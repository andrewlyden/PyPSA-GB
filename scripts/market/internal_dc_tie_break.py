"""
Tie-break cost on GB-internal HVDC flow (user decision, 16 Oct 2026).

GB-internal DC links (Eastern Green Links 1 and 2, the Western HVDC, the
Peterhead-Drax / Spittal-Sizewell projects, the Caithness-Moray-Shetland hub
legs, ...) are lossless and free in both directions in the model. Wherever such
a link runs in parallel with the AC network, or with another DC leg, its flow
is a degenerate tie: the LP can place any flow on it at the same cost, and the
barrier lands wherever it lands. On the 16 Oct 2026 cal_rb1 balancing results
EGL1 (``BRNX4A_HAWP41_HVDC_2029``) ran at full south-to-north for 915 hours of
the year with no reason in the objective.

The fix is a small cost on the ABSOLUTE flow of every internal DC link, about
its converter losses times the price: ``market.internal_dc_tie_break_gbp_per_mwh``
(default 0.75 GBP/MWh). A plain ``marginal_cost`` cannot do this on a two-way
link: PyPSA charges ``marginal_cost x p``, so with ``p_min_pu = -1`` reverse flow
would earn money. The penalty is therefore built in the linopy model:

    p[l, t] = fwd[l, t] - rev[l, t],   fwd, rev >= 0
    objective += cost x sum_t w_t x (fwd[l, t] + rev[l, t])

With a positive cost the optimum never holds both legs positive, so
``fwd + rev = |p|``. ``w_t`` is ``snapshot_weightings.objective`` (a coarse
look-ahead block carries its hours).

Scope. Internal links only, identified by ``internal_dc_link_names``: carrier in
``carriers`` (``DC``), neither end an External (foreign) bus by name or by a
non-GB ``country``, both ends electrical buses (carrier AC or DC). Cross-border
interconnectors (priced and fixed separately) and the hydrogen links
(electrolysis, H2_turbine) are never touched.

Both stages. Stage 1 (``solve_wholesale.py``: single, rolling day-ahead and the
look-ahead windows) and the balancing stage (``solve_balancing.py``: full period
and rolling windows) add the term through the same ``extra_functionality``
callback, after any callback that replaces the objective. In the collapsed
copperplate the mainland-internal links are removed (their flow is zero on one
bus), so in Stage 1 the term reaches the island links only; it removes the free
loop mainland -> Central DC Substation -> mainland there.

Reporting. The penalty is objective only: each stage strips it from the cost it
reports (Stage 1 objective, BM cost) and records it separately in
``network.meta['internal_dc_tie_break']``; ``analyze_market.py`` reports it as
``internal_dc_tie_break_cost_GBP``. Redispatch volumes, ``constraint_costs.csv``
and ``modelled_redispatch_cost*`` are built from accepted bids and offers, so
they change only through dispatch.
"""

import logging

import numpy as np
import pandas as pd

# Must match config/defaults/ market.internal_dc_tie_break_gbp_per_mwh.
DEFAULT_TIE_BREAK_GBP_PER_MWH = 0.75
DEFAULT_CARRIERS = ("DC",)
EXTERNAL_BUS_PATTERN = "External"
ELECTRICAL_BUS_CARRIERS = ("AC", "DC")
META_KEY = "internal_dc_tie_break"
FWD_VARIABLE = "Link-internal_dc_fwd"
REV_VARIABLE = "Link-internal_dc_rev"
CONSTRAINT_NAME = "Link-internal_dc_tie_break"


def resolve_internal_dc_tie_break(market_config) -> float:
    """``market.internal_dc_tie_break_gbp_per_mwh`` (default 0.75), validated; 0 or null is off."""
    value = (market_config or {}).get("internal_dc_tie_break_gbp_per_mwh", DEFAULT_TIE_BREAK_GBP_PER_MWH)
    value = 0.0 if value is None else float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"market.internal_dc_tie_break_gbp_per_mwh must be >= 0, got {value}")
    return value


def internal_dc_link_names(network, carriers=DEFAULT_CARRIERS, external_pattern=EXTERNAL_BUS_PATTERN) -> list:
    """GB-internal DC links of ``network``: the links the tie-break prices.

    A link qualifies when its carrier is in ``carriers``, it has a rating
    (``p_nom > 0`` or extendable), both ends are buses of the network, neither end
    is foreign (name containing ``external_pattern``, or a ``country`` other than
    GB where one is set) and both ends are electrical buses (carrier AC or DC
    where the bus carries one). The same rule as ``find_boundary_crossings`` and
    ``relax_island_hvdc_links`` (carrier DC, no External end), with the country
    and bus-carrier checks added so a hydrogen or foreign bus can never qualify.
    """
    links = network.links
    if not len(links):
        return []
    carrier = links["carrier"].astype(str) if "carrier" in links.columns else pd.Series("", index=links.index)
    mask = carrier.isin({str(c) for c in carriers})
    bus0, bus1 = links["bus0"].astype(str), links["bus1"].astype(str)
    mask &= ~bus0.str.contains(external_pattern, regex=False) & ~bus1.str.contains(external_pattern, regex=False)
    buses = network.buses
    mask &= bus0.isin(buses.index) & bus1.isin(buses.index)
    rated = pd.to_numeric(links["p_nom"], errors="coerce").fillna(0.0) > 0.0
    if "p_nom_extendable" in links.columns:
        rated |= links["p_nom_extendable"].fillna(False).astype(bool)
    mask &= rated
    if "country" in buses.columns:
        country = buses["country"].fillna("").astype(str).str.strip()
        foreign = set(country.index[(country != "") & (country != "GB")])
        mask &= ~bus0.isin(foreign) & ~bus1.isin(foreign)
    if "carrier" in buses.columns:
        bus_carrier = buses["carrier"].fillna("").astype(str)
        electrical = set(bus_carrier.index[bus_carrier.isin(ELECTRICAL_BUS_CARRIERS) | (bus_carrier == "")])
        mask &= bus0.isin(electrical) & bus1.isin(electrical)
    return list(links.index[mask])


def build_internal_dc_tie_break_callback(network, cost_gbp_per_mwh: float, logger: logging.Logger = None,
                                         names=None, stage: str = ""):
    """``extra_functionality`` adding ``cost x |p|`` on every internal DC link, or None when off.

    Built once per stage on the network that is solved (the collapsed copy in a
    collapsed Stage 1); each call (every window, every re-solve of the fallback
    ladder) adds the split variables, the identity ``p = fwd - rev`` and the
    objective term to that model. Must come after any callback that REPLACES the
    objective (the balancing stage's ``_build_balancing_extra_functionality``).
    """
    cost = float(cost_gbp_per_mwh or 0.0)
    logger = logger or logging.getLogger(__name__)
    if cost <= 0.0:
        logger.info(f"Internal HVDC tie-break off{f' ({stage})' if stage else ''}")
        return None
    names = internal_dc_link_names(network) if names is None else list(names)
    if not names:
        logger.info(f"Internal HVDC tie-break: no GB-internal DC link{f' in {stage}' if stage else ''}")
        return None
    logger.info(
        f"Internal HVDC tie-break{f' ({stage})' if stage else ''}: {cost:.2f} GBP/MWh on |flow| of "
        f"{len(names)} GB-internal DC link(s) (objective only, stripped from the reported cost): "
        + ", ".join(names[:12]) + (" ..." if len(names) > 12 else "")
    )

    def internal_dc_tie_break(n, snapshots):
        import xarray as xr

        model = n.model
        if "Link-p" not in model.variables:
            return
        link_p = model.variables["Link-p"]
        dims = [d for d in link_p.dims if d != "snapshot"]
        dim = dims[0] if dims else "Link"
        present = set(map(str, link_p.coords[dim].values))
        live = [name for name in names if name in present]
        if not live:
            return
        sns = pd.Index(snapshots)
        coords = {"snapshot": sns, dim: pd.Index(live, name=dim)}
        zeros = xr.DataArray(np.zeros((len(sns), len(live))), dims=["snapshot", dim], coords=coords)
        fwd = model.add_variables(lower=zeros, name=FWD_VARIABLE)
        rev = model.add_variables(lower=zeros, name=REV_VARIABLE)
        model.add_constraints(link_p.sel({dim: live}) - fwd + rev == 0.0, name=CONSTRAINT_NAME)
        weights = n.snapshot_weightings.objective.reindex(sns).fillna(1.0)
        weight_da = xr.DataArray(weights.to_numpy(dtype=float), dims=["snapshot"], coords={"snapshot": sns})
        model.objective = model.objective + ((fwd + rev) * weight_da).sum() * cost

    internal_dc_tie_break.cost = cost
    internal_dc_tie_break.links = tuple(names)
    return internal_dc_tie_break


def internal_dc_tie_break_cost(network, callback, snapshots=None) -> tuple:
    """``(GBP, MWh)`` the tie-break put in the network's last solve; ``(0, 0)`` when off.

    Read from the solved split variables, so it is exactly the objective term
    (``cost x sum w (fwd + rev)``), which equals ``cost x sum w |p|`` at an optimum.
    ``snapshots`` limits it to those rows (the kept hours of a look-ahead window).
    """
    if callback is None:
        return 0.0, 0.0
    model = getattr(network, "model", None)
    if model is None:
        return 0.0, 0.0
    total = 0.0
    for name in (FWD_VARIABLE, REV_VARIABLE):
        if name not in model.variables:
            continue
        solution = model.variables[name].solution.fillna(0.0).clip(min=0.0)
        frame = solution.to_pandas()
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        if snapshots is not None:
            frame = frame.loc[frame.index.isin(pd.Index(snapshots))]
        weights = network.snapshot_weightings.objective.reindex(frame.index).fillna(1.0)
        total += float(frame.mul(weights, axis=0).to_numpy().sum())
    return total * float(callback.cost), total


def internal_dc_flow_by_link(network, names, snapshots=None) -> pd.Series:
    """Weighted |p0| (MWh) per internal DC link from a solved network's ``links_t.p0``."""
    p0 = getattr(network.links_t, "p0", None)
    names = [x for x in names if p0 is not None and x in p0.columns]
    if not names:
        return pd.Series(dtype=float)
    flows = p0[names]
    if snapshots is not None:
        flows = flows.loc[flows.index.isin(pd.Index(snapshots))]
    weights = network.snapshot_weightings.objective.reindex(flows.index).fillna(1.0)
    return flows.abs().mul(weights, axis=0).sum()


def record_internal_dc_tie_break(network, stage: str, cost_gbp_per_mwh: float, names, cost_gbp: float,
                                 flow_mwh: float, logger: logging.Logger = None) -> dict:
    """Store one stage's tie-break in ``network.meta['internal_dc_tie_break'][stage]`` and log it.

    ``flow_by_link_MWh`` is read from the network's saved ``links_t.p0``.
    """
    names = list(names or [])
    record = {
        "cost_gbp_per_mwh": float(cost_gbp_per_mwh or 0.0),
        "links": names,
        "cost_GBP": round(float(cost_gbp), 2),
        "abs_flow_MWh": round(float(flow_mwh), 3),
        "flow_by_link_MWh": {str(k): round(float(v), 3)
                             for k, v in internal_dc_flow_by_link(network, names).items()},
        "basis": "objective only: stripped from the reported cost of this stage",
    }
    meta = dict(getattr(network, "meta", None) or {})
    block = dict(meta.get(META_KEY) or {})
    block[stage] = record
    meta[META_KEY] = block
    network.meta = meta
    if logger is not None and record["cost_gbp_per_mwh"] > 0.0:
        logger.info(
            f"Internal HVDC tie-break ({stage}): {flow_mwh / 1e3:,.1f} GWh of |flow| on {len(names)} link(s), "
            f"{cost_gbp:,.2f} GBP at {record['cost_gbp_per_mwh']:.2f} GBP/MWh stripped from the reported cost"
        )
    return record


def tie_break_summary(wholesale_network, balancing_network) -> dict:
    """The analysis summary's block: each stage's recorded tie-break (empty stages left out)."""
    out = {}
    for stage, net in (("wholesale", wholesale_network), ("balancing", balancing_network)):
        meta = dict(getattr(net, "meta", None) or {}) if net is not None else {}
        block = meta.get(META_KEY) or {}
        if isinstance(block, str):  # a meta round trip may hand back JSON text
            import json
            block = json.loads(block)
        record = block.get(stage)
        if record:
            out[stage] = record
    return out
