"""Stage 1 marginal-cost basis.

The network's generator marginal costs are *market-shaped*: a ROC generator
carries ``-(banding x buyout)`` because that is what it forgoes when curtailed,
a CfD generator carries the CfD dispatch cost (zero), and empirically calibrated
renewables carry their balancing-market bids.  A copperplate market schedules on
those, correctly, because that is what the plant bids.  A public operator
minimising the cost of the system does not: certificates it pays to itself are
not a cost, so it dispatches on fuel, carbon and variable O&M alone.  Without
this switch a central dispatch ranks ROC wind above CfD wind for reasons that
vanish under nationalisation, and the resource part of the answer is
contaminated by the transfer part.

``mode: as_built`` leaves the network alone (the default, every existing run).
``mode: resource_cost`` sets every generator of a renewable carrier to
``renewable_floor`` (a positive cost just above zero: spilled wind costs nothing,
and a zero cost is degenerate), and, where the marginal-cost audit written at
build time is available, sets every other generator whose cost carries a subsidy
layer (``source`` starting ``subsidy_``, ``renewable_empirical`` or containing
``_ROC(``) to its fuel plus carbon plus variable O&M.  Nuclear (an override that
is already a cost), thermal formula costs, storage and load shedding are never
touched.  The switch is applied before the forecast error and the copperplate
step, so the collapsed market and the central dispatch see the same basis.

Stage 2 in the cost-basis scenarios already floors renewables through
``mc_floors``; with this switch on, the two stages of institution I3 are on one
basis.

``support_order`` (on by default, 27 Oct 2026; the 14 Oct decision that central
dispatch turns subsidised renewables down after merchant, Ofgem 2023 A1.18):
every renewable at one floor left which unit gave way to the solver, and the
central year cut CfD and ROC units while running other CfD and merchant units
above their market schedule (40.6 TWh of lost CfD volume against the market's
32.8; about GBP 0.3 bn of support paid twice). The floor is therefore spread
by the support a unit loses when turned down: merchant units at ``renewable_floor
+ spread_gbp_per_mwh``, a CfD unit by its strike, a ROC unit by its certificate
value (minus its as-built cost), each at ``floor + spread x (1 - value / largest
value)``, so the highest-support plant gives way last. Resource cost among
renewables is zero either way, so the spread only resolves the tie.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

MODES = ("as_built", "resource_cost")
DEFAULT_RENEWABLE_CARRIERS = (
    "wind_onshore", "wind_offshore", "solar_pv", "embedded_wind", "embedded_solar",
    "large_hydro", "small_hydro", "marine", "tidal_stream", "shoreline_wave", "tidal_lagoon",
)
SUBSIDY_SOURCE_PREFIXES = ("subsidy_", "renewable_empirical")
SUBSIDY_SOURCE_MARKERS = ("_ROC(",)
UNTOUCHED_SOURCES = ("nuclear_override", "protected")
#: Defaults of market.wholesale.marginal_cost_basis.support_order.
#: ``labels_file`` (11 Nov 2026): a FES long-range network carries no support labels (they are in the scenario's
#: support-labels file, one row per contract cohort); with it set, a generator's support value is its cohorts'
#: weighted live CfD strike, so central dispatch turns subsidised plant down after merchant as in 2030.
DEFAULT_SUPPORT_ORDER = {"enabled": True, "spread_gbp_per_mwh": 1.0, "labels_file": None}


def settings(block: Mapping | None) -> dict:
    block = dict(block or {})
    mode = str(block.get("mode", "as_built")).lower()
    if mode not in MODES:
        raise ValueError(f"market.wholesale.marginal_cost_basis.mode must be one of {MODES}, got {mode!r}")
    floor = float(block.get("renewable_floor", 0.01))
    if floor <= 0:
        raise ValueError("market.wholesale.marginal_cost_basis.renewable_floor must be positive (a zero cost is degenerate)")
    carriers = tuple(block.get("renewable_carriers") or DEFAULT_RENEWABLE_CARRIERS)
    order = {**DEFAULT_SUPPORT_ORDER, **dict(block.get("support_order") or {})}
    if unknown := sorted(set(order) - set(DEFAULT_SUPPORT_ORDER)):
        raise ValueError(f"market.wholesale.marginal_cost_basis.support_order has unknown keys {unknown}")
    order = {"enabled": bool(order["enabled"]), "spread_gbp_per_mwh": float(order["spread_gbp_per_mwh"]),
             "labels_file": (str(order["labels_file"]) if order.get("labels_file") else None)}
    if not 0.0 < order["spread_gbp_per_mwh"] <= 5.0:
        raise ValueError("market.wholesale.marginal_cost_basis.support_order.spread_gbp_per_mwh must be small and positive (0 to 5)")
    return {"mode": mode, "renewable_floor": floor, "renewable_carriers": carriers, "support_order": order}


def _is_subsidy_shaped(source: str) -> bool:
    source = str(source or "")
    if source.startswith(UNTOUCHED_SOURCES):
        return False
    return source.startswith(SUBSIDY_SOURCE_PREFIXES) or any(marker in source for marker in SUBSIDY_SOURCE_MARKERS)


def labels_support_value(index: pd.Index, labels: pd.DataFrame) -> pd.Series:
    """GBP/MWh of CfD support each generator loses when turned down, from the support labels (one row per cohort):
    the weighted sum of its CfD cohorts' strikes times their live share; 0 for a generator with none."""
    rows = labels.loc[labels["component_type"].astype(str).eq("Generator")
                      & labels["support_type"].fillna("").astype(str).str.strip().str.lower().eq("cfd")]
    if rows.empty:
        return pd.Series(0.0, index=index)
    weight = pd.to_numeric(rows["weight"], errors="coerce").fillna(0.0)
    live = pd.to_numeric(rows["live_share"], errors="coerce").fillna(1.0)
    strike = pd.to_numeric(rows["strike_gbp_per_mwh_2024"], errors="coerce").fillna(0.0)
    value = (weight * live * strike.clip(lower=0.0)).groupby(rows["component"].astype(str)).sum()
    return value.reindex(index).fillna(0.0)


def support_value(generators: pd.DataFrame, as_built: pd.Series, strike_table,
                  labels: pd.DataFrame | None = None) -> pd.Series:
    """GBP/MWh of support a renewable loses when turned down: a CfD unit's strike, a ROC
    unit's certificate value (minus its as-built cost), 0 for merchant and anything else.

    A network without support labels takes them from ``labels`` (the FES support-labels file) when given.
    """
    support = generators.get("support_type", pd.Series("", index=generators.index)).fillna("").astype(str).str.strip()
    if labels is not None and not support.isin(["CfD", "ROC"]).any():
        return labels_support_value(generators.index, labels)
    value = pd.Series(0.0, index=generators.index)
    roc = support.eq("ROC")
    value[roc] = (-as_built[roc]).clip(lower=0.0)
    cfd = generators.index[support.eq("CfD")]
    if len(cfd):
        if strike_table is None:
            raise ValueError("marginal_cost_basis.support_order needs the scenario's CfD strike table "
                             f"({len(cfd)} CfD renewables to order)")
        from scripts.market.cfd_strike_table import contract_technology, strike_lookup

        rounds = generators.get("cfd_round", pd.Series("", index=generators.index)).fillna("").astype(str).str.strip()
        technology = contract_technology(generators)
        for name in cfd:
            strike, _, _ = strike_lookup(strike_table, str(generators.at[name, "carrier"]), rounds[name], technology[name])
            if strike is None:
                raise ValueError(f"marginal_cost_basis.support_order: no strike for {name} "
                                 f"({generators.at[name, 'carrier']}, {rounds[name]!r})")
            value[name] = max(float(strike), 0.0)
    return value


def resource_cost_table(generators: pd.DataFrame, cfg: Mapping, audit: pd.DataFrame | None,
                        strike_table=None, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per generator: the as-built cost, the resource cost and the rule that set it.

    Pure: takes ``network.generators`` (needs ``carrier`` and ``marginal_cost``;
    ``support_type``, ``cfd_round`` for the support order), the optional audit frame
    (``generator``, ``source``, ``fuel_cost_gbp_per_mwh``, ``carbon_cost_gbp_per_mwh``,
    ``variable_om_gbp_per_mwh``) and the CfD strike table the support order reads.
    Rows whose rule is ``unchanged`` keep the as-built cost.
    """
    table = pd.DataFrame({
        "carrier": generators["carrier"].astype(str),
        "as_built": pd.to_numeric(generators["marginal_cost"], errors="coerce").fillna(0.0),
    }, index=generators.index)
    table["resource_cost"] = table["as_built"]
    table["rule"] = "unchanged"
    renewable = table["carrier"].isin(cfg["renewable_carriers"])
    table.loc[renewable, "resource_cost"] = cfg["renewable_floor"]
    table.loc[renewable, "rule"] = "renewable_floor"
    order = cfg.get("support_order") or {}
    if order.get("enabled") and renewable.any():
        value = support_value(generators.loc[renewable], table.loc[renewable, "as_built"], strike_table, labels)
        largest = float(value.max())
        share = value / largest if largest > 0 else value * 0.0
        table.loc[renewable, "resource_cost"] = (cfg["renewable_floor"]
                                                 + order["spread_gbp_per_mwh"] * (1.0 - share)).to_numpy()
        table.loc[renewable, "rule"] = "renewable_floor_support_order"
        table.loc[renewable, "support_value"] = value.to_numpy()
    if audit is not None and not audit.empty:
        keyed = audit.drop_duplicates("generator").set_index("generator")
        keyed = keyed.reindex(table.index)
        components = sum(
            pd.to_numeric(keyed[column], errors="coerce").fillna(0.0)
            for column in ("fuel_cost_gbp_per_mwh", "carbon_cost_gbp_per_mwh", "variable_om_gbp_per_mwh")
        )
        subsidised = keyed["source"].map(_is_subsidy_shaped).fillna(False).astype(bool) & ~renewable
        table.loc[subsidised, "resource_cost"] = components[subsidised]
        table.loc[subsidised, "rule"] = "audit_fuel_carbon_vom"
        table["audit_source"] = keyed["source"]
    else:
        table["audit_source"] = np.nan
    return table


def needs_strike_table(block: Mapping | None) -> bool:
    """True when the basis orders renewables by support, so the caller must pass the strike table."""
    cfg = settings({k: v for k, v in dict(block or {}).items() if k != "dpa_dispatch"})
    # The labels carry their own strikes (11 Nov 2026).
    return cfg["mode"] != "as_built" and cfg["support_order"]["enabled"] and not cfg["support_order"]["labels_file"]


def apply_marginal_cost_basis(network, block: Mapping | None, logger=None, audit: pd.DataFrame | None = None,
                              strike_table=None) -> dict:
    """Apply the configured basis to ``network.generators.marginal_cost`` in place; return a summary."""
    logger = logger or _LOG
    dpa = dict((block or {}).get("dpa_dispatch") or {})
    if dpa:
        # Task BM2050 (coordinator follow-up): DPA plant dispatched ahead of the
        # reference CCGT in Stage 1 and the annual pass, from the 2050 support labels.
        # No 2030 scenario sets the key.
        block = {k: v for k, v in dict(block).items() if k != "dpa_dispatch"}
        if dpa.get("enabled", False):
            from scripts.market.support_bids import apply_dpa_dispatch

            apply_dpa_dispatch(network, dpa, logger)
    cfg = settings(block)
    if cfg["mode"] == "as_built":
        return {"mode": "as_built"}
    generators = network.generators
    labels = None
    if cfg["support_order"]["enabled"] and cfg["support_order"]["labels_file"]:
        labels = pd.read_csv(cfg["support_order"]["labels_file"], low_memory=False)
    table = resource_cost_table(generators, cfg, audit, strike_table, labels)
    changed = table["rule"].ne("unchanged") & ~np.isclose(table["as_built"], table["resource_cost"])
    negative_before = int((table["as_built"] < 0).sum())
    network.generators.loc[changed.index[changed], "marginal_cost"] = table.loc[changed, "resource_cost"].to_numpy()
    # A time-varying marginal cost would silently override the static one.
    varying = getattr(network.generators_t, "marginal_cost", None)
    dropped_varying = 0
    if varying is not None and not varying.empty:
        overlap = [c for c in varying.columns if c in changed.index[changed]]
        if overlap:
            network.generators_t.marginal_cost = varying.drop(columns=overlap)
            dropped_varying = len(overlap)
    p_nom = pd.to_numeric(generators["p_nom"], errors="coerce").fillna(0.0)
    by_rule = table.loc[changed].assign(p_nom=p_nom[changed]).groupby(["rule", "carrier"]).agg(
        generators=("as_built", "size"), capacity_mw=("p_nom", "sum"),
        as_built_mean=("as_built", "mean"), resource_cost_mean=("resource_cost", "mean"),
    )
    still_negative = int((pd.to_numeric(network.generators["marginal_cost"], errors="coerce") < 0).sum())
    summary = {
        "mode": cfg["mode"], "renewable_floor": cfg["renewable_floor"], "support_order": cfg["support_order"],
        "audit_available": audit is not None and not audit.empty,
        "generators_recosted": int(changed.sum()),
        "capacity_recosted_mw": float(p_nom[changed].sum()),
        "negative_costs_before": negative_before, "negative_costs_after": still_negative,
        "time_varying_costs_dropped": dropped_varying,
        "by_rule_and_carrier": {
            f"{rule}:{carrier}": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}
            for (rule, carrier), row in by_rule.iterrows()
        },
    }
    logger.info("=" * 80)
    logger.info("MARGINAL COST BASIS: RESOURCE COST (item N1)")
    logger.info("=" * 80)
    logger.info(
        "Re-costed %s generators (%.0f MW); negative costs %s -> %s; audit %s",
        summary["generators_recosted"], summary["capacity_recosted_mw"], negative_before, still_negative,
        "used" if summary["audit_available"] else "not available (renewable floor only)",
    )
    for (rule, carrier), row in by_rule.iterrows():
        logger.info("  %-24s %-16s %5d gens %9.0f MW  %8.2f -> %6.2f GBP/MWh",
                    rule, carrier, row["generators"], row["capacity_mw"], row["as_built_mean"], row["resource_cost_mean"])
    if still_negative:
        logger.warning("%s generators still carry a negative marginal cost after the resource-cost basis", still_negative)
    if not summary["audit_available"]:
        logger.warning("No marginal-cost audit: only the renewable floor was applied; subsidy-shaped thermal costs, if any, remain")
    try:
        network.meta["marginal_cost_basis"] = summary
    except Exception:  # pragma: no cover - meta is optional on older pypsa
        pass
    return summary
