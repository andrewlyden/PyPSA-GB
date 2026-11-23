"""Item N10: the nationalised bill under each dispatch institution.

The nationalised tariff (:mod:`scripts.public_ownership.nationalised_tariff`) is
built on a register that was settled against one solved year, institution I1:
the market machinery kept, with every counterparty public.  This module answers
the question that closes the programme: **does the bill change if the state also
runs the system differently?**

It does not rebuild the register on the central-dispatch year.  It does not need
to.  Under nationalisation every asset is owned, so the acquisition charge,
new-build capital and fixed O&M are capacity-driven and identical whichever way
the operator dispatches; the same fleet meets the same demand on the same
network.  Only four things can move, and all four are measured here directly
from the solved networks of both years, the same way in each:

* **fuel and carbon**, energy times the build-time fuel and carbon rates;
* **variable O&M**, energy times the build-time variable rate;
* **imports and exports**, cross-border energy times its price;
* the **institution lines**: the interconnector re-trading cost (I1, gone
  under I3), and under I3 the forecast-error charge borrowed from FTI, which
  replaces the model's own real-time correction (already inside the physical
  fuel, O&M and trade lines, so the real-time line takes it back out; review
  26 Sep 2026; since 27 Sep measured on those lines' own basis, physical less
  the Stage 1 schedule, :func:`realtime_correction_on_line_basis`).  The model barely produces a forecast-error cost itself
  (:mod:`scripts.market.forecast_error_charge`; read from the year settlement,
  audit 26 Sep 2026).  It is charged to I3 alone because FTI define it as the
  increment central dispatch adds over self-dispatch with intra-day trading.

Each line is converted to a household figure at the rate the tariff recovers it
(``allocation`` in the tariff summary: the like-for-like pool rate for the cost of
owned plant, final demand for the balancing lines), never flat by default.

Everything else in the tariff is carried across unchanged and said to be so.

**The finding this exists to state.** The constraint payments that central
dispatch removes are the same money nationalisation removes by owning both
sides of them.  They are substitutes, not complements: a nationalisation saving
and a central-dispatch saving must never be added together.  What the
institution adds on top of ownership is the resource difference alone, which on
this run is a fraction of a percent of the fuel bill.

Usage::

    python scripts/public_ownership/nationalised_institutions.py <i1_bill_summary.json>
        <out_summary.json> <out_comparison.csv>
        --market-scenarios <12 names> --central-scenarios <12 names>
        --marginal-costs <breakdown.csv> [--fti-settlement <year_settlement.csv>]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.nationalised_tariff import (  # noqa: E402
    LINE_FORECAST_ERROR, LINE_FUEL, LINE_IC_ACQUISITION, LINE_IC_FOREIGN_RENT, LINE_IC_RETRADING, LINE_IMPORTS,
    LINE_REALTIME, LINE_VARIABLE_OM, VALUE, household, line_rate,
)
from config.config_loader import read_config

# F20 phase 0: a parity run in a worktree reads the solved months from another tree's
# resources (PYPSA_GB_INPUT_RESOURCES); unset, this is resources/market as before.
MARKET_DIR = Path(os.environ.get("PYPSA_GB_INPUT_RESOURCES") or "resources") / "market"
IMPORT_CARRIERS = ("EU_market", "EU_import")
# Review B13: the heat a CHP unit sells is credited at the cost of the boiler it
# displaces, so only the electricity's share of its fuel and carbon is measured.
# Same rule and values as marginal_costs.byproduct_adjustments in
# config/defaults/method/brownfield.yaml, which the CLI reads when given --config.
HEAT_CREDIT_DEFAULTS: dict[str, dict[str, float]] = {"CHP": {"heat_to_power": 1.1, "boiler_efficiency": 0.85}}


def heat_credit_settings(config: Mapping | None, overlays: tuple = ()) -> dict[str, dict[str, float]]:
    """``{carrier: {heat_to_power, boiler_efficiency}}`` from the byproduct adjustments in a config.

    ``overlays`` (config trees, e.g. ``load_layer("method", "brownfield")``) are read after ``config``; the
    first carrier found wins.
    """
    if not config and not overlays:
        return {k: dict(v) for k, v in HEAT_CREDIT_DEFAULTS.items()}
    blocks = [((c or {}).get("marginal_costs") or {}) for c in ((config or {}),) + tuple(overlays)]
    out: dict[str, dict[str, float]] = {}
    for block in blocks:
        for carrier, adj in (block.get("byproduct_adjustments") or {}).items():
            if (adj or {}).get("mode") == "heat_credit" and carrier not in out:
                out[str(carrier)] = {"heat_to_power": float(adj["heat_to_power"]),
                                     "boiler_efficiency": float(adj["boiler_efficiency"])}
    return out or {k: dict(v) for k, v in HEAT_CREDIT_DEFAULTS.items()}


def config_heat_credit_settings(path) -> dict[str, dict[str, float]]:
    """:func:`heat_credit_settings` of the config at ``path`` (``read_config``); the CHP heat credit is in
    the base defaults (``marginal_costs.byproduct_adjustments``) since the config restructure."""
    return heat_credit_settings(read_config(path) or {})


def heat_credit_rates(rates: pd.DataFrame, settings: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Per generator, the heat credit split into its fuel and carbon parts, GBP per MWh of electricity.

    ``heat_to_power x (fuel + carbon per MWh of fuel) / boiler efficiency``, the
    rule ``apply_marginal_costs`` prices the credit with; fuel and carbon per MWh
    of fuel are the breakdown's per-MWh-electric figures times ``efficiency_used``.
    Capped at the unit's own fuel and carbon, so no unit is credited more than it burns.
    """
    out = pd.DataFrame(0.0, index=rates.index,
                       columns=["heat_credit_fuel_gbp_per_mwh", "heat_credit_carbon_gbp_per_mwh"])
    if not settings or "carrier" not in rates.columns:
        return out
    carrier = rates["carrier"].astype(str)
    nan = pd.Series(np.nan, index=rates.index)
    fuel = pd.to_numeric(rates.get("fuel_cost_gbp_per_mwh", nan), errors="coerce").fillna(0.0)
    carbon = pd.to_numeric(rates.get("carbon_cost_gbp_per_mwh", nan), errors="coerce").fillna(0.0)
    eff = pd.to_numeric(rates.get("efficiency_used", nan), errors="coerce")
    for name, adj in settings.items():
        rows = carrier.eq(name)
        if not rows.any():
            continue
        if eff[rows].isna().any():
            raise ValueError(f"heat credit for {name} needs efficiency_used in the marginal-cost breakdown")
        factor = (float(adj["heat_to_power"]) * eff[rows] / float(adj["boiler_efficiency"])).clip(upper=1.0)
        out.loc[rows, "heat_credit_fuel_gbp_per_mwh"] = fuel[rows] * factor
        out.loc[rows, "heat_credit_carbon_gbp_per_mwh"] = carbon[rows] * factor
    return out
# Lines the institution cannot move: the fleet, its capital and its fixed costs
# are the same under either operator.
CARRIED_UNCHANGED_NOTE = (
    "capacity-driven: the same fleet on the same network, so the acquisition charge, "
    "new-build capital and fixed O&M do not depend on how it is dispatched"
)


def _balancing_path(scenario: str) -> Path:
    return MARKET_DIR / f"{scenario}_balancing.nc"


def _stage_1_network_path(scenario: str) -> Path:
    """The saved network that carries a Stage 1 month's interconnector price tables.

    Its balancing network when it has one; otherwise (a Stage-1-only month, such as the
    s1_zf parent the calibrated months take through ``wholesale_from``, 13 Oct 2026) its
    own Stage 1 network, whose link and ask tables equal the balancing networks' exactly.
    """
    balancing = _balancing_path(scenario)
    return balancing if balancing.exists() else MARKET_DIR / f"{scenario}_wholesale.nc"


def _paired_stage_1(scenarios: Sequence[str], stage_1_scenarios: Sequence[str] | None) -> list[str]:
    """Each month's Stage 1 month (its ``wholesale_from`` parent), paired month by month; default itself."""
    stage_1 = list(stage_1_scenarios) if stage_1_scenarios is not None else list(scenarios)
    if len(stage_1) != len(scenarios):
        raise ValueError(f"{len(scenarios)} months but {len(stage_1)} Stage 1 months; they must pair month by month")
    return stage_1


#: F19 (user decision, 6 Oct 2026): the gap between two years' solve windows beyond which they
#: are different solve runs (public_ownership.institution_comparison.mixed_vintage_gap_hours).
MIXED_VINTAGE_GAP_HOURS_DEFAULT = 24.0


def solve_identity(scenarios, stage_1_scenarios: Sequence[str] | None = None) -> dict:
    """One solved year's identity: when and what its Stage 2 networks and Stage 1 schedules were written as.

    The saved networks carry no solve stamp (their ``meta`` names the scenario, not the run), so the
    identity is the files' own: the first and last write time (UTC) of the twelve months' balancing
    networks and Stage 1 dispatch files, and a fingerprint of every file's name, size and write time.

    ``stage_1_scenarios`` pairs each month with the month whose Stage 1 files it reads (its
    ``wholesale_from`` parent, 18 Oct 2026: the market family ``cp30_network`` takes Stage 1 from
    ``stage1``); default each month itself.
    """
    import hashlib
    from datetime import datetime, timezone

    stage_1 = _paired_stage_1(scenarios, stage_1_scenarios)
    files = []
    for scenario, stage_1_scenario in zip(scenarios, stage_1):
        for path in (_balancing_path(scenario), MARKET_DIR / f"{stage_1_scenario}_wholesale_dispatch.csv"):
            if path.exists():
                stat = path.stat()
                files.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    if not files:
        return {"scenarios": list(scenarios), "files": 0, "first_written_utc": None, "last_written_utc": None,
                "fingerprint": None}
    stamp = lambda ns: datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    digest = hashlib.sha1("\n".join(f"{n}|{s}|{t}" for n, s, t in sorted(files)).encode("utf-8")).hexdigest()[:16]
    return {"scenarios": list(scenarios), "files": len(files),
            "first_written_utc": stamp(min(t for _, _, t in files)),
            "last_written_utc": stamp(max(t for _, _, t in files)),
            "fingerprint": digest,
            "basis": "file write times and sizes (the saved networks carry no solve stamp)"}


def solve_vintage(market: Mapping, central: Mapping, gap_hours: float = MIXED_VINTAGE_GAP_HOURS_DEFAULT) -> dict:
    """The two years' identities and whether they come from different solve runs (``mixed_vintage``).

    Two years are one solve run when their write windows overlap or lie within ``gap_hours`` of each
    other; otherwise the comparison sets one run's market year against another run's central year.
    """
    from datetime import datetime

    out = {"market_year": dict(market), "central_dispatch_year": dict(central), "mixed_vintage_gap_hours": float(gap_hours)}
    if not market.get("first_written_utc") or not central.get("first_written_utc"):
        out.update(mixed_vintage=None, gap_between_runs_hours=None)
        return out
    t = lambda key, d: datetime.fromisoformat(d[key])  # noqa: E731
    gap = max(t("first_written_utc", market) - t("last_written_utc", central),
              t("first_written_utc", central) - t("last_written_utc", market)).total_seconds() / 3600.0
    out["gap_between_runs_hours"] = max(gap, 0.0)
    out["mixed_vintage"] = bool(gap > float(gap_hours))
    out["note"] = ("user decision (6 Oct 2026): I3 stays P3's central case; when the market year was re-solved after "
                   "the central-dispatch year (different solve runs), the institution step compares two vintages "
                   "and is flagged so" if out["mixed_vintage"] else "both years from one solve run")
    return out


#: Where each stage's flows are read: the physical (balancing) dispatch from the
#: saved Stage 2 network, the day-ahead schedule from the Stage 1 CSVs. Both use
#: the Stage 2 network's static tables and snapshot weights (one shared build).
STAGES = ("balancing", "wholesale")


def _schedule_frame(scenario: str, kind: str, index: pd.Index) -> pd.DataFrame:
    """A Stage 1 schedule (``wholesale_dispatch`` or ``wholesale_links``) on the network's snapshots."""
    path = MARKET_DIR / f"{scenario}_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; every month must be solved")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.columns = frame.columns.astype(str)
    return frame.reindex(index)


def _stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return stage


def counter_trade_premium(scenario: str, flow: pd.DataFrame, links: pd.DataFrame,
                          schedule_scenario: str | None = None,
                          ask_t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Premium per snapshot and signed cable, ``premium x |flow - Stage 1 schedule|`` (unweighted).

    Zero, without reading the schedule, when the network was solved with no premium.
    ``schedule_scenario`` names the month whose Stage 1 schedule the balancing run
    anchored to (its ``wholesale_from`` parent, item BM1); default the month itself.
    ``ask_t`` (the hourly ask table) values a premium solved under the ``all_in``
    rule on the prices the LP saw.
    """
    from scripts.interconnectors.market_coupling import (
        COUNTER_TRADE_BID_PREMIUM_COLUMN, COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN,
        COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN, COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN, COUNTER_TRADE_PREMIUM_COLUMN,
        counter_trade_premium_cost,
    )

    # 18 Oct 2026: a move is paid for when any of the premium, the bid premium or the
    # foreign-plant uplift is set (the NESO-consistent families, cp30_network among them, set
    # premium 0 and price moves through the uplift, which the premium-only test dropped).
    paid_columns = [c for c in (COUNTER_TRADE_PREMIUM_COLUMN, COUNTER_TRADE_BID_PREMIUM_COLUMN,
                                COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN, COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN,
                                COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN) if c in links.columns]
    if COUNTER_TRADE_PREMIUM_COLUMN not in links.columns or not any(
            (pd.to_numeric(links[c], errors="coerce").fillna(0.0) > 0.0).any() for c in paid_columns):
        return pd.DataFrame(0.0, index=flow.index, columns=flow.columns)
    schedule = _schedule_frame(schedule_scenario or scenario, "wholesale_links", flow.index)
    schedule = schedule.reindex(columns=flow.columns).fillna(0.0)
    return counter_trade_premium_cost(flow, schedule, links, ask_t)


def measure_link_trade(scenarios: Sequence[str], stage: str = "balancing",
                       schedule_scenarios: Sequence[str] | None = None) -> dict:
    """Cross-border and internal link energy and cost over a solved year.

    The foreign price sits on the interconnector links, not on the import
    generators, which carry no marginal cost at all; a measurement that reads
    only generators values every import at nothing.  Flows are not floored at
    zero, because an export earns the foreign price as genuinely as an import
    pays it, so a net-exporting year shows a negative cost.

    Signed interconnectors are valued as the LP charged them
    (``market_coupling.interconnector_trade_cost``: import at the ask, export at
    efficiency times the bid, plus the volume tranches).  Before item IC1 every
    flow was valued at the import ask in both directions, which is where most of
    the institution effect on the nationalised bill came from.  Internal links
    are still energy times their marginal cost.

    ``stage="wholesale"`` measures the Stage 1 schedule (``wholesale_links.csv``)
    on the same link tables and weights, for the real-time correction on this
    module's own basis (:func:`realtime_correction_on_line_basis`).

    ``schedule_scenarios`` pairs each month with the month whose Stage 1 files it
    reads (its ``wholesale_from`` parent: the P2/P3 balancing months share their
    parent's Stage 1, item BM1); default each month itself.
    """
    from scripts.interconnectors.market_coupling import (
        _piecewise_links, dataset_snapshots, interconnector_trade_cost, link_price_tables_from_dataset,
    )

    _stage(stage)
    schedules = list(schedule_scenarios) if schedule_scenarios is not None else list(scenarios)
    if len(schedules) != len(scenarios):
        raise ValueError(f"{len(scenarios)} months but {len(schedules)} Stage 1 months; they must pair month by month")
    totals = {"interconnector_energy_mwh": 0.0, "interconnector_cost_gbp": 0.0,
              "internal_energy_mwh": 0.0, "internal_cost_gbp": 0.0, "interconnector_links": 0,
              # Inside interconnector_cost_gbp, reported apart (balancing stage only).
              "counter_trade_premium_gbp": 0.0}
    for scenario, schedule_scenario in zip(scenarios, schedules):
        path = _balancing_path(scenario)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; every month must be solved")
        with xr.open_dataset(path, engine="netcdf4") as ds:
            links, ask_t = link_price_tables_from_dataset(ds)
            weights = pd.Series(ds["snapshots_generators"].values, dtype=float).to_numpy()
            flow = ds["links_t_p0"].to_pandas()
            flow.index = dataset_snapshots(ds, flow.index)
        flow.columns = flow.columns.astype(str)
        if stage == "wholesale":
            flow = _schedule_frame(schedule_scenario, "wholesale_links", flow.index).reindex(
                columns=flow.columns).fillna(0.0)
        names = links.index
        mc = pd.to_numeric(links["marginal_cost"], errors="coerce").astype(float)
        bus0, bus1 = links["bus0"].astype(str), links["bus1"].astype(str)
        weighted = flow.mul(weights, axis=0)
        energy = weighted.sum()
        energy = energy[[c for c in energy.index if c in mc.index]]
        cross_border = pd.Index(energy.index).isin(names[bus0.str.contains("External") | bus1.str.contains("External")])
        signed = [c for c in _piecewise_links(links).values() if c in energy.index]
        flat = [c for c in energy.index[cross_border] if c not in signed]
        cost = float((energy[flat] * mc[flat]).sum())
        if signed:
            cost += float(interconnector_trade_cost(flow[signed], links, ask_t).mul(weights, axis=0).to_numpy().sum())
            if stage == "balancing":
                # Review 28 Sep 2026: the counter-trade premium the balancing stage paid
                # to move a cable off its day-ahead schedule, as analyze_market adds it to
                # re-trading. It is paid abroad, so it is a cost under every institution;
                # measuring trade value alone dropped it from the nationalised tariff.
                premium = counter_trade_premium(scenario, flow[signed], links, schedule_scenario, ask_t)
                totals["counter_trade_premium_gbp"] += float(premium.mul(weights, axis=0).to_numpy().sum())
                cost += float(premium.mul(weights, axis=0).to_numpy().sum())
        rate = mc.reindex(energy.index)
        totals["interconnector_energy_mwh"] += float(energy[cross_border].sum())
        totals["interconnector_cost_gbp"] += cost
        totals["internal_energy_mwh"] += float(energy[~cross_border].sum())
        totals["internal_cost_gbp"] += float((energy[~cross_border] * rate[~cross_border]).sum())
        totals["interconnector_links"] = int(cross_border.sum())
    return totals


def measure_congestion_rent(scenarios: Sequence[str], price_scenarios: Sequence[str] | None = None,
                            price_paths: Sequence | None = None) -> pd.Series:
    """Whole-cable Stage 1 congestion rent per interconnector over a solved year, GBP by cable.

    Each month's day-ahead schedule (``{scenario}_wholesale_links.csv``) valued at
    the GB price of ``price_scenarios`` (default the month itself): the market's
    own copperplate price for the market year, and the market twin's copperplate
    price, FTI's national price, for a central-dispatch year.  The foreign side is
    what the LP charged (:func:`scripts.market.fti_settlement.congestion_rent_by_cable`,
    corrected export-loss convention); link prices come from the month's saved
    balancing network, which shares the Stage 1 build.

    ``price_paths`` (item F12, 12 Oct 2026; one per month) are price files read in place
    of ``{price_scenario}_wholesale_price.csv``: a price-level twin run group's mapped
    month prices (the GB side re-priced, the foreign side and the flows as solved).
    """
    from scripts.interconnectors.market_coupling import dataset_snapshots, link_price_tables_from_dataset
    from scripts.market.fti_settlement import congestion_rent_by_cable

    prices = list(price_scenarios) if price_scenarios is not None else list(scenarios)
    if len(prices) != len(scenarios):
        raise ValueError(f"{len(scenarios)} months but {len(prices)} price months; they must pair month by month")
    files = ([Path(p) for p in price_paths] if price_paths is not None
             else [MARKET_DIR / f"{price_scenario}_wholesale_price.csv" for price_scenario in prices])
    if len(files) != len(scenarios):
        raise ValueError(f"{len(scenarios)} months but {len(files)} price files; they must pair month by month")
    total = pd.Series(dtype=float)
    for scenario, price_file in zip(scenarios, files):
        paths = {"links": MARKET_DIR / f"{scenario}_wholesale_links.csv",
                 "price": price_file, "network": _stage_1_network_path(scenario)}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"congestion rent needs {missing}; every month must be solved")
        with xr.open_dataset(paths["network"], engine="netcdf4") as ds:
            links, ask_t = link_price_tables_from_dataset(ds)
            weights = pd.Series(ds["snapshots_generators"].values, dtype=float)
            weights.index = dataset_snapshots(ds, weights.index)
        flows = pd.read_csv(paths["links"], index_col=0, parse_dates=True)
        price = pd.read_csv(paths["price"], index_col=0, parse_dates=True)["wholesale_price"]
        rent = congestion_rent_by_cable(flows, price.reindex(flows.index), links, ask_t, weights.reindex(flows.index))
        total = total.add(rent, fill_value=0.0)
    return total


def interconnector_capacity_by_cable(scenario: str) -> dict[str, float]:
    """Rating (MW) of each signed interconnector in a solved month, keyed by cable name."""
    from scripts.interconnectors.market_coupling import _piecewise_links, link_price_tables_from_dataset

    path = _balancing_path(scenario)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    with xr.open_dataset(path, engine="netcdf4") as ds:
        links, _ = link_price_tables_from_dataset(ds)
    return {cable: float(pd.to_numeric(links.at[link, "p_nom"], errors="coerce"))
            for cable, link in _piecewise_links(links).items()}


#: The build-time rates :func:`measure_dispatch` prices energy at.
MEASURED_RATE_COLUMNS = ("fuel_cost_gbp_per_mwh", "carbon_cost_gbp_per_mwh", "variable_om_gbp_per_mwh")


def _dispatch_on_turn_down_basis(dispatch: pd.DataFrame, path: Path, rates: pd.DataFrame, scenario: str,
                                 totals: dict) -> pd.DataFrame:
    """One balancing month's generator output on the turn-down basis its bill reads (24 Oct 2026).

    The I1 tariff's register reads each market month through ``balancing_dispatch_in_force``
    (pro rata zonal on the CENTRAL network), so the market measurement the institution step
    subtracts must be on that basis too. The re-share moves wind turn-down between wind units
    with the zone-hour volume conserved; at build-time rates those units carry no fuel, carbon
    or variable O&M (checked on CENTRAL, 24 Oct 2026: 30.6 TWh moved, GBP 0 on every line), so
    the re-shared output is read only when a re-shared unit has a rate; either way the month's
    basis is recorded in ``totals["turn_down_basis_by_month"]``.
    """
    from scripts.market.turn_down_allocation import balancing_dispatch_in_force, month_basis, month_files

    basis = month_basis(path)
    totals["turn_down_basis_by_month"][scenario] = basis
    if basis == "cheapest_first":
        return dispatch
    units = pd.read_csv(month_files(path)[0], usecols=["unit"])["unit"].astype(str).unique()
    rated = rates.reindex(units).reindex(columns=list(MEASURED_RATE_COLUMNS))
    if not float(rated.apply(pd.to_numeric, errors="coerce").fillna(0.0).abs().to_numpy().sum()) > 0.0:
        return dispatch
    totals["turn_down_reshared_units_rated"] = True
    dispatch = dispatch.copy()
    dispatch.columns = dispatch.columns.astype(str)
    return balancing_dispatch_in_force(dispatch, path)


def measure_dispatch(scenarios: Sequence[str], rates: pd.DataFrame,
                     heat_credit: Mapping[str, Mapping[str, float]] | None = None,
                     stage: str = "balancing") -> dict:
    """Energy, fuel, carbon, variable O&M and cross-border trade over a solved year.

    ``rates`` is the build-time marginal-cost breakdown, indexed by generator,
    carrying ``fuel_cost_gbp_per_mwh``, ``carbon_cost_gbp_per_mwh`` and
    ``variable_om_gbp_per_mwh``.  Both years share one network build, so the
    same rates apply to both and any difference is dispatch alone.

    Review B13: ``fuel_gbp`` and ``carbon_gbp`` are the electricity's share, net of
    the heat credit of CHP units (``heat_credit``, :func:`heat_credit_settings`;
    default :data:`HEAT_CREDIT_DEFAULTS`), and the credit is reported beside them.
    ``carbon_gbp`` is split into UK ETS and carbon price support by the
    breakdown's per-tonne rates when it carries them (else all UK ETS): carbon is
    an Exchequer receipt, not a resource cost, and callers keep it apart.

    Imports are valued at the generator's own marginal cost, which for the
    cross-border carriers is the modelled foreign price; exports (negative
    energy) therefore net off at that price.

    ``stage="wholesale"`` measures the Stage 1 schedule
    (``wholesale_dispatch.csv``) at the same rates, so the schedule and the
    physical dispatch are on one basis (:func:`realtime_correction_on_line_basis`).

    Turn-down basis (24 Oct 2026): a balancing month is read on the basis its bill reads
    (``scripts/market/turn_down_allocation.py``; :func:`_dispatch_on_turn_down_basis`), so the
    market year's measurement and the I1 tariff it is subtracted from share one basis. A
    central-dispatch month has no BM turn-down and is read as solved.
    """
    _stage(stage)
    totals = {"energy_mwh": 0.0, "fuel_gbp": 0.0, "carbon_gbp": 0.0, "variable_om_gbp": 0.0,
              "carbon_uk_ets_gbp": 0.0, "carbon_price_support_gbp": 0.0,
              "chp_heat_credit_fuel_gbp": 0.0, "chp_heat_credit_carbon_gbp": 0.0,
              "import_energy_mwh": 0.0, "import_cost_gbp": 0.0, "months": 0,
              "turn_down_basis_by_month": {}, "turn_down_reshared_units_rated": False}
    credit = heat_credit_rates(rates, HEAT_CREDIT_DEFAULTS if heat_credit is None else heat_credit)
    nan = pd.Series(np.nan, index=rates.index)
    ets = pd.to_numeric(rates.get("carbon_uk_ets_gbp_per_tco2", nan), errors="coerce")
    cps = pd.to_numeric(rates.get("carbon_price_support_gbp_per_tco2", nan), errors="coerce")
    ets_share = (ets / (ets + cps)).where((ets + cps) > 0).fillna(1.0)
    by_carrier: dict[str, float] = {}
    for scenario in scenarios:
        path = _balancing_path(scenario)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; every month must be solved")
        with xr.open_dataset(path, engine="netcdf4") as ds:
            names = pd.Index(ds["generators_i"].values)
            carrier = pd.Series(ds["generators_carrier"].values, index=names).astype(str)
            mc = pd.Series(ds["generators_marginal_cost"].values, index=names, dtype=float)
            weights = pd.Series(ds["snapshots_generators"].values, dtype=float).to_numpy()
            dispatch = ds["generators_t_p"].to_pandas()
            if stage == "wholesale":
                from scripts.interconnectors.market_coupling import dataset_snapshots

                snapshots = dataset_snapshots(ds, dispatch.index)
        if stage == "wholesale":
            dispatch.columns = dispatch.columns.astype(str)
            dispatch = _schedule_frame(scenario, "wholesale_dispatch", snapshots) \
                .reindex(columns=dispatch.columns).fillna(0.0)
        else:
            dispatch = _dispatch_on_turn_down_basis(dispatch, path, rates, scenario, totals)
        energy = dispatch.mul(weights, axis=0).sum()
        aligned = rates.reindex(energy.index)
        gross = energy.clip(lower=0.0)
        totals["energy_mwh"] += float(gross.sum())
        heat = credit.reindex(energy.index).fillna(0.0)
        fuel_rate = aligned["fuel_cost_gbp_per_mwh"].fillna(0.0) - heat["heat_credit_fuel_gbp_per_mwh"]
        carbon = gross * (aligned["carbon_cost_gbp_per_mwh"].fillna(0.0) - heat["heat_credit_carbon_gbp_per_mwh"])
        share = ets_share.reindex(energy.index).fillna(1.0)
        totals["fuel_gbp"] += float((gross * fuel_rate).sum())
        totals["carbon_gbp"] += float(carbon.sum())
        totals["carbon_uk_ets_gbp"] += float((carbon * share).sum())
        totals["carbon_price_support_gbp"] += float((carbon * (1.0 - share)).sum())
        totals["variable_om_gbp"] += float((gross * aligned["variable_om_gbp_per_mwh"].fillna(0.0)).sum())
        totals["chp_heat_credit_fuel_gbp"] += float((gross * heat["heat_credit_fuel_gbp_per_mwh"]).sum())
        totals["chp_heat_credit_carbon_gbp"] += float((gross * heat["heat_credit_carbon_gbp_per_mwh"]).sum())
        imports = carrier.reindex(energy.index).isin(IMPORT_CARRIERS)
        totals["import_energy_mwh"] += float(energy[imports].sum())
        # Reported for information only: the import generators carry no marginal
        # cost, so this is not the cost of the energy. The trade price sits on
        # the links and :func:`measure_link_trade` is what values it.
        totals["import_generator_cost_info_gbp"] = totals.get("import_generator_cost_info_gbp", 0.0) + float(
            (energy[imports] * mc.reindex(energy.index)[imports].fillna(0.0)).sum())
        grouped = gross.groupby(carrier.reindex(gross.index).values).sum()
        for name, value in grouped.items():
            by_carrier[name] = by_carrier.get(name, 0.0) + float(value)
        totals["months"] += 1
    totals["fuel_carbon_gbp"] = totals["fuel_gbp"] + totals["carbon_gbp"]
    totals["energy_by_carrier_mwh"] = by_carrier
    return totals


#: The measurement keys the three moved lines are built from (fuel and carbon,
#: variable O&M, cross-border trade); the real-time correction is taken on these.
LINE_BASIS_KEYS = ("fuel_carbon_gbp", "variable_om_gbp", "interconnector_cost_gbp")


def realtime_correction_on_line_basis(physical: Mapping, schedule: Mapping) -> float:
    """The central year's real-time correction on the basis of the moved lines, GBP.

    The fuel, O&M and trade lines are measured on physical dispatch at the build's
    fuel, carbon and variable O&M rates (net of the CHP heat credit) and the
    links' trade prices, so the correction inside them is exactly that same
    measurement on physical dispatch less the same measurement on the Stage 1
    schedule. The settlement's ``central_realtime_correction_gbp_m`` is on another
    basis (the network's as-built marginal costs floored at zero, every internal
    link at its marginal cost, the start and counter-trade premia), so taking it
    out of these lines removed amounts they never contained (review 27 Sep 2026).
    """
    return float(sum(float(physical[k]) - float(schedule[k]) for k in LINE_BASIS_KEYS))


def substitute_lines(i1_lines: Mapping[str, float], market: Mapping, central: Mapping, *,
                     realtime_correction_gbp: float,
                     forecast_error_charge_gbp: float = 0.0,
                     replace_realtime_correction: bool | None = None,
                     central_foreign_rent_gbp: float | None = None) -> tuple[dict[str, float], dict]:
    """The I3 tariff lines: I1 carried across, with the four measured lines moved.

    Each moved line is the I1 amount plus the measured difference between the
    two years, so the level stays on the register's own construction and only
    the dispatch effect is taken from the measurement.  The interconnector
    re-trading line goes to zero (there is no day-ahead position to unwind).

    The measured lines are read from the central year's *physical* dispatch, so
    the model's own real-time correction (physical less schedule resource cost,
    ``realtime_correction_gbp``) is already inside them and is never added again.
    When the borrowed forecast-error charge replaces it (user decision, 26 Sep
    2026, as in the settlement's headline) the real-time line takes it back out,
    so the I3 bill is the schedule plus the charge.  ``replace_realtime_correction``
    defaults to "a charge is supplied"; the CLI passes the settlement's own flag.

    Review B13: the fuel line pays fuel *and* carbon, but carbon is a receipt of
    the Exchequer, not a resource cost, so the audit splits the line's delta and
    the generator resource cost excludes carbon (``carbon_delta_gbp`` is reported
    as an Exchequer receipt change).  Both measurements are net of the CHP heat
    credit (:func:`measure_dispatch`).

    Item D3: the congestion rent owed to the foreign halves of the cables (and to
    whole other-jurisdiction cables) is re-measured on the central schedule at the
    national price (``central_foreign_rent_gbp``); the GB-half acquisition annuity
    is capacity-driven, a price paid once for the asset, and is carried across.
    """
    lines = dict(i1_lines)
    if float(i1_lines.get(LINE_IC_FOREIGN_RENT, 0.0)) != 0.0 and central_foreign_rent_gbp is None:
        raise ValueError(f"the I1 tariff carries {LINE_IC_FOREIGN_RENT}; the I3 case needs the foreign-half rent "
                         "on the central schedule (central_foreign_rent_gbp)")
    deltas = {
        LINE_FUEL: central["fuel_carbon_gbp"] - market["fuel_carbon_gbp"],
        LINE_VARIABLE_OM: central["variable_om_gbp"] - market["variable_om_gbp"],
        # Cross-border energy is priced on the links, never on the import
        # generators; the tariff's own import line is on the same basis.
        LINE_IMPORTS: central["interconnector_cost_gbp"] - market["interconnector_cost_gbp"],
    }
    for key, delta in deltas.items():
        lines[key] = float(lines.get(key, 0.0)) + float(delta)
    foreign_rent_delta = 0.0
    if central_foreign_rent_gbp is not None:
        foreign_rent_delta = float(central_foreign_rent_gbp) - float(i1_lines.get(LINE_IC_FOREIGN_RENT, 0.0))
        lines[LINE_IC_FOREIGN_RENT] = float(central_foreign_rent_gbp)
        deltas[LINE_IC_FOREIGN_RENT] = foreign_rent_delta
    lines[LINE_IC_RETRADING] = 0.0
    # Review 26 Sep 2026: the physical lines already carry the correction, so it is
    # taken back out when the charge replaces it and otherwise left where it is.
    replace = (float(forecast_error_charge_gbp) != 0.0 if replace_realtime_correction is None
               else bool(replace_realtime_correction))
    removed = float(realtime_correction_gbp) if replace else 0.0
    lines[LINE_REALTIME] = -removed
    lines[LINE_FORECAST_ERROR] = float(forecast_error_charge_gbp)
    audit = {
        "measured_deltas_gbp": {k: float(v) for k, v in deltas.items()},
        "interconnector_retrading_removed_gbp": float(i1_lines.get(LINE_IC_RETRADING, 0.0)),
        "realtime_correction_in_physical_lines_gbp": float(realtime_correction_gbp),
        # 24 Oct 2026: the flag itself, so the step can be re-derived on another tariff case (derive_case_step).
        "realtime_correction_replaced": bool(replace),
        "realtime_correction_replaced_by_charge_gbp": removed,
        "forecast_error_charge_added_gbp": float(forecast_error_charge_gbp),
        # Review B13: the fuel line's delta, split; carbon is an Exchequer receipt.
        "fuel_delta_gbp": float(central["fuel_gbp"] - market["fuel_gbp"]),
        "carbon_delta_gbp": float(central["carbon_gbp"] - market["carbon_gbp"]),
        "chp_heat_credit_market_gbp": float(market.get("chp_heat_credit_fuel_gbp", 0.0)
                                            + market.get("chp_heat_credit_carbon_gbp", 0.0)),
        "chp_heat_credit_central_gbp": float(central.get("chp_heat_credit_fuel_gbp", 0.0)
                                             + central.get("chp_heat_credit_carbon_gbp", 0.0)),
        "interconnector_foreign_half_rent_delta_gbp": foreign_rent_delta,
        "interconnector_gb_half_acquisition_carried_gbp": float(i1_lines.get(LINE_IC_ACQUISITION, 0.0)),
        # Fuel and variable O&M only (review B13: carbon is not a resource cost).
        "generator_resource_cost_market_gbp": market["fuel_gbp"] + market["variable_om_gbp"],
        "generator_resource_cost_central_gbp": central["fuel_gbp"] + central["variable_om_gbp"],
    }
    if abs(audit["fuel_delta_gbp"] + audit["carbon_delta_gbp"] - deltas[LINE_FUEL]) > 1.0:
        raise ValueError("fuel plus carbon does not add up to the measured fuel-line delta")
    resource_market = audit["generator_resource_cost_market_gbp"]
    audit["generator_resource_cost_change_share"] = (
        (audit["generator_resource_cost_central_gbp"] - resource_market) / resource_market
        if resource_market else None)
    return lines, audit


def compare(i1_summary: Mapping, market: Mapping, central: Mapping, *, realtime_correction_gbp: float,
            forecast_error_charge_gbp: float = 0.0,
            replace_realtime_correction: bool | None = None,
            central_foreign_rent_gbp: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-line I1 against I3, in GBP per household, and the summary.

    Each line is converted at the rate the tariff recovers it
    (``i1_summary["allocation"]``); a summary written before the like-for-like
    allocation has none and falls back to flat over final demand.  The household
    change is the sum of the per-line changes, so the two ledgers agree.

    F19 P3 decision 6 (user, 6 Oct 2026): the I1 tariff keeps the supplier imbalance
    allowance as its forecast-error cost (``i1_summary["i1_forecast_error_cost"]``); I3 has
    no supplier imbalance and carries FTI's borrowed charge, so the allowance is removed
    here, a row of its own, and counted with the real-time correction the charge replaces
    (``realtime_correction_replaced_by_charge_gbp``, whose part it is now reported as).
    """
    conv = i1_summary["conversion"]
    allocation = i1_summary.get("allocation")
    i1_lines = i1_summary["revenue_requirement_gbp"]
    i3_lines, audit = substitute_lines(i1_lines, market, central, realtime_correction_gbp=realtime_correction_gbp,
                                       forecast_error_charge_gbp=forecast_error_charge_gbp,
                                       replace_realtime_correction=replace_realtime_correction,
                                       central_foreign_rent_gbp=central_foreign_rent_gbp)
    imbalance = i1_summary.get("i1_forecast_error_cost") or {}
    imbalance_system = float(imbalance.get("system_gbp", 0.0) or 0.0)
    imbalance_hh = float(imbalance.get("gbp_per_3100kwh_ex_vat", 0.0) or 0.0)
    audit["i1_imbalance_allowance_removed_gbp"] = imbalance_system
    audit["realtime_correction_model_own_replaced_gbp"] = audit["realtime_correction_replaced_by_charge_gbp"]
    audit["realtime_correction_replaced_by_charge_gbp"] = (
        float(audit["realtime_correction_replaced_by_charge_gbp"]) + imbalance_system)

    def hh(line_id: str, gbp: float) -> float:
        return float(gbp) * line_rate(allocation, conv, line_id)

    rows = []
    for line_id in dict.fromkeys(list(i1_lines) + list(i3_lines)):
        i1_gbp, i3_gbp = float(i1_lines.get(line_id, 0.0)), float(i3_lines.get(line_id, 0.0))
        rows.append({
            "component_id": line_id,
            "i1_system_gbp": i1_gbp, "i3_system_gbp": i3_gbp, "change_system_gbp": i3_gbp - i1_gbp,
            "i1_gbp_per_3100kwh_ex_vat": hh(line_id, i1_gbp), "i3_gbp_per_3100kwh_ex_vat": hh(line_id, i3_gbp),
            "change_gbp_per_3100kwh_ex_vat": hh(line_id, i3_gbp - i1_gbp),
            "basis": ("measured from both solved years" if line_id in audit["measured_deltas_gbp"]
                      else "institution line" if line_id in (LINE_IC_RETRADING, LINE_REALTIME)
                      else "borrowed from FTI (the model produces almost no forecast-error cost)"
                      if line_id == LINE_FORECAST_ERROR else CARRIED_UNCHANGED_NOTE),
        })
    if imbalance_system:
        rows.append({
            "component_id": str(imbalance.get("component_id", "procurement_imbalance")),
            "i1_system_gbp": imbalance_system, "i3_system_gbp": 0.0, "change_system_gbp": -imbalance_system,
            "i1_gbp_per_3100kwh_ex_vat": imbalance_hh, "i3_gbp_per_3100kwh_ex_vat": 0.0,
            "change_gbp_per_3100kwh_ex_vat": -imbalance_hh,
            "basis": "institution line: I1's supplier imbalance allowance, replaced under I3 by FTI's charge (F19)",
        })
    frame = pd.DataFrame(rows).sort_values("change_system_gbp", key=abs, ascending=False).reset_index(drop=True)
    change_system = float(frame["change_system_gbp"].sum())
    change_hh = float(frame["change_gbp_per_3100kwh_ex_vat"].sum())
    forecast_error_hh = hh(LINE_FORECAST_ERROR, forecast_error_charge_gbp)
    removed_hh = hh(LINE_REALTIME, audit["realtime_correction_model_own_replaced_gbp"]) + imbalance_hh
    vat = 1.0 + float(conv["vat_rate"])
    bill = i1_summary["bill"]
    i1_ex = float(bill["ex_vat_gbp"])
    private_inc = float(bill["private_total_gbp_inc_vat"])
    # Both deltas are computed here from the bill quantities rather than taken
    # from the tariff's reported delta, so the identity
    # delta = bill - private holds on each side whatever rounding the input
    # carried; the tariff's own figure is kept beside them as a cross-check.
    i1_inc, i3_inc = i1_ex * vat, (i1_ex + change_hh) * vat
    summary = {
        "i1_ex_vat_gbp": i1_ex, "i3_ex_vat_gbp": i1_ex + change_hh,
        "i1_inc_vat_gbp": i1_inc, "i3_inc_vat_gbp": i3_inc,
        "private_ex_vat_gbp": float(bill["private_ex_vat_gbp"]),
        "private_inc_vat_gbp": private_inc,
        "i1_delta_vs_private_inc_vat_gbp": i1_inc - private_inc,
        "i3_delta_vs_private_inc_vat_gbp": i3_inc - private_inc,
        "i1_delta_reported_by_tariff_inc_vat_gbp": float(bill["delta_inc_vat_gbp"]),
        "institution_change_ex_vat_gbp": change_hh,
        "institution_change_inc_vat_gbp": change_hh * vat,
        "institution_change_system_gbp": change_system,
        # The institution change before the borrowed charge, i.e. what the model
        # itself measures on physical dispatch, its own correction included; the
        # headline above is net of the charge, which replaces that correction.
        "forecast_error_charge_gbp": float(forecast_error_charge_gbp),
        "forecast_error_charge_ex_vat_gbp": forecast_error_hh,
        "institution_change_gross_inc_vat_gbp": (change_hh - forecast_error_hh + removed_hh) * vat,
        # Review B13: the carbon inside the fuel line is a receipt of the Exchequer.
        # The bill carries it (the state's plant pays its carbon), so the headline
        # above is unchanged; net of that receipt the institution changes the
        # country's cost by the resource figure below.
        "exchequer_carbon_receipt_change_gbp": audit["carbon_delta_gbp"],
        "exchequer_carbon_receipt_change_inc_vat_gbp": hh(LINE_FUEL, audit["carbon_delta_gbp"]) * vat,
        "institution_change_net_of_carbon_receipt_inc_vat_gbp": (change_hh - hh(LINE_FUEL, audit["carbon_delta_gbp"])) * vat,
        "measurement": {
            "market": {k: v for k, v in market.items() if k != "energy_by_carrier_mwh"},
            "central": {k: v for k, v in central.items() if k != "energy_by_carrier_mwh"},
        },
        "audit": audit,
    }
    return frame, summary


#: User decision (24 Oct 2026): the label on every I3 figure derived for a P3 sensitivity.
DERIVED_STEP_LABEL = ("central dispatch, additive approximation: the case's market-dispatch (I1) figure plus the "
                      "institution step measured on P3's central case (fuel and carbon, variable O&M, cross-border "
                      "trade, the institution lines), a dispatch effect independent of the purchase terms")


def derive_case_step(i1_case: Mapping, institution: Mapping) -> dict:
    """The measured I1-to-I3 step applied to another tariff case that owns the same fleet (24 Oct 2026).

    User decision: P3's headline is central dispatch (I3), but only one case is solved under it;
    every other same-fleet tariff case is reported on I3 as its own I1 tariff plus the measured
    step. The step is re-run through :func:`compare` on the case's own I1 lines and recovery
    rates with the institution summary's two measured years, real-time correction, forecast-error
    charge and central foreign-half rent, so the institution case's own twin reproduces the
    measured I3 bill exactly and a case recovering a line at another rate is converted at its
    own. Nothing the purchase terms set (acquisition annuity, thermal basis, term) is a moved
    line, so the step does not depend on them; a case owning another fleet is the caller's to
    exclude (:func:`same_fleet_as`).
    """
    measurement = institution.get("measurement") or {}
    market, central = measurement.get("market"), measurement.get("central")
    if not market or not central:
        raise ValueError("the institution summary carries no measurement of its two years; rebuild it")
    audit = institution.get("audit") or {}
    correction = float(audit["realtime_correction_in_physical_lines_gbp"])
    charge = float(institution.get("forecast_error_charge_gbp") or 0.0)
    replace = audit.get("realtime_correction_replaced")
    if replace is None:
        # A summary written before 24 Oct 2026 states what was removed, not the flag.
        own = audit.get("realtime_correction_model_own_replaced_gbp")
        if own is None:
            raise ValueError("the institution summary predates F19 (no realtime_correction_model_own_replaced_gbp); "
                             "rebuild it")
        replace = float(own) != 0.0 if correction != 0.0 else charge != 0.0
    rent = (institution.get("interconnector_rent_i3") or {}).get("central_foreign_half_rent_gbp")
    _, summary = compare(i1_case, market, central, realtime_correction_gbp=correction,
                         forecast_error_charge_gbp=charge, replace_realtime_correction=bool(replace),
                         central_foreign_rent_gbp=None if rent is None else float(rent))
    summary["i1_case_id"] = i1_case.get("case_id")
    summary["derived_from_institution_case_id"] = institution.get("institution_case_id")
    summary["approximation"] = DERIVED_STEP_LABEL
    return summary


#: The tariff diagnostics that say which plant the state owns and costs (the fleet the step moves).
FLEET_DIAGNOSTICS = ("owned_costed_assets", "owned_costed_capacity_mw", "paid_as_today_capacity_mw")


def same_fleet_as(case: Mapping, twin: Mapping) -> tuple[bool, str | None]:
    """Whether a tariff case owns and costs the same plant as the institution case's twin, and why not.

    The step carries plant paid as today across at its I1 payments and books every moved line
    to owned plant, so it applies only where the owned set is the twin's (the
    ``legacy_*_acquired_contracts_cancelled*`` steps from P2 to P3 own less).
    """
    d, t = case.get("diagnostics") or {}, twin.get("diagnostics") or {}
    for key in FLEET_DIAGNOSTICS:
        if key not in d or key not in t:
            return False, f"the tariff diagnostics carry no {key}"
        if abs(float(d[key]) - float(t[key])) > 1e-6 * max(1.0, abs(float(t[key]))):
            return False, f"owns another fleet ({key} {float(d[key]):,.1f} against the twin's {float(t[key]):,.1f})"
    if sorted(d.get("paid_as_today_carriers") or []) != sorted(t.get("paid_as_today_carriers") or []):
        return False, "pays another set of carriers as today"
    return True, None


def energy_table(market: Mapping, central: Mapping, *, threshold_mwh: float = 1.0e6) -> pd.DataFrame:
    """Energy by carrier under each institution, for the carriers that move."""
    carriers = sorted(set(market["energy_by_carrier_mwh"]) | set(central["energy_by_carrier_mwh"]))
    rows = []
    for name in carriers:
        m = market["energy_by_carrier_mwh"].get(name, 0.0)
        c = central["energy_by_carrier_mwh"].get(name, 0.0)
        if max(abs(m), abs(c)) < threshold_mwh:
            continue
        rows.append({"carrier": name, "market_twh": m / 1e6, "central_twh": c / 1e6,
                     "change_twh": (c - m) / 1e6})
    return pd.DataFrame(rows).sort_values("change_twh", key=abs, ascending=False).reset_index(drop=True)


def replace_flag_from_settlement(frame: pd.DataFrame, column: str, default=None):
    """Whether the settlement replaced the real-time correction with the FTI charge.

    The settlement replaces it whenever the charge is enabled (``fti_settlement``),
    and labels the charge basis ``disabled`` otherwise; that label is the flag.
    Testing the replaced amount for zero instead would silently stop replacing in a
    year whose correction is exactly zero, as a perfect-foresight central year can
    be (review 28 Sep 2026). Settlements written without the label fall back to it.
    """
    if "forecast_error_charge_basis" in frame.index:
        return str(frame.loc["forecast_error_charge_basis", column]).strip().lower() != "disabled"
    if "central_realtime_correction_replaced_gbp_m" in frame.index:
        return float(frame.loc["central_realtime_correction_replaced_gbp_m", column]) != 0.0
    return default


def settlement_fleet_ratio(basis: str) -> float:
    """The fleet ratio a settlement's forecast-error charge was computed at, read off its basis text."""
    import re

    match = re.search(r"fleet ratio ([0-9.]+)", str(basis))
    if not match or float(match.group(1)) <= 0:
        raise ValueError(f"the settlement's forecast-error basis states no fleet ratio: {basis!r}")
    return float(match.group(1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("i1_bill_summary")
    parser.add_argument("out_summary")
    parser.add_argument("out_comparison")
    parser.add_argument("--market-scenarios", nargs="+", required=True)
    parser.add_argument("--market-wholesale-scenarios", nargs="+", default=None,
                        help="Each market month's Stage 1 month (its wholesale_from parent), paired month by month: "
                             "the schedule its counter-trade is measured against, the copperplate price the central "
                             "schedule's rent is valued at, and the Stage 1 files in its solve identity. Default: the "
                             "market months themselves (18 Oct 2026: cp30_network reads stage1's)")
    parser.add_argument("--central-scenarios", nargs="+", required=True)
    parser.add_argument("--marginal-costs", required=True,
                        help="Build-time marginal-cost breakdown CSV shared by both years")
    parser.add_argument("--fti-settlement", default=None,
                        help="Year FTI settlement CSV: the borrowed forecast-error charge and whether it "
                             "replaces the real-time correction (the correction itself is measured here on the moved "
                             "lines' basis; the settlement's figure is reported for information)")
    parser.add_argument("--realtime-correction-gbp", type=float, default=None)
    parser.add_argument("--forecast-error-charge-gbp", type=float, default=None,
                        help="Borrowed FTI forecast-error charge under I3 (default: the settlement's YEAR figure)")
    parser.add_argument("--forecast-error-fleet-ratio", type=float, default=None,
                        help="FES 2050 decision 18: rescale the settlement's charge from its own fleet ratio to this one "
                             "(a later year's group block; 2030 never passes it)")
    parser.add_argument("--no-forecast-error-charge", action="store_true",
                        help="Leave the borrowed forecast-error charge off the I3 bill")
    parser.add_argument("--case", default=None,
                        help="The tariff case taken as I1 (the institution case's market twin); default the primary")
    parser.add_argument("--institution-case", default=None, help="The I3 case id, recorded in the summary")
    parser.add_argument("--config", default=None,
                        help="the config directory: the CHP heat credit (marginal_costs.byproduct_adjustments)")
    parser.add_argument("--assumptions", default=None,
                        help="Forecast bill assumptions JSON: the interconnector GB-share table (item D3). "
                             "Required when the I1 tariff carries the foreign-half rent line")
    args = parser.parse_args(argv)

    tariff = json.loads(Path(args.i1_bill_summary).read_text(encoding="utf-8"))
    i1_case = args.case or tariff.get("primary_case_id")
    if args.case is not None and args.case not in (tariff.get("cases") or {}):
        raise ValueError(f"{args.case} is not a case in {args.i1_bill_summary}; "
                         f"tariff cases: {sorted(tariff.get('cases') or {})}")
    i1 = tariff["cases"][args.case] if args.case is not None else tariff
    rates = pd.read_csv(args.marginal_costs).drop_duplicates("generator").set_index("generator")
    correction = args.realtime_correction_gbp
    charge = args.forecast_error_charge_gbp
    settlement_note = "supplied on the command line"
    charge_note = "supplied on the command line"
    basis = ""
    replace = None  # None: replace whenever a charge is supplied
    settlement_correction = None  # the settlement's own figure, another basis: information only
    if args.fti_settlement:
        frame = pd.read_csv(args.fti_settlement).set_index("metric")
        column = "YEAR" if "YEAR" in frame.columns else frame.columns[-1]
        settlement_correction = float(frame.loc["central_realtime_correction_gbp_m", column]) * 1e6
        if charge is None and "forecast_error_charge_gbp_m" in frame.index:
            charge = float(frame.loc["forecast_error_charge_gbp_m", column]) * 1e6
            basis = frame.loc["forecast_error_charge_basis", column] if "forecast_error_charge_basis" in frame.index else ""
            charge_note = f"{args.fti_settlement} ({column}; {basis})"
        replace = replace_flag_from_settlement(frame, column, default=replace)
    applied_ratio = None
    if args.forecast_error_fleet_ratio is not None and not args.no_forecast_error_charge:
        # FES 2050 decision 57: a group's declared ratio must reach the charge, never be skipped
        # silently (the 3 Oct 2026 I3 bill was built by a command without the flag).
        if args.forecast_error_charge_gbp is not None:
            raise ValueError("--forecast-error-fleet-ratio rescales the settlement's charge; it cannot be combined "
                             "with --forecast-error-charge-gbp")
        if not args.fti_settlement or charge is None:
            raise ValueError("--forecast-error-fleet-ratio needs an FTI settlement carrying a forecast-error charge "
                             "(--fti-settlement); without it decision 18's ratio would not reach the I3 bill")
        # Decision 18: the borrowed charge scales with the wind and solar fleet; the settlement
        # states the ratio it was computed at ("fleet ratio 1.1").
        own = settlement_fleet_ratio(str(basis))
        charge = float(charge) * float(args.forecast_error_fleet_ratio) / own
        applied_ratio = float(args.forecast_error_fleet_ratio)
        charge_note += f"; rescaled from fleet ratio {own:g} to {float(args.forecast_error_fleet_ratio):g} (decision 18)"
    if args.no_forecast_error_charge:
        charge, charge_note, replace = 0.0, "excluded on the command line", False
    elif charge is None:
        charge_note = "no settlement figure: a settlement written before 26 Sep 2026 carries no charge"
    charge = float(charge or 0.0)

    heat = HEAT_CREDIT_DEFAULTS
    if args.config:
        import yaml

        heat = config_heat_credit_settings(args.config)
    # A wholesale_from market year (cp30_network on stage1) owns no Stage 1 files: its parent's are read.
    market_stage_1 = _paired_stage_1(args.market_scenarios, args.market_wholesale_scenarios)
    market = {**measure_dispatch(args.market_scenarios, rates, heat),
              **measure_link_trade(args.market_scenarios, schedule_scenarios=market_stage_1)}
    central = {**measure_dispatch(args.central_scenarios, rates, heat), **measure_link_trade(args.central_scenarios)}
    if correction is None:
        # Review 27 Sep 2026: the correction taken out of the physical lines is
        # measured on those lines' own basis, physical less the Stage 1 schedule.
        central_schedule = {**measure_dispatch(args.central_scenarios, rates, heat, stage="wholesale"),
                            **measure_link_trade(args.central_scenarios, stage="wholesale")}
        correction = realtime_correction_on_line_basis(central, central_schedule)
        settlement_note = ("measured on the moved lines' basis: central physical dispatch less its Stage 1 "
                           "schedule (fuel and carbon, variable O&M, cross-border trade)")
    correction = float(correction)
    # Item D3: the foreign halves' congestion rent on the central schedule, at the
    # national price (the market twin's copperplate price), month by month.
    central_foreign_rent, rent_detail = None, None
    if float((i1.get("revenue_requirement_gbp") or {}).get(LINE_IC_FOREIGN_RENT, 0.0)) != 0.0:
        if not args.assumptions:
            raise ValueError("the I1 tariff carries the foreign-half rent line; pass --assumptions for the GB shares")
        from scripts.market.fti_settlement import gb_half_shares, load_interconnector_rule

        shares = pd.Series(gb_half_shares(load_interconnector_rule(args.assumptions)))
        rent = measure_congestion_rent(args.central_scenarios, price_scenarios=market_stage_1)
        unknown = sorted(set(rent.index) - set(shares.index))
        if unknown:
            raise ValueError(f"interconnectors with rent but no declared regime: {unknown}")
        central_foreign_rent = float((rent * (1.0 - shares.reindex(rent.index))).sum())
        rent_detail = {"central_schedule_rent_at_national_price_gbp": {k: float(v) for k, v in rent.items()},
                       "gb_share": {k: float(shares[k]) for k in rent.index},
                       "central_foreign_half_rent_gbp": central_foreign_rent}
    frame, summary = compare(i1, market, central, realtime_correction_gbp=correction,
                             forecast_error_charge_gbp=charge, replace_realtime_correction=replace,
                             central_foreign_rent_gbp=central_foreign_rent)
    summary["chp_heat_credit_settings"] = heat
    if rent_detail is not None:
        summary["interconnector_rent_i3"] = rent_detail
    energy = energy_table(market, central)
    summary["energy_by_carrier_twh"] = energy.to_dict("records")
    summary["realtime_correction_source"] = settlement_note
    summary["realtime_correction_settlement_basis_info_gbp"] = settlement_correction
    summary["forecast_error_charge_source"] = charge_note
    if applied_ratio is not None:
        # Decision 57: the ratio the charge was rescaled to, which the Exchequer ledger checks.
        summary["forecast_error_fleet_ratio"] = applied_ratio
    summary["i1_case_id"] = i1_case
    summary["institution_case_id"] = args.institution_case
    # F19 (user decision, 6 Oct 2026): both solved years' identities, and the mixed-vintage flag.
    gap_hours = MIXED_VINTAGE_GAP_HOURS_DEFAULT
    if args.config:
        import yaml

        cfg = read_config(args.config) or {}
        block = ((cfg.get("public_ownership") or cfg).get("institution_comparison") or {})
        gap_hours = float(block.get("mixed_vintage_gap_hours", MIXED_VINTAGE_GAP_HOURS_DEFAULT))
    summary["solve_vintage"] = solve_vintage(solve_identity(args.market_scenarios, market_stage_1),
                                             solve_identity(args.central_scenarios), gap_hours)
    if market_stage_1 != list(args.market_scenarios):
        summary["market_stage_1_scenarios"] = market_stage_1
    summary["mixed_vintage"] = summary["solve_vintage"]["mixed_vintage"]
    summary["notes"] = [
        "I1 = the market machinery kept with every counterparty public (the solved market year). "
        "I3 = central dispatch on a day-ahead forecast with the error corrected at cost.",
        "The constraint payments central dispatch removes are the same money nationalisation removes by "
        "owning both sides of them. They are substitutes, not complements: never add a nationalisation "
        "saving to a central-dispatch saving.",
        "Only four lines can move: fuel and carbon, variable O&M, cross-border trade, and the institution "
        "line. Everything else is capacity-driven and is carried across, which is stated on each row.",
        "Cross-border trade is valued on the links at the foreign price the LP itself cleared on: the item IC1 "
        "supply curve (volume tranches on three coupled markets and an hourly ENTSO-E shape; decision 18, "
        "22 Sep 2026), so each further MWh exported earns less. Read the line as a change in trade position, "
        "not as an efficiency gain.",
        "The model's own forecast-error cost is small because Stage 2 re-optimises each 24-hour window on the "
        "full outturn, so the operator corrects with a day of hindsight and holds no reserve. The I3 bill "
        "therefore carries FTI's forecast-error cost, borrowed and scaled (forecast_error_charge_*), in place "
        "of the model's own correction: the fuel, O&M and trade lines are measured on physical dispatch, so "
        "the real-time line takes that correction back out and I3 is the schedule plus the charge. "
        "institution_change_gross_inc_vat_gbp is the change on physical dispatch without the charge.",
        "Every line is converted at the rate the tariff recovers it (like-for-like allocation, 26 Sep 2026).",
        "Review B13: the fuel line pays fuel and carbon, but carbon is an Exchequer receipt, not a resource cost. "
        "audit.fuel_delta_gbp and audit.carbon_delta_gbp split the line; the generator resource cost and its "
        "change share exclude carbon; institution_change_net_of_carbon_receipt_inc_vat_gbp nets the receipt off. "
        "CHP is measured net of its heat credit (the heat valued at the boiler it displaces).",
        "Item D3: the congestion rent owed to foreign halves (and whole other-jurisdiction cables) is re-measured "
        "on the central schedule at the national price; the GB-half acquisition annuity is capacity-driven and "
        "carried across.",
        "Turn-down basis (24 Oct 2026): the market year is measured on the basis its I1 tariff reads "
        "(measurement.market.turn_down_basis_by_month; pro rata zonal on the CENTRAL network) and the central year as "
        "solved (no BM turn-down). The re-share moves wind between wind units, which carry no fuel, carbon or variable "
        "O&M at build-time rates, so the step is the same on either basis (GBP 0 on CENTRAL); the pro rata uplift on "
        "wind bids sits in the private comparator's constraint line, which the tariff replaces, never in the step.",
        "Every other tariff case owning the same fleet is reported on I3 as its own I1 tariff plus this step "
        "(derive_case_step; the Exchequer ledger's i3_approx_* columns), an additive approximation.",
    ]
    for path in (args.out_summary, args.out_comparison):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_comparison, index=False)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "private_inc_vat": round(summary["private_inc_vat_gbp"], 2),
        "nationalised_I1_inc_vat": round(summary["i1_inc_vat_gbp"], 2),
        "nationalised_I3_inc_vat": round(summary["i3_inc_vat_gbp"], 2),
        "institution_change_inc_vat": round(summary["institution_change_inc_vat_gbp"], 2),
        "generator_resource_cost_change_share": round(audit_share(summary), 5),
    }))


def audit_share(summary: Mapping) -> float:
    value = summary["audit"].get("generator_resource_cost_change_share")
    return float(value) if value is not None else float("nan")


if __name__ == "__main__":
    main()
