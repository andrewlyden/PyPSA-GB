"""Seasonal (hydrogen) storage across independently solved months: the two-pass approach.

Plan decision 33 (user, 1 October 2026), task Y8a. Each market month is solved on
its own as rolling 24 h day-ahead windows, so nothing tells a month how much of the
11.67 TWh hydrogen store it may draw or must leave. The Y7 gate week showed what
that costs: with ``month_end: at_least_initial`` the last day refilled the store on
the coldest, stillest day and shed 815 GWh; with ``free`` and a flat end-of-window
value the store started at an arbitrary 50% and emptied on day 7.

**Pass 1, the annual pass** (this module, rule ``solve_seasonal_storage_pass``): one
copperplate LP over all 8,760 hours with perfect foresight, on the full-year parent
network aggregated by carrier and cost class:

* every GB AC bus becomes one bus (``GB_mainland``). This is the Stage 1 copperplate
  exactly: the mainland collapse is exact (``copperplate.py``) and Stage 1 lifts every
  island HVDC link above what its islands can push (``relax_island_hvdc_links``), so the
  island links cannot bind either; External buses, the hydrogen and gas buses, the
  interconnector links (with their tranches) and the hydrogen store keep their identity;
* generators are grouped by (bus class, carrier, marginal cost): ``p_nom`` summed and
  ``p_max_pu`` / ``p_min_pu`` capacity-weighted, so available energy and must-run are
  preserved hour by hour (exact on one bus: any aggregate dispatch splits back pro rata);
  a unit with a time-varying cost stays on its own; ramp limits are capacity-weighted
  (the aggregate limit is implied by the units', so it is a relaxation);
* storage units by (bus class, carrier, efficiencies, standing loss, costs, max_hours),
  with summed inflows: exact when the members' per-unit profiles coincide (they do for the
  FES consumer-flexibility units, which are one national profile spread by bus share; the
  report records the largest deviation per group);
* links between the same bus classes with the same carrier, efficiency and cost, summed;
  links with both ends on GB AC buses are dropped (zero flow on one bus); loads summed by
  carrier; large hydro becomes a reservoir storage unit (the Stage 1 water budget
  cumulates over every snapshot, which is quadratic in a year; declared).

The pass writes the store's level path ``e(t)`` and its hourly value, the dual of the
store's energy balance (GBP/MWh_H2, equal to the hydrogen bus price whenever the store is
between its bounds), plus diagnostics.

**Pass 2, the monthly runs** (``market_utils``): a scenario with
``market.hydrogen_store.seasonal_trajectory.enabled``

* **starts** each carried store at the trajectory's level at the period start (hard;
  both stages, ``prepare_carried_stores``);
* **values** each Stage 1 window's end level at the annual dual for that hour
  (``end_value: dual``) instead of the flat ``end_of_window_value_gbp_per_mwh_h2``;
* **does not pin the end.** ``end_condition: value`` leaves the period end to that
  value alone; ``soft_floor`` also charges ``shortfall_penalty_gbp_per_mwh_h2`` (P, a
  premium over the end value V) for every MWh_H2 a window ends below the trajectory's
  level at its last hour: every window (``soft_floor_windows: every_window``, default)
  or the period's last only (``period_end``). The store's value is then kinked at the
  path, V above and V + P below. Objective only, stripped from the reported objective.
  V + P must be below the VOLL-equivalent (VOLL x the best turbine efficiency), so no
  window sheds load to keep hydrogen. A hard end pin is not offered: the Y8a gate week
  (``HT50_1week_gate_seasonal``, 1 Oct 2026) pinned the end at the trajectory and shed
  815 GWh on its last day, because the store's flat value (36.75 GBP/MWh_H2, the
  reformer's cost: the store never reaches a bound, so its dual is the same every hour)
  cannot stop myopic days drawing below the path, and the pin then made the coldest day
  refill it. ``month_end`` must be ``free`` with the trajectory;
* with ``storage_units: true``, also **starts every storage unit** (batteries, LDES,
  pumped hydro and the FES consumer-flexibility fleets) at the annual pass's state of
  charge for its carrier at the period start (``apply_initial_storage_soc``, both
  stages) and **values the period's end** state at the annual pass's storage dual for
  that carrier and hour (last window only, so short-duration storage keeps its
  day-ahead myopia inside the month). ``market.storage_soc.month_end`` must then be
  ``free``. Found in the gate week: the annual pass enters 10 January with the EV and
  V2G fleets holding 670 GWh charged ahead of the cold spell, while the rolling run
  starts them at zero and must charge the week's driving inside it (about 550 GWh more
  electricity on days 1 to 6), which the hydrogen turbines supplied.

The BM is unchanged (it reads both starting states through the same helpers).
Defaults are today's behaviour: without the key nothing here runs.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypsa

from scripts.market.copperplate import EXTERNAL_BUS_PATTERN, MAINLAND_HUB

LOG = logging.getLogger(__name__)

#: Annual-pass settings (``market.seasonal_pass`` on the parent scenario). Code
#: defaults; no HT50 entry overrides them yet (declared in the Y8a report).
ANNUAL_PASS_DEFAULTS = {
    "resolution_hours": 1,            # 1, 2 or 3; the trajectory is always written hourly
    "cost_band_gbp_per_mwh": 0.0,     # 0 = one group per distinct marginal cost (exact)
    "store_cyclic": True,             # the year ends where it starts (a representative year)
    "storage_cyclic": True,           # storage units likewise (no free starting energy)
    "large_hydro_as_storage": True,   # reservoir unit instead of the per-snapshot water budget
    "threads": 4,
    # Decision 40 option (2 Oct 2026): hold the carried hydrogen stores at or above this
    # many TWh all year (e_min, split pro rata to e_nom), so the copperplate plan keeps a
    # margin for constraint management. 0 = none (the Y8a pass).
    "hydrogen_reserve_twh": 0.0,
}

#: ``market.hydrogen_store.seasonal_trajectory``: off by default (today's behaviour).
TRAJECTORY_DEFAULTS = {
    "enabled": False,
    "file": None,               # the annual pass output (rules/market.smk fills the parent's path)
    "end_value": "dual",        # dual: the annual dual at each window's last snapshot; flat: keep V
    "end_condition": "value",   # value: no end condition | soft_floor: penalised shortfall at the period end
    "shortfall_penalty_gbp_per_mwh_h2": None,   # soft_floor only: premium P over V; V + P < VOLL x turbine efficiency
    "soft_floor_windows": "every_window",       # soft_floor only: every_window | period_end
    # soft_floor only: GBP/MWh_H2 taken off V for the level above the path (null: V both sides of it)
    "surplus_discount_gbp_per_mwh_h2": None,
    "storage_units": False,     # also start storage units on the pass and value their period end
}
TRAJECTORY_END_VALUES = ("dual", "flat")
TRAJECTORY_END_CONDITIONS = ("value", "soft_floor")
SOFT_FLOOR_WINDOWS = ("every_window", "period_end")
#: Keys of earlier drafts (Y8a, 1 Oct 2026) that held the end hard; refused with a pointer.
RETIRED_TRAJECTORY_KEYS = {
    "window_floor": "a hard floor at every window end; removed (it forces refills on still days)",
}
RETIRED_END_CONDITIONS = ("equal", "at_least")
TRAJECTORY_COLUMNS = ("e_start_mwh", "e_end_mwh", "value_gbp_per_mwh_h2")
#: Storage-unit columns of the trajectory: ``su_soc_end_mwh|<carrier>`` and ``su_value|<carrier>``.
SU_LEVEL_PREFIX = "su_soc_end_mwh|"
SU_VALUE_PREFIX = "su_value|"
#: Carriers the pass models differently from Stage 1 (large hydro: a reservoir here, a
#: water budget there), so their states are not handed to the months.
SU_TRAJECTORY_EXCLUDED_CARRIERS = ("large_hydro",)

LARGE_HYDRO_CARRIERS = ("large_hydro",)
LOAD_SHEDDING_CARRIERS = ("load_shedding", "load shedding", "voll", "VOLL")
VRE_CARRIERS = ("wind_offshore", "wind_onshore", "solar_pv", "marine", "large_hydro", "small_hydro")
H2_STORE_CARRIERS_DEFAULT = ("H2_gas",)


# ── Configuration ─────────────────────────────────────────────────────────

def resolve_seasonal_pass_config(market_config: Optional[dict]) -> dict:
    """``market.seasonal_pass`` with the code defaults filled and validated.

    Args:
        market_config: the scenario's ``market`` block.

    Returns:
        A dict with every key of ``ANNUAL_PASS_DEFAULTS``.
    """
    cfg = dict(ANNUAL_PASS_DEFAULTS)
    cfg.update(dict((market_config or {}).get("seasonal_pass", {}) or {}))
    res = int(cfg["resolution_hours"])
    if res not in (1, 2, 3):
        raise ValueError(f"market.seasonal_pass.resolution_hours must be 1, 2 or 3, got {cfg['resolution_hours']}")
    cfg["resolution_hours"] = res
    cfg["cost_band_gbp_per_mwh"] = float(cfg["cost_band_gbp_per_mwh"] or 0.0)
    if cfg["cost_band_gbp_per_mwh"] < 0:
        raise ValueError("market.seasonal_pass.cost_band_gbp_per_mwh must be >= 0")
    for key in ("store_cyclic", "storage_cyclic", "large_hydro_as_storage"):
        cfg[key] = bool(cfg[key])
    cfg["threads"] = int(cfg["threads"])
    cfg["hydrogen_reserve_twh"] = float(cfg["hydrogen_reserve_twh"] or 0.0)
    if cfg["hydrogen_reserve_twh"] < 0:
        raise ValueError("market.seasonal_pass.hydrogen_reserve_twh must be >= 0")
    return cfg


def apply_hydrogen_reserve(agg, names: List[str], reserve_twh: float, logger: logging.Logger) -> dict:
    """Hold the carried stores at or above ``reserve_twh`` (e_min_pu, pro rata to e_nom); return a report.

    Raises if the reserve is not below the stores' combined e_nom (the pass would be
    infeasible or the store pinned full).
    """
    present = [s for s in names if s in agg.stores.index]
    if reserve_twh <= 0.0 or not present:
        return {"reserve_twh": 0.0}
    e_nom = pd.to_numeric(agg.stores.loc[present, "e_nom"], errors="coerce").fillna(0.0)
    total = float(e_nom.sum())
    reserve_mwh = reserve_twh * 1e6
    if reserve_mwh >= total:
        raise ValueError(f"market.seasonal_pass.hydrogen_reserve_twh {reserve_twh} must be below the stores' "
                         f"e_nom ({total / 1e6:.2f} TWh)")
    floor_pu = reserve_mwh / total
    agg.stores.loc[present, "e_min_pu"] = floor_pu
    logger.info(f"Seasonal pass: hydrogen reserve {reserve_twh:.2f} TWh held all year "
                f"(e_min_pu {floor_pu:.4f} on {len(present)} store(s), e_nom {total / 1e6:.2f} TWh)")
    return {"reserve_twh": reserve_twh, "e_min_pu": floor_pu, "stores": present}


def resolve_seasonal_trajectory_config(market_config: Optional[dict]) -> dict:
    """``market.hydrogen_store.seasonal_trajectory`` with defaults filled and validated.

    Args:
        market_config: the scenario's ``market`` block.

    Returns:
        A dict with every key of ``TRAJECTORY_DEFAULTS``; ``enabled`` False unless set.
    """
    block = ((market_config or {}).get("hydrogen_store", {}) or {}).get("seasonal_trajectory", {}) or {}
    for key, why in RETIRED_TRAJECTORY_KEYS.items():
        if key in block:
            raise ValueError(f"market.hydrogen_store.seasonal_trajectory.{key} is retired: {why}")
    unknown = sorted(set(block) - set(TRAJECTORY_DEFAULTS))
    if unknown:
        raise ValueError(f"market.hydrogen_store.seasonal_trajectory: unknown keys {unknown}")
    cfg = dict(TRAJECTORY_DEFAULTS)
    cfg.update(dict(block))
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["storage_units"] = bool(cfg["storage_units"])
    cfg["end_value"] = str(cfg["end_value"]).lower()
    cfg["end_condition"] = str(cfg["end_condition"]).lower()
    if cfg["end_value"] not in TRAJECTORY_END_VALUES:
        raise ValueError(f"seasonal_trajectory.end_value must be one of {TRAJECTORY_END_VALUES}, got {cfg['end_value']!r}")
    if cfg["end_condition"] in RETIRED_END_CONDITIONS:
        raise ValueError(
            f"seasonal_trajectory.end_condition {cfg['end_condition']!r} is retired: a hard end pin made the Y8a "
            f"gate week's last day refill the store and shed 815 GWh; use 'value' or 'soft_floor'")
    if cfg["end_condition"] not in TRAJECTORY_END_CONDITIONS:
        raise ValueError(
            f"seasonal_trajectory.end_condition must be one of {TRAJECTORY_END_CONDITIONS}, got {cfg['end_condition']!r}")
    penalty = cfg["shortfall_penalty_gbp_per_mwh_h2"]
    if cfg["end_condition"] == "soft_floor":
        if penalty is None or not float(penalty) > 0.0:
            raise ValueError("seasonal_trajectory.end_condition 'soft_floor' needs shortfall_penalty_gbp_per_mwh_h2 > 0")
        cfg["shortfall_penalty_gbp_per_mwh_h2"] = float(penalty)
        cfg["soft_floor_windows"] = str(cfg["soft_floor_windows"]).lower()
        if cfg["soft_floor_windows"] not in SOFT_FLOOR_WINDOWS:
            raise ValueError(f"seasonal_trajectory.soft_floor_windows must be one of {SOFT_FLOOR_WINDOWS}, "
                             f"got {cfg['soft_floor_windows']!r}")
    elif penalty is not None:
        raise ValueError("seasonal_trajectory.shortfall_penalty_gbp_per_mwh_h2 is used only with end_condition 'soft_floor'")
    discount = cfg["surplus_discount_gbp_per_mwh_h2"]
    if discount is not None:
        if cfg["end_condition"] != "soft_floor":
            raise ValueError("seasonal_trajectory.surplus_discount_gbp_per_mwh_h2 is used only with end_condition 'soft_floor'")
        if not float(discount) >= 0.0:
            raise ValueError("seasonal_trajectory.surplus_discount_gbp_per_mwh_h2 must be >= 0")
        cfg["surplus_discount_gbp_per_mwh_h2"] = float(discount)
    if cfg["enabled"] and not cfg["file"]:
        raise ValueError("market.hydrogen_store.seasonal_trajectory.enabled needs a file (the annual pass output)")
    return cfg


def default_trajectory_path(build_source: str, resources_path: str = "resources") -> str:
    """Where rule ``solve_seasonal_storage_pass`` writes the trajectory of a parent network."""
    return f"{resources_path}/market/{build_source}_seasonal_trajectory.csv"


# ── Reading the trajectory (pass 2) ───────────────────────────────────────

_TRAJECTORY_CACHE: Dict[Tuple[str, float], pd.DataFrame] = {}


def load_seasonal_trajectory(path) -> pd.DataFrame:
    """The hourly trajectory CSV indexed by snapshot (cached per path and modification time).

    Args:
        path: CSV written by the annual pass.

    Returns:
        DataFrame with at least ``TRAJECTORY_COLUMNS``.
    """
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"Seasonal trajectory {p} not found; run rule solve_seasonal_storage_pass first")
    key = (str(p.resolve()), p.stat().st_mtime)
    if key not in _TRAJECTORY_CACHE:
        frame = pd.read_csv(p, index_col=0, parse_dates=True)
        missing = [c for c in TRAJECTORY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Seasonal trajectory {p} lacks columns {missing}")
        frame.index = pd.DatetimeIndex(frame.index)
        _TRAJECTORY_CACHE[key] = frame
    return _TRAJECTORY_CACHE[key]


def _row(trajectory: pd.DataFrame, snapshot, what: str) -> pd.Series:
    ts = pd.Timestamp(snapshot)
    if ts not in trajectory.index:
        raise ValueError(
            f"Seasonal trajectory has no row for {ts} ({what}); it covers "
            f"{trajectory.index[0]} to {trajectory.index[-1]} hourly")
    return trajectory.loc[ts]


def trajectory_start_level(trajectory: pd.DataFrame, snapshot) -> float:
    """Store level (MWh) just before ``snapshot`` (the level a period starting there inherits)."""
    return float(_row(trajectory, snapshot, "period start")["e_start_mwh"])


def trajectory_end_level(trajectory: pd.DataFrame, snapshot) -> float:
    """Store level (MWh) at the end of ``snapshot`` (PyPSA's ``e`` at that snapshot)."""
    return float(_row(trajectory, snapshot, "period end")["e_end_mwh"])


def trajectory_value(trajectory: pd.DataFrame, snapshot) -> float:
    """The annual pass's store value (GBP/MWh_H2) at ``snapshot``."""
    return float(_row(trajectory, snapshot, "window end value")["value_gbp_per_mwh_h2"])


def trajectory_storage_carriers(trajectory: pd.DataFrame) -> List[str]:
    """Storage-unit carriers the trajectory carries (``su_soc_end_mwh|<carrier>`` columns)."""
    return [c[len(SU_LEVEL_PREFIX):] for c in trajectory.columns if str(c).startswith(SU_LEVEL_PREFIX)]


def trajectory_storage_start(trajectory: pd.DataFrame, snapshot) -> pd.Series:
    """Per-carrier storage-unit state (MWh) just before ``snapshot``: the previous hour's end.

    The first hour of the trajectory takes its last hour's end (the pass is cyclic).
    """
    carriers = trajectory_storage_carriers(trajectory)
    if not carriers:
        raise ValueError("Seasonal trajectory has no storage-unit columns; rerun rule solve_seasonal_storage_pass")
    ts = pd.Timestamp(snapshot)
    _row(trajectory, ts, "storage start")
    pos = trajectory.index.get_loc(ts)
    row = trajectory.iloc[pos - 1] if pos > 0 else trajectory.iloc[-1]
    return pd.Series({c: float(row[SU_LEVEL_PREFIX + c]) for c in carriers})


def trajectory_storage_value(trajectory: pd.DataFrame, snapshot) -> pd.Series:
    """Per-carrier storage-unit value (GBP/MWh of state of charge) at ``snapshot``."""
    carriers = trajectory_storage_carriers(trajectory)
    row = _row(trajectory, snapshot, "storage end value")
    return pd.Series({c: float(row[SU_VALUE_PREFIX + c]) for c in carriers})


def store_shares(stores: pd.DataFrame, names) -> pd.Series:
    """Each carried store's share of their combined ``e_nom`` (the trajectory is one total)."""
    e_nom = pd.to_numeric(stores.loc[list(names), "e_nom"], errors="coerce").fillna(0.0)
    total = float(e_nom.sum())
    if total <= 0:
        return pd.Series(1.0 / max(len(e_nom), 1), index=e_nom.index)
    return e_nom / total


# ── Aggregation (pass 1) ──────────────────────────────────────────────────

def _bus_class(network, external_pattern: str = EXTERNAL_BUS_PATTERN) -> pd.Series:
    """``GB_mainland`` for every GB AC bus (mainland and islands), else the bus itself."""
    buses = network.buses
    carrier = buses["carrier"].astype(str) if "carrier" in buses.columns else pd.Series("AC", index=buses.index)
    gb_ac = carrier.isin(["AC", ""]) & ~buses.index.to_series().str.contains(external_pattern, regex=False)
    return pd.Series(np.where(gb_ac, MAINLAND_HUB, buses.index.astype(str)), index=buses.index)


def _fmt(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return str(v)
    return f"{v:.6g}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@#+-]+", "_", str(name))


def _group_names(keys: pd.Series, prefix: str) -> Dict[str, str]:
    """A unique, readable component name per group key."""
    out, used = {}, set()
    for key in pd.unique(keys):
        parts = str(key).split("|")
        base = _safe(f"{prefix}{parts[1]}@{parts[0]}") if len(parts) > 1 else _safe(f"{prefix}{key}")
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}#{k}"
        used.add(name)
        out[key] = name
    return out


def _weighted_pu(dense: pd.DataFrame, p_nom: pd.Series, keys: pd.Series, p_nom_g: pd.Series) -> pd.DataFrame:
    """Capacity-weighted per-unit series per group: sum(pu_i p_nom_i) / sum(p_nom_i)."""
    mw = dense.mul(p_nom.reindex(dense.columns), axis=1)
    summed = mw.T.groupby(keys.reindex(dense.columns)).sum().T
    denom = p_nom_g.reindex(summed.columns).replace(0.0, np.nan)
    return summed.div(denom, axis=1).fillna(0.0)


def _max_member_deviation(dense: pd.DataFrame, grouped: pd.DataFrame, keys: pd.Series) -> pd.Series:
    """Largest |member pu - group pu| over time, per group (0 for one-member groups)."""
    expanded = grouped[keys.reindex(dense.columns).to_numpy()]
    expanded.columns = dense.columns
    dev = (dense - expanded).abs().max(axis=0)
    return dev.groupby(keys.reindex(dense.columns)).max()


def _static_attrs(agg, list_name: str, frame: pd.DataFrame) -> Tuple[dict, List[str]]:
    """Split a static frame into standard PyPSA attributes (for ``add``) and extra columns."""
    standard = set(getattr(agg, list_name).columns)
    std = {c: frame[c] for c in frame.columns if c in standard and not c.endswith("_opt")
           and c not in ("sub_network", "control", "type")}
    extra = [c for c in frame.columns if c not in standard]
    return std, extra


def _copy_components(agg, network, component: str, list_name: str, names, bus_map: Dict[str, pd.Series],
                     cyclic: Optional[dict] = None) -> None:
    """Copy components unchanged (except buses) with their extra columns and time series."""
    names = list(names)
    if not names:
        return
    frame = getattr(network, list_name).loc[names].copy()
    for col, mapped in bus_map.items():
        frame[col] = mapped.reindex(names).to_numpy()
    for col, value in (cyclic or {}).items():
        frame[col] = value
    std, extra = _static_attrs(agg, list_name, frame)
    agg.add(component, names, **std)
    static = getattr(agg, list_name)
    for col in extra:
        static.loc[names, col] = frame[col].to_numpy()
    series = getattr(network, f"{list_name}_t")
    target = getattr(agg, f"{list_name}_t")
    for attr in list(series.keys()):
        df = series[attr]
        if df is None or df.empty:
            continue
        cols = [c for c in names if c in df.columns]
        if cols:
            existing = target[attr]
            target[attr] = pd.concat([existing, df[cols]], axis=1) if not existing.empty else df[cols].copy()


def aggregate_for_seasonal_pass(network, scenario_config: dict, pass_cfg: dict, logger: logging.Logger,
                                energy_floor: Optional[pd.DataFrame] = None,
                                external_pattern: str = EXTERNAL_BUS_PATTERN):
    """Build the aggregated copperplate network for the annual pass (see the module note).

    Call on the full-period network after the Stage 1 pre-solve preparations
    (``prepare_full_network``). The input network is not modified.

    Args:
        network: the prepared full network.
        scenario_config: the parent scenario's merged config (hydro block used).
        pass_cfg: ``resolve_seasonal_pass_config`` output.
        logger: logger.
        energy_floor: ``prepare_min_output_energy_budget`` output on ``network`` (MW per unit).
        external_pattern: substring that marks foreign buses.

    Returns:
        ``(aggregated network, report dict, aggregated energy floor)``.
    """
    sns = network.snapshots
    cls = _bus_class(network, external_pattern)
    agg = pypsa.Network()
    agg.set_snapshots(sns)
    agg.snapshot_weightings = network.snapshot_weightings.copy()
    report: Dict[str, dict] = {}

    # Carriers and buses.
    carriers = network.carriers
    if len(carriers):
        keep = [c for c in ("co2_emissions", "nice_name", "color") if c in carriers.columns]
        agg.add("Carrier", carriers.index, **{c: carriers[c] for c in keep})
    main = network.buses.loc[cls.index[cls == MAINLAND_HUB]]
    hub_kwargs = {"v_nom": 400.0, "carrier": "AC"}
    for coord in ("x", "y"):
        if coord in main.columns and len(main):
            hub_kwargs[coord] = float(pd.to_numeric(main[coord], errors="coerce").mean())
    agg.add("Bus", MAINLAND_HUB, **hub_kwargs)
    others = network.buses.loc[cls.index[cls != MAINLAND_HUB]]
    if len(others):
        cols = [c for c in ("v_nom", "carrier", "x", "y") if c in others.columns]
        agg.add("Bus", others.index, **{c: others[c] for c in cols})
        if "country" in others.columns:
            agg.buses.loc[others.index, "country"] = others["country"].to_numpy()
    agg.buses.loc[MAINLAND_HUB, "country"] = "GB"
    report["buses"] = {"original": int(len(network.buses)), "gb_ac_collapsed": int((cls == MAINLAND_HUB).sum()),
                       "aggregated": int(len(agg.buses))}

    # ── Generators ──
    g = network.generators.copy()
    g["bus_class"] = g["bus"].map(cls).fillna(g["bus"])
    p_nom = pd.to_numeric(g["p_nom"], errors="coerce").fillna(0.0)
    mc = pd.to_numeric(g["marginal_cost"], errors="coerce").fillna(0.0)
    band = float(pass_cfg.get("cost_band_gbp_per_mwh", 0.0))
    mc_key = (mc / band).round() * band if band > 0 else mc.round(6)
    tv_mc = set(getattr(network.generators_t, "marginal_cost", pd.DataFrame()).columns)
    keys = g["bus_class"].astype(str) + "|" + g["carrier"].astype(str) + "|" + mc_key.map(_fmt)
    single = g.index.isin(tv_mc)
    keys[single] = keys[single] + "|" + g.index[single].astype(str)
    names = _group_names(keys, "")
    gkey = keys.map(names)
    p_nom_g = p_nom.groupby(gkey).sum()
    pmax = network.get_switchable_as_dense("Generator", "p_max_pu", sns)
    pmin = network.get_switchable_as_dense("Generator", "p_min_pu", sns)
    pmax_g = _weighted_pu(pmax, p_nom, gkey, p_nom_g)
    pmin_g = _weighted_pu(pmin, p_nom, gkey, p_nom_g)
    gdev = _max_member_deviation(pmax, pmax_g, gkey)
    del pmax, pmin
    first = g.groupby(gkey).first()
    mc_g = (mc * p_nom).groupby(gkey).sum() / p_nom_g.replace(0.0, np.nan)
    mc_g = mc_g.fillna(mc.groupby(gkey).mean())
    ramp = {}
    for col in ("ramp_limit_up", "ramp_limit_down"):
        if col in g.columns:
            r = pd.to_numeric(g[col], errors="coerce")
            has = r.notna().groupby(gkey).any()
            val = (r.fillna(1.0) * p_nom).groupby(gkey).sum() / p_nom_g.replace(0.0, np.nan)
            ramp[col] = val.where(has, np.nan)

    hydro_cfg = (((scenario_config or {}).get("renewables", {}) or {}).get("hydro", {})) or {}
    large_cfg = hydro_cfg.get("large_hydro", {}) or {}
    to_storage = (bool(pass_cfg.get("large_hydro_as_storage", True)) and bool(hydro_cfg.get("enabled", False))
                  and bool(large_cfg.get("reservoir_model", False)))
    hydro_groups = [k for k in p_nom_g.index if to_storage and str(first.at[k, "carrier"]) in LARGE_HYDRO_CARRIERS]
    gen_groups = [k for k in p_nom_g.index if k not in hydro_groups]

    agg.add("Generator", gen_groups,
            bus=first.loc[gen_groups, "bus_class"], carrier=first.loc[gen_groups, "carrier"],
            p_nom=p_nom_g.loc[gen_groups], marginal_cost=mc_g.loc[gen_groups],
            p_max_pu=pmax_g[gen_groups], p_min_pu=pmin_g[gen_groups],
            **{col: ramp[col].loc[gen_groups] for col in ramp})
    if tv_mc:
        tv_groups = [names[k] for k in keys[single]]
        tv = network.generators_t.marginal_cost[list(g.index[single])]
        tv.columns = tv_groups
        agg.generators_t.marginal_cost = tv
    members = gkey.groupby(gkey).size()
    by_carrier = g.groupby("carrier").agg(units=("p_nom", "size"), p_nom_mw=("p_nom", "sum"))
    by_carrier["groups"] = first.groupby("carrier").size().reindex(by_carrier.index).fillna(0).astype(int)
    report["generators"] = {
        "original": int(len(g)), "aggregated": int(len(gen_groups)), "large_hydro_to_storage": len(hydro_groups),
        "by_carrier": {c: {"units": int(r.units), "groups": int(r.groups), "p_nom_mw": round(float(r.p_nom_mw), 1)}
                       for c, r in by_carrier.iterrows()},
        "p_max_pu_member_deviation_max": round(float(gdev[members[members > 1].index].max()), 6)
        if (members > 1).any() else 0.0,
        "p_nom_mw": round(float(p_nom.sum()), 3), "p_nom_mw_aggregated": round(float(p_nom_g.sum()), 3),
    }

    # ── Energy floor (minimum output as an energy budget) ──
    agg_floor = pd.DataFrame(index=sns)
    if energy_floor is not None and energy_floor.shape[1]:
        fkeys = gkey.reindex(energy_floor.columns)
        agg_floor = energy_floor.T.groupby(fkeys).sum().T
        hours = energy_floor.attrs.get("hours", {})
        agg_floor.attrs["hours"] = {
            grp: float(pd.Series([hours.get(u, 24.0) for u in fkeys.index[fkeys == grp]]).max())
            for grp in agg_floor.columns
        }

    # ── Storage units (and large hydro as a reservoir) ──
    su = network.storage_units.copy()
    cyclic_su = bool(pass_cfg.get("storage_cyclic", True))
    if len(su):
        su["bus_class"] = su["bus"].map(cls).fillna(su["bus"])
        su_p_nom = pd.to_numeric(su["p_nom"], errors="coerce").fillna(0.0)
        attr_cols = ["efficiency_store", "efficiency_dispatch", "standing_loss", "marginal_cost",
                     "marginal_cost_storage", "spill_cost", "max_hours"]
        attr_cols = [c for c in attr_cols if c in su.columns]
        skeys = su["bus_class"].astype(str) + "|" + su["carrier"].astype(str)
        for col in attr_cols:
            skeys = skeys + "|" + pd.to_numeric(su[col], errors="coerce").fillna(0.0).round(6).map(_fmt)
        tv_su_mc = set(getattr(network.storage_units_t, "marginal_cost", pd.DataFrame()).columns)
        s_single = su.index.isin(tv_su_mc)
        skeys[s_single] = skeys[s_single] + "|" + su.index[s_single].astype(str)
        snames = _group_names(skeys, "")
        skey = skeys.map(snames)
        su_p_nom_g = su_p_nom.groupby(skey).sum()
        sfirst = su.groupby(skey).first()
        spmax = network.get_switchable_as_dense("StorageUnit", "p_max_pu", sns)
        spmin = network.get_switchable_as_dense("StorageUnit", "p_min_pu", sns)
        inflow = network.get_switchable_as_dense("StorageUnit", "inflow", sns)
        spmax_g = _weighted_pu(spmax, su_p_nom, skey, su_p_nom_g)
        spmin_g = _weighted_pu(spmin, su_p_nom, skey, su_p_nom_g)
        inflow_g = inflow.T.groupby(skey.reindex(inflow.columns)).sum().T
        inflow_pu = inflow.div(su_p_nom.replace(0.0, np.nan), axis=1).fillna(0.0)
        inflow_pu_g = inflow_g.div(su_p_nom_g.replace(0.0, np.nan), axis=1).fillna(0.0)
        dev = pd.DataFrame({
            "p_max_pu": _max_member_deviation(spmax, spmax_g, skey),
            "p_min_pu": _max_member_deviation(spmin, spmin_g, skey),
            "inflow_pu": _max_member_deviation(inflow_pu, inflow_pu_g, skey),
        })
        del spmax, spmin, inflow, inflow_pu
        soc0 = pd.to_numeric(su["state_of_charge_initial"], errors="coerce").fillna(0.0).groupby(skey).sum()
        groups = list(su_p_nom_g.index)
        kwargs = {col: sfirst.loc[groups, col] for col in attr_cols}
        agg.add("StorageUnit", groups, bus=sfirst.loc[groups, "bus_class"], carrier=sfirst.loc[groups, "carrier"],
                p_nom=su_p_nom_g.loc[groups], p_max_pu=spmax_g[groups], p_min_pu=spmin_g[groups],
                state_of_charge_initial=soc0.loc[groups], cyclic_state_of_charge=cyclic_su, **kwargs)
        nonzero = [c for c in groups if c in inflow_g.columns and float(inflow_g[c].abs().sum()) > 0]
        if nonzero:
            agg.storage_units_t.inflow = inflow_g[nonzero]
        if s_single.any():
            tv = network.storage_units_t.marginal_cost[list(su.index[s_single])]
            tv.columns = [snames[k] for k in skeys[s_single]]
            agg.storage_units_t.marginal_cost = tv
        smembers = skey.groupby(skey).size()
        sc = su.groupby("carrier").agg(units=("p_nom", "size"), p_nom_mw=("p_nom", "sum"))
        energy = (su_p_nom * pd.to_numeric(su["max_hours"], errors="coerce").fillna(0.0)).groupby(su["carrier"]).sum()
        report["storage_units"] = {
            "original": int(len(su)), "aggregated": int(len(groups)), "cyclic": cyclic_su,
            "by_carrier": {c: {"units": int(r.units), "groups": int(sfirst["carrier"].eq(c).sum()),
                               "p_nom_mw": round(float(r.p_nom_mw), 1), "energy_mwh": round(float(energy[c]), 1)}
                           for c, r in sc.iterrows()},
            "max_member_deviation": {
                str(sfirst.at[k, "carrier"]): {a: round(float(dev.at[k, a]), 6) for a in dev.columns}
                for k in smembers.index if smembers[k] > 1},
        }
    if hydro_groups:
        inflow_factor = float(large_cfg.get("inflow_profile_factor", 0.55))
        res_hours = float(large_cfg.get("reservoir_capacity_hours", 168.0))
        init_hours = float(large_cfg.get("initial_storage_hours", 48.0))
        hp = p_nom_g.loc[hydro_groups]
        agg.add("StorageUnit", hydro_groups, bus=first.loc[hydro_groups, "bus_class"],
                carrier=first.loc[hydro_groups, "carrier"], p_nom=hp, max_hours=res_hours,
                p_max_pu=pmax_g[hydro_groups], p_min_pu=0.0, efficiency_store=1.0, efficiency_dispatch=1.0,
                marginal_cost=mc_g.loc[hydro_groups], spill_cost=0.0,
                state_of_charge_initial=hp * min(init_hours, res_hours), cyclic_state_of_charge=cyclic_su)
        # Per-unit inflow summed by group: p_max_pu x inflow_profile_factor, or the
        # unit's own inflow series when its ceiling is its availability (OW3,
        # renewables.hydro.large_hydro.power_ceiling; scripts/solve/hydro_constraints.py).
        # Networks without the separate inflow series (every HT50 build) keep the
        # original expression exactly, so the 2050 annual pass is unchanged bit for bit.
        from scripts.solve.hydro_constraints import LARGE_HYDRO_INFLOW_SERIES, large_hydro_inflow_power

        hydro_units = list(g.index[gkey.isin(hydro_groups)])
        series = getattr(network.generators_t, LARGE_HYDRO_INFLOW_SERIES, None)
        if series is None or series.empty or not set(hydro_units) & set(series.columns):
            hydro_inflow = pmax_g[hydro_groups].mul(hp, axis=1) * inflow_factor
        else:
            unit_inflow = large_hydro_inflow_power(network, sns, hydro_cfg, hydro_units)
            hydro_inflow = (unit_inflow.T.groupby(gkey.reindex(unit_inflow.columns)).sum().T
                            .reindex(columns=hydro_groups).fillna(0.0))
        agg.storage_units_t.inflow = pd.concat([agg.storage_units_t.inflow, hydro_inflow], axis=1) \
            if not agg.storage_units_t.inflow.empty else hydro_inflow
        report["large_hydro"] = {"groups": len(hydro_groups), "p_nom_mw": round(float(hp.sum()), 1),
                                 "reservoir_hours": res_hours, "inflow_profile_factor": inflow_factor,
                                 "inflow_twh": round(float(hydro_inflow.sum().sum() / 1e6), 4)}

    # ── Links ──
    lk = network.links.copy()
    if len(lk):
        b0 = lk["bus0"].map(cls).fillna(lk["bus0"])
        b1 = lk["bus1"].map(cls).fillna(lk["bus1"])
        external = (lk["bus0"].astype(str).str.contains(external_pattern, regex=False)
                    | lk["bus1"].astype(str).str.contains(external_pattern, regex=False))
        internal = (b0 == MAINLAND_HUB) & (b1 == MAINLAND_HUB) & ~external
        tv_cols = set()
        for attr in network.links_t.keys():
            df = network.links_t[attr]
            if df is not None and not df.empty:
                tv_cols |= set(df.columns)
        individual = (external | lk.index.isin(tv_cols)) & ~internal
        pooled = ~individual & ~internal
        report_links = {"original": int(len(lk)), "dropped_gb_internal": int(internal.sum()),
                        "dropped_gb_internal_mw": round(float(lk.loc[internal, "p_nom"].sum()), 1),
                        "kept_individual": int(individual.sum())}
        _copy_components(agg, network, "Link", "links", lk.index[individual],
                         {"bus0": b0, "bus1": b1})
        if pooled.any():
            lp = lk.loc[pooled]
            lkeys = (b0[pooled].astype(str) + "|" + lp["carrier"].astype(str) + "|" + b1[pooled].astype(str))
            for col in ("efficiency", "marginal_cost", "p_min_pu", "p_max_pu"):
                lkeys = lkeys + "|" + pd.to_numeric(lp[col], errors="coerce").fillna(0.0).round(6).map(_fmt)
            lnames = _group_names(lkeys, "")
            lkey = lkeys.map(lnames)
            lfirst = lp.groupby(lkey).first()
            lp_nom = pd.to_numeric(lp["p_nom"], errors="coerce").fillna(0.0).groupby(lkey).sum()
            groups = list(lp_nom.index)
            agg.add("Link", groups, bus0=b0[pooled].groupby(lkey).first().loc[groups],
                    bus1=b1[pooled].groupby(lkey).first().loc[groups], carrier=lfirst.loc[groups, "carrier"],
                    p_nom=lp_nom.loc[groups], efficiency=lfirst.loc[groups, "efficiency"],
                    marginal_cost=lfirst.loc[groups, "marginal_cost"], p_min_pu=lfirst.loc[groups, "p_min_pu"],
                    p_max_pu=lfirst.loc[groups, "p_max_pu"])
            report_links["pooled"] = {str(lfirst.at[k, "carrier"]): {"units": int((lkey == k).sum()),
                                                                     "p_nom_mw": round(float(lp_nom[k]), 1)}
                                      for k in groups}
        report_links["aggregated"] = int(len(agg.links))
        report["links"] = report_links

    # ── Stores ──
    if len(network.stores):
        cyclic_store = bool(pass_cfg.get("store_cyclic", True))
        _copy_components(agg, network, "Store", "stores", network.stores.index,
                         {"bus": network.stores["bus"].map(cls).fillna(network.stores["bus"])},
                         cyclic={"e_cyclic": cyclic_store})
        report["stores"] = {"names": list(network.stores.index), "e_cyclic": cyclic_store,
                            "e_nom_mwh": round(float(network.stores["e_nom"].sum()), 1)}

    # ── Loads ──
    loads = network.loads
    if len(loads):
        lcls = loads["bus"].map(cls).fillna(loads["bus"])
        dkeys = lcls.astype(str) + "|" + loads["carrier"].astype(str)
        dnames = _group_names(dkeys, "load_")
        dkey = dkeys.map(dnames)
        p_set = network.get_switchable_as_dense("Load", "p_set", sns)
        p_set_g = p_set.T.groupby(dkey.reindex(p_set.columns)).sum().T
        dfirst = pd.DataFrame({"bus": lcls, "carrier": loads["carrier"]}).groupby(dkey).first()
        groups = list(p_set_g.columns)
        agg.add("Load", groups, bus=dfirst.loc[groups, "bus"], carrier=dfirst.loc[groups, "carrier"],
                p_set=p_set_g[groups])
        report["loads"] = {"original": int(len(loads)), "aggregated": len(groups),
                           "energy_twh": round(float(p_set.mul(network.snapshot_weightings.generators, axis=0)
                                                     .sum().sum() / 1e6), 4),
                           "energy_twh_aggregated": round(float(p_set_g.mul(network.snapshot_weightings.generators,
                                                                            axis=0).sum().sum() / 1e6), 4)}
        del p_set

    k = int(pass_cfg.get("resolution_hours", 1))
    if k > 1:
        agg, agg_floor = coarsen_network(agg, k, agg_floor)
        report["resolution_hours"] = k
    logger.info(
        "Seasonal pass aggregation: %d buses, %d generators (from %d), %d storage units (from %d), %d links "
        "(from %d), %d stores, %d loads (from %d); %d snapshots at %d h",
        len(agg.buses), len(agg.generators), len(network.generators), len(agg.storage_units),
        len(network.storage_units), len(agg.links), len(network.links), len(agg.stores), len(agg.loads),
        len(network.loads), len(agg.snapshots), k)
    return agg, report, agg_floor


def coarsen_network(agg, hours: int, floor: Optional[pd.DataFrame] = None):
    """Average every time series over blocks of ``hours`` snapshots (declared approximation).

    Snapshot weights become the block length, so energies are preserved.
    """
    sns = agg.snapshots
    block = pd.Series(np.arange(len(sns)) // hours, index=sns)
    new_index = pd.DatetimeIndex([sns[i] for i in range(0, len(sns), hours)])
    weights = block.groupby(block).size().astype(float).to_numpy()
    frames = {}
    for list_name in ("generators", "storage_units", "links", "loads", "stores"):
        series = getattr(agg, f"{list_name}_t")
        for attr in list(series.keys()):
            df = series[attr]
            if df is None or df.empty:
                continue
            coarse = df.groupby(block.reindex(df.index).to_numpy()).mean()
            coarse.index = new_index
            frames[(list_name, attr)] = coarse
    agg.set_snapshots(new_index)
    for col in agg.snapshot_weightings.columns:
        agg.snapshot_weightings[col] = weights
    for (list_name, attr), coarse in frames.items():
        getattr(agg, f"{list_name}_t")[attr] = coarse
    if floor is not None and floor.shape[1]:
        hours_attr = floor.attrs.get("hours", {})
        floor = floor.groupby(block.reindex(floor.index).to_numpy()).mean()
        floor.index = new_index
        floor.attrs["hours"] = hours_attr
    return agg, floor


# ── Pre-solve preparation (the Stage 1 steps, on the full network) ─────────

def prepare_full_network(network, scenario_config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Apply the Stage 1 pre-solve steps the annual pass must share; return the energy floor.

    Mirrors ``solve_wholesale.py`` for a copperplate rolling Stage 1: cost validation,
    the marginal-cost basis, the storage dispatch cost floor, numerical conditioning,
    load-shedding caps, feasible operating bounds, unit commitment off, the
    minimum-output energy budget. Not applied, declared: the solve period (the pass is
    the whole year), forecast error (perfect foresight by design), a contract-aware
    curtailment order and scarcity pricing (both off for HT50; a warning names them if on).

    Args:
        network: the full parent network (modified in place).
        scenario_config: its merged config.
        logger: logger.

    Returns:
        The minimum-output energy floor (MW per unit; empty when none is configured).
    """
    from scripts.market.marginal_cost_basis import apply_marginal_cost_basis, needs_strike_table
    from scripts.market.market_utils import (
        apply_initial_storage_soc, apply_storage_dispatch_cost_floor, prepare_min_output_energy_budget,
        resolve_store_level_config,
    )
    from scripts.solve.solve_network import (
        apply_load_shedding_limits, enforce_feasible_generator_operating_bounds,
        improve_numerical_conditioning, validate_network_costs,
    )

    market = (scenario_config or {}).get("market", {}) or {}
    wholesale = market.get("wholesale", {}) or {}
    validate_network_costs(network, logger)
    basis_block = wholesale.get("marginal_cost_basis", {})
    basis_strikes = None
    if needs_strike_table(basis_block):
        from scripts.market.cfd_bm_bids import load_scenario_strike_table

        basis_strikes = load_scenario_strike_table(scenario_config)
    apply_marginal_cost_basis(network, basis_block, logger, strike_table=basis_strikes)
    for key, label in (("renewable_curtailment_order", "contract-aware curtailment order"),):
        mode = str((wholesale.get(key, {}) or {}).get("mode", "as_built"))
        if mode != "as_built":
            logger.warning(f"Seasonal pass: {label} ({mode}) is not applied in the annual pass (declared)")
    if bool((wholesale.get("scarcity_pricing", {}) or {}).get("enabled", False)):
        logger.warning("Seasonal pass: scarcity pricing is not applied in the annual pass (declared)")
    if bool((wholesale.get("forecast_error", {}) or {}).get("enabled", False)):
        logger.warning("Seasonal pass: forecast error is not applied; the annual pass is perfect foresight")
    if str(wholesale.get("copperplate_mode", "collapse")).lower() == "none":
        logger.warning("Seasonal pass: the scenario's Stage 1 is a central dispatch; the annual pass is a copperplate")
    if not bool(wholesale.get("relax_island_hvdc", True)):
        logger.warning("Seasonal pass: relax_island_hvdc is off; the pass still merges islands into one GB bus")
    apply_storage_dispatch_cost_floor(network, wholesale, logger)
    improve_numerical_conditioning(network, logger)
    apply_load_shedding_limits(network, logger)
    enforce_feasible_generator_operating_bounds(network, logger)
    if "committable" in network.generators.columns:
        network.generators["committable"] = False
    if (scenario_config.get("optimization", {}) or {}).get("remove_must_run", False):
        network.generators["p_min_pu"] = 0.0
    # Starting levels for a non-cyclic pass (market.seasonal_pass.storage_cyclic /
    # store_cyclic false): the scenario's own storage_soc and hydrogen_store fractions.
    apply_initial_storage_soc(network, market, logger)
    store_cfg = resolve_store_level_config({"hydrogen_store": {
        k: v for k, v in (market.get("hydrogen_store") or {}).items() if k not in ("seasonal_trajectory", "month_end")}})
    names = carried_store_names_for_pass(network, market)
    if names and store_cfg["initial_fraction"] is not None:
        e_nom = pd.to_numeric(network.stores.loc[names, "e_nom"], errors="coerce").fillna(0.0)
        network.stores.loc[names, "e_initial"] = store_cfg["initial_fraction"] * e_nom
    return prepare_min_output_energy_budget(network, scenario_config, logger)


# ── The annual LP ────────────────────────────────────────────────────────

def carried_store_names_for_pass(network, market_config: dict) -> List[str]:
    """The hydrogen stores the trajectory is for (``hydrogen_store.carriers``, e_nom > 0)."""
    carriers = tuple(str(c) for c in (((market_config or {}).get("hydrogen_store", {}) or {}).get("carriers")
                                      or H2_STORE_CARRIERS_DEFAULT))
    stores = network.stores
    if not len(stores):
        return []
    e_nom = pd.to_numeric(stores["e_nom"], errors="coerce").fillna(0.0)
    return list(stores.index[stores["carrier"].isin(set(carriers)) & (e_nom > 0)])


def solve_annual_pass(agg, scenario_config: dict, solver_name: str, solver_options: dict,
                      logger: logging.Logger, energy_floor: Optional[pd.DataFrame] = None,
                      optimize_kwargs: Optional[dict] = None):
    """Solve the aggregated year with the Stage 1 callbacks; return the solver outcome.

    Callbacks: interconnector shared capacity and piecewise bid/ask with tranches,
    the pumped-hydro wholesale SoC floor, the minimum-output energy budget. The
    counterflow lock and the fallback ladder are the Stage 1 ones.
    """
    from scripts.interconnectors.market_coupling import (
        build_interconnector_shared_capacity_callback, build_piecewise_interconnector_cost_callback,
        lock_material_counterflows,
    )
    from scripts.market.market_utils import build_min_output_energy_callback, solve_with_counterflow_lock
    from scripts.solve.hydro_constraints import build_hydro_constraints_callback, combine_extra_functionalities

    pumped = ((((scenario_config or {}).get("renewables", {}) or {}).get("hydro", {})) or {}).get(
        "pumped_hydro", {}) or {}
    callbacks = combine_extra_functionalities(
        build_interconnector_shared_capacity_callback(agg, logger),
        build_hydro_constraints_callback(agg, scenario_config,
                                         pumped_min_soc_override=pumped.get("wholesale_min_soc_fraction"),
                                         logger=logger),
        build_piecewise_interconnector_cost_callback(agg, logger),
        build_min_output_energy_callback(energy_floor),
    )
    return solve_with_counterflow_lock(
        agg, solver_name, solver_options, callbacks, logger, "Seasonal annual pass",
        lock_counterflows=lock_material_counterflows, **(optimize_kwargs or {}),
    )


def store_dual(agg, names) -> pd.DataFrame:
    """Dual of each store's energy balance (GBP/MWh), snapshots x stores; empty if unavailable."""
    model = getattr(agg, "model", None)
    try:
        dual = model.constraints["Store-energy_balance"].dual.to_pandas()
    except Exception:  # pragma: no cover - no model or no duals
        return pd.DataFrame(index=agg.snapshots)
    if isinstance(dual, pd.Series):
        dual = dual.to_frame()
    dual.index = pd.DatetimeIndex(dual.index)
    return dual.reindex(columns=[n for n in names if n in dual.columns])


def extract_trajectory(agg, names: List[str], hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """The hourly trajectory table (one total over ``names``) from the solved aggregated year.

    ``e_end_mwh`` is PyPSA's ``e`` (level after the snapshot), ``e_start_mwh`` the level
    before it (the previous ``e``; the first takes the cyclic end or ``e_initial``).
    At a coarser resolution, ``e_end`` is interpolated linearly to the hour and the value
    held over each block.
    """
    e = agg.stores_t.e.reindex(columns=names).fillna(0.0).sum(axis=1)
    e_nom = float(pd.to_numeric(agg.stores.loc[names, "e_nom"], errors="coerce").fillna(0.0).sum())
    dual = store_dual(agg, names)
    shares = store_shares(agg.stores, names)
    value = dual.mul(shares.reindex(dual.columns).fillna(0.0), axis=1).sum(axis=1) if dual.shape[1] else \
        pd.Series(np.nan, index=agg.snapshots)
    prices = agg.buses_t.marginal_price
    h2_bus = agg.stores.loc[names[0], "bus"] if names else None
    sns = agg.snapshots
    frame = pd.DataFrame(index=sns)
    frame["e_end_mwh"] = e
    frame["value_gbp_per_mwh_h2"] = value.reindex(sns)
    if h2_bus is not None and h2_bus in prices.columns:
        frame["h2_bus_price_gbp_per_mwh_h2"] = prices[h2_bus]
    if MAINLAND_HUB in prices.columns:
        frame["gb_price_gbp_per_mwh"] = prices[MAINLAND_HUB]
    gp = agg.generators_t.p
    shed = agg.generators.index[agg.generators["carrier"].isin(LOAD_SHEDDING_CARRIERS)]
    frame["load_shedding_mw"] = gp.reindex(columns=shed).fillna(0.0).sum(axis=1)
    p0 = agg.links_t.p0
    for carrier, col, side in (("electrolysis", "electrolysis_mw_el", "p0"), ("H2_turbine", "h2_turbine_mw_h2", "p0"),
                               ("gas_reforming_ccs", "reformer_mw_h2", "p1")):
        links = agg.links.index[agg.links["carrier"] == carrier]
        src = p0 if side == "p0" else -agg.links_t.p1
        frame[col] = src.reindex(columns=links).fillna(0.0).sum(axis=1)
    su_frame = storage_unit_trajectory(agg)
    if su_frame.shape[1]:
        frame = pd.concat([frame, su_frame.reindex(sns)], axis=1)
    if len(sns) != len(hourly_index) or not sns.equals(hourly_index):
        frame = _to_hourly(frame, hourly_index)
    initial = float(pd.to_numeric(agg.stores.loc[names, "e_initial"], errors="coerce").fillna(0.0).sum())
    cyclic = bool(agg.stores.loc[names, "e_cyclic"].astype(bool).all()) if names else False
    start = frame["e_end_mwh"].shift(1)
    start.iloc[0] = float(frame["e_end_mwh"].iloc[-1]) if cyclic else initial
    frame.insert(0, "e_start_mwh", start)
    frame.insert(2, "e_nom_mwh", e_nom)
    frame.index.name = "snapshot"
    return frame


def storage_unit_dual(agg) -> pd.DataFrame:
    """Dual of each storage unit's energy balance (GBP/MWh of state of charge); empty if unavailable.

    PyPSA writes the storage-unit and store balances the same way (``-level +
    previous level ... = rhs``), so the sign convention is the store's: positive is
    the value of energy held.
    """
    model = getattr(agg, "model", None)
    try:
        dual = model.constraints["StorageUnit-energy_balance"].dual.to_pandas()
    except Exception:  # pragma: no cover - no model or no duals
        return pd.DataFrame(index=agg.snapshots)
    if isinstance(dual, pd.Series):
        dual = dual.to_frame()
    dual.index = pd.DatetimeIndex(dual.index)
    return dual


def storage_unit_trajectory(agg) -> pd.DataFrame:
    """Per-carrier storage-unit state (``su_soc_end_mwh|c``) and value (``su_value|c``) from the solved pass.

    The value is the energy-capacity-weighted mean of the carrier's groups' duals
    (one group per carrier on FES25_HT_2050). Carriers in
    ``SU_TRAJECTORY_EXCLUDED_CARRIERS`` are left out.
    """
    su = agg.storage_units
    soc = getattr(agg.storage_units_t, "state_of_charge", pd.DataFrame())
    if not len(su) or soc is None or soc.empty:
        return pd.DataFrame(index=agg.snapshots)
    keep = su.index[~su["carrier"].isin(SU_TRAJECTORY_EXCLUDED_CARRIERS)]
    dual = storage_unit_dual(agg)
    energy = (pd.to_numeric(su["p_nom"], errors="coerce").fillna(0.0)
              * pd.to_numeric(su["max_hours"], errors="coerce").fillna(0.0))
    cols = {}
    for carrier in sorted(su.loc[keep, "carrier"].unique()):
        units = [u for u in keep[su.loc[keep, "carrier"] == carrier] if u in soc.columns]
        if not units:
            continue
        cols[SU_LEVEL_PREFIX + carrier] = soc[units].sum(axis=1)
        present = [u for u in units if u in dual.columns]
        if present:
            w = energy.reindex(present)
            w = w / w.sum() if float(w.sum()) > 0 else pd.Series(1.0 / len(present), index=present)
            cols[SU_VALUE_PREFIX + carrier] = dual[present].mul(w, axis=1).sum(axis=1)
        else:
            cols[SU_VALUE_PREFIX + carrier] = pd.Series(np.nan, index=agg.snapshots)
    return pd.DataFrame(cols, index=agg.snapshots)


def _to_hourly(frame: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Coarse blocks to hours: levels interpolated between block ends, rates and values held."""
    sns = frame.index
    step = sns[1] - sns[0] if len(sns) > 1 else pd.Timedelta(hours=1)
    block_end = sns + step - pd.Timedelta(hours=1)
    out = frame.reindex(hourly_index, method="ffill")
    for col in ["e_end_mwh"] + [c for c in frame.columns if str(c).startswith(SU_LEVEL_PREFIX)]:
        levels = pd.Series(frame[col].to_numpy(), index=block_end).reindex(hourly_index)
        out[col] = levels.interpolate(limit_direction="both")
    return out


# ── Diagnostics ───────────────────────────────────────────────────────────

def annual_summary(agg, trajectory: pd.DataFrame, names: List[str]) -> dict:
    """Annual totals, monthly store and value statistics and load shedding by day."""
    w = agg.snapshot_weightings.generators
    g = agg.generators
    gp = agg.generators_t.p.reindex(columns=g.index).fillna(0.0)
    pmax = agg.get_switchable_as_dense("Generator", "p_max_pu")
    avail = pmax.mul(g["p_nom"], axis=1).mul(w, axis=0).sum()
    disp = gp.mul(w, axis=0).sum()
    twh = lambda s: round(float(s) / 1e6, 3)  # noqa: E731
    out: Dict[str, object] = {}
    vre = g.index[g["carrier"].isin(VRE_CARRIERS)]
    out["vre_twh"] = {str(c): {"available": twh(avail[vre][g.loc[vre, "carrier"] == c].sum()),
                               "dispatched": twh(disp[vre][g.loc[vre, "carrier"] == c].sum())}
                      for c in sorted(g.loc[vre, "carrier"].unique())}
    vre_av = float(avail[vre].sum())
    vre_d = float(disp[vre].sum())
    out["vre_total_twh"] = {"available": twh(vre_av), "dispatched": twh(vre_d), "curtailed": twh(vre_av - vre_d),
                            "curtailed_share": round(1 - vre_d / vre_av, 4) if vre_av else None}
    su = agg.storage_units
    if len(su) and "large_hydro" in set(su["carrier"]):
        hyd = su.index[su["carrier"] == "large_hydro"]
        out["large_hydro_twh"] = twh(agg.storage_units_t.p_dispatch.reindex(columns=hyd).fillna(0.0)
                                     .mul(w, axis=0).sum().sum())
    out["generation_twh_by_carrier"] = {str(c): twh(v) for c, v in disp.groupby(g["carrier"]).sum().items()}
    loads = agg.loads
    p_set = agg.get_switchable_as_dense("Load", "p_set")
    out["load_twh_by_carrier"] = {str(c): twh(v) for c, v in p_set.mul(w, axis=0).sum().groupby(loads["carrier"]).sum().items()}
    inflow = agg.storage_units_t.inflow
    if len(su) and not inflow.empty:
        # Consumer-flexibility demand sits in storage-unit drains (negative inflow), not loads.
        drain = (-inflow.clip(upper=0.0)).mul(w, axis=0).sum()
        out["storage_drain_twh_by_carrier"] = {str(c): twh(v) for c, v in
                                               drain.groupby(su["carrier"].reindex(drain.index)).sum().items() if v > 0}
    gb_loads = loads.index[loads["bus"] == MAINLAND_HUB]
    out["gb_load_twh"] = twh(p_set[gb_loads].mul(w, axis=0).sum().sum())
    out["gb_peak_load_gw"] = round(float(p_set[gb_loads].sum(axis=1).max() / 1e3), 2)
    shed = g.index[g["carrier"].isin(LOAD_SHEDDING_CARRIERS)]
    shed_t = gp[shed].sum(axis=1)
    out["load_shedding"] = {"twh": twh(shed_t.mul(w).sum()), "max_gw": round(float(shed_t.max() / 1e3), 3),
                            "hours": int((shed_t > 1.0).sum())}
    daily = shed_t.mul(w).groupby(shed_t.index.normalize()).sum() / 1e3
    out["load_shedding"]["days_gwh"] = {str(d.date()): round(float(v), 2) for d, v in daily[daily > 0.01].items()}
    # Links: hydrogen and interconnectors.
    lk = agg.links
    p0 = agg.links_t.p0.reindex(columns=lk.index).fillna(0.0)
    p1 = agg.links_t.p1.reindex(columns=lk.index).fillna(0.0)
    sum_w = lambda df: float(df.mul(w, axis=0).sum().sum())  # noqa: E731
    ely = lk.index[lk["carrier"] == "electrolysis"]
    tur = lk.index[lk["carrier"] == "H2_turbine"]
    ref = lk.index[lk["carrier"] == "gas_reforming_ccs"]
    h2_load = loads.index[loads["carrier"] == "H2_demand"]
    out["hydrogen_twh"] = {
        "electrolysis_electricity": twh(sum_w(p0[ely])), "electrolysis_h2": twh(-sum_w(p1[ely])),
        "reformer_h2": twh(-sum_w(p1[ref])), "turbine_h2": twh(sum_w(p0[tur])),
        "turbine_electricity": twh(-sum_w(p1[tur])), "non_power_demand": twh(p_set[h2_load].mul(w, axis=0).sum().sum()),
    }
    external = lk.index[lk["bus0"].astype(str).str.contains(EXTERNAL_BUS_PATTERN, regex=False)
                        | lk["bus1"].astype(str).str.contains(EXTERNAL_BUS_PATTERN, regex=False)]
    gb_side = pd.DataFrame(index=agg.snapshots)
    for name in external:
        gb_side[name] = -p1[name] if lk.at[name, "bus1"] == MAINLAND_HUB else -p0[name]
    imports = gb_side.clip(lower=0.0)
    exports = (-gb_side).clip(lower=0.0)
    out["interconnectors_twh"] = {"imports": twh(sum_w(imports)), "exports": twh(sum_w(exports)),
                                  "net_imports": twh(sum_w(gb_side))}
    if "gb_price_gbp_per_mwh" in trajectory.columns:
        price = trajectory["gb_price_gbp_per_mwh"]
        out["gb_price"] = {"mean": round(float(price.mean()), 2), "min": round(float(price.min()), 2),
                           "max": round(float(price.max()), 2),
                           "monthly_mean": {str(k): round(float(v), 2) for k, v in price.groupby(price.index.month).mean().items()}}
    level = trajectory["e_end_mwh"]
    value = trajectory["value_gbp_per_mwh_h2"]
    months = level.index.month
    out["store_twh"] = {
        "e_nom": round(float(trajectory["e_nom_mwh"].iloc[0]) / 1e6, 3),
        "start": round(float(trajectory["e_start_mwh"].iloc[0]) / 1e6, 3), "end": round(float(level.iloc[-1]) / 1e6, 3),
        "min": round(float(level.min()) / 1e6, 3), "min_at": str(level.idxmin()),
        "max": round(float(level.max()) / 1e6, 3), "max_at": str(level.idxmax()),
        "monthly": {str(m): {"min": round(float(level[months == m].min()) / 1e6, 3),
                             "max": round(float(level[months == m].max()) / 1e6, 3),
                             "end": round(float(level[months == m].iloc[-1]) / 1e6, 3)} for m in sorted(set(months))},
    }
    out["store_value_gbp_per_mwh_h2"] = {
        "mean": round(float(value.mean()), 2), "min": round(float(value.min()), 2), "max": round(float(value.max()), 2),
        "monthly": {str(m): {"mean": round(float(value[months == m].mean()), 2),
                             "min": round(float(value[months == m].min()), 2),
                             "max": round(float(value[months == m].max()), 2)} for m in sorted(set(months))},
    }
    su_carriers = trajectory_storage_carriers(trajectory)
    if su_carriers:
        first = trajectory.groupby(months).head(1)
        out["storage_units_month_start_gwh"] = {
            c: {str(ts.month): round(float(trajectory_storage_start(trajectory, ts)[c]) / 1e3, 1) for ts in first.index}
            for c in su_carriers}
        out["storage_units_value_gbp_per_mwh"] = {
            c: {"mean": round(float(trajectory[SU_VALUE_PREFIX + c].mean()), 2),
                "min": round(float(trajectory[SU_VALUE_PREFIX + c].min()), 2),
                "max": round(float(trajectory[SU_VALUE_PREFIX + c].max()), 2)} for c in su_carriers}
    if "h2_bus_price_gbp_per_mwh_h2" in trajectory.columns:
        gap = (trajectory["h2_bus_price_gbp_per_mwh_h2"] - value).abs()
        out["store_value_vs_h2_bus_price_abs_gap"] = {"mean": round(float(gap.mean()), 4), "max": round(float(gap.max()), 4)}
    return out


def write_outputs(trajectory: pd.DataFrame, summary: dict, trajectory_path, summary_path) -> None:
    """Write the trajectory CSV and the JSON summary (with provenance)."""
    Path(str(trajectory_path)).parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(trajectory_path)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def run_annual_pass(network, scenario_config: dict, solver_name: str, solver_options: dict,
                    logger: logging.Logger, optimize_kwargs: Optional[dict] = None,
                    provenance: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """Prepare, aggregate and solve the annual pass; return ``(hourly trajectory, summary)``.

    Args:
        network: the full parent network (consumed: modified, then released), or a
            one-element list holding it so the caller keeps no reference.
        scenario_config: its merged config.
        solver_name: solver.
        solver_options: solver options (``threads`` is replaced by the pass setting).
        logger: logger.
        optimize_kwargs: extra ``optimize`` kwargs (``io_api``).
        provenance: extra provenance fields for the summary.
    """
    if isinstance(network, list):
        network = network.pop()   # the caller's holder: this frame keeps the only reference
    market = (scenario_config or {}).get("market", {}) or {}
    pass_cfg = resolve_seasonal_pass_config(market)
    hourly_index = pd.DatetimeIndex(network.snapshots)
    started = time.perf_counter()
    floor = prepare_full_network(network, scenario_config, logger)
    names = carried_store_names_for_pass(network, market)
    if not names:
        raise ValueError("Seasonal pass: the network has no hydrogen store (market.hydrogen_store.carriers)")
    agg, report, agg_floor = aggregate_for_seasonal_pass(network, scenario_config, pass_cfg, logger,
                                                         energy_floor=floor)
    report["hydrogen_reserve"] = apply_hydrogen_reserve(agg, names, pass_cfg["hydrogen_reserve_twh"], logger)
    # Release the full network (the caller passes it without keeping a reference).
    del network, floor
    import gc

    gc.collect()
    prep_s = time.perf_counter() - started
    options = dict(solver_options or {})
    for key in list(options):
        if key.lower() == "threads":
            options.pop(key)
    options["threads"] = pass_cfg["threads"]
    t0 = time.perf_counter()
    outcome = solve_annual_pass(agg, scenario_config, solver_name, options, logger, energy_floor=agg_floor,
                                optimize_kwargs=optimize_kwargs)
    solve_s = time.perf_counter() - t0
    model = getattr(agg, "model", None)
    size = {}
    if model is not None:
        try:
            size = {"variables": int(model.nvars), "constraints": int(model.ncons)}
        except Exception:  # pragma: no cover
            size = {}
    trajectory = extract_trajectory(agg, names, hourly_index)
    summary = {
        "provenance": {"source": "rule solve_seasonal_storage_pass (scripts/market/seasonal_storage.py)",
                       "status": "model output (annual copperplate pass, plan decision 33, task Y8a)",
                       "stores": names, **(provenance or {})},
        "pass_config": pass_cfg,
        "solver": {"name": solver_name, "status": outcome.status, "condition": outcome.condition,
                   "rung": outcome.rung, "rung_name": outcome.rung_name,
                   "optimize_seconds": round(outcome.elapsed, 1), "attempts": [list(a) for a in outcome.attempts],
                   "bound_violation_mw": outcome.bound_violation},
        "runtime_seconds": {"prepare_and_aggregate": round(prep_s, 1), "solve_total": round(solve_s, 1)},
        "lp_size": size, "snapshots": int(len(agg.snapshots)),
        "objective_gbp": float(getattr(agg, "objective", float("nan"))),
        "aggregation": report,
        "annual": annual_summary(agg, trajectory, names),
    }
    try:
        import psutil

        mem = psutil.Process().memory_info()
        summary["rss_gb_at_end"] = round(mem.rss / 1e9, 2)
        summary["peak_working_set_gb"] = round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)
    except Exception:  # pragma: no cover
        pass
    logger.info(
        "Seasonal pass solved: %s (%s) on rung %d in %.0f s; store %0.2f to %0.2f TWh (min %0.2f at %s); "
        "value %0.1f-%0.1f GBP/MWh_H2 (mean %0.1f); load shed %.3f TWh",
        outcome.status, outcome.condition, outcome.rung, solve_s,
        summary["annual"]["store_twh"]["start"], summary["annual"]["store_twh"]["end"],
        summary["annual"]["store_twh"]["min"], summary["annual"]["store_twh"]["min_at"],
        summary["annual"]["store_value_gbp_per_mwh_h2"]["min"], summary["annual"]["store_value_gbp_per_mwh_h2"]["max"],
        summary["annual"]["store_value_gbp_per_mwh_h2"]["mean"], summary["annual"]["load_shedding"]["twh"])
    return trajectory, summary
