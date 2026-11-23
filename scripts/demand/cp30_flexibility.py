"""CP30 2030 consumer demand flexibility (follow-up F5; user decision 27 Oct 2026: central).

CP30's 2030 demand (``scripts/demand/cp30_demand.py``) is a component shape calibrated to the
ES1 peak and was inflexible: only electrolysers and storage moved. CP30 ED1 (FFR 2030, GW at
the 5-6 pm peak) books consumer DSR 4.72 (residential 2.69, I&C 2.03), EV smart charging 4.46,
V2G 0.94 at peak and 1.87 maximum, and heat flexibility 6.14 (storage heaters 4.12, thermal
storage and district heating 1.51 of which thermal storage 0.43, hybrid heat pumps 0.08).

**One method with the FES long-range chain.** The flexibility is built on the 2050 chain's
conventions (``scripts/demand/fes_demand.py``, ``scripts/demand/demand_carriers.py``): the same
carriers, ``StorageUnit`` components on the AC buses of the demand they shift, a fixed drain
(``inflow < 0``) met by charging when Stage 1 chooses, units starting empty (``state_of_charge_
initial = 0``, not cyclic; rolling market solves carry the state of charge), and the same
parameters (``fes_demand.DEFAULT_CONFIG`` blocks; store_days 1.0 as the 2050 study). Stage 1
dispatches them; the balancing mechanism holds them at their Stage 1 schedule (participation
``exclude_carriers`` and 6000 GBP/MWh overrides, as in the 2050 study).

**No double counting: carve, then add.** The CP30 shape is kept as built (its ES1 peak
calibration stands). For each flexible part, the consumption the shape already put on the
network (its *reference*) is removed from the base loads at the buses where that component
sits, and a storage unit takes the same electricity through its drain:

* EV smart charging and V2G: the smart-charged share of residential EV energy, which the CP30
  build placed on a fixed overnight window (``cp30_demand.shape.ev_smart_window``), becomes the
  flexible pool; drain = the fleet's driving profile, plug-in availability from V2G at peak /
  maximum (2050 rule). The V2G fleet is part of that pool, on the same buses.
* Heat pumps and district heat: the share of their temperature shape whose load at the
  ACS-peak hours is ED1's flexibility at peak (2050 rule; capped at the whole load).
* Storage heaters (``SH_flexibility``; FES 2025 FLX1 has them too, the 2050 chain does not):
  resistive space heat whose direct-acting load at the ACS-peak hours is ED1's 4.12 GW. They
  already charge overnight in the ESPENI 2013 base, so the reference carved from the base is an
  Economy 7 window holding each day's energy; the drain is the day's space heat; the unit may
  charge at any hour up to the fleet's charging power (coldest day / ``charge_hours``) and holds
  ``store_days`` of the coldest day.
* Demand response (I&C and residential, user decision 27 Oct 2026): energy neutral, nothing carved.

**Spatial.** Each component's GSP energy goes to buses through the base loads' own
distribution (``load.aggregate_demand_to_network_topology`` with the build's ETYS options),
then the carve is checked bus by bus: a bus whose load would fall below zero in any hour keeps
part of its flexible energy as fixed load (logged; zero on the 2030 build).

Library functions are pure and testable on synthetic inputs; the network step is called from
``scripts/demand/load.py`` (build_base_demand, CP30 branch) when
``demand.build.flexibility.enabled`` is true. Code default: off.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from scripts.demand import demand_carriers as dc
from scripts.demand import fes_demand as f50

logger = logging.getLogger(__name__)


def _fes_block(key: str) -> dict:
    return copy.deepcopy(f50.DEFAULT_CONFIG[key])


#: Declared defaults (``demand.build.flexibility``). The EV, V2G, heat-pump, district-heat and DSR
#: blocks are the 2050 chain's own (``fes_demand.DEFAULT_CONFIG``), with the 2050 study's
#: store_days 1.0; the storage-heater block is new. Off unless a scenario sets ``enabled: true``.
DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    # ED1 pathway for rows a pathway lacks (CP30 New Dispatch publishes DSR only).
    "ed1_fallback_pathway": "Further Flex and Renewables",
    # FES building block giving the battery-EV fleet (cars, vans, motorbikes), read from the
    # same FES file and pathway as the CP30 GSP shares (FES 2024 Holistic Transition).
    "ev_fleet_block": "Lct_BB001",
    "electric_vehicles": {**{k: v for k, v in _fes_block("electric_vehicles").items()
                             if k in ("driving_weekday", "driving_weekend", "driving_weekend_factor",
                                      "charger_kw", "battery_kwh", "usable_battery_fraction",
                                      "charge_efficiency", "plugged_in_max", "store_days")},
                          "store_days": 1.0},
    "v2g": _fes_block("v2g"),
    "heat_pump_flexibility": _fes_block("heat_pump_flexibility"),
    "district_heat_flexibility": _fes_block("district_heat_flexibility"),
    # FES 2024's ED1/FLX1 publish "thermal storage and district heating flex" and "thermal
    # storage flex"; FES 2025 splits district heating out. "combined_less_thermal_storage" reads
    # the combined row as including thermal storage (FES 2025's 2030 split, 0.85 + 0.84 GW, is
    # closer to 0.43 + 1.07 than to 0.43 + 1.51); "combined" takes it whole. Capped at the DH load.
    "district_heat_flex_basis": "combined_less_thermal_storage",
    # The 2050 chain's storage-heater block (charge hours 7, one-day store, no standing loss), plus:
    "storage_heater_flexibility": {
        **_fes_block("storage_heater_flexibility"), "enabled": True,
        # Where the base holds their charge today: an Economy 7 night block of about seven hours,
        # 00:00-07:00 (declared; three circular Gaussians: weight, mean hour, sd hours).
        "reference_window": [[1.0, 1.0, 1.5], [1.0, 3.5, 1.5], [1.0, 6.0, 1.5]],
        # Location: FES baseline domestic demand (no FES storage-heater block exists).
        "allocation_blocks": ["Dem_BB008a"],
    },
    "ic_dsr": {**_fes_block("ic_dsr"), "allocation_blocks": ["Srg_BB006a"]},
    # Residential DSR on in both years (user decision, 27 Oct 2026); the 2050 chain's parameters.
    "residential_dsr": {**_fes_block("residential_dsr"), "enabled": True, "allocation_blocks": ["Dem_BB008a"]},
    # Holding cost of the drained units, GBP/MWh per hour held (the 2050 chain's; see fes_demand).
    "drained_holding_cost_gbp_per_mwh_h": f50.DEFAULT_CONFIG["drained_holding_cost_gbp_per_mwh_h"],
    # A bus share below this is dropped (shares renormalised), as in the 2050 allocation.
    "min_share": 1e-6,
}

#: CP30 ED1 rows (``resources/CP30/CP30_ED1.csv``, the CP30 data workbook's ED1 sheet; FES 2024
#: vocabulary). GW, negative in the workbook for peak reductions; read positive.
ED1_FLEX_ITEMS = {
    "smart_charging_at_peak": ["Smart charging impact at peak"],
    "v2g_at_peak": ["V2G potential at peak"],
    "v2g_max": ["V2G maximum potential"],
    "residential_dsr": ["Residential DSR impact at peak"],
    "ic_dsr": ["I&C DSR impact at peak"],
    "storage_heater": ["Residential peak reduction - storage heater flex",
                       "Non-residential peak reduction - storage heater flex"],
    "hybrid_hp": ["Residential peak reduction - hybrid heat pump flex",
                  "Non-residential peak reduction - hybrid heat pump flex"],
    "thermal_storage": ["Residential peak reduction - thermal storage flex",
                        "Non-residential peak reduction - thermal storage flex"],
    "thermal_storage_and_dh": ["Residential peak reduction - thermal storage and district heating flex",
                               "Non-residential peak reduction - thermal storage and district heating flex"],
}

#: Carriers whose energy is carved out of the CP30 loads (demand response is energy neutral).
CARVED_CARRIERS = (dc.EV_SMART_CHARGING, dc.V2G, dc.HEAT_PUMP_FLEXIBILITY, dc.DISTRICT_HEAT_FLEXIBILITY,
                   dc.STORAGE_HEATER_FLEXIBILITY)

META_KEY = "cp30_flexibility"


def resolve_config(cfg: Optional[dict]) -> dict:
    """``demand.build.flexibility`` merged over :data:`DEFAULT_CONFIG` (nested dicts merged)."""
    def _merge(base: dict, over: dict) -> dict:
        out = copy.deepcopy(base)
        for k, v in (over or {}).items():
            out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        return out
    return _merge(DEFAULT_CONFIG, cfg or {})


def is_enabled(cfg: Optional[dict]) -> bool:
    """True only when the block says ``enabled: true``."""
    return bool((cfg or {}).get("enabled", False))


# ─────────────────────────────────── ED1 figures ───────────────────────────────────

def _norm(value) -> str:
    return " ".join(str(value).strip().lower().split())


def ed1_flexibility(ed1: pd.DataFrame, pathway: str, year: int, fallback_pathway: str) -> Dict[str, Dict[str, Any]]:
    """CP30 ED1 flexibility figures (GW, positive) for a pathway and year, with their rows.

    Each item sums its rows; an item the pathway lacks (CP30 New Dispatch publishes DSR and the
    peak only) is read from ``fallback_pathway``. Returns item -> {gw, pathway, rows}.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for key, items in ED1_FLEX_ITEMS.items():
        names = {_norm(i) for i in items}
        for source in (pathway, fallback_pathway):
            rows = ed1[(ed1["Pathway"].map(_norm) == _norm(source)) & ed1["Data item"].map(_norm).isin(names)]
            if len(rows) == len(items):
                break
        else:
            raise ValueError(f"CP30 ED1 has no rows {items} for {pathway} or {fallback_pathway}")
        values = pd.to_numeric(rows[str(year)], errors="coerce")
        if values.isna().any():
            raise ValueError(f"CP30 ED1 {items} {source} {year}: missing values")
        out[key] = {"gw": float(values.abs().sum()), "pathway": source, "rows": list(items)}
    return out


def district_heat_flex_gw(flex: Dict[str, Dict[str, Any]], basis: str) -> float:
    """District-heating flexibility at peak (GW) from the FES 2024 rows (see DEFAULT_CONFIG)."""
    combined = flex["thermal_storage_and_dh"]["gw"]
    if basis == "combined":
        return combined
    if basis == "combined_less_thermal_storage":
        return max(combined - flex["thermal_storage"]["gw"], 0.0)
    raise ValueError("district_heat_flex_basis must be 'combined' or 'combined_less_thermal_storage'")


# ──────────────────────────────── national build ────────────────────────────────

def _unit(shape: np.ndarray, dt: float) -> np.ndarray:
    shape = np.asarray(shape, dtype=float)
    return shape / (shape.sum() * dt)


def daily_window(index: pd.DatetimeIndex, daily_mwh: np.ndarray, window) -> np.ndarray:
    """MW that puts each day's energy on a within-day window (circular Gaussian mixture).

    Args:
        index: Snapshots.
        daily_mwh: Energy per snapshot's day, repeated on every snapshot of the day.
        window: (weight, mean hour, sd) mixture.
    """
    idx = pd.DatetimeIndex(index)
    dt = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    w = pd.Series(f50.gaussian_day(hours, window), index=idx)
    day_sum = w.groupby(idx.normalize()).transform("sum").to_numpy() * dt
    return np.asarray(daily_mwh, dtype=float) * w.to_numpy() / day_sum


def build_flexibility(snapshots: pd.DatetimeIndex, temp_c: pd.Series, prep: Dict[str, Any], demand_cfg: dict,
                      flex_cfg: dict, ed1_flex: Dict[str, Dict[str, Any]], ev_fleet: float) -> Dict[str, Any]:
    """National flexibility specs, their profiles and the carve references (MW, system basis).

    Args:
        snapshots: Model snapshots.
        temp_c: Demand-weighted temperature on ``snapshots`` (``cp30_demand.weather_temperature``).
        prep: ``cp30_demand.prepare`` output (components, energies, lines, loss factor, EV shares).
        demand_cfg: Resolved ``cp30_demand`` config (its shape block gives the EV smart window).
        flex_cfg: Resolved :data:`DEFAULT_CONFIG`.
        ed1_flex: :func:`ed1_flexibility`.
        ev_fleet: Battery EVs (``ev_fleet_block``).

    Returns:
        ``flex`` (carrier -> spec, as ``fes_demand.build_components``'s, with ``reference``: the
        profile column carved from the loads, or None), ``profiles`` (national MW and per-unit
        columns) and ``calibration``.
    """
    from scripts.demand import cp30_demand

    idx = pd.DatetimeIndex(snapshots)
    dt = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    lf = float(prep["accounting"]["loss_factor"])
    comp_gwh = prep["component_gwh"]
    sh = cp30_demand.national_shapes(idx, temp_c, demand_cfg, prep["lines"],
                                     comp_gwh.get("electric_vehicles", 0.0) / lf, lf,
                                     prep["ev_residential_share"], prep["ev_depot_share"])
    mask = sh["mask"]
    if mask is None:
        raise ValueError("CP30 flexibility needs the weather-year temperature")
    pcfg = f50.resolve_config(demand_cfg["shape"].get("profiles") or {})
    prof = pd.DataFrame(index=idx)
    flex: Dict[str, Dict[str, Any]] = {}
    cal: Dict[str, Any] = {"calibration_hours": int(mask.sum()), "loss_factor": lf,
                           "ed1": {k: v["gw"] for k, v in ed1_flex.items()}}

    def _at_peak(mw) -> float:
        return float(np.asarray(mw)[mask].mean())

    # ── EV smart charging and V2G: the smart-charged share the CP30 shape put overnight ──
    e = flex_cfg["electric_vehicles"]
    s = float(sh["record"].get("ev_smart_share_of_residential", 0.0))
    ev_mwh = float(comp_gwh.get("electric_vehicles", 0.0)) * 1e3
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    smart_unit = _unit(f50.gaussian_day(hours, demand_cfg["shape"]["ev_smart_window"]), dt)
    ev_flex_mwh = ev_mwh * float(prep["ev_residential_share"]) * s
    reference_ev = ev_flex_mwh * smart_unit
    cal.update(ev_smart_share_of_residential=s, ev_flex_gwh=ev_flex_mwh / 1e3)
    if ev_flex_mwh > 0:
        eta_c = float(e["charge_efficiency"])
        drive = f50.ev_profiles(idx, {**pcfg, "electric_vehicles": {**pcfg["electric_vehicles"], **e}})["driving"]
        drive = drive.to_numpy()
        drive_cal = float(drive[mask].mean())
        v2g_max = ed1_flex["v2g_max"]["gw"]
        ratio = ed1_flex["v2g_at_peak"]["gw"] / v2g_max if v2g_max > 0 else 1.0
        avail = float(e["plugged_in_max"]) * np.clip(1.0 - (1.0 - ratio) * drive / drive_cal, 0.02, 1.0)
        prof["ev_plugged_in"] = avail
        v2g_cfg = flex_cfg["v2g"]
        flex_vehicles = s * float(ev_fleet)
        v2g_vehicles = 0.0
        if v2g_cfg.get("enabled", True) and v2g_max > 0 and flex_vehicles > 0:
            v2g_vehicles = min(flex_vehicles, v2g_max * 1e6 / float(v2g_cfg["bidirectional_charger_kw"]))
        v2g_frac = v2g_vehicles / flex_vehicles if flex_vehicles > 0 else 0.0
        usable_mwh = float(e["battery_kwh"]) * float(e["usable_battery_fraction"]) / 1e3
        for carrier, frac, vehicles in ((dc.EV_SMART_CHARGING, 1.0 - v2g_frac, flex_vehicles - v2g_vehicles),
                                        (dc.V2G, v2g_frac, v2g_vehicles)):
            mwh = ev_flex_mwh * frac
            if mwh <= 0 or vehicles <= 0:
                continue
            drain = _unit(drive, dt) * mwh * eta_c
            charge_mw = vehicles * float(e["charger_kw"]) / 1e3
            energy_mwh = vehicles * usable_mwh
            if e.get("store_days") is not None:
                energy_mwh = min(energy_mwh, float(e["store_days"]) * float(drain.mean()) * 24.0)
            spec = {"energy_mwh": energy_mwh, "efficiency_store": eta_c, "efficiency_dispatch": 1.0,
                    "standing_loss": 0.0, "marginal_cost": 0.0, "inflow": f"{carrier}_drain",
                    "reference": f"{carrier}_reference", "allocation": "electric_vehicles",
                    "electricity_gwh": mwh / 1e3, "vehicles": vehicles, "charge_mw": charge_mw,
                    "source": ("CP30 smart-charged EV share (cp30_demand: FES 2024 'EVs at peak'); fleet "
                               f"{flex_cfg['ev_fleet_block']}" if carrier == dc.EV_SMART_CHARGING else
                               "CP30 ED1 'V2G maximum potential' and 'V2G potential at peak'")}
            if carrier == dc.V2G:
                discharge_mw = v2g_max * 1e3
                p_nom = max(charge_mw, discharge_mw)
                spec.update(p_nom_mw=p_nom, efficiency_dispatch=float(v2g_cfg["discharge_efficiency"]),
                            marginal_cost=float(v2g_cfg["degradation_cost_gbp_per_mwh"]),
                            p_min_pu="V2G_fleet_p_min_pu", p_max_pu="V2G_fleet_p_max_pu", discharge_mw=discharge_mw)
                prof["V2G_fleet_p_min_pu"] = -avail * charge_mw / p_nom
                prof["V2G_fleet_p_max_pu"] = (avail / float(e["plugged_in_max"])) * discharge_mw / p_nom
            else:
                spec.update(p_nom_mw=charge_mw, p_min_pu="EV_smart_charging_p_min_pu", p_max_pu=0.0)
                prof["EV_smart_charging_p_min_pu"] = -avail
            flex[carrier] = spec
            prof[f"{carrier}_drain"] = -drain
            prof[f"{carrier}_reference"] = reference_ev * frac
        cal.update(ev_flexible_vehicles=flex_vehicles, v2g_vehicles=v2g_vehicles,
                   ev_smart_impact_at_peak_mw=float(sh["record"].get("ev_smart_impact_at_peak_mw", np.nan)) * lf,
                   ev_smart_impact_ed1_mw=ed1_flex["smart_charging_at_peak"]["gw"] * lf * 1e3)

    # ── Heat pumps and district heat: the share at peak that ED1 calls flexible ──
    shapes = sh["shapes"]
    comps = prep["components"]
    for carrier, comp, block, flex_gw, source in (
            (dc.HEAT_PUMP_FLEXIBILITY, "heat_pumps", "heat_pump_flexibility",
             ed1_flex["hybrid_hp"]["gw"] + ed1_flex["thermal_storage"]["gw"],
             "CP30 ED1 hybrid heat pump + thermal storage flex at peak (residential + non-residential)"),
            (dc.DISTRICT_HEAT_FLEXIBILITY, "district_heat", "district_heat_flexibility",
             district_heat_flex_gw(ed1_flex, flex_cfg["district_heat_flex_basis"]),
             f"CP30 ED1 thermal storage and district heating flex ({flex_cfg['district_heat_flex_basis']})")):
        hcfg = flex_cfg[block]
        if not hcfg.get("enabled", True) or comp not in shapes.columns or comp not in comps.columns:
            continue
        mw = shapes[comp].to_numpy() * float(comps[comp].sum()) * 1e3
        share = float(np.clip(flex_gw * lf * 1e3 / max(_at_peak(mw), 1e-9), 0.0, 1.0))
        cal[f"{comp}_flex_share"] = share
        cal[f"{comp}_flex_target_at_peak_mw"] = flex_gw * lf * 1e3
        cal[f"{comp}_at_peak_mw"] = _at_peak(mw)
        if share <= 0:
            continue
        drain = share * mw
        flex[carrier] = {"p_nom_mw": float(hcfg["compressor_headroom"] * drain.max()),
                         "energy_mwh": float(hcfg["storage_hours_at_peak"] * _at_peak(drain)),
                         "efficiency_store": 1.0, "efficiency_dispatch": 1.0,
                         "standing_loss": float(hcfg["standing_loss_per_hour"]), "marginal_cost": 0.0,
                         "p_min_pu": -1.0, "p_max_pu": 0.0, "inflow": f"{carrier}_drain",
                         "reference": f"{carrier}_reference", "allocation": comp,
                         "electricity_gwh": float(drain.sum() * dt / 1e3), "source": source}
        prof[f"{carrier}_drain"] = -drain
        prof[f"{carrier}_reference"] = drain

    # ── Storage heaters: resistive space heat, charged on a night window in the base ──
    scfg = flex_cfg["storage_heater_flexibility"]
    sh_gw = ed1_flex["storage_heater"]["gw"]
    if scfg.get("enabled", True) and sh_gw > 0:
        # The 2050 chain's unit (fes_demand.storage_heater_flexibility); 2030 adds the night reference.
        unit = f50.storage_heater_flexibility(temp_c, mask, sh_gw * lf * 1e3, scfg, pcfg)
        drain, daily = unit["drain"], unit["daily_mwh"]
        reference = daily_window(idx, daily, scfg["reference_window"])
        coldest_day = float(daily.max())
        flex[dc.STORAGE_HEATER_FLEXIBILITY] = {
            **unit["spec"], "reference": f"{dc.STORAGE_HEATER_FLEXIBILITY}_reference",
            "source": ("CP30 ED1 storage heater flex at peak (residential + non-residential) = direct-acting load at "
                       "the ACS-peak hours; energy derived from the space-heat shape; base reference Economy 7 window"),
        }
        prof[f"{dc.STORAGE_HEATER_FLEXIBILITY}_drain"] = -drain
        prof[f"{dc.STORAGE_HEATER_FLEXIBILITY}_reference"] = reference
        if "p_min_pu" in unit:   # Economy 10-style charge windows (storage_heater_flexibility.charge_windows)
            prof[unit["spec"]["p_min_pu"]] = unit["p_min_pu"]
        cal.update(storage_heater_gwh=float(drain.sum() * dt / 1e3), storage_heater_coldest_day_mwh=coldest_day,
                   storage_heater_direct_at_peak_mw=_at_peak(drain))

    # ── Demand response: energy-neutral shifting (2050 form) ──
    for key, carrier, ed1_key in (("ic_dsr", dc.IC_DSR, "ic_dsr"), ("residential_dsr", dc.RESIDENTIAL_DSR,
                                                                     "residential_dsr")):
        d = flex_cfg[key]
        gw = ed1_flex[ed1_key]["gw"]
        if not d.get("enabled", False) or gw <= 0:
            continue
        p_nom = gw * 1e3
        flex[carrier] = {"p_nom_mw": p_nom, "energy_mwh": p_nom * float(d["shift_hours"]),
                         "efficiency_store": 1.0, "efficiency_dispatch": 1.0, "standing_loss": 0.0,
                         "marginal_cost": float(d["marginal_cost_gbp_per_mwh"]), "p_min_pu": -1.0, "p_max_pu": 1.0,
                         "inflow": None, "reference": None, "allocation": key, "electricity_gwh": 0.0,
                         "source": f"CP30 ED1 {ED1_FLEX_ITEMS[ed1_key][0]} ({ed1_flex[ed1_key]['pathway']})"}
    f50.holding_costs(flex, flex_cfg)   # DSR and drained-unit holding costs (marginal_cost_storage), as 2050
    return {"flex": flex, "profiles": prof, "calibration": cal}


# ─────────────────────────────── spatial allocation ───────────────────────────────

def allocation_vectors(prep: Dict[str, Any], fes: pd.DataFrame, demand_cfg: dict, flex_cfg: dict,
                       year: int) -> Dict[str, pd.Series]:
    """GSP vectors (any unit) of every allocation key the specs use.

    EVs, heat pumps and district heat are the CP30 build's own component energies by GSP (so a
    unit sits where the demand it shifts was put); storage heaters, DSR on their FES blocks.
    """
    from scripts.demand.cp30_demand import fes_gsp_blocks

    sp = demand_cfg["spatial"]
    gsp_index = prep["gsp_gwh"].index
    out: Dict[str, pd.Series] = {}
    for comp in ("electric_vehicles", "heat_pumps", "district_heat"):
        if comp in prep["components"].columns:
            out[comp] = prep["components"][comp].astype(float)
    for key, block in (("storage_heaters", "storage_heater_flexibility"), ("ic_dsr", "ic_dsr"),
                       ("residential_dsr", "residential_dsr")):
        values = fes_gsp_blocks(fes, sp["FES_scenario"], year, flex_cfg[block]["allocation_blocks"],
                                sp.get("direct_rows")).abs()
        out[key] = values.reindex(gsp_index).fillna(0.0)
    return {k: v[v > 0] for k, v in out.items()}


def etys_bus_allocator(network, fes_year: int, etys_options: Optional[dict], log: Optional[logging.Logger] = None
                       ) -> Callable[[pd.Series], pd.Series]:
    """GSP vector -> bus shares through the base loads' ETYS distribution (with its options).

    The same path ``load.add_FES_demand_data_to_network`` takes (Dem_per_node, unlisted-GSP
    resolution, the capped-demand step), on a flat day so only the energy split matters.
    """
    from scripts.demand.load import aggregate_demand_to_network_topology

    log = log or logger
    opts = {k: v for k, v in (etys_options or {}).items() if k != "direct_node_shares"}
    index = pd.date_range("2030-01-01", periods=24, freq="h")

    def allocate(gsp_values: pd.Series) -> pd.Series:
        gwh = gsp_values.astype(float)
        ts = pd.DataFrame(np.outer(np.ones(len(index)), gwh.to_numpy() * 1e3 / 8760.0), index=index,
                          columns=gwh.index)
        bus_demand, _ = aggregate_demand_to_network_topology(gwh.to_frame("demand"), ts, "ETYS", network, None, log,
                                                             is_historical=False, fes_year=fes_year,
                                                             etys_options=opts)
        values = bus_demand.iloc[:, 0].astype(float)
        return values / values.sum()

    return allocate


def bus_shares(vectors: Dict[str, pd.Series], allocate: Callable[[pd.Series], pd.Series],
               min_share: float = 0.0) -> pd.DataFrame:
    """Bus x allocation-key shares (columns sum to 1)."""
    table = pd.DataFrame({k: allocate(v) for k, v in vectors.items()}).fillna(0.0).clip(lower=0.0)
    table = table.div(table.sum(axis=0), axis=1)
    if min_share > 0:
        table = table.where(table >= min_share, 0.0)
        table = table.div(table.sum(axis=0), axis=1)
    table.index = table.index.astype(str)
    table.index.name = "bus"
    return table


# ──────────────────────────────── network step ────────────────────────────────

def _dt(n) -> pd.Series:
    return n.snapshot_weightings.generators.astype(float)


def integrate_flexibility(n, built: Dict[str, Any], shares: pd.DataFrame,
                          log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Carve the flexible references from the base loads and add the flexibility units.

    Args:
        n: Network with the CP30 loads (one per bus, base carriers), modified in place.
        built: :func:`build_flexibility` output; profiles on ``n.snapshots``.
        shares: :func:`bus_shares`.
        log: Logger.

    Returns:
        Stats: carved GWh by carrier, energy kept as fixed load where a bus could not give it,
        units added, and the demand total before and after (carved + drains' electricity).
    """
    log = log or logger
    flex = built["flex"]
    prof = built["profiles"].reindex(n.snapshots)
    if prof.isna().any().any():
        raise ValueError("CP30 flexibility profiles do not cover the network snapshots")
    dt = _dt(n).to_numpy()
    base = f50._base_load_names(n)
    p_set = n.loads_t.p_set
    load_bus = n.loads.loc[base, "bus"].astype(str)
    bus_load = p_set[base].T.groupby(load_bus.to_numpy()).sum().T            # snapshots x bus
    before = float((p_set[base].sum(axis=1).to_numpy() * dt).sum())

    carved = [c for c in flex if flex[c].get("reference")]
    buses = shares.index[shares.index.isin(n.buses.index)]
    # Hourly carve per bus (MW): sum over carriers of share x national reference.
    carve = pd.DataFrame(0.0, index=n.snapshots, columns=buses)
    for c in carved:
        col = shares.loc[buses, flex[c]["allocation"]].to_numpy()
        carve += np.outer(prof[flex[c]["reference"]].to_numpy(), col)
    load_on = bus_load.reindex(columns=buses).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(carve.to_numpy() > 1e-9, load_on.to_numpy() / carve.to_numpy(), np.inf)
    keep = pd.Series(np.clip(ratio.min(axis=0), 0.0, 1.0), index=buses)        # per-bus factor
    short = keep[keep < 1.0 - 1e-9]
    kept_gwh = float(((carve * (1.0 - keep)).to_numpy() * dt[:, None]).sum() / 1e3)
    if len(short):
        log.warning("CP30 flexibility: %d buses cannot give their whole flexible reference; %.1f GWh stays "
                    "fixed load there (lowest factor %.3f at %s)", len(short), kept_gwh, float(short.min()),
                    str(short.idxmin()))
    carve = carve * keep
    # Remove the carve from each bus's base loads pro rata to their hourly load.
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(load_on.to_numpy() > 0, 1.0 - carve.to_numpy() / load_on.to_numpy(), 1.0)
    factor = pd.DataFrame(np.clip(factor, 0.0, 1.0), index=n.snapshots, columns=buses)
    cols = [l for l in base if load_bus[l] in factor.columns]
    p_set.loc[:, cols] = p_set[cols].to_numpy() * factor[load_bus[cols].to_numpy()].to_numpy()

    carved_gwh: Dict[str, float] = {}
    added = 0
    for carrier, spec in flex.items():
        col = shares[spec["allocation"]]
        col = col[(col > 0) & col.index.isin(n.buses.index)]
        scale = col * (keep.reindex(col.index).fillna(1.0) if spec.get("reference") else 1.0)
        scale = scale[scale > 0]
        if scale.empty:
            continue
        names = [f"{bus} {carrier}" for bus in scale.index]
        kwargs: Dict[str, Any] = dict(
            bus=list(scale.index), carrier=carrier, p_nom=spec["p_nom_mw"] * scale.to_numpy(), p_nom_extendable=False,
            max_hours=float(spec["energy_mwh"]) / float(spec["p_nom_mw"]),
            efficiency_store=float(spec["efficiency_store"]), efficiency_dispatch=float(spec["efficiency_dispatch"]),
            standing_loss=float(spec["standing_loss"]), marginal_cost=float(spec["marginal_cost"]),
            marginal_cost_storage=float(spec.get("marginal_cost_storage", 0.0)),
            cyclic_state_of_charge=False, state_of_charge_initial=0.0,
        )
        for attr in ("p_min_pu", "p_max_pu"):
            value = spec[attr]
            if isinstance(value, str):
                series = prof[value].to_numpy()
                kwargs[attr] = pd.DataFrame(np.repeat(series[:, None], len(names), axis=1), index=n.snapshots,
                                            columns=names)
            else:
                kwargs[attr] = float(value)
        if spec.get("inflow"):
            kwargs["inflow"] = pd.DataFrame(prof[spec["inflow"]].to_numpy()[:, None] * scale.to_numpy()[None, :],
                                            index=n.snapshots, columns=names)
        n.add("StorageUnit", names, **kwargs)
        added += len(names)
        if spec.get("reference"):
            carved_gwh[carrier] = float((prof[spec["reference"]].to_numpy() * dt).sum() * scale.sum() / 1e3)
    for carrier in flex:
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier, co2_emissions=0.0)
    after_loads = float((n.loads_t.p_set[base].sum(axis=1).to_numpy() * dt).sum())
    drains = 0.0
    su = n.storage_units
    inflow = n.storage_units_t.inflow
    for name in su.index[su["carrier"].isin(carved)]:
        if name in inflow.columns:
            drains += float((-inflow[name].to_numpy() * dt).sum()) / float(su.at[name, "efficiency_store"])
    stats = {
        "loads_before_gwh": before / 1e3, "loads_after_gwh": after_loads / 1e3, "carved_gwh": carved_gwh,
        "drain_electricity_gwh": drains / 1e3, "kept_as_fixed_gwh": kept_gwh, "buses_short": int(len(short)),
        "units_added": added,
        "units_p_nom_mw": {c: float(su.loc[su["carrier"].eq(c), "p_nom"].sum()) for c in flex},
        "units_energy_mwh": {c: float((su.loc[su["carrier"].eq(c), "p_nom"]
                                       * su.loc[su["carrier"].eq(c), "max_hours"]).sum()) for c in flex},
    }
    total_after = (after_loads + drains) / 1e3
    if abs(total_after - before / 1e3) > 1e-6 * max(before / 1e3, 1.0) + 1e-6:
        raise ValueError(f"CP30 flexibility changed the demand total: {before / 1e3:,.3f} -> {total_after:,.3f} GWh")
    log.info("CP30 flexibility: %d units; carved %s GWh; fixed loads %.1f -> %.1f GWh; kept as fixed %.1f GWh",
             added, {k: round(v, 1) for k, v in carved_gwh.items()}, before / 1e3, after_loads / 1e3, kept_gwh)
    return stats


def build_and_integrate(n, prep: Dict[str, Any], temp_c: pd.Series, demand_cfg: dict, flex_cfg: dict,
                        ed1: pd.DataFrame, fes: pd.DataFrame, pathway: str, year: int,
                        allocate: Callable[[pd.Series], pd.Series], log: Optional[logging.Logger] = None
                        ) -> Dict[str, Any]:
    """The build step: ED1 figures, national specs, bus shares, carve and units; record in ``n.meta``."""
    log = log or logger
    cfg = resolve_config(flex_cfg)
    ed1_flex = ed1_flexibility(ed1, pathway, year, cfg["ed1_fallback_pathway"])
    fleet = _block_total(fes, demand_cfg["spatial"]["FES_scenario"], year, cfg["ev_fleet_block"])
    temp = temp_c.reindex(n.snapshots)
    built = build_flexibility(n.snapshots, temp, prep, demand_cfg, cfg, ed1_flex, fleet)
    vectors = allocation_vectors(prep, fes, demand_cfg, cfg, year)
    used = {spec["allocation"] for spec in built["flex"].values()}
    shares = bus_shares({k: v for k, v in vectors.items() if k in used}, allocate, float(cfg["min_share"]))
    stats = integrate_flexibility(n, built, shares, log)
    record = {"calibration": built["calibration"], "stats": stats,
              "specs": {c: {k: v for k, v in s.items() if k not in ("inflow", "reference", "p_min_pu", "p_max_pu")}
                        for c, s in built["flex"].items()},
              "ev_fleet": fleet,
              "reference": reference_record(n.snapshots, built)}
    n.meta = {**dict(getattr(n, "meta", None) or {}), META_KEY: _jsonable(record)}
    for carrier, spec in built["flex"].items():
        log.info("  %s: %.0f MW, %.0f MWh, %.1f GWh drained (%s)", carrier, spec["p_nom_mw"], spec["energy_mwh"],
                 spec["electricity_gwh"], spec["source"])
    return record


# ─────────────────────────── the as-built (inflexible) demand ───────────────────────────
#
# The build checks judge where demand sits and how much a node must import (CP30 locational
# validation, the electrolyser import relief). They judge the inflexible demand the CP30
# build calibrated, so they add the carved reference back: each unit's reference is its
# carrier's national reference times p_nom / the carrier's national p_nom (p_nom is share x
# keep factor x national p_nom, exactly the carve's weight at that bus).

def reference_record(snapshots: pd.DatetimeIndex, built: Dict[str, Any]) -> Dict[str, Any]:
    """The national carve references and p_nom per carrier, for ``n.meta`` (JSON-able)."""
    idx = pd.DatetimeIndex(snapshots)
    step = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    out = {"start": str(idx[0]) if len(idx) else None, "step_hours": step, "carriers": {}}
    for carrier, spec in built["flex"].items():
        if spec.get("reference"):
            out["carriers"][carrier] = {
                "spec_p_nom_mw": float(spec["p_nom_mw"]),
                "national_mw": [round(float(v), 3) for v in built["profiles"][spec["reference"]].to_numpy()]}
    return out


def reference_by_bus(storage_units: pd.DataFrame, meta: Optional[dict], snapshots=None, n_rows: Optional[int] = None
                     ) -> pd.DataFrame:
    """Rows x bus MW of the carved references (the inflexible demand the flexibility replaced).

    Args:
        storage_units: The network's storage units (bus, carrier, p_nom).
        meta: ``n.meta`` (or the netCDF ``meta`` attribute, parsed).
        snapshots: Timestamps of the rows (a solved month or week); the reference is aligned by time.
        n_rows: Row count when timestamps are unknown (rows then start at the reference's start).

    Returns:
        Empty when the network has no CP30 flexibility record.
    """
    rec = ((meta or {}).get(META_KEY) or {}).get("reference") or {}
    carriers = rec.get("carriers") or {}
    if not carriers or storage_units is None or len(storage_units) == 0:
        return pd.DataFrame()
    first = next(iter(carriers.values()))["national_mw"]
    start = pd.Timestamp(rec["start"])
    full_index = pd.date_range(start, periods=len(first), freq=pd.Timedelta(hours=float(rec["step_hours"])))
    if snapshots is not None and isinstance(pd.Index(snapshots), pd.DatetimeIndex):
        rows = pd.DatetimeIndex(snapshots)
    else:
        rows = full_index[: (len(first) if n_rows is None else int(n_rows))]
    out = pd.DataFrame(0.0, index=rows, columns=pd.Index([], dtype=object))
    su = storage_units
    for carrier, item in carriers.items():
        units = su[su["carrier"].astype(str).eq(carrier)]
        if units.empty:
            continue
        national = pd.Series(item["national_mw"], index=full_index).reindex(rows).fillna(0.0).to_numpy()
        weight = pd.to_numeric(units["p_nom"], errors="coerce").fillna(0.0) / float(item["spec_p_nom_mw"])
        by_bus = weight.groupby(units["bus"].astype(str)).sum()
        block = pd.DataFrame(np.outer(national, by_bus.to_numpy()), index=rows, columns=by_bus.index)
        out = out.add(block, fill_value=0.0)
    return out.fillna(0.0)


def restore_reference_loads(loads: pd.DataFrame, load_p: pd.DataFrame, storage_units: pd.DataFrame,
                            meta: Optional[dict], snapshots=None) -> pd.DataFrame:
    """``load_p`` (rows x loads) with each bus's carved reference added back to its first base load.

    A no-op for a network without CP30 flexibility (every network before F5, and the 2050 chain).
    """
    ref = reference_by_bus(storage_units, meta, snapshots, n_rows=len(load_p))
    if ref.empty:
        return load_p
    out = load_p.copy()
    carrier = loads.get("carrier", pd.Series("", index=loads.index)).fillna("").astype(str)
    base = loads[carrier.isin(f50.BASE_LOAD_CARRIERS)]
    first = base.reset_index().drop_duplicates("bus").set_index("bus").iloc[:, 0] if len(base) else pd.Series(dtype=str)
    missing = sorted(set(ref.columns) - set(first.index.astype(str)))
    if missing:
        raise ValueError(f"CP30 flexibility reference on buses without a base load: {missing[:5]}")
    values = ref.to_numpy()
    for j, bus in enumerate(ref.columns):
        out[first[bus]] = out[first[bus]].to_numpy() + values[:, j]
    return out


def demand_storage_mask(storage_units: pd.DataFrame) -> pd.Series:
    """True for consumer-flexibility storage units: demand, not storage plant (``DEMAND_FLEX_STORAGE_CARRIERS``)."""
    carrier = storage_units.get("carrier", pd.Series("", index=storage_units.index)).astype(str)
    return carrier.isin(dc.DEMAND_FLEX_STORAGE_CARRIERS)


def _block_total(fes: pd.DataFrame, pathway: str, year: int, block: str) -> float:
    col = next(c for c in ("FES Pathway", "FES Scenario") if c in fes.columns)
    rows = fes[fes["Building Block ID Number"].eq(block) & fes[col].astype(str).str.strip().eq(pathway)]
    return float(pd.to_numeric(rows[str(year)], errors="coerce").sum())


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
