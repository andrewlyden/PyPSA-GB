"""FES long-range demand components and consumer flexibility (plan Y6a; year-generic in stage 5).

Written for FES 2025 Holistic Transition 2050 and run for any FES 2025 pathway and year (the
numbers quoted below are HT 2050's). The demand is a non-electrified base plus explicit components, each pinned to a FES
line:

====================  ==========================================  ==============================
Component             FES 2025 HT 2050 line                        Hourly shape
====================  ==========================================  ==============================
base (residual)       ED1 consumer total less every row below      ESPENI 2013, day-of-year
heat pumps            Dem_BB004 + Dem_BB005                        2013 temperature: space heat
                                                                   (HDH x daily shape) + hot
                                                                   water, temperature COPs
electric vehicles     Dem_BB006 + Dem_BB007 (ED1 res./comm. EVs)   plug-in charging (home,
                                                                   public, depot)
district heat         ED1 Residential District Heat Demand         daily HDH + hot water, large
                                                                   heat-pump COP
data centres          ED1 Data Centre Demand (inside commercial)   flat
DACCS                 ES1 DACCS electrical demand (not in ED1)     flexible, per-window volume
====================  ==========================================  ==============================

Flexibility at FES scale (FLX1) is consumer-owned (``scripts/demand/demand_carriers.py``):
EV smart charging, V2G, heat-pump and district-heat pre-heating, I&C and residential
demand response and DACCS are ``StorageUnit`` components on AC buses. A unit meets a fixed
drain (``inflow < 0``) by charging when the optimiser chooses, within its plug-in or
compressor limit and its energy buffer. Why storage units and not the existing
Store-on-its-own-bus modules (``electric_vehicles.py``, ``heat_pumps.py``):

* rolling market solves carry a storage unit's state of charge between windows and hold
  the month's end at or above its start (``market.storage_soc``), while an ``e_cyclic``
  store is cyclic per window (a 1 h balancing window could not move it);
* no extra buses, so the boundary clustering (which keeps non-AC buses individual) does
  not multiply the LP, and ``aggregateoneport`` merges them exactly: every unit of a
  carrier has the same per-unit profiles and ``max_hours``.

**Windowing (DACCS and every drained unit).** A storage unit's energy balance survives any
window, but a myopic window places no value on energy left at its end. Units therefore
start empty (``state_of_charge_initial = 0``): each window meets its own drain, and can move
consumption EARLIER (pre-charge, pre-heat, bank DACCS) but never later. DACCS's annual
volume is met as a per-window pro-rata volume (drain x window hours), exactly, month by
month; within a window it chooses its hours. Declared.

The module is a library (profile, calibration, allocation, integration and check
functions, all testable on synthetic inputs) and the script of Snakemake rule
``build_fes_demand_components`` (``rules/demand.smk``). The network step is
:func:`integrate_fes_demand`, called from ``finalize_demand_integration.py`` only when
the scenario sets ``demand.disaggregation.fes.enabled: true``. The settings sit under
``demand.disaggregation`` (with ``demand.disaggregation.enabled: false``) so that
``finalize_demand``'s params are unchanged for every other scenario.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.demand import demand_carriers as dc  # noqa: E402
from scripts.utilities.calendar_year import hours_in_year  # noqa: E402

logger = logging.getLogger(__name__)

# ─────────────────────────────── declared defaults ───────────────────────────────
#
# Every value is declared (plan Y6a). A scenario overrides any of them under
# ``demand.disaggregation.fes``; the HT50 snippet repeats them.

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    # ED1 and the building blocks are fiscal years (April Y to March Y+1). "calendar" uses
    # Y1's rule (0.75 FY(Y) + 0.25 FY(Y-1)), which is what the energy validation targets;
    # "fiscal" keeps the load builder's FY consumer total (+0.19% in HT 2050).
    "year_basis": "calendar",
    # Loss basis of the loads: "system" = ED1 system demand excl. electrolysis, incl. losses,
    # the basis of the CP30 2030 loads and of the bill's recovery volumes (loss factor 1.092
    # to the meter); "consumer" = at the meter. See component_energies.
    "demand_basis": "system",
    # Base loads after the carve-out: "per_bus" keeps each bus's total at the base (Dem_BB003
    # GSP shares) and removes the components allocated there; "flat" is the old scalar.
    "residual_allocation": "per_bus",
    "min_share": 1e-6,
    "temperature": {
        # Heating degree hours on a demand-weighted GB temperature (cutout cells nearest the
        # load buses, weighted by base demand), not the grid mean (which includes the sea).
        "hdh_base_c": 15.5,                     # scripts/demand/build_heat_profiles.py
    },
    "heat_pumps": {
        # Hot-water share of heat-pump heat output (thermal). ECUK: water heating is about a
        # fifth of domestic heat; declared 0.20.
        "hot_water_share": 0.20,
        "hot_water_seasonal_amplitude": 0.10,   # colder mains water in January (declared)
        # Space-heating daily shape, hours 0-23 (declared; shape of the RHPP heat pump field
        # trial profiles, Love et al. 2017, Applied Energy 204: overnight set-back, a morning
        # peak and an evening plateau). Normalised to mean 1 inside the code.
        "space_heating_diurnal": [0.75, 0.75, 0.75, 0.75, 0.78, 0.85, 1.10, 1.30, 1.25, 1.10,
                                  0.95, 0.95, 0.95, 0.95, 0.95, 0.98, 1.05, 1.15, 1.20, 1.20,
                                  1.15, 1.05, 0.90, 0.80],
        # Hot-water draw-off shape (declared; shape of EST 2008 "Measurement of domestic hot
        # water consumption in dwellings"). Normalised inside the code.
        "hot_water_diurnal": [0.30, 0.25, 0.25, 0.25, 0.30, 0.50, 1.00, 2.00, 1.80, 1.30,
                              0.90, 0.80, 0.80, 0.80, 0.80, 0.80, 0.90, 1.20, 1.60, 1.70,
                              1.50, 1.30, 0.90, 0.50],
        # COP: Staffell et al. (2012) ASHP fit, COP = a - b dT + c dT^2 (the repository's
        # build_heat_profiles.py constants), dT = sink - ambient. Space-heating sink on a
        # radiator heating curve sink = 40 - 1.0 T (Ruhnau et al. 2019, when2heat), clipped
        # to [30, 50] C; hot water at 50 C. All heat pumps treated as air source (declared).
        "cop_a": 6.81, "cop_b": 0.121, "cop_c": 0.000630,
        "space_sink_intercept_c": 40.0, "space_sink_slope": -1.0,
        "space_sink_min_c": 30.0, "space_sink_max_c": 50.0,
        "hot_water_sink_c": 50.0,
        "cop_min": 1.5, "cop_max": 6.0,
    },
    "district_heat": {
        # Heat networks buffer the diurnal swing: daily-mean HDH, plus hot water.
        "hot_water_share": 0.30,
        # Large heat pump COP = carnot_efficiency x T_sink(K) / (T_sink - T); sink 70 C.
        # (Declared; Danish Energy Agency technology data puts large HP at ~0.45-0.55 of Carnot.)
        "sink_c": 70.0, "carnot_efficiency": 0.50, "cop_min": 1.5, "cop_max": 4.5,
    },
    "electric_vehicles": {
        # Plug-in (unmanaged) charging shapes, hours of day: Gaussian mixtures
        # (weight, mean hour, sd hours), circular (declared parametric; home charging peaks
        # on evening arrival as in the My Electric Avenue and Electric Nation trials; public
        # and workplace charging in the working day; HGV and bus depots overnight). The
        # weekday means and spreads are calibrated so the unmanaged EV load at the peak hours
        # is FES's (ED5 row 21, 19.09 GW in HT 2050: 19.0 here); the smart-charging reduction
        # then lands on FLX1 row 115 (10.09 GW: 10.0 here) with no further tuning.
        "residential_weekday": [[0.85, 19.0, 2.0], [0.15, 8.0, 1.5]],
        "residential_weekend": [[1.0, 15.0, 3.5]],
        "public_weekday": [[0.7, 11.0, 2.5], [0.3, 14.0, 3.0]],
        "public_weekend": [[1.0, 13.5, 3.5]],
        "depot_weekday": [[1.0, 1.0, 2.5]],
        "depot_weekend": [[1.0, 1.0, 2.5]],
        # Daily energy on a weekend day relative to a weekday (declared).
        "weekend_factor": {"residential": 1.0, "public": 0.8, "depot": 0.6},
        # Driving (battery drain) shape of the flexible fleet: the existing synthetic driving
        # profile (electric_vehicles.py), by calendar weekday.
        "driving_weekday": [[0.30, 8.0, 1.5], [0.30, 17.5, 1.5], [0.40, 13.0, 3.0]],
        "driving_weekend": [[1.0, 13.0, 3.5]],
        "driving_weekend_factor": 0.8,
        # Commercial EVs are left inflexible (FES books 7.6 GW at peak, about their mean);
        # the residential flexible share is solved so EV load at the ED1 peak hours is the
        # ED1 figure (1.4 + 7.6 GW). Cap on the residential share.
        "commercial_flex_share": 0.0,
        "max_flex_share": 0.95,
        "charger_kw": 7.0,                      # home charger (existing module default)
        "battery_kwh": 60.0,                    # existing module default
        "usable_battery_fraction": 0.60,        # 20-80% window (declared)
        "charge_efficiency": 0.90,              # existing module default
        "plugged_in_max": 0.90,                 # overnight plug-in share (declared)
        # Cap on the smart-charging and V2G stores, in days of the fleet's mean driving energy
        # (None: the usable battery window, about 9 days). One day keeps the shifting within the
        # day, as FES describes it (FLX1 gives smart charging and V2G only as GW at peak).
        "store_days": None,
    },
    "v2g": {
        "enabled": True,
        "discharge_efficiency": 0.90,           # existing module default
        "degradation_cost_gbp_per_mwh": 50.0,   # existing module default
        "bidirectional_charger_kw": 7.0,
    },
    "heat_pump_flexibility": {
        "enabled": True,
        "storage_hours_at_peak": 2.0,           # hot-water cylinders and fabric (declared)
        "standing_loss_per_hour": 0.02,         # between tank (~1%/h) and fabric (~5%/h)
        "compressor_headroom": 1.5,             # existing module link_sizing_margin
    },
    "district_heat_flexibility": {
        "enabled": True,
        "storage_hours_at_peak": 12.0,          # network thermal stores (declared)
        "standing_loss_per_hour": 0.005,
        "compressor_headroom": 2.0,
    },
    # Storage heaters (user decision 27 Oct 2026; F5, the CP30 2030 method): FLX1's storage
    # heater peak reduction is the direct-acting resistive load at the ACS-peak hours; the energy
    # is that over the space-heat shape at those hours, carved from the residual (it is inside
    # ED1's residential base). Charged up to coldest day / charge_hours (Economy 7: 7 h), a
    # store of store_days x the coldest day; standing loss 0 keeps demand exact. Locations:
    # Dem_BB008a (no FES storage-heater block). Off until the study switches it on with its BM keys.
    # charge_windows ([start, end) hours; None = any hour, F5 fix 28 Oct 2026): Economy 10-style, the
    # night block plus an afternoon boost (00-07 + 13-16). Economy 10 gives ten off-peak hours, usually
    # three in the afternoon, two in the evening and five overnight; we keep the night and afternoon
    # blocks and drop the evening one, which falls in the 16-20 h system peak. Ten hours >= charge_hours,
    # so the coldest day still charges. Without it a flat-price day let Stage 1 put 4-5.5 GW of heater
    # charging at 14:00-20:00 (F5 test, 17 Jan 2030).
    "storage_heater_flexibility": {"enabled": False, "charge_hours": 7.0, "store_days": 1.0,
                                   "standing_loss_per_hour": 0.0, "charge_windows": [[0, 7], [13, 16]]},
    # holding_cost_gbp_per_mwh_h (PyPSA marginal_cost_storage, F5 fix 28 Oct 2026): DSR moves load within
    # shift_hours, so holding energy longer must not pay. marginal_cost / shift_hours (30 / 2 = 15) makes a
    # 2-hour hold cost one activation; without it DSR held its energy 18-117 h (a lossless 2 h battery).
    "ic_dsr": {"enabled": True, "shift_hours": 2.0, "marginal_cost_gbp_per_mwh": 30.0,
               "holding_cost_gbp_per_mwh_h": 15.0},
    "residential_dsr": {"enabled": False, "shift_hours": 2.0, "marginal_cost_gbp_per_mwh": 30.0,
                        "holding_cost_gbp_per_mwh_h": 15.0},
    # Holding cost of every drained unit (EV smart charging, V2G, heat pump, district heat, storage
    # heaters, DACCS; GBP/MWh per hour held; F5 fix 28 Oct 2026). A tie-breaker: on flat or zero prices
    # the units were indifferent to charging early, so the solve filled their stores with surplus they
    # never used (autumn test week: 59 GWh, +1.1% of demand, at the period end). It is small against any
    # real price spread (a 12 h hold costs GBP 0.06/MWh, below the curtailment order's 0.5 merchant step)
    # and is objective only. Tested at 0.05 and 0.005: both leave under 0.04 GWh at the period end, and 0.005
    # keeps more of the flexibility's surplus absorption (spring test week 1,417 against 1,459 GWh lost).
    "drained_holding_cost_gbp_per_mwh_h": 0.005,
    "daccs": {
        "enabled": True,
        # FES gives energy only (ES1 row 611). Capacity at a declared utilisation.
        "utilisation": 0.50,
        "buffer_hours_of_mean_load": 24.0,
        # Declared sites: the CO2 transport and storage clusters.
        "sites": [
            {"name": "Humber", "lat": 53.62, "lon": -0.23, "share": 0.35},
            {"name": "Teesside", "lat": 54.62, "lon": -1.15, "share": 0.30},
            {"name": "St Fergus", "lat": 57.57, "lon": -1.84, "share": 0.20},
            {"name": "HyNet North West", "lat": 53.25, "lon": -2.90, "share": 0.15},
        ],
    },
    # The ED1 peak is an average-cold-spell (ACS) weekday 17:00-18:00. Declared model
    # analogue: 17:00 on the ten coldest winter (Nov-Mar) weekdays of the weather year
    # (2013's cold spell is in March).
    "calibration": {"hour": 17, "months": [11, 12, 1, 2, 3], "coldest_days": 10},
}


#: ``demand.disaggregation.<key>`` of the components.
CONFIG_KEY = "fes"


def fes_demand_config(disaggregation_config: Optional[dict]) -> dict:
    """``demand.disaggregation.fes`` (given the ``demand.disaggregation`` block), else ``{}``."""
    return dict((disaggregation_config or {}).get(CONFIG_KEY) or {})


def resolve_config(cfg: Optional[dict]) -> dict:
    """``demand.disaggregation.fes`` merged over :data:`DEFAULT_CONFIG` (nested dicts merged)."""
    def _merge(base: dict, over: dict) -> dict:
        out = dict(base)
        for k, v in (over or {}).items():
            out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        return out
    return _merge(DEFAULT_CONFIG, cfg or {})


def is_enabled(cfg: Optional[dict]) -> bool:
    """True only when a scenario explicitly sets ``demand.disaggregation.fes.enabled: true``."""
    return bool((cfg or {}).get("enabled", False))


# ───────────────────────────── FES targets and flexibility ─────────────────────────────

# FLX1 rows (FES 2025 workbook, sheet FLX1). Values are GW,
# negative in the workbook for peak reductions; returned positive.
FLX1_ITEMS = {
    "smart_charging_at_peak": "Smart charging impact at peak",
    "v2g_at_peak": "V2G potential at peak",
    "v2g_max": "V2G maximum potential",
    "residential_appliance_dsr": "Residential Appliance DSR impact at peak",
    "ic_dsr": "I&C DSR impact at peak",
    "res_hybrid_hp_flex": "Residential peak reduction - hybrid heat pump flex",
    "res_thermal_storage_flex": "Residential peak reduction - thermal storage flex",
    # Separate rows, not inside thermal storage: FLX1 row 185 (combined consumer flexibility) is the
    # plain sum of every row here, storage heaters and thermal storage each once (HT 2050: 81.56 GW).
    "res_storage_heater_flex": "Residential peak reduction - storage heater flex",
    "nonres_storage_heater_flex": "Non-residential peak reduction - storage heater flex",
    "res_district_heat_flex": "Residential peak reduction - district heating flex",
    "nonres_hybrid_hp_flex": "Non-residential peak reduction - hybrid heat pump flex",
    "nonres_thermal_storage_flex": "Non-residential peak reduction - thermal storage flex",
    "nonres_district_heat_flex": "Non-residential peak reduction - district heating flex",
    "unconstrained_heat_peak": "Unconstrained peak electricity demand for heat",
    "total_consumer_flex": "Combined flexibility capacity from all consumer led sources",
}


def flx1_flexibility(flx1: pd.DataFrame, pathway: str, year: int) -> Dict[str, Dict[str, Any]]:
    """FLX1 consumer-flexibility items for a pathway and year, GW (positive), with rows.

    Args:
        flx1: FLX1 table (``scripts.FES.fes2025_targets.read_flx1``).
        pathway: FES pathway.
        year: Year column (FLX1 is published per year; peaks have no fiscal basis).

    Returns:
        item -> {gw, excel_row, detail}.
    """
    rows = flx1[flx1["Pathway"].astype(str).str.strip().eq(pathway)]
    out: Dict[str, Dict[str, Any]] = {}
    for key, detail in FLX1_ITEMS.items():
        match = rows[rows["Detail"].astype(str).str.strip().eq(detail)]
        if len(match) != 1:
            raise ValueError(f"FLX1 {detail!r} ({pathway}): expected one row, found {len(match)}")
        row = match.iloc[0]
        out[key] = {"gw": abs(float(pd.to_numeric(row[str(year)], errors="coerce"))),
                    "excel_row": int(row["excel_row"]), "detail": detail}
    return out


def component_energies(targets: dict, year_basis: str = "calendar", demand_basis: str = "system") -> Dict[str, Any]:
    """National annual energies (GWh) of every component, from the Y1 targets JSON.

    **Loss basis.** The CP30 2030 loads total "GB Total System Demand - Minus electrolysis"
    (315.16 TWh), which includes transmission and distribution losses, and the bill's
    recovery file (``data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json``)
    treats modelled load as that loss-inclusive GSP-level volume and divides by the 1.092
    GSP-to-meter factor for metered supply. ``demand_basis: system`` (the default) puts 2050
    on the same basis: loads total ED1 row 9 less row 74 (system demand excluding
    electrolysis, including losses), and every component's FES line (an at-the-meter
    consumer figure) is uplifted by the one loss factor ``system / consumer`` (1.0866 in HT
    2050), so losses fall pro rata on every use. ``consumer`` keeps the meter basis.

    Args:
        targets: ``fes2025_energy_targets`` output as JSON (``targets_to_json``) or dict.
        year_basis: "calendar" or "fiscal" (see :data:`DEFAULT_CONFIG`).
        demand_basis: "system" (loss-inclusive, as CP30 2030) or "consumer" (at the meter).

    Returns:
        ``demand_total_gwh`` and each component on the model basis, the FES line values
        before the uplift (``fes_line_gwh``), the loss factor, the ED1 EV split and DACCS
        (ES1, not uplifted: transmission-connected, like electrolysis), with sources.
    """
    if year_basis not in {"calendar", "fiscal"}:
        raise ValueError("year_basis must be 'calendar' or 'fiscal'")
    if demand_basis not in {"system", "consumer"}:
        raise ValueError("demand_basis must be 'system' or 'consumer'")
    key = "calendar_gwh" if year_basis == "calendar" else "fy_gwh"
    demand = targets["demand"]
    annual = demand["annual"]
    bb = targets["demand_components"]["building_blocks"]
    consumer = float(demand["totals"]["consumer_excluding_electrolysis"][key])
    system = float(demand["totals"]["system_excluding_electrolysis"][key])
    loss_factor = system / consumer if demand_basis == "system" else 1.0
    lines = {
        "heat_pumps": float(bb["Dem_BB004"][key] + bb["Dem_BB005"][key]),
        "evs": float(bb["Dem_BB006"][key] + bb["Dem_BB007"][key]),
        "district_heat": float(annual["residential_district_heat"][key]),
        "data_centres": float(annual["data_centres"][key]),
    }
    lines["residual"] = consumer - sum(lines.values())
    if lines["residual"] <= 0:
        raise ValueError(f"Residual demand is not positive: {lines['residual']:.1f} GWh")
    ev_res = float(annual["residential_evs"][key])
    ev_com = float(annual["commercial_evs"][key])
    daccs = float(targets.get("daccs_gwh", targets["demand_components"].get("daccs_gwh", 0.0)))
    total_label = ("ED1 system demand excl. electrolysis, incl. losses (row 9 - row 74)" if demand_basis == "system"
                   else "ED1 consumer demand excl. electrolysis and losses")
    out = {
        "year_basis": year_basis,
        "demand_basis": demand_basis,
        "loss_factor": loss_factor,
        "consumer_gwh": consumer,
        "system_gwh": system,
        "demand_total_gwh": consumer * loss_factor,
        **{f"{k}_gwh": v * loss_factor for k, v in lines.items()},
        "fes_line_gwh": dict(lines, consumer_total=consumer),
        # ED1 residential and commercial EVs close on Dem_BB006 + BB007 (research note);
        # their split sets the home/public/depot shape mix. BB007 (HGVs and buses) is depot.
        "ev_residential_share": ev_res / (ev_res + ev_com),
        "ev_depot_share": min(1.0, float(bb["Dem_BB007"][key]) / lines["evs"]),
        "daccs_gwh": daccs,
        "sources": {
            "demand_total": f"{total_label} ({year_basis})",
            "heat_pumps": "Dem_BB004 + Dem_BB005",
            "evs": "Dem_BB006 + Dem_BB007 (= ED1 rows 49 + 104)",
            "district_heat": f"ED1 row {annual['residential_district_heat'].get('excel_row')}",
            "data_centres": f"ED1 row {annual['data_centres'].get('excel_row')} (L5, inside commercial)",
            "residual": "ED1 consumer total less the four lines above",
            "daccs": "ES1 row 611 (not in ED1; not loss-uplifted)",
        },
    }
    if demand_basis == "system":
        for k in ("heat_pumps", "evs", "district_heat", "data_centres", "residual"):
            out["sources"][k] += f" x loss factor {loss_factor:.4f}"
    return out


def peak_targets(targets: dict, flex: Dict[str, Dict[str, Any]], loss_factor: float = 1.0) -> Dict[str, float]:
    """ED1 and FLX1 peak figures the calibration and the checks use (GW).

    The per-component peak figures (EVs, heat pumps, district heat and their FLX1 flexibility)
    are consumer-level; they are multiplied by ``loss_factor`` so they sit on the same basis
    as the loads (:func:`component_energies`). Capacities (V2G, DSR) are not uplifted.
    """
    peaks = targets["demand"]["peak_gw"]

    def _p(label: str) -> float:
        return float(peaks[label]["gw"])

    losses = sum(_p(k) for k in ("Residential Losses", "Industrial Losses", "Commercial Losses"))
    total = _p("GBFES Peak Customer Demand: Total Consumption plus Losses")
    lf = float(loss_factor)
    return {
        "loss_factor": lf,
        "ed1_peak_including_losses_gw": total,
        "ed1_peak_losses_gw": losses,
        "ed1_peak_consumer_gw": total - losses,
        # The ED1 peak on the loads' own basis: incl. losses when the loads are (loss_factor > 1).
        "ed1_peak_model_basis_gw": total if lf > 1.0 else total - losses,
        "ev_at_peak_gw": lf * (_p("Residential EVs at Peak") + _p("Commercial EVs at Peak")),
        "heat_pumps_at_peak_gw": lf * (_p("Residential Heat Pumps at Peak") + _p("Commercial Heat Pumps at Peak")),
        "district_heat_at_peak_gw": lf * (_p("Residential District Heat at Peak") + _p("Commercial District Heat at Peak")
                                          + _p("Industrial District Heat at Peak")),
        "smart_charging_at_peak_gw": lf * flex["smart_charging_at_peak"]["gw"],
        "hp_flex_at_peak_gw": lf * sum(flex[k]["gw"] for k in ("res_hybrid_hp_flex", "res_thermal_storage_flex",
                                                               "nonres_hybrid_hp_flex", "nonres_thermal_storage_flex")),
        "dh_flex_at_peak_gw": lf * (flex["res_district_heat_flex"]["gw"] + flex["nonres_district_heat_flex"]["gw"]),
        "storage_heater_flex_at_peak_gw": lf * (flex["res_storage_heater_flex"]["gw"]
                                                + flex["nonres_storage_heater_flex"]["gw"]),
        "v2g_max_gw": flex["v2g_max"]["gw"],
        "v2g_at_peak_gw": flex["v2g_at_peak"]["gw"],
        "ic_dsr_gw": flex["ic_dsr"]["gw"],
        "residential_dsr_gw": flex["residential_appliance_dsr"]["gw"],
    }


# ───────────────────────────────── temperature ─────────────────────────────────

def bus_lonlat(buses: pd.DataFrame) -> pd.DataFrame:
    """Bus longitude/latitude, converting OSGB36 metres when the x range says so."""
    x = pd.to_numeric(buses["x"], errors="coerce")
    y = pd.to_numeric(buses["y"], errors="coerce")
    if (x.max() - x.min()) > 1000:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x.to_numpy(), y.to_numpy())
        return pd.DataFrame({"lon": lon, "lat": lat}, index=buses.index)
    return pd.DataFrame({"lon": x.to_numpy(), "lat": y.to_numpy()}, index=buses.index)


def demand_weighted_temperature(temperature, lon: np.ndarray, lat: np.ndarray,
                                weights: np.ndarray) -> pd.Series:
    """GB temperature (deg C) weighted by demand at the nearest cutout cell.

    Args:
        temperature: xarray DataArray (time, y, x) in K or C, x = lon, y = lat.
        lon, lat: Locations of the demand (degrees).
        weights: Demand at each location (any unit).

    Returns:
        Hourly series indexed by the cutout's time.
    """
    xs = np.asarray(temperature["x"].values, dtype=float)
    ys = np.asarray(temperature["y"].values, dtype=float)
    ix = np.abs(xs[None, :] - np.asarray(lon, dtype=float)[:, None]).argmin(axis=1)
    iy = np.abs(ys[None, :] - np.asarray(lat, dtype=float)[:, None]).argmin(axis=1)
    cells = pd.DataFrame({"iy": iy, "ix": ix, "w": np.asarray(weights, dtype=float)})
    cells = cells[cells["w"] > 0].groupby(["iy", "ix"], as_index=False)["w"].sum()
    if cells.empty:
        raise ValueError("No positive demand weights for the temperature average")
    values = np.stack([temperature.isel(y=int(r.iy), x=int(r.ix)).values for r in cells.itertuples()], axis=1)
    t = values @ (cells["w"].to_numpy() / cells["w"].sum())
    if np.nanmean(t) > 200:
        t = t - 273.15
    return pd.Series(t, index=pd.DatetimeIndex(temperature["time"].values), name="temperature_c")


def map_weather_year(series: pd.Series, snapshots: pd.DatetimeIndex) -> pd.Series:
    """Put a weather-year series on the model snapshots by month, day and hour.

    The same day-of-year alignment the ESPENI base and the renewable profiles use. A 29
    February missing from the source takes 28 February's value.
    """
    src = series.copy()
    key = pd.MultiIndex.from_arrays([src.index.month, src.index.day, src.index.hour])
    src = pd.Series(src.to_numpy(), index=key).groupby(level=[0, 1, 2]).mean()
    snaps = pd.DatetimeIndex(snapshots)
    day = snaps.day.to_numpy().copy()
    has_leap_day = bool(((src.index.get_level_values(0) == 2) & (src.index.get_level_values(1) == 29)).any())
    if not has_leap_day:
        day[(snaps.month == 2) & (snaps.day == 29)] = 28
    target = pd.MultiIndex.from_arrays([snaps.month, day, snaps.hour])
    values = src.reindex(target).to_numpy()
    out = pd.Series(values, index=snaps, name=series.name)
    if out.isna().any():
        out = out.interpolate(limit_direction="both")
    return out


# ───────────────────────────────── profiles ─────────────────────────────────

def _diurnal(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (24,):
        raise ValueError("A diurnal shape needs 24 hourly values")
    return arr / arr.mean()


def gaussian_day(hours: np.ndarray, mixture: Iterable[Iterable[float]]) -> np.ndarray:
    """Circular Gaussian mixture over hour of day (weight, mean hour, sd), integral per day 1."""
    h = np.asarray(hours, dtype=float)
    out = np.zeros_like(h)
    total_w = 0.0
    for weight, mean, sd in mixture:
        d = (h - float(mean) + 12.0) % 24.0 - 12.0
        out += float(weight) * np.exp(-0.5 * (d / float(sd)) ** 2) / (float(sd) * np.sqrt(2 * np.pi))
        total_w += float(weight)
    return out / total_w


def ashp_cop(delta_t: np.ndarray, p: dict) -> np.ndarray:
    """Staffell et al. (2012) ASHP COP, clipped."""
    d = np.asarray(delta_t, dtype=float)
    return np.clip(p["cop_a"] - p["cop_b"] * d + p["cop_c"] * d ** 2, p["cop_min"], p["cop_max"])


def heat_pump_profiles(temp_c: pd.Series, cfg: dict) -> pd.DataFrame:
    """Heat pump thermal and electric shape (decision 21).

    Space heat = HDH (on ``hdh_base_c``) x a daily heating shape; hot water = a daily
    draw-off shape with a mild seasonal swing; each divided by its own temperature COP.

    Args:
        temp_c: Hourly demand-weighted temperature on the model snapshots.
        cfg: Resolved config (keys ``temperature`` and ``heat_pumps``).

    Returns:
        Columns space_heat_th, hot_water_th, cop_space, cop_water, electric (MW per unit;
        the thermal columns sum to 1 over the year, electric is in the same units).
    """
    p = cfg["heat_pumps"]
    t = temp_c.to_numpy(dtype=float)
    idx = temp_c.index
    hdh = np.clip(cfg["temperature"]["hdh_base_c"] - t, 0.0, None)
    space = hdh * _diurnal(p["space_heating_diurnal"])[idx.hour]
    doy = idx.dayofyear.to_numpy()
    days = np.where(idx.is_leap_year, 366.0, 365.0)   # the cycle spans the calendar year, leap or not
    seasonal = 1.0 + p["hot_water_seasonal_amplitude"] * np.cos(2 * np.pi * (doy - 15) / days)
    water = _diurnal(p["hot_water_diurnal"])[idx.hour] * seasonal
    share = float(p["hot_water_share"])
    space_th = (1 - share) * space / space.sum()
    water_th = share * water / water.sum()
    sink = np.clip(p["space_sink_intercept_c"] + p["space_sink_slope"] * t, p["space_sink_min_c"], p["space_sink_max_c"])
    cop_space = ashp_cop(sink - t, p)
    cop_water = ashp_cop(p["hot_water_sink_c"] - t, p)
    electric = space_th / cop_space + water_th / cop_water
    return pd.DataFrame({"space_heat_th": space_th, "hot_water_th": water_th, "cop_space": cop_space,
                         "cop_water": cop_water, "electric": electric}, index=idx)


def district_heat_profile(temp_c: pd.Series, cfg: dict) -> pd.DataFrame:
    """District heat electricity shape: daily-mean HDH plus hot water, large heat-pump COP."""
    p = cfg["district_heat"]
    t = temp_c.to_numpy(dtype=float)
    idx = temp_c.index
    hdh = pd.Series(np.clip(cfg["temperature"]["hdh_base_c"] - t, 0.0, None), index=idx)
    daily = hdh.groupby(idx.normalize()).transform("mean").to_numpy()
    share = float(p["hot_water_share"])
    heat = (1 - share) * daily / daily.sum() + share / len(idx)
    sink = float(p["sink_c"])
    cop = np.clip(p["carnot_efficiency"] * (sink + 273.15) / np.maximum(sink - t, 1.0), p["cop_min"], p["cop_max"])
    return pd.DataFrame({"heat_th": heat, "cop": cop, "electric": heat / cop}, index=idx)


def _daily_mixture_profile(index: pd.DatetimeIndex, weekday_mix, weekend_mix, weekend_factor: float) -> np.ndarray:
    hours = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    weekend = index.dayofweek.to_numpy() >= 5
    shape = np.where(weekend, weekend_factor * gaussian_day(hours, weekend_mix), gaussian_day(hours, weekday_mix))
    return shape / shape.mean()


def ev_profiles(index: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """EV plug-in charging shapes (home, public, depot) and the driving drain, mean 1 each.

    Weekdays come from the model calendar (the old synthetic profile counted weekdays from
    snapshot 0).
    """
    p = cfg["electric_vehicles"]
    wf = p["weekend_factor"]
    out = pd.DataFrame(index=index)
    out["residential"] = _daily_mixture_profile(index, p["residential_weekday"], p["residential_weekend"],
                                                wf["residential"])
    out["public"] = _daily_mixture_profile(index, p["public_weekday"], p["public_weekend"], wf["public"])
    out["depot"] = _daily_mixture_profile(index, p["depot_weekday"], p["depot_weekend"], wf["depot"])
    out["driving"] = _daily_mixture_profile(index, p["driving_weekday"], p["driving_weekend"],
                                            p["driving_weekend_factor"])
    return out


def storage_heater_flexibility(temp_c: pd.Series, mask: np.ndarray, flex_at_peak_mw: float, block: dict,
                               cfg: dict) -> Dict[str, Any]:
    """Storage-heater unit (``demand_carriers.STORAGE_HEATER_FLEXIBILITY``), shared by 2030 and 2050.

    The FLX1/ED1 storage heater peak reduction is read as the direct-acting resistive load at the
    ACS-peak hours (``mask``): energy = that MW over the space-heat shape (:func:`heat_pump_profiles`,
    COP 1) at those hours. Drain = the space heat; charging power = coldest day / ``charge_hours``;
    store = ``store_days`` x the coldest day.

    Returns ``spec`` (StorageUnit spec, inflow column ``SH_flexibility_drain``), ``drain`` (MW) and
    ``daily_mwh`` (each snapshot's day energy).
    """
    idx = temp_c.index
    dt = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    space = heat_pump_profiles(temp_c, cfg)["space_heat_th"].to_numpy()
    space = space / (space.sum() * dt)                       # MW per MWh a year
    energy = float(flex_at_peak_mw) / float(space[mask].mean())
    drain = space * energy
    daily = pd.Series(drain * dt, index=idx).groupby(idx.normalize()).transform("sum").to_numpy()
    coldest = float(daily.max())
    spec = {"p_nom_mw": coldest / float(block["charge_hours"]), "energy_mwh": float(block["store_days"]) * coldest,
            "efficiency_store": 1.0, "efficiency_dispatch": 1.0, "standing_loss": float(block["standing_loss_per_hour"]),
            "marginal_cost": 0.0, "p_min_pu": -1.0, "p_max_pu": 0.0, "inflow": f"{dc.STORAGE_HEATER_FLEXIBILITY}_drain",
            "allocation": "storage_heaters", "electricity_gwh": float(drain.sum() * dt / 1e3)}
    out = {"spec": spec, "drain": drain, "daily_mwh": daily}
    windows = block.get("charge_windows")
    if windows:
        out["p_min_pu"] = charge_window_pu(idx, windows, float(block["charge_hours"]))
        spec["p_min_pu"] = f"{dc.STORAGE_HEATER_FLEXIBILITY}_p_min_pu"
    return out


def charge_window_pu(index: pd.DatetimeIndex, windows, charge_hours: float) -> np.ndarray:
    """p_min_pu that lets a unit charge only inside ``windows`` ([start, end) hours of the day): -1 inside, 0 outside.

    Refuses windows shorter than ``charge_hours`` a day (the coldest day could not charge).
    """
    spans = [(int(a), int(b)) for a, b in windows]
    if any(not 0 <= a < b <= 24 for a, b in spans):
        raise ValueError(f"charge_windows must be [start, end) hours within 0-24, got {windows}")
    hour = pd.DatetimeIndex(index).hour.to_numpy()
    inside = np.zeros(len(hour), dtype=bool)
    for a, b in spans:
        inside |= (hour >= a) & (hour < b)
    if sum(b - a for a, b in spans) < float(charge_hours) - 1e-9:
        raise ValueError(f"charge_windows {windows} give fewer than charge_hours = {charge_hours} hours a day")
    return np.where(inside, -1.0, 0.0)


def holding_costs(flex: Dict[str, Dict[str, Any]], cfg: dict) -> None:
    """Set ``marginal_cost_storage`` on every spec (in place): DSR its block's holding cost, drained units
    the config's ``drained_holding_cost_gbp_per_mwh_h`` (F5 fix, 28 Oct 2026)."""
    drained = float(cfg.get("drained_holding_cost_gbp_per_mwh_h") or 0.0)
    for carrier, spec in flex.items():
        if carrier == dc.IC_DSR:
            spec["marginal_cost_storage"] = float(cfg["ic_dsr"].get("holding_cost_gbp_per_mwh_h") or 0.0)
        elif carrier == dc.RESIDENTIAL_DSR:
            spec["marginal_cost_storage"] = float(cfg["residential_dsr"].get("holding_cost_gbp_per_mwh_h") or 0.0)
        elif spec.get("inflow"):
            spec["marginal_cost_storage"] = drained
        else:
            spec.setdefault("marginal_cost_storage", 0.0)


def calibration_mask(temp_c: pd.Series, cfg: dict) -> pd.Series:
    """ACS-peak analogue: ``hour`` on the ``coldest_days`` coldest winter weekdays."""
    c = cfg["calibration"]
    idx = temp_c.index
    daily = temp_c.groupby(idx.normalize()).mean()
    days = daily[(daily.index.month.isin(c["months"])) & (daily.index.dayofweek < 5)]
    cold = set(days.nsmallest(int(c["coldest_days"])).index)
    mask = pd.Series(idx.normalize().isin(list(cold)) & (idx.hour == int(c["hour"])), index=idx)
    if not mask.any():
        raise ValueError("Calibration mask is empty: check calibration.months and snapshots")
    return mask


def solve_flex_share(unmanaged_at_peak_gw: float, other_fixed_at_peak_gw: float, target_fixed_at_peak_gw: float,
                     max_share: float) -> float:
    """Flexible share s so that other + (1 - s) x unmanaged = the target at the peak hours."""
    if unmanaged_at_peak_gw <= 0:
        return 0.0
    s = 1.0 - (target_fixed_at_peak_gw - other_fixed_at_peak_gw) / unmanaged_at_peak_gw
    return float(np.clip(s, 0.0, max_share))


# ─────────────────────────────── component build ───────────────────────────────

def build_components(snapshots: pd.DatetimeIndex, temp_c: pd.Series, energies: Dict[str, Any],
                     peaks: Dict[str, float], ev_fleet: float, cfg: dict) -> Dict[str, Any]:
    """National hourly component profiles, flexibility specs and the calibration record.

    Args:
        snapshots: Model snapshots (hourly or finer; energy uses the snapshot length).
        temp_c: Demand-weighted temperature on ``snapshots``.
        energies: :func:`component_energies`.
        peaks: :func:`peak_targets`.
        ev_fleet: Number of battery EVs (Lct_BB001, cars/vans/motorbikes).
        cfg: Resolved config.

    Returns:
        ``profiles`` (national MW, and per-unit availability columns), ``flex`` (per carrier
        spec, national), ``calibration`` (shares and at-peak values), ``energies``.
    """
    idx = pd.DatetimeIndex(snapshots)
    dt = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    hours = len(idx) * dt
    mask = calibration_mask(temp_c, cfg).to_numpy()
    prof = pd.DataFrame(index=idx)

    def _mw(shape: np.ndarray, gwh: float) -> np.ndarray:
        shape = np.asarray(shape, dtype=float)
        return shape / (shape.sum() * dt) * gwh * 1e3

    def _at_peak(mw: np.ndarray) -> float:
        return float(np.asarray(mw)[mask].mean() / 1e3)

    # Heat pumps.
    hp = heat_pump_profiles(temp_c, cfg)
    hp_mw = _mw(hp["electric"].to_numpy(), energies["heat_pumps_gwh"])
    # District heat and data centres.
    dh = district_heat_profile(temp_c, cfg)
    dh_mw = _mw(dh["electric"].to_numpy(), energies["district_heat_gwh"])
    dc_mw = np.full(len(idx), energies["data_centres_gwh"] * 1e3 / hours)
    # EVs by segment.
    evp = ev_profiles(idx, cfg)
    e = cfg["electric_vehicles"]
    ev_gwh = energies["evs_gwh"]
    res_gwh = ev_gwh * energies["ev_residential_share"]
    depot_gwh = min(ev_gwh * energies["ev_depot_share"], ev_gwh - res_gwh)
    public_gwh = ev_gwh - res_gwh - depot_gwh
    res_mw = _mw(evp["residential"].to_numpy(), res_gwh)
    com_mw = _mw(evp["public"].to_numpy(), public_gwh) + _mw(evp["depot"].to_numpy(), depot_gwh)
    s_com = float(e["commercial_flex_share"])
    s_res = solve_flex_share(_at_peak(res_mw), (1 - s_com) * _at_peak(com_mw), peaks["ev_at_peak_gw"],
                             float(e["max_flex_share"]))
    ev_fixed_mw = (1 - s_res) * res_mw + (1 - s_com) * com_mw
    ev_flex_gwh = s_res * res_gwh + s_com * public_gwh + s_com * depot_gwh

    calibration = {
        "calibration_hours": int(mask.sum()),
        "ev_residential_flex_share": s_res, "ev_commercial_flex_share": s_com,
        "ev_unmanaged_at_peak_gw": _at_peak(res_mw + com_mw),
        "ev_fixed_at_peak_gw": _at_peak(ev_fixed_mw), "ev_fixed_target_at_peak_gw": peaks["ev_at_peak_gw"],
        "ev_smart_impact_at_peak_gw": _at_peak(res_mw + com_mw) - _at_peak(ev_fixed_mw),
        "ev_smart_impact_fes_gw": peaks["smart_charging_at_peak_gw"],
        "hp_unmanaged_at_peak_gw": _at_peak(hp_mw), "hp_annual_peak_gw": float(hp_mw.max() / 1e3),
        "hp_fes_at_peak_gw": peaks["heat_pumps_at_peak_gw"],
        "hp_effective_cop": float((hp["space_heat_th"] + hp["hot_water_th"]).sum() / hp["electric"].sum()),
        "hp_min_cop_space": float(hp["cop_space"].min()),
        "dh_unmanaged_at_peak_gw": _at_peak(dh_mw), "dh_fes_at_peak_gw": peaks["district_heat_at_peak_gw"],
        "coldest_hour_temperature_c": float(temp_c.min()),
        "calibration_mean_temperature_c": float(temp_c.to_numpy()[mask].mean()),
    }

    flex: Dict[str, Dict[str, Any]] = {}
    # Heat-pump pre-heating: the flexible part at the peak hours is the FLX1 figure.
    hpf = cfg["heat_pump_flexibility"]
    s_hp = 0.0
    if hpf.get("enabled", True):
        s_hp = float(np.clip(peaks["hp_flex_at_peak_gw"] / max(_at_peak(hp_mw), 1e-9), 0.0, 1.0))
        drain = s_hp * hp_mw
        p_nom = float(hpf["compressor_headroom"] * drain.max())
        flex[dc.HEAT_PUMP_FLEXIBILITY] = {
            "p_nom_mw": p_nom, "energy_mwh": float(hpf["storage_hours_at_peak"] * _at_peak(drain) * 1e3),
            "efficiency_store": 1.0, "efficiency_dispatch": 1.0, "standing_loss": float(hpf["standing_loss_per_hour"]),
            "marginal_cost": 0.0, "p_min_pu": -1.0, "p_max_pu": 0.0, "inflow": "HP_flexibility_drain",
            "allocation": "heat_pumps", "electricity_gwh": s_hp * energies["heat_pumps_gwh"],
            "source": "FLX1 rows 135, 140, 160, 165 (hybrid + thermal storage flex at peak); electric-equivalent store",
        }
        prof["HP_flexibility_drain"] = -drain
    calibration["hp_flex_share"] = s_hp
    prof[dc.HEAT_PUMP_LOAD] = (1 - s_hp) * hp_mw
    calibration["hp_fixed_at_peak_gw"] = _at_peak((1 - s_hp) * hp_mw)

    dhf = cfg["district_heat_flexibility"]
    s_dh = 0.0
    if dhf.get("enabled", True) and energies["district_heat_gwh"] > 0:
        s_dh = float(np.clip(peaks["dh_flex_at_peak_gw"] / max(_at_peak(dh_mw), 1e-9), 0.0, 1.0))
        drain = s_dh * dh_mw
        flex[dc.DISTRICT_HEAT_FLEXIBILITY] = {
            "p_nom_mw": float(dhf["compressor_headroom"] * drain.max()),
            "energy_mwh": float(dhf["storage_hours_at_peak"] * _at_peak(drain) * 1e3),
            "efficiency_store": 1.0, "efficiency_dispatch": 1.0, "standing_loss": float(dhf["standing_loss_per_hour"]),
            "marginal_cost": 0.0, "p_min_pu": -1.0, "p_max_pu": 0.0, "inflow": "DH_flexibility_drain",
            "allocation": "district_heat", "electricity_gwh": s_dh * energies["district_heat_gwh"],
            "source": "FLX1 rows 145, 170 (district heating flex at peak); capped at the whole DH load",
        }
        prof["DH_flexibility_drain"] = -drain
    calibration["dh_flex_share"] = s_dh
    prof[dc.DISTRICT_HEAT_LOAD] = (1 - s_dh) * dh_mw
    prof[dc.DATA_CENTRE_LOAD] = dc_mw
    prof[dc.EV_LOAD] = ev_fixed_mw

    # EV smart charging and V2G (the V2G fleet is part of the flexible pool).
    eta_c = float(e["charge_efficiency"])
    drive = evp["driving"].to_numpy()
    drive_cal = float(drive[mask].mean())
    avail = float(e["plugged_in_max"]) * np.clip(1.0 - (1.0 - peaks["v2g_at_peak_gw"] / peaks["v2g_max_gw"])
                                                 * drive / drive_cal, 0.02, 1.0)
    prof["ev_plugged_in"] = avail
    v2g_cfg = cfg["v2g"]
    flex_vehicles = s_res * ev_fleet
    v2g_vehicles = 0.0
    if v2g_cfg.get("enabled", True) and peaks["v2g_max_gw"] > 0 and flex_vehicles > 0:
        v2g_vehicles = min(flex_vehicles, peaks["v2g_max_gw"] * 1e6 / float(v2g_cfg["bidirectional_charger_kw"]))
    v2g_frac = v2g_vehicles / flex_vehicles if flex_vehicles > 0 else 0.0
    usable_mwh = float(e["battery_kwh"]) * float(e["usable_battery_fraction"]) / 1e3
    for carrier, frac, vehicles in ((dc.EV_SMART_CHARGING, 1.0 - v2g_frac, flex_vehicles - v2g_vehicles),
                                    (dc.V2G, v2g_frac, v2g_vehicles)):
        gwh = ev_flex_gwh * frac
        if gwh <= 0 or vehicles <= 0:
            continue
        drain = _mw(drive, gwh * eta_c)
        charge_mw = vehicles * float(e["charger_kw"]) / 1e3
        energy_mwh = vehicles * usable_mwh
        if e.get("store_days") is not None:
            energy_mwh = min(energy_mwh, float(e["store_days"]) * float(drain.mean()) * 24.0)
        spec = {
            "energy_mwh": energy_mwh, "efficiency_store": eta_c, "efficiency_dispatch": 1.0,
            "standing_loss": 0.0, "marginal_cost": 0.0, "inflow": f"{carrier}_drain",
            "allocation": "electric_vehicles" if carrier == dc.EV_SMART_CHARGING else "v2g",
            "electricity_gwh": gwh, "vehicles": vehicles, "charge_mw": charge_mw,
            "source": ("ED1 EVs at peak (rows 154, 244) sets the flexible share; fleet Lct_BB001"
                       if carrier == dc.EV_SMART_CHARGING else
                       "FLX1 row 125 (V2G maximum) and row 120 (at peak); Srg_BB005 locations"),
        }
        if carrier == dc.V2G:
            discharge_mw = peaks["v2g_max_gw"] * 1e3
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
    calibration["v2g_vehicles"] = v2g_vehicles
    calibration["ev_flexible_vehicles"] = flex_vehicles

    # Storage heaters (27 Oct 2026): carved from the residual through comp_bus (integrate_fes_demand).
    shf = cfg.get("storage_heater_flexibility") or {}
    if shf.get("enabled", False) and peaks.get("storage_heater_flex_at_peak_gw", 0.0) > 0:
        sh = storage_heater_flexibility(temp_c, mask, peaks["storage_heater_flex_at_peak_gw"] * 1e3, shf, cfg)
        flex[dc.STORAGE_HEATER_FLEXIBILITY] = {
            **sh["spec"], "source": "FLX1 rows 130, 155 (storage heater flex at peak) = direct-acting load at the "
                                    "ACS-peak hours; energy over the space-heat shape; inside the ED1 residual"}
        prof[sh["spec"]["inflow"]] = -sh["drain"]
        if "p_min_pu" in sh:
            prof[sh["spec"]["p_min_pu"]] = sh["p_min_pu"]
        calibration["storage_heater_gwh"] = sh["spec"]["electricity_gwh"]

    # Demand response: energy-neutral load shifting.
    for key, carrier, alloc, peak_key, row in (("ic_dsr", dc.IC_DSR, "ic_dsr", "ic_dsr_gw", "FLX1 row 110; Srg_BB006a"),
                                               ("residential_dsr", dc.RESIDENTIAL_DSR, "residential_dsr",
                                                "residential_dsr_gw", "FLX1 row 105; Dem_BB008a locations")):
        d = cfg[key]
        if not d.get("enabled", False) or peaks[peak_key] <= 0:
            continue
        p_nom = peaks[peak_key] * 1e3
        flex[carrier] = {"p_nom_mw": p_nom, "energy_mwh": p_nom * float(d["shift_hours"]),
                         "efficiency_store": 1.0, "efficiency_dispatch": 1.0, "standing_loss": 0.0,
                         "marginal_cost": float(d["marginal_cost_gbp_per_mwh"]), "p_min_pu": -1.0, "p_max_pu": 1.0,
                         "inflow": None, "allocation": alloc, "electricity_gwh": 0.0, "source": row}

    # DACCS: annual volume as a constant drain, met per window.
    dcfg = cfg["daccs"]
    if dcfg.get("enabled", False) and energies["daccs_gwh"] > 0:
        # The annual volume over the modelled calendar year's hours (8,784 in a leap year).
        mean_mw = energies["daccs_gwh"] * 1e3 / float(hours_in_year(idx[0].year))
        flex[dc.DACCS] = {"p_nom_mw": mean_mw / float(dcfg["utilisation"]),
                          "energy_mwh": mean_mw * float(dcfg["buffer_hours_of_mean_load"]),
                          "efficiency_store": 1.0, "efficiency_dispatch": 1.0, "standing_loss": 0.0,
                          "marginal_cost": 0.0, "p_min_pu": -1.0, "p_max_pu": 0.0, "inflow": "DACCS_drain",
                          "allocation": "daccs", "electricity_gwh": mean_mw * hours / 1e3,
                          "source": "ES1 row 611 (17.8 TWh HT 2050); declared utilisation and sites"}
        prof["DACCS_drain"] = -mean_mw

    holding_costs(flex, cfg)

    # Unmanaged references for the checks (not network inputs).
    prof["heat_pumps_unmanaged"] = hp_mw
    prof["electric_vehicles_unmanaged"] = res_mw + com_mw
    prof["district_heat_unmanaged"] = dh_mw
    prof["temperature_c"] = temp_c.to_numpy()
    prof["calibration_hour"] = mask.astype(float)
    return {"profiles": prof, "flex": flex, "calibration": calibration, "energies": energies}


# ─────────────────────────────── spatial allocation ───────────────────────────────

ALLOCATION_BLOCKS = {
    "heat_pumps": ["Dem_BB004", "Dem_BB005"],
    "electric_vehicles": ["Dem_BB006", "Dem_BB007"],
    "data_centres": ["Dem_BB008b"],        # declared: FES has no data-centre location block
    "district_heat": ["Dem_BB008a"],       # declared: residential baseline locations
    "v2g": ["Srg_BB005"],
    "ic_dsr": ["Srg_BB006a"],
    "storage_heaters": ["Dem_BB008a"],     # declared: residential baseline locations (no FES block)
    "residential_dsr": ["Dem_BB008a"],
}


def fes_gsp_values(fes: pd.DataFrame, pathway: str, year: int, blocks: Iterable[str]) -> pd.Series:
    """GSP sums of FES building blocks (absolute values, so V2G's negative MW count)."""
    col = next(c for c in ("FES Pathway", "FES Scenario", "﻿FES Scenario") if c in fes.columns)
    rows = fes[fes["Building Block ID Number"].isin(list(blocks))
               & fes[col].astype(str).str.strip().eq(pathway) & fes["GSP"].notna()]
    values = pd.to_numeric(rows[str(year)], errors="coerce").abs()
    out = values.groupby(rows["GSP"].astype(str).str.strip()).sum()
    return out[out > 0]


def etys_allocator(network, fes_year: int, network_model: str = "ETYS", log: Optional[logging.Logger] = None
                   ) -> Callable[[pd.Series, np.ndarray], pd.Series]:
    """GSP -> bus allocation through the base demand path (``load.aggregate_demand_to_network_topology``).

    The same mapping the base loads took (Dem_per_node, isolated-island and transformer
    redistribution), so a component lands where the base put that GSP's demand.
    """
    from scripts.demand.load import aggregate_demand_to_network_topology, load_spatial_mapping_data

    log = log or logger

    def allocate(gsp_values: pd.Series, national_shape: np.ndarray) -> pd.Series:
        shape = np.asarray(national_shape, dtype=float)
        shape = shape / shape.mean()
        gwh = gsp_values.astype(float)
        demand = gwh.to_frame(name="demand")
        ts = pd.DataFrame(np.outer(shape, gwh.to_numpy() * 1e3 / len(shape)), columns=gwh.index)
        mapping = load_spatial_mapping_data(network_model, list(gwh.index), fes_year, log)
        bus_demand, _ = aggregate_demand_to_network_topology(demand, ts, network_model, network, mapping, log,
                                                             is_historical=False, fes_year=fes_year)
        return bus_demand.iloc[:, 0].astype(float)

    return allocate


def nearest_bus_shares(buses: pd.DataFrame, sites: Iterable[dict]) -> pd.Series:
    """Shares on the AC bus nearest each declared site (lat, lon, share)."""
    ac = buses[buses.get("carrier", pd.Series("AC", index=buses.index)).fillna("AC").astype(str).isin(["AC", ""])]
    ac = ac[~ac.index.astype(str).str.contains("External", case=False)]
    ll = bus_lonlat(ac)
    out: Dict[str, float] = {}
    for site in sites:
        d = (ll["lat"] - site["lat"]) ** 2 + ((ll["lon"] - site["lon"]) * np.cos(np.radians(site["lat"]))) ** 2
        bus = str(d.idxmin())
        out[bus] = out.get(bus, 0.0) + float(site["share"])
    s = pd.Series(out)
    return s / s.sum()


def build_allocation(fes: pd.DataFrame, pathway: str, year: int, allocate: Callable, profiles: pd.DataFrame,
                     buses: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Bus x allocation-key shares (each column sums to 1).

    Args:
        fes: FES building-block CSV (``resources/FES/FES_2025_data.csv``).
        pathway, year: FES pathway and year column.
        allocate: ``f(gsp_values, national_shape) -> bus values`` (:func:`etys_allocator`).
        profiles: :func:`build_components` profiles (shapes for the redistribution step).
        buses: Network buses (for DACCS sites).
        cfg: Resolved config.
    """
    shape_of = {"heat_pumps": dc.HEAT_PUMP_LOAD, "electric_vehicles": dc.EV_LOAD,
                "data_centres": dc.DATA_CENTRE_LOAD, "district_heat": "district_heat_unmanaged"}
    cols: Dict[str, pd.Series] = {}
    for key, blocks in ALLOCATION_BLOCKS.items():
        gsp = fes_gsp_values(fes, pathway, year, blocks)
        if gsp.empty:
            raise ValueError(f"No FES GSP values for {key} ({blocks})")
        col = shape_of.get(key)
        shape = profiles[col].to_numpy() if col in profiles else np.ones(len(profiles))
        cols[key] = allocate(gsp, shape)
    cols["daccs"] = nearest_bus_shares(buses, cfg["daccs"]["sites"])
    table = pd.DataFrame(cols).fillna(0.0).clip(lower=0.0)
    table = table.div(table.sum(axis=0), axis=1)
    min_share = float(cfg.get("min_share", 0.0))
    if min_share > 0:
        table = table.where(table >= min_share, 0.0)
        table = table.div(table.sum(axis=0), axis=1)
    table.index.name = "bus"
    return table


# ──────────────────────────────── network step ────────────────────────────────

BASE_LOAD_CARRIERS = ("", "electricity", "AC")


def _base_load_names(n) -> pd.Index:
    loads = n.loads
    carrier = loads.get("carrier", pd.Series("", index=loads.index)).fillna("").astype(str)
    bus_carrier = loads["bus"].map(n.buses["carrier"]).fillna("AC").astype(str)
    mask = carrier.isin(BASE_LOAD_CARRIERS) & bus_carrier.eq("AC") & ~loads["bus"].astype(str).str.contains(
        "External", case=False, na=False)
    return loads.index[mask]


def _dt(n) -> pd.Series:
    return n.snapshot_weightings.generators.astype(float)


def integrate_fes_demand(n, components: Dict[str, Any], allocation: pd.DataFrame, cfg: dict,
                             log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Carve the components out of the base loads and add them, with the flexibility.

    Args:
        n: Network with the base (ED1-scaled ESPENI) loads, modified in place.
        components: :func:`build_components` output (or its CSV/JSON round trip, see
            :func:`load_components`); profiles on ``n.snapshots``.
        allocation: :func:`build_allocation`.
        cfg: Resolved config.
        log: Logger.

    Returns:
        Stats: base, target and residual GWh, clipped energy, loads and units added.
    """
    log = log or logger
    prof = components["profiles"].reindex(n.snapshots)
    if prof.isna().any().any():
        raise ValueError("Component profiles do not cover the network snapshots")
    flex = components["flex"]
    energies = components["energies"]
    dt = _dt(n)
    buses = n.buses.index

    base = _base_load_names(n)
    p_set = n.loads_t.p_set
    base_gwh = p_set[base].mul(dt, axis=0).sum() / 1e3
    base_bus = base_gwh.groupby(n.loads.loc[base, "bus"]).sum()
    t_base = float(base_bus.sum())
    consumer = float(energies["demand_total_gwh"])

    # Electric energy each bus hands to the components (fixed loads + flexible drains).
    comp_bus = pd.Series(0.0, index=allocation.index)
    for load in (dc.HEAT_PUMP_LOAD, dc.EV_LOAD, dc.DISTRICT_HEAT_LOAD, dc.DATA_CENTRE_LOAD):
        comp_bus = comp_bus.add(allocation[load] * float((prof[load] * dt).sum() / 1e3), fill_value=0.0)
    for carrier, spec in flex.items():
        if carrier == dc.DACCS or not spec.get("electricity_gwh"):
            continue
        comp_bus = comp_bus.add(allocation[spec["allocation"]] * float(spec["electricity_gwh"]), fill_value=0.0)
    comp_total = float(comp_bus.sum())
    residual_total = consumer - comp_total
    if residual_total <= 0:
        raise ValueError(f"Components ({comp_total:.0f} GWh) exceed the consumer total ({consumer:.0f} GWh)")

    scaled = base_bus * (consumer / t_base)
    mode = cfg.get("residual_allocation", "per_bus")
    clipped = 0.0
    if mode == "per_bus":
        diff = scaled.sub(comp_bus.reindex(scaled.index).fillna(0.0))
        clipped = float((-diff.clip(upper=0)).sum()) + float(comp_bus.drop(scaled.index, errors="ignore").sum())
        target = diff.clip(lower=0)
        target = target / target.sum() * residual_total
    elif mode == "flat":
        target = base_bus * (residual_total / t_base)
    else:
        raise ValueError("residual_allocation must be 'per_bus' or 'flat'")
    ratio = target.reindex(base_bus.index).fillna(0.0).div(base_bus.where(base_bus > 0))
    factor = ratio.reindex(n.loads.loc[base, "bus"]).fillna(0.0).to_numpy()
    p_set.loc[:, base] = p_set[base].to_numpy() * factor[None, :]
    log.info(f"Base loads {t_base:,.0f} GWh -> residual {residual_total:,.0f} GWh ({mode}); consumer total "
             f"{consumer:,.0f} GWh; components {comp_total:,.0f} GWh; clipped {clipped:,.1f} GWh")

    # Fixed component loads, carrier = component name.
    added_loads = 0
    new_loads: Dict[str, pd.Series] = {}
    load_rows = []
    for load in (dc.HEAT_PUMP_LOAD, dc.EV_LOAD, dc.DISTRICT_HEAT_LOAD, dc.DATA_CENTRE_LOAD):
        shares = allocation[load]
        shares = shares[(shares > 0) & shares.index.isin(buses)]
        national = prof[load].to_numpy()
        for bus, share in shares.items():
            name = f"{load}_{bus}"
            load_rows.append((name, bus, load))
            new_loads[name] = pd.Series(national * share, index=n.snapshots)
        added_loads += len(shares)
    if load_rows:
        names = [r[0] for r in load_rows]
        n.add("Load", names, bus=[r[1] for r in load_rows], carrier=[r[2] for r in load_rows],
              p_set=pd.DataFrame(new_loads, index=n.snapshots)[names])

    # Flexibility as storage units.
    added_units = 0
    for carrier, spec in flex.items():
        shares = allocation[spec["allocation"]]
        shares = shares[(shares > 0) & shares.index.isin(buses)]
        if shares.empty:
            continue
        names = [f"{bus} {carrier}" for bus in shares.index]
        p_nom = spec["p_nom_mw"] * shares.to_numpy()
        kwargs: Dict[str, Any] = dict(
            bus=list(shares.index), carrier=carrier, p_nom=p_nom, p_nom_extendable=False,
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
                kwargs[attr] = pd.DataFrame(np.repeat(series[:, None], len(names), axis=1),
                                            index=n.snapshots, columns=names)
            else:
                kwargs[attr] = float(value)
        if spec.get("inflow"):
            national = prof[spec["inflow"]].to_numpy()
            kwargs["inflow"] = pd.DataFrame(national[:, None] * shares.to_numpy()[None, :],
                                            index=n.snapshots, columns=names)
        n.add("StorageUnit", names, **kwargs)
        added_units += len(names)

    for carrier in (*dc.DEMAND_COMPONENT_LOAD_CARRIERS[:4], *flex.keys()):
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier, co2_emissions=0.0)

    stats = {"base_gwh": t_base, "demand_total_gwh": consumer, "demand_basis": energies.get("demand_basis"),
             "loss_factor": energies.get("loss_factor", 1.0), "components_gwh": comp_total,
             "residual_gwh": residual_total, "residual_allocation": mode, "clipped_gwh": clipped,
             "fixed_loads_added": added_loads, "storage_units_added": added_units}
    log.info(f"FES demand: {added_loads} component loads, {added_units} consumer flexibility units")
    return stats


# ──────────────────────────────────── checks ────────────────────────────────────

def network_demand_breakdown(n) -> pd.Series:
    """Electricity demand (GWh) by line on a built network, before any solve.

    Fixed loads count at their ``p_set``; drained storage units count at the electricity
    their drain needs (``-inflow / efficiency_store``); demand response is energy neutral.
    """
    dt = _dt(n)
    out: Dict[str, float] = {}
    loads = n.loads
    bus_carrier = loads["bus"].map(n.buses["carrier"]).fillna("AC").astype(str)
    ac = bus_carrier.eq("AC") & ~loads["bus"].astype(str).str.contains("External", case=False, na=False)
    p_set = n.loads_t.p_set
    for name in loads.index[ac]:
        carrier = str(loads.at[name, "carrier"] or "")
        line = "residual" if carrier in BASE_LOAD_CARRIERS else dc.DEMAND_LINE_OF_CARRIER.get(carrier, carrier)
        e = float((p_set[name] * dt).sum()) if name in p_set.columns else float(loads.at[name, "p_set"] * dt.sum())
        out[line] = out.get(line, 0.0) + e / 1e3
    su = n.storage_units
    inflow = n.storage_units_t.inflow
    for name in su.index[su["carrier"].isin(dc.CONSUMER_FLEX_CARRIERS + dc.NON_ELECTRICITY_ASSET_CARRIERS)]:
        if name not in inflow.columns:
            continue
        drain = float((-inflow[name] * dt).sum()) / float(su.at[name, "efficiency_store"])
        line = dc.DEMAND_LINE_OF_CARRIER.get(str(su.at[name, "carrier"]), str(su.at[name, "carrier"]))
        out[line] = out.get(line, 0.0) + drain / 1e3
    return pd.Series(out, name="gwh")


def demand_check(n, energies: Dict[str, Any], peaks: Dict[str, float], calibration: Optional[dict] = None,
                 rel_tol: float = 0.005) -> pd.DataFrame:
    """Acceptance table: component energy against its FES line, and the peak against ED1.

    Args:
        n: Built network (after :func:`integrate_fes_demand`).
        energies: :func:`component_energies`.
        peaks: :func:`peak_targets`.
        calibration: :func:`build_components` calibration record (optional, reported).
        rel_tol: Energy tolerance (the validation's demand tolerance).

    Returns:
        One row per check: item, target, model, relative difference, status, note.
    """
    b = network_demand_breakdown(n)
    rows = []

    def _row(item, target, model, tol, note=""):
        rel = (model - target) / target if target else np.nan
        status = "INFO" if tol is None else ("PASS" if abs(rel) <= tol else "WARN")
        rows.append({"item": item, "target": target, "model": model, "relative_difference": rel,
                     "status": status, "note": note})

    total = float(b.drop(labels=["daccs"], errors="ignore").sum())
    _row(f"demand total, {energies.get('demand_basis', 'consumer')} basis (GWh)", energies["demand_total_gwh"], total,
         rel_tol, energies["sources"]["demand_total"])
    for line, key in (("heat_pumps", "heat_pumps_gwh"), ("evs", "evs_gwh"), ("district_heat", "district_heat_gwh"),
                      ("data_centres", "data_centres_gwh"), ("residual", "residual_gwh"), ("daccs", "daccs_gwh")):
        _row(f"{line} (GWh)", energies[key], float(b.get(line, 0.0)), rel_tol, energies["sources"].get(line, ""))
    _row("demand response (GWh, energy neutral)", 0.0, float(b.get("demand_response", 0.0)), None)

    dt = _dt(n)
    loads = n.loads
    ac = loads["bus"].map(n.buses["carrier"]).fillna("AC").astype(str).eq("AC") & \
        ~loads["bus"].astype(str).str.contains("External", case=False, na=False)
    fixed = n.loads_t.p_set[[c for c in loads.index[ac] if c in n.loads_t.p_set.columns]].sum(axis=1)
    su = n.storage_units
    drains = pd.Series(0.0, index=n.snapshots)
    for name in su.index[su["carrier"].isin([dc.HEAT_PUMP_FLEXIBILITY, dc.DISTRICT_HEAT_FLEXIBILITY])]:
        if name in n.storage_units_t.inflow.columns:
            drains = drains - n.storage_units_t.inflow[name] / float(su.at[name, "efficiency_store"])
    basis = "incl. losses" if peaks.get("loss_factor", 1.0) > 1.0 else "consumer basis, losses excluded"
    target = peaks.get("ed1_peak_model_basis_gw", peaks["ed1_peak_consumer_gw"])
    note = ("ED1 row 134 (ACS peak incl. losses)" if peaks.get("loss_factor", 1.0) > 1.0
            else "ED1 row 134 less losses at peak (rows 179, 209, 249)")
    _row(f"peak, inflexible loads (GW, {basis})", target, float(fixed.max() / 1e3), None,
         note + "; flexible parts excluded; the ED1 peak is after FES flexibility")
    _row(f"peak, inflexible loads + heat drains at their unmanaged time (GW, {basis})", target,
         float((fixed + drains).max() / 1e3), None, "EV flexible pool, V2G and DSR excluded")
    if calibration:
        for key in ("ev_fixed_at_peak_gw", "ev_unmanaged_at_peak_gw", "ev_smart_impact_at_peak_gw",
                    "hp_unmanaged_at_peak_gw", "hp_fixed_at_peak_gw", "dh_unmanaged_at_peak_gw",
                    "ev_residential_flex_share", "hp_flex_share", "dh_flex_share", "hp_effective_cop"):
            if key in calibration:
                _row(key, np.nan, float(calibration[key]), None, "calibration record")
    return pd.DataFrame(rows)


# ─────────────────────────────── file round trip ───────────────────────────────

def save_components(components: Dict[str, Any], allocation: pd.DataFrame, profiles_path: str, allocation_path: str,
                    spec_path: str) -> None:
    """Write the rule outputs: profiles CSV, allocation CSV, spec JSON (flex, energies, calibration)."""
    for p in (profiles_path, allocation_path, spec_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    components["profiles"].to_csv(profiles_path)
    allocation.to_csv(allocation_path)
    spec = {k: components[k] for k in ("flex", "calibration", "energies", "peaks", "config") if k in components}
    Path(spec_path).write_text(json.dumps(spec, indent=1, default=float), encoding="utf-8")


def load_components(profiles_path: str, allocation_path: str, spec_path: str) -> tuple:
    """Read :func:`save_components` outputs back: (components dict, allocation)."""
    profiles = pd.read_csv(profiles_path, index_col=0, parse_dates=True)
    allocation = pd.read_csv(allocation_path, index_col=0)
    allocation.index = allocation.index.astype(str)
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    return {"profiles": profiles, **spec}, allocation


# ──────────────────────────────── Snakemake rule ────────────────────────────────

def _ev_fleet(fes: pd.DataFrame, pathway: str, year: int) -> float:
    col = next(c for c in ("FES Pathway", "FES Scenario") if c in fes.columns)
    rows = fes[fes["Building Block ID Number"].eq("Lct_BB001") & fes[col].astype(str).str.strip().eq(pathway)]
    return float(pd.to_numeric(rows[str(year)], errors="coerce").sum())


def run_rule(smk) -> None:
    """Script body of ``build_fes_demand_components``."""
    import xarray as xr
    from scripts.FES.fes2025_targets import read_flx1
    from scripts.utilities.logging_config import setup_logging
    from scripts.utilities.network_io import load_network

    log = setup_logging(smk.log[0], "INFO")
    cfg = resolve_config(smk.params.fes_demand)
    pathway = str(smk.params.fes_scenario)
    year = int(smk.params.modelled_year)
    targets = json.loads(Path(smk.input.targets).read_text(encoding="utf-8"))
    flex = flx1_flexibility(read_flx1(smk.input.workbook), pathway, year)
    energies = component_energies(targets, cfg["year_basis"], cfg["demand_basis"])
    peaks = peak_targets(targets, flex, energies["loss_factor"])
    n = load_network(smk.input.base_network, skip_time_series=False, custom_logger=log)
    base = _base_load_names(n)
    weights = (n.loads_t.p_set[base].sum() if len(n.loads_t.p_set) else n.loads.loc[base, "p_set"])
    by_bus = weights.groupby(n.loads.loc[base, "bus"]).sum()
    ll = bus_lonlat(n.buses.loc[by_bus.index])
    with xr.open_dataset(smk.input.cutout) as ds:
        temp = demand_weighted_temperature(ds["temperature"], ll["lon"].to_numpy(), ll["lat"].to_numpy(),
                                           by_bus.to_numpy())
    temp_c = map_weather_year(temp, n.snapshots)
    fes = pd.read_csv(smk.input.fes_data, low_memory=False)
    comps = build_components(n.snapshots, temp_c, energies, peaks, _ev_fleet(fes, pathway, year), cfg)
    allocate = etys_allocator(n, int(smk.params.fes_year), str(smk.params.network_model), log)
    allocation = build_allocation(fes, pathway, year, allocate, comps["profiles"], n.buses, cfg)
    comps.update(peaks=peaks, config=cfg, flx1=flex)
    comps["energies"]["flx1_rows"] = {k: v["excel_row"] for k, v in flex.items()}
    save_components(comps, allocation, smk.output.profiles, smk.output.allocation, smk.output.spec)
    log.info("Calibration: %s", json.dumps(comps["calibration"], default=float))
    log.info("Energies: %s", json.dumps({k: v for k, v in energies.items() if k != "sources"}, default=float))


if __name__ == "__main__" and "snakemake" in globals():
    run_rule(snakemake)  # noqa: F821
