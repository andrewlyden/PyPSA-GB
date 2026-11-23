"""CP30 2030 demand: FES 2024 spatial basis, named large loads and a component-built shape.

The demand audit of 15 Oct 2026 (CP30 2030 rebuild, fixes 1, 2 and 4; fix 3 is the
capacity redirect in ``load.py``) found the CP30 loads:

1. spread on FES **2025** Holistic Transition GSP shares, whose northern DNO bases are
   well above FES 2024's (the vintage CP30 was built on) and DESNZ meter data, so 39.4%
   of demand sat north of B8 against about 34-35%;
2. scaled 1.22x pro rata, so the 18% of demand that is not GSP demand (transmission
   data centres, direct transmission customers, losses) followed the GSP pattern instead
   of sitting at the named sites FES places it (1.54 GW of data centres, 92% in the
   South East and East; Port Talbot; St Fergus; HS2 feeders);
3. one ESPENI 2013 shape on every bus: no EV, heat pump or data-centre shape.

This module builds, for scenarios with ``demand.build.enabled: true``:

* **Energy by GSP and by named site** (:func:`build_energy_table`). The consumer total is
  CP30 ED1 (residential + I&C + road transport, 287.2 TWh); ED1's system total (315.16
  TWh, excluding electrolysis) carries the losses, spread pro rata on every load through
  one loss factor (system / consumer). Named loads take FES 2024 ED1's national lines
  (transmission data centres 13,185 GWh, direct transmission customers 7,913 GWh), split
  over the sites of ``resources/demand/cp30_2030_named_large_loads.csv`` by their ETYS
  2030/31 peak. Only the residual is scaled onto the FES 2024 Dem_BB003 GSP shares. The
  named-loads table and the FES 2024 lines are made from the workbooks by rule
  ``extract_cp30_demand_inputs`` (``scripts/demand/extract_cp30_demand_inputs.py``);
  :func:`named_load_reference` gives the CP30 locational validation the same placement.
  FES 2024's "Direct(SPTL)"/"Direct(SHETL)" rows are spread within their own TO area,
  not across GB.
* **Components inside each GSP's energy** (:func:`allocate_components`): EVs (CP30 ED1
  road transport, at FES 2024 Dem_BB006+BB007 shares), heat pumps (ED1 residential +
  commercial, Dem_BB004+BB005), district heat (ED1, Dem_BB008a, as the 2050 build),
  distribution-connected data centres (FES 2024 ED1, Dem_BB008b). The base (ESPENI) part
  of each GSP is what is left.
* **Shapes** (:func:`national_shapes`, reusing ``fes_demand``'s profile functions):
  heat pumps and district heat on the weather year's demand-weighted temperature, EVs on
  plug-in charging profiles with the smart-charged share moved to a fixed overnight
  window (demand flexibility stays off) and solved so EV load at the ACS-peak analogue
  is FES 2024's "EVs at peak"; data centres and named loads flat.
* **Peak calibration** (:func:`calibrate_base_exponent`): the base ESPENI shape is raised
  to a power alpha and renormalised (energy exact) so the GB peak equals the target.
  ``peak_target: es1`` is CP30 ES1 "Demand (MW) - Excluding Electrolysis" (56,284.67
  MW, CP30's own dispatch-model peak, the same table and basis as the 315.16 TWh);
  ``ed1`` is ED1 "GBFES Peak Customer Demand: Total Consumption plus Losses" (62,357
  MW, FES's average-cold-spell peak); a number is MW.
"""

from __future__ import annotations

import copy

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.demand import fes_demand as f50

logger = logging.getLogger(__name__)

# Declared defaults; demand.build in config/defaults/method/brownfield.yaml repeats them.
DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "spatial": {
        "FES_year": 2024,
        "FES_scenario": "Holistic Transition",
        # FES GSP rows that are transmission-connected demand inside Dem_BB003: spread
        # pro rata over the GSPs of the named DNO licence area (same TO), not over GB.
        "direct_rows": {"Direct(SPTL)": "SPEN", "Direct(SHETL)": "SSEN (SHEPD)"},
    },
    "named_loads": {
        "enabled": True,
        # Output of rule extract_cp30_demand_inputs (a declared build_base_demand input).
        "file": "resources/demand/cp30_2030_named_large_loads.csv",
        # National energy (FES 2024 ED1 line) of each named-load group; sites share their
        # group's energy pro rata to their ETYS 2030/31 peak MW. Shape: flat (ETYS gives
        # the same MW at winter peak and both summer minima).
        "group_lines": {"data_centre": "data_centres_transmission_gwh",
                        "other": "direct_transmission_customers_gwh"},
    },
    # Output of rule extract_cp30_demand_inputs (FES 2024 ED1 HT 2030 lines).
    "national_lines_file": "resources/demand/fes24_ht_2030_demand_lines.csv",
    # ED1 pathway whose component rows (HP, EV, DH, consumer totals) are used when the
    # scenario's pathway has none (CP30 New Dispatch publishes only the total).
    "ed1_component_pathway": "Further Flex and Renewables",
    # Fix 3: excess at capped <=132 kV buses goes to the same-site supergrid bus.
    "capacity_redistribution": "same_site",
    "capacity_safety_margin": 0.90,
    "min_supergrid_kv": 275.0,
    "shape": {
        "enabled": True,
        "peak_target": "es1",
        "components": ["electric_vehicles", "heat_pumps", "district_heat", "data_centres"],
        # "fes_at_peak": smart share solved so EV load at the calibration hours is FES 2024
        # ED1 residential + commercial "EVs at peak" (times the loss factor); or a number.
        "ev_smart_share": "fes_at_peak",
        "ev_smart_max_share": 0.95,
        # Smart-charged energy as a fixed overnight window (flexibility off): circular
        # Gaussian (weight, mean hour, sd hours); declared, Octopus Go/Intelligent-style.
        "ev_smart_window": [[1.0, 2.5, 2.0]],
        "exponent_bounds": [0.2, 4.0],
        # Overrides of fes_demand.DEFAULT_CONFIG profile blocks (heat_pumps,
        # district_heat, electric_vehicles, temperature, calibration).
        "profiles": {},
    },
}

COMPONENT_BLOCKS = {
    "electric_vehicles": ["Dem_BB006", "Dem_BB007"],
    "heat_pumps": ["Dem_BB004", "Dem_BB005"],
    "district_heat": ["Dem_BB008a"],
    "data_centres": ["Dem_BB008b"],
}

NAMED_PREFIX = "NAMED::"


def resolve_config(cfg: Optional[dict]) -> dict:
    """``demand.build`` merged over :data:`DEFAULT_CONFIG` (nested dicts merged; a deep copy, so a
    caller editing the result cannot change the defaults)."""
    def _merge(base: dict, over: dict) -> dict:
        out = dict(base)
        for k, v in (over or {}).items():
            out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        return out
    return copy.deepcopy(_merge(DEFAULT_CONFIG, cfg or {}))


def is_enabled(cfg: Optional[dict]) -> bool:
    return bool((cfg or {}).get("enabled", False))


# ─────────────────────────────── national inputs ───────────────────────────────

def _norm(value) -> str:
    return " ".join(str(value).strip().lower().split())


def read_national_lines(path: str | Path) -> Dict[str, float]:
    lines = pd.read_csv(path)
    return {str(k): float(v) for k, v in zip(lines["key"], lines["value"])}


def _ed1_sum(ed1: pd.DataFrame, pathway: str, items: Iterable[str], year: int) -> Optional[float]:
    names = {_norm(i) for i in items}
    rows = ed1[(ed1["Pathway"].map(_norm) == _norm(pathway)) & (ed1["Data item"].map(_norm).isin(names))]
    if rows.empty:
        return None
    return float(pd.to_numeric(rows[str(year)], errors="coerce").sum())


ED1_COMPONENT_ITEMS = {
    "evs": ["Total Road Transport"],
    "heat_pumps": ["Residential Heat Pump Flex", "Residential Heat Pump NonFlex",
                   "Commercial Heat Pump Flex", "Commercial Heat Pump NonFlex"],
    "district_heat": ["Residential District Heat Flex", "Residential District Heat NonFlex",
                      "Commercial District Heat Flex", "Commercial District Heat NonFlex"],
    "consumer_total": ["Total Residential", "Total Industrial and Commercial", "Total Road Transport"],
}


def cp30_component_energies(ed1: pd.DataFrame, pathway: str, year: int, fallback_pathway: str) -> Dict[str, Any]:
    """CP30 ED1 consumer-level energies (GWh) of the components and the consumer total.

    A pathway without component rows (New Dispatch) takes them from ``fallback_pathway``.
    """
    out: Dict[str, Any] = {}
    for key, items in ED1_COMPONENT_ITEMS.items():
        value = _ed1_sum(ed1, pathway, items, year)
        source = pathway
        if value is None:
            value = _ed1_sum(ed1, fallback_pathway, items, year)
            source = fallback_pathway
        if value is None or value <= 0:
            raise ValueError(f"CP30 ED1 has no {key} rows for {pathway} or {fallback_pathway} {year}")
        out[f"{key}_gwh"] = value
        out.setdefault("sources", {})[key] = f"CP30 ED1 {source}: " + " + ".join(items)
    return out


def peak_target_mw(target, pathway: str, year: int, es1: Optional[pd.DataFrame],
                   ed1: Optional[pd.DataFrame]) -> Tuple[float, str]:
    """Resolve ``shape.peak_target`` to MW and a source label."""
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        return float(target), "configured MW"
    key = str(target).strip().lower()
    if key == "es1":
        rows = es1[(es1["Pathway"].map(_norm) == _norm(pathway))
                   & (es1["Type"].map(_norm) == _norm("Demand (MW) - Excluding Electrolysis"))]
        value = pd.to_numeric(rows[str(year)], errors="coerce").dropna()
        if len(value) != 1:
            raise ValueError(f"CP30 ES1 'Demand (MW) - Excluding Electrolysis' for {pathway}: {len(value)} rows")
        return float(value.iloc[0]), f"CP30 ES1 {pathway} 'Demand (MW) - Excluding Electrolysis'"
    if key == "ed1":
        rows = ed1[(ed1["Pathway"].map(_norm) == _norm(pathway))
                   & (ed1["Data item"].map(_norm) == _norm("GBFES Peak Customer Demand: Total Consumption plus Losses"))]
        value = pd.to_numeric(rows[str(year)], errors="coerce").dropna()
        if len(value) != 1:
            raise ValueError(f"CP30 ED1 peak for {pathway}: {len(value)} rows")
        return float(value.iloc[0]) * 1e3, f"CP30 ED1 {pathway} 'GBFES Peak Customer Demand: Total Consumption plus Losses'"
    raise ValueError(f"peak_target must be 'es1', 'ed1' or a number of MW, not {target!r}")


# ─────────────────────────────── spatial energy ───────────────────────────────

def _scenario_column(fes: pd.DataFrame) -> str:
    for col in ("FES Pathway", "FES Scenario", "﻿FES Scenario"):
        if col in fes.columns:
            return col
    raise ValueError("FES data has no pathway/scenario column")


def fes_gsp_blocks(fes: pd.DataFrame, pathway: str, year: int, blocks: Iterable[str],
                   direct_rows: Optional[Dict[str, str]] = None) -> pd.Series:
    """GSP sums of FES building blocks, with the "Direct(TO)" rows spread within their area.

    A Direct row's value is added to the GSPs of its DNO licence area pro rata to their own
    value of the same blocks; a Direct row with no mapping is left as it is (the GSP split
    spreads an unlocated row pro rata over GB and logs it).
    """
    col = _scenario_column(fes)
    rows = fes[fes["Building Block ID Number"].isin(list(blocks))
               & (fes[col].map(_norm) == _norm(pathway)) & fes["GSP"].notna()].copy()
    rows["value"] = pd.to_numeric(rows[str(year)], errors="coerce").fillna(0.0)
    rows["GSP"] = rows["GSP"].astype(str).str.strip()
    by_gsp = rows.groupby("GSP")["value"].sum()
    area_of = rows.groupby("GSP")["DNO License Area"].first() if "DNO License Area" in rows else pd.Series(dtype=str)
    for direct, area in (direct_rows or {}).items():
        if direct not in by_gsp.index:
            continue
        value = float(by_gsp.pop(direct))
        members = [g for g in area_of.index[area_of.astype(str) == area] if g in by_gsp.index and by_gsp[g] > 0]
        if not members:
            raise ValueError(f"FES row {direct}: no GSPs in DNO area {area!r} to spread {value:.1f} over")
        weights = by_gsp[members] / by_gsp[members].sum()
        by_gsp.loc[members] += value * weights
    return by_gsp[by_gsp > 0]


def read_named_loads(path: str | Path) -> pd.DataFrame:
    named = pd.read_csv(path)
    for col in ("etys_id", "category", "peak_mw_ht_2030_31", "nodes"):
        if col not in named.columns:
            raise ValueError(f"{path}: missing column {col}")
    named = named[named["peak_mw_ht_2030_31"] > 0].copy()
    if named["etys_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated etys_id")
    return named.set_index("etys_id")


def named_load_energies(named: pd.DataFrame, lines: Dict[str, float], group_lines: Dict[str, str]) -> pd.Series:
    """Consumer-level GWh of every named site: its group's FES line pro rata to peak MW."""
    group = np.where(named["category"].eq("data_centre"), "data_centre", "other")
    out = pd.Series(0.0, index=named.index)
    for g in ("data_centre", "other"):
        mask = group == g
        if not mask.any():
            continue
        total = float(lines[group_lines[g]])
        mw = named.loc[mask, "peak_mw_ht_2030_31"].astype(float)
        out[mask] = total * mw / mw.sum()
    return out


def loss_factor(system_total_gwh: float, consumer_total_gwh: float) -> float:
    """The one loss factor spread pro rata on every load: system / consumer energy."""
    return float(system_total_gwh) / float(consumer_total_gwh)


def named_system_energies(named: pd.DataFrame, lines: Dict[str, float], group_lines: Dict[str, str],
                          lf: float) -> pd.Series:
    """System-level GWh of every named site (consumer GWh times the loss factor), by etys_id."""
    return named_load_energies(named, lines, group_lines) * lf


def named_load_reference(cfg: Optional[dict], pathway: str, year: int, ed1_path: str | Path,
                         es1_path: Optional[str | Path] = None,
                         log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """The transmission-connected named loads as this build places them, one row per (site, node).

    The same files, energy split and loss factor as :func:`prepare`: each site's FES line
    energy pro rata to its ETYS peak (:func:`named_load_energies`), times system / consumer
    (:func:`loss_factor`; system = CP30 ED1 demand excluding electrolysis, consumer = CP30
    ED1 residential + I&C + road transport), split over its nodes by their shares. A site
    with no node is one row with ``node`` "". ``share_of_system_demand`` is the row's GWh
    over the system total. Empty when ``named_loads.enabled`` is false. ``attrs`` carries
    the totals and the loss factor.

    Read by ``scripts/CP30/locational_validation.py`` (the demand check's reference), so the
    check and the build cannot drift apart.
    """
    from scripts.demand.load import get_cp30_electricity_demand_gwh

    cfg = resolve_config(cfg)
    cols = ["etys_id", "name", "category", "node", "node_share", "gwh", "share_of_system_demand",
            "latitude", "longitude", "source"]
    if not cfg["named_loads"].get("enabled", True):
        return pd.DataFrame(columns=cols)
    named = read_named_loads(cfg["named_loads"]["file"])
    lines = read_national_lines(cfg["national_lines_file"])
    consumer = cp30_component_energies(pd.read_csv(ed1_path), pathway, year,
                                       cfg["ed1_component_pathway"])["consumer_total_gwh"]
    system = get_cp30_electricity_demand_gwh(pathway, year, ed1_path, log, cp30_es1_path=es1_path)
    lf = loss_factor(system, consumer)
    gwh = named_system_energies(named, lines, cfg["named_loads"]["group_lines"], lf)
    source = (f"{cfg['named_loads']['file']} (FES24 ETYS HT 2030/31 type T sites and nodes); energy = "
              f"{cfg['national_lines_file']} group lines {cfg['named_loads']['group_lines']} split by site peak, "
              f"x loss factor {lf:.4f} (CP30 ED1 {system:,.2f} / {consumer:,.2f} GWh): demand.build.named_load_reference")
    rows = []
    for etys_id, r in named.iterrows():
        for node, share in parse_node_shares(r["nodes"]) or [("", 1.0)]:
            rows.append({"etys_id": etys_id, "name": r.get("name", etys_id), "category": r["category"],
                         "node": node, "node_share": share, "gwh": float(gwh[etys_id]) * share,
                         "share_of_system_demand": float(gwh[etys_id]) * share / system,
                         "latitude": r.get("latitude"), "longitude": r.get("longitude"), "source": source})
    out = pd.DataFrame(rows, columns=cols)
    out.attrs.update({"system_total_gwh": system, "consumer_total_gwh": consumer, "loss_factor": lf})
    return out


def build_energy_table(fes: pd.DataFrame, cfg: dict, year: int, system_total_gwh: float,
                       consumer_total_gwh: float, named: Optional[pd.DataFrame],
                       lines: Dict[str, float], log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """System-basis annual GWh by GSP name and by named site (``NAMED::<id>``).

    Named sites carry their FES line energy times the loss factor; the GSPs share the rest
    on FES Dem_BB003 (Direct rows within their TO). The two sum to ``system_total_gwh``.
    """
    log = log or logger
    sp = cfg["spatial"]
    lf = loss_factor(system_total_gwh, consumer_total_gwh)
    bb003 = fes_gsp_blocks(fes, sp["FES_scenario"], year, ["Dem_BB003"], sp.get("direct_rows"))
    named_sys = pd.Series(dtype=float)
    if named is not None and len(named):
        named_sys = named_system_energies(named, lines, cfg["named_loads"]["group_lines"], lf)
        named_sys.index = [NAMED_PREFIX + str(i) for i in named_sys.index]
    gsp_total = system_total_gwh - float(named_sys.sum())
    if gsp_total <= 0:
        raise ValueError("Named loads exceed the system total")
    gsp_sys = bb003 / bb003.sum() * gsp_total
    accounting = {
        "system_total_gwh": system_total_gwh, "consumer_total_gwh": consumer_total_gwh,
        "loss_factor": lf, "fes_bb003_gwh": float(bb003.sum()),
        "named_loads_gwh": float(named_sys.sum()), "named_sites": int(len(named_sys)),
        "gsp_total_gwh": gsp_total, "gsp_scale_on_bb003": gsp_total / float(bb003.sum()),
    }
    log.info("CP30 demand energy table: system %.1f GWh = GSP %.1f (FES %s BB003 %.1f x %.4f) + named %.1f "
             "(%d sites); loss factor %.5f", system_total_gwh, gsp_total, sp["FES_year"], bb003.sum(),
             accounting["gsp_scale_on_bb003"], named_sys.sum(), len(named_sys), lf)
    return {"gsp_gwh": gsp_sys, "named_gwh": named_sys, "accounting": accounting}


def allocate_components(fes: pd.DataFrame, cfg: dict, year: int, gsp_gwh: pd.Series,
                        energies_gwh: Dict[str, float], log: Optional[logging.Logger] = None,
                        max_share_of_gsp: float = 0.95) -> Tuple[pd.DataFrame, float]:
    """GSP x component GWh (system basis), each component at its FES block shares.

    A GSP whose components would exceed ``max_share_of_gsp`` of its energy has them scaled
    down and the excess spread over the other GSPs pro rata (component totals exact).

    Returns:
        (table indexed by GSP, columns = components; GWh moved by the cap).
    """
    log = log or logger
    sp = cfg["spatial"]
    table = pd.DataFrame(index=gsp_gwh.index)
    for comp, gwh in energies_gwh.items():
        values = fes_gsp_blocks(fes, sp["FES_scenario"], year, COMPONENT_BLOCKS[comp], sp.get("direct_rows"))
        values = values.reindex(gsp_gwh.index).fillna(0.0)
        if values.sum() <= 0:
            raise ValueError(f"No FES GSP values for component {comp}")
        table[comp] = values / values.sum() * gwh
    moved = 0.0
    for _ in range(50):
        cap = gsp_gwh * max_share_of_gsp
        over = table.sum(axis=1) - cap
        hit = over > 1e-9
        if not hit.any():
            break
        factor = (cap[hit] / table.loc[hit].sum(axis=1))
        excess = table.loc[hit].mul(1 - factor, axis=0).sum()
        moved += float(excess.sum())
        table.loc[hit] = table.loc[hit].mul(factor, axis=0)
        free = ~hit & (table.sum(axis=1) < cap)
        for comp in table.columns:
            w = table.loc[free, comp]
            w = w / w.sum() if w.sum() > 0 else gsp_gwh[free] / gsp_gwh[free].sum()
            table.loc[free, comp] += excess[comp] * w
    if moved > 0:
        log.warning("Components capped at %.0f%% of their GSP's energy: %.1f GWh moved to other GSPs",
                    100 * max_share_of_gsp, moved)
    return table, moved


# ─────────────────────────────────── shapes ───────────────────────────────────

def _unit(shape: np.ndarray, dt: float) -> np.ndarray:
    """Scale a non-negative shape so sum(shape * dt) = 1 (MW per MWh of annual energy)."""
    shape = np.asarray(shape, dtype=float)
    return shape / (shape.sum() * dt)


def national_shapes(snapshots: pd.DatetimeIndex, temp_c: Optional[pd.Series], cfg: dict,
                    lines: Dict[str, float], ev_gwh: float, loss_factor: float,
                    ev_residential_share: float, ev_depot_share: float) -> Dict[str, Any]:
    """Unit shapes (sum x dt = 1) of each component and the EV smart-share calibration.

    ``ev_gwh`` is consumer-level (at the meter), like FES's "EVs at peak"; the MW records
    are consumer-level too (``loss_factor`` is kept for the caller's records).
    """
    idx = pd.DatetimeIndex(snapshots)
    dt = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    pcfg = f50.resolve_config(cfg["shape"].get("profiles") or {})
    comps = cfg["shape"]["components"]
    shapes: Dict[str, np.ndarray] = {}
    record: Dict[str, Any] = {}
    mask = None
    if temp_c is not None:
        mask = f50.calibration_mask(temp_c, pcfg).to_numpy()
        record["calibration_hours"] = int(mask.sum())
        record["calibration_mean_temperature_c"] = float(temp_c.to_numpy()[mask].mean())
    if "heat_pumps" in comps or "district_heat" in comps:
        if temp_c is None:
            raise ValueError("Heat pump and district heat shapes need the weather-year temperature")
    if "heat_pumps" in comps:
        hp = f50.heat_pump_profiles(temp_c, pcfg)
        shapes["heat_pumps"] = _unit(hp["electric"].to_numpy(), dt)
        record["hp_effective_cop"] = float((hp["space_heat_th"] + hp["hot_water_th"]).sum() / hp["electric"].sum())
    if "district_heat" in comps:
        shapes["district_heat"] = _unit(f50.district_heat_profile(temp_c, pcfg)["electric"].to_numpy(), dt)
    if "data_centres" in comps:
        shapes["data_centres"] = _unit(np.ones(len(idx)), dt)
    if "electric_vehicles" in comps:
        evp = f50.ev_profiles(idx, pcfg)
        res = _unit(evp["residential"].to_numpy(), dt)
        com_gwh_share = 1.0 - ev_residential_share
        depot = min(ev_depot_share, com_gwh_share)
        public = com_gwh_share - depot
        com = public * _unit(evp["public"].to_numpy(), dt) + depot * _unit(evp["depot"].to_numpy(), dt)
        hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
        smart = _unit(f50.gaussian_day(hours, cfg["shape"]["ev_smart_window"]), dt)
        setting = cfg["shape"].get("ev_smart_share", "fes_at_peak")
        if isinstance(setting, (int, float)) and not isinstance(setting, bool):
            s = float(setting)
        else:
            if mask is None:
                raise ValueError("ev_smart_share 'fes_at_peak' needs the weather-year temperature")
            # Consumer basis on both sides (ev_gwh and FES's at-peak figures are at the meter);
            # shape() reports the MW records times the loss factor.
            mw = ev_gwh * 1e3
            target = (lines["residential_evs_at_peak_gw"] + lines["commercial_evs_at_peak_gw"]) * 1e3
            res_at = ev_residential_share * mw * res[mask].mean()
            com_at = mw * com[mask].mean()
            smart_at = ev_residential_share * mw * smart[mask].mean()
            s = (res_at + com_at - target) / max(res_at - smart_at, 1e-9)
            record["ev_target_at_peak_mw"] = target
            record["ev_unmanaged_at_peak_mw"] = res_at + com_at
        s = float(np.clip(s, 0.0, float(cfg["shape"]["ev_smart_max_share"])))
        shape = ev_residential_share * ((1 - s) * res + s * smart) + com
        shapes["electric_vehicles"] = shape / (shape.sum() * dt)
        record["ev_smart_share_of_residential"] = s
        if mask is not None:
            mw = ev_gwh * 1e3
            record["ev_at_peak_mw"] = float(mw * shapes["electric_vehicles"][mask].mean())
            record["ev_smart_impact_at_peak_mw"] = float(
                ev_residential_share * mw * s * (res[mask].mean() - smart[mask].mean()))
    return {"shapes": pd.DataFrame(shapes, index=idx), "record": record, "dt": dt, "mask": mask}


def transform_base(base: np.ndarray, alpha: float) -> np.ndarray:
    """``base ** alpha`` renormalised to the same mean (energy exact)."""
    b = np.asarray(base, dtype=float)
    b = b / b.mean()
    t = np.power(b, alpha)
    return t / t.mean()


def calibrate_base_exponent(base: np.ndarray, base_mwh: float, other_mw: np.ndarray, target_mw: float,
                            bounds: Tuple[float, float] = (0.2, 4.0), tol_mw: float = 0.01,
                            hours: Optional[float] = None) -> Tuple[float, float]:
    """Exponent alpha so max(base_mwh-scaled base**alpha + other) = target_mw (bisection).

    Returns:
        (alpha, achieved peak MW). Raises if the target is outside what the bounds reach.
    """
    b = np.asarray(base, dtype=float)
    hours = float(hours if hours is not None else len(b))

    def peak(alpha: float) -> float:
        return float((transform_base(b, alpha) * base_mwh / hours + other_mw).max())

    lo, hi = float(bounds[0]), float(bounds[1])
    p_lo, p_hi = peak(lo), peak(hi)
    if not (p_lo <= target_mw <= p_hi):
        raise ValueError(f"Peak target {target_mw:,.0f} MW outside {p_lo:,.0f}-{p_hi:,.0f} MW for exponent "
                         f"bounds {bounds}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = peak(mid)
        if abs(p - target_mw) <= tol_mw:
            return mid, p
        if p < target_mw:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, peak(mid)


def build_profiles(base_shape: pd.Series, gsp_gwh: pd.Series, components: pd.DataFrame,
                   named_gwh: pd.Series, shapes: pd.DataFrame, dt: float,
                   peak_target: Optional[float], bounds=(0.2, 4.0)) -> Dict[str, Any]:
    """MW per GSP column and per named site, with the base calibrated to the peak target.

    Args:
        base_shape: ESPENI base on the snapshots (any scale; only its shape is used).
        gsp_gwh: System GWh per GSP.
        components: GSP x component GWh (columns present in ``shapes``).
        named_gwh: GWh per named site (flat).
        shapes: Unit shapes per component (sum x dt = 1).
        dt: Snapshot length (hours).
        peak_target: GB peak MW, or None for no calibration (alpha = 1).
    """
    idx = shapes.index
    comps = components.reindex(gsp_gwh.index).fillna(0.0)
    comps = comps[[c for c in comps.columns if c in shapes.columns]]
    base_gwh = gsp_gwh - comps.sum(axis=1)
    if (base_gwh < -1e-9).any():
        raise ValueError("A GSP's components exceed its energy")
    hours = len(idx) * dt
    comp_mw = shapes[comps.columns].to_numpy() @ (comps.sum().to_numpy() * 1e3)
    named_mw = float(named_gwh.sum()) * 1e3 / hours
    b = base_shape.reindex(idx).to_numpy(dtype=float)
    base_mwh = float(base_gwh.sum()) * 1e3
    if peak_target is None:
        alpha, peak = 1.0, float((transform_base(b, 1.0) * base_mwh / hours + comp_mw + named_mw).max())
    else:
        alpha, peak = calibrate_base_exponent(b, base_mwh, comp_mw + named_mw, peak_target, bounds, hours=hours)
    b_unit = transform_base(b, alpha) / hours  # MW per MWh
    cols = (np.outer(b_unit, base_gwh.to_numpy() * 1e3)
            + shapes[comps.columns].to_numpy() @ (comps.to_numpy().T * 1e3))
    p_set = pd.DataFrame(cols, index=idx, columns=gsp_gwh.index)
    for name, gwh in named_gwh.items():
        p_set[name] = gwh * 1e3 / hours
    total = p_set.sum(axis=1)
    record = {
        "base_exponent": alpha, "peak_mw": float(total.max()), "peak_time": str(total.idxmax()),
        "energy_gwh": float(total.sum() * dt / 1e3), "load_factor": float(total.mean() / total.max()),
        "base_gwh": float(base_gwh.sum()), "components_gwh": {c: float(comps[c].sum()) for c in comps.columns},
        "named_gwh": float(named_gwh.sum()), "peak_target_mw": peak_target,
    }
    return {"p_set": p_set, "record": record,
            "national": pd.DataFrame({"base": b_unit * base_mwh,
                                      **{c: shapes[c].to_numpy() * comps[c].sum() * 1e3 for c in comps.columns},
                                      "named": np.full(len(idx), named_mw)}, index=idx)}


def gsp_lonlat(metadata: pd.DataFrame) -> pd.DataFrame:
    """GSP name (lower case) -> lon/lat from the FES regional-breakdown GSP metadata."""
    m = metadata.dropna(subset=["Latitude", "Longitude"]).copy()
    m["key"] = m["Name"].astype(str).str.strip().str.lower()
    return m.drop_duplicates("key").set_index("key")[["Longitude", "Latitude"]].rename(
        columns={"Longitude": "lon", "Latitude": "lat"})


def weather_temperature(cutout_path: str | Path, gsp_gwh: pd.Series, metadata: pd.DataFrame,
                        snapshots: pd.DatetimeIndex) -> pd.Series:
    """Demand-weighted GB temperature (deg C) on the snapshots, from the weather-year cutout."""
    import xarray as xr

    ll = gsp_lonlat(metadata)
    keys = gsp_gwh.index.astype(str).str.strip().str.lower()
    located = ll.reindex(keys)
    ok = located["lon"].notna().to_numpy()
    with xr.open_dataset(cutout_path) as ds:
        temp = f50.demand_weighted_temperature(ds["temperature"], located["lon"].to_numpy()[ok],
                                               located["lat"].to_numpy()[ok], gsp_gwh.to_numpy()[ok])
    return f50.map_weather_year(temp, snapshots)


def parse_node_shares(nodes: str) -> list:
    """``"A:0.5;B:0.5"`` -> [("A", 0.5), ("B", 0.5)] (a bare node has share 1)."""
    out = []
    for part in str(nodes or "").split(";"):
        part = part.strip()
        if not part:
            continue
        node, _, share = part.partition(":")
        out.append((node.strip(), float(share) if share else 1.0))
    return out


# ─────────────────────────────── build entry points ───────────────────────────────

def prepare(cfg: dict, pathway: str, year: int, fes_path: str | Path, ed1_path: str | Path,
            es1_path: Optional[str | Path], system_total_gwh: float,
            log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Annual energy by GSP and named site, the component split and the peak target.

    Returns a dict with ``demand_data`` (GWh, index = GSP names and ``NAMED::<id>``, one
    column named after the year: the frame the load builder distributes),
    ``named_node_shares`` (column -> {"nodes": [(node, share)], "lonlat"}), ``components`` (GSP x component
    GWh), ``energies``, ``lines``, ``accounting`` and ``peak_target``.
    """
    log = log or logger
    fes = pd.read_csv(fes_path, low_memory=False)
    ed1 = pd.read_csv(ed1_path)
    es1 = pd.read_csv(es1_path) if es1_path and Path(es1_path).exists() else None
    lines = read_national_lines(cfg["national_lines_file"])
    energies = cp30_component_energies(ed1, pathway, year, cfg["ed1_component_pathway"])
    named = read_named_loads(cfg["named_loads"]["file"]) if cfg["named_loads"].get("enabled", True) else None
    table = build_energy_table(fes, cfg, year, system_total_gwh, energies["consumer_total_gwh"], named, lines, log)
    lf = table["accounting"]["loss_factor"]
    sp = cfg["spatial"]
    comp_gwh = {
        "electric_vehicles": energies["evs_gwh"] * lf,
        "heat_pumps": energies["heat_pumps_gwh"] * lf,
        "district_heat": energies["district_heat_gwh"] * lf,
        "data_centres": lines["data_centres_distribution_gwh"] * lf,
    }
    wanted = list(cfg["shape"]["components"]) if cfg["shape"].get("enabled", True) else []
    unknown = set(wanted) - set(comp_gwh)
    if unknown:
        raise ValueError(f"Unknown demand.build.shape.components: {sorted(unknown)}")
    components, moved = allocate_components(fes, cfg, year, table["gsp_gwh"],
                                            {k: v for k, v in comp_gwh.items() if k in wanted}, log)
    ev_blocks = fes_gsp_blocks(fes, sp["FES_scenario"], year, ["Dem_BB006"], sp.get("direct_rows")).sum()
    ev_depot = fes_gsp_blocks(fes, sp["FES_scenario"], year, ["Dem_BB007"], sp.get("direct_rows")).sum()
    ev_res_share = lines["residential_evs_gwh"] / (lines["residential_evs_gwh"] + lines["commercial_evs_gwh"])
    demand_data = pd.concat([table["gsp_gwh"], table["named_gwh"]]).to_frame(str(year))
    named_shares = {}
    if named is not None:
        for etys_id, row in named.iterrows():
            shares = parse_node_shares(row["nodes"])
            lat, lon = row.get("latitude"), row.get("longitude")
            lonlat = None if pd.isna(lat) or pd.isna(lon) else (float(lon), float(lat))
            if not shares and lonlat is None:
                raise ValueError(f"Named load {etys_id} has neither nodes nor a location")
            named_shares[NAMED_PREFIX + str(etys_id)] = {"nodes": shares, "lonlat": lonlat}
    target = None
    if cfg["shape"].get("enabled", True):
        target = peak_target_mw(cfg["shape"]["peak_target"], pathway, year, es1, ed1)
    return {
        "demand_data": demand_data, "named_node_shares": named_shares, "named": named,
        "components": components, "component_cap_moved_gwh": moved, "component_gwh": comp_gwh,
        "energies": energies, "lines": lines, "accounting": table["accounting"],
        "gsp_gwh": table["gsp_gwh"], "named_gwh": table["named_gwh"], "peak_target": target,
        "ev_residential_share": ev_res_share,
        "ev_depot_share": float(ev_depot / (ev_blocks + ev_depot)) if (ev_blocks + ev_depot) > 0 else 0.0,
    }


def shape(prep: Dict[str, Any], p_set_espeni: pd.DataFrame, temp_c: Optional[pd.Series], cfg: dict,
          log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Replace the one-shape ESPENI columns with base + component shapes, peak calibrated.

    Args:
        prep: :func:`prepare` output.
        p_set_espeni: The ESPENI-scaled columns the load builder made (one per GSP and named
            site, all one shape); their sum is the base shape.
        temp_c: Weather-year demand-weighted temperature on the same index.
        cfg: Resolved config.

    Returns:
        ``p_set`` (MW, same columns), ``record`` (calibration and checks) and ``national``
        (GB MW by component).
    """
    log = log or logger
    idx = p_set_espeni.index
    lf = prep["accounting"]["loss_factor"]
    sh = national_shapes(idx, temp_c, cfg, prep["lines"], prep["component_gwh"]["electric_vehicles"] / lf
                         if "electric_vehicles" in prep["component_gwh"] else 0.0,
                         lf, prep["ev_residential_share"], prep["ev_depot_share"])
    # EV energy in national_shapes is consumer-level; scale its MW record to system level.
    rec = dict(sh["record"])
    for k in ("ev_at_peak_mw", "ev_unmanaged_at_peak_mw", "ev_smart_impact_at_peak_mw", "ev_target_at_peak_mw"):
        if k in rec:
            rec[k] *= lf
    target_mw, target_label = prep["peak_target"] if prep["peak_target"] else (None, "none")
    base = p_set_espeni.sum(axis=1)
    out = build_profiles(base, prep["gsp_gwh"], prep["components"], prep["named_gwh"], sh["shapes"], sh["dt"],
                         target_mw, tuple(cfg["shape"]["exponent_bounds"]))
    p_set = out["p_set"][list(p_set_espeni.columns)]
    record = {**out["record"], **rec, "peak_target_source": target_label}
    mask = sh["mask"]
    if mask is not None:
        nat = out["national"]
        for c in nat.columns:
            record[f"{c}_at_calibration_hours_mw"] = float(nat[c].to_numpy()[mask].mean())
        record["fes_heat_pumps_at_peak_mw"] = (prep["lines"]["residential_heat_pumps_at_peak_gw"]
                                               + prep["lines"]["commercial_heat_pumps_at_peak_gw"]) * lf * 1e3
    before_peak = float(base.max())
    log.info("CP30 demand shape: GB peak %.0f MW at %s (target %s %s; ESPENI-only %.0f MW), load factor %.3f, "
             "base exponent %.4f, energy %.1f GWh", record["peak_mw"], record["peak_time"],
             f"{target_mw:,.0f} MW" if target_mw else "none", target_label, before_peak,
             record["load_factor"], record["base_exponent"], record["energy_gwh"])
    log.info("CP30 demand components (GWh): %s; EV smart share of residential %.3f",
             {k: round(v, 1) for k, v in record["components_gwh"].items()},
             record.get("ev_smart_share_of_residential", float("nan")))
    return {"p_set": p_set, "record": record, "national": out["national"]}
