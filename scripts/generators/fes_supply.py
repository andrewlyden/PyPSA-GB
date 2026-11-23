"""FES supply for the long-range (gsp_block) build, for any FES year.

Everything here runs only when a scenario sets ``fleet.fes_supply.enabled: true`` (the FES long-range
studies: ``HT50_*`` today). With the key absent every caller keeps today's behaviour, so the CP30
2030 build is unchanged.

What it does on top of the generic FES path:

* **Thermal carriers by FES technology.** ``Gen_BB009`` "CCGTs (non CHP)" carries ES1's gas CCS
  and its unabated CCGTs together. In HT 2050 it is 22,205 MW, all CCS (ES1 has no unabated gas
  that year); in HT 2040 it is 25,033 MW, 19,625 CCS and 5,408 unabated. ``thermal.gas_ccs_split``
  splits every row of the technology in that year's ES1 proportion (:func:`es1_gas_ccs_share`);
  ``thermal.technology_carriers`` maps a whole technology to one carrier.
* **Declared station schedule** (``resources/FES/fes2025_<pathway>_<year>_station_schedule.csv``,
  built from ``data/generators/fes/station_list.csv`` by ``build_fes_station_schedule.py``). The
  FES "Direct" rows of nuclear, gas CCS, transmission hydrogen power and the BECCS part of biomass
  are replaced by named or declared sites at their buses. The total of every (carrier, licence
  area) is conserved exactly; BECCS takes only its ES1 capacity out of the biomass Direct row.
* **Offshore landing.** The FES Direct offshore rows (102,971 MW in 2050) are replaced by task
  Y5's landing table for the year (``data/network/ETYS/fes<year>_offshore_landing.csv``), one row
  per landing bus. Each row takes the weather profile of a named project that lands there, else
  of the nearest profiled offshore site, so offshore output keeps its geography.
* **Declared buses survive the generic heuristics.** Rows placed by the schedule or the
  landing table carry ``fes_declared_bus``; :func:`restore_declared_buses` puts them
  back after the ETYS BMU and weak-bus corrections. Clustering then moves them with their
  bus through the busmap like every other component.
* **Storage by type** at FES ES1 power and energy (:func:`build_fes_storage`), with the
  compressed-air energy at the FES Economics duration (E.01, 20 h) instead of ES1's 1 h.
* **Off-grid offshore** (``Gen_BB024``, Non-Networked, 442.9 MW in 2050) is excluded: its FES
  Technology "Offshore-Wind (off-Grid)" matches neither the renewable nor the thermal
  loader, and :func:`check_offgrid_excluded` makes the exclusion an assertion.

Names written into the network (site names, ``data_source``) carry the modelled year, so a
2050 build writes exactly the names it wrote before (``FES2050_...``).
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

#: The block's key under the scenario's ``fleet`` block.
CONFIG_KEY = "fes_supply"
DECLARED_BUS_COLUMN = "fes_declared_bus"
LICENCE_AREAS = {"Direct(NGET)": "NGET", "Direct(SHETL)": "SHETL", "Direct(SPTL)": "SPTL"}
OFFGRID_TECHNOLOGY = "Offshore-Wind (off-Grid)"
OFFGRID_BLOCK = "Gen_BB024"
#: ES1 lines of unabated gas that share Gen_BB009 "CCGTs (non CHP)" with gas CCS: (Type, SubType).
UNABATED_CCGT_ES1_LINES = (("Gas", "CCGT"),)

REQUIRED_SCHEDULE_COLUMNS = {
    "station_name", "carrier", "fes_source_carrier", "fes_region", "capacity_mw", "bus",
    "basis", "partial_of_direct", "source", "status",
}

logger = logging.getLogger(__name__)


def schedule_data_source(year: int) -> str:
    """``data_source`` of the declared stations of ``year`` (``FES2050_station_schedule``)."""
    return f"FES{int(year)}_station_schedule"


def landing_data_source(year: int) -> str:
    """``data_source`` of the offshore landing rows of ``year`` (``FES2050_offshore_landing``)."""
    return f"FES{int(year)}_offshore_landing"


# ───────────────────────────────── configuration ─────────────────────────────────

def fes_supply_config(scenario_config: Optional[dict]) -> dict:
    """The scenario's ``fleet.fes_supply`` block, or ``{}`` when absent or disabled."""
    block = dict(((scenario_config or {}).get("fleet") or {}).get(CONFIG_KEY) or {})
    return block if block.get("enabled", False) else {}


def fes_supply_part(scenario_config: Optional[dict], part: str) -> dict:
    """One enabled sub-block (``thermal``, ``offshore_landing``, ``storage``, ``hydrogen``,
    ``interconnectors``) of an enabled ``fleet.fes_supply`` block, else ``{}``.

    A sub-block is on unless it says ``enabled: false``.
    """
    block = fes_supply_config(scenario_config)
    if not block:
        return {}
    sub = dict(block.get(part) or {})
    if not sub or not sub.get("enabled", True):
        return {}
    return sub


def load_targets(path) -> dict:
    """The Y1 FES 2025 targets JSON (``build_fes2025_energy_targets``)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ───────────────────────────────── bus helpers ─────────────────────────────────

def resolve_node_bus(node_id, buses: pd.DataFrame) -> Optional[str]:
    """ETYS bus for an FES 2025 ``Node ID`` (FES 2025 names the ETYS node behind a GSP row).

    Exact name first; else a bus of the same four-letter site with the same voltage digit;
    else the highest-voltage bus of that site. ``None`` if the site is not in the network.
    """
    if node_id is None or (isinstance(node_id, float) and np.isnan(node_id)):
        return None
    node = str(node_id).strip()
    if not node or node.startswith(("Direct", "Not")):
        return None
    if node in buses.index:
        return node
    site = node[:4]
    same_site = buses[buses.index.str[:4] == site]
    same_site = same_site[same_site.get("carrier", pd.Series("AC", index=same_site.index)).fillna("AC").isin(["AC", ""])]
    if same_site.empty:
        return None
    if len(node) >= 5:
        same_digit = same_site[same_site.index.str[4:5] == node[4:5]]
        if not same_digit.empty:
            return str(sorted(same_digit.index)[0])
    return str(same_site["v_nom"].astype(float).idxmax())


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def restore_declared_buses(sites: pd.DataFrame, log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Put rows that carry ``fes_declared_bus`` back on that bus.

    The generic ETYS corrections (BMU-to-node, weak-bus and export relief) move large units to
    other buses by heuristic; a declared station or landing bus is evidence, not a guess.
    """
    log = log or logger
    if DECLARED_BUS_COLUMN not in sites.columns:
        return sites
    declared = sites[DECLARED_BUS_COLUMN]
    mask = declared.notna() & declared.astype(str).str.len().gt(0)
    moved = mask & sites["bus"].astype(str).ne(declared.astype(str))
    if moved.any():
        log.info("FES supply: restored %d declared buses moved by the generic corrections (%.0f MW): %s",
                 int(moved.sum()), float(sites.loc[moved, "capacity_mw"].sum()) if "capacity_mw" in sites else 0.0,
                 "; ".join(f"{b0}->{b1}" for b0, b1 in zip(sites.loc[moved, "bus"], declared[moved]))[:2000])
    out = sites.copy()
    out.loc[mask, "bus"] = declared[mask]
    return out


# ───────────────────────────────── thermal ─────────────────────────────────

def load_station_schedule(path) -> pd.DataFrame:
    """Read and validate a declared station schedule (one year's, ``build_fes_station_schedule``)."""
    frame = pd.read_csv(path)
    if missing := REQUIRED_SCHEDULE_COLUMNS.difference(frame.columns):
        raise ValueError(f"FES station schedule is missing columns: {sorted(missing)}")
    if frame["source"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every FES station schedule row needs a source")
    if frame["station_name"].duplicated().any():
        dup = frame.loc[frame["station_name"].duplicated(), "station_name"].tolist()
        raise ValueError(f"FES station schedule has duplicate station names: {dup}")
    frame["capacity_mw"] = pd.to_numeric(frame["capacity_mw"], errors="raise").astype(float)
    if (frame["capacity_mw"] <= 0).any():
        raise ValueError("FES station schedule capacities must be positive")
    frame["partial_of_direct"] = frame["partial_of_direct"].astype(str).str.lower().isin(["true", "1", "yes"])
    for column in ("lat", "lon"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame


def remap_thermal_carriers(fes_df: pd.DataFrame, technology_carriers: Dict[str, str],
                           log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Carrier override by FES technology (``Gen_BB009`` 'CCGTs (non CHP)' -> 'CCS Gas')."""
    log = log or logger
    out = fes_df.copy()
    for technology, carrier in (technology_carriers or {}).items():
        mask = out["technology"].astype(str).eq(technology)
        if mask.any():
            log.info("FES supply: %d rows of FES technology %r (%.0f MW) mapped to carrier %r",
                     int(mask.sum()), technology, float(out.loc[mask, "capacity_mw"].sum()), carrier)
            out.loc[mask, "fuel_type"] = carrier
            out.loc[mask, "fes_technology_remap"] = f"{technology} -> {carrier}"
    return out


def es1_gas_ccs_share(es1_lines: pd.DataFrame, ccs_carrier: str = "CCS Gas",
                      unabated_lines: Iterable[Tuple[str, str]] = UNABATED_CCGT_ES1_LINES) -> Dict[str, float]:
    """Share of ``Gen_BB009`` "CCGTs (non CHP)" that is gas CCS, from one year's ES1 capacity lines.

    FES carries gas CCS and unabated CCGTs in the same building block; ES1 splits them. The
    share is ES1 CCS gas over CCS gas plus the unabated CCGT lines (``Gas`` / ``CCGT``, every
    connection). OCGTs and gas engines have their own building blocks and are not in it.

    Args:
        es1_lines: The classified ES1 lines of the year (``fes2025_<pathway>_<year>_es1_lines.csv``).
        ccs_carrier: ES1 carrier of gas CCS.
        unabated_lines: (Type, SubType) of the unabated ES1 lines that share the block.

    Returns:
        ``{"share": ..., "ccs_mw": ..., "unabated_mw": ...}``.

    Raises:
        ValueError: ES1 has no capacity on either side.
    """
    capacity = es1_lines[es1_lines["quantity"].astype(str).eq("capacity_mw")]
    ccs_mw = float(capacity.loc[capacity["carrier"].astype(str).eq(ccs_carrier), "value_std"].sum())
    pairs = {(str(t), str(st)) for t, st in unabated_lines}
    unabated = capacity[[(str(t), str(st)) in pairs for t, st in zip(capacity["type"], capacity["subtype"])]]
    unabated_mw = float(unabated["value_std"].sum())
    total = ccs_mw + unabated_mw
    if total <= 0:
        raise ValueError(f"ES1 has no {ccs_carrier} or unabated CCGT capacity: the gas CCS split is undefined")
    return {"share": ccs_mw / total, "ccs_mw": ccs_mw, "unabated_mw": unabated_mw}


def split_gas_ccs(fes_df: pd.DataFrame, split: Dict[str, float], technology: str = "CCGTs (non CHP)",
                  ccs_carrier: str = "CCS Gas", unabated_carrier: str = "CCGT",
                  log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Split every row of ``technology`` into gas CCS and unabated CCGT at the ES1 share.

    Each row keeps its GSP and node and becomes a CCS row of share x capacity and, when the
    share is below one, an unabated row of the rest. At a share of one (HT 2050) the table is
    exactly what the carrier remap ``{technology: ccs_carrier}`` gives.
    """
    log = log or logger
    share = float(split["share"])
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"gas CCS share {share} is outside [0, 1]")
    mask = fes_df["technology"].astype(str).eq(technology)
    out = fes_df.copy()
    if not mask.any():
        return out
    rows = out.loc[mask].copy()
    if share >= 1.0:
        # All CCS: exactly the carrier remap, so the table does not move.
        return remap_thermal_carriers(fes_df, {technology: ccs_carrier}, log)
    note = f"{technology} -> {ccs_carrier} {share:.6f} / {unabated_carrier} {1 - share:.6f} (ES1)"
    out.loc[mask, "fuel_type"] = ccs_carrier
    out.loc[mask, "capacity_mw"] = rows["capacity_mw"] * share
    out.loc[mask, "fes_technology_remap"] = note
    unabated = rows.assign(fuel_type=unabated_carrier, capacity_mw=rows["capacity_mw"] * (1.0 - share),
                           fes_technology_remap=note)
    out = pd.concat([out, unabated], ignore_index=True, sort=False)
    out = out[out["capacity_mw"] > 1e-6].reset_index(drop=True)
    total = float(rows["capacity_mw"].sum())
    log.info("FES supply: %d rows of FES technology %r (%.0f MW) split by ES1: %s %.0f MW (ES1 %.0f), "
             "%s %.0f MW (ES1 %.0f)", int(mask.sum()), technology, total, ccs_carrier, total * share,
             split.get("ccs_mw", float("nan")), unabated_carrier, total * (1.0 - share),
             split.get("unabated_mw", float("nan")))
    return out


def _direct_region(gsp) -> Optional[str]:
    return LICENCE_AREAS.get(str(gsp).strip()) if gsp is not None and not (isinstance(gsp, float) and np.isnan(gsp)) else None


def apply_station_schedule(fes_df: pd.DataFrame, schedule: pd.DataFrame, atol_mw: float = 1.0,
                           log: Optional[logging.Logger] = None, year: int = 2050) -> pd.DataFrame:
    """Replace FES Direct rows with the declared stations, conserving every total.

    For each (``fes_source_carrier``, licence area) the schedule rows must sum to the FES
    Direct capacity of that carrier there, unless every row is ``partial_of_direct`` (BECCS):
    then the schedule takes its capacity out of the Direct row, which keeps the rest.
    ``year`` is the modelled year; it names the rows' ``data_source``.
    """
    log = log or logger
    out = fes_df.copy()
    region = out["gsp"].map(_direct_region)
    template_cols = list(out.columns)
    keep = pd.Series(True, index=out.index)
    new_rows: List[dict] = []
    for (source_carrier, area), group in schedule.groupby(["fes_source_carrier", "fes_region"]):
        direct = region.eq(area) & out["fuel_type"].astype(str).eq(source_carrier)
        direct_mw = float(out.loc[direct, "capacity_mw"].sum())
        scheduled_mw = float(group["capacity_mw"].sum())
        partial = bool(group["partial_of_direct"].all())
        if group["partial_of_direct"].any() and not partial:
            raise ValueError(f"FES station schedule mixes partial and full rows for {source_carrier} {area}")
        if partial:
            if scheduled_mw > direct_mw + atol_mw:
                raise ValueError(f"FES station schedule takes {scheduled_mw:,.1f} MW of {source_carrier} {area} "
                                 f"but FES Direct holds {direct_mw:,.1f} MW")
            factor = (direct_mw - scheduled_mw) / direct_mw if direct_mw > 0 else 0.0
            out.loc[direct, "capacity_mw"] = out.loc[direct, "capacity_mw"] * factor
            out.loc[direct, "fes_schedule_note"] = f"reduced by {scheduled_mw:,.1f} MW taken by the schedule"
        else:
            if abs(scheduled_mw - direct_mw) > atol_mw:
                raise ValueError(f"FES station schedule has {scheduled_mw:,.1f} MW of {source_carrier} {area} but "
                                 f"FES Direct holds {direct_mw:,.1f} MW; rebuild the schedule")
            keep &= ~direct
        for row in group.itertuples(index=False):
            record = {col: np.nan for col in template_cols}
            record.update({
                "technology": row.carrier, "fuel_type": row.carrier, "capacity_mw": float(row.capacity_mw),
                "bus": row.bus, DECLARED_BUS_COLUMN: row.bus, "station_name": row.station_name,
                "site_name": row.station_name, "source_site_name": row.station_name,
                "lat": row.lat, "lon": row.lon, "gsp": np.nan, "data_source": schedule_data_source(year),
                "spatial_proxy_source": row.source, "fes_schedule_basis": row.basis,
                "fes_schedule_status": row.status, "fes_region": area,
            })
            new_rows.append(record)
        log.info("FES station schedule: %s %s Direct %.1f MW -> %d stations %.1f MW%s", source_carrier, area,
                 direct_mw, len(group), scheduled_mw, " (partial take)" if partial else "")
    result = pd.concat([out[keep], pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    result = result[result["capacity_mw"] > 1e-6].reset_index(drop=True)
    return result


def check_offgrid_excluded(frame: pd.DataFrame, label: str) -> None:
    """Fail if the Non-Networked offshore block reached a generator table (Gen_BB024)."""
    tech = frame.get("technology", pd.Series(dtype=object)).astype(str)
    if tech.eq(OFFGRID_TECHNOLOGY).any():
        raise ValueError(f"{label}: off-grid offshore ({OFFGRID_BLOCK}, {OFFGRID_TECHNOLOGY}) must not be on the grid")


def apply_fes_thermal(fes_df: pd.DataFrame, config: dict, log: Optional[logging.Logger] = None,
                      year: int = 2050, es1_lines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The long-range thermal table: gas CCS split or carrier remap, then the declared station schedule.

    Args:
        fes_df: ``load_fes_generators`` output for the modelled year.
        config: ``fleet.fes_supply.thermal``. ``gas_ccs_split`` (``enabled``, ``technology``,
            ``ccs_carrier``, ``unabated_carrier``) splits the CCGT block at the ES1 share and needs
            ``es1_lines``; ``technology_carriers`` remaps whole technologies; ``station_schedule``
            is the year's schedule file.
        log: Logger.
        year: Modelled year (names the scheduled rows' ``data_source``).
        es1_lines: The year's classified ES1 lines (for ``gas_ccs_split``).
    """
    log = log or logger
    check_offgrid_excluded(fes_df, "FES thermal")
    split_cfg = dict(config.get("gas_ccs_split") or {})
    remap = dict(config.get("technology_carriers") or {})
    out = fes_df
    if split_cfg and split_cfg.get("enabled", True):
        if es1_lines is None:
            raise ValueError("fleet.fes_supply.thermal.gas_ccs_split needs the year's ES1 lines (es1_lines_file)")
        technology = split_cfg.get("technology", "CCGTs (non CHP)")
        ccs_carrier = split_cfg.get("ccs_carrier", "CCS Gas")
        split = es1_gas_ccs_share(es1_lines, ccs_carrier)
        out = split_gas_ccs(out, split, technology, ccs_carrier, split_cfg.get("unabated_carrier", "CCGT"), log)
        remap.pop(technology, None)   # the split decides this technology's carriers
    out = remap_thermal_carriers(out, remap, log)
    schedule_path = config.get("station_schedule")
    if schedule_path:
        out = apply_station_schedule(out, load_station_schedule(schedule_path), log=log, year=year)
    summary = out.groupby("fuel_type")["capacity_mw"].sum().sort_values(ascending=False)
    log.info("FES %s thermal by carrier (MW): %s", year, ", ".join(f"{k} {v:,.0f}" for k, v in summary.items()))
    return out


# ───────────────────────────────── offshore landing ─────────────────────────────────

def _norm_name(value) -> str:
    return " ".join(str(value).split()).strip().lower()


def choose_landing_profiles(landing: pd.DataFrame, projects: Optional[pd.DataFrame],
                            profile_columns: Iterable[str],
                            profile_sites: Optional[pd.DataFrame],
                            min_nearest_site_mw: float = 100.0) -> pd.DataFrame:
    """Profile column for each landing bus: its largest profiled project, else the nearest profiled site.

    ``profile_sites`` has ``site_name``, ``lat``, ``lon`` (and ``capacity_mw``) for the
    profile columns (the offshore site files); ``projects`` is Y5's per-project landing
    table. The nearest-site fallback skips sites under ``min_nearest_site_mw`` (declared:
    demonstrators and single test turbines sit near shore and are not representative).
    """
    columns = {}
    for column in profile_columns:
        # Profile columns may carry one source tag, "<site> (REPD)"; match on both forms.
        columns.setdefault(_norm_name(column), column)
        untagged = _norm_name(pd.Series([column]).str.replace(r"\s*\([^()]*\)$", "", regex=True).iloc[0])
        columns.setdefault(untagged, column)
    chosen, basis = [], []
    coords = None
    if profile_sites is not None and len(profile_sites):
        sites = profile_sites.dropna(subset=["lat", "lon"]).copy()
        sites["key"] = sites["site_name"].map(_norm_name)
        sites = sites[sites["key"].isin(columns)].drop_duplicates("key")
        if "capacity_mw" in sites.columns:
            capacity = pd.to_numeric(sites["capacity_mw"], errors="coerce")
            sites = sites[capacity.isna() | (capacity >= min_nearest_site_mw)]
        coords = sites
    for row in landing.itertuples(index=False):
        pick = None
        if projects is not None and len(projects):
            own = projects[projects["landing_bus"].eq(row.landing_bus)].copy()
            own["key"] = own["project"].map(_norm_name)
            own = own[own["key"].isin(columns)].sort_values("capacity_mw", ascending=False)
            if len(own):
                pick = (columns[own["key"].iloc[0]], f"landing project {own['project'].iloc[0]}")
        if pick is None and coords is not None and len(coords):
            d = _haversine_km(float(row.lat), float(row.lon), coords["lat"].to_numpy(), coords["lon"].to_numpy())
            i = int(np.argmin(d))
            pick = (columns[coords["key"].iloc[i]], f"nearest profiled site {coords['site_name'].iloc[i]} ({d[i]:.0f} km)")
        chosen.append(pick[0] if pick else np.nan)
        basis.append(pick[1] if pick else "no profile site: carrier fleet average")
    return landing.assign(profile_column=chosen, profile_basis=basis)


def apply_fes_offshore_landing(sites: pd.DataFrame, landing_path, network_buses: pd.DataFrame,
                                   projects_path=None, profile_columns: Iterable[str] = (),
                                   profile_sites: Optional[pd.DataFrame] = None, modelled_year: int = 2050,
                                   atol_mw: float = 1.0, log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Replace the FES Direct offshore rows by the Y5 landing table (bus, MW).

    Regional totals are conserved: the landing table's NGET, SHETL and SPTL sums must equal the
    Direct offshore capacity the FES loader produced for each licence area. Every landing bus
    must exist in the network (the dead-island rule: one row per surviving bus).
    """
    log = log or logger
    check_offgrid_excluded(sites, f"FES {modelled_year} renewables")
    landing = pd.read_csv(landing_path)
    projects = pd.read_csv(projects_path) if projects_path else None
    absent = sorted(set(landing["landing_bus"]) - set(network_buses.index))
    if absent:
        raise ValueError(f"FES {modelled_year} offshore landing buses absent from the network: {absent}")
    is_direct = (sites["technology"].eq("wind_offshore")
                 & sites.get("data_source", pd.Series(index=sites.index, dtype=object)).eq("FES_Direct"))
    region = sites["gsp"].map(_direct_region)
    for area, target in landing.groupby("fes_region")["capacity_mw"].sum().items():
        direct_mw = float(sites.loc[is_direct & region.eq(area), "capacity_mw"].sum())
        if abs(direct_mw - float(target)) > atol_mw:
            raise ValueError(f"FES {modelled_year} offshore landing {area}: table {target:,.1f} MW but FES Direct "
                             f"offshore {direct_mw:,.1f} MW")
    leftover = sorted(set(region[is_direct].dropna()) - set(landing["fes_region"]))
    if leftover:
        raise ValueError(f"FES Direct offshore in {leftover} has no landing rows")
    landing = choose_landing_profiles(landing, projects, profile_columns, profile_sites)
    rows = []
    for row in landing.itertuples(index=False):
        rows.append({
            "technology": "wind_offshore", "capacity_mw": float(row.capacity_mw), "bus": row.landing_bus,
            DECLARED_BUS_COLUMN: row.landing_bus, "lat": float(row.lat), "lon": float(row.lon),
            "site_name": f"FES{modelled_year}_offshore_{row.landing_bus}_{modelled_year}",
            "source_site_name": row.profile_column if isinstance(row.profile_column, str) else np.nan,
            "data_source": landing_data_source(modelled_year), "gsp": f"Direct({row.fes_region})",
            "spatial_proxy_source": str(row.source), "landing_evidence": str(row.landing_basis),
            "fes_profile_basis": row.profile_basis, "fes_region": row.fes_region,
        })
    result = pd.concat([sites[~is_direct], pd.DataFrame(rows)], ignore_index=True, sort=False)
    log.info("FES %s offshore landing: %d Direct rows (%.1f MW) replaced by %d landing buses (%.1f MW); "
             "%d take a landing project's profile, %d the nearest profiled site", modelled_year,
             int(is_direct.sum()), float(sites.loc[is_direct, "capacity_mw"].sum()), len(rows),
             float(landing["capacity_mw"].sum()),
             int(landing["profile_basis"].str.startswith("landing project").sum()),
             int(landing["profile_basis"].str.startswith("nearest").sum()))
    return result


# ───────────────────────────────── storage ─────────────────────────────────

# FES storage building blocks -> model storage carriers. Srg_BB004 "Other" is LAES plus
# CAES in ES1 (5,454.9 + 4,003.3 = 9,458.2 MW); it is split on those ES1 shares, row by row.
STORAGE_BLOCKS = {
    "Srg_BB001": ["Battery"],
    "Srg_BB002": ["Domestic Battery"],
    "Srg_BB003": ["Pumped Storage Hydroelectricity"],
    "Srg_BB004": ["LAES", "CAES"],
}
# ES1 carrier of each model storage carrier (the domestic battery is inside ES1 "Battery").
ES1_STORAGE_CARRIER = {"Battery": "Battery", "Domestic Battery": "Battery",
                       "Pumped Storage Hydroelectricity": "Pumped Storage Hydroelectricity",
                       "LAES": "LAES", "CAES": "CAES"}


def storage_targets(targets: dict) -> pd.DataFrame:
    """ES1 storage power (MW), energy (MWh), duration and round trip by ES1 carrier."""
    frame = pd.DataFrame(targets["storage"]).set_index("carrier")
    frame["energy_mwh"] = frame["energy_gwh"] * 1e3
    frame["duration_h"] = frame["energy_mwh"] / frame["capacity_mw"]
    frame["round_trip"] = frame["generation_gwh"] / frame["consumption_gwh"]
    return frame


def build_fes_storage(fes_path, targets: dict, config: dict, network_buses: pd.DataFrame,
                          repd_storage: pd.DataFrame, pathway: str = "Holistic Transition",
                          year: int = 2050, round_trip_efficiencies: Optional[dict] = None,
                          log: Optional[logging.Logger] = None, licence_areas=None) -> pd.DataFrame:
    """Storage units for one FES pathway and year at ES1 power and energy by type.

    * GSP rows are placed at the ETYS bus FES names (``Node ID``); rows whose node is not in
      the network go to the nearest REPD site of the carrier's licence area (declared).
    * Direct rows are spread over REPD storage sites of the same technology family in the
      licence area in proportion to their power (pumped hydro over pumped-hydro sites only,
      and over the declared ``pumped_hydro_sites`` file if given), as the generic loader does.
    * Each carrier is then scaled to its ES1 power, and its energy set from the ES1 duration,
      except where ``durations_h`` declares one (CAES 20 h, FES Economics E.01 row 118).
    """
    log = log or logger
    round_trip_efficiencies = dict(round_trip_efficiencies or {})
    es1 = storage_targets(targets)
    fes = pd.read_csv(fes_path, encoding="utf-8-sig", low_memory=False)
    fes = fes[fes["FES Pathway"].eq(pathway) & fes["Building Block ID Number"].isin(list(STORAGE_BLOCKS))]
    fes = fes[pd.to_numeric(fes[str(year)], errors="coerce").fillna(0) > 0]
    other_split = es1.loc[["LAES", "CAES"], "capacity_mw"]
    other_split = other_split / other_split.sum()
    durations = {**{c: float(es1.loc[ES1_STORAGE_CARRIER[c], "duration_h"]) for c in ES1_STORAGE_CARRIER},
                 **{k: float(v) for k, v in (config.get("durations_h") or {}).items()}}
    phs_sites = None
    if config.get("pumped_hydro_sites"):
        phs_sites = pd.read_csv(config["pumped_hydro_sites"])

    def _region_sites(carrier: str, area: str) -> pd.DataFrame:
        if carrier == "Pumped Storage Hydroelectricity" and phs_sites is not None:
            pool = phs_sites.rename(columns={"capacity_mw": "power_mw"})
            return pool[pool["fes_region"].eq(area)]
        family = {"Battery": "Battery", "Domestic Battery": "Battery",
                  "Pumped Storage Hydroelectricity": "Pumped Storage Hydroelectricity"}.get(carrier)
        # The other storage (LAES, CAES) is spread over every REPD storage site but pumped hydro (8 Nov 2026):
        # weighted by power, Cruachan's record drew 464 MW of SPT LAES and CAES to the Cruachan site.
        pool = (repd_storage[repd_storage["technology"].eq(family)] if family
                else repd_storage[~repd_storage["technology"].eq("Pumped Storage Hydroelectricity")])
        pool = pool[pool["power_mw"] > 0]
        lat = pool["lat"]
        if licence_areas is not None:   # the GSP polygon's licence area (27 Oct 2026)
            pool = pool[licence_areas.site_licence(lat, pool["lon"]).eq(area).to_numpy()]
        elif area == "SHETL":
            pool = pool[lat > 57.0]
        elif area == "SPTL":
            pool = pool[(lat > 55.5) & (lat <= 57.0)]
        else:
            pool = pool[lat <= 55.5]
        return pool if len(pool) else repd_storage[repd_storage["power_mw"] > 0]

    rows = []
    for _, record in fes.iterrows():
        block = record["Building Block ID Number"]
        mw_total = float(record[str(year)])
        gsp = record["GSP"]
        node = record.get("Node ID")
        carriers = STORAGE_BLOCKS[block]
        for carrier in carriers:
            mw = mw_total * (float(other_split[carrier]) if block == "Srg_BB004" else 1.0)
            area = _direct_region(gsp)
            if area is None:
                bus = resolve_node_bus(node, network_buses)
                if bus is not None:
                    rows.append({"site_name": f"FES{year}_{carrier.replace(' ', '_')}_{bus}", "technology": carrier,
                                 "power_mw": mw, "bus": bus, "lat": network_buses.at[bus, "lat"] if "lat" in network_buses else np.nan,
                                 "lon": network_buses.at[bus, "lon"] if "lon" in network_buses else np.nan,
                                 "source": f"FES{year}_GSP_{block}", "placement": f"FES Node ID {node}"})
                    continue
                area = "NGET"   # unplaced GSP row: spread like a Direct row (declared)
                placement = f"GSP {gsp} (node {node}) not in network: spread over REPD sites"
            else:
                placement = f"{gsp}: spread over REPD sites"
            pool = _region_sites(carrier, area)
            weights = pool["power_mw"] / pool["power_mw"].sum()
            for site, share in zip(pool.itertuples(index=False), weights):
                # With licence areas the bus is the nearest of the site's own licence (Cruachan on CRUA,
                # not the SHE buses beside it); without, the storage build maps it to the nearest bus.
                bus = (licence_areas.nearest_bus(site.lat, site.lon, area) if licence_areas is not None
                       else np.nan)
                # A declared pumped-hydro site carries its TEC substation's code (8 Nov 2026), so the storage
                # build pins it there and weak-bus relief leaves it: relief had moved Cruachan's 1.29 GW from
                # CRUA (SPT) to Kilchrenan and Fernoch, north of B4.
                code = str(getattr(site, "site_code", "") or "")
                code = "" if code.lower() == "nan" else code
                rows.append({"site_name": f"FES{year}_{carrier.replace(' ', '_')}_{site.site_name}",
                             "technology": carrier, "power_mw": mw * float(share), "bus": bus,
                             "lat": site.lat, "lon": site.lon, "source": f"FES{year}_Direct_{block}",
                             "placement": placement, "site_code": code})
    table = pd.DataFrame(rows)
    # Aggregate duplicates (one row per carrier and bus or site).
    keys = ["technology", "site_name"]
    table = table.groupby(keys, as_index=False).agg(
        power_mw=("power_mw", "sum"), bus=("bus", "first"), lat=("lat", "first"), lon=("lon", "first"),
        source=("source", "first"), placement=("placement", "first"),
        site_code=("site_code", lambda s: next((c for c in s.fillna("").astype(str) if c), "")))
    # Scale each carrier to its ES1 power; energy from the declared duration.
    es1_power = {c: float(es1.loc[ES1_STORAGE_CARRIER[c], "capacity_mw"]) for c in ES1_STORAGE_CARRIER}
    built = table.groupby("technology")["power_mw"].sum()
    battery_built = float(built.get("Battery", 0.0) + built.get("Domestic Battery", 0.0))
    for carrier in table["technology"].unique():
        if carrier in ("Battery", "Domestic Battery"):
            factor = es1_power["Battery"] / battery_built if battery_built > 0 else 1.0
        else:
            factor = es1_power[carrier] / float(built[carrier])
        mask = table["technology"].eq(carrier)
        table.loc[mask, "power_mw"] *= factor
        table.loc[mask, "es1_scale_factor"] = factor
    table["duration_h"] = table["technology"].map(durations)
    table["energy_mwh"] = table["power_mw"] * table["duration_h"]
    rt = {c: float(round_trip_efficiencies.get(c, es1.loc[ES1_STORAGE_CARRIER[c], "round_trip"]))
          for c in ES1_STORAGE_CARRIER}
    table["round_trip"] = table["technology"].map(rt)
    table["eta_charge"] = np.sqrt(table["round_trip"])
    table["eta_discharge"] = np.sqrt(table["round_trip"])
    for carrier, group in table.groupby("technology"):
        log.info("FES %s storage %s: %.1f MW, %.1f MWh (%.2f h), round trip %.3f, ES1 scale %.4f, %d units",
                 year, carrier, group["power_mw"].sum(), group["energy_mwh"].sum(), group["duration_h"].iloc[0],
                 group["round_trip"].iloc[0], group["es1_scale_factor"].iloc[0], len(group))
    return table


def storage_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Power, energy and duration by carrier (for logs and tests)."""
    g = table.groupby("technology").agg(power_mw=("power_mw", "sum"), energy_mwh=("energy_mwh", "sum"))
    g["duration_h"] = g["energy_mwh"] / g["power_mw"]
    return g
