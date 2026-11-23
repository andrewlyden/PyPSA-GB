"""Build the CP30 renewable fleet from named projects, proxying only the residual.

CP30 says how much capacity there is in 2030 (ES1 nationally, ES5 by region)
and nothing about which projects make it up. The proxy-only approach scales an
FES spatial proxy to those totals, which produces a fleet with no station
identity: no contract round, no accreditation date, no real connection point,
and one load-factor shape per carrier.

This module keeps CP30 authoritative for *how much* and lets the REPD register
decide *what and where* inside those envelopes:

    named_mw    = operational projects still within their technical lifetime
                + pipeline projects at a declared maturity, expected online
    residual    = ES5 regional target - named_mw          (per carrier, region)
    residual > 0 -> the FES proxy fills exactly the residual in that region
    residual < 0 -> a declared overshoot policy resolves it, least mature first

Total capacity per (carrier, region) therefore equals ES5 exactly, so the
existing CP30 totals validation keeps its meaning. Every decision - retirement,
trimming, landing evidence - is written to an audit table rather than logged and
forgotten.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.market.cfd_strike_table import MODELLED_ROUND


def _parse_operational_dates(values: pd.Series) -> pd.Series:
    """Parse REPD day-first dates and ISO dates explicitly.

    Mirrors the parser in ``integrate_renewable_generators`` rather than
    importing it: that module imports this one, and a top-level import back
    would be circular.
    """
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", na=False)
    parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask].str[:10], format="%Y-%m-%d", errors="coerce")
    parsed.loc[~iso_mask] = pd.to_datetime(text.loc[~iso_mask], format="%d/%m/%Y", errors="coerce")
    return parsed


BROWNFIELD_CARRIERS = ("wind_offshore", "wind_onshore", "solar_pv")

DEFAULT_STATUSES = ("Operational", "Under Construction", "Awaiting Construction")

# Lower is more mature. Used to decide what gives way when the named fleet in a
# region exceeds what CP30 allows there.
MATURITY_RANK = {
    "Operational": 0,
    "Under Construction": 1,
    "Awaiting Construction": 2,
    "Application Submitted": 3,
}

OVERSHOOT_POLICIES = ("trim_least_mature", "error")
LANDING_POLICIES = ("nearest_bus_fallback", "error")

# Columns carried on every named row so downstream stages and the audit can
# tell a REPD project from a proxy tranche.
NAMED_COLUMNS = [
    "site_name", "source_site_name", "repd_ref_id", "technology", "capacity_mw", "lat", "lon",
    "cfd_round", "cfd_capacity_mw", "ro_banding", "operational_date", "development_status",
    "maturity_rank", "fleet_origin",
]


def _log(logger):
    return logger or logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named fleet
# ---------------------------------------------------------------------------

def _read_sites(path, technology: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Site file for {technology} not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["technology"] = technology
    frame["capacity_mw"] = pd.to_numeric(frame.get("capacity_mw"), errors="coerce")
    frame = frame[frame["capacity_mw"].gt(0) & frame["lat"].notna() & frame["lon"].notna()]
    for column in ("cfd_round", "cfd_capacity_mw", "ro_banding", "operational_date",
                   "repd_ref_id", "source_site_name"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["repd_ref_id"] = frame["repd_ref_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return frame


def retire_by_lifetime(operational: pd.DataFrame, modelled_year: int,
                       lifetimes_years: dict | None, logger=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop operational projects past a declared technical lifetime by the model year.

    REPD records no decommissioning, so without this a 1995 turbine runs in
    2030. The lifetimes are stated assumptions; what they remove is returned so
    the audit can show it.
    """
    logger = _log(logger)
    lifetimes = lifetimes_years or {}
    if operational.empty or not lifetimes:
        return operational, operational.iloc[0:0]
    commissioned = _parse_operational_dates(operational["operational_date"])
    lifetime = operational["technology"].map(lifetimes)
    horizon = pd.Timestamp(f"{int(modelled_year)}-01-01")
    expiry = pd.Series(pd.NaT, index=operational.index, dtype="datetime64[ns]")
    has_rule = lifetime.notna() & commissioned.notna()
    expiry[has_rule] = [
        date + pd.DateOffset(years=int(years))
        for date, years in zip(commissioned[has_rule], lifetime[has_rule])
    ]
    retired_mask = has_rule & (expiry <= horizon)
    retired = operational[retired_mask].copy()
    retired["retirement_expiry"] = expiry[retired_mask].dt.date
    if len(retired):
        logger.info(
            "Retired %d operational projects (%.0f MW) past their declared lifetime by %s",
            len(retired), retired["capacity_mw"].sum(), horizon.date(),
        )
    return operational[~retired_mask].copy(), retired


def load_named_fleet(operational_files: dict, pipeline_files: dict, modelled_year: int,
                     config: dict, logger=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (named fleet, retired projects) for the brownfield carriers.

    ``operational_files`` and ``pipeline_files`` map carrier -> site CSV. The
    inclusion rule comes from ``config``: ``statuses`` (default operational,
    under construction, awaiting construction), ``retirement_lifetimes_years``
    by carrier, and ``require_expected_date`` (default false: a consented
    project with no developer date is assumed online by the model year, which
    is CP30's own assumption for its pathway).
    """
    logger = _log(logger)
    statuses = set(config.get("statuses") or DEFAULT_STATUSES)
    unknown = statuses - set(MATURITY_RANK)
    if unknown:
        raise ValueError(f"Unknown brownfield statuses {sorted(unknown)}; allowed {sorted(MATURITY_RANK)}")
    year_end = pd.Timestamp(f"{int(modelled_year)}-12-31")

    frames = []
    retired_frames = []
    if "Operational" in statuses:
        operational = pd.concat(
            [_read_sites(path, carrier) for carrier, path in operational_files.items()
             if carrier in BROWNFIELD_CARRIERS],
            ignore_index=True,
        ) if operational_files else pd.DataFrame()
        if not operational.empty:
            dates = _parse_operational_dates(operational["operational_date"])
            operational = operational[(dates <= year_end) | dates.isna()].copy()
            operational["development_status"] = "Operational"
            operational, retired = retire_by_lifetime(
                operational, modelled_year, config.get("retirement_lifetimes_years"), logger
            )
            frames.append(operational)
            retired_frames.append(retired)

    pipeline_statuses = statuses - {"Operational"}
    if pipeline_statuses and pipeline_files:
        pipeline = pd.concat(
            [_read_sites(path, carrier) for carrier, path in pipeline_files.items()
             if carrier in BROWNFIELD_CARRIERS],
            ignore_index=True,
        )
        if not pipeline.empty:
            pipeline = pipeline[pipeline["development_status"].isin(pipeline_statuses)].copy()
            if "expected_operational_date" not in pipeline.columns:
                pipeline["expected_operational_date"] = pd.NaT
            expected = pd.to_datetime(pipeline["expected_operational_date"], errors="coerce")
            late = expected.notna() & (expected > year_end)
            if late.any():
                logger.info(
                    "Excluding %d pipeline projects (%.0f MW) whose developer-stated date is after %s",
                    int(late.sum()), pipeline.loc[late, "capacity_mw"].sum(), year_end.date(),
                )
            pipeline = pipeline[~late]
            if config.get("require_expected_date", False):
                undated = expected.reindex(pipeline.index).isna()
                logger.info("Excluding %d undated pipeline projects (require_expected_date)", int(undated.sum()))
                pipeline = pipeline[~undated]
            pipeline["operational_date"] = pipeline.get("expected_operational_date")
            frames.append(pipeline)

    if not frames:
        return pd.DataFrame(columns=NAMED_COLUMNS), pd.DataFrame()
    fleet = pd.concat(frames, ignore_index=True, sort=False)
    fleet["maturity_rank"] = fleet["development_status"].map(MATURITY_RANK)
    fleet["fleet_origin"] = "named"
    fleet["source_site_name"] = fleet["source_site_name"].fillna(fleet["site_name"])
    for column in NAMED_COLUMNS:
        if column not in fleet.columns:
            fleet[column] = np.nan
    retired = pd.concat(retired_frames, ignore_index=True) if retired_frames else pd.DataFrame()
    logger.info(
        "Named fleet: %d projects, %.0f MW (%s)", len(fleet), fleet["capacity_mw"].sum(),
        "; ".join(f"{k}: {v:,.0f} MW" for k, v in fleet.groupby("technology")["capacity_mw"].sum().items()),
    )
    return fleet, retired


# Declared named projects that cannot deliver in early modelled years even
# though REPD's status would admit them (e.g. the Lewis farms, which need the
# Western Isles HVDC link that CP30 does not include for 2030). Read through
# cp30_brownfield.named_exclusions_file; this is its default.
DEFAULT_NAMED_EXCLUSIONS_FILE = "data/renewables/cp30_named_fleet_exclusions.csv"


EXCLUSION_COLUMNS = ("repd_ref_id", "site_name", "first_modelled_year", "reason", "evidence")


def apply_named_fleet_exclusions(fleet: pd.DataFrame, exclusions_path, modelled_year: int,
                                 logger=None, extra: pd.DataFrame | None = None
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop named projects listed for exclusion before ``first_modelled_year``.

    Matched on REPD Ref ID. The dropped capacity is not lost from CP30: ES5
    reconciliation runs afterwards, so the region's proxy residual grows by the
    same amount. ``extra`` is a further table in the file's schema (the checked
    CfD table's ``not_built_refill`` sites: terminated or never-contracted
    projects, support_register_labels.checked_cfd_exclusions). Returns (kept
    fleet, excluded rows with their reason).
    """
    logger = _log(logger)
    if (not exclusions_path and (extra is None or extra.empty)) or fleet.empty:
        return fleet, fleet.iloc[0:0]
    tables, sources = [], []
    if exclusions_path:
        path = Path(exclusions_path)
        if not path.is_file():
            raise FileNotFoundError(f"Named-fleet exclusion file not found: {path}")
        tables.append(pd.read_csv(path, dtype=str).fillna(""))
        sources.append(str(path))
    if extra is not None and not extra.empty:
        tables.append(extra.astype(str).fillna(""))
        sources.append("checked CfD contracts")
    path = " + ".join(sources)
    table = pd.concat(tables, ignore_index=True)
    required = set(EXCLUSION_COLUMNS)
    if missing := required.difference(table.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    table["repd_ref_id"] = table["repd_ref_id"].str.strip().str.replace(r"\.0$", "", regex=True)
    if table["repd_ref_id"].duplicated().any():
        raise ValueError(f"{path} lists a REPD Ref ID twice")
    table["first_modelled_year"] = pd.to_numeric(table["first_modelled_year"], errors="raise").astype(int)
    active = table[table["first_modelled_year"] > int(modelled_year)].set_index("repd_ref_id")
    ids = fleet["repd_ref_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    mask = ids.isin(active.index)
    excluded = fleet[mask].copy()
    if len(excluded):
        excluded["exclusion_reason"] = ids[mask].map(active["reason"]).values
        excluded["exclusion_evidence"] = ids[mask].map(active["evidence"]).values
        excluded["first_modelled_year"] = ids[mask].map(active["first_modelled_year"]).values
        logger.info(
            "Excluded %d declared named projects (%.1f MW) not deliverable in %d (%s):",
            len(excluded), excluded["capacity_mw"].sum(), int(modelled_year), path,
        )
        for _, row in excluded.iterrows():
            logger.info("  - %s (REPD %s, %.1f MW): %s", row["site_name"], row["repd_ref_id"],
                        row["capacity_mw"], row["exclusion_reason"])
    unmatched = sorted(set(active.index) - set(ids))
    if unmatched:
        logger.warning("Named-fleet exclusions not found in the named fleet (REPD IDs): %s", unmatched)
    return fleet[~mask].copy(), excluded


# ---------------------------------------------------------------------------
# Offshore landing
# ---------------------------------------------------------------------------

def _nearest_bus_at_voltage(network, lat: float, lon: float, min_voltage_kv: float):
    """Nearest bus whose nominal voltage is at least ``min_voltage_kv``.

    The ETYS build stores bus ``x``/``y`` in OSGB metres and carries separate
    ``lon``/``lat`` columns; comparing a site's degrees against eastings sent
    a Norfolk farm to Cornwall. Degrees are used when the network has them,
    with ``x``/``y`` only as a fallback for networks built in degrees. A
    great-circle metric is unnecessary to choose among substations tens of
    kilometres apart, but longitude is scaled by cos(latitude) so east-west
    distances are not overstated at GB latitudes.
    """
    buses = network.buses
    candidates = buses[pd.to_numeric(buses["v_nom"], errors="coerce") >= float(min_voltage_kv)]
    if {"lon", "lat"}.issubset(candidates.columns) and candidates[["lon", "lat"]].notna().any().any():
        x_col, y_col = "lon", "lat"
    else:
        x_col, y_col = "x", "y"
    candidates = candidates.dropna(subset=[x_col, y_col])
    if candidates.empty:
        return None
    scale = float(np.cos(np.deg2rad(float(lat)))) if y_col == "lat" else 1.0
    distance = (
        ((candidates[x_col].astype(float) - float(lon)) * scale) ** 2
        + (candidates[y_col].astype(float) - float(lat)) ** 2
    )
    return str(distance.idxmin())


def _enforce_fallback_voltage(result: pd.DataFrame, mask: pd.Series, network, min_voltage_kv,
                              min_capacity_mw, logger) -> pd.DataFrame:
    """Move large fallback-landed offshore projects off sub-transmission buses.

    A nearest-bus placement for a 2 GW farm can land on a 132 kV busbar, which
    would manufacture a constraint the real connection agreement does not have.
    Small farms that genuinely connect at 132 kV are left alone.
    """
    if min_voltage_kv is None or "v_nom" not in network.buses.columns:
        return result
    voltage = result.loc[mask, "bus"].map(pd.to_numeric(network.buses["v_nom"], errors="coerce"))
    large = result.loc[mask, "capacity_mw"].astype(float) >= float(min_capacity_mw)
    to_move = mask.copy()
    to_move.loc[mask] = (voltage < float(min_voltage_kv)) & large
    for idx in result.index[to_move]:
        replacement = _nearest_bus_at_voltage(
            network, result.at[idx, "lat"], result.at[idx, "lon"], min_voltage_kv
        )
        if replacement is None:
            continue
        logger.info(
            "Offshore fallback: %s (%.0f MW) moved from %s (%.0f kV) to nearest >=%.0f kV bus %s",
            result.at[idx, "site_name"], float(result.at[idx, "capacity_mw"]), result.at[idx, "bus"],
            float(voltage.loc[idx]), float(min_voltage_kv), replacement,
        )
        result.at[idx, "bus"] = replacement
        result.at[idx, "landing_evidence"] = f"{result.at[idx, 'landing_evidence']}_nearest_transmission_bus"
    return result


def apply_offshore_landing_crosswalk(fleet: pd.DataFrame, crosswalk_path, network,
                                     policy: str = "nearest_bus_fallback", logger=None,
                                     fallback_min_voltage_kv=275.0,
                                     fallback_min_voltage_capacity_mw=300.0) -> pd.DataFrame:
    """Place named offshore projects at their evidenced landing bus.

    The crosswalk is keyed by REPD Ref ID and records the landing bus and the
    evidence for it. A row with an empty ``landing_bus`` documents a known gap
    (a substation the ETYS network does not represent) so the fallback is a
    recorded decision rather than silence. Projects with no row keep the
    nearest-bus placement and are flagged ``landing_evidence = none``; those at
    or above ``fallback_min_voltage_capacity_mw`` are moved to the nearest bus
    at ``fallback_min_voltage_kv`` or above.
    """
    logger = _log(logger)
    if policy not in LANDING_POLICIES:
        raise ValueError(f"Offshore landing policy must be one of {LANDING_POLICIES}, got {policy!r}")
    result = fleet.copy()
    result["landing_evidence"] = np.where(result["technology"].eq("wind_offshore"), "none", "not_applicable")
    offshore = result["technology"].eq("wind_offshore")
    if not offshore.any():
        return result
    if crosswalk_path is None or not Path(crosswalk_path).is_file():
        if policy == "error":
            raise FileNotFoundError(f"Offshore landing crosswalk required but missing: {crosswalk_path}")
        logger.warning("No offshore landing crosswalk; named offshore projects use nearest-bus placement")
        return _enforce_fallback_voltage(result, offshore, network, fallback_min_voltage_kv,
                                         fallback_min_voltage_capacity_mw, logger)

    crosswalk = pd.read_csv(crosswalk_path, dtype=str).fillna("")
    required = {"repd_ref_id", "landing_bus", "evidence", "source"}
    if missing := required.difference(crosswalk.columns):
        raise ValueError(f"Offshore landing crosswalk is missing columns: {sorted(missing)}")
    crosswalk["repd_ref_id"] = crosswalk["repd_ref_id"].str.strip()
    if crosswalk["repd_ref_id"].duplicated().any():
        raise ValueError("Offshore landing crosswalk has duplicate REPD Ref IDs")
    unknown = sorted(set(crosswalk.loc[crosswalk.landing_bus.ne(""), "landing_bus"]) - set(network.buses.index))
    if unknown:
        raise ValueError(f"Offshore landing crosswalk names buses absent from this network: {unknown}")
    lookup = crosswalk.set_index("repd_ref_id")

    ids = result.loc[offshore, "repd_ref_id"].astype(str).str.strip()
    matched = ids.isin(lookup.index)
    for idx, ref in ids[matched].items():
        row = lookup.loc[ref]
        if row["landing_bus"]:
            result.at[idx, "bus"] = row["landing_bus"]
            # The same marker the explicit CP30 connection allocation uses, so
            # the later BMU/voltage heuristic does not undo an evidenced landing.
            result.at[idx, "data_source"] = "CP30_connection_allocation"
            result.at[idx, "connection_allocation_source"] = row["source"]
            result.at[idx, "landing_evidence"] = row["evidence"] or "crosswalk"
        else:
            result.at[idx, "landing_evidence"] = row["evidence"] or "not_represented_in_etys"
            result.at[idx, "connection_allocation_source"] = row["source"]
    unmatched_mask = offshore & result["landing_evidence"].isin(["none", "not_represented_in_etys"])
    unmatched = result.loc[offshore & result["landing_evidence"].eq("none")]
    if len(unmatched):
        message = (
            f"{len(unmatched)} named offshore projects ({unmatched['capacity_mw'].sum():,.0f} MW) have "
            "no landing crosswalk row and are placed at the nearest bus to their REPD coordinates"
        )
        if policy == "error":
            raise ValueError(message + f": {sorted(unmatched['site_name'])}")
        logger.warning(message)
    return _enforce_fallback_voltage(result, unmatched_mask, network, fallback_min_voltage_kv,
                                     fallback_min_voltage_capacity_mw, logger)


# ---------------------------------------------------------------------------
# Reconciliation with ES5
# ---------------------------------------------------------------------------

def _has_contract(frame: pd.DataFrame) -> pd.Series:
    """A named project REPD records an allocation round for (the modelled AR8 round is no contract yet)."""
    rounds = frame.get("cfd_round", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    return rounds.ne("") & rounds.ne(MODELLED_ROUND) & rounds.str.lower().ne("nan")


def _trim_region(group: pd.DataFrame, excess: float, contracted_last: bool = False) -> tuple[pd.DataFrame, list]:
    """Remove ``excess`` MW from a region's named pipeline, least mature first.

    Whole projects are dropped in maturity order (Application Submitted before
    Awaiting Construction before Under Construction; most recently consented
    first within a status); the last project needed is derated so the region
    fits exactly. Operational projects are never touched here. With
    ``contracted_last`` every pipeline project without a CfD round goes before
    any project holding one, so a contracted project is trimmed only when the
    uncontracted pipeline is exhausted.
    """
    candidates = group[group["maturity_rank"].gt(0)].copy()
    dates = pd.to_datetime(candidates.get("planning_granted_date"), errors="coerce")
    candidates["_sort_date"] = dates.fillna(pd.Timestamp("2100-01-01"))
    candidates["_contracted"] = _has_contract(candidates).astype(int) if contracted_last else 0
    order = candidates.sort_values(["_contracted", "maturity_rank", "_sort_date"],
                                   ascending=[True, False, False]).index
    trimmed = []
    remaining = float(excess)
    kept = group.copy()
    for idx in order:
        if remaining <= 1e-9:
            break
        capacity = float(kept.at[idx, "capacity_mw"])
        removed = min(capacity, remaining)
        record = group.loc[idx].to_dict()
        record["trimmed_mw"] = removed
        record["trim_action"] = "dropped" if removed >= capacity - 1e-9 else "derated"
        trimmed.append(record)
        if removed >= capacity - 1e-9:
            kept = kept.drop(index=idx)
        else:
            kept.at[idx, "capacity_mw"] = capacity - removed
        remaining -= removed
    return kept, trimmed


#: Columns an excluded project's row keeps when it becomes an anonymous refill
#: row: where it is (bus, region, coordinates for the resource region) and how
#: much. Everything that names it or its support (site name, REPD ID, CfD
#: round, ROC banding, status, dates) is dropped.
REFILL_KEEP_COLUMNS = ("technology", "capacity_mw", "bus", "cp30_region", "lat", "lon")


def build_landing_refill(placed: pd.DataFrame, modelled_year: int, logger=None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn excluded not-built projects into anonymous proxy rows at their own landing bus.

    ``placed`` is the excluded rows after the named fleet's own placement
    (nearest bus, offshore landing crosswalk, CP30 region), so each carries the
    bus and region the project would have had. A row with no bus is left to
    the regional proxy. The rest are summed per (carrier, region, bus) into
    proxy increment rows: no site name, REPD ID or support label, so the
    increment's support treatment (future CfD fractions, vintage schedule)
    applies to them as it did to the pro-rata refill. Coordinates are the
    capacity-weighted mean, which keeps the resource region the project had.

    Returns (refill rows, per-project detail for the audit).
    """
    logger = _log(logger)
    if placed is None or placed.empty:
        return pd.DataFrame(columns=list(REFILL_KEEP_COLUMNS)), pd.DataFrame()
    rows = placed.copy()
    has_bus = rows["bus"].notna() & rows["bus"].astype(str).str.strip().ne("")
    detail = rows.reindex(columns=["site_name", "repd_ref_id", "technology", "capacity_mw", "bus",
                                   "cp30_region", "landing_evidence"]).copy()
    detail["refill"] = np.where(has_bus, "at_excluded_landing", "regional_proxy")
    rows = rows[has_bus]
    if rows.empty:
        return pd.DataFrame(columns=list(REFILL_KEEP_COLUMNS)), detail
    rows["capacity_mw"] = pd.to_numeric(rows["capacity_mw"], errors="raise").astype(float)
    for column in ("lat", "lon"):
        rows[column] = pd.to_numeric(rows.get(column), errors="coerce")
        rows[f"_w_{column}"] = rows[column] * rows["capacity_mw"]
    grouped = rows.groupby(["technology", "cp30_region", "bus"], sort=True)
    refill = grouped[["capacity_mw", "_w_lat", "_w_lon"]].sum().reset_index()
    refill["lat"] = refill["_w_lat"] / refill["capacity_mw"]
    refill["lon"] = refill["_w_lon"] / refill["capacity_mw"]
    refill = refill.drop(columns=["_w_lat", "_w_lon"])
    region_slug = refill["cp30_region"].astype(str).str.replace(" ", "_", regex=False)
    refill["site_name"] = ("CP30_increment_" + refill["technology"] + "_refill_" + region_slug + "_"
                           + refill["bus"].astype(str) + f"_{int(modelled_year)}")
    refill["fleet_origin"] = "increment"
    # The offshore rows hold an evidenced landing (crosswalk or TEC), so the
    # generic BMU/voltage heuristics must not move them, as for a landed named
    # project. Onshore and solar rows stay movable like any proxy increment.
    offshore = refill["technology"].eq("wind_offshore")
    refill["data_source"] = np.where(offshore, "CP30_connection_allocation", "CP30_refill_at_excluded_site")
    refill["connection_allocation_source"] = np.where(
        offshore,
        "Refill of not-built offshore capacity at the excluded project's evidenced landing bus "
        "(repd_offshore_landing_crosswalk / TEC register)",
        "Refill of not-built capacity at the excluded site's nearest bus (REPD coordinates)",
    )
    refill["spatial_proxy_source"] = (
        "CP30 ES5 refill of excluded not-built capacity, placed at the excluded project's own "
        "connection point; anonymous proxy capacity"
    )
    refill["cp30_spatial_method"] = "refill_at_excluded_landing"
    for _, row in refill.iterrows():
        logger.info("Refill at excluded landing: %.1f MW %s at %s (%s)", row["capacity_mw"],
                    row["technology"], row["bus"], row["cp30_region"])
    return refill.reset_index(drop=True), detail


SMALL_SCALE_SOLAR_METHOD = "small_scale_solar_reserved"


def build_small_scale_solar_reservation(rows: pd.DataFrame, modelled_year: int, logger=None) -> pd.DataFrame:
    """FES small-scale solar (G98/G83, Gen_BB013) as proxy rows held at each GSP node.

    ``rows`` has one row per FES GSP with ``capacity_mw``, ``bus`` (the GSP's
    demand node), ``cp30_region``, ``lat``/``lon`` and ``gsp``. Rooftop solar
    connects at the GSP it serves, so it is placed there before the named REPD
    pipeline fills the region's ES5 envelope (``reconcile_with_es5(reserved=)``).
    The rows are proxy increment, not CfD-eligible: the subsidy step labels
    them merchant (``SMALL_SCALE_SOLAR_METHOD``), so they take no allocation round.
    """
    logger = _log(logger)
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["technology", "capacity_mw", "bus", "cp30_region"])
    frame = rows.copy()
    frame["capacity_mw"] = pd.to_numeric(frame["capacity_mw"], errors="raise").astype(float)
    frame = frame[frame["capacity_mw"] > 0].copy()
    frame["technology"] = "solar_pv"
    slug = frame["gsp"].astype(str).str.replace(r"[^A-Za-z0-9]+", "_", regex=True).str.strip("_")
    frame["site_name"] = "CP30_increment_solar_pv_small_scale_" + slug + f"_{int(modelled_year)}"
    frame["fleet_origin"] = "increment"
    frame["data_source"] = "CP30_small_scale_solar_gsp"
    frame["connection_allocation_source"] = "FES small-scale solar (Gen_BB013) at its GSP's demand node"
    frame["spatial_proxy_source"] = "FES small-scale (G98/G83) solar by GSP, held before the named pipeline"
    frame["cp30_spatial_method"] = SMALL_SCALE_SOLAR_METHOD
    logger.info("Small-scale solar reservation: %.0f MW over %d GSPs (%s)", frame["capacity_mw"].sum(), len(frame),
                frame.groupby("cp30_region")["capacity_mw"].sum().round(0).to_dict())
    return frame.reset_index(drop=True)


def _place_refill(refill: pd.DataFrame, residuals: dict, covered: set, logger
                  ) -> tuple[pd.DataFrame, dict]:
    """Fit refill rows inside each region's proxy residual; return (rows kept, MW per key).

    A refill row stands in for capacity the excluded project would have held,
    so it takes the region's proxy residual first; the pro-rata proxy fills
    only what remains. A region whose residual is smaller than its refill
    (its named fleet already fills ES5) scales its refill rows down to fit,
    which is exactly what the pro-rata refill would have delivered there.
    Rows for a (carrier, region) outside ES5 pass through untouched.
    """
    if refill is None or refill.empty:
        return pd.DataFrame(), {}
    rows = refill.copy()
    rows["capacity_mw"] = rows["capacity_mw"].astype(float)
    placed = {}
    for key, group in rows.groupby(["technology", "cp30_region"], sort=True):
        if key not in covered:
            continue
        wanted = float(group["capacity_mw"].sum())
        available = float(residuals.get(key, 0.0))
        take = min(wanted, available)
        if take < wanted - 1e-6:
            logger.warning(
                "Refill at excluded landing: %s in %s wants %.1f MW but the proxy residual is %.1f MW; "
                "refill rows scaled to fit", key[0], key[1], wanted, available,
            )
        rows.loc[group.index, "capacity_mw"] = group["capacity_mw"] * (take / wanted if wanted > 0 else 0.0)
        residuals[key] = available - take
        placed[key] = take
    rows = rows[rows["capacity_mw"] > 1e-9]
    return rows, placed


def reconcile_with_es5(named: pd.DataFrame, es5_targets: pd.DataFrame, policy: str = "trim_least_mature",
                       logger=None, refill: pd.DataFrame | None = None,
                       reserved: pd.DataFrame | None = None, contracted_last: bool = False,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the named fleet inside the ES5 regional envelopes.

    ``es5_targets`` has columns carrier, cp30_region, capacity_mw. Returns
    (named fleet after trimming, residual targets for the proxy increment,
    trimmed projects, regional audit).

    An operational fleet that alone exceeds a regional target is never trimmed.
    The excess is carried nationally instead: it reduces the positive residuals
    of the other regions for that carrier in proportion, so the ES1 national
    total is preserved while the regional split departs from ES5 by exactly the
    amount the built fleet forces. The audit records both.

    ``refill`` (``build_landing_refill``) is proxy capacity already placed at
    excluded projects' own landing buses. It is counted on the named side of
    the region after the named fleet is fitted, so the pro-rata proxy residual
    does not refill the same MW a second time; it is returned in the kept
    fleet (``fleet_origin`` increment) and never trims a named project.

    ``reserved`` (``build_small_scale_solar_reservation``) is proxy capacity
    held for a (carrier, region) BEFORE the named fleet fills it: FES
    small-scale (G98/G83) solar at its GSP node. The named pipeline is fitted
    to the target less the reservation (trimmed least mature first, and with
    ``contracted_last`` uncontracted projects before any holding a CfD round).
    A reservation never forces operational plant out: it is capped at the
    target less the region's operational named MW, its rows scaled to fit.
    The kept fleet carries the reserved rows; named + reserved + refill +
    proxy residual = ES5, region by region.
    """
    logger = _log(logger)
    if policy not in OVERSHOOT_POLICIES:
        raise ValueError(f"Overshoot policy must be one of {OVERSHOOT_POLICIES}, got {policy!r}")
    if "cp30_region" not in named.columns:
        raise ValueError("Named fleet needs a cp30_region column before reconciliation")

    kept_parts, trimmed_records, audit_rows = [], [], []
    residuals = {}
    operational_excess = {}
    targets = es5_targets[es5_targets["carrier"].isin(BROWNFIELD_CARRIERS)]
    target_index = targets.set_index(["carrier", "cp30_region"])["capacity_mw"]

    reserved_rows = reserved.copy() if reserved is not None and len(reserved) else pd.DataFrame()
    if len(reserved_rows):
        reserved_rows["capacity_mw"] = pd.to_numeric(reserved_rows["capacity_mw"], errors="raise").astype(float)
    reserved_parts = []
    for (carrier, region), target in target_index.items():
        target = float(target)
        group = named[named["technology"].eq(carrier) & named["cp30_region"].eq(region)]
        named_mw = float(group["capacity_mw"].sum())
        operational_mw = float(group.loc[group["maturity_rank"].eq(0), "capacity_mw"].sum())
        held = (reserved_rows[reserved_rows["technology"].eq(carrier) & reserved_rows["cp30_region"].eq(region)]
                if len(reserved_rows) else pd.DataFrame())
        reserve_wanted = float(held["capacity_mw"].sum()) if len(held) else 0.0
        reserve = min(reserve_wanted, max(target - operational_mw, 0.0))
        if reserve_wanted > 0:
            held = held.copy()
            held["capacity_mw"] = held["capacity_mw"] * (reserve / reserve_wanted)
            reserved_parts.append(held[held["capacity_mw"] > 1e-9])
        effective = target - reserve
        trimmed_mw = 0.0
        if named_mw > effective + 1e-6:
            excess = named_mw - effective
            if policy == "error":
                raise ValueError(
                    f"Named {carrier} in {region} is {named_mw:,.0f} MW against a CP30 ES5 target of "
                    f"{target:,.0f} MW ({reserve:,.0f} MW reserved); overshoot policy is 'error'"
                )
            trimmable = named_mw - operational_mw
            group, trimmed = _trim_region(group, min(excess, trimmable), contracted_last=contracted_last)
            for record in trimmed:
                record["cp30_region"] = region
                record["trim_reason"] = "small_scale_reservation" if named_mw <= target + 1e-6 else "es5_overshoot"
            trimmed_records.extend(trimmed)
            trimmed_mw = float(sum(record["trimmed_mw"] for record in trimmed))
            if excess > trimmable + 1e-6:
                # The reservation is capped at target - operational, so only built plant can still exceed.
                operational_excess[(carrier, region)] = max(operational_mw - target, 0.0)
                if operational_excess[(carrier, region)] <= 1e-6:
                    operational_excess.pop((carrier, region))
        kept_parts.append(group)
        residual = max(target - float(group["capacity_mw"].sum()) - reserve, 0.0)
        residuals[(carrier, region)] = residual
        audit_rows.append({
            "carrier": carrier, "cp30_region": region, "es5_target_mw": target,
            "named_mw_before_trim": named_mw, "named_operational_mw": operational_mw,
            "named_pipeline_mw": float(group.loc[group["maturity_rank"].gt(0), "capacity_mw"].sum()),
            "trimmed_mw": trimmed_mw, "operational_excess_mw": operational_excess.get((carrier, region), 0.0),
            "reserved_small_scale_mw": reserve,
            "residual_for_proxy_mw": residual,
        })

    # Carry any built-fleet excess nationally: reduce the other regions' residuals
    # for the same carrier in proportion so the ES1 total is preserved.
    for carrier in BROWNFIELD_CARRIERS:
        excess = sum(v for (c, _), v in operational_excess.items() if c == carrier)
        if excess <= 0:
            continue
        pool = {key: v for key, v in residuals.items() if key[0] == carrier and v > 0}
        available = sum(pool.values())
        if excess > available + 1e-6:
            raise ValueError(
                f"Operational {carrier} exceeds CP30 nationally by {excess - available:,.0f} MW; the "
                "named fleet cannot be reconciled to ES1 without retiring built plant"
            )
        for key, value in pool.items():
            residuals[key] = value - excess * value / available
        logger.warning(
            "Operational %s exceeds ES5 in %d region(s) by %.0f MW; the excess reduces other regions' "
            "proxy residual so the national total holds",
            carrier, sum(1 for (c, _) in operational_excess if c == carrier), excess,
        )
    # Named rows outside every ES5 region/carrier pair pass through untouched.
    covered = set(target_index.index)
    refill_rows, refill_placed = _place_refill(refill, residuals, covered, logger)
    for row in audit_rows:
        key = (row["carrier"], row["cp30_region"])
        row["residual_for_proxy_mw"] = residuals[key]
        row["refill_at_landing_mw"] = refill_placed.get(key, 0.0)
        row["named_share_of_target"] = (
            (row["es5_target_mw"] - row["residual_for_proxy_mw"] - row["refill_at_landing_mw"]
             - row["reserved_small_scale_mw"])
            / row["es5_target_mw"]
            if row["es5_target_mw"] > 0 else np.nan
        )

    outside = named[[
        (t, r) not in covered for t, r in zip(named["technology"], named["cp30_region"])
    ]]
    kept = pd.concat([*kept_parts, outside], ignore_index=True) if kept_parts else named.copy()
    if len(refill_rows):
        kept = pd.concat([kept, refill_rows], ignore_index=True, sort=False)
    if reserved_parts:
        kept = pd.concat([kept, *reserved_parts], ignore_index=True, sort=False)
    residual_targets = pd.DataFrame(
        [{"carrier": c, "cp30_region": r, "capacity_mw": v} for (c, r), v in residuals.items()]
    )
    trimmed_frame = pd.DataFrame(trimmed_records)
    audit = pd.DataFrame(audit_rows)
    if len(trimmed_frame):
        logger.info("Trimmed %d pipeline projects (%.0f MW) to fit CP30 ES5 regional totals",
                    len(trimmed_frame), trimmed_frame["trimmed_mw"].sum())
    return kept, residual_targets, trimmed_frame, audit


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def build_fleet_audit(final_sites: pd.DataFrame, regional_audit: pd.DataFrame,
                      retired: pd.DataFrame, trimmed: pd.DataFrame,
                      excluded: pd.DataFrame | None = None) -> dict:
    """Summarise what the brownfield build did, per carrier and overall."""
    sites = final_sites.copy()
    origin = sites.get("fleet_origin", pd.Series("increment", index=sites.index)).fillna("increment")
    sites["fleet_origin"] = origin
    carriers = {}
    for carrier in BROWNFIELD_CARRIERS:
        sub = sites[sites["technology"].eq(carrier)]
        named = sub[sub["fleet_origin"].eq("named")]
        total = float(sub["capacity_mw"].sum())
        named_mw = float(named["capacity_mw"].sum())
        cfd = named[named.get("support_type", pd.Series(index=named.index)).eq("CfD")] if len(named) else named
        entry = {
            "total_mw": total,
            "named_mw": named_mw,
            "increment_mw": total - named_mw,
            "named_share": named_mw / total if total else None,
            "named_projects": int(len(named)),
            "named_by_status_mw": {
                str(k): float(v) for k, v in named.groupby("development_status")["capacity_mw"].sum().items()
            } if len(named) else {},
            "named_cfd_mw": float(cfd["capacity_mw"].sum()) if len(cfd) else 0.0,
            # The modelled AR8 round is the residual, not a published contract round.
            "named_cfd_with_round_share": (
                float(cfd.loc[~cfd["cfd_round"].fillna("").astype(str).str.strip().isin(["", MODELLED_ROUND]),
                              "capacity_mw"].sum() / cfd["capacity_mw"].sum())
                if len(cfd) and cfd["capacity_mw"].sum() > 0 else None
            ),
            "named_roc_mw": float(named.loc[
                named.get("support_type", pd.Series(index=named.index)).eq("ROC"), "capacity_mw"].sum())
                if len(named) else 0.0,
            "profile_basis": {"named": "site_coordinates", "increment": "carrier_fleet_average"},
        }
        method = sub.get("cp30_spatial_method", pd.Series("", index=sub.index)).fillna("")
        refill = sub[method.eq("refill_at_excluded_landing")]
        small = sub[method.eq(SMALL_SCALE_SOLAR_METHOD)]
        if len(small):
            entry["small_scale_reserved_mw"] = float(small["capacity_mw"].sum())
        if len(refill):
            entry["refill_at_landing_mw"] = float(refill["capacity_mw"].sum())
            entry["refill_at_landing_by_bus_mw"] = {
                str(k): float(v) for k, v in refill.groupby("bus")["capacity_mw"].sum().items()
            }
        if carrier == "wind_offshore" and len(named):
            evidence = named.get("landing_evidence", pd.Series("none", index=named.index)).fillna("none")
            entry["landing_evidence_mw"] = {
                str(k): float(v) for k, v in named.groupby(evidence)["capacity_mw"].sum().items()
            }
        carriers[carrier] = entry
    return {
        "method": "brownfield_named_fleet_plus_proxy_increment",
        "carriers": carriers,
        "retired_projects": int(len(retired)),
        "retired_mw": float(retired["capacity_mw"].sum()) if len(retired) else 0.0,
        "excluded_projects": int(len(excluded)) if excluded is not None else 0,
        "excluded_mw": float(excluded["capacity_mw"].sum()) if excluded is not None and len(excluded) else 0.0,
        "excluded_projects_detail": (
            excluded.reindex(columns=["site_name", "repd_ref_id", "technology", "capacity_mw",
                                      "first_modelled_year", "exclusion_reason"])
            .astype(str).to_dict("records") if excluded is not None and len(excluded) else []
        ),
        "trimmed_projects": int(len(trimmed)),
        "trimmed_mw": float(trimmed["trimmed_mw"].sum()) if len(trimmed) else 0.0,
        "regional": regional_audit.to_dict("records") if len(regional_audit) else [],
    }
