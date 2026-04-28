"""
NESO Constraint Validation — Compare Model BM vs NESO Thermal Constraints

Compares PyPSA-GB two-stage market results against NESO published data:
  1. Thermal Constraint Costs  — model total BM cost vs NESO boundary costs
  2. Day-Ahead Constraint Flows — model boundary flows vs NESO DA flows/limits
  3. Boundary congestion hours  — model congested hours vs NESO at-limit hours

Data sources:
  - NESO Thermal Constraint Costs: https://www.neso.energy/data-portal/thermal-constraint-costs
  - NESO DA Constraint Flows: https://www.neso.energy/data-portal/day-ahead-constraint-flows-and-limits

Inputs:
  - Model: balancing.nc (line flows), constraint_costs.csv, congestion.csv
  - NESO: downloaded/cached XLSX/CSV from NESO API
  - Boundary mapping: data/network/neso_boundary_mapping.yaml

Outputs:
  - {scenario}_neso_validation.csv  — tabular comparison metrics
  - {scenario}_neso_validation.html — multi-panel comparison dashboard

Called by Snakemake rule `validate_neso_constraints` in rules/market.smk.
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pypsa
import requests
import yaml

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("validate_neso")


# ═══════════════════════════════════════════════════════════════════════════════
# NESO API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

NESO_THERMAL_DATASET_ID = "thermal-constraint-costs"
NESO_DA_FLOWS_DATASET_ID = "day-ahead-constraint-flows-and-limits"
NESO_DA_FLOWS_RESOURCE_ID = "38a18ec1-9e40-465d-93fb-301e80fd1352"

# Maps financial year label to (start_month, start_year_offset)
# FY "19-20" = Apr 2019 – Mar 2020; data may start later (Aug 2019 for 19-20)
_FY_RESOURCE_NAMES = {
    # resource_name_pattern: (fy_start_year, fy_end_year)
    "19-20": (2019, 2020),
    "20-21": (2020, 2021),
    "21-22": (2021, 2022),
    "22-23": (2022, 2023),
    "23-24": (2023, 2024),
    "24-25": (2024, 2025),
    "25-26": (2025, 2026),
    "26-27": (2026, 2027),
}


def _financial_years_for_range(start_date, end_date):
    """Return list of FY labels (e.g. '19-20') that overlap [start_date, end_date]."""
    needed = []
    for fy_label, (fy_start_yr, fy_end_yr) in _FY_RESOURCE_NAMES.items():
        # FY runs Apr fy_start_yr to Mar fy_end_yr
        fy_start = pd.Timestamp(fy_start_yr, 4, 1)
        fy_end = pd.Timestamp(fy_end_yr, 3, 31, 23, 59, 59)
        if fy_start <= end_date and fy_end >= start_date:
            needed.append(fy_label)
    return needed


def _get_neso_resource_urls(dataset_id, logger):
    """Fetch resource download URLs from NESO CKAN API."""
    url = f"https://api.neso.energy/api/3/action/datapackage_show?id={dataset_id}"
    logger.info(f"Fetching NESO API: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    resources = []
    for res in data["result"]["resources"]:
        resources.append(
            {
                "name": res.get("name", ""),
                "path": res.get("path", ""),
                "format": res.get("format", "").upper(),
                "id": res.get("id", ""),
            }
        )
    return resources


# ═══════════════════════════════════════════════════════════════════════════════
# NESO DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def _load_thermal_costs(start_date, end_date, cache_dir, logger):
    """
    Load NESO thermal constraint costs for a date range.

    Downloads financial-year files from NESO API if not cached locally.
    Returns a DataFrame with columns: date, constraint_group, daily_cost_gbp.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fy_labels = _financial_years_for_range(start_date, end_date)
    logger.info(
        f"Thermal costs: need FY files {fy_labels} for "
        f"{start_date.date()} to {end_date.date()}"
    )

    frames = []
    for fy in fy_labels:
        # Check local cache first
        xlsx_path = cache_dir / f"thermal_constraint_costs_{fy}.xlsx"
        csv_path = cache_dir / f"thermal_constraint_costs_{fy}.csv"

        if xlsx_path.exists():
            logger.info(f"  Loading cached: {xlsx_path.name}")
            df = pd.read_excel(xlsx_path)
        elif csv_path.exists():
            logger.info(f"  Loading cached: {csv_path.name}")
            df = pd.read_csv(csv_path)
        else:
            # Download from NESO API
            logger.info(f"  Downloading FY {fy} from NESO API...")
            resources = _get_neso_resource_urls(NESO_THERMAL_DATASET_ID, logger)
            downloaded = False
            for res in resources:
                if fy in res["name"]:
                    logger.info(f"  Fetching: {res['name']} [{res['format']}]")
                    r = requests.get(res["path"], timeout=120)
                    r.raise_for_status()
                    if res["format"] in ("XLSX", "XLS"):
                        out_path = cache_dir / f"thermal_constraint_costs_{fy}.xlsx"
                        out_path.write_bytes(r.content)
                        df = pd.read_excel(out_path)
                    else:
                        out_path = cache_dir / f"thermal_constraint_costs_{fy}.csv"
                        out_path.write_bytes(r.content)
                        df = pd.read_csv(out_path)
                    downloaded = True
                    break
            if not downloaded:
                logger.warning(f"  FY {fy} not found in NESO API — skipping")
                continue

        # Normalise column names
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if "date" in cl:
                col_map[c] = "date"
            elif "group" in cl or "constraint" in cl:
                col_map[c] = "constraint_group"
            elif "cost" in cl:
                col_map[c] = "daily_cost_gbp"
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df[["date", "constraint_group", "daily_cost_gbp"]])

    if not frames:
        logger.warning("No thermal cost data loaded")
        return pd.DataFrame(columns=["date", "constraint_group", "daily_cost_gbp"])

    all_costs = pd.concat(frames, ignore_index=True)
    # Filter to requested date range
    mask = (all_costs["date"] >= start_date) & (all_costs["date"] <= end_date)
    filtered = all_costs.loc[mask].copy()
    logger.info(
        f"  Thermal costs: {len(filtered)} daily records in range "
        f"({filtered['date'].min().date()} to {filtered['date'].max().date()})"
    )
    return filtered


def _load_da_flows(start_date, end_date, cache_dir, logger):
    """
    Load NESO Day-Ahead constraint flows for a date range.

    Uses a single large CSV (all history). Downloads if not cached.
    Returns a DataFrame with columns: constraint_group, datetime, limit_mw, flow_mw.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / "day_ahead_constraint_flows_limits.csv"

    if local_path.exists():
        logger.info(f"Loading cached DA flows: {local_path.name}")
        df = pd.read_csv(local_path, low_memory=False)
    else:
        logger.info("Downloading DA constraint flows from NESO API...")
        resources = _get_neso_resource_urls(NESO_DA_FLOWS_DATASET_ID, logger)
        downloaded = False
        for res in resources:
            if res["format"] == "CSV":
                logger.info(f"  Fetching: {res['name']} ({res['id']})")
                r = requests.get(res["path"], timeout=300)
                r.raise_for_status()
                local_path.write_bytes(r.content)
                df = pd.read_csv(local_path, low_memory=False)
                downloaded = True
                break
        if not downloaded:
            logger.warning("DA flows CSV not found in NESO API")
            return pd.DataFrame(
                columns=["constraint_group", "datetime", "limit_mw", "flow_mw"]
            )

    # Normalise columns
    date_col = [c for c in df.columns if "Date" in c][0]
    df = df.rename(
        columns={
            "Constraint Group": "constraint_group",
            date_col: "datetime",
            "Limit (MW)": "limit_mw",
            "Flow (MW)": "flow_mw",
        }
    )
    # Drop unnamed columns
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]

    # Parse dates (ISO8601 mixed precision — some have seconds, some don't)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df["datetime"] = df["datetime"].dt.tz_localize(None)  # drop tz for comparison

    # Filter to date range
    mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
    filtered = df.loc[mask].copy()
    if not filtered.empty:
        requested_days = pd.date_range(
            start_date.normalize(), end_date.normalize(), freq="D"
        ).date
        available_days = set(filtered["datetime"].dt.date.unique())
        missing_days = [day for day in requested_days if day not in available_days]
        if missing_days:
            logger.warning(
                f"  DA flows cache missing requested day(s): {missing_days}. "
                f"Validation will only compare overlapping timestamps."
            )
    logger.info(
        f"  DA flows: {len(filtered)} records for {filtered['constraint_group'].nunique()} "
        f"boundaries in range"
    )
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDARY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════


def _load_boundary_mapping(mapping_path, logger, boundary_include=None):
    """
    Load NESO boundary → PyPSA line mapping from YAML.

        Returns dict:
            {
                boundary_name: {
                    "lines": [...],
                    "flow_groups": {"positive": [...], "negative": [...]},
                }
            }
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        logger.warning(f"Boundary mapping not found: {mapping_path}")
        return {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    include_set = None
    if boundary_include:
        include_set = {str(name).strip() for name in boundary_include}

    boundaries = {}
    for name, info in data.get("boundaries", {}).items():
        if include_set is not None and name not in include_set:
            continue
        lines = list(dict.fromkeys(info.get("lines", [])))
        flow_groups = info.get("flow_groups", {}) or {}
        positive = list(dict.fromkeys(flow_groups.get("positive", [])))
        negative = list(dict.fromkeys(flow_groups.get("negative", [])))
        transformers = list(dict.fromkeys(info.get("transformers", [])))
        transformer_groups = info.get("transformer_flow_groups", {}) or {}
        positive_transformers = list(dict.fromkeys(transformer_groups.get("positive", [])))
        negative_transformers = list(dict.fromkeys(transformer_groups.get("negative", [])))

        if positive or negative:
            signed_union = list(dict.fromkeys(positive + negative))
            if not lines:
                lines = signed_union
            else:
                lines = list(dict.fromkeys(lines + signed_union))

        if positive_transformers or negative_transformers:
            transformer_union = list(dict.fromkeys(
                positive_transformers + negative_transformers
            ))
            if not transformers:
                transformers = transformer_union
            else:
                transformers = list(dict.fromkeys(transformers + transformer_union))

        if not lines and not transformers:
            continue

        if not positive and not negative:
            positive = lines.copy()
        if transformers and not positive_transformers and not negative_transformers:
            positive_transformers = transformers.copy()

        link_defs = list(info.get("links", []))  # list of {name, sign} dicts

        boundaries[name] = {
            "lines": lines,
            "transformers": transformers,
            "flow_groups": {
                "positive": positive,
                "negative": negative,
            },
            "transformer_flow_groups": {
                "positive": positive_transformers,
                "negative": negative_transformers,
            },
            "links": link_defs,
        }
        logger.info(
            f"  Boundary {name}: {len(lines)} lines + {len(transformers)} transformers "
            f"+ {len(link_defs)} links "
            f"(signed lines +{len(positive)} / -{len(negative)}, "
            f"transformers +{len(positive_transformers)} / -{len(negative_transformers)})"
        )
    return boundaries


def _get_boundary_lines(boundary_def):
    """Return the full union of lines for a boundary definition."""
    if isinstance(boundary_def, dict):
        return boundary_def.get("lines", [])
    return boundary_def


def _get_boundary_flow_groups(boundary_def, available_lines=None):
    """Return positive/negative signed groups for a boundary definition."""
    lines = _get_boundary_lines(boundary_def)
    if isinstance(boundary_def, dict):
        groups = boundary_def.get("flow_groups", {}) or {}
        positive = groups.get("positive", []) or lines
        negative = groups.get("negative", []) or []
    else:
        positive = lines
        negative = []

    if available_lines is not None:
        positive = [lid for lid in positive if lid in available_lines]
        negative = [lid for lid in negative if lid in available_lines]

    return positive, negative


def _get_boundary_transformers(boundary_def):
    """Return the full union of transformers for a boundary definition."""
    if isinstance(boundary_def, dict):
        return boundary_def.get("transformers", [])
    return []


def _get_boundary_transformer_flow_groups(boundary_def, available_transformers=None):
    """Return positive/negative signed transformer groups for a boundary definition."""
    transformers = _get_boundary_transformers(boundary_def)
    if isinstance(boundary_def, dict):
        groups = boundary_def.get("transformer_flow_groups", {}) or {}
        positive = groups.get("positive", []) or transformers
        negative = groups.get("negative", []) or []
    else:
        positive = transformers
        negative = []

    if available_transformers is not None:
        positive = [tid for tid in positive if tid in available_transformers]
        negative = [tid for tid in negative if tid in available_transformers]

    return positive, negative


def _compute_model_boundary_flows(network, boundary_mapping, logger):
    """
    Aggregate model line and link flows into NESO boundary-level flows.

    For each boundary, computes signed area-to-area transfer across the cut
    using figure-backed positive/negative line groups when available.
    Optional ``links`` entries in the boundary definition add HVDC link flows
    (each entry has ``name`` and ``sign`` fields; sign=-1 flips the p0 direction).
    Returns a DataFrame indexed by snapshot with one column per boundary
    (MW aggregate flow magnitude).
    """
    p0 = network.lines_t.p0
    tp0 = network.transformers_t.p0
    lp0 = network.links_t.p0
    results = {}

    for boundary, boundary_def in boundary_mapping.items():
        line_ids = _get_boundary_lines(boundary_def)
        present = [lid for lid in line_ids if lid in p0.columns]
        missing = [lid for lid in line_ids if lid not in p0.columns]
        transformer_ids = _get_boundary_transformers(boundary_def)
        present_transformers = [tid for tid in transformer_ids if tid in tp0.columns]
        missing_transformers = [tid for tid in transformer_ids if tid not in tp0.columns]
        if missing:
            logger.warning(
                f"  Boundary {boundary}: lines not in network: {missing}"
            )
        if missing_transformers:
            logger.warning(
                f"  Boundary {boundary}: transformers not in network: {missing_transformers}"
            )
        if (
            not present
            and not present_transformers
            and not (isinstance(boundary_def, dict) and boundary_def.get("links"))
        ):
            logger.warning(
                f"  Boundary {boundary}: no lines, transformers, or links found in network"
            )
            continue

        positive, negative = _get_boundary_flow_groups(boundary_def, present)
        positive_transformers, negative_transformers = _get_boundary_transformer_flow_groups(
            boundary_def,
            present_transformers,
        )
        if present or present_transformers:
            signed_flow = pd.Series(0.0, index=p0.index)
        else:
            signed_flow = pd.Series(0.0, index=lp0.index if not lp0.empty else p0.index)
        if positive:
            signed_flow = signed_flow.add(p0[positive].sum(axis=1), fill_value=0.0)
        if negative:
            signed_flow = signed_flow.sub(p0[negative].sum(axis=1), fill_value=0.0)
        if positive_transformers:
            signed_flow = signed_flow.add(
                tp0[positive_transformers].sum(axis=1),
                fill_value=0.0,
            )
        if negative_transformers:
            signed_flow = signed_flow.sub(
                tp0[negative_transformers].sum(axis=1),
                fill_value=0.0,
            )

        # Add HVDC link flows if defined
        link_defs = boundary_def.get("links", []) if isinstance(boundary_def, dict) else []
        n_links_added = 0
        for link_entry in link_defs:
            lid = link_entry.get("name") if isinstance(link_entry, dict) else link_entry
            sign = float(link_entry.get("sign", 1)) if isinstance(link_entry, dict) else 1.0
            if lid in lp0.columns:
                signed_flow = signed_flow.add(sign * lp0[lid], fill_value=0.0)
                n_links_added += 1
            else:
                logger.debug(f"  Boundary {boundary}: link {lid} not in links_t.p0 (no flow data)")

        boundary_flow = signed_flow.abs()
        results[boundary] = boundary_flow

        s_nom_total = network.lines.loc[present, "s_nom"].sum() if present else 0.0
        tf_s_nom_total = (
            network.transformers.loc[present_transformers, "s_nom"].sum()
            if present_transformers else 0.0
        )
        logger.info(
            f"  Boundary {boundary}: {len(present)} lines + {len(present_transformers)} "
            f"transformers + {n_links_added} links, "
            f"total branch s_nom={s_nom_total + tf_s_nom_total:.0f} MVA, "
            f"signed groups=+{len(positive) + len(positive_transformers)}/"
            f"-{len(negative) + len(negative_transformers)}, "
            f"mean flow={boundary_flow.mean():.0f} MW, "
            f"max flow={boundary_flow.max():.0f} MW"
        )

    return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON LOGIC
# ═══════════════════════════════════════════════════════════════════════════════


def _compare_costs(model_costs_csv, neso_costs, start_date, end_date, logger):
    """
    Compare model total BM cost vs NESO thermal constraint costs.

    Returns a dict with comparison metrics.
    """
    # Model total BM cost
    model_costs = pd.read_csv(model_costs_csv)
    total_row = model_costs[model_costs["carrier"] == "TOTAL"]
    if total_row.empty:
        model_total = model_costs["net_cost"].sum()
    else:
        model_total = total_row["net_cost"].values[0]

    # NESO total thermal cost
    neso_total = neso_costs["daily_cost_gbp"].sum()

    # NESO by boundary
    neso_by_boundary = (
        neso_costs.groupby("constraint_group")["daily_cost_gbp"]
        .sum()
        .sort_values(ascending=False)
    )

    ratio = model_total / neso_total if neso_total > 0 else float("nan")
    n_days = (end_date - start_date).days + 1

    result = {
        "period_start": str(start_date.date()),
        "period_end": str(end_date.date()),
        "period_days": n_days,
        "model_total_bm_cost_gbp": model_total,
        "neso_total_thermal_cost_gbp": neso_total,
        "model_neso_ratio": ratio,
        "neso_by_boundary": neso_by_boundary.to_dict(),
    }

    logger.info(f"  Model total BM cost:  £{model_total / 1e6:,.1f}M")
    logger.info(f"  NESO thermal cost:    £{neso_total / 1e6:,.1f}M")
    logger.info(f"  Model/NESO ratio:     {ratio:.2f}x")
    for b, v in neso_by_boundary.items():
        pct = v / neso_total * 100 if neso_total > 0 else 0
        logger.info(f"    {b}: £{v / 1e6:,.1f}M ({pct:.0f}%)")

    return result


def _compare_boundary_flows(model_flows, neso_da_flows, boundary_mapping, logger):
    """
    Compare model boundary flows vs NESO DA flows.

    Returns a list of dicts, one per boundary, with flow statistics.
    """
    records = []
    boundaries_in_neso = neso_da_flows["constraint_group"].unique()

    for boundary in boundary_mapping:
        rec = {"boundary": boundary}
        nf = neso_da_flows[neso_da_flows["constraint_group"] == boundary]
        compare_times = None
        if len(nf) > 0:
            compare_times = pd.DatetimeIndex(
                pd.to_datetime(nf["datetime"]).dropna().sort_values().unique()
            )

        # Model stats
        if boundary in model_flows.columns:
            mf = model_flows[boundary]
            if compare_times is not None and len(compare_times) > 0:
                mf = mf.reindex(compare_times).dropna()
            rec["model_mean_flow_mw"] = mf.mean()
            rec["model_max_flow_mw"] = mf.max()
            rec["model_hours"] = len(mf)
        else:
            rec["model_mean_flow_mw"] = np.nan
            rec["model_max_flow_mw"] = np.nan
            rec["model_hours"] = 0

        # NESO stats
        if boundary in boundaries_in_neso:
            rec["neso_mean_flow_mw"] = nf["flow_mw"].mean()
            rec["neso_max_flow_mw"] = nf["flow_mw"].max()
            rec["neso_mean_limit_mw"] = nf["limit_mw"].mean()
            rec["neso_records"] = len(nf)

            # Utilisation: flow / limit
            valid = nf[nf["limit_mw"] > 0]
            if len(valid) > 0:
                util = valid["flow_mw"].abs() / valid["limit_mw"]
                rec["neso_mean_utilisation"] = util.mean()
                rec["neso_pct_above_90"] = (util >= 0.9).mean() * 100
            else:
                rec["neso_mean_utilisation"] = np.nan
                rec["neso_pct_above_90"] = np.nan
        else:
            for k in [
                "neso_mean_flow_mw",
                "neso_max_flow_mw",
                "neso_mean_limit_mw",
                "neso_records",
                "neso_mean_utilisation",
                "neso_pct_above_90",
            ]:
                rec[k] = np.nan

        # Model utilisation (using s_nom from boundary lines)
        if boundary in model_flows.columns and rec.get("model_hours", 0) > 0:
            mf = model_flows[boundary]
            # Get total s_nom for the boundary
            rec["model_mean_utilisation"] = rec.get("model_mean_flow_mw", 0) / max(
                rec.get("neso_mean_limit_mw", mf.max()), 1
            )
        else:
            rec["model_mean_utilisation"] = np.nan

        records.append(rec)
        logger.info(
            f"  {boundary}: model mean={rec['model_mean_flow_mw']:.0f} MW, "
            f"NESO mean={rec.get('neso_mean_flow_mw', float('nan')):.0f} MW, "
            f"NESO util={rec.get('neso_mean_utilisation', float('nan')):.1%}"
        )

    return records


def _compare_congestion_hours(
    model_congestion_csv, model_flows, boundary_mapping, neso_da_flows, network, logger
):
    """
    Compare model congestion hours on boundary lines vs NESO at-limit hours.

    Returns a list of dicts, one per boundary.
    """
    model_cong = pd.read_csv(model_congestion_csv)
    records = []

    for boundary, boundary_def in boundary_mapping.items():
        rec = {"boundary": boundary}
        line_ids = _get_boundary_lines(boundary_def)
        valid_lines = [lid for lid in line_ids if lid in network.lines.index]
        nf = neso_da_flows[neso_da_flows["constraint_group"] == boundary]
        compare_times = None
        if len(nf) > 0:
            compare_times = pd.DatetimeIndex(
                pd.to_datetime(nf["datetime"]).dropna().sort_values().unique()
            )

        # Model: compute congestion using effective capacity (s_nom * s_max_pu)
        # when time-varying NESO limits are present. Fall back to the static
        # congestion summary CSV for runs without s_max_pu.
        if valid_lines and not network.lines_t.p0.empty:
            flows = network.lines_t.p0.loc[:, valid_lines].abs()
            if compare_times is not None and len(compare_times) > 0:
                flows = flows.reindex(compare_times).dropna(how="all")
            s_nom = network.lines.loc[valid_lines, "s_nom"].astype(float)

            s_max_pu = pd.DataFrame(1.0, index=flows.index, columns=valid_lines)
            if hasattr(network.lines_t, "s_max_pu") and not network.lines_t.s_max_pu.empty:
                dynamic_cols = [lid for lid in valid_lines if lid in network.lines_t.s_max_pu.columns]
                if dynamic_cols:
                    s_max_pu.loc[:, dynamic_cols] = (
                        network.lines_t.s_max_pu.loc[:, dynamic_cols]
                        .reindex(index=flows.index)
                        .fillna(1.0)
                    )

            effective_capacity = s_max_pu.mul(s_nom, axis=1).clip(lower=1e-9)
            line_loading = flows.div(effective_capacity)

            line_hours = (line_loading >= 0.95).sum(axis=0)
            rec["model_max_hours_congested"] = int(line_hours.max()) if len(line_hours) else 0
            rec["model_mean_loading"] = float(line_loading.to_numpy().mean())
            rec["model_lines_congested"] = int((line_hours > 0).sum())
        else:
            present = [lid for lid in line_ids if lid in model_cong["component"].values]
            if present:
                bc = model_cong[model_cong["component"].isin(present)]
                rec["model_max_hours_congested"] = int(bc["hours_congested"].max())
                rec["model_mean_loading"] = float(bc["mean_loading_fraction"].mean())
                rec["model_lines_congested"] = len(bc)
            else:
                rec["model_max_hours_congested"] = 0
                rec["model_mean_loading"] = 0.0
                rec["model_lines_congested"] = 0

        # Also compute aggregate boundary loading using effective boundary
        # capacity. When NESO limits are available, compare directly against the
        # published boundary limit series rather than inferring capacity from
        # per-line s_max_pu; this works for both uniform derating and aggregate
        # boundary-constraint modes.
        if boundary in model_flows.columns:
            mf = model_flows[boundary]
            if compare_times is not None and len(compare_times) > 0:
                mf = mf.reindex(compare_times).dropna()
            if len(nf) > 0 and len(mf) > 0:
                limit_series = (
                    nf.assign(datetime=pd.to_datetime(nf["datetime"]))
                    .dropna(subset=["datetime"])
                    .drop_duplicates(subset=["datetime"], keep="last")
                    .set_index("datetime")["limit_mw"]
                    .sort_index()
                )
                limit_series = limit_series.reindex(mf.index).dropna()
                mf_aligned = mf.reindex(limit_series.index).dropna()
                if len(mf_aligned) > 0:
                    loading = mf_aligned / limit_series.clip(lower=1e-9)
                    rec["model_boundary_mean_loading"] = float(loading.mean())
                    rec["model_boundary_hours_above_90"] = int((loading >= 0.9).sum())
                else:
                    rec["model_boundary_mean_loading"] = 0.0
                    rec["model_boundary_hours_above_90"] = 0
            elif valid_lines:
                s_nom = network.lines.loc[valid_lines, "s_nom"].astype(float)
                s_max_pu = pd.DataFrame(1.0, index=mf.index, columns=valid_lines)
                if hasattr(network.lines_t, "s_max_pu") and not network.lines_t.s_max_pu.empty:
                    dynamic_cols = [lid for lid in valid_lines if lid in network.lines_t.s_max_pu.columns]
                    if dynamic_cols:
                        s_max_pu.loc[:, dynamic_cols] = (
                            network.lines_t.s_max_pu.loc[:, dynamic_cols]
                            .reindex(index=mf.index)
                            .fillna(1.0)
                        )

                effective_boundary_capacity = s_max_pu.mul(s_nom, axis=1).sum(axis=1)
                effective_boundary_capacity = effective_boundary_capacity.clip(lower=1e-9)
                loading = mf / effective_boundary_capacity
                rec["model_boundary_mean_loading"] = float(loading.mean())
                rec["model_boundary_hours_above_90"] = int((loading >= 0.9).sum())
            else:
                rec["model_boundary_mean_loading"] = 0.0
                rec["model_boundary_hours_above_90"] = 0

        # NESO: hours at limit
        if len(nf) > 0:
            valid = nf[nf["limit_mw"] > 0]
            if len(valid) > 0:
                util = valid["flow_mw"].abs() / valid["limit_mw"]
                rec["neso_hours_above_90"] = (util >= 0.9).sum()
                rec["neso_total_periods"] = len(valid)
            else:
                rec["neso_hours_above_90"] = 0
                rec["neso_total_periods"] = 0
        else:
            rec["neso_hours_above_90"] = 0
            rec["neso_total_periods"] = 0

        records.append(rec)
        logger.info(
            f"  {boundary}: model line-max={rec['model_max_hours_congested']}h, "
            f"model boundary ≥90%={rec.get('model_boundary_hours_above_90', 0)}h, "
            f"NESO ≥90%={rec['neso_hours_above_90']} periods"
        )

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def _build_validation_csv(cost_comparison, flow_comparison, congestion_comparison, logger):
    """Build a single validation summary DataFrame."""
    rows = []

    # Cost metrics
    rows.append(
        {
            "category": "total_cost",
            "metric": "model_bm_cost_gbp",
            "value": cost_comparison["model_total_bm_cost_gbp"],
        }
    )
    rows.append(
        {
            "category": "total_cost",
            "metric": "neso_thermal_cost_gbp",
            "value": cost_comparison["neso_total_thermal_cost_gbp"],
        }
    )
    rows.append(
        {
            "category": "total_cost",
            "metric": "model_neso_ratio",
            "value": cost_comparison["model_neso_ratio"],
        }
    )

    # NESO cost by boundary
    for boundary, cost in cost_comparison["neso_by_boundary"].items():
        rows.append(
            {
                "category": "neso_boundary_cost",
                "metric": f"{boundary}_gbp",
                "value": cost,
            }
        )

    # Flow comparison
    for fc in flow_comparison:
        b = fc["boundary"]
        for key in [
            "model_mean_flow_mw",
            "model_max_flow_mw",
            "neso_mean_flow_mw",
            "neso_max_flow_mw",
            "neso_mean_limit_mw",
            "neso_mean_utilisation",
            "neso_pct_above_90",
        ]:
            if key in fc:
                rows.append(
                    {
                        "category": f"flow_{b}",
                        "metric": key,
                        "value": fc[key],
                    }
                )

    # Congestion comparison
    for cc in congestion_comparison:
        b = cc["boundary"]
        for key in [
            "model_max_hours_congested",
            "model_mean_loading",
            "model_boundary_mean_loading",
            "model_boundary_hours_above_90",
            "neso_hours_above_90",
        ]:
            if key in cc:
                rows.append(
                    {
                        "category": f"congestion_{b}",
                        "metric": key,
                        "value": cc[key],
                    }
                )

    return pd.DataFrame(rows)


def _build_dashboard(
    cost_comparison,
    flow_comparison,
    congestion_comparison,
    model_flows,
    neso_da_flows,
    neso_costs,
    scenario_id,
    logger,
):
    """Build a multi-panel Plotly HTML dashboard."""
    if not HAS_PLOTLY:
        logger.warning("Plotly not available — skipping dashboard")
        return None

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Total Cost Comparison",
            "NESO Thermal Cost by Boundary",
            "Model vs NESO Mean Boundary Flow (MW)",
            "Boundary Binding Hours",
            "NESO Boundary Utilisation (%)",
            "Model Boundary Flow Time Series",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
    )

    # ── Panel 1: Total cost comparison ──────────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=["Model BM Cost", "NESO Thermal Cost"],
            y=[
                cost_comparison["model_total_bm_cost_gbp"] / 1e6,
                cost_comparison["neso_total_thermal_cost_gbp"] / 1e6,
            ],
            marker_color=["#1f77b4", "#ff7f0e"],
            text=[
                f"£{cost_comparison['model_total_bm_cost_gbp'] / 1e6:,.1f}M",
                f"£{cost_comparison['neso_total_thermal_cost_gbp'] / 1e6:,.1f}M",
            ],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="£ millions", row=1, col=1)

    # ── Panel 2: NESO cost by boundary ──────────────────────────────────────
    boundary_costs = cost_comparison["neso_by_boundary"]
    if boundary_costs:
        names = list(boundary_costs.keys())
        values = [boundary_costs[n] / 1e6 for n in names]
        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                marker_color="#ff7f0e",
                text=[f"£{v:,.1f}M" for v in values],
                textposition="outside",
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="£ millions", row=1, col=2)

    # ── Panel 3: Mean flow comparison ───────────────────────────────────────
    boundaries = [fc["boundary"] for fc in flow_comparison]
    model_means = [fc.get("model_mean_flow_mw", 0) for fc in flow_comparison]
    neso_means = [fc.get("neso_mean_flow_mw", 0) for fc in flow_comparison]

    fig.add_trace(
        go.Bar(
            x=boundaries,
            y=model_means,
            name="Model",
            marker_color="#1f77b4",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=boundaries,
            y=neso_means,
            name="NESO DA",
            marker_color="#ff7f0e",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="MW", row=2, col=1)

    # ── Panel 4: Congestion hours ───────────────────────────────────────────
    bounds = [cc["boundary"] for cc in congestion_comparison]
    model_line_hours = [cc.get("model_max_hours_congested", 0) for cc in congestion_comparison]
    model_boundary_hours = [cc.get("model_boundary_hours_above_90", 0) for cc in congestion_comparison]
    neso_hours = [cc.get("neso_hours_above_90", 0) for cc in congestion_comparison]

    fig.add_trace(
        go.Bar(
            x=bounds,
            y=model_line_hours,
            name="Model line max (≥95%)",
            marker_color="#1f77b4",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=bounds,
            y=model_boundary_hours,
            name="Model boundary (≥90%)",
            marker_color="#9467bd",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=bounds,
            y=neso_hours,
            name="NESO (≥90% util)",
            marker_color="#ff7f0e",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="Hours / Periods", row=2, col=2)

    # ── Panel 5: NESO utilisation by boundary ───────────────────────────────
    utils = [fc.get("neso_mean_utilisation", 0) * 100 for fc in flow_comparison]
    pct90 = [fc.get("neso_pct_above_90", 0) for fc in flow_comparison]

    fig.add_trace(
        go.Bar(
            x=boundaries,
            y=utils,
            name="Mean Utilisation %",
            marker_color="#2ca02c",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=boundaries,
            y=pct90,
            name="% Periods ≥90%",
            marker_color="#d62728",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="%", row=3, col=1)

    # ── Panel 6: Model boundary flow time series ────────────────────────────
    if not model_flows.empty:
        for col in model_flows.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_flows.index,
                    y=model_flows[col],
                    mode="lines",
                    name=col,
                    showlegend=False,
                ),
                row=3,
                col=2,
            )
    fig.update_yaxes(title_text="MW", row=3, col=2)

    # Layout
    ratio_str = f"{cost_comparison['model_neso_ratio']:.2f}x"
    fig.update_layout(
        title_text=(
            f"NESO Constraint Validation — {scenario_id} "
            f"({cost_comparison['period_start']} to {cost_comparison['period_end']}) "
            f"[Model/NESO = {ratio_str}]"
        ),
        height=1100,
        width=1400,
        barmode="group",
        showlegend=True,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def validate_neso_constraints(
    scenario_config,
    balancing_network_path,
    constraint_costs_csv,
    congestion_csv,
    output_csv,
    output_html,
    boundary_mapping_path="data/network/neso_boundary_mapping.yaml",
    cache_dir="data/validation",
    logger=None,
):
    """
    Compare model BM results against NESO published constraint data.

    Parameters
    ----------
    scenario_config : dict
        Scenario configuration (must include modelled_year, solve_period).
    balancing_network_path : str
        Path to solved balancing network (.nc) with line flows.
    constraint_costs_csv : str
        Path to model constraint_costs.csv (by carrier).
    congestion_csv : str
        Path to model congestion.csv (by line).
    output_csv : str
        Path for output validation CSV.
    output_html : str
        Path for output validation dashboard HTML.
    boundary_mapping_path : str
        Path to NESO boundary mapping YAML.
    cache_dir : str
        Directory for caching NESO downloads.
    logger : logging.Logger, optional
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    scenario_id = scenario_config.get("scenario_id", "unknown")
    modelled_year = scenario_config.get("modelled_year", 9999)

    # ── Future scenario guard ───────────────────────────────────────────────
    if modelled_year > 2024:
        logger.info(
            f"Scenario {scenario_id} is future (year={modelled_year}) "
            f"— no NESO actuals available. Writing stub outputs."
        )
        pd.DataFrame(
            [{"category": "info", "metric": "status", "value": "future_scenario_stub"}]
        ).to_csv(output_csv, index=False)
        Path(output_html).write_text(
            f"<html><body><p>No NESO validation for future scenario "
            f"{scenario_id} (year={modelled_year})</p></body></html>"
        )
        return

    # ── Determine date range ────────────────────────────────────────────────
    solve_period = scenario_config.get("solve_period", {})
    if solve_period.get("enabled", False):
        start_date = pd.Timestamp(solve_period["start"])
        end_date = pd.Timestamp(solve_period["end"])
    else:
        start_date = pd.Timestamp(f"{modelled_year}-01-01")
        end_date = pd.Timestamp(f"{modelled_year}-12-31 23:00")

    logger.info(f"Validation period: {start_date} to {end_date}")

    # ── Load data ───────────────────────────────────────────────────────────
    logger.info("Loading NESO thermal constraint costs...")
    neso_costs = _load_thermal_costs(start_date, end_date, cache_dir, logger)

    logger.info("Loading NESO DA constraint flows...")
    neso_da_flows = _load_da_flows(start_date, end_date, cache_dir, logger)

    logger.info("Loading boundary mapping...")
    transmission = scenario_config.get("transmission", {}) or {}
    outage_cfg = transmission.get("outage_schedule", {}) or {}
    neso_cfg = outage_cfg.get("neso", {}) or {}
    boundary_include = neso_cfg.get("boundary_include")

    boundary_mapping = _load_boundary_mapping(
        boundary_mapping_path,
        logger,
        boundary_include=boundary_include,
    )

    logger.info("Loading model balancing network...")
    network = pypsa.Network(balancing_network_path)

    # ── Compute model boundary flows ────────────────────────────────────────
    logger.info("Computing model boundary flows...")
    model_flows = _compute_model_boundary_flows(network, boundary_mapping, logger)

    # ── Comparisons ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("COST COMPARISON")
    logger.info("=" * 60)
    cost_comparison = _compare_costs(
        constraint_costs_csv, neso_costs, start_date, end_date, logger
    )

    logger.info("=" * 60)
    logger.info("BOUNDARY FLOW COMPARISON")
    logger.info("=" * 60)
    flow_comparison = _compare_boundary_flows(
        model_flows, neso_da_flows, boundary_mapping, logger
    )

    logger.info("=" * 60)
    logger.info("CONGESTION HOURS COMPARISON")
    logger.info("=" * 60)
    congestion_comparison = _compare_congestion_hours(
        congestion_csv, model_flows, boundary_mapping, neso_da_flows, network, logger
    )

    # ── Write outputs ───────────────────────────────────────────────────────
    logger.info("Writing validation CSV...")
    validation_df = _build_validation_csv(
        cost_comparison, flow_comparison, congestion_comparison, logger
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(output_csv, index=False)
    logger.info(f"  Written: {output_csv} ({len(validation_df)} rows)")

    logger.info("Building validation dashboard...")
    fig = _build_dashboard(
        cost_comparison,
        flow_comparison,
        congestion_comparison,
        model_flows,
        neso_da_flows,
        neso_costs,
        scenario_id,
        logger,
    )
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    if fig is not None:
        fig.write_html(output_html, include_plotlyjs="cdn")
        logger.info(f"  Written: {output_html}")
    else:
        Path(output_html).write_text(
            "<html><body><p>Plotly not available for dashboard.</p></body></html>"
        )

    logger.info("NESO constraint validation complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_path = (
        snakemake.log[0]
        if hasattr(snakemake, "log") and snakemake.log
        else "validate_neso"
    )
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("NESO CONSTRAINT VALIDATION — MODEL vs NESO ACTUALS")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        scenario_config = snakemake.params.scenario_config
        scenario_id = scenario_config.get("scenario_id", snakemake.wildcards.scenario)
        scenario_config["scenario_id"] = scenario_id

        validate_neso_constraints(
            scenario_config=scenario_config,
            balancing_network_path=snakemake.input.balancing_network,
            constraint_costs_csv=snakemake.input.constraint_costs_csv,
            congestion_csv=snakemake.input.congestion_csv,
            output_csv=snakemake.output.validation_csv,
            output_html=snakemake.output.validation_html,
            boundary_mapping_path=snakemake.params.get(
                "boundary_mapping_path", "data/network/neso_boundary_mapping.yaml"
            ),
            cache_dir=snakemake.params.get("cache_dir", "data/validation"),
            logger=logger,
        )

        elapsed = time.time() - start_time
        logger.info(f"NESO validation completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"FATAL ERROR in NESO validation: {e}", exc_info=True)
        raise
