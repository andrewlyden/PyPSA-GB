"""
BM Validation — Compare PyPSA-GB Balancing Mechanism Against ELEXON Actuals

Fetches (or loads cached) ELEXON BM data and produces a validation report
comparing the model's two-stage dispatch results against historical actuals.

Comparisons:
  1. Redispatch volumes  — model vs BOALF acceptance volumes by fuel type
  2. System prices       — model wholesale SMP vs ELEXON MID & SBP
  3. Constraint costs    — model total BM cost vs annualised NESO benchmarks
  4. Dispatch levels     — model physical dispatch vs B1610 actual generation
  5. Price duration curve— sorted price profile comparison

Inputs:
  - Model results: redispatch_summary, constraint_costs, price_comparison,
    wholesale_dispatch, balancing_dispatch CSVs
  - ELEXON data: fetched from API or loaded from cache

Outputs:
  - {scenario}_bm_validation.csv  — tabular comparison metrics
  - {scenario}_bm_validation.html — multi-panel comparison dashboard

Called by Snakemake rule `validate_bm_results` in rules/market.smk.
"""

import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("validate_bm")


# ─── BMU fuel type classification ────────────────────────────────────────────
# Maps BMU name prefixes to PyPSA carrier names for cross-comparison.
# Based on ELEXON REMIT Fuel Type Codes and common BMU naming conventions.
BMU_FUEL_MAP = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "COAL": "coal",
    "OIL": "oil",
    "NPSHYD": "pumped_hydro",
    "PS": "pumped_hydro",
    "WIND": "wind_onshore",
    "BIOMASS": "biomass",
    "NUCLEAR": "nuclear",
}

# Known BMU prefixes for fuel type inference from the BMU ID itself.
# Ordered longest-first so longer prefix takes precedence over shorter one.
BMU_PREFIX_FUEL = {
    # Nuclear
    "T_SIZB": "nuclear", "T_SIZA": "nuclear",
    "T_HINB": "nuclear", "T_HINK": "nuclear",
    "T_TORN": "nuclear",
    "T_HUNB": "nuclear", "T_HUNT": "nuclear",
    "T_HEYM": "nuclear", "T_HEYS": "nuclear",
    "T_DUNG": "nuclear",
    "T_HART": "nuclear", "T_HRTL": "nuclear",
    "T_WYLF": "nuclear",
    # Coal
    "T_COTPS": "coal", "T_COTT": "coal",
    "T_RATS": "coal", "T_RATSGT": "coal",
    "T_ABTH": "coal",
    "T_WBUPS": "coal", "T_WBURB": "coal", "T_WBUGT": "coal",
    # Note: T_DRAXX is biomass (Drax converted 2013-2021), see below
    "T_EGGPS": "coal",
    "T_FIDR": "coal", "T_FIDL": "coal",
    "T_FERR": "coal",
    "T_RUGG": "coal",
    # CCGT / gas
    "T_PEMB": "CCGT",
    "T_DAMC": "CCGT", "T_DAMH": "CCGT",
    "T_SEAB": "CCGT",
    "T_CARR": "CCGT",
    "T_DIDCB": "CCGT",
    "T_MRWD": "CCGT", "T_MARC": "CCGT",
    "T_SHBA": "CCGT",
    "T_STAY": "CCGT",
    "T_CNQPS": "CCGT", "T_CNQP": "CCGT",
    "T_LBAR": "CCGT",
    "T_MEDP": "CCGT", "T_MEDW": "CCGT",
    "T_RYHPS": "CCGT",
    "T_GRAI": "CCGT",
    "T_SUTB": "CCGT",
    "T_KEAD": "CCGT",
    "T_HUMR": "CCGT",
    "T_PEHE": "CCGT", "T_PEHED": "CCGT",
    "T_SEEL": "CCGT", "T_SPLN": "CCGT", "T_SPAE": "CCGT",
    "T_SCCL": "CCGT", "T_SALD": "CCGT",
    "T_SVRP": "CCGT",
    "T_LAGA": "CCGT", "T_LANG": "CCGT",
    "T_EECL": "CCGT",
    "T_ROCK": "CCGT",
    "T_COSO": "CCGT", "T_CORY": "CCGT",
    "T_CDCL": "CCGT",
    # Offshore wind
    "T_BEATO": "wind_offshore", "T_BEATD": "wind_offshore",
    "T_DDGNO": "wind_offshore", "T_DDGN": "wind_offshore",
    "T_WLNYO": "wind_offshore", "T_WLNYW": "wind_offshore",
    "T_WDNSO": "wind_offshore",
    "T_BOWLW": "wind_offshore",
    "T_WTMSO": "wind_offshore",
    "T_HMGTO": "wind_offshore", "T_HMGT": "wind_offshore",
    "T_EAAO": "wind_offshore",
    "T_GRGBW": "wind_offshore",
    "T_RMPNO": "wind_offshore", "T_RMPN": "wind_offshore",
    "T_LNCSW": "wind_offshore", "T_LNCS": "wind_offshore",
    "T_SHRSW": "wind_offshore",
    "T_BRBEO": "wind_offshore",
    "T_HOWAO": "wind_offshore", "T_HOWA": "wind_offshore",
    "T_HOWBO": "wind_offshore", "T_HOWB": "wind_offshore",
    "T_SGRW": "wind_offshore",
    "T_RCBKO": "wind_offshore",
    "T_THNTO": "wind_offshore",
    "T_GYMR": "wind_offshore",
    # Onshore wind
    "T_WHILW": "wind_onshore",
    "T_CLDCW": "wind_onshore", "T_CLDNW": "wind_onshore", "T_CLDSW": "wind_onshore",
    "T_STLGW": "wind_onshore",
    "T_PNYCW": "wind_onshore",
    "T_AKGLW": "wind_onshore",
    "T_ABRBO": "wind_onshore",
    # Pumped storage hydro
    "T_CRUA": "Pumped Storage Hydroelectricity",
    "T_CRUAD": "Pumped Storage Hydroelectricity",
    "T_FOYE": "Pumped Storage Hydroelectricity",
    "T_FOYED": "Pumped Storage Hydroelectricity",
    "T_DINO": "Pumped Storage Hydroelectricity",
    "T_FFES": "Pumped Storage Hydroelectricity",
    "T_FFESST": "Pumped Storage Hydroelectricity",
    # Large hydro (reservoir)
    "T_SLOY": "large_hydro",
    "T_ERRO": "large_hydro",
    "T_FASN": "large_hydro",
    "T_GLND": "large_hydro",
    "T_NANT": "large_hydro",
    "T_FINL": "large_hydro",
    # Biomass
    "T_DRAXX": "biomass",     # Drax biomass units (DRAXX vs DRAX)
    "T_LYNES": "biomass",
    "T_LYNE": "biomass",
    "T_IRON": "biomass",
}


def _classify_bmu_carrier(bmu_id: str) -> str:
    """Infer PyPSA carrier from BMU ID prefix."""
    for prefix, carrier in BMU_PREFIX_FUEL.items():
        if bmu_id.startswith(prefix):
            return carrier
    # Fallback: check if common fuel word appears
    bmu_upper = bmu_id.upper()
    for keyword, carrier in BMU_FUEL_MAP.items():
        if keyword in bmu_upper:
            return carrier
    return "unknown"


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed-type flag columns to boolean."""
    if series.dtype == bool:
        return series.fillna(False)
    text = series.fillna(False).astype(str).str.strip().str.lower()
    return text.isin(["true", "t", "1", "y", "yes"])


def _normalise_boalf_dataframe(boalf_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise BOALF column names and flag columns."""
    if boalf_df.empty:
        return boalf_df.copy()

    df = boalf_df.copy()
    rename_map = {
        "bmUnit": "bmu_id",
        "acceptanceNumber": "acceptance_number",
        "soFlag": "so_flag",
        "storFlag": "stor_flag",
        "storProviderFlag": "stor_flag",
        "rrFlag": "rr_flag",
        "levelFrom": "level_from",
        "levelTo": "level_to",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "timeFrom" in df.columns:
        df["timeFrom"] = pd.to_datetime(df["timeFrom"], utc=True, errors="coerce").dt.tz_localize(None)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False, errors="coerce")

    for flag_col in ["so_flag", "stor_flag", "rr_flag"]:
        if flag_col in df.columns:
            df[flag_col] = _coerce_bool_series(df[flag_col])
        else:
            df[flag_col] = False

    for level_col in ["level_from", "level_to"]:
        if level_col in df.columns:
            df[level_col] = pd.to_numeric(df[level_col], errors="coerce").fillna(0.0)

    return df


def _prepare_boalf_acceptance_records(boalf_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse BOALF ramp segments into one net change per acceptance."""
    df = _normalise_boalf_dataframe(boalf_df)
    empty_cols = [
        "bmu_id", "acceptance_number", "carrier", "net_delta", "increase_mw",
        "decrease_mw", "so_flag", "stor_flag", "rr_flag", "any_flag",
        "unflagged",
    ]
    if df.empty or "bmu_id" not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    if "level_from" not in df.columns or "level_to" not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    df["carrier"] = df["bmu_id"].astype(str).apply(_classify_bmu_carrier)
    time_col = "timeFrom" if "timeFrom" in df.columns else "datetime" if "datetime" in df.columns else None

    acc_records = []
    group_cols = ["bmu_id", "acceptance_number"] if "acceptance_number" in df.columns else ["bmu_id"]

    for _, grp in df.groupby(group_cols, dropna=False):
        if time_col:
            grp = grp.sort_values(time_col)

        first_from = float(grp["level_from"].iloc[0])
        last_to = float(grp["level_to"].iloc[-1])
        net_delta = last_to - first_from
        so_flag = bool(grp["so_flag"].any())
        stor_flag = bool(grp["stor_flag"].any())
        rr_flag = bool(grp["rr_flag"].any())
        any_flag = so_flag or stor_flag or rr_flag

        acc_records.append({
            "bmu_id": grp["bmu_id"].iloc[0],
            "acceptance_number": grp["acceptance_number"].iloc[0] if "acceptance_number" in grp.columns else pd.NA,
            "carrier": grp["carrier"].iloc[0],
            "net_delta": net_delta,
            "increase_mw": max(net_delta, 0.0),
            "decrease_mw": max(-net_delta, 0.0),
            "so_flag": so_flag,
            "stor_flag": stor_flag,
            "rr_flag": rr_flag,
            "any_flag": any_flag,
            "unflagged": not any_flag,
        })

    return pd.DataFrame(acc_records)


def _normalise_disbsad_dataframe(disbsad_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise DISBSAD column names and basic types."""
    if disbsad_df.empty:
        return disbsad_df.copy()

    df = disbsad_df.copy()
    rename_map = {
        "settlementDate": "settlement_date",
        "settlementPeriod": "settlement_period",
        "soFlag": "so_flag",
        "storFlag": "stor_flag",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False, errors="coerce")
    elif "settlement_date" in df.columns and "settlement_period" in df.columns:
        df["datetime"] = pd.to_datetime(df["settlement_date"], errors="coerce") + pd.to_timedelta(
            (pd.to_numeric(df["settlement_period"], errors="coerce").fillna(1).astype(int) - 1) * 30,
            unit="min",
        )

    for flag_col in ["so_flag", "stor_flag"]:
        if flag_col in df.columns:
            df[flag_col] = _coerce_bool_series(df[flag_col])
        else:
            df[flag_col] = False

    for num_col in ["cost", "volume"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)

    if "service" not in df.columns:
        df["service"] = pd.NA

    return df


def _aggregate_boalf_by_carrier(boalf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate BOALF acceptance data into increase/decrease volumes by carrier.

    Each BOALF acceptance can have multiple profile rows (ramp segments).
    We compute the *net change* per acceptance (first level_from → last
    level_to, sorted by timeFrom) to avoid double-counting within-acceptance
    ramp-up/ramp-down segments.

    Parameters
    ----------
    boalf_df : DataFrame
        Raw BOALF data with columns: bmu_id, level_from, level_to,
        acceptance_number, timeFrom.

    Returns
    -------
    DataFrame with columns: carrier, increase_MW, decrease_MW, n_acceptances.
    """
    if boalf_df.empty:
        return pd.DataFrame(columns=["carrier", "increase_MW", "decrease_MW", "n_acceptances"])

    acc_df = _prepare_boalf_acceptance_records(boalf_df)
    if acc_df.empty:
        return pd.DataFrame(columns=["carrier", "increase_MW", "decrease_MW", "n_acceptances"])

    result = acc_df.groupby("carrier").agg(
        increase_MW=("increase_mw", "sum"),
        decrease_MW=("decrease_mw", "sum"),
        n_acceptances=("net_delta", "count"),
    ).reset_index()

    return result


def create_boalf_flag_summary(boalf_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise BOALF acceptance volumes for flagged and unflagged subsets."""
    acc_df = _prepare_boalf_acceptance_records(boalf_df)
    columns = [
        "scope", "group", "carrier", "n_acceptances", "increase_mw",
        "decrease_mw", "increase_mwh", "decrease_mwh",
    ]
    if acc_df.empty:
        return pd.DataFrame(columns=columns)

    groups = {
        "all": acc_df.index == acc_df.index,
        "unflagged": acc_df["unflagged"],
        "flagged_any": acc_df["any_flag"],
        "so_flagged": acc_df["so_flag"],
        "stor_flagged": acc_df["stor_flag"],
        "rr_flagged": acc_df["rr_flag"],
    }

    rows = []
    for label, mask in groups.items():
        subset = acc_df[mask].copy()
        rows.append({
            "scope": "total",
            "group": label,
            "carrier": "ALL",
            "n_acceptances": int(len(subset)),
            "increase_mw": float(subset["increase_mw"].sum()),
            "decrease_mw": float(subset["decrease_mw"].sum()),
            "increase_mwh": float(subset["increase_mw"].sum() * 0.5),
            "decrease_mwh": float(subset["decrease_mw"].sum() * 0.5),
        })
        if subset.empty:
            continue
        by_carrier = subset.groupby("carrier").agg(
            n_acceptances=("net_delta", "count"),
            increase_mw=("increase_mw", "sum"),
            decrease_mw=("decrease_mw", "sum"),
        )
        for carrier, carrier_row in by_carrier.iterrows():
            rows.append({
                "scope": "carrier",
                "group": label,
                "carrier": carrier,
                "n_acceptances": int(carrier_row["n_acceptances"]),
                "increase_mw": float(carrier_row["increase_mw"]),
                "decrease_mw": float(carrier_row["decrease_mw"]),
                "increase_mwh": float(carrier_row["increase_mw"] * 0.5),
                "decrease_mwh": float(carrier_row["decrease_mw"] * 0.5),
            })

    return pd.DataFrame(rows, columns=columns)


def create_disbsad_summary(disbsad_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise DISBSAD non-BM balancing-service adjustments."""
    df = _normalise_disbsad_dataframe(disbsad_df)
    columns = [
        "scope", "group", "service", "n_records", "abs_volume_mwh",
        "net_volume_mwh", "positive_volume_mwh", "negative_volume_mwh",
        "cost_gbp",
    ]
    if df.empty or "volume" not in df.columns or "cost" not in df.columns:
        return pd.DataFrame(columns=columns)

    groups = {
        "all": df.index == df.index,
        "unflagged": ~(df["so_flag"] | df["stor_flag"]),
        "flagged_any": (df["so_flag"] | df["stor_flag"]),
        "so_flagged": df["so_flag"],
        "stor_flagged": df["stor_flag"],
    }

    rows = []
    for label, mask in groups.items():
        subset = df[mask].copy()
        rows.append({
            "scope": "flag_group",
            "group": label,
            "service": "ALL",
            "n_records": int(len(subset)),
            "abs_volume_mwh": float(subset["volume"].abs().sum()),
            "net_volume_mwh": float(subset["volume"].sum()),
            "positive_volume_mwh": float(subset["volume"].clip(lower=0).sum()),
            "negative_volume_mwh": float(subset["volume"].clip(upper=0).sum()),
            "cost_gbp": float(subset["cost"].sum()),
        })

    service_series = df["service"].fillna("unlabelled").replace("", "unlabelled")
    for service, subset in df.assign(service_clean=service_series).groupby("service_clean"):
        rows.append({
            "scope": "service",
            "group": "all",
            "service": service,
            "n_records": int(len(subset)),
            "abs_volume_mwh": float(subset["volume"].abs().sum()),
            "net_volume_mwh": float(subset["volume"].sum()),
            "positive_volume_mwh": float(subset["volume"].clip(lower=0).sum()),
            "negative_volume_mwh": float(subset["volume"].clip(upper=0).sum()),
            "cost_gbp": float(subset["cost"].sum()),
        })

    return pd.DataFrame(rows, columns=columns)


def _load_or_fetch_validation_data(
    start_date: str,
    end_date: str,
    cache_dir: str,
    logger: logging.Logger,
) -> dict:
    """
    Load BM validation data from cache, or fetch from ELEXON API.

    Checks for existing cached files first; only fetches missing data.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    files = {
        "boalf": cache_path / "boalf_data.csv",
        "system_prices": cache_path / "system_prices.csv",
        "b1610": cache_path / "b1610_actual.csv",
        "disbsad": cache_path / "disbsad_data.csv",
    }

    # Check which files exist
    missing = [k for k, v in files.items() if not v.exists() or v.stat().st_size < 100]

    if not missing:
        logger.info(f"All BM validation data cached in {cache_dir}")
        return {f"{k}_file": str(v) for k, v in files.items()}

    logger.info(f"Missing BM validation data: {missing} — fetching from ELEXON API")

    from scripts.market.elexon_data import retrieve_bm_validation_data

    result = retrieve_bm_validation_data(
        start_date=start_date,
        end_date=end_date,
        output_dir=str(cache_path),
        logger=logger,
    )
    return result


def _load_or_fetch_validation_data_checked(
    start_date: str,
    end_date: str,
    cache_dir: str,
    logger: logging.Logger,
) -> dict:
    """
    Coverage-aware wrapper around ``_load_or_fetch_validation_data``.

    Existing cached files are only reused when they span the requested
    validation window; otherwise they are refreshed in-place.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path / "coverage_meta.json"

    files = {
        "boalf": cache_path / "boalf_data.csv",
        "system_prices": cache_path / "system_prices.csv",
        "b1610": cache_path / "b1610_actual.csv",
        "disbsad": cache_path / "disbsad_data.csv",
    }
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

    def _extract_coverage(kind: str, path: Path):
        try:
            if kind == "boalf":
                df = pd.read_csv(path, usecols=["timeFrom", "timeTo"])
                ts_from = pd.to_datetime(
                    df["timeFrom"], utc=True, errors="coerce"
                ).dt.tz_localize(None)
                ts_to = pd.to_datetime(
                    df["timeTo"], utc=True, errors="coerce"
                ).dt.tz_localize(None)
                ts = pd.concat([ts_from, ts_to], ignore_index=True)
            elif kind == "disbsad":
                df = pd.read_csv(path, usecols=["datetime"])
                ts = pd.to_datetime(df["datetime"], errors="coerce")
            else:
                df = pd.read_csv(path, usecols=[0])
                ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            ts = ts.dropna()
            if ts.empty:
                return None
            return {"start": ts.min(), "end": ts.max()}
        except Exception as e:
            logger.warning(
                f"Could not inspect cached validation coverage for {path}: {e}"
            )
            return None

    cached_coverage = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "coverage" in meta:
                cached_coverage = {
                    key: {
                        "start": pd.Timestamp(value["start"]),
                        "end": pd.Timestamp(value["end"]),
                    }
                    for key, value in meta["coverage"].items()
                    if isinstance(value, dict)
                    and "start" in value
                    and "end" in value
                }
        except Exception as e:
            logger.warning(f"Could not read validation cache metadata {meta_path}: {e}")

    missing = [k for k, v in files.items() if not v.exists() or v.stat().st_size < 100]
    stale = []

    if not missing:
        if cached_coverage is None:
            cached_coverage = {
                kind: _extract_coverage(kind, path)
                for kind, path in files.items()
            }

        for kind, coverage in cached_coverage.items():
            if coverage is None:
                stale.append(kind)
                continue
            if coverage["start"] > requested_start or coverage["end"] < requested_end:
                stale.append(kind)

    if not missing and not stale:
        meta = {
            "requested_start": start_date,
            "requested_end": end_date,
            "coverage": {
                key: {
                    "start": str(value["start"]),
                    "end": str(value["end"]),
                }
                for key, value in cached_coverage.items()
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(
            f"All BM validation data cached in {cache_dir} "
            f"(coverage {requested_start} to {requested_end})"
        )
        return {f"{k}_file": str(v) for k, v in files.items()}

    to_fetch = sorted(set(missing + stale))
    logger.info(
        f"Missing/stale BM validation data: {to_fetch} - fetching from ELEXON API"
    )

    from scripts.market.elexon_data import retrieve_bm_validation_data

    result = retrieve_bm_validation_data(
        start_date=start_date,
        end_date=end_date,
        output_dir=str(cache_path),
        logger=logger,
    )

    refreshed_coverage = {
        kind: _extract_coverage(kind, path)
        for kind, path in files.items()
        if path.exists()
    }
    meta = {
        "requested_start": start_date,
        "requested_end": end_date,
        "coverage": {
            key: {
                "start": str(value["start"]),
                "end": str(value["end"]),
            }
            for key, value in refreshed_coverage.items()
            if value is not None
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


def _read_csv_or_empty(
    path: str | Path,
    logger: logging.Logger,
    label: str,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV cache, treating empty or corrupt cache files as no data."""
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        logger.warning(f"{label} cache is empty: {path}")
    except FileNotFoundError:
        logger.warning(f"{label} cache not found: {path}")
    except Exception as e:
        logger.warning(f"Failed to load {label} cache {path}: {e}")
    return pd.DataFrame()


def create_validation_report(
    model_redispatch: pd.DataFrame,
    model_costs: pd.DataFrame,
    model_prices: pd.DataFrame,
    model_wholesale_dispatch: pd.DataFrame,
    model_balancing_dispatch: pd.DataFrame,
    boalf_df: pd.DataFrame,
    disbsad_df: pd.DataFrame,
    system_prices: pd.DataFrame,
    b1610_df: pd.DataFrame,
    mid_prices: pd.Series,
    scenario_id: str,
    modelled_year: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Create a tabular validation report comparing model vs ELEXON actuals.

    Returns
    -------
    DataFrame with columns: metric, model_value, elexon_value, unit, ratio, note
    """
    rows = []

    # ── 1. Total BM cost comparison ──────────────────────────────────────
    total_row = model_costs[model_costs.iloc[:, 0] == "TOTAL"]
    model_bm_cost = total_row["net_cost"].values[0] if len(total_row) else model_redispatch["net_cost"].sum()
    n_hours = len(model_prices) if len(model_prices) > 0 else 24
    annual_equiv = model_bm_cost * (8760 / n_hours) if n_hours > 0 else 0

    # Published NESO BM cost benchmarks
    benchmarks = {2019: 1.2e9, 2020: 1.4e9, 2021: 2.1e9, 2022: 3.5e9, 2023: 2.8e9, 2024: 2.3e9}
    elexon_bm = benchmarks.get(modelled_year, None)

    rows.append({
        "metric": "Total BM cost (solve period)",
        "model_value": f"£{model_bm_cost:,.0f}",
        "elexon_value": "—",
        "unit": "£",
        "ratio": "—",
        "note": f"Over {n_hours} hours",
    })
    rows.append({
        "metric": "Annualised BM cost",
        "model_value": f"£{annual_equiv:,.0f}",
        "elexon_value": f"£{elexon_bm:,.0f}" if elexon_bm else "N/A",
        "unit": "£/year",
        "ratio": f"{annual_equiv / elexon_bm:.2f}" if elexon_bm else "—",
        "note": "NESO published benchmark" if elexon_bm else "",
    })

    # ── 2. Redispatch volume comparison: model vs BOALF ──────────────────
    model_inc = model_redispatch["increase_MWh"].sum() if "increase_MWh" in model_redispatch.columns else 0
    model_dec = model_redispatch["decrease_MWh"].sum() if "decrease_MWh" in model_redispatch.columns else 0

    boalf_summary = create_boalf_flag_summary(boalf_df)
    if not boalf_df.empty:
        boalf_agg = _aggregate_boalf_by_carrier(boalf_df)
        elexon_inc = boalf_agg["increase_MW"].sum()  # MW-level, convert approx to MWh
        elexon_dec = boalf_agg["decrease_MW"].sum()
        # BOALF level changes are MW at each SP (not energy); approximate MWh as MW * 0.5h
        elexon_inc_mwh = elexon_inc * 0.5
        elexon_dec_mwh = elexon_dec * 0.5
    else:
        elexon_inc_mwh = None
        elexon_dec_mwh = None
        boalf_agg = pd.DataFrame()

    rows.append({
        "metric": "Total increase volume",
        "model_value": f"{model_inc:,.0f}",
        "elexon_value": f"{elexon_inc_mwh:,.0f}" if elexon_inc_mwh is not None else "N/A",
        "unit": "MWh",
        "ratio": f"{model_inc / elexon_inc_mwh:.2f}" if elexon_inc_mwh else "—",
        "note": "BOALF acceptance-level increases (approx MWh)",
    })
    rows.append({
        "metric": "Total decrease volume",
        "model_value": f"{model_dec:,.0f}",
        "elexon_value": f"{elexon_dec_mwh:,.0f}" if elexon_dec_mwh is not None else "N/A",
        "unit": "MWh",
        "ratio": f"{model_dec / elexon_dec_mwh:.2f}" if elexon_dec_mwh else "—",
        "note": "BOALF acceptance-level decreases (approx MWh)",
    })

    if not boalf_summary.empty:
        total_rows = boalf_summary[boalf_summary["scope"] == "total"].set_index("group")
        for label in ["unflagged", "flagged_any", "so_flagged", "stor_flagged"]:
            if label not in total_rows.index:
                continue
            inc_ref = float(total_rows.loc[label, "increase_mwh"])
            dec_ref = float(total_rows.loc[label, "decrease_mwh"])
            pretty = label.replace("_", " ")
            rows.append({
                "metric": f"Total increase volume vs BOALF ({pretty})",
                "model_value": f"{model_inc:,.0f}",
                "elexon_value": f"{inc_ref:,.0f}",
                "unit": "MWh",
                "ratio": f"{model_inc / inc_ref:.2f}" if inc_ref else "—",
                "note": "BOALF subset diagnostic; flagged means any of so/stor/rr",
            })
            rows.append({
                "metric": f"Total decrease volume vs BOALF ({pretty})",
                "model_value": f"{model_dec:,.0f}",
                "elexon_value": f"{dec_ref:,.0f}",
                "unit": "MWh",
                "ratio": f"{model_dec / dec_ref:.2f}" if dec_ref else "—",
                "note": "BOALF subset diagnostic; flagged means any of so/stor/rr",
            })

        if "flagged_any" in total_rows.index and "all" in total_rows.index:
            total_inc_ref = float(total_rows.loc["all", "increase_mwh"])
            flagged_inc_ref = float(total_rows.loc["flagged_any", "increase_mwh"])
            total_dec_ref = float(total_rows.loc["all", "decrease_mwh"])
            flagged_dec_ref = float(total_rows.loc["flagged_any", "decrease_mwh"])
            rows.append({
                "metric": "BOALF flagged share of increase volume",
                "model_value": "—",
                "elexon_value": f"{(100 * flagged_inc_ref / total_inc_ref):.1f}%" if total_inc_ref else "N/A",
                "unit": "%",
                "ratio": "—",
                "note": "Share of BOALF increase volume in flagged subsets",
            })
            rows.append({
                "metric": "BOALF flagged share of decrease volume",
                "model_value": "—",
                "elexon_value": f"{(100 * flagged_dec_ref / total_dec_ref):.1f}%" if total_dec_ref else "N/A",
                "unit": "%",
                "ratio": "—",
                "note": "Share of BOALF decrease volume in flagged subsets",
            })

    # ── 2b. Non-BM balancing-services context (DISBSAD) ─────────────────
    disbsad_summary = create_disbsad_summary(disbsad_df)
    if not disbsad_summary.empty:
        disbsad_totals = disbsad_summary[
            (disbsad_summary["scope"] == "flag_group")
            & (disbsad_summary["group"] == "all")
        ]
        if not disbsad_totals.empty:
            total = disbsad_totals.iloc[0]
            rows.append({
                "metric": "DISBSAD absolute volume",
                "model_value": "—",
                "elexon_value": f"{total['abs_volume_mwh']:,.0f}",
                "unit": "MWh",
                "ratio": "—",
                "note": "Non-BM balancing-service adjustments outside BOALF benchmark",
            })
            rows.append({
                "metric": "DISBSAD net volume",
                "model_value": "—",
                "elexon_value": f"{total['net_volume_mwh']:,.0f}",
                "unit": "MWh",
                "ratio": "—",
                "note": "Signed non-BM balancing-service adjustment volume",
            })
            rows.append({
                "metric": "DISBSAD total cost",
                "model_value": "—",
                "elexon_value": f"£{total['cost_gbp']:,.0f}",
                "unit": "£",
                "ratio": "—",
                "note": "Non-BM balancing-service adjustment cost over solve period",
            })

    # ── 3. Wholesale price comparison ────────────────────────────────────
    if "wholesale_price" in model_prices.columns:
        model_smp_mean = model_prices["wholesale_price"].mean()
    else:
        model_smp_mean = None

    if mid_prices is not None and len(mid_prices) > 0:
        if "wholesale_price" in model_prices.columns:
            common_idx = model_prices.index.intersection(mid_prices.index)
            elexon_mid_mean = (
                mid_prices.loc[common_idx].mean()
                if len(common_idx) > 0 else mid_prices.mean()
            )
            model_smp_mean = (
                model_prices.loc[common_idx, "wholesale_price"].mean()
                if len(common_idx) > 0 else model_smp_mean
            )
        else:
            elexon_mid_mean = mid_prices.mean()
    else:
        elexon_mid_mean = None

    if model_smp_mean is not None:
        rows.append({
            "metric": "Mean wholesale price (SMP)",
            "model_value": f"£{model_smp_mean:.2f}",
            "elexon_value": f"£{elexon_mid_mean:.2f}" if elexon_mid_mean else "N/A",
            "unit": "£/MWh",
            "ratio": f"{model_smp_mean / elexon_mid_mean:.2f}" if elexon_mid_mean else "—",
            "note": "Model SMP vs ELEXON MID (N2EX)",
        })

    # ── 4. System prices — effective price comparison ─────────────────
    # SBP is NOT a nodal price — it's total balancing cost / net imbalance
    # volume, including ancillary services the model doesn't capture.
    # Fairer metric: model_effective_price = SMP + BM_adder, where
    # BM_adder = total_BM_cost / total_demand_MWh.
    if not system_prices.empty and "system_buy_price" in system_prices.columns:
        common_idx = model_prices.index.intersection(system_prices.index)
        sbp_col = system_prices.loc[common_idx, "system_buy_price"] if len(common_idx) > 0 else system_prices["system_buy_price"]
        sbp_mean = sbp_col.mean()

        # Compute BM adder: total constraint cost spread over total demand
        bm_adder = None
        if model_smp_mean is not None and n_hours > 0:
            # model_bm_cost already computed above; total demand in MWh
            # Use model_prices length * average demand (approx)
            # Better: try to get total demand from model_prices if available
            if model_bm_cost != 0:
                # Approximate total demand as hours * average GB demand (~30 GW in 2020)
                # But we have the actual SMP and BM cost, so:
                # effective_price = SMP + BM_cost / (n_hours * avg_demand_MW)
                # For now, compute BM cost per hour as a simpler proxy
                bm_cost_per_hour = model_bm_cost / n_hours if n_hours > 0 else 0
                rows.append({
                    "metric": "BM constraint cost per hour",
                    "model_value": f"£{bm_cost_per_hour:,.0f}",
                    "elexon_value": "—",
                    "unit": "£/hour",
                    "ratio": "—",
                    "note": "Total BM cost / solve hours (constraint management cost rate)",
                })

        # SMP + BM adder vs SBP (fairer comparison)
        if model_smp_mean is not None:
            rows.append({
                "metric": "Mean wholesale price (SMP) vs SBP",
                "model_value": f"£{model_smp_mean:.2f}",
                "elexon_value": f"£{sbp_mean:.2f}",
                "unit": "£/MWh",
                "ratio": f"{model_smp_mean / sbp_mean:.2f}" if sbp_mean != 0 else "—",
                "note": (
                    "SBP includes ancillary services + reserve that the model "
                    "does not capture — expect SMP < SBP. Gap indicates "
                    "non-constraint balancing costs."
                ),
            })

        # Retain nodal price info as a separate diagnostic (not a benchmark)
        if "mean_nodal_price" in model_prices.columns:
            model_nodal_mean = (
                model_prices.loc[common_idx, "mean_nodal_price"].mean()
                if len(common_idx) > 0 else model_prices["mean_nodal_price"].mean()
            )
            rows.append({
                "metric": "Mean BM nodal price (diagnostic)",
                "model_value": f"£{model_nodal_mean:.2f}",
                "elexon_value": "—",
                "unit": "£/MWh",
                "ratio": "—",
                "note": (
                    "LP shadow price at demand buses — reflects constraint "
                    "cost of local power balance. Not directly comparable to SBP."
                ),
            })

        # SBP/SSP spread: proxy for system long/short imbalance
        if "system_sell_price" in system_prices.columns:
            ssp_col = system_prices.loc[common_idx, "system_sell_price"] if len(common_idx) > 0 else system_prices["system_sell_price"]
            sbp_ssp_spread = (sbp_col - ssp_col).mean()
            # Model spread: nodal price range
            if "max_nodal_spread" in model_prices.columns:
                model_spread = model_prices["max_nodal_spread"].mean()
                rows.append({
                    "metric": "Price spread (SBP-SSP / nodal range)",
                    "model_value": f"£{model_spread:.2f}",
                    "elexon_value": f"£{sbp_ssp_spread:.2f}",
                    "unit": "£/MWh",
                    "ratio": f"{model_spread / sbp_ssp_spread:.2f}" if sbp_ssp_spread > 0 else "—",
                    "note": "Model max nodal spread vs ELEXON SBP-SSP mean spread",
                })
            # Fraction of time system was long (SBP > SSP + threshold)
            long_threshold = 5.0  # £5/MWh gap = meaningfully long
            pct_long = (sbp_col > ssp_col + long_threshold).mean() * 100
            rows.append({
                "metric": "% hours system long (SBP-SSP > £5)",
                "model_value": "—",
                "elexon_value": f"{pct_long:.1f}%",
                "unit": "%",
                "ratio": "—",
                "note": "Fraction of periods where system was net long (ELEXON actuals)",
            })

    # ── 5. Dispatch level comparison (B1610) ─────────────────────────────
    if not b1610_df.empty:
        b1610_total = b1610_df.select_dtypes(include="number").clip(lower=0).sum(axis=1)
        if len(b1610_total) > 1:
            inferred_step = b1610_total.index.to_series().diff().dropna().min()
            if pd.notna(inferred_step) and inferred_step < pd.Timedelta(hours=1):
                b1610_total = b1610_total.resample("h").mean()
        b1610_mean_gw = b1610_total.mean() / 1000 if len(b1610_total) > 0 else 0

        model_total = model_balancing_dispatch.select_dtypes(include="number").sum(axis=1)
        common_idx = model_total.index.intersection(b1610_total.index)
        if len(common_idx) > 0:
            b1610_mean_gw = b1610_total.loc[common_idx].mean() / 1000
            model_mean_gw = model_total.loc[common_idx].mean() / 1000
        else:
            model_mean_gw = model_total.mean() / 1000 if len(model_total) > 0 else 0

        rows.append({
            "metric": "Mean physical dispatch",
            "model_value": f"{model_mean_gw:.1f} GW",
            "elexon_value": f"{b1610_mean_gw:.1f} GW (BM units only)",
            "unit": "GW",
            "ratio": f"{model_mean_gw / b1610_mean_gw:.2f}" if b1610_mean_gw > 0 else "—",
            "note": "B1610 covers T_ BMUs only (~35-65% of total)",
        })

    # ── 6. Per-carrier redispatch comparison ─────────────────────────────
    if not boalf_agg.empty and "carrier" in model_redispatch.columns:
        model_by_carrier = model_redispatch.groupby("carrier")[["increase_MWh", "decrease_MWh"]].sum()

        for carrier in boalf_agg["carrier"].unique():
            if carrier == "unknown":
                continue
            elexon_row = boalf_agg[boalf_agg["carrier"] == carrier]
            elexon_inc_c = elexon_row["increase_MW"].values[0] * 0.5 if len(elexon_row) else 0
            model_inc_c = model_by_carrier.loc[carrier, "increase_MWh"] if carrier in model_by_carrier.index else 0

            if elexon_inc_c > 10 or model_inc_c > 10:
                rows.append({
                    "metric": f"Increase volume: {carrier}",
                    "model_value": f"{model_inc_c:,.0f}",
                    "elexon_value": f"{elexon_inc_c:,.0f}",
                    "unit": "MWh",
                    "ratio": f"{model_inc_c / elexon_inc_c:.2f}" if elexon_inc_c > 0 else "—",
                    "note": "Per-carrier BOALF comparison",
                })

    # ── 7. Per-carrier B1610 actual generation vs model dispatch ──────────
    # Classify B1610 BMU columns by carrier and compare hourly generation.
    # NOTE: B1610 covers only BM-participating T_ units (~35-65% of total).
    # We compare against the FULL model physical dispatch per carrier.
    if not b1610_df.empty and not model_balancing_dispatch.empty:
        b1610_carrier_gen = {}
        for col in b1610_df.columns:
            carrier = _classify_bmu_carrier(col)
            if carrier != "unknown":
                vals = b1610_df[col].clip(lower=0)
                b1610_carrier_gen[carrier] = b1610_carrier_gen.get(carrier, 0) + vals

        # Build generator→carrier mapping from redispatch summary
        gen_carrier_map = {}
        if "component" in model_redispatch.columns and "carrier" in model_redispatch.columns:
            for _, rd_row in model_redispatch.iterrows():
                gen_carrier_map[rd_row["component"]] = rd_row["carrier"]

        # Aggregate model balancing dispatch (physical) by carrier
        model_carrier_dispatch = {}
        dispatch_numeric = model_balancing_dispatch.select_dtypes(include="number")
        for col in dispatch_numeric.columns:
            c = gen_carrier_map.get(col)
            if c is not None:
                if c not in model_carrier_dispatch:
                    model_carrier_dispatch[c] = dispatch_numeric[col].copy()
                else:
                    model_carrier_dispatch[c] = model_carrier_dispatch[c] + dispatch_numeric[col]

        # Compute per-carrier mean generation
        for carrier, b1610_series in sorted(b1610_carrier_gen.items()):
            if hasattr(b1610_series, "resample"):
                b1610_hourly = b1610_series.resample("h").mean()
            else:
                b1610_hourly = b1610_series
            elexon_mean_mw = b1610_hourly.mean()
            if elexon_mean_mw < 50:  # skip very small carriers
                continue

            # Use actual model physical dispatch for this carrier
            model_mean_mw = None
            if carrier in model_carrier_dispatch:
                model_series = model_carrier_dispatch[carrier]
                # Align to common timestamps if possible
                common_idx = model_series.index.intersection(b1610_hourly.index)
                if len(common_idx) > 0:
                    model_mean_mw = model_series.loc[common_idx].mean()
                else:
                    model_mean_mw = model_series.mean()

            rows.append({
                "metric": f"B1610 generation: {carrier}",
                "model_value": f"{model_mean_mw:.0f} MW" if model_mean_mw is not None else "—",
                "elexon_value": f"{elexon_mean_mw:.0f} MW",
                "unit": "MW mean",
                "ratio": f"{model_mean_mw / elexon_mean_mw:.2f}" if (
                    model_mean_mw is not None and elexon_mean_mw > 0) else "—",
                "note": "B1610 actual (BM units only) vs model full physical dispatch",
            })

    # ── 8. Per-carrier BM cost diagnostics ───────────────────────────────
    # Show cost breakdown by carrier to highlight dominant cost drivers.
    if "carrier" in model_redispatch.columns and "net_cost" in model_redispatch.columns:
        carrier_costs = model_redispatch.groupby("carrier")["net_cost"].sum().sort_values()
        total_cost = carrier_costs.sum()
        # Show carriers that contribute >5% of total cost magnitude
        cost_threshold = abs(total_cost) * 0.05 if abs(total_cost) > 0 else 1e6
        for carrier, cost in carrier_costs.items():
            if abs(cost) >= cost_threshold:
                pct = (cost / total_cost * 100) if total_cost != 0 else 0
                rows.append({
                    "metric": f"BM cost: {carrier}",
                    "model_value": f"£{cost:,.0f}",
                    "elexon_value": "—",
                    "unit": "£",
                    "ratio": f"{pct:.1f}%",
                    "note": "% of total BM cost",
                })

    # ── 9. Nuclear / wind curtailment diagnostics ─────────────────────────
    if "carrier" in model_redispatch.columns:
        for diag_carrier in ["nuclear", "wind_offshore", "wind_onshore"]:
            row = model_redispatch[model_redispatch["carrier"] == diag_carrier]
            if not row.empty:
                dec_mwh = row["decrease_MWh"].sum()
                inc_mwh = row["increase_MWh"].sum()
                rows.append({
                    "metric": f"Curtailment: {diag_carrier}",
                    "model_value": f"{dec_mwh:,.0f} MWh decrease / {inc_mwh:,.0f} MWh increase",
                    "elexon_value": "—",
                    "unit": "MWh",
                    "ratio": f"{dec_mwh / (inc_mwh + 1):.1f}x" if inc_mwh >= 0 else "—",
                    "note": "dec/inc ratio (>1 means net turn-down)",
                })

    report = pd.DataFrame(rows)
    return report


def create_validation_dashboard(
    model_redispatch: pd.DataFrame,
    model_prices: pd.DataFrame,
    model_balancing_dispatch: pd.DataFrame,
    boalf_df: pd.DataFrame,
    system_prices: pd.DataFrame,
    b1610_df: pd.DataFrame,
    mid_prices: Optional[pd.Series],
    output_path: str,
    scenario_id: str,
    logger: logging.Logger,
):
    """
    Create a multi-panel Plotly dashboard for BM validation.

    Panels:
      1. Price duration curve: model SMP vs ELEXON MID vs SBP
      2. Redispatch volumes: model vs BOALF by carrier
      3. System price time series: model nodal vs ELEXON SBP
      4. Dispatch comparison: model vs B1610 total generation
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available — skipping BM validation dashboard")
        # Write a placeholder HTML
        Path(output_path).write_text(
            "<html><body><h1>Plotly not installed — BM validation dashboard unavailable</h1></body></html>"
        )
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Price Duration Curve: Model vs ELEXON",
            "Redispatch Volumes by Carrier: Model vs BOALF",
            "Price Time Series: Model Nodal vs ELEXON SBP",
            "Total Dispatch: Model vs B1610",
            "Per-Carrier Generation: Model vs B1610",
            "BM Cost Breakdown by Carrier",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # ── Panel 1: Price duration curve ────────────────────────────────────
    if "wholesale_price" in model_prices.columns:
        smp = model_prices["wholesale_price"].dropna().sort_values(ascending=False).values
        x_pct = np.linspace(0, 100, len(smp))
        fig.add_trace(
            go.Scatter(x=x_pct, y=smp, mode="lines", name="Model SMP",
                       line=dict(color="#1f77b4", width=2)),
            row=1, col=1,
        )

    if mid_prices is not None and len(mid_prices) > 0:
        mid_sorted = mid_prices.dropna().sort_values(ascending=False).values
        x_pct_mid = np.linspace(0, 100, len(mid_sorted))
        fig.add_trace(
            go.Scatter(x=x_pct_mid, y=mid_sorted, mode="lines", name="ELEXON MID",
                       line=dict(color="#ff7f0e", width=2)),
            row=1, col=1,
        )

    if not system_prices.empty and "system_buy_price" in system_prices.columns:
        sbp_sorted = system_prices["system_buy_price"].dropna().sort_values(ascending=False).values
        x_pct_sbp = np.linspace(0, 100, len(sbp_sorted))
        fig.add_trace(
            go.Scatter(x=x_pct_sbp, y=sbp_sorted, mode="lines", name="ELEXON SBP",
                       line=dict(color="#2ca02c", width=2, dash="dash")),
            row=1, col=1,
        )
    fig.update_xaxes(title_text="% of time", row=1, col=1)
    fig.update_yaxes(title_text="£/MWh", row=1, col=1)

    # ── Panel 2: Redispatch by carrier comparison ────────────────────────
    if not boalf_df.empty and "carrier" in model_redispatch.columns:
        boalf_agg = _aggregate_boalf_by_carrier(boalf_df)
        model_by_c = model_redispatch.groupby("carrier")["increase_MWh"].sum()

        # Merge on common carriers
        carriers = sorted(set(boalf_agg["carrier"].unique()) | set(model_by_c.index))
        carriers = [c for c in carriers if c != "unknown"]
        model_vals = [model_by_c.get(c, 0) for c in carriers]
        elexon_vals = []
        for c in carriers:
            row = boalf_agg[boalf_agg["carrier"] == c]
            elexon_vals.append(row["increase_MW"].values[0] * 0.5 if len(row) else 0)

        fig.add_trace(
            go.Bar(x=carriers, y=model_vals, name="Model (MWh)", marker_color="#1f77b4"),
            row=1, col=2,
        )
        fig.add_trace(
            go.Bar(x=carriers, y=elexon_vals, name="BOALF (approx MWh)", marker_color="#ff7f0e"),
            row=1, col=2,
        )
        fig.update_yaxes(title_text="MWh", row=1, col=2)

    # ── Panel 3: Price time series comparison ────────────────────────────
    if "wholesale_price" in model_prices.columns:
        fig.add_trace(
            go.Scatter(x=model_prices.index, y=model_prices["wholesale_price"],
                       mode="lines", name="Model SMP", line=dict(color="#1f77b4", width=2)),
            row=2, col=1,
        )
    if "mean_nodal_price" in model_prices.columns:
        fig.add_trace(
            go.Scatter(x=model_prices.index, y=model_prices["mean_nodal_price"],
                       mode="lines", name="Model Nodal Mean",
                       line=dict(color="#9467bd", width=1.5)),
            row=2, col=1,
        )
    if not system_prices.empty and "system_buy_price" in system_prices.columns:
        # Align to model timestamps
        common_idx = model_prices.index.intersection(system_prices.index)
        if len(common_idx) > 0:
            fig.add_trace(
                go.Scatter(x=common_idx, y=system_prices.loc[common_idx, "system_buy_price"],
                           mode="lines", name="ELEXON SBP",
                           line=dict(color="#2ca02c", width=2, dash="dash")),
                row=2, col=1,
            )
    if mid_prices is not None and len(mid_prices) > 0:
        common_idx = model_prices.index.intersection(mid_prices.index)
        if len(common_idx) > 0:
            fig.add_trace(
                go.Scatter(x=common_idx, y=mid_prices.loc[common_idx],
                           mode="lines", name="ELEXON MID",
                           line=dict(color="#ff7f0e", width=1.5, dash="dot")),
                row=2, col=1,
            )
    fig.update_yaxes(title_text="£/MWh", row=2, col=1)

    # ── Panel 4: Dispatch comparison ─────────────────────────────────────
    model_total = model_balancing_dispatch.select_dtypes(include="number").sum(axis=1) / 1000
    fig.add_trace(
        go.Scatter(x=model_total.index, y=model_total.values,
                   mode="lines", name="Model Physical (GW)",
                   line=dict(color="#1f77b4", width=2)),
        row=2, col=2,
    )

    if not b1610_df.empty:
        b1610_total = b1610_df.select_dtypes(include="number").clip(lower=0).sum(axis=1) / 1000
        # Resample to hourly if needed
        if hasattr(b1610_total.index, "freq") and b1610_total.index.freq != "h":
            b1610_total = b1610_total.resample("h").mean()
        fig.add_trace(
            go.Scatter(x=b1610_total.index, y=b1610_total.values,
                       mode="lines", name="ELEXON B1610 BM units (GW)",
                       line=dict(color="#ff7f0e", width=2, dash="dash")),
            row=2, col=2,
        )
    fig.update_yaxes(title_text="GW", row=2, col=2)

    # ── Panel 5: Per-carrier B1610 vs model dispatch ──────────────────
    _n_hours = len(model_balancing_dispatch) if len(model_balancing_dispatch) > 0 else 24
    if not b1610_df.empty:
        b1610_carrier_totals = {}
        for col in b1610_df.columns:
            carrier = _classify_bmu_carrier(col)
            if carrier != "unknown":
                vals = b1610_df[col].clip(lower=0)
                b1610_carrier_totals[carrier] = b1610_carrier_totals.get(carrier, 0) + vals.mean()

        # Build generator→carrier mapping and aggregate physical dispatch by carrier
        gen_carrier_map = {}
        if "component" in model_redispatch.columns and "carrier" in model_redispatch.columns:
            for _, rd_row in model_redispatch.iterrows():
                gen_carrier_map[rd_row["component"]] = rd_row["carrier"]

        dispatch_numeric = model_balancing_dispatch.select_dtypes(include="number")
        model_carrier_totals = {}
        for col in dispatch_numeric.columns:
            c = gen_carrier_map.get(col)
            if c is not None:
                model_carrier_totals[c] = model_carrier_totals.get(c, 0) + dispatch_numeric[col].mean()
        model_carrier_totals = pd.Series(model_carrier_totals)

        carriers_5 = sorted(
            set(b1610_carrier_totals.keys()) | set(model_carrier_totals.index)
        )
        carriers_5 = [c for c in carriers_5 if
                      b1610_carrier_totals.get(c, 0) > 50 or
                      model_carrier_totals.get(c, 0) > 50]
        b1610_vals = [b1610_carrier_totals.get(c, 0) for c in carriers_5]
        model_vals_5 = [model_carrier_totals.get(c, 0) for c in carriers_5]

        fig.add_trace(
            go.Bar(x=carriers_5, y=b1610_vals, name="B1610 actual (MW mean)",
                   marker_color="#2ca02c"),
            row=3, col=1,
        )
        fig.add_trace(
            go.Bar(x=carriers_5, y=model_vals_5, name="Model redispatch (MW mean)",
                   marker_color="#1f77b4"),
            row=3, col=1,
        )
        fig.update_yaxes(title_text="MW mean", row=3, col=1)

    # ── Panel 6: BM cost breakdown by carrier ─────────────────────────
    if "carrier" in model_redispatch.columns and "net_cost" in model_redispatch.columns:
        cost_by_carrier = model_redispatch.groupby("carrier")["net_cost"].sum().sort_values()
        fig.add_trace(
            go.Bar(
                x=cost_by_carrier.index.tolist(),
                y=cost_by_carrier.values.tolist(),
                name="Net BM cost (£)",
                marker_color=[
                    "#d62728" if v > 0 else "#2ca02c"
                    for v in cost_by_carrier.values
                ],
            ),
            row=3, col=2,
        )
        fig.update_yaxes(title_text="£", row=3, col=2)

    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=1300, width=1400,
        title_text=f"BM Validation: {scenario_id} vs ELEXON Actuals",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08),
        template="plotly_white",
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    logger.info(f"BM validation dashboard saved to {output_path}")


def validate_bm_results(
    scenario_config: dict,
    redispatch_csv: str,
    costs_csv: str,
    prices_csv: str,
    wholesale_dispatch_csv: str,
    balancing_dispatch_csv: str,
    output_csv: str,
    output_html: str,
    boalf_by_flag_csv: str,
    disbsad_summary_csv: str,
    logger: logging.Logger,
):
    """
    Main entry point: run full BM validation for a historical scenario.

    Parameters
    ----------
    scenario_config : dict
        Scenario configuration dict (must include modelled_year, solve_period).
    redispatch_csv, costs_csv, prices_csv : str
        Paths to model result CSVs.
    wholesale_dispatch_csv, balancing_dispatch_csv : str
        Paths to dispatch CSVs.
    output_csv, output_html : str
        Output paths for validation report and dashboard.
    logger : Logger
    """
    scenario_id = scenario_config.get("scenario_id", "unknown")
    modelled_year = scenario_config.get("modelled_year", 2020)

    if modelled_year > 2024:
        logger.info(f"Future scenario (year {modelled_year}) — ELEXON BM validation not available")
        pd.DataFrame([{"metric": "Status", "model_value": "Future scenario",
                        "elexon_value": "N/A", "unit": "", "ratio": "", "note": "No ELEXON data for future years"}]).to_csv(output_csv, index=False)
        pd.DataFrame().to_csv(boalf_by_flag_csv, index=False)
        pd.DataFrame().to_csv(disbsad_summary_csv, index=False)
        Path(output_html).write_text(
            f"<html><body><h1>BM Validation: {scenario_id}</h1>"
            f"<p>Future scenario ({modelled_year}) — no ELEXON data available for validation.</p></body></html>"
        )
        return

    # Load model results
    model_redispatch = pd.read_csv(redispatch_csv)
    model_costs = pd.read_csv(costs_csv)
    model_prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    model_wholesale = pd.read_csv(wholesale_dispatch_csv, index_col=0, parse_dates=True)
    model_balancing = pd.read_csv(balancing_dispatch_csv, index_col=0, parse_dates=True)

    # Determine solve period
    solve_period = scenario_config.get("solve_period", {})
    start_date = solve_period.get("start", "2020-01-01 00:00")[:10]
    end_date = solve_period.get("end", "2020-01-07 23:00")[:10]

    # Load or fetch ELEXON validation data
    cache_dir = f"resources/market/elexon/validation/{modelled_year}"
    data = _load_or_fetch_validation_data_checked(
        start_date, end_date, cache_dir, logger
    )
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

    # Load BOALF
    boalf_path = data.get("boalf_file", "")
    boalf_df = pd.DataFrame()
    if boalf_path and Path(boalf_path).exists():
        boalf_df = pd.read_csv(boalf_path)
        if "timeFrom" in boalf_df.columns:
            boalf_df["timeFrom"] = pd.to_datetime(
                boalf_df["timeFrom"], utc=True, errors="coerce"
            ).dt.tz_localize(None)
            boalf_df = boalf_df[
                (boalf_df["timeFrom"] >= start_ts)
                & (boalf_df["timeFrom"] <= end_ts + pd.Timedelta(minutes=59))
            ]
        if "datetime" in boalf_df.columns:
            boalf_df["datetime"] = pd.to_datetime(boalf_df["datetime"])
            boalf_df = boalf_df[
                (boalf_df["datetime"] >= start_ts)
                & (boalf_df["datetime"] <= end_ts)
            ]
        logger.info(f"Loaded BOALF: {len(boalf_df)} records")

    # Load system prices
    sys_prices_path = data.get("system_prices_file", "")
    sys_prices = pd.DataFrame()
    if sys_prices_path and Path(sys_prices_path).exists():
        try:
            sys_prices = pd.read_csv(sys_prices_path, index_col=0, parse_dates=True)
            sys_prices = sys_prices.loc[start_ts:end_ts]
            logger.info(f"Loaded system prices: {len(sys_prices)} periods")
        except Exception:
            pass

    # Load B1610
    b1610_path = data.get("b1610_file", "")
    b1610_df = pd.DataFrame()
    if b1610_path and Path(b1610_path).exists():
        try:
            b1610_df = pd.read_csv(b1610_path, index_col=0, parse_dates=True)
            b1610_df = b1610_df.loc[start_ts:end_ts + pd.Timedelta(minutes=59)]
            logger.info(f"Loaded B1610: {b1610_df.shape}")
        except Exception:
            pass

    # Load DISBSAD
    disbsad_path = data.get("disbsad_file", "")
    disbsad_df = pd.DataFrame()
    if disbsad_path and Path(disbsad_path).exists():
        try:
            disbsad_df = pd.read_csv(disbsad_path)
            disbsad_df = _normalise_disbsad_dataframe(disbsad_df)
            if "datetime" in disbsad_df.columns:
                disbsad_df = disbsad_df[
                    (disbsad_df["datetime"] >= start_ts)
                    & (disbsad_df["datetime"] <= end_ts + pd.Timedelta(minutes=59))
                ]
            logger.info(f"Loaded DISBSAD: {len(disbsad_df)} records")
        except Exception as e:
            logger.warning(f"Failed to load DISBSAD: {e}")

    boalf_flag_summary = create_boalf_flag_summary(boalf_df)
    boalf_flag_summary.to_csv(boalf_by_flag_csv, index=False)
    logger.info(f"BOALF flag summary: {boalf_by_flag_csv} ({len(boalf_flag_summary)} rows)")

    disbsad_summary = create_disbsad_summary(disbsad_df)
    disbsad_summary.to_csv(disbsad_summary_csv, index=False)
    logger.info(f"DISBSAD summary: {disbsad_summary_csv} ({len(disbsad_summary)} rows)")

    # Load MID prices from calibration cache
    mid_path = Path(f"resources/market/elexon/mid_prices_{modelled_year}.csv")
    mid_prices = None
    if mid_path.exists():
        try:
            mid_raw = pd.read_csv(mid_path, index_col=0)
            mid_raw.index = pd.to_datetime(mid_raw.index)
            mid_period = mid_raw.loc[start_ts:end_ts]
            if len(mid_period) > 0 and "mid_price" in mid_period.columns:
                mid_prices = mid_period["mid_price"]
            elif len(mid_period.columns) == 1:
                mid_prices = mid_period.iloc[:, 0]
            # Resample to hourly if half-hourly
            if mid_prices is not None and len(mid_prices) > 0:
                mid_prices = mid_prices.resample("h").mean()
                logger.info(f"Loaded MID prices: {len(mid_prices)} hourly periods")
        except Exception as e:
            logger.warning(f"Failed to load MID prices: {e}")

    # Create validation report
    report = create_validation_report(
        model_redispatch=model_redispatch,
        model_costs=model_costs,
        model_prices=model_prices,
        model_wholesale_dispatch=model_wholesale,
        model_balancing_dispatch=model_balancing,
        boalf_df=boalf_df,
        disbsad_df=disbsad_df,
        system_prices=sys_prices,
        b1610_df=b1610_df,
        mid_prices=mid_prices,
        scenario_id=scenario_id,
        modelled_year=modelled_year,
        logger=logger,
    )

    report.to_csv(output_csv, index=False)
    logger.info(f"Validation report: {output_csv} ({len(report)} metrics)")

    # Create validation dashboard
    create_validation_dashboard(
        model_redispatch=model_redispatch,
        model_prices=model_prices,
        model_balancing_dispatch=model_balancing,
        boalf_df=boalf_df,
        system_prices=sys_prices,
        b1610_df=b1610_df,
        mid_prices=mid_prices,
        output_path=output_html,
        scenario_id=scenario_id,
        logger=logger,
    )

    logger.info("BM validation complete")


if __name__ == "__main__":
    log_path = (
        snakemake.log[0]
        if hasattr(snakemake, "log") and snakemake.log
        else "validate_bm"
    )
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("BM VALIDATION — MODEL vs ELEXON ACTUALS")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        scenario_config = snakemake.params.scenario_config
        scenario_id = scenario_config.get("scenario_id", snakemake.wildcards.scenario)
        scenario_config["scenario_id"] = scenario_id

        validate_bm_results(
            scenario_config=scenario_config,
            redispatch_csv=snakemake.input.redispatch_summary_csv,
            costs_csv=snakemake.input.constraint_costs_csv,
            prices_csv=snakemake.input.price_comparison_csv,
            wholesale_dispatch_csv=snakemake.input.wholesale_dispatch_csv,
            balancing_dispatch_csv=snakemake.input.balancing_dispatch_csv,
            output_csv=snakemake.output.validation_csv,
            output_html=snakemake.output.validation_html,
            boalf_by_flag_csv=snakemake.output.boalf_by_flag_csv,
            disbsad_summary_csv=snakemake.output.disbsad_summary_csv,
            logger=logger,
        )

        elapsed = time.time() - start_time
        logger.info(f"BM validation completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"FATAL ERROR in BM validation: {e}", exc_info=True)
        raise
