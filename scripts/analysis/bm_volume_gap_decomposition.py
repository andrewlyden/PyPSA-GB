"""
BM volume gap decomposition.

This is a post-processing diagnostic for historical market validation runs. It
compares model redispatch against ELEXON BOALF at carrier, BMU, and hourly level
to identify why model BM volumes differ from observed acceptances.

Example:
    python scripts/analysis/bm_volume_gap_decomposition.py --scenario Validation_Jun2020
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market.validate_bm import (
    _classify_bmu_carrier,
    _normalise_boalf_dataframe,
    _normalise_disbsad_dataframe,
)


def _read_timeseries_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
    df = df.rename(columns={first_col: "snapshot"}).set_index("snapshot")
    return df


def _period_from_model(price_comparison_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    prices = pd.read_csv(price_comparison_path, usecols=["snapshot"])
    snapshots = pd.to_datetime(prices["snapshot"], errors="coerce").dropna()
    if snapshots.empty:
        raise ValueError(f"Could not infer solve period from {price_comparison_path}")
    return snapshots.min(), snapshots.max()


def _prepare_boalf_acceptance_records(boalf_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse BOALF ramp rows into acceptance-level net changes."""
    df = _normalise_boalf_dataframe(boalf_df)
    columns = [
        "datetime",
        "hour",
        "bmu_id",
        "acceptance_number",
        "carrier",
        "net_delta_mw",
        "increase_mwh",
        "decrease_mwh",
        "so_flag",
        "stor_flag",
        "rr_flag",
        "any_flag",
        "unflagged",
    ]
    if df.empty or "bmu_id" not in df.columns:
        return pd.DataFrame(columns=columns)
    if "level_from" not in df.columns or "level_to" not in df.columns:
        return pd.DataFrame(columns=columns)

    time_col = "timeFrom" if "timeFrom" in df.columns else "datetime" if "datetime" in df.columns else None
    group_cols = ["bmu_id", "acceptance_number"] if "acceptance_number" in df.columns else ["bmu_id"]
    records = []

    for _, group in df.groupby(group_cols, dropna=False):
        if time_col:
            group = group.sort_values(time_col)
            timestamp = pd.to_datetime(group[time_col].iloc[0], errors="coerce")
        else:
            timestamp = pd.NaT

        net_delta = float(group["level_to"].iloc[-1]) - float(group["level_from"].iloc[0])
        so_flag = bool(group["so_flag"].any())
        stor_flag = bool(group["stor_flag"].any())
        rr_flag = bool(group["rr_flag"].any())
        any_flag = so_flag or stor_flag or rr_flag

        records.append(
            {
                "datetime": timestamp,
                "hour": timestamp.floor("h") if pd.notna(timestamp) else pd.NaT,
                "bmu_id": str(group["bmu_id"].iloc[0]),
                "acceptance_number": group["acceptance_number"].iloc[0]
                if "acceptance_number" in group.columns
                else pd.NA,
                "carrier": _classify_bmu_carrier(str(group["bmu_id"].iloc[0])),
                "net_delta_mw": net_delta,
                "increase_mwh": max(net_delta, 0.0) * 0.5,
                "decrease_mwh": max(-net_delta, 0.0) * 0.5,
                "so_flag": so_flag,
                "stor_flag": stor_flag,
                "rr_flag": rr_flag,
                "any_flag": any_flag,
                "unflagged": not any_flag,
            }
        )

    return pd.DataFrame(records, columns=columns)


def _load_boalf(boalf_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    boalf = pd.read_csv(boalf_path)
    boalf = _normalise_boalf_dataframe(boalf)
    end_inclusive = end + pd.Timedelta(minutes=59)

    if "timeFrom" in boalf.columns:
        boalf = boalf[
            (boalf["timeFrom"] >= start) & (boalf["timeFrom"] <= end_inclusive)
        ]
    elif "datetime" in boalf.columns:
        boalf = boalf[
            (boalf["datetime"] >= start) & (boalf["datetime"] <= end_inclusive)
        ]
    return boalf


def _load_disbsad(disbsad_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not disbsad_path.exists():
        return pd.DataFrame()
    disbsad = pd.read_csv(disbsad_path)
    disbsad = _normalise_disbsad_dataframe(disbsad)
    if "datetime" in disbsad.columns:
        disbsad = disbsad[
            (disbsad["datetime"] >= start)
            & (disbsad["datetime"] <= end + pd.Timedelta(minutes=59))
        ]
    return disbsad


def _model_by_carrier(redispatch: pd.DataFrame) -> pd.DataFrame:
    required = {"carrier", "increase_MWh", "decrease_MWh", "offer_cost", "bid_cost"}
    missing = required - set(redispatch.columns)
    if missing:
        raise ValueError(f"Redispatch CSV missing columns: {sorted(missing)}")

    out = (
        redispatch.groupby("carrier", dropna=False)
        .agg(
            model_inc_mwh=("increase_MWh", "sum"),
            model_dec_mwh=("decrease_MWh", "sum"),
            model_offer_cost=("offer_cost", "sum"),
            model_bid_cost=("bid_cost", "sum"),
            n_model_components=("component", "nunique"),
        )
        .reset_index()
    )
    out["model_gross_mwh"] = out["model_inc_mwh"] + out["model_dec_mwh"]
    return out


def _boalf_by_carrier(acc: pd.DataFrame, label: str, mask: pd.Series) -> pd.DataFrame:
    subset = acc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["carrier", f"boalf_{label}_inc_mwh", f"boalf_{label}_dec_mwh"])
    out = (
        subset.groupby("carrier", dropna=False)
        .agg(
            **{
                f"boalf_{label}_inc_mwh": ("increase_mwh", "sum"),
                f"boalf_{label}_dec_mwh": ("decrease_mwh", "sum"),
                f"boalf_{label}_acceptances": ("acceptance_number", "count"),
                f"boalf_{label}_bmus": ("bmu_id", "nunique"),
            }
        )
        .reset_index()
    )
    out[f"boalf_{label}_gross_mwh"] = (
        out[f"boalf_{label}_inc_mwh"] + out[f"boalf_{label}_dec_mwh"]
    )
    return out


def build_carrier_gap(redispatch: pd.DataFrame, boalf_acc: pd.DataFrame) -> pd.DataFrame:
    out = _model_by_carrier(redispatch)
    groups = {
        "all": boalf_acc.index == boalf_acc.index,
        "unflagged": boalf_acc["unflagged"],
        "flagged": boalf_acc["any_flag"],
        "so": boalf_acc["so_flag"],
        "stor": boalf_acc["stor_flag"],
        "rr": boalf_acc["rr_flag"],
    }
    for label, mask in groups.items():
        out = out.merge(_boalf_by_carrier(boalf_acc, label, mask), on="carrier", how="outer")

    numeric_cols = [c for c in out.columns if c != "carrier"]
    out[numeric_cols] = out[numeric_cols].fillna(0.0)

    for scope in ["all", "unflagged", "flagged"]:
        out[f"inc_gap_vs_{scope}_mwh"] = out["model_inc_mwh"] - out[f"boalf_{scope}_inc_mwh"]
        out[f"dec_gap_vs_{scope}_mwh"] = out["model_dec_mwh"] - out[f"boalf_{scope}_dec_mwh"]
        out[f"gross_gap_vs_{scope}_mwh"] = out["model_gross_mwh"] - out[f"boalf_{scope}_gross_mwh"]
        out[f"inc_ratio_vs_{scope}"] = out["model_inc_mwh"] / out[f"boalf_{scope}_inc_mwh"].replace(0, np.nan)
        out[f"dec_ratio_vs_{scope}"] = out["model_dec_mwh"] / out[f"boalf_{scope}_dec_mwh"].replace(0, np.nan)
        out[f"gross_ratio_vs_{scope}"] = out["model_gross_mwh"] / out[f"boalf_{scope}_gross_mwh"].replace(0, np.nan)

    return out.sort_values("boalf_all_gross_mwh", ascending=False)


def build_bmu_gap(
    redispatch: pd.DataFrame,
    boalf_acc: pd.DataFrame,
    bmu_mapping: pd.DataFrame,
) -> pd.DataFrame:
    model_component = (
        redispatch.groupby("component", dropna=False)
        .agg(
            model_component_inc_mwh=("increase_MWh", "sum"),
            model_component_dec_mwh=("decrease_MWh", "sum"),
            model_carrier=("carrier", "first"),
            model_type=("type", "first"),
        )
        .reset_index()
        .rename(columns={"component": "generator_name"})
    )
    model_component["model_component_gross_mwh"] = (
        model_component["model_component_inc_mwh"] + model_component["model_component_dec_mwh"]
    )

    bmu = bmu_mapping.copy()
    bmu = bmu.rename(columns={"carrier": "mapped_carrier"})
    bmu = bmu.merge(model_component, on="generator_name", how="left")

    bmu_model = (
        bmu.groupby("bmu_id", dropna=False)
        .agg(
            generator_name=("generator_name", "first"),
            station_name=("station_name", "first"),
            mapped_carrier=("mapped_carrier", "first"),
            match_method=("match_method", "first"),
            model_component_inc_mwh=("model_component_inc_mwh", "sum"),
            model_component_dec_mwh=("model_component_dec_mwh", "sum"),
            model_component_gross_mwh=("model_component_gross_mwh", "sum"),
        )
        .reset_index()
    )

    acc = boalf_acc.copy()
    acc["unflagged_inc_mwh"] = acc["increase_mwh"].where(acc["unflagged"], 0.0)
    acc["unflagged_dec_mwh"] = acc["decrease_mwh"].where(acc["unflagged"], 0.0)
    acc["flagged_inc_mwh"] = acc["increase_mwh"].where(acc["any_flag"], 0.0)
    acc["flagged_dec_mwh"] = acc["decrease_mwh"].where(acc["any_flag"], 0.0)
    boalf_bmu = (
        acc.groupby(["bmu_id", "carrier"], dropna=False)
        .agg(
            boalf_all_inc_mwh=("increase_mwh", "sum"),
            boalf_all_dec_mwh=("decrease_mwh", "sum"),
            boalf_unflagged_inc_mwh=("unflagged_inc_mwh", "sum"),
            boalf_unflagged_dec_mwh=("unflagged_dec_mwh", "sum"),
            boalf_flagged_inc_mwh=("flagged_inc_mwh", "sum"),
            boalf_flagged_dec_mwh=("flagged_dec_mwh", "sum"),
            n_acceptances=("acceptance_number", "count"),
        )
        .reset_index()
    )
    boalf_bmu["boalf_all_gross_mwh"] = boalf_bmu["boalf_all_inc_mwh"] + boalf_bmu["boalf_all_dec_mwh"]
    boalf_bmu["boalf_unflagged_gross_mwh"] = (
        boalf_bmu["boalf_unflagged_inc_mwh"] + boalf_bmu["boalf_unflagged_dec_mwh"]
    )
    boalf_bmu["boalf_flagged_gross_mwh"] = (
        boalf_bmu["boalf_flagged_inc_mwh"] + boalf_bmu["boalf_flagged_dec_mwh"]
    )

    # The model is generator-based, while BOALF is BMU-based. The mapping below
    # connects BOALF BMUs to model generators and attributes the full generator
    # redispatch to each BMU. This is a diagnostic for coverage, not a strict
    # one-to-one volume reconciliation.
    mapped = bmu_model.merge(boalf_bmu, on="bmu_id", how="right")

    for col in [
        "model_component_inc_mwh",
        "model_component_dec_mwh",
        "model_component_gross_mwh",
    ]:
        mapped[col] = pd.to_numeric(mapped[col], errors="coerce").fillna(0.0)

    mapped["mapped_to_model"] = mapped["generator_name"].notna()
    mapped["model_generator_has_redispatch"] = mapped["model_component_gross_mwh"] > 1e-6
    mapped["unmodelled_or_unmapped"] = ~mapped["mapped_to_model"]
    return mapped.sort_values("boalf_all_gross_mwh", ascending=False)


def build_mapping_summary(bmu_gap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for carrier, group in bmu_gap.groupby("carrier", dropna=False):
        total = float(group["boalf_all_gross_mwh"].sum())
        unmapped = float(group.loc[group["unmodelled_or_unmapped"], "boalf_all_gross_mwh"].sum())
        inactive = float(
            group.loc[
                group["mapped_to_model"] & ~group["model_generator_has_redispatch"],
                "boalf_all_gross_mwh",
            ].sum()
        )
        active = float(group.loc[group["model_generator_has_redispatch"], "boalf_all_gross_mwh"].sum())
        rows.append(
            {
                "carrier": carrier,
                "boalf_all_gross_mwh": total,
                "unmapped_boalf_gross_mwh": unmapped,
                "mapped_but_no_model_redispatch_boalf_gross_mwh": inactive,
                "mapped_and_model_redispatched_boalf_gross_mwh": active,
                "unmapped_share": unmapped / total if total else np.nan,
                "inactive_mapped_share": inactive / total if total else np.nan,
                "n_boalf_bmus": int(group["bmu_id"].nunique()),
                "n_unmapped_bmus": int(group.loc[group["unmodelled_or_unmapped"], "bmu_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("boalf_all_gross_mwh", ascending=False)


def build_mapped_carrier_gap(redispatch: pd.DataFrame, bmu_gap: pd.DataFrame) -> pd.DataFrame:
    """Compare model volumes to BOALF after applying the BMU crosswalk carrier."""
    mapped = bmu_gap.copy()
    mapped["comparison_carrier"] = mapped["mapped_carrier"].where(
        mapped["mapped_carrier"].notna() & (mapped["mapped_carrier"].astype(str).str.len() > 0),
        mapped["carrier"],
    )

    boalf = (
        mapped.groupby("comparison_carrier", dropna=False)
        .agg(
            boalf_mapped_all_inc_mwh=("boalf_all_inc_mwh", "sum"),
            boalf_mapped_all_dec_mwh=("boalf_all_dec_mwh", "sum"),
            boalf_mapped_all_gross_mwh=("boalf_all_gross_mwh", "sum"),
            boalf_mapped_unflagged_inc_mwh=("boalf_unflagged_inc_mwh", "sum"),
            boalf_mapped_unflagged_dec_mwh=("boalf_unflagged_dec_mwh", "sum"),
            boalf_mapped_unflagged_gross_mwh=("boalf_unflagged_gross_mwh", "sum"),
            boalf_mapped_flagged_inc_mwh=("boalf_flagged_inc_mwh", "sum"),
            boalf_mapped_flagged_dec_mwh=("boalf_flagged_dec_mwh", "sum"),
            boalf_mapped_flagged_gross_mwh=("boalf_flagged_gross_mwh", "sum"),
            n_boalf_bmus=("bmu_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"comparison_carrier": "carrier"})
    )

    out = _model_by_carrier(redispatch).merge(boalf, on="carrier", how="outer")
    numeric_cols = [c for c in out.columns if c != "carrier"]
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    for scope in ["all", "unflagged", "flagged"]:
        out[f"gross_gap_vs_mapped_{scope}_mwh"] = (
            out["model_gross_mwh"] - out[f"boalf_mapped_{scope}_gross_mwh"]
        )
        out[f"gross_ratio_vs_mapped_{scope}"] = (
            out["model_gross_mwh"] / out[f"boalf_mapped_{scope}_gross_mwh"].replace(0, np.nan)
        )
    return out.sort_values("boalf_mapped_all_gross_mwh", ascending=False)


def build_carrier_conflicts(bmu_gap: pd.DataFrame) -> pd.DataFrame:
    """List BOALF-prefix carrier classifications that disagree with BMU mapping."""
    conflicts = bmu_gap[
        bmu_gap["mapped_carrier"].notna()
        & bmu_gap["carrier"].notna()
        & (bmu_gap["mapped_carrier"].astype(str) != bmu_gap["carrier"].astype(str))
    ].copy()
    if conflicts.empty:
        return pd.DataFrame(
            columns=[
                "bmu_id",
                "carrier",
                "mapped_carrier",
                "boalf_all_gross_mwh",
                "boalf_unflagged_gross_mwh",
                "generator_name",
                "station_name",
                "match_method",
            ]
        )
    return conflicts[
        [
            "bmu_id",
            "carrier",
            "mapped_carrier",
            "boalf_all_gross_mwh",
            "boalf_unflagged_gross_mwh",
            "generator_name",
            "station_name",
            "match_method",
        ]
    ].sort_values("boalf_all_gross_mwh", ascending=False)


def build_hourly_gap(redispatch: pd.DataFrame, boalf_acc: pd.DataFrame) -> pd.DataFrame:
    # Redispatch summary is component-total only. For hourly model volumes use
    # the solved schedule difference files when available in main().
    boalf_groups = {
        "all": boalf_acc.index == boalf_acc.index,
        "unflagged": boalf_acc["unflagged"],
        "flagged": boalf_acc["any_flag"],
    }
    frames = []
    for label, mask in boalf_groups.items():
        subset = boalf_acc[mask & boalf_acc["hour"].notna()]
        hourly = (
            subset.groupby("hour")
            .agg(
                **{
                    f"boalf_{label}_inc_mwh": ("increase_mwh", "sum"),
                    f"boalf_{label}_dec_mwh": ("decrease_mwh", "sum"),
                    f"boalf_{label}_gross_mwh": ("increase_mwh", lambda s: float(s.sum())),
                }
            )
            .reset_index()
        )
        hourly[f"boalf_{label}_gross_mwh"] = (
            hourly[f"boalf_{label}_inc_mwh"] + hourly[f"boalf_{label}_dec_mwh"]
        )
        frames.append(hourly)

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="hour", how="outer")
    return out.sort_values("hour")


def add_model_hourly_gap(
    hourly_gap: pd.DataFrame,
    wholesale_dispatch_path: Path,
    balancing_dispatch_path: Path,
    wholesale_storage_path: Path | None,
    balancing_storage_path: Path | None,
) -> pd.DataFrame:
    wholesale = _read_timeseries_csv(wholesale_dispatch_path)
    balancing = _read_timeseries_csv(balancing_dispatch_path)
    common_cols = wholesale.columns.intersection(balancing.columns)
    diff = balancing[common_cols] - wholesale[common_cols]
    model_hourly = pd.DataFrame(
        {
            "hour": diff.index,
            "model_gen_inc_mwh": diff.clip(lower=0).sum(axis=1),
            "model_gen_dec_mwh": (-diff.clip(upper=0)).sum(axis=1),
        }
    )

    if wholesale_storage_path and balancing_storage_path and wholesale_storage_path.exists() and balancing_storage_path.exists():
        w_store = _read_timeseries_csv(wholesale_storage_path)
        b_store = _read_timeseries_csv(balancing_storage_path)
        common_store = w_store.columns.intersection(b_store.columns)
        store_diff = b_store[common_store] - w_store[common_store]
        model_hourly["model_storage_inc_mwh"] = store_diff.clip(lower=0).sum(axis=1).values
        model_hourly["model_storage_dec_mwh"] = (-store_diff.clip(upper=0)).sum(axis=1).values
    else:
        model_hourly["model_storage_inc_mwh"] = 0.0
        model_hourly["model_storage_dec_mwh"] = 0.0

    model_hourly["model_inc_mwh"] = (
        model_hourly["model_gen_inc_mwh"] + model_hourly["model_storage_inc_mwh"]
    )
    model_hourly["model_dec_mwh"] = (
        model_hourly["model_gen_dec_mwh"] + model_hourly["model_storage_dec_mwh"]
    )
    model_hourly["model_gross_mwh"] = model_hourly["model_inc_mwh"] + model_hourly["model_dec_mwh"]

    out = hourly_gap.merge(model_hourly, on="hour", how="outer").fillna(0.0)
    for scope in ["all", "unflagged", "flagged"]:
        out[f"gross_gap_vs_{scope}_mwh"] = out["model_gross_mwh"] - out[f"boalf_{scope}_gross_mwh"]
    return out.sort_values("hour")


def build_headroom_summary(
    network_path: Path,
    wholesale_dispatch_path: Path,
    wholesale_storage_path: Path | None,
) -> pd.DataFrame:
    try:
        import pypsa
    except ImportError:
        return pd.DataFrame([{"status": "pypsa unavailable; headroom skipped"}])
    if not network_path.exists():
        return pd.DataFrame([{"status": f"network file not found: {network_path}"}])

    network = pypsa.Network(str(network_path))
    wholesale = _read_timeseries_csv(wholesale_dispatch_path)
    cols = wholesale.columns.intersection(network.generators.index)
    gen = network.generators.loc[cols].copy()

    p_nom = gen["p_nom"].reindex(cols).astype(float)
    p_min_pu_static = gen.get("p_min_pu", pd.Series(0.0, index=cols)).reindex(cols).fillna(0.0).astype(float)
    p_max_pu_static = gen.get("p_max_pu", pd.Series(1.0, index=cols)).reindex(cols).fillna(1.0).astype(float)

    p_max_pu = pd.DataFrame(
        np.tile(p_max_pu_static.values, (len(wholesale), 1)),
        index=wholesale.index,
        columns=cols,
    )
    dyn_max = getattr(network.generators_t, "p_max_pu", pd.DataFrame())
    if not dyn_max.empty:
        dyn_max = dyn_max.reindex(index=wholesale.index, columns=cols)
        p_max_pu = dyn_max.fillna(p_max_pu)

    p_min_pu = pd.DataFrame(
        np.tile(p_min_pu_static.values, (len(wholesale), 1)),
        index=wholesale.index,
        columns=cols,
    )
    dyn_min = getattr(network.generators_t, "p_min_pu", pd.DataFrame())
    if not dyn_min.empty:
        dyn_min = dyn_min.reindex(index=wholesale.index, columns=cols)
        p_min_pu = dyn_min.fillna(p_min_pu)

    max_output = p_max_pu.mul(p_nom, axis=1)
    min_output = p_min_pu.mul(p_nom, axis=1)
    headroom = (max_output - wholesale[cols]).clip(lower=0.0)
    footroom = (wholesale[cols] - min_output).clip(lower=0.0)

    rows = []
    for carrier, carrier_gens in gen.groupby("carrier"):
        names = carrier_gens.index.intersection(cols)
        rows.append(
            {
                "component_type": "generator",
                "carrier": carrier,
                "capacity_mw": float(gen.loc[names, "p_nom"].sum()),
                "avg_headroom_mw": float(headroom[names].sum(axis=1).mean()),
                "avg_footroom_mw": float(footroom[names].sum(axis=1).mean()),
                "total_headroom_mwh": float(headroom[names].sum().sum()),
                "total_footroom_mwh": float(footroom[names].sum().sum()),
            }
        )

    if wholesale_storage_path and wholesale_storage_path.exists() and len(network.storage_units) > 0:
        w_store = _read_timeseries_csv(wholesale_storage_path)
        store_cols = w_store.columns.intersection(network.storage_units.index)
        storage = network.storage_units.loc[store_cols].copy()
        if len(store_cols):
            p_nom_store = storage["p_nom"].astype(float)
            max_dispatch = pd.DataFrame(
                np.tile(p_nom_store.values, (len(w_store), 1)),
                index=w_store.index,
                columns=store_cols,
            )
            max_store = max_dispatch.copy()
            store_headroom = (max_dispatch - w_store[store_cols]).clip(lower=0.0)
            store_footroom = (w_store[store_cols] + max_store).clip(lower=0.0)
            for carrier, carrier_units in storage.groupby("carrier"):
                names = carrier_units.index.intersection(store_cols)
                rows.append(
                    {
                        "component_type": "storage_unit",
                        "carrier": carrier,
                        "capacity_mw": float(storage.loc[names, "p_nom"].sum()),
                        "avg_headroom_mw": float(store_headroom[names].sum(axis=1).mean()),
                        "avg_footroom_mw": float(store_footroom[names].sum(axis=1).mean()),
                        "total_headroom_mwh": float(store_headroom[names].sum().sum()),
                        "total_footroom_mwh": float(store_footroom[names].sum().sum()),
                    }
                )

    return pd.DataFrame(rows).sort_values("capacity_mw", ascending=False)


def _metric_lookup(validation: pd.DataFrame, metric: str, column: str) -> float:
    row = validation.loc[validation["metric"].eq(metric)]
    if row.empty:
        return np.nan
    return pd.to_numeric(row.iloc[0][column], errors="coerce")


def write_summary(
    output_path: Path,
    scenario: str,
    carrier_gap: pd.DataFrame,
    mapped_carrier_gap: pd.DataFrame,
    mapping_summary: pd.DataFrame,
    carrier_conflicts: pd.DataFrame,
    bmu_gap: pd.DataFrame,
    disbsad: pd.DataFrame,
    validation_path: Path,
) -> None:
    validation = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame()

    total_model_inc = float(carrier_gap["model_inc_mwh"].sum())
    total_model_dec = float(carrier_gap["model_dec_mwh"].sum())
    total_boalf_all_inc = float(carrier_gap["boalf_all_inc_mwh"].sum())
    total_boalf_all_dec = float(carrier_gap["boalf_all_dec_mwh"].sum())
    total_boalf_uf_inc = float(carrier_gap["boalf_unflagged_inc_mwh"].sum())
    total_boalf_uf_dec = float(carrier_gap["boalf_unflagged_dec_mwh"].sum())
    total_boalf_flag_inc = float(carrier_gap["boalf_flagged_inc_mwh"].sum())
    total_boalf_flag_dec = float(carrier_gap["boalf_flagged_dec_mwh"].sum())

    disbsad_abs = np.nan
    disbsad_cost = np.nan
    if not disbsad.empty:
        if "volume" in disbsad.columns:
            disbsad_abs = float(disbsad["volume"].abs().sum())
        if "cost" in disbsad.columns:
            disbsad_cost = float(disbsad["cost"].sum())
    if np.isnan(disbsad_abs) and not validation.empty:
        disbsad_abs = _metric_lookup(validation, "DISBSAD absolute volume", "elexon_value")
    if np.isnan(disbsad_cost) and not validation.empty:
        disbsad_cost = _metric_lookup(validation, "DISBSAD total cost", "elexon_value")

    top_carriers = carrier_gap.head(10)[
        [
            "carrier",
            "model_gross_mwh",
            "boalf_all_gross_mwh",
            "boalf_unflagged_gross_mwh",
            "gross_gap_vs_all_mwh",
        ]
    ]
    top_missing_mapping = mapping_summary.head(10)
    top_mapped_carriers = mapped_carrier_gap.head(10)[
        [
            "carrier",
            "model_gross_mwh",
            "boalf_mapped_all_gross_mwh",
            "boalf_mapped_unflagged_gross_mwh",
            "gross_gap_vs_mapped_all_mwh",
        ]
    ]
    top_bmus = bmu_gap.head(15)[
        [
            "bmu_id",
            "carrier",
            "boalf_all_gross_mwh",
            "boalf_unflagged_gross_mwh",
            "mapped_to_model",
            "model_generator_has_redispatch",
            "generator_name",
            "station_name",
        ]
    ]
    top_conflicts = carrier_conflicts.head(15)

    lines = [
        f"# BM volume gap decomposition: {scenario}",
        "",
        "## Totals",
        "",
        f"- Model increase: {total_model_inc:,.0f} MWh",
        f"- Model decrease: {total_model_dec:,.0f} MWh",
        f"- BOALF all increase: {total_boalf_all_inc:,.0f} MWh",
        f"- BOALF all decrease: {total_boalf_all_dec:,.0f} MWh",
        f"- BOALF unflagged increase: {total_boalf_uf_inc:,.0f} MWh",
        f"- BOALF unflagged decrease: {total_boalf_uf_dec:,.0f} MWh",
        f"- BOALF flagged increase: {total_boalf_flag_inc:,.0f} MWh",
        f"- BOALF flagged decrease: {total_boalf_flag_dec:,.0f} MWh",
        f"- DISBSAD absolute volume: {disbsad_abs:,.0f} MWh",
        f"- DISBSAD cost: £{disbsad_cost:,.0f}",
        "",
        "## Interpretation",
        "",
        "- The model is a network redispatch LP, so it should not be expected to match all BOALF volume.",
        "- The largest useful comparator is BOALF unflagged, but even that includes real-world unit constraints and operational instructions that are not fully represented.",
        "- A large BOALF flagged and DISBSAD volume means a material part of the observed BM volume is outside the current model scope.",
        "- The BMU mapping summary shows whether the remaining gap is missing assets, mapped assets that do not move in the model, or genuine model redispatch behaviour.",
        "",
        "## Top Carrier Gaps",
        "",
        top_carriers.to_markdown(index=False),
        "",
        "## Top Carrier Gaps After BMU Crosswalk Carrier Mapping",
        "",
        top_mapped_carriers.to_markdown(index=False),
        "",
        "## Largest BOALF Carrier Classification Conflicts",
        "",
        top_conflicts.to_markdown(index=False) if not top_conflicts.empty else "No material carrier classification conflicts found.",
        "",
        "## BOALF Mapping Coverage By Carrier",
        "",
        top_missing_mapping.to_markdown(index=False),
        "",
        "## Largest BOALF BMUs",
        "",
        top_bmus.to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `carrier_gap.csv`: model vs BOALF by carrier and flag scope.",
        "- `carrier_gap_mapped.csv`: same comparison after applying the BMU crosswalk carrier where available.",
        "- `carrier_classification_conflicts.csv`: BMUs where BOALF-prefix carrier differs from crosswalk carrier.",
        "- `bmu_gap.csv`: BOALF BMU volumes with model mapping/redispatch coverage.",
        "- `mapping_summary.csv`: carrier-level attribution of BOALF volume to unmapped/mapped/inactive model assets.",
        "- `hourly_gap.csv`: hourly model gross volume vs BOALF flagged/unflagged/all.",
        "- `headroom_footroom.csv`: wholesale schedule headroom/footroom by model carrier.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    scenario = args.scenario
    market_dir = ROOT / "resources" / "market"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "resources" / "analysis" / f"{scenario}_bm_volume_gap"
    output_dir.mkdir(parents=True, exist_ok=True)

    redispatch_path = Path(args.redispatch) if args.redispatch else market_dir / f"{scenario}_redispatch_summary.csv"
    validation_path = Path(args.validation) if args.validation else market_dir / f"{scenario}_bm_validation.csv"
    price_path = Path(args.price_comparison) if args.price_comparison else market_dir / f"{scenario}_price_comparison.csv"
    wholesale_dispatch_path = Path(args.wholesale_dispatch) if args.wholesale_dispatch else market_dir / f"{scenario}_wholesale_dispatch.csv"
    balancing_dispatch_path = Path(args.balancing_dispatch) if args.balancing_dispatch else market_dir / f"{scenario}_balancing_dispatch.csv"
    wholesale_storage_path = Path(args.wholesale_storage) if args.wholesale_storage else market_dir / f"{scenario}_wholesale_storage.csv"
    balancing_storage_path = Path(args.balancing_storage) if args.balancing_storage else market_dir / f"{scenario}_balancing_storage.csv"
    bmu_mapping_path = Path(args.bmu_mapping) if args.bmu_mapping else ROOT / "resources" / "generators" / f"{scenario}_bmu_mapping.csv"
    network_path = Path(args.network) if args.network else market_dir / f"{scenario}_balancing.nc"

    start, end = _period_from_model(price_path)
    if args.boalf:
        boalf_path = Path(args.boalf)
    else:
        scenario_boalf = (
            ROOT
            / "resources"
            / "market"
            / "elexon"
            / "validation"
            / str(start.year)
            / scenario
            / "boalf_data.csv"
        )
        shared_boalf = (
            ROOT
            / "resources"
            / "market"
            / "elexon"
            / "validation"
            / str(start.year)
            / "boalf_data.csv"
        )
        boalf_path = scenario_boalf if scenario_boalf.exists() else shared_boalf

    if args.disbsad:
        disbsad_path = Path(args.disbsad)
    else:
        scenario_disbsad = (
            ROOT
            / "resources"
            / "market"
            / "elexon"
            / "validation"
            / str(start.year)
            / scenario
            / "disbsad_data.csv"
        )
        shared_disbsad = (
            ROOT
            / "resources"
            / "market"
            / "elexon"
            / "validation"
            / str(start.year)
            / "disbsad_data.csv"
        )
        disbsad_path = scenario_disbsad if scenario_disbsad.exists() else shared_disbsad

    redispatch = pd.read_csv(redispatch_path)
    bmu_mapping = pd.read_csv(bmu_mapping_path)
    boalf = _load_boalf(boalf_path, start, end)
    boalf_acc = _prepare_boalf_acceptance_records(boalf)
    disbsad = _load_disbsad(disbsad_path, start, end)

    carrier_gap = build_carrier_gap(redispatch, boalf_acc)
    bmu_gap = build_bmu_gap(redispatch, boalf_acc, bmu_mapping)
    mapping_summary = build_mapping_summary(bmu_gap)
    mapped_carrier_gap = build_mapped_carrier_gap(redispatch, bmu_gap)
    carrier_conflicts = build_carrier_conflicts(bmu_gap)
    hourly_gap = build_hourly_gap(redispatch, boalf_acc)
    hourly_gap = add_model_hourly_gap(
        hourly_gap,
        wholesale_dispatch_path,
        balancing_dispatch_path,
        wholesale_storage_path,
        balancing_storage_path,
    )
    headroom = build_headroom_summary(network_path, wholesale_dispatch_path, wholesale_storage_path)

    carrier_gap.to_csv(output_dir / "carrier_gap.csv", index=False)
    mapped_carrier_gap.to_csv(output_dir / "carrier_gap_mapped.csv", index=False)
    carrier_conflicts.to_csv(output_dir / "carrier_classification_conflicts.csv", index=False)
    bmu_gap.to_csv(output_dir / "bmu_gap.csv", index=False)
    mapping_summary.to_csv(output_dir / "mapping_summary.csv", index=False)
    hourly_gap.to_csv(output_dir / "hourly_gap.csv", index=False)
    headroom.to_csv(output_dir / "headroom_footroom.csv", index=False)
    boalf_acc.to_csv(output_dir / "boalf_acceptance_records.csv", index=False)

    write_summary(
        output_dir / "summary.md",
        scenario,
        carrier_gap,
        mapped_carrier_gap,
        mapping_summary,
        carrier_conflicts,
        bmu_gap,
        disbsad,
        validation_path,
    )

    print(f"BM volume gap decomposition written to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="Validation_Jun2020")
    parser.add_argument("--output-dir")
    parser.add_argument("--redispatch")
    parser.add_argument("--validation")
    parser.add_argument("--price-comparison")
    parser.add_argument("--wholesale-dispatch")
    parser.add_argument("--balancing-dispatch")
    parser.add_argument("--wholesale-storage")
    parser.add_argument("--balancing-storage")
    parser.add_argument("--bmu-mapping")
    parser.add_argument("--network")
    parser.add_argument("--boalf")
    parser.add_argument("--disbsad")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
