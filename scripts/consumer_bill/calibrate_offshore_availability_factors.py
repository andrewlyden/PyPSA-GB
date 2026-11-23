"""Derive resource-region availability factors from LCCC outturn plus BM curtailment.

A renewable performance factor should describe *availability* -- reanalysis
bias, turbine modelling, wakes and outages -- and nothing else. LCCC reports
**settled** generation, which is net of curtailment, so calibrating directly
against it folds a network outcome into a resource assumption and the model
then curtails a second time on top.

This script reconstructs potential output as settled generation plus
bid-accepted curtailment from Elexon BOAV, and divides by the raw
(pre-factor) site-profile energy:

    factor = (LCCC settled MWh + BM bid-accepted MWh) / raw profile MWh

It is a calibration tool, not a forecast. The result is only as good as the
contract-to-BMU mapping it is given, and settled generation remains a lower
bound on metered output wherever a contract stops paying while still running
(negative-price suspension, for example), so a year with heavy suspension
cannot be fully cleaned by curtailment add-back alone.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_CONTRACT_COLUMNS = {
    "CfD_ID", "resource_region", "actual_generation_mwh", "raw_profile_mwh",
}


def load_bmu_map(mapping_paths, carrier: str | None) -> dict[str, list[str]]:
    """Union the CfD-to-BMU mapping across every supplied mapping file."""
    bmus: dict[str, set[str]] = {}
    for path in mapping_paths:
        frame = pd.read_csv(path)
        if carrier and "model_carrier" in frame.columns:
            frame = frame.loc[frame["model_carrier"].eq(carrier)]
        if "lccc_bmu_ids" not in frame.columns:
            raise ValueError(f"{path} has no lccc_bmu_ids column")
        for row in frame.to_dict("records"):
            raw = row.get("lccc_bmu_ids")
            if pd.isna(raw):
                continue
            ids = {part.strip() for part in str(raw).split(";") if part.strip()}
            bmus.setdefault(str(row["CfD_ID"]), set()).update(ids)
    return {key: sorted(value) for key, value in bmus.items()}


BOAV_IDENTITY = ["bmu_id", "acceptance_id", "settlement_date", "settlement_period", "side"]


def load_bid_volumes(boav_paths) -> pd.Series:
    """Return total bid-accepted (curtailed) MWh per BMU, as a positive number.

    Caches for adjacent periods can overlap, so records are de-duplicated on
    their settlement identity. Two genuinely distinct acceptances may carry the
    same volume in the same period, so the acceptance id has to be part of that
    key -- dropping it silently merges them and understates curtailment.
    """
    columns = BOAV_IDENTITY + ["accepted_volume_mwh"]
    frames = []
    for path in boav_paths:
        frame = pd.read_csv(path)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing BOAV columns: {missing}")
        frames.append(frame[columns])
    if not frames:
        return pd.Series(dtype=float)
    boav = pd.concat(frames, ignore_index=True).drop_duplicates(subset=BOAV_IDENTITY)
    bids = boav.loc[boav["side"].eq("bid")]
    volumes = -bids.groupby("bmu_id")["accepted_volume_mwh"].sum()
    return volumes.clip(lower=0.0)


def calibrate(contracts: pd.DataFrame, bmu_map: dict, bid_volumes: pd.Series) -> pd.DataFrame:
    """Attribute curtailment to contracts, then aggregate factors by region."""
    missing = REQUIRED_CONTRACT_COLUMNS - set(contracts.columns)
    if missing:
        raise ValueError(f"Contract calibration input is missing columns: {sorted(missing)}")
    contracts = contracts.dropna(subset=["raw_profile_mwh"]).copy()
    contracts = contracts.loc[contracts["raw_profile_mwh"] > 0]
    if contracts.empty:
        raise ValueError("No contracts with positive raw profile energy to calibrate against")

    # A BMU shared by several contract phases is counted once, split evenly,
    # so overlapping phase mappings cannot inflate the curtailment total.
    owners: dict[str, list[str]] = {}
    for cfd_id in contracts["CfD_ID"].astype(str):
        for bmu in bmu_map.get(cfd_id, []):
            owners.setdefault(bmu, []).append(cfd_id)

    curtailed = {cfd_id: 0.0 for cfd_id in contracts["CfD_ID"].astype(str)}
    for bmu, holders in owners.items():
        volume = float(bid_volumes.get(bmu, 0.0))
        if not volume:
            continue
        for cfd_id in holders:
            curtailed[cfd_id] += volume / len(holders)

    contracts["curtailed_mwh"] = contracts["CfD_ID"].astype(str).map(curtailed).fillna(0.0)
    contracts["bmu_mapped"] = contracts["CfD_ID"].astype(str).isin(bmu_map)
    contracts["potential_generation_mwh"] = (
        contracts["actual_generation_mwh"] + contracts["curtailed_mwh"]
    )

    grouped = contracts.groupby("resource_region", as_index=False).agg(
        contracts=("CfD_ID", "nunique"),
        contracts_without_bmu_mapping=("bmu_mapped", lambda s: int((~s).sum())),
        settled_mwh=("actual_generation_mwh", "sum"),
        curtailed_mwh=("curtailed_mwh", "sum"),
        raw_profile_mwh=("raw_profile_mwh", "sum"),
    )
    grouped["potential_mwh"] = grouped["settled_mwh"] + grouped["curtailed_mwh"]
    grouped["factor_from_settled_output"] = grouped["settled_mwh"] / grouped["raw_profile_mwh"]
    grouped["factor_from_potential_output"] = grouped["potential_mwh"] / grouped["raw_profile_mwh"]
    grouped["curtailment_share_of_settled"] = (
        grouped["curtailed_mwh"] / grouped["settled_mwh"].where(grouped["settled_mwh"] > 0)
    )
    return grouped.sort_values("resource_region").reset_index(drop=True)


def write_calibration(contracts_path, mapping_paths, boav_paths, carrier, output_csv, output_json):
    contracts = pd.read_csv(contracts_path)
    if carrier and "carrier" in contracts.columns:
        contracts = contracts.loc[contracts["carrier"].eq(carrier)]
    bmu_map = load_bmu_map(mapping_paths, carrier)
    bid_volumes = load_bid_volumes(boav_paths)
    regions = calibrate(contracts, bmu_map, bid_volumes)

    unmapped = int(regions["contracts_without_bmu_mapping"].sum())
    summary = {
        "carrier": carrier,
        "contracts_calibrated": int(regions["contracts"].sum()),
        "contracts_without_bmu_mapping": unmapped,
        "settled_mwh": float(regions["settled_mwh"].sum()),
        "curtailed_mwh": float(regions["curtailed_mwh"].sum()),
        "raw_profile_mwh": float(regions["raw_profile_mwh"].sum()),
        "factor_from_settled_output": float(
            regions["settled_mwh"].sum() / regions["raw_profile_mwh"].sum()
        ),
        "factor_from_potential_output": float(
            regions["potential_mwh"].sum() / regions["raw_profile_mwh"].sum()
        ),
        "treatment": (
            "Availability calibration only. Curtailment is added back so the "
            "factor does not embed a network outcome; the scenario model "
            "applies its own curtailment. Settled output remains a lower bound "
            "on metered output where a contract suspends payment while running."
        ),
        "sources": {
            "contracts": str(contracts_path),
            "mappings": [str(path) for path in mapping_paths],
            "boav": [str(path) for path in boav_paths],
        },
    }
    if unmapped:
        summary["warning"] = (
            f"{unmapped} contracts had no BMU mapping; their curtailment is "
            "counted as zero and their factor is therefore a lower bound."
        )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    regions.to_csv(output_csv, index=False)
    Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return regions, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", required=True, help="*_lccc_profile_calibration_contracts.csv")
    parser.add_argument("--mapping", nargs="+", required=True, help="*_lccc_cfd_contract_mapping.csv")
    parser.add_argument("--boav", nargs="+", required=True, help="Elexon boav_data.csv files")
    parser.add_argument("--carrier", default="wind_offshore")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    regions, summary = write_calibration(
        args.contracts, args.mapping, args.boav, args.carrier,
        args.output_csv, args.output_summary,
    )
    print(regions.to_string(index=False))
    print(json.dumps(summary, indent=2))
