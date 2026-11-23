"""Extract the named REPD pipeline for future scenarios.

``prepare_renewable_site_data.py`` keeps only ``Operational`` REPD projects,
which is right for a historical replay and useless for 2030: the projects that
will make up most of the 2030 fleet are today under construction, consented, or
in planning. This script emits a companion site file per weather-variable
technology holding exactly those projects, with the columns the brownfield
CP30 fleet needs to place, label and date them:

* location (WGS84 from the OSGB36 planning coordinates),
* support labels straight from REPD (``cfd_round``, ``cfd_capacity_mw``,
  ``ro_banding``), so contract vintages are read rather than scheduled,
* status and the planning milestones REPD records for it, so a declared
  inclusion rule can be applied per scenario and the maturity ordering used to
  resolve regional overshoot is auditable.

The historical site files are untouched; this writes new files alongside them.
Nothing here decides which projects a scenario includes. That is the scenario's
``spatial_proxy.brownfield`` configuration, applied in
``scripts/CP30/brownfield_fleet.py``.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from scripts.renewables.repd_support_labels import (
    apply_cfd_round_corrections, load_cfd_round_corrections, normalise_support_labels,
)


logger = logging.getLogger("prepare_future_pipeline_sites")

# REPD technology labels for the three carriers the ES5 regional split covers.
TECHNOLOGIES = {
    "wind_offshore": ["Wind Offshore"],
    "wind_onshore": ["Wind Onshore"],
    "solar_pv": ["Solar Photovoltaics"],
}

# Every non-operational status REPD uses that can plausibly deliver capacity.
# Which of these a scenario includes is configured per scenario; the file keeps
# them all so the choice is made where it can be audited.
PIPELINE_STATUSES = (
    "Under Construction",
    "Awaiting Construction",
    "Application Submitted",
)

REQUIRED_COLUMNS = [
    "Ref ID", "Old Ref ID", "Site Name", "Technology Type",
    "Installed Capacity (MWelec)", "Country", "Development Status",
    "Development Status (short)", "X-coordinate", "Y-coordinate",
    "CfD Allocation Round", "RO Banding (ROC/MWh)", "CfD Capacity (MW)",
    "Offshore Wind Round", "Planning Permission Granted", "Under Construction",
    "Planning Application Submitted", "Operational",
]

OUTPUT_COLUMNS = [
    "site_name", "source_site_name", "repd_ref_id", "repd_old_ref_id", "technology",
    "capacity_mw", "lat", "lon", "cfd_round", "cfd_capacity_mw", "ro_banding",
    "development_status", "planning_application_date", "planning_granted_date",
    "under_construction_date", "expected_operational_date", "offshore_wind_round",
    "country",
]


def _to_wgs84(frame: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)
    x = pd.to_numeric(frame["X-coordinate"], errors="coerce")
    y = pd.to_numeric(frame["Y-coordinate"], errors="coerce")
    valid = x.notna() & y.notna()
    lon = pd.Series(np.nan, index=frame.index)
    lat = pd.Series(np.nan, index=frame.index)
    if valid.any():
        lon_values, lat_values = transformer.transform(x[valid].to_numpy(), y[valid].to_numpy())
        lon[valid] = lon_values
        lat[valid] = lat_values
    return frame.assign(lat=lat, lon=lon)


def _clean_name(values: pd.Series) -> pd.Series:
    # REPD carries stray non-breaking spaces and mojibake in a few site names
    # ("Triton Knoll\xa0"); profile columns are matched on normalised names so
    # the raw text is kept but trailing junk is trimmed.
    return (values.astype(str).str.replace("\xa0", " ", regex=False)
            .str.replace("�", "", regex=False).str.strip())


def load_repd_pipeline(repd_path, offshore_pipeline_path=None,
                       round_corrections_path=None) -> pd.DataFrame:
    """Return every non-operational REPD project for the three carriers."""
    frame = pd.read_csv(repd_path, encoding="latin-1", low_memory=False,
                        usecols=lambda c: c in REQUIRED_COLUMNS)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"REPD extract lacks columns: {missing}")
    frame = frame[frame["Country"].ne("Northern Ireland")]
    frame = frame[frame["Development Status (short)"].isin(PIPELINE_STATUSES)].copy()

    technology = pd.Series(pd.NA, index=frame.index, dtype="object")
    for carrier, labels in TECHNOLOGIES.items():
        technology[frame["Technology Type"].isin(labels)] = carrier
    frame = frame[technology.notna()].copy()
    frame["technology"] = technology[frame.index]

    frame["capacity_mw"] = pd.to_numeric(frame["Installed Capacity (MWelec)"], errors="coerce")
    frame = frame[frame["capacity_mw"].gt(0)]
    frame = _to_wgs84(frame)
    dropped = int(frame[["lat", "lon"]].isna().any(axis=1).sum())
    if dropped:
        logger.warning("Dropping %d pipeline projects with no usable coordinates", dropped)
    frame = frame.dropna(subset=["lat", "lon"])

    for column, source in (
        ("planning_application_date", "Planning Application Submitted"),
        ("planning_granted_date", "Planning Permission Granted"),
        ("under_construction_date", "Under Construction"),
    ):
        frame[column] = pd.to_datetime(frame[source], format="%d/%m/%Y", errors="coerce").dt.date

    frame["expected_operational_date"] = pd.NaT
    if offshore_pipeline_path and Path(offshore_pipeline_path).is_file():
        # The maintained offshore pipeline file carries developer-stated
        # expected operational dates for the largest projects; REPD itself has
        # none. Join on site name and keep REPD as the authority for everything else.
        pipeline = pd.read_csv(offshore_pipeline_path, encoding="latin-1")
        if {"Site Name", "Expected Operational"}.issubset(pipeline.columns):
            expected = pipeline.assign(
                name=_clean_name(pipeline["Site Name"]).str.lower(),
                date=pd.to_datetime(pipeline["Expected Operational"], format="%d/%m/%Y",
                                    errors="coerce"),
            ).dropna(subset=["date"]).drop_duplicates("name").set_index("name")["date"]
            key = _clean_name(frame["Site Name"]).str.lower()
            frame["expected_operational_date"] = key.map(expected).dt.date

    out = pd.DataFrame({
        "site_name": _clean_name(frame["Site Name"]),
        "source_site_name": _clean_name(frame["Site Name"]),
        "repd_ref_id": frame["Ref ID"].astype(str).str.strip(),
        "repd_old_ref_id": frame["Old Ref ID"],
        "technology": frame["technology"],
        "capacity_mw": frame["capacity_mw"],
        "lat": frame["lat"],
        "lon": frame["lon"],
        "cfd_round": frame["CfD Allocation Round"],
        "cfd_capacity_mw": pd.to_numeric(frame["CfD Capacity (MW)"], errors="coerce"),
        "ro_banding": pd.to_numeric(frame["RO Banding (ROC/MWh)"], errors="coerce"),
        "development_status": frame["Development Status (short)"],
        "planning_application_date": frame["planning_application_date"],
        "planning_granted_date": frame["planning_granted_date"],
        "under_construction_date": frame["under_construction_date"],
        "expected_operational_date": frame["expected_operational_date"],
        "offshore_wind_round": frame["Offshore Wind Round"],
        "country": frame["Country"],
    })
    # Two REPD rows can share a site name (phases, re-applications). The Ref ID
    # is the identity; make the site name unique so profile columns and
    # generator names cannot collide.
    duplicated = out["site_name"].duplicated(keep=False)
    out.loc[duplicated, "site_name"] = (
        out.loc[duplicated, "site_name"] + " [REPD " + out.loc[duplicated, "repd_ref_id"] + "]"
    )
    out = normalise_support_labels(out, logger=logger)
    if round_corrections_path:
        out = apply_cfd_round_corrections(
            out, load_cfd_round_corrections(round_corrections_path), logger=logger
        )
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def write_pipeline_sites(repd_path, offshore_pipeline_path, outputs: dict, summary_path=None,
                         round_corrections_path=None):
    pipeline = load_repd_pipeline(repd_path, offshore_pipeline_path, round_corrections_path)
    summary = []
    for carrier, path in outputs.items():
        subset = pipeline[pipeline["technology"].eq(carrier)]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(path, index=False)
        by_status = subset.groupby("development_status")["capacity_mw"].sum()
        logger.info("%s: %d pipeline projects, %.0f MW (%s)", carrier, len(subset),
                    subset["capacity_mw"].sum(),
                    ", ".join(f"{k} {v:,.0f}" for k, v in by_status.items()))
        for status, mw in by_status.items():
            summary.append({"technology": carrier, "development_status": status,
                            "projects": int(subset["development_status"].eq(status).sum()),
                            "capacity_mw": float(mw)})
    if summary_path:
        pd.DataFrame(summary).to_csv(summary_path, index=False)
    return pipeline


if "snakemake" in globals():
    logging.basicConfig(level=logging.INFO)
    write_pipeline_sites(
        snakemake.input.repd, snakemake.input.offshore_pipeline,
        {"wind_offshore": snakemake.output.wind_offshore,
         "wind_onshore": snakemake.output.wind_onshore,
         "solar_pv": snakemake.output.solar_pv},
        snakemake.output.summary,
        round_corrections_path=snakemake.input.cfd_round_corrections,
    )
elif __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repd", default="data/renewables/repd-q2-jul-2025.csv")
    parser.add_argument("--offshore-pipeline",
                        default="data/renewables/future_offshore_sites/offshore_pipeline.csv")
    parser.add_argument("--output-dir", default="resources/renewable")
    parser.add_argument("--round-corrections",
                        default="data/market/lccc/cfd_round_corrections.csv")
    args = parser.parse_args()
    out = Path(args.output_dir)
    write_pipeline_sites(
        args.repd, args.offshore_pipeline,
        {carrier: out / f"{carrier}_pipeline_sites.csv" for carrier in TECHNOLOGIES},
        out / "pipeline_sites_summary.csv",
        round_corrections_path=args.round_corrections,
    )
