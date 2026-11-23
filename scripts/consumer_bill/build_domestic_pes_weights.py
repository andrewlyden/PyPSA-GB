"""Aggregate official domestic electricity consumption to PES/DNO areas.

DESNZ publishes annual domestic consumption by local authority, while NESO
publishes the GSP-to-PES lookup.  A local authority is assigned to the nearest
NESO GSP using its official centroid.  This is an explicit geographic
allocation approximation: it avoids using total PyPSA load as a household
weight, but should be replaced by sub-local-authority sales data when available.
"""


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.consumer_bill.build_bus_dno_mapping import NESO_GSP_DNO_LOOKUP_URL, _gsp_areas, _haversine_km


DESNZ_DOMESTIC_CONSUMPTION_URL = (
    "https://assets.publishing.service.gov.uk/media/694295bd9273c48f554cf4ef/"
    "elec_LA_stacked_2005-2024.csv"
)
ONS_LAD_CENTROIDS_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_DEC_2025_Boundaries_UK_BFE/FeatureServer/0/query?where=1%3D1%20AND%20LAD25CD%20IS%20NOT%20NULL&"
    "outFields=LAD25CD%2CLAD25NM&returnGeometry=false&returnCentroid=true&outSR=4326&f=json"
)

# Ofgem's fourteen price-cap regions correspond to the PES/DNO areas.
PES_TO_OFGEM_REGION = {
    10: "Eastern", 11: "East Midlands", 12: "London", 13: "N Wales and Mersey",
    14: "Midlands", 15: "Northern", 16: "North West", 17: "Southern",
    18: "South East", 19: "South Wales", 20: "Southern Western", 21: "Yorkshire",
    22: "Southern Scotland", 23: "Northern Scotland",
}


def _centroids(path: str | Path) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for item in raw.get("features", []):
        attributes = item.get("attributes", {})
        code = attributes.get("LAD25CD") or attributes.get("LAD20CD")
        name = attributes.get("LAD25NM") or attributes.get("LAD20NM")
        centroid = item.get("centroid")
        if code and name and centroid:
            rows.append({"LA_Code": code, "LA": name, "lon": centroid["x"], "lat": centroid["y"]})
    if not rows:
        raise ValueError("ONS local-authority centroid input contains no usable features")
    return pd.DataFrame(rows).drop_duplicates("LA_Code")


def build_domestic_pes_weights(domestic: pd.DataFrame, centroids: pd.DataFrame,
                               gsp_lookup: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    required = {"Year", "LA_Code", "Domestic_consumption_GWh", "Domestic_meters_thouasands"}
    if missing := required.difference(domestic.columns):
        raise ValueError(f"DESNZ consumption input is missing columns: {sorted(missing)}")
    source = domestic.loc[pd.to_numeric(domestic["Year"], errors="coerce").eq(year)].copy()
    source["Domestic_consumption_GWh"] = pd.to_numeric(source["Domestic_consumption_GWh"], errors="coerce")
    source["Domestic_meters_thouasands"] = pd.to_numeric(source["Domestic_meters_thouasands"], errors="coerce")
    source = source.dropna(subset=["LA_Code", "Domestic_consumption_GWh"])
    if source.empty:
        raise ValueError(f"No domestic local-authority consumption found for {year}")
    joined = source.merge(centroids[["LA_Code", "lat", "lon"]], on="LA_Code", how="left", validate="one_to_one")
    unmatched = joined.loc[joined[["lat", "lon"]].isna().any(axis=1)]
    assigned = joined.dropna(subset=["lat", "lon"]).copy()
    gsp = _gsp_areas(gsp_lookup)
    assignments = []
    for row in assigned.itertuples(index=False):
        distances = _haversine_km(row.lat, row.lon, gsp["gsp_lat"].to_numpy(), gsp["gsp_lon"].to_numpy())
        nearest = gsp.iloc[int(np.argmin(distances))]
        assignments.append((nearest.gsp_name, int(nearest.pes_id), nearest.pes_name, float(np.min(distances))))
    assigned[["nearest_gsp_name", "pes_id", "pes_name", "la_centroid_to_gsp_km"]] = assignments
    weights = assigned.groupby(["pes_id", "pes_name"], as_index=False).agg(
        domestic_consumption_gwh=("Domestic_consumption_GWh", "sum"),
        domestic_meters_thousands=("Domestic_meters_thouasands", "sum"),
        local_authority_count=("LA_Code", "nunique"),
    )
    weights["domestic_consumption_weight"] = weights["domestic_consumption_gwh"] / weights["domestic_consumption_gwh"].sum()
    weights["domestic_meter_weight"] = weights["domestic_meters_thousands"] / weights["domestic_meters_thousands"].sum()
    weights["ofgem_region"] = weights["pes_id"].map(PES_TO_OFGEM_REGION)
    if weights["ofgem_region"].isna().any():
        raise ValueError("No Ofgem regional label for one or more PES areas")
    weights["geographic_allocation"] = "local-authority centroid to nearest NESO GSP, then PES/DNO area"
    weights = weights.sort_values("pes_id").reset_index(drop=True)
    summary = {
        "purpose": f"{year} domestic electricity consumption weights for PES/DNO tariff allocation",
        "desnz_source": DESNZ_DOMESTIC_CONSUMPTION_URL,
        "ons_centroid_source": ONS_LAD_CENTROIDS_URL,
        "geography_reference_year": 2025,
        "neso_gsp_dno_source": NESO_GSP_DNO_LOOKUP_URL,
        "domestic_consumption_gwh": float(weights["domestic_consumption_gwh"].sum()),
        "domestic_meters_thousands": float(weights["domestic_meters_thousands"].sum()),
        "unmapped_local_authority_count": int(len(unmatched)),
        "unmapped_domestic_consumption_gwh": float(unmatched["Domestic_consumption_GWh"].sum()),
        "maximum_la_centroid_to_gsp_km": float(assigned["la_centroid_to_gsp_km"].max()),
        "treatment": "Domestic sales/account geography input, not PyPSA demand. DESNZ 2020 values are published on current local-authority codes and use 2025 centroids. Local authorities spanning a PES boundary are assigned by centroid; retain this limitation in tariff results.",
    }
    return weights, summary


def write_weights(domestic_path, centroids_path, gsp_lookup_path, year, output_csv, output_summary) -> None:
    weights, summary = build_domestic_pes_weights(
        # DESNZ's published CSV uses Windows-1252 punctuation in authority names.
        pd.read_csv(domestic_path, encoding="cp1252"),
        _centroids(centroids_path),
        pd.read_csv(gsp_lookup_path),
        year,
    )
    for output in (output_csv, output_summary):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if "snakemake" in globals():
    write_weights(
        snakemake.input.domestic_consumption, snakemake.input.ons_centroids,
        snakemake.input.gsp_dno_lookup, snakemake.params.year,
        snakemake.output.weights_csv, snakemake.output.summary_json,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domestic-consumption", required=True)
    parser.add_argument("--ons-centroids", required=True)
    parser.add_argument("--gsp-dno-lookup", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_weights(args.domestic_consumption, args.ons_centroids, args.gsp_dno_lookup, args.year,
                  args.output_csv, args.output_summary)
