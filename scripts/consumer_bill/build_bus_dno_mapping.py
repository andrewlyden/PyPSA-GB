"""Map PyPSA demand buses to DNO/PES areas using NESO's GSP lookup.

The mapping is a geographic crosswalk only.  It does not claim that system
load at a bus is domestic demand; that requires separately supplied domestic
sales/account weights before it can be used in a consumer-bill tariff bridge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


NESO_GSP_DNO_LOOKUP_URL = (
    "https://api.neso.energy/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/"
    "resource/bbe2cc72-a6c6-46e6-8f4e-48b879467368/download/"
    "gsp_gnode_directconnect_region_lookup.csv"
)


def _haversine_km(lat, lon, reference_lat, reference_lon) -> np.ndarray:
    """Return great-circle distances from one point to reference points."""
    lat1, lon1 = np.radians([lat, lon])
    lat2, lon2 = np.radians([reference_lat, reference_lon])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _gsp_areas(lookup: pd.DataFrame) -> pd.DataFrame:
    required = {"gsp_name", "gsp_lat", "gsp_lon", "pes_id", "pes_name"}
    if missing := required.difference(lookup.columns):
        raise ValueError(f"NESO GSP lookup is missing columns: {sorted(missing)}")
    gsp = lookup[list(required)].copy()
    for column in ("gsp_lat", "gsp_lon", "pes_id"):
        gsp[column] = pd.to_numeric(gsp[column], errors="coerce")
    gsp = gsp.dropna(subset=["gsp_name", "gsp_lat", "gsp_lon", "pes_id", "pes_name"])
    conflicts = gsp.groupby("gsp_name")["pes_id"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"NESO GSP lookup assigns multiple PES areas to: {conflicts[conflicts > 1].index.tolist()}")
    return gsp.drop_duplicates("gsp_name").reset_index(drop=True)


def build_bus_dno_mapping(load_buses: pd.DataFrame, gsp_lookup: pd.DataFrame) -> pd.DataFrame:
    """Assign each load bus to its nearest NESO GSP and associated PES area."""
    required = {"bus", "lat", "lon"}
    if missing := required.difference(load_buses.columns):
        raise ValueError(f"Load-bus input is missing columns: {sorted(missing)}")
    buses = load_buses.copy()
    buses["lat"] = pd.to_numeric(buses["lat"], errors="coerce")
    buses["lon"] = pd.to_numeric(buses["lon"], errors="coerce")
    if buses["bus"].duplicated().any():
        raise ValueError("Load-bus input must have one row per bus")
    if buses[["lat", "lon"]].isna().any().any():
        raise ValueError("Every load bus must have valid latitude and longitude")
    gsp = _gsp_areas(gsp_lookup)
    assignments = []
    for row in buses.itertuples(index=False):
        distances = _haversine_km(row.lat, row.lon, gsp["gsp_lat"].to_numpy(), gsp["gsp_lon"].to_numpy())
        nearest = gsp.iloc[int(np.argmin(distances))]
        assignments.append({
            "bus": row.bus,
            "nearest_gsp_name": nearest.gsp_name,
            "pes_id": int(nearest.pes_id),
            "pes_name": nearest.pes_name,
            "distance_to_nearest_gsp_km": float(np.min(distances)),
        })
    return pd.DataFrame(assignments).sort_values("bus").reset_index(drop=True)


def load_bus_coordinates(network_path: str | Path) -> pd.DataFrame:
    network = pypsa.Network(network_path)
    buses = pd.Index(network.loads.bus.dropna().unique(), name="bus")
    coordinates = network.buses.reindex(buses)[["lat", "lon"]].reset_index(names="bus")
    return coordinates


def write_mapping(network_path, lookup_path, output_csv, output_summary) -> None:
    coordinates = load_bus_coordinates(network_path)
    unmapped_non_gb = coordinates.loc[coordinates[["lat", "lon"]].isna().any(axis=1), "bus"].tolist()
    mapping = build_bus_dno_mapping(
        coordinates.dropna(subset=["lat", "lon"]), pd.read_csv(lookup_path)
    )
    summary = {
        "purpose": "Geographic PyPSA demand-bus to NESO GSP/PES-DNO mapping",
        "source_lookup": NESO_GSP_DNO_LOOKUP_URL,
        "load_bus_count": len(mapping),
        "excluded_non_geographic_load_buses": unmapped_non_gb,
        "pes_area_count": int(mapping["pes_id"].nunique()),
        "maximum_distance_to_assigned_gsp_km": float(mapping["distance_to_nearest_gsp_km"].max()),
        "treatment": "Geographic crosswalk only. Pair it with domestic sales/accounts and DNO tariff-class inputs before calculating consumer network charges.",
    }
    for output in (output_csv, output_summary):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--gsp-dno-lookup", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    write_mapping(args.network, args.gsp_dno_lookup, args.output_csv, args.output_summary)
