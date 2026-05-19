"""
Build a local BMU crosswalk from the OSUKED Power Station Dictionary.

The raw dictionary keeps multiple BMU IDs in comma-separated fields.  This
script expands those fields into one row per BMU ID so build_bmu_mapping.py can
use the dictionary as an exact BMU lookup before falling back to prefix
heuristics.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import pandas as pd


DEFAULT_IDS_URL = (
    "https://raw.githubusercontent.com/OSUKED/Power-Station-Dictionary/"
    "shiro/data/dictionary/ids.csv"
)
DEFAULT_ATTRS_URL = (
    "https://osuked.github.io/Power-Station-Dictionary/"
    "object_attrs/dictionary_attributes.csv"
)
DEFAULT_IDS_PATH = Path("data/generators/power_station_dictionary_ids.csv")
DEFAULT_ATTRS_PATH = Path("data/generators/power_station_dictionary_attributes.csv")
DEFAULT_OUTPUT_PATH = Path("data/generators/power_station_dictionary_bmu_crosswalk.csv")


def _split_ids(value: object) -> list[str]:
    """Split a comma-separated dictionary ID field into clean IDs."""
    if value is None or pd.isna(value):
        return []
    items = []
    for item in str(value).split(","):
        item = re.sub(r"\s+", " ", item).strip()
        if item and item.lower() != "nan":
            items.append(item)
    return items


def _first_attribute(
    attrs: pd.DataFrame,
    dictionary_id: int,
    names: tuple[str, ...],
    id_type: str | None = "dictionary_id",
) -> object:
    """Return the first matching attribute value for a dictionary object."""
    mask = (attrs["dictionary_id"] == dictionary_id) & attrs["attribute"].isin(names)
    if id_type is not None:
        mask &= attrs["id_type"].eq(id_type)
    values = attrs.loc[mask, "value"].dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def _attribute_by_id(
    attrs: pd.DataFrame,
    bmu_id: str,
    names: tuple[str, ...],
    id_type: str,
) -> object:
    """Return an attribute attached directly to a specific BMU ID."""
    mask = (
        attrs["id"].astype(str).eq(bmu_id)
        & attrs["id_type"].eq(id_type)
        & attrs["attribute"].isin(names)
    )
    values = attrs.loc[mask, "value"].dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def _to_numeric(value: object) -> float | pd.NA:
    """Parse capacity-like values such as '1,200 MW'."""
    if value is None or pd.isna(value):
        return pd.NA
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    if not match:
        return pd.NA
    return float(match.group(0).replace(",", ""))


def _first_present(*values: object) -> object:
    """Return the first non-null value."""
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return pd.NA


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def build_crosswalk(ids_path: Path, attrs_path: Path, output_path: Path) -> pd.DataFrame:
    ids = pd.read_csv(ids_path)
    attrs = pd.read_csv(attrs_path, dtype={"id": "string", "value": "string"})

    rows: list[dict[str, object]] = []
    for _, source in ids.iterrows():
        dictionary_id = int(source["dictionary_id"])
        station_name = source["name"]
        if pd.isna(station_name) or not str(station_name).strip():
            continue
        settlement_ids = _split_ids(source.get("sett_bmu_id"))
        national_grid_ids = _split_ids(source.get("ngc_bmu_id"))

        primary_fuel = _first_present(
            _first_attribute(
                attrs,
                dictionary_id,
                ("Primary Fuel Type", "Fuel Type"),
            ),
            _first_attribute(
                attrs,
                dictionary_id,
                ("Primary Fuel Type", "Fuel Type"),
                id_type=None,
            ),
        )
        plant_type = _first_attribute(
            attrs,
            dictionary_id,
            ("Plant Type", "Technology Type", "Reactor Type"),
        )
        capacity = _to_numeric(
            _first_present(
                _first_attribute(
                    attrs,
                    dictionary_id,
                    (
                        "Installed Capacity (MW)",
                        "Installed Capacity (MWelec)",
                        "Capacity (MW)",
                    ),
                ),
                _first_attribute(
                    attrs,
                    dictionary_id,
                    (
                        "Installed Capacity (MW)",
                        "Installed Capacity (MWelec)",
                        "Capacity (MW)",
                    ),
                    id_type=None,
                ),
            )
        )
        latitude = _to_numeric(
            _first_present(
                _first_attribute(attrs, dictionary_id, ("Latitude",)),
                _first_attribute(attrs, dictionary_id, ("Latitude",), id_type=None),
            )
        )
        longitude = _to_numeric(
            _first_present(
                _first_attribute(attrs, dictionary_id, ("Longitude",)),
                _first_attribute(attrs, dictionary_id, ("Longitude",), id_type=None),
            )
        )

        for i, bmu_id in enumerate(settlement_ids):
            ngc_bmu_id = national_grid_ids[i] if i < len(national_grid_ids) else pd.NA
            bmu_fuel = (
                _attribute_by_id(attrs, ngc_bmu_id, ("Fuel Type",), "ngc_bmu_id")
                if pd.notna(ngc_bmu_id)
                else pd.NA
            )
            rows.append(
                {
                    "bmu_id": bmu_id,
                    "station_name": station_name,
                    "fuel": bmu_fuel if pd.notna(bmu_fuel) else primary_fuel,
                    "plant_type": plant_type,
                    "installed_capacity_mw": capacity,
                    "node_id": pd.NA,
                    "dictionary_id": dictionary_id,
                    "national_grid_bmu_id": ngc_bmu_id,
                    "gppd_id": source.get("gppd_idnr"),
                    "old_repd_id": source.get("old_repd_id"),
                    "new_repd_id": source.get("new_repd_id"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "source": "OSUKED Power Station Dictionary",
                }
            )

    crosswalk = pd.DataFrame(rows)
    if crosswalk.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crosswalk.to_csv(output_path, index=False)
        return crosswalk

    crosswalk = crosswalk.drop_duplicates(subset=["bmu_id"], keep="first")
    crosswalk = crosswalk.sort_values(["station_name", "bmu_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(output_path, index=False)
    return crosswalk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument("--attributes", type=Path, default=DEFAULT_ATTRS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--ids-url", default=DEFAULT_IDS_URL)
    parser.add_argument("--attributes-url", default=DEFAULT_ATTRS_URL)
    args = parser.parse_args()

    if args.download:
        _download(args.ids_url, args.ids)
        _download(args.attributes_url, args.attributes)

    crosswalk = build_crosswalk(args.ids, args.attributes, args.output)
    print(f"Wrote {len(crosswalk):,} BMU crosswalk rows to {args.output}")


if __name__ == "__main__":
    main()
