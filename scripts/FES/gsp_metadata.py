"""Utilities for locating and generating FES GSP metadata files."""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

METADATA_COLUMNS = ["GSP ID", "GSP Group", "Minor FLOP", "Name", "Latitude", "Longitude", "Comments"]
DEFAULT_METADATA_DIR = Path("data/network/ETYS")
DEFAULT_GSP_DIR = Path("data/network/GSP")
DEFAULT_FES_RESOURCES_DIR = Path("resources/FES")


def _log(logger: Optional[logging.Logger]) -> logging.Logger:
    return logger or logging.getLogger(__name__)


def canonical_gsp_name(value) -> str:
    """Normalize GSP labels across FES releases for joins."""

    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", "and")
    text = text.replace("'", "")
    text = text.replace("/", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\d+\s*k?v", " ", text)
    text = re.sub(r"\b(lpn|spn|enw|spen)\b", " ", text)
    text = re.sub(r"\b(_[a-z])\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    text = text.replace("bridgwater", "bridgewater")
    text = text.replace("stjohnswood", "stjohnwood")
    text = text.replace("hartmoor", "hartmoor")
    return text


def gsp_name_variants(value) -> list[str]:
    raw = str(value)
    variants = [canonical_gsp_name(raw)]
    variants.extend(canonical_gsp_name(part) for part in re.findall(r"\(([^)]*)\)", raw))
    variants.append(canonical_gsp_name(raw.split("(", 1)[0]))
    seen = set()
    ordered = []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            ordered.append(variant)
    return ordered


def metadata_path_for_year(fes_year: int, metadata_dir: Path = DEFAULT_METADATA_DIR) -> Path:
    return metadata_dir / f"fes{fes_year}_regional_breakdown_gsp_info.csv"


def save_gsp_metadata_file(
    gsp_info: pd.DataFrame,
    fes_year: int,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Save a GSP metadata dataframe using the repository's standard filename."""

    logger = _log(logger)
    missing = [column for column in METADATA_COLUMNS if column not in gsp_info.columns]
    if missing:
        raise ValueError(f"GSP metadata missing required columns: {missing}")

    output_path = metadata_path_for_year(fes_year, metadata_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = gsp_info[METADATA_COLUMNS].copy()
    cleaned = cleaned.drop_duplicates(subset=["GSP ID"], keep="first")
    cleaned.to_csv(output_path, index=False)
    logger.info("Saved FES %s GSP metadata to %s (%d rows)", fes_year, output_path, len(cleaned))
    return output_path


def available_metadata_files(metadata_dir: Path = DEFAULT_METADATA_DIR) -> list[tuple[int, Path]]:
    files = []
    for candidate in metadata_dir.glob("fes*_regional_breakdown_gsp_info.csv"):
        match = re.search(r"fes(\d{4})_regional_breakdown_gsp_info\.csv", candidate.name)
        if match:
            files.append((int(match.group(1)), candidate))
    return sorted(files)


def _latest_existing_metadata(
    fes_year: int,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    exclude_year: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    candidates = [item for item in available_metadata_files(metadata_dir) if item[0] <= fes_year]
    if exclude_year is not None:
        candidates = [item for item in candidates if item[0] != exclude_year]
    if not candidates:
        return None
    _, path = candidates[-1]
    return pd.read_csv(path)


def _read_fes_name_pairs(fes_year: int, fes_data=None, fes_resources_dir: Path = DEFAULT_FES_RESOURCES_DIR) -> pd.DataFrame:
    if fes_data is not None:
        df = fes_data.copy()
    else:
        path = fes_resources_dir / f"FES_{fes_year}_data.csv"
        if not path.exists():
            return pd.DataFrame(columns=["GSP", "GSP ID"])
        usecols = [column for column in ["GSP", "GSP ID"] if column in pd.read_csv(path, nrows=0).columns]
        if len(usecols) < 2:
            return pd.DataFrame(columns=["GSP", "GSP ID"])
        df = pd.read_csv(path, usecols=usecols)

    if not {"GSP", "GSP ID"}.issubset(df.columns):
        return pd.DataFrame(columns=["GSP", "GSP ID"])

    pairs = df[["GSP", "GSP ID"]].dropna().copy()
    pairs["GSP"] = pairs["GSP"].astype(str)
    pairs["GSP ID"] = pairs["GSP ID"].astype(str)
    pairs = pairs[~pairs["GSP"].str.contains("Direct", na=False)]
    pairs = pairs[~pairs["GSP ID"].str.contains("Direct", na=False)]
    return pairs.drop_duplicates(subset=["GSP ID"], keep="first")


def _name_sources_by_id(fes_year: int, fes_data=None, metadata_dir: Path = DEFAULT_METADATA_DIR) -> dict[str, str]:
    names: dict[str, str] = {}

    older = _latest_existing_metadata(fes_year, metadata_dir, exclude_year=fes_year)
    if older is not None and {"GSP ID", "Name"}.issubset(older.columns):
        for _, row in older.dropna(subset=["GSP ID", "Name"]).iterrows():
            names[str(row["GSP ID"])] = str(row["Name"])

    pairs = _read_fes_name_pairs(fes_year, fes_data=fes_data)
    for _, row in pairs.iterrows():
        names[str(row["GSP ID"])] = str(row["GSP"])

    return names


def _old_metadata_by_id(fes_year: int, metadata_dir: Path = DEFAULT_METADATA_DIR) -> dict[str, pd.Series]:
    older = _latest_existing_metadata(fes_year, metadata_dir, exclude_year=fes_year)
    if older is None or "GSP ID" not in older.columns:
        return {}
    return {str(row["GSP ID"]): row for _, row in older.dropna(subset=["GSP ID"]).iterrows()}


def _spatial_candidates(gsp_dir: Path = DEFAULT_GSP_DIR) -> list[tuple[int, Path]]:
    candidates = []
    for pattern in ["**/GSP_regions_*.shp", "**/GSP_regions_*.geojson"]:
        for path in gsp_dir.glob(pattern):
            match = re.search(r"(20\d{6})", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    return sorted(set(candidates), key=lambda item: (item[0], str(item[1])))


def _select_spatial_file(fes_year: int, gsp_dir: Path = DEFAULT_GSP_DIR) -> Optional[Path]:
    candidates = _spatial_candidates(gsp_dir)
    if not candidates:
        return None

    target = fes_year * 10000 + 1231
    previous = [item for item in candidates if item[0] <= target]
    if previous:
        same_or_previous_year = [item for item in previous if item[0] // 10000 <= fes_year]
        pool = same_or_previous_year or previous
        selected_date = max(item[0] for item in pool)
    else:
        selected_date = min(item[0] for item in candidates)

    dated = [path for date, path in candidates if date == selected_date]
    # Prefer explicit WGS84 GeoJSON if available, otherwise use the shapefile.
    dated = sorted(dated, key=lambda path: ("4326" not in path.name, path.suffix != ".shp", str(path)))
    return dated[0]


def _split_gsp_codes(value) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def derive_gsp_metadata_from_spatial(
    fes_year: int,
    fes_data=None,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    gsp_dir: Path = DEFAULT_GSP_DIR,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """Build FES-style GSP metadata from the local GSP region spatial layer."""

    logger = _log(logger)
    spatial_path = _select_spatial_file(fes_year, gsp_dir)
    if spatial_path is None:
        logger.warning("No GSP spatial file available under %s", gsp_dir)
        return None

    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas is not installed; cannot derive GSP metadata from %s", spatial_path)
        return None

    gdf = gpd.read_file(spatial_path)
    if not {"GSPs", "GSPGroup"}.issubset(gdf.columns):
        logger.warning("GSP spatial file %s does not contain GSPs/GSPGroup columns", spatial_path)
        return None

    if gdf.crs is None:
        bounds = gdf.total_bounds
        if -20 <= bounds[0] <= 20 and -20 <= bounds[2] <= 20 and 40 <= bounds[1] <= 70 and 40 <= bounds[3] <= 70:
            logger.warning("GSP spatial file %s has no CRS; assuming EPSG:4326 from bounds", spatial_path)
            gdf = gdf.set_crs("EPSG:4326")
        else:
            logger.warning("GSP spatial file %s has no CRS; assuming EPSG:27700", spatial_path)
            gdf = gdf.set_crs("EPSG:27700")

    bounds = gdf.total_bounds
    looks_lonlat = -20 <= bounds[0] <= 20 and -20 <= bounds[2] <= 20 and 40 <= bounds[1] <= 70 and 40 <= bounds[3] <= 70
    if looks_lonlat:
        # The 2025 GSP GeoJSON is already WGS84. Local PROJ transforms can fail
        # in some environments, so use geographic centroids for nearest-bus
        # assignment and suppress geopandas' projection warning at this point.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Geometry is in a geographic CRS.*",
                category=UserWarning,
            )
            centroids_wgs84 = gdf.geometry.centroid
    else:
        try:
            centroid_source = gdf.to_crs("EPSG:27700")
            centroids = centroid_source.geometry.centroid
            centroids_wgs84 = gpd.GeoSeries(centroids, crs="EPSG:27700").to_crs("EPSG:4326")
        except Exception as exc:
            logger.warning("Could not transform %s to WGS84 for centroids: %s", spatial_path, exc)
            centroids_wgs84 = gpd.GeoSeries([None] * len(gdf), crs="EPSG:4326")

    names_by_id = _name_sources_by_id(fes_year, fes_data=fes_data, metadata_dir=metadata_dir)
    old_by_id = _old_metadata_by_id(fes_year, metadata_dir)

    rows = []
    seen = set()
    for idx, source_row in gdf.iterrows():
        point = centroids_wgs84.iloc[idx]
        if point is None or point.is_empty:
            latitude = pd.NA
            longitude = pd.NA
        else:
            latitude = point.y
            longitude = point.x
        for gsp_id in _split_gsp_codes(source_row["GSPs"]):
            if gsp_id in seen:
                continue
            seen.add(gsp_id)
            old_row = old_by_id.get(gsp_id)
            rows.append(
                {
                    "GSP ID": gsp_id,
                    "GSP Group": old_row.get("GSP Group") if old_row is not None and "GSP Group" in old_row else source_row.get("GSPGroup"),
                    "Minor FLOP": old_row.get("Minor FLOP") if old_row is not None and "Minor FLOP" in old_row else "",
                    "Name": names_by_id.get(gsp_id, old_row.get("Name") if old_row is not None and "Name" in old_row else gsp_id),
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "Comments": f"Derived from {spatial_path.as_posix()} centroid",
                }
            )

    # Retain older metadata rows that are not represented by the spatial file.
    for gsp_id, old_row in old_by_id.items():
        if gsp_id in seen:
            continue
        rows.append(
            {
                "GSP ID": gsp_id,
                "GSP Group": old_row.get("GSP Group", ""),
                "Minor FLOP": old_row.get("Minor FLOP", ""),
                "Name": names_by_id.get(gsp_id, old_row.get("Name", gsp_id)),
                "Latitude": old_row.get("Latitude", pd.NA),
                "Longitude": old_row.get("Longitude", pd.NA),
                "Comments": old_row.get("Comments", ""),
            }
        )

    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    logger.info(
        "Derived FES %s GSP metadata from %s: %d rows, %d with coordinates",
        fes_year,
        spatial_path,
        len(metadata),
        metadata.dropna(subset=["Latitude", "Longitude"]).shape[0],
    )
    return metadata


def ensure_gsp_metadata_file(
    fes_year: int,
    fes_data=None,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    gsp_dir: Path = DEFAULT_GSP_DIR,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Return an exact-year metadata file, generating it from spatial data if needed."""

    logger = _log(logger)
    output_path = metadata_path_for_year(fes_year, metadata_dir)
    if output_path.exists():
        return output_path

    metadata = derive_gsp_metadata_from_spatial(
        fes_year,
        fes_data=fes_data,
        metadata_dir=metadata_dir,
        gsp_dir=gsp_dir,
        logger=logger,
    )
    if metadata is None or metadata.empty:
        return None
    return save_gsp_metadata_file(metadata, fes_year, metadata_dir, logger)


def map_gsp_names_to_ids(gsp_names: pd.Series, gsp_info: pd.DataFrame) -> pd.Series:
    """Map GSP names to IDs using exact match first, then canonical labels."""

    exact = dict(zip(gsp_info["Name"].astype(str), gsp_info["GSP ID"].astype(str)))
    canonical = {}
    for _, row in gsp_info.dropna(subset=["Name", "GSP ID"]).iterrows():
        key = canonical_gsp_name(row["Name"])
        if key and key not in canonical:
            canonical[key] = str(row["GSP ID"])

    def lookup(name):
        if pd.isna(name):
            return pd.NA
        raw = str(name)
        compact = re.sub(r"[^a-z0-9]+", "", raw.lower())
        if "tynemouth" in compact and re.search(r"(^|[^0-9])32\s*kv", raw.lower()) and "TYNE_2" in set(gsp_info["GSP ID"].astype(str)):
            return "TYNE_2"
        if "tynemouth" in compact and re.search(r"(^|[^0-9])1\s*kv", raw.lower()) and "TYNE_1" in set(gsp_info["GSP ID"].astype(str)):
            return "TYNE_1"
        if raw in exact:
            return exact[raw]
        for key in gsp_name_variants(raw):
            if key in canonical:
                return canonical[key]
        return pd.NA

    return gsp_names.map(lookup)