"""FES building blocks per edition: cached raw download, then normalisation to FES_{year}_data.csv.

Two steps (year-generic workflow plan, stage 2):

1. **Raw.** ``scripts/sources/fes_raw.py`` downloads each table the FES API file
   (``data/FES/FES_api_urls.yaml``) lists, once per edition, to
   ``resources/sources/raw/fes_{edition}/`` with its URL recorded. Later runs read the cache.
2. **Normalise.** The building blocks are merged with their definitions, the GSP names are cleaned
   by the edition's rows of ``data/FES/gsp_aliases.csv`` (``scripts/sources/registry.py``), mapped to
   GSP ids with the edition's committed GSP info table
   (``data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv``) and to network nodes with
   ``GB_network.xlsx`` ``Dem_per_node``.

An edition with no alias rows or no committed GSP info table fails: there is no fallback to another
edition's names or coordinates.
"""

import io
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.FES.gsp_metadata import map_gsp_names_to_ids, metadata_path_for_year  # noqa: E402
from scripts.sources import fes_raw  # noqa: E402
from scripts.sources.registry import apply_gsp_aliases  # noqa: E402

# Configure logging
try:
    from scripts.utilities.logging_config import setup_logging
    logger = setup_logging("FES_data")
except Exception:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("FES_data")

# Retry configuration
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

#: Cache root of the raw downloads (tests point it at a temporary directory).
RAW_DIR = fes_raw.DEFAULT_RAW_DIR
#: Committed GSP info tables, one per edition.
GSP_INFO_DIR = project_root / "data" / "network" / "ETYS"
#: Node id of each GSP id (sheet Dem_per_node).
NODE_MAP_FILE = project_root / "data" / "network" / "ETYS" / "GB_network.xlsx"
#: The two tables merged into FES_{year}_data.csv.
BUILDING_BLOCK_TABLES = ("building_blocks", "building_block_definitions")
#: How a raw building-block CSV was read before 2026 (kept: FES_{year}_data.csv is unchanged).
RAW_ENCODING = "ISO-8859-1"
#: A UTF-8 byte-order mark as the ISO-8859-1 read sees it.
MOJIBAKE_BOM = "ï»¿"

# Load configuration from YAML file
snk = globals().get('snakemake')
if snk:
    config_file_path = snakemake.input[0]
else:
    config_file_path = str(fes_raw.DEFAULT_URLS_FILE)
try:
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.exception(f"Failed to load FES API config YAML at {config_file_path}: {e}")
    raise


def _clean_text_cells(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop non-ASCII characters and apostrophes from every string cell (the names GSP info uses)."""
    try:
        return df.apply(lambda col: col.map(
            lambda x: re.sub(r"[^\x00-\x7F]+", '', x).replace("'", '').replace("’", '') if isinstance(x, str) else x))
    except Exception as e:
        logger.debug(f"Non-ASCII cleanup failed for {label}: {e}")
        return df


def parse_raw_table(content: bytes, category: str, year: int) -> pd.DataFrame:
    """A raw building-block table as a DataFrame with its string cells cleaned.

    Args:
        content: The bytes NESO served.
        category: Table name (for messages).
        year: FES edition (for messages).

    Returns:
        The table, column names as served.
    """
    df = pd.read_csv(io.BytesIO(content), encoding=RAW_ENCODING, low_memory=False)
    return _clean_text_cells(df, f"{category} {year}")


def fetch_data(category, year):
    """One FES API table for an edition, from the raw cache (downloaded on first use).

    Args:
        category: Table name in the FES API file, e.g. "building_blocks".
        year: FES edition.

    Returns:
        The parsed table, or None when the API file has no URL or the download or parse fails.
    """
    url = config.get(category, {}).get(year)
    if not url:
        logger.error(f"Data for year {year} in category {category} is not available.")
        return None
    try:
        path = fes_raw.fetch_raw(year, category, urls=config, raw_dir=RAW_DIR, session=http, logger=logger)
        df = parse_raw_table(path.read_bytes(), category, year)
        logger.info(f"Fetched {category} for {year}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve data from {url}: {e}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV data for {category} {year}: {e}")
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode CSV data for {category} {year}: {e}")
        return None


def _promote_header_if_needed(definitions: pd.DataFrame) -> pd.DataFrame:
    """Use the first data row as the header when it, not the header, names the id column.

    The FES 2021 and 2022 definitions were once served with a title row above the header; NESO's
    current files (cached 26 Oct 2026) have the header first, so this now changes nothing.
    """
    key = 'Building Block ID Number'
    if key in definitions.columns or definitions.empty or key not in set(definitions.iloc[0].astype(str)):
        return definitions
    promoted = definitions.copy()
    promoted.columns = promoted.iloc[0]
    return promoted[1:]


def normalise_building_blocks(building_blocks: pd.DataFrame, building_block_definitions: pd.DataFrame,
                              FES_year: int) -> pd.DataFrame:
    """Merge the building blocks with their definitions on the block id.

    The building blocks' column names lose a byte-order mark NESO left on the first column
    (FES 2020 'Building Block ID Number', FES 2021-2023 'FES Scenario'). The definitions keep their
    column names as served (FES 2024's first column stays 'ï»¿Template' in FES_2024_data.csv).

    Args:
        building_blocks: Raw building blocks.
        building_block_definitions: Raw definitions.
        FES_year: FES edition (for messages).

    Returns:
        Building blocks with definition columns, indexed by 'Building Block ID Number'.
    """
    definitions = _promote_header_if_needed(building_block_definitions)
    definitions = definitions.set_index('Building Block ID Number')
    blocks = building_blocks.rename(columns=lambda c: str(c).replace(MOJIBAKE_BOM, '').replace('﻿', ''))
    blocks = blocks.set_index('Building Block ID Number')
    # FES 2021-2023 name the pathway column 'FES Scenario'; FES 2024+ 'FES Pathway', which the build reads.
    # The older editions get the 2024 name as well (19 Nov 2026: a FES 2023 build failed on KeyError 'FES Pathway').
    if 'FES Pathway' not in blocks.columns and 'FES Scenario' in blocks.columns:
        blocks['FES Pathway'] = blocks['FES Scenario']
    merged = blocks.merge(definitions, left_index=True, right_index=True)
    logger.info(f"Merged FES {FES_year} building blocks and definitions: {merged.shape[0]} rows, "
                f"{merged.shape[1]} columns")
    return merged


def raw_FES_data(FES_year):
    """FES building blocks merged with their definitions for one edition.

    Raises:
        RuntimeError: Either table is not available.
    """
    year = str(FES_year)
    logger.info(f"Loading raw FES data for year {year}")
    tables = {}
    for category in BUILDING_BLOCK_TABLES:
        if category not in config:
            continue
        df = fetch_data(category, FES_year)
        if df is None:
            logger.warning(f"No data returned for category {category} year {year}")
            continue
        tables[category] = df

    if 'building_blocks' not in tables:
        logger.error("building_blocks data not available, aborting")
        raise RuntimeError("building_blocks data not available")
    if 'building_block_definitions' not in tables:
        logger.error("building_block_definitions data not available, aborting")
        raise RuntimeError("building_block_definitions data not available")
    return normalise_building_blocks(tables['building_blocks'], tables['building_block_definitions'], FES_year)


def load_gsp_info(FES_year: int, gsp_info_dir: Path = GSP_INFO_DIR) -> pd.DataFrame:
    """The committed GSP info table of one FES edition.

    FES 2021-2024 are NESO's regional-breakdown tables (re-encoded as UTF-8); FES 2025 has no NESO
    table and is derived from the GSP region polygons (``gsp_metadata.derive_gsp_metadata_from_spatial``).

    Raises:
        FileNotFoundError: The edition has no committed table. Cache NESO's table with
            ``python -m scripts.sources.fes_raw --edition N --table gsp_info`` (or derive one from the
            GSP polygons), check it and commit it under ``data/network/ETYS/``.
    """
    path = metadata_path_for_year(int(FES_year), Path(gsp_info_dir))
    if not path.is_file():
        raise FileNotFoundError(
            f"No GSP info for FES {FES_year}: {path} is missing. Cache NESO's table "
            f"(python -m scripts.sources.fes_raw --edition {FES_year} --table gsp_info) or derive one from the "
            "GSP polygons (scripts/FES/gsp_metadata.py), check it and commit it; no other edition is used.")
    gsp_info = pd.read_csv(path, encoding='utf-8-sig')
    logger.info(f"GSP info for FES {FES_year}: {len(gsp_info)} rows from {path}")
    return gsp_info


def add_node_id_to_gsp(FES_year, raw_FES_data, gsp_info: Optional[pd.DataFrame] = None,
                       outpath: Optional[str | Path] = None, node_map_file: str | Path = NODE_MAP_FILE):
    """Clean GSP names, map them to GSP and node ids, and write FES_{year}_data.csv.

    GSP names change between FES editions (offshore wind named with its connection point from FES
    2024, London LPN/SPN subcircuits, island GSPs not in ETYS); the edition's rows of
    ``data/FES/gsp_aliases.csv`` say how each is cleaned or dropped. A new edition needs its rows.

    Args:
        FES_year: FES edition.
        raw_FES_data: Output of :func:`raw_FES_data`.
        gsp_info: GSP info table; default the committed one (:func:`load_gsp_info`).
        outpath: Output CSV; default the Snakemake output, else resources/FES/FES_{year}_data.csv.
        node_map_file: Workbook with the Dem_per_node sheet.

    Returns:
        The table written.
    """
    logger.info(f"Cleaning GSP names and mapping Node IDs for FES year {FES_year}")
    raw_FES_data = apply_gsp_aliases(raw_FES_data, int(FES_year), column='GSP')
    logger.info(f"Applied the FES {FES_year} GSP aliases: {len(raw_FES_data)} rows kept")

    GSP_info = load_gsp_info(FES_year) if gsp_info is None else gsp_info.copy()
    GSP_info['Name'] = GSP_info['Name'].str.replace("'", "")
    GSP_info['Name'] = GSP_info['Name'].str.replace("’", '')

    # Map GSP names to IDs using exact match first, then canonical labels.
    raw_FES_data['GSP ID'] = map_gsp_names_to_ids(raw_FES_data['GSP'], GSP_info)
    missing_gsp_ids = raw_FES_data['GSP ID'].isna().sum()
    logger.info(f"Mapped GSP IDs; missing GSP ID count: {missing_gsp_ids}")

    node_info = pd.read_excel(node_map_file, sheet_name="Dem_per_node")
    gsp_to_node_mapping = dict(zip(node_info['GSP Id'], node_info['Node Id']))
    raw_FES_data['Node ID'] = raw_FES_data['GSP ID'].map(gsp_to_node_mapping)
    missing_node_ids = raw_FES_data['Node ID'].isna().sum()
    logger.info(f"Mapped Node IDs; missing Node ID count: {missing_node_ids}")

    # Transmission-connected rows ('Direct(...)') keep the GSP label as both ids.
    direct_mask = raw_FES_data['GSP'].str.contains('Direct', na=False)
    raw_FES_data.loc[direct_mask, 'Node ID'] = raw_FES_data.loc[direct_mask, 'GSP']
    raw_FES_data.loc[direct_mask, 'GSP ID'] = raw_FES_data.loc[direct_mask, 'GSP']

    if outpath is None:
        snk = globals().get('snakemake')
        outpath = snk.output[0] if snk else os.path.join('resources', 'FES', f'FES_{FES_year}_data.csv')
    logger.info(f"Writing output FES file to {outpath}")
    raw_FES_data.to_csv(outpath, index=True)
    return raw_FES_data


def build_fes_data(FES_year: int, outpath: Optional[str | Path] = None) -> pd.DataFrame:
    """Raw building blocks to FES_{year}_data.csv for one edition."""
    return add_node_id_to_gsp(FES_year, raw_FES_data(FES_year), outpath=outpath)


if __name__ == "__main__":
    snk = globals().get('snakemake')

    # Reinitialize logger with Snakemake log path if available
    if snk and hasattr(snk, 'log') and snk.log:
        logger = setup_logging(snk.log[0])

    out = None
    if snk:
        FES_year = snk.params.fes_year
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Build resources/FES/FES_{year}_data.csv for one FES edition")
        parser.add_argument("--edition", type=int, default=2024)
        parser.add_argument("--out", default=None)
        args = parser.parse_args()
        FES_year, out = args.edition, args.out
    try:
        logger.info(f"Starting FES data pipeline for year {FES_year}")
        build_fes_data(FES_year, outpath=out)
        logger.info("FES data pipeline finished successfully")
    except Exception as e:
        logger.exception(f"FES data pipeline failed: {e}")
        raise
