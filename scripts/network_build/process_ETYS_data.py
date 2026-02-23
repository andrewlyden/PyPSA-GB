"""
Process raw ETYS Excel data into standardized intermediate CSV files.

This script separates the slow Excel I/O and data parsing from the network
model construction step. It reads the ETYS Appendix B Excel file, GB_network.xlsx,
and FES regional data, then outputs:
  - components CSV: all lines, transformers, and links with standardized columns
  - buses CSV: unique buses with voltage levels, carriers, GSP coordinates,
    and offshore classification

The outputs are consumed by ETYS_network.py which handles coordinate guessing
and PyPSA network assembly.

Inputs (via snakemake.input):
  [0] ETYS Appendix B Excel file (e.g. "ETYS 2024 Appendix-B V1.xlsx")
  [1] GB_network.xlsx (extra WF/BMU edges)
  [2] FES regional data (Excel .xlsb/.xlsx or CSV fallback)

Outputs (via snakemake.output):
  [0] components CSV
  [1] buses CSV
"""

import pandas as pd
import numpy as np
import logging
import time
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.utilities.logging_config import setup_logging, log_dataframe_info
from scripts.network_build.etys_file_registry import (
    VOLTAGE_LEVELS, ELECTRICAL_DEFAULTS, ETYS_BASE_SHEETS,
    EXTRA_WF_EDGE_RATINGS, DEFAULT_WF_RATING,
)


def sort_raw_ETYS_data(etys_file: str, gb_network_file: str,
                       logger: Optional[logging.Logger] = None):
    """
    Parse and process raw ETYS data from Excel sheets.

    Returns:
        Tuple of (components_df, offshore_wf_buses_set)
        - components_df: DataFrame with columns [component, carrier, bus0, bus1, r, x, b, s_nom, length_km]
        - offshore_wf_buses: set of bus IDs that are offshore wind farm buses
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Reading ETYS raw data from: {etys_file}")
    xls = pd.ExcelFile(etys_file)
    logger.info(f"Loaded Excel file with {len(xls.sheet_names)} sheets")

    # Parse all base network sheets from the registry
    all_sheet_names = (ETYS_BASE_SHEETS['circuits']
                       + ETYS_BASE_SHEETS['transformers']
                       + [ETYS_BASE_SHEETS['hvdc']])
    sheets = {name: xls.parse(name, skiprows=1) for name in all_sheet_names}

    circuit_sheet_names = ETYS_BASE_SHEETS['circuits']
    logger.info(f"Processing line data sheets ({circuit_sheet_names[0]} to {circuit_sheet_names[-1]})")
    dfa, dfb, dfc, dfd = [sheets[s] for s in circuit_sheet_names]
    for df in [dfa, dfb, dfc, dfd]:
        df.loc[:, 'component'] = 'line'
        df.loc[:, 'carrier'] = 'AC'
    # Normalize OFTO column names to match main ETYS format
    # OFTO sheets use slightly different naming: no space before (km), no space before MVA
    dfd.rename(columns={
        'R (% on 100MVA)': 'R (% on 100 MVA)',
        'X (% on 100MVA)': 'X (% on 100 MVA)',
        'B (% on 100MVA)': 'B (% on 100 MVA)',
        'Rating (MVA)': 'Winter Rating (MVA)',
        'OHL Length(km)': 'OHL Length (km)',
        'Cable Length(km)': 'Cable Length (km)',
    }, inplace=True)

    xfmr_sheet_names = ETYS_BASE_SHEETS['transformers']
    logger.info(f"Processing transformer data sheets ({xfmr_sheet_names[0]} to {xfmr_sheet_names[-1]})")
    dfe, dff, dfg, dfh = [sheets[s] for s in xfmr_sheet_names]
    dfe.rename(columns={'Rating (MVA)': 'Winter Rating (MVA)'}, inplace=True)
    for df in [dff, dfg, dfh]:
        df.rename(columns={
            'R (% on 100MVA)': 'R (% on 100 MVA)',
            'X (% on 100MVA)': 'X (% on 100 MVA)',
            'B (% on 100MVA)': 'B (% on 100 MVA)',
            'Node1': 'Node 1', 'Node2': 'Node 2',
            'Rating (MVA)': 'Winter Rating (MVA)'
        }, inplace=True)
    for df in [dfe, dff, dfg, dfh]:
        df.loc[:, 'component'] = 'transformer'
        df.loc[:, 'carrier'] = 'AC'

    hvdc_sheet = ETYS_BASE_SHEETS['hvdc']
    logger.info(f"Processing interconnector data ({hvdc_sheet})")
    dfi = sheets[hvdc_sheet]
    dfi = dfi.loc[dfi['Existing'] == 'Yes'].copy()
    logger.info(f"Found {len(dfi)} existing interconnectors")
    dfi.loc[:, 'component'] = 'link'
    dfi.loc[:, 'carrier'] = 'DC'

    logger.info(f"Loading additional wind farm and BMU edges from: {gb_network_file}")
    dfj = pd.read_excel(gb_network_file, sheet_name='Extra_WF_edges')
    dfj.loc[:, 'component'] = 'line'
    dfj.loc[:, 'carrier'] = 'AC'
    DEFAULT_BMU_RATING = 500  # MVA, conservative default for BMU connections

    node1_col = 'Node 1' if 'Node 1' in dfj.columns else 'Node1'
    dfj.loc[:, 'Winter Rating (MVA)'] = dfj[node1_col].map(EXTRA_WF_EDGE_RATINGS).fillna(DEFAULT_WF_RATING)
    rated_count = (dfj['Winter Rating (MVA)'] != DEFAULT_WF_RATING).sum()
    unmapped_mask = ~dfj[node1_col].isin(EXTRA_WF_EDGE_RATINGS)
    if unmapped_mask.any():
        unmapped_nodes = dfj.loc[unmapped_mask, node1_col].unique().tolist()
        logger.warning(
            f"{len(unmapped_nodes)} WF edge nodes have no OFTO-derived rating, "
            f"using default {DEFAULT_WF_RATING} MVA: {unmapped_nodes}"
        )
    logger.info(f"Loaded {len(dfj)} extra wind farm edges ({rated_count} with OFTO-derived ratings)")

    # Set cable-appropriate electrical parameters for offshore WF connections.
    # Submarine cables have higher R and B, lower X than overhead lines.
    cable_defaults = ELECTRICAL_DEFAULTS['cable']
    dfj.loc[:, 'R (% on 100 MVA)'] = cable_defaults['r'] * 100  # Will be /100 later
    dfj.loc[:, 'X (% on 100 MVA)'] = cable_defaults['x'] * 100
    dfj.loc[:, 'B (% on 100 MVA)'] = cable_defaults['b'] * 100

    dfk = pd.read_excel(gb_network_file, sheet_name='Extra_BMUs_edges')
    dfk.loc[:, 'component'] = 'line'
    dfk.loc[:, 'carrier'] = 'AC'
    dfk.loc[:, 'Winter Rating (MVA)'] = DEFAULT_BMU_RATING
    logger.info(f"Loaded {len(dfk)} extra BMU edges (default rating: {DEFAULT_BMU_RATING} MVA)")

    # Identify true offshore buses: buses that appear ONLY in OFTO data (B-2-1d,
    # B-3-1d) and NOT in any main ETYS sheet (B-2-1a/b/c, B-3-1a/b/c).
    logger.info("Identifying offshore wind farm buses")
    main_etys_buses = set()
    for _df in [dfa, dfb, dfc, dfe, dff, dfg]:
        main_etys_buses.update(_df['Node 1'].dropna().tolist())
        main_etys_buses.update(_df['Node 2'].dropna().tolist())

    ofto_buses = set()
    for _df in [dfd, dfh]:
        ofto_buses.update(_df['Node 1'].dropna().tolist())
        ofto_buses.update(_df['Node 2'].dropna().tolist())

    offshore_wf_buses = ofto_buses - main_etys_buses
    logger.info(f"Identified {len(offshore_wf_buses)} offshore buses (in OFTO data only)")

    # Concatenate all components
    logger.info("Concatenating all network components")
    df = pd.concat([dfa, dfb, dfc, dfd, dfe, dff, dfg, dfh, dfi, dfj, dfk], ignore_index=True)
    df.rename(columns={'Node 1': 'bus0', 'Node 2': 'bus1'}, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Generate descriptive component IDs: {bus0}_{bus1}_{n} for parallel circuits
    counter = {}
    ids = []
    for _, row in df.iterrows():
        key = f"{row['bus0']}_{row['bus1']}"
        n = counter.get(key, 0)
        counter[key] = n + 1
        ids.append(f"{key}_{n}")
    df.index = ids
    df.index.name = 'name'
    df.rename(columns={
        'R (% on 100 MVA)': 'r',
        'X (% on 100 MVA)': 'x',
        'B (% on 100 MVA)': 'b',
        'Winter Rating (MVA)': 's_nom'
    }, inplace=True)

    # Calculate total length for distance-based coordinate estimation
    if 'OHL Length (km)' in df.columns and 'Cable Length (km)' in df.columns:
        df['length_km'] = df['OHL Length (km)'].fillna(0) + df['Cable Length (km)'].fillna(0)
    elif 'Length(km)' in df.columns:
        df['length_km'] = df['Length(km)'].fillna(0)
    else:
        df['length_km'] = 0

    df = df[['component', 'carrier', 'bus0', 'bus1', 'r', 'x', 'b', 's_nom', 'length_km']]

    # Convert electrical parameters
    logger.info("Processing electrical parameters")
    df['r'] = df['r'].astype('float64')
    df['x'] = df['x'].astype('float64')
    df['b'] = df['b'].astype('float64')

    # Convert from "% on 100 MVA" to per-unit
    logger.info("Converting electrical parameters from % on 100 MVA base to per-unit")
    df['r'] = df['r'] / 100.0
    df['x'] = df['x'] / 100.0
    df['b'] = df['b'] / 100.0

    # Apply component-type-aware defaults (lines vs transformers vs links)
    for comp_type in ['line', 'transformer']:
        defaults = ELECTRICAL_DEFAULTS[comp_type]
        mask = df['component'] == comp_type
        for col in ['r', 'x', 'b']:
            zero_or_nan = mask & ((df[col] == 0) | df[col].isna())
            df.loc[zero_or_nan, col] = defaults[col]

    # Links use p_nom not impedance — zero out R/X/B
    df.loc[df['component'] == 'link', ['r', 'x', 'b']] = 0

    # Validate parameter ranges (warn on out-of-range values, skip links)
    non_link = df[df['component'] != 'link']
    for param, (lo, hi) in [('r', (0.0001, 0.05)), ('x', (0.0001, 0.1)), ('b', (0.0, 5.0))]:
        out_of_range = ~non_link[param].between(lo, hi)
        count = out_of_range.sum()
        if count > 0:
            logger.warning(f"{count} components have {param} outside range [{lo}, {hi}]")

    logger.info(f"Processed {len(df)} network components")
    log_dataframe_info(df, logger, "Network components summary")

    return df, offshore_wf_buses


def buses_from_line_data(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Extract unique buses from network component data and add voltage levels."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Extracting buses from line data")
    df_buses = pd.concat([df['bus0'], df['bus1']]).unique()
    df_buses = pd.DataFrame(df_buses, columns=['name'])
    df_buses.index = df_buses['name']
    df_buses.index.name = 'name'
    logger.info(f"Found {len(df_buses)} unique buses")

    # Add voltage data using the ETYS naming convention: XXXX# where position [4]
    # is the voltage digit (1=132kV, 2=275kV, 3=33kV, 4=400kV, etc.)
    logger.info("Adding voltage level data to buses")
    voltage_digit = df_buses['name'].str[4]
    df_buses['v_nom'] = voltage_digit.map(VOLTAGE_LEVELS)

    unmapped = df_buses['v_nom'].isna()
    if unmapped.any():
        logger.warning(f"{unmapped.sum()} buses have unmapped voltage digits, "
                       f"defaulting to 400kV: "
                       f"{df_buses.loc[unmapped, 'name'].head(10).tolist()}")
        df_buses.loc[unmapped, 'v_nom'] = 400

    # All buses get AC carrier
    df_buses['carrier'] = 'AC'

    logger.info(f"Completed bus processing with voltage levels and carriers")
    log_dataframe_info(df_buses, logger, "Buses summary")

    return df_buses


def GSP_locations_from_FES_data(fes_file: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load Grid Supply Point (GSP) location data from FES data.

    Handles multiple file formats:
      - .xlsb (FES 2023 binary Excel)
      - .xlsx (FES 2024 standard Excel)
      - .csv  (CSV fallback for 2021/2022)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading GSP location data from: {fes_file}")

    if fes_file.endswith('.csv'):
        df2 = pd.read_csv(fes_file)
        df2 = df2.set_index('GSP ID')
        df2.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
    else:
        # Excel format (.xlsb or .xlsx)
        df2 = pd.read_excel(fes_file, sheet_name='GSP info', skiprows=4, index_col=1)
        df2.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
        df2.drop(columns=['Name'], inplace=True, errors='ignore')

    df2 = df2[~df2.index.duplicated(keep='first')]
    df2.index.name = 'name'
    df2['name'] = df2.index

    logger.info(f"Loaded {len(df2)} GSP locations")
    log_dataframe_info(df2, logger, "GSP locations summary")
    return df2


def load_node_to_gsp_mapping(gb_network_file: str,
                              logger: Optional[logging.Logger] = None) -> dict:
    """
    Load explicit Node ID → GSP ID mapping from GB_network.xlsx Dem_per_node sheet.

    This gives a much more accurate bus-to-GSP mapping than the 4-character prefix
    approach, especially in Scotland where many substations share prefixes.

    Returns:
        dict mapping Node ID (str) → GSP ID (str)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading Node-to-GSP mapping from: {gb_network_file} (Dem_per_node)")
    df_dem = pd.read_excel(gb_network_file, sheet_name='Dem_per_node')

    # Build node → GSP mapping (take the GSP with highest demand share per node)
    node_to_gsp = {}
    for _, row in df_dem.iterrows():
        node_id = str(row['Node Id']).strip()
        gsp_id = str(row['GSP Id']).strip()
        pct = row.get('Dem as % of demand within the GSP Group ID per each node', 0)
        if node_id not in node_to_gsp or pct > node_to_gsp[node_id][1]:
            node_to_gsp[node_id] = (gsp_id, pct)

    # Simplify to node → gsp_id only
    node_to_gsp = {k: v[0] for k, v in node_to_gsp.items()}
    logger.info(f"Loaded {len(node_to_gsp)} node-to-GSP mappings")
    return node_to_gsp


def load_substation_coordinates(substation_coords_file: str,
                                 logger: Optional[logging.Logger] = None) -> dict:
    """
    Load supplementary substation coordinates from CSV lookup file.

    This file maps ETYS 4-char site codes to lat/lon coordinates, generated
    by cross-referencing ETYS substation names with generator databases and
    geocoding services.

    Returns:
        dict mapping site_code (str) → (lat, lon) tuple
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        df = pd.read_csv(substation_coords_file)
        coords = {}
        for _, row in df.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                coords[str(row['site_code']).strip()] = (row['lat'], row['lon'])
        logger.info(f"Loaded {len(coords)} substation coordinates from {substation_coords_file}")
        return coords
    except FileNotFoundError:
        logger.warning(f"Substation coordinates file not found: {substation_coords_file}")
        return {}


def add_GSP_location_data(df_buses: pd.DataFrame, df_gsp: pd.DataFrame,
                          node_to_gsp: dict,
                          substation_coords: Optional[dict] = None,
                          logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Add coordinate data to buses using a three-tier matching strategy:

    1. Explicit mapping: Use Dem_per_node (Node ID → GSP ID) for exact matches
    2. Substation lookup: Use geocoded substation coordinates (4-char site code)
    3. Prefix fallback: Match first 4 characters of bus name to GSP ID

    This avoids the Scotland problem where many buses share 4-char prefixes
    but belong to different GSPs at different locations.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if substation_coords is None:
        substation_coords = {}

    logger.info("Matching buses to GSP locations")
    explicit_matches = 0
    substation_matches = 0
    prefix_matches = 0
    unmatched = 0

    lons = []
    lats = []
    coord_sources = []  # Track which tier assigned coordinates
    for bus_name in df_buses.index:
        lon, lat = None, None

        # Tier 1: Explicit Dem_per_node mapping
        if bus_name in node_to_gsp:
            gsp_id = node_to_gsp[bus_name]
            if gsp_id in df_gsp.index:
                lon = df_gsp.loc[gsp_id, 'lon']
                lat = df_gsp.loc[gsp_id, 'lat']
                if pd.notna(lon) and pd.notna(lat):
                    explicit_matches += 1
                    lons.append(lon)
                    lats.append(lat)
                    coord_sources.append('gsp_explicit')
                    continue

        # Tier 2: Substation coordinates lookup (4-char site code)
        site_code = bus_name[:4] if len(bus_name) >= 4 else bus_name
        if site_code in substation_coords:
            lat, lon = substation_coords[site_code]
            substation_matches += 1
            lons.append(lon)
            lats.append(lat)
            coord_sources.append('substation_lookup')
            continue

        # Tier 3: 4-character prefix match to GSP
        matching = df_gsp[df_gsp.index.str[:4] == bus_name[:4]]
        if not matching.empty:
            lon = matching.iloc[0]['lon']
            lat = matching.iloc[0]['lat']
            if pd.notna(lon) and pd.notna(lat):
                prefix_matches += 1
                lons.append(lon)
                lats.append(lat)
                coord_sources.append('gsp_prefix')
                continue

        unmatched += 1
        lons.append(None)
        lats.append(None)
        coord_sources.append(None)

    df_buses['lon'] = pd.to_numeric(lons, errors='coerce')
    df_buses['lat'] = pd.to_numeric(lats, errors='coerce')
    df_buses['coord_source'] = coord_sources

    df_buses = df_buses.drop(columns=['name'])

    total = len(df_buses)
    logger.info(f"GSP matching: {explicit_matches} explicit, {substation_matches} substation, "
                f"{prefix_matches} prefix, {unmatched} unmatched (total {total})")

    return df_buses


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION (called by Snakemake)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = time.time()

    log_path = None
    if 'snakemake' in globals() and hasattr(snakemake, 'log') and snakemake.log:
        log_path = snakemake.log[0]
    logger = setup_logging(log_path or "process_ETYS_data")

    logger.info("=" * 50)
    logger.info("STARTING ETYS DATA PROCESSING")

    try:
        etys_file = str(snakemake.input[0])
        gb_network_file = str(snakemake.input[1])
        fes_file = str(snakemake.input[2])

        logger.info(f"ETYS file: {etys_file}")
        logger.info(f"GB network file: {gb_network_file}")
        logger.info(f"FES file: {fes_file}")

        logger.info("Step 1: Processing raw ETYS data")
        df_components, offshore_wf_buses = sort_raw_ETYS_data(etys_file, gb_network_file, logger)

        logger.info("Step 2: Extracting buses from line data")
        df_buses = buses_from_line_data(df_components, logger)

        logger.info("Step 3: Loading GSP location data")
        df_gsp = GSP_locations_from_FES_data(fes_file, logger)

        logger.info("Step 4: Loading Node-to-GSP mapping from GB_network.xlsx")
        node_to_gsp = load_node_to_gsp_mapping(gb_network_file, logger)

        logger.info("Step 4b: Loading substation coordinates lookup")
        substation_coords_file = str(snakemake.input.substation_coords)
        substation_coords = load_substation_coordinates(substation_coords_file, logger)

        logger.info("Step 5: Adding GSP location data to buses")
        df_buses = add_GSP_location_data(df_buses, df_gsp, node_to_gsp, substation_coords, logger)

        # Add offshore classification column
        df_buses['is_offshore'] = df_buses.index.isin(offshore_wf_buses)
        offshore_count = df_buses['is_offshore'].sum()
        logger.info(f"Marked {offshore_count} buses as offshore")

        # Clear coordinates for offshore buses that got onshore GSP positions
        # (Tier 1 gsp_explicit or Tier 3 gsp_prefix). Preserve Tier 2
        # substation_lookup coords which include manually corrected offshore
        # platform positions (e.g. BEIW, EAAW, MOWE).
        offshore_with_gsp_coords = (
            df_buses['is_offshore'] &
            df_buses['lat'].notna() &
            df_buses['coord_source'].isin(['gsp_explicit', 'gsp_prefix'])
        )
        if offshore_with_gsp_coords.sum() > 0:
            logger.info(f"Clearing {offshore_with_gsp_coords.sum()} offshore buses that "
                        f"incorrectly got onshore GSP coordinates")
            df_buses.loc[offshore_with_gsp_coords, ['lat', 'lon']] = np.nan

        # Log preserved offshore coordinates (manual/substation corrections)
        offshore_preserved = (
            df_buses['is_offshore'] &
            df_buses['lat'].notna() &
            (df_buses['coord_source'] == 'substation_lookup')
        )
        if offshore_preserved.sum() > 0:
            logger.info(f"Preserved {offshore_preserved.sum()} offshore buses with "
                        f"manual/substation coordinates")

        # Drop coord_source before export (internal metadata only)
        df_buses = df_buses.drop(columns=['coord_source'])

        # Export to CSV
        components_path = str(snakemake.output[0])
        buses_path = str(snakemake.output[1])

        logger.info(f"Exporting components to: {components_path}")
        df_components.to_csv(components_path, index=True)

        logger.info(f"Exporting buses to: {buses_path}")
        df_buses.to_csv(buses_path, index=True)

        elapsed = time.time() - start_time
        logger.info(f"ETYS DATA PROCESSING COMPLETED in {elapsed:.1f}s")
        logger.info(f"  Components: {len(df_components)} rows")
        logger.info(f"  Buses: {len(df_buses)} rows ({offshore_count} offshore)")

    except Exception as e:
        logger.exception(f"FATAL ERROR in ETYS data processing: {e}")
        raise
