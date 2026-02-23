"""
ETYS File Registry - Maps ETYS publication years to their actual filenames.

NESO changed file naming conventions between years:
  - 2022: "ETYS Appendix B 2022.xlsx"
  - 2023: "ETYS Appendix B 2023.xlsx"
  - 2024: "ETYS 2024 Appendix-B V1.xlsx"  (different pattern)

FES regional data also changed:
  - 2022: No FES regional Excel file, CSV fallback only
  - 2023: "Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb" (binary)
  - 2024: "Regional breakdown of FES24 data.xlsx" (standard xlsx)

This registry centralizes all year-specific filename knowledge so that
Snakemake rules and scripts can select ETYS data by year without
hardcoding filenames.
"""

ETYS_FILES = {
    2022: {
        "appendix_b": "ETYS Appendix B 2022.xlsx",
        "fes_regional": None,
        "fes_gsp_csv": "fes2022_regional_breakdown_gsp_info.csv",
        "gb_network": "GB_network.xlsx",
    },
    2023: {
        "appendix_b": "ETYS Appendix B 2023.xlsx",
        "fes_regional": "Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb",
        "fes_gsp_csv": "fes2023_regional_breakdown_gsp_info.csv",
        "gb_network": "GB_network.xlsx",
    },
    2024: {
        "appendix_b": "ETYS 2024 Appendix-B V1.xlsx",
        # TODO: Implement boundary transfer capability loading from Appendix G
        "appendix_g": "ETYS 2024 Appendix-G.xlsx",  # Boundary transfer capability
        "fes_regional": "Regional breakdown of FES24 data.xlsx",
        "fes_gsp_csv": None,
        "gb_network": "GB_network.xlsx",
    },
}

ETYS_DATA_DIR = "network/ETYS"

# Shared ETYS constants — imported by process_ETYS_data.py, ETYS_network.py, ETYS_upgrades.py
# Voltage level mapping from bus name suffix digit to kV
VOLTAGE_LEVELS = {
    '1': 132,   # 132kV distribution
    '2': 275,   # 275kV transmission
    '3': 33,    # 33kV distribution
    '4': 400,   # 400kV transmission (primary)
    '5': 11,    # 11kV distribution
    '6': 66,    # 66kV distribution
    '7': 20.5,  # 20.5kV (uncommon)
}

# Default electrical parameters by component type (per-unit on 100 MVA base).
# Shared by process_ETYS_data.py and ETYS_upgrades.py to ensure consistency.
ELECTRICAL_DEFAULTS = {
    'line': {
        'r': 0.002,   # Typical 400kV OHL resistance
        'x': 0.02,    # Typical 400kV OHL reactance
        'b': 0.5,     # Line charging susceptance
    },
    'transformer': {
        'r': 0.002,   # Typical transformer resistance
        'x': 0.08,    # Typical transformer reactance
        'b': 0.0,     # Shunt susceptance (negligible for transformers)
    },
    'cable': {
        'r': 0.005,   # Higher resistance for submarine cables
        'x': 0.015,   # Lower reactance for cables vs OHL
        'b': 1.5,     # Much higher charging susceptance for cables
    },
}

# ETYS Appendix B sheet names — base network data
ETYS_BASE_SHEETS = {
    'circuits': ['B-2-1a', 'B-2-1b', 'B-2-1c', 'B-2-1d'],
    'transformers': ['B-3-1a', 'B-3-1b', 'B-3-1c', 'B-3-1d'],
    'hvdc': 'B-5-1',
}

# ETYS Appendix B sheet names — upgrade data (sheet -> operator mapping)
ETYS_UPGRADE_SHEETS = {
    'circuits': {'B-2-2a': 'SHE', 'B-2-2b': 'SPT', 'B-2-2c': 'NGET', 'B-2-2d': 'OFTO'},
    'transformers': {'B-3-2a': 'SHE', 'B-3-2b': 'SPT', 'B-3-2c': 'NGET', 'B-3-2d': 'OFTO'},
    'hvdc': 'B-5-1',
}

# Placeholder/invalid node name patterns (case-insensitive substring match)
INVALID_NODE_PATTERNS = ['converter station', 'offshore', 'onshore', 'nan', 'tbc', 'n/a']

# Default ratings when ETYS data has no rating specified
DEFAULT_RATINGS = {
    'line': 1000,         # MVA, conservative for 400kV OHL
    'transformer': 500,   # MVA, conservative for SGT
    'hvdc': 1000,         # MW, conservative for HVDC link
}

# Substation coordinates file (relative to ETYS data directory)
SUBSTATION_COORDS_FILE = "substation_coordinates.csv"

# GSP regions boundary file for land validation
GSP_REGIONS_FILE = "data/network/GSP/GSP_regions_4326_20250109.geojson"

# ─── Extra wind farm edge ratings ────────────────────────────────────────────
# Ratings for Extra_WF_edges in GB_network.xlsx, derived from real ETYS data.
# These replace the default s_nom=9999 (infinite capacity) to properly
# represent the actual transmission capability of offshore connections.
# Sources: ETYS B-2-1d (OFTO data) and B-2-1c (NGET data).
EXTRA_WF_EDGE_RATINGS = {
    # From OFTO data (B-2-1d) — offshore export cable ratings
    'BOSO11': 106,    # Barrow Offshore WF
    'BRST42': 369,    # East Anglia One
    'ORMO11': 158,    # Ormonde Offshore WF
    'SALL11': 178,    # Sheringham Shoal WF
    'SALL12': 178,    # Sheringham Shoal WF (circuit 2)
    'GUNS11': 158,    # Gunfleet Sands WF
    'LONO4A': 360,    # London Array WF
    'LONO4B': 360,    # London Array WF (circuit 2)
    'BODE41': 277,    # Burbo Bank Extension WF
    'LINO41': 492,    # Lincs WF
    'RORE11': 103,    # Robin Rigg East WF
    'RORW11': 103,    # Robin Rigg West WF
    'THAW11': 155,    # Thanet WF
    'THAW12': 155,    # Thanet WF (circuit 2)
    'WAAO11': 192,    # Walney 1 WF
    'WABO11': 192,    # Walney 2 WF
    # From main ETYS data (B-2-1c) — large onshore connection stubs
    'CREB2A': 1749,   # Creyke Beck (Dogger Bank)
    'CREB2B': 1749,   # Creyke Beck (Dogger Bank, circuit 2)
    'NECT41': 3326,   # Necton (Norfolk projects)
    # Inferred from connected OFTO circuit ratings
    'BEAT4A': 321,    # Beatrice WF (from BEAT41→BLHI4- rating)
    'BEAT4B': 321,    # Beatrice WF (circuit 2)
    'BLHI41': 321,    # Blackhillock WF stub (Beatrice connection)
    'BLHI42': 321,    # Blackhillock WF stub (circuit 2)
    'LINO42': 492,    # Lincs WF (parallel to LINO41)
    'GANW14': 181,    # Galloper WF (from GALO→GANW rating 180.6)
}

# Conservative default for unmapped WF connections (MVA)
DEFAULT_WF_RATING = 200


def get_etys_paths(etys_year, data_path="data"):
    """
    Return a dict of full paths for all ETYS files for the given year.

    Args:
        etys_year: ETYS publication year (2022, 2023, or 2024)
        data_path: Base data directory (default "data")

    Returns:
        dict with keys: appendix_b, fes_regional, fes_gsp_csv, gb_network
        Values are full paths (str) or None if that file doesn't exist for this year.
    """
    etys_year = int(etys_year)
    if etys_year not in ETYS_FILES:
        raise ValueError(
            f"ETYS year {etys_year} not supported. "
            f"Available: {sorted(ETYS_FILES.keys())}"
        )
    files = ETYS_FILES[etys_year]
    paths = {}
    for key, filename in files.items():
        if filename is not None:
            paths[key] = f"{data_path}/{ETYS_DATA_DIR}/{filename}"
        else:
            paths[key] = None
    return paths


def get_etys_input_files(etys_year, data_path="data"):
    """
    Return the ordered list of input files needed for Snakemake rules.

    The list is ordered as:
      [0] ETYS Appendix B file (base network topology + upgrades)
      [1] GB_network.xlsx (coordinates + extra WF/BMU edges)
      [2] FES regional data (GSP locations) - Excel preferred, CSV fallback

    Args:
        etys_year: ETYS publication year (2022, 2023, or 2024)
        data_path: Base data directory (default "data")

    Returns:
        list of 3 file paths (strings)

    Raises:
        ValueError: If etys_year is not supported or no FES data is available
    """
    paths = get_etys_paths(etys_year, data_path)
    inputs = [paths["appendix_b"], paths["gb_network"]]

    # For GSP locations, prefer the full FES regional Excel file, fall back to CSV
    if paths.get("fes_regional"):
        inputs.append(paths["fes_regional"])
    elif paths.get("fes_gsp_csv"):
        inputs.append(paths["fes_gsp_csv"])
    else:
        raise ValueError(
            f"No FES regional data file available for ETYS year {etys_year}. "
            f"Please add either an Excel or CSV file to {data_path}/{ETYS_DATA_DIR}/"
        )

    return inputs
