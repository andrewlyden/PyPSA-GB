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
        # Appendix G is user-submitted GSP winter peak demand ("Appendix G - User
        # Submitted Grid Supply Point GB Winter Peak Demands"), not boundary
        # capability. Boundary capabilities come from the ETYS 2025 boundary chart
        # data: data/network/etys_boundary_capability.csv (see
        # scripts/solve/secured_boundaries.py).
        "appendix_g": "ETYS 2024 Appendix-G.xlsx",  # GSP winter peak demand (unused)
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

# Default electrical parameters by component type, in PER UNIT ON A 100 MVA BASE
# (the ETYS "% on 100 MVA" value / 100), the unit every branch carries during the
# build. They are NOT PyPSA units: branch_units.convert_network_pu100_to_pypsa()
# converts them with the ETYS values, once, at the end of the build (a 0.02 pu100
# line is 32 ohm at 400 kV, 3.5 ohm at 132 kV and 0.22 ohm at 33 kV; the same
# 0.02 pu100 on any voltage is 2e-4 pu on PyPSA's 1 MVA base).
# Shared by process_ETYS_data.py and ETYS_upgrades.py; the policy that decides
# when a default applies is branch_units.resolve_source_impedance_pu100().
#   'coupler': zero-reactance bus sections, tees and short (<= 1 km) Appendix B
#   connections, and inferred same-substation repair couplers. X = 0.01 % on
#   100 MVA, R = B = 0: the values ETYS itself publishes for its 9,999 MVA
#   couplers (e.g. KERG1J-KERG1K, BANN4J-BANN4K).
COUPLER_MAX_LENGTH_KM = 1.0
ELECTRICAL_DEFAULTS = {
    'coupler': {
        'r': 0.0,
        'x': 1.0e-4,
        'b': 0.0,
    },
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

# ─── HVDC place-name aliases ─────────────────────────────────────────────────
# ETYS B-5-1 sometimes uses descriptive place names instead of ETYS bus codes
# for planned HVDC converter stations.  This lookup maps those names to the
# nearest existing ETYS bus so the link can still be created.
#
# Entries are case-insensitive (lowered at lookup time).  Values are ETYS bus
# codes that will be resolved via _resolve_hvdc_bus (so partial codes like
# 'KILS4' are fine — they'll match KILS4T via prefix matching).
HVDC_PLACE_NAME_ALIASES = {
    'ballantrae':       'AUCH2-',   # Planned converter station near Ballantrae → Auchencrosh 275kV (3.7 km)
    'kilmarnock south': 'KILS4T',   # Planned converter station → Kilmarnock South 400kV (3.0 km)
    'shetland hub':     'KERG1J',   # Planned Shetland aggregation hub → Kergord 132kV (existing HVDC terminal)
}

# ─── HVDC links completed with their later-year terminal works ────────────────
# Appendix B can date an HVDC link (B-5-1) before the AC additions that create
# its converter buses; ETYS_upgrades.add_hvdc_enabling_works() then pulls those
# terminal works forward so the link can be built. Only links named here are
# completed, because doing so adds transmission ahead of its Appendix B date.
# Keys are B-5-1 'Interconnector Name' values (case-insensitive).
#   PSDC HVDC Link: Spittal (Banniskirk) -- Longside (Peterhead 2), BANN4R--LOSD4R,
#     2,000 MW, B-5-1 2030; CP30 workbook sheet N1 lists PSDC as required for
#     CP30 with 2030 delivery. Its Longside terminal works are 2032 rows.
# Deliberately NOT listed: 'Western Isles HVDC Link' (LEWI4R--FNLN4R, B-5-1
#   2030). CP30 sheet N1 does not list it and the TEC register dates the Lewis
#   wind farms it connects to Oct--Dec 2030, so the 2030 model leaves it (and
#   those farms, see data/renewables/cp30_named_fleet_exclusions.csv) out. From
#   2032 the ordinary year filter builds it with its own terminals.
HVDC_TERMINAL_ENABLING_WORKS = {
    'PSDC HVDC Link',
}

# ─── GB_network.xlsx repair-edge impedance corrections ──────────────────────
# process_ETYS_data.repair_disconnected_components() reads bridging edges from
# the GB_network.xlsx AC sheet, whose columns are labelled "% on 100 MVA".
# A few OFTO transformer rows there are on the transformer's OWN rating
# instead: their R and X equal the Appendix B twin x rating / 100. Each entry
# below replaces such a row with its Appendix B twin (values in % on 100 MVA).
#   BEAT42-BEAT22 (Beatrice OFTO, 320 MVA 400/220 kV): GB_network R 0.173 %,
#   X 16.03 % = 3.2 x the ETYS 2024 B-3-1d rows BEAT41-BEAT21/BEAT22
#   (R 0.054 %, X 5.009 %, 320 MVA). Uncorrected it was 3.2x its twin in a loop.
# The same own-rating pattern appears in AC-sheet rows for BEAT41-BEAT21,
# BEIW21/22, LINO41/42, RAMW11/12 and TWNE41/42; those rows are never used
# (their ETYS rows are in the base sheets), so they need no entry.
GB_NETWORK_REPAIR_IMPEDANCE_OVERRIDES = {
    ('BEAT42', 'BEAT22'): {
        'R (% on 100 MVA)': 0.054,
        'X (% on 100 MVA)': 5.009,
        'B (% on 100 MVA)': 0.0,
        'source': 'ETYS 2024 Appendix B B-3-1d BEAT41-BEAT22 (TC Beatrice OFTO), 320 MVA',
    },
}

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
