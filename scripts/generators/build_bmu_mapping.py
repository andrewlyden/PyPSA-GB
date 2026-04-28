"""
Build BMU-to-generator name mapping for ELEXON bid/offer integration.

Creates `data/generators/bmus_prepared.csv` which maps ELEXON BMU IDs
(e.g., T_PEMB-11) to PyPSA generator names (e.g., Pembroke).

Data sources:
- ETYS Dir_con_BMUs_to_node sheet:  BMU ID → ETYS Node ID
- STATION_TO_BMU_PREFIX in spatial_utils.py: station name → 4-char prefix
- Solved/finalized network: actual PyPSA generator names

The mapping uses prefix matching: BMU IDs that start with a known station
prefix are mapped to the generator whose name matches that station.

Usage:
    # Standalone (requires a finalized network):
    python scripts/generators/build_bmu_mapping.py \\
        --network resources/network/Historical_2023_etys.nc \\
        --output data/generators/bmus_prepared.csv

    # Or via Snakemake rule (see rules/market.smk)
"""

import pandas as pd
import numpy as np
import logging
import argparse
import os
import re
from pathlib import Path

try:
    import pypsa
except ImportError:
    pypsa = None

from scripts.utilities.logging_config import setup_logging
from scripts.generators.calibrate_renewable_mc import RENEWABLE_PREFIX_TO_STATION

logger = setup_logging("build_bmu_mapping")


# ══════════════════════════════════════════════════════════════════════════════
# STATION NAME → BMU PREFIX MAPPING
# ══════════════════════════════════════════════════════════════════════════════
# Bidirectional: we can go station_name → prefix → matching BMU IDs,
# or BMU ID → prefix → station_name → generator_name.
#
# This is the same mapping used by spatial_utils.apply_etys_bmu_mapping().
# Duplicated here to avoid circular imports and allow standalone use.

STATION_TO_BMU_PREFIX = {
    # Nuclear
    'torness': 'TORN',
    'hunterston': 'HUNT',
    'hinkley': 'HINK',
    'heysham': 'HEYM',
    'hartlepool': 'HART',
    'sizewell': 'SIZE',
    'dungeness': 'DUNG',
    # Gas (CCGT / OCGT)
    'pembroke': 'PEMB',
    'west burton': 'WBUR',
    'drax': 'DRAX',
    'cottam': 'COTT',
    'ratcliffe': 'RATS',
    'didcot': 'DIDC',
    'grain': 'GRAI',
    'seabank': 'SEAB',
    'sutton bridge': 'SUTB',
    'south humber': 'SHBA',
    'saltend': 'SALD',
    'peterhead': 'PEHE',
    'fiddler': 'FIDL',
    'baglan': 'BAGB',
    'staythorpe': 'STAY',
    'keadby': 'KEAD',
    'spalding': 'SPAE',
    'damhead': 'DAMH',
    'cockenzie': 'COCK',
    'longannet': 'LONG',
    'killingholme': 'KILL',
    'little barford': 'LITB',
    'carrington': 'CARR',
    'rocksavage': 'ROCK',
    'immingham': 'IMMM',
    'enfield': 'ENFI',
    'medway': 'MEDW',
    'shoreham': 'SHOR',
    'marchwood': 'MARC',
    'coryton': 'CORY',
    'teesside': 'TEES',
    'wilton': 'WILT',
    'connahs quay': 'CNQP',
    'deeside': 'DEES',
    'langage': 'LANG',
    'sellafield': 'SELL',
    'barking': 'BARK',
    'brigg': 'BRIG',
    'corby': 'CORB',
    'cowes': 'COWE',
    'damhead creek': 'DAMH',
    'great yarmouth': 'GRYA',
    'fellside': 'FELL',
    'indian queens': 'INDQ',
    'rye house': 'RYEH',
    'shotton': 'SHOT',
    'south humber bank': 'SHBA',
    # Offshore wind
    'hornsea': 'HORN',
    'beatrice': 'BEAT',
    'moray': 'MORA',
    'triton knoll': 'TRIK',
    'east anglia': 'EANG',
    'london array': 'LOAD',
    'gwynt y mor': 'GYMR',
    'walney': 'WALN',
    'rampion': 'RAMP',
    'race bank': 'RACB',
    'dudgeon': 'DUDG',
    'greater gabbard': 'GRGB',
    'thanet': 'THAN',
    'sheringham shoal': 'SHER',
    'westermost rough': 'WEST',
    'lincs': 'LINC',
    'humber gateway': 'HUMG',
    'robin rigg': 'ROBI',
    'ormonde': 'ORMO',
    'burbo bank': 'BURB',
    'barrow': 'BARW',
    'gunfleet': 'GUNF',
    'kentish flats': 'KENT',
    'seagreen': 'SGRW',
    'dogger bank': 'DGRB',
    # Pumped storage
    'cruachan': 'CRUA',
    'foyers': 'FOYE',
    'dinorwig': 'DINO',
    'ffestiniog': 'FFES',
    # Large hydro (Scottish) - active BM participants
    'sloy': 'SLOY',
    'errochty': 'ERRO',
    'fasnakyle': 'FASN',
    'glendoe': 'GLND',
    'nant': 'NANT',
    'finlarig': 'FINL',
    # Biomass
    'lynemouth': 'LYNE',
    'ironbridge': 'IRON',
    'steven': 'STEV',
    'uskmouth': 'USKM',
    'rugeley': 'RUGG',
    # Additional stations (ELEXON-registered names)
    'severn power': 'SVRP',
    'vpi': 'HUMR',
    'cottam development centre': 'CDCL',
    'wylfa': 'WYLF',
}

# Reverse: prefix → station name (for BMU → generator lookup)
BMU_PREFIX_TO_STATION = {v: k for k, v in STATION_TO_BMU_PREFIX.items()}

# ELEXON BMU IDs often use different prefixes from ETYS naming.
# Add the actual ELEXON registration prefixes so B1610 data can be matched.
_ELEXON_EXTRA_PREFIXES = {
    # Thermal / gas (alternative ELEXON prefixes)
    'SCCL': 'saltend',                  # Saltend Cogeneration Company Ltd
    'COSO': 'coryton',                  # Coryton Energy Company
    'MEDP': 'medway',                   # Medway Power Ltd
    'MRWD': 'marchwood',                # Marchwood Power Limited
    'LBAR': 'little barford',           # RWE Generation UK plc
    'EECL': 'enfield',                  # Enfield Energy Centre (Uniper)
    'SIZB': 'sizewell',                 # Sizewell B (EDF)
    'LAGA': 'langage',                  # EP Langage Ltd
    'RYHP': 'rye house',               # Rye House Power Station
    'SPLN': 'spalding',                 # Spalding Energy Company Ltd
    'SEEL': 'spalding',                 # Spalding Energy Expansion Ltd
    'SVRP': 'severn power',             # Severn Power Limited
    'DAMC': 'damhead creek',            # Damhead Creek
    'HRTL': 'hartlepool',               # Hartlepool (EDF Nuclear)
    'CDCL': 'cottam development centre', # Cottam Development Centre Ltd
    'HUMR': 'vpi',                       # VPI Immingham (Humber Refinery)
    # Nuclear (Hinkley B and Hunterston B use different prefixes from ETYS)
    'HINB': 'hinkley',                   # Hinkley Point B (EDF) — HINB ≠ HINK
    'HUNB': 'hunterston',                # Hunterston B (EDF) — HUNB ≠ HUNT
    # Offshore wind (ELEXON-registered prefixes differ from ETYS naming)
    'WLNY': 'walney',                    # Walney Extension (Ørsted, ~659 MW) — WLNY ≠ WALN
    'WDNS': 'west of duddon sands',      # West of Duddon Sands (Ørsted, 389 MW)
    'HOWA': 'hornsea',                   # Hornsea One (Ørsted, 1.2 GW) — HOWA ≠ HORN
    'HOWB': 'hornsea',                   # Hornsea Two (Ørsted, 1.32 GW)
    'EAAO': 'east anglia',               # East Anglia One (ScottishPower, 714 MW)
    'RMPN': 'rampion',                   # Rampion (EDF/E.ON, 400 MW) — RMPN ≠ RAMP
    'LNCS': 'lincs',                     # Lincs (Centrica, 270 MW) — LNCS ≠ LINC
    'SHRS': 'sheringham shoal',          # Sheringham Shoal — SHRS ≠ SHER
    'BRBE': 'burbo bank',                # Burbo Bank Extension (Ørsted, 258 MW)
    'HMGT': 'humber gateway',            # Humber Gateway — HMGT ≠ HUMG
    'WTMS': 'westermost rough',          # Westermost Rough — WTMS ≠ WEST
    'LARY': 'london array',              # London Array — LARY ≠ LOAD
    'DDGN': 'dudgeon',                   # Dudgeon — DDGN ≠ DUDG
    'OMND': 'ormonde',                   # Ormonde — OMND ≠ ORMO
    'RCBK': 'race bank',                 # Race Bank — RCBK ≠ RACB
    'THNM': 'thanet',                    # Thanet (alternative)
    'THNT': 'thanet',                    # Thanet — THNT ≠ THAN
    'GAOF': 'galloper',                  # Galloper (Innogy, 353 MW)
    'NNGA': 'neart na gaoithe',          # Neart na Gaoithe (EDF, 448 MW)
    'MOWE': 'moray east',                # Moray East (Iberdrola, 900 MW) — MOWE ≠ MORA
    'MOWW': 'moray west',                # Moray West (Corio, 860 MW)
    'DBBW': 'dogger bank b',             # Dogger Bank B (Equinor, 1.2 GW)
    'DBAW': 'dogger bank a',             # Dogger Bank A (Equinor, 1.2 GW)
    'TKNW': 'triton knoll',              # Triton Knoll West — TKNW ≠ TRIK
    'TKNE': 'triton knoll',              # Triton Knoll East
    'SGRW': 'seagreen',                  # Seagreen (already in main dict, belt+braces)
    'VKNG': 'viking energy',             # Viking Energy (SSE Renewables, 443 MW)
    # Onshore wind (large Scottish / English sites)
    'STLG': 'stronelairg',               # Stronelairg (SSE, 228 MW)
    'DORE': 'dorenell',                  # Dorenell (RES, 177 MW)
    'GRIF': 'griffin',                   # Griffin (Vattenfall, 156 MW)
    'KLGL': 'kilgallioch',               # Kilgallioch (SSE, 239 MW)
    'CRYR': 'crystal rig',               # Crystal Rig (Fred. Olsen, 213 MW)
    'BDCH': 'bad a cheo',                # Bad a Cheo (SSE, 245 MW)
    'PNYC': 'pen y cymoedd',             # Pen y Cymoedd (Vattenfall, 228 MW)
    'AKGL': 'aikengall',                 # Aikengall II (EDF, 227 MW)
    'ABRB': 'aberdeen bay',              # Vattenfall Aberdeen Bay (93 MW)
    'ACHR': 'achrua ch',                 # AChruach (42 MW)
    'AFTO': 'afton',                     # Afton (50 MW)
    # Large onshore wind (alternative/ELEXON-specific prefixes)
    'WHIL': 'whitelee',                  # Whitelee (SP Energy, 539 MW) — T_WHILW
    'CLDC': 'clyde',                     # Clyde Central (EDF, 200 MW)
    'CLDN': 'clyde',                     # Clyde North (EDF, 200 MW)
    'CLDS': 'clyde',                     # Clyde South (EDF, 150 MW)
    'RREW': 'robin rigg',                # Robin Rigg East — RREW ≠ ROBI
    'RRWW': 'robin rigg',                # Robin Rigg West — RRWW ≠ ROBI
    'GNFS': 'gunfleet sands',            # Gunfleet Sands — GNFS ≠ GUNF
    'ARCH': 'arecleoch',                 # Arecleoch (114 MW)
    'FALG': 'fallago',                   # Fallago Rig (144 MW)
    'HRST': 'harestanes',                # Harestanes (142 MW)
    'HADH': 'hadyard hill',              # Hadyard Hill (130 MW)
    'BEIN': 'beinneun',                  # Beinneun (109 MW)
    'BHLA': 'bhlaraidh',                 # Bhlaraidh (108 MW)
    'BOWL': 'barrow',                    # Barrow Offshore — BOWL ≠ BARW
    'BLLA': 'blyth',                     # Blyth Offshore (118 MW)
    'STRN': 'strathy north',             # Strathy North (70 MW)
    'DRSL': 'dersalloch',                # Dersalloch (71 MW)
    'LCLT': 'lochluichart',              # Lochluichart (69 MW)
    # Additional CCGT/thermal (expanded mapping from ELEXON data)
    'WBUP': 'west burton',               # West Burton CCGT P-units (T_WBUPS-1..4)
    'WBUG': 'west burton',               # West Burton GT units (T_WBUGT-1..4)
    'GRCH': 'grain',                     # Grain CHP (InterGen, 1517 MW)
    'GRYA': 'great yarmouth',            # Great Yarmouth CCGT (RWE, 420 MW)
    'FELL': 'fellside',                  # Fellside CHP (Sellafield, 155 MW)
    'BAGE': 'baglan',                    # Baglan Bay (GE, 520 MW) — BAGE vs BAGB
    'FAWL': 'fawley',                    # Fawley (oil/gas)
    'RUGP': 'rugeley',                   # Rugeley P-units
    'SFGS': 'south ferriby',             # South Ferriby gas storage
    'GANW': 'galloway',                  # Galloway hydro
    'FARR': 'farr',                      # Farr wind farm (92 MW)
    'CRDE': 'corriemoillie',             # Corriemoillie wind farm
    'BPGR': 'baglan generation',         # Baglan Generation (alt prefix)
}
BMU_PREFIX_TO_STATION.update(_ELEXON_EXTRA_PREFIXES)

# Reuse the richer renewable station dictionary built for renewable MC
# calibration so BMU mapping and pricing use the same station vocabulary.
for _prefix, _station in RENEWABLE_PREFIX_TO_STATION.items():
    BMU_PREFIX_TO_STATION.setdefault(_prefix, _station)


DEFAULT_POWER_STATION_DICTIONARY_PATH = (
    "data/generators/power_station_dictionary_bmu_crosswalk.csv"
)

_PSD_COLUMN_ALIASES = {
    "bmu_id": [
        "bmu_id",
        "BMU ID",
        "Settlement BMU ID",
        "settlement_bmu_id",
        "nationalGridBmUnit",
    ],
    "station_name": [
        "station_name",
        "Station Name",
        "station",
        "Common Name",
        "common_name",
        "bmUnitName",
    ],
    "generator_name": [
        "generator_name",
        "Generator Name",
        "pypsa_generator_name",
        "PyPSA Generator Name",
    ],
    "fuel": [
        "fuel",
        "Fuel",
        "fuel_type",
        "Fuel Type",
    ],
    "plant_type": [
        "plant_type",
        "Plant Type",
        "type",
        "Type",
    ],
    "installed_capacity_mw": [
        "installed_capacity_mw",
        "Installed Capacity (MW)",
        "capacity_mw",
        "generationCapacity",
    ],
    "node_id": [
        "node_id",
        "Node Id",
        "node",
        "bus",
    ],
}


def _normalise_station_name(value: str) -> str:
    """Normalise station names so cross-source alias matching is stable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""

    # Remove common formatting noise seen across DUKES / TEC / PSD / PyPSA names.
    text = text.replace("\xa0", " ")
    text = text.replace("_", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("*", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(power station|wind farm|offshore|onshore|extension|demonstration site)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first present column from a list of aliases."""
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _load_power_station_dictionary(
    dictionary_path: str | None,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load an optional Power Station Dictionary BMU crosswalk.

    Expected minimum fields:
      - BMU ID / settlement_bmu_id
      - Station Name / common_name

    Optional fields improve disambiguation:
      - generator_name
      - fuel / plant_type
      - installed_capacity_mw
      - node_id
    """
    if not dictionary_path:
        return pd.DataFrame()

    path = Path(dictionary_path)
    if not path.exists():
        logger.info(
            f"Power Station Dictionary crosswalk not found at {path} "
            "(continuing with ETYS + prefix heuristics)"
        )
        return pd.DataFrame()

    df = pd.read_csv(path)
    rename_map = {}
    for target, aliases in _PSD_COLUMN_ALIASES.items():
        source = _resolve_column(df, aliases)
        if source:
            rename_map[source] = target

    df = df.rename(columns=rename_map)
    required = {"bmu_id", "station_name"}
    missing = required.difference(df.columns)
    if missing:
        logger.warning(
            f"Power Station Dictionary crosswalk missing required columns {sorted(missing)}; "
            "ignoring file"
        )
        return pd.DataFrame()

    keep_cols = [
        c for c in [
            "bmu_id", "station_name", "generator_name", "fuel",
            "plant_type", "installed_capacity_mw", "node_id",
        ] if c in df.columns
    ]
    df = df[keep_cols].copy()
    df["bmu_id"] = df["bmu_id"].astype(str).str.strip()
    df["station_name"] = df["station_name"].astype(str).str.strip()
    df["station_name_norm"] = df["station_name"].map(_normalise_station_name)
    if "generator_name" in df.columns:
        df["generator_name_norm"] = df["generator_name"].map(_normalise_station_name)
    if "installed_capacity_mw" in df.columns:
        df["installed_capacity_mw"] = pd.to_numeric(
            df["installed_capacity_mw"], errors="coerce"
        )

    df = df.dropna(subset=["bmu_id"])
    df = df[df["bmu_id"] != ""]
    df = df.drop_duplicates(subset=["bmu_id"], keep="first")

    logger.info(
        f"Loaded Power Station Dictionary crosswalk: {len(df)} BMU rows from {path}"
    )
    return df


def _fuel_to_carrier_candidates(
    fuel: str | None,
    plant_type: str | None,
) -> set[str]:
    """Map dictionary fuel/type hints onto PyPSA carriers."""
    text = " ".join(
        [str(x).lower() for x in [fuel, plant_type] if x is not None and not pd.isna(x)]
    )
    if not text:
        return set()

    carriers = set()
    if "nuclear" in text:
        carriers.add("nuclear")
    if "combined cycle" in text or "ccgt" in text:
        carriers.add("CCGT")
    if "ocgt" in text or "gas oil" in text:
        carriers.update({"OCGT", "oil"})
    if "coal" in text:
        carriers.add("coal")
    if "oil" in text:
        carriers.add("oil")
    if "biomass" in text or "bioenergy" in text:
        carriers.add("biomass")
    if "waste" in text:
        carriers.add("waste_to_energy")
    if "hydro" in text and "pumped" in text:
        carriers.update({"Pumped Storage Hydroelectricity", "pumped_hydro"})
    elif "hydro" in text:
        carriers.update({"large_hydro", "small_hydro"})
    if "offshore" in text:
        carriers.add("wind_offshore")
    if "onshore" in text:
        carriers.add("wind_onshore")
    if "wind" in text and "offshore" not in text and "onshore" not in text:
        carriers.update({"wind_offshore", "wind_onshore"})
    if "solar" in text or "photo" in text or "pv" in text:
        carriers.add("solar_pv")

    return carriers


def _build_component_lookup(network) -> tuple[pd.DataFrame, dict]:
    """Build a lookup table of generators/storage units for station matching."""
    rows = []
    if network is None:
        return pd.DataFrame(), {}

    for component_name, component_df, component_type in [
        ("Generator", network.generators, "generator"),
        ("StorageUnit", network.storage_units, "storage"),
    ]:
        if component_df is None or component_df.empty:
            continue

        for idx, row in component_df.iterrows():
            rows.append(
                {
                    "name": idx,
                    "name_norm": _normalise_station_name(idx),
                    "carrier": row.get("carrier", ""),
                    "p_nom": pd.to_numeric(row.get("p_nom", np.nan), errors="coerce"),
                    "bus": row.get("bus", ""),
                    "component_type": component_type,
                    "data_source": row.get("data_source", ""),
                }
            )

    lookup_df = pd.DataFrame(rows)
    by_name = {}
    if not lookup_df.empty:
        by_name = dict(zip(lookup_df["name"], lookup_df["carrier"]))
    return lookup_df, by_name


def _select_generator_match(
    station_name: str,
    lookup_df: pd.DataFrame,
    explicit_generator_name: str | None = None,
    expected_carriers: set[str] | None = None,
    expected_bus: str | None = None,
    expected_capacity_mw: float | None = None,
) -> tuple[str | None, str]:
    """Choose the best network component for a station using scored matching."""
    if lookup_df.empty:
        return None, "unmatched"

    station_norm = _normalise_station_name(station_name)
    if not station_norm:
        return None, "unmatched"

    if explicit_generator_name:
        explicit_norm = _normalise_station_name(explicit_generator_name)
        explicit = lookup_df[lookup_df["name_norm"] == explicit_norm]
        if not explicit.empty:
            return explicit.iloc[0]["name"], "power_station_dictionary"

    has_dictionary_metadata = bool(
        expected_carriers
        or expected_bus
        or (expected_capacity_mw is not None and not pd.isna(expected_capacity_mw))
    )
    pseudo_carriers = {"load_shedding", "EU_import", "embedded_solar", "embedded_wind"}
    named_lookup_df = lookup_df[
        ~lookup_df["carrier"].isin(pseudo_carriers)
        & ~lookup_df["name"].astype(str).str.startswith("gen_")
        & ~lookup_df["name"].astype(str).str.contains("__agg", regex=False)
    ].copy()
    if named_lookup_df.empty:
        named_lookup_df = lookup_df.copy()

    # Stay conservative when we only have a prefix-derived station name. This
    # preserves the old behaviour unless the Power Station Dictionary provides
    # extra metadata that justifies a scored disambiguation.
    if not has_dictionary_metadata:
        exact = named_lookup_df[named_lookup_df["name_norm"] == station_norm]
        if not exact.empty:
            return exact.iloc[0]["name"], "direct"

        candidates = named_lookup_df[
            named_lookup_df["name_norm"].map(
                lambda name: station_norm in name.replace(" ", "")
                if isinstance(name, str) else False
            )
        ]
        if candidates.empty:
            candidates = named_lookup_df[
                named_lookup_df["name_norm"].map(
                    lambda name: station_norm.replace(" ", "") in name.replace(" ", "")
                    if isinstance(name, str) else False
                )
            ]

        if len(candidates) == 1:
            return candidates.iloc[0]["name"], "partial"
        if len(candidates) > 1:
            candidates = candidates.assign(
                _length_delta=(candidates["name_norm"].str.len() - len(station_norm)).abs()
            ).sort_values(["_length_delta", "p_nom"], ascending=[True, False])
            return candidates.iloc[0]["name"], "partial"
        return None, "unmatched"

    candidates = lookup_df.copy()
    candidates["score"] = 0.0

    exact = candidates["name_norm"] == station_norm
    candidates.loc[exact, "score"] += 100

    contains = candidates["name_norm"].str.contains(
        re.escape(station_norm), na=False
    ) | candidates["name_norm"].map(lambda v: station_norm in v if isinstance(v, str) else False)
    candidates.loc[contains, "score"] += 60

    station_tokens = set(station_norm.split())
    if station_tokens:
        overlap = candidates["name_norm"].map(
            lambda name: len(station_tokens.intersection(set(name.split())))
            if isinstance(name, str) else 0
        )
        candidates["score"] += overlap * 8

    if expected_carriers:
        candidates.loc[candidates["carrier"].isin(expected_carriers), "score"] += 80
        candidates.loc[~candidates["carrier"].isin(expected_carriers), "score"] -= 25

    if expected_bus:
        candidates.loc[candidates["bus"] == expected_bus, "score"] += 45

    if expected_capacity_mw is not None and not pd.isna(expected_capacity_mw):
        delta = (candidates["p_nom"] - expected_capacity_mw).abs()
        candidates.loc[delta <= max(5.0, expected_capacity_mw * 0.05), "score"] += 45
        candidates.loc[
            (delta > max(5.0, expected_capacity_mw * 0.05))
            & (delta <= max(20.0, expected_capacity_mw * 0.25)),
            "score"
        ] += 25
        candidates.loc[delta > max(50.0, expected_capacity_mw * 0.75), "score"] -= 20

    # De-prioritise pseudo-generators when a real plant is available.
    candidates.loc[candidates["carrier"].isin(pseudo_carriers), "score"] -= 100

    candidates = candidates.sort_values(
        ["score", "component_type", "p_nom"],
        ascending=[False, True, False],
    )
    best = candidates.iloc[0]
    if best["score"] <= 0:
        return None, "unmatched"

    match_method = "direct" if best["name_norm"] == station_norm else "partial"
    return best["name"], match_method


def build_bmu_mapping(
    network_path: str = None,
    network=None,
    etys_path: str = "data/network/ETYS/GB_network.xlsx",
    elexon_offers_path: str = None,
    power_station_dictionary_path: str = DEFAULT_POWER_STATION_DICTIONARY_PATH,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Build a mapping from ELEXON BMU IDs to PyPSA generator names.

    Strategy:
    1. Load all T_ (transmission) BMU IDs from ETYS Dir_con_BMUs_to_node
    2. Extract the 4-char station prefix from each BMU ID
    3. Look up station name from prefix via BMU_PREFIX_TO_STATION
    4. Fuzzy-match station name against actual PyPSA generator names
    5. (Optional) Supplementary pass: scan ELEXON offer CSV columns for
       T_ BMU IDs not covered by ETYS, apply the same prefix matching
    6. Output: bmu_id, generator_name, station_name, carrier, match_method

    Parameters
    ----------
    network_path : str, optional
        Path to a PyPSA network .nc file to get actual generator names.
    network : pypsa.Network, optional
        Pre-loaded network (used instead of network_path if given).
    etys_path : str
        Path to GB_network.xlsx with Dir_con_BMUs_to_node sheet.
    elexon_offers_path : str, optional
        Path to ELEXON offers CSV. If provided, column headers are scanned
        as supplementary BMU IDs (catches BMUs not in ETYS register).
    power_station_dictionary_path : str, optional
        Optional path to a local Power Station Dictionary BMU crosswalk CSV.
        If present, explicit BMU rows are used ahead of prefix heuristics and
        its fuel / plant type metadata is used to disambiguate sites.
    logger : Logger

    Returns
    -------
    DataFrame with columns: bmu_id, generator_name, station_name, carrier, match_method
    """
    # Load network
    if network is None and network_path is not None:
        network = pypsa.Network(network_path)
    if network is None:
        logger.warning("No network provided — building prefix-only mapping (no generator name verification)")

    # Load ETYS BMU data
    if not os.path.exists(etys_path):
        raise FileNotFoundError(f"ETYS network file not found: {etys_path}")

    bmu_df = pd.read_excel(etys_path, sheet_name='Dir_con_BMUs_to_node')
    logger.info(f"Loaded {len(bmu_df)} BMU-to-node entries from ETYS")

    # Filter to transmission-connected BMUs (T_ prefix = generation)
    t_bmus = bmu_df[bmu_df['BM Unit Id'].str.startswith('T_', na=False)].copy()
    # Also include M_ (embedded medium) for pumped storage etc
    m_bmus = bmu_df[bmu_df['BM Unit Id'].str.startswith('M_', na=False)].copy()
    all_bmus = pd.concat([t_bmus, m_bmus], ignore_index=True)
    logger.info(f"Filtered to {len(all_bmus)} generation BMUs (T_ and M_ prefix)")

    # Exclude non-generation BMUs: ELEXON registers automatic demand-intertrip /
    # protection units with the same T_ prefix but a trailing "-D" suffix
    # (e.g., T_HUNB-D, T_TORN-D, T_HINB-D, T_HEYM2-D, T_SIZB-D).  These units
    # have extreme "protection" bid prices (e.g., -£49,999.5/MWh) that inflate
    # the averaged ELEXON bid price for their parent generator by an order of
    # magnitude.  Real generation BMUs use numeric or alphanumeric suffixes.
    n_before = len(all_bmus)
    all_bmus = all_bmus[~all_bmus['BM Unit Id'].str.endswith('-D', na=False)]
    n_removed = n_before - len(all_bmus)
    if n_removed:
        logger.info(f"Removed {n_removed} intertrip/demand BMUs (-D suffix): {len(all_bmus)} remaining")

    # Get unique BMU IDs
    unique_bmus = all_bmus['BM Unit Id'].str.strip().unique()
    bmu_node_lookup = dict(
        zip(
            all_bmus["BM Unit Id"].astype(str).str.strip(),
            all_bmus["Node Id"].astype(str).str.strip(),
        )
    )

    component_lookup_df, gen_carrier_lookup = _build_component_lookup(network)
    psd_df = _load_power_station_dictionary(power_station_dictionary_path, logger)
    psd_by_bmu = {}
    if not psd_df.empty:
        psd_by_bmu = psd_df.set_index("bmu_id").to_dict("index")

    results = []

    for bmu_id in unique_bmus:
        bmu_clean = bmu_id.strip()
        psd_row = psd_by_bmu.get(bmu_clean, {})
        expected_bus = (
            psd_row.get("node_id")
            or (bmu_node_lookup.get(bmu_clean) if psd_row else None)
            or None
        )
        expected_carriers = _fuel_to_carrier_candidates(
            psd_row.get("fuel"), psd_row.get("plant_type")
        )
        expected_capacity = psd_row.get("installed_capacity_mw")

        station_name = psd_row.get("station_name")
        if not station_name:
            # Extract prefix: strip T_/E_/M_ prefix, then take first 4 chars before - or digit
            core = bmu_clean
            for prefix_strip in ['T_', 'E_', 'M_', '2_']:
                if core.startswith(prefix_strip):
                    core = core[len(prefix_strip):]
                    break

            for prefix_len in [4, 5, 3]:
                candidate_prefix = core[:prefix_len].upper()
                if candidate_prefix in BMU_PREFIX_TO_STATION:
                    station_name = BMU_PREFIX_TO_STATION[candidate_prefix]
                    break

        if station_name is None:
            # No station match — skip (will fall back to derived pricing)
            continue

        # Find matching generator name in the network
        generator_name = None
        carrier = ''
        match_method = 'prefix_only'

        if not component_lookup_df.empty:
            generator_name, match_method = _select_generator_match(
                station_name=station_name,
                lookup_df=component_lookup_df,
                explicit_generator_name=psd_row.get("generator_name"),
                expected_carriers=expected_carriers,
                expected_bus=expected_bus,
                expected_capacity_mw=expected_capacity,
            )
            if generator_name:
                carrier = gen_carrier_lookup.get(generator_name, '')
                if psd_row and match_method in {"direct", "partial"}:
                    match_method = f"power_station_dictionary_{match_method}"
        else:
            # No network — use station name as-is (capitalised)
            generator_name = station_name.title()
            match_method = 'no_network'

        if generator_name:
            results.append({
                'bmu_id': bmu_clean,
                'generator_name': generator_name,
                'station_name': station_name,
                'carrier': carrier,
                'match_method': match_method,
            })

    df = pd.DataFrame(results)

    # ── Supplementary pass: scan ELEXON offer column headers ──────────────
    # Catches T_ BMU IDs that are registered in ELEXON but missing from the
    # ETYS Dir_con_BMUs_to_node sheet (e.g., CHP units, newer registrations).
    if elexon_offers_path and os.path.exists(elexon_offers_path):
        elexon_cols = pd.read_csv(elexon_offers_path, index_col=0, nrows=0).columns.tolist()
        already_mapped = set(df['bmu_id'].tolist()) if not df.empty else set()
        elexon_t_bmus = [
            c for c in elexon_cols
            if c.startswith(('T_', 'E_', 'M_')) and c not in already_mapped
        ]
        # Filter out intertrip/demand BMUs (-D suffix)
        elexon_t_bmus = [b for b in elexon_t_bmus if not b.endswith('-D')]
        logger.info(
            f"Supplementary ELEXON pass: {len(elexon_t_bmus)} unmapped "
            "T_/E_/M_ BMU columns to scan"
        )

        elexon_results = []
        for bmu_id in elexon_t_bmus:
            psd_row = psd_by_bmu.get(bmu_id, {})
            expected_carriers = _fuel_to_carrier_candidates(
                psd_row.get("fuel"), psd_row.get("plant_type")
            )
            expected_capacity = psd_row.get("installed_capacity_mw")
            station_name = psd_row.get("station_name")
            if not station_name:
                core = bmu_id[2:]  # Strip T_/E_/M_
                for prefix_len in [4, 5, 3]:
                    candidate_prefix = core[:prefix_len].upper()
                    if candidate_prefix in BMU_PREFIX_TO_STATION:
                        station_name = BMU_PREFIX_TO_STATION[candidate_prefix]
                        break
            if station_name is None:
                continue

            generator_name = None
            carrier = ''
            match_method = 'elexon_supplementary'

            if not component_lookup_df.empty:
                generator_name, generator_match_method = _select_generator_match(
                    station_name=station_name,
                    lookup_df=component_lookup_df,
                    explicit_generator_name=psd_row.get("generator_name"),
                    expected_carriers=expected_carriers,
                    expected_bus=(
                        psd_row.get("node_id")
                        or (bmu_node_lookup.get(bmu_id) if psd_row else None)
                    ),
                    expected_capacity_mw=expected_capacity,
                )
                if generator_name:
                    carrier = gen_carrier_lookup.get(generator_name, '')
                    if psd_row:
                        match_method = f"power_station_dictionary_{generator_match_method}"
                    else:
                        match_method = f"elexon_supplementary_{generator_match_method}"

            if generator_name:
                elexon_results.append({
                    'bmu_id': bmu_id,
                    'generator_name': generator_name,
                    'station_name': station_name,
                    'carrier': carrier,
                    'match_method': match_method,
                })

        if elexon_results:
            elexon_df = pd.DataFrame(elexon_results)
            df = pd.concat([df, elexon_df], ignore_index=True)
            n_new_gens = elexon_df['generator_name'].nunique()
            logger.info(f"ELEXON supplementary: added {len(elexon_results)} BMU entries "
                        f"({n_new_gens} unique generators)")

    # Deduplicate: multiple BMU IDs mapping to same generator is fine (many-to-one)
    # But log the counts
    if not df.empty:
        n_bmus = df['bmu_id'].nunique()
        n_gens = df['generator_name'].nunique()
        logger.info(f"BMU mapping built: {n_bmus} BMU IDs → {n_gens} generator names")
        logger.info(f"Match methods: {df['match_method'].value_counts().to_dict()}")
    else:
        logger.warning("No BMU mappings could be built!")

    return df


def main():
    """Command-line / Snakemake entry point."""
    global logger

    # Check if running under Snakemake
    try:
        _ = snakemake
        is_snakemake = True
    except NameError:
        is_snakemake = False

    if is_snakemake:
        log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "build_bmu_mapping"
        logger = setup_logging(log_path)

        network_path = snakemake.input.network
        output_path = snakemake.output.bmu_mapping
        etys_path = snakemake.params.get('etys_path', 'data/network/ETYS/GB_network.xlsx')
        elexon_offers_path = snakemake.params.get('elexon_offers_path', None)
        power_station_dictionary_path = snakemake.params.get(
            'power_station_dictionary_path',
            DEFAULT_POWER_STATION_DICTIONARY_PATH,
        )

        logger.info("=" * 80)
        logger.info("BUILDING BMU-TO-GENERATOR MAPPING")
        logger.info("=" * 80)

        df = build_bmu_mapping(
            network_path=network_path,
            etys_path=etys_path,
            elexon_offers_path=elexon_offers_path,
            power_station_dictionary_path=power_station_dictionary_path,
            logger=logger,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved BMU mapping: {output_path} ({len(df)} entries)")

    else:
        parser = argparse.ArgumentParser(description="Build BMU-to-generator mapping")
        parser.add_argument("--network", type=str, default=None,
                            help="Path to PyPSA network .nc file (optional)")
        parser.add_argument("--etys", type=str, default="data/network/ETYS/GB_network.xlsx",
                            help="Path to GB_network.xlsx")
        parser.add_argument("--output", type=str, default="data/generators/bmus_prepared.csv",
                            help="Output CSV path")
        parser.add_argument("--elexon-offers", type=str, default=None,
                            help="Path to ELEXON offers CSV (optional, for supplementary matching)")
        parser.add_argument(
            "--power-station-dictionary",
            type=str,
            default=DEFAULT_POWER_STATION_DICTIONARY_PATH,
            help="Optional Power Station Dictionary BMU crosswalk CSV",
        )
        args = parser.parse_args()

        logger = setup_logging("build_bmu_mapping")
        logger.info("=" * 80)
        logger.info("BUILDING BMU-TO-GENERATOR MAPPING (standalone)")
        logger.info("=" * 80)

        df = build_bmu_mapping(
            network_path=args.network,
            etys_path=args.etys,
            elexon_offers_path=args.elexon_offers,
            power_station_dictionary_path=args.power_station_dictionary,
            logger=logger,
        )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"Saved: {args.output} ({len(df)} entries)")


if __name__ == "__main__":
    main()
