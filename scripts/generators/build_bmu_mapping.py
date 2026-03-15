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
from pathlib import Path

try:
    import pypsa
except ImportError:
    pypsa = None

from scripts.utilities.logging_config import setup_logging

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
    # Biomass
    'lynemouth': 'LYNE',
    'ironbridge': 'IRON',
    'steven': 'STEV',
    # Additional stations (ELEXON-registered names)
    'severn power': 'SVRP',
    'vpi': 'HUMR',
    'cottam development centre': 'CDCL',
}

# Reverse: prefix → station name (for BMU → generator lookup)
BMU_PREFIX_TO_STATION = {v: k for k, v in STATION_TO_BMU_PREFIX.items()}

# ELEXON BMU IDs often use different prefixes from ETYS naming.
# Add the actual ELEXON registration prefixes so B1610 data can be matched.
_ELEXON_EXTRA_PREFIXES = {
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
}
BMU_PREFIX_TO_STATION.update(_ELEXON_EXTRA_PREFIXES)


def build_bmu_mapping(
    network_path: str = None,
    network=None,
    etys_path: str = "data/network/ETYS/GB_network.xlsx",
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Build a mapping from ELEXON BMU IDs to PyPSA generator names.

    Strategy:
    1. Load all T_ (transmission) BMU IDs from ETYS Dir_con_BMUs_to_node
    2. Extract the 4-char station prefix from each BMU ID
    3. Look up station name from prefix via BMU_PREFIX_TO_STATION
    4. Fuzzy-match station name against actual PyPSA generator names
    5. Output: bmu_id, generator_name, station_name, carrier, match_method

    Parameters
    ----------
    network_path : str, optional
        Path to a PyPSA network .nc file to get actual generator names.
    network : pypsa.Network, optional
        Pre-loaded network (used instead of network_path if given).
    etys_path : str
        Path to GB_network.xlsx with Dir_con_BMUs_to_node sheet.
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

    # Get unique BMU IDs
    unique_bmus = all_bmus['BM Unit Id'].str.strip().unique()

    # Build generator name lookup from network
    gen_name_lookup = {}  # lowercase station → actual generator name
    gen_carrier_lookup = {}  # generator name → carrier
    if network is not None:
        for gen_name in network.generators.index:
            gen_name_lookup[gen_name.lower().replace('_', ' ')] = gen_name
            gen_name_lookup[gen_name.lower()] = gen_name
            if hasattr(network.generators, 'carrier'):
                gen_carrier_lookup[gen_name] = network.generators.loc[gen_name, 'carrier']

        # Also add storage units (pumped hydro has BMUs)
        for su_name in network.storage_units.index:
            gen_name_lookup[su_name.lower().replace('_', ' ')] = su_name
            gen_name_lookup[su_name.lower()] = su_name

    results = []
    matched_count = 0

    for bmu_id in unique_bmus:
        bmu_clean = bmu_id.strip()

        # Extract prefix: strip T_/M_ prefix, then take first 4 chars before - or digit
        core = bmu_clean
        for prefix_strip in ['T_', 'M_']:
            if core.startswith(prefix_strip):
                core = core[len(prefix_strip):]
                break

        # Try progressively shorter prefixes (4, 5, 3 chars)
        station_name = None
        match_prefix = None
        for prefix_len in [4, 5, 3]:
            candidate_prefix = core[:prefix_len].upper()
            if candidate_prefix in BMU_PREFIX_TO_STATION:
                station_name = BMU_PREFIX_TO_STATION[candidate_prefix]
                match_prefix = candidate_prefix
                break

        if station_name is None:
            # No station match — skip (will fall back to derived pricing)
            continue

        # Find matching generator name in the network
        generator_name = None
        carrier = ''
        match_method = 'prefix_only'

        if gen_name_lookup:
            # Try direct station name match
            if station_name in gen_name_lookup:
                generator_name = gen_name_lookup[station_name]
                match_method = 'direct'
            else:
                # Try partial match: any generator whose name contains the station name
                for key, gen_name in gen_name_lookup.items():
                    if station_name.replace(' ', '') in key.replace(' ', ''):
                        generator_name = gen_name
                        match_method = 'partial'
                        break

            if generator_name:
                carrier = gen_carrier_lookup.get(generator_name, '')
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
            matched_count += 1

    df = pd.DataFrame(results)

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

        logger.info("=" * 80)
        logger.info("BUILDING BMU-TO-GENERATOR MAPPING")
        logger.info("=" * 80)

        df = build_bmu_mapping(
            network_path=network_path,
            etys_path=etys_path,
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
        args = parser.parse_args()

        logger = setup_logging("build_bmu_mapping")
        logger.info("=" * 80)
        logger.info("BUILDING BMU-TO-GENERATOR MAPPING (standalone)")
        logger.info("=" * 80)

        df = build_bmu_mapping(
            network_path=args.network,
            etys_path=args.etys,
            logger=logger,
        )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info(f"Saved: {args.output} ({len(df)} entries)")


if __name__ == "__main__":
    main()
