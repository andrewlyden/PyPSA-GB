"""
Calibrate renewable marginal costs using ELEXON bid data.

One-time prerun script (NOT part of the per-scenario Snakemake DAG).
Estimates per-generator empirical marginal costs for wind and solar generators
by analysing their Bid-Offer Data (BOD) bid prices from the ELEXON BMRS API.

Methodology (inspired by GBPower build_roc_values.py):
    Renewable generators in the Balancing Mechanism submit bid prices that
    reflect their willingness-to-pay to keep running (i.e., their opportunity
    cost of curtailment):
      - ROC generators bid negative (~-£50 to -£60/MWh) because curtailment
        loses their ROC subsidy income.
      - CfD generators bid near £0 because CfD difference payments top them
        up to strike price regardless of curtailment.
      - Merchant generators bid near £0 (no subsidy income to protect).

    The median bid price thus directly reveals each generator's effective MC.

    We also infer the support type (CfD / ROC / merchant) from the bid price
    distribution: strongly negative bids → ROC; near-zero bids → CfD or merchant.

Output:
    data/market/renewable_empirical_mc.csv
    Columns: generator, carrier, empirical_mc, support_type_inferred, n_bmus,
             median_bid, mean_bid, std_bid

Usage:
    python scripts/generators/calibrate_renewable_mc.py \\
        --start-date 2023-01-01 --end-date 2023-12-31 \\
        --output data/market/renewable_empirical_mc.csv

    Or with pre-downloaded BOD data:
    python scripts/generators/calibrate_renewable_mc.py \\
        --bids-file resources/market/elexon/elexon_bids.csv \\
        --output data/market/renewable_empirical_mc.csv
"""

import pandas as pd
import numpy as np
import argparse
import logging
import time
from pathlib import Path
from datetime import timedelta
from typing import Optional

from scripts.utilities.logging_config import setup_logging

logger = setup_logging("calibrate_renewable_mc")


# ══════════════════════════════════════════════════════════════════════════════
# RENEWABLE BMU PREFIX MAPPING
# ══════════════════════════════════════════════════════════════════════════════
# Maps station names (lowercase) to their nationalGridBmUnit 4-char prefix.
# These are the NGBMU format used by the ELEXON BOD stream endpoint, NOT the
# old elexonBmUnit format (T_HORN-1 etc.).
#
# The primary discovery mechanism is now fetch_renewable_bmu_registry() which
# queries the ELEXON /reference/bmunits/all endpoint and filters by fuelType.
# This static dict serves as a fallback for offline use and station name mapping.
#
# Source: ELEXON BMU registration API (nationalGridBmUnit field), June 2024.

RENEWABLE_STATION_TO_PREFIX = {
    # ── Offshore Wind (nationalGridBmUnit prefixes) ──────
    'aberdeen bay': 'ABRB',
    'beatrice': 'BEAT',
    'burbo bank extension': 'BRBE',
    'burbo bank': 'BURB',
    'dogger bank a': 'DBAW',
    'dogger bank b': 'DBBW',
    'dudgeon': 'DDGN',
    'east anglia one': 'EAAO',
    'galloper': 'GAOF',
    'greater gabbard': 'GRGB',
    'gwynt y mor': 'GYMR',
    'humber gateway': 'HMGT',
    'hornsea': 'HOWA',
    'hornsea 2': 'HOWB',
    'kincardine offshore': 'KINC',
    'london array': 'LARY',
    'lincs': 'LNCS',
    'moray east': 'MOWE',
    'moray west': 'MOWW',
    'neart na gaoithe': 'NNGA',
    'ormonde': 'OMND',
    'race bank': 'RCBK',
    'rampion': 'RMPN',
    'seagreen': 'SGRW',
    'sheringham shoal': 'SHRS',
    'thanet': 'THNT',
    'walney': 'WLNY',
    'west of duddon sands': 'WDNS',
    'westermost rough': 'WTMS',
    'barrow offshore': 'BOWL',
    'gunfleet sands': 'GNFS',
    # ── Onshore Wind (large BM-participating farms) ──────
    'auchrobert': 'ABRT',
    'a chruach': 'ACHR',
    'afton': 'AFTO',
    'airies': 'AIRS',
    'aikengall': 'AKGL',
    'an suidhe': 'ANSU',
    'andershaw': 'ASHW',
    'arecleoch': 'ARCH',
    'assel valley': 'ASLV',
    'bad a cheo': 'BDCH',
    'baillie': 'BABA',
    'beinneun': 'BEIN',
    'benbrack': 'BENB',
    'bhlaraidh': 'BHLA',
    'black law': 'BLLA',
    'blackcraig': 'BLKW',
    'berry burn': 'BRYB',
    'beinn an tuirc': 'BTUI',
    'braes of doune': 'BRDU',
    'broken cross': 'BROC',
    'corriegarth': 'CGTH',
    'clyde central': 'CLDC',
    'clyde north': 'CLDN',
    'clyde south': 'CLDS',
    'clyde': 'CLDC',
    'clachan flats': 'CLFL',
    'camster': 'CMST',
    'coire na cloiche': 'CNCL',
    'cour': 'COUW',
    'crossdykes': 'CRDE',
    'creag riabhach': 'CREA',
    'carraig gheal': 'CRGH',
    'corriemoillie': 'CRML',
    'crystal rig': 'CRYR',
    'cumberhead': 'CUMH',
    'dalquhandy': 'DALQ',
    'dalswinton': 'DALS',
    'dunlaw': 'DNLW',
    'dorenell': 'DORE',
    'dersalloch': 'DRSL',
    'dunmaglass': 'DUNG',
    'douglas west': 'DWEX',
    'edinbane': 'EDIN',
    'enoch hill': 'ENHL',
    'ewe hill': 'EWHL',
    'farr': 'FAAR',
    'fallago rig': 'FALG',
    'freasdail': 'FSDL',
    'galawhistle': 'GLWS',
    'glenchamber': 'GLCH',
    'glen kyllachy': 'GLNK',
    'glens of foudland': 'GLOF',
    'glen app': 'GNAP',
    'gordonbush': 'GORD',
    'greengairs': 'GRGR',
    'griffin': 'GRIF',
    'hadyard hill': 'HADH',
    'hagshaw': 'HAHA',
    'halsary': 'HALS',
    'harburnhead': 'HBHD',
    'harestanes': 'HRST',
    'hare hill': 'HRHL',
    'hill of glaschyle': 'HLGL',
    'hill of towie': 'HLTW',
    'hywind': 'HYWD',
    'kennoxhead': 'KENN',
    'kilbraur': 'KILB',
    'kilgallioch': 'KLGL',
    'kype muir extension': 'KYPE',
    'kype muir': 'KPMR',
    'limekiln': 'LIMK',
    'lochluichart': 'LCLT',
    'mark hill': 'MKHL',
    'middle muir': 'MIDM',
    'minnygap': 'MYGP',
    'moy': 'MOYE',
    'north kyle': 'NOKY',
    'pines burn': 'PIBU',
    'penyclun': 'PNYC',
    'pogbie': 'PGBI',
    'robin rigg': 'RRWW',
    'sandy knowe': 'SAKN',
    'sanquhar': 'SANQ',
    'south kyle': 'SOKY',
    'solwaybank': 'SWBK',
    'stronelairg': 'STLG',
    'strathy north': 'STRN',
    'toddleburn': 'TDBN',
    'triton knoll': 'TKNE',
    'twenty shilling': 'TWSH',
    'viking energy': 'VKNG',
    'whitelee': 'WHIL',
    'whitelee extension': 'WHIL',
    'whiteside hill': 'WHIH',
    'windy rig': 'WDRG',
    'brockloch rig': 'WIST',
    # ── Solar (large BM-participating farms) ─────────────
    'cleve hill solar': 'CLVH',
    'larks green solar': 'LARK',
    'sutton bridge solar': 'SUTB',
    'beechgreen': 'BURW',
    'shotwick solar': 'SHSW',
    'bradenstoke solar': 'BSOL',
    'manor farm solar': 'MFGN',
}

# Reverse: prefix → station name
RENEWABLE_PREFIX_TO_STATION = {v: k for k, v in RENEWABLE_STATION_TO_PREFIX.items()}

# Carrier assignment: offshore detection uses BMU suffix 'O', onshore uses 'W'.
# The static sets below serve as fallback when the ELEXON registry is not available.
_OFFSHORE_PREFIXES = {
    'ABRB', 'BEAT', 'BRBE', 'BOWL', 'BURB', 'DBAW', 'DBBW', 'DDGN', 'EAAO',
    'GAOF', 'GNFS', 'GRGB', 'GYMR', 'HMGT', 'HOWA', 'HOWB', 'KINC', 'LARY',
    'LNCS', 'MOWE', 'MOWW', 'NNGA', 'OMND', 'RCBK', 'RMPN', 'SGRW', 'SHRS',
    'THNT', 'WDNS', 'WLNY', 'WTMS',
}
_ONSHORE_PREFIXES = {
    'ABRT', 'ACHR', 'AFTO', 'AIRS', 'AKGL', 'ANSU', 'ARCH', 'ASHW', 'ASLV',
    'BABA', 'BDCH', 'BEIN', 'BENB', 'BHLA', 'BLLA', 'BLKW', 'BNWK', 'BRDU',
    'BROC', 'BRYB', 'BTUI', 'CGTH', 'CLDC', 'CLDN', 'CLDR', 'CLDS', 'CLFL',
    'CMST', 'CNCL', 'COUW', 'CRDE', 'CREA', 'CRGH', 'CRML', 'CRYR', 'CUMH',
    'DALQ', 'DALS', 'DNLW', 'DORE', 'DOUG', 'DRSL', 'DUNG', 'DWEX', 'EDIN',
    'ENHL', 'EWHL', 'FAAR', 'FALG', 'FSDL', 'GLCH', 'GLNK', 'GLOF', 'GLWS',
    'GNAP', 'GORD', 'GRGR', 'GRIF', 'HADH', 'HAHA', 'HALS', 'HBHD', 'HLGL',
    'HLTW', 'HRHL', 'HRST', 'HYWD', 'KENN', 'KILB', 'KLGL', 'KPMR', 'KYPE',
    'LCLT', 'LIMK', 'MIDM', 'MKHL', 'MOYE', 'MYGP', 'NOKY', 'PGBI', 'PIBU',
    'PNYC', 'RRWW', 'RREW', 'SAKN', 'SANQ', 'SOKY', 'STLG', 'STRN', 'SWBK',
    'TDBN', 'TKNE', 'TKNW', 'TRLG', 'TULW', 'TWSH', 'VKNG', 'WDRG', 'WHIH',
    'WHIL', 'WIST',
}
_SOLAR_PREFIXES = {'SHSW', 'BSOL', 'MFGN', 'CLVH', 'LARK', 'SUTB', 'BURW'}

# ── Runtime registry cache (populated by fetch_renewable_bmu_registry) ───────
_RENEWABLE_BMU_REGISTRY: Optional[set] = None
_BMU_REGISTRY_DETAILS: Optional[pd.DataFrame] = None


def _prefix_to_carrier(prefix: str, bmu_id: str = '') -> str:
    """Map a BMU prefix to its renewable carrier.

    Uses BMU suffix convention as primary signal:
      - 'O' suffix → wind_offshore  (e.g. BEATO-1, SGRWO-3)
      - 'W' suffix → wind_onshore   (e.g. GRIFW-1, STLGW-2)
      - 'S' suffix → solar_pv       (e.g. CLVHS-1, SUTBS-1)
      - 'B' suffix for some onshore  (e.g. WHLWB-1)
    Falls back to static prefix sets.
    """
    # Use the full BMU prefix (all alpha chars) to detect suffix
    if bmu_id:
        alpha = ''.join(ch for ch in bmu_id.split('-')[0] if ch.isalpha())
        if alpha:
            last = alpha[-1].upper()
            if last == 'O':
                return 'wind_offshore'
            elif last in ('W', 'B'):
                return 'wind_onshore'
            elif last == 'S':
                return 'solar_pv'
    # Fallback to static sets
    if prefix in _OFFSHORE_PREFIXES:
        return 'wind_offshore'
    elif prefix in _ONSHORE_PREFIXES:
        return 'wind_onshore'
    elif prefix in _SOLAR_PREFIXES:
        return 'solar_pv'
    return 'unknown_renewable'


def fetch_renewable_bmu_registry(
    cache_dir: Optional[str] = None,
    logger: logging.Logger = logger,
) -> set:
    """
    Fetch the set of all renewable BMU IDs from the ELEXON BMU registry.

    Queries ``/reference/bmunits/all`` and filters to fuelType='WIND' plus
    known solar BMUs.  Returns a set of nationalGridBmUnit strings that can
    be used to filter BOD records without a hardcoded prefix dictionary.

    Results are cached in-memory and optionally to a CSV file.

    Returns
    -------
    set of str
        nationalGridBmUnit IDs for all BM-registered renewable generators.
    """
    global _RENEWABLE_BMU_REGISTRY, _BMU_REGISTRY_DETAILS

    if _RENEWABLE_BMU_REGISTRY is not None:
        return _RENEWABLE_BMU_REGISTRY

    # Try loading from cache file first
    if cache_dir:
        cache_path = Path(cache_dir) / "renewable_bmu_registry.csv"
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                _RENEWABLE_BMU_REGISTRY = set(df['nationalGridBmUnit'].dropna())
                _BMU_REGISTRY_DETAILS = df
                logger.info(f"Loaded BMU registry from cache: {len(_RENEWABLE_BMU_REGISTRY)} renewable BMUs")
                return _RENEWABLE_BMU_REGISTRY
            except Exception as e:
                logger.warning(f"Failed to load BMU registry cache: {e}")

    try:
        import requests
    except ImportError:
        logger.warning("requests not available; falling back to static prefix dict")
        _RENEWABLE_BMU_REGISTRY = set()
        return _RENEWABLE_BMU_REGISTRY

    from scripts.market.elexon_data import ELEXON_API_BASE

    try:
        url = f"{ELEXON_API_BASE}/reference/bmunits/all"
        logger.info("Fetching BMU registry from ELEXON API...")
        resp = requests.get(url, params={'format': 'json'}, timeout=60)
        resp.raise_for_status()
        all_bmus = resp.json()
        df = pd.DataFrame(all_bmus)

        # Select WIND fuel type (T and E types participate in BM)
        wind_mask = (df['fuelType'] == 'WIND') & (df['bmUnitType'].isin(['T', 'E']))
        # Also pick up solar BMUs from OTHER fuel type by name matching
        solar_mask = (
            (df['fuelType'].isin(['OTHER', 'SOLAR']))
            & (df['bmUnitType'].isin(['T', 'E']))
            & (df['bmUnitName'].str.contains('solar|pv|photo', case=False, na=False))
        )
        renewable_df = df[wind_mask | solar_mask].copy()

        _RENEWABLE_BMU_REGISTRY = set(renewable_df['nationalGridBmUnit'].dropna())
        _BMU_REGISTRY_DETAILS = renewable_df
        logger.info(f"ELEXON BMU registry: {len(_RENEWABLE_BMU_REGISTRY)} renewable BMUs "
                     f"({wind_mask.sum()} wind, {solar_mask.sum()} solar)")

        # Cache to file
        if cache_dir:
            cache_path = Path(cache_dir) / "renewable_bmu_registry.csv"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            renewable_df[['nationalGridBmUnit', 'bmUnitName', 'fuelType',
                          'bmUnitType', 'generationCapacity', 'gspGroupName']].to_csv(
                cache_path, index=False)
            logger.info(f"Cached BMU registry to {cache_path}")

        return _RENEWABLE_BMU_REGISTRY

    except Exception as e:
        logger.warning(f"Failed to fetch BMU registry: {e}; falling back to static prefix dict")
        _RENEWABLE_BMU_REGISTRY = set()
        return _RENEWABLE_BMU_REGISTRY


# ══════════════════════════════════════════════════════════════════════════════
# ELEXON BOD FETCHING (RENEWABLE-TARGETED)
# ══════════════════════════════════════════════════════════════════════════════


def _extract_prefix(bmu_id: str) -> Optional[str]:
    """Extract the 4-char prefix from a BMU ID (e.g., T_HORN-1 → HORN)."""
    if not isinstance(bmu_id, str):
        return None
    for strip in ('T_', 'E_', 'M_', '2_'):
        if bmu_id.startswith(strip):
            bmu_id = bmu_id[len(strip):]
            break
    # Take chars before first digit or hyphen
    prefix = ''
    for ch in bmu_id:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix.upper() if len(prefix) >= 3 else None


def _is_renewable_bmu(bmu_id: str, registry: Optional[set] = None) -> bool:
    """Check if a BMU ID belongs to a known renewable station.

    Uses the ELEXON BMU registry (if available) for authoritative matching,
    falling back to the static prefix dictionary.
    """
    # Primary: check against the live ELEXON registry
    if registry and bmu_id in registry:
        return True
    if _RENEWABLE_BMU_REGISTRY and bmu_id in _RENEWABLE_BMU_REGISTRY:
        return True

    # Fallback: static prefix dictionary
    prefix = _extract_prefix(bmu_id)
    if prefix is None:
        return False
    for length in [4, 3, 5]:
        if prefix[:length] in RENEWABLE_PREFIX_TO_STATION:
            return True
    return False


def fetch_renewable_bids(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Fetch BOD bid prices for renewable BMUs over a date range.

    Downloads BOD data from ELEXON for each day, filters to renewable BMUs,
    and extracts bid prices. Returns a long-form DataFrame.

    Uses the ``/datasets/BOD/stream`` bulk endpoint (no pagination), processing
    one calendar day per request to bound memory usage (~40 MB per day).
    The ``nationalGridBmUnit`` field is matched against the ELEXON BMU
    registry (fuelType='WIND') to capture ALL renewable BM participants.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD), inclusive.
    output_dir : str, optional
        If given, saves a combined cache CSV so subsequent runs skip the fetch.
    logger : Logger

    Returns
    -------
    DataFrame with columns: bmu_id, datetime, bid_price, settlement_period
    """
    try:
        import requests  # noqa: F811
    except ImportError:
        raise ImportError("The 'requests' package is required. pip install requests")

    from scripts.market.elexon_data import ELEXON_API_BASE

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq="D")

    logger.info(f"Fetching renewable BOD bids via stream: {start_date} to {end_date} ({len(dates)} days)")

    if output_dir:
        cache_path = Path(output_dir) / "renewable_bids_cache.csv"
        if cache_path.exists():
            logger.info(f"Loading cached bids from {cache_path}")
            return pd.read_csv(cache_path, parse_dates=['datetime'])

        # Try restoring from compressed archive before hitting the API
        try:
            from scripts.utilities.elexon_cache import ensure_cache_for_year
            year = start.year
            if ensure_cache_for_year(year, logger=logger) and cache_path.exists():
                logger.info(f"Restored from archive — loading cached bids for {year}")
                return pd.read_csv(cache_path, parse_dates=['datetime'])
        except ImportError:
            pass  # elexon_cache not available — fall through to API fetch

    # Fetch the authoritative set of renewable BMU IDs from ELEXON registry
    registry = fetch_renewable_bmu_registry(cache_dir=output_dir, logger=logger)
    if registry:
        logger.info(f"Using ELEXON registry: {len(registry)} renewable BMU IDs for filtering")
    else:
        logger.warning("Registry unavailable — falling back to static prefix dictionary")

    all_bids = []
    url = f"{ELEXON_API_BASE}/datasets/BOD/stream"

    for i, date in enumerate(dates):
        dt_from = date.strftime("%Y-%m-%dT00:00Z")
        dt_to = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00Z")

        if (i + 1) % 7 == 0 or i == 0:
            logger.info(f"  Day {i + 1}/{len(dates)}: {date.date()}")

        try:
            resp = requests.get(url, params={"from": dt_from, "to": dt_to, "format": "json"},
                                timeout=120)
            resp.raise_for_status()
            records = resp.json()
            if isinstance(records, dict):
                records = records.get("data", [])

            n_found = 0
            for record in records:
                ng_bmu = record.get("nationalGridBmUnit") or ""
                if not ng_bmu or not _is_renewable_bmu(ng_bmu, registry=registry):
                    continue

                bid_price = record.get("bid")
                if bid_price is None:
                    continue

                sp = int(record.get("settlementPeriod") or 0)
                settle_date = pd.Timestamp(record.get("settlementDate") or str(date.date()))
                dt = settle_date + timedelta(minutes=30 * (sp - 1))

                all_bids.append({
                    'bmu_id': ng_bmu,          # e.g. "HORN-1" (nationalGridBmUnit)
                    'datetime': dt,
                    'bid_price': float(bid_price),
                    'settlement_period': sp,
                    'pair_id': int(record.get("pairId") or 0),
                })
                n_found += 1

            logger.debug(f"    {date.date()}: {len(records)} total records, {n_found} renewable")

        except Exception as e:
            logger.warning(f"  Failed to fetch {date.date()}: {e}")
            continue

        time.sleep(0.5)  # gentle rate limiting between daily requests

    df = pd.DataFrame(all_bids)
    logger.info(f"Collected {len(df)} renewable bid records across "
                f"{df['bmu_id'].nunique() if not df.empty else 0} BMUs")

    if output_dir and not df.empty:
        cache_path = Path(output_dir) / "renewable_bids_cache.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached bids to {cache_path}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BID PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════


def _infer_support_type(median_bid: float, std_bid: float) -> str:
    """
    Infer generator support type from bid price distribution.

    Heuristic thresholds (based on GBPower analysis):
      - median_bid < -20 £/MWh  → ROC (losing significant subsidy if curtailed)
      - median_bid >= -20        → CfD or merchant (near-zero opportunity cost)
    """
    if median_bid < -20.0:
        return 'ROC'
    else:
        return 'CfD_or_merchant'


def analyse_renewable_bids(
    bids_df: pd.DataFrame,
    network_path: Optional[str] = None,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Analyse renewable bid prices to estimate per-generator empirical MCs.

    For each renewable BMU:
      1. Compute median, mean, and std of bid prices over the calibration period
      2. Infer support type (ROC / CfD_or_merchant) from the distribution
      3. Set empirical_mc = median(bid_price)

    BMUs are then mapped to PyPSA generator names where possible.

    Parameters
    ----------
    bids_df : DataFrame
        Long-form bid data with columns: bmu_id, datetime, bid_price.
    network_path : str, optional
        Path to PyPSA network for generator name matching.
    logger : Logger

    Returns
    -------
    DataFrame with columns: generator, carrier, empirical_mc, support_type_inferred,
        n_bmus, median_bid, mean_bid, std_bid
    """
    if bids_df.empty:
        logger.warning("No bid data to analyse")
        return pd.DataFrame()

    # Filter extreme protective / null bids before computing statistics.
    # Realistic renewable bids: ROC generators bid ~ -50 to -150 £/MWh;
    # CfD/merchant near 0.  Values outside ±500 are BM exception bids (e.g.
    # Whitelee's -9999 "do not reduce" bid) and would corrupt the median.
    n_raw = len(bids_df)
    bids_df = bids_df[(bids_df['bid_price'] >= -500.0) & (bids_df['bid_price'] <= 500.0)].copy()
    n_filtered = n_raw - len(bids_df)
    if n_filtered:
        logger.info(f"Filtered {n_filtered} extreme bid records (outside ±£500/MWh)")

    # Per-BMU statistics
    bmu_stats = bids_df.groupby('bmu_id').agg(
        median_bid=('bid_price', 'median'),
        mean_bid=('bid_price', 'mean'),
        std_bid=('bid_price', 'std'),
        n_records=('bid_price', 'count'),
    ).reset_index()

    # Filter: require at least 50 bid records for reliable estimate
    bmu_stats = bmu_stats[bmu_stats['n_records'] >= 50].copy()
    logger.info(f"Bid statistics for {len(bmu_stats)} BMUs (≥50 records each)")

    if bmu_stats.empty:
        logger.warning("No BMUs with sufficient bid data")
        return pd.DataFrame()

    # Assign carrier and support type
    bmu_stats['prefix'] = bmu_stats['bmu_id'].apply(_extract_prefix)
    bmu_stats['carrier'] = bmu_stats.apply(
        lambda row: _prefix_to_carrier(
            row['prefix'][:4] if row['prefix'] and len(row['prefix']) >= 4 else '',
            bmu_id=row['bmu_id']
        ), axis=1
    )
    bmu_stats['support_type_inferred'] = bmu_stats.apply(
        lambda row: _infer_support_type(row['median_bid'], row['std_bid']), axis=1
    )
    bmu_stats['empirical_mc'] = bmu_stats['median_bid']

    # Map to station name — try registry first, then static dict
    def _resolve_station(row):
        prefix = row['prefix']
        bmu_id = row['bmu_id']
        # Try ELEXON registry details (authoritative station name)
        if _BMU_REGISTRY_DETAILS is not None and not _BMU_REGISTRY_DETAILS.empty:
            match = _BMU_REGISTRY_DETAILS[
                _BMU_REGISTRY_DETAILS['nationalGridBmUnit'] == bmu_id
            ]
            if not match.empty:
                name = match.iloc[0].get('bmUnitName', '')
                if name and name != bmu_id:
                    return name.lower()
        # Fallback: static prefix dict
        if prefix:
            return RENEWABLE_PREFIX_TO_STATION.get(prefix[:4], None)
        return None

    bmu_stats['station'] = bmu_stats.apply(_resolve_station, axis=1)

    logger.info("Per-BMU bid analysis:")
    for _, row in bmu_stats.sort_values('median_bid').iterrows():
        logger.info(f"  {row['bmu_id']:15s}  {row['station'] or '?':25s}  "
                     f"{row['carrier']:15s}  "
                     f"median=£{row['median_bid']:>7.1f}  "
                     f"mean=£{row['mean_bid']:>7.1f}  "
                     f"std=£{row['std_bid']:>5.1f}  "
                     f"n={row['n_records']:>5d}  "
                     f"→ {row['support_type_inferred']}")

    # Map BMUs to PyPSA generator names
    bmu_stats = _map_bmus_to_generators(bmu_stats, network_path, logger)

    # Aggregate: multiple BMUs per generator → median
    if 'generator' in bmu_stats.columns and bmu_stats['generator'].notna().any():
        mapped = bmu_stats[bmu_stats['generator'].notna()].copy()
        unmapped = bmu_stats[bmu_stats['generator'].isna()].copy()

        agg = mapped.groupby(['generator', 'carrier']).agg(
            empirical_mc=('empirical_mc', 'median'),
            support_type_inferred=('support_type_inferred', 'first'),
            n_bmus=('bmu_id', 'count'),
            median_bid=('median_bid', 'median'),
            mean_bid=('mean_bid', 'mean'),
            std_bid=('std_bid', 'mean'),
        ).reset_index()

        if not unmapped.empty:
            logger.info(f"\n  {len(unmapped)} BMUs not mapped to generators:")
            for _, row in unmapped.iterrows():
                logger.info(f"    {row['bmu_id']} ({row['station'] or '?'}): "
                             f"£{row['median_bid']:.1f}/MWh")
    else:
        # No generator mapping — use station names as identifiers
        agg = bmu_stats.groupby(['station', 'carrier']).agg(
            empirical_mc=('empirical_mc', 'median'),
            support_type_inferred=('support_type_inferred', 'first'),
            n_bmus=('bmu_id', 'count'),
            median_bid=('median_bid', 'median'),
            mean_bid=('mean_bid', 'mean'),
            std_bid=('std_bid', 'mean'),
        ).reset_index()
        agg = agg.rename(columns={'station': 'generator'})

    logger.info(f"\nPer-generator renewable MCs for {len(agg)} generators:")
    for _, row in agg.sort_values('empirical_mc').iterrows():
        logger.info(f"  {row['generator']:30s} ({row['carrier']:15s}): "
                     f"£{row['empirical_mc']:>7.1f}/MWh  "
                     f"({row['support_type_inferred']}, {row['n_bmus']} BMUs)")

    # Summary by carrier and support type
    logger.info("\nSummary by carrier:")
    for carrier in agg['carrier'].unique():
        subset = agg[agg['carrier'] == carrier]
        logger.info(f"  {carrier}: {len(subset)} generators, "
                     f"median MC £{subset['empirical_mc'].median():.1f}/MWh")
        for st in subset['support_type_inferred'].unique():
            st_subset = subset[subset['support_type_inferred'] == st]
            logger.info(f"    {st}: {len(st_subset)} generators, "
                         f"median MC £{st_subset['empirical_mc'].median():.1f}/MWh")

    return agg


def _map_bmus_to_generators(
    bmu_stats: pd.DataFrame,
    network_path: Optional[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Map BMU IDs to PyPSA generator names.

    Matching strategy (in priority order):
      1. Direct station name match against generator index
      2. Station name substring match (both directions)
      3. Key-word extraction from ELEXON bmUnitName for broader matching
      4. Falls back to using station name as identifier

    If a network path is provided, matches BMU station names against actual
    generator names in the network. Otherwise, uses the station name directly.
    """
    bmu_stats = bmu_stats.copy()
    bmu_stats['generator'] = None

    if network_path and Path(network_path).exists():
        try:
            import pypsa
            network = pypsa.Network(network_path)

            # Build renewable generator lookup
            renewable_carriers = {'wind_offshore', 'wind_onshore', 'solar_pv',
                                  'large_hydro', 'small_hydro', 'tidal_stream',
                                  'shoreline_wave', 'marine'}
            gen_lookup = {}
            for gen_name in network.generators.index:
                carrier = network.generators.loc[gen_name, 'carrier']
                if carrier not in renewable_carriers:
                    continue
                base = gen_name.split('__agg')[0].lower().strip()
                gen_lookup[base] = gen_name
                gen_lookup[gen_name.lower()] = gen_name

            # Sort gen_lookup entries longest-first for better partial matching
            gen_keys_sorted = sorted(gen_lookup.keys(), key=len, reverse=True)

            for idx, row in bmu_stats.iterrows():
                station = row.get('station')
                if station is None:
                    continue
                station_lower = station.lower().strip()
                station_nospace = station_lower.replace(' ', '').replace('_', '')

                # 1. Direct match
                if station_lower in gen_lookup:
                    bmu_stats.at[idx, 'generator'] = gen_lookup[station_lower]
                    continue

                # 2. Station name is a substring of generator name (or vice versa)
                matched = False
                for key in gen_keys_sorted:
                    key_nospace = key.replace(' ', '').replace('_', '')
                    if station_nospace in key_nospace or key_nospace in station_nospace:
                        bmu_stats.at[idx, 'generator'] = gen_lookup[key]
                        matched = True
                        break

                if matched:
                    continue

                # 3. Try matching on key words (e.g. "hornsea" from "hornsea_1a")
                #    Extract significant words (≥4 chars) from station name
                words = [w for w in station_lower.replace('_', ' ').split()
                         if len(w) >= 4 and w not in ('wind', 'farm', 'offshore',
                                                       'onshore', 'solar', 'extension',
                                                       'windfarm', 'unit', 'generator')]
                if words:
                    keyword = words[0]  # Use the most significant word
                    for key in gen_keys_sorted:
                        if keyword in key:
                            bmu_stats.at[idx, 'generator'] = gen_lookup[key]
                            matched = True
                            break

            n_mapped = bmu_stats['generator'].notna().sum()
            n_total = len(bmu_stats)
            logger.info(f"Mapped {n_mapped}/{n_total} BMUs to network generators "
                         f"({n_total - n_mapped} unmapped)")

        except Exception as e:
            logger.warning(f"Could not load network for mapping: {e}")
            bmu_stats['generator'] = bmu_stats['station']
    else:
        # No network — use station name as generator identifier
        bmu_stats['generator'] = bmu_stats['station']

    return bmu_stats


# ══════════════════════════════════════════════════════════════════════════════
# CARRIER-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_renewable_carrier_stats(
    per_gen_df: pd.DataFrame,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Compute carrier-level summary statistics from per-generator MCs.

    Useful for setting default ROC banding values in config:
      estimated_ro_banding ≈ |median_bid| / roc_buyout_price

    Parameters
    ----------
    per_gen_df : DataFrame
        Output of analyse_renewable_bids().

    Returns
    -------
    DataFrame with carrier, mean_mc, median_mc, n_generators, support_type_mix
    """
    if per_gen_df.empty:
        return pd.DataFrame()

    rows = []
    for carrier in per_gen_df['carrier'].unique():
        subset = per_gen_df[per_gen_df['carrier'] == carrier]
        support_counts = subset['support_type_inferred'].value_counts().to_dict()

        rows.append({
            'carrier': carrier,
            'mean_mc': subset['empirical_mc'].mean(),
            'median_mc': subset['empirical_mc'].median(),
            'n_generators': len(subset),
            'support_type_mix': str(support_counts),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FROM PRE-FETCHED BIDS CSV
# ══════════════════════════════════════════════════════════════════════════════


def load_bids_from_csv(
    bids_file: str,
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Load bid data from a pre-fetched CSV and filter to renewable BMUs.

    The CSV is expected to be in the format output by
    retrieve_elexon_market_data() — wide format with datetime index and
    one column per BMU containing bid price (£/MWh).

    Parameters
    ----------
    bids_file : str
        Path to elexon_bids.csv (wide format: datetime × bmu_id).

    Returns
    -------
    Long-form DataFrame with columns: bmu_id, datetime, bid_price
    """
    df = pd.read_csv(bids_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded bids CSV: {df.shape[0]} periods × {df.shape[1]} BMUs")

    # Filter to renewable BMU columns
    renewable_cols = [col for col in df.columns if _is_renewable_bmu(col)]
    logger.info(f"Found {len(renewable_cols)} renewable BMU columns")

    if not renewable_cols:
        logger.warning("No renewable BMUs found in bids CSV")
        return pd.DataFrame()

    # Melt to long form
    df_renew = df[renewable_cols].copy()
    df_renew.index.name = 'datetime'
    long = df_renew.reset_index().melt(
        id_vars='datetime', var_name='bmu_id', value_name='bid_price'
    ).dropna(subset=['bid_price'])

    logger.info(f"Extracted {len(long)} renewable bid records")
    return long


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CALIBRATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def calibrate_renewable_mc(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bids_file: Optional[str] = None,
    network_path: Optional[str] = None,
    output_file: str = "data/market/renewable_empirical_mc.csv",
    data_dir: str = "resources/market/elexon",
    logger: logging.Logger = logger,
) -> pd.DataFrame:
    """
    Main entry point: calibrate renewable MCs from ELEXON bid data.

    Either fetches BOD data from ELEXON API (if start/end dates given) or
    loads from a pre-fetched CSV (if bids_file given).

    Parameters
    ----------
    start_date : str, optional
        Start date for ELEXON fetch (YYYY-MM-DD).
    end_date : str, optional
        End date for ELEXON fetch (YYYY-MM-DD).
    bids_file : str, optional
        Path to pre-fetched bids CSV (skips API fetch).
    network_path : str, optional
        Path to solved PyPSA network for generator name matching.
    output_file : str
        Path to write per-generator MC CSV.
    data_dir : str
        Directory for cached ELEXON data.
    logger : Logger

    Returns
    -------
    DataFrame with per-generator renewable empirical MCs.
    """
    logger.info("=" * 80)
    logger.info("RENEWABLE MARGINAL COST CALIBRATION (ELEXON BID PRICES)")
    logger.info("=" * 80)

    # Step 0: Pre-fetch BMU registry so _is_renewable_bmu can use it
    fetch_renewable_bmu_registry(cache_dir=data_dir, logger=logger)

    # Step 1: Get bid data
    if bids_file and Path(bids_file).exists():
        logger.info(f"Loading bids from CSV: {bids_file}")
        bids_df = load_bids_from_csv(bids_file, logger)
    elif start_date and end_date:
        logger.info(f"Fetching bids from ELEXON: {start_date} to {end_date}")
        bids_df = fetch_renewable_bids(start_date, end_date, data_dir, logger)
    else:
        logger.error("Either --bids-file or --start-date + --end-date required")
        return pd.DataFrame()

    if bids_df.empty:
        logger.error("No bid data available")
        return pd.DataFrame()

    # Step 2: Analyse bids per BMU and aggregate per generator
    result = analyse_renewable_bids(bids_df, network_path, logger)

    # Step 3: Carrier-level summary
    if not result.empty:
        carrier_stats = compute_renewable_carrier_stats(result, logger)
        logger.info("\nCarrier-level summary:")
        for _, row in carrier_stats.iterrows():
            logger.info(f"  {row['carrier']:15s}: {row['n_generators']:3d} generators, "
                         f"median MC £{row['median_mc']:.1f}/MWh, "
                         f"support mix: {row['support_type_mix']}")

    # Step 4: Save output
    if not result.empty and output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        logger.info(f"\nSaved renewable MCs: {len(result)} generators → {output_file}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def snakemake_main():
    """Entry point when called from Snakemake rule.

    Determines calibration year from scenario config. For future scenarios
    (modelled_year > 2024), writes an empty CSV since no ELEXON data exists.
    """
    global logger

    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "calibrate_renewable_mc"
    logger = setup_logging(log_path)

    logger.info("=" * 80)
    logger.info("RENEWABLE EMPIRICAL MC CALIBRATION (SNAKEMAKE)")
    logger.info("=" * 80)

    network_path = snakemake.input.network
    output_file = snakemake.output.renewable_mc
    scenario_config = snakemake.params.scenario_config
    data_dir = getattr(snakemake.params, 'data_dir', 'resources/market/elexon')

    modelled_year = scenario_config.get('modelled_year', 2023)

    def _write_empty_csv(reason: str) -> None:
        logger.info(f"{reason}: writing empty renewable MC file")
        empty_df = pd.DataFrame(columns=['generator', 'carrier', 'empirical_mc',
                                          'support_type_inferred', 'n_bmus'])
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_file, index=False)
        logger.info(f"Saved empty CSV -> {output_file}")

    # Future scenarios: no ELEXON data available
    if modelled_year > 2024:
        _write_empty_csv(f"Future scenario (year {modelled_year})")
        return

    # Feature disabled: skip expensive API fetch
    empirical_enabled = (
        scenario_config
        .get('marginal_costs', {})
        .get('empirical_calibration', {})
        .get('enabled', False)
    )
    if not empirical_enabled:
        _write_empty_csv("empirical_calibration.enabled: false")
        return

    # Historical scenario: calibrate from ELEXON BOD data
    cal_year = modelled_year

    # Scope fetch to solve period if defined (avoids fetching a full year for short runs)
    solve_period = scenario_config.get('solve_period', {})
    if solve_period.get('enabled', False) and solve_period.get('start') and solve_period.get('end'):
        sp_start = pd.Timestamp(solve_period['start'])
        sp_end = pd.Timestamp(solve_period['end'])
        # Use solve period ± 14 days for statistical robustness
        buffer = pd.Timedelta(days=14)
        fetch_start = max(sp_start - buffer, pd.Timestamp(f"{cal_year}-01-01"))
        fetch_end = min(sp_end + buffer, pd.Timestamp(f"{cal_year}-12-31"))
        start_date = fetch_start.strftime("%Y-%m-%d")
        end_date = fetch_end.strftime("%Y-%m-%d")
        logger.info(f"Scoped to solve_period ± 14d: {start_date} to {end_date}")
    else:
        start_date = f"{cal_year}-01-01"
        end_date = f"{cal_year}-12-31"

    # Year-specific cache directory so each year fetches its own data
    year_data_dir = str(Path(data_dir) / str(cal_year))

    logger.info(f"Calibration year: {cal_year}")
    logger.info(f"Network: {network_path}")
    logger.info(f"Date range: {start_date} to {end_date}")

    calibrate_renewable_mc(
        start_date=start_date,
        end_date=end_date,
        network_path=network_path,
        output_file=output_file,
        data_dir=year_data_dir,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("RENEWABLE CALIBRATION COMPLETE")
    logger.info("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate renewable marginal costs from ELEXON bid data"
    )
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for ELEXON data fetch (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for ELEXON data fetch (YYYY-MM-DD)")
    parser.add_argument("--bids-file", type=str, default=None,
                        help="Path to pre-fetched elexon_bids.csv (skips API fetch)")
    parser.add_argument("--network", type=str, default=None,
                        help="Path to solved PyPSA network .nc file (for name matching)")
    parser.add_argument("--output", type=str,
                        default="data/market/renewable_empirical_mc.csv",
                        help="Output CSV path")
    parser.add_argument("--data-dir", type=str, default="resources/market/elexon",
                        help="Directory for cached ELEXON data")

    args = parser.parse_args()

    calibrate_renewable_mc(
        start_date=args.start_date,
        end_date=args.end_date,
        bids_file=args.bids_file,
        network_path=args.network,
        output_file=args.output,
        data_dir=args.data_dir,
        logger=logger,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
        snakemake_main()
    except NameError:
        main()
