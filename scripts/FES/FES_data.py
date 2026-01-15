import requests
import pandas as pd
import io
import re
import time
import logging
import yaml
import os
import sys
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

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

# Load configuration from YAML file
snk = globals().get('snakemake')
if snk:
    config_file_path = snakemake.input[0]
else:
    config_file_path = os.path.join('data', 'FES', 'FES_api_urls.yaml')
try:
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.exception(f"Failed to load FES API config YAML at {config_file_path}: {e}")
    raise

def fetch_data(category, year):
    url = config.get(category, {}).get(year)
    if not url:
        logger.error(f"Data for year {year} in category {category} is not available.")
        return None
    
    # polite delay between requests
    time.sleep(1)
    try:
        logger.info(f"Fetching {category} for {year} from {url}")
        response = http.get(url, timeout=30)
        response.raise_for_status()
        csv_content = response.content
        # read into dataframe
        df = pd.read_csv(io.BytesIO(csv_content), encoding='ISO-8859-1', low_memory=False)

        # Remove non-standard characters from string columns
        try:
            df = df.apply(lambda col: col.map(lambda x: re.sub(r"[^\x00-\x7F]+", '', x).replace("'", '').replace("’", '') if isinstance(x, str) else x))
        except Exception as e:
            logger.debug(f"Non-ASCII cleanup failed for {category} {year}: {e}")

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

def raw_FES_data(FES_year):
    # Placeholder function for matching definitions to building blocks
    # read the csvs first

    year = str(FES_year)
    logger.info(f"Loading raw FES data for year {year}")
    building_blocks = None
    building_block_definitions = None
    for category in config.keys():
        df = fetch_data(category, FES_year)
        if df is None:
            logger.warning(f"No data returned for category {category} year {year}")
            continue
        if category == 'building_blocks':
            building_blocks = df
        elif category == 'building_block_definitions':
            building_block_definitions = df

    if building_blocks is None:
        logger.error("building_blocks data not available, aborting")
        raise RuntimeError("building_blocks data not available")
    if building_block_definitions is None:
        logger.error("building_block_definitions data not available, aborting")
        raise RuntimeError("building_block_definitions data not available")

    # if the year is 2022 then make the first row the column names
    if FES_year == 2022 or FES_year == 2021:
        logger.debug("Adjusting header row for building_block_definitions for year %s", year)
        building_block_definitions.columns = building_block_definitions.iloc[0]
        building_block_definitions = building_block_definitions[1:]
    
    building_block_definitions.set_index('Building Block ID Number', inplace=True)
    # for 2020 set the first column name to 'Building Block ID Number'
    if FES_year == 2020:
        building_blocks.rename(columns={building_blocks.columns[0]: 'Building Block ID Number'}, inplace=True)
    # set index
    building_blocks.set_index('Building Block ID Number', inplace=True)
    # merge
    merged = building_blocks.merge(building_block_definitions, left_index=True, right_index=True)
    logger.info(f"Merged building blocks and definitions: {merged.shape[0]} rows, {merged.shape[1]} columns")
    # need to rename for 2022
    if FES_year == 2022:
        merged.rename(columns={merged.columns[0]: 'FES Scenario'}, inplace=True)
        logger.debug("Renamed first merged column to 'FES Scenario' for year 2022")
    return merged

def add_node_id_to_gsp(FES_year, raw_FES_data):
    """
    Clean GSP names and map to Node IDs.
    
    GSP CLEANING RATIONALE:
    - NESO FES data uses informal/evolving GSP names
    - ETYS network requires exact GSP name matches
    - Offshore wind sites (2024+) include connection points in parentheses
    - London LPN/SPN circuits need subcircuit disambiguation
    - Some FES GSPs don't exist in ETYS network (islands, decommissioned sites)
    
    YEAR-SPECIFIC PATCHES:
    - 2021-2022: Header format fixes, basic GSP standardization (Dunbar A/B, Ferrybridge)
    - 2023: Broadford-Skye correction, Grange→Margam mapping, Grangemouth cleanup
    - 2024: Extensive offshore wind (Robin Rigg, Walney, Sheringham) and London subcircuit cleaning
    
    NOTE: Future FES years will likely need new patches as:
    - Offshore wind portfolio continues to expand
    - Network topology evolves (new GSPs, decommissioning)
    - NESO updates naming conventions in FES releases
    
    Args:
        FES_year: Year of FES data being processed
        raw_FES_data: Merged FES DataFrame with GSP column
        
    Returns:
        None (writes output CSV via Snakemake output path)
    """


    logger.info(f"Cleaning GSP names and mapping Node IDs for FES year {FES_year}")
    # Work on an explicit copy to avoid SettingWithCopyWarning when the
    # incoming `raw_FES_data` is a slice/view of another DataFrame.
    # This makes in-place assignments and .loc operations safe.
    raw_FES_data = raw_FES_data.copy()
    if FES_year == 2021:
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        # remove whitespace at end of strings
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.strip()
        # remove row with Cain, which I cant find existence of
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Cain']
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_F)', 'Ferrybridge B')
        # exact match for these
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Dumfries 11' if x == 'Dumfries' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Crookston A' if x == 'Crookston' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Grangemouth A' if x == 'Grangemouth' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kilbowie 11' if x == 'Kilbowie' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Saltcoats A' if x == 'Saltcoats' else x)
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year == 2022:
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Crookston A' if x == 'Crookston' else x)
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year == 2023:
        # replace the 951 row with CLYM
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Broadford-Skye', 'Broadford')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].replace({'Grange': 'Margam'})
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Grangemouth A' if x == 'Grangemouth' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kearsley 132kV' if x == 'Kearsley' else x)
        # delete row with Roosecote as can't see it has anything, bar some battery storage
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Roosecote']
        # delete row with Stornoway as only has little wind and no coordinates
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Stornoway']
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year == 2024:
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_H)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_G)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_D)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ormonde', 'Ormonde (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Robin Rigg', 'Robin Rigg (Harker)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Walney 2', 'Walney 2 (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 1', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 2', 'Tynemouth 132kV')
        # raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 132kV 2', 'Tynemouth 132kV')
        # raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 132kV 1', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hams Hall', 'Hams Hall (Lea Marston 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hart Moor', 'Hartmoor')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Camblesforth', 'Camblesforth 66kV (Drax)')
        # raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Skelton Margam (Grange 66kV)', 'Margam (Grange 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Fiddlers Ferry', 'Fiddlers Ferry (Cuerdley 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('St. Asaph', 'St Asaph (Bodelwyddan)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Broadford-Skye', 'Broadford')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Axminster_H', 'Axminster H')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Clacton', 'Clacton (Bramford)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Loudwater', 'Loudwater (Amersham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Sheringham Shoal', 'Sheringham Shoal (Norwich)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_C)', 'Beddington LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Canal Bank', 'Canal Bank (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Greenford', 'Greenford (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_C)', 'Littlebrook LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Lodge Road', 'Lodge Road (St Johns Wood)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('New Cross', 'New Cross 66kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Perivale', 'Perivale (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Wesley Avenue (Willesden)', 'Wesley Avenue (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_J)', 'Beddington SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_J)', 'Littlebrook SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].replace({'Drax': 'Camblesforth 66kV (Drax)', 'Grange': 'Margam (Grange 66kV)'})
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kearsley 132kV' if x == 'Kearsley' else x)
        # delete row with Port Ham as only small and no gsp data
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Port Ham']
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Roosecote']
        # remove these too Kellwood Road, Long Park, New Cumnock, Well Street Paisley, Stornoway
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Kellwood Road')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Long Park')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('New Cumnock')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Well Street Paisley')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Stornoway')]
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year >= 2025:
        # FES 2025+ - Start with 2024 logic and expand as needed
        logger.info(f"Using default GSP cleaning logic for FES {FES_year} (based on FES 2024)")
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_H)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_G)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_D)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ormonde', 'Ormonde (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Robin Rigg', 'Robin Rigg (Harker)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Walney 2', 'Walney 2 (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 1', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 2', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hams Hall', 'Hams Hall (Lea Marston 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hart Moor', 'Hartmoor')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Camblesforth', 'Camblesforth 66kV (Drax)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Fiddlers Ferry', 'Fiddlers Ferry (Cuerdley 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('St. Asaph', 'St Asaph (Bodelwyddan)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Broadford-Skye', 'Broadford')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Axminster_H', 'Axminster H')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Clacton', 'Clacton (Bramford)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Loudwater', 'Loudwater (Amersham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Sheringham Shoal', 'Sheringham Shoal (Norwich)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_C)', 'Beddington LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Canal Bank', 'Canal Bank (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Greenford', 'Greenford (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_C)', 'Littlebrook LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Lodge Road', 'Lodge Road (St Johns Wood)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('New Cross', 'New Cross 66kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Perivale', 'Perivale (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Wesley Avenue (Willesden)', 'Wesley Avenue (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_J)', 'Beddington SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_J)', 'Littlebrook SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].replace({'Drax': 'Camblesforth 66kV (Drax)', 'Grange': 'Margam (Grange 66kV)'})
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kearsley 132kV' if x == 'Kearsley' else x)
        # Remove unmapped GSPs
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Port Ham']
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Roosecote']
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Kellwood Road', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Long Park', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('New Cumnock', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Well Street Paisley', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Stornoway', na=False)]
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']
        logger.info(f"Applied default GSP cleaning rules for FES {FES_year}")

    # Download GSP info from API, with fallback to alternate years
    GSP_info = None
    
    # Try to fetch from API for the requested year first
    gsp_url = config.get('gsp_info', {}).get(FES_year)
    if gsp_url:
        logger.info(f"Fetching GSP info for {FES_year} from API")
        try:
            time.sleep(1)  # polite delay
            response = http.get(gsp_url, timeout=30)
            response.raise_for_status()
            # Use utf-8-sig to properly handle BOM (Byte Order Mark) in CSV files
            GSP_info = pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
            logger.info(f"Downloaded GSP info from API: {GSP_info.shape[0]} rows")
        except Exception as e:
            logger.warning(f"Failed to download GSP info for {FES_year} from API: {e}")
            GSP_info = None
    
    # If primary year failed, try alternate years (2024, 2023, 2022, 2021)
    if GSP_info is None:
        fallback_years = [2024, 2023, 2022, 2021]
        for fallback_year in fallback_years:
            if fallback_year == FES_year:
                continue  # Skip the year we already tried
            
            fallback_url = config.get('gsp_info', {}).get(fallback_year)
            if fallback_url:
                logger.info(f"GSP info for {FES_year} not available, trying {fallback_year} from API")
                try:
                    time.sleep(1)  # polite delay
                    response = http.get(fallback_url, timeout=30)
                    response.raise_for_status()
                    GSP_info = pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
                    logger.info(f"Downloaded GSP info from {fallback_year}: {GSP_info.shape[0]} rows")
                    break
                except Exception as e:
                    logger.debug(f"Failed to download GSP info for {fallback_year}: {e}")
                    continue
    
    # If all API downloads failed, raise an error
    if GSP_info is None:
        raise RuntimeError(f"Could not download GSP info for FES {FES_year} or any fallback year from NESO API. Check internet connection and configuration.")
    # remove any ' characters
    GSP_info['Name'] = GSP_info['Name'].str.replace("'", "")
    # .replace("’", '')
    GSP_info['Name'] = GSP_info['Name'].str.replace("’", '')

    GSP_info['Name'] = GSP_info['Name'].str.replace("'", '')

    # Create a mapping dictionary from Name to GSP ID
    name_to_gsp_mapping = dict(zip(GSP_info['Name'], GSP_info['GSP ID']))

    # New Node ID column to the corresponding GSP ID values without overwriting non-matching values
    raw_FES_data['GSP ID'] = raw_FES_data['GSP'].map(name_to_gsp_mapping, na_action='ignore')
    # # remove row where GSP contains Direct
    # raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Direct')]

    # read the Node ID to GSP ID key from "../data/network/ETYS/GB_network.xlsx", sheetname="Dem_per_node"
    node_info = pd.read_excel('data/network/ETYS/GB_network.xlsx', sheet_name="Dem_per_node")

    # Create a mapping dictionary from GSP Id to Node Id
    gsp_to_node_mapping = dict(zip(node_info['GSP Id'], node_info['Node Id']))

    raw_FES_data['Node ID'] = raw_FES_data['GSP ID'].map(gsp_to_node_mapping)
    missing_node_ids = raw_FES_data['Node ID'].isna().sum()
    logger.info(f"Mapped Node IDs; missing Node ID count: {missing_node_ids}")
    
    # If 'GSP' contains 'Direct', copy the 'GSP' value to both 'GSP ID' and 'Node ID'
    direct_mask = raw_FES_data['GSP'].str.contains('Direct', na=False)
    raw_FES_data.loc[direct_mask, 'Node ID'] = raw_FES_data.loc[direct_mask, 'GSP']
    raw_FES_data.loc[direct_mask, 'GSP ID'] = raw_FES_data.loc[direct_mask, 'GSP']

    snk = globals().get('snakemake')
    outpath = snk.output[0] if snk else os.path.join('resources', 'FES', f'FES_{FES_year}_data.csv')
    logger.info(f"Writing output FES file to {outpath}")
    raw_FES_data.to_csv(outpath, index=True)

if __name__ == "__main__":
    # Get FES year from Snakemake params
    snk = globals().get('snakemake')
    
    # Reinitialize logger with Snakemake log path if available
    if snk and hasattr(snk, 'log') and snk.log:
        logger = setup_logging(snk.log[0])
    
    if snk:
        FES_year = snk.params.fes_year
    else:
        FES_year = 2024  # Default for standalone testing
    try:
        logger.info(f"Starting FES data pipeline for year {FES_year}")
        FES_data = raw_FES_data(FES_year)
        add_node_id_to_gsp(FES_year, FES_data)
        logger.info("FES data pipeline finished successfully")
    except Exception as e:
        logger.exception(f"FES data pipeline failed: {e}")
        raise

