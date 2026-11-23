#!/usr/bin/env python3
"""
Add Interconnectors to PyPSA Network
====================================

This script integrates interconnectors into PyPSA networks as DC links with
proper representation of European electricity supply.

Architecture:
- External buses represent connected European bidding zones
- Future cables use paired, one-way import and export links so losses are
  applied in the physical direction of flow
- Foreign supply generators and signed export sinks represent a price-taking
  external market with a configurable bid/ask spread
- A shared-capacity callback prevents both directional links from independently
  using the full cable rating
- For historical scenarios, flows are fixed using actual ESPENI data

This architecture ensures:
1. European supply is properly costed (not an infinite free source)
2. Optimization correctly balances GB generation vs. imports
3. Transmission efficiency losses are modeled via link efficiency
4. Historical validation uses actual interconnector flows

Key features:
- PyPSA link integration with DC parameters
- External generators for European supply modeling
- Efficiency calculation from losses
- Availability profile integration
- Bidirectional market coupling using paired directional links
- Historical flow fixing for validation scenarios
- Network validation and consistency checking

Author: PyPSA-GB Team
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pypsa
import logging
import time
import warnings
import os

# Suppress PyPSA warnings about unoptimized networks (expected during network building)
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network
from scripts.utilities.calendar_year import fill_leap_day
from scripts.CP30.storage_interconnectors import load_cp30_interconnector_capacity_mw
from scripts.interconnectors.market_coupling import EXPORT_BID_FLOOR_COLUMN, floor_export_bid

try:
    from scripts.utilities.logging_config import setup_logging, log_execution_summary
    from scripts.utilities.carrier_definitions import add_carriers_to_network
except ImportError:
    import logging
    def setup_logging(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

# Check if running in Snakemake context
if 'snakemake' in globals():
    SNAKEMAKE_MODE = True
    input_network = snakemake.input.network
    input_mapped = snakemake.input.interconnectors
    # Optional inputs based on scenario type
    input_availability = snakemake.input.get('availability', None)
    input_price_diff = snakemake.input.get('price_differentials', None)
    input_historical_flows = snakemake.input.get('historical_flows', None)
    input_historical_flows_metadata = snakemake.input.get('historical_flows_metadata', None)
    input_entsoe_prices = snakemake.input.get('entsoe_prices', None)
    input_cp30_es1 = snakemake.input.get('cp30_es1', None)
    input_cp30_interconnector_prices = snakemake.input.get('cp30_interconnector_prices', None)
    input_market_demand = snakemake.input.get('market_demand', None)
    # Task Y6b (FES 2050 plan): an HT50_* scenario declares the FES file whose
    # Gen_BB022 total the cables are scaled to; absent = the FES 2024 default.
    input_fes_capacity_data = snakemake.input.get('fes_capacity_data', None)
    # CP30 cable composition (interconnectors.cp30_composition.file, 15 Oct 2026); future CP30 builds.
    input_cp30_composition = snakemake.input.get('cp30_interconnector_composition', None)
    output_network = snakemake.output[0]
    is_historical = snakemake.params.get('is_historical', False)
    modelled_year = snakemake.params.get('modelled_year', None)
    fes_pathway = snakemake.params.get('fes_pathway', None)
    scenario_source = snakemake.params.get('scenario_source', None)
    cp30_pathway = snakemake.params.get('cp30_pathway', None)
    pricing_config = snakemake.params.get('pricing_config', {})
    # Historical flows are fixed by default. The market_coupled alternative is
    # deliberately opt-in and is used only to validate model-selected flows
    # against the separate ESPENI reference file.
    historical_flow_mode = str(
        snakemake.params.get('historical_flow_mode', 'fixed')
    ).strip().lower()
    renewables_year = snakemake.params.get('renewables_year', None)
else:
    SNAKEMAKE_MODE = False
    modelled_year = None
    fes_pathway = None
    scenario_source = None
    cp30_pathway = None
    input_cp30_es1 = None
    input_cp30_interconnector_prices = None
    input_market_demand = None
    input_fes_capacity_data = None
    input_cp30_composition = None
    pricing_config = {}
    historical_flow_mode = 'fixed'
    renewables_year = None
    input_entsoe_prices = None
    input_historical_flows_metadata = None

# Optional: force-zero interconnectors for diagnostics
FORCE_ZERO_INTERCONNECTORS = os.environ.get("ZERO_INTERCONNECTORS", "0").lower() in ("1", "true", "yes")
ZEROED_CAPACITY_MW = 1e-3  # tiny non-zero to keep time series/variables present

# =============================================================================
# Name Normalization for ESPENI Column Matching
# =============================================================================
# Interconnector names in CSV may differ from ESPENI column names
# This mapping ensures correct matching between metadata and flow data
ESPENI_NAME_MAPPING = {
    # Format: {CSV_name: ESPENI_column_name}
    'Britned': 'BritNed',
    'IFA Interconnector': 'IFA',
    'East West Interconnector': 'EastWest',
    'Nemo Link': 'Nemo',
    'Auchencrosh (interconnector CCT)': 'Moyle',
    'NS Link': 'NorthSeaLink',
    'Viking Link Denmark Interconnector': 'VikingLink',
    # These already match exactly:
    # 'IFA2', 'ElecLink', 'Greenlink'
    # These are not in ESPENI (future or no data):
    # 'Isle of Man Interconnector', 'NeuConnect Interconnector', 'NEMO', 'NSL'
}

def normalize_interconnector_name(name: str, available_columns: list = None) -> str:
    """
    Normalize interconnector name to match ESPENI column naming.
    
    Args:
        name: Original interconnector name from CSV
        available_columns: List of available ESPENI columns (for validation)
        
    Returns:
        Normalized name that should match ESPENI column
    """
    # First check direct mapping
    if name in ESPENI_NAME_MAPPING:
        normalized = ESPENI_NAME_MAPPING[name]
    else:
        # No mapping needed - use as-is
        normalized = name
    
    # If available columns provided, validate the match
    if available_columns is not None:
        if normalized not in available_columns:
            # Try case-insensitive match
            for col in available_columns:
                if col.lower() == normalized.lower():
                    return col
    
    return normalized

def load_availability_profiles(availability_file: str) -> pd.DataFrame:
    """
    Load interconnector availability profiles.
    
    Args:
        availability_file: Path to availability CSV file
        
    Returns:
        DataFrame with availability profiles
    """
    logger = logging.getLogger(__name__)
    
    if not Path(availability_file).exists():
        logger.warning(f"Availability file not found: {availability_file}")
        return pd.DataFrame()
    
    try:
        availability_df = pd.read_csv(availability_file)
        logger.info(f"Loaded availability profiles: {len(availability_df)} records")
        
        # Ensure time column is datetime
        if 'time' in availability_df.columns:
            availability_df['time'] = pd.to_datetime(availability_df['time'])
        
        return availability_df
        
    except Exception as e:
        logger.error(f"Error loading availability profiles: {e}")
        return pd.DataFrame()


def enrich_historical_commissioning_years(interconnectors_df: pd.DataFrame, metadata_file: str | None) -> pd.DataFrame:
    """Fill missing mapped commissioning years from ESPENI metadata.

    This is an availability registry, not an optimisation input. It is used only
    for explicit historical market-coupled back-tests, to prevent modern or
    unverified cables from entering an earlier model year.
    """
    logger = logging.getLogger(__name__)
    if not metadata_file or not Path(metadata_file).exists():
        raise FileNotFoundError(
            "Historical market-coupled mode requires ESPENI flow metadata: "
            f"{metadata_file}"
        )
    with open(metadata_file, encoding="utf-8") as handle:
        registry = json.load(handle).get("interconnectors", {})
    known_years = {
        str(name).lower(): item.get("commissioning_year")
        for name, item in registry.items()
    }
    result = interconnectors_df.copy()
    result["commissioning_year"] = pd.to_numeric(
        result.get("commissioning_year"), errors="coerce"
    )
    filled = []
    for index, row in result.loc[result["commissioning_year"].isna()].iterrows():
        canonical = ESPENI_NAME_MAPPING.get(str(row["name"]), str(row["name"]))
        year = known_years.get(canonical.lower())
        if year is not None:
            result.at[index, "commissioning_year"] = float(year)
            filled.append(f"{row['name']}={int(year)}")
    if filled:
        logger.info("Filled historical commissioning years from ESPENI metadata: %s", ", ".join(filled))
    return result


def restrict_market_coupled_historical_interconnectors(interconnectors_df: pd.DataFrame, scenario_year: int) -> pd.DataFrame:
    """Keep only cables with a source-backed commissioning year by scenario year."""
    logger = logging.getLogger(__name__)
    result = interconnectors_df.copy()
    # ESPENI sometimes has one series for a named legacy cable while the mapped
    # data also contains an auxiliary/alias record. Retain the exact-name record
    # and exclude aliases from a market-coupled calibration to avoid creating a
    # second independent cable from the same observation.
    result["_espeni_reference"] = result["name"].map(
        lambda name: ESPENI_NAME_MAPPING.get(str(name), str(name))
    )
    direct_reference = (
        result["name"].astype(str).str.lower()
        == result["_espeni_reference"].astype(str).str.lower()
    )
    has_direct_reference = direct_reference.groupby(
        result["_espeni_reference"].astype(str).str.lower()
    ).transform("any")
    unique_reference = direct_reference | ~has_direct_reference
    operational = (
        result["commissioning_year"].notna()
        & (result["commissioning_year"] <= scenario_year)
        & unique_reference
    )
    excluded = result.loc[~operational]
    if not excluded.empty:
        logger.info(
            "Historical market-coupled eligibility excludes %d cable(s) without a source-backed commissioning year by %d: %s",
            len(excluded), scenario_year, ", ".join(excluded["name"].astype(str)),
        )
    return interconnectors_df.loc[operational].copy()

#: Commissioning years of GB cables missing from DUKES 5.13 (data/interconnectors/DUKES_5.13_2025.csv).
#: East West Interconnector: EirGrid, commercial operation December 2012. Greenlink: Greenlink Interconnector
#: Ltd, commercial operation 2025.
DECLARED_COMMISSIONING_YEARS = {"East West Interconnector": 2012, "Greenlink": 2025}


def filter_interconnectors_by_commissioning_year(
    interconnectors_df: pd.DataFrame, 
    scenario_year: int,
    is_historical: bool = True
) -> pd.DataFrame:
    """
    Filter interconnectors to only include those operational by the scenario year.
    
    For historical scenarios, this ensures only interconnectors that were commissioned
    by the modelled year are included (e.g., 2020 scenario excludes IFA2, NSL, ElecLink, Viking).
    
    Args:
        interconnectors_df: DataFrame with interconnector data including 'commissioning_year'
        scenario_year: The year being modelled (e.g., 2020)
        is_historical: Whether this is a historical scenario (filtering only applies for historical)
        
    Returns:
        Filtered DataFrame containing only interconnectors operational by scenario_year
    """
    logger = logging.getLogger(__name__)
    
    if not is_historical:
        logger.info("Future scenario - no commissioning year filtering applied")
        return interconnectors_df
    
    if scenario_year is None:
        logger.warning("No scenario year provided - cannot filter by commissioning year")
        return interconnectors_df
    
    initial_count = len(interconnectors_df)
    
    # Check if commissioning_year column exists
    if 'commissioning_year' not in interconnectors_df.columns:
        logger.warning("No 'commissioning_year' column found - cannot filter by commissioning year")
        return interconnectors_df
    
    # Cables the DUKES 5.13 table omits: their commissioning years, declared (19 Nov 2026: Greenlink, which
    # opened in 2025, was in the 2015 Reduced network because its year was blank).
    interconnectors_df = interconnectors_df.copy()
    years = pd.to_numeric(interconnectors_df['commissioning_year'], errors='coerce')
    for name, year in DECLARED_COMMISSIONING_YEARS.items():
        rows = interconnectors_df['name'].astype(str).str.strip().str.lower().eq(name.lower()) & years.isna()
        years.loc[rows] = year
    interconnectors_df['commissioning_year'] = years
    unknown = interconnectors_df.loc[years.isna(), 'name'].astype(str).tolist()
    if unknown:
        logger.warning(f"No commissioning year for {unknown}: kept in the {scenario_year} network")

    # Filter: keep if commissioning_year <= scenario_year OR commissioning_year is NaN
    # (NaN means unknown, so we include it by default)
    mask = (
        interconnectors_df['commissioning_year'].isna() | 
        (interconnectors_df['commissioning_year'] <= scenario_year)
    )
    
    filtered_df = interconnectors_df[mask].copy()
    excluded_df = interconnectors_df[~mask]
    
    if len(excluded_df) > 0:
        logger.info(f"=== COMMISSIONING YEAR FILTER (scenario year: {scenario_year}) ===")
        for _, row in excluded_df.iterrows():
            logger.info(f"  EXCLUDED: {row['name']} (commissioned {int(row['commissioning_year'])}, after {scenario_year})")
    
    logger.info(f"Commissioning year filter: {initial_count} → {len(filtered_df)} interconnectors "
                f"(excluded {initial_count - len(filtered_df)} post-{scenario_year} interconnectors)")
    
    return filtered_df


def calculate_link_efficiency(losses_percent: float) -> float:
    """
    Calculate link efficiency from losses percentage.
    
    Args:
        losses_percent: Losses as percentage (0-100)
        
    Returns:
        Efficiency factor (0-1)
    """
    if pd.isna(losses_percent) or losses_percent < 0:
        losses_percent = 2.5  # Default 2.5% losses
    
    # Ensure losses are reasonable
    losses_percent = min(losses_percent, 50.0)  # Cap at 50%
    
    efficiency = 1.0 - (losses_percent / 100.0)
    return max(efficiency, 0.1)  # Minimum 10% efficiency

def add_external_buses(network: pypsa.Network, interconnectors_df: pd.DataFrame) -> None:
    """
    Add external buses for interconnector endpoints with coordinates.
    
    CRITICAL: External bus coordinates MUST match the network's coordinate system.
    ETYS networks use OSGB36 (British National Grid in meters), so we convert
    WGS84 lat/lon coordinates to OSGB36 before adding buses.
    
    Args:
        network: PyPSA network object
        interconnectors_df: Mapped interconnector data with international_latitude/longitude
    """
    logger = logging.getLogger(__name__)
    
    # Import coordinate conversion utilities
    try:
        from scripts.utilities.spatial_utils import get_bus_coordinates_for_external, detect_coordinate_system
    except ImportError:
        # Fallback if spatial_utils not available
        from pyproj import Transformer
        def get_bus_coordinates_for_external(lon, lat, network):
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            return transformer.transform(lon, lat)
        def detect_coordinate_system(x, y):
            return 'OSGB36' if max(x) > 1000 else 'WGS84'
    
    # Log the network's coordinate system for debugging
    if not network.buses.empty:
        coord_sys = detect_coordinate_system(
            network.buses['x'].values, 
            network.buses['y'].values
        )
        logger.info(f"Network coordinate system: {coord_sys}")
    
    # Get unique external buses
    external_buses = interconnectors_df['to_bus'].unique()
    existing_buses = set(network.buses.index)
    
    buses_to_add = []
    for bus_name in external_buses:
        if bus_name not in existing_buses:
            # Find the interconnector(s) using this bus
            ic_rows = interconnectors_df[interconnectors_df['to_bus'] == bus_name]
            
            if len(ic_rows) > 0:
                # Get international coordinates from first interconnector using this bus
                ic = ic_rows.iloc[0]
                int_lat = ic.get('international_latitude', None)
                int_lon = ic.get('international_longitude', None)
                country = ic.get('counterparty_country', 'External')
                
                # Convert WGS84 coordinates to match network coordinate system
                if int_lat is not None and int_lon is not None and not pd.isna(int_lat) and not pd.isna(int_lon):
                    x, y = get_bus_coordinates_for_external(float(int_lon), float(int_lat), network)
                    logger.info(f"External bus {bus_name}: WGS84({int_lat:.4f}, {int_lon:.4f}) → OSGB36({x:.0f}, {y:.0f})")
                else:
                    # No coordinates available - place at center of GB (approximate)
                    x, y = 400000, 500000  # Central England in OSGB36
                    logger.warning(f"External bus {bus_name}: No international coordinates, using default ({x}, {y})")
            else:
                x, y = 400000, 500000
                country = 'External'
            
            buses_to_add.append({
                'Bus': bus_name,
                'x': x,
                'y': y,
                'v_nom': 400,  # Assume 400kV for HVDC
                'country': country,
                'carrier': 'AC'
            })
    
    if buses_to_add:
        bus_df = pd.DataFrame(buses_to_add).set_index('Bus')
        # Replace deprecated import_components_from_dataframe with modern n.add
        for bus_name, row in bus_df.iterrows():
            network.add('Bus', 
                       name=bus_name,
                       x=row['x'],
                       y=row['y'],
                       v_nom=row.get('v_nom', 400),
                       country=row.get('country', 'External'),
                       carrier=row.get('carrier', 'AC'))
        logger.info(f"Added {len(buses_to_add)} external buses with OSGB36 coordinates (consistent with ETYS network)")
    else:
        logger.info("No new external buses needed")

def load_price_differentials(price_diff_file: str) -> pd.DataFrame:
    """
    Load European price differential data.
    
    Args:
        price_diff_file: Path to price differentials CSV file
        
    Returns:
        DataFrame with price differentials by country, scenario, year
    """
    logger = logging.getLogger(__name__)
    
    if not Path(price_diff_file).exists():
        logger.warning(f"Price differential file not found: {price_diff_file}")
        logger.info("Interconnectors will be added without marginal cost profiles")
        return pd.DataFrame()
    
    try:
        price_df = pd.read_csv(price_diff_file)
        logger.info(f"Loaded price differentials: {len(price_df)} records")
        logger.info(f"  Countries: {price_df['country'].nunique()}")
        logger.info(f"  Years: {price_df['year'].min()}-{price_df['year'].max()}")
        
        return price_df
        
    except Exception as e:
        logger.error(f"Error loading price differentials: {e}")
        return pd.DataFrame()


def load_cp30_interconnector_prices(price_file: str) -> pd.DataFrame:
    """Load the CP.29 pathway-level interconnector trade-price assumptions."""
    if not price_file or not Path(price_file).exists():
        raise FileNotFoundError(f"CP30 interconnector price file not found: {price_file}")
    prices = pd.read_csv(price_file)
    required = {"year", "pathway", "import_price_gbp_per_mwh", "export_price_gbp_per_mwh"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"CP30 interconnector price file missing columns: {sorted(missing)}")
    return prices

def load_entsoe_prices(entsoe_file: str) -> pd.DataFrame:
    """
    Load ENTSO-E day-ahead price CSV produced by fetch_entsoe_prices.py.

    Args:
        entsoe_file: Path to CSV (datetime index, one column per country in £/MWh).

    Returns:
        DataFrame with DatetimeIndex and country columns, or empty DataFrame on failure.
    """
    logger = logging.getLogger(__name__)

    if not entsoe_file or not Path(entsoe_file).exists():
        logger.warning(f"ENTSO-E price file not found: {entsoe_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(entsoe_file, index_col=0, parse_dates=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        logger.info(f"Loaded ENTSO-E prices: {len(df)} hours × {len(df.columns)} countries")
        return df
    except Exception as e:
        logger.error(f"Error loading ENTSO-E prices: {e}")
        return pd.DataFrame()


def _apply_time_varying_mc(
    network: pypsa.Network,
    gen_name: str,
    hourly_prices: pd.Series,
    logger: logging.Logger,
) -> None:
    """Set time-varying marginal cost on a generator using generators_t.

    If ``generators_t['marginal_cost']`` doesn't exist yet it is created;
    otherwise the new column is added to the existing DataFrame.

    Parameters
    ----------
    network : pypsa.Network
    gen_name : str
        Generator name (must already exist in `
etwork.generators``).
    hourly_prices : pd.Series
        Hourly prices (£/MWh) with a DatetimeIndex.  Will be reindexed to
        `
etwork.snapshots``; missing values are forward-filled then
        back-filled, then filled with the series mean.
    logger : logging.Logger
    """
    # Align year if ENTSO-E data year differs from network snapshot year
    # (e.g. future scenarios use 2023 weather-year prices for 2035 snapshots)
    snap_year = network.snapshots[0].year
    price_year = hourly_prices.index[0].year
    if snap_year != price_year:
        logger.debug(
            f"  {gen_name}: shifting ENTSO-E price index "
            f"{price_year} → {snap_year}"
        )
        import calendar

        target_is_leap = calendar.isleap(snap_year)

        def _shift_ts(ts):
            if ts.month == 2 and ts.day == 29 and not target_is_leap:
                # Feb 29 doesn't exist in target year — map to Feb 28
                return ts.replace(year=snap_year, day=28)
            return ts.replace(year=snap_year)

        shifted_index = pd.DatetimeIndex([_shift_ts(ts) for ts in hourly_prices.index])
        # Remove any duplicates introduced by the Feb-28/29 collapse
        hourly_prices = hourly_prices.copy()
        hourly_prices.index = shifted_index
        hourly_prices = hourly_prices[~hourly_prices.index.duplicated(keep="first")]
        # A leap target year from a source without 29 February: 28 February's hours, not a gap fill.
        hourly_prices = fill_leap_day(hourly_prices)

    # Reindex to network snapshots
    prices = hourly_prices.reindex(network.snapshots)

    n_missing = prices.isna().sum()
    if n_missing > 0:
        logger.debug(f"  {gen_name}: {n_missing}/{len(prices)} snapshots missing — gap-filling")
        prices = prices.interpolate(method="linear", limit=6, limit_direction="both")
        prices = prices.ffill().bfill()
        if prices.isna().any():
            prices = prices.fillna(prices.mean())

    # Ensure generators_t.marginal_cost DataFrame exists
    mc_key = "marginal_cost"
    if mc_key not in network.generators_t or network.generators_t[mc_key].empty:
        network.generators_t[mc_key] = pd.DataFrame(index=network.snapshots)

    network.generators_t[mc_key][gen_name] = prices.values

    logger.info(
        f"  {gen_name}: time-varying MC set — "
        f"mean £{prices.mean():.2f}/MWh, "
        f"range £{prices.min():.2f}–£{prices.max():.2f}/MWh"
    )


def _apply_time_varying_link_mc(
    network: pypsa.Network,
    link_name: str,
    hourly_costs: pd.Series,
    logger: logging.Logger,
) -> None:
    """Set a time-varying marginal cost directly on a Link."""
    costs = hourly_costs.copy()
    snap_year = network.snapshots[0].year
    cost_year = costs.index[0].year
    if snap_year != cost_year:
        import calendar

        target_is_leap = calendar.isleap(snap_year)

        def _shift_ts(ts):
            if ts.month == 2 and ts.day == 29 and not target_is_leap:
                return ts.replace(year=snap_year, day=28)
            return ts.replace(year=snap_year)

        costs.index = pd.DatetimeIndex([_shift_ts(ts) for ts in costs.index])
        costs = costs[~costs.index.duplicated(keep="first")]
        costs = fill_leap_day(costs)

    costs = costs.reindex(network.snapshots)
    if costs.isna().any():
        costs = costs.interpolate(method="linear", limit=6, limit_direction="both")
        costs = costs.ffill().bfill()
        if costs.isna().any():
            costs = costs.fillna(costs.mean())

    if "marginal_cost" not in network.links_t or network.links_t.marginal_cost.empty:
        network.links_t["marginal_cost"] = pd.DataFrame(index=network.snapshots)
    network.links_t.marginal_cost[link_name] = costs.to_numpy(dtype=float)
    network.links_t.marginal_cost.columns.name = "name"
    logger.info(
        "  %s: time-varying link MC set - mean %.2f GBP/MWh, range %.2f to %.2f",
        link_name,
        float(costs.mean()),
        float(costs.min()),
        float(costs.max()),
    )
def load_historical_flows(historical_flows_file: str) -> pd.DataFrame:
    """
    Load historical interconnector flows from ESPENI data.
    
    Args:
        historical_flows_file: Path to historical flows CSV file
        
    Returns:
        DataFrame with datetime index and interconnector flow columns (MW)
    """
    logger = logging.getLogger(__name__)
    
    if not Path(historical_flows_file).exists():
        raise FileNotFoundError(
            f"Historical flows file not found: {historical_flows_file}"
        )
    
    try:
        flows_df = pd.read_csv(historical_flows_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded historical interconnector flows: {len(flows_df)} timesteps")
        logger.info(f"  Date range: {flows_df.index.min()} to {flows_df.index.max()}")
        logger.info(f"  Interconnectors: {', '.join(flows_df.columns)}")
        
        # Calculate total net imports
        total_net_import = flows_df.sum().sum() * 0.5  # Half-hourly to MWh
        total_net_import_twh = total_net_import / 1e6
        logger.info(f"  Total net import: {total_net_import_twh:.2f} TWh")
        
        return flows_df
        
    except Exception as e:
        logger.error(f"Error loading historical flows: {e}")
        raise


def scale_interconnectors_to_fes(interconnectors_df: pd.DataFrame,
                                  modelled_year: int,
                                  fes_pathway: str,
                                  fes_data_path: str = None) -> pd.DataFrame:
    """
    Scale existing interconnector capacities to match FES projections for future scenarios.
    
    FES provides total interconnector capacity projections by year. This function
    uniformly scales existing interconnector capacities to match the FES total while
    preserving the relative distribution between countries.
    
    Args:
        interconnectors_df: DataFrame with mapped interconnector data including capacity_mw
        modelled_year: Target year (e.g., 2035, 2050)
        fes_pathway: FES pathway name (e.g., 'Holistic Transition')
        fes_data_path: Path to FES data CSV (defaults to resources/FES/FES_2024_data.csv)
        
    Returns:
        DataFrame with scaled capacity_mw column
    """
    logger = logging.getLogger(__name__)
    
    if modelled_year is None or modelled_year <= 2024:
        logger.info("Historical scenario - not scaling interconnectors to FES")
        return interconnectors_df
    
    if fes_pathway is None:
        logger.warning("No FES pathway specified - cannot scale interconnectors to FES targets")
        return interconnectors_df
    
    # Default FES data path
    if fes_data_path is None:
        fes_data_path = project_root / "resources" / "FES" / "FES_2024_data.csv"
    
    if not Path(fes_data_path).exists():
        logger.warning(f"FES data file not found: {fes_data_path} - cannot scale interconnectors")
        return interconnectors_df
    
    try:
        # Load FES data
        fes_df = pd.read_csv(fes_data_path)
        
        # Filter to Gen_BB022 (Interconnectors) and target pathway
        ic_fes = fes_df[
            (fes_df['Building Block ID Number'] == 'Gen_BB022') &
            (fes_df['FES Pathway'] == fes_pathway)
        ]
        
        if len(ic_fes) == 0:
            logger.warning(f"No FES interconnector data found for pathway '{fes_pathway}'")
            return interconnectors_df
        
        # Get the target year column
        year_col = str(modelled_year)
        if year_col not in fes_df.columns:
            logger.warning(f"Year {modelled_year} not found in FES data columns")
            return interconnectors_df
        
        # Get FES target capacity (sum across all GSPs/regions)
        fes_target_mw = ic_fes[year_col].sum()
        
        # Calculate current total capacity
        current_total_mw = interconnectors_df['capacity_mw'].sum()
        
        if current_total_mw <= 0:
            logger.warning("No current interconnector capacity to scale")
            return interconnectors_df
        
        # Calculate scaling factor
        scale_factor = fes_target_mw / current_total_mw
        
        logger.info(f"=== Scaling Interconnectors to FES {fes_pathway} {modelled_year} ===")
        logger.info(f"  Current total capacity: {current_total_mw:.0f} MW")
        logger.info(f"  FES target capacity:    {fes_target_mw:.0f} MW")
        logger.info(f"  Scale factor:           {scale_factor:.3f}x")
        
        # Apply scaling
        scaled_df = interconnectors_df.copy()
        scaled_df['capacity_mw_original'] = scaled_df['capacity_mw']
        scaled_df['capacity_mw'] = scaled_df['capacity_mw'] * scale_factor
        
        # Log individual scaling
        for idx, row in scaled_df.iterrows():
            logger.debug(f"  {row['name']}: {row['capacity_mw_original']:.0f} MW → {row['capacity_mw']:.0f} MW")
        
        logger.info(f"  Scaled {len(scaled_df)} interconnectors by {scale_factor:.2f}x")
        
        return scaled_df
        
    except Exception as e:
        logger.error(f"Error scaling interconnectors to FES: {e}")
        return interconnectors_df


def apply_cp30_interconnector_composition(
    interconnectors_df: pd.DataFrame,
    network: pypsa.Network,
    composition_path,
    cp30_es1_path: str,
    cp30_pathway: str,
    modelled_year: int,
) -> pd.DataFrame:
    """CP30's cables from the declared composition, checked against the ES1 total (15 Oct 2026).

    Replaced the pro-rata uplift (every cable x 1.0559 to reach 12,450 MW in 2030): kept cables
    take CP30's nominal ratings, an ``add`` row (MaresConnect, Bodelwyddan, 750 MW) joins the set
    priced on its counterparty market like any mapped cable, and ``exclude`` rows (the Isle of Man
    cable) leave it. An added cable whose GB bus is not in this network (a Reduced or Zonal build)
    lands on the nearest AC bus to its GB coordinates.
    """
    logger = logging.getLogger(__name__)
    if isinstance(composition_path, (list, tuple)):
        composition_path = composition_path[0]
    if not cp30_es1_path or not cp30_pathway:
        raise ValueError("CP30 interconnector composition needs the cp30_es1 input and the CP30 pathway")
    from scripts.CP30.storage_interconnectors import (
        compose_cp30_interconnectors, load_cp30_interconnector_composition)
    composition = load_cp30_interconnector_composition(composition_path)
    target_mw = load_cp30_interconnector_capacity_mw(cp30_es1_path, cp30_pathway, modelled_year)
    before = float(pd.to_numeric(interconnectors_df["capacity_mw"], errors="coerce").fillna(0.0).sum())
    composed = compose_cp30_interconnectors(interconnectors_df, target_mw, composition, modelled_year)
    added = composed["cp30_composition_action"].eq("add")
    for idx in composed.index[added & ~composed["from_bus"].isin(network.buses.index)]:
        from scripts.utilities.spatial_utils import get_bus_coordinates_for_external
        candidates = network.buses[~network.buses.index.astype(str).str.contains("External", case=False)]
        candidates = candidates[candidates.get("carrier", pd.Series("AC", index=candidates.index)).fillna("AC").eq("AC")]
        x, y = get_bus_coordinates_for_external(float(composed.at[idx, "gb_longitude"]),
                                                float(composed.at[idx, "gb_latitude"]), network)
        nearest = np.hypot(candidates["x"].astype(float) - x, candidates["y"].astype(float) - y).idxmin()
        logger.info(f"  {composed.at[idx, 'name']}: GB bus {composed.at[idx, 'from_bus']} not in this network, "
                    f"landed at nearest bus {nearest}")
        composed.at[idx, "from_bus"] = nearest
        composed.at[idx, "gb_bus"] = nearest
    excluded = sorted(set(interconnectors_df["name"]) - set(composed["name"]))
    logger.info(f"=== CP30 interconnector composition, {cp30_pathway} {modelled_year} ===")
    logger.info(f"  Mapped register total {before:,.0f} MW -> composed {composed['capacity_mw'].sum():,.0f} MW "
                f"(CP30 ES1 {target_mw:,.0f} MW); no pro-rata scaling")
    for _, row in composed.iterrows():
        logger.info(f"  {row['name']} ({row['counterparty_country']}, {row['from_bus']}): "
                    f"{row['capacity_mw']:,.0f} MW [{row['cp30_composition_action']}; register "
                    f"{row['capacity_mw_original']:,.0f}]")
    if excluded:
        logger.info(f"  Outside the CP30 count, not built: {excluded}")
    return composed


# Country name mapping for European price data
# Maps interconnector country names to price differential country names
COUNTRY_PRICE_MAPPING = {
    'ireland': 'sem',
    'northern ireland': 'sem',
    'isle of man': 'sem',  # Use SEM prices as proxy for Isle of Man
}

ENTSOE_COUNTRY_MAPPING = {
    'northern ireland': 'Ireland',
    'isle of man': 'Ireland',
}

#: ``interconnectors.pricing.foreign_price_source`` values whose price file is used
#: at its own level, with no CP.29 re-centring: the 2050 TYNDP shape (task Y3) and
#: TYNDP NT2030 rebased to CP30's fuel (item F13).
OWN_LEVEL_PRICE_SOURCES = ("tyndp", "cp30_basis")

def _case_insensitive_column(df, requested):
    '''Return a case-insensitive column match.'''
    wanted = str(requested).strip().lower()
    for column in df.columns:
        if str(column).strip().lower() == wanted:
            return column
    return None


def _apply_price_case(target, country, pricing):
    """Apply configured price sensitivities and country overrides."""
    case = str(pricing.get("future_price_case", "central")).lower()
    multipliers = pricing.get(
        "price_case_multipliers", {"low": 0.75, "central": 1.0, "high": 1.25}
    )
    target = float(target) * float(multipliers.get(case, 1.0))
    price_country = COUNTRY_PRICE_MAPPING.get(country.lower(), country.lower())
    overrides = pricing.get("country_price_overrides_gbp_per_mwh", {})
    case_overrides = overrides.get(case, {}) if isinstance(overrides, dict) else {}
    for key, value in case_overrides.items():
        if str(key).lower() in {country.lower(), price_country}:
            return float(value)
    return target


def _foreign_price_target(country, filtered_prices, pricing):
    """Return the annual import-price target for one foreign bidding zone."""
    if str(pricing.get("source", "derived")).lower() == "cp30":
        if "cp30_import_price_gbp_per_mwh" not in pricing:
            raise ValueError("CP30 interconnector pricing was selected but no CP.29 import price was loaded")
        return _apply_price_case(pricing["cp30_import_price_gbp_per_mwh"], country, pricing)

    price_country = COUNTRY_PRICE_MAPPING.get(country.lower(), country.lower())
    target = float(pricing.get("gb_price", 50.0))
    if not filtered_prices.empty and "country" in filtered_prices:
        rows = filtered_prices[
            filtered_prices["country"].astype(str).str.lower() == price_country
        ]
        if not rows.empty:
            candidate = pd.to_numeric(
                rows["estimated_price_gbp_per_mwh"], errors="coerce"
            ).mean()
            if pd.notna(candidate):
                target = float(candidate)
    return _apply_price_case(target, country, pricing)


def _foreign_export_price_target(country, filtered_prices, pricing):
    """Return the annual export-price target; CP.29 provides it separately."""
    if str(pricing.get("source", "derived")).lower() != "cp30":
        return _foreign_price_target(country, filtered_prices, pricing)
    if "cp30_export_price_gbp_per_mwh" not in pricing:
        raise ValueError("CP30 interconnector pricing was selected but no CP.29 export price was loaded")
    return _apply_price_case(pricing["cp30_export_price_gbp_per_mwh"], country, pricing)


def _prepare_historical_entsoe_prices(hourly, price_floor, price_cap):
    """Fill and clip actual prices without changing their observed level."""
    hourly = pd.to_numeric(hourly, errors="coerce").astype(float)
    if hourly.notna().sum() == 0:
        raise ValueError("Historical ENTSO-E price series contains no numeric observations")
    return hourly.interpolate(limit_direction="both").ffill().bfill().clip(price_floor, price_cap)

def _recenter_and_clip_prices(hourly, target_mean, price_floor, price_cap):
    '''Preserve an hourly shape while matching its clipped target mean.'''
    hourly = pd.to_numeric(hourly, errors='coerce').astype(float)
    if hourly.notna().sum() == 0:
        return pd.Series(target_mean, index=hourly.index, dtype=float)
    hourly = hourly.interpolate(limit_direction='both').ffill().bfill()
    deviations = hourly - hourly.mean()
    target = min(max(float(target_mean), price_floor), price_cap)
    low = price_floor - float(deviations.max()) - abs(price_cap - price_floor)
    high = price_cap - float(deviations.min()) + abs(price_cap - price_floor)
    for _ in range(60):
        midpoint = (low + high) / 2.0
        mean = deviations.add(midpoint).clip(price_floor, price_cap).mean()
        if mean < target:
            low = midpoint
        else:
            high = midpoint
    return deviations.add((low + high) / 2.0).clip(price_floor, price_cap)


def _foreign_bid_ask_prices(reference, efficiency, pricing):
    '''Apply a bid/ask spread that keeps the signed-link cost curve convex.

    The export slope is ``efficiency**2 x bid`` (item D2): a signed PyPSA link
    running backwards delivers ``1/efficiency`` of what GB sends, so the loss
    is charged on the price. Convexity needs ``ask >= efficiency**2 x bid``.
    '''
    base_half = max(
        float(pricing.get('transaction_spread_gbp_per_mwh', 1.0)) / 2.0,
        0.0,
    )
    epsilon = max(
        float(pricing.get('circulation_epsilon_gbp_per_mwh', 0.01)),
        0.0,
    )
    eta = max(float(efficiency), 0.01) ** 2
    factor = (1.0 - eta) / (1.0 + eta)
    if isinstance(reference, pd.Series):
        required = (-reference).clip(lower=0.0) * factor + epsilon
        half_spread = required.clip(lower=base_half)
        return reference + half_spread, reference - half_spread
    required = max(-float(reference), 0.0) * factor + epsilon
    half_spread = max(base_half, required)
    return float(reference) + half_spread, float(reference) - half_spread


def single_foreign_price_basis(pricing) -> bool:
    """True when each foreign market trades at ONE hourly price (item D2)."""
    return str((pricing or {}).get("foreign_price_basis", "separate")).strip().lower() == "single"


def single_price_market_of(country, pricing) -> str:
    """The coupled market (volume-tranche market) a counterparty's level offset belongs to."""
    markets = ((pricing.get("volume_tranches") or {}).get("markets") or {})
    wanted = str(country).strip().lower()
    for market, spec in markets.items():
        if wanted in {str(c).strip().lower() for c in spec.get("counterparties", [])}:
            return str(market)
    raise ValueError(
        f"single foreign price: counterparty {country!r} belongs to no market in "
        "interconnectors.pricing.volume_tranches.markets"
    )


def single_price_market_shapes(entsoe_prices_df, interconnectors_df, pricing):
    """One hourly shape per coupled market: the GB-capacity-weighted mean of its zones' prices.

    France, Belgium, the Netherlands, Germany and Denmark clear in one coupled
    continental market, so GB trades with them at one price; separate zone
    prices would let GB wheel energy between them (import from France, export
    to the Netherlands), which the continent's own borders do directly.
    Weights are GB's cable capacity to each zone.
    """
    shapes = {}
    capacity = interconnectors_df.groupby("counterparty_country")["capacity_mw"].sum()
    by_market = {}
    for country, mw in capacity.items():
        by_market.setdefault(single_price_market_of(country, pricing), []).append((country, float(mw)))
    for market, members in by_market.items():
        total, weight = None, 0.0
        for country, mw in members:
            column = _case_insensitive_column(
                entsoe_prices_df, ENTSOE_COUNTRY_MAPPING.get(str(country).lower(), country))
            if column is None:
                raise ValueError(f"single foreign price: no ENTSO-E column for {country}")
            series = pd.to_numeric(entsoe_prices_df[column], errors="coerce").astype(float)
            series = series.interpolate(limit_direction="both").ffill().bfill()
            total = series * mw if total is None else total + series * mw
            weight += mw
        if weight <= 0:
            raise ValueError(f"single foreign price: market {market!r} has no cable capacity")
        shapes[market] = total / weight
    return shapes


def single_price_link_pricing(pricing):
    """Pricing for a single-basis link: ``single_price_transaction_spread_gbp_per_mwh`` if set.

    The single basis's spread is a calibrated trading hurdle (item D2), set with the
    level offset; unset (null) it falls back to ``transaction_spread_gbp_per_mwh``.
    """
    hurdle = pricing.get("single_price_transaction_spread_gbp_per_mwh")
    if hurdle is None:
        return pricing
    if float(hurdle) < 0:
        raise ValueError("single_price_transaction_spread_gbp_per_mwh must be non-negative")
    return dict(pricing, transaction_spread_gbp_per_mwh=float(hurdle))


def single_price_export_bid_floor(pricing):
    """The single basis's export-bid floor (GBP/MWh), or None when off.

    ``single_price_export_bid_floor_gbp_per_mwh``: where the foreign price is
    positive the export bid is at least ``min(foreign price, floor)``, so the
    trading hurdle never stops GB exporting its surplus
    (``market_coupling.floor_export_bid``). Imports keep the full hurdle.
    """
    floor = pricing.get("single_price_export_bid_floor_gbp_per_mwh")
    if floor is None:
        return None
    if float(floor) < 0:
        raise ValueError("single_price_export_bid_floor_gbp_per_mwh must be non-negative")
    return float(floor)


def single_price_reference(hourly, country, pricing, price_floor, price_cap):
    """One hourly foreign price: the market's ENTSO-E (conditioned) shape plus its calibrated offset.

    ``price = market shape + offset[market]``: the shape year's own level is
    kept and one offset per coupled market
    (``single_price_level_offsets_gbp_per_mwh``) moves it. The offsets are
    calibrated so the model's gross trade matches CP30 (see
    config/defaults/method/brownfield.yaml ``interconnectors.pricing``). The configured price case scales
    the resulting annual level and a country override replaces it, as for the
    separate CP.29 basis. ``hourly`` is the market shape
    (``single_price_market_shapes``).
    """
    offsets = pricing.get("single_price_level_offsets_gbp_per_mwh") or {}
    market = single_price_market_of(country, pricing)
    if market not in offsets:
        raise ValueError(f"single foreign price: no level offset for market {market!r} ({country})")
    hourly = pd.to_numeric(hourly, errors="coerce").astype(float)
    if hourly.notna().sum() == 0:
        raise ValueError(f"single foreign price: no hourly shape observations for {country}")
    hourly = hourly.interpolate(limit_direction="both").ffill().bfill()
    shape_mean = float(hourly.mean())
    target = _apply_price_case(shape_mean + float(offsets[market]), country, pricing)
    return hourly.add(target - shape_mean).clip(price_floor, price_cap)


def _separate_import_export_prices(import_reference, export_reference, efficiency, pricing):
    """Construct a convex bid/ask pair from separately sourced trade prices."""
    if import_reference is export_reference:
        return _foreign_bid_ask_prices(import_reference, efficiency, pricing)
    half_spread = max(
        float(pricing.get("transaction_spread_gbp_per_mwh", 1.0)) / 2.0,
        0.0,
    )
    epsilon = max(float(pricing.get("circulation_epsilon_gbp_per_mwh", 0.01)), 0.0)
    eta = max(float(efficiency), 0.01) ** 2   # export slope efficiency**2 x bid (item D2)
    import_ask = import_reference + half_spread
    export_bid = export_reference - half_spread
    maximum_export_bid = (import_ask - epsilon) / eta
    if isinstance(export_bid, pd.Series):
        export_bid = pd.concat([export_bid, maximum_export_bid], axis=1).min(axis=1)
    else:
        export_bid = min(float(export_bid), float(maximum_export_bid))
    return import_ask, export_bid

def _add_external_generators_legacy(network: pypsa.Network,
                           interconnectors_df: pd.DataFrame,
                           price_differentials_df: pd.DataFrame = None,
                           modelled_year: int = None,
                           fes_pathway: str = None,
                           entsoe_prices_df: pd.DataFrame = None,
                           pricing_config: dict = None) -> None:
    """
    Add generators on external buses to represent European electricity supply.
    
    This models the European generation stack as a generator at each external bus
    with a marginal cost representing the European wholesale price for the specific
    modelled year and FES pathway. Generator capacity is limited to the total
    interconnector capacity to each country, providing a realistic constraint.
    
    When *entsoe_prices_df* is provided the MC becomes **time-varying**: hourly
    ENTSO-E day-ahead prices are scaled so the annual mean matches the FES-derived
    price, preserving the hourly shape (weather-correlated).
    
    Args:
        network: PyPSA network object
        interconnectors_df: Mapped interconnector data with external buses
        price_differentials_df: European price differentials (optional)
        modelled_year: The year being modelled (for filtering price data)
        fes_pathway: The FES pathway name (e.g., 'Holistic Transition')
        entsoe_prices_df: Hourly ENTSO-E day-ahead prices (optional).
            Columns are country names (France, Belgium, …), index is DatetimeIndex.
    """
    logger = logging.getLogger(__name__)
    pricing_config = pricing_config or {}
    
    # Calculate total interconnector capacity by country
    # This will be used to set realistic EU_supply generator capacities
    country_capacities = interconnectors_df.groupby('counterparty_country')['capacity_mw'].sum()
    logger.info(f"Interconnector capacities by country: {dict(country_capacities)}")
    
    # Get unique external buses and their countries
    external_buses = interconnectors_df[['to_bus', 'counterparty_country']].drop_duplicates()
    
    # Filter price differentials by year and pathway if available
    filtered_prices = pd.DataFrame()
    if price_differentials_df is not None and len(price_differentials_df) > 0:
        filtered_prices = price_differentials_df.copy()
        
        # Filter by modelled year
        if modelled_year is not None and 'year' in filtered_prices.columns:
            year_filtered = filtered_prices[filtered_prices['year'] == modelled_year]
            if len(year_filtered) > 0:
                filtered_prices = year_filtered
                logger.info(f"Filtered prices to year {modelled_year}: {len(filtered_prices)} records")
            else:
                logger.warning(f"No price data for year {modelled_year}, using all years")
        
        # Filter by FES pathway
        if fes_pathway is not None and 'pathway' in filtered_prices.columns:
            pathway_filtered = filtered_prices[
                filtered_prices['pathway'].str.lower() == fes_pathway.lower()
            ]
            if len(pathway_filtered) > 0:
                filtered_prices = pathway_filtered
                logger.info(f"Filtered prices to pathway '{fes_pathway}': {len(filtered_prices)} records")
            else:
                logger.warning(f"No price data for pathway '{fes_pathway}', using all pathways")
    
    generators_added = 0
    
    for _, row in external_buses.iterrows():
        external_bus = row['to_bus']
        country = row['counterparty_country']
        
        # Map country name for price lookup (e.g., Ireland → SEM)
        price_country = COUNTRY_PRICE_MAPPING.get(country.lower(), country.lower())
        
        # Determine marginal cost for this country
        if len(filtered_prices) > 0:
            country_prices = filtered_prices[
                filtered_prices['country'].str.lower() == price_country
            ]
            
            if len(country_prices) > 0:
                marginal_cost = country_prices['estimated_price_gbp_per_mwh'].mean()
                logger.info(f"  {country} generator: £{marginal_cost:.2f}/MWh "
                          f"(from {price_country} data, year={modelled_year}, pathway={fes_pathway})")
            else:
                marginal_cost = 50.0  # Default European wholesale price
                logger.warning(f"  {country}: No price data for '{price_country}', using default £{marginal_cost:.2f}/MWh")
        else:
            marginal_cost = 50.0  # Default European wholesale price
            logger.info(f"  {country} generator: £{marginal_cost:.2f}/MWh (no price data available)")
        
        # Calculate capacity for this external bus
        # Sum capacity of all interconnectors connected to this bus
        bus_interconnectors = interconnectors_df[interconnectors_df['to_bus'] == external_bus]
        bus_capacity = bus_interconnectors['capacity_mw'].sum()
        
        # Add small margin (10%) for headroom
        generator_capacity = bus_capacity * 1.1
        
        gen_name = f"EU_supply_{country}_{external_bus}"
        
        network.add(
            "Generator",
            gen_name,
            bus=external_bus,
            p_nom=generator_capacity,  # Capacity = interconnector capacity + 10% margin
            marginal_cost=marginal_cost,  # European wholesale price (scalar fallback)
            carrier='EU_import',
            # Additional metadata
            country=country,
            source='European generation mix'
        )
        
        generators_added += 1
        logger.info(f"  Added EU generator: {gen_name}, capacity={generator_capacity:.0f} MW, "
                   f"marginal_cost=£{marginal_cost:.2f}/MWh")
        
        # ── Apply time-varying MC from ENTSO-E prices if available ──
        if entsoe_prices_df is not None and len(entsoe_prices_df) > 0 and country in entsoe_prices_df.columns:
            hourly = entsoe_prices_df[country].copy()
            entsoe_annual_mean = hourly.mean()

            if entsoe_annual_mean > 0:
                # Scale hourly shape so its mean equals the FES-derived annual MC
                scale = marginal_cost / entsoe_annual_mean
                scaled_hourly = hourly * scale
                logger.info(
                    f"  {gen_name}: Scaling ENTSO-E shape (mean £{entsoe_annual_mean:.2f}) "
                    f"× {scale:.3f} → target mean £{marginal_cost:.2f}/MWh"
                )
            else:
                scaled_hourly = hourly
                logger.warning(f"  {gen_name}: ENTSO-E mean ≤ 0 — using unscaled prices")

            _apply_time_varying_mc(network, gen_name, scaled_hourly, logger)
        elif entsoe_prices_df is not None and len(entsoe_prices_df) > 0:
            logger.warning(f"  {gen_name}: No ENTSO-E column for '{country}' — using scalar MC")
    
    logger.info(f"Added {generators_added} European supply generators on external buses")


def add_external_generators(
    network,
    interconnectors_df,
    price_differentials_df=None,
    modelled_year=None,
    fes_pathway=None,
    entsoe_prices_df=None,
    pricing_config=None,
):
    '''Add price-taking foreign supply and export demand at external buses.'''
    logger = logging.getLogger(__name__)
    pricing = pricing_config or {}
    filtered = (
        price_differentials_df.copy()
        if price_differentials_df is not None
        else pd.DataFrame()
    )
    if not filtered.empty and modelled_year is not None and 'year' in filtered:
        selected = filtered[filtered['year'] == modelled_year]
        if not selected.empty:
            filtered = selected
    if not filtered.empty and fes_pathway and 'pathway' in filtered:
        selected = filtered[
            filtered['pathway'].astype(str).str.lower() == fes_pathway.lower()
        ]
        if not selected.empty:
            filtered = selected

    external_buses = interconnectors_df[
        ['to_bus', 'counterparty_country']
    ].drop_duplicates()
    generators_added = 0
    for row in external_buses.itertuples(index=False):
        external_bus = row.to_bus
        country = row.counterparty_country
        cables = interconnectors_df[interconnectors_df['to_bus'] == external_bus]
        capacity = float(cables['capacity_mw'].sum()) * 1.1
        efficiencies = cables.get(
            'losses_percent', pd.Series(2.5, index=cables.index)
        ).apply(calculate_link_efficiency)
        efficiency = float(efficiencies.min()) if not efficiencies.empty else 0.975
        target = _foreign_price_target(country, filtered, pricing)
        reference = target

        entsoe_country = ENTSOE_COUNTRY_MAPPING.get(country.lower(), country)
        entsoe_column = None
        if entsoe_prices_df is not None and not entsoe_prices_df.empty:
            entsoe_column = _case_insensitive_column(
                entsoe_prices_df, entsoe_country
            )
        if entsoe_column is not None:
            if entsoe_prices_df[entsoe_column].min() >= 0.0:
                logger.warning(
                    '%s hourly prices contain no negative values; refresh older '
                    'cached ENTSO-E files if they were created with a zero floor',
                    country,
                )
            reference = _recenter_and_clip_prices(
                entsoe_prices_df[entsoe_column],
                target,
                float(pricing.get('price_floor', -500.0)),
                float(pricing.get('price_cap', 500.0)),
            )

        import_ask, export_bid = _foreign_bid_ask_prices(
            reference, efficiency, pricing
        )
        static_ask, static_bid = _foreign_bid_ask_prices(
            target, efficiency, pricing
        )
        supply_name = f'EU_supply_{country}_{external_bus}'
        export_name = f'EU_export_sink_{country}_{external_bus}'

        network.add(
            'Generator',
            supply_name,
            bus=external_bus,
            p_nom=capacity,
            p_min_pu=0.0,
            p_max_pu=1.0,
            marginal_cost=static_ask,
            carrier='EU_import',
            country=country,
            foreign_market_role='import_supply',
            source='Foreign day-ahead market',
        )
        network.add(
            'Generator',
            export_name,
            bus=external_bus,
            p_nom=capacity,
            p_min_pu=-1.0,
            p_max_pu=0.0,
            marginal_cost=static_bid,
            carrier='EU_export',
            country=country,
            foreign_market_role='export_demand',
            source='Foreign day-ahead market',
        )
        generators_added += 2
        if isinstance(import_ask, pd.Series):
            _apply_time_varying_mc(network, supply_name, import_ask, logger)
            _apply_time_varying_mc(network, export_name, export_bid, logger)
        elif entsoe_prices_df is not None and not entsoe_prices_df.empty:
            logger.warning(
                'No ENTSO-E price column for %s; using static price', country
            )
        logger.info(
            '%s foreign market: target %.2f, ask %.2f, bid %.2f GBP/MWh',
            country, target, static_ask, static_bid,
        )
    logger.info('Added %d foreign market counterparties', generators_added)


def _add_interconnector_links_legacy(network: pypsa.Network,
                             interconnectors_df: pd.DataFrame,
                             availability_df: pd.DataFrame,
                             price_differentials_df: pd.DataFrame = None,
                             modelled_year: int = None,
                             fes_pathway: str = None,
                             entsoe_prices_df: pd.DataFrame = None) -> None:
    """
    Add interconnector links to the PyPSA network.
    
    Links are added with near-zero marginal cost since the cost of electricity
    is now represented by generators on the external buses. The link marginal_cost
    only represents transmission losses/costs.
    
    Args:
        network: PyPSA network object
        interconnectors_df: Mapped interconnector data
        availability_df: Availability profiles
        price_differentials_df: European price differentials (used for external generators)
        modelled_year: The year being modelled (for filtering price data)
        fes_pathway: The FES pathway name (e.g., 'Holistic Transition')
        entsoe_prices_df: Hourly ENTSO-E day-ahead prices (optional)
    """
    logger = logging.getLogger(__name__)
    
    links_added = 0
    links_failed = 0
    
    # First add generators on external buses
    add_external_generators(
        network, 
        interconnectors_df, 
        price_differentials_df,
        modelled_year=modelled_year,
        fes_pathway=fes_pathway,
        entsoe_prices_df=entsoe_prices_df
    )
    
    logger.info("Adding interconnector links (with external generators already in place)...")
    
    for idx, row in interconnectors_df.iterrows():
        try:
            link_name = f"IC_{row['name']}"
            from_bus = row['from_bus']
            to_bus = row['to_bus']
            capacity_mw = ZEROED_CAPACITY_MW if FORCE_ZERO_INTERCONNECTORS else row['capacity_mw']
            losses_percent = row.get('losses_percent', 2.5)
            counterparty = row.get('counterparty_country', 'Unknown')
            
            # Skip if no valid bus mapping
            if pd.isna(from_bus) or pd.isna(to_bus):
                logger.warning(f"Skipping {row['name']}: invalid bus mapping")
                links_failed += 1
                continue
            
            # Check if buses exist in network
            if from_bus not in network.buses.index:
                logger.warning(f"From bus '{from_bus}' not found in network for {row['name']}")
                links_failed += 1
                continue
            
            if to_bus not in network.buses.index:
                logger.warning(f"To bus '{to_bus}' not found in network for {row['name']}")
                links_failed += 1
                continue
            
            # Calculate efficiency
            efficiency = calculate_link_efficiency(losses_percent)
            
            # Link marginal cost is now ZERO (or very small) because:
            # - The cost of European electricity is on the external generator
            # - Efficiency already handles transmission losses
            # - We want the optimizer to freely use the link based on price differentials
            link_marginal_cost = 0.0  # Zero cost - economics handled by external generators
            
            # Add the link
            network.add(
                "Link",
                link_name,
                bus0=from_bus,
                bus1=to_bus,
                p_nom=capacity_mw,
                p_min_pu=-1.0,  # Bidirectional
                p_max_pu=1.0,
                efficiency=efficiency,
                marginal_cost=link_marginal_cost,  # Zero - cost is on external generator
                carrier='DC',
                # Additional metadata
                interconnector_name=row['name'],
                counterparty_country=counterparty,
                commissioning_year=row.get('commissioning_year', np.nan),
                source=row.get('source', 'Unknown')
            )
            
            links_added += 1
            logger.debug(f"Added link: {link_name} ({from_bus} -> {to_bus}, {capacity_mw:.1f} MW)")
            
        except Exception as e:
            logger.error(f"Error adding link for {row['name']}: {e}")
            links_failed += 1
    
    logger.info(f"Interconnector links: {links_added} added, {links_failed} failed")

def _filter_price_differentials(price_differentials_df, modelled_year, fes_pathway):
    """Select one explicit FES price pathway, with no all-pathway fallback."""
    filtered = price_differentials_df.copy() if price_differentials_df is not None else pd.DataFrame()
    if filtered.empty:
        return filtered
    if modelled_year is not None and "year" in filtered:
        filtered = filtered[filtered["year"] == modelled_year]
        if filtered.empty:
            raise ValueError(f"No foreign price data for model year {modelled_year}")
    if fes_pathway and "pathway" in filtered:
        # The pathway's own rows first; a table that groups the two net zero pathways in one label (an earlier
        # ES2 layout) is matched through it. NESO's current ES2 table lists them separately.
        aliases = {
            "holistic transition": "holistic transition, electric engagement",
            "electric engagement": "holistic transition, electric engagement",
        }
        own = str(fes_pathway).strip().lower()
        labels = filtered["pathway"].astype(str).str.strip().str.lower()
        wanted = own if (labels == own).any() else aliases.get(own, own)
        filtered = filtered[labels == wanted]
        if filtered.empty:
            raise ValueError(
                f"No foreign price data matches pathway '{fes_pathway}' (resolved as '{wanted}')"
            )
    return filtered


def add_external_market_balancers(
    network,
    interconnectors_df,
    price_differentials_df=None,
    modelled_year=None,
    fes_pathway=None,
    entsoe_prices_df=None,
    pricing_config=None,
):
    """Add neutral foreign balances and construct the import/export price series."""
    logger = logging.getLogger(__name__)
    pricing = pricing_config or {}
    filtered = _filter_price_differentials(
        price_differentials_df, modelled_year, fes_pathway
    )
    historical_direct = modelled_year is not None and int(modelled_year) <= 2024
    hourly_shape = str(pricing.get("hourly_shape", "entsoe")).lower()
    # Task Y3: absent key (every scenario today) = False = unchanged behaviour.
    # Item F13 (7 Oct 2026): "cp30_basis" (TYNDP NT2030 rebased to CP30's fuel,
    # scripts/interconnectors/cp30_foreign_prices.py) is also used at its own level.
    price_source = str(pricing.get("foreign_price_source", "entsoe")).strip().lower()
    tyndp_prices = price_source in OWN_LEVEL_PRICE_SOURCES
    if tyndp_prices and historical_direct:
        raise ValueError(f"foreign_price_source {price_source!r} is for future scenarios, not historical ones")
    # Item F13: one price per counterparty country instead of one per coupled
    # market. GB then carries power between countries when their prices part by
    # more than the hurdle, as it did in 2024 (ESPENI: 9.2 TWh of 10.6 TWh of
    # exports in periods GB also imported). Off (absent key) = unchanged.
    per_country = bool(pricing.get("single_price_per_country", False))

    if "EU_market" not in network.carriers.index:
        network.add("Carrier", "EU_market", nice_name="Foreign day-ahead market balance", color="#64748B")

    market_prices = {}
    market_shapes = None   # item D2 single basis: built on first use
    external_buses = interconnectors_df[["to_bus", "counterparty_country"]].drop_duplicates()
    for row in external_buses.itertuples(index=False):
        external_bus, country = row.to_bus, row.counterparty_country
        cables = interconnectors_df[interconnectors_df["to_bus"] == external_bus]
        capacity = float(cables["capacity_mw"].sum()) * 1.1
        import_target = _foreign_price_target(country, filtered, pricing)
        export_target = _foreign_export_price_target(country, filtered, pricing)
        import_reference = import_target
        export_reference = export_target

        entsoe_country = ENTSOE_COUNTRY_MAPPING.get(country.lower(), country)
        entsoe_column = (
            _case_insensitive_column(entsoe_prices_df, entsoe_country)
            if entsoe_prices_df is not None and not entsoe_prices_df.empty
            else None
        )
        if entsoe_column is not None:
            floor = float(pricing.get("price_floor", -500.0))
            cap = float(pricing.get("price_cap", 500.0))
            hourly = entsoe_prices_df[entsoe_column]
            if historical_direct:
                # A historical back-test validates against actual price levels.
                import_reference = _prepare_historical_entsoe_prices(hourly, floor, cap)
                export_reference = import_reference
            elif tyndp_prices:
                # Task Y3 (FES 2050 plan): TYNDP's 2050 price at its own level,
                # with no CP.29 re-centring. On the single basis the continent is
                # one GB-capacity-weighted series, as for IC1; the level offset
                # is 0 unless the scenario sets one. The annual mean becomes the
                # import and export target, so it is also the tranche level.
                from scripts.interconnectors.tyndp_prices import tyndp_market_reference
                if single_foreign_price_basis(pricing):
                    market = single_price_market_of(country, pricing)
                    offset = float((pricing.get("single_price_level_offsets_gbp_per_mwh") or {}).get(market, 0.0))
                    if per_country:
                        hourly = pd.to_numeric(hourly, errors="coerce").astype(float) + offset
                    else:
                        if market_shapes is None:
                            market_shapes = single_price_market_shapes(entsoe_prices_df, interconnectors_df, pricing)
                        hourly = market_shapes[market] + offset
                import_reference, import_target = tyndp_market_reference(
                    hourly, country, pricing, floor, cap, _apply_price_case)
                export_reference, export_target = import_reference, import_target
            elif single_foreign_price_basis(pricing):
                # Item D2: one hourly price per market, the shape plus a
                # calibrated offset; import and export differ only by the
                # transaction spread and the loss. The CP.29 targets stay as
                # the tranche levels (market_import_level/export_level).
                if market_shapes is None:
                    market_shapes = single_price_market_shapes(entsoe_prices_df, interconnectors_df, pricing)
                import_reference = single_price_reference(
                    market_shapes[single_price_market_of(country, pricing)], country, pricing, floor, cap)
                export_reference = import_reference
            else:
                # Future CP30/FES cases use ENTSO-E only as an hourly shape.
                import_reference = _recenter_and_clip_prices(hourly, import_target, floor, cap)
                export_reference = _recenter_and_clip_prices(hourly, export_target, floor, cap)
        elif single_foreign_price_basis(pricing) and not historical_direct and hourly_shape != "flat":
            raise ValueError(
                f"foreign_price_basis 'single' needs an hourly ENTSO-E shape for {country}; "
                "its level is an offset on that shape"
            )
        elif entsoe_prices_df is not None and not entsoe_prices_df.empty:
            if not historical_direct:
                # A shaped run with one counterparty left flat is silently wrong:
                # NeuConnect (Germany) had no column until 26 Sep 2026. Add the
                # country to fetch_entsoe_prices.COUNTRY_ZONE_MAP instead.
                raise ValueError(
                    f"No ENTSO-E price column for {country} (looked for {entsoe_country!r} in "
                    f"{list(entsoe_prices_df.columns)}); every counterparty needs an hourly shape"
                )
            logger.warning("No ENTSO-E price column for %s; using annual trade-price targets", country)
        if tyndp_prices and entsoe_column is None:
            raise ValueError(f"foreign_price_source {price_source!r}: the price file has no column for {country}")

        balance_name = f"EU_market_balance_{country}_{external_bus}"
        network.add(
            "Generator", balance_name, bus=external_bus, p_nom=capacity,
            p_min_pu=-1.0, p_max_pu=1.0, marginal_cost=0.0,
            carrier="EU_market", country=country,
            foreign_market_role="market_balance", source="Neutral foreign market energy balance",
        )
        market_prices[external_bus] = {
            "country": country,
            "import_target": import_target,
            "export_target": export_target,
            "import_reference": import_reference,
            "export_reference": export_reference,
        }
        if single_foreign_price_basis(pricing) and hourly_shape == "flat" and not historical_direct:
            # The single basis is an offset on an hourly shape; a flat run has
            # none, so it keeps the flat CP.29 import/export targets (declared).
            logger.warning(
                "%s: hourly_shape 'flat' has no shape for the single foreign price; "
                "using the separate flat CP.29 import/export targets", country,
            )
        if isinstance(import_reference, pd.Series) and import_reference is export_reference and not historical_direct:
            # Single basis: the static targets are the shaped price's own mean,
            # one object so the spread logic treats import and export as one price.
            single_mean = float(import_reference.mean())
            market_prices[external_bus]["static_reference"] = single_mean
        if historical_direct:
            price_basis = "raw historical ENTSO-E"
        elif tyndp_prices and price_source == "cp30_basis":
            price_basis = "TYNDP NT2030 rebased to CP30 fuel, own level, %s (item F13)" % (
                "per country" if per_country else "per coupled market")
        elif tyndp_prices:
            price_basis = "TYNDP 2050 hourly price at its own level, no CP.29 re-centring (task Y3)"
        elif single_foreign_price_basis(pricing) and "static_reference" in market_prices[external_bus]:
            price_basis = "single hourly price: ENTSO-E shape + calibrated offset (mean %.2f)" % float(
                import_reference.mean())
        elif entsoe_column is not None:
            price_basis = "future target + ENTSO-E shape"
        elif hourly_shape == "flat":
            price_basis = "flat annual trade-price target"
        else:
            price_basis = "annual trade-price target (no hourly shape available)"
        logger.info(
            "%s foreign market: import target %.2f, export target %.2f GBP/MWh (%s)",
            country, import_target, export_target,
            price_basis,
        )
    return market_prices

def _add_directional_interconnector_links(
    network,
    interconnectors_df,
    availability_df,
    price_differentials_df=None,
    modelled_year=None,
    fes_pathway=None,
    entsoe_prices_df=None,
    pricing_config=None,
):
    '''Add a loss-correct directional import/export pair for every cable.'''
    logger = logging.getLogger(__name__)
    pricing = pricing_config or {}
    market_prices = add_external_market_balancers(
        network,
        interconnectors_df,
        price_differentials_df,
        modelled_year=modelled_year,
        fes_pathway=fes_pathway,
        entsoe_prices_df=entsoe_prices_df,
        pricing_config=pricing,
    )
    added = 0
    failed = 0
    for row in interconnectors_df.to_dict(orient='records'):
        try:
            name = row['name']
            gb_bus = row['from_bus']
            foreign_bus = row['to_bus']
            if pd.isna(gb_bus) or pd.isna(foreign_bus):
                raise ValueError('invalid bus mapping')
            if gb_bus not in network.buses.index or foreign_bus not in network.buses.index:
                raise ValueError('mapped bus is absent from network')

            capacity = (
                ZEROED_CAPACITY_MW
                if FORCE_ZERO_INTERCONNECTORS
                else float(row['capacity_mw'])
            )
            efficiency = calculate_link_efficiency(row.get('losses_percent', 2.5))
            counterparty = row.get('counterparty_country', 'Unknown')
            market = market_prices.get(foreign_bus)
            if market is None:
                raise ValueError(f'no foreign market balance for {foreign_bus}')
            import_ask, export_bid = _separate_import_export_prices(
                market['import_reference'], market['export_reference'], efficiency, pricing
            )
            static_ask, static_bid = _separate_import_export_prices(
                market['import_target'], market['export_target'], efficiency, pricing
            )
            import_name = f'IC_IMPORT_{name}'
            export_name = f'IC_EXPORT_{name}'
            common = {
                'p_nom': capacity,
                'p_min_pu': 0.0,
                'p_max_pu': 1.0,
                'efficiency': efficiency,
                'carrier': 'DC',
                'interconnector_name': name,
                'interconnector_pair_id': name,
                'counterparty_country': counterparty,
                'commissioning_year': row.get('commissioning_year', np.nan),
                'source': row.get('source', 'Unknown'),
                'foreign_market_pricing': 'direct_link',
            }
            ramp_limit = pricing.get('ramp_limit_per_hour')
            if ramp_limit is not None:
                common['ramp_limit_up'] = float(ramp_limit)
                common['ramp_limit_down'] = float(ramp_limit)

            network.add(
                'Link',
                import_name,
                bus0=foreign_bus,
                bus1=gb_bus,
                interconnector_direction='import',
                marginal_cost=static_ask,
                foreign_market_price=static_ask,
                **common,
            )
            network.add(
                'Link',
                export_name,
                bus0=gb_bus,
                bus1=foreign_bus,
                interconnector_direction='export',
                marginal_cost=-efficiency * static_bid,
                foreign_market_price=static_bid,
                **common,
            )
            if isinstance(import_ask, pd.Series):
                _apply_time_varying_link_mc(
                    network, import_name, import_ask, logger
                )
                _apply_time_varying_link_mc(
                    network, export_name, -efficiency * export_bid, logger
                )
            logger.info(
                '%s direct link prices: import ask %.2f, export bid %.2f GBP/MWh',
                name,
                static_ask,
                static_bid,
            )
            added += 1
        except Exception as error:
            logger.error('Could not add interconnector %s: %s', row.get('name'), error)
            failed += 1
    logger.info(
        'Interconnectors: %d directional pairs added, %d failed', added, failed
    )




def _apply_time_varying_link_market_attribute(network, link_name, attribute, values):
    """Store a custom time-varying price attribute for one Link."""
    table = network.links_t.get(attribute, pd.DataFrame(index=network.snapshots))
    table = table.reindex(index=network.snapshots)
    table[link_name] = pd.Series(values).reindex(network.snapshots).ffill().bfill()
    network.links_t[attribute] = table


def add_interconnector_links(
    network,
    interconnectors_df,
    availability_df,
    price_differentials_df=None,
    modelled_year=None,
    fes_pathway=None,
    entsoe_prices_df=None,
    pricing_config=None,
):
    """Add one signed physical Link per cable with piecewise bid/ask pricing.

    Link flow is positive from the foreign bus into GB (an import) and negative
    from GB into the foreign bus (an export). The market objective is added by
    ``build_piecewise_interconnector_cost_callback``; the native Link marginal
    cost remains zero so it cannot incorrectly apply one price to both signs.
    """
    logger = logging.getLogger(__name__)
    pricing = pricing_config or {}
    market_prices = add_external_market_balancers(
        network,
        interconnectors_df,
        price_differentials_df,
        modelled_year=modelled_year,
        fes_pathway=fes_pathway,
        entsoe_prices_df=entsoe_prices_df,
        pricing_config=pricing,
    )
    added = 0
    failed = 0
    for row in interconnectors_df.to_dict(orient="records"):
        try:
            name = row["name"]
            gb_bus = row["from_bus"]
            foreign_bus = row["to_bus"]
            if pd.isna(gb_bus) or pd.isna(foreign_bus):
                raise ValueError("invalid bus mapping")
            if gb_bus not in network.buses.index or foreign_bus not in network.buses.index:
                raise ValueError("mapped bus is absent from network")

            capacity = (
                ZEROED_CAPACITY_MW
                if FORCE_ZERO_INTERCONNECTORS
                else float(row["capacity_mw"])
            )
            efficiency = calculate_link_efficiency(row.get("losses_percent", 2.5))
            market = market_prices.get(foreign_bus)
            if market is None:
                raise ValueError(f"no foreign market balance for {foreign_bus}")
            single = "static_reference" in market
            # The single basis trades through its own calibrated hurdle (item D2),
            # so the separate basis keeps transaction_spread unchanged.
            link_pricing = single_price_link_pricing(pricing) if single else pricing
            import_ask, export_bid = _separate_import_export_prices(
                market["import_reference"], market["export_reference"], efficiency, link_pricing
            )
            # Single basis only: exports keep a bid of min(foreign price, floor)
            # when the foreign price is positive, so the hurdle cannot make GB
            # curtail surplus it could sell (item D2 surplus floor).
            export_bid_floor = single_price_export_bid_floor(pricing) if single else None
            if single:
                static_ask, static_bid = _separate_import_export_prices(
                    market["static_reference"], market["static_reference"], efficiency, link_pricing
                )
                static_bid = floor_export_bid(static_bid, market["static_reference"], export_bid_floor)
            else:
                static_ask, static_bid = _separate_import_export_prices(
                    market["import_target"], market["export_target"], efficiency, link_pricing
                )
            link_name = f"IC_{name}"
            link_data = {
                "bus0": foreign_bus,
                "bus1": gb_bus,
                "p_nom": capacity,
                "p_min_pu": -1.0,
                "p_max_pu": 1.0,
                "efficiency": efficiency,
                "marginal_cost": static_ask,
                "carrier": "DC",
                "interconnector_name": name,
                "interconnector_pair_id": name,
                "interconnector_direction": "bidirectional",
                "counterparty_country": row.get("counterparty_country", "Unknown"),
                "commissioning_year": row.get("commissioning_year", np.nan),
                "source": row.get("source", "Unknown"),
                "foreign_market_pricing": "piecewise_bidirectional",
                "market_export_bid": static_bid,
                "market_export_bid_offset": static_ask - static_bid,
                # The annual levels before any spread: the volume tranches scale
                # their price steps on these (item IC1).
                "market_import_level": float(market["import_target"]),
                "market_export_level": float(market["export_target"]),
                # Separate CP.29 import/export series only on the legacy basis; the
                # single basis (item D2) rebuilds the bid from the ask and spread.
                "market_separate_import_export_prices": (
                    str(pricing.get("source", "derived")).lower() == "cp30" and not single
                ),
                "market_price_basis": "single" if single else "separate",
                "market_spread_half": max(float(link_pricing.get("transaction_spread_gbp_per_mwh", 1.0)) / 2.0, 0.0),
                "market_circulation_epsilon": max(float(link_pricing.get("circulation_epsilon_gbp_per_mwh", 0.01)), 0.0),
                # Read by piecewise_market_prices when it rebuilds the bid; NaN = no floor.
                EXPORT_BID_FLOOR_COLUMN: np.nan if export_bid_floor is None else export_bid_floor,
            }
            ramp_limit = pricing.get("ramp_limit_per_hour")
            if ramp_limit is not None:
                link_data["ramp_limit_up"] = float(ramp_limit)
                link_data["ramp_limit_down"] = float(ramp_limit)
            network.add("Link", link_name, **link_data)
            if isinstance(import_ask, pd.Series):
                _apply_time_varying_link_mc(network, link_name, import_ask, logger)

            logger.info(
                "%s signed Link prices: import ask %.2f, export bid %.2f GBP/MWh",
                name,
                static_ask,
                static_bid,
            )
            added += 1
        except Exception as error:
            logger.error("Could not add interconnector %s: %s", row.get("name"), error)
            failed += 1
    logger.info("Interconnectors: %d signed Links added, %d failed", added, failed)


def apply_availability_profiles(network: pypsa.Network, availability_df: pd.DataFrame) -> None:
    """
    Apply availability profiles to interconnector links.
    
    For future scenarios where the modelled year differs from the availability data year,
    the availability profiles are mapped using hour-of-year to carry over the seasonal
    patterns to the target year.
    
    Args:
        network: PyPSA network object
        availability_df: Availability profiles DataFrame with columns: time, name, p_max_pu
    """
    logger = logging.getLogger(__name__)
    
    if len(availability_df) == 0:
        logger.warning("No availability profiles available - using default p_max_pu=1.0")
        return
    
    # Get interconnector links
    ic_links = network.links[network.links.index.str.startswith('IC_')]
    
    if len(ic_links) == 0:
        logger.warning("No interconnector links found in network")
        return
    
    # Get network snapshots
    if not hasattr(network, 'snapshots') or len(network.snapshots) == 0:
        logger.warning("Network has no snapshots - cannot apply availability profiles")
        return
    
    network_snapshots = pd.DatetimeIndex(network.snapshots)
    network_year = network_snapshots[0].year
    
    # Determine availability data year
    availability_df = availability_df.copy()
    availability_df['time'] = pd.to_datetime(availability_df['time'])
    availability_year = availability_df['time'].iloc[0].year
    
    if availability_year != network_year:
        logger.info(f"Availability profiles are for {availability_year}, network is for {network_year}")
        logger.info("Mapping availability profiles using hour-of-year alignment")
    import calendar as _calendar
    # Exactly one of the two years a leap year (2013 onto 2040, or 2020 onto 2030): match by month, day
    # and hour, not day of year, which reads every day from 1 March a day off. A target 29 February the
    # source lacks takes 28 February's; a source 29 February the target lacks is unused.
    match_by_calendar = _calendar.isleap(network_year) != _calendar.isleap(availability_year)
    
    # Create availability time series for each link. An outage limits both
    # directions: a signed (bidirectional) link, static p_min_pu < 0, also gets
    # links_t.p_min_pu = p_min_pu * availability. Before 27 Sep 2026 (fix S1)
    # only p_max_pu was written, so an out-of-service cable could not import
    # but could still export at full rating.
    profiles_applied = 0
    profiles_defaulted = 0
    outage_hours = {}

    def _write(link_name: str, availability: pd.Series) -> None:
        static_max = float(pd.to_numeric(ic_links.at[link_name, 'p_max_pu'], errors='coerce'))
        static_min = float(pd.to_numeric(ic_links.at[link_name, 'p_min_pu'], errors='coerce'))
        static_max = 1.0 if not np.isfinite(static_max) else static_max
        static_min = 0.0 if not np.isfinite(static_min) else static_min
        availability = availability.clip(lower=0.0, upper=1.0)
        network.links_t.p_max_pu[link_name] = (
            availability * static_max if static_max > 0.0 else pd.Series(static_max, index=availability.index)
        )
        if static_min < 0.0:
            network.links_t.p_min_pu[link_name] = availability * static_min
        outage_hours[link_name] = int((availability < 1.0 - 1e-9).sum())

    for link_name in ic_links.index:
        # Directional pairs share the original cable name and availability.
        if 'interconnector_name' in ic_links.columns:
            ic_name = ic_links.at[link_name, 'interconnector_name']
        else:
            ic_name = link_name.replace('IC_', '')
        
        # Find matching availability profile
        link_availability = availability_df[availability_df['name'] == ic_name].copy()
        
        if len(link_availability) > 0:
            if match_by_calendar:
                times = link_availability['time'].dt
                link_availability['hour_of_year'] = list(zip(times.month, times.day, times.hour))
                source_has_leap_day = bool(((times.month == 2) & (times.day == 29)).any())
                leap_day = (network_snapshots.month == 2) & (network_snapshots.day == 29)
                days = np.where(leap_day & (not source_has_leap_day), 28, network_snapshots.day)
                network_hours = list(zip(network_snapshots.month, days, network_snapshots.hour))
            else:
                # Hour-of-year index for matching across years
                link_availability['hour_of_year'] = (
                    link_availability['time'].dt.dayofyear * 24 +
                    link_availability['time'].dt.hour
                )
                network_hours = network_snapshots.dayofyear * 24 + network_snapshots.hour
            
            # Build lookup dictionary from availability data
            hour_to_availability = dict(zip(
                link_availability['hour_of_year'], 
                link_availability['p_max_pu']
            ))
            
            # Map to network snapshots
            p_max_pu_values = pd.Series(
                [hour_to_availability.get(h, 1.0) for h in network_hours],
                index=network_snapshots
            )
            
            # Replace any NaN values with 1.0 (full availability)
            p_max_pu_values = p_max_pu_values.fillna(1.0)
            
            # Time-varying limits in both directions
            _write(link_name, p_max_pu_values)

            profiles_applied += 1
            logger.debug(f"Applied availability profile to {link_name} "
                        f"(mean={p_max_pu_values.mean():.3f})")
        else:
            # No availability profile found - full availability for all snapshots
            _write(link_name, pd.Series(1.0, index=network_snapshots))
            profiles_defaulted += 1
            logger.debug(f"No availability profile for {ic_name} - using default p_max_pu=1.0")

    logger.info(f"Availability profiles: {profiles_applied} applied, {profiles_defaulted} defaulted to 1.0")
    reduced = {name: hours for name, hours in outage_hours.items() if hours}
    if reduced:
        logger.info(
            "Interconnector hours below full availability (both directions limited): "
            + ", ".join(f"{name} {hours}" for name, hours in sorted(reduced.items()))
        )


def add_historical_interconnector_links(network: pypsa.Network,
                                       interconnectors_df: pd.DataFrame,
                                       historical_flows_df: pd.DataFrame,
                                       entsoe_prices_df: pd.DataFrame = None) -> None:
    """
    Add interconnector links with FIXED historical flows from ESPENI.
    
    For historical scenarios, interconnector flows are actual observations
    and should not be optimized. Positive flows = imports, negative = exports.
    
    When *entsoe_prices_df* is provided, EU_supply generators are created on
    the external buses with time-varying marginal costs.  The generators have
    p_nom = 0 and serve only to inject price signals into the nodal shadow
    prices; they do NOT affect dispatch (flows remain fixed via p_set).
    
    This function also tracks which external buses have no corresponding links
    (due to missing flow data) and removes them from the network.
    
    IMPORTANT: Handles cases where multiple interconnectors map to the same ESPENI
    column (e.g., Moyle and Auchencrosh both map to 'Moyle'). In such cases, the
    flows are split proportionally by capacity to avoid double-counting.
    
    Args:
        network: PyPSA network object
        interconnectors_df: Mapped interconnector data with capacities
        historical_flows_df: Historical flow time series from ESPENI (MW)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Adding interconnectors with FIXED historical flows")
    
    # ==========================================================================
    # PRE-COMPUTE ESPENI MAPPING AND CAPACITY SHARES
    # ==========================================================================
    # Some interconnectors share the same ESPENI data column (e.g., Moyle and 
    # Auchencrosh both map to 'Moyle'). We need to split flows proportionally
    # by capacity to avoid double-counting.
    
    espeni_mapping = {}  # {espeni_column: [(ic_name, capacity_mw), ...]}
    
    for idx, row in interconnectors_df.iterrows():
        ic_name = row['name']
        capacity_mw = row.get('capacity_mw', 0)
        
        # Skip if no valid bus mapping
        if pd.isna(row.get('from_bus')) or pd.isna(row.get('to_bus')):
            continue
            
        # Normalize to ESPENI column name
        espeni_name = normalize_interconnector_name(
            ic_name, 
            available_columns=historical_flows_df.columns.tolist()
        )
        
        if espeni_name in historical_flows_df.columns:
            if espeni_name not in espeni_mapping:
                espeni_mapping[espeni_name] = []
            espeni_mapping[espeni_name].append((ic_name, capacity_mw))
    
    # Compute capacity shares for each ESPENI column
    capacity_shares = {}  # {ic_name: share (0-1)}
    for espeni_name, interconnectors in espeni_mapping.items():
        total_capacity = sum(cap for _, cap in interconnectors)
        if total_capacity > 0:
            for ic_name, cap in interconnectors:
                capacity_shares[ic_name] = cap / total_capacity
                if len(interconnectors) > 1:
                    logger.info(
                        f"  Flow sharing: {ic_name} gets {cap/total_capacity*100:.1f}% of '{espeni_name}' "
                        f"flows ({cap:.0f}/{total_capacity:.0f} MW)"
                    )
        else:
            for ic_name, _ in interconnectors:
                capacity_shares[ic_name] = 1.0 / len(interconnectors)
    
    # Log any shared ESPENI columns
    shared_columns = {k: v for k, v in espeni_mapping.items() if len(v) > 1}
    if shared_columns:
        logger.info(f"Detected {len(shared_columns)} ESPENI column(s) shared by multiple interconnectors:")
        for espeni_name, ics in shared_columns.items():
            logger.info(f"  '{espeni_name}': {[ic for ic, _ in ics]}")
    
    links_added = 0
    links_failed = 0
    
    # Track which external buses have links added
    external_buses_with_links = set()
    
    for idx, row in interconnectors_df.iterrows():
        try:
            link_name = f"IC_{row['name']}"
            from_bus = row['from_bus']
            to_bus = row['to_bus']
            capacity_mw = ZEROED_CAPACITY_MW if FORCE_ZERO_INTERCONNECTORS else row['capacity_mw']
            losses_percent = row.get('losses_percent', 2.5)
            counterparty = row.get('counterparty_country', 'Unknown')
            ic_name = row['name']
            
            # Skip if no valid bus mapping
            if pd.isna(from_bus) or pd.isna(to_bus):
                logger.warning(f"Skipping {ic_name}: invalid bus mapping")
                links_failed += 1
                continue
            
            # Check if buses exist in network
            if from_bus not in network.buses.index:
                logger.warning(f"From bus '{from_bus}' not found for {ic_name}")
                links_failed += 1
                continue
            
            if to_bus not in network.buses.index:
                logger.warning(f"To bus '{to_bus}' not found for {ic_name}")
                links_failed += 1
                continue
            
            # Normalize interconnector name to match ESPENI columns
            espeni_name = normalize_interconnector_name(
                ic_name, 
                available_columns=historical_flows_df.columns.tolist()
            )
            
            # Check if we have historical flow data for this interconnector
            if espeni_name not in historical_flows_df.columns:
                logger.info(
                    f"Skipping {ic_name} - no historical flow data available (normalized: {espeni_name}). "
                    f"This is expected for new/planned interconnectors."
                )
                links_failed += 1
                continue
            
            # Get historical flows (MW, positive = import)
            historical_flows = historical_flows_df[espeni_name].copy()
            
            # ====================================================================
            # APPLY CAPACITY SHARE FOR SHARED ESPENI COLUMNS
            # ====================================================================
            # If multiple interconnectors share the same ESPENI column (e.g., Moyle
            # and Auchencrosh both use 'Moyle' data), split flows proportionally
            # by capacity to avoid double-counting.
            share = capacity_shares.get(ic_name, 1.0)
            if share < 1.0:
                logger.debug(f"  {ic_name}: Applying {share*100:.1f}% share of '{espeni_name}' flows")
                historical_flows = historical_flows * share
            
            # Strip timezone from flows if network is timezone-naive
            # Network snapshots are typically timezone-naive, but ESPENI flows are UTC-aware
            if historical_flows.index.tz is not None and network.snapshots.tz is None:
                historical_flows.index = historical_flows.index.tz_localize(None)
            
            # Reindex flows to match network snapshots
            # This ensures flows align with the network's time index
            original_nans = historical_flows.isna().sum()
            historical_flows = historical_flows.reindex(network.snapshots)
            reindexed_nans = historical_flows.isna().sum()
            
            # Log gap statistics before gap-filling
            if reindexed_nans > 0:
                gap_fraction = reindexed_nans / len(historical_flows) * 100
                logger.debug(
                    f"  {ic_name}: Found {reindexed_nans} ({gap_fraction:.2f}%) missing values. "
                    f"Original data: {original_nans} NaNs before reindexing."
                )
            
            # Fill NaN values using interpolation + zero-fill fallback
            # Linear interpolation handles continuous gaps better than zero-fill
            historical_flows = historical_flows.interpolate(
                method='linear', 
                limit_direction='both',
                limit=6  # Allow up to 6 consecutive NaNs (3 hours at 30-min resolution)
            )
            
            # Fill any remaining NaNs with 0 (edge cases or gaps > 3 hours)
            remaining_nans = historical_flows.isna().sum()
            if remaining_nans > 0:
                logger.debug(
                    f"  {ic_name}: {remaining_nans} values still NaN after interpolation - "
                    f"filling with zero (likely edge or extended gap)"
                )
                historical_flows = historical_flows.fillna(0.0)
            
            # Optionally zero flows entirely for diagnostics
            if FORCE_ZERO_INTERCONNECTORS:
                historical_flows = historical_flows * 0.0
            
            # CRITICAL: Clip historical flows to interconnector capacity
            # ESPENI metered data may slightly exceed rated capacity due to:
            #   - Measurement tolerance/rounding in grid meters
            #   - Short-term overloading allowed in practice
            #   - Capacity ratings being conservative design values
            # Clipping ensures feasibility while preserving the direction and timing
            # of actual flows. Without this, fixed p_set may exceed link bounds.
            # ESPENI convention: positive = import, negative = export
            # So we clip to: -capacity_mw <= historical_flows <= +capacity_mw
            flows_before_clip = historical_flows.copy()
            historical_flows = historical_flows.clip(lower=-capacity_mw, upper=capacity_mw)
            
            # Log any clipping that occurred
            clipped_count = (flows_before_clip != historical_flows).sum()
            if clipped_count > 0:
                max_exceed = max(
                    abs(flows_before_clip.max()) - capacity_mw if flows_before_clip.max() > capacity_mw else 0,
                    abs(flows_before_clip.min()) - capacity_mw if flows_before_clip.min() < -capacity_mw else 0
                )
                logger.debug(
                    f"  {ic_name}: Clipped {clipped_count} flow values exceeding capacity of {capacity_mw:.0f} MW. "
                    f"Max exceedance: {max_exceed:.1f} MW ({max_exceed/capacity_mw*100:.1f}% over capacity)"
                )

            # CRITICAL: For historical scenarios with FIXED flows (p_set), efficiency MUST be 1.0
            # According to PyPSA documentation, when p_set is used, efficiency should be 1.0
            # because the fixed flow data already accounts for any transmission losses.
            # Using efficiency < 1.0 with p_set creates energy imbalances → unbounded/infeasible.
            # See: https://pypsa.readthedocs.io/en/latest/user-guide/components/links/
            efficiency = 1.0  # MUST be 1.0 for fixed flows per PyPSA docs
            
            # Link marginal cost is ZERO because:
            # 1. Historical flows are FIXED (p_set) - optimizer doesn't choose flows
            # 2. The cost is already embedded in the historical flow data
            # 3. PyPSA docs require marginal_cost=0 for bidirectional lossless links
            interconnector_marginal_cost = 0.0  # Zero cost per PyPSA docs
            
            # Add the link with p_nom capacity
            network.add(
                "Link",
                link_name,
                bus0=from_bus,
                bus1=to_bus,
                p_nom=capacity_mw,
                p_min_pu=-1.0,  # Allow bidirectional
                p_max_pu=1.0,
                efficiency=efficiency,
                marginal_cost=interconnector_marginal_cost,  # Zero - flows are fixed
                carrier='DC',
                # Additional metadata
                interconnector_name=ic_name,
                counterparty_country=counterparty,
                commissioning_year=row.get('commissioning_year', np.nan),
                source='ESPENI_historical',
                flow_type='fixed'
            )
            
            # DEBUG: Log historical flow statistics
            logger.debug(f"  {ic_name} - Historical flows: mean={historical_flows.mean():.1f} MW, "
                        f"isna={historical_flows.isna().sum()}, len={len(historical_flows)}")
            
            # APPLY FIXED FLOWS: Use p_set to fix interconnector flows to ESPENI values
            # This ensures historical scenarios use actual observed interconnector flows
            # rather than optimizing them (which would change the dispatch)
            # Note: p_set on a Link fixes the power flow at bus0 (positive = flow from bus0 to bus1)
            # ESPENI convention: positive = import to GB, which means flow FROM external TO GB
            # PyPSA Link convention: p0 is power at bus0 (from_bus = GB side)
            # So we need to NEGATE the ESPENI flows since:
            #   - ESPENI: +1000 MW = 1000 MW flowing INTO GB
            #   - Link p0: power at bus0 (GB), so p0 = -1000 means power leaves GB → goes to external
            #   - We want power to ARRIVE at GB, so p0 should be negative for imports
            # Actually, let's check the bus assignment: bus0=from_bus (GB side), bus1=to_bus (external)
            # p_set on Link: positive = flow from bus0 to bus1 = export
            # ESPENI: positive = import to GB
            # Therefore: p_set = -historical_flows (negate to convert import to export convention)
            
            # Wait - let me reconsider. The ESPENI data shows IMPORT to GB.
            # In PyPSA Link: p0 is power at bus0, p1 = -efficiency * p0 (by conservation)
            # If bus0 = GB side, and we want IMPORT (power arriving at GB):
            #   - Power arriving at bus0 means p0 < 0 (power is consumed at bus0 from the link)
            #   - Actually no: p0 represents power ENTERING the link from bus0
            #   - If p0 > 0, power flows FROM bus0 TO bus1 (export)
            #   - If p0 < 0, power flows FROM bus1 TO bus0 (import)
            # So for ESPENI import (positive = into GB): p_set = -historical_flows
            network.links_t.p_set[link_name] = -historical_flows
            
            # Store for validation/logging
            if not hasattr(network, 'historical_interconnector_flows'):
                network.historical_interconnector_flows = {}
            network.historical_interconnector_flows[link_name] = historical_flows
            
            logger.info(
                f"  ✓ Applied FIXED p_set flows from ESPENI (link flows are constrained)"
            )
            
            # Track that this external bus has a link
            external_buses_with_links.add(to_bus)
            
            links_added += 1
            
            # Log statistics - note: ESPENI uses hourly resolution after resampling
            mean_flow = historical_flows.mean()
            # For hourly data, sum gives MWh directly, divide by 1e6 for TWh
            net_import_twh = historical_flows.sum() / 1e6
            
            logger.info(
                f"Added FIXED link: {link_name} ({counterparty}) "
                f"- mean flow: {mean_flow:.1f} MW, "
                f"net import: {net_import_twh:.2f} TWh"
            )
            
        except Exception as e:
            logger.error(f"Error adding historical link for {row['name']}: {e}")
            links_failed += 1
    
    logger.info(f"Historical interconnector links: {links_added} added, {links_failed} failed")
    
    # Remove external buses that have no links (e.g., Isle of Man with no flow data)
    # External buses are those with country != 'GB' (not country == 'External')
    external_buses = network.buses[network.buses['country'] != 'GB'].index
    orphaned_buses = [bus for bus in external_buses if bus not in external_buses_with_links]
    
    if orphaned_buses:
        logger.info(f"Removing {len(orphaned_buses)} external bus(es) with no interconnector links:")
        for bus in orphaned_buses:
            # Find the interconnector name from the original mapping
            ic_name = "Unknown"
            for idx, row in interconnectors_df.iterrows():
                if row['to_bus'] == bus:
                    ic_name = row['name']
                    break
            
            logger.info(f"  - Removed bus '{bus}' (interconnector: {ic_name}, reason: no flow data available)")
            # PyPSA 1.0.2: mremove() was replaced with remove()
            network.remove("Bus", bus)
            logger.info(f"  ✓ Successfully removed orphaned external bus: {bus}")
    
    if links_added > 0:
        # Add zero-cost generators on external buses to enable imports
        # For historical scenarios with FIXED flows, we need generators on external buses
        # to provide the supply potential for imports (negative p_set values).
        # These generators have:
        #   - p_nom matching interconnector capacity (to enable full import potential)
        #   - marginal_cost = 0.0 (cost is already zero on link; this is just supply potential)
        #   - carrier = EU_import (to track European supply separately)
        logger.info("Adding European supply generators on external buses (for import potential)...")
        
        # Ensure EU_import carrier exists
        if 'EU_import' not in network.carriers.index:
            network.add("Carrier", "EU_import", nice_name="European Imports", color="#3B5998")
            logger.info("  Added EU_import carrier")
        
        # Ensure EU_export carrier exists for export loads
        if 'EU_export' not in network.carriers.index:
            network.add("Carrier", "EU_export", nice_name="European Exports", color="#228B22")
            logger.info("  Added EU_export carrier")
        
        generators_added = 0
        loads_added = 0
        
        for link_name in network.links[network.links.index.str.startswith('IC_')].index:
            link = network.links.loc[link_name]
            external_bus = link.bus1  # bus1 is the external/foreign side
            capacity_mw = ZEROED_CAPACITY_MW if FORCE_ZERO_INTERCONNECTORS else link.p_nom
            
            # Create generator name from link name
            gen_name = f"EU_supply_{link_name.replace('IC_', '')}"
            
            # For historical scenarios with FIXED flows, we need:
            # 1. A generator on external bus to provide power for imports (when p_set < 0)
            # 2. A load on external bus to absorb power for exports (when p_set > 0)
            # Note: p_set < 0 means flow FROM external TO GB (import)
            #       p_set > 0 means flow FROM GB TO external (export)
            
            # Check if this is a fixed-flow scenario (has p_set)
            has_fixed_flows = link_name in network.links_t.p_set.columns
            
            if has_fixed_flows:
                # For fixed flows, generator/load just needs enough capacity
                # The actual flow is determined by p_set
                eu_marginal_cost = 0.0  # Zero cost - flows are fixed, cost doesn't matter
                
                # Add generator on external bus (for imports)
                network.add(
                    "Generator",
                    gen_name,
                    bus=external_bus,
                    p_nom=capacity_mw,  # Match interconnector capacity
                    marginal_cost=eu_marginal_cost,
                    carrier="EU_import",
                    p_min_pu=0.0,
                    p_max_pu=1.0
                )
                generators_added += 1
                
                # Apply ENTSO-E time-varying MC if available (for price signal accuracy)
                counterparty = link.get('counterparty_country', '')
                if (entsoe_prices_df is not None and len(entsoe_prices_df) > 0
                        and counterparty in entsoe_prices_df.columns):
                    _apply_time_varying_mc(
                        network, gen_name,
                        entsoe_prices_df[counterparty],
                        logger,
                    )
                
                # Add load on external bus (for exports)
                # This absorbs power when GB exports to Europe
                load_name = f"EU_demand_{link_name.replace('IC_', '')}"
                network.add(
                    "Load",
                    load_name,
                    bus=external_bus,
                    p_set=0.0,  # Will be overwritten by time-varying p_set below
                    carrier="EU_export"
                )
                loads_added += 1
                
                # CRITICAL: Set time-varying p_set for EU_demand to absorb exports
                # When link p_set > 0 (export from GB to EU), EU_demand must absorb that power
                # When link p_set < 0 (import to GB), EU_demand = 0 (EU_supply provides power)
                link_p_set = network.links_t.p_set[link_name]
                export_flows = link_p_set.clip(lower=0)  # Only positive values (exports)
                
                # Add to loads_t.p_set (create if needed)
                if network.loads_t.p_set.empty:
                    network.loads_t.p_set = pd.DataFrame(index=network.snapshots)
                network.loads_t.p_set[load_name] = export_flows
                
                export_total = export_flows.sum() / 1e6  # TWh
                if export_total > 0:
                    logger.debug(f"  Added {gen_name} + {load_name} on {external_bus}: {capacity_mw:.0f} MW (FIXED flows, {export_total:.2f} TWh exports)")
                else:
                    logger.debug(f"  Added {gen_name} + {load_name} on {external_bus}: {capacity_mw:.0f} MW (FIXED flows)")
            else:
                # For optimized flows, use economic marginal cost
                eu_marginal_cost = 45.0
                if FORCE_ZERO_INTERCONNECTORS:
                    eu_marginal_cost = 1e6
                
                network.add(
                    "Generator",
                    gen_name,
                    bus=external_bus,
                    p_nom=capacity_mw,
                    marginal_cost=eu_marginal_cost,
                    carrier="EU_import",
                    p_min_pu=0.0,
                    p_max_pu=1.0
                )
                generators_added += 1
                logger.debug(f"  Added {gen_name} on {external_bus}: {capacity_mw:.0f} MW @ £{eu_marginal_cost:.2f}/MWh")
        
        logger.info(f"✓ Added {generators_added} European supply generators")
        if loads_added > 0:
            logger.info(f"✓ Added {loads_added} European demand loads (for export absorption)")
        
        # Calculate and log total EXPECTED imports/exports from historical data
        if hasattr(network, 'historical_interconnector_flows'):
            total_expected_import = sum(
                flows.sum() / 1e6  # Hourly data: sum gives MWh, /1e6 for TWh
                for flows in network.historical_interconnector_flows.values()
            )
            logger.info(
                f"Historical interconnector data: {total_expected_import:.2f} TWh net import "
                f"(flows are FIXED via p_set)"
            )
        else:
            logger.warning("No historical interconnector flows stored")


def validate_network_with_interconnectors(network: pypsa.Network) -> bool:
    """
    Validate the network after adding interconnectors.
    
    Checks:
    1. Network consistency
    2. External generators exist on external buses
    3. Interconnector links have valid bus connections
    4. No unbounded power sources/sinks
    
    Args:
        network: PyPSA network object
        
    Returns:
        True if validation passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Basic consistency check
        network.consistency_check()
        logger.info("✓ Network consistency check passed")
        
        # Check interconnector links
        ic_links = network.links[network.links.index.str.startswith('IC_')]
        logger.info(f"✓ Network contains {len(ic_links)} interconnector links")
        
        if len(ic_links) > 0:
            if 'interconnector_pair_id' in ic_links.columns:
                paired = ic_links.dropna(subset=['interconnector_pair_id'])
                total_ic_capacity = paired.groupby(
                    'interconnector_pair_id'
                )['p_nom'].first().sum()
            else:
                total_ic_capacity = ic_links['p_nom'].sum()
            logger.info(f"  Total interconnector capacity: {total_ic_capacity:.1f} MW")
            
            # Check for valid bus connections
            invalid_links = 0
            for link_name, link in ic_links.iterrows():
                bus0_exists = link['bus0'] in network.buses.index
                bus1_exists = link['bus1'] in network.buses.index
                
                if not (bus0_exists and bus1_exists):
                    logger.error(f"✗ Invalid bus connection for link {link_name}")
                    invalid_links += 1
            
            if invalid_links > 0:
                logger.error(f"✗ Found {invalid_links} links with invalid bus connections")
                return False
            
            logger.info(f"✓ All interconnector links have valid bus connections")
            
            direct_pricing = (
                'foreign_market_pricing' in ic_links.columns
                and ic_links['foreign_market_pricing']
                .astype(str).str.lower().isin(['direct_link', 'piecewise_bidirectional']).any()
            )
            if direct_pricing:
                if 'foreign_market_role' not in network.generators.columns:
                    logger.error(
                        'Direct-link pricing requires neutral foreign market balances'
                    )
                    return False
                market_balances = network.generators[
                    network.generators['foreign_market_role']
                    .astype(str).str.lower().eq('market_balance')
                ]
                external_buses = {
                    bus
                    for bus in set(ic_links['bus0']) | set(ic_links['bus1'])
                    if 'External' in str(bus)
                }
                missing_balances = (
                    external_buses - set(market_balances['bus'].unique())
                )
                if missing_balances:
                    logger.error(
                        'External buses without neutral market balances: %s',
                        missing_balances,
                    )
                    return False
                invalid_balances = market_balances[
                    (market_balances['p_min_pu'] != -1.0)
                    | (market_balances['p_max_pu'] != 1.0)
                    | (market_balances['marginal_cost'] != 0.0)
                ]
                if not invalid_balances.empty:
                    logger.error(
                        'Foreign market balances must be signed and zero-cost: %s',
                        invalid_balances.index.tolist(),
                    )
                    return False
                logger.info(
                    'All %d external buses use bounded neutral market balances; '
                    'import asks and export bids are priced directly on Links',
                    len(external_buses),
                )
                return True

            # Check for external generators
            eu_generators = network.generators[
                network.generators.index.str.startswith('EU_supply_')
            ]
            eu_export_sinks = network.generators[
                network.generators.index.str.startswith('EU_export_sink_')
            ]
            
            if len(eu_generators) == 0:
                logger.warning("⚠ No European supply generators found on external buses")
                logger.warning("  External buses may act as unbounded sources/sinks")
            else:
                logger.info(f"✓ Found {len(eu_generators)} European supply generators")
                
                # Check that each external bus has a generator
                external_buses = {
                    bus
                    for bus in set(ic_links['bus0']) | set(ic_links['bus1'])
                    if 'External' in str(bus)
                }
                buses_with_generators = set(eu_generators['bus'].unique())
                
                missing_generators = external_buses - buses_with_generators
                if missing_generators:
                    logger.warning(f"⚠ External buses without generators: {missing_generators}")
                else:
                    logger.info(f"✓ All {len(external_buses)} external buses have supply generators")
                
                directional = (
                    'interconnector_direction' in ic_links.columns
                    and ic_links['interconnector_direction'].notna().any()
                )
                if directional:
                    buses_with_export_sinks = set(eu_export_sinks['bus'].unique())
                    missing_export_sinks = external_buses - buses_with_export_sinks
                    if missing_export_sinks:
                        logger.error(
                            'External buses without export sinks: %s',
                            missing_export_sinks,
                        )
                        return False

                # Log generator details
                for gen_name, gen in eu_generators.iterrows():
                    logger.debug(
                        f"  {gen_name}: bus={gen['bus']}, "
                        f"p_nom={gen['p_nom']:.0f} MW, "
                        f"marginal_cost={gen['marginal_cost']:.2f} £/MWh"
                    )
            
            # Check link marginal costs (should be zero or very low)
            high_cost_links = ic_links[ic_links['marginal_cost'] > 5.0]
            if len(high_cost_links) > 0:
                logger.warning(
                    f"⚠ {len(high_cost_links)} interconnector links have high marginal costs (>£5/MWh)"
                )
                logger.warning(
                    "  Expected: near-zero costs with European supply on external generators"
                )
                for link_name, link in high_cost_links.iterrows():
                    logger.warning(f"    {link_name}: £{link['marginal_cost']:.2f}/MWh")
            else:
                logger.info(f"✓ All interconnector links have appropriate marginal costs (≤£5/MWh)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Network validation failed: {e}")
        return False

def main():
    """Main processing function."""
    logger = setup_logging("add_interconnectors_to_network")
    start_time = time.time()
    
    try:
        logger.info("Starting interconnector network integration...")
        logger.info(f"Running in {'Snakemake' if SNAKEMAKE_MODE else 'standalone'} mode")
        
        if SNAKEMAKE_MODE:
            network_file = input_network
            mapped_file = input_mapped
            availability_file = input_availability
            price_diff_file = input_price_diff
            historical_flows_file = input_historical_flows
            entsoe_prices_file = input_entsoe_prices
            output_file = output_network
            is_historical_scenario = is_historical
            target_network_model = "Unknown"
        else:
            # Standalone mode defaults (future scenario)
            network_file = "resources/network/ETYS_base_demand.nc"
            mapped_file = "resources/interconnectors/interconnectors_mapped_ETYS.csv"
            availability_file = "resources/interconnectors/interconnector_availability.csv"
            price_diff_file = "resources/interconnectors/price_differentials_2024.csv"
            historical_flows_file = None
            entsoe_prices_file = None
            output_file = "resources/network/ETYS_with_interconnectors.nc"
            is_historical_scenario = False
            target_network_model = "ETYS"
        
        logger.info(f"Base network file: {network_file}")
        logger.info(f"Mapped interconnectors: {mapped_file}")
        logger.info(f"Scenario type: {'HISTORICAL' if is_historical_scenario else 'FUTURE'}")
        logger.info(f"Modelled year: {modelled_year}")
        if scenario_source == 'CP30':
            logger.info(f"CP30 pathway: {cp30_pathway}")
            logger.info(f"Spatial/pricing FES proxy pathway: {fes_pathway}")
        else:
            logger.info(f"FES pathway: {fes_pathway}")
        
        if is_historical_scenario:
            if historical_flow_mode not in {'fixed', 'market_coupled'}:
                raise ValueError(
                    "interconnectors.historical_flow_mode must be 'fixed' or "
                    f"'market_coupled', got {historical_flow_mode!r}"
                )
            logger.info(f"Historical flow mode: {historical_flow_mode}")
            logger.info(f"Historical ESPENI reference flows: {historical_flows_file}")
        else:
            logger.info(f"Availability profiles: {availability_file}")
            logger.info(f"Price differentials: {price_diff_file}")
        
        logger.info(f"Output network: {output_file}")
        logger.info(f"Network model: {target_network_model}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Load base network
        if not Path(network_file).exists():
            raise FileNotFoundError(f"Base network file not found: {network_file}")
        
        logger.info("Loading base PyPSA network...")
        network = load_network(network_file, custom_logger=logger)
        logger.info(f"Loaded network with {len(network.buses)} buses, {len(network.lines)} lines")
        
        # Add carrier definitions to ensure all carriers are defined
        try:
            network = add_carriers_to_network(network, logger)
        except Exception as e:
            logger.warning(f"Could not add carrier definitions: {e}")
        
        # Load mapped interconnector data
        if not Path(mapped_file).exists():
            logger.warning(f"Mapped interconnectors file not found: {mapped_file}")
            interconnectors_df = pd.DataFrame()
        else:
            interconnectors_df = pd.read_csv(mapped_file)
            logger.info(f"Loaded {len(interconnectors_df)} mapped interconnectors")
        
        if len(interconnectors_df) > 0:
            # Get modelled year for commissioning filtering
            scenario_year = modelled_year if SNAKEMAKE_MODE else None
            if scenario_year is not None:
                logger.info(f"Modelled year: {scenario_year}")
            
            # The opt-in historical market-coupled calibration uses ESPENI's
            # commissioning metadata as an availability registry. Unlike the
            # legacy fixed-flow path, unknown-year cables are excluded rather
            # than assumed available.
            if is_historical_scenario and historical_flow_mode == 'market_coupled':
                interconnectors_df = enrich_historical_commissioning_years(
                    interconnectors_df, input_historical_flows_metadata
                )
                interconnectors_df = restrict_market_coupled_historical_interconnectors(
                    interconnectors_df, scenario_year
                )
            else:
                interconnectors_df = filter_interconnectors_by_commissioning_year(
                    interconnectors_df,
                    scenario_year=scenario_year,
                    is_historical=is_historical_scenario,
                )
            
            # CP30 cable set from the declared composition (15 Oct 2026): CP30's nominal
            # ratings, MaresConnect added, the Isle of Man cable outside the count; no
            # pro-rata uplift. Before the external buses, so an added cable gets its own.
            if not is_historical_scenario and scenario_source == 'CP30' and len(interconnectors_df) > 0:
                if not input_cp30_composition:
                    raise ValueError("A future CP30 build needs the cp30_interconnector_composition input "
                                     "(config/defaults/method/brownfield.yaml interconnectors.cp30_composition.file)")
                interconnectors_df = apply_cp30_interconnector_composition(
                    interconnectors_df, network, input_cp30_composition, input_cp30_es1,
                    cp30_pathway, modelled_year,
                )

            if len(interconnectors_df) == 0:
                logger.warning("No interconnectors remaining after commissioning year filter!")
            else:
                # Add external buses for interconnector endpoints
                add_external_buses(network, interconnectors_df)
                
                # Validate and fix coordinate system consistency
                # External buses must use OSGB36 coordinates to match ETYS network
                try:
                    from scripts.utilities.spatial_utils import ensure_osgb36_coordinates, validate_network_coordinates
                    validation = validate_network_coordinates(network, fix=False)
                    if validation['wgs84_count'] > 0:
                        logger.warning(f"Found {validation['wgs84_count']} buses with WGS84 coordinates - fixing...")
                        fixed = ensure_osgb36_coordinates(network)
                        logger.info(f"Coordinate normalization: Fixed {fixed} buses")
                    else:
                        logger.info(f"Coordinate validation: All {validation['osgb36_count']} buses use OSGB36 coordinates ✓")
                except ImportError as e:
                    logger.warning(f"Could not validate coordinates (spatial_utils import failed): {e}")
            
            # Branch based on scenario type
            #
            # Optionally load ENTSO-E hourly prices for time-varying EU_supply MC
            entsoe_prices_df = pd.DataFrame()
            if entsoe_prices_file:
                entsoe_prices_df = load_entsoe_prices(entsoe_prices_file)
                if len(entsoe_prices_df) > 0:
                    logger.info(f"ENTSO-E prices loaded: {len(entsoe_prices_df)} hours × {len(entsoe_prices_df.columns)} countries")
                else:
                    logger.warning("ENTSO-E price file could not be loaded — falling back to derived MC")

            if (
                is_historical_scenario
                and historical_flow_mode == 'fixed'
                and len(interconnectors_df) > 0
            ):
                # HISTORICAL DEFAULT: use observed ESPENI flows as fixed constraints.
                logger.info("=== HISTORICAL SCENARIO: Using fixed ESPENI interconnector flows ===")
                
                if not historical_flows_file or not Path(historical_flows_file).exists():
                    raise FileNotFoundError(
                        f"Historical scenario requires ESPENI flow data: {historical_flows_file}"
                    )
                
                historical_flows_df = load_historical_flows(historical_flows_file)
                add_historical_interconnector_links(
                    network, 
                    interconnectors_df, 
                    historical_flows_df,
                    entsoe_prices_df=entsoe_prices_df if len(entsoe_prices_df) > 0 else None
                )
                
            elif len(interconnectors_df) > 0:
                # Future and opt-in historical back-tests use optimizable Links.
                # Historical market-coupled mode retains model-year cable capacity;
                # it never applies FES/CP30 scaling.
                if is_historical_scenario:
                    logger.info(
                        "=== HISTORICAL MARKET-COUPLED BACK-TEST: Optimizing flows against ESPENI reference ==="
                    )
                    if not historical_flows_file or not Path(historical_flows_file).exists():
                        raise FileNotFoundError(
                            "Historical market-coupled back-test requires ESPENI reference flows: "
                            f"{historical_flows_file}"
                        )
                else:
                    logger.info("=== FUTURE SCENARIO: Using optimizable interconnector flows ===")
                
                # Scale only forward scenarios. CP30 uses ES1 aggregate MW; FES is
                # only used for pricing/availability inputs.
                if not is_historical_scenario and scenario_source == 'CP30':
                    logger.info("CP30 interconnectors from the declared composition: no pro-rata scaling")
                elif not is_historical_scenario:
                    interconnectors_df = scale_interconnectors_to_fes(
                        interconnectors_df,
                        modelled_year=modelled_year,
                        fes_pathway=fes_pathway,
                        fes_data_path=input_fes_capacity_data or None,
                    )
                    if input_fes_capacity_data:
                        # HT50_*: a silent fall-back to unscaled cables is wrong, not degraded.
                        _fes_ic = pd.read_csv(input_fes_capacity_data, low_memory=False)
                        _target = float(_fes_ic.loc[
                            _fes_ic['Building Block ID Number'].eq('Gen_BB022')
                            & _fes_ic['FES Pathway'].eq(fes_pathway), str(modelled_year)].sum())
                        _total = float(interconnectors_df['capacity_mw'].sum())
                        if abs(_total - _target) > 1.0:
                            raise ValueError(
                                f"Interconnectors total {_total:,.1f} MW after FES scaling, but "
                                f"{input_fes_capacity_data} gives {_target:,.1f} MW for {fes_pathway} {modelled_year}")
                
                # Historical back-tests use commissioned capacity throughout.
                # Documented outage data can later replace this all-available
                # assumption without changing the default fixed-flow path.
                availability_df = (
                    pd.DataFrame()
                    if is_historical_scenario
                    else load_availability_profiles(availability_file)
                )
                
                # Load price differentials (optional)
                price_differentials_df = pd.DataFrame()
                if price_diff_file and Path(price_diff_file).exists():
                    price_differentials_df = load_price_differentials(price_diff_file)
                elif price_diff_file:
                    logger.warning(f"Price differential file not found: {price_diff_file}")
                
                # CP30 scenarios take boundary trade prices from the authoritative
                # CP.29 workbook extract. FES remains only an explicitly declared
                # spatial proxy elsewhere in the CP30 workflow.
                effective_pricing_config = pricing_config
                if str(effective_pricing_config.get("source", "derived")).lower() == "cp30":
                    cp30_prices = load_cp30_interconnector_prices(input_cp30_interconnector_prices)
                    matched = cp30_prices[
                        (pd.to_numeric(cp30_prices["year"], errors="coerce") == int(modelled_year))
                        & (cp30_prices["pathway"].astype(str).str.strip().str.lower()
                           == str(cp30_pathway).strip().lower())
                    ]
                    if len(matched) != 1:
                        raise ValueError(
                            "CP30 pricing requires exactly one CP.29 price row for "
                            f"year={modelled_year}, pathway={cp30_pathway!r}; found {len(matched)}"
                        )
                    effective_pricing_config = dict(effective_pricing_config)
                    effective_pricing_config["cp30_import_price_gbp_per_mwh"] = float(
                        matched.iloc[0]["import_price_gbp_per_mwh"]
                    )
                    effective_pricing_config["cp30_export_price_gbp_per_mwh"] = float(
                        matched.iloc[0]["export_price_gbp_per_mwh"]
                    )
                    logger.info(
                        "CP30 CP.29 interconnector prices: import %.2f, export %.2f GBP/MWh",
                        effective_pricing_config["cp30_import_price_gbp_per_mwh"],
                        effective_pricing_config["cp30_export_price_gbp_per_mwh"],
                    )
                # Add optimizable interconnector links
                add_interconnector_links(
                    network, 
                    interconnectors_df, 
                    availability_df, 
                    price_differentials_df,
                    modelled_year=modelled_year,
                    fes_pathway=fes_pathway,
                    entsoe_prices_df=entsoe_prices_df if len(entsoe_prices_df) > 0 else None,
                    pricing_config=effective_pricing_config,
                )
                tranche_config = effective_pricing_config.get("volume_tranches") or {}
                if tranche_config.get("enabled", False):
                    # Item IC1: the foreign price responds to GB's trade volume.
                    from scripts.interconnectors.volume_tranches import (
                        apply_volume_tranches,
                        load_market_demand_twh,
                    )
                    if not input_market_demand:
                        raise ValueError("volume_tranches is enabled but no market_demand input was given")
                    demand_twh = load_market_demand_twh(
                        input_market_demand,
                        tranche_config["eu_scenario"],
                        tranche_config["pathway_alignment"],
                        int(tranche_config["demand_year"]),
                    )
                    apply_volume_tranches(network, tranche_config, demand_twh, logger=logger)

                # Apply availability profiles
                if len(availability_df) > 0:
                    apply_availability_profiles(network, availability_df)
            
            # Validate the updated network
            if not validate_network_with_interconnectors(network):
                raise ValueError("Network validation failed after adding interconnectors")
        else:
            logger.warning("No interconnectors to add - saving network unchanged")
        
        # DEBUG: Check p_set before saving
        ic_links_in_network = [l for l in network.links.index if l.startswith('IC_')]
        logger.info(f"DEBUG: Before save - IC links in network: {len(ic_links_in_network)}")
        if len(ic_links_in_network) > 0:
            p_set_cols = [c for c in network.links_t.p_set.columns if c.startswith('IC_')]
            logger.info(f"DEBUG: p_set columns for IC links: {len(p_set_cols)}")
            if p_set_cols:
                logger.info(f"DEBUG: Sample p_set values for {p_set_cols[0]}: "
                          f"mean={network.links_t.p_set[p_set_cols[0]].mean():.1f}, "
                          f"nonzero={( network.links_t.p_set[p_set_cols[0]] != 0).sum()}")
        
        # Save the updated network
        save_network(network, output_file, custom_logger=logger)
        
        # DEBUG: Check if p_set was saved
        logger.info("DEBUG: Verifying p_set was saved to NetCDF...")
        verify_n = load_network(output_file, custom_logger=logger)
        verify_p_set_cols = [c for c in verify_n.links_t.p_set.columns if c.startswith('IC_')]
        logger.info(f"DEBUG: After reload - p_set columns: {len(verify_p_set_cols)}")
        if len(verify_p_set_cols) == 0 and len(p_set_cols) > 0:
            logger.error(f"WARNING: p_set was NOT saved! Had {len(p_set_cols)} columns before save, 0 after!")
        
        logger.info(f"Saved network with interconnectors to: {output_file}")
        
        # Log final summary
        ic_links = network.links[network.links.index.str.startswith('IC_')]
        logger.info(f"Final network summary:")
        logger.info(f"  Buses: {len(network.buses)}")
        logger.info(f"  Lines: {len(network.lines)}")
        logger.info(f"  Interconnector links: {len(ic_links)}")
        
        if len(ic_links) > 0:
            if 'interconnector_pair_id' in ic_links.columns:
                total_capacity = (
                    ic_links.dropna(subset=['interconnector_pair_id'])
                    .groupby('interconnector_pair_id')['p_nom']
                    .first()
                    .sum()
                )
            else:
                total_capacity = ic_links['p_nom'].sum()
            logger.info(f"  Total interconnector capacity: {total_capacity:.1f} MW")
            
            # Check if historical validation data is available
            if hasattr(network, 'historical_interconnector_flows'):
                logger.info(
                    f"  Flow type: OPTIMIZABLE with historical validation data "
                    f"({len(network.historical_interconnector_flows)} flows stored)"
                )
            else:
                logger.info(f"  Flow type: OPTIMIZABLE (no historical validation data)")
        
        # Execution summary
        log_execution_summary(
            logger,
            "Add Interconnectors to Network",
            start_time,
            inputs={
                'network': network_file,
                'interconnectors': mapped_file,
                'availability': availability_file if not is_historical_scenario else None,
                'historical_flows': historical_flows_file if is_historical_scenario else None
            },
            outputs={'network': output_file},
            context={
                'interconnectors_added': len(interconnectors_df),
                'total_capacity_mw': float(interconnectors_df['capacity_mw'].sum()),
                'external_buses_created': len([b for b in network.buses.index if '_external' in str(b)]),
                'scenario_type': 'historical' if is_historical_scenario else 'future',
                'is_historical': is_historical_scenario
            }
        )
        logger.info("Interconnector network integration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in interconnector network integration: {e}")
        if SNAKEMAKE_MODE:
            raise
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()

