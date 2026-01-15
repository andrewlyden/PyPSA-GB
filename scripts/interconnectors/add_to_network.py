#!/usr/bin/env python3
"""
Add Interconnectors to PyPSA Network
====================================

This script integrates interconnectors into PyPSA networks as DC links with
proper representation of European electricity supply.

Architecture:
- External buses represent connection points to European countries
- Large generators on external buses represent European electricity supply
  with marginal costs based on European generation mix data
- DC links connect GB buses to external buses with near-zero marginal cost
  (only transmission losses) since economics are handled by external generators
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
- Bidirectional link configuration
- Historical flow fixing for validation scenarios
- Network validation and consistency checking

Author: PyPSA-GB Team
"""

import sys
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
    output_network = snakemake.output[0]
    is_historical = snakemake.params.get('is_historical', False)
    modelled_year = snakemake.params.get('modelled_year', None)
    fes_pathway = snakemake.params.get('fes_pathway', None)
else:
    SNAKEMAKE_MODE = False
    modelled_year = None
    fes_pathway = None

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


# Country name mapping for European price data
# Maps interconnector country names to price differential country names
COUNTRY_PRICE_MAPPING = {
    'ireland': 'sem',
    'northern ireland': 'sem',
    'isle of man': 'sem',  # Use SEM prices as proxy for Isle of Man
}


def add_external_generators(network: pypsa.Network,
                           interconnectors_df: pd.DataFrame,
                           price_differentials_df: pd.DataFrame = None,
                           modelled_year: int = None,
                           fes_pathway: str = None) -> None:
    """
    Add generators on external buses to represent European electricity supply.
    
    This models the European generation stack as a generator at each external bus
    with a marginal cost representing the European wholesale price for the specific
    modelled year and FES pathway. Generator capacity is limited to the total
    interconnector capacity to each country, providing a realistic constraint.
    
    Args:
        network: PyPSA network object
        interconnectors_df: Mapped interconnector data with external buses
        price_differentials_df: European price differentials (optional)
        modelled_year: The year being modelled (for filtering price data)
        fes_pathway: The FES pathway name (e.g., 'Holistic Transition')
    """
    logger = logging.getLogger(__name__)
    
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
            marginal_cost=marginal_cost,  # European wholesale price
            carrier='EU_import',
            # Additional metadata
            country=country,
            source='European generation mix'
        )
        
        generators_added += 1
        logger.info(f"  Added EU generator: {gen_name}, capacity={generator_capacity:.0f} MW, "
                   f"marginal_cost=£{marginal_cost:.2f}/MWh")
    
    logger.info(f"Added {generators_added} European supply generators on external buses")


def add_interconnector_links(network: pypsa.Network, 
                             interconnectors_df: pd.DataFrame,
                             availability_df: pd.DataFrame,
                             price_differentials_df: pd.DataFrame = None,
                             modelled_year: int = None,
                             fes_pathway: str = None) -> None:
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
        fes_pathway=fes_pathway
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
    
    # Create availability time series for each link
    profiles_applied = 0
    profiles_defaulted = 0
    
    for link_name in ic_links.index:
        # Extract interconnector name from link name
        ic_name = link_name.replace('IC_', '')
        
        # Find matching availability profile
        link_availability = availability_df[availability_df['name'] == ic_name].copy()
        
        if len(link_availability) > 0:
            # Create hour-of-year index for matching across years
            link_availability['hour_of_year'] = (
                link_availability['time'].dt.dayofyear * 24 + 
                link_availability['time'].dt.hour
            )
            
            # Create the same for network snapshots
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
            
            # Add time-varying p_max_pu
            network.links_t.p_max_pu[link_name] = p_max_pu_values
            
            profiles_applied += 1
            logger.debug(f"Applied availability profile to {link_name} "
                        f"(mean={p_max_pu_values.mean():.3f})")
        else:
            # No availability profile found - set default p_max_pu=1.0 for all snapshots
            network.links_t.p_max_pu[link_name] = pd.Series(
                1.0, 
                index=network_snapshots
            )
            profiles_defaulted += 1
            logger.debug(f"No availability profile for {ic_name} - using default p_max_pu=1.0")
    
    logger.info(f"Availability profiles: {profiles_applied} applied, {profiles_defaulted} defaulted to 1.0")


def add_historical_interconnector_links(network: pypsa.Network,
                                       interconnectors_df: pd.DataFrame,
                                       historical_flows_df: pd.DataFrame) -> None:
    """
    Add interconnector links with FIXED historical flows from ESPENI.
    
    For historical scenarios, interconnector flows are actual observations
    and should not be optimized. Positive flows = imports, negative = exports.
    
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
                logger.warning(
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
            
            # Check for external generators
            eu_generators = network.generators[
                network.generators.index.str.startswith('EU_supply_')
            ]
            
            if len(eu_generators) == 0:
                logger.warning("⚠ No European supply generators found on external buses")
                logger.warning("  External buses may act as unbounded sources/sinks")
            else:
                logger.info(f"✓ Found {len(eu_generators)} European supply generators")
                
                # Check that each external bus has a generator
                external_buses = set(ic_links['bus1'].unique())
                buses_with_generators = set(eu_generators['bus'].unique())
                
                missing_generators = external_buses - buses_with_generators
                if missing_generators:
                    logger.warning(f"⚠ External buses without generators: {missing_generators}")
                else:
                    logger.info(f"✓ All {len(external_buses)} external buses have supply generators")
                
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
            output_file = "resources/network/ETYS_with_interconnectors.nc"
            is_historical_scenario = False
            target_network_model = "ETYS"
        
        logger.info(f"Base network file: {network_file}")
        logger.info(f"Mapped interconnectors: {mapped_file}")
        logger.info(f"Scenario type: {'HISTORICAL' if is_historical_scenario else 'FUTURE'}")
        logger.info(f"Modelled year: {modelled_year}")
        logger.info(f"FES pathway: {fes_pathway}")
        
        if is_historical_scenario:
            logger.info(f"Historical flows: {historical_flows_file}")
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
            
            # Filter interconnectors by commissioning year for historical scenarios
            # This ensures only interconnectors that existed in the modelled year are included
            interconnectors_df = filter_interconnectors_by_commissioning_year(
                interconnectors_df, 
                scenario_year=scenario_year,
                is_historical=is_historical_scenario
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
            if is_historical_scenario and len(interconnectors_df) > 0:
                # HISTORICAL: Use fixed flows from ESPENI
                logger.info("=== HISTORICAL SCENARIO: Using fixed interconnector flows ===")
                
                if not historical_flows_file or not Path(historical_flows_file).exists():
                    raise FileNotFoundError(
                        f"Historical scenario requires ESPENI flow data: {historical_flows_file}"
                    )
                
                historical_flows_df = load_historical_flows(historical_flows_file)
                add_historical_interconnector_links(
                    network, 
                    interconnectors_df, 
                    historical_flows_df
                )
                
            elif len(interconnectors_df) > 0:
                # FUTURE: Use optimizable links with availability and prices
                logger.info("=== FUTURE SCENARIO: Using optimizable interconnector flows ===")
                
                # Scale interconnector capacities to FES projections
                interconnectors_df = scale_interconnectors_to_fes(
                    interconnectors_df,
                    modelled_year=modelled_year,
                    fes_pathway=fes_pathway
                )
                
                # Load availability profiles
                availability_df = load_availability_profiles(availability_file)
                
                # Load price differentials (optional)
                price_differentials_df = pd.DataFrame()
                if price_diff_file and Path(price_diff_file).exists():
                    price_differentials_df = load_price_differentials(price_diff_file)
                elif price_diff_file:
                    logger.warning(f"Price differential file not found: {price_diff_file}")
                
                # Add optimizable interconnector links
                add_interconnector_links(
                    network, 
                    interconnectors_df, 
                    availability_df, 
                    price_differentials_df,
                    modelled_year=modelled_year,
                    fes_pathway=fes_pathway
                )
                
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

