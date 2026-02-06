"""
Define carrier attributes for PyPSA-GB generators.

This module provides carrier definitions for all generator types in the model.
Carriers define technology-specific attributes including:
- Visualization (color, nice_name)
- Emissions (co2_emissions in t_CO2/MWh_thermal)
- Constraints (max_growth, max_relative_growth)

Author: PyPSA-GB Team
Date: October 2025
"""

import pandas as pd

def get_carrier_definitions():
    """
    Return carrier definitions for all generator types.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with carrier attributes indexed by carrier name
    """
    
    # Carrier definitions
    # co2_emissions: t_CO2/MWh_thermal (will be divided by efficiency to get t_CO2/MWh_electrical)
    carriers = {
        # Renewable - Wind (standardized names)
        'wind_onshore': {
            'color': '#3B6182',  # Dark blue
            'nice_name': 'Onshore Wind',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'wind_offshore': {
            'color': '#6BAED6',  # Light blue
            'nice_name': 'Offshore Wind', 
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        # Renewable - Wind (legacy names with parentheses)
        'Wind (Onshore)': {
            'color': '#3B6182',  # Dark blue
            'nice_name': 'Onshore Wind',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Wind (Offshore)': {
            'color': '#6BAED6',  # Light blue
            'nice_name': 'Offshore Wind', 
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Renewable - Solar
        'solar_pv': {
            'color': '#FFBB00',  # Yellow/gold
            'nice_name': 'Solar PV',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Solar': {
            'color': '#FFBB00',  # Yellow/gold
            'nice_name': 'Solar PV',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Renewable - Hydro
        'small_hydro': {
            'color': '#08519C',  # Navy blue
            'nice_name': 'Small Hydro',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'large_hydro': {
            'color': '#0868AC',  # Medium blue
            'nice_name': 'Large Hydro',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Hydro': {
            'color': '#0868AC',  # Medium blue
            'nice_name': 'Hydro',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Hydro / pumped storage': {
            'color': '#084594',  # Dark blue
            'nice_name': 'Pumped Hydro Storage',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Pumped Storage': {
            'color': '#084594',  # Dark blue
            'nice_name': 'Pumped Hydro Storage',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'pumped_hydro': {
            'color': '#084594',  # Dark blue
            'nice_name': 'Pumped Hydro Storage',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Renewable - Marine
        'tidal_stream': {
            'color': '#4EB3D3',  # Cyan
            'nice_name': 'Tidal Stream',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'shoreline_wave': {
            'color': '#7BCCC4',  # Turquoise
            'nice_name': 'Wave Power',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'tidal_lagoon': {
            'color': '#A8DDB5',  # Light cyan
            'nice_name': 'Tidal Lagoon',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Renewable - Geothermal
        'geothermal': {
            'color': '#D95F0E',  # Orange
            'nice_name': 'Geothermal',
            'co2_emissions': 0.0,  # Very low emissions
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Biomass and Waste
        'biomass': {
            'color': '#238B45',  # Dark green
            'nice_name': 'Biomass',
            'co2_emissions': 0.0,  # Carbon neutral assumption (grown = burned)
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Bioenergy': {
            'color': '#238B45',  # Dark green
            'nice_name': 'Bioenergy',
            'co2_emissions': 0.0,  # Carbon neutral assumption
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'waste_to_energy': {
            'color': '#66C2A4',  # Light green
            'nice_name': 'Waste to Energy',
            'co2_emissions': 0.2,  # Some CO2 from non-biogenic waste
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'advanced_biofuel': {
            'color': '#41AB5D',  # Medium green
            'nice_name': 'Advanced Biofuel',
            'co2_emissions': 0.0,  # Carbon neutral
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Gas-based Biomass
        'biogas': {
            'color': '#74C476',  # Light green
            'nice_name': 'Biogas',
            'co2_emissions': 0.0,  # Carbon neutral
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'landfill_gas': {
            'color': '#A1D99B',  # Very light green
            'nice_name': 'Landfill Gas',
            'co2_emissions': 0.0,  # Carbon neutral (would be released anyway)
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'sewage_gas': {
            'color': '#C7E9C0',  # Pale green
            'nice_name': 'Sewage Gas',
            'co2_emissions': 0.0,  # Carbon neutral
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Fossil Fuels - Gas
        'gas_reciprocating': {
            'color': '#969696',  # Gray
            'nice_name': 'Gas Reciprocating',
            'co2_emissions': 0.202,  # t_CO2/MWh_thermal for natural gas
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'CCGT': {
            'color': '#8B8B8B',  # Dark gray
            'nice_name': 'Combined Cycle Gas Turbine',
            'co2_emissions': 0.202,  # t_CO2/MWh_thermal for natural gas
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'ccgt': {
            'color': '#8B8B8B',  # Dark gray
            'nice_name': 'Combined Cycle Gas Turbine',
            'co2_emissions': 0.202,  # t_CO2/MWh_thermal for natural gas
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'OCGT': {
            'color': '#A9A9A9',  # Light gray
            'nice_name': 'Open Cycle Gas Turbine',
            'co2_emissions': 0.202,  # t_CO2/MWh_thermal for natural gas
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'ocgt': {
            'color': '#A9A9A9',  # Light gray
            'nice_name': 'Open Cycle Gas Turbine',
            'co2_emissions': 0.202,  # t_CO2/MWh_thermal for natural gas
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Fossil Fuels - Coal/Steam
        'Conventional steam': {
            'color': '#2F2F2F',  # Very dark gray (coal)
            'nice_name': 'Conventional Steam',
            'co2_emissions': 0.341,  # t_CO2/MWh_thermal for coal
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Conventional Steam': {
            'color': '#2F2F2F',  # Very dark gray (coal)
            'nice_name': 'Conventional Steam',
            'co2_emissions': 0.341,  # t_CO2/MWh_thermal for coal
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'conventional_steam': {
            'color': '#2F2F2F',  # Very dark gray (coal)
            'nice_name': 'Conventional Steam',
            'co2_emissions': 0.341,  # t_CO2/MWh_thermal for coal
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'coal': {
            'color': '#2F2F2F',  # Very dark gray
            'nice_name': 'Coal',
            'co2_emissions': 0.341,  # t_CO2/MWh_thermal for coal
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Nuclear
        'nuclear': {
            'color': '#CC4C02',  # Orange-red
            'nice_name': 'Nuclear',
            'co2_emissions': 0.0,  # Zero operational emissions
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'PWR': {
            'color': '#CC4C02',  # Orange-red
            'nice_name': 'Pressurized Water Reactor',
            'co2_emissions': 0.0,  # Zero operational emissions
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'AGR': {
            'color': '#D95F0E',  # Orange
            'nice_name': 'Advanced Gas-cooled Reactor',
            'co2_emissions': 0.0,  # Zero operational emissions
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Hydrogen and Fuel Cells
        'H2': {
            'color': '#00CED1',  # Dark turquoise
            'nice_name': 'Hydrogen Generation',
            'co2_emissions': 0.0,  # Zero emissions (green hydrogen) or depends on source
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'hydrogen': {
            'color': '#00CED1',  # Dark turquoise
            'nice_name': 'Hydrogen Generation',
            'co2_emissions': 0.0,  # Zero emissions (green hydrogen)
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'H2_gas': {
            'color': '#FF69B4',  # Hot pink
            'nice_name': 'Hydrogen Gas',
            'co2_emissions': 0.0,  # Zero emissions (energy carrier)
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'electrolysis': {
            'color': '#8A2BE2',  # Blue violet
            'nice_name': 'Electrolysis',
            'co2_emissions': 0.0,  # Zero emissions (power-to-gas)
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'H2_turbine': {
            'color': '#FF1493',  # Deep pink
            'nice_name': 'H2 Power Generation',
            'co2_emissions': 0.0,  # Zero emissions (hydrogen fuel)
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'fuel_cell': {
            'color': '#20B2AA',  # Light sea green
            'nice_name': 'Fuel Cell',
            'co2_emissions': 0.0,  # Depends on hydrogen source
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Combined Heat and Power (CHP)
        'CHP': {
            'color': '#B22222',  # Firebrick red
            'nice_name': 'Combined Heat & Power',
            'co2_emissions': 0.202,  # Assumes natural gas CHP
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'micro_CHP': {
            'color': '#CD5C5C',  # Indian red
            'nice_name': 'Micro CHP',
            'co2_emissions': 0.202,  # Assumes natural gas
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Gas Engines and Oil
        'gas_engine': {
            'color': '#708090',  # Slate gray
            'nice_name': 'Gas Engine',
            'co2_emissions': 0.202,  # Natural gas
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'oil': {
            'color': '#4A4A4A',  # Dark gray
            'nice_name': 'Oil',
            'co2_emissions': 0.264,  # t_CO2/MWh_thermal for oil
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Waste (FES naming)
        'waste': {
            'color': '#66C2A4',  # Light green
            'nice_name': 'Waste Incineration',
            'co2_emissions': 0.2,  # Some CO2 from non-biogenic waste
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # FES Wind Naming Aliases
        'onwind': {
            'color': '#3B6182',  # Dark blue (matches wind_onshore)
            'nice_name': 'Onshore Wind',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'offwind': {
            'color': '#6BAED6',  # Light blue (matches wind_offshore)
            'nice_name': 'Offshore Wind',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # FES Solar/Marine Naming Aliases
        'solar': {
            'color': '#FFBB00',  # Yellow/gold (matches solar_pv)
            'nice_name': 'Solar PV',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'marine': {
            'color': '#4EB3D3',  # Cyan (matches tidal_stream)
            'nice_name': 'Marine Energy',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'hydro': {
            'color': '#0868AC',  # Medium blue (matches large_hydro)
            'nice_name': 'Hydro',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Other/Unclassified
        'thermal_other': {
            'color': '#8C6BB1',  # Purple
            'nice_name': 'Other Thermal',
            'co2_emissions': 0.0,  # Unknown - assume low
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'unclassified': {
            'color': '#BDBDBD',  # Light gray
            'nice_name': 'Unclassified',
            'co2_emissions': 0.0,  # Unknown
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Load Shedding (Emergency Capacity)
        'load_shedding': {
            'color': '#FF0000',  # Red - critical/emergency
            'nice_name': 'Load Shedding',
            'co2_emissions': 0.0,  # Represents lost load, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Flexible Demand / Demand-Side Response
        'heat_pumps': {
            'color': '#E377C2',  # Pink - flexible heating demand
            'nice_name': 'Heat Pumps',
            'co2_emissions': 0.0,  # Electric load, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'EVs': {
            'color': '#17BECF',  # Cyan - flexible transport demand
            'nice_name': 'Electric Vehicles',
            'co2_emissions': 0.0,  # Electric load, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'electricity_demand': {
            'color': '#7F7F7F',  # Gray - base electricity demand
            'nice_name': 'Electricity Demand',
            'co2_emissions': 0.0,  # Electric load, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # European Imports (External Generators)
        'EU_import': {
            'color': '#9467BD',  # Purple - represents external/foreign source
            'nice_name': 'European Imports',
            'co2_emissions': 0.2,  # Approx. European grid average (varies by country/year)
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Energy Storage
        'Battery': {
            'color': '#FFD700',  # Gold
            'nice_name': 'Battery Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'battery': {
            'color': '#FFD700',  # Gold
            'nice_name': 'Battery Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'LAES': {
            'color': '#DAA520',  # Goldenrod
            'nice_name': 'Liquid Air Energy Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Liquid_Air_Energy_Storage': {
            'color': '#DAA520',  # Goldenrod
            'nice_name': 'Liquid Air Energy Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Liquid Air Energy Storage': {
            'color': '#DAA520',  # Goldenrod
            'nice_name': 'Liquid Air Energy Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'CAES': {
            'color': '#CD853F',  # Peru/tan color
            'nice_name': 'Compressed Air Energy Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Compressed Air Energy Storage': {
            'color': '#CD853F',  # Peru/tan color
            'nice_name': 'Compressed Air Energy Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Domestic Battery': {
            'color': '#FFEC8B',  # Light goldenrod - slightly different from grid battery
            'nice_name': 'Domestic Battery Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Flywheel': {
            'color': '#B8860B',  # Dark goldenrod
            'nice_name': 'Flywheel Storage',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Pumped_Storage_Hydroelectricity': {
            'color': '#084594',  # Dark blue
            'nice_name': 'Pumped Storage Hydroelectricity',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'Pumped Storage Hydroelectricity': {
            'color': '#084594',  # Dark blue
            'nice_name': 'Pumped Storage Hydroelectricity',
            'co2_emissions': 0.0,  # Storage system, not generation
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        
        # Network carriers (already exist but include for completeness)
        'AC': {
            'color': '#000000',
            'nice_name': 'AC',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        },
        'DC': {
            'color': '#000000',
            'nice_name': 'DC',
            'co2_emissions': 0.0,
            'max_growth': float('inf'),
            'max_relative_growth': 0.0
        }
    }
    
    # Convert to DataFrame
    carriers_df = pd.DataFrame.from_dict(carriers, orient='index')
    
    return carriers_df


def add_carriers_to_network(network, logger=None):
    """
    Add carrier definitions to PyPSA network.
    
    Will not overwrite existing carriers - only adds missing ones.
    
    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    pypsa.Network
        Network with carriers added
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Get carrier definitions
    carriers_df = get_carrier_definitions()
    
    # Check existing carriers
    existing_carriers = set(network.carriers.index) if not network.carriers.empty else set()
    
    logger.info(f"Existing carriers in network: {len(existing_carriers)}")
    if existing_carriers:
        logger.info(f"  {sorted(existing_carriers)}")
    
    # Add missing carriers
    carriers_to_add = set(carriers_df.index) - existing_carriers
    
    if carriers_to_add:
        logger.info(f"Adding {len(carriers_to_add)} new carriers:")
        for carrier in sorted(carriers_to_add):
            carrier_attrs = carriers_df.loc[carrier].to_dict()
            network.add("Carrier", carrier, **carrier_attrs)
            logger.info(f"  + {carrier}: {carrier_attrs['nice_name']}")
    else:
        logger.info("All carriers already defined in network")
    
    # Verify all generator carriers are defined
    if hasattr(network, 'generators') and not network.generators.empty:
        generator_carriers = set(network.generators.carrier.unique())
        undefined_carriers = generator_carriers - set(network.carriers.index)
        
        if undefined_carriers:
            logger.warning(f"⚠️  {len(undefined_carriers)} generator carriers still undefined:")
            logger.warning(f"  {sorted(undefined_carriers)}")
            logger.warning("  These generators may not work correctly in optimization!")
        else:
            logger.info(f"✅ All {len(generator_carriers)} generator carriers properly defined")
    
    return network


def export_carrier_table(output_path="resources/generators/carrier_definitions.csv"):
    """
    Export carrier definitions to CSV for reference.
    
    Parameters
    ----------
    output_path : str or Path
        Path to save carrier definitions CSV
    """
    carriers_df = get_carrier_definitions()
    carriers_df.to_csv(output_path)
    print(f"Carrier definitions exported to: {output_path}")
    return carriers_df


if __name__ == "__main__":
    # Export carrier definitions when run as script
    import sys
    from pathlib import Path
    
    # Ensure output directory exists
    output_dir = Path("resources/generators")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export carriers
    carriers = export_carrier_table()
    
    print("\nCarrier Definitions Summary:")
    print("=" * 80)
    print(carriers.to_string())
    print("\n" + "=" * 80)
    print(f"Total carriers defined: {len(carriers)}")
    
    # Group by technology type
    print("\nBy Category:")
    print("  Wind: wind_onshore, wind_offshore")
    print("  Solar: solar_pv")
    print("  Hydro: small_hydro, large_hydro")
    print("  Marine: tidal_stream, shoreline_wave, tidal_lagoon")
    print("  Biomass: biomass, advanced_biofuel")
    print("  Gas-based Biomass: biogas, landfill_gas, sewage_gas")
    print("  Waste: waste_to_energy")
    print("  Geothermal: geothermal")
    print("  Fossil: gas_reciprocating")
    print("  Nuclear: nuclear")
    print("  Other: thermal_other, unclassified")
    print("  Network: AC, DC")

