"""
Hydrogen System Integration Rules for PyPSA-GB

This module integrates a simplified hydrogen sector coupling:
- Electrolysis (power-to-hydrogen)
- Hydrogen storage (underground caverns / tanks)
- H2 power generation (H2-ready turbines)

Architecture:
=============
The current implementation uses a "copper-plate" hydrogen network:
- Single GB-wide hydrogen bus (no spatial resolution)
- All electrolysers feed into common H2 storage
- All H2 turbines draw from common H2 storage

This is a simplification - future work will model regional H2 networks.

Data Sources:
=============
- FES: Electrolysis capacity projections by GSP
- FES: Hydrogen generation capacity (already integrated as generators)

Technical Parameters:
====================
- Electrolysis efficiency: 70% (electrical → H2 LHV)
- H2 turbine efficiency: 50% (H2 → electrical)
- Round-trip efficiency: 35% (electricity → H2 → electricity)
- Storage: Sized for seasonal flexibility (168 hours of generation)

Usage:
======
# Run hydrogen integration for a scenario
snakemake resources/network/{scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl --cores 1

# Run full workflow including hydrogen
snakemake --cores 4
"""

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

resources_path = "resources"
data_path = "data"


# =============================================================================
# HYDROGEN SYSTEM INTEGRATION
# =============================================================================

rule add_hydrogen_system:
    """
    Add hydrogen system components to PyPSA network.
    
    This rule transforms the network by:
    1. Creating a single GB-wide hydrogen bus
    2. Adding electrolysis Links (electricity → H2)
    3. Adding hydrogen storage Store
    4. Converting H2 generators to Links (H2 → electricity)
    
    The result is a proper power-to-gas-to-power (P2G2P) system where:
    - Electrolysis consumes electricity to produce hydrogen
    - Hydrogen is stored (weekly/seasonal buffer)
    - H2 turbines consume hydrogen to produce electricity
    
    This creates the correct energy balance constraint - H2 turbines can
    only generate if there's sufficient hydrogen from electrolysis.
    
    For historical scenarios (≤2024), this rule is a no-op as there was
    minimal hydrogen infrastructure.
    
    Parameters:
    -----------
    electrolysis_efficiency: 0.70 (70% electrical → H2)
    h2_turbine_efficiency: 0.50 (50% H2 → electrical)
    storage_hours: 168 (1 week of H2 generation capacity)
    
    Performance:
    ------------
    ~5-10 seconds (network modification, no heavy computation)
    
    Transforms:
    -----------
    {scenario}_network_demand_renewables_thermal_generators_storage.pkl
    → {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage.pkl",
        fes_data=f"{resources_path}/FES/FES_2024_data.csv"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl"
    params:
        scenario=lambda wc: wc.scenario,
        modelled_year=lambda wc: scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year'),
        is_historical=lambda wc: (scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year', 9999)) <= 2024,
        fes_scenario=lambda wc: scenarios[wc.scenario].get('FES_scenario', None)
    message:
        "Adding hydrogen system to network for {wildcards.scenario}"
    log:
        "logs/storage/add_hydrogen_system_{scenario}.log"
    benchmark:
        "benchmarks/storage/add_hydrogen_system_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/hydrogen/add_hydrogen_system.py"
