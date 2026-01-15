"""
Network Clustering Rules for PyPSA-GB

Reduces network complexity by clustering buses into regions.

Pipeline Stages:
  1. cluster_network - Cluster complete network using configured method
  2. validate_network_clustered - Validate clustered network properties

Inputs:
  - Complete network with all components (.nc)
  - Boundaries file (for spatial clustering)
  - Busmap file (for explicit clustering)

Outputs:
  - Clustered network (.nc)
  - Bus mapping CSV

See Also:
  - network_build.smk - Builds base networks before clustering
  - config/clustering.yaml - Clustering method presets
"""

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _get_clustering_config(wildcards):
    """Get method-specific clustering configuration from scenario.
    
    The config_loader.py resolves presets (e.g., 'gsp_spatial') to full dicts,
    so we can access properties like 'method', 'boundaries_path', etc. directly.
    """
    scenario_config = scenarios.get(wildcards.scenario, {})
    clustering = scenario_config.get('clustering', {})
    
    if not isinstance(clustering, dict):
        return {}
    
    # Return all config except 'enabled' and 'method' which are handled separately
    return {k: v for k, v in clustering.items() 
            if k not in ['enabled', 'method']}


def _get_aggregation_strategies(wildcards):
    """Get aggregation strategies for clustering."""
    scenario_config = scenarios.get(wildcards.scenario, {})
    clustering = scenario_config.get('clustering', {})
    
    if not isinstance(clustering, dict):
        return None
    return clustering.get('strategies', None)


def _get_clustering_method(wildcards):
    """Get clustering method from scenario config.
    
    The config_loader.py resolves presets to full dicts with 'method' key.
    """
    scenario_config = scenarios.get(wildcards.scenario, {})
    clustering = scenario_config.get('clustering', {})
    
    if not isinstance(clustering, dict):
        return 'spatial'  # Default method
    return clustering.get('method', 'spatial')


def _get_clustering_boundaries(wildcards):
    """Get boundaries file path for spatial clustering.
    
    The config_loader.py resolves presets to full dicts with 'boundaries_path' key.
    """
    scenario_config = scenarios.get(wildcards.scenario, {})
    clustering = scenario_config.get('clustering', {})
    
    if not isinstance(clustering, dict):
        return []
    
    method = clustering.get('method', '')
    if method != 'spatial':
        return []
    
    # boundaries_path is now at top level (resolved by config_loader)
    boundaries = clustering.get('boundaries_path', '')
    
    if not boundaries:
        return []
    
    return boundaries


def _get_clustering_busmap(wildcards):
    """Get busmap file path for explicit clustering.
    
    The config_loader.py resolves presets to full dicts with 'busmap_source' key.
    """
    scenario_config = scenarios.get(wildcards.scenario, {})
    clustering = scenario_config.get('clustering', {})
    
    if not isinstance(clustering, dict):
        return []
    
    method = clustering.get('method', '')
    if method != 'busmap':
        return []
    
    # busmap_source is now at top level (resolved by config_loader)
    busmap = clustering.get('busmap_source', '')
    
    if not busmap:
        return []
    
    return busmap
    return busmap


# ══════════════════════════════════════════════════════════════════════════════
# RULES
# ══════════════════════════════════════════════════════════════════════════════

rule cluster_network:
    """
    Cluster complete network using the configured method.
    
    Supports multiple clustering methods:
    - spatial: Cluster to geographic boundaries (e.g., GSP regions)
    - kmeans: K-means clustering based on coordinates
    - busmap: Explicit bus-to-cluster mapping from CSV
    
    Transforms: {scenario}_network_..._interconnectors.nc → {scenario}_network_clustered_...nc
    
    Performance: ~30-120s depending on network size and method
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc",
        boundaries=_get_clustering_boundaries,
        busmap=_get_clustering_busmap
    output:
        clustered_network=f"{resources_path}/network/{{scenario}}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc",
        busmap_csv=f"{resources_path}/network/{{scenario}}_clustering_busmap.csv"
    params:
        scenario=lambda wc: wc.scenario,
        clustering_method=_get_clustering_method,
        method_config=_get_clustering_config,
        aggregation_strategies=_get_aggregation_strategies
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    log:
        "logs/network_clustering/cluster_network_{scenario}.log"
    benchmark:
        "benchmarks/network_clustering/cluster_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/network_clustering/cluster_network.py"


rule validate_network_clustered:
    """
    Validate clustered network connectivity and properties.
    
    Checks:
    - All buses connected (no isolated nodes)
    - Generator/load balance preserved
    - Line capacities aggregated correctly
    
    Transforms: clustered network → validation report HTML
    
    Performance: ~10-20s
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    output:
        validation_report=f"{resources_path}/validation/{{scenario}}_clustered_network_validation_report.html"
    params:
        scenario=lambda wc: wc.scenario
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    log:
        "logs/network_clustering/validate_{scenario}_clustered.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/network_clustering/validate_network.py"

