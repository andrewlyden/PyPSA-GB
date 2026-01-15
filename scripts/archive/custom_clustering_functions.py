"""
Custom Clustering Functions for PyPSA-GB

This module provides example custom clustering functions that can be used
with the network clustering framework. Users can create their own functions
following these patterns.

Each custom clustering function should:
1. Accept a PyPSA Network object as the first parameter
2. Accept a configuration dictionary as the second parameter  
3. Return a pandas Series mapping buses to cluster names
"""

import pandas as pd
import numpy as np
import pypsa
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def cluster_by_voltage_level(network: pypsa.Network, config: Dict) -> pd.Series:
    """
    Cluster buses based on voltage levels.
    
    Args:
        network: PyPSA Network to cluster
        config: Configuration dictionary with:
            - voltage_thresholds: List of voltage thresholds for clustering
            - cluster_names: List of cluster names for each voltage range
            
    Returns:
        pandas.Series: Bus-to-cluster mapping
    """
    voltage_thresholds = config.get('voltage_thresholds', [132, 275, 400])
    cluster_names = config.get('cluster_names', ['LV', 'MV', 'HV', 'EHV'])
    
    if len(cluster_names) != len(voltage_thresholds) + 1:
        raise ValueError("cluster_names should have one more element than voltage_thresholds")
    
    logger.info(f"Clustering by voltage levels: {voltage_thresholds} kV")
    
    # Get bus voltage levels
    bus_voltages = network.buses['v_nom']
    
    # Create cluster mapping based on voltage thresholds
    busmap = pd.Series(index=bus_voltages.index, dtype='object')
    
    for i, bus in enumerate(bus_voltages.index):
        voltage = bus_voltages[bus]
        
        # Find appropriate cluster
        cluster_idx = 0
        for threshold in voltage_thresholds:
            if voltage <= threshold:
                break
            cluster_idx += 1
        
        busmap[bus] = cluster_names[cluster_idx]
    
    logger.info(f"Voltage clustering result: {busmap.value_counts().to_dict()}")
    return busmap


def cluster_by_geographical_regions(network: pypsa.Network, config: Dict) -> pd.Series:
    """
    Cluster buses based on geographical regions using coordinate ranges.
    
    Args:
        network: PyPSA Network to cluster
        config: Configuration dictionary with:
            - regions: Dict mapping region names to coordinate bounds
                      {'region_name': {'lon_min': x1, 'lon_max': x2, 'lat_min': y1, 'lat_max': y2}}
            - default_region: Name for buses not in any defined region
            
    Returns:
        pandas.Series: Bus-to-cluster mapping
    """
    regions = config.get('regions', {})
    default_region = config.get('default_region', 'Other')
    
    if not regions:
        raise ValueError("No regions defined in configuration")
    
    logger.info(f"Clustering by geographical regions: {list(regions.keys())}")
    
    # Initialize busmap with default region
    busmap = pd.Series(default_region, index=network.buses.index)
    
    # Assign buses to regions based on coordinates
    for region_name, bounds in regions.items():
        lon_min = bounds.get('lon_min', -180)
        lon_max = bounds.get('lon_max', 180)
        lat_min = bounds.get('lat_min', -90)
        lat_max = bounds.get('lat_max', 90)
        
        # Find buses within this region
        mask = (
            (network.buses['x'] >= lon_min) & 
            (network.buses['x'] <= lon_max) &
            (network.buses['y'] >= lat_min) & 
            (network.buses['y'] <= lat_max)
        )
        
        busmap.loc[mask] = region_name
    
    logger.info(f"Geographical clustering result: {busmap.value_counts().to_dict()}")
    return busmap


def cluster_by_generation_capacity(network: pypsa.Network, config: Dict) -> pd.Series:
    """
    Cluster buses based on total generation capacity connected to them.
    
    Args:
        network: PyPSA Network to cluster
        config: Configuration dictionary with:
            - capacity_thresholds: List of capacity thresholds (MW) for clustering
            - cluster_names: List of cluster names for each capacity range
            - include_carriers: List of generator carriers to include (optional)
            
    Returns:
        pandas.Series: Bus-to-cluster mapping
    """
    capacity_thresholds = config.get('capacity_thresholds', [100, 500, 1000])
    cluster_names = config.get('cluster_names', ['Small', 'Medium', 'Large', 'Very_Large'])
    include_carriers = config.get('include_carriers', None)
    
    if len(cluster_names) != len(capacity_thresholds) + 1:
        raise ValueError("cluster_names should have one more element than capacity_thresholds")
    
    logger.info(f"Clustering by generation capacity: {capacity_thresholds} MW")
    
    # Calculate total generation capacity per bus
    generators = network.generators.copy()
    if include_carriers:
        generators = generators[generators['carrier'].isin(include_carriers)]
    
    bus_capacity = generators.groupby('bus')['p_nom'].sum()
    bus_capacity = bus_capacity.reindex(network.buses.index, fill_value=0)
    
    # Create cluster mapping based on capacity thresholds
    busmap = pd.Series(index=network.buses.index, dtype='object')
    
    for bus in network.buses.index:
        capacity = bus_capacity[bus]
        
        # Find appropriate cluster
        cluster_idx = 0
        for threshold in capacity_thresholds:
            if capacity <= threshold:
                break
            cluster_idx += 1
        
        busmap[bus] = cluster_names[cluster_idx]
    
    logger.info(f"Generation capacity clustering result: {busmap.value_counts().to_dict()}")
    return busmap


def cluster_by_load_density(network: pypsa.Network, config: Dict) -> pd.Series:
    """
    Cluster buses based on load density (load per area).
    
    Args:
        network: PyPSA Network to cluster
        config: Configuration dictionary with:
            - density_thresholds: List of density thresholds for clustering
            - cluster_names: List of cluster names for each density range
            - area_method: Method to estimate area ('voronoi', 'radius', 'uniform')
            
    Returns:
        pandas.Series: Bus-to-cluster mapping
    """
    density_thresholds = config.get('density_thresholds', [10, 50, 100])  # MW/km²
    cluster_names = config.get('cluster_names', ['Rural', 'Suburban', 'Urban', 'Metropolitan'])
    area_method = config.get('area_method', 'uniform')
    
    if len(cluster_names) != len(density_thresholds) + 1:
        raise ValueError("cluster_names should have one more element than density_thresholds")
    
    logger.info(f"Clustering by load density: {density_thresholds} MW/km²")
    
    # Calculate load per bus
    bus_load = network.loads.groupby('bus')['p_nom'].sum()
    bus_load = bus_load.reindex(network.buses.index, fill_value=0)
    
    # Estimate area per bus (simplified approach)
    if area_method == 'uniform':
        # Assume uniform area distribution
        total_buses = len(network.buses)
        area_per_bus = 1000  # Simplified: 1000 km² per bus on average
        bus_area = pd.Series(area_per_bus, index=network.buses.index)
    else:
        # For now, fall back to uniform
        logger.warning(f"Area method '{area_method}' not implemented, using uniform")
        area_per_bus = 1000
        bus_area = pd.Series(area_per_bus, index=network.buses.index)
    
    # Calculate density
    bus_density = bus_load / bus_area
    
    # Create cluster mapping based on density thresholds
    busmap = pd.Series(index=network.buses.index, dtype='object')
    
    for bus in network.buses.index:
        density = bus_density[bus]
        
        # Find appropriate cluster
        cluster_idx = 0
        for threshold in density_thresholds:
            if density <= threshold:
                break
            cluster_idx += 1
        
        busmap[bus] = cluster_names[cluster_idx]
    
    logger.info(f"Load density clustering result: {busmap.value_counts().to_dict()}")
    return busmap


def cluster_by_network_topology(network: pypsa.Network, config: Dict) -> pd.Series:
    """
    Cluster buses based on network topology characteristics.
    
    Args:
        network: PyPSA Network to cluster
        config: Configuration dictionary with:
            - connectivity_thresholds: List of connectivity thresholds for clustering
            - cluster_names: List of cluster names for each connectivity range
            
    Returns:
        pandas.Series: Bus-to-cluster mapping
    """
    connectivity_thresholds = config.get('connectivity_thresholds', [2, 4, 6])
    cluster_names = config.get('cluster_names', ['Isolated', 'Low_Connect', 'Med_Connect', 'High_Connect'])
    
    if len(cluster_names) != len(connectivity_thresholds) + 1:
        raise ValueError("cluster_names should have one more element than connectivity_thresholds")
    
    logger.info(f"Clustering by network connectivity: {connectivity_thresholds}")
    
    # Calculate connectivity (number of lines connected to each bus)
    bus_connectivity = pd.Series(0, index=network.buses.index)
    
    # Count lines connected to each bus
    for _, line in network.lines.iterrows():
        bus_connectivity[line['bus0']] += 1
        bus_connectivity[line['bus1']] += 1
    
    # Count transformers connected to each bus
    for _, transformer in network.transformers.iterrows():
        bus_connectivity[transformer['bus0']] += 1
        bus_connectivity[transformer['bus1']] += 1
    
    # Create cluster mapping based on connectivity thresholds
    busmap = pd.Series(index=network.buses.index, dtype='object')
    
    for bus in network.buses.index:
        connectivity = bus_connectivity[bus]
        
        # Find appropriate cluster
        cluster_idx = 0
        for threshold in connectivity_thresholds:
            if connectivity <= threshold:
                break
            cluster_idx += 1
        
        busmap[bus] = cluster_names[cluster_idx]
    
    logger.info(f"Topology clustering result: {busmap.value_counts().to_dict()}")
    return busmap


def cluster_mixed_criteria(network: pypsa.Network, config: Dict) -> pd.Series:
    """
    Example of clustering using multiple criteria combined.
    
    Args:
        network: PyPSA Network to cluster
        config: Configuration dictionary with criteria weights and thresholds
        
    Returns:
        pandas.Series: Bus-to-cluster mapping
    """
    logger.info("Performing mixed-criteria clustering")
    
    # This is a more complex example that could combine multiple factors
    # For demonstration, we'll use a simple combination of voltage and geography
    
    voltage_weight = config.get('voltage_weight', 0.5)
    geography_weight = config.get('geography_weight', 0.5)
    n_clusters = config.get('n_clusters', 8)
    
    # Normalize voltage levels
    voltages = network.buses['v_nom']
    voltage_norm = (voltages - voltages.min()) / (voltages.max() - voltages.min())
    
    # Normalize geographical coordinates
    x_norm = (network.buses['x'] - network.buses['x'].min()) / (network.buses['x'].max() - network.buses['x'].min())
    y_norm = (network.buses['y'] - network.buses['y'].min()) / (network.buses['y'].max() - network.buses['y'].min())
    geo_norm = (x_norm + y_norm) / 2
    
    # Combine criteria
    combined_score = voltage_weight * voltage_norm + geography_weight * geo_norm
    
    # Use quantile-based clustering
    quantiles = np.linspace(0, 1, n_clusters + 1)
    thresholds = combined_score.quantile(quantiles[1:-1])
    
    busmap = pd.Series(index=network.buses.index, dtype='object')
    
    for bus in network.buses.index:
        score = combined_score[bus]
        
        # Find appropriate cluster
        cluster_idx = 0
        for threshold in thresholds:
            if score <= threshold:
                break
            cluster_idx += 1
        
        busmap[bus] = f"mixed_cluster_{cluster_idx}"
    
    logger.info(f"Mixed criteria clustering result: {busmap.value_counts().to_dict()}")
    return busmap

