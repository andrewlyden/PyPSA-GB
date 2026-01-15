"""
General Network Clustering Framework for PyPSA-GB

This script provides a flexible framework for clustering network data based on 
user-defined bus maps. It supports various clustering methods and integrates 
seamlessly with the PyPSA-GB workflow.

Supported clustering methods:
- spatial: Spatial clustering based on GeoJSON boundaries (like GSP regions)
- busmap: Direct bus-to-cluster mapping from CSV/dictionary
- custom: Custom clustering using user-defined functions
- kmeans: K-means clustering based on geographical coordinates
- hierarchical: Hierarchical clustering based on electrical distances
"""

import logging
import pandas as pd
import geopandas as gpd
import pypsa
from pypsa.clustering.spatial import get_clustering_from_busmap
import numpy as np
from pathlib import Path
import yaml
import json
from typing import Dict, List, Optional, Union, Callable
import warnings
from time import time

# Suppress PyPSA warnings about unoptimized networks (expected during network building)
warnings.filterwarnings('ignore', message='The network has not been optimized yet')

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network

# Try importing logging_config from same directory
try:
    from scripts.utilities.logging_config import setup_logging, log_network_info, log_execution_summary
except ImportError:
    # Fallback for when script is run standalone
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from logging_config import setup_logging, log_network_info, log_execution_summary

# Configure logging
logger = setup_logging("cluster_network", log_level="INFO")

class NetworkClusterer:
    """
    A comprehensive network clustering framework for PyPSA networks.
    
    This class handles various clustering approaches and integrates with 
    the PyPSA-GB scenario configuration system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the NetworkClusterer.
        
        Args:
            config_path: Path to clustering configuration file
        """
        self.clustering_config = {}
        if config_path and Path(config_path).exists():
            self.load_clustering_config(config_path)
            
        # Default clustering strategies for different network components
        # CRITICAL: Lines aggregated between same bus pairs should be treated as PARALLEL
        # This means: s_nom sums, impedances (r,x) combine in parallel (1/r_total = sum(1/r_i))
        self.default_strategies = {
            'bus': {
                'x': 'mean', 
                'y': 'mean',
                'lon': 'mean',  # WGS84 longitude - use mean for cluster centroid
                'lat': 'mean',  # WGS84 latitude - use mean for cluster centroid
                'x_osgb36': 'mean',  # Original OSGB36 x-coordinate (preserved during WGS84 conversion)
                'y_osgb36': 'mean',  # Original OSGB36 y-coordinate (preserved during WGS84 conversion)
            },
            'line': {
                'v_nom': 'max', 
                's_nom': 'sum',  # Parallel lines add capacity
                'length': 'mean',  # Average length is more reasonable than sum
                'length_km': 'mean',  # Some networks use length_km instead of length
                'r': self._parallel_impedance,  # Parallel aggregation: 1/R_total = sum(1/R_i)
                'x': self._parallel_impedance,  # Parallel aggregation: 1/X_total = sum(1/X_i)
                'b': 'sum',  # Susceptances add in parallel
            },
            'transformer': {'v_nom': 'max', 's_nom': 'sum'},
            'load': {'p_nom': 'sum'},
            'generator': {'p_nom': 'sum', 'marginal_cost': 'mean'},
            'storage_unit': {'p_nom': 'sum', 'max_hours': 'mean'},
            'store': {'e_nom': 'sum'},
            'link': {'p_nom': 'sum', 'efficiency': 'mean'}
        }
    
    @staticmethod
    def _parallel_impedance(impedances):
        """
        Aggregate impedances in parallel (as for transmission lines).
        
        For resistances/reactances in parallel:
        1/R_total = sum(1/R_i)
        
        This prevents unrealistically high impedances after clustering.
        """
        # Filter out zeros and NaNs to avoid division errors
        valid_impedances = impedances[impedances > 0].dropna()
        
        if len(valid_impedances) == 0:
            return 0.0001  # Small default if all zeros
        elif len(valid_impedances) == 1:
            return valid_impedances.iloc[0]
        else:
            # Parallel: 1/R_total = sum(1/R_i) → R_total = 1/sum(1/R_i)
            return 1.0 / (1.0 / valid_impedances).sum()
    
    @staticmethod
    def _capacity_weighted_impedance(df):
        """
        Aggregate impedances weighted by line capacity (s_nom).
        
        This gives more weight to high-capacity lines, which is more realistic
        for power flow. Formula: R_eff = sum(R_i * s_i) / sum(s_i)
        
        Args:
            df: DataFrame with columns 'r' (or 'x') and 's_nom'
            
        Returns:
            Weighted average impedance
        """
        # Get impedance column name (r or x)
        imp_col = 'r' if 'r' in df.columns else 'x' if 'x' in df.columns else None
        if imp_col is None:
            return 0.0001
            
        # Filter valid values
        valid = (df[imp_col] > 0) & (df['s_nom'] > 0) & df[imp_col].notna() & df['s_nom'].notna()
        df_valid = df[valid]
        
        if len(df_valid) == 0:
            return 0.0001
        elif len(df_valid) == 1:
            return df_valid[imp_col].iloc[0]
        else:
            # Capacity-weighted average impedance
            total_capacity = df_valid['s_nom'].sum()
            if total_capacity > 0:
                return (df_valid[imp_col] * df_valid['s_nom']).sum() / total_capacity
            else:
                return df_valid[imp_col].mean()
    
    def _resolve_strategy_functions(self, strategies: Dict) -> Dict:
        """
        Convert string references in strategies to actual function references.
        
        Allows YAML config to specify custom aggregation functions by name.
        E.g., "parallel_impedance" → self._parallel_impedance
        """
        resolved = {}
        
        # Mapping of string names to actual functions
        function_map = {
            'parallel_impedance': self._parallel_impedance,
            'capacity_weighted_impedance': self._capacity_weighted_impedance,
        }
        
        for component, component_strategies in strategies.items():
            if isinstance(component_strategies, dict):
                resolved[component] = {}
                for param, strategy in component_strategies.items():
                    # If strategy is a string matching a known function, replace it
                    if isinstance(strategy, str) and strategy in function_map:
                        resolved[component][param] = function_map[strategy]
                        logger.debug(f"Resolved '{strategy}' to function for {component}.{param}")
                    else:
                        resolved[component][param] = strategy
            else:
                resolved[component] = component_strategies
        
        return resolved
    
    def load_clustering_config(self, config_path: str) -> None:
        """Load clustering configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                self.clustering_config = yaml.safe_load(f)
            logger.info(f"Loaded clustering configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load clustering config from {config_path}: {e}")
    
    def spatial_clustering(self, network: pypsa.Network, 
                          boundaries_path: str, 
                          method_config: Dict) -> pd.Series:
        """
        Perform spatial clustering based on geographical boundaries.
        
        Args:
            network: PyPSA Network to cluster
            boundaries_path: Path to GeoJSON file with cluster boundaries
            method_config: Configuration for spatial clustering
            
        Returns:
            pandas.Series: Bus-to-cluster mapping (busmap)
        """
        logger.info(f"Performing spatial clustering using boundaries from {boundaries_path}")
        
        # Load boundary data
        try:
            boundaries = gpd.read_file(boundaries_path)
        except Exception as e:
            raise ValueError(f"Could not load boundaries from {boundaries_path}: {e}")
        
        # Get configuration parameters
        cluster_col = method_config.get('cluster_column', boundaries.columns[0])
        boundary_crs = method_config.get('boundary_crs', 'EPSG:27700')
        fallback_method = method_config.get('fallback', 'nearest_centroid')
        
        # Auto-detect network coordinate system based on coordinate range
        # OSGB36 (British National Grid) has x values in the range 0-700,000 meters
        # WGS84 has longitude values in the range -180 to 180 degrees
        x_range = network.buses['x'].max() - network.buses['x'].min()
        if x_range > 1000:
            # Coordinates are in meters (OSGB36)
            detected_bus_crs = 'EPSG:27700'
            logger.info(f"Detected OSGB36 coordinates in network buses (x range: {x_range:.0f}m)")
        else:
            # Coordinates are in degrees (WGS84)
            detected_bus_crs = 'EPSG:4326'
            logger.info(f"Detected WGS84 coordinates in network buses (x range: {x_range:.4f}°)")
        
        # Use detected CRS unless overridden by config
        bus_crs = method_config.get('bus_crs', detected_bus_crs)
        if bus_crs != detected_bus_crs:
            logger.warning(f"Config specifies bus_crs={bus_crs} but detected {detected_bus_crs} - using config value")
        
        # Ensure boundaries have correct CRS
        if boundaries.crs is None:
            logger.warning(f"Boundaries have no CRS, assuming {boundary_crs}")
            boundaries = boundaries.set_crs(boundary_crs)
        elif boundaries.crs.to_string() != boundary_crs:
            logger.info(f"Reprojecting boundaries from {boundaries.crs} to {boundary_crs}")
            boundaries = boundaries.to_crs(boundary_crs)
        
        # Create GeoDataFrame of network buses with detected CRS
        bus_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(network.buses.x, network.buses.y),
            index=network.buses.index,
            crs=bus_crs
        )
        # Reproject to boundary CRS for spatial join if needed
        if bus_crs != boundary_crs:
            logger.info(f"Reprojecting bus coordinates from {bus_crs} to {boundary_crs} for spatial join")
            bus_gdf = bus_gdf.to_crs(boundary_crs)
        bus_gdf['bus_id'] = bus_gdf.index
        
        # Separate external buses (for interconnectors) from GB buses
        # External buses should not be clustered with GB buses
        # CRITICAL: Buses with empty/missing country should be treated as GB buses, not external
        # CRITICAL: Non-AC buses (like H2 buses) should also be kept separate from clustering
        
        # First, identify non-AC buses (hydrogen, DC, etc.) - these should not be clustered with AC buses
        if 'carrier' in network.buses.columns:
            is_non_ac = (network.buses['carrier'] != 'AC') & (network.buses['carrier'].notna())
            non_ac_buses = network.buses[is_non_ac].index
            if len(non_ac_buses) > 0:
                logger.info(f"Found {len(non_ac_buses)} non-AC buses to keep separate from clustering:")
                for carrier in network.buses.loc[non_ac_buses, 'carrier'].unique():
                    count = (network.buses.loc[non_ac_buses, 'carrier'] == carrier).sum()
                    logger.info(f"  - {carrier}: {count} buses")
        else:
            non_ac_buses = pd.Index([])
        
        if 'country' in network.buses.columns:
            # Only treat buses with explicit non-GB country values (like 'France', 'Norway') as external
            # Empty strings, NaN, and 'GB' are all treated as GB buses
            is_external = (
                (network.buses['country'].notna()) & 
                (network.buses['country'] != '') & 
                (network.buses['country'] != 'GB')
            )
            external_buses = network.buses[is_external].index
            # GB AC buses = not external AND AC carrier (or no carrier info)
            gb_buses = network.buses[~is_external & ~network.buses.index.isin(non_ac_buses)].index
            logger.info(f"Found {len(gb_buses)} GB AC buses for clustering")
            logger.info(f"Found {len(gb_buses)} GB buses (including {(~is_external & ((network.buses['country'] == '') | network.buses['country'].isna())).sum()} with empty country)")
            logger.info(f"Found {len(external_buses)} external buses to handle separately")
            bus_gdf_gb = bus_gdf.loc[gb_buses]
        else:
            gb_buses = network.buses.index
            external_buses = []
            bus_gdf_gb = bus_gdf
        
        # Perform spatial join only for GB buses
        logger.info(f"Performing spatial join with {len(boundaries)} boundary regions")
        joined = gpd.sjoin(
            boundaries[[cluster_col, 'geometry']], 
            bus_gdf_gb, 
            how="right", 
            predicate="contains"
        )
        
        # Create busmap for GB buses only
        busmap = joined.reset_index().set_index("bus_id")[cluster_col].reindex(gb_buses)
        unassigned = busmap[busmap.isna()]
        
        logger.info(f"Spatial join assigned {len(busmap) - len(unassigned)} buses to clusters")
        
        if not unassigned.empty:
            logger.info(f"Handling {len(unassigned)} unassigned buses using {fallback_method}")
            busmap = self._handle_unassigned_buses(
                busmap, unassigned, bus_gdf, boundaries, 
                cluster_col, fallback_method
            )
        
        # Assign external buses to their own individual clusters
        # Each external bus gets its own cluster to prevent mixing GB and External countries
        if len(external_buses) > 0:
            logger.info(f"Assigning {len(external_buses)} external buses to individual clusters")
            for ext_bus in external_buses:
                # Use the bus name itself as the cluster ID for external buses
                busmap.loc[ext_bus] = f"External_{ext_bus}"
        
        # Assign non-AC buses (hydrogen, DC, etc.) to their own individual clusters
        # Each non-AC bus gets its own cluster to prevent carrier mismatch during aggregation
        if len(non_ac_buses) > 0:
            logger.info(f"Assigning {len(non_ac_buses)} non-AC buses to individual clusters")
            for non_ac_bus in non_ac_buses:
                carrier = network.buses.loc[non_ac_bus, 'carrier']
                # Use carrier type and bus name as the cluster ID
                busmap.loc[non_ac_bus] = f"{carrier}_{non_ac_bus}"
        
        # Validate busmap
        if busmap.isna().any():
            raise ValueError(f"Failed to assign {busmap.isna().sum()} buses to clusters")
        
        n_gb_clusters = busmap.nunique() - len(external_buses) - len(non_ac_buses)
        logger.info(f"Successfully created busmap with {busmap.nunique()} clusters ({n_gb_clusters} GB AC clusters + {len(external_buses)} external + {len(non_ac_buses)} non-AC)")
        return busmap
    
    def busmap_clustering(self, network: pypsa.Network, 
                         busmap_source: Union[str, Dict, pd.Series],
                         method_config: Dict) -> pd.Series:
        """
        Perform clustering based on explicit bus-to-cluster mapping.
        
        Args:
            network: PyPSA Network to cluster
            busmap_source: Source of bus mapping (file path, dict, or Series)
            method_config: Configuration for busmap clustering
            
        Returns:
            pandas.Series: Bus-to-cluster mapping (busmap)
        """
        logger.info("Performing explicit busmap clustering")
        
        if isinstance(busmap_source, str):
            # Load from file
            if busmap_source.endswith('.csv'):
                busmap_df = pd.read_csv(busmap_source, index_col=0)
                busmap = busmap_df.iloc[:, 0]  # Take first column
            elif busmap_source.endswith('.json'):
                with open(busmap_source, 'r') as f:
                    busmap_dict = json.load(f)
                busmap = pd.Series(busmap_dict)
            else:
                raise ValueError(f"Unsupported busmap file format: {busmap_source}")
        elif isinstance(busmap_source, dict):
            busmap = pd.Series(busmap_source)
        elif isinstance(busmap_source, pd.Series):
            busmap = busmap_source.copy()
        else:
            raise ValueError(f"Unsupported busmap_source type: {type(busmap_source)}")
        
        # Remove duplicates by keeping the first occurrence
        busmap = busmap[~busmap.index.duplicated(keep='first')]
        
        # Reindex to match network buses and handle missing entries
        busmap = busmap.reindex(network.buses.index)
        missing_buses = busmap[busmap.isna()].index
        
        if not missing_buses.empty:
            default_cluster = method_config.get('default_cluster', 'unassigned')
            logger.warning(f"Assigning {len(missing_buses)} missing buses to '{default_cluster}'")
            busmap.loc[missing_buses] = default_cluster
        
        logger.info(f"Successfully created busmap with {busmap.nunique()} clusters")
        return busmap
    
    def kmeans_clustering(self, network: pypsa.Network, 
                         method_config: Dict) -> pd.Series:
        """
        Perform K-means clustering based on geographical coordinates.
        
        Args:
            network: PyPSA Network to cluster
            method_config: Configuration for K-means clustering
            
        Returns:
            pandas.Series: Bus-to-cluster mapping (busmap)
            
        Note:
            Non-AC buses (like hydrogen, DC) and external buses (non-GB) are excluded 
            from clustering and assigned to their own individual clusters to prevent 
            carrier/country mismatch during aggregation.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn is required for K-means clustering")
        
        n_clusters = method_config.get('n_clusters', 10)
        random_state = method_config.get('random_state', 42)
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters")
        
        # Initialize busmap with bus indices
        busmap = pd.Series(index=network.buses.index, dtype=str)
        
        # Build mask for buses that should NOT be clustered
        exclude_mask = pd.Series(False, index=network.buses.index)
        
        # Exclude non-AC buses (hydrogen, DC, etc.)
        if 'carrier' in network.buses.columns:
            is_non_ac = (network.buses['carrier'] != 'AC') & (network.buses['carrier'].notna())
            exclude_mask |= is_non_ac
            non_ac_count = is_non_ac.sum()
            if non_ac_count > 0:
                logger.info(f"Excluding {non_ac_count} non-AC buses from K-means clustering:")
                for carrier in network.buses.loc[is_non_ac, 'carrier'].unique():
                    count = (network.buses.loc[is_non_ac, 'carrier'] == carrier).sum()
                    logger.info(f"  - {carrier}: {count} buses")
        
        # Exclude external buses (non-GB countries)
        if 'country' in network.buses.columns:
            is_external = (network.buses['country'] != 'GB') & (network.buses['country'] != '') & (network.buses['country'].notna())
            exclude_mask |= is_external
            external_count = is_external.sum()
            if external_count > 0:
                logger.info(f"Excluding {external_count} external buses from K-means clustering:")
                for country in network.buses.loc[is_external, 'country'].unique():
                    count = (network.buses.loc[is_external, 'country'] == country).sum()
                    logger.info(f"  - {country}: {count} buses")
        
        # Assign excluded buses to their own individual clusters
        excluded_buses = network.buses.index[exclude_mask]
        for excluded_bus in excluded_buses:
            # Use a unique identifier based on bus name
            country = network.buses.loc[excluded_bus, 'country'] if 'country' in network.buses.columns else 'unknown'
            carrier = network.buses.loc[excluded_bus, 'carrier'] if 'carrier' in network.buses.columns else 'AC'
            busmap.loc[excluded_bus] = f"external_{country}_{excluded_bus}"
        
        # Get buses to cluster (GB AC buses only)
        cluster_buses = network.buses.index[~exclude_mask]
        
        if len(cluster_buses) == 0:
            logger.warning("No buses found for K-means clustering")
            return busmap
        
        logger.info(f"Clustering {len(cluster_buses)} GB AC buses")
        
        # Prepare coordinates for GB AC buses only
        coords = network.buses.loc[cluster_buses, ['x', 'y']].values
        
        # Adjust n_clusters if we have fewer buses than requested clusters
        actual_n_clusters = min(n_clusters, len(cluster_buses))
        if actual_n_clusters < n_clusters:
            logger.warning(f"Reducing clusters from {n_clusters} to {actual_n_clusters} (only {len(cluster_buses)} GB AC buses)")
        
        # Perform clustering on GB AC buses only
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Create busmap for clustered buses
        for bus, label in zip(cluster_buses, cluster_labels):
            busmap.loc[bus] = f"cluster_{label}"
        
        logger.info(f"Successfully created K-means busmap with {busmap.nunique()} clusters")
        return busmap
    
    def hierarchical_clustering(self, network: pypsa.Network,
                               method_config: Dict) -> pd.Series:
        """
        Perform hierarchical clustering based on electrical distances.
        
        Args:
            network: PyPSA Network to cluster  
            method_config: Configuration for hierarchical clustering
            
        Returns:
            pandas.Series: Bus-to-cluster mapping (busmap)
            
        Note:
            Non-AC buses (like hydrogen, DC) and external buses (non-GB) are excluded 
            from clustering and assigned to their own individual clusters to prevent 
            carrier/country mismatch during aggregation.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import pairwise_distances
        except ImportError:
            raise ImportError("scikit-learn is required for hierarchical clustering")
        
        n_clusters = method_config.get('n_clusters', 10)
        linkage = method_config.get('linkage', 'ward')
        
        logger.info(f"Performing hierarchical clustering with {n_clusters} clusters")
        
        # Initialize busmap with bus indices
        busmap = pd.Series(index=network.buses.index, dtype=str)
        
        # Build mask for buses that should NOT be clustered
        exclude_mask = pd.Series(False, index=network.buses.index)
        
        # Exclude non-AC buses (hydrogen, DC, etc.)
        if 'carrier' in network.buses.columns:
            is_non_ac = (network.buses['carrier'] != 'AC') & (network.buses['carrier'].notna())
            exclude_mask |= is_non_ac
            non_ac_count = is_non_ac.sum()
            if non_ac_count > 0:
                logger.info(f"Excluding {non_ac_count} non-AC buses from hierarchical clustering:")
                for carrier in network.buses.loc[is_non_ac, 'carrier'].unique():
                    count = (network.buses.loc[is_non_ac, 'carrier'] == carrier).sum()
                    logger.info(f"  - {carrier}: {count} buses")
        
        # Exclude external buses (non-GB countries)
        if 'country' in network.buses.columns:
            is_external = (network.buses['country'] != 'GB') & (network.buses['country'] != '') & (network.buses['country'].notna())
            exclude_mask |= is_external
            external_count = is_external.sum()
            if external_count > 0:
                logger.info(f"Excluding {external_count} external buses from hierarchical clustering:")
                for country in network.buses.loc[is_external, 'country'].unique():
                    count = (network.buses.loc[is_external, 'country'] == country).sum()
                    logger.info(f"  - {country}: {count} buses")
        
        # Assign excluded buses to their own individual clusters
        excluded_buses = network.buses.index[exclude_mask]
        for excluded_bus in excluded_buses:
            country = network.buses.loc[excluded_bus, 'country'] if 'country' in network.buses.columns else 'unknown'
            busmap.loc[excluded_bus] = f"external_{country}_{excluded_bus}"
        
        # Get buses to cluster (GB AC buses only)
        cluster_buses = network.buses.index[~exclude_mask]
        
        if len(cluster_buses) == 0:
            logger.warning("No buses found for hierarchical clustering")
            return busmap
        
        logger.info(f"Clustering {len(cluster_buses)} GB AC buses")
        
        # Calculate electrical distances for GB AC buses only (simplified - using geographical distance as proxy)
        coords = network.buses.loc[cluster_buses, ['x', 'y']].values
        distances = pairwise_distances(coords, metric='euclidean')
        
        # Adjust n_clusters if we have fewer buses than requested clusters
        actual_n_clusters = min(n_clusters, len(cluster_buses))
        if actual_n_clusters < n_clusters:
            logger.warning(f"Reducing clusters from {n_clusters} to {actual_n_clusters} (only {len(cluster_buses)} GB AC buses)")
        
        # Perform clustering on GB AC buses only
        clustering = AgglomerativeClustering(
            n_clusters=actual_n_clusters, 
            linkage=linkage,
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distances)
        
        # Create busmap for clustered buses
        for bus, label in zip(cluster_buses, cluster_labels):
            busmap.loc[bus] = f"cluster_{label}"
        
        logger.info(f"Successfully created hierarchical busmap with {busmap.nunique()} clusters")
        return busmap
    
    def custom_clustering(self, network: pypsa.Network,
                         method_config: Dict) -> pd.Series:
        """
        Perform custom clustering using user-defined function.
        
        Args:
            network: PyPSA Network to cluster
            method_config: Configuration for custom clustering
            
        Returns:
            pandas.Series: Bus-to-cluster mapping (busmap)
        """
        function_path = method_config.get('function_path')
        function_name = method_config.get('function_name', 'cluster_buses')
        
        if not function_path:
            raise ValueError("Custom clustering requires 'function_path' in config")
        
        logger.info(f"Performing custom clustering using {function_name} from {function_path}")
        
        # Import custom function
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_clustering", function_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        
        custom_function = getattr(custom_module, function_name)
        
        # Apply custom function
        busmap = custom_function(network, method_config)
        
        if not isinstance(busmap, pd.Series):
            raise ValueError("Custom clustering function must return a pandas Series")
        
        logger.info(f"Successfully created custom busmap with {busmap.nunique()} clusters")
        return busmap
    
    def cluster_network(self, network: pypsa.Network, 
                       clustering_method: str,
                       method_config: Dict,
                       aggregation_strategies: Optional[Dict] = None) -> pypsa.Network:
        """
        Cluster a PyPSA network using the specified method.
        
        Args:
            network: PyPSA Network to cluster
            clustering_method: Method to use ('spatial', 'busmap', 'kmeans', etc.)
            method_config: Configuration for the clustering method
            aggregation_strategies: Custom aggregation strategies for network components
            
        Returns:
            pypsa.Network: Clustered network
        """
        logger.info(f"Starting network clustering using method: {clustering_method}")
        
        # Get busmap using appropriate method
        if clustering_method == 'spatial':
            boundaries_path = method_config.get('boundaries_path')
            if not boundaries_path:
                raise ValueError("Spatial clustering requires 'boundaries_path' in config")
            busmap = self.spatial_clustering(network, boundaries_path, method_config)
            
        elif clustering_method == 'busmap':
            busmap_source = method_config.get('busmap_source')
            if not busmap_source:
                raise ValueError("Busmap clustering requires 'busmap_source' in config")
            busmap = self.busmap_clustering(network, busmap_source, method_config)
            
        elif clustering_method == 'kmeans':
            busmap = self.kmeans_clustering(network, method_config)
            
        elif clustering_method == 'hierarchical':
            busmap = self.hierarchical_clustering(network, method_config)
            
        elif clustering_method == 'custom':
            busmap = self.custom_clustering(network, method_config)
            
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        # Store busmap as instance variable for later retrieval
        self.last_busmap = busmap
        
        # Prepare aggregation strategies
        strategies = self.default_strategies.copy()
        if aggregation_strategies:
            strategies.update(aggregation_strategies)
        
        # Convert string references to actual functions
        strategies = self._resolve_strategy_functions(strategies)
        
        # Handle transformers -> lines conversion if specified
        if method_config.get('convert_transformers_to_lines', False):
            network = self._convert_transformers_to_lines(network)
        
        # CRITICAL: PyPSA's clustering uses haversine distance for line lengths, which expects WGS84 coordinates
        # If network buses are in OSGB36 (meters), we need to convert to WGS84 BEFORE clustering
        x_range = network.buses['x'].max() - network.buses['x'].min()
        original_crs_is_osgb36 = x_range > 1000  # OSGB36 has x values in 0-700,000 range
        
        if original_crs_is_osgb36:
            logger.info("Converting bus coordinates from OSGB36 to WGS84 for clustering (required for haversine distance)")
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                
                # Store original coordinates for reference
                network.buses['x_osgb36'] = network.buses['x'].copy()
                network.buses['y_osgb36'] = network.buses['y'].copy()
                
                # Convert to WGS84
                new_x = []
                new_y = []
                for idx in network.buses.index:
                    x_osgb = network.buses.at[idx, 'x']
                    y_osgb = network.buses.at[idx, 'y']
                    if pd.notna(x_osgb) and pd.notna(y_osgb):
                        lon, lat = transformer.transform(x_osgb, y_osgb)
                        new_x.append(lon)
                        new_y.append(lat)
                    else:
                        new_x.append(x_osgb)
                        new_y.append(y_osgb)
                
                network.buses['x'] = new_x
                network.buses['y'] = new_y
                logger.info(f"Converted {len(network.buses)} bus coordinates to WGS84")
                logger.info(f"  New x range: {network.buses['x'].min():.4f} to {network.buses['x'].max():.4f} (lon)")
                logger.info(f"  New y range: {network.buses['y'].min():.4f} to {network.buses['y'].max():.4f} (lat)")
            except Exception as e:
                logger.error(f"Failed to convert coordinates: {e}")
                raise
        
        # CRITICAL: Dynamically add 'mean' aggregation for any coordinate-related bus columns
        # This MUST happen AFTER x_osgb36/y_osgb36 columns are created (above)
        # This prevents errors when new coordinate columns are added (e.g., x_osgb36, lat_rounded)
        coordinate_column_patterns = ['x_', 'y_', 'lat_', 'lon_', 'coord', 'osgb', 'wgs']
        for col in network.buses.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in coordinate_column_patterns):
                if col not in strategies.get('bus', {}):
                    if 'bus' not in strategies:
                        strategies['bus'] = {}
                    strategies['bus'][col] = 'mean'
                    logger.debug(f"Auto-added 'mean' strategy for coordinate column '{col}'")
        
        # CRITICAL: Ensure Line 'under_construction' column has consistent values BEFORE clustering
        # PyPSA aggregation fails if different lines in a cluster have different under_construction values
        # (e.g., NaN vs 0.0). Fill NaN with 0.0 since most lines are constructed.
        if 'under_construction' in network.lines.columns:
            network.lines['under_construction'] = network.lines['under_construction'].astype(float)
            nan_count = network.lines['under_construction'].isna().sum()
            if nan_count > 0:
                logger.info(f"Filling {nan_count} NaN values in lines 'under_construction' with 0.0 (before clustering)")
                network.lines['under_construction'] = network.lines['under_construction'].fillna(0.0)
        
        # Also clean up any other columns that might have mixed types
        if 'terrain_factor' in network.lines.columns:
            network.lines['terrain_factor'] = network.lines['terrain_factor'].astype(float).fillna(1.0)
        
        # Perform clustering
        logger.info("Applying PyPSA spatial clustering...")
        clustering = get_clustering_from_busmap(
            network,
            busmap,
            bus_strategies=strategies.get('bus', {}),
            line_strategies=strategies.get('line', {}),
            generator_strategies=strategies.get('generator', {}),
            one_port_strategies={
                'Load': strategies.get('load', {}),
                'StorageUnit': strategies.get('storage_unit', {}),
                'Store': strategies.get('store', {}),
            }
        )
        
        clustered_network = clustering.n
            
        # Add metadata
        clustered_network.meta = getattr(network, 'meta', {}).copy()
        clustered_network.meta.update({
            'clustering_method': clustering_method,
            'clustering_config': method_config,
            'original_buses': len(network.buses),
            'clustered_buses': len(clustered_network.buses),
            'reduction_ratio': len(clustered_network.buses) / len(network.buses)
        })
        
        logger.info(f"Clustering complete: {len(network.buses)} -> {len(clustered_network.buses)} buses")
        logger.info(f"Reduction ratio: {clustered_network.meta['reduction_ratio']:.3f}")
        
        # CRITICAL: Convert clustered bus coordinates BACK to OSGB36 if the original network was in OSGB36
        # This ensures consistent coordinate systems throughout the workflow
        if original_crs_is_osgb36:
            logger.info("Converting clustered bus coordinates back to OSGB36 for consistency")
            try:
                from pyproj import Transformer
                transformer_back = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
                
                # Store WGS84 coordinates for reference (useful for mapping)
                clustered_network.buses['lon'] = clustered_network.buses['x'].copy()
                clustered_network.buses['lat'] = clustered_network.buses['y'].copy()
                
                # Convert back to OSGB36
                new_x = []
                new_y = []
                for idx in clustered_network.buses.index:
                    lon = clustered_network.buses.at[idx, 'x']
                    lat = clustered_network.buses.at[idx, 'y']
                    if pd.notna(lon) and pd.notna(lat):
                        x_osgb, y_osgb = transformer_back.transform(lon, lat)
                        new_x.append(x_osgb)
                        new_y.append(y_osgb)
                    else:
                        new_x.append(lon)
                        new_y.append(lat)
                
                clustered_network.buses['x'] = new_x
                clustered_network.buses['y'] = new_y
                logger.info(f"Converted {len(clustered_network.buses)} clustered bus coordinates back to OSGB36")
                logger.info(f"  X range: {clustered_network.buses['x'].min():.0f} to {clustered_network.buses['x'].max():.0f} (meters)")
                logger.info(f"  Y range: {clustered_network.buses['y'].min():.0f} to {clustered_network.buses['y'].max():.0f} (meters)")
                logger.info(f"  WGS84 coordinates preserved in 'lon' and 'lat' columns for mapping")
            except Exception as e:
                logger.error(f"Failed to convert clustered coordinates back to OSGB36: {e}")
                raise
        
        # Verify final coordinate system
        x_range = clustered_network.buses['x'].max() - clustered_network.buses['x'].min()
        if x_range > 1000:
            logger.info(f"Clustered bus coordinates are in OSGB36 (x range: {x_range:.0f}m)")
        else:
            logger.info(f"Clustered bus coordinates are in WGS84 (x range: {x_range:.4f}°)")
        
        # Post-processing: Ensure no zero or missing r, x, b in clustered lines
        DEFAULT_R = 0.0001
        DEFAULT_X = 0.0001
        DEFAULT_B = 0.0001
        for param, default in [('r', DEFAULT_R), ('x', DEFAULT_X), ('b', DEFAULT_B)]:
            if param in clustered_network.lines.columns:
                before = clustered_network.lines[param].isna().sum() + (clustered_network.lines[param] == 0).sum()
                clustered_network.lines[param] = clustered_network.lines[param].replace(0, default).fillna(default)
                after = clustered_network.lines[param].isna().sum() + (clustered_network.lines[param] == 0).sum()
                replaced = before - after
                if replaced > 0:
                    logger.info(f"Replaced {replaced} zero or missing values in clustered lines '{param}' with default {default}.")
        return clustered_network
    
    def _handle_unassigned_buses(self, busmap: pd.Series, unassigned: pd.Series,
                                bus_gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame,
                                cluster_col: str, fallback_method: str) -> pd.Series:
        """Handle buses that weren't assigned to any cluster."""
        if fallback_method == 'nearest_centroid':
            # Find nearest cluster centroid
            centroids = boundaries.copy()
            centroids['centroid'] = boundaries.geometry.centroid
            
            for bus_id in unassigned.index:
                bus_point = bus_gdf.loc[bus_id].geometry
                distances = centroids['centroid'].distance(bus_point)
                nearest_cluster = centroids.loc[distances.idxmin(), cluster_col]
                busmap.at[bus_id] = nearest_cluster
                
        elif fallback_method == 'nearest_neighbor':
            # Assign to same cluster as nearest assigned bus
            assigned_buses = busmap.dropna()
            
            for bus_id in unassigned.index:
                bus_point = bus_gdf.loc[bus_id].geometry
                assigned_points = bus_gdf.loc[assigned_buses.index]
                distances = assigned_points.geometry.distance(bus_point)
                nearest_bus = distances.idxmin()
                busmap.at[bus_id] = assigned_buses.loc[nearest_bus]
                
        elif fallback_method == 'default_cluster':
            # Assign all unassigned buses to a default cluster
            default_name = f"unassigned_cluster"
            busmap.loc[unassigned.index] = default_name
            
        else:
            raise ValueError(f"Unknown fallback method: {fallback_method}")
            
        return busmap
    
    def _convert_transformers_to_lines(self, network: pypsa.Network) -> pypsa.Network:
        """Convert transformers to lines for simplified clustering."""
        logger.info("Converting transformers to lines for clustering...")
        
        for name, transformer in network.transformers.iterrows():
            s_nom = transformer.s_nom or 1.0
            network.add("Line",
                       name=f"tf_as_line_{name}",
                       bus0=transformer.bus0,
                       bus1=transformer.bus1,
                       s_nom=s_nom,
                       x=0.1,
                       r=0.0,
                       carrier="AC",
                       under_construction=0.0)  # Use 0.0 (float) to match ETYS lines
        
        # Remove original transformers
        network.transformers.drop(network.transformers.index, inplace=True)
        
        # Clean up non-standard attributes from lines before clustering to avoid PyPSA warnings
        attrs_to_remove = ['component', 'v_nom']
        for attr in attrs_to_remove:
            if attr in network.lines.columns:
                logger.debug(f"Removing non-standard attribute '{attr}' from lines before clustering")
                network.lines.drop(columns=[attr], inplace=True)
        
        # Clean up non-standard attributes from links before clustering
        link_attrs_to_remove = ['b', 'r']
        for attr in link_attrs_to_remove:
            if attr in network.links.columns:
                logger.debug(f"Removing non-standard attribute '{attr}' from links before clustering")
                network.links.drop(columns=[attr], inplace=True)
        
        # Ensure under_construction column has consistent float dtype for NetCDF export
        # NetCDF cannot save columns with mixed bool/float types
        if 'under_construction' in network.lines.columns:
            # Convert any boolean values to float (True->1.0, False->0.0)
            network.lines['under_construction'] = network.lines['under_construction'].astype(float)
            # Fill any remaining NaN values with 0.0
            nan_count = network.lines['under_construction'].isna().sum()
            if nan_count > 0:
                logger.info(f"Filling {nan_count} NaN values in lines 'under_construction' with 0.0")
                network.lines['under_construction'] = network.lines['under_construction'].fillna(0.0)
        
        return network


def main():
    """
    Main function for Snakemake integration.
    
    Expected Snakemake inputs:
    - snakemake.input[0]: Network file to cluster
    - snakemake.input[1:]: Additional inputs (boundaries, busmap files, etc.)
    
    Expected Snakemake params:
    - clustering_method: Method to use for clustering
    - method_config: Configuration dictionary for the method
    - aggregation_strategies: Optional custom aggregation strategies
    """
    global logger
    start_time = time()
    
    # Reinitialize logger with Snakemake log path if available
    if 'snakemake' in globals() and hasattr(snakemake, 'log') and snakemake.log:
        logger = setup_logging(snakemake.log[0])
    
    # Initialize clusterer
    clusterer = NetworkClusterer()
    
    # Load network
    network_path = snakemake.input[0]
    logger.info(f"Loading network from {network_path}")
    network = load_network(network_path, custom_logger=logger)
    
    # Get clustering parameters
    clustering_method = snakemake.params.get('clustering_method', 'spatial')
    method_config = snakemake.params.get('method_config', {})
    aggregation_strategies = snakemake.params.get('aggregation_strategies', None)
    
    # Add additional inputs to config if needed
    if len(snakemake.input) > 1:
        if clustering_method == 'spatial':
            method_config['boundaries_path'] = snakemake.input[1]
        elif clustering_method == 'busmap':
            method_config['busmap_source'] = snakemake.input[1]
    
    try:
        # Perform clustering and get busmap
        clustered_network = clusterer.cluster_network(
            network=network,
            clustering_method=clustering_method,
            method_config=method_config,
            aggregation_strategies=aggregation_strategies
        )

        # Extract busmap from clusterer (stored during clustering)
        if hasattr(clusterer, 'last_busmap'):
            busmap = clusterer.last_busmap
            logger.info(f"Retrieved busmap with {busmap.nunique()} clusters")
        else:
            logger.warning("No busmap found in clusterer")
            busmap = None

        # Clean up non-standard attributes to avoid PyPSA 1.0.2 warnings
        # These can appear during clustering when components are aggregated
        cleanup_attrs = {
            'lines': ['v_nom'],  # v_nom is for buses, not lines
            'links': ['b', 'r'],  # b, r are for lines, not links
        }
        for component, attrs in cleanup_attrs.items():
            if hasattr(clustered_network, component):
                comp_df = getattr(clustered_network, component)
                for attr in attrs:
                    if attr in comp_df.columns:
                        logger.debug(f"Removing non-standard attribute '{attr}' from {component}")
                        comp_df.drop(columns=[attr], inplace=True)

        # Save busmap CSV if output is specified and busmap exists
        if busmap is not None and len(snakemake.output) > 1 and hasattr(snakemake.output, 'busmap_csv'):
            busmap_path = snakemake.output.busmap_csv
            logger.info(f"Saving busmap to {busmap_path}")
            busmap_df = pd.DataFrame({'bus_id': busmap.index, 'cluster': busmap.values})
            busmap_df.to_csv(busmap_path, index=False)
        
        # Set network name and version metadata
        clustered_network.name = f"{clustering_method.capitalize()} Clustered Network"
        clustered_network.version = getattr(pypsa, "__version__", "unknown")

        # Save clustered network
        output_path = snakemake.output[0]
        logger.info(f"Saving clustered network to {output_path}")
        save_network(clustered_network, output_path, custom_logger=logger)

        logger.info("Network clustering completed successfully!")
        
        # Log execution summary
        log_execution_summary(
            logger,
            "Network Clustering",
            start_time,
            inputs={'network': network_path, 'method': clustering_method},
            outputs={'clustered_network': output_path, 'busmap': snakemake.output.busmap_csv if len(snakemake.output) > 1 else 'N/A'},
            context={
                'original_buses': len(network.buses),
                'clustered_buses': len(clustered_network.buses),
                'reduction_ratio': f"{clustered_network.meta.get('reduction_ratio', 0):.3f}",
                'n_clusters': busmap.nunique() if busmap is not None else 'N/A'
            }
        )

    except Exception as e:
        logger.error(f"Network clustering failed: {e}")
        raise


if __name__ == "__main__":
    main()

