"""
Timeseries Synchronization Utilities for PyPSA-GB

This module ensures that renewable generation profiles and other timeseries data
are properly synchronized with the network's temporal structure.

Key Issues Addressed:
1. Mismatch between renewable profile timesteps and network snapshots
2. Inconsistent temporal indexing across different data sources
3. Missing timesteps or gaps in renewable generation data
4. Proper handling of half-hourly vs hourly resolution

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pypsa
from datetime import datetime, timedelta
import pytz

# Configure logging
logger = None
try:
    from scripts.utilities.logging_config import setup_logging
    logger = setup_logging("timeseries_synchronization")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging
        logger = setup_logging("timeseries_synchronization")
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("timeseries_synchronization")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TimeseriesSynchronizer:
    """
    Synchronize timeseries data with PyPSA network temporal structure.
    
    This class handles temporal alignment, resampling, and validation
    of timeseries data to ensure consistency with PyPSA network snapshots.
    """
    
    def __init__(self, reference_snapshots: pd.DatetimeIndex = None):
        """
        Initialize synchronizer with reference temporal structure.
        
        Parameters
        ----------
        reference_snapshots : pd.DatetimeIndex, optional
            Reference snapshots to align data to. If None, must be set later.
        """
        self.reference_snapshots = reference_snapshots
        self.timezone = 'UTC'  # Standard timezone for energy modeling
        
    def set_reference_snapshots(self, snapshots: pd.DatetimeIndex):
        """Set reference snapshots for synchronization."""
        self.reference_snapshots = snapshots
        logger.info(f"Set reference snapshots: {len(snapshots)} periods from {snapshots[0]} to {snapshots[-1]}")
    
    def validate_timeseries_index(self, ts_data: pd.DataFrame, data_name: str = "timeseries") -> Dict[str, any]:
        """
        Validate timeseries index for consistency issues.
        
        Parameters
        ----------
        ts_data : pd.DataFrame
            Timeseries data to validate
        data_name : str
            Name of the dataset for logging
            
        Returns
        -------
        Dict[str, any]
            Validation report with issues and recommendations
        """
        issues = []
        warnings_list = []
        
        # Check index type
        if not isinstance(ts_data.index, pd.DatetimeIndex):
            issues.append("Index is not DatetimeIndex")
            try:
                ts_data.index = pd.to_datetime(ts_data.index)
                warnings_list.append("Converted index to DatetimeIndex")
            except Exception as e:
                issues.append(f"Cannot convert index to datetime: {e}")
        
        # Check for timezone
        if ts_data.index.tz is None:
            warnings_list.append("Index has no timezone information")
        elif str(ts_data.index.tz) != self.timezone:
            warnings_list.append(f"Index timezone {ts_data.index.tz} differs from standard {self.timezone}")
        
        # Check for duplicates
        duplicates = ts_data.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")
        
        # Check for gaps
        if len(ts_data) > 1:
            time_diffs = ts_data.index[1:] - ts_data.index[:-1]
            median_freq = time_diffs.median()
            irregular_gaps = (time_diffs != median_freq).sum()
            if irregular_gaps > 0:
                warnings_list.append(f"Found {irregular_gaps} irregular time gaps")
        
        # Check alignment with reference if available
        if self.reference_snapshots is not None:
            overlap = len(set(ts_data.index).intersection(set(self.reference_snapshots)))
            coverage = overlap / len(self.reference_snapshots) * 100
            if coverage < 100:
                issues.append(f"Timeseries covers only {coverage:.1f}% of reference snapshots")
        
        report = {
            'data_name': data_name,
            'length': len(ts_data),
            'start_time': ts_data.index[0] if len(ts_data) > 0 else None,
            'end_time': ts_data.index[-1] if len(ts_data) > 0 else None,
            'frequency': self._detect_frequency(ts_data.index),
            'timezone': str(ts_data.index.tz) if ts_data.index.tz else None,
            'issues': issues,
            'warnings': warnings_list,
            'valid': len(issues) == 0
        }
        
        return report
    
    def _detect_frequency(self, index: pd.DatetimeIndex) -> str:
        """Detect the frequency of a datetime index."""
        if len(index) < 2:
            return "Unknown"
        
        time_diffs = index[1:] - index[:-1]
        median_diff = time_diffs.median()
        
        if median_diff == timedelta(minutes=30):
            return "30min"
        elif median_diff == timedelta(hours=1):
            return "1H"
        elif median_diff == timedelta(days=1):
            return "1D"
        else:
            return f"Custom: {median_diff}"
    
    def resample_to_reference(self, ts_data: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        """
        Resample timeseries data to match reference snapshots.
        
        Parameters
        ----------
        ts_data : pd.DataFrame
            Input timeseries data
        method : str
            Resampling method: 'interpolate', 'ffill', 'bfill', 'mean'
            
        Returns
        -------
        pd.DataFrame
            Resampled data aligned to reference snapshots
        """
        if self.reference_snapshots is None:
            raise ValueError("Reference snapshots not set. Call set_reference_snapshots() first.")
        
        logger.info(f"Resampling {len(ts_data)} records to {len(self.reference_snapshots)} reference snapshots")
        
        # Ensure timezone consistency
        if ts_data.index.tz is None and self.reference_snapshots.tz is not None:
            ts_data.index = ts_data.index.tz_localize(self.reference_snapshots.tz)
        elif ts_data.index.tz is not None and self.reference_snapshots.tz is None:
            self.reference_snapshots = self.reference_snapshots.tz_localize(ts_data.index.tz)
        
        # Create target index
        target_index = pd.DatetimeIndex(self.reference_snapshots, name=ts_data.index.name or 'time')
        
        # Reindex to target timestamps
        if method == "interpolate":
            # First reindex to include all needed timestamps
            combined_index = ts_data.index.union(target_index).sort_values()
            ts_expanded = ts_data.reindex(combined_index)
            
            # Interpolate missing values
            ts_expanded = ts_expanded.interpolate(method='time', limit_direction='both')
            
            # Select only target timestamps
            result = ts_expanded.loc[target_index]
            
        elif method == "ffill":
            result = ts_data.reindex(target_index, method='ffill')
            
        elif method == "bfill":
            result = ts_data.reindex(target_index, method='bfill')
            
        elif method == "mean":
            # Resample using pandas resample with mean aggregation
            freq = self._detect_reference_frequency()
            ts_resampled = ts_data.resample(freq).mean()
            result = ts_resampled.reindex(target_index, method='nearest')
            
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        # Fill any remaining NaN values
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Resampling complete. Output shape: {result.shape}")
        return result
    
    def _detect_reference_frequency(self) -> str:
        """Detect frequency of reference snapshots for resampling."""
        if self.reference_snapshots is None or len(self.reference_snapshots) < 2:
            return "1H"  # Default to hourly
        
        freq_str = self._detect_frequency(self.reference_snapshots)
        if freq_str == "30min":
            return "30min"
        elif freq_str == "1H":
            return "1H"
        else:
            return "1H"  # Default fallback
    
    def synchronize_renewable_profiles(self, profiles_data: pd.DataFrame, 
                                     profile_type: str = "renewable") -> pd.DataFrame:
        """
        Synchronize renewable generation profiles with network snapshots.
        
        Parameters
        ----------
        profiles_data : pd.DataFrame
            Renewable generation profiles (columns = sites, index = time)
        profile_type : str
            Type of profiles for logging
            
        Returns
        -------
        pd.DataFrame
            Synchronized profiles
        """
        logger.info(f"Synchronizing {profile_type} profiles")
        
        # Validate input data
        validation_report = self.validate_timeseries_index(profiles_data, profile_type)
        
        if not validation_report['valid']:
            logger.warning(f"Validation issues found in {profile_type} data:")
            for issue in validation_report['issues']:
                logger.warning(f"  - {issue}")
        
        # Log warnings
        for warning in validation_report.get('warnings', []):
            logger.info(f"  - {warning}")
        
        # Perform synchronization
        synchronized_data = self.resample_to_reference(profiles_data, method="interpolate")
        
        # Validate output
        final_validation = self.validate_timeseries_index(synchronized_data, f"{profile_type}_synchronized")
        if final_validation['valid']:
            logger.info(f"Successfully synchronized {profile_type} profiles")
        else:
            logger.error(f"Synchronization failed for {profile_type} profiles")
        
        return synchronized_data
    
    def add_timeseries_to_network(self, network: pypsa.Network, 
                                timeseries_data: Dict[str, pd.DataFrame],
                                component_type: str = "Generator") -> None:
        """
        Add synchronized timeseries data to PyPSA network components.
        
        Parameters
        ----------
        network : pypsa.Network
            PyPSA network object
        timeseries_data : Dict[str, pd.DataFrame]
            Dictionary mapping component names to timeseries data
        component_type : str
            Type of PyPSA component ('Generator', 'Load', 'StorageUnit', etc.)
        """
        logger.info(f"Adding timeseries data to {component_type} components")
        
        # Set network snapshots if not already set
        if len(network.snapshots) <= 1 and self.reference_snapshots is not None:
            network.set_snapshots(self.reference_snapshots)
            logger.info(f"Set network snapshots to {len(self.reference_snapshots)} periods")
        
        # Validate network snapshots alignment
        if self.reference_snapshots is not None:
            if not network.snapshots.equals(self.reference_snapshots):
                logger.warning("Network snapshots don't match reference snapshots")
        
        # Add timeseries data
        added_count = 0
        for component_name, ts_data in timeseries_data.items():
            try:
                # Check if component exists
                if component_type == "Generator" and component_name in network.generators.index:
                    # Add p_max_pu timeseries for generators
                    network.generators_t.p_max_pu[component_name] = ts_data.iloc[:, 0] if len(ts_data.columns) > 0 else ts_data
                    added_count += 1
                    
                elif component_type == "Load" and component_name in network.loads.index:
                    # Add p_set timeseries for loads
                    network.loads_t.p_set[component_name] = ts_data.iloc[:, 0] if len(ts_data.columns) > 0 else ts_data
                    added_count += 1
                    
                elif component_type == "StorageUnit" and component_name in network.storage_units.index:
                    # Add p_max_pu or inflow timeseries for storage
                    network.storage_units_t.inflow[component_name] = ts_data.iloc[:, 0] if len(ts_data.columns) > 0 else ts_data
                    added_count += 1
                    
                else:
                    logger.warning(f"Component {component_name} not found in network {component_type} components")
                    
            except Exception as e:
                logger.error(f"Failed to add timeseries for {component_name}: {e}")
        
        logger.info(f"Successfully added timeseries data to {added_count} {component_type} components")
    
    def check_network_timeseries_consistency(self, network: pypsa.Network) -> Dict[str, any]:
        """
        Check PyPSA network for timeseries consistency issues.
        
        Parameters
        ----------
        network : pypsa.Network
            PyPSA network to check
            
        Returns
        -------
        Dict[str, any]
            Consistency report
        """
        logger.info("Checking network timeseries consistency")
        
        issues = []
        warnings_list = []
        
        # Check snapshots
        if len(network.snapshots) <= 1:
            issues.append("Network has no or minimal snapshots defined")
        
        # Check generators timeseries
        if hasattr(network, 'generators_t') and hasattr(network.generators_t, 'p_max_pu'):
            gen_ts = network.generators_t.p_max_pu
            if not gen_ts.empty:
                # Check index alignment
                if not gen_ts.index.equals(network.snapshots):
                    issues.append("Generator timeseries index doesn't match network snapshots")
                
                # Check for missing data
                missing_data = gen_ts.isna().sum().sum()
                if missing_data > 0:
                    warnings_list.append(f"Generator timeseries has {missing_data} missing values")
        
        # Check loads timeseries
        if hasattr(network, 'loads_t') and hasattr(network.loads_t, 'p_set'):
            load_ts = network.loads_t.p_set
            if not load_ts.empty:
                if not load_ts.index.equals(network.snapshots):
                    issues.append("Load timeseries index doesn't match network snapshots")
                
                missing_data = load_ts.isna().sum().sum()
                if missing_data > 0:
                    warnings_list.append(f"Load timeseries has {missing_data} missing values")
        
        report = {
            'snapshots_count': len(network.snapshots),
            'snapshots_start': network.snapshots[0] if len(network.snapshots) > 0 else None,
            'snapshots_end': network.snapshots[-1] if len(network.snapshots) > 0 else None,
            'generators_with_timeseries': len(network.generators_t.p_max_pu.columns) if hasattr(network, 'generators_t') else 0,
            'loads_with_timeseries': len(network.loads_t.p_set.columns) if hasattr(network, 'loads_t') else 0,
            'issues': issues,
            'warnings': warnings_list,
            'consistent': len(issues) == 0
        }
        
        return report


def create_standard_snapshots(start_date: str, end_date: str, freq: str = "1H", 
                            timezone: str = "UTC") -> pd.DatetimeIndex:
    """
    Create standard snapshot structure for PyPSA-GB networks.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format  
    freq : str
        Frequency string ('30min', '1H', '1D')
    timezone : str
        Timezone for snapshots
        
    Returns
    -------
    pd.DatetimeIndex
        Standard snapshots for the specified period
    """
    snapshots = pd.date_range(
        start=start_date,
        end=end_date, 
        freq=freq,
        tz=timezone,
        name='snapshots'
    )
    
    logger.info(f"Created {len(snapshots)} snapshots from {start_date} to {end_date} at {freq} frequency")
    return snapshots


def fix_renewable_profiles_timesteps(profiles_file: str, network_file: str, 
                                   output_file: str, freq: str = "1H") -> None:
    """
    Fix renewable profiles to match network timestep structure.
    
    Parameters
    ----------
    profiles_file : str
        Path to renewable profiles CSV
    network_file : str  
        Path to PyPSA network file
    output_file : str
        Path for synchronized profiles output
    freq : str
        Target frequency for synchronization
    """
    logger.info(f"Fixing renewable profiles timestep alignment")
    
    # Load network to get reference snapshots
    network = pypsa.Network(network_file)
    
    # Load renewable profiles
    profiles_df = pd.read_csv(profiles_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded renewable profiles: {profiles_df.shape}")
    
    # Create synchronizer
    synchronizer = TimeseriesSynchronizer(network.snapshots)
    
    # Synchronize profiles
    synchronized_profiles = synchronizer.synchronize_renewable_profiles(profiles_df, "renewable_generation")
    
    # Save synchronized profiles
    synchronized_profiles.to_csv(output_file)
    logger.info(f"Saved synchronized profiles to {output_file}")
    
    return synchronized_profiles


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create test snapshots
    test_snapshots = create_standard_snapshots("2020-01-01", "2020-01-02", freq="1H")
    
    # Initialize synchronizer
    sync = TimeseriesSynchronizer(test_snapshots)
    
    # Create test data with misaligned timestamps
    test_index = pd.date_range("2020-01-01 00:30", "2020-01-02 00:30", freq="1H")
    test_data = pd.DataFrame(
        data=np.random.rand(len(test_index), 3),
        index=test_index,
        columns=['site_1', 'site_2', 'site_3']
    )
    
    # Test synchronization
    print("Original data shape:", test_data.shape)
    synchronized = sync.synchronize_renewable_profiles(test_data)
    print("Synchronized data shape:", synchronized.shape)
    print("Reference snapshots length:", len(test_snapshots))

