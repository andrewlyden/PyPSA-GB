"""
Time utilities for PyPSA-GB project.
Handles time series resampling and alignment to config timesteps.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from scripts.utilities.logging_config import setup_logging

logger = setup_logging("time_utils")

def load_config_timestep(scenario: str = None) -> int:
    """
    Load timestep_minutes from scenario config.
    
    Parameters
    ----------
    scenario : str, optional
        Scenario name to get timestep from. If None, tries to get from snakemake context.
        
    Returns
    -------
    int
        Timestep in minutes (default: 60)
    """
    # Try to get from snakemake context first
    try:
        if 'snakemake' in globals():
            snk = snakemake  # type: ignore
            if hasattr(snk, 'wildcards') and hasattr(snk.wildcards, 'scenario'):
                scenario = snk.wildcards.scenario
            elif hasattr(snk, 'params') and 'timestep_minutes' in snk.params:
                timestep = snk.params['timestep_minutes']
                logger.info(f"Loaded timestep_minutes from snakemake params: {timestep}")
                return timestep
    except NameError:
        pass
    
    # Load from scenario config if scenario is provided
    if scenario:
        scenarios_path = Path("config/scenarios_master.yaml")
        if scenarios_path.exists():
            with open(scenarios_path, 'r') as f:
                scenarios_config = yaml.safe_load(f)
            
            if 'scenarios' in scenarios_config and scenario in scenarios_config['scenarios']:
                timestep = scenarios_config['scenarios'][scenario].get('timestep_minutes', 60)
                logger.info(f"Loaded timestep_minutes for scenario '{scenario}': {timestep}")
                return timestep
    
    # Fallback to old config.yaml for backward compatibility
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'timestep_minutes' in config:
            timestep = config['timestep_minutes']
            logger.info(f"Loaded timestep_minutes from legacy config.yaml: {timestep}")
            return timestep
    
    # Default fallback
    logger.warning(f"Could not load timestep_minutes, using default 60 minutes")
    return 60

def create_canonical_snapshots(start_date: str, end_date: str, timestep_minutes: int) -> pd.DatetimeIndex:
    """
    Create canonical snapshot index for the scenario.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format  
    timestep_minutes : int
        Timestep in minutes
        
    Returns
    -------
    pd.DatetimeIndex
        Canonical snapshot index
    """
    freq = f"{timestep_minutes}min"
    snapshots = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive='left')
    logger.info(f"Created {len(snapshots)} snapshots from {start_date} to {end_date} at {timestep_minutes}-minute intervals")
    return snapshots

def ensure_timestep(ts, timestep_minutes: int, how: dict = None, name: str = "timeseries") -> pd.Series:
    """
    Resample time series to match config timestep.
    
    Parameters
    ----------
    ts : pd.Series or pd.DataFrame
        Time series with datetime index
    timestep_minutes : int
        Target timestep in minutes
    how : dict, optional
        Resampling method per scenario:
        - 'interpolate': linear interpolation (default for continuous data)
        - 'ffill': forward fill (discrete states) 
        - 'mean': average (for downsampling power)
        - 'sum': sum (for energy totals)
        - 'max': maximum (for peaks)
    name : str
        Name for logging purposes
        
    Returns
    -------
    pd.Series or pd.DataFrame
        Resampled time series
    """
    if how is None:
        how = {'default': 'interpolate'}
    
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: Index must be datetime, got {type(ts.index)}")
    
    # Determine current frequency
    try:
        current_freq = pd.infer_freq(ts.index)
        if current_freq is None:
            # Try to infer from first few intervals
            if len(ts.index) > 1:
                delta = ts.index[1] - ts.index[0]
                current_minutes = delta.total_seconds() / 60
            else:
                raise ValueError(f"{name}: Cannot infer frequency from single timestamp")
        else:
            # Parse frequency string to minutes
            if current_freq.endswith('H'):
                current_minutes = int(current_freq[:-1]) * 60
            elif current_freq.endswith('min'):
                current_minutes = int(current_freq[:-3])
            else:
                raise ValueError(f"{name}: Unsupported frequency {current_freq}")
    except:
        # Fallback: compute from time differences
        if len(ts.index) > 1:
            delta = ts.index[1] - ts.index[0]
            current_minutes = delta.total_seconds() / 60
            logger.warning(f"{name}: Could not infer frequency, estimated {current_minutes} minutes from time diff")
        else:
            raise ValueError(f"{name}: Cannot determine time frequency")
    
    logger.info(f"{name}: Current timestep {current_minutes} min, target {timestep_minutes} min")
    
    # If already at target resolution, return as-is
    if abs(current_minutes - timestep_minutes) < 0.1:  # Allow small numerical differences
        logger.info(f"{name}: Already at target timestep")
        return ts
    
    # Determine target frequency string
    target_freq = f"{timestep_minutes}min"
    method = how.get('default', 'interpolate')
    
    if current_minutes > timestep_minutes:
        # Upsampling (e.g., hourly -> half-hourly)
        logger.info(f"{name}: Upsampling from {current_minutes} to {timestep_minutes} min using {method}")
        
        if method == 'interpolate':
            # Create new index and interpolate
            new_index = pd.date_range(start=ts.index[0], end=ts.index[-1], freq=target_freq)
            ts_resampled = ts.reindex(new_index).interpolate(method='linear')
        elif method == 'ffill':
            new_index = pd.date_range(start=ts.index[0], end=ts.index[-1], freq=target_freq)
            ts_resampled = ts.reindex(new_index, method='ffill')
        else:
            raise ValueError(f"Unsupported upsampling method: {method}")
            
    else:
        # Downsampling (e.g., 15-min -> hourly)
        logger.info(f"{name}: Downsampling from {current_minutes} to {timestep_minutes} min using {method}")
        
        if method == 'mean':
            ts_resampled = ts.resample(target_freq).mean()
        elif method == 'sum':
            ts_resampled = ts.resample(target_freq).sum()
        elif method == 'max':
            ts_resampled = ts.resample(target_freq).max()
        elif method == 'ffill':
            # Take first value in each period
            ts_resampled = ts.resample(target_freq).first()
        else:
            # Default to mean for downsampling
            logger.warning(f"{name}: Unknown downsampling method {method}, using mean")
            ts_resampled = ts.resample(target_freq).mean()
    
    # Log resampling results
    original_length = len(ts)
    new_length = len(ts_resampled)
    logger.info(f"{name}: Resampled from {original_length} to {new_length} timesteps")
    
    # Check for NaNs
    if isinstance(ts_resampled, pd.Series):
        nan_count = ts_resampled.isna().sum()
    else:
        nan_count = ts_resampled.isna().sum().sum()
    
    if nan_count > 0:
        logger.warning(f"{name}: {nan_count} NaN values after resampling")
    
    return ts_resampled

def align_to_snapshots(ts, snapshots: pd.DatetimeIndex, method: str = 'interpolate', name: str = "timeseries"):
    """
    Align time series to specific snapshot index.
    
    Parameters
    ----------
    ts : pd.Series or pd.DataFrame
        Time series to align
    snapshots : pd.DatetimeIndex
        Target snapshot index
    method : str
        Alignment method ('interpolate', 'ffill', 'nearest')
    name : str
        Name for logging
        
    Returns
    -------
    pd.Series or pd.DataFrame
        Aligned time series
    """
    logger.info(f"{name}: Aligning to {len(snapshots)} snapshots")
    
    if method == 'interpolate':
        # Reindex and interpolate to snapshots
        aligned = ts.reindex(ts.index.union(snapshots)).interpolate(method='linear').reindex(snapshots)
    elif method == 'ffill':
        aligned = ts.reindex(snapshots, method='ffill')
    elif method == 'nearest':
        aligned = ts.reindex(snapshots, method='nearest')
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # Check results
    nan_count = aligned.isna().sum() if isinstance(aligned, pd.Series) else aligned.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"{name}: {nan_count} NaN values after alignment to snapshots")
    
    logger.info(f"{name}: Successfully aligned to snapshots")
    return aligned

def validate_timeseries(ts, expected_length: int = None, value_range: tuple = None, name: str = "timeseries"):
    """
    Validate time series properties.
    
    Parameters
    ----------
    ts : pd.Series or pd.DataFrame
        Time series to validate
    expected_length : int, optional
        Expected number of timesteps
    value_range : tuple, optional
        Expected (min, max) value range
    name : str
        Name for logging
    """
    logger.info(f"Validating {name}")
    
    # Check length
    if expected_length is not None:
        if len(ts) != expected_length:
            raise ValueError(f"{name}: Expected length {expected_length}, got {len(ts)}")
        logger.info(f"{name}: Length validation passed ({len(ts)} timesteps)")
    
    # Check for NaNs
    if isinstance(ts, pd.Series):
        nan_count = ts.isna().sum()
        if value_range:
            valid_values = ts.dropna()
            if len(valid_values) > 0:
                actual_min, actual_max = valid_values.min(), valid_values.max()
                if actual_min < value_range[0] or actual_max > value_range[1]:
                    logger.warning(f"{name}: Values outside expected range {value_range}, got [{actual_min:.3f}, {actual_max:.3f}]")
    else:
        nan_count = ts.isna().sum().sum()
        if value_range:
            valid_values = ts.dropna()
            if not valid_values.empty:
                actual_min, actual_max = valid_values.min().min(), valid_values.max().max()
                if actual_min < value_range[0] or actual_max > value_range[1]:
                    logger.warning(f"{name}: Values outside expected range {value_range}, got [{actual_min:.3f}, {actual_max:.3f}]")
    
    if nan_count > 0:
        logger.warning(f"{name}: Contains {nan_count} NaN values")
    else:
        logger.info(f"{name}: No NaN values found")
    
    logger.info(f"{name}: Validation completed")

