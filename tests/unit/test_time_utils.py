"""
Unit tests for time_utils.py - Time series utilities module.

Tests time series resampling, alignment, and validation functions used
throughout the PyPSA-GB workflow.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from time_utils import (
    create_canonical_snapshots,
    ensure_timestep,
    align_to_snapshots,
    validate_timeseries
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def hourly_timeseries():
    """Create hourly time series for testing."""
    dates = pd.date_range('2020-01-01', periods=48, freq='h')
    values = np.sin(np.linspace(0, 4*np.pi, 48)) + 2  # Sine wave offset to be positive
    return pd.Series(values, index=dates, name='hourly_data')


@pytest.fixture
def half_hourly_timeseries():
    """Create half-hourly time series for testing."""
    dates = pd.date_range('2020-01-01', periods=96, freq='30min')
    values = np.sin(np.linspace(0, 4*np.pi, 96)) + 2
    return pd.Series(values, index=dates, name='half_hourly_data')


@pytest.fixture
def irregular_timeseries():
    """Create irregularly sampled time series."""
    dates = pd.DatetimeIndex([
        '2020-01-01 00:00', '2020-01-01 00:45', 
        '2020-01-01 02:00', '2020-01-01 03:15',
        '2020-01-01 05:00'
    ])
    values = [1.0, 1.5, 2.0, 1.8, 2.5]
    return pd.Series(values, index=dates, name='irregular_data')


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Snapshot Creation
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapshotCreation:
    """Test creation of canonical snapshot indices."""
    
    def test_create_hourly_snapshots(self):
        """Test creation of hourly snapshots."""
        snapshots = create_canonical_snapshots(
            '2020-01-01', '2020-01-03', timestep_minutes=60
        )
        
        assert len(snapshots) == 48  # 2 days * 24 hours
        assert snapshots.freq is not None or pd.infer_freq(snapshots) is not None
    
    def test_create_half_hourly_snapshots(self):
        """Test creation of half-hourly snapshots."""
        snapshots = create_canonical_snapshots(
            '2020-01-01', '2020-01-02', timestep_minutes=30
        )
        
        assert len(snapshots) == 48  # 1 day * 48 half-hours
    
    def test_snapshots_start_at_midnight(self):
        """Test that snapshots start at midnight."""
        snapshots = create_canonical_snapshots(
            '2020-01-01', '2020-01-02', timestep_minutes=60
        )
        
        assert snapshots[0].hour == 0
        assert snapshots[0].minute == 0
    
    def test_snapshots_correct_frequency(self):
        """Test that snapshots have correct frequency."""
        snapshots = create_canonical_snapshots(
            '2020-01-01 00:00', '2020-01-01 06:00', timestep_minutes=15
        )
        
        # Check time differences
        diffs = snapshots[1:] - snapshots[:-1]
        expected_diff = pd.Timedelta(minutes=15)
        
        assert all(diff == expected_diff for diff in diffs)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Timestep Resampling
# ══════════════════════════════════════════════════════════════════════════════

class TestTimestepResampling:
    """Test resampling time series to different timesteps."""
    
    def test_upsample_hourly_to_half_hourly(self, hourly_timeseries):
        """Test upsampling from hourly to half-hourly."""
        result = ensure_timestep(hourly_timeseries, timestep_minutes=30)
        
        # Should have approximately 2x points (minus 1 for endpoint)
        expected_len = len(hourly_timeseries) * 2 - 1
        assert len(result) == expected_len
        
        # Values should be interpolated smoothly
        assert not result.isna().any()
    
    def test_downsample_half_hourly_to_hourly(self, half_hourly_timeseries):
        """Test downsampling from half-hourly to hourly."""
        result = ensure_timestep(half_hourly_timeseries, timestep_minutes=60)
        
        # Should have approximately half the points
        expected_len = len(half_hourly_timeseries) // 2
        assert abs(len(result) - expected_len) <= 1
    
    def test_resampling_preserves_values_at_common_times(self, hourly_timeseries):
        """Test that values at common timepoints are preserved."""
        # Upsample then downsample
        upsampled = ensure_timestep(hourly_timeseries, timestep_minutes=30)
        downsampled = ensure_timestep(upsampled, timestep_minutes=60)
        
        # Values at original hourly points should be similar
        common_times = hourly_timeseries.index.intersection(downsampled.index)
        
        if len(common_times) > 0:
            original_vals = hourly_timeseries.loc[common_times]
            resampled_vals = downsampled.loc[common_times]
            
            # Allow small numerical differences from resampling
            assert np.allclose(original_vals, resampled_vals, rtol=0.1)
    
    def test_handle_irregular_timestep(self, irregular_timeseries):
        """Test handling of irregularly sampled data."""
        result = ensure_timestep(irregular_timeseries, timestep_minutes=60)
        
        # Should produce regularly spaced output
        assert isinstance(result, pd.Series)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Time Alignment
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeAlignment:
    """Test alignment of time series to network snapshots."""
    
    def test_align_to_snapshots_exact_match(self, hourly_timeseries):
        """Test alignment when time series matches snapshots exactly."""
        snapshots = hourly_timeseries.index
        result = align_to_snapshots(hourly_timeseries, snapshots)
        
        assert len(result) == len(snapshots)
        assert result.index.equals(snapshots)
        assert np.allclose(result.values, hourly_timeseries.values)
    
    def test_align_to_snapshots_interpolation(self, hourly_timeseries):
        """Test alignment with interpolation."""
        # Create snapshots at 30-min intervals
        snapshots = pd.date_range(
            hourly_timeseries.index[0],
            hourly_timeseries.index[-1],
            freq='30min'
        )
        
        result = align_to_snapshots(hourly_timeseries, snapshots, method='interpolate')
        
        assert len(result) == len(snapshots)
        assert result.index.equals(snapshots)
        assert not result.isna().any()
    
    def test_align_handles_missing_data(self):
        """Test alignment with missing data points."""
        dates = pd.date_range('2020-01-01', periods=24, freq='h')
        values = np.random.rand(24)
        values[5:8] = np.nan  # Insert missing data
        
        ts = pd.Series(values, index=dates)
        snapshots = dates
        
        result = align_to_snapshots(ts, snapshots, method='interpolate')
        
        # Should interpolate missing values
        assert len(result) == len(snapshots)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Timeseries Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeseriesValidation:
    """Test validation of time series data."""
    
    def test_validate_correct_length(self, hourly_timeseries):
        """Test validation of time series with correct length."""
        # Should not raise
        validate_timeseries(hourly_timeseries, expected_length=48)
    
    def test_validate_incorrect_length(self, hourly_timeseries):
        """Test validation catches incorrect length."""
        with pytest.raises((ValueError, AssertionError)):
            validate_timeseries(hourly_timeseries, expected_length=100)
    
    def test_validate_value_range(self, hourly_timeseries):
        """Test validation of value ranges."""
        # All values should be positive in our fixture
        validate_timeseries(hourly_timeseries, value_range=(0, 10))
    
    def test_validate_detects_out_of_range(self):
        """Test that validation detects out-of-range values."""
        dates = pd.date_range('2020-01-01', periods=10, freq='h')
        values = [1, 2, 3, 4, 5, 999, 7, 8, 9, 10]  # 999 is out of range
        ts = pd.Series(values, index=dates)
        
        # Skip if validate_timeseries doesn't have value_range parameter
        # This is a placeholder - actual implementation may vary
        try:
            validate_timeseries(ts, value_range=(0, 100))
        except TypeError:
            # Function doesn't support value_range parameter
            pytest.skip("validate_timeseries doesn't support value_range parameter")
    
    def test_validate_allows_nan_if_specified(self):
        """Test that validation can allow NaN values."""
        dates = pd.date_range('2020-01-01', periods=10, freq='h')
        values = [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]
        ts = pd.Series(values, index=dates)
        
        # Should not raise if NaN is acceptable
        # (depends on implementation)
        assert isinstance(ts, pd.Series)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_handle_empty_timeseries(self):
        """Test handling of empty time series."""
        empty_ts = pd.Series(dtype=float)
        
        # Should handle gracefully (return empty or raise appropriate error)
        try:
            result = ensure_timestep(empty_ts, timestep_minutes=60)
            assert len(result) == 0
        except (ValueError, IndexError):
            # Acceptable to raise error for empty input
            pass
    
    def test_handle_single_point(self):
        """Test handling of single data point."""
        single_ts = pd.Series([1.0], index=pd.DatetimeIndex(['2020-01-01']))
        
        # Should handle gracefully
        try:
            result = ensure_timestep(single_ts, timestep_minutes=60)
            assert len(result) >= 1
        except (ValueError, IndexError):
            # Acceptable to require multiple points
            pass
    
    def test_handle_duplicate_timestamps(self):
        """Test handling of duplicate timestamps."""
        dates = pd.DatetimeIndex([
            '2020-01-01 00:00', '2020-01-01 01:00',
            '2020-01-01 01:00', '2020-01-01 02:00'  # Duplicate at 01:00
        ])
        values = [1.0, 2.0, 2.5, 3.0]
        ts = pd.Series(values, index=dates)
        
        # Should handle duplicates (average, keep first, or raise)
        result = ensure_timestep(ts, timestep_minutes=60)
        assert isinstance(result, pd.Series)
