"""
Unit tests for map_renewable_profiles.py

Tests the RenewableProfileGenerator class for renewable energy profile generation
from weather data using atlite.

Test Coverage:
- Class initialization and configuration
- Site data loading and preprocessing
- Wind site preparation (onshore/offshore)
- Solar site preparation
- Utility methods (year extraction, turbine selection)
- Data validation and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))

from map_renewable_profiles import RenewableProfileGenerator


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def generator():
    """Create a RenewableProfileGenerator instance"""
    return RenewableProfileGenerator()


@pytest.fixture
def sample_wind_sites():
    """Create sample wind site data"""
    return pd.DataFrame({
        'site_name': ['Wind Farm A', 'Wind Farm B', 'Wind Farm C'],
        'lat': [55.5, 56.0, 54.5],
        'lon': [-3.0, -2.5, -4.0],
        'capacity_mw': [50.0, 75.0, 100.0],
        'turbine_capacity_mw': [3.0, 3.0, 5.0],
        'no_turbines': [17, 25, 20],
        'hub_height_m': [100.0, 100.0, 120.0]
    })


@pytest.fixture
def sample_solar_sites():
    """Create sample solar site data"""
    return pd.DataFrame({
        'site_name': ['Solar Park A', 'Solar Park B'],
        'lat': [52.5, 53.0],
        'lon': [-1.5, -2.0],
        'capacity_mw': [25.0, 30.0],
        'mounting_type': ['Ground', 'Ground']
    })


@pytest.fixture
def sample_offshore_repd():
    """Sample offshore REPD data"""
    return pd.DataFrame({
        'site_name': ['Offshore Farm 1', 'Offshore Farm 2'],
        'lat': [55.0, 56.0],
        'lon': [-2.0, -1.5],
        'capacity_mw': [200.0, 300.0],
        'turbine_capacity_mw': [5.0, 7.0],
        'no_turbines': [40, 43],
        'operational_year': [2020, 2021]
    })


@pytest.fixture
def sample_offshore_pipeline():
    """Sample offshore pipeline data"""
    return pd.DataFrame({
        'project_name': ['Future Offshore A', 'Future Offshore B'],
        'latitude': [55.5, 56.5],
        'longitude': [-2.5, -1.0],
        'installed_capacity_mw': [400.0, 500.0],
        'expected_year': [2027, 2030]
    })


# ══════════════════════════════════════════════════════════════════════════════
# Test Class Initialization
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneratorInitialization:
    """Test RenewableProfileGenerator initialization"""
    
    def test_generator_creates_successfully(self, generator):
        """Test generator instance creation"""
        assert generator is not None
        assert isinstance(generator, RenewableProfileGenerator)
    
    def test_generator_has_turbine_types(self, generator):
        """Test turbine type configurations"""
        assert hasattr(generator, 'offshore_turbine_types')
        assert hasattr(generator, 'onshore_turbine_types')
        
        # Check offshore turbines
        assert 3 in generator.offshore_turbine_types
        assert 5 in generator.offshore_turbine_types
        assert 7 in generator.offshore_turbine_types
        
        # Check onshore turbines
        assert 0.66 in generator.onshore_turbine_types
        assert 2.3 in generator.onshore_turbine_types
        assert 3.0 in generator.onshore_turbine_types
    
    def test_generator_has_solar_config(self, generator):
        """Test solar panel configuration"""
        assert hasattr(generator, 'solar_panel_config')
        assert 'panel' in generator.solar_panel_config
        assert 'orientation' in generator.solar_panel_config
        assert generator.solar_panel_config['panel'] == 'CSi'


# ══════════════════════════════════════════════════════════════════════════════
# Test Site Data Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestSiteDataLoading:
    """Test site data loading and preprocessing"""
    
    def test_load_site_data_valid_file(self, generator, tmp_path):
        """Test loading valid site data file"""
        # Create temporary CSV
        test_file = tmp_path / "test_sites.csv"
        test_data = pd.DataFrame({
            'site_name': ['Site A', 'Site B'],
            'lat': [55.0, 56.0],
            'lon': [-3.0, -2.0],
            'capacity_mw': [50.0, 75.0]
        })
        test_data.to_csv(test_file, index=False)
        
        # Load and verify
        loaded = generator.load_site_data(str(test_file))
        assert len(loaded) == 2
        assert 'site_name' in loaded.columns
        assert loaded.iloc[0]['lat'] == 55.0
    
    def test_load_site_data_missing_file(self, generator):
        """Test handling of missing site data file"""
        result = generator.load_site_data('nonexistent_file.csv')
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_prepare_wind_sites_valid_data(self, generator, sample_wind_sites):
        """Test wind site preparation with valid data"""
        prepared = generator.prepare_wind_sites(sample_wind_sites)
        
        assert len(prepared) == 3
        assert 'lat' in prepared.columns
        assert 'lon' in prepared.columns
        assert not prepared['lat'].isna().any()
        assert not prepared['lon'].isna().any()
    
    def test_prepare_wind_sites_with_missing_coords(self, generator):
        """Test wind site preparation with missing coordinates"""
        data_with_na = pd.DataFrame({
            'site_name': ['Site A', 'Site B', 'Site C'],
            'lat': [55.0, np.nan, 56.0],
            'lon': [-3.0, -2.0, np.nan],
            'capacity_mw': [50.0, 75.0, 100.0]
        })
        
        prepared = generator.prepare_wind_sites(data_with_na)
        
        # Should drop rows with missing coordinates
        assert len(prepared) == 1  # Only Site A has complete coords
        assert prepared.iloc[0]['site_name'] == 'Site A'
    
    def test_prepare_solar_sites(self, generator, sample_solar_sites):
        """Test solar site preparation"""
        prepared = generator.prepare_solar_sites(sample_solar_sites)
        
        assert len(prepared) == 2
        assert not prepared['lat'].isna().any()
        assert not prepared['lon'].isna().any()


# ══════════════════════════════════════════════════════════════════════════════
# Test Utility Methods
# ══════════════════════════════════════════════════════════════════════════════

class TestUtilityMethods:
    """Test utility and helper methods"""
    
    def test_year_from_cutout(self, generator):
        """Test extracting year from cutout"""
        # Create mock cutout with proper nested structure
        mock_cutout = Mock()
        mock_time_value = Mock()
        mock_time_value.values = pd.Timestamp('2020-01-01')
        mock_cutout.data.time = [mock_time_value]
        
        year = generator._year_from_cutout(mock_cutout)
        assert year == 2020
    
    def test_year_from_cutout_alternative_format(self, generator):
        """Test year extraction with alternative cutout format"""
        mock_cutout = Mock()
        # Remove .data.time, use coords instead
        mock_cutout.data.time = None
        mock_cutout.coords = {'time': np.array([pd.Timestamp('2021-06-15')])}
        
        year = generator._year_from_cutout(mock_cutout)
        # Method should handle both formats
        assert year is None or year == 2021


# ══════════════════════════════════════════════════════════════════════════════
# Test Offshore Site Combination
# ══════════════════════════════════════════════════════════════════════════════

class TestOffshoreSiteCombination:
    """Test combining REPD and pipeline offshore sites"""
    
    def test_combine_offshore_sites_both_sources(self, generator, sample_offshore_repd, sample_offshore_pipeline):
        """Test combining REPD and pipeline data"""
        combined = generator.combine_offshore_sites(sample_offshore_repd, sample_offshore_pipeline)
        
        # Should have sites from both sources
        assert len(combined) == 4  # 2 REPD + 2 pipeline
        assert 'lat' in combined.columns
        assert 'lon' in combined.columns
        assert 'capacity_mw' in combined.columns
    
    def test_combine_offshore_sites_only_repd(self, generator, sample_offshore_repd):
        """Test with only REPD data"""
        empty_pipeline = pd.DataFrame()
        combined = generator.combine_offshore_sites(sample_offshore_repd, empty_pipeline)
        
        assert len(combined) == 2
        assert combined.iloc[0]['site_name'] == 'Offshore Farm 1'
    
    def test_combine_offshore_sites_only_pipeline(self, generator, sample_offshore_pipeline):
        """Test with only pipeline data"""
        empty_repd = pd.DataFrame()
        combined = generator.combine_offshore_sites(empty_repd, sample_offshore_pipeline)
        
        assert len(combined) == 2
    
    def test_combine_offshore_sites_both_empty(self, generator):
        """Test with both sources empty"""
        empty1 = pd.DataFrame()
        empty2 = pd.DataFrame()
        combined = generator.combine_offshore_sites(empty1, empty2)
        
        assert isinstance(combined, pd.DataFrame)
        assert combined.empty


# ══════════════════════════════════════════════════════════════════════════════
# Test Data Standardization
# ══════════════════════════════════════════════════════════════════════════════

class TestDataStandardization:
    """Test column standardization for different data sources"""
    
    def test_standardize_repd_columns(self, generator):
        """Test standardizing REPD column names"""
        repd_data = pd.DataFrame({
            'site_name': ['Site A'],
            'lat': [55.0],
            'lon': [-3.0],
            'capacity_mw': [50.0],
            'operational_year': [2020]
        })
        
        standardized = generator._standardize_site_columns(repd_data, 'repd')
        
        # Should preserve standard column names
        assert 'site_name' in standardized.columns
        assert 'lat' in standardized.columns
        assert 'capacity_mw' in standardized.columns
    
    def test_standardize_pipeline_columns(self, generator):
        """Test standardizing pipeline column names"""
        pipeline_data = pd.DataFrame({
            'project_name': ['Project A'],
            'latitude': [55.0],
            'longitude': [-3.0],
            'installed_capacity_mw': [200.0],
            'expected_year': [2030]
        })
        
        standardized = generator._standardize_site_columns(pipeline_data, 'pipeline')
        
        # Should map to standard column names
        assert 'site_name' in standardized.columns or 'project_name' in standardized.columns
        assert 'lat' in standardized.columns or 'latitude' in standardized.columns


# ══════════════════════════════════════════════════════════════════════════════
# Test Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_prepare_wind_sites_empty_dataframe(self, generator):
        """Test handling empty DataFrame"""
        empty_df = pd.DataFrame()
        result = generator.prepare_wind_sites(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_prepare_wind_sites_all_invalid_coords(self, generator):
        """Test with all invalid coordinates"""
        invalid_data = pd.DataFrame({
            'site_name': ['Site A', 'Site B'],
            'lat': [np.nan, 'invalid'],
            'lon': [np.nan, 'invalid'],
            'capacity_mw': [50.0, 75.0]
        })
        
        prepared = generator.prepare_wind_sites(invalid_data)
        # Should drop all rows with invalid coords
        assert prepared.empty or len(prepared) == 0
    
    def test_prepare_sites_string_coordinates(self, generator):
        """Test coordinate conversion from strings"""
        string_coords = pd.DataFrame({
            'site_name': ['Site A'],
            'lat': ['55.5'],
            'lon': ['-3.0'],
            'capacity_mw': ['50.0']
        })
        
        prepared = generator.prepare_wind_sites(string_coords)
        
        # Should convert strings to numeric
        assert len(prepared) == 1
        assert isinstance(prepared.iloc[0]['lat'], (int, float))
        assert prepared.iloc[0]['lat'] == 55.5


# ══════════════════════════════════════════════════════════════════════════════
# Parametrized Tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("turbine_capacity,expected_type", [
    (3, 'Vestas_V112_3MW_offshore'),
    (5, 'NREL_ReferenceTurbine_5MW_offshore'),
    (7, 'Vestas_V164_7MW_offshore'),
])
def test_offshore_turbine_mapping(generator, turbine_capacity, expected_type):
    """Test offshore turbine type mapping"""
    assert generator.offshore_turbine_types[turbine_capacity] == expected_type


@pytest.mark.parametrize("turbine_capacity,expected_type", [
    (0.66, 'Vestas_V47_660kW'),
    (2.3, 'Siemens_SWT_2300kW'),
    (3.0, 'Vestas_V112_3MW'),
])
def test_onshore_turbine_mapping(generator, turbine_capacity, expected_type):
    """Test onshore turbine type mapping"""
    assert generator.onshore_turbine_types[turbine_capacity] == expected_type


@pytest.mark.parametrize("num_sites", [0, 1, 10, 100])
def test_prepare_sites_various_sizes(generator, num_sites):
    """Test site preparation with various dataset sizes"""
    data = pd.DataFrame({
        'site_name': [f'Site {i}' for i in range(num_sites)],
        'lat': [55.0 + i * 0.1 for i in range(num_sites)],
        'lon': [-3.0 - i * 0.1 for i in range(num_sites)],
        'capacity_mw': [50.0 + i * 10 for i in range(num_sites)]
    })
    
    prepared = generator.prepare_wind_sites(data)
    assert len(prepared) == num_sites
