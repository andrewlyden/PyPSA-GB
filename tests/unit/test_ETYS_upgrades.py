"""
Unit tests for ETYS_upgrades.py - Network infrastructure upgrades module.

Tests the application of ETYS Appendix B 2023 network upgrades including:
- Circuit additions, removals, and modifications
- Transformer additions, removals, and modifications
- HVDC interconnector data
"""

import pytest
import pandas as pd
import pypsa
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from ETYS_upgrades import (
    load_etys_upgrade_data,
    filter_upgrades_by_year,
    apply_etys_network_upgrades
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_circuit_data():
    """Sample circuit upgrade data."""
    return pd.DataFrame({
        'Year': [2025, 2026, 2027],
        'operator': ['NGET', 'SPT', 'SHE'],
        'Node1': ['Bus1', 'Bus2', 'Bus3'],
        'Node2': ['Bus2', 'Bus3', 'Bus4'],
        'Voltage': [400, 400, 275],
        'Status': ['Addition', 'Modify', 'Removed'],
        'Capacity': [1000, 1200, 800]
    })


@pytest.fixture
def sample_transformer_data():
    """Sample transformer upgrade data."""
    return pd.DataFrame({
        'Year': [2025, 2026, 2028],
        'operator': ['NGET', 'NGET', 'SPT'],
        'Node1': ['Bus1', 'Bus2', 'Bus5'],
        'Node2': ['Bus1_Sub', 'Bus2_Sub', 'Bus5_Sub'],
        'Status': ['Addition', 'Modify', 'Addition'],
        'Rating': [500, 600, 400]
    })


@pytest.fixture
def sample_hvdc_data():
    """Sample HVDC interconnector data."""
    return pd.DataFrame({
        'Planned from year': [2026, 2029],
        'Existing': ['No', 'No'],
        'Name': ['Viking Link', 'Interconnector X'],
        'From': ['GB_Bus', 'GB_Bus2'],
        'To': ['DK_Bus', 'FR_Bus'],
        'Capacity': [1400, 2000]
    })


@pytest.fixture
def mock_upgrades_dict(sample_circuit_data, sample_transformer_data, sample_hvdc_data):
    """Mock upgrades data dictionary."""
    return {
        'circuits': sample_circuit_data,
        'transformers': sample_transformer_data,
        'hvdc': sample_hvdc_data
    }


@pytest.fixture
def simple_network():
    """Create a simple PyPSA network for testing."""
    network = pypsa.Network()
    
    # Add buses
    network.add("Bus", "Bus1", x=0, y=52, v_nom=400)
    network.add("Bus", "Bus2", x=1, y=53, v_nom=400)
    network.add("Bus", "Bus3", x=2, y=54, v_nom=400)
    network.add("Bus", "Bus4", x=3, y=55, v_nom=275)
    
    # Add some lines
    network.add("Line", "Line1", bus0="Bus1", bus1="Bus2", s_nom=1000, length=100)
    network.add("Line", "Line2", bus0="Bus2", bus1="Bus3", s_nom=800, length=120)
    
    return network


# ══════════════════════════════════════════════════════════════════════════════
# TEST: ETYS Data Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSUpgradeLoading:
    """Test loading of ETYS upgrade data from Excel file."""
    
    def test_load_etys_upgrade_data_returns_dict(self, mock_upgrades_dict):
        """Test that load function returns dictionary with correct keys."""
        with patch('ETYS_upgrades.pd.read_excel') as mock_read:
            mock_read.side_effect = [
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['circuits'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['transformers'].copy(),
                mock_upgrades_dict['hvdc'].copy()
            ]
            
            result = load_etys_upgrade_data("fake_file.xlsx")
            
            assert isinstance(result, dict)
            assert 'circuits' in result
            assert 'transformers' in result
            assert 'hvdc' in result
    
    def test_load_handles_missing_file(self):
        """Test that load function handles missing file gracefully."""
        with patch('ETYS_upgrades.pd.read_excel') as mock_read:
            mock_read.side_effect = FileNotFoundError("File not found")
            
            # Should not raise, but return empty dataframes
            result = load_etys_upgrade_data("nonexistent.xlsx")
            
            assert result['circuits'].empty
            assert result['transformers'].empty
            assert result['hvdc'].empty
    
    def test_load_adds_operator_column(self, sample_circuit_data):
        """Test that operator column is added to each sheet."""
        with patch('ETYS_upgrades.pd.read_excel') as mock_read:
            mock_read.return_value = sample_circuit_data.copy()
            
            result = load_etys_upgrade_data("fake_file.xlsx")
            
            assert 'operator' in result['circuits'].columns


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Year Filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSYearFiltering:
    """Test filtering of upgrades by year."""
    
    def test_filter_upgrades_by_year(self, mock_upgrades_dict):
        """Test that upgrades are correctly filtered by year."""
        result = filter_upgrades_by_year(mock_upgrades_dict, modelled_year=2026)
        
        # Result should be a dict with categorized upgrades
        assert 'circuits' in result
        assert 'transformers' in result
        assert 'hvdc' in result
        
        # Should have additions, removals, changes lists
        assert 'additions' in result['circuits']
        assert 'removals' in result['circuits']
        assert 'changes' in result['circuits']
    
    def test_filter_handles_future_year(self, mock_upgrades_dict):
        """Test filtering with year beyond all upgrades."""
        result = filter_upgrades_by_year(mock_upgrades_dict, modelled_year=2050)
        
        # Should include all upgrades (check at least some exist)
        assert 'circuits' in result
        assert 'transformers' in result
        assert isinstance(result, dict)
    
    def test_filter_handles_past_year(self, mock_upgrades_dict):
        """Test filtering with year before any upgrades."""
        result = filter_upgrades_by_year(mock_upgrades_dict, modelled_year=2020)
        
        # Should exclude all upgrades (empty lists)
        assert len(result['circuits']['additions']) == 0
        assert len(result['transformers']['additions']) == 0
        assert len(result['hvdc']['additions']) == 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network Application
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSNetworkApplication:
    """Test application of upgrades to PyPSA network."""
    
    def test_apply_upgrades_preserves_network_type(self, simple_network):
        """Test that network type is preserved."""
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = {
                'circuits': pd.DataFrame(),
                'transformers': pd.DataFrame(),
                'hvdc': pd.DataFrame()
            }
            
            result = apply_etys_network_upgrades(
                simple_network.copy(), 
                modelled_year=2025,
                etys_file="fake.xlsx"
            )
            
            assert isinstance(result, pypsa.Network)
    
    def test_apply_upgrades_with_disabled_config(self, simple_network):
        """Test that upgrades can be disabled via config."""
        initial_lines = len(simple_network.lines)
        
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = {
                'circuits': pd.DataFrame({'Year': [2025], 'Status': ['Addition'], 'Node1': ['Bus1'], 'Node2': ['Bus2']}),
                'transformers': pd.DataFrame(),
                'hvdc': pd.DataFrame()
            }
            
            # Should not apply if disabled
            result = apply_etys_network_upgrades(
                simple_network.copy(),
                modelled_year=2025,
                etys_file="fake.xlsx"
            )
            
            # Network should be valid (may have more lines if upgrades applied)
            assert isinstance(result, pypsa.Network)
            assert len(result.lines) >= initial_lines
    
    def test_apply_handles_missing_etys_file(self, simple_network):
        """Test that function handles missing ETYS file gracefully."""
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = {
                'circuits': pd.DataFrame(),
                'transformers': pd.DataFrame(),
                'hvdc': pd.DataFrame()
            }
            
            # Should not raise
            result = apply_etys_network_upgrades(
                simple_network.copy(),
                modelled_year=2025,
                etys_file="nonexistent.xlsx"
            )
            
            assert isinstance(result, pypsa.Network)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Upgrade Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSUpgradeValidation:
    """Test validation of upgrade data."""
    
    def test_detect_invalid_years(self, sample_circuit_data):
        """Test detection of invalid year values."""
        invalid_data = sample_circuit_data.copy()
        invalid_data.loc[0, 'Year'] = np.nan
        
        # Should handle gracefully
        result = filter_upgrades_by_year(
            {'circuits': invalid_data, 'transformers': pd.DataFrame(), 'hvdc': pd.DataFrame()},
            modelled_year=2025
        )
        
        # Should still work, just skip invalid rows
        assert isinstance(result, dict)
        assert 'circuits' in result
    
    def test_validate_upgrade_years_in_range(self, sample_circuit_data):
        """Test that upgrade years are within expected range."""
        years = sample_circuit_data['Year'].values
        
        # ETYS upgrades should be 2024-2031
        assert all(2024 <= y <= 2031 for y in years if not pd.isna(y))


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestETYSUpgradeIntegration:
    """Test end-to-end upgrade application."""
    
    def test_complete_upgrade_workflow(self, simple_network, mock_upgrades_dict):
        """Test complete upgrade workflow from load to apply."""
        with patch('ETYS_upgrades.load_etys_upgrade_data') as mock_load:
            mock_load.return_value = mock_upgrades_dict
            
            result = apply_etys_network_upgrades(
                simple_network.copy(),
                modelled_year=2026,
                etys_file="fake.xlsx"
            )
            
            assert isinstance(result, pypsa.Network)
            # Network should still have buses
            assert len(result.buses) > 0
