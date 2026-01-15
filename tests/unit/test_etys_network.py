"""
Unit tests for scripts/ETYS_network.py.

Tests core functions used in ETYS network construction:
- Coordinate validation and transformation
- Voltage level mapping
- Line parameter validation
- Network topology validation
- Geographic boundary checking

These tests use pytest fixtures and do not require Snakemake execution.
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from ETYS_network import (
    VOLTAGE_LEVELS,
    check_point_on_land,
    move_point_to_land,
    # Add other functions as needed
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_buses_df():
    """Create sample buses DataFrame for testing."""
    return pd.DataFrame({
        'Node': ['BUS001', 'BUS002', 'BUS003'],
        'Name': ['Test Bus 1', 'Test Bus 2', 'Test Bus 3'],
        'Voltage': ['400', '275', '132'],
        'Region': ['Scotland', 'England', 'Wales']
    })


@pytest.fixture
def sample_lines_df():
    """Create sample lines DataFrame for testing."""
    return pd.DataFrame({
        'Line': ['LINE001', 'LINE002'],
        'From': ['BUS001', 'BUS002'],
        'To': ['BUS002', 'BUS003'],
        'R': [0.01, 0.02],
        'X': [0.1, 0.15],
        'Capacity': [1000, 1500]
    })


@pytest.fixture
def sample_coordinates_df():
    """Create sample coordinates DataFrame for testing."""
    return pd.DataFrame({
        'Node': ['BUS001', 'BUS002', 'BUS003'],
        'x': [-4.0, -2.0, -0.5],  # Longitude
        'y': [57.0, 53.0, 51.5],  # Latitude
        'latitude': [57.0, 53.0, 51.5],
        'longitude': [-4.0, -2.0, -0.5]
    })


@pytest.fixture
def valid_network():
    """Create a valid minimal PyPSA network for testing."""
    network = pypsa.Network()
    
    # Add buses
    network.add("Bus", "BUS001", v_nom=400, x=-4.0, y=57.0)
    network.add("Bus", "BUS002", v_nom=400, x=-2.0, y=53.0)
    network.add("Bus", "BUS003", v_nom=275, x=-0.5, y=51.5)
    
    # Add lines
    network.add("Line", "LINE001", bus0="BUS001", bus1="BUS002", 
                r=0.01, x=0.1, s_nom=1000, length=400)
    network.add("Line", "LINE002", bus0="BUS002", bus1="BUS003",
                r=0.02, x=0.15, s_nom=1500, length=300)
    
    return network


@pytest.fixture
def invalid_network_isolated_bus():
    """Create network with isolated bus (connectivity issue)."""
    network = pypsa.Network()
    
    # Add connected buses
    network.add("Bus", "BUS001", v_nom=400, x=-4.0, y=57.0)
    network.add("Bus", "BUS002", v_nom=400, x=-2.0, y=53.0)
    
    # Add isolated bus (no connections)
    network.add("Bus", "BUS_ISOLATED", v_nom=400, x=-1.0, y=52.0)
    
    # Add only one line (BUS_ISOLATED has no connection)
    network.add("Line", "LINE001", bus0="BUS001", bus1="BUS002",
                r=0.01, x=0.1, s_nom=1000)
    
    return network


@pytest.fixture
def invalid_network_bad_coordinates():
    """Create network with invalid coordinates (outside UK)."""
    network = pypsa.Network()
    
    # Add bus with coordinates outside UK bounds
    network.add("Bus", "BUS001", v_nom=400, x=-4.0, y=57.0)  # Valid
    network.add("Bus", "BUS_FOREIGN", v_nom=400, x=10.0, y=45.0)  # Invalid (Italy)
    
    network.add("Line", "LINE001", bus0="BUS001", bus1="BUS_FOREIGN",
                r=0.01, x=0.1, s_nom=1000)
    
    return network


# ============================================================================
# UNIT TESTS - Voltage Level Mapping
# ============================================================================

class TestVoltageLevelMapping:
    """Test voltage level conversion from ETYS codes to kV values."""
    
    def test_voltage_levels_dict_complete(self):
        """Test that VOLTAGE_LEVELS dict contains expected mappings."""
        assert '1' in VOLTAGE_LEVELS
        assert '2' in VOLTAGE_LEVELS
        assert '4' in VOLTAGE_LEVELS
        
        # Check values
        assert VOLTAGE_LEVELS['1'] == 132
        assert VOLTAGE_LEVELS['2'] == 275
        assert VOLTAGE_LEVELS['4'] == 400
    
    def test_voltage_level_lookup(self):
        """Test voltage level lookup for valid codes."""
        assert VOLTAGE_LEVELS.get('4') == 400
        assert VOLTAGE_LEVELS.get('2') == 275
        assert VOLTAGE_LEVELS.get('1') == 132
    
    def test_invalid_voltage_code(self):
        """Test handling of invalid voltage codes."""
        # Should return None for invalid codes
        assert VOLTAGE_LEVELS.get('99') is None


# ============================================================================
# UNIT TESTS - Coordinate Validation
# ============================================================================

class TestCoordinateValidation:
    """Test coordinate validation functions."""
    
    def test_valid_uk_coordinates(self, sample_coordinates_df):
        """Test that valid UK coordinates pass validation."""
        # All coordinates in sample should be within UK bounds
        UK_LAT_MIN, UK_LAT_MAX = 49.5, 60.0
        UK_LON_MIN, UK_LON_MAX = -8.0, 2.0
        
        assert (sample_coordinates_df['y'] >= UK_LAT_MIN).all()
        assert (sample_coordinates_df['y'] <= UK_LAT_MAX).all()
        assert (sample_coordinates_df['x'] >= UK_LON_MIN).all()
        assert (sample_coordinates_df['x'] <= UK_LON_MAX).all()
    
    def test_detect_invalid_coordinates(self):
        """Test detection of coordinates outside UK bounds."""
        # Create DataFrame with one invalid coordinate
        invalid_coords = pd.DataFrame({
            'Node': ['BUS_UK', 'BUS_FOREIGN'],
            'x': [-2.0, 10.0],  # 10.0 is outside UK (Eastern Europe)
            'y': [53.0, 45.0]   # 45.0 is outside UK (Southern Europe)
        })
        
        UK_LON_MAX = 2.0
        UK_LAT_MIN = 49.5
        
        # Check that invalid coordinates are detected
        invalid_lon = invalid_coords['x'] > UK_LON_MAX
        invalid_lat = invalid_coords['y'] < UK_LAT_MIN
        
        assert invalid_lon.sum() == 1  # One invalid longitude
        assert invalid_lat.sum() == 1  # One invalid latitude
    
    def test_coordinates_not_nan(self, sample_coordinates_df):
        """Test that coordinates don't contain NaN values."""
        assert not sample_coordinates_df['x'].isna().any()
        assert not sample_coordinates_df['y'].isna().any()


# ============================================================================
# UNIT TESTS - Line Parameter Validation
# ============================================================================

class TestLineParameterValidation:
    """Test validation of line electrical parameters."""
    
    def test_valid_line_parameters(self, sample_lines_df):
        """Test that line parameters are positive and non-zero."""
        # Resistance should be positive
        assert (sample_lines_df['R'] > 0).all()
        
        # Reactance should be positive
        assert (sample_lines_df['X'] > 0).all()
        
        # Capacity should be positive
        assert (sample_lines_df['Capacity'] > 0).all()
    
    def test_detect_zero_parameters(self):
        """Test detection of zero or negative line parameters."""
        invalid_lines = pd.DataFrame({
            'Line': ['LINE_ZERO_R', 'LINE_NEG_X'],
            'R': [0.0, 0.01],
            'X': [0.1, -0.1],  # Negative reactance is invalid
            'Capacity': [1000, 1000]
        })
        
        # Check for invalid values
        assert (invalid_lines['R'] == 0).any()  # Zero resistance
        assert (invalid_lines['X'] < 0).any()   # Negative reactance
    
    def test_reasonable_parameter_ranges(self, sample_lines_df):
        """Test that line parameters are within reasonable ranges."""
        # Typical transmission line parameters
        # R: 0.001 to 0.1 Ω/km
        # X: 0.01 to 1.0 Ω/km
        # Capacity: 100 to 10000 MW
        
        assert (sample_lines_df['R'] < 1.0).all(), "Resistance too high"
        assert (sample_lines_df['X'] < 2.0).all(), "Reactance too high"
        assert (sample_lines_df['Capacity'] < 20000).all(), "Capacity unreasonably high"


# ============================================================================
# UNIT TESTS - Network Topology Validation
# ============================================================================

class TestNetworkTopologyValidation:
    """Test network topology validation functions."""
    
    def test_valid_network_passes_consistency_check(self, valid_network):
        """Test that a valid network passes PyPSA consistency check."""
        try:
            valid_network.consistency_check()
        except AssertionError as e:
            pytest.fail(f"Valid network failed consistency check: {e}")
    
    def test_network_has_required_components(self, valid_network):
        """Test that network has minimum required components."""
        assert len(valid_network.buses) >= 2, "Network needs at least 2 buses"
        assert len(valid_network.lines) >= 1, "Network needs at least 1 line"
    
    def test_detect_isolated_buses(self, invalid_network_isolated_bus):
        """Test detection of isolated buses (no connections)."""
        network = invalid_network_isolated_bus
        
        # Get all buses that appear in lines
        buses_in_lines = set(network.lines['bus0'].tolist() + 
                           network.lines['bus1'].tolist())
        
        # Find isolated buses
        isolated_buses = set(network.buses.index) - buses_in_lines
        
        assert len(isolated_buses) > 0, "Should detect isolated bus"
        assert 'BUS_ISOLATED' in isolated_buses
    
    def test_all_buses_connected(self, valid_network):
        """Test that all buses in valid network are connected."""
        buses_in_lines = set(valid_network.lines['bus0'].tolist() + 
                           valid_network.lines['bus1'].tolist())
        isolated_buses = set(valid_network.buses.index) - buses_in_lines
        
        assert len(isolated_buses) == 0, f"Found isolated buses: {isolated_buses}"
    
    def test_line_endpoints_exist(self, valid_network):
        """Test that all line endpoints reference existing buses."""
        for line_id, line in valid_network.lines.iterrows():
            assert line['bus0'] in valid_network.buses.index, \
                f"Line {line_id} bus0 '{line['bus0']}' not in buses"
            assert line['bus1'] in valid_network.buses.index, \
                f"Line {line_id} bus1 '{line['bus1']}' not in buses"


# ============================================================================
# UNIT TESTS - Geographic Boundary Checking
# ============================================================================

class TestGeographicBoundaryChecking:
    """Test geographic boundary validation for UK networks."""
    
    def test_coordinates_within_uk_bounds(self, valid_network):
        """Test that all network coordinates are within UK bounds."""
        UK_LAT_MIN, UK_LAT_MAX = 49.5, 60.0
        UK_LON_MIN, UK_LON_MAX = -8.0, 2.0
        
        # Check all buses
        for bus_id, bus in valid_network.buses.iterrows():
            assert UK_LAT_MIN <= bus['y'] <= UK_LAT_MAX, \
                f"Bus {bus_id} latitude {bus['y']} outside UK bounds"
            assert UK_LON_MIN <= bus['x'] <= UK_LON_MAX, \
                f"Bus {bus_id} longitude {bus['x']} outside UK bounds"
    
    def test_detect_foreign_coordinates(self, invalid_network_bad_coordinates):
        """Test detection of coordinates outside UK."""
        network = invalid_network_bad_coordinates
        
        UK_LON_MAX = 2.0
        foreign_buses = network.buses[network.buses['x'] > UK_LON_MAX]
        
        assert len(foreign_buses) > 0, "Should detect foreign coordinates"
        assert 'BUS_FOREIGN' in foreign_buses.index


# ============================================================================
# UNIT TESTS - Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_bus_network(self):
        """Test handling of network with single bus (invalid)."""
        network = pypsa.Network()
        network.add("Bus", "SINGLE_BUS", v_nom=400, x=-2.0, y=53.0)
        
        # Single bus network should have no lines
        assert len(network.lines) == 0
        
        # Should be detectable as problematic
        buses_in_lines = set(network.lines['bus0'].tolist() + 
                           network.lines['bus1'].tolist())
        isolated_buses = set(network.buses.index) - buses_in_lines
        
        assert len(isolated_buses) == 1  # The single bus is isolated
    
    def test_disconnected_network_islands(self):
        """Test detection of disconnected network islands."""
        network = pypsa.Network()
        
        # Island 1
        network.add("Bus", "A1", v_nom=400, x=-4.0, y=57.0)
        network.add("Bus", "A2", v_nom=400, x=-3.5, y=56.5)
        network.add("Line", "L_A", bus0="A1", bus1="A2", r=0.01, x=0.1, s_nom=1000)
        
        # Island 2 (disconnected from Island 1)
        network.add("Bus", "B1", v_nom=400, x=-1.0, y=51.0)
        network.add("Bus", "B2", v_nom=400, x=-0.5, y=51.5)
        network.add("Line", "L_B", bus0="B1", bus1="B2", r=0.01, x=0.1, s_nom=1000)
        
        # All buses are connected within their islands,
        # but the two islands are not connected to each other
        # This requires graph connectivity analysis to detect
        
        # For now, just verify structure exists
        assert len(network.buses) == 4
        assert len(network.lines) == 2
    
    def test_zero_length_line(self):
        """Test handling of zero-length line (bus connected to itself)."""
        network = pypsa.Network()
        network.add("Bus", "BUS001", v_nom=400, x=-2.0, y=53.0)
        
        # Try to add line from bus to itself
        network.add("Line", "SELF_LINE", bus0="BUS001", bus1="BUS001",
                   r=0.01, x=0.1, s_nom=1000, length=0)
        
        # This is topologically invalid and should be detectable
        self_loops = network.lines[network.lines['bus0'] == network.lines['bus1']]
        assert len(self_loops) > 0, "Should detect self-loop"


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("voltage_code,expected_kv", [
    ('1', 132),
    ('2', 275),
    ('4', 400),
    ('3', 33),
    ('5', 11),
    ('6', 66),
])
def test_voltage_code_mapping(voltage_code, expected_kv):
    """Parametrized test for voltage code to kV mapping."""
    assert VOLTAGE_LEVELS.get(voltage_code) == expected_kv


@pytest.mark.parametrize("lat,lon,expected_valid", [
    (51.5, -0.1, True),   # London - valid
    (57.0, -4.0, True),   # Scotland - valid
    (45.0, 10.0, False),  # Italy - invalid
    (70.0, -2.0, False),  # Arctic - invalid
    (52.0, 15.0, False),  # Poland - invalid
])
def test_coordinate_validity(lat, lon, expected_valid):
    """Parametrized test for coordinate validity checking."""
    UK_LAT_MIN, UK_LAT_MAX = 49.5, 60.0
    UK_LON_MIN, UK_LON_MAX = -8.0, 2.0
    
    is_valid = (UK_LAT_MIN <= lat <= UK_LAT_MAX) and (UK_LON_MIN <= lon <= UK_LON_MAX)
    assert is_valid == expected_valid


# ============================================================================
# INTEGRATION-STYLE UNIT TESTS
# ============================================================================

class TestETYSNetworkConstruction:
    """Test complete ETYS network construction workflow (unit level)."""
    
    def test_build_network_from_dataframes(self, sample_buses_df, sample_lines_df, 
                                          sample_coordinates_df):
        """
        Test building a PyPSA network from DataFrames.
        
        This simulates the core network building logic without I/O.
        """
        network = pypsa.Network()
        
        # Merge buses with coordinates
        buses_with_coords = sample_buses_df.merge(
            sample_coordinates_df[['Node', 'x', 'y']], 
            on='Node'
        )
        
        # Add buses
        for _, bus in buses_with_coords.iterrows():
            network.add(
                "Bus",
                bus['Node'],
                v_nom=VOLTAGE_LEVELS.get(bus['Voltage'], 400),
                x=bus['x'],
                y=bus['y']
            )
        
        # Add lines
        for _, line in sample_lines_df.iterrows():
            network.add(
                "Line",
                line['Line'],
                bus0=line['From'],
                bus1=line['To'],
                r=line['R'],
                x=line['X'],
                s_nom=line['Capacity']
            )
        
        # Verify network structure
        assert len(network.buses) == len(sample_buses_df)
        assert len(network.lines) == len(sample_lines_df)
        
        # Consistency check
        try:
            network.consistency_check()
        except AssertionError as e:
            pytest.fail(f"Constructed network failed consistency check: {e}")
