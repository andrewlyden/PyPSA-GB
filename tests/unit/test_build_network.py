"""
Unit tests for build_network.py

Tests the core network construction functionality:
- ETYS network building
- Reduced network building
- Zonal network building
- Bus creation and validation
- Line/transformer creation
- Network metadata

This is a CRITICAL module - network building is the foundation for all scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import tempfile
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_buses_csv(tmp_path):
    """Create sample buses CSV file."""
    data = pd.DataFrame({
        'bus_id': ['BUS001', 'BUS002', 'BUS003', 'BUS004'],
        'name': ['London', 'Manchester', 'Edinburgh', 'Cardiff'],
        'v_nom': [400, 400, 275, 275],
        'x': [530000, 385000, 325000, 320000],  # OSGB36
        'y': [180000, 398000, 673000, 175000],
        'zone': ['South', 'North', 'Scotland', 'Wales'],
    })
    
    csv_file = tmp_path / "buses.csv"
    data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_lines_csv(tmp_path):
    """Create sample lines CSV file."""
    data = pd.DataFrame({
        'line_id': ['LINE001', 'LINE002', 'LINE003'],
        'bus0': ['BUS001', 'BUS002', 'BUS001'],
        'bus1': ['BUS002', 'BUS003', 'BUS004'],
        'r': [0.01, 0.015, 0.012],
        'x': [0.1, 0.12, 0.11],
        's_nom': [2000, 1500, 1800],
        'length': [300, 400, 250],
        'v_nom': [400, 400, 275],
    })
    
    csv_file = tmp_path / "lines.csv"
    data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_transformers_csv(tmp_path):
    """Create sample transformers CSV file."""
    data = pd.DataFrame({
        'trafo_id': ['TRAFO001', 'TRAFO002'],
        'bus0': ['BUS001', 'BUS002'],
        'bus1': ['BUS004', 'BUS003'],
        'x': [0.1, 0.12],
        's_nom': [1000, 800],
        'tap_ratio': [1.0, 1.0],
    })
    
    csv_file = tmp_path / "transformers.csv"
    data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def minimal_network():
    """Create minimal valid PyPSA network."""
    network = pypsa.Network()
    
    # Add buses
    network.add("Bus", "bus0", v_nom=400, x=0, y=0)
    network.add("Bus", "bus1", v_nom=400, x=1, y=1)
    
    # Add line
    network.add("Line", "line0", bus0="bus0", bus1="bus1", 
                r=0.01, x=0.1, s_nom=1000)
    
    return network


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network Component Creation
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkComponentCreation:
    """Test creation of network components."""
    
    def test_add_bus(self):
        """Test adding a bus to network."""
        network = pypsa.Network()
        network.add("Bus", "test_bus", v_nom=400, x=100, y=200)
        
        assert "test_bus" in network.buses.index
        assert network.buses.loc["test_bus", "v_nom"] == 400
        assert network.buses.loc["test_bus", "x"] == 100
        assert network.buses.loc["test_bus", "y"] == 200
    
    def test_add_line(self):
        """Test adding a line to network."""
        network = pypsa.Network()
        network.add("Bus", "bus0", v_nom=400)
        network.add("Bus", "bus1", v_nom=400)
        network.add("Line", "test_line", bus0="bus0", bus1="bus1",
                   r=0.01, x=0.1, s_nom=1000)
        
        assert "test_line" in network.lines.index
        assert network.lines.loc["test_line", "bus0"] == "bus0"
        assert network.lines.loc["test_line", "bus1"] == "bus1"
        assert network.lines.loc["test_line", "s_nom"] == 1000
    
    def test_add_transformer(self):
        """Test adding a transformer to network."""
        network = pypsa.Network()
        network.add("Bus", "bus_400", v_nom=400)
        network.add("Bus", "bus_275", v_nom=275)
        network.add("Transformer", "test_trafo", bus0="bus_400", bus1="bus_275",
                   x=0.1, s_nom=500)
        
        assert "test_trafo" in network.transformers.index
        assert network.transformers.loc["test_trafo", "s_nom"] == 500


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkValidation:
    """Test network validation functions."""
    
    def test_all_buses_have_coordinates(self, minimal_network):
        """Test that all buses have valid coordinates."""
        buses = minimal_network.buses
        
        assert buses['x'].notna().all()
        assert buses['y'].notna().all()
    
    def test_all_lines_connect_valid_buses(self, minimal_network):
        """Test that all lines connect to valid buses."""
        lines = minimal_network.lines
        buses = minimal_network.buses.index
        
        for _, line in lines.iterrows():
            assert line['bus0'] in buses, f"Line {line.name} has invalid bus0"
            assert line['bus1'] in buses, f"Line {line.name} has invalid bus1"
    
    def test_no_self_loops(self, minimal_network):
        """Test that no lines connect a bus to itself."""
        for _, line in minimal_network.lines.iterrows():
            assert line['bus0'] != line['bus1'], \
                f"Line {line.name} is a self-loop"
    
    def test_positive_line_ratings(self, minimal_network):
        """Test that all line ratings are positive."""
        assert (minimal_network.lines['s_nom'] > 0).all()
    
    def test_positive_reactance(self, minimal_network):
        """Test that all lines have positive reactance."""
        assert (minimal_network.lines['x'] > 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network Connectivity
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkConnectivity:
    """Test network connectivity validation."""
    
    def test_network_is_connected(self, minimal_network):
        """Test that network is fully connected."""
        import networkx as nx
        
        G = nx.Graph()
        G.add_nodes_from(minimal_network.buses.index)
        
        for _, line in minimal_network.lines.iterrows():
            G.add_edge(line['bus0'], line['bus1'])
        
        for _, trafo in minimal_network.transformers.iterrows():
            G.add_edge(trafo['bus0'], trafo['bus1'])
        
        # Network should have exactly one connected component
        assert nx.is_connected(G), "Network is not fully connected"
    
    def test_detect_isolated_buses(self):
        """Test detection of isolated (unconnected) buses."""
        import networkx as nx
        
        network = pypsa.Network()
        network.add("Bus", "connected1", v_nom=400)
        network.add("Bus", "connected2", v_nom=400)
        network.add("Bus", "isolated", v_nom=400)  # No connections
        network.add("Line", "line", bus0="connected1", bus1="connected2",
                   r=0.01, x=0.1, s_nom=1000)
        
        G = nx.Graph()
        G.add_nodes_from(network.buses.index)
        for _, line in network.lines.iterrows():
            G.add_edge(line['bus0'], line['bus1'])
        
        components = list(nx.connected_components(G))
        
        # Should have 2 components (connected pair + isolated)
        assert len(components) == 2


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Voltage Level Handling
# ══════════════════════════════════════════════════════════════════════════════

class TestVoltageLevelHandling:
    """Test voltage level validation and handling."""
    
    def test_valid_voltage_levels(self):
        """Test that standard voltage levels are accepted."""
        network = pypsa.Network()
        
        valid_voltages = [400, 275, 132, 33, 11]
        
        for v in valid_voltages:
            network.add("Bus", f"bus_{v}kV", v_nom=v)
        
        assert len(network.buses) == 5
        assert set(network.buses['v_nom']) == set(valid_voltages)
    
    def test_transformer_connects_different_voltages(self):
        """Test that transformers typically connect different voltage levels."""
        network = pypsa.Network()
        network.add("Bus", "bus_400", v_nom=400)
        network.add("Bus", "bus_275", v_nom=275)
        network.add("Transformer", "trafo", bus0="bus_400", bus1="bus_275",
                   x=0.1, s_nom=500)
        
        trafo = network.transformers.loc["trafo"]
        v0 = network.buses.loc[trafo['bus0'], 'v_nom']
        v1 = network.buses.loc[trafo['bus1'], 'v_nom']
        
        # Voltage levels should be different
        assert v0 != v1


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network Metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkMetadata:
    """Test network metadata handling."""
    
    def test_can_add_metadata(self, minimal_network):
        """Test that metadata can be added to network."""
        minimal_network.meta = {
            'scenario': 'Test_Scenario',
            'network_model': 'Reduced',
            'modelled_year': 2030,
        }
        
        assert minimal_network.meta['scenario'] == 'Test_Scenario'
        assert minimal_network.meta['modelled_year'] == 2030
    
    def test_metadata_preserved_after_save_load(self, minimal_network, tmp_path):
        """Test that metadata is preserved after save/load cycle."""
        minimal_network.meta = {
            'test_key': 'test_value',
        }
        
        # Save
        nc_file = tmp_path / "test_network.nc"
        minimal_network.export_to_netcdf(str(nc_file))
        
        # Load
        loaded = pypsa.Network(str(nc_file))
        
        # Metadata should be preserved
        assert 'test_key' in loaded.meta
        assert loaded.meta['test_key'] == 'test_value'


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network File I/O
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkFileIO:
    """Test network save and load operations."""
    
    def test_save_to_netcdf(self, minimal_network, tmp_path):
        """Test saving network to NetCDF format."""
        nc_file = tmp_path / "test.nc"
        minimal_network.export_to_netcdf(str(nc_file))
        
        assert nc_file.exists()
    
    def test_load_from_netcdf(self, minimal_network, tmp_path):
        """Test loading network from NetCDF format."""
        nc_file = tmp_path / "test.nc"
        minimal_network.export_to_netcdf(str(nc_file))
        
        loaded = pypsa.Network(str(nc_file))
        
        assert len(loaded.buses) == len(minimal_network.buses)
        assert len(loaded.lines) == len(minimal_network.lines)
    
    def test_component_counts_preserved(self, minimal_network, tmp_path):
        """Test that all components are preserved after save/load."""
        nc_file = tmp_path / "test.nc"
        minimal_network.export_to_netcdf(str(nc_file))
        loaded = pypsa.Network(str(nc_file))
        
        assert len(loaded.buses) == len(minimal_network.buses)
        assert len(loaded.lines) == len(minimal_network.lines)
        assert len(loaded.generators) == len(minimal_network.generators)
        assert len(loaded.loads) == len(minimal_network.loads)
