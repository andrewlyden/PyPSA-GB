"""
Smoke tests for PyPSA-GB

Quick sanity checks that verify basic functionality works.
These should run in under 30 seconds total.

Run with: pytest tests/test_smoke.py -v
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestImports:
    """Test that key modules can be imported."""
    
    def test_import_pypsa(self):
        """Test PyPSA import."""
        import pypsa
        assert hasattr(pypsa, 'Network')
    
    def test_import_scenario_detection(self):
        """Test scenario_detection module import."""
        from scenario_detection import is_historical_scenario
        assert callable(is_historical_scenario)
    
    def test_import_carrier_definitions(self):
        """Test carrier_definitions module import."""
        from carrier_definitions import get_carrier_definitions
        assert callable(get_carrier_definitions)
    
    def test_import_spatial_utils(self):
        """Test spatial_utils module import."""
        from spatial_utils import map_sites_to_buses
        assert callable(map_sites_to_buses)
    
    def test_import_solve_network(self):
        """Test solve_network module import."""
        from solve_network import configure_solver
        assert callable(configure_solver)


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK CREATION SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestNetworkCreation:
    """Test basic network creation."""
    
    def test_create_empty_network(self):
        """Test creating empty PyPSA network."""
        network = pypsa.Network()
        assert network is not None
        assert len(network.buses) == 0
    
    def test_add_bus(self):
        """Test adding a bus."""
        network = pypsa.Network()
        network.add("Bus", "test_bus", v_nom=400)
        assert "test_bus" in network.buses.index
    
    def test_add_generator(self):
        """Test adding a generator."""
        network = pypsa.Network()
        network.add("Bus", "bus1", v_nom=400)
        network.add("Generator", "gen1", bus="bus1", p_nom=100)
        assert "gen1" in network.generators.index
    
    def test_set_snapshots(self):
        """Test setting snapshots."""
        network = pypsa.Network()
        snapshots = pd.date_range("2020-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)
        assert len(network.snapshots) == 24


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO DETECTION SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestScenarioDetection:
    """Test scenario detection basics."""
    
    def test_historical_detection(self):
        """Test historical scenario is detected."""
        from scenario_detection import is_historical_scenario
        
        scenario = {'modelled_year': 2020}
        assert is_historical_scenario(scenario) is True
    
    def test_future_detection(self):
        """Test future scenario is detected."""
        from scenario_detection import is_historical_scenario
        
        scenario = {'modelled_year': 2035}
        assert is_historical_scenario(scenario) is False
    
    def test_boundary_year_2024(self):
        """Test 2024 is historical."""
        from scenario_detection import is_historical_scenario
        
        scenario = {'modelled_year': 2024}
        assert is_historical_scenario(scenario) is True
    
    def test_boundary_year_2025(self):
        """Test 2025 is future."""
        from scenario_detection import is_historical_scenario
        
        scenario = {'modelled_year': 2025}
        assert is_historical_scenario(scenario) is False


# ══════════════════════════════════════════════════════════════════════════════
# CARRIER DEFINITIONS SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestCarrierDefinitions:
    """Test carrier definitions basics."""
    
    def test_get_definitions(self):
        """Test getting carrier definitions."""
        from carrier_definitions import get_carrier_definitions
        
        carriers = get_carrier_definitions()
        assert isinstance(carriers, pd.DataFrame)
        assert len(carriers) > 0
    
    def test_has_wind(self):
        """Test wind carriers exist."""
        from carrier_definitions import get_carrier_definitions
        
        carriers = get_carrier_definitions()
        index_str = ' '.join(carriers.index.astype(str))
        
        assert 'wind' in index_str.lower()
    
    def test_has_colors(self):
        """Test carriers have colors."""
        from carrier_definitions import get_carrier_definitions
        
        carriers = get_carrier_definitions()
        assert 'color' in carriers.columns


# ══════════════════════════════════════════════════════════════════════════════
# DATA FILE SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestDataFiles:
    """Test that key data files exist."""
    
    def test_config_exists(self):
        """Test config.yaml exists."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        assert config_path.exists()
    
    def test_scenarios_yaml_exists(self):
        """Test scenarios.yaml exists."""
        scenarios_path = Path(__file__).parent.parent / "config" / "scenarios.yaml"
        assert scenarios_path.exists()
    
    def test_defaults_yaml_exists(self):
        """Test defaults.yaml exists."""
        defaults_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
        assert defaults_path.exists()
    
    def test_snakefile_exists(self):
        """Test Snakefile exists."""
        snakefile = Path(__file__).parent.parent / "Snakefile"
        assert snakefile.exists()


# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE SYSTEM SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
class TestCoordinateSystems:
    """Test coordinate system handling."""
    
    def test_pyproj_available(self):
        """Test pyproj is available for coordinate conversion."""
        try:
            from pyproj import Transformer
            assert True
        except ImportError:
            pytest.skip("pyproj not installed")
    
    def test_wgs84_to_osgb36_conversion(self):
        """Test WGS84 to OSGB36 coordinate conversion."""
        try:
            from pyproj import Transformer
            
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            
            # London coordinates
            wgs84_lon, wgs84_lat = -0.1278, 51.5074
            osgb_x, osgb_y = transformer.transform(wgs84_lon, wgs84_lat)
            
            # Should be in the 500000, 180000 range for London
            assert 500000 < osgb_x < 550000
            assert 150000 < osgb_y < 200000
        except ImportError:
            pytest.skip("pyproj not installed")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.smoke
def test_minimal_network_solves(minimal_network):
    """Test that a minimal network can be solved."""
    network = minimal_network
    
    # Add simple demand profile
    network.loads_t.p_set = pd.DataFrame(
        {"load_1": [300] * 24},
        index=network.snapshots
    )
    
    # Add simple wind profile
    network.generators_t.p_max_pu = pd.DataFrame(
        {"gen_wind": np.random.uniform(0.3, 0.8, 24)},
        index=network.snapshots
    )
    
    try:
        # Try to solve with HiGHS (open source)
        network.optimize(solver_name='highs')
        assert network.objective is not None
    except Exception as e:
        # Solver might not be available
        if 'solver' in str(e).lower() or 'highs' in str(e).lower():
            pytest.skip("HiGHS solver not available")
        raise
