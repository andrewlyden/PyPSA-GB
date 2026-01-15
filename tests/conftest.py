"""
PyPSA-GB Core Test Suite Configuration

This conftest.py defines shared fixtures and markers for the simplified testing regime.
Tests are organized into three tiers:
  - Smoke tests: Quick sanity checks (~30 seconds)
  - Unit tests: Individual function tests (~2 minutes)
  - Integration tests: Full pipeline validation (~10 minutes)
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add scripts and project root to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Also add all first-level subdirectories of scripts to sys.path so tests can
# import modules placed under subpackages directly by module name (e.g.
# "ETYS_upgrades" from scripts/network_build/ETYS_upgrades.py).
for _sub in SCRIPTS_DIR.iterdir():
    if _sub.is_dir():
        sys.path.insert(0, str(_sub))


# ══════════════════════════════════════════════════════════════════════════════
# PYTEST MARKERS
# ══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Quick sanity check tests (<5s each)")
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests requiring multiple components")
    config.addinivalue_line("markers", "slow: Slow tests that may be skipped in quick runs")
    config.addinivalue_line("markers", "requires_data: Tests requiring specific data files")
    config.addinivalue_line("markers", "requires_solver: Tests requiring optimization solver")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def scripts_dir():
    """Return the scripts directory."""
    return SCRIPTS_DIR


@pytest.fixture(scope="function")
def temp_workspace():
    """Create a temporary workspace directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="pypsa_gb_test_")
    workspace = Path(temp_dir)
    
    # Create basic directory structure
    (workspace / "data").mkdir()
    (workspace / "resources").mkdir()
    (workspace / "logs").mkdir()
    (workspace / "config").mkdir()
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def minimal_network():
    """
    Create a minimal valid PyPSA network for testing.
    
    This network has:
    - 3 buses at different voltage levels
    - 2 lines connecting the buses
    - 1 transformer
    - Basic generators and loads
    """
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
    
    # Add buses
    network.add("Bus", "bus_400_1", v_nom=400, x=0, y=0)
    network.add("Bus", "bus_400_2", v_nom=400, x=1, y=1)
    network.add("Bus", "bus_275", v_nom=275, x=0.5, y=0.5)
    
    # Add lines
    network.add("Line", "line_400", bus0="bus_400_1", bus1="bus_400_2",
                r=0.01, x=0.1, s_nom=2000, length=100)
    
    # Add transformer
    network.add("Transformer", "trafo_1", bus0="bus_400_2", bus1="bus_275",
                x=0.1, s_nom=1000)
    
    # Add generators
    network.add("Generator", "gen_wind", bus="bus_400_1", p_nom=500,
                carrier="wind_onshore", marginal_cost=0)
    network.add("Generator", "gen_ccgt", bus="bus_400_2", p_nom=400,
                carrier="CCGT", marginal_cost=50)
    
    # Add load
    network.add("Load", "load_1", bus="bus_275", p_set=300)
    
    return network


@pytest.fixture(scope="function")
def etys_network():
    """
    Create a more realistic ETYS-like network for testing.
    
    This simulates the ETYS network structure with:
    - Multiple buses with OSGB36 coordinates
    - 400/275/132 kV voltage levels
    - Transformers connecting voltage levels
    """
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
    
    # ETYS-style buses with OSGB36 coordinates
    buses = [
        ("LOND1", 400, 530000, 180000),   # London
        ("LOND2", 275, 531000, 181000),   # London 275kV
        ("MANC1", 400, 385000, 398000),   # Manchester
        ("EDIN1", 400, 325000, 673000),   # Edinburgh
        ("BRIS1", 275, 358000, 173000),   # Bristol
    ]
    
    for name, v_nom, x, y in buses:
        network.add("Bus", name, v_nom=v_nom, x=x, y=y)
    
    # Lines
    network.add("Line", "LOND_MANC", bus0="LOND1", bus1="MANC1",
                r=0.005, x=0.05, s_nom=3000, length=300)
    network.add("Line", "MANC_EDIN", bus0="MANC1", bus1="EDIN1",
                r=0.006, x=0.06, s_nom=2500, length=350)
    
    # Transformers
    network.add("Transformer", "LOND_TRAFO", bus0="LOND1", bus1="LOND2",
                x=0.1, s_nom=2000)
    
    # Add metadata
    network.meta = {
        'network_model': 'ETYS',
        'scenario': 'test',
    }
    
    return network


@pytest.fixture(scope="function")
def sample_renewable_sites():
    """Create sample renewable site DataFrame for testing."""
    return pd.DataFrame({
        'site_name': ['Wind Farm 1', 'Wind Farm 2', 'Solar Park 1', 'Solar Park 2'],
        'technology': ['wind_onshore', 'wind_onshore', 'solar_pv', 'solar_pv'],
        'lat': [51.5, 53.5, 52.0, 55.0],
        'lon': [-0.1, -2.2, -1.0, -3.5],
        'capacity_mw': [100, 200, 50, 150],
        'status': ['Operational', 'Operational', 'Operational', 'Under Construction'],
    })


@pytest.fixture(scope="function")
def sample_thermal_generators():
    """Create sample thermal generator DataFrame for testing."""
    return pd.DataFrame({
        'name': ['CCGT Plant 1', 'Nuclear Station', 'Gas Peaker'],
        'technology': ['CCGT', 'nuclear', 'OCGT'],
        'lat': [51.5, 52.2, 53.0],
        'lon': [-0.1, 1.6, -2.0],
        'capacity_mw': [500, 1200, 150],
        'efficiency': [0.55, 0.35, 0.40],
    })


@pytest.fixture(scope="function")
def sample_fes_data():
    """Create sample FES building blocks data for testing."""
    return pd.DataFrame({
        'Technology': ['Wind', 'Wind', 'Solar', 'Nuclear', 'CCGT'],
        'Technology Detail': ['Offshore Wind', 'Onshore Wind', 'Solar PV', 'Nuclear', 'CCGT'],
        'GSP': ['Norwich', 'Direct(NGET)', 'London', 'Hinkley', 'Grain'],
        'FES Pathway': ['Holistic Transition'] * 5,
        '2030': [10000, 5000, 8000, 3200, 15000],
        '2035': [15000, 7000, 12000, 6400, 12000],
        '2040': [20000, 9000, 18000, 9600, 10000],
    })


@pytest.fixture(scope="function") 
def historical_scenario_config():
    """Create historical scenario configuration for testing."""
    return {
        'scenario_id': 'Test_Historical',
        'modelled_year': 2020,
        'renewables_year': 2020,
        'demand_year': 2020,
        'network_model': 'Reduced',
        'timestep_minutes': 60,
        'voll': 6000.0,
        'demand_timeseries': 'ESPENI',
    }


@pytest.fixture(scope="function")
def future_scenario_config():
    """Create future scenario configuration for testing."""
    return {
        'scenario_id': 'Test_Future',
        'modelled_year': 2035,
        'renewables_year': 2019,
        'demand_year': 2035,
        'network_model': 'ETYS',
        'timestep_minutes': 60,
        'voll': 6000.0,
        'FES_year': 2024,
        'FES_scenario': 'Holistic Transition',
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def logger():
    """Create a test logger."""
    import logging
    return logging.getLogger("test")


@pytest.fixture
def mock_snakemake():
    """Create a mock Snakemake object for testing scripts."""
    from unittest.mock import Mock
    
    snakemake = Mock()
    snakemake.input = Mock()
    snakemake.output = Mock()
    snakemake.params = Mock()
    snakemake.wildcards = Mock()
    snakemake.log = Mock()
    snakemake.config = {}
    
    return snakemake
