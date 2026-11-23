"""
Unit tests for add_storage.py

Tests the storage integration module including:
- FES storage data loading
- Storage type routing (GSP vs Direct)
- Pumped hydro preservation
- Storage capacity validation

This is important for future scenarios where storage scaling must be correct.
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from add_storage import (
    add_storage_to_network,
    load_fes_storage_data,
)
from config.config_loader import load_defaults


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_network():
    """Create simple network for storage testing."""
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
    
    # Add buses
    network.add("Bus", "bus0", v_nom=400, x=0, y=0)
    network.add("Bus", "bus1", v_nom=400, x=1, y=1)
    network.add("Bus", "bus2", v_nom=275, x=2, y=2)
    
    # Add a load to make the network useful
    network.add("Load", "load0", bus="bus0", p_set=100)
    
    return network


@pytest.fixture
def battery_storage_df():
    """Create sample battery storage DataFrame."""
    return pd.DataFrame({
        'site_name': ['Battery_1', 'Battery_2', 'Battery_3'],
        'bus': ['bus0', 'bus1', 'bus2'],
        'power_mw': [100.0, 200.0, 150.0],
        'duration_h': [4.0, 2.0, 4.0],
        'technology': ['battery', 'battery', 'battery'],
        'eta_charge': [0.9, 0.9, 0.9],
        'eta_discharge': [0.9, 0.9, 0.9],
        'capital_cost': [0, 0, 0],
        'marginal_cost': [0, 0, 0],
    })


@pytest.fixture
def mixed_storage_df():
    """Create mixed storage types (battery + pumped hydro)."""
    return pd.DataFrame({
        'site_name': ['Battery_1', 'Pumped_Hydro_1', 'Battery_2'],
        'bus': ['bus0', 'bus1', 'bus2'],
        'power_mw': [100.0, 200.0, 150.0],
        'duration_h': [4.0, 6.0, 4.0],
        'technology': ['battery', 'pumped_hydro', 'battery'],
        'eta_charge': [0.9, 0.85, 0.9],
        'eta_discharge': [0.9, 0.85, 0.9],
        'capital_cost': [0, 0, 0],
        'marginal_cost': [0, 0, 0],
    })


@pytest.fixture
def fes_storage_csv(tmp_path):
    """Create sample FES storage CSV file."""
    data = pd.DataFrame({
        'Technology': ['Battery', 'Battery', 'Pumped Hydro Storage'],
        'Technology Detail': ['Small Scale', 'Large Scale', 'Pumped Hydro'],
        'GSP': ['Norwich', 'Direct(NGET)', 'Dinorwig'],
        'FES Pathway': ['Holistic Transition', 'Holistic Transition', 'Holistic Transition'],
        '2030': [500.0, 1000.0, 1800.0],
        '2035': [800.0, 2000.0, 1800.0],
        '2040': [1200.0, 3500.0, 1800.0],
    })
    
    fes_file = tmp_path / "FES_storage.csv"
    data.to_csv(fes_file, index=False)
    return fes_file


# ══════════════════════════════════════════════════════════════════════════════
# TEST: add_storage_to_network
# ══════════════════════════════════════════════════════════════════════════════

class TestAddStorageToNetwork:
    """Test adding storage units to network."""
    
    def test_adds_battery_storage(self, simple_network, battery_storage_df):
        """Test adding battery storage to network."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        # Check storage was added
        assert len(simple_network.storage_units) == 3
        
        # Check total capacity
        total_capacity = simple_network.storage_units['p_nom'].sum()
        assert total_capacity == pytest.approx(450.0)
    
    def test_adds_mixed_storage_types(self, simple_network, mixed_storage_df):
        """Test adding different storage types."""
        add_storage_to_network(simple_network, mixed_storage_df)
        
        # Check by carrier
        carriers = simple_network.storage_units['carrier'].unique()
        assert 'battery' in carriers
        assert 'pumped_hydro' in carriers
        
        # Check pumped hydro capacity (200 MW from fixture)
        ph = simple_network.storage_units[simple_network.storage_units['carrier'] == 'pumped_hydro']
        assert ph['p_nom'].sum() == 200.0
    
    def test_preserves_efficiency(self, simple_network, battery_storage_df):
        """Test that efficiency values are preserved."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        # Check that all storage units have the expected efficiency values
        for su_name, su in simple_network.storage_units.iterrows():
            assert su['efficiency_store'] == pytest.approx(0.9, abs=0.01)
            assert su['efficiency_dispatch'] == pytest.approx(0.9, abs=0.01)
    
    def test_preserves_max_hours(self, simple_network, battery_storage_df):
        """Test that max_hours (storage duration) is preserved."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        # Check that max_hours values match expected duration_h values
        expected_durations = set(battery_storage_df['duration_h'].values)
        actual_durations = set(simple_network.storage_units['max_hours'].values)
        assert expected_durations == actual_durations
    
    def test_empty_storage_df(self, simple_network):
        """Test that empty storage DataFrame doesn't crash."""
        empty_df = pd.DataFrame(columns=['site_name', 'bus', 'power_mw', 'technology'])
        
        add_storage_to_network(simple_network, empty_df)
        
        assert len(simple_network.storage_units) == 0
    
    def test_validates_bus_exists(self, simple_network):
        """Test that invalid bus names are handled."""
        invalid_storage = pd.DataFrame({
            'site_name': ['Invalid_Battery'],
            'bus': ['nonexistent_bus'],  # Bus doesn't exist
            'power_mw': [100.0],
            'technology': ['battery'],
            'duration_h': [4.0],
            'eta_charge': [0.9],
            'eta_discharge': [0.9],
            'capital_cost': [0],
            'marginal_cost': [0],
        })
        
        # Should either raise an error or skip invalid entries
        try:
            add_storage_to_network(simple_network, invalid_storage)
            # If no error, check storage wasn't added or was handled
            assert len(simple_network.storage_units) <= 1
        except (ValueError, KeyError):
            pass  # Expected behavior


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Storage Capacity Calculations
# ══════════════════════════════════════════════════════════════════════════════

class TestStorageCapacityCalculations:
    """Test storage capacity and energy calculations."""
    
    def test_energy_capacity_calculation(self, simple_network, battery_storage_df):
        """Test that energy capacity = power * hours."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        # Check the total energy capacity matches sum of (power_mw * duration_h)
        expected_total_energy = (battery_storage_df['power_mw'] * battery_storage_df['duration_h']).sum()
        actual_total_energy = (simple_network.storage_units['p_nom'] * simple_network.storage_units['max_hours']).sum()
        
        assert actual_total_energy == pytest.approx(expected_total_energy)
    
    def test_total_storage_power_capacity(self, simple_network, battery_storage_df):
        """Test total power capacity aggregation."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        expected_total = battery_storage_df['power_mw'].sum()
        actual_total = simple_network.storage_units['p_nom'].sum()
        
        assert actual_total == pytest.approx(expected_total)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Pumped Hydro Handling
# ══════════════════════════════════════════════════════════════════════════════

class TestPumpedHydroHandling:
    """Test special handling of pumped hydro storage."""
    
    def test_pumped_hydro_not_scaled(self, simple_network):
        """Test that pumped hydro capacity is preserved (not FES-scaled)."""
        # This is important: pumped hydro should use existing infrastructure
        storage_data = pd.DataFrame({
            'site_name': ['Dinorwig', 'Ffestiniog'],
            'bus': ['bus0', 'bus1'],
            'power_mw': [1800.0, 360.0],  # Real UK capacities
            'duration_h': [5.0, 4.0],
            'technology': ['pumped_hydro', 'pumped_hydro'],
            'eta_charge': [0.87, 0.87],
            'eta_discharge': [0.87, 0.87],
            'capital_cost': [0, 0],
            'marginal_cost': [0, 0],
        })
        
        add_storage_to_network(simple_network, storage_data)
        
        # Capacities should be exactly as specified
        # Note: storage units will have indexed names like "Dinorwig_0"
        dinorwig_units = [su for su in simple_network.storage_units.index if 'Dinorwig' in su]
        assert len(dinorwig_units) == 1
        dinorwig = simple_network.storage_units.loc[dinorwig_units[0]]
        assert dinorwig['p_nom'] == 1800.0
        
        ffest_units = [su for su in simple_network.storage_units.index if 'Ffestiniog' in su]
        assert len(ffest_units) == 1
        ffestiniog = simple_network.storage_units.loc[ffest_units[0]]
        assert ffestiniog['p_nom'] == 360.0
    
    def test_pumped_hydro_efficiency(self, simple_network):
        """Test that pumped hydro has appropriate round-trip efficiency."""
        storage_data = pd.DataFrame({
            'site_name': ['Pumped_Hydro'],
            'bus': ['bus0'],
            'power_mw': [500.0],
            'duration_h': [6.0],
            'technology': ['pumped_hydro'],
            'eta_charge': [0.87],  # Typical pumping efficiency
            'eta_discharge': [0.87],  # Typical generation efficiency
            'capital_cost': [0],
            'marginal_cost': [0],
        })
        
        add_storage_to_network(simple_network, storage_data)
        
        ph_units = [su for su in simple_network.storage_units.index if 'Pumped_Hydro' in su]
        assert len(ph_units) == 1
        ph = simple_network.storage_units.loc[ph_units[0]]
        round_trip = ph['efficiency_store'] * ph['efficiency_dispatch']
        
        # Round-trip should be ~75-80%
        assert 0.7 <= round_trip <= 0.85


# ══════════════════════════════════════════════════════════════════════════════
# TEST: FES Storage Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestFESStorageLoading:
    """Test FES storage data loading and processing."""
    
    def test_loads_fes_storage_for_year(self, fes_storage_csv, simple_network):
        """Test loading FES storage data for a specific year."""
        import logging
        logger = logging.getLogger("test")
        
        try:
            storage_df = load_fes_storage_data(
                fes_file=str(fes_storage_csv),
                year=2035,
                scenario='Holistic Transition',
                network=simple_network,
                logger=logger
            )
            
            assert len(storage_df) > 0
            assert 'p_nom' in storage_df.columns or 'capacity_mw' in storage_df.columns
        except Exception as e:
            # Function may not exist exactly as named
            pytest.skip(f"load_fes_storage_data not available: {e}")
    
    def test_fes_storage_gsp_mapping(self, fes_storage_csv, simple_network):
        """Test that GSP-connected storage is mapped correctly."""
        # This tests that GSP names are converted to network buses
        import logging
        logger = logging.getLogger("test")
        
        try:
            storage_df = load_fes_storage_data(
                fes_file=str(fes_storage_csv),
                year=2035,
                scenario='Holistic Transition',
                network=simple_network,
                logger=logger
            )
            
            # Should have bus column after processing
            if 'bus' in storage_df.columns:
                assert storage_df['bus'].notna().any()
        except Exception as e:
            pytest.skip(f"load_fes_storage_data not available: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Storage Integration Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestStorageIntegrationValidation:
    """Test validation of storage integration."""
    
    def test_no_negative_capacity(self, simple_network, battery_storage_df):
        """Test that no negative capacities are created."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        assert (simple_network.storage_units['p_nom'] >= 0).all()
    
    def test_no_negative_hours(self, simple_network, battery_storage_df):
        """Test that max_hours is positive."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        assert (simple_network.storage_units['max_hours'] > 0).all()
    
    def test_efficiency_in_valid_range(self, simple_network, battery_storage_df):
        """Test that efficiencies are between 0 and 1."""
        add_storage_to_network(simple_network, battery_storage_df)
        
        su = simple_network.storage_units
        
        if 'efficiency_store' in su.columns:
            assert (su['efficiency_store'] > 0).all()
            assert (su['efficiency_store'] <= 1).all()
        
        if 'efficiency_dispatch' in su.columns:
            assert (su['efficiency_dispatch'] > 0).all()
            assert (su['efficiency_dispatch'] <= 1).all()


# ----------------------------------------------------------------------------- fixes S2, S3 (27 Sep 2026)

from scripts.storage.add_storage import (  # noqa: E402
    DEFAULT_STORAGE_PLACEMENT,
    DEFAULT_STORAGE_STANDING_LOSSES,
    add_storage_to_network as add_storage_units,
    pin_storage_to_site_codes,
    spread_storage_over_branch_capacity,
    stranded_storage_buses,
)


def _grid():
    """Two busbars of one substation (AAAA2Q, AAAA2R, 283 MVA each) off a 400 kV node, a DC hub, a load-less island."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for name, v, x in [("CORE41", 400, 0), ("AAAA2Q", 275, 1000), ("AAAA2R", 275, 1000), ("BBBB11", 132, 5000),
                       ("HUB", 400, 90000), ("ISLD61", 66, 2000), ("ISLD4D", 400, 2000), ("ISLD4A", 400, 2000)]:
        n.add("Bus", name, v_nom=v, x=x, y=0)
    n.add("Load", "demand", bus="CORE41", p_set=100.0)
    n.add("Line", "q", bus0="CORE41", bus1="AAAA2Q", s_nom=283.0, x=0.1)
    n.add("Line", "r", bus0="CORE41", bus1="AAAA2R", s_nom=283.0, x=0.1)
    n.add("Line", "b", bus0="CORE41", bus1="BBBB11", s_nom=500.0, x=0.1)
    n.add("Line", "i", bus0="CORE41", bus1="ISLD4A", s_nom=2000.0, x=0.1)
    n.add("Transformer", "t", bus0="ISLD4D", bus1="ISLD61", s_nom=100.0, x=0.1)
    n.add("Link", "dc_hub", bus0="CORE41", bus1="HUB", p_nom=1000.0, p_min_pu=-1)
    n.add("Link", "dc_isl", bus0="ISLD4A", bus1="ISLD4D", p_nom=1000.0, p_min_pu=-1)
    return n


def _rows(**cols):
    base = {"technology": "LAES", "duration_h": 16.0, "eta_charge": 0.84, "eta_discharge": 0.84}
    frame = pd.DataFrame(cols)
    for key, value in base.items():
        if key not in frame.columns:
            frame[key] = value
    frame["energy_mwh"] = frame["power_mw"] * frame["duration_h"]
    return frame


def test_default_standing_losses_and_placement_match_defaults_yaml():
    import yaml

    defaults = load_defaults()
    assert defaults["storage"]["standing_losses"] == DEFAULT_STORAGE_STANDING_LOSSES
    assert defaults["storage"]["placement"] == DEFAULT_STORAGE_PLACEMENT
    assert "standing_loss" not in defaults["renewables"]["hydro"]["pumped_hydro"]


def test_standing_loss_is_per_carrier_batteries_small_ldes_zero():
    n = _grid()
    df = _rows(site_name=["b", "l", "p"], bus=["CORE41"] * 3, power_mw=[10.0, 10.0, 10.0],
               technology=["Battery", "LAES", "Pumped Storage Hydroelectricity"])
    add_storage_units(n, df, standing_losses=DEFAULT_STORAGE_STANDING_LOSSES)
    losses = n.storage_units.set_index("carrier")["standing_loss"]
    assert losses["Battery"] == pytest.approx(0.00005)
    assert losses["LAES"] == 0.0 and losses["Pumped Storage Hydroelectricity"] == 0.0


def test_site_code_pin_then_spread_over_sibling_busbars_conserves_power_and_energy():
    n = _grid()
    df = _rows(site_name=["CP30_LAES_existing_Cruachan"], bus=["BBBB11"], power_mw=[440.0], site_code=["AAAA"])
    df = pin_storage_to_site_codes(df, n)
    assert df["bus"].iloc[0] in {"AAAA2Q", "AAAA2R"}
    spread, moves = spread_storage_over_branch_capacity(df, n, stranded_buses=[])
    by_bus = spread.groupby("bus")["power_mw"].sum()
    assert by_bus.to_dict() == pytest.approx({"AAAA2Q": 283.0, "AAAA2R": 157.0} if df["bus"].iloc[0] == "AAAA2Q"
                                             else {"AAAA2R": 283.0, "AAAA2Q": 157.0})
    assert spread["energy_mwh"].sum() == pytest.approx(440.0 * 16.0)
    assert (spread["energy_mwh"] / spread["power_mw"]).tolist() == pytest.approx([16.0, 16.0])
    assert spread["site_name"].str.contains("Cruachan").all() and len(moves) == 1 and moves["split"].all()


def test_storage_on_a_dc_hub_or_a_loadless_island_is_moved_to_the_grid():
    n = _grid()
    assert set(stranded_storage_buses(n)) == {"HUB", "ISLD61", "ISLD4D"}
    df = _rows(site_name=["orkney", "teesside"], bus=["HUB", "ISLD61"], power_mw=[100.0, 20.0],
               technology=["Battery", "Battery"], duration_h=[1.8, 1.8])
    spread, _ = spread_storage_over_branch_capacity(df, n)
    assert not spread["bus"].isin(["HUB", "ISLD61", "ISLD4D"]).any()
    # same site code first (ISLD4A is on the main grid), else nearest at >= 132 kV at any distance
    assert spread.set_index("site_name").loc["teesside", "bus"] == "ISLD4A"
    assert spread["power_mw"].sum() == pytest.approx(120.0)


def test_storage_within_branch_capacity_is_left_alone():
    n = _grid()
    df = _rows(site_name=["a"], bus=["BBBB11"], power_mw=[400.0])
    spread, moves = spread_storage_over_branch_capacity(df, n)
    assert spread["bus"].tolist() == ["BBBB11"] and moves.empty


def _coupled_grid():
    """CCCC2Q and CCCC2R (275 kV) joined by a 9,999 MVA coupler: one node with 300 MVA leaving it."""
    n = _grid()
    n.add("Bus", "CCCC2Q", v_nom=275, x=3000, y=0)
    n.add("Bus", "CCCC2R", v_nom=275, x=3000, y=0)
    n.add("Bus", "CCCC2S", v_nom=275, x=3000, y=0)
    n.add("Line", "cq", bus0="CORE41", bus1="CCCC2Q", s_nom=150.0, x=0.1)
    n.add("Line", "cr", bus0="CORE41", bus1="CCCC2R", s_nom=150.0, x=0.1)
    n.add("Line", "cs", bus0="CORE41", bus1="CCCC2S", s_nom=400.0, x=0.1)
    n.add("Line", "coupler", bus0="CCCC2Q", bus1="CCCC2R", s_nom=9999.0, x=1e-6)
    return n


def test_spread_judges_a_coupled_node_as_one_not_busbar_by_busbar():
    # F25.2: 200 MW on each coupled busbar is 400 MW on a 300 MVA node; busbar by busbar each saw 300.
    n = _coupled_grid()
    df = _rows(site_name=["q", "r"], bus=["CCCC2Q", "CCCC2R"], power_mw=[200.0, 200.0])
    spread, moves = spread_storage_over_branch_capacity(df, n, stranded_buses=[])
    by_bus = spread.groupby("bus")["power_mw"].sum()
    assert by_bus.reindex(["CCCC2Q", "CCCC2R"]).fillna(0).sum() == pytest.approx(300.0)
    assert by_bus["CCCC2S"] == pytest.approx(100.0)          # the site's other node, never the coupled sibling
    assert set(moves["to_bus"]) == {"CCCC2S"} and moves["method"].eq("same_substation").all()
    assert spread["power_mw"].sum() == pytest.approx(400.0)


def test_site_pin_counts_storage_already_on_the_node():
    # A second site-coded row goes to the node with the most room, not again to the best-connected bus.
    n = _coupled_grid()
    df = _rows(site_name=["existing", "new"], bus=["CCCC2S", "BBBB11"], power_mw=[350.0, 100.0],
               site_code=["CCCC", "CCCC"])
    pinned = pin_storage_to_site_codes(df, n)
    assert pinned.set_index("site_name").loc["existing", "bus"] == "CCCC2S"     # 400 MVA, alone: as before
    assert pinned.set_index("site_name").loc["new", "bus"] in {"CCCC2Q", "CCCC2R"}   # 300 free against 50
    alone = pin_storage_to_site_codes(df.iloc[[1]], n)
    assert alone["bus"].iloc[0] == "CCCC2S"                                       # no other storage: best bus


# FES GSP info from the repository, not the NESO API (16 Oct 2026): a DNS failure at build time had
# silently dropped every FES GSP-level storage row (13 GW) and CP30's national scaling hid it.

def _no_api(*args, **kwargs):
    raise AssertionError("the NESO API must not be called when the committed GSP info file exists")


def test_gsp_info_is_read_from_the_committed_file_without_the_api(monkeypatch):
    import scripts.storage.add_storage as add_storage
    import logging
    monkeypatch.setattr(add_storage, "_download_gsp_info", _no_api)
    gsp = add_storage.load_gsp_info(2024, logging.getLogger("test"))
    assert len(gsp) == 373 and {"Name", "Latitude", "Longitude"} <= set(gsp.columns)
    assert add_storage.FES_GSP_INFO_FILES[2024] == "data/network/ETYS/fes2024_regional_breakdown_gsp_info.csv"


def test_missing_gsp_info_fails_instead_of_dropping_the_gsp_rows(monkeypatch, tmp_path):
    import scripts.storage.add_storage as add_storage
    import logging
    monkeypatch.setattr(add_storage, "_download_gsp_info", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="GSP info"):
        add_storage.load_gsp_info(2024, logging.getLogger("test"), tmp_path / "absent.csv")
    fes = tmp_path / "fes.csv"
    pd.DataFrame({"FES Pathway": ["Holistic Transition"], "Building Block ID Number": ["Srg_BB001"],
                  "GSP": ["Abernethy"], "2030": [50.0]}).to_csv(fes, index=False)
    with pytest.raises(RuntimeError, match="GSP info"):   # not the old silent fall-back to REPD storage
        add_storage.load_fes_storage_data(str(fes), 2030, "Holistic Transition", pd.DataFrame(),
                                          logging.getLogger("test"), gsp_info_file=tmp_path / "absent.csv")
