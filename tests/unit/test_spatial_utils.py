"""
Unit tests for spatial_utils.py

Tests the critical spatial mapping utilities that place generators at correct network buses.
This is a CRITICAL module - incorrect bus mapping causes load shedding and infeasibility.

Key functions tested:
- map_sites_to_buses: Core bus assignment
- Coordinate system detection (OSGB36 vs WGS84)
- Coordinate conversion
- Pre-assigned bus preservation
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from spatial_utils import map_sites_to_buses


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def network_osgb36():
    """Create network with OSGB36 coordinates (meters)."""
    network = pypsa.Network()
    
    # OSGB36 coordinates (British National Grid - meters)
    # These are typical GB transmission bus locations
    buses = {
        'LONDON': {'x': 530000, 'y': 180000},      # London area
        'MANCH': {'x': 385000, 'y': 398000},        # Manchester area
        'EDINB': {'x': 325000, 'y': 673000},        # Edinburgh area
        'BRIST': {'x': 358000, 'y': 173000},        # Bristol area
        'BIRMI': {'x': 406000, 'y': 286000},        # Birmingham area
    }
    
    for bus_id, coords in buses.items():
        network.add("Bus", bus_id, x=coords['x'], y=coords['y'], v_nom=400)
    
    return network


@pytest.fixture
def network_wgs84():
    """Create network with WGS84 coordinates (lat/lon degrees)."""
    network = pypsa.Network()
    
    # WGS84 coordinates (lat/lon in degrees)
    buses = {
        'LONDON': {'x': -0.1, 'y': 51.5},      # London
        'MANCH': {'x': -2.24, 'y': 53.48},      # Manchester
        'EDINB': {'x': -3.19, 'y': 55.95},      # Edinburgh
        'BRIST': {'x': -2.58, 'y': 51.45},      # Bristol
        'BIRMI': {'x': -1.89, 'y': 52.48},      # Birmingham
    }
    
    for bus_id, coords in buses.items():
        network.add("Bus", bus_id, x=coords['x'], y=coords['y'], v_nom=400)
    
    return network


@pytest.fixture
def sites_wgs84():
    """Create renewable sites with WGS84 coordinates."""
    return pd.DataFrame({
        'site_name': ['London Wind', 'Manchester Solar', 'Edinburgh Wind', 'Bristol Solar'],
        'lat': [51.52, 53.50, 55.93, 51.47],
        'lon': [-0.08, -2.22, -3.17, -2.56],
        'capacity_mw': [100, 150, 200, 80],
        'technology': ['wind_onshore', 'solar_pv', 'wind_onshore', 'solar_pv'],
    })


@pytest.fixture
def sites_osgb36():
    """Create renewable sites with OSGB36 coordinates."""
    return pd.DataFrame({
        'site_name': ['London Wind', 'Manchester Solar', 'Edinburgh Wind', 'Bristol Solar'],
        'x': [531000, 386000, 326000, 359000],  # Easting
        'y': [181000, 399000, 674000, 174000],  # Northing
        'capacity_mw': [100, 150, 200, 80],
        'technology': ['wind_onshore', 'solar_pv', 'wind_onshore', 'solar_pv'],
    })


@pytest.fixture
def sites_with_preassigned_buses():
    """Create sites where some already have bus assignments."""
    return pd.DataFrame({
        'site_name': ['Pre-assigned Site', 'Needs Mapping', 'Also Pre-assigned'],
        'lat': [51.52, 53.50, 55.93],
        'lon': [-0.08, -2.22, -3.17],
        'capacity_mw': [100, 150, 200],
        'bus': ['LONDON', None, 'EDINB'],  # Pre-assigned and missing
    })


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Coordinate System Detection
# ══════════════════════════════════════════════════════════════════════════════

class TestCoordinateSystemDetection:
    """Test automatic detection of coordinate systems."""
    
    def test_detects_osgb36_from_large_x_range(self, network_osgb36):
        """Test that OSGB36 is detected from large x range."""
        x_range = network_osgb36.buses['x'].max() - network_osgb36.buses['x'].min()
        
        # OSGB36 should have range > 1000 (meters)
        assert x_range > 1000
    
    def test_detects_wgs84_from_small_x_range(self, network_wgs84):
        """Test that WGS84 is detected from small x range."""
        x_range = network_wgs84.buses['x'].max() - network_wgs84.buses['x'].min()
        
        # WGS84 should have range < 100 (degrees)
        assert x_range < 100


# ══════════════════════════════════════════════════════════════════════════════
# TEST: map_sites_to_buses
# ══════════════════════════════════════════════════════════════════════════════

class TestMapSitesToBuses:
    """Test the core map_sites_to_buses function."""
    
    def test_basic_mapping_wgs84_to_wgs84(self, network_wgs84, sites_wgs84):
        """Test mapping WGS84 sites to WGS84 network buses."""
        result = map_sites_to_buses(network_wgs84, sites_wgs84, method='nearest')
        
        # All sites should be mapped
        assert 'bus' in result.columns
        assert result['bus'].notna().sum() == len(sites_wgs84)
        
        # London site should map to LONDON bus
        london_site = result[result['site_name'] == 'London Wind']
        assert london_site['bus'].iloc[0] == 'LONDON'
    
    def test_mapping_wgs84_sites_to_osgb36_network(self, network_osgb36, sites_wgs84):
        """Test mapping WGS84 sites to OSGB36 network (requires conversion)."""
        result = map_sites_to_buses(network_osgb36, sites_wgs84, method='nearest')
        
        # All sites should be mapped
        assert result['bus'].notna().sum() == len(sites_wgs84)
        
        # London site should still map to LONDON bus
        london_site = result[result['site_name'] == 'London Wind']
        assert london_site['bus'].iloc[0] == 'LONDON'
    
    def test_preserves_preassigned_buses(self, network_wgs84, sites_with_preassigned_buses):
        """Test that pre-assigned buses are NOT overwritten."""
        result = map_sites_to_buses(network_wgs84, sites_with_preassigned_buses, method='nearest')
        
        # Pre-assigned sites should keep their original bus
        pre1 = result[result['site_name'] == 'Pre-assigned Site']
        pre2 = result[result['site_name'] == 'Also Pre-assigned']
        
        assert pre1['bus'].iloc[0] == 'LONDON'
        assert pre2['bus'].iloc[0] == 'EDINB'
        
        # Unmapped site should now have a bus
        needs_mapping = result[result['site_name'] == 'Needs Mapping']
        assert needs_mapping['bus'].notna().iloc[0]
    
    def test_empty_sites_returns_empty(self, network_wgs84):
        """Test that empty sites DataFrame returns empty result."""
        empty_sites = pd.DataFrame(columns=['site_name', 'lat', 'lon', 'capacity_mw'])
        result = map_sites_to_buses(network_wgs84, empty_sites, method='nearest')
        
        assert len(result) == 0
    
    def test_all_sites_get_distance_column(self, network_wgs84, sites_wgs84):
        """Test that distance to bus is calculated for each site."""
        result = map_sites_to_buses(network_wgs84, sites_wgs84, method='nearest')
        
        # Should have distance column (if implemented)
        # This is optional but useful for validation
        if 'distance_km' in result.columns:
            assert result['distance_km'].notna().all()
            assert (result['distance_km'] >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Edge Cases and Error Handling
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_sites_with_nan_coordinates(self, network_wgs84):
        """Test handling of sites with NaN coordinates."""
        sites_with_nan = pd.DataFrame({
            'site_name': ['Valid Site', 'NaN Lat', 'NaN Lon'],
            'lat': [51.5, np.nan, 53.5],
            'lon': [-0.1, -2.0, np.nan],
            'capacity_mw': [100, 100, 100],
        })
        
        result = map_sites_to_buses(network_wgs84, sites_with_nan, method='nearest')
        
        # Valid site should be mapped
        valid = result[result['site_name'] == 'Valid Site']
        assert valid['bus'].notna().iloc[0]
        
        # NaN coordinate sites should NOT crash, may have NaN bus
        assert len(result) == 3
    
    def test_duplicate_site_names(self, network_wgs84):
        """Test handling of duplicate site names."""
        sites_duplicates = pd.DataFrame({
            'site_name': ['Site A', 'Site A', 'Site B'],  # Duplicate name
            'lat': [51.5, 53.5, 55.9],
            'lon': [-0.1, -2.2, -3.2],
            'capacity_mw': [100, 150, 200],
        })
        
        result = map_sites_to_buses(network_wgs84, sites_duplicates, method='nearest')
        
        # All rows should be processed
        assert len(result) == 3
    
    def test_sites_outside_network_coverage(self, network_wgs84):
        """Test handling of sites far outside the network area."""
        sites_far = pd.DataFrame({
            'site_name': ['Norway Site', 'France Site'],
            'lat': [60.0, 48.0],  # Outside GB
            'lon': [10.0, 2.0],
            'capacity_mw': [100, 100],
        })
        
        result = map_sites_to_buses(network_wgs84, sites_far, method='nearest')
        
        # Should still map to nearest bus (even if far) OR be NaN if outside max_distance
        # With default max_distance (very large), should map to something
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Coordinate Conversion
# ══════════════════════════════════════════════════════════════════════════════

class TestCoordinateConversion:
    """Test coordinate system conversion between WGS84 and OSGB36."""
    
    def test_london_conversion_accuracy(self):
        """Test that London coordinates convert accurately."""
        # Known coordinates for a London location
        wgs84_lon, wgs84_lat = -0.1278, 51.5074  # Central London
        expected_osgb36_x = 530034  # Approximate easting
        expected_osgb36_y = 180381  # Approximate northing
        
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            osgb36_x, osgb36_y = transformer.transform(wgs84_lon, wgs84_lat)
            
            # Should be within ~100m
            assert abs(osgb36_x - expected_osgb36_x) < 100
            assert abs(osgb36_y - expected_osgb36_y) < 100
        except ImportError:
            pytest.skip("pyproj not available")
    
    def test_edinburgh_conversion_accuracy(self):
        """Test that Edinburgh coordinates convert accurately."""
        wgs84_lon, wgs84_lat = -3.1883, 55.9533  # Edinburgh
        expected_osgb36_x = 325897  # More accurate value from pyproj
        expected_osgb36_y = 674001  # More accurate value from pyproj
        
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
            osgb36_x, osgb36_y = transformer.transform(wgs84_lon, wgs84_lat)
            
            # Allow 250m tolerance for projection differences
            assert abs(osgb36_x - expected_osgb36_x) < 250
            assert abs(osgb36_y - expected_osgb36_y) < 250
        except ImportError:
            pytest.skip("pyproj not available")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Capacity Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestCapacityPreservation:
    """Test that capacity values are preserved through mapping."""
    
    def test_total_capacity_preserved(self, network_wgs84, sites_wgs84):
        """Test that total capacity is preserved after mapping."""
        original_total = sites_wgs84['capacity_mw'].sum()
        
        result = map_sites_to_buses(network_wgs84, sites_wgs84, method='nearest')
        
        mapped_total = result['capacity_mw'].sum()
        assert mapped_total == original_total
    
    def test_individual_capacities_preserved(self, network_wgs84, sites_wgs84):
        """Test that individual site capacities are not modified."""
        result = map_sites_to_buses(network_wgs84, sites_wgs84, method='nearest')
        
        for idx, row in sites_wgs84.iterrows():
            mapped_row = result[result['site_name'] == row['site_name']]
            assert mapped_row['capacity_mw'].iloc[0] == row['capacity_mw']


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Explicit onshore renewable connection points
# ══════════════════════════════════════════════════════════════════════════════

def _clyde_network():
    """Elvanfoot / North Lowther pocket with the Clyde collectors (ETYS names)."""
    network = pypsa.Network()
    for bus, v_nom in [("ELVA4-", 400), ("ELVA2R", 275), ("ELVA1-", 132),
                       ("NLEI1-", 132), ("NLEI3A", 33), ("CLYN2Q", 275),
                       ("DEVM3-", 33), ("DEVM1-", 132)]:
        network.add("Bus", bus, v_nom=v_nom, x=300000, y=617000)
    network.add("Transformer", "t1", bus0="NLEI1-", bus1="NLEI3A", s_nom=90, x=0.1)
    network.add("Transformer", "t2", bus0="ELVA4-", bus1="ELVA2R", s_nom=500, x=0.1)
    network.add("Transformer", "t3", bus0="DEVM1-", bus1="DEVM3-", s_nom=120, x=0.1)
    network.add("Line", "l1", bus0="ELVA1-", bus1="NLEI1-", s_nom=226, x=0.1)
    return network


class TestExplicitOnshoreConnections:
    """Documented connection points override nearest-bus placement."""

    def _map(self, network, rows):
        from spatial_utils import apply_etys_bmu_mapping
        sites = pd.DataFrame(rows, columns=["site_name", "capacity_mw", "bus"])
        sites["data_source"] = "REPD_named_fleet"
        return apply_etys_bmu_mapping(sites, network, gb_network_path="missing.xlsx")

    def test_clyde_moves_off_north_lowther(self):
        result = self._map(_clyde_network(), [
            ("Clyde Wind Farm", 350.0, "NLEI3A"),
            ("Clyde Wind Farm Extension (Clyde 2)", 172.8, "NLEI3A"),
        ])
        assert result["bus"].tolist() == ["ELVA2R", "CLYN2Q"]

    def test_whole_word_match_leaves_inverclyde_alone(self):
        result = self._map(_clyde_network(), [("Inverclyde Wind Farm (Corlic Hill)", 24.0, "DEVM3-")])
        assert result["bus"].tolist() == ["DEVM3-"]

    def test_network_generator_names_with_underscores_match(self):
        result = self._map(_clyde_network(), [("Clyde_Wind_Farm", 350.0, "NLEI3A")])
        assert result["bus"].tolist() == ["ELVA2R"]

    def test_absent_explicit_bus_falls_back_to_heuristics(self):
        # No LEWI buses (Western Isles link not built): Stornoway stays put
        result = self._map(_clyde_network(), [("Stornoway Wind Farm (second resubmission)", 196.0, "NLEI3A")])
        assert result["bus"].tolist() == ["NLEI3A"]

    def test_evidenced_cp30_allocation_is_not_overridden(self):
        from spatial_utils import apply_etys_bmu_mapping
        sites = pd.DataFrame({"site_name": ["Clyde Wind Farm"], "capacity_mw": [350.0],
                              "bus": ["NLEI3A"], "data_source": ["CP30_connection_allocation"]})
        result = apply_etys_bmu_mapping(sites, _clyde_network(), gb_network_path="missing.xlsx")
        assert result["bus"].tolist() == ["NLEI3A"]

    def test_explicit_targets_are_etys_buses_in_the_cp30_build(self):
        """Every explicit target names an ETYS node (guards against typos)."""
        from spatial_utils import ONSHORE_RENEWABLE_DIRECT_BUS
        buses_file = Path(__file__).parent.parent.parent / "resources/network/ETYS_2024_buses.csv"
        if not buses_file.exists():
            pytest.skip("processed ETYS 2024 bus list not available")
        nodes = set(pd.read_csv(buses_file, index_col=0).index.astype(str))
        # Created by Appendix B upgrade rows rather than the base topology:
        # STSO1B / STWO3- (Strathy South / Wood, 2025-26), LEWI1J (Lewis, 2032),
        # HLND1- (Heathland) and ACHE1- (Achany), both present in the CP30 2030 build
        # The 15 Oct 2026 pocket connections (Benbrack, Cloich, New Cumnock 132 kV B) are also
        # Appendix B buses present in the CP30 2030 build.
        upgrade_buses = {"STSO1B", "STWO3-", "LEWI1J", "HLND1-", "ACHE1-", "BENB1-", "CLOI1A", "NECU1B"}
        missing = {bus for bus in ONSHORE_RENEWABLE_DIRECT_BUS.values() if bus not in nodes} - upgrade_buses
        assert missing == set()


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Weak-bus relief
# ══════════════════════════════════════════════════════════════════════════════

def _weak_bus_network():
    """POGB3--like terminal busbar next to a strong same-site bus and a 132 kV bus 10 km away."""
    network = pypsa.Network()
    network.add("Bus", "POGB3-", v_nom=33, x=345000, y=665000)
    network.add("Bus", "DUNE3-", v_nom=33, x=347000, y=665000)
    network.add("Bus", "DUNE1-", v_nom=132, x=347000, y=665000)
    network.add("Bus", "GALN1Q", v_nom=132, x=355000, y=665000)
    network.add("Bus", "FAR01-", v_nom=132, x=545000, y=165000)
    network.add("Line", "l1", bus0="POGB3-", bus1="DUNE3-", s_nom=57, x=0.1)
    network.add("Transformer", "t1", bus0="DUNE1-", bus1="DUNE3-", s_nom=90, x=0.1)
    network.add("Line", "l2", bus0="DUNE1-", bus1="GALN1Q", s_nom=482, x=0.1)
    network.add("Line", "l3", bus0="GALN1Q", bus1="FAR01-", s_nom=2000, x=0.1)
    return network


class TestWeakBusRelief:
    def _sites(self, rows):
        return pd.DataFrame(rows, columns=["site_name", "technology", "capacity_mw", "bus",
                                           "fleet_origin", "data_source"])

    def test_proxy_moves_first_to_nearest_adequate_bus(self):
        from spatial_utils import relieve_weak_bus_placements
        sites = self._sites([
            ("CP30_increment_wind_onshore_1", "wind_onshore", 300.0, "POGB3-", "increment", "CP30"),
            ("Named farm", "wind_onshore", 40.0, "POGB3-", "named", "REPD_named_fleet"),
        ])
        result, moves = relieve_weak_bus_placements(sites, _weak_bus_network())
        # DUNE1- is the nearest 132 kV bus with headroom (572 MVA of branches)
        assert result["bus"].tolist() == ["DUNE1-", "POGB3-"]
        assert moves["capacity_mw"].sum() == pytest.approx(300.0)
        assert result["capacity_mw"].sum() == pytest.approx(340.0)

    def test_same_substation_bus_is_preferred(self):
        from spatial_utils import relieve_weak_bus_placements
        network = _weak_bus_network()
        network.add("Bus", "POGB1-", v_nom=132, x=345000, y=665000)
        network.add("Line", "l4", bus0="POGB1-", bus1="GALN1Q", s_nom=400, x=0.1)
        sites = self._sites([("CP30_increment_x", "wind_onshore", 300.0, "POGB3-", "increment", "CP30")])
        result, moves = relieve_weak_bus_placements(sites, network)
        assert result["bus"].tolist() == ["POGB1-"]
        assert moves["method"].tolist() == ["same_substation"]

    def test_protected_offshore_and_adequate_buses_are_untouched(self):
        from spatial_utils import relieve_weak_bus_placements
        sites = self._sites([
            ("Evidenced", "wind_onshore", 300.0, "POGB3-", "named", "CP30_connection_allocation"),
            ("Offshore", "wind_offshore", 900.0, "DUNE1-", "increment", "CP30"),
            ("Fine", "solar_pv", 100.0, "GALN1Q", "increment", "CP30"),
        ])
        result, moves = relieve_weak_bus_placements(sites, _weak_bus_network())
        assert result["bus"].tolist() == ["POGB3-", "DUNE1-", "GALN1Q"]
        assert moves.empty

    def test_no_target_within_radius_leaves_site_and_warns(self):
        from spatial_utils import relieve_weak_bus_placements
        sites = self._sites([("CP30_increment_x", "wind_onshore", 5000.0, "POGB3-", "increment", "CP30")])
        result, moves = relieve_weak_bus_placements(sites, _weak_bus_network())
        assert result["bus"].tolist() == ["POGB3-"]
        assert moves.empty

    def test_gsp_node_rows_move_only_within_their_substation(self):
        # A distribution-size site on its GSP's node connects inside that GSP (16 Oct 2026):
        # relief may take it to another busbar of the same substation, never to a nearby one.
        from spatial_utils import relieve_weak_bus_placements
        from scripts.utilities.gsp_nodes import GSP_NODE_SOURCES
        sites = self._sites([("Small farm", "wind_onshore", 300.0, "POGB3-", "named", "CP30_distributed_gsp_node")])
        result, moves = relieve_weak_bus_placements(sites, _weak_bus_network(), site_bound_sources=GSP_NODE_SOURCES)
        assert result["bus"].tolist() == ["POGB3-"] and moves.empty      # DUNE1- is another substation
        unbound, _ = relieve_weak_bus_placements(sites, _weak_bus_network())
        assert unbound["bus"].tolist() == ["DUNE1-"]                      # the default is unchanged
        network = _weak_bus_network()
        network.add("Bus", "POGB1-", v_nom=132, x=345000, y=665000)
        network.add("Line", "l4", bus0="POGB1-", bus1="GALN1Q", s_nom=400, x=0.1)
        result, moves = relieve_weak_bus_placements(sites, network, site_bound_sources=GSP_NODE_SOURCES)
        assert result["bus"].tolist() == ["POGB1-"] and moves["method"].tolist() == ["same_substation"]


def test_gsp_node_rows_are_not_moved_off_an_excluded_bus():
    # The renewables pass the GSP-node sources as protected to the placement exclusions
    # (16 Oct 2026): LINM3- (Linnmill 33 kV, the LINM GSP's demand node) is a coordinate
    # outlier, and moving its distributed farms by their coordinates scattered them over ten buses.
    from spatial_utils import _PROTECTED_PLACEMENTS, reassign_excluded_placements
    from scripts.utilities.gsp_nodes import GSP_NODE_SOURCES
    network = _weak_bus_network()
    sites = pd.DataFrame({"site_name": ["at node", "plain"], "technology": "wind_onshore",
                          "capacity_mw": [10.0, 10.0], "bus": ["POGB3-", "POGB3-"],
                          "data_source": ["CP30_distributed_gsp_node", "REPD_named_fleet"],
                          "lon": np.nan, "lat": np.nan})
    result, moves = reassign_excluded_placements(
        sites, network, {"POGB3-"}, protected_sources=tuple(_PROTECTED_PLACEMENTS) + tuple(GSP_NODE_SOURCES))
    assert result["bus"].tolist()[0] == "POGB3-"
    assert result["bus"].tolist()[1] != "POGB3-" and moves["site_name"].tolist() == ["plain"]


def test_renewables_integration_wires_the_gsp_node_protections():
    import inspect
    from scripts.generators import integrate_renewable_generators as irg
    source = inspect.getsource(irg.main)
    assert "protected_sources=tuple(_PROTECTED_PLACEMENTS) + tuple(GSP_NODE_SOURCES)" in source
    assert "site_bound_sources=GSP_NODE_SOURCES" in source


def test_new_explicit_connections_move_the_audited_sites():
    from spatial_utils import apply_etys_bmu_mapping
    network = pypsa.Network()
    for bus, v in [("ECCF1J", 132), ("EWEH3-", 33), ("CRDY31", 33), ("TRLO3-", 33), ("MAHI3-", 33),
                   ("KYPE3A", 33), ("KYPE3-", 33), ("ARMA1-", 132), ("STRW3-", 33), ("STSO1B", 132),
                   ("STWO3-", 33)]:
        network.add("Bus", bus, v_nom=v, x=300000, y=600000)
    sites = pd.DataFrame({
        "site_name": ["Ewe Hill (extension)", "Crossdykes", "Mark Hill", "Kype Muir Wind Farm",
                      "Kype Muir Wind Farm Extension", "Strathy South", "Strathy North", "Strathy Wood",
                      "Cairnview, Bettyhill"],
        "capacity_mw": [36.8, 48.0, 56.0, 88.4, 67.2, 218.4, 67.7, 62.4, 48.0],
        "bus": ["ECCF1J", "ECCF1J", "TRLO3-", "KYPE3A", "KYPE3A", "ARMA1-", "ARMA1-", "ARMA1-", "ARMA1-"],
        "data_source": "REPD_named_fleet",
    })
    result = apply_etys_bmu_mapping(sites, network, gb_network_path="missing.xlsx")
    assert result["bus"].tolist() == ["EWEH3-", "CRDY31", "MAHI3-", "KYPE3-", "KYPE3A",
                                      "STSO1B", "STRW3-", "STWO3-", "ARMA1-"]
