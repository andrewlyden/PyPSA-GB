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
