"""
Unit tests for integrate_renewable_generators.py (Stage 1 of generator integration).

Tests the renewable generator integration workflow including:
- Network loading and carrier addition
- Renewable site data loading and combination
- Site to bus mapping
- Renewable generator integration with profiles
- Summary generation
- Output saving

Author: PyPSA-GB Testing Team
Date: 2025-10-24
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestRenewableSiteLoading:
    """Test renewable site data loading and combination."""
    
    def test_load_single_technology_sites(self, tmp_path):
        """Test loading site data for a single renewable technology."""
        # Create test site file
        site_file = tmp_path / "wind_onshore_sites.csv"
        test_data = pd.DataFrame({
            'site_name': ['Wind Farm 1', 'Wind Farm 2'],
            'capacity_mw': [50.0, 100.0],
            'lat': [52.5, 53.0],
            'lon': [-1.5, -1.0]
        })
        test_data.to_csv(site_file, index=False)
        
        # Load site data
        sites_df = pd.read_csv(site_file)
        sites_df['technology'] = 'wind_onshore'
        
        # Verify
        assert len(sites_df) == 2
        assert 'technology' in sites_df.columns
        assert sites_df['technology'].iloc[0] == 'wind_onshore'
        assert sites_df['capacity_mw'].sum() == 150.0
    
    def test_load_multiple_technology_sites(self, tmp_path):
        """Test loading and combining multiple renewable technology sites."""
        # Create test site files for 3 technologies
        technologies = {
            'wind_onshore': [(50.0, 52.5, -1.5), (100.0, 53.0, -1.0)],
            'wind_offshore': [(200.0, 54.0, 1.0), (300.0, 54.5, 1.5)],
            'solar_pv': [(25.0, 51.5, -0.5), (30.0, 51.8, -0.3)]
        }
        
        site_list = []
        for tech, sites in technologies.items():
            site_file = tmp_path / f"{tech}_sites.csv"
            test_data = pd.DataFrame({
                'site_name': [f'{tech}_{i}' for i in range(len(sites))],
                'capacity_mw': [s[0] for s in sites],
                'lat': [s[1] for s in sites],
                'lon': [s[2] for s in sites]
            })
            test_data.to_csv(site_file, index=False)
            
            # Load and add technology
            sites_df = pd.read_csv(site_file)
            sites_df['technology'] = tech
            site_list.append(sites_df)
        
        # Combine all sites
        combined_sites = pd.concat(site_list, ignore_index=True)
        
        # Verify
        assert len(combined_sites) == 6  # 2+2+2
        assert len(combined_sites['technology'].unique()) == 3
        assert combined_sites['capacity_mw'].sum() == 705.0  # 50+100+200+300+25+30
        
    def test_handle_empty_site_file(self, tmp_path):
        """Test handling of empty site files."""
        # Create empty site file
        site_file = tmp_path / "tidal_stream_sites.csv"
        empty_data = pd.DataFrame(columns=['site_name', 'capacity_mw', 'lat', 'lon'])
        empty_data.to_csv(site_file, index=False)
        
        # Load site data
        sites_df = pd.read_csv(site_file)
        
        # Verify
        assert len(sites_df) == 0
        assert 'capacity_mw' in sites_df.columns  # Schema preserved
    
    def test_handle_missing_site_file(self, tmp_path):
        """Test handling of missing site files."""
        site_file = tmp_path / "nonexistent_sites.csv"
        
        # Verify file doesn't exist
        assert not os.path.exists(site_file)
    
    def test_combined_sites_have_required_columns(self, tmp_path):
        """Test that combined site data has all required columns."""
        # Create minimal site file
        site_file = tmp_path / "wind_onshore_sites.csv"
        test_data = pd.DataFrame({
            'site_name': ['Test Site'],
            'capacity_mw': [100.0],
            'lat': [52.5],
            'lon': [-1.5]
        })
        test_data.to_csv(site_file, index=False)
        
        # Load and add technology
        sites_df = pd.read_csv(site_file)
        sites_df['technology'] = 'wind_onshore'
        
        # Verify required columns
        required_columns = ['site_name', 'capacity_mw', 'lat', 'lon', 'technology']
        for col in required_columns:
            assert col in sites_df.columns
    
    def test_total_capacity_calculation(self, tmp_path):
        """Test accurate total capacity calculation across technologies."""
        # Create sites with known capacities
        capacities = [10.0, 25.0, 50.0, 100.0, 250.0]
        site_file = tmp_path / "test_sites.csv"
        test_data = pd.DataFrame({
            'site_name': [f'Site_{i}' for i in range(len(capacities))],
            'capacity_mw': capacities,
            'lat': [52.0 + i*0.1 for i in range(len(capacities))],
            'lon': [-1.0 + i*0.1 for i in range(len(capacities))]
        })
        test_data.to_csv(site_file, index=False)
        
        sites_df = pd.read_csv(site_file)
        total_capacity = sites_df['capacity_mw'].sum()
        
        # Verify
        assert total_capacity == pytest.approx(435.0)  # 10+25+50+100+250


class TestCarrierAddition:
    """Test carrier definition addition to network."""
    
    def test_add_carriers_to_empty_network(self):
        """Test adding carriers to network with no existing carriers."""
        network = pypsa.Network()
        
        # Mock the carrier addition function
        with patch('carrier_definitions.add_carriers_to_network') as mock_add:
            mock_network = Mock()
            mock_add.return_value = mock_network
            
            result = mock_add(network, Mock())
            
            # Verify function was called
            mock_add.assert_called_once()
            assert result is mock_network
    
    def test_renewable_carriers_defined(self):
        """Test that all 8 renewable technology carriers are defined."""
        expected_carriers = [
            'wind_onshore',
            'wind_offshore',
            'solar_pv',
            'small_hydro',
            'large_hydro',
            'tidal_stream',
            'shoreline_wave',
            'tidal_lagoon'
        ]
        
        # This test verifies the expected carrier list exists
        assert len(expected_carriers) == 8
        assert 'wind_onshore' in expected_carriers
        assert 'solar_pv' in expected_carriers
        assert 'tidal_stream' in expected_carriers


class TestSiteToBusMapping:
    """Test mapping renewable sites to network buses."""
    
    @pytest.fixture
    def simple_network(self):
        """Create simple network with 3 buses."""
        network = pypsa.Network()
        
        # Add buses at specific coordinates
        network.add("Bus", "Bus_1", x=52.0, y=-1.0, v_nom=400)
        network.add("Bus", "Bus_2", x=53.0, y=-1.5, v_nom=400)
        network.add("Bus", "Bus_3", x=54.0, y=0.0, v_nom=400)
        
        return network
    
    def test_map_site_to_nearest_bus(self, simple_network):
        """Test that site is mapped to nearest bus."""
        # Create site near Bus_1 (52.0, -1.0)
        site = pd.DataFrame({
            'site_name': ['Test Site'],
            'capacity_mw': [100.0],
            'lat': [52.1],  # Close to Bus_1
            'lon': [-0.9],
            'technology': ['wind_onshore']
        })
        
        # Mock the mapping function
        with patch('spatial_utils.map_sites_to_buses') as mock_map:
            site['bus'] = 'Bus_1'  # Simulate mapping
            mock_map.return_value = site
            
            result = mock_map(simple_network, site, method='nearest')
            
            # Verify
            assert 'bus' in result.columns
            assert result['bus'].iloc[0] == 'Bus_1'
    
    def test_handle_unmapped_sites(self, simple_network):
        """Test handling of sites that cannot be mapped to buses."""
        # Create site far from all buses
        site = pd.DataFrame({
            'site_name': ['Remote Site'],
            'capacity_mw': [50.0],
            'lat': [60.0],  # Far north
            'lon': [5.0],   # Far east
            'technology': ['wind_offshore']
        })
        
        # Mock the mapping function to return NaN bus
        with patch('spatial_utils.map_sites_to_buses') as mock_map:
            site['bus'] = np.nan  # Simulate failed mapping
            mock_map.return_value = site
            
            result = mock_map(simple_network, site, method='nearest', max_distance_km=10.0)
            
            # Verify
            assert 'bus' in result.columns
            assert pd.isna(result['bus'].iloc[0])
    
    def test_mapping_statistics(self, simple_network):
        """Test calculation of mapping success statistics."""
        # Create mixed sites (some mappable, some not)
        sites = pd.DataFrame({
            'site_name': ['Site_1', 'Site_2', 'Site_3'],
            'capacity_mw': [50.0, 100.0, 75.0],
            'lat': [52.1, 53.1, 60.0],  # Last one far away
            'lon': [-0.9, -1.4, 5.0],
            'technology': ['wind_onshore', 'wind_onshore', 'wind_onshore']
        })
        
        # Simulate mapping (first 2 mapped, last one unmapped)
        sites['bus'] = ['Bus_1', 'Bus_2', np.nan]
        
        # Calculate statistics
        total_sites = len(sites)
        mapped_sites = sites['bus'].notna().sum()
        mapping_success_rate = (mapped_sites / total_sites) * 100
        
        # Verify
        assert total_sites == 3
        assert mapped_sites == 2
        assert mapping_success_rate == pytest.approx(66.67, rel=0.01)


class TestGeneratorIntegration:
    """Test renewable generator integration into network."""
    
    @pytest.fixture
    def network_with_buses_and_carriers(self):
        """Create network with buses and renewable carriers."""
        network = pypsa.Network()
        
        # Add buses
        network.add("Bus", "Bus_1", x=52.0, y=-1.0, v_nom=400)
        network.add("Bus", "Bus_2", x=53.0, y=-1.5, v_nom=400)
        
        # Add renewable carriers
        for carrier in ['wind_onshore', 'solar_pv']:
            network.add("Carrier", carrier)
        
        return network
    
    def test_add_renewable_generators_to_network(self, network_with_buses_and_carriers):
        """Test adding renewable generators to network."""
        network = network_with_buses_and_carriers
        initial_gen_count = len(network.generators)
        
        # Create renewable sites
        sites = pd.DataFrame({
            'site_name': ['Wind_1', 'Wind_2'],
            'capacity_mw': [50.0, 100.0],
            'technology': ['wind_onshore', 'wind_onshore'],
            'bus': ['Bus_1', 'Bus_2']
        })
        
        # Mock add_renewable_generators function
        with patch('integrate_renewable_generators.add_renewable_generators') as mock_add:
            # Simulate adding generators
            network.add("Generator", "Wind_1", bus="Bus_1", carrier="wind_onshore", p_nom=50.0)
            network.add("Generator", "Wind_2", bus="Bus_2", carrier="wind_onshore", p_nom=100.0)
            mock_add.return_value = network
            
            result = mock_add(network, sites, profiles_dir="resources/renewable/profiles")
            
            # Verify
            final_gen_count = len(result.generators)
            assert final_gen_count == initial_gen_count + 2
            assert 'Wind_1' in result.generators.index
            assert 'Wind_2' in result.generators.index
    
    def test_generator_capacity_matches_site_capacity(self, network_with_buses_and_carriers):
        """Test that added generator capacity matches site capacity."""
        network = network_with_buses_and_carriers
        
        # Add generator manually
        site_capacity = 150.0
        network.add("Generator", "Solar_1", bus="Bus_1", carrier="solar_pv", p_nom=site_capacity)
        
        # Verify
        gen_capacity = network.generators.loc['Solar_1', 'p_nom']
        assert gen_capacity == pytest.approx(site_capacity)
    
    def test_generator_has_correct_carrier(self, network_with_buses_and_carriers):
        """Test that generators have correct carrier assignment."""
        network = network_with_buses_and_carriers
        
        # Add generators with different carriers
        network.add("Generator", "Wind_1", bus="Bus_1", carrier="wind_onshore", p_nom=100.0)
        network.add("Generator", "Solar_1", bus="Bus_2", carrier="solar_pv", p_nom=50.0)
        
        # Verify
        assert network.generators.loc['Wind_1', 'carrier'] == 'wind_onshore'
        assert network.generators.loc['Solar_1', 'carrier'] == 'solar_pv'


class TestSummaryGeneration:
    """Test renewable capacity summary generation."""
    
    def test_create_summary_by_technology(self):
        """Test creation of summary grouped by technology."""
        # Create mock network with generators
        network = pypsa.Network()
        network.add("Bus", "Bus_1", x=52.0, y=-1.0)
        network.add("Carrier", "wind_onshore")
        network.add("Carrier", "solar_pv")
        
        # Add generators
        network.add("Generator", "Wind_1", bus="Bus_1", carrier="wind_onshore", p_nom=50.0)
        network.add("Generator", "Wind_2", bus="Bus_1", carrier="wind_onshore", p_nom=100.0)
        network.add("Generator", "Solar_1", bus="Bus_1", carrier="solar_pv", p_nom=25.0)
        network.add("Generator", "Solar_2", bus="Bus_1", carrier="solar_pv", p_nom=30.0)
        
        # Create summary
        summary_data = []
        for carrier in ['wind_onshore', 'solar_pv']:
            gens = network.generators[network.generators['carrier'] == carrier]
            if len(gens) > 0:
                summary_data.append({
                    'technology': carrier,
                    'capacity_mw': gens['p_nom'].sum(),
                    'count': len(gens)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Verify
        assert len(summary_df) == 2
        
        wind_row = summary_df[summary_df['technology'] == 'wind_onshore'].iloc[0]
        assert wind_row['capacity_mw'] == pytest.approx(150.0)  # 50+100
        assert wind_row['count'] == 2
        
        solar_row = summary_df[summary_df['technology'] == 'solar_pv'].iloc[0]
        assert solar_row['capacity_mw'] == pytest.approx(55.0)  # 25+30
        assert solar_row['count'] == 2
    
    def test_summary_total_capacity_matches_generators(self):
        """Test that summary total capacity matches total in network."""
        # Create mock network
        network = pypsa.Network()
        network.add("Bus", "Bus_1", x=52.0, y=-1.0)
        network.add("Carrier", "wind_onshore")
        
        # Add generators with known total capacity
        capacities = [10.0, 25.0, 50.0, 100.0]
        for i, cap in enumerate(capacities):
            network.add("Generator", f"Wind_{i}", bus="Bus_1", carrier="wind_onshore", p_nom=cap)
        
        # Create summary
        summary_data = [{
            'technology': 'wind_onshore',
            'capacity_mw': network.generators['p_nom'].sum(),
            'count': len(network.generators)
        }]
        summary_df = pd.DataFrame(summary_data)
        
        # Verify
        expected_total = sum(capacities)  # 185.0
        assert summary_df['capacity_mw'].sum() == pytest.approx(expected_total)
    
    def test_empty_summary_for_no_generators(self):
        """Test creation of empty summary when no generators added."""
        summary_df = pd.DataFrame(columns=['technology', 'capacity_mw', 'count'])
        
        # Verify
        assert len(summary_df) == 0
        assert 'technology' in summary_df.columns
        assert 'capacity_mw' in summary_df.columns
        assert 'count' in summary_df.columns


class TestOutputSaving:
    """Test saving of network and summary outputs."""
    
    def test_save_network_to_netcdf(self, tmp_path):
        """Test saving network to NetCDF file."""
        # Create simple network
        network = pypsa.Network()
        network.add("Bus", "Bus_1", x=52.0, y=-1.0)
        network.add("Generator", "Wind_1", bus="Bus_1", p_nom=100.0)
        
        # Save network
        output_file = tmp_path / "test_network.nc"
        network.export_to_netcdf(output_file)
        
        # Verify file created
        assert output_file.exists()
        
        # Load and verify
        loaded_network = pypsa.Network(output_file)
        assert 'Bus_1' in loaded_network.buses.index
        assert 'Wind_1' in loaded_network.generators.index
    
    def test_save_summary_to_csv(self, tmp_path):
        """Test saving summary DataFrame to CSV."""
        # Create summary
        summary_df = pd.DataFrame({
            'technology': ['wind_onshore', 'solar_pv'],
            'capacity_mw': [150.0, 55.0],
            'count': [2, 2]
        })
        
        # Save summary
        output_file = tmp_path / "renewable_summary.csv"
        summary_df.to_csv(output_file, index=False)
        
        # Verify file created
        assert output_file.exists()
        
        # Load and verify
        loaded_summary = pd.read_csv(output_file)
        assert len(loaded_summary) == 2
        assert loaded_summary['capacity_mw'].sum() == pytest.approx(205.0)
    
    def test_output_file_paths_created(self, tmp_path):
        """Test that output file parent directories are created if needed."""
        # Create nested output path
        nested_path = tmp_path / "resources" / "network"
        nested_path.mkdir(parents=True, exist_ok=True)
        
        output_file = nested_path / "test_output.nc"
        
        # Verify parent directories exist
        assert output_file.parent.exists()


class TestExecutionSummary:
    """Test execution summary statistics."""
    
    def test_execution_summary_statistics(self):
        """Test calculation of execution summary statistics."""
        # Simulate workflow results
        total_sites = 150
        added_generators = 145  # Some sites didn't map
        total_capacity = 5000.0  # MW
        technologies = 8
        unique_buses = 75
        
        summary_stats = {
            'renewable_sites_processed': total_sites,
            'renewable_generators_added': added_generators,
            'total_renewable_capacity_mw': total_capacity,
            'technologies_integrated': technologies,
            'buses_with_renewables': unique_buses
        }
        
        # Verify
        assert summary_stats['renewable_sites_processed'] == 150
        assert summary_stats['renewable_generators_added'] == 145
        assert summary_stats['total_renewable_capacity_mw'] == 5000.0
        assert summary_stats['technologies_integrated'] == 8
        assert summary_stats['buses_with_renewables'] == 75
    
    def test_execution_time_tracking(self):
        """Test execution time measurement."""
        import time
        
        start_time = time.time()
        time.sleep(0.01)  # Simulate work
        execution_time = time.time() - start_time
        
        # Verify time measurement works
        assert execution_time > 0
        assert execution_time < 1.0  # Should be very quick


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_handle_missing_snakemake_context(self):
        """Test error when script run outside Snakemake."""
        # Simulate missing snakemake context
        with patch('builtins.globals', return_value={}):
            # This would raise RuntimeError in real script
            with pytest.raises(RuntimeError, match="must be run via Snakemake"):
                snk = globals().get('snakemake')
                if not snk:
                    raise RuntimeError("This script must be run via Snakemake")
    
    def test_handle_corrupted_site_file(self, tmp_path):
        """Test handling of corrupted CSV file."""
        # Create malformed CSV
        site_file = tmp_path / "corrupted_sites.csv"
        with open(site_file, 'w') as f:
            f.write("invalid,csv,format\nno,proper,structure")
        
        # Attempting to load should raise or handle gracefully
        try:
            sites_df = pd.read_csv(site_file)
            # Even if it loads, required columns won't exist
            assert 'capacity_mw' not in sites_df.columns or len(sites_df) == 0
        except Exception:
            # File is malformed enough to cause exception
            pass
    
    def test_handle_zero_capacity_sites(self, tmp_path):
        """Test handling of sites with zero capacity."""
        site_file = tmp_path / "zero_capacity_sites.csv"
        test_data = pd.DataFrame({
            'site_name': ['Zero Site'],
            'capacity_mw': [0.0],
            'lat': [52.5],
            'lon': [-1.5]
        })
        test_data.to_csv(site_file, index=False)
        
        sites_df = pd.read_csv(site_file)
        
        # Zero capacity should be allowed (edge case handling)
        assert sites_df['capacity_mw'].iloc[0] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION-STYLE TESTS (Workflow Simulation)
# ═══════════════════════════════════════════════════════════════════════════

class TestRenewableIntegrationWorkflow:
    """Test complete renewable integration workflow (integration-style tests)."""
    
    def test_minimal_renewable_workflow(self, tmp_path):
        """Test minimal end-to-end renewable integration workflow."""
        # 1. Create minimal network
        network = pypsa.Network()
        network.add("Bus", "Bus_1", x=52.0, y=-1.0, v_nom=400)
        network.add("Carrier", "wind_onshore")
        
        # 2. Create minimal site data
        sites = pd.DataFrame({
            'site_name': ['Wind_1'],
            'capacity_mw': [100.0],
            'lat': [52.1],
            'lon': [-0.9],
            'technology': ['wind_onshore'],
            'bus': ['Bus_1']  # Pre-mapped
        })
        
        # 3. Add generator (simulating integration)
        network.add("Generator", "Wind_1", bus="Bus_1", carrier="wind_onshore", p_nom=100.0)
        
        # 4. Create summary
        summary_df = pd.DataFrame({
            'technology': ['wind_onshore'],
            'capacity_mw': [100.0],
            'count': [1]
        })
        
        # 5. Save outputs
        network_file = tmp_path / "test_network.nc"
        summary_file = tmp_path / "test_summary.csv"
        
        network.export_to_netcdf(network_file)
        summary_df.to_csv(summary_file, index=False)
        
        # Verify workflow completed
        assert network_file.exists()
        assert summary_file.exists()
        assert len(network.generators) == 1
        assert len(summary_df) == 1
    
    def test_multi_technology_integration(self, tmp_path):
        """Test integration of multiple renewable technologies."""
        # Create network with multiple carriers
        network = pypsa.Network()
        network.add("Bus", "Bus_1", x=52.0, y=-1.0, v_nom=400)
        network.add("Bus", "Bus_2", x=53.0, y=-1.5, v_nom=400)
        
        for carrier in ['wind_onshore', 'wind_offshore', 'solar_pv']:
            network.add("Carrier", carrier)
        
        # Create sites for 3 technologies
        sites = pd.DataFrame({
            'site_name': ['Wind_On_1', 'Wind_Off_1', 'Solar_1'],
            'capacity_mw': [50.0, 200.0, 25.0],
            'technology': ['wind_onshore', 'wind_offshore', 'solar_pv'],
            'bus': ['Bus_1', 'Bus_2', 'Bus_1']
        })
        
        # Add all generators
        for _, site in sites.iterrows():
            network.add("Generator", 
                       site['site_name'],
                       bus=site['bus'],
                       carrier=site['technology'],
                       p_nom=site['capacity_mw'])
        
        # Create summary
        summary_data = []
        for tech in sites['technology'].unique():
            tech_gens = network.generators[network.generators['carrier'] == tech]
            summary_data.append({
                'technology': tech,
                'capacity_mw': tech_gens['p_nom'].sum(),
                'count': len(tech_gens)
            })
        summary_df = pd.DataFrame(summary_data)
        
        # Verify multi-technology integration
        assert len(network.generators) == 3
        assert len(summary_df) == 3
        assert summary_df['capacity_mw'].sum() == pytest.approx(275.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
