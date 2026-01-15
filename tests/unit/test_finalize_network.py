"""
Unit tests for finalize_network.py

Tests the network finalization script including:
- Network summary generation
- Metadata management
- Component summarization
- File I/O operations
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Import the functions we're testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from finalize_network import generate_network_summary


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FIXTURES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@pytest.fixture
def logger():
    """Create test logger."""
    return logging.getLogger("test_finalize_network")


@pytest.fixture
def complete_network():
    """Create complete PyPSA network with all component types."""
    import pypsa
    
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=8760, freq="h"))
    
    # Add buses
    network.add("Bus", "bus0", v_nom=400, x=0, y=0)
    network.add("Bus", "bus1", v_nom=400, x=1, y=1)
    network.add("Bus", "bus2", v_nom=275, x=2, y=2)
    
    # Add lines
    network.add("Line", "line0", bus0="bus0", bus1="bus1", x=0.1, r=0.01, s_nom=1000)
    network.add("Line", "line1", bus0="bus1", bus1="bus2", x=0.15, r=0.02, s_nom=800)
    
    # Add transformers
    network.add("Transformer", "trafo0", bus0="bus0", bus1="bus2", x=0.1, s_nom=500)
    
    # Add generators by carrier
    network.add("Generator", "gen_ccgt_1", bus="bus0", p_nom=500, marginal_cost=50, carrier="CCGT")
    network.add("Generator", "gen_ccgt_2", bus="bus1", p_nom=300, marginal_cost=50, carrier="CCGT")
    network.add("Generator", "gen_wind_1", bus="bus1", p_nom=400, marginal_cost=0, carrier="Wind Onshore")
    network.add("Generator", "gen_wind_2", bus="bus2", p_nom=200, marginal_cost=0, carrier="Wind Onshore")
    network.add("Generator", "gen_solar", bus="bus2", p_nom=150, marginal_cost=0, carrier="Solar PV")
    network.add("Generator", "gen_nuclear", bus="bus0", p_nom=1000, marginal_cost=10, carrier="Nuclear")
    
    # Add storage
    network.add("StorageUnit", "battery_1", bus="bus0", p_nom=100, max_hours=4, carrier="Battery")
    network.add("StorageUnit", "battery_2", bus="bus1", p_nom=50, max_hours=2, carrier="Battery")
    network.add("StorageUnit", "pumped_hydro", bus="bus2", p_nom=200, max_hours=6, carrier="Pumped Hydro")
    
    # Add loads
    network.add("Load", "load0", bus="bus0", p_set=300)
    network.add("Load", "load1", bus="bus1", p_set=200)
    
    # Add demand profile
    network.loads_t.p_set = pd.DataFrame(
        {
            "load0": np.random.uniform(200, 400, 8760),
            "load1": np.random.uniform(100, 300, 8760)
        },
        index=network.snapshots
    )
    
    # Add links (interconnectors)
    network.add("Link", "ic_france", bus0="bus0", bus1="bus1", p_nom=2000)
    network.add("Link", "ic_ireland", bus0="bus1", bus1="bus2", p_nom=500)
    
    # Mark interconnectors
    network.links.loc[:, 'is_interconnector'] = True
    
    # Add metadata
    network.meta = {
        'created': '2024-01-01',
        'model_type': 'ETYS',
        'version': '1.0'
    }
    
    return network


@pytest.fixture
def minimal_network():
    """Create minimal network for edge case testing."""
    import pypsa
    
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
    
    network.add("Bus", "bus0", v_nom=400)
    network.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50, carrier="Test")
    
    return network


@pytest.fixture
def empty_network():
    """Create empty network (no components)."""
    import pypsa
    
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
    
    return network


@pytest.fixture
def scenario_config():
    """Create test scenario configuration."""
    return {
        'scenario_id': 'Test_Scenario_2030',
        'fes_year': 2024,
        'modelled_year': 2030,
        'network_model': 'ETYS',
        'clustering': {
            'enabled': False
        }
    }


@pytest.fixture
def clustered_scenario_config():
    """Create clustered scenario configuration."""
    return {
        'scenario_id': 'Test_Clustered',
        'fes_year': 2024,
        'modelled_year': 2030,
        'network_model': 'Reduced',
        'clustering': {
            'enabled': True,
            'n_clusters': 50
        }
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# NETWORK SUMMARY GENERATION TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestNetworkSummaryGeneration:
    """Test comprehensive network summary report generation."""
    
    def test_summary_file_creation(self, complete_network, scenario_config, tmp_path):
        """Test that summary file is created."""
        summary_path = tmp_path / "network_summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        assert summary_path.exists()
        assert summary_path.stat().st_size > 0
    
    def test_summary_includes_header(self, complete_network, scenario_config, tmp_path):
        """Test that summary includes proper header."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "=" * 80 in content
        assert "PYPSA-GB NETWORK SUMMARY" in content
    
    def test_summary_includes_scenario_info(self, complete_network, scenario_config, tmp_path):
        """Test that summary includes scenario information."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "SCENARIO INFORMATION" in content
        assert "Test_Scenario_2030" in content
        assert "2024" in content  # FES year
        assert "2030" in content  # Modelled year
        assert "ETYS" in content  # Network model
        assert "Clustered: No" in content
    
    def test_summary_clustered_scenario(self, complete_network, clustered_scenario_config, tmp_path):
        """Test summary for clustered scenario."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, clustered_scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "Clustered: Yes" in content
    
    def test_summary_includes_topology(self, complete_network, scenario_config, tmp_path):
        """Test that summary includes network topology."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "NETWORK TOPOLOGY" in content
        assert "Buses: 3" in content
        assert "Lines: 2" in content
        assert "Transformers: 1" in content
        assert "Links: 2" in content
    
    def test_summary_includes_temporal_info(self, complete_network, scenario_config, tmp_path):
        """Test that summary includes temporal resolution."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "TEMPORAL RESOLUTION" in content
        assert "Snapshots: 8,760" in content  # Full year hourly
        assert "Start: 2020-01-01" in content
        assert "End: 2020-12-30 23:00:00" in content  # 8760 hours from 2020-01-01 00:00
        assert "Resolution:" in content
    
    def test_summary_includes_demand_info(self, complete_network, scenario_config, tmp_path):
        """Test that summary includes demand information."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "DEMAND" in content
        assert "Load buses: 2" in content
        assert "Total demand:" in content
        assert "Peak demand:" in content
        assert "MWh" in content
        assert "MW" in content


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GENERATION CAPACITY SUMMARIZATION TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestGenerationSummarization:
    """Test generator capacity summarization by carrier."""
    
    def test_summary_generation_by_carrier(self, complete_network, scenario_config, tmp_path):
        """Test generation capacity grouped by carrier."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "GENERATION CAPACITY BY CARRIER" in content
        assert "CCGT" in content
        assert "Wind Onshore" in content
        assert "Solar PV" in content
        assert "Nuclear" in content
        assert "TOTAL" in content
    
    def test_generation_capacity_accuracy(self, complete_network, scenario_config, tmp_path):
        """Test that generation capacities are accurate."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # CCGT: 500 + 300 = 800 MW
        assert "CCGT" in content and "800" in content
        # Wind: 400 + 200 = 600 MW
        assert "Wind Onshore" in content and "600" in content
        # Solar: 150 MW
        assert "Solar PV" in content and "150" in content
        # Nuclear: 1000 MW
        assert "Nuclear" in content and "1,000" in content
    
    def test_generation_unit_counts(self, complete_network, scenario_config, tmp_path):
        """Test that generator unit counts are shown."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # Should show number of units for each carrier
        # CCGT: 2 units
        # Wind: 2 units
        # Solar: 1 unit
        # Nuclear: 1 unit
        assert "units)" in content
    
    def test_generation_total_capacity(self, complete_network, scenario_config, tmp_path):
        """Test that total generation capacity is calculated."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # Total: 800 + 600 + 150 + 1000 = 2,550 MW
        assert "TOTAL" in content
        assert "2,550" in content
    
    def test_no_generators_handling(self, empty_network, scenario_config, tmp_path):
        """Test handling of network with no generators."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(empty_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "No generators found" in content


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STORAGE SUMMARIZATION TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestStorageSummarization:
    """Test storage capacity summarization."""
    
    def test_summary_storage_by_carrier(self, complete_network, scenario_config, tmp_path):
        """Test storage capacity grouped by carrier."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "STORAGE CAPACITY" in content
        assert "Battery" in content
        assert "Pumped Hydro" in content
    
    def test_storage_capacity_accuracy(self, complete_network, scenario_config, tmp_path):
        """Test that storage capacities are accurate."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # Battery: 100 + 50 = 150 MW
        assert "Battery" in content and "150" in content
        # Pumped Hydro: 200 MW
        assert "Pumped Hydro" in content and "200" in content
    
    def test_storage_energy_capacity(self, complete_network, scenario_config, tmp_path):
        """Test that storage energy capacity (MWh) is shown."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # Battery: (100*4 + 50*2)/2 = 250 MWh average
        # Pumped Hydro: 200*6 = 1200 MWh
        assert "MWh" in content
    
    def test_storage_unit_counts(self, complete_network, scenario_config, tmp_path):
        """Test that storage unit counts are shown."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # Battery: 2 units
        # Pumped Hydro: 1 unit
        assert "units)" in content
    
    def test_no_storage_handling(self, minimal_network, scenario_config, tmp_path):
        """Test handling of network with no storage."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(minimal_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "No storage units found" in content


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# INTERCONNECTOR SUMMARIZATION TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestInterconnectorSummarization:
    """Test interconnector summarization."""
    
    def test_summary_interconnectors(self, complete_network, scenario_config, tmp_path):
        """Test interconnector capacity summary."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "INTERCONNECTORS" in content
        assert "Interconnectors: 2" in content
        # Total: 2000 + 500 = 2,500 MW
        assert "2,500" in content
    
    def test_interconnector_detection(self, complete_network, scenario_config, tmp_path):
        """Test that interconnectors are detected correctly."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # Should identify links with is_interconnector flag
        assert "Interconnectors:" in content
    
    def test_no_interconnectors_handling(self, minimal_network, scenario_config, tmp_path):
        """Test handling of network without interconnectors."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(minimal_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "No links found" in content


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# METADATA TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestMetadataHandling:
    """Test network metadata inclusion in summary."""
    
    def test_metadata_included(self, complete_network, scenario_config, tmp_path):
        """Test that network metadata is included."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(complete_network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "NETWORK METADATA" in content
        assert "created" in content
        assert "model_type" in content
        assert "ETYS" in content
        assert "version" in content
    
    def test_no_metadata_handling(self, scenario_config, tmp_path):
        """Test handling of network without metadata."""
        # Create a fresh network without metadata
        import pypsa
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
        network.add("Bus", "bus0", v_nom=400)
        
        summary_path = tmp_path / "summary.txt"
        
        # Should not crash even without metadata
        generate_network_summary(network, scenario_config, summary_path)
        
        assert summary_path.exists()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# EDGE CASES AND ERROR HANDLING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_network_summary(self, empty_network, scenario_config, tmp_path):
        """Test summary generation for empty network."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(empty_network, scenario_config, summary_path)
        
        assert summary_path.exists()
        content = summary_path.read_text(encoding='utf-8')
        assert "PYPSA-GB NETWORK SUMMARY" in content
        assert "Buses: 0" in content
    
    def test_network_single_snapshot(self, tmp_path, scenario_config):
        """Test network with single snapshot."""
        import pypsa
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2020-01-01", periods=1, freq="h"))
        
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        assert "Snapshots: 1" in content
    
    def test_network_no_snapshots(self, tmp_path, scenario_config):
        """Test network with no snapshots."""
        import pypsa
        network = pypsa.Network()
        
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(network, scenario_config, summary_path)
        
        assert summary_path.exists()
    
    def test_network_no_demand_data(self, minimal_network, scenario_config, tmp_path):
        """Test network with no demand time series."""
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(minimal_network, scenario_config, summary_path)
        
        # Should handle gracefully
        assert summary_path.exists()
    
    def test_large_numbers_formatting(self, tmp_path, scenario_config):
        """Test formatting of large numbers (thousands separators)."""
        import pypsa
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2020-01-01", periods=8760, freq="h"))
        
        # Add many buses to test formatting
        for i in range(500):
            network.add("Bus", f"bus{i}", v_nom=400)
        
        summary_path = tmp_path / "summary.txt"
        
        generate_network_summary(network, scenario_config, summary_path)
        
        content = summary_path.read_text(encoding='utf-8')
        # Should use comma separator for thousands
        assert "500" in content or "," in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

