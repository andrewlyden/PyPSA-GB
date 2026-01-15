"""
Unit tests for solve_network.py

Tests the network optimization script including:
- Solver configuration
- Solve period handling (full year, explicit dates, auto-select)
- Results export to CSV
- Optimization summary generation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import logging

# Import the functions we're testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from solve_network import (
    configure_solver,
    export_optimization_results,
    generate_optimization_summary
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def set_network_objective(network, value):
    """Set network objective value compatible with PyPSA 0.35+"""
    # Directly set the _objective attribute to avoid deprecation warning
    # This is what PyPSA does internally when accessing network.objective
    network._objective = value


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def logger():
    """Create test logger."""
    return logging.getLogger("test_solve_network")


@pytest.fixture
def minimal_network():
    """Create minimal PyPSA network with optimization results."""
    import pypsa
    
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
    
    # Add basic components
    network.add("Bus", "bus0", v_nom=400)
    network.add("Bus", "bus1", v_nom=400)
    
    # Add line
    network.add("Line", "line0", bus0="bus0", bus1="bus1", x=0.1, r=0.01, s_nom=1000)
    
    # Add generators
    network.add("Generator", "gen_ccgt", bus="bus0", p_nom=500, marginal_cost=50, carrier="CCGT")
    network.add("Generator", "gen_wind", bus="bus1", p_nom=300, marginal_cost=0, carrier="Wind Onshore")
    network.add("Generator", "gen_load_shedding", bus="bus0", p_nom=1000, marginal_cost=10000, carrier="load_shedding")
    
    # Add storage
    network.add("StorageUnit", "battery", bus="bus0", p_nom=100, max_hours=4, carrier="Battery")
    
    # Add load
    network.add("Load", "load0", bus="bus0", p_set=200)
    
    # Add mock optimization results
    network.generators_t.p = pd.DataFrame(
        {
            "gen_ccgt": np.random.uniform(0, 500, 24),
            "gen_wind": np.random.uniform(0, 300, 24),
            "gen_load_shedding": np.zeros(24)  # No load shedding
        },
        index=network.snapshots
    )
    
    network.storage_units_t.p = pd.DataFrame(
        {"battery": np.random.uniform(-100, 100, 24)},
        index=network.snapshots
    )
    
    network.storage_units_t.state_of_charge = pd.DataFrame(
        {"battery": np.random.uniform(0, 400, 24)},
        index=network.snapshots
    )
    
    network.lines_t.p0 = pd.DataFrame(
        {"line0": np.random.uniform(-500, 500, 24)},
        index=network.snapshots
    )
    
    # Add objective (total cost) - use helper to avoid deprecation warning
    set_network_objective(network, 123456.78)
    
    # Add CO2 emissions
    network.generators.loc[:, 'co2_emissions'] = [0.4, 0, 0]  # CCGT, Wind, Load shedding
    
    return network


@pytest.fixture
def scenario_config():
    """Create test scenario configuration."""
    return {
        'scenario_id': 'Test_Scenario',
        'fes_year': 2024,
        'modelled_year': 2030,
        'network_model': 'Reduced',
        'solver': 'highs',
        'solver_options': {'threads': 4}
    }


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER CONFIGURATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSolverConfiguration:
    """Test solver configuration validation and setup."""
    
    def test_valid_solver_gurobi(self, logger):
        """Test valid Gurobi solver configuration."""
        solver_name = 'gurobi'
        solver_options = {'method': 2, 'threads': 8}
        
        name, options = configure_solver(None, solver_name, solver_options, logger)
        
        assert name == 'gurobi'
        assert options == solver_options
    
    def test_valid_solver_highs(self, logger):
        """Test valid HiGHS solver configuration."""
        solver_name = 'highs'
        solver_options = {'threads': 4}
        
        name, options = configure_solver(None, solver_name, solver_options, logger)
        
        assert name == 'highs'
        assert options == solver_options
    
    def test_valid_solver_glpk(self, logger):
        """Test valid GLPK solver configuration."""
        solver_name = 'glpk'
        solver_options = {}
        
        name, options = configure_solver(None, solver_name, solver_options, logger)
        
        assert name == 'glpk'
        assert options == solver_options
    
    def test_valid_solver_cplex(self, logger):
        """Test valid CPLEX solver configuration."""
        solver_name = 'cplex'
        solver_options = {'threads': 4}
        
        name, options = configure_solver(None, solver_name, solver_options, logger)
        
        assert name == 'cplex'
        assert options == solver_options
    
    def test_invalid_solver_fallback(self, logger):
        """Test that invalid solver falls back to HiGHS."""
        solver_name = 'invalid_solver'
        solver_options = {}
        
        name, options = configure_solver(None, solver_name, solver_options, logger)
        
        assert name == 'highs'  # Should fallback
        assert options == solver_options
    
    def test_solver_options_preserved(self, logger):
        """Test that solver options are preserved."""
        solver_name = 'highs'
        solver_options = {
            'threads': 8,
            'time_limit': 3600,
            'mip_rel_gap': 0.01
        }
        
        name, options = configure_solver(None, solver_name, solver_options, logger)
        
        assert options == solver_options
        assert options['threads'] == 8
        assert options['time_limit'] == 3600
    
    def test_empty_solver_options(self, logger):
        """Test solver configuration with empty options."""
        solver_name = 'highs'
        solver_options = {}
        
        name, options = configure_solver(None, solver_name, solver_options, logger)
        
        assert name == 'highs'
        assert options == {}


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPORT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestResultsExport:
    """Test export of optimization results to CSV files."""
    
    def test_export_generation_results(self, minimal_network, tmp_path, logger):
        """Test generation results export."""
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_Scenario", logger
        )
        
        # Check generation DataFrame
        assert not generation_df.empty
        assert len(generation_df) == 24  # 24 snapshots
        assert len(generation_df.columns) == 3  # 3 generators
        
        # Check column naming includes carrier
        assert any('CCGT' in col for col in generation_df.columns)
        assert any('Wind' in col for col in generation_df.columns)
    
    def test_export_storage_results(self, minimal_network, tmp_path, logger):
        """Test storage results export."""
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_Scenario", logger
        )
        
        # Check storage DataFrame
        assert not storage_df.empty
        assert 'snapshot' in storage_df.columns
        assert 'storage_unit' in storage_df.columns
        assert 'carrier' in storage_df.columns
        assert 'p_dispatch' in storage_df.columns
        assert 'state_of_charge' in storage_df.columns
        
        # Check data
        assert len(storage_df) == 24  # 24 snapshots for 1 storage unit
        assert storage_df['carrier'].iloc[0] == 'Battery'
    
    def test_export_flows_results(self, minimal_network, tmp_path, logger):
        """Test line flows export."""
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_Scenario", logger
        )
        
        # Check flows DataFrame
        assert not flows_df.empty
        assert len(flows_df) == 24  # 24 snapshots
        assert len(flows_df.columns) == 1  # 1 line
        assert 'line_line0' in flows_df.columns
    
    def test_export_costs_results(self, minimal_network, tmp_path, logger):
        """Test cost breakdown export."""
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_Scenario", logger
        )
        
        # Check costs DataFrame
        assert not costs_df.empty
        assert 'component_type' in costs_df.columns
        assert 'component_name' in costs_df.columns
        assert 'carrier' in costs_df.columns
        assert 'output_MWh' in costs_df.columns
        assert 'marginal_cost' in costs_df.columns
        assert 'total_cost' in costs_df.columns
        
        # Check data
        assert all(costs_df['component_type'] == 'generator')
        assert len(costs_df) == 3  # 3 generators
    
    def test_export_emissions_results(self, minimal_network, tmp_path, logger):
        """Test emissions export."""
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_Scenario", logger
        )
        
        # Check emissions DataFrame
        assert not emissions_df.empty
        assert 'generator' in emissions_df.columns
        assert 'carrier' in emissions_df.columns
        assert 'generation_MWh' in emissions_df.columns
        assert 'co2_intensity_kg_per_MWh' in emissions_df.columns
        assert 'total_emissions_kg' in emissions_df.columns
        
        # Check data
        assert len(emissions_df) == 3  # 3 generators
        # CCGT should have emissions, wind should have zero
        ccgt_emissions = emissions_df[emissions_df['carrier'] == 'CCGT']['co2_intensity_kg_per_MWh'].values[0]
        wind_emissions = emissions_df[emissions_df['carrier'] == 'Wind Onshore']['co2_intensity_kg_per_MWh'].values[0]
        assert ccgt_emissions > 0
        assert wind_emissions == 0
    
    def test_export_empty_network(self, tmp_path, logger):
        """Test export with empty network (no results)."""
        import pypsa
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
        
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            network, tmp_path, "Test_Empty", logger
        )
        
        # All should be empty
        assert generation_df.empty
        assert storage_df.empty
        assert flows_df.empty
        assert costs_df.empty
        assert emissions_df.empty
    
    def test_cost_calculation_accuracy(self, minimal_network, tmp_path, logger):
        """Test that cost calculations are accurate."""
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_Scenario", logger
        )
        
        # Check cost calculation for each generator
        for idx, row in costs_df.iterrows():
            gen_name = row['component_name']
            expected_output = minimal_network.generators_t.p[gen_name].sum()
            expected_marginal = minimal_network.generators.loc[gen_name, 'marginal_cost']
            expected_cost = expected_output * expected_marginal
            
            assert np.isclose(row['output_MWh'], expected_output, rtol=1e-5)
            assert np.isclose(row['marginal_cost'], expected_marginal, rtol=1e-5)
            assert np.isclose(row['total_cost'], expected_cost, rtol=1e-5)
    
    def test_emissions_calculation_accuracy(self, minimal_network, tmp_path, logger):
        """Test that emissions calculations are accurate."""
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_Scenario", logger
        )
        
        # Check emissions calculation
        for idx, row in emissions_df.iterrows():
            gen_name = row['generator']
            expected_output = minimal_network.generators_t.p[gen_name].sum()
            expected_intensity = minimal_network.generators.loc[gen_name, 'co2_emissions']
            expected_emissions = expected_output * expected_intensity
            
            assert np.isclose(row['generation_MWh'], expected_output, rtol=1e-5)
            assert np.isclose(row['co2_intensity_kg_per_MWh'], expected_intensity, rtol=1e-5)
            assert np.isclose(row['total_emissions_kg'], expected_emissions, rtol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION SUMMARY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizationSummary:
    """Test optimization summary report generation."""
    
    def test_summary_generation(self, minimal_network, scenario_config, tmp_path, logger):
        """Test basic summary generation."""
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network,
            scenario_config,
            'highs',
            'ok',
            45.67,
            summary_path,
            logger
        )
        
        # Check file created
        assert summary_path.exists()
        
        # Check content
        content = summary_path.read_text()
        assert "PYPSA-GB OPTIMIZATION SUMMARY" in content
        assert "Test_Scenario" in content
        assert "highs" in content
        assert "ok" in content
        assert "45.67 seconds" in content
    
    def test_summary_includes_network_info(self, minimal_network, scenario_config, tmp_path, logger):
        """Test that summary includes network information."""
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        assert "2 buses" in content
        assert "3 generators" in content
        assert "24" in content  # snapshots
    
    def test_summary_includes_costs(self, minimal_network, scenario_config, tmp_path, logger):
        """Test that summary includes cost information."""
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        assert "Total system cost" in content
        assert "£123,456.78" in content  # From network.objective
        assert "Average cost per hour" in content
    
    def test_summary_includes_generation_breakdown(self, minimal_network, scenario_config, tmp_path, logger):
        """Test that summary includes generation by carrier."""
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        assert "GENERATION" in content
        assert "CCGT" in content
        assert "Wind Onshore" in content
        assert "MWh" in content
        assert "TOTAL" in content
    
    def test_summary_detects_load_shedding(self, minimal_network, scenario_config, tmp_path, logger):
        """Test that summary detects and warns about load shedding."""
        # Add load shedding
        minimal_network.generators_t.p.loc[:, 'gen_load_shedding'] = 100  # 100 MWh per hour
        
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        assert "WARNING: LOAD SHEDDING DETECTED" in content
        assert "Total load shedding" in content
    
    def test_summary_no_load_shedding_warning(self, minimal_network, scenario_config, tmp_path, logger):
        """Test that summary doesn't warn when no load shedding."""
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        assert "WARNING: LOAD SHEDDING DETECTED" not in content
    
    def test_summary_with_solve_period(self, minimal_network, scenario_config, tmp_path, logger):
        """Test summary with solve period information."""
        scenario_config['solve_period'] = {
            'enabled': True,
            'auto_select': 'peak_demand_week'
        }
        
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        assert "Solve period:" in content
        assert "auto-selected" in content or "days" in content
    
    def test_summary_full_year(self, minimal_network, scenario_config, tmp_path, logger):
        """Test summary indicates full year when no solve period."""
        scenario_config['solve_period'] = {'enabled': False}
        
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        assert "Full year" in content
    
    def test_summary_format_consistency(self, minimal_network, scenario_config, tmp_path, logger):
        """Test that summary has consistent formatting."""
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        
        # Check for section headers
        assert "=" * 80 in content
        assert "-" * 80 in content
        
        # Check main sections present
        assert "SCENARIO" in content
        assert "OPTIMIZATION" in content
        assert "SYSTEM COSTS" in content
        assert "GENERATION" in content
    
    def test_summary_without_objective(self, scenario_config, tmp_path, logger):
        """Test summary generation when network has no objective value."""
        # Create a fresh network without optimization
        import pypsa
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
        network.add("Bus", "bus0", v_nom=400)
        network.add("Generator", "gen0", bus="bus0", p_nom=100, carrier="Test")
        
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        # Should not crash, just skip cost section
        assert summary_path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES AND ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_network_with_no_snapshots(self, tmp_path, logger):
        """Test handling network with no snapshots."""
        import pypsa
        network = pypsa.Network()
        
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            network, tmp_path, "Test_NoSnapshots", logger
        )
        
        assert generation_df.empty
        assert storage_df.empty
        assert flows_df.empty
    
    def test_network_with_single_snapshot(self, tmp_path, logger):
        """Test handling network with single snapshot."""
        import pypsa
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2020-01-01", periods=1, freq="h"))
        
        network.add("Bus", "bus0")
        network.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=50, carrier="Test")
        network.generators_t.p = pd.DataFrame({"gen0": [50]}, index=network.snapshots)
        
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            network, tmp_path, "Test_SingleSnapshot", logger
        )
        
        assert len(generation_df) == 1
        assert not generation_df.empty
    
    def test_generator_without_results(self, minimal_network, tmp_path, logger):
        """Test handling generator that exists but has no results."""
        # Add generator but don't add results
        minimal_network.add("Generator", "gen_new", bus="bus0", p_nom=100, marginal_cost=30, carrier="New")
        
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            minimal_network, tmp_path, "Test_MissingResults", logger
        )
        
        # Should only include generators with results
        assert 'gen_new_New' not in generation_df.columns
    
    def test_very_large_cost_formatting(self, minimal_network, scenario_config, tmp_path, logger):
        """Test formatting of very large cost values."""
        set_network_objective(minimal_network, 1_234_567_890.12)
        
        summary_path = tmp_path / "summary.txt"
        
        generate_optimization_summary(
            minimal_network, scenario_config, 'highs', 'ok', 30.0, summary_path, logger
        )
        
        content = summary_path.read_text()
        # Should use thousands separators
        assert "1,234,567,890" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
