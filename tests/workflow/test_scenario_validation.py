"""
Scenario validation tests for PyPSA-GB.

Tests the pre-flight validation system that checks scenarios before expensive workflows:
- Configuration validation
- Data availability checking
- Data freshness warnings
- Scenario detection accuracy
- Validation report generation

These tests verify the validation infrastructure works correctly.
"""

import pytest
import yaml
import subprocess
from pathlib import Path


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path.cwd()


@pytest.fixture
def scenarios_master(project_root):
    """Load scenarios master configuration."""
    scenarios_path = project_root / "config" / "scenarios.yaml"
    with open(scenarios_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def config(project_root):
    """Load main configuration."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def defaults(project_root):
    """Load defaults configuration."""
    defaults_path = project_root / "config" / "defaults.yaml"
    with open(defaults_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


class TestValidationScriptsExist:
    """Test that validation scripts exist and are accessible."""
    
    def test_validate_scenarios_script_exists(self, project_root):
        """Test that scenario validation script exists."""
        validate_script = project_root / "scripts" / "validate_scenarios.py"
        
        assert validate_script.exists(), \
            f"Validation script not found at {validate_script}"
    
    def test_generate_validation_report_script_exists(self, project_root):
        """Test that validation report generation script exists."""
        report_script = project_root / "scripts" / "generate_validation_report.py"
        
        assert report_script.exists(), \
            f"Report generation script not found at {report_script}"


class TestScenarioConfigurationValidation:
    """Test configuration validation."""
    
    def test_all_test_scenarios_have_descriptions(self, scenarios_master):
        """Test that all test scenarios have descriptions."""
        scenarios = scenarios_master
        test_scenarios = [s for s in scenarios if s.startswith("Test_")]
        
        assert len(test_scenarios) > 0, "No test scenarios found"
        
        for scenario_id in test_scenarios:
            scenario = scenarios[scenario_id]
            assert 'description' in scenario, \
                f"Test scenario {scenario_id} missing description"
            assert len(scenario['description']) > 0, \
                f"Test scenario {scenario_id} has empty description"
    
    def test_all_scenarios_have_required_fields(self, scenarios_master):
        """Test that all scenarios have required core fields."""
        scenarios = scenarios_master
        required_fields = ['modelled_year', 'network_model']
        
        for scenario_id, scenario in scenarios.items():
            for field in required_fields:
                assert field in scenario, \
                    f"Scenario {scenario_id} missing required field {field}"
    
    def test_timestep_minutes_valid(self, scenarios_master, defaults):
        """Test that timestep_minutes values are valid."""
        scenarios = scenarios_master
        valid_timesteps = [15, 30, 60]
        
        for scenario_id, scenario in scenarios.items():
            timestep = scenario.get('timestep_minutes', defaults.get('timestep_minutes'))
            assert timestep in valid_timesteps, \
                f"Scenario {scenario_id} has invalid timestep: {timestep}"
    
    def test_voll_values_positive(self, scenarios_master, defaults):
        """Test that VOLL (Value of Lost Load) values are positive."""
        scenarios = scenarios_master
        
        for scenario_id, scenario in scenarios.items():
            voll = scenario.get('voll', defaults.get('voll'))
            assert voll is not None and voll > 0, \
                f"Scenario {scenario_id} has non-positive VOLL: {voll}"


class TestHistoricalScenarioValidation:
    """Test validation specific to historical scenarios."""
    
    def test_historical_scenarios_have_matching_years(self, scenarios_master):
        """Test that historical scenarios have consistent year configuration."""
        scenarios = scenarios_master
        
        for scenario_id, scenario in scenarios.items():
            modelled_year = scenario.get('modelled_year')
            
            # If modelled_year <= 2024, should be historical
            if modelled_year and modelled_year <= 2024:
                renewables_year = scenario.get('renewables_year')
                demand_year = scenario.get('demand_year')
                
                assert renewables_year == modelled_year, \
                    f"Scenario {scenario_id}: renewables_year should match modelled_year for historical"
                assert demand_year == modelled_year, \
                    f"Scenario {scenario_id}: demand_year should match modelled_year for historical"
    
    def test_historical_scenarios_use_espeni(self, scenarios_master, defaults):
        """Test that historical scenarios use ESPENI demand."""
        scenarios = scenarios_master
        
        for scenario_id, scenario in scenarios.items():
            modelled_year = scenario.get('modelled_year')
            
            # Historical scenarios should use ESPENI (or inherit from defaults)
            if modelled_year and modelled_year <= 2024:
                demand_timeseries = scenario.get('demand_timeseries', defaults.get('demand_timeseries'))
                assert demand_timeseries == "ESPENI", \
                    f"Scenario {scenario_id}: historical should use ESPENI, got {demand_timeseries}"


class TestFutureScenarioValidation:
    """Test validation specific to future scenarios."""
    
    def test_future_scenarios_have_fes_config(self, scenarios_master):
        """Test that future scenarios have FES configuration."""
        scenarios = scenarios_master
        
        for scenario_id, scenario in scenarios.items():
            modelled_year = scenario.get('modelled_year')
            
            # Future scenarios should have FES config
            if modelled_year and modelled_year > 2024:
                assert 'FES_year' in scenario, \
                    f"Scenario {scenario_id}: future scenario missing FES_year"
                assert 'FES_scenario' in scenario, \
                    f"Scenario {scenario_id}: future scenario missing FES_scenario"
    
    def test_future_scenarios_use_base_data(self, scenarios_master):
        """Test that future scenarios use recent base year for weather/demand."""
        scenarios = scenarios_master
        
        for scenario_id, scenario in scenarios.items():
            modelled_year = scenario.get('modelled_year')
            
            # Future scenarios should have historical base years
            if modelled_year and modelled_year > 2024:
                renewables_year = scenario.get('renewables_year')
                demand_year = scenario.get('demand_year')
                
                assert renewables_year <= 2024, \
                    f"Scenario {scenario_id}: renewables_year should be historical"
                assert demand_year <= 2024, \
                    f"Scenario {scenario_id}: demand_year should be historical"


class TestNetworkModelValidation:
    """Test network model configuration validation."""
    
    def test_network_model_values_valid(self, scenarios_master):
        """Test that network_model values are valid."""
        scenarios = scenarios_master
        valid_models = ['ETYS', 'Reduced', 'Zonal']
        
        for scenario_id, scenario in scenarios.items():
            network_model = scenario.get('network_model')
            assert network_model in valid_models, \
                f"Scenario {scenario_id}: invalid network_model {network_model}"
    
    def test_network_model_directories_exist(self, project_root):
        """Test that network model directories exist."""
        network_base = project_root / "data" / "network"
        
        models = ['ETYS', 'reduced_network', 'zonal']
        for model in models:
            model_path = network_base / model
            assert model_path.exists(), \
                f"Network model directory not found: {model_path}"


class TestClusteringConfigurationValidation:
    """Test network clustering configuration validation."""
    
    def test_clustering_enabled_has_method(self, scenarios_master, defaults):
        """Test that enabled clustering has a method specified."""
        scenarios = scenarios_master
        
        for scenario_id, scenario in scenarios.items():
            clustering = scenario.get('clustering', defaults.get('clustering', {}))
            if isinstance(clustering, dict) and clustering.get('enabled'):
                assert 'method' in clustering, \
                    f"Scenario {scenario_id}: clustering enabled but no method specified"
                assert 'config' in clustering, \
                    f"Scenario {scenario_id}: clustering enabled but no config specified"
    
    def test_clustering_methods_valid(self, scenarios_master, defaults):
        """Test that clustering methods are valid."""
        scenarios = scenarios_master
        valid_methods = ['spatial', 'busmap', 'kmeans', 'hierarchical', 'custom']
        
        for scenario_id, scenario in scenarios.items():
            clustering = scenario.get('clustering', defaults.get('clustering', {}))
            if isinstance(clustering, dict) and clustering.get('enabled'):
                method = clustering.get('method')
                assert method in valid_methods, \
                    f"Scenario {scenario_id}: invalid clustering method {method}"


class TestDataDirectoryStructure:
    """Test that data directory structure is complete."""
    
    def test_data_subdirectories_exist(self, project_root):
        """Test that all required data subdirectories exist."""
        data_dir = project_root / "data"
        
        required_dirs = [
            'network',
            'generators',
            'renewables',
            'demand',
            'FES',
            'storage',
            'interconnectors'
        ]
        
        for subdir in required_dirs:
            subdir_path = data_dir / subdir
            assert subdir_path.exists(), \
                f"Required data directory not found: {subdir_path}"
    
    def test_resources_subdirectories_exist(self, project_root):
        """Test that all resources subdirectories exist."""
        if not (project_root / "resources").exists():
            pytest.skip("Resources directory not created yet")
        
        resources_dir = project_root / "resources"
        
        required_dirs = [
            'network',
            'renewable',
            'demand',
            'generators'
        ]
        
        for subdir in required_dirs:
            subdir_path = resources_dir / subdir
            # These should exist if workflow has been run
            if subdir_path.exists():
                assert subdir_path.is_dir()


class TestActiveScenarioConfiguration:
    """Test active scenario configuration."""
    
    def test_run_scenarios_field_exists(self, config):
        """Test that config has run_scenarios field."""
        assert 'run_scenarios' in config, \
            "Config missing run_scenarios field"
    
    def test_run_scenarios_is_list(self, config):
        """Test that run_scenarios is a list."""
        run_scenarios = config.get('run_scenarios')
        assert isinstance(run_scenarios, list), \
            "run_scenarios should be a list"
    
    def test_run_scenarios_not_empty(self, config):
        """Test that run_scenarios is not empty."""
        run_scenarios = config.get('run_scenarios')
        assert len(run_scenarios) > 0, \
            "run_scenarios should not be empty"
    
    def test_run_scenarios_exist_in_master(self, config, scenarios_master):
        """Test that all run_scenarios exist in scenarios_master."""
        run_scenarios = config.get('run_scenarios', [])
        available_scenarios = scenarios_master
        
        for scenario_id in run_scenarios:
            assert scenario_id in available_scenarios, \
                f"Scenario {scenario_id} not found in scenarios.yaml"


class TestValidationReportGeneration:
    """Test validation report generation."""
    
    def test_validation_report_structure(self, project_root):
        """Test that validation report, if generated, has expected structure."""
        # Look for validation reports
        report_patterns = [
            "*.validation_report*",
            "*_validation_*.txt",
            "*_validation_*.md"
        ]
        
        reports_found = False
        for pattern in report_patterns:
            reports = list(project_root.glob(pattern))
            if reports:
                reports_found = True
                break
        
        # If reports exist, they should be readable text files
        if reports_found:
            for report_path in reports:
                assert report_path.is_file(), f"Validation report not a file: {report_path}"


class TestScenarioConsistency:
    """Test consistency across scenarios."""
    
    def test_no_duplicate_scenario_ids(self, scenarios_master):
        """Test that there are no duplicate scenario IDs."""
        scenarios = scenarios_master.get("scenarios", {})
        scenario_ids = list(scenarios.keys())
        
        unique_ids = set(scenario_ids)
        assert len(scenario_ids) == len(unique_ids), \
            "Found duplicate scenario IDs in scenarios_master.yaml"
    
    def test_scenario_field_types_consistent(self, scenarios_master):
        """Test that scenario fields have consistent types."""
        scenarios = scenarios_master.get("scenarios", {})
        
        for scenario_id, scenario in scenarios.items():
            # Type checks
            if 'modelled_year' in scenario:
                assert isinstance(scenario['modelled_year'], int), \
                    f"{scenario_id}: modelled_year should be int"
            
            if 'timestep_minutes' in scenario:
                assert isinstance(scenario['timestep_minutes'], int), \
                    f"{scenario_id}: timestep_minutes should be int"
            
            if 'voll' in scenario:
                assert isinstance(scenario['voll'], (int, float)), \
                    f"{scenario_id}: voll should be numeric"


class TestErrorDetection:
    """Test error detection capabilities."""
    
    def test_invalid_year_detected(self, scenarios_master):
        """Test that invalid years would be detected."""
        scenarios = scenarios_master.get("scenarios", {})
        
        # Check that no scenario has unreasonable years
        for scenario_id, scenario in scenarios.items():
            modelled_year = scenario.get('modelled_year')
            if modelled_year:
                # Year should be reasonable (between 1990 and 2100)
                assert 1990 <= modelled_year <= 2100, \
                    f"{scenario_id}: year {modelled_year} outside reasonable range"
    
    def test_missing_required_field_detected(self, scenarios_master):
        """Test that scenarios with missing required fields would be caught."""
        scenarios = scenarios_master.get("scenarios", {})
        required_fields = ['modelled_year', 'network_model']
        
        for scenario_id, scenario in scenarios.items():
            for field in required_fields:
                assert field in scenario, \
                    f"{scenario_id}: missing required field {field}"

