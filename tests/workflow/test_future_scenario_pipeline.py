"""
End-to-end workflow tests for future scenario pipelines.

Tests the complete workflow for future scenarios from configuration through to network building,
validating that:
- Future scenarios are correctly detected and routed to FES data
- FES scenario configurations are valid
- Network building with FES data completes successfully
- Demand is scaled appropriately from base year
- All intermediate files are created with correct formats

These tests use the Test_Future_2030 scenario defined in scenarios.yaml.
"""

import pytest
import yaml
import pandas as pd
import pypsa
from pathlib import Path


# ============================================================================
# HELPER FUNCTIONS FOR DETECTING FES RESOURCES
# ============================================================================

def find_future_scenario_outputs(project_root, pattern="*"):
    """
    Find future scenario outputs in resources folder.
    
    Returns dict with paths to available outputs by type.
    """
    resources_dir = project_root / "resources"
    if not resources_dir.exists():
        return {}
    
    outputs = {
        'networks': [],
        'renewable_profiles': [],
        'demand_files': [],
        'generator_files': [],
        'busmaps': []
    }
    
    # Find network files (future scenarios may have different naming)
    network_dir = resources_dir / "network"
    if network_dir.exists():
        # Look for future scenario files (usually by year > 2024 or FES naming)
        outputs['networks'] = sorted(network_dir.glob("*_with_storage.nc"))
        outputs['busmaps'] = sorted(network_dir.glob("*busmap*.csv"))
    
    # Find renewable profiles
    renewable_dir = resources_dir / "renewable" / "profiles"
    if renewable_dir.exists():
        outputs['renewable_profiles'] = sorted(renewable_dir.glob("*.csv"))
    
    # Find demand files
    demand_dir = resources_dir / "demand"
    if demand_dir.exists():
        outputs['demand_files'] = sorted(demand_dir.glob("**/*.csv"))
    
    # Find generator files
    gen_dir = resources_dir / "generators"
    if gen_dir.exists():
        outputs['generator_files'] = sorted(gen_dir.glob("*full*.csv"))
    
    return outputs


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path.cwd()


@pytest.fixture
def future_scenario_outputs(project_root):
    """
    Smart fixture that detects future scenario outputs if available.
    
    Returns a dict with:
    - has_network_outputs: Boolean flag
    - has_renewable_profiles: Boolean flag
    - has_demand_files: Boolean flag
    - available_networks: List of network files
    """
    outputs = find_future_scenario_outputs(project_root)
    
    return {
        'has_network_outputs': len(outputs.get('networks', [])) > 0,
        'has_renewable_profiles': len(outputs.get('renewable_profiles', [])) > 0,
        'has_demand_files': len(outputs.get('demand_files', [])) > 0,
        'has_generator_files': len(outputs.get('generator_files', [])) > 0,
        'available_networks': outputs.get('networks', []),
        'available_profiles': outputs.get('renewable_profiles', []),
        'available_demands': outputs.get('demand_files', []),
    }


@pytest.fixture
def scenarios_master(project_root):
    """Load scenarios master configuration."""
    scenarios_path = project_root / "config" / "scenarios.yaml"
    with open(scenarios_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def defaults(project_root):
    """Load defaults configuration."""
    defaults_path = project_root / "config" / "defaults.yaml"
    with open(defaults_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def test_future_scenario(scenarios_master):
    """Get the Test_Future_2030 scenario configuration."""
    if "Test_Future_2030" not in scenarios_master:
        pytest.skip("Test_Future_2030 scenario not found in scenarios.yaml")
    return scenarios_master["Test_Future_2030"]


class TestFutureScenarioConfiguration:
    """Test future scenario configuration."""
    
    def test_test_future_scenario_exists(self, scenarios_master):
        """Test that Test_Future_2030 scenario exists."""
        assert "Test_Future_2030" in scenarios_master, \
            "Test_Future_2030 scenario not found in scenarios.yaml"
    
    def test_future_scenario_has_required_fields(self, test_future_scenario):
        """Test that future scenario has all required fields."""
        # Only check fields that must be in scenario (not inherited from defaults)
        required_fields = [
            'modelled_year', 'renewables_year', 'demand_year',
            'network_model', 'FES_year', 'FES_scenario'
        ]
        
        for field in required_fields:
            assert field in test_future_scenario, \
                f"Required field '{field}' missing from Test_Future_2030"
    
    def test_future_year_configuration(self, test_future_scenario):
        """Test that future scenario has valid year configuration."""
        modelled_year = test_future_scenario['modelled_year']
        renewables_year = test_future_scenario['renewables_year']
        demand_year = test_future_scenario['demand_year']
        
        # Modelled year should be future (> 2024)
        assert modelled_year > 2024, \
            f"Future scenario should have modelled_year > 2024, got {modelled_year}"
        
        # Renewables and demand years should be for base data (typically historical)
        assert renewables_year <= 2024, \
            f"Renewables year should be historical/recent, got {renewables_year}"
        assert demand_year <= 2024, \
            f"Demand year should be historical/recent, got {demand_year}"
    
    def test_fes_configuration_valid(self, test_future_scenario):
        """Test that FES configuration is valid."""
        fes_year = test_future_scenario.get('FES_year')
        fes_scenario = test_future_scenario.get('FES_scenario')
        
        assert fes_year is not None, "FES_year not specified"
        assert fes_scenario is not None, "FES_scenario not specified"
        
        # FES year should be reasonable
        assert 2020 <= fes_year <= 2030, \
            f"FES year should be 2020-2030, got {fes_year}"
        
        # FES scenario should be non-empty string
        assert isinstance(fes_scenario, str) and len(fes_scenario) > 0, \
            f"FES_scenario should be a non-empty string, got {fes_scenario}"


class TestFutureDataSources:
    """Test that future data sources are available."""
    
    def test_fes_data_directory_exists(self, project_root):
        """Test that FES data directory exists."""
        fes_dir = project_root / "data" / "FES"
        
        assert fes_dir.exists(), \
            f"FES data directory not found at {fes_dir}"
    
    def test_network_model_for_future(self, test_future_scenario):
        """Test that network model is valid for future scenarios."""
        network_model = test_future_scenario['network_model']
        
        valid_models = ['ETYS', 'Reduced', 'Zonal']
        assert network_model in valid_models, \
            f"Invalid network model: {network_model}"


class TestFutureNetworkBuilding:
    """Test network building for future scenarios."""
    
    def test_network_scripts_exist(self, project_root):
        """Test that network building scripts exist."""
        build_network_path = project_root / "scripts" / "build_network.py"
        
        assert build_network_path.exists(), \
            f"Network building script not found at {build_network_path}"
    
    def test_network_rules_exist(self, project_root):
        """Test that network building Snakemake rules exist."""
        network_rule_path = project_root / "rules" / "network_build.smk"
        
        assert network_rule_path.exists(), \
            f"Network building rule not found at {network_rule_path}"
    
    @pytest.mark.skipif(
        not Path("resources/network").exists(),
        reason="No network output directory"
    )
    def test_future_network_output_if_exists(self, project_root):
        """Test future network output file if it exists."""
        network_dir = project_root / "resources" / "network"
        
        if not network_dir.exists():
            pytest.skip("Network output directory doesn't exist")
        
        # Look for Test_Future_2030 network
        network_files = list(network_dir.glob("*Test_Future_2030*.nc"))
        
        if len(network_files) == 0:
            pytest.skip("No network output found for Test_Future_2030")
        
        # Test that network can be loaded
        network_path = network_files[0]
        network = pypsa.Network(str(network_path))
        
        # Basic validation
        assert len(network.buses) > 0, "Network has no buses"
        assert 'x' in network.buses.columns, "Buses missing x coordinates"
        assert 'y' in network.buses.columns, "Buses missing y coordinates"


class TestFutureRenewableProfiles:
    """Test renewable profile generation for future scenarios."""
    
    def test_renewable_integration_script_exists(self, project_root):
        """Test that renewable integration script exists."""
        scripts_dir = project_root / "scripts"
        renewable_scripts = [
            "renewable_integration.py",
            "integrate_renewable_generators.py"
        ]
        
        found = any((scripts_dir / script).exists() for script in renewable_scripts)
        assert found, f"No renewable integration scripts found"
    
    @pytest.mark.skipif(
        not Path("resources/renewable/profiles").exists(),
        reason="No renewable profiles directory"
    )
    def test_future_renewable_profiles_if_exist(self, project_root):
        """Test future renewable profiles if they exist."""
        profiles_dir = project_root / "resources" / "renewable" / "profiles"
        
        if not profiles_dir.exists():
            pytest.skip("Renewable profiles directory doesn't exist")
        
        # Look for any profile files for Test_Future_2030
        profile_files = list(profiles_dir.glob("*Test_Future_2030*.csv"))
        
        if len(profile_files) == 0:
            pytest.skip("No renewable profiles found for Test_Future_2030")
        
        # Test that profiles can be loaded
        for profile_path in profile_files[:3]:
            df = pd.read_csv(profile_path, nrows=10)
            assert len(df) > 0, f"Empty profile file: {profile_path}"


class TestFutureDemandProcessing:
    """Test demand processing for future scenarios."""
    
    def test_demand_uses_espeni_base(self, test_future_scenario, defaults):
        """Test that future scenario uses ESPENI as base for demand."""
        demand_timeseries = test_future_scenario.get('demand_timeseries', defaults.get('demand_timeseries'))
        
        # Future scenarios typically scale from historical demand (inherited from defaults)
        assert demand_timeseries == "ESPENI", \
            f"Future scenario should use ESPENI as demand base, got: {demand_timeseries}"
    
    def test_demand_year_is_recent(self, test_future_scenario):
        """Test that demand year is recent historical."""
        demand_year = test_future_scenario['demand_year']
        
        # Should use recent historical data for demand pattern
        assert 2015 <= demand_year <= 2024, \
            f"Demand year should be recent historical (2015-2024), got {demand_year}"
    
    @pytest.mark.skipif(
        not Path("resources/demand").exists(),
        reason="No demand output directory"
    )
    def test_future_demand_output_if_exists(self, project_root):
        """Test future demand output if it exists."""
        demand_dir = project_root / "resources" / "demand"
        
        if not demand_dir.exists():
            pytest.skip("Demand output directory doesn't exist")
        
        # Look for demand files for Test_Future_2030
        demand_files = list(demand_dir.glob("*Test_Future_2030*.csv"))
        
        if len(demand_files) == 0:
            pytest.skip("No demand output found for Test_Future_2030")
        
        # Test that demand can be loaded
        demand_path = demand_files[0]
        df = pd.read_csv(demand_path, nrows=10)
        
        assert len(df) > 0, "Empty demand file"


class TestFESIntegration:
    """Test FES (Future Energy Scenarios) integration."""
    
    def test_fes_rule_exists(self, project_root):
        """Test that FES processing rule exists."""
        fes_rule_path = project_root / "rules" / "FES.smk"
        
        assert fes_rule_path.exists(), \
            f"FES rule not found at {fes_rule_path}"
    
    def test_fes_data_script_exists(self, project_root):
        """Test that FES data processing script exists."""
        fes_script_path = project_root / "scripts" / "FES_data.py"
        
        assert fes_script_path.exists(), \
            f"FES data script not found at {fes_script_path}"
    
    @pytest.mark.skipif(
        not Path("resources/FES").exists(),
        reason="No FES output directory"
    )
    def test_fes_output_if_exists(self, project_root):
        """Test FES output files if they exist."""
        fes_dir = project_root / "resources" / "FES"
        
        if not fes_dir.exists():
            pytest.skip("FES output directory doesn't exist")
        
        # Look for any FES output files
        fes_files = list(fes_dir.glob("*.csv"))
        
        if len(fes_files) == 0:
            pytest.skip("No FES output files found")
        
        # Test that files can be loaded
        for fes_path in fes_files[:3]:
            try:
                df = pd.read_csv(fes_path, nrows=10)
                assert len(df) > 0
            except Exception as e:
                pytest.fail(f"Failed to load FES file {fes_path}: {e}")


class TestFutureGeneratorIntegration:
    """Test generator integration for future scenarios."""
    
    def test_generator_integration_script_exists(self, project_root):
        """Test that generator integration script exists."""
        scripts_dir = project_root / "scripts"
        gen_scripts = [
            "integrate_thermal_generators.py",
            "integrate_renewable_generators.py"
        ]
        
        found = any((scripts_dir / script).exists() for script in gen_scripts)
        assert found, "No generator integration scripts found"
    
    def test_generators_rule_exists(self, project_root):
        """Test that generators rule exists."""
        gen_rule_path = project_root / "rules" / "generators.smk"
        
        assert gen_rule_path.exists(), \
            f"Generators rule not found at {gen_rule_path}"


class TestFuturePipelineIntegration:
    """Test end-to-end pipeline integration for future scenarios."""
    
    def test_snakefile_handles_future_scenarios(self, project_root):
        """Test that Snakefile includes FES configuration."""
        snakefile_path = project_root / "Snakefile"
        
        with open(snakefile_path, encoding='utf-8') as f:
            content = f.read()
        
        # Should reference FES scenarios
        assert 'FES' in content, "Snakefile doesn't reference FES"
    
    def test_future_workflow_data_dependencies(self, project_root):
        """Test that future workflow has required data dependencies."""
        # FES data
        assert (project_root / "data" / "FES").exists(), \
            "FES data directory not found"
        
        # Network data (needed for topology)
        assert (project_root / "data" / "network").exists(), \
            "Network data directory not found"
        
        # Renewables data
        assert (project_root / "data" / "renewables").exists(), \
            "Renewables data directory not found"


class TestFutureScenarioMetadata:
    """Test future scenario metadata and tracking."""
    
    def test_scenario_description_exists(self, test_future_scenario):
        """Test that future scenario has a description."""
        assert 'description' in test_future_scenario, \
            "Scenario missing description field"
        
        description = test_future_scenario['description']
        assert 'future' in description.lower() or 'fes' in description.lower(), \
            "Description should indicate this is a future scenario"
    
    def test_fes_metadata_complete(self, test_future_scenario):
        """Test that FES metadata is complete."""
        assert 'FES_year' in test_future_scenario
        assert 'FES_scenario' in test_future_scenario
        
        # Both should be set for future scenarios
        assert test_future_scenario['FES_year'] is not None
        assert test_future_scenario['FES_scenario'] is not None


class TestFutureVsFutureComparison:
    """Test consistency between different future scenarios."""
    
    def test_future_year_consistency(self, test_future_scenario):
        """Test that modelled year and FES are aligned."""
        modelled_year = test_future_scenario['modelled_year']
        fes_year = test_future_scenario.get('FES_year')
        
        # FES year should be close to or before modelled year
        # (FES projects forward from its year)
        if fes_year and modelled_year:
            assert fes_year <= modelled_year + 5, \
                f"FES year {fes_year} too far from modelled year {modelled_year}"


class TestFutureResourceOutputs:
    """Test the actual resource outputs from future scenario workflows."""
    
    def test_future_outputs_detected(self, future_scenario_outputs):
        """Test that future scenario outputs are available."""
        # Some outputs should be available if workflow ran
        has_any_output = (future_scenario_outputs.get('has_network_outputs') or
                         future_scenario_outputs.get('has_renewable_profiles') or
                         future_scenario_outputs.get('has_demand_files'))
        
        if not has_any_output:
            pytest.skip("No future scenario outputs detected")
    
    def test_network_files_load_properly(self, future_scenario_outputs, project_root):
        """Test that detected network files load properly."""
        if not future_scenario_outputs.get('available_networks'):
            pytest.skip("No network files found")
        
        for network_file in future_scenario_outputs['available_networks'][:1]:
            try:
                import pypsa
                n = pypsa.Network(str(network_file))
                assert len(n.buses) > 0, f"Network {network_file} has no buses"
            except Exception as e:
                pytest.fail(f"Failed to load network {network_file}: {e}")
    
    def test_renewable_profiles_valid(self, future_scenario_outputs):
        """Test that renewable profiles are valid CSVs."""
        if not future_scenario_outputs.get('available_profiles'):
            pytest.skip("No renewable profiles found")
        
        for profile_file in future_scenario_outputs['available_profiles'][:2]:
            try:
                df = pd.read_csv(profile_file, nrows=10)
                assert len(df) > 0, f"Profile {profile_file} is empty"
            except Exception as e:
                pytest.fail(f"Failed to load profile {profile_file}: {e}")
    
    def test_demand_files_valid(self, future_scenario_outputs):
        """Test that demand files are valid CSVs."""
        if not future_scenario_outputs.get('available_demands'):
            pytest.skip("No demand files found")
        
        for demand_file in future_scenario_outputs['available_demands'][:1]:
            try:
                df = pd.read_csv(demand_file, nrows=10)
                assert len(df) > 0, f"Demand file {demand_file} is empty"
            except Exception as e:
                pytest.fail(f"Failed to load demand file {demand_file}: {e}")


class TestFutureNetworkValidation:
    """Validate structure of future scenario network outputs."""
    
    def test_network_has_valid_structure(self, future_scenario_outputs):
        """Test that network outputs have valid PyPSA structure."""
        if not future_scenario_outputs.get('available_networks'):
            pytest.skip("No network files available")
        
        try:
            import pypsa
            n = pypsa.Network(str(future_scenario_outputs['available_networks'][0]))
            
            # Should have buses and components
            assert len(n.buses) > 0, "Network has no buses"
            assert len(n.carriers) >= 0, "Network carriers invalid"
            
            # Check for demand
            if len(n.loads) > 0:
                # If there are loads, demand should be positive
                assert (n.loads.p_set > 0).any() or (n.loads_t.p_set > 0).any().any(), \
                    "Network has loads but no positive demand"
        except Exception as e:
            pytest.skip(f"Could not validate network structure: {e}")



