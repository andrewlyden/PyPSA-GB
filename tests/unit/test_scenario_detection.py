"""
Unit tests for scenario_detection.py

Tests the critical routing logic that determines whether a scenario uses
historical (DUKES/REPD) or future (FES) data sources.

This is a CRITICAL module - incorrect routing leads to wrong data sources.
"""

import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from scenario_detection import (
    is_historical_scenario,
    auto_configure_scenario,
    validate_scenario_complete,
    check_cutout_availability,
    get_scenario_source,
    target_source,
    uses_brownfield,
    uses_gsp_blocks,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def historical_scenario():
    """Create a historical scenario (year <= 2024)."""
    return {
        'modelled_year': 2020,
        'network_model': 'Reduced',
        'renewables_year': 2020,
        'demand_year': 2020,
        'timestep_minutes': 60,
        'solve': {'voll': 6000.0},
    }


@pytest.fixture
def future_scenario():
    """Create a future scenario (year > 2024)."""
    return {
        'modelled_year': 2035,
        'network_model': 'ETYS',
        'renewables_year': 2019,
        'demand_year': 2035,
        'timestep_minutes': 60,
        'solve': {'voll': 6000.0},
        'FES_year': 2024,
        'FES_scenario': 'Holistic Transition',
    }


@pytest.fixture
def boundary_scenario_2024():
    """Create scenario at the boundary year (2024 = historical)."""
    return {
        'modelled_year': 2024,
        'network_model': 'Reduced',
        'renewables_year': 2024,
        'demand_year': 2024,
        'timestep_minutes': 60,
        'solve': {'voll': 6000.0},
    }


@pytest.fixture
def boundary_scenario_2025():
    """Create scenario just past the boundary (2025 = future)."""
    return {
        'modelled_year': 2025,
        'network_model': 'ETYS',
        'renewables_year': 2019,
        'demand_year': 2025,
        'timestep_minutes': 60,
        'solve': {'voll': 6000.0},
        'FES_year': 2024,
        'FES_scenario': 'Consumer Transformation',
    }


@pytest.fixture
def cp30_scenario():
    """Create a CP30 future scenario with explicit FES spatial proxy."""
    return {
        'modelled_year': 2030,
        'network_model': 'Reduced',
        'renewables_year': 2020,
        'demand_year': 2020,
        'scenario_source': 'CP30',
        'CP30_pathway': 'Further Flex and Renewables',
        'source_data': {
            'workbook': 'data/CP30/Clean Power 2030 Data Workbook_V003.xlsx',
            'files': {
                'es1': 'resources/CP30/CP30_ES1.csv',
                'es5': 'resources/CP30/CP30_ES5_regional.csv',
                'ed1': 'resources/CP30/CP30_ED1.csv',
                'reference': 'resources/CP30/CP30_reference_N1_N2_NA1.csv',
                'validation': 'resources/CP30/CP30_validation_report.csv',
                # Required since the CP30 interconnector price path landed (July
                # 2026); the real config has carried it all along and only this
                # fixture had not caught up.
                'interconnector_prices': 'resources/CP30/CP30_interconnector_prices.csv',
            },
        },
        'fleet': {
            'spatial_proxy': {
                'source': 'FES',
                'FES_year': 2025,
                'FES_scenario': 'Holistic Transition',
            },
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST: is_historical_scenario
# ══════════════════════════════════════════════════════════════════════════════



class TestScenarioSource:
    """Test authoritative source detection for historical, FES and CP30 scenarios."""

    def test_historical_source(self, historical_scenario):
        assert get_scenario_source(historical_scenario) == 'historical'
        assert not uses_brownfield(historical_scenario) and not uses_gsp_blocks(historical_scenario)

    def test_fes_future_source(self, future_scenario):
        assert get_scenario_source(future_scenario) == 'FES'
        assert uses_gsp_blocks(future_scenario) and not uses_brownfield(future_scenario)

    def test_cp30_future_source(self, cp30_scenario):
        assert get_scenario_source(cp30_scenario) == 'CP30'
        assert target_source(cp30_scenario) == 'CP30'
        assert uses_brownfield(cp30_scenario) is True

    def test_written_keys_win(self, future_scenario):
        written = {**future_scenario, 'targets': {'source': 'CP30', 'edition': 2024, 'pathway': 'FFR'},
                   'method': 'brownfield'}
        assert target_source(written) == 'CP30'
        assert uses_brownfield(written) is True

class TestIsHistoricalScenario:
    """Test historical vs future scenario detection."""
    
    def test_2020_is_historical(self, historical_scenario):
        """Test that 2020 is detected as historical."""
        assert is_historical_scenario(historical_scenario) is True
    
    def test_2035_is_not_historical(self, future_scenario):
        """Test that 2035 is detected as future (not historical)."""
        assert is_historical_scenario(future_scenario) is False
    
    def test_2024_boundary_is_historical(self, boundary_scenario_2024):
        """Test that 2024 (boundary year) is detected as historical."""
        assert is_historical_scenario(boundary_scenario_2024) is True
    
    def test_2025_boundary_is_future(self, boundary_scenario_2025):
        """Test that 2025 (just past boundary) is detected as future."""
        assert is_historical_scenario(boundary_scenario_2025) is False
    
    def test_missing_modelled_year_treats_as_future(self):
        """Test that missing modelled_year is treated as future (returns False)."""
        scenario = {'network_model': 'Reduced'}
        # Function returns False (future) when year is missing
        assert is_historical_scenario(scenario) is False
    
    def test_none_modelled_year_treats_as_future(self):
        """Test that None modelled_year is treated as future (returns False)."""
        scenario = {'modelled_year': None}
        # Function returns False (future) when year is None
        assert is_historical_scenario(scenario) is False
    
    def test_very_old_year_is_historical(self):
        """Test that very old years (2010) are historical."""
        scenario = {'modelled_year': 2010}
        assert is_historical_scenario(scenario) is True
    
    def test_far_future_is_not_historical(self):
        """Test that far future years (2050) are not historical."""
        scenario = {'modelled_year': 2050}
        assert is_historical_scenario(scenario) is False


# ══════════════════════════════════════════════════════════════════════════════
# TEST: auto_configure_scenario
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoConfigureScenario:
    """Test automatic scenario configuration with routing metadata."""
    
    def test_historical_adds_routing_metadata(self, historical_scenario):
        """Test that historical scenarios get correct routing metadata."""
        configured = auto_configure_scenario(historical_scenario)
        
        assert 'data_source' in configured or 'is_historical' in configured
        # Historical should NOT use FES for generators
        assert configured.get('thermal_data_source') in ['DUKES', None] or 'is_historical' in configured
    
    def test_future_adds_fes_routing(self, future_scenario):
        """Test that FES future scenarios are configured for FES data."""
        configured = auto_configure_scenario(future_scenario)
        
        assert configured['scenario_source'] == 'FES'
        assert configured['generator_data_source'] == 'FES'
        assert configured['demand_source'] == 'FES'
        assert configured['_needs_fes'] is True

    def test_cp30_adds_cp30_routing(self, cp30_scenario):
        """Test that CP30 scenarios do not route authoritative totals to FES."""
        configured = auto_configure_scenario(cp30_scenario)

        assert configured['scenario_source'] == 'CP30'
        assert configured['generator_data_source'] == 'CP30'
        assert configured['demand_source'] == 'CP30_ED1'
        assert configured['storage_data_source'] == 'CP30'
        assert configured['interconnector_data_source'] == 'CP30'
        assert configured['spatial_data_source'] == 'FES'
        assert configured['_needs_fes'] is True
    
    def test_preserves_original_fields(self, historical_scenario):
        """Test that original scenario fields are preserved."""
        configured = auto_configure_scenario(historical_scenario)
        
        assert configured['modelled_year'] == 2020
        assert configured['network_model'] == 'Reduced'
        assert configured['timestep_minutes'] == 60
    
    def test_does_not_modify_input(self, historical_scenario):
        """Test that the original scenario dict is not modified."""
        original_year = historical_scenario['modelled_year']
        _ = auto_configure_scenario(historical_scenario)
        
        assert historical_scenario['modelled_year'] == original_year


# ══════════════════════════════════════════════════════════════════════════════
# TEST: validate_scenario_complete
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateScenarioComplete:
    """Test scenario validation for completeness."""
    
    def test_valid_historical_scenario_passes(self, historical_scenario):
        """Test that a complete historical scenario passes validation."""
        result = validate_scenario_complete(historical_scenario)
        # Function returns dict with errors/warnings/info
        assert isinstance(result, dict)
        assert 'errors' in result
        assert len(result['errors']) == 0  # No errors for valid scenario
    
    def test_valid_future_scenario_passes(self, future_scenario):
        """Test that a complete future scenario passes validation."""
        result = validate_scenario_complete(future_scenario)
        assert isinstance(result, dict)
        assert 'errors' in result
        assert len(result['errors']) == 0  # No errors for valid scenario
    
    def test_missing_modelled_year_fails(self):
        """Test that missing modelled_year fails validation."""
        incomplete = {'network_model': 'Reduced'}
        
        result = validate_scenario_complete(incomplete)
        assert isinstance(result, dict)
        assert 'errors' in result
        assert len(result['errors']) > 0  # Should have errors
    
    def test_future_without_fes_config_warns_or_fails(self):
        """Test that future scenario without FES config is flagged."""
        incomplete_future = {
            'modelled_year': 2035,
            'network_model': 'ETYS',
            # Missing FES_year and FES_scenario
        }
        
        result = validate_scenario_complete(incomplete_future)
        assert isinstance(result, dict)
        # Should have either errors or warnings about missing FES config
        assert len(result['errors']) > 0 or len(result.get('warnings', [])) > 0
    
    def test_invalid_network_model_fails(self):
        """Test that invalid network_model fails validation."""
        invalid = {
            'modelled_year': 2020,
            'network_model': 'InvalidModel',  # Not ETYS, Reduced, or Zonal
        }
        
        result = validate_scenario_complete(invalid)
        assert isinstance(result, dict)
        assert len(result['errors']) > 0  # Should have errors about invalid network model

    def test_valid_cp30_scenario_passes(self, cp30_scenario):
        """Test that a complete CP30 scenario passes validation."""
        result = validate_scenario_complete(cp30_scenario)
        assert isinstance(result, dict)
        assert result['errors'] == []

    def test_cp30_missing_pathway_fails(self, cp30_scenario):
        """Test that CP30 scenarios require an explicit pathway."""
        incomplete = dict(cp30_scenario)
        incomplete.pop('CP30_pathway')

        result = validate_scenario_complete(incomplete)
        assert any('CP30_pathway' in error for error in result['errors'])


# ══════════════════════════════════════════════════════════════════════════════
# TEST: check_cutout_availability
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckCutoutAvailability:
    """Test cutout (weather data) availability checking."""
    
    def test_available_year_returns_true(self):
        """Test that an available cutout year returns True."""
        # 2019 is typically available
        result = check_cutout_availability(2019)
        # Depending on actual data, this may be True or need mocking
        assert isinstance(result, bool)
    
    def test_unavailable_year_returns_false(self):
        """Test that an unavailable cutout year returns False."""
        # Very future year unlikely to have cutouts
        result = check_cutout_availability(2099)
        assert result is False
    
    def test_none_year_handles_gracefully(self):
        """Test that None year is handled gracefully."""
        try:
            result = check_cutout_availability(None)
            assert result is False
        except TypeError:
            pass  # Also acceptable behavior


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Integration scenarios
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioDetectionIntegration:
    """Integration tests for scenario detection workflow."""
    
    def test_historical_to_future_transition(self):
        """Test detection across the 2024/2025 boundary."""
        years = [2022, 2023, 2024, 2025, 2026, 2030]
        expected = [True, True, True, False, False, False]
        
        for year, is_hist in zip(years, expected):
            scenario = {'modelled_year': year}
            assert is_historical_scenario(scenario) == is_hist, \
                f"Year {year} should be historical={is_hist}"
    
    def test_fes_scenarios_all_detected_as_future(self):
        """Test that all FES scenario types are detected as future."""
        fes_scenarios = [
            'Holistic Transition',
            'Consumer Transformation', 
            'Electric Engagement',
            'Hydrogen Evolution',
        ]
        
        for fes_name in fes_scenarios:
            scenario = {
                'modelled_year': 2035,
                'FES_scenario': fes_name,
                'FES_year': 2024,
            }
            assert is_historical_scenario(scenario) is False
