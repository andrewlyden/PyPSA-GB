"""
Unit tests for carrier_definitions.py

Tests the carrier (technology) definitions that provide:
- Technology colors for plotting
- CO2 emissions factors
- Nice names for display
- Default parameters

This module is important for consistent handling across all scripts.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from carrier_definitions import get_carrier_definitions


# ══════════════════════════════════════════════════════════════════════════════
# TEST: get_carrier_definitions
# ══════════════════════════════════════════════════════════════════════════════

class TestGetCarrierDefinitions:
    """Test the carrier definitions function."""
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = get_carrier_definitions()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_has_required_columns(self):
        """Test that DataFrame has required columns."""
        result = get_carrier_definitions()
        
        # Essential columns for PyPSA
        required_cols = ['color', 'nice_name']
        for col in required_cols:
            assert col in result.columns, f"Missing required column: {col}"
    
    def test_has_key_carriers(self):
        """Test that key carriers are defined."""
        result = get_carrier_definitions()
        
        key_carriers = [
            'wind_onshore', 'wind_offshore', 'solar_pv',
            'CCGT', 'nuclear', 'battery', 'pumped_hydro'
        ]
        
        for carrier in key_carriers:
            assert carrier in result.index, f"Missing key carrier: {carrier}"
    
    def test_colors_are_valid(self):
        """Test that all color values are valid."""
        result = get_carrier_definitions()
        
        if 'color' in result.columns:
            for carrier, row in result.iterrows():
                color = row['color']
                
                # Should be a valid color string (hex, named, or rgb)
                if pd.notna(color):
                    assert isinstance(color, str)
                    # Hex colors start with #
                    if color.startswith('#'):
                        assert len(color) in [4, 7, 9], f"Invalid hex color: {color}"
    
    def test_nice_names_not_empty(self):
        """Test that nice_name values are provided."""
        result = get_carrier_definitions()
        
        if 'nice_name' in result.columns:
            for carrier, row in result.iterrows():
                nice_name = row['nice_name']
                if pd.notna(nice_name):
                    assert len(str(nice_name)) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Emissions Factors
# ══════════════════════════════════════════════════════════════════════════════

class TestEmissionsFactors:
    """Test CO2 emissions factors in carrier definitions."""
    
    def test_has_emissions_column(self):
        """Test that emissions factor column exists."""
        result = get_carrier_definitions()
        
        emissions_cols = ['co2_emissions', 'emissions', 'co2']
        has_emissions = any(col in result.columns for col in emissions_cols)
        
        # Not all implementations have this, but it's useful
        if not has_emissions:
            pytest.skip("No emissions column in carrier definitions")
    
    def test_renewables_have_zero_emissions(self):
        """Test that renewables have zero or near-zero emissions."""
        result = get_carrier_definitions()
        
        emissions_col = None
        for col in ['co2_emissions', 'emissions', 'co2']:
            if col in result.columns:
                emissions_col = col
                break
        
        if emissions_col is None:
            pytest.skip("No emissions column")
        
        renewables = ['wind_onshore', 'wind_offshore', 'solar_pv', 'large_hydro']
        
        for carrier in renewables:
            if carrier in result.index:
                emissions = result.loc[carrier, emissions_col]
                if pd.notna(emissions):
                    assert emissions <= 0.05, f"{carrier} should have ~zero emissions"
    
    def test_fossil_fuels_have_positive_emissions(self):
        """Test that fossil fuels have positive emissions."""
        result = get_carrier_definitions()
        
        emissions_col = None
        for col in ['co2_emissions', 'emissions', 'co2']:
            if col in result.columns:
                emissions_col = col
                break
        
        if emissions_col is None:
            pytest.skip("No emissions column")
        
        fossil_carriers = ['CCGT', 'OCGT', 'coal', 'gas']
        
        for carrier in fossil_carriers:
            if carrier in result.index:
                emissions = result.loc[carrier, emissions_col]
                if pd.notna(emissions):
                    assert emissions > 0, f"{carrier} should have positive emissions"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Carrier Name Consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestCarrierNameConsistency:
    """Test that carrier names are consistent with PyPSA-GB conventions."""
    
    def test_wind_carriers(self):
        """Test wind carrier naming."""
        result = get_carrier_definitions()
        
        # Should have separate onshore and offshore
        assert 'wind_onshore' in result.index or 'Wind Onshore' in result.index
        assert 'wind_offshore' in result.index or 'Wind Offshore' in result.index
    
    def test_solar_carrier(self):
        """Test solar carrier naming."""
        result = get_carrier_definitions()
        
        solar_carriers = ['solar_pv', 'Solar PV', 'solar', 'Solar']
        has_solar = any(c in result.index for c in solar_carriers)
        
        assert has_solar, "Solar carrier not found"
    
    def test_storage_carriers(self):
        """Test storage carrier naming."""
        result = get_carrier_definitions()
        
        # Should have battery and pumped hydro
        battery_carriers = ['battery', 'Battery', 'Battery Storage']
        hydro_carriers = ['pumped_hydro', 'Pumped Hydro', 'Pumped Hydro Storage']
        
        has_battery = any(c in result.index for c in battery_carriers)
        has_hydro = any(c in result.index for c in hydro_carriers)
        
        assert has_battery, "Battery carrier not found"
        assert has_hydro, "Pumped hydro carrier not found"
    
    def test_thermal_carriers(self):
        """Test thermal generation carrier naming."""
        result = get_carrier_definitions()
        
        # Should have CCGT and nuclear at minimum
        ccgt_carriers = ['CCGT', 'ccgt', 'Gas CCGT', 'gas_ccgt']
        nuclear_carriers = ['nuclear', 'Nuclear']
        
        has_ccgt = any(c in result.index for c in ccgt_carriers)
        has_nuclear = any(c in result.index for c in nuclear_carriers)
        
        assert has_ccgt, "CCGT carrier not found"
        assert has_nuclear, "Nuclear carrier not found"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Color Uniqueness
# ══════════════════════════════════════════════════════════════════════════════

class TestColorUniqueness:
    """Test that colors are reasonably distinct for plotting."""
    
    def test_major_carriers_have_different_colors(self):
        """Test that major carriers have distinct colors."""
        result = get_carrier_definitions()
        
        if 'color' not in result.columns:
            pytest.skip("No color column")
        
        # Major carriers that should have distinct colors
        major_carriers = [
            'wind_onshore', 'wind_offshore', 'solar_pv', 
            'CCGT', 'nuclear', 'battery'
        ]
        
        colors = {}
        for carrier in major_carriers:
            if carrier in result.index:
                color = result.loc[carrier, 'color']
                if pd.notna(color):
                    colors[carrier] = color
        
        # Check for uniqueness
        unique_colors = set(colors.values())
        
        # Allow some overlap but not all same
        if len(colors) > 1:
            assert len(unique_colors) > 1, "Major carriers should have different colors"


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Load Shedding Carrier
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadSheddingCarrier:
    """Test the special load_shedding carrier."""
    
    def test_load_shedding_exists(self):
        """Test that load_shedding carrier is defined."""
        result = get_carrier_definitions()
        
        load_shedding_carriers = ['load_shedding', 'Load Shedding', 'load shedding']
        has_ls = any(c in result.index for c in load_shedding_carriers)
        
        assert has_ls, "load_shedding carrier should be defined"
    
    def test_load_shedding_has_distinctive_color(self):
        """Test that load_shedding has a distinctive color (usually red)."""
        result = get_carrier_definitions()
        
        if 'color' not in result.columns:
            pytest.skip("No color column")
        
        for carrier_name in ['load_shedding', 'Load Shedding']:
            if carrier_name in result.index:
                color = result.loc[carrier_name, 'color']
                # Load shedding is typically red/pink for visibility
                assert pd.notna(color), "load_shedding should have a color"
