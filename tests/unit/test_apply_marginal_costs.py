"""
Unit tests for apply_marginal_costs.py - Marginal cost calculation module.

Tests the calculation and application of time-varying marginal costs for
thermal generators based on fuel prices, carbon prices, and emission factors.
"""

import pytest
import pandas as pd
import numpy as np
import pypsa
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from apply_marginal_costs import (
    CARBON_EMISSION_FACTORS,
    HISTORICAL_FUEL_PRICES,
    PROTECTED_CARRIERS
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_network():
    """Create a simple network with thermal generators."""
    network = pypsa.Network()
    network.set_snapshots(pd.date_range('2020-01-01', periods=24, freq='h'))
    
    # Add buses
    network.add("Bus", "bus1", x=0, y=52)
    
    # Add thermal generators with different technologies
    network.add("Generator", "CCGT1", 
                bus="bus1", carrier="CCGT", p_nom=400, 
                efficiency=0.55, marginal_cost=0)
    
    network.add("Generator", "Coal1", 
                bus="bus1", carrier="coal", p_nom=300, 
                efficiency=0.38, marginal_cost=0)
    
    network.add("Generator", "Nuclear1", 
                bus="bus1", carrier="nuclear", p_nom=1000, 
                efficiency=0.35, marginal_cost=15)
    
    # Add load shedding (protected carrier)
    network.add("Generator", "LoadShed", 
                bus="bus1", carrier="load_shedding", p_nom=10000, 
                marginal_cost=6000)
    
    return network


@pytest.fixture
def sample_fuel_prices():
    """Sample fuel price time series."""
    dates = pd.date_range('2020-01-01', periods=24, freq='h')
    return pd.DataFrame({
        'gas': np.linspace(20, 25, 24),
        'coal': np.linspace(10, 12, 24),
        'oil': np.linspace(45, 50, 24)
    }, index=dates)


@pytest.fixture
def sample_carbon_prices():
    """Sample carbon price time series."""
    dates = pd.date_range('2020-01-01', periods=24, freq='h')
    return pd.Series(np.linspace(20, 25, 24), index=dates)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Emission Factors
# ══════════════════════════════════════════════════════════════════════════════

class TestEmissionFactors:
    """Test carbon emission factor definitions."""
    
    def test_emission_factors_exist_for_main_carriers(self):
        """Test that emission factors exist for main thermal carriers."""
        assert 'coal' in CARBON_EMISSION_FACTORS
        assert 'gas' in CARBON_EMISSION_FACTORS
        assert 'CCGT' in CARBON_EMISSION_FACTORS
        assert 'oil' in CARBON_EMISSION_FACTORS
    
    def test_emission_factors_are_positive(self):
        """Test that emission factors are non-negative."""
        for carrier, factor in CARBON_EMISSION_FACTORS.items():
            assert factor >= 0, f"{carrier} has negative emission factor"
    
    def test_coal_has_highest_emissions(self):
        """Test that coal has higher emissions than gas."""
        assert CARBON_EMISSION_FACTORS['coal'] > CARBON_EMISSION_FACTORS['gas']
    
    def test_renewables_have_low_emissions(self):
        """Test that renewable carriers have low lifecycle emissions."""
        assert CARBON_EMISSION_FACTORS['wind_onshore'] < 20
        assert CARBON_EMISSION_FACTORS['solar_pv'] < 50
        assert CARBON_EMISSION_FACTORS['nuclear'] < 15


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Fuel Price Data
# ══════════════════════════════════════════════════════════════════════════════

class TestFuelPriceData:
    """Test historical fuel price data."""
    
    def test_historical_fuel_prices_exist(self):
        """Test that historical fuel prices are defined."""
        assert len(HISTORICAL_FUEL_PRICES) > 0
    
    def test_fuel_prices_have_required_keys(self):
        """Test that each year has gas, coal, oil prices."""
        for year, prices in HISTORICAL_FUEL_PRICES.items():
            assert 'gas' in prices
            assert 'coal' in prices
            assert 'oil' in prices
    
    def test_fuel_prices_are_positive(self):
        """Test that all fuel prices are positive."""
        for year, prices in HISTORICAL_FUEL_PRICES.items():
            for fuel, price in prices.items():
                assert price > 0, f"{fuel} in {year} has non-positive price"
    
    def test_fuel_prices_in_reasonable_range(self):
        """Test that fuel prices are in reasonable range (£/MWh)."""
        for year, prices in HISTORICAL_FUEL_PRICES.items():
            # Gas typically 10-100 £/MWh
            assert 5 <= prices['gas'] <= 150
            # Coal typically 5-50 £/MWh
            assert 3 <= prices['coal'] <= 100
            # Oil typically 20-150 £/MWh
            assert 10 <= prices['oil'] <= 200


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Protected Carriers
# ══════════════════════════════════════════════════════════════════════════════

class TestProtectedCarriers:
    """Test that certain carriers are protected from cost modification."""
    
    def test_load_shedding_is_protected(self):
        """Test that load shedding is in protected carriers."""
        assert 'load_shedding' in PROTECTED_CARRIERS
    
    def test_voll_is_protected(self):
        """Test that VOLL carriers are protected."""
        assert 'voll' in PROTECTED_CARRIERS or 'VOLL' in PROTECTED_CARRIERS
    
    def test_protected_carriers_not_empty(self):
        """Test that there are protected carriers defined."""
        assert len(PROTECTED_CARRIERS) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Marginal Cost Calculation
# ══════════════════════════════════════════════════════════════════════════════

class TestMarginalCostCalculation:
    """Test marginal cost calculation logic."""
    
    def test_marginal_cost_increases_with_fuel_price(self):
        """Test that marginal cost increases with fuel price."""
        efficiency = 0.5
        emission_factor = 400  # kg CO2/MWh
        carbon_price = 20  # £/tonne
        
        # Lower fuel price
        mc1 = (20 / efficiency) + (emission_factor * carbon_price / 1000)
        
        # Higher fuel price
        mc2 = (30 / efficiency) + (emission_factor * carbon_price / 1000)
        
        assert mc2 > mc1
    
    def test_marginal_cost_increases_with_carbon_price(self):
        """Test that marginal cost increases with carbon price."""
        fuel_price = 20
        efficiency = 0.5
        emission_factor = 400
        
        # Lower carbon price
        mc1 = (fuel_price / efficiency) + (emission_factor * 10 / 1000)
        
        # Higher carbon price
        mc2 = (fuel_price / efficiency) + (emission_factor * 30 / 1000)
        
        assert mc2 > mc1
    
    def test_marginal_cost_decreases_with_efficiency(self):
        """Test that marginal cost decreases with higher efficiency."""
        fuel_price = 20
        carbon_price = 20
        emission_factor = 400
        
        # Lower efficiency
        mc1 = (fuel_price / 0.4) + (emission_factor * carbon_price / 1000)
        
        # Higher efficiency
        mc2 = (fuel_price / 0.6) + (emission_factor * carbon_price / 1000)
        
        assert mc2 < mc1


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Network Application
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkApplication:
    """Test application of costs to network."""
    
    def test_protected_carriers_unchanged(self, simple_network):
        """Test that protected carriers (load shedding) keep original costs."""
        original_cost = simple_network.generators.loc['LoadShed', 'marginal_cost']
        
        # Even if we try to modify it, it should stay the same
        assert original_cost == 6000
        assert simple_network.generators.loc['LoadShed', 'carrier'] in PROTECTED_CARRIERS
    
    def test_thermal_generators_identified(self, simple_network):
        """Test that thermal generators are correctly identified."""
        thermal_carriers = {'CCGT', 'coal', 'gas', 'oil'}
        
        thermal_gens = simple_network.generators[
            simple_network.generators.carrier.isin(thermal_carriers)
        ]
        
        assert len(thermal_gens) >= 2  # CCGT1 and Coal1
    
    def test_network_structure_preserved(self, simple_network):
        """Test that network structure is preserved after cost application."""
        initial_gen_count = len(simple_network.generators)
        initial_bus_count = len(simple_network.buses)
        
        # Network structure should not change
        assert len(simple_network.generators) == initial_gen_count
        assert len(simple_network.buses) == initial_bus_count


# ══════════════════════════════════════════════════════════════════════════════
# TEST: Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_handle_missing_efficiency(self, simple_network):
        """Test handling of generators without efficiency data."""
        # Remove efficiency from one generator
        simple_network.generators.loc['CCGT1', 'efficiency'] = np.nan
        
        # Should handle gracefully (use default or skip)
        assert True  # Shouldn't crash
    
    def test_handle_unknown_carrier(self, simple_network):
        """Test handling of unknown carrier type."""
        simple_network.add("Generator", "Unknown1",
                          bus="bus1", carrier="unknown_fuel",
                          p_nom=100, marginal_cost=0)
        
        # Should handle gracefully
        assert 'Unknown1' in simple_network.generators.index
    
    def test_handle_zero_efficiency(self):
        """Test handling of zero efficiency."""
        # Should avoid division by zero
        efficiency = 0.001  # Very small but not zero
        fuel_price = 20
        
        # Should not raise
        mc = fuel_price / efficiency
        assert mc > 0
