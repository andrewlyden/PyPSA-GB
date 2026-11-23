"""
Deep Network Validation for PyPSA-GB

Comprehensive validation to prevent:
- Unbounded optimization (infinite cheap imports/generation)
- Infeasibility (demand > supply with no backup)
- Slow solves (poor problem conditioning, unnecessary complexity)
- Data errors (NaN, negative values, inconsistent time series)

Validates: Demand → Renewables → Thermal → Storage → Interconnectors → Clustering
"""

import pypsa
import pandas as pd
import numpy as np
import logging
from pathlib import Path


def validate_demand(network, logger):
    """Validate demand/load configuration."""
    logger.info("")
    logger.info("1️⃣  DEMAND VALIDATION")
    logger.info("-" * 80)
    
    errors = []
    warnings = []
    
    if len(network.loads) == 0:
        errors.append("No loads found in network")
        logger.error("❌ CRITICAL: No loads found in network!")
        return errors, warnings
    
    logger.info(f"✓ Found {len(network.loads):,} load buses")
    
    # Check for p_set time series
    if 'p_set' not in network.loads_t or len(network.loads_t.p_set.columns) == 0:
        errors.append("No load time series (p_set) found")
        logger.error("❌ CRITICAL: No load time series (p_set) found!")
        return errors, warnings
    
    demand_ts = network.loads_t.p_set
    total_demand = demand_ts.sum().sum()
    peak_demand = demand_ts.sum(axis=1).max()
    min_demand = demand_ts.sum(axis=1).min()
    
    # Check for valid demand values
    if total_demand <= 0:
        errors.append(f"Total demand is zero or negative: {total_demand:.0f} MWh")
        logger.error(f"❌ CRITICAL: Total demand ≤ 0: {total_demand:.0f} MWh")
    elif total_demand < 100_000_000:  # <100 TWh for full year is suspicious
        warnings.append(f"Total demand low: {total_demand/1e6:.1f} TWh (expected ~300 TWh/yr for GB)")
        logger.warning(f"⚠ Total demand low: {total_demand/1e6:.1f} TWh")
    else:
        logger.info(f"  ✓ Total demand: {total_demand/1e6:.1f} TWh")
        logger.info(f"  ✓ Peak demand: {peak_demand:,.0f} MW")
        logger.info(f"  ✓ Min demand: {min_demand:,.0f} MW")
    
    # Check for NaN or negative values
    nan_count = demand_ts.isna().sum().sum()
    negative_count = (demand_ts < 0).sum().sum()
    
    if nan_count > 0:
        errors.append(f"{nan_count:,} NaN values in demand")
        logger.error(f"❌ CRITICAL: {nan_count:,} NaN values in demand")
    
    if negative_count > 0:
        errors.append(f"{negative_count:,} negative demand values")
        logger.error(f"❌ CRITICAL: {negative_count:,} negative demand values")
    
    # Check time series alignment
    if len(demand_ts) != len(network.snapshots):
        errors.append(f"Demand length ({len(demand_ts)}) != snapshots ({len(network.snapshots)})")
        logger.error(f"❌ CRITICAL: Demand/snapshot length mismatch")
    
    return errors, warnings


def validate_renewables(network, logger):
    """Validate renewable generation configuration."""
    logger.info("")
    logger.info("2️⃣  RENEWABLE GENERATION VALIDATION")
    logger.info("-" * 80)
    
    errors = []
    warnings = []
    
    renewable_carriers = ['wind_onshore', 'wind_offshore', 'solar_pv', 'Solar',
                         'Wind (Onshore)', 'Wind (Offshore)', 'tidal_stream',
                         'shoreline_wave', 'large_hydro', 'small_hydro']
    
    renewable_gens = network.generators[network.generators.carrier.isin(renewable_carriers)]
    
    if len(renewable_gens) == 0:
        warnings.append("No renewable generators found")
        logger.warning("⚠ No renewable generators (unusual for GB)")
        return errors, warnings
    
    logger.info(f"✓ Found {len(renewable_gens):,} renewable generators")
    logger.info(f"  Total capacity: {renewable_gens.p_nom.sum():,.0f} MW")
    
    # Check for p_max_pu time series
    if 'p_max_pu' not in network.generators_t or len(network.generators_t.p_max_pu.columns) == 0:
        warnings.append("No renewable p_max_pu time series")
        logger.warning("⚠ No p_max_pu profiles (will use static availability)")
        return errors, warnings
    
    renewable_cols = [col for col in network.generators_t.p_max_pu.columns 
                     if col in renewable_gens.index]
    
    if len(renewable_cols) == 0:
        warnings.append("Renewables have no p_max_pu profiles")
        logger.warning("⚠ Renewables exist but no p_max_pu profiles")
        return errors, warnings
    
    p_max_pu = network.generators_t.p_max_pu[renewable_cols]
    
    # Check for invalid values
    invalid_low = (p_max_pu < 0).sum().sum()
    invalid_high = (p_max_pu > 1.0001).sum().sum()
    nan_count = p_max_pu.isna().sum().sum()
    
    if invalid_low > 0:
        errors.append(f"{invalid_low:,} negative p_max_pu values")
        logger.error(f"❌ CRITICAL: {invalid_low:,} negative p_max_pu")
    
    if invalid_high > 0:
        errors.append(f"{invalid_high:,} p_max_pu values >1.0")
        logger.error(f"❌ CRITICAL: {invalid_high:,} p_max_pu >1.0")
    
    if nan_count > 0:
        warnings.append(f"{nan_count:,} NaN p_max_pu values")
        logger.warning(f"⚠ {nan_count:,} NaN p_max_pu (will use p_nom)")
    
    # Check for constant profiles (data issue indicator)
    constant_count = 0
    for col in renewable_cols[:10]:
        profile = p_max_pu[col].dropna()
        if len(profile) > 0 and profile.nunique() <= 1:
            constant_count += 1
    
    if constant_count > 0:
        warnings.append(f"{constant_count} generators with constant p_max_pu")
        logger.warning(f"⚠ {constant_count} generators have constant profiles")
    
    avg_cf = p_max_pu.mean().mean()
    logger.info(f"  ✓ Avg renewable capacity factor: {avg_cf:.1%}")
    logger.info(f"  ✓ Profiles for {len(renewable_cols):,} generators")
    
    return errors, warnings


def validate_thermal_generators(network, logger):
    """Validate thermal/dispatchable generation."""
    logger.info("")
    logger.info("3️⃣  THERMAL GENERATION VALIDATION")
    logger.info("-" * 80)
    
    errors = []
    warnings = []
    
    thermal_carriers = ['CCGT', 'OCGT', 'coal', 'Conventional steam', 'nuclear',
                       'PWR', 'AGR', 'biomass', 'biogas', 'waste_to_energy']
    
    thermal_gens = network.generators[network.generators.carrier.isin(thermal_carriers)]
    
    if len(thermal_gens) == 0:
        warnings.append("No thermal generators")
        logger.warning("⚠ No thermal generators (renewable-only system)")
        return errors, warnings
    
    logger.info(f"✓ Found {len(thermal_gens):,} thermal generators")
    logger.info(f"  Total capacity: {thermal_gens.p_nom.sum():,.0f} MW")
    
    # Check for unbounded capacity
    unbounded = thermal_gens[
        (thermal_gens.p_nom.isna()) |
        (thermal_gens.p_nom == 0) |
        (thermal_gens.p_nom > 100_000)
    ]
    if len(unbounded) > 0:
        errors.append(f"{len(unbounded)} thermal gens with invalid capacity")
        logger.error(f"❌ CRITICAL: {len(unbounded)} with invalid p_nom")
        for idx in unbounded.index[:3]:
            gen = thermal_gens.loc[idx]
            logger.error(f"     {idx}: p_nom={gen.p_nom}, carrier={gen.carrier}")
    
    # Check marginal costs
    no_cost = thermal_gens[thermal_gens.marginal_cost.isna() | (thermal_gens.marginal_cost == 0)]
    if len(no_cost) > 0:
        warnings.append(f"{len(no_cost)} thermal gens with zero cost")
        logger.warning(f"⚠ {len(no_cost)} thermal gens with zero cost")
    
    very_cheap = thermal_gens[(thermal_gens.marginal_cost > 0) & (thermal_gens.marginal_cost < 5)]
    if len(very_cheap) > 0:
        warnings.append(f"{len(very_cheap)} thermal gens <£5/MWh")
        logger.warning(f"⚠ {len(very_cheap)} thermal gens very cheap (<£5/MWh)")
    
    avg_cost = thermal_gens.marginal_cost.mean()
    logger.info(f"  ✓ Avg marginal cost: £{avg_cost:.1f}/MWh")
    
    return errors, warnings


def validate_load_shedding(network, logger):
    """Validate load shedding backup (VOLL)."""
    logger.info("")
    logger.info("4️⃣  LOAD SHEDDING VALIDATION")
    logger.info("-" * 80)
    
    errors = []
    warnings = []
    
    ls_gens = network.generators[network.generators.carrier == 'load_shedding']
    
    if len(ls_gens) == 0:
        warnings.append("No load shedding backup")
        logger.warning("⚠ No load shedding (may be infeasible)")
        return errors, warnings
    
    logger.info(f"✓ Found {len(ls_gens):,} load shedding generators")
    
    total_ls = ls_gens.p_nom.sum()
    avg_voll = ls_gens.marginal_cost.mean()
    
    if avg_voll < 1000:
        warnings.append(f"Low VOLL: £{avg_voll:.0f}/MWh")
        logger.warning(f"⚠ VOLL low: £{avg_voll:.0f}/MWh (expected >£1000/MWh)")
    else:
        logger.info(f"  ✓ VOLL: £{avg_voll:,.0f}/MWh")
    
    # Check capacity vs peak demand
    if 'p_set' in network.loads_t and len(network.loads_t.p_set.columns) > 0:
        peak_demand = network.loads_t.p_set.sum(axis=1).max()
        if total_ls < peak_demand:
            errors.append(f"Load shedding ({total_ls:.0f} MW) < peak demand ({peak_demand:.0f} MW)")
            logger.error(f"❌ CRITICAL: Insufficient load shedding capacity")
        else:
            logger.info(f"  ✓ Capacity: {total_ls:,.0f} MW (covers {peak_demand:,.0f} MW peak)")
    
    return errors, warnings


def validate_storage(network, logger):
    """Validate storage units."""
    logger.info("")
    logger.info("5️⃣  STORAGE VALIDATION")
    logger.info("-" * 80)
    
    errors = []
    warnings = []
    
    if len(network.storage_units) == 0:
        logger.info("ℹ️  No storage units")
        return errors, warnings
    
    logger.info(f"✓ Found {len(network.storage_units):,} storage units")
    
    # Check power capacity
    invalid_power = network.storage_units[
        (network.storage_units.p_nom.isna()) |
        (network.storage_units.p_nom <= 0) |
        (network.storage_units.p_nom > 50_000)
    ]
    if len(invalid_power) > 0:
        errors.append(f"{len(invalid_power)} storage with invalid power")
        logger.error(f"❌ CRITICAL: {len(invalid_power)} invalid p_nom")
    
    # Check efficiencies
    high_eff = network.storage_units[
        (network.storage_units.efficiency_store > 1.0) |
        (network.storage_units.efficiency_dispatch > 1.0)
    ]
    if len(high_eff) > 0:
        errors.append(f"{len(high_eff)} storage with efficiency >100%")
        logger.error(f"❌ CRITICAL: {len(high_eff)} efficiency >100%")
    
    total_power = network.storage_units.p_nom.sum()
    total_energy = (network.storage_units.p_nom * network.storage_units.max_hours).sum()
    logger.info(f"  ✓ Total power: {total_power:,.0f} MW")
    logger.info(f"  ✓ Total energy: {total_energy:,.0f} MWh")
    
    return errors, warnings


def validate_interconnectors(network, scenario_config, logger):
    """Validate interconnectors - CRITICAL for unbounded prevention."""
    logger.info("")
    logger.info("6️⃣  INTERCONNECTOR VALIDATION (Unbounded Risk)")
    logger.info("-" * 80)
    
    errors = []
    warnings = []
    
    if len(network.links) == 0:
        logger.info("ℹ️  No links found")
        return errors, warnings
    
    ic_links = network.links[network.links.index.str.startswith('IC_')]
    
    if len(ic_links) == 0:
        logger.info("ℹ️  No interconnector links")
        return errors, warnings
    
    logger.info(f"✓ Found {len(ic_links)} interconnector links")
    logger.info(f"  Total capacity: {ic_links.p_nom.sum():,.0f} MW")
    
    # Check if historical (fixed) or future (optimizable)
    is_historical = False
    has_p_set = 'p_set' in network.links_t and len(network.links_t.p_set.columns) > 0
    
    if has_p_set:
        p_set_ic = network.links_t.p_set[[col for col in network.links_t.p_set.columns 
                                          if col.startswith('IC_')]]
        if len(p_set_ic.columns) > 0:
            nonzero_flows = (p_set_ic != 0).sum().sum()
            if nonzero_flows > 0:
                is_historical = True
                logger.info(f"  ✓ Historical: FIXED flows ({nonzero_flows:,} timesteps)")
    
    if not is_historical:
        logger.info("  ✓ Future/optimizable scenario")
        
        # CRITICAL: Check for external generators
        external_buses = network.buses[network.buses.get('country', '') != 'GB']
        
        if len(external_buses) == 0:
            errors.append("No external buses for interconnectors")
            logger.error("❌ CRITICAL: No external buses!")
        else:
            logger.info(f"     ✓ {len(external_buses)} external buses")
            
            external_gens = network.generators[network.generators.bus.isin(external_buses.index)]
            
            if len(external_gens) == 0:
                errors.append("UNBOUNDED: No external generators!")
                logger.error("❌ CRITICAL UNBOUNDED: No external generators!")
                logger.error("   Future ICs need external gens to prevent unbounded")
            else:
                logger.info(f"     ✓ {len(external_gens)} external gens (prevents unbounded)")
                
                eu_gens = external_gens[external_gens.carrier == 'EU_import']
                if len(eu_gens) > 0:
                    avg_cost = eu_gens.marginal_cost.mean()
                    logger.info(f"     ✓ External cost: £{avg_cost:.1f}/MWh")
    
    # Check link costs (should be low)
    high_cost = ic_links[ic_links.marginal_cost > 5.0]
    if len(high_cost) > 0:
        warnings.append(f"{len(high_cost)} ICs with high link cost")
        logger.warning(f"⚠ {len(high_cost)} ICs with cost >£5/MWh")
    
    return errors, warnings


def validate_network_consistency(network, logger):
    """Validate network connectivity and consistency."""
    logger.info("")
    logger.info("7️⃣  NETWORK CONSISTENCY")
    logger.info("-" * 80)
    
    errors = []
    warnings = []
    
    # Check bus connections (include lines, links, AND transformers)
    buses_with_gens = set(network.generators.bus.unique())
    buses_with_loads = set(network.loads.bus.unique())
    buses_with_storage = set(network.storage_units.bus.unique()) if len(network.storage_units) > 0 else set()
    
    # Buses in topology: lines + links + transformers
    buses_in_lines = set(network.lines.bus0.unique()) | set(network.lines.bus1.unique())
    buses_in_links = set(network.links.bus0.unique()) | set(network.links.bus1.unique()) if len(network.links) > 0 else set()
    buses_in_transformers = set()
    if len(network.transformers) > 0:
        buses_in_transformers = set(network.transformers.bus0.unique()) | set(network.transformers.bus1.unique())
    
    buses_in_topology = buses_in_lines | buses_in_links | buses_in_transformers
    
    # Connected: any bus with a component OR in the topology
    connected = buses_with_gens | buses_with_loads | buses_with_storage | buses_in_topology
    isolated = set(network.buses.index) - connected
    
    # CRITICAL: Check for orphaned external buses (interconnector endpoints with no links)
    # These can break optimization and should have been removed during interconnector integration
    if len(isolated) > 0:
        external_isolated = []
        gb_isolated = []
        
        for bus in isolated:
            bus_data = network.buses.loc[bus]
            if bus_data.get('country', 'GB') != 'GB':
                external_isolated.append(bus)
            else:
                gb_isolated.append(bus)
        
        if external_isolated:
            errors.append(f"{len(external_isolated)} orphaned external buses (interconnector endpoints with no links)")
            logger.error(f"❌ CRITICAL: {len(external_isolated)} orphaned external buses found!")
            logger.error("   These are interconnector endpoints that have no links attached.")
            logger.error("   This indicates a bug in interconnector integration - buses should have been removed.")
            for bus in external_isolated[:5]:
                bus_country = network.buses.loc[bus, 'country']
                logger.error(f"     - {bus} (country: {bus_country})")
            if len(external_isolated) > 5:
                logger.error(f"     ... and {len(external_isolated) - 5} more")
            logger.error("   FIX: Re-run interconnector integration to remove orphaned buses")
        
        if gb_isolated:
            warnings.append(f"{len(gb_isolated)} isolated GB buses")
            logger.warning(f"⚠ {len(gb_isolated)} isolated GB buses (no generators, loads, storage, or connections)")
            for bus in gb_isolated[:5]:
                logger.warning(f"  - {bus}")
            if len(gb_isolated) > 5:
                logger.warning(f"  ... and {len(gb_isolated) - 5} more")
    else:
        logger.info("✓ All buses connected")
    
    logger.info(f"  Buses with generators: {len(buses_with_gens):,}")
    logger.info(f"  Buses with loads: {len(buses_with_loads):,}")
    logger.info(f"  Buses with storage: {len(buses_with_storage):,}")
    logger.info(f"  Buses in lines: {len(buses_in_lines):,}")
    logger.info(f"  Buses in links: {len(buses_in_links):,}")
    logger.info(f"  Buses in transformers: {len(buses_in_transformers):,}")
    logger.info(f"  Total topology buses: {len(buses_in_topology):,}")
    
    # Run PyPSA's built-in consistency check
    logger.info("")
    logger.info("Running PyPSA consistency_check()...")
    try:
        # consistency_check() will log warnings/errors automatically
        network.consistency_check()
        logger.info("✓ PyPSA consistency check passed")
    except Exception as e:
        errors.append(f"PyPSA consistency check failed: {str(e)}")
        logger.error(f"❌ PyPSA consistency check failed: {e}")
    
    return errors, warnings


def deep_validate_network(network, scenario_config, logger):
    """
    Perform comprehensive deep validation.
    
    Returns:
        (passed, errors, warnings): Tuple of (bool, list, list)
    """
    logger.info("=" * 80)
    logger.info("DEEP NETWORK VALIDATION FOR OPTIMIZATION")
    logger.info("Checking: Demand → Renewables → Thermal → Storage → Interconnectors → Consistency")
    logger.info("=" * 80)
    
    all_errors = []
    all_warnings = []
    
    # Run all validations
    e, w = validate_demand(network, logger)
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = validate_renewables(network, logger)
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = validate_thermal_generators(network, logger)
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = validate_load_shedding(network, logger)
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = validate_storage(network, logger)
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = validate_interconnectors(network, scenario_config, logger)
    all_errors.extend(e)
    all_warnings.extend(w)
    
    e, w = validate_network_consistency(network, logger)
    all_errors.extend(e)
    all_warnings.extend(w)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    if len(all_errors) == 0:
        logger.info("✅ VALIDATION PASSED")
        if len(all_warnings) > 0:
            logger.info(f"   {len(all_warnings)} warnings (non-critical):")
            for w in all_warnings:
                logger.info(f"   - {w}")
        logger.info("=" * 80)
        return True, [], all_warnings
    else:
        logger.error("❌ VALIDATION FAILED")
        logger.error(f"   {len(all_errors)} critical errors:")
        for e in all_errors:
            logger.error(f"   - {e}")
        if len(all_warnings) > 0:
            logger.warning(f"   {len(all_warnings)} warnings:")
            for w in all_warnings:
                logger.warning(f"   - {w}")
        logger.error("=" * 80)
        logger.error("DO NOT SOLVE THIS NETWORK - FIX ERRORS FIRST")
        logger.error("=" * 80)
        return False, all_errors, all_warnings


if __name__ == "__main__":
    # For standalone use or Snakemake integration
    from scripts.utilities.logging_config import setup_logging
    
    logger = setup_logging("deep_validation", log_level="INFO", log_to_file=True)
    
    # Load network
    network_path = snakemake.input.network if hasattr(snakemake, 'input') else "network.nc"
    scenario_config = snakemake.params.scenario_config if hasattr(snakemake, 'params') else {}
    
    logger.info(f"Loading network from: {network_path}")
    network = pypsa.Network(network_path)
    
    # Run validation
    passed, errors, warnings = deep_validate_network(network, scenario_config, logger)
    
    if not passed:
        raise ValueError(f"Network validation failed with {len(errors)} errors")
    
    logger.info("Deep validation completed successfully")

