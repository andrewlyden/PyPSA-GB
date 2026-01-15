"""
Validate PyPSA Network Generators
==================================

Comprehensive validation of generator integration in the PyPSA network:
1. Check carriers are defined and assigned correctly
2. Verify static generator attributes (p_nom, marginal_cost, etc.)
3. Validate time-varying inputs (p_max_pu)
4. Check for data quality issues

Author: PyPSA-GB Team
Date: October 2025
"""

import pypsa
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utilities.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

def load_network(network_path):
    """Load PyPSA network from file."""
    logger.info(f"Loading network from: {network_path}")
    network = pypsa.Network(network_path)
    logger.info(f"Network loaded: {len(network.snapshots)} snapshots, {len(network.generators)} generators")
    return network

def check_carriers(network):
    """Check carrier definitions and assignments."""
    logger.info("\n" + "="*80)
    logger.info("CARRIER VALIDATION")
    logger.info("="*80)
    
    # Check carriers table
    if hasattr(network, 'carriers') and not network.carriers.empty:
        logger.info(f"\n✅ Carriers defined: {len(network.carriers)}")
        logger.info("\nCarrier attributes:")
        print(network.carriers)
    else:
        logger.warning("\n⚠️  No carriers table found in network")
    
    # Check generator carrier assignments
    logger.info(f"\n📊 Generator Carrier Distribution:")
    carrier_counts = network.generators.carrier.value_counts().sort_index()
    print("\n" + carrier_counts.to_string())
    
    # Check for missing carriers
    unique_carriers = network.generators.carrier.unique()
    if hasattr(network, 'carriers') and not network.carriers.empty:
        missing_carriers = set(unique_carriers) - set(network.carriers.index)
        if missing_carriers:
            logger.error(f"\n❌ Generators with undefined carriers: {missing_carriers}")
        else:
            logger.info(f"\n✅ All generator carriers defined in carriers table")
    
    # Check for null carriers
    null_carriers = network.generators[network.generators.carrier.isna()]
    if not null_carriers.empty:
        logger.error(f"\n❌ {len(null_carriers)} generators with null carriers:")
        print(null_carriers[['bus', 'p_nom', 'carrier']])
    else:
        logger.info("✅ No generators with null carriers")
    
    return carrier_counts

def check_static_attributes(network):
    """Validate static generator attributes."""
    logger.info("\n" + "="*80)
    logger.info("STATIC GENERATOR ATTRIBUTES")
    logger.info("="*80)
    
    generators = network.generators
    
    # Key attributes to check
    attributes = {
        'p_nom': 'Nominal power (MW)',
        'p_nom_extendable': 'Extendable capacity',
        'marginal_cost': 'Marginal cost (£/MWh)',
        'capital_cost': 'Capital cost',
        'efficiency': 'Efficiency',
        'carrier': 'Carrier type',
        'bus': 'Bus connection'
    }
    
    logger.info("\n📋 Attribute Summary:")
    for attr, description in attributes.items():
        if attr in generators.columns:
            non_null = generators[attr].notna().sum()
            pct = 100 * non_null / len(generators)
            
            if attr in ['p_nom', 'marginal_cost', 'efficiency']:
                # Numeric attributes - show statistics
                stats = generators[attr].describe()
                logger.info(f"\n{attr} ({description}):")
                logger.info(f"  Count: {non_null}/{len(generators)} ({pct:.1f}%)")
                logger.info(f"  Min:   {stats['min']:.2f}")
                logger.info(f"  Mean:  {stats['mean']:.2f}")
                logger.info(f"  Max:   {stats['max']:.2f}")
            elif attr == 'p_nom_extendable':
                # Boolean attribute
                if non_null > 0:
                    extendable_count = generators[attr].sum()
                    logger.info(f"\n{attr} ({description}):")
                    logger.info(f"  Extendable: {extendable_count}/{len(generators)} ({100*extendable_count/len(generators):.1f}%)")
            else:
                logger.info(f"\n{attr} ({description}): {non_null}/{len(generators)} ({pct:.1f}%)")
        else:
            logger.warning(f"\n⚠️  {attr} ({description}): NOT FOUND")
    
    # Check for suspicious values
    logger.info("\n🔍 Data Quality Checks:")
    
    # Negative p_nom
    negative_pnom = generators[generators.p_nom < 0]
    if not negative_pnom.empty:
        logger.error(f"❌ {len(negative_pnom)} generators with negative p_nom:")
        print(negative_pnom[['carrier', 'p_nom', 'bus']])
    else:
        logger.info("✅ No negative p_nom values")
    
    # Zero p_nom
    zero_pnom = generators[generators.p_nom == 0]
    if not zero_pnom.empty:
        logger.warning(f"⚠️  {len(zero_pnom)} generators with zero p_nom:")
        print(zero_pnom[['carrier', 'p_nom', 'bus']].head(10))
    else:
        logger.info("✅ No zero p_nom values")
    
    # Negative marginal cost
    if 'marginal_cost' in generators.columns:
        negative_mc = generators[generators.marginal_cost < 0]
        if not negative_mc.empty:
            logger.warning(f"⚠️  {len(negative_mc)} generators with negative marginal_cost:")
            print(negative_mc[['carrier', 'marginal_cost', 'p_nom']].head(10))
        else:
            logger.info("✅ No negative marginal_cost values")
    
    # Efficiency > 1.0
    if 'efficiency' in generators.columns:
        high_eff = generators[generators.efficiency > 1.0]
        if not high_eff.empty:
            logger.warning(f"⚠️  {len(high_eff)} generators with efficiency > 1.0:")
            print(high_eff[['carrier', 'efficiency', 'p_nom']].head(10))
        else:
            logger.info("✅ No generators with efficiency > 1.0")
    
    # Check by carrier
    logger.info("\n📊 Statistics by Carrier:")
    carrier_stats = generators.groupby('carrier').agg({
        'p_nom': ['count', 'sum', 'mean'],
        'marginal_cost': 'mean' if 'marginal_cost' in generators.columns else lambda x: None
    })
    print("\n" + carrier_stats.to_string())
    
    return generators

def check_time_varying_attributes(network):
    """Validate time-varying generator attributes (p_max_pu, etc.)."""
    logger.info("\n" + "="*80)
    logger.info("TIME-VARYING ATTRIBUTES (p_max_pu)")
    logger.info("="*80)
    
    generators = network.generators
    
    # Check if p_max_pu exists
    if not hasattr(network.generators_t, 'p_max_pu'):
        logger.warning("⚠️  No p_max_pu time series found")
        return None
    
    p_max_pu = network.generators_t.p_max_pu
    
    if p_max_pu.empty:
        logger.warning("⚠️  p_max_pu DataFrame is empty")
        return None
    
    logger.info(f"\n📊 p_max_pu Overview:")
    logger.info(f"  Timesteps: {len(p_max_pu)}")
    logger.info(f"  Generators with profiles: {len(p_max_pu.columns)}")
    logger.info(f"  Total generators: {len(generators)}")
    
    # Check which generators have time-varying profiles
    generators_with_profiles = set(p_max_pu.columns)
    all_generators = set(generators.index)
    generators_without_profiles = all_generators - generators_with_profiles
    
    logger.info(f"\n  Generators WITH time-varying profiles: {len(generators_with_profiles)}")
    logger.info(f"  Generators WITHOUT time-varying profiles: {len(generators_without_profiles)}")
    
    # Show breakdown by carrier
    if generators_without_profiles:
        static_carriers = generators.loc[list(generators_without_profiles), 'carrier'].value_counts()
        logger.info(f"\n📊 Generators without profiles (by carrier):")
        print(static_carriers.to_string())
    
    if generators_with_profiles:
        varying_carriers = generators.loc[list(generators_with_profiles), 'carrier'].value_counts()
        logger.info(f"\n📊 Generators with profiles (by carrier):")
        print(varying_carriers.to_string())
    
    # Validate p_max_pu values
    logger.info("\n🔍 p_max_pu Data Quality:")
    
    # Check range (should be 0-1)
    min_val = p_max_pu.min().min()
    max_val = p_max_pu.max().max()
    logger.info(f"  Value range: {min_val:.4f} to {max_val:.4f}")
    
    if min_val < 0:
        logger.error(f"❌ Negative values found (min: {min_val:.4f})")
        negative_cols = p_max_pu.columns[(p_max_pu < 0).any()]
        logger.error(f"   Generators with negative values: {len(negative_cols)}")
        print(f"   Examples: {list(negative_cols[:5])}")
    else:
        logger.info("✅ No negative values")
    
    if max_val > 1.0:
        logger.error(f"❌ Values > 1.0 found (max: {max_val:.4f})")
        over_one_cols = p_max_pu.columns[(p_max_pu > 1.0).any()]
        logger.error(f"   Generators with values > 1.0: {len(over_one_cols)}")
        print(f"   Examples: {list(over_one_cols[:5])}")
    else:
        logger.info("✅ No values > 1.0")
    
    # Check for NaN values
    nan_count = p_max_pu.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠️  {nan_count} NaN values found")
        cols_with_nan = p_max_pu.columns[p_max_pu.isna().any()]
        logger.warning(f"   Generators with NaN: {len(cols_with_nan)}")
        print(f"   Examples: {list(cols_with_nan[:5])}")
    else:
        logger.info("✅ No NaN values")
    
    # Check for constant profiles (might indicate issues)
    logger.info("\n🔍 Profile Variation Check:")
    std_devs = p_max_pu.std()
    constant_profiles = std_devs[std_devs == 0].index.tolist()
    if constant_profiles:
        logger.warning(f"⚠️  {len(constant_profiles)} generators with constant profiles:")
        # Show carrier breakdown
        const_carriers = generators.loc[constant_profiles, 'carrier'].value_counts()
        print(const_carriers.to_string())
        logger.info(f"   Examples: {constant_profiles[:5]}")
    else:
        logger.info("✅ All profiles have variation")
    
    # Check for alternating zeros (the bug we fixed!)
    logger.info("\n🔍 Alternating Zero Pattern Check:")
    sample_generators = p_max_pu.columns[:min(50, len(p_max_pu.columns))]
    alternating_zeros = 0
    
    for gen in sample_generators:
        series = p_max_pu[gen]
        non_zero = series[series > 0.001]  # Avoid floating point issues
        if len(non_zero) > 0:
            # Check if non-zero values appear at regular intervals (e.g., every other timestep)
            non_zero_indices = non_zero.index
            if len(non_zero_indices) > 1:
                # Check first 10 intervals
                intervals = []
                for i in range(min(10, len(non_zero_indices)-1)):
                    interval = non_zero_indices[i+1] - non_zero_indices[i]
                    intervals.append(interval)
                
                # If all intervals are exactly 2 timesteps, it's alternating
                if intervals and all(i == intervals[0] and intervals[0] == pd.Timedelta('1H') for i in intervals):
                    # Check if there are zeros in between
                    first_nonzero_idx = network.snapshots.get_loc(non_zero_indices[0])
                    if first_nonzero_idx < len(network.snapshots) - 1:
                        next_idx = first_nonzero_idx + 1
                        if series.iloc[next_idx] < 0.001:  # Next value is zero
                            alternating_zeros += 1
    
    if alternating_zeros > 0:
        logger.error(f"❌ {alternating_zeros}/{len(sample_generators)} sampled generators have alternating zero pattern!")
    else:
        logger.info(f"✅ No alternating zero patterns detected (sampled {len(sample_generators)} generators)")
    
    # Sample some profiles
    logger.info("\n📈 Sample Profiles (first 10 timesteps):")
    sample_cols = p_max_pu.columns[:3]
    for col in sample_cols:
        carrier = generators.loc[col, 'carrier']
        p_nom = generators.loc[col, 'p_nom']
        logger.info(f"\n{col} ({carrier}, {p_nom:.2f} MW):")
        sample = p_max_pu[col].head(10)
        for idx, val in sample.items():
            logger.info(f"  {idx}: {val:.4f} (Power: {val*p_nom:.2f} MW)")
    
    # Statistics by carrier
    logger.info("\n📊 Profile Statistics by Carrier:")
    for carrier in varying_carriers.index:
        carrier_gens = generators[generators.carrier == carrier].index
        carrier_gens_with_profiles = list(set(carrier_gens) & generators_with_profiles)
        
        if carrier_gens_with_profiles:
            carrier_data = p_max_pu[carrier_gens_with_profiles]
            logger.info(f"\n{carrier}:")
            logger.info(f"  Generators: {len(carrier_gens_with_profiles)}")
            logger.info(f"  Mean CF: {carrier_data.mean().mean():.3f}")
            logger.info(f"  Min CF: {carrier_data.min().min():.3f}")
            logger.info(f"  Max CF: {carrier_data.max().max():.3f}")
            logger.info(f"  Non-zero %: {(carrier_data > 0.001).sum().sum() / (len(carrier_data) * len(carrier_gens_with_profiles)) * 100:.1f}%")
    
    return p_max_pu

def main():
    """Main validation routine."""
    # Network path
    network_path = project_root / "resources" / "network" / "HT35_clustered_gsp_base_demand_generators.nc"
    
    if not network_path.exists():
        logger.error(f"Network file not found: {network_path}")
        logger.info("Run generators.smk workflow first:")
        logger.info("  snakemake resources/network/HT35_clustered_gsp_base_demand_generators.nc")
        return 1
    
    # Load network
    network = load_network(network_path)
    
    # Run validation checks
    logger.info("\n" + "#"*80)
    logger.info("# PYPSA NETWORK GENERATOR VALIDATION")
    logger.info("#"*80)
    
    # 1. Check carriers
    carrier_counts = check_carriers(network)
    
    # 2. Check static attributes
    generators = check_static_attributes(network)
    
    # 3. Check time-varying attributes
    p_max_pu = check_time_varying_attributes(network)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Network loaded: {len(network.snapshots)} snapshots, {len(network.generators)} generators")
    logger.info(f"✅ Carriers: {len(carrier_counts)} unique types")
    logger.info(f"✅ Total capacity: {generators.p_nom.sum():.1f} MW")
    if p_max_pu is not None:
        logger.info(f"✅ Time-varying profiles: {len(p_max_pu.columns)} generators")
    logger.info("\n🎉 Validation complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

