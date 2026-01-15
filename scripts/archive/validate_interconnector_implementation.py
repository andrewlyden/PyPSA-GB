#!/usr/bin/env python3
"""
Validate Interconnector Implementation
======================================

Quick validation script to check if a network has the correct interconnector
implementation with external generators and zero-cost links.

Usage:
    python scripts/validate_interconnector_implementation.py <network_file.nc>

Example:
    python scripts/validate_interconnector_implementation.py resources/network/HT35_network_demand_renewables_thermal_storage_interconnectors.nc

Author: PyPSA-GB Team
Date: November 4, 2025
"""

import sys
from pathlib import Path
import pypsa
import pandas as pd

def validate_interconnector_implementation(network_file: str) -> dict:
    """
    Validate interconnector implementation in a PyPSA network.
    
    Checks:
    1. External generators exist (EU_import carrier)
    2. Link marginal costs are zero/minimal
    3. External buses have generators attached
    4. Link efficiency is reasonable (95-99%)
    
    Args:
        network_file: Path to PyPSA network NetCDF file
        
    Returns:
        dict: Validation results with status and messages
    """
    
    results = {
        'network_file': network_file,
        'valid': True,
        'warnings': [],
        'errors': [],
        'info': []
    }
    
    try:
        # Load network
        print(f"\n{'='*80}")
        print(f"Validating Interconnector Implementation")
        print(f"{'='*80}")
        print(f"Network file: {network_file}")
        
        n = pypsa.Network(network_file)
        results['info'].append(f"✓ Loaded network with {len(n.buses)} buses, {len(n.generators)} generators, {len(n.links)} links")
        
        # Check for interconnector links
        ic_links = n.links[n.links.index.str.startswith('IC_')]
        
        if len(ic_links) == 0:
            results['warnings'].append("⚠ No interconnector links found (links starting with 'IC_')")
            print("\n⚠ WARNING: No interconnector links found")
            print("  This network may not have interconnectors added yet.")
            return results
        
        results['info'].append(f"✓ Found {len(ic_links)} interconnector links")
        total_capacity = ic_links['p_nom'].sum()
        results['info'].append(f"  Total interconnector capacity: {total_capacity:.0f} MW")
        
        # Check 1: External generators
        print("\n1. Checking for European supply generators...")
        eu_generators = n.generators[n.generators['carrier'] == 'EU_import']
        
        if len(eu_generators) == 0:
            results['errors'].append("✗ No European supply generators found (carrier='EU_import')")
            results['valid'] = False
            print("  ✗ CRITICAL: No European supply generators found")
            print("    External buses may act as unbounded power sources!")
        else:
            results['info'].append(f"✓ Found {len(eu_generators)} European supply generators")
            print(f"  ✓ Found {len(eu_generators)} European supply generators")
            
            # List generators
            for gen_name, gen in eu_generators.iterrows():
                mc = gen.get('marginal_cost', 0)
                pnom = gen.get('p_nom', 0)
                bus = gen.get('bus', 'unknown')
                print(f"    - {gen_name}: bus={bus}, p_nom={pnom:.0f} MW, marginal_cost=£{mc:.2f}/MWh")
                results['info'].append(f"  Generator {gen_name}: {pnom:.0f} MW at £{mc:.2f}/MWh")
        
        # Check 2: Link marginal costs
        print("\n2. Checking link marginal costs...")
        high_cost_links = ic_links[ic_links['marginal_cost'] > 5.0]
        
        if len(high_cost_links) > 0:
            results['warnings'].append(f"⚠ {len(high_cost_links)} interconnector links have marginal costs >£5/MWh")
            print(f"  ⚠ WARNING: {len(high_cost_links)} links have high marginal costs")
            print("    Expected: near-zero costs (economics on external generators)")
            
            for link_name, link in high_cost_links.iterrows():
                mc = link['marginal_cost']
                print(f"    - {link_name}: £{mc:.2f}/MWh")
                results['warnings'].append(f"  Link {link_name}: £{mc:.2f}/MWh")
        else:
            results['info'].append(f"✓ All {len(ic_links)} interconnector links have appropriate costs (≤£5/MWh)")
            print(f"  ✓ All {len(ic_links)} links have appropriate marginal costs")
            
            # Show cost distribution
            max_cost = ic_links['marginal_cost'].max()
            mean_cost = ic_links['marginal_cost'].mean()
            print(f"    Range: £{ic_links['marginal_cost'].min():.2f} - £{max_cost:.2f}/MWh (mean: £{mean_cost:.2f}/MWh)")
        
        # Check 3: External buses have generators
        print("\n3. Checking external bus generators...")
        external_buses = set(ic_links['bus1'].unique())  # bus1 is the external side
        
        if len(eu_generators) > 0:
            buses_with_generators = set(eu_generators['bus'].unique())
            missing_generators = external_buses - buses_with_generators
            
            if missing_generators:
                results['warnings'].append(f"⚠ {len(missing_generators)} external buses without generators: {missing_generators}")
                print(f"  ⚠ WARNING: {len(missing_generators)} external buses without generators")
                for bus in missing_generators:
                    print(f"    - {bus}")
            else:
                results['info'].append(f"✓ All {len(external_buses)} external buses have generators")
                print(f"  ✓ All {len(external_buses)} external buses have generators")
        
        # Check 4: Link efficiency
        print("\n4. Checking link efficiency...")
        low_eff_links = ic_links[ic_links['efficiency'] < 0.90]
        high_eff_links = ic_links[ic_links['efficiency'] > 1.0]
        
        if len(low_eff_links) > 0:
            results['warnings'].append(f"⚠ {len(low_eff_links)} links have efficiency <90%")
            print(f"  ⚠ WARNING: {len(low_eff_links)} links have efficiency <90%")
            for link_name, link in low_eff_links.iterrows():
                eff = link['efficiency']
                print(f"    - {link_name}: {eff*100:.1f}% (losses: {(1-eff)*100:.1f}%)")
        
        if len(high_eff_links) > 0:
            results['errors'].append(f"✗ {len(high_eff_links)} links have efficiency >100% (impossible)")
            results['valid'] = False
            print(f"  ✗ ERROR: {len(high_eff_links)} links have efficiency >100%")
            for link_name, link in high_eff_links.iterrows():
                eff = link['efficiency']
                print(f"    - {link_name}: {eff*100:.1f}%")
        
        if len(low_eff_links) == 0 and len(high_eff_links) == 0:
            results['info'].append(f"✓ All links have reasonable efficiency (90-100%)")
            print(f"  ✓ All links have reasonable efficiency")
            mean_eff = ic_links['efficiency'].mean()
            print(f"    Mean efficiency: {mean_eff*100:.1f}% (losses: {(1-mean_eff)*100:.1f}%)")
        
        # Check 5: Historical flows (if present)
        print("\n5. Checking for historical fixed flows...")
        if hasattr(n, 'links_t') and hasattr(n.links_t, 'p_set'):
            ic_with_pset = [link for link in ic_links.index if link in n.links_t.p_set.columns]
            
            if ic_with_pset:
                results['info'].append(f"✓ Found {len(ic_with_pset)} links with fixed flows (historical scenario)")
                print(f"  ✓ Found {len(ic_with_pset)} links with fixed flows (historical scenario)")
                
                for link in ic_with_pset[:3]:  # Show first 3
                    flows = n.links_t.p_set[link]
                    mean_flow = flows.mean()
                    print(f"    - {link}: mean flow = {mean_flow:.1f} MW")
                
                if len(ic_with_pset) > 3:
                    print(f"    ... and {len(ic_with_pset)-3} more")
            else:
                results['info'].append("  No fixed flows (future scenario with optimizable interconnectors)")
                print("  ℹ No fixed flows (future scenario with optimizable interconnectors)")
        
        # Summary
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        if results['valid'] and len(results['warnings']) == 0:
            print("✓ ALL CHECKS PASSED")
            print("  Interconnector implementation is correct!")
        elif results['valid']:
            print("⚠ PASSED WITH WARNINGS")
            print(f"  Found {len(results['warnings'])} warnings (see above)")
        else:
            print("✗ VALIDATION FAILED")
            print(f"  Found {len(results['errors'])} errors (see above)")
        
        print(f"\nInterconnector links: {len(ic_links)}")
        print(f"European generators: {len(eu_generators)}")
        print(f"External buses: {len(external_buses)}")
        
        return results
        
    except Exception as e:
        results['errors'].append(f"✗ Failed to validate network: {e}")
        results['valid'] = False
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return results


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python validate_interconnector_implementation.py <network_file.nc>")
        print("\nExample:")
        print("  python scripts/validate_interconnector_implementation.py resources/network/HT35_network_interconnectors.nc")
        sys.exit(1)
    
    network_file = sys.argv[1]
    
    if not Path(network_file).exists():
        print(f"✗ ERROR: Network file not found: {network_file}")
        sys.exit(1)
    
    results = validate_interconnector_implementation(network_file)
    
    # Exit with appropriate code
    if results['valid'] and len(results['warnings']) == 0:
        sys.exit(0)  # Perfect
    elif results['valid']:
        sys.exit(0)  # Warnings but valid
    else:
        sys.exit(1)  # Errors found


if __name__ == "__main__":
    main()

