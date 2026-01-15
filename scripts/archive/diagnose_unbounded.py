"""
Diagnose unbounded optimization problem.
"""
import pypsa
import pandas as pd
import sys

def diagnose_unbounded_model(network_path):
    """Find components that could cause unbounded optimization."""
    print("=" * 80)
    print("UNBOUNDED MODEL DIAGNOSTICS")
    print("=" * 80)
    
    n = pypsa.Network(network_path)
    
    print(f"\nNetwork: {n.name}")
    print(f"Buses: {len(n.buses)}, Generators: {len(n.generators)}, Storage: {len(n.storage_units)}")
    
    issues_found = []
    
    # 1. Check for negative or zero marginal costs
    print("\n" + "=" * 80)
    print("1. GENERATOR MARGINAL COSTS")
    print("=" * 80)
    
    if 'marginal_cost' in n.generators.columns:
        negative_mc = n.generators[n.generators.marginal_cost < 0]
        zero_mc = n.generators[n.generators.marginal_cost == 0]
        
        if len(negative_mc) > 0:
            print(f"\n❌ PROBLEM: {len(negative_mc)} generators with NEGATIVE marginal cost!")
            print("   These can produce infinite power for infinite profit (unbounded)")
            for idx, row in negative_mc.head(20).iterrows():
                print(f"   - {idx}: {row['carrier']} @ £{row['marginal_cost']}/MWh, capacity: {row.get('p_nom', 'N/A')} MW")
            issues_found.append(f"Negative marginal cost generators: {len(negative_mc)}")
        else:
            print("✓ No generators with negative marginal cost")
        
        if len(zero_mc) > 0:
            print(f"\n⚠️  {len(zero_mc)} generators with ZERO marginal cost")
            print("   (Usually OK, but check if intended)")
            print(f"   Carriers: {zero_mc.carrier.value_counts().head(10).to_dict()}")
        
        # Summary statistics
        print(f"\nMarginal cost statistics:")
        print(f"  Min: £{n.generators.marginal_cost.min():.2f}/MWh")
        print(f"  Max: £{n.generators.marginal_cost.max():.2f}/MWh")
        print(f"  Mean: £{n.generators.marginal_cost.mean():.2f}/MWh")
    
    # 2. Check storage costs
    print("\n" + "=" * 80)
    print("2. STORAGE UNITS")
    print("=" * 80)
    
    if len(n.storage_units) > 0:
        print(f"\nStorage units: {len(n.storage_units)}")
        
        # Check for negative standing loss or marginal costs
        if 'marginal_cost' in n.storage_units.columns:
            neg_storage_mc = n.storage_units[n.storage_units.marginal_cost < 0]
            if len(neg_storage_mc) > 0:
                print(f"\n❌ PROBLEM: {len(neg_storage_mc)} storage units with NEGATIVE marginal cost!")
                issues_found.append(f"Negative storage marginal cost: {len(neg_storage_mc)}")
            else:
                print("✓ No storage with negative marginal cost")
        
        # Check for cycling profit potential
        if 'marginal_cost' in n.storage_units.columns:
            storage_mc_range = n.storage_units.marginal_cost.describe()
            print(f"\nStorage marginal cost range:")
            print(f"  Min: £{storage_mc_range['min']:.2f}/MWh")
            print(f"  Max: £{storage_mc_range['max']:.2f}/MWh")
        
        # Check standing loss (should be small positive number)
        if 'standing_loss' in n.storage_units.columns:
            print(f"\nStanding loss (self-discharge):")
            print(f"  Min: {n.storage_units.standing_loss.min():.6f}")
            print(f"  Max: {n.storage_units.standing_loss.max():.6f}")
            if n.storage_units.standing_loss.min() < 0:
                print(f"  ❌ PROBLEM: Negative standing loss detected!")
                issues_found.append("Negative storage standing loss")
    
    # 3. Check for missing capacity limits
    print("\n" + "=" * 80)
    print("3. CAPACITY LIMITS")
    print("=" * 80)
    
    # Generators without capacity limits
    unlimited_gens = n.generators[
        (n.generators.p_nom.isna()) | 
        (n.generators.p_nom == 0) | 
        (n.generators.p_nom == float('inf'))
    ]
    
    if len(unlimited_gens) > 0 and len(unlimited_gens) < len(n.generators):
        print(f"\n⚠️  {len(unlimited_gens)} generators with unlimited/zero capacity")
        print(f"   Carriers: {unlimited_gens.carrier.value_counts().head(10).to_dict()}")
        # This is OK if all generators have p_nom_extendable=True
    
    # Check for extendable generators with zero capital cost
    if 'p_nom_extendable' in n.generators.columns and 'capital_cost' in n.generators.columns:
        extendable = n.generators[n.generators.p_nom_extendable == True]
        if len(extendable) > 0:
            zero_capex = extendable[extendable.capital_cost == 0]
            if len(zero_capex) > 0:
                print(f"\n❌ PROBLEM: {len(zero_capex)} extendable generators with ZERO capital cost!")
                print("   These can be built infinitely for free (unbounded)")
                for idx, row in zero_capex.head(10).iterrows():
                    print(f"   - {idx}: {row['carrier']}, MC: £{row.get('marginal_cost', 'N/A')}/MWh")
                issues_found.append(f"Extendable generators with zero capital cost: {len(zero_capex)}")
    
    # 4. Check link/interconnector configuration
    print("\n" + "=" * 80)
    print("4. LINKS/INTERCONNECTORS")
    print("=" * 80)
    
    if len(n.links) > 0:
        print(f"\nLinks: {len(n.links)}")
        
        # Check for negative marginal costs
        if 'marginal_cost' in n.links.columns:
            neg_link_mc = n.links[n.links.marginal_cost < 0]
            if len(neg_link_mc) > 0:
                print(f"\n❌ PROBLEM: {len(neg_link_mc)} links with NEGATIVE marginal cost!")
                issues_found.append(f"Negative link marginal cost: {len(neg_link_mc)}")
        
        # Check if p_set (fixed flows) is configured
        has_p_set = hasattr(n, 'links_t') and hasattr(n.links_t, 'p_set') and not n.links_t.p_set.empty
        if has_p_set:
            print(f"✓ Links have p_set (fixed flows) - should prevent arbitrage")
        else:
            print(f"⚠️  Links do NOT have p_set - could allow arbitrage if marginal_cost != 0")
    
    # 5. Check for missing constraints
    print("\n" + "=" * 80)
    print("5. CONSTRAINT COMPLETENESS")
    print("=" * 80)
    
    # Check if loads exist
    if len(n.loads) == 0:
        print("❌ PROBLEM: No loads in network! Unbounded generation.")
        issues_found.append("No loads (no demand constraint)")
    else:
        print(f"✓ {len(n.loads)} loads present")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if len(issues_found) == 0:
        print("\n✓ No obvious unboundedness issues found in network data")
        print("\nPossible causes:")
        print("  1. Price arbitrage opportunity (buy low, sell high cycles)")
        print("  2. Missing temporal coupling constraints")
        print("  3. Interaction between storage, interconnectors, and price variations")
        print("\nSuggestions:")
        print("  - Try adding small positive marginal costs to zero-cost generators (£0.01/MWh)")
        print("  - Add standing loss to storage (e.g., 0.001 per hour = 0.1%/hour)")
        print("  - Check if renewable profiles have unusual values (>1.0 capacity factors)")
    else:
        print(f"\n❌ Found {len(issues_found)} potential unboundedness issues:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    
    print("=" * 80)
    
    return issues_found

if __name__ == "__main__":
    network_path = sys.argv[1] if len(sys.argv) > 1 else "resources/network/Historical_2020_clustered.nc"
    issues = diagnose_unbounded_model(network_path)
    sys.exit(0 if len(issues) == 0 else 1)

