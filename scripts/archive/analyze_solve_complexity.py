"""
Analyze network complexity and suggest optimizations for faster solving.
"""
import pypsa
import pandas as pd
import sys

def analyze_network_complexity(network_path):
    """Analyze network and suggest solve time optimizations."""
    print("=" * 80)
    print("NETWORK COMPLEXITY ANALYSIS - SOLVE TIME OPTIMIZATION")
    print("=" * 80)
    
    n = pypsa.Network(network_path)
    
    print(f"\n📊 NETWORK SCALE:")
    print(f"   Buses: {len(n.buses):,}")
    print(f"   Lines: {len(n.lines):,}")
    print(f"   Generators: {len(n.generators):,}")
    print(f"   Storage units: {len(n.storage_units):,}")
    print(f"   Links: {len(n.links):,}")
    print(f"   Loads: {len(n.loads):,}")
    print(f"   Snapshots: {len(n.snapshots):,}")
    
    # Calculate problem size
    total_variables = (
        len(n.generators) * len(n.snapshots) +  # Generator dispatch
        len(n.storage_units) * len(n.snapshots) * 2 +  # Storage charge/discharge
        len(n.links) * len(n.snapshots) +  # Link flows
        len(n.lines) * len(n.snapshots)  # Line flows
    )
    
    print(f"\n🔢 OPTIMIZATION PROBLEM SIZE:")
    print(f"   Estimated decision variables: {total_variables:,}")
    print(f"   Estimated constraints: ~{total_variables * 2:,}")
    
    # Check for unit commitment constraints
    has_committable = 'committable' in n.generators.columns and n.generators.committable.any()
    if has_committable:
        n_committable = n.generators.committable.sum()
        print(f"\n⚠️  UNIT COMMITMENT ACTIVE: {n_committable} committable generators")
        print(f"   This adds binary variables making problem MILP (much slower!)")
    
    # Check for ramp limits
    has_ramp = False
    if 'ramp_limit_up' in n.generators.columns:
        has_ramp = n.generators.ramp_limit_up.notna().any()
        if has_ramp:
            n_ramp = n.generators.ramp_limit_up.notna().sum()
            print(f"\n⚠️  RAMP CONSTRAINTS: {n_ramp} generators with ramp limits")
            print(f"   Adds temporal coupling between snapshots (slower)")
    
    # Check generator types
    print(f"\n⚡ GENERATOR BREAKDOWN:")
    gen_by_carrier = n.generators.groupby('carrier').size().sort_values(ascending=False)
    for carrier, count in gen_by_carrier.head(10).items():
        cap = n.generators[n.generators.carrier == carrier].p_nom.sum()
        print(f"   {carrier:30s}: {count:4d} units, {cap:10,.1f} MW")
    
    # Check for load shedding
    load_shedding = n.generators[n.generators.carrier == 'load_shedding']
    if len(load_shedding) > 0:
        print(f"\n💡 LOAD SHEDDING:")
        print(f"   Units: {len(load_shedding):,}")
        print(f"   Total capacity: {load_shedding.p_nom.sum():,.0f} MW")
        print(f"   Marginal cost: £{load_shedding.marginal_cost.iloc[0]:,.0f}/MWh")
        
        # Suggestion: Could reduce number of load shedding units
        if len(load_shedding) > 100:
            print(f"   ⚡ OPTIMIZATION: Consider reducing to 1 large unit per bus")
            print(f"      Current: {len(load_shedding)} units")
            print(f"      Suggested: {len(n.buses)} units (one per bus)")
            print(f"      Variables saved: {(len(load_shedding) - len(n.buses)) * len(n.snapshots):,}")
    
    # Check storage
    if len(n.storage_units) > 0:
        print(f"\n🔋 STORAGE:")
        print(f"   Units: {len(n.storage_units)}")
        storage_by_carrier = n.storage_units.groupby('carrier').size()
        for carrier, count in storage_by_carrier.items():
            print(f"   {carrier}: {count} units")
    
    # Interconnector configuration
    if len(n.links) > 0:
        print(f"\n🔗 INTERCONNECTORS/LINKS:")
        print(f"   Total links: {len(n.links)}")
        
        # Check for p_set (fixed flows)
        has_p_set = hasattr(n, 'links_t') and hasattr(n.links_t, 'p_set') and not n.links_t.p_set.empty
        if has_p_set:
            n_fixed = len([c for c in n.links_t.p_set.columns if not n.links_t.p_set[c].isna().all()])
            print(f"   Links with FIXED flows (p_set): {n_fixed}")
            print(f"   ⚡ Fixed flows reduce variables (good for solve time!)")
        
    # Line configuration
    if len(n.lines) > 0:
        print(f"\n⚡ TRANSMISSION LINES:")
        print(f"   Total lines: {len(n.lines)}")
        # Check for line limits
        has_limits = 's_nom' in n.lines.columns and (n.lines.s_nom > 0).any()
        if has_limits:
            n_limited = (n.lines.s_nom > 0).sum()
            print(f"   Lines with thermal limits: {n_limited}")
    
    print("\n" + "=" * 80)
    print("🚀 OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 80)
    
    suggestions = []
    
    # Snapshot reduction
    if len(n.snapshots) > 500:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'Temporal Resolution',
            'issue': f'{len(n.snapshots):,} snapshots creates {total_variables:,} variables',
            'solution': 'Use solve_period to optimize subset of year (e.g., 1 week = 336 snapshots)',
            'impact': f'~{int(100 * (1 - 336/len(n.snapshots)))}% reduction in solve time'
        })
    
    # Load shedding reduction
    if len(load_shedding) > len(n.buses):
        n_current = len(load_shedding)
        n_suggested = len(n.buses)
        var_reduction = (n_current - n_suggested) * len(n.snapshots)
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'Generator Count',
            'issue': f'{n_current:,} load shedding generators (one per load)',
            'solution': f'Consolidate to {n_suggested} units (one per bus)',
            'impact': f'Reduces {var_reduction:,} variables'
        })
    
    # Unit commitment
    if has_committable:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'Problem Type',
            'issue': f'Unit commitment active ({n_committable} committable units)',
            'solution': 'Disable unit commitment (set committable=False) for LP instead of MILP',
            'impact': 'Can reduce solve time by 10-100x (MILP → LP)'
        })
    
    # Ramp constraints
    if has_ramp:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'Temporal Coupling',
            'issue': f'{n_ramp} generators with ramp limits',
            'solution': 'Remove ramp constraints for faster solve (if acceptable)',
            'impact': 'Reduces constraint matrix density'
        })
    
    # Solver settings
    suggestions.append({
        'priority': 'HIGH',
        'category': 'Solver Configuration',
        'issue': 'Default solver settings may not be optimal',
        'solution': 'Use Gurobi barrier method with crossover=0, increase threads',
        'impact': '2-5x faster for large LPs'
    })
    
    # Network clustering
    if len(n.buses) > 50:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'Network Clustering',
            'issue': f'{len(n.buses)} buses creates many line flow variables',
            'solution': f'Further cluster to ~50 buses (currently {len(n.buses)})',
            'impact': f'~{int(100 * (1 - 50/len(n.buses)))}% reduction in network complexity'
        })
    
    # Print suggestions
    for i, sug in enumerate(suggestions, 1):
        print(f"\n{i}. [{sug['priority']}] {sug['category']}")
        print(f"   Issue: {sug['issue']}")
        print(f"   ✓ Solution: {sug['solution']}")
        print(f"   📈 Impact: {sug['impact']}")
    
    print("\n" + "=" * 80)
    print("💡 QUICK WIN: Use solve_period in scenario config!")
    print("   Instead of solving full year (17,568 snapshots),")
    print("   solve 1 week (336 snapshots) = ~50x faster")
    print("=" * 80)
    
    return suggestions

if __name__ == "__main__":
    network_path = sys.argv[1] if len(sys.argv) > 1 else "resources/network/Historical_2020_clustered.nc"
    suggestions = analyze_network_complexity(network_path)

