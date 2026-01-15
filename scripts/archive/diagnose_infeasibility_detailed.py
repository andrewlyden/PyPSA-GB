"""
Detailed infeasibility diagnosis for PyPSA network.
"""
import pypsa
import pandas as pd
import numpy as np
import sys

def diagnose_network(network_path):
    """Comprehensive infeasibility diagnosis."""
    print("=" * 80)
    print("INFEASIBILITY DIAGNOSIS")
    print("=" * 80)
    
    n = pypsa.Network(network_path)
    snapshots = n.snapshots[:168]  # First week
    
    # 1. Check sub-networks
    print("\n1. SUB-NETWORK ANALYSIS")
    print("-" * 40)
    n.determine_network_topology()
    sub_sizes = n.buses.groupby('sub_network').size()
    print(f"Total sub-networks: {len(sub_sizes)}")
    print(sub_sizes)
    
    # 2. Check external HVDC buses
    print("\n2. EXTERNAL BUS BALANCE")
    print("-" * 40)
    
    external_buses = n.buses[n.buses.index.str.contains('HVDC_External|Central DC')]
    print(f"External buses: {len(external_buses)}")
    
    for bus_id in external_buses.index:
        print(f"\n  {bus_id}:")
        
        # Generators
        gens = n.generators[n.generators.bus == bus_id]
        total_gen_cap = gens.p_nom.sum()
        print(f"    Generators: {len(gens)}, capacity={total_gen_cap:.0f} MW")
        for g in gens.index:
            carrier = n.generators.loc[g, 'carrier']
            p_nom = n.generators.loc[g, 'p_nom']
            print(f"      - {g}: {carrier}, {p_nom:.0f} MW")
        
        # Loads
        loads = n.loads[n.loads.bus == bus_id]
        print(f"    Loads: {len(loads)}")
        for l in loads.index:
            if l in n.loads_t.p_set.columns:
                pset = n.loads_t.p_set.loc[snapshots, l]
                print(f"      - {l}: p_set range [{pset.min():.1f}, {pset.max():.1f}]")
            else:
                static = n.loads.loc[l, 'p_set']
                print(f"      - {l}: static p_set={static}")
        
        # Links
        links_at_bus = n.links[(n.links.bus0 == bus_id) | (n.links.bus1 == bus_id)]
        print(f"    Links: {len(links_at_bus)}")
        for lk in links_at_bus.index:
            link = n.links.loc[lk]
            direction = "bus0->bus1" if link.bus0 == bus_id else "bus1<-bus0"
            has_pset = lk in n.links_t.p_set.columns
            print(f"      - {lk}: {direction}, p_nom={link.p_nom:.0f}, fixed={has_pset}")
            if has_pset:
                pset = n.links_t.p_set.loc[snapshots, lk]
                print(f"        p_set range: [{pset.min():.1f}, {pset.max():.1f}]")
    
    # 3. Check global supply/demand balance
    print("\n3. SUPPLY/DEMAND BALANCE")
    print("-" * 40)
    
    # Total load p_set
    total_load = n.loads_t.p_set.loc[snapshots].sum(axis=1)
    print(f"Total load: mean={total_load.mean():.0f}, min={total_load.min():.0f}, max={total_load.max():.0f} MW")
    
    # Total available generation
    total_gen_cap = n.generators.p_nom.sum()
    print(f"Total generator capacity: {total_gen_cap:.0f} MW")
    
    # Renewable available (with p_max_pu)
    renewable_carriers = ['wind_onshore', 'wind_offshore', 'solar_pv', 'small_hydro', 'large_hydro']
    renewable_gens = n.generators[n.generators.carrier.isin(renewable_carriers)]
    
    # Calculate available renewable power
    if 'p_max_pu' in n.generators_t and len(n.generators_t.p_max_pu.columns) > 0:
        renewable_available = pd.DataFrame(0, index=snapshots, columns=['total'])
        for g in renewable_gens.index:
            if g in n.generators_t.p_max_pu.columns:
                p_nom = n.generators.loc[g, 'p_nom']
                p_max_pu = n.generators_t.p_max_pu.loc[snapshots, g]
                renewable_available['total'] += p_nom * p_max_pu
        print(f"Renewable available: mean={renewable_available['total'].mean():.0f} MW")
    
    # Dispatchable capacity
    dispatchable_carriers = ['CCGT', 'OCGT', 'coal', 'nuclear', 'biomass']
    dispatchable = n.generators[n.generators.carrier.isin(dispatchable_carriers)]
    print(f"Dispatchable capacity: {dispatchable.p_nom.sum():.0f} MW")
    
    # 4. Check for NaN/inf values
    print("\n4. DATA QUALITY CHECK")
    print("-" * 40)
    
    # Loads p_set
    if 'p_set' in n.loads_t:
        nan_loads = n.loads_t.p_set.isna().sum().sum()
        inf_loads = np.isinf(n.loads_t.p_set.values).sum()
        neg_loads = (n.loads_t.p_set < 0).sum().sum()
        print(f"Load p_set: NaN={nan_loads}, Inf={inf_loads}, Negative={neg_loads}")
        if neg_loads > 0:
            print("  WARNING: Negative loads found!")
            neg_cols = n.loads_t.p_set.columns[(n.loads_t.p_set < 0).any()]
            for c in neg_cols[:5]:
                neg_vals = n.loads_t.p_set[c][n.loads_t.p_set[c] < 0]
                print(f"    {c}: {len(neg_vals)} negative values, min={neg_vals.min():.1f}")
    
    # Generator p_max_pu
    if 'p_max_pu' in n.generators_t:
        nan_pmax = n.generators_t.p_max_pu.isna().sum().sum()
        neg_pmax = (n.generators_t.p_max_pu < 0).sum().sum()
        gt1_pmax = (n.generators_t.p_max_pu > 1.0001).sum().sum()
        print(f"Generator p_max_pu: NaN={nan_pmax}, Negative={neg_pmax}, >1={gt1_pmax}")
    
    # Links p_set
    if 'p_set' in n.links_t:
        nan_links = n.links_t.p_set.isna().sum().sum()
        print(f"Links p_set: NaN={nan_links}")
    
    # 5. Check load shedding
    print("\n5. LOAD SHEDDING CHECK")
    print("-" * 40)
    load_shedding = n.generators[n.generators.carrier == 'load_shedding']
    print(f"Load shedding generators: {len(load_shedding)}")
    print(f"Load shedding capacity: {load_shedding.p_nom.sum():.0f} MW")
    print(f"Peak load: {total_load.max():.0f} MW")
    
    if load_shedding.p_nom.sum() < total_load.max():
        print("  WARNING: Load shedding capacity less than peak load!")
    
    # 6. Check line/transformer ratings
    print("\n6. NETWORK CONSTRAINTS")
    print("-" * 40)
    
    # Lines with zero s_nom
    zero_lines = n.lines[n.lines.s_nom == 0]
    print(f"Lines with s_nom=0: {len(zero_lines)}")
    if len(zero_lines) > 0:
        print("  WARNING: Some lines have zero capacity!")
        print(f"  Examples: {list(zero_lines.index[:5])}")
    
    # Transformers with zero s_nom
    zero_transformers = n.transformers[n.transformers.s_nom == 0]
    print(f"Transformers with s_nom=0: {len(zero_transformers)}")
    
    # 7. Check interconnector balance specifically
    print("\n7. INTERCONNECTOR BALANCE")
    print("-" * 40)
    
    # For historical scenarios with fixed p_set, check the balance
    ic_links = n.links[n.links.index.str.startswith('IC_')]
    print(f"Interconnector links: {len(ic_links)}")
    
    for lk in ic_links.index:
        link = n.links.loc[lk]
        bus0 = link.bus0
        bus1 = link.bus1
        
        # Get p_set
        if lk in n.links_t.p_set.columns:
            pset = n.links_t.p_set.loc[snapshots, lk]
            
            # Power flows FROM bus0 TO bus1 when p_set > 0
            # So positive = export from GB to external
            # Negative = import to GB from external
            
            avg_flow = pset.mean()
            max_export = pset.max()
            max_import = -pset.min()
            
            print(f"\n  {lk}:")
            print(f"    bus0 (GB): {bus0}")
            print(f"    bus1 (ext): {bus1}")
            print(f"    Avg flow: {avg_flow:.1f} MW (+ve=export from GB)")
            print(f"    Max export: {max_export:.1f} MW")
            print(f"    Max import: {max_import:.1f} MW")
            
            # Check if external bus can handle the flows
            ext_bus = bus1
            ext_gens = n.generators[n.generators.bus == ext_bus]
            ext_loads = n.loads[n.loads.bus == ext_bus]
            
            ext_gen_cap = ext_gens.p_nom.sum()
            print(f"    External gen capacity: {ext_gen_cap:.0f} MW (for imports)")
            
            # Check if load can absorb exports
            if len(ext_loads) > 0:
                for l in ext_loads.index:
                    if l in n.loads_t.p_set.columns:
                        load_pset = n.loads_t.p_set.loc[snapshots, l]
                        print(f"    External load {l}: range [{load_pset.min():.1f}, {load_pset.max():.1f}]")
            
            # Check for balance issues
            if max_import > ext_gen_cap:
                print(f"    WARNING: Max import ({max_import:.0f}) > gen capacity ({ext_gen_cap:.0f})")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    network_path = sys.argv[1] if len(sys.argv) > 1 else "resources/network/Historical_2020_ETYS.nc"
    diagnose_network(network_path)

