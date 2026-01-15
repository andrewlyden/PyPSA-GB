"""
Export PyPSA network to Excel files for inspection.
"""
import pypsa
import pandas as pd
from pathlib import Path
import sys

def export_network_to_excel(network_path, output_dir="network_export"):
    """Export all network components to Excel files."""
    print("=" * 80)
    print("EXPORTING PYPSA NETWORK TO EXCEL")
    print("=" * 80)
    
    n = pypsa.Network(network_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nNetwork: {n.name}")
    print(f"Output directory: {output_path.absolute()}")
    
    # 1. Export static component data
    print("\n1. Exporting static component data...")
    
    components = {
        'buses': n.buses,
        'generators': n.generators,
        'loads': n.loads,
        'storage_units': n.storage_units,
        'lines': n.lines,
        'links': n.links,
        'carriers': n.carriers,
    }
    
    for name, df in components.items():
        if len(df) > 0:
            filepath = output_path / f"{name}.xlsx"
            df.to_excel(filepath)
            print(f"   ✓ {name}: {len(df)} rows → {filepath.name}")
        else:
            print(f"   - {name}: (empty)")
    
    # 2. Export time series data (generators_t, loads_t, etc.)
    print("\n2. Exporting time series data...")
    
    # Generators time series
    if hasattr(n, 'generators_t'):
        if hasattr(n.generators_t, 'p_max_pu') and not n.generators_t.p_max_pu.empty:
            filepath = output_path / "generators_t_p_max_pu.xlsx"
            # Limit to first 100 columns to avoid Excel limits
            df = n.generators_t.p_max_pu.iloc[:, :100]
            df.to_excel(filepath)
            print(f"   ✓ generators_t.p_max_pu: {df.shape} (first 100 columns) → {filepath.name}")
    
    # Loads time series
    if hasattr(n, 'loads_t'):
        if hasattr(n.loads_t, 'p_set') and not n.loads_t.p_set.empty:
            filepath = output_path / "loads_t_p_set.xlsx"
            # Limit columns
            df = n.loads_t.p_set.iloc[:, :100]
            df.to_excel(filepath)
            print(f"   ✓ loads_t.p_set: {df.shape} (first 100 columns) → {filepath.name}")
    
    # Links time series (p_set for interconnectors)
    if hasattr(n, 'links_t'):
        if hasattr(n.links_t, 'p_set') and not n.links_t.p_set.empty:
            filepath = output_path / "links_t_p_set.xlsx"
            df = n.links_t.p_set
            df.to_excel(filepath)
            print(f"   ✓ links_t.p_set: {df.shape} → {filepath.name}")
    
    # 3. Export summary statistics
    print("\n3. Creating summary workbook...")
    
    with pd.ExcelWriter(output_path / "network_summary.xlsx", engine='openpyxl') as writer:
        # Sheet 1: Overview
        overview_data = {
            'Property': ['Network Name', 'Buses', 'Generators', 'Loads', 'Storage Units', 
                        'Lines', 'Links', 'Snapshots', 'Start Time', 'End Time'],
            'Value': [
                n.name,
                len(n.buses),
                len(n.generators),
                len(n.loads),
                len(n.storage_units),
                len(n.lines),
                len(n.links),
                len(n.snapshots),
                str(n.snapshots[0]) if len(n.snapshots) > 0 else 'N/A',
                str(n.snapshots[-1]) if len(n.snapshots) > 0 else 'N/A'
            ]
        }
        pd.DataFrame(overview_data).to_excel(writer, sheet_name='Overview', index=False)
        
        # Sheet 2: Generator capacity by carrier
        if len(n.generators) > 0:
            gen_summary = n.generators.groupby('carrier').agg({
                'p_nom': ['count', 'sum', 'mean', 'min', 'max'],
                'marginal_cost': ['mean', 'min', 'max']
            }).round(2)
            gen_summary.to_excel(writer, sheet_name='Generator Summary')
        
        # Sheet 3: Storage summary
        if len(n.storage_units) > 0:
            storage_summary = n.storage_units.groupby('carrier').agg({
                'p_nom': ['count', 'sum', 'mean'],
                'max_hours': 'mean',
                'efficiency_store': 'mean',
                'efficiency_dispatch': 'mean',
                'marginal_cost': ['mean', 'min', 'max'],
                'standing_loss': 'mean'
            }).round(4)
            storage_summary.to_excel(writer, sheet_name='Storage Summary')
        
        # Sheet 4: Load summary
        if len(n.loads) > 0 and hasattr(n, 'loads_t') and hasattr(n.loads_t, 'p_set'):
            load_stats = n.loads_t.p_set.describe()
            load_stats.to_excel(writer, sheet_name='Load Statistics')
    
    print(f"   ✓ network_summary.xlsx created")
    
    # 4. Export specific diagnostic data
    print("\n4. Creating diagnostic files...")
    
    # Generators with zero marginal cost
    if len(n.generators) > 0:
        zero_mc = n.generators[n.generators.marginal_cost == 0][['carrier', 'p_nom', 'marginal_cost']]
        if len(zero_mc) > 0:
            filepath = output_path / "generators_zero_marginal_cost.xlsx"
            zero_mc.to_excel(filepath)
            print(f"   ✓ Zero marginal cost generators: {len(zero_mc)} → {filepath.name}")
    
    # Storage details
    if len(n.storage_units) > 0:
        filepath = output_path / "storage_units_detailed.xlsx"
        n.storage_units[['carrier', 'p_nom', 'max_hours', 'efficiency_store', 
                        'efficiency_dispatch', 'marginal_cost', 'standing_loss']].to_excel(filepath)
        print(f"   ✓ Storage units detailed → {filepath.name}")
    
    # Links/Interconnectors details
    if len(n.links) > 0:
        filepath = output_path / "links_detailed.xlsx"
        n.links.to_excel(filepath)
        print(f"   ✓ Links detailed → {filepath.name}")
    
    print("\n" + "=" * 80)
    print(f"✓ Export complete! Files saved to: {output_path.absolute()}")
    print("=" * 80)
    
    return output_path

if __name__ == "__main__":
    network_path = sys.argv[1] if len(sys.argv) > 1 else "resources/network/Historical_2020_clustered.nc"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "network_export"
    
    export_network_to_excel(network_path, output_dir)

