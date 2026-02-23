"""
Verify found coordinates by checking against network topology
"""
import pandas as pd
import numpy as np
from pathlib import Path

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def main():
    base_path = Path(r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1")

    print("="*80)
    print("COORDINATE VERIFICATION USING NETWORK TOPOLOGY")
    print("="*80)

    # Load the coordinates we found
    coords_df = pd.read_csv(base_path / "tmp" / "missing_bus_coordinates.csv")

    print("\nFound coordinates summary:")
    print(coords_df.to_string(index=False))

    # Load GSP coordinates for reference
    gsp_df = pd.read_csv(base_path / "data" / "network" / "ETYS" / "fes2023_regional_breakdown_gsp_info.csv")

    # Create coordinate lookup
    coord_lookup = {}
    for _, row in coords_df.iterrows():
        bus_code = row['bus_code']
        if bus_code not in coord_lookup:
            coord_lookup[bus_code] = (row['lat'], row['lon'])

    # Add GSP coordinates
    for _, row in gsp_df.iterrows():
        gsp_id = row['GSP ID'].split('_')[0]  # Extract base code
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            if gsp_id not in coord_lookup:
                coord_lookup[gsp_id] = (row['Latitude'], row['Longitude'])

    print(f"\nTotal coordinates available: {len(coord_lookup)}")

    # Load network topology
    gb_network_path = base_path / "data" / "network" / "ETYS" / "GB_network.xlsx"
    df_ac = pd.read_excel(gb_network_path, sheet_name='AC')

    # Target buses to verify
    target_buses = ['TEAL', 'KINT', 'CASS', 'TUMM', 'ERRO', 'FOYE', 'TORN']

    print("\n" + "="*80)
    print("VERIFICATION BY NETWORK NEIGHBORS")
    print("="*80)

    for target in target_buses:
        if target not in coord_lookup:
            print(f"\n{target}: No coordinates found")
            continue

        target_lat, target_lon = coord_lookup[target]

        print(f"\n{target}: ({target_lat:.4f}, {target_lon:.4f})")

        # Find connections
        mask = (df_ac['Node 1'].str.contains(target, na=False) |
                df_ac['Node 2'].str.contains(target, na=False))
        connections = df_ac[mask]

        if connections.empty:
            print("  No connections found")
            continue

        # Extract neighbor bus codes
        neighbors = set()
        line_lengths = {}

        for _, row in connections.iterrows():
            node1 = str(row['Node 1'])[:4]
            node2 = str(row['Node 2'])[:4]
            line_length = row['OHL Length (km)'] + row['Cable Length (km)']

            if node1 == target:
                neighbors.add(node2)
                line_lengths[node2] = line_length
            elif node2 == target:
                neighbors.add(node1)
                line_lengths[node1] = line_length

        print(f"  Connected to {len(neighbors)} neighbors")

        # Calculate distances to neighbors with known coordinates
        neighbor_distances = []
        for neighbor in neighbors:
            if neighbor in coord_lookup:
                n_lat, n_lon = coord_lookup[neighbor]
                calc_dist = haversine_distance(target_lat, target_lon, n_lat, n_lon)
                line_len = line_lengths.get(neighbor, 0)

                neighbor_distances.append({
                    'neighbor': neighbor,
                    'calc_dist_km': calc_dist,
                    'line_length_km': line_len,
                    'difference_km': calc_dist - line_len,
                    'percent_diff': ((calc_dist - line_len) / line_len * 100) if line_len > 0 else 0
                })

        if neighbor_distances:
            print(f"\n  Distance verification (to neighbors with known coordinates):")
            print(f"  {'Neighbor':<8} {'Calc Dist':<12} {'Line Length':<13} {'Diff':<10} {'% Diff':<10}")
            print(f"  {'-'*8} {'-'*12} {'-'*13} {'-'*10} {'-'*10}")

            for nd in sorted(neighbor_distances, key=lambda x: abs(x['percent_diff'])):
                print(f"  {nd['neighbor']:<8} {nd['calc_dist_km']:>10.2f} km {nd['line_length_km']:>11.2f} km "
                      f"{nd['difference_km']:>8.2f} km {nd['percent_diff']:>8.1f}%")

            # Summary statistics
            avg_diff = np.mean([abs(nd['percent_diff']) for nd in neighbor_distances if nd['line_length_km'] > 0])
            print(f"\n  Average absolute % difference: {avg_diff:.1f}%")

            if avg_diff < 20:
                print(f"  ✓ GOOD: Coordinates match network topology well")
            elif avg_diff < 50:
                print(f"  ⚠ FAIR: Coordinates roughly match, but verify if possible")
            else:
                print(f"  ✗ POOR: Coordinates may be inaccurate, manual verification needed")

        else:
            print(f"  ⚠ Cannot verify: No neighbors with known coordinates")

    print("\n" + "="*80)
    print("COORDINATE QUALITY SUMMARY")
    print("="*80)

    quality_summary = coords_df.groupby('confidence').size()
    print("\nBy confidence level:")
    for conf, count in quality_summary.items():
        print(f"  {conf}: {count} coordinate entries")

    print("\n✓ High confidence coordinates can be used directly")
    print("⚠ Medium confidence coordinates should be verified if possible")
    print("✗ Low confidence coordinates need manual verification or conversion")

if __name__ == "__main__":
    main()
