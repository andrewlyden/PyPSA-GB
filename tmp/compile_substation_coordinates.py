"""
Compile substation coordinates from available sources and suggest manual lookups
"""
import pandas as pd
from pathlib import Path

def main():
    base_path = Path(r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1")

    # Target substations
    target_substations = {
        'TEAL': {'name': 'TEALING', 'voltages': [275, 132], 'operator': 'SHE'},
        'KINT': {'name': 'KINTORE', 'voltages': [400, 275, 132], 'operator': 'SHE'},
        'CASS': {'name': 'CASSLEY', 'voltages': [132], 'operator': 'SHE'},
        'TUMM': {'name': 'TUMMEL', 'voltages': [400, 275, 132], 'operator': 'SHE'},
        'ERRO': {'name': 'ERROCHTY', 'voltages': [132], 'operator': 'SHE'},
        'FOYE': {'name': 'FOYERS', 'voltages': [275], 'operator': 'SHE'},
        'TORN': {'name': 'TORNESS', 'voltages': [400, 132], 'operator': 'SPT'}
    }

    print("="*80)
    print("SUBSTATION COORDINATE COMPILATION")
    print("="*80)

    # Check GSP info files
    gsp_files = [
        'fes2021_regional_breakdown_gsp_info.csv',
        'fes2022_regional_breakdown_gsp_info.csv',
        'fes2023_regional_breakdown_gsp_info.csv'
    ]

    found_coords = {}

    for gsp_file in gsp_files:
        gsp_path = base_path / "data" / "network" / "ETYS" / gsp_file
        if gsp_path.exists():
            df = pd.read_csv(gsp_path)
            print(f"\nChecking {gsp_file}:")

            for code, info in target_substations.items():
                # Check for matches
                mask = df['GSP ID'].str.contains(code, case=False, na=False)
                if mask.any():
                    matches = df[mask]
                    for _, row in matches.iterrows():
                        if code not in found_coords:
                            found_coords[code] = {
                                'lat': row['Latitude'],
                                'lon': row['Longitude'],
                                'name': row['Name'],
                                'source': gsp_file
                            }
                            print(f"  FOUND {code}: {row['Name']} at ({row['Latitude']}, {row['Longitude']})")

    print("\n" + "="*80)
    print("SUMMARY OF FOUND COORDINATES")
    print("="*80)

    for code, info in target_substations.items():
        print(f"\n{code} - {info['name']} ({info['operator']})")
        print(f"  Voltages: {info['voltages']} kV")

        if code in found_coords:
            coord = found_coords[code]
            print(f"  ✓ FOUND: Lat {coord['lat']}, Lon {coord['lon']}")
            print(f"    Source: {coord['source']}")
        else:
            print(f"  ✗ NOT FOUND - Manual lookup required")
            print(f"    Search term: '{info['name']} substation Scotland'")
            print(f"    Approximate region:")

            # Provide regional hints based on known nearby substations
            if code == 'TEAL':
                print(f"      Near Dundee/Forfar area")
            elif code == 'TUMM':
                print(f"      Perthshire, near Pitlochry")
            elif code == 'ERRO':
                print(f"      Perthshire, near Tummel (upstream on River Tummel)")
            elif code == 'FOYE':
                print(f"      Near Loch Ness, Scottish Highlands")
            elif code == 'TORN':
                print(f"      East Lothian coast, near Torness Power Station")

    print("\n" + "="*80)
    print("MANUAL LOOKUP RECOMMENDATIONS")
    print("="*80)

    missing = [code for code in target_substations if code not in found_coords]

    if missing:
        print(f"\nThe following {len(missing)} substations need manual coordinate lookup:")

        for code in missing:
            info = target_substations[code]
            print(f"\n{code} - {info['name']}")
            print(f"  1. Google Maps search: '{info['name']} substation Scotland'")
            print(f"  2. OpenStreetMap search: '{info['name']} electrical substation'")
            print(f"  3. SSEN website: Search for {info['name']} in their network maps")
            print(f"  4. National Grid ESO: Check transmission network diagrams")

            # Provide specific hints for power station connections
            if code == 'FOYE':
                print(f"  5. Note: Connected to Foyers pumped storage power station")
            elif code == 'TORN':
                print(f"  5. Note: Connected to Torness nuclear power station")
            elif code == 'ERRO':
                print(f"  5. Note: Errochty Dam and hydroelectric station")

    print("\n" + "="*80)
    print("COORDINATE FORMAT FOR MANUAL ENTRY")
    print("="*80)
    print("\nOnce you find the coordinates, create a CSV file with this format:")
    print("\nbus_code,bus_name,lat,lon,voltage_kv,source")

    for code, info in target_substations.items():
        if code not in found_coords:
            for voltage in info['voltages']:
                if voltage in [132, 275]:  # Only missing transmission voltages
                    print(f"{code}{voltage//100}*,{info['name']},{'{lat}'},{'{lon}'},{voltage},manual_lookup")

    print("\n" + "="*80)
    print("COORDINATE ESTIMATION FROM NETWORK TOPOLOGY")
    print("="*80)
    print("\nAs a fallback, we can estimate coordinates based on connected buses")
    print("by analyzing the AC network topology in GB_network.xlsx")

    # Read the AC sheet to find connections
    gb_network_path = base_path / "data" / "network" / "ETYS" / "GB_network.xlsx"
    df_ac = pd.read_excel(gb_network_path, sheet_name='AC')

    for code in missing:
        print(f"\n{code} network connections:")
        # Find all edges involving this bus
        mask = (df_ac['Node 1'].str.contains(code, na=False) |
                df_ac['Node 2'].str.contains(code, na=False))
        connections = df_ac[mask]

        if not connections.empty:
            print(f"  Connected to {len(connections)} other buses:")
            unique_connected = set()
            for _, row in connections.iterrows():
                node1 = str(row['Node 1'])[:4]
                node2 = str(row['Node 2'])[:4]
                if node1 != code:
                    unique_connected.add(node1)
                if node2 != code:
                    unique_connected.add(node2)

            for connected in sorted(unique_connected):
                print(f"    - {connected}")

            print(f"  → Could estimate {code} coordinates by averaging neighbors with known locations")
        else:
            print(f"  No direct connections found in AC sheet")

if __name__ == "__main__":
    main()
