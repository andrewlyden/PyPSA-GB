"""
Check if there's any additional location data available beyond what we see in the index sheets
"""
import pandas as pd
from pathlib import Path
import os

def main():
    base_path = Path(r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1")
    etys_path = base_path / "data" / "network" / "ETYS"

    print("="*80)
    print("CHECKING ETYS DIRECTORY FOR ADDITIONAL FILES WITH LOCATION DATA")
    print("="*80)

    # List all files in ETYS directory
    if etys_path.exists():
        print(f"\nFiles in {etys_path}:")
        for file in etys_path.iterdir():
            print(f"  {file.name} ({file.stat().st_size / 1024:.1f} KB)")

    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)

    print("\n1. ETYS Appendix B 2023.xlsx - Substation Index Sheets (B-1-1a, B-1-1b, B-1-1c, B-1-1d):")
    print("   - Contains Site Code, Site Name, and Voltage (kV)")
    print("   - NO latitude, longitude, easting, northing, or postcode data")
    print("   - Found all 7 target buses:")

    target_info = {
        'TEAL': ('B-1-1a (SHE)', 'TEALING', [275, 132, 33]),
        'KINT': ('B-1-1a (SHE)', 'KINTORE', [400, 275, 132, 33]),
        'CASS': ('B-1-1a (SHE)', 'CASSLEY', [132, 33]),
        'TUMM': ('B-1-1a (SHE)', 'TUMMEL', [400, 275, 132, 33]),
        'ERRO': ('B-1-1a (SHE)', 'ERROCHTY', [132, 11]),
        'FOYE': ('B-1-1a (SHE)', 'FOYERS', [275]),
        'TORN': ('B-1-1b (SPT)', 'TORNESS', [400, 132])
    }

    for code, (sheet, name, voltages) in target_info.items():
        print(f"\n     {code}: {name}")
        print(f"        Sheet: {sheet}")
        print(f"        Voltages: {voltages} kV")

    print("\n2. GB_network.xlsx - AC Sheet:")
    print("   - Contains network topology (Node 1, Node 2)")
    print("   - NO coordinate data")
    print("   - Found multiple nodes for each target bus:")

    node_examples = {
        'CASS': ['CASS1Q', 'CASS3-'],
        'ERRO': ['ERRO1A', 'ERRO1B', 'ERRO1J', 'ERRO1K', 'ERRO1T', 'ERRO5J', 'ERRO5L'],
        'FOYE': ['FOYE2-', 'FOYE2J'],
        'KINT': ['KINT1-', 'KINT1B', 'KINT1P', 'KINT1R', 'KINT1T', 'KINT1U', 'KINT2J', 'KINT2K', 'KINT3-'],
        'TEAL': ['TEAL1-', 'TEAL2J', 'TEAL2K'],
        'TORN': ['TORN1-', 'TORN4-'],
        'TUMM': ['TUMM1J', 'TUMM1K', 'TUMM2J', 'TUMM2K', 'TUMM4-']
    }

    for code, nodes in node_examples.items():
        print(f"     {code}: {', '.join(nodes[:3])}")
        if len(nodes) > 3:
            print(f"           ... and {len(nodes) - 3} more")

    print("\n3. CONCLUSION:")
    print("   - The ETYS Excel files contain substation NAMES but NO geographic coordinates")
    print("   - To place these buses on the map, we need to:")
    print("     a) Use substation names to search for coordinates in other sources")
    print("     b) Use Google Maps/OpenStreetMap APIs to geocode substation names")
    print("     c) Manually look up coordinates from National Grid or SSEN documentation")
    print("     d) Estimate coordinates based on connected substations (less accurate)")

    print("\n4. RECOMMENDED APPROACH:")
    print("   - Create a manual coordinates file mapping substation codes to lat/lon")
    print("   - Use substation names for geocoding:")

    print("\n     Substations to geocode:")
    for code, (sheet, name, voltages) in target_info.items():
        print(f"       {code} -> {name} substation, Scotland")

    # Check if there's already a coordinates file
    print("\n5. CHECKING FOR EXISTING COORDINATE FILES:")
    possible_coord_files = [
        base_path / "data" / "network" / "buses.csv",
        base_path / "data" / "network" / "substations.csv",
        base_path / "data" / "network" / "coordinates.csv",
        etys_path / "coordinates.csv",
        etys_path / "substation_locations.csv",
    ]

    for coord_file in possible_coord_files:
        if coord_file.exists():
            print(f"\n   Found: {coord_file}")
            try:
                df = pd.read_csv(coord_file)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   First 5 rows:")
                print(df.head().to_string())
            except Exception as e:
                print(f"   Error reading: {e}")
        else:
            print(f"   Not found: {coord_file.name}")

if __name__ == "__main__":
    main()
