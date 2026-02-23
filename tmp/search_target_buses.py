"""
Search for target buses in ETYS Appendix B substation index sheets
"""
import pandas as pd
from pathlib import Path

# Target buses we're looking for
TARGET_BUSES = ['TEAL', 'KINT', 'CASS', 'TUMM', 'ERRO', 'FOYE', 'TORN']

def search_in_sheet(file_path, sheet_name, skip_rows=1):
    """Search for target buses in a sheet"""
    try:
        # Read the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows)

        print(f"\n{'='*80}")
        print(f"SHEET: {sheet_name}")
        print(f"{'='*80}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Check if any columns exist
        if df.empty or len(df.columns) == 0:
            print("Empty sheet or no columns")
            return

        # Search for target buses in all columns
        found_buses = []
        for bus in TARGET_BUSES:
            for col in df.columns:
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                    # Search for exact matches and partial matches
                    mask = df[col].astype(str).str.contains(bus, case=False, na=False)
                    if mask.any():
                        found_buses.append(bus)
                        print(f"\n*** FOUND '{bus}' in column '{col}' ***")
                        print(df[mask][list(df.columns)].to_string())
                        break  # Move to next bus once found

        if not found_buses:
            print(f"\nNo target buses found in this sheet")
        else:
            print(f"\nFound buses: {found_buses}")

        # Show first 20 rows for context
        print(f"\nFirst 20 rows of data:")
        print(df.head(20).to_string())

    except Exception as e:
        print(f"Error reading {sheet_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    base_path = Path(r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1")
    file_path = base_path / "data" / "network" / "ETYS" / "ETYS Appendix B 2023.xlsx"

    # Check the substation index sheets
    index_sheets = ['B-1-1a', 'B-1-1b', 'B-1-1c', 'B-1-1d']

    for sheet in index_sheets:
        search_in_sheet(file_path, sheet, skip_rows=1)

    # Also check the GB_network.xlsx nodes
    print("\n\n" + "="*80)
    print("CHECKING GB_network.xlsx FOR NODE/BUS INFORMATION")
    print("="*80)

    gb_network_path = base_path / "data" / "network" / "ETYS" / "GB_network.xlsx"

    # Read the AC sheet which has Node 1 and Node 2
    try:
        df_ac = pd.read_excel(gb_network_path, sheet_name='AC')
        print("\nAC Sheet - All unique nodes:")
        all_nodes = pd.concat([df_ac['Node 1'], df_ac['Node 2']]).unique()
        all_nodes_sorted = sorted([str(n) for n in all_nodes])
        print(f"Total unique nodes: {len(all_nodes_sorted)}")

        # Filter to show nodes that contain our target bus codes
        target_nodes = [n for n in all_nodes_sorted if any(bus in n for bus in TARGET_BUSES)]
        if target_nodes:
            print(f"\n*** Nodes containing target buses: ***")
            for node in target_nodes:
                print(f"  {node}")
        else:
            print("\nNo nodes found containing target bus codes")

        # Show sample of all nodes for context
        print(f"\nSample of all nodes (first 50):")
        for node in all_nodes_sorted[:50]:
            print(f"  {node}")

    except Exception as e:
        print(f"Error reading GB_network.xlsx AC sheet: {e}")

if __name__ == "__main__":
    main()
