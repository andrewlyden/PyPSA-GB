"""
Analyze Excel files for bus/node location data
"""
import pandas as pd
import openpyxl
from pathlib import Path

# Target buses we're looking for
TARGET_BUSES = ['TEAL', 'KINT', 'CASS', 'TUMM', 'ERRO', 'FOYE', 'TORN']

def analyze_sheet(file_path, sheet_name):
    """Analyze a single sheet for location data"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=3)

        # Check if sheet has location-related columns
        location_keywords = ['lat', 'lon', 'x', 'y', 'easting', 'northing',
                            'postcode', 'coordinate', 'location', 'position',
                            'grid_ref', 'os_grid', 'node', 'bus', 'substation']

        columns = [str(col).lower() for col in df.columns]
        has_location_data = any(keyword in col for col in columns for keyword in location_keywords)

        # Check if any target buses are in the data
        has_target_bus = False
        if has_location_data:
            # Check all columns for target bus names
            for col in df.columns:
                if df[col].dtype == 'object':
                    values_str = ' '.join(df[col].astype(str).tolist())
                    if any(bus in values_str for bus in TARGET_BUSES):
                        has_target_bus = True
                        break

        return {
            'columns': list(df.columns),
            'has_location_data': has_location_data,
            'has_target_bus': has_target_bus,
            'first_rows': df,
            'shape': df.shape
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    base_path = Path(r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1")

    files_to_check = [
        base_path / "data" / "network" / "ETYS" / "GB_network.xlsx",
        base_path / "data" / "network" / "ETYS" / "ETYS Appendix B 2023.xlsx"
    ]

    for file_path in files_to_check:
        if not file_path.exists():
            print(f"\n{'='*80}")
            print(f"FILE NOT FOUND: {file_path}")
            print(f"{'='*80}\n")
            continue

        print(f"\n{'='*80}")
        print(f"ANALYZING: {file_path.name}")
        print(f"{'='*80}\n")

        # Get all sheet names
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            print(f"Total sheets found: {len(sheet_names)}")
            print(f"Sheet names: {sheet_names}\n")

            # Analyze each sheet
            for sheet_name in sheet_names:
                print(f"\n{'-'*80}")
                print(f"SHEET: {sheet_name}")
                print(f"{'-'*80}")

                result = analyze_sheet(file_path, sheet_name)

                if 'error' in result:
                    print(f"Error reading sheet: {result['error']}")
                    continue

                print(f"Shape: {result['shape']}")
                print(f"Columns ({len(result['columns'])}): {result['columns']}")
                print(f"Has location-related data: {result['has_location_data']}")
                print(f"Contains target buses: {result['has_target_bus']}")

                # Show first rows if has location data or target buses
                if result['has_location_data'] or result['has_target_bus']:
                    print(f"\n*** POTENTIALLY USEFUL SHEET ***")
                    print(f"\nFirst 3 rows:")
                    print(result['first_rows'].to_string())

                    # Try to read more rows if target buses found
                    if result['has_target_bus']:
                        print(f"\n*** CONTAINS TARGET BUSES - Reading more data ***")
                        try:
                            df_full = pd.read_excel(file_path, sheet_name=sheet_name, nrows=50)
                            # Filter rows mentioning target buses
                            for col in df_full.columns:
                                if df_full[col].dtype == 'object':
                                    mask = df_full[col].astype(str).str.contains('|'.join(TARGET_BUSES), na=False)
                                    if mask.any():
                                        print(f"\nRows containing target buses in column '{col}':")
                                        print(df_full[mask].to_string())
                        except Exception as e:
                            print(f"Error reading more rows: {e}")

                print()  # Blank line between sheets

        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
