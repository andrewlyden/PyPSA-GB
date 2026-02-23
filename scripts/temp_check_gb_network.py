import pandas as pd
import openpyxl

# Check what sheets GB_network.xlsx has
wb = openpyxl.load_workbook('data/network/ETYS/GB_network.xlsx', read_only=True)
print("GB_network.xlsx sheets:", wb.sheetnames)
wb.close()

# Check each sheet for coordinate data
xls = pd.ExcelFile('data/network/ETYS/GB_network.xlsx')
for sheet in xls.sheet_names:
    df = xls.parse(sheet, nrows=5)
    print(f"\n=== {sheet} ===")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    if len(df) > 0:
        print(df.head(3).to_string())

# Look for a bus coordinates sheet
for sheet in xls.sheet_names:
    df = xls.parse(sheet)
    cols_lower = [str(c).lower() for c in df.columns]
    if any('lat' in c or 'lon' in c or 'easting' in c or 'northing' in c or 'x' == c or 'y' == c for c in cols_lower):
        print(f"\n*** COORDINATE SHEET FOUND: {sheet} ***")
        print(f"Columns: {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print(df.head(10).to_string())

xls.close()
