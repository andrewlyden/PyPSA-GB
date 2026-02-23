import pandas as pd
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)

fpath = r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\data\network\ETYS\Regional breakdown of FES24 data.xlsx"

xls = pd.ExcelFile(fpath)
print(f"Sheet names ({len(xls.sheet_names)}):")
for i, name in enumerate(xls.sheet_names):
    print(f"  {i}: '{name}'")

# ---- GSP info sheet ----
print(f"\n{'=' * 80}")
print("SHEET: 'GSP info'")
print(f"{'=' * 80}")

df = pd.read_excel(xls, sheet_name='GSP info')
print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  {i}: '{col}' (dtype={df[col].dtype}, nunique={df[col].nunique()}, nulls={df[col].isna().sum()})")

print(f"\nFirst 30 rows:")
print(df.head(30).to_string())

print(f"\nLast 5 rows:")
print(df.tail(5).to_string())

# Show unique values for columns with few unique values
for col in df.columns:
    nuniq = df[col].nunique()
    if nuniq <= 40:
        vals = df[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except TypeError:
            vals = [str(v) for v in vals]
            vals.sort()
        print(f"\n  '{col}' ({nuniq} unique): {vals}")

# ---- MAIN DATA sheet peek ----
print(f"\n{'=' * 80}")
print("SHEET: 'MAIN DATA' (first 5 rows)")
print(f"{'=' * 80}")
df2 = pd.read_excel(xls, sheet_name='MAIN DATA', nrows=5)
print(f"Shape hint: {df2.shape}")
print(f"Columns: {list(df2.columns)}")
print(df2.head().to_string())

# ---- MAIN DATA Flopzones peek ----
print(f"\n{'=' * 80}")
print("SHEET: 'MAIN DATA Flopzones' (first 5 rows)")
print(f"{'=' * 80}")
df3 = pd.read_excel(xls, sheet_name='MAIN DATA Flopzones', nrows=5)
print(f"Shape hint: {df3.shape}")
print(f"Columns: {list(df3.columns)}")
print(df3.head().to_string())

xls.close()
print("\nDone.")
