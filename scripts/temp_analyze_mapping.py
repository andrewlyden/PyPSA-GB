"""Temporary script to analyze GSP/Node mapping between GB_network.xlsx and FES data."""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\data\network\ETYS"

# 1. Load Dem_per_node from GB_network.xlsx
print("=" * 80)
print("1. LOADING Dem_per_node FROM GB_network.xlsx")
print("=" * 80)
dem = pd.read_excel(f"{DATA_DIR}\GB_network.xlsx", sheet_name="Dem_per_node")
print(f"Columns: {list(dem.columns)}")
print(f"Shape: {dem.shape}")

# 2. Load FES24 GSP info
print("\n" + "=" * 80)
print("2. LOADING FES24 GSP INFO")
print("=" * 80)
fes24 = pd.read_excel(
    f"{DATA_DIR}\Regional breakdown of FES24 data.xlsx",
    sheet_name="GSP info",
    skiprows=4,
    index_col=1,
)
print(f"FES24 GSP info shape: {fes24.shape}")
print(f"FES24 columns: {list(fes24.columns)}")
print(f"FES24 index name: {fes24.index.name}")
print(f"FES24 first 5 index values: {list(fes24.index[:5])}")

# 3. Load FES23 GSP info
print("\n" + "=" * 80)
print("3. LOADING FES23 GSP INFO")
print("=" * 80)
fes23 = pd.read_excel(
    f"{DATA_DIR}\Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb",
    sheet_name="GSP info",
    skiprows=4,
    index_col=1,
    engine="pyxlsb",
)
print(f"FES23 GSP info shape: {fes23.shape}")
print(f"FES23 columns: {list(fes23.columns)}")
print(f"FES23 index name: {fes23.index.name}")
print(f"FES23 first 5 index values: {list(fes23.index[:5])}")

# 4a. Unique Node IDs
print("\n" + "=" * 80)
print("4a. UNIQUE NODE IDs IN Dem_per_node")
print("=" * 80)
node_ids = dem["Node Id"].unique()
print(f"Number of unique Node IDs: {len(node_ids)}")

# 4b. Unique GSP IDs
print("\n" + "=" * 80)
print("4b. UNIQUE GSP IDs IN Dem_per_node")
print("=" * 80)
gsp_ids = dem["GSP Id"].unique()
print(f"Number of unique GSP IDs: {len(gsp_ids)}")
gsp_group_ids = dem["GSP Group ID"].unique()
print(f"Number of unique GSP Group IDs: {len(gsp_group_ids)}")

# 4c. First 30 rows
print("\n" + "=" * 80)
print("4c. FIRST 30 ROWS OF Dem_per_node")
print("=" * 80)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(dem.head(30).to_string())

# 4d. Scottish GSP examples
print("\n" + "=" * 80)
print("4d. SCOTTISH GSP EXAMPLES (Node IDs starting with D, S, K, B)")
print("=" * 80)
for prefix in ["D", "S", "K", "B"]:
    subset = dem[dem["Node Id"].str.startswith(prefix, na=False)]
    if len(subset) > 0:
        print(f"\n--- Nodes starting with '{prefix}' ({len(subset)} rows) ---")
        print(subset.head(10).to_string())
    else:
        print(f"\n--- No nodes starting with '{prefix}' ---")

# Also check GSP IDs starting with these prefixes
print("\n\n--- GSP IDs starting with D, S, K, B ---")
for prefix in ["D", "S", "K", "B"]:
    matching = [g for g in gsp_ids if str(g).startswith(prefix)]
    print(f"  '{prefix}': {sorted(matching)[:20]}  (total: {len(matching)})")

# 4e. Match Dem_per_node GSP IDs with FES24
print("\n" + "=" * 80)
print("4e. GSP ID MATCHING: Dem_per_node vs FES24")
print("=" * 80)
dem_gsp_set = set(str(g).strip() for g in gsp_ids)
fes24_gsp_set = set(str(g).strip() for g in fes24.index.dropna())
matched = dem_gsp_set & fes24_gsp_set
unmatched_dem = dem_gsp_set - fes24_gsp_set
unmatched_fes24 = fes24_gsp_set - dem_gsp_set

print(f"Dem_per_node GSP IDs: {len(dem_gsp_set)}")
print(f"FES24 GSP IDs: {len(fes24_gsp_set)}")
print(f"Matched: {len(matched)}")
print(f"In Dem_per_node but NOT in FES24: {len(unmatched_dem)}")
if unmatched_dem:
    print(f"  -> {sorted(unmatched_dem)[:30]}")
print(f"In FES24 but NOT in Dem_per_node: {len(unmatched_fes24)}")
if unmatched_fes24:
    print(f"  -> {sorted(unmatched_fes24)[:30]}")

# Also try matching GSP Group IDs
dem_group_set = set(str(g).strip() for g in gsp_group_ids)
matched_group = dem_group_set & fes24_gsp_set
print(f"\nGSP Group ID matching with FES24: {len(matched_group)} of {len(dem_group_set)}")
unmatched_group = dem_group_set - fes24_gsp_set
if unmatched_group:
    print(f"  Unmatched Group IDs: {sorted(unmatched_group)[:30]}")

# 4f. Compare FES23 vs FES24 for Scottish GSPs
print("\n" + "=" * 80)
print("4f. FES23 vs FES24 COMPARISON (Scottish GSPs)")
print("=" * 80)
fes23_gsp_set = set(str(g).strip() for g in fes23.index.dropna())

print(f"FES23 total GSP IDs: {len(fes23_gsp_set)}")
print(f"FES24 total GSP IDs: {len(fes24_gsp_set)}")

only_in_23 = fes23_gsp_set - fes24_gsp_set
only_in_24 = fes24_gsp_set - fes23_gsp_set
print(f"\nOnly in FES23 (not in FES24): {len(only_in_23)}")
if only_in_23:
    print(f"  -> {sorted(only_in_23)[:30]}")
print(f"Only in FES24 (not in FES23): {len(only_in_24)}")
if only_in_24:
    print(f"  -> {sorted(only_in_24)[:30]}")

# Compare coordinates for Scottish GSPs
scottish_prefixes = ["D", "S", "K", "B"]
print("\n--- Coordinate comparison for Scottish GSPs ---")

# Find lat/lon columns
lat_col_24 = [c for c in fes24.columns if "lat" in c.lower()]
lon_col_24 = [c for c in fes24.columns if "lon" in c.lower()]
lat_col_23 = [c for c in fes23.columns if "lat" in c.lower()]
lon_col_23 = [c for c in fes23.columns if "lon" in c.lower()]
print(f"FES24 lat/lon cols: {lat_col_24}, {lon_col_24}")
print(f"FES23 lat/lon cols: {lat_col_23}, {lon_col_23}")

if lat_col_24 and lon_col_24 and lat_col_23 and lon_col_23:
    lat24, lon24 = lat_col_24[0], lon_col_24[0]
    lat23, lon23 = lat_col_23[0], lon_col_23[0]

    common_gsps = fes23_gsp_set & fes24_gsp_set
    scottish_common = sorted([g for g in common_gsps if any(g.startswith(p) for p in scottish_prefixes)])

    print(f"\nScottish GSPs in both FES23 and FES24: {len(scottish_common)}")
    if scottish_common:
        diffs = []
        for gsp in scottish_common:
            try:
                lat_23_val = fes23.loc[gsp, lat23]
                lon_23_val = fes23.loc[gsp, lon23]
                lat_24_val = fes24.loc[gsp, lat24]
                lon_24_val = fes24.loc[gsp, lon24]
                # Handle Series (duplicate indices)
                if isinstance(lat_23_val, pd.Series):
                    lat_23_val = lat_23_val.iloc[0]
                if isinstance(lon_23_val, pd.Series):
                    lon_23_val = lon_23_val.iloc[0]
                if isinstance(lat_24_val, pd.Series):
                    lat_24_val = lat_24_val.iloc[0]
                if isinstance(lon_24_val, pd.Series):
                    lon_24_val = lon_24_val.iloc[0]
                if (abs(float(lat_23_val) - float(lat_24_val)) > 0.001 or
                        abs(float(lon_23_val) - float(lon_24_val)) > 0.001):
                    diffs.append((gsp, lat_23_val, lon_23_val, lat_24_val, lon_24_val))
            except Exception as e:
                print(f"  Error for {gsp}: {e}")

        if diffs:
            print(f"\n  Coordinate differences found ({len(diffs)}):")
            for gsp, la23, lo23, la24, lo24 in diffs:
                print(f"    {gsp}: FES23=({la23}, {lo23}) vs FES24=({la24}, {lo24})")
        else:
            print("  No coordinate differences found for Scottish GSPs.")

        # Show all Scottish GSPs from FES24
        print(f"\n  All Scottish GSPs in FES24 (first 20):")
        scottish_fes24 = sorted([g for g in fes24_gsp_set if any(g.startswith(p) for p in scottish_prefixes)])
        for g in scottish_fes24[:20]:
            try:
                print(f"    {g}: lat={fes24.loc[g, lat24]}, lon={fes24.loc[g, lon24]}")
            except Exception:
                print(f"    {g}: (error reading coords)")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
