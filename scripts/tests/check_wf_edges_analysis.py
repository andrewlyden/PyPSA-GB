"""Analyse which Extra_WF_edges already exist in ETYS raw data and identify offshore stubs."""
import pandas as pd

wf = pd.read_excel('data/network/ETYS/GB_network.xlsx', sheet_name='Extra_WF_edges')

# Read ETYS line data to find all existing node pairs
xls = pd.ExcelFile('data/network/ETYS/ETYS Appendix B 2023.xlsx')
etys_pairs = set()
for sheet in ['B-2-1a', 'B-2-1b', 'B-2-1c', 'B-2-1d']:
    df = xls.parse(sheet, skiprows=1)
    if 'Node 1' not in df.columns:
        continue
    for _, row in df.iterrows():
        n1 = str(row['Node 1']).strip()
        n2 = str(row['Node 2']).strip()
        etys_pairs.add((n1, n2))
        etys_pairs.add((n2, n1))

print("=== CHECKING WHICH EXTRA_WF_EDGES ALREADY EXIST IN ETYS ===")
for _, row in wf.iterrows():
    n1, n2 = row['Node 1'], row['Node 2']
    in_etys = (n1, n2) in etys_pairs
    status = "ALREADY IN ETYS" if in_etys else "NEW (not in ETYS)"
    uid = row['Unique ID']
    print(f"  {uid:>15s}  {n1:>8s} -> {n2:>8s}  {status}")

# Identify offshore vs onshore stubs
print()
print("=== IDENTIFYING OFFSHORE vs ONSHORE STUBS ===")
print("(If Node 1 and Node 2 share 4-char prefix, likely internal substation link)")
print("(If different prefix, likely offshore-to-onshore cable)")
for _, row in wf.iterrows():
    n1, n2 = row['Node 1'], row['Node 2']
    same_prefix = n1[:4] == n2[:4]
    uid = row['Unique ID']
    kind = "SAME-SITE stub" if same_prefix else "CROSS-SITE cable"
    print(f"  {uid:>15s}  {n1:>8s} -> {n2:>8s}  {kind}")

# Now check the B-2-1d sheet specifically - it has OFTO data (offshore transmission owners)
print()
print("=== ETYS B-2-1d (OFTO LINES) - ALL ENTRIES ===")
dfd = xls.parse('B-2-1d', skiprows=1)
cols_show = [c for c in dfd.columns if c != 'Unnamed: 0']
print(f"Columns: {list(dfd.columns)}")
print(dfd.to_string())

# Check which buses in WF edges DON'T get GSP coordinates
print()
print("=== CHECKING WHICH WF BUSES HAVE NO GSP MATCH ===")
fes = pd.read_excel('data/FES/FES 2024/Regional breakdown of FES data.xlsx',
                     sheet_name='GSP info', skiprows=4, index_col=1)
gsp_prefixes = set(fes.index.astype(str).str[:4])

all_wf_buses = set(wf['Node 1'].tolist() + wf['Node 2'].tolist())
for bus in sorted(all_wf_buses):
    prefix = bus[:4]
    has_gsp = prefix in gsp_prefixes
    print(f"  {bus:>8s}  prefix={prefix}  GSP_match={'YES' if has_gsp else 'NO'}")
