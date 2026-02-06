"""Map Extra_WF_edges to OFTO data to determine real ratings and offshore status."""
import pandas as pd

wf = pd.read_excel('data/network/ETYS/GB_network.xlsx', sheet_name='Extra_WF_edges')
xls = pd.ExcelFile('data/network/ETYS/ETYS Appendix B 2023.xlsx')
dfd = xls.parse('B-2-1d', skiprows=1)

print("=== MAPPING EXTRA_WF_EDGES TO OFTO DATA ===")
print()
for _, row in wf.iterrows():
    n1, n2 = row['Node 1'], row['Node 2']
    uid = row['Unique ID']
    
    # Find OFTO entries involving Node 1
    ofto_matches = dfd[(dfd['Node 1'] == n1) | (dfd['Node 2'] == n1)]
    
    if len(ofto_matches) > 0:
        farm_name_series = ofto_matches['Station'].dropna()
        farm_name = farm_name_series.iloc[0] if len(farm_name_series) > 0 else '?'
        max_rating = ofto_matches['Rating (MVA)'].max()
        total_rating = ofto_matches['Rating (MVA)'].sum()
        cable_len = ofto_matches['Cable Length(km)'].max()
        n_circuits = len(ofto_matches)
        print(f"  {uid:>15s}  {n1:>8s} -> {n2:>8s}  OFTO: {farm_name}")
        print(f"                  Rating: max={max_rating:.0f} MVA, sum={total_rating:.0f} MVA, cable={cable_len:.1f} km, circuits={n_circuits}")
    else:
        found_in = []
        for sheet in ['B-2-1a', 'B-2-1b', 'B-2-1c']:
            dfs = xls.parse(sheet, skiprows=1)
            if 'Node 1' in dfs.columns:
                matches = dfs[(dfs['Node 1'] == n1) | (dfs['Node 2'] == n1)]
                if len(matches) > 0:
                    found_in.append(f"{sheet}({len(matches)})")
        location = ", ".join(found_in) if found_in else "NOWHERE"
        print(f"  {uid:>15s}  {n1:>8s} -> {n2:>8s}  NOT in OFTO. Found in: {location}")
    print()

# Now identify which Node1 buses are OFFSHORE PLATFORMS
# An offshore platform bus appears in B-2-1d connected to a cable going out to sea
# We can identify them by: they have long submarine cables connecting to offshore turbines
print()
print("=== ALL OFFSHORE PLATFORM BUSES (from B-2-1d OFTO data) ===")
print("Looking for buses that are onshore connection points for offshore cables...")
print()

# Get all unique buses from B-2-1d 
all_ofto_buses = set(dfd['Node 1'].dropna().tolist() + dfd['Node 2'].dropna().tolist())
# Remove non-string entries
all_ofto_buses = {b for b in all_ofto_buses if isinstance(b, str)}

# Find buses that appear in OFTO but NOT in main ETYS (B-2-1a/b/c or B-3-1a/b/c/d)
main_buses = set()
for sheet in ['B-2-1a', 'B-2-1b', 'B-2-1c', 'B-3-1a', 'B-3-1b', 'B-3-1c', 'B-3-1d']:
    dfs = xls.parse(sheet, skiprows=1)
    if 'Node 1' in dfs.columns:
        main_buses.update(dfs['Node 1'].dropna().astype(str).tolist())
    if 'Node 2' in dfs.columns:
        main_buses.update(dfs['Node 2'].dropna().astype(str).tolist())

# Buses ONLY in OFTO data = likely offshore platforms
ofto_only = all_ofto_buses - main_buses
print(f"Buses that appear ONLY in OFTO (B-2-1d), not in main ETYS:")
for b in sorted(ofto_only):
    # Get their connections from B-2-1d
    conns = dfd[(dfd['Node 1'] == b) | (dfd['Node 2'] == b)]
    ratings = conns['Rating (MVA)'].tolist()
    cables = conns['Cable Length(km)'].tolist()
    print(f"  {b:>10s}  ratings={ratings}  cables={cables}")

print()
print(f"Total OFTO-only buses: {len(ofto_only)}")
print(f"Total OFTO buses: {len(all_ofto_buses)}")
print(f"Total main ETYS buses: {len(main_buses)}")

# Now the key question: which Extra_WF_edges Node1 buses are the offshore ones?
print()
print("=== EXTRA_WF_EDGES NODE1: OFFSHORE OR ONSHORE? ===")
wf_node1s = set(wf['Node 1'].tolist())
for bus in sorted(wf_node1s):
    in_ofto = bus in all_ofto_buses
    in_main = bus in main_buses
    in_ofto_only = bus in ofto_only
    
    # Get OFTO connection info
    if in_ofto:
        conns = dfd[(dfd['Node 1'] == bus) | (dfd['Node 2'] == bus)]
        cable_max = conns['Cable Length(km)'].max()
        rating_max = conns['Rating (MVA)'].max()
        info = f"cable_max={cable_max:.1f}km, rating_max={rating_max:.0f}MVA"
    else:
        info = "no OFTO data"
    
    status = "OFFSHORE-ONLY" if in_ofto_only else ("BOTH" if in_ofto and in_main else "MAIN-ONLY")
    print(f"  {bus:>8s}  {status:>14s}  {info}")
