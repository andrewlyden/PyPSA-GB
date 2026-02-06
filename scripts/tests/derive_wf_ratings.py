"""Derive proper s_nom ratings for Extra_WF_edges from raw ETYS OFTO data."""
import pandas as pd

xls = pd.ExcelFile('data/network/ETYS/ETYS Appendix B 2023.xlsx')
dfd = xls.parse('B-2-1d', skiprows=1)
wf = pd.read_excel('data/network/ETYS/GB_network.xlsx', sheet_name='Extra_WF_edges')

rating_col_d = 'Rating (MVA)'
cable_col_d = 'Cable Length(km)'

print("=== DERIVING PROPER RATINGS FOR EXTRA_WF_EDGES ===")
print()

results = []

for _, row in wf.iterrows():
    n1, n2 = row['Node 1'], row['Node 2']
    uid = row['Unique ID']
    
    # Get OFTO circuits where Node1 appears
    ofto_from = dfd[dfd['Node 1'] == n1]
    ofto_to = dfd[dfd['Node 2'] == n1]
    
    if len(ofto_from) > 0 or len(ofto_to) > 0:
        all_ratings = []
        all_cables = []
        if len(ofto_from) > 0:
            all_ratings.extend(ofto_from[rating_col_d].tolist())
            all_cables.extend(ofto_from[cable_col_d].tolist())
        if len(ofto_to) > 0:
            all_ratings.extend(ofto_to[rating_col_d].tolist())
            all_cables.extend(ofto_to[cable_col_d].tolist())
        
        max_rating = max(all_ratings)
        max_cable = max(all_cables)
        
        # The stub capacity should equal the sum of all export circuits from that platform
        # (since multiple circuits can export in parallel)
        total_export = sum(all_ratings)
        
        print(f"  {uid:>15s}  {n1:>8s} -> {n2:>8s}")
        print(f"    OFTO ratings: {all_ratings}")
        print(f"    OFTO cables:  {all_cables}")
        print(f"    -> s_nom = {max_rating:.0f} MVA (max circuit), total = {total_export:.0f} MVA")
        results.append({"uid": uid, "n1": n1, "n2": n2, "s_nom": max_rating, "source": "OFTO"})
        print()
    else:
        # Check main ETYS sheets for rating data
        found = False
        for sheet in ['B-2-1a', 'B-2-1b', 'B-2-1c']:
            dfs = xls.parse(sheet, skiprows=1)
            matches = dfs[(dfs['Node 1'] == n1) | (dfs['Node 2'] == n1)]
            if len(matches) > 0:
                rc = 'Winter Rating (MVA)' if 'Winter Rating (MVA)' in matches.columns else 'Rating (MVA)'
                if rc in matches.columns:
                    ratings = [r for r in matches[rc].tolist() if r != 9999 and pd.notna(r)]
                    if ratings:
                        max_r = max(ratings)
                        print(f"  {uid:>15s}  {n1:>8s} -> {n2:>8s}")
                        print(f"    Main ETYS ({sheet}) ratings: {ratings}")
                        print(f"    -> s_nom = {max_r:.0f} MVA (max from main ETYS)")
                        results.append({"uid": uid, "n1": n1, "n2": n2, "s_nom": max_r, "source": sheet})
                        found = True
                        print()
                        break
        if not found:
            print(f"  {uid:>15s}  {n1:>8s} -> {n2:>8s}  NO RATING DATA FOUND")
            results.append({"uid": uid, "n1": n1, "n2": n2, "s_nom": None, "source": "NONE"})
            print()

print()
print("=== SUMMARY TABLE ===")
print(f"{'UID':>15s}  {'Node1':>8s}  {'Node2':>8s}  {'s_nom':>8s}  Source")
for r in results:
    snom = f"{r['s_nom']:.0f}" if r['s_nom'] is not None else "???"
    print(f"{r['uid']:>15s}  {r['n1']:>8s}  {r['n2']:>8s}  {snom:>8s}  {r['source']}")
