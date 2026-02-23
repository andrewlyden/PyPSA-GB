"""
Fix wrongly-geocoded substation coordinates using REPD data.

Matches ETYS substation names to REPD renewable energy project names
and extracts accurate coordinates where available.

Only uses high-confidence matches to avoid false positives.
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
import re
import os
import sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPD_PATH = os.path.join(BASE, "data", "renewables", "repd-q2-jul-2025.csv")
ETYS_PATH = os.path.join(BASE, "data", "network", "ETYS", "ETYS Appendix B 2023.xlsx")
COORDS_PATH = os.path.join(BASE, "data", "network", "ETYS", "substation_coordinates.csv")

# -- 1. Load REPD ---------------------------------------------------------------
print("Loading REPD...")
repd = pd.read_csv(REPD_PATH, encoding="latin-1")
print(f"  {len(repd)} REPD records loaded")

# Convert OSGB to WGS84
transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# Clean coordinates
repd["X-coordinate"] = pd.to_numeric(repd["X-coordinate"], errors="coerce")
repd["Y-coordinate"] = pd.to_numeric(repd["Y-coordinate"], errors="coerce")
repd = repd.dropna(subset=["X-coordinate", "Y-coordinate"])

# Convert all coordinates
lons, lats = transformer.transform(
    repd["X-coordinate"].values, repd["Y-coordinate"].values
)
repd["repd_lat"] = lats
repd["repd_lon"] = lons

# Normalize site names for matching - handle NaN
repd["site_name_clean"] = repd["Site Name"].fillna("").astype(str).str.upper().str.strip()
repd["site_name_alnum"] = repd["site_name_clean"].apply(
    lambda x: re.sub(r"[^\w\s]", "", x)
)

print(f"  {len(repd)} REPD records with valid coordinates")

# -- 2. Load current substation coordinates --------------------------------------
print("\nLoading current substation_coordinates.csv...")
coords_orig = pd.read_csv(COORDS_PATH)
# Work on a copy so we can compare
coords = coords_orig.copy()
print(f"  {len(coords)} entries")

# -- 3. Define MANUAL matches only -----------------------------------------------
# Only match entries where we have HIGH CONFIDENCE the REPD project corresponds
# to the ETYS substation. This avoids false positives from substring matching.
#
# Format: site_code -> (search_term, description)
MANUAL_MATCHES = {
    # Onshore wind farms - high confidence
    "GRIF": "GRIFFIN WIND",
    "FAAR": "FARR WIND FARM",
    "TUMM": "TUMMEL BRIDGE POWER STATION",
    # Offshore wind - Moray East
    "MORF": "MORAY EAST",
    "MORO": "MORAY EAST",
    "MOWE": "MORAY EAST",
    # Offshore wind - Beatrice
    "BEIW": "BEATRICE",
    # Offshore wind - Dudgeon
    "DUDO": "DUDGEON",
    "DUDW": "DUDGEON",
    # Offshore wind - East Anglia
    "EAAW": "EAST ANGLIA 1",
    # Offshore wind - Greater Gabbard
    "GGON": "GREATER GABBARD",
    "GREG": "GREATER GABBARD",
    # Offshore wind - London Array
    "LOAW": "LONDON ARRAY",
    "LONO": "LONDON ARRAY",
    "LOYW": "LONDON ARRAY",
    # Offshore wind - other
    "HOWW": "HORNSEA",
    "HUMO": "HUMBER GATEWAY",
    "HUMW": "HUMBER GATEWAY",
    "NNGO": "NEART NA GAOITHE",
    "NNGW": "NEART NA GAOITHE",
    "ORMO": "ORMONDE OFFSHORE",
    "ORMW": "ORMONDE OFFSHORE",
    "RAMW": "RAMPION OFFSHORE",
    "TKNO": "TRITON KNOLL",
    "WACW": "WALNEY 3",
    "WDSO": "WEST OF DUDDON SANDS",
    "WDSW": "WEST OF DUDDON SANDS",
    "GALO": "GALLOPER WIND",
    "GANW": "GALLOPER WIND",
    # Onshore wind farms - specific matches
    "ANSU": "AN SUIDHE",
    "BHLA": "BHLARAIDH",
    "BLCW": "BLACKCRAIG",
    "BLKS": "BLACKCRAIG",
    "COGA": "CORRIEGARTH",
    "CRDY": "CROSSDYKES",
    "CREA": "CREAG RIABHACH",
    "CLYN": "CLYDE WIND FARM",
    "CLYS": "CLYDE WIND FARM",
    "EWEH": "EWE HILL",
    "GLAP": "GLEN APP",
    "HARE": "HARESTANES",
    "KYLL": "GLEN KYLLACHY",
    "MAHI": "MARK HILL",
    "SAKN": "SANDY KNOWE WIND",
    "STRW": "STRATHY NORTH",
    "WHLL": "WHITESIDE HILL",
    "WLEE": "WHITELEE WIND FARM",
    "WLEX": "WHITELEE WIND FARM",
    # Hydro
    "FFES": "FFESTINIOG",
    # Solar / other with clear matches
    "LOFI": "LONGFIELD",
    "DAIN": "DAINES BATTERY",
}


def find_repd_by_search(search_term, repd_df):
    """Find best REPD entry matching a search term."""
    term_upper = search_term.upper().strip()

    # Try exact substring in cleaned name
    matches = repd_df[repd_df["site_name_clean"].str.contains(
        re.escape(term_upper), case=False, na=False
    )]

    if matches.empty:
        # Try with special chars removed
        term_alnum = re.sub(r"[^\w\s]", "", term_upper)
        matches = repd_df[repd_df["site_name_alnum"].str.contains(
            re.escape(term_alnum), case=False, na=False
        )]

    if matches.empty:
        return None

    # Prefer operational, then largest capacity
    operational = matches[matches["Development Status (short)"] == "Operational"]
    if len(operational) > 0:
        matches = operational

    matches = matches.sort_values("Installed Capacity (MWelec)", ascending=False)
    return matches.iloc[0]


# -- 4. Process manual matches ---------------------------------------------------
print("\n" + "=" * 80)
print("MATCHING RESULTS")
print("=" * 80)

updates = {}

print("\n-- Curated matches --")
for site_code, search_term in MANUAL_MATCHES.items():
    row = coords[coords["site_code"] == site_code]
    if row.empty:
        print(f"  {site_code}: NOT in substation_coordinates.csv, skipping")
        continue

    current_lat = row.iloc[0]["lat"]
    current_lon = row.iloc[0]["lon"]
    current_name = row.iloc[0]["site_name"]

    best = find_repd_by_search(search_term, repd)

    if best is None:
        print(f"  {site_code} ({current_name}): No REPD match for '{search_term}'")
        continue

    repd_lat = best["repd_lat"]
    repd_lon = best["repd_lon"]
    dist_km = np.sqrt((current_lat - repd_lat)**2 + (current_lon - repd_lon)**2) * 111

    print(f"  {site_code} ({current_name}):")
    print(f"    REPD: '{best['Site Name']}' ({best['Technology Type']}, {best['Installed Capacity (MWelec)']} MW)")
    print(f"    Current:  ({current_lat:.6f}, {current_lon:.6f})")
    print(f"    REPD:     ({repd_lat:.6f}, {repd_lon:.6f})")
    print(f"    Distance: {dist_km:.1f} km")

    if dist_km > 10:
        print(f"    >>> WILL UPDATE")
        updates[site_code] = (repd_lat, repd_lon, best["Site Name"])
    else:
        print(f"    OK - coordinates close enough")

# -- 5. Report entries NOT in REPD -----------------------------------------------
print("\n-- Entries not matchable in REPD (pure substations) --")
not_in_repd = ["STEW", "SFEE", "SPIT", "WIYH"]
for sc in not_in_repd:
    row = coords[coords["site_code"] == sc]
    if not row.empty:
        print(f"  {sc} ({row.iloc[0]['site_name']}): ({row.iloc[0]['lat']:.6f}, {row.iloc[0]['lon']:.6f}) - skipping, not a generation project")

# -- 6. Apply updates -----------------------------------------------------------
print("\n" + "=" * 80)
print(f"APPLYING {len(updates)} UPDATES to substation_coordinates.csv")
print("=" * 80)

for site_code, (new_lat, new_lon, repd_name) in sorted(updates.items()):
    mask = coords["site_code"] == site_code
    old_lat = coords.loc[mask, "lat"].values[0]
    old_lon = coords.loc[mask, "lon"].values[0]
    coords.loc[mask, "lat"] = round(new_lat, 6)
    coords.loc[mask, "lon"] = round(new_lon, 6)
    coords.loc[mask, "source"] = "repd_match"
    print(f"  {site_code}: ({old_lat:.6f}, {old_lon:.6f}) -> ({new_lat:.6f}, {new_lon:.6f})  [REPD: {repd_name}]")

# Save
coords.to_csv(COORDS_PATH, index=False)
print(f"\nSaved {len(updates)} updated coordinates to {COORDS_PATH}")
print("Done!")
