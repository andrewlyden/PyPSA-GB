"""
Generate substation_coordinates.csv by cross-referencing ETYS substation names
with generator coordinate data from DUKES, power_stations_locations, and nominatim_cache.

Only resolves the 343 site codes that have missing coordinates in ETYS_2023_buses.csv.

Usage:
    python scripts/utilities/generate_substation_coordinates.py

Output:
    data/network/ETYS/substation_coordinates.csv
"""

import pandas as pd
import numpy as np
import os
import re
import time

BASE = r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1"

# Common geographic/directional words that should not be used for first-word matching
# as they produce false positives (e.g. "SOUTH KYLE" matching "South Humber Bank")
STOP_WORDS = {
    "NORTH", "SOUTH", "EAST", "WEST", "UPPER", "LOWER", "GREAT", "LITTLE",
    "NEW", "OLD", "BLACK", "WHITE", "GREEN", "LONDON", "PORT", "LOCH",
    "GLEN", "STRATH", "BRIDGE", "CASTLE", "CROSS", "POINT", "WATER",
}

# ============================================================
# 1. Load ETYS site names from Appendix B sheets
# ============================================================
print("=" * 60)
print("1. Loading ETYS site names...")
etys_path = os.path.join(BASE, "data", "network", "ETYS", "ETYS Appendix B 2023.xlsx")
sheets = ["B-1-1a", "B-1-1b", "B-1-1c", "B-1-1d"]

site_map = {}  # site_code -> site_name
site_operator = {}  # site_code -> operator (SHE, SPT, NGET, OFTO)
sheet_operators = {"B-1-1a": "SHE", "B-1-1b": "SPT", "B-1-1c": "NGET", "B-1-1d": "OFTO"}
for sheet in sheets:
    df = pd.read_excel(etys_path, sheet_name=sheet, skiprows=1)
    operator = sheet_operators[sheet]
    for _, row in df.iterrows():
        code = str(row.get("Site Code", "")).strip()
        name = str(row.get("Site Name", "")).strip()
        if len(code) == 4 and code != "nan" and name != "nan":
            site_map[code] = name
            if code not in site_operator:  # first sheet wins
                site_operator[code] = operator

print(f"  Loaded {len(site_map)} ETYS site codes")

# Operator-specific geocoding config: viewbox and region hints
# Viewbox format: (max_lat, min_lon), (min_lat, max_lon) — NW corner, SE corner
OPERATOR_GEO = {
    "SHE": {
        "viewbox": [(61, -8), (55.5, -1)],  # Scotland north
        "region_hint": "Scotland",
        "lat_range": (55.5, 61),
    },
    "SPT": {
        "viewbox": [(57, -6), (54.5, -1)],  # Central Scotland / N. England
        "region_hint": "Scotland",
        "lat_range": (54.5, 57),
    },
    "NGET": {
        "viewbox": [(56, -6), (49, 3)],  # England and Wales
        "region_hint": "England",
        "lat_range": (49, 56),
    },
    "OFTO": {
        "viewbox": [(61, -9), (49, 3)],  # Offshore — wide range
        "region_hint": "",
        "lat_range": (49, 61),
    },
}

# ============================================================
# 2. Identify the 343 missing site codes from ETYS buses
# ============================================================
print("\n2. Identifying missing site codes from ETYS buses...")
buses = pd.read_csv(os.path.join(BASE, "resources", "network", "ETYS_2023_buses.csv"))
missing_buses = buses[buses["lat"].isna()]
missing_codes = set(missing_buses["name"].str[:4].unique())
print(f"  Found {len(missing_codes)} unique site codes with missing coordinates")

# ============================================================
# 3. Load generator coordinate sources
# ============================================================
print("\n3. Loading generator coordinate sources...")

# 3a. DUKES generators - IMPORTANT: filter out regional centroid coordinates
# Most DUKES entries have round x_coord/y_coord (e.g. 280000/680000) which are
# regional centroids, not actual station locations. Only keep rows where at least
# one of x_coord or y_coord is not divisible by 10000.
dukes = pd.read_csv(os.path.join(BASE, "resources", "generators", "DUKES", "DUKES_2023_generators.csv"))

# Filter to valid WGS84 GB range
dukes_wgs84 = dukes[
    dukes["lat"].between(49, 61) &
    dukes["lon"].between(-9, 3) &
    dukes["lat"].notna() &
    dukes["lon"].notna()
].copy()

# Remove regional centroids (x_coord and y_coord both divisible by 10000)
is_centroid = (dukes_wgs84["x_coord"] % 10000 == 0) & (dukes_wgs84["y_coord"] % 10000 == 0)
dukes_valid = dukes_wgs84[~is_centroid].copy()
print(f"  DUKES: {len(dukes_valid)} rows with station-specific coordinates "
      f"(filtered {is_centroid.sum()} centroid rows from {len(dukes_wgs84)} valid WGS84)")

# 3b. Power stations locations
ps = pd.read_csv(
    os.path.join(BASE, "data", "generators", "power_stations_locations.csv"),
    encoding="latin-1"
)
# Parse Geolocation column - contains "lat, lon" with possible special chars
ps_coords = []
for _, row in ps.iterrows():
    name = str(row.get("Station Name", "")).strip()
    geo = str(row.get("Geolocation", ""))
    # Remove non-ASCII chars (there are special chars in the data)
    geo_clean = re.sub(r"[^\d.,\-\s]", "", geo)
    parts = [p.strip() for p in geo_clean.split(",") if p.strip()]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if 49 <= lat <= 61 and -9 <= lon <= 3:
                # Clean station name - remove parenthetical suffixes like "(9)"
                clean_name = re.sub(r"\s*\(\d+\)\s*$", "", name).strip()
                ps_coords.append({"station_name": clean_name, "lat": lat, "lon": lon})
        except ValueError:
            pass
ps_df = pd.DataFrame(ps_coords)
print(f"  Power stations: {len(ps_df)} entries with valid coordinates")

# 3c. Nominatim cache
nom = pd.read_csv(os.path.join(BASE, "data", "generators", "nominatim_cache.csv"))
nom = nom.rename(columns={"latitude": "lat", "longitude": "lon"})
nom = nom[nom["lat"].between(49, 61) & nom["lon"].between(-9, 3)].copy()
# Clean station names - remove asterisks
nom["station_name"] = nom["station_name"].str.replace("*", "", regex=False).str.strip()
print(f"  Nominatim: {len(nom)} entries with valid coordinates")

# Combine all generator sources into one lookup
# Priority order: power_stations > nominatim > dukes (power_stations has hand-verified coords)
all_generators = pd.concat([
    ps_df[["station_name", "lat", "lon"]].assign(source="power_stations"),
    nom[["station_name", "lat", "lon"]].assign(source="nominatim"),
    dukes_valid[["station_name", "lat", "lon"]].assign(source="dukes"),
], ignore_index=True)
all_generators["station_name_clean"] = all_generators["station_name"].str.strip().str.upper()
print(f"  Combined: {len(all_generators)} generator entries for matching")

# ============================================================
# 4. Match ETYS sites to generator data
# ============================================================
print("\n4. Matching ETYS sites to generator coordinates...")

results = {}  # site_code -> {site_name, lat, lon, source}


def add_result(code, name, lat, lon, source):
    """Add a result, only if code is in missing_codes and not already matched."""
    if code in missing_codes and code not in results:
        results[code] = {
            "site_code": code,
            "site_name": name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "source": source,
        }


# Build a lookup of generator names to average coordinates
# Use drop_duplicates to prefer first source (power_stations > nominatim > dukes)
gen_name_coords = (
    all_generators.groupby("station_name_clean")
    .agg(lat=("lat", "mean"), lon=("lon", "mean"), source=("source", "first"))
    .reset_index()
)
gen_name_lookup = {
    row["station_name_clean"]: row for _, row in gen_name_coords.iterrows()
}

# Strategy 1: Exact match of full ETYS site name
for code, name in site_map.items():
    name_upper = name.upper()
    if name_upper in gen_name_lookup:
        r = gen_name_lookup[name_upper]
        add_result(code, name, r["lat"], r["lon"], f"exact_name_match ({r['source']})")

print(f"  After exact name match: {len(results)} resolved")

# Strategy 2: ETYS site name is a substring of a generator name
for code, name in site_map.items():
    if code in results:
        continue
    name_upper = name.upper()
    if len(name_upper) < 4:
        continue
    for gen_name, r in gen_name_lookup.items():
        if name_upper in gen_name:
            add_result(code, name, r["lat"], r["lon"], f"name_in_gen ({r['source']})")
            break

print(f"  After substring (site in gen): {len(results)} resolved")

# Strategy 3: Generator name is a substring of ETYS site name
for code, name in site_map.items():
    if code in results:
        continue
    name_upper = name.upper()
    for gen_name, r in gen_name_lookup.items():
        if len(gen_name) >= 5 and gen_name in name_upper:
            add_result(code, name, r["lat"], r["lon"], f"gen_in_name ({r['source']})")
            break

print(f"  After substring (gen in site): {len(results)} resolved")

# Strategy 4: First-word matching (bidirectional, word length >= 5)
# Skip common geographic stop words to avoid false positives
for code, name in site_map.items():
    if code in results:
        continue
    name_upper = name.upper()
    first_word = name_upper.split()[0] if name_upper else ""
    if len(first_word) >= 5 and first_word not in STOP_WORDS:
        for gen_name, r in gen_name_lookup.items():
            gen_words = gen_name.split()
            if first_word in gen_words:
                add_result(code, name, r["lat"], r["lon"], f"first_word_match ({r['source']})")
                break

# Also try reverse: generator first word in ETYS name words
remaining_codes = missing_codes - set(results.keys())
remaining_site_names = {
    code: site_map.get(code, "").upper() for code in remaining_codes if code in site_map
}
for gen_name, r in gen_name_lookup.items():
    gen_first_word = gen_name.split()[0] if gen_name else ""
    if len(gen_first_word) >= 5 and gen_first_word not in STOP_WORDS:
        for code, etys_name in remaining_site_names.items():
            if code not in results:
                etys_words = etys_name.split()
                if gen_first_word in etys_words:
                    add_result(
                        code,
                        site_map.get(code, etys_name),
                        r["lat"],
                        r["lon"],
                        f"reverse_first_word ({r['source']})",
                    )

print(f"  After first-word matching: {len(results)} resolved")

# ============================================================
# 4a. REPD matching for renewable energy sites
# ============================================================
print("\n4a. Matching to REPD (renewable energy projects)...")

# Keywords that indicate a site IS a renewable project (worth matching to REPD)
# Match explicit renewable project names — require "WIND FARM" or "WINDFARM",
# not just "WIND" (which would match place names like WINDYHILL)
RENEWABLE_KEYWORDS = re.compile(
    r"WIND\s*FARM|WINDFARM|SOLAR|HYDRO|WAVE|TIDAL|BATTERY|STORAGE",
    re.IGNORECASE,
)

repd_path = os.path.join(BASE, "data", "renewables", "repd-q2-jul-2025.csv")
try:
    repd = pd.read_csv(repd_path, encoding="latin-1")
    # REPD has X-coordinate/Y-coordinate in OSGB — convert to WGS84
    from pyproj import Transformer
    osgb_to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    repd_coords = {}  # cleaned name -> (lat, lon)
    for _, row in repd.iterrows():
        try:
            x = float(row.get("X-coordinate", 0))
            y = float(row.get("Y-coordinate", 0))
        except (ValueError, TypeError):
            continue
        site_name = str(row.get("Site Name", "")).strip()
        if pd.notna(x) and pd.notna(y) and x > 0 and y > 0 and site_name:
            lon, lat = osgb_to_wgs.transform(x, y)
            if 49 <= lat <= 61 and -9 <= lon <= 3:
                clean = site_name.upper().strip()
                # Keep first occurrence (REPD can have multiple entries per site)
                if clean not in repd_coords:
                    repd_coords[clean] = (lat, lon)
    print(f"  REPD: {len(repd_coords)} unique projects with coordinates")

    repd_matched = 0
    for code in sorted(missing_codes - set(results.keys())):
        name = site_map.get(code, "")
        if not name:
            continue
        # Only use REPD for sites that look like renewable projects
        if not RENEWABLE_KEYWORDS.search(name):
            continue

        name_upper = name.upper()
        # Clean the ETYS name for matching
        clean_etys = re.sub(
            r"\s*(WIND\s*FARM|WINDFARM|SUBSTATION|\(SSE\)|\(SPT\)|\(NGET\)|\(OFTO\))\s*",
            "", name_upper,
        ).strip()

        # Try exact match on cleaned name
        for repd_name, (lat, lon) in repd_coords.items():
            repd_clean = re.sub(
                r"\s*(WIND\s*FARM|WINDFARM|EXTENSION|PHASE\s*\d+)\s*",
                "", repd_name,
            ).strip()
            if clean_etys == repd_clean or clean_etys in repd_name or repd_clean == clean_etys:
                add_result(code, name, lat, lon, "repd_match")
                repd_matched += 1
                break

    print(f"  REPD matched: {repd_matched} sites")
    print(f"  After REPD: {len(results)} resolved")

except FileNotFoundError:
    print(f"  REPD file not found: {repd_path}")
except ImportError:
    print("  pyproj not available for OSGB conversion, skipping REPD")

# ============================================================
# 4b. Nominatim geocoding for remaining unresolved sites
# ============================================================
print("\n4b. Geocoding remaining sites via Nominatim...")

# Load previous geocoding results as cache to avoid re-querying Nominatim
nominatim_cache = {}
cache_path = os.path.join(BASE, "data", "network", "ETYS", "substation_coordinates.csv")
if os.path.exists(cache_path):
    prev = pd.read_csv(cache_path)
    for _, row in prev.iterrows():
        if row.get("source") == "nominatim_geocode" and pd.notna(row["lat"]):
            nominatim_cache[str(row["site_code"]).strip()] = (row["lat"], row["lon"])
    print(f"  Loaded {len(nominatim_cache)} cached Nominatim results")

remaining_to_geocode = sorted(missing_codes - set(results.keys()))
remaining_with_names = [(code, site_map[code]) for code in remaining_to_geocode if code in site_map]
print(f"  {len(remaining_with_names)} sites to geocode")

try:
    from geopy.geocoders import Nominatim as NominatimGeocoder
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = NominatimGeocoder(user_agent="pypsa-gb-substation-geocoder/1.0", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

    geocoded_count = 0
    failed_geocode = []

    for i, (code, name) in enumerate(remaining_with_names):
        if i % 50 == 0 and i > 0:
            print(f"    Progress: {i}/{len(remaining_with_names)} ({geocoded_count} resolved)")

        # Use cached result if available
        if code in nominatim_cache:
            lat, lon = nominatim_cache[code]
            # Validate cached result against operator region
            operator = site_operator.get(code, "NGET")
            geo_cfg = OPERATOR_GEO.get(operator, OPERATOR_GEO["NGET"])
            lat_min, lat_max = geo_cfg["lat_range"]
            if lat_min <= lat <= lat_max:
                add_result(code, name, lat, lon, "nominatim_geocode")
                geocoded_count += 1
                continue
            # Cached result outside region — re-geocode
            print(f"    Re-geocoding {code} ({name}): cached {lat:.1f} outside {operator} range {lat_min}-{lat_max}")

        # Clean name for geocoding: remove suffixes like "WINDFARM", "WIND FARM",
        # "SUBSTATION", "CONVERTER STATION", "ONSHORE", "OFFSHORE", "(SSE)", "(NGET)"
        clean_name = re.sub(
            r"\s*(WIND\s*FARM|WINDFARM|SUBSTATION|CONVERTER\s*STATION|ONSHORE|OFFSHORE"
            r"|POWER\s*STATION|\(SSE\)|\(SPT\)|\(NGET\)|\(OFTO\))\s*",
            "", name, flags=re.IGNORECASE
        ).strip()

        # Get operator-specific geocoding config
        operator = site_operator.get(code, "NGET")
        geo_cfg = OPERATOR_GEO.get(operator, OPERATOR_GEO["NGET"])
        viewbox = geo_cfg["viewbox"]
        region_hint = geo_cfg["region_hint"]
        lat_min, lat_max = geo_cfg["lat_range"]

        # Build queries with region hint for disambiguation
        queries = []
        if region_hint:
            queries.append(f"{clean_name}, {region_hint}")
        queries.append(f"{clean_name}, Great Britain")
        queries.append(f"{clean_name}, United Kingdom")
        # For multi-word names, also try first word with region
        words = clean_name.split()
        if len(words) > 1 and len(words[0]) >= 4:
            if region_hint:
                queries.append(f"{words[0]}, {region_hint}")
            queries.append(f"{words[0]}, Great Britain")

        location = None
        for query in queries:
            try:
                location = geocode(
                    query,
                    exactly_one=True,
                    viewbox=viewbox,
                    bounded=True,
                    country_codes=["gb"],
                )
            except Exception:
                pass
            # Validate result is within operator's expected lat range
            if (location and
                    lat_min <= location.latitude <= lat_max and
                    -9 <= location.longitude <= 3):
                break
            location = None

        if location:
            add_result(code, name, location.latitude, location.longitude, "nominatim_geocode")
            geocoded_count += 1
        else:
            failed_geocode.append((code, name))

    print(f"  Nominatim geocoding: {geocoded_count} resolved, {len(failed_geocode)} failed")
    print(f"  Total resolved after all strategies: {len(results)}")

except ImportError:
    print("  WARNING: geopy not available, skipping Nominatim geocoding")
    print("  Install with: pip install geopy")

# ============================================================
# 5. Output results
# ============================================================
print("\n5. Writing output...")

if results:
    output_df = pd.DataFrame(list(results.values()))
    output_df = output_df.sort_values("site_code").reset_index(drop=True)
else:
    output_df = pd.DataFrame(columns=["site_code", "site_name", "lat", "lon", "source"])

output_path = os.path.join(BASE, "data", "network", "ETYS", "substation_coordinates.csv")
output_df.to_csv(output_path, index=False)
print(f"  Written {len(output_df)} entries to {output_path}")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total missing site codes: {len(missing_codes)}")
print(f"Resolved: {len(results)}")
print(f"Remaining unresolved: {len(missing_codes) - len(results)}")

# Source breakdown
if len(output_df) > 0:
    print("\nMatches by source type:")
    source_counts = output_df["source"].value_counts()
    for src, count in source_counts.items():
        print(f"  {src}: {count}")

    print("\nSample resolved entries:")
    for _, row in output_df.head(15).iterrows():
        print(f"  {row['site_code']} ({row['site_name']}): "
              f"{row['lat']}, {row['lon']} [{row['source']}]")

# List unresolved
unresolved = sorted(missing_codes - set(results.keys()))
print(f"\nUnresolved codes ({len(unresolved)}):")
for code in unresolved:
    name = site_map.get(code, "UNKNOWN")
    print(f"  {code}: {name}")
