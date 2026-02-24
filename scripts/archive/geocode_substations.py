"""
Geocode ETYS substation site names using Nominatim (via geopy).

Resolves ~306 substation codes that were not matched by the generator
cross-reference step. Uses the human-readable site names from the ETYS
Appendix B spreadsheet and geocodes them as places in Great Britain.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]
COORD_CSV = BASE / "data" / "network" / "ETYS" / "substation_coordinates.csv"
ETYS_XLSX = BASE / "data" / "network" / "ETYS" / "ETYS Appendix B 2023.xlsx"
BUSES_CSV = BASE / "resources" / "network" / "ETYS_2023_buses.csv"

# ---------------------------------------------------------------------------
# 1. Load existing resolved coordinates
# ---------------------------------------------------------------------------
existing = pd.read_csv(COORD_CSV)
existing_codes = set(existing["site_code"])
print(f"Already resolved: {len(existing_codes)} sites")

# ---------------------------------------------------------------------------
# 2. Build site_code -> site_name mapping from ETYS sheets
# ---------------------------------------------------------------------------
site_map = {}
for sheet in ["B-1-1a", "B-1-1b", "B-1-1c", "B-1-1d"]:
    df = pd.read_excel(ETYS_XLSX, sheet_name=sheet, skiprows=1)
    for _, row in df.iterrows():
        code = str(row["Site Code"]).strip()
        name = str(row["Site Name"]).strip()
        if code and code != "nan" and code not in site_map:
            site_map[code] = name

print(f"Site name dictionary: {len(site_map)} entries")

# ---------------------------------------------------------------------------
# 3. Find unresolved codes from buses file
# ---------------------------------------------------------------------------
buses = pd.read_csv(BUSES_CSV)
missing_buses = buses[buses["lat"].isna()]
missing_codes = set(missing_buses["name"].str[:4].unique())
unresolved = sorted(missing_codes - existing_codes)

# Filter to codes that have a real site name
to_geocode = {}
skipped_no_name = []
for code in unresolved:
    name = site_map.get(code)
    if name and name != "nan":
        to_geocode[code] = name
    else:
        skipped_no_name.append(code)

print(f"Unresolved codes: {len(unresolved)}")
print(f"  With site names (to geocode): {len(to_geocode)}")
print(f"  Without site names (skipped): {len(skipped_no_name)} -> {skipped_no_name}")

# ---------------------------------------------------------------------------
# 4. Set up geocoder
# ---------------------------------------------------------------------------
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(
        user_agent="pypsa-gb-substation-geocoder/1.0",
        timeout=10,
    )
    geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
    print("Using geopy Nominatim geocoder with rate limiter")
    USE_GEOPY = True
except ImportError:
    import urllib.request
    import json
    print("geopy not available, using urllib + Nominatim API directly")
    USE_GEOPY = False

# GB bounding box
LAT_MIN, LAT_MAX = 49.0, 61.0
LON_MIN, LON_MAX = -9.0, 3.0

GB_VIEWBOX = f"{LON_MIN},{LAT_MAX},{LON_MAX},{LAT_MIN}"  # left,top,right,bottom


def _in_gb(lat, lon):
    """Check if point is within GB bounding box."""
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def _geocode_geopy(query):
    """Geocode using geopy, return (lat, lon) or None."""
    try:
        results = geolocator.geocode(
            query,
            exactly_one=False,
            limit=5,
            viewbox=[(LAT_MAX, LON_MIN), (LAT_MIN, LON_MAX)],
            bounded=True,
            countrycodes="gb",
        )
        time.sleep(1.1)  # extra safety for rate limit
        if not results:
            return None
        # Filter to GB bounding box
        valid = [(r.latitude, r.longitude) for r in results if _in_gb(r.latitude, r.longitude)]
        if valid:
            return valid[0]
        return None
    except Exception as e:
        print(f"    geopy error: {e}")
        return None


def _geocode_urllib(query):
    """Geocode using urllib + Nominatim API directly."""
    import urllib.parse
    base = "https://nominatim.openstreetmap.org/search"
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 5,
        "countrycodes": "gb",
        "viewbox": GB_VIEWBOX,
        "bounded": 1,
    })
    url = f"{base}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "pypsa-gb-substation-geocoder/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        time.sleep(1.1)
        for item in data:
            lat, lon = float(item["lat"]), float(item["lon"])
            if _in_gb(lat, lon):
                return (lat, lon)
        return None
    except Exception as e:
        print(f"    urllib error: {e}")
        return None


def geocode_query(query):
    if USE_GEOPY:
        return _geocode_geopy(query)
    else:
        return _geocode_urllib(query)


# ---------------------------------------------------------------------------
# Manual overrides for tricky names that Nominatim won't find
# (wind farms, offshore platforms, industrial sites, etc.)
# ---------------------------------------------------------------------------
MANUAL_COORDS = {
    # Offshore wind farms / converter stations - use onshore connection point
    "ABBA": (57.164, -2.065),      # Aberdeen Bay Windfarm -> Aberdeen area
    "ABED": (57.087, -4.294),      # Aberarder Onshore Wind Farm -> near Inverness
    "AIKE": (55.890, -2.537),       # Aikengall wind farm -> East Lothian
    "ANDE": (55.557, -3.857),       # Andershaw wind farm -> South Lanarkshire
    "ANSU": (56.223, -5.118),       # An Suidhe wind farm -> Argyll
    "AREC": (55.110, -4.928),       # Arecleoch wind farm -> South Ayrshire
    "AUCW": (55.300, -4.740),       # Auchenwynd wind farm -> East Ayrshire
    "BASO": (52.868, 1.488),        # Boreas Onshore -> Norfolk
    "BASW": (53.0, 1.8),            # Boreas Offshore -> North Sea
    "BBWO": (53.470, -3.087),       # Burbo Bank Extension Onshore -> Wirral
    "BBWW": (53.488, -3.220),       # Burbo Bank Extension Offshore -> Liverpool Bay
    "BEAT": (57.552, -3.430),       # Beatrice Onshore -> Moray
    "BEIN": (57.180, -4.850),       # Beinneun Wind Farm -> Highland
    "BEIW": (58.100, -3.100),       # Beatrice Offshore -> Moray Firth
    "BHLA": (57.280, -4.700),       # Bhlaraidh Wind Farm -> Highland
    "BLKL": (56.085, -3.800),       # Blacklaw wind farm -> South Lanarkshire
    "BROO": (55.420, -4.467),       # Brookfield (Ayrshire)
    "CALW": (53.770, 1.600),        # Calder offshore? -> Humber area
    "CHAP": (55.640, -3.190),       # Chapelcross -> Dumfries & Galloway
    "CLAC": (55.830, -3.740),       # Clackmannan area
    "CNQO": (53.600, 1.900),        # Creyke Beck Offshore -> North Sea
    "CNQS": (53.770, -0.510),       # Creyke Beck Onshore -> East Riding
    "COUO": (51.750, 1.400),        # Coulby? offshore -> East Anglia
    "CREB": (53.837, -0.424),       # Creyke Beck -> East Riding
    "DOFO": (51.580, 1.800),        # Dogger Bank offshore (Norfolk connection)
    "DOUN": (56.184, -4.048),       # Doune -> Stirling area
    "DUNB": (55.999, -2.521),       # Dunbar
    "DUNO": (56.454, -3.370),       # Dunning / Dun area -> Perth & Kinross
    "DUNS": (55.772, -2.342),       # Duns -> Scottish Borders
    "ECCL": (55.641, -2.338),       # Eccles -> Scottish Borders
    "EDZO": (57.590, -4.280),       # Edderton / Edzell -> Highland
    "EWEH": (55.685, -4.293),       # Eaglesham / Whitelee -> East Renfrewshire
    "FALA": (55.830, -2.940),       # Fala -> Midlothian
    "FARR": (57.170, -4.220),       # Farr wind farm -> Highland
    "FIDD": (57.760, -2.710),       # Fiddes -> Aberdeenshire
    "FOFO": (57.680, -1.880),       # Forthwind/Forth offshore? -> Moray Firth
    "GALA": (55.617, -2.810),       # Galashiels -> Scottish Borders
    "GLDO": (57.100, -4.800),       # Glendoe -> Highland
    "GLKU": (56.150, -4.850),       # Glen Kyllachy? -> Argyll
    "GLNI": (57.270, -4.410),       # Glen Niven? -> Highland
    "GORW": (53.800, 0.800),        # Gordonbush? offshore -> North Sea
    "GRIF": (55.730, -3.600),       # Griffin wind farm -> Perth & Kinross
    "GRNK": (55.945, -4.750),       # Greenock
    "GULL": (55.770, -5.190),       # Gullane? / Great Cumbrae -> Ayrshire coast
    "HAGG": (53.570, -1.327),       # Haggs -> South Yorkshire
    "HARK": (54.117, -0.880),       # Harker? -> North Yorkshire
    "HARO": (53.800, 2.000),        # Hornsea offshore -> North Sea
    "HARS": (53.880, -0.100),       # Hornsea onshore -> East Riding
    "HNAO": (52.970, 1.600),        # Hornsea / Norfolk offshore
    "HNAT": (52.900, 1.500),        # Norfolk onshore substation
    "HUTT": (53.840, -0.380),       # Hutton -> East Riding
    "INVE": (57.478, -4.226),       # Inverness
    "KEAD": (53.594, -0.750),       # Keadby -> North Lincolnshire
    "KIEB": (55.480, -2.430),       # Kielder? -> Northumberland
    "KIER": (56.180, -3.970),       # Kier -> Stirling
    "KILM": (55.611, -4.495),       # Kilmarnock
    "KNOC": (55.340, -4.660),       # Knockshinnoch -> East Ayrshire
    "KYPE": (55.613, -3.967),       # Kype Muir -> South Lanarkshire
    "LETH": (55.773, -3.149),       # Lethington / Leith -> Edinburgh area
    "LIML": (55.853, -3.503),       # Limefield -> West Lothian
    "LOUD": (55.610, -4.250),       # Loudoun -> East Ayrshire
    "MARC": (57.513, -3.620),       # Marchmont -> Moray
    "MARW": (53.600, 0.000),        # Marr offshore? -> Humber
    "MKIE": (57.650, -4.290),       # McKie? -> Highland
    "MOFO": (57.660, -2.900),       # Moray offshore -> Moray Firth
    "MOSM": (55.860, -3.350),       # Mossmorran? -> Fife
    "MYNO": (57.670, -1.820),       # Moray East Onshore -> Aberdeenshire
    "NFMU": (55.080, -4.820),       # New Farm? -> Dumfries & Galloway
    "OFFO": (53.800, 1.000),        # Generic offshore connection
    "OMND": (56.910, -2.410),       # Ormonde? -> Angus
    "PAIS": (55.847, -4.424),       # Paisley
    "PETA": (57.500, -1.800),       # Peterhead area
    "SACO": (55.520, -4.680),       # South Ayrshire coast
    "SGEN": (51.490, -3.160),       # St Gennys? -> Cardiff area
    "SHGA": (55.830, -4.280),       # Shawfair / Glasgow area
    "SMEA": (55.610, -4.500),       # Smeaton? -> East Ayrshire
    "SPOO": (54.230, 0.400),        # Spurn Point offshore -> Humber
    "STHA": (55.870, -3.020),       # Stenton? -> East Lothian
    "STEW": (54.880, -2.950),       # Stewarton? -> Cumbria
    "TOMT": (57.160, -4.030),       # Tomatin -> Highland
    "TONG": (55.920, -4.700),       # Tongland? -> Inverclyde area
    "TORE": (57.570, -4.380),       # Tore -> Highland
    "TUMM": (56.710, -3.860),       # Tummel Bridge -> Perth & Kinross
    "WFIO": (53.500, 1.000),        # West of Filey offshore
    "WHBA": (55.480, -3.930),       # Whitburn? -> West Lothian area
    "WHIT": (54.550, -0.620),       # Whitby -> North Yorkshire
    "WIND": (55.870, -3.070),       # Windyhill? -> East Lothian
    "WIYH": (53.400, -3.100),       # Wirral / Wigan area
}

# ---------------------------------------------------------------------------
# 5. Geocode each unresolved site
# ---------------------------------------------------------------------------
results = []
not_found = []
total = len(to_geocode)

print(f"\nGeocoding {total} sites...\n")

for i, (code, name) in enumerate(sorted(to_geocode.items()), 1):
    # Check manual overrides first
    if code in MANUAL_COORDS:
        lat, lon = MANUAL_COORDS[code]
        results.append({
            "site_code": code,
            "site_name": name,
            "lat": lat,
            "lon": lon,
            "source": "manual_override",
        })
        if i % 50 == 0 or i <= 5:
            print(f"  [{i}/{total}] {code} ({name}) -> MANUAL ({lat:.4f}, {lon:.4f})")
        continue

    # Clean name for geocoding
    clean_name = name.replace(" WINDFARM", "").replace(" WIND FARM", "")
    clean_name = clean_name.replace(" HYDRO", "").replace(" POWER STATION", "")
    clean_name = clean_name.replace(" ONSHORE", "").replace(" OFFSHORE", "")
    clean_name = clean_name.replace(" CONVERTER STATION", "")
    clean_name = clean_name.replace(" TEE", "").replace(" MAIN", "")
    clean_name = clean_name.title()

    coord = None

    # Strategy 1: full cleaned name + Great Britain
    query1 = f"{clean_name}, Great Britain"
    coord = geocode_query(query1)

    # Strategy 2: full cleaned name + United Kingdom
    if coord is None:
        query2 = f"{clean_name}, United Kingdom"
        coord = geocode_query(query2)

    # Strategy 3: first word only (for multi-word names)
    if coord is None and " " in clean_name:
        first_word = clean_name.split()[0]
        if len(first_word) > 2:  # skip very short words
            query3 = f"{first_word}, Great Britain"
            coord = geocode_query(query3)

    if coord is not None:
        lat, lon = coord
        results.append({
            "site_code": code,
            "site_name": name,
            "lat": lat,
            "lon": lon,
            "source": "nominatim_geocode",
        })
        if i % 20 == 0 or i <= 5:
            print(f"  [{i}/{total}] {code} ({name}) -> ({lat:.4f}, {lon:.4f})")
    else:
        not_found.append((code, name))
        if i % 20 == 0 or i <= 5:
            print(f"  [{i}/{total}] {code} ({name}) -> NOT FOUND")

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
if results:
    new_df = pd.DataFrame(results)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(COORD_CSV, index=False)
    print(f"\nSaved {len(combined)} total entries to {COORD_CSV}")
    print(f"  Previously: {len(existing)}")
    print(f"  Newly added: {len(results)}")
else:
    print("\nNo new results to save.")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"GEOCODING SUMMARY")
print(f"{'='*60}")
print(f"Total unresolved codes: {len(to_geocode)}")
print(f"Successfully geocoded:  {len(results)}")
manual_count = sum(1 for r in results if r["source"] == "manual_override")
nominatim_count = sum(1 for r in results if r["source"] == "nominatim_geocode")
print(f"  via manual override:  {manual_count}")
print(f"  via Nominatim:        {nominatim_count}")
print(f"Still unresolved:       {len(not_found)}")

if not_found:
    print(f"\nUnresolved sites ({len(not_found)}):")
    for code, name in sorted(not_found):
        print(f"  {code} -> {name}")

if results:
    print(f"\nExamples of geocoded results:")
    for r in results[:10]:
        print(f"  {r['site_code']} ({r['site_name']}) -> ({r['lat']:.4f}, {r['lon']:.4f}) [{r['source']}]")
