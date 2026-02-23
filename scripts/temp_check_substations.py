import pandas as pd
import math

csv_path = r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\data\network\ETYS\substation_coordinates.csv"

df = pd.read_csv(csv_path)

# Expected coordinates (lat, lon)
expected = {
    'BEAT': (57.5, -3.0, 'Beatrice Onshore'),
    'BEIW': (58.1, -3.0, 'Beatrice Offshore'),
    'THUS': (58.6, -3.5, 'Thurso South'),
    'SPIT': (58.4, -3.5, 'Spittal'),
    'FYRI': (57.7, -4.4, 'Fyrish'),
    'GLFA': (57.4, -3.3, 'Glenfarclas'),
    'GRIF': (56.7, -3.9, 'Griffin'),
    'FAAR': (57.4, -4.1, 'Farr'),
    'MORF': (57.5, -3.0, 'Moray East'),
    'MORO': (57.5, -3.0, 'Moray East'),
    'MOWE': (57.5, -3.0, 'Moray East'),
    'TUMM': (56.7, -3.9, 'Tummel'),
    'TEAL': (56.5, -3.0, 'Tealing'),
    'WIYH': (55.9, -4.3, 'Windyhill'),
    'STEW': (54.97, -1.76, 'Stella West'),
    'CLYS': (55.5, -3.6, 'Clyde South'),
    'GLAP': (55.0, -5.1, 'Glen App'),
    'CULL': (57.4, -4.85, 'Culligran'),
    'KNOC': (57.45, -4.22, 'Knocknagael'),
    'DENN': (56.0, -3.9, 'Denny North'),
    'FOYE': (57.25, -4.5, 'Foyers'),
    'PUDM': (51.5, 0.0, 'Pudding Mill Lane'),
    'SFEE': (57.6, -1.8, 'St Fergus East'),
    'ERRO': (56.8, -4.1, 'Errochty'),
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points"""
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

print("Checking substation coordinates:\n")
print(f"{'Code':<6} {'Name':<30} {'Actual (lat,lon)':<25} {'Expected (lat,lon)':<25} {'Distance (km)':<15} {'Status':<10}")
print("-" * 120)

for code, (exp_lat, exp_lon, name) in expected.items():
    rows = df[df['site_code'] == code]

    if rows.empty:
        print(f"{code:<6} {name:<30} {'NOT FOUND':<25} {f'({exp_lat}, {exp_lon})':<25} {'--':<15} {'MISSING':<10}")
        continue

    row = rows.iloc[0]
    act_lat = row['lat']
    act_lon = row['lon']
    actual_name = row['site_name']

    distance = haversine_distance(act_lat, act_lon, exp_lat, exp_lon)

    if distance <= 30:
        status = "OK"
    elif distance <= 100:
        status = "CLOSE"
    else:
        status = "WRONG"

    print(f"{code:<6} {actual_name:<30} ({act_lat:>7.4f}, {act_lon:>8.4f})   ({exp_lat:>7.2f}, {exp_lon:>8.2f})   {distance:>10.1f}        {status:<10}")

print("\n" + "="*120)
print("\nSummary:")
print("- OK: Within 30 km")
print("- CLOSE: Within 30-100 km")
print("- WRONG: > 100 km")
