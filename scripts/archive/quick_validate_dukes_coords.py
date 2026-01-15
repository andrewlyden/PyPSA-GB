"""
Quick validation of critical DUKES coordinates (nuclear + large generators).
"""

import pandas as pd
import requests
import time

def geocode_postcode(postcode: str) -> dict:
    """Geocode a UK postcode using postcodes.io API."""
    if pd.isna(postcode) or not postcode:
        return {'success': False}
    
    clean_postcode = str(postcode).replace(' ', '').strip().upper()
    
    try:
        response = requests.get(
            f'https://api.postcodes.io/postcodes/{clean_postcode}',
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 200:
                result = data['result']
                return {
                    'success': True,
                    'easting': result.get('eastings'),
                    'northing': result.get('northings'),
                    'latitude': result.get('latitude'),
                    'longitude': result.get('longitude'),
                }
        return {'success': False}
    except:
        return {'success': False}


# Load DUKES data
df = pd.read_excel('data/generators/DUKES_5.11_2025.xlsx', sheet_name='5.11 Full list', skiprows=5)

# Filter to critical generators (nuclear + capacity >500 MW)
critical = df[(df['Primary Fuel'].str.contains('Nuclear', na=False, case=False)) | 
              (df['InstalledCapacity (MW)'] > 500)].copy()

print(f"Checking {len(critical)} critical generators (nuclear + >500 MW)")
print("="*100)

errors_found = []

for idx, row in critical.iterrows():
    if pd.isna(row['Postcode']) or pd.isna(row['X-Coordinate']):
        continue
    
    result = geocode_postcode(row['Postcode'])
    
    if result['success']:
        dukes_x, dukes_y = row['X-Coordinate'], row['Y-Coordinate']
        geocode_x, geocode_y = result['easting'], result['northing']
        
        # Calculate error
        error_m = ((dukes_x - geocode_x)**2 + (dukes_y - geocode_y)**2)**0.5
        
        if error_m > 10000:  # >10 km error
            errors_found.append({
                'site': row['Site Name'],
                'fuel': row['Primary Fuel'],
                'capacity_mw': row['InstalledCapacity (MW)'],
                'postcode': row['Postcode'],
                'error_km': error_m / 1000,
                'dukes_x': dukes_x,
                'dukes_y': dukes_y,
                'correct_x': geocode_x,
                'correct_y': geocode_y,
                'correct_lat': result['latitude'],
                'correct_lon': result['longitude']
            })
            
            print(f"❌ {row['Site Name']:40s} | Error: {error_m/1000:6.1f} km | {row['Primary Fuel']:20s} | {row['InstalledCapacity (MW)']:6.1f} MW")
        
    time.sleep(0.15)  # Rate limiting

print("\n" + "="*100)
print(f"\nFound {len(errors_found)} generators with coordinate errors >10 km")

if errors_found:
    print("\nCORRECT COORDINATES (from postcodes):")
    print("="*100)
    for error in sorted(errors_found, key=lambda x: -x['error_km']):
        print(f"\n{error['site']} ({error['fuel']}, {error['capacity_mw']:.0f} MW)")
        print(f"  Postcode: {error['postcode']}")
        print(f"  Error: {error['error_km']:.1f} km")
        print(f"  DUKES coordinates: X={error['dukes_x']:.0f}, Y={error['dukes_y']:.0f}")
        print(f"  CORRECT coordinates: X={error['correct_x']}, Y={error['correct_y']} (OSGB36)")
        print(f"                      Lat={error['correct_lat']:.6f}, Lon={error['correct_lon']:.6f} (WGS84)")

    # Save corrections
    corrections_df = pd.DataFrame(errors_found)
    corrections_df.to_csv('resources/validation/dukes_critical_coordinate_errors.csv', index=False)
    print(f"\n✅ Saved corrections to: resources/validation/dukes_critical_coordinate_errors.csv")

