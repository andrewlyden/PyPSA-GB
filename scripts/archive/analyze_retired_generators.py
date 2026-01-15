"""
Analyze retired generators missing from 5.11 Full list but present in DUKES 2020
"""
import pandas as pd
import numpy as np

# Load both sheets
df_2020 = pd.read_excel('data/generators/DUKES_5.11_2025.xlsx', sheet_name='DUKES 2020', skiprows=5)
df_full = pd.read_excel('data/generators/DUKES_5.11_2025.xlsx', sheet_name='5.11 Full list', skiprows=5)

print("=" * 80)
print("RETIRED GENERATORS ANALYSIS")
print("=" * 80)

# Clean capacity columns
capacity_col_2020 = 'Installed Capacity\n(MW)'
capacity_col_full = 'InstalledCapacity (MW)'

# Convert to numeric, handling any strings
df_2020[capacity_col_2020] = pd.to_numeric(df_2020[capacity_col_2020], errors='coerce')
df_full[capacity_col_full] = pd.to_numeric(df_full[capacity_col_full], errors='coerce')

print(f"\nDUKES 2020 Historical Sheet:")
print(f"  Generators: {len(df_2020)}")
print(f"  Total Capacity: {df_2020[capacity_col_2020].sum():.1f} MW")

# Filter 5.11 Full list to generators commissioned by 2020
df_full_2020 = df_full[df_full['Year Commissioned'] <= 2020].copy()
print(f"\n5.11 Full List (Year Commissioned <= 2020):")
print(f"  Generators: {len(df_full_2020)}")
print(f"  Total Capacity: {df_full_2020[capacity_col_full].sum():.1f} MW")

print(f"\n⚠️  MISSING (Retired between 2020 and 2025):")
print(f"  Generators: {len(df_2020) - len(df_full_2020)}")
print(f"  Capacity: {df_2020[capacity_col_2020].sum() - df_full_2020[capacity_col_full].sum():.1f} MW")

print("\n" + "=" * 80)
print("LARGE THERMAL STATIONS IN DUKES 2020 (>100 MW)")
print("=" * 80)

large = df_2020[df_2020[capacity_col_2020] > 100].copy()
large = large.sort_values(capacity_col_2020, ascending=False)
print(f"\nFound {len(large)} large stations (>100 MW):")
print(large[['Station Name', 'Fuel', capacity_col_2020, 'Location\nScotland, Wales, Northern Ireland or English region']].head(30).to_string(index=False))

# Check which large stations are in 5.11 Full list
print("\n" + "=" * 80)
print("CHECKING WHICH LARGE STATIONS ARE MISSING FROM 5.11 FULL LIST")
print("=" * 80)

# Try to match by station name
large_names = large['Station Name'].str.strip().str.lower()
full_names = df_full['Site Name'].str.strip().str.lower()

missing_large = []
for idx, row in large.iterrows():
    station = str(row['Station Name']).strip().lower()
    # Check if station name appears in 5.11 Full list
    match = full_names.str.contains(station, case=False, na=False, regex=False).any()
    if not match:
        missing_large.append({
            'Station': row['Station Name'],
            'Fuel': row['Fuel'],
            'Capacity_MW': row[capacity_col_2020],
            'Location': row['Location\nScotland, Wales, Northern Ireland or English region']
        })

if missing_large:
    print(f"\n🚨 {len(missing_large)} LARGE STATIONS (>100 MW) NOT FOUND IN 5.11 FULL LIST:")
    missing_df = pd.DataFrame(missing_large)
    print(missing_df.to_string(index=False))
    print(f"\nTotal missing capacity from large stations: {missing_df['Capacity_MW'].sum():.1f} MW")
else:
    print("\n✅ All large stations found in 5.11 Full list")

print("\n" + "=" * 80)
print("FUEL TYPE BREAKDOWN - DUKES 2020")
print("=" * 80)
fuel_summary = df_2020.groupby('Fuel').agg({
    capacity_col_2020: ['count', 'sum']
}).round(1)
fuel_summary.columns = ['Count', 'Capacity_MW']
fuel_summary = fuel_summary.sort_values('Capacity_MW', ascending=False)
print(fuel_summary.to_string())

# Look for coal specifically
print("\n" + "=" * 80)
print("COAL GENERATORS IN DUKES 2020")
print("=" * 80)
coal_2020 = df_2020[df_2020['Fuel'].str.contains('Coal', case=False, na=False)]
print(f"Found {len(coal_2020)} coal generators, {coal_2020[capacity_col_2020].sum():.1f} MW")
if len(coal_2020) > 0:
    print("\nCoal stations:")
    print(coal_2020[['Station Name', 'Fuel', capacity_col_2020, 'Location\nScotland, Wales, Northern Ireland or English region']].to_string(index=False))

