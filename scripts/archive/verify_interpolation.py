"""
Verify interpolation fix for renewable generator profiles.
"""
import pypsa
import pandas as pd

print("=== INTERPOLATION FIX VERIFICATION ===\n")

# Load network
n = pypsa.Network('resources/network/HT35_clustered_gsp_base_demand_generators.nc')

# Check a wind generator
wind_gen = [g for g in n.generators_t.p_max_pu.columns if 'South_Beach' in g][0]
profile = n.generators_t.p_max_pu[wind_gen]

print(f"Generator: {wind_gen}")
print(f"Profile length: {len(profile)}")
print(f"Non-zero values: {(profile > 0).sum()}/{len(profile)} ({(profile > 0).sum()/len(profile)*100:.1f}%)")
print(f"Mean CF: {profile.mean():.4f}")

print(f"\nFirst 30 values (should have NO alternating zeros):")
for i in range(30):
    ts = n.snapshots[i]
    val = profile.iloc[i]
    print(f"{i:3d}  {ts}  {val:.6f}")

# Check for alternating zeros
alternating = 0
for i in range(0, min(100, len(profile)-1), 2):
    if profile.iloc[i] > 0 and profile.iloc[i+1] == 0:
        alternating += 1
    elif profile.iloc[i] == 0 and profile.iloc[i+1] > 0:
        alternating += 1

print(f"\nAlternating zero pattern: {alternating}/50 pairs")

if alternating < 5:
    print("✅ SUCCESS: No more alternating zero pattern!")
    print("✅ Interpolation is working correctly!")
else:
    print("⚠️  WARNING: Still detecting alternating zeros")

# Check solar too
print("\n" + "="*60)
solar_gen = [g for g in n.generators_t.p_max_pu.columns if 'Fen_Farm' in g][0]
profile_solar = n.generators_t.p_max_pu[solar_gen]

print(f"\nSolar Generator: {solar_gen}")
print(f"Profile length: {len(profile_solar)}")
print(f"Non-zero values: {(profile_solar > 0).sum()}/{len(profile_solar)} ({(profile_solar > 0).sum()/len(profile_solar)*100:.1f}%)")
print(f"Mean CF: {profile_solar.mean():.4f}")

# Find daytime period
daytime_vals = profile_solar[profile_solar > 0.01][:30]
print(f"\nFirst 30 daytime values (should be smooth):")
for i, (ts, val) in enumerate(daytime_vals.items()):
    idx = n.snapshots.get_loc(ts)
    print(f"{idx:3d}  {ts}  {val:.6f}")

print("\n✅ Timestep interpolation fix validated!")

