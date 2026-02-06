# EV Model Deep Dive Analysis

## Executive Summary

The EV model is **conceptually correct** - it tracks state of charge and coordinates charging vs driving. However, there's a **fundamental sizing mismatch** causing load shedding.

---

## Model Architecture (Correct)

```
Main Grid Bus (e.g., WALP_1)
      |
      | [Charger Link: 142 MW capacity, 80% availability]
      |
      V
EV Battery Bus (internal: "EV battery_WALP11 EV battery INT")
      |
      +--- Store (EV fleet battery): 1,219 MWh capacity
      +--- Load (EV driving demand): 310 MW peak
      +--- Load Shedding Generator (backup)
```

**How It Works:**
1. **Load (EV driving)**: Represents when EVs consume energy from their batteries while driving
2. **Store (EV battery)**: Buffers energy, tracks state of charge (SoC)
3. **Charger Link**: Imports power from main grid when EVs are plugged in (controlled by availability profile)
4. **PyPSA Optimizer**: Coordinates charging to minimize cost while maintaining SoC feasibility

**Energy Balance (verified):**
```
Store Energy: e(t) = e(t-1) - Store_p(t) × 1hr
Store_p > 0: Discharging to meet driving demand
Store_p < 0: Charging from grid (via Charger Link)
```

---

## The Root Cause: Profile Mismatch

### 1. Driving Demand Profile (EV consumption)
```
Hour    Demand (MW)    Description
00-10   0              No driving overnight/morning
11-14   1-406          Gradual increase
15-17   1,607-11,873   Ramp up
18-20   22,182-36,572  PEAK (evening commute)
21-23   32,275-11,873  Decline
```
**Peak/Mean Ratio: 5.5x** (system average: 40.4 GW peak, 7.3 GW mean)

### 2. Charger Availability Profile
```
Hour    Availability    Expected Behavior
00-07   86-95%         EVs at home, plugged in ✓
08-17   67-75%         Mix of home/work/driving
18-23   71-89%         **PROBLEM: High availability during peak driving**
```

### 3. The Contradiction

**At 20:00 (peak driving hour):**
- Driving demand: 40.4 GW (EVs consuming battery energy)
- Availability: 79.7% (EVs supposedly available to charge)
- **Physical reality**: EVs can't drive AND charge simultaneously!

**What's Happening:**
The availability profile suggests **79.7% of EVs are plugged in** at 20:00, but the driving demand shows **peak consumption** at the same time. This is contradictory.

---

## Energy Deficit Analysis

### System-Wide Balance (Peak Hour)
```
Supply:
  Charging from grid: 9.7 GW
  Store discharge: 22.5 GW
  Load shedding: 8.8 GW
  Total: 41.0 GW

Demand:
  EV driving: 40.4 GW
```

### Weekly Energy Balance (Sample Bus)
```
Total EV driving demand: 14,220 MWh
Total charging from grid: 11,818 MWh
Deficit: 2,402 MWh (covered by load shedding)

Charger capacity: 142 MW × 80% avg availability × 168 hrs = 19,112 MWh (theoretical max)
But actual utilization: only 11,818 MWh (62%)
```

**Why isn't the charger at full capacity?**
- The Store constraint! The battery can't accept unlimited charge
- Peak driving depletes the battery to e_min
- Recharging takes time but availability drops during the day
- Result: Insufficient time to recharge before next peak

---

## Root Causes Identified

### 1. **Profile Generation Issue**
The driving demand profile is generated from a synthetic Gaussian:
```python
# In scripts/demand/electric_vehicles.py ~line 1530
evening_peak = np.exp(-((hour_of_day - 20) ** 2) / (2 * 2 ** 2))  # Peak at 20:00
```
**σ = 2 hours → 95% of demand concentrated in 4-hour window (18:00-22:00)**

This creates an unrealistic profile:
- Real UK driving: Morning commute (07:00-09:00) + evening commute (17:00-19:00) + throughout day
- Model: 95% of driving in 4-hour evening window

### 2. **Availability Profile Doesn't Match Driving**
The availability profile should be the **inverse** of driving:
- When driving = high → availability = low (not plugged in)
- When driving = low → availability = high (plugged in at home/work)

Currently: availability = 70-95% almost all day, including peak driving hours

### 3. **Charger Sizing Assumption**
Charger capacity is sized at **7 kW per vehicle**:
```python
charger_power_mw = n_vehicles * 7 / 1000
```

For 2 million flexible EVs → 14 GW total capacity

But with peak demand of 40 GW and only 4 hours to deliver the weekly energy:
- Required: ~40 GW charger capacity (to charge during peak)
- OR: Spread driving demand across 24 hours (realistic usage)
- Current: 14 GW capacity with 4-hour peak → **mismatch**

---

## Proposed Fixes

### Option 1: Fix the Driving Profile (RECOMMENDED)
**Change the synthetic profile to be realistic:**

```python
# Multi-modal: morning commute + evening commute + throughout day
hour_of_day = (timesteps % steps_per_day) * timestep_hours

# Morning commute peak (08:00)
morning_peak = 0.3 * np.exp(-((hour_of_day - 8) ** 2) / (2 * 1.5 ** 2))

# Evening commute peak (18:00)
evening_peak = 0.3 * np.exp(-((hour_of_day - 18) ** 2) / (2 * 1.5 ** 2))

# Throughout day (10:00-16:00)
day_usage = 0.4 * np.where((hour_of_day >= 10) & (hour_of_day <= 16), 1.0, 0.0)

# Combined
synthetic_profile = morning_peak + evening_peak + day_usage
```

**Expected result:**
- Peak/mean ratio: ~2-3x (instead of 5.5x)
- Peak demand: ~15-20 GW (instead of 40 GW)
- Charger capacity sufficient to recharge between peaks

### Option 2: Fix the Availability Profile
**Make availability the inverse of driving:**

```python
# Availability = 1 - (driving_profile_normalized * 0.5)
# When driving is 100% of peak → availability = 50%
# When driving is 0% → availability = 100%
```

**Issue**: This doesn't fix the fundamental problem - the driving profile is still too peaky

### Option 3: Increase Charger Capacity (WORKAROUND)
**Scale charger capacity to handle peak:**

```python
# Instead of 7 kW per vehicle
# Use 20 kW per vehicle (fast chargers)
charger_power_kw = 20  # Up from 7
```

**Issue**: This is unrealistic - not all EVs have fast chargers at home

### Option 4: Use Historical Data (BEST if available)
**Replace synthetic profile with real UK EV charging/driving data**

Sources:
- National Grid ESO EV profiles
- EVHS (Electric Vehicle Homecharge Scheme) data
- Academic studies (e.g., "EV charging patterns" research)

**Benefits:**
- Realistic peak/mean ratio
- Proper morning/evening commute patterns
- Coordinated availability profiles

---

## Recommended Implementation

### Immediate Fix (Phase 1)
1. **Update synthetic driving profile** to have two peaks (morning + evening) plus daytime usage
2. **Verify charger capacity** is adequate for the new profile
3. **Test with 1 week solve** to ensure no load shedding

### Longer-term (Phase 2)
1. **Acquire real UK EV driving/charging data**
2. **Implement data loading** in `electric_vehicles.py`
3. **Validate against FES building blocks**

---

## Code Changes Required

### File: `scripts/demand/electric_vehicles.py`

**Line ~1530 (synthetic profile generation):**

```python
# BEFORE
evening_peak = np.exp(-((hour_of_day - 20) ** 2) / (2 * 2 ** 2))

# AFTER
morning_peak = 0.3 * np.exp(-((hour_of_day - 8) ** 2) / (2 * 1.5 ** 2))
evening_peak = 0.3 * np.exp(-((hour_of_day - 18) ** 2) / (2 * 1.5 ** 2))
midday_usage = 0.4 * np.where((hour_of_day >= 10) & (hour_of_day <= 16), 
                               (1 - np.abs(hour_of_day - 13) / 3) * 0.5, 0)
synthetic_profile = morning_peak + evening_peak + midday_usage
```

**Explanation:**
- Morning peak at 08:00 (30% of daily driving)
- Evening peak at 18:00 (30% of daily driving)
- Midday usage 10:00-16:00 (40% of daily driving, lower intensity)
- Total peak/mean ratio: ~2.5x (realistic)

---

## Testing Plan

1. **Regenerate EV profile** with new synthetic formula
2. **Re-run disaggregation** for HT35_flex scenario
3. **Check profile statistics**:
   - Peak/mean ratio should be 2-3x
   - Peak should be ~20 GW (not 40 GW)
4. **Re-run flexibility integration**
5. **Solve network** and verify:
   - Load shedding < 1 TWh/year
   - Store SoC stays within bounds
   - Charger utilization is reasonable

---

## Key Learnings

1. **The model IS tracking SoC correctly** - PyPSA's Store component works as expected
2. **The issue is input data** - the driving profile is unrealistic
3. **Availability ≠ Driving** - these should be complementary, not simultaneous
4. **Charger sizing matters** - 7 kW works for realistic profiles, not for extreme peaks
5. **Synthetic profiles need validation** - always compare to real-world data

---

*Analysis Date: 2026-02-05*
*Scenario: HT35_flex (Holistic Transition 2035)*
*Network: ETYS Clustered (1,235 buses, 936 EV battery buses)*
