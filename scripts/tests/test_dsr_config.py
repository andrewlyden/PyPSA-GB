"""
Test script to validate DSR (Demand Side Response) configuration settings.
Tests all event_response modes and event_window options.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.demand.event_flex import generate_event_schedule


def load_defaults():
    """Load default configuration."""
    with open("config/defaults.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_event_modes():
    """Test all three event frequency modes."""
    print("\n" + "=" * 80)
    print("TEST 1: EVENT FREQUENCY MODES (regular, winter, both)")
    print("=" * 80)
    
    # Create year-long index
    snapshots = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    
    configs = {
        "regular": {
            "mode": "regular",
            "event_window": ["17:00", "19:00"],
            "winter_months": [10, 11, 12, 1, 2, 3],
        },
        "winter": {
            "mode": "winter",
            "event_window": ["17:00", "19:00"],
            "winter_months": [10, 11, 12, 1, 2, 3],
        },
        "both": {
            "mode": "both",
            "event_window": ["17:00", "19:00"],
            "winter_months": [10, 11, 12, 1, 2, 3],
        },
    }
    
    window_start = 17
    window_end = 19
    
    for mode_name, config in configs.items():
        schedule = generate_event_schedule(
            snapshots=snapshots,
            mode=config["mode"],
            winter_months=config["winter_months"],
            event_window_start=window_start,
            event_window_end=window_end,
        )
        
        # Statistics
        event_hours = schedule.sum()
        event_days = (schedule.groupby(schedule.index.date).sum() > 0).sum()
        
        # Split by season
        summer_mask = ~schedule.index.month.isin([10, 11, 12, 1, 2, 3])
        winter_mask = schedule.index.month.isin([10, 11, 12, 1, 2, 3])
        
        summer_events = (schedule[summer_mask].groupby(schedule[summer_mask].index.date).sum() > 0).sum()
        winter_events = (schedule[winter_mask].groupby(schedule[winter_mask].index.date).sum() > 0).sum()
        
        summer_weeks = len(set(schedule[summer_mask].index.isocalendar().week))
        winter_weeks = len(set(schedule[winter_mask].index.isocalendar().week))
        
        print(f"\n📊 MODE: '{mode_name.upper()}'")
        print(f"   Total event hours: {event_hours:.0f}")
        print(f"   Total event days: {event_days:.0f}")
        print(f"   Summer events (Apr-Sep): {summer_events} days in {summer_weeks} weeks")
        if summer_weeks > 0:
            print(f"      → {summer_events/summer_weeks:.1f} events/week")
        print(f"   Winter events (Oct-Mar): {winter_events} days in {winter_weeks} weeks")
        if winter_weeks > 0:
            print(f"      → {winter_events/winter_weeks:.1f} events/week")
        
        # Validate expectations
        if mode_name == "regular":
            expected_summer = 2
            expected_winter = 2
        elif mode_name == "winter":
            expected_summer = 0
            expected_winter = 5
        elif mode_name == "both":
            expected_summer = 2
            expected_winter = 7
        
        if summer_weeks > 0 and mode_name != "winter":
            actual_summer = summer_events / summer_weeks
            status_summer = "✓" if abs(actual_summer - expected_summer) < 0.5 else "✗"
            print(f"   {status_summer} Expected {expected_summer}/week in summer: {actual_summer:.1f}/week")
        
        if winter_weeks > 0:
            actual_winter = winter_events / winter_weeks
            status_winter = "✓" if abs(actual_winter - expected_winter) < 0.5 else "✗"
            print(f"   {status_winter} Expected {expected_winter}/week in winter: {actual_winter:.1f}/week")


def test_event_windows():
    """Test all three event time windows."""
    print("\n" + "=" * 80)
    print("TEST 2: EVENT TIME WINDOWS (17:00-19:00, 16:00-20:00, 07:00-22:00)")
    print("=" * 80)
    
    # Create one week of data
    snapshots = pd.date_range("2025-01-06", "2025-01-12 23:00", freq="h")  # Week starts Monday
    
    windows = {
        "evening_peak": (["17:00", "19:00"], 17, 19),
        "extended_evening": (["16:00", "20:00"], 16, 20),
        "full_day": (["07:00", "22:00"], 7, 22),
    }
    
    config = {
        "mode": "regular",
        "winter_months": [10, 11, 12, 1, 2, 3],
    }
    
    for window_name, (window_list, start_hour, end_hour) in windows.items():
        schedule = generate_event_schedule(
            snapshots=snapshots,
            mode=config["mode"],
            winter_months=config["winter_months"],
            event_window_start=start_hour,
            event_window_end=end_hour,
        )
        
        # Check which hours have events
        event_hours = set()
        for idx in snapshots[schedule > 0]:
            event_hours.add(idx.hour)
        
        event_hours_sorted = sorted(event_hours)
        window_size = end_hour - start_hour
        
        print(f"\n📊 WINDOW: '{window_name.upper()}' ({window_list})")
        print(f"   Duration: {window_size} hours")
        print(f"   Hours with events: {event_hours_sorted if event_hours_sorted else 'None'}")
        
        # Validate
        expected_hours = set(range(start_hour, end_hour))
        if event_hours and event_hours.issubset(expected_hours):
            print(f"   ✓ Events only occur within specified window")
        elif not event_hours:
            print(f"   ⚠ No events scheduled this week (randomness in scheduling)")
        else:
            print(f"   ✗ Events occur outside specified window!")


def test_load_from_config():
    """Test loading DSR config from defaults.yaml."""
    print("\n" + "=" * 80)
    print("TEST 3: LOAD DSR CONFIG FROM defaults.yaml")
    print("=" * 80)
    
    defaults = load_defaults()
    dsr_config = defaults.get("demand_flexibility", {}).get("event_response", {})
    
    print(f"\n📋 Current defaults.yaml DSR configuration:")
    print(f"   enabled: {dsr_config.get('enabled')}")
    print(f"   mode: {dsr_config.get('mode')}")
    print(f"   event_window: {dsr_config.get('event_window')}")
    print(f"   dsr_capacity_mw: {dsr_config.get('dsr_capacity_mw')}")
    print(f"   participation_rate: {dsr_config.get('participation_rate')}")
    print(f"   max_reduction_fraction: {dsr_config.get('max_reduction_fraction')}")
    print(f"   marginal_cost: {dsr_config.get('marginal_cost')}")
    print(f"   winter_months: {dsr_config.get('winter_months')}")
    
    # Validate all required fields present
    required_fields = [
        "enabled", "mode", "event_window", "dsr_capacity_mw",
        "participation_rate", "max_reduction_fraction", "marginal_cost",
        "winter_months"
    ]
    
    missing = [f for f in required_fields if f not in dsr_config]
    if missing:
        print(f"\n✗ Missing fields: {missing}")
    else:
        print(f"\n✓ All required fields present")
    
    # Validate mode
    if dsr_config.get("mode") in ["regular", "winter", "both"]:
        print(f"✓ Mode '{dsr_config.get('mode')}' is valid")
    else:
        print(f"✗ Mode '{dsr_config.get('mode')}' is invalid")
    
    # Validate event window
    window = dsr_config.get("event_window")
    if isinstance(window, list) and len(window) == 2:
        try:
            start = int(window[0].split(":")[0])
            end = int(window[1].split(":")[0])
            if 0 <= start < 24 and 0 <= end < 24 and start < end:
                print(f"✓ Event window {window} is valid ({end-start}h duration)")
            else:
                print(f"✗ Event window {window} has invalid hour range")
        except:
            print(f"✗ Event window {window} format invalid")
    else:
        print(f"✗ Event window must be [HH:MM, HH:MM] list")
    
    # Validate marginal cost
    cost = dsr_config.get("marginal_cost")
    if isinstance(cost, (int, float)) and cost > 0:
        print(f"✓ Marginal cost £{cost}/MWh is valid (positive)")
    else:
        print(f"✗ Marginal cost must be positive number")
    
    # Validate rates
    part_rate = dsr_config.get("participation_rate")
    max_red = dsr_config.get("max_reduction_fraction")
    if 0 <= part_rate <= 1 and 0 <= max_red <= 1:
        max_potential = part_rate * max_red * 100
        print(f"✓ Participation rate {part_rate*100:.0f}% × Max reduction {max_red*100:.0f}% = {max_potential:.0f}% capacity potential")
    else:
        print(f"✗ Rates must be between 0 and 1")


def test_realistic_combinations():
    """Test recommended realistic scenario combinations."""
    print("\n" + "=" * 80)
    print("TEST 4: REALISTIC SCENARIO COMBINATIONS")
    print("=" * 80)
    
    scenarios = {
        "Baseline (tight)": {
            "mode": "regular",
            "event_window": ["17:00", "19:00"],
            "desc": "Conservative: baseline flexibility, peak-only"
        },
        "Extended (realistic)": {
            "mode": "both",
            "event_window": ["16:00", "20:00"],
            "desc": "Most realistic: year-round + winter peaks, extended window"
        },
        "Flexible (optimistic)": {
            "mode": "both",
            "event_window": ["07:00", "22:00"],
            "desc": "Optimistic: full-day flexibility with winter peaks"
        },
    }
    
    snapshots = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    winter_months = [10, 11, 12, 1, 2, 3]
    
    for scenario_name, scenario_config in scenarios.items():
        window = scenario_config["event_window"]
        start = int(window[0].split(":")[0])
        end = int(window[1].split(":")[0])
        
        schedule = generate_event_schedule(
            snapshots=snapshots,
            mode=scenario_config["mode"],
            winter_months=winter_months,
            event_window_start=start,
            event_window_end=end,
        )
        
        event_days = (schedule.groupby(schedule.index.date).sum() > 0).sum()
        winter_mask = schedule.index.month.isin(winter_months)
        winter_days = (schedule[winter_mask].groupby(schedule[winter_mask].index.date).sum() > 0).sum()
        summer_days = event_days - winter_days
        
        print(f"\n🎯 {scenario_name}")
        print(f"   {scenario_config['desc']}")
        print(f"   Mode: {scenario_config['mode']}, Window: {window}")
        print(f"   Total events: {event_days} days/year")
        print(f"   Summer (Apr-Sep): {summer_days} events")
        print(f"   Winter (Oct-Mar): {winter_days} events")


if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  DSR (Demand Side Response) CONFIGURATION TEST SUITE".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        test_load_from_config()
        test_event_modes()
        test_event_windows()
        test_realistic_combinations()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
