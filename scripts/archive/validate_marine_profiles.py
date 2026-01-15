"""
Quick validation and visualization of marine renewable profiles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def validate_marine_profiles():
    """Load and validate marine renewable profiles."""
    
    profiles_dir = Path("resources/renewable/profiles")
    
    # Load marine profiles
    tidal_stream = pd.read_csv(profiles_dir / "tidal_stream_2020.csv", index_col=0, parse_dates=True)
    shoreline_wave = pd.read_csv(profiles_dir / "shoreline_wave_2020.csv", index_col=0, parse_dates=True)
    tidal_lagoon = pd.read_csv(profiles_dir / "tidal_lagoon_2020.csv", index_col=0, parse_dates=True)
    
    print("=== Marine Renewable Profiles Validation ===\n")
    
    # Tidal Stream
    print("🌊 TIDAL STREAM PROFILES:")
    print(f"  Sites: {len(tidal_stream.columns)}")
    print(f"  Time periods: {len(tidal_stream)}")
    if len(tidal_stream.columns) > 0:
        total_capacity = tidal_stream.sum(axis=1)
        print(f"  Total capacity range: {total_capacity.min():.2f} - {total_capacity.max():.2f} MW")
        print(f"  Average capacity factor: {(total_capacity.mean() / (2+6+2+0.5+1.4)):.2f}")  # Sum of site capacities
        print(f"  Sites: {list(tidal_stream.columns)}")
    
    print("\n🌊 SHORELINE WAVE PROFILES:")
    print(f"  Sites: {len(shoreline_wave.columns)}")
    print(f"  Time periods: {len(shoreline_wave)}")
    if len(shoreline_wave.columns) > 0:
        total_capacity = shoreline_wave.sum(axis=1)
        print(f"  Total capacity range: {total_capacity.min():.2f} - {total_capacity.max():.2f} MW")
        print(f"  Average capacity factor: {(total_capacity.mean() / (1+23)):.2f}")  # Sum of site capacities
        print(f"  Sites: {list(shoreline_wave.columns)}")
    
    print("\n🌊 TIDAL LAGOON PROFILES:")
    print(f"  Sites: {len(tidal_lagoon.columns)}")
    print(f"  Time periods: {len(tidal_lagoon)}")
    if len(tidal_lagoon.columns) == 0:
        print("  No tidal lagoon sites currently in dataset")
    
    # Show some sample patterns
    if len(tidal_stream.columns) > 0:
        print("\n📊 SAMPLE TIDAL PATTERNS (first 48 hours):")
        sample_site = tidal_stream.columns[0]
        sample_data = tidal_stream[sample_site].iloc[:48]
        
        # Find tidal cycles (approximate 12.4 hour periods)
        peaks = []
        for i in range(1, len(sample_data)-1):
            if sample_data.iloc[i] > sample_data.iloc[i-1] and sample_data.iloc[i] > sample_data.iloc[i+1]:
                peaks.append(i)
        
        if len(peaks) >= 2:
            avg_period = np.mean(np.diff(peaks))
            print(f"  Average tidal cycle: {avg_period:.1f} hours (expected ~12.4)")
        
        print(f"  Max generation: {sample_data.max():.2f} MW")
        print(f"  Min generation: {sample_data.min():.2f} MW")
    
    return tidal_stream, shoreline_wave, tidal_lagoon

def create_sample_plots():
    """Create sample plots of marine renewable patterns."""
    
    tidal_stream, shoreline_wave, tidal_lagoon = validate_marine_profiles()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot first week of tidal stream
    if len(tidal_stream.columns) > 0:
        week_data = tidal_stream.iloc[:168]  # First week (168 hours)
        
        axes[0].plot(week_data.index, week_data.sum(axis=1), 'b-', linewidth=2, label='Total Tidal Stream')
        axes[0].set_title('Tidal Stream Generation - First Week of 2020')
        axes[0].set_ylabel('Power (MW)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Plot first week of wave
    if len(shoreline_wave.columns) > 0:
        week_data = shoreline_wave.iloc[:168]  # First week
        
        axes[1].plot(week_data.index, week_data.sum(axis=1), 'g-', linewidth=2, label='Total Wave Power')
        axes[1].set_title('Shoreline Wave Generation - First Week of 2020')
        axes[1].set_ylabel('Power (MW)')
        axes[1].set_xlabel('Time')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('resources/renewable/marine_profiles_sample.png', dpi=150, bbox_inches='tight')
    print("\n📈 Sample plots saved to: resources/renewable/marine_profiles_sample.png")
    
    plt.show()

if __name__ == "__main__":
    validate_marine_profiles()
    create_sample_plots()

