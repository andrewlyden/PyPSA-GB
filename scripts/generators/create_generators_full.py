#!/usr/bin/env python3
"""
Create Full Generators Dataset
=============================

This script combines renewable and dispatchable generators into a unified
dataset for comprehensive mapping and analysis.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_renewable_sites():
    """Load all renewable site data."""
    renewable_files = [
        ("resources/renewable/wind_onshore_sites.csv", "wind_onshore"),
        ("resources/renewable/wind_offshore_sites.csv", "wind_offshore"), 
        ("resources/renewable/solar_pv_sites.csv", "solar_pv"),
        ("resources/renewable/geothermal_sites.csv", "geothermal"),
        ("resources/renewable/small_hydro_sites.csv", "small_hydro"),
        ("resources/renewable/large_hydro_sites.csv", "large_hydro"),
        ("resources/renewable/tidal_stream_sites.csv", "tidal_stream"),
        ("resources/renewable/shoreline_wave_sites.csv", "shoreline_wave"),
        ("resources/renewable/tidal_lagoon_sites.csv", "tidal_lagoon")
    ]
    
    renewable_dfs = []
    
    for file_path, technology in renewable_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Set technology based on filename
                df['technology'] = technology
                renewable_dfs.append(df)
                print(f"Loaded {file_path}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if renewable_dfs:
        combined_renewable = pd.concat(renewable_dfs, ignore_index=True)
        print(f"Combined renewable: {len(combined_renewable)} records")
        return combined_renewable
    else:
        return pd.DataFrame()

def load_dispatchable_generators():
    """Load dispatchable generators."""
    file_path = "resources/generators/dispatchable_generators_complete.csv"
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded dispatchable generators: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading dispatchable generators: {e}")
            return pd.DataFrame()
    else:
        print(f"Dispatchable generators file not found: {file_path}")
        return pd.DataFrame()

def standardize_columns(df, source_type):
    """Standardize column names and add source information."""
    
    # Standard column mapping
    column_mappings = {
        'capacity_mw': ['capacity_mw', 'Installed Capacity MW', 'capacity', 'Installed Capacity'],
        'technology': ['technology', 'Technology', 'tech_type', 'fuel_type'],
        'name': ['name', 'site_name', 'Site Name', 'Plant Name', 'generator_name'],
        'lat': ['lat', 'latitude', 'Latitude', 'y'],
        'lon': ['lon', 'longitude', 'Longitude', 'x'],
        'status': ['status', 'Status', 'development_status'],
        'commissioning_year': ['commissioning_year', 'Commissioned', 'commissioned', 'year_commissioned'],
        'bus': ['bus', 'nearest_bus', 'mapped_bus']
    }
    
    # Apply mappings
    for std_col, possible_cols in column_mappings.items():
        for col in possible_cols:
            if col in df.columns and std_col not in df.columns:
                df[std_col] = df[col]
                break
    
    # Add source type
    df['source_type'] = source_type
    
    # Ensure required columns exist
    required_cols = ['name', 'technology', 'capacity_mw', 'lat', 'lon', 'source_type']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    return df[required_cols + [c for c in df.columns if c not in required_cols]]

def main():
    """Main execution function."""
    print("Creating full generators dataset...")
    
    # Load data
    renewable_df = load_renewable_sites()
    dispatchable_df = load_dispatchable_generators()
    
    # Standardize
    if not renewable_df.empty:
        renewable_df = standardize_columns(renewable_df, 'renewable')
    
    if not dispatchable_df.empty:
        dispatchable_df = standardize_columns(dispatchable_df, 'dispatchable')
    
    # Combine
    if not renewable_df.empty and not dispatchable_df.empty:
        full_df = pd.concat([renewable_df, dispatchable_df], ignore_index=True)
    elif not renewable_df.empty:
        full_df = renewable_df
    elif not dispatchable_df.empty:
        full_df = dispatchable_df
    else:
        print("No generator data found!")
        return
    
    # Clean and deduplicate
    full_df['capacity_mw'] = pd.to_numeric(full_df['capacity_mw'], errors='coerce')
    full_df['lat'] = pd.to_numeric(full_df['lat'], errors='coerce')
    full_df['lon'] = pd.to_numeric(full_df['lon'], errors='coerce')
    
    # Save
    output_path = Path("resources/generators/generators_full.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_df.to_csv(output_path, index=False)
    
    print(f"Saved full generators dataset: {output_path}")
    print(f"Total records: {len(full_df)}")
    print(f"Renewable: {len(full_df[full_df['source_type'] == 'renewable'])}")
    print(f"Dispatchable: {len(full_df[full_df['source_type'] == 'dispatchable'])}")
    print(f"Total capacity: {full_df['capacity_mw'].sum():.0f} MW")

if __name__ == "__main__":
    main()

