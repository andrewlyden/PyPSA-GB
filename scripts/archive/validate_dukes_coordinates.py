"""
Validate DUKES coordinates by comparing with postcode geocoding.

This script:
1. Loads DUKES data with postcodes and coordinates
2. Geocodes postcodes using postcodes.io API
3. Compares geocoded coordinates with DUKES X-Coordinate/Y-Coordinate
4. Flags generators with significant coordinate errors
5. Generates a correction file for import into DUKES processing
"""

import pandas as pd
import requests
import time
from pathlib import Path
import logging
from scripts.utilities.logging_config import setup_logging

logger = setup_logging("validate_dukes_coordinates", log_level="INFO", log_to_file=True)


def geocode_postcode(postcode: str, max_retries: int = 3) -> dict:
    """
    Geocode a UK postcode using postcodes.io API.
    
    Args:
        postcode: UK postcode (spaces will be removed)
        max_retries: Number of retry attempts for failed requests
        
    Returns:
        Dictionary with 'easting', 'northing', 'latitude', 'longitude', 'success'
    """
    if pd.isna(postcode) or not postcode:
        return {'success': False, 'error': 'No postcode provided'}
    
    # Clean postcode (remove spaces)
    clean_postcode = str(postcode).replace(' ', '').strip().upper()
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f'https://api.postcodes.io/postcodes/{clean_postcode}',
                timeout=5
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
                        'postcode_clean': result.get('postcode')
                    }
            elif response.status_code == 404:
                return {'success': False, 'error': 'Postcode not found'}
            
            # Rate limiting or server error - retry
            time.sleep(0.5 * (attempt + 1))
            
        except requests.RequestException as e:
            logger.warning(f"Geocoding attempt {attempt + 1} failed for {clean_postcode}: {e}")
            time.sleep(0.5 * (attempt + 1))
    
    return {'success': False, 'error': 'Max retries exceeded'}


def calculate_coordinate_error(x1, y1, x2, y2) -> float:
    """
    Calculate Euclidean distance between two coordinate pairs in meters.
    
    Args:
        x1, y1: First coordinate pair (OSGB36)
        x2, y2: Second coordinate pair (OSGB36)
        
    Returns:
        Distance in meters
    """
    if any(pd.isna([x1, y1, x2, y2])):
        return float('inf')
    
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def validate_dukes_coordinates(dukes_file: str, output_dir: str = "resources/validation"):
    """
    Validate all DUKES coordinates against postcode geocoding.
    
    Args:
        dukes_file: Path to DUKES Excel file
        output_dir: Directory for output files
    """
    logger.info("="*80)
    logger.info("DUKES Coordinate Validation")
    logger.info("="*80)
    
    # Load DUKES data
    logger.info(f"Loading DUKES data from: {dukes_file}")
    df = pd.read_excel(dukes_file, sheet_name='5.11 Full list', skiprows=5)
    logger.info(f"Loaded {len(df)} generators")
    
    # Filter to generators with both postcodes and coordinates
    df_with_data = df[df['Postcode'].notna() & df['X-Coordinate'].notna()].copy()
    logger.info(f"Generators with both postcode and coordinates: {len(df_with_data)}")
    
    # Geocode postcodes
    logger.info("Geocoding postcodes (this may take a few minutes)...")
    geocoded_results = []
    
    for idx, row in df_with_data.iterrows():
        postcode = row['Postcode']
        result = geocode_postcode(postcode)
        
        geocoded_results.append({
            'index': idx,
            'site_name': row['Site Name'],
            'postcode': postcode,
            'dukes_x': row['X-Coordinate'],
            'dukes_y': row['Y-Coordinate'],
            'geocode_success': result.get('success', False),
            'geocode_x': result.get('easting'),
            'geocode_y': result.get('northing'),
            'geocode_lat': result.get('latitude'),
            'geocode_lon': result.get('longitude'),
            'primary_fuel': row.get('Primary Fuel'),
            'capacity_mw': row.get('InstalledCapacity (MW)'),
            'region': row.get('Region'),
            'error_message': result.get('error')
        })
        
        # Rate limiting - be nice to the free API
        if len(geocoded_results) % 100 == 0:
            logger.info(f"  Geocoded {len(geocoded_results)}/{len(df_with_data)}...")
            time.sleep(1)
        else:
            time.sleep(0.1)
    
    # Create results dataframe
    results_df = pd.DataFrame(geocoded_results)
    
    # Calculate coordinate errors
    results_df['error_meters'] = results_df.apply(
        lambda row: calculate_coordinate_error(
            row['dukes_x'], row['dukes_y'],
            row['geocode_x'], row['geocode_y']
        ) if row['geocode_success'] else float('nan'),
        axis=1
    )
    
    # Analyze results
    logger.info("\n" + "="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)
    
    successful_geocodes = results_df[results_df['geocode_success']].copy()
    logger.info(f"Successfully geocoded: {len(successful_geocodes)}/{len(results_df)}")
    
    if len(successful_geocodes) > 0:
        # Define error thresholds
        threshold_minor = 1000  # 1 km
        threshold_major = 10000  # 10 km
        
        minor_errors = successful_geocodes[
            (successful_geocodes['error_meters'] > threshold_minor) &
            (successful_geocodes['error_meters'] <= threshold_major)
        ]
        major_errors = successful_geocodes[successful_geocodes['error_meters'] > threshold_major]
        
        logger.info(f"\nCoordinate Error Summary:")
        logger.info(f"  Mean error: {successful_geocodes['error_meters'].mean():.1f} m")
        logger.info(f"  Median error: {successful_geocodes['error_meters'].median():.1f} m")
        logger.info(f"  Max error: {successful_geocodes['error_meters'].max():.1f} m")
        logger.info(f"  Within 1 km: {(successful_geocodes['error_meters'] <= threshold_minor).sum()}")
        logger.info(f"  Errors 1-10 km: {len(minor_errors)}")
        logger.info(f"  Errors >10 km: {len(major_errors)}")
        
        # Report major errors
        if len(major_errors) > 0:
            logger.warning(f"\n⚠️  Found {len(major_errors)} generators with coordinate errors >10 km:")
            for _, row in major_errors.sort_values('error_meters', ascending=False).iterrows():
                logger.warning(
                    f"  {row['site_name']:40s} | "
                    f"Error: {row['error_meters']/1000:.1f} km | "
                    f"Fuel: {row['primary_fuel']:15s} | "
                    f"Capacity: {row['capacity_mw']:.1f} MW"
                )
        
        # Create correction file for major errors
        if len(major_errors) > 0:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            corrections_file = output_path / "dukes_coordinate_corrections.csv"
            corrections_df = major_errors[[
                'site_name', 'postcode', 
                'geocode_x', 'geocode_y', 
                'geocode_lat', 'geocode_lon',
                'dukes_x', 'dukes_y', 
                'error_meters',
                'primary_fuel', 'capacity_mw', 'region'
            ]].copy()
            
            corrections_df.columns = [
                'station_name', 'postcode',
                'correct_x_osgb36', 'correct_y_osgb36',
                'correct_lat', 'correct_lon',
                'wrong_x_osgb36', 'wrong_y_osgb36',
                'error_meters',
                'fuel_type', 'capacity_mw', 'region'
            ]
            
            corrections_df.to_csv(corrections_file, index=False)
            logger.info(f"\n✅ Coordinate corrections saved to: {corrections_file}")
    
    # Save full results
    full_results_file = Path(output_dir) / "dukes_coordinate_validation_full.csv"
    results_df.to_csv(full_results_file, index=False)
    logger.info(f"✅ Full validation results saved to: {full_results_file}")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    # Default DUKES file
    dukes_file = "data/generators/DUKES_5.11_2025.xlsx"
    
    if len(sys.argv) > 1:
        dukes_file = sys.argv[1]
    
    results = validate_dukes_coordinates(dukes_file)
    
    logger.info("\n" + "="*80)
    logger.info("Validation complete!")
    logger.info("="*80)

