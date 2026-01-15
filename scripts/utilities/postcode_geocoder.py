"""
UK Postcode Geocoding Utility with Caching

This module provides postcode geocoding for UK postcodes using the postcodes.io API,
with intelligent caching to minimize API calls.

Features:
- Cache postcodes to CSV (data/generators/postcode_cache.csv)
- Automatic cache updates for new postcodes
- Batch geocoding with rate limiting
- OSGB36 (Easting/Northing) and WGS84 (Lat/Lon) outputs
"""

import pandas as pd
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PostcodeGeocoder:
    """
    Geocode UK postcodes with persistent caching.
    """
    
    def __init__(self, cache_file: str = "data/generators/postcode_cache.csv"):
        """
        Initialize geocoder with cache file.
        
        Args:
            cache_file: Path to CSV file for caching postcode lookups
        """
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, Dict] = {}
        self.new_entries = 0
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from CSV file."""
        if self.cache_file.exists():
            try:
                cache_df = pd.read_csv(self.cache_file)
                for _, row in cache_df.iterrows():
                    postcode = row['postcode_clean'].upper()
                    self.cache[postcode] = {
                        'easting': row['easting'],
                        'northing': row['northing'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'postcode_clean': row['postcode_clean']
                    }
                logger.info(f"Loaded {len(self.cache)} postcodes from cache: {self.cache_file}")
            except Exception as e:
                logger.warning(f"Could not load cache file {self.cache_file}: {e}")
                logger.info("Starting with empty cache")
        else:
            logger.info(f"No existing cache found at {self.cache_file}. Starting fresh.")
    
    def _save_cache(self):
        """Save cache to CSV file."""
        if not self.cache:
            return
        
        # Create directory if needed
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert cache to DataFrame
        cache_list = []
        for postcode, data in self.cache.items():
            cache_list.append({
                'postcode_clean': data['postcode_clean'],
                'easting': data['easting'],
                'northing': data['northing'],
                'latitude': data['latitude'],
                'longitude': data['longitude']
            })
        
        cache_df = pd.DataFrame(cache_list)
        cache_df = cache_df.sort_values('postcode_clean')
        cache_df.to_csv(self.cache_file, index=False)
        
        logger.info(f"Saved {len(self.cache)} postcodes to cache: {self.cache_file}")
        if self.new_entries > 0:
            logger.info(f"  Added {self.new_entries} new postcodes to cache")
    
    @staticmethod
    def _clean_postcode(postcode: str) -> str:
        """Clean and normalize postcode."""
        if pd.isna(postcode) or not postcode:
            return ""
        return str(postcode).replace(' ', '').strip().upper()
    
    def _geocode_api(self, postcode: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Geocode a single postcode using postcodes.io API.
        
        Args:
            postcode: Cleaned UK postcode
            max_retries: Number of retry attempts
            
        Returns:
            Dictionary with geocoding results or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f'https://api.postcodes.io/postcodes/{postcode}',
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 200:
                        result = data['result']
                        return {
                            'easting': result.get('eastings'),
                            'northing': result.get('northings'),
                            'latitude': result.get('latitude'),
                            'longitude': result.get('longitude'),
                            'postcode_clean': result.get('postcode')
                        }
                elif response.status_code == 404:
                    logger.warning(f"Postcode not found: {postcode}")
                    return None
                
                # Rate limiting or server error - retry
                time.sleep(0.5 * (attempt + 1))
                
            except requests.RequestException as e:
                logger.warning(f"Geocoding attempt {attempt + 1} failed for {postcode}: {e}")
                time.sleep(0.5 * (attempt + 1))
        
        logger.error(f"Failed to geocode {postcode} after {max_retries} attempts")
        return None
    
    def geocode(self, postcode: str) -> Optional[Dict]:
        """
        Geocode a single postcode (using cache if available).
        
        Args:
            postcode: UK postcode (spaces optional)
            
        Returns:
            Dictionary with easting, northing, latitude, longitude or None if failed
        """
        clean = self._clean_postcode(postcode)
        
        if not clean:
            return None
        
        # Check cache first
        if clean in self.cache:
            return self.cache[clean].copy()
        
        # Call API
        result = self._geocode_api(clean)
        
        if result:
            # Add to cache
            self.cache[clean] = result
            self.new_entries += 1
        
        return result
    
    def geocode_batch(self, postcodes: List[str], delay: float = 0.1, 
                     save_every: int = 50) -> pd.DataFrame:
        """
        Geocode multiple postcodes with progress logging.
        
        Args:
            postcodes: List of UK postcodes
            delay: Delay between API calls (seconds)
            save_every: Save cache every N new postcodes
            
        Returns:
            DataFrame with original postcodes and geocoded coordinates
        """
        results = []
        api_calls = 0
        
        logger.info(f"Geocoding {len(postcodes)} postcodes...")
        
        for i, postcode in enumerate(postcodes):
            clean = self._clean_postcode(postcode)
            
            # Check if we need to call API
            needs_api_call = clean and (clean not in self.cache)
            
            result = self.geocode(postcode)
            
            results.append({
                'postcode_original': postcode,
                'postcode_clean': clean if clean else None,
                'easting': result['easting'] if result else None,
                'northing': result['northing'] if result else None,
                'latitude': result['latitude'] if result else None,
                'longitude': result['longitude'] if result else None,
                'geocoded': result is not None
            })
            
            if needs_api_call:
                api_calls += 1
                time.sleep(delay)  # Rate limiting
                
                # Periodic cache save
                if api_calls % save_every == 0:
                    self._save_cache()
                    logger.info(f"  Progress: {i+1}/{len(postcodes)} processed, {api_calls} API calls")
        
        # Final cache save
        if self.new_entries > 0:
            self._save_cache()
        
        logger.info(f"Geocoding complete: {api_calls} new API calls, {len(self.cache)} total in cache")
        
        return pd.DataFrame(results)
    
    def get_osgb36_coordinates(self, postcode: str) -> Optional[Tuple[float, float]]:
        """
        Get OSGB36 (Easting, Northing) coordinates for a postcode.
        
        Args:
            postcode: UK postcode
            
        Returns:
            Tuple of (easting, northing) or None if geocoding failed
        """
        result = self.geocode(postcode)
        if result and result['easting'] and result['northing']:
            return (result['easting'], result['northing'])
        return None
    
    def get_wgs84_coordinates(self, postcode: str) -> Optional[Tuple[float, float]]:
        """
        Get WGS84 (Latitude, Longitude) coordinates for a postcode.
        
        Args:
            postcode: UK postcode
            
        Returns:
            Tuple of (latitude, longitude) or None if geocoding failed
        """
        result = self.geocode(postcode)
        if result and result['latitude'] and result['longitude']:
            return (result['latitude'], result['longitude'])
        return None


def geocode_postcodes_from_dataframe(
    df: pd.DataFrame,
    postcode_column: str = 'Postcode',
    cache_file: str = "data/generators/postcode_cache.csv"
) -> pd.DataFrame:
    """
    Geocode postcodes from a DataFrame column.
    
    Args:
        df: DataFrame containing postcodes
        postcode_column: Name of column with postcodes
        cache_file: Path to cache file
        
    Returns:
        DataFrame with added columns: postcode_easting, postcode_northing, 
                                     postcode_lat, postcode_lon
    """
    geocoder = PostcodeGeocoder(cache_file=cache_file)
    
    # Get unique postcodes
    unique_postcodes = df[postcode_column].dropna().unique()
    
    # Geocode batch
    geocoded = geocoder.geocode_batch(unique_postcodes.tolist())
    
    # Create lookup dictionary
    lookup = {}
    for _, row in geocoded.iterrows():
        if row['geocoded']:
            lookup[row['postcode_original']] = {
                'easting': row['easting'],
                'northing': row['northing'],
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
    
    # Add coordinates to original DataFrame
    df_copy = df.copy()
    df_copy['postcode_easting'] = df_copy[postcode_column].map(
        lambda x: lookup.get(x, {}).get('easting')
    )
    df_copy['postcode_northing'] = df_copy[postcode_column].map(
        lambda x: lookup.get(x, {}).get('northing')
    )
    df_copy['postcode_lat'] = df_copy[postcode_column].map(
        lambda x: lookup.get(x, {}).get('latitude')
    )
    df_copy['postcode_lon'] = df_copy[postcode_column].map(
        lambda x: lookup.get(x, {}).get('longitude')
    )
    
    success_rate = (df_copy['postcode_easting'].notna().sum() / len(df_copy)) * 100
    logger.info(f"Geocoded {df_copy['postcode_easting'].notna().sum()}/{len(df_copy)} "
                f"rows ({success_rate:.1f}% success rate)")
    
    return df_copy


if __name__ == "__main__":
    # Example usage
    from logging_config import setup_logging
    
    setup_logging("postcode_geocoder", log_level="INFO")
    
    # Test with sample postcodes
    geocoder = PostcodeGeocoder()
    
    test_postcodes = [
        "EH42 1QS",  # Torness nuclear
        "BS10 7SP",  # Seabank gas
        "SW1A 1AA",  # 10 Downing Street
        "M1 4AE",    # Manchester
    ]
    
    print("\nTest Geocoding:")
    print("=" * 80)
    for postcode in test_postcodes:
        result = geocoder.geocode(postcode)
        if result:
            print(f"{postcode:15s} → E: {result['easting']:6d}, N: {result['northing']:6d} | "
                  f"Lat: {result['latitude']:8.5f}, Lon: {result['longitude']:8.5f}")
        else:
            print(f"{postcode:15s} → FAILED")
    
    print(f"\nCache statistics:")
    print(f"  Total cached: {len(geocoder.cache)}")
    print(f"  New entries: {geocoder.new_entries}")

