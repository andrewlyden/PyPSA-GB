"""
Enhanced TEC Location Mapping with Geocoding Backup for PyPSA-GB

This script enhances TEC generator location mapping by:
1. Using existing enhanced location mapping from process_tec_generators
2. Adding geocoding backup for sites without coordinates
3. Improving location matching accuracy
4. Providing comprehensive location reporting

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import logging
import warnings
import time
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic

# Configure logging
logger = None
try:
    from logging_config import setup_logging, log_dataframe_info
    logger = setup_logging("enhance_tec_locations_with_geocoding")
except Exception:
    try:
        from scripts.utilities.logging_config import setup_logging, log_dataframe_info
        logger = setup_logging("enhance_tec_locations_with_geocoding")
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("enhance_tec_locations_with_geocoding")
        def log_dataframe_info(df, logger, name: str):
            logger.info(f"{name} shape: {df.shape}")

class AutomatedGeocoder:
    """Automated geocoding service for unmapped generator locations."""
    
    def __init__(self, user_agent="pypsa-gb-geocoder", timeout=10, delay=1.0):
        """
        Initialize geocoder with rate limiting.
        
        Parameters
        ----------
        user_agent : str
            User agent string for Nominatim API
        timeout : int
            Request timeout in seconds
        delay : float
            Delay between requests to respect rate limits
        """
        self.geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        self.delay = delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def geocode_site(self, site_name: str, country="United Kingdom") -> Optional[Tuple[float, float]]:
        """
        Geocode a site name to coordinates.
        
        Parameters
        ----------
        site_name : str
            Name of the site to geocode
        country : str
            Country to restrict search to
            
        Returns
        -------
        tuple or None
            (latitude, longitude) if successful, None otherwise
        """
        if not site_name or pd.isna(site_name):
            return None
            
        # Clean site name for geocoding
        cleaned_name = self._clean_site_name(site_name)
        
        # Try different query variations
        queries = [
            f"{cleaned_name}, {country}",
            f"{cleaned_name} power station, {country}",
            f"{cleaned_name} power plant, {country}",
            cleaned_name
        ]
        
        for query in queries:
            try:
                self._rate_limit()
                logger.debug(f"Geocoding query: {query}")
                location = self.geolocator.geocode(query, exactly_one=True)
                
                if location:
                    lat, lon = location.latitude, location.longitude
                    logger.info(f"Geocoded {site_name} -> ({lat:.6f}, {lon:.6f}) via '{query}'")
                    return lat, lon
                    
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Geocoding error for {site_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected geocoding error for {site_name}: {e}")
                continue
                
        logger.debug(f"Failed to geocode: {site_name}")
        return None
    
    def _clean_site_name(self, site_name: str) -> str:
        """Clean site name for better geocoding results."""
        if not site_name:
            return ""
            
        # Remove common suffixes that might confuse geocoding
        cleaned = str(site_name).strip()
        
        # Remove capacity indicators
        cleaned = re.sub(r'\d+\.?\d*\s*MW\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d+\.?\d*\s*KW\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\d+\.?\d*\s*GW\b', '', cleaned, flags=re.IGNORECASE)
        
        # Remove common technical suffixes
        suffixes_to_remove = [
            r'\b\d+kV\s*Substation\b',
            r'\bSubstation\b',
            r'\bGSP\b',
            r'\bPrimary\b',
            r'\bSecondary\b',
            r'\bTertiary\b',
            r'\b\d+kV\b',
            r'\b\d+KV\b',
            r'\bConnection\b',
            r'\bSite\b',
            r'\bFacility\b'
        ]
        
        for suffix in suffixes_to_remove:
            cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

def enhance_tec_locations_with_geocoding(
    processed_tec_file: str,
    output_enhanced_file: str,
    output_geocoding_report: str,
    max_geocoding_attempts: int = 50
) -> pd.DataFrame:
    """
    Enhance TEC locations by adding geocoding backup for unmapped sites.
    
    Parameters
    ----------
    processed_tec_file : str
        Path to processed TEC file with initial location mapping
    output_enhanced_file : str
        Path for enhanced TEC file with geocoding backup
    output_geocoding_report : str
        Path for geocoding report
    max_geocoding_attempts : int
        Maximum number of sites to attempt geocoding (rate limiting)
        
    Returns
    -------
    pd.DataFrame
        Enhanced TEC data with additional geocoded coordinates
    """
    logger.info("Starting TEC location enhancement with geocoding backup")
    
    # Load processed TEC data
    logger.info(f"Loading processed TEC data from {processed_tec_file}")
    tec_df = pd.read_csv(processed_tec_file)
    log_dataframe_info(tec_df, logger, "Processed TEC data")
    
    # Identify sites without coordinates
    missing_coords_mask = (
        tec_df['x_coord'].isna() | 
        tec_df['y_coord'].isna() | 
        (tec_df['location_source'] == 'not_found')
    )
    
    sites_without_coords = tec_df[missing_coords_mask].copy()
    sites_with_coords = tec_df[~missing_coords_mask].copy()
    
    logger.info(f"Sites with existing coordinates: {len(sites_with_coords)}")
    logger.info(f"Sites needing geocoding: {len(sites_without_coords)}")
    
    # Initialize geocoding results tracking
    geocoding_results = []
    
    if len(sites_without_coords) > 0:
        logger.info(f"Attempting geocoding for up to {min(max_geocoding_attempts, len(sites_without_coords))} sites")
        
        # Initialize geocoder
        geocoder = AutomatedGeocoder(
            user_agent="pypsa-gb-enhanced-mapping",
            timeout=15,
            delay=1.2  # Respectful rate limiting
        )
        
        # Process sites without coordinates
        sites_to_geocode = sites_without_coords.head(max_geocoding_attempts)
        
        for idx, row in sites_to_geocode.iterrows():
            site_name = row.get('site_name', '')
            
            logger.info(f"Geocoding site {idx}: {site_name}")
            
            # Try geocoding
            coords = geocoder.geocode_site(site_name)
            
            if coords:
                lat, lon = coords
                # Update the dataframe
                sites_without_coords.loc[idx, 'x_coord'] = lon  # x = longitude
                sites_without_coords.loc[idx, 'y_coord'] = lat  # y = latitude  
                sites_without_coords.loc[idx, 'location_source'] = 'geocoded_nominatim'
                
                geocoding_results.append({
                    'site_name': site_name,
                    'status': 'success',
                    'lat': lat,
                    'lon': lon,
                    'source': 'nominatim'
                })
            else:
                geocoding_results.append({
                    'site_name': site_name,
                    'status': 'failed',
                    'lat': np.nan,
                    'lon': np.nan,
                    'source': 'nominatim'
                })
    
    # Combine enhanced data
    enhanced_tec_df = pd.concat([sites_with_coords, sites_without_coords], ignore_index=True)
    
    # Create geocoding report
    geocoding_report_df = pd.DataFrame(geocoding_results)
    
    # Summary statistics
    total_sites = len(enhanced_tec_df)
    sites_with_coords_final = enhanced_tec_df[['x_coord', 'y_coord']].notna().all(axis=1).sum()
    sites_geocoded = len([r for r in geocoding_results if r['status'] == 'success'])
    
    logger.info(f"Enhanced location mapping completed:")
    logger.info(f"  Total sites: {total_sites}")
    logger.info(f"  Sites with coordinates: {sites_with_coords_final}/{total_sites} ({100*sites_with_coords_final/total_sites:.1f}%)")
    logger.info(f"  Sites geocoded this run: {sites_geocoded}")
    
    # Save enhanced TEC data
    Path(output_enhanced_file).parent.mkdir(parents=True, exist_ok=True)
    enhanced_tec_df.to_csv(output_enhanced_file, index=False)
    log_dataframe_info(enhanced_tec_df, logger, "Enhanced TEC data")
    logger.info(f"Saved enhanced TEC data to {output_enhanced_file}")
    
    # Save geocoding report
    if geocoding_results:
        Path(output_geocoding_report).parent.mkdir(parents=True, exist_ok=True)
        geocoding_report_df.to_csv(output_geocoding_report, index=False)
        logger.info(f"Saved geocoding report to {output_geocoding_report}")
    
    return enhanced_tec_df

def main(
    processed_tec_file: str,
    output_enhanced_file: str,
    output_geocoding_report: str,
    max_geocoding_attempts: int = 50
):
    """
    Main function for TEC location enhancement with geocoding.
    
    Parameters
    ----------
    processed_tec_file : str
        Input processed TEC file
    output_enhanced_file : str
        Output enhanced TEC file
    output_geocoding_report : str
        Output geocoding report
    max_geocoding_attempts : int
        Maximum geocoding attempts
    """
    return enhance_tec_locations_with_geocoding(
        processed_tec_file=processed_tec_file,
        output_enhanced_file=output_enhanced_file,
        output_geocoding_report=output_geocoding_report,
        max_geocoding_attempts=max_geocoding_attempts
    )

if __name__ == "__main__":
    import sys
    
    # Check if running from Snakemake
    if 'snakemake' in globals():
        # Snakemake execution
        processed_tec_file = snakemake.input.tec_processed
        output_enhanced_file = snakemake.output.tec_enhanced
        output_geocoding_report = snakemake.output.geocoding_report
        
        # Get max attempts from params if provided
        max_attempts = getattr(snakemake.params, 'max_geocoding_attempts', 50)
        
        main(
            processed_tec_file=processed_tec_file,
            output_enhanced_file=output_enhanced_file,
            output_geocoding_report=output_geocoding_report,
            max_geocoding_attempts=max_attempts
        )
    else:
        # Command line execution
        if len(sys.argv) < 4:
            print("Usage: python enhance_tec_locations_with_geocoding.py <processed_tec_file> <output_enhanced_file> <output_geocoding_report> [max_attempts]")
            sys.exit(1)
            
        processed_tec_file = sys.argv[1]
        output_enhanced_file = sys.argv[2]
        output_geocoding_report = sys.argv[3]
        max_attempts = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        
        main(
            processed_tec_file=processed_tec_file,
            output_enhanced_file=output_enhanced_file,
            output_geocoding_report=output_geocoding_report,
            max_geocoding_attempts=max_attempts
        )

