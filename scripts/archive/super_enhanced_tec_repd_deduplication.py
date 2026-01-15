"""
Super Enhanced TEC-REPD Deduplication Analysis

This script performs even more sophisticated matching between TEC and REPD datasets
to find subtle duplicates that were missed in the initial analysis.

Enhanced matching techniques:
1. Location-based matching using coordinates
2. Capacity range matching (within tolerance)
3. Advanced name normalization with technology variants
4. Partial location name matching
5. Operator name matching
6. Multi-step fuzzy matching with different thresholds

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import re
import logging
from pathlib import Path
from geopy.distance import geodesic

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_name_advanced(name: str) -> str:
    """
    Advanced name normalization for better matching.
    
    Args:
        name: Site name to normalize
        
    Returns:
        Normalized name
    """
    if pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Remove common suffixes and technology indicators
    removals = [
        r'\s*-?\s*(battery\s*storage|battery|bess|energy\s*storage)\s*$',
        r'\s*-?\s*(power\s*station|power\s*plant|power)\s*$',
        r'\s*-?\s*(energy\s*park|energy\s*centre|energy\s*center)\s*$',
        r'\s*-?\s*(wind\s*farm|windfarm)\s*$',
        r'\s*-?\s*(solar\s*farm|solar\s*park)\s*$',
        r'\s*-?\s*(expansion|extension|phase\s*\d+|phase\s*[ivx]+)\s*$',
        r'\s*-?\s*(stage\s*\d+|stage\s*[ivx]+)\s*$',
        r'\s*-?\s*(unit\s*\d+|block\s*\d+)\s*$',
        r'\s*-?\s*(grid\s*scale|utility\s*scale)\s*$',
        r'\s*-?\s*(co-located|collocated)\s*$',
    ]
    
    for pattern in removals:
        name = re.sub(pattern, '', name)
    
    # Remove common prefixes
    prefix_removals = [
        r'^(the\s+)',
        r'^(new\s+)',
        r'^(great\s+britain\s+)',
        r'^(uk\s+)',
    ]
    
    for pattern in prefix_removals:
        name = re.sub(pattern, '', name)
    
    # Standardize common words
    replacements = {
        r'\bst\b': 'saint',
        r'\bmount\b': 'mt',
        r'\bmountain\b': 'mt',
        r'\bcentre\b': 'center',
        r'\benergy\s*park\b': 'energy park',
        r'\benergy\s*centre\b': 'energy center',
        r'\bsubstation\b': 'sub',
        r'\bsubst\b': 'sub',
        r'\bconnection\b': 'conn',
        r'\bterminal\b': 'term',
        r'\btertiary\b': 'tert',
        r'\bprimary\b': 'prim',
        r'\bsecondary\b': 'sec',
    }
    
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)
    
    # Remove extra whitespace and punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def extract_location_keywords(name: str) -> set:
    """Extract location-related keywords from a name."""
    if pd.isna(name):
        return set()
    
    name = str(name).lower()
    
    # Common location indicators
    location_patterns = [
        r'\b([a-z]+(?:ford|bury|ham|ton|bridge|field|hill|wood|gate|dale|vale|moor|land|port|mouth|head|well|wick|thorpe|by|leigh))\b',
        r'\b(north|south|east|west|upper|lower|great|little|new|old)\s+([a-z]+)\b',
        r'\b([a-z]{3,})\s+(park|farm|common|green|heath|marsh|downs|ridge|valley)\b',
        r'\b([a-z]+)\s+(power|energy|grid|sub|substation|terminal)\b',
    ]
    
    keywords = set()
    for pattern in location_patterns:
        matches = re.findall(pattern, name)
        for match in matches:
            if isinstance(match, tuple):
                keywords.update([word for word in match if len(word) > 2])
            else:
                if len(match) > 2:
                    keywords.add(match)
    
    # Also extract main words (3+ chars)
    words = re.findall(r'\b[a-z]{3,}\b', name)
    keywords.update(words[:3])  # Take first 3 main words
    
    return keywords

def calculate_distance_km(coord1, coord2):
    """Calculate distance between two coordinate pairs in km."""
    try:
        if any(pd.isna([coord1[0], coord1[1], coord2[0], coord2[1]])):
            return float('inf')
        return geodesic(coord1, coord2).kilometers
    except:
        return float('inf')

def enhanced_tec_repd_matching(
    tec_file: str = "data/generators/tec-register-02-september-2025.csv",
    repd_file: str = "data/renewables/repd-q2-july-2024.csv",
    generators_file: str = "resources/generators/dispatchable_generators_final_with_repd_duplicates.csv",
    output_file: str = "resources/generators/super_enhanced_tec_repd_duplicates.csv"
) -> pd.DataFrame:
    """
    Perform super enhanced matching between TEC and REPD datasets.
    
    Returns:
        DataFrame with potential matches
    """
    logger.info("Starting super enhanced TEC-REPD matching analysis")
    
    # Load datasets
    logger.info("Loading datasets...")
    tec_df = pd.read_csv(tec_file, encoding='utf-8-sig')
    repd_df = pd.read_csv(repd_file, encoding='utf-8-sig')
    generators_df = pd.read_csv(generators_file)
    
    # Filter operational REPD sites
    repd_operational = repd_df[repd_df['Development Status (short)'].isin(['Operational'])].copy()
    
    logger.info(f"Loaded {len(tec_df)} TEC generators, {len(repd_operational)} operational REPD sites")
    
    # Prepare TEC data with normalization
    tec_processed = tec_df.copy()
    tec_processed['normalized_name'] = tec_processed['Project Name'].apply(normalize_name_advanced)
    tec_processed['location_keywords'] = tec_processed['Project Name'].apply(extract_location_keywords)
    
    # Get TEC coordinates from generators database
    tec_coords = generators_df[generators_df['data_source'] == 'TEC'][['site_name', 'x_coord', 'y_coord']].copy()
    tec_coords_dict = dict(zip(tec_coords['site_name'], zip(tec_coords['x_coord'], tec_coords['y_coord'])))
    
    # Prepare REPD data with normalization
    repd_processed = repd_operational.copy()
    repd_processed['normalized_name'] = repd_processed['Site Name'].apply(normalize_name_advanced)
    repd_processed['location_keywords'] = repd_processed['Site Name'].apply(extract_location_keywords)
    
    # Convert REPD coordinates
    repd_processed['repd_x'] = pd.to_numeric(repd_processed['X-coordinate'], errors='coerce')
    repd_processed['repd_y'] = pd.to_numeric(repd_processed['Y-coordinate'], errors='coerce')
    
    matches = []
    
    logger.info("Performing enhanced matching...")
    
    for tec_idx, tec_row in tec_processed.iterrows():
        tec_name = tec_row['Project Name']
        tec_capacity = pd.to_numeric(tec_row['Cumulative Total Capacity (MW)'], errors='coerce')
        tec_normalized = tec_row['normalized_name']
        tec_keywords = tec_row['location_keywords']
        
        # Get TEC coordinates if available
        tec_coords_pair = tec_coords_dict.get(tec_name, (np.nan, np.nan))
        
        best_matches = []
        
        for repd_idx, repd_row in repd_processed.iterrows():
            repd_name = repd_row['Site Name']
            repd_capacity = pd.to_numeric(repd_row['Installed Capacity (MWelec)'], errors='coerce')
            repd_normalized = repd_row['normalized_name']
            repd_keywords = repd_row['location_keywords']
            repd_coords_pair = (repd_row['repd_x'], repd_row['repd_y'])
            
            # Skip if either name is too short after normalization
            if len(tec_normalized) < 3 or len(repd_normalized) < 3:
                continue
            
            match_scores = {}
            
            # 1. Exact normalized match
            if tec_normalized == repd_normalized:
                match_scores['exact_normalized'] = 100.0
            
            # 2. Fuzzy matching with different algorithms
            match_scores['fuzzy_ratio'] = fuzz.ratio(tec_normalized, repd_normalized)
            match_scores['fuzzy_partial'] = fuzz.partial_ratio(tec_normalized, repd_normalized)
            match_scores['fuzzy_token_sort'] = fuzz.token_sort_ratio(tec_normalized, repd_normalized)
            match_scores['fuzzy_token_set'] = fuzz.token_set_ratio(tec_normalized, repd_normalized)
            
            # 3. Location keyword matching
            if tec_keywords and repd_keywords:
                common_keywords = tec_keywords.intersection(repd_keywords)
                keyword_score = len(common_keywords) / max(len(tec_keywords), len(repd_keywords)) * 100
                match_scores['keyword_match'] = keyword_score
            else:
                match_scores['keyword_match'] = 0.0
            
            # 4. Capacity matching
            if pd.notna(tec_capacity) and pd.notna(repd_capacity):
                capacity_diff = abs(tec_capacity - repd_capacity)
                max_capacity = max(tec_capacity, repd_capacity)
                if max_capacity > 0:
                    capacity_similarity = max(0, (1 - capacity_diff / max_capacity)) * 100
                    match_scores['capacity_match'] = capacity_similarity
                else:
                    match_scores['capacity_match'] = 0.0
            else:
                match_scores['capacity_match'] = 0.0
            
            # 5. Distance-based matching (if coordinates available)
            distance_km = calculate_distance_km(tec_coords_pair, repd_coords_pair)
            if distance_km != float('inf'):
                # Score inversely related to distance (max 100 for same location, 0 for >10km)
                distance_score = max(0, min(100, (10 - distance_km) / 10 * 100))
                match_scores['distance_match'] = distance_score
            else:
                match_scores['distance_match'] = 0.0
            
            # 6. Operator matching
            tec_operator = str(tec_row.get('Customer Name', '')).lower()
            repd_operator = str(repd_row.get('Operator (or Applicant)', '')).lower()
            if tec_operator and repd_operator and len(tec_operator) > 3 and len(repd_operator) > 3:
                operator_score = fuzz.token_set_ratio(tec_operator, repd_operator)
                match_scores['operator_match'] = operator_score
            else:
                match_scores['operator_match'] = 0.0
            
            # Calculate overall confidence score
            # Weight the different matching methods
            weights = {
                'exact_normalized': 3.0,
                'fuzzy_token_set': 2.0,
                'fuzzy_token_sort': 1.5,
                'keyword_match': 2.0,
                'capacity_match': 1.5,
                'distance_match': 2.5,
                'operator_match': 1.0,
                'fuzzy_ratio': 1.0,
                'fuzzy_partial': 1.0
            }
            
            weighted_score = sum(score * weights.get(method, 1.0) for method, score in match_scores.items())
            total_weight = sum(weights.values())
            overall_score = weighted_score / total_weight
            
            # Store match if above threshold
            if overall_score >= 40.0:  # Lower threshold to catch more matches
                match_data = {
                    'tec_generator': tec_name,
                    'tec_capacity': tec_capacity,
                    'tec_normalized': tec_normalized,
                    'repd_site': repd_name,
                    'repd_capacity': repd_capacity,
                    'repd_normalized': repd_normalized,
                    'repd_x_coord': repd_row['repd_x'],
                    'repd_y_coord': repd_row['repd_y'],
                    'overall_score': overall_score,
                    'distance_km': distance_km if distance_km != float('inf') else np.nan,
                    'capacity_diff_pct': abs(tec_capacity - repd_capacity) / max(tec_capacity, repd_capacity) * 100 if pd.notna(tec_capacity) and pd.notna(repd_capacity) else np.nan,
                    **match_scores
                }
                best_matches.append(match_data)
        
        # Keep top 3 matches for each TEC generator
        if best_matches:
            best_matches.sort(key=lambda x: x['overall_score'], reverse=True)
            matches.extend(best_matches[:3])
    
    # Convert to DataFrame
    matches_df = pd.DataFrame(matches)
    
    if len(matches_df) == 0:
        logger.warning("No matches found!")
        return pd.DataFrame()
    
    # Sort by overall score
    matches_df = matches_df.sort_values('overall_score', ascending=False)
    
    # Add confidence categories
    def categorize_confidence(score):
        if score >= 80:
            return 'very_high'
        elif score >= 65:
            return 'high'
        elif score >= 50:
            return 'medium'
        else:
            return 'low'
    
    matches_df['confidence'] = matches_df['overall_score'].apply(categorize_confidence)
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(output_file, index=False)
    
    # Summary statistics
    total_matches = len(matches_df)
    confidence_counts = matches_df['confidence'].value_counts()
    
    logger.info(f"Super enhanced matching completed!")
    logger.info(f"Total potential matches: {total_matches}")
    logger.info(f"Confidence distribution:")
    for conf, count in confidence_counts.items():
        logger.info(f"  {conf}: {count}")
    
    # Show top matches
    logger.info("\nTop 10 matches:")
    top_matches = matches_df.head(10)[['tec_generator', 'repd_site', 'overall_score', 'confidence', 'distance_km', 'capacity_diff_pct']]
    for _, match in top_matches.iterrows():
        logger.info(f"  {match['tec_generator']} → {match['repd_site']} (Score: {match['overall_score']:.1f}, Conf: {match['confidence']})")
    
    return matches_df

def main():
    """Main function to run super enhanced TEC-REPD matching."""
    matches_df = enhanced_tec_repd_matching()
    
    if len(matches_df) > 0:
        logger.info(f"Results saved to: resources/generators/super_enhanced_tec_repd_duplicates.csv")
    else:
        logger.warning("No matches found to save")

if __name__ == "__main__":
    main()

