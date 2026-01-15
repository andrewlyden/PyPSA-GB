"""
Single Word Matching for TEC-REPD Sites

This script looks for single word matches between unmapped TEC generators
and REPD sites to find additional location matches.

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re
import time
from collections import defaultdict

# Set up logging - compatible with both standalone and Snakemake execution
try:
    from scripts.utilities.logging_config import setup_logging, get_snakemake_logger, log_execution_summary
    # Check if we're running under Snakemake
    if 'snakemake' in globals():
        logger = get_snakemake_logger()
    else:
        logger = setup_logging("single_word_tec_repd_matching")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

def extract_meaningful_words(name: str, min_length: int = 4) -> set:
    """Extract meaningful words from a site name."""
    if pd.isna(name):
        return set()
    
    name = str(name).lower().strip()
    
    # Remove common technology/business words that aren't location-specific
    exclude_words = {
        'energy', 'power', 'battery', 'storage', 'bess', 'wind', 'farm', 'solar', 
        'park', 'centre', 'center', 'station', 'plant', 'facility', 'site',
        'expansion', 'extension', 'phase', 'stage', 'development', 'project',
        'limited', 'ltd', 'company', 'corp', 'plc', 'group', 'holdings',
        'substation', 'connection', 'terminal', 'grid', 'national', 'electric',
        'generation', 'renewable', 'green', 'clean', 'sustainable', 'carbon',
        'technology', 'technologies', 'systems', 'solutions', 'services',
        'operation', 'operations', 'management', 'investment', 'infrastructure'
    }
    
    # Clean the name
    name = re.sub(r'[^\w\s]', ' ', name)  # Remove punctuation
    name = re.sub(r'\s+', ' ', name)      # Normalize whitespace
    
    # Extract words
    words = set()
    for word in name.split():
        # Skip if too short or in exclude list
        if len(word) >= min_length and word not in exclude_words:
            # Skip numbers unless they're part of a larger word
            if not word.isdigit():
                words.add(word)
    
    return words

def find_single_word_matches():
    """Find matches based on meaningful single words."""
    
    logger.info("Starting single word matching analysis")
    
    # Load data
    generators_df = pd.read_csv("resources/generators/dispatchable_generators_final_with_manual_matches.csv")
    repd_df = pd.read_csv("data/renewables/repd-q2-july-2024.csv", encoding='utf-8-sig')
    
    # Get unmapped TEC generators
    unmapped = generators_df[
        (generators_df['data_source'] == 'TEC') & 
        (generators_df[['x_coord', 'y_coord']].isna().any(axis=1))
    ].copy()
    
    # Get REPD sites with coordinates (all statuses to catch more matches)
    repd_with_coords = repd_df[
        (pd.notna(repd_df['X-coordinate'])) &
        (pd.notna(repd_df['Y-coordinate']))
    ].copy()
    
    logger.info(f"Checking {len(unmapped)} unmapped TEC generators against {len(repd_with_coords)} REPD sites")
    
    # Extract meaningful words for each dataset
    logger.info("Extracting meaningful words...")
    unmapped['meaningful_words'] = unmapped['site_name'].apply(extract_meaningful_words)
    repd_with_coords['meaningful_words'] = repd_with_coords['Site Name'].apply(extract_meaningful_words)
    
    # Create word-to-sites mapping for REPD
    repd_word_map = defaultdict(list)
    for idx, row in repd_with_coords.iterrows():
        for word in row['meaningful_words']:
            repd_word_map[word].append(idx)
    
    matches = []
    
    logger.info("Finding word-based matches...")
    for _, tec_row in unmapped.iterrows():
        tec_name = tec_row['site_name']
        tec_capacity = tec_row['capacity_mw']
        tec_words = tec_row['meaningful_words']
        
        if not tec_words:
            continue
        
        # Find REPD sites that share words
        potential_repd_indices = set()
        shared_words = set()
        
        for word in tec_words:
            if word in repd_word_map:
                potential_repd_indices.update(repd_word_map[word])
                shared_words.add(word)
        
        if not potential_repd_indices:
            continue
        
        # Evaluate each potential match
        for repd_idx in potential_repd_indices:
            repd_row = repd_with_coords.loc[repd_idx]
            repd_name = repd_row['Site Name']
            repd_capacity = pd.to_numeric(repd_row['Installed Capacity (MWelec)'], errors='coerce')
            repd_words = repd_row['meaningful_words']
            
            # Calculate word overlap
            common_words = tec_words.intersection(repd_words)
            if not common_words:
                continue
            
            word_overlap_score = len(common_words) / max(len(tec_words), len(repd_words)) * 100
            
            # Calculate name similarity
            name_similarity = fuzz.token_set_ratio(tec_name.lower(), repd_name.lower())
            
            # Calculate capacity similarity
            capacity_match = 0
            capacity_diff_pct = np.nan
            if pd.notna(tec_capacity) and pd.notna(repd_capacity) and max(tec_capacity, repd_capacity) > 0:
                capacity_diff_pct = abs(tec_capacity - repd_capacity) / max(tec_capacity, repd_capacity) * 100
                capacity_match = max(0, 100 - capacity_diff_pct)
            
            # Overall score (weighted combination)
            overall_score = (word_overlap_score * 0.5 + name_similarity * 0.3 + capacity_match * 0.2)
            
            # Only include if there's meaningful overlap
            if len(common_words) >= 1 and overall_score >= 30:
                matches.append({
                    'tec_generator': tec_name,
                    'tec_capacity': tec_capacity,
                    'tec_words': ', '.join(sorted(tec_words)),
                    'repd_site': repd_name,
                    'repd_capacity': repd_capacity,
                    'repd_words': ', '.join(sorted(repd_words)),
                    'repd_status': repd_row['Development Status (short)'],
                    'repd_x_coord': repd_row['X-coordinate'],
                    'repd_y_coord': repd_row['Y-coordinate'],
                    'common_words': ', '.join(sorted(common_words)),
                    'num_common_words': len(common_words),
                    'word_overlap_score': word_overlap_score,
                    'name_similarity': name_similarity,
                    'capacity_match': capacity_match,
                    'capacity_diff_pct': capacity_diff_pct,
                    'overall_score': overall_score
                })
    
    # Convert to DataFrame and process
    matches_df = pd.DataFrame(matches)
    
    if len(matches_df) == 0:
        logger.info("No single word matches found")
        return matches_df
    
    # Sort by overall score and remove duplicates (keep best match per TEC generator)
    matches_df = matches_df.sort_values('overall_score', ascending=False)
    matches_df = matches_df.drop_duplicates('tec_generator', keep='first')
    
    # Add confidence categories
    def categorize_confidence(row):
        score = row['overall_score']
        num_words = row['num_common_words']
        
        if score >= 70 and num_words >= 2:
            return 'high'
        elif score >= 50 and num_words >= 1:
            return 'medium'
        else:
            return 'low'
    
    matches_df['confidence'] = matches_df.apply(categorize_confidence, axis=1)
    
    # Save results
    matches_df.to_csv("resources/generators/single_word_tec_repd_matches.csv", index=False)
    
    logger.info(f"Found {len(matches_df)} single word matches")
    
    # Show summary by confidence
    confidence_counts = matches_df['confidence'].value_counts()
    logger.info("Confidence distribution:")
    for conf, count in confidence_counts.items():
        logger.info(f"  {conf}: {count}")
    
    # Show top matches
    logger.info("\nTop 10 single word matches:")
    top_matches = matches_df.head(10)
    for _, match in top_matches.iterrows():
        logger.info(f"  {match['tec_generator']} → {match['repd_site']}")
        logger.info(f"    Score: {match['overall_score']:.1f} | Common words: {match['common_words']} | Confidence: {match['confidence']}")
        logger.info(f"    Capacity: {match['tec_capacity']:.1f} MW vs {match['repd_capacity']:.1f} MW")
        logger.info("")
    
    return matches_df

def apply_single_word_matches(confidence_threshold: str = 'medium'):
    """Apply coordinates from high-confidence single word matches."""
    
    logger.info(f"Applying single word matches with {confidence_threshold}+ confidence")
    
    # Load data
    generators_df = pd.read_csv("resources/generators/dispatchable_generators_final_with_manual_matches.csv")
    
    try:
        matches_df = pd.read_csv("resources/generators/single_word_tec_repd_matches.csv")
    except FileNotFoundError:
        logger.warning("No single word matches file found. Run matching first.")
        return generators_df
    
    if len(matches_df) == 0:
        logger.info("No matches to apply")
        return generators_df
    
    # Filter by confidence
    if confidence_threshold == 'high':
        valid_confidences = ['high']
    elif confidence_threshold == 'medium':
        valid_confidences = ['high', 'medium']
    else:
        valid_confidences = ['high', 'medium', 'low']
    
    high_confidence = matches_df[matches_df['confidence'].isin(valid_confidences)].copy()
    
    logger.info(f"Applying {len(high_confidence)} matches with {confidence_threshold}+ confidence")
    
    updated_generators = generators_df.copy()
    applied_count = 0
    
    for _, match in high_confidence.iterrows():
        tec_name = match['tec_generator']
        repd_x = match['repd_x_coord']
        repd_y = match['repd_y_coord']
        
        # Find matching generator
        gen_mask = updated_generators['site_name'] == tec_name
        matching_gens = updated_generators[gen_mask]
        
        if len(matching_gens) == 0:
            continue
        
        # Check if already has coordinates
        if updated_generators.loc[gen_mask, 'x_coord'].notna().any():
            logger.info(f"Generator '{tec_name}' already has coordinates, skipping")
            continue
        
        # Apply coordinates
        updated_generators.loc[gen_mask, 'x_coord'] = repd_x
        updated_generators.loc[gen_mask, 'y_coord'] = repd_y
        updated_generators.loc[gen_mask, 'location_source'] = 'single_word_match'
        
        applied_count += len(matching_gens)
        logger.info(f"Applied coordinates to '{tec_name}': ({repd_x}, {repd_y})")
        logger.info(f"  Common words: {match['common_words']}")
    
    # Save updated database
    updated_generators.to_csv("resources/generators/dispatchable_generators_with_single_word_matches.csv", index=False)
    
    # Calculate improvement
    original_located = generators_df[generators_df[['x_coord', 'y_coord']].notna().all(axis=1)]
    updated_located = updated_generators[updated_generators[['x_coord', 'y_coord']].notna().all(axis=1)]
    
    improvement = len(updated_located) - len(original_located)
    new_success_rate = len(updated_located) / len(updated_generators) * 100
    
    # Still unmapped
    still_unmapped = updated_generators[updated_generators[['x_coord', 'y_coord']].isna().any(axis=1)]
    unmapped_count = len(still_unmapped)
    unmapped_capacity = still_unmapped['capacity_mw'].sum()
    
    logger.info(f"\nSingle word matching results:")
    logger.info(f"  Coordinates applied: {applied_count}")
    logger.info(f"  New locations: {len(updated_located)}/{len(updated_generators)}")
    logger.info(f"  New success rate: {new_success_rate:.1f}%")
    logger.info(f"  Improvement: +{improvement} generators located")
    logger.info(f"  Still unmapped: {unmapped_count} generators ({unmapped_capacity:.0f} MW)")
    
    return updated_generators

def load_tec_data(tec_file_path):
    """Load TEC register data."""
    tec_df = pd.read_csv(tec_file_path, encoding='utf-8-sig')
    # Filter for Built status only
    tec_df = tec_df[tec_df['Project Status'] == 'Built']
    return tec_df

def load_repd_data(repd_file_path):
    """Load REPD data."""
    repd_df = pd.read_csv(repd_file_path, encoding='utf-8-sig')
    # Filter for operational sites
    repd_df = repd_df[repd_df['Development Status (short)'] == 'Operational']
    return repd_df

def find_single_word_matches_for_snakemake(unmapped_generators, repd_df):
    """Find single word matches between unmapped generators and REPD sites."""
    logger.info("Extracting meaningful words...")
    
    # Extract meaningful words from TEC generator names
    tec_words = {}
    for idx, row in unmapped_generators.iterrows():
        words = extract_meaningful_words(row['site_name'])
        if words:
            tec_words[idx] = {
                'name': row['site_name'],
                'words': words,
                'capacity': row.get('capacity_mw', 0)
            }
    
    # Extract meaningful words from REPD site names  
    repd_words = {}
    for idx, row in repd_df.iterrows():
        words = extract_meaningful_words(row['Site Name'])
        if words:
            repd_words[idx] = {
                'name': row['Site Name'],
                'words': words,
                'capacity': row.get('Installed Capacity (MWelec)', 0),
                'x_coord': row.get('X-coordinate', None),
                'y_coord': row.get('Y-coordinate', None)
            }
    
    logger.info("Finding word-based matches...")
    
    matches = []
    for tec_idx, tec_data in tec_words.items():
        best_score = 0
        best_match = None
        
        for repd_idx, repd_data in repd_words.items():
            # Find common words
            common_words = set(tec_data['words']) & set(repd_data['words'])
            
            if common_words:
                # Calculate score based on word overlap and fuzzy similarity
                overlap_score = len(common_words) / max(len(tec_data['words']), len(repd_data['words']))
                name_similarity = fuzz.ratio(tec_data['name'], repd_data['name']) / 100
                
                # Combined score with emphasis on word overlap
                total_score = (overlap_score * 0.7) + (name_similarity * 0.3)
                
                if total_score > best_score and total_score > 0.4:  # Minimum threshold
                    best_score = total_score
                    best_match = {
                        'repd_idx': repd_idx,
                        'repd_data': repd_data,
                        'common_words': common_words,
                        'overlap_score': overlap_score,
                        'name_similarity': name_similarity
                    }
        
        if best_match:
            # Determine confidence level
            if best_score > 0.8:
                confidence = 'high'
            elif best_score > 0.6:
                confidence = 'medium'  
            else:
                confidence = 'low'
            
            matches.append({
                'tec_generator_idx': tec_idx,
                'tec_generator_name': tec_data['name'],
                'tec_capacity': tec_data['capacity'],
                'repd_site_name': best_match['repd_data']['name'],
                'repd_capacity': best_match['repd_data']['capacity'],
                'x_coordinate': best_match['repd_data']['x_coord'],
                'y_coordinate': best_match['repd_data']['y_coord'],
                'score': best_score * 100,
                'common_words': ', '.join(sorted(best_match['common_words'])),
                'confidence': confidence
            })
    
    matches_df = pd.DataFrame(matches)
    logger.info(f"Found {len(matches_df)} single word matches")
    
    if len(matches_df) > 0:
        confidence_counts = matches_df['confidence'].value_counts()
        logger.info("Confidence distribution:")
        for confidence, count in confidence_counts.items():
            logger.info(f"  {confidence}: {count}")
        
        # Show top matches
        logger.info("\nTop 10 single word matches:")
        top_matches = matches_df.nlargest(10, 'score')
        for _, match in top_matches.iterrows():
            logger.info(f"  {match['tec_generator_name']} → {match['repd_site_name']}")
            logger.info(f"    Score: {match['score']:.1f} | Common words: {match['common_words']} | Confidence: {match['confidence']}")
            logger.info(f"    Capacity: {match['tec_capacity']} MW vs {match['repd_capacity']} MW")
            logger.info("")
    
    return matches_df

def apply_single_word_matches_for_snakemake(generators_df, matches_df):
    """Apply single word matches to the generator database."""
    logger.info("Applying single word matches with medium+ confidence")
    
    # Filter for medium and high confidence matches
    good_matches = matches_df[matches_df['confidence'].isin(['medium', 'high'])]
    logger.info(f"Applying {len(good_matches)} matches with medium+ confidence")
    
    updated_generators = generators_df.copy()
    
    for _, match in good_matches.iterrows():
        tec_idx = match['tec_generator_idx']
        
        if pd.notna(match['x_coordinate']) and pd.notna(match['y_coordinate']):
            # Update coordinates
            updated_generators.loc[tec_idx, 'longitude'] = match['x_coordinate']
            updated_generators.loc[tec_idx, 'latitude'] = match['y_coordinate']
            updated_generators.loc[tec_idx, 'location_source'] = f"REPD_single_word_match"
            
            logger.info(f"Applied coordinates to '{match['tec_generator_name']}': ({match['x_coordinate']}, {match['y_coordinate']})")
            logger.info(f"  Common words: {match['common_words']}")
    
    return updated_generators

def main():
    """Main function for Snakemake execution."""
    start_time = time.time()
    
    # Get input and output files from snakemake
    input_generators_file = snakemake.input[0]  # dispatchable_generators_with_dukes_locations.csv
    input_tec_file = snakemake.input[1]         # tec-register-02-september-2025.csv
    input_repd_file = snakemake.input[2]        # repd-q2-july-2024.csv
    
    output_generators_file = snakemake.output[0]  # dispatchable_generators_final.csv
    output_matches_file = snakemake.output[1]     # single_word_tec_repd_matches.csv
    
    logger.info("Starting single word matching analysis")
    
    # Load the generator database
    generators_df = pd.read_csv(input_generators_file)
    logger.info(f"Loaded {len(generators_df)} generators from {input_generators_file}")
    
    # Load TEC and REPD data
    tec_df = load_tec_data(input_tec_file)
    repd_df = load_repd_data(input_repd_file)
    logger.info(f"Loaded {len(tec_df)} TEC entries and {len(repd_df)} REPD sites")
    
    # Find unmapped generators
    unmapped_generators = generators_df[generators_df['x_coord'].isna() | generators_df['y_coord'].isna()]
    logger.info(f"Checking {len(unmapped_generators)} unmapped TEC generators against {len(repd_df)} REPD sites")
    
    # Perform single word matching
    matches_df = find_single_word_matches_for_snakemake(unmapped_generators, repd_df)
    
    # Save matches to output file
    matches_df.to_csv(output_matches_file, index=False)
    logger.info(f"Saved {len(matches_df)} single word matches to {output_matches_file}")
    
    # Apply matches to generator database
    updated_generators = apply_single_word_matches_for_snakemake(generators_df, matches_df)
    
    # Save final generator database
    updated_generators.to_csv(output_generators_file, index=False)
    logger.info(f"Saved updated generator database to {output_generators_file}")
    
    # Report results
    original_with_coords = len(generators_df.dropna(subset=['x_coord', 'y_coord']))
    final_with_coords = len(updated_generators.dropna(subset=['x_coord', 'y_coord']))
    improvement = final_with_coords - original_with_coords
    success_rate = (final_with_coords / len(updated_generators)) * 100
    
    logger.info(f"""
Single word matching results:
  Coordinates applied: {improvement}
  New locations: {final_with_coords}/{len(updated_generators)}
  New success rate: {success_rate:.1f}%
  Improvement: +{improvement} generators located
  Still unmapped: {len(updated_generators) - final_with_coords} generators ({updated_generators[updated_generators['x_coord'].isna()]['capacity_mw'].sum():.0f} MW)""")
    
    # Log execution summary if available
    try:
        log_execution_summary(logger, "single_word_matching", start_time, 
                            inputs=[input_generators_file, input_tec_file, input_repd_file],
                            outputs=[output_generators_file, output_matches_file])
    except:
        duration = time.time() - start_time
        logger.info(f"Single word matching completed in {duration:.2f} seconds")
    
    logger.info("Single word matching completed!")

if __name__ == "__main__":
    if 'snakemake' in globals():
        main()
    else:
        # Find single word matches using the original logic
        matches_df = find_single_word_matches()
        
        if len(matches_df) > 0:
            # Apply medium+ confidence matches
            apply_single_word_matches(confidence_threshold='medium')
        
        logger.info("Single word matching completed!")

