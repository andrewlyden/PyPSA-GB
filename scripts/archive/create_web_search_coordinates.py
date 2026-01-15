"""
Web Search Results for Unmapped Generators

This script creates a CSV file with coordinates found through web searches
for the remaining unmapped generators.

Author: PyPSA-GB Development Team
Date: September 2025
"""

import pandas as pd
import numpy as np

def create_web_search_coordinates_csv():
    """Create CSV with coordinates found through web searches."""
    
    # Data found through web searches
    web_search_data = [
        {
            'site_name': 'Zenobe Blackhillock 300 MW',
            'description': 'Blackhillock Battery Energy Storage - between Aberdeen and Inverness',
            'latitude': 57.2,  # Approximate - between Aberdeen (57.1°N) and Inverness (57.5°N)
            'longitude': -3.5,  # Approximate - northeast Scotland
            'x_coord': 340000,  # Approximate British National Grid
            'y_coord': 850000,  # Approximate British National Grid  
            'capacity_mw': 300,
            'technology': 'battery_generic',
            'source': 'Web search - NS Energy Business, Zenobe company website',
            'location_quality': 'approximate',
            'notes': 'Multiple TEC entries for same project - 300MW total capacity in phases'
        },
        {
            'site_name': 'Green Frog @ Alcoa',
            'description': 'Green Frog Power operates gas turbines, likely at Alcoa site near Swansea',
            'latitude': 51.6446,
            'longitude': -4.0183,
            'x_coord': 249000,  # Converted from lat/lon to approximate BNG
            'y_coord': 194000,  # Converted from lat/lon to approximate BNG
            'capacity_mw': 40,
            'technology': 'ocgt',
            'source': 'Web search - Global Energy Monitor, Green Frog Power operations',
            'location_quality': 'approximate',
            'notes': 'Green Frog Power Ltd operates gas turbines at Swansea, likely same location'
        },
        {
            'site_name': 'Hunningley Stairfoot BESS',
            'description': 'Battery storage at Hunningley 132kV Substation, Barnsley',
            'latitude': 53.535,  # From Mapcarta coordinates for Hunningley
            'longitude': -1.44932,  # From Mapcarta coordinates  
            'x_coord': 441000,  # Converted from lat/lon to approximate BNG
            'y_coord': 405000,  # Converted from lat/lon to approximate BNG
            'capacity_mw': 40,
            'technology': 'battery_generic', 
            'source': 'Web search - Wikidata, TEC Register, Mapcarta',
            'location_quality': 'good',
            'notes': 'CARE POWER (BARNSLEY) LIMITED at Hunningley 132kV Substation'
        },
        {
            'site_name': 'Redditch',
            'description': 'CCGT plant in Redditch, Worcestershire',
            'latitude': 52.3089,
            'longitude': -1.9455,
            'x_coord': 404000,  # Approximate for Redditch town center
            'y_coord': 267000,  # Approximate for Redditch town center
            'capacity_mw': 29,
            'technology': 'ccgt',
            'source': 'Web search - Redditch town coordinates',
            'location_quality': 'town_center',
            'notes': 'Location assumed at Redditch town center - actual plant location unknown'
        },
        {
            'site_name': 'Blackpool BESS',
            'description': 'Battery storage in Blackpool area',
            'latitude': 53.8175,
            'longitude': -3.0357,
            'x_coord': 330000,  # Approximate for Blackpool
            'y_coord': 437000,  # Approximate for Blackpool
            'capacity_mw': 25.3,
            'technology': 'battery_generic',
            'source': 'Web search - Blackpool town coordinates',
            'location_quality': 'town_center',
            'notes': 'Location assumed at Blackpool area - actual plant location unknown'
        },
        {
            'site_name': 'Dowlais',
            'description': 'Thermal plant in Dowlais, Merthyr Tydfil, Wales',
            'latitude': 51.7634,
            'longitude': -3.3464,
            'x_coord': 306000,  # Approximate for Dowlais
            'y_coord': 208000,  # Approximate for Dowlais
            'capacity_mw': 21,
            'technology': 'thermal_other',
            'source': 'Web search - Dowlais town coordinates',
            'location_quality': 'town_center', 
            'notes': 'Historic industrial area in Wales - actual plant location unknown'
        }
    ]
    
    # Convert to DataFrame
    web_coords_df = pd.DataFrame(web_search_data)
    
    # Save to CSV in data/generators folder
    output_file = 'data/generators/web_search_generator_coordinates.csv'
    web_coords_df.to_csv(output_file, index=False)
    
    print(f"Created web search coordinates file: {output_file}")
    print(f"Found coordinates for {len(web_coords_df)} generators:")
    
    for _, row in web_coords_df.iterrows():
        print(f"  {row['site_name']} - {row['capacity_mw']} MW ({row['location_quality']} quality)")
    
    print(f"\nTotal capacity: {web_coords_df['capacity_mw'].sum():.1f} MW")
    print(f"Remaining unmapped: ~{911 - web_coords_df['capacity_mw'].sum():.0f} MW")
    
    return web_coords_df

def main():
    """Main function."""
    create_web_search_coordinates_csv()
    print("\nWeb search coordinate generation completed!")
    print("\nNote: Some coordinates are approximate based on town/area locations.")
    print("The Zenobe Blackhillock entries appear to be duplicates of the same 300MW project.")

if __name__ == "__main__":
    main()

