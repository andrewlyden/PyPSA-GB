#!/usr/bin/env python3
"""
Create Interconnector Pipeline Placeholder
==========================================

This script creates a placeholder CSV file for future interconnector pipeline
projects and capacity expansion planning. It provides a structured schema for
future expansion of interconnector capacity.

Key features:
- Standardized schema for pipeline projects
- Template for future capacity planning
- Integration with expansion planning workflows
- Metadata and documentation structure

Author: PyPSA-GB Team
"""

import sys
import pandas as pd
import logging
from pathlib import Path
import time

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.utilities.logging_config import setup_logging, log_execution_summary
except ImportError:
    import logging
    def setup_logging(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

# Check if running in Snakemake context
if 'snakemake' in globals():
    SNAKEMAKE_MODE = True
    output_placeholder = snakemake.output[0]
else:
    SNAKEMAKE_MODE = False

def create_pipeline_schema() -> pd.DataFrame:
    """
    Create the schema for interconnector pipeline data.
    
    Returns:
        DataFrame with pipeline schema and example entries
    """
    logger = logging.getLogger(__name__)
    
    # Define the schema columns
    schema_columns = [
        'name',                    # Project name
        'landing_point_gb',        # GB landing point
        'counterparty_country',    # Destination country
        'counterparty_landing_point',  # Foreign landing point
        'capacity_mw',             # Planned capacity in MW
        'dc',                      # DC technology (True/False)
        'losses_percent',          # Expected losses percentage
        'target_year',             # Target commissioning year
        'status',                  # Project status
        'cost_millions',           # Estimated cost in millions
        'developer',               # Project developer
        'technology',              # Cable technology (e.g., HVDC VSC)
        'cable_length_km',         # Approximate cable length
        'voltage_kv',              # Operating voltage
        'planning_consent',        # Planning consent status
        'marine_license',          # Marine licensing status
        'grid_connection_gb',      # GB grid connection status
        'grid_connection_foreign', # Foreign grid connection status
        'environmental_impact',    # Environmental impact assessment
        'commercial_framework',    # Commercial/regulatory framework
        'notes',                   # Additional notes
        'source',                  # Data source
        'last_updated'             # Last update date
    ]
    
    # Create empty DataFrame with schema
    pipeline_df = pd.DataFrame(columns=schema_columns)
    
    # Add example entries to demonstrate the schema
    example_entries = [
        {
            'name': 'Example_Future_Interconnector_1',
            'landing_point_gb': 'Example GB Terminal',
            'counterparty_country': 'Example Country',
            'counterparty_landing_point': 'Example Foreign Terminal',
            'capacity_mw': 1000.0,
            'dc': True,
            'losses_percent': 2.5,
            'target_year': 2030,
            'status': 'Planning',
            'cost_millions': 1500.0,
            'developer': 'Example Developer',
            'technology': 'HVDC VSC',
            'cable_length_km': 150.0,
            'voltage_kv': 400.0,
            'planning_consent': 'Pending',
            'marine_license': 'Not Started',
            'grid_connection_gb': 'Under Review',
            'grid_connection_foreign': 'Not Started',
            'environmental_impact': 'In Progress',
            'commercial_framework': 'Under Development',
            'notes': 'Example pipeline project for schema demonstration',
            'source': 'Pipeline Placeholder',
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d')
        },
        {
            'name': 'Example_Expansion_Project_2',
            'landing_point_gb': 'Another GB Terminal',
            'counterparty_country': 'Another Country',
            'counterparty_landing_point': 'Another Foreign Terminal',
            'capacity_mw': 1400.0,
            'dc': True,
            'losses_percent': 3.0,
            'target_year': 2035,
            'status': 'Concept',
            'cost_millions': 2000.0,
            'developer': 'Another Developer',
            'technology': 'HVDC LCC',
            'cable_length_km': 200.0,
            'voltage_kv': 500.0,
            'planning_consent': 'Not Started',
            'marine_license': 'Not Started',
            'grid_connection_gb': 'Not Started',
            'grid_connection_foreign': 'Not Started',
            'environmental_impact': 'Not Started',
            'commercial_framework': 'Concept Stage',
            'notes': 'Longer-term expansion project example',
            'source': 'Pipeline Placeholder',
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
    ]
    
    # Add example entries to DataFrame
    for entry in example_entries:
        pipeline_df = pd.concat([pipeline_df, pd.DataFrame([entry])], ignore_index=True)
    
    logger.info(f"Created pipeline schema with {len(schema_columns)} columns and {len(example_entries)} example entries")
    
    return pipeline_df

def create_metadata_documentation() -> str:
    """
    Create documentation for the pipeline schema.
    
    Returns:
        Multiline string with schema documentation
    """
    documentation = """
# Interconnector Pipeline Schema Documentation

This file provides a schema for future interconnector pipeline projects and capacity expansion planning.

## Column Descriptions:

### Basic Project Information
- **name**: Unique project identifier/name
- **landing_point_gb**: GB landing point or terminal location
- **counterparty_country**: Destination country for the interconnector
- **counterparty_landing_point**: Foreign terminal or landing point

### Technical Specifications
- **capacity_mw**: Planned transmission capacity in MW
- **dc**: Whether the interconnector uses DC technology (True/False)
- **losses_percent**: Expected transmission losses as percentage
- **technology**: Cable technology type (e.g., HVDC VSC, HVDC LCC)
- **cable_length_km**: Approximate total cable length
- **voltage_kv**: Operating voltage in kilovolts

### Project Timeline and Status
- **target_year**: Target commissioning year
- **status**: Current project status (Concept, Planning, Under Construction, etc.)
- **last_updated**: Date of last data update

### Commercial and Financial
- **cost_millions**: Estimated project cost in millions (currency unspecified)
- **developer**: Primary project developer or consortium
- **commercial_framework**: Commercial/regulatory framework status

### Regulatory and Consents
- **planning_consent**: Planning permission status
- **marine_license**: Marine licensing status (for subsea cables)
- **grid_connection_gb**: GB grid connection agreement status
- **grid_connection_foreign**: Foreign grid connection status
- **environmental_impact**: Environmental impact assessment status

### Additional Information
- **notes**: Free-text additional information
- **source**: Data source or reference
- **last_updated**: Date when record was last updated

## Usage Notes:

1. This is a template/placeholder file for future pipeline projects
2. Example entries are included to demonstrate the schema
3. Real pipeline data should replace or supplement the example entries
4. All monetary values should specify currency in notes if relevant
5. Status values should be standardized across projects
6. Dates should follow YYYY-MM-DD format

## Integration with PyPSA-GB:

This schema is designed to integrate with:
- Network expansion planning workflows
- Cost-benefit analysis tools
- Scenario planning and sensitivity analysis
- Long-term capacity planning studies

## Data Sources:

Potential data sources for pipeline projects include:
- National Grid ESO Future Energy Scenarios
- Ofgem Impact Assessments
- Developer announcements and feasibility studies
- European Network of Transmission System Operators (ENTSO-E) reports
- Government policy documents and strategies
"""
    
    return documentation

def main():
    """Main processing function."""
    logger = setup_logging("interconnector_pipeline_placeholder")
    start_time = time.time()
    
    try:
        logger.info("Creating interconnector pipeline placeholder...")
        logger.info(f"Running in {'Snakemake' if SNAKEMAKE_MODE else 'standalone'} mode")
        
        if SNAKEMAKE_MODE:
            output_file = output_placeholder
        else:
            output_file = "resources/interconnectors/pipeline_placeholder.csv"
        
        logger.info(f"Output file: {output_file}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline schema
        pipeline_df = create_pipeline_schema()
        
        # Save the placeholder file
        pipeline_df.to_csv(output_file, index=False)
        logger.info(f"Saved pipeline placeholder to: {output_file}")
        
        # Create documentation file
        doc_file = output_file.replace('.csv', '_documentation.md')
        documentation = create_metadata_documentation()
        
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(documentation)
        logger.info(f"Saved documentation to: {doc_file}")
        
        # Calculate statistics
        schema_columns = len(pipeline_df.columns)
        example_entries = len(pipeline_df)
        
        # Log execution summary
        log_execution_summary(
            logger,
            "interconnector_pipeline_placeholder",
            start_time,
            inputs={},
            outputs={'placeholder': output_file, 'documentation': doc_file},
            context={
                'schema_columns': schema_columns,
                'example_entries': example_entries
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating pipeline placeholder: {e}")
        if SNAKEMAKE_MODE:
            raise
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()

