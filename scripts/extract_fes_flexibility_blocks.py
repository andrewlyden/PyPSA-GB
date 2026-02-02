"""
Extract and document FES Storage & Flexibility Building Blocks

This script extracts the flexibility-related building blocks from FES data
and saves them to a structured format for easy reference.
"""
import pandas as pd
from pathlib import Path

# Define flexibility building blocks based on FES documentation
FLEXIBILITY_BLOCKS = {
    'Srg_BB005': {
        'name': 'V2G',
        'category': 'Storage',
        'unit': 'MW availability',
        'description': 'Potential MW available to participate in V2G',
        'notes': 'Storage (GW) - Vehicle-to-Grid bidirectional flexibility'
    },
    'Srg_BB006a': {
        'name': 'I&C Flexibility (TouT)',
        'category': 'I&C Flexibility',
        'unit': 'MW availability',
        'description': 'Potential MW available to participate in DSR',
        'notes': 'Total Industrial & Commercial demand available for Demand Response (Non-contracted)'
    },
    'Srg_BB006b': {
        'name': 'I&C Flexibility (TouT) %',
        'category': 'I&C Flexibility',
        'unit': '% customers',
        'description': '% Potential participating customers',
        'notes': 'Total Industrial & Commercial demand available for Demand Response (Non-contracted)'
    },
    'Srg_BB007a': {
        'name': 'Electric Vehicle Smart Charging (TouT)',
        'category': 'Domestic Flexibility',
        'unit': 'MW availability',
        'description': 'Potential MW available to participate in DSR',
        'notes': 'Total domestic EV demand available for Demand Response. Separate EV-based tariff assumed'
    },
    'Srg_BB007b': {
        'name': 'Electric Vehicle Smart Charging (TouT) %',
        'category': 'Domestic Flexibility',
        'unit': '% customers',
        'description': '% Potential participating customers',
        'notes': 'Total domestic EV demand available for Demand Response. Separate EV-based tariff assumed'
    },
    'Srg_BB008a': {
        'name': 'Domestic Flexibility (TouT)',
        'category': 'Domestic Flexibility',
        'unit': 'MW availability',
        'description': 'Potential MW available to participate in DSR',
        'notes': 'Total domestic demand (non-EV) available for Demand Response'
    },
    'Srg_BB008b': {
        'name': 'Domestic Flexibility (TouT) %',
        'category': 'Domestic Flexibility',
        'unit': '% customers',
        'description': '% Potential participating customers',
        'notes': 'Total domestic demand (non-EV) available for Demand Response'
    },
}

def extract_flexibility_data(fes_path: Path, output_dir: Path, fes_year: int = 2024):
    """Extract flexibility building blocks from FES data."""

    print(f"\nExtracting FES {fes_year} Flexibility Building Blocks")
    print("=" * 80)

    # Read FES data
    fes_data = pd.read_csv(fes_path, low_memory=False)

    # Clean column names (remove BOM)
    fes_data.columns = [col.replace('\ufeff', '').strip() for col in fes_data.columns]

    print(f"Loaded FES data: {len(fes_data)} rows")
    print(f"Columns: {list(fes_data.columns[:10])}...")

    # Extract flexibility blocks
    flexibility_data = fes_data[fes_data['Building Block ID Number'].isin(FLEXIBILITY_BLOCKS.keys())].copy()

    print(f"\nFound {len(flexibility_data)} rows for flexibility building blocks")

    if len(flexibility_data) == 0:
        print("WARNING: No flexibility building blocks found!")
        return

    # Get unique blocks found
    blocks_found = flexibility_data['Building Block ID Number'].unique()
    print(f"\nBuilding blocks found: {sorted(blocks_found)}")

    # Save extracted data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"FES_{fes_year}_flexibility_building_blocks.csv"
    flexibility_data.to_csv(output_path, index=False)
    print(f"\n[OK] Saved flexibility data to: {output_path}")

    # Create summary document
    summary_path = output_dir / f"FES_{fes_year}_flexibility_blocks_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"FES {fes_year} STORAGE & FLEXIBILITY BUILDING BLOCKS\n")
        f.write("=" * 80 + "\n\n")

        f.write("These building blocks define the potential for demand-side flexibility\n")
        f.write("across different technologies and customer segments.\n\n")

        for bb_id, info in FLEXIBILITY_BLOCKS.items():
            f.write(f"\n{bb_id}: {info['name']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Category:    {info['category']}\n")
            f.write(f"Unit:        {info['unit']}\n")
            f.write(f"Description: {info['description']}\n")
            f.write(f"Notes:       {info['notes']}\n")

            # Check if this block was found
            if bb_id in blocks_found:
                block_data = flexibility_data[flexibility_data['Building Block ID Number'] == bb_id]
                n_rows = len(block_data)
                scenarios = block_data['FES Pathway'].unique()
                f.write(f"Found:       {n_rows} entries across scenarios: {', '.join(scenarios)}\n")
            else:
                f.write(f"Found:       NOT FOUND IN DATA\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DATA STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        f.write("Each building block has:\n")
        f.write("- GSP-level granularity (Grid Supply Points)\n")
        f.write("- Yearly projections from 2023 to 2050\n")
        f.write("- Multiple FES pathways (scenarios)\n\n")

        f.write("Common FES Pathways:\n")
        f.write("- Leading the Way: Fastest decarbonization, highest ambition\n")
        f.write("- Consumer Transformation: Consumer-led change\n")
        f.write("- System Transformation: System-led change\n")
        f.write("- Falling Short: Slower progress\n")
        f.write("- Holistic Transition: Balanced pathway (NEW in FES 2024)\n")
        f.write("- Counterfactual: Business-as-usual baseline\n")

    print(f"[OK] Saved summary to: {summary_path}")

    # Create aggregated summary by scenario and year
    if len(flexibility_data) > 0:
        pivot_path = output_dir / f"FES_{fes_year}_flexibility_summary_by_scenario.csv"

        # Select key years
        key_years = ['2025', '2030', '2035', '2040', '2045', '2050']
        cols = ['Building Block ID Number', 'FES Pathway', 'Unit'] + key_years

        # Filter to columns that exist
        available_years = [y for y in key_years if y in flexibility_data.columns]
        cols = ['Building Block ID Number', 'FES Pathway', 'Unit'] + available_years

        summary = flexibility_data.groupby(['Building Block ID Number', 'FES Pathway', 'Unit'])[available_years].sum().reset_index()
        summary.to_csv(pivot_path, index=False)
        print(f"[OK] Saved scenario summary to: {pivot_path}")

        # Print quick overview
        print(f"\n{'Block':<12} {'Scenario':<25} {'Unit':<15} {' '.join([f'{y:>8}' for y in available_years])}")
        print("-" * 120)
        for _, row in summary.iterrows():
            values = ' '.join([f"{row[y]:>8.1f}" for y in available_years])
            print(f"{row['Building Block ID Number']:<12} {row['FES Pathway']:<25} {row['Unit']:<15} {values}")

if __name__ == "__main__":
    # Process FES 2024
    fes_2024_path = Path("resources/FES/FES_2024_data.csv")
    output_dir = Path("resources/FES/building_blocks")

    if fes_2024_path.exists():
        extract_flexibility_data(fes_2024_path, output_dir, fes_year=2024)
    else:
        print(f"ERROR: FES data not found at {fes_2024_path}")

    # Process FES 2025 if available
    fes_2025_path = Path("resources/FES/FES_2025_data.csv")
    if fes_2025_path.exists():
        print("\n" + "=" * 80)
        extract_flexibility_data(fes_2025_path, output_dir, fes_year=2025)

    print("\n" + "=" * 80)
    print("[OK] Flexibility building block extraction complete!")
    print("=" * 80)
