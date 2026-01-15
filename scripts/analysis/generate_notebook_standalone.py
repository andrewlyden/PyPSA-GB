#!/usr/bin/env python3
"""
Standalone script to generate analysis notebooks for solved PyPSA networks.

This script bypasses Snakemake and directly generates a Jupyter notebook
for analyzing a solved network.

Usage:
    python generate_notebook_standalone.py HT35_clustered
"""

import sys
import os
import json
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from analysis.generate_analysis_notebook import create_analysis_notebook

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_notebook_standalone.py <scenario_id>")
        print("\nAvailable solved scenarios:")
        network_dir = Path("resources/network")
        solved_networks = sorted([f.stem.replace("_solved", "") for f in network_dir.glob("*_solved.nc")])
        for scenario in solved_networks:
            print(f"  - {scenario}")
        sys.exit(1)
    
    scenario_id = sys.argv[1]
    
    # Paths
    network_path = f"resources/network/{scenario_id}_solved.nc"
    output_path = f"resources/analysis/{scenario_id}_notebook.ipynb"
    
    # Check if network exists
    if not Path(network_path).exists():
        print(f"❌ Error: Solved network not found at {network_path}")
        print("\nAvailable solved scenarios:")
        network_dir = Path("resources/network")
        solved_networks = sorted([f.stem.replace("_solved", "") for f in network_dir.glob("*_solved.nc")])
        for s in solved_networks:
            print(f"  - {s}")
        sys.exit(1)
    
    print(f"📓 Generating analysis notebook for scenario: {scenario_id}")
    print(f"  Input:  {network_path}")
    print(f"  Output: {output_path}")
    print()
    
    try:
        # Generate the notebook
        notebook = create_analysis_notebook(scenario_id, network_path, output_path)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write notebook to file as valid JSON with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        # Verify the file is valid JSON
        with open(output_path, 'r', encoding='utf-8') as f:
            json.load(f)
        
        file_size_kb = Path(output_path).stat().st_size / 1024
        print(f"✓ Notebook generated successfully: {output_path}")
        print(f"  File size: {file_size_kb:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error generating notebook: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
