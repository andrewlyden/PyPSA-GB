#!/usr/bin/env python3
"""
Generate tutorial notebooks from analysis notebooks.

This script takes the analysis notebooks from resources/analysis and creates
tutorial versions with more explanatory text for the documentation.
"""

import json
import sys
from pathlib import Path

# Tutorial mappings: (analysis_notebook, tutorial_number, title, description)
TUTORIALS = [
    {
        'source': 'Historical_2015_reduced_notebook.ipynb',
        'target': '1-historical-baseload-2015.ipynb',
        'title': 'Historical Baseload (2015)',
        'description': """This tutorial demonstrates PyPSA-GB's network-constrained optimal power flow (LOPF) modeling using the **Historical_2015_reduced** scenario. The year 2015 represents the traditional "baseload era" with significant coal and nuclear generation, before the rapid renewable energy expansion of recent years.

## What You'll Learn

- Loading and inspecting a solved PyPSA-GB network
- Analyzing conventional generation dispatch patterns  
- Understanding marginal prices and network congestion
- Visualizing power flows across the GB transmission system
- Creating baseline metrics for comparing energy transition progress

## Prerequisites

Run the workflow to generate the solved network:
```bash
snakemake resources/network/Historical_2015_reduced_solved.nc -j 4
```

## Scenario Overview

| Parameter | Value |
|-----------|-------|
| **Modelled Year** | 2015 |
| **Network Model** | Reduced (32 buses) |
| **Data Source** | DUKES (thermal), REPD (renewables), ESPENI (demand) |
| **Solve Period** | First week of January |
| **Key Features** | Coal baseload, early renewables |""",
        'scenario_id': 'Historical_2015_reduced'
    },
    {
        'source': 'Historical_2023_etys_notebook.ipynb',
        'target': '2-historical-renewables-2023.ipynb',
        'title': 'Historical Renewables Era (2023)',
        'description': """This tutorial explores the **Historical_2023_etys** scenario using the full ETYS network topology (2000+ buses). By 2023, Great Britain's electricity system had undergone significant transformation with coal phase-out complete and wind/solar providing substantial generation.

## What You'll Learn

- Working with the detailed ETYS transmission network
- Analyzing high renewable penetration scenarios
- Identifying transmission constraints and congestion hotspots
- Understanding renewable curtailment patterns
- Evaluating system flexibility requirements

## Prerequisites

Run the workflow to generate the solved network:
```bash
snakemake resources/network/Historical_2023_etys_solved.nc -j 4
```
⚠️ **Note**: ETYS network optimization takes longer (~30-60 minutes for one week)

## Scenario Overview

| Parameter | Value |
|-----------|-------|
| **Modelled Year** | 2023 |
| **Network Model** | ETYS (2000+ buses) |
| **Data Source** | DUKES (thermal), REPD (renewables), ESPENI (demand) |
| **Solve Period** | First week of January |
| **Key Features** | No coal, high wind/solar, network detail |""",
        'scenario_id': 'Historical_2023_etys'
    },
    {
        'source': 'HT35_clustered_notebook.ipynb',
        'target': '3-future-holistic-transition-2035.ipynb',
        'title': 'Future Scenario: Holistic Transition (2035)',
        'description': """This tutorial examines the **Holistic Transition** 2035 scenario from NESO's Future Energy Scenarios (FES). This pathway represents a coordinated approach to decarbonization with strong policy support, balanced technology deployment, and system integration.

## What You'll Learn

- Exploring NESO Future Energy Scenarios (FES) projections
- Analyzing offshore wind expansion and grid integration
- Understanding storage dispatch and arbitrage patterns
- Evaluating transmission network upgrades and requirements
- Assessing hydrogen system integration (electrolysis + turbines)

## Prerequisites

Run the workflow to generate the solved network:
```bash
snakemake resources/network/HT35_clustered_solved.nc -j 4
```

## Scenario Overview

| Parameter | Value |
|-----------|-------|
| **Modelled Year** | 2035 |
| **FES Pathway** | Holistic Transition |
| **Network Model** | Clustered (~150 buses) |
| **Data Source** | FES 2024 projections |
| **Solve Period** | Representative week |
| **Key Features** | Major offshore wind, hydrogen, storage |""",
        'scenario_id': 'HT35_clustered'
    },
    {
        'source': 'EE50_clustered_notebook.ipynb',
        'target': '4-future-electric-engagement-2050.ipynb',
        'title': 'Future Scenario: Electric Engagement (2050)',
        'description': """This tutorial explores the **Electric Engagement** 2050 scenario from NESO's FES - an ambitious pathway achieving Net Zero with extensive electrification across transport, heating, and industry. This represents the most aggressive decarbonization trajectory.

## What You'll Learn

- Modeling highly electrified future energy systems
- Analyzing massive renewable capacity deployment
- Understanding seasonal storage and flexibility requirements
- Evaluating hydrogen's role in long-duration energy storage
- Identifying critical infrastructure and investment needs

## Prerequisites

Run the workflow to generate the solved network:
```bash
snakemake resources/network/EE50_clustered_solved.nc -j 4
```

## Scenario Overview

| Parameter | Value |
|-----------|-------|
| **Modelled Year** | 2050 |
| **FES Pathway** | Electric Engagement |
| **Network Model** | Clustered (~150 buses) |
| **Data Source** | FES 2024 projections |
| **Solve Period** | Representative week |
| **Key Features** | 100+ GW wind/solar, large H2 system, high demand |""",
        'scenario_id': 'EE50_clustered'
    }
]


def add_tutorial_intro(notebook, tutorial_info):
    """Add explanatory introduction to tutorial notebook."""
    # Replace first cell with tutorial introduction
    intro_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': f"# {tutorial_info['title']}\n\n{tutorial_info['description']}"
    }
    
    # Insert at beginning (after the title cell is removed)
    notebook['cells'][0] = intro_cell
    return notebook


def add_explanatory_text(notebook):
    """Add more explanatory markdown cells throughout the notebook."""
    
    # Add explanation before network topology section
    topology_explanations = [
        {
            'after': 'Interactive Network Topology',
            'text': """
### Understanding Network Topology

The maps below show the physical layout of Great Britain's electricity infrastructure:
- **Buses**: Connection points (substations, generation sites) shown as orange markers
- **Lines/Links**: Transmission corridors carrying power between buses
- **Geography**: Real GB coordinates using British National Grid projection

The network model captures how electricity flows are constrained by transmission capacity, creating locational price differences and potential congestion."""
        },
        {
            'after': 'Transmission Loading',
            'text': """
### Reading the Transmission Loading Map

Line colors indicate congestion levels:
- 🟢 **Green** (<60%): Uncongested, spare capacity available
- 🟡 **Gold** (60-80%): Moderate loading, approaching limits
- 🟠 **Orange** (80-95%): Heavy loading, limited spare capacity  
- 🔴 **Red** (>95%): Congested, at or near thermal limits

Line thickness scales with loading percentage - thicker lines carry more power relative to their rated capacity."""
        },
        {
            'after': 'Generator Dispatch',
            'text': """
### Understanding Generator Dispatch

This map visualizes where electricity generation is occurring:
- **Marker size**: Proportional to total generation at each bus
- **Marker color**: Intensity indicates generation level (light → dark red for low → high)
- **Background network**: Light gray lines show transmission connections

Large coastal markers typically indicate offshore wind farms, while inland concentrations show thermal power stations or large solar farms."""
        }
    ]
    
    # Add explanation before hydrogen section if it exists
    h2_explanation = """
### About the Hydrogen System

The hydrogen subsystem models the Power-to-Gas-to-Power pathway:
1. **Electrolysis**: Converts surplus renewable electricity to hydrogen (dashed turquoise links)
2. **H2 Storage**: Stores hydrogen for later use (purple markers)
3. **H2 Turbines**: Converts hydrogen back to electricity during high demand (red markers)

This provides seasonal energy storage, shifting summer renewable surplus to winter demand peaks."""
    
    # Find and add explanations
    new_cells = []
    for i, cell in enumerate(notebook['cells']):
        new_cells.append(cell)
        
        # Add explanations after specific section headers
        if cell['cell_type'] == 'markdown':
            source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            for expl in topology_explanations:
                if expl['after'] in source_text:
                    new_cells.append({
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': expl['text']
                    })
            
            if 'Hydrogen System Analysis' in source_text:
                new_cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': h2_explanation
                })
    
    notebook['cells'] = new_cells
    return notebook


def create_tutorial_notebook(source_path, target_path, tutorial_info):
    """Create tutorial notebook from analysis notebook."""
    
    print(f"Creating tutorial: {tutorial_info['target']}")
    print(f"  Source: {source_path}")
    print(f"  Target: {target_path}")
    
    # Load source notebook
    with open(source_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Add tutorial introduction
    notebook = add_tutorial_intro(notebook, tutorial_info)
    
    # Add explanatory text throughout
    notebook = add_explanatory_text(notebook)
    
    # Update any scenario references
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = source.replace('Generated automatically by PyPSA-GB workflow', 
                                      f'PyPSA-GB Tutorial: {tutorial_info["title"]}')
                cell['source'] = source
            elif isinstance(source, list):
                cell['source'] = [s.replace('Generated automatically by PyPSA-GB workflow',
                                           f'PyPSA-GB Tutorial: {tutorial_info["title"]}') 
                                 for s in source]
    
    # Save tutorial notebook
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Created: {target_path.stat().st_size / 1024:.1f} KB\n")


def main():
    """Main execution."""
    repo_root = Path(__file__).parent.parent.parent
    analysis_dir = repo_root / 'resources' / 'analysis'
    tutorial_dir = repo_root / 'docs' / 'source' / 'tutorials'
    
    print("=" * 80)
    print("GENERATING TUTORIAL NOTEBOOKS")
    print("=" * 80)
    print(f"Analysis notebooks: {analysis_dir}")
    print(f"Tutorial directory: {tutorial_dir}")
    print()
    
    created_count = 0
    for tutorial_info in TUTORIALS:
        source_file = analysis_dir / tutorial_info['source']
        target_file = tutorial_dir / tutorial_info['target']
        
        if not source_file.exists():
            print(f"⚠️  Source not found: {source_file}")
            print(f"   Skipping {tutorial_info['target']}\n")
            continue
        
        create_tutorial_notebook(source_file, target_file, tutorial_info)
        created_count += 1
    
    print("=" * 80)
    print(f"✓ Created {created_count}/{len(TUTORIALS)} tutorial notebooks")
    print("=" * 80)


if __name__ == '__main__':
    main()
