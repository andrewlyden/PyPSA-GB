#!/usr/bin/env python3
"""
Generate comprehensive validation report for PyPSA-GB scenarios.

This script produces a detailed report showing:
- All validation checks performed
- Data availability status
- Scenario configuration summary
- Recommendations for improvements

Usage:
    python scripts/generate_validation_report.py
    python scripts/generate_validation_report.py --output report.md
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from scripts.utilities.scenario_detection import (
    validate_all_active_scenarios,
    summarize_scenario_configuration,
    CURRENT_YEAR,
    HISTORICAL_YEAR_THRESHOLD,
    LATEST_DUKES_YEAR,
    LATEST_REPD_YEAR,
    LATEST_ESPENI_YEAR,
    LATEST_CUTOUT_YEAR
)
import yaml


def generate_markdown_report() -> str:
    """Generate comprehensive validation report in Markdown format."""
    
    # Load scenarios
    with open("config/scenarios_master.yaml", 'r', encoding='utf-8') as f:
        all_scenarios = yaml.safe_load(f)["scenarios"]
    
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    active_scenario_ids = cfg.get("run_scenarios", [])
    
    # Run validation
    validation_result = validate_all_active_scenarios(verbose=False, check_files=True)
    
    # Get summary
    summary = summarize_scenario_configuration(
        {sid: all_scenarios[sid] for sid in active_scenario_ids if sid in all_scenarios}
    )
    
    # Build report
    report = []
    report.append(f"# PyPSA-GB Validation Report")
    report.append(f"")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**System Year**: {CURRENT_YEAR}")
    report.append(f"**Historical Threshold**: ≤ {HISTORICAL_YEAR_THRESHOLD}")
    report.append(f"")
    
    # Validation summary
    report.append(f"## Validation Summary")
    report.append(f"")
    if validation_result['valid']:
        report.append(f"✅ **Status**: ALL SCENARIOS VALID")
    else:
        report.append(f"❌ **Status**: VALIDATION ERRORS FOUND")
    report.append(f"")
    report.append(f"- Total Scenarios: {validation_result['summary']['total_scenarios']}")
    report.append(f"- Errors: {validation_result['summary']['total_errors']}")
    report.append(f"- Warnings: {validation_result['summary']['total_warnings']}")
    report.append(f"")
    
    # Scenario breakdown
    report.append(f"## Scenario Breakdown")
    report.append(f"")
    report.append(f"### Historical Scenarios ({len(summary['historical'])})")
    report.append(f"")
    if summary['historical']:
        for sid in summary['historical']:
            config = all_scenarios[sid]
            report.append(f"- **{sid}**")
            report.append(f"  - Modelled Year: {config['modelled_year']}")
            report.append(f"  - Renewables Year: {config['renewables_year']}")
            report.append(f"  - Network: {config.get('network_model', 'N/A')}")
            report.append(f"  - Data Source: DUKES + REPD (primary), FES (fallback)")
            report.append(f"")
    else:
        report.append(f"*No historical scenarios active*")
        report.append(f"")
    
    report.append(f"### Future Scenarios ({len(summary['future'])})")
    report.append(f"")
    if summary['future']:
        for sid in summary['future']:
            config = all_scenarios[sid]
            report.append(f"- **{sid}**")
            report.append(f"  - Modelled Year: {config['modelled_year']}")
            report.append(f"  - FES Year: {config.get('FES_year', 'N/A')}")
            report.append(f"  - FES Scenario: {config.get('FES_scenario', 'N/A')}")
            report.append(f"  - Network: {config.get('network_model', 'N/A')}")
            report.append(f"")
    else:
        report.append(f"*No future scenarios active*")
        report.append(f"")
    
    # Data requirements
    report.append(f"## Data Requirements")
    report.append(f"")
    
    report.append(f"### Weather Cutouts")
    report.append(f"")
    report.append(f"Required years: {summary['cutout_years_needed']}")
    report.append(f"")
    cutout_dir = Path("resources/atlite/cutouts")
    for year in summary['cutout_years_needed']:
        cutout_file = cutout_dir / f"uk-{year}.nc"
        if cutout_file.exists():
            size_mb = cutout_file.stat().st_size / (1024 * 1024)
            report.append(f"- ✅ `uk-{year}.nc` ({size_mb:.1f} MB)")
        else:
            report.append(f"- ❌ `uk-{year}.nc` (MISSING)")
    report.append(f"")
    
    if summary['dukes_years_needed']:
        report.append(f"### DUKES Data")
        report.append(f"")
        report.append(f"Required for historical years: {summary['dukes_years_needed']}")
        report.append(f"")
        dukes_file = Path("data/generators/DUKES/DUKES_5.11.xls")
        if dukes_file.exists():
            report.append(f"- ✅ `DUKES_5.11.xls` (contains years 2010-2020)")
        else:
            report.append(f"- ❌ `DUKES_5.11.xls` (MISSING)")
        report.append(f"")
    
    if summary['fes_years_needed']:
        report.append(f"### FES Data")
        report.append(f"")
        report.append(f"Required FES years: {summary['fes_years_needed']}")
        report.append(f"")
        for year in summary['fes_years_needed']:
            fes_file = Path(f"resources/FES/FES_{year}_data.csv")
            if fes_file.exists():
                report.append(f"- ✅ `FES_{year}_data.csv`")
            else:
                report.append(f"- ⚠️  `FES_{year}_data.csv` (will be downloaded)")
        report.append(f"")
    
    # Data currency
    report.append(f"## Data Currency Status")
    report.append(f"")
    report.append(f"| Dataset | Latest Available | Age (years) | Status |")
    report.append(f"|---------|------------------|-------------|--------|")
    
    dukes_age = CURRENT_YEAR - LATEST_DUKES_YEAR
    dukes_status = "✅ Current" if dukes_age <= 2 else "⚠️ Aging"
    report.append(f"| DUKES 5.11 | {LATEST_DUKES_YEAR} | {dukes_age} | {dukes_status} |")
    
    repd_age = CURRENT_YEAR - LATEST_REPD_YEAR
    repd_status = "✅ Current" if repd_age <= 2 else "⚠️ Aging"
    report.append(f"| REPD | {LATEST_REPD_YEAR} | {repd_age} | {repd_status} |")
    
    espeni_age = CURRENT_YEAR - LATEST_ESPENI_YEAR
    espeni_status = "✅ Current" if espeni_age <= 2 else "⚠️ Aging"
    report.append(f"| ESPENI Demand | {LATEST_ESPENI_YEAR} | {espeni_age} | {espeni_status} |")
    
    cutout_age = CURRENT_YEAR - LATEST_CUTOUT_YEAR
    cutout_status = "✅ Current" if cutout_age <= 2 else "⚠️ Aging"
    report.append(f"| Weather Cutouts | {LATEST_CUTOUT_YEAR} | {cutout_age} | {cutout_status} |")
    report.append(f"")
    
    # Detailed validation results
    report.append(f"## Detailed Validation Results")
    report.append(f"")
    
    for scenario_id in active_scenario_ids:
        if scenario_id not in validation_result['scenarios']:
            report.append(f"### ❌ {scenario_id}")
            report.append(f"")
            report.append(f"*Scenario not found in scenarios_master.yaml*")
            report.append(f"")
            continue
        
        validation = validation_result['scenarios'][scenario_id]
        config = all_scenarios.get(scenario_id, {})
        
        if validation['errors']:
            report.append(f"### ❌ {scenario_id}")
        elif validation['warnings']:
            report.append(f"### ⚠️  {scenario_id}")
        else:
            report.append(f"### ✅ {scenario_id}")
        
        report.append(f"")
        report.append(f"**Configuration**:")
        report.append(f"- Modelled Year: {config.get('modelled_year', 'N/A')}")
        report.append(f"- Network Model: {config.get('network_model', 'N/A')}")
        report.append(f"- Renewables Year: {config.get('renewables_year', 'N/A')}")
        if config.get('modelled_year', 0) > HISTORICAL_YEAR_THRESHOLD:
            report.append(f"- FES Year: {config.get('FES_year', 'N/A')}")
            report.append(f"- FES Scenario: {config.get('FES_scenario', 'N/A')}")
        report.append(f"")
        
        if validation['errors']:
            report.append(f"**Errors ({len(validation['errors'])})**:")
            report.append(f"")
            for error in validation['errors']:
                report.append(f"- {error}")
            report.append(f"")
        
        if validation['warnings']:
            report.append(f"**Warnings ({len(validation['warnings'])})**:")
            report.append(f"")
            for warning in validation['warnings']:
                report.append(f"- {warning}")
            report.append(f"")
        
        if validation['info'] and not validation['errors']:
            report.append(f"**Info**:")
            report.append(f"")
            for info in validation['info']:
                report.append(f"- {info}")
            report.append(f"")
    
    # Recommendations
    report.append(f"## Recommendations")
    report.append(f"")
    
    if validation_result['valid']:
        report.append(f"✅ All scenarios are properly configured and ready to run!")
        report.append(f"")
    else:
        report.append(f"❌ Fix the errors above before running workflows.")
        report.append(f"")
    
    if validation_result['summary']['total_warnings'] > 0:
        report.append(f"⚠️  Review warnings - they may indicate data updates needed.")
        report.append(f"")
    
    # Add tips
    report.append(f"### Tips")
    report.append(f"")
    report.append(f"- Run `python scripts/validate_scenarios.py` for detailed validation")
    report.append(f"- Use `python scripts/validate_scenarios.py --no-files` for quick config checks")
    report.append(f"- Generate cutouts with `snakemake -s Snakefile_cutouts --cores 2`")
    report.append(f"- Update REPD data from: https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract")
    report.append(f"")
    
    report.append(f"---")
    report.append(f"")
    report.append(f"*Report generated by `scripts/generate_validation_report.py`*")
    
    return "\n".join(report)


def main():
    """Generate validation report."""
    parser = argparse.ArgumentParser(description="Generate PyPSA-GB validation report")
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output file (default: print to console)'
    )
    
    args = parser.parse_args()
    
    try:
        report = generate_markdown_report()
        
        if args.output:
            args.output.write_text(report, encoding='utf-8')
            print(f"✅ Validation report written to: {args.output}")
        else:
            print(report)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

