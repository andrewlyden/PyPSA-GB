#!/usr/bin/env python3
"""
Scenario Validation Pre-Flight Check

This script performs comprehensive validation of active scenarios before
running the workflow. It checks:
- Scenario configuration completeness
- Data file availability (cutouts, DUKES, etc.)
- Historical vs Future scenario requirements
- Data freshness and currency

Usage:
    python scripts/validate_scenarios.py              # Full validation
    python scripts/validate_scenarios.py --no-files   # Config only (skip file checks)
    python scripts/validate_scenarios.py --quiet      # Minimal output

Exit codes:
    0 - All scenarios valid
    1 - Validation errors found
    2 - Configuration file errors
"""

import sys
from pathlib import Path
import argparse

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from scripts.utilities.scenario_detection import validate_all_active_scenarios


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate PyPSA-GB scenario configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_scenarios.py
      Full validation including file checks
      
  python scripts/validate_scenarios.py --no-files
      Validate configuration only (faster, good for config changes)
      
  python scripts/validate_scenarios.py --quiet
      Show only summary (good for CI/CD pipelines)
        """
    )
    
    parser.add_argument(
        '--no-files',
        action='store_true',
        help='Skip file existence checks (only validate configuration)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (only show summary)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/config.yaml'),
        help='Path to config.yaml (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--scenarios',
        type=Path,
        default=Path('config/scenarios.yaml'),
        help='Path to scenarios.yaml (default: config/scenarios.yaml)'
    )
    
    args = parser.parse_args()
    
    # Check if config files exist
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        return 2
    
    if not args.scenarios.exists():
        print(f"ERROR: Scenarios file not found: {args.scenarios}")
        return 2
    
    # Run validation
    try:
        result = validate_all_active_scenarios(
            config_path=args.config,
            scenarios_path=args.scenarios,
            check_files=not args.no_files,
            verbose=not args.quiet
        )
        
        # Return appropriate exit code
        if result['valid']:
            if not args.quiet:
                print("\nSUCCESS: All scenarios validated successfully!")
            return 0
        else:
            if not args.quiet:
                print(f"\nERROR: Validation failed with {result['summary']['total_errors']} error(s)")
            return 1
            
    except Exception as e:
        print(f"\nVALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

