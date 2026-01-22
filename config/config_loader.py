"""
Configuration Loading Utility for PyPSA-GB

Handles the hierarchical configuration system:
  1. defaults.yaml    - Base defaults for all scenarios
  2. scenarios.yaml   - Scenario definitions (override defaults)
  3. config.yaml      - Run selection + global overrides
  4. clustering.yaml  - Clustering presets

Usage:
    from config_loader import load_config, get_scenario
    
    config = load_config()
    scenario = get_scenario("HT35", config)
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.
    Nested dicts are merged recursively, not replaced entirely.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(filepath: Path) -> Dict:
    """Load a YAML file, return empty dict if not found."""
    if not filepath.exists():
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def resolve_clustering(clustering_value: Any, clustering_presets: Dict) -> Dict:
    """
    Resolve clustering configuration.
    
    Args:
        clustering_value: Either a string (preset name) or dict (inline config)
        clustering_presets: Dictionary of preset configurations
    
    Returns:
        Resolved clustering configuration dict
        
    Note:
        If a dict has 'method' specified, clustering is enabled regardless of 
        the 'enabled' key (which may come from defaults). The 'enabled: false'
        only disables clustering if no method is specified.
    """
    if clustering_value is None:
        return {"enabled": False}
    
    if isinstance(clustering_value, str):
        # Reference to preset
        if clustering_value not in clustering_presets:
            raise ValueError(f"Unknown clustering preset: '{clustering_value}'. "
                           f"Available: {list(clustering_presets.keys())}")
        config = deepcopy(clustering_presets[clustering_value])
        config["enabled"] = True
        return config
    
    if isinstance(clustering_value, dict):
        # Allow referencing preset while extending with custom options
        if "preset" in clustering_value:
            preset_name = clustering_value.get("preset")
            if preset_name not in clustering_presets:
                raise ValueError(f"Unknown clustering preset: '{preset_name}'. "
                               f"Available: {list(clustering_presets.keys())}")
            config = deep_merge(
                deepcopy(clustering_presets[preset_name]),
                {k: v for k, v in clustering_value.items() if k != "preset"}
            )
            config["enabled"] = True
            return config
        
        # If 'method' is specified, clustering should be enabled
        # (even if 'enabled: false' was inherited from defaults)
        if "method" in clustering_value:
            config = deepcopy(clustering_value)
            config["enabled"] = True
            return config
        
        # No method specified - respect the enabled flag
        if not clustering_value.get("enabled", False):
            return {"enabled": False}
        
        # Enabled but no method - unusual, but keep the config
        config = deepcopy(clustering_value)
        config["enabled"] = True
        return config
    
    return {"enabled": False}


def load_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the complete configuration with proper inheritance.
    
    Returns a dict with:
        - defaults: Default settings
        - scenarios: Dict of scenario_id -> merged scenario config
        - run_scenarios: List of scenario IDs to run
        - clustering_presets: Available clustering presets
        - global_overrides: Overrides from config.yaml
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    # Load all config files
    defaults = load_yaml(config_dir / "defaults.yaml")
    scenarios_raw = load_yaml(config_dir / "scenarios.yaml")
    main_config = load_yaml(config_dir / "config.yaml")
    clustering_config = load_yaml(config_dir / "clustering.yaml")
    
    # Support both old (scenarios_master.yaml) and new (scenarios.yaml) format
    if not scenarios_raw and (config_dir / "scenarios_master.yaml").exists():
        scenarios_master = load_yaml(config_dir / "scenarios_master.yaml")
        scenarios_raw = scenarios_master.get("scenarios", {})
    
    # Extract clustering presets
    clustering_presets = clustering_config.get("presets", {})
    aggregation_strategies = clustering_config.get("aggregation_strategies", {})
    
    # Extract global overrides from main config (excluding special keys)
    # Note: logging and network_naming are now in defaults.yaml, not config.yaml
    special_keys = {"run_scenarios", "solve_mode"}
    global_overrides = {k: v for k, v in main_config.items() if k not in special_keys}
    
    # Build merged scenarios
    merged_scenarios = {}
    for scenario_id, scenario_config in scenarios_raw.items():
        if scenario_config is None:
            continue
            
        # Skip template comments (strings that start with template markers)
        if isinstance(scenario_config, str):
            continue
        
        # Start with defaults
        merged = deepcopy(defaults)
        
        # Apply global overrides
        merged = deep_merge(merged, global_overrides)
        
        # Apply scenario-specific settings
        merged = deep_merge(merged, scenario_config)
        
        # Resolve clustering references
        if "clustering" in merged:
            merged["clustering"] = resolve_clustering(
                merged["clustering"], 
                clustering_presets
            )
        
        # Add scenario ID for reference
        merged["_scenario_id"] = scenario_id
        
        merged_scenarios[scenario_id] = merged
    
    return {
        "defaults": defaults,
        "scenarios": merged_scenarios,
        "run_scenarios": main_config.get("run_scenarios", []),
        "clustering_presets": clustering_presets,
        "aggregation_strategies": aggregation_strategies,
        "global_overrides": global_overrides,
        "logging": defaults.get("logging", {}),
        "network_naming": defaults.get("network_naming", {}),
        "solve_mode": main_config.get("solve_mode", defaults.get("solve_mode", "LP")),
    }


def get_scenario(scenario_id: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get a fully resolved scenario configuration.
    
    Args:
        scenario_id: The scenario identifier
        config: Pre-loaded config (optional, will load if not provided)
    
    Returns:
        Merged scenario configuration dict
    """
    if config is None:
        config = load_config()
    
    if scenario_id not in config["scenarios"]:
        available = list(config["scenarios"].keys())
        raise ValueError(f"Unknown scenario: '{scenario_id}'. Available: {available}")
    
    return config["scenarios"][scenario_id]


def get_active_scenarios(config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Get list of scenarios that are marked to run in config.yaml.
    
    Returns:
        List of merged scenario configuration dicts
    """
    if config is None:
        config = load_config()
    
    scenarios = []
    for scenario_id in config["run_scenarios"]:
        if scenario_id in config["scenarios"]:
            scenarios.append(config["scenarios"][scenario_id])
        else:
            print(f"Warning: Scenario '{scenario_id}' in run_scenarios not found")
    
    return scenarios


def validate_scenario(scenario: Dict[str, Any]) -> List[str]:
    """
    Validate a scenario configuration.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    scenario_id = scenario.get("_scenario_id", "unknown")
    
    # Required fields
    required = ["modelled_year", "renewables_year", "demand_year"]
    for field in required:
        if field not in scenario:
            errors.append(f"[{scenario_id}] Missing required field: {field}")
    
    # Future scenarios need FES config
    modelled_year = scenario.get("modelled_year", 0)
    if modelled_year > 2024:
        if "FES_year" not in scenario:
            errors.append(f"[{scenario_id}] Future scenario (year {modelled_year}) requires FES_year")
        if "FES_scenario" not in scenario:
            errors.append(f"[{scenario_id}] Future scenario (year {modelled_year}) requires FES_scenario")
    
    # Validate network_model
    valid_networks = ["ETYS", "Reduced", "Zonal"]
    network_model = scenario.get("network_model", "ETYS")
    if network_model not in valid_networks:
        errors.append(f"[{scenario_id}] Invalid network_model: {network_model}. Must be one of {valid_networks}")
    
    return errors


# ══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="PyPSA-GB Configuration Loader")
    parser.add_argument("--scenario", "-s", help="Show specific scenario config")
    parser.add_argument("--list", "-l", action="store_true", help="List all scenarios")
    parser.add_argument("--active", "-a", action="store_true", help="List active scenarios")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate all scenarios")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    config = load_config()
    
    if args.list:
        print("Available scenarios:")
        for sid in sorted(config["scenarios"].keys()):
            desc = config["scenarios"][sid].get("description", "")
            print(f"  {sid}: {desc}")
    
    elif args.active:
        print("Active scenarios (from run_scenarios):")
        for sid in config["run_scenarios"]:
            if sid in config["scenarios"]:
                desc = config["scenarios"][sid].get("description", "")
                print(f"  ✓ {sid}: {desc}")
            else:
                print(f"  ✗ {sid}: NOT FOUND")
    
    elif args.validate:
        print("Validating scenarios...")
        all_errors = []
        for sid, scenario in config["scenarios"].items():
            errors = validate_scenario(scenario)
            all_errors.extend(errors)
        
        if all_errors:
            print(f"\n{len(all_errors)} validation errors:")
            for error in all_errors:
                print(f"  ✗ {error}")
        else:
            print(f"  ✓ All {len(config['scenarios'])} scenarios valid")
    
    elif args.scenario:
        scenario = get_scenario(args.scenario, config)
        if args.json:
            print(json.dumps(scenario, indent=2, default=str))
        else:
            print(f"Scenario: {args.scenario}")
            print("-" * 40)
            for key, value in sorted(scenario.items()):
                if not key.startswith("_"):
                    print(f"  {key}: {value}")
    
    else:
        parser.print_help()
