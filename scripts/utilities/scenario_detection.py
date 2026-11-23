"""
Scenario Detection and Auto-Configuration Module

This module handles intelligent detection of historical vs. future scenarios
and automatically configures data sources appropriately.

Key Functions:
- is_historical_scenario: Determine if scenario year is historical (≤2024)
- target_source: Where a scenario's targets come from (CP30 | FES | historical)
- uses_brownfield / uses_gsp_blocks / build_method: How its fleet and network are built
- auto_configure_scenario: Automatically set data source based on scenario year
- validate_historical_scenario: Validate historical scenario configuration
- validate_future_scenario: Validate future scenario configuration
- validate_scenario_complete: Complete scenario validation
- summarize_scenario_configuration: Generate summary of scenario setup
"""

from pathlib import Path
from datetime import datetime
import logging
import os

from scripts.sources import registry as _registry

logger = logging.getLogger(__name__)

#: Names of NESO's Clean Power 2030 pathways (data/scenario_sources/pathways.csv).
CP30_PATHWAYS = {p.name for p in _registry.pathways("CP30")}
CP30_REQUIRED_RESOURCE_KEYS = {"es1", "es5", "ed1", "reference", "validation", "interconnector_prices"}


def is_historical_scenario(scenario_year):
    """
    Determine if a scenario year is historical (≤2024) or future (>2024).
    
    Args:
        scenario_year (int or dict): The year of the scenario, or scenario config dict
        
    Returns:
        bool: True if historical (≤2024), False if future (>2024)
    """
    # Handle both int and dict inputs
    if isinstance(scenario_year, dict):
        scenario_year = scenario_year.get('modelled_year') or scenario_year.get('year')
    
    if scenario_year is None:
        return False  # Treat None as future/unknown
    
    current_year = datetime.now().year
    threshold = min(2024, current_year)  # Dynamic threshold: never go beyond current year
    return scenario_year <= threshold


def _legacy_target_source(scenario_dict):
    """The target source of a dict that has no ``targets`` key, from the old keys.

    A merged scenario from ``config_loader.load_config`` always carries ``targets`` and
    ``method``; this covers dicts built elsewhere (tests, standalone script runs). It is the
    rule ``is_cp30_scenario`` used before stage 4: a historical year, then an explicit CP30
    marker (scenario_source/source/data_source "CP30" or a CP30_pathway key), else FES.
    """
    if is_historical_scenario(scenario_dict):
        return "historical"

    source = (
        scenario_dict.get("scenario_source")
        or scenario_dict.get("source")
        or scenario_dict.get("data_source")
        or ""
    )
    if str(source).upper() == "CP30" or "CP30_pathway" in scenario_dict:
        return "CP30"

    return "FES"


#: The method an old-style scenario is built with, by its target source (config_loader.DERIVED_METHOD).
_DERIVED_METHOD = {"historical": "historical", "CP30": "brownfield", "FES": "gsp_block"}


def target_source(scenario_dict):
    """Where a scenario's targets come from: "CP30", "FES" or "historical".

    Reads ``targets.source`` (written, or derived by config_loader). Use it for gates about
    numbers and source files: CP30 ES1/ES5/ED1 resources, which FES edition serves as the GSP
    proxy, CP30 validation targets.
    """
    targets = (scenario_dict or {}).get("targets")
    if isinstance(targets, dict) and targets.get("source"):
        return targets["source"]
    return _legacy_target_source(scenario_dict or {})


def build_method(scenario_dict):
    """How a scenario's fleet and network are built: "brownfield", "gsp_block" or "historical"."""
    method = (scenario_dict or {}).get("method")
    if method:
        return method
    return _DERIVED_METHOD[target_source(scenario_dict)]


def uses_brownfield(scenario_dict):
    """True when the fleet is built brownfield (named REPD/TEC projects, ES5 regions, station schedules).

    Use it for gates about how the fleet or network is built and for the method's defaults
    (config/defaults/method/brownfield.yaml: fleet.spatial_proxy placement, fleet.brownfield, regional allocation,
    demand.build).
    """
    return build_method(scenario_dict) == "brownfield"


def uses_gsp_blocks(scenario_dict):
    """True when the fleet is built from FES GSP building blocks (the FES long-range chain)."""
    return build_method(scenario_dict) == "gsp_block"


def get_scenario_source(scenario_dict):
    """The authoritative scenario data source (``target_source``; kept for older callers)."""
    return target_source(scenario_dict)


def _uses_fes_spatial_proxy(scenario_dict):
    """Return True if a CP30 scenario explicitly requests FES as a proxy."""
    proxy = (scenario_dict.get("fleet") or {}).get("spatial_proxy") or {}
    return str(proxy.get("source", "")).upper() == "FES"


def auto_configure_scenario(scenario_dict):
    """
    Automatically configure scenario based on year (historical vs future).
    
    Args:
        scenario_dict (dict): The scenario configuration dictionary
        
    Returns:
        dict: Enhanced scenario configuration with auto-configured data sources
    """
    enhanced = dict(scenario_dict)
    # Try both 'modelled_year' and 'year' for compatibility
    scenario_year = scenario_dict.get("modelled_year") or scenario_dict.get("year", 2025)
    
    scenario_source = target_source(scenario_dict)
    is_hist = scenario_source == "historical"
    
    if is_hist:
        # Historical scenario: use DUKES + REPD
        enhanced["scenario_source"] = "historical"
        enhanced["data_source"] = "historical"
        enhanced["generator_data_source"] = "DUKES"
        enhanced["demand_source"] = "ESPENI"
        enhanced["_needs_fes"] = False
    elif scenario_source == "CP30":
        # CP30 future scenario: CP30 totals are authoritative. FES may still be
        # used explicitly for spatial proxying if configured under fleet.spatial_proxy.
        enhanced["scenario_source"] = "CP30"
        enhanced["data_source"] = "CP30"
        enhanced["generator_data_source"] = "CP30"
        enhanced["demand_source"] = "CP30_ED1"
        enhanced["storage_data_source"] = "CP30"
        enhanced["interconnector_data_source"] = "CP30"
        enhanced["spatial_data_source"] = (
            "FES" if _uses_fes_spatial_proxy(scenario_dict) else "CP30"
        )
        enhanced["_needs_fes"] = _uses_fes_spatial_proxy(scenario_dict)
    else:
        # Future scenario: use FES
        enhanced["scenario_source"] = "FES"
        enhanced["data_source"] = "future"
        enhanced["generator_data_source"] = "FES"
        enhanced["demand_source"] = "FES"
        enhanced["_needs_fes"] = True
    
    return enhanced


def validate_historical_scenario(scenario_dict, scenario_id):
    """
    Validate a historical scenario configuration.
    
    Args:
        scenario_dict (dict): The scenario configuration
        scenario_id (str): The scenario identifier
        
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check year (try both field names)
    year = scenario_dict.get("modelled_year") or scenario_dict.get("year")
    if not year:
        errors.append(f"Scenario {scenario_id}: Missing 'modelled_year' or 'year' field")
    elif year > 2024:
        errors.append(f"Scenario {scenario_id}: Year {year} > 2024, should be future scenario")
    
    # Check network model
    if "network_model" not in scenario_dict:
        errors.append(f"Scenario {scenario_id}: Missing 'network_model'")
    
    # Warnings for old data
    if year and year < 2020:
        warnings.append(f"Scenario {scenario_id}: Using data from year {year} (>4 years old)")
    
    return (len(errors) == 0, errors, warnings)


def validate_future_scenario(scenario_dict, scenario_id):
    """
    Validate a future scenario configuration.
    
    Args:
        scenario_dict (dict): The scenario configuration
        scenario_id (str): The scenario identifier
        
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Check year (try both field names)
    year = scenario_dict.get("modelled_year") or scenario_dict.get("year")
    if not year:
        errors.append(f"Scenario {scenario_id}: Missing 'modelled_year' or 'year' field")
    elif year <= 2024:
        errors.append(f"Scenario {scenario_id}: Year {year} <= 2024, should be historical scenario")
    
    # Check network model
    if "network_model" not in scenario_dict:
        errors.append(f"Scenario {scenario_id}: Missing 'network_model'")
    
    scenario_source = target_source(scenario_dict)
    if scenario_source == "CP30":
        cp30_valid, cp30_errors, cp30_warnings = validate_cp30_scenario(scenario_dict, scenario_id)
        errors.extend(cp30_errors)
        warnings.extend(cp30_warnings)
    else:
        # Check FES scenario
        if "FES_year" not in scenario_dict:
            warnings.append(f"Scenario {scenario_id}: No FES year specified")
        if "FES_scenario" not in scenario_dict:
            warnings.append(f"Scenario {scenario_id}: No FES scenario specified")
    
    return (len(errors) == 0, errors, warnings)


def validate_cp30_scenario(scenario_dict, scenario_id):
    """
    Validate CP30-specific scenario configuration.

    CP30 scenarios must not rely on FES scenario fields for authoritative totals.
    FES proxy use is allowed only when explicitly declared under fleet.spatial_proxy.
    """
    errors = []
    warnings = []

    pathway = scenario_dict.get("CP30_pathway")
    if not pathway:
        errors.append(f"Scenario {scenario_id}: Missing CP30_pathway")
    elif pathway not in CP30_PATHWAYS:
        errors.append(
            f"Scenario {scenario_id}: Invalid CP30_pathway '{pathway}'. "
            f"Must be one of {sorted(CP30_PATHWAYS)}"
        )

    if "FES_year" in scenario_dict or "FES_scenario" in scenario_dict:
        warnings.append(
            f"Scenario {scenario_id}: CP30 scenario has top-level FES fields; "
            "use fleet.spatial_proxy for explicit FES proxy assumptions"
        )

    source_data = scenario_dict.get("source_data") or {}
    workbook = source_data.get("workbook")
    if not workbook:
        errors.append(f"Scenario {scenario_id}: Missing source_data.workbook")

    resources = source_data.get("files")
    if not isinstance(resources, dict):
        errors.append(f"Scenario {scenario_id}: Missing source_data.files mapping")
    else:
        missing_keys = sorted(CP30_REQUIRED_RESOURCE_KEYS - set(resources))
        if missing_keys:
            errors.append(
                f"Scenario {scenario_id}: source_data.files missing keys {missing_keys}"
            )

    proxy = (scenario_dict.get("fleet") or {}).get("spatial_proxy")
    if proxy:
        if str(proxy.get("source", "")).upper() != "FES":
            warnings.append(
                f"Scenario {scenario_id}: fleet.spatial_proxy source is not FES; "
                "confirm CP30 spatial data is available"
            )
        elif "FES_year" not in proxy or "FES_scenario" not in proxy:
            errors.append(
                f"Scenario {scenario_id}: FES fleet.spatial_proxy requires FES_year and FES_scenario"
            )
    else:
        warnings.append(
            f"Scenario {scenario_id}: No fleet.spatial_proxy configured; later stages need "
            "explicit CP30 spatial data or a declared proxy"
        )

    return (len(errors) == 0, errors, warnings)


def validate_scenario_complete(scenario_dict):
    """
    Perform complete validation of a scenario.
    
    Args:
        scenario_dict (dict): The scenario configuration
        
    Returns:
        dict: Validation result with 'errors', 'warnings', 'info' keys
    """
    errors = []
    warnings = []
    info = []
    
    # Get year (try both field names)
    year = scenario_dict.get("modelled_year") or scenario_dict.get("year")
    
    # Determine type and validate accordingly
    if is_historical_scenario(year):
        valid, hist_errors, hist_warnings = validate_historical_scenario(scenario_dict, "unknown")
        errors.extend(hist_errors)
        warnings.extend(hist_warnings)
    else:
        valid, fut_errors, fut_warnings = validate_future_scenario(scenario_dict, "unknown")
        errors.extend(fut_errors)
        warnings.extend(fut_warnings)
    
    # Common validation
    if "network_model" in scenario_dict:
        model = scenario_dict["network_model"]
        if model not in ["ETYS", "Reduced", "Zonal"]:
            errors.append(f"Invalid network_model '{model}'")
    
    return {
        "errors": errors,
        "warnings": warnings,
        "info": info
    }


def summarize_scenario_configuration(scenarios_dict_or_dict):
    """
    Generate a summary of scenario configuration(s).
    
    Args:
        scenarios_dict_or_dict: Either a dict of {scenario_id: scenario_config} or a single scenario config
        
    Returns:
        dict: Summary with 'historical', 'future', 'dukes_years_needed', 'fes_years_needed', 'cutout_years_needed'
    """
    # Handle both dict of scenarios and single scenario
    if not isinstance(scenarios_dict_or_dict, dict):
        return {}
    
    # Check if this is a dict of scenarios or a single scenario
    first_val = next(iter(scenarios_dict_or_dict.values()), None)
    
    if first_val and isinstance(first_val, dict) and "network_model" in first_val:
        # This is {scenario_id: scenario_config}
        scenarios_to_summarize = scenarios_dict_or_dict
    else:
        # Single scenario config
        return {
            "historical": [],
            "future": [],
            "dukes_years_needed": set(),
            "fes_years_needed": set(),
            "cutout_years_needed": set()
        }
    
    historical = []
    future = []
    cp30 = []
    dukes_years = set()
    fes_years = set()
    cp30_pathways = set()
    cutout_years = set()
    
    for scenario_id, config in scenarios_to_summarize.items():
        year = config.get("modelled_year") or config.get("year")
        
        scenario_source = target_source(config)

        if scenario_source == "historical":
            historical.append(scenario_id)
            if year:
                dukes_years.add(year)
                cutout_years.add(year)
        elif scenario_source == "CP30":
            cp30.append(scenario_id)
            future.append(scenario_id)
            pathway = config.get("CP30_pathway")
            if pathway:
                cp30_pathways.add(pathway)
            renewables_year = config.get("renewables_year")
            if renewables_year:
                cutout_years.add(renewables_year)
            proxy = (config.get("fleet") or {}).get("spatial_proxy") or {}
            if str(proxy.get("source", "")).upper() == "FES" and proxy.get("FES_year"):
                fes_years.add(proxy["FES_year"])
        else:
            future.append(scenario_id)
            # Future scenarios use renewables_year for weather cutouts
            renewables_year = config.get("renewables_year")
            if renewables_year:
                cutout_years.add(renewables_year)
            fes_year = config.get("FES_year", 2025)
            if fes_year:
                fes_years.add(fes_year)
    
    return {
        "historical": historical,
        "future": future,
        "cp30": cp30,
        "dukes_years_needed": list(sorted(dukes_years)),
        "fes_years_needed": list(sorted(fes_years)),
        "cp30_pathways_needed": list(sorted(cp30_pathways)),
        "cutout_years_needed": list(sorted(cutout_years))
    }


def check_cutout_availability(year):
    """
    Check if a weather cutout (Atlite cutout) is available for a given year.
    
    Args:
        year (int): The year to check
        
    Returns:
        bool: True if cutout exists, False otherwise
    """
    # Canonical workflow location, followed by the legacy locations used by
    # older checkouts and manually prepared cutouts.
    candidate_paths = (
        Path("resources/atlite/cutouts") / f"uk-{year}.nc",
        Path("resources/atlite") / f"uk-{year}.nc",
        Path("data/atlite") / f"uk-{year}.nc",
    )
    return any(path.exists() for path in candidate_paths)


def validate_all_active_scenarios(config_path, scenarios_path, check_files=True, verbose=True):
    """
    Validate all active scenarios defined in config.yaml.
    
    Args:
        config_path (Path): Path to config.yaml
        scenarios_path (Path): Path to scenarios.yaml
        check_files (bool): Whether to check for file existence
        verbose (bool): Whether to print detailed output
        
    Returns:
        dict: Validation result with 'valid' (bool) and 'summary' (dict)
    """
    
    # Load configs: the merged scenarios (defaults, overlays and entry), as the Snakefile validates them,
    # and run_scenarios with family names expanded to their months.
    try:
        from config.config_loader import load_config
        loaded = load_config(Path(config_path).parent)
    except Exception as e:
        if verbose:
            print(f"Error loading config files: {e}")
        return {'valid': False, 'summary': {'total_errors': 1}}

    active_scenarios = loaded['run_scenarios']
    scenarios_def = loaded['scenarios']

    total_errors = 0
    total_warnings = 0
    
    if verbose:
        print(f"Validating {len(active_scenarios)} active scenarios...")
        
    for scenario_id in active_scenarios:
        if scenario_id not in scenarios_def:
            if verbose:
                print(f"ERROR: Scenario '{scenario_id}' listed in config.yaml but not found in config/scenarios.yaml "
                      "or a study listed under studies in config/config.yaml")
            total_errors += 1
            continue
            
        scenario_config = scenarios_def[scenario_id]
        result = validate_scenario_complete(scenario_config)
        
        errors = result['errors']
        warnings = result['warnings']
        
        if errors:
            total_errors += len(errors)
            if verbose:
                print(f"[FAIL] Scenario '{scenario_id}': {len(errors)} errors")
                for err in errors:
                    print(f"  - {err}")
        elif verbose:
            print(f"[ok] Scenario '{scenario_id}': Valid")
            
        if warnings:
            total_warnings += len(warnings)
            if verbose:
                for warn in warnings:
                    print(f"  - Warning: {warn}")
                    
        # File checks (basic implementation)
        if check_files:
            # Check cutout. Historical scenarios use modelled_year; future
            # scenarios use renewables_year as the weather proxy year.
            if is_historical_scenario(scenario_config):
                cutout_year = scenario_config.get("modelled_year") or scenario_config.get("year")
            else:
                cutout_year = scenario_config.get("renewables_year")
            # A missing cutout is not an error: rule retrieve_cutout (rules/renewables.smk) downloads it.
            if cutout_year and not check_cutout_availability(cutout_year):
                if verbose:
                    print(f"  - Note: weather cutout for year {cutout_year} not present yet; "
                          "fetched on first run by rule retrieve_cutout")

    return {
        'valid': total_errors == 0,
        'summary': {
            'total_errors': total_errors,
            'total_warnings': total_warnings
        }
    }

