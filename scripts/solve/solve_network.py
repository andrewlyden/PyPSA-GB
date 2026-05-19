"""
Solve Network - PyPSA Optimization

This script performs optimal power flow optimization on the network
to determine least-cost dispatch, storage operation, and interconnector flows.
"""

import pypsa
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from scripts.utilities.logging_config import setup_logging
from scripts.solve.hydro_constraints import (
    build_hydro_constraints_callback,
    combine_extra_functionalities,
    log_hydro_constraint_setup,
)

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network


RENEWABLE_CARRIERS = {
    "wind",
    "Wind",
    "Wind (Onshore)",
    "Wind (Offshore)",
    "wind_onshore",
    "wind_offshore",
    "solar",
    "Solar",
    "solar_pv",
    "embedded_wind",
    "embedded_solar",
    "Hydro",
    "hydro",
    "large_hydro",
    "small_hydro",
    "tidal_stream",
    "shoreline_wave",
    "tidal_lagoon",
    "marine",
    "geothermal",
}


def get_solve_mode_from_config() -> str:
    """
    Load the solve_mode from config.yaml.
    
    Returns:
        "LP" or "MILP" - controls optimization problem type.
        Defaults to "LP" (no ramp limits, no unit commitment) if not specified.
    """
    config_path = Path("config/config.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            optimization = config.get('optimization', {})
            solve_mode = optimization.get('solve_mode', 'LP')
            return solve_mode.upper()
        except Exception:
            return 'LP'
    return 'LP'

def configure_solver(network, solver_name, solver_options, logger):
    """
    Configure optimization solver settings.
    
    Parameters
    ----------
    network : pypsa.Network
        Network to optimize
    solver_name : str
        Solver to use ('gurobi', 'highs', 'glpk', 'cplex')
    solver_options : dict
        Solver-specific options
    logger : logging.Logger
        Logger instance
    """
    logger.info(f"Configuring solver: {solver_name}")
    logger.info(f"Solver options: {solver_options}")
    
    # Validate solver availability
    available_solvers = ['gurobi', 'highs', 'glpk', 'cplex']
    if solver_name not in available_solvers:
        logger.warning(f"Unknown solver '{solver_name}', falling back to 'highs'")
        solver_name = 'highs'
    
    return solver_name, solver_options


def validate_network_costs(network, logger):
    """
    Validate network marginal costs before solving.
    
    This function checks for common issues that cause unbounded optimization:
    1. Load shedding generators with zero cost (should have VOLL ~£6000/MWh)
    2. Thermal generators with zero cost (should have fuel + carbon costs)
    
    Parameters
    ----------
    network : pypsa.Network
        Network to validate
    logger : logging.Logger
        Logger instance
        
    Raises
    ------
    ValueError
        If critical cost issues are detected that would cause unbounded optimization
    """
    logger.info("=" * 80)
    logger.info("VALIDATING NETWORK MARGINAL COSTS")
    logger.info("=" * 80)
    
    issues = []
    warnings = []
    
    # Check 1: Load shedding must have high cost (VOLL)
    load_shedding_carriers = ['load_shedding', 'load shedding', 'voll', 'VOLL']
    load_shedding_mask = network.generators['carrier'].isin(load_shedding_carriers)
    load_shedding_gens = network.generators[load_shedding_mask]
    
    if len(load_shedding_gens) > 0:
        zero_cost_ls = load_shedding_gens[load_shedding_gens['marginal_cost'] == 0]
        if len(zero_cost_ls) > 0:
            issues.append(
                f"CRITICAL: {len(zero_cost_ls)} load shedding generators have ZERO cost!\n"
                f"  This allows the LP to shed all demand for free → unbounded optimization.\n"
                f"  Load shedding MUST have high cost (VOLL ~£6000/MWh)."
            )
        else:
            min_voll = load_shedding_gens['marginal_cost'].min()
            max_voll = load_shedding_gens['marginal_cost'].max()
            logger.info(f"[OK] Load shedding: {len(load_shedding_gens)} units @ £{min_voll:,.0f}-£{max_voll:,.0f}/MWh")
    else:
        warnings.append("No load shedding generators found - network may be infeasible if supply < demand")
    
    # Check 2: Thermal generators should have positive marginal cost
    thermal_carriers = ['CCGT', 'OCGT', 'Coal', 'Oil', 'gas', 'coal', 'oil',
                       'Conventional Steam', 'Conventional steam', 'AGR', 'PWR',
                       'Bioenergy', 'biomass', 'Biomass']
    thermal_mask = network.generators['carrier'].isin(thermal_carriers)
    thermal_gens = network.generators[thermal_mask]
    
    if len(thermal_gens) > 0:
        zero_cost_thermal = thermal_gens[thermal_gens['marginal_cost'] == 0]
        if len(zero_cost_thermal) > 0:
            total_capacity = zero_cost_thermal['p_nom'].sum()
            carriers_affected = zero_cost_thermal['carrier'].unique()
            issues.append(
                f"CRITICAL: {len(zero_cost_thermal)} thermal generators have ZERO cost ({total_capacity:,.0f} MW)!\n"
                f"  Carriers: {', '.join(carriers_affected)}\n"
                f"  Thermal units should have fuel + carbon costs (typically £50-150/MWh).\n"
                f"  With zero-cost generation + zero-cost load shedding → unbounded optimization."
            )
        else:
            mean_mc = thermal_gens['marginal_cost'].mean()
            min_mc = thermal_gens['marginal_cost'].min()
            max_mc = thermal_gens['marginal_cost'].max()
            logger.info(f"[OK] Thermal generators: {len(thermal_gens)} units, MC: £{mean_mc:.2f}/MWh (£{min_mc:.2f}-£{max_mc:.2f})")
    
    # Check 3: Overall marginal cost distribution
    total_gens = len(network.generators)
    zero_cost_gens = (network.generators['marginal_cost'] == 0).sum()
    positive_cost_gens = (network.generators['marginal_cost'] > 0).sum()
    
    logger.info(f"\nMarginal cost distribution:")
    logger.info(f"  Total generators: {total_gens}")
    logger.info(f"  Zero cost: {zero_cost_gens} ({100*zero_cost_gens/total_gens:.1f}%)")
    logger.info(f"  Positive cost: {positive_cost_gens} ({100*positive_cost_gens/total_gens:.1f}%)")
    
    # If too many zero-cost generators (excluding renewables), warn
    non_renewable_mask = ~network.generators["carrier"].isin(RENEWABLE_CARRIERS)
    zero_cost_mask = network.generators["marginal_cost"] == 0

    zero_cost_fixed_imports = network.generators[
        (network.generators["carrier"] == "EU_import") & zero_cost_mask
    ]
    suspicious_zero_cost = network.generators[
        non_renewable_mask
        & zero_cost_mask
        & (network.generators["carrier"] != "EU_import")
    ]

    if len(zero_cost_fixed_imports) > 0:
        capacity = zero_cost_fixed_imports["p_nom"].sum()
        logger.info(
            "[OK] Fixed-flow interconnector supply: "
            f"{len(zero_cost_fixed_imports)} EU_import generators at zero cost "
            f"({capacity:,.0f} MW). This is expected for historical fixed-link flows."
        )

    if len(suspicious_zero_cost) > 0:
        capacity = suspicious_zero_cost["p_nom"].sum()
        if capacity > 1000:  # More than 1 GW
            carrier_summary = (
                suspicious_zero_cost.groupby("carrier")["p_nom"]
                .sum()
                .sort_values(ascending=False)
            )
            carrier_text = ", ".join(
                f"{carrier} ({cap:,.0f} MW)"
                for carrier, cap in carrier_summary.items()
            )
            warnings.append(
                f"{len(suspicious_zero_cost)} non-renewable generators have zero cost "
                f"({capacity:,.0f} MW)\n"
                f"  Carriers: {carrier_text}\n"
                f"  This may indicate missing fuel price mappings."
            )
    
    # Report issues
    if issues:
        logger.error("\n" + "!" * 80)
        logger.error("CRITICAL COST VALIDATION ERRORS DETECTED:")
        logger.error("!" * 80)
        for i, issue in enumerate(issues, 1):
            logger.error(f"\n{i}. {issue}")
        logger.error("\n" + "!" * 80)
        logger.error("FIX REQUIRED: Re-run apply_marginal_costs.py with corrected mappings")
        logger.error("!" * 80)
        raise ValueError("Network has critical marginal cost issues that would cause unbounded optimization")
    
    if warnings:
        logger.warning("\n⚠️  WARNINGS:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    logger.info("=" * 80)
    logger.info("[OK] NETWORK COST VALIDATION PASSED")
    logger.info("=" * 80)


def apply_load_shedding_limits(network, logger):
    """
    Cap load-shedding generators to same-bus demand at each snapshot.

    Load shedding is modelled as high-cost generation so the LP remains
    feasible, but it should behave like local demand interruption. Without a
    time-varying cap these units can produce more than the local load and
    export through the network, which turns them into emergency generators.
    """
    if len(network.generators) == 0 or len(network.snapshots) == 0:
        return False

    load_shedding_carriers = ["load_shedding", "load shedding", "voll", "VOLL"]
    ls_mask = network.generators["carrier"].isin(load_shedding_carriers)
    ls_gens = network.generators[ls_mask]

    if ls_gens.empty:
        return False

    pmax = network.generators_t.p_max_pu.copy()
    if pmax.empty:
        pmax = pd.DataFrame(index=network.snapshots)
    else:
        pmax = pmax.reindex(index=network.snapshots)

    if network.loads.empty or network.loads_t.p_set.empty:
        logger.warning(
            "Load shedding demand caps skipped: no load time series available"
        )
        return False

    load_p_set = network.loads_t.p_set.reindex(index=network.snapshots).fillna(0.0)
    no_load_bus_count = 0

    for gen_name, gen in ls_gens.iterrows():
        bus = gen["bus"]
        p_nom = float(gen.get("p_nom", 0.0) or 0.0)

        if p_nom <= 0:
            pmax[gen_name] = 0.0
            continue

        load_names = network.loads.index[network.loads["bus"] == bus]
        if len(load_names) == 0:
            local_load = pd.Series(0.0, index=network.snapshots)
            no_load_bus_count += 1
        else:
            local_load = (
                load_p_set.reindex(columns=load_names)
                .fillna(0.0)
                .sum(axis=1)
                .clip(lower=0.0)
            )

        pmax[gen_name] = (local_load / p_nom).clip(lower=0.0, upper=1.0)

    network.generators_t.p_max_pu = pmax

    ls_caps = (
        pmax.reindex(columns=ls_gens.index)
        .fillna(0.0)
        .mul(ls_gens["p_nom"], axis=1)
        .sum(axis=1)
    )
    static_capacity = float(ls_gens["p_nom"].sum())
    logger.info(
        "Applied load shedding demand caps: "
        f"{len(ls_gens)} generators, static capacity {static_capacity:,.0f} MW, "
        f"hourly cap max {ls_caps.max():,.0f} MW, mean {ls_caps.mean():,.0f} MW"
    )
    if no_load_bus_count:
        logger.info(
            f"Load shedding capped to zero at {no_load_bus_count} buses with no load"
        )

    return True


def apply_transmission_relaxation(network, scenario_config, logger):
    """
    Apply transmission constraint relaxation for feasibility.
    
    For detailed networks (like ETYS), actual line ratings may be too restrictive
    for the power flow to be feasible. This function applies:
    1. Minimum line capacity floor
    2. Minimum transformer capacity floor  
    3. Capacity scaling factor
    
    Parameters
    ----------
    network : pypsa.Network
        Network to modify
    scenario_config : dict
        Scenario configuration (may contain 'transmission' key)
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    bool
        True if any modifications were made
    """
    # Load global defaults
    defaults_path = Path("config/defaults.yaml")
    global_transmission = {}
    if defaults_path.exists():
        try:
            with open(defaults_path, 'r', encoding='utf-8') as f:
                defaults = yaml.safe_load(f)
            global_transmission = defaults.get('transmission', {})
        except Exception as e:
            logger.warning(f"Could not load defaults.yaml: {e}")
    
    # Scenario config overrides global defaults
    transmission = scenario_config.get('transmission', global_transmission)
    
    min_line_s_nom = transmission.get('min_line_s_nom', 0)
    min_transformer_s_nom = transmission.get('min_transformer_s_nom', 0)
    capacity_scale = transmission.get('capacity_scale', 1.0)
    
    # Skip if no modifications needed
    if min_line_s_nom == 0 and min_transformer_s_nom == 0 and capacity_scale == 1.0:
        return False
    
    logger.info("=" * 80)
    logger.info("APPLYING TRANSMISSION CONSTRAINT RELAXATION")
    logger.info("=" * 80)
    
    modifications = False
    
    # Apply minimum line capacity
    if min_line_s_nom > 0 and len(network.lines) > 0:
        low_cap_lines = network.lines['s_nom'] < min_line_s_nom
        n_modified = low_cap_lines.sum()
        if n_modified > 0:
            network.lines.loc[low_cap_lines, 's_nom'] = min_line_s_nom
            logger.info(f"Lines: Set minimum s_nom={min_line_s_nom} MVA for {n_modified} lines")
            modifications = True
    
    # Apply minimum transformer capacity
    if min_transformer_s_nom > 0 and len(network.transformers) > 0:
        low_cap_trafos = network.transformers['s_nom'] < min_transformer_s_nom
        n_modified = low_cap_trafos.sum()
        if n_modified > 0:
            network.transformers.loc[low_cap_trafos, 's_nom'] = min_transformer_s_nom
            logger.info(f"Transformers: Set minimum s_nom={min_transformer_s_nom} MVA for {n_modified} transformers")
            modifications = True
    
    # Apply capacity scaling
    if capacity_scale != 1.0:
        if len(network.lines) > 0:
            network.lines['s_nom'] = network.lines['s_nom'] * capacity_scale
            logger.info(f"Lines: Scaled all s_nom by {capacity_scale}x")
            modifications = True
        if len(network.transformers) > 0:
            network.transformers['s_nom'] = network.transformers['s_nom'] * capacity_scale
            logger.info(f"Transformers: Scaled all s_nom by {capacity_scale}x")
            modifications = True
    
    if modifications:
        logger.info("Transmission constraints relaxed for feasibility")
        logger.info(f"  Line s_nom range: {network.lines.s_nom.min():.0f} - {network.lines.s_nom.max():.0f} MVA")
        if len(network.transformers) > 0:
            logger.info(f"  Transformer s_nom range: {network.transformers.s_nom.min():.0f} - {network.transformers.s_nom.max():.0f} MVA")
    
    logger.info("=" * 80)
    return modifications


def apply_line_rating_overrides(network, scenario_config, logger):
    """Apply per-line s_nom overrides and voltage floor from config.

    Two mechanisms, applied in order:

    1. min_bm_constraint_voltage_kv — set s_nom = 9999 MVA for all lines/transformers
       with v_nom strictly below this threshold.  Removes sub-transmission loop-flow
       artefacts caused by the DC power flow distributing currents through incomplete
       132 kV ring topologies in ETYS.  Set to 0 to disable.

    2. line_rating_overrides — explicit per-component s_nom values.  Applied after
       the voltage floor so targeted corrections always take precedence.
    """
    # Load global defaults
    defaults_path = Path("config/defaults.yaml")
    global_transmission = {}
    if defaults_path.exists():
        try:
            with open(defaults_path, 'r', encoding='utf-8') as f:
                defaults = yaml.safe_load(f)
            global_transmission = defaults.get('transmission', {})
        except Exception:
            pass

    transmission = scenario_config.get('transmission', global_transmission)

    logger.info("=" * 80)
    logger.info("APPLYING LINE RATING OVERRIDES")
    logger.info("=" * 80)

    applied = 0

    # ── 1. Voltage floor ─────────────────────────────────────────────────────
    min_v = float(transmission.get('min_bm_constraint_voltage_kv', 0))
    if min_v > 0:
        # Only applies to lines — transformers span two voltage levels so v_nom
        # is ambiguous, and transformer constraints are not the source of loop-flow
        # artefacts in the Highland 132 kV network.
        low_lines = network.lines[network.lines['v_nom'] < min_v]
        if len(low_lines) > 0:
            network.lines.loc[low_lines.index, 's_nom'] = 9999.0
            v_counts = low_lines['v_nom'].value_counts().sort_index()
            logger.info(
                f"  Voltage floor {min_v:.0f} kV: relaxed {len(low_lines)} lines "
                f"({v_counts.to_dict()})"
            )
            applied += len(low_lines)
        else:
            logger.info(f"  Voltage floor {min_v:.0f} kV: no lines below threshold")
    else:
        logger.info("  Voltage floor disabled (min_bm_constraint_voltage_kv=0)")

    # ── 2. Explicit per-line overrides ───────────────────────────────────────
    overrides = transmission.get('line_rating_overrides', {}) or {}
    if not isinstance(overrides, dict):
        logger.warning(
            "  Ignoring line_rating_overrides because it is not a mapping "
            f"({type(overrides).__name__})"
        )
        overrides = {}
    for line_id, new_s_nom in overrides.items():
        new_s_nom = float(new_s_nom)
        if line_id in network.lines.index:
            old_val = network.lines.at[line_id, 's_nom']
            network.lines.at[line_id, 's_nom'] = new_s_nom
            logger.info(f"  {line_id}: s_nom {old_val:.0f} → {new_s_nom:.0f} MVA")
            applied += 1
        elif line_id in network.transformers.index:
            old_val = network.transformers.at[line_id, 's_nom']
            network.transformers.at[line_id, 's_nom'] = new_s_nom
            logger.info(f"  {line_id} (transformer): s_nom {old_val:.0f} → {new_s_nom:.0f} MVA")
            applied += 1
        else:
            logger.warning(f"  {line_id}: not found in network lines or transformers")

    if applied:
        logger.info(f"Applied {applied} total line/transformer rating changes")
    return applied > 0


def apply_outage_schedule(network, scenario_config, logger):
    """Apply time-varying transmission outage schedule via s_max_pu.

    Supports three modes configured under ``transmission.outage_schedule``:

    ``csv``
        Load from a CSV file with columns:
        component, component_id, start, end, s_max_pu

    ``synthetic``
        Generate stochastic maintenance outages from parameters
        (voltage-dependent forced outage rates, seasonal weighting).

    ``neso``
        Apply NESO day-ahead boundary limits, with configurable gap filling
        when the published source has missing timestamps.

    The function sets ``network.lines_t.s_max_pu`` (and
    ``network.transformers_t.s_max_pu``) so that PyPSA reduces the effective
    line rating at each snapshot.  Values default to 1.0 (fully available).

    Parameters
    ----------
    network : pypsa.Network
    scenario_config : dict
    logger : logging.Logger

    Returns
    -------
    bool  — True if any outages were applied.
    """
    defaults_path = Path("config/defaults.yaml")
    global_transmission = {}
    if defaults_path.exists():
        try:
            with open(defaults_path, 'r', encoding='utf-8') as f:
                defaults = yaml.safe_load(f)
            global_transmission = defaults.get('transmission', {})
        except Exception:
            pass

    transmission = scenario_config.get('transmission', global_transmission)
    outage_cfg = transmission.get('outage_schedule', {})

    if not outage_cfg.get('enabled', False):
        return False

    source = outage_cfg.get('source', 'csv')
    if not hasattr(network, 'meta') or network.meta is None:
        network.meta = {}
    if source == 'neso':
        network.meta['neso_constraint_mode'] = _get_neso_constraint_mode(outage_cfg)
    logger.info("=" * 80)
    logger.info(f"APPLYING OUTAGE SCHEDULE (source={source})")
    logger.info("=" * 80)

    snapshots = network.snapshots
    solve_period_cfg = scenario_config.get('solve_period', {})
    if solve_period_cfg.get('enabled', False) and 'start' in solve_period_cfg and 'end' in solve_period_cfg:
        solve_start = pd.Timestamp(solve_period_cfg['start'])
        solve_end = pd.Timestamp(solve_period_cfg['end'])
        mask = (snapshots >= solve_start) & (snapshots <= solve_end)
        active_snapshots = snapshots[mask]
        if len(active_snapshots) > 0:
            logger.info(
                f"  Outage schedule using active solve-period snapshots: "
                f"{active_snapshots[0]} to {active_snapshots[-1]} ({len(active_snapshots)} snapshots)"
            )
            snapshots = active_snapshots

    if source == 'csv':
        return _apply_csv_outage_schedule(network, outage_cfg, snapshots, logger)
    elif source == 'synthetic':
        return _apply_synthetic_outage_schedule(network, outage_cfg, snapshots, logger)
    elif source == 'neso':
        constraint_mode = _get_neso_constraint_mode(outage_cfg)
        if constraint_mode == 'aggregate_boundary':
            logger.info(
                "  NESO constraint_mode=aggregate_boundary: limits will be enforced "
                "inside the optimization model, not via line s_max_pu derating"
            )
            return False
        return _apply_neso_boundary_limits(network, outage_cfg, snapshots, logger)
    else:
        logger.warning(f"Unknown outage_schedule source '{source}', skipping")
        return False


def _inspect_neso_boundary_coverage(bnd_df, snapshots, tolerance_minutes=45):
    """Summarise how well a NESO boundary DataFrame covers requested snapshots."""
    if bnd_df.empty:
        return {
            'matched_snapshots': 0,
            'missing_snapshots': len(snapshots),
            'missing_days': sorted({ts.date() for ts in snapshots}),
            'available_start': None,
            'available_end': None,
        }

    ts_index = pd.DatetimeIndex(pd.to_datetime(bnd_df['date'])).sort_values()
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    nearest_idx = ts_index.get_indexer(snapshots, method='nearest', tolerance=tolerance)
    missing_mask = nearest_idx == -1
    missing_days = sorted({ts.date() for ts in snapshots[missing_mask]})
    return {
        'matched_snapshots': int((~missing_mask).sum()),
        'missing_snapshots': int(missing_mask.sum()),
        'missing_days': missing_days,
        'available_start': ts_index.min(),
        'available_end': ts_index.max(),
    }


def _load_cached_boundary_file(cache_dir, boundary_name, date_tag, snapshots):
    """Load the best cached NESO boundary file for the requested snapshot window."""
    exact_file = cache_dir / f"neso_{boundary_name}_{date_tag}.csv"
    candidates = []
    if exact_file.exists():
        candidates.append(exact_file)

    for path in sorted(cache_dir.glob(f"neso_{boundary_name}_*.csv")):
        if path != exact_file:
            candidates.append(path)

    best_df = None
    best_coverage = None
    best_path = None
    for cache_file in candidates:
        bnd_df = pd.read_csv(cache_file, parse_dates=['date'])
        coverage = _inspect_neso_boundary_coverage(bnd_df, snapshots)
        if best_coverage is None or coverage['matched_snapshots'] > best_coverage['matched_snapshots']:
            best_df = bnd_df
            best_coverage = coverage
            best_path = cache_file
        if coverage['missing_snapshots'] == 0:
            return bnd_df, coverage, cache_file

    return best_df, best_coverage, best_path


def _fill_missing_neso_limits(limit_series, bnd_data, gap_fill_mode, total_s_nom, logger, boundary_name):
    """Fill missing NESO boundary limits according to the configured gap policy."""
    missing_mask = limit_series.isna()
    if not missing_mask.any():
        return limit_series

    missing_snapshots = limit_series.index[missing_mask]
    missing_days = sorted({ts.date() for ts in missing_snapshots})

    if gap_fill_mode == 'interpolate':
        filled = limit_series.astype(float).interpolate(method='time').ffill().bfill()
        if filled.isna().any():
            raise RuntimeError(
                f"{boundary_name}: unable to interpolate NESO limits for all missing snapshots"
            )
        logger.warning(
            f"  {boundary_name}: filled {missing_mask.sum()} missing snapshots "
            f"(days: {missing_days}) using interpolated nearby NESO limits"
        )
        return filled

    if gap_fill_mode == 'nearest_available':
        nearest_idx = bnd_data.index.get_indexer(missing_snapshots, method='nearest')
        if (nearest_idx < 0).any():
            raise RuntimeError(
                f"{boundary_name}: unable to find nearby NESO limits for all missing snapshots"
            )
        source_times = bnd_data.index[nearest_idx]
        filled = limit_series.copy()
        filled.loc[missing_snapshots] = bnd_data.iloc[nearest_idx]['limit_mw'].to_numpy()
        source_days = sorted({ts.date() for ts in source_times})
        logger.warning(
            f"  {boundary_name}: filled {missing_mask.sum()} missing snapshots "
            f"(days: {missing_days}) using nearest available NESO limits "
            f"from {source_days}"
        )
        return filled

    if gap_fill_mode == 'unconstrained':
        logger.warning(
            f"  {boundary_name}: no nearby NESO limit for {missing_mask.sum()} snapshots "
            f"(days: {missing_days}); filling with unconstrained total_s_nom={total_s_nom:.0f} MW"
        )
        return limit_series.fillna(total_s_nom)

    if gap_fill_mode == 'fail':
        raise RuntimeError(
            f"{boundary_name}: NESO limit data missing for {missing_mask.sum()} snapshots "
            f"(days: {missing_days}) and gap_fill_mode='fail'"
        )

    raise ValueError(f"Unknown NESO gap_fill_mode '{gap_fill_mode}'")


def _get_neso_constraint_mode(outage_cfg):
    """Return the configured NESO boundary enforcement mode."""
    neso_cfg = outage_cfg.get('neso', {})
    return str(neso_cfg.get('constraint_mode', 'uniform_s_max_pu')).strip().lower()


def _load_neso_boundary_inputs(outage_cfg, snapshots, logger):
    """Load mapping plus cached/fetched NESO boundary data for the requested snapshots."""
    import requests

    neso_cfg = outage_cfg.get('neso', {})
    mapping_path = Path(neso_cfg.get(
        'boundary_mapping', 'data/network/neso_boundary_mapping.yaml'
    ))
    resource_id = neso_cfg.get(
        'resource_id', '38a18ec1-9e40-465d-93fb-301e80fd1352'
    )
    api_base = neso_cfg.get(
        'api_base', 'https://api.neso.energy/api/3/action/datastore_search_sql'
    )
    cache_dir = Path(neso_cfg.get('cache_dir', 'resources/neso_cache'))
    min_s_max_pu = float(neso_cfg.get('min_s_max_pu', 0.05))
    tolerance_minutes = int(neso_cfg.get('nearest_tolerance_minutes', 45))
    gap_fill_mode = str(neso_cfg.get('gap_fill_mode', 'interpolate')).strip().lower()

    if not mapping_path.exists():
        logger.error(f"Boundary mapping file not found: {mapping_path}")
        return None

    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = yaml.safe_load(f)

    raw_boundaries = mapping.get('boundaries', {})
    boundary_include = neso_cfg.get('boundary_include')
    if boundary_include:
        include_set = {str(name).strip() for name in boundary_include}
        boundaries = {}
        for name, info in raw_boundaries.items():
            if name in include_set:
                boundary_info = dict(info)
                boundary_info.setdefault('neso_boundary', name)
                boundaries[name] = boundary_info
            for sub_name, sub_info in (info.get('subconstraints') or {}).items():
                flat_name = f"{name}::{sub_name}"
                if flat_name in include_set:
                    boundary_info = dict(sub_info)
                    boundary_info['neso_boundary'] = name
                    boundary_info['parent_boundary'] = name
                    boundaries[flat_name] = boundary_info
    else:
        # Subconstraints are diagnostic variants. Only enforce them when a
        # scenario explicitly asks for the flattened NAME::SUBCONSTRAINT key.
        boundaries = {}
        for name, info in raw_boundaries.items():
            boundary_info = dict(info)
            boundary_info.setdefault('neso_boundary', name)
            boundaries[name] = boundary_info
    if not boundaries:
        logger.warning("No boundaries defined in mapping file")
        return None

    start_date = snapshots[0].strftime('%Y-%m-%dT%H:%M:%S')
    end_date = (snapshots[-1] + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')

    cache_dir.mkdir(parents=True, exist_ok=True)
    date_tag = f"{snapshots[0].strftime('%Y%m%d')}_{snapshots[-1].strftime('%Y%m%d')}"

    all_boundary_dfs = []
    fetched_boundary_names = set()
    for boundary_name, boundary_def in boundaries.items():
        neso_boundary_name = str(boundary_def.get('neso_boundary', boundary_name))
        if neso_boundary_name in fetched_boundary_names:
            continue
        fetched_boundary_names.add(neso_boundary_name)

        cache_file = cache_dir / f"neso_{neso_boundary_name}_{date_tag}.csv"
        bnd_df, coverage, loaded_from = _load_cached_boundary_file(
            cache_dir, neso_boundary_name, date_tag, snapshots
        )

        if bnd_df is not None and coverage and coverage['missing_snapshots'] == 0:
            logger.debug(
                f"  {neso_boundary_name}: loaded {len(bnd_df)} cached records "
                f"from {loaded_from.name}"
            )
        else:
            if bnd_df is not None and coverage and loaded_from is not None:
                logger.warning(
                    f"  {neso_boundary_name}: cached file {loaded_from.name} is incomplete "
                    f"for requested window ({coverage['matched_snapshots']}/{len(snapshots)} "
                    f"snapshots matched; missing days: {coverage['missing_days']}) - refreshing"
                )
            sql = (
                f'SELECT "Constraint Group", "Date (GMT/BST)", '
                f'"Limit (MW)", "Flow (MW)" '
                f'FROM "{resource_id}" '
                f"WHERE \"Date (GMT/BST)\" >= '{start_date}' "
                f"AND \"Date (GMT/BST)\" < '{end_date}' "
                f"AND \"Constraint Group\" = '{neso_boundary_name}' "
                f'ORDER BY "Date (GMT/BST)" LIMIT 32000'
            )
            try:
                resp = requests.get(api_base, params={'sql': sql}, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                if not result.get('success'):
                    logger.warning(f"  {neso_boundary_name}: API error, skipping")
                    continue
                records = result['result']['records']
            except Exception as e:
                logger.warning(f"  {neso_boundary_name}: fetch failed ({e}), skipping")
                continue

            if not records:
                logger.debug(f"  {neso_boundary_name}: no NESO records for this period")
                if bnd_df is None:
                    continue
            else:
                bnd_df = pd.DataFrame(records)
                bnd_df.rename(columns={
                    'Constraint Group': 'boundary',
                    'Date (GMT/BST)': 'date',
                    'Limit (MW)': 'limit_mw',
                    'Flow (MW)': 'flow_mw',
                }, inplace=True)
                bnd_df['date'] = pd.to_datetime(bnd_df['date'])
                bnd_df['limit_mw'] = pd.to_numeric(bnd_df['limit_mw'])
                bnd_df['flow_mw'] = pd.to_numeric(bnd_df['flow_mw'])
                bnd_df.to_csv(cache_file, index=False)
                logger.info(f"  {neso_boundary_name}: fetched {len(bnd_df)} records from NESO API")

            coverage = _inspect_neso_boundary_coverage(
                bnd_df, snapshots, tolerance_minutes=tolerance_minutes
            )
            if coverage['missing_snapshots'] > 0:
                logger.warning(
                    f"  {neso_boundary_name}: NESO data still missing "
                    f"{coverage['missing_snapshots']}/{len(snapshots)} requested snapshots "
                    f"after refresh (days: {coverage['missing_days']}). "
                    f"Applying gap_fill_mode='{gap_fill_mode}'."
                )

        all_boundary_dfs.append(bnd_df)

    if not all_boundary_dfs:
        logger.warning("No NESO boundary data retrieved for any boundary")
        return None

    neso_df = pd.concat(all_boundary_dfs, ignore_index=True)
    logger.info(
        f"  Total NESO records: {len(neso_df)} across "
        f"{neso_df['boundary'].nunique()} boundaries"
    )

    return {
        'boundaries': boundaries,
        'neso_df': neso_df,
        'tolerance_minutes': tolerance_minutes,
        'gap_fill_mode': gap_fill_mode,
        'min_s_max_pu': min_s_max_pu,
    }


def _align_neso_boundary_limit_series(
    bnd_data, snapshots, tolerance_minutes, gap_fill_mode, fallback_limit, logger, boundary_name
):
    """Align a boundary's NESO limit time series to model snapshots."""
    bnd_data = bnd_data.set_index('date').sort_index()
    bnd_data['limit_mw'] = pd.to_numeric(bnd_data['limit_mw'], errors='coerce')
    negative_limits = bnd_data['limit_mw'] < 0
    if negative_limits.any():
        logger.warning(
            f"  {boundary_name}: interpreting {int(negative_limits.sum())} signed "
            "negative NESO limit values as transfer-capacity magnitudes"
        )
        bnd_data['limit_mw'] = bnd_data['limit_mw'].abs()

    limit_series = pd.Series(dtype=float, index=snapshots)
    nearest_idx = bnd_data.index.get_indexer(
        snapshots,
        method='nearest',
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    for i, ts in enumerate(snapshots):
        idx = nearest_idx[i]
        if 0 <= idx < len(bnd_data):
            limit_series.iloc[i] = bnd_data.iloc[idx]['limit_mw']

    return _fill_missing_neso_limits(
        limit_series=limit_series,
        bnd_data=bnd_data,
        gap_fill_mode=gap_fill_mode,
        total_s_nom=fallback_limit,
        logger=logger,
        boundary_name=boundary_name,
    )


def _build_neso_boundary_constraints_callback(network, scenario_config, logger):
    """Build an extra_functionality callback enforcing NESO limits as aggregate boundary constraints."""
    defaults_path = Path("config/defaults.yaml")
    global_transmission = {}
    if defaults_path.exists():
        try:
            with open(defaults_path, 'r', encoding='utf-8') as f:
                defaults = yaml.safe_load(f)
            global_transmission = defaults.get('transmission', {})
        except Exception:
            pass

    transmission = scenario_config.get('transmission', global_transmission)
    outage_cfg = transmission.get('outage_schedule', {})
    if not outage_cfg.get('enabled', False) or outage_cfg.get('source', 'csv') != 'neso':
        return None

    constraint_mode = _get_neso_constraint_mode(outage_cfg)
    if constraint_mode != 'aggregate_boundary':
        return None

    snapshots = network.snapshots
    neso_inputs = _load_neso_boundary_inputs(outage_cfg, snapshots, logger)
    if not neso_inputs:
        return None

    boundaries = neso_inputs['boundaries']
    neso_df = neso_inputs['neso_df']
    tolerance_minutes = neso_inputs['tolerance_minutes']
    gap_fill_mode = neso_inputs['gap_fill_mode']
    neso_cfg = outage_cfg.get('neso', {}) or {}
    soft_boundary_cfg = neso_cfg.get('soft_boundaries', {}) or {}
    if not isinstance(soft_boundary_cfg, dict):
        soft_boundary_cfg = {}

    specs = []
    for boundary_name, boundary_def in boundaries.items():
        neso_boundary_name = str(boundary_def.get('neso_boundary', boundary_name))
        line_ids = list(dict.fromkeys(
            boundary_def.get('lines') or boundary_def.get('constraint_lines', [])
        ))
        flow_groups = boundary_def.get('flow_groups', {}) or {}
        positive = list(dict.fromkeys(flow_groups.get('positive', []) or line_ids))
        negative = list(dict.fromkeys(flow_groups.get('negative', []) or []))
        if not positive and not negative:
            positive = line_ids.copy()

        positive = [lid for lid in positive if lid in network.lines.index]
        negative = [lid for lid in negative if lid in network.lines.index]

        transformer_ids = list(dict.fromkeys(boundary_def.get('transformers', []) or []))
        transformer_groups = boundary_def.get('transformer_flow_groups', {}) or {}
        positive_transformers = list(dict.fromkeys(
            transformer_groups.get('positive', []) or transformer_ids
        ))
        negative_transformers = list(dict.fromkeys(
            transformer_groups.get('negative', []) or []
        ))
        if transformer_ids and not positive_transformers and not negative_transformers:
            positive_transformers = transformer_ids.copy()

        positive_transformers = [
            tid for tid in positive_transformers if tid in network.transformers.index
        ]
        negative_transformers = [
            tid for tid in negative_transformers if tid in network.transformers.index
        ]

        link_entries = []
        for entry in boundary_def.get('links', []) or []:
            if isinstance(entry, dict):
                name = entry.get('name')
                sign = float(entry.get('sign', 1.0))
            else:
                name = entry
                sign = 1.0
            if name in network.links.index:
                link_entries.append({'name': name, 'sign': sign})

        if (
            not positive
            and not negative
            and not positive_transformers
            and not negative_transformers
            and not link_entries
        ):
            logger.debug(
                f"  {boundary_name}: no matching lines, transformers, or links "
                f"for aggregate constraint"
            )
            continue

        bnd_data = neso_df[neso_df['boundary'] == neso_boundary_name].copy()
        if bnd_data.empty:
            logger.debug(
                f"  {boundary_name}: no NESO data found for aggregate constraint "
                f"(source {neso_boundary_name})"
            )
            continue

        fallback_limit = 0.0
        used_line_ids = list(dict.fromkeys(positive + negative))
        used_transformer_ids = list(dict.fromkeys(
            positive_transformers + negative_transformers
        ))
        if used_line_ids:
            fallback_limit += float(network.lines.loc[used_line_ids, 's_nom'].sum())
        if used_transformer_ids:
            fallback_limit += float(
                network.transformers.loc[used_transformer_ids, 's_nom'].sum()
            )
        if link_entries:
            link_names = [entry['name'] for entry in link_entries]
            if 'p_nom' in network.links.columns:
                fallback_limit += float(network.links.loc[link_names, 'p_nom'].abs().sum())
        fallback_limit = max(fallback_limit, 1.0)

        limit_series = _align_neso_boundary_limit_series(
            bnd_data=bnd_data,
            snapshots=snapshots,
            tolerance_minutes=tolerance_minutes,
            gap_fill_mode=gap_fill_mode,
            fallback_limit=fallback_limit,
            logger=logger,
            boundary_name=neso_boundary_name,
        )

        soft_cfg = None
        for cfg_key in (boundary_name, neso_boundary_name):
            candidate = soft_boundary_cfg.get(cfg_key)
            if isinstance(candidate, dict) and candidate.get('enabled', True):
                soft_cfg = candidate
                break

        soft_penalty = None
        soft_max_violation = None
        if soft_cfg is not None:
            soft_penalty = float(soft_cfg.get('penalty_gbp_per_mwh', 1000.0))
            soft_max_violation = soft_cfg.get('max_violation_mw')
            if soft_max_violation is None:
                soft_max_violation = np.inf
            else:
                soft_max_violation = max(0.0, float(soft_max_violation))

        logger.info(
            f"  {boundary_name}: aggregate boundary constraint prepared with "
            f"{len(used_line_ids)} lines + {len(used_transformer_ids)} transformers "
            f"+ {len(link_entries)} links, "
            f"NESO source={neso_boundary_name}, "
            f"limit={limit_series.min():.0f}-{limit_series.max():.0f} MW "
            f"(avg {limit_series.mean():.0f})"
            + (
                f", soft slack penalty={soft_penalty:.0f} GBP/MWh, "
                f"max_violation={soft_max_violation:.0f} MW"
                if soft_cfg is not None and np.isfinite(soft_max_violation)
                else (
                    f", soft slack penalty={soft_penalty:.0f} GBP/MWh, "
                    "max_violation=unbounded"
                    if soft_cfg is not None
                    else ""
                )
            )
        )

        specs.append({
            'boundary_name': boundary_name,
            'positive_lines': positive,
            'negative_lines': negative,
            'positive_transformers': positive_transformers,
            'negative_transformers': negative_transformers,
            'links': link_entries,
            'limit_series': limit_series,
            'soft_penalty_gbp_per_mwh': soft_penalty,
            'soft_max_violation_mw': soft_max_violation,
        })

    if not specs:
        logger.warning("No aggregate NESO boundary constraints were prepared")
        return None

    def boundary_extra_functionality(n, snapshots):
        import xarray as xr

        model = n.model
        line_var = model.variables["Line-s"] if "Line-s" in model.variables else None
        transformer_var = (
            model.variables["Transformer-s"] if "Transformer-s" in model.variables else None
        )
        link_var = model.variables["Link-p"] if "Link-p" in model.variables else None

        for spec in specs:
            expr = None

            if line_var is not None and spec['positive_lines']:
                expr = line_var.sel({"name": spec['positive_lines']}).sum("name")
            if line_var is not None and spec['negative_lines']:
                neg_expr = line_var.sel({"name": spec['negative_lines']}).sum("name")
                expr = -neg_expr if expr is None else expr - neg_expr

            if transformer_var is not None and spec['positive_transformers']:
                tf_expr = transformer_var.sel({"name": spec['positive_transformers']}).sum("name")
                expr = tf_expr if expr is None else expr + tf_expr
            if transformer_var is not None and spec['negative_transformers']:
                neg_tf_expr = transformer_var.sel({"name": spec['negative_transformers']}).sum("name")
                expr = -neg_tf_expr if expr is None else expr - neg_tf_expr

            if link_var is not None and spec['links']:
                link_names = [entry['name'] for entry in spec['links']]
                coeffs = xr.DataArray(
                    [entry['sign'] for entry in spec['links']],
                    dims=["name"],
                    coords={"name": pd.Index(link_names, name="name")},
                )
                link_expr = (link_var.sel({"name": link_names}) * coeffs).sum("name")
                expr = link_expr if expr is None else expr + link_expr

            if expr is None:
                continue

            limit = spec['limit_series'].reindex(snapshots).ffill().bfill()
            limit_da = xr.DataArray(
                limit.values,
                dims=["snapshot"],
                coords={"snapshot": snapshots},
            )
            safe_name = (
                spec['boundary_name']
                .replace("-", "_")
                .replace("+", "_")
                .replace(":", "_")
            )
            if spec['soft_penalty_gbp_per_mwh'] is None:
                model.add_constraints(
                    expr <= limit_da,
                    name=f"NESO_boundary_{safe_name}_upper",
                )
                model.add_constraints(
                    -expr <= limit_da,
                    name=f"NESO_boundary_{safe_name}_lower",
                )
                continue

            upper_bound = spec['soft_max_violation_mw']
            if upper_bound is None or not np.isfinite(upper_bound):
                slack_upper = np.inf
            else:
                slack_upper = float(upper_bound)
            slack_upper_da = xr.DataArray(
                np.full(len(snapshots), slack_upper),
                dims=["snapshot"],
                coords={"snapshot": snapshots},
            )
            slack_lower_da = xr.zeros_like(slack_upper_da)

            model.add_variables(
                lower=slack_lower_da,
                upper=slack_upper_da,
                name=f"NESO_boundary_{safe_name}_upper_slack",
            )
            model.add_variables(
                lower=slack_lower_da,
                upper=slack_upper_da,
                name=f"NESO_boundary_{safe_name}_lower_slack",
            )
            upper_slack = model.variables[f"NESO_boundary_{safe_name}_upper_slack"]
            lower_slack = model.variables[f"NESO_boundary_{safe_name}_lower_slack"]

            model.add_constraints(
                expr - upper_slack <= limit_da,
                name=f"NESO_boundary_{safe_name}_upper",
            )
            model.add_constraints(
                -expr - lower_slack <= limit_da,
                name=f"NESO_boundary_{safe_name}_lower",
            )

            snapshot_hours = pd.Series(1.0, index=pd.Index(snapshots))
            if len(snapshot_hours) > 1:
                inferred_hours = (
                    pd.Index(snapshots)
                    .to_series()
                    .diff()
                    .dt.total_seconds()
                    .dropna()
                    .median()
                    / 3600
                )
                if pd.notna(inferred_hours) and inferred_hours > 0:
                    snapshot_hours.iloc[:] = float(inferred_hours)
            snapshot_hours_da = xr.DataArray(
                snapshot_hours.values,
                dims=["snapshot"],
                coords={"snapshot": snapshots},
            )
            slack_cost = (
                (upper_slack + lower_slack)
                * float(spec['soft_penalty_gbp_per_mwh'])
                * snapshot_hours_da
            ).sum()
            model.objective = model.objective + slack_cost

    logger.info(f"Prepared {len(specs)} aggregate NESO boundary constraints")
    return boundary_extra_functionality


def _apply_csv_outage_schedule(network, outage_cfg, snapshots, logger):
    """Load outage schedule from CSV and apply to network."""
    csv_path = outage_cfg.get('file')
    if not csv_path:
        logger.warning("outage_schedule.file is null — no outages applied")
        return False

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning(f"Outage schedule file not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    required_cols = {'component', 'component_id', 'start', 'end', 's_max_pu'}
    if not required_cols.issubset(df.columns):
        logger.error(f"Outage CSV must have columns {required_cols}, got {set(df.columns)}")
        return False

    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    # Initialise s_max_pu DataFrames at 1.0
    lines_smax = pd.DataFrame(1.0, index=snapshots, columns=network.lines.index)
    xfmr_smax = pd.DataFrame(1.0, index=snapshots, columns=network.transformers.index)

    applied = 0
    for _, row in df.iterrows():
        comp_type = str(row['component']).strip().lower()
        comp_id = str(row['component_id']).strip()
        s_max_pu = float(row['s_max_pu'])
        mask = (snapshots >= row['start']) & (snapshots < row['end'])

        if comp_type == 'line' and comp_id in network.lines.index:
            # Take the minimum if overlapping outages exist
            lines_smax.loc[mask, comp_id] = np.minimum(
                lines_smax.loc[mask, comp_id], s_max_pu
            )
            applied += 1
        elif comp_type == 'transformer' and comp_id in network.transformers.index:
            xfmr_smax.loc[mask, comp_id] = np.minimum(
                xfmr_smax.loc[mask, comp_id], s_max_pu
            )
            applied += 1
        else:
            logger.debug(f"Outage row skipped: {comp_type}/{comp_id} not in network")

    # Only set time-varying columns that differ from 1.0
    affected_lines = lines_smax.columns[lines_smax.min() < 1.0]
    affected_xfmrs = xfmr_smax.columns[xfmr_smax.min() < 1.0]

    if len(affected_lines) > 0:
        network.lines_t.s_max_pu = lines_smax[affected_lines]
    if len(affected_xfmrs) > 0:
        network.transformers_t.s_max_pu = xfmr_smax[affected_xfmrs]

    total_outage_hours = 0
    for col in affected_lines:
        total_outage_hours += (lines_smax[col] < 1.0).sum()
    for col in affected_xfmrs:
        total_outage_hours += (xfmr_smax[col] < 1.0).sum()

    logger.info(f"  Loaded {applied} outage events from {csv_path}")
    logger.info(f"  Affected: {len(affected_lines)} lines, {len(affected_xfmrs)} transformers")
    logger.info(f"  Total component-hours with reduced capacity: {total_outage_hours:,}")
    return applied > 0


def _apply_neso_boundary_limits(network, outage_cfg, snapshots, logger):
    """Fetch NESO limits and apply the legacy uniform line-derating interpretation.

    For each NESO constraint boundary, this function:
    1. Fetches time-varying transfer limits from the NESO API (one boundary per query
       to avoid the API's ~200-record cap on multi-group queries)
    2. Maps boundary groups to PyPSA lines via a YAML mapping file
    3. Computes s_max_pu = NESO_limit / sum(s_nom of boundary lines)
    4. Applies the resulting time-varying s_max_pu to all lines in each boundary

    When a line appears in multiple boundaries (e.g. SSHARN lines also in ESTEX),
    the tightest (minimum) s_max_pu across all boundaries is applied.

    The NESO limits already incorporate N-1/N-2 security margins and real
    transmission outages, so no further outage modelling is needed.

    Parameters
    ----------
    network : pypsa.Network
    outage_cfg : dict  — the ``outage_schedule`` config subtree
    snapshots : pd.DatetimeIndex
    logger : logging.Logger

    Returns
    -------
    bool — True if any boundary limits were applied
    """
    neso_inputs = _load_neso_boundary_inputs(outage_cfg, snapshots, logger)
    if not neso_inputs:
        return False

    boundaries = neso_inputs['boundaries']
    neso_df = neso_inputs['neso_df']
    tolerance_minutes = neso_inputs['tolerance_minutes']
    gap_fill_mode = neso_inputs['gap_fill_mode']
    min_s_max_pu = neso_inputs['min_s_max_pu']

    # ── Apply boundary limits to network lines ───────────────────────────
    lines_smax = pd.DataFrame(1.0, index=snapshots, columns=network.lines.index)
    applied_boundaries = 0

    for boundary_name, boundary_def in boundaries.items():
        # Prefer constraint_lines (narrow derating set) over lines (full validation union).
        # constraint_lines is optional; if absent, fall back to lines.
        line_ids = boundary_def.get('constraint_lines') or boundary_def.get('lines', [])
        # Filter to lines that exist in this network
        valid_lines = [lid for lid in line_ids if lid in network.lines.index]
        if not valid_lines:
            logger.debug(f"  {boundary_name}: no matching lines in network, skipping")
            continue

        total_s_nom = network.lines.loc[valid_lines, 's_nom'].sum()
        if total_s_nom <= 0:
            continue

        # Get NESO data for this boundary
        bnd_data = neso_df[neso_df['boundary'] == boundary_name].copy()
        if bnd_data.empty:
            logger.debug(f"  {boundary_name}: no NESO data found, skipping")
            continue

        limit_series = _align_neso_boundary_limit_series(
            bnd_data=bnd_data,
            snapshots=snapshots,
            tolerance_minutes=tolerance_minutes,
            gap_fill_mode=gap_fill_mode,
            fallback_limit=total_s_nom,
            logger=logger,
            boundary_name=boundary_name,
        )

        # Compute s_max_pu for all lines in the boundary
        # s_max_pu = NESO_limit / total_s_nom (clamped to [min_s_max_pu, 1.0])
        s_max_pu_series = (limit_series / total_s_nom).clip(min_s_max_pu, 1.0)

        # Apply to all lines in the boundary (minimum across overlapping boundaries)
        for lid in valid_lines:
            lines_smax[lid] = np.minimum(lines_smax[lid].values, s_max_pu_series.values)

        avg_limit = limit_series.mean()
        min_limit = limit_series.min()
        max_limit = limit_series.max()
        avg_smax = s_max_pu_series.mean()
        logger.info(
            f"  {boundary_name}: {len(valid_lines)} lines, "
            f"total_s_nom={total_s_nom:.0f} MVA, "
            f"NESO limit={min_limit:.0f}-{max_limit:.0f} MW (avg {avg_limit:.0f}), "
            f"avg s_max_pu={avg_smax:.3f}"
        )
        applied_boundaries += 1

    # Only set time-varying columns that differ from 1.0
    affected = lines_smax.columns[lines_smax.min() < 1.0]
    if len(affected) > 0:
        network.lines_t.s_max_pu = lines_smax[affected]

    total_constrained_hours = sum((lines_smax[c] < 1.0).sum() for c in affected)
    logger.info(f"  Applied {applied_boundaries} NESO boundary constraints to "
                f"{len(affected)} lines ({total_constrained_hours:,} line-hours constrained)")
    return applied_boundaries > 0


def _apply_synthetic_outage_schedule(network, outage_cfg, snapshots, logger):
    """Generate stochastic maintenance outages from configuration parameters."""
    synth_cfg = outage_cfg.get('synthetic', {})
    seed = synth_cfg.get('seed', 42)
    min_s_nom = synth_cfg.get('min_s_nom_mva', 100)
    for_rates = synth_cfg.get('forced_outage_rate', {400: 0.03, 275: 0.04, 132: 0.05})
    maint_hours = synth_cfg.get('maintenance_duration_hours', 168)
    seasonal_weights = synth_cfg.get('seasonal_weights', {})

    rng = np.random.RandomState(seed)
    n_snapshots = len(snapshots)
    hours_per_year = 8760

    # Build seasonal probability array aligned to snapshots
    seasonal_prob = np.ones(n_snapshots)
    if seasonal_weights:
        for i, ts in enumerate(snapshots):
            month = ts.month
            seasonal_prob[i] = seasonal_weights.get(month, 1.0)
        # Normalise so mean = 1.0
        seasonal_prob = seasonal_prob / seasonal_prob.mean()

    # Process lines and transformers together
    applied_count = 0
    for comp_name, comp_df in [('lines', network.lines), ('transformers', network.transformers)]:
        eligible = comp_df[comp_df['s_nom'] >= min_s_nom].copy()
        if len(eligible) == 0:
            continue

        smax = pd.DataFrame(1.0, index=snapshots, columns=eligible.index)

        for comp_id, row in eligible.iterrows():
            v_nom = row.get('v_nom', 0)
            # Find matching forced outage rate (nearest voltage level)
            rate = 0.0
            for v_level in sorted(for_rates.keys(), key=lambda x: abs(float(x) - v_nom)):
                rate = for_rates[float(v_level)]
                break

            if rate <= 0:
                continue

            # Number of maintenance events = target unavailable hours / duration
            target_hours = rate * hours_per_year * (n_snapshots / hours_per_year)
            n_events = max(1, int(round(target_hours / maint_hours)))

            # Place events with seasonal weighting
            for _ in range(n_events):
                # Weighted random start hour
                probs = seasonal_prob.copy()
                # Can't start so late that the window exceeds snapshots
                latest_start = max(0, n_snapshots - maint_hours)
                probs[latest_start:] = 0.0
                if probs.sum() == 0:
                    continue
                probs = probs / probs.sum()
                start_idx = rng.choice(n_snapshots, p=probs)
                end_idx = min(start_idx + maint_hours, n_snapshots)
                smax.iloc[start_idx:end_idx][comp_id] = 0.0
                applied_count += 1

        affected = smax.columns[smax.min() < 1.0]
        if len(affected) > 0:
            if comp_name == 'lines':
                network.lines_t.s_max_pu = smax[affected]
            else:
                network.transformers_t.s_max_pu = smax[affected]

            total_hours = sum((smax[c] < 1.0).sum() for c in affected)
            logger.info(f"  {comp_name}: {len(affected)} components with outages, "
                        f"{total_hours:,} total component-hours")

    logger.info(f"  Generated {applied_count} synthetic outage events "
                f"(seed={seed}, min_s_nom={min_s_nom} MVA)")
    return applied_count > 0


def improve_numerical_conditioning(network, logger):
    """
    Improve numerical conditioning by removing very small components and
    clamping extreme parameter values.
    
    Large LP problems with coefficient ranges exceeding ~1e6 cause Gurobi
    to report "Numerical trouble encountered". This function:
    1. Removes generators, storage units, and links with p_nom < 0.1 MW
       (negligible capacity that introduces extreme coefficient ratios)
    2. Clamps transformer reactance values that are unreasonably high
    
    Parameters
    ----------
    network : pypsa.Network
        Network to clean up
    logger : logging.Logger
        Logger instance
    """
    logger.info("=" * 80)
    logger.info("IMPROVING NUMERICAL CONDITIONING")
    logger.info("=" * 80)
    
    min_pnom = 0.1  # MW - components below this are negligible
    
    # 1. Remove very small generators (excluding load_shedding)
    non_ls_gens = network.generators[network.generators.carrier != 'load_shedding']
    small_gens = non_ls_gens[non_ls_gens.p_nom < min_pnom]
    if len(small_gens) > 0:
        total_removed_cap = small_gens.p_nom.sum()
        logger.info(f"Removing {len(small_gens)} generators with p_nom < {min_pnom} MW "
                     f"(total: {total_removed_cap:.2f} MW - negligible)")
        # Also remove from time-varying DataFrames
        for attr in ['p_max_pu', 'p_min_pu', 'marginal_cost']:
            df = getattr(network.generators_t, attr, None)
            if df is not None and len(df) > 0:
                cols_to_drop = [c for c in small_gens.index if c in df.columns]
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
        network.generators.drop(small_gens.index, inplace=True)
    
    # 2. Remove very small storage units
    small_storage = network.storage_units[network.storage_units.p_nom < min_pnom]
    if len(small_storage) > 0:
        total_removed_stor = small_storage.p_nom.sum()
        logger.info(f"Removing {len(small_storage)} storage units with p_nom < {min_pnom} MW "
                     f"(total: {total_removed_stor:.2f} MW - negligible)")
        for attr in ['p_max_pu', 'p_min_pu', 'state_of_charge_set', 'inflow']:
            df = getattr(network.storage_units_t, attr, None)
            if df is not None and len(df) > 0:
                cols_to_drop = [c for c in small_storage.index if c in df.columns]
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
        network.storage_units.drop(small_storage.index, inplace=True)
    
    # 3. Remove very small links (but not HVDC interconnectors)
    if len(network.links) > 0:
        small_links = network.links[network.links.p_nom < min_pnom]
        if len(small_links) > 0:
            total_removed_links = small_links.p_nom.sum()
            logger.info(f"Removing {len(small_links)} links with p_nom < {min_pnom} MW "
                         f"(total: {total_removed_links:.2f} MW - negligible)")
            for attr in ['p_max_pu', 'p_min_pu', 'efficiency']:
                df = getattr(network.links_t, attr, None)
                if df is not None and len(df) > 0:
                    cols_to_drop = [c for c in small_links.index if c in df.columns]
                    if cols_to_drop:
                        df.drop(columns=cols_to_drop, inplace=True)
            network.links.drop(small_links.index, inplace=True)
    
    # 4. Clamp transformer reactance to reasonable range
    # Very high x values (>10) create extreme coefficient ratios
    if len(network.transformers) > 0:
        high_x = network.transformers.x > 10.0
        n_high_x = high_x.sum()
        if n_high_x > 0:
            logger.info(f"Clamping {n_high_x} transformer reactance values from "
                         f"max {network.transformers.loc[high_x, 'x'].max():.1f} to 10.0")
            network.transformers.loc[high_x, 'x'] = 10.0
    
    # Report final coefficient ranges
    gen_pnom = network.generators[network.generators.p_nom > 0]['p_nom']
    if len(gen_pnom) > 0:
        ratio = gen_pnom.max() / gen_pnom.min()
        logger.info(f"Generator p_nom range: {gen_pnom.min():.2f} - {gen_pnom.max():.0f} MW (ratio: {ratio:.0f})")
    
    logger.info(f"Final network: {len(network.generators)} generators, "
                 f"{len(network.storage_units)} storage, {len(network.links)} links")
    logger.info("=" * 80)


def export_optimization_results(network, output_dir, scenario_id, logger):
    """
    Export optimization results to CSV files.
    
    Parameters
    ----------
    network : pypsa.Network
        Solved network
    output_dir : str or Path
        Directory to write CSV files
    scenario_id : str
        Scenario identifier
    logger : logging.Logger
        Logger instance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Exporting optimization results to CSV...")
    
    # Generator dispatch
    if len(network.generators_t.p) > 0:
        generation_df = network.generators_t.p.copy()
        # Add carrier information
        generation_df.columns = [
            f"{gen}_{network.generators.loc[gen, 'carrier']}" 
            for gen in generation_df.columns
        ]
        logger.info(f"  Generation time series: {len(generation_df)} snapshots, {len(generation_df.columns)} generators")
    else:
        generation_df = pd.DataFrame()
        logger.warning("  No generation results found")
    
    # Storage state of charge and dispatch
    if len(network.storage_units_t.p) > 0:
        storage_df = pd.DataFrame({
            'snapshot': network.snapshots.tolist() * len(network.storage_units),
            'storage_unit': [su for su in network.storage_units.index for _ in network.snapshots],
            'carrier': [network.storage_units.loc[su, 'carrier'] for su in network.storage_units.index for _ in network.snapshots],
            'p_dispatch': network.storage_units_t.p.values.flatten(),
        })
        # Only add state_of_charge if it has data
        if len(network.storage_units_t.state_of_charge) > 0 and len(network.storage_units_t.state_of_charge.values.flatten()) > 0:
            storage_df['state_of_charge'] = network.storage_units_t.state_of_charge.values.flatten()
        logger.info(f"  Storage results: {len(storage_df)} records")
    else:
        storage_df = pd.DataFrame()
        logger.warning("  No storage results found")
    
    # Line flows
    if len(network.lines_t.p0) > 0:
        flows_df = network.lines_t.p0.copy()
        flows_df.columns = [f"line_{line}" for line in flows_df.columns]
        logger.info(f"  Line flows: {len(flows_df)} snapshots, {len(flows_df.columns)} lines")
    else:
        flows_df = pd.DataFrame()
        logger.warning("  No line flow results found")
    
    # System costs
    if hasattr(network, 'objective') and network.objective is not None:
        total_cost = network.objective
        logger.info(f"  Total system cost: £{total_cost:,.2f}")
    else:
        total_cost = np.nan
        logger.warning("  No objective value found")
    
    # Cost breakdown by component
    costs_data = []
    
    # Generator costs
    if len(network.generators_t.p) > 0:
        for gen in network.generators.index:
            if gen in network.generators_t.p.columns:
                gen_output = network.generators_t.p[gen].sum()
                marginal_cost = network.generators.loc[gen, 'marginal_cost']
                gen_cost = gen_output * marginal_cost
                costs_data.append({
                    'component_type': 'generator',
                    'component_name': gen,
                    'carrier': network.generators.loc[gen, 'carrier'],
                    'output_MWh': gen_output,
                    'marginal_cost': marginal_cost,
                    'total_cost': gen_cost
                })
    
    costs_df = pd.DataFrame(costs_data)
    if len(costs_df) > 0:
        logger.info(f"  Cost breakdown: {len(costs_df)} components")
    else:
        logger.warning("  No cost data available")
    
    # Emissions (if CO2 intensity available)
    emissions_data = []
    if 'co2_emissions' in network.generators.columns:
        for gen in network.generators.index:
            if gen in network.generators_t.p.columns:
                gen_output = network.generators_t.p[gen].sum()
                co2_intensity = network.generators.loc[gen, 'co2_emissions']
                emissions = gen_output * co2_intensity
                emissions_data.append({
                    'generator': gen,
                    'carrier': network.generators.loc[gen, 'carrier'],
                    'generation_MWh': gen_output,
                    'co2_intensity_kg_per_MWh': co2_intensity,
                    'total_emissions_kg': emissions
                })
    
    emissions_df = pd.DataFrame(emissions_data)
    if len(emissions_df) > 0:
        total_emissions = emissions_df['total_emissions_kg'].sum()
        logger.info(f"  Total CO2 emissions: {total_emissions:,.0f} kg ({total_emissions/1e6:,.2f} tonnes)")
    else:
        logger.warning("  No emissions data available")
    
    return generation_df, storage_df, flows_df, costs_df, emissions_df


def generate_optimization_summary(network, scenario_config, solver_name, 
                                 solver_status, solve_time, output_path, logger):
    """
    Generate optimization summary report.
    
    Parameters
    ----------
    network : pypsa.Network
        Solved network
    scenario_config : dict
        Scenario configuration
    solver_name : str
        Solver used
    solver_status : str
        Optimization status
    solve_time : float
        Solution time in seconds
    output_path : str or Path
        Path to write summary
    logger : logging.Logger
        Logger instance
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("PYPSA-GB OPTIMIZATION SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Scenario info
    summary_lines.append("SCENARIO")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Scenario: {scenario_config.get('scenario_id', 'Unknown')}")
    summary_lines.append(f"Network: {len(network.buses)} buses, {len(network.generators)} generators")
    summary_lines.append(f"Snapshots: {len(network.snapshots)}")
    
    # Add solve period info if applicable
    solve_period_config = scenario_config.get('solve_period', {})
    if solve_period_config.get('enabled', False):
        if len(network.snapshots) > 0:
            period_start = network.snapshots[0]
            period_end = network.snapshots[-1]
            duration = period_end - period_start
            summary_lines.append(f"Solve period: {period_start.date()} to {period_end.date()}")
            summary_lines.append(f"Duration: {duration.days} days, {duration.seconds//3600} hours")
            if 'auto_select' in solve_period_config:
                summary_lines.append(f"Period selection: {solve_period_config['auto_select']} (auto-selected)")
            else:
                summary_lines.append(f"Period selection: explicit dates")
    else:
        summary_lines.append(f"Solve period: Full year")
    
    summary_lines.append("")
    
    # Optimization info
    summary_lines.append("OPTIMIZATION")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Solver: {solver_name}")
    summary_lines.append(f"Status: {solver_status}")
    summary_lines.append(f"Solution time: {solve_time:.2f} seconds ({solve_time/60:.2f} minutes)")
    summary_lines.append("")
    
    # Objective value
    if hasattr(network, 'objective') and network.objective is not None:
        summary_lines.append("SYSTEM COSTS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total system cost: £{network.objective:,.2f}")
        if len(network.snapshots) > 0:
            cost_per_snapshot = network.objective / len(network.snapshots)
            summary_lines.append(f"Average cost per hour: £{cost_per_snapshot:,.2f}")
        summary_lines.append("")
    
    # Generation summary
    if len(network.generators_t.p) > 0:
        summary_lines.append("GENERATION")
        summary_lines.append("-" * 80)
        gen_by_carrier = {}
        for gen in network.generators.index:
            carrier = network.generators.loc[gen, 'carrier']
            if gen in network.generators_t.p.columns:
                output = network.generators_t.p[gen].sum()
                if carrier not in gen_by_carrier:
                    gen_by_carrier[carrier] = 0
                gen_by_carrier[carrier] += output
        
        total_generation = sum(gen_by_carrier.values())
        for carrier in sorted(gen_by_carrier.keys(), key=lambda x: gen_by_carrier[x], reverse=True):
            output = gen_by_carrier[carrier]
            pct = (output / total_generation * 100) if total_generation > 0 else 0
            summary_lines.append(f"  {carrier:30s}: {output:12,.0f} MWh ({pct:5.1f}%)")
        summary_lines.append(f"  {'TOTAL':30s}: {total_generation:12,.0f} MWh")
        summary_lines.append("")
    
    # Check for load shedding
    load_shedding_carriers = ['load_shedding', 'load shedding', 'voll']
    load_shedding = 0
    for gen in network.generators.index:
        carrier = network.generators.loc[gen, 'carrier'].lower()
        if any(ls in carrier for ls in load_shedding_carriers):
            if gen in network.generators_t.p.columns:
                load_shedding += network.generators_t.p[gen].sum()
    
    if load_shedding > 1:  # More than 1 MWh
        summary_lines.append("WARNING: LOAD SHEDDING DETECTED")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total load shedding: {load_shedding:,.0f} MWh")
        summary_lines.append("This indicates insufficient generation capacity or transmission constraints.")
        summary_lines.append("")
    
    summary_lines.append("=" * 80)
    
    # Write to file
    summary_text = "\n".join(summary_lines)
    Path(output_path).write_text(summary_text, encoding='utf-8')
    logger.info(f"Optimization summary written to {output_path}")
    
    # Also log key results
    logger.info("\n" + summary_text)


if __name__ == "__main__":
    import time
    
    # Set up logging - use Snakemake log path if available
    log_path = snakemake.log[0] if hasattr(snakemake, 'log') and snakemake.log else "solve_network"
    logger = setup_logging(log_path)
    
    logger.info("=" * 80)
    logger.info("SOLVING NETWORK WITH PYPSA OPTIMIZATION")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Load network
        input_path = snakemake.input.network
        logger.info(f"Loading network from: {input_path}")
        network = load_network(input_path, custom_logger=logger)
        
        # Get parameters
        scenario_config = snakemake.params.scenario_config
        solver_name = snakemake.params.solver
        solver_options = snakemake.params.solver_options
        scenario_id = scenario_config.get('scenario_id', snakemake.wildcards.scenario)
        
        logger.info(f"Scenario: {scenario_id}")
        logger.info(f"Network loaded: {len(network.buses)} buses, {len(network.generators)} generators")
        logger.info(f"Full year snapshots: {len(network.snapshots)} (from {network.snapshots[0]} to {network.snapshots[-1]})")
        
        # CRITICAL: Validate marginal costs before proceeding
        # This catches unbounded optimization issues (zero-cost load shedding, zero-cost thermal)
        validate_network_costs(network, logger)
        
        # Apply transmission constraint relaxation if configured
        # This helps feasibility for detailed networks (ETYS) with tight line limits
        apply_transmission_relaxation(network, scenario_config, logger)
        
        # Apply per-line s_nom overrides (fix known ETYS data errors)
        apply_line_rating_overrides(network, scenario_config, logger)
        
        # Apply transmission outage schedule if configured
        # (reduces line capacity at specific timesteps via s_max_pu)
        apply_outage_schedule(network, scenario_config, logger)
        
        # Improve numerical conditioning by removing tiny components
        # and clamping extreme parameter values (prevents "Numerical trouble" errors)
        improve_numerical_conditioning(network, logger)
        
        # Apply solve period if configured
        solve_period_config = scenario_config.get('solve_period', {})
        if solve_period_config.get('enabled', False):
            logger.info("=" * 80)
            logger.info("APPLYING SOLVE PERIOD RESTRICTION")
            logger.info("=" * 80)
            
            # Auto-select period if requested
            if 'auto_select' in solve_period_config:
                auto_mode = solve_period_config['auto_select']
                logger.info(f"Auto-selecting period: {auto_mode}")
                
                if auto_mode == 'peak_demand_week':
                    # Find week with highest total demand
                    if len(network.loads_t.p_set) > 0:
                        total_demand = network.loads_t.p_set.sum(axis=1)
                        # Resample to weekly and find peak
                        weekly_demand = total_demand.resample('W').sum()
                        peak_week_end = weekly_demand.idxmax()
                        solve_start = peak_week_end - pd.Timedelta(days=6)
                        solve_end = peak_week_end
                        logger.info(f"Peak demand week: {solve_start.date()} to {solve_end.date()}")
                    else:
                        raise ValueError("Cannot auto-select peak demand week: no load data found")
                
                elif auto_mode == 'peak_wind_week':
                    # Find week with highest wind generation potential
                    wind_gens = network.generators[network.generators.carrier.str.contains('wind', case=False, na=False)]
                    if len(wind_gens) > 0 and len(network.generators_t.p_max_pu) > 0:
                        wind_profiles = network.generators_t.p_max_pu[[g for g in wind_gens.index if g in network.generators_t.p_max_pu.columns]]
                        total_wind_cf = wind_profiles.mean(axis=1)
                        weekly_wind = total_wind_cf.resample('W').mean()
                        peak_week_end = weekly_wind.idxmax()
                        solve_start = peak_week_end - pd.Timedelta(days=6)
                        solve_end = peak_week_end
                        logger.info(f"Peak wind week: {solve_start.date()} to {solve_end.date()}")
                    else:
                        raise ValueError("Cannot auto-select peak wind week: no wind generator data found")
                
                elif auto_mode == 'low_wind_week':
                    # Find week with lowest wind generation potential
                    wind_gens = network.generators[network.generators.carrier.str.contains('wind', case=False, na=False)]
                    if len(wind_gens) > 0 and len(network.generators_t.p_max_pu) > 0:
                        wind_profiles = network.generators_t.p_max_pu[[g for g in wind_gens.index if g in network.generators_t.p_max_pu.columns]]
                        total_wind_cf = wind_profiles.mean(axis=1)
                        weekly_wind = total_wind_cf.resample('W').mean()
                        low_week_end = weekly_wind.idxmin()
                        solve_start = low_week_end - pd.Timedelta(days=6)
                        solve_end = low_week_end
                        logger.info(f"Low wind week: {solve_start.date()} to {solve_end.date()}")
                    else:
                        raise ValueError("Cannot auto-select low wind week: no wind generator data found")
                
                else:
                    raise ValueError(f"Unknown auto_select mode: {auto_mode}")
            
            else:
                # Use explicit start/end dates
                solve_start = pd.Timestamp(solve_period_config['start'])
                solve_end = pd.Timestamp(solve_period_config['end'])
                logger.info(f"Explicit solve period: {solve_start.date()} to {solve_end.date()}")
            
            # Filter snapshots to solve period
            original_snapshots = len(network.snapshots)
            mask = (network.snapshots >= solve_start) & (network.snapshots <= solve_end)
            selected_snapshots = network.snapshots[mask]
            
            if len(selected_snapshots) == 0:
                raise ValueError(f"No snapshots found in period {solve_start} to {solve_end}")
            
            # Set network snapshots to selected period
            network.set_snapshots(selected_snapshots)
            
            period_duration = solve_end - solve_start
            logger.info(f"Solve period snapshots: {len(network.snapshots)} ({period_duration.days} days + {period_duration.seconds//3600} hours)")
            logger.info(f"Reduction: {original_snapshots} → {len(network.snapshots)} snapshots ({100*(1-len(network.snapshots)/original_snapshots):.1f}% reduction)")
            logger.info("=" * 80)
        else:
            logger.info("Solving full year (no period restriction)")
        
        apply_load_shedding_limits(network, logger)

        logger.info(f"Optimization will run for: {len(network.snapshots)} snapshots")
        
        # Get solve mode from global config
        # LP mode: Fast solving, no ramp limits, no unit commitment
        # MILP mode: Slower solving, with ramp limits and unit commitment
        global_solve_mode = get_solve_mode_from_config()
        logger.info(f"Global solve mode from config.yaml: {global_solve_mode}")
        
        # Determine if unit commitment should be enabled
        # Priority: 1) global solve_mode=LP disables all, 2) scenario_config.unit_commitment.enabled
        if global_solve_mode == 'LP':
            # LP mode: disable unit commitment regardless of scenario config
            unit_commitment_enabled = False
            logger.info("LP mode: Unit commitment disabled (from global solve_mode)")
        else:
            # MILP mode: check scenario config (default True for MILP)
            unit_commitment_enabled = scenario_config.get('unit_commitment', {}).get('enabled', True)
        
        if not unit_commitment_enabled:
            logger.info("=" * 80)
            logger.info("DISABLING UNIT COMMITMENT (LP MODE)")
            logger.info("=" * 80)
            if 'committable' in network.generators.columns:
                n_committable_before = network.generators.committable.sum() if network.generators.committable.any() else 0
                logger.info(f"Generators with committable=True before: {n_committable_before}")
                network.generators['committable'] = False
                logger.info("Set all generators committable=False")
                logger.info("This converts problem from MILP to LP (much faster!)")
            else:
                logger.info("No 'committable' column found in generators (already LP)")
            logger.info("=" * 80)
        else:
            if 'committable' in network.generators.columns:
                n_committable = network.generators.committable.sum() if network.generators.committable.any() else 0
                if n_committable > 0:
                    logger.info(f"Unit commitment ENABLED: {n_committable} committable generators (MILP problem)")
                    logger.info("  This may significantly increase solve time.")
                    logger.info("  To disable: set unit_commitment.enabled: false in scenario config")
        
        # Remove must-run constraints (p_min_pu) if configured
        # Default: False (respect data), but can be True to prevent infeasibility
        remove_must_run = scenario_config.get('optimization', {}).get('remove_must_run', False)
        
        logger.info("=" * 80)
        if remove_must_run:
            logger.info("REMOVING MUST-RUN CONSTRAINTS (p_min_pu = 0)")
            logger.info("=" * 80)
            if 'p_min_pu' in network.generators.columns:
                n_must_run_before = (network.generators.p_min_pu > 0).sum()
                logger.info(f"Generators with p_min_pu > 0 before: {n_must_run_before}")
                network.generators['p_min_pu'] = 0.0
                logger.info("Set all generators p_min_pu=0.0")
                logger.info("This allows all generators to turn off when not economically needed")
            else:
                logger.info("No 'p_min_pu' column found in generators")
        else:
            logger.info("PRESERVING MUST-RUN CONSTRAINTS (p_min_pu from data)")
            logger.info("=" * 80)
            if 'p_min_pu' in network.generators.columns:
                n_must_run = (network.generators.p_min_pu > 0).sum()
                logger.info(f"Generators with p_min_pu > 0: {n_must_run}")
                logger.info("These generators (e.g. nuclear, biomass) will be forced to run if available")
            else:
                logger.info("No 'p_min_pu' column found - all generators can turn off")
        logger.info("=" * 80)
        
        # Check generator aggregation status
        if hasattr(network, 'meta') and network.meta.get('aggregated', False):
            logger.info("NOTE: Network uses GENERATOR AGGREGATION")
            logger.info("  Generators of same carrier at same bus are aggregated")
            logger.info("  This significantly improves solve speed")

        
        # Configure solver
        solver_name, solver_options = configure_solver(network, solver_name, solver_options, logger)
        
        # Run optimization
        logger.info("Starting optimization...")
        logger.info(f"This may take several minutes depending on network size and solver...")
        
        solve_start = time.time()
        
        # Note: skip_objective and track_iterations are PyPSA parameters, NOT solver options
        # They should not be passed in solver_options to avoid Gurobi errors
        log_hydro_constraint_setup(network, scenario_config, logger)
        hydro_callback = build_hydro_constraints_callback(network, scenario_config)
        neso_boundary_callback = _build_neso_boundary_constraints_callback(
            network, scenario_config, logger
        )

        status, termination_condition = network.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=combine_extra_functionalities(
                hydro_callback, neso_boundary_callback
            ),
        )
        
        solve_time = time.time() - solve_start
        
        logger.info(f"Optimization completed in {solve_time:.2f} seconds")
        logger.info(f"Status: {status}")
        logger.info(f"Termination condition: {termination_condition}")
        
        # Check if optimization was successful
        if status != "ok":
            logger.error(f"Optimization failed with status: {status}")
            logger.error(f"Termination condition: {termination_condition}")
            raise RuntimeError(f"Optimization failed: {status}")
        
        # Log objective value
        if hasattr(network, 'objective'):
            logger.info(f"Total system cost: £{network.objective:,.2f}")
        
        # Export results to CSV
        logger.info("Exporting results to CSV files...")
        results_dir = Path(snakemake.output.generation_csv).parent
        
        generation_df, storage_df, flows_df, costs_df, emissions_df = export_optimization_results(
            network, results_dir, scenario_id, logger
        )
        
        # Write CSV files
        if not generation_df.empty:
            generation_df.to_csv(snakemake.output.generation_csv)
            logger.info(f"  Saved: {snakemake.output.generation_csv}")
        
        if not storage_df.empty:
            storage_df.to_csv(snakemake.output.storage_csv, index=False)
            logger.info(f"  Saved: {snakemake.output.storage_csv}")
        else:
            # Create empty CSV file so Snakemake output file exists (e.g., networks without storage)
            with open(snakemake.output.storage_csv, 'w') as f:
                f.write("# No storage results available (network has no storage units)\n")
            logger.info(f"  Created empty storage file: {snakemake.output.storage_csv}")

        if not flows_df.empty:
            flows_df.to_csv(snakemake.output.flows_csv)
            logger.info(f"  Saved: {snakemake.output.flows_csv}")
        else:
            # Create empty CSV file so Snakemake output file exists (e.g., zonal networks have no lines)
            with open(snakemake.output.flows_csv, 'w') as f:
                f.write("# No line flows available (network may use links instead of lines)\n")
            logger.warning(f"  Created empty flows file: {snakemake.output.flows_csv}")
        
        if not costs_df.empty:
            costs_df.to_csv(snakemake.output.costs_csv, index=False)
            logger.info(f"  Saved: {snakemake.output.costs_csv}")
        
        if not emissions_df.empty:
            emissions_df.to_csv(snakemake.output.emissions_csv, index=False)
            logger.info(f"  Saved: {snakemake.output.emissions_csv}")
        else:
            # Create empty CSV file so Snakemake output file exists
            with open(snakemake.output.emissions_csv, 'w') as f:
                f.write("# No emissions data available\n")
            logger.warning(f"  Created empty emissions file: {snakemake.output.emissions_csv}")
        
        # Save solved network
        output_path = snakemake.output.network
        logger.info(f"Saving solved network to: {output_path}")
        save_network(network, output_path, custom_logger=logger)
        
        # Generate summary report
        summary_path = snakemake.output.summary
        logger.info(f"Generating optimization summary: {summary_path}")
        generate_optimization_summary(
            network, scenario_config, solver_name, status, solve_time, summary_path, logger
        )
        
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"OPTIMIZATION COMPLETED SUCCESSFULLY (Total time: {total_time:.2f}s)")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"FATAL ERROR in network optimization: {e}", exc_info=True)
        raise

