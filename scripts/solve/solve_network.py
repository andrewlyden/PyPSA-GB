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

# Fast I/O for network loading/saving
from scripts.utilities.network_io import load_network, save_network


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
    renewable_carriers = ['wind', 'Wind', 'Wind (Onshore)', 'Wind (Offshore)', 
                         'wind_onshore', 'wind_offshore', 'solar', 'Solar', 'solar_pv',
                         'Hydro', 'hydro', 'large_hydro', 'small_hydro', 'tidal_stream',
                         'shoreline_wave', 'geothermal']
    
    non_renewable_mask = ~network.generators['carrier'].isin(renewable_carriers)
    non_renewable_zero_cost = network.generators[non_renewable_mask & (network.generators['marginal_cost'] == 0)]
    
    if len(non_renewable_zero_cost) > 0:
        capacity = non_renewable_zero_cost['p_nom'].sum()
        if capacity > 1000:  # More than 1 GW
            warnings.append(
                f"{len(non_renewable_zero_cost)} non-renewable generators have zero cost ({capacity:,.0f} MW)\n"
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
        status, termination_condition = network.optimize(
            solver_name=solver_name,
            solver_options=solver_options
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

