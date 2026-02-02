# Workflow Refactoring Plan: Fix Demand Integration Order

## Problem Statement

**Current Issue**: All generators are lost during demand disaggregation/flexibility integration, resulting in 100% load shedding in optimization.

**Root Cause**: Demand disaggregation creates `_final.nc` with only loads (0 generators), which is then used as input for flexibility integration. This happens AFTER all generators have been added, causing them to be lost.

## Current (Broken) Workflow

```
1. Network topology                        → {scenario}_network.nc
2. + Base demand integration               → {scenario}_network_demand.pkl
3. + Renewable generators                  → {scenario}_network_demand_renewables.pkl
4. + Thermal generators                    → {scenario}_network_demand_renewables_thermal.pkl
5. + Generator finalization (load shed)    → {scenario}_network_demand_renewables_thermal_generators.pkl
6. + Marginal costs                        → {scenario}_network_demand_renewables_thermal_generators_costs.pkl
7. + Storage units                         → {scenario}_network_demand_renewables_thermal_generators_storage.pkl
8. + Hydrogen infrastructure               → {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl
9. + Interconnectors                       → {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc

--- IF DISAGGREGATION ENABLED ---
10. Demand disaggregation (PARALLEL)       → {scenario}_final.nc (ONLY LOADS - 0 GENERATORS!)
    - Heat pump disaggregation
    - EV disaggregation
    - Integration into network

11. + Flexibility integration              → {scenario}_network_..._flexibility.nc
    Input: {scenario}_final.nc (0 generators) ❌

12. Finalize network                       → {scenario}.nc (ONLY LOAD SHEDDING)
```

**Result**: Final network has 0 actual generators, only 941 load shedding → 100% load shedding in solve

## Proposed (Fixed) Workflow

```
1. Network topology                        → {scenario}_network.nc

--- DEMAND STAGE (Complete all demand-side before generators) ---
2. + Base demand integration               → {scenario}_network_demand_base.pkl

3. IF DISAGGREGATION ENABLED:
   a. Disaggregate components              → CSV files in resources/demand/components/
      - Heat pump disaggregation           → {scenario}_heat_pumps_profile.csv
      - EV disaggregation                  → {scenario}_ev_profile.csv

   b. Integrate disaggregated loads        → {scenario}_network_demand_disaggregated.pkl
      - Remove HP/EV portions from base
      - Add separate HP/EV loads with carriers

   c. Add flexibility components           → {scenario}_network_demand.pkl ✓
      - Heat pump stores/links
      - EV stores/links
      - Event response

   ELSE (no disaggregation):
      Just rename base → demand             → {scenario}_network_demand.pkl ✓

--- GENERATION STAGE (Supply-side) ---
4. + Renewable generators                  → {scenario}_network_demand_renewables.pkl
5. + Thermal generators                    → {scenario}_network_demand_renewables_thermal.pkl
6. + Generator finalization (load shed)    → {scenario}_network_demand_renewables_thermal_generators.pkl
7. + Marginal costs                        → {scenario}_network_demand_renewables_thermal_generators_costs.pkl

--- INFRASTRUCTURE STAGE ---
8. + Storage units                         → {scenario}_network_demand_renewables_thermal_generators_storage.pkl
9. + Hydrogen infrastructure               → {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl
10. + Interconnectors                      → {scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc

--- FINALIZATION ---
11. Finalize network                       → {scenario}.nc (ALL COMPONENTS) ✓
```

**Result**: Final network has all generators + all demand flexibility components

## File Changes Required

### 1. rules/demand.smk

#### A. Modify `integrate_demand` rule

**Current**:
```python
rule integrate_demand:
    input:
        network=f"{resources_path}/network/{{scenario}}_network.nc"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand.pkl"
```

**New**:
```python
rule integrate_demand_base:
    """Integrate baseline demand (before disaggregation/flexibility)"""
    input:
        network=f"{resources_path}/network/{{scenario}}_network.nc"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl"
    script:
        "../scripts/demand/integrate.py"  # Same script, just baseline
```

#### B. Create new `finalize_demand` rule

```python
rule finalize_demand:
    """
    Finalize complete demand-side network (disaggregation + flexibility).

    This rule orchestrates:
    1. Load disaggregated component data (if enabled)
    2. Integrate disaggregated loads into network
    3. Add flexibility components (if enabled)

    Output is complete demand-side network ready for generator integration.
    """
    input:
        base_network=f"{resources_path}/network/{{scenario}}_network_demand_base.pkl",
        # Disaggregation inputs (conditional)
        hp_profile=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_profile.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        hp_allocation=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_allocation.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        hp_cop=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_cop.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        hp_thermal=lambda w: f"{resources_path}/demand/components/{w.scenario}_heat_pumps_thermal.csv" if is_disaggregation_enabled(w.scenario) and "heat_pumps" in get_component_names(w.scenario) else [],
        ev_profile=lambda w: f"{resources_path}/demand/components/{w.scenario}_ev_profile.csv" if is_disaggregation_enabled(w.scenario) and "electric_vehicles" in get_component_names(w.scenario) else [],
        ev_allocation=lambda w: f"{resources_path}/demand/components/{w.scenario}_ev_allocation.csv" if is_disaggregation_enabled(w.scenario) and "electric_vehicles" in get_component_names(w.scenario) else [],
        # Flexibility inputs (conditional)
        heat_demand=lambda w: f"{resources_path}/demand/heat_demand_{w.scenario}.nc" if is_hp_flexibility_enabled(w.scenario) else [],
        cop_ashp=lambda w: f"{resources_path}/demand/cop_ashp_{w.scenario}.nc" if is_hp_flexibility_enabled(w.scenario) else [],
        ev_availability=lambda w: f"{resources_path}/demand/ev_availability_{w.scenario}.csv" if is_ev_flexibility_enabled(w.scenario) else [],
        ev_dsm=lambda w: f"{resources_path}/demand/ev_dsm_{w.scenario}.csv" if is_ev_flexibility_enabled(w.scenario) else []
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand.pkl",
        summary=f"{resources_path}/demand/{{scenario}}_demand_integration_summary.csv"
    params:
        disaggregation_config=lambda w: scenarios[w.scenario].get("demand_disaggregation", {}),
        flexibility_config=lambda w: get_flexibility_config(w.scenario)
    log:
        "logs/demand/finalize_demand_{scenario}.log"
    script:
        "../scripts/demand/finalize_demand_integration.py"  # NEW SCRIPT
```

#### C. Remove/deprecate old rules

- `integrate_demand_disaggregated` (replaced by finalize_demand)
- `add_demand_flexibility` rule (now part of finalize_demand)

### 2. Create new script: scripts/demand/finalize_demand_integration.py

```python
"""
Finalize Demand Integration

Orchestrates:
1. Load base demand network
2. Integrate disaggregated components (if enabled)
3. Add flexibility (if enabled)
4. Output complete demand-side network
"""

import pypsa
import pandas as pd
import logging
from pathlib import Path

# Import existing modules
from scripts.demand.integrate import integrate_disaggregated_components
from scripts.demand.heat_pumps import add_heat_pump_flexibility
from scripts.demand.electric_vehicles import add_ev_flexibility
from scripts.demand.event_flex import add_event_flexibility


def finalize_demand_integration(
    n: pypsa.Network,
    disaggregation_config: dict,
    flexibility_config: dict,
    component_data: dict,
    logger: logging.Logger
) -> pypsa.Network:
    """
    Complete demand-side integration.

    Args:
        n: Base network with baseline demand
        disaggregation_config: Disaggregation settings
        flexibility_config: Flexibility settings
        component_data: Dict of component profiles/allocations
        logger: Logger instance

    Returns:
        Network with complete demand-side (loads + flexibility components)
    """

    # Step 1: Integrate disaggregated components
    if disaggregation_config.get("enabled", False):
        logger.info("="*80)
        logger.info("STEP 1: INTEGRATING DISAGGREGATED DEMAND COMPONENTS")
        logger.info("="*80)

        n = integrate_disaggregated_components(
            n=n,
            config=disaggregation_config,
            component_data=component_data,
            logger=logger
        )
    else:
        logger.info("Disaggregation disabled - using baseline demand only")

    # Step 2: Add flexibility components
    if flexibility_config.get("enabled", False):
        logger.info("="*80)
        logger.info("STEP 2: ADDING DEMAND FLEXIBILITY COMPONENTS")
        logger.info("="*80)

        # Heat pump flexibility
        if flexibility_config.get("heat_pumps", {}).get("enabled", False):
            n = add_heat_pump_flexibility(
                n=n,
                config=flexibility_config["heat_pumps"],
                hp_data=component_data.get("heat_pumps"),
                logger=logger
            )

        # EV flexibility
        if flexibility_config.get("electric_vehicles", {}).get("enabled", False):
            n = add_ev_flexibility(
                n=n,
                config=flexibility_config["electric_vehicles"],
                ev_data=component_data.get("electric_vehicles"),
                logger=logger
            )

        # Event response
        if flexibility_config.get("event_response", {}).get("enabled", False):
            n = add_event_flexibility(
                n=n,
                config=flexibility_config["event_response"],
                logger=logger
            )
    else:
        logger.info("Flexibility disabled - loads only")

    return n


if __name__ == "__main__":
    from scripts.utilities.logging_config import setup_logging

    logger = setup_logging(snakemake.log[0], "INFO")

    # Load base network
    n = pypsa.Network(snakemake.input.base_network)
    logger.info(f"Loaded base demand network: {len(n.buses)} buses, {len(n.loads)} loads")

    # Collect component data
    component_data = {}

    # Heat pump data
    if hasattr(snakemake.input, "hp_profile") and snakemake.input.hp_profile:
        component_data["heat_pumps"] = {
            "profile": pd.read_csv(snakemake.input.hp_profile, index_col=0, parse_dates=True),
            "allocation": pd.read_csv(snakemake.input.hp_allocation),
            "cop": pd.read_csv(snakemake.input.hp_cop, index_col=0, parse_dates=True),
            "thermal": pd.read_csv(snakemake.input.hp_thermal, index_col=0, parse_dates=True)
        }

    # EV data
    if hasattr(snakemake.input, "ev_profile") and snakemake.input.ev_profile:
        component_data["electric_vehicles"] = {
            "profile": pd.read_csv(snakemake.input.ev_profile, index_col=0, parse_dates=True),
            "allocation": pd.read_csv(snakemake.input.ev_allocation),
            "availability": pd.read_csv(snakemake.input.ev_availability, index_col=0, parse_dates=True) if hasattr(snakemake.input, "ev_availability") and snakemake.input.ev_availability else None,
            "dsm": pd.read_csv(snakemake.input.ev_dsm, index_col=0, parse_dates=True) if hasattr(snakemake.input, "ev_dsm") and snakemake.input.ev_dsm else None
        }

    # Finalize demand integration
    n = finalize_demand_integration(
        n=n,
        disaggregation_config=snakemake.params.disaggregation_config,
        flexibility_config=snakemake.params.flexibility_config,
        component_data=component_data,
        logger=logger
    )

    # Save
    n.export_to_netcdf(snakemake.output.network)
    logger.info(f"Saved demand network: {snakemake.output.network}")

    # Generate summary
    summary = generate_demand_summary(n, logger)
    summary.to_csv(snakemake.output.summary, index=False)
```

### 3. Update scripts/demand/integrate.py

Add function to integrate disaggregated components:

```python
def integrate_disaggregated_components(
    n: pypsa.Network,
    config: dict,
    component_data: dict,
    logger: logging.Logger
) -> pypsa.Network:
    """
    Integrate disaggregated demand components into network.

    This function:
    1. Removes the disaggregated portions from base demand
    2. Adds separate loads for each component with proper carriers

    Args:
        n: Network with baseline demand
        config: Disaggregation configuration
        component_data: Dict with component profiles/allocations
        logger: Logger instance

    Returns:
        Network with disaggregated loads
    """

    components = config.get("components", [])

    for component_name in components:
        if component_name not in component_data:
            logger.warning(f"Component '{component_name}' enabled but no data provided")
            continue

        data = component_data[component_name]

        logger.info(f"Integrating {component_name}...")

        # Get allocation (which buses get this component)
        allocation = data["allocation"]  # bus, annual_gwh, fraction_of_bus

        # Get profile (time series)
        profile = data["profile"]  # timesteps x buses

        # Remove this component's demand from base loads
        fraction = config.get("use_fes_fraction", True)
        if fraction:
            fes_fraction = get_fes_component_fraction(component_name, n)
            logger.info(f"  Removing {fes_fraction*100:.1f}% of base demand for {component_name}")

            for load_idx in n.loads.index:
                if load_idx in n.loads_t.p_set.columns:
                    n.loads_t.p_set[load_idx] *= (1 - fes_fraction)

        # Add disaggregated loads
        for _, row in allocation.iterrows():
            bus = row["bus"]

            # Create load name
            load_name = f"{bus} {component_name}"

            # Add load
            n.add("Load",
                  load_name,
                  bus=bus,
                  carrier=component_name,
                  p_set=profile[bus].values if bus in profile.columns else 0
            )

            logger.info(f"  Added load: {load_name}")

    logger.info(f"Disaggregation complete: {len(n.loads)} total loads")
    return n
```

### 4. Update rules/generators.smk

Change renewable integration input:

**Current**:
```python
input:
    network=f"{resources_path}/network/{{scenario}}_network_demand.pkl"
```

**New** (no change needed - still uses `_demand.pkl`, but now it has complete demand-side)

### 5. Update rules/solve.smk

#### A. Modify `finalize_network` rule

**Current**:
```python
input:
    network=lambda wildcards: (
        _clustered_network_output(wildcards.scenario)
        if _is_clustering_enabled(wildcards.scenario)
        else f"{resources_path}/network/{wildcards.scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors_flexibility.nc"
    )
```

**New**:
```python
input:
    network=lambda wildcards: (
        _clustered_network_output(wildcards.scenario)
        if _is_clustering_enabled(wildcards.scenario)
        else f"{resources_path}/network/{wildcards.scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    )
```

(Remove `_flexibility` suffix since flexibility is now in `_demand`)

#### B. Remove flexibility integration rule

The `add_demand_flexibility` rule is no longer needed as standalone step.

## Migration Steps

1. ✅ **Fix 1: Heat pump year mismatch** (ALREADY DONE)
   - Fixed heat_pumps.py reindexing logic

2. ✅ **Fix 2: Empty storage CSV** (ALREADY DONE)
   - Fixed solve_network.py to always create storage CSV

3. **Implement new workflow**:
   a. Create `scripts/demand/finalize_demand_integration.py`
   b. Update `scripts/demand/integrate.py` with disaggregation function
   c. Modify `rules/demand.smk`:
      - Rename `integrate_demand` → `integrate_demand_base`
      - Create `finalize_demand` rule
      - Deprecate old flexibility rule
   d. Update `rules/solve.smk` to remove `_flexibility` suffix
   e. Test with HT35_flex scenario

4. **Clean up old files** (after testing):
   - Remove old `_final.nc` networks
   - Remove old `_flexibility.nc` intermediate files

## Testing Plan

1. Delete all HT35_flex network files
2. Run: `snakemake resources/network/HT35_flex.nc --cores 1 -f`
3. Verify:
   - `HT35_flex_network_demand.pkl` has loads + flexibility components + 0 generators ✓
   - `HT35_flex_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc` has all generators ✓
   - `HT35_flex.nc` has all generators + all flexibility ✓
4. Run solve and verify non-zero generation from actual generators

## Expected Results After Fix

- Final network should have:
  - ~5,000 generators (not just 941 load shedding)
  - ~1,250 loads (including disaggregated HP/EV)
  - ~940 links (flexibility heat pump connections)
  - ~620 stores (thermal storage)
- Solve should show:
  - Renewable generation (wind, solar)
  - Thermal generation (CCGT, nuclear, biomass, etc.)
  - Load shedding < 1% (not 87%!)

---

## Next Actions

**Immediate**: Implement the new workflow structure outlined above.

**Future**: Consider consolidating the entire demand pipeline into a single comprehensive rule with better internal orchestration.
