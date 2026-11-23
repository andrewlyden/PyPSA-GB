"""Explicit CP30 operational assumptions.

The CP30 workbook provides scenario totals but not a full operational-cost or
hydrogen technical dataset.  Keep these assumptions centralised so CP30 runs do
not silently inherit FES capacities or hidden defaults.
"""

from pathlib import Path
from typing import Any

import pandas as pd


CP30_CARRIER_MARGINAL_COST_OVERRIDES = {
    "CCS Gas": 100.0,
    "CCS Biomass": 120.0,
    "Hydrogen": 80.0,
}

CP30_ELECTROLYSIS_FULL_LOAD_HOURS = 4000.0

CP30_FLEXIBILITY_DECISION = (
    "CP30 workbook V003 does not provide PyPSA-ready operational flexibility "
    "time series or constraints. CP30 scenarios therefore do not automatically "
    "enable demand-side flexibility; storage and hydrogen are represented by "
    "their explicit CP30 ES1 totals and documented technical assumptions."
)


def is_cp30_config(scenario_config: dict[str, Any] | None) -> bool:
    """Return True when a scenario's targets come from CP30 (its carriers are CP30's ES1 lines)."""
    from scripts.utilities.scenario_detection import target_source

    return target_source(scenario_config or {}) == "CP30"


def get_cp30_marginal_cost_overrides(scenario_config: dict[str, Any] | None) -> dict[str, float]:
    """Return CP30-specific carrier marginal-cost overrides.

    Defaults are existing PyPSA-GB fuel-row assumptions for the clean
    dispatchable CP30 carriers. Scenario YAML may override them under:

        marginal_costs:
          cp30_carrier_overrides:
            Hydrogen: 90.0
    """
    if not is_cp30_config(scenario_config):
        return {}

    overrides = CP30_CARRIER_MARGINAL_COST_OVERRIDES.copy()
    configured = (
        (scenario_config or {})
        .get("marginal_costs", {})
        .get("cp30_carrier_overrides", {})
    )
    for carrier, value in configured.items():
        overrides[str(carrier)] = float(value)
    return overrides


def load_cp30_electrolysis_demand_gwh(
    es1_path: str | Path,
    pathway: str,
    modelled_year: int,
) -> float:
    """Load CP30 annual electrolysis demand from ES1 in GWh."""
    path = Path(es1_path)
    if not path.exists():
        raise FileNotFoundError(f"CP30 ES1 file not found: {path}")

    df = pd.read_csv(path)
    year_col = str(modelled_year)
    required = {"Pathway", "Variable", year_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CP30 ES1 missing required columns: {sorted(missing)}")

    rows = df[
        (df["Pathway"] == pathway)
        & (df["Variable"].astype(str).str.strip() == "Electrolysis Demand (GWh)")
    ].copy()
    if rows.empty:
        raise ValueError(f"No CP30 electrolysis demand row for pathway '{pathway}'")

    values = pd.to_numeric(rows[year_col], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric CP30 electrolysis demand for {pathway} {modelled_year}")

    return float(values.sum())



def load_cp30_grid_connected_electrolysis_capacity_mw(
    ed1_path: str | Path,
    pathway: str,
    modelled_year: int,
) -> float:
    """Load ED1 grid-connected electrolysis installed electric capacity in MW."""
    path = Path(ed1_path)
    if not path.exists():
        raise FileNotFoundError(f"CP30 ED1 file not found: {path}")

    df = pd.read_csv(path)
    year_col = str(modelled_year)
    required = {"Data item", "Unit", "Pathway", year_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CP30 ED1 missing required columns: {sorted(missing)}")

    rows = df[
        (df["Pathway"] == pathway)
        & (
            df["Data item"].astype(str).str.strip()
            == "Grid-connected electrolysis capacity (installed electric capacity)"
        )
    ].copy()
    if rows.empty:
        raise ValueError(f"No CP30 ED1 electrolysis capacity row for pathway '{pathway}'")

    values = pd.to_numeric(rows[year_col], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric CP30 ED1 electrolysis capacity for {pathway} {modelled_year}")

    total = float(values.sum())
    units = {str(v).strip().lower() for v in rows["Unit"].dropna()}
    if units == {"gw"}:
        return total * 1000.0
    if units == {"mw"}:
        return total
    raise ValueError(f"Unsupported CP30 ED1 electrolysis capacity unit(s): {sorted(units)}")

def derive_cp30_electrolysis_capacity_mw(
    es1_path: str | Path,
    pathway: str,
    modelled_year: int,
    full_load_hours: float = CP30_ELECTROLYSIS_FULL_LOAD_HOURS,
) -> float:
    """Derive electrolyser MW from CP30 annual electrolysis demand.

    CP30 ES1 reports electrolysis demand as annual electricity consumption, not
    installed electrolyser capacity.  The conversion is:

        capacity_mw = demand_gwh * 1000 / full_load_hours
    """
    if full_load_hours <= 0:
        raise ValueError("full_load_hours must be positive")

    demand_gwh = load_cp30_electrolysis_demand_gwh(es1_path, pathway, modelled_year)
    return demand_gwh * 1000.0 / float(full_load_hours)

def load_or_derive_cp30_electrolysis_capacity_mw(
    ed1_path: str | Path,
    es1_path: str | Path,
    pathway: str,
    modelled_year: int,
    full_load_hours: float = CP30_ELECTROLYSIS_FULL_LOAD_HOURS,
) -> tuple[float, str]:
    """Return CP30 electrolyser capacity and the source used.

    Prefer ED1 installed electric capacity. If the row is absent, derive capacity
    from CP30 ES1 annual electrolysis demand using explicit full-load hours.
    """
    try:
        capacity = load_cp30_grid_connected_electrolysis_capacity_mw(ed1_path, pathway, modelled_year)
        return capacity, "CP30_ED1_grid_connected_electrolysis_capacity"
    except ValueError:
        capacity = derive_cp30_electrolysis_capacity_mw(
            es1_path,
            pathway,
            modelled_year,
            full_load_hours=full_load_hours,
        )
        return capacity, "CP30_ES1_electrolysis_demand_derived"
