"""CP30 network/reference validation helpers.

CP30 network tables are reference metadata, not a full PyPSA branch dataset.  The
validation here therefore has two jobs:

1. Record the explicit network assumption used by CP30 scenarios.
2. Check that built network component totals still match authoritative CP30
   pathway targets after spatial proxying and integration.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.CP30.assumptions import load_or_derive_cp30_electrolysis_capacity_mw
from scripts.CP30.generation import load_cp30_generation_capacities, target_member_carriers
from scripts.CP30.storage_interconnectors import (
    load_cp30_interconnector_capacity_mw,
    load_cp30_storage_targets,
)


CP30_NETWORK_ASSUMPTION = (
    "CP30 workbook V003 N1/N2/NA1/CP.08/CP.12 data is treated as network "
    "reference metadata. CP30 scenarios use the configured PyPSA-GB network_model "
    "and existing ETYS topology/upgrade representation unless a later CP30 branch "
    "parameter mapping is implemented."
)


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return pd.read_csv(path)


def summarise_cp30_network_reference(reference_path: str | Path) -> pd.DataFrame:
    """Summarise extracted CP30 network/reference tables.

    This intentionally avoids interpreting projects as PyPSA branches. It simply
    exposes what reference material is available for audit/reporting.
    """
    df = _read_csv(reference_path, "CP30 reference")
    if "source_sheet" not in df.columns:
        raise ValueError("CP30 reference data missing source_sheet column")

    rows: list[dict[str, Any]] = []
    for sheet, group in df.groupby("source_sheet", dropna=True):
        summary: dict[str, Any] = {
            "source_sheet": sheet,
            "rows": int(len(group)),
            "non_empty_rows": int(group.dropna(how="all").shape[0]),
        }
        if sheet == "N1" and "Code" in group.columns:
            summary["network_projects"] = int(group["Code"].notna().sum())
        if sheet == "N2" and "Boundary name" in group.columns:
            summary["boundaries"] = int(group["Boundary name"].notna().sum())
        if "Latest Delivery Year (updated September 2024)" in group.columns:
            years = pd.to_numeric(group["Latest Delivery Year (updated September 2024)"], errors="coerce")
            summary["latest_delivery_year_min"] = int(years.min()) if years.notna().any() else None
            summary["latest_delivery_year_max"] = int(years.max()) if years.notna().any() else None
        rows.append(summary)

    return pd.DataFrame(rows).sort_values("source_sheet").reset_index(drop=True)


def network_assumption_record(scenario_config: dict[str, Any]) -> pd.DataFrame:
    """Return the explicit CP30 network modelling assumption as a one-row table."""
    return pd.DataFrame(
        [
            {
                "scenario_source": scenario_config.get("scenario_source"),
                "network_model": scenario_config.get("network_model"),
                "network_assumption": scenario_config.get("source_data", {}).get("network_assumption",
                                                                   CP30_NETWORK_ASSUMPTION),
                "cp30_reference_use": "reference_metadata_only",
            }
        ]
    )


def _comparison_row(
    component: str,
    item: str,
    target: float,
    actual: float,
    units: str,
    tolerance: float,
    source: str,
) -> dict[str, Any]:
    diff = float(actual) - float(target)
    passed = abs(diff) <= tolerance
    return {
        "component": component,
        "item": item,
        "target": float(target),
        "actual": float(actual),
        "difference": diff,
        "units": units,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
        "source": source,
    }


def _actual_generation_by_carrier(generators: pd.DataFrame) -> pd.Series:
    if generators is None or generators.empty:
        return pd.Series(dtype=float)
    required = {"carrier", "p_nom"}
    missing = required - set(generators.columns)
    if missing:
        raise ValueError(f"Generator table missing columns: {sorted(missing)}")
    values = generators.copy()
    values["p_nom"] = pd.to_numeric(values["p_nom"], errors="coerce").fillna(0.0)
    return values.groupby("carrier")["p_nom"].sum()


def _actual_storage_by_carrier(storage_units: pd.DataFrame) -> pd.DataFrame:
    if storage_units is None or storage_units.empty:
        return pd.DataFrame(columns=["carrier", "power_mw", "energy_mwh"])
    required = {"carrier", "p_nom", "max_hours"}
    missing = required - set(storage_units.columns)
    if missing:
        raise ValueError(f"StorageUnit table missing columns: {sorted(missing)}")
    from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS

    # Consumer flexibility is demand (F5), not storage plant: it has no ES1 storage line.
    values = storage_units[~storage_units["carrier"].astype(str).isin(DEMAND_FLEX_STORAGE_CARRIERS)].copy()
    values["p_nom"] = pd.to_numeric(values["p_nom"], errors="coerce").fillna(0.0)
    values["max_hours"] = pd.to_numeric(values["max_hours"], errors="coerce").fillna(0.0)
    values["energy_mwh"] = values["p_nom"] * values["max_hours"]
    return values.groupby("carrier", as_index=False).agg(
        power_mw=("p_nom", "sum"),
        energy_mwh=("energy_mwh", "sum"),
    )


def _actual_interconnector_capacity(links: pd.DataFrame) -> float:
    if links is None or links.empty:
        return 0.0
    if "p_nom" not in links.columns:
        raise ValueError("Link table missing p_nom column")
    names = links.index.astype(str)
    carrier = links.get("carrier", pd.Series("", index=links.index)).astype(str)
    bus0 = links.get("bus0", pd.Series("", index=links.index)).astype(str)
    bus1 = links.get("bus1", pd.Series("", index=links.index)).astype(str)
    mask = (
        names.str.startswith("IC_")
        | (
            carrier.eq("DC")
            & (bus0.str.contains("HVDC_External", case=False, na=False) | bus1.str.contains("HVDC_External", case=False, na=False))
        )
    )
    interconnectors = links.loc[mask].copy()
    interconnectors["p_nom"] = pd.to_numeric(
        interconnectors["p_nom"], errors="coerce"
    ).fillna(0.0)

    # Future market coupling represents one physical cable by paired directional
    # links. Count its shared physical rating once; legacy signed IC links do
    # not carry a pair identifier and remain additive.
    if "interconnector_pair_id" not in interconnectors.columns:
        return float(interconnectors["p_nom"].sum())

    pair_id = interconnectors["interconnector_pair_id"]
    paired = interconnectors.loc[pair_id.notna()]
    legacy = interconnectors.loc[pair_id.isna()]
    paired_capacity = (
        paired.groupby("interconnector_pair_id")["p_nom"].min().sum()
        if not paired.empty
        else 0.0
    )
    return float(paired_capacity + legacy["p_nom"].sum())


def _actual_link_capacity_by_carrier(links: pd.DataFrame, carrier_name: str) -> float:
    if links is None or links.empty:
        return 0.0
    if not {"carrier", "p_nom"}.issubset(links.columns):
        raise ValueError("Link table missing carrier or p_nom column")
    mask = links["carrier"].astype(str) == carrier_name
    return float(pd.to_numeric(links.loc[mask, "p_nom"], errors="coerce").fillna(0.0).sum())

def _actual_link_output_capacity_by_carrier(links: pd.DataFrame, carrier_name: str) -> float:
    """Output-side capacity of a carrier's links (p_nom limits the input, bus0)."""
    if links is None or links.empty:
        return 0.0
    mask = links["carrier"].astype(str) == carrier_name
    p_nom = pd.to_numeric(links.loc[mask, "p_nom"], errors="coerce").fillna(0.0)
    efficiency = (pd.to_numeric(links.loc[mask, "efficiency"], errors="coerce").fillna(1.0)
                  if "efficiency" in links.columns else 1.0)
    return float((p_nom * efficiency).sum())


def _actual_electrolysis_capacity(links: pd.DataFrame) -> float:
    return _actual_link_capacity_by_carrier(links, "electrolysis")


def validate_cp30_component_totals(
    es1_path: str | Path,
    ed1_path: str | Path,
    pathway: str,
    modelled_year: int,
    generators: pd.DataFrame | None = None,
    storage_units: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
    tolerance_mw: float = 1.0,
    tolerance_mwh: float = 1.0,
    electrolysis_full_load_hours: float = 4000.0,
    overlay_capacity: dict | None = None,
    overlay_carved_into: dict | None = None,
) -> pd.DataFrame:
    """Compare network component totals against CP30 targets.

    ``overlay_capacity`` ({carrier: MW}) is committed plant CP30 omits
    (scripts/CP30/committed_overlay.py): it is checked on its own row, so CP30's
    totals are still compared exactly. ``overlay_carved_into`` ({overlay carrier:
    CP30 line carrier}) declares an overlay carved out of a CP30 line (NZT out of
    Gas, user decision 14 Oct 2026): that line is checked on its members plus the
    overlay carrier, so CP30's total is still compared exactly.
    """
    rows: list[dict[str, Any]] = []

    gen_actual = _actual_generation_by_carrier(generators)
    overlay_capacity = {k: float(v) for k, v in (overlay_capacity or {}).items()}
    overlay_carved_into = dict(overlay_carved_into or {})
    unknown = sorted(set(overlay_carved_into) - set(overlay_capacity))
    if unknown:
        raise ValueError(f"overlay_carved_into names carriers with no overlay capacity: {unknown}")
    # Hydrogen turbines are H2 -> electricity links: CP30's capacity is electrical,
    # the output side (electrolysis stays on its input, electrical side).
    h2_turbine_capacity = _actual_link_output_capacity_by_carrier(links, "H2_turbine")
    gen_targets = load_cp30_generation_capacities(es1_path, pathway, modelled_year)
    for _, target in gen_targets.iterrows():
        carrier = target["carrier"]
        # A split line (CP30 "Gas": CCGT, OCGT and gas engines) is checked on
        # the sum of its member carriers.
        members = target_member_carriers(target) + tuple(
            c for c, line in overlay_carved_into.items() if line == carrier)
        actual_capacity = sum(gen_actual.get(member, 0.0) for member in members)
        if carrier == "Hydrogen":
            actual_capacity += h2_turbine_capacity
        rows.append(
            _comparison_row(
                "generation",
                carrier,
                target["capacity_mw"],
                actual_capacity,
                "MW",
                tolerance_mw,
                "CP30_ES1",
            )
        )

    for carrier, mw in overlay_capacity.items():
        label = (f"{carrier} (committed overlay, carved out of CP30 {overlay_carved_into[carrier]})"
                 if carrier in overlay_carved_into else f"{carrier} (committed overlay, not in CP30)")
        rows.append(_comparison_row("generation", label, mw,
                                    float(gen_actual.get(carrier, 0.0)), "MW", tolerance_mw, "committed_overlay"))

    storage_actual = _actual_storage_by_carrier(storage_units)
    storage_actual_by_carrier = storage_actual.set_index("carrier") if not storage_actual.empty else storage_actual
    storage_targets = load_cp30_storage_targets(es1_path, pathway, modelled_year)
    for _, target in storage_targets.iterrows():
        carrier = target["carrier"]
        if carrier in storage_actual_by_carrier.index:
            actual_power = storage_actual_by_carrier.loc[carrier, "power_mw"]
            actual_energy = storage_actual_by_carrier.loc[carrier, "energy_mwh"]
        else:
            actual_power = 0.0
            actual_energy = 0.0
        rows.append(_comparison_row("storage_power", carrier, target["power_mw"], actual_power, "MW", tolerance_mw, "CP30_ES1"))
        rows.append(_comparison_row("storage_energy", carrier, target["energy_mwh"], actual_energy, "MWh", tolerance_mwh, "CP30_ES1"))

    target_ic = load_cp30_interconnector_capacity_mw(es1_path, pathway, modelled_year)
    rows.append(
        _comparison_row(
            "interconnector_capacity",
            "Interconnectors",
            target_ic,
            _actual_interconnector_capacity(links),
            "MW",
            tolerance_mw,
            "CP30_ES1",
        )
    )

    target_electrolysis, electrolysis_source = load_or_derive_cp30_electrolysis_capacity_mw(
        ed1_path,
        es1_path,
        pathway,
        modelled_year,
        full_load_hours=electrolysis_full_load_hours,
    )
    rows.append(
        _comparison_row(
            "electrolysis_capacity",
            "Grid-connected electrolysis",
            target_electrolysis,
            _actual_electrolysis_capacity(links),
            "MW",
            tolerance_mw,
            electrolysis_source,
        )
    )

    return pd.DataFrame(rows)


def validate_cp30_network(
    network: Any,
    es1_path: str | Path,
    ed1_path: str | Path,
    pathway: str,
    modelled_year: int,
    **kwargs: Any,
) -> pd.DataFrame:
    """Validate a PyPSA network object against CP30 component totals."""
    return validate_cp30_component_totals(
        es1_path=es1_path,
        ed1_path=ed1_path,
        pathway=pathway,
        modelled_year=modelled_year,
        generators=getattr(network, "generators", None),
        storage_units=getattr(network, "storage_units", None),
        links=getattr(network, "links", None),
        **kwargs,
    )
