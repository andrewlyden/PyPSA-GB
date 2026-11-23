"""CP30 energy validation: build-time availability and post-solve dispatch.

``scripts/CP30/validation.py`` checks capacity only, so a network could match
every CP30 MW total and still miss the pathway's energy by a wide margin; the
26 Sep 2026 audit found onshore wind available at 78.6 TWh against CP30's
70.2 TWh gross, waste running at CF 0.76 against 0.59, a flat hydro
placeholder, marine at twice its energy, and 4.7 TWh of "Other Renewables"
dropped silently. This module reconciles energy on two bases:

* **build** (``stage="build"``): AVAILABLE energy by technology, before
  curtailment, against CP30 GROSS energy (ES1 generation + ES1 curtailment),
  because curtailment is a model output and must not be calibrated away; plus
  the network's demand against CP30 demand.
* **dispatch** (``stage="dispatch"``): the FINAL balancing dispatch summed over
  the solved months against ES1 generation, model curtailment against ES1
  curtailment, demand, storage, interconnector and electrolysis lines.

Every positive ES1 supply line must be mapped to a model carrier: an unmapped
line FAILS, as does a demand mismatch. Energy differences WARN only: they are
results (or, at build time, a calibration signal), not errors.

Large hydro is reservoir-constrained (``scripts/solve/hydro_constraints.py``):
its available energy is the INFLOW, ``p_max_pu x p_nom x
inflow_profile_factor``, not the ``p_max_pu`` ceiling; or, on a network built
with ``hydro.large_hydro.power_ceiling: availability``, the inflow series
``generators_t.hydro_inflow_pu`` x p_nom (OW3, 3 Oct 2026).

Target sources (task Y1, 30 Sep 2026): :func:`energy_targets` takes the source,
pathway and year and returns the same target schema for either workbook:

* ``"CP30"``: the CP30 ES1 CSV, through :func:`cp30_energy_targets` (unchanged);
* ``"FES2025"``: the FES 2025 Data Workbook, through
  ``scripts/FES/fes2025_targets.py`` (ES1 in TWh, demand from ED1, fiscal to
  calendar).

:func:`build_energy_validation` labels its notes with the target source, and
names the target column ``cp30_gwh`` for CP30 (as before) and ``target_gwh``
for any other source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.CP30.generation import (
    GAS_SPLIT_CARRIERS,
    mapping_for_es1_row,
    unmapped_cp30_lines,
)
from scripts.CP30.storage_interconnectors import CP30_STORAGE_MAP
from scripts.solve.hydro_constraints import (
    LARGE_HYDRO_INFLOW_SERIES,
    _existing_pumped_hydro_site_names,
    _generator_series,
)
from scripts.demand.demand_carriers import (
    CONSUMER_FLEX_CARRIERS,
    DACCS,
    DEMAND_COMPONENT_LOAD_CARRIERS,
    DEMAND_LINE_OF_CARRIER,
    IC_DSR,
    RESIDENTIAL_DSR,
)


# Tolerances. Suggested config home: validation.energy in config/defaults/method/brownfield.yaml;
# these code defaults are what applies until it exists.
DEFAULT_TOLERANCES = {
    "available_rel_tol": 0.05,     # build: available vs CP30 gross (WARN)
    "dispatch_rel_tol": 0.10,      # dispatch: generation vs ES1 generation (WARN)
    "curtailment_rel_tol": 0.25,   # dispatch: curtailment vs ES1 curtailment (WARN)
    "demand_rel_tol": 0.005,       # both: network demand vs CP30 demand (FAIL)
    "min_abs_gwh": 100.0,          # differences below this never warn
}

# Model carriers that together make up one CP30 generation carrier.
MEMBER_CARRIERS = {"gas": GAS_SPLIT_CARRIERS}
# FES 2025 (task Y14): ES1's Waste line includes the renewable engines (building block
# Gen_BB004: landfill gas, sewage gas, AD biogas), which the HT50 build carries as
# ``biogas``; so for FES 2025 targets the waste row compares waste_to_energy + biogas.
# An ES1 "Other Renewables" line, when present, is folded into the same row so biogas is
# never counted twice. CP30 targets keep MEMBER_CARRIERS exactly.
FES2025_SOURCE = "FES2025"
FES2025_MEMBER_CARRIERS = {**MEMBER_CARRIERS, "waste_to_energy": ("waste_to_energy", "biogas")}
# Model carriers that are not GB generation in the ES1 sense.
EXCLUDED_GENERATOR_CARRIERS = {"EU_market", "load_shedding"}
RESERVOIR_CARRIERS = {"large_hydro"}
CURTAILMENT_CARRIERS = {"wind_offshore", "wind_onshore", "solar_pv"}
# Resource-limited carriers whose availability IS their energy. For a
# dispatchable carrier availability is only a ceiling (CCGT: 187 TWh available
# against 15 TWh generated), so at build time it is reported, not judged.
RESOURCE_CARRIERS = CURTAILMENT_CARRIERS | {"marine", "large_hydro"}
DEMAND_VARIABLE = "Demand Total (GWh) - Excluding Electrolysis"

# GB electricity demand on the network side (task Y6a, 30 Sep 2026). Base loads carry
# "", "electricity" or "AC"; the disaggregated components carry their component name
# (scripts/demand/demand_carriers.py). Consumer flexibility is storage units: the
# electricity their drain needs (build: -inflow / efficiency_store; dispatch: p_store, net
# of p_dispatch for demand response) is demand too. DACCS is outside ED1: it is reported
# in the breakdown but not in ``demand_mwh``. A network with none of these carriers (every
# CP30 build) gives exactly the old total.
BASE_DEMAND_LOAD_CARRIERS = ("", "electricity", "AC")
GB_DEMAND_LOAD_CARRIERS = BASE_DEMAND_LOAD_CARRIERS + tuple(DEMAND_COMPONENT_LOAD_CARRIERS)
DEMAND_RESPONSE_CARRIERS = (IC_DSR, RESIDENTIAL_DSR)
DEMAND_FLEX_CARRIERS = tuple(CONSUMER_FLEX_CARRIERS) + (DACCS,)


def members_of(carrier: str, members: dict | None = None) -> tuple[str, ...]:
    return tuple((MEMBER_CARRIERS if members is None else members).get(carrier, (carrier,)))


def overlay_carved_into(scenario_config: dict | None) -> dict[str, str]:
    """{overlay carrier: CP30 line carrier} from a scenario's declared overlay carve-out.

    ``fleet.spatial_proxy.committed_overlay_carve_out`` (scripts/CP30/committed_overlay.py; user
    decision 14 Oct 2026) takes NZT's 742 MW of ``CCS Gas`` out of CP30's Gas line, so the
    energy check counts that carrier inside the line, as the capacity check does
    (``validation.py``, ``overlay_carved_into``). Empty when nothing is carved out.
    """
    from scripts.CP30.committed_overlay import carved_line_by_carrier, resolve_overlay_carve_out

    block = ((((scenario_config or {}).get("fleet") or {}).get("spatial_proxy") or {})
             .get("committed_overlay_carve_out"))
    return carved_line_by_carrier(resolve_overlay_carve_out(block))


def _with_carved_members(members: dict, carved_into: dict | None, target_carriers) -> dict:
    """``members`` with each carved overlay carrier added to the CP30 line it was carved out of.

    Refuses a carrier that also has its own ES1 line: the build refuses such an overlay
    (it can only add what CP30 omits), so counting it in two rows would be a double count.
    """
    if not carved_into:
        return members
    own = sorted(set(carved_into) & set(target_carriers))
    if own:
        raise ValueError(f"carved overlay carriers {own} also have their own ES1 line; "
                         "an overlay carve-out must name a carrier CP30 omits")
    out = dict(members)
    for carrier, line in carved_into.items():
        out[line] = tuple(dict.fromkeys(members_of(line, out) + (str(carrier),)))
    return out


def _fold_member_targets(generation: pd.DataFrame, members: dict) -> pd.DataFrame:
    """Fold a target row into the row whose members include its carrier (FES 2025 biogas into waste)."""
    generation = generation.copy()
    for carrier, group in members.items():
        if carrier not in generation.index:
            continue
        for member in group:
            if member == carrier or member not in generation.index:
                continue
            for column in ("generation_gwh", "curtailment_gwh", "gross_gwh", "capacity_mw"):
                if column in generation.columns:
                    generation.loc[carrier, column] = (float(generation.loc[carrier, column])
                                                       + float(generation.loc[member, column]))
            if "es1_lines" in generation.columns:
                lines = {*str(generation.loc[carrier, "es1_lines"]).split("; "),
                         *str(generation.loc[member, "es1_lines"]).split("; ")} - {""}
                generation.loc[carrier, "es1_lines"] = "; ".join(sorted(lines))
            generation = generation.drop(index=member)
    return generation


# ─────────────────────────────── CP30 targets ────────────────────────────────

def cp30_energy_targets(es1_path: str | Path, pathway: str, modelled_year: int) -> dict[str, Any]:
    """CP30 ES1 energy lines for a pathway and year, keyed to model carriers."""
    es1 = pd.read_csv(es1_path)
    year_col = str(modelled_year)
    rows = es1[es1["Pathway"] == pathway].copy()
    rows["_gwh"] = pd.to_numeric(rows[year_col], errors="coerce")

    generation = rows[rows["Variable"].eq("Generation (GWh)")
                      & ~rows["Category"].astype(str).str.strip().eq("Storage")]
    records = []
    for _, row in generation.iterrows():
        mapping = mapping_for_es1_row(row)
        if mapping is None or pd.isna(row["_gwh"]):
            continue
        records.append({"carrier": mapping.carrier, "es1_line": f"{row['Type']} / {row['SubType']}",
                        "connection": row.get("Connection"), "generation_gwh": float(row["_gwh"])})
    gen = pd.DataFrame(records, columns=["carrier", "es1_line", "connection", "generation_gwh"])
    gen_by_carrier = gen.groupby("carrier").agg(
        generation_gwh=("generation_gwh", "sum"),
        es1_lines=("es1_line", lambda v: "; ".join(sorted(set(v)))),
    )

    curtailment = rows[rows["Variable"].eq("Curtailment (TWh)")]
    curt_by_carrier: dict[str, float] = {}
    for _, row in curtailment.iterrows():
        mapping = mapping_for_es1_row(row)
        if mapping is not None and pd.notna(row["_gwh"]):
            curt_by_carrier[mapping.carrier] = curt_by_carrier.get(mapping.carrier, 0.0) + float(row["_gwh"]) * 1e3
    gen_by_carrier["curtailment_gwh"] = pd.Series(curt_by_carrier).reindex(gen_by_carrier.index).fillna(0.0)
    gen_by_carrier["gross_gwh"] = gen_by_carrier["generation_gwh"] + gen_by_carrier["curtailment_gwh"]

    storage_carrier = {m.subtype: m.carrier for m in CP30_STORAGE_MAP}
    storage_rows = rows[rows["Variable"].eq("Generation (GWh)") & rows["Category"].astype(str).str.strip().eq("Storage")]
    storage = {}
    for _, row in storage_rows.iterrows():
        carrier = storage_carrier.get(str(row["SubType"]).strip())
        if carrier is not None and pd.notna(row["_gwh"]):
            storage[carrier] = storage.get(carrier, 0.0) + float(row["_gwh"])

    def _scalar(variable: str) -> float:
        return float(rows.loc[rows["Variable"].eq(variable), "_gwh"].sum())

    unmapped = unmapped_cp30_lines(es1, pathway, modelled_year,
                                   variables=("Capacity (MW)", "Generation (GWh)", "Curtailment (TWh)"))
    unmapped_storage = storage_rows[storage_rows["_gwh"].gt(0)
                                    & ~storage_rows["SubType"].astype(str).str.strip().isin(storage_carrier)]
    return {
        "generation": gen_by_carrier,
        "storage_gwh": storage,
        "demand_gwh": _scalar(DEMAND_VARIABLE),
        "electrolysis_gwh": _scalar("Electrolysis Demand (GWh)"),
        "import_gwh": _scalar("Import (GWh)"),
        "export_gwh": _scalar("Export (GWh)"),
        "unmapped": pd.concat([unmapped, unmapped_storage], ignore_index=True),
    }


TARGET_SOURCES = ("CP30", "FES2025")


def energy_targets(source: str, path: str | Path, pathway: str, modelled_year: int,
                   **kwargs: Any) -> dict[str, Any]:
    """Energy targets keyed on source, pathway and year, in the one target schema.

    Args:
        source: "CP30" (``path`` is the CP30 ES1 CSV) or "FES2025" (``path`` is the FES 2025
            Data Workbook).
        path: The source file.
        pathway: Pathway name as the source prints it.
        modelled_year: Year.
        **kwargs: Source-specific options: for FES2025, ``year_basis``, ``demand_basis``,
            ``include_building_blocks`` and ``strict`` (see ``fes2025_energy_targets``).

    Returns:
        The targets dict ``build_energy_validation`` takes.
    """
    if source == "CP30":
        if kwargs:
            raise TypeError(f"CP30 targets take no options; got {sorted(kwargs)}")
        return cp30_energy_targets(path, pathway, modelled_year)
    if source == "FES2025":
        from scripts.FES.fes2025_targets import fes2025_energy_targets
        return fes2025_energy_targets(path, pathway, modelled_year, **kwargs)
    raise ValueError(f"Unknown energy target source {source!r}; expected one of {TARGET_SOURCES}")


def target_column(table: pd.DataFrame) -> str:
    """The target-energy column of a validation table: ``cp30_gwh`` or ``target_gwh``."""
    return "cp30_gwh" if "cp30_gwh" in table.columns else "target_gwh"


# ─────────────────────────────── model energy ────────────────────────────────

def _empty_tables() -> dict[str, Any]:
    return {"generators": pd.DataFrame(columns=["available_mwh", "dispatched_mwh", "p_nom"]),
            "storage_dispatch_mwh": pd.Series(dtype=float), "demand_mwh": 0.0,
            "storage_dispatch_existing_pumped_hydro_mwh": pd.Series(dtype=float),
            "load_shedding_mwh": 0.0, "import_mwh": 0.0, "export_mwh": 0.0,
            "electrolysis_mwh": 0.0, "h2_turbine_mwh": 0.0, "hours": 0.0,
            "demand_components_mwh": pd.Series(dtype=float)}


def existing_pumped_hydro_units(names: Iterable[str], hydro_config: dict | None) -> list[str]:
    """Storage units that are the configured existing pumped-hydro stations (follow-up F1, 4 Oct 2026).

    CP30 books Dinorwig, Cruachan, Ffestiniog and Foyers inside its long-duration storage, so they carry
    the ``LAES`` carrier; they are found by the name the build gives them (``..._existing_<site>_<n>``),
    as the pumped-hydro SoC floors find them (``hydro_constraints.get_pumped_hydro_names``).
    """
    sites = _existing_pumped_hydro_site_names(hydro_config)
    return [str(n) for n in names
            if any(f"_existing_{s}_" in str(n) or str(n).endswith(f"_existing_{s}") for s in sites)]


def _existing_pumped_hydro_dispatch(dispatch: pd.DataFrame, carriers: pd.Series, weights,
                                    hydro_config: dict | None) -> pd.Series:
    """Discharge (MWh) of the existing pumped-hydro stations, by the network carrier they sit in."""
    units = [u for u in existing_pumped_hydro_units(dispatch.columns, hydro_config) if u in carriers.index]
    if not units:
        return pd.Series(dtype=float)
    w = weights.to_numpy() if hasattr(weights, "to_numpy") else weights
    return dispatch[units].mul(w, axis=0).sum().groupby(carriers.loc[units].astype(str).to_numpy()).sum()


def _inflow_factor(hydro_config: dict | None) -> float | None:
    hydro_config = hydro_config or {}
    large = hydro_config.get("large_hydro", {}) or {}
    if hydro_config.get("enabled", False) and large.get("reservoir_model", False):
        return float(large.get("inflow_profile_factor", 0.55))
    return None


def _gb_demand_mask(load_carrier: pd.Series, bus_carrier: pd.Series, bus_name: pd.Series) -> pd.Series:
    carrier = load_carrier.fillna("").astype(str)
    return (carrier.isin(GB_DEMAND_LOAD_CARRIERS)
            & bus_carrier.fillna("AC").astype(str).eq("AC")
            & ~bus_name.astype(str).str.contains("External", case=False, na=False))


def _demand_line(carrier: str) -> str:
    return "residual" if carrier in BASE_DEMAND_LOAD_CARRIERS else DEMAND_LINE_OF_CARRIER.get(carrier, carrier)


def _load_energy_by_line(loads: pd.DataFrame, names: pd.Index, p_set: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Energy (MWh) of the counted loads by demand line (breakdown only; totals stay as before)."""
    carrier = loads.loc[names].get("carrier", pd.Series("", index=names)).fillna("").astype(str)
    tv = [n for n in names if n in p_set.columns]
    energy = pd.Series(0.0, index=names)
    if tv:
        energy.loc[tv] = p_set[tv].mul(np.asarray(weights, dtype=float), axis=0).sum().to_numpy()
    static = [n for n in names if n not in p_set.columns]
    if static and "p_set" in loads:
        energy.loc[static] = loads.loc[static, "p_set"].astype(float).to_numpy() * float(np.sum(weights))
    return energy.groupby(carrier.map(_demand_line).to_numpy()).sum()


def _flex_demand_energy(carriers: pd.Series, efficiency_store: pd.Series, weights: pd.Series,
                        inflow: pd.DataFrame | None = None, p_store: pd.DataFrame | None = None,
                        p_dispatch: pd.DataFrame | None = None) -> pd.Series:
    """Electricity (MWh) consumer-flexibility storage units draw, by demand line.

    Build (``inflow`` given): the drain's electricity, ``-inflow / efficiency_store``.
    Dispatch (``p_store`` given): the electricity charged; for demand response the net of
    ``p_store - p_dispatch`` (load shifted, not added).
    """
    units = carriers.index[carriers.astype(str).isin(DEMAND_FLEX_CARRIERS)]
    if not len(units):
        return pd.Series(dtype=float)
    w = np.asarray(weights, dtype=float)
    energy = pd.Series(0.0, index=units)
    if p_store is not None:
        cols = [u for u in units if u in p_store.columns]
        if cols:
            energy.loc[cols] = p_store[cols].mul(w, axis=0).sum().to_numpy()
        # Net of whatever the unit gives back (V2G and demand response), as the bill and market
        # accounting read it (market_accounting.consumer_flex_draw: p_store - p_dispatch).
        dr = [u for u in cols if p_dispatch is not None and u in p_dispatch.columns]
        if dr:
            energy.loc[dr] -= p_dispatch[dr].mul(w, axis=0).sum().to_numpy()
    elif inflow is not None:
        cols = [u for u in units if u in inflow.columns]
        if cols:
            eff = pd.to_numeric(efficiency_store.reindex(cols), errors="coerce").fillna(1.0).to_numpy()
            energy.loc[cols] = (-inflow[cols].mul(w, axis=0).sum()).to_numpy() / eff
    return energy.groupby(carriers.reindex(units).astype(str).map(_demand_line).to_numpy()).sum()


def _record_demand_components(tables: dict[str, Any], by_line: pd.Series, flex: pd.Series) -> None:
    """Store the demand breakdown (MWh by line) and add consumer flexibility, except DACCS, to ``demand_mwh``.

    ``flex`` is empty on every network without consumer-flexibility units, so the total is
    then left exactly as the loads gave it.
    """
    if len(flex):
        tables["demand_mwh"] = tables["demand_mwh"] + float(flex.drop(labels=["daccs"], errors="ignore").sum())
        by_line = by_line.add(flex, fill_value=0.0)
    tables["demand_components_mwh"] = by_line


def _generator_energy(static: pd.DataFrame, p_max_t: pd.DataFrame, p_t: pd.DataFrame | None,
                      weights: pd.Series, inflow_factor: float | None,
                      inflow_t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Available and dispatched energy per generator.

    ``inflow_t`` is the large-hydro inflow series (``generators_t.hydro_inflow_pu``,
    MW per MW) a network built with ``hydro.large_hydro.power_ceiling:
    availability`` carries: those units' available energy is their inflow, not
    their (availability) ceiling.
    """
    hours = float(weights.sum())
    pu_hours = static["p_max_pu"].astype(float) * hours
    tv = [c for c in p_max_t.columns if c in static.index]
    if tv:
        pu_hours.loc[tv] = p_max_t[tv].mul(weights.to_numpy(), axis=0).sum()
    available = pu_hours * static["p_nom"].astype(float)
    if inflow_factor is not None:
        reservoir = static["carrier"].isin(RESERVOIR_CARRIERS)
        available.loc[reservoir] = available.loc[reservoir] * inflow_factor
    if inflow_t is not None and not inflow_t.empty:
        own = [c for c in inflow_t.columns if c in static.index]
        if own:
            available.loc[own] = (inflow_t[own].mul(weights.to_numpy(), axis=0).sum()
                                  * static.loc[own, "p_nom"].astype(float))
    dispatched = pd.Series(np.nan, index=static.index)
    if p_t is not None:
        dispatched = pd.Series(0.0, index=static.index)
        cols = [c for c in p_t.columns if c in static.index]
        dispatched.loc[cols] = p_t[cols].mul(weights.to_numpy(), axis=0).sum()
    table = pd.DataFrame({"carrier": static["carrier"], "bus": static["bus"],
                          "p_nom": static["p_nom"].astype(float),
                          "available_mwh": available, "dispatched_mwh": dispatched})
    return table


def _net_external_flows(table: pd.DataFrame, p_t: pd.DataFrame | None, weights: pd.Series) -> tuple[float, float]:
    """Imports and exports from the EU_market generators, netted per external bus and hour."""
    if p_t is None:
        return 0.0, 0.0
    eu = table.index[table["carrier"].eq("EU_market")]
    eu = [g for g in eu if g in p_t.columns]
    if not eu:
        return 0.0, 0.0
    by_bus = p_t[eu].T.groupby(table.loc[eu, "bus"].to_numpy()).sum().T
    net = by_bus.mul(weights.to_numpy(), axis=0)
    return float(net.clip(lower=0).to_numpy().sum()), float((-net.clip(upper=0)).to_numpy().sum())


def network_energy_tables(network: Any, hydro_config: dict | None = None,
                          include_dispatch: bool = False) -> dict[str, Any]:
    """Energy tables from a PyPSA network (build time: availability and demand)."""
    tables = _empty_tables()
    weights = network.snapshot_weightings.generators.astype(float)
    tables["hours"] = float(weights.sum())
    gens = network.generators
    p_t = network.generators_t.p if include_dispatch and not network.generators_t.p.empty else None
    table = _generator_energy(gens.assign(carrier=gens["carrier"].astype(str)),
                              network.generators_t.p_max_pu, p_t, weights, _inflow_factor(hydro_config),
                              _generator_series(network, LARGE_HYDRO_INFLOW_SERIES))
    tables["generators"] = table
    shedding = table["carrier"].eq("load_shedding")
    tables["load_shedding_mwh"] = float(table.loc[shedding, "dispatched_mwh"].fillna(0).sum())
    tables["import_mwh"], tables["export_mwh"] = _net_external_flows(table, p_t, weights)

    loads = network.loads
    if len(loads):
        mask = _gb_demand_mask(loads.get("carrier", pd.Series("", index=loads.index)),
                               loads["bus"].map(network.buses["carrier"]), loads["bus"])
        names = loads.index[mask]
        p_set = network.loads_t.p_set
        tv = [n for n in names if n in p_set.columns]
        static = [n for n in names if n not in p_set.columns]
        energy = float(p_set[tv].mul(weights, axis=0).to_numpy().sum()) if tv else 0.0
        energy += float(loads.loc[static, "p_set"].sum() * weights.sum()) if static else 0.0
        tables["demand_mwh"] = energy
        by_line = _load_energy_by_line(loads, names, p_set, weights)
    else:
        by_line = pd.Series(dtype=float)
    su = network.storage_units
    if len(su):
        if include_dispatch:
            flex = _flex_demand_energy(su["carrier"], su["efficiency_store"], weights,
                                       p_store=network.storage_units_t.get("p_store", pd.DataFrame()),
                                       p_dispatch=network.storage_units_t.get("p_dispatch", pd.DataFrame()))
        else:
            flex = _flex_demand_energy(su["carrier"], su["efficiency_store"], weights,
                                       inflow=network.storage_units_t.get("inflow", pd.DataFrame()))
    else:
        flex = pd.Series(dtype=float)
    _record_demand_components(tables, by_line, flex)
    if include_dispatch:
        tables.update(_link_and_storage_energy_from_network(network, weights, hydro_config))
    return tables


def _link_and_storage_energy_from_network(network: Any, weights: pd.Series,
                                           hydro_config: dict | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    links = network.links
    for carrier, key in (("electrolysis", "electrolysis_mwh"), ("H2_turbine", "h2_turbine_mwh")):
        names = [n for n in links.index[links["carrier"].eq(carrier)] if n in network.links_t.p0.columns]
        scale = links.loc[names, "efficiency"].astype(float) if carrier == "H2_turbine" else 1.0
        out[key] = float((network.links_t.p0[names] * scale).mul(weights, axis=0).to_numpy().sum()) if names else 0.0
    su = network.storage_units
    dispatch = network.storage_units_t.get("p_dispatch", pd.DataFrame())
    if dispatch is None or dispatch.empty:
        dispatch = network.storage_units_t.p.clip(lower=0)
    # Consumer flexibility is demand (F5): its V2G and DSR output is netted in the demand line.
    cols = [c for c in dispatch.columns if c in su.index and str(su.at[c, "carrier"]) not in DEMAND_FLEX_CARRIERS]
    out["storage_dispatch_mwh"] = (dispatch[cols].mul(weights, axis=0).sum()
                                   .groupby(su.loc[cols, "carrier"]).sum() if cols else pd.Series(dtype=float))
    out["storage_dispatch_existing_pumped_hydro_mwh"] = _existing_pumped_hydro_dispatch(
        dispatch[cols], su["carrier"], weights, hydro_config)
    return out


def dataset_energy_tables(path: str | Path, hydro_config: dict | None = None) -> dict[str, Any]:
    """Energy tables from one solved month's NetCDF, read with xarray.

    Reads the arrays directly rather than through ``pypsa.Network`` (seconds
    rather than minutes for twelve ETYS months) and never joins on snapshot
    labels, which xarray reads as integers (see the NetCDF snapshot-index trap).
    """
    import xarray as xr

    tables = _empty_tables()
    with xr.open_dataset(path) as ds:
        weights = pd.Series(ds["snapshots_generators"].values.astype(float)) if "snapshots_generators" in ds \
            else pd.Series(np.ones(ds.sizes.get("snapshots", 0)))
        tables["hours"] = float(weights.sum())

        def _static(prefix: str, columns: Iterable[str]) -> pd.DataFrame:
            index = ds[f"{prefix}_i"].values.astype(str)
            data = {c: ds[f"{prefix}_{c}"].values for c in columns if f"{prefix}_{c}" in ds}
            return pd.DataFrame(data, index=index)

        def _series(prefix: str, attr: str) -> pd.DataFrame:
            key = f"{prefix}_t_{attr}"
            if key not in ds:
                return pd.DataFrame()
            return pd.DataFrame(ds[key].values, columns=ds[f"{key}_i"].values.astype(str))

        gens = _static("generators", ["carrier", "bus", "p_nom", "p_max_pu"])
        gens["carrier"] = gens["carrier"].astype(str)
        gens["bus"] = gens["bus"].astype(str)
        if "p_max_pu" not in gens:
            gens["p_max_pu"] = 1.0
        p_t = _series("generators", "p")
        table = _generator_energy(gens, _series("generators", "p_max_pu"), p_t, weights, _inflow_factor(hydro_config),
                                  _series("generators", LARGE_HYDRO_INFLOW_SERIES))
        tables["generators"] = table
        tables["load_shedding_mwh"] = float(table.loc[table["carrier"].eq("load_shedding"), "dispatched_mwh"].sum())
        tables["import_mwh"], tables["export_mwh"] = _net_external_flows(table, p_t, weights)

        buses = _static("buses", ["carrier"])
        loads = _static("loads", ["carrier", "bus", "p_set"])
        if len(loads):
            mask = _gb_demand_mask(loads.get("carrier", pd.Series("", index=loads.index)).astype(str),
                                   loads["bus"].astype(str).map(buses.get("carrier", pd.Series(dtype=str)).astype(str)),
                                   loads["bus"].astype(str))
            names = loads.index[mask]
            p_set = _series("loads", "p_set")
            tv = [n for n in names if n in p_set.columns]
            static = [n for n in names if n not in p_set.columns]
            energy = float(p_set[tv].mul(weights.to_numpy(), axis=0).to_numpy().sum()) if tv else 0.0
            if static and "p_set" in loads:
                energy += float(loads.loc[static, "p_set"].astype(float).sum() * weights.sum())
            tables["demand_mwh"] = energy
            by_line = _load_energy_by_line(loads, names, p_set, weights)
        else:
            by_line = pd.Series(dtype=float)

        links = _static("links", ["carrier", "efficiency"])
        p0_t = _series("links", "p0")
        for carrier, key in (("electrolysis", "electrolysis_mwh"), ("H2_turbine", "h2_turbine_mwh")):
            names = [n for n in links.index[links["carrier"].astype(str).eq(carrier)] if n in p0_t.columns]
            if not names:
                tables[key] = 0.0
                continue
            # Electricity in for electrolysis; electricity out (p0 x efficiency,
            # since the solved files do not keep p1) for hydrogen turbines.
            scale = (links.loc[names, "efficiency"].astype(float).to_numpy()
                     if carrier == "H2_turbine" and "efficiency" in links else 1.0)
            tables[key] = float((p0_t[names] * scale).mul(weights.to_numpy(), axis=0).to_numpy().sum())

        su = _static("storage_units", ["carrier"])
        dispatch = _series("storage_units", "p_dispatch")
        if dispatch.empty:
            dispatch = _series("storage_units", "p").clip(lower=0)
        # Consumer flexibility is demand (F5): its V2G and DSR output is netted in the demand line.
        cols = [c for c in dispatch.columns if c in su.index and str(su.at[c, "carrier"]) not in DEMAND_FLEX_CARRIERS]
        tables["storage_dispatch_mwh"] = (
            dispatch[cols].mul(weights.to_numpy(), axis=0).sum().groupby(su.loc[cols, "carrier"].astype(str).to_numpy()).sum()
            if cols else pd.Series(dtype=float))
        tables["storage_dispatch_existing_pumped_hydro_mwh"] = _existing_pumped_hydro_dispatch(
            dispatch[cols], su["carrier"].astype(str), weights, hydro_config)
        flex = pd.Series(dtype=float)
        if len(su) and su["carrier"].astype(str).isin(DEMAND_FLEX_CARRIERS).any():
            store = _series("storage_units", "p_store")
            if store.empty:
                store = (-_series("storage_units", "p")).clip(lower=0)
            flex = _flex_demand_energy(su["carrier"].astype(str), pd.Series(1.0, index=su.index), weights,
                                       p_store=store, p_dispatch=dispatch)
        _record_demand_components(tables, by_line, flex)
    return tables


def sum_energy_tables(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum monthly tables into one period (generators summed by name)."""
    total = _empty_tables()
    if not parts:
        return total
    gens = pd.concat([p["generators"] for p in parts])
    total["generators"] = gens.groupby(level=0).agg(
        carrier=("carrier", "first"), bus=("bus", "first"), p_nom=("p_nom", "max"),
        available_mwh=("available_mwh", "sum"), dispatched_mwh=("dispatched_mwh", "sum"))
    for key in ("demand_mwh", "load_shedding_mwh", "import_mwh", "export_mwh",
                "electrolysis_mwh", "h2_turbine_mwh", "hours"):
        total[key] = float(sum(p[key] for p in parts))
    lines = [p["demand_components_mwh"] for p in parts if len(p.get("demand_components_mwh", ()))]
    total["demand_components_mwh"] = pd.concat(lines).groupby(level=0).sum() if lines else pd.Series(dtype=float)
    storage = [p["storage_dispatch_mwh"] for p in parts if len(p["storage_dispatch_mwh"])]
    total["storage_dispatch_mwh"] = pd.concat(storage).groupby(level=0).sum() if storage else pd.Series(dtype=float)
    existing = [p["storage_dispatch_existing_pumped_hydro_mwh"] for p in parts
                if len(p.get("storage_dispatch_existing_pumped_hydro_mwh", ()))]
    total["storage_dispatch_existing_pumped_hydro_mwh"] = (pd.concat(existing).groupby(level=0).sum() if existing
                                                           else pd.Series(dtype=float))
    return total


# ──────────────────────────────── comparison ─────────────────────────────────

def _row(component: str, item: str, target_gwh: float | None, model_gwh: float | None,
         rel_tol: float | None, fail: bool, min_abs_gwh: float, note: str = "",
         es1_lines: str = "") -> dict[str, Any]:
    status = "INFO"
    difference = rel = np.nan
    if target_gwh is not None and model_gwh is not None and rel_tol is not None:
        difference = model_gwh - target_gwh
        rel = difference / target_gwh if target_gwh else (np.inf if difference else 0.0)
        within = abs(rel) <= rel_tol or abs(difference) < min_abs_gwh
        status = "PASS" if within else ("FAIL" if fail else "WARN")
    return {"component": component, "item": item, "es1_lines": es1_lines,
            "cp30_gwh": target_gwh, "model_gwh": model_gwh, "difference_gwh": difference,
            "relative_difference": rel, "tolerance": rel_tol, "status": status, "note": note}


def build_energy_validation(targets: dict[str, Any], tables: dict[str, Any], stage: str,
                            tolerances: dict[str, float] | None = None,
                            carved_into: dict[str, str] | None = None) -> pd.DataFrame:
    """Reconcile CP30 energy lines with model energy (``stage`` build | dispatch).

    ``carved_into`` ({overlay carrier: CP30 line carrier}, from :func:`overlay_carved_into`)
    counts a carved-out overlay carrier inside its CP30 line (NZT's ``CCS Gas`` in ``gas``).
    """
    if stage not in {"build", "dispatch"}:
        raise ValueError("stage must be 'build' or 'dispatch'")
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    min_abs = float(tol["min_abs_gwh"])
    rows: list[dict[str, Any]] = []
    label_source = str(targets.get("source_label", "CP30"))
    members = FES2025_MEMBER_CARRIERS if targets.get("source") == FES2025_SOURCE else MEMBER_CARRIERS
    generation = (_fold_member_targets(targets["generation"], members) if members is not MEMBER_CARRIERS
                  else targets["generation"])
    members = _with_carved_members(members, carved_into, generation.index)
    carved_note = {line: f"{' + '.join(sorted(c for c, l in carved_into.items() if l == line))} carved out of "
                         f"{label_source}'s {line} line (fleet.spatial_proxy.committed_overlay_carve_out)"
                   for line in set((carved_into or {}).values())}

    for _, line in targets["unmapped"].iterrows():
        rows.append(_row("unmapped_es1_line", f"{line['Type']} / {line['SubType']}", None, None, None, True, min_abs,
                         note=f"{line['Variable']} {line.get('Connection')}: no model carrier mapping"))
        rows[-1]["status"] = "FAIL"

    gens = tables["generators"]
    gens = gens[~gens["carrier"].isin(EXCLUDED_GENERATOR_CARRIERS)]
    by_carrier = gens.groupby("carrier").agg(p_nom=("p_nom", "sum"), available=("available_mwh", "sum"),
                                             dispatched=("dispatched_mwh", "sum"))
    covered: set[str] = set()
    for carrier, target in generation.iterrows():
        present = [m for m in members_of(carrier, members) if m in by_carrier.index]
        covered.update(members_of(carrier, members))
        available = float(by_carrier.loc[present, "available"].sum()) / 1e3
        dispatched = float(by_carrier.loc[present, "dispatched"].sum()) / 1e3
        label = (carrier if len(members_of(carrier, members)) == 1
                 else f"{carrier} ({' + '.join(members_of(carrier, members))})")
        if carrier == "Hydrogen":
            covered.add("Hydrogen")
            dispatched += tables["h2_turbine_mwh"] / 1e3
            available = np.nan
        note = "inflow energy (reservoir model)" if carrier in RESERVOIR_CARRIERS else carved_note.get(carrier, "")
        if stage == "build":
            if carrier == "Hydrogen":
                rows.append(_row("available_energy", label, target["gross_gwh"], None, None, False, min_abs,
                                 "hydrogen turbines are links; dispatch validated post-solve", target["es1_lines"]))
                continue
            if carrier in RESOURCE_CARRIERS:
                rows.append(_row("available_energy", label, target["gross_gwh"], available, tol["available_rel_tol"],
                                 False, min_abs, note or ("gross = ES1 generation + curtailment"
                                                          if target["curtailment_gwh"] else
                                                          "no ES1 curtailment line: gross = generation"),
                                 target["es1_lines"]))
            else:
                rows.append(_row("available_energy", label, target["gross_gwh"], available, None, False, min_abs,
                                 "dispatchable: availability is a ceiling; judged on dispatch post-solve",
                                 target["es1_lines"]))
        else:
            rows.append(_row("dispatched_energy", label, target["generation_gwh"], dispatched, tol["dispatch_rel_tol"],
                             False, min_abs, note, target["es1_lines"]))
            if carrier in CURTAILMENT_CARRIERS or target["curtailment_gwh"]:
                rows.append(_row("curtailment", label, target["curtailment_gwh"], available - dispatched,
                                 tol["curtailment_rel_tol"], False, min_abs,
                                 "model curtailment = available - dispatched", target["es1_lines"]))
            if carrier not in {"Hydrogen"}:
                rows.append(_row("available_energy", label, target["gross_gwh"], available, None, False, min_abs,
                                 note or "context for the curtailment row", target["es1_lines"]))

    for carrier in sorted(set(by_carrier.index) - covered):
        value = by_carrier.loc[carrier, "available" if stage == "build" else "dispatched"] / 1e3
        rows.append(_row("model_only_carrier", carrier, None, float(value), None, False, min_abs,
                         f"model carrier with no {label_source} ES1 generation line (e.g. committed overlay)"))

    demand_gwh = tables["demand_mwh"] / 1e3
    rows.append(_row("demand", "Electricity demand excluding electrolysis", targets["demand_gwh"], demand_gwh,
                     tol["demand_rel_tol"], True, 0.0, "GB electricity loads (p_set) on the network",
                     str(targets.get("demand_line", DEMAND_VARIABLE))))
    if stage == "dispatch":
        rows.append(_row("demand", "Load shedding", None, tables["load_shedding_mwh"] / 1e3, None, False, min_abs,
                         f"unserved energy; {label_source} has none"))
        existing = tables.get("storage_dispatch_existing_pumped_hydro_mwh", pd.Series(dtype=float))
        for carrier, target_gwh in targets["storage_gwh"].items():
            model = float(tables["storage_dispatch_mwh"].get(carrier, 0.0)) / 1e3
            note = "ES1 storage generation"
            if float(existing.get(carrier, 0.0)) > 0:
                note += ("; CP30's long-duration storage, which includes the existing pumped hydro "
                         "(memo row below), so the comparison is on the whole carrier")
            rows.append(_row("storage_discharge", carrier, target_gwh, model, tol["dispatch_rel_tol"], False, min_abs,
                             note))
            # Follow-up F1 (4 Oct 2026): the existing pumped hydro inside the carrier, reported on its own.
            if float(existing.get(carrier, 0.0)) > 0:
                rows.append(_row("storage_discharge", f"{carrier}: of which existing pumped hydro", None,
                                 float(existing[carrier]) / 1e3, None, False, min_abs,
                                 "memo: Dinorwig, Cruachan, Ffestiniog and Foyers, which CP30 books inside its "
                                 "long-duration storage; the rest of the carrier is new-build long-duration storage"))
        for carrier, value in tables["storage_dispatch_mwh"].items():
            if carrier not in targets["storage_gwh"]:
                rows.append(_row("storage_discharge", str(carrier), None, float(value) / 1e3, None, False, min_abs,
                                 f"model storage carrier with no {label_source} ES1 line"))
        rows.append(_row("interconnectors", "Imports", targets["import_gwh"], tables["import_mwh"] / 1e3,
                         tol["dispatch_rel_tol"], False, min_abs, "net per external bus and hour"))
        rows.append(_row("interconnectors", "Exports", targets["export_gwh"], tables["export_mwh"] / 1e3,
                         tol["dispatch_rel_tol"], False, min_abs, "net per external bus and hour"))
        rows.append(_row("electrolysis", "Electrolysis demand", targets["electrolysis_gwh"],
                         tables["electrolysis_mwh"] / 1e3, tol["dispatch_rel_tol"], False, min_abs,
                         "electricity into electrolysis links"))
    result = pd.DataFrame(rows)
    result.insert(0, "stage", stage)
    if label_source != "CP30":
        result = result.rename(columns={"cp30_gwh": "target_gwh"})
    return result


def energy_validation_failures(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["status"].eq("FAIL")]


def format_energy_validation(table: pd.DataFrame) -> str:
    target = target_column(table)
    target_twh = "cp30_TWh" if target == "cp30_gwh" else "target_TWh"
    view = table[["component", "item", target, "model_gwh", "relative_difference", "status"]].copy()
    view[target_twh] = view.pop(target) / 1e3
    view["model_TWh"] = view.pop("model_gwh") / 1e3
    view["rel"] = view.pop("relative_difference")
    return view[["component", "item", target_twh, "model_TWh", "rel", "status"]].to_string(
        index=False, float_format=lambda v: f"{v:,.3f}")
