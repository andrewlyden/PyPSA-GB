"""Apply CP30-required accelerated N1 reinforcements with ETYS parameters.

This layer is deliberately narrow.  It contains only CP30 projects for which
an Appendix B row gives both the physical terminals and electrical parameters.
It is applied after the normal ETYS date-filtered upgrade layer, so it can
bring a CP30-required 2031/33 project into the 2030 comparator without changing
the timing of unrelated ETYS works.
"""

from pathlib import Path

import pandas as pd

from scripts.network_build.branch_units import branch_values_for_network

# Units a components file may declare for a line's r, x and b.  The layer
# always hands the network pu on 100 MVA (the build unit; see
# scripts/network_build/branch_units.py), or PyPSA units if the network has
# already been converted.
_IMPEDANCE_UNIT_DIVISORS = {
    "percent_on_100MVA": 100.0,   # as published in ETYS Appendix B
    "pu_on_100MVA": 1.0,
}


def _voltage_from_bus_name(bus):
    """Return the ETYS nominal voltage encoded by a bus name's fifth character."""
    levels = {"1": 132.0, "2": 275.0, "3": 33.0, "4": 400.0, "5": 11.0, "6": 66.0}
    return levels.get(str(bus)[4:5], 400.0)


def _ensure_bus(network, bus, coordinate_reference):
    if bus in network.buses.index:
        return
    if coordinate_reference not in network.buses.index:
        raise ValueError(
            f"Cannot add CP30 reinforcement bus {bus}: coordinate reference "
            f"{coordinate_reference} is absent from the ETYS network"
        )
    source = network.buses.loc[coordinate_reference]
    network.add(
        "Bus", bus, v_nom=_voltage_from_bus_name(bus),
        x=float(source["x"]), y=float(source["y"]), carrier="AC", country="GB",
    )
    for field in ("lat", "lon"):
        if field in network.buses.columns:
            network.buses.at[bus, field] = source[field]


def _read_components(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CP30 reinforcement component file not found: {path}")
    components = pd.read_csv(path)
    required = {
        "project_code", "component", "name", "in_service_year", "bus0", "bus1",
        "bus0_coordinate_reference", "bus1_coordinate_reference", "source",
    }
    missing = required - set(components.columns)
    if missing:
        raise ValueError(
            "CP30 reinforcement component file is missing required columns: "
            f"{sorted(missing)}"
        )
    if components["name"].duplicated().any():
        raise ValueError("CP30 reinforcement component names must be unique")
    return components


def apply_cp30_reinforcement_layer(network, config, modelled_year, logger):
    """Add source-backed accelerated components due by ``modelled_year``.

    Returns a set of project codes whose every requested component is present.
    Existing components with the same names are accepted only when they connect
    the same two terminals, making repeat application safe but not silent.
    """
    config = config or {}
    if not config.get("enabled", False):
        return set()
    components = _read_components(config.get("components_file", ""))
    components["in_service_year"] = pd.to_numeric(
        components["in_service_year"], errors="raise"
    ).astype(int)
    active = components[components["in_service_year"] <= int(modelled_year)].copy()
    if active.empty:
        logger.info("CP30 accelerated-reinforcement layer: no components due by %s", modelled_year)
        return set()

    applied = set()
    for _, row in active.iterrows():
        component = str(row["component"]).strip().lower()
        name = str(row["name"]).strip()
        bus0 = str(row["bus0"]).strip()
        bus1 = str(row["bus1"]).strip()
        _ensure_bus(network, bus0, str(row["bus0_coordinate_reference"]).strip())
        _ensure_bus(network, bus1, str(row["bus1_coordinate_reference"]).strip())
        if component == "line":
            units = str(row.get("impedance_units", "")).strip()
            if units not in _IMPEDANCE_UNIT_DIVISORS:
                raise ValueError(
                    f"CP30 line {name} must declare impedance_units as one of "
                    f"{sorted(_IMPEDANCE_UNIT_DIVISORS)}, got {units!r}"
                )
            divisor = _IMPEDANCE_UNIT_DIVISORS[units]
            values = {field: float(row[field]) / divisor for field in ("r", "x", "b")}
            values["s_nom"] = float(row["s_nom"])
            values = branch_values_for_network(network, "Line", values, bus0)
            for col in ("s_nom_spring", "s_nom_summer", "s_nom_autumn"):
                value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                if pd.notna(value):
                    values[col] = float(value)
            if str(row.get("etys_licensee", "")).strip():
                values["etys_licensee"] = str(row["etys_licensee"]).strip()
            if name in network.lines.index:
                existing = network.lines.loc[name]
                if {existing["bus0"], existing["bus1"]} != {bus0, bus1}:
                    raise ValueError(f"Existing CP30 line {name} has different terminals")
            else:
                network.add(
                    "Line", name, bus0=bus0, bus1=bus1, carrier="AC",
                    under_construction=False, cp30_project_code=row["project_code"],
                    cp30_source=row["source"], **values,
                )
        elif component == "link":
            p_nom = float(row["p_nom"])
            efficiency = float(row.get("efficiency", 1.0))
            if name in network.links.index:
                existing = network.links.loc[name]
                if {existing["bus0"], existing["bus1"]} != {bus0, bus1}:
                    raise ValueError(f"Existing CP30 link {name} has different terminals")
            else:
                network.add(
                    "Link", name, bus0=bus0, bus1=bus1, p_nom=p_nom,
                    p_min_pu=-1.0, p_max_pu=1.0, efficiency=efficiency,
                    carrier="DC", under_construction=False,
                    cp30_project_code=row["project_code"], cp30_source=row["source"],
                )
        else:
            raise ValueError(f"Unsupported CP30 reinforcement component: {component}")
        applied.add(str(row["project_code"]).strip())

    logger.info(
        "CP30 accelerated-reinforcement layer applied %d components across %d projects: %s",
        len(active), len(applied), ", ".join(sorted(applied)),
    )
    return applied
