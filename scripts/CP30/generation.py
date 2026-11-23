"""Clean Power 2030 generation capacity helpers.

CP30 provides national capacity totals in ES1.  It does not provide
generator-level or bus-level placements, so integration code should use these
helpers for authoritative MW totals and apply an explicit spatial proxy
separately.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CP30TechnologyMapping:
    category: str
    type: str
    subtype: str
    carrier: str
    group: str
    # Proxy carriers are tried in order; the first with any capacity places
    # the target. (Until 23 Sep 2026 they were pooled, which spread CHP over
    # CCGT rows and oil over OCGT and CCGT rows.)
    proxy_carriers: tuple[str, ...]
    # A CP30 line that covers several PyPSA carriers. The target is divided in
    # proportion to the proxy's own capacity of each, and each part is placed
    # on that carrier's proxy rows (item MR2).
    split_carriers: tuple[str, ...] = ()


# CP30 ES1 publishes unabated gas as one "Gas" line with no technology split;
# FES carries CCGT, OCGT and gas engines separately, so the line is split on the
# FES proxy's own shares instead of being labelled CCGT whole.
GAS_SPLIT_CARRIERS = ("CCGT", "OCGT", "gas_engine")

CP30_GENERATION_MAP: tuple[CP30TechnologyMapping, ...] = (
    CP30TechnologyMapping("Renewable", "Offshore Wind", "Offshore Wind", "wind_offshore", "renewable", ("wind_offshore",)),
    CP30TechnologyMapping("Renewable", "Onshore Wind", "Onshore Wind", "wind_onshore", "renewable", ("wind_onshore",)),
    CP30TechnologyMapping("Renewable", "Solar", "Solar PV", "solar_pv", "renewable", ("solar_pv",)),
    CP30TechnologyMapping("Renewable", "Hydro", "Hydro", "large_hydro", "renewable", ("large_hydro", "hydro")),
    CP30TechnologyMapping("Renewable", "Marine", "Marine", "marine", "renewable", ("marine",)),
    CP30TechnologyMapping("Thermal", "Gas", "Gas", "gas", "thermal", ("CCGT",), GAS_SPLIT_CARRIERS),
    CP30TechnologyMapping("Thermal", "Gas", "Gas CHP", "CHP", "thermal", ("CHP", "CCGT")),
    CP30TechnologyMapping("Low Carbon", "Nuclear", "Nuclear - Large", "nuclear", "thermal", ("nuclear",)),
    CP30TechnologyMapping("Low Carbon", "Nuclear", "Nuclear - SMR", "nuclear", "thermal", ("nuclear",)),
    CP30TechnologyMapping("Low Carbon", "CCS", "CCS Biomass", "CCS Biomass", "thermal", ("biomass",)),
    CP30TechnologyMapping("Low Carbon", "CCS", "CCS Gas", "CCS Gas", "thermal", ("CCGT",)),
    CP30TechnologyMapping("Thermal", "Hydrogen", "Hydrogen", "Hydrogen", "thermal", ("Hydrogen", "H2", "CCGT")),
    CP30TechnologyMapping("Thermal", "Hydrogen", "Hydrogen CHP", "Hydrogen", "thermal", ("Hydrogen", "H2", "CHP", "CCGT")),
    CP30TechnologyMapping("Renewable", "Biomass", "Biomass", "biomass", "thermal", ("biomass",)),
    CP30TechnologyMapping("Renewable", "Biomass", "Biomass CHP", "biomass", "thermal", ("biomass",)),
    CP30TechnologyMapping("Renewable", "Waste", "Waste", "waste_to_energy", "thermal", ("waste_to_energy", "waste", "biomass")),
    CP30TechnologyMapping("Renewable", "Waste", "Waste CHP", "waste_to_energy", "thermal", ("waste_to_energy", "waste", "biomass")),
    CP30TechnologyMapping("Thermal", "Other Thermal", "Fuel Oil", "oil", "thermal", ("oil", "OCGT", "CCGT")),
    CP30TechnologyMapping("Thermal", "Other Thermal", "Diesel", "oil", "thermal", ("oil", "OCGT", "CCGT")),
    CP30TechnologyMapping("Thermal", "Coal", "Coal", "coal", "thermal", ("coal",)),
    # ES1 "Other Renewables" (FFR 2030: 1,071 MW, 4.69 TWh, CF 0.50, the same
    # capacity every year 2023-2030) is distribution-connected renewable engines:
    # landfill gas, sewage gas and AD biogas, FES building block Gen_BB004. It had
    # no row here and was skipped silently until 26 Sep 2026, so 4.7 TWh of
    # baseload supply was missing. Placed on the FES Gen_BB004 rows, which the
    # FES loader labels biogas; the other names cover REPD-derived proxies.
    CP30TechnologyMapping("Renewable", "Other Renewables", "Other Renewables", "biogas", "thermal",
                          ("biogas", "landfill_gas", "sewage_gas")),
)

# ES1 capacity lines that are not generation and are placed by their own
# modules (storage_interconnectors.py). Every other positive capacity line must
# map to CP30_GENERATION_MAP, or load_cp30_generation_capacities refuses it.
CP30_NON_GENERATION_CATEGORIES = frozenset({"Storage", "Interconnectors"})

# ES1 "Generation (GWh)" rows that are keyed differently from their capacity
# row: hydrogen generation is published under Type "Low Carbon".
CP30_GENERATION_ENERGY_ALIASES: dict[tuple[str, str, str], tuple[str, str, str]] = {
    ("Thermal", "Low Carbon", "Hydrogen"): ("Thermal", "Hydrogen", "Hydrogen"),
}


def _mapping_lookup() -> dict[tuple[str, str, str], CP30TechnologyMapping]:
    return {(m.category, m.type, m.subtype): m for m in CP30_GENERATION_MAP}


def cp30_row_key(row: pd.Series) -> tuple[str, str, str]:
    return (str(row["Category"]).strip(), str(row["Type"]).strip(), str(row["SubType"]).strip())


def mapping_for_es1_row(row: pd.Series) -> CP30TechnologyMapping | None:
    """Generation mapping for an ES1 capacity or generation row, or None."""
    key = cp30_row_key(row)
    return _mapping_lookup().get(CP30_GENERATION_ENERGY_ALIASES.get(key, key))


def unmapped_cp30_lines(
    es1: pd.DataFrame, pathway: str, modelled_year: int, variables: Iterable[str] = ("Capacity (MW)",),
) -> pd.DataFrame:
    """ES1 supply lines with a positive value that no mapping places.

    Storage and interconnector capacity lines are excluded: they are not
    generation and are placed by ``storage_interconnectors.py``.
    """
    year_col = str(modelled_year)
    rows = es1[(es1["Pathway"] == pathway) & es1["Variable"].isin(list(variables))].copy()
    rows = rows[~rows["Category"].astype(str).str.strip().isin(CP30_NON_GENERATION_CATEGORIES)]
    rows["_value"] = pd.to_numeric(rows[year_col], errors="coerce")
    rows = rows[rows["_value"].gt(0.0)]
    mask = rows.apply(lambda row: mapping_for_es1_row(row) is None, axis=1) if len(rows) else pd.Series(dtype=bool)
    return rows.loc[mask] if len(rows) else rows


def _read_cp30_es1(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CP30 ES1 file not found: {path}")
    return pd.read_csv(path)


def load_cp30_generation_capacities(
    es1_path: str | Path,
    pathway: str,
    modelled_year: int,
    groups: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return CP30 ES1 generation capacity totals by PyPSA carrier.

    The result is aggregated across CP30 connection classes, so Transmission and
    Distributed rows are both included where present.
    """
    df = _read_cp30_es1(es1_path)
    year_col = str(modelled_year)
    required = {"Pathway", "Variable", "Category", "Type", "SubType", year_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CP30 ES1 missing required columns: {sorted(missing)}")

    selected_groups = set(groups) if groups is not None else None
    lookup = _mapping_lookup()
    rows = []

    unmapped = unmapped_cp30_lines(df, pathway, modelled_year)
    if not unmapped.empty:
        detail = unmapped[["Connection", "Category", "Type", "SubType", year_col]].to_string(index=False)
        raise ValueError(
            "CP30 ES1 capacity lines with no PyPSA carrier mapping (add them to "
            f"CP30_GENERATION_MAP rather than dropping their capacity):\n{detail}"
        )

    capacity = df[(df["Pathway"] == pathway) & (df["Variable"] == "Capacity (MW)")].copy()
    for _, row in capacity.iterrows():
        key = (
            str(row["Category"]).strip(),
            str(row["Type"]).strip(),
            str(row["SubType"]).strip(),
        )
        mapping = lookup.get(key)
        if mapping is None:
            continue
        if selected_groups is not None and mapping.group not in selected_groups:
            continue
        value = pd.to_numeric(row[year_col], errors="coerce")
        if pd.isna(value) or float(value) <= 0.0:
            continue
        rows.append(
            {
                "cp30_category": mapping.category,
                "cp30_type": mapping.type,
                "cp30_subtype": mapping.subtype,
                "carrier": mapping.carrier,
                "group": mapping.group,
                "capacity_mw": float(value),
                "connection": row.get("Connection"),
                "pathway": pathway,
                "modelled_year": int(modelled_year),
                "proxy_carriers": mapping.proxy_carriers,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "carrier",
                "group",
                "capacity_mw",
                "pathway",
                "modelled_year",
                "proxy_carriers",
                "split_carriers",
            ]
        )

    result = pd.DataFrame(rows)
    grouped = result.groupby(["carrier", "group", "pathway", "modelled_year"], as_index=False).agg(
        capacity_mw=("capacity_mw", "sum"),
        cp30_category=("cp30_category", lambda values: "; ".join(sorted(set(values)))),
        cp30_type=("cp30_type", lambda values: "; ".join(sorted(set(values)))),
        cp30_subtype=("cp30_subtype", lambda values: "; ".join(sorted(set(values)))),
        connection=("connection", lambda values: "; ".join(sorted({str(v) for v in values if pd.notna(v)}))),
    )

    proxy_by_carrier = {
        mapping.carrier: mapping.proxy_carriers for mapping in CP30_GENERATION_MAP if mapping.group in (selected_groups or {mapping.group})
    }
    grouped["proxy_carriers"] = grouped["carrier"].map(proxy_by_carrier)
    split_by_carrier = {mapping.carrier: mapping.split_carriers for mapping in CP30_GENERATION_MAP}
    grouped["split_carriers"] = grouped["carrier"].map(split_by_carrier)
    return grouped.sort_values(["group", "carrier"]).reset_index(drop=True)


def target_member_carriers(target: pd.Series) -> tuple[str, ...]:
    """PyPSA carriers that together make up one CP30 target row."""
    split = target.get("split_carriers")
    return tuple(split) if isinstance(split, (tuple, list)) and len(split) else (target["carrier"],)


def _positive_rows(proxy_df: pd.DataFrame, proxy_carrier_col: str, capacity_col: str, carrier: str) -> pd.DataFrame:
    rows = proxy_df[proxy_df[proxy_carrier_col].eq(carrier)].copy()
    rows[capacity_col] = pd.to_numeric(rows[capacity_col], errors="coerce").fillna(0.0)
    return rows[rows[capacity_col] > 0.0]


def split_cp30_target(
    target: pd.Series, proxy_df: pd.DataFrame, proxy_carrier_col: str, capacity_col: str = "capacity_mw",
) -> dict[str, float]:
    """Divide a split CP30 target across its carriers on the proxy's shares."""
    carriers = target_member_carriers(target)
    proxy_mw = {
        carrier: float(_positive_rows(proxy_df, proxy_carrier_col, capacity_col, carrier)[capacity_col].sum())
        for carrier in carriers
    }
    total = sum(proxy_mw.values())
    if total <= 0.0:
        return {}
    return {carrier: float(target["capacity_mw"]) * mw / total for carrier, mw in proxy_mw.items() if mw > 0.0}


def scale_proxy_to_cp30_targets(
    proxy_df: pd.DataFrame,
    targets: pd.DataFrame,
    proxy_carrier_col: str,
    output_carrier_col: str,
    capacity_col: str = "capacity_mw",
    name_prefix: str = "CP30",
) -> pd.DataFrame:
    """Scale proxy rows so each carrier exactly matches the CP30 target MW."""
    if proxy_df.empty or targets.empty:
        return pd.DataFrame()
    if proxy_carrier_col not in proxy_df.columns:
        raise ValueError(f"Proxy dataframe missing carrier column: {proxy_carrier_col}")
    if capacity_col not in proxy_df.columns:
        raise ValueError(f"Proxy dataframe missing capacity column: {capacity_col}")

    scaled_rows = []
    for _, target in targets.iterrows():
        target_capacity = float(target["capacity_mw"])
        if len(target_member_carriers(target)) > 1:
            # One CP30 line, several carriers: each part on its own proxy rows.
            placements = [
                (carrier, capacity, (carrier,))
                for carrier, capacity in split_cp30_target(target, proxy_df, proxy_carrier_col, capacity_col).items()
            ]
        else:
            proxy_carriers = tuple(target.get("proxy_carriers") or (target["carrier"],))
            placements = [(target["carrier"], target_capacity, proxy_carriers)]

        for output_carrier, capacity, proxy_carriers in placements:
            carrier_proxy = pd.DataFrame()
            for proxy_carrier in proxy_carriers:
                carrier_proxy = _positive_rows(proxy_df, proxy_carrier_col, capacity_col, proxy_carrier)
                if not carrier_proxy.empty:
                    break
            if carrier_proxy.empty:
                continue

            proxy_total = float(carrier_proxy[capacity_col].sum())
            carrier_proxy[capacity_col] = carrier_proxy[capacity_col] / proxy_total * capacity
            carrier_proxy[output_carrier_col] = output_carrier
            carrier_proxy["data_source"] = "CP30"
            carrier_proxy["spatial_proxy_source"] = "FES"
            carrier_proxy["cp30_pathway"] = target["pathway"]
            carrier_proxy["cp30_target_mw"] = target_capacity
            carrier_proxy["cp30_carrier_target_mw"] = capacity
            carrier_proxy["cp30_source_type"] = target.get("cp30_type")
            carrier_proxy["cp30_source_subtype"] = target.get("cp30_subtype")
            carrier_proxy["cp30_proxy_carriers"] = ",".join(proxy_carriers)
            names = [f"{name_prefix}_{output_carrier}_{idx}" for idx in range(len(carrier_proxy))]
            if "site_name" in carrier_proxy.columns:
                carrier_proxy["site_name"] = names
            if "station_name" in carrier_proxy.columns or output_carrier_col == "fuel_type":
                carrier_proxy["station_name"] = names

            scaled_rows.append(carrier_proxy)

    if not scaled_rows:
        return pd.DataFrame()
    return pd.concat(scaled_rows, ignore_index=True)
