"""Fail-closed HND additions to an ETYS network and CP30 capacity allocation.

The HND Implementation Plan is a project portfolio, not a PyPSA-ready network
model.  This module deliberately requires a separately evidenced asset register
before it adds a bus, HVAC circuit, or HVDC link to ETYS.  It therefore supports
an auditable *ETYS--HND hybrid* sensitivity without implying that it is NESO's
internal HND model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.network_build.branch_units import branch_values_for_network

# Units contract for the asset register. Bus rows: ``x``/``y`` are OSGB36
# coordinates (m). Line rows: ``r_pu100``, ``x_pu100``, ``b_pu100`` are per unit
# on 100 MVA (the ETYS "% on 100 MVA" value / 100), the build unit; they are
# converted to PyPSA units with every other branch (branch_units.py). The
# register once carried a reactance column also named ``x``, which pandas read
# as ``x.1`` while the line code took the bus coordinate.
LINE_IMPEDANCE_COLUMNS = {"r": "r_pu100", "x": "x_pu100", "b": "b_pu100"}


ASSET_COLUMNS = {
    "asset_id", "component_type", "name", "bus0", "bus1", "in_service_year",
    "etys_overlap_status", "source_url", "evidence_reference",
}
OVERLAP_STATUSES = {"not_in_etys", "represented_in_etys"}


def _require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _as_text(value, field, asset_id):
    value = str(value).strip()
    if not value or value.lower() == "nan":
        raise ValueError(f"HND asset {asset_id!r} is missing {field}")
    return value


def _as_float(row, field):
    value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
    if pd.isna(value):
        raise ValueError(f"HND asset {row['asset_id']!r} is missing numeric {field}")
    return float(value)


def _read_asset_register(path):
    assets = pd.read_csv(_require_file(path, "HND hybrid asset register"))
    mangled = [c for c in assets.columns if str(c).split(".")[0] in {"r", "x", "b"} and "." in str(c)]
    if mangled:
        raise ValueError(
            f"HND hybrid asset register has duplicate impedance/coordinate headers {mangled}; "
            f"line impedances go in {sorted(LINE_IMPEDANCE_COLUMNS.values())}"
        )
    missing = ASSET_COLUMNS - set(assets.columns)
    if missing:
        raise ValueError(
            "HND hybrid asset register is missing required columns: "
            f"{sorted(missing)}"
        )
    if assets.empty:
        raise ValueError("HND hybrid asset register contains no asset rows")
    assets = assets.copy()
    for field in ASSET_COLUMNS - {"in_service_year"}:
        assets[field] = assets[field].astype(str).str.strip()
    if assets["asset_id"].duplicated().any():
        raise ValueError("HND hybrid asset IDs must be unique")
    assets["in_service_year"] = pd.to_numeric(
        assets["in_service_year"], errors="coerce"
    )
    if assets["in_service_year"].isna().any():
        raise ValueError("HND hybrid asset register has non-numeric in_service_year values")
    assets["in_service_year"] = assets["in_service_year"].astype(int)
    invalid = sorted(set(assets["etys_overlap_status"]) - OVERLAP_STATUSES)
    if invalid:
        raise ValueError(
            "HND hybrid etys_overlap_status must be one of "
            f"{sorted(OVERLAP_STATUSES)}; found {invalid}"
        )
    return assets


def _add_hnd_bus(network, row):
    asset_id = row["asset_id"]
    bus = _as_text(row["bus0"], "bus0", asset_id)
    exists = bus in network.buses.index
    represented = row["etys_overlap_status"] == "represented_in_etys"
    if exists:
        if not represented:
            raise ValueError(
                f"HND bus {bus!r} already exists in ETYS but is marked not_in_etys"
            )
        return "represented_in_etys"
    if represented:
        raise ValueError(
            f"HND bus {bus!r} is marked represented_in_etys but is absent from ETYS"
        )
    v_nom = _as_float(row, "v_nom")
    x, y = _as_float(row, "x"), _as_float(row, "y")
    network.add(
        "Bus", bus, v_nom=v_nom, x=x, y=y,
        carrier=str(row.get("carrier", "AC")).strip() or "AC", country="GB",
        hnd_asset_id=asset_id, hnd_source=row["source_url"],
    )
    for field in ("lat", "lon"):
        if field in network.buses.columns and field in row and pd.notna(row[field]):
            network.buses.at[bus, field] = float(row[field])
    return "added"


def _component_exists(network, component, name, bus0, bus1):
    table = network.lines if component == "line" else network.links
    if name not in table.index:
        return False
    existing = table.loc[name]
    if {str(existing["bus0"]), str(existing["bus1"])} != {bus0, bus1}:
        raise ValueError(
            f"Existing HND {component} {name!r} has different terminals than its register row"
        )
    return True


def _add_hnd_component(network, row):
    asset_id = row["asset_id"]
    component = _as_text(row["component_type"], "component_type", asset_id).lower()
    if component == "bus":
        return _add_hnd_bus(network, row)
    if component not in {"line", "link"}:
        raise ValueError(f"HND asset {asset_id!r} has unsupported component_type {component!r}")
    name = _as_text(row["name"], "name", asset_id)
    bus0, bus1 = _as_text(row["bus0"], "bus0", asset_id), _as_text(row["bus1"], "bus1", asset_id)
    absent = sorted({bus0, bus1} - set(network.buses.index))
    if absent:
        raise ValueError(
            f"HND {component} {name!r} references buses absent from ETYS/hybrid register: {absent}"
        )
    exists = _component_exists(network, component, name, bus0, bus1)
    represented = row["etys_overlap_status"] == "represented_in_etys"
    if exists:
        if not represented:
            raise ValueError(
                f"HND {component} {name!r} already exists in ETYS but is marked not_in_etys"
            )
        return "represented_in_etys"
    if represented:
        raise ValueError(
            f"HND {component} {name!r} is marked represented_in_etys but is absent"
        )
    common = {"hnd_asset_id": asset_id, "hnd_source": row["source_url"], "under_construction": False}
    if component == "line":
        values = {key: _as_float(row, column) for key, column in LINE_IMPEDANCE_COLUMNS.items()}
        values["s_nom"] = _as_float(row, "s_nom")
        values = branch_values_for_network(network, "Line", values, bus0)
        network.add(
            "Line", name, bus0=bus0, bus1=bus1, carrier="AC", **values, **common,
        )
    else:
        efficiency = _as_float(row, "efficiency")
        if not 0.0 < efficiency <= 1.0:
            raise ValueError(f"HND link {name!r} efficiency must be in (0, 1]")
        network.add(
            "Link", name, bus0=bus0, bus1=bus1, carrier="DC",
            p_nom=_as_float(row, "p_nom"), efficiency=efficiency,
            p_min_pu=float(row.get("p_min_pu", -1.0)),
            p_max_pu=float(row.get("p_max_pu", 1.0)), **common,
        )
    return "added"


def apply_hnd_hybrid_topology(network, config, modelled_year, audit_path, logger):
    """Apply documented HND assets due by ``modelled_year`` to the ETYS backbone.

    Each asset must declare whether ETYS already represents it.  The function
    rejects unreviewed overlap rather than silently double-counting a circuit.
    It writes an audit for both applied and deferred assets.
    """
    config = config or {}
    if not config.get("enabled", False):
        pd.DataFrame([{
            "audit_status": "hnd_hybrid_disabled",
            "message": "No HND assets were added to the ETYS backbone.",
        }]).to_csv(audit_path, index=False)
        return pd.DataFrame()

    assets = _read_asset_register(config.get("asset_register", ""))
    # Buses must be created before circuits regardless of arbitrary map/asset
    # identifiers.  Within each class retain the register's source order.
    assets = assets.assign(_component_order=assets["component_type"].str.lower().ne("bus"))
    audit = []
    for _, row in assets.sort_values(["in_service_year", "_component_order"], kind="stable").iterrows():
        record = row.to_dict()
        record.pop("_component_order", None)
        if row["in_service_year"] > int(modelled_year):
            record["audit_status"] = "deferred_after_model_year"
        else:
            record["audit_status"] = _add_hnd_component(network, row)
        audit.append(record)
    audit = pd.DataFrame(audit)
    audit.to_csv(audit_path, index=False)
    added = int(audit["audit_status"].eq("added").sum())
    logger.info(
        "ETYS--HND hybrid: added %d source-backed assets; audit written to %s",
        added, audit_path,
    )
    return audit


def apply_hnd_project_allocation(scaled, network, config, logger):
    """Allocate CP30 regional wind capacity to HND project nodes conservatively.

    The HND project's published MW sets its weight within an explicitly mapped
    CP30 region.  Every original FES proxy row is split by those weights, so
    CP30 ES5 regional capacity and the aggregate regional availability profile
    are both conserved exactly.  The crosswalk supplies the non-publicly
    derivable project-to-CP30-region and project-to-HND-node correspondence.
    """
    config = config or {}
    if not config.get("enabled", False):
        return scaled
    allocation_mode = str(config.get("allocation_mode", "reviewed_crosswalk")).strip()
    if allocation_mode not in {"reviewed_crosswalk", "public_portfolio_proxy"}:
        raise ValueError(
            "HND project allocation_mode must be 'reviewed_crosswalk' or "
            f"'public_portfolio_proxy', got {allocation_mode!r}"
        )
    portfolio = pd.read_csv(_require_file(config.get("portfolio_file", ""), "HND project portfolio"))
    crosswalk = pd.read_csv(_require_file(config.get("crosswalk_file", ""), "HND project crosswalk"))
    portfolio_required = {"map_id", "capacity_mw"}
    crosswalk_required = {"map_id", "carrier", "cp30_region", "connection_bus", "allocation_share", "source"}
    missing = portfolio_required - set(portfolio.columns)
    if missing:
        raise ValueError(f"HND project portfolio is missing columns: {sorted(missing)}")
    missing = crosswalk_required - set(crosswalk.columns)
    if missing:
        raise ValueError(f"HND project crosswalk is missing columns: {sorted(missing)}")
    if allocation_mode == "public_portfolio_proxy":
        proxy_columns = {"mapping_basis", "evidence_status"}
        missing = proxy_columns - set(crosswalk.columns)
        if missing:
            raise ValueError(
                "HND public-portfolio proxy requires explicit proxy columns: "
                f"{sorted(missing)}"
            )
    portfolio = portfolio[["map_id", "capacity_mw"]].copy()
    portfolio["map_id"] = portfolio["map_id"].astype(str).str.strip()
    portfolio["capacity_mw"] = pd.to_numeric(portfolio["capacity_mw"], errors="coerce")
    if portfolio["map_id"].duplicated().any() or portfolio["capacity_mw"].isna().any():
        raise ValueError("HND portfolio map_id values must be unique with numeric capacity_mw")
    crosswalk = crosswalk.copy()
    for field in ("map_id", "carrier", "cp30_region", "connection_bus", "source"):
        crosswalk[field] = crosswalk[field].astype(str).str.strip()
        if crosswalk[field].eq("").any() or crosswalk[field].str.lower().eq("nan").any():
            raise ValueError(f"HND project crosswalk has blank {field} values")
    crosswalk["allocation_share"] = pd.to_numeric(crosswalk["allocation_share"], errors="coerce")
    if crosswalk["allocation_share"].isna().any() or (crosswalk["allocation_share"] <= 0).any():
        raise ValueError("HND project crosswalk allocation_share must be positive numeric")
    unknown_projects = sorted(set(crosswalk["map_id"]) - set(portfolio["map_id"]))
    if unknown_projects:
        raise ValueError(f"HND project crosswalk has unknown map IDs: {unknown_projects}")
    unknown_buses = sorted(set(crosswalk["connection_bus"]) - set(network.buses.index))
    if unknown_buses:
        raise ValueError(
            "HND project crosswalk references HND/ETYS buses absent from the hybrid network: "
            f"{unknown_buses}"
        )
    share_sum = crosswalk.groupby("map_id")["allocation_share"].sum()
    invalid_share = share_sum[~np.isclose(share_sum, 1.0, rtol=0.0, atol=1e-8)]
    if not invalid_share.empty:
        raise ValueError(
            "HND project crosswalk allocation_share must sum to 1.0 per map_id: "
            f"{invalid_share.to_dict()}"
        )
    weights = crosswalk.merge(portfolio, on="map_id", validate="many_to_one")
    weights["weight_mw"] = weights["capacity_mw"] * weights["allocation_share"]
    result_parts = []
    allocated_regions = set()
    for (carrier, region), destinations in weights.groupby(["carrier", "cp30_region"], sort=True):
        original = scaled[(scaled["technology"] == carrier) & (scaled["cp30_region"] == region)]
        if original.empty:
            raise ValueError(
                f"HND allocation maps {carrier!r} in CP30 region {region!r}, but that region has no CP30 proxy capacity"
            )
        total_weight = float(destinations["weight_mw"].sum())
        if total_weight <= 0:
            raise ValueError(f"HND allocation has zero published weight for {carrier} / {region}")
        allocated_regions.add((carrier, region))
        for _, destination in destinations.iterrows():
            part = original.copy()
            fraction = float(destination["weight_mw"]) / total_weight
            part["capacity_mw"] *= fraction
            part["connection_bus_original"] = part["bus"]
            part["bus"] = destination["connection_bus"]
            part["hnd_map_id"] = destination["map_id"]
            part["hnd_connection_source"] = destination["source"]
            if allocation_mode == "public_portfolio_proxy":
                part["hnd_mapping_basis"] = destination["mapping_basis"]
                part["hnd_evidence_status"] = destination["evidence_status"]
            part["hnd_profile_method"] = "regional_fes_proxy_split_preserve_total"
            part["data_source"] = (
                "CP30_HND_public_portfolio_proxy"
                if allocation_mode == "public_portfolio_proxy"
                else "CP30_HND_hybrid_project_allocation"
            )
            result_parts.append(part)
    keep = scaled.apply(lambda row: (row.get("technology"), row.get("cp30_region")) not in allocated_regions, axis=1)
    result = pd.concat([scaled.loc[keep], *result_parts], ignore_index=True)
    before = scaled.groupby(["technology", "cp30_region"])["capacity_mw"].sum().sort_index()
    after = result.groupby(["technology", "cp30_region"])["capacity_mw"].sum().sort_index()
    if not before.equals(after):
        difference = after.subtract(before, fill_value=0.0)
        if not np.allclose(difference.values, 0.0, rtol=0.0, atol=1e-8):
            raise AssertionError(f"HND allocation failed to conserve CP30 regional MW: {difference.to_dict()}")
    logger.info(
        "HND %s allocation: split CP30 capacity across %d project/node entries in %d regional carrier groups; regional MW and profiles are conserved.",
        allocation_mode, len(weights), len(allocated_regions),
    )
    return result
