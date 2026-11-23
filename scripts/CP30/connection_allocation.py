"""Source-backed spatial allocation for CP30 future connection capacity."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pypsa


def _offshore_component_name(prefix: str, source_bus: str) -> str:
    """Create stable component names from ETYS bus IDs."""
    return f"{prefix}{str(source_bus).strip().replace('-', '_')}"


def _add_explicit_export_links(scaled, mapping, network, config, logger):
    """Put offshore proxy generation behind directional export links.

    Link capacity equals the CP30-scaled connected capacity at each retained FES
    proxy location.  This follows CP30's stated whole-year connection-capacity
    assumption without inventing a project-by-project cable schedule.  ES5
    anchor fallbacks are deliberately excluded: they are regional onshore
    allocations, not evidence of an offshore collector or export cable.
    """
    required = {"carrier", "source_bus", "landing_bus", "source"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            "Explicit CP30 offshore export-link mapping is missing required "
            f"columns: {sorted(missing)}"
        )
    mapping = mapping.copy()
    for column in ("carrier", "source_bus", "landing_bus"):
        mapping[column] = mapping[column].astype(str).str.strip()
    if mapping.duplicated(["carrier", "source_bus"]).any():
        raise ValueError("Explicit CP30 offshore export-link mapping has duplicate source buses")
    unknown = sorted(set(mapping["landing_bus"]) - set(network.buses.index))
    if unknown:
        raise ValueError(
            "Explicit CP30 offshore export-link mapping references ETYS buses "
            f"absent from this network: {unknown}"
        )
    efficiency = float(config.get("link_efficiency", 1.0))
    capacity_factor = float(config.get("export_capacity_factor", 1.0))
    same_bus_policy = str(config.get("same_bus_policy", "export_link")).lower()
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("CP30 offshore export-link efficiency must be in (0, 1]")
    if capacity_factor <= 0.0:
        raise ValueError("CP30 offshore export capacity factor must be positive")
    if same_bus_policy not in {"export_link", "direct"}:
        raise ValueError(
            "CP30 offshore same_bus_policy must be 'export_link' or 'direct'"
        )

    result = scaled.copy()
    offshore_prefix = str(config.get("offshore_bus_prefix", "CP30_OFF_"))
    link_prefix = str(config.get("link_prefix", "CP30_EXPORT_"))
    links_added = 0
    for carrier, carrier_mapping in mapping.groupby("carrier", sort=True):
        mask = result["technology"].eq(carrier)
        if not mask.any():
            raise ValueError(f"Export-link mapping carrier '{carrier}' has no FES proxy rows")
        anchor_proxy = result.get(
            "cp30_anchor_proxy", pd.Series(False, index=result.index)
        ).fillna(False).astype(bool)
        source_to_landing = carrier_mapping.set_index("source_bus")["landing_bus"].to_dict()
        source_to_evidence = carrier_mapping.set_index("source_bus")["source"].to_dict()
        direct_same_bus = pd.Series(False, index=result.index)
        if same_bus_policy == "direct":
            direct_same_bus = mask & result["bus"].astype(str).map(source_to_landing).eq(
                result["bus"].astype(str)
            )
        export_mask = mask & ~anchor_proxy & ~direct_same_bus
        source_buses = set(result.loc[export_mask, "bus"].astype(str))
        absent = sorted(source_buses - set(source_to_landing))
        if absent:
            raise ValueError(
                f"Export-link mapping for {carrier} is missing active FES proxy buses: {absent}"
            )
        if (mask & anchor_proxy).any():
            result.loc[mask & anchor_proxy, "connection_allocation_source"] = (
                "CP30 ES5 regional anchor fallback; allocated directly at the ETYS "
                "regional anchor (no offshore export link)"
            )
            result.loc[mask & anchor_proxy, "data_source"] = "CP30_ES5_regional_anchor_proxy"
        if direct_same_bus.any():
            result.loc[direct_same_bus, "connection_allocation_source"] = (
                "CP30 mapped landing bus equals source bus; retained directly without "
                "an artificial offshore export link"
            )
            result.loc[direct_same_bus, "data_source"] = "CP30_connection_allocation_direct"
        for source_bus, indices in result.loc[export_mask].groupby("bus").groups.items():
            source_bus = str(source_bus)
            landing_bus = str(source_to_landing[source_bus])
            offshore_bus = _offshore_component_name(offshore_prefix, source_bus)
            link_name = _offshore_component_name(link_prefix, source_bus)
            if offshore_bus in network.buses.index or link_name in network.links.index:
                raise ValueError(
                    f"CP30 explicit offshore component already exists: {offshore_bus} / {link_name}"
                )
            source = network.buses.loc[source_bus]
            network.add(
                "Bus", offshore_bus, v_nom=float(source["v_nom"]),
                x=float(source["x"]), y=float(source["y"]), carrier="AC", country="GB",
            )
            for coordinate in ("lat", "lon"):
                if coordinate in network.buses.columns:
                    network.buses.at[offshore_bus, coordinate] = source[coordinate]
            network.buses.at[offshore_bus, "is_offshore"] = True
            network.add(
                "Link", link_name, bus0=offshore_bus, bus1=landing_bus,
                p_nom=float(result.loc[indices, "capacity_mw"].sum()) * capacity_factor,
                p_min_pu=0.0, p_max_pu=1.0, efficiency=efficiency,
                carrier="offshore_export",
            )
            result.loc[indices, "connection_bus_original"] = source_bus
            result.loc[indices, "bus"] = offshore_bus
            result.loc[indices, "data_source"] = "CP30_connection_allocation"
            result.loc[indices, "connection_allocation_source"] = source_to_evidence[source_bus]
            result.loc[indices, "offshore_export_link"] = link_name
            links_added += 1
    logger.info(
        "CP30 explicit offshore export links: added %d links (efficiency %.3f, capacity factor %.3f).",
        links_added, efficiency, capacity_factor,
    )
    return result


def _reassign_buses_preserving_profiles(
    scaled: pd.DataFrame,
    allocation: pd.DataFrame,
    network: pypsa.Network,
    config: dict,
    logger,
) -> pd.DataFrame:
    """Move connection buses without changing proxy geography or capacity."""
    required = {"carrier", "source_bus", "landing_bus", "source"}
    missing = required - set(allocation.columns)
    if missing:
        raise ValueError(
            "Profile-preserving CP30 connection mapping is missing required columns: "
            f"{sorted(missing)}"
        )
    mapping = allocation.copy()
    for column in ("carrier", "source_bus", "landing_bus"):
        mapping[column] = mapping[column].astype(str).str.strip()
    if mapping.empty or mapping[list(required)].isna().any().any():
        raise ValueError("Profile-preserving CP30 connection mapping contains incomplete rows")
    if mapping.duplicated(["carrier", "source_bus"]).any():
        raise ValueError("Profile-preserving CP30 connection mapping has duplicate source buses")

    landing_buses = set(mapping["landing_bus"])
    unknown_buses = sorted(landing_buses - set(network.buses.index))
    if unknown_buses:
        raise ValueError(
            "Profile-preserving CP30 connection mapping references ETYS buses absent "
            f"from this network: {unknown_buses}"
        )
    min_voltage = config.get("min_bus_voltage_kv")
    if min_voltage is not None:
        voltages = network.buses.loc[sorted(landing_buses), "v_nom"]
        low_voltage = voltages[voltages < float(min_voltage)]
        if not low_voltage.empty:
            details = ", ".join(
                f"{bus} ({voltage:g} kV)" for bus, voltage in low_voltage.items()
            )
            raise ValueError(
                "Profile-preserving CP30 connection mapping violates its minimum "
                f"landing voltage of {float(min_voltage):g} kV: {details}"
            )

    result = scaled.copy()
    for carrier, rows in mapping.groupby("carrier", sort=True):
        mask = result["technology"].eq(carrier)
        if not mask.any():
            raise ValueError(f"Mapping carrier '{carrier}' has no FES spatial proxy rows")
        anchor_proxy = result.get(
            "cp30_anchor_proxy", pd.Series(False, index=result.index)
        ).fillna(False).astype(bool)
        reassign_mask = mask & ~anchor_proxy
        bus_map = rows.set_index("source_bus")["landing_bus"]
        source_buses = set(result.loc[reassign_mask, "bus"])
        missing_sources = sorted(source_buses - set(bus_map.index))
        unused_sources = sorted(set(bus_map.index) - source_buses)
        if missing_sources:
            raise ValueError(
                f"Profile-preserving CP30 mapping for {carrier} is missing active "
                f"FES proxy buses: {missing_sources}"
            )
        if unused_sources:
            logger.info(
                "CP30 profile-preserving mapping has %d inactive %s reference buses "
                "in this scenario; they are ignored: %s",
                len(unused_sources), carrier, unused_sources,
            )
        if (mask & anchor_proxy).any():
            result.loc[mask & anchor_proxy, "connection_allocation_source"] = (
                "CP30 ES5 regional anchor fallback; retained directly at the ETYS "
                "regional anchor (no FES landing-map reassignment)"
            )
            result.loc[mask & anchor_proxy, "data_source"] = "CP30_ES5_regional_anchor_proxy"
        result.loc[reassign_mask, "connection_bus_original"] = result.loc[reassign_mask, "bus"]
        result.loc[reassign_mask, "bus"] = result.loc[reassign_mask, "bus"].map(bus_map)
        # Generic ETYS BMU correction runs later in the renewable pipeline. Mark
        # this as an explicit connection choice so it does not undo the landing bus.
        result.loc[reassign_mask, "data_source"] = "CP30_connection_allocation"
        result.loc[reassign_mask, "connection_allocation_source"] = result.loc[
            reassign_mask, "connection_bus_original"
        ].map(rows.set_index("source_bus")["source"])
        logger.info(
            "CP30 profile-preserving connection mapping moved %d %s proxy sites to %d landing buses; "
            "capacity and lat/lon are unchanged.",
            reassign_mask.sum(), carrier, len(set(bus_map)),
        )
    return result


#: Connection-allocation modes that move proxy rows from their FES source bus to
#: a mapped landing bus (source_bus -> landing_bus), so a row's landing is known
#: before the regional allocation runs.
LANDING_MAP_MODES = ("reassign_bus_preserve_profile", "explicit_export_links_preserve_profile")


def landing_buses_for_regions(scaled: pd.DataFrame, allocation_config: Optional[dict]) -> pd.Series:
    """Landing bus each proxy row will be moved to, for labelling its CP30 region.

    An offshore proxy's FES source bus can be an offshore platform (Hornsea
    West, HOWW2A, 112 km offshore) whose nearest GSP polygon is not the region
    the capacity lands in (Killingholme, North England). The ES5 region is a
    property of where the capacity connects, so the regional allocation labels
    such rows by their landing bus. Returns a Series on ``scaled``'s index:
    the mapped landing bus, or NaN where no landing map applies (allocation
    disabled, another mode, a carrier or source bus the map does not list).
    """
    config = allocation_config or {}
    result = pd.Series(np.nan, index=scaled.index, dtype=object)
    if not config.get("enabled", False):
        return result
    mode = str(config.get("mode", "replace_capacity_allocation")).strip()
    if mode not in LANDING_MAP_MODES:
        return result
    path = Path(config.get("file", ""))
    if not path.is_file():
        raise FileNotFoundError(f"CP30 offshore connection allocation file not found: {path}")
    mapping = pd.read_csv(path, dtype=str)
    if missing := {"carrier", "source_bus", "landing_bus"} - set(mapping.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for column in ("carrier", "source_bus", "landing_bus"):
        mapping[column] = mapping[column].astype(str).str.strip()
    lookup = mapping.set_index(["carrier", "source_bus"])["landing_bus"]
    if lookup.index.duplicated().any():
        raise ValueError(f"{path} has duplicate (carrier, source_bus) rows")
    keys = pd.MultiIndex.from_arrays([scaled["technology"].astype(str), scaled["bus"].astype(str)])
    result[:] = lookup.reindex(keys).to_numpy()
    anchor_proxy = scaled.get("cp30_anchor_proxy", pd.Series(False, index=scaled.index)).fillna(False)
    result[anchor_proxy.astype(bool)] = np.nan
    return result


def apply_connection_allocation(
    scaled: pd.DataFrame,
    targets: pd.DataFrame,
    network: pypsa.Network,
    allocation_config: Optional[dict],
    modelled_year: int,
    logger,
) -> pd.DataFrame:
    """Replace CP30 spatial-proxy weights with an explicit allocation.

    CP30 ES1 is authoritative for national MW, but does not provide project or
    node-level capacity. An enabled allocation can alter only a carrier's
    location; each carrier total is conserved exactly and each row must state
    its source. Usually that source is an ETYS/TO/OFTO record. A paired
    topology test may instead use a clearly labelled frozen reference allocation
    to prevent the available-bus set from changing generator placement. The
    default configuration leaves this disabled.
    """
    config = allocation_config or {}
    if not config.get("enabled", False):
        return scaled

    purpose = str(config.get("purpose", "source_backed")).strip()
    if purpose != "source_backed":
        logger.warning(
            "CP30 offshore connection allocation purpose '%s' is a sensitivity, "
            "not a project-by-project connection register.",
            purpose,
        )
    path = Path(config.get("file", ""))
    if not path.is_file():
        raise FileNotFoundError(
            "CP30 offshore connection allocation is enabled but its source file "
            f"is unavailable: {path}"
        )
    allocation = pd.read_csv(path)
    mode = str(config.get("mode", "replace_capacity_allocation")).strip()
    if mode == "explicit_export_links_preserve_profile":
        return _add_explicit_export_links(scaled, allocation, network, config, logger)
    if mode == "reassign_bus_preserve_profile":
        return _reassign_buses_preserving_profiles(
            scaled, allocation, network, config, logger
        )
    if mode != "replace_capacity_allocation":
        raise ValueError(f"Unknown CP30 offshore connection allocation mode: {mode}")

    required = {"carrier", "bus", "capacity_share", "source"}
    missing = required - set(allocation.columns)
    if missing:
        raise ValueError(
            "CP30 offshore/reference allocation is missing required columns: "
            f"{sorted(missing)}"
        )
    allocation = allocation.copy()
    allocation["carrier"] = allocation["carrier"].astype(str).str.strip()
    allocation["bus"] = allocation["bus"].astype(str).str.strip()
    allocation["capacity_share"] = pd.to_numeric(
        allocation["capacity_share"], errors="coerce"
    )
    if allocation.empty or allocation[["carrier", "bus", "capacity_share", "source"]].isna().any().any():
        raise ValueError("CP30 offshore connection allocation contains no complete source-backed rows")
    unknown_buses = sorted(set(allocation["bus"]) - set(network.buses.index))
    if unknown_buses:
        raise ValueError(
            "CP30 offshore connection allocation references ETYS buses absent "
            f"from this network: {unknown_buses}"
        )
    min_bus_voltage_kv = config.get("min_bus_voltage_kv")
    if min_bus_voltage_kv is not None:
        min_bus_voltage_kv = float(min_bus_voltage_kv)
        allocated_buses = allocation["bus"].drop_duplicates()
        voltages = network.buses.loc[allocated_buses, "v_nom"]
        low_voltage = voltages[voltages < min_bus_voltage_kv]
        if not low_voltage.empty:
            details = ", ".join(
                f"{bus} ({voltage:g} kV)" for bus, voltage in low_voltage.items()
            )
            raise ValueError(
                "CP30 offshore connection allocation violates its minimum landing "
                f"voltage of {min_bus_voltage_kv:g} kV: {details}"
            )
    target_by_carrier = targets.groupby("carrier")["capacity_mw"].sum()
    replacement_rows = []
    for carrier, rows in allocation.groupby("carrier", sort=True):
        if carrier not in target_by_carrier.index:
            raise ValueError(f"Allocation carrier '{carrier}' has no CP30 ES1 target")
        share = float(rows["capacity_share"].sum())
        if not np.isclose(share, 1.0, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                f"Allocation shares for {carrier} must sum to 1.0; got {share}"
            )
        proxy_rows = scaled[scaled["technology"] == carrier]
        if proxy_rows.empty:
            raise ValueError(f"Allocation carrier '{carrier}' has no FES spatial proxy rows")
        template = proxy_rows.iloc[0].copy()
        for _, row in rows.iterrows():
            allocated = template.copy()
            bus = row["bus"]
            allocated["capacity_mw"] = float(target_by_carrier[carrier]) * float(row["capacity_share"])
            allocated["bus"] = bus
            allocated["lat"] = network.buses.at[bus, "lat"]
            allocated["lon"] = network.buses.at[bus, "lon"]
            allocated["site_name"] = f"CP30_{carrier}_connection_{bus}_{modelled_year}"
            allocated["data_source"] = "CP30_connection_allocation"
            allocated["spatial_proxy_source"] = str(row["source"])
            replacement_rows.append(allocated)
        scaled = scaled[scaled["technology"] != carrier]
        logger.info(
            "CP30 source-backed connection allocation replaced the %s proxy: %.1f MW across %d buses.",
            carrier, float(target_by_carrier[carrier]), len(rows),
        )

    return pd.concat([scaled, pd.DataFrame(replacement_rows)], ignore_index=True)
