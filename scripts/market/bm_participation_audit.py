"""Audit BM eligibility and model redispatch coverage for historical ELEXON runs.

The standard BM calibration scorecard compares carrier-level prices and accepted
volumes.  This companion audit makes the comparison scope explicit: it reports
which model assets are directly BMU-mapped, eligible under the configured BM
participation policy, and active in model redispatch.  Geographic bands are a
transparent latitude-based diagnostic, not ESO constraint boundaries.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pypsa

from scripts.utilities.logging_config import setup_logging


DEFAULT_LATITUDE_BANDS = (
    {"name": "Scotland", "min_lat": 55.5},
    {"name": "North_England_Wales", "min_lat": 53.5},
    {"name": "Midlands", "min_lat": 51.5},
    {"name": "South_England_Wales", "min_lat": -90.0},
)


def _latitude_bands(config: dict[str, Any]) -> list[dict[str, float | str]]:
    """Return validated broad geographic bands, ordered north to south."""
    raw = (
        config.get("latitude_bands", DEFAULT_LATITUDE_BANDS)
        if config else DEFAULT_LATITUDE_BANDS
    )
    bands: list[dict[str, float | str]] = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item or "min_lat" not in item:
            raise ValueError(
                "market.balancing.participation_audit.latitude_bands entries "
                "must each contain name and min_lat"
            )
        bands.append({"name": str(item["name"]), "min_lat": float(item["min_lat"])})
    if not bands:
        raise ValueError("At least one participation-audit latitude band is required")
    return sorted(bands, key=lambda band: float(band["min_lat"]), reverse=True)


def _bus_latitudes(buses: pd.DataFrame) -> pd.Series:
    """Latitude of every bus: its ``lat`` column, else from ``x``/``y`` (OSGB36 metres or WGS84 degrees).

    The ETYS network carries ``lat``; the Reduced and Zonal networks have only OSGB36 ``x``/``y``, and the
    audit failed on them (KeyError 'lat', Test_Market, 19 Nov 2026).
    """
    if "lat" in buses.columns:
        return pd.to_numeric(buses["lat"], errors="coerce")
    if not {"x", "y"} <= set(buses.columns):
        return pd.Series(pd.NA, index=buses.index, dtype="Float64")
    x = pd.to_numeric(buses["x"], errors="coerce")
    y = pd.to_numeric(buses["y"], errors="coerce")
    if float((x.max() - x.min()) if x.notna().any() else 0.0) <= 1000.0:
        return y                                    # already degrees
    from pyproj import Transformer

    _, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(
        x.to_numpy(float), y.to_numpy(float))
    return pd.Series(lat, index=buses.index)


def _region_from_bus(bus: str, buses: pd.DataFrame, bands: list[dict[str, float | str]]) -> str:
    """Map a network bus to a documented broad geographic band (``buses`` carrying ``lat``)."""
    lat = pd.to_numeric(buses.loc[bus, "lat"], errors="coerce") if bus in buses.index else pd.NA
    if pd.isna(lat):
        return "Unknown"
    for band in bands:
        if float(lat) >= float(band["min_lat"]):
            return str(band["name"])
    return "Unknown"


def _eligible_components(
    components: pd.DataFrame,
    direct_matches: set[str],
    policy: dict[str, Any],
) -> pd.Index:
    """Reproduce the configured component eligibility filter for audit purposes."""
    mode = str(policy.get("mode", "all")).lower()
    index = components.index
    if mode == "all":
        eligible = set(index)
    elif mode == "none":
        eligible = set()
    elif mode == "elexon_mapped":
        eligible = set(index).intersection(direct_matches)
    else:
        raise ValueError(f"Unsupported participation mode '{mode}' in audit")

    min_p_nom_mw = float(policy.get("min_p_nom_mw", 0.0) or 0.0)
    if min_p_nom_mw > 0.0 and "p_nom" in components.columns:
        eligible &= set(components.index[components["p_nom"].fillna(0.0) >= min_p_nom_mw])

    include_carriers = set(policy.get("include_carriers", []) or [])
    if include_carriers and "carrier" in components.columns:
        eligible &= set(components.index[components["carrier"].isin(include_carriers)])

    exclude_carriers = set(policy.get("exclude_carriers", []) or [])
    if exclude_carriers and "carrier" in components.columns:
        eligible -= set(components.index[components["carrier"].isin(exclude_carriers)])

    include_names = {str(name) for name in (policy.get("include_names", []) or [])}
    if include_names:
        eligible &= include_names
    exclude_names = {str(name) for name in (policy.get("exclude_names", []) or [])}
    if exclude_names:
        eligible -= exclude_names
    return pd.Index([name for name in index if name in eligible])


def _component_frame(
    network: pypsa.Network,
    bmu_mapping: pd.DataFrame,
    participation: dict[str, Any],
    audit_config: dict[str, Any],
) -> pd.DataFrame:
    """Create one metadata record per generator or storage unit."""
    direct_matches = set()
    if "generator_name" in bmu_mapping.columns:
        direct_matches = set(bmu_mapping["generator_name"].dropna().astype(str))
    bands = _latitude_bands(audit_config)
    bus_lat = _bus_latitudes(network.buses).rename("lat").to_frame()
    frames = []

    specs = (
        ("generator", network.generators, participation.get("generators", {}) or {}, direct_matches),
        ("storage_unit", network.storage_units, participation.get("storage_units", {}) or {}, set()),
    )
    for component_type, components, policy, matches in specs:
        if components.empty:
            continue
        eligible = set(_eligible_components(components, matches, policy))
        behavior = str(policy.get("behavior", "priced_out")).lower()
        frame = pd.DataFrame(
            {
                "component": components.index.astype(str),
                "component_type": component_type,
                "carrier": components["carrier"].fillna("unknown").astype(str).values,
                "capacity_MW": pd.to_numeric(components["p_nom"], errors="coerce").fillna(0.0).values,
                "direct_elexon_match": components.index.astype(str).isin(matches),
                "eligible": components.index.astype(str).isin(eligible),
                "participation_behavior": behavior,
                "latitude_band": [
                    _region_from_bus(str(bus), bus_lat, bands)
                    for bus in components["bus"].astype(str)
                ],
            }
        )
        frame["eligibility_class"] = frame["eligible"].map(
            {True: "eligible", False: "non_participant"}
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_participation_audit(
    network: pypsa.Network,
    redispatch: pd.DataFrame,
    bmu_mapping: pd.DataFrame,
    balancing_config: dict[str, Any],
) -> pd.DataFrame:
    """Summarise coverage and model activity by carrier and latitude band."""
    participation = balancing_config.get("participation", {}) or {}
    audit_config = balancing_config.get("participation_audit", {}) or {}
    components = _component_frame(network, bmu_mapping, participation, audit_config)
    columns = [
        "scope", "component_type", "carrier", "latitude_band", "eligibility_class", "participation_behavior",
        "n_components", "capacity_MW", "direct_elexon_matched_components",
        "direct_elexon_matched_capacity_MW", "eligible_components", "eligible_capacity_MW",
        "non_participant_components", "non_participant_capacity_MW", "model_increase_MWh",
        "model_decrease_MWh", "model_offer_cost_GBP", "model_bid_cost_GBP",
    ]
    if components.empty:
        return pd.DataFrame(columns=columns)

    dispatch_cols = ["component", "type", "increase_MWh", "decrease_MWh", "offer_cost", "bid_cost"]
    available = [column for column in dispatch_cols if column in redispatch.columns]
    dispatch = redispatch[available].copy() if available else pd.DataFrame()
    if not dispatch.empty:
        dispatch = dispatch.rename(columns={"type": "component_type"})
        for column in ["increase_MWh", "decrease_MWh", "offer_cost", "bid_cost"]:
            if column not in dispatch.columns:
                dispatch[column] = 0.0
        dispatch = (
            dispatch.groupby(["component", "component_type"], as_index=False)[
                ["increase_MWh", "decrease_MWh", "offer_cost", "bid_cost"]
            ].sum()
        )
        components = components.merge(dispatch, on=["component", "component_type"], how="left")
    for column in ["increase_MWh", "decrease_MWh", "offer_cost", "bid_cost"]:
        if column not in components.columns:
            components[column] = 0.0
        components[column] = pd.to_numeric(components[column], errors="coerce").fillna(0.0)

    components["direct_elexon_matched_capacity_MW"] = components["capacity_MW"].where(
        components["direct_elexon_match"], 0.0
    )
    components["eligible_capacity_MW"] = components["capacity_MW"].where(
        components["eligible"], 0.0
    )
    components["non_participant_capacity_MW"] = components["capacity_MW"].where(
        ~components["eligible"], 0.0
    )
    components["direct_elexon_matched_components"] = components["direct_elexon_match"].astype(int)
    components["eligible_components"] = components["eligible"].astype(int)
    components["non_participant_components"] = (~components["eligible"]).astype(int)

    value_columns = [
        "capacity_MW", "direct_elexon_matched_components", "direct_elexon_matched_capacity_MW",
        "eligible_components", "eligible_capacity_MW", "non_participant_components",
        "non_participant_capacity_MW", "increase_MWh", "decrease_MWh", "offer_cost", "bid_cost",
    ]

    def summarise(group_columns: list[str], scope: str) -> pd.DataFrame:
        summary = (
            components.groupby(group_columns, dropna=False)
            .agg(
                n_components=("component", "nunique"),
                **{column: (column, "sum") for column in value_columns},
            )
            .reset_index()
            .rename(
                columns={
                    "increase_MWh": "model_increase_MWh",
                    "decrease_MWh": "model_decrease_MWh",
                    "offer_cost": "model_offer_cost_GBP",
                    "bid_cost": "model_bid_cost_GBP",
                }
            )
        )
        summary["scope"] = scope
        if "latitude_band" not in summary.columns:
            summary["latitude_band"] = "All"
        return summary

    carrier = summarise(["component_type", "carrier", "eligibility_class", "participation_behavior"], "carrier")
    band = summarise(
        ["component_type", "carrier", "latitude_band", "eligibility_class", "participation_behavior"],
        "carrier_latitude_band",
    )
    result = pd.concat([carrier, band], ignore_index=True)
    return result.reindex(columns=columns).sort_values(
        ["scope", "component_type", "carrier", "latitude_band", "eligibility_class"], kind="stable"
    )


def main(snakemake) -> None:
    logger = setup_logging(snakemake.log[0])
    scenario_config = snakemake.params.scenario_config
    balancing = scenario_config.get("market", {}).get("balancing", {})
    network = pypsa.Network(str(snakemake.input.network))
    redispatch = pd.read_csv(snakemake.input.redispatch_summary_csv)
    bmu_mapping_input = getattr(snakemake.input, "bmu_mapping", None)
    bmu_mapping = (
        pd.read_csv(bmu_mapping_input)
        if bmu_mapping_input
        else pd.DataFrame(columns=["generator_name"])
    )
    audit = build_participation_audit(network, redispatch, bmu_mapping, balancing)
    output = Path(snakemake.output.participation_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    logger.info("Wrote BM participation audit: %s (%d rows)", output, len(audit))


if __name__ == "__main__":
    main(snakemake)