"""Create an auditable component/cohort register from solved balancing networks.

Vintages are assigned at asset level wherever the network carries the evidence
(an operational date, a REPD development status) and only the undated residual
of each carrier is split pro rata against the completed baseline year.  CfD
contracts additionally carry the allocation round they settle at, the year
that round was awarded and the resulting contract vintage, so the consumer-bill
link can choose between contract-vintage and commissioning-vintage eligibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd

# The rule runs ``python scripts/public_ownership/build_asset_finance_register.py``: the
# vintaged path (task Y10) imports scripts.* lazily, so the project root must be importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# churn_sink (12 Oct 2026): the market stage's negative-only loss sink at storage buses
# (market.wholesale.churn_sink, s1_zf and every family on it), a modelling device, not an asset.
NON_FINANCE_CARRIERS = {"load_shedding", "load shedding", "voll", "unserved_energy", "churn_sink"}
DEFAULT_COMPONENT_TYPES = ("Generator", "StorageUnit")

#: FES 2050 task Y14-prep: the Link carriers a vintaged register takes in.  The HT50
#: build turns FES hydrogen power into Links from the hydrogen bus to an AC bus
#: (``scripts/hydrogen/fes_hydrogen.py``, carrier ``H2_turbine``), about 26 GW that a
#: Generator/StorageUnit scope misses.  Declared: electrolysers and gas reforming are
#: hydrogen-production assets, not electricity assets of any ownership case, so their
#: Links stay out (:data:`EXCLUDED_HYDROGEN_LINK_CARRIERS`); a 2030 register takes no Link.
HYDROGEN_TURBINE_LINK_CARRIERS = ("H2_turbine",)
EXCLUDED_HYDROGEN_LINK_CARRIERS = ("electrolysis", "gas_reforming_ccs")
#: The 2050 crosswalk supplement a vintaged register reads over the base crosswalk.
DEFAULT_CROSSWALK_2050 = (Path(__file__).resolve().parents[2] / "data" / "public_ownership" / "financing"
                          / "pypsa_finance_technology_crosswalk_2050.csv")

# Existing plant a model carrier absorbs (CP30 labels existing pumped hydro as
# LAES): its capacity at the named bus is the existing site, the rest expansion.
EXISTING_SITE_VINTAGE_METHOD = "existing_storage_site"
EXISTING_SITE_EXPANSION_METHOD = "existing_storage_site_expansion"
# Vintage methods that rest on evidence about the individual asset.  Rows with
# any other method are the undated residual that the baseline-year split
# allocates between the legacy and post-policy cohorts.
ASSET_LEVEL_VINTAGE_METHODS = frozenset({
    "asset_date", "development_status", EXISTING_SITE_VINTAGE_METHOD, EXISTING_SITE_EXPANSION_METHOD,
})
RESIDUAL_VINTAGE_METHOD = "2024_to_model_capacity_delta_by_carrier"

# REPD development statuses.  A project that is still being built in the
# source snapshot cannot have been commissioned before the policy start; one
# recorded as operational without a date is treated as legacy because the
# named fleet was assembled from a pre-2026 register snapshot.
PIPELINE_DEVELOPMENT_STATUSES = frozenset({
    "under construction", "awaiting construction", "application submitted",
    "planning permission granted", "consented", "application approved",
})
OPERATIONAL_DEVELOPMENT_STATUSES = frozenset({"operational"})

CONTRACT_VINTAGES = ("post_policy", "pre_policy", "unknown_round", "not_applicable")

#: F28 (user, 26 Oct 2026): small-scale plant, what FES calls "Distributed - Micro" (on homes' and
#: businesses' own premises), stays out of public ownership in every case and is never bought.
#: Such rows carry ``ownership_exclusion`` = this value and are finance-ineligible (``excluded``).
SMALL_SCALE_EXCLUSION = "small_scale"
MICRO_CONNECTION = "Distributed - Micro"
#: The register columns a small-scale split shares out (``fleet_vintages.SPLIT_COLUMNS``).
SMALL_SCALE_SPLIT_COLUMNS = ("capacity_mw", "energy_capacity_mwh", "dispatch_mwh", "cohort_share")


def _snapshot_hours(network) -> pd.Series:
    snapshots = pd.Index(network.snapshots)
    weights = getattr(network, "snapshot_weightings", None)
    if weights is not None:
        for column in ("generators", "objective"):
            if column in weights.columns:
                candidate = pd.to_numeric(weights[column].reindex(snapshots), errors="coerce")
                if candidate.notna().all():
                    return candidate.astype(float)
    return pd.Series(1.0, index=snapshots, dtype=float)


def _text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _commissioning_year(static: pd.Series) -> float | None:
    operational_date = pd.to_datetime(
        static.get("operational_date"), errors="coerce", format="mixed", dayfirst=True
    )
    if pd.notna(operational_date):
        return float(operational_date.year)
    build_year = pd.to_numeric(static.get("build_year"), errors="coerce")
    return float(build_year) if pd.notna(build_year) and build_year > 0 else None


def _finance_vintage(static: pd.Series, policy_start_year: int) -> tuple[str, str]:
    commissioning_year = _commissioning_year(static)
    if commissioning_year is not None:
        return ("new_build" if commissioning_year >= policy_start_year else "legacy", "asset_date")
    status = _text(static.get("development_status")).lower()
    if status in PIPELINE_DEVELOPMENT_STATUSES:
        return "new_build", "development_status"
    if status in OPERATIONAL_DEVELOPMENT_STATUSES:
        return "legacy", "development_status"
    source = _text(static.get("data_source")).lower()
    asset_id = _text(static.name).lower()
    if source.startswith("cp30") or asset_id.startswith("cp30_"):
        return "new_build", "cp30_source_attribute"
    # A carried-forward asset without an attributable identity is kept on its
    # existing private financing rather than guessed to be in construction.
    return "legacy", "preexisting_asset_not_cp30"


LEGACY_PLANT_BRIDGE = "legacy_plant_bridge"


def load_legacy_plant_rounds(path: str | Path) -> frozenset:
    """Rounds the award-year table marks as bridging contracts on legacy plant.

    Such a contract (Drax's 2025 bridging CfD) is awarded after the policy start
    but finances nothing new, so it is classed ``legacy_plant_contract``: never
    re-priced and never new-build state-owned (item MR5).
    """
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "contract_kind" not in frame.columns:
        return frozenset()
    kinds = frame["contract_kind"].str.strip()
    unknown = sorted(set(kinds) - {"", LEGACY_PLANT_BRIDGE})
    if unknown:
        raise ValueError(f"Award-year table has unknown contract_kind values: {unknown}")
    return frozenset(frame.loc[kinds.eq(LEGACY_PLANT_BRIDGE), "cfd_round"].str.strip())


def load_award_years(path: str | Path) -> dict[str, int]:
    """Read the allocation-round award-year table (``cfd_round`` -> year)."""
    frame = pd.read_csv(path)
    required = {"cfd_round", "award_year", "source"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Award-year table lacks columns: {sorted(missing)}")
    if frame["cfd_round"].astype(str).str.strip().duplicated().any():
        raise ValueError("Award-year table repeats a cfd_round")
    if frame["source"].isna().any() or frame["source"].astype(str).str.strip().eq("").any():
        raise ValueError("Every award-year row needs a source")
    years = pd.to_numeric(frame["award_year"], errors="raise")
    return {str(r).strip(): int(y) for r, y in zip(frame["cfd_round"], years)}


def contract_vintage(
    support_type: str, cfd_round: str, award_years: Mapping[str, int] | None, eligible_from: int | None,
    legacy_plant_rounds: frozenset = frozenset(),
) -> tuple[str, float | None]:
    """Classify a contract by the year its allocation round was awarded.

    Returns ``(contract_vintage, cfd_award_year)``.  ``not_applicable`` for
    anything that is not a CfD, ``unknown_round`` for a CfD whose round is
    blank or absent from the award-year table (it then settles at the declared
    future strike in the bill, but its award year is not evidenced).
    """
    if _text(support_type).lower() != "cfd":
        return "not_applicable", None
    if award_years is None or eligible_from is None:
        return "unknown_round", None
    year = award_years.get(_text(cfd_round))
    if year is None:
        return "unknown_round", None
    if _text(cfd_round) in legacy_plant_rounds:
        return "legacy_plant_contract", float(year)
    return ("post_policy" if year >= int(eligible_from) else "pre_policy"), float(year)


def _component_rows(
    network, component_type: str, model_year: int, policy_start_year: int, *,
    support_type_overrides: Mapping[str, str] | None = None,
    award_years: Mapping[str, int] | None = None, eligible_award_years_from: int | None = None,
    legacy_plant_rounds: frozenset = frozenset(),
) -> list[dict]:
    if component_type == "Generator":
        static, dispatch = network.generators, network.generators_t.p
    elif component_type == "StorageUnit":
        static, dispatch = network.storage_units, network.storage_units_t.p
    else:
        raise ValueError(f"Unsupported component type: {component_type}")
    overrides = {str(k): str(v) for k, v in (support_type_overrides or {}).items()}
    hours = _snapshot_hours(network)
    dispatch_energy = (
        dispatch.reindex(index=hours.index, columns=static.index, fill_value=0.0)
        .clip(lower=0.0)
        .mul(hours, axis=0)
        .sum(axis=0)
    )
    # F5 (27 Oct 2026): CP30 2030 now carries consumer flexibility. An unvintaged (2030) register
    # leaves every non-ownership carrier out here; a vintaged one drops them in vintage_register,
    # which reports what it excluded.
    from scripts.public_ownership.model_year import is_vintaged_model_year

    outside = set() if is_vintaged_model_year(model_year) else set(non_ownership_carriers())
    rows = []
    for asset_id, asset in static.iterrows():
        carrier = str(asset.get("carrier", ""))
        if carrier.strip().lower() in NON_FINANCE_CARRIERS or carrier in outside:
            continue
        dispatch_mwh = float(dispatch_energy.get(asset_id, 0.0))
        vintage, method = _finance_vintage(asset, policy_start_year)
        capacity_value = pd.to_numeric(asset.get("p_nom_opt"), errors="coerce")
        # Market-result files retain the built capacity in p_nom but commonly
        # serialise p_nom_opt as zero.  A zero optimisation value is not a
        # commissioned zero-MW asset.
        if pd.isna(capacity_value) or capacity_value <= 0:
            capacity_value = pd.to_numeric(asset.get("p_nom"), errors="coerce")
        capacity = float(capacity_value) if pd.notna(capacity_value) else 0.0
        max_hours = pd.to_numeric(asset.get("max_hours"), errors="coerce")
        network_support = _text(asset.get("support_type", None))
        support = overrides.get(carrier, network_support)
        cfd_round = _text(asset.get("cfd_round", None))
        vintage_label, award_year = contract_vintage(
            support, cfd_round, award_years, eligible_award_years_from, legacy_plant_rounds
        )
        rows.append({
            "model_year": model_year,
            "component_type": component_type,
            "asset_id": str(asset_id),
            "carrier": carrier,
            "bus": _text(asset.get("bus", None)) or None,
            "finance_vintage": vintage,
            "vintage_method": method,
            "commissioning_year": _commissioning_year(asset),
            "development_status": _text(asset.get("development_status", None)) or None,
            "capacity_mw": capacity,
            "energy_capacity_mwh": capacity * float(max_hours) if pd.notna(max_hours) else 0.0,
            "dispatch_mwh": dispatch_mwh,
            "support_type": support or None,
            "network_support_type": network_support or None,
            "cfd_round": cfd_round or None,
            # The contract's own strike technology (floating, permitted reduction, ...),
            # so CfD re-pricing settles each contract at the strike the bill uses.
            "cfd_technology": _text(asset.get("cfd_technology", None)) or None,
            "cfd_award_year": award_year,
            "contract_vintage": vintage_label,
            "data_source": asset.get("data_source", None),
            "revenue_evidence_status": "not_linked_to_asset_settlement",
        })
    return rows


def _link_electrical_side(network, static: pd.DataFrame) -> pd.Series:
    """Which end of each link is the AC (electrical) end: 1 where bus1 is AC, 0 where only bus0 is.

    Raises:
        ValueError: A link has no AC end.
    """
    bus_carrier = network.buses["carrier"].astype(str) if "carrier" in network.buses.columns else pd.Series(dtype=str)
    ac1 = static["bus1"].astype(str).map(bus_carrier).eq("AC")
    ac0 = static["bus0"].astype(str).map(bus_carrier).eq("AC")
    if (~(ac0 | ac1)).any():
        raise ValueError(f"hydrogen-turbine links with no AC bus: {static.index[~(ac0 | ac1)].tolist()[:5]}")
    return pd.Series(np.where(ac1, 1, 0), index=static.index)


def hydrogen_turbine_link_rows(network, model_year: int, policy_start_year: int, *,
                               carriers: tuple[str, ...] = HYDROGEN_TURBINE_LINK_CARRIERS) -> list[dict]:
    """Register rows for the hydrogen-turbine Links of a vintaged (2050) network (task Y14-prep).

    Capacity is the electrical output rating: the HT50 build puts ``p_nom`` on the
    hydrogen side (bus0) and the AC bus on bus1, so it is ``p_nom x efficiency``
    (``p_nom`` where the AC end is bus0). Dispatch is the energy delivered to the AC
    bus, ``-p1`` (or ``efficiency x p0`` when ``p1`` is not stored), over the snapshot
    weights, never negative.

    Args:
        network: A loaded PyPSA network.
        model_year: The register's model year.
        policy_start_year: Build years before it are legacy.
        carriers: The Link carriers taken (:data:`HYDROGEN_TURBINE_LINK_CARRIERS`).

    Returns:
        Rows with the same columns :func:`_component_rows` writes, ``component_type = Link``.
    """
    links = network.links
    if links.empty or "carrier" not in links.columns:
        return []
    static = links.loc[links["carrier"].astype(str).isin(carriers)]
    if static.empty:
        return []
    hours = _snapshot_hours(network)
    side = _link_electrical_side(network, static)
    efficiency = pd.to_numeric(static.get("efficiency"), errors="coerce").fillna(1.0)
    p1 = getattr(network.links_t, "p1", pd.DataFrame())
    p0 = getattr(network.links_t, "p0", pd.DataFrame())
    delivered = pd.DataFrame(0.0, index=hours.index, columns=static.index)
    for name in static.index:
        if side[name] == 1:
            if name in p1.columns:
                series = -p1[name]
            elif name in p0.columns:
                series = p0[name] * float(efficiency[name])
            else:
                continue
        else:
            if name not in p0.columns:
                continue
            series = -p0[name]
        delivered[name] = pd.to_numeric(series.reindex(hours.index), errors="coerce").fillna(0.0).to_numpy()
    energy = delivered.clip(lower=0.0).mul(hours, axis=0).sum(axis=0)
    rows = []
    for asset_id, asset in static.iterrows():
        nominal = pd.to_numeric(asset.get("p_nom_opt"), errors="coerce")
        if pd.isna(nominal) or nominal <= 0:
            nominal = pd.to_numeric(asset.get("p_nom"), errors="coerce")
        nominal = float(nominal) if pd.notna(nominal) else 0.0
        capacity = nominal * float(efficiency[asset_id]) if side[asset_id] == 1 else nominal
        vintage, method = _finance_vintage(asset, policy_start_year)
        rows.append({
            "model_year": model_year, "component_type": "Link", "asset_id": str(asset_id),
            "carrier": str(asset.get("carrier", "")), "bus": _text(asset.get("bus1" if side[asset_id] == 1 else "bus0")) or None,
            "finance_vintage": vintage, "vintage_method": method, "commissioning_year": _commissioning_year(asset),
            "development_status": _text(asset.get("development_status", None)) or None,
            "capacity_mw": capacity, "energy_capacity_mwh": 0.0, "dispatch_mwh": float(energy.get(asset_id, 0.0)),
            "support_type": _text(asset.get("support_type", None)) or None,
            "network_support_type": _text(asset.get("support_type", None)) or None,
            "cfd_round": None, "cfd_award_year": None, "contract_vintage": "not_applicable",
            "data_source": asset.get("data_source", None),
            "revenue_evidence_status": "not_linked_to_asset_settlement",
        })
    return rows


def _static_table(network, component_type: str) -> pd.DataFrame:
    if component_type == "Generator":
        return network.generators
    if component_type == "StorageUnit":
        return network.storage_units
    raise ValueError(f"Unsupported component type: {component_type}")


def _component_capacities(network, component_types: tuple[str, ...]) -> pd.Series:
    """Return installed MW by component/carrier for a completed baseline run."""
    values = []
    for component_type in component_types:
        static = _static_table(network, component_type)
        capacities = pd.to_numeric(static.get("p_nom"), errors="coerce").fillna(0.0)
        frame = pd.DataFrame({
            "component_type": component_type,
            "carrier": static["carrier"].astype(str),
            "capacity_mw": capacities,
        })
        values.append(frame)
    return pd.concat(values).groupby(["component_type", "carrier"])["capacity_mw"].sum()


def _split_post_policy_cohorts(register: pd.DataFrame, baseline_capacity: pd.Series) -> pd.DataFrame:
    """Split each carrier's undated residual into legacy and post-policy cohorts.

    Assets with asset-level evidence (a date or a development status) keep the
    vintage that evidence gives.  The completed baseline run fixes the size of
    the legacy fleet; whatever part of it the dated legacy assets do not
    already account for is carried by the undated residual, pro rata to
    capacity, and the rest of the residual is post-policy build.  Dispatch is
    allocated in the same proportion, preserving the original total.  With no
    dated assets this reduces to the carrier-level capacity delta.
    """
    rows = []
    for (component_type, carrier), group in register.groupby(["component_type", "carrier"], dropna=False):
        asset_level = group["vintage_method"].isin(ASSET_LEVEL_VINTAGE_METHODS)
        dated, undated = group[asset_level], group[~asset_level]
        prior_capacity = float(baseline_capacity.get((component_type, carrier), 0.0))
        dated_legacy = float(dated.loc[dated["finance_vintage"].eq("legacy"), "capacity_mw"].sum())
        undated_capacity = float(undated["capacity_mw"].sum())
        legacy_residual = min(undated_capacity, max(0.0, prior_capacity - dated_legacy))
        legacy_share = legacy_residual / undated_capacity if undated_capacity > 0 else 0.0
        for _, asset in dated.iterrows():
            kept = asset.copy()
            # An existing-site split has already set its own share.
            if pd.isna(pd.to_numeric(kept.get("cohort_legacy_share"), errors="coerce")):
                kept["cohort_legacy_share"] = 1.0 if kept["finance_vintage"] == "legacy" else 0.0
            rows.append(kept)
        for _, asset in undated.iterrows():
            for vintage, share in (("legacy", legacy_share), ("new_build", 1.0 - legacy_share)):
                if share <= 0:
                    continue
                cohort = asset.copy()
                cohort["asset_id"] = f"{asset['asset_id']}::{vintage}"
                cohort["finance_vintage"] = vintage
                cohort["vintage_method"] = RESIDUAL_VINTAGE_METHOD
                cohort["cohort_legacy_share"] = legacy_share
                for column in ("capacity_mw", "energy_capacity_mwh", "dispatch_mwh"):
                    cohort[column] = float(cohort[column]) * share
                rows.append(cohort)
    return pd.DataFrame(rows).reset_index(drop=True)


def parse_existing_storage_sites(sites) -> list[dict]:
    """Validate ``public_ownership.existing_storage_sites`` (a list of mappings) or ``--existing-storage-site`` tokens.

    Each site is ``{name, bus, network_carrier, existing_mw, register_carrier}``;
    a command-line token is ``name=bus:network_carrier:existing_mw:register_carrier``.
    """
    out = []
    for site in sites or []:
        if isinstance(site, str):
            if "=" not in site or site.count(":") != 3:
                raise ValueError(f"existing storage site must be name=bus:network_carrier:existing_mw:register_carrier, got {site!r}")
            name, rest = site.split("=", 1)
            bus, network_carrier, existing_mw, register_carrier = rest.split(":")
            site = {"name": name, "bus": bus, "network_carrier": network_carrier,
                    "existing_mw": existing_mw, "register_carrier": register_carrier}
        missing = {"name", "bus", "network_carrier", "existing_mw", "register_carrier"} - set(site)
        if missing:
            raise ValueError(f"existing storage site {site!r} lacks {sorted(missing)}")
        existing = float(site["existing_mw"])
        if not existing > 0:
            raise ValueError(f"existing storage site {site['name']!r} needs a positive existing_mw")
        out.append({"name": str(site["name"]), "bus": str(site["bus"]), "network_carrier": str(site["network_carrier"]),
                    "existing_mw": existing, "register_carrier": str(site["register_carrier"])})
    # Sites match on the bus's four-character substation code, so two sites may not share one.
    buses = [s["bus"][:4].upper() for s in out]
    if len(set(buses)) != len(buses):
        raise ValueError(f"existing storage sites repeat a bus (substation): "
                         f"{sorted(b for b in buses if buses.count(b) > 1)}")
    return out


def existing_storage_site_flags(sites) -> str:
    """The ``--existing-storage-site`` tokens for a configured site list (used by the Snakemake rule)."""
    parsed = parse_existing_storage_sites(sites)
    if not parsed:
        return ""
    return "--existing-storage-site " + " ".join(
        f"{s['name']}={s['bus']}:{s['network_carrier']}:{s['existing_mw']:g}:{s['register_carrier']}" for s in parsed)


def split_existing_storage_sites(register: pd.DataFrame, sites) -> pd.DataFrame:
    """Register existing plant a model carrier absorbs as legacy, under its own carrier.

    CP30 labels every long-duration store ``LAES``, including the existing
    pumped hydro at Dinorwig, Cruachan, Ffestiniog and Foyers, so the carrier-level
    baseline split (the 2024 run calls them ``Pumped Storage Hydroelectricity``)
    finds no legacy LAES and registers them as new build at the novel-LDES
    capex. Each configured site is matched by bus and network carrier: up to
    ``existing_mw`` of the unit is a ``::legacy`` cohort under
    ``register_carrier`` (``pumped_hydro``, which the crosswalk holds unchanged
    across cases, as every other run treats existing pumped hydro), and any
    capacity above it (a CP30 expansion) stays a ``::new_build`` cohort of the
    network carrier. Capacity, energy and dispatch are split in proportion and
    ``cohort_legacy_share`` records the split for the settlement. A run that
    does not use the network carrier at all (the historical runs label pumped
    hydro as such) is left alone; a run that uses it but not at the configured
    substation is refused, so a moved site cannot silently turn the split off.

    A site matches on the four-character substation code of its bus, not the
    exact busbar: the build pins a site to its best-connected busbar and shares
    it over siblings that cannot carry it alone (Cruachan moved CRUA2R -> CRUA2Q
    between two builds and now spans both). Units named after the site (the
    build names them ``..._existing_<site>``) take the legacy share first, then
    the largest other units at the substation.
    """
    sites = parse_existing_storage_sites(sites)
    if not sites:
        return register
    if "bus" not in register.columns:
        raise ValueError("existing storage sites are matched by bus; the register carries none")
    rows, touched = [], set()
    storage = register["component_type"].eq("StorageUnit")
    site_codes = register["bus"].astype(str).str[:4].str.upper()
    for site in sites:
        carrier_rows = storage & register["carrier"].astype(str).eq(site["network_carrier"])
        if not carrier_rows.any():
            continue
        match = carrier_rows & site_codes.eq(site["bus"][:4].upper())
        if not match.any():
            raise ValueError(f"existing storage site {site['name']!r}: no {site['network_carrier']} StorageUnit at bus "
                             f"{site['bus']!r} (substation {site['bus'][:4]!r}) in the networks")
        candidates = register.loc[match].copy()
        candidates["_named"] = candidates["asset_id"].astype(str).str.lower().str.contains(
            site["name"].lower(), regex=False)
        remaining = site["existing_mw"]
        for index in candidates.sort_values(["_named", "capacity_mw"], ascending=[False, False]).index:
            if remaining <= 1e-9:
                break
            asset = register.loc[index]
            capacity = float(asset["capacity_mw"])
            legacy = min(remaining, capacity)
            remaining -= legacy
            share = legacy / capacity if capacity > 0 else 0.0
            touched.add(index)
            for vintage, part, carrier, method in (
                ("legacy", share, site["register_carrier"], EXISTING_SITE_VINTAGE_METHOD),
                ("new_build", 1.0 - share, asset["carrier"], EXISTING_SITE_EXPANSION_METHOD),
            ):
                if part <= 1e-12:
                    continue
                cohort = asset.copy()
                cohort["asset_id"] = f"{asset['asset_id']}::{vintage}"
                cohort["carrier"] = carrier
                cohort["finance_vintage"] = vintage
                cohort["vintage_method"] = method
                cohort["cohort_legacy_share"] = share
                cohort["existing_site"] = site["name"]
                for column in ("capacity_mw", "energy_capacity_mwh", "dispatch_mwh"):
                    cohort[column] = float(cohort[column]) * part
                rows.append(cohort)
    kept = register.drop(index=sorted(touched))
    return pd.concat([kept, pd.DataFrame(rows)], ignore_index=True)


def build_asset_finance_register(
    network_paths: list[str | Path], crosswalk: pd.DataFrame, *, model_year: int,
    policy_start_year: int = 2025, baseline_network_paths: list[str | Path] | None = None,
    component_types: tuple[str, ...] = DEFAULT_COMPONENT_TYPES,
    support_type_overrides: Mapping[str, str] | None = None,
    award_years: Mapping[str, int] | None = None,
    eligible_award_years_from: int | None = None,
    legacy_plant_rounds: frozenset = frozenset(),
    existing_storage_sites=None,
    fleet_vintage_mix: pd.DataFrame | None = None,
    fleet_vintage_lives=None,
    fleet_vintage_mode: str = "pro_rata",
    site_vintages: pd.DataFrame | None = None,
    crosswalk_supplement: pd.DataFrame | None = None,
    fleet_vintage_repowering=None,
    no_cfd_on_repowering=(),
    small_scale_data_sources=(),
    micro_mw_by_carrier=None,
    legislated_year: int | None = None,
    transfer_year: int | None = None,
    lead_times: pd.DataFrame | None = None,
    snapshot_year: int = 2025,
    rebuild_lives: Mapping[str, float] | None = None,
    technology_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Union static asset identities and sum physical dispatch across full-year networks.

    ``support_type_overrides`` records a configured label correction per
    carrier (the network label is kept in ``network_support_type``).
    ``award_years`` and ``eligible_award_years_from`` give every CfD row its
    ``cfd_award_year`` and ``contract_vintage``.  ``existing_storage_sites``
    (config ``public_ownership.existing_storage_sites``) registers existing
    plant a model carrier absorbs as legacy (:func:`split_existing_storage_sites`).

    FES 2050 plan, task Y10: ``fleet_vintage_mix`` (the stock-flow's target-year
    mix, ``fleet_vintages.py``) replaces the baseline split for a vintaged model
    year (2040 on): every row takes a build year by the join in
    :func:`fleet_vintages.assign_build_years` (dated rows, then bus-located
    ``site_vintages``, then ``fleet_vintage_mode``), and its finance vintage follows
    the build year.  A 2030 register never takes this path.

    Task Y14-prep, on the vintaged path only: the hydrogen-turbine Links join the
    register (:func:`hydrogen_turbine_link_rows`), and ``crosswalk_supplement`` (the
    2050 crosswalk, :data:`DEFAULT_CROSSWALK_2050` when None) is read over the base
    crosswalk (:func:`apply_crosswalk_supplement`), giving the rows that need it a
    ``cost_technology`` distinct from their rate technology.

    FES 2050 decision 62: ``fleet_vintage_repowering`` (``fleet_vintages.RepoweringRule`` by
    technology) reaches the join, and ``no_cfd_on_repowering`` names the technologies whose
    repowered capacity takes no post-AR7 CfD (solar). Both empty: unchanged.

    F28 (user, 26 Oct 2026): ``small_scale_data_sources`` and ``micro_mw_by_carrier`` mark the
    small-scale plant no case owns or buys (:func:`exclude_small_scale`); ``legislated_year`` re-classes
    the new build committed before the legislation year (``legislation_year.apply_legislation_year``,
    P3's purchase on ``transfer_year``, the model year when None). None leaves the old boundary
    (commissioning from ``policy_start_year``).  ``rebuild_lives`` (economic life by finance technology)
    rebuilds dated existing plant at the end of its life (``legislation_year.rebuild_at_end_of_life``,
    F28 (d)); a vintaged register takes it from the stock-flow instead.
    """
    import pypsa  # Keep PyPSA optional for pure finance/unit-test imports.

    component_types = tuple(component_types)
    if not component_types:
        raise ValueError("component_types must name at least one PyPSA component type")

    required_crosswalk = {"component_type", "pypsa_carrier", "workbook_technology", "finance_eligible"}
    missing = required_crosswalk - set(crosswalk.columns)
    if missing:
        raise ValueError(f"Crosswalk lacks columns: {sorted(missing)}")
    vintaged = fleet_vintage_mix is not None
    rows = []
    for path in network_paths:
        network = pypsa.Network(path)
        for component_type in component_types:
            rows.extend(_component_rows(
                network, component_type, model_year, policy_start_year,
                support_type_overrides=support_type_overrides, award_years=award_years,
                eligible_award_years_from=eligible_award_years_from,
                legacy_plant_rounds=legacy_plant_rounds,
            ))
        if vintaged:
            rows.extend(hydrogen_turbine_link_rows(network, model_year, policy_start_year))
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("No finance-register components were found")
    static_columns = [
        "model_year", "component_type", "asset_id", "carrier", "bus", "finance_vintage", "vintage_method",
        "commissioning_year", "development_status", "capacity_mw", "energy_capacity_mwh",
        "support_type", "network_support_type", "cfd_round", "cfd_technology", "cfd_award_year",
        "contract_vintage", "data_source",
    ]
    duplicate_capacity = raw.groupby(["component_type", "asset_id"])["capacity_mw"].nunique()
    # Historical monthly runs can have different embedded-capacity snapshots.
    # Use the observed annual maximum rather than an arbitrary month and retain
    # the method in the register for audit; dispatch is still summed exactly.
    static = (
        raw.sort_values("capacity_mw")
        .drop_duplicates(["component_type", "asset_id"], keep="last")[static_columns]
        .copy()
    )
    static["capacity_aggregation_method"] = [
        "annual_max_across_monthly_runs" if duplicate_capacity.loc[(row.component_type, row.asset_id)] > 1
        else "static_consistent_across_reference_runs"
        for row in static.itertuples()
    ]
    annual_dispatch = raw.groupby(["component_type", "asset_id"], as_index=False)["dispatch_mwh"].sum()
    register = static.merge(annual_dispatch, on=["component_type", "asset_id"], validate="one_to_one")
    register = split_existing_storage_sites(register, existing_storage_sites)
    if fleet_vintage_mix is not None:
        register = vintage_register(register, fleet_vintage_mix, model_year=model_year, lives=fleet_vintage_lives,
                                    mode=fleet_vintage_mode, site_vintages=site_vintages,
                                    policy_start_year=policy_start_year,
                                    baseline_given=bool(baseline_network_paths), award_years=award_years,
                                    repowering=fleet_vintage_repowering, no_cfd_on_repowering=no_cfd_on_repowering)
    elif baseline_network_paths:
        baseline_capacity = pd.concat([
            _component_capacities(pypsa.Network(path), component_types)
            for path in baseline_network_paths
        ]).groupby(level=[0, 1]).sum()
        register = _split_post_policy_cohorts(register, baseline_capacity)
    else:
        default_share = register["finance_vintage"].eq("legacy").astype(float)
        existing = (pd.to_numeric(register["cohort_legacy_share"], errors="coerce")
                    if "cohort_legacy_share" in register.columns else pd.Series(float("nan"), index=register.index))
        register["cohort_legacy_share"] = existing.fillna(default_share)
    mapping = crosswalk.copy()
    mapping["finance_eligible"] = mapping["finance_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
    # F27 item 5 (23 Oct 2026): a base row may cost on another row than its rate technology
    # (OCGT and gas engines take CCGT's rate and their own DESNZ costs).
    base_columns = ["component_type", "pypsa_carrier", "workbook_technology", "finance_eligible"]
    if "cost_technology" in mapping.columns:
        mapping["cost_technology"] = mapping["cost_technology"].replace("", np.nan)
        base_columns.append("cost_technology")
    register = register.merge(
        mapping[base_columns],
        how="left", left_on=["component_type", "carrier"], right_on=["component_type", "pypsa_carrier"],
        validate="many_to_one",
    ).rename(columns={"workbook_technology": "finance_technology"})
    if vintaged:
        supplement = (pd.read_csv(DEFAULT_CROSSWALK_2050) if crosswalk_supplement is None
                      else crosswalk_supplement)
        register = apply_crosswalk_supplement(register, supplement)
    register = cost_existing_storage_sites(register, mapping, existing_storage_sites)
    register = apply_finance_technology_overrides(register, technology_overrides, model_year=int(model_year))
    register["evidence_status"] = "resolved"
    register.loc[register["finance_technology"].isna(), "evidence_status"] = "unmapped_carrier"
    register.loc[~register["finance_eligible"].eq(True), "evidence_status"] = "excluded"
    if rebuild_lives is not None and not vintaged:
        from scripts.public_ownership.legislation_year import rebuild_at_end_of_life

        register = rebuild_at_end_of_life(register, rebuild_lives, policy_start_year=int(policy_start_year),
                                          model_year=int(model_year))
    if legislated_year is not None:
        from scripts.public_ownership.legislation_year import apply_legislation_year

        register = apply_legislation_year(
            register, legislated_year=int(legislated_year), policy_start_year=int(policy_start_year),
            model_year=int(model_year), transfer_year=int(transfer_year if transfer_year is not None else model_year),
            leads=lead_times, snapshot_year=int(snapshot_year), vintaged=vintaged)
    if small_scale_data_sources or micro_mw_by_carrier:
        register = exclude_small_scale(register, data_sources=small_scale_data_sources,
                                       micro_mw_by_carrier=micro_mw_by_carrier)
    return register.drop(columns=["pypsa_carrier"], errors="ignore")


#: Hydrogen-system components of the 2050 build that are not electricity assets of
#: any ownership case (Y6b): the gas reformer with CCS, its gas supply, the hydrogen
#: bus, store and non-power demand.
HYDROGEN_NON_OWNERSHIP_CARRIERS = ("gas_reforming_ccs", "gas_supply", "H2_demand", "H2_gas")
#: User, 3 Oct 2026: domestic batteries are consumer flexibility (as V2G, decision 24), outside
#: every ownership case. Vintaged registers only (no 2030 register carries the carrier).
CONSUMER_STORAGE_NON_OWNERSHIP_CARRIERS = ("Domestic Battery",)


def non_ownership_carriers() -> tuple[str, ...]:
    """Carriers no ownership case may own in a vintaged register (FES 2050 decisions 24 and 25, Y6a/Y6b).

    Consumer-owned flexibility and the demand component loads
    (``scripts.demand.demand_carriers.CONSUMER_OWNED_CARRIERS``), DACCS
    (``NON_ELECTRICITY_ASSET_CARRIERS``) and the hydrogen system outside electricity
    (:data:`HYDROGEN_NON_OWNERSHIP_CARRIERS`).  Matched on carrier, never a custom
    column (pypsa clustering refuses unknown static columns).
    """
    from scripts.demand.demand_carriers import CONSUMER_OWNED_CARRIERS, NON_ELECTRICITY_ASSET_CARRIERS

    return tuple(dict.fromkeys((*CONSUMER_OWNED_CARRIERS, *NON_ELECTRICITY_ASSET_CARRIERS,
                                *HYDROGEN_NON_OWNERSHIP_CARRIERS, *CONSUMER_STORAGE_NON_OWNERSHIP_CARRIERS)))


def vintage_register(register: pd.DataFrame, mix: pd.DataFrame, *, model_year: int, lives=None,
                     mode: str = "pro_rata", site_vintages: pd.DataFrame | None = None,
                     policy_start_year: int = 2025, baseline_given: bool = False,
                     support_contracts: pd.DataFrame | None = None, label_support: bool = True,
                     award_years: Mapping[str, int] | None = None, repowering=None,
                     no_cfd_on_repowering=()) -> pd.DataFrame:
    """Task Y10: give every register row a build year from the stock-flow mix (a vintaged model year only).

    Args:
        register: The unioned register (one row per network component or existing-site cohort).
        mix: The target-year vintage mix (``fleet_vintages.stock_flow``).
        model_year: The register's model year; must be a vintaged year (2040 on).
        lives: Economic life by stock-flow technology (default: Y2's table).
        mode: ``pro_rata`` or ``whole_units`` (:func:`fleet_vintages.assign_build_years`).
        site_vintages: Optional bus-located survivors.
        policy_start_year: Build years before it are the legacy fleet.
            Rows whose carrier is in :func:`non_ownership_carriers` are dropped first.
        baseline_given: A baseline run group was also given; refused (one split only).
        support_contracts: Y11's live contracts (``cfd_strikes.known_contracts``); read for the
            model year when None and ``label_support`` is on.
        label_support: Label each generation row with the contract it holds in the model
            year (:func:`fleet_vintages.label_support_contracts`: known AR7/AR7a/HPC/SZC
            phases first, then ``PAR7_<year>`` CfDs and ``DPA_<year>`` agreements).
        award_years: ``cfd_round -> award year``.
        repowering: FES 2050 decision 62's rules for the join (``fleet_vintages.RepoweringRule``).
        no_cfd_on_repowering: Technologies whose repowered share of a build year (the mix's
            ``replacement_share``) holds no CfD in the model year (decision 62: solar).

    Returns:
        The register as build-year cohorts, each with ``build_year``, ``finance_vintage``,
        ``vintage_method``, ``cohort_share`` and ``cohort_legacy_share``.
    """
    from scripts.public_ownership.fleet_vintages import assign_build_years, load_lives
    from scripts.public_ownership.model_year import is_vintaged_model_year

    if not is_vintaged_model_year(model_year):
        raise ValueError(f"the stock-flow vintage join is for vintaged model years; this register is for {model_year}")
    if baseline_given:
        raise ValueError("a vintaged register takes its legacy/new-build split from the stock-flow, not a baseline "
                         "run group; drop baseline_run_group or the fleet vintages")
    # Consumer-owned flexibility, DACCS and the hydrogen system are no case's asset: out of the register.
    outside = register["carrier"].astype(str).isin(non_ownership_carriers())
    excluded = register.loc[outside, ["component_type", "carrier", "capacity_mw"]].copy()
    register = register.loc[~outside].reset_index(drop=True)
    register["site_cohort_share"] = site_cohort_shares(register)
    cohorts, report = assign_build_years(register, mix, model_year=int(model_year), lives=lives or load_lives(),
                                         site_vintages=site_vintages, mode=mode, policy_start_year=policy_start_year,
                                         repowering=repowering or None)
    if no_cfd_on_repowering and "replacement_share" in mix.columns:
        # Decision 62: the share of each cohort that replaced retired plant (its repowering).
        share = mix.set_index(["technology", "build_year"])["replacement_share"]
        keys = list(zip(cohorts["stock_flow_technology"], pd.to_numeric(cohorts["build_year"], errors="coerce")))
        cohorts["cohort_replacement_share"] = [float(share.get((t, int(y)), 0.0)) if pd.notna(y) else 0.0
                                               for t, y in keys]
    if "refurbishment_share" in mix.columns:
        # F28 (d): the share of each cohort that is refurbished plant (hydro, pumped hydro), costed apart.
        refurbished = mix.set_index(["technology", "build_year"])["refurbishment_share"]
        keys = list(zip(cohorts["stock_flow_technology"], pd.to_numeric(cohorts["build_year"], errors="coerce")))
        cohorts["cohort_refurbishment_share"] = [float(refurbished.get((t, int(y)), 0.0)) if pd.notna(y) else 0.0
                                                 for t, y in keys]
    unassigned = cohorts["vintage_method"].eq("unassigned")
    if unassigned.any():
        raise ValueError(f"{int(unassigned.sum())} register rows found no cell of the vintage mix (their technology "
                         f"has no FES capacity): {cohorts.loc[unassigned, 'asset_id'].head(5).tolist()}")
    if label_support:
        from scripts.public_ownership.fleet_vintages import label_support_contracts

        if support_contracts is None:
            from scripts.market.cfd_strikes import known_contracts

            support_contracts = known_contracts(int(model_year))
        cohorts, contracts_report = label_support_contracts(cohorts, support_contracts, model_year=int(model_year),
                                                            award_years=award_years,
                                                            no_cfd_on_repowering=no_cfd_on_repowering)
        # Year-generic stage 5b: the RO (and FiT) on cohorts outside a contract, as the labels carry them
        # (a model year after 31 March 2037 has none, so a 2050 register is unchanged).
        from scripts.market.support_labels import label_support_schemes, load_support_schemes

        cohorts = label_support_schemes(cohorts, load_support_schemes(), model_year=int(model_year))
        cohorts.attrs["support_contract_report"] = contracts_report
    cohorts.attrs["vintage_join_report"] = report
    cohorts.attrs["non_ownership_excluded"] = excluded.groupby(["component_type", "carrier"])["capacity_mw"].sum()
    return cohorts


def site_cohort_shares(register: pd.DataFrame) -> pd.Series:
    """Each row's share of its network asset before the build-year join (task Y14-prep).

    An existing-site cohort (:func:`split_existing_storage_sites`) is ``cohort_legacy_share``
    of its unit (legacy) or the rest (expansion); every other row is the whole asset (1.0).
    The build-year join overwrites ``cohort_legacy_share``, so the settlement reads this
    column (``settlement.vintaged_settlement_share``) to keep the site split.

    Args:
        register: The unioned register, before the join.

    Returns:
        The share per row.
    """
    share = pd.Series(1.0, index=register.index)
    if "cohort_legacy_share" not in register.columns or "vintage_method" not in register.columns:
        return share
    legacy_share = pd.to_numeric(register["cohort_legacy_share"], errors="coerce")
    method = register["vintage_method"].astype(str)
    legacy = method.eq(EXISTING_SITE_VINTAGE_METHOD)
    expansion = method.eq(EXISTING_SITE_EXPANSION_METHOD)
    share[legacy] = legacy_share[legacy]
    share[expansion] = 1.0 - legacy_share[expansion]
    if share.isna().any():
        raise ValueError("existing-site cohorts need cohort_legacy_share")
    return share


#: Columns the 2050 crosswalk supplement must carry.
SUPPLEMENT_COLUMNS = ("component_type", "pypsa_carrier", "stock_flow_technology", "workbook_technology",
                      "cost_technology", "finance_eligible", "source", "status")


def apply_crosswalk_supplement(register: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    """Read the 2050 crosswalk over the base mapping of a vintaged register (task Y14-prep).

    A supplement row keyed on ``(component_type, pypsa_carrier, stock_flow_technology)``
    wins for the rows of that stock-flow technology (a pooled carrier such as
    ``nuclear`` or ``H2_turbine`` maps each cohort by what the Y10 join named it); a row
    with a blank ``stock_flow_technology`` applies to the whole carrier. The row sets
    ``finance_technology`` (the workbook rate technology), ``finance_eligible`` and
    ``cost_technology`` (the vintage-table row, where it differs from the rate
    technology), and records ``crosswalk_source``. Rows the supplement does not name
    keep the base mapping, with ``cost_technology`` equal to ``finance_technology``.

    Args:
        register: The register after the base crosswalk merge (``finance_technology``,
            ``finance_eligible``; ``stock_flow_technology`` from the join).
        supplement: ``pypsa_finance_technology_crosswalk_2050.csv``.

    Returns:
        The register with the supplement applied.

    Raises:
        ValueError: The supplement lacks a column, repeats a key or a row lacks provenance.
    """
    if missing := set(SUPPLEMENT_COLUMNS) - set(supplement.columns):
        raise ValueError(f"crosswalk supplement lacks columns: {sorted(missing)}")
    table = supplement.copy()
    for column in ("component_type", "pypsa_carrier", "stock_flow_technology", "workbook_technology",
                   "cost_technology"):
        table[column] = table[column].fillna("").astype(str).str.strip()
    blank_source = table["source"].isna() | table["source"].astype(str).str.strip().eq("")
    if blank_source.any():
        raise ValueError("every crosswalk supplement row needs a source")
    keys = ["component_type", "pypsa_carrier", "stock_flow_technology"]
    if table.duplicated(keys).any():
        raise ValueError(f"crosswalk supplement repeats a key: {table.loc[table.duplicated(keys), keys].values.tolist()}")
    table["finance_eligible"] = table["finance_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
    out = register.copy()
    stock_flow = (out["stock_flow_technology"].fillna("").astype(str) if "stock_flow_technology" in out.columns
                  else pd.Series("", index=out.index))
    exact = {tuple(k): r for k, r in zip(table[keys].values.tolist(), table.to_dict("records"))}
    carrier_wide = {(r["component_type"], r["pypsa_carrier"]): r for r in table.to_dict("records")
                    if r["stock_flow_technology"] == ""}
    base_cost = (out["cost_technology"] if "cost_technology" in out.columns
                 else pd.Series(np.nan, index=out.index, dtype=object))
    out["cost_technology"] = base_cost.where(base_cost.notna(), out["finance_technology"]).astype(object)
    out["crosswalk_source"] = "base"
    out["finance_technology"] = out["finance_technology"].astype(object)
    out["finance_eligible"] = out["finance_eligible"].astype(object)
    for index in out.index:
        key = (str(out.at[index, "component_type"]), str(out.at[index, "carrier"]))
        row = exact.get((*key, stock_flow[index])) or carrier_wide.get(key)
        if row is None:
            continue
        rate = row["workbook_technology"] or None
        out.at[index, "finance_technology"] = rate
        out.at[index, "cost_technology"] = row["cost_technology"] or rate
        out.at[index, "finance_eligible"] = bool(row["finance_eligible"])
        out.at[index, "crosswalk_source"] = "2050_supplement"
    out["finance_eligible"] = out["finance_eligible"].map(lambda v: v is True or v == True)  # noqa: E712
    return out


#: Declared per-asset cost technologies (15 Nov 2026): plant a model carrier mislabels, costed on its own row.
DEFAULT_TECHNOLOGY_OVERRIDES = (Path(__file__).resolve().parents[2] / "data" / "public_ownership" / "financing"
                                / "finance_technology_overrides.csv")


def apply_finance_technology_overrides(register: pd.DataFrame, table: pd.DataFrame | None, *,
                                       model_year: int) -> pd.DataFrame:
    """Give the assets a declared table names their own finance and cost technology (user decision E5, 15 Nov 2026).

    CP30 labels its pumped-storage projects ``LAES``, so the carrier-level crosswalk costed them on the novel-LDES
    proxy (Coire Glas at about 5,660 GBP/kW against FES 2025's 2,037 for pumped storage). A row matches assets of its
    model year whose ``asset_id`` contains ``asset_match``; one that matches none is refused (a renamed asset must
    not silently fall back to the proxy), unless the register holds none of the table's carriers at all (a test fleet).
    """
    if table is None or table.empty or register.empty or "asset_id" not in register.columns:
        return register
    rows = table.loc[pd.to_numeric(table["model_year"], errors="coerce").eq(int(model_year))]
    if rows.empty:
        return register
    register = register.copy()
    if "finance_technology_source" not in register.columns:
        register["finance_technology_source"] = None
    if "cost_technology" not in register.columns:
        register["cost_technology"] = np.nan
    ids = register["asset_id"].astype(str)
    missing = []
    for row in rows.itertuples(index=False):
        hit = ids.str.contains(str(row.asset_match), regex=False)
        if not hit.any():
            missing.append(str(row.asset_match))
            continue
        register.loc[hit, "finance_technology"] = row.finance_technology
        register.loc[hit, "cost_technology"] = row.cost_technology
        register.loc[hit, "finance_eligible"] = True
        register.loc[hit, "finance_technology_source"] = f"override:{row.asset_match}"
    if missing and ids.str.contains("CP30_", regex=False).any():
        raise ValueError(f"finance technology overrides match no asset: {missing}; the table and the fleet have "
                         "drifted apart (renamed or removed assets)")
    return register


def cost_existing_storage_sites(register: pd.DataFrame, crosswalk: pd.DataFrame, sites) -> pd.DataFrame:
    """Give the legacy cohort of each existing storage site a cost mapping, so it is acquired and costed.

    Decided 26 Sep 2026: existing pumped hydro is acquired under nationalisation
    like any other legacy plant (priced on the thermal-and-storage basis) and
    costed at the public rate, not held unchanged and paid as today.  The
    crosswalk's ``pumped_hydro`` row is not finance-eligible (it was the "held
    unchanged" treatment), so the cohort takes the network carrier's crosswalk
    technology (LAES: the workbook's long-duration storage category, whose WACC,
    O&M and capex apply as a proxy), recorded in ``finance_technology_source``.
    """
    parsed = {s["name"]: s for s in parse_existing_storage_sites(sites)}
    if not parsed or "existing_site" not in register.columns:
        return register
    register = register.copy()
    eligible = crosswalk.loc[crosswalk["finance_eligible"].eq(True)]
    by_carrier = eligible.set_index(["component_type", "pypsa_carrier"])["workbook_technology"]
    if "finance_technology_source" not in register.columns:
        register["finance_technology_source"] = None
    legacy = register["vintage_method"].eq(EXISTING_SITE_VINTAGE_METHOD) & register["existing_site"].notna()
    for index in register.index[legacy]:
        site = parsed.get(str(register.at[index, "existing_site"]))
        if site is None:
            continue
        technology = by_carrier.get(("StorageUnit", site["network_carrier"]))
        if not technology:
            raise ValueError(f"existing storage site {site['name']!r}: no finance technology for its legacy cohort; "
                             f"map {site['network_carrier']!r} as finance-eligible in the crosswalk")
        register.at[index, "finance_technology"] = technology
        register.at[index, "finance_eligible"] = True
        register.at[index, "finance_technology_source"] = f"existing_site_network_carrier:{site['network_carrier']}"
    return register


def micro_capacity_by_carrier(es1_lines: pd.DataFrame) -> dict[str, float]:
    """FES "Distributed - Micro" supply capacity (MW) by model carrier, from a study's ES1 lines table.

    The table is the FES chain's ``*_es1_lines.csv`` (``connection``, ``role``, ``quantity``,
    ``carrier``, ``value_std``). Storage is left out: domestic batteries are a non-ownership
    carrier of their own (:data:`CONSUMER_STORAGE_NON_OWNERSHIP_CARRIERS`).
    """
    missing = {"connection", "role", "quantity", "carrier", "value_std"} - set(es1_lines.columns)
    if missing:
        raise ValueError(f"ES1 lines table lacks columns: {sorted(missing)}")
    micro = es1_lines.loc[es1_lines["connection"].astype(str).eq(MICRO_CONNECTION)
                          & es1_lines["role"].astype(str).eq("supply")
                          & es1_lines["quantity"].astype(str).eq("capacity_mw")]
    values = pd.to_numeric(micro["value_std"], errors="coerce")
    if values.isna().any():
        raise ValueError("ES1 micro capacity lines carry a non-numeric value")
    return {str(c): float(v) for c, v in values.groupby(micro["carrier"].astype(str)).sum().items() if v > 0}


def exclude_small_scale(register: pd.DataFrame, *, data_sources=(), micro_mw_by_carrier=None,
                        micro_data_source: str = "FES") -> pd.DataFrame:
    """Keep small-scale plant out of public ownership (F28, user 26 Oct 2026).

    Two ways a row is small-scale: its ``data_source`` is one of ``data_sources`` (2030: CP30's
    ``CP30_small_scale_solar_gsp``, rooftop solar placed at its GSP), or it is the micro share
    of a carrier's ``micro_data_source`` rows (2050: the FES rows do not separate micro from
    distributed plant, so each FES row of a carrier is split into ``micro MW / FES-row MW`` of
    itself, ``<asset_id>::small_scale``, and the rest, which keeps its id). A split shares
    :data:`SMALL_SCALE_SPLIT_COLUMNS` and keeps every label, support contract included, pro rata.
    Small-scale rows get ``ownership_exclusion`` = :data:`SMALL_SCALE_EXCLUSION`, are finance
    ineligible and ``excluded``: private in every case and outside every purchase.

    Args:
        register: The register after the crosswalk merge.
        data_sources: Data sources whose rows are wholly small-scale.
        micro_mw_by_carrier: Micro capacity by carrier (:func:`micro_capacity_by_carrier`).
        micro_data_source: The data source of the rows the micro share is taken from.

    Returns:
        The register with the small-scale rows marked (and split).

    Raises:
        ValueError: A carrier's micro capacity exceeds its rows of ``micro_data_source``.
    """
    out = register.copy()
    if "ownership_exclusion" not in out.columns:
        out["ownership_exclusion"] = None
    out["ownership_exclusion"] = out["ownership_exclusion"].astype(object)
    source = out["data_source"].astype(str) if "data_source" in out.columns else pd.Series("", index=out.index)
    out.loc[source.isin([str(s) for s in data_sources or ()]), "ownership_exclusion"] = SMALL_SCALE_EXCLUSION
    parts = []
    contract = (out["support_type"].fillna("").astype(str).str.strip().str.lower().isin(["cfd", "dpa", "rab"])
                if "support_type" in out.columns else pd.Series(False, index=out.index))
    for carrier, micro_mw in sorted((micro_mw_by_carrier or {}).items()):
        pool = (out["carrier"].astype(str).eq(carrier) & source.eq(micro_data_source)
                & out["ownership_exclusion"].isna())
        # S3 scope audit (14 Nov 2026): micro plant holds no CfD, so its share comes from the carrier's rows off a
        # contract first (taking it pro rata from CfD parks too left 4.6 GW of contracted solar the 2030 purchase
        # bought private in 2040); contracted rows only where the rest cannot hold it.
        off_contract = pool & ~contract
        if float(out.loc[off_contract, "capacity_mw"].sum()) >= micro_mw * (1.0 - 1e-9):
            pool = off_contract
        total = float(out.loc[pool, "capacity_mw"].sum())
        if micro_mw > total * (1.0 + 1e-9):
            raise ValueError(f"{carrier}: {micro_mw:.1f} MW of micro capacity exceeds its {micro_data_source} rows "
                             f"({total:.1f} MW)")
        share = micro_mw / total if total > 0 else 0.0
        if share <= 0:
            continue
        micro = out.loc[pool].copy()
        micro["asset_id"] = micro["asset_id"].astype(str) + f"::{SMALL_SCALE_EXCLUSION}"
        micro["ownership_exclusion"] = SMALL_SCALE_EXCLUSION
        for column in SMALL_SCALE_SPLIT_COLUMNS:
            if column in out.columns:
                values = pd.to_numeric(out.loc[pool, column], errors="coerce")
                out.loc[pool, column] = values * (1.0 - share)
                micro[column] = values * share
        parts.append(micro)
    if parts:
        out = pd.concat([out, *parts], ignore_index=True)
    small = out["ownership_exclusion"].eq(SMALL_SCALE_EXCLUSION)
    out.loc[small, "finance_eligible"] = False
    if "evidence_status" in out.columns:
        out.loc[small, "evidence_status"] = "excluded"
    return out


def parse_support_type_overrides(values: list[str] | None) -> dict[str, str]:
    """Parse ``carrier=label`` command-line pairs."""
    overrides = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"support-type override must be carrier=label, got {item!r}")
        carrier, label = item.split("=", 1)
        overrides[carrier.strip()] = label.strip()
    return overrides


def _load_lead_times(path):
    from scripts.public_ownership.legislation_year import load_lead_times

    return load_lead_times(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crosswalk")
    parser.add_argument("output")
    parser.add_argument("networks", nargs="+")
    parser.add_argument("--model-year", type=int, required=True)
    parser.add_argument("--policy-start-year", type=int, default=2025)
    parser.add_argument("--baseline-networks", nargs="*")
    parser.add_argument(
        "--component-types", nargs="+", default=list(DEFAULT_COMPONENT_TYPES),
        help="PyPSA component types to register (config public_ownership.ownership_scope)",
    )
    parser.add_argument(
        "--support-type-override", nargs="*", default=[],
        help="carrier=label corrections (config public_ownership.label_corrections.support_type)",
    )
    parser.add_argument("--award-years", help="CSV of cfd_round -> award_year with sources")
    parser.add_argument(
        "--eligible-award-years-from", type=int, default=None,
        help="First award year counted as post-policy (config cfd_repricing.eligible_award_years_from)",
    )
    parser.add_argument(
        "--existing-storage-site", nargs="*", default=[],
        help="name=bus:network_carrier:existing_mw:register_carrier (config public_ownership.existing_storage_sites)",
    )
    parser.add_argument("--fleet-vintages", default=None,
                        help="FES 2050 task Y10: the stock-flow vintage mix CSV (fleet_vintages.py); vintaged model "
                             "years only, in place of --baseline-networks")
    parser.add_argument("--fleet-vintage-mode", default="pro_rata", choices=("pro_rata", "whole_units"))
    parser.add_argument("--site-vintages", default=None, help="optional bus-located survivors CSV")
    parser.add_argument("--life-source", default="central", choices=("central", "fes_e02"))
    parser.add_argument("--life-override", nargs="*", default=[],
                        help="FES 2050 decision 48: TECHNOLOGY=YEARS over the life source, as the stock-flow got it")
    parser.add_argument("--repowering", nargs="*", default=[],
                        help="FES 2050 decision 62: TECHNOLOGY=SHARE:MAX_LIFE, as the stock-flow got it")
    parser.add_argument("--no-cfd-on-repowering", nargs="*", default=[],
                        help="FES 2050 decision 62: technologies whose repowered capacity takes no CfD ('Solar PV')")
    parser.add_argument("--legislated-year", type=int, default=None,
                        help="F28: the legislation year; new build committed before it is not eligible "
                             "(config public_ownership.legislation.year, a group's legislated_year)")
    parser.add_argument("--transfer-year", type=int, default=None,
                        help="F28: P3's purchase date (config public_ownership.legislation.transfer_year)")
    parser.add_argument("--lead-times", default=None, help="F28: the lead-time table (legislation_lead_times.csv)")
    parser.add_argument("--rebuild-lives", default=None,
                        help="F28 (d): technology_costs.csv; dated existing plant reaching the end of its economic life "
                             "is rebuilt as new build of that year (config public_ownership.legislation."
                             "rebuild_at_end_of_life)")
    parser.add_argument("--snapshot-year", type=int, default=2025,
                        help="F28: the REPD snapshot's year (pipeline projects commission after it)")
    parser.add_argument("--small-scale-data-source", nargs="*", default=[],
                        help="F28: data sources whose rows are small-scale plant (config "
                             "public_ownership.small_scale_exclusion.data_sources)")
    parser.add_argument("--micro-capacity-lines", default=None,
                        help="F28: the study's ES1 lines CSV; its 'Distributed - Micro' supply capacity is taken "
                             "from the FES rows as small-scale plant")
    parser.add_argument("--technology-overrides", default=None,
                        help="declared per-asset finance and cost technologies (finance_technology_overrides.csv)")
    parser.add_argument("--crosswalk-supplement", default=None,
                        help="task Y14-prep: the 2050 crosswalk read over the base one on the vintaged path "
                             f"(default {DEFAULT_CROSSWALK_2050.name})")
    args = parser.parse_args()
    fleet_mix = pd.read_csv(args.fleet_vintages) if args.fleet_vintages else None
    fleet_lives, repowering = None, None
    if fleet_mix is not None:
        from scripts.public_ownership.fleet_vintages import load_lives, parse_life_overrides, parse_repowering

        fleet_lives = load_lives(source=args.life_source, overrides=parse_life_overrides(args.life_override))
        repowering = parse_repowering(args.repowering) or None
    register = build_asset_finance_register(
        args.networks, pd.read_csv(args.crosswalk), model_year=args.model_year,
        policy_start_year=args.policy_start_year, baseline_network_paths=args.baseline_networks,
        component_types=tuple(args.component_types),
        support_type_overrides=parse_support_type_overrides(args.support_type_override),
        award_years=load_award_years(args.award_years) if args.award_years else None,
        eligible_award_years_from=args.eligible_award_years_from,
        legacy_plant_rounds=load_legacy_plant_rounds(args.award_years) if args.award_years else frozenset(),
        existing_storage_sites=args.existing_storage_site,
        fleet_vintage_mix=fleet_mix, fleet_vintage_lives=fleet_lives, fleet_vintage_mode=args.fleet_vintage_mode,
        site_vintages=pd.read_csv(args.site_vintages) if args.site_vintages else None,
        crosswalk_supplement=pd.read_csv(args.crosswalk_supplement) if args.crosswalk_supplement else None,
        technology_overrides=pd.read_csv(args.technology_overrides) if args.technology_overrides else None,
        fleet_vintage_repowering=repowering, no_cfd_on_repowering=tuple(args.no_cfd_on_repowering),
        small_scale_data_sources=tuple(args.small_scale_data_source),
        legislated_year=args.legislated_year, transfer_year=args.transfer_year,
        lead_times=None if args.lead_times is None else _load_lead_times(args.lead_times),
        snapshot_year=args.snapshot_year,
        rebuild_lives=(None if args.rebuild_lives is None else
                       pd.read_csv(args.rebuild_lives).set_index("technology")["economic_life_years"].astype(float).to_dict()),
        micro_mw_by_carrier=(micro_capacity_by_carrier(pd.read_csv(args.micro_capacity_lines))
                             if args.micro_capacity_lines else None),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    register.to_csv(output, index=False)


if __name__ == "__main__":
    main()
