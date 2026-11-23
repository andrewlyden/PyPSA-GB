"""
Market Simulation Rules for PyPSA-GB  Two-Stage Dispatch

Pipeline Stages:
  1. solve_wholesale_market    - Stage 1: copperplate dispatch (no network constraints)
  2. solve_balancing_mechanism - Stage 2: anchored redispatch with full constraints
  3. analyze_market_results    - Post-processing dashboard and summary

Inputs:
  - Finalized network: {scenario}.nc (from main workflow)

Outputs:
  - Wholesale solved network + dispatch CSVs
  - Balancing solved network + redispatch/congestion CSVs
  - Market dashboard (HTML) + summary (JSON)

Configuration:
  Enable per-scenario via market.enabled: true in scenarios.yaml.
  See config/defaults/base/market.yaml and config/defaults/base/market/ for full configuration.

See Also:
  - solve.smk - Standard (single-stage) network optimisation
  - analysis.smk - Standard post-solve analysis
  - scripts/market/ - Python modules for market simulation
"""

import re

# Use scenarios and run_ids from main Snakefile (no config reloading)
_scenarios = scenarios
_run_ids = run_ids
_cfg = config


def _get_solver_config_market(scenario_id=None):
    """
    Get solver configuration for market rules.

    Mirrors _get_solver_config from solve.smk but is local to this module.
    """
    if scenario_id and scenario_id in _scenarios:
        solver_config = _scenarios[scenario_id].get("solve", {}).get("solver", {})
    elif _run_ids:
        solver_config = _scenarios.get(_run_ids[0], {}).get("solve", {}).get("solver", {})
    else:
        solver_config = _cfg.get("solve", {}).get("solver", {})

    solver_name = solver_config.get("name", "gurobi")

    # Fallback values mirror config/defaults/base/solve.yaml solver: (fast profile, 18 Sep
    # 2026); the conservative numerics live in the solve_with_fallback ladder.
    if solver_name == "gurobi":
        solver_options = {
            "threads": solver_config.get("threads", 2),
            "method": solver_config.get("method", 2),
            "crossover": solver_config.get("crossover", 0),
            "BarHomogeneous": solver_config.get("BarHomogeneous", 0),
            "BarConvTol": solver_config.get("BarConvTol", 1.0e-4),
            "FeasibilityTol": solver_config.get("FeasibilityTol", 1.0e-4),
            "OptimalityTol": solver_config.get("OptimalityTol", 1.0e-4),
            "NumericFocus": solver_config.get("NumericFocus", 0),
            "ScaleFlag": solver_config.get("ScaleFlag", -1),
            "DualReductions": solver_config.get("DualReductions", 1),
            "BarIterLimit": solver_config.get("BarIterLimit", 200),
        }
    elif solver_name == "highs":
        solver_options = {
            "threads": solver_config.get("threads", 2),
            "log_to_console": False,
        }
    else:
        solver_options = {"threads": solver_config.get("threads", 2)}

    return solver_name, solver_options


def _is_market_enabled(scenario_id):
    """Check if market simulation is enabled for a scenario."""
    return _scenarios.get(scenario_id, {}).get("market", {}).get("enabled", False)


def _build_source(scenario_id):
    """Scenario whose build chain produced this scenario's network and generator files.

    A scenario with ``network_from: <parent>`` (the twelve CP30 months share
    one full-year build) reads the parent's artefacts; the main Snakefile
    checks that its build-relevant config equals the parent's and puts the
    parent in ``build_ids`` so the build chain runs exactly once.
    """
    return _scenarios.get(scenario_id, {}).get("network_from") or scenario_id


def _finalized_network(scenario_id):
    """Path of the finalized network a market scenario solves."""
    return f"{resources_path}/network/{_build_source(scenario_id)}.nc"


def _wholesale_source(scenario_id):
    """Scenario whose Stage 1 solve this scenario's balancing stage reads.

    A scenario with ``wholesale_from: <parent>`` takes the parent's dispatch,
    price, storage and link CSVs rather than re-solving an identical wholesale
    market (item BM1: a public-ownership balancing counterfactual changes only
    ``market.balancing``). The Snakefile refuses the declaration at parse time
    unless everything the wholesale stage reads matches, so this is a shortcut
    through a provably identical solve, not an approximation.
    """
    return _scenarios.get(scenario_id, {}).get("wholesale_from") or scenario_id


def _regex_from_list(items):
    """Generate regex pattern that matches exactly the items in the list."""
    if not items:
        return r"(?!x)x"  # Never-matching pattern
    return r"(?:%s)" % "|".join(map(re.escape, items))


# Build regex for scenarios with market enabled
_market_scenario_ids = [rid for rid in _run_ids if _is_market_enabled(rid)]

_offshore_link_audit_ids = [
    rid for rid in _market_scenario_ids
    if _scenarios.get(rid, {}).get("fleet", {}).get("spatial_proxy", {}).get(
        "offshore_connection_allocation", {}
    ).get("audit", False)
]
OFFSHORE_LINK_AUDIT_REGEX = (
    _regex_from_list(_offshore_link_audit_ids)
    if _offshore_link_audit_ids else r"(?!x)x"
)

_offshore_landing_mapping_audit_ids = [
    rid for rid in _market_scenario_ids
    if _scenarios.get(rid, {}).get("fleet", {}).get("spatial_proxy", {}).get(
        "offshore_connection_allocation", {}
    ).get("audit_mapping", False)
]
OFFSHORE_LANDING_MAPPING_AUDIT_REGEX = (
    _regex_from_list(_offshore_landing_mapping_audit_ids)
    if _offshore_landing_mapping_audit_ids else r"(?!x)x"
)

_offshore_project_crosswalk_audit_ids = [
    rid for rid in _market_scenario_ids
    if _scenarios.get(rid, {}).get("fleet", {}).get("spatial_proxy", {}).get(
        "offshore_connection_allocation", {}
    ).get("audit_project_crosswalk", False)
]
OFFSHORE_PROJECT_CROSSWALK_AUDIT_REGEX = (
    _regex_from_list(_offshore_project_crosswalk_audit_ids)
    if _offshore_project_crosswalk_audit_ids else r"(?!x)x"
)

MARKET_SCENARIO_REGEX = _regex_from_list(_market_scenario_ids)

# Scenarios whose Stage 1 is actually solved: a wholesale_from child reads its
# parent's files, so it never solves its own, and the parent is solved even when
# only the child was selected (e.g. --config scenario=<child>).
_wholesale_solve_ids = list(dict.fromkeys(
    _wholesale_source(rid) for rid in _market_scenario_ids
))
# Central dispatch Stage 1 jobs (copperplate_mode none: the full network, ~9x a
# collapsed solve) are the long pole of a full run, so they get scheduling
# priority through a twin rule (solve_wholesale_market_central below; Snakemake
# priorities are static per rule). The two regexes partition the solve ids.
_central_wholesale_solve_ids = [
    sid for sid in _wholesale_solve_ids
    if str((_scenarios.get(sid, {}).get("market", {}).get("wholesale", {}) or {})
           .get("copperplate_mode", "collapse")).lower() == "none"
]
WHOLESALE_SOLVE_REGEX = _regex_from_list(
    [sid for sid in _wholesale_solve_ids if sid not in set(_central_wholesale_solve_ids)]
)
WHOLESALE_CENTRAL_SOLVE_REGEX = _regex_from_list(_central_wholesale_solve_ids)
CENTRAL_WHOLESALE_PRIORITY = 50


_topology_adequacy_ids = [
    rid for rid in _market_scenario_ids
    if _scenarios.get(rid, {}).get("network", {}).get("topology_adequacy", {}).get("enabled", False)
]

TOPOLOGY_ADEQUACY_SCENARIO_REGEX = _regex_from_list(_topology_adequacy_ids)

def _neso_boundary_mapping_path(scenario_id):
    """Resolve the NESO boundary mapping file used by a scenario."""
    sc = _scenarios.get(scenario_id, {})
    neso_cfg = (
        sc.get("network", {}).get("transmission", {})
        .get("outage_schedule", {})
        .get("neso", {})
    )
    if neso_cfg.get("boundary_mapping"):
        return neso_cfg["boundary_mapping"]

    global_neso_cfg = (
        _cfg.get("network", {}).get("transmission", {})
        .get("outage_schedule", {})
        .get("neso", {})
    )
    return global_neso_cfg.get(
        "boundary_mapping",
        "data/network/neso_boundary_mapping.yaml",
    )


def _effective_bid_offer_source(scenario_id):
    """Resolve the effective bid/offer source for a scenario."""
    sc = _scenarios.get(scenario_id, {})
    if not sc.get("market", {}).get("enabled", False):
        return None

    source = (
        sc.get("market", {})
        .get("balancing", {})
        .get("bid_offer_source", "auto")
    )
    if source != "auto":
        return source

    modelled_year = sc.get("modelled_year", 9999)
    return "elexon" if modelled_year <= 2024 else "derived"


def _uses_elexon_bid_offer_data(scenario_id):
    """Check if scenario should retrieve/use ELEXON bid/offer data."""
    sc = _scenarios.get(scenario_id, {})
    if not sc.get("market", {}).get("enabled", False):
        return False

    modelled_year = sc.get("modelled_year", 9999)
    if modelled_year > 2024:
        return False

    return _effective_bid_offer_source(scenario_id) == "elexon"


def _uses_elexon_price_ladders(scenario_id):
    """Check if scenario should use full ELEXON BOD price ladders in BM."""
    if not _uses_elexon_bid_offer_data(scenario_id):
        return False
    sc = _scenarios.get(scenario_id, {})
    return bool(
        sc.get("market", {})
        .get("balancing", {})
        .get("elexon", {})
        .get("price_ladders", {})
        .get("enabled", False)
    )


def _elexon_solve_dates(scenario_id):
    """Return start/end dates used to retrieve ELEXON BOD data.

    The solve period when one is enabled, else the modelled calendar year: a
    full-year historical scenario without explicit dates used to fetch the
    first week of 2023, whatever its year.
    """
    sc = _scenarios.get(scenario_id, {})
    solve_period = sc.get("solve_period", {}) or {}
    if solve_period.get("enabled", False) and solve_period.get("start") and solve_period.get("end"):
        return str(solve_period["start"])[:10], str(solve_period["end"])[:10]
    year = int(sc["modelled_year"])
    return f"{year}-01-01", f"{year}-12-31"


def _elexon_ladder_settings(scenario_id):
    """Return ELEXON ladder settings that affect processed BOD output."""
    sc = _scenarios.get(scenario_id, {})
    ladder_cfg = (
        sc.get("market", {})
        .get("balancing", {})
        .get("elexon", {})
        .get("price_ladders", {})
    )
    build_ladders = bool(ladder_cfg.get("enabled", False))
    max_blocks = int(ladder_cfg.get("max_blocks_per_side", 10)) if build_ladders else 0
    return build_ladders, max_blocks


def _elexon_cache_key(scenario_id):
    """Shared processed ELEXON BOD cache key for a scenario."""
    start_date, end_date = _elexon_solve_dates(scenario_id)
    build_ladders, max_blocks = _elexon_ladder_settings(scenario_id)
    return (
        f"{start_date}_to_{end_date}_"
        f"ladders-{int(build_ladders)}_blocks-{max_blocks}"
    )


def _shared_elexon_files_for_key(cache_key):
    """Return shared processed ELEXON BOD files for a cache key."""
    base = f"{resources_path}/market/elexon_shared/{cache_key}"
    return {
        "offers_file": f"{base}/elexon_offers.csv",
        "bids_file": f"{base}/elexon_bids.csv",
        "offer_ladders_file": f"{base}/elexon_offer_ladders.csv",
        "bid_ladders_file": f"{base}/elexon_bid_ladders.csv",
    }


def _shared_elexon_files_for_scenario(scenario_id):
    """Return shared processed ELEXON BOD files for a scenario."""
    return _shared_elexon_files_for_key(_elexon_cache_key(scenario_id))


def _uses_future_bid_stack(scenario_id):
    """Check whether a scenario uses empirical historical-to-future pricing."""
    return _effective_bid_offer_source(scenario_id) == "future_bid_stack"


def _uses_empirical_support_pricing(scenario_id):
    """Check whether a future scenario uses historic CfD/ROC BM evidence."""
    cfg = (
        _scenarios.get(scenario_id, {})
        .get("support", {})
        .get("bm_bids", {})
    )
    return bool(cfg.get("enabled", False)) and str(cfg.get("mode", "fixed")).lower() == "empirical"


def _support_bid_stack_config(scenario_id):
    return (
        _scenarios.get(scenario_id, {})
        .get("support", {})
        .get("bm_bids", {})
    )


def _future_bid_stack_config(scenario_id):
    return (
        _scenarios.get(scenario_id, {})
        .get("market", {})
        .get("balancing", {})
        .get("future_bid_stack", {})
    )


def _future_bid_stack_calibration_inputs(scenario_id, filename):
    """Return declared historical ELEXON artefacts for a future stack."""
    calibration_scenarios = _future_bid_stack_config(scenario_id).get(
        "calibration_scenarios", []
    )
    if not calibration_scenarios:
        raise ValueError(
            f"future_bid_stack scenario '{scenario_id}' must declare calibration_scenarios"
        )
    if filename == "network":
        return [f"{resources_path}/network/{sid}.nc" for sid in calibration_scenarios]
    if filename == "offers":
        return [f"{resources_path}/market/{sid}/elexon/elexon_offers.csv" for sid in calibration_scenarios]
    if filename == "bids":
        return [f"{resources_path}/market/{sid}/elexon/elexon_bids.csv" for sid in calibration_scenarios]
    if filename == "mapping":
        return [f"{resources_path}/generators/{sid}_bmu_mapping.csv" for sid in calibration_scenarios]
    raise ValueError(f"Unknown future bid-stack calibration input '{filename}'")

def _support_bid_stack_extra_inputs(wildcards):
    """Declare optional legal CfD evidence consumed by a support calibration."""
    legal_cfg = _support_bid_stack_config(wildcards.scenario).get("legal_cfd_crosswalk", {}) or {}
    if legal_cfg.get("enabled", False):
        return {"legal_cfd_crosswalk": legal_cfg["file"]}
    return {}

def _support_bid_stack_calibration_inputs(scenario_id, filename):
    """Return declared historical artefacts for an empirical support stack."""
    cfg = _support_bid_stack_config(scenario_id)
    calibration_scenarios = cfg.get("calibration_scenarios", [])
    if not calibration_scenarios:
        raise ValueError(
            f"empirical support-pricing scenario '{scenario_id}' must declare calibration_scenarios"
        )
    if filename == "network":
        return [f"{resources_path}/network/{sid}.nc" for sid in calibration_scenarios]
    if filename == "offers":
        return [f"{resources_path}/market/{sid}/elexon/elexon_offers.csv" for sid in calibration_scenarios]
    if filename == "bids":
        return [f"{resources_path}/market/{sid}/elexon/elexon_bids.csv" for sid in calibration_scenarios]
    if filename == "mapping":
        return [f"{resources_path}/generators/{sid}_bmu_mapping.csv" for sid in calibration_scenarios]
    raise ValueError(f"Unknown support bid-stack calibration input '{filename}'")


def _uses_future_participation_audit(scenario_id):
    """Return true for enabled participation audits without an ELEXON BMU mapping.

    Future scenarios, and historical ones whose BM bids are not ELEXON-sourced
    (``derived``, ``csv``, ``future_bid_stack``): neither has the BMU mapping
    that ``audit_bm_participation`` reads.
    """
    sc = _scenarios.get(scenario_id, {})
    return (
        not _uses_elexon_bid_offer_data(scenario_id)
        and bool(
            sc.get("market", {})
            .get("balancing", {})
            .get("participation_audit", {})
            .get("enabled", False)
        )
    )
def _station_identity_registry_config(scenario_id):
    """Return the source-registry settings inherited by a scenario."""
    return _scenarios.get(scenario_id, {}).get("fleet", {}).get("station_registry", {})


def _uses_station_identity_registry(scenario_id):
    return bool(_station_identity_registry_config(scenario_id).get("enabled", False))


_support_bid_stack_scenario_ids = [rid for rid in _run_ids if _uses_empirical_support_pricing(rid)]
SUPPORT_BID_STACK_SCENARIO_REGEX = (
    _regex_from_list(_support_bid_stack_scenario_ids)
    if _support_bid_stack_scenario_ids
    else r"(?!x)x"
)

_future_bid_stack_scenario_ids = [rid for rid in _run_ids if _uses_future_bid_stack(rid)]
FUTURE_BID_STACK_SCENARIO_REGEX = (
    _regex_from_list(_future_bid_stack_scenario_ids)
    if _future_bid_stack_scenario_ids
    else r"(?!x)x"
)

_future_participation_audit_ids = [
    rid for rid in _run_ids if _uses_future_participation_audit(rid)
]
FUTURE_PARTICIPATION_AUDIT_SCENARIO_REGEX = (
    _regex_from_list(_future_participation_audit_ids)
    if _future_participation_audit_ids
    else r"(?!x)x"
)

_station_identity_registry_ids = sorted({
    str(_station_identity_registry_config(rid).get("registry_id", "gb_public_current"))
    for rid in _run_ids if _uses_station_identity_registry(rid)
})
STATION_IDENTITY_REGISTRY_REGEX = (
    _regex_from_list(_station_identity_registry_ids)
    if _station_identity_registry_ids
    else r"(?!x)x"
)
_station_identity_registry_config_by_id = {
    registry_id: next(
        _station_identity_registry_config(rid)
        for rid in _run_ids
        if _uses_station_identity_registry(rid)
        and str(_station_identity_registry_config(rid).get("registry_id", "gb_public_current")) == registry_id
    )
    for registry_id in _station_identity_registry_ids
}


def _station_identity_registry_inputs(wildcards):
    """Declare only the optional, user-supplied Ofgem report when configured."""
    cfg = _station_identity_registry_config_by_id[wildcards.registry_id]
    inputs = {
        "elexon": f"{resources_path}/market/station_identity/{wildcards.registry_id}/sources/elexon_bmu_reference.csv",
        "lccc": f"{resources_path}/market/station_identity/{wildcards.registry_id}/sources/lccc_cfd_contract_portfolio.csv",
        "neso": f"{resources_path}/market/station_identity/{wildcards.registry_id}/sources/neso_tec_register.csv",
        "repd": cfg.get("repd_file", "data/renewables/repd-q2-jul-2025.csv"),
        "dictionary_ids": cfg.get("power_station_dictionary_ids_file", "data/generators/power_station_dictionary_ids.csv"),
    }
    if ofgem_path := cfg.get("ofgem_ro_accredited_stations_file"):
        inputs["ofgem_ro"] = ofgem_path
    return inputs


_elexon_scenario_ids = [rid for rid in _run_ids if _uses_elexon_bid_offer_data(rid)]
ELEXON_SCENARIO_REGEX = _regex_from_list(_elexon_scenario_ids) if _elexon_scenario_ids else r"(?!x)x"
_elexon_cache_key_to_scenario = {
    _elexon_cache_key(sid): sid for sid in _elexon_scenario_ids
}
ELEXON_CACHE_KEY_REGEX = (
    _regex_from_list(_elexon_cache_key_to_scenario.keys())
    if _elexon_cache_key_to_scenario
    else r"(?!x)x"
)


def _wholesale_position_source_inputs(wildcards):
    """Declare reference position files for controlled adequacy comparisons."""
    source = (
        _scenarios.get(wildcards.scenario, {})
        .get("market", {})
        .get("wholesale", {})
        .get("position_source", {})
        .get("scenario")
    )
    if not source:
        return {}
    base = f"{resources_path}/market/{source}"
    return {
        "reference_wholesale_dispatch_csv": f"{base}_wholesale_dispatch.csv",
        "reference_wholesale_storage_csv": f"{base}_wholesale_storage.csv",
        "reference_wholesale_links_csv": f"{base}_wholesale_links.csv",
        "reference_wholesale_price_csv": f"{base}_wholesale_price.csv",
    }


def _uses_contract_value_bids(scenario_id):
    """True when the BM prices CfD bids on contract value (plan item B1)."""
    cfg = (
        _scenarios.get(scenario_id, {})
        .get("support", {})
        .get("bm_bids", {})
        .get("contract_value_bids", {})
    ) or {}
    return bool(cfg.get("enabled", False))


def _contract_value_strike_table(scenario_id):
    """The published strike table the contract-value bids settle against."""
    block = (
        _scenarios.get(scenario_id, {})
        .get("support", {}).get("labels", {})
        .get("cfd_strike_table", {})
    ) or {}
    return block.get("file")


def _public_ownership_bids_shares_file(scenario_id):
    """The per-generator state-share table the BM bid stack reads (item BM1), if enabled."""
    block = (
        _scenarios.get(scenario_id, {})
        .get("support", {})
        .get("bm_bids", {})
        .get("contract_value_bids", {})
        .get("public_ownership_bids", {})
    ) or {}
    return block.get("shares_file") if block.get("enabled", False) else None


def _public_ownership_bids_years_file(scenario_id):
    """The commissioning-year table P3's tie-break ranks on (decision 20), if declared."""
    block = (
        _scenarios.get(scenario_id, {})
        .get("support", {})
        .get("bm_bids", {})
        .get("contract_value_bids", {})
        .get("public_ownership_bids", {})
    ) or {}
    tie = block.get("tie_break", {}) or {}
    return tie.get("years_file") if block.get("enabled", False) and tie.get("enabled", False) else None


def _wholesale_curtailment_order_inputs(wildcards):
    """The strike table Stage 1 prices the contract-aware curtailment order on.

    Declared so an edited table re-runs Stage 1 as well as the BM, which would
    otherwise order curtailment on old strikes (review 26 Sep 2026).
    """
    sid = wildcards.scenario
    order = (
        _scenarios.get(sid, {}).get("market", {}).get("wholesale", {})
        .get("renewable_curtailment_order", {})
    ) or {}
    if order.get("mode", "as_built") != "contract_aware":
        return {}
    strike_table = _contract_value_strike_table(sid)
    if not strike_table:
        raise ValueError(f"{sid}: renewable_curtailment_order is contract_aware but "
                         "support.labels.cfd_strike_table.file is not configured")
    return {"cfd_strike_table": strike_table}


def _is_central_stage1(scenario_id):
    """True when Stage 1 is a central dispatch on the real network (copperplate_mode none)."""
    return str(
        (_scenarios.get(scenario_id, {}).get("market", {}).get("wholesale", {}) or {})
        .get("copperplate_mode", "collapse")
    ).lower() == "none"


def _wholesale_extra_inputs(wildcards):
    """Data files the Stage 1 solve reads outside the network, declared so an edit re-runs it.

    The scarcity curve (market.wholesale.scarcity_pricing, item C6) and, for a
    central dispatch Stage 1, the secured boundary geometry and capability it
    now enforces (item C1).
    """
    sid = wildcards.scenario
    inputs = {}
    scarcity = (
        _scenarios.get(sid, {}).get("market", {}).get("wholesale", {}).get("scarcity_pricing", {})
    ) or {}
    if scarcity.get("enabled", False):
        inputs["scarcity_curve"] = scarcity.get("curve_file", "data/market/scarcity/scarcity_curve_2024.json")
    if _is_central_stage1(sid):
        inputs.update(_secured_boundary_inputs(sid))
        # A central Stage 1 applies the balancing stage's network delivery case too
        # (solve_wholesale.apply_central_network_delivery, 18 Oct 2026), Appendix B reverts included.
        inputs.update(_appendix_b_projects_inputs(sid))
    # Decision 33: the annual pass trajectory, only when the scenario sets the key.
    inputs.update(_seasonal_trajectory_inputs(sid))
    # Task BM2050: DPA plant dispatched from the rate-free DPA dispatch table (rule
    # build_dpa_dispatch), only when set: a financing-rate change does not re-run Stage 1.
    if _dpa_dispatch_block(sid).get("enabled", False):
        inputs["dpa_dispatch"] = _dpa_dispatch_block(sid)["file"]
    # Decision 29: a TYNDP price sensitivity on a shared 2050 network, only when set.
    inputs.update(_foreign_price_override_inputs(sid))
    # Resource-cost Stage 1 orders renewables at the floor by the support they lose, read from the
    # support labels when the network carries none (marginal_cost_basis.support_order.labels_file,
    # 11 Nov 2026). Declared (14 Nov 2026) so a label rebuild re-runs these solves after it, not before.
    labels_file = _support_order_labels_file(sid)
    if labels_file:
        inputs["support_order_labels"] = labels_file
    return inputs


def _support_order_labels_file(scenario_id):
    """The labels file a resource-cost Stage 1 orders renewables by, or None (scripts/market/marginal_cost_basis.py)."""
    basis = ((_scenarios.get(scenario_id, {}).get("market", {}) or {}).get("wholesale", {}) or {}).get(
        "marginal_cost_basis", {}) or {}
    if str(basis.get("mode", "as_built")) == "as_built":
        return None
    order = basis.get("support_order", {}) or {}
    if not order.get("enabled", True):
        return None
    return order.get("labels_file") or None


def _foreign_price_override_inputs(scenario_id):
    """The TYNDP shape (and tranche demand) a market.foreign_price_override re-prices from.

    Absent in every scenario but the decision-29 gate weeks, so empty everywhere else
    (scripts/interconnectors/foreign_price_override.py).
    """
    sc = _scenarios.get(scenario_id, {})
    inputs = {}
    # Y8 follow-up (2050 only): the IC1 tranche refit reads the ES2 market demand file.
    if ((sc.get("market", {}) or {}).get("ic1_tranche_refit", {}) or {}).get("enabled", False):
        from scripts.interconnectors.foreign_price_override import refit_market_demand_path
        inputs["ic1_refit_market_demand"] = refit_market_demand_path(sc)
    if not ((sc.get("market", {}) or {}).get("foreign_price_override", {}) or {}).get("enabled", False):
        return inputs
    from scripts.interconnectors.foreign_price_override import market_demand_path, shape_path
    inputs["foreign_price_shape"] = shape_path(sc, resources_path)
    demand = market_demand_path(sc)
    if demand:
        inputs["foreign_price_market_demand"] = demand
    return inputs


def _secured_boundary_inputs(scenario_id):
    """Geometry and capability files read by the ETYS secured boundary limits
    (network.transmission.outage_schedule.source: secured_capability), declared so a
    changed data file re-runs the BM solve."""
    outage = (_scenarios.get(scenario_id, {}).get("network", {}).get("transmission", {}) or {}).get("outage_schedule", {}) or {}
    if not (outage.get("enabled", False) and outage.get("source") == "secured_capability"):
        return {}
    sec = outage.get("secured_capability", {}) or {}
    return {
        "secured_boundary_geometry": sec.get(
            "geometry_file", "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson"),
        "secured_boundary_capability": sec.get(
            "capability_file", "data/network/etys_boundary_capability.csv"),
    }


def _appendix_b_projects_inputs(scenario_id):
    """A network-delivery case that reverts named Appendix B projects reads their rows from the
    projects table (scripts/market/network_delivery.py, 15 Oct 2026)."""
    delivery = (((_scenarios.get(scenario_id, {}) or {}).get("market") or {}).get("balancing") or {}).get(
        "network_delivery") or {}
    revert = delivery.get("revert_appendix_b") or {}
    if revert.get("projects"):
        return {"appendix_b_projects": revert.get("projects_file") or "data/network/ETYS/appendix_b_projects.csv"}
    return {}


def _balancing_extra_inputs(wildcards):
    """Return additional calibration or ELEXON inputs required by the BM solve."""
    sid = wildcards.scenario
    inputs = {}
    if _uses_contract_value_bids(sid):
        if _uses_support_labels(sid):
            # Task BM2050: a long-range fleet bids from its support label table (rule
            # build_support_labels), which carries its own strikes.
            inputs["support_labels"] = _support_labels_file(sid)
        else:
            strike_table = _contract_value_strike_table(sid)
            if not strike_table:
                raise ValueError(
                    f"{sid}: contract_value_bids is enabled but "
                    "support.labels.cfd_strike_table.file is not configured"
                )
            inputs["cfd_strike_table"] = strike_table
        # Written by rule build_public_ownership_state_shares, which rebuilds only
        # when forced (its register input is ancient); declared so a rebuilt table
        # re-runs the BM solves. verify_public_ownership_ownership_tables guards it.
        shares_file = _public_ownership_bids_shares_file(sid)
        if shares_file:
            inputs["public_ownership_shares"] = shares_file
        years_file = _public_ownership_bids_years_file(sid)
        if years_file:
            inputs["public_ownership_commissioning_years"] = years_file
    inputs.update(_secured_boundary_inputs(sid))
    inputs.update(_appendix_b_projects_inputs(sid))
    if _uses_future_bid_stack(sid):
        inputs["future_bid_stack_parameters"] = (
            f"{resources_path}/market/{sid}_future_bid_stack_parameters.csv"
        )
    if _uses_empirical_support_pricing(sid):
        inputs["support_bid_stack_parameters"] = (
            f"{resources_path}/market/{sid}_support_bid_stack_parameters.csv"
        )
    if _uses_elexon_bid_offer_data(sid):
        inputs.update({
            "elexon_offers": f"{resources_path}/market/{sid}/elexon/elexon_offers.csv",
            "elexon_bids": f"{resources_path}/market/{sid}/elexon/elexon_bids.csv",
            "bmu_mapping": f"{resources_path}/generators/{sid}_bmu_mapping.csv",
        })
        if _uses_elexon_price_ladders(sid):
            inputs.update({
                "elexon_offer_ladders": f"{resources_path}/market/{sid}/elexon/elexon_offer_ladders.csv",
                "elexon_bid_ladders": f"{resources_path}/market/{sid}/elexon/elexon_bid_ladders.csv",
            })
    # Storage anchored on Stage 1's gross legs, saved on its network (10 Oct 2026).
    anchor = str(
        _scenarios.get(sid, {}).get("market", {}).get("balancing", {}).get("storage_anchor", "net_sign")
    ).lower()
    if anchor == "stage1_legs":
        inputs["wholesale_network"] = f"{resources_path}/market/{_wholesale_source(sid)}_wholesale.nc"
    # Decision 33: the BM starts the hydrogen store at the trajectory's level too.
    inputs.update(_seasonal_trajectory_inputs(sid))
    # Decision 29: the BM re-prices the interconnectors as Stage 1 did.
    inputs.update(_foreign_price_override_inputs(sid))
    return inputs

#
# PUBLIC STATION-IDENTITY REGISTRY
#
# This is deliberately a data-preparation stage, not a heuristic mapping rule.
# It captures the authoritative source records and their provenance so a later
# matching stage can score BMU -> station -> CfD/RO evidence transparently.
#

rule fetch_station_identity_sources:
    """Fetch the current public Elexon, LCCC and NESO source tables once."""
    output:
        elexon=f"{resources_path}/market/station_identity/{{registry_id}}/sources/elexon_bmu_reference.csv",
        lccc=f"{resources_path}/market/station_identity/{{registry_id}}/sources/lccc_cfd_contract_portfolio.csv",
        neso=f"{resources_path}/market/station_identity/{{registry_id}}/sources/neso_tec_register.csv",
        manifest=f"{resources_path}/market/station_identity/{{registry_id}}/sources/source_manifest.json",
    log:
        "logs/market/fetch_station_identity_sources_{registry_id}.log"
    benchmark:
        "benchmarks/market/fetch_station_identity_sources_{registry_id}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        registry_id=STATION_IDENTITY_REGISTRY_REGEX
    params:
        mode="fetch",
        source_dir=lambda wildcards: f"{resources_path}/market/station_identity/{wildcards.registry_id}/sources",
    script:
        "../scripts/market/station_identity_registry.py"


rule build_station_identity_registry:
    """Create an auditable, source-preserving public station registry."""
    input:
        unpack(_station_identity_registry_inputs)
    output:
        registry=f"{resources_path}/market/station_identity/{{registry_id}}/station_identity_registry.csv",
        evidence=f"{resources_path}/market/station_identity/{{registry_id}}/station_identity_registry_evidence.json",
    log:
        "logs/market/build_station_identity_registry_{registry_id}.log"
    benchmark:
        "benchmarks/market/build_station_identity_registry_{registry_id}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        registry_id=STATION_IDENTITY_REGISTRY_REGEX
    params:
        mode="build",
    script:
        "../scripts/market/station_identity_registry.py"

rule build_bmu_support_crosswalk:
    """Score explicit dictionary BMU-to-CfD/ROC evidence without pricing changes."""
    input:
        dictionary=lambda wildcards: _station_identity_registry_config_by_id[wildcards.registry_id].get(
            "power_station_dictionary_ids_file", "data/generators/power_station_dictionary_ids.csv"
        ),
        registry=f"{resources_path}/market/station_identity/{{registry_id}}/station_identity_registry.csv",
        repd=lambda wildcards: _station_identity_registry_config_by_id[wildcards.registry_id].get(
            "repd_file", "data/renewables/repd-q2-jul-2025.csv"
        ),
    output:
        crosswalk=f"{resources_path}/market/station_identity/{{registry_id}}/bmu_station_support_crosswalk.csv",
        review=f"{resources_path}/market/station_identity/{{registry_id}}/bmu_station_support_review.csv",
        summary=f"{resources_path}/market/station_identity/{{registry_id}}/bmu_station_support_summary.json",
    log:
        "logs/market/build_bmu_support_crosswalk_{registry_id}.log"
    benchmark:
        "benchmarks/market/build_bmu_support_crosswalk_{registry_id}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        registry_id=STATION_IDENTITY_REGISTRY_REGEX
    script:
        "../scripts/market/build_bmu_support_crosswalk.py"
# 
# OPTIONAL: BUILD BMU-TO-GENERATOR MAPPING (ELEXON SCENARIOS ONLY)
# 

rule build_bmu_mapping:
    """
    Build mapping from ELEXON BMU IDs to PyPSA generator names.

    Uses ETYS Dir_con_BMUs_to_node sheet and station-to-prefix lookups
    to map BMU IDs (e.g., T_PEMB-11) to generator names (e.g., Pembroke).

    Only runs for scenarios with bid_offer_source: "elexon".
    """
    input:
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        elexon_offers=f"{resources_path}/market/{{scenario}}/elexon/elexon_offers.csv",
        power_station_dictionary="data/generators/power_station_dictionary_bmu_crosswalk.csv",
    output:
        bmu_mapping=f"{resources_path}/generators/{{scenario}}_bmu_mapping.csv",
    log:
        "logs/market/build_bmu_mapping_{scenario}.log"
    benchmark:
        "benchmarks/market/build_bmu_mapping_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=ELEXON_SCENARIO_REGEX
    params:
        etys_path="data/network/ETYS/GB_network.xlsx",
        elexon_offers_path=lambda wildcards: f"{resources_path}/market/{wildcards.scenario}/elexon/elexon_offers.csv",
        power_station_dictionary_path="data/generators/power_station_dictionary_bmu_crosswalk.csv",
    script:
        "../scripts/generators/build_bmu_mapping.py"


# 
# OPTIONAL: RETRIEVE ELEXON MARKET DATA (HISTORICAL SCENARIOS ONLY)
# 

rule retrieve_elexon_market_data_shared:
    """
    Build a shared processed ELEXON BMRS bid/offer cache for a date range.

    Identical scenario date ranges and ladder settings share this output, so
    scenario variants do not rebuild the expensive BOD aggregation.

    Output:
      - ELEXON offer prices: elexon/elexon_offers.csv
      - ELEXON bid prices: elexon/elexon_bids.csv
    """
    output:
        offers_file=f"{resources_path}/market/elexon_shared/{{cache_key}}/elexon_offers.csv",
        bids_file=f"{resources_path}/market/elexon_shared/{{cache_key}}/elexon_bids.csv",
        offer_ladders_file=f"{resources_path}/market/elexon_shared/{{cache_key}}/elexon_offer_ladders.csv",
        bid_ladders_file=f"{resources_path}/market/elexon_shared/{{cache_key}}/elexon_bid_ladders.csv",
    log:
        "logs/market/retrieve_elexon_shared_{cache_key}.log"
    benchmark:
        "benchmarks/market/retrieve_elexon_shared_{cache_key}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        cache_key=ELEXON_CACHE_KEY_REGEX
    params:
        scenario_config=lambda wildcards: scenario_param(_scenarios.get(_elexon_cache_key_to_scenario[wildcards.cache_key], {})),
        # The window the cache key was built from, so script and key agree.
        solve_dates=lambda wildcards: list(_elexon_solve_dates(_elexon_cache_key_to_scenario[wildcards.cache_key])),
    script:
        "../scripts/market/elexon_data.py"


rule retrieve_elexon_market_data:
    """
    Copy shared ELEXON bid/offer files into the scenario output location.

    Downstream rules keep using resources/market/{scenario}/elexon/*.csv, while
    the expensive retrieval/aggregation is keyed by date range and ladder config.
    """
    input:
        offers_file=lambda wildcards: _shared_elexon_files_for_scenario(wildcards.scenario)["offers_file"],
        bids_file=lambda wildcards: _shared_elexon_files_for_scenario(wildcards.scenario)["bids_file"],
        offer_ladders_file=lambda wildcards: _shared_elexon_files_for_scenario(wildcards.scenario)["offer_ladders_file"],
        bid_ladders_file=lambda wildcards: _shared_elexon_files_for_scenario(wildcards.scenario)["bid_ladders_file"],
    output:
        offers_file=f"{resources_path}/market/{{scenario}}/elexon/elexon_offers.csv",
        bids_file=f"{resources_path}/market/{{scenario}}/elexon/elexon_bids.csv",
        offer_ladders_file=f"{resources_path}/market/{{scenario}}/elexon/elexon_offer_ladders.csv",
        bid_ladders_file=f"{resources_path}/market/{{scenario}}/elexon/elexon_bid_ladders.csv",
    log:
        "logs/market/retrieve_elexon_{scenario}.log"
    benchmark:
        "benchmarks/market/retrieve_elexon_{scenario}.txt"
    wildcard_constraints:
        scenario=ELEXON_SCENARIO_REGEX
    run:
        import shutil
        from pathlib import Path

        for source, target in zip(input, output):
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_path)


# 
# STAGE 1: WHOLESALE MARKET (COPPERPLATE DISPATCH)
# Consumer-bill rules are owned by rules/consumer_bill.smk.

# STAGE 2: BALANCING MECHANISM (ANCHORED REDISPATCH)
# 

rule calibrate_support_bid_stack:
    """Calibrate future CfD/ROC BM prices from high-quality historic ELEXON matches."""
    input:
        unpack(_support_bid_stack_extra_inputs),
        calibration_networks=lambda wildcards: _support_bid_stack_calibration_inputs(wildcards.scenario, "network"),
        calibration_offers=lambda wildcards: _support_bid_stack_calibration_inputs(wildcards.scenario, "offers"),
        calibration_bids=lambda wildcards: _support_bid_stack_calibration_inputs(wildcards.scenario, "bids"),
        calibration_mappings=lambda wildcards: _support_bid_stack_calibration_inputs(wildcards.scenario, "mapping"),
    output:
        parameters_csv=f"{resources_path}/market/{{scenario}}_support_bid_stack_parameters.csv",
        evidence_csv=f"{resources_path}/market/{{scenario}}_support_bid_stack_evidence.csv",
    log:
        "logs/market/calibrate_support_bid_stack_{scenario}.log"
    benchmark:
        "benchmarks/market/calibrate_support_bid_stack_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=SUPPORT_BID_STACK_SCENARIO_REGEX
    params:
        support_bid_stack=lambda wildcards: _support_bid_stack_config(wildcards.scenario),
    script:
        "../scripts/market/support_bid_stack.py"

rule calibrate_future_bid_stack:
    """Calibrate portable ELEXON residuals for a CP30 or FES future scenario."""
    input:
        calibration_networks=lambda wildcards: _future_bid_stack_calibration_inputs(wildcards.scenario, "network"),
        calibration_offers=lambda wildcards: _future_bid_stack_calibration_inputs(wildcards.scenario, "offers"),
        calibration_bids=lambda wildcards: _future_bid_stack_calibration_inputs(wildcards.scenario, "bids"),
        calibration_mappings=lambda wildcards: _future_bid_stack_calibration_inputs(wildcards.scenario, "mapping"),
    output:
        parameters_csv=f"{resources_path}/market/{{scenario}}_future_bid_stack_parameters.csv",
        evidence_csv=f"{resources_path}/market/{{scenario}}_future_bid_stack_evidence.csv",
    log:
        "logs/market/calibrate_future_bid_stack_{scenario}.log"
    benchmark:
        "benchmarks/market/calibrate_future_bid_stack_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=FUTURE_BID_STACK_SCENARIO_REGEX
    params:
        future_bid_stack=lambda wildcards: _future_bid_stack_config(wildcards.scenario),
    script:
        "../scripts/market/future_bid_stack.py"

rule solve_wholesale_market:
    """
    Stage 1: Solve wholesale market with copperplate (unconstrained) dispatch.

    Removes all network constraints by setting line/transformer ratings to
    a very large value, producing a single uniform clearing price. This
    represents the GB wholesale electricity market.

    Input:
      - Finalized network with all components: {scenario}.nc

    Output:
      - Wholesale solved network: {scenario}_wholesale.nc
      - Generator dispatch: {scenario}_wholesale_dispatch.csv
      - Storage dispatch: {scenario}_wholesale_storage.csv
      - Link dispatch: {scenario}_wholesale_links.csv
      - Uniform price: {scenario}_wholesale_price.csv
    """
    input:
        unpack(_wholesale_position_source_inputs),
        unpack(_wholesale_curtailment_order_inputs),
        unpack(_wholesale_extra_inputs),
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        # Item N1: the build-time cost audit lets a resource-cost basis strip
        # subsidy layers exactly (scripts/market/marginal_cost_basis.py).
        marginal_costs_csv=lambda wildcards: f"{resources_path}/generators/{_build_source(wildcards.scenario)}_marginal_costs_breakdown.csv"
    output:
        network=f"{resources_path}/market/{{scenario}}_wholesale.nc",
        wholesale_dispatch_csv=f"{resources_path}/market/{{scenario}}_wholesale_dispatch.csv",
        wholesale_storage_csv=f"{resources_path}/market/{{scenario}}_wholesale_storage.csv",
        wholesale_links_csv=f"{resources_path}/market/{{scenario}}_wholesale_links.csv",
        wholesale_price_csv=f"{resources_path}/market/{{scenario}}_wholesale_price.csv",
    log:
        "logs/market/solve_wholesale_{scenario}.log"
    benchmark:
        "benchmarks/market/solve_wholesale_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    # Two threads: barrier gains nothing past two on these LPs, and 2 lets
    # `--cores 8` run four months at once (see solve.solver.threads in config/defaults/base/solve.yaml).
    threads: 2
    resources:
        # Measured peak RSS per solve on the CP30 2030 network; only enforced
        # when the launch passes `--resources mem_mb=<budget>`. See the
        # balancing rule for why the budget matters. A central dispatch
        # (copperplate_mode "none") keeps the whole network and frees every
        # generator against it, so it needs far more than a collapsed solve.
        # Measured 5.84 GB peak for a central dispatch day on this network.
        mem_mb=lambda wildcards: (
            6500 if str(
                _scenarios.get(wildcards.scenario, {})
                .get("market", {}).get("wholesale", {})
                .get("copperplate_mode", "collapse")
            ).lower() == "none" else 3500
        )
    wildcard_constraints:
        scenario=WHOLESALE_SOLVE_REGEX
    params:
        scenario_config=lambda wildcards: model_scenario_param(_scenarios.get(wildcards.scenario, {})),
        solver=lambda wildcards: _get_solver_config_market(wildcards.scenario)[0],
        solver_options=lambda wildcards: _get_solver_config_market(wildcards.scenario)[1],
    script:
        "../scripts/market/solve_wholesale.py"


# The same rule for central dispatch Stage 1 jobs, with scheduling priority so a
# full run starts its longest solves first (item C12, 27 Sep 2026). Identical
# inputs, outputs and script; only the wildcard set and the priority differ.
use rule solve_wholesale_market as solve_wholesale_market_central with:
    priority: CENTRAL_WHOLESALE_PRIORITY
    wildcard_constraints:
        scenario=WHOLESALE_CENTRAL_SOLVE_REGEX


rule solve_balancing_mechanism:
    """
    Stage 2: Solve balancing mechanism with full network constraints.

    Starting from wholesale dispatch positions, re-solves with actual
    line/transformer ratings. Generators and storage are anchored to their
    wholesale position, with increase/decrease variables penalised by
    bid/offer prices  mimicking the GB Balancing Mechanism.

    Input:
      - Finalized network (original constraints): {scenario}.nc
      - Wholesale dispatch CSVs from Stage 1

    Output:
      - Balancing solved network: {scenario}_balancing.nc
      - Physical dispatch: {scenario}_balancing_dispatch.csv
      - Physical storage dispatch: {scenario}_balancing_storage.csv
      - Redispatch summary: {scenario}_redispatch_summary.csv
      - Constraint costs: {scenario}_constraint_costs.csv
      - Congestion analysis: {scenario}_congestion.csv
      - Price comparison: {scenario}_price_comparison.csv
    """
    input:
        unpack(_balancing_extra_inputs),
        boundary_mapping_yaml=lambda wildcards: _neso_boundary_mapping_path(wildcards.scenario),
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        wholesale_dispatch_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_dispatch.csv",
        wholesale_storage_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_storage.csv",
        wholesale_links_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_links.csv",
        wholesale_price_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_price.csv",
    output:
        network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        balancing_dispatch_csv=f"{resources_path}/market/{{scenario}}_balancing_dispatch.csv",
        balancing_storage_csv=f"{resources_path}/market/{{scenario}}_balancing_storage.csv",
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        congestion_csv=f"{resources_path}/market/{{scenario}}_congestion.csv",
        price_comparison_csv=f"{resources_path}/market/{{scenario}}_price_comparison.csv",
        contract_value_bids_csv=f"{resources_path}/market/{{scenario}}_contract_value_bids.csv",
        adequacy_relaxation_csv=f"{resources_path}/analysis/{{scenario}}_adequacy_relaxation.csv",
    log:
        "logs/market/solve_balancing_{scenario}.log"
    benchmark:
        "benchmarks/market/solve_balancing_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    # Two threads: barrier gains nothing past two on these LPs, and 2 lets
    # `--cores 8` run four months at once (see solve.solver.threads in config/defaults/base/solve.yaml).
    threads: 2
    resources:
        # Measured peak RSS per solve on the CP30 2030 network (the anchored BM
        # model is the memory-hungry stage). Four at once needs ~24 GB, which is
        # what this 31 GB box has left after the OS and background apps: on
        # 19 Sep 2026 the September solve died in create_model with "Unable to
        # allocate 350 MiB" after 45 minutes. Snakemake only enforces this when
        # the launch passes a budget, so use
        #   --cores 8 --resources mem_mb=20000
        # to cap the BM stage at three concurrent solves.
        mem_mb=6500
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: model_scenario_param(_scenarios.get(wildcards.scenario, {})),
        solver=lambda wildcards: _get_solver_config_market(wildcards.scenario)[0],
        solver_options=lambda wildcards: _get_solver_config_market(wildcards.scenario)[1],
    script:
        "../scripts/market/solve_balancing.py"


# 
# ANALYSIS: MARKET DASHBOARD AND SUMMARY
# 

rule analyze_market_results:
    """
    Post-processing analysis of two-stage market simulation results.

    Compares wholesale and balancing dispatch to produce:
    - Interactive dashboard (HTML) with 6 panels
    - Machine-readable summary (JSON) with key metrics

    Input:
      - Both solved networks (.nc) and all result CSVs

    Output:
      - Market dashboard: {scenario}_market_dashboard.html
      - Market summary: {scenario}_market_summary.json
    """
    input:
        # A scenario declaring wholesale_from (item BM1) analyses the parent's
        # Stage 1 beside its own Stage 2, as its balancing solve did; naming its
        # own wholesale files would re-solve an identical Stage 1.
        wholesale_links_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_links.csv",
        wholesale_network=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale.nc",
        balancing_network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        wholesale_dispatch_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_dispatch.csv",
        balancing_dispatch_csv=f"{resources_path}/market/{{scenario}}_balancing_dispatch.csv",
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        congestion_csv=f"{resources_path}/market/{{scenario}}_congestion.csv",
        price_comparison_csv=f"{resources_path}/market/{{scenario}}_price_comparison.csv",
    output:
        accounting_csv=f"{resources_path}/analysis/{{scenario}}_market_accounting.csv",
        interconnector_flows_csv=f"{resources_path}/analysis/{{scenario}}_interconnector_flows.csv",
        dashboard=f"{resources_path}/analysis/{{scenario}}_market_dashboard.html",
        summary_json=f"{resources_path}/analysis/{{scenario}}_market_summary.json",
    log:
        "logs/market/analyze_market_{scenario}.log"
    benchmark:
        "benchmarks/market/analyze_market_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: scenario_param(_scenarios.get(wildcards.scenario, {})),
    script:
        "../scripts/market/analyze_market.py"


rule audit_cp30_offshore_export_links:
    """Audit offshore export capacities, landing buses and realised BM use."""
    input:
        base_network=lambda wildcards: _finalized_network(wildcards.scenario),
        balancing_network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        mapping_csv=lambda wildcards: _scenarios[wildcards.scenario]["fleet"]["spatial_proxy"][
            "offshore_connection_allocation"
        ]["file"],
    output:
        audit_csv=f"{resources_path}/analysis/{{scenario}}_offshore_link_audit.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_offshore_link_audit_summary.json",
    log:
        "logs/market/audit_cp30_offshore_export_links_{scenario}.log"
    benchmark:
        "benchmarks/market/audit_cp30_offshore_export_links_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=OFFSHORE_LINK_AUDIT_REGEX
    script:
        "../scripts/market/audit_cp30_offshore_links.py"


rule audit_cp30_offshore_landing_mapping:
    """Create the evidence/review register for an offshore landing sensitivity."""
    input:
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        mapping_csv=lambda wildcards: _scenarios[wildcards.scenario]["fleet"]["spatial_proxy"][
            "offshore_connection_allocation"
        ]["file"],
    output:
        review_csv=f"{resources_path}/analysis/{{scenario}}_offshore_landing_mapping_review.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_offshore_landing_mapping_review_summary.json",
    log:
        "logs/market/audit_cp30_offshore_landing_mapping_{scenario}.log"
    benchmark:
        "benchmarks/market/audit_cp30_offshore_landing_mapping_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=OFFSHORE_LANDING_MAPPING_AUDIT_REGEX
    script:
        "../scripts/CP30/audit_offshore_landing_mapping.py"


rule audit_cp30_offshore_project_crosswalk:
    """Check landing evidence against a versioned TEC snapshot, without inferring proxy projects."""
    input:
        mapping_csv=lambda wildcards: _scenarios[wildcards.scenario]["fleet"]["spatial_proxy"][
            "offshore_connection_allocation"
        ]["file"],
        evidence_register=lambda wildcards: _scenarios[wildcards.scenario]["fleet"]["spatial_proxy"][
            "offshore_connection_allocation"
        ]["project_evidence_register"],
        tec_register=lambda wildcards: _scenarios[wildcards.scenario]["fleet"]["spatial_proxy"][
            "offshore_connection_allocation"
        ]["tec_register"],
    output:
        audit_csv=f"{resources_path}/analysis/{{scenario}}_offshore_project_crosswalk_audit.csv",
        summary_json=f"{resources_path}/analysis/{{scenario}}_offshore_project_crosswalk_summary.json",
    log:
        "logs/market/audit_cp30_offshore_project_crosswalk_{scenario}.log"
    benchmark:
        "benchmarks/market/audit_cp30_offshore_project_crosswalk_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=OFFSHORE_PROJECT_CROSSWALK_AUDIT_REGEX
    params:
        modelled_year=lambda wildcards: _scenarios[wildcards.scenario]["modelled_year"]
    script:
        "../scripts/CP30/audit_offshore_project_crosswalk.py"


# 
# BM VALIDATION: COMPARE MODEL vs ELEXON ACTUALS (HISTORICAL ONLY)
# 

def _is_historical_market(scenario_id):
    """Check if scenario is a historical market scenario (year  2024)."""
    sc = _scenarios.get(scenario_id, {})
    if not sc.get("market", {}).get("enabled", False):
        return False
    year = sc.get("modelled_year", 9999)
    return year <= 2024


def _bm_validation_mapping_inputs(wildcards):
    """The BMU mapping, for the historical market scenarios that have one."""
    if _uses_elexon_bid_offer_data(wildcards.scenario):
        return {"bmu_mapping": f"{resources_path}/generators/{wildcards.scenario}_bmu_mapping.csv"}
    return {}


_historical_market_ids = [rid for rid in _run_ids if _is_historical_market(rid)]
HISTORICAL_MARKET_REGEX = _regex_from_list(_historical_market_ids) if _historical_market_ids else r"(?!x)x"


rule diagnose_topology_adequacy:
    """Controlled WP4 adequacy diagnosis and capacity-relaxation audit."""
    input:
        balancing_network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        relaxation_csv=f"{resources_path}/analysis/{{scenario}}_adequacy_relaxation.csv",
    output:
        summary_json=f"{resources_path}/analysis/{{scenario}}_topology_adequacy_summary.json",
        binding_components_csv=f"{resources_path}/analysis/{{scenario}}_adequacy_binding_components.csv",
        shedding_by_bus_csv=f"{resources_path}/analysis/{{scenario}}_adequacy_shedding_by_bus.csv",
    log:
        "logs/analysis/diagnose_topology_adequacy_{scenario}.log"
    benchmark:
        "benchmarks/analysis/diagnose_topology_adequacy_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=TOPOLOGY_ADEQUACY_SCENARIO_REGEX
    params:
        topology_adequacy=lambda wc: _scenarios[wc.scenario].get("network", {}).get("topology_adequacy", {})
    script:
        "../scripts/analysis/diagnose_adequacy_controls.py"

rule validate_bm_results:
    """
    Validate BM results against ELEXON actuals for historical scenarios.

    Fetches (or loads cached) ELEXON BOAV, BOALF, system prices, and full B1610
    generation data and compares against model redispatch volumes, constraint
    costs, prices, and dispatch levels.

    Output:
      - Validation report CSV: per-metric model vs ELEXON comparison
      - Validation dashboard HTML: multi-panel interactive comparison
            - BOAV carrier CSV: accepted bid/offer volumes by carrier
            - BOALF flag split CSV: timing/flag diagnostic volumes
            - DISBSAD summary CSV: non-BM balancing-service adjustment totals
    """
    input:
        # The BMU mapping only exists for ELEXON-bid scenarios; validate_bm.py
        # falls back to inferred BMU carriers without it.
        unpack(_bm_validation_mapping_inputs),
        wholesale_dispatch_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_dispatch.csv",
        balancing_dispatch_csv=f"{resources_path}/market/{{scenario}}_balancing_dispatch.csv",
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        price_comparison_csv=f"{resources_path}/market/{{scenario}}_price_comparison.csv",
    output:
        validation_csv=f"{resources_path}/market/{{scenario}}_bm_validation.csv",
        validation_html=f"{resources_path}/analysis/{{scenario}}_bm_validation.html",
        boav_by_carrier_csv=f"{resources_path}/market/{{scenario}}_boav_by_carrier.csv",
        boalf_by_flag_csv=f"{resources_path}/market/{{scenario}}_boalf_by_flag.csv",
        disbsad_summary_csv=f"{resources_path}/market/{{scenario}}_disbsad_summary.csv",
        mapped_redispatch_csv=f"{resources_path}/market/{{scenario}}_redispatch_mapped_unmapped.csv",
    log:
        "logs/market/validate_bm_{scenario}.log"
    benchmark:
        "benchmarks/market/validate_bm_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=HISTORICAL_MARKET_REGEX
    params:
        scenario_config=lambda wildcards: {
            **_scenarios.get(wildcards.scenario, {}),
            "scenario_id": wildcards.scenario,
        },
    script:
        "../scripts/market/validate_bm.py"


# 
# BM CALIBRATION SCORECARD: PER-CARRIER MODEL vs ELEXON PRICES + VOLUMES
# 


rule validate_bm_calibration:
    """
    Per-carrier scorecard comparing model effective bid/offer prices against
    ELEXON BOD medians and model redispatch volumes against BOAV accepted
    bid/offer volumes. Flags carriers that drift beyond the configured
    tolerance (default 20%) and optionally fails the rule.

    Only runs for scenarios that actually consume ELEXON data (historical
    scenarios with bid_offer_source resolving to "elexon"), since the
    scorecard requires the BMU mapping + BOD CSVs.

    Output:
      - Calibration CSV: per-carrier ratios and volumes
      - Calibration dashboard HTML: 4-panel ratio bars + regression flags
    """
    input:
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        boav_by_carrier_csv=f"{resources_path}/market/{{scenario}}_boav_by_carrier.csv",
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        elexon_offers=f"{resources_path}/market/{{scenario}}/elexon/elexon_offers.csv",
        elexon_bids=f"{resources_path}/market/{{scenario}}/elexon/elexon_bids.csv",
        bmu_mapping=f"{resources_path}/generators/{{scenario}}_bmu_mapping.csv",
    output:
        calibration_csv=f"{resources_path}/market/{{scenario}}_bm_calibration.csv",
        calibration_html=f"{resources_path}/analysis/{{scenario}}_bm_calibration.html",
    log:
        "logs/market/validate_bm_calibration_{scenario}.log"
    benchmark:
        "benchmarks/market/validate_bm_calibration_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=ELEXON_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: {
            **_scenarios.get(wildcards.scenario, {}),
            "scenario_id": wildcards.scenario,
        },
    script:
        "../scripts/market/bm_calibration_scorecard.py"


rule audit_bm_participation:
    """Audit ELEXON mapping and configured BM participation by carrier and geography."""
    input:
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        bmu_mapping=f"{resources_path}/generators/{{scenario}}_bmu_mapping.csv",
    output:
        participation_csv=f"{resources_path}/market/{{scenario}}_bm_participation.csv",
    log:
        "logs/market/audit_bm_participation_{scenario}.log"
    benchmark:
        "benchmarks/market/audit_bm_participation_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=ELEXON_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: {
            **_scenarios.get(wildcards.scenario, {}),
            "scenario_id": wildcards.scenario,
        },
    script:
        "../scripts/market/bm_participation_audit.py"


# 
rule audit_future_bm_participation:
    """Audit CP30/future BM eligibility without claiming historic BMU coverage."""
    input:
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        network=lambda wildcards: _finalized_network(wildcards.scenario),
    output:
        participation_csv=f"{resources_path}/market/{{scenario}}_bm_participation.csv",
    log:
        "logs/market/audit_bm_participation_{scenario}.log"
    benchmark:
        "benchmarks/market/audit_bm_participation_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=FUTURE_PARTICIPATION_AUDIT_SCENARIO_REGEX
    params:
        scenario_config=lambda wildcards: {
            **_scenarios.get(wildcards.scenario, {}),
            "scenario_id": wildcards.scenario,
        },
    script:
        "../scripts/market/bm_participation_audit.py"

# NESO CONSTRAINT VALIDATION: MODEL vs NESO THERMAL COSTS & DA FLOWS
# 

rule validate_neso_constraints:
    """
    Validate model BM results against NESO published constraint data.

    Compares model boundary flows and total BM cost against:
      - NESO Thermal Constraint Costs (daily, by boundary)
      - NESO Day-Ahead Constraint Flows & Limits (half-hourly, by boundary)

    Downloads NESO data from the API on first run and caches in data/validation/.
    Uses the boundary-to-line mapping in data/network/neso_boundary_mapping.yaml
    to aggregate model line flows into NESO-comparable boundary transfers.

    Output:
      - Validation CSV: per-metric model vs NESO comparison
      - Validation dashboard HTML: 6-panel interactive comparison
    """
    input:
        balancing_network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        congestion_csv=f"{resources_path}/market/{{scenario}}_congestion.csv",
        boundary_mapping_yaml=lambda wildcards: _neso_boundary_mapping_path(wildcards.scenario),
    output:
        validation_csv=f"{resources_path}/market/{{scenario}}_neso_validation.csv",
        validation_html=f"{resources_path}/analysis/{{scenario}}_neso_validation.html",
    log:
        "logs/market/validate_neso_{scenario}.log"
    benchmark:
        "benchmarks/market/validate_neso_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=HISTORICAL_MARKET_REGEX
    params:
        scenario_config=lambda wildcards: {
            **_scenarios.get(wildcards.scenario, {}),
            "scenario_id": wildcards.scenario,
        },
        boundary_mapping_path=lambda wildcards: _neso_boundary_mapping_path(wildcards.scenario),
        cache_dir="data/validation",
    script:
        "../scripts/market/validate_neso_constraints.py"


# 
# ANALYSIS NOTEBOOK: PER-SCENARIO MARKET ANALYSIS
# 

rule generate_market_analysis_notebook:
    """
    Generate a per-scenario Jupyter notebook for two-stage market analysis.

    Produces a self-contained .ipynb with eight analysis sections:
      1. Wholesale dispatch overview (generation + storage by carrier)
      2. Wholesale price time series and copperplate diagnostic
      3. BM redispatch volume by carrier
      4. BM constraint costs by carrier (stacked bar + pie)
      5. Top BM assets by increase / decrease
      6. Network congestion (hours congested, mean loading fraction)
      7. Wholesale vs BM nodal prices comparison
      8. Summary statistics table

    Paths to all input CSVs are embedded in the notebook at generation time,
    so the notebook can be opened and re-run from any working directory.

    Input:
      - All market CSVs produced by Stage 1 + Stage 2 rules
      - Both solved network .nc files (for carrier lookup)

    Output:
      - {scenario}_market_notebook.ipynb   resources/analysis/

    Usage:
      snakemake resources/analysis/HT35_market_market_notebook.ipynb --cores 1
    """
    input:
        wholesale_network=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale.nc",
        balancing_network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        wholesale_dispatch_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_dispatch.csv",
        wholesale_storage_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_storage.csv",
        wholesale_links_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_links.csv",
        wholesale_price_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_price.csv",
        balancing_dispatch_csv=f"{resources_path}/market/{{scenario}}_balancing_dispatch.csv",
        redispatch_summary_csv=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        constraint_costs_csv=f"{resources_path}/market/{{scenario}}_constraint_costs.csv",
        congestion_csv=f"{resources_path}/market/{{scenario}}_congestion.csv",
        price_comparison_csv=f"{resources_path}/market/{{scenario}}_price_comparison.csv",
    output:
        notebook=f"{resources_path}/analysis/{{scenario}}_market_notebook.ipynb",
    log:
        "logs/market/generate_market_notebook_{scenario}.log"
    benchmark:
        "benchmarks/market/generate_market_notebook_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    script:
        "../scripts/market/generate_market_analysis_notebook.py"


# WHOLESALE-ONLY ANALYSIS NOTEBOOK
# 

# Historical direct-Link interconnector calibration. This rule is targeted only
# when a scenario explicitly selects interconnectors.historical_flow_mode:
# market_coupled. It reports comparison metrics; it never fixes model flows.
rule analyse_historical_interconnector_backtest:
    input:
        wholesale_links=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_links.csv",
        historical_flows=lambda wc: (
            f"{resources_path}/interconnectors/historical_flows_"
            f"{scenarios[wc.scenario]['modelled_year']}.csv"
        ),
        interconnectors=lambda wc: (
            f"{resources_path}/interconnectors/{_build_source(wc.scenario)}_interconnectors_mapped.csv"
        )
    output:
        summary=f"{resources_path}/analysis/{{scenario}}_interconnector_backtest_summary.csv",
        timeseries=f"{resources_path}/analysis/{{scenario}}_interconnector_backtest_timeseries.csv"
    log:
        "logs/market/analyse_interconnector_backtest_{scenario}.log"
    benchmark:
        "benchmarks/market/analyse_interconnector_backtest_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    script:
        "../scripts/interconnectors/analyse_historical_market_coupling.py"
rule generate_wholesale_notebook:
    """
    Generate a per-scenario Jupyter notebook for wholesale market analysis.

    Produces a self-contained .ipynb with four analysis sections:
      1. Setup  libraries, file paths, data loading
      2. Merit Order  supply curve (unit-level)
      3. Dispatch Stack  stacked area chart by carrier
      4. Wholesale Price vs Market  SMP overlaid with ELEXON MID

    Used for scenarios with market.wholesale_only: true (Stage 1 only, no BM).

    Input:
      - Wholesale CSVs (dispatch, storage, links, price) from solve_wholesale_market
      - Solved wholesale network .nc (for carrier/capacity lookup)
      - Marginal costs breakdown CSV (for merit order plot)
      - scenario_config param (for modelled_year + solve_period for MID fetch)

    Output:
      - {scenario}_wholesale_notebook.ipynb   resources/analysis/

    Usage:
      snakemake resources/analysis/Historical_2024_lowwind_wholesale_notebook.ipynb --cores 1
    """
    input:
        wholesale_network=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale.nc",
        wholesale_dispatch_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_dispatch.csv",
        wholesale_storage_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_storage.csv",
        wholesale_links_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_links.csv",
        wholesale_price_csv=lambda wildcards: f"{resources_path}/market/{_wholesale_source(wildcards.scenario)}_wholesale_price.csv",
        marginal_costs_csv=lambda wildcards: f"{resources_path}/generators/{_build_source(wildcards.scenario)}_marginal_costs_breakdown.csv",
    output:
        notebook=f"{resources_path}/analysis/{{scenario}}_wholesale_notebook.ipynb",
    params:
        scenario_config=lambda wildcards: scenario_param(_scenarios.get(wildcards.scenario, {})),
    log:
        "logs/market/generate_wholesale_notebook_{scenario}.log"
    benchmark:
        "benchmarks/market/generate_wholesale_notebook_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=MARKET_SCENARIO_REGEX
    script:
        "../scripts/market/generate_wholesale_notebook.py"


#
# SEASONAL STORAGE: THE ANNUAL PASS (plan decision 33, task Y8a)
#
# A scenario with market.hydrogen_store.seasonal_trajectory.enabled solves its months
# on the hydrogen store trajectory of one annual copperplate pass over its parent
# network (network_from). The pass runs once per parent; its scenario wildcard
# matches only parents of run ids that set the key, so no other scenario's DAG
# changes (the 2030 scenarios never set it).
#

def _seasonal_trajectory_block(scenario_id):
    """``market.hydrogen_store.seasonal_trajectory`` of a scenario ({} when absent)."""
    return (
        (_scenarios.get(scenario_id, {}).get("market", {}) or {}).get("hydrogen_store", {}) or {}
    ).get("seasonal_trajectory", {}) or {}


def _uses_seasonal_trajectory(scenario_id):
    return bool(_seasonal_trajectory_block(scenario_id).get("enabled", False))


def _seasonal_trajectory_file(scenario_id):
    """The trajectory a scenario reads: its ``file``, else its parent's annual pass output."""
    block = _seasonal_trajectory_block(scenario_id)
    return block.get("file") or f"{resources_path}/market/{_build_source(scenario_id)}_seasonal_trajectory.csv"


def _seasonal_trajectory_inputs(scenario_id):
    """``{"seasonal_trajectory": path}`` for a scenario that sets the key, else {}."""
    if not _uses_seasonal_trajectory(scenario_id):
        return {}
    return {"seasonal_trajectory": _seasonal_trajectory_file(scenario_id)}


_seasonal_pass_ids = sorted({_build_source(rid) for rid in _run_ids if _uses_seasonal_trajectory(rid)})
SEASONAL_PASS_REGEX = _regex_from_list(_seasonal_pass_ids)


def _dpa_dispatch_block(scenario_id):
    """``market.wholesale.marginal_cost_basis.dpa_dispatch`` of a scenario ({} when absent; task BM2050)."""
    return ((((_scenarios.get(scenario_id, {}).get("market", {}) or {}).get("wholesale", {}) or {})
             .get("marginal_cost_basis", {}) or {}).get("dpa_dispatch", {}) or {})


def _seasonal_pass_dpa(build_id):
    """The DPA dispatch block the scenarios on this build's trajectory declare, or None (task BM2050).

    Read from every scenario on the build that uses the trajectory (not only the run
    ids), so the pass is configured the same whatever is selected; they must agree.
    """
    blocks = [_dpa_dispatch_block(sid) for sid in _scenarios
              if _build_source(sid) == build_id and _uses_seasonal_trajectory(sid)
              and _dpa_dispatch_block(sid).get("enabled", False)]
    if not blocks:
        return None
    if any(b != blocks[0] for b in blocks[1:]):
        raise ValueError(f"{build_id}: the scenarios on its seasonal trajectory declare different dpa_dispatch blocks")
    return blocks[0]


def _seasonal_pass_config(build_id):
    """The pass's config: the build's own, with the months' DPA dispatch costs when they declare them.

    The annual pass must see the Stage 1 costs the months dispatch on (decision 33);
    a build none of whose scenarios declares the block gets its own entry unchanged.
    """
    dpa = _seasonal_pass_dpa(build_id)
    base = _scenarios.get(build_id, {})
    if dpa is None:
        return base
    import copy

    cfg = copy.deepcopy(base)
    basis = (cfg.setdefault("market", {}).setdefault("wholesale", {}).setdefault("marginal_cost_basis", {}))
    basis["dpa_dispatch"] = copy.deepcopy(dpa)
    return cfg


def _seasonal_pass_extra_inputs(wildcards):
    """The DPA dispatch table the DPA costs read, when the pass applies them (task BM2050), and the
    foreign price shape when the build itself sets market.foreign_price_override (2050 price calibration,
    25 Oct 2026: the central pass re-prices as its months do)."""
    dpa = _seasonal_pass_dpa(wildcards.scenario)
    inputs = {"dpa_dispatch": dpa["file"]} if dpa else {}
    inputs.update(_foreign_price_override_inputs(wildcards.scenario))
    return inputs


rule solve_seasonal_storage_pass:
    """
    Decision 33, pass 1: one copperplate LP over the parent's whole year on a fleet
    aggregated by carrier and cost class (scripts/market/seasonal_storage.py).

    Input:
      - Full-year parent network: resources/network/{scenario}.nc

    Output:
      - Hydrogen store trajectory (hourly level and store-balance dual) and each
        storage-unit carrier's state and storage dual (su_soc_end_mwh|c, su_value|c):
        resources/market/{scenario}_seasonal_trajectory.csv
      - Solver status, LP size, aggregation report and annual diagnostics:
        resources/market/{scenario}_seasonal_pass.json
    """
    input:
        unpack(_seasonal_pass_extra_inputs),
        network=lambda wildcards: _finalized_network(wildcards.scenario),
    output:
        trajectory=f"{resources_path}/market/{{scenario}}_seasonal_trajectory.csv",
        summary=f"{resources_path}/market/{{scenario}}_seasonal_pass.json",
    log:
        "logs/market/solve_seasonal_storage_pass_{scenario}.log"
    benchmark:
        "benchmarks/market/solve_seasonal_storage_pass_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    # The pass sets Gurobi threads itself (market.seasonal_pass.threads, default 4).
    threads: 4
    resources:
        # Measured on FES25_HT_2050 (8,760 hourly snapshots, aggregated): see the Y8a report.
        mem_mb=12000
    wildcard_constraints:
        scenario=SEASONAL_PASS_REGEX
    params:
        scenario_config=lambda wildcards: model_scenario_param(_seasonal_pass_config(wildcards.scenario)),
        solver=lambda wildcards: _get_solver_config_market(wildcards.scenario)[0],
        solver_options=lambda wildcards: _get_solver_config_market(wildcards.scenario)[1],
    script:
        "../scripts/market/solve_seasonal_storage_pass.py"


# Decision 29 (FES 2050 plan): a foreign-price sensitivity year needs its own annual
# pass. A scenario that sets market.foreign_price_override and reads a trajectory named
# resources/market/<build>_<suffix>_seasonal_trajectory.csv gets that file from this
# twin of solve_seasonal_storage_pass: the build's pass config (reserve, DPA costs)
# plus the override, on the build's network. Only scenarios that set both keys define
# a pass id here, so no other scenario's DAG changes. A scenario that keeps the build's
# own trajectory (the decision-29 gate weeks) stays on the central pass, declared.

def _variant_pass_specs():
    specs = {}
    for sid in _scenarios:
        override = ((_scenarios[sid].get("market", {}) or {}).get("foreign_price_override", {}) or {})
        if not (override.get("enabled", False) and _uses_seasonal_trajectory(sid)):
            continue
        build = _build_source(sid)
        path = _seasonal_trajectory_file(sid)
        if path == f"{resources_path}/market/{build}_seasonal_trajectory.csv":
            continue   # declared: on the build's own (central-price) pass, e.g. the decision-29 gate weeks
        prefix, suffix = f"{resources_path}/market/{build}_", "_seasonal_trajectory.csv"
        variant_name = path.startswith(prefix) and path.endswith(suffix) and len(path) > len(prefix) + len(suffix)
        build_override = ((_scenarios.get(build, {}).get("market", {}) or {}).get("foreign_price_override", {}) or {})
        if override == build_override and not variant_name:
            # 2050 price calibration (25 Oct 2026): the build itself sets the override, so every pass of the build
            # (its own and the decision-33 reserve variants) carries it; no price-variant pass is needed.
            continue
        if not variant_name:
            raise ValueError(
                f"{sid}: a foreign_price_override scenario must read its own pass's trajectory, "
                f"{prefix}<suffix>{suffix}, not {path}")
        pass_id = path[len(f"{resources_path}/market/"):-len(suffix)]
        spec = (build, override)
        if specs.setdefault(pass_id, spec) != spec:
            raise ValueError(f"{pass_id}: scenarios on this pass declare different overrides or builds")
    return specs


_VARIANT_PASS_SPECS = _variant_pass_specs()
VARIANT_PASS_REGEX = _regex_from_list(sorted(_VARIANT_PASS_SPECS))


def _variant_pass_config(pass_id):
    import copy

    build, override = _VARIANT_PASS_SPECS[pass_id]
    cfg = copy.deepcopy(_seasonal_pass_config(build))
    cfg.setdefault("market", {})["foreign_price_override"] = copy.deepcopy(override)
    return cfg


def _variant_pass_inputs(wildcards):
    from scripts.interconnectors.foreign_price_override import market_demand_path, shape_path

    build = _VARIANT_PASS_SPECS[wildcards.pass_id][0]
    cfg = _variant_pass_config(wildcards.pass_id)
    inputs = {"network": _finalized_network(build), "foreign_price_shape": shape_path(cfg, resources_path)}
    if market_demand_path(cfg):
        inputs["foreign_price_market_demand"] = market_demand_path(cfg)
    dpa = _seasonal_pass_dpa(build)
    if dpa:
        inputs["dpa_dispatch"] = dpa["file"]
    return inputs


rule solve_seasonal_storage_pass_price_variant:
    """Decision 29: the annual pass of a foreign-price sensitivity year (see above)."""
    input:
        unpack(_variant_pass_inputs),
    output:
        trajectory=f"{resources_path}/market/{{pass_id}}_seasonal_trajectory.csv",
        summary=f"{resources_path}/market/{{pass_id}}_seasonal_pass.json",
    log:
        "logs/market/solve_seasonal_storage_pass_{pass_id}.log"
    benchmark:
        "benchmarks/market/solve_seasonal_storage_pass_{pass_id}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    threads: 4
    resources:
        mem_mb=12000
    wildcard_constraints:
        pass_id=VARIANT_PASS_REGEX
    params:
        scenario_config=lambda wildcards: model_scenario_param(_variant_pass_config(wildcards.pass_id)),
        solver=lambda wildcards: _get_solver_config_market(_VARIANT_PASS_SPECS[wildcards.pass_id][0])[0],
        solver_options=lambda wildcards: _get_solver_config_market(_VARIANT_PASS_SPECS[wildcards.pass_id][0])[1],
    script:
        "../scripts/market/solve_seasonal_storage_pass.py"

#
# TASK BM2050: SUPPORT LABELS FOR A LONG-RANGE BALANCING STAGE (year-generic stage 5)
#
# A long-range scenario (2040, 2050, ...) that sets
# support.bm_bids.contract_value_bids.support_labels
# bids its CfD, DPA, RO and merchant plant from a label table built from its parent
# network by the vintaged register's own labelling code
# (scripts/market/support_labels.py), for the parent's modelled_year. The regex
# matches only the parents of run ids that set the key, so no 2030 scenario's DAG
# changes. Stage 1 and the seasonal pass read the DPA dispatch table instead
# (rule build_dpa_dispatch), which holds no strike.
#

def _support_labels_block(scenario_id):
    """``...contract_value_bids.support_labels`` of a scenario ({} when absent)."""
    return (
        ((_scenarios.get(scenario_id, {}).get("support", {}) or {})
         .get("bm_bids", {}) or {}).get("contract_value_bids", {}) or {}
    ).get("support_labels", {}) or {}


def _uses_support_labels(scenario_id):
    return bool(_support_labels_block(scenario_id)) and _uses_contract_value_bids(scenario_id)


def _support_labels_file(scenario_id):
    """The label table a scenario reads: its ``file``, else its parent build's."""
    return (_support_labels_block(scenario_id).get("file")
            or f"{resources_path}/market/{_build_source(scenario_id)}_support_labels.csv")


_support_label_build_ids = sorted({_build_source(rid) for rid in _run_ids if _uses_support_labels(rid)})
SUPPORT_LABEL_BUILD_REGEX = _regex_from_list(_support_label_build_ids)


def _support_labels_settings(build_id, scenario_ids=None):
    """The label block every run id on this build declares (they must agree).

    ``scenario_ids``: the ids to read (default the selected run ids; the DPA dispatch rule
    reads every scenario on the build, as the seasonal pass does).
    """
    ids = _run_ids if scenario_ids is None else scenario_ids
    blocks = [_support_labels_block(rid) for rid in ids
              if _build_source(rid) == build_id and _uses_support_labels(rid)]
    settings = [{k: v for k, v in b.items() if k != "dpa_carriers"} for b in blocks]
    if not settings:
        raise ValueError(f"{build_id}: no run id declares contract_value_bids.support_labels")
    if any(s != settings[0] for s in settings[1:]):
        raise ValueError(f"{build_id}: run ids on this build declare different support_labels settings")
    return settings[0]


def _support_labels_workbook(build_id):
    """The FES Data Workbook of the build's own edition (``targets.edition``): the stock-flow trajectory's source."""
    from scripts.sources.fes_adapter import WORKBOOKS

    targets = _scenarios.get(build_id, {}).get("targets") or {}
    edition = targets.get("edition")
    if targets.get("source") != "FES" or edition is None or int(edition) not in WORKBOOKS:
        raise ValueError(f"{build_id}: support labels need a FES build with a registered Data Workbook "
                         f"(targets {targets}; scripts/sources/fes_adapter.WORKBOOKS)")
    path = WORKBOOKS[int(edition)]
    return str(path.relative_to(path.parents[2])).replace("\\", "/")


def _support_labels_model_year(build_id):
    """The parent build's modelled_year: the year the labels' contracts, stock-flow and strikes are for."""
    year = _scenarios.get(build_id, {}).get("modelled_year")
    if year is None:
        raise ValueError(f"{build_id}: support labels need the build's modelled_year")
    return int(year)


def _support_labels_fleet(build_id, key, default):
    return (_support_labels_settings(build_id).get("fleet_vintages") or {}).get(key, default)


def _life_override_flag(overrides):
    """FES 2050 decision 48: ``fleet_vintages.life_overrides`` as ``--life-override "TECH=YEARS" ...`` ('' if none)."""
    items = [f'"{tech}={float(life):g}"' for tech, life in sorted((overrides or {}).items())]
    return f" --life-override {' '.join(items)}" if items else ""


def _fleet_decision62_flags(fleet, *, stock_flow=True, labels=True):
    """FES 2050 decision 62: a fleet_vintages block's repowering rule, evidence exclusions and no-CfD list as flags.

    '' for a block without them (every group before decision 62, and every 2030 group).
    ``stock_flow``: the flags the stock-flow takes (repowering, exclusions, and F28 (d)'s
    refurbishment rule, as ``_po_refurbishment_flag`` gives the register group's); ``labels``: the
    flag the labelling takes (the technologies whose repowered capacity holds no CfD).
    """
    from scripts.public_ownership.fleet_vintages import (parse_refurbishment, parse_repowering, refurbishment_flag,
                                                         repowering_flag)

    fleet = fleet or {}
    out = repowering_flag(parse_repowering(fleet.get("repowering")))
    if stock_flow:
        countries = [f'"{c}"' for c in (fleet.get("evidence_exclude_countries") or [])]
        if countries:
            out += f" --exclude-country {' '.join(countries)}"
        if fleet.get("evidence_stale_records"):
            out += f" --stale-records {fleet['evidence_stale_records']}"
        out += refurbishment_flag(parse_refurbishment(fleet.get("refurbishment")))
    if labels:
        techs = [f'"{t}"' for t in (fleet.get("no_cfd_on_repowering") or [])]
        if techs:
            out += f" --no-cfd-on-repowering {' '.join(techs)}"
    return out


def _support_labels_small_scale_lines(group):
    """F28: the ES1 lines file a register group's small-scale exclusion names ([] without one)."""
    lines = (_po_cfg(group).get("small_scale_exclusion") or {}).get("es1_lines") if group else None
    return [lines] if lines else []


def _fleet_stale_records_input(fleet):
    """Decision 62: the stale-REPD-records file a fleet_vintages block names ([] without one)."""
    path = (fleet or {}).get("evidence_stale_records")
    return [path] if path else []


def _support_labels_calibrated(build_id):
    return _support_labels_settings(build_id).get("strike_variant", "central") == "calibrated"


def _support_labels_calibration_inputs(build_id):
    """Decision 49 (b): the calibrated strikes read the study's decision-41 files. Ancient: the BM
    bids on one pass of the calibration, so re-deriving it after the solve rebuilds no labels."""
    if not _support_labels_calibrated(build_id):
        return []
    return [ancient(p) for p in _po_vintage_capital_files(_PO_STUDIES.configs[build_id].get("public_ownership"))]


def _support_labels_group(build_id):
    """The vintaged run group whose register labels these must reproduce (its flags and award years)."""
    group = _support_labels_settings(build_id).get("register_group")
    if not group:
        raise ValueError(f"{build_id}: support_labels needs register_group (the vintaged run group)")
    return group


def _support_labels_future_costs(build_id):
    """The scenario's fuel and carbon price file (marginal_costs.future_costs), for the DPA reference CCGT."""
    future = (_scenarios.get(build_id, {}).get("marginal_costs", {}) or {}).get("future_costs", {}) or {}
    case = (future.get("cases", {}) or {}).get(future.get("case"), {}) or {}
    path = case.get("source_file")
    if not path:
        raise ValueError(f"{build_id}: support labels need marginal_costs.future_costs.cases.<case>.source_file")
    return path


def _solve_boundary(files):
    """Inputs of a rule a solve reads, as ancient() (15 Nov 2026): an edit never re-solves by itself."""
    return [ancient(f) for f in files]


rule build_support_labels:
    """Task BM2050: the contract each build-year cohort of a long-range network holds (CfD, DPA, RO, merchant, RAB).

    The vintaged register's own labelling (stock-flow join and
    fleet_vintages.label_support_contracts), run on the unsolved parent network for
    its modelled_year, with the strike each CfD settles at (cfd_strikes), the RO
    banding of a cohort the RO still supports (support_schemes.csv) and the
    reference CCGT a DPA's variable payment is set against (support_regime_2050).
    Read by the BM of every run id that sets contract_value_bids.support_labels.
    """
    input:
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        workbook=lambda wildcards: _support_labels_workbook(wildcards.scenario),
        repd="data/renewables/repd-q2-jul-2025.csv",
        dukes=ancient(f"{resources_path}/generators/DUKES/DUKES_2023_generators.csv"),
        # Solve boundary (user decision, 15 Nov 2026): the three cost tables the bill layer regenerates are ancient()
        # here; their regeneration (OCGT/gas-engine rows) left every label field the solves read unchanged (measured).
        lives=ancient("data/public_ownership/financing/technology_economic_lives.csv"),
        closures="data/generators/nuclear_phase_out_dates.csv",
        crosswalk=ancient("data/public_ownership/financing/pypsa_finance_technology_crosswalk.csv"),
        crosswalk_2050=ancient("data/public_ownership/financing/pypsa_finance_technology_crosswalk_2050.csv"),
        award_years=lambda wildcards: _po_group_award_years(_support_labels_group(wildcards.scenario)),
        known_contracts=ancient("data/market/support/cfd_contracts.csv"),
        support_schemes="data/market/support/support_schemes.csv",
        strike_inputs="data/market/support/cfd_strike_inputs_2050.csv",
        strike_cost_calibration="data/market/support/cfd_strike_cost_calibration.csv",
        support_regime="data/market/support/support_regime_2050.json",
        vintage_costs=ancient("data/public_ownership/financing/technology_costs_vintage.csv"),
        hurdle_rates=ancient("data/public_ownership/financing/fes2025_hurdle_rates.csv"),
        published_strikes="data/market/lccc/cfd_strike_prices_by_round.csv",
        # The workbook rate snapshot the strikes price on; rewritten by every 2030
        # bill-layer run without changing a rate, so read as ancient.
        snapshot=ancient(f"{resources_path}/public_ownership/workbook_v5_assumptions.yaml"),
        future_costs=lambda wildcards: _support_labels_future_costs(wildcards.scenario),
        # Decision 49 (b): strike_variant calibrated (the bill's capture prices and calibration).
        calibration=lambda wildcards: _support_labels_calibration_inputs(wildcards.scenario),
        # Decision 62: the REPD records left out of the stock-flow's evidence.
        stale_records=lambda wildcards: _fleet_stale_records_input(_support_labels_settings(wildcards.scenario).get("fleet_vintages")),
        # F28: the register group's ES1 lines (its small-scale exclusion), as _po_register_flags names them.
        small_scale_lines=lambda wildcards: _support_labels_small_scale_lines(_support_labels_group(wildcards.scenario)),
        # The labelling code is the vintaged register's (task Y10/SETTLE2050) and Y11's, so an
        # edit there rebuilds the labels and the BM solves that bid on them.
        # Solve boundary (15 Nov 2026): the code, the crosswalks and the contract table are ancient() here, so an edit
        # never re-solves by itself; the results audit's code and content checks stop the run for the user instead.
        code=_solve_boundary(["scripts/market/support_labels.py", "scripts/public_ownership/fleet_vintages.py",
              "scripts/public_ownership/build_asset_finance_register.py", "scripts/market/cfd_strikes.py",
              "scripts/public_ownership/support_regime_2050.py", "scripts/market/capacity_market_2050.py"]),
    output:
        labels=f"{resources_path}/market/{{scenario}}_support_labels.csv",
        summary=f"{resources_path}/market/{{scenario}}_support_labels.json",
    log:
        "logs/market/build_support_labels_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    resources:
        mem_mb=6000
    wildcard_constraints:
        scenario=SUPPORT_LABEL_BUILD_REGEX
    params:
        pathway=lambda wildcards: _support_labels_fleet(wildcards.scenario, "pathway", "Holistic Transition"),
        rule=lambda wildcards: _support_labels_fleet(wildcards.scenario, "retirement_rule", "economic_life_with_extension"),
        life_source=lambda wildcards: _support_labels_fleet(wildcards.scenario, "life_source", "central"),
        join_mode=lambda wildcards: _support_labels_fleet(wildcards.scenario, "join_mode", "pro_rata"),
        flags=lambda wildcards: _po_register_flags(storage_sites=False, group=_support_labels_group(wildcards.scenario)),
        strike_variant=lambda wildcards: _support_labels_settings(wildcards.scenario).get("strike_variant", "central"),
        # Year-generic stage 5: the parent build's modelled_year.
        model_year=lambda wildcards: _support_labels_model_year(wildcards.scenario),
        # Decision 48 (life overrides); the study the strikes' insurance and the calibrated
        # strikes (decision 49 (b)) resolve (a non-finance scenario falls back to config/defaults/).
        extra=lambda wildcards: (_life_override_flag(_support_labels_fleet(wildcards.scenario, "life_overrides", {}))
                                 + f" --study {wildcards.scenario}"
                                 # Decision 62: the repowering rule, exclusions and no-CfD list ('' before it).
                                 + _fleet_decision62_flags(_support_labels_settings(wildcards.scenario).get("fleet_vintages"))),
    shell:
        "python scripts/market/support_labels.py {input.network} {output.labels} --summary {output.summary}"
        " --model-year {params.model_year} --snapshot {input.snapshot}"
        " --workbook \"{input.workbook}\" --pathway \"{params.pathway}\" --rule {params.rule}"
        " --life-source {params.life_source} --join-mode {params.join_mode}"
        " --repd {input.repd} --dukes {input.dukes} --lives {input.lives} --closures {input.closures}"
        " --crosswalk {input.crosswalk} --crosswalk-2050 {input.crosswalk_2050} {params.flags}"
        " --future-costs {input.future_costs} --strike-variant {params.strike_variant}{params.extra} > {log} 2>&1"


# Year-generic stage 5: the rate-free DPA dispatch table Stage 1 and the seasonal pass read
# (market.wholesale.marginal_cost_basis.dpa_dispatch.file). It holds each component's live
# DPA weight, reference CCGT cost and margin, and no strike: a financing-rate change
# rebuilds the label table and the balancing months, never this table or Stage 1.
def _dpa_dispatch_settings(build_id):
    """The support-label settings of the build's scenarios (every scenario on it, not only the run ids)."""
    return _support_labels_settings(build_id, scenario_ids=list(_scenarios))


def _dpa_dispatch_group(build_id):
    """The vintaged run group whose register labels the DPA table must reproduce (as _support_labels_group)."""
    group = _dpa_dispatch_settings(build_id).get("register_group")
    if not group:
        raise ValueError(f"{build_id}: dpa_dispatch needs a support_labels block with register_group on its build")
    return group


_dpa_dispatch_build_ids = sorted({_build_source(sid) for sid in _scenarios
                                  if _dpa_dispatch_block(sid).get("enabled", False)})
DPA_DISPATCH_BUILD_REGEX = _regex_from_list(_dpa_dispatch_build_ids)


rule build_dpa_dispatch:
    """Task BM2050 / stage 5: the DPA share, reference CCGT cost and margin of each long-range component.

    The same cohorts as build_support_labels (the vintaged register's labelling on the
    unsolved parent network, for its modelled_year), without the strikes, the snapshot or
    the calibration, so Stage 1 and the seasonal pass do not depend on a financing rate.
    """
    input:
        network=lambda wildcards: _finalized_network(wildcards.scenario),
        workbook=lambda wildcards: _support_labels_workbook(wildcards.scenario),
        repd="data/renewables/repd-q2-jul-2025.csv",
        dukes=ancient(f"{resources_path}/generators/DUKES/DUKES_2023_generators.csv"),
        lives=ancient("data/public_ownership/financing/technology_economic_lives.csv"),  # solve boundary (user decision, 15 Nov 2026; measured identical)
        closures="data/generators/nuclear_phase_out_dates.csv",
        crosswalk=ancient("data/public_ownership/financing/pypsa_finance_technology_crosswalk.csv"),
        crosswalk_2050=ancient("data/public_ownership/financing/pypsa_finance_technology_crosswalk_2050.csv"),
        award_years=lambda wildcards: _po_group_award_years(_dpa_dispatch_group(wildcards.scenario)),
        known_contracts=ancient("data/market/support/cfd_contracts.csv"),
        support_schemes="data/market/support/support_schemes.csv",
        # Contract terms only (modelled_contract_term); no rate is read from it.
        strike_inputs="data/market/support/cfd_strike_inputs_2050.csv",
        support_regime="data/market/support/support_regime_2050.json",
        future_costs=lambda wildcards: _support_labels_future_costs(wildcards.scenario),
        stale_records=lambda wildcards: _fleet_stale_records_input(_dpa_dispatch_settings(wildcards.scenario).get("fleet_vintages")),
        small_scale_lines=lambda wildcards: _support_labels_small_scale_lines(_dpa_dispatch_group(wildcards.scenario)),
        # Solve boundary (15 Nov 2026): the code, the crosswalks and the contract table are ancient() here, so an edit
        # never re-solves by itself; the results audit's code and content checks stop the run for the user instead.
        code=_solve_boundary(["scripts/market/support_labels.py", "scripts/public_ownership/fleet_vintages.py",
              "scripts/public_ownership/build_asset_finance_register.py",
              "scripts/public_ownership/support_regime_2050.py", "scripts/market/capacity_market_2050.py"]),
    output:
        table=f"{resources_path}/market/{{scenario}}_dpa_dispatch.csv",
        summary=f"{resources_path}/market/{{scenario}}_dpa_dispatch.json",
    log:
        "logs/market/build_dpa_dispatch_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    resources:
        mem_mb=6000
    wildcard_constraints:
        scenario=DPA_DISPATCH_BUILD_REGEX
    params:
        pathway=lambda wildcards: (_dpa_dispatch_settings(wildcards.scenario).get("fleet_vintages") or {}).get("pathway", "Holistic Transition"),
        rule=lambda wildcards: (_dpa_dispatch_settings(wildcards.scenario).get("fleet_vintages") or {}).get("retirement_rule", "economic_life_with_extension"),
        life_source=lambda wildcards: (_dpa_dispatch_settings(wildcards.scenario).get("fleet_vintages") or {}).get("life_source", "central"),
        join_mode=lambda wildcards: (_dpa_dispatch_settings(wildcards.scenario).get("fleet_vintages") or {}).get("join_mode", "pro_rata"),
        flags=lambda wildcards: _po_register_flags(storage_sites=False, group=_dpa_dispatch_group(wildcards.scenario)),
        model_year=lambda wildcards: _support_labels_model_year(wildcards.scenario),
        extra=lambda wildcards: (_life_override_flag((_dpa_dispatch_settings(wildcards.scenario).get("fleet_vintages") or {}).get("life_overrides", {}))
                                 + _fleet_decision62_flags(_dpa_dispatch_settings(wildcards.scenario).get("fleet_vintages"))),
    shell:
        "python scripts/market/support_labels.py {input.network} {output.table} --summary {output.summary}"
        " --kind dpa_dispatch --model-year {params.model_year}"
        " --workbook \"{input.workbook}\" --pathway \"{params.pathway}\" --rule {params.rule}"
        " --life-source {params.life_source} --join-mode {params.join_mode}"
        " --repd {input.repd} --dukes {input.dukes} --lives {input.lives} --closures {input.closures}"
        " --crosswalk {input.crosswalk} --crosswalk-2050 {input.crosswalk_2050} {params.flags}"
        " --future-costs {input.future_costs}{params.extra} > {log} 2>&1"
