# =============================================================================
# Pro rata zonal turn-down re-pricing (user decision, 20 Oct 2026)
# =============================================================================
# scripts/market/turn_down_allocation.py. Report and bill input; never read by
# the solves. Each month of a monthly_aggregates family is re-priced on
# BOTH bases: as solved (cheapest_first) and with every NESO CP.08 zone's hourly
# wind turn-down re-shared pro rata to Stage 1 output (pro_rata_zonal), each unit
# at its own bid. The family summary names the headline basis per
# market.balancing.turn_down_allocation.mode (cheapest_first | pro_rata_zonal |
# closest_to_neso: the basis nearer NESO's cost for the family's network).
# The bill and ownership layers read the basis of their network's market family
# (turn_down_allocation.use_network_basis). Needs rules/CP30.smk and
# rules/market.smk (_wholesale_source) loaded first.

_TURN_DOWN_MONTHS = sorted({s for study in CP30_MONTHLY_AGGREGATES
                            for s in (CP30_MONTHLY_AGGREGATES[study].get("scenarios") or [])})
TURN_DOWN_MONTH_REGEX = "|".join(map(re.escape, _TURN_DOWN_MONTHS)) if _TURN_DOWN_MONTHS else r"(?!x)x"


def _turn_down_month_settings(wildcards):
    """The few month settings the re-pricing reads (technologies, zones, ROC rate); not the mode."""
    from scripts.market.turn_down_allocation import scenario_settings

    return scenario_settings(scenarios[wildcards.scenario])


def _turn_down_family_settings(wildcards):
    """The family's mode and network (its first month's config)."""
    from scripts.market.turn_down_allocation import family_settings

    return family_settings(scenarios[_cp30_monthly_aggregate_config(wildcards.study)["scenarios"][0]])


def _turn_down_zone_files(wildcards):
    from scripts.market.neso_zones import ZONES_FILE
    from scripts.solve.secured_boundaries import DEFAULT_GEOMETRY

    s = _turn_down_month_settings(wildcards)
    return [s.get("zones_file") or ZONES_FILE,
            (s.get("secured_capability") or {}).get("geometry_file") or DEFAULT_GEOMETRY]


def turn_down_summary_path(study):
    """The family summary (both bases, headline_basis) the bill reads its network's basis from."""
    return f"{resources_path}/analysis/{study}_pro_rata_turn_down_summary.json"


def _turn_down_network_study(scenario):
    """The market family whose headline basis this month's bill reads (its network's), or None."""
    from scripts.market.turn_down_allocation import network_basis_study

    cases = globals().get("_consumer_all_future_cases") or {}
    return network_basis_study(scenario, cases) if scenario in _TURN_DOWN_MONTHS else None


def turn_down_month_inputs(scenario_ids):
    """The month files a bill or ownership rule reads BM turn-down through (turn_down_allocation helpers):
    each covered month's basis file and pro rata table and totals. Empty for months the re-pricing does
    not cover (2050, historical), which read as solved."""
    out = []
    for s in scenario_ids:
        if _turn_down_network_study(s) is None:
            continue
        out += [f"{resources_path}/market/{s}_{suffix}"
                for suffix in ("turn_down_basis.json", "pro_rata_turn_down.csv", "pro_rata_turn_down.json")]
    return out


rule reprice_turn_down_pro_rata:
    """
    One BM month on both turn-down bases: per unit (zone, support, state share,
    as-solved and pro rata MWh and cost) and totals. Fails if the rebuilt bids do
    not reproduce the solve's wind bid cost (0.1% of gross).
    """
    input:
        network=f"{resources_path}/market/{{scenario}}_balancing.nc",
        redispatch=f"{resources_path}/market/{{scenario}}_redispatch_summary.csv",
        contract_value_bids=f"{resources_path}/market/{{scenario}}_contract_value_bids.csv",
        stage1_network=lambda wc: f"{resources_path}/market/{_wholesale_source(wc.scenario)}_wholesale.nc",
        stage1_price=lambda wc: f"{resources_path}/market/{_wholesale_source(wc.scenario)}_wholesale_price.csv",
        # 13 Nov 2026: a long-range month bids from its build's support label table; the re-pricing rebuilds
        # the hourly CfD part from it (empty for the 2030 months, which carry their strikes in the audit).
        support_labels=lambda wc: [_support_labels_file(wc.scenario)] if _uses_support_labels(wc.scenario) else [],
        references=_turn_down_zone_files,
        logic=["scripts/market/turn_down_allocation.py", "scripts/market/neso_zones.py"],
    output:
        csv=f"{resources_path}/market/{{scenario}}_pro_rata_turn_down.csv",
        json=f"{resources_path}/market/{{scenario}}_pro_rata_turn_down.json",
    params:
        settings=_turn_down_month_settings,
    log:
        "logs/market/reprice_turn_down_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=TURN_DOWN_MONTH_REGEX
    script:
        "../scripts/market/turn_down_allocation.py"


rule summarise_turn_down_allocation:
    """
    A family's twelve months on both bases: wind bid cost and the NESO-comparable
    cost (aggregate.neso_comparable_redispatch_cost_GBP + pro rata uplift), NESO's
    figure for the family's network, and the headline basis.
    """
    input:
        months=lambda wc: [f"{resources_path}/market/{s}_pro_rata_turn_down.json"
                           for s in _cp30_monthly_aggregate_config(wc.study)["scenarios"]],
        annual_summary=f"{resources_path}/analysis/{{study}}_annual_summary.json",
        benchmarks="data/CP30/validation_references/neso_balancing_benchmarks.csv",
        logic="scripts/market/turn_down_allocation.py",
    output:
        json=f"{resources_path}/analysis/{{study}}_pro_rata_turn_down_summary.json",
    params:
        settings=_turn_down_family_settings,
    log:
        "logs/market/summarise_turn_down_{study}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        study=CP30_MONTHLY_AGGREGATE_REGEX
    script:
        "../scripts/market/turn_down_allocation.py"


rule select_turn_down_basis:
    """
    A month's turn-down basis for the bill and ownership layers: the headline
    basis of its network's market family (the base bill case of its network
    group; P2/P3 bid families and price-level twins follow their network).
    """
    input:
        summary=lambda wc: turn_down_summary_path(_turn_down_network_study(wc.scenario)),
        month=f"{resources_path}/market/{{scenario}}_pro_rata_turn_down.json",
        logic="scripts/market/turn_down_allocation.py",
    output:
        json=f"{resources_path}/market/{{scenario}}_turn_down_basis.json",
    params:
        settings={},
    log:
        "logs/market/select_turn_down_basis_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario=TURN_DOWN_MONTH_REGEX
    script:
        "../scripts/market/turn_down_allocation.py"
