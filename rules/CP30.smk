"""
Clean Power 2030 data extraction rules.

Stage 1 materializes CP30 workbook sheets into stable CSV resources. Later
workflow stages decide how those resources are routed into the network model.
"""

import re


# Complete-year aggregates: each loaded study's study.yaml monthly_aggregates (config_loader.study_settings).
CP30_MONTHLY_AGGREGATES = _full_config["study_settings"]["monthly_aggregates"]
CP30_MONTHLY_AGGREGATE_REGEX = (
    "|".join(map(re.escape, CP30_MONTHLY_AGGREGATES))
    if CP30_MONTHLY_AGGREGATES
    else r"(?!x)x"
)


def _cp30_monthly_aggregate_config(study):
    """Return a validated, explicitly configured twelve-month study."""
    aggregate = CP30_MONTHLY_AGGREGATES[study]
    scenarios_for_study = aggregate.get("scenarios", [])
    if len(scenarios_for_study) != 12 or len(set(scenarios_for_study)) != 12:
        raise ValueError(
            f"CP30 monthly aggregate {study!r} must declare twelve unique scenario IDs"
        )
    return aggregate


def _cp30_monthly_aggregate_inputs(wildcards):
    aggregate = _cp30_monthly_aggregate_config(wildcards.study)
    return expand(
        f"{resources_path}/analysis/{{scenario}}_market_summary.json",
        scenario=aggregate["scenarios"],
    )


rule CP30_data:
    """
    Extract normalized CP30 data tables from the Clean Power 2030 workbook.
    """
    input:
        workbook=f"{data_path}/CP30/Clean Power 2030 Data Workbook_V003.xlsx"
    output:
        es1=f"{resources_path}/CP30/CP30_ES1.csv",
        es5=f"{resources_path}/CP30/CP30_ES5_regional.csv",
        ed1=f"{resources_path}/CP30/CP30_ED1.csv",
        reference=f"{resources_path}/CP30/CP30_reference_N1_N2_NA1.csv",
        validation=f"{resources_path}/CP30/CP30_validation_report.csv",
        interconnector_prices=f"{resources_path}/CP30/CP30_interconnector_prices.csv"
    log:
        "logs/CP30/CP30_data.log"
    benchmark:
        "benchmarks/CP30/CP30_data.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/CP30/CP30_data.py"

rule validate_CP30_network:
    """
    Validate a built CP30 network against authoritative CP30 pathway totals.

    This rule is intended for explicit validation runs once the full network has
    been built. It records CP30 network-reference metadata but does not translate
    CP30 N1/N2 projects into branch parameters.
    """
    input:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc",
        cp30_es1=lambda wc: scenarios[wc.scenario].get("source_data", {}).get("files", {}).get("es1", []),
        cp30_ed1=lambda wc: scenarios[wc.scenario].get("source_data", {}).get("files", {}).get("ed1", []),
        cp30_reference=lambda wc: scenarios[wc.scenario].get("source_data", {}).get("files", {}).get("reference", []),
        committed_overlay=lambda wc: scenarios[wc.scenario].get("fleet", {}).get("spatial_proxy", {}).get("committed_overlay") or [],
        energy_validation_logic="scripts/CP30/energy_validation.py"
    output:
        validation=f"{resources_path}/CP30/validation/{{scenario}}_component_totals.csv",
        reference_summary=f"{resources_path}/CP30/validation/{{scenario}}_network_reference_summary.csv",
        network_assumption=f"{resources_path}/CP30/validation/{{scenario}}_network_assumption.csv",
        # Available energy vs CP30 gross (ES1 generation + curtailment) and
        # demand vs CP30 (audit 26 Sep 2026; scripts/CP30/energy_validation.py).
        energy_validation=f"{resources_path}/CP30/validation/{{scenario}}_energy_totals.csv"
    params:
        scenario_config=lambda wc: scenario_param(scenarios[wc.scenario]),
        cp30_pathway=lambda wc: scenarios[wc.scenario].get("CP30_pathway"),
        modelled_year=lambda wc: scenarios[wc.scenario].get("modelled_year")
    log:
        "logs/CP30/validate_network_{scenario}.log"
    benchmark:
        "benchmarks/CP30/validate_network_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/CP30/validate_cp30_network.py"


def _cp30_locational_config(wildcards):
    return (scenarios[wildcards.scenario].get("validation") or {}).get("locational") or {}


def _cp30_locational_references(wildcards):
    """The committed reference tables (hand judgements, each row with its source)."""
    import glob as _glob

    folder = _cp30_locational_config(wildcards).get("reference_dir", "data/CP30/validation_references")
    return sorted(_glob.glob(f"{folder}/*.csv"))


def _cp30_locational_fes_reference(wildcards):
    return _cp30_locational_config(wildcards).get("fes_reference") or {}


def _cp30_locational_build_inputs(wildcards):
    """The build's own placement files the checks judge against (regions, landings, CfD contracts)."""
    sc = scenarios[wildcards.scenario]
    fleet = sc.get("fleet") or {}
    regional = (fleet.get("spatial_proxy") or {}).get("regional_allocation") or {}
    brownfield = {**(fleet.get("brownfield") or {}), **((fleet.get("spatial_proxy") or {}).get("brownfield") or {})}
    return {
        "gsp_polygons": regional.get("gsp_polygon_file", "data/network/GSP/GSP_regions_27700_20250109.geojson"),
        "gsp_group_regions": regional.get("gsp_group_region_file", "data/network/ETYS/cp30_es5_region_by_gsp_group.csv"),
        "offshore_crosswalk": brownfield.get("offshore_landing_crosswalk",
                                             "data/network/ETYS/repd_offshore_landing_crosswalk.csv"),
        "cfd_contracts": _support_register_input(wildcards.scenario, "checked_cfd_contracts") or [],
    }


rule validate_CP30_locational:
    """
    Locational and network validation of a built CP30 network (15 Oct 2026).

    Checks where capacity sits, not only how much: ES5 regional totals by bus region,
    offshore landings (named, refill, proxy boundary crossings, region tags), named
    stations at their TEC buses, electrolysers and LDES behind the Scottish boundaries,
    interconnector landings and total, demand shares, total and peak, distributed
    generation at its GSP node, radial pockets, and a boundary exposure summary.
    Writes one table; a FAIL row fails the rule after the table is written
    (validation.locational.fail_on_fail). scripts/CP30/locational_validation.py.

    References derived from repository sources are built at run time from the inputs
    declared here (FES data, offshore crosswalk, checked CfD contracts, the outputs of
    extract_cp30_demand_inputs with the demand build's own functions, the CP30 cable
    composition); only hand judgements are committed (references).
    """
    input:
        network=f"{resources_path}/network/{{scenario}}.nc",
        cp30_es5=lambda wc: scenarios[wc.scenario].get("source_data", {}).get("files", {}).get("es5", []),
        cp30_ed1=lambda wc: scenarios[wc.scenario].get("source_data", {}).get("files", {}).get("ed1", []),
        cp30_es1=lambda wc: scenarios[wc.scenario].get("source_data", {}).get("files", {}).get("es1", []),
        references=_cp30_locational_references,
        boundaries=lambda wc: _cp30_locational_config(wc).get(
            "boundary_file", "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson"),
        # The build's GSP polygons and region map, landing crosswalk and checked CfD contracts.
        gsp_polygons=lambda wc: _cp30_locational_build_inputs(wc)["gsp_polygons"],
        gsp_group_regions=lambda wc: _cp30_locational_build_inputs(wc)["gsp_group_regions"],
        # Sources of the references derived at run time (locational_validation.derive_references).
        fes_data=lambda wc: f"{resources_path}/FES/FES_{_cp30_locational_fes_reference(wc).get('FES_year', 2024)}_data.csv",
        # The GSP table of fes_data's own FES edition (stage 5: no FES 2024 default for another edition).
        gsp_info=lambda wc: _cp30_locational_fes_reference(wc).get(
            "gsp_info_file",
            f"data/network/ETYS/fes{_cp30_locational_fes_reference(wc).get('FES_year', 2024)}_regional_breakdown_gsp_info.csv"),
        offshore_crosswalk=lambda wc: _cp30_locational_build_inputs(wc)["offshore_crosswalk"],
        cfd_contracts=lambda wc: _cp30_locational_build_inputs(wc)["cfd_contracts"],
        # The named loads as the CP30 demand build places them (rule extract_cp30_demand_inputs's
        # outputs, demand.build.named_load_reference); empty when the scenario does not enable it.
        named_loads=lambda wc: cp30_demand_inputs(wc.scenario).get("cp30_named_loads", []),
        demand_lines=lambda wc: cp30_demand_inputs(wc.scenario).get("cp30_national_lines", []),
        demand_logic=lambda wc: cp30_demand_inputs(wc.scenario).get("cp30_logic", []),
        etys_peak=CP30_ETYS_PEAK_FILE,
        # CP30's cable composition (interconnectors.cp30_composition): capacities and which cables count.
        ic_composition=lambda wc: get_cp30_interconnector_composition(scenarios[wc.scenario]),
        # Dem_per_node, read by the build's GSP-node locator (check 8: the node a site belongs at).
        gsp_dem_per_node="data/network/ETYS/GB_network.xlsx",
        logic=["scripts/CP30/locational_validation.py", "scripts/CP30/spatial_allocation.py",
               "scripts/utilities/gsp_nodes.py"]
    output:
        checks=f"{resources_path}/CP30/validation/{{scenario}}_locational_checks.csv"
    params:
        scenario_config=lambda wc: scenario_param(scenarios[wc.scenario]),
        cp30_pathway=lambda wc: scenarios[wc.scenario].get("CP30_pathway"),
        modelled_year=lambda wc: scenarios[wc.scenario].get("modelled_year"),
        cp30_demand=lambda wc: get_cp30_demand_config(wc.scenario)
    log:
        "logs/CP30/validate_locational_{scenario}.log"
    benchmark:
        "benchmarks/CP30/validate_locational_{scenario}.txt"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/CP30/locational_validation.py"


rule aggregate_CP30_months:
    """Sum an explicitly declared complete CP30 calendar-year study."""
    input:
        summary_jsons=_cp30_monthly_aggregate_inputs
    output:
        monthly_csv=f"{resources_path}/analysis/{{study}}_monthly_metrics.csv",
        annual_json=f"{resources_path}/analysis/{{study}}_annual_summary.json"
    params:
        scenarios=lambda wc: _cp30_monthly_aggregate_config(wc.study)["scenarios"],
        scenario_family=lambda wc: _cp30_monthly_aggregate_config(wc.study).get(
            "scenario_family", wc.study
        )
    log:
        "logs/CP30/aggregate_{study}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        study=CP30_MONTHLY_AGGREGATE_REGEX
    script:
        "../scripts/market/aggregate_neso_comparator.py"


def _cp30_monthly_balancing_networks(wildcards):
    aggregate = _cp30_monthly_aggregate_config(wildcards.study)
    return expand(
        f"{resources_path}/market/{{scenario}}_balancing.nc",
        scenario=aggregate["scenarios"],
    )


def _cp30_monthly_wholesale_networks(wildcards):
    """Each month's Stage 1 network, from the scenario its Stage 1 comes from (follow-up F15)."""
    aggregate = _cp30_monthly_aggregate_config(wildcards.study)
    return [f"{resources_path}/market/{_wholesale_source(scenario)}_wholesale.nc"
            for scenario in aggregate["scenarios"]]


def _cp30_study_config(wildcards):
    return scenarios[_cp30_monthly_aggregate_config(wildcards.study)["scenarios"][0]]


rule validate_CP30_dispatch:
    """
    Validate a solved CP30 calendar year's FINAL (balancing) dispatch against
    CP30 ES1 energy: dispatched energy and curtailment by technology, demand,
    storage, interconnectors, electrolysis, and every unmapped ES1 line.
    Energy differences warn; an unmapped line or a demand mismatch fails
    after the table is written (audit 26 Sep 2026). Follow-up F15 (6 Oct 2026):
    ES1 is CP30's unconstrained dispatch, so its generation, curtailment, storage
    and interconnector lines are judged on Stage 1, the final values kept as INFO;
    CP.08's constraint bid volume is set against the balancing-stage turn-down.
    """
    input:
        networks=_cp30_monthly_balancing_networks,
        wholesale_networks=_cp30_monthly_wholesale_networks,
        cp30_cp08="data/CP30/cp30_cp08_constraint_bid_volumes_2030.csv",
        cp30_es1=lambda wc: _cp30_study_config(wc).get("source_data", {}).get("files", {}).get("es1", []),
        energy_validation_logic="scripts/CP30/energy_validation.py",
    output:
        csv=f"{resources_path}/analysis/{{study}}_cp30_energy_validation.csv",
        json=f"{resources_path}/analysis/{{study}}_cp30_energy_validation.json"
    params:
        scenario_config=_cp30_study_config
    log:
        "logs/CP30/validate_dispatch_{study}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        study=CP30_MONTHLY_AGGREGATE_REGEX
    script:
        "../scripts/CP30/validate_cp30_dispatch.py"


# --- Stage 1 against CP30 (16 Oct 2026) ---------------------------------------
# A Stage 1 family is twelve CP30 month scenarios <head>_<Mon>_<tag> (older ids:
# <head>_<Mon><year>_<tag>) that solve their own Stage 1 (no wholesale_from); its
# study name is the scenario name without the month, <head>_<tag>
# (CP30_FFR_2030_stage1). A monthly_aggregates study also resolves, on its months'
# Stage 1 sources (_wholesale_source), so CP30_2030_no_acceleration checks the
# stage1 files. Only the
# families in run_scenarios with validation.stage1.enabled (default true)
# join rule all (CP30_STAGE1_VALIDATION_TARGETS, Snakefile).

_CP30_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_CP30_FAMILY_MONTH_RE = re.compile(r"^(?P<head>.+?)_(?P<mon>" + "|".join(_CP30_MONTHS) + r")(?P<year>\d{4})?_(?P<tag>.+)$")


def _cp30_stage1_families(ids):
    """{study: twelve month scenario ids, Jan to Dec} for the complete CP30 Stage 1 families in ``ids``."""
    from scripts.utilities.scenario_detection import target_source  # judged against CP30's ES1 and CP.32
    found = {}
    for rid in ids:
        match = _CP30_FAMILY_MONTH_RE.match(str(rid))
        sc = scenarios.get(rid, {})
        if (match is None or target_source(sc) != "CP30"
                or not (sc.get("market") or {}).get("enabled", False) or sc.get("wholesale_from")):
            continue
        months = found.setdefault((match["head"], match["year"], match["tag"]), {})
        months[match["mon"]] = rid
    families = {}
    for (head, _year, tag), months in found.items():
        study = f"{head}_{tag}"
        if len(months) == 12 and study not in families and study not in CP30_MONTHLY_AGGREGATES:
            families[study] = [months[m] for m in _CP30_MONTHS]
    return families


CP30_STAGE1_FAMILIES = _cp30_stage1_families(scenarios)
CP30_STAGE1_STUDY_REGEX = (
    "|".join(map(re.escape, [*CP30_STAGE1_FAMILIES, *CP30_MONTHLY_AGGREGATES]))
    if (CP30_STAGE1_FAMILIES or CP30_MONTHLY_AGGREGATES)
    else r"(?!x)x"
)
CP30_STAGE1_VALIDATION_TARGETS = [
    f"{resources_path}/analysis/{study}_cp30_stage1_validation.csv"
    for study, months in _cp30_stage1_families(run_ids).items()
    if all(m in run_ids for m in months)
    and ((scenarios[months[0]].get("validation") or {}).get("stage1") or {}).get("enabled", True)
    # 18 Oct 2026: a central dispatch Stage 1 (copperplate_mode none) schedules on the real
    # network, so it is not CP30's unconstrained ES1 run and is not judged against it (the
    # key cannot be switched off per family: it is build-relevant under network_from).
    and str(((scenarios[months[0]].get("market") or {}).get("wholesale") or {})
            .get("copperplate_mode", "collapse")).lower() != "none"
]


def _cp30_stage1_months(wildcards):
    """The study's twelve Stage 1 scenarios (the scenarios whose *_wholesale files hold its Stage 1)."""
    if wildcards.study in CP30_STAGE1_FAMILIES:
        return CP30_STAGE1_FAMILIES[wildcards.study]
    return [_wholesale_source(s) for s in _cp30_monthly_aggregate_config(wildcards.study)["scenarios"]]


def _cp30_stage1_config(wildcards):
    return scenarios[_cp30_stage1_months(wildcards)[0]]


rule validate_CP30_stage1:
    """
    Validate a CP30 calendar year's Stage 1 (wholesale) dispatch against CP30 as
    soon as its twelve Stage 1 months exist, before any balancing solve: ES1
    energy by technology (NZT's CCS Gas inside the Gas line), curtailment,
    demand, interconnectors, electrolysis and hydrogen turbines, storage (INFO),
    and the price against CP.32's duration curve (mean, the price exceeded in
    1/10/30/50/70/90% of hours, the share of hours at or below GBP 0.5/MWh).
    A FAIL row fails the rule after the outputs are written, with a
    *_failed_run.csv copy kept. scripts/CP30/stage1_validation.py; tolerances
    in its DEFAULT_CONFIG (scenario block validation.stage1).
    """
    input:
        wholesale_networks=lambda wc: [f"{resources_path}/market/{s}_wholesale.nc" for s in _cp30_stage1_months(wc)],
        wholesale_prices=lambda wc: [f"{resources_path}/market/{s}_wholesale_price.csv" for s in _cp30_stage1_months(wc)],
        cp30_es1=lambda wc: _cp30_stage1_config(wc).get("source_data", {}).get("files", {}).get("es1", []),
        price_curve=lambda wc: ((_cp30_stage1_config(wc).get("validation") or {}).get("stage1") or {}).get(
            "price_curve_file", "data/CP30/cp32_price_duration_curve_2030.csv"),
        logic=["scripts/CP30/stage1_validation.py", "scripts/CP30/energy_validation.py",
               "scripts/consumer_bill/price_level_mapping.py"],
    output:
        csv=f"{resources_path}/analysis/{{study}}_cp30_stage1_validation.csv",
        json=f"{resources_path}/analysis/{{study}}_cp30_stage1_validation.json"
    params:
        scenario_config=_cp30_stage1_config
    log:
        "logs/CP30/validate_stage1_{study}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        study=CP30_STAGE1_STUDY_REGEX
    script:
        "../scripts/CP30/stage1_validation.py"


# --- Balancing family against NESO (18 Oct 2026) ------------------------------
# Report only (never fails a run): a monthly_aggregates study's twelve
# balancing months against NESO's published figures (CP.18 volumes, the
# GBP 3,570 m constraint cost, N2 final dispatch, CP.08 zones, the central-case
# score; delay families against NESO's delay-case costs).
# scripts/CP30/balancing_validation.py; benchmarks committed in
# data/CP30/validation_references/neso_balancing_benchmarks.csv (re-read from
# the workbook by a test). Any study can be requested; only the families named in
# studies/cp30_2030/study.yaml's balancing_validation_families (a tag such as cp30_network, or a
# study name; default none) join rule all (CP30_BALANCING_VALIDATION_TARGETS,
# Snakefile). The key is workflow selection, kept out of every scenario config
# (config_loader special_keys), so editing the list never re-triggers a solve.


def _cp30_balancing_study(name):
    """A family tag (cp30_network) or study name (CP30_2030_cp30_network) -> its monthly_aggregates study."""
    name = str(name)
    for study in (name, f"CP30_2030_{name}"):
        if study in CP30_MONTHLY_AGGREGATES:
            return study
    raise ValueError(
        f"balancing_validation_families: {name!r} is not a monthly_aggregates study "
        f"(nor CP30_2030_{name})"
    )


CP30_BALANCING_VALIDATION_FAMILIES = [
    _cp30_balancing_study(f) for f in _full_config["study_settings"]["balancing_validation_families"]
]
CP30_BALANCING_VALIDATION_TARGETS = [
    f"{resources_path}/analysis/{study}_cp30_balancing_validation.csv"
    for study in dict.fromkeys(CP30_BALANCING_VALIDATION_FAMILIES)
]


def _cp30_balancing_months(wildcards):
    return _cp30_monthly_aggregate_config(wildcards.study)["scenarios"]


def _cp30_balancing_settings(wildcards):
    """The few scenario settings the check reads (first month), not the whole config."""
    from scripts.CP30.balancing_validation import scenario_settings

    return scenario_settings(scenarios[_cp30_balancing_months(wildcards)[0]])


def _cp30_balancing_files(wildcards):
    from scripts.CP30.balancing_validation import balancing_config
    from scripts.solve.secured_boundaries import DEFAULT_GEOMETRY

    settings = _cp30_balancing_settings(wildcards)
    cfg = balancing_config(settings["cp30_balancing_validation"])
    geometry = settings["secured_capability"].get("geometry_file") or DEFAULT_GEOMETRY
    return [cfg["benchmarks_file"], cfg["zones_file"], geometry]


rule validate_CP30_balancing:
    """
    Compare a solved CP30 2030 balancing family (a monthly_aggregates
    study, each month's Stage 1 from its wholesale_from scenario) with NESO's
    published figures: CP.18 balancing volumes (and the ES1-start restatement),
    the NESO-comparable redispatch cost against GBP 3,570 m with its split by
    technology, final dispatch against N2 (wind, gas incl. CHP, net export, the
    unabated gas share), clean turn-down by NESO's 17 CP.08 zones (buses zoned on
    the family's own network at run time), the central-case score, and, for a
    delay family, NESO's delay-case cost. Report only: PASS/WARN/INFO rows, never
    a failed run. Tolerances in DEFAULT_CONFIG (scenario block
    validation.balancing).
    """
    input:
        networks=lambda wc: [f"{resources_path}/market/{s}_balancing.nc" for s in _cp30_balancing_months(wc)],
        redispatch=lambda wc: [f"{resources_path}/market/{s}_redispatch_summary.csv" for s in _cp30_balancing_months(wc)],
        constraint_costs=lambda wc: [f"{resources_path}/market/{s}_constraint_costs.csv" for s in _cp30_balancing_months(wc)],
        stage1_links=lambda wc: [f"{resources_path}/market/{_wholesale_source(s)}_wholesale_links.csv"
                                 for s in _cp30_balancing_months(wc)],
        market_summaries=lambda wc: [f"{resources_path}/analysis/{s}_market_summary.json"
                                     for s in _cp30_balancing_months(wc)],
        annual_summary=f"{resources_path}/analysis/{{study}}_annual_summary.json",
        # Both turn-down bases and the headline one (rules/turn_down_allocation.smk, 20 Oct 2026).
        turn_down_summary=lambda wc: turn_down_summary_path(wc.study),
        references=_cp30_balancing_files,
        logic=["scripts/CP30/balancing_validation.py", "scripts/solve/secured_boundaries.py",
               "scripts/market/neso_zones.py"],
    output:
        csv=f"{resources_path}/analysis/{{study}}_cp30_balancing_validation.csv",
        json=f"{resources_path}/analysis/{{study}}_cp30_balancing_validation.json"
    params:
        months=_cp30_balancing_months,
        settings=_cp30_balancing_settings
    log:
        "logs/CP30/validate_balancing_{study}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        study=CP30_MONTHLY_AGGREGATE_REGEX
    script:
        "../scripts/CP30/balancing_validation.py"


# --- Fleet location (19 Oct 2026) ---------------------------------------------
# Where the built fleet sits against CP30 ES5, NESO's CP.08 zones and the secured
# boundaries; offshore and interconnector landings against their references; bus
# positions against circuit lengths. Runs on every CP30 build that runs
# validate_CP30_locational (Snakefile, cp30_validation_targets).
# scripts/CP30/fleet_location_validation.py; a FAIL row fails the rule after the
# outputs are written (validation.fleet_location.fail_on_fail); the position
# check FAILs only on misplaced buses that matter (position audit, 19 Oct 2026), with
# reviewed exemptions in position_exemptions_file. Params are the few settings it
# reads, not the whole config.


def _cp30_fleet_location_settings(wildcards):
    from scripts.CP30.fleet_location_validation import scenario_settings

    return scenario_settings(scenarios[wildcards.scenario])


def _cp30_fleet_location_files(wildcards):
    from scripts.CP30.fleet_location_validation import fleet_location_config

    cfg = fleet_location_config(_cp30_fleet_location_settings(wildcards)["cp30_fleet_location_validation"])
    return [f for f in (cfg["zones_file"], cfg["landing_additions_file"], cfg["ic_reference_file"],
                        cfg["position_exemptions_file"]) if f]


def _cp30_fleet_location_etys_components(wildcards):
    if str(scenarios[wildcards.scenario].get("network_model", "")).upper() != "ETYS":
        return []
    return f"{resources_path}/network/ETYS_{_get_etys_year(wildcards.scenario)}_components.csv"


rule validate_CP30_fleet_location:
    """
    Fleet location of a built CP30 network: capacity by ES5 region against ES5,
    by NESO CP.08 zone (INFO), each side of the secured boundaries (B4/B6 against
    ES5's Scottish regions), named offshore farms at their reference landings,
    interconnectors at their NESO register substations and CP30 ratings, and bus
    positions against circuit lengths (inconsistent circuits WARN; misplaced buses,
    and circuits over by more than 5 km across a boundary, region or zone, FAIL).
    Boundary sides use the scenario's secured_capability include block.
    """
    input:
        network=f"{resources_path}/network/{{scenario}}.nc",
        cp30_es5=lambda wc: scenarios[wc.scenario].get("source_data", {}).get("files", {}).get("es5", []),
        boundaries=lambda wc: (_cp30_fleet_location_settings(wc).get("geometry_file")
                               or "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson"),
        gsp_polygons=lambda wc: _cp30_locational_build_inputs(wc)["gsp_polygons"],
        gsp_group_regions=lambda wc: _cp30_locational_build_inputs(wc)["gsp_group_regions"],
        offshore_crosswalk=lambda wc: _cp30_locational_build_inputs(wc)["offshore_crosswalk"],
        ic_composition=lambda wc: get_cp30_interconnector_composition(scenarios[wc.scenario]),
        etys_components=_cp30_fleet_location_etys_components,
        references=_cp30_fleet_location_files,
        logic=["scripts/CP30/fleet_location_validation.py", "scripts/CP30/spatial_allocation.py",
               "scripts/solve/secured_boundaries.py"]
    output:
        csv=f"{resources_path}/CP30/validation/{{scenario}}_fleet_location.csv",
        json=f"{resources_path}/CP30/validation/{{scenario}}_fleet_location.json"
    params:
        settings=_cp30_fleet_location_settings
    log:
        "logs/CP30/validate_fleet_location_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    script:
        "../scripts/CP30/fleet_location_validation.py"


# --- Secured boundary capability by family (19 Oct 2026) ----------------------
# Report only: the secured-boundary capability every market family in
# run_scenarios resolves to in its balancing stage (base outage block +
# network_delivery.secured_capability, as solve_balancing), one row per family x
# boundary, with WARN flags (a delay family above its central twin; a capability
# file other than the central family's). From the config alone; the params are the
# resolved settings only (scripts/solve/secured_capability_audit.py audit_spec), so
# an unrelated config edit does not rerun it. The central family's tag is
# config.yaml's secured_capability_audit_central (default cp30_network; a workflow key,
# config_loader special_keys).

from scripts.solve.secured_capability_audit import (
    DEFAULT_CENTRAL_TAG as _SCA_DEFAULT_CENTRAL,
    audit_spec as _secured_capability_audit_spec,
    spec_files as _secured_capability_spec_files,
)

SECURED_CAPABILITY_AUDIT_SPEC = _secured_capability_audit_spec(
    scenarios, run_ids, str(_full_config["study_settings"]["secured_capability_audit_central"] or _SCA_DEFAULT_CENTRAL))
SECURED_CAPABILITY_AUDIT_TARGETS = (
    [f"{resources_path}/analysis/secured_capability_by_family.csv"]
    if SECURED_CAPABILITY_AUDIT_SPEC["families"] else []
)


rule audit_secured_capability:
    """
    Secured boundary capability by balancing family (report only): boundary,
    capability year asked and used, capability, NOA seasonal factors, historic
    derate, mean limit over the family's hours, capability file, and the case.
    """
    input:
        files=lambda wc: [f for v in _secured_capability_spec_files(SECURED_CAPABILITY_AUDIT_SPEC).values() for f in v],
        logic=["scripts/solve/secured_capability_audit.py", "scripts/solve/secured_boundaries.py",
               "scripts/market/network_delivery.py"]
    output:
        csv=f"{resources_path}/analysis/secured_capability_by_family.csv"
    params:
        spec=SECURED_CAPABILITY_AUDIT_SPEC
    log:
        "logs/CP30/audit_secured_capability.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/solve/secured_capability_audit.py"


# --- Gas station schedule retention check (F25.9, 27 Oct 2026) ----------------
# The schedule (spatial_proxy.gas_station_schedule) is a build input and its full
# builder needs a built network, so it cannot be a rule output without a cycle.
# This report rule recomputes only its retention columns (CM register coverage,
# the declared stations, the FES 2025 HT 2030 claims) and fails with the refresh
# command on drift. It joins rule all for every schedule an active CP30
# brownfield build reads; no build reads its stamp, so it moves no result.
from scripts.utilities.scenario_detection import uses_brownfield as _gas_check_uses_brownfield

GAS_SCHEDULE_CHECK_CM_REGISTER = "data/market/capacity_market/cm_register_cmu_20260923.csv"
GAS_SCHEDULE_CHECK_FES_YEAR = 2025   # FES_2030 in the builder cites FES 2025 HT Gen_BB009


def _gas_station_schedules(ids):
    """Schedule file stem -> path, for every schedule an active brownfield build reads."""
    found = {}
    for rid in ids:
        sc = scenarios.get(rid, {})
        if not _gas_check_uses_brownfield(sc):
            continue
        path = ((sc.get("fleet") or {}).get("spatial_proxy") or {}).get("gas_station_schedule")
        if path:
            found.setdefault(Path(path).stem, str(path).replace("\\", "/"))
    return found


GAS_STATION_SCHEDULES = _gas_station_schedules(build_ids)
GAS_SCHEDULE_CHECK_TARGETS = [
    f"{resources_path}/CP30/validation/{stem}_retention_check.json" for stem in GAS_STATION_SCHEDULES
]


rule verify_CP30_gas_station_schedule:
    """The committed gas station schedule's CM / FES retention columns still reproduce (report only)."""
    input:
        script="scripts/CP30/build_gas_station_schedule.py",
        schedule=lambda wc: GAS_STATION_SCHEDULES[wc.schedule],
        cm_register=GAS_SCHEDULE_CHECK_CM_REGISTER,
        cm_logic="scripts/market/capacity_market.py",
        fes_data=f"{resources_path}/FES/FES_{GAS_SCHEDULE_CHECK_FES_YEAR}_data.csv",
    output:
        stamp=f"{resources_path}/CP30/validation/{{schedule}}_retention_check.json"
    wildcard_constraints:
        schedule="|".join(map(re.escape, GAS_STATION_SCHEDULES)) or r"(?!x)x"
    conda:
        "../envs/pypsa-gb.yaml"
    shell:
        "python {input.script} --check --output {input.schedule} --cm-register {input.cm_register} "
        "--fes-data {input.fes_data} --stamp {output.stamp}"
