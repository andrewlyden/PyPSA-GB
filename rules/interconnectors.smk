"""
Interconnectors workflow for PyPSA-GB with European generation mix integration.

This workflow processes interconnector data from DUKES and NESO sources,
enriches it with locations and capacities, and integrates it into the PyPSA network.
Additionally, downloads European electricity generation mix data for border pricing.
"""

resources_path = "resources"
data_path = "data"


# Every CP30 gate here is about where the numbers come from (CP30 ES1, CP.29 prices, CP30's
# cable set, which FES edition serves as the proxy), so it reads target_source().
from scripts.utilities.scenario_detection import target_source


def get_interconnector_fes_year(scenario_config):
    if target_source(scenario_config) == "CP30":
        return scenario_config.get("fleet", {}).get("spatial_proxy", {}).get("FES_year", 2025)
    return scenario_config.get("FES_year", 2024)


def get_interconnector_fes_scenario(scenario_config):
    if target_source(scenario_config) == "CP30":
        return scenario_config.get("fleet", {}).get("spatial_proxy", {}).get("FES_scenario")
    return scenario_config.get("FES_scenario", None)


def get_interconnector_cp30_resource(scenario_config, resource_name):
    return scenario_config.get("source_data", {}).get("files", {}).get(resource_name, []) if target_source(scenario_config) == "CP30" else []

# =============================================================================
# European Generation Mix Integration
# =============================================================================

rule download_european_generation_mix:
    """Download European electricity generation mix data from NESO FES API."""
    output:
        generation_mix=f"{resources_path}/interconnectors/european_generation_mix_{{fes_year}}.csv",
        metadata=f"{resources_path}/interconnectors/european_generation_mix_{{fes_year}}_metadata.json"
    params:
        api_endpoint="https://api.neso.energy/api/3/action/datapackage_show",
        dataset_id="fes-european-electricity-supply-data-table-es2"
    log:
        "logs/download_european_generation_mix_{fes_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/download_european_generation.py"

rule process_european_generation_mix:
    """Process European generation mix into marginal costs and price differentials."""
    input:
        generation_mix=f"{resources_path}/interconnectors/european_generation_mix_{{fes_year}}.csv",
        fuel_prices=f"{resources_path}/marginal_costs/fuel_prices_{{fes_year}}.csv",
        carbon_prices=f"{resources_path}/marginal_costs/carbon_prices_{{fes_year}}.csv"
    output:
        marginal_costs=f"{resources_path}/interconnectors/european_marginal_costs_{{fes_year}}.csv",
        price_differentials=f"{resources_path}/interconnectors/price_differentials_{{fes_year}}.csv"
    params:
        gb_price=config.get("interconnectors", {}).get("pricing", {}).get("gb_price", 50.0)
    log:
        "logs/process_european_generation_mix_{fes_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/process_european_generation.py"

# =============================================================================
# Historical Interconnector Flows
# =============================================================================

rule extract_historical_interconnector_flows:
    """Extract actual historical interconnector flows from ESPENI data."""
    input:
        espeni=data_path + "/demand/espeni.csv"
    output:
        flows=resources_path + "/interconnectors/historical_flows_{year}.csv",
        metadata=resources_path + "/interconnectors/historical_flows_{year}_metadata.json"
    params:
        year=lambda wildcards: wildcards.year
    log:
        "logs/extract_historical_interconnector_flows_{year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/extract_historical_flows.py"

# =============================================================================
# ENTSO-E Day-Ahead Price Fetching (optional)
# =============================================================================

def _entsoe_archive_input(wildcards):
    """The shipped ENTSO-E prices of the year, where the repository has them (fetch_entsoe_prices.archive_path)."""
    path = Path(data_path) / "interconnectors" / "entsoe_day_ahead" / f"day_ahead_prices_{wildcards.entsoe_year}_eur_per_mwh.csv"
    return str(path).replace("\\", "/") if path.exists() else []


rule fetch_entsoe_day_ahead_prices:
    """Hourly day-ahead prices by country: the shipped file of the year (2015-2024), else the ENTSO-E API.

    The API needs a token in .env (ENTSOE_API_TOKEN=...) or in the environment. The shipped file is a declared
    input, so a cached output older than it is rebuilt (a May 2026 cache without Germany failed HT35_clustered).
    """
    input:
        archive=_entsoe_archive_input
    output:
        prices=resources_path + "/interconnectors/entsoe_day_ahead_prices_{entsoe_year}.csv"
    params:
        year=lambda wildcards: wildcards.entsoe_year,
        pricing_config=lambda wildcards: config.get("interconnectors", {}).get("pricing", {})
    log:
        "logs/fetch_entsoe_day_ahead_prices_{entsoe_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/fetch_entsoe_prices.py"

# =============================================================================
# Core Interconnector Data Processing
# =============================================================================

rule ingest_dukes:
    """Ingest DUKES interconnector data."""
    input:
        data_path + "/interconnectors/DUKES_5.13_2025.csv"
    output:
        resources_path + "/interconnectors/interconnectors_dukes.csv"
    log:
        "logs/ingest_dukes_interconnectors.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/ingest_dukes.py"

rule ingest_neso_register:
    """Ingest NESO Interconnector Register data."""
    input:
        data_path + "/interconnectors/NESO_interconnector_register.csv"
    output:
        resources_path + "/interconnectors/interconnectors_neso.csv"
    log:
        "logs/ingest_neso_register.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/ingest_neso_register.py"

rule combine_datasets:
    """Combine DUKES and NESO register datasets."""
    input:
        dukes=resources_path + "/interconnectors/interconnectors_dukes.csv",
        neso_register=resources_path + "/interconnectors/interconnectors_neso.csv"
    output:
        resources_path + "/interconnectors/interconnector_data_combined.csv"
    log:
        "logs/combine_interconnector_datasets.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/combine_datasets.py"

rule enrich_locations:
    """Enrich interconnector data with location information."""
    input:
        resources_path + "/interconnectors/interconnector_data_combined.csv",
        data_path + "/interconnectors/NESO_interconnector_register.csv"
    output:
        resources_path + "/interconnectors/interconnector_data_with_locations.csv"
    log:
        "logs/enrich_interconnector_locations.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/enrich_locations.py"

rule clean:
    """Clean and standardize interconnector data."""
    input:
        resources_path + "/interconnectors/interconnector_data_with_locations.csv"
    output:
        resources_path + "/interconnectors/interconnectors_clean.csv"
    params:
        overrides_file="data/interconnectors/overrides.csv"
    log:
        "logs/clean_interconnector_data.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/clean_interconnectors.py"

rule pipeline_placeholder:
    """Create placeholder for future pipeline data."""
    output:
        resources_path + "/interconnectors/pipeline_placeholder.csv"
    log:
        "logs/pipeline_placeholder.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/pipeline_placeholder.py"

rule map_to_buses:
    """Map interconnectors to the buses of this scenario's own built network.

    Keyed on the scenario, not the network model: two ETYS builds can differ in
    topology (ETYS year, reinforcements, dead-island pruning, HND hybrid), and a
    landing bus missing from the network is skipped with only a warning, so a
    mapping made on another scenario's topology could silently drop a link. It
    also means the mapping no longer depends on which scenarios are selected.
    """
    input:
        interconnectors=resources_path + "/interconnectors/interconnectors_clean.csv",
        network=resources_path + "/network/{scenario}_network.nc",
        # Curated ETYS landing buses override the coordinate match (Viking, 26 Sep 2026).
        bus_mapping="data/interconnectors/bus_mapping.csv",
    output:
        resources_path + "/interconnectors/{scenario}_interconnectors_mapped.csv"
    wildcard_constraints:
        scenario=_regex_from_list(build_ids)
    params:
        network_model=lambda wc: scenarios[wc.scenario]["network_model"]
    log:
        "logs/map_interconnectors_to_buses_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/map_to_buses.py"

def get_cp30_interconnector_composition(scenario_config):
    """The CP30 cable composition file (interconnectors.cp30_composition.file, 15 Oct 2026), else [].

    CP30's cable set for a future CP30 build: CP30 nominal ratings, MaresConnect added, the Isle of
    Man cable outside the count, summing to ES1 with nothing scaled. Future CP30 scenarios only.
    """
    block = (scenario_config.get('interconnectors', {}) or {}).get('cp30_composition', {}) or {}
    modelled_year = scenario_config.get('modelled_year') or scenario_config.get('year') or 9999
    if not (target_source(scenario_config) == "CP30" and int(modelled_year) > 2024):
        return []
    return [block.get('file', data_path + '/interconnectors/cp30_interconnector_composition.csv')]


rule availability:
    """Generate scenario-specific interconnector availability time series."""
    input:
        resources_path + "/interconnectors/interconnectors_clean.csv",
        # Cables a CP30 composition adds get profiles too, appended after the register cables.
        cp30_composition=lambda wc: get_cp30_interconnector_composition(scenarios[wc.scenario]),
    output:
        resources_path + "/interconnectors/{scenario}_interconnector_availability.csv"
    params:
        demand_year=lambda wc: scenarios[wc.scenario].get(
            'demand_year', scenarios[wc.scenario].get('modelled_year', 2020)
        ),
        availability_config=lambda wc: scenarios[wc.scenario].get(
            'interconnectors', {}
        ).get('availability', {}),
    log:
        "logs/interconnector_availability_{scenario}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/availability.py"

def get_interconnector_inputs(wildcards):
    """
    Get appropriate interconnector inputs based on scenario type.
    
    For historical scenarios: Use actual ESPENI flows
    For future scenarios: Use availability profiles + price differentials
    """
    import sys
    from pathlib import Path
    
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.utilities.scenario_detection import is_historical_scenario
    
    scenario_config = scenarios[wildcards.scenario]
    modelled_year = scenario_config.get('modelled_year') or scenario_config.get('year')
    
    # Base inputs always required
    inputs = {
        'interconnectors': f"{resources_path}/interconnectors/{wildcards.scenario}_interconnectors_mapped.csv",
    }
    
    # Determine interconnector pricing source
    pricing_cfg = scenario_config.get('interconnectors', {}).get('pricing', {})
    pricing_source = pricing_cfg.get('source', 'derived')
    
    if is_historical_scenario(modelled_year):
        # Historical: use actual flows from ESPENI
        inputs['historical_flows'] = f"{resources_path}/interconnectors/historical_flows_{modelled_year}.csv"
        inputs['historical_flows_metadata'] = f"{resources_path}/interconnectors/historical_flows_{modelled_year}_metadata.json"
        # Optionally add ENTSO-E prices for accurate shadow pricing
        if pricing_source == 'entsoe':
            inputs['entsoe_prices'] = f"{resources_path}/interconnectors/entsoe_day_ahead_prices_{modelled_year}.csv"
    else:
        # Future: use availability + price differentials
        inputs['availability'] = f"{resources_path}/interconnectors/{wildcards.scenario}_interconnector_availability.csv"
        if not (target_source(scenario_config) == "CP30" and pricing_source == 'cp30'):
            inputs['price_differentials'] = f"{resources_path}/interconnectors/price_differentials_{get_interconnector_fes_year(scenario_config)}.csv"
        if target_source(scenario_config) == "CP30":
            inputs['cp30_es1'] = get_interconnector_cp30_resource(scenario_config, 'es1')
            composition = get_cp30_interconnector_composition(scenario_config)
            if composition:
                inputs['cp30_interconnector_composition'] = composition
        if pricing_source == 'cp30':
            inputs['cp30_interconnector_prices'] = get_interconnector_cp30_resource(scenario_config, 'interconnector_prices')
        # An hourly ENTSO-E shape can be re-centred to the CP30 annual trade
        # price.  ``flat`` instead uses the annual CP.29 import/export targets
        # unchanged in every snapshot, with no historical-price data dependency.
        # The default preserves the established ENTSO-E-shape behaviour.
        hourly_shape = str(pricing_cfg.get('hourly_shape', 'entsoe')).lower()
        if hourly_shape not in {'entsoe', 'flat'}:
            raise ValueError(
                f"Unsupported interconnector pricing hourly_shape={hourly_shape!r}; "
                "expected 'entsoe' or 'flat'"
            )
        # The single foreign price basis (item D2, the default) also prices on the ENTSO-E shape, whatever the
        # level's source (add_to_network refuses it without one; 18 Nov 2026: a fresh Test_Market failed there).
        single_basis = str(pricing_cfg.get('foreign_price_basis', 'separate')).strip().lower() == 'single'
        if (pricing_source in {'entsoe', 'cp30'} or single_basis) and hourly_shape == 'entsoe':
            weather_year = scenario_config.get('renewables_year', modelled_year)
            hourly_shape_year = pricing_cfg.get('hourly_shape_year', weather_year)
            inputs['entsoe_prices'] = f"{resources_path}/interconnectors/entsoe_day_ahead_prices_{hourly_shape_year}.csv"
            # Item IC1: the shape year's GB-wind term swapped for the weather
            # year's, so a shape borrowed from another year keeps the
            # wind-price correlation.
            if (pricing_cfg.get('shape_conditioning') or {}).get('enabled', False):
                inputs['entsoe_prices'] = (
                    f"{resources_path}/interconnectors/entsoe_price_shape_{hourly_shape_year}_wind{weather_year}.csv"
                )
        # Task Y3 (FES 2050 plan): foreign_price_source "tyndp" replaces the
        # ENTSO-E shape with the TYNDP shape of the modelled year at its own level.
        # The key is absent in every other scenario, so their inputs are unchanged.
        if is_tyndp_foreign_pricing(pricing_cfg):
            inputs['entsoe_prices'] = tyndp_shape_path(scenario_config)
        # Item F13 (7 Oct 2026): "cp30_basis" = TYNDP NT2030 rebased to CP30's
        # fuel, at its own level. Absent in every scenario that does not set it.
        if is_cp30_basis_foreign_pricing(pricing_cfg):
            inputs['entsoe_prices'] = cp30_basis_price_path(scenario_config)
        tranches = pricing_cfg.get('volume_tranches') or {}
        if tranches.get('enabled', False):
            inputs['market_demand'] = tranches['demand_file']
        # Task Y6b: scale the cables to the scenario's own FES release (FES 2025 HT
        # 2050: 21.8 GW) instead of the FES 2024 file scale_interconnectors_to_fes
        # defaults to. Only fleet.fes_supply.interconnectors sets it.
        if fes_supply_block(wildcards.scenario, 'interconnectors').get('capacity_from_scenario_fes_year', False):
            inputs['fes_capacity_data'] = f"{resources_path}/FES/FES_{scenario_config['FES_year']}_data.csv"

    return inputs


rule build_interconnector_price_shape:
    """ENTSO-E hourly prices of one year, conditioned on another year's GB wind (item IC1).

    Per country, the part of the price GB offshore output explains in the shape
    year is replaced by the same term driven by the weather year's output, on
    the shape year's calendar. The level is re-centred to CP.29 later, in
    add_to_network, so only the shape matters. See scripts/interconnectors/price_shape.py.
    """
    input:
        prices=resources_path + "/interconnectors/entsoe_day_ahead_prices_{shape_year}.csv",
        wind_shape=resources_path + "/renewable/profiles/wind_offshore_{shape_year}.csv",
        wind_weather=resources_path + "/renewable/profiles/wind_offshore_{weather_year}.csv",
    output:
        shape=resources_path + "/interconnectors/entsoe_price_shape_{shape_year}_wind{weather_year}.csv",
        audit=resources_path + "/interconnectors/entsoe_price_shape_{shape_year}_wind{weather_year}_audit.json",
    wildcard_constraints:
        shape_year=r"\d{4}",
        weather_year=r"\d{4}",
    log:
        "logs/interconnectors/build_interconnector_price_shape_{shape_year}_wind{weather_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/build_price_shape.py"

rule add_interconnectors_to_network:
    """
    Add cross-border interconnectors to network with European supply modeling.
    
    Takes the network with base demand, renewables, thermal generators, and storage,
    then integrates international electricity interconnector links. Interconnectors
    enable electricity exchange between Great Britain and neighboring countries.
    
    Transforms: {scenario}_network_demand_renewables_thermal_storage.nc → {scenario}_network_demand_renewables_thermal_storage_interconnectors.nc
    
    Architecture:
      - External buses: Connection points to European countries (France, Belgium, etc.)
      - Foreign counterparties: bounded import supply and signed export demand,
        priced from hourly foreign day-ahead shapes or static scenario prices
      - Future DC cables: paired non-negative import/export links with losses
        applied in the physical direction of flow
      - Shared-capacity constraints: both directions use one cable rating
      - Historical fixed-flow links retain the legacy observed-flow formulation
    
    Interconnector Coverage:
      - France: Interconnecteur France-Angleterre (IFA, IFA2, ElecLink)
      - Netherlands: BritNed
      - Belgium: Nemolink
      - Norway: NSL (Norwegian connection)
      - Denmark: Viking Link
      - Ireland: EuroConnector, EastWest, Moyle
    
    Processing:
      1. Load network with demand, renewables, thermal generators, storage
      2. Load interconnector mapping data (locations, capacities)
      3. Create external buses at international connection points
      4. [Future] Add foreign import supply and export demand counterparties
      5. [Future] Add directional import and export Link objects with:
         - non-negative flow in the declared physical direction
         - one shared cable capacity enforced during optimization
         - directional efficiency for transmission losses
         - optional ramp limits and a circulation-safe bid/ask spread
      6. [Future] Apply availability profiles (time-varying capacity if needed)
      7. [Historical] Apply actual ESPENI flows as fixed constraints (p_set)
      8. Validate network (capacity, bidirectionality, external generators)
      9. Export to NetCDF
    
    Inputs:
      - network: Network with all domestic components (NetCDF)
      - interconnector_data: Mapped interconnector links with capacities (CSV)
      - [Historical] historical_flows: Actual interconnector flows from ESPENI (CSV)
      - [Future] availability: Interconnector availability profiles (CSV)
      - [Future] price_differentials: European marginal costs by country (CSV)
    
    Output:
      - network: Fully assembled network ready for clustering or solving
    
    Network Model Handling:
      - ETYS: Interconnectors mapped to transmission nodes (typically southern UK)
      - Reduced: Interconnectors at representative regional buses
      - Zonal: Interconnectors at zone buses closest to connection points
    
    Capacity Assumptions (GW):
      - France (IFA/IFA2/ElecLink): 2-5 GW total
      - Netherlands (BritNed): 1 GW
      - Belgium (Nemolink): 1 GW
      - Norway (NSL): 0.7-1.4 GW
      - Denmark (Viking Link): 1.4 GW
      - Note: Capacities increase over time with new interconnectors
    
    Transmission Efficiency:
      - HVAC submarine cables: 95-98%
      - HVDC submarine cables: 96-99%
      - Overhead lines: 99%+
      - Typical loss: 0.5-2% per interconnector
    
    European Supply Modeling:
      - External generators have 100 GW capacity (non-binding constraint)
      - Marginal costs from European generation mix analysis:
        * Calculated from European renewable/thermal generation shares
        * Based on fuel prices, carbon costs, and merit order dispatch
        * Typical range: £30-70/MWh depending on country and scenario
      - Allows optimizer to balance GB generation vs. imports economically
    
    Historical vs Future Data:
      - Historical (≤2024): ESPENI actual flows stored for post-solve validation/comparison
        * Flows are NOT constrained (p_set) - this causes infeasibility
        * Instead, stored in network.historical_interconnector_flows for validation
        * Used as initial guess (p0) to aid solver convergence
        * Post-solve analysis compares optimized vs actual flows
      - Future (>2024): Optimizable flows with European marginal costs + availability profiles
      - Both scenarios allow full optimization, historical validates against market data
    
    Performance:
      - ETYS: ~15-25s (many domestic nodes to connect to)
      - Reduced: ~5-10s (fewer regional nodes)
      - Zonal: ~2-5s (minimal zone nodes)
    
    See Also:
      - scripts/interconnectors/add_to_network.py - Integration implementation
      - scripts/interconnectors/process_european_generation.py - European cost calculation
      - Fully assembled network ready for: network_clustering.smk or solve.smk
    """
    input:
        unpack(get_interconnector_inputs),
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl"
    output:
        network=f"{resources_path}/network/{{scenario}}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc"
    params:
        scenario=lambda wc: wc.scenario,
        network_model=lambda wc: scenarios[wc.scenario]["network_model"],
        modelled_year=lambda wc: scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year'),
        is_historical=lambda wc: (scenarios[wc.scenario].get('modelled_year') or scenarios[wc.scenario].get('year', 9999)) <= 2024,
        fes_pathway=lambda wc: get_interconnector_fes_scenario(scenarios[wc.scenario]),
        scenario_source=lambda wc: scenarios[wc.scenario].get('scenario_source', None),
        cp30_pathway=lambda wc: scenarios[wc.scenario].get('CP30_pathway', None),
        pricing_config=lambda wc: scenarios[wc.scenario].get('interconnectors', {}).get('pricing', {}),
        historical_flow_mode=lambda wc: scenarios[wc.scenario].get('interconnectors', {}).get('historical_flow_mode', 'fixed'),
        renewables_year=lambda wc: scenarios[wc.scenario].get('renewables_year', None)
    message:
        "Adding interconnectors to network for {wildcards.scenario} (model: {params.network_model})"
    benchmark:
        "benchmarks/interconnectors/add_interconnectors_to_network_{scenario}.txt"
    log:
        "logs/interconnectors/add_interconnectors_to_network_{scenario}.log"
    wildcard_constraints:
        scenario="[A-Za-z0-9_-]+"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/add_to_network.py"


# =============================================================================
# European prices from TYNDP for the
# modelled year (year-generic in stage 5; the TYNDP files of that year must be extracted)
# =============================================================================
# Opt-in per scenario, for the FES long-range entries (HT50_* today):
#
#   interconnectors:
#     pricing:
#       foreign_price_source: "tyndp"      # absent / "entsoe" = today's behaviour
#       tyndp:
#         scenario: "DE"                   # DE (central) | GA | NTplus2026 (decision 27)
#         route: "tyndp_shape"             # tyndp_shape (central) | ic1_rescale (decision 8)
#
# See scripts/interconnectors/tyndp_prices.py for the method and
# data/interconnectors/tyndp2024/README.md for the data and licence.

TYNDP_DATA_DIR = f"{data_path}/interconnectors/tyndp2024"


def is_tyndp_foreign_pricing(pricing_cfg):
    return str((pricing_cfg or {}).get('foreign_price_source', 'entsoe')).strip().lower() == 'tyndp'


def tyndp_variant_token(pricing_cfg):
    from scripts.interconnectors.tyndp_prices import tyndp_variant
    return tyndp_variant(pricing_cfg)


def tyndp_shape_path(scenario_config):
    pricing_cfg = scenario_config.get('interconnectors', {}).get('pricing', {})
    year = scenario_config.get('modelled_year') or scenario_config.get('year')
    weather_year = scenario_config.get('renewables_year', year)
    return (
        f"{resources_path}/interconnectors/tyndp_{tyndp_variant_token(pricing_cfg)}_{year}"
        f"_shape_wind{weather_year}.csv"
    )


def tyndp_eur_to_gbp():
    """The EUR->GBP rate of the scenarios that use TYNDP prices (one value, or the defaults' 0.86)."""
    rates = {
        float(cfg.get('interconnectors', {}).get('pricing', {}).get('eur_to_gbp', 0.86))
        for cfg in scenarios.values()
        if is_tyndp_foreign_pricing(cfg.get('interconnectors', {}).get('pricing', {}))
    }
    if len(rates) > 1:
        raise ValueError(f"TYNDP-priced scenarios disagree on interconnectors.pricing.eur_to_gbp: {sorted(rates)}")
    return rates.pop() if rates else 0.86


def tyndp_shape_inputs(wildcards):
    from scripts.interconnectors.tyndp_prices import ntplus_bracket, parse_variant, tyndp_input_files
    scenario, route = parse_variant(wildcards.variant)
    if scenario == 'NTplus2026' and ntplus_bracket(int(wildcards.year)) is not None:
        # User decision (11 Nov 2026): a year NT+ does not publish (2045) blends the shapes of the NT+ years either
        # side, each built by this rule on the same weather year.
        y0, y1, _ = ntplus_bracket(int(wildcards.year))
        shape = resources_path + "/interconnectors/tyndp_{v}_{y}_shape_wind{w}"
        return {f'bracket_{side}_{kind}': shape.format(v=wildcards.variant, y=y, w=wildcards.weather_year) + ext
                for side, y in (('lo', y0), ('hi', y1)) for kind, ext in (('shape', '.csv'), ('audit', '_audit.json'))}
    inputs = {
        'wind_weather': f"{resources_path}/renewable/profiles/wind_offshore_{wildcards.weather_year}.csv",
    }
    # The TYNDP year is the modelled year (stage 5): its own files, which fail as missing
    # inputs until they are extracted (scripts/interconnectors/extract_tyndp_hourly.py).
    inputs.update(tyndp_input_files(wildcards.variant, int(wildcards.year), TYNDP_DATA_DIR))
    if route == 'ic1_rescale':
        # Decision 8 sensitivity: the IC1 2024 ENTSO-E shape on the weather year's wind.
        inputs['ic1_shape'] = (
            f"{resources_path}/interconnectors/entsoe_price_shape_2024_wind{wildcards.weather_year}.csv"
        )
    return inputs


rule build_tyndp_price_shape:
    """TYNDP hourly foreign prices of the modelled year, conditioned on the weather year's GB wind (task Y3).

    The shape climate year (TYNDP 2024 CY2009, NT+ WS092) is laid on the weather
    year's calendar and conditioned on its GB offshore wind with IC1's
    condition_prices_on_wind, using TYNDP's own UK00 offshore wind as the
    shape-year regressor; the level is TYNDP's climate-weighted mean in GBP 2024.
    add_to_network uses it at its own level (no CP.29 re-centring) when a scenario
    sets interconnectors.pricing.foreign_price_source: "tyndp".
    """
    input:
        unpack(tyndp_shape_inputs)
    output:
        shape=resources_path + "/interconnectors/tyndp_{variant}_{year}_shape_wind{weather_year}.csv",
        audit=resources_path + "/interconnectors/tyndp_{variant}_{year}_shape_wind{weather_year}_audit.json",
    params:
        eur_to_gbp=lambda wc: tyndp_eur_to_gbp(),
    wildcard_constraints:
        variant=r"(DE|GA|NTplus2026)(_ic1rescale)?",
        year=r"20[2-9][0-9]",
        weather_year=r"\d{4}",
    log:
        "logs/interconnectors/build_tyndp_price_shape_{variant}_{year}_wind{weather_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/build_tyndp_price_shape.py"


# =============================================================================
# 2030 foreign prices on CP30's fuel basis
# =============================================================================
# Opt-in per scenario (the CP30 family sets it in config/defaults/method/brownfield.yaml):
#
#   interconnectors:
#     pricing:
#       foreign_price_source: "cp30_basis"
#       single_price_per_country: true     # one price per country, not per coupled market
#       cp30_basis:
#         tyndp_new_ccgt_eur: 82.0         # TYNDP NT2030's new-CCGT price plateau
#         foreign_new_ccgt_gbp: 100.62     # the same plant on CP30's fuel and EU carbon
#
# Method: scripts/interconnectors/cp30_foreign_prices.py. Data and licence:
# data/interconnectors/tyndp2024/README.md.

CP30_BASIS_TYNDP_FILE = f"{data_path}/interconnectors/tyndp2024/tyndp2024_NT2030CY2009_hourly.csv"


def is_cp30_basis_foreign_pricing(pricing_cfg):
    return str((pricing_cfg or {}).get('foreign_price_source', 'entsoe')).strip().lower() == 'cp30_basis'


def cp30_basis_price_path(scenario_config):
    year = scenario_config.get('modelled_year') or scenario_config.get('year')
    weather_year = scenario_config.get('renewables_year', year)
    return f"{resources_path}/interconnectors/cp30_basis_foreign_prices_wind{weather_year}.csv"


CP30_REFIT_ZONAL_FILE = f"{data_path}/interconnectors/tyndp2024/tyndp2024_NT2030CY2009_zonal_hourly.csv"
CP30_REFIT_CF_FILE = f"{data_path}/interconnectors/tyndp2024/renewables_ninja_2013_cf.csv"


def cp30_basis_parameter(key):
    """One value of interconnectors.pricing.cp30_basis.<key> across the scenarios that use it."""
    from scripts.interconnectors import cp30_foreign_prices as cfp
    default = {"tyndp_new_ccgt_eur": cfp.DEFAULT_TYNDP_NEW_CCGT_EUR,
               "foreign_new_ccgt_gbp": cfp.DEFAULT_FOREIGN_NEW_CCGT_GBP,
               "method": cfp.DEFAULT_METHOD,
               "target_price_cap_eur": cfp.DEFAULT_TARGET_PRICE_CAP_EUR,
               "ireland_source": cfp.DEFAULT_IRELAND_SOURCE,
               "ireland_offset_gbp": cfp.DEFAULT_IRELAND_OFFSET_GBP}[key]
    values = {
        (cfg.get('interconnectors', {}).get('pricing', {}).get('cp30_basis') or {}).get(key, default)
        for cfg in scenarios.values()
        if is_cp30_basis_foreign_pricing(cfg.get('interconnectors', {}).get('pricing', {}))
    }
    if len(values) > 1:
        raise ValueError(f"cp30_basis scenarios disagree on interconnectors.pricing.cp30_basis.{key}: {sorted(map(str, values))}")
    return values.pop() if values else default


def cp30_basis_composition_file():
    """The CP30 cable composition the cp30_basis scenarios build with (one file), else None (F25.3).

    Weights the TYNDP SEM nodes (IE00, UKNI) by CP30's 2030 cables: MaresConnect's 750 MW counts on IE00.
    """
    files = {f for cfg in scenarios.values()
             if is_cp30_basis_foreign_pricing(cfg.get('interconnectors', {}).get('pricing', {}))
             for f in get_cp30_interconnector_composition(cfg)}
    if len(files) > 1:
        raise ValueError(f"cp30_basis scenarios disagree on interconnectors.cp30_composition.file: {sorted(files)}")
    return files.pop() if files else None


def cp30_foreign_price_inputs(wildcards):
    """Inputs of build_cp30_foreign_prices for the configured method (rebase_cy2009 | refit_2013).

    The CP30 cable composition is an input only where it moves a price (F25.3): rebase_cy2009, and
    refit_2013 with ireland_source refit. Under refit_2013 + entsoe_2024 (central) the SEM column is
    2024 ENTSO-E, so the SEM node weights do not reach the file.
    """
    method = str(cp30_basis_parameter("method"))
    composition = cp30_basis_composition_file()
    if method == "refit_2013":
        inputs = {"zonal": CP30_REFIT_ZONAL_FILE, "cf": CP30_REFIT_CF_FILE}
        if str(cp30_basis_parameter("ireland_source")) == "entsoe_2024":
            inputs["ireland_shape"] = (
                f"{resources_path}/interconnectors/entsoe_price_shape_2024_wind{wildcards.weather_year}.csv"
            )
        elif composition:
            inputs["composition"] = composition
        return inputs
    if method == "rebase_cy2009":
        inputs = {"tyndp": CP30_BASIS_TYNDP_FILE,
                  "wind_weather": f"{resources_path}/renewable/profiles/wind_offshore_{wildcards.weather_year}.csv"}
        if composition:
            inputs["composition"] = composition
        return inputs
    raise ValueError(f"interconnectors.pricing.cp30_basis.method {method!r}: expected rebase_cy2009 or refit_2013")


rule build_cp30_foreign_prices:
    """Hourly foreign price per GB counterparty on CP30's fuel basis (item F13).

    TYNDP 2024 National Trends 2030 zone prices rebased by one ratio that maps
    TYNDP's new-CCGT plateau to the same plant on CP30's gas and EU carbon.
    Method refit_2013 (central since 7 Oct 2026): TYNDP's price response to
    residual load, learnt in its own climate year, refitted to the weather
    year's wind and solar; the SEM column from 2024 ENTSO-E (declared).
    Method rebase_cy2009: TYNDP's CY2009 prices laid on the weather year and
    conditioned on its GB offshore wind. add_to_network uses the file at its
    own level when interconnectors.pricing.foreign_price_source is "cp30_basis".
    """
    input:
        unpack(cp30_foreign_price_inputs)
    output:
        prices=resources_path + "/interconnectors/cp30_basis_foreign_prices_wind{weather_year}.csv",
        audit=resources_path + "/interconnectors/cp30_basis_foreign_prices_wind{weather_year}_audit.json",
    params:
        method=lambda wc: cp30_basis_parameter("method"),
        tyndp_new_ccgt_eur=lambda wc: cp30_basis_parameter("tyndp_new_ccgt_eur"),
        foreign_new_ccgt_gbp=lambda wc: cp30_basis_parameter("foreign_new_ccgt_gbp"),
        target_price_cap_eur=lambda wc: cp30_basis_parameter("target_price_cap_eur"),
        ireland_source=lambda wc: cp30_basis_parameter("ireland_source"),
        ireland_offset_gbp=lambda wc: cp30_basis_parameter("ireland_offset_gbp"),
    wildcard_constraints:
        weather_year=r"\d{4}",
    log:
        "logs/interconnectors/build_cp30_foreign_prices_wind{weather_year}.log"
    conda:
        "../envs/pypsa-gb.yaml"
    script:
        "../scripts/interconnectors/build_cp30_foreign_prices.py"
