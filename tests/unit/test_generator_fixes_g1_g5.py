"""Generator fixes G1-G5 of the 27 Sep 2026 system audit.

G1 CfD rounds netted against the named fleet; G2 MW site profiles normalised by
the capacity they were built on; G3 AR7 -> AR7a and one round per site; G4 the
ROC value on ROC-labelled biofuel; G5 placement exclusions and declared buses.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

LOG = logging.getLogger("test_generator_fixes")
SCHEDULE = pd.DataFrame({
    "carrier": ["wind_offshore"] * 3,
    "cfd_round": ["AR2", "AR3", "AR4"],
    "capacity_mw": [100.0, 200.0, 300.0],
    "source": ["s"] * 3,
})


# --------------------------------------------------------------------------- G1

def test_carrier_allocation_nets_named_capacity_and_spills_to_the_ar8_round():
    from scripts.generators.cfd_vintage_labels import allocate_carrier_rounds
    allocation = allocate_carrier_rounds(500.0, SCHEDULE, {"AR2": 50.0, "AR3": 250.0})
    assert allocation["AR2"] == pytest.approx(50.0)        # 100 awarded - 50 named
    assert "AR3" not in allocation                         # named already exceed the award
    assert allocation["AR4"] == pytest.approx(300.0)
    assert allocation["AR8_modelled"] == pytest.approx(150.0)
    assert sum(allocation.values()) == pytest.approx(500.0)


def test_carrier_allocation_scales_down_when_the_increment_is_small():
    from scripts.generators.cfd_vintage_labels import allocate_carrier_rounds
    allocation = allocate_carrier_rounds(175.0, SCHEDULE, {"AR3": 200.0})
    # 400 MW open (AR2 100, AR4 300); 175 fills 43.75% of each.
    assert allocation == pytest.approx({"AR2": 43.75, "AR4": 131.25})
    assert "AR8_modelled" not in allocation


def test_labelling_applies_carrier_fractions_to_every_row(tmp_path):
    from scripts.generators.cfd_vintage_labels import apply_cfd_vintage_labels
    path = tmp_path / "schedule.csv"
    SCHEDULE.to_csv(path, index=False)
    sites = pd.DataFrame({
        "site_name": ["a", "b"], "technology": ["wind_offshore"] * 2,
        "capacity_mw": [300.0, 200.0], "support_type": ["CfD", "CfD"],
    })
    named = pd.DataFrame({"carrier": ["wind_offshore"] * 2, "cfd_round": ["AR2", "AR3"],
                          "capacity_mw": [50.0, 250.0]})
    result = apply_cfd_vintage_labels(sites, path, LOG, named_capacity=named)
    assert result.capacity_mw.sum() == pytest.approx(500.0)
    by_round = result.groupby("cfd_round").capacity_mw.sum()
    assert by_round.to_dict() == pytest.approx({"AR2": 50.0, "AR4": 300.0, "AR8_modelled": 150.0})
    # each row carries the same fractions: row a is 60% of the increment
    a = result[result.site_name.str.startswith("a_")].set_index("cfd_round").capacity_mw
    assert a["AR4"] == pytest.approx(180.0) and a["AR8_modelled"] == pytest.approx(90.0)


def test_named_round_capacity_ignores_ar8_and_unlabelled_rows():
    from scripts.generators.cfd_vintage_labels import named_round_capacity
    rows = pd.DataFrame({
        "technology": ["wind_offshore"] * 4, "capacity_mw": [10.0, 20.0, 30.0, 40.0],
        "support_type": ["CfD", "CfD", "CfD", "ROC"], "cfd_round": ["AR3", "AR8_modelled", None, "AR3"],
    })
    netted = named_round_capacity(rows)
    assert netted.to_dict("records") == [{"carrier": "wind_offshore", "cfd_round": "AR3", "cfd_technology": "",
                                          "capacity_mw": 10.0}]


# --------------------------------------------------------------------------- G3

def test_ar7_on_onshore_and_solar_is_ar7a():
    from scripts.generators.cfd_vintage_labels import normalise_cfd_round, normalise_cfd_rounds
    assert normalise_cfd_round("AR7", "solar_pv") == "AR7a"
    assert normalise_cfd_round("AR7 ", "wind_onshore") == "AR7a"
    assert normalise_cfd_round("AR7", "wind_offshore") == "AR7"
    assert normalise_cfd_round("AR6", "solar_pv") == "AR6"
    assert pd.isna(normalise_cfd_round(np.nan, "solar_pv"))
    frame = normalise_cfd_rounds(pd.DataFrame({"technology": ["solar_pv", "wind_offshore"],
                                               "cfd_round": ["AR7", "AR7"]}))
    assert frame.cfd_round.tolist() == ["AR7a", "AR7"]


def test_lccc_portfolio_files_ar7_onshore_and_solar_as_ar7a(tmp_path):
    from scripts.renewables.support_register_labels import load_lccc_portfolio
    path = tmp_path / "lccc.csv"
    pd.DataFrame([
        ("AR7-A", "Solar A", "Allocation Round 7", "Solar PV", "Pre-MDD", 50.0),
        ("AR7-B", "Offshore B", "Allocation Round 7", "Offshore Wind", "Pre-MDD", 500.0),
    ], columns=["CFD_ID", "Name_of_CFD_Unit", "Allocation_Round", "Technology_Type", "Status",
                "Maximum_Contract_Capacity_MW"]).to_csv(path, index=False)
    portfolio = load_lccc_portfolio(path).set_index("unit_id")
    assert portfolio.at["AR7-A", "cfd_round"] == "AR7a"
    assert portfolio.at["AR7-B", "cfd_round"] == "AR7"
    assert portfolio.at["AR7-A", "award_year"] == 2025


def test_a_site_matched_to_two_rounds_takes_the_dominant_one(monkeypatch):
    import scripts.renewables.support_register_labels as labels
    sites = pd.DataFrame([{
        "site_name": "West Burton Solar Project", "source_site_name": "West Burton Solar Project",
        "technology": "solar_pv", "capacity_mw": 201.5, "operational_date": None,
        "development_status": "Awaiting Construction", "fleet_origin": "named", "repd_ref_id": "10916",
        "cfd_round": None, "cfd_capacity_mw": np.nan, "lat": 53.3, "lon": -0.8,
    }])
    portfolio = pd.DataFrame({
        "unit_id": ["AR6-BTS-700", "AR7-WBS-710"], "unit_name": ["Burton Top", "West Burton Site 3"],
        "cfd_round": ["AR6", "AR7a"], "Status": ["Pre-MDD", "Pre-MDD"],
    })

    def fake_match(units, table, overrides=None, register="lccc_cfd"):
        return pd.DataFrame({"unit_id": ["AR6-BTS-700", "AR7-WBS-710"], "site": [0, 0],
                             "allocated_mw": [8.0, 193.5], "method": ["name", "name"]})

    monkeypatch.setattr(labels, "match_units_to_sites", fake_match)
    result, _, audit = labels.apply_lccc_cfd_evidence(sites, portfolio, logger=LOG)
    assert result.at[0, "cfd_round"] == "AR7a"
    assert audit.iloc[0]["round_lccc"] == "AR6/AR7a"


# --------------------------------------------------------------------------- G2

def test_profile_column_capacities_match_the_profile_rule():
    pytest.importorskip("atlite")
    from scripts.renewables.map_renewable_profiles import RenewableProfileGenerator
    from scripts.generators.integrate_renewable_generators import profile_column_capacities
    base = pd.DataFrame({
        "site_name": ["Grange Farm", "Grange Farm", "Mona", "No Coords"],
        "capacity_mw": [5.0, 7.5, 1500.0, 3.0],
        "lat": [52.0, 53.0, 53.8, np.nan], "lon": [-1.0, -0.5, -3.7, np.nan],
    })
    pipeline = pd.DataFrame({
        "site_name": ["Mona", "New Farm", "New Farm"], "capacity_mw": [1500.0, 40.0, 49.9],
        "lat": [53.8, 52.2, 52.2], "lon": [-3.7, 0.1, 0.1],
    })
    processed = pd.DataFrame({"name": ["Big Offshore"], "max capacity (GW)": [1.2],
                              "lat": [54.0], "lon": [1.5]})
    generator = RenewableProfileGenerator()
    combined = generator.prepare_wind_sites(generator.combine_with_pipeline(base, processed, pipeline))
    expected = {}
    for name, capacity in zip(combined.site_name, combined.capacity_mw):
        if capacity > 0:
            expected[name] = float(capacity)          # a later duplicate overwrites, as in the rule
    assert profile_column_capacities(base, processed, pipeline) == pytest.approx(expected)
    assert expected["Grange Farm"] == 7.5
    assert "Mona (Pipeline)" not in expected      # an operational name shadows the pipeline row


def _profile_network(tmp_path, capacity_mw):
    from scripts.generators.integrate_renewable_generators import add_renewable_generators
    snapshots = pd.date_range("2030-01-01", periods=4, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "B1", x=0.0, y=52.0, v_nom=132.0)
    cf = pd.Series([0.1, 0.5, 0.9, 0.3], index=pd.date_range("2013-01-01", periods=4, freq="h"))
    profiles = tmp_path / "profiles"
    profiles.mkdir(exist_ok=True)
    pd.DataFrame({"Site A": cf * 100.0}).to_csv(profiles / "wind_onshore_2013.csv")
    sites = pd.DataFrame({"site_name": ["Site A"], "technology": ["wind_onshore"],
                          "capacity_mw": [capacity_mw], "bus": ["B1"], "lat": [52.0], "lon": [0.0]})
    return network, sites, profiles, cf, add_renewable_generators


def test_a_trimmed_unit_keeps_its_site_capacity_factor(tmp_path):
    network, sites, profiles, cf, add = _profile_network(tmp_path, 60.0)
    add(network, sites, profiles_dir=str(profiles), renewables_year=2013,
        profile_capacities={"wind_onshore": {"Site A": 100.0}})
    np.testing.assert_allclose(network.generators_t.p_max_pu["Site_A"].to_numpy(), cf.to_numpy())


def test_without_a_column_capacity_the_p_nom_divisor_is_kept(tmp_path):
    network, sites, profiles, cf, add = _profile_network(tmp_path, 60.0)
    add(network, sites, profiles_dir=str(profiles), renewables_year=2013)
    np.testing.assert_allclose(network.generators_t.p_max_pu["Site_A"].to_numpy(),
                               np.clip(cf.to_numpy() * 100.0 / 60.0, 0, 1))


# --------------------------------------------------------------------------- G4

def _biomass_network():
    network = pypsa.Network()
    network.add("Bus", "bus")
    for name, support, banding in (("roc", "ROC", 1.5461), ("merchant", "merchant", np.nan),
                                   ("roc_no_band", "ROC", np.nan)):
        network.add("Generator", name, bus="bus", carrier="biomass", p_nom=10.0, efficiency=0.35,
                    support_type=support, ro_banding=banding)
    return network


def _biomass_config(follow=None):
    biofuel = {"enabled": False, "mc_floor": 5.0, "default_roc_bandings": {"biomass": 1.5}}
    if follow is not None:
        biofuel["roc_labelled_follow_subsidies"] = follow
    return {"modelled_year": 2030,
            "support": {"dispatch_costs": {"enabled": True, "roc_buyout_prices": {2030: 78.04}}},
            "marginal_costs": {"biofuel_subsidy": biofuel}}


def test_roc_biomass_takes_the_roc_value_with_the_subsidy_layer_on():
    from scripts.generators.apply_marginal_costs import apply_marginal_costs_to_network
    result = apply_marginal_costs_to_network(
        _biomass_network(), fuel_prices={"biomass": 45.0}, carbon_price=0.0,
        scenario_config=_biomass_config(),
    ).set_index("generator")
    formula = 45.0 / 0.35
    assert result.at["merchant", "marginal_cost"] == pytest.approx(formula)
    assert result.at["roc", "marginal_cost"] == pytest.approx(formula - 1.5461 * 78.04)
    assert "_ROC(" in result.at["roc", "source"]
    # no banding on the unit: the carrier default applies
    assert result.at["roc_no_band", "marginal_cost"] == pytest.approx(formula - 1.5 * 78.04)


def test_roc_biomass_value_is_floored_and_can_be_switched_off():
    from scripts.generators.apply_marginal_costs import apply_marginal_costs_to_network
    cheap = apply_marginal_costs_to_network(
        _biomass_network(), fuel_prices={"biomass": 20.0}, carbon_price=0.0,
        scenario_config=_biomass_config(),
    ).set_index("generator")
    assert cheap.at["roc", "marginal_cost"] == pytest.approx(5.0)
    off = apply_marginal_costs_to_network(
        _biomass_network(), fuel_prices={"biomass": 45.0}, carbon_price=0.0,
        scenario_config=_biomass_config(follow=False),
    ).set_index("generator")
    assert off.at["roc", "marginal_cost"] == pytest.approx(45.0 / 0.35)


# --------------------------------------------------------------------------- G5

def _exclusion_network(tmp_path):
    """Grid A (load) - export substation EXPO - offshore platform OFFW; a DC hub;
    site COST with one bus 300 km from its two siblings."""
    import geopandas as gpd
    from shapely.geometry import box
    land = tmp_path / "land.geojson"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(300000, 300000, 500000, 700000)],
                     crs=27700).to_file(land, driver="GeoJSON")
    network = pypsa.Network()
    for name, x, y in (("GRDA41", 400000, 400000), ("GRDB41", 420000, 420000),
                       ("EXPO41", 401000, 401000), ("EXPO21", 401000, 401000),
                       ("OFFW21", 600000, 400000), ("HUB", 450000, 450000),
                       ("COST1A", 350000, 650000), ("COST3-", 350000, 650000), ("COST1-", 350000, 350000)):
        network.add("Bus", name, x=x, y=y, v_nom=275.0)
    network.add("Line", "l1", bus0="GRDA41", bus1="GRDB41", s_nom=2000, x=0.1)
    network.add("Line", "l2", bus0="GRDA41", bus1="EXPO41", s_nom=300, x=0.1)
    network.add("Transformer", "t1", bus0="EXPO41", bus1="EXPO21", s_nom=300, x=0.1)
    network.add("Line", "l3", bus0="EXPO21", bus1="OFFW21", s_nom=300, x=0.1)
    network.add("Line", "l4", bus0="COST1A", bus1="COST1-", s_nom=100, x=0.1)
    network.add("Transformer", "t2", bus0="COST1-", bus1="COST3-", s_nom=45, x=0.1)
    network.add("Line", "l5", bus0="COST1A", bus1="GRDB41", s_nom=100, x=0.1)
    network.add("Link", "k1", bus0="HUB", bus1="GRDB41", p_nom=800)
    network.add("Load", "d", bus="GRDA41", p_set=100)
    network.add("Load", "d2", bus="COST1A", p_set=10)
    return network, str(land)


def test_placement_exclusions_find_hubs_platforms_export_substations_and_misplaced_buses(tmp_path):
    from scripts.utilities.spatial_utils import generation_placement_exclusions
    network, land = _exclusion_network(tmp_path)
    reasons = generation_placement_exclusions(network, land_boundary_file=land, land_buffer_m=0.0)
    assert reasons.to_dict() == {
        "COST1-": "coordinate_outlier", "EXPO21": "offshore_export", "EXPO41": "offshore_export",
        "HUB": "dc_only", "OFFW21": "offshore",
    }


def test_rows_on_excluded_buses_move_to_the_nearest_allowed_bus(tmp_path):
    from scripts.utilities.spatial_utils import (
        generation_placement_exclusions, osgb36_to_wgs84, reassign_excluded_placements,
    )
    network, land = _exclusion_network(tmp_path)
    reasons = generation_placement_exclusions(network, land_boundary_file=land, land_buffer_m=0.0)
    lon, lat = osgb36_to_wgs84(401500, 401500)
    sites = pd.DataFrame({
        "site_name": ["onshore farm", "anchored", "offshore farm"],
        "technology": ["wind_onshore", "wind_onshore", "wind_offshore"],
        "bus": ["EXPO21", "EXPO41", "EXPO41"], "lon": [lon] * 3, "lat": [lat] * 3,
        "data_source": ["REPD_named_fleet", "CP30_connection_allocation", "REPD_named_fleet"],
    })
    moved, log = reassign_excluded_placements(sites, network, reasons,
                                              carriers=["wind_onshore", "solar_pv"])
    assert moved.bus.tolist() == ["GRDA41", "EXPO41", "EXPO41"]
    assert len(log) == 1


def test_documented_connections_for_crystal_rig_and_arecleoch():
    from scripts.utilities.spatial_utils import _explicit_onshore_connection
    index = pd.Index(["CRYR1-", "AREC1-", "NNGO21"])
    assert _explicit_onshore_connection("Crystal Rig Extension II", index) == ("crystal rig", "CRYR1-")
    assert _explicit_onshore_connection("Crystal_Rig_Wind_Farm_Phase_2a", index)[1] == "CRYR1-"
    assert _explicit_onshore_connection("Arecleoch Windfarm", index) == ("arecleoch", "AREC1-")


def test_peterborough_is_declared_at_walpole_132kv():
    schedule = pd.read_csv("data/market/CP30/gas_station_schedule_2030.csv")
    rows = schedule[schedule.station_name.str.startswith("Peterborough Power Station")]
    assert len(rows) == 2 and rows.bus.eq("WALP11").all()
    assert rows.bus_source.str.contains("TEC register").all()


def test_island_sites_go_to_the_island_hub_and_mainland_sites_leave_it(tmp_path):
    from scripts.utilities.spatial_utils import apply_island_networks
    network = pypsa.Network()
    for name, x, y in (("FINS1-", 300000, 960000), ("THUS1J", 311000, 968000), ("HUB", 360000, 1000000)):
        network.add("Bus", name, x=x, y=y, v_nom=132.0)
    network.add("Line", "l", bus0="FINS1-", bus1="THUS1J", s_nom=200, x=0.1)
    sites = pd.DataFrame({
        "site_name": ["orkney farm", "caithness farm", "orkney anchored"],
        "technology": ["wind_onshore"] * 3,
        "bus": ["HUB", "FINS1-", "HUB"],
        "lon": [-3.1, -3.74, -3.0], "lat": [59.0, 58.53, 59.1],
        "data_source": ["REPD_named_fleet", "REPD_named_fleet", "CP30_connection_allocation"],
    })
    moved, log = apply_island_networks(sites, network, carriers=["wind_onshore"], exclusions={"HUB"})
    assert moved.bus.tolist() == ["FINS1-", "THUS1J", "HUB"]
    assert set(log["rule"]) == {"Orkney_hub", "off_Orkney"}


def test_a_reviewed_override_replaces_the_repd_round(monkeypatch):
    import scripts.renewables.support_register_labels as labels
    sites = pd.DataFrame([{
        "site_name": "Hornsea 3", "source_site_name": "Hornsea 3", "technology": "wind_offshore",
        "capacity_mw": 2955.0, "operational_date": None, "development_status": "Awaiting Construction",
        "fleet_origin": "named", "repd_ref_id": "2472", "cfd_round": "AR6", "cfd_capacity_mw": 1080.0,
        "lat": 53.7, "lon": 1.4,
    }])
    portfolio = pd.DataFrame({"unit_id": ["AR4-HRN-010", "AR6-HPT-010"], "unit_name": ["H3", "H3 A"],
                              "cfd_round": ["AR4", "AR6"], "Status": ["Pre-Start Date", "Pre-MDD"]})

    def fake_match(units, table, overrides=None, register="lccc_cfd"):
        return pd.DataFrame({"unit_id": ["AR4-HRN-010", "AR6-HPT-010"], "site": [0, 0],
                             "allocated_mw": [2146.1, 360.0], "method": ["explicit_override"] * 2})

    monkeypatch.setattr(labels, "match_units_to_sites", fake_match)
    result, _, audit = labels.apply_lccc_cfd_evidence(sites, portfolio, logger=LOG)
    assert result.at[0, "cfd_round"] == "AR4"
    assert result.at[0, "cfd_capacity_mw"] == pytest.approx(2506.1)
    assert audit.iloc[0]["note"].startswith("reviewed override")


def test_a_support_tranche_finds_its_profile_through_the_source_name(tmp_path):
    network, sites, profiles, cf, add = _profile_network(tmp_path, 30.0)
    sites = sites.assign(site_name="Site A_CfD", source_site_name="Site A")
    add(network, sites, profiles_dir=str(profiles), renewables_year=2013,
        profile_capacities={"wind_onshore": {"Site A": 100.0}})
    np.testing.assert_allclose(network.generators_t.p_max_pu["Site_A_CfD"].to_numpy(), cf.to_numpy())


def test_fleet_average_profile_is_normalised_by_capacity_not_peak(tmp_path):
    """Unmatched (proxy) units take the fleet average on the same basis as named units."""
    from scripts.generators.integrate_renewable_generators import add_renewable_generators
    snapshots = pd.date_range("2030-06-01", periods=4, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "B1", x=0.0, y=52.0, v_nom=132.0)
    index = pd.date_range("2013-06-01", periods=4, freq="h")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    # two sites of 100 MW whose output never reaches rated capacity (solar-like)
    pd.DataFrame({"Farm A": [0.0, 40.0, 80.0, 20.0], "Farm B": [0.0, 30.0, 60.0, 10.0]},
                 index=index).to_csv(profiles / "solar_pv_2013.csv")
    sites = pd.DataFrame({"site_name": ["CP30_increment_solar_pv_1"], "technology": ["solar_pv"],
                          "capacity_mw": [50.0], "bus": ["B1"], "lat": [52.0], "lon": [0.0]})
    add_renewable_generators(network, sites, profiles_dir=str(profiles), renewables_year=2013,
                             profile_capacities={"solar_pv": {"Farm A": 100.0, "Farm B": 100.0}})
    expected = np.array([0.0, 70.0, 140.0, 30.0]) / 200.0          # not / (80 + 60)
    np.testing.assert_allclose(network.generators_t.p_max_pu["CP30_increment_solar_pv_1"].to_numpy(), expected)


def test_profile_capacity_map_covers_marine_and_hydro(tmp_path):
    from scripts.generators.integrate_renewable_generators import build_profile_capacity_map

    class Inputs:
        pass

    inputs = Inputs()
    path = tmp_path / "tidal.csv"
    pd.DataFrame({"site_name": ["MeyGen", "EMEC"], "capacity_mw": [6.0, 2.0]}).to_csv(path, index=False)
    inputs.tidal_stream_sites = str(path)
    result = build_profile_capacity_map(inputs, LOG)
    assert result["tidal_stream"] == {"MeyGen": 6.0, "EMEC": 2.0}
    assert result["marine"] == result["tidal_stream"]


# --------------------------------------------------------------------------- Scottish 132 kV export relief

def _pocket_network():
    """Scottish 132 kV pocket: P132 -(112 MVA)- Q132 -(240 MVA trafo)- G400; spare 132 kV bus S132
    on its own 300 MVA transformer; E&W 132 kV bus E132 (relaxed) far south."""
    network = pypsa.Network()
    for name, x, y, v in (("P132", 380000, 780000, 132), ("Q132", 385000, 780000, 132),
                          ("G400", 386000, 780000, 400), ("S132", 390000, 785000, 132),
                          ("E132", 450000, 300000, 132), ("E400", 451000, 300000, 400)):
        network.add("Bus", name, x=x, y=y, v_nom=v)
    network.add("Line", "pq", bus0="P132", bus1="Q132", s_nom=112, x=0.1)
    network.add("Transformer", "qg", bus0="G400", bus1="Q132", s_nom=240, x=0.1)
    network.add("Transformer", "sg", bus0="G400", bus1="S132", s_nom=300, x=0.1)
    network.add("Transformer", "ee", bus0="E400", bus1="E132", s_nom=50, x=0.1)
    network.lines["etys_licensee"] = "SHE"
    network.transformers["etys_licensee"] = ["SHE", "SHE", "NGET"]
    return network


def test_bm_capacities_keep_scottish_132kv_and_relax_english():
    from scripts.utilities.spatial_utils import bm_branch_capacities, _INF_CAPACITY
    caps = bm_branch_capacities(_pocket_network(), {"min_bm_constraint_voltage_kv": 220,
                                                    "bm_voltage_floor": {"scottish_transmission_constrained": True,
                                                                         "scottish_min_constrained_kv": 132}})
    assert caps.loc["pq", "capacity"] == 112 and caps.loc["qg", "capacity"] == 240
    assert caps.loc["ee", "capacity"] == _INF_CAPACITY


def test_export_relief_moves_proxies_out_of_a_constrained_pocket_and_keeps_named():
    from scripts.utilities.spatial_utils import relieve_export_constrained_proxies
    network = _pocket_network()
    sites = pd.DataFrame({
        "site_name": ["named farm", "proxy big", "proxy small", "english proxy"],
        "technology": ["wind_onshore"] * 4, "capacity_mw": [100.0, 120.0, 60.0, 400.0],
        "bus": ["P132", "P132", "Q132", "E132"],
        "fleet_origin": ["named", "increment", "increment", "increment"],
        "data_source": ["REPD_named_fleet", "CP30", "CP30", "CP30"],
    })
    moved, log, pockets = relieve_export_constrained_proxies(
        sites, network, {"min_bm_constraint_voltage_kv": 220}, max_local_loading=10.0)
    buses = moved.set_index("site_name").bus
    assert buses["named farm"] == "P132"                 # named stays at its connection point
    assert buses["proxy big"] != "P132"                  # 220 MW behind 112 MVA: proxy out
    assert buses["english proxy"] == "E132"              # E&W sub-220 kV is relaxed in the BM
    before = pockets[pockets.state.eq("before")]
    assert before.deficit_mw.sum() == pytest.approx(108.0)
    assert pockets[pockets.state.eq("after")].empty
    assert set(log.to_bus) <= {"G400", "S132"}


def test_export_relief_leaves_a_pocket_of_named_farms_alone():
    from scripts.utilities.spatial_utils import relieve_export_constrained_proxies
    network = _pocket_network()
    sites = pd.DataFrame({"site_name": ["a", "b"], "technology": ["wind_onshore"] * 2,
                          "capacity_mw": [100.0, 80.0], "bus": ["P132", "P132"],
                          "fleet_origin": ["named", "named"], "data_source": ["REPD_named_fleet"] * 2})
    moved, log, pockets = relieve_export_constrained_proxies(sites, network, {"min_bm_constraint_voltage_kv": 220})
    assert moved.bus.tolist() == ["P132", "P132"] and log.empty
    after = pockets[pockets.state.eq("after")]
    assert after.deficit_mw.sum() == pytest.approx(68.0)


def test_scottish_132kv_connection_points_from_tec_and_bmu():
    from scripts.utilities.spatial_utils import _explicit_onshore_connection
    index = pd.Index(["HLND1-", "BLLX1-", "FAAR1R", "MAYB1-", "LINM1R", "MILW3-", "SHIN1-", "ACHE1-"])
    for name, bus in (("Heathland_Wind_Farm", "HLND1-"), ("Black_Law_Extension___1a", "BLLX1-"),
                      ("Farr_Wind_Farm", "FAAR1R"), ("Assel_Valley_Wind_Farm", "MAYB1-"),
                      ("Kirk_Hill_Wind_Farm", "MAYB1-"), ("Auchrobert_Wind_Farm", "LINM1R"),
                      ("Millennium_Windfarm", "MILW3-"), ("Millennium_Extension_II", "MILW3-"),
                      ("Strath_Oykel_Wind_Farm", "SHIN1-"), ("Meall_Buidhe_Wind_Farm", "SHIN1-"),
                      ("Achany_Extension_Wind_Farm_CfD", "ACHE1-")):
        assert _explicit_onshore_connection(name, index)[1] == bus, name
    assert _explicit_onshore_connection("Black_Law", index) is None
