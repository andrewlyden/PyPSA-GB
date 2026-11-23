"""Offshore placement fixes for the CP30 2030 build (15 Oct 2026).

1. Not-built sites (the checked CfD table's ``not_built_refill`` rows) refill as
   anonymous proxy rows at the excluded project's own landing bus, counted
   against the ES5 residual so the pro-rata proxy does not refill them twice
   (the brownfield build's behaviour since the 15 Oct 2026 consolidation).
2. The Fife proxy LEVT1Q, which carries the South Scotland (SPT) offshore
   residual (Berwick Bank), lands at Branxton (BRNX4-), not Blackhillock.
3. Offshore proxies take the CP30 region of their landing bus, not of their FES
   source bus.
"""
from pathlib import Path
from types import SimpleNamespace

import logging

import pandas as pd
import pytest
import yaml

import scripts.generators.integrate_renewable_generators as irg
from scripts.CP30.brownfield_fleet import (
    MATURITY_RANK, build_fleet_audit, build_landing_refill, reconcile_with_es5,
)
from scripts.CP30.connection_allocation import landing_buses_for_regions
from scripts.CP30.spatial_allocation import apply_cp30_es5_regional_allocation
from scripts.generators.proxy_profiles import proxy_profile_settings
from config.config_loader import load_scenario_entries, scenario_files
from config.config_loader import defaults_files, load_defaults, load_layer

ROOT = Path(__file__).resolve().parents[2]
LANDING_MAP = ROOT / "data" / "network" / "ETYS" / "cp30_offshore_landing_point_profile_preserving.csv"
LOG = logging.getLogger("test_cp30_offshore_placement_fixes")


def _targets(rows):
    return pd.DataFrame(rows, columns=["carrier", "cp30_region", "capacity_mw"])


def _named(rows):
    frame = pd.DataFrame(rows)
    frame["maturity_rank"] = frame["development_status"].map(MATURITY_RANK)
    frame["fleet_origin"] = "named"
    for column in ("cfd_round", "ro_banding", "planning_granted_date", "repd_ref_id"):
        if column not in frame.columns:
            frame[column] = None
    return frame


def _placed_excluded():
    """Excluded not-built projects after the named fleet's placement step."""
    return pd.DataFrame([
        {"site_name": "Hornsea 4", "repd_ref_id": "2473", "technology": "wind_offshore", "capacity_mw": 2600.0,
         "bus": "CREB41", "cp30_region": "North England", "lat": 54.0, "lon": 1.5, "cfd_round": "AR6",
         "development_status": "Awaiting Construction", "landing_evidence": "evidence_register"},
        {"site_name": "Norfolk Boreas", "repd_ref_id": "6483", "technology": "wind_offshore", "capacity_mw": 1400.0,
         "bus": "NECT41", "cp30_region": "East England", "lat": 53.0, "lon": 2.0, "cfd_round": "AR4",
         "development_status": "Awaiting Construction", "landing_evidence": "tec_register"},
        {"site_name": "Solar A", "repd_ref_id": "1", "technology": "solar_pv", "capacity_mw": 30.0,
         "bus": "SOL1", "cp30_region": "Midlands", "lat": 52.0, "lon": -1.0},
        {"site_name": "Solar B", "repd_ref_id": "2", "technology": "solar_pv", "capacity_mw": 10.0,
         "bus": "SOL1", "cp30_region": "Midlands", "lat": 52.4, "lon": -1.4},
    ])


# --------------------------------------------------------------------------
# Fix 1: refill at the excluded project's landing
# --------------------------------------------------------------------------

def test_refill_rows_are_anonymous_proxy_rows_at_the_named_landing_bus():
    refill, detail = build_landing_refill(_placed_excluded(), 2030, LOG)
    off = refill[refill["technology"].eq("wind_offshore")].set_index("bus")
    assert off.loc["CREB41", "capacity_mw"] == 2600.0
    assert off.loc["NECT41", "capacity_mw"] == 1400.0
    assert off.loc["CREB41", "cp30_region"] == "North England"
    # No name, REPD ID or support label survives; the row is proxy increment.
    assert not {"repd_ref_id", "cfd_round", "development_status", "source_site_name"} & set(refill.columns)
    assert refill["site_name"].str.startswith("CP30_increment_").all()
    assert not refill["site_name"].str.contains("Hornsea|Boreas|Solar A", regex=True).any()
    assert refill["fleet_origin"].eq("increment").all()
    # Offshore keeps its evidenced landing (protected from BMU heuristics); onshore stays movable.
    assert off["data_source"].eq("CP30_connection_allocation").all()
    assert refill.loc[refill["technology"].eq("solar_pv"), "data_source"].eq("CP30_refill_at_excluded_site").all()
    # Two sites at one bus become one row, at their capacity-weighted coordinates.
    solar = refill[refill["technology"].eq("solar_pv")]
    assert len(solar) == 1 and solar["capacity_mw"].iloc[0] == 40.0
    assert solar["lat"].iloc[0] == pytest.approx((52.0 * 30 + 52.4 * 10) / 40)
    assert set(detail["refill"]) == {"at_excluded_landing"}


def test_excluded_site_with_no_bus_is_left_to_the_regional_proxy():
    placed = _placed_excluded()
    placed.loc[placed["site_name"].eq("Norfolk Boreas"), "bus"] = None
    refill, detail = build_landing_refill(placed, 2030, LOG)
    assert "NECT41" not in set(refill["bus"])
    assert detail.set_index("site_name").at["Norfolk Boreas", "refill"] == "regional_proxy"


def test_refill_counts_against_the_residual_so_it_is_not_refilled_twice():
    named = _named([
        {"site_name": "Hornsea 2", "technology": "wind_offshore", "cp30_region": "North England",
         "capacity_mw": 1320.0, "development_status": "Operational"},
        {"site_name": "EA1", "technology": "wind_offshore", "cp30_region": "East England",
         "capacity_mw": 714.0, "development_status": "Operational"},
    ])
    targets = _targets([("wind_offshore", "North England", 10000.0), ("wind_offshore", "East England", 5000.0),
                        ("solar_pv", "Midlands", 1000.0)])
    refill, _ = build_landing_refill(_placed_excluded(), 2030, LOG)

    _, residual_off, _, _ = reconcile_with_es5(named, targets, logger=LOG)
    kept, residual_on, _, audit = reconcile_with_es5(named, targets, logger=LOG, refill=refill)
    off = residual_off.set_index(["carrier", "cp30_region"])["capacity_mw"]
    on = residual_on.set_index(["carrier", "cp30_region"])["capacity_mw"]
    # The pro-rata residual shrinks by exactly the refill placed at landing.
    assert on[("wind_offshore", "North England")] == pytest.approx(off[("wind_offshore", "North England")] - 2600.0)
    assert on[("wind_offshore", "East England")] == pytest.approx(off[("wind_offshore", "East England")] - 1400.0)
    assert on[("solar_pv", "Midlands")] == pytest.approx(off[("solar_pv", "Midlands")] - 40.0)
    # ES5 is met exactly: named + refill + proxy residual = target, in every region.
    built = kept.groupby(["technology", "cp30_region"])["capacity_mw"].sum()
    for key, target in targets.set_index(["carrier", "cp30_region"])["capacity_mw"].items():
        assert built.get(key, 0.0) + on[key] == pytest.approx(target)
    assert kept.loc[kept["fleet_origin"].eq("increment"), "capacity_mw"].sum() == pytest.approx(4040.0)
    row = audit.set_index(["carrier", "cp30_region"]).loc[("wind_offshore", "North England")]
    assert row["refill_at_landing_mw"] == 2600.0
    assert row["named_share_of_target"] == pytest.approx(0.132)   # named only, not the refill


def test_refill_is_capped_at_the_residual_the_pro_rata_refill_would_have_had():
    named = _named([
        {"site_name": "big", "technology": "wind_offshore", "cp30_region": "North England",
         "capacity_mw": 9000.0, "development_status": "Operational"},
    ])
    targets = _targets([("wind_offshore", "North England", 10000.0)])
    refill, _ = build_landing_refill(_placed_excluded().iloc[[0]], 2030, LOG)
    kept, residual, _, _ = reconcile_with_es5(named, targets, logger=LOG, refill=refill)
    assert kept.loc[kept["fleet_origin"].eq("increment"), "capacity_mw"].sum() == pytest.approx(1000.0)
    assert residual["capacity_mw"].sum() == pytest.approx(0.0)
    assert kept["capacity_mw"].sum() == pytest.approx(10000.0)


def test_refill_never_trims_a_named_project():
    named = _named([
        {"site_name": "pipeline", "technology": "wind_offshore", "cp30_region": "North England",
         "capacity_mw": 9500.0, "development_status": "Awaiting Construction"},
    ])
    targets = _targets([("wind_offshore", "North England", 10000.0)])
    refill, _ = build_landing_refill(_placed_excluded().iloc[[0]], 2030, LOG)
    kept, _, trimmed, _ = reconcile_with_es5(named, targets, logger=LOG, refill=refill)
    assert trimmed.empty
    assert kept.loc[kept["site_name"].eq("pipeline"), "capacity_mw"].iloc[0] == 9500.0


def test_audit_reports_refill_at_landing_separately_from_named():
    refill, _ = build_landing_refill(_placed_excluded(), 2030, LOG)
    named = _named([
        {"site_name": "Hornsea 2", "technology": "wind_offshore", "cp30_region": "North England",
         "capacity_mw": 1320.0, "development_status": "Operational", "bus": "KILL41"},
    ])
    audit = build_fleet_audit(pd.concat([named, refill], ignore_index=True), pd.DataFrame(),
                              pd.DataFrame(), pd.DataFrame())
    entry = audit["carriers"]["wind_offshore"]
    assert entry["named_mw"] == 1320.0
    assert entry["refill_at_landing_mw"] == 4000.0
    assert entry["refill_at_landing_by_bus_mw"] == {"CREB41": 2600.0, "NECT41": 1400.0}


# The brownfield assembly, end to end on toy inputs (placement and regions stubbed).

_REGION = {"CREB41": "North England", "KILL41": "North England", "NECT41": "East England",
           "ONS1": "South Scotland", "LEW1": "North Scotland"}


def _toy_fleet():
    return _named([
        {"site_name": "Hornsea 4", "repd_ref_id": "2473", "technology": "wind_offshore", "capacity_mw": 2600.0,
         "development_status": "Awaiting Construction", "lat": 54.0, "lon": 1.5, "_bus": "HOWW2A",
         "cfd_round": "AR6"},
        {"site_name": "Hornsea 2", "repd_ref_id": "2502", "technology": "wind_offshore", "capacity_mw": 1320.0,
         "development_status": "Operational", "lat": 53.9, "lon": 1.7, "_bus": "KILL41"},
        {"site_name": "Chirmorie Wind Farm", "repd_ref_id": "2767", "technology": "wind_onshore",
         "capacity_mw": 80.0, "development_status": "Awaiting Construction", "lat": 55.1, "lon": -4.8,
         "_bus": "ONS1"},
        # A declared exclusion (the Lewis farms' kind): its connection does not exist, so no landing refill.
        {"site_name": "Lewis", "repd_ref_id": "999", "technology": "wind_onshore", "capacity_mw": 100.0,
         "development_status": "Awaiting Construction", "lat": 58.2, "lon": -6.4, "_bus": "LEW1"},
    ])


@pytest.fixture
def toy_build(tmp_path, monkeypatch):
    crosswalk = tmp_path / "crosswalk.csv"
    pd.DataFrame([{"repd_ref_id": "2473", "landing_bus": "CREB41", "evidence": "tec_register",
                   "source": "TEC register"}]).to_csv(crosswalk, index=False)
    exclusions = tmp_path / "exclusions.csv"
    pd.DataFrame([{"repd_ref_id": "999", "site_name": "Lewis", "first_modelled_year": 2031,
                   "reason": "link after 2030", "evidence": "TEC"}]).to_csv(exclusions, index=False)
    checked = pd.DataFrame([
        {"repd_ref_id": "2473", "site_name": "Hornsea 4", "first_modelled_year": 2031, "reason": "terminated",
         "evidence": "LCCC Terminated"},
        {"repd_ref_id": "2767", "site_name": "Chirmorie Wind Farm", "first_modelled_year": 2031,
         "reason": "terminated", "evidence": "LCCC Terminated"},
    ])
    import scripts.renewables.support_register_labels as labels
    monkeypatch.setattr(labels, "load_checked_cfd_contracts", lambda path: "table")
    monkeypatch.setattr(labels, "checked_cfd_exclusions", lambda table: checked)
    monkeypatch.setattr(irg, "load_named_fleet", lambda *a, **k: (_toy_fleet(), pd.DataFrame()))

    def _map(network, sites, **kwargs):
        out = sites.copy()
        out["bus"] = out["_bus"]
        return out

    def _regions(sites, network, anchors, config=None):
        out = sites.copy()
        out["cp30_region"] = out["bus"].map(_REGION).fillna("East England")
        return out

    class _NoGspNodes:  # the GSP-node placement is tested on its own below
        def __init__(self, *args, **kwargs):
            pass

        def assign(self, sites):
            return sites.assign(gsp_node=None, gsp_group=None, gsp_ids=None)

    monkeypatch.setattr(irg, "map_sites_to_buses", _map)
    monkeypatch.setattr(irg, "assign_cp30_regions", _regions)
    monkeypatch.setattr(irg, "GspNodeLocator", _NoGspNodes)
    monkeypatch.setattr(irg, "read_es5_targets", lambda *a: pd.DataFrame([
        {"carrier": "wind_offshore", "cp30_region": "North England", "capacity_mw": 10000.0},
        {"carrier": "wind_offshore", "cp30_region": "East England", "capacity_mw": 5000.0},
        {"carrier": "wind_onshore", "cp30_region": "South Scotland", "capacity_mw": 3000.0},
        {"carrier": "wind_onshore", "cp30_region": "North Scotland", "capacity_mw": 2000.0},
    ]))
    buses = pd.DataFrame({"v_nom": 400.0, "lon": 0.0, "lat": 54.0},
                         index=["CREB41", "KILL41", "NECT41", "ONS1", "LEW1", "HOWW2A"])
    network = SimpleNamespace(buses=buses)

    def run():
        brownfield = {"named_exclusions_file": str(exclusions), "offshore_landing_crosswalk": str(crosswalk),
                      "_checked_cfd_contracts": "checked.csv"}
        return irg._build_brownfield_named_fleet(
            brownfield, {}, "es5.csv", "FFR", 2030, network, {"enabled": True}, LOG)
    return run


def _residual(targets):
    return targets.set_index(["carrier", "cp30_region"])["capacity_mw"]


def test_brownfield_refills_not_built_offshore_at_its_landing(toy_build):
    named, residual, state = toy_build()
    refill = named[named["fleet_origin"].eq("increment")]
    assert set(zip(refill["technology"], refill["bus"], refill["capacity_mw"])) == {
        ("wind_offshore", "CREB41", 2600.0), ("wind_onshore", "ONS1", 80.0)}
    assert refill["repd_ref_id"].isna().all()
    assert not refill["site_name"].str.contains("Hornsea|Chirmorie").any()
    assert "_refill_candidate" not in named.columns
    r = _residual(residual)
    assert r[("wind_offshore", "North England")] == pytest.approx(10000.0 - 1320.0 - 2600.0)
    assert r[("wind_onshore", "South Scotland")] == pytest.approx(3000.0 - 80.0)
    # The declared exclusion is not refilled at its own (non-existent) connection.
    assert "LEW1" not in set(named["bus"])
    assert r[("wind_onshore", "North Scotland")] == pytest.approx(2000.0)
    detail = state["refill_detail"].set_index("site_name")
    assert detail.at["Hornsea 4", "bus"] == "CREB41"
    assert "Lewis" not in detail.index


def test_regional_totals_still_equal_es5(toy_build):
    """Named + refill + proxy residual = the ES5 target, region by region (as the pro-rata refill gave)."""
    named, residual, _ = toy_build()
    built = named.groupby(["technology", "cp30_region"])["capacity_mw"].sum()
    totals = _residual(residual).add(built.rename_axis(["carrier", "cp30_region"]), fill_value=0.0)
    assert totals.to_dict() == pytest.approx({
        ("wind_offshore", "North England"): 10000.0, ("wind_offshore", "East England"): 5000.0,
        ("wind_onshore", "South Scotland"): 3000.0, ("wind_onshore", "North Scotland"): 2000.0})


# --------------------------------------------------------------------------
# Fix 2: the South Scotland proxy lands at Branxton
# --------------------------------------------------------------------------

def test_fife_proxy_lands_at_branxton_south_of_b4_with_its_evidence():
    landing = pd.read_csv(LANDING_MAP)
    assert not landing.duplicated(["carrier", "source_bus"]).any()
    row = landing.set_index("source_bus").loc["LEVT1Q"]
    assert row["landing_bus"] == "BRNX4-"
    for evidence in ("TEC register", "Branxton 400/132kV", "Berwick Bank", "AR7"):
        assert evidence in row["source"]
    # Only the Moray Firth proxies (north of B1a, as Blackhillock is) still land at BLHI4-.
    assert set(landing.loc[landing["landing_bus"].eq("BLHI4-"), "source_bus"]) == {"BEIW21", "MOWE21", "MOWW63"}


# --------------------------------------------------------------------------
# Fix 3: offshore proxies take the region of their landing bus
# --------------------------------------------------------------------------

def _region_network():
    buses = pd.DataFrame({
        "x": [0.0, 100.0, 10.0, 90.0], "y": [0.0, 0.0, 0.0, 0.0], "v_nom": 400.0,
    }, index=["EAST_ANCHOR", "NORTH_ANCHOR", "PLATFORM", "LANDING"])
    return SimpleNamespace(buses=buses)


@pytest.fixture
def region_inputs(tmp_path):
    anchors = tmp_path / "anchors.csv"
    pd.DataFrame([{"cp30_region": "East England", "bus": "EAST_ANCHOR", "source": "t"},
                  {"cp30_region": "North England", "bus": "NORTH_ANCHOR", "source": "t"}]).to_csv(anchors, index=False)
    landing = tmp_path / "landing.csv"
    pd.DataFrame([{"carrier": "wind_offshore", "source_bus": "PLATFORM", "landing_bus": "LANDING", "source": "t"},
                  {"carrier": "wind_offshore", "source_bus": "EAST_ANCHOR", "landing_bus": "EAST_ANCHOR",
                   "source": "t"}]).to_csv(landing, index=False)
    scaled = pd.DataFrame([
        {"technology": "wind_offshore", "bus": "PLATFORM", "capacity_mw": 100.0, "site_name": "CP30_a"},
        {"technology": "wind_offshore", "bus": "EAST_ANCHOR", "capacity_mw": 50.0, "site_name": "CP30_b"},
        {"technology": "wind_offshore", "bus": "NORTH_ANCHOR", "capacity_mw": 50.0, "site_name": "CP30_c"},
    ])
    targets = _targets([("wind_offshore", "East England", 300.0), ("wind_offshore", "North England", 600.0)])
    regional = {"enabled": True, "region_anchor_file": str(anchors), "region_method": "nearest_anchor"}
    connection = {"enabled": True, "mode": "reassign_bus_preserve_profile", "file": str(landing)}
    return scaled, targets, regional, connection


def test_platform_proxy_takes_the_region_of_its_landing_bus(region_inputs):
    scaled, targets, regional, connection = region_inputs
    network = _region_network()
    region_buses = landing_buses_for_regions(scaled, connection)
    assert region_buses.tolist()[:2] == ["LANDING", "EAST_ANCHOR"]
    assert pd.isna(region_buses.iloc[2])     # a source bus the map does not list keeps its own bus
    by_source = apply_cp30_es5_regional_allocation(scaled, None, "FFR", 2030, network, regional, LOG,
                                                   targets_override=targets)
    by_landing = apply_cp30_es5_regional_allocation(scaled, None, "FFR", 2030, network, regional, LOG,
                                                    targets_override=targets, region_buses=region_buses)
    assert by_source.set_index("site_name").at["CP30_a", "cp30_region"] == "East England"
    assert by_landing.set_index("site_name").at["CP30_a", "cp30_region"] == "North England"
    # Only the label moves: the bus is still the source bus (the landing map moves it later).
    assert by_landing.set_index("site_name").at["CP30_a", "bus"] == "PLATFORM"
    # ES5 regional totals still reconcile exactly; the MW moves between the two residuals.
    for result in (by_source, by_landing):
        got = result.groupby("cp30_region")["capacity_mw"].sum()
        assert got["East England"] == pytest.approx(300.0)
        assert got["North England"] == pytest.approx(600.0)
    assert by_landing.set_index("site_name").at["CP30_b", "capacity_mw"] == pytest.approx(300.0)


def test_no_landing_map_or_another_mode_leaves_regions_by_source_bus(region_inputs):
    scaled, _, _, connection = region_inputs
    assert landing_buses_for_regions(scaled, dict(connection, enabled=False)).isna().all()
    assert landing_buses_for_regions(scaled, dict(connection, mode="replace_capacity_allocation")).isna().all()
    assert landing_buses_for_regions(scaled, None).isna().all()


# --------------------------------------------------------------------------
# Configuration (consolidated 15 Oct 2026): corrections are plain behaviour,
# the two placement choices live once in the brownfield overlay (config/defaults/method/brownfield.yaml)
# --------------------------------------------------------------------------

RETIRED_SWITCHES = ("refill_at_excluded_landing", "offshore_region_bus", "distributed_to_gsp_node",
                    "proxy_relocation_file", "distributed_thresholds_mw", "distributed_carriers")


def test_retired_placement_switches_are_named_nowhere_in_config():
    for path in [*defaults_files(ROOT / "config"), *scenario_files(ROOT / "config")]:
        text = path.read_text(encoding="utf-8")
        for key in RETIRED_SWITCHES:
            assert f"{key}:" not in text, (path.name, key)


def test_placement_choices_live_in_cp30_defaults_and_new_dispatch_pins_national():
    defaults = load_defaults()
    cp30 = load_layer("method", "brownfield")
    assert proxy_profile_settings(cp30["renewables"])["basis"] == "regional"
    assert cp30["fleet"]["spatial_proxy"]["brownfield"]["small_scale_solar_reservation"] is True
    # Code defaults are the earlier rules (no global key in the base defaults).
    assert proxy_profile_settings(None)["basis"] == "national"
    assert "small_scale_solar_reservation" not in ((defaults.get("fleet") or {}).get("brownfield") or {})
    scenarios = load_scenario_entries(ROOT / "config", include_archive=True)   # CP30 New Dispatch is archived
    ffr = scenarios["CP30_FFR_2030"]
    nd = scenarios["CP30_ND_2030_ETYS_market_neso_es5_landing_reassignment_reinforced"]
    assert "proxy_profile_basis" not in (ffr.get("renewables") or {})
    assert "small_scale_solar_reservation" not in ffr["fleet"]["spatial_proxy"]["brownfield"]
    assert nd["renewables"]["proxy_profile_basis"] == "national"
    assert ffr["fleet"]["spatial_proxy"]["offshore_connection_allocation"]["file"].endswith(
        "cp30_offshore_landing_point_profile_preserving.csv")


# --------------------------------------------------------------------------
# Fix 4: regional weather profiles for proxy units
# --------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pypsa  # noqa: E402

from scripts.generators.proxy_profiles import (  # noqa: E402
    NATIONAL_KEY, area_keys, build_regional_proxy_profiles, offshore_basin,
)

HOURS = pd.date_range("2013-01-01", periods=48, freq="h")


def _site_profiles():
    """MW profiles: three northern sites (windy, one shape), three southern (calm, another)."""
    t = np.arange(len(HOURS))
    north = 0.5 + 0.3 * np.sin(t / 5.0)
    south = 0.2 + 0.1 * np.cos(t / 3.0)
    data = {}
    for name, shape, mw in (("N1", north, 100.0), ("N2", north, 200.0), ("N3", north * 0.9, 100.0),
                            ("S1", south, 150.0), ("S2", south, 150.0), ("S3", south * 1.1, 100.0)):
        data[name] = shape * mw
    return pd.DataFrame(data, index=HOURS), {"N1": 100.0, "N2": 200.0, "N3": 100.0,
                                             "S1": 150.0, "S2": 150.0, "S3": 100.0}


def test_offshore_basins_follow_the_named_fleet_geography():
    assert offshore_basin(58.2, -3.0) == "moray_aberdeen"         # Beatrice
    assert offshore_basin(56.17, -3.0) == "forth_tay"             # Methil and the Fife proxy
    assert offshore_basin(55.7, -4.9) == "west_coast_irish_sea"   # Hunterston
    assert offshore_basin(54.05, -3.5) == "west_coast_irish_sea"  # Walney
    assert offshore_basin(54.9, 2.2) == "humber_dogger_teesside"  # Dogger Bank
    assert offshore_basin(52.2, 2.5) == "wash_east_anglia"        # East Anglia ONE
    assert offshore_basin(51.6, 1.5) == "thames_south_coast"      # London Array
    assert offshore_basin(None, 1.0) is None
    site = {"lat": 54.9, "lon": 2.2, "cp30_region": "North England"}
    assert area_keys(site, "wind_offshore") == [("basin", "humber_dogger_teesside"),
                                                ("region", "North England"), NATIONAL_KEY]
    assert area_keys(site, "wind_onshore") == [("region", "North England"), NATIONAL_KEY]


def _members(capacities):
    rows = []
    for col, cap in capacities.items():
        region = "North" if col.startswith("N") else "South"
        rows.append({"column": col, "capacity": cap, "keys": [("region", region), NATIONAL_KEY]})
    return pd.DataFrame(rows)


def test_proxies_in_different_regions_get_different_profiles_and_keep_the_carrier_energy():
    profiles, caps = _site_profiles()
    national = profiles.sum(axis=1) / sum(caps.values())
    proxies = pd.DataFrame([
        {"pos": 10, "keys": [("region", "North"), NATIONAL_KEY], "weight": 300.0},
        {"pos": 11, "keys": [("region", "South"), NATIONAL_KEY], "weight": 100.0},
    ]).set_index("pos")
    settings = proxy_profile_settings({"proxy_profile_basis": "regional"})
    built, assign, groups, scale = build_regional_proxy_profiles(
        "wind_onshore", profiles, national, _members(caps), proxies, caps, settings, LOG)
    assert set(assign["group"]) == {"North", "South"}
    assert not np.allclose(built[10].to_numpy(), built[11].to_numpy())
    # The northern proxy follows the northern sites, not the national blend.
    assert built[10].corr(profiles["N1"]) == pytest.approx(1.0, abs=1e-6)
    # One scalar per carrier: the proxy fleet keeps the national-basis energy.
    regional_energy = 300.0 * built[10].mean() + 100.0 * built[11].mean()
    assert regional_energy == pytest.approx(400.0 * national.mean())
    by_group = groups.set_index("group")
    assert by_group.loc["North", "cf_raw"] > by_group.loc["South", "cf_raw"]
    # Without energy preservation the raw regional CFs are used and the total moves.
    raw, *_ = build_regional_proxy_profiles(
        "wind_onshore", profiles, national, _members(caps), proxies, caps,
        dict(settings, preserve_carrier_energy=False), LOG)
    assert 300.0 * raw[10].mean() + 100.0 * raw[11].mean() != pytest.approx(400.0 * national.mean())


def test_thin_groups_fall_back_to_region_then_gb():
    profiles, caps = _site_profiles()
    national = profiles.sum(axis=1) / sum(caps.values())
    members = _members(caps)
    # A basin with one named site falls back to its region; an unknown region falls back to GB.
    members.at[int(members.index[members["column"].eq("N1")][0]), "keys"] = [
        ("basin", "lonely"), ("region", "North"), NATIONAL_KEY]
    proxies = pd.DataFrame([
        {"pos": 1, "keys": [("basin", "lonely"), ("region", "North"), NATIONAL_KEY], "weight": 50.0},
        {"pos": 2, "keys": [("region", "Nowhere"), NATIONAL_KEY], "weight": 50.0},
    ]).set_index("pos")
    settings = proxy_profile_settings({"proxy_profile_basis": "regional"})
    built, assign, _, _ = build_regional_proxy_profiles(
        "wind_onshore", profiles, national, members, proxies, caps, settings, LOG)
    a = assign.set_index("row")
    assert (a.at[1, "level"], a.at[1, "group"]) == ("region", "North")
    assert "basin:lonely (1 sites" in a.at[1, "fallback_from"]
    assert (a.at[2, "level"], a.at[2, "group"]) == NATIONAL_KEY
    pd.testing.assert_series_equal(built[2], national)      # GB is the old fleet average, unscaled
    # The MW threshold bites too: three sites but under the minimum capacity.
    thin = dict(settings, min_mw=1000.0)
    _, assign, _, _ = build_regional_proxy_profiles(
        "wind_onshore", profiles, national, _members(caps), proxies.iloc[[0]], caps, thin, LOG)
    assert assign["level"].iloc[0] == "national"


def test_proxy_profile_settings_default_to_national_and_validate():
    assert proxy_profile_settings(None)["basis"] == "national"
    assert proxy_profile_settings({})["min_sites"] == 3
    assert proxy_profile_settings({})["min_mw"] == 300.0
    with pytest.raises(ValueError, match="proxy_profile_basis"):
        proxy_profile_settings({"proxy_profile_basis": "local"})


@pytest.fixture
def profile_build(tmp_path):
    profiles, caps = _site_profiles()
    (tmp_path / "profiles").mkdir()
    profiles.to_csv(tmp_path / "profiles" / "wind_onshore_2013.csv")
    rows = [{"site_name": c, "technology": "wind_onshore", "capacity_mw": cap, "bus": "B1",
             "cp30_region": "North" if c.startswith("N") else "South", "fleet_origin": "named",
             "lat": 57.0 if c.startswith("N") else 51.0, "lon": -3.0}
            for c, cap in caps.items()]
    rows += [
        {"site_name": "CP30_increment_wind_onshore_north", "technology": "wind_onshore", "capacity_mw": 300.0,
         "bus": "B1", "cp30_region": "North", "fleet_origin": "increment", "lat": 57.0, "lon": -3.0},
        {"site_name": "CP30_increment_wind_onshore_south", "technology": "wind_onshore", "capacity_mw": 100.0,
         "bus": "B1", "cp30_region": "South", "fleet_origin": "increment", "lat": 51.0, "lon": -3.0},
    ]
    sites = pd.DataFrame(rows)

    def run(basis):
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2030-01-01", periods=48, freq="h"))
        network.add("Bus", "B1")
        audit = {}
        network = irg.add_renewable_generators(
            network, sites, profiles_dir=str(tmp_path / "profiles"), renewables_year=2013,
            performance_factors={"wind_onshore": 0.8}, profile_capacities={"wind_onshore": caps},
            proxy_profiles={"proxy_profile_basis": basis}, profile_audit=audit,
        )
        return network, audit
    return run


def _energy(network, names):
    p = network.generators_t.p_max_pu[names]
    return float((p.mean() * network.generators.loc[names, "p_nom"]).sum())


def test_regional_basis_gives_area_profiles_and_keeps_the_national_total(profile_build):
    national, _ = profile_build("national")
    regional, audit = profile_build("regional")
    north, south = "CP30_increment_wind_onshore_north", "CP30_increment_wind_onshore_south"
    pn, pr = national.generators_t.p_max_pu, regional.generators_t.p_max_pu
    assert np.allclose(pn[north].to_numpy(), pn[south].to_numpy())          # old: one GB profile
    assert not np.allclose(pr[north].to_numpy(), pr[south].to_numpy())      # new: area profiles
    assert pr[north].mean() > pr[south].mean()
    # Named units keep their own site profiles.
    named = [c for c in pn.columns if c not in (north, south)]
    pd.testing.assert_frame_equal(pn[named], pr[named])
    # Carrier total available energy, after the performance factor, is unchanged.
    everyone = list(pn.columns)
    assert _energy(regional, everyone) == pytest.approx(_energy(national, everyone), rel=1e-6)
    assert regional.generators.at[north, "proxy_profile_group"] == "region:North"
    assert ("proxy_profile_group" not in national.generators.columns
            or national.generators["proxy_profile_group"].fillna("").eq("").all())
    assert audit["basis"] == "regional"
    assert set(audit["assignments"]["generator"]) == {north, south}
    groups = audit["groups"].set_index("group")
    assert set(groups.index) == {"North", "South", "GB"}       # GB is the reference row
    assert groups.at["GB", "proxy_units"] == 0


def test_energy_scalar_above_one_never_zeroes_a_proxy_profile(tmp_path):
    """A scaled regional profile can peak above 1.0; it is a CF, not MW (offline check, 15 Oct 2026)."""
    t = np.arange(len(HOURS))
    north = np.clip(0.55 + 0.5 * np.sin(t / 5.0), 0.0, 1.0)     # reaches rated output
    south = 0.2 + 0.1 * np.cos(t / 3.0)
    caps = {"N1": 100.0, "N2": 100.0, "N3": 100.0, "S1": 100.0, "S2": 100.0, "S3": 100.0}
    profiles = pd.DataFrame({c: (north if c.startswith("N") else south) * mw for c, mw in caps.items()},
                            index=HOURS)
    (tmp_path / "profiles").mkdir()
    profiles.to_csv(tmp_path / "profiles" / "wind_onshore_2013.csv")
    rows = [{"site_name": c, "technology": "wind_onshore", "capacity_mw": cap, "bus": "B1",
             "cp30_region": "North" if c.startswith("N") else "South", "fleet_origin": "named",
             "lat": 55.0, "lon": -3.0} for c, cap in caps.items()]
    # Most proxy capacity is in the calm south, so the carrier scalar is above 1.
    rows += [{"site_name": "CP30_increment_n", "technology": "wind_onshore", "capacity_mw": 50.0, "bus": "B1",
              "cp30_region": "North", "fleet_origin": "increment", "lat": 55.0, "lon": -3.0},
             {"site_name": "CP30_increment_s", "technology": "wind_onshore", "capacity_mw": 500.0, "bus": "B1",
              "cp30_region": "South", "fleet_origin": "increment", "lat": 55.0, "lon": -3.0}]
    sites = pd.DataFrame(rows)
    built = {}
    for basis in ("national", "regional"):
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2030-01-01", periods=48, freq="h"))
        network.add("Bus", "B1")
        audit = {}
        built[basis] = irg.add_renewable_generators(
            network, sites, profiles_dir=str(tmp_path / "profiles"), renewables_year=2013,
            performance_factors={"wind_onshore": 0.76}, profile_capacities={"wind_onshore": caps},
            proxy_profiles={"proxy_profile_basis": basis}, profile_audit=audit,
        )
        if basis == "regional":
            assert audit["groups"]["energy_scalar"].iloc[0] > 1.0
    p = built["regional"].generators_t.p_max_pu
    assert p["CP30_increment_n"].max() > 0.5 and p["CP30_increment_s"].mean() > 0.1
    names = ["CP30_increment_n", "CP30_increment_s"]
    # The toy scalar is large (about 1.8), so the clip at 1.0 after the factor takes a few
    # percent; the bug this guards against zeroed the profile (about -95%).
    assert _energy(built["regional"], names) == pytest.approx(_energy(built["national"], names), rel=0.05)


# --------------------------------------------------------------------------
# Audit items (15 Oct 2026): NE Scotland landings, Inch Cape, Drax, GSP nodes, rooftop solar
# --------------------------------------------------------------------------

CROSSWALK = ROOT / "data" / "network" / "ETYS" / "repd_offshore_landing_crosswalk.csv"
BIOMASS_SCHEDULE = ROOT / "data" / "market" / "CP30" / "biomass_station_schedule_2030.csv"


def test_ne_scotland_proxies_land_south_of_b1a_not_at_blackhillock():
    landing = pd.read_csv(LANDING_MAP).set_index("source_bus")
    expected = {"ABBA1-": "KINT4J", "REDM1Q": "KINT4J", "KINM2A": "PEHE4J", "PEHG1Q": "PEHE4J", "LOSD4A": "PEHE4J"}
    for source, target in expected.items():
        assert landing.at[source, "landing_bus"] == target
        assert "south of B1a" in landing.at[source, "source"]
    # The LOSD4A row keeps its earlier provenance note.
    assert "Successor to the KINM2A row" in landing.at["LOSD4A", "source"]


def test_inch_cape_lands_at_cockenzie_275_kv_and_seagreen_1a_stays_at_400_kv():
    crosswalk = pd.read_csv(CROSSWALK, dtype=str).set_index("repd_ref_id")
    assert crosswalk.at["2527", "landing_bus"] == "COCK2-"
    assert "B-2-2d" in crosswalk.at["2527", "notes"]
    assert crosswalk.at["14144", "landing_bus"] == "COCK4Q"


def test_drax_biomass_is_split_across_its_two_400_kv_sections_by_bmu_node():
    from scripts.CP30.gas_schedule import load_gas_station_schedule
    from scripts.public_ownership.legacy_valuation import _base_name

    schedule = load_gas_station_schedule(BIOMASS_SCHEDULE, carriers=("biomass",))
    drax = schedule[schedule["station_name"].map(_base_name).eq("Drax Power Station (biomass units)")]
    assert drax.set_index("bus")["capacity_mw"].astype(float).to_dict() == {"DRAX41": 1330.0, "DRAX42": 665.0}
    assert drax["capacity_mw"].astype(float).sum() == 1995.0           # unchanged station total
    assert set(drax["cfd_round"]) == {"DRAX_BRIDGE"}
    assert drax["commissioning_year"].nunique() == 1
    assert all("Dir_con_BMUs_to_node" in text for text in drax["bus_source"])


# Item 4: distribution-size sites at their own GSP's demand node.

from shapely.geometry import box  # noqa: E402


def _gsp_inputs():
    import geopandas as gpd

    polygons = gpd.GeoDataFrame({
        "GSPs": ["AAAA_1", "BBBB_1|BBBB_2", "CCCC_1"],
        "GSPGroup": ["_E", "_P", "_E"],
    }, geometry=[box(400000, 400000, 410000, 410000), box(410000, 400000, 420000, 410000),
                 box(420000, 400000, 430000, 410000)], crs="EPSG:27700")
    dem = pd.DataFrame({
        "GSP Id": ["AAAA_1", "BBBB_1", "BBBB_2"],
        "Node Id": ["AAAA11", "BBBB11", "BBBB12"],
        "GSP Group ID": ["_E", "_P", "_P"],
        "Dem as % of demand within the GSP Group ID per each node": [1.0, 0.3, 0.7],
    })
    network = pypsa.Network()
    for name, x, v in (("AAAA11", 405000, 132), ("AAAA41", 401000, 400), ("BBBB11", 412000, 132),
                       ("BBBB12", 418000, 132), ("CCCC11", 425000, 132)):
        network.add("Bus", name, x=x, y=405000, v_nom=v)
    for name in ("AAAA11", "BBBB11", "BBBB12", "CCCC11"):
        network.add("Load", f"load_{name}", bus=name, p_set=10.0)
    return network, polygons, dem


def _wgs(x, y):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(x, y)
    return lat, lon


def test_gsp_node_locator_uses_the_demand_nodes_of_each_polygon():
    from scripts.utilities.gsp_nodes import GspNodeLocator

    network, polygons, dem = _gsp_inputs()
    locator = GspNodeLocator(network, polygons=polygons, dem_per_node=dem, log=LOG)
    assert locator.node_for(*_wgs(401500, 405000)) == "AAAA11"   # nearer AAAA41 (no load), but its GSP is AAAA11
    assert locator.node_for(*_wgs(411000, 405000)) == "BBBB12"   # a split GSP takes its largest-share node
    assert locator.node_for(*_wgs(426000, 405000)) == "CCCC11"   # no Dem_per_node row: nearest load bus
    table = locator.table.set_index("gsp_ids")
    assert table.at["CCCC_1", "method"] == "nearest_load_bus"


def test_only_distribution_size_sites_move_to_their_gsp_node_by_licence_threshold():
    from scripts.utilities.gsp_nodes import GspNodeLocator, place_distributed_at_gsp_nodes

    network, polygons, dem = _gsp_inputs()
    locator = GspNodeLocator(network, polygons=polygons, dem_per_node=dem, log=LOG)
    sites = pd.DataFrame([
        {"site_name": "small_solar_E", "technology": "solar_pv", "capacity_mw": 40.0, "bus": "AAAA41"},
        {"site_name": "large_solar_E", "technology": "solar_pv", "capacity_mw": 150.0, "bus": "AAAA41"},
        {"site_name": "wind_P_9", "technology": "wind_onshore", "capacity_mw": 9.0, "bus": "AAAA41"},
        {"site_name": "wind_P_20", "technology": "wind_onshore", "capacity_mw": 20.0, "bus": "AAAA41"},
        {"site_name": "offshore", "technology": "wind_offshore", "capacity_mw": 5.0, "bus": "AAAA41"},
    ])
    for i, (x, y) in enumerate([(401500, 405000), (401500, 405000), (411000, 405000), (411000, 405000),
                                (401500, 405000)]):
        sites.loc[i, "lat"], sites.loc[i, "lon"] = _wgs(x, y)
    placed, moves = place_distributed_at_gsp_nodes(sites, locator, log=LOG)
    bus = placed.set_index("site_name")["bus"]
    assert bus["small_solar_E"] == "AAAA11"           # 40 MW < 100 MW (England and Wales)
    assert bus["large_solar_E"] == "AAAA41"           # transmission-size: keeps its bus
    assert bus["wind_P_9"] == "BBBB12"                # 9 MW < 10 MW (SHET)
    assert bus["wind_P_20"] == "AAAA41"               # 20 MW >= 10 MW in SHET
    assert bus["offshore"] == "AAAA41"                # not a listed carrier
    assert placed.set_index("site_name").at["small_solar_E", "data_source"] == "CP30_distributed_gsp_node"
    assert set(moves["site_name"]) == {"small_solar_E", "wind_P_9"}
    assert placed["capacity_mw"].sum() == sites["capacity_mw"].sum()


def test_gsp_node_placements_are_not_moved_by_the_bmu_voltage_heuristic():
    from scripts.utilities.spatial_utils import apply_etys_bmu_mapping

    network, _, _ = _gsp_inputs()
    sites = pd.DataFrame([{"site_name": "rooftop", "technology": "solar_pv", "capacity_mw": 5.0, "bus": "AAAA11",
                           "data_source": "CP30_distributed_gsp_node"},
                          {"site_name": "rooftop_plain", "technology": "solar_pv", "capacity_mw": 5.0,
                           "bus": "AAAA11", "data_source": "REPD_named_fleet"}])
    mapped = apply_etys_bmu_mapping(sites.copy(), network).set_index("site_name")["bus"]
    assert mapped["rooftop"] == "AAAA11"


# Item 5: FES small-scale solar reserved at its GSP before the named pipeline fills ES5.

def test_small_scale_reservation_trims_uncontracted_pipeline_first_and_keeps_es5_exact():
    from scripts.CP30.brownfield_fleet import build_small_scale_solar_reservation

    named = _named([
        {"site_name": "built", "technology": "solar_pv", "cp30_region": "Midlands", "capacity_mw": 3000.0,
         "development_status": "Operational"},
        {"site_name": "contracted", "technology": "solar_pv", "cp30_region": "Midlands", "capacity_mw": 1000.0,
         "development_status": "Awaiting Construction", "cfd_round": "AR6", "planning_granted_date": "2024-01-01"},
        {"site_name": "consented", "technology": "solar_pv", "cp30_region": "Midlands", "capacity_mw": 2500.0,
         "development_status": "Awaiting Construction", "planning_granted_date": "2023-01-01"},
        {"site_name": "building", "technology": "solar_pv", "cp30_region": "Midlands", "capacity_mw": 755.0,
         "development_status": "Under Construction"},
    ])
    targets = _targets([("solar_pv", "Midlands", 7255.0)])
    reserved = build_small_scale_solar_reservation(pd.DataFrame([
        {"gsp": "Gsp A", "capacity_mw": 2000.0, "bus": "A11", "cp30_region": "Midlands", "lat": 52.5, "lon": -1.9},
        {"gsp": "Gsp B", "capacity_mw": 1225.0, "bus": "B11", "cp30_region": "Midlands", "lat": 52.6, "lon": -1.5},
    ]), 2030, LOG)
    kept, residual, trimmed, audit = reconcile_with_es5(named, targets, logger=LOG, reserved=reserved,
                                                       contracted_last=True)
    assert kept["capacity_mw"].sum() + residual["capacity_mw"].sum() == pytest.approx(7255.0)
    small = kept[kept["cp30_spatial_method"].eq("small_scale_solar_reserved")]
    assert small["capacity_mw"].sum() == pytest.approx(3225.0)
    assert small["fleet_origin"].eq("increment").all() and small["repd_ref_id"].isna().all()
    # 3,225 MW of room: the uncontracted consented project goes first (2,500), then the
    # under-construction one (725 of 755); the contracted and operational projects stay whole.
    t = trimmed.set_index("site_name")["trimmed_mw"]
    assert t.to_dict() == pytest.approx({"consented": 2500.0, "building": 725.0})
    by_name = kept.set_index("site_name")["capacity_mw"]
    assert by_name["contracted"] == 1000.0 and by_name["built"] == 3000.0
    assert set(trimmed["trim_reason"]) == {"small_scale_reservation"}
    assert audit.iloc[0]["reserved_small_scale_mw"] == pytest.approx(3225.0)


def test_reservation_never_forces_operational_plant_out():
    from scripts.CP30.brownfield_fleet import build_small_scale_solar_reservation

    named = _named([{"site_name": "built", "technology": "solar_pv", "cp30_region": "Wales",
                     "capacity_mw": 900.0, "development_status": "Operational"}])
    reserved = build_small_scale_solar_reservation(pd.DataFrame([
        {"gsp": "G", "capacity_mw": 500.0, "bus": "W11", "cp30_region": "Wales", "lat": 52.0, "lon": -3.5}]), 2030, LOG)
    kept, residual, trimmed, _ = reconcile_with_es5(named, _targets([("solar_pv", "Wales", 1000.0)]), logger=LOG,
                                                   reserved=reserved, contracted_last=True)
    assert trimmed.empty
    assert kept.set_index("site_name").at["built", "capacity_mw"] == 900.0
    assert kept["capacity_mw"].sum() == pytest.approx(1000.0)                 # the reservation shrinks to 100
    assert residual["capacity_mw"].sum() == pytest.approx(0.0)


def test_without_a_reservation_trimming_is_unchanged():
    named = _named([
        {"site_name": "contracted", "technology": "solar_pv", "cp30_region": "Midlands", "capacity_mw": 1000.0,
         "development_status": "Awaiting Construction", "cfd_round": "AR6", "planning_granted_date": "2025-01-01"},
        {"site_name": "consented", "technology": "solar_pv", "cp30_region": "Midlands", "capacity_mw": 1000.0,
         "development_status": "Awaiting Construction", "planning_granted_date": "2023-01-01"},
    ])
    _, _, trimmed, _ = reconcile_with_es5(named, _targets([("solar_pv", "Midlands", 1500.0)]), logger=LOG)
    # Old rule, most recently consented first, regardless of contract.
    assert trimmed.set_index("site_name")["trimmed_mw"].to_dict() == {"contracted": 500.0}


def test_small_scale_solar_is_labelled_merchant_not_cfd():
    from scripts.CP30.brownfield_fleet import SMALL_SCALE_SOLAR_METHOD

    sites = pd.DataFrame([
        {"site_name": "named", "technology": "solar_pv", "capacity_mw": 10.0, "fleet_origin": "named",
         "development_status": "Operational", "lat": 52.0, "lon": -1.0, "operational_date": "2015-06-01"},
        {"site_name": "CP30_increment_solar_pv_small_scale_X_2030", "technology": "solar_pv", "capacity_mw": 100.0,
         "fleet_origin": "increment", "cp30_spatial_method": SMALL_SCALE_SOLAR_METHOD, "lat": 52.0, "lon": -1.0},
        {"site_name": "CP30_increment_solar_pv_1", "technology": "solar_pv", "capacity_mw": 100.0,
         "fleet_origin": "increment", "lat": 52.0, "lon": -1.0},
    ])
    out = irg._assign_subsidy_attributes_by_origin(sites, 2030, 20, {"solar_pv": 0.5}, LOG, "2017-04-01", {})
    small = out[out["site_name"].str.contains("small_scale")]
    assert set(small["support_type"]) == {"merchant"}
    assert small["capacity_mw"].sum() == pytest.approx(100.0)
    proxy = out[out["site_name"].str.startswith("CP30_increment_solar_pv_1")]   # split into CfD/merchant tranches
    assert "CfD" in set(proxy["support_type"])          # the ordinary proxy keeps the FES split


# Network-fix hand-over (15 Oct 2026): radial 132 kV pockets that were placement errors.

def test_pocket_farms_take_their_tec_connection_points():
    from scripts.utilities.spatial_utils import _explicit_onshore_connection

    buses = {"GLGL1Q", "NECU1-", "NECU1B", "NECU1C", "BENB1-", "CLOI1A", "ENHI1-", "WLEE2-"}
    expected = {
        "Sanquhar 2 Community Wind Farm": "GLGL1Q",
        "North_Kyle_Energy_Project": "NECU1-",
        "Benbrack Wind Farm": "BENB1-",
        "Cloiche Wind Farm": "CLOI1A",
        "Cloiche_Wind_Farm_CfD": "CLOI1A",
        "Enoch Hill": "ENHI1-",
        "Pencloe Wind Farm": "NECU1C",
        "Windy Standard III Wind Farm (Brockloch Rig Phase 2)": "NECU1C",
        "Greenburn Wind Park": "NECU1B",
        "Over Hill Wind Farm": "NECU1B",
    }
    for name, bus in expected.items():
        assert _explicit_onshore_connection(name, buses)[1] == bus, name
    # Different farms with similar names keep the ordinary placement.
    for name in ("Coire Na Cloiche Wind Farm", "Sanquhar Community Windfarm", "Dersalloch",
                 "Brochloch Rig (formerly Windy Standard II)"):
        assert _explicit_onshore_connection(name, buses) is None, name
    # Inert on a network without the substation.
    assert _explicit_onshore_connection("Cloiche Wind Farm", {"BHLA11"}) is None


def test_documented_connection_wins_over_the_gsp_node_placement():
    from scripts.utilities.gsp_nodes import GspNodeLocator, place_distributed_at_gsp_nodes
    from scripts.utilities.spatial_utils import documented_connection_mask

    network, polygons, dem = _gsp_inputs()
    network.add("Bus", "CLOI1A", x=411500, y=405000, v_nom=132)
    locator = GspNodeLocator(network, polygons=polygons, dem_per_node=dem, log=LOG)
    lat, lon = _wgs(411000, 405000)                    # a SHET (_P) polygon: 10 MW threshold
    sites = pd.DataFrame([
        {"site_name": "Cloiche Wind Farm", "technology": "wind_onshore", "capacity_mw": 5.0, "bus": "CLOI1A",
         "lat": lat, "lon": lon},
        {"site_name": "Some Small Farm", "technology": "wind_onshore", "capacity_mw": 5.0, "bus": "CLOI1A",
         "lat": lat, "lon": lon},
    ])
    placed, _ = place_distributed_at_gsp_nodes(sites, locator, log=LOG,
                                               keep=documented_connection_mask(sites, network.buses.index))
    bus = placed.set_index("site_name")["bus"]
    assert bus["Cloiche Wind Farm"] == "CLOI1A"
    assert bus["Some Small Farm"] == "BBBB12"


def test_proxy_relocation_moves_only_listed_carriers_and_is_inert_without_the_buses(tmp_path):
    network = pypsa.Network()
    for name in ("WLEE2-", "EKIS2-", "OTHER"):
        network.add("Bus", name)
    table = tmp_path / "relocations.csv"
    pd.DataFrame([
        {"bus": "WLEE2-", "target_bus": "EKIS2-", "carriers": "wind_onshore", "reason": "r", "source": "s"},
        {"bus": "GONE", "target_bus": "EKIS2-", "carriers": "wind_onshore", "reason": "r", "source": "s"},
    ]).to_csv(table, index=False)
    sites = pd.DataFrame([
        {"site_name": "p1", "technology": "wind_onshore", "capacity_mw": 202.0, "bus": "WLEE2-"},
        {"site_name": "p2", "technology": "solar_pv", "capacity_mw": 5.0, "bus": "WLEE2-"},
        {"site_name": "p3", "technology": "wind_onshore", "capacity_mw": 10.0, "bus": "OTHER"},
    ])
    out, moves = irg.relocate_proxy_rows(sites, table, network, LOG)
    assert out.set_index("site_name")["bus"].to_dict() == {"p1": "EKIS2-", "p2": "WLEE2-", "p3": "OTHER"}
    assert out["capacity_mw"].sum() == sites["capacity_mw"].sum()
    assert moves.to_dict("records") == [{"bus": "WLEE2-", "target_bus": "EKIS2-", "rows": 1, "capacity_mw": 202.0}]


def test_shipped_proxy_relocation_table_is_well_formed():
    table = pd.read_csv(ROOT / "data" / "network" / "ETYS" / "cp30_proxy_relocations.csv", dtype=str)
    assert {"bus", "target_bus", "carriers", "reason", "source"} <= set(table.columns)
    assert not table["bus"].duplicated().any()
    assert table.set_index("bus").at["WLEE2-", "target_bus"] == "EKIS2-"
    assert table["reason"].str.len().gt(40).all() and table["source"].str.strip().ne("").all()
