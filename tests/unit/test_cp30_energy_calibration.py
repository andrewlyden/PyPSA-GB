"""Generation-calibration fixes of 26 Sep 2026 (CP30 energy audit).

Covers the weather-pinned performance-factor registry, the large-hydro inflow
profile, capacity-normalised marine profiles, the Other Renewables mapping and
unmapped-line refusal, the thermal floor-vs-cap guard, the wave profile, and
the CP30 energy validation on both bases.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.CP30.energy_validation import (
    build_energy_validation,
    cp30_energy_targets,
    network_energy_tables,
    sum_energy_tables,
)
from scripts.CP30.generation import load_cp30_generation_capacities
from scripts.generators.integrate_renewable_generators import (
    add_renewable_generators,
    log_applied_performance_factors,
)
from scripts.generators.integrate_thermal_generators import (
    apply_renewable_profiles_to_fes,
    capacity_weighted_fleet_profile,
    check_min_output_within_availability,
)
from scripts.generators.renewable_performance_factors import (
    load_performance_factor_registry,
    select_performance_factor,
)
from scripts.renewables.generate_marine_profiles import MarineRenewableGenerator
from scripts.renewables.large_hydro_inflow import (
    inflow_capacity_factor,
    large_hydro_proxy_p_max_pu,
    load_monthly_shape,
)

REGISTRY = Path("data/market/support/renewable_performance_factor_registry.csv")
HYDRO_SHAPE = Path("data/renewables/hydro/espeni_npshyd_monthly_shape.csv")
CP30_ES1 = Path("resources/CP30/CP30_ES1.csv")
PATHWAY = "Further Flex and Renewables"


# ── performance-factor registry ─────────────────────────────────────────────

def _registry(rows):
    frame = pd.DataFrame(rows)
    for column in ("calendar_year", "calendar_month", "weather_year"):
        frame[column] = pd.array(frame.get(column, [pd.NA] * len(frame)), dtype="Int64")
    return frame


def test_weather_pinned_row_matches_only_its_weather_year():
    registry = _registry([
        {"carrier": "wind_onshore", "resource_region": "GB", "calendar_year": 2030,
         "calendar_month": pd.NA, "weather_year": 2013, "factor": 0.76},
    ])
    assert select_performance_factor(registry, "wind_onshore", "scotland", 2030, None, 0.805,
                                     weather_year=2013) == (pytest.approx(0.76), "registry")
    # Another weather year, or an unknown one, falls back to the carrier config.
    assert select_performance_factor(registry, "wind_onshore", "scotland", 2030, None, 0.805,
                                     weather_year=2019) == (0.805, "carrier_config_fallback")
    assert select_performance_factor(registry, "wind_onshore", "scotland", 2030, None, 0.805) == (
        0.805, "carrier_config_fallback")


def test_weather_pinned_row_beats_unpinned_gb_row_but_not_a_regional_row():
    registry = _registry([
        {"carrier": "wind_offshore", "resource_region": "GB", "calendar_year": 2030, "weather_year": pd.NA,
         "factor": 0.60},
        {"carrier": "wind_offshore", "resource_region": "GB", "calendar_year": 2030, "weather_year": 2013,
         "factor": 0.70},
        {"carrier": "wind_offshore", "resource_region": "north_sea", "calendar_year": 2030,
         "weather_year": pd.NA, "factor": 0.90},
    ])
    assert select_performance_factor(registry, "wind_offshore", "irish_sea", 2030, None, 0.5,
                                     weather_year=2013)[0] == pytest.approx(0.70)
    assert select_performance_factor(registry, "wind_offshore", "north_sea", 2030, None, 0.5,
                                     weather_year=2013)[0] == pytest.approx(0.90)


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not available")
def test_repo_registry_has_no_stale_config_copies_and_a_cp30_onshore_row():
    registry = load_performance_factor_registry(REGISTRY)
    # The physical_default rows were frozen copies of an old defaults stack
    # (onshore 0.850, solar 0.861) that silently overrode the live config.
    assert not registry["calibration_scope"].eq("physical_default").any()
    assert not registry["evidence_source"].eq("config/defaults.yaml").any()
    onshore = registry[registry["carrier"].eq("wind_onshore") & registry["calendar_year"].eq(2030)]
    assert len(onshore) == 1
    assert int(onshore["weather_year"].iloc[0]) == 2013
    assert 0.70 < float(onshore["factor"].iloc[0]) < 0.80
    marine = registry[registry["carrier"].eq("marine") & registry["calendar_year"].eq(2030)]
    assert len(marine) == 1 and int(marine["weather_year"].iloc[0]) == 2013
    assert float(marine["factor"].iloc[0]) == pytest.approx(403.59 / 486.49, abs=5e-4)
    # CP30 solar is calibrated to CP30 gross like onshore and marine (27 Sep 2026):
    # ES1 generation + curtailment over the rebuilt fleet's pre-factor energy.
    solar = registry[registry["carrier"].eq("solar_pv") & registry["calendar_year"].eq(2030)]
    assert len(solar) == 1
    assert int(solar["weather_year"].iloc[0]) == 2013
    assert solar["resource_region"].iloc[0] == "GB"
    assert solar["calibration_scope"].iloc[0] == "cp30_gross_energy_calibration"
    assert float(solar["factor"].iloc[0]) == pytest.approx(49669.66 / 48490.55, abs=5e-4)
    # Onshore refitted on the same rebuilt fleet.
    assert float(onshore["factor"].iloc[0]) == pytest.approx(70199.24 / 91761.61, abs=5e-4)
    # Offshore keeps its regional rows (no single 2030 GB row).
    assert registry[registry["carrier"].eq("wind_offshore") & registry["calendar_year"].eq(2030)
                    & registry["resource_region"].eq("GB")].empty


def test_applied_factor_log_reports_registry_value_not_config():
    network = pypsa.Network()
    network.add("Bus", "b")
    network.add("Generator", "g1", bus="b", carrier="wind_onshore", p_nom=100.0)
    network.generators["renewable_performance_factor"] = [0.76]
    network.generators["renewable_performance_factor_source"] = ["registry"]
    table = log_applied_performance_factors(network, {"wind_onshore": 0.805}, logging.getLogger("test"))
    assert table.loc[("wind_onshore", "registry"), "factor"] == pytest.approx(0.76)


# ── large hydro ─────────────────────────────────────────────────────────────

def test_large_hydro_profile_hits_annual_cf_and_follows_the_month_shape():
    snapshots = pd.date_range("2030-01-01", periods=8760, freq="h")
    shape = pd.Series([1.4, 1.2, 0.5, 0.9, 1.1, 0.5, 0.4, 0.5, 0.8, 1.2, 1.5, 2.0], index=range(1, 13))
    profile = large_hydro_proxy_p_max_pu(snapshots, 0.21, shape, 0.55)
    assert inflow_capacity_factor(profile, 0.55) == pytest.approx(0.21, rel=1e-9)
    monthly = profile.groupby(profile.index.month).mean()
    assert monthly[12] > 3 * monthly[7]
    assert profile.max() <= 1.0
    # Reservoir model off: p_max_pu is the energy itself.
    assert large_hydro_proxy_p_max_pu(snapshots, 0.21, shape, None).mean() == pytest.approx(0.21, rel=1e-9)


def test_large_hydro_profile_on_a_one_month_network_keeps_that_months_share():
    shape = pd.Series([2.0] + [12 / 11 - 2 / 11] * 11, index=range(1, 13))
    january = pd.date_range("2030-01-01", periods=744, freq="h")
    full = large_hydro_proxy_p_max_pu(pd.date_range("2030-01-01", periods=8760, freq="h"), 0.2, shape, None)
    month = large_hydro_proxy_p_max_pu(january, 0.2, shape, None)
    assert month.mean() == pytest.approx(full[full.index.month == 1].mean())
    assert month.mean() > 0.3


@pytest.mark.skipif(not HYDRO_SHAPE.exists(), reason="hydro shape data not available")
def test_espeni_hydro_shape_uses_the_weather_year_and_falls_back_to_the_mean():
    shape_2013, label = load_monthly_shape(HYDRO_SHAPE, 2013)
    assert label == "ESPENI NPSHYD 2013"
    assert shape_2013.mean() == pytest.approx(1.0)
    assert shape_2013[12] > shape_2013[7]
    _, fallback = load_monthly_shape(HYDRO_SHAPE, 1990)
    assert fallback.startswith("ESPENI NPSHYD mean")


def test_unmatched_large_hydro_takes_the_default_profile_not_flat_point_four(tmp_path):
    snapshots = pd.date_range("2030-01-01", periods=48, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "bus", v_nom=132.0)
    pd.DataFrame({"Named Station": np.full(48, 10.0)},
                 index=pd.date_range("2013-01-01", periods=48, freq="h")).to_csv(tmp_path / "large_hydro_2013.csv")
    sites = pd.DataFrame([{"site_name": "CP30_large_hydro_0", "technology": "large_hydro", "bus": "bus",
                           "capacity_mw": 50.0}])
    default = pd.Series(0.33, index=snapshots)
    add_renewable_generators(network, sites, profiles_dir=str(tmp_path), renewables_year=2013,
                             default_profiles={"large_hydro": default})
    assert network.generators_t.p_max_pu["CP30_large_hydro_0"].to_numpy() == pytest.approx(0.33)


# ── marine ──────────────────────────────────────────────────────────────────

def test_capacity_weighted_fleet_profile_is_a_capacity_factor():
    profiles = pd.DataFrame({"a": [2.0, 0.0, 2.0], "b": [10.0, 5.0, 0.0]})
    fleet = capacity_weighted_fleet_profile(profiles)
    assert fleet.tolist() == pytest.approx([1.0, 5 / 12, 2 / 12])
    # The old mean(axis=1).clip(0, 1) read MW as CF: [1, 1, 1].
    assert profiles.mean(axis=1).clip(0, 1).tolist() == [1.0, 1.0, 1.0]


def test_marine_takes_capacity_normalised_tidal_profile_in_stage_one(tmp_path):
    snapshots = pd.date_range("2030-01-01", periods=4, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "bus", v_nom=33.0)
    pd.DataFrame({"MeyGen": [6.0, 0.0, 3.0, 0.0], "EMEC": [2.0, 2.0, 0.0, 0.0]},
                 index=pd.date_range("2013-01-01", periods=4, freq="h")).to_csv(tmp_path / "tidal_stream_2013.csv")
    sites = pd.DataFrame([{"site_name": "CP30_marine_0", "technology": "marine", "bus": "bus", "capacity_mw": 100.0}])
    add_renewable_generators(network, sites, profiles_dir=str(tmp_path), renewables_year=2013)
    assert network.generators_t.p_max_pu["CP30_marine_0"].tolist() == pytest.approx([1.0, 0.25, 0.375, 0.0])


def test_thermal_stage_fallback_divides_repd_profile_by_its_capacity(tmp_path):
    snapshots = pd.date_range("2030-01-01", periods=3, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "b", x=0.0, y=0.0)
    network.add("Generator", "Site A", bus="b", carrier="tidal_stream", p_nom=4.0)
    network.add("Generator", "fes_marine", bus="b", carrier="marine", p_nom=50.0)
    pd.DataFrame({"Site A": [4.0, 2.0, 0.0]},
                 index=pd.date_range("2013-01-01", periods=3, freq="h")).to_csv(tmp_path / "tidal_stream_2013.csv")
    apply_renewable_profiles_to_fes(network, ["marine"], 2013, profiles_dir=str(tmp_path))
    assert network.generators_t.p_max_pu["fes_marine"].tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_wave_profile_mean_is_configured_cf_and_peaks_in_winter():
    generator = MarineRenewableGenerator()
    time_index = generator.create_time_index(2013)
    sites = pd.DataFrame([{"site_name": "wave A", "capacity_mw": 2.0, "lat": 58.9, "lon": -3.3}])
    first = generator.generate_site_profiles(sites, "shoreline_wave", time_index)
    second = generator.generate_site_profiles(sites, "shoreline_wave", time_index)
    cf = first["wave A"] / 2.0
    assert cf.mean() == pytest.approx(0.25, abs=0.01)
    monthly = cf.groupby(cf.index.month).mean()
    assert monthly[[1, 12]].mean() > 2 * monthly[[6, 7]].mean()
    pd.testing.assert_frame_equal(first, second)  # seeded per site and year


# ── thermal availability guard ─────────────────────────────────────────────

def test_min_output_above_availability_is_refused():
    check_min_output_within_availability({"factors": {"waste_to_energy": 0.62}, "min_output": {"waste_to_energy": 0.55}})
    with pytest.raises(ValueError, match="waste_to_energy"):
        check_min_output_within_availability({"factors": {"waste_to_energy": 0.50},
                                              "min_output": {"waste_to_energy": 0.55}})


# ── CP30 ES1 mapping ────────────────────────────────────────────────────────

def _es1(tmp_path, extra=()):
    base = [
        ("Capacity (MW)", "Renewable", "Onshore Wind", "Onshore Wind", 1000.0),
        ("Generation (GWh)", "Renewable", "Onshore Wind", "Onshore Wind", 2500.0),
        ("Curtailment (TWh)", "Renewable", "Onshore Wind", "Onshore Wind", 0.5),
        ("Capacity (MW)", "Renewable", "Other Renewables", "Other Renewables", 100.0),
        ("Generation (GWh)", "Renewable", "Other Renewables", "Other Renewables", 438.0),
        ("Capacity (MW)", "Renewable", "Hydro", "Hydro", 100.0),
        ("Generation (GWh)", "Renewable", "Hydro", "Hydro", 184.0),
        ("Capacity (MW)", "Storage", "Storage", "Battery", 50.0),
        ("Generation (GWh)", "Storage", "Storage", "Battery", 40.0),
        ("Capacity (MW)", "Interconnectors", "Interconnectors", "Interconnectors", 500.0),
        ("Demand Total (GWh) - Excluding Electrolysis", "Demand", None, None, 5000.0),
        *extra,
    ]
    frame = pd.DataFrame(base, columns=["Variable", "Category", "Type", "SubType", "2030"])
    frame.insert(0, "Pathway", PATHWAY)
    frame.insert(0, "Connection", "Both")
    path = tmp_path / "es1.csv"
    frame.to_csv(path, index=False)
    return path


def test_other_renewables_maps_to_biogas(tmp_path):
    targets = load_cp30_generation_capacities(_es1(tmp_path), PATHWAY, 2030).set_index("carrier")
    assert targets.loc["biogas", "capacity_mw"] == pytest.approx(100.0)
    assert targets.loc["biogas", "group"] == "thermal"
    assert tuple(targets.loc["biogas", "proxy_carriers"]) == ("biogas", "landfill_gas", "sewage_gas")


def test_unmapped_es1_capacity_line_is_refused(tmp_path):
    path = _es1(tmp_path, [("Capacity (MW)", "Renewable", "Geothermal", "Geothermal", 30.0)])
    with pytest.raises(ValueError, match="Geothermal"):
        load_cp30_generation_capacities(path, PATHWAY, 2030)


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 resource not available")
def test_real_es1_has_no_unmapped_supply_lines():
    targets = cp30_energy_targets(CP30_ES1, PATHWAY, 2030)
    assert targets["unmapped"].empty
    gen = targets["generation"]
    assert gen.loc["biogas", "generation_gwh"] == pytest.approx(4688.93)
    assert gen.loc["wind_onshore", "gross_gwh"] == pytest.approx(57758.45 + 12440.79)
    assert gen.loc["Hydrogen", "generation_gwh"] == pytest.approx(123.51 + 214.89)


# ── CP30 energy validation ─────────────────────────────────────────────────

def _network(demand_mw=5000.0 / 8.76, onshore_cf=0.34):
    snapshots = pd.date_range("2030-01-01", periods=8760, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "gb", carrier="AC")
    n.add("Load", "demand", bus="gb", carrier="electricity", p_set=pd.Series(demand_mw, index=snapshots))
    n.add("Generator", "on", bus="gb", carrier="wind_onshore", p_nom=1000.0,
          p_max_pu=pd.Series(onshore_cf, index=snapshots))
    n.add("Generator", "hydro", bus="gb", carrier="large_hydro", p_nom=100.0,
          p_max_pu=pd.Series(0.21 / 0.55, index=snapshots))
    n.add("Generator", "bio", bus="gb", carrier="biogas", p_nom=100.0, p_max_pu=0.5)
    return n


HYDRO_ON = {"enabled": True, "large_hydro": {"reservoir_model": True, "inflow_profile_factor": 0.55}}


def test_build_validation_compares_available_energy_with_cp30_gross(tmp_path):
    targets = cp30_energy_targets(_es1(tmp_path), PATHWAY, 2030)
    table = build_energy_validation(targets, network_energy_tables(_network(), HYDRO_ON), "build")
    rows = table.set_index(["component", "item"])
    onshore = rows.loc[("available_energy", "wind_onshore")]
    assert onshore["cp30_gwh"] == pytest.approx(3000.0)      # 2,500 generation + 500 curtailment
    assert onshore["model_gwh"] == pytest.approx(0.34 * 8760)
    assert onshore["status"] == "PASS"
    hydro = rows.loc[("available_energy", "large_hydro")]
    assert hydro["model_gwh"] == pytest.approx(0.21 * 100 * 8.76)   # inflow, not the ceiling
    assert rows.loc[("demand", "Electricity demand excluding electrolysis"), "status"] == "PASS"
    assert set(table["status"]) <= {"PASS", "WARN", "INFO"}


def test_build_validation_warns_on_energy_and_fails_on_demand(tmp_path):
    targets = cp30_energy_targets(_es1(tmp_path), PATHWAY, 2030)
    table = build_energy_validation(targets, network_energy_tables(_network(demand_mw=500.0, onshore_cf=0.40),
                                                                   HYDRO_ON), "build")
    rows = table.set_index(["component", "item"])
    assert rows.loc[("available_energy", "wind_onshore"), "status"] == "WARN"
    assert rows.loc[("demand", "Electricity demand excluding electrolysis"), "status"] == "FAIL"


def test_unmapped_line_fails_the_energy_validation(tmp_path):
    path = _es1(tmp_path, [("Generation (GWh)", "Renewable", "Geothermal", "Geothermal", 200.0)])
    table = build_energy_validation(cp30_energy_targets(path, PATHWAY, 2030),
                                    network_energy_tables(_network(), HYDRO_ON), "build")
    unmapped = table[table["component"].eq("unmapped_es1_line")]
    assert unmapped["status"].tolist() == ["FAIL"]


def test_dispatch_validation_reports_curtailment_and_sums_months(tmp_path):
    n = _network()
    snapshots = n.snapshots
    n.generators_t.p = pd.DataFrame({"on": 0.285 * 1000.0, "hydro": 21.0, "bio": 50.0}, index=snapshots)
    n.storage_units_t.p_dispatch = pd.DataFrame(index=snapshots)
    tables = network_energy_tables(n, HYDRO_ON, include_dispatch=True)
    half = {k: (v / 2 if isinstance(v, float) else v) for k, v in tables.items()}
    half["generators"] = tables["generators"].assign(available_mwh=tables["generators"]["available_mwh"] / 2,
                                                     dispatched_mwh=tables["generators"]["dispatched_mwh"] / 2)
    summed = sum_energy_tables([half, half])
    table = build_energy_validation(cp30_energy_targets(_es1(tmp_path), PATHWAY, 2030), summed, "dispatch")
    rows = table.set_index(["component", "item"])
    assert rows.loc[("dispatched_energy", "wind_onshore"), "model_gwh"] == pytest.approx(0.285 * 8760)
    curtailment = rows.loc[("curtailment", "wind_onshore")]
    assert curtailment["model_gwh"] == pytest.approx((0.34 - 0.285) * 8760)
    assert curtailment["cp30_gwh"] == pytest.approx(500.0)
    assert rows.loc[("dispatched_energy", "biogas"), "model_gwh"] == pytest.approx(438.0)
    assert rows.loc[("dispatched_energy", "biogas"), "status"] == "PASS"


def test_only_gross_energy_calibrations_may_exceed_one(tmp_path):
    base = {"carrier": "solar_pv", "resource_region": "GB", "calendar_year": 2030, "calendar_month": "",
            "weather_year": 2013, "evidence_source": "x", "notes": "x"}
    ok = tmp_path / "ok.csv"
    pd.DataFrame([{**base, "factor": 1.0243, "calibration_scope": "cp30_gross_energy_calibration"}]).to_csv(ok, index=False)
    assert float(load_performance_factor_registry(ok)["factor"].iloc[0]) == pytest.approx(1.0243)
    physical = tmp_path / "physical.csv"
    pd.DataFrame([{**base, "factor": 1.02, "calibration_scope": "lccc_raw_profile_regional_calibration"}]).to_csv(physical, index=False)
    with pytest.raises(ValueError):
        load_performance_factor_registry(physical)
    runaway = tmp_path / "runaway.csv"
    pd.DataFrame([{**base, "factor": 1.5, "calibration_scope": "cp30_gross_energy_calibration"}]).to_csv(runaway, index=False)
    with pytest.raises(ValueError):
        load_performance_factor_registry(runaway)
