"""Supply, network integration and scenario keys for the FES long-range
build (HT50_*), for any FES year."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generators import fes_supply as fs
from scripts.generators import build_fes_station_schedule as bss
from scripts.generators.build_fes_station_schedule import check_schedule_totals, fes_direct_totals

ROOT = Path(__file__).resolve().parents[2]
STATION_LIST = ROOT / "data" / "generators" / "fes" / "station_list.csv"
LANDING = ROOT / "data" / "network" / "ETYS" / "fes2050_offshore_landing.csv"
FES_2025 = ROOT / "resources" / "FES" / "FES_2025_data.csv"
ES1_2050 = ROOT / "resources" / "FES" / "fes2025_holistic_transition_2050_es1_lines.csv"
# The HT50 base network (ETYS with the 2050 upgrades), not the 2030 one: Kintore 275 kV
# (KINM2A) exists in 2030 but is 400 kV only by 2050, which broke the first HT50 build.
BASE_NETWORK = ROOT / "resources" / "network" / "FES25_HT_2050_network.nc"


# ───────────────────────────── configuration guards ─────────────────────────────

def test_config_absent_or_disabled_is_empty():
    assert fs.fes_supply_config({}) == {}
    assert fs.fes_supply_config({"fleet": {"fes_supply": {"enabled": False, "thermal": {"x": 1}}}}) == {}
    assert fs.fes_supply_part({"fleet": {"fes_supply": {"enabled": False, "thermal": {"x": 1}}}}, "thermal") == {}


def test_config_parts():
    cfg = {"fleet": {"fes_supply": {"enabled": True, "thermal": {"x": 1}, "storage": {"enabled": False, "y": 2}}}}
    assert fs.fes_supply_part(cfg, "thermal") == {"x": 1}
    assert fs.fes_supply_part(cfg, "storage") == {}
    assert fs.fes_supply_part(cfg, "offshore_landing") == {}


def test_old_keys_are_refused_not_aliased():
    from config.config_loader import check_renamed_keys

    # The loader refuses the old paths and names the new one; the module reads only fleet.fes_supply.
    for old in ("fes2050_supply", "fes_supply"):
        with pytest.raises(ValueError, match=r"write fleet[.]fes_supply "):
            check_renamed_keys({old: {"enabled": True}}, "X")
        assert fs.fes_supply_config({old: {"enabled": True, "thermal": {"x": 1}}}) == {}
    assert not hasattr(fs, "LEGACY_CONFIG_KEY")


# ───────────────────────────── thermal ─────────────────────────────

def _fes_thermal():
    return pd.DataFrame([
        {"technology": "CCGTs (non CHP)", "fuel_type": "CCGT", "capacity_mw": 900.0, "gsp": "Direct(NGET)", "data_source": "FES"},
        {"technology": "CCGTs (non CHP)", "fuel_type": "CCGT", "capacity_mw": 100.0, "gsp": "Direct(SHETL)", "data_source": "FES"},
        {"technology": "Nuclear", "fuel_type": "nuclear", "capacity_mw": 3000.0, "gsp": "Direct(NGET)", "data_source": "FES"},
        {"technology": "Biomass & Energy Crops (including CHP)", "fuel_type": "biomass", "capacity_mw": 500.0,
         "gsp": "Direct(NGET)", "data_source": "FES"},
        {"technology": "Biomass & Energy Crops (including CHP)", "fuel_type": "biomass", "capacity_mw": 50.0,
         "gsp": "Abingdon", "data_source": "FES"},
    ])


def _schedule(rows):
    frame = pd.DataFrame(rows)
    for col, default in (("basis", "named_project"), ("partial_of_direct", False), ("source", "test source"),
                         ("status", "declared"), ("lat", 52.0), ("lon", -1.0)):
        frame[col] = frame.get(col, default)
    return frame


def test_bb009_is_gas_ccs_in_2050():
    out = fs.remap_thermal_carriers(_fes_thermal(), {"CCGTs (non CHP)": "CCS Gas"})
    assert set(out.loc[out["technology"].eq("CCGTs (non CHP)"), "fuel_type"]) == {"CCS Gas"}
    assert not out["fuel_type"].eq("CCGT").any()


def test_schedule_conserves_direct_and_keeps_gsp_rows():
    fes = fs.remap_thermal_carriers(_fes_thermal(), {"CCGTs (non CHP)": "CCS Gas"})
    schedule = _schedule([
        {"station_name": "A", "carrier": "CCS Gas", "fes_source_carrier": "CCS Gas", "fes_region": "NGET", "capacity_mw": 600.0, "bus": "B1"},
        {"station_name": "B", "carrier": "CCS Gas", "fes_source_carrier": "CCS Gas", "fes_region": "NGET", "capacity_mw": 300.0, "bus": "B2"},
        {"station_name": "C", "carrier": "CCS Gas", "fes_source_carrier": "CCS Gas", "fes_region": "SHETL", "capacity_mw": 100.0, "bus": "B3"},
        {"station_name": "N", "carrier": "nuclear", "fes_source_carrier": "nuclear", "fes_region": "NGET", "capacity_mw": 3000.0, "bus": "B4"},
        {"station_name": "D", "carrier": "CCS Biomass", "fes_source_carrier": "biomass", "fes_region": "NGET",
         "capacity_mw": 400.0, "bus": "B5", "partial_of_direct": True},
    ])
    out = fs.apply_station_schedule(fes, schedule)
    by = out.groupby("fuel_type")["capacity_mw"].sum()
    assert by["CCS Gas"] == pytest.approx(1000.0)
    assert by["nuclear"] == pytest.approx(3000.0)
    assert by["CCS Biomass"] == pytest.approx(400.0)
    # BECCS is a partial take of the biomass Direct row; the GSP biomass row is untouched.
    assert by["biomass"] == pytest.approx(100.0 + 50.0)
    scheduled = out[out["data_source"].eq(fs.schedule_data_source(2050))]
    assert fs.schedule_data_source(2050) == "FES2050_station_schedule"     # the 2050 names do not move
    assert (scheduled["bus"] == scheduled[fs.DECLARED_BUS_COLUMN]).all()
    assert scheduled["gsp"].isna().all()
    assert not out.loc[~out["data_source"].eq(fs.schedule_data_source(2050))
                       & out["fuel_type"].isin(["CCS Gas", "nuclear"]), "capacity_mw"].gt(0).any()


def test_schedule_total_mismatch_fails():
    fes = fs.remap_thermal_carriers(_fes_thermal(), {"CCGTs (non CHP)": "CCS Gas"})
    schedule = _schedule([{"station_name": "N", "carrier": "nuclear", "fes_source_carrier": "nuclear",
                           "fes_region": "NGET", "capacity_mw": 2000.0, "bus": "B4"}])
    with pytest.raises(ValueError, match="rebuild the schedule"):
        fs.apply_station_schedule(fes, schedule)


def test_schedule_partial_take_cannot_exceed_direct():
    schedule = _schedule([{"station_name": "D", "carrier": "CCS Biomass", "fes_source_carrier": "biomass",
                           "fes_region": "NGET", "capacity_mw": 600.0, "bus": "B5", "partial_of_direct": True}])
    with pytest.raises(ValueError, match="takes"):
        fs.apply_station_schedule(_fes_thermal(), schedule)


def test_restore_declared_buses():
    sites = pd.DataFrame({"bus": ["X", "Y", "Z"], "capacity_mw": [1.0, 2.0, 3.0],
                          fs.DECLARED_BUS_COLUMN: ["A", np.nan, "Z"]})
    out = fs.restore_declared_buses(sites)
    assert out["bus"].tolist() == ["A", "Y", "Z"]


def test_offgrid_offshore_rejected():
    with pytest.raises(ValueError, match="off-grid"):
        fs.check_offgrid_excluded(pd.DataFrame({"technology": ["Offshore-Wind (off-Grid)"]}), "t")


def test_station_list_file():
    stations = bss.load_station_list(str(STATION_LIST))
    assert stations["source"].astype(str).str.len().gt(0).all()
    assert stations["dates_source"].astype(str).str.len().gt(0).all()
    nuclear = stations[stations["carrier"].eq("nuclear")]
    # ES1 2050: large 6,680 (HPC + SZC at TEC) and SMR 7,520.
    large = nuclear[nuclear["station_name"].isin(["Hinkley Point C", "Sizewell C"])]["capacity_mw"].astype(float).sum()
    assert large == pytest.approx(6680.0)
    # TEC dates: first full calendar year at the cumulative capacity.
    years = stations.set_index("station_name")["from_year"]
    assert years["Hinkley Point C"] == 2031 and years["Sizewell C"] == 2037
    assert years["Net Zero Teesside Power"] == 2028
    assert stations.loc[stations["carrier"].eq("CCS Biomass"), "allocation"].eq("es1_capacity").all()


def _built_2050_schedule():
    return bss.build_schedule(2050, str(FES_2025), str(STATION_LIST), pd.read_csv(ES1_2050))


@pytest.mark.skipif(not (FES_2025.exists() and ES1_2050.exists()), reason="FES 2025 resources not present")
def test_2050_schedule_matches_fes_direct_totals():
    schedule = _built_2050_schedule()
    check_schedule_totals(schedule, fes_direct_totals(str(FES_2025), 2050))
    by = schedule.groupby(["carrier", "fes_region"])["capacity_mw"].sum()
    assert by[("nuclear", "NGET")] == pytest.approx(14200.0)
    assert by[("CCS Gas", "NGET")] + by[("CCS Gas", "SHETL")] == pytest.approx(22205.0)
    assert by[("H2", "NGET")] + by[("H2", "SHETL")] + by[("H2", "SPTL")] == pytest.approx(19097.0)
    assert by[("CCS Biomass", "NGET")] == pytest.approx(2662.0)


@pytest.mark.skipif(not (BASE_NETWORK.exists() and FES_2025.exists() and ES1_2050.exists()),
                    reason="HT50 base network or FES resources not built")
def test_schedule_and_landing_buses_exist_in_the_base_network():
    import netCDF4
    with netCDF4.Dataset(BASE_NETWORK) as ds:
        buses = set(np.asarray(ds.variables["buses_i"][:]).astype(str))
    schedule = _built_2050_schedule()
    assert set(schedule["bus"]) <= buses
    assert set(pd.read_csv(LANDING)["landing_bus"]) <= buses


# ───────────────────────────── offshore landing ─────────────────────────────

def _landing_tables(tmp_path):
    landing = pd.DataFrame([
        {"landing_bus": "L1", "fes_region": "NGET", "capacity_mw": 700.0, "lat": 53.0, "lon": 0.0,
         "source": "s", "landing_basis": "b"},
        {"landing_bus": "L2", "fes_region": "NGET", "capacity_mw": 300.0, "lat": 51.0, "lon": 1.0,
         "source": "s", "landing_basis": "b"},
        {"landing_bus": "L3", "fes_region": "SHETL", "capacity_mw": 500.0, "lat": 57.5, "lon": -1.8,
         "source": "s", "landing_basis": "b"},
    ])
    projects = pd.DataFrame([{"project": "Farm One", "landing_bus": "L1", "capacity_mw": 700.0}])
    lp, pp = tmp_path / "landing.csv", tmp_path / "projects.csv"
    landing.to_csv(lp, index=False)
    projects.to_csv(pp, index=False)
    return lp, pp


def _renewable_sites():
    return pd.DataFrame([
        {"technology": "wind_offshore", "capacity_mw": 600.0, "bus": "X1", "gsp": "Direct(NGET)", "data_source": "FES_Direct"},
        {"technology": "wind_offshore", "capacity_mw": 400.0, "bus": "X2", "gsp": "Direct(NGET)", "data_source": "FES_Direct"},
        {"technology": "wind_offshore", "capacity_mw": 500.0, "bus": "X3", "gsp": "Direct(SHETL)", "data_source": "FES_Direct"},
        {"technology": "wind_offshore", "capacity_mw": 50.0, "bus": "G1", "gsp": "Abingdon", "data_source": "FES"},
        {"technology": "solar_pv", "capacity_mw": 80.0, "bus": "G1", "gsp": "Direct(NGET)", "data_source": "FES_Direct"},
    ])


def test_landing_replaces_direct_offshore(tmp_path):
    lp, pp = _landing_tables(tmp_path)
    buses = pd.DataFrame(index=["L1", "L2", "L3", "G1"])
    sites = pd.DataFrame({"site_name": ["Farm One", "Farm Far"], "lat": [53.1, 57.4], "lon": [0.1, -1.9]})
    out = fs.apply_fes_offshore_landing(_renewable_sites(), lp, buses, projects_path=pp,
                                            profile_columns=["Farm One (REPD)", "Farm Far"], profile_sites=sites)
    landed = out[out["data_source"].eq(fs.landing_data_source(2050))]
    assert landed["capacity_mw"].sum() == pytest.approx(1500.0)
    assert set(landed["bus"]) == {"L1", "L2", "L3"}
    assert (landed["bus"] == landed[fs.DECLARED_BUS_COLUMN]).all()
    # A landing project with a profile wins; otherwise the nearest profiled site.
    assert landed.set_index("bus").loc["L1", "source_site_name"] == "Farm One (REPD)"
    assert landed.set_index("bus").loc["L3", "source_site_name"] == "Farm Far"
    # GSP offshore and Direct solar are untouched.
    assert out.loc[out["gsp"].eq("Abingdon"), "capacity_mw"].sum() == pytest.approx(50.0)
    assert out.loc[out["technology"].eq("solar_pv"), "capacity_mw"].sum() == pytest.approx(80.0)


def test_landing_regional_total_mismatch_fails(tmp_path):
    lp, pp = _landing_tables(tmp_path)
    sites = _renewable_sites()
    sites.loc[0, "capacity_mw"] = 900.0
    with pytest.raises(ValueError, match="NGET"):
        fs.apply_fes_offshore_landing(sites, lp, pd.DataFrame(index=["L1", "L2", "L3"]), projects_path=pp)


def test_landing_bus_absent_fails(tmp_path):
    lp, pp = _landing_tables(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        fs.apply_fes_offshore_landing(_renewable_sites(), lp, pd.DataFrame(index=["L1", "L2"]), projects_path=pp)


def test_committed_landing_table_totals():
    landing = pd.read_csv(LANDING)
    by = landing.groupby("fes_region")["capacity_mw"].sum()
    assert by["NGET"] == pytest.approx(72335.65, abs=0.5)
    assert by["SHETL"] == pytest.approx(23899.5, abs=0.5)
    assert by["SPTL"] == pytest.approx(6736.0, abs=0.5)
    assert not landing["landing_bus"].duplicated().any()


# ───────────────────────────── node buses ─────────────────────────────

def test_resolve_node_bus():
    buses = pd.DataFrame({"v_nom": [132.0, 400.0, 400.0], "carrier": ["AC", "AC", "AC"]},
                         index=["CARR11", "CARR41", "HEYS41"])
    assert fs.resolve_node_bus("CARR11", buses) == "CARR11"
    assert fs.resolve_node_bus("CARR12", buses) == "CARR11"      # same site, same voltage digit
    assert fs.resolve_node_bus("HEYS11", buses) == "HEYS41"      # same site, highest voltage
    assert fs.resolve_node_bus("ZZZZ11", buses) is None
    assert fs.resolve_node_bus("Direct(NGET)", buses) is None
    assert fs.resolve_node_bus(np.nan, buses) is None


# ───────────────────────────── storage ─────────────────────────────

def _targets():
    return {"storage": [
        {"carrier": "Battery", "capacity_mw": 1000.0, "energy_gwh": 1.5, "generation_gwh": 85.0, "consumption_gwh": 100.0},
        {"carrier": "CAES", "capacity_mw": 400.0, "energy_gwh": 0.4, "generation_gwh": 60.0, "consumption_gwh": 100.0},
        {"carrier": "LAES", "capacity_mw": 600.0, "energy_gwh": 2.4, "generation_gwh": 55.0, "consumption_gwh": 100.0},
        {"carrier": "Pumped Storage Hydroelectricity", "capacity_mw": 200.0, "energy_gwh": 3.4,
         "generation_gwh": 75.0, "consumption_gwh": 100.0},
    ]}


def test_fes_storage_by_type(tmp_path):
    fes = pd.DataFrame([
        {"Building Block ID Number": "Srg_BB001", "FES Pathway": "Holistic Transition", "GSP": "Carrington", "Node ID": "CARR11", "2050": 500.0},
        {"Building Block ID Number": "Srg_BB001", "FES Pathway": "Holistic Transition", "GSP": "Direct(NGET)", "Node ID": "Direct(NGET)", "2050": 300.0},
        {"Building Block ID Number": "Srg_BB002", "FES Pathway": "Holistic Transition", "GSP": "Nowhere", "Node ID": "ZZZZ11", "2050": 100.0},
        {"Building Block ID Number": "Srg_BB003", "FES Pathway": "Holistic Transition", "GSP": "Direct(NGET)", "Node ID": "Direct(NGET)", "2050": 200.0},
        {"Building Block ID Number": "Srg_BB004", "FES Pathway": "Holistic Transition", "GSP": "Direct(NGET)", "Node ID": "Direct(NGET)", "2050": 1000.0},
        {"Building Block ID Number": "Srg_BB004", "FES Pathway": "Electric Engagement", "GSP": "Direct(NGET)", "Node ID": "Direct(NGET)", "2050": 9999.0},
    ])
    path = tmp_path / "fes.csv"
    fes.to_csv(path, index=False)
    buses = pd.DataFrame({"v_nom": [132.0], "carrier": ["AC"], "lat": [53.4], "lon": [-2.4]}, index=["CARR11"])
    repd = pd.DataFrame([
        {"site_name": "Bat A", "technology": "Battery", "power_mw": 50.0, "lat": 52.0, "lon": -1.0},
        {"site_name": "Bat B", "technology": "Battery", "power_mw": 150.0, "lat": 51.0, "lon": 0.0},
        {"site_name": "Dinorwig", "technology": "Pumped Storage Hydroelectricity", "power_mw": 1728.0, "lat": 53.1, "lon": -4.1},
    ])
    table = fs.build_fes_storage(path, _targets(), {"durations_h": {"CAES": 20.0}}, buses, repd,
                                     round_trip_efficiencies={"Battery": 0.8464})
    summary = fs.storage_summary(table)
    assert summary.loc[["Battery", "Domestic Battery"], "power_mw"].sum() == pytest.approx(1000.0)
    assert summary.loc["CAES", "power_mw"] == pytest.approx(400.0)
    assert summary.loc["LAES", "power_mw"] == pytest.approx(600.0)
    assert summary.loc["CAES", "duration_h"] == pytest.approx(20.0)          # FES Economics, not ES1's 1 h
    assert summary.loc["LAES", "duration_h"] == pytest.approx(4.0)           # ES1 energy / power
    assert summary.loc["Pumped Storage Hydroelectricity", "duration_h"] == pytest.approx(17.0)
    battery = table[table["technology"].eq("Battery")]
    assert (battery["eta_charge"] ** 2).round(4).eq(0.8464).all()            # configured round trip wins
    caes = table[table["technology"].eq("CAES")]
    assert (caes["eta_charge"] ** 2).round(4).eq(0.6).all()                  # else the ES1-implied one
    assert "CARR11" in set(table["bus"].dropna())                             # GSP row at its FES node


# ───────────────────────────── hydrogen ─────────────────────────────

def _h2_targets():
    return {"hydrogen": {
        "annual_twh_calendar": {"demand": {"Industrial": 30.0, "Aviation": 20.0, "Power Generation": 10.0}},
        "non_power_demand_twh": 52.0,
        "electrolysis_capacity_gw_electric": 2.0, "electrolysis_efficiency": 0.8,
        "h2_turbines": {"efficiency_electric_per_h2": 0.4, "capacity_mw": 1000.0},
        "storage_twh": 1.0,
        "gas_reforming_ccs": {"capacity_gw_h2": 6.0, "output_twh": 40.0, "gas_feedstock_twh": 50.0},
    }}


def test_hydrogen_quantities_bases():
    from scripts.hydrogen.fes_hydrogen import hydrogen_quantities
    assert hydrogen_quantities(_h2_targets(), "calendar")["non_power_demand_mwh"] == pytest.approx(50e6)
    assert hydrogen_quantities(_h2_targets(), "fiscal")["non_power_demand_mwh"] == pytest.approx(52e6)
    with pytest.raises(ValueError):
        hydrogen_quantities(_h2_targets(), "annual")


def test_fes_hydrogen_system():
    import pypsa
    from scripts.hydrogen import fes_hydrogen as fh
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=3, freq="h"))
    for name, x in (("CARR11", 0.0), ("HEYS41", 1.0)):
        n.add("Bus", name, v_nom=400.0, x=x, y=0.0, carrier="AC")
    n.add("Generator", "H2_gen_1", bus="HEYS41", carrier="H2", p_nom=1000.0)
    fes = pd.DataFrame([
        {"Building Block ID Number": "Dem_BB009", "FES Pathway": "Holistic Transition", "GSP": "Carrington", "Node ID": "CARR11", "2050": 1500.0},
        {"Building Block ID Number": "Dem_BB009", "FES Pathway": "Holistic Transition", "GSP": "Heysham", "Node ID": "HEYS11", "2050": 500.0},
    ])
    audit = fh.add_fes_hydrogen_system(n, fes, _h2_targets(), {"gas": 20.0, "carbon": 100.0},
                                           config={"non_power_demand_basis": "calendar"})
    links = n.links
    assert links.loc[links.carrier.eq("electrolysis"), "p_nom"].sum() == pytest.approx(2000.0)
    assert links.loc[links.carrier.eq("electrolysis"), "efficiency"].eq(0.8).all()
    turbine = links[links.carrier.eq("H2_turbine")].iloc[0]
    assert turbine["p_nom"] * turbine["efficiency"] == pytest.approx(1000.0)
    assert "H2_gen_1" not in n.generators.index
    assert n.stores.loc[fh.STORE, "e_nom"] == pytest.approx(1e6)
    assert n.stores.loc[fh.STORE, "standing_loss"] == 0.0
    assert n.loads.loc[fh.NON_POWER_LOAD, "p_set"] == pytest.approx(50e6 / 8760)
    reformer = links.loc[fh.REFORMER]
    assert reformer["bus0"] == fh.GAS_BUS and reformer["bus1"] == fh.H2_BUS
    assert reformer["efficiency"] == pytest.approx(0.8)
    assert reformer["p_nom"] * reformer["efficiency"] == pytest.approx(6000.0)
    assert n.generators.loc[fh.GAS_SUPPLY, "marginal_cost"] == pytest.approx(20.0)
    # residual 0.183 x 5% x 100 + captured 0.183 x 95% x 34.5 per MWh gas
    assert reformer["marginal_cost"] == pytest.approx(0.183 * 0.05 * 100 + 0.183 * 0.95 * 34.5)
    assert audit["feasibility"]["supply_max_twh_h2"] >= audit["feasibility"]["demand_twh_h2"]


def test_hydrogen_paper_balance_infeasible_raises():
    import pypsa
    from scripts.hydrogen import fes_hydrogen as fh
    targets = _h2_targets()
    targets["hydrogen"]["gas_reforming_ccs"]["capacity_gw_h2"] = 0.0
    n = pypsa.Network()
    n.add("Bus", "CARR11", v_nom=400.0, x=0.0, y=0.0, carrier="AC")
    n.add("Generator", "H2_gen_1", bus="CARR11", carrier="H2", p_nom=100.0)
    fes = pd.DataFrame([{"Building Block ID Number": "Dem_BB009", "FES Pathway": "Holistic Transition",
                         "GSP": "Carrington", "Node ID": "CARR11", "2050": 2000.0}])
    with pytest.raises(ValueError, match="infeasible"):
        fh.add_fes_hydrogen_system(n, fes, targets, {"gas": 20.0, "carbon": 100.0},
                                       config={"gas_reforming_ccs": {"enabled": False}})


# ───────────────────────────── interconnectors and clustering ─────────────────────────────

def test_interconnector_scaling_reads_the_given_fes_file(tmp_path):
    from scripts.interconnectors.add_to_network import scale_interconnectors_to_fes
    fes = pd.DataFrame([
        {"Building Block ID Number": "Gen_BB022", "FES Pathway": "Holistic Transition", "2050": 20600.0},
        {"Building Block ID Number": "Gen_BB022", "FES Pathway": "Holistic Transition", "2050": 1200.0},
    ])
    path = tmp_path / "FES_2025_data.csv"
    fes.to_csv(path, index=False)
    cables = pd.DataFrame({"capacity_mw": [1000.0, 3000.0], "name": ["a", "b"]})
    out = scale_interconnectors_to_fes(cables, 2050, "Holistic Transition", fes_data_path=str(path))
    assert out["capacity_mw"].sum() == pytest.approx(21800.0)
    assert out["capacity_mw"].iloc[1] / out["capacity_mw"].iloc[0] == pytest.approx(3.0)


def test_cluster_defaults_cover_seasonal_and_provenance_columns():
    from scripts.network_clustering.boundary_busmap import LINE_STRATEGIES
    from scripts.network_clustering.cluster_network import NetworkClusterer
    defaults = NetworkClusterer().default_strategies["line"]
    for key, value in LINE_STRATEGIES.items():
        assert defaults.get(key) == value, key


# ───────────────────────────── scenario entries ─────────────────────────────

def _scenarios():
    import sys
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config
    return load_config(ROOT / "config", include_archive=True)["scenarios"]   # the gate weeks are archived


def test_ht50_scenario_entries():
    sc = _scenarios()
    parent = sc["FES25_HT_2050"]
    assert parent["modelled_year"] == 2050 and parent["FES_year"] == 2025
    assert parent["FES_scenario"] == "Holistic Transition" and parent["renewables_year"] == 2013
    assert "scenario_source" not in parent and "CP30_pathway" not in parent   # the generic FES path
    assert parent["fleet"]["fes_supply"]["enabled"] is True and "fes2050_supply" not in parent
    assert "fes_supply" not in parent
    assert parent["demand"]["disaggregation"]["fes"]["enabled"] is True
    assert parent["hydrogen"]["fes"]["enabled"] is True
    # Calibrated scalars come from data/FES/fes_year_parameters.csv for (2050, HT).
    assert parent["market"]["hydrogen_supply_price"]["gbp_per_mwh_h2"] == 64.0
    assert parent["renewables"]["performance_factors"]["factors"]["wind_offshore"] == 0.9261
    assert parent["interconnectors"]["pricing"]["foreign_price_source"] == "tyndp"
    assert parent["interconnectors"]["pricing"].get("source", "derived") != "cp30"
    assert parent["network"]["clustering"]["method"] == "custom"
    sec = parent["network"]["transmission"]["outage_schedule"]["secured_capability"]
    assert sec["capability_year"] == 2050 and sec["hold_last_year"] is True
    assert sec["branch_scaling"] == {"enabled": True, "margin": 1.25}
    assert parent["marginal_costs"]["future_costs"]["cases"]["fes2025_ht"]["source_file"].endswith(
        "fes2025_future_costs_2050.json")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = [f"FES25_HT_2050_{m}_market" for m in names]                  # pattern FES25_HT_2050_{month}_market
    central = [f"FES25_HT_2050_{m}_central_dispatch" for m in names]   # pattern FES25_HT_2050_{month}_central_dispatch
    for rid in months + central + ["HT50_1week_gate", "HT50_1week_gate_central_dispatch_fe"]:
        assert sc[rid]["network_from"] == "FES25_HT_2050", rid
        assert sc[rid]["solve_period"]["enabled"] is True
    for rid in central:
        assert sc[rid]["market"]["dispatch"] == "central"
        assert sc[rid]["market"]["wholesale"]["copperplate_mode"] == "none"
        assert sc[rid]["market"]["wholesale"]["forecast_error"]["enabled"] is False
    assert sc["HT50_1week_gate"]["solve_period"]["start"] == "2050-01-10 00:00"


def test_ht50_children_differ_from_parent_only_in_solve_time_keys():
    # The live scenarios solved on the FES25_HT_2050 network (its monthly families). The archived gate
    # weeks are frozen copies of an earlier parent, so they are not held to today's.
    import sys
    sys.path.insert(0, str(ROOT / "config"))
    from config_loader import load_config
    sc = load_config(ROOT / "config")["scenarios"]
    solve_time = {"network_from", "description", "solve_period", "_scenario_id", "market", "wholesale_from"}
    # Solve-time keys inside build-time blocks: the solver settings and the BM bid settings.
    solve_time_paths = (("solve", "solver"), ("support", "bm_bids"))

    def build_time(cfg):
        out = {k: v for k, v in cfg.items() if k not in solve_time}
        for top, key in solve_time_paths:
            if isinstance(out.get(top), dict):
                out[top] = {k: v for k, v in out[top].items() if k != key}
        return out

    parent = build_time(sc["FES25_HT_2050"])
    children = [k for k in sc if k != "FES25_HT_2050" and sc[k].get("network_from") == "FES25_HT_2050"]
    assert children, "no scenario is solved on the FES25_HT_2050 network"
    for rid in children:
        assert build_time(sc[rid]) == parent, rid


def test_direct_electrolysis_follows_its_licence_area():
    from scripts.hydrogen.fes_hydrogen import licence_area, place_electrolysers
    assert licence_area("SSEN (SHEPD)") == "SHETL" and licence_area("SPEN") == "SPTL"
    assert licence_area("SPEN Manweb") == "NGET" and licence_area("UKPN Eastern") == "NGET"
    sites = pd.DataFrame([
        {"GSP": "Peterhead", "bus": "PEHE11", "area": "SHETL", "mw": 100.0},
        {"GSP": "Carrington", "bus": "CARR11", "area": "NGET", "mw": 300.0},
        {"GSP": "Heysham", "bus": "HEYS11", "area": "NGET", "mw": 100.0},
        {"GSP": "Direct(SHETL)", "bus": np.nan, "area": "SHETL", "mw": 50.0},
        {"GSP": "Direct(NGET)", "bus": np.nan, "area": "NGET", "mw": 200.0},
        {"GSP": "Direct(SPTL)", "bus": np.nan, "area": "SPTL", "mw": 40.0},   # no SPTL node: spread over all
    ])
    by = place_electrolysers(sites)
    assert by.sum() == pytest.approx(790.0)
    assert by["PEHE11"] == pytest.approx(100.0 + 50.0 + 40.0 * 100 / 500)
    assert by["CARR11"] == pytest.approx(300.0 + 200.0 * 0.75 + 40.0 * 300 / 500)


def test_hydrogen_offtake_spreads_over_a_leap_year():
    """2040 has 8,784 hours: the flat offtake is the annual volume over them, not over 8,760."""
    import pypsa
    from scripts.hydrogen import fes_hydrogen as fh
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2040-01-01", periods=3, freq="h"))
    n.add("Bus", "CARR11", v_nom=400.0, x=0.0, y=0.0, carrier="AC")
    n.add("Generator", "H2_gen_1", bus="CARR11", carrier="H2", p_nom=1000.0)
    fes = pd.DataFrame([{"Building Block ID Number": "Dem_BB009", "FES Pathway": "Holistic Transition",
                         "GSP": "Carrington", "Node ID": "CARR11", "2040": 2000.0}])
    audit = fh.add_fes_hydrogen_system(n, fes, _h2_targets(), {"gas": 20.0, "carbon": 100.0},
                                       config={"non_power_demand_basis": "calendar"}, year=2040)
    assert n.loads.loc[fh.NON_POWER_LOAD, "p_set"] == pytest.approx(50e6 / 8784)
    assert n.generators.loc[fh.GAS_SUPPLY, "data_source"] == "FES2040_hydrogen"
    assert audit["feasibility"]["hourly_flat_offtake_mw_h2"] == pytest.approx(50e6 / 8784)


# ───────────────────────────── gas CCS split (stage 5) ─────────────────────────────

def _es1_lines(ccs_mw, ccgt_mw):
    rows = [{"type": "CCS Gas", "subtype": "CCS Gas", "carrier": "CCS Gas", "quantity": "capacity_mw", "value_std": ccs_mw},
            {"type": "Gas", "subtype": "OCGT", "carrier": "gas", "quantity": "capacity_mw", "value_std": 2462.63},
            {"type": "Gas", "subtype": "CCS Gas", "carrier": "CCS Gas", "quantity": "generation_gwh", "value_std": 9.0}]
    if ccgt_mw:
        rows.append({"type": "Gas", "subtype": "CCGT", "carrier": "gas", "quantity": "capacity_mw", "value_std": ccgt_mw})
    return pd.DataFrame(rows)


def test_es1_gas_ccs_share():
    # HT 2040: Gen_BB009 25,033 MW = ES1 CCS gas 19,625 + unabated CCGT 5,408; OCGTs are another block.
    split = fs.es1_gas_ccs_share(_es1_lines(19625.0, 5408.0))
    assert split["share"] == pytest.approx(19625.0 / 25033.0)
    assert split["unabated_mw"] == pytest.approx(5408.0)
    assert fs.es1_gas_ccs_share(_es1_lines(22205.0, 0.0))["share"] == 1.0     # HT 2050: all CCS
    with pytest.raises(ValueError, match="undefined"):
        fs.es1_gas_ccs_share(_es1_lines(0.0, 0.0))


def test_gas_split_at_share_one_is_the_remap():
    fes = _fes_thermal()
    split = fs.split_gas_ccs(fes, {"share": 1.0})
    pd.testing.assert_frame_equal(split, fs.remap_thermal_carriers(fes, {"CCGTs (non CHP)": "CCS Gas"}))


def test_gas_split_by_es1_share():
    cfg = {"technology_carriers": {"CCGTs (non CHP)": "CCS Gas"},
           "gas_ccs_split": {"enabled": True, "technology": "CCGTs (non CHP)", "ccs_carrier": "CCS Gas",
                             "unabated_carrier": "CCGT"}}
    out = fs.apply_fes_thermal(_fes_thermal(), cfg, year=2040, es1_lines=_es1_lines(750.0, 250.0))
    by = out.groupby("fuel_type")["capacity_mw"].sum()
    assert by["CCS Gas"] == pytest.approx(750.0) and by["CCGT"] == pytest.approx(250.0)
    # Each GSP keeps its share of both carriers.
    shetl = out[out["gsp"].eq("Direct(SHETL)")].set_index("fuel_type")["capacity_mw"]
    assert shetl["CCS Gas"] == pytest.approx(75.0) and shetl["CCGT"] == pytest.approx(25.0)
    with pytest.raises(ValueError, match="ES1 lines"):
        fs.apply_fes_thermal(_fes_thermal(), cfg, year=2040)


# ───────────────────────────── station schedule by year (stage 5) ─────────────────────────────

def _station_list(tmp_path, rows):
    base = {"es1_carrier": "", "share_weight_mw": np.nan, "capacity_mw": np.nan, "lat": 52.0, "lon": -1.0,
            "basis": "declared_site", "from_year": np.nan, "to_year": np.nan, "dates_source": "test",
            "source": "test {direct_mw:,.0f} MW in {year}", "status": "declared", "notes": ""}
    path = tmp_path / "stations.csv"
    pd.DataFrame([{**base, **r} for r in rows]).to_csv(path, index=False)
    return str(path)


def _fes_blocks(tmp_path, nuclear_mw, year=2040):
    path = tmp_path / "fes.csv"
    pd.DataFrame([{"Building Block ID Number": "Gen_BB020", "FES Pathway": "Holistic Transition",
                   "GSP": "Direct(NGET)", str(year): nuclear_mw}]).to_csv(path, index=False)
    return str(path)


def test_schedule_filters_by_year_and_shares_the_rest(tmp_path):
    stations = _station_list(tmp_path, [
        {"station_name": "Big", "carrier": "nuclear", "fes_region": "NGET", "allocation": "fixed", "capacity_mw": 3000.0,
         "bus": "B1", "from_year": 2031},
        {"station_name": "Late", "carrier": "nuclear", "fes_region": "NGET", "allocation": "fixed", "capacity_mw": 3000.0,
         "bus": "B2", "from_year": 2045},
        {"station_name": "Shut", "carrier": "nuclear", "fes_region": "NGET", "allocation": "fixed", "capacity_mw": 900.0,
         "bus": "B5", "to_year": 2035},
        {"station_name": "SMR A", "carrier": "nuclear", "fes_region": "NGET", "allocation": "share", "capacity_mw": 1410.0,
         "bus": "B3"},
        {"station_name": "SMR B", "carrier": "nuclear", "fes_region": "NGET", "allocation": "share", "capacity_mw": 470.0,
         "bus": "B4"},
    ])
    out = bss.build_schedule(2040, _fes_blocks(tmp_path, 5000.0), stations)
    mw = out.set_index("station_name")["capacity_mw"]
    assert set(mw.index) == {"Big", "SMR A", "SMR B"}                  # Late not open, Shut closed
    assert mw["Big"] == 3000.0
    assert mw["SMR A"] == pytest.approx(1500.0) and mw["SMR B"] == pytest.approx(500.0)
    assert out["capacity_mw"].sum() == pytest.approx(5000.0)
    assert out.set_index("station_name").loc["SMR A", "source"] == "test 5,000 MW in 2040"


def test_schedule_scales_fixed_down_and_refuses_a_shortfall(tmp_path):
    stations = _station_list(tmp_path, [
        {"station_name": "Big", "carrier": "nuclear", "fes_region": "NGET", "allocation": "fixed", "capacity_mw": 3000.0,
         "bus": "B1"},
    ])
    out = bss.build_schedule(2040, _fes_blocks(tmp_path, 2000.0), stations)
    assert out["capacity_mw"].sum() == pytest.approx(2000.0)
    with pytest.raises(ValueError, match="none takes the rest"):
        bss.build_schedule(2040, _fes_blocks(tmp_path, 4000.0), stations)


def test_schedule_refuses_a_region_with_no_station(tmp_path):
    stations = _station_list(tmp_path, [
        {"station_name": "Big", "carrier": "nuclear", "fes_region": "SHETL", "allocation": "fixed", "capacity_mw": 3000.0,
         "bus": "B1"},
    ])
    with pytest.raises(ValueError, match="no open station"):
        bss.build_schedule(2040, _fes_blocks(tmp_path, 2000.0), stations)
