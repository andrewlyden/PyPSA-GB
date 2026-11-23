"""FES 2050 demand components and consumer flexibility (task Y6a), on synthetic inputs."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.CP30 import energy_validation as ev
from scripts.demand import demand_carriers as dc
from scripts.demand import fes_demand as fd

YEAR = pd.date_range("2050-01-01", periods=8760, freq="h")


def _temperature(index=YEAR) -> pd.Series:
    """Synthetic GB temperature: 4 C mean in January, 16 C in July, a 4 C daily swing."""
    doy = index.dayofyear.to_numpy()
    t = 10.0 - 6.0 * np.cos(2 * np.pi * (doy - 15) / 365.0) + 2.0 * np.sin(2 * np.pi * (index.hour.to_numpy() - 9) / 24)
    return pd.Series(t, index=index)


def _targets() -> dict:
    """The slice of the Y1 targets JSON the module reads (HT 2050 magnitudes, GWh)."""
    def item(fy, cal, row=0):
        return {"fy_gwh": fy, "calendar_gwh": cal, "excel_row": row}

    bb = {"Dem_BB004": item(53878, 53590), "Dem_BB005": item(25021, 24831),
          "Dem_BB006": item(87226, 87400), "Dem_BB007": item(38850, 38977)}
    annual = {"residential_evs": item(57590, 57700, 49), "commercial_evs": item(68530, 68677, 104),
              "residential_district_heat": item(19600, 19432, 59), "data_centres": item(65320, 65013, 129)}
    peaks = {"GBFES Peak Customer Demand: Total Consumption plus Losses": 120.05, "Residential Losses": 4.466,
             "Industrial Losses": 1.839, "Commercial Losses": 4.393, "Residential EVs at Peak": 1.4,
             "Commercial EVs at Peak": 7.6, "Residential Heat Pumps at Peak": 22.243,
             "Commercial Heat Pumps at Peak": 7.371, "Residential District Heat at Peak": 5.898,
             "Commercial District Heat at Peak": 1.426, "Industrial District Heat at Peak": 0.216}
    return {"demand": {"annual": annual,
                       "totals": {"consumer_excluding_electrolysis": {"fy_gwh": 567384, "calendar_gwh": 566294},
                                  "system_excluding_electrolysis": {"fy_gwh": 616513, "calendar_gwh": 615324}},
                       "peak_gw": {k: {"gw": v, "excel_row": 0} for k, v in peaks.items()}},
            "demand_components": {"building_blocks": bb}, "daccs_gwh": 17828.38}


def _flex() -> dict:
    gw = {"smart_charging_at_peak": 10.09, "v2g_at_peak": 40.6, "v2g_max": 81.21, "residential_appliance_dsr": 6.3,
          "ic_dsr": 3.9, "res_hybrid_hp_flex": 0.192, "res_thermal_storage_flex": 6.578,
          "res_district_heat_flex": 4.882, "nonres_hybrid_hp_flex": 0.043, "nonres_thermal_storage_flex": 0.621,
          "nonres_district_heat_flex": 1.359, "unconstrained_heat_peak": 65.976, "total_consumer_flex": 81.56,
          "res_storage_heater_flex": 4.62, "nonres_storage_heater_flex": 2.377}
    return {k: {"gw": v, "excel_row": 0, "detail": k} for k, v in gw.items()}


@pytest.fixture(scope="module")
def built():
    cfg = fd.resolve_config({"enabled": True})
    energies = fd.component_energies(_targets(), "calendar")
    peaks = fd.peak_targets(_targets(), _flex(), energies["loss_factor"])
    temp = _temperature()
    comps = fd.build_components(YEAR, temp, energies, peaks, 38.7e6, cfg)
    return cfg, energies, peaks, temp, comps


def _network(buses=("A", "B", "C"), base_share=(0.5, 0.3, 0.2), total_gwh=567384.0) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(YEAR)
    for i, b in enumerate(buses):
        n.add("Bus", b, carrier="AC", x=-2.0 + i, y=52.0 + i)
    n.add("Bus", "H2", carrier="H2")
    shape = 1.0 + 0.3 * np.cos(2 * np.pi * (YEAR.dayofyear.to_numpy() - 15) / 365.0)
    shape = shape / shape.sum() * total_gwh * 1e3
    for b, s in zip(buses, base_share):
        n.add("Load", f"base {b}", bus=b, carrier="electricity", p_set=pd.Series(shape * s, index=YEAR))
    n.add("Load", "H2 offtake", bus="H2", carrier="H2_demand", p_set=100.0)
    return n


def _allocation(buses=("A", "B", "C")) -> pd.DataFrame:
    keys = ["heat_pumps", "electric_vehicles", "data_centres", "district_heat", "v2g", "ic_dsr",
            "residential_dsr", "storage_heaters", "daccs"]
    table = pd.DataFrame({k: [0.4, 0.35, 0.25] for k in keys}, index=list(buses))
    table["daccs"] = [0.0, 0.0, 1.0]
    table.index.name = "bus"
    return table


# ───────────────────────────── energies and targets ─────────────────────────────

def test_component_energies_close_on_consumer_total():
    e = fd.component_energies(_targets(), "calendar", "consumer")
    assert e["loss_factor"] == 1.0
    assert e["heat_pumps_gwh"] == pytest.approx(53590 + 24831)
    assert e["evs_gwh"] == pytest.approx(87400 + 38977)
    parts = e["heat_pumps_gwh"] + e["evs_gwh"] + e["district_heat_gwh"] + e["data_centres_gwh"] + e["residual_gwh"]
    assert parts == pytest.approx(e["demand_total_gwh"]) and e["demand_total_gwh"] == 566294
    assert e["residual_gwh"] == pytest.approx(566294 - 78421 - 126377 - 19432 - 65013)
    fiscal = fd.component_energies(_targets(), "fiscal", "consumer")
    assert fiscal["demand_total_gwh"] == 567384
    with pytest.raises(ValueError):
        fd.component_energies(_targets(), "monthly")
    with pytest.raises(ValueError):
        fd.component_energies(_targets(), "calendar", "meter")


def test_system_basis_uplifts_every_line_by_one_loss_factor():
    """Default basis = CP30 2030's: ED1 system demand excl. electrolysis, incl. losses."""
    e = fd.component_energies(_targets())
    assert e["demand_basis"] == "system"
    assert e["demand_total_gwh"] == pytest.approx(615324)
    lf = 615324 / 566294
    assert e["loss_factor"] == pytest.approx(lf)
    assert e["heat_pumps_gwh"] == pytest.approx((53590 + 24831) * lf)
    assert e["fes_line_gwh"]["heat_pumps"] == pytest.approx(53590 + 24831)
    parts = e["heat_pumps_gwh"] + e["evs_gwh"] + e["district_heat_gwh"] + e["data_centres_gwh"] + e["residual_gwh"]
    assert parts == pytest.approx(e["demand_total_gwh"])
    assert e["daccs_gwh"] == 17828.38                    # transmission-connected: no uplift
    p = fd.peak_targets(_targets(), _flex(), e["loss_factor"])
    assert p["ev_at_peak_gw"] == pytest.approx(9.0 * lf)
    assert p["ed1_peak_model_basis_gw"] == 120.05
    assert p["v2g_max_gw"] == 81.21                      # capacities not uplifted


def test_peak_targets_consumer_basis():
    p = fd.peak_targets(_targets(), _flex())
    assert p["ed1_peak_consumer_gw"] == pytest.approx(120.05 - 4.466 - 1.839 - 4.393)
    assert p["ev_at_peak_gw"] == pytest.approx(9.0)
    assert p["hp_flex_at_peak_gw"] == pytest.approx(0.192 + 6.578 + 0.043 + 0.621)
    assert p["dh_flex_at_peak_gw"] == pytest.approx(4.882 + 1.359)


def test_flx1_reader_returns_positive_gw_and_rows():
    rows = []
    for key, detail in fd.FLX1_ITEMS.items():
        rows.append({"Pathway": "Holistic Transition", "Detail": detail, "2050": -1.5, "excel_row": 7})
    rows.append({"Pathway": "Electric Engagement", "Detail": fd.FLX1_ITEMS["v2g_max"], "2050": -9.0, "excel_row": 8})
    out = fd.flx1_flexibility(pd.DataFrame(rows), "Holistic Transition", 2050)
    assert out["v2g_max"]["gw"] == 1.5 and out["v2g_max"]["excel_row"] == 7
    with pytest.raises(ValueError):
        fd.flx1_flexibility(pd.DataFrame(rows[:-2]), "Holistic Transition", 2050)


# ─────────────────────────────────── shapes ───────────────────────────────────

def test_heat_pump_profile_hot_water_cop_and_daily_shape():
    cfg = fd.resolve_config({})
    temp = _temperature()
    hp = fd.heat_pump_profiles(temp, cfg)
    assert hp["space_heat_th"].sum() + hp["hot_water_th"].sum() == pytest.approx(1.0)
    assert hp["hot_water_th"].sum() == pytest.approx(0.20)
    july = hp.index.month == 7
    assert hp.loc[july, "electric"].sum() > 0          # hot water keeps summer load
    assert np.corrcoef(temp, hp["cop_space"])[0, 1] > 0.99   # COP rises with temperature
    assert (hp["cop_water"] <= hp["cop_space"] + 1e-12).all()  # hotter sink, lower COP
    jan = hp[hp.index.month == 1]
    by_hour = jan["space_heat_th"].groupby(jan.index.hour).mean()
    assert by_hour.idxmax() in (6, 7, 8)                # morning heating peak
    assert by_hour.loc[3] < by_hour.loc[18]             # overnight set-back


def test_district_heat_daily_shape_and_cop():
    cfg = fd.resolve_config({})
    temp = _temperature()
    dh = fd.district_heat_profile(temp, cfg)
    assert dh["heat_th"].sum() == pytest.approx(1.0)
    day = dh.loc["2050-01-15", "heat_th"]
    assert day.max() == pytest.approx(day.min())        # networks buffer the daily swing
    assert dh["cop"].between(1.5, 4.5).all()


def test_ev_plugin_profiles_calendar_weekends():
    cfg = fd.resolve_config({})
    p = fd.ev_profiles(YEAR, cfg)
    for col in ("residential", "public", "depot", "driving"):
        assert p[col].mean() == pytest.approx(1.0)
    wk = p[p.index.dayofweek < 5]
    assert wk["residential"].groupby(wk.index.hour).mean().idxmax() in (18, 19, 20)
    assert wk["depot"].groupby(wk.index.hour).mean().idxmax() in (0, 1, 2)
    # 1 January 2050 is a Saturday: the weekend shape applies (the old profile used index 0).
    sat = p.loc["2050-01-01", "residential"]
    assert sat.idxmax().hour == 15


def test_map_weather_year_by_month_day_hour():
    src = pd.Series(np.arange(8760, dtype=float), index=pd.date_range("2013-01-01", periods=8760, freq="h"))
    out = fd.map_weather_year(src, YEAR)
    assert (out.to_numpy() == src.to_numpy()).all()
    leap = pd.date_range("2052-02-28", periods=72, freq="h")
    mapped = fd.map_weather_year(src, leap)
    assert (mapped.loc["2052-02-29"].to_numpy() == src.loc["2013-02-28"].to_numpy()).all()


def test_demand_weighted_temperature_uses_demand_cells():
    import xarray as xr
    t = np.zeros((3, 2, 2)) + 273.15
    t[:, 0, 0] += 10.0   # cell (lat 50, lon -3): 10 C
    t[:, 1, 1] += 0.0    # cell (lat 51, lon -2): 0 C
    da = xr.DataArray(t, dims=("time", "y", "x"),
                      coords={"time": pd.date_range("2013-01-01", periods=3, freq="h"), "y": [50.0, 51.0],
                              "x": [-3.0, -2.0]})
    out = fd.demand_weighted_temperature(da, np.array([-3.0, -2.0]), np.array([50.0, 51.0]), np.array([3.0, 1.0]))
    assert out.iloc[0] == pytest.approx(7.5)


def test_solve_flex_share():
    assert fd.solve_flex_share(20.0, 7.6, 9.0, 0.95) == pytest.approx(1 - 1.4 / 20.0)
    assert fd.solve_flex_share(1.0, 20.0, 9.0, 0.95) == 0.95   # capped
    assert fd.solve_flex_share(0.0, 1.0, 9.0, 0.95) == 0.0


# ────────────────────────────── component build ──────────────────────────────

def test_build_components_conserves_energy_and_calibrates(built):
    cfg, energies, peaks, temp, comps = built
    prof, flex, cal = comps["profiles"], comps["flex"], comps["calibration"]
    gwh = lambda col: prof[col].sum() / 1e3
    # Heat pumps: fixed + flexible drain = Dem_BB004 + BB005.
    assert gwh(dc.HEAT_PUMP_LOAD) - gwh("HP_flexibility_drain") == pytest.approx(energies["heat_pumps_gwh"])
    assert flex[dc.HEAT_PUMP_FLEXIBILITY]["electricity_gwh"] == pytest.approx(-gwh("HP_flexibility_drain"))
    # EVs: fixed + (drains / charge efficiency) = Dem_BB006 + BB007.
    eta = cfg["electric_vehicles"]["charge_efficiency"]
    ev_flex = -(gwh("EV_smart_charging_drain") + gwh("V2G_fleet_drain")) / eta
    assert gwh(dc.EV_LOAD) + ev_flex == pytest.approx(energies["evs_gwh"])
    # EV load at the ED1 peak hours is ED1's 1.4 + 7.6 GW.
    assert cal["ev_fixed_at_peak_gw"] == pytest.approx(peaks["ev_at_peak_gw"])
    mask = prof["calibration_hour"].astype(bool)
    assert prof.loc[mask, dc.EV_LOAD].mean() / 1e3 == pytest.approx(9.0 * energies["loss_factor"])
    # V2G: 81.2 GW maximum, half available at the peak hours (FLX1 rows 120, 125).
    v2g = flex[dc.V2G]
    assert v2g["discharge_mw"] == pytest.approx(81210.0)
    at_peak = (prof.loc[mask, "V2G_fleet_p_max_pu"] * v2g["p_nom_mw"]).mean() / 1e3
    assert at_peak == pytest.approx(40.6, rel=1e-6)
    # Heat-pump flexibility at the peak hours is FLX1's.
    hp_flex_at_peak = -prof.loc[mask, "HP_flexibility_drain"].mean() / 1e3
    assert hp_flex_at_peak == pytest.approx(peaks["hp_flex_at_peak_gw"])
    # Data centres flat, DACCS volume, DSR energy neutral.
    assert prof[dc.DATA_CENTRE_LOAD].std() / prof[dc.DATA_CENTRE_LOAD].mean() < 1e-12
    assert -gwh("DACCS_drain") == pytest.approx(17828.38)
    assert flex[dc.IC_DSR]["p_nom_mw"] == pytest.approx(3900.0) and flex[dc.IC_DSR]["inflow"] is None
    assert dc.RESIDENTIAL_DSR not in flex           # off by default
    for spec in flex.values():
        assert spec["energy_mwh"] > 0 and spec["p_nom_mw"] > 0


def test_ev_store_days_caps_the_smart_charging_and_v2g_stores(built):
    cfg, energies, peaks, temp, comps = built
    capped_cfg = fd.resolve_config({"enabled": True, "electric_vehicles": {"store_days": 1.0}})
    capped = fd.build_components(YEAR, temp, energies, peaks, 38.7e6, capped_cfg)
    for carrier in (dc.EV_SMART_CHARGING, dc.V2G):
        drain = -capped["profiles"][f"{carrier}_drain"]
        assert capped["flex"][carrier]["energy_mwh"] == pytest.approx(drain.mean() * 24.0)
        # Only the store shrinks: power, drain and volume are the uncapped build's.
        assert capped["flex"][carrier]["energy_mwh"] < comps["flex"][carrier]["energy_mwh"]
        assert capped["flex"][carrier]["p_nom_mw"] == pytest.approx(comps["flex"][carrier]["p_nom_mw"])
        pd.testing.assert_series_equal(drain, -comps["profiles"][f"{carrier}_drain"])


# ─────────────────────────────── network step ───────────────────────────────

@pytest.fixture(scope="module")
def integrated(built):
    cfg, energies, peaks, temp, comps = built
    n = _network()
    stats = fd.integrate_fes_demand(n, comps, _allocation(), cfg)
    return n, stats, energies, peaks, comps


def test_integrate_conserves_every_line(integrated):
    n, stats, energies, peaks, comps = integrated
    check = fd.demand_check(n, energies, peaks, comps["calibration"])
    energy_rows = check[check["item"].str.endswith("(GWh)")]
    assert (energy_rows["status"] == "PASS").all(), energy_rows
    b = fd.network_demand_breakdown(n)
    assert b.drop("daccs").sum() == pytest.approx(energies["demand_total_gwh"])
    assert stats["clipped_gwh"] == pytest.approx(0.0)
    # The H2 load is untouched and not demand.
    assert "H2 offtake" not in n.loads_t.p_set.columns and n.loads.at["H2 offtake", "p_set"] == 100.0


def test_integrate_per_bus_keeps_base_bus_totals(integrated):
    n, stats, energies, *_ = integrated
    dt = n.snapshot_weightings.generators
    ac = n.loads[n.loads.bus.isin(["A", "B", "C"])]
    fixed = n.loads_t.p_set[ac.index].mul(dt, axis=0).sum().groupby(ac.bus).sum() / 1e3
    su = n.storage_units
    drains = su[su.carrier.isin(dc.CONSUMER_FLEX_CARRIERS) & su.index.isin(n.storage_units_t.inflow.columns)]
    drain = (-n.storage_units_t.inflow[drains.index]).mul(dt, axis=0).sum() / drains.efficiency_store / 1e3
    per_bus = fixed.add(drain.groupby(drains.bus).sum(), fill_value=0.0)
    expected = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"]) * energies["demand_total_gwh"]
    assert per_bus.to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-9)


def test_integrate_adds_tagged_storage_units_that_aggregate_exactly(integrated):
    n, *_ = integrated
    su = n.storage_units
    assert set(su.carrier) == {dc.HEAT_PUMP_FLEXIBILITY, dc.DISTRICT_HEAT_FLEXIBILITY, dc.EV_SMART_CHARGING,
                               dc.V2G, dc.IC_DSR, dc.DACCS}
    assert set(su.carrier) <= set(dc.CONSUMER_OWNED_CARRIERS) | set(dc.NON_ELECTRICITY_ASSET_CARRIERS)
    assert not su.cyclic_state_of_charge.any() and (su.state_of_charge_initial == 0).all()
    for carrier, group in su.groupby("carrier"):
        assert group.max_hours.nunique() == 1           # aggregateoneport keeps energy exact
    assert su.loc[su.carrier == dc.DACCS, "bus"].tolist() == ["C"]
    assert {dc.HEAT_PUMP_LOAD, dc.EV_LOAD, dc.DATA_CENTRE_LOAD} <= set(n.loads.carrier)
    assert {"electricity", "H2_demand"} <= set(n.loads.carrier)


def test_flat_residual_mode_and_overrun_guard(built):
    cfg, energies, peaks, temp, comps = built
    n = _network()
    stats = fd.integrate_fes_demand(n, comps, _allocation(), {**cfg, "residual_allocation": "flat"})
    base = n.loads_t.p_set[["base A", "base B", "base C"]].sum()
    assert (base / base.sum()).to_numpy() == pytest.approx([0.5, 0.3, 0.2])
    small = {**comps, "energies": {**comps["energies"], "demand_total_gwh": 1000.0}}
    with pytest.raises(ValueError):
        fd.integrate_fes_demand(_network(), small, _allocation(), cfg)


def test_components_round_trip(tmp_path, built):
    cfg, energies, peaks, temp, comps = built
    comps = {**comps, "peaks": peaks, "config": cfg}
    paths = [str(tmp_path / f) for f in ("p.csv", "a.csv", "s.json")]
    fd.save_components(comps, _allocation(), *paths)
    back, alloc = fd.load_components(*paths)
    assert back["flex"][dc.V2G]["p_nom_mw"] == pytest.approx(comps["flex"][dc.V2G]["p_nom_mw"])
    n1, n2 = _network(), _network()
    fd.integrate_fes_demand(n1, comps, _allocation(), cfg)
    fd.integrate_fes_demand(n2, back, alloc, cfg)
    assert np.allclose(n1.loads_t.p_set.sum().sort_index(), n2.loads_t.p_set.sum().sort_index())


def test_nearest_bus_shares():
    buses = pd.DataFrame({"x": [-0.2, -1.2, -1.8, -2.9, 0.0], "y": [53.6, 54.6, 57.5, 53.3, 51.5],
                          "carrier": ["AC"] * 5}, index=["HUM", "TEE", "SFE", "LIV", "LON"])
    s = fd.nearest_bus_shares(buses, fd.DEFAULT_CONFIG["daccs"]["sites"])
    assert s.sum() == pytest.approx(1.0)
    assert s.to_dict() == pytest.approx({"HUM": 0.35, "TEE": 0.30, "SFE": 0.20, "LIV": 0.15})


def test_build_allocation_columns_sum_to_one(built):
    cfg, energies, peaks, temp, comps = built
    blocks = sorted({b for v in fd.ALLOCATION_BLOCKS.values() for b in v})
    rows = [{"Building Block ID Number": b, "FES Pathway": "Holistic Transition", "GSP": g, "2050": v}
            for b in blocks for g, v in (("A", 3.0), ("B", 1.0))]
    fes = pd.DataFrame(rows)
    buses = pd.DataFrame({"x": [-0.2, -1.2], "y": [53.6, 54.6], "carrier": ["AC", "AC"]}, index=["A", "B"])
    table = fd.build_allocation(fes, "Holistic Transition", 2050, lambda gsp, shape: gsp.astype(float),
                                comps["profiles"], buses, cfg)
    assert np.allclose(table.sum(), 1.0)
    assert table.loc["A", "heat_pumps"] == pytest.approx(0.75)


# ────────────────────────── the optimiser sees them ──────────────────────────

def test_storage_units_meet_drain_and_daccs_volume_in_a_window(built):
    """A 48 h LP: EV, heat-pump and DACCS units meet their drains, charging in cheap hours."""
    cfg, energies, peaks, temp, comps = built
    sns = YEAR[:48]
    n = _network()
    fd.integrate_fes_demand(n, comps, _allocation(), cfg)
    n.set_snapshots(sns)
    price = np.where(sns.hour.isin(range(0, 6)), 5.0, 80.0)
    for b in ("A", "B", "C"):
        n.add("Generator", f"cheap {b}", bus=b, p_nom=4e5, marginal_cost=5.0,
              p_max_pu=pd.Series(np.where(price == 5.0, 1.0, 0.0), index=sns))
        n.add("Generator", f"dear {b}", bus=b, p_nom=4e5, marginal_cost=80.0)
    n.add("Generator", "h2 supply", bus="H2", p_nom=1e3)
    n.add("Line", "AB", bus0="A", bus1="B", x=0.1, s_nom=1e6)
    n.add("Line", "BC", bus0="B", bus1="C", x=0.1, s_nom=1e6)
    status, condition = n.optimize(solver_name="highs")
    assert status == "ok"
    su = n.storage_units
    dac = su.index[su.carrier == dc.DACCS]
    charged = n.storage_units_t.p_store[dac].sum().sum()
    drained = -n.storage_units_t.inflow[dac].sum().sum()
    assert charged == pytest.approx(drained, rel=1e-6)      # per-window volume met exactly
    cheap = sns.hour.isin(range(0, 6))
    # DACCS runs at its capacity (twice the mean drain) in every cheap hour: half the volume.
    assert n.storage_units_t.p_store[dac].loc[cheap].sum().sum() >= 0.5 * charged * (1 - 1e-6)
    evu = su.index[su.carrier == dc.EV_SMART_CHARGING]
    ev_in = n.storage_units_t.p_store[evu].sum().sum() * su.loc[evu, "efficiency_store"].iloc[0]
    ev_soc_end = n.storage_units_t.state_of_charge[evu].iloc[-1].sum()
    assert ev_in == pytest.approx(-n.storage_units_t.inflow[evu].sum().sum() + ev_soc_end, rel=1e-6)
    assert (n.storage_units_t.state_of_charge.min() >= -1e-6).all()


# ──────────────────────────── energy validation ────────────────────────────

def test_energy_validation_counts_components_but_not_daccs(integrated):
    n, stats, energies, *_ = integrated
    tables = ev.network_energy_tables(n)
    b = fd.network_demand_breakdown(n)
    assert tables["demand_mwh"] / 1e3 == pytest.approx(b.drop("daccs").sum())
    lines = tables["demand_components_mwh"] / 1e3
    assert lines["heat_pumps"] == pytest.approx(energies["heat_pumps_gwh"])
    assert lines["evs"] == pytest.approx(energies["evs_gwh"])
    assert lines["daccs"] == pytest.approx(energies["daccs_gwh"])


def test_energy_validation_base_only_network_unchanged():
    n = _network(total_gwh=1000.0)
    tables = ev.network_energy_tables(n)
    weights = n.snapshot_weightings.generators
    names = ["base A", "base B", "base C"]
    old = float(n.loads_t.p_set[names].mul(weights, axis=0).to_numpy().sum())
    assert tables["demand_mwh"] == old                   # exactly the pre-Y6a formula
    assert list(tables["demand_components_mwh"].index) == ["residual"]


def test_energy_validation_dispatch_counts_flex_charging(integrated):
    n, *_ = integrated
    m = n.copy()
    m.storage_units_t.p_store = pd.DataFrame(0.0, index=m.snapshots, columns=m.storage_units.index)
    m.storage_units_t.p_dispatch = pd.DataFrame(0.0, index=m.snapshots, columns=m.storage_units.index)
    ev_units = m.storage_units.index[m.storage_units.carrier == dc.EV_SMART_CHARGING]
    dr_units = m.storage_units.index[m.storage_units.carrier == dc.IC_DSR]
    m.storage_units_t.p_store[ev_units] = 1.0
    m.storage_units_t.p_store[dr_units] = 2.0
    m.storage_units_t.p_dispatch[dr_units] = 2.0
    m.generators_t.p = pd.DataFrame(index=m.snapshots)
    tables = ev.network_energy_tables(m, include_dispatch=True)
    lines = tables["demand_components_mwh"]
    assert lines["demand_response"] == pytest.approx(0.0)         # shifted, not added
    fixed_ev = m.loads_t.p_set[[c for c in m.loads.index if m.loads.at[c, "carrier"] == dc.EV_LOAD]].sum().sum()
    assert lines["evs"] == pytest.approx(fixed_ev + len(ev_units) * 8760)   # charged electricity
    assert tables["demand_mwh"] == pytest.approx(lines.drop(labels=["daccs"], errors="ignore").sum())


def test_finalize_guard_against_double_counting(built):
    from scripts.demand.finalize_demand_integration import finalize_demand_integration
    import logging
    cfg, energies, peaks, temp, comps = built
    with pytest.raises(ValueError):
        finalize_demand_integration(_network(), pd.DataFrame(), {"enabled": True}, {"enabled": False}, {}, None,
                                    None, None, logging.getLogger("t"),
                                    fes_components={"config": {"enabled": True}, "components": comps,
                                             "allocation": _allocation()})


# ──────────────── storage heaters and residential DSR (27 Oct 2026, F5 with 2050) ────────────────

def test_flx1_storage_heaters_are_separate_rows_so_nothing_is_counted_twice():
    """FLX1 row 185 is the plain sum of the peak-reduction rows: storage heaters sit beside thermal storage."""
    flex = _flex()
    parts = sum(flex[k]["gw"] for k in fd.FLX1_ITEMS
                if k not in ("v2g_max", "unconstrained_heat_peak", "total_consumer_flex"))
    assert parts == pytest.approx(flex["total_consumer_flex"]["gw"], abs=0.01)
    p = fd.peak_targets(_targets(), flex)
    assert p["hp_flex_at_peak_gw"] == pytest.approx(0.192 + 6.578 + 0.043 + 0.621)      # no storage heaters in it
    assert p["storage_heater_flex_at_peak_gw"] == pytest.approx(4.62 + 2.377)


@pytest.fixture(scope="module")
def with_storage_heaters(built):
    cfg, energies, peaks, temp, comps = built
    on = fd.resolve_config({"enabled": True, "storage_heater_flexibility": {"enabled": True}})
    c2 = fd.build_components(YEAR, temp, energies, peaks, 38.7e6, on)
    n = _network()
    fd.integrate_fes_demand(n, c2, _allocation(), on)
    return n, energies, peaks, c2


def test_storage_heaters_off_by_default(built):
    assert dc.STORAGE_HEATER_FLEXIBILITY not in built[4]["flex"]


def test_storage_heaters_built_on_the_2030_method(with_storage_heaters):
    n, energies, peaks, comps = with_storage_heaters
    spec = comps["flex"][dc.STORAGE_HEATER_FLEXIBILITY]
    drain = -comps["profiles"][spec["inflow"]]
    mask = comps["profiles"]["calibration_hour"].astype(bool).to_numpy()
    assert drain.to_numpy()[mask].mean() == pytest.approx(peaks["storage_heater_flex_at_peak_gw"] * 1e3)
    coldest = drain.groupby(drain.index.normalize()).sum().max()
    assert spec["p_nom_mw"] == pytest.approx(coldest / 7.0) and spec["energy_mwh"] == pytest.approx(coldest)
    assert spec["electricity_gwh"] == pytest.approx(drain.sum() / 1e3)
    assert spec["standing_loss"] == 0.0 and spec["allocation"] == "storage_heaters"


def test_storage_heater_energy_comes_out_of_the_residual_and_totals_close(with_storage_heaters):
    n, energies, peaks, comps = with_storage_heaters
    b = fd.network_demand_breakdown(n)
    assert b["residual"] == pytest.approx(energies["residual_gwh"], rel=1e-6)   # base loads + storage heaters
    total = float(b.drop(labels=["daccs"], errors="ignore").sum())
    assert total == pytest.approx(energies["demand_total_gwh"], rel=1e-6)
    sh = n.storage_units[n.storage_units.carrier.eq(dc.STORAGE_HEATER_FLEXIBILITY)]
    sh_gwh = float(-n.storage_units_t.inflow[sh.index].sum().sum()) / 1e3
    assert sh_gwh == pytest.approx(comps["flex"][dc.STORAGE_HEATER_FLEXIBILITY]["electricity_gwh"], rel=1e-9)
    base = n.loads_t.p_set[["base A", "base B", "base C"]].sum().sum() / 1e3
    assert base + sh_gwh == pytest.approx(energies["residual_gwh"], rel=1e-6)
    # the other heat units are unchanged by it: thermal storage is not counted twice
    assert comps["flex"][dc.HEAT_PUMP_FLEXIBILITY]["p_nom_mw"] == pytest.approx(
        fd.build_components(YEAR, _temperature(), energies, peaks, 38.7e6,
                            fd.resolve_config({"enabled": True}))["flex"][dc.HEAT_PUMP_FLEXIBILITY]["p_nom_mw"])


def test_residential_dsr_switched_on_builds_energy_neutral_units(built):
    cfg, energies, peaks, temp, comps = built
    on = fd.resolve_config({"enabled": True, "residential_dsr": {"enabled": True}})
    c2 = fd.build_components(YEAR, temp, energies, peaks, 38.7e6, on)
    spec = c2["flex"][dc.RESIDENTIAL_DSR]
    assert spec["p_nom_mw"] == pytest.approx(6300.0) and spec["energy_mwh"] == pytest.approx(12600.0)
    assert spec["inflow"] is None and spec["electricity_gwh"] == 0.0 and spec["allocation"] == "residential_dsr"
    n = _network()
    fd.integrate_fes_demand(n, c2, _allocation(), on)
    units = n.storage_units[n.storage_units.carrier.eq(dc.RESIDENTIAL_DSR)]
    assert units["p_nom"].sum() == pytest.approx(6300.0) and set(units.bus) == {"A", "B", "C"}
    assert fd.network_demand_breakdown(n).get("demand_response", 0.0) == 0.0
    assert dc.RESIDENTIAL_DSR not in comps["flex"]                                    # off by default
