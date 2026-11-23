"""Pro rata zonal turn-down re-pricing (scripts/market/turn_down_allocation.py, 20 Oct 2026)."""
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.market import neso_zones
from scripts.market import turn_down_allocation as tda
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "resources" / "market"


# ── Allocation ──────────────────────────────────────────────────────────────
def _case():
    rng = np.random.default_rng(1)
    s1 = rng.uniform(0, 100, size=(48, 6))
    bm = s1 * rng.uniform(0.3, 1.0, size=s1.shape)
    bm[:, 5] = s1[:, 5] + 5.0                       # a unit turned up, never down
    dec = np.clip(s1 - bm, 0, None)
    groups = np.array(["Z1", "Z1", "Z1", "Z2", "Z2", None], dtype=object)
    return s1, bm, dec, groups


def test_volume_conserved_per_zone_hour():
    s1, _, dec, groups = _case()
    pr = tda.pro_rata_turn_down(s1, dec, groups)
    for z in ("Z1", "Z2"):
        k = groups == z
        np.testing.assert_allclose(pr[:, k].sum(axis=1), dec[:, k].sum(axis=1), rtol=0, atol=1e-9)
    assert (pr <= s1 + 1e-9).all() and (pr >= -1e-12).all()


def test_shares_follow_stage1_output():
    s1 = np.array([[30.0, 10.0, 60.0]])
    dec = np.array([[20.0, 0.0, 0.0]])
    pr = tda.pro_rata_turn_down(s1, dec, ["Z1", "Z1", "Z1"])
    np.testing.assert_allclose(pr, [[6.0, 2.0, 12.0]])


def test_unzoned_unit_keeps_its_own_volume():
    s1, _, dec, groups = _case()
    pr = tda.pro_rata_turn_down(s1, dec, groups)
    np.testing.assert_allclose(pr[:, 5], dec[:, 5])


def test_reallocated_dispatch_keeps_turn_up_and_totals():
    s1, bm, _, groups = _case()
    new = tda.reallocated_dispatch(s1, bm, groups)
    np.testing.assert_allclose(new[:, 5], bm[:, 5])
    for z in ("Z1", "Z2"):
        k = groups == z
        np.testing.assert_allclose(new[:, k].sum(axis=1), bm[:, k].sum(axis=1), atol=1e-9)


# ── Bids ────────────────────────────────────────────────────────────────────
def _audit(rows):
    cols = ["generator", "carrier", "support_type", "cfd_round", "pricing", "strike_price_gbp_per_mwh",
            "reference_price_basis", "mean_reference_price_gbp_per_mwh", "mean_bid_gbp_per_mwh", "state_share",
            "bid_basis", "contract_value_floor", "negative_price_rule", "tie_break_step_gbp_per_mwh"]
    return pd.DataFrame(rows, columns=cols)


def test_rebuilt_bids_match_the_bm_blend_private_and_state():
    """P2/P3: a state share bids (1 - s) x contract value + s x public price + its tie step."""
    from scripts.market.cfd_bm_bids import foregone_rate

    idx = pd.date_range("2030-01-01", periods=24, freq="h")
    ref = pd.Series(np.linspace(-20, 120, 24), index=idx)
    strike, s, public, step = 80.0, 0.6, 0.0, 0.3
    cv = foregone_rate(strike, ref, "contract_value", "none")
    lp_bid = (1 - s) * cv + s * public + step * s          # what build_contract_value_bid_frames gives the LP
    audit = _audit([
        ["A", "wind_offshore", "CfD", "AR4", "cfd_contract_value", strike, "hourly_wholesale", ref.mean(),
         lp_bid.mean(), s, "contract_value", "none", "none", step * s],
        ["B", "wind_onshore", "CfD", "AR4", "cfd_contract_value", strike, "hourly_wholesale", ref.mean(),
         cv.mean(), 0.0, "contract_value", "none", "none", 0.0],
        ["C", "wind_onshore", "merchant", "", "merchant_short_run_cost", np.nan, "not_applicable", ref.mean(),
         0.0, 0.0, "", "", "none", 0.0],
        ["D", "wind_onshore", "roc", "", "roc_state_at_cost", np.nan, "not_applicable", ref.mean(),
         0.0, 1.0, "", "", "none", 0.0],
    ])
    units = pd.DataFrame({"support_type": ["CfD", "CfD", "merchant", "ROC", "ROC"],
                          "ro_banding": [np.nan, np.nan, np.nan, 0.9, 0.9], "cfd_round": ["AR4", "AR4", "", "", ""]},
                         index=["A", "B", "C", "D", "E"])
    bids, label = tda.rebuild_bids(units, audit, ref, roc_rate=78.04)
    np.testing.assert_allclose(bids["A"], lp_bid, atol=1e-9)
    np.testing.assert_allclose(bids["B"], cv, atol=1e-9)
    assert (bids["C"] == 0).all() and (bids["D"] == 0).all()        # state-owned RO at cost
    np.testing.assert_allclose(bids["E"], 0.9 * 78.04)               # private RO: banding x buyout
    assert label.to_dict() == {"A": "cfd_contract_value", "B": "cfd_contract_value", "C": "merchant_short_run_cost",
                               "D": "roc_state_at_cost", "E": "roc_banding_x_buyout"}


def test_negative_price_rule_zeroes_the_contract_value():
    idx = pd.date_range("2030-01-01", periods=4, freq="h")
    ref = pd.Series([50.0, -5.0, 10.0, -1.0], index=idx)
    audit = _audit([["A", "wind_onshore", "CfD", "AR4", "cfd_contract_value", 60.0, "hourly_wholesale", ref.mean(),
                     np.mean([10.0, 0.0, 50.0, 0.0]), 0.0, "contract_value", "zero",
                     "ar4_zero_payment_when_price_le_0", 0.0]])
    units = pd.DataFrame({"support_type": ["CfD"], "ro_banding": [np.nan], "cfd_round": ["AR4"]}, index=["A"])
    bids, _ = tda.rebuild_bids(units, audit, ref, None)
    np.testing.assert_allclose(bids["A"], [10.0, 0.0, 50.0, 0.0])


@pytest.mark.parametrize("state_share", [0.0, 0.3])
def test_label_priced_units_rebuild_the_solve_bid_hour_by_hour(tmp_path, state_share):
    """2040/2050 (13 Nov 2026): a unit bidding from the support label table gets the solve's own hourly bid.

    Each unit blends CfD cohorts at different strikes (the contract value moves with the hour and the
    negative-price rule zeroes it), merchant and static parts; under P2 the state takes a share of the
    post-2025 cohorts. The rebuild must equal ``support_bids.build_label_bid_frames`` in every hour.
    """
    import pypsa

    from scripts.market import support_bids as sb
    from scripts.market import support_labels as sl

    def lab(component, carrier, weight, support, year, strike=np.nan, live=1.0, rule="ar4_plus"):
        return {"component": component, "component_type": "Generator", "carrier": carrier,
                "cohort_id": f"{component}::b{year}", "stock_flow_technology": "", "build_year": year,
                "cohort_capacity_mw": weight, "weight": weight, "support_type": support, "cfd_round": "",
                "cfd_contract_id": None, "live_share": live, "strike_gbp_per_mwh_2024": strike, "strike_key": "",
                "negative_price_rule": rule if support == "cfd" else "", "reference_price":
                "intermittent_hourly" if support == "cfd" else "", "dpa_reference_srmc_gbp_per_mwh": np.nan,
                "dpa_margin_gbp_per_mwh": np.nan, "support_basis": ""}

    labels = pd.DataFrame([
        lab("off", "wind_offshore", 0.2, "cfd", 2018, 140.0, rule="none"),     # pre-2025 CfD, no negative rule
        lab("off", "wind_offshore", 0.5, "cfd", 2035, 70.0),
        lab("off", "wind_offshore", 0.3, "merchant", 2036),
        lab("on", "wind_onshore", 0.4, "cfd", 2030, 55.0, live=0.5),
        lab("on", "wind_onshore", 0.6, "cfd", 2040, 90.0),
    ], columns=list(sl.LABEL_COLUMNS))
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2040-03-01", periods=6, freq="h"))
    n.add("Bus", "bus", v_nom=400, carrier="AC")
    n.add("Load", "load", bus="bus", p_set=1000.0)
    for name, carrier in (("off", "wind_offshore"), ("on", "wind_onshore")):
        n.add("Generator", name, bus="bus", carrier=carrier, p_nom=1000.0, marginal_cost=0.0)
    ref = pd.Series([150.0, 80.0, 60.0, 30.0, -5.0, 0.0], index=n.snapshots)
    cvb = {"enabled": True, "bid_basis": "contract_value", "negative_price_rule": "gb_contract_terms",
           "merchant": {"bid_price": 0.0, "carriers": ["wind_offshore", "wind_onshore", "solar_pv"]},
           "support_labels": {"file": "unused.csv"}}
    if state_share:
        table = tmp_path / "shares.csv"
        pd.DataFrame({"case_id": "public_ownership@deployment_share_50", "generator": ["off", "on"],
                      "capacity_mw": 1000.0, "state_capacity_mw": 1000.0 * state_share, "state_share": state_share,
                      "state_commissioning_year": 2040.0}).to_csv(table, index=False)
        cvb["public_ownership_bids"] = {"enabled": True, "case_id": "public_ownership@deployment_share_50",
                                        "shares_file": str(table), "public_bid_price": 0.0}
    scenario = {"support": {"bm_bids": {"enabled": True, "mode": "fixed", "exclude_carriers": ["nuclear"],
                                        "contract_value_bids": cvb}}}
    static = pd.Series(25.0, index=n.generators.index)
    frames = sb.build_label_bid_frames(n, scenario, {}, static, static, ref, logging.getLogger("t"), labels=labels)
    units = pd.DataFrame({"support_type": ["mixed", "mixed"], "ro_banding": [np.nan, np.nan], "cfd_round": ["", ""]},
                         index=["off", "on"])
    audit = frames.audit.copy()
    assert set(audit["pricing"]) == {tda.LABEL_PRICING}
    bids, label = tda.rebuild_bids(units, audit, ref, None, labels=labels)
    np.testing.assert_allclose(bids[["off", "on"]].to_numpy(), frames.bid_tv[["off", "on"]].to_numpy(), atol=1e-9)
    assert label.eq(tda.LABEL_PRICING).all()
    # The flat mean the re-pricing used before 13 Nov is not the hourly bid
    assert not np.allclose(bids["on"].to_numpy(), audit.set_index("generator").at["on", "mean_bid_gbp_per_mwh"])
    with pytest.raises(ValueError, match="label table"):
        tda.rebuild_bids(units, audit, ref, None)


def test_unpriced_unit_takes_its_implied_bid():
    idx = pd.date_range("2030-01-01", periods=3, freq="h")
    units = pd.DataFrame({"support_type": ["other"], "ro_banding": [np.nan], "cfd_round": [""]}, index=["X"])
    bids, label = tda.rebuild_bids(units, _audit([]), pd.Series(0.0, index=idx), None, pd.Series({"X": 12.5}))
    assert label["X"] == "implied_from_solve" and (bids["X"] == 12.5).all()


# ── Config and basis selection ──────────────────────────────────────────────
def test_config_defaults_and_refusals():
    cfg = tda.allocation_config(None)
    assert cfg["mode"] == "closest_to_neso" and cfg["technologies"] == ["wind_onshore", "wind_offshore"]
    assert cfg["zone_basis"] == "neso_cp08"
    with pytest.raises(ValueError):
        tda.allocation_config({"mode": "lowest"})
    with pytest.raises(ValueError):
        tda.allocation_config({"bogus": 1})


def test_defaults_yaml_block_matches_code():
    import yaml

    d = load_defaults()
    block = ((d.get("market") or {}).get("balancing") or {}).get("turn_down_allocation")
    assert block is not None, "market.balancing.turn_down_allocation must be in config/defaults/base/market/balancing.yaml"
    assert tda.allocation_config(block) == tda.allocation_config(None)


def test_select_basis():
    costs = {"cheapest_first": 6677e6, "pro_rata_zonal": 7650e6}
    assert tda.select_basis("closest_to_neso", costs, 7790e6)[0] == "pro_rata_zonal"
    assert tda.select_basis("closest_to_neso", {"cheapest_first": 3460e6, "pro_rata_zonal": 4250e6}, 3570e6)[0] \
        == "cheapest_first"
    assert tda.select_basis("closest_to_neso", {"cheapest_first": 90.0, "pro_rata_zonal": 110.0}, 100.0)[0] \
        == "cheapest_first"                                     # a tie stays as solved
    assert tda.select_basis("pro_rata_zonal", costs, None) == ("pro_rata_zonal", "configured")
    with pytest.raises(ValueError):
        tda.select_basis("closest_to_neso", costs, None)


def test_network_benchmarks_are_neso_figures():
    assert tda.network_benchmark(None)["value_GBP"] == pytest.approx(3570e6)
    assert tda.network_benchmark("no_acceleration")["value_GBP"] == pytest.approx(7790e6)
    assert tda.network_benchmark("no_upgrades")["value_GBP"] == pytest.approx(12700e6)
    assert tda.network_benchmark("seven_late")["value_GBP"] == pytest.approx(10240e6)


def test_family_summary_reports_both_bases_and_names_the_headline():
    months = [{"scenario": "m1", "turn_down_MWh": 10.0, "wind_bid_cost_as_solved_GBP": 100.0,
               "wind_bid_cost_as_solved_rebuilt_GBP": 100.0, "wind_bid_cost_pro_rata_GBP": 180.0,
               "pro_rata_uplift_GBP": 80.0, "check": {"passed": True}, "by_zone": {"Z1": {"our_MWh": 10.0}}}]
    annual = {"aggregate": {"neso_comparable_redispatch_cost_GBP": 1000.0}}
    out = tda.summarise_family("S", months, annual, {"mode": "closest_to_neso"},
                               {"network": "x", "value_GBP": 1100.0})
    assert out["neso_comparable_redispatch_cost_by_basis_GBP"] == {"cheapest_first": 1000.0, "pro_rata_zonal": 1080.0}
    assert out["headline_basis"] == "pro_rata_zonal" and out["headline_neso_comparable_redispatch_cost_GBP"] == 1080.0
    out = tda.summarise_family("S", months, annual, {"mode": "cheapest_first"}, None)
    assert out["headline_basis"] == "cheapest_first" and out["headline_wind_bid_cost_GBP"] == 100.0


# ── The helper the bill reads through ───────────────────────────────────────
def _month_files(tmp_path):
    name = "CP30_FFR_2030_ETYS_market_Jan2030_t"
    rs = pd.DataFrame({"component": ["A", "B", "G", "S"], "type": ["generator"] * 3 + ["storage_unit"],
                       "carrier": ["wind_onshore", "wind_onshore", "CCGT", "battery"],
                       "increase_MWh": [0.0, 0.0, 5.0, 1.0], "decrease_MWh": [10.0, 0.0, 0.0, 0.0],
                       "offer_cost": [0.0, 0.0, 400.0, 10.0], "bid_cost": [100.0, 0.0, 0.0, 0.0],
                       "net_cost": [100.0, 0.0, 400.0, 10.0]})
    rs_path = tmp_path / f"{name}_redispatch_summary.csv"
    rs.to_csv(rs_path, index=False)
    pd.DataFrame({"unit": ["A", "B"], "zone": ["Z1", "Z1"], "our_MWh": [10.0, 0.0], "pro_rata_MWh": [4.0, 6.0],
                  "our_cost_GBP": [100.0, 0.0], "pro_rata_cost_GBP": [40.0, 300.0]}).to_csv(
        tmp_path / f"{name}_pro_rata_turn_down.csv", index=False)
    (tmp_path / f"{name}_pro_rata_turn_down.json").write_text(json.dumps({"pro_rata_uplift_GBP": 240.0}))
    return rs_path


def test_redispatch_in_force_moves_only_wind_and_by_the_uplift(tmp_path):
    path = _month_files(tmp_path)
    solved = tda.read_redispatch_in_force(path, "cheapest_first")
    pro = tda.read_redispatch_in_force(path, "pro_rata_zonal")
    assert pro["bid_cost"].sum() - solved["bid_cost"].sum() == pytest.approx(240.0)
    assert pro["net_cost"].sum() - solved["net_cost"].sum() == pytest.approx(240.0)
    assert pro["decrease_MWh"].sum() == pytest.approx(solved["decrease_MWh"].sum())
    assert pro.set_index("component").loc["B", "bid_cost"] == pytest.approx(300.0)
    pd.testing.assert_frame_equal(pro[pro.component.isin(["G", "S"])], solved[solved.component.isin(["G", "S"])])
    indexed = tda.read_redispatch_in_force(path, "pro_rata_zonal", index_col=0)
    assert indexed.loc["A", "decrease_MWh"] == pytest.approx(4.0)
    assert tda.bid_cost_uplift_in_force(path, "pro_rata_zonal") == pytest.approx(240.0)
    assert tda.bid_cost_uplift_in_force(path, "cheapest_first") == 0.0


def test_month_basis_file_decides_and_an_uncovered_month_is_as_solved(tmp_path):
    path = _month_files(tmp_path)
    pd.testing.assert_frame_equal(tda.read_redispatch_in_force(path), pd.read_csv(path))   # no basis file
    tda.month_basis_path(path).write_text(json.dumps(tda.month_basis_record(
        "m", {"study": "S", "headline_basis": "pro_rata_zonal", "mode": "closest_to_neso"})))
    assert tda.month_basis(path) == "pro_rata_zonal"
    assert tda.read_redispatch_in_force(path)["bid_cost"].sum() == pytest.approx(340.0)
    assert tda.basis_record([path]) == {"pro_rata_zonal": [path.name]}
    other = tmp_path / "CP30_FFR_2030_ETYS_market_Feb2030_t_redispatch_summary.csv"
    pd.read_csv(path).to_csv(other, index=False)
    pd.testing.assert_frame_equal(tda.read_redispatch_in_force(other), pd.read_csv(other))
    with pytest.raises(FileNotFoundError):                         # pro rata named, month never re-priced
        tda.read_redispatch_in_force(other, "pro_rata_zonal")
    wholesale = tmp_path / "CP30_FFR_2030_ETYS_market_Jan2030_s1_wholesale.nc"
    frame = pd.DataFrame({"A": [1.0]})
    assert tda.balancing_dispatch_in_force(frame, wholesale, "pro_rata_zonal") is frame


def test_network_basis_study_follows_the_network_group():
    m = "CP30_FFR_2030_ETYS_market_Jan2030_"
    cases = {
        "CP30_2030_cp30_network": {"annual_aggregate_study": "CP30_2030_cp30_network", "source_scenarios": [m + "cp30_network"]},
        "CP30_2030_cp30_network_cp32": {"annual_aggregate_study": "CP30_2030_cp30_network", "source_scenarios": [m + "cp30_network"]},
        "CP30_2030_cp30_network_public_bids_p2": {"annual_aggregate_study": "CP30_2030_cp30_network_p2_bids",
                                     "source_scenarios": [m + "cp30_network_p2_bids"]},
        "CP30_2030_no_acceleration": {"annual_aggregate_study": "CP30_2030_no_acceleration",
                                      "register_account_group": "cp30_2030_no_acceleration",
                                      "source_scenarios": [m + "no_acceleration"]},
        "CP30_2030_no_acceleration_public_bids_p3": {
            "annual_aggregate_study": "CP30_2030_no_acceleration_p3_bids",
            "register_account_group": "cp30_2030_no_acceleration",
            "source_scenarios": [m + "no_acceleration_p3_bids"]},
    }
    assert tda.network_basis_study(m + "cp30_network", cases) == "CP30_2030_cp30_network"
    assert tda.network_basis_study(m + "cp30_network_p2_bids", cases) == "CP30_2030_cp30_network"
    assert tda.network_basis_study(m + "no_acceleration_p3_bids", cases) == "CP30_2030_no_acceleration"
    assert tda.network_basis_study(m + "unread", cases) is None


# ── Zones ───────────────────────────────────────────────────────────────────
def test_zone_code_is_shared_with_the_balancing_check():
    from scripts.CP30 import balancing_validation as bv

    assert bv.bus_zone_map is neso_zones.bus_zone_map and bv.allowed_zones is neso_zones.allowed_zones
    assert bv.ZONES == neso_zones.ZONES and len(neso_zones.ZONES) == 17


def test_allowed_zones_follow_the_boundaries():
    assert neso_zones.allowed_zones({"EC5": 1}) == ["Z12"]
    assert neso_zones.allowed_zones({"B6": 1, "EC5": -1, "LE1": -1, "SW1": 1, "B13": 1}) == ["Z4"]


def test_polygon_zones_on_land():
    polys = neso_zones.load_zone_polygons(ROOT / neso_zones.ZONES_FILE)
    buses = pd.DataFrame({"lon": [-4.25, -0.12], "lat": [55.86, 51.50], "country": ["GB", "GB"]},
                         index=["glasgow", "london"])
    z = neso_zones.polygon_zones(buses, polys)["neso_zone"]
    assert z["glasgow"] in neso_zones.SCOTLAND and z["london"] not in neso_zones.SCOTLAND


# ── The solved central month (skipped where the outputs are absent) ──────────
@pytest.mark.parametrize("tag", ["no_acceleration", "no_acceleration_p3_bids"])
def test_solved_month_reproduces_the_solve_wind_bid_cost(tag):
    name = f"CP30_FFR_2030_Jan_{tag}"
    paths = tda.month_paths(name, "CP30_FFR_2030_Jan_stage1", str(ROOT / "resources"))
    if not all(Path(p).exists() for p in paths.values()):
        pytest.skip(f"{name} not solved here")
    bz = neso_zones.network_bus_zones(paths["network"], ROOT / neso_zones.ZONES_FILE)
    table, totals = tda.reprice_month(name, paths["network"], paths["stage1_network"], paths["stage1_price"],
                                      paths["contract_value_bids"], paths["redispatch"], bz["neso_zone"],
                                      tda.DEFAULT_CONFIG, 78.04)
    assert totals["check"]["passed"]
    assert abs(totals["check"]["relative_to_gross"]) < 1e-3
    assert totals["check"]["max_zone_volume_gap_MWh"] < 1e-3
    assert table["pro_rata_MWh"].sum() == pytest.approx(table["our_MWh"].sum())


@pytest.mark.parametrize("name,stage1,build", [
    ("FES25_HT_2040_Mar_p2_bids_50pct", "FES25_HT_2040_Mar_market", "FES25_HT_2040"),
    ("FES25_HT_2050_Aug_market", "FES25_HT_2050_Aug_market", "FES25_HT_2050")])
def test_solved_long_range_month_reproduces_the_solve_wind_bid_cost(name, stage1, build):
    """13 Nov 2026: label-priced months (hourly contract-value bids since 11 Nov) pass the as-solved check."""
    paths = tda.month_paths(name, stage1, str(ROOT / "resources"))
    paths["support_labels"] = str(MARKET / f"{build}_support_labels.csv")
    if not all(Path(p).exists() for p in paths.values()):
        pytest.skip(f"{name} not solved here")
    bz = neso_zones.network_bus_zones(paths["network"], ROOT / neso_zones.ZONES_FILE)
    _, totals = tda.reprice_month(name, paths["network"], paths["stage1_network"], paths["stage1_price"],
                                  paths["contract_value_bids"], paths["redispatch"], bz["neso_zone"],
                                  tda.DEFAULT_CONFIG, None, labels_csv=paths["support_labels"])
    assert totals["check"]["passed"]
    assert abs(totals["check"]["relative_to_gross"]) < 1e-6


def _write_nc(path, p: pd.DataFrame, carriers, buses):
    import netCDF4

    d = netCDF4.Dataset(str(path), "w")
    try:
        d.createDimension("snapshots", len(p.index))
        d.createDimension("generators_i", len(carriers))
        d.createDimension("generators_t_p_i", len(p.columns))
        snap = d.createVariable("snapshots_snapshot", "i8", ("snapshots",))
        snap.units = "hours since 2030-01-01 00:00:00"
        snap[:] = np.arange(len(p.index))
        d.createVariable("snapshots_generators", "f8", ("snapshots",))[:] = np.ones(len(p.index))
        for name, values in (("generators_i", list(carriers)), ("generators_carrier", list(carriers.values())),
                             ("generators_bus", buses), ("generators_support_type", ["CfD"] * len(carriers)),
                             ("generators_cfd_round", ["AR4"] * len(carriers)), ("generators_t_p_i", list(p.columns))):
            dim = "generators_t_p_i" if name == "generators_t_p_i" else "generators_i"
            d.createVariable(name, str, (dim,))[:] = np.array(values, dtype=object)
        d.createVariable("generators_ro_banding", "f8", ("generators_i",))[:] = np.full(len(carriers), np.nan)
        d.createVariable("generators_t_p", "f8", ("snapshots", "generators_t_p_i"))[:] = p.to_numpy()
    finally:
        d.close()


def test_hourly_dispatch_in_force_matches_the_unit_table(tmp_path):
    """The hourly path (CfD payments, settlement) and the per-unit table re-share the same volumes."""
    s1, bm, _, groups = _case()
    names = [f"U{i}" for i in range(6)]
    carriers = {n: ("wind_onshore" if i % 2 else "wind_offshore") for i, n in enumerate(names)}
    stem = "CP30_FFR_2030_ETYS_market_Jan2030_t"
    s1_path, bm_path = tmp_path / f"{stem}_s1_wholesale.nc", tmp_path / f"{stem}_balancing.nc"
    idx = pd.date_range("2030-01-01", periods=len(s1), freq="h")
    _write_nc(s1_path, pd.DataFrame(s1, index=idx, columns=names), carriers, list(groups.astype(str)))
    _write_nc(bm_path, pd.DataFrame(bm, index=idx, columns=names), carriers, list(groups.astype(str)))
    dec = np.clip(s1 - bm, 0, None)
    pr = tda.pro_rata_turn_down(s1, dec, groups)
    pd.DataFrame({"unit": names, "zone": groups, "pricing": "cfd_contract_value", "our_MWh": dec.sum(0),
                  "pro_rata_MWh": pr.sum(0), "our_cost_GBP": 0.0, "pro_rata_cost_GBP": 0.0}).to_csv(
        tmp_path / f"{stem}_pro_rata_turn_down.csv", index=False)
    (tmp_path / f"{stem}_pro_rata_turn_down.json").write_text(json.dumps(
        {"config": {"technologies": ["wind_onshore", "wind_offshore"]}, "inputs": {"stage1_network": str(s1_path)},
         "pro_rata_uplift_GBP": 0.0}))
    frame = pd.DataFrame(bm, columns=names)                       # read by position, as xarray gives it
    frame["OTHER"] = 1.0
    out = tda.balancing_dispatch_in_force(frame, bm_path, "pro_rata_zonal")
    np.testing.assert_allclose((s1 - out[names].to_numpy()).clip(0).sum(0), pr.sum(0), atol=1e-9)
    assert (out["OTHER"] == 1.0).all()
    assert tda.balancing_dispatch_in_force(frame, bm_path, "cheapest_first") is frame


def test_a_month_older_than_its_stage1_is_refused(tmp_path):
    import os
    import time

    bm, s1 = tmp_path / "x_balancing.nc", tmp_path / "x_s1_wholesale.nc"
    bm.write_text("")
    s1.write_text("")
    old = time.time() - 3600
    os.utime(bm, (old, old))
    with pytest.raises(RuntimeError, match="older than"):
        tda.check_month_vintage("x", bm, [s1])
    os.utime(bm, None)
    tda.check_month_vintage("x", bm, [s1])                          # current: no error


def test_a_repricing_older_than_its_month_is_refused(tmp_path):
    import os
    import time

    path = _month_files(tmp_path)
    old = time.time() - 3600
    for f in tmp_path.glob("*_pro_rata_turn_down.*"):
        os.utime(f, (old, old))
    with pytest.raises(RuntimeError, match="re-solved"):
        tda.read_redispatch_in_force(path, "pro_rata_zonal")
