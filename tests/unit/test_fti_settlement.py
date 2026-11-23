"""FTI-style settlement: the pure functions on small hand-checkable frames."""

import numpy as np
import pandas as pd
import pytest

from scripts.market.fti_settlement import (
    cost_recovery,
    firm_access,
    national_price,
    resource_cost,
)

IDX = pd.date_range("2030-01-01", periods=3, freq="h")


def test_national_price_is_demand_weighted_over_mapped_loads():
    mp = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [50.0, 50.0, 50.0], "X": [999.0] * 3}, index=IDX)
    load = pd.DataFrame({"la": [100.0, 100.0, 0.0], "lb": [300.0, 100.0, 100.0], "lx": [1.0] * 3}, index=IDX)
    bus = pd.Series({"la": "A", "lb": "B"})  # lx has no bus mapping and must be ignored
    p = national_price(mp, load, bus)
    assert p.iloc[0] == pytest.approx((100 * 10 + 300 * 50) / 400)
    assert p.iloc[1] == pytest.approx((100 * 20 + 100 * 50) / 200)
    assert p.iloc[2] == pytest.approx(50.0)


def test_national_price_needs_a_mapped_load():
    mp = pd.DataFrame({"A": [1.0] * 3}, index=IDX)
    load = pd.DataFrame({"l": [1.0] * 3}, index=IDX)
    with pytest.raises(ValueError):
        national_price(mp, load, pd.Series({"l": "Z"}))


def test_cost_recovery_pays_only_the_shortfall_on_scheduled_output():
    disp = pd.DataFrame({"g_cheap": [100.0] * 3, "g_dear": [50.0, 0.0, 50.0], "g_nocost": [1.0] * 3}, index=IDX)
    mc = pd.Series({"g_cheap": 10.0, "g_dear": 80.0})
    price = pd.Series([60.0, 60.0, 100.0], index=IDX)
    cr = cost_recovery(disp, mc, price)
    assert cr["g_cheap"] == 0.0
    assert cr["g_dear"] == pytest.approx((80 - 60) * 50 + 0 + 0)  # hour 3 price covers cost
    assert "g_nocost" not in cr.index


def test_firm_access_pays_lost_margin_only_where_output_fell():
    central = pd.DataFrame({"g": [0.0, 100.0, 100.0]}, index=IDX)
    market = pd.DataFrame({"g": [100.0, 100.0, 50.0]}, index=IDX)
    mc = pd.Series({"g": 30.0})
    price = pd.Series([50.0, 50.0, 20.0], index=IDX)  # hour 3 price below cost: no margin lost
    fa = firm_access(central, market, mc, price)
    assert fa["g"] == pytest.approx((50 - 30) * 100)


def test_resource_cost_ignores_negative_output_and_unknown_units():
    disp = pd.DataFrame({"g": [10.0, -5.0, 10.0], "u": [1.0] * 3}, index=IDX)
    assert resource_cost(disp, pd.Series({"g": 2.0})) == pytest.approx(40.0)


def test_cost_recovery_increment_charges_only_output_above_self_dispatch():
    from scripts.market.fti_settlement import cost_recovery_increment
    central = pd.DataFrame({"mustrun": [100.0] * 3, "gas": [50.0, 80.0, 80.0]}, index=IDX)
    selfd = pd.DataFrame({"mustrun": [100.0] * 3, "gas": [50.0, 50.0, 90.0]}, index=IDX)
    mc = pd.Series({"mustrun": 150.0, "gas": 120.0})
    price = pd.Series([55.0, 55.0, 130.0], index=IDX)  # hour 3 price covers gas
    cr = cost_recovery_increment(central, selfd, mc, price)
    assert cr["mustrun"] == 0.0  # same output both sides: nothing to make whole
    assert cr["gas"] == pytest.approx((120 - 55) * 30)


# link_trade_cost is tested in test_interconnector_trade_valuation.py (item IC1). The
# test that stood here pinned exports earning the import ask, which was the defect.


def test_unplanned_start_cost_counts_only_output_brought_on_from_off():
    from scripts.market.fti_settlement import unplanned_start_cost
    sched = pd.DataFrame({"g_off": [0.0, 0.0, 0.0], "g_on": [50.0, 50.0, 50.0], "w": [0.0] * 3}, index=IDX)
    phys = pd.DataFrame({"g_off": [10.0, 0.0, 30.0], "g_on": [80.0, 20.0, 50.0], "w": [5.0] * 3}, index=IDX)
    cost = unplanned_start_cost(phys, sched, premium=50.0, units=["g_off", "g_on"], threshold_mw=1.0)
    assert cost == pytest.approx(50.0 * (10.0 + 30.0))  # g_on was committed; w is not a listed unit
    assert unplanned_start_cost(phys, sched, 0.0, ["g_off"], 1.0) == 0.0


# ── audit 26 Sep 2026: the market's interconnector re-trading ─────────────────

def _ic_links():
    return pd.DataFrame({
        "marginal_cost": [90.0, 5.0],
        "efficiency": [0.975, 1.0],
        "foreign_market_pricing": ["piecewise_bidirectional", ""],
        "interconnector_name": ["ic", ""],
        "market_separate_import_export_prices": [True, False],
        "market_export_bid_offset": [50.0, 0.0],
        "market_spread_half": [0.5, 0.0],
        "market_circulation_epsilon": [0.01, 0.0],
    }, index=["IC_ic", "internal"])


def test_retrading_is_the_physical_trade_value_less_the_day_ahead_value():
    from scripts.market.fti_settlement import interconnector_retrading_cost, link_trade_cost
    links = _ic_links()
    sched = pd.DataFrame({"IC_ic": [100.0, 0.0, -50.0], "internal": [10.0] * 3}, index=IDX)
    phys = pd.DataFrame({"IC_ic": [40.0, -20.0, -50.0], "internal": [30.0] * 3}, index=IDX)
    cost = interconnector_retrading_cost(phys, sched, links)
    # Only the signed cable counts; the internal link is not re-trading.
    signed = ["IC_ic"]
    assert cost == pytest.approx(link_trade_cost(phys[signed], links) - link_trade_cost(sched[signed], links))
    # Importing 60 less at the 90 ask saves 5,400; exporting 20 more earns 0.975**2 x 40 x 20 = 760.5
    # (item D2, 27 Sep 2026: the export pays the cable loss, market_coupling.export_efficiency).
    assert cost == pytest.approx(-60 * 90 - 20 * 0.975 ** 2 * 40)
    assert interconnector_retrading_cost(sched, sched, links) == 0.0


def test_retrading_over_a_window_is_the_window_own_flows():
    from scripts.market.fti_settlement import interconnector_retrading_cost
    links = _ic_links()
    sched = pd.DataFrame({"IC_ic": [100.0, 100.0, 100.0]}, index=IDX)
    phys = pd.DataFrame({"IC_ic": [100.0, 50.0, 100.0]}, index=IDX)
    whole = interconnector_retrading_cost(phys, sched, links)
    hour = interconnector_retrading_cost(phys.iloc[1:2], sched.iloc[1:2], links)
    assert hour == pytest.approx(whole) == pytest.approx(-50 * 90)
    assert interconnector_retrading_cost(phys.iloc[:1], sched.iloc[:1], links) == 0.0


def test_a_constraint_cost_that_already_holds_retrading_is_refused():
    from scripts.market.fti_settlement import check_retrading_not_in_constraint_cost
    accounting = {"modelled_redispatch_cost_GBP": 152.1e6, "interconnector_retrading_cost_GBP": 205.6e6,
                  "modelled_redispatch_cost_including_retrading_GBP": 357.7e6}
    check_retrading_not_in_constraint_cost(152.1e6, accounting)  # the file as it is: generators and storage only
    check_retrading_not_in_constraint_cost(152.1e6, None)
    with pytest.raises(ValueError, match="twice"):
        check_retrading_not_in_constraint_cost(357.7e6, accounting)


def test_market_consumer_cost_carries_retrading_once_and_the_headline_is_net():
    from scripts.market import forecast_error_charge as fec
    from scripts.market.fti_settlement import HEADLINE_METRIC, consumer_totals
    r = {"market_energy_gbp_m": 1000.0, "market_constraint_cost_gbp_m": 150.0,
         "market_interconnector_retrading_gbp_m": 200.0, "central_energy_gbp_m": 1000.0,
         "central_cost_recovery_gbp_m": 10.0, "central_firm_access_gbp_m": 80.0,
         "central_storage_make_whole_gbp_m": 0.0, "central_ic_make_whole_gbp_m": 0.0, "market_constraint_pro_rata_uplift_gbp_m": 0.0, "market_support_lines_cfd_gbp_m": 0.0, "market_support_lines_roc_gbp_m": 0.0, "central_support_lines_cfd_gbp_m": 0.0, "central_support_lines_roc_gbp_m": 0.0, "central_realtime_correction_gbp_m": 30.0, "gb_demand_twh": 30.0,
         # information-only lines must enter no total
         "market_interconnector_retrading_accounting_info_gbp_m": 200.0,
         "central_realtime_bm_net_gbp_m": 70.0,
         # Item D4(b): each side's pass-through; the rent transfer enters no total.
         "market_ic_pass_through_credit_gbp_m": 0.0, "central_ic_pass_through_credit_gbp_m": 0.0,
         "ic_congestion_rent_transfer_info_gbp_m": 999.0}
    fe = fec.settings_from_config(None)
    t = consumer_totals(r, fe, hours=744)
    assert t["market_consumer_cost_gbp_m"] == pytest.approx(1350.0)
    assert t["central_consumer_cost_gbp_m"] == pytest.approx(1120.0)
    assert t["consumer_saving_gross_gbp_m"] == pytest.approx(230.0)
    assert t["forecast_error_charge_gbp_m"] == pytest.approx(fec.forecast_error_charge_gbp_bn(0, 0) * 1e3 * 744 / 8760)
    # The borrowed charge replaces the model's own real-time correction (26 Sep 2026):
    # the 30 already inside the central cost comes back out before the charge goes in.
    assert t["central_realtime_correction_replaced_gbp_m"] == pytest.approx(30.0)
    assert t[HEADLINE_METRIC] == pytest.approx(230.0 + 30.0 - t["forecast_error_charge_gbp_m"])
    off = consumer_totals(r, fec.settings_from_config({"enabled": False}), hours=744)
    assert off["forecast_error_charge_gbp_m"] == 0.0 and off[HEADLINE_METRIC] == pytest.approx(230.0)
    assert off["central_realtime_correction_replaced_gbp_m"] == 0.0


# ── item D4(b): interconnector pass-through on both sides and the rent transfer ──

def _rule():
    return {
        "default_gb_share_of_rent": 0.5,
        "index": {"to_price_year": 2024, "to_index": 100.0},
        "cables": {
            "capped": {"regime": "cap_and_floor", "gb_share_of_rent": 0.5, "levels_apply_to": "gb_share",
                       "cap_gbp_m": 10.0, "floor_gbp_m": 4.0, "index_from": 100.0},
            "merchant": {"regime": "exempt_merchant"},
            "irish": {"regime": "other_jurisdiction"},
        },
    }


def test_gb_half_shares_follow_the_regime_table():
    from scripts.market.fti_settlement import gb_half_shares
    shares = gb_half_shares(_rule())
    assert shares == {"capped": 0.5, "merchant": 0.5, "irish": 0.0}
    rule = _rule()
    rule["cables"]["merchant"]["gb_share_of_ownership"] = 0.0  # a declared override wins
    assert gb_half_shares(rule)["merchant"] == 0.0
    rule["cables"]["irish"]["regime"] = "mystery"
    with pytest.raises(ValueError, match="regime"):
        gb_half_shares(rule)


def test_congestion_rent_gives_up_efficiency_times_the_export():
    from scripts.interconnectors.market_coupling import interconnector_trade_cost
    from scripts.market.fti_settlement import congestion_rent_by_cable
    links = _ic_links()
    flows = pd.DataFrame({"IC_ic": [100.0, -50.0, 0.0], "internal": [5.0] * 3}, index=IDX)
    price = pd.Series([60.0, 70.0, 80.0], index=IDX)
    rent = congestion_rent_by_cable(flows, price, links)
    trade = interconnector_trade_cost(flows[["IC_ic"]], links)["IC_ic"]
    # GB receives 0.975 x 100 on the import and gives up 0.975 x 50 on the export.
    expected = 60 * 0.975 * 100 - trade.iloc[0] + (70 * 0.975 * -50 - trade.iloc[1])
    assert list(rent.index) == ["ic"]  # keyed by cable name, the bill's regime key
    assert rent["ic"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="GB price missing"):
        congestion_rent_by_cable(flows, price.iloc[:2], links)


def test_pass_through_scales_the_cap_and_floor_to_the_window():
    from scripts.market.fti_settlement import interconnector_regime_pass_through
    rule = _rule()
    rent = {"capped": 30.0e6, "merchant": 5.0e6, "irish": 1.0e6}  # GB half 15 m, above the 10 m cap
    year = interconnector_regime_pass_through(rent, rule, 1.0)
    assert year["credit_to_consumers_gbp"] == pytest.approx(5.0e6)
    month = interconnector_regime_pass_through({"capped": 2.0e6}, rule, 1.0 / 12)
    # GB half 1.0 m against a floor of 4/12 m and a cap of 10/12 m: 1.0 - 0.8333 returned
    assert month["credit_to_consumers_gbp"] == pytest.approx(1.0e6 - 10.0e6 / 12)
    low = interconnector_regime_pass_through({"capped": 2.0e6}, rule, 1.0)
    assert low["credit_to_consumers_gbp"] == pytest.approx(-(4.0e6 - 1.0e6))  # floor topped up by consumers


def test_rent_transfer_is_split_and_enters_no_consumer_total():
    from scripts.market import forecast_error_charge as fec
    from scripts.market.fti_settlement import consumer_totals, interconnector_rent_rows
    rent_m = pd.Series({"capped": 30.0e6, "irish": 4.0e6})
    rent_c = pd.Series({"capped": 20.0e6, "irish": 1.0e6})
    rows = interconnector_rent_rows(rent_m, rent_c, _rule(), hours=8760)
    assert rows["market_ic_pass_through_credit_gbp_m"] == pytest.approx(5.0)   # 15 - 10 cap
    assert rows["central_ic_pass_through_credit_gbp_m"] == pytest.approx(0.0)  # 10 at the cap
    assert rows["ic_congestion_rent_transfer_info_gbp_m"] == pytest.approx(13.0)
    assert rows["ic_congestion_rent_transfer_gb_half_info_gbp_m"] == pytest.approx(5.0)
    assert rows["ic_congestion_rent_transfer_foreign_half_info_gbp_m"] == pytest.approx(8.0)
    assert rows["market_ic_rent_gbp_m::capped"] == pytest.approx(30.0)
    base = {"market_energy_gbp_m": 1000.0, "market_constraint_cost_gbp_m": 100.0,
            "market_interconnector_retrading_gbp_m": 0.0, "central_energy_gbp_m": 1000.0,
            "central_cost_recovery_gbp_m": 0.0, "central_firm_access_gbp_m": 50.0,
            "central_storage_make_whole_gbp_m": 0.0, "central_ic_make_whole_gbp_m": 0.0, "market_constraint_pro_rata_uplift_gbp_m": 0.0, "market_support_lines_cfd_gbp_m": 0.0, "market_support_lines_roc_gbp_m": 0.0, "central_support_lines_cfd_gbp_m": 0.0, "central_support_lines_roc_gbp_m": 0.0, "central_realtime_correction_gbp_m": 0.0, "gb_demand_twh": 300.0, **rows}
    t = consumer_totals(base, fec.settings_from_config({"enabled": False}), hours=8760)
    # Market consumers get 5 back; central consumers nothing; the 13 transfer is in no total.
    assert t["market_consumer_cost_gbp_m"] == pytest.approx(1095.0)
    assert t["central_consumer_cost_gbp_m"] == pytest.approx(1050.0)
    with pytest.raises(ValueError, match="no declared regime"):
        interconnector_rent_rows(pd.Series({"unknown": 1.0}), rent_c, _rule(), hours=8760)
    with pytest.raises(KeyError, match="pass-through"):
        consumer_totals({k: v for k, v in base.items() if k != "central_ic_pass_through_credit_gbp_m"},
                        fec.settings_from_config(None), hours=8760)


def test_year_is_assessed_once_on_summed_rents():
    from scripts.market import forecast_error_charge as fec
    from scripts.market.fti_settle_year import reassess_year
    from scripts.market.fti_settlement import HEADLINE_METRIC, interconnector_rent_rows
    rule = _rule()
    months = {}
    # Two half-years: one far above the half-year cap, one below its floor.
    for name, rent in (("H1", 20.0e6), ("H2", 2.0e6)):
        r = {"market_energy_gbp_m": 500.0, "market_constraint_cost_gbp_m": 50.0,
             "market_interconnector_retrading_gbp_m": 0.0, "central_energy_gbp_m": 500.0,
             "central_cost_recovery_gbp_m": 0.0, "central_firm_access_gbp_m": 20.0,
             "central_storage_make_whole_gbp_m": 0.0, "central_ic_make_whole_gbp_m": 0.0, "market_constraint_pro_rata_uplift_gbp_m": 0.0, "market_support_lines_cfd_gbp_m": 0.0, "market_support_lines_roc_gbp_m": 0.0, "central_support_lines_cfd_gbp_m": 0.0, "central_support_lines_roc_gbp_m": 0.0, "central_realtime_correction_gbp_m": 0.0, "gb_demand_twh": 150.0, "hours": 4380.0}
        r.update(interconnector_rent_rows(pd.Series({"capped": rent}), pd.Series({"capped": rent}), rule, 4380))
        months[name] = pd.Series(r)
    table = pd.DataFrame(months)
    fe = fec.settings_from_config({"enabled": False})
    year = reassess_year(table, rule, fe, hours=8760.0)
    # Annual GB half 11 m against the 10 m cap: 1 m returned on each side.
    assert year["market_ic_pass_through_credit_gbp_m"] == pytest.approx(1.0)
    monthly_sum = table.loc["market_ic_pass_through_credit_gbp_m"].sum()
    assert monthly_sum == pytest.approx(5.0 - 1.0)  # (10 - 5) returned in H1, (2 - 1) topped up in H2
    assert year["market_consumer_cost_gbp_m"] == pytest.approx(1100.0 - 1.0)
    assert year[HEADLINE_METRIC] == pytest.approx(year["consumer_saving_gross_gbp_m"])


def test_d7_firm_access_and_cost_recovery_take_the_self_dispatch_on_forecast_price():
    """Decision D7: both transfers compare two forecast schedules, so they use that price."""
    from scripts.market.fti_settlement import fa_cr_price
    outturn = pd.Series([-1.0, 0.0, 40.0], index=IDX)
    forecast = pd.Series([12.0, 30.0, 45.0], index=IDX)
    price, basis = fa_cr_price(outturn, "copperplate", "sd_fe", forecast)
    assert price.tolist() == forecast.tolist() and "sd_fe" in basis
    # Firm access on the forecast price: 12 and 30 above a 10 cost, where the outturn gives nothing.
    central = pd.DataFrame({"g": [0.0, 0.0, 100.0]}, index=IDX)
    selfd = pd.DataFrame({"g": [100.0, 100.0, 100.0]}, index=IDX)
    mc = pd.Series({"g": 10.0})
    assert firm_access(central, selfd, mc, price)["g"] == pytest.approx(2.0 * 100 + 20.0 * 100)
    assert firm_access(central, selfd, mc, outturn)["g"] == 0.0
    # No forecast counterfactual, or the nodal basis: the energy line's price.
    for args in (("copperplate", None, None), ("nodal", "sd_fe", forecast)):
        same, _ = fa_cr_price(outturn, *args)
        assert same is outturn
    with pytest.raises(ValueError, match="missing"):
        fa_cr_price(outturn, "copperplate", "sd_fe", forecast.iloc[:2])


# ── decisions of 14 Oct 2026: lost support, storage and interconnector make-whole ──

def _support_generators():
    return pd.DataFrame({"carrier": ["wind_onshore", "wind_onshore", "wind_onshore", "CCGT"],
                         "support_type": ["CfD", "ROC", "merchant", "nan"],
                         "ro_banding": [np.nan, 0.9, np.nan, np.nan]},
                        index=["cfd", "roc", "merchant", "gas"])


def _cfd_audit(rule="none"):
    return pd.DataFrame({"generator": ["cfd", "merchant"], "support_type": ["CfD", "merchant"],
                         "strike_price_gbp_per_mwh": [80.0, np.nan], "negative_price_rule": [rule, "none"]})


def test_lost_support_pays_cfd_top_up_and_roc_value_on_the_lost_volume():
    from scripts.market.fti_settlement import lost_support, support_terms
    terms = support_terms(_support_generators(), _cfd_audit(), roc_buyout=60.0)
    assert terms["roc_value"]["roc"] == pytest.approx(54.0)  # 0.9 x 60
    assert list(terms["cfd_strike"].index) == ["cfd"]
    price = pd.Series([-10.0, 50.0, 100.0], index=IDX)
    market = pd.DataFrame({"cfd": [10.0, 10.0, 10.0], "roc": [5.0, 5.0, 0.0], "merchant": [9.0] * 3}, index=IDX)
    central = pd.DataFrame({"cfd": [0.0, 4.0, 0.0], "roc": [5.0, 0.0, 0.0], "merchant": [0.0] * 3}, index=IDX)
    s = lost_support(central, market, price, terms)
    # CfD: 90 x 10 + 30 x 6 + 0 (price above strike); ROC: 54 x 5 in hour two only.
    assert s.loc["cfd", "cfd"] == pytest.approx(90 * 10 + 30 * 6)
    assert s.loc["roc", "roc"] == pytest.approx(54.0 * 5)
    assert "merchant" not in s.index or s.loc["merchant"].sum() == 0.0


def test_negative_price_rule_removes_the_top_up_below_zero():
    from scripts.market.fti_settlement import lost_support, support_terms
    terms = support_terms(_support_generators(), _cfd_audit(rule="six_hour"), roc_buyout=60.0)
    price = pd.Series([-10.0, 50.0, 100.0], index=IDX)
    market = pd.DataFrame({"cfd": [10.0, 10.0, 10.0]}, index=IDX)
    central = pd.DataFrame({"cfd": [0.0, 4.0, 0.0]}, index=IDX)
    assert lost_support(central, market, price, terms).loc["cfd", "cfd"] == pytest.approx(30 * 6)


def test_a_roc_unit_without_banding_is_refused():
    from scripts.market.fti_settlement import support_terms
    gens = _support_generators()
    gens.loc["roc", "ro_banding"] = np.nan
    with pytest.raises(ValueError, match="ro_banding"):
        support_terms(gens, _cfd_audit(), roc_buyout=60.0)


def test_storage_is_made_whole_week_by_week_to_its_self_dispatch_earnings():
    from scripts.market.fti_settlement import storage_make_whole
    idx = pd.date_range("2030-01-05", periods=4, freq="D")  # Sat, Sun | Mon, Tue: two calendar weeks
    price = pd.Series([10.0, 100.0, 10.0, 100.0], index=idx)
    sd = pd.DataFrame({"b": [-1.0, 1.0, -1.0, 1.0]}, index=idx)      # earns 90 a week
    central = pd.DataFrame({"b": [-1.0, 0.5, 1.0, -1.0]}, index=idx)  # week 1: 40; week 2: -90
    mw = storage_make_whole(central, sd, price, price)
    assert mw["b"] == pytest.approx((90 - 40) + (90 - (-90)))
    better = pd.DataFrame({"b": [0.0, 2.0, 0.0, 2.0]}, index=idx)   # earns more than self-dispatch
    assert storage_make_whole(better, sd, price, price)["b"] == 0.0


def test_interconnector_holders_are_made_whole_day_by_day():
    from scripts.market.fti_settlement import interconnector_make_whole
    idx = pd.date_range("2030-01-01", periods=4, freq="12h")  # two days
    sd = pd.DataFrame({"ic": [10.0, 10.0, 5.0, 5.0]}, index=idx)
    central = pd.DataFrame({"ic": [2.0, 3.0, 20.0, 0.0]}, index=idx)
    # Day 1: 20 - 5 = 15 owed; day 2: central earns more, nothing owed and nothing clawed back.
    assert interconnector_make_whole(sd, central)["ic"] == pytest.approx(15.0)


def test_make_whole_lines_are_central_costs_and_the_headline_is_as_solved():
    from scripts.market import forecast_error_charge as fec
    from scripts.market.fti_settlement import HEADLINE_METRIC, HEADLINE_METRIC_IN_FORCE, consumer_totals
    r = {"market_energy_gbp_m": 1000.0, "market_constraint_cost_gbp_m": 150.0,
         "market_interconnector_retrading_gbp_m": 0.0, "central_energy_gbp_m": 1000.0,
         "central_cost_recovery_gbp_m": 10.0, "central_firm_access_gbp_m": 80.0,
         "central_realtime_correction_gbp_m": 0.0, "gb_demand_twh": 30.0,
         "market_ic_pass_through_credit_gbp_m": 0.0, "central_ic_pass_through_credit_gbp_m": 0.0,
         "central_storage_make_whole_gbp_m": 7.0, "central_ic_make_whole_gbp_m": 3.0,
         "market_constraint_pro_rata_uplift_gbp_m": 40.0, "market_support_lines_cfd_gbp_m": 0.0, "market_support_lines_roc_gbp_m": 0.0, "central_support_lines_cfd_gbp_m": 0.0, "central_support_lines_roc_gbp_m": 0.0}
    t = consumer_totals(r, fec.settings_from_config({"enabled": False}), hours=744)
    assert t["central_consumer_cost_gbp_m"] == pytest.approx(1100.0)
    assert t[HEADLINE_METRIC] == pytest.approx(1150.0 - 1100.0)
    # The market on its in-force (pro rata) basis: the uplift sits beside the headline.
    assert t["market_consumer_cost_in_force_basis_gbp_m"] == pytest.approx(1190.0)
    assert t[HEADLINE_METRIC_IN_FORCE] == pytest.approx(90.0)
    with pytest.raises(KeyError, match="make-whole"):
        consumer_totals({k: v for k, v in r.items() if k != "central_storage_make_whole_gbp_m"},
                        fec.settings_from_config({"enabled": False}), hours=744)


def test_support_lines_pay_two_way_top_ups_and_roc_value_on_metered_output():
    from scripts.market.fti_settlement import support_on_output, support_terms
    terms = support_terms(_support_generators(), _cfd_audit(rule="six_hour"), roc_buyout=60.0)
    price = pd.Series([-10.0, 50.0, 100.0], index=IDX)
    output = pd.DataFrame({"cfd": [10.0, 10.0, 10.0], "roc": [5.0, 5.0, -1.0], "merchant": [9.0] * 3}, index=IDX)
    s = support_on_output(output, price, terms)
    # CfD: nothing below zero (negative-price rule), 30 x 10, then pays back 20 x 10 above the strike.
    assert s.loc["cfd", "cfd"] == pytest.approx(30 * 10 - 20 * 10)
    # ROC: 54 per MWh generated; negative output earns nothing.
    assert s.loc["roc", "roc"] == pytest.approx(54.0 * 10)
    assert "merchant" not in s.index or s.loc["merchant"].sum() == 0.0


def test_support_lines_offset_the_lost_support_firm_access_pays():
    """Turn-down paid its lost top-up through firm access no longer earns it on the CfD line."""
    from scripts.market import forecast_error_charge as fec
    from scripts.market.fti_settlement import HEADLINE_METRIC, consumer_totals
    r = {"market_energy_gbp_m": 1000.0, "market_constraint_cost_gbp_m": 0.0,
         "market_interconnector_retrading_gbp_m": 0.0, "central_energy_gbp_m": 1000.0,
         "central_cost_recovery_gbp_m": 0.0, "central_firm_access_gbp_m": 60.0,  # 60 of lost CfD support
         "central_realtime_correction_gbp_m": 0.0, "gb_demand_twh": 30.0,
         "market_ic_pass_through_credit_gbp_m": 0.0, "central_ic_pass_through_credit_gbp_m": 0.0,
         "central_storage_make_whole_gbp_m": 0.0, "central_ic_make_whole_gbp_m": 0.0,
         "market_constraint_pro_rata_uplift_gbp_m": 0.0,
         "market_support_lines_cfd_gbp_m": 500.0, "market_support_lines_roc_gbp_m": 0.0,
         "central_support_lines_cfd_gbp_m": 440.0, "central_support_lines_roc_gbp_m": 0.0}
    t = consumer_totals(r, fec.settings_from_config({"enabled": False}), hours=744)
    assert t[HEADLINE_METRIC] == pytest.approx(0.0)
    assert t["consumer_saving_fti_comparable_net_excluding_support_lines_info_gbp_m"] == pytest.approx(-60.0)
    with pytest.raises(KeyError, match="support lines"):
        consumer_totals({k: v for k, v in r.items() if k != "central_support_lines_roc_gbp_m"},
                        fec.settings_from_config({"enabled": False}), hours=744)


def test_gb_resource_saving_counts_each_side_premium_as_paid_abroad():
    from scripts.market.fti_settlement import gb_resource_saving
    r = {"resource_saving_gbp_m": 820.0, "market_counter_trade_premium_gbp_m": 1842.0,
         "central_counter_trade_premium_gbp_m": 5.0}
    assert gb_resource_saving(r) == pytest.approx(820.0 + 1842.0 - 5.0)

# ── P1/P2 under central dispatch (20 Oct 2026): per-unit transfers and the payment abroad ──

def test_unit_transfers_pay_each_unit_its_own_transfers_and_sum_to_the_rows():
    from scripts.market.fti_settlement import UNIT_TRANSFER_COLUMNS, check_unit_transfers, unit_transfers
    cr = pd.Series({"gas": 5.0e6, "EU_x": 1.0e6, "wind": 0.0})
    fa = pd.Series({"wind": 3.0e6, "gas": 0.0})
    support = pd.DataFrame({"cfd": [2.0e6], "roc": [0.5e6]}, index=["wind"])
    store = pd.Series({"bat": 0.25e6, "idle": 0.0})
    carriers = pd.Series({"gas": "CCGT", "EU_x": "EU_market", "wind": "wind_offshore"})
    units = unit_transfers(cr, fa, support, store, carriers, pd.Series({"bat": "battery", "idle": "battery"}))
    assert list(units.columns) == UNIT_TRANSFER_COLUMNS
    by = units.set_index("asset_id")
    assert set(by.index) == {"gas", "EU_x", "wind", "bat"}            # nothing paid, nothing listed
    assert by.at["wind", "firm_access_margin_gbp"] == 3.0e6 and by.at["wind", "firm_access_cfd_support_gbp"] == 2.0e6
    assert by.at["bat", "component_type"] == "StorageUnit" and by.at["bat", "carrier"] == "battery"
    assert by.at["EU_x", "carrier"] == "EU_market"                      # cost recovery on an import is a purchase
    rows = {"central_cost_recovery_gbp_m": 6.0, "central_firm_access_margin_gbp_m": 3.0,
            "central_firm_access_cfd_support_gbp_m": 2.0, "central_firm_access_roc_support_gbp_m": 0.5,
            "central_storage_make_whole_gbp_m": 0.25}
    check_unit_transfers(units, rows)
    with pytest.raises(ValueError, match="cost_recovery_gbp"):
        check_unit_transfers(units, {**rows, "central_cost_recovery_gbp_m": 6.1})


def test_signed_trade_cost_is_what_the_rent_nets_from_the_gb_end():
    from scripts.market.fti_settlement import congestion_rent_by_cable, signed_trade_cost
    links = _ic_links()
    flows = pd.DataFrame({"IC_ic": [100.0, -50.0, 0.0], "internal": [5.0] * 3}, index=IDX)
    price = pd.Series([60.0, 70.0, 80.0], index=IDX)
    gb_end = float((flows["IC_ic"] * 0.975 * price).sum())
    # The GB end of the trade splits into the payment abroad and the congestion rent, exactly.
    assert signed_trade_cost(flows, links) + congestion_rent_by_cable(flows, price, links)["ic"] == pytest.approx(gb_end)
    assert signed_trade_cost(flows[["internal"]], links) == 0.0          # internal links pay nobody abroad
