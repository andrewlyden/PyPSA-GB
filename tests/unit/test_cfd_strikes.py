"""Task Y11 (FES 2050 plan) and year-generic stage 5: post-AR7 CfD strikes, every round's contracts live in a
model year, the AR7 offshore calibration and CfD settlement."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.market import cfd_strikes as s
from scripts.market.capacity_market_2050 import live_fraction
from scripts.public_ownership import vintage_costs as vc
from scripts.public_ownership.finance import capital_recovery_factor

needs_snapshot = pytest.mark.skipif(not Path(s.DEFAULT_SNAPSHOT).exists(), reason="workbook snapshot not extracted")


@pytest.fixture(scope="module")
def snapshot():
    return s.load_snapshot()


def test_live_fraction_spreads_expiry_through_the_year():
    # AR7 phase starting in 2030/31 (1 April 2030 onward), 20 years: expires April 2050 to March 2051.
    assert live_fraction(2030.25, 20, 2050) == pytest.approx(0.71875)
    assert live_fraction(2029.25, 20, 2050) == pytest.approx(0.03125)
    assert live_fraction(2028.25, 20, 2050) == 0.0 and live_fraction(2031.25, 20, 2050) == 1.0
    assert live_fraction(2035, 15, 2050) == pytest.approx(0.5) and live_fraction(2050, 20, 2050) == 1.0
    assert live_fraction(2040, 0, 2050) == 0.0


@needs_snapshot
def test_strike_is_the_vintage_levelised_cost_at_the_private_rate(snapshot):
    row = s.levelised_strike("Offshore wind", 2040, snapshot=snapshot)
    cost = vc.costs_for("Offshore wind", 2040)
    rate = [t for t in snapshot["technologies"] if t["technology"] == "Offshore wind"][0]["wacc"]["private"]
    inputs = s.strike_inputs_for("Offshore wind", 2040)
    # Insurance on the register's basis (Y14-prep): DESNZ insurance out, the D4(a) line (11.22 offshore) in.
    fixed = cost["fixed_om_gbp_per_kw_year"] - cost["insurance_gbp_per_kw_year"] + 11.22
    # Offshore capex is calibrated to AR7 (user, 27 Oct 2026): one factor on DESNZ capex.
    k = row["capex_calibration_factor"]
    assert k != 1.0 and row["capex_gbp_per_kw"] == pytest.approx(cost["capex_power_gbp_per_kw"] * k)
    expected = ((cost["capex_power_gbp_per_kw"] * k * capital_recovery_factor(rate, cost["economic_life_years"])
                 + fixed + inputs["use_of_system_gbp_per_kw_year"])
                / (8.76 * inputs["expected_load_factor"]) + cost["variable_om_gbp_per_mwh"])
    assert row["strike_gbp_per_mwh_2024"] == pytest.approx(expected, rel=1e-12)
    assert row["private_rate"] == pytest.approx(rate) and row["load_factor"] == 0.50
    assert "Annex A" in row["cost_source"] and "WACC Scenarios" in row["rate_source"]
    assert row["term_years"] == 20 and row["contract_end_year"] == 2060


@needs_snapshot
def test_2030_offshore_vintage_lands_on_the_ar7_strike(snapshot):
    """Calibration check on the declared inputs: the 2030 fixed-offshore vintage is within 1 GBP of AR7's 91.20."""
    assert s.levelised_strike("Offshore wind", 2030, snapshot=snapshot)["strike_gbp_per_mwh_2024"] == pytest.approx(
        91.20, abs=1.0)


@needs_snapshot
def test_sensitivities_move_strikes_without_a_resolve(snapshot):
    central = s.levelised_strike("Offshore wind", 2040, snapshot=snapshot)["strike_gbp_per_mwh_2024"]
    ramp = s.levelised_strike("Offshore wind", 2040, snapshot=snapshot, rate_path="neso_wind_ramp")
    assert ramp["rate_shift"] == pytest.approx(-0.02) and ramp["strike_gbp_per_mwh_2024"] < central
    assert s.levelised_strike("Solar PV", 2040, snapshot=snapshot, rate_path="neso_wind_ramp")["rate_shift"] == 0.0
    assert s.levelised_strike("Offshore wind", 2040, snapshot=snapshot, learning=True)[
        "strike_gbp_per_mwh_2024"] < central
    assert s.levelised_strike("Offshore wind", 2040, snapshot=snapshot, amortisation="contract_term")[
        "strike_gbp_per_mwh_2024"] > central
    # A 2030 build is before the ramp: no shift.
    assert s.levelised_strike("Offshore wind", 2030, snapshot=snapshot, rate_path="neso_wind_ramp")["rate_shift"] == 0.0


@needs_snapshot
def test_insurance_is_counted_once(snapshot):
    central = s.levelised_strike("Onshore wind", 2040, snapshot=snapshot)          # exclude + D4(a), the register's
    inc = s.levelised_strike("Onshore wind", 2040, snapshot=snapshot, insurance="include")
    bare = s.levelised_strike("Onshore wind", 2040, snapshot=snapshot, finance_insurance=False)
    cost = vc.costs_for("Onshore wind", 2040)
    ins, d4a = cost["insurance_gbp_per_kw_year"], 0.004 * cost["capex_power_gbp_per_kw"]
    assert ins > 0 and central["insurance_rule"] == "vintage_insurance_exclude_plus_d4a"
    assert inc["strike_gbp_per_mwh_2024"] - bare["strike_gbp_per_mwh_2024"] == pytest.approx(ins / (8.76 * 0.37))
    assert central["strike_gbp_per_mwh_2024"] - bare["strike_gbp_per_mwh_2024"] == pytest.approx(d4a / (8.76 * 0.37))


def test_load_factor_override_is_the_solved_year_hook():
    snapshot = s.load_snapshot() if Path(s.DEFAULT_SNAPSHOT).exists() else None
    if snapshot is None:
        pytest.skip("workbook snapshot not extracted")
    a = s.levelised_strike("Solar PV", 2040, snapshot=snapshot)
    b = s.levelised_strike("Solar PV", 2040, snapshot=snapshot, load_factor=0.10)
    assert b["load_factor_source"] == "override"
    fixed_part = a["strike_gbp_per_mwh_2024"] - a["variable_om_gbp_per_mwh"]
    assert b["strike_gbp_per_mwh_2024"] - b["variable_om_gbp_per_mwh"] == pytest.approx(fixed_part * 0.12 / 0.10)


def test_known_contracts_in_2050():
    known = s.known_contracts(2050)
    by_round = known.groupby("cfd_round")["live_mw"].sum()
    for expired in ("AR1", "AR2", "AR3", "AR4", "AR5", "AR6", "Investment Contract", "DRAX_BRIDGE", "POWER_BECCS",
                    "MGT_IC"):
        assert by_round.get(expired, 0.0) == 0.0, expired
    # Contracts audit (13 Nov 2026): AR7a starts on LCCC's unit dates (2029-31), so on 20 years part is live in 2050.
    assert 0 < by_round.get("AR7a", 0.0) < known.loc[known["cfd_round"].eq("AR7a"), "capacity_mw"].sum()
    ar7 = known[known["cfd_round"].eq("AR7")]
    assert ar7["capacity_mw"].sum() == pytest.approx(8437.5)          # 6,865 + 1,380 + 192.5 MW
    assert 0 < ar7["live_mw"].sum() < ar7["capacity_mw"].sum()
    hpc = known[known["cfd_round"].eq("HPC")]
    assert hpc["live_mw"].sum() == pytest.approx(3260.0)
    assert hpc["strike_gbp_per_mwh_2024"].iloc[0] == pytest.approx(89.50 * 133.9 / 96.1)
    assert known.loc[known["contract_id"].eq("SIZEWELL_C"), "scheme"].iloc[0] == "RAB"
    scot = known[known["contract_id"].eq("AR7_BERWICK_BANK_B")]
    assert scot["strike_gbp_per_mwh_2024"].iloc[0] == pytest.approx(89.49, abs=0.02)


def test_cfd_settlement_on_synthetic_hours():
    index = pd.date_range("2050-01-01", periods=4, freq="h")
    price = pd.Series([50.0, -5.0, 100.0, 20.0], index=index)
    dispatch = pd.DataFrame({"wind": [10.0, 10.0, 10.0, 10.0], "hpc": [5.0, 5.0, 5.0, 5.0]}, index=index)
    contracts = pd.DataFrame([
        {"unit": "wind", "strike_gbp_per_mwh_2024": 80.0, "reference_price": "intermittent_hourly",
         "negative_price_rule": "ar4_plus", "live_share": 1.0},
        {"unit": "hpc", "strike_gbp_per_mwh_2024": 124.7, "reference_price": "baseload_seasonal",
         "negative_price_rule": "none", "live_share": 0.5}])
    demand = pd.Series([1.0, 3.0, 1.0, 1.0], index=index)
    out = s.cfd_settlement(dispatch, contracts, price, demand_mw=demand).set_index("unit")
    # Wind: the -5 hour is unpaid (AR4 rule); the rest pays strike - price, and pays back above strike.
    assert out.loc["wind", "unpaid_mwh"] == 10.0
    assert out.loc["wind", "payment_gbp"] == pytest.approx(10 * ((80 - 50) + (80 - 100) + (80 - 20)))
    # HPC: one winter season, demand-weighted reference; half the year live; paid in negative hours.
    ref = (50 * 1 - 5 * 3 + 100 * 1 + 20 * 1) / 6
    assert out.loc["hpc", "payment_gbp"] == pytest.approx(0.5 * 20 * (124.7 - ref))
    assert out.loc["hpc", "settled_mwh"] == pytest.approx(10.0) and out.loc["hpc", "unpaid_mwh"] == 0.0


@needs_snapshot
def test_bill_strike_table_settles_through_the_2030_calculator(snapshot):
    from scripts.market.revenue_tracking import compute_cfd_payments

    strikes = s.modelled_strikes(technologies=["Offshore wind"], build_years=[2040], snapshot=snapshot,
                                 variants={"central": {}}, model_year=2050)
    table = s.strike_table_for_bill(strikes, s.known_contracts(2050))
    assert ("PAR7_2040", "offshore") in table.strikes and ("HPC", "nuclear") in table.strikes
    # 13 Nov 2026: the 2029-30 cohort's AR8 strikes are in the long-range table, but a labelled round the table lacks
    # never falls back to them (cfd_strike_table.StrikeTable.resolve).
    assert ("AR8_modelled", "offshore") in table.strikes
    assert table.resolve("wind_offshore", "PAR7_2099")[0] is None
    index = pd.date_range("2050-01-01", periods=2, freq="h")
    gens = pd.DataFrame({"support_type": ["CfD"], "carrier": ["wind_offshore"], "bus": ["b"], "cfd_round": ["PAR7_2040"]},
                        index=["g"])
    network = SimpleNamespace(generators=gens, snapshots=index,
                              generators_t=SimpleNamespace(p=pd.DataFrame({"g": [100.0, 100.0]}, index=index)),
                              snapshot_weightings=pd.DataFrame({"generators": [1.0, 1.0]}, index=index))
    out = compute_cfd_payments(network, table, reference_price=pd.Series([40.0, 60.0], index=index))
    strike = strikes["strike_gbp_per_mwh_2024"].iloc[0]
    assert out["net_cfd_cost"].iloc[0] == pytest.approx(100 * (strike - 40) + 100 * (strike - 60))


def test_strike_inputs_carry_provenance():
    table = s.load_strike_inputs()
    assert table[["source", "source_page", "status"]].notna().all().all()
    for technology in s.MAIN_TECHNOLOGIES:
        assert s.strike_inputs_for(technology, 2045)["technology"] == technology


# ── year-generic stage 5 (27 Oct 2026) ───────────────────────────────────────

def test_the_contract_table_holds_every_round_and_matches_the_published_one():
    table = s.load_contracts()                       # the published cross-check runs inside
    rounds = set(table["cfd_round"])
    for expected in ("AR1", "AR2", "AR3", "AR4", "AR5", "AR6", "AR7", "AR7a", "Investment Contract", "MGT_IC",
                     "DRAX_BRIDGE", "POWER_BECCS", "HPC", "SZC_RAB"):
        assert expected in rounds, expected
    assert table[["source", "source_page", "status"]].notna().all().all()
    assert set(table["negative_price_rule"].fillna("none")) <= set(s.NEGATIVE_PRICE_RULES)


def test_a_drifted_contract_table_is_refused(tmp_path):
    table = pd.read_csv(s.CONTRACTS)
    table.loc[table["contract_id"].eq("AR4_OFFSHORE_2026_27"), "capacity_mw"] = 1.0
    path = tmp_path / "contracts.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="AR4 offshore"):
        s.load_contracts(path)


def test_2040_has_ar3_to_ar6_partly_live_and_2050_none_of_them():
    k40 = s.known_contracts(2040).groupby("cfd_round")["live_mw"].sum()
    k50 = s.known_contracts(2050).groupby("cfd_round")["live_mw"].sum()
    # Neart na Gaoithe (AR1, LCCC start Feb 2025) and the last AR2 phases are still live early in 2040 (LCCC dates).
    assert 0 < k40["AR1"] < 450 and k40["AR2"] < 100 and k40["Investment Contract"] == 0.0
    for live in ("AR3", "AR4", "AR5", "AR6"):
        assert k40[live] > 0, live
        assert k50[live] == 0.0, live
    # User decision (13 Nov 2026): AR3 starts on LCCC's unit dates (offshore units 2026-08 to 2028-03), so the
    # offshore contracts are wholly live in 2040; the advanced-conversion units are terminated and hold nothing.
    k = s.known_contracts(2040)
    for cid in ("AR3_OFFSHORE_2023_24", "AR3_OFFSHORE_2024_25"):
        rows = k[k["contract_id"].eq(cid)]
        assert rows["live_share"].eq(1.0).all() and rows["start_year"].between(2026.6, 2028.3).all()
    off = k[k["contract_id"].isin(["AR3_OFFSHORE_2023_24", "AR3_OFFSHORE_2024_25"])]
    assert off["live_mw"].sum() == pytest.approx(5466.0 - 12.0, abs=0.5)  # Forthwind (12 MW) terminated
    act = k[k["contract_id"].isin(["AR3_ADVANCED_CONVERSION_2023_24", "AR3_ADVANCED_CONVERSION_2024_25"])]
    assert act["live_mw"].sum() == 0.0
    assert k40["AR7"] == pytest.approx(8437.5) and k40["HPC"] == pytest.approx(3260.0)
    # The Drax bridge ends in March 2031; the declared Power BECCS CfD (2030/31, 15 years) is live in 2040.
    assert k40["DRAX_BRIDGE"] == 0.0 and k40["POWER_BECCS"] == pytest.approx(450.0)


def test_2050_live_contracts_are_the_ones_the_2050_table_held():
    known = s.known_contracts(2050)
    live = known.loc[known["live_share"].gt(0)]
    assert set(live["cfd_round"]) == {"AR7", "AR7a", "HPC", "SZC_RAB"}
    # On LCCC's AR7 unit dates (contracts audit, 13 Nov 2026), more of AR7 is live in 2050 than on the target years.
    assert 5212.3 < known.loc[known["cfd_round"].eq("AR7"), "live_mw"].sum() < 8437.5
    # The 2050 alias is the same table.
    pd.testing.assert_frame_equal(s.known_contracts_2050(), known)


def test_negative_pricing_terms_by_round():
    table = s.load_contracts().set_index("contract_id")
    assert table.loc["AR1_OFFSHORE_2017_18", "negative_price_rule"] == "none"
    assert table.loc["AR3_OFFSHORE_2023_24", "negative_price_rule"] == "six_hour"
    assert table.loc["AR6_OFFSHORE_2028_29", "negative_price_rule"] == "ar4_plus"
    index = pd.date_range("2040-01-01", periods=8, freq="h")
    price = pd.Series([-1.0] * 7 + [50.0], index=index)
    dispatch = pd.DataFrame({"u": 1.0}, index=index)
    out = s.cfd_settlement(dispatch, pd.DataFrame([{"unit": "u", "strike_gbp_per_mwh_2024": 80.0,
                                                     "reference_price": "intermittent_hourly",
                                                     "negative_price_rule": "six_hour"}]), price)
    assert out["unpaid_mwh"].iloc[0] == 7.0          # a run of seven negative hours: none paid


def test_old_module_names_are_the_same_module():
    from scripts.market import cfd_strikes_2050, support_labels, support_labels_2050

    assert cfd_strikes_2050 is s and support_labels_2050 is support_labels
    assert s.KNOWN_CONTRACTS == s.CONTRACTS


@needs_snapshot
def test_the_calibration_lands_2030_on_the_anchor_and_scales_every_year(snapshot):
    anchor = s.load_cost_calibration().set_index("technology").loc["Offshore wind"]
    assert anchor["anchor_build_year"] == 2030 and anchor["anchor_strike_gbp_per_mwh_2024"] == 91.20
    k = s.cost_calibration_factor("Offshore wind", snapshot=snapshot)
    assert 0.9 < k < 1.0
    assert s.levelised_strike("Offshore wind", 2030, snapshot=snapshot)["strike_gbp_per_mwh_2024"] == pytest.approx(
        91.20, abs=1e-9)
    raw = s.levelised_strike("Offshore wind", 2040, snapshot=snapshot, cost_calibration=False)
    cal = s.levelised_strike("Offshore wind", 2040, snapshot=snapshot)
    assert raw["capex_calibration_factor"] == 1.0 and cal["capex_calibration_factor"] == pytest.approx(k)
    assert cal["capex_gbp_per_kw"] == pytest.approx(raw["capex_gbp_per_kw"] * k)
    assert cal["strike_gbp_per_mwh_2024"] < raw["strike_gbp_per_mwh_2024"]
    # Technologies without an anchor are unchanged.
    assert s.cost_calibration_factor("Onshore wind", snapshot=snapshot) is None
    # Sensitivities move from the calibrated cost.
    ramp = s.levelised_strike("Offshore wind", 2040, snapshot=snapshot, rate_path="neso_wind_ramp")
    assert ramp["capex_calibration_factor"] == pytest.approx(k)


@needs_snapshot
def test_modelled_strikes_run_to_the_model_year(snapshot):
    strikes = s.modelled_strikes(technologies=["Onshore wind"], snapshot=snapshot, variants={"central": {}},
                                 model_year=2040)
    assert sorted(strikes["build_year"]) == list(range(2031, 2041))
    assert (strikes["model_year"] == 2040).all()
    row = strikes.set_index("build_year").loc[2030 + 1]
    assert row["live_share"] == pytest.approx(live_fraction(2031, int(row["term_years"]), 2040))
    old = s.strike_table_2050(technologies=["Onshore wind"], build_years=[2040], snapshot=snapshot,
                              variants={"central": {}})
    assert (old["live_share_2050"] == old["live_share"]).all() and (old["model_year"] == 2050).all()


def test_insurance_falls_back_to_defaults_for_a_non_finance_scenario():
    with pytest.raises(KeyError):
        s.finance_insurance_config(study="NO_SUCH_STUDY")
    block = s.finance_insurance_config(study="NO_SUCH_STUDY", defaults_without_study=True)
    assert block == s.finance_insurance_config(s.DEFAULTS_CONFIG)


@needs_snapshot
def test_modelled_strikes_without_the_calibration_price_desnz_capex(snapshot):
    off = s.modelled_strikes(technologies=["Offshore wind"], build_years=[2030], snapshot=snapshot,
                             variants={"central": {}}, model_year=2050, cost_calibration=False).iloc[0]
    assert off["capex_calibration_factor"] == 1.0
    assert off["strike_gbp_per_mwh_2024"] == pytest.approx(
        s.levelised_strike("Offshore wind", 2030, snapshot=snapshot, cost_calibration=False)["strike_gbp_per_mwh_2024"])
    assert off["strike_gbp_per_mwh_2024"] > 91.20 + 1.0          # v2.0's rate: 93.19 before the calibration


@needs_snapshot
def test_the_register_can_take_the_same_capex_factor(snapshot):
    factors = s.vintage_capex_factors(["Offshore wind", "Onshore wind", "Offshore wind", "Lithium batteries"],
                                      snapshot=snapshot)
    k = s.cost_calibration_factor("Offshore wind", snapshot=snapshot)
    assert list(factors) == [pytest.approx(k), 1.0, pytest.approx(k), 1.0]



def test_every_round_follows_lccc_units_terminations_and_permitted_reductions():
    """Contracts audit (13 Nov 2026): LCCC's live units, their dates, and AR4/AR6 counted once."""
    k = s.known_contracts(2040)
    by = k.groupby(["cfd_round", "contract_id"])["capacity_mw"].sum()
    ar4_off = k[k["cfd_round"].eq("AR4") & k["contract_id"].str.contains("OFFSHORE") & ~k["contract_id"].str.contains("FLOATING")]
    assert ar4_off["capacity_mw"].sum() == pytest.approx(4231.1, abs=1.0)            # Norfolk Boreas terminated
    ar6 = k[k["cfd_round"].eq("AR6") & k["technology"].eq("Offshore wind")]
    assert ar6["capacity_mw"].sum() == pytest.approx(2541.6, abs=1.0)                 # Hornsea 4 terminated
    assert k[k["cfd_round"].eq("AR7")]["capacity_mw"].sum() == pytest.approx(8437.5, abs=0.5)   # DESNZ MW kept
    # Neart na Gaoithe (AR1, LCCC start 28 Feb 2025) is still under contract early in 2040.
    ar1 = k[k["cfd_round"].eq("AR1") & k["technology"].eq("Offshore wind")]
    assert ar1["live_mw"].sum() > 0
    # A round LCCC does not list keeps the table (the Investment Contracts, HPC).
    assert k[k["cfd_round"].eq("HPC")]["capacity_mw"].sum() == pytest.approx(3260.0)
