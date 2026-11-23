"""FES 2050 review fixes, decisions 53 to 62 (user, 3 Oct 2026), on small synthetic fixtures.

53 carbon per balancing group; 54 the NESO wind-ramp strike variant; 55 HPC unit 1 and Sizewell C
under P3, and the acquisition-pricing check; 56 the LDES floor by row owner; 57 decision 18's ratio
reaches the I3 bill; 58 support lines; 59 fractional P2 shares (test_fes2050_l4.py); 60 the
electrolyser exemption sensitivity; 61 minor items; 62 the onshore repowering rule.
"""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import fleet_vintages as fv


# ── decision 62: the onshore repowering rule ─────────────────────────────────

LIVES = {name: 10.0 for name in fv.TECHNOLOGIES}
LIVES["Onshore wind"] = 25.0
RULE = {"Onshore wind": fv.RepoweringRule(0.3, 35)}


def _trajectory(series, start=2024, end=2050, tech="Onshore wind"):
    rows = [{"technology": tech, "year": y, "capacity_mw": float(series)} for y in range(start, end + 1)]
    rows += [{"technology": t, "year": y, "capacity_mw": 0.0} for t in fv.TECHNOLOGIES if t != tech
             for y in range(start, end + 1)]
    return pd.DataFrame(rows)


def _evidence(rows):
    return pd.DataFrame(rows, columns=["technology", "commissioning_year", "capacity_mw", "source", "name"])


EVIDENCE = _evidence([("Onshore wind", 1985, 10.0, "REPD", "a"),   # past the cap at the base: leaves in 2025
                      ("Onshore wind", 1995, 20.0, "REPD", "b"),   # past the decision age: runs to 2030
                      ("Onshore wind", 2010, 30.0, "REPD", "c"),   # splits: 30% at 2035, 70% at 2045
                      ("Onshore wind", 2020, 40.0, "REPD", "d")])  # splits: 30% at 2045, 70% at 2055


def test_parse_repowering_from_config_and_command_line():
    rules = fv.parse_repowering({"Onshore wind": {"share": 0.3, "max_life_years": 35}})
    assert rules == {"Onshore wind": fv.RepoweringRule(0.3, 35.0)}
    assert fv.parse_repowering(["Onshore wind=0.5:35"]) == {"Onshore wind": fv.RepoweringRule(0.5, 35.0)}
    assert fv.parse_repowering(None) == {} and fv.repowering_flag({}) == ""
    assert fv.repowering_flag(rules) == ' --repowering "Onshore wind=0.3:35"'
    with pytest.raises(ValueError):
        fv.parse_repowering({"Wind": {"share": 0.3, "max_life_years": 35}})
    with pytest.raises(ValueError):
        fv.parse_repowering({"Onshore wind": {"share": 1.3, "max_life_years": 35}})


def test_the_base_stock_splits_young_cohorts_and_caps_old_ones():
    stock = fv.base_stock({"Onshore wind": 100.0}, EVIDENCE, LIVES, repowering=RULE)["Onshore wind"]
    got = {(c.build_year, c.retirement_year, c.schedule_reason): round(c.capacity_mw, 9) for c in stock}
    assert got == {(1985, 2025, fv.REASON_PAST_MAX_LIFE_AT_BASE): 10.0,
                   (1995, 2030, fv.REASON_LIFE_EXTENSION_END): 20.0,
                   (2010, 2035, fv.REASON_REPOWERED_AT_LIFE): 9.0, (2010, 2045, fv.REASON_LIFE_EXTENSION_END): 21.0,
                   (2020, 2045, fv.REASON_REPOWERED_AT_LIFE): 12.0, (2020, 2055, fv.REASON_LIFE_EXTENSION_END): 28.0}
    # the strict bound ignores the rule (every cohort at its life)
    strict = fv.base_stock({"Onshore wind": 100.0}, EVIDENCE, LIVES, rule=fv.RULE_STRICT, repowering=RULE)
    assert {c.retirement_year for c in strict["Onshore wind"]} == {2025, 2035, 2045}


def test_the_stock_flow_conserves_fes_and_no_cohort_outlives_the_cap():
    out = fv.build_fleet_vintages(_trajectory(100.0), EVIDENCE, LIVES, repowering=RULE)
    assert out["summary"]["conservation_max_abs_gap_mw"] < 1e-9
    mix = out["mix"].loc[out["mix"]["technology"].eq("Onshore wind")]
    assert mix["capacity_mw"].sum() == pytest.approx(100.0)
    assert (2050 - mix["build_year"]).max() < 35
    # 70% of the 2020 cohort survives to 2050 as legacy; the 30% repowered in 2045 is new build of 2045
    assert mix.set_index("build_year").loc[2020, "capacity_mw"] == pytest.approx(28.0)
    flows = out["flows"].set_index(["technology", "year"]).loc["Onshore wind"]
    assert flows.loc[2035, "repowered_at_decision_age_mw"] == pytest.approx(9.0)
    assert flows.loc[2045, "life_extension_end_mw"] == pytest.approx(21.0)
    assert flows.loc[2045, "repowered_at_decision_age_mw"] == pytest.approx(12.0)
    # each gross addition splits too: no new cohort survives past 35 either
    assert out["summary"]["repowering"]["Onshore wind"]["share_at_decision_age"] == 0.3
    assert (mix["repowering_share"].between(0.0, 1.0)).all()


def test_without_the_rule_the_stock_flow_is_unchanged():
    a = fv.build_fleet_vintages(_trajectory(100.0), EVIDENCE, LIVES)
    b = fv.build_fleet_vintages(_trajectory(100.0), EVIDENCE, LIVES, repowering=None)
    for key in ("mix", "flows"):
        pd.testing.assert_frame_equal(a[key], b[key])
    assert "repowering_share" not in a["mix"].columns and "repowered_at_decision_age_mw" not in a["flows"].columns


def test_standing_cohorts_follow_the_rule_tree():
    rule = fv.RepoweringRule(0.3, 35)
    assert fv.standing_cohorts(2020, 25, rule, 2050) == [(2020, pytest.approx(0.7)), (2045, pytest.approx(0.3))]
    assert fv.standing_cohorts(1995, 25, rule, 2050) == [(2030, 1.0)]     # decision taken: runs to 2030, then new
    assert fv.standing_cohorts(2030, 25, rule, 2050) == [(2030, 1.0)]
    parts = fv.standing_cohorts(2010, 25, rule, 2050)
    assert sum(s for _, s in parts) == pytest.approx(1.0) and dict(parts) == {2035: pytest.approx(0.3),
                                                                              2045: pytest.approx(0.7)}


def test_the_evidence_leaves_out_northern_ireland_and_stale_records(tmp_path):
    repd = pd.DataFrame({
        "Development Status (short)": ["Operational"] * 3, "Operational": ["01/01/2010"] * 3,
        "Installed Capacity (MWelec)": [10.0, 5.0, 3.0], "Technology Type": ["Wind Onshore"] * 3,
        "Site Name": ["GB", "NI", "Stale"], "Country": ["Scotland", "Northern Ireland", "England"],
        "Ref ID": [1, 2, 3]})
    stale = tmp_path / "stale.csv"
    pd.DataFrame({"ref_id": [3], "site": ["Stale"]}).to_csv(stale, index=False)
    ev = fv.existing_fleet_evidence(repd, None, exclude_countries=["Northern Ireland"],
                                    exclude_ref_ids=fv.load_stale_records(stale))
    assert ev["name"].tolist() == ["GB"]
    assert len(fv.existing_fleet_evidence(repd, None)) == 3                  # no exclusion: unchanged


def test_a_solar_repowering_takes_no_cfd():
    cohorts = pd.DataFrame({
        "asset_id": ["S1::b2049", "S2::b2049", "W1::b2049"], "component_type": "Generator",
        "stock_flow_technology": ["Solar PV", "Solar PV", "Onshore wind"], "build_year": [2049, 2049, 2049],
        "capacity_mw": [10.0, 10.0, 10.0], "cohort_replacement_share": [0.25, 0.25, 0.4]})
    contracts = pd.DataFrame(columns=["technology", "start_year", "capacity_mw", "live_share", "contract_id", "phase",
                                      "scheme", "cfd_round"])
    out, _ = fv.label_support_contracts(cohorts, contracts, model_year=2050, no_cfd_on_repowering=("Solar PV",))
    solar = out.loc[out["stock_flow_technology"].eq("Solar PV")]
    assert solar["cfd_live_share"].tolist() == pytest.approx([0.75, 0.75])
    assert solar["repowered_merchant_share"].tolist() == [0.25, 0.25]
    assert out.loc[2, "cfd_live_share"] == pytest.approx(1.0)               # onshore repowering keeps its CfD
    plain, _ = fv.label_support_contracts(cohorts, contracts, model_year=2050)
    assert "repowered_merchant_share" not in plain.columns and plain["cfd_live_share"].eq(1.0).all()


def test_the_legacy_valuation_takes_the_same_rule():
    from scripts.public_ownership import legacy_valuation as lv

    rows = pd.DataFrame({"model_year": 2030, "carrier": ["wind_onshore", "wind_onshore", "solar_pv"],
                         "commissioning_year": [2010.0, 2000.0, 2010.0], "asset_id": ["a", "b", "c"]})
    settings = lv.settings_from_config({"repowering_share_by_carrier": {"wind_onshore": 0.3},
                                        "max_life_years_by_carrier": {"wind_onshore": 35}})
    years = np.arange(2030, 2071)
    w = lv.repowering_weights(rows, settings, years)
    assert w["applies"].tolist() == [True, True, False]
    a = dict(zip(years, w["weight"][0]))
    assert a[2034] == 1.0 and a[2035] == pytest.approx(0.7) and a[2044] == pytest.approx(0.7) and a[2045] == 0.0
    b = dict(zip(years, w["weight"][1]))   # past the decision age in 2030: whole to the cap (2035)
    assert b[2034] == 1.0 and b[2035] == 0.0
    assert lv.settings_from_config(None)["repowering_share_by_carrier"] == {}


def test_the_survival_schedule_splits_repowered_rows():
    from scripts.public_ownership.acquisition_carry_forward import repowering_pieces

    rows = pd.DataFrame({"capacity_mw": [100.0, 50.0], "retire": [2045.0, 2040.0],
                         "repowering_year": [2035.0, np.nan], "repowering_share": [0.3, np.nan]})
    pieces = repowering_pieces(rows)
    assert pieces["capacity_mw"].tolist() == pytest.approx([50.0, 30.0, 70.0]) or \
        sorted(pieces["capacity_mw"].round(9).tolist()) == [30.0, 50.0, 70.0]
    assert sorted(pieces["retire"].tolist()) == [2035.0, 2040.0, 2045.0]
    plain = rows.drop(columns=["repowering_year", "repowering_share"])
    assert repowering_pieces(plain) is plain


# ── decision 53: carbon per balancing group ──────────────────────────────────

def test_balancing_months_are_split_by_group_and_carbon_read_per_group():
    from scripts.public_ownership.exchequer_ledger import balancing_carbon, balancing_scenarios_by_group

    patterns = {f"p2_{p}": f"FES25_HT_2050_{{month}}_p2_bids_{p}pct" for p in (25, 100)}
    months = [f"FES25_HT_2050_{m}_p2_bids_{p}pct" for p in (25, 100) for m in ("Jan", "Feb")]
    split = balancing_scenarios_by_group(months, patterns)
    assert split == {"p2_25": months[:2], "p2_100": months[2:]}
    with pytest.raises(ValueError, match="matches no"):
        balancing_scenarios_by_group(months + ["FES25_HT_2050_Jan_market"], patterns)
    carbon = {"balancing_by_group": {"p2_25": {"carbon_receipts_gbp": 1.0}, "p2_100": {"carbon_receipts_gbp": 2.0}}}
    assert balancing_carbon(carbon, "p2_100")["carbon_receipts_gbp"] == 2.0
    with pytest.raises(ValueError):
        balancing_carbon(carbon, "other")
    # one group (every 2030 ledger): the single measure, as before
    assert balancing_carbon({"balancing": {"carbon_receipts_gbp": 5.0}}, "cp30_2030_cp30_network_public_bids_p2")[
        "carbon_receipts_gbp"] == 5.0


# ── decision 56: the LDES floor by row owner ─────────────────────────────────

def test_the_ldes_floor_rate_follows_the_row_owner():
    from scripts.public_ownership.finance import capital_recovery_factor
    from scripts.public_ownership.public_account import _floor_required_revenue

    frame = pd.DataFrame({"case_id": ["public_ownership@deployment_share_25"] * 2, "ownership_share": [0.25, 0.0],
                          "annual_capital_recovery_gbp": [100.0, 100.0], "wacc": [0.05, 0.08],
                          "remaining_life_years": [30.0, 30.0], "annual_fixed_om_gbp": [0.0, 0.0],
                          "annual_variable_om_gbp": [0.0, 0.0]})
    rates = {"private": 0.045, "public_ownership": None}
    by_case = _floor_required_revenue(frame, rates)              # 2030's rule: the case's rate (null: own WACC)
    assert by_case.tolist() == pytest.approx([100.0, 100.0])
    by_owner = _floor_required_revenue(frame, rates, "row_owner")
    assert by_owner[0] == pytest.approx(100.0)                   # state-held: the case's rate (its own WACC)
    expect = 100.0 / capital_recovery_factor(0.08, 30) * capital_recovery_factor(0.045, 30)
    assert by_owner[1] == pytest.approx(expect)                  # private: the private 4.5%


def test_the_floor_rate_key_is_validated_and_defaults_to_case():
    from scripts.public_ownership.public_account import account_settings

    assert "ldes_floor_rate_key" not in account_settings({})
    s = account_settings({"consumer_bill": {"ldes_cap_and_floor": {"floor_return": {"rate_key": "row_owner"}}}})
    assert s["ldes_floor_rate_key"] == "row_owner"
    with pytest.raises(ValueError):
        account_settings({"consumer_bill": {"ldes_cap_and_floor": {"floor_return": {"rate_key": "owner"}}}})


# ── decision 57: decision 18's ratio reaches the I3 bill ─────────────────────

def test_the_ledger_refuses_an_institution_built_without_the_group_ratio():
    from scripts.public_ownership.exchequer_ledger import check_institution_fleet_ratio

    block = {"reference_run_groups": {"g": {"institution_comparison": {"forecast_error_charge": {"fleet_ratio": 2.32}}}}}
    check_institution_fleet_ratio({"forecast_error_fleet_ratio": 2.32}, block, "g")
    with pytest.raises(ValueError, match="2.32"):
        check_institution_fleet_ratio({"forecast_error_charge_source": "fleet ratio 1.1"}, block, "g")
    check_institution_fleet_ratio({}, {"reference_run_groups": {"cp30": {}}}, "cp30")   # 2030: no ratio, no check


def test_the_institution_refuses_a_ratio_it_cannot_apply(tmp_path):
    from scripts.public_ownership import nationalised_institutions as ni

    tariff = tmp_path / "t.json"
    tariff.write_text(json.dumps({"primary_case_id": "p3", "cases": {}}), encoding="utf-8")
    mc = tmp_path / "mc.csv"
    pd.DataFrame({"generator": ["g"]}).to_csv(mc, index=False)
    with pytest.raises(ValueError, match="decision 18"):
        ni.main([str(tariff), str(tmp_path / "o.json"), str(tmp_path / "o.csv"), "--market-scenarios", "m",
                 "--central-scenarios", "c", "--marginal-costs", str(mc), "--forecast-error-fleet-ratio", "2.32"])


# ── decision 55: HPC unit 1 and Sizewell C under P3 ──────────────────────────

def _nat(capital, legacy, carried=None, rav=None, basis=None):
    n = len(capital)
    nat = pd.DataFrame({"capacity_mw": [100.0] * n, "capex_power_gbp_per_kw": [1000.0] * n,
                        "support_basis": basis or [""] * n}, index=[f"a{i}" for i in range(n)])
    s = lambda v: pd.Series(v if v is not None else [False] * n, index=nat.index)  # noqa: E731
    return nat, pd.Series(capital, index=nat.index), s([True] * n), s(legacy), s(carried), s(rav)


def test_the_check_fails_on_an_asset_both_acquired_and_paid_as_new_build():
    from scripts.public_ownership.nationalised_tariff import check_acquisition_pricing

    nat, capital, owned, legacy, carried, rav = _nat([0.0, 50.0], [False, False], carried=[False, True])
    with pytest.raises(ValueError, match="also paid capital as new build"):
        check_acquisition_pricing(nat, capital, owned=owned, legacy=legacy, carried=carried, rav=rav)


def test_the_check_fails_on_a_state_held_asset_with_no_price():
    from scripts.public_ownership.nationalised_tariff import check_acquisition_pricing

    nat, capital, owned, legacy, carried, rav = _nat([50.0, 0.0], [False, False])
    with pytest.raises(ValueError, match="no price"):
        check_acquisition_pricing(nat, capital, owned=owned, legacy=legacy, carried=carried, rav=rav)
    # priced, carried, acquired at RAV, legacy or sunk on an ended contract: all pass
    nat, capital, owned, legacy, carried, rav = _nat([50.0, 0.0, 0.0, 0.0, 0.0], [False, True, False, False, False],
                                                     carried=[False, False, True, False, False],
                                                     rav=[False, False, False, True, False],
                                                     basis=["", "", "", "", "contract_expired_merchant"])
    capital.iloc[3] = 0.0
    check_acquisition_pricing(nat, capital, owned=owned, legacy=legacy, carried=carried, rav=rav)


def test_the_regulated_asset_value_is_the_capital_the_allowance_pays():
    from scripts.public_ownership.finance import capital_recovery_factor
    from scripts.public_ownership.nationalised_tariff import regulated_asset_value

    rows = pd.DataFrame({"annual_capital_recovery_gbp": [1000.0 * capital_recovery_factor(0.0673, 45)],
                         "wacc": [0.0673], "remaining_life_years": [45.0]})
    assert regulated_asset_value(rows).iloc[0] == pytest.approx(1000.0)


# ── decision 54: the NESO wind-ramp strike variant ───────────────────────────

class _Table:
    def __init__(self, strikes):
        self.strikes = strikes


def test_rate_path_offsets_are_the_flat_strike_less_the_paths_own():
    from scripts.public_ownership.contract_book_2050 import load_rate_path_offsets, rate_path_strike_offsets

    flat = _Table({("PAR7_2035", "onshore"): 80.0, ("PAR7_2035", "solar"): 50.0, ("AR7", "offshore"): 91.0})
    ramp = _Table({("PAR7_2035", "onshore"): 74.0, ("PAR7_2035", "solar"): 50.0, ("AR7", "offshore"): 85.0})
    assert rate_path_strike_offsets(flat, ramp) == {("PAR7_2035", "onshore"): 6.0}   # known contracts never move
    assert load_rate_path_offsets(None) == {}


def test_the_register_cuts_every_ramp_contract_by_its_offset():
    from scripts.public_ownership.calculate_financing_scenarios import rate_path_row_offsets

    merged = pd.DataFrame({"support_type": ["CfD", "CfD", "merchant"], "cfd_round": ["PAR7_2035", "AR7", ""],
                           "stock_flow_technology": ["Onshore wind", "Offshore wind", "Onshore wind"],
                           "cfd_paid_mwh": [1000.0, 1000.0, 0.0]})
    out = rate_path_row_offsets(merged, {"neso_wind_ramp": {("PAR7_2035", "onshore"): 6.0}})
    assert out["neso_wind_ramp"]["per_mwh"].tolist() == [6.0, 0.0, 0.0]
    assert out["neso_wind_ramp"]["gbp"].tolist() == [6000.0, 0.0, 0.0]
    assert rate_path_row_offsets(merged, None) is None


def test_strike_adjustments_carry_the_offset_only_when_given():
    from scripts.public_ownership.project_public_ownership_consumer_bill import strike_basis_offset_saving

    adj = pd.DataFrame({"case_id": ["a", "a", "b"], "strike_basis_offset_gbp_per_mwh_2024": [6.0, 0.0, 0.0],
                        "eligible_dispatch_mwh": [100.0, 50.0, 100.0]})
    assert strike_basis_offset_saving(adj, "a") == 600.0
    assert strike_basis_offset_saving(adj.drop(columns=["strike_basis_offset_gbp_per_mwh_2024"]), "a") == 0.0


def test_the_ramp_calibration_reads_the_offset_in_use():
    from scripts.public_ownership import vintage_capital_2050 as vk

    assert vk.rate_path_private_case("neso_wind_ramp") == "private@private_rate_neso_wind_ramp"
    reg = pd.DataFrame({"case_id": ["private@private_rate_neso_wind_ramp"] * 2, "support_type": "CfD",
                        "cfd_round": "PAR7_2035", "carrier": "wind_onshore", "producer_rent_gbp": [600.0, 0.0],
                        "cfd_paid_mwh": [100.0, 100.0], "cfd_strike_offset_gbp_per_mwh": [2.0, 2.0],
                        "capacity_mw": 1.0, "dispatch_mwh": 100.0})

    class T:
        strikes = {("PAR7_2035", "onshore"): 80.0}

        def resolve(self, carrier, rnd):
            return ("PAR7_2035", "onshore"), False

    frame = vk.derive_calibration(reg, pd.DataFrame(columns=["scheme", "unit"]), T(), amortisation="contract_term",
                                  base_strikes={("PAR7_2035", "onshore"): 70.0}, base_dpa={}, source="t",
                                  case_id="private@private_rate_neso_wind_ramp", include_dpa=False)
    # in use 80 - 2 = 78; rent 600 over 200 MWh paid: break-even 75
    assert frame["value_in_use"].iloc[0] == pytest.approx(78.0)
    assert frame["calibrated_value"].iloc[0] == pytest.approx(75.0)


# ── decision 58: support lines ───────────────────────────────────────────────

def _ancillary(tmp_path):
    rows = [("dc", "frequency_response", "Battery", 0.9, "bsuos_other_services", 100.0),
            ("dc", "frequency_response", "other", 0.1, "bsuos_other_services", 100.0),
            ("volt", "reactive", "CCS Gas", 1.0, "bsuos_voltage_and_stability_constraints", 50.0)]
    frame = pd.DataFrame(rows, columns=["service", "service_group", "provider_carrier", "participation_share",
                                        "in_bill_line", "spend_2050_central_gbp_m_2024"])
    frame["spend_2050_low_gbp_m_2024"] = frame["spend_2050_central_gbp_m_2024"] / 2
    frame["spend_2050_high_gbp_m_2024"] = frame["spend_2050_central_gbp_m_2024"] * 2
    path = tmp_path / "anc.csv"
    frame.to_csv(path, index=False)
    return path


def test_ancillary_revenue_goes_to_the_providers_by_capacity(tmp_path):
    from scripts.public_ownership import support_lines_2050 as sl

    assets = pd.DataFrame({"asset_id": ["B1", "B2", "G1", "W1"], "carrier": ["Battery", "Battery", "CCS Gas", "wind_onshore"],
                           "capacity_mw": [100.0, 300.0, 500.0, 50.0]})
    out, report = sl.ancillary_by_row(assets, sl.load_ancillary(_ancillary(tmp_path)))
    assert out.loc["B1", sl.COLUMN_ANCILLARY] == pytest.approx(90e6 * 0.25)
    assert out.loc["G1", sl.COLUMN_ANCILLARY] == pytest.approx(50e6)
    assert out.loc["G1", sl.COLUMN_ANCILLARY_VOLTAGE] == pytest.approx(50e6)
    assert out.loc["W1", sl.COLUMN_ANCILLARY] == 0.0
    assert report["unallocated_gbp"]["other"] == pytest.approx(10e6)
    assert report["rate_gbp_per_kw_year"]["Battery"] == pytest.approx(90e6 / 400e3)


def test_gate_fees_top_ups_and_booking(tmp_path):
    from scripts.public_ownership import support_lines_2050 as sl

    assets = pd.DataFrame({"asset_id": ["E", "W"], "carrier": ["waste_to_energy", "wind_offshore"],
                           "dispatch_mwh": [1000.0, 5000.0]})
    fee = sl.gate_fee_by_row(assets, 200.0)
    assert fee.tolist() == [200000.0, 0.0]
    private = pd.DataFrame({"case_id": "private", "asset_id": ["W", "B", "X"],
                            "stock_flow_technology": ["Offshore wind", "Biomass - unabated", "Offshore wind"],
                            "support_basis": ["contract_expired_merchant", "", "post_ar7_modelled"],
                            "market_revenue_gbp": [50.0, 10.0, 0.0], "annual_fixed_om_gbp": [80.0, 30.0, 99.0],
                            "annual_variable_om_gbp": [0.0, 5.0, 0.0], "fuel_carbon_gbp": [0.0, 20.0, 0.0],
                            sl.COLUMN_ANCILLARY: [10.0, 0.0, 0.0]})
    top = sl.top_up_by_row(private, {"Offshore wind": "expired", "Biomass - unabated": "all"})
    assert top.to_dict() == {"W": 20.0, "B": 45.0, "X": 0.0}
    reg = pd.DataFrame({"case_id": ["private", "p2"], "asset_id": ["W", "W"], "producer_rent_gbp": [-20.0, -20.0],
                        "public_surplus_before_recycling_gbp": [0.0, -20.0], "ownership_share": [0.0, 1.0]})
    booked = sl.book_support(reg, {sl.COLUMN_TOP_UP: top})
    assert booked["producer_rent_gbp"].tolist() == [0.0, 0.0]
    assert booked["public_surplus_before_recycling_gbp"].tolist() == [0.0, 0.0]
    assert sl.settings(None) is None
    with pytest.raises(sl.SupportLinesError):
        sl.settings({"ancillary": {}})


def test_the_public_account_rent_takes_the_support_income():
    from scripts.public_ownership.public_account import SUPPORT_INCOME_COLUMNS

    assert set(SUPPORT_INCOME_COLUMNS) == {"ancillary_services_revenue_gbp", "efw_gate_fee_revenue_gbp",
                                           "life_extension_top_up_gbp", "battery_revenue_top_up_gbp",
                                           "battery_subhourly_revenue_gbp"}


# ── decision 60: the electrolyser exemption sensitivity ──────────────────────

def test_the_exemption_case_is_exact_on_the_runs_volumes():
    from scripts.consumer_bill.project_future_bill import levy_and_cm_sensitivities

    frame = pd.DataFrame({"component_id": ["cfd", "capacity_market", "bsuos_other_services"],
                          "annual_gbp_per_3100kwh_ex_vat": [100.0, 20.0, 10.0]})
    recovery = {"cfd_eligible_supplier_sales_mwh": 600.0, "cfd_eligible_share": 0.95, "electrolyser_draw_mwh": 109.2,
                "gsp_to_meter_loss_factor": 1.092, "gsp_chargeable_volume_mwh": 700.0}
    assumptions = {"levy_exemptions": {"sensitivities": {"electrolysers_2050": {"electrolysers_exempt": True}}}}
    out = levy_and_cm_sensitivities(frame, assumptions, recovery, 3100)
    eligible_after = 600.0 - 109.2 / 1.092 * 0.95
    expect = 100.0 * (600.0 / eligible_after - 1.0) + 20.0 * (700.0 / (700.0 - 109.2) - 1.0)
    assert out["levy_exemption_electrolysers_2050"] == pytest.approx(expect)


def test_the_2050_bill_names_the_exemption_case_and_keeps_electrolysers_central():
    a = json.loads(open("data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json",
                        encoding="utf-8").read())
    levy = a["levy_exemptions"]
    assert levy["electrolysers_exempt"] is False
    assert levy["sensitivities"]["electrolysers_2050"]["electrolysers_exempt"] is True
    assert "life_extension_operating_top_up" in a["sourced_externals"]


# ── decision 61: the interconnector pass-through's system amount ─────────────

def test_the_pass_through_line_takes_the_bills_own_credit():
    from scripts.public_ownership.nationalised_tariff import INTERCONNECTOR_PASS_THROUGH_LINE, _replaced_system

    conv = {"interconnector_pass_through_household_gbp": -0.683, "interconnector_pass_through_system_gbp": -73.0e6,
            "final_demand_mwh": 651.6e6, "factor": 1.0384, "mwh": 3.1}
    assert _replaced_system(-0.683, INTERCONNECTOR_PASS_THROUGH_LINE, conv, None) == pytest.approx(-73.0e6)
    assert _replaced_system(0.683, INTERCONNECTOR_PASS_THROUGH_LINE, conv, None) == pytest.approx(73.0e6)
    plain = {k: v for k, v in conv.items() if not k.startswith("interconnector")}
    assert _replaced_system(-0.683, INTERCONNECTOR_PASS_THROUGH_LINE, plain, None) == pytest.approx(
        -0.683 * 651.6e6 / 1.0384 / 3.1)


# ── decision 59: the co-ownership rate ───────────────────────────────────────

def test_the_co_ownership_rate_blends_the_owners_annuities():
    from scripts.public_ownership.finance import capital_recovery_factor, co_ownership_rate

    r = co_ownership_rate(0.0325, 0.08, 0.25, 30)
    assert capital_recovery_factor(r, 30) == pytest.approx(
        0.25 * capital_recovery_factor(0.0325, 30) + 0.75 * capital_recovery_factor(0.08, 30), rel=1e-12)
    assert co_ownership_rate(0.0325, 0.08, 1.0, 30) == 0.0325 and co_ownership_rate(0.0325, 0.08, 0.0, 30) == 0.08


# ── the HT50 configuration ───────────────────────────────────────────────────

def test_the_ht50_entry_carries_the_decisions():
    from scripts.public_ownership.study_config import load_studies

    po = load_studies("config", validate=False).configs["FES25_HT_2050"]["public_ownership"]
    fleet = po["reference_run_groups"]["fes_ht_2050"]["fleet_vintages"]
    # F28 (d, user 26 Oct 2026): no private life extension in the central; decision 62's rule is the toggle's.
    assert fleet["retirement_rule"] == "strict_economic_life" and "repowering" not in fleet
    assert fleet["no_cfd_on_repowering"] == ["Solar PV"]
    toggle = po["reference_run_groups"]["fes_ht_2050_retirement_bound"]["fleet_vintages"]
    assert toggle["retirement_rule"] == "economic_life_with_extension"
    assert toggle["repowering"] == {"Onshore wind": {"share": 0.3, "max_life_years": 35}}
    bound = po["reference_run_groups"]["fes_ht_2050_repowering_bound"]["fleet_vintages"]["repowering"]
    assert bound["Onshore wind"]["share"] == 0.5
    val = po["acquisition"]["valuation"]
    assert "repowering_share_by_carrier" not in val
    vc = po["vintage_capital_2050"]
    assert vc["carried_from_earlier_purchase_contracts"] == ["HPC_UNIT_1"] and vc["rab_acquired_at_rav"] is True
    assert set(po["support_lines"]) == {"ancillary_services", "efw_gate_fee", "life_extension_top_up", "battery_top_up",
                                     "battery_subhourly_uplift"}
    assert "neso_wind_ramp" in vc["strike_calibration_by_rate_path"]
    assert po["consumer_bill"]["ldes_cap_and_floor"]["floor_return"]["rate_key"] == "row_owner"
    fv.parse_repowering(toggle["repowering"])
