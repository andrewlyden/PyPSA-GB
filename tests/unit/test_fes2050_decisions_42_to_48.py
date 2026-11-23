"""FES 2050 decisions 30, 32, 42, 43, 44, 45 and 48 (user, 3 Oct 2026), on the decision-41 toy fleet.

42: legacy plant whose RO or CfD has ended is at operating cost only. 44: Power BECCS's carbon
CfD runs on its own term. 32: Sizewell C's RAB outside P2's boundary (a switch, off centrally).
45: the declared-output strike beside the calibrated one. 30: the Capacity Market memo of who is
paid. 43 and 48 are configuration on the FES25_HT_2050 entry, checked on the resolved block.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import vintage_capital_2050 as vk
from scripts.public_ownership.calculate_financing_scenarios import (
    book_carbon_removal_cfd, carbon_cfd_live_share, vintage_capital_options,
)
from tests.unit.test_fes2050_decision41 import INSURANCE, _assets, _capture, _run


def _basis(tmp_path, **extra):
    block = {vk.CONFIG_KEY: {"amortisation": vk.CONTRACT_TERM_WITH_TAIL, "capture_prices": str(_capture(tmp_path)),
                             **extra},
             "generator_costs": INSURANCE}
    return vintage_capital_options(block)


# ── decision 42 ──────────────────────────────────────────────────────────────

def test_the_legacy_rule_is_off_without_its_key(tmp_path):
    assert _basis(tmp_path)["legacy_expired"] is None
    rule = vk.legacy_expired_rule({"labelled": True, "technologies": {"Hydropower": [2002, 2017]}})
    assert rule == {"labelled": True, "technologies": {"Hydropower": (2002.0, 2017.0)}}


def test_legacy_rows_whose_contract_ended_are_sunk(tmp_path):
    """The labelled legacy row (ON_2020) loses its capital; new build and Sizewell C are as decision 41 left them."""
    before = _run(_assets(), vintage_capital=_basis(tmp_path))
    after = _run(_assets(), vintage_capital=_basis(tmp_path, legacy_expired_contracts={"labelled": True}))
    for case in ("private", "public_ownership"):
        a = after.loc[after["case_id"].eq(case)].set_index("asset_id")
        b = before.loc[before["case_id"].eq(case)].set_index("asset_id")
        assert b.loc["ON_2020", "annual_capital_recovery_gbp"] > 0.0
        assert a.loc["ON_2020", "annual_capital_recovery_gbp"] == 0.0
        assert a.loc["ON_2020", "annual_required_revenue_gbp"] == pytest.approx(
            b.loc["ON_2020", "annual_required_revenue_gbp"] - b.loc["ON_2020", "annual_capital_recovery_gbp"])
        for asset in ("ON_2040", "ON_2031", "ON_2028", "SZC_2039"):
            assert a.loc[asset, "annual_capital_recovery_gbp"] == pytest.approx(b.loc[asset, "annual_capital_recovery_gbp"])


def test_legacy_rows_by_technology_and_build_year():
    frame = pd.DataFrame({"finance_vintage": ["legacy", "legacy", "legacy", "new_build", "legacy"],
                          "support_basis": [None, None, "contract_expired_merchant", "contract_expired_merchant", None],
                          "stock_flow_technology": ["Hydropower", "Hydropower", "Solar PV", "Solar PV",
                                                    "Pumped hydro energy storage (PHES)"],
                          "build_year": [1950, 2010, 2015, 2028, 1983]})
    rule = vk.legacy_expired_rule({"labelled": True, "technologies": {"Hydropower": [2002, 2017]}})
    assert vk.legacy_expired_rows(frame, rule).tolist() == [False, True, True, False, False]
    assert not vk.legacy_expired_rows(frame, None).any()


# ── decision 44 ──────────────────────────────────────────────────────────────

def test_the_carbon_cfd_live_share_is_its_own_term():
    merged = pd.DataFrame({"asset_id": ["B30", "B35", "B40", "W40"], "carrier": ["CCS Biomass"] * 3 + ["wind_onshore"],
                           "build_year": [2030, 2035, 2040, 2040], "model_year": 2050})
    share = carbon_cfd_live_share(merged, ["CCS Biomass"], 15)
    assert share.to_dict() == {"B30": 0.0, "B35": 0.5, "B40": 1.0}
    assert carbon_cfd_live_share(merged, ["CCS Biomass"], None) is None
    assert carbon_cfd_live_share(merged, ["CCS Biomass"], 25)["B30"] == pytest.approx(1.0)


def test_an_ended_carbon_cfd_pays_nothing():
    register = pd.DataFrame({
        "case_id": ["private", "private"], "asset_id": ["B30", "B40"], "carrier": "CCS Biomass",
        "component_type": "Generator", "annual_required_revenue_gbp": [100.0, 100.0],
        "producer_rent_gbp": [-50.0, -50.0], "public_surplus_before_recycling_gbp": [0.0, 0.0],
        "ownership_share": [0.0, 0.0], "market_revenue_gbp": [0.0, 0.0], "support_revenue_gbp": [0.0, 0.0],
        "fuel_carbon_gbp": [0.0, 0.0]})
    live = pd.Series({"B30": 0.0, "B40": 1.0})
    full = book_carbon_removal_cfd(register, ["CCS Biomass"])
    termed = book_carbon_removal_cfd(register, ["CCS Biomass"], live_share=live)
    assert full["carbon_removal_cfd_revenue_gbp"].tolist() == [100.0, 100.0]
    assert termed["carbon_removal_cfd_revenue_gbp"].tolist() == [0.0, 100.0]
    assert termed["producer_rent_gbp"].tolist() == [-50.0, 50.0]


# ── decision 32 ──────────────────────────────────────────────────────────────

def _run_with_p1(assets, vintage_capital):
    import pandas as pd

    from scripts.public_ownership.calculate_financing_scenarios import (calculate_financing_scenarios,
                                                                        vintage_cost_registry)
    from scripts.public_ownership.levers import expand_cases, parse_scenarios, parse_sensitivities
    from tests.unit.test_fes2050_decision41 import NO_OVERHEAD, SCENARIOS, _snapshot

    scenarios = dict(SCENARIOS, public_control={"new_build_generation": "state_stake"})
    bundles = parse_scenarios(scenarios)
    specs = expand_cases(bundles, parse_sensitivities({"bounding_pair": False}, bundles))
    costs = pd.read_csv("data/public_ownership/financing/technology_costs.csv")
    register, _ = calculate_financing_scenarios(
        assets, costs, _snapshot(), price_year=2024, state_operating_overhead=NO_OVERHEAD, generator_costs=INSURANCE,
        case_specs=specs, vintage_costs=vintage_cost_registry(), vintage_capital=vintage_capital)
    return register


def test_sizewell_c_leaves_p1_and_p2_only_when_the_switch_is_on(tmp_path):
    """Decision 32 (user, 3 Oct 2026): Sizewell C's RAB is outside P1's stake and P2's boundary."""
    off = _run_with_p1(_assets(), _basis(tmp_path))
    on = _run_with_p1(_assets(), _basis(tmp_path, rab_outside_propositions=True))
    share = lambda r, case, asset: float(r.loc[r["case_id"].eq(case)].set_index("asset_id").loc[asset, "ownership_share"])  # noqa: E731
    assert share(off, "public_ownership", "SZC_2039") == 1.0
    assert share(off, "public_control", "SZC_2039") > 0.0
    assert share(on, "public_ownership", "SZC_2039") == 0.0
    assert share(on, "public_control", "SZC_2039") == 0.0
    # Nothing else moves.
    for case in ("public_ownership", "public_control"):
        for asset in ("ON_2040", "ON_2031", "ON_2028"):
            assert share(on, case, asset) == share(off, case, asset)
    assert share(on, "private", "SZC_2039") == 0.0


def test_the_2050_ownership_check_exempts_only_rab_rows():
    """Decision 32: the check accepts Sizewell C's rows leaving the state, and nothing else."""
    from scripts.public_ownership import verify_ownership_tables_2050 as v

    def table(state):
        return pd.DataFrame({"case_id": "p2", "generator": ["NUC", "WIND"], "capacity_mw": [100.0, 50.0],
                             "state_capacity_mw": state, "state_share": [s / c for s, c in zip(state, [100.0, 50.0])],
                             "state_commissioning_year": [2039.0, 2040.0]})

    exempt = pd.DataFrame({"case_id": ["p2"], "generator": ["NUC"], "exempt_mw": [40.0]})
    saved = table([100.0, 50.0])
    assert v.compare_with_exemption(table([60.0, 50.0]), saved, exempt).get("failed_columns") is None
    # More than the RAB rows left the state on that generator: refused.
    assert "exempt_bound" in v.compare_with_exemption(table([50.0, 50.0]), saved, exempt)["failed_columns"]
    # Any change on a generator without RAB rows: refused, as before.
    assert v.compare_with_exemption(table([60.0, 40.0]), saved, exempt)["failed_columns"]
    with pytest.raises(SystemExit):
        v.main(["r", "p", "s", "--exempt-support-types", "CfD"])


def test_the_whole_legacy_fleet_can_be_sunk():
    rule = vk.legacy_expired_rule({"labelled": True}, {"technologies": ["Hydropower", "Energy from waste"]})
    frame = pd.DataFrame({"finance_vintage": ["legacy", "legacy", "new_build"], "support_basis": [None, None, None],
                          "stock_flow_technology": ["Hydropower", "Energy from waste", "Hydropower"],
                          "build_year": [1927, 2015, 2040]})
    assert vk.legacy_expired_rows(frame, rule).tolist() == [True, True, False]
    assert vk.legacy_expired_rule(None, None) is None


def test_expired_beccs_sells_its_removals_at_market_value():
    from scripts.public_ownership.calculate_financing_scenarios import merchant_removal_value

    assert vk.beccs_removal_value(None) is None
    price = vk.beccs_removal_value("ets_value")
    assert price == pytest.approx(114.82, abs=0.01)          # 1.036 tCO2/MWh x 110.80 GBP/tCO2
    with pytest.raises(vk.VintageCapitalError):
        vk.beccs_removal_value("strike")
    merged = pd.DataFrame({"asset_id": ["B30", "B35"], "dispatch_mwh": [1000.0, 1000.0]})
    value = merchant_removal_value(merged, pd.Series({"B30": 0.0, "B35": 0.5}), price)
    assert value["B30"] == pytest.approx(1000.0 * price)
    assert value["B35"] == pytest.approx(500.0 * price)
    register = pd.DataFrame({
        "case_id": ["private"], "asset_id": ["B30"], "carrier": "CCS Biomass", "component_type": "Generator",
        "annual_required_revenue_gbp": [100.0], "producer_rent_gbp": [-50.0],
        "public_surplus_before_recycling_gbp": [0.0], "ownership_share": [0.0], "market_revenue_gbp": [0.0],
        "support_revenue_gbp": [0.0], "fuel_carbon_gbp": [0.0]})
    out = book_carbon_removal_cfd(register, ["CCS Biomass"], live_share=pd.Series({"B30": 0.0}),
                                  merchant_removals=pd.Series({"B30": 30.0}))
    assert out["carbon_removal_cfd_revenue_gbp"].tolist() == [30.0]
    assert out["producer_rent_gbp"].tolist() == [-20.0]


# ── decision 45 ──────────────────────────────────────────────────────────────

def test_the_declared_output_strike_recovers_required_revenue_on_declared_output():
    from scripts.market.cfd_strikes_2050 import strike_inputs_for

    rows = pd.DataFrame({"capacity_mw": [100.0], "cfd_live_share": [1.0], "dispatch_mwh": [100.0 * 8760 * 0.19],
                         "annual_required_revenue_gbp": [20e6], "annual_variable_om_gbp": [1e6], "fuel_carbon_gbp": [0.0]})
    out = vk.declared_output_strike(rows, "onshore", 2040)
    lf = float(strike_inputs_for("Onshore wind", 2040)["expected_load_factor"])
    declared = 100.0 * 8760 * lf
    assert out["declared_load_factor"] == lf
    assert out["delivered_load_factor"] == pytest.approx(0.19)
    assert out["declared_output_strike"] == pytest.approx(19e6 / declared + 1e6 / (100.0 * 8760 * 0.19))
    assert np.isnan(vk.declared_output_strike(rows, "beccs", 2040)["declared_output_strike"])


# ── decision 30 ──────────────────────────────────────────────────────────────

def test_the_capacity_market_memo_is_2050_only():
    import json
    from pathlib import Path

    from scripts.public_ownership.exchequer_ledger import consumer_flexibility_memo

    root = Path(__file__).resolve().parents[2]
    a2030 = json.loads((root / "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json").read_text(encoding="utf-8"))
    a2050 = json.loads((root / "data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json").read_text(encoding="utf-8"))
    assert consumer_flexibility_memo(a2030) is None
    memo = consumer_flexibility_memo(a2050)
    assert memo["consumers_pay_gbp"] == memo["consumers_receive_gbp"] > 1e9
    assert set(memo["change_vs_private_gbp"].values()) == {0.0}
    assert sum(v for k, v in memo["payments_by_payee_gbp"].items() if k.endswith(
        ("households", "residential", "commercial", "interconnectors"))) == pytest.approx(memo["agreement_cost_gbp"], abs=1.0)


# ── decisions 43 and 48: the FES25_HT_2050 entry ─────────────────────────────────

def test_the_study_entry_carries_decisions_42_to_48():
    from scripts.public_ownership.study_config import load_studies

    block = load_studies("config").configs["FES25_HT_2050"]["public_ownership"]
    assert set(block["consumer_bill"]["ldes_cap_and_floor"]["carriers"]) == {
        "LAES", "CAES", "Pumped Storage Hydroelectricity"}
    lives = block["acquisition"]["valuation"]["retirement_life_years_by_carrier"]
    assert (lives["solar_pv"], lives["wind_offshore"], lives["wind_onshore"]) == (38, 30, 25)
    vc = block[vk.CONFIG_KEY]
    assert vc["beccs_carbon_cfd_term_years"] == 15 and vc["legacy_expired_contracts"]["labelled"]
    assert vc["rab_outside_propositions"] is True and vc["beccs_removals_after_cfd"] == "ets_value"
    assert set(vc["legacy_sunk_capital"]["technologies"]) == {
        "Pumped hydro energy storage (PHES)", "Hydropower", "Energy from waste"}


# ── batch of 3 Oct 2026 (decision 49 and two answers) ─────────────────────────

def test_the_y12_network_drivers_are_the_network_account():
    """Item 6: the bill's roll-forward is the fes_ht_2050 network account's (skipped when absent; never pinned)."""
    import json
    from pathlib import Path

    from scripts.consumer_bill.build_2050_bill_inputs import network_2050, y12_driver

    path = Path(__file__).resolve().parents[2] / "resources/public_ownership/fes_ht_2050_network_finance_account.json"
    if not path.exists():
        pytest.skip("2050 network account not built")
    roll = json.loads(path.read_text(encoding="utf-8"))["roll_forward"]
    Y12_NETWORK = network_2050()
    t = {r["year"]: r for r in roll["transmission_path"]}
    d = {r["year"]: r for r in roll["distribution_path"]}
    for y in (2030, 2049, 2050):
        assert Y12_NETWORK["transmission"]["rav_gbp"][y] == pytest.approx(t[y]["closing_rav_gbp"], rel=1e-12)
        assert Y12_NETWORK["transmission"]["depreciation_gbp"][y] == pytest.approx(t[y]["depreciation_gbp"], rel=1e-12)
        assert Y12_NETWORK["distribution"]["rav_gbp"][y] == pytest.approx(d[y]["closing_rav_gbp"], rel=1e-12)
    assert y12_driver("transmission", 2030) == 1.0 and y12_driver("distribution", 2030) == 1.0
    assert 1.5 < y12_driver("transmission", 2050) < 2.0


def test_interconnectors_leave_cap_and_floor_with_their_own_regime():
    """Item 5: a cable whose 25-year regime has ended by 2050 is merchant; a live one passes through."""
    import json
    from pathlib import Path

    a = json.loads((Path(__file__).resolve().parents[2]
                    / "data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json").read_text(encoding="utf-8"))
    cables = a["model_linked_rules"]["interconnector_cap_and_floor"]["cables"]
    for name, spec in cables.items():
        if "regime_end_year" in spec:
            assert spec["regime_end_year"] == int(spec["operational"]) + 25
            assert spec["regime"] == ("exempt_merchant" if spec["regime_end_year"] <= 2050 else "cap_and_floor"), name
    assert cables["NeuConnect Interconnector"]["regime"] == "cap_and_floor"
    assert cables["Nemo Link"]["regime"] == "exempt_merchant"
    assert cables["IFA"]["regime"] == "use_of_revenues_share"
    assert "hpc_strike_reduction" not in a["model_linked_rules"]["cfd"]          # the 2030 session's patch


def test_the_forecast_error_charge_scales_to_the_group_fleet_ratio():
    """Item 7 (decision 18): the settlement states its ratio; a group's own ratio rescales the charge."""
    from scripts.public_ownership.nationalised_institutions import settlement_fleet_ratio

    assert settlement_fleet_ratio("renewable_capacity, NPV reading, fleet ratio 1.1, 2024 prices") == 1.1
    with pytest.raises(ValueError):
        settlement_fleet_ratio("renewable_capacity")


def test_a_2050_register_refuses_a_config_without_its_group():
    """Item 9: a vintaged register handed the base defaults fails loudly; 2030 is never checked."""
    from scripts.public_ownership.calculate_financing_scenarios import require_study_config

    with pytest.raises(ValueError, match="study's resolved config"):
        require_study_config({"reference_run_groups": {"cp30_2030_cp30_network": {}}}, "fes_ht_2050", 2050)
    require_study_config({"reference_run_groups": {"fes_ht_2050": {}}}, "fes_ht_2050", 2050)
    require_study_config({"reference_run_groups": {}}, "cp30_2030_cp30_network", 2030)


def test_domestic_batteries_and_held_carriers():
    """Item 4: domestic batteries leave the vintaged register; item 3: geothermal is costed but held private."""
    from scripts.public_ownership.build_asset_finance_register import non_ownership_carriers
    from scripts.public_ownership.vintage_costs import costs_for

    assert "Domestic Battery" in non_ownership_carriers()
    cost = costs_for("Deep geothermal", 2040)
    assert float(cost["capex_power_gbp_per_kw"]) > 0 and float(cost["economic_life_years"]) == 30.0
    assert vk.basis_from_config({vk.CONFIG_KEY: {"held_private_carriers": ["geothermal"]}})["held_private_carriers"] == ["geothermal"]


def test_a_private_row_that_left_the_register_is_the_same_bid():
    """Item 4 and the check: a saved zero-state row with no register row is dropped; a state-owned one never."""
    from scripts.public_ownership import verify_ownership_tables_2050 as v

    expected = pd.DataFrame({"case_id": ["p2"], "generator": ["WIND"], "capacity_mw": [50.0], "state_capacity_mw": [50.0],
                             "state_share": [1.0], "state_commissioning_year": [2040.0]})
    saved = pd.concat([expected, pd.DataFrame({"case_id": ["p2", "p2"], "generator": ["DOMBAT", "OWNED"],
                                               "capacity_mw": [5.0, 5.0], "state_capacity_mw": [0.0, 5.0],
                                               "state_share": [0.0, 1.0], "state_commissioning_year": [np.nan, 2040.0]})])
    kept, dropped = v.drop_private_rows_left_the_register(expected, saved)
    assert dropped == 1 and kept["generator"].tolist() == ["WIND", "OWNED"]


def test_smr_and_floating_offshore_carry_no_d4a_insurance_in_2050():
    """Item 2: their fixed O&M already holds insurance (Annex A floating; FES E.03 SMR)."""
    from scripts.market.cfd_strikes_2050 import finance_insurance_gbp_per_kw_year
    from scripts.public_ownership.study_config import load_studies

    block = load_studies("config").configs["FES25_HT_2050"]["public_ownership"]["generator_costs"]["insurance"]
    assert finance_insurance_gbp_per_kw_year("Small modular reactors (SMRs)", 8000.0, block) == 0.0
    assert finance_insurance_gbp_per_kw_year("Floating offshore wind", 3000.0, block) == 0.0
    assert finance_insurance_gbp_per_kw_year("Onshore wind", 1500.0, block) == pytest.approx(6.0)
