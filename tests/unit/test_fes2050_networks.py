"""FES 2050 plan, task Y12: the networks in 2050 are a declared RAV roll-forward.

No official RAV or TNUoS projection exists after 2031, so every 2050 figure must come
from declared inputs with provenance, be converted to one price basis before it is
added, close under both network treatments, and leave the CP30 year untouched.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership import network_finance_account as nfa
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.network_rav_roll_forward import (
    NO_PROJECTION_STATEMENT, capex_between, distribution_path, load_inputs, peak_gw, rav_at, rebase_factor,
    roll_forward, transmission_path,
)
from config.config_loader import load_defaults

INPUTS_PATH = "data/market/network_cost_sources/rav_roll_forward_2050.yaml"
TABLE_2050 = "data/market/network_cost_sources/network_guarantee_pass_through_2050.csv"
CONV = {"factor": 1.038374717833, "final_demand_mwh": 616.5e6, "mwh": 3.1}
PUBLIC_RATE = 0.027450980392157
# The real extracted building blocks (resources/analysis/network_price_control_building_blocks.json).
BLOCKS = {
    "distribution": {
        "price_basis_key": "2020/21", "closing_rav_gbp_m_2027_28": 69027.0,
        "net_additions_gbp_m_2027_28": 6076.7, "depreciation_gbp_m_2027_28": 4000.4,
        "tax_allowance_gbp_m_period_average": 735.5, "vanilla_wacc_cpih_real": 0.0393,
        "opening_rav_gbp_m_period_average": 62271.4,
    },
    "transmission": {
        "price_basis_key": "2023/24", "total_rav_gbp_m": 54210.222222222226,
        "wacc_semi_nominal": 0.0564, "notional_gearing": 0.55,
    },
}
YEAR_2050 = {
    "distribution_roll_forward": "peak_demand", "transmission_roll_forward": "declared",
    "roll_forward_inputs": INPUTS_PATH, "acquisition_year": 2030, "guarantee_pass_through_table": TABLE_2050,
    # As config/studies/fes2050.yaml: F27 item 7 (23 Oct 2026) is a 2030 change.
    "distribution_tax_allowance_basis": "period_average", "excess_return": None,
}


@pytest.fixture(scope="module")
def inputs():
    return load_inputs(INPUTS_PATH)


@pytest.fixture(scope="module")
def anchors():
    return {"transmission_anchor_rav_gbp": nfa.transmission_anchor_rav(BLOCKS),
            "distribution_anchor_rav_gbp": nfa.ed2_anchor_rav(BLOCKS, nfa.settings({})),
            "distribution_anchor_depreciation_gbp": 4000.4e6 * nfa.price_factor("2020/21")}


def _base_block():
    block = load_defaults()["public_ownership"]["network_account"]
    return dict(block)


def _cfg_2050(**extra):
    block = _base_block()
    block["model_years"] = {2050: {**YEAR_2050, **extra}}
    return nfa.settings(block, 2050)


class TestInputs:
    def test_every_input_carries_its_source_and_status(self, inputs, tmp_path):
        assert "No official RAV or TNUoS projection exists after 2031" in inputs["statement"]
        broken = yaml.safe_load(Path(INPUTS_PATH).read_text(encoding="utf-8"))
        del broken["transmission"]["capitalisation_rate"]["source"]
        path = tmp_path / "broken.yaml"
        path.write_text(yaml.safe_dump(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="capitalisation_rate needs a source"):
            load_inputs(path)

    def test_prices_are_rebased_before_they_are_added(self, inputs):
        # NESO's 2026 figure is taken as 2025 prices (declared): ONS CPIH 132.9 / 138.0.
        assert rebase_factor("2025", inputs) == pytest.approx(132.9 / 138.0)
        # DESNZ's framework is 2020 prices.
        assert rebase_factor("2020", inputs) == pytest.approx(132.9 / 108.9)
        # Financial-year bases are the 2030 account's own factors (RIIO-3 2023/24).
        assert rebase_factor("2023/24", inputs) == pytest.approx(nfa.price_factor("2023/24"))
        with pytest.raises(ValueError, match="no CPIH calendar average"):
            rebase_factor("2019", inputs)


class TestTransmission:
    def test_programmes_enter_at_the_capitalisation_rate_and_the_network_is_held_after_2039(self, inputs, anchors):
        path, programme = transmission_path(anchors["transmission_anchor_rav_gbp"], inputs, 2050)
        f3, fb = nfa.price_factor("2023/24"), 132.9 / 138.0
        by_year = path.set_index("year")
        # The anchor lacks the uncommitted 26 bn's first three years.
        assert by_year.at[2028, "additions_gbp"] == pytest.approx(0.85 * 26e9 * f3 * 3 / 5)
        assert by_year.at[2029, "capex_gbp"] == pytest.approx(70e9 * f3 / 5)
        assert by_year.at[2035, "capex_gbp"] == pytest.approx(89e9 * fb / 9)
        assert by_year.at[2035, "additions_gbp"] == pytest.approx(0.85 * 89e9 * fb / 9)
        # Capex equals depreciation from 2040 (decision 10): the RAV is flat.
        after = path.loc[path["year"].ge(2040)]
        assert (after["additions_gbp"] - after["depreciation_gbp"]).abs().max() < 1e-3
        assert rav_at(path, 2050) == pytest.approx(rav_at(path, 2039))
        # Each year closes: opening + additions - depreciation.
        assert ((path["opening_rav_gbp"] + path["additions_gbp"] - path["depreciation_gbp"]
                 - path["closing_rav_gbp"]).abs().max() < 1e-3)
        assert programme["beyond_2030_price_basis"] == "2025"

    def test_the_2050_rav_and_the_workbook_cross_check(self, inputs, anchors):
        result = roll_forward(anchors, inputs, 2050, fy_price_factor=nfa.price_factor)
        assert result["transmission_rav_2050_gbp"] / 1e9 == pytest.approx(133.88, abs=0.01)
        assert result["distribution_rav_2050_gbp"] / 1e9 == pytest.approx(174.72, abs=0.01)
        check = result["cross_checks"]["workbook_network_capex"]
        assert check["workbook_gbp_bn"] == 160.0
        # The roll-forward's 2031-2050 transmission capex is 144 bn, 0.90 of the workbook's 160 bn.
        assert check["roll_forward_transmission_capex_2031_to_end_gbp_bn"] == pytest.approx(144.30, abs=0.01)
        assert 0.8 < check["ratio_2031_to_end"] < 1.0
        # The low sensitivity (committed RIIO-3, the 2024 Beyond 2030 plan) sits below the central.
        assert result["transmission_sensitivities"]["low"]["transmission_rav_2050_gbp"] < result["transmission_rav_2050_gbp"]
        assert result["statement"] == NO_PROJECTION_STATEMENT


class TestDistribution:
    def test_the_rav_follows_fes_peak_demand_at_the_declared_elasticity(self, inputs, anchors):
        path = distribution_path(anchors["distribution_anchor_rav_gbp"], anchors["distribution_anchor_depreciation_gbp"],
                                 inputs, 2050)
        assert rav_at(path, 2050) == pytest.approx(anchors["distribution_anchor_rav_gbp"] * 120.05 / 62.09)
        assert peak_gw(2040, inputs) == pytest.approx(62.09 * (120.05 / 62.09) ** 0.5)
        # Gross additions are backed out: never negative, and each year closes.
        assert path["additions_gbp"].min() >= 0.0
        assert ((path["opening_rav_gbp"] + path["additions_gbp"] - path["depreciation_gbp"]
                 - path["closing_rav_gbp"]).abs().max() < 1e-3)
        assert capex_between(path, 2031, 2050) > 0


class TestAccount2050:
    def test_outside_riio3_without_a_roll_forward_it_still_refuses(self):
        cfg = nfa.settings({"distribution_roll_forward": "explicit", "distribution_rav_2050_gbp": 1.0e11}, 2050)
        with pytest.raises(ValueError, match="needs a declared roll-forward"):
            nfa.build_account(BLOCKS, PUBLIC_RATE, cfg, CONV, model_year=2050)
        with pytest.raises(ValueError, match="needs roll_forward_inputs"):
            nfa.settings({"model_years": {2050: {"distribution_roll_forward": "peak_demand"}}}, 2050)

    @pytest.fixture(scope="class")
    def account(self):
        cfg = _cfg_2050()
        # F29: the cash rate at each of workbook v2.0's lending margins; the debt guarantee is retired.
        margins = {"pwlb_hra": 0.0040, "pwlb_certainty": 0.0080, "pwlb_standard": 0.0100, "gilt_flat": 0.0}
        snapshot = {"assumptions": {"public_borrowing_real": {"value": PUBLIC_RATE}},
                    "public_rate_by_lending_margin": {k: {"value": PUBLIC_RATE - 0.0011 + m} for k, m in margins.items()}}
        frame, summary = nfa.build_all(
            BLOCKS, snapshot, cfg, CONV, premiums=nfa.load_premiums(cfg), pass_through=nfa.load_pass_through(cfg, 2050),
            debt_compression=None, model_year=2050)
        return cfg, frame, summary

    def test_it_closes_under_both_treatments(self, account):
        _, frame, summary = account
        closed = nfa.check_account_closes(frame, summary)
        assert closed["rows_checked"] == len(frame) and closed["largest_residual_gbp"] < 1.0
        assert set(frame["treatment"]) == {nfa.TREATMENT_ACQUIRED}
        assert summary["transmission_rav_2050_gbp"] / 1e9 == pytest.approx(133.88, abs=0.01)
        assert summary["distribution_rav_2050_gbp"] / 1e9 == pytest.approx(174.72, abs=0.01)
        assert "declared roll-forward" in summary["transmission"]["rav_basis"]
        assert "No official RAV or TNUoS projection exists after 2031" in " ".join(summary["notes"])

    def test_the_premium_was_paid_on_the_2030_rav_and_runs_to_2059(self, account):
        cfg, _, summary = account
        t = summary["transmission"]
        crf = capital_recovery_factor(PUBLIC_RATE, 30.0)
        assert t["premium_rav_gbp"] == pytest.approx(nfa.transmission_anchor_rav(BLOCKS))
        assert t["premium_annual_cost_gbp"] == pytest.approx(0.5 * t["premium_rav_gbp"] * crf)
        assert t["financing_saving_gbp"] == pytest.approx(
            t["rav_gbp"] * (t["allowed_wacc_cpih_real"] - PUBLIC_RATE) - t["premium_annual_cost_gbp"])
        later = nfa.build_account(BLOCKS, PUBLIC_RATE, _cfg_2060(), CONV, premium=0.5, model_year=2060)[1]
        assert later["transmission"]["premium_payable"] is False
        assert later["transmission"]["premium_annual_cost_gbp"] == 0.0

    def test_the_distribution_tax_allowance_scales_with_the_rav(self, account):
        _, _, summary = account
        d = summary["distribution"]
        expected = 735.5e6 * nfa.price_factor("2020/21") * d["rav_gbp"] / (62271.4e6 * nfa.price_factor("2020/21"))
        assert d["tax_allowance_saving_gbp"] == pytest.approx(expected)
        assert "scaled by the rolled-forward RAV" in d["tax_basis"]


def _cfg_2060():
    block = _base_block()
    block["model_years"] = {2060: dict(YEAR_2050)}
    return nfa.settings(block, 2060)


class TestGuaranteeWindow:
    def test_by_2050_every_trailing_year_is_inside_the_window(self):
        derived = nfa.derived_pass_through(2050)
        table = nfa.load_pass_through({"guarantee_pass_through_table": TABLE_2050})
        assert derived == table
        assert all(v == 1.0 for shares in table.values() for v in shares.values())
        assert nfa.issuance_window_share(2039, trailing_years=14) == 1.0
        assert nfa.issuance_window_share(2038, trailing_years=14) is None

    def test_2030_straddles_the_window_and_needs_the_cost_of_debt_table(self):
        with pytest.raises(ValueError, match="straddles the 2026 window start"):
            nfa.derived_pass_through(2030)
        assert nfa.load_pass_through({"guarantee_window": {}}, 2050)["new_and_refinanced"]["transmission"] == 1.0


class TestCp30YearUnchanged:
    def test_a_model_years_block_does_not_touch_the_cp30_settings(self):
        base = _base_block()
        with_years = {**base, "model_years": {2050: dict(YEAR_2050)}}
        assert nfa.settings(with_years, 2030) == nfa.settings(base, 2030)
        assert not set(nfa.settings(with_years, 2030)) & set(nfa.YEAR_KEYS)

    def test_the_cp30_account_has_no_roll_forward_block(self):
        cfg = nfa.settings(_base_block(), 2030)
        _, summary = nfa.build_account(BLOCKS, PUBLIC_RATE, cfg, CONV, model_year=2030)
        assert "roll_forward" not in summary and "premium_rav_gbp" not in summary["transmission"]
        assert "Transmission needs no roll-forward: RIIO-3 runs to 31 March 2031." in summary["notes"]
