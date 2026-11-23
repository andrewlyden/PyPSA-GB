"""OW10 (was G4): Sizewell C's RAB levy with its return at the public rate, in the public cases only.

The levy's return element is scaled by public rate / RAB rate on the state's
share of its own new build; private and comparator cases never move; a register
that books the RAB itself (2050) is never adjusted.  The bill line, the tariff's
two ledgers, the three-party identity and the itemised flows all close with it.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership import investor_ledger as inv
from scripts.public_ownership import party_flows
from scripts.public_ownership import sizewell_c_rab as szc
from scripts.public_ownership.nationalised_tariff import VALUE, build_bill, reconcile, revenue_requirement, system
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    LINE_SIZEWELL, NEW_LINES, sizewell_row, sizewell_terms,
)
from config.config_loader import load_defaults

M = 1.0e6
ON = szc.settings_from_config({szc.CONFIG_KEY: {"enabled": True}})
#: F19 decision 1: the share of Sizewell C's capital the state already funds (NAO HC 33).
EXISTING = (0.449 * 8.5 + 36.6) / (8.5 + 41.6)
#: A rate step for the arithmetic tests (the case rate itself carries no premium since F29).
H = 0.0125
#: F19 decision 1, refined: equity at the NAO's 12.5% nominal IRR (CPIH 2%), debt at the WACC-consistent rate.
R_E = 1.125 / 1.02 - 1.0
R_D = (50.1 * 0.0673 - 8.5 * R_E) / 41.6
E_P, D_P = 0.551 * 8.5, 5.0


def _saving(rate, fraction=1.0):
    """The share of the whole return saved: f x (R_private - K_private x rate) / (K x WACC)."""
    return fraction * (E_P * R_E + D_P * R_D - (E_P + D_P) * rate) / (50.1 * 0.0673)
NETWORK = pd.DataFrame({
    "treatment": ["acquired", "acquired", "acquired", "acquired", "private"],
    "sector": ["transmission", "distribution", "transmission", "distribution", "transmission"],
    "public_rate_basis": ["real_cpi", "real_cpi", "pwlb_standard", "pwlb_standard", None],
    "public_real_rate": [0.02745, 0.02745, 0.03599, 0.03599, None],
})


class TestSettingsAndArithmetic:
    def test_off_by_default_and_on_in_the_repository_config(self):
        assert szc.settings_from_config({}) == {**szc.DEFAULT_SETTINGS, "enabled": False}
        bill = load_defaults()["public_ownership"]["consumer_bill"]
        settings = szc.settings_from_config(bill)
        assert settings["enabled"] is True
        # The code defaults agree with the config on every value (house rule).
        assert {k: v for k, v in settings.items() if k != "enabled"} == {
            k: v for k, v in szc.DEFAULT_SETTINGS.items() if k != "enabled"}
        assert szc.existing_state_capital_share(settings) == pytest.approx(EXISTING)
        assert EXISTING == pytest.approx(0.8067, abs=1e-4)

    def test_bad_settings_are_refused(self):
        for block, message in (({"rate": 0.05}, "unknown keys"), ({"rab_real_rate": 6.73}, "rab_real_rate"),
                               ({"return_share": 1.5}, "return_share"), ({"enabled": "yes"}, "enabled"),
                               ({"state_equity_share": 1.2}, "state_equity_share"),
                               ({"nwf_debt_gbp_bn": 50.0}, "nwf_debt_gbp_bn"),
                               ({"construction_premium": {"graded": 0.01}}, "construction_premium")):
            with pytest.raises(ValueError, match=message):
                szc.settings_from_config({szc.CONFIG_KEY: block})

    def test_the_rates_are_consistent_with_the_wacc(self):
        r_e, r_d = szc.capital_rates(ON)
        assert r_e == pytest.approx(0.10294, abs=1e-5) and r_d == pytest.approx(0.06002, abs=1e-5)
        assert (8.5 * r_e + 41.6 * r_d) / 50.1 == pytest.approx(0.0673)
        assert szc.private_capital(ON) == (pytest.approx(E_P), pytest.approx(D_P))
        # An equity return the WACC cannot carry is refused.
        with pytest.raises(ValueError, match="inconsistent"):
            szc.settings_from_config({szc.CONFIG_KEY: {"equity_return_nominal": 0.60}})

    def test_only_the_return_on_the_privately_financed_capital_moves(self):
        # F19 decision 1 (refined): all of it under P2 takes on the 19.3% the state does not already fund; what moves
        # is that capital's return (equity at 10.29%, debt at 6.00%) less its cost at the given rate.
        rate = 0.02745 + H
        full = szc.household_adjustment(14.37, ON, state_share=1.0, rate=rate)
        assert full == pytest.approx(-14.37 * _saving(rate))
        assert _saving(rate) == pytest.approx(0.1173, abs=1e-4)
        # Larger than the capital-share rule it replaces (19.3% at the blended 6.73%).
        assert -full > 14.37 * (1 - EXISTING) * (1 - rate / 0.0673)
        # A 30% stake (P1) is below the existing share of capital: nothing moves.
        assert szc.household_adjustment(14.37, ON, state_share=0.3, rate=rate) == 0.0
        assert szc.refinanced_share(ON, 0.3) == 0.0
        assert szc.refinanced_share(ON, 0.9) == pytest.approx(0.9 - EXISTING)
        # Part of the private capital: pro rata across its equity and debt.
        assert szc.household_adjustment(14.37, ON, state_share=0.9, rate=rate) == pytest.approx(
            -14.37 * _saving(rate, (0.9 - EXISTING) / (1 - EXISTING)))
        assert szc.household_adjustment(14.37, dict(ON, return_share=0.5), state_share=1.0,
                                        rate=rate) == pytest.approx(0.5 * full)
        # With no existing state finance the whole return moves: all of it at the WACC, less the state's cost.
        bare = dict(ON, state_equity_share=0.0, nwf_debt_gbp_bn=0.0)
        assert szc.household_adjustment(14.37, bare, state_share=1.0, rate=0.02745) == pytest.approx(
            -14.37 * (1 - 0.02745 / 0.0673))
        assert szc.household_adjustment(14.37, ON, state_share=0.0, rate=rate) == 0.0
        assert szc.household_adjustment(14.37, dict(ON, enabled=False), state_share=1.0, rate=rate) == 0.0

    def test_the_state_pays_its_cash_rate_with_no_construction_premium(self):
        # F29: workbook v2.0 charges the state its cash rate; the construction premium is retired.
        assert szc.premium(ON, {}) == 0.0
        assert szc.premium(ON, {"public_rate_basis": "pwlb_standard"}) == 0.0

    def test_the_public_rate_follows_the_case_basis_on_the_network_account(self):
        assert szc.public_rate(NETWORK, {"public_rate_basis": "real_cpi"}) == pytest.approx(0.02745)
        assert szc.public_rate(NETWORK, {"public_rate_basis": "pwlb_standard"}) == pytest.approx(0.03599)
        assert szc.public_rate(None, {}, fallback=0.03) == 0.03
        with pytest.raises(ValueError, match="network finance account"):
            szc.public_rate(None, {"public_rate_basis": "gilt_flat"})

    def test_the_state_share_is_its_share_of_its_own_new_build(self):
        rows = pd.DataFrame({"finance_vintage": ["legacy", "new_build", "new_build"],
                             "ownership_share": [1.0, 0.3, 0.0]})
        assert szc.state_new_build_share(rows) == pytest.approx(0.3)
        assert szc.state_new_build_share(rows.iloc[:1]) == 0.0
        assert szc.state_new_build_share(pd.DataFrame()) == 0.0


class TestMarketBill:
    def test_only_a_public_case_without_a_booked_rab_moves(self):
        public = {"state_new_build_share": 1.0, "levers": {"public_rate_basis": "real_cpi"}}
        assert sizewell_terms(public, settings=ON, network_lines=NETWORK) == (1.0, pytest.approx(0.02745))
        # Private (and any comparator) owns no new build; a stake below the existing share; switch off;
        # a 2050 register booking the RAB.
        assert sizewell_terms({"state_new_build_share": 0.0}, settings=ON, network_lines=NETWORK) is None
        assert sizewell_terms({"state_new_build_share": 0.3}, settings=ON, network_lines=NETWORK) is None
        assert sizewell_terms(public, settings=dict(ON, enabled=False), network_lines=NETWORK) is None
        assert sizewell_terms(dict(public, rab_support_gbp=1.8e9), settings=ON, network_lines=NETWORK) is None
        assert sizewell_terms(public, settings={}, network_lines=NETWORK) is None

    def test_the_row_is_a_new_line_on_the_policy_stack(self):
        owned = {"state_new_build_share": 1.0, "levers": {"public_rate_basis": "real_cpi"}}
        rows = sizewell_row(owned, private_line=14.37, settings=ON, network_lines=NETWORK, account_source="acct")
        assert len(rows) == 1 and rows[0]["component_id"] == LINE_SIZEWELL and LINE_SIZEWELL in NEW_LINES
        assert rows[0]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-14.37 * _saving(0.02745))
        method = rows[0]["method"]
        assert all(text in method for text in ("6.73%", "NAO", "19.3%", "10.29%", "6.00%", "2.23"))
        assert sizewell_row({"state_new_build_share": 0.3, "levers": {}}, private_line=14.37, settings=ON,
                            network_lines=NETWORK, account_source="acct") == []
        assert sizewell_row({"state_new_build_share": 0.0}, private_line=14.37, settings=ON, network_lines=NETWORK,
                            account_source="acct") == []


CONV = {"factor": 1.04, "final_demand_mwh": 300.0e6, "cfd_eligible_mwh": 280.0e6, "ro_eligible_mwh": 285.0e6,
        "mwh": 3.1, "vat_rate": 0.05}


class TestTariff:
    @pytest.fixture(scope="class")
    def built(self):
        from tests.unit.test_public_ownership_nationalised_tariff import _private_bill, _register

        private = _private_bill()
        lines, diag = revenue_requirement(_register())
        adjustment = szc.household_adjustment(12.0, ON, state_share=1.0, rate=0.02745)
        components, summary = build_bill(private, lines, CONV, sizewell_adjustment=adjustment, sizewell_note="n")
        recon, recon_summary = reconcile(private, components, lines, diag, CONV, {}, sizewell_adjustment=adjustment)
        plain, plain_summary = build_bill(private, lines, CONV)
        return adjustment, components, summary, recon, recon_summary, plain_summary

    def test_the_kept_levy_falls_and_both_ledgers_close(self, built):
        adjustment, components, summary, recon, recon_summary, plain_summary = built
        assert components.set_index("component_id").loc["sizewell_c_rab", VALUE] == pytest.approx(12.0 + adjustment)
        assert summary["ex_vat_gbp"] - plain_summary["ex_vat_gbp"] == pytest.approx(adjustment)
        assert summary["sizewell_c_rab_public_rate_adjustment_gbp_per_3100kwh_ex_vat"] == pytest.approx(adjustment)
        change = summary["delta_ex_vat_gbp"]
        for ledger in ("bill", "register"):
            assert recon.loc[recon["ledger"].eq(ledger), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(change)
        expected = system(adjustment, CONV["final_demand_mwh"], CONV["factor"], CONV["mwh"])
        assert recon_summary["sizewell_c_rab_public_rate_system_gbp"] == pytest.approx(expected)
        assert any("OW10" in step for step in recon_summary["attribution_gbp"])
        assert abs(recon_summary["unexplained_gbp"]) < 1.0

    def test_a_tariff_without_the_switch_is_unchanged(self, built):
        *_, plain_summary = built
        assert plain_summary["sizewell_c_rab_public_rate_adjustment_gbp_per_3100kwh_ex_vat"] == 0.0


class TestThreePartyIdentity:
    POSITION = {"private_rent_gbp": 0.0, "required_revenue_gbp": 0.0, "capital_recovery_gbp": 0.0,
                "corporation_tax_forgone_gbp": 0.0}

    def _bill(self, **kw):
        from tests.unit.test_public_ownership_investor_ledger import _bill_case

        return _bill_case(**kw)

    def test_the_market_line_is_a_cost_of_capital_saving_and_a_wash_for_investors(self):
        parties = inv.market_case_parties(
            bill_case=self._bill(sizewell_c_rab_public_rate_gbp_2024=-80 * M), bill_private=self._bill(),
            account_case={}, position=self.POSITION, position_private=self.POSITION,
            network=inv.network_parties(None, "private", None))
        assert parties["resource_parts_gbp"][inv.CAPITAL_SIZEWELL] == pytest.approx(80 * M)
        row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=0.0, case_id="p2")
        assert row["consumer_change_vs_private_gbp"] == pytest.approx(80 * M)
        assert row["investor_change_vs_private_gbp"] == pytest.approx(0.0)
        assert row["resource_saving_vs_private_gbp"] == pytest.approx(80 * M)
        # Observed-return reading: the RAB's capital providers are credited with the return paid.
        assert row["investor_change_observed_return_basis_gbp"] == pytest.approx(-80 * M)

        # The itemised flows: one flow from resource to consumers, and the three checks close.
        flows = party_flows.Flows("p2", lambda gbp: gbp / 1e6)
        components = {k: 0.0 for k in ("state_capital", "private_capital", "state_operating", "private_operating",
                                       "corporation_tax", "takeover_positive", "takeover_negative", "carbon_cfd_private", "tax_equivalent_private",
                                       "ldes_state", "ldes_private")}
        party_flows.market_flows(flows, parties=parties, components=components, account_case={},
                                 network=inv.network_parties(None, "private", None), levers={})
        assert [(r["payer"], r["payee"], r["gbp"]) for r in flows.rows] == [("resource", "consumers", 80 * M)]
        party_flows.check(flows.rows, {"case_id": "p2", "exchequer_net_vs_private_gbp": 0.0, **row})

    def test_the_tariff_item_is_assigned_to_resource(self):
        from tests.unit.test_public_ownership_investor_ledger import TestTariffCase, _tariff_case

        case = _tariff_case(bill_change_extra=-80 * M)
        case["reconciliation"]["sizewell_c_rab_public_rate_system_gbp"] = -80 * M
        parties = inv.tariff_case_parties(case, corporation_tax_legacy_gbp=8 * M, corporation_tax_new_build_gbp=9 * M,
                                          network=TestTariffCase.NETWORK)
        row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=-(8 + 9 + 5) * M, case_id="nat")
        assert row["resource_saving_vs_private_gbp"] == pytest.approx(36 * M + 80 * M)
        assert row["investor_change_vs_private_gbp"] == pytest.approx(-38.5 * M)
