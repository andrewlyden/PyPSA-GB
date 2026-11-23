"""The private-investor column and the three-party identity (review 29 Sep 2026).

Consumers + Exchequer + investors = resource saving, per case, to GBP 1.  Small
hand-checkable inputs: a market case where the state owns and re-prices a wind
farm, pure transfers that must net to zero (pay-as-bid, merchant contracts at
cost, a network premium over RAV), the acquisition tariff's register ledger, the
bid stack on the private fleet and the I3 institution step; and the public
account re-pricing R1 rows as the bill does, the defect the identity found.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import investor_ledger as inv
from scripts.public_ownership.exchequer_ledger import build, guarantee_expected_loss, guarantee_terms
from scripts.public_ownership.levers import LEVER_GUARANTEE_COST, LEVERS
from scripts.public_ownership.nationalised_tariff import (
    LINE_CANCELLED_RESIDUAL, LINE_CM_NOT_ACQUIRED, LINE_FORECAST_ERROR, LINE_IC_ACQUISITION, LINE_IC_FOREIGN_RENT,
    LINE_IC_RETRADING, LINE_IMPORTS, LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, LINE_NEW_BUILD, LINE_REALTIME,
    LINE_STATE_OVERHEAD, LINE_VARIABLE_OM,
)

M = 1.0e6
CONV = {"factor": 1.04, "final_demand_mwh": 300.0e6, "mwh": 3.1, "vat_rate": 0.05, "domestic_mwh": 100.0e6}


def _row(case, asset, *, share, rr, market, support, fuel=0.0, ct=0.0, vintage="new_build", price=np.nan,
         capital=np.nan):
    rent = market + support - fuel - rr
    return {"case_id": case, "asset_id": asset, "finance_vintage": vintage, "finance_technology": "Offshore wind",
            "ownership_share": share, "state_owned": share > 0, "annual_capital_recovery_gbp": capital,
            "wacc": 0.05, "remaining_life_years": 25.0, "acquisition_price_gbp": price, "public_return": "public_rate",
            "corporation_tax_forgone_gbp": ct, "annual_required_revenue_gbp": rr, "market_revenue_gbp": market,
            "support_revenue_gbp": support, "capacity_market_revenue_gbp": 0.0, "ldes_floor_topup_paid_gbp": 0.0,
            "fuel_carbon_gbp": fuel, "producer_rent_gbp": rent, "public_surplus_before_recycling_gbp": share * rent}


def _register():
    return pd.DataFrame([
        _row("private", "wind", share=0.0, rr=100 * M, market=60 * M, support=50 * M, capital=100 * M),
        _row("private", "nuke", share=0.0, rr=300 * M, market=350 * M, support=0.0, fuel=20 * M, vintage="legacy"),
        # P2 owns the wind farm outright at the public rate: 70 m instead of 100 m, 12 m of it tax.
        _row("p2", "wind", share=1.0, rr=70 * M, market=60 * M, support=50 * M, ct=12 * M, capital=70 * M),
        _row("p2", "nuke", share=0.0, rr=300 * M, market=350 * M, support=0.0, fuel=20 * M, vintage="legacy"),
    ])


def _bill_case(**kw):
    base = {"ex_vat_gbp": 900.0, "total_gbp_inc_vat": 945.0, "financing_saving_gbp_2024": 0.0,
            "levers": {"networks": "private"}, "bill_allocated_r1_gbp_2024": 0.0, "bill_allocated_ldes_gbp_2024": 0.0,
            "bill_allocated_w1_gbp_2024": 0.0, "bill_allocated_public_cfd_at_cost_gbp_2024": 0.0,
            "bill_allocated_rebate_gbp_2024": 0.0, "shortfall_on_bill_gbp_2024": 0.0,
            "route_booking_booked_gbp_2024": {}, "new_build_contracts_bill_gbp_2024": 0.0,
            "network_adjustments_system_gbp_2024": {}, "pay_as_bid_ceiling_gbp_2024": 0.0}
    return {**base, **kw}


def _account(p2_retained):
    return {"cases": {
        "private": {"state_deficit_exchequer_funded_gbp": 0.0, "state_surplus_retained_gbp": 0.0},
        # The state's rent on the wind farm is 110 - 70 = 40 m; the strike cut (at cost) takes 20 m of it.
        "p2": {"state_deficit_exchequer_funded_gbp": 0.0, "state_surplus_retained_gbp": p2_retained,
               "r1_bill_cut_gbp": 20 * M, "r1_cut_borne_by_state_gbp": 20 * M},
    }}


NETWORK = pd.DataFrame({"treatment": ["acquired"], "sector": ["distribution"], "rav_gbp": [1000 * M],
                        "allowed_wacc_cpih_real": [0.05], "financing_saving_gbp": [17 * M],
                        "tax_allowance_saving_gbp": [5 * M]})
CARBON = {"market": {"carbon_receipts_gbp": 1.5e9, "carbon_uk_ets_gbp": 1.5e9, "carbon_price_support_gbp": 0.0,
                     "chp_heat_credit_carbon_gbp": 0.0}}


def _build(account):
    bill = {"cases": {"private": _bill_case(),
                      "p2": _bill_case(ex_vat_gbp=899.8, total_gbp_inc_vat=899.8 * 1.05, bill_allocated_r1_gbp_2024=20 * M)}}
    return build(register=_register(), bill=bill, account=account, network_lines=NETWORK, conv=CONV, wedges={},
                 carbon=CARBON, investor={"network_public_rate": 0.03})


class TestMarketCase:
    def test_the_identity_closes_and_splits_the_saving(self):
        frame, summary = _build(_account(p2_retained=20 * M))
        p2 = summary["cases"]["p2"]
        # Investors lose the wind farm's 10 m rent; the resource saving is the 30 m
        # financing saving less the 12 m of tax inside it; consumers get the 20 m cut.
        assert p2["investor_change_vs_private_gbp"] == pytest.approx(-10 * M)
        assert p2["resource_saving_vs_private_gbp"] == pytest.approx(18 * M)
        assert p2["consumer_change_vs_private_gbp"] == pytest.approx(20 * M - p2["vat_change_vs_private_gbp"])
        assert p2["exchequer_net_vs_private_gbp"] == pytest.approx(p2["vat_change_vs_private_gbp"] - 12 * M + 20 * M)
        assert abs(p2["three_party_identity_residual_gbp"]) <= inv.IDENTITY_TOLERANCE_GBP
        assert summary["cases"]["private"]["investor_change_vs_private_gbp"] == 0.0
        assert "three_party_identity" in summary and summary["three_party_identity"]["tolerance_gbp"] == 1.0
        assert "investor_change_vs_private_gbp_per_household" in frame.columns

    def test_a_transfer_booked_to_one_party_only_is_refused(self):
        # An account that forgets the strike cut leaves the state 20 m richer than
        # the bill made consumers poorer: the pound is counted twice.
        with pytest.raises(ValueError, match="a transfer is booked to one party"):
            _build(_account(p2_retained=40 * M))

    def test_without_the_investor_inputs_the_ledger_is_unchanged(self):
        bill = {"cases": {"private": _bill_case()}}
        frame, summary = build(register=_register(), bill=bill, account=_account(0.0), network_lines=NETWORK, conv=CONV,
                               wedges={}, carbon=CARBON)
        assert "investor_change_vs_private_gbp" not in frame.columns and "three_party_identity" not in summary


class TestPureTransfers:
    POSITION = {"private_rent_gbp": 0.0, "required_revenue_gbp": 0.0, "capital_recovery_gbp": 0.0,
                "corporation_tax_forgone_gbp": 0.0}

    def _parties(self, bill_case, network=None):
        return inv.market_case_parties(bill_case=bill_case, bill_private=_bill_case(), account_case={},
                                       position=self.POSITION, position_private=self.POSITION,
                                       network=network or inv.network_parties(None, "private", None))

    def test_pay_as_bid_and_merchant_contracts_at_cost_net_to_zero(self):
        for bill_case in (_bill_case(pay_as_bid_ceiling_gbp_2024=50 * M),
                          _bill_case(new_build_contracts_bill_gbp_2024=30 * M)):
            parties = self._parties(bill_case)
            row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=0.0, case_id="x")
            assert row["resource_saving_vs_private_gbp"] == 0.0
            assert row["consumer_change_vs_private_gbp"] == pytest.approx(-row["investor_change_vs_private_gbp"])

    def test_a_network_premium_over_rav_goes_to_the_vendors_and_the_tax_to_the_exchequer(self):
        network = inv.network_parties(NETWORK, "acquired", 0.03)
        # 1,000 m RAV at 5% allowed against 3%: a 20 m wedge; bought at 1.1 x RAV the
        # consumer saves 17 m of it plus the 5 m tax allowance, the vendors get 3 m.
        assert network == pytest.approx({"bill_gbp": -22 * M, "tax_gbp": 5 * M, "premium_gbp": 3 * M, "wedge_gbp": 20 * M,
                                         "guarantee_expected_loss_gbp": 0.0, "fee_gbp": 0.0, "retained_gbp": 0.0,
                                         "gross_wedge_gbp": 20 * M, "excess_return_gbp": 0.0})
        parties = self._parties(_bill_case(network_adjustments_system_gbp_2024={"duos_public_financing_adjustment": -22 * M}),
                                network)
        row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=-5 * M, case_id="x")
        assert row["investor_change_vs_private_gbp"] == pytest.approx(3 * M)
        assert row["resource_saving_vs_private_gbp"] == pytest.approx(20 * M)

    def test_an_acquired_network_needs_the_public_rate(self):
        with pytest.raises(ValueError, match="public_real_rate"):
            inv.network_parties(NETWORK, "acquired", None)


def _tariff_case(bill_change_extra=0.0):
    lines = {LINE_LEGACY_RENEWABLES: 40 * M, LINE_LEGACY_THERMAL: 10 * M, LINE_NEW_BUILD: 70 * M,
             LINE_STATE_OVERHEAD: 5 * M, LINE_IMPORTS: -6 * M, LINE_IC_FOREIGN_RENT: 15 * M, LINE_IC_ACQUISITION: 7 * M,
             LINE_CM_NOT_ACQUIRED: 8 * M, LINE_CANCELLED_RESIDUAL: 0.0, LINE_IC_RETRADING: 0.0, LINE_REALTIME: 0.0,
             LINE_FORECAST_ERROR: 0.0}
    diagnostics = {"legacy_operating_margin_gbp": 110 * M, "new_build_private_rent_gbp": -50 * M,
                   "new_build_capital_private_gbp": 100 * M, "private_roc_support_gbp": 60 * M,
                   "interconnector_gb_half_rent_to_tariff_gbp": 11 * M, "import_register_domestic_revenue_gbp": 14 * M,
                   "cancelled_contract_residual_exchequer_transfer_gbp": 0.0}
    removed = {"capacity_market": 25 * M, "ldes_cap_and_floor": 12 * M, "procurement_transaction_cost": 4 * M,
               "renewables_obligation": 62 * M, "interconnector_cap_and_floor_pass_through": -2 * M}
    # Investor items -46.5 m, resource items -25 m, networks -25 m (see the module test notes).
    change = -96.5 * M + bill_change_extra
    return {"case_id": "nat", "revenue_requirement_gbp": lines, "diagnostics": diagnostics,
            "bill_detail": {"interconnector_retrading_cost_gbp": 6 * M},
            "reconciliation": {"bill_change_system_gbp": change, "replaced_lines_system_gbp_by_line": removed,
                               "residual_gbp": 0.5 * M}}


class TestTariffCase:
    NETWORK = {"bill_gbp": -25 * M, "tax_gbp": 5 * M, "premium_gbp": 0.0, "wedge_gbp": 20 * M}

    def test_every_register_ledger_item_has_one_party_and_the_identity_closes(self):
        parties = inv.tariff_case_parties(_tariff_case(), corporation_tax_legacy_gbp=8 * M,
                                          corporation_tax_new_build_gbp=9 * M, network=self.NETWORK)
        # Exchequer: the tax on all owned plant and the distribution tax allowance forgone.
        row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=-(8 + 9 + 5) * M, case_id="nat")
        assert row["investor_change_vs_private_gbp"] == pytest.approx(-38.5 * M)
        assert row["resource_saving_vs_private_gbp"] == pytest.approx(36 * M)
        # Cross-border: foreign counterparties get 14 - 11 + 6 = 9 m under private, -6 + 15 = 9 m under the tariff.
        trade = parties["resource_parts_gbp"]
        # F19 P3 decision 5: one net item (re-trading is paid inside the measured import line under I1).
        assert trade["cross-border energy and interconnector re-trading, net of what private paid abroad "
                     "(foreign price on the links, the foreign halves' rent)"] == pytest.approx(0.0)
        assert not any("re-trading no longer paid abroad" in label for label in trade)

    def test_the_o_and_m_residual_is_a_resource_change_not_a_payment_to_investors(self):
        """Money audit (13 Nov 2026): the tariff's O&M residual (self-insurance, any public O&M premium) moves from
        investors to the resource saving, as in S1 and S2; the identity still closes."""
        from scripts.public_ownership.nationalised_tariff import RESIDUAL_OM_PART

        base = _tariff_case()
        case = _tariff_case()
        case["reconciliation"] = dict(case["reconciliation"], residual_parts_gbp={RESIDUAL_OM_PART: -0.3 * M,
                                                                                  "other": 0.8 * M})
        kw = dict(corporation_tax_legacy_gbp=8 * M, corporation_tax_new_build_gbp=9 * M, network=self.NETWORK)
        r0 = inv.three_party_row(inv.tariff_case_parties(base, **kw), vat_change_gbp=0.0,
                                 exchequer_change_gbp=-(8 + 9 + 5) * M, case_id="nat")
        r1 = inv.three_party_row(inv.tariff_case_parties(case, **kw), vat_change_gbp=0.0,
                                 exchequer_change_gbp=-(8 + 9 + 5) * M, case_id="nat")
        # A 0.3 m O&M saving is the economy's, not a 0.3 m loss to owners.
        assert r1["resource_saving_vs_private_gbp"] - r0["resource_saving_vs_private_gbp"] == pytest.approx(0.3 * M)
        assert r1["investor_change_vs_private_gbp"] - r0["investor_change_vs_private_gbp"] == pytest.approx(0.3 * M)

    def test_a_tariff_line_nobody_is_assigned_is_refused(self):
        with pytest.raises(ValueError, match="no party in the investor ledger"):
            inv.tariff_case_parties(_tariff_case(bill_change_extra=3 * M), corporation_tax_legacy_gbp=0.0,
                                    corporation_tax_new_build_gbp=0.0, network=self.NETWORK)

    def test_the_acquisition_memo_splits_the_annuity_and_both_readings_close(self):
        rows = pd.DataFrame({"asset_id": ["a", "b"], "acquisition_price_gbp": [600 * M, 400 * M],
                             "annual_capital_recovery_gbp": [30 * M, 20 * M]})
        valuation = pd.DataFrame({"asset_id": ["a", "b"], "pv_state_gbp": [700 * M, 300 * M],
                                  "pv_market_gbp": [500 * M, 250 * M]})
        memo = inv.legacy_acquisition_memo(rows, valuation)
        assert memo["legacy_annuity_factor"] == pytest.approx(0.05)
        assert memo["legacy_premium_over_owners_valuation_annuitised_gbp"] == pytest.approx(250 * M * 0.05)
        assert memo["legacy_cost_of_capital_gap_annuitised_gbp"] == pytest.approx(250 * M * 0.05)
        row = {"investor_change_vs_private_gbp": -40 * M, "resource_saving_vs_private_gbp": 36 * M}
        inv.acquisition_gap_as_resource(row, memo)
        # The gap moves from investors to the resource saving: the sum is unchanged.
        assert (row["resource_saving_acquisition_gap_as_resource_gbp"]
                - row["investor_change_acquisition_gap_as_resource_gbp"] == pytest.approx(76 * M))
        with pytest.raises(ValueError, match="no legacy valuation"):
            inv.legacy_acquisition_memo(rows, valuation.iloc[:1])


class TestDispatchAndInstitution:
    def _position(self, **kw):
        base = {k: 0.0 for k in ("private_rent_gbp", "producer_rent_gbp", "required_revenue_gbp",
                                 "corporation_tax_forgone_gbp", "revenue_gbp", "unresolved_net_cash_gbp",
                                 "fuel_carbon_gbp", "wholesale_revenue_gbp", "balancing_settlement_gbp",
                                 "cfd_support_gbp", "roc_support_gbp", "ldes_floor_topup_paid_gbp")}
        return {**base, **kw}

    def test_the_bid_stack_on_the_private_fleet_closes_on_the_registers_own_rent(self):
        market = self._position()
        # Balancing payments fall 95 m, ROC income rises 10 m, fuel and carbon +5 m, VOM +1 m.
        group = self._position(balancing_settlement_gbp=-95 * M, roc_support_gbp=10 * M, revenue_gbp=-85 * M,
                               fuel_carbon_gbp=5 * M, required_revenue_gbp=1 * M, producer_rent_gbp=-91 * M,
                               private_rent_gbp=-91 * M)
        parties = inv.dispatch_step_parties(line_changes={"bsuos_constraints": -100 * M, "renewables_obligation": 3 * M},
                                            position_group=group, position_market=market, carbon_change_gbp=1.5 * M,
                                            retrading_change_gbp=2 * M)
        row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=1.5 * M, case_id="bids")
        assert row["resource_saving_vs_private_gbp"] == pytest.approx(-6.5 * M)
        assert row["investor_change_vs_private_gbp"] == pytest.approx(-105 * M)
        with pytest.raises(ValueError, match="cannot assign"):
            inv.dispatch_step_parties(line_changes={"tnuos": 1.0}, position_group=group, position_market=market,
                                      carbon_change_gbp=0.0, retrading_change_gbp=0.0)

    def test_the_institution_step_is_resource_and_carbon_and_must_read_the_same_carbon(self):
        institution = {"institution_change_system_gbp": -3 * M, "audit": {
            "fuel_delta_gbp": 30 * M, "carbon_delta_gbp": 10 * M, "interconnector_retrading_removed_gbp": 0.0,
            "realtime_correction_replaced_by_charge_gbp": 4 * M, "forecast_error_charge_added_gbp": 6 * M,
            "measured_deltas_gbp": {LINE_VARIABLE_OM: 2 * M, LINE_IMPORTS: -50 * M, LINE_IC_FOREIGN_RENT: 3 * M}}}
        step = inv.institution_step_parties(institution, carbon_change_gbp=10 * M)
        row = inv.three_party_row(step, vat_change_gbp=0.0, exchequer_change_gbp=10 * M, case_id="i3")
        assert row["resource_saving_vs_private_gbp"] == pytest.approx(13 * M)
        assert row["investor_change_vs_private_gbp"] == 0.0
        with pytest.raises(ValueError, match="read different months or rates"):
            inv.institution_step_parties(institution, carbon_change_gbp=9 * M)


class TestAccountRepricesAsTheBillDoes:
    def test_an_undeliverable_at_cost_contract_stays_on_the_financing_basis_in_the_account(self):
        from scripts.public_ownership.build_public_account import reprice_as_the_bill_does
        from tests.unit.test_public_ownership_at_cost_limit import SETTLED, _savings, _table

        case = "public_ownership@all_alternatives"
        adjustments = pd.DataFrame({
            "case_id": ["private", case], "strike_key": ["AR3_remote_island_wind"] * 2,
            "eligible_dispatch_mwh": [SETTLED, SETTLED], "r1_saving_gbp_2024": [0.0, 1_000_000.0]})
        savings = reprice_as_the_bill_does(_savings(10_267_000.0), adjustments, _table())
        row = savings.loc[savings["case_id"].eq(case)].iloc[0]
        assert row["saving_bill_term_gbp"] == pytest.approx(1_000_000.0) and not row["at_cost_applied"]
        # A strike adjustments file the account cannot reproduce is refused.
        with pytest.raises(ValueError, match="re-price differently"):
            reprice_as_the_bill_does(_savings(10_267_000.0), adjustments.assign(r1_saving_gbp_2024=[0.0, 2.0e6]),
                                     _table())


class TestObservedReturnDiagnostic:
    def test_the_two_investor_bases_are_labelled_and_both_close(self):
        _, summary = _build(_account(p2_retained=20 * M))
        p2 = summary["cases"]["p2"]
        assert p2["investor_basis"] == "required_return"
        assert p2["investor_change_required_return_basis_gbp"] == p2["investor_change_vs_private_gbp"]
        # The 30 m capital-charge fall less its 12 m of tax is the cost-of-capital saving; on the
        # observed-return basis capital providers lose it and it is no longer a resource saving.
        assert p2["cost_of_capital_saving_gbp"] == pytest.approx(18 * M)
        assert p2["investor_change_observed_return_basis_gbp"] == pytest.approx(-28 * M)
        assert p2["resource_saving_observed_return_basis_gbp"] == pytest.approx(0.0, abs=1.0)
        closes = (p2["consumer_change_vs_private_gbp"] + p2["exchequer_net_vs_private_gbp"]
                  + p2["investor_change_observed_return_basis_gbp"] - p2["resource_saving_observed_return_basis_gbp"])
        assert abs(closes) <= inv.IDENTITY_TOLERANCE_GBP


GUARANTEE_TABLE = pd.DataFrame({
    "setting": ["central", "low", "high"], "annual_default_probability": [0.00231, 0.00169, 0.00549],
    "loss_given_default": [0.35, 0.212, 0.35], "expected_loss_bps": [8.085, 3.5828, 19.215], "sources": ["s"] * 3})
GUARANTEE_ACCOUNT = {"treatments": {"debt_guarantee": {
    "transmission": {"rav_gbp": 1000 * M, "notional_gearing": 0.55, "debt_compression": 0.0075,
                     "financing_saving_gbp": 1000 * M * 0.55 * 0.0075},
    "distribution": {"rav_gbp": 2000 * M, "notional_gearing": 0.6, "debt_compression": 0.0075,
                     "financing_saving_gbp": 2000 * M * 0.6 * 0.0075}}}}


class TestDebtGuarantee:
    """F29 (26 Oct 2026): networks: debt_guarantee and its guarantee_cost lever are retired (workbook v2.0 has no
    network rate); the ledger prices no guarantee and a case without one is charged nothing."""

    def test_the_lever_is_retired_and_the_terms_are_refused(self):
        assert LEVER_GUARANTEE_COST not in LEVERS
        with pytest.raises(ValueError, match="retired 26 Oct 2026"):
            guarantee_terms(GUARANTEE_TABLE, GUARANTEE_ACCOUNT)

    def test_a_case_without_a_guarantee_is_charged_nothing(self):
        assert guarantee_expected_loss({"networks": "private"}, None, "p") == {"guarantee_expected_loss_gbp": 0.0}
        assert guarantee_expected_loss({"networks": "acquired"}, None, "p") == {"guarantee_expected_loss_gbp": 0.0}


def test_bid_stack_retrading_goes_abroad_not_to_generators():
    """The re-trading inside the constraint line is paid abroad (30 Sep 2026 rerun: a GBP 2.45 m gap).

    dispatch_step_parties removes it from the investors' constraint-line change and
    books it as paid abroad; the flow table must split the line the same way.
    """
    from scripts.public_ownership import party_flows as pf

    step = {"bill_lines_gbp": {"bsuos_constraints": 100.0, "cfd": 20.0},
            "resource_parts_gbp": {pf.RETRADING_ABROAD: -30.0, "fuel, net of the carbon": 5.0}}
    flows = pf.Flows("p2_bids", lambda gbp: gbp)
    pf.dispatch_step_flows(flows, step, carbon_change_gbp=0.0)
    net = pf.net_by_party(flows.rows)
    assert net["foreign"] == pytest.approx(30.0)
    # Generators get the constraint line less the re-trading, the CfD line and the fuel part.
    assert net["private_investors"] == pytest.approx(70.0 + 20.0 + 5.0)
    assert not any(r["payer"] == "resource" and r["payee"] == "foreign" for r in flows.rows)
