"""Items N5, N7, N4 and N8: the nationalised tariff replaces the market lines with cost, and reconciles.

A small register with the four things the tariff must treat differently: a
legacy renewable at its acquisition charge, legacy gas at book value, new build
at the public rate, an import valued at what GB paid it, and a CHP unit without a
cost mapping paid as today.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.finance import NATIONALISED_CASE, capital_recovery_factor
from scripts.public_ownership.nationalised_tariff import (
    LINE_FIXED_OM, LINE_FUEL, LINE_IC_RETRADING, LINE_IMPORTS, LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL,
    LINE_NEW_BUILD, LINE_REALTIME, LINE_UNACQUIRED, LINE_VARIABLE_OM, REPLACED_LINES, VALUE, build_bill,
    conversion, household, legacy_capital_variants, multi_year, reconcile, revenue_requirement,
)

RATE = 0.02745
CRF30 = capital_recovery_factor(RATE, 30)
CONV = {"factor": 1.04, "final_demand_mwh": 300.0e6, "cfd_eligible_mwh": 280.0e6, "ro_eligible_mwh": 285.0e6,
        "mwh": 3.1, "vat_rate": 0.05}


def _row(case, asset_id, carrier, vintage, *, owned, scope=None, capital=0.0, fixed=0.0, var=0.0, fuel=0.0,
         market=0.0, support=0.0, price=np.nan, resolved=True, dispatch=1.0e6):
    rr = capital + fixed + var if resolved else np.nan
    return {
        "case_id": case, "asset_id": asset_id, "carrier": carrier, "finance_vintage": vintage, "state_owned": owned,
        "legacy_scope_group": scope, "annual_capital_recovery_gbp": capital if resolved else np.nan,
        "annual_fixed_om_gbp": fixed if resolved else np.nan, "annual_variable_om_gbp": var if resolved else np.nan,
        "annual_required_revenue_gbp": rr, "fuel_carbon_gbp": fuel, "market_revenue_gbp": market,
        "support_revenue_gbp": support, "acquisition_price_gbp": price, "capacity_mw": 100.0, "dispatch_mwh": dispatch,
        "model_year": 2030, "wholesale_revenue_gbp": market, "balancing_settlement_gbp": 0.0,
        "cfd_support_gbp": support if carrier == "wind_offshore" and vintage == "new_build" else 0.0,
        "roc_support_gbp": support if carrier == "wind_offshore" and vintage == "legacy" else 0.0,
    }


def _register():
    rows = []
    # private
    rows += [
        _row("private", "old_wind", "wind_offshore", "legacy", owned=False, capital=50.0e6, fixed=10.0e6, market=60.0e6, support=40.0e6),
        _row("private", "old_gas", "CCGT", "legacy", owned=False, capital=30.0e6, fixed=8.0e6, var=2.0e6, fuel=50.0e6, market=45.0e6),
        _row("private", "new_wind", "wind_offshore", "new_build", owned=False, capital=80.0e6, fixed=12.0e6, market=70.0e6, support=40.0e6),
        _row("private", "import", "EU_market", "new_build", owned=False, market=20.0e6, resolved=False),
        _row("private", "chp", "CHP", "legacy", owned=False, market=9.0e6, fuel=15.0e6, resolved=False),
    ]
    # nationalised
    rows += [
        _row(NATIONALISED_CASE, "old_wind", "wind_offshore", "legacy", owned=True, scope="valued", capital=200.0e6 * CRF30,
             fixed=10.0e6, market=60.0e6, support=40.0e6, price=200.0e6),
        _row(NATIONALISED_CASE, "old_gas", "CCGT", "legacy", owned=True, scope="thermal", capital=100.0e6 * CRF30,
             fixed=8.0e6, var=2.0e6, fuel=50.0e6, market=45.0e6, price=100.0e6),
        _row(NATIONALISED_CASE, "new_wind", "wind_offshore", "new_build", owned=True, capital=50.0e6, fixed=12.0e6, market=70.0e6, support=40.0e6),
        _row(NATIONALISED_CASE, "import", "EU_market", "new_build", owned=False, scope="excluded", market=20.0e6, resolved=False),
        _row(NATIONALISED_CASE, "chp", "CHP", "legacy", owned=True, scope="thermal", market=9.0e6, fuel=15.0e6, resolved=False),
    ]
    return pd.DataFrame(rows)


def _private_bill():
    values = {"direct_energy": 150.0, "shaping_adjustment": 1.0, "procurement_losses": 14.0, "cfd": 130.0,
              "bsuos_constraints": 40.0, "renewables_obligation": 55.0, "capacity_market": 10.0, "tnuos": 110.0,
              "duos": 120.0, "supplier_costs_and_margin": 160.0, "feed_in_tariffs": 26.0, "allocated_ccs": 4.0,
              "ldes_cap_and_floor": 20.0, "sizewell_c_rab": 12.0}
    return pd.DataFrame([{
        "case_id": "private", "scenario": "CP30_2030_cp30_network", "modelled_year": 2030, "component_id": k, "stack_group": "x",
        "category": "x", "component": k, VALUE: v, "source_class": "x", "method": "m", "source": "s", "currency": "GBP",
        "reference_gbp_per_3100kwh_ex_vat": np.nan,
    } for k, v in values.items()])


class TestRevenueRequirement:
    def test_lines_come_from_the_right_rows(self):
        lines, diag = revenue_requirement(_register(), realtime_correction_gbp=3.0e6, interconnector_retrading_gbp=2.0e6)
        assert lines[LINE_LEGACY_RENEWABLES] == pytest.approx(200.0e6 * CRF30)
        assert lines[LINE_LEGACY_THERMAL] == pytest.approx(100.0e6 * CRF30)
        assert lines[LINE_NEW_BUILD] == pytest.approx(50.0e6)
        assert lines[LINE_FIXED_OM] == pytest.approx(30.0e6) and lines[LINE_VARIABLE_OM] == pytest.approx(2.0e6)
        assert lines[LINE_FUEL] == pytest.approx(50.0e6)  # CHP fuel is not costed: it is paid as today
        assert lines[LINE_IMPORTS] == pytest.approx(20.0e6)
        assert lines[LINE_UNACQUIRED] == pytest.approx(9.0e6)
        assert lines[LINE_REALTIME] == pytest.approx(3.0e6)
        assert lines[LINE_IC_RETRADING] == pytest.approx(2.0e6)
        assert diag["paid_as_today_carriers"] == ["CHP"]
        assert diag["import_mwh"] == pytest.approx(1.0e6)
        assert diag["acquisition_price_gbp"] == pytest.approx(300.0e6)
        assert diag["legacy_operating_margin_gbp"] == pytest.approx((60 + 40 - 10) * 1e6 + (45 - 50 - 8 - 2) * 1e6)
        assert diag["new_build_private_rent_gbp"] == pytest.approx((70 + 40 - 92) * 1e6)

    def test_override_replaces_legacy_capital_row_by_row(self):
        override = pd.Series({"old_gas": 0.0})
        lines, _ = revenue_requirement(_register(), legacy_capital_override=override)
        assert lines[LINE_LEGACY_THERMAL] == 0.0 and lines[LINE_LEGACY_RENEWABLES] == pytest.approx(200.0e6 * CRF30)


class TestBillAndReconciliation:
    @pytest.fixture(scope="class")
    def built(self):
        register, private = _register(), _private_bill()
        lines, diag = revenue_requirement(register)
        components, summary = build_bill(private, lines, CONV)
        detail = {"cfd_economic_curtailment_payment_gbp": 5.0e6, "interconnector_retrading_cost_gbp": 3.0e6}
        recon, recon_summary = reconcile(private, components, lines, diag, CONV, detail)
        return private, lines, components, summary, recon, recon_summary

    def test_replaced_lines_go_and_kept_lines_stay(self, built):
        private, lines, components, summary, _, _ = built
        ids = set(components["component_id"])
        assert not ids & set(REPLACED_LINES)
        assert {"tnuos", "duos", "supplier_costs_and_margin", "feed_in_tariffs", "sizewell_c_rab"} <= ids
        # Losses are replaced (27 Sep 2026): under metered recovery the owned fleet
        # already generates for GSP-level load, so keeping the line counted it twice.
        assert "procurement_losses" not in ids
        assert summary["replaced_lines_total_gbp_per_3100kwh_ex_vat"] == pytest.approx(
            150 + 1 + 14 + 130 + 40 + 55 + 10 + 4 + 20)
        assert (components["case_id"] == NATIONALISED_CASE).all()

    def test_new_lines_convert_flat_over_final_demand(self, built):
        _, lines, components, summary, _, _ = built
        got = components.set_index("component_id")[VALUE]
        for line_id, amount in lines.items():
            assert got[line_id] == pytest.approx(household(amount, CONV["final_demand_mwh"], CONV["factor"], CONV["mwh"]))
        assert summary["vat_gbp"] == pytest.approx(summary["ex_vat_gbp"] * 0.05)
        assert summary["delta_inc_vat_gbp"] == pytest.approx((summary["ex_vat_gbp"] - summary["private_ex_vat_gbp"]) * 1.05)

    def test_both_ledgers_close_to_the_bill_change(self, built):
        private, _, components, summary, recon, recon_summary = built
        change = summary["delta_ex_vat_gbp"]
        bill = recon.loc[recon["ledger"].eq("bill"), "gbp_per_3100kwh_ex_vat"].sum()
        register = recon.loc[recon["ledger"].eq("register"), "gbp_per_3100kwh_ex_vat"].sum()
        assert bill == pytest.approx(change) and register == pytest.approx(change)
        assert recon_summary["bill_change_gbp_per_3100kwh_ex_vat"] == pytest.approx(change)
        # CfD and RO lines are converted at their own recovery denominators.
        cfd_row = recon.loc[recon["component_id"].eq("cfd")].iloc[0]
        assert cfd_row["system_gbp_model_year"] == pytest.approx(-130.0 * CONV["cfd_eligible_mwh"] / CONV["factor"] / 3.1)
        # Named non-register payments, then the attribution, then a declared residual.
        ledger = recon_summary["register_ledger"]
        assert any("LDES floor" in s for s in ledger) and any("Capacity Market" in s for s in ledger)
        curtailment_key = next(k for k in recon_summary["not_register_payments_gbp"] if k.startswith("CfD economic-curtailment"))
        assert recon_summary["not_register_payments_gbp"][curtailment_key] == pytest.approx(-5.0e6)
        assert any(s.startswith("residual") for s in ledger)
        # The residual is not a plug: it is the sum of its declared parts, and nothing is left over.
        assert len(recon_summary["residual_parts_gbp"]) == 6
        assert recon_summary["residual_gbp"] == pytest.approx(sum(recon_summary["residual_parts_gbp"].values()))
        assert abs(recon_summary["unexplained_gbp"]) < 1.0
        # The state's operating overhead (item OH1) is attributed by name, never left in the residual.
        assert any("item OH1" in s for s in recon_summary["attribution_gbp"])

    def test_the_rent_credit_goes_only_when_imports_are_paid_at_the_foreign_price(self):
        # User decision 26 Sep 2026: no congestion rent is collected when the public
        # buyer pays the foreign price on the links, so none is returned; the pool
        # (the rate of every new line) does not move, and both ledgers still close.
        from scripts.public_ownership.nationalised_tariff import (
            INTERCONNECTOR_PASS_THROUGH_LINE, dropped_lines_for,
        )
        register = _register()
        extra_row = _private_bill().iloc[[0]].assign(component_id=INTERCONNECTOR_PASS_THROUGH_LINE, **{VALUE: -1.18})
        private = pd.concat([_private_bill(), extra_row], ignore_index=True)
        detail = {"cfd_economic_curtailment_payment_gbp": 5.0e6, "interconnector_retrading_cost_gbp": 3.0e6}
        # Register basis: imports at the GB price, the credit stays.
        lines, diag = revenue_requirement(register)
        assert dropped_lines_for(diag) == {}
        # Foreign-price basis: the credit is dropped.
        lines_f, diag_f = revenue_requirement(register, interconnector_trade_gbp=15.0e6)
        dropped = dropped_lines_for(diag_f)
        assert set(dropped) == {INTERCONNECTOR_PASS_THROUGH_LINE}
        kept_comp, kept_summary = build_bill(private, lines_f, CONV)
        comp, summary = build_bill(private, lines_f, CONV, dropped_lines=dropped)
        assert INTERCONNECTOR_PASS_THROUGH_LINE in set(kept_comp["component_id"])
        assert INTERCONNECTOR_PASS_THROUGH_LINE not in set(comp["component_id"])
        assert summary["ex_vat_gbp"] == pytest.approx(kept_summary["ex_vat_gbp"] + 1.18)
        assert summary["dropped_lines_gbp_per_3100kwh_ex_vat"] == {INTERCONNECTOR_PASS_THROUGH_LINE: -1.18}
        recon, recon_summary = reconcile(private, comp, lines_f, diag_f, CONV, detail, dropped_lines=dropped)
        for ledger in ("bill", "register"):
            assert recon.loc[recon["ledger"].eq(ledger), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(
                summary["delta_ex_vat_gbp"])
        assert abs(recon_summary["unexplained_gbp"]) < 1.0

    def test_a_tariff_line_without_an_attribution_is_refused(self, built, monkeypatch):
        """The register ledger used to plug the residual as change - explained; an unattributed line now fails."""
        from scripts.public_ownership import nationalised_tariff as tariff

        private, lines, _, _, _, _ = built
        _, diag = revenue_requirement(_register())
        monkeypatch.setitem(tariff.NEW_LINE_LABELS, "nationalised_unattributed", "a line nobody attributed")
        extra = dict(lines, nationalised_unattributed=5.0e6)
        # A new line is first refused for having no presentation group (bill_groupings.csv, 6 Oct 2026) ...
        with pytest.raises(ValueError, match="no 'ofgem' group"):
            build_bill(private, extra, CONV)
        # ... and, grouped, still refused by the register ledger for having no attribution.
        monkeypatch.setattr(tariff, "annotate_bill_groups", lambda frame: frame)
        monkeypatch.setattr(tariff, "bill_group_totals", lambda frame: {})
        components, _ = build_bill(private, extra, CONV)
        with pytest.raises(ValueError, match="neither attributed nor in a declared residual part"):
            reconcile(private, components, extra, diag, CONV, {})

    def test_an_attribution_that_stops_matching_the_register_is_refused(self, built):
        private, lines, components, _, _, _ = built
        _, diag = revenue_requirement(_register())
        # Moving one register payment moves its declared residual part by the same amount: still closes.
        shifted = dict(diag, private_wholesale_revenue_gbp=diag["private_wholesale_revenue_gbp"] + 1.0e6)
        _, summary = reconcile(private, components, lines, shifted, CONV, {})
        assert abs(summary["unexplained_gbp"]) < 1.0
        # An attribution computed on different rows from the lines does not.
        broken = dict(diag, legacy_operating_margin_gbp=diag["legacy_operating_margin_gbp"] + 1.0e6)
        with pytest.raises(ValueError, match="register ledger does not close"):
            reconcile(private, components, lines, broken, CONV, {})

    def test_capacity_market_agreements_the_state_does_not_buy_stay_on_the_bill(self):
        from scripts.public_ownership.nationalised_tariff import LINE_CM_NOT_ACQUIRED, capacity_market_not_acquired

        register = _register()
        # Owned and costed: old gas 30 m and new wind 0; CHP (paid as today) 6 m stays with the book.
        cm = {"old_gas": 30.0e6, "chp": 6.0e6}
        register["capacity_market_revenue_gbp"] = register["asset_id"].map(cm).fillna(0.0)
        lines, diag = revenue_requirement(register)
        assert diag["capacity_market_revenue_owned_costed_gbp"] == pytest.approx(30.0e6)
        assert diag["capacity_market_revenue_paid_as_today_gbp"] == pytest.approx(6.0e6)
        private = _private_bill()
        # A 104 m book in report money is 100 m in model money at factor 1.04: 30% is owned, 70% kept.
        kept, detail = capacity_market_not_acquired(private, diag, CONV, 104.0e6)
        assert detail["capacity_market_kept_share"] == pytest.approx(0.70)
        lines[LINE_CM_NOT_ACQUIRED] = kept
        components, summary = build_bill(private, lines, CONV)
        got = components.set_index("component_id")[VALUE]
        assert got[LINE_CM_NOT_ACQUIRED] == pytest.approx(0.70 * 10.0)  # the private line is 10.00
        assert "capacity_market" not in set(components["component_id"])
        recon, recon_summary = reconcile(private, components, lines, diag, CONV, {})
        assert abs(recon_summary["unexplained_gbp"]) < 1.0
        # Without a book total nothing is kept (the old behaviour), never a guess.
        assert capacity_market_not_acquired(private, diag, CONV, None)[0] == 0.0

    def test_reconciliation_refuses_a_tampered_bill(self, built):
        private, lines, components, _, _, _ = built
        _, diag = revenue_requirement(_register())
        broken = components.copy()
        broken.loc[broken["component_id"].eq("tnuos"), VALUE] += 5.0
        with pytest.raises(ValueError, match="does not close"):
            reconcile(private, broken, lines, diag, CONV, {})


class TestVariantsAndYears:
    def test_variants_recompute_legacy_capital_without_the_register(self):
        register = _register()
        valuation = pd.DataFrame({"asset_id": ["old_wind"], "pv_state_gbp": [200.0e6], "pv_market_gbp": [150.0e6],
                                  "public_real_rate": [RATE]})
        acquisition = {"purchase_price_basis": "state_present_value", "thermal_basis": "market_value",
                       "term_years": 30}
        variants = legacy_capital_variants(register, valuation, acquisition, RATE)
        # F29: the workbook's A1/A2 no-support/A3 bases need the valuation's own columns; this one carries none.
        assert set(variants) == {"price_market_value", "thermal_zero", "term_15y"}
        assert variants["price_market_value"]["override"]["old_wind"] == pytest.approx(150.0e6 * CRF30)
        assert variants["price_market_value"]["purchase_price_gbp"] == pytest.approx(150.0e6 + 100.0e6)
        # With them, every acquired row is priced on each basis (old_gas at its own book value).
        full = valuation.assign(net_book_value_gbp=[90.0e6], pv_no_support_gbp=[120.0e6], support_type=["ROC"])
        full = pd.concat([full, pd.DataFrame({"asset_id": ["old_gas"], "pv_market_gbp": [100.0e6],
                                              "net_book_value_gbp": [60.0e6], "pv_no_support_gbp": [100.0e6],
                                              "support_type": ["merchant"]})], ignore_index=True)
        every = legacy_capital_variants(register, full, acquisition, RATE)
        assert every["price_book_value"]["purchase_price_gbp"] == pytest.approx(150.0e6)
        assert every["price_no_support"]["override"]["old_wind"] == pytest.approx(120.0e6 * CRF30)
        assert every["price_ro_pass_through"]["purchase_price_gbp"] == pytest.approx(120.0e6 + 100.0e6)
        assert variants["thermal_zero"]["override"]["old_gas"] == 0.0
        assert variants["thermal_zero"]["purchase_price_gbp"] == pytest.approx(200.0e6)
        assert variants["term_15y"]["override"]["old_wind"] == pytest.approx(200.0e6 * capital_recovery_factor(RATE, 15))
        assert variants["term_15y"]["term_years"] == 15.0
        # Applying a variant moves only the legacy line it targets.
        lines, _ = revenue_requirement(register, legacy_capital_override=variants["price_market_value"]["override"])
        assert lines[LINE_LEGACY_RENEWABLES] == pytest.approx(150.0e6 * CRF30)
        assert lines[LINE_LEGACY_THERMAL] == pytest.approx(100.0e6 * CRF30)

    def test_multi_year_is_streams_less_the_annuity_while_it_runs(self):
        profile = pd.DataFrame({
            "scope_group": ["valued"] * 3 + ["thermal"] * 3 + ["excluded"],
            "carrier": ["wind_offshore"] * 3 + ["CCGT"] * 3 + ["EU_market"],
            "support_type": [""] * 7, "stream": ["net_operating_cash"] * 7,
            "year": [2030, 2035, 2040, 2030, 2035, 2040, 2030],
            "gbp": [90.0e6, 60.0e6, 0.0, -15.0e6, -15.0e6, 0.0, 999.0e6],
        })
        lines = {LINE_LEGACY_RENEWABLES: 10.0e6, LINE_LEGACY_THERMAL: 5.0e6}
        frame = multi_year(profile, lines, CONV, term_years=30, model_year=2030, years=(2030, 2035, 2040, 2060),
                           bill_delta_model_year_gbp_per_hh=-100.0)
        by = frame.set_index("year")
        assert by.loc[2030, "legacy_component_system_gbp"] == pytest.approx(75.0e6 - 15.0e6)
        assert by.loc[2035, "legacy_component_system_gbp"] == pytest.approx(45.0e6 - 15.0e6)
        assert by.loc[2040, "legacy_component_system_gbp"] == pytest.approx(0.0 - 15.0e6)
        assert by.loc[2060, "acquisition_annuity_gbp"] == 0.0  # the 30-year annuity has ended
        # The bill change moves by the change in the legacy component only; 2030 equals the model-year delta.
        assert by.loc[2030, "bill_delta_gbp_per_3100kwh_ex_vat"] == pytest.approx(-100.0)
        shift = household(30.0e6, CONV["final_demand_mwh"], CONV["factor"], CONV["mwh"])
        assert by.loc[2035, "bill_delta_gbp_per_3100kwh_ex_vat"] == pytest.approx(-100.0 + shift)
        assert {"15-year average", "30-year average"} <= set(frame["year"].astype(str))

    def test_conversion_reads_the_recovery_denominators(self):
        assumptions = {"currency_rebasing": {"factor": 1.04}, "benchmark": {"annual_consumption_kwh": 3100, "vat_rate": 0.05}}
        recovery = {"final_demand_mwh": {"value": 300.0e6}, "ro_obligation_eligible_supply_mwh": {"value": 285.0e6}}
        conv = conversion(assumptions, recovery, {"cfd_recovery_denominator_mwh": 280.0e6})
        assert conv["cfd_eligible_mwh"] == 280.0e6 and conv["ro_eligible_mwh"] == 285.0e6
        assert conversion(assumptions, recovery)["cfd_eligible_mwh"] == 285.0e6  # falls back to RO-eligible supply


class TestLikeForLikeAllocation:
    """Audit 26 Sep 2026: the tariff recovers each new line by the rules of the lines it replaces."""

    CONV_CM = {**CONV, "cm_household_allocation_factor": 1.3}

    def test_each_replaced_line_is_costed_at_its_own_rule(self):
        from scripts.public_ownership.nationalised_tariff import recovery_rates

        alloc = recovery_rates(_private_bill(), self.CONV_CM)
        flat = household(1.0, 300.0e6, 1.04, 3.1)
        rates = alloc["replaced_rates"]
        assert alloc["flat_rate"] == pytest.approx(flat)
        assert rates["cfd"] == pytest.approx(flat * 300.0 / 280.0)
        assert rates["allocated_ccs"] == pytest.approx(flat * 300.0 / 280.0)
        assert rates["renewables_obligation"] == pytest.approx(flat * 300.0 / 285.0)
        assert rates["capacity_market"] == pytest.approx(flat * 1.3)
        # The shaping line moves cost between consumers and costs nothing in aggregate.
        assert rates["shaping_adjustment"] is None and alloc["replaced_system_gbp"]["shaping_adjustment"] == 0.0
        hh = sum(alloc["replaced_gbp_per_3100kwh_ex_vat"].values())
        system = sum(alloc["replaced_system_gbp"].values())
        assert alloc["pool_rate"] == pytest.approx(hh / system)
        # Households carry more of the replaced lines than a flat share: EII exemption and peak factor.
        assert alloc["pool_rate"] > flat

    def test_replacing_cost_with_equal_cost_leaves_the_household_bill_unchanged(self):
        """The property the audit asked for: the comparison is a saving, not a reallocation."""
        from scripts.public_ownership.nationalised_tariff import recovery_rates

        private = _private_bill()
        alloc = recovery_rates(private, self.CONV_CM)
        same_cost = {LINE_NEW_BUILD: sum(alloc["replaced_system_gbp"].values())}
        _, like_for_like = build_bill(private, same_cost, self.CONV_CM, allocation=alloc)
        _, flat = build_bill(private, same_cost, self.CONV_CM)
        assert like_for_like["delta_ex_vat_gbp"] == pytest.approx(0.0, abs=1e-9)
        # Flat recovery would have shown households a "saving" that is only cost moved onto non-domestic users.
        assert flat["delta_ex_vat_gbp"] < -1.0
        assert like_for_like["tariff_allocation"]["delta_inc_vat_gbp_if_flat"] == pytest.approx(flat["delta_inc_vat_gbp"])

    def test_kept_parts_go_back_at_their_own_rule_and_the_ledgers_close(self):
        from scripts.public_ownership.nationalised_tariff import (
            LINE_CM_NOT_ACQUIRED, capacity_market_not_acquired, recovery_rates,
        )

        register = _register()
        register["capacity_market_revenue_gbp"] = register["asset_id"].map({"old_gas": 30.0e6}).fillna(0.0)
        lines, diag = revenue_requirement(register, interconnector_retrading_gbp=3.0e6)
        private = _private_bill()
        kept, _ = capacity_market_not_acquired(private, diag, self.CONV_CM, 104.0e6)
        lines[LINE_CM_NOT_ACQUIRED] = kept
        alloc = recovery_rates(private, self.CONV_CM, {LINE_CM_NOT_ACQUIRED: kept, LINE_IC_RETRADING: 3.0e6})
        components, summary = build_bill(private, lines, self.CONV_CM, allocation=alloc)
        got = components.set_index("component_id")[VALUE]
        flat = household(1.0, 300.0e6, 1.04, 3.1)
        # The kept Capacity Market agreements reproduce their share of the private line, peak factor and all.
        assert got[LINE_CM_NOT_ACQUIRED] == pytest.approx(0.70 * 10.0)
        assert kept == pytest.approx(0.70 * 10.0 / (flat * 1.3))
        # Re-trading is part of the constraint line: final demand, flat.
        assert got[LINE_IC_RETRADING] == pytest.approx(3.0e6 * flat)
        # The cost of owned plant takes the pool rate.
        assert got[LINE_NEW_BUILD] == pytest.approx(lines[LINE_NEW_BUILD] * alloc["pool_rate"])
        detail = {"cfd_economic_curtailment_payment_gbp": 5.0e6, "interconnector_retrading_cost_gbp": 3.0e6}
        recon, recon_summary = reconcile(private, components, lines, diag, self.CONV_CM, detail, allocation=alloc)
        change = summary["delta_ex_vat_gbp"]
        assert recon.loc[recon["ledger"].eq("bill"), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(change)
        assert recon.loc[recon["ledger"].eq("register"), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(change)
        assert abs(recon_summary["unexplained_gbp"]) < 1.0
        # The Capacity Market line stands for its book, not its household value over a flat share.
        cm = recon.loc[recon["component_id"].eq("capacity_market")].iloc[0]
        assert cm["system_gbp_model_year"] == pytest.approx(-10.0 / (flat * 1.3))

    def test_conversion_carries_the_capacity_market_peak_factor(self):
        assumptions = {"currency_rebasing": {"factor": 1.04}, "benchmark": {"annual_consumption_kwh": 3100, "vat_rate": 0.05},
                       "sourced_externals": {"capacity_market": {"derivation": {"household_allocation_factor": 1.348308}}}}
        recovery = {"final_demand_mwh": {"value": 300.0e6}, "ro_obligation_eligible_supply_mwh": {"value": 285.0e6}}
        assert conversion(assumptions, recovery)["cm_household_allocation_factor"] == pytest.approx(1.348308)


class TestInterconnectorAcquisition:
    """Item D3: every tariff case buys the GB halves; foreign halves paid their rent; both ledgers close."""

    RENT = {"capped": 200.0e6, "merchant": 40.0e6, "irish": 30.0e6}
    SHARES = {"capped": 0.5, "merchant": 0.5, "irish": 0.0}
    # F28 (f): each cable's remaining years (interconnector_remaining_years); 25 for every half here.
    YEARS = {"capped": 25.0, "merchant": 25.0, "new": 25.0}

    def _settings(self, **kw):
        from scripts.public_ownership.nationalised_tariff import interconnector_acquisition_settings
        return interconnector_acquisition_settings(kw)

    def test_acquired_gb_halves_accrue_and_are_annuitised(self):
        from scripts.public_ownership.nationalised_tariff import (
            LINE_IC_ACQUISITION, LINE_IC_FOREIGN_RENT, interconnector_lines,
        )
        settings = self._settings(owner_real_rate=0.067)
        lines, detail = interconnector_lines(self.RENT, self.SHARES, settings=settings, public_rate=RATE, term_years=30, remaining_years=self.YEARS)
        assert set(lines) == {LINE_IC_FOREIGN_RENT, LINE_IC_ACQUISITION}  # no private GB-half path
        assert lines[LINE_IC_FOREIGN_RENT] == pytest.approx(100.0e6 + 20.0e6 + 30.0e6)
        assert detail["interconnector_gb_halves_acquired"] is True
        # F29: the owners' value after corporation tax (workbook v2.0's Approach 2, 25%).
        factor = 0.75 * (1 - 1.067 ** -25) / 0.067
        assert detail["interconnector_acquisition_price_gbp"] == pytest.approx(120.0e6 * factor)
        assert lines[LINE_IC_ACQUISITION] == pytest.approx(120.0e6 * factor * CRF30)
        assert detail["interconnector_gb_half_rent_to_tariff_gbp"] == pytest.approx(120.0e6)

    def test_explicit_price_and_refusals(self):
        from scripts.public_ownership.nationalised_tariff import LINE_IC_ACQUISITION, interconnector_lines
        settings = self._settings(purchase_price_basis="explicit", price_gbp=1.0e9)
        lines, detail = interconnector_lines(self.RENT, self.SHARES, settings=settings, public_rate=RATE, term_years=30, remaining_years=self.YEARS)
        assert detail["interconnector_acquisition_price_gbp"] == pytest.approx(1.0e9)
        assert lines[LINE_IC_ACQUISITION] == pytest.approx(1.0e9 * CRF30)
        with pytest.raises(ValueError, match="no declared GB share"):
            interconnector_lines({"unknown": 1.0}, self.SHARES, settings=self._settings(), public_rate=RATE,
                                 term_years=30, remaining_years=self.YEARS)
        with pytest.raises(ValueError, match="unknown keys"):
            self._settings(owner_rate=0.05)

    def test_a_cable_built_in_the_model_year_is_bought_at_cost_not_at_market_value(self):
        """15 Oct 2026: MaresConnect (2030) is new build; its GB half is priced at its floor's PV at the public rate."""
        from scripts.public_ownership.nationalised_tariff import LINE_IC_ACQUISITION, interconnector_lines

        rent = {**self.RENT, "new": 80.0e6}
        shares = {**self.SHARES, "new": 0.5}
        settings = self._settings(owner_real_rate=0.067)
        lines, detail = interconnector_lines(rent, shares, settings=settings, public_rate=RATE, term_years=30, remaining_years=self.YEARS,
                                             new_build_floor_gbp={"new": 30.0e6, "not_built": 9.0e6})
        owner = 0.75 * (1 - 1.067 ** -25) / 0.067  # F29: after corporation tax
        public = (1 - (1 + RATE) ** -25) / RATE
        price = detail["interconnector_acquisition_price_by_cable_gbp"]
        assert price["capped"] == pytest.approx(100.0e6 * owner)           # existing halves: market value
        assert price["new"] == pytest.approx(30.0e6 * public)              # new build: at cost, not 40 m x owner
        assert set(detail["interconnector_new_build_at_cost"]) == {"new"}  # a cable with no rent is not built
        assert lines[LINE_IC_ACQUISITION] == pytest.approx((120.0e6 * owner + 30.0e6 * public) * CRF30)
        # Its rent still accrues to the tariff: the state owns the GB half either way.
        assert detail["interconnector_gb_half_rent_to_tariff_gbp"] == pytest.approx(160.0e6)
        # An explicit price is for the existing halves only.
        explicit = self._settings(purchase_price_basis="explicit", price_gbp=1.0e9)
        _, detail = interconnector_lines(rent, shares, settings=explicit, public_rate=RATE, term_years=30, remaining_years=self.YEARS,
                                         new_build_floor_gbp={"new": 30.0e6})
        assert detail["interconnector_acquisition_price_gbp"] == pytest.approx(1.0e9 + 30.0e6 * public)
        with pytest.raises(ValueError, match="need a GB half"):
            interconnector_lines(rent, shares, settings=settings, public_rate=RATE, term_years=30, remaining_years=self.YEARS,
                                 new_build_floor_gbp={"irish": 1.0e6})

    def test_new_build_floors_come_from_the_2030_regime_table(self):
        from scripts.market.fti_settlement import load_interconnector_rule
        from scripts.public_ownership.nationalised_tariff import interconnector_new_build_floors

        rule = load_interconnector_rule("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
        floors = interconnector_new_build_floors(rule)
        assert set(floors) == {"MaresConnect"}
        # 27.90 m (2020/21, GB share) RPI-indexed to 2024 prices, as the bill indexes it.
        assert floors["MaresConnect"] == pytest.approx(27.90e6 * 386.7 / 294.17)
        bad = {**rule, "cables": {"X": {"regime": "exempt_merchant", "public_ownership_treatment": "new_build_at_cost"}}}
        with pytest.raises(ValueError, match="cap-and-floor"):
            interconnector_new_build_floors(bad)
        bad = {**rule, "cables": {"X": {"regime": "cap_and_floor", "public_ownership_treatment": "gift"}}}
        with pytest.raises(ValueError, match="public_ownership_treatment"):
            interconnector_new_build_floors(bad)

    def test_cm_share_is_capacity_weighted_and_leaves_the_kept_line(self):
        from scripts.public_ownership.nationalised_tariff import (
            capacity_market_not_acquired, interconnector_cm_acquired_share,
        )
        share = interconnector_cm_acquired_share({"capped": 1000.0, "irish": 500.0, "merchant": 500.0}, self.SHARES)
        assert share == pytest.approx((500.0 + 250.0) / 2000.0)
        register = _register()
        register["capacity_market_revenue_gbp"] = register["asset_id"].map({"old_gas": 30.0e6}).fillna(0.0)
        _, diag = revenue_requirement(register)
        kept, detail = capacity_market_not_acquired(_private_bill(), diag, CONV, 104.0e6,
                                                    interconnector_acquired_gbp=10.4e6)
        # 30 m acquired fleet + 10 m acquired GB halves of a 100 m book: 60% kept.
        assert detail["capacity_market_kept_share"] == pytest.approx(0.60)
        assert detail["capacity_market_interconnector_gb_halves_acquired_model_year_gbp"] == pytest.approx(10.0e6)

    def test_rent_lines_need_the_foreign_price_basis_and_both_ledgers_close(self):
        from scripts.public_ownership.nationalised_tariff import (
            INTERCONNECTOR_PASS_THROUGH_LINE, LINE_IC_ACQUISITION, LINE_IC_FOREIGN_RENT, dropped_lines_for,
            interconnector_lines,
        )
        register = _register()
        extra_row = _private_bill().iloc[[0]].assign(component_id=INTERCONNECTOR_PASS_THROUGH_LINE, **{VALUE: -1.18})
        private = pd.concat([_private_bill(), extra_row], ignore_index=True)
        detail = {"cfd_economic_curtailment_payment_gbp": 5.0e6, "interconnector_retrading_cost_gbp": 3.0e6}
        ic, ic_detail = interconnector_lines(self.RENT, self.SHARES, settings=self._settings(), public_rate=RATE,
                                             term_years=30, remaining_years=self.YEARS)
        with pytest.raises(ValueError, match="measured link trade"):
            revenue_requirement(register, interconnector=ic, interconnector_detail=ic_detail)
        lines, diag = revenue_requirement(register, interconnector_trade_gbp=15.0e6, interconnector=ic,
                                          interconnector_detail=ic_detail)
        # The acquired GB halves leave their regimes: the pass-through goes.
        dropped = dropped_lines_for(diag)
        assert INTERCONNECTOR_PASS_THROUGH_LINE in dropped
        comp, summary = build_bill(private, lines, CONV, dropped_lines=dropped)
        recon, recon_summary = reconcile(private, comp, lines, diag, CONV, detail, dropped_lines=dropped)
        for ledger in ("bill", "register"):
            assert recon.loc[recon["ledger"].eq(ledger), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(
                summary["delta_ex_vat_gbp"])
        assert abs(recon_summary["unexplained_gbp"]) < 1.0
        attributed = recon_summary["attribution_gbp"]
        assert attributed[next(k for k in attributed if k.startswith("interconnector congestion rent paid to "
                                                                      "foreign owners"))] == pytest.approx(
            lines[LINE_IC_FOREIGN_RENT])
        assert attributed[next(k for k in attributed if "acquisition annuity" in k and "interconnector" in k)] == \
            pytest.approx(lines[LINE_IC_ACQUISITION])

    def test_the_ro_ledger_step_is_labelled_as_volume(self):
        """Review B16: the RO step is a volume gap, not recycle or headroom value."""
        register, private = _register(), _private_bill()
        lines, diag = revenue_requirement(register)
        comp, _ = build_bill(private, lines, CONV)
        _, summary = reconcile(private, comp, lines, diag, CONV, {})
        keys = list(summary["not_register_payments_gbp"])
        assert any(k.startswith("RO line above the register's certificate income (a volume gap") for k in keys)
        assert not any("recycle and headroom" in k for k in keys)


class TestFixedOmSplit:
    """Item D4(a): the register's fixed O&M carries generator TNUoS and insurance; the tariff shows them apart."""

    def _register(self):
        from scripts.public_ownership.nationalised_tariff import FIXED_OM_COMPONENTS  # noqa: F401
        register = _register()
        tnuos = {"old_wind": 3.0e6, "old_gas": 1.0e6, "new_wind": 4.0e6}
        insurance = {"old_wind": 1.0e6, "old_gas": 0.5e6, "new_wind": 2.0e6}
        register["annual_generator_tnuos_gbp"] = register["asset_id"].map(tnuos).fillna(0.0)
        register["annual_insurance_gbp"] = register["asset_id"].map(insurance).fillna(0.0)
        register["annual_site_fixed_om_gbp"] = register["annual_fixed_om_gbp"]
        extra = register["annual_generator_tnuos_gbp"] + register["annual_insurance_gbp"]
        # As calculate_financing_scenarios does: folded into fixed O&M and required revenue.
        register["annual_fixed_om_gbp"] = register["annual_fixed_om_gbp"] + extra
        register["annual_required_revenue_gbp"] = register["annual_required_revenue_gbp"] + extra
        return register

    def test_the_three_lines_add_up_to_the_register_fixed_om_and_the_ledgers_close(self):
        from scripts.public_ownership.nationalised_tariff import LINE_GENERATOR_TNUOS, LINE_INSURANCE
        register, private = self._register(), _private_bill()
        lines, diag = revenue_requirement(register)
        assert lines[LINE_FIXED_OM] == pytest.approx(30.0e6)
        assert lines[LINE_GENERATOR_TNUOS] == pytest.approx(8.0e6)
        assert lines[LINE_INSURANCE] == pytest.approx(3.5e6)
        comp, summary = build_bill(private, lines, CONV)
        got = comp.set_index("component_id")[VALUE]
        for line in (LINE_FIXED_OM, LINE_GENERATOR_TNUOS, LINE_INSURANCE):
            assert got[line] == pytest.approx(household(lines[line], CONV["final_demand_mwh"], CONV["factor"], CONV["mwh"]))
        recon, recon_summary = reconcile(private, comp, lines, diag, CONV, {})
        for ledger in ("bill", "register"):
            assert recon.loc[recon["ledger"].eq(ledger), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(
                summary["delta_ex_vat_gbp"])
        assert abs(recon_summary["unexplained_gbp"]) < 1.0

    def test_components_that_do_not_add_up_are_refused_and_an_old_register_is_all_site_om(self):
        from scripts.public_ownership.nationalised_tariff import LINE_GENERATOR_TNUOS
        register = self._register()
        register.loc[register["asset_id"].eq("new_wind"), "annual_insurance_gbp"] += 5.0e6
        with pytest.raises(ValueError, match="do not add up"):
            revenue_requirement(register)
        lines, _ = revenue_requirement(_register())
        assert lines[LINE_FIXED_OM] == pytest.approx(30.0e6) and lines[LINE_GENERATOR_TNUOS] == 0.0


class TestMeteredSupplyDenominators:
    """Item D1 follow-up: the tariff inverts each replaced line at the bill's own denominator."""

    def test_capacity_market_is_inverted_over_the_gsp_volume(self):
        from scripts.public_ownership.nationalised_tariff import _rule_rate
        assumptions = {"currency_rebasing": {"factor": 1.04}, "benchmark": {"annual_consumption_kwh": 3100, "vat_rate": 0.05},
                       "sourced_externals": {"capacity_market": {"derivation": {
                           "household_allocation_factor": 1.348, "recovery_denominator": "gsp_chargeable_volume_mwh"}}}}
        recovery = {"final_demand_mwh": {"value": 295.79e6}, "ro_obligation_eligible_supply_mwh": {"value": 282.93e6},
                    "metered_supply_basis": {"gsp_chargeable_volume_mwh": {"value": 322.98e6}}}
        conv = conversion(assumptions, recovery)
        # F19 integration: the D1 conversion names the Capacity Market's volume cm_volume_mwh.
        assert conv["cm_volume_mwh"] == pytest.approx(322.98e6)
        # The bill: book / GSP volume x 3.1 x factor; the inversion must use the same volume.
        assert _rule_rate(conv, "capacity_market") == pytest.approx(1.04 * 3.1 / 322.98e6 * 1.348)
        assert _rule_rate(conv, "final_demand") == pytest.approx(1.04 * 3.1 / 295.79e6)
        del recovery["metered_supply_basis"]
        with pytest.raises(ValueError, match="gsp_chargeable_volume_mwh"):
            conversion(assumptions, recovery)

    def test_imbalance_goes_like_hedging_and_voltage_constraints_stay(self):
        from scripts.public_ownership.nationalised_tariff import HEDGING_LINES, KEPT_DECLARED
        assert "procurement_imbalance" in HEDGING_LINES and "procurement_imbalance" in REPLACED_LINES
        assert "bsuos_voltage_and_stability_constraints" in KEPT_DECLARED
        register = _register()
        extra = _private_bill().iloc[[0, 0]].assign(
            component_id=["procurement_imbalance", "bsuos_voltage_and_stability_constraints"], **{VALUE: [2.0, 6.0]})
        private = pd.concat([_private_bill(), extra], ignore_index=True)
        lines, diag = revenue_requirement(register)
        comp, summary = build_bill(private, lines, CONV)
        ids = set(comp["component_id"])
        assert "procurement_imbalance" not in ids and "bsuos_voltage_and_stability_constraints" in ids
        recon, recon_summary = reconcile(private, comp, lines, diag, CONV, {})
        for ledger in ("bill", "register"):
            assert recon.loc[recon["ledger"].eq(ledger), "gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(
                summary["delta_ex_vat_gbp"])
        assert abs(recon_summary["unexplained_gbp"]) < 1.0


def test_the_supplier_margin_follows_the_tariffs_own_costs():
    """F27 item 10 (23 Oct 2026): EBIT and headroom at Ofgem's percentages of the tariff's costs."""
    import pandas as pd

    from scripts.public_ownership.nationalised_tariff import VALUE, supplier_margin_adjustment

    rule = {"ebit_variable_share": 0.02, "headroom_share": 0.01,
            "network_group": {"view": "ofgem", "group": "Network costs"}}
    # The private bill: costs 900 (of which networks 200), EBIT 0.02 x 900 + 10 = 28, headroom 0.01 x (700 + 28).
    old_ebit, old_headroom = 0.02 * 900 + 10.0, 0.01 * (700 + 28.0)
    margin = {"rule": rule, "ebit_gbp": old_ebit, "headroom_gbp": old_headroom, "levelisation_gbp": 5.0,
              "ebit_fixed_gbp": 10.0}
    tariff = pd.DataFrame({"component_id": ["duos", "nationalised_generation_cost", "supplier_costs_and_margin"],
                           VALUE: [200.0, 500.0, 100.0 + 5.0 + old_ebit + old_headroom]})
    # Tariff costs: 200 + 500 + 100 = 800; EBIT 0.02 x 800 + 10 = 26, headroom 0.01 x (600 + 26).
    assert supplier_margin_adjustment(tariff, margin) == pytest.approx((26.0 + 6.26) - (old_ebit + old_headroom))
    assert supplier_margin_adjustment(tariff, {}) == 0.0


def test_capacity_added_after_the_purchase_is_built_by_the_state_at_cost(tmp_path):
    """D2 (user, 14 Nov 2026): a later year's GB halves beyond the purchase's cable ratings are state new build, at the
    cost-based floor per GB-half MW of the cables the purchase built at cost (MaresConnect in 2030)."""
    from scripts.public_ownership.nationalised_tariff import interconnector_post_acquisition_new_build

    comp = tmp_path / "comp.csv"
    pd.DataFrame({"name": ["A", "M", "X"], "action": ["keep", "add", "exclude"], "cp30_mw": [1000.0, 500.0, 0.0]}
                 ).to_csv(comp, index=False)
    shares = {"A": 0.5, "M": 0.5, "X": 0.0}
    floors = {"M": 25.0e6}                                 # GBP 25 m for M's 250 MW GB half: GBP 100k per MW
    out = interconnector_post_acquisition_new_build({"A": 1600.0, "M": 500.0, "X": 300.0}, shares, floors,
                                                    composition=comp)
    assert out["gb_half_mw_by_cable"] == {"A": pytest.approx(300.0)}       # 600 MW added, its GB half
    assert out["annual_gbp"] == pytest.approx(300.0 * 100_000.0)
    assert interconnector_post_acquisition_new_build({"A": 900.0}, shares, floors, composition=comp)["annual_gbp"] == 0.0
