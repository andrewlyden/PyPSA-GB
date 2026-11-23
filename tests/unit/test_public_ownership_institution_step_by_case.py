"""P3's sensitivities on central dispatch (user decision, 24 Oct 2026).

Only P3's central case is solved under central dispatch (I3); every other tariff case that owns
the same fleet is reported on I3 as its own I1 tariff plus the measured institution step, an
additive approximation. Small hand-checkable inputs: the step re-derived on another case's I1
lines equals the measured one; the measured case is its twin plus the step exactly; a case
priced at the market bill moves the Treasury's margin, not the bill; a case owning another
fleet is excluded; every derived row closes the three-party identity; and the market year is
read on the turn-down basis its tariff reads only where a re-shared unit has a rate.
"""

import copy

import pandas as pd
import pytest

from scripts.public_ownership import investor_ledger as inv
from scripts.public_ownership import nationalised_institutions as ni
from scripts.public_ownership.exchequer_ledger import (
    I3_APPROX_PREFIX, fill_three_party, household_totals, household_to_system, institution_step_columns, ledger_row,
)
from scripts.public_ownership.nationalised_tariff import (
    LINE_FUEL, LINE_IC_RETRADING, LINE_IMPORTS, LINE_LEGACY_RENEWABLES, LINE_NEW_BUILD, LINE_REALTIME,
    LINE_TREASURY_MARGIN, LINE_VARIABLE_OM,
)

M = 1.0e6
CONV = {"factor": 1.04, "final_demand_mwh": 300.0e6, "mwh": 3.1, "vat_rate": 0.05, "domestic_mwh": 100.0e6}
MARKET = {"fuel_carbon_gbp": 5.0e9, "fuel_gbp": 3.6e9, "carbon_gbp": 1.4e9, "variable_om_gbp": 0.5e9,
          "interconnector_cost_gbp": 0.7e9, "energy_mwh": 300.0e6}
CENTRAL = {"fuel_carbon_gbp": 4.95e9, "fuel_gbp": 3.57e9, "carbon_gbp": 1.38e9, "variable_om_gbp": 0.49e9,
           "interconnector_cost_gbp": 0.2e9, "energy_mwh": 300.0e6}
FLEET = {"owned_costed_assets": 100, "owned_costed_capacity_mw": 1.0e5, "paid_as_today_capacity_mw": 1.0e4,
         "paid_as_today_carriers": ["CHP", "oil"]}
PRIVATE_EX = 1000.0


def _report(case_id, *, legacy, ex_vat, pricing="cost", fleet=None, margin=0.0):
    lines = {LINE_LEGACY_RENEWABLES: legacy, LINE_NEW_BUILD: 5.0e9, LINE_FUEL: 4.5e9, LINE_VARIABLE_OM: 0.6e9,
             LINE_IMPORTS: 0.7e9, LINE_REALTIME: 0.0, LINE_IC_RETRADING: 0.0}
    if margin:
        lines[LINE_TREASURY_MARGIN] = margin
    return {"case_id": case_id, "levers": {"tariff_pricing": pricing}, "conversion": dict(CONV), "allocation": None,
            "revenue_requirement_gbp": lines, "diagnostics": dict(fleet or FLEET),
            "bill": {"ex_vat_gbp": ex_vat, "private_ex_vat_gbp": PRIVATE_EX, "private_total_gbp_inc_vat": PRIVATE_EX * 1.05,
                     "delta_inc_vat_gbp": (ex_vat - PRIVATE_EX) * 1.05}}


def _tariff():
    return {"cases": {
        "nat": _report("nat", legacy=3.0e9, ex_vat=900.0),
        # The purchase annuity over 15 years: a dearer I1 bill, the same fleet.
        "nat_15y": _report("nat_15y", legacy=4.5e9, ex_vat=960.0),
        # tariff_pricing market_bill: the comparator's bill, the Treasury keeps the margin.
        "nat_tf": _report("nat_tf", legacy=3.0e9, ex_vat=PRIVATE_EX, pricing="market_bill", margin=1.0e9),
        # A step from P2 towards P3: less of the fleet owned.
        "nat_less": _report("nat_less", legacy=1.0e9, ex_vat=950.0, fleet=dict(FLEET, owned_costed_assets=80)),
    }}


def _institution(tariff):
    _, summary = ni.compare(tariff["cases"]["nat"], MARKET, CENTRAL, realtime_correction_gbp=20 * M,
                            forecast_error_charge_gbp=60 * M, replace_realtime_correction=True)
    summary.update(institution_case_id="nat_central", i1_case_id="nat")
    return summary


def _carbon(measurement):
    return {"carbon_receipts_gbp": measurement["carbon_gbp"], "carbon_uk_ets_gbp": measurement["carbon_gbp"],
            "carbon_price_support_gbp": 0.0, "chp_heat_credit_carbon_gbp": 0.0}


CARBON = {"market": _carbon(MARKET), "central": _carbon(CENTRAL)}
ZERO = {"corporation_tax_forgone_gbp": 0.0, "network_tax_forgone_gbp": 0.0, "state_deficits_funded_gbp": 0.0,
        "state_surplus_retained_gbp": 0.0, "exchequer_residual_gbp": 0.0, "financing_saving_gbp": 0.0,
        "financing_saving_tax_transfer_gbp": 0.0, "levers_for_flows": {}}


def _ledger(tariff, institution):
    """Rows as the ledger builds them: I1 tariff cases, the measured I3 case on its twin plus the step."""
    cases = {"private": {"case_id": "private", "stage": "market_bill", "bill_ex_vat": PRIVATE_EX,
                         "bill_inc_vat": PRIVATE_EX * 1.05, **CARBON["market"], **ZERO}}
    parties = {"private": {"bill_change_system_gbp": 0.0, "investor_parts_gbp": {}, "resource_parts_gbp": {}}}
    for case_id, report in tariff["cases"].items():
        margin = report["revenue_requirement_gbp"].get(LINE_TREASURY_MARGIN, 0.0)
        cases[case_id] = {"case_id": case_id, "stage": "acquisition_tariff", "bill_ex_vat": report["bill"]["ex_vat_gbp"],
                          "bill_inc_vat": report["bill"]["ex_vat_gbp"] * 1.05, **CARBON["market"], **ZERO,
                          "treasury_margin_gbp": margin}
        bill_change = household_to_system(report["bill"]["ex_vat_gbp"] - PRIVATE_EX, CONV)
        # Investors lose 50 m; the resource saving is what closes the identity.
        parties[case_id] = {"bill_change_system_gbp": bill_change, "investor_parts_gbp": {"owners": -50 * M},
                            "resource_parts_gbp": {"resource": -bill_change + margin - 50 * M}}
    cases["nat_central"] = {**cases["nat"], "case_id": "nat_central", "stage": "institution_comparison",
                            "bill_ex_vat": institution["i3_ex_vat_gbp"], "bill_inc_vat": institution["i3_inc_vat_gbp"],
                            **CARBON["central"], "institution_twin": "nat"}
    step = inv.institution_step_parties(institution, carbon_change_gbp=CENTRAL["carbon_gbp"] - MARKET["carbon_gbp"])
    parties["nat_central"] = inv.combine(parties["nat"], step)
    rows = [ledger_row(entry, cases["private"], CONV) for entry in cases.values()]
    for row in rows:
        fill_three_party(row, parties[row["case_id"]], None, two_volume=False, investor={}, conv=CONV)
    household_totals(rows, CONV, catalogue={}, incidence=None, contested=None, register=None, tariff=tariff,
                     network_lines=None)
    return rows, cases, {"parties": parties, "memos": {}, "two_volume": False}


def _apply(tariff=None, institution=None, mutate=None):
    tariff = tariff or _tariff()
    institution = institution or _institution(tariff)
    rows, cases, kept = _ledger(tariff, institution)
    if mutate:
        mutate({r["case_id"]: r for r in rows})
    block = institution_step_columns(rows, cases, tariff=tariff, institution=institution, carbon=CARBON, conv=CONV,
                                     kept=kept, investor={}, incidence=None)
    return {r["case_id"]: r for r in rows}, block, institution


class TestDerivedStep:
    def test_the_twin_reproduces_the_measured_step_and_the_purchase_terms_do_not_move_it(self):
        tariff = _tariff()
        institution = _institution(tariff)
        twin = ni.derive_case_step(tariff["cases"]["nat"], institution)
        assert twin["i3_ex_vat_gbp"] == pytest.approx(institution["i3_ex_vat_gbp"], abs=1e-12)
        assert twin["institution_change_system_gbp"] == pytest.approx(institution["institution_change_system_gbp"])
        other = ni.derive_case_step(tariff["cases"]["nat_15y"], institution)
        assert other["institution_change_system_gbp"] == pytest.approx(twin["institution_change_system_gbp"])
        assert other["i3_ex_vat_gbp"] - 960.0 == pytest.approx(twin["i3_ex_vat_gbp"] - 900.0)
        assert other["approximation"] == ni.DERIVED_STEP_LABEL

    def test_a_summary_without_the_flag_reads_it_off_what_was_removed(self):
        tariff = _tariff()
        institution = _institution(tariff)
        old = copy.deepcopy(institution)
        del old["audit"]["realtime_correction_replaced"]
        assert (ni.derive_case_step(tariff["cases"]["nat"], old)["i3_ex_vat_gbp"]
                == pytest.approx(institution["i3_ex_vat_gbp"], abs=1e-12))
        del old["audit"]["realtime_correction_model_own_replaced_gbp"]
        with pytest.raises(ValueError, match="predates F19"):
            ni.derive_case_step(tariff["cases"]["nat"], old)

    def test_a_case_owning_another_fleet_is_not_the_same_fleet(self):
        tariff = _tariff()
        same, why = ni.same_fleet_as(tariff["cases"]["nat_less"], tariff["cases"]["nat"])
        assert not same and "owns another fleet" in why
        assert ni.same_fleet_as(tariff["cases"]["nat_15y"], tariff["cases"]["nat"]) == (True, None)


class TestLedgerColumns:
    def test_the_twin_on_i3_is_the_measured_case(self):
        rows, block, _ = _apply()
        twin, measured = rows["nat"], rows["nat_central"]
        for column in ("bill_inc_vat", "household_total_gbp_inc_vat", "consumer_change_vs_private_gbp",
                       "exchequer_net_vs_private_gbp", "investor_change_vs_private_gbp", "resource_saving_vs_private_gbp"):
            assert twin[I3_APPROX_PREFIX + column] == pytest.approx(measured[column], abs=1e-6)
        assert block["twin_case_id"] == "nat" and block["institution_case_id"] == "nat_central"
        # The measured case itself carries no derived columns: it is not derived again.
        assert I3_APPROX_PREFIX + "bill_inc_vat" not in measured

    def test_a_sensitivity_gets_the_same_step_and_closes_the_identity(self):
        rows, block, institution = _apply()
        twin, case = rows["nat"], rows["nat_15y"]
        step = twin[I3_APPROX_PREFIX + "bill_inc_vat"] - twin["bill_inc_vat"]
        assert case[I3_APPROX_PREFIX + "bill_inc_vat"] - case["bill_inc_vat"] == pytest.approx(step)
        assert step == pytest.approx(institution["institution_change_inc_vat_gbp"])
        assert abs(case[I3_APPROX_PREFIX + "three_party_identity_residual_gbp"]) <= inv.IDENTITY_TOLERANCE_GBP
        # Investors do not move; the resource saving moves by the step net of the carbon receipt.
        assert case[I3_APPROX_PREFIX + "investor_change_vs_private_gbp"] == pytest.approx(case["investor_change_vs_private_gbp"])
        carbon = CENTRAL["carbon_gbp"] - MARKET["carbon_gbp"]
        assert (case[I3_APPROX_PREFIX + "resource_saving_vs_private_gbp"] - case["resource_saving_vs_private_gbp"]
                == pytest.approx(-institution["institution_change_system_gbp"] + carbon))
        assert case[I3_APPROX_PREFIX + "basis"].startswith("central dispatch, additive approximation")
        assert block["applied"]["nat_15y"]["step_bill_gbp_inc_vat"] == pytest.approx(step)

    def test_market_bill_pricing_moves_the_margin_not_the_bill(self):
        rows, block, institution = _apply()
        case = rows["nat_tf"]
        assert case[I3_APPROX_PREFIX + "bill_inc_vat"] == pytest.approx(case["bill_inc_vat"])
        carbon = CENTRAL["carbon_gbp"] - MARKET["carbon_gbp"]
        assert (case[I3_APPROX_PREFIX + "exchequer_net_vs_private_gbp"] - case["exchequer_net_vs_private_gbp"]
                == pytest.approx(-institution["institution_change_system_gbp"] + carbon))
        assert abs(case[I3_APPROX_PREFIX + "three_party_identity_residual_gbp"]) <= inv.IDENTITY_TOLERANCE_GBP
        assert "market_bill" in case[I3_APPROX_PREFIX + "basis"] and block["applied"]["nat_tf"]["market_bill_pricing"]

    def test_another_fleet_is_excluded_and_said_so(self):
        rows, block, _ = _apply()
        assert I3_APPROX_PREFIX + "bill_inc_vat" not in rows["nat_less"]
        assert "owns another fleet" in block["excluded"]["nat_less"]

    def test_a_stale_institution_summary_is_refused(self):
        def stale(rows):
            rows["nat_central"]["bill_inc_vat"] += 0.5
        with pytest.raises(ValueError, match="not on this tariff"):
            _apply(mutate=stale)


class TestTurnDownBasis:
    def _patch(self, monkeypatch, tmp_path, basis):
        import scripts.market.turn_down_allocation as tda

        csv = tmp_path / "m_pro_rata_turn_down.csv"
        pd.DataFrame({"unit": ["w1", "w2"]}).to_csv(csv, index=False)
        calls = []
        monkeypatch.setattr(tda, "month_basis", lambda path: basis)
        monkeypatch.setattr(tda, "month_files", lambda path: (csv, tmp_path / "m.json"))
        monkeypatch.setattr(tda, "balancing_dispatch_in_force",
                            lambda frame, path: calls.append(path) or frame.assign(w1=frame["w1"] + 1.0, w2=frame["w2"] - 1.0))
        return calls

    def test_rate_free_reshared_units_read_as_solved_and_the_basis_is_recorded(self, monkeypatch, tmp_path):
        calls = self._patch(monkeypatch, tmp_path, "pro_rata_zonal")
        rates = pd.DataFrame({"fuel_cost_gbp_per_mwh": [0.0, 0.0, 50.0], "carbon_cost_gbp_per_mwh": [0.0, 0.0, 20.0],
                              "variable_om_gbp_per_mwh": [0.0, 0.0, 3.0]}, index=["w1", "w2", "ccgt"])
        totals = {"turn_down_basis_by_month": {}, "turn_down_reshared_units_rated": False}
        frame = pd.DataFrame({"w1": [5.0], "w2": [5.0], "ccgt": [1.0]})
        out = ni._dispatch_on_turn_down_basis(frame, tmp_path / "m_balancing.nc", rates, "m", totals)
        assert out.equals(frame) and not calls
        assert totals["turn_down_basis_by_month"] == {"m": "pro_rata_zonal"}

    def test_a_rated_reshared_unit_is_read_on_the_basis_in_force(self, monkeypatch, tmp_path):
        calls = self._patch(monkeypatch, tmp_path, "pro_rata_zonal")
        rates = pd.DataFrame({"fuel_cost_gbp_per_mwh": [0.0, 0.0], "carbon_cost_gbp_per_mwh": [0.0, 0.0],
                              "variable_om_gbp_per_mwh": [2.0, 4.0]}, index=["w1", "w2"])
        totals = {"turn_down_basis_by_month": {}, "turn_down_reshared_units_rated": False}
        out = ni._dispatch_on_turn_down_basis(pd.DataFrame({"w1": [5.0], "w2": [5.0]}), tmp_path / "m_balancing.nc",
                                              rates, "m", totals)
        assert calls and totals["turn_down_reshared_units_rated"]
        assert list(out.iloc[0]) == [6.0, 4.0]

    def test_an_as_solved_month_is_untouched(self, monkeypatch, tmp_path):
        calls = self._patch(monkeypatch, tmp_path, "cheapest_first")
        totals = {"turn_down_basis_by_month": {}, "turn_down_reshared_units_rated": False}
        frame = pd.DataFrame({"w1": [5.0]})
        assert ni._dispatch_on_turn_down_basis(frame, tmp_path / "m_balancing.nc", pd.DataFrame(), "m", totals) is frame
        assert not calls and totals["turn_down_basis_by_month"] == {"m": "cheapest_first"}


class TestExplorer:
    def test_a_case_with_i3_columns_gets_an_i3_block_and_one_without_none(self):
        from scripts.analysis.public_ownership_explorer import extract_explorer_data as x

        rows, block, _ = _apply()
        frame = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows.values()])
        by_id = frame.set_index("case_id")
        twin = x.i3_entry(by_id.loc["nat"], 1.0e-8)
        assert twin["b"] == pytest.approx(round(rows["nat_central"]["bill_inc_vat"], 2))
        assert twin["step"] == pytest.approx(round(rows["nat"][I3_PREFIX_STEP], 2))
        assert len(twin["p"]) == 4 and twin["mb"] is False
        assert x.i3_entry(by_id.loc["nat_tf"], 1.0e-8)["mb"] is True
        assert x.i3_entry(by_id.loc["nat_less"], 1.0e-8) is None
        assert x.i3_entry(by_id.loc["nat_central"], 1.0e-8) is None

    def test_the_view_carries_the_label_and_is_absent_without_the_ledger_block(self):
        from scripts.analysis.public_ownership_explorer import extract_explorer_data as x

        _, block, _ = _apply()
        view = x.p3_i3_view({"institution_step_by_case": block})
        # User (11 Nov 2026): the view carries no approximation label; the note says how it is built.
        assert view["label"] == "Central dispatch" and "measured on the headline case" in view["note"]
        assert view["twin"] == "nat" and view["excluded"] == ["nat_less"] and view["cases"] == 3
        assert x.p3_i3_view({}) is None


I3_PREFIX_STEP = I3_APPROX_PREFIX + "step_bill_gbp_inc_vat"
