"""P1/P2 under central dispatch (20 Oct 2026): the institution step, the merge, and the forbidden sum.

Each case's I3 bill against private-I1 is a chain of two measured steps (institution given owner,
owner given institution); the three-party identity closes on each and on the sum; a saving measured
under I1 is never added to one measured on private under I3.
"""

import pandas as pd
import pytest

from scripts.public_ownership import investor_ledger as investors
from scripts.public_ownership import merge_institution_groups as mig

VALUE = mig.VALUE


def _position(rent, revenue, fuel, required, wholesale, balancing, cfd, roc, ldes=0.0, battery=0.0, cfdc=0.0):
    return {"producer_rent_gbp": rent, "unresolved_net_cash_gbp": 0.0, "revenue_gbp": revenue,
            "fuel_carbon_gbp": fuel, "required_revenue_gbp": required, "wholesale_revenue_gbp": wholesale,
            "balancing_settlement_gbp": balancing, "cfd_support_gbp": cfd, "roc_support_gbp": roc,
            "ldes_floor_topup_paid_gbp": ldes, "battery_revenue_top_up_gbp": battery, "carbon_removal_cfd_gbp": cfdc,
            "private_rent_gbp": rent, "capital_recovery_gbp": 500.0, "corporation_tax_forgone_gbp": 0.0,
            "tax_equivalent_gbp": 0.0, "carbon_removal_cfd_gbp_": 0.0}


def _positions():
    """Two private positions whose rent is revenue + battery top-up - fuel - required revenue."""
    market = dict(wholesale=1000.0, balancing=300.0, cfd=200.0, roc=50.0, ldes=10.0, cfdc=20.0)
    central = dict(wholesale=990.0, balancing=280.0, cfd=210.0, roc=48.0, ldes=12.0, cfdc=22.0)
    out = []
    for p, fuel, req in ((market, 400.0, 600.0), (central, 380.0, 595.0)):
        revenue = p["wholesale"] + p["balancing"] + p["cfd"] + p["roc"] + p["ldes"] + p["cfdc"]
        out.append(_position(revenue - fuel - req, revenue, fuel, req, p["wholesale"], p["balancing"], p["cfd"],
                             p["roc"], p["ldes"], 0.0, p["cfdc"]))
    return out


MEASURED = {"schedule_trade_change_gbp": 7.0, "rent_change_before_make_whole_gbp": -3.0, "make_whole_gbp": 4.0,
            "gb_half_revenue_change_gbp": 0.5, "foreign_half_revenue_change_gbp": 0.5, "realtime_links_gbp": 1.5,
            "market_retrading_gbp": 30.0, "forecast_error_charge_gbp": 40.0, "counter_trade_premium_gbp": 0.25,
            "realtime_at_cost_register_gbp": -2.0, "exchequer_paid_revenue_change_gbp": 2.0}
LINES = {"direct_energy": 0.0, "bsuos_constraints": -25.0, "cfd": 10.0, "renewables_obligation": -1.5,
         "ldes_cap_and_floor": 2.0, "interconnector_cap_and_floor_pass_through": -0.75,
         "supplier_costs_and_margin": -0.4}


def test_the_institution_step_closes_with_the_exchequers_carbon_vat_and_carbon_cfd():
    market, central = _positions()
    carbon = -12.0
    step = investors.institution_step_parties_market(line_changes=LINES, position_central=central,
                                                     position_market=market, carbon_change_gbp=carbon,
                                                     measured=MEASURED)
    vat = -1.1
    exchequer = carbon + vat - MEASURED["exchequer_paid_revenue_change_gbp"]
    row = investors.three_party_row(step, vat_change_gbp=vat, exchequer_change_gbp=exchequer, case_id="step")
    assert abs(row["three_party_identity_residual_gbp"]) < 1e-9
    parts = row["resource_saving_vs_private_parts_gbp"]
    # The forecast-error resource: the charge and premium less the real-time correction they pay (at cost, abroad).
    assert parts[investors.INSTITUTION_STEP_FORECAST_ERROR] == pytest.approx(-(40.0 + 0.25 + 2.0 - 1.5))
    # Abroad: the Stage 1 trade, the foreign halves, the central real-time link trade; re-trading gone.
    assert parts[investors.INSTITUTION_STEP_ABROAD] == pytest.approx(-(7.0 + 0.5 + 1.5 - 30.0))
    inv = row["investor_change_vs_private_parts_gbp"]
    assert inv["interconnector GB halves: Stage 1 rent and the day-ahead make-whole"] == 0.5
    assert inv["interconnector GB halves: the cap-and-floor pass-through they pay (bill line)"] == -0.75
    with pytest.raises(ValueError, match="cannot assign"):
        investors.institution_step_parties_market(line_changes={**LINES, "capacity_market": 1.0},
                                                  position_central=central, position_market=market,
                                                  carbon_change_gbp=carbon, measured=MEASURED)
    with pytest.raises(ValueError, match="GB and foreign halves"):
        investors.institution_step_parties_market(line_changes=LINES, position_central=central,
                                                  position_market=market, carbon_change_gbp=carbon,
                                                  measured={**MEASURED, "gb_half_revenue_change_gbp": 9.0})
    with pytest.raises(ValueError, match="needs the settlement"):
        investors.institution_step_parties_market(line_changes=LINES, position_central=central,
                                                  position_market=market, carbon_change_gbp=carbon,
                                                  measured={k: v for k, v in MEASURED.items()
                                                            if k != "realtime_links_gbp"})


def test_only_chains_of_meeting_steps_add():
    inst = mig.Step(("private", "I1"), ("private", "I3"), -5.0, "institution given owner")
    own = mig.Step(("private", "I3"), ("public_ownership", "I3"), -20.0, "owner given institution")
    total = mig.chain(inst, own)
    assert total.value == -25.0 and total.start == ("private", "I1") and total.end == ("public_ownership", "I3")
    p2_i1 = mig.Step(("private", "I1"), ("public_ownership", "I1"), -30.0, "P2-I1 saving")
    with pytest.raises(ValueError, match="refused"):
        mig.chain(p2_i1, inst)                     # a P2-I1 saving plus a private-I3 saving: the forbidden sum
    with pytest.raises(ValueError, match="refused"):
        mig.chain(inst, p2_i1)
    # P2-I1 then the institution given P2 is a chain, and so allowed.
    assert mig.chain(p2_i1, mig.Step(("public_ownership", "I1"), ("public_ownership", "I3"), 5.0)).value == -25.0


# ── the merge on small synthetic bills ────────────────────────────────────────

CONV = {"factor": 1.0, "final_demand_mwh": 1e6, "exchequer_key_mwh": 1e6, "mwh": 3.1, "vat_rate": 0.05,
        "domestic_mwh": 1e5}


def _by_case(bills):
    rows = [{"case_id": case, "component_id": line, "component": line, VALUE: value}
            for case, lines in bills.items() for line, value in lines.items()]
    return pd.DataFrame(rows)


def _summary(bills, institution):
    cases = {}
    for case, lines in bills.items():
        ex = float(sum(lines.values()))
        cases[case] = {"ex_vat_gbp": ex, "vat_gbp": 0.05 * ex, "total_gbp_inc_vat": 1.05 * ex, "levers": {"x": case}}
    return {"cases": cases, "institution": institution, "forecast_case": institution,
            "private_reproduces_forecast_case": True}


def _ledger_row(case, bill_ex_vat, carbon, **extra):
    return {"case_id": case, "bill_ex_vat": bill_ex_vat, "bill_inc_vat": 1.05 * bill_ex_vat,
            "carbon_receipts_gbp": carbon, "corporation_tax_forgone_gbp": 0.0, "network_tax_forgone_gbp": 0.0,
            "state_deficits_funded_gbp": 0.0, "state_surplus_retained_gbp": 0.0, "exchequer_residual_gbp": 0.0,
            "financing_saving_gbp": 0.0, "financing_saving_tax_transfer_gbp": 0.0,
            "carbon_removal_cfd_exchequer_gbp": extra.pop("cfdc", 0.0), **extra}


def test_the_merge_splits_each_case_and_closes_the_identity(monkeypatch):
    from scripts.public_ownership import bill_recovery, exchequer_ledger
    import scripts.public_ownership.merge_balancing_groups as mbg

    # One household pound of any line is GBP 1m of system cost here (the bills' own entries elsewhere).
    monkeypatch.setattr(bill_recovery, "system_from_household", lambda summary, line, hh, case_id="private": hh * 1e6)
    monkeypatch.setattr(mbg, "check_comparable", lambda *a, **k: None)
    i1 = {"private": {"direct_energy": 500.0, "bsuos_constraints": 60.0, "cfd": 40.0},
          "public_ownership": {"direct_energy": 500.0, "bsuos_constraints": 60.0, "cfd": 20.0}}
    lines_m = {k: v / 1e6 for k, v in LINES.items()}
    i3_private = {"direct_energy": 500.0, "bsuos_constraints": 60.0 + lines_m["bsuos_constraints"],
                  "cfd": 40.0 + lines_m["cfd"]}
    i3 = {"private": i3_private, "public_ownership": {**i3_private, "cfd": i3_private["cfd"] - 21.0}}
    # Keep only the lines the synthetic bills carry, so the step moves exactly those.
    moved = {k: LINES[k] for k in ("bsuos_constraints", "cfd")}
    market, central = _positions()
    carbon_m, carbon_c = 100.0, 88.0
    vat = lambda ex: exchequer_ledger.domestic_vat(ex, CONV)  # noqa: E731
    step_exch = (carbon_c - carbon_m) + (vat(sum(i3_private.values())) - vat(sum(i1["private"].values()))) \
        - MEASURED["exchequer_paid_revenue_change_gbp"]
    step = investors.institution_step_parties_market(line_changes=moved, position_central=central,
                                                     position_market=market, carbon_change_gbp=carbon_c - carbon_m,
                                                     measured=MEASURED)
    # The ownership step on the twin, closed as its own ledger closes it.
    own_bill, own_vat = -21.0e6, vat(sum(i3["public_ownership"].values())) - vat(sum(i3_private.values()))
    own = {"consumer_bill_change_system_gbp": own_bill, "investor_change_vs_private_parts_gbp": {"rent": -5e6},
           "resource_saving_vs_private_parts_gbp": {"capital": 16e6 - 0.0}, "vat_change_vs_private_gbp": own_vat,
           "exchequer_net_vs_private_gbp": own_vat}
    own["resource_saving_vs_private_parts_gbp"] = {"capital": -own_bill - own_vat + own_vat - 5e6}
    ledger_m = {"cases": {"private": _ledger_row("private", sum(i1["private"].values()), carbon_m, cfdc=20.0)}}
    ledger_t = {"cases": {"private": {**_ledger_row("private", sum(i3_private.values()), carbon_c, cfdc=22.0),
                                      "consumer_bill_change_system_gbp": 0.0,
                                      "investor_change_vs_private_parts_gbp": {},
                                      "resource_saving_vs_private_parts_gbp": {},
                                      "vat_change_vs_private_gbp": 0.0, "exchequer_net_vs_private_gbp": 0.0},
                          "public_ownership": {**_ledger_row("public_ownership", sum(i3["public_ownership"].values()),
                                                             carbon_c, cfdc=22.0), **own}}}
    components, summary = mig.merge(
        market_group="m", twin_group="m_cd", market_by_case=_by_case(i1), market_summary=_summary(i1, "market"),
        twin_by_case=_by_case(i3), twin_summary=_summary(i3, "central_dispatch"), market_ledger=ledger_m,
        twin_ledger=ledger_t, position_market=market, position_central=central, measured=MEASURED, conv=CONV)
    assert abs(summary["institution_step"]["three_party"]["three_party_identity_residual_gbp"]) < 1.0
    entry = summary["cases"]["public_ownership"]
    assert entry[mig.HEADLINE_METRIC] == pytest.approx(entry["institution_given_owner_gbp_inc_vat"]
                                                       + entry["owner_given_institution_gbp_inc_vat"])
    assert entry[mig.HEADLINE_METRIC] == pytest.approx(1.05 * (sum(i3["public_ownership"].values())
                                                              - sum(i1["private"].values())))
    assert abs(entry["three_party_vs_private_i1"]["three_party_identity_residual_gbp"]) < 1.0
    # The I1 ownership saving is beside it, for information only.
    assert entry["ownership_under_i1_info_gbp_inc_vat"] == pytest.approx(1.05 * -20.0)
    rows = components.loc[components["case_id"].eq("public_ownership")].set_index("component_id")
    assert rows.loc["cfd", "i3_vs_private_i1"] == pytest.approx(rows.loc["cfd", "institution_given_owner"]
                                                                + rows.loc["cfd", "owner_given_institution"])
    with pytest.raises(ValueError, match="not a central-dispatch bill"):
        mig.merge(market_group="m", twin_group="m_cd", market_by_case=_by_case(i1),
                  market_summary=_summary(i1, "market"), twin_by_case=_by_case(i3), twin_summary=_summary(i3, "market"),
                  market_ledger=ledger_m, twin_ledger=ledger_t, position_market=market, position_central=central,
                  measured=MEASURED, conv=CONV)


def test_the_measured_quantities_come_from_the_settlement_and_the_bills():
    rows = {"central_ic_schedule_trade_cost_info_gbp_m": 10.0, "market_ic_schedule_trade_cost_info_gbp_m": 8.0,
            "central_ic_congestion_rent_before_make_whole_info_gbp_m": 5.0, "market_ic_congestion_rent_gbp_m": 6.0,
            "central_ic_make_whole_gbp_m": 2.0, "central_ic_physical_trade_cost_info_gbp_m": 10.5,
            "central_counter_trade_premium_gbp_m": 0.1,
            "market_ic_rent_gbp_m::A": 4.0, "central_ic_rent_gbp_m::A": 5.0,
            "market_ic_rent_gbp_m::B": 2.0, "central_ic_rent_gbp_m::B": 2.0}
    table = pd.DataFrame({"YEAR": rows})
    twin = {"balancing_costs": {"institution": "central_dispatch", "interconnector_make_whole_gbp": 2.0e6,
                                "forecast_error_charge_gbp": 40e6, "counter_trade_premium_gbp": 0.1e6}}
    market = {"balancing_costs": {"include_interconnector_retrading": True, "interconnector_retrading_cost_gbp": 30e6}}
    m = mig.measured_quantities(table, twin, market, {"A": 0.5, "B": 0.0}, -1e6, 2e6)
    assert m["schedule_trade_change_gbp"] == pytest.approx(2e6)
    assert m["rent_change_before_make_whole_gbp"] == pytest.approx(-1e6)
    assert m["gb_half_revenue_change_gbp"] == pytest.approx(0.5e6)
    assert m["foreign_half_revenue_change_gbp"] == pytest.approx(0.5e6)
    assert m["realtime_links_gbp"] == pytest.approx(0.6e6)
    assert m["market_retrading_gbp"] == 30e6 and m["forecast_error_charge_gbp"] == 40e6
    with pytest.raises(ValueError, match="no GB share"):
        mig.measured_quantities(table, twin, market, {"A": 0.5}, 0.0, 0.0)
    with pytest.raises(ValueError, match="make-whole"):
        mig.measured_quantities(table, {"balancing_costs": {**twin["balancing_costs"],
                                                            "interconnector_make_whole_gbp": 9e6}},
                                market, {"A": 0.5, "B": 0.0}, 0.0, 0.0)
