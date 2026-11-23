"""Item BM1, P3: the acquisition tariff on its own bid stack beside the same case on the market months.

The case's tariff splits exactly into market private, private on the P3 months
less private on the market months (not a feasible case, a term of the sum only),
and the ownership effect on the stack. The headline is the bid-stack effect, the
same case's tariff on the two stacks, with the purchase-price lines held at the
market year's value: the price is struck before the bid stack changes.
"""

import copy
import json

import pandas as pd
import pytest

from scripts.public_ownership.merge_tariff_balancing_groups import main, merge, tariff_components

CASE = "public_ownership@fully_nationalised_market_value"
COMMON = {"benchmark_kwh": 3100.0, "price_basis": "GBP real 2025", "cfd_recovery_denominator_mwh": 300e6,
          "final_demand_recovery_denominator_mwh": 310e6, "model_rebasing_factor": 1.0,
          "private_reproduces_forecast_case": True}
CONVERSION = {"factor": 1.0, "final_demand_mwh": 310e6, "cfd_eligible_mwh": 300e6, "mwh": 3.1, "vat_rate": 0.05}
PRICE = "nationalised_legacy_acquisition_renewables_nuclear"


def _rows(values):
    rows = []
    for case_id, lines in values.items():
        for component_id, value in lines.items():
            rows.append({"case_id": case_id, "component_id": component_id, "component": component_id.upper(),
                         "stack_group": "g", "annual_gbp_per_3100kwh_ex_vat": value})
    return pd.DataFrame(rows)


def _bill_summary(total, forecast_case):
    return {**COMMON, "forecast_case": forecast_case,
            "cases": {"private": {"total_gbp_inc_vat": total * 1.05, "ex_vat_gbp": total, "vat_gbp": total * 0.05}}}


def _tariff_summary(total, private_total, price):
    return {"cases": {CASE: {
        "institution": "I1", "conversion": dict(CONVERSION), "levers": {"legacy_fleet": "acquire_all"},
        "bill": {"ex_vat_gbp": total, "vat_gbp": total * 0.05, "total_gbp_inc_vat": total * 1.05,
                 "private_total_gbp_inc_vat": private_total * 1.05},
        "diagnostics": {"acquisition_price_gbp": price}}}}


@pytest.fixture()
def inputs():
    # Private: the constraint line falls on the P3 months (a term of the sum, never a saving).
    market_bill = _rows({"private": {"bsuos": 40.0, "cfd": 130.0, "networks": 100.0}})
    group_bill = _rows({"private": {"bsuos": 30.0, "cfd": 131.0, "networks": 100.0}})
    # Tariff: bsuos and cfd replaced; fuel falls 2 on the P3 dispatch; the purchase line
    # moves 1.5 only because the price was valued on each group's own settlement.
    market_tariff = _rows({CASE: {"networks": 100.0, PRICE: 30.0, "nationalised_fuel_carbon": 50.0}})
    group_tariff = _rows({CASE: {"networks": 100.0, PRICE: 28.5, "nationalised_fuel_carbon": 48.0}})
    return {
        "market_bill": market_bill, "market_summary": _bill_summary(270.0, "CP30_2030_cp30_network"),
        "market_tariff": market_tariff, "market_tariff_summary": _tariff_summary(180.0, 270.0, 84e9),
        "group": {"name": "p3", "bid_case": CASE, "by_case": group_bill,
                  "summary": _bill_summary(261.0, "CP30_2030_cp30_network_public_bids_p3"), "tariff": group_tariff,
                  "tariff_summary": _tariff_summary(176.5, 261.0, 82e9),
                  "decomposition": {"year": {"delta_bid_cost_gbp": -9e8, "tie_break_step_cost_gbp": 3e6}}},
    }


def _merge(inputs, **group_overrides):
    group = {**inputs["group"], **group_overrides}
    return merge(inputs["market_bill"], inputs["market_summary"], inputs["market_tariff"],
                 inputs["market_tariff_summary"], [group])


def test_each_line_splits_and_the_tariff_carries_its_own_lines(inputs):
    g = inputs["group"]
    lines = tariff_components(inputs["market_tariff"], g["tariff"], inputs["market_bill"], g["by_case"],
                              CASE).set_index("component_id")
    # Replaced private lines are zero on the tariff; tariff-only lines are zero on private.
    assert lines.loc["bsuos", "case_on_group"] == 0.0 and lines.loc[PRICE, "private_on_group"] == 0.0
    assert lines.loc["nationalised_fuel_carbon", "bid_stack_effect"] == pytest.approx(-2.0)
    assert bool(lines.loc[PRICE, "purchase_price_line"]) and not bool(lines.loc["networks", "purchase_price_line"])
    parts = lines["private_on_market"] + lines["private_bid_stack_decomposition_term"] + lines["ownership_effect_on_stack"]
    assert (parts - lines["case_on_group"]).abs().max() < 1e-12


def test_the_headline_holds_the_purchase_price_at_the_market_year(inputs):
    components, summary = _merge(inputs)
    entry = summary["cases"][CASE]
    assert entry["headline_metric"] == "bid_stack_effect_purchase_price_held_gbp_inc_vat"
    assert entry["bid_stack_effect_gbp_inc_vat"] == pytest.approx(-3.5 * 1.05)
    assert entry["purchase_price_line_effect_gbp_inc_vat"] == pytest.approx(-1.5 * 1.05)
    assert entry["bid_stack_effect_purchase_price_held_gbp_inc_vat"] == pytest.approx(-2.0 * 1.05)
    assert entry["purchase_price_gbp"]["on_market"]["acquisition_price_gbp"] == 84e9
    assert entry["purchase_price_gbp"]["on_group"]["acquisition_price_gbp"] == 82e9
    # The telescoping parts, and private on the stack is only a term of the sum.
    assert entry["private_bid_stack_decomposition_term_gbp_inc_vat"] == pytest.approx(-9.0 * 1.05)
    assert entry["ownership_effect_on_stack_gbp_inc_vat"] == pytest.approx((176.5 - 261.0) * 1.05)
    assert entry["delta_vs_private_gbp_inc_vat"] == pytest.approx((176.5 - 270.0) * 1.05)
    assert set(entry["bid_stack_effect_by_line_gbp_inc_vat"]) == {PRICE, "nationalised_fuel_carbon"}
    assert entry["decomposition"]["tie_break_step_cost_gbp"] == 3e6
    assert set(components["run_group"]) == {"p3"}


def test_a_group_that_projects_more_than_private_is_refused(inputs):
    summary = copy.deepcopy(inputs["group"]["summary"])
    summary["cases"]["public_ownership"] = dict(summary["cases"]["private"])
    with pytest.raises(ValueError, match="projects private alone"):
        _merge(inputs, summary=summary)


TWIN = "public_ownership@fully_nationalised_market_value_term_15y"


def _with_twin(inputs, twin_on_market=True):
    """The 15-year-term twin on both stacks: same lines but a larger purchase annuity."""
    def add(frame, summary, price_line, total):
        rows = frame.loc[frame["case_id"].eq(CASE)].copy()
        rows["case_id"] = TWIN
        rows.loc[rows["component_id"].eq(PRICE), "annual_gbp_per_3100kwh_ex_vat"] = price_line
        frame = pd.concat([frame, rows], ignore_index=True)
        summary = copy.deepcopy(summary)
        twin = copy.deepcopy(summary["cases"][CASE])
        twin["bill"].update(ex_vat_gbp=total, vat_gbp=total * 0.05, total_gbp_inc_vat=total * 1.05)
        summary["cases"][TWIN] = twin
        return frame, summary

    group = dict(inputs["group"])
    group["tariff"], group["tariff_summary"] = add(group["tariff"], group["tariff_summary"], 40.0, 188.0)
    market_tariff, market_summary = inputs["market_tariff"], inputs["market_tariff_summary"]
    if twin_on_market:
        market_tariff, market_summary = add(market_tariff, market_summary, 42.0, 192.0)
    return merge(inputs["market_bill"], inputs["market_summary"], market_tariff, market_summary, [group])


def test_every_case_on_the_stack_gets_its_own_entry_and_the_bid_case_heads(inputs):
    components, summary = _with_twin(inputs)
    assert set(summary["cases"]) == {CASE, TWIN}
    twin = summary["cases"][TWIN]
    assert twin["bid_case"] == CASE and not twin["is_bid_case"] and summary["cases"][CASE]["is_bid_case"]
    assert twin["bid_stack_effect_gbp_inc_vat"] == pytest.approx(-4.0 * 1.05)
    assert twin["bid_stack_effect_purchase_price_held_gbp_inc_vat"] == pytest.approx(-2.0 * 1.05)
    headline = summary["headline_by_group"]["p3"]
    assert headline["case_id"] == CASE and headline["cases_on_stack"] == [CASE, TWIN]
    assert headline["gbp_inc_vat"] == pytest.approx(-2.0 * 1.05)
    assert set(components["case_id"]) == {CASE, TWIN}


def test_a_case_the_market_group_does_not_tariff_is_refused(inputs):
    with pytest.raises(ValueError, match="has no comparator|does not price"):
        _with_twin(inputs, twin_on_market=False)


def test_a_tariff_without_the_bid_case_is_refused(inputs):
    tariff = copy.deepcopy(inputs["group"]["tariff_summary"])
    tariff["cases"] = {"public_ownership@fully_nationalised": tariff["cases"][CASE]}
    with pytest.raises(ValueError, match="must price its bid case"):
        _merge(inputs, tariff_summary=tariff)


def test_a_tariff_built_on_another_private_bill_is_refused(inputs):
    tariff = copy.deepcopy(inputs["group"]["tariff_summary"])
    tariff["cases"][CASE]["bill"]["private_total_gbp_inc_vat"] = 270.0 * 1.05
    with pytest.raises(ValueError, match="not the same projection"):
        _merge(inputs, tariff_summary=tariff)


def test_tariffs_on_different_conversions_are_refused(inputs):
    tariff = copy.deepcopy(inputs["group"]["tariff_summary"])
    tariff["cases"][CASE]["conversion"]["final_demand_mwh"] = 300e6
    with pytest.raises(ValueError, match="conversion final_demand_mwh"):
        _merge(inputs, tariff_summary=tariff)


def test_lines_that_do_not_sum_to_the_total_are_refused(inputs):
    tariff = inputs["group"]["tariff"].copy()
    tariff.loc[tariff["component_id"].eq("networks"), "annual_gbp_per_3100kwh_ex_vat"] = 101.0
    with pytest.raises(ValueError, match="not the same projection"):
        _merge(inputs, tariff=tariff)


def test_the_command_line_writes_both_outputs(inputs, tmp_path):
    g = inputs["group"]
    paths = {}
    for name, frame in (("market_bill", inputs["market_bill"]), ("market_tariff", inputs["market_tariff"]),
                        ("group_bill", g["by_case"]), ("group_tariff", g["tariff"])):
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    for name, document in (("market_summary", inputs["market_summary"]),
                           ("market_tariff_summary", inputs["market_tariff_summary"]),
                           ("group_summary", g["summary"]), ("group_tariff_summary", g["tariff_summary"]),
                           ("decomposition", g["decomposition"])):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(document), encoding="utf-8")
    out_csv, out_json = tmp_path / "out.csv", tmp_path / "out.json"
    main([str(paths["market_bill"]), str(paths["market_summary"]), str(paths["market_tariff"]),
          str(paths["market_tariff_summary"]), str(out_csv), str(out_json),
          "--group", "p3", CASE, str(paths["group_bill"]), str(paths["group_summary"]), str(paths["group_tariff"]),
          str(paths["group_tariff_summary"]), str(paths["decomposition"])])
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written["cases"][CASE]["bid_stack_effect_purchase_price_held_gbp_inc_vat"] == pytest.approx(-2.1)
    assert len(pd.read_csv(out_csv)) == 5
