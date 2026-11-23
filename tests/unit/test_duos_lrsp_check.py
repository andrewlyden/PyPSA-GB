"""The DUoS line on the clean-base RIIO-ED2 path, and the LRSP check behind it (29 Sep 2026).

The earlier ED2-path candidate rolled forward the 2024/25 Ofgem cap DUoS. The DNOs'
LC14 charging statements (Annex 7) publish the domestic Supplier of Last Resort fixed
adders that the cap's fixed charges include. They show that the 2024/25 base carries
LRSP (GBP0.78) and bad debt (GBP1.90). It also carries a correction factor of about
GBP19.7 for the 2023/24 under-recovery.

The user's decision is to apply the same ED2 path on the clean 2026/27 base (cap 17a,
zero adders), with low and high as the sensitivity. These tests pin the applied value,
the ED2 ratios (to the tracked Finance Annex), the reported sensitivities and the
not-applied record. Since 6 Oct 2026 (user decision) the ED3 years grow on CP30's
distribution capex rather than the ED2 RAV growth (scripts/consumer_bill/duos_cp30_growth.py).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.consumer_bill import project_future_bill as bill

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
ADDERS = Path("data/market/network_cost_sources/duos_lrsp/dno_supplier_failure_fixed_adders.csv")
FY2024_25_TO_2025 = 138.0 / 134.175
BLEND = lambda g: 0.25 * g**2 + 0.75 * g**3                    # calendar 2030 = 3/12 FY29/30 + 9/12 FY30/31


def _duos():
    return json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]["duos"]


def _adders(year):
    frame = pd.read_csv(ADDERS)
    return frame[frame.charging_year.eq(year) & frame.status.eq("retrieved")]


def _ed2_path(base, revenue_ratio, rav_growth):
    low = base * revenue_ratio
    return low, low * BLEND(rav_growth), base * BLEND(1.05)


@pytest.fixture(scope="module")
def ed2_totals():
    """Calculated revenue and closing RAV summed over the 14 DNOs, parsed from the ED2 FD Finance Annex."""
    from scripts.consumer_bill.extract_network_price_control_building_blocks import extract_ed2_appendix5

    frame = extract_ed2_appendix5(Path(_duos()["derivation"]["ed2_growth_source_file"]))
    assert frame.licensee.nunique() == 14
    return {item: frame[frame["item"].eq(item)][["2024/25", "2026/27", "2027/28"]].sum()
            for item in ("Calculated revenue", "Closing RAV")}


def test_the_applied_line_is_the_clean_base_on_cp30s_ed3_path():
    """User decisions, 6 Oct 2026: ED2 (Appendix 8) to 2027/28, then CP30's distribution capex into ED3,
    with the unit rate spread over CP30's demand growth."""
    from scripts.consumer_bill import duos_cp30_growth as dg
    from scripts.consumer_bill.extract_network_price_control_building_blocks import extract_ed2_appendix8

    duos = _duos()
    ed2 = duos["derivation"]
    frame = extract_ed2_appendix8(Path(ed2["ed2_growth_source_file"]))
    revenue = frame[frame["item"].eq("Calculated revenue")][["2026/27", "2027/28"]].sum()
    rav = frame[frame["item"].eq("Closing RAV")][["2026/27", "2027/28"]].sum()
    assert ed2["ed2_calculated_revenue_gbp_m_2020_21"] == pytest.approx(revenue.to_dict())
    assert ed2["ed2_closing_rav_gbp_m_2020_21"] == pytest.approx(
        {"closing_2026_27": rav["2026/27"], "closing_2027_28": rav["2027/28"]})
    assert ed2["ed2_revenue_ratio_2027_28_over_2026_27"] == pytest.approx(revenue["2027/28"] / revenue["2026/27"])
    assert "ed3_real_growth_per_year" not in ed2                       # the ED2 RAV growth rule is superseded
    central, low, high = (dg.path(c) for c in ("central", "low", "high"))
    to_2025 = duos["rebasing"]["factor"]
    # The projector's own calendar blend gives the central value in 2025 prices.
    value, _ = bill.blended_household_value("duos", duos, 2030, 2025)
    assert value == pytest.approx(central["calendar_2030_gbp_2025"], abs=1e-3) and round(value, 2) == 123.34
    assert duos["value"] * to_2025 == pytest.approx(value, abs=1e-6)
    assert ed2["central_gbp_2025_prices"] == pytest.approx(value, abs=1e-3)
    assert bill.check_period_basis("duos", duos, 2030) == "calendar_blend"
    # The ED3 years at the ED3 allowed return: RAV x (4.35% - 3.93%) on the CP30 path's own RAV, added to
    # allowed revenue and charged through the same revenue ratio and charging base as the rest of the line.
    uplift = ed2["ed3_allowed_return_uplift"]
    assert (uplift["ed2_vanilla_wacc_cpih_real"], uplift["ed3_wacc_cpih_real"]) == (0.0393, 0.0435)
    assert "final_demand_mwh" not in uplift
    for period, part in uplift["by_period"].items():
        year = central["by_fiscal_year"][period]
        assert part["rav_gbp_2024"] == pytest.approx(year["average_rav_gbp_2024"])
        assert part["uplift_revenue_gbp_m_2020_21"] == pytest.approx(year["opening_rav_gbp_m_2020_21"] / 2 * 0.0042
                                                                     + year["closing_rav_gbp_m_2020_21"] / 2 * 0.0042)
        assert part["gbp_per_household_2026_prices"] == pytest.approx(
            126.4094 * part["uplift_revenue_gbp_m_2020_21"] / revenue["2026/27"] * year["charging_base_factor"])
    assert uplift["calendar_2030_gbp_per_household_2025_prices"] == pytest.approx(central["calendar_2030_uplift_gbp_2025"])
    sensitivity = ed2["sensitivity_gbp_per_household_year"]
    assert sensitivity["low"]["value"] == pytest.approx(low["calendar_2030_gbp_2025"], abs=1e-3)
    assert sensitivity["high"]["value"] == pytest.approx(high["calendar_2030_gbp_2025"], abs=1e-3)
    assert (round(sensitivity["low"]["value"], 2), round(sensitivity["high"]["value"], 2)) == (117.18, 124.81)
    held = dg.path("central", spread=False)
    assert sensitivity["charging_base_held"]["value"] == pytest.approx(held["calendar_2030_gbp_2025"], abs=1e-3)
    assert sensitivity["charging_base_held"]["value"] > value
    assert {s["price_year"] for s in sensitivity.values()} == {2025}
    walk = ed2["previous_value"]["walk_to_new_central_gbp_2025"]
    assert walk["previous"] == 136.0134
    assert walk["previous"] + sum(v for k, v in walk.items() if k not in ("previous", "new")) == pytest.approx(walk["new"])


def test_the_low_and_high_are_reported_as_bill_sensitivities():
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    # Only DUoS is on this frame; a register-booked line's sensitivities need its own line (OW16), and the
    # BSUoS services lines' cost sensitivities need their own lines and a recovery file (6 Oct 2026).
    for key in ("battery_revenue_top_up", "bsuos_other_services", "bsuos_voltage_and_stability_constraints"):
        assumptions["sourced_externals"].pop(key, None)
    sensitivity = assumptions["sourced_externals"]["duos"]["derivation"]["sensitivity_gbp_per_household_year"]
    central = assumptions["sourced_externals"]["duos"]["derivation"]["central_gbp_2025_prices"]
    frame = pd.DataFrame({"component_id": ["duos"], "annual_gbp_per_3100kwh_ex_vat": [central]})
    out = bill.levy_and_cm_sensitivities(frame, assumptions, None, 3100)
    assert out["duos_low"] == pytest.approx(sensitivity["low"]["value"] - central, abs=1e-3)
    assert out["duos_high"] == pytest.approx(sensitivity["high"]["value"] - central, abs=1e-3)
    assert out["duos_low"] < 0 < out["duos_high"]
    sensitivity["low"]["price_year"] = 2026
    with pytest.raises(ValueError, match="report price year"):
        bill.levy_and_cm_sensitivities(frame, assumptions, None, 3100)


def test_the_2024_25_lrsp_recorded_is_the_mean_of_the_published_adders():
    year = _adders("2024/25")
    assert len(year) == 9 and year.solr_fixed_adder_p_per_mpan_day.between(0.19, 0.24).all()
    assert (year.excess_solr_fixed_adder_p_per_mpan_day.fillna(0) == 0).all()
    # The cap takes the DNOs' final fixed charges, and those already include the adders.
    assert (year.annex1_domestic_fixed_charge_p_per_mpan_day == year.annex3_v1_25_fixed_charge_p_per_day).all()
    solr = (year.solr_fixed_adder_p_per_mpan_day + year.excess_solr_fixed_adder_p_per_mpan_day.fillna(0)).mean()
    check = _duos()["ed2_path_candidate_not_applied"]["lrsp_check"]
    derivation = check["derivation"]
    assert Path(check["source_file"]) == ADDERS and check["source_url"].startswith("https://")
    assert derivation["solr_plus_excess_adder_gb_mean_p_per_mpan_day"] == pytest.approx(solr, abs=1e-6)
    assert derivation["lrsp_gbp_2025_prices"] == pytest.approx(solr * 365 / 100 * FY2024_25_TO_2025, abs=1e-5)
    assert derivation["rebasing"]["factor"] == pytest.approx(FY2024_25_TO_2025)
    assert set(derivation["regions_retrieved_2024_25"]) == set(year.dno_licensee)


def test_the_2024_25_candidates_are_recorded_as_not_applied(ed2_totals):
    candidate = _duos()["ed2_path_candidate_not_applied"]
    assert "Not applied" in candidate["status"] and "correction factor" in candidate["status"]
    assert "period_basis" not in candidate                       # a record, not a line
    revenue, rav = ed2_totals["Calculated revenue"], ed2_totals["Closing RAV"]
    ratio, g = revenue["2027/28"] / revenue["2024/25"], rav["2027/28"] / rav["2026/27"]
    base = 144.587 * FY2024_25_TO_2025                           # Ofgem Annex 3 v1.25 tab 2b, Apr-Jun 2024, GB mean
    recorded = candidate["gbp_per_3100kwh_2025_prices"]
    assert [recorded[k] for k in ("low_ed2_then_flat", "central_ed3_at_ed2_rav_growth_3_1pct",
                                  "high_plus_5pct_real_a_year_in_ed3")] == pytest.approx(_ed2_path(base, ratio, g), abs=0.1)
    lrsp = candidate["lrsp_check"]["derivation"]["lrsp_gbp_2025_prices"]
    stripped = candidate["lrsp_stripped_gbp_per_3100kwh_2025_prices"]
    assert stripped["status"].startswith("not applied")
    assert [stripped["low_ed2_then_flat"], stripped["central_ed3_at_ed2_rav_growth_3_1pct"],
            stripped["high_plus_5pct_real_a_year_in_ed3"]] == pytest.approx(_ed2_path(base - lrsp, ratio, g), abs=0.01)
    assert candidate["clean_base_path_gbp_per_3100kwh_2025_prices"]["status"].startswith("superseded 6 Oct 2026")


def test_the_clean_base_carries_no_supplier_failure_adders():
    held_year = _adders("2026/27")
    assert len(held_year) >= 6
    assert (held_year.solr_fixed_adder_p_per_mpan_day == 0).all()
    assert (held_year.eligible_bad_debt_fixed_adder_p_per_mpan_day == 0).all()
