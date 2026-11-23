"""DUoS into RIIO-ED3 on CP30's distribution capex (user decision, 6 Oct 2026)."""

import json
from pathlib import Path

import pytest

from scripts.consumer_bill import duos_cp30_growth as dg

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
WORKBOOK = Path("data/CP30/Clean Power 2030 Data Workbook_V003.xlsx")
ANNEX = Path(dg.ED2_REVISED["source_file"])


def _duos():
    return json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]["duos"]


def test_the_ed2_inputs_are_appendix_8_alone():
    from scripts.consumer_bill.extract_network_price_control_building_blocks import (
        extract_ed2_appendix5, extract_ed2_appendix8)

    revised = extract_ed2_appendix8(ANNEX)
    assert revised.licensee.nunique() == 14 and len(revised) == 140
    total = lambda frame, item, year: float(frame.loc[frame["item"].eq(item), year].sum())  # noqa: E731
    e = dg.ED2_REVISED
    for year in ("2026/27", "2027/28"):
        assert total(revised, "Calculated revenue", year) == pytest.approx(e["calculated_revenue"][year])
        assert total(revised, "Closing RAV", year) == pytest.approx(e["closing_rav"][year])
    for item, key in (("Opening RAV", "opening_rav_2027_28"), ("Net additions (after disposals)", "net_additions_2027_28"),
                      ("Depreciation", "depreciation_2027_28"), ("Return", "return_2027_28"),
                      ("Tax allowance", "tax_allowance_2027_28"), ("Fast pot expenditure", "fast_pot_2027_28")):
        assert total(revised, item, "2027/28") == pytest.approx(e[key])
    # The finding behind it: the Appendix 5 reader returns every DNO twice (Appendix 5 and 8), so the
    # building-blocks levels the network account reads are about double the 14 DNOs'.
    both = extract_ed2_appendix5(ANNEX)
    assert (both.groupby(["licensee", "item"]).size() == 2).all()
    assert total(both, "Closing RAV", "2027/28") == pytest.approx(2 * e["closing_rav"]["2027/28"], rel=0.005)


def test_the_cp30_inputs_are_the_workbooks():
    if not WORKBOOK.exists():
        pytest.skip("CP30 workbook not present")
    import openpyxl

    book = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    cp13 = {row[0]: row for row in book["CP.13"].iter_rows(min_row=6, max_row=11, min_col=11, max_col=13, values_only=True)}
    assert cp13["Distribution network CAPEX"][1] == pytest.approx(dg.CP13_DISTRIBUTION_CAPEX_GBP_BN_2024)
    rows = list(book["ED1"].iter_rows(values_only=True))
    header = next(r for r in rows if r and "Data item" in r)
    item, pathway = header.index("Data item"), header.index("Pathway")
    peak = next(r for r in rows if r and str(r[item] or "").startswith("GBFES Peak Customer Demand")
                and r[pathway] == "Further Flex and Renewables")
    for year, value in dg.PEAK_GW.items():
        assert peak[header.index(year)] == pytest.approx(value, abs=1e-3)
    demand = next(r for r in rows if r and str(r[item] or "").startswith("GB Total System Demand - Minus electrolysis")
                  and r[pathway] == "Further Flex and Renewables")
    for year, value in dg.DEMAND_GWH.items():
        assert demand[header.index(year)] == pytest.approx(value, rel=1e-12)


def test_the_path_rolls_the_rav_and_revenue_forward():
    central, low, high = (dg.path(c) for c in ("central", "low", "high"))
    e = dg.ED2_REVISED
    previous = e["closing_rav"]["2027/28"]
    for year in central["by_fiscal_year"].values():
        assert year["opening_rav_gbp_m_2020_21"] == pytest.approx(previous)
        assert year["closing_rav_gbp_m_2020_21"] == pytest.approx(
            year["opening_rav_gbp_m_2020_21"] + dg.cp13_additions_gbp_m_2020_21() - year["depreciation_gbp_m_2020_21"])
        previous = year["closing_rav_gbp_m_2020_21"]
    assert low["calendar_2030_gbp_2025"] < central["calendar_2030_gbp_2025"] < high["calendar_2030_gbp_2025"]
    # The low case is ED2 to 2027/28 then flat: the cap base x the ED2 revenue ratio (on the held charging base).
    ratio = e["calculated_revenue"]["2027/28"] / e["calculated_revenue"]["2026/27"]
    assert dg.path("low", spread=False)["calendar_2030_at_ed2_return_gbp_2025"] == pytest.approx(
        dg.CAP_BASE_2026 * ratio * dg.CAP_2026_TO_2025)
    # CP30's capex is below ED2's own RAV additions, so ED3 grows more slowly than ED2's last year.
    assert dg.comparison()["cp30_over_ed2_rav_additions"] < 1
    rav_growth = (central["by_fiscal_year"]["FY2030/31"]["closing_rav_gbp_m_2020_21"] / e["closing_rav"]["2027/28"]) ** (1 / 3)
    assert 1.0 < rav_growth < e["closing_rav"]["2027/28"] / e["closing_rav"]["2026/27"]


def test_the_unit_rate_is_spread_over_cp30s_demand_growth():
    """User decision, 6 Oct 2026: the unit part of the charge falls as CP30's demand grows; the fixed part does not."""
    assert dg.UNIT_SHARE == pytest.approx(27.104 * 3.1 / 126.4094)
    assert dg.charging_base_factor(2026) == pytest.approx(1.0)
    assert dg.charging_base_factor(2030, spread=False) == 1.0
    d26, d30 = dg.charging_year_demand_gwh(2026), dg.charging_year_demand_gwh(2030)
    assert d26 == pytest.approx(0.75 * dg.DEMAND_GWH[2026] + 0.25 * dg.DEMAND_GWH[2027])
    assert d30 == pytest.approx(0.75 * dg.DEMAND_GWH[2030] + 0.25 * (2 * dg.DEMAND_GWH[2030] - dg.DEMAND_GWH[2029]))
    assert dg.charging_base_factor(2030) == pytest.approx(1 - dg.UNIT_SHARE + dg.UNIT_SHARE * d26 / d30)
    spread, held = dg.path("central"), dg.path("central", spread=False)
    assert spread["calendar_2030_gbp_2025"] < held["calendar_2030_gbp_2025"]
    # The same revenue path either way: only the base differs.
    for period, year in spread["by_fiscal_year"].items():
        assert year["allowed_revenue_gbp_m_2020_21"] == held["by_fiscal_year"][period]["allowed_revenue_gbp_m_2020_21"]
        assert year["value_gbp_2026"] == pytest.approx(
            held["by_fiscal_year"][period]["value_gbp_2026"] * year["charging_base_factor"])


def test_the_bill_block_is_the_path():
    duos = _duos()
    fy = dg.path("central")["by_fiscal_year"]
    for component in duos["period_blend"]:
        assert component["value"] == pytest.approx(fy[component["period"]]["value_gbp_2026"], abs=1e-6)
        assert component["value_at_ed2_return"] == pytest.approx(fy[component["period"]]["value_at_ed2_return_gbp_2026"], abs=1e-6)
    growth = duos["derivation"]["ed3_cp30_growth"]
    assert growth["cp30_input"]["value_gbp_bn_per_year"] == dg.CP13_DISTRIBUTION_CAPEX_GBP_BN_2024
    assert growth["cp30_input"]["price_basis"].startswith("real 2024")


def test_the_2050_duos_keeps_its_committed_base():
    from scripts.consumer_bill.build_2050_bill_inputs import DUOS_FY2030_31_BASE_2050

    committed = json.loads(Path("data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json")
                           .read_text(encoding="utf-8"))["sourced_externals"]["duos"]["derivation"]
    assert committed["base_fy2030_31_gbp_2026_money"] == DUOS_FY2030_31_BASE_2050
    assert _duos()["period_blend"][1]["value"] != DUOS_FY2030_31_BASE_2050   # the 2030 line itself moved
