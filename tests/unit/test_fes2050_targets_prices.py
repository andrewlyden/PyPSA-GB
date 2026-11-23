"""FES commodity prices (task Y1): CP1/CP2 parsing, conversion to GBP real 2024, no fallback."""

from pathlib import Path

import pytest

from scripts.FES.fes_commodity_prices import (
    CPIH_INDEX,
    FESPriceError,
    future_cost_source,
    parse_price_blocks,
    read_fes_commodity_prices,
    select_case,
    standard_price_tables,
    to_model_money,
)

ROOT = Path(__file__).resolve().parents[2]
FES2025 = ROOT / "data" / "FES" / "Future Energy Scenarios 2025 Data Workbook V005_0.xlsx"
FES2024 = ROOT / "data" / "FES" / "Future Energy Scenarios 2024 Data Workbook_V006_0.xlsx"
needs_2025 = pytest.mark.skipif(not FES2025.exists(), reason="FES 2025 workbook not present")


def _rows(table):
    return [(i, tuple(r)) for i, r in enumerate(table, start=1)]


def test_parse_block_layout_and_block_end():
    rows = _rows([
        (None, "Wholesale NBP gas price (2025 EUR/MWh)"),
        (None, "NBP Gas Price (EUR/MWh)"),
        (),
        (None, None, "Year", 2049, 2050),
        (None, None, "Base Case", 28.73, 28.72),
        (),
        (None, None, "All prices converted to Euro/MWh"),
        (None, None, "Conversion Rates", 1.0, 2.0),
    ])
    block = parse_price_blocks(rows, "CP1")
    assert len(block) == 2                                   # the text after the blank row is not a case
    assert set(block["case"]) == {"Base Case"}
    assert block.set_index("year").loc[2050, "value"] == 28.72
    assert block["price_year"].iloc[0] == 2025 and block["currency"].iloc[0] == "EUR"


def test_title_without_price_basis_fails():
    with pytest.raises(FESPriceError):
        parse_price_blocks(_rows([(None, "Wholesale NBP gas price (EUR/MWh)")]), "CP1")


def test_case_selection_does_not_fall_back():
    rows = _rows([
        (None, "Wholesale NBP gas price (2024 EUR/MWh)"),
        (None, "Year", 2050),
        (None, "High Case", 47.0),
        (None, "Low Case", 15.0),
    ])
    with pytest.raises(FESPriceError, match="no 'Base Case' row"):
        select_case(parse_price_blocks(rows, "CP1"))


def test_midpoint_is_the_mean_of_high_and_low_only_when_asked():
    rows = _rows([
        (None, "Wholesale NBP gas price (2024 EUR/MWh)"),
        (None, "Year", 2030, 2050),
        (None, "High Case", 47.0, 40.0),
        (None, "Low Case", 15.0, 10.0),
    ])
    picked = select_case(parse_price_blocks(rows, "CP1"), missing_case="midpoint")
    assert picked.set_index("year")["value"].to_dict() == {2030: 31.0, 2050: 25.0}
    assert picked["case"].str.startswith("midpoint of High Case (row 3) and Low Case (row 4)").all()
    with pytest.raises(FESPriceError, match="missing_case"):
        select_case(parse_price_blocks(rows, "CP1"), missing_case="mean")


@needs_2025
def test_fes2025_published_values_and_conversion():
    read = read_fes_commodity_prices(FES2025)
    prices = read["prices"].set_index(["commodity", "year"])
    assert prices.loc[("gas", 2050), "value"] == pytest.approx(28.72)
    assert prices.loc[("carbon_gb_total", 2050), "value"] == pytest.approx(134.61, abs=0.005)
    assert prices.loc[("coal", 2050), "value"] == pytest.approx(6.74)
    assert prices.loc[("oil", 2050), "value"] == pytest.approx(30.21)
    assert prices.loc[("gas", 2050), "excel_row"] == 10 and prices.loc[("carbon_gb_total", 2050), "excel_row"] == 10
    conv = read["conversion"].set_index("year")
    assert conv.loc[2025, "gbp_to_eur"] == pytest.approx(1.17)
    assert conv.loc[2024, "gdp_deflator_rebased"] == pytest.approx(0.976)

    model = to_model_money(read["prices"], read["conversion"]).set_index(["commodity", "year"])
    cpih = CPIH_INDEX[2025] / CPIH_INDEX[2024]
    assert model.loc[("gas", 2050), "value_gbp_price_year"] == pytest.approx(28.72 / 1.17)          # 24.55
    assert model.loc[("gas", 2050), "value_model_gbp"] == pytest.approx(28.72 / 1.17 / cpih)        # 23.64
    assert model.loc[("carbon_gb_total", 2050), "value_model_gbp"] == pytest.approx(110.80, abs=0.01)
    assert model.loc[("gas", 2050), "value_model_gbp_gdp_deflator"] == pytest.approx(28.72 / 1.17 * 0.976)

    fuel, carbon = standard_price_tables(to_model_money(read["prices"], read["conversion"]))
    assert {"year", "fuel", "price_gbp_per_mwh_thermal"} <= set(fuel.columns)
    assert set(fuel["fuel"]) == {"gas", "coal", "oil"}                 # no invented biomass price
    assert {"year", "carbon_price_gbp_per_tco2"} <= set(carbon.columns)
    placeholder = {2030: 24.0, 2050: 32.0}                             # the old fallback values
    for year, value in placeholder.items():
        gas = fuel[(fuel["fuel"] == "gas") & (fuel["year"] == year)]["price_gbp_per_mwh_thermal"].iloc[0]
        assert gas != pytest.approx(value, abs=0.5)


@needs_2025
def test_future_cost_source_schema():
    read = read_fes_commodity_prices(FES2025)
    payload = future_cost_source(to_model_money(read["prices"], read["conversion"]), 2050, FES2025)
    assert payload["price_year"] == 2024 and payload["currency"] == "GBP"
    assert payload["assumptions"]["fuel_prices"]["gas"] == pytest.approx(23.64, abs=0.01)
    assert set(payload["assumptions"]["carbon"]) == {"uk_ets", "carbon_price_support"}
    assert payload["assumptions"]["carbon"]["carbon_price_support"] == 0.0


@pytest.mark.skipif(not FES2024.exists(), reason="FES 2024 workbook not present")
def test_fes2024_fails_loudly_without_a_base_case_for_gas():
    with pytest.raises(FESPriceError, match="gas"):
        read_fes_commodity_prices(FES2024)


def test_missing_sheet_fails(tmp_path):
    import openpyxl

    path = tmp_path / "empty.xlsx"
    openpyxl.Workbook().save(path)
    with pytest.raises(FESPriceError):
        read_fes_commodity_prices(path)
