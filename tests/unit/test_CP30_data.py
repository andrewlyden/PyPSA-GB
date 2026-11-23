from pathlib import Path

import pandas as pd
import pytest

from scripts.CP30.CP30_data import (
    EXPECTED_PATHWAYS,
    build_validation_report,
    normalize_columns,
    read_cp30_core_tables,
    read_reference_tables,
    year_columns,
)


CP30_WORKBOOK = Path("data/CP30/Clean Power 2030 Data Workbook_V003.xlsx")


def test_normalize_columns_converts_excel_years():
    df = pd.DataFrame({2030.0: [1], "  Annual/   Peak/ Flex  ": ["Annual"], "2035.0": [2]})

    result = normalize_columns(df)

    assert "2030" in result.columns
    assert "2035" in result.columns
    assert "Annual/ Peak/ Flex" in result.columns


@pytest.mark.skipif(not CP30_WORKBOOK.exists(), reason="CP30 workbook not available")
def test_cp30_core_tables_have_expected_pathways():
    tables = read_cp30_core_tables(CP30_WORKBOOK)

    es1_pathways = set(tables["es1"]["Pathway"].dropna().unique())
    es5_pathways = set(tables["es5"]["Scenario"].dropna().unique())
    ed1_pathways = set(tables["ed1"]["Pathway"].dropna().unique())

    assert EXPECTED_PATHWAYS.issubset(es1_pathways)
    assert EXPECTED_PATHWAYS.issubset(es5_pathways)
    assert EXPECTED_PATHWAYS.issubset(ed1_pathways)


@pytest.mark.skipif(not CP30_WORKBOOK.exists(), reason="CP30 workbook not available")
def test_cp30_core_tables_have_2030_data():
    tables = read_cp30_core_tables(CP30_WORKBOOK)

    assert "2030" in year_columns(tables["es1"])
    assert "2030" in year_columns(tables["ed1"])
    assert "2030" in tables["es1"].columns


@pytest.mark.skipif(not CP30_WORKBOOK.exists(), reason="CP30 workbook not available")
def test_cp30_es1_has_required_model_inputs():
    es1 = read_cp30_core_tables(CP30_WORKBOOK)["es1"]
    variables = set(es1["Variable"].dropna().unique())

    assert "Capacity (MW)" in variables
    assert "Demand Total (GWh) - Excluding Electrolysis" in variables
    assert "Storage Capacity (GWh)" in variables
    assert "CO2 Emissions (Mt)" in variables


@pytest.mark.skipif(not CP30_WORKBOOK.exists(), reason="CP30 workbook not available")
def test_cp30_es5_has_expected_regional_technologies():
    es5 = read_cp30_core_tables(CP30_WORKBOOK)["es5"]
    technologies = set(es5["Technology"].dropna().unique())

    assert {"Solar", "Offshore Wind", "Onshore Wind", "Batteries", "Storage"}.issubset(technologies)


@pytest.mark.skipif(not CP30_WORKBOOK.exists(), reason="CP30 workbook not available")
def test_cp30_validation_report_passes_core_checks():
    tables = read_cp30_core_tables(CP30_WORKBOOK)
    reference = read_reference_tables(CP30_WORKBOOK)
    report = build_validation_report(CP30_WORKBOOK, tables, reference)

    failed = report[(report["kind"] == "check") & (report["status"] == "FAIL")]
    assert failed.empty, failed.to_string(index=False)
