"""Task Y2 (FES 2050 plan): the vintage technology cost table and its loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import vintage_costs as vc
from tests.helpers import requires_outputs

ROOT = Path(__file__).resolve().parents[2]
FINANCING = ROOT / "data" / "public_ownership" / "financing"
TECHNOLOGY_COSTS = FINANCING / "technology_costs.csv"
VINTAGE = FINANCING / "technology_costs_vintage.csv"

#: Technologies the FES HT 2050 fleet needs (plan task Y2, deliverable 2).
FES_2050_TECHNOLOGIES = [
    "Offshore wind", "Floating offshore wind", "Onshore wind", "Solar PV", "Large-scale nuclear",
    "Small modular reactors (SMRs)", "Gas generation - CCUS - mature", "Hydrogen CCHT", "Hydrogen OCHT",
    "Biomass - with CCUS - mature", "Lithium batteries - 2h", "Lithium batteries", "Lithium batteries - 8h",
    "Liquid air energy storage - 6h", "Compressed air energy storage - 6h",
    "Pumped hydro energy storage (PHES) - 20h", "Vanadium redox flow battery", "Hydrogen electrolyser",
    "Hydrogen storage - salt cavern", "Tidal stream",
]
#: Declared exceptions to "capex does not rise with build year after 2030".
CAPEX_RISE_EXCEPTIONS = {"Hydrogen CCHT"}  # Annex A: 2030 FOAK 50% blend, 100% H2 from 2035
LEARNING_TECHNOLOGIES = {"Solar PV", "Onshore wind", "Offshore wind", "Floating offshore wind",
                         "Lithium batteries", "Lithium batteries - 1h", "Lithium batteries - 2h",
                         "Lithium batteries - 6h", "Lithium batteries - 8h", "Lithium batteries - 9h"}


@pytest.fixture(scope="module")
def table():
    return vc.load_vintage_costs(VINTAGE, use_cache=False)


@pytest.fixture(scope="module")
def anchors():
    from scripts.public_ownership.build_vintage_technology_costs import read_anchors
    return read_anchors(TECHNOLOGY_COSTS)


def _value(table, technology, year, column="capex_power_gbp_per_kw"):
    return float(table.loc[table["technology"].eq(technology) & table["build_year"].eq(year), column].iloc[0])


# ── 2030 parity (decision 26) ────────────────────────────────────────────────


def test_2030_row_minus_adjustment_reproduces_technology_costs(table, anchors):
    basis = vc.technology_costs_basis(table, 2030).set_index("technology")
    for technology, anchor in anchors.iterrows():
        assert technology in basis.index, technology
        row = basis.loc[technology]
        for column in ("capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh", "fixed_om_gbp_per_kw_year",
                       "variable_om_gbp_per_mwh", "economic_life_years"):
            assert row[column] == pytest.approx(float(anchor[column]), abs=1e-6), (technology, column)
        if pd.notna(anchor["duration_hours"]):
            assert float(row["duration_hours"]) == float(anchor["duration_hours"])
        assert row["price_year"] == anchor["price_year"] and row["real_or_nominal"] == anchor["real_or_nominal"]


def test_declared_adjustments_match_the_research(table):
    adj = vc.basis_adjustments(table, 2030).set_index("technology")
    capex = adj["basis_adjustment_capex_power_gbp_per_kw"]
    fom = adj["basis_adjustment_fixed_om_gbp_per_kw_year"]
    assert capex["Solar PV"] == pytest.approx(6100 / 52, abs=1e-5)  # ~ +117
    assert capex["Onshore wind"] == pytest.approx(16300 / 51.6, abs=1e-5)  # ~ +316
    assert capex["Offshore wind"] == 0.0  # connection excluded by construction (OFTO)
    assert capex["Tidal stream"] == pytest.approx(4430 - 5598.98, abs=1e-6)
    assert fom["Solar PV"] == pytest.approx(1.6) and fom["Onshore wind"] == pytest.approx(3.8)
    assert fom["Offshore wind"] == pytest.approx(8.9)
    for technology in ("Large-scale nuclear", "Lithium batteries", "Hydropower", "Small modular reactors (SMRs)"):
        assert capex[technology] == 0.0 and fom[technology] == 0.0
    assert _value(table, "Tidal stream", 2030) == pytest.approx(4430.0)


def test_desnz_rows_are_on_the_full_annex_a_basis(table):
    # pre-development + construction + infrastructure / plant size, Medium.
    assert _value(table, "Solar PV", 2050) == pytest.approx(30 + 400 + 6100 / 52, abs=1e-5)
    assert _value(table, "Onshore wind", 2050) == pytest.approx(80 + 1100 + 16300 / 51.6, abs=1e-5)
    assert _value(table, "Offshore wind", 2050) == pytest.approx(140 + 2200)
    assert _value(table, "Offshore wind", 2040, "fixed_om_gbp_per_kw_year") == pytest.approx(41.1 + 8.9)
    assert _value(table, "Floating offshore wind", 2030) == pytest.approx(3700.0)
    # Annual values are linear between the published years.
    assert _value(table, "Offshore wind", 2032) == pytest.approx(2670 + 0.4 * (2570 - 2670))


# ── Coverage, values, provenance ─────────────────────────────────────────────


def test_every_technology_has_every_build_year(table, anchors):
    names = set(table["technology"])
    assert set(anchors.index) <= names
    assert set(FES_2050_TECHNOLOGIES) <= names
    counts = table.groupby("technology")["build_year"].agg(["min", "max", "count"])
    assert (counts["min"] == 2025).all() and (counts["max"] == 2050).all() and (counts["count"] == 26).all()


def test_values_are_finite_and_non_negative(table):
    assert vc.is_finite_table(table)
    for column in ("capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh", "fixed_om_gbp_per_kw_year",
                   "variable_om_gbp_per_mwh", "economic_life_years", "learning_capex_gbp_per_kw",
                   "insurance_gbp_per_kw_year", "fixed_om_gbp_per_kwh_year"):
        assert (table[column] >= 0).all(), column
    assert (table["economic_life_years"] > 0).all()
    cost = table["capex_power_gbp_per_kw"] + table["capex_energy_gbp_per_kwh"]
    assert (cost > 0).all()


def test_capex_does_not_rise_after_2030_except_declared(table):
    for technology, rows in table[table["build_year"] >= 2030].groupby("technology"):
        rows = rows.sort_values("build_year")
        for column in ("capex_power_gbp_per_kw", "capex_energy_gbp_per_kwh"):
            steps = np.diff(rows[column].to_numpy())
            if technology in CAPEX_RISE_EXCEPTIONS:
                continue
            assert (steps <= 1e-9).all(), (technology, column)
    ccht = table[table["technology"].eq("Hydrogen CCHT")].set_index("build_year")["capex_power_gbp_per_kw"]
    assert ccht[2035] > ccht[2030] and ccht.loc[2035:].nunique() == 1


def test_flat_after_last_source_year(table):
    for technology in ("Large-scale nuclear", "Hydrogen storage - salt cavern"):
        rows = table[table["technology"].eq(technology)]
        assert rows["capex_power_gbp_per_kw"].nunique() == 1
        assert rows["capex_energy_gbp_per_kwh"].nunique() == 1
    assert _value(table, "Hydrogen storage - salt cavern", 2040, "capex_energy_gbp_per_kwh") == pytest.approx(
        0.54432 * 0.97727, abs=1e-4)


def test_every_row_has_provenance(table):
    for column in ("source", "source_page", "status", "anchor_rule", "year_basis", "basis_adjustment_note"):
        assert table[column].notna().all() and table[column].astype(str).str.strip().ne("").all(), column
    assert set(table["status"]) <= {"reviewed", "provisional"}
    assert (table["price_year"] == 2024).all() and (table["real_or_nominal"] == "real").all()
    provisional = set(table.loc[table["status"].eq("provisional"), "technology"])
    assert {"Vanadium redox flow battery", "Hydrogen storage - salt cavern",
            "Novel long duration energy storage (LDES)"} <= provisional


def test_ratio_anchoring_of_fes_rows(table, anchors):
    # BECCS: technology_costs.csv 2030 x FES ER.01 2035/2030 (4,127.134 / 4,519.02).
    expected = float(anchors.loc["Biomass - with CCUS - mature", "capex_power_gbp_per_kw"]) * 4127.134 / 4519.02
    assert _value(table, "Biomass - with CCUS - mature", 2035) == pytest.approx(expected, rel=1e-5)
    # Battery family anchored on the 4 h row: 2 h / 4 h = FES 406.83 / 654.01 in 2030.
    assert _value(table, "Lithium batteries - 2h", 2030) == pytest.approx(732.67 * 406.83 / 654.01, rel=1e-5)
    assert _value(table, "Lithium batteries - 8h", 2030) > _value(table, "Lithium batteries - 6h", 2030)


def test_electrolyser_is_per_kw_electrical(table):
    expected = 2152.52 * (72.27 / 88.71) * 0.97727
    assert _value(table, "Hydrogen electrolyser", 2030) == pytest.approx(expected, rel=1e-4)


def test_learning_column_below_central_after_2030(table):
    for technology, rows in table.groupby("technology"):
        rows = rows.set_index("build_year")
        central, learning = rows["capex_power_gbp_per_kw"], rows["learning_capex_gbp_per_kw"]
        assert (learning.loc[:2030] == central.loc[:2030]).all(), technology
        if technology in LEARNING_TECHNOLOGIES:
            assert (learning.loc[2031:] < central.loc[2031:]).all(), technology
        else:
            assert (learning == central).all(), technology
    ratio = _value(table, "Solar PV", 2050, "learning_capex_gbp_per_kw") / _value(table, "Solar PV", 2050)
    assert ratio == pytest.approx(0.8 ** 2)  # two declared doublings 2030-2050 at 20%


# ── Loader ───────────────────────────────────────────────────────────────────


def test_costs_for_matches_names_like_the_finance_layer(table):
    row = vc.costs_for("offshore  wind", 2040, table=table)
    assert row["technology"] == "Offshore wind" and row["build_year"] == 2040
    assert row["capex_power_gbp_per_kw"] == pytest.approx(2460.0)
    assert vc.costs_for("Biomass — with CCUS — mature", 2030, table=table)["technology"] == "Biomass - with CCUS - mature"


def test_costs_for_holds_flat_after_2050_and_rejects_before_2025(table):
    later = vc.costs_for("Solar PV", 2060, table=table)
    assert later["held_flat"] and later["capex_power_gbp_per_kw"] == _value(table, "Solar PV", 2050)
    assert not vc.costs_for("Solar PV", 2050, table=table)["held_flat"]
    with pytest.raises(ValueError):
        vc.costs_for("Solar PV", 2024, table=table)
    with pytest.raises(KeyError):
        vc.costs_for("Fusion", 2040, table=table)


def test_costs_for_learning_switch(table):
    central = vc.costs_for("Onshore wind", 2045, table=table)
    learning = vc.costs_for("Onshore wind", 2045, learning=True, table=table)
    assert learning["capex_case"] == "learning" and central["capex_case"] == "central"
    assert learning["capex_power_gbp_per_kw"] < central["capex_power_gbp_per_kw"]


def test_vintage_cost_table_is_drop_in_for_the_finance_layer(table):
    from scripts.public_ownership.calculate_financing_scenarios import REQUIRED_COST_COLUMNS, _validated_costs

    rows = vc.vintage_cost_table(2045, table=table)
    assert list(rows.columns[:len(vc.TECHNOLOGY_COST_COLUMNS)]) == list(vc.TECHNOLOGY_COST_COLUMNS)
    assert REQUIRED_COST_COLUMNS <= set(rows.columns)
    accepted = _validated_costs(rows, price_year=2024, rate_basis="real_cpi")
    assert len(accepted) == len(rows) and rows["technology"].is_unique


def test_insurance_exclusion_removes_only_insurance(table):
    included = vc.vintage_cost_table(2030, table=table).set_index("technology")
    excluded = vc.vintage_cost_table(2030, insurance="exclude", table=table).set_index("technology")
    assert excluded.loc["Offshore wind", "fixed_om_gbp_per_kw_year"] == pytest.approx(46.0)
    assert excluded.loc["Lithium batteries", "fixed_om_gbp_per_kw_year"] == included.loc[
        "Lithium batteries", "fixed_om_gbp_per_kw_year"]
    with pytest.raises(ValueError):
        vc.vintage_cost_table(2030, insurance="both", table=table)


def test_loader_rejects_a_row_without_provenance(tmp_path, table):
    broken = table.drop(columns=["_key"]).copy()
    broken.loc[0, "source"] = ""
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source"):
        vc.load_vintage_costs(path, use_cache=False)


def test_hurdle_rates_carry_the_neso_2035_wind_step():
    rates = vc.load_hurdle_rates()
    assert vc.hurdle_rate("Offshore wind", 2030, table=rates) == pytest.approx(0.083)
    assert vc.hurdle_rate("Offshore wind", 2035, table=rates) == pytest.approx(0.063)
    assert vc.hurdle_rate("Onshore wind", 2040, table=rates) == pytest.approx(0.052)
    assert vc.hurdle_rate("Floating offshore wind", 2050, table=rates) == pytest.approx(0.078)
    # DESNZ steps between published commissioning years.
    assert vc.hurdle_rate("Floating offshore wind", 2033, source="DESNZ", table=rates) == pytest.approx(0.114)
    assert vc.hurdle_rate("Floating offshore wind", 2035, source="DESNZ", table=rates) == pytest.approx(0.101)
    assert vc.hurdle_rate("Offshore wind", 2027, source="DESNZ", case="low", table=rates) == pytest.approx(0.073)


# ── Rebuild from the sources (slow: reads the workbooks) ─────────────────────


@pytest.mark.slow
@requires_outputs("data/fes_2050_sources/fes_econ_inputs.xlsx")
def test_builder_reproduces_the_committed_table(table):
    from scripts.public_ownership import build_vintage_technology_costs as builder

    deflators = builder.read_fes_deflators(builder.DEFAULT_FES_INPUTS)
    assert deflators["to_2024"] == pytest.approx(0.97727, abs=1e-5)
    rebuilt = builder.build_vintage_table()
    committed = table.drop(columns=["_key"])
    text = [c for c in committed.columns if committed[c].dtype == object or rebuilt[c].dtype == object]
    rebuilt[text] = rebuilt[text].replace("", np.nan)
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), committed.reset_index(drop=True),
                                  check_dtype=False, atol=1e-6)
