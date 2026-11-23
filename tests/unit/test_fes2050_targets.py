"""FES 2025 energy targets (task Y1): mapping, units, fiscal to calendar, hydrogen, CP30 parity."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.CP30.energy_validation import (
    build_energy_validation,
    cp30_energy_targets,
    energy_targets,
    format_energy_validation,
    target_column,
)
from scripts.FES.fes2025_targets import (
    FES2025_WORKBOOK,
    classify_es1,
    fiscal_to_calendar,
    split_variable,
    to_gwh,
    to_mw,
    unmapped_fes_lines,
)

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / FES2025_WORKBOOK
CP30_ES1 = ROOT / "resources" / "CP30" / "CP30_ES1.csv"
HT = "Holistic Transition"

needs_workbook = pytest.mark.skipif(not WORKBOOK.exists(), reason="FES 2025 workbook not present")


# ─────────────────────────────── pure helpers ───────────────────────────────

def test_split_variable_and_units():
    assert split_variable("Generation (TWh)") == ("Generation", "TWh")
    assert split_variable("Storage Capacity (GWh)") == ("Storage Capacity", "GWh")
    assert split_variable("Curtailment including DACCS (TWh)") == ("Curtailment including DACCS", "TWh")
    assert to_gwh(1.5, "TWh") == 1500.0
    assert to_gwh(7.0, "GWh") == 7.0
    assert to_mw(2.0, "GW") == 2000.0
    with pytest.raises(ValueError):
        to_gwh(1.0, "Mt")


def test_fiscal_to_calendar_rule():
    # HT 2050 system demand, ED1 row 9: FY2049 700,725 and FY2050 705,223 GWh -> 704.1 TWh.
    assert fiscal_to_calendar({2049: 700725.0, 2050: 705223.0}, 2050) == pytest.approx(704098.5)
    with pytest.raises(ValueError):
        fiscal_to_calendar({2050: 1.0}, 2050)


def _es1(rows):
    columns = ["Connection", "Pathway", "Variable", "Category", "Type", "SubType", "2050", "excel_row"]
    return pd.DataFrame([dict(zip(columns, r)) for r in rows])


def test_classification_maps_units_and_excludes_off_grid():
    es1 = _es1([
        ("Transmission", HT, "Capacity (MW)", "Renewable", "Offshore Wind", "Offshore Wind", 1000.0, 1),
        ("Transmission", HT, "Generation (TWh)", "Renewable", "Offshore Wind", "Offshore Wind", 4.0, 2),
        ("Both", HT, "Curtailment (TWh)", "Renewable", "Offshore Wind", "Offshore Wind", 0.5, 3),
        ("Non-Networked", HT, "Capacity (MW)", "Renewable", "Offshore Wind", "Offshore Wind", 40.0, 4),
        ("Transmission", HT, "Export (TWh)", "Interconnectors", "Interconnectors", "Interconnectors", -2.0, 5),
        ("Transmission", HT, "Capacity (MW)", "Base Load", "Nuclear", "Nuclear - Small", 500.0, 6),
        ("Transmission", HT, "Capacity (MW)", "Dispatchable", "Gas", "CCGT", 0.0, 7),
    ])
    lines = classify_es1(es1, HT, 2050).set_index("excel_row")
    assert 7 not in lines.index                                     # zero lines are not listed
    assert lines.loc[2, "value_std"] == pytest.approx(4000.0)       # TWh -> GWh
    assert lines.loc[3, ["role", "carrier"]].tolist() == ["supply", "wind_offshore"]
    assert lines.loc[4, "role"] == "excluded"
    assert lines.loc[5, "value_std"] == pytest.approx(2000.0)       # exports positive
    assert lines.loc[6, "carrier"] == "nuclear"
    assert unmapped_fes_lines(lines.reset_index()).empty


def test_unknown_line_is_unmapped():
    es1 = _es1([
        ("Transmission", HT, "Capacity (MW)", "Dispatchable", "Fusion", "Fusion", 100.0, 1),
        ("Transmission", HT, "Capacity (MW)", "Storage", "Long Duration Energy Storage", "Flywheel", 5.0, 2),
        ("Both", HT, "Mystery (TWh)", "Other", "X", "Y", 1.0, 3),
    ])
    unmapped = unmapped_fes_lines(classify_es1(es1, HT, 2050))
    assert sorted(unmapped["excel_row"]) == [1, 2, 3]
    assert {"Connection", "Variable", "Type", "SubType"} <= set(unmapped.columns)


# ────────────────────────────── the real workbook ───────────────────────────

@pytest.fixture(scope="module")
def ht2050():
    if not WORKBOOK.exists():
        pytest.skip("FES 2025 workbook not present")
    return energy_targets("FES2025", WORKBOOK, HT, 2050, strict=True)


@needs_workbook
def test_every_es1_line_is_mapped_ht_2050_and_2030(ht2050):
    assert ht2050["unmapped"].empty
    t2030 = energy_targets("FES2025", WORKBOOK, HT, 2030, include_building_blocks=False, strict=True)
    assert t2030["unmapped"].empty
    lines = ht2050["es1_lines"]
    assert set(lines["role"]) <= {"supply", "excluded", "storage", "interconnector", "daccs", "dsr", "information"}


@needs_workbook
def test_ht_2050_supply_in_gw_and_twh(ht2050):
    gen = ht2050["generation"]
    cap_gw = gen["capacity_mw"] / 1e3
    gen_twh = gen["generation_gwh"] / 1e3
    assert cap_gw["wind_offshore"] == pytest.approx(103.93, abs=0.01)      # grid only; +0.44 GW off-grid
    assert gen_twh["wind_offshore"] == pytest.approx(395.51, abs=0.01)
    assert cap_gw["wind_onshore"] == pytest.approx(47.45, abs=0.01)
    assert cap_gw["solar_pv"] == pytest.approx(97.04, abs=0.01)
    assert cap_gw["nuclear"] == pytest.approx(14.20, abs=0.01)
    assert gen_twh["nuclear"] == pytest.approx(93.52, abs=0.01)
    assert cap_gw["CCS Gas"] == pytest.approx(22.205, abs=0.001)
    assert cap_gw["Hydrogen"] == pytest.approx(26.08, abs=0.01)
    assert gen_twh["Hydrogen"] == pytest.approx(10.40, abs=0.01)
    assert cap_gw["CCS Biomass"] == pytest.approx(2.662, abs=0.001)
    assert "gas" not in gen.index or gen.loc["gas", "capacity_mw"] == 0.0  # no unabated gas in HT 2050
    assert gen["curtailment_gwh"].sum() / 1e3 == pytest.approx(63.69, abs=0.01)
    assert gen.loc["wind_onshore", "gross_gwh"] / 1e3 == pytest.approx(118.99, abs=0.01)
    excluded = ht2050["excluded"]
    assert excluded["connection"].eq("Non-Networked").all()
    assert ht2050["daccs_gwh"] / 1e3 == pytest.approx(17.83, abs=0.01)
    assert ht2050["interconnectors"]["capacity_mw"] == pytest.approx(21800.0)
    assert ht2050["export_gwh"] - ht2050["import_gwh"] == pytest.approx(17824.8, abs=0.1)
    storage = ht2050["storage"]
    assert storage["capacity_mw"].sum() / 1e3 == pytest.approx(55.85, abs=0.01)
    assert storage["energy_gwh"].sum() == pytest.approx(204.66, abs=0.01)


@needs_workbook
def test_ht_2050_demand_fiscal_to_calendar_and_electrolysis(ht2050):
    totals = ht2050["demand"]["totals"]
    system_fy = totals["system_excluding_electrolysis"]["fy_gwh"] + totals["electrolysis"]["fy_gwh"]
    system_cal = totals["system_excluding_electrolysis"]["calendar_gwh"] + totals["electrolysis"]["calendar_gwh"]
    assert system_fy / 1e3 == pytest.approx(705.22, abs=0.01)
    assert system_cal / 1e3 == pytest.approx(704.10, abs=0.01)
    assert totals["system_excluding_electrolysis"]["fy_gwh"] / 1e3 == pytest.approx(616.51, abs=0.01)
    assert totals["electrolysis"]["fy_gwh"] / 1e3 == pytest.approx(88.71, abs=0.01)
    assert totals["consumer_excluding_electrolysis"]["fy_gwh"] / 1e3 == pytest.approx(567.38, abs=0.01)
    # The validation target: consumer basis, calendar year, electrolysis separate.
    assert ht2050["demand_gwh"] == pytest.approx(totals["consumer_excluding_electrolysis"]["calendar_gwh"])
    assert ht2050["electrolysis_gwh"] == pytest.approx(totals["electrolysis"]["calendar_gwh"])
    assert ht2050["demand"]["peak_gw"]["GBFES Peak Customer Demand: Total Consumption plus Losses"]["gw"] == \
        pytest.approx(120.05)


@needs_workbook
def test_ht_2050_demand_components(ht2050):
    comp = ht2050["demand_components"]
    bb = comp["building_blocks"]
    assert (bb["Dem_BB004"]["fy_gwh"] + bb["Dem_BB005"]["fy_gwh"]) / 1e3 == pytest.approx(78.90, abs=0.01)
    evs_fy = (bb["Dem_BB006"]["fy_gwh"] + bb["Dem_BB007"]["fy_gwh"]) / 1e3
    assert evs_fy == pytest.approx(126.11, abs=0.01)
    assert evs_fy * 1e3 == pytest.approx(ht2050["demand"]["totals"]["evs"]["fy_gwh"], rel=1e-3)   # = ED1 rows 49 + 104
    assert bb["Dem_BB004"]["fy_gwh"] == pytest.approx(ht2050["demand"]["annual"]["residential_heat_pumps"]["fy_gwh"],
                                                      rel=1e-6)
    assert bb["Dem_BB009"]["mw"] == pytest.approx(15660.0, abs=1.0)
    assert comp["data_centres_gwh"] / 1e3 == pytest.approx(0.75 * 65.317 + 0.25 * 64.101, abs=0.001)
    assert comp["district_heat_residential_gwh"] / 1e3 == pytest.approx(0.75 * 19.598 + 0.25 * 18.934, abs=0.001)
    # Residual = consumer total less heat pumps and EVs (plan Y6), about 362 TWh.
    assert comp["residual_gwh"] / 1e3 == pytest.approx(361.5, abs=0.1)


@needs_workbook
def test_ht_2050_hydrogen_balance(ht2050):
    h2 = ht2050["hydrogen"]
    assert h2["electrolysis_capacity_gw_electric"] == pytest.approx(15.66)
    assert h2["annual_twh"]["supply"]["Networked Electrolysis"] == pytest.approx(72.27)
    assert h2["gas_reforming_ccs"]["capacity_gw_h2"] == pytest.approx(5.38)
    assert h2["gas_reforming_ccs"]["output_twh"] == pytest.approx(44.51)
    assert h2["non_power_demand_twh"] == pytest.approx(92.23, abs=0.01)
    assert h2["total_demand_twh"] == pytest.approx(119.56, abs=0.01)
    assert h2["storage_twh"] == pytest.approx(11.67)
    assert h2["h2_turbines"]["capacity_mw"] / 1e3 == pytest.approx(26.08, abs=0.01)
    # Electrolysis alone (~112 TWh) cannot meet ~119.6 TWh of demand; with reforming it can.
    assert h2["feasibility"]["electrolysis_max_twh_h2"] == pytest.approx(111.76, abs=0.05)
    assert h2["feasibility"]["electrolysis_max_twh_h2"] < h2["total_demand_twh"]
    assert h2["feasibility"]["electrolysis_plus_reforming_max_twh_h2"] > h2["total_demand_twh"]


@needs_workbook
def test_fes_targets_feed_the_energy_validation(ht2050):
    gens = pd.DataFrame({"carrier": ["wind_offshore", "nuclear", "overlay"], "bus": ["a", "b", "c"],
                         "p_nom": [1.0, 1.0, 1.0],
                         "available_mwh": [ht2050["generation"].loc["wind_offshore", "gross_gwh"] * 1e3, 1e8, 1e6],
                         "dispatched_mwh": [np.nan, np.nan, np.nan]}, index=["g1", "g2", "g3"])
    tables = {"generators": gens, "storage_dispatch_mwh": pd.Series(dtype=float),
              "demand_mwh": ht2050["demand_gwh"] * 1e3, "load_shedding_mwh": 0.0, "import_mwh": 0.0,
              "export_mwh": 0.0, "electrolysis_mwh": 0.0, "h2_turbine_mwh": 0.0, "hours": 8760.0}
    table = build_energy_validation(ht2050, tables, "build")
    assert target_column(table) == "target_gwh" and "cp30_gwh" not in table.columns
    offshore = table[table["item"].eq("wind_offshore")].iloc[0]
    assert offshore["status"] == "PASS"
    assert table.loc[table["component"].eq("demand"), "status"].eq("PASS").all()
    assert not table["status"].eq("FAIL").any()
    assert "model carrier with no FES 2025 ES1 generation line (e.g. committed overlay)" in set(table["note"])
    assert table.loc[table["component"].eq("demand"), "es1_lines"].iloc[0] == ht2050["demand_line"]
    assert "target_TWh" in format_energy_validation(table)


# ─────────────────────────────── CP30 unchanged ─────────────────────────────

@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 not built")
def test_cp30_targets_identical_through_the_dispatcher():
    for pathway in pd.read_csv(CP30_ES1)["Pathway"].dropna().unique():
        direct = cp30_energy_targets(CP30_ES1, pathway, 2030)
        routed = energy_targets("CP30", CP30_ES1, pathway, 2030)
        assert direct.keys() == routed.keys()
        pd.testing.assert_frame_equal(direct["generation"], routed["generation"], check_exact=True)
        pd.testing.assert_frame_equal(direct["unmapped"], routed["unmapped"], check_exact=True)
        for key in ("storage_gwh", "demand_gwh", "electrolysis_gwh", "import_gwh", "export_gwh"):
            assert direct[key] == routed[key]
    ffr = cp30_energy_targets(CP30_ES1, "Further Flex and Renewables", 2030)
    assert ffr["demand_gwh"] == pytest.approx(315161.95)            # the published CP30 2030 demand
    assert "source_label" not in ffr                                 # CP30 targets carry no new keys


@pytest.mark.skipif(not CP30_ES1.exists(), reason="CP30 ES1 not built")
def test_cp30_validation_keeps_cp30_column_and_notes():
    targets = cp30_energy_targets(CP30_ES1, "Further Flex and Renewables", 2030)
    gens = pd.DataFrame({"carrier": ["extra"], "bus": ["a"], "p_nom": [1.0], "available_mwh": [1e5],
                         "dispatched_mwh": [1e5]}, index=["g"])
    tables = {"generators": gens, "storage_dispatch_mwh": pd.Series({"Other": 1e4}), "demand_mwh": 3.15e8,
              "load_shedding_mwh": 0.0, "import_mwh": 0.0, "export_mwh": 0.0, "electrolysis_mwh": 0.0,
              "h2_turbine_mwh": 0.0, "hours": 8760.0}
    table = build_energy_validation(targets, tables, "dispatch")
    assert "cp30_gwh" in table.columns and "target_gwh" not in table.columns
    notes = set(table["note"])
    assert "model carrier with no CP30 ES1 generation line (e.g. committed overlay)" in notes
    assert "unserved energy; CP30 has none" in notes
    assert "model storage carrier with no CP30 ES1 line" in notes
    assert table.loc[table["item"].eq("Electricity demand excluding electrolysis"), "es1_lines"].iloc[0] == \
        "Demand Total (GWh) - Excluding Electrolysis"
    assert "cp30_TWh" in format_energy_validation(table)


def test_unknown_source_fails():
    with pytest.raises(ValueError):
        energy_targets("FES2019", "x", HT, 2050)
    with pytest.raises(TypeError):
        energy_targets("CP30", "x", HT, 2030, strict=True)
