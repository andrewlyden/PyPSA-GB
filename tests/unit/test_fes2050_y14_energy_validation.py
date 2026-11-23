"""FES 2050 plan, task Y14: FES 2025's ES1 Waste line includes the renewable engines.

For FES 2025 targets the waste row compares waste_to_energy + biogas (an ES1 Other
Renewables line, when present, folds into the same row); CP30 targets keep biogas on
its own "Other Renewables" row exactly as before.
"""

import pandas as pd
import pytest

from scripts.CP30.energy_validation import build_energy_validation


def _targets(source=None, biogas_line=False):
    rows = {"waste_to_energy": {"generation_gwh": 9_100.0, "curtailment_gwh": 0.0, "gross_gwh": 9_100.0,
                                "es1_lines": "Waste / Waste; Waste / Waste CHP"}}
    if biogas_line:
        rows["biogas"] = {"generation_gwh": 400.0, "curtailment_gwh": 0.0, "gross_gwh": 400.0,
                          "es1_lines": "Other Renewables / Other Renewables"}
    generation = pd.DataFrame.from_dict(rows, orient="index")
    generation.index.name = "carrier"
    targets = {"generation": generation, "storage_gwh": {}, "demand_gwh": 1_000.0, "electrolysis_gwh": 0.0,
               "import_gwh": 0.0, "export_gwh": 0.0,
               "unmapped": pd.DataFrame(columns=["Type", "SubType", "Variable", "Connection"])}
    if source:
        targets.update(source=source, source_label="FES 2025")
    return targets


def _tables():
    gens = pd.DataFrame({"carrier": ["waste_to_energy", "biogas"], "bus": ["A", "B"], "p_nom": [2_600.0, 1_000.0],
                         "available_mwh": [16e6, 8e6], "dispatched_mwh": [6.0e6, 3.0e6]}, index=["W", "G"])
    return {"generators": gens, "storage_dispatch_mwh": pd.Series(dtype=float), "demand_mwh": 1_000e3,
            "load_shedding_mwh": 0.0, "import_mwh": 0.0, "export_mwh": 0.0, "electrolysis_mwh": 0.0,
            "h2_turbine_mwh": 0.0, "hours": 8760.0, "demand_components_mwh": pd.Series(dtype=float)}


def _row(table, component, item):
    rows = table.loc[table["component"].eq(component) & table["item"].eq(item)]
    assert len(rows) == 1, (component, item, table[["component", "item"]].to_string())
    return rows.iloc[0]


def test_fes2025_waste_compares_waste_plus_biogas():
    table = build_energy_validation(_targets("FES2025"), _tables(), stage="dispatch")
    row = _row(table, "dispatched_energy", "waste_to_energy (waste_to_energy + biogas)")
    assert row["target_gwh"] == 9_100.0 and row["model_gwh"] == pytest.approx(9_000.0)
    assert not table["item"].eq("biogas").any()          # no model_only_carrier row for biogas
    build = build_energy_validation(_targets("FES2025"), _tables(), stage="build")
    assert _row(build, "available_energy", "waste_to_energy (waste_to_energy + biogas)")["model_gwh"] == \
        pytest.approx(24_000.0)


def test_an_fes2025_other_renewables_line_folds_into_the_waste_row():
    table = build_energy_validation(_targets("FES2025", biogas_line=True), _tables(), stage="dispatch")
    row = _row(table, "dispatched_energy", "waste_to_energy (waste_to_energy + biogas)")
    assert row["target_gwh"] == pytest.approx(9_500.0)
    assert "Other Renewables / Other Renewables" in row["es1_lines"]
    assert not table["item"].eq("biogas").any()


def test_cp30_targets_keep_biogas_on_its_own_line():
    table = build_energy_validation(_targets(None, biogas_line=True), _tables(), stage="dispatch")
    assert _row(table, "dispatched_energy", "waste_to_energy")["model_gwh"] == pytest.approx(6_000.0)
    assert _row(table, "dispatched_energy", "biogas")["cp30_gwh"] == 400.0
    without = build_energy_validation(_targets(None), _tables(), stage="dispatch")
    assert _row(without, "model_only_carrier", "biogas")["model_gwh"] == pytest.approx(3_000.0)
