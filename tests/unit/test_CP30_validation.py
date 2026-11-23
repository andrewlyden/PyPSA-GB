from pathlib import Path

import pandas as pd
import pytest

from scripts.CP30.assumptions import load_or_derive_cp30_electrolysis_capacity_mw
from scripts.CP30.generation import load_cp30_generation_capacities
from scripts.CP30.storage_interconnectors import (
    load_cp30_interconnector_capacity_mw,
    load_cp30_storage_targets,
)
from scripts.CP30.validation import (
    CP30_NETWORK_ASSUMPTION,
    _actual_interconnector_capacity,
    network_assumption_record,
    summarise_cp30_network_reference,
    validate_cp30_component_totals,
)


def test_directional_interconnector_pair_counts_once_in_cp30_validation():
    links = pd.DataFrame(
        {
            "carrier": ["DC", "DC", "DC"],
            "p_nom": [12450.0, 12450.0, 100.0],
            "interconnector_pair_id": ["TestCable", "TestCable", None],
        },
        index=["IC_IMPORT_TestCable", "IC_EXPORT_TestCable", "IC_legacy"],
    )

    assert _actual_interconnector_capacity(links) == pytest.approx(12550.0)


CP30_ES1 = Path("resources/CP30/CP30_ES1.csv")
CP30_ED1 = Path("resources/CP30/CP30_ED1.csv")
CP30_REFERENCE = Path("resources/CP30/CP30_reference_N1_N2_NA1.csv")
PATHWAY = "Further Flex and Renewables"
YEAR = 2030


@pytest.mark.skipif(not CP30_REFERENCE.exists(), reason="CP30 reference resource not available")
def test_cp30_network_reference_summary_exposes_available_reference_tables():
    summary = summarise_cp30_network_reference(CP30_REFERENCE)
    by_sheet = summary.set_index("source_sheet")

    assert {"N1", "N2", "NA1", "CP.08", "CP.12"}.issubset(set(by_sheet.index))
    assert by_sheet.loc["N1", "network_projects"] > 0
    assert by_sheet.loc["N2", "rows"] > 0


def test_cp30_network_assumption_record_is_explicit():
    record = network_assumption_record(
        {
            "scenario_source": "CP30",
            "network_model": "Reduced",
        }
    ).iloc[0]

    assert record["scenario_source"] == "CP30"
    assert record["network_model"] == "Reduced"
    assert record["cp30_reference_use"] == "reference_metadata_only"
    assert record["network_assumption"] == CP30_NETWORK_ASSUMPTION


@pytest.mark.skipif(
    not CP30_ES1.exists() or not CP30_ED1.exists(),
    reason="CP30 ES1/ED1 resources not available",
)
def test_validate_cp30_component_totals_passes_when_targets_are_preserved():
    gen_targets = load_cp30_generation_capacities(CP30_ES1, PATHWAY, YEAR)
    generators = gen_targets.rename(columns={"capacity_mw": "p_nom"})[["carrier", "p_nom"]]
    # The split "gas" line is built as CCGT, OCGT and gas engines; validation
    # checks their sum, so any division of the total passes.
    gas = generators["carrier"].eq("gas")
    gas_mw = float(generators.loc[gas, "p_nom"].sum())
    generators = pd.concat([
        generators[~gas],
        pd.DataFrame({"carrier": ["CCGT", "OCGT", "gas_engine"], "p_nom": [0.8 * gas_mw, 0.12 * gas_mw, 0.08 * gas_mw]}),
    ], ignore_index=True)

    storage_targets = load_cp30_storage_targets(CP30_ES1, PATHWAY, YEAR)
    storage_units = pd.DataFrame(
        {
            "carrier": storage_targets["carrier"],
            "p_nom": storage_targets["power_mw"],
            "max_hours": storage_targets["duration_h"],
        }
    )

    interconnector_target = load_cp30_interconnector_capacity_mw(CP30_ES1, PATHWAY, YEAR)
    electrolysis_target, _ = load_or_derive_cp30_electrolysis_capacity_mw(CP30_ED1, CP30_ES1, PATHWAY, YEAR)
    links = pd.DataFrame(
        {
            "carrier": ["DC", "electrolysis"],
            "p_nom": [interconnector_target, electrolysis_target],
        },
        index=["IC_total_proxy", "electrolysis_total_proxy"],
    )

    report = validate_cp30_component_totals(
        CP30_ES1,
        CP30_ED1,
        PATHWAY,
        YEAR,
        generators=generators,
        storage_units=storage_units,
        links=links,
    )

    assert set(report["status"]) == {"PASS"}


@pytest.mark.skipif(not CP30_ES1.exists() or not CP30_ED1.exists(), reason="CP30 resources not available")
def test_validate_cp30_component_totals_flags_capacity_drift():
    gen_targets = load_cp30_generation_capacities(CP30_ES1, PATHWAY, YEAR)
    generators = gen_targets.rename(columns={"capacity_mw": "p_nom"})[["carrier", "p_nom"]].copy()
    generators.loc[generators["carrier"] == "solar_pv", "p_nom"] += 10.0

    storage_targets = load_cp30_storage_targets(CP30_ES1, PATHWAY, YEAR)
    storage_units = pd.DataFrame(
        {
            "carrier": storage_targets["carrier"],
            "p_nom": storage_targets["power_mw"],
            "max_hours": storage_targets["duration_h"],
        }
    )
    interconnector_target = load_cp30_interconnector_capacity_mw(CP30_ES1, PATHWAY, YEAR)
    electrolysis_target, _ = load_or_derive_cp30_electrolysis_capacity_mw(CP30_ED1, CP30_ES1, PATHWAY, YEAR)
    links = pd.DataFrame(
        {"carrier": ["DC", "electrolysis"], "p_nom": [interconnector_target, electrolysis_target]},
        index=["IC_total_proxy", "electrolysis_total_proxy"],
    )

    report = validate_cp30_component_totals(
        CP30_ES1,
        CP30_ED1,
        PATHWAY,
        YEAR,
        generators=generators,
        storage_units=storage_units,
        links=links,
    )
    solar = report[(report["component"] == "generation") & (report["item"] == "solar_pv")].iloc[0]

    assert solar["status"] == "FAIL"
    assert solar["difference"] == pytest.approx(10.0)
