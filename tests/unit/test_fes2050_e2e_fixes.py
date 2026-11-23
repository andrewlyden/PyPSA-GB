"""Task E2E (1 Oct 2026): the fixes that let the 2050 finance and bill chain run end to end.
Each test pins one fix and, where it touches a shared path, that the 2030 behaviour is unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.public_ownership.fleet_vintages as fv
from scripts.public_ownership.build_public_account import capacity_market_line_is_nil
from scripts.public_ownership.cfd_repricing import register_settings_by_case
from scripts.public_ownership.levers import (
    DEFAULT_SETTINGS, LEVER_PRIVATE_RATE_PATH, NEW_BUILD_PUBLIC, RATE_PATH_NESO_WIND_RAMP, comparator_for_settings,
)
from scripts.public_ownership.project_public_ownership_consumer_bill import axis_settings
from scripts.public_ownership.reference_runs import configured_group

A2030 = "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json"
A2050 = "data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json"


# ── the vintaged register ────────────────────────────────────────────────────


def _register(rows):
    return pd.DataFrame([{"component_type": "Generator", "asset_id": a, "carrier": c, "bus": "B", "capacity_mw": mw,
                          "dispatch_mwh": mw * 1000.0, "commissioning_year": np.nan} for a, c, mw in rows])


def test_a_pooled_carrier_names_the_technology_in_each_cohort_id():
    """Nuclear pools large plant and SMRs: two technologies of one build year must not share an id."""
    mix = pd.DataFrame([{"technology": "Large-scale nuclear", "build_year": 2035, "capacity_mw": 60.0},
                        {"technology": "Small modular reactors (SMRs)", "build_year": 2035, "capacity_mw": 40.0}])
    cohorts, _ = fv.assign_build_years(_register([("NUC", "nuclear", 100.0)]), mix, model_year=2050,
                                       lives=fv.load_lives())
    assert cohorts["asset_id"].is_unique
    assert set(cohorts["asset_id"]) == {"NUC::b2035::large_scale_nuclear", "NUC::b2035::small_modular_reactors_smrs"}
    assert cohorts["capacity_mw"].sum() == pytest.approx(100.0)


def test_a_single_technology_carrier_keeps_its_cohort_ids():
    mix = pd.DataFrame([{"technology": "Offshore wind", "build_year": y, "capacity_mw": 50.0} for y in (2030, 2040)])
    cohorts, _ = fv.assign_build_years(_register([("OFF", "wind_offshore", 100.0)]), mix, model_year=2050,
                                       lives=fv.load_lives())
    assert set(cohorts["asset_id"]) == {"OFF::b2030", "OFF::b2040"}


def test_biogas_takes_its_vintages_from_the_waste_line():
    """FES 2025 ES1 has no Other Renewables line; its Waste line covers waste and biogas."""
    assert fv.carrier_technologies()["biogas"] == ("Energy from waste",)


# ── gated levers reach the comparator ────────────────────────────────────────


def test_register_settings_carry_an_off_default_gated_lever(tmp_path):
    register = tmp_path / "g_asset_finance_register.parquet"
    register.write_bytes(b"")                       # read off the settings table beside it
    rows = [{"case_id": "private"}, {"case_id": "private@private_rate_neso_wind_ramp",
                                     LEVER_PRIVATE_RATE_PATH: RATE_PATH_NESO_WIND_RAMP},
            {"case_id": "public_ownership@private_rate_neso_wind_ramp", LEVER_PRIVATE_RATE_PATH: RATE_PATH_NESO_WIND_RAMP,
             "new_build_generation": NEW_BUILD_PUBLIC}]
    pd.DataFrame(rows).to_csv(tmp_path / "g_case_settings.csv", index=False)
    settings = register_settings_by_case(register)
    assert LEVER_PRIVATE_RATE_PATH not in settings["private"]
    assert settings["private@private_rate_neso_wind_ramp"][LEVER_PRIVATE_RATE_PATH] == RATE_PATH_NESO_WIND_RAMP
    assert comparator_for_settings("public_ownership@private_rate_neso_wind_ramp", settings) == \
        "private@private_rate_neso_wind_ramp"


def test_a_settings_table_without_gated_columns_is_read_as_before(tmp_path):
    register = tmp_path / "g_asset_finance_register.parquet"
    register.write_bytes(b"")
    pd.DataFrame([{"case_id": "private"}, {"case_id": "public_ownership"}]).to_csv(
        tmp_path / "g_case_settings.csv", index=False)
    settings = register_settings_by_case(register)
    assert settings == {"private": dict(DEFAULT_SETTINGS), "public_ownership": dict(DEFAULT_SETTINGS)}


def test_the_projector_keeps_a_declared_gated_lever_and_nothing_else():
    ramp = {"sensitivity_axes": {LEVER_PRIVATE_RATE_PATH: RATE_PATH_NESO_WIND_RAMP}}
    assert axis_settings(ramp)[LEVER_PRIVATE_RATE_PATH] == RATE_PATH_NESO_WIND_RAMP
    assert axis_settings({"sensitivity_axes": {LEVER_PRIVATE_RATE_PATH: "flat"}}) == axis_settings(None)
    assert set(axis_settings(None)) == set(DEFAULT_SETTINGS)


# ── the Capacity Market line and the case catalogue ─────────────────────────


def test_only_a_declared_nil_capacity_market_line_books_nothing(tmp_path):
    nil = tmp_path / "nil.json"
    nil.write_text(json.dumps({"sourced_externals": {"capacity_market": {"value_basis": "nil"}}}), encoding="utf-8")
    assert capacity_market_line_is_nil(nil) is True
    # Decision 30 (3 Oct 2026): V2G bids in as DSR, so the 2050 auction clears and the line is not nil.
    assert capacity_market_line_is_nil(A2050) is False
    assert capacity_market_line_is_nil(A2030) is False
    assert capacity_market_line_is_nil(None) is False


def test_configured_group_falls_back_to_the_global_list_for_unknown_names():
    config = {"reference_run_groups": {"cp30_2030_cp30_network": {}}}
    assert configured_group(config, "cp30_2030_cp30_network") == "cp30_2030_cp30_network"
    assert configured_group(config, "scratch_group") is None
    assert configured_group(config, None) is None


def test_the_2050_bill_files_still_declare_their_capacity_market_source():
    block = json.loads(Path(A2050).read_text(encoding="utf-8"))["sourced_externals"]["capacity_market"]
    assert "value_basis" not in block and block["derivation"]["agreement_cost_gbp"] > 0.0   # decision 30
    assert block["period_basis"] == "calendar_year"
