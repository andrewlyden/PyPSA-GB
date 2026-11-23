"""User decision (14 Nov 2026): CP30's 2030 pumped-storage projects without an FID are not bought in 2030."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.public_ownership import legacy_valuation as lv

ASSETS_2030 = Path("resources/public_ownership/cp30_2030_no_acceleration_asset_inputs.csv")
PROJECTS = {"CoireGlas": 1296.0, "CRUA_expansion": 600.0, "LochNaCathrach": 450.0, "Glenmuckloch": 210.0}


def _rows(year=2030, coire_mw=1296.0):
    rows = [{"asset_id": f"CP30_LAES_committed_{name}_5{i}::pre_legislation", "carrier": "LAES",
             "capacity_mw": coire_mw if name == "CoireGlas" else mw, "model_year": year,
             "component_type": "StorageUnit", "finance_vintage": "legacy"}
            for i, (name, mw) in enumerate(PROJECTS.items())]
    rows += [{"asset_id": "CP30_LAES_12_584::pre_legislation", "carrier": "LAES", "capacity_mw": 502.3, "model_year": year,
              "component_type": "StorageUnit", "finance_vintage": "legacy"},
             {"asset_id": "CP30_LAES_existing_Dinorwig_572::legacy", "carrier": "pumped_hydro", "capacity_mw": 1728.0,
              "model_year": year, "component_type": "StorageUnit", "finance_vintage": "legacy"}]
    return pd.DataFrame(rows)


def test_the_four_projects_are_out_of_the_2030_purchase_and_nothing_else_is():
    settings = lv.settings_from_config({}, 2030)
    rows = _rows()
    mask = lv.not_operating_mask(rows, settings, check=True)
    assert mask.tolist() == [True, True, True, True, False, False]
    group = lv.scope_group(rows, settings)
    assert set(group[mask]) == {lv.GROUP_EXCLUDED}
    assert lv.GROUP_EXCLUDED not in set(group[~mask])


def test_a_later_model_year_is_untouched():
    settings = lv.settings_from_config({}, 2050)
    assert not lv.not_operating_mask(_rows(year=2050), settings, check=True).any()


def test_a_resized_or_renamed_project_is_refused():
    settings = lv.settings_from_config({}, 2030)
    with pytest.raises(ValueError, match="CoireGlas"):
        lv.not_operating_mask(_rows(coire_mw=2000.0), settings, check=True)      # resized above the declaration
    # Split by the legislation year (2025-law bound): only the legacy half is seen, which is allowed.
    assert lv.not_operating_mask(_rows(coire_mw=648.0), settings, check=True).sum() == 4
    renamed = _rows().assign(asset_id=lambda f: f["asset_id"].str.replace("Glenmuckloch", "Glenmuck"))
    with pytest.raises(ValueError, match="Glenmuckloch"):
        lv.not_operating_mask(renamed, settings, check=True)


def test_the_table_is_sourced_and_the_config_matches_the_code_default():
    table = lv.not_operating_table(lv.DEFAULT_SETTINGS)
    assert set(table["asset_match"].str.extract(r"committed_(\w+?)_$")[0]) == set(PROJECTS)
    assert table["reason"].notna().all() and table["source"].notna().all()
    import yaml
    block = yaml.safe_load(Path("config/defaults/base/public_ownership/acquisition.yaml").read_text(encoding="utf-8"))
    block = block.get("acquisition", block)
    assert block["valuation"]["not_operating_at_acquisition"] == lv.DEFAULT_SETTINGS["not_operating_at_acquisition"]


@pytest.mark.skipif(not ASSETS_2030.exists(), reason="2030 register inputs not built")
def test_the_2030_register_holds_exactly_the_declared_projects():
    assets = pd.read_csv(ASSETS_2030, low_memory=False)
    mask = lv.not_operating_mask(assets, lv.settings_from_config({}, 2030), check=True)
    assert assets.loc[mask, "capacity_mw"].sum() == pytest.approx(sum(PROJECTS.values()), abs=1.0)


def test_a_case_that_buys_the_fleet_completes_the_projects_as_its_own_new_build():
    """User decision (15 Nov 2026, option a): S3 is full ownership, so the projects the purchase cannot buy because
    they are not yet operating are completed by the state as new build; a case that buys nothing leaves them private."""
    import numpy as np
    from scripts.public_ownership import calculate_financing_scenarios as cfs
    from scripts.public_ownership.levers import DEFAULT_SETTINGS

    rows = cfs._RowInputs.__new__(cfs._RowInputs)
    n = 3
    rows.n = n
    rows.base_vintage = np.array(["legacy", "legacy", "legacy"], dtype=object)
    rows._vintage_override = None
    rows.pre_legislation_contract = np.zeros(n, dtype=bool)
    rows.excluded = np.zeros(n, dtype=bool)
    rows.resolved = np.ones(n, dtype=bool)
    rows.is_storage = np.ones(n, dtype=bool)
    rows.pre_policy_cfd = np.zeros(n, dtype=bool)
    rows.pre_2025_dpa = np.zeros(n, dtype=bool)
    rows.in_construction_carried_round = np.zeros(n, dtype=bool)
    rows.in_construction_hpc = np.zeros(n, dtype=bool)
    rows.acquired_in_construction = np.zeros(n, dtype=bool)
    rows.baseline_acquired = None
    rows.baseline_legacy_not_bought = None
    rows.gbe_owned_rows = lambda settings, plan: None
    rows.completed_by_state = np.array([True, False, False])
    s3 = dict(DEFAULT_SETTINGS, legacy_fleet="acquire_all", new_build_generation="public")
    plan = pd.DataFrame({"acquirable": [False, True, False], "legacy_scope_group": ["excluded", "valued", "excluded"]})
    cfs._RowInputs.use_vintage(rows, s3, plan)
    assert list(rows.vintage) == ["new_build", "legacy", "legacy"]
    assert list(cfs._state_owned_vector(rows, s3, plan)) == [True, True, False]
    # A case that buys nothing (no plan) keeps the project legacy and private.
    s2 = dict(DEFAULT_SETTINGS, new_build_generation="public")
    cfs._RowInputs.use_vintage(rows, s2, None)
    assert list(rows.vintage) == ["legacy", "legacy", "legacy"]
    assert not cfs._state_owned_vector(rows, s2, None)[0]
