"""The compact register (29 Sep 2026): Parquet body plus a case-settings table, read back as the CSV was."""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.levers import DEFAULT_SETTINGS
from scripts.public_ownership.register_io import (
    read_register, read_settings, settings_path, write_register,
)


def _register():
    rows = []
    for case, stake in (("private", 0.0), ("public_ownership", 1.0)):
        for i, (asset, carrier) in enumerate((("a::legacy", "wind_offshore"), ("b", "solar_pv"), ("c", "nuclear"))):
            rows.append({
                "asset_id": asset, "carrier": carrier, "capacity_mw": 100.0 * (i + 1), "dispatch_mwh": 1e5 * i,
                "commissioning_year": np.nan if i == 2 else 2020.0 + i, "cfd_round": ["AR4", None, "HPC"][i],
                "state_owned": bool(stake), "existing_site": [True, None, False][i],
                "acquisition_price_gbp": None, "mixed": [1, "x", None][i], "notes": ["2025", "2026", None][i],
                "case_id": case, **{k: (v if case == "private" else ("public" if k == "new_build_generation" else v))
                                    for k, v in DEFAULT_SETTINGS.items()},
                "ownership_share": stake,
            })
    return pd.DataFrame(rows)


def test_the_register_reads_back_as_the_csv_did(tmp_path):
    register = _register()
    csv = tmp_path / "g_asset_finance_register.csv"
    register.to_csv(csv, index=False)
    expected = pd.read_csv(csv, low_memory=False)
    parquet = tmp_path / "g_asset_finance_register.parquet"
    sizes = write_register(register, parquet)
    assert sizes["register_bytes"] > 0 and settings_path(parquet).name == "g_case_settings.csv"
    got = read_register(parquet)
    assert list(got.columns) == list(expected.columns)
    levers = [c for c in DEFAULT_SETTINGS if c in got.columns]
    assert all(isinstance(got[c].dtype, pd.CategoricalDtype) for c in levers)
    # Body columns: exactly the CSV reader's values and dtypes.
    body = [c for c in expected.columns if c not in levers]
    pd.testing.assert_frame_equal(got[body], expected[body])
    # Lever columns: the same text as the CSV (read as str).
    for lever in levers:
        assert got[lever].astype(str).tolist() == expected[lever].astype(str).tolist(), lever
    # One settings row per case.
    settings = read_settings(parquet)
    assert settings["case_id"].tolist() == ["private", "public_ownership"]
    assert settings.set_index("case_id").loc["public_ownership", "new_build_generation"] == "public"


def test_a_subset_reads_only_its_cases_and_columns(tmp_path):
    parquet = tmp_path / "g_asset_finance_register.parquet"
    write_register(_register(), parquet)
    got = read_register(parquet, columns=["asset_id", "capacity_mw", "new_build_generation"], cases=["public_ownership"])
    assert set(got["case_id"]) == {"public_ownership"} and len(got) == 3
    assert list(got.columns) == ["asset_id", "capacity_mw", "case_id", "new_build_generation"]
    assert set(got["new_build_generation"].astype(str)) == {"public"}


def test_a_case_with_two_settings_rows_is_refused(tmp_path):
    register = _register()
    register.loc[0, "new_build_generation"] = "state_stake"
    with pytest.raises(ValueError, match="more than one lever setting"):
        write_register(register, tmp_path / "g_asset_finance_register.parquet")


def test_an_old_csv_register_still_reads(tmp_path):
    csv = tmp_path / "g_asset_finance_register.csv"
    _register().to_csv(csv, index=False)
    got = read_register(csv, columns=["asset_id", "carrier"], cases=["private"])
    assert got["case_id"].eq("private").all() and set(got.columns) == {"case_id", "asset_id", "carrier"}


def test_an_empty_string_reads_back_missing_as_the_csv_did(tmp_path):
    register = _register()
    register["route"] = ""
    parquet = tmp_path / "g_asset_finance_register.parquet"
    write_register(register, parquet)
    assert read_register(parquet, columns=["route"])["route"].isna().all()
