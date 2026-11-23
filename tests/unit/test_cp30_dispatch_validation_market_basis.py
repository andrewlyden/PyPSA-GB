"""Follow-up F15 (6 Oct 2026): CP30 ES1 is judged on Stage 1, CP.08 on the balancing-stage turn-down."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.CP30.validate_cp30_dispatch import (
    CP08_NETWORK,
    clean_power_turn_down_mwh,
    constraint_curtailment_row,
    cp08_bid_volume_gwh,
    market_basis_table,
)

COLUMNS = ["stage", "component", "item", "es1_lines", "cp30_gwh", "model_gwh", "difference_gwh",
           "relative_difference", "tolerance", "status", "note"]


def _table(rows):
    return pd.DataFrame([dict(zip(COLUMNS, ["dispatch", *r])) for r in rows], columns=COLUMNS)


def _row(component, item, cp30, model, status="WARN", note=""):
    return [component, item, "", cp30, model, model - cp30 if cp30 else np.nan, np.nan, 0.1, status, note]


def test_market_basis_takes_es1_rows_from_stage1_and_keeps_final_as_info():
    stage1 = _table([_row("dispatched_energy", "wind_onshore", 57.0, 47.5, "WARN"),
                     _row("curtailment", "wind_onshore", 12.4, 22.6, "WARN"),
                     _row("dispatched_energy", "Hydrogen", 0.34, 0.0, "WARN"),
                     _row("demand", "Electricity demand excluding electrolysis", 315.0, 315.0, "PASS")])
    final = _table([_row("dispatched_energy", "wind_onshore", 57.0, 40.0, "WARN"),
                    _row("curtailment", "wind_onshore", 12.4, 30.0, "WARN"),
                    _row("dispatched_energy", "Hydrogen", 0.34, 0.35, "PASS"),
                    _row("demand", "Electricity demand excluding electrolysis", 315.0, 315.0, "PASS"),
                    _row("electrolysis", "Electrolysis demand", 14.9, 15.2, "PASS")])
    out = market_basis_table(stage1, final).set_index(["component", "item"])
    assert out.loc[("curtailment", "wind_onshore"), "model_gwh"] == 22.6
    assert out.loc[("curtailment", "wind_onshore"), "stage"] == "dispatch_stage1"
    assert out.loc[("curtailment_final", "wind_onshore"), "model_gwh"] == 30.0
    assert out.loc[("curtailment_final", "wind_onshore"), "status"] == "INFO"
    # Hydrogen turbines are links, absent from Stage 1 files: the final row is the judged one.
    assert out.loc[("dispatched_energy", "Hydrogen"), "model_gwh"] == 0.35
    assert out.loc[("dispatched_energy", "Hydrogen"), "stage"] == "dispatch_final"
    assert ("dispatched_energy_final", "Hydrogen") not in out.index
    assert out.loc[("electrolysis", "Electrolysis demand"), "stage"] == "dispatch_final"
    assert not out.index.duplicated().any()
    # The explorer reads dispatched_energy and curtailment by item: one row each, the Stage 1 one.
    assert list(market_basis_table(stage1, final).columns) == COLUMNS


def _month(path, p, carriers):
    names = list(p.columns)
    ds = xr.Dataset({
        "generators_carrier": ("generators_i", np.array(carriers, dtype=object)),
        "generators_t_p": (("snapshots", "generators_t_p_i"), p.to_numpy()),
        "snapshots_generators": ("snapshots", np.ones(len(p))),
    }, coords={"generators_i": names, "generators_t_p_i": names, "snapshots": np.arange(len(p))})
    ds.to_netcdf(path)


def test_turn_down_is_gross_clean_power_only(tmp_path):
    carriers = ["wind_onshore", "wind_onshore", "CCGT"]
    s1 = pd.DataFrame({"w1": [100.0, 50.0], "w2": [80.0, 80.0], "gas": [10.0, 10.0]})
    s2 = pd.DataFrame({"w1": [60.0, 70.0], "w2": [80.0, 30.0], "gas": [0.0, 40.0]})
    _month(tmp_path / "w.nc", s1, carriers)
    _month(tmp_path / "b.nc", s2, carriers)
    down = clean_power_turn_down_mwh(tmp_path / "w.nc", tmp_path / "b.nc")
    # w1: 40 down then 20 up (up is not netted); w2: 50 down; gas is not clean power.
    assert down.to_dict() == {"wind_onshore": pytest.approx(90.0)}


def test_cp08_row_reads_the_recommended_network_and_warns_beyond_tolerance():
    target = cp08_bid_volume_gwh("data/CP30/cp30_cp08_constraint_bid_volumes_2030.csv", "Further Flex and Renewables")
    assert target == pytest.approx(25683.0)
    assert CP08_NETWORK.startswith("Network capacity we are recommending")
    row = constraint_curtailment_row(target, pd.Series({"wind_onshore": 9.6e6, "wind_offshore": 7.8e6}))
    assert row["model_gwh"] == pytest.approx(17400.0)
    assert row["status"] == "WARN"          # -32% against a 25% tolerance
    assert constraint_curtailment_row(target, pd.Series({"wind_onshore": 24.0e6}))["status"] == "PASS"
