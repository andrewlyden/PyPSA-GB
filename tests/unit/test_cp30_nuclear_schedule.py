"""Nuclear at named stations: the CP30 total conserved, HPC labelled CfD."""

import pandas as pd
import pytest

from scripts.CP30.nuclear_schedule import (
    DATA_SOURCE, apply_nuclear_station_schedule, load_nuclear_station_schedule,
)

SCHEDULE = "data/market/CP30/nuclear_station_schedule_2030.csv"


def _proxy_rows():
    return pd.DataFrame({
        "technology": ["nuclear", "nuclear", "CCGT"],
        "fuel_type": ["nuclear", "nuclear", "CCGT"],
        "capacity_mw": [2000.0, 1515.0, 500.0],
        "bus": ["AAA41", "BBB41", "CCC41"],
        "station_name": ["CP30_nuclear_0", "CP30_nuclear_1", "CP30_CCGT_0"],
        "gsp": ["Direct(NGET)", "Direct(NGET)", "Direct(NGET)"],
        "data_source": ["CP30"] * 3,
        "cp30_pathway": ["Further Flex and Renewables"] * 3,
    })


def test_shipped_schedule_conserves_the_cp30_nuclear_total():
    result = apply_nuclear_station_schedule(_proxy_rows(), SCHEDULE, 3515.0)
    nuclear = result[result.fuel_type.eq("nuclear")]
    assert nuclear.capacity_mw.sum() == pytest.approx(3515.0)
    by_station = nuclear.set_index("station_name")
    assert by_station.loc["Sizewell B", "capacity_mw"] == pytest.approx(1198.0)
    assert by_station.loc["Hinkley Point C", "capacity_mw"] == pytest.approx(3515.0 - 1198.0)
    assert by_station.loc["Hinkley Point C", "support_type"] == "CfD"
    assert by_station.loc["Hinkley Point C", "cfd_round"] == "HPC"
    assert by_station.loc["Sizewell B", "support_type"] == "merchant"
    assert set(by_station.bus) == {"SIZE41", "HINP41"}
    assert (nuclear.data_source == DATA_SOURCE).all()
    # No GSP, so the later "Direct" spatial distribution leaves them alone.
    assert nuclear.gsp.isna().all()
    # Non-nuclear rows pass through untouched.
    assert result[result.fuel_type.eq("CCGT")].capacity_mw.sum() == 500.0


def test_hpc_residual_matches_tec_unit_dates_to_within_ten_megawatts():
    """1,198 + 1,630 (unit 1) + 5/12 x 1,630 (unit 2 from 31 July) ~ CP30's 3,515."""
    residual = 3515.0 - 1198.0
    inferred = 1630.0 + 1630.0 * 5 / 12
    assert abs(residual - inferred) < 10


def test_fixed_stations_larger_than_the_target_are_an_error(tmp_path):
    path = tmp_path / "s.csv"
    pd.DataFrame({
        "station_name": ["A", "B"], "bus": ["X", "Y"], "capacity_mw": ["3000", "residual"],
        "support_type": ["merchant", "CfD"], "cfd_round": ["", "HPC"], "source": ["s", "s"],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exceed the CP30 nuclear total"):
        apply_nuclear_station_schedule(_proxy_rows(), path, 2500.0)


def test_schedule_without_a_residual_must_match_the_target(tmp_path):
    path = tmp_path / "s.csv"
    pd.DataFrame({
        "station_name": ["A"], "bus": ["X"], "capacity_mw": ["1000"],
        "support_type": ["merchant"], "cfd_round": [""], "source": ["s"],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no station takes the residual"):
        apply_nuclear_station_schedule(_proxy_rows(), path, 3515.0)


def test_schedule_validation(tmp_path):
    path = tmp_path / "s.csv"
    pd.DataFrame({
        "station_name": ["A", "A"], "bus": ["X", "Y"], "capacity_mw": ["1", "residual"],
        "support_type": ["merchant", "CfD"], "cfd_round": ["", ""], "source": ["s", "s"],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate station names"):
        load_nuclear_station_schedule(path)
    pd.DataFrame({
        "station_name": ["A"], "bus": ["X"], "capacity_mw": ["residual"],
        "support_type": ["Subsidy"], "cfd_round": [""], "source": ["s"],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="support_type"):
        load_nuclear_station_schedule(path)


def test_hpc_round_resolves_in_the_strike_table():
    from scripts.market.cfd_strike_table import load_strike_table
    from tests.unit.test_future_consumer_bill import STRIKE_TABLE_PATH, _indexation
    table = load_strike_table(STRIKE_TABLE_PATH, _indexation(), 2024)
    strike, key, is_fallback = table.lookup("nuclear", "HPC")
    assert key == "HPC_nuclear" and is_fallback is False
    # 89.50 since Sizewell C's financial close (4 Nov 2025; OW12 moved it into the table).
    assert strike == pytest.approx(89.5 * 1.393340270551509)


def test_cp30_scenarios_declare_the_nuclear_schedule():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("config").resolve()))
    from config_loader import load_config
    scenario = load_config()["scenarios"][
        "CP30_FFR_2030_Jan_cp30_network"]
    assert scenario["fleet"]["spatial_proxy"]["nuclear_station_schedule"] == SCHEDULE
    assert "nuclear" not in scenario["support"]["labels"]["fes_cfd_fractions"]
