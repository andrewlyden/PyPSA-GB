"""A performance factor must measure availability, not settled output.

LCCC settled generation is net of curtailment. Calibrating on it folds a
network outcome into a resource assumption, and the scenario model then
curtails a second time on top.
"""

import pandas as pd
import pytest

from scripts.consumer_bill.calibrate_offshore_availability_factors import (
    calibrate,
    load_bid_volumes,
)


def _contracts():
    return pd.DataFrame({
        "CfD_ID": ["A", "B"],
        "resource_region": ["north_sea", "scottish_offshore"],
        "actual_generation_mwh": [800.0, 400.0],
        "raw_profile_mwh": [1000.0, 1000.0],
    })


def test_curtailment_is_added_back_before_the_factor_is_formed():
    result = calibrate(_contracts(), {"A": ["T_A-1"], "B": ["T_B-1"]},
                       pd.Series({"T_A-1": 100.0, "T_B-1": 500.0})).set_index("resource_region")

    # Settled-only would report the Scottish farm as a 0.40 resource when it is
    # really a 0.90 resource that was curtailed.
    assert result.loc["scottish_offshore", "factor_from_settled_output"] == pytest.approx(0.40)
    assert result.loc["scottish_offshore", "factor_from_potential_output"] == pytest.approx(0.90)
    assert result.loc["north_sea", "factor_from_potential_output"] == pytest.approx(0.90)


def test_a_bmu_shared_by_two_contract_phases_is_not_double_counted():
    contracts = pd.DataFrame({
        "CfD_ID": ["A", "B"],
        "resource_region": ["north_sea", "north_sea"],
        "actual_generation_mwh": [400.0, 400.0],
        "raw_profile_mwh": [500.0, 500.0],
    })
    shared = {"A": ["T_S-1"], "B": ["T_S-1"]}
    result = calibrate(contracts, shared, pd.Series({"T_S-1": 100.0})).iloc[0]
    assert result["curtailed_mwh"] == pytest.approx(100.0)
    assert result["factor_from_potential_output"] == pytest.approx(0.90)


def test_contracts_without_a_bmu_mapping_are_reported_not_silently_zeroed():
    result = calibrate(_contracts(), {"A": ["T_A-1"]}, pd.Series({"T_A-1": 100.0}))
    unmapped = result.set_index("resource_region")["contracts_without_bmu_mapping"]
    assert unmapped["scottish_offshore"] == 1
    assert unmapped["north_sea"] == 0


def test_distinct_acceptances_with_equal_volumes_are_both_counted(tmp_path):
    """Deduplicating without the acceptance id silently halves curtailment."""
    boav = pd.DataFrame({
        "bmu_id": ["T_A-1", "T_A-1"],
        "acceptance_id": [1, 2],
        "settlement_date": ["2020-01-01", "2020-01-01"],
        "settlement_period": [7, 7],
        "side": ["bid", "bid"],
        "accepted_volume_mwh": [-25.0, -25.0],
    })
    path = tmp_path / "boav_data.csv"
    boav.to_csv(path, index=False)
    assert load_bid_volumes([path])["T_A-1"] == pytest.approx(50.0)


def test_overlapping_caches_do_not_double_count_the_same_acceptance(tmp_path):
    boav = pd.DataFrame({
        "bmu_id": ["T_A-1"], "acceptance_id": [1], "settlement_date": ["2020-01-31"],
        "settlement_period": [48], "side": ["bid"], "accepted_volume_mwh": [-30.0],
    })
    first, second = tmp_path / "jan.csv", tmp_path / "feb.csv"
    boav.to_csv(first, index=False); boav.to_csv(second, index=False)
    assert load_bid_volumes([first, second])["T_A-1"] == pytest.approx(30.0)


def test_offers_are_ignored_because_only_bids_curtail():
    assert calibrate(_contracts(), {"A": ["T_A-1"]}, pd.Series(dtype=float)).iloc[0][
        "curtailed_mwh"
    ] == pytest.approx(0.0)
