"""Ownership tables must name the network they bid on (scripts/public_ownership/share_coverage.py)."""

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.public_ownership.share_coverage import check_share_coverage, share_coverage


def _table(tmp_path, rows):
    path = tmp_path / "shares.csv"
    pd.DataFrame(rows, columns=["case_id", "generator", "state_capacity_mw", "state_share"]).to_csv(path, index=False)
    return path


def _network(generators=(), storage_units=()):
    return SimpleNamespace(generators=pd.DataFrame(index=list(generators)),
                           storage_units=pd.DataFrame(index=list(storage_units)),
                           links=pd.DataFrame(), stores=pd.DataFrame())


def test_matching_table_passes(tmp_path):
    path = _table(tmp_path, [("p2", "wind_a", 100.0, 1.0), ("p2", "bat_a", 50.0, 1.0), ("p2", "gone", 0.0, 0.0)])
    result = check_share_coverage(path, _network(["wind_a"], ["bat_a"]), logger=logging.getLogger("t"))
    assert result.loc[0, "unmatched_mw"] == 0.0
    assert result.loc[0, "owned_units"] == 2  # a private-only row is not judged


def test_stale_table_fails_with_rebuild_hint(tmp_path):
    path = _table(tmp_path, [("p2", "wind_a", 990.0, 1.0), ("p2", "renamed", 10.0, 1.0)])
    with pytest.raises(ValueError, match="build_public_ownership_state_shares"):
        check_share_coverage(path, _network(["wind_a"]), logger=logging.getLogger("t"))


def test_threshold_and_case_filter(tmp_path):
    path = _table(tmp_path, [("p2", "wind_a", 999.5, 1.0), ("p2", "renamed", 0.5, 1.0),
                             ("p3", "renamed", 100.0, 1.0)])
    net = _network(["wind_a"])
    result = check_share_coverage(path, net, case_id="p2", logger=logging.getLogger("t"))
    assert result.loc[0, "unmatched_share"] == pytest.approx(0.0005)
    with pytest.raises(ValueError, match="p3"):
        check_share_coverage(path, net, logger=logging.getLogger("t"))
    with pytest.raises(ValueError, match="no rows for case"):
        check_share_coverage(path, net, case_id="p9", logger=logging.getLogger("t"))


def test_share_coverage_lists_largest_missing_first():
    table = pd.DataFrame({"case_id": ["p2"] * 3, "generator": ["a", "b", "c"],
                          "state_capacity_mw": [5.0, 50.0, 1.0]})
    result = share_coverage(table, {"a"})
    assert result.loc[0, "examples"] == ["b", "c"]
