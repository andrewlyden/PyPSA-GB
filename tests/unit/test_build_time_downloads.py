"""Build-time downloads fail loudly or serve a committed cache (F25.12).

- NESO day-ahead boundary limits for historical NESO outage-schedule solves are
  read from the committed snapshot; a missing boundary raises instead of being
  skipped, and the API is only queried when ``allow_api_fetch`` is set.
- ELEXON BOD retrieval raises when a day failed, before writing outputs or the
  processed cache.
- DUKES geocoding never records a network error as a permanent lookup failure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.solve import solve_network as sn
from scripts.solve import build_neso_boundary_limit_cache as cache_builder

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("test_build_time_downloads")


def _snapshots(start="2020-03-01 00:00", periods=48):
    return pd.date_range(start, periods=periods, freq="h")


def _limits(boundary, start, periods, limit=1000.0):
    dates = pd.date_range(start, periods=periods, freq="30min")
    return pd.DataFrame({
        "boundary": boundary,
        "date": dates,
        "limit_mw": limit,
        "flow_mw": limit / 2,
    })


def _mapping(tmp_path, names=("ESTEX",)):
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump({
        "boundaries": {name: {"lines": ["L1"]} for name in names}
    }), encoding="utf-8")
    return path


def _committed(tmp_path, frames):
    path = tmp_path / "committed.csv.gz"
    pd.concat(frames, ignore_index=True).to_csv(
        path, index=False, date_format=cache_builder.DATE_FORMAT,
        compression={"method": "gzip", "mtime": 0},
    )
    return path


def _outage_cfg(tmp_path, mapping, committed, **extra):
    neso = {
        "boundary_mapping": str(mapping),
        "cache_dir": str(tmp_path / "cache"),
        "committed_cache": str(committed),
        **extra,
    }
    return {"neso": neso}


@pytest.fixture
def no_network(monkeypatch):
    import requests

    calls = []

    def _refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise requests.exceptions.ConnectionError("network blocked in test")

    monkeypatch.setattr(requests, "get", _refuse)
    return calls


def test_boundary_limits_served_from_committed_cache(tmp_path, no_network):
    snapshots = _snapshots()
    committed = _committed(tmp_path, [_limits("ESTEX", "2020-02-28", 6 * 48)])
    cfg = _outage_cfg(tmp_path, _mapping(tmp_path), committed)

    inputs = sn._load_neso_boundary_inputs(cfg, snapshots, LOGGER)

    assert not no_network
    df = inputs["neso_df"]
    assert set(df["boundary"]) == {"ESTEX"}
    # Same window as the API query: [first snapshot, last snapshot + 1 h)
    assert df["date"].min() == snapshots[0]
    assert df["date"].max() < snapshots[-1] + pd.Timedelta(hours=1)
    assert len(df) == 2 * len(snapshots)


def test_missing_boundary_raises_instead_of_skipping(tmp_path, no_network):
    committed = _committed(tmp_path, [_limits("ESTEX", "2020-02-28", 6 * 48)])
    cfg = _outage_cfg(tmp_path, _mapping(tmp_path, ("ESTEX", "SCOTEX")), committed)

    with pytest.raises(RuntimeError, match="SCOTEX.*build_neso_boundary_limit_cache"):
        sn._load_neso_boundary_inputs(cfg, _snapshots(), LOGGER)
    assert not no_network


def test_window_outside_committed_range_raises(tmp_path, no_network):
    committed = _committed(tmp_path, [_limits("ESTEX", "2020-02-28", 6 * 48)])
    cfg = _outage_cfg(tmp_path, _mapping(tmp_path), committed)

    with pytest.raises(RuntimeError, match="ESTEX"):
        sn._load_neso_boundary_inputs(cfg, _snapshots("2015-01-01"), LOGGER)


def test_partial_coverage_uses_gap_fill_without_fetching(tmp_path, no_network):
    snapshots = _snapshots()
    committed = _committed(tmp_path, [_limits("ESTEX", "2020-03-01", 24)])  # 12 h only
    cfg = _outage_cfg(tmp_path, _mapping(tmp_path), committed)

    inputs = sn._load_neso_boundary_inputs(cfg, snapshots, LOGGER)

    assert not no_network
    assert inputs["gap_fill_mode"] == "interpolate"
    assert len(inputs["neso_df"]) == 24


def test_api_fetch_failure_raises_when_enabled(tmp_path, no_network):
    committed = _committed(tmp_path, [_limits("ESTEX", "2020-03-01", 24)])
    cfg = _outage_cfg(tmp_path, _mapping(tmp_path), committed, allow_api_fetch=True)

    with pytest.raises(RuntimeError, match="API fetch failed"):
        sn._load_neso_boundary_inputs(cfg, _snapshots(), LOGGER)
    assert len(no_network) == 1


def test_complete_window_cache_still_read_first(tmp_path, no_network):
    snapshots = _snapshots()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    tag = f"{snapshots[0]:%Y%m%d}_{snapshots[-1]:%Y%m%d}"
    _limits("ESTEX", "2020-03-01", 2 * len(snapshots), limit=777.0).to_csv(
        cache_dir / f"neso_ESTEX_{tag}.csv", index=False
    )
    committed = _committed(tmp_path, [_limits("ESTEX", "2020-02-28", 6 * 48)])
    cfg = _outage_cfg(tmp_path, _mapping(tmp_path), committed)

    inputs = sn._load_neso_boundary_inputs(cfg, snapshots, LOGGER)

    assert set(inputs["neso_df"]["limit_mw"]) == {777.0}


def test_builder_keeps_mapped_boundaries_and_writes_sidecar(tmp_path):
    raw = pd.DataFrame({
        "Constraint Group": ["ESTEX", "ESTEX", "OTHER"],
        "Date (GMT/BST)": ["2020-01-01T00:00:00", "2020-01-01T00:30", "2020-01-01T00:00:00"],
        "Limit (MW)": [100, 110, 5],
        "Flow (MW)": [50, 55, 1],
        "Unnamed: 4": [None, None, None],
    })
    source = tmp_path / "export.csv"
    raw.to_csv(source, index=False)
    output = tmp_path / "limits.csv.gz"

    meta = cache_builder.build_cache(source, output, [_mapping(tmp_path)])

    df = pd.read_csv(output)
    assert list(df.columns) == ["boundary", "date", "limit_mw", "flow_mw"]
    assert df["boundary"].tolist() == ["ESTEX", "ESTEX"]
    assert df["date"].tolist() == ["2020-01-01T00:00:00", "2020-01-01T00:30:00"]
    assert meta["rows"] == 2 and meta["boundaries"] == ["ESTEX"]
    sidecar = json.loads(cache_builder.sidecar_path(output).read_text(encoding="utf-8"))
    assert sidecar["source_sha256"] == meta["source_sha256"]


def test_builder_refuses_export_without_a_mapped_boundary(tmp_path):
    raw = pd.DataFrame({
        "Constraint Group": ["OTHER"], "Date (GMT/BST)": ["2020-01-01T00:00:00"],
        "Limit (MW)": [1], "Flow (MW)": [1],
    })
    with pytest.raises(ValueError, match="ESTEX"):
        cache_builder.normalise_neso_export(raw, ["ESTEX"])


def test_committed_cache_covers_every_mapped_boundary():
    committed = REPO_ROOT / sn.NESO_COMMITTED_BOUNDARY_CACHE
    meta = json.loads(cache_builder.sidecar_path(committed).read_text(encoding="utf-8"))
    mapped = cache_builder.mapped_neso_boundaries(
        [REPO_ROOT / p for p in cache_builder.DEFAULT_MAPPINGS]
    )
    assert meta["boundaries"] == mapped
    df = pd.read_csv(committed, usecols=["boundary"])
    assert sorted(df["boundary"].unique()) == mapped
    assert len(df) == meta["rows"]
    # The historical validation years (2020, 2024) are inside the snapshot.
    assert meta["date_min"] <= "2020-01-01" and meta["date_max"] >= "2024-12-31"


def test_elexon_bod_failed_day_raises_before_writing(tmp_path, monkeypatch):
    from scripts.market import elexon_data

    def _process(task):
        if task["date"] == "2020-01-02":
            raise ConnectionError("blocked")
        return {"date": task["date"], "from_cache": True, "rows": 0,
                "offers": pd.DataFrame(), "bids": pd.DataFrame(),
                "offer_ladders": pd.DataFrame(), "bid_ladders": pd.DataFrame()}

    monkeypatch.setattr(elexon_data, "_check_requests_available", lambda: None)
    monkeypatch.setattr(elexon_data, "_process_bod_day", _process)
    processed = tmp_path / "processed"
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="1 of 3 days: 2020-01-02"):
        elexon_data.retrieve_elexon_market_data(
            "2020-01-01", "2020-01-03", str(out), logger=LOGGER,
            raw_bod_cache_dir=None, processed_bod_cache_dir=str(processed),
        )
    assert not processed.exists()
    assert not any(out.iterdir())


@pytest.fixture
def dukes(tmp_path, monkeypatch):
    """The DUKES script module (it reads ``snakemake.log`` at import)."""
    import builtins
    import importlib
    import sys
    from types import SimpleNamespace

    if "scripts.generators.DUKES_generator_data" not in sys.modules:
        monkeypatch.setattr(
            builtins, "snakemake",
            SimpleNamespace(log=[str(tmp_path / "dukes.log")]), raising=False,
        )
    module = importlib.import_module("scripts.generators.DUKES_generator_data")
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    return module


def _dukes_frame():
    return pd.DataFrame({
        "station_name": ["Newplace"],
        "fuel_type": ["Natural Gas"],
        "x_coord": [float("nan")],
        "y_coord": [float("nan")],
        "location": ["Somewhere"],
    })


@pytest.mark.parametrize("function", ["geocode_from_nominatim", "geocode_from_location_name"])
def test_nominatim_network_error_is_not_cached_as_failure(tmp_path, dukes, no_network, function):
    cache = tmp_path / "nominatim_cache.csv"

    with pytest.raises(RuntimeError, match="Newplace"):
        getattr(dukes, function)(_dukes_frame(), cache_file=str(cache))
    assert no_network
    assert not (tmp_path / "nominatim_cache_failures.txt").exists()


def test_nominatim_empty_answer_still_cached_as_failure(tmp_path, monkeypatch, dukes):
    class _Empty:
        status_code = 200

        @staticmethod
        def json():
            return []

    monkeypatch.setattr(dukes.requests, "get", lambda *a, **k: _Empty())
    cache = tmp_path / "nominatim_cache.csv"

    dukes.geocode_from_nominatim(_dukes_frame(), cache_file=str(cache))

    failures = (tmp_path / "nominatim_cache_failures.txt").read_text().split()
    assert failures == ["Newplace"]


def test_postcode_network_failure_recorded(tmp_path, monkeypatch, no_network):
    from scripts.utilities.postcode_geocoder import PostcodeGeocoder
    from scripts.utilities import postcode_geocoder as pg

    monkeypatch.setattr(pg.time, "sleep", lambda *_: None)
    geocoder = PostcodeGeocoder(cache_file=str(tmp_path / "postcodes.csv"))

    assert geocoder.geocode("AB1 2CD") is None
    assert geocoder.network_failures == ["AB12CD"]
