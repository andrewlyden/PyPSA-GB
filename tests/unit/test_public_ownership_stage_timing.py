"""F20 phase 0: opt-in stage timing (scripts/public_ownership/stage_timing.py) changes nothing unless switched on."""

import types

import pandas as pd
import pytest

from scripts.public_ownership import stage_timing


@pytest.fixture(autouse=True)
def _fresh_records():
    stage_timing._records.clear()
    yield
    stage_timing._records.clear()


def _module():
    module = types.ModuleType("fake_stage")

    def add(a, b=1):
        return a + b

    def rows(n):
        yield from range(n)

    def fails():
        raise ValueError("boom")

    class Stream:
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            yield from range(self.n)

        def total(self):
            return sum(self)

        @classmethod
        def make(cls, n):
            return cls(n)

    module.add, module.rows, module.fails, module.Stream = add, rows, fails, Stream
    return module


def test_off_by_default_nothing_is_wrapped(monkeypatch):
    monkeypatch.delenv(stage_timing.ENV_VAR, raising=False)
    module = _module()
    original = module.add
    assert stage_timing.instrument(module, ["add", "rows"], prefix="x") == []
    assert module.add is original
    assert stage_timing.instrument_register_io() == [] and stage_timing.instrument_cfd_repricing() == []
    lap = stage_timing.laps("x")
    lap("a")
    with stage_timing.timed("x.block"):
        pass
    assert stage_timing._records == {} and stage_timing.flush() is None


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_falsy_values_keep_it_off(monkeypatch, value):
    monkeypatch.setenv(stage_timing.ENV_VAR, value)
    assert not stage_timing.enabled()


def test_on_it_times_without_changing_results(monkeypatch, tmp_path):
    monkeypatch.setenv(stage_timing.ENV_VAR, str(tmp_path / "timing"))
    module = _module()
    labels = stage_timing.instrument(module, ["add", "rows", "fails", "absent"], prefix="stage")
    assert labels == ["stage.add", "stage.rows", "stage.fails"]
    # Wrapped twice is wrapped once.
    assert stage_timing.instrument(module, ["add"], prefix="stage") == []
    assert module.add(2, b=3) == 5 and module.add.__name__ == "add"
    assert list(module.rows(3)) == [0, 1, 2]
    with pytest.raises(ValueError, match="boom"):
        module.fails()
    assert stage_timing.instrument(module.Stream, ["__iter__", "total", "make"], prefix="stage.Stream") == [
        "stage.Stream.__iter__", "stage.Stream.total"]
    stream = module.Stream.make(4)
    assert list(stream) == [0, 1, 2, 3] and stream.total() == 6
    lap = stage_timing.laps("stage.main")
    lap("setup")
    with stage_timing.timed("stage.block"):
        pass
    records = stage_timing._records
    assert records["stage.add"][0] == 1
    assert records["stage.rows"][0] == 4          # three items and the exhausting call
    assert records["stage.fails"][0] == 1         # a raising call is still timed
    assert records["stage.Stream.__iter__"][0] == 10 and records["stage.Stream.total"][0] == 1
    assert {"stage.main.setup", "stage.block"} <= set(records)
    path = stage_timing.flush()
    table = pd.read_csv(path)
    assert list(table.columns) == list(stage_timing.COLUMNS)
    assert table["label"].iloc[0].endswith(".total") and "stage.add" in set(table["label"])
    assert stage_timing.flush() is None           # once per process


def test_mapping_namespaces_and_the_summary(monkeypatch, tmp_path):
    folder = tmp_path / "timing"
    monkeypatch.setenv(stage_timing.ENV_VAR, str(folder))
    namespace = {"double": lambda x: 2 * x}
    stage_timing.instrument(namespace, ["double"], prefix="ns")
    assert namespace["double"](4) == 8
    stage_timing.flush()
    stage_timing.record("ns.double", 1.0)
    stage_timing.flush()
    table = stage_timing.summarise(folder)
    row = table.loc[table["label"].eq("ns.double")].iloc[0]
    assert row["calls"] == 2 and row["processes"] == 1 and row["total_s"] >= 1.0
    text = stage_timing.to_markdown(table)
    assert text.startswith("| Script |") and "ns.double" in text
    stage_timing.main(["summarise", str(folder), "--out", str(tmp_path / "table.md")])
    assert (tmp_path / "table.md").read_text(encoding="utf-8") == text


def test_run_group_is_read_off_the_command_line():
    argv = ["resources/public_ownership/cp30_2030_cp30_network_public_bids_p3_asset_finance_register.parquet", "x.csv"]
    assert stage_timing.run_group_from_argv(argv) == "cp30_2030_cp30_network_public_bids_p3"
    assert stage_timing.run_group_from_argv(["--cases", "a"]) == ""


def test_the_register_readers_are_timed_and_read_the_same(monkeypatch, tmp_path):
    import numpy as np

    from scripts.public_ownership import register_io
    from scripts.public_ownership.levers import DEFAULT_SETTINGS

    rows = [{"case_id": case, "asset_id": a, "capacity_mw": float(i), **DEFAULT_SETTINGS}
            for case in ("private", "public_ownership") for i, a in enumerate("abc")]
    path = tmp_path / "g_asset_finance_register.parquet"
    register_io.write_register(pd.DataFrame(rows), path)
    before = register_io.read_register(path)
    originals = {name: getattr(register_io, name) for name in stage_timing.REGISTER_IO_FUNCTIONS}
    method = register_io.CaseRows._case
    monkeypatch.setenv(stage_timing.ENV_VAR, str(tmp_path / "timing"))
    try:
        assert "register_io.read_register" in stage_timing.instrument_register_io()
        after = register_io.read_register(path)
        pd.testing.assert_frame_equal(before, after)
        batches = list(register_io.iter_register_cases(path, batch_cases=1))
        assert [b["case_id"].iloc[0] for b in batches] == ["private", "public_ownership"]
        assert np.array_equal(register_io.CaseRows(path).rows("public_ownership")["capacity_mw"], [0.0, 1.0, 2.0])
        assert stage_timing._records["register_io.iter_register_cases"][0] == 3
        assert stage_timing._records["register_io.CaseRows._case"][0] == 1
    finally:
        for name, function in originals.items():
            setattr(register_io, name, function)
        register_io.CaseRows._case = method
        for cls, methods in stage_timing.REGISTER_IO_METHODS.items():
            for name in methods:
                current = getattr(getattr(register_io, cls), name)
                if getattr(current, stage_timing.WRAPPED_ATTR, None):
                    setattr(getattr(register_io, cls), name, current.__wrapped__)
