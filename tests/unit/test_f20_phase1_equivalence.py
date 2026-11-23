"""F20 phase 1 (13 Oct 2026): every speed-up gives the numbers the code it replaces gave, bit for bit.

Each change keeps its old form callable (``*_reference``, or the direct call a memo
stands in front of) and is checked against it here on small synthetic inputs, with
the edge cases each one has to get right: missing values, ``-0.0`` against ``0.0``,
``None`` against ``NaN``, empty inputs, a single case, a missing comparator.
"""

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import register_io as rio
from scripts.public_ownership.levers import case_settings, lever_default, LEVERS
from config.config_loader import load_defaults


def assert_identical(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Same columns, dtypes, index and values, with object values of the same type and floats as bits."""
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.index.equals(b.index)
    for column in a.columns:
        x, y = a[column], b[column]
        assert x.dtype == y.dtype, column
        if isinstance(x.dtype, pd.CategoricalDtype):
            assert list(x.cat.categories) == list(y.cat.categories), column
            assert np.array_equal(x.cat.codes, y.cat.codes), column
        elif x.dtype == object:
            for u, v in zip(x, y):
                assert type(u) is type(v), (column, u, v)
                if isinstance(u, float):
                    assert math.copysign(1.0, u) == math.copysign(1.0, v) and ((u != u and v != v) or u == v)
        elif x.dtype.kind == "f":
            assert np.array_equal(x.to_numpy().view(np.int64), y.to_numpy().view(np.int64)), column


# ── register_io: restore, levers, case split, settings, reader ────────────────

TEXT_COLUMNS = {
    "all_numeric": ["1", "2.5", "-3e2", "4", "5", "6"],
    "numeric_missing": ["1", None, "2", None, "3", "4"],
    "mixed": ["a", "1", None, "b", "2", "c"],
    "nan_text": ["nan", "1", "2", "3", "4", "5"],
    "all_missing": [None] * 6,
    "big_ints": ["12345678901234567890", "-1", "2", "3", "4", "5"],
    "padded": [" 1", "2 ", "3", "4", "5", "6"],
}
BOOL_COLUMNS = {
    "bool_plain": ["True", "False", "True", "True", "False", "True"],
    "bool_missing": ["True", None, "False", None, "True", "False"],
    "bool_all_missing": [None] * 6,
}


def _raw_frame() -> tuple[pd.DataFrame, dict]:
    data = {name: pd.Series(values, dtype=object) for name, values in {**TEXT_COLUMNS, **BOOL_COLUMNS}.items()}
    data["odd_bool"] = pd.Series(["True", 1, None, "False", True, np.nan], dtype=object)
    data["empty_kind"] = pd.Series([None] * 6, dtype=object)
    data["number"] = pd.Series([1.0, -0.0, np.nan, 2.0, 3.0, 4.0])
    kinds = {**{c: "text" for c in TEXT_COLUMNS}, **{c: "bool" for c in BOOL_COLUMNS}, "odd_bool": "bool",
             "empty_kind": "empty", "absent": "text"}
    return pd.DataFrame(data), kinds


def test_restore_matches_the_reference_column_by_column():
    frame, kinds = _raw_frame()
    old = rio._restore_reference(frame.copy(), kinds)
    new = rio._restore(frame.copy(), kinds)
    assert_identical(old, new)
    assert new["all_numeric"].dtype == float and new["mixed"].dtype == object and new["bool_plain"].dtype == bool
    # Empty frames and frames of one row keep the same typing.
    for rows in (frame.iloc[:0], frame.iloc[:1]):
        assert_identical(rio._restore_reference(rows.copy(), kinds), rio._restore(rows.copy(), kinds))


def _lever_settings() -> pd.DataFrame:
    lever_a, lever_b = list(LEVERS)[:2]
    return pd.DataFrame({
        "case_id": ["private", "a", "b", "c"],
        lever_a: [lever_default(lever_a), "an_uncatalogued_value", None, "another_one"],
        lever_b: [lever_default(lever_b)] * 4,
    }).astype(object).set_index("case_id"), [lever_a, lever_b]


@pytest.mark.parametrize("lever_dtype", ["category", "object"])
@pytest.mark.parametrize("case_ids", [
    ["b", "a", "b", "private", "a", "c", "c", "b"],        # interleaved, uncatalogued values in an odd order
    ["a"],                                                   # one row
    [],                                                      # none
])
def test_levers_attached_by_case_match_the_row_by_row_reference(lever_dtype, case_ids):
    settings, levers = _lever_settings()
    frame = pd.DataFrame({"case_id": pd.Series(case_ids, dtype=object), "x": np.arange(len(case_ids), dtype=float)})
    old, new = frame.copy(), frame.copy()
    rio._attach_levers_reference(old, settings, levers, lever_dtype)
    rio._attach_levers(new, settings, levers, lever_dtype)
    assert_identical(old, new)


def test_case_settings_reads_categoricals_as_the_row_by_row_text_did():
    settings, levers = _lever_settings()
    frame = pd.DataFrame({"case_id": pd.Series(["a"] * 3 + ["c"] * 2, dtype=object)})
    rio._attach_levers(frame, settings, levers, "category")

    def reference(rows):
        out = {}
        for name in LEVERS:
            values = rows[name].dropna().astype(str).unique() if name in rows.columns else []
            if len(values) > 1:
                raise ValueError(f"a case carries more than one {name}: {sorted(values)}")
            out[name] = str(values[0]) if len(values) else lever_default(name)
        return out

    for case in ("a", "c"):
        rows = frame.loc[frame["case_id"].eq(case)]
        assert case_settings(rows) == reference(rows)
        assert case_settings(rows.astype({c: object for c in levers})) == reference(rows)
    # Two settings in one "case" fail the same way, with the same message.
    with pytest.raises(ValueError) as new_error:
        case_settings(frame)
    with pytest.raises(ValueError) as old_error:
        reference(frame)
    assert str(new_error.value) == str(old_error.value)


def test_distinct_settings_match_drop_duplicates():
    settings, levers = _lever_settings()
    frame = pd.DataFrame({"case_id": pd.Series(["a", "a", "b", "c", "b", "c"], dtype=object)})
    rio._attach_levers(frame, settings, levers, "category")
    assert_identical(rio._distinct_settings_reference(frame, levers), rio._distinct_settings(frame, levers))
    as_text = frame.astype({c: object for c in levers})
    assert_identical(rio._distinct_settings_reference(as_text, levers), rio._distinct_settings(as_text, levers))
    # A case carrying two settings falls back to the whole-frame rule, so the writer still refuses it.
    broken = as_text.copy()
    broken.loc[4, levers[1]] = "something_else"
    assert_identical(rio._distinct_settings_reference(broken, levers), rio._distinct_settings(broken, levers))
    assert len(rio._distinct_settings(broken, levers)) == 4


def _write_register(tmp_path, row_group_rows=10) -> tuple:
    """Five cases of seven assets; small row groups so private has its own groups and shares one."""
    lever_a = "networks"
    rows = []
    for case in ("private", "a", "b", "c", "d"):
        for i in range(7):
            rows.append({"case_id": case, lever_a: "private" if case in ("private", "a") else "acquired",
                         "asset_id": f"asset{i}", "carrier": "wind" if i % 2 else None,
                         "cfd_round": str(i) if case != "d" else ("AR4" if i == 0 else str(i)),
                         "finance_eligible": (i % 3 == 0) if case != "c" else None,
                         "value": float(i) * (1.0 if case != "b" else -1.0), "zero": -0.0 if case == "c" else 0.0})
    frame = pd.DataFrame(rows)
    path = tmp_path / "t_asset_finance_register.parquet"
    with rio.RegisterWriter(path, row_group_rows=row_group_rows) as writer:
        writer.write(frame.loc[frame["case_id"].isin(["private"])])
        writer.write(frame.loc[frame["case_id"].isin(["a", "b"])])
        writer.write(frame.loc[frame["case_id"].isin(["c"])])
        writer.write(frame.loc[frame["case_id"].isin(["d"])])
    return path, frame


@pytest.mark.parametrize("row_group_rows", [10, 1000])
def test_the_reader_with_pinned_cases_reads_what_read_register_reads(tmp_path, row_group_rows):
    path, _ = _write_register(tmp_path, row_group_rows)
    for columns in (None, ["case_id", "asset_id", "value", "networks"], ["value"]):
        reader = rio.RegisterReader(path, columns=columns, pinned=["private"])
        for cases in (["private", "a"], ["private", "c"], ["private"], ["private", "a", "b", "d"], ["d"], ["zzz"],
                      ["private", "a"], ["private", "zzz"]):
            assert_identical(rio.read_register(path, columns=columns, cases=cases), reader.read(cases))


def test_case_rows_split_a_run_as_the_filter_did(tmp_path):
    path, _ = _write_register(tmp_path, 1000)
    rows = rio.CaseRows(path)
    for case in ("private", "a", "b", "c", "d"):
        run = rows._runs[case]
        whole = rio.read_register(path, cases=list(run))
        expected = whole.loc[whole["case_id"].eq(case)] if len(run) > 1 else whole
        assert_identical(expected, rows.rows(case))


def test_memo_and_prefetch_give_the_direct_results_and_raise_where_the_stage_asked(tmp_path):
    path, _ = _write_register(tmp_path, 10)
    rows = rio.CaseRows(path, cache_cases=1)
    total = lambda frame: {"value": float(frame["value"].sum())}  # noqa: E731

    def fails(frame):
        raise KeyError("not this case")

    requests = [("total", (case,), total) for case in ("d", "a", "private", "b")]
    requests += [("pair", ("a", "private"), lambda f: len(f)), ("fails", ("c",), fails), ("missing", ("zzz",), total)]
    assert rows.prefetch(requests) == 5
    for case in ("d", "a", "private", "b", "c"):
        direct = total(rio.read_register(path, cases=[case]))
        assert rows.memo("total", (case,), total) == direct
    out = rows.memo("total", ("a",), total)
    out["value"] = -1.0                                # each caller has its own copy
    assert rows.memo("total", ("a",), total)["value"] == total(rio.read_register(path, cases=["a"]))["value"]
    assert rows.memo("pair", ("a", "private"), lambda f: len(f)) == 14
    with pytest.raises(KeyError, match="not this case"):
        rows.memo("fails", ("c",), fails)
    rows.clear_memo()
    assert rows._memo == {}


def test_frames_identical_is_strict():
    base = pd.DataFrame({"x": [1.0, np.nan, 0.0], "t": ["a", None, "c"], "n": [1, 2, 3]})
    assert rio.frames_identical(base, base.copy())
    assert not rio.frames_identical(base, base.assign(x=[1.0, np.nan, -0.0]))
    assert not rio.frames_identical(base, base.assign(t=["a", np.nan, "c"]))
    assert not rio.frames_identical(base, base.assign(n=[1.0, 2.0, 3.0]))
    assert not rio.frames_identical(base, base[["t", "x", "n"]])
    cat = base.assign(t=pd.Categorical(["a", None, "c"], categories=["a", "c"]))
    assert rio.frames_identical(cat, cat.copy())
    assert not rio.frames_identical(cat, base.assign(t=pd.Categorical(["a", None, "c"], categories=["c", "a"])))
    assert rio.frames_identical(base.iloc[:0], base.iloc[:0].copy())


# ── the bill bridge ───────────────────────────────────────────────────────────

def _bridge_register() -> tuple[pd.DataFrame, pd.DataFrame]:
    def case_rows(case, rr_scale=1.0, zero=0.0, support="CfD"):
        return [{"case_id": case, "asset_id": f"a{i}", "finance_technology": ["wind", "solar", "gas"][i % 3],
                 "finance_vintage": "new_build" if i % 4 else "legacy", "capacity_mw": 10.0 + i,
                 "dispatch_mwh": 100.0 * i, "annual_required_revenue_gbp": (1000.0 + 7.1 * i) * rr_scale + zero,
                 "annual_capital_recovery_gbp": (600.0 + 3.3 * i) * rr_scale,
                 "support_type": [support, "merchant", None, "cfd"][i % 4]} for i in range(9)]
    rows = case_rows("private", rr_scale=1.1) + case_rows("private@levy", rr_scale=1.2)
    rows += case_rows("public_ownership") + case_rows("public_ownership@grid_x") + case_rows("public_ownership@grid_y")
    rows += case_rows("public_ownership@levy") + case_rows("public_control", rr_scale=0.9)
    rows += case_rows("public_control@zero", rr_scale=0.9, zero=-0.0) + case_rows("public_control@nan", support=None)
    register = pd.DataFrame(rows)
    register.loc[register["case_id"].eq("public_control@nan") & register["asset_id"].eq("a3"),
                 "annual_required_revenue_gbp"] = np.nan
    # A zero against a negative zero: equal under ==, never reused.
    for case, zero in (("public_control", 0.0), ("public_control@zero", -0.0)):
        register.loc[register["case_id"].eq(case) & register["asset_id"].eq("a0"), "annual_required_revenue_gbp"] = zero
    cases = [c for c in dict.fromkeys(register["case_id"])]
    summary = pd.DataFrame({"case_id": cases,
                            "annual_financing_saving_vs_private_gbp": [0.0] + [100.0 + i for i in range(len(cases) - 1)],
                            "consumer_equivalent_saving_gbp_per_mwh_demand": [0.0] + [0.5] * (len(cases) - 1),
                            "comparator_case_id": ["private" if "@levy" not in c else "private@levy" for c in cases]})
    summary.loc[summary["case_id"].eq("public_control"), "comparator_case_id"] = "private@missing"
    return register, summary


def _bridge_reference(monkeypatch, run):
    from scripts.public_ownership import build_consumer_bill_financing_bridge as bridge

    with monkeypatch.context() as patch:
        patch.setattr(bridge._SavingsMemo, "savings",
                      lambda self, case_id, rows, private: bridge._technology_savings(case_id, rows, private))
        patch.setattr(bridge, "_new_cfd_saving_by_case",
                      lambda table: (lambda case_id: bridge._new_cfd_saving_reference(table, case_id)))
        return run()


def test_the_bridge_matches_its_reference_whole_and_streamed(monkeypatch, tmp_path):
    from scripts.public_ownership import build_consumer_bill_financing_bridge as bridge

    register, summary = _bridge_register()
    whole = lambda: bridge.build_consumer_bill_financing_bridge(register, summary)  # noqa: E731
    path = tmp_path / "b_asset_finance_register.parquet"
    rio.write_register(register, path)
    streamed = lambda: bridge.build_consumer_bill_financing_bridge_from_file(path, summary, batch_cases=2)  # noqa: E731
    for run in (whole, streamed):
        new, old = run(), _bridge_reference(monkeypatch, run)
        assert_identical(old[0], new[0])
        assert_identical(old[1], new[1])
        assert json.dumps(old[2]) == json.dumps(new[2])


def test_the_savings_memo_reuses_only_identical_rows_against_the_same_comparator():
    from scripts.public_ownership import build_consumer_bill_financing_bridge as bridge

    register, _ = _bridge_register()
    private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
    other = register.loc[register["case_id"].eq("private@levy")].set_index("asset_id")
    memo = bridge._SavingsMemo()
    for case, comparator in (("public_ownership", private), ("public_ownership@grid_x", private),
                             ("public_ownership@grid_y", other), ("public_control", private),
                             ("public_control@zero", private), ("public_control@nan", private)):
        rows = register.loc[register["case_id"].eq(case)]
        assert_identical(bridge._technology_savings(case, rows, comparator), memo.savings(case, rows, comparator))
    assert memo.hits == 1 and memo.misses == 5      # grid_x reuses public_ownership; -0.0 and the comparator do not


def test_the_new_cfd_saving_lookup_matches_the_scan():
    from scripts.public_ownership import build_consumer_bill_financing_bridge as bridge

    register, summary = _bridge_register()
    private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
    table = pd.concat([bridge._technology_savings(c, r, private) for c, r in register.groupby("case_id")
                       if c != "private"], ignore_index=True)
    table.loc[3, "annual_financing_saving_gbp"] = np.nan
    lookup = bridge._new_cfd_saving_by_case(table)
    for case in [*summary["case_id"], "absent"]:
        old, new = bridge._new_cfd_saving_reference(table, case), lookup(case)
        assert np.float64(old).view(np.int64) == np.float64(new).view(np.int64)
    assert bridge._new_cfd_saving_by_case(pd.DataFrame())("x") == bridge._new_cfd_saving_reference(pd.DataFrame(), "x")


# ── the nationalised tariff's revenue requirement ─────────────────────────────

def _tariff_register() -> pd.DataFrame:
    from tests.unit.test_public_ownership_nationalised_tariff import _register

    base = _register()
    nat = base.loc[~base["case_id"].eq("private")]
    frames = [base]
    for case, change in (("p3@same", None), ("p3@other", ("annual_fixed_om_gbp", 1.0)),
                         ("p3@zero", ("fuel_carbon_gbp", -0.0)), ("p3@none", ("legacy_scope_group", None))):
        rows = nat.assign(case_id=case)
        if change is not None:
            column, value = change
            rows = rows.copy()
            if value is None:
                rows[column] = rows[column].where(rows[column].notna(), None).astype(object)
                rows.loc[rows.index[-1], column] = None
            elif value == 0.0 and math.copysign(1.0, value) < 0:
                rows.loc[rows[column].eq(0.0), column] = -0.0
            else:
                rows[column] = rows[column] + value
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def _same_result(a, b) -> None:
    assert json.dumps(a, sort_keys=True, default=repr) == json.dumps(b, sort_keys=True, default=repr)


def test_the_revenue_requirement_memo_gives_the_direct_result_and_reuses_only_identical_cases():
    from scripts.public_ownership.finance import NATIONALISED_CASE
    from scripts.public_ownership.nationalised_tariff import RevenueRequirementMemo, revenue_requirement

    register = _tariff_register()
    memo = RevenueRequirementMemo()
    gate = pd.Series({"old_gas": 5.0e6})
    kwargs = dict(realtime_correction_gbp=3.0e6, interconnector_retrading_gbp=2.0e6, legacy_gate_fee_gbp=gate)
    order = [NATIONALISED_CASE, "p3@same", "p3@other", "p3@zero", "p3@none", NATIONALISED_CASE, "p3@same"]
    for case in order:
        _same_result(revenue_requirement(register, case_id=case, **kwargs), memo(register, case_id=case, **kwargs))
    assert memo.misses == 4 and memo.hits == 3
    # A changed argument, or another gate-fee series with the same values, is never reused.
    _same_result(revenue_requirement(register, case_id="p3@same", interconnector_trade_gbp=15.0e6, **kwargs),
                 memo(register, case_id="p3@same", interconnector_trade_gbp=15.0e6, **kwargs))
    other_gate = dict(kwargs, legacy_gate_fee_gbp=gate.copy())
    _same_result(revenue_requirement(register, case_id="p3@same", **other_gate),
                 memo(register, case_id="p3@same", **other_gate))
    assert memo.misses == 6
    # The caller may change what it gets back without changing what the next case gets.
    lines, diagnostics = memo(register, case_id="p3@same", **kwargs)
    lines.clear()
    diagnostics["paid_as_today_carriers"].append("x")
    _same_result(revenue_requirement(register, case_id="p3@same", **kwargs), memo(register, case_id="p3@same", **kwargs))
    # Another frame with private's rows unchanged keeps the entries; changed private rows drop them.
    copy_frame = register.copy()
    hits = memo.hits
    memo(copy_frame, case_id="p3@same", **kwargs)
    assert memo.hits == hits + 1
    moved = register.copy()
    moved.loc[moved["case_id"].eq("private"), "market_revenue_gbp"] += 1.0
    _same_result(revenue_requirement(moved, case_id="p3@same", **kwargs), memo(moved, case_id="p3@same", **kwargs))
    assert memo.hits == hits + 1
    with pytest.raises(ValueError, match="register has no absent rows"):
        memo(register, case_id="absent", **kwargs)


# ── the LDES floor and the strike adjustments ─────────────────────────────────

def _floor_frame() -> pd.DataFrame:
    n = 8
    return pd.DataFrame({
        "case_id": ["public_ownership", "private", "public_ownership@x", "public_ownership", "private",
                    "public_ownership", "public_ownership@x", "private"],
        "component_type": ["StorageUnit", "StorageUnit", "Generator", "StorageUnit", "Generator", "StorageUnit",
                           "StorageUnit", "StorageUnit"],
        "carrier": ["LAES", "LAES", "wind", "Battery", "gas", "LAES", "LAES", "LAES"],
        "finance_vintage": ["new_build", "new_build", "new_build", "new_build", "legacy", "legacy", "new_build",
                            "new_build"],
        "annual_required_revenue_gbp": [100.0, 120.0, 50.0, 80.0, 40.0, 90.0, np.nan, 130.0],
        "annual_capital_recovery_gbp": [60.0, 80.0, 30.0, 50.0, 20.0, 50.0, 40.0, 90.0],
        "annual_fixed_om_gbp": [10.0] * n, "annual_variable_om_gbp": [1.0] * n,
        "wacc": [0.04, 0.07, 0.05, 0.04, np.nan, 0.04, 0.04, 0.07],
        "remaining_life_years": [25.0, 25.0, 20.0, 15.0, 10.0, 0.0, 25.0, 25.0],
        "market_revenue_gbp": [30.0, 30.0, 40.0, 20.0, 50.0, 30.0, 30.0, 30.0],
        "capacity_market_revenue_gbp": [5.0] * n, "ownership_share": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        "ldes_floor_basis": ["cost_of_debt_like", "cost_of_debt_like", "cost_of_debt_like", "technology_wacc",
                             "cost_of_debt_like", "cost_of_debt_like", "cost_of_debt_like", "technology_wacc"],
    }, index=[10, 11, 12, 13, 14, 15, 16, 17])


@pytest.mark.parametrize("rate_key", ["case", "row_owner"])
def test_the_ldes_floor_on_ldes_rows_matches_the_whole_frame(monkeypatch, rate_key):
    from scripts.public_ownership import public_account as pa

    settings = {"ldes_carriers": ["LAES"], "ldes_enabled": True, "ldes_floor_rate_key": rate_key,
                "ldes_floor_real_rate_by_case": {"public_ownership": 0.02, "private": 0.05}}
    for frame in (_floor_frame(), _floor_frame().iloc[:0], _floor_frame().assign(carrier="wind")):
        new = pa.ldes_floor_topups(frame, settings)
        with monkeypatch.context() as patch:
            patch.setattr(pa, "_floor_parts_on", lambda f, rows, rates, key: pa._floor_parts(f, rates, key))
            old = pa.ldes_floor_topups(frame, settings)
        assert_identical(old, new)


def test_strike_adjustments_add_matches_the_per_row_lookups(tmp_path):
    from scripts.market.cfd_strike_table import load_strike_table
    from scripts.public_ownership.cfd_repricing import (
        ROUTE_CFD_BILL, StrikeAdjustments, SavingsStream, _priced_rows, limit_at_cost_to_deliverable,
        repricing_settings,
    )
    from tests.unit.test_public_ownership_cfd_repricing import ASSUMPTIONS, STRIKE_TABLE
    from tests.unit.test_public_ownership_streaming import _streamed
    import yaml

    block = load_defaults()["public_ownership"]
    path = tmp_path / "g_asset_finance_register.parquet"
    _streamed(block, path)
    settings = repricing_settings(block["consumer_bill"])
    stream = SavingsStream(path, settings, batch_cases=2)
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    table = load_strike_table(STRIKE_TABLE, assumptions["cfd_strike_indexation"], 2024)
    paid = {f"{r}_{t}": 1.0e6 for r, t in table.strikes}

    class Reference(StrikeAdjustments):
        def add(self, savings):                       # the loop as it was before F20
            if self.empty or savings.empty:
                return
            priced = _priced_rows(savings, self.table)
            r1 = priced.loc[priced["route"].eq(ROUTE_CFD_BILL)]
            numerator_term = r1.groupby(["case_id", "strike_key"])["saving_bill_term_gbp"].sum()
            numerator_life = r1.groupby(["case_id", "strike_key"])["saving_economic_life_gbp"].sum()
            eligible_assets = r1.groupby(["case_id", "strike_key"])["asset_id"].size()
            eligible_by_case = priced.groupby(["case_id", "strike_key"])["r1_eligible"].any()
            batch_any = priced.groupby("strike_key")["r1_eligible"].any()
            self._eligible_any = self._eligible_any | batch_any.reindex(self._eligible_any.index, fill_value=False)
            for case_id in savings["case_id"].drop_duplicates():
                method = (savings.loc[savings["case_id"].eq(case_id), "method"].iloc[0] if "method" in savings
                          else self.settings["method"])
                for strike_key, info in self.detail.iterrows():
                    strike_private = float(self.table.strikes[(info["cfd_round"], info["technology"])])
                    dispatch = float(self.dispatch_by_key.get(strike_key, 0.0))
                    saving_term = float(numerator_term.get((case_id, strike_key), 0.0))
                    saving_life = float(numerator_life.get((case_id, strike_key), 0.0))
                    delta = saving_term / dispatch if dispatch > 0 else 0.0
                    offset = float((self.case_offsets or {}).get(case_id, {}).get(strike_key, 0.0))
                    delta += offset
                    self._rows.append({
                        "case_id": case_id, "carrier": info["carrier"], "technology": info["technology"],
                        "cfd_round": info["cfd_round"], "strike_key": strike_key,
                        "cfd_award_year": info["cfd_award_year"], "contract_vintage": info["contract_vintage"],
                        "eligible": (bool(eligible_by_case[(case_id, strike_key)])
                                     if (case_id, strike_key) in eligible_by_case.index else None),
                        "is_fallback_strike": bool(info["is_fallback_strike"]),
                        "generators": int(self.assets_by_key.get(strike_key, 0)),
                        "eligible_generators": int(eligible_assets.get((case_id, strike_key), 0)),
                        "strike_private_gbp_per_mwh_2024": strike_private, "delta_gbp_per_mwh_2024": delta,
                        "strike_case_gbp_per_mwh_2024": strike_private - delta, "eligible_dispatch_mwh": dispatch,
                        "r1_saving_gbp_2024": saving_term, "register_saving_gbp_2024": saving_life, "method": method,
                        **({"strike_basis_offset_gbp_per_mwh_2024": offset} if self.case_offsets is not None else {}),
                    })

    for offsets in (None, {"public_ownership": {key: 0.5 for key in list(paid)[:2]}}):
        new = StrikeAdjustments(stream.private_savings, table, settings, None, paid, case_offsets=offsets)
        old = Reference(stream.private_savings, table, settings, None, paid, case_offsets=offsets)
        for _, savings in stream:
            savings, _ = limit_at_cost_to_deliverable(savings, table, None, paid)
            new.add(savings)
            old.add(savings)
            no_method = savings.drop(columns=["method"]) if "method" in savings.columns else savings
            new.add(no_method)
            old.add(no_method)
        assert_identical(old.frame(), new.frame())


# ── the Exchequer ledger: one sweep, the same ledger ──────────────────────────

def _direct(monkeypatch):
    """CaseRows as before F20: nothing prefetched, every memo computed again from the rows."""
    monkeypatch.setattr(rio.CaseRows, "prefetch", lambda self, requests: 0)
    monkeypatch.setattr(rio.CaseRows, "memo", lambda self, label, case_ids, compute: compute(self.rows(*case_ids)))


@pytest.mark.parametrize("source", ["frame", "file"])
def test_the_ledger_from_memoised_quantities_matches_the_direct_passes(monkeypatch, tmp_path, source):
    from tests.unit import test_public_ownership_investor_ledger as fixture

    frame = fixture._register()

    def run():
        register = frame.copy()
        if source == "file":
            path = tmp_path / "l_asset_finance_register.parquet"
            if not path.exists():
                rio.write_register(register, path)
            from scripts.public_ownership.exchequer_ledger import load_register

            register = load_register(path, list(dict.fromkeys(register["case_id"])))
        monkeypatch.setattr(fixture, "_register", lambda: register)
        return fixture._build(fixture._account(p2_retained=20 * fixture.M))

    new_frame, new_summary = run()
    with monkeypatch.context() as patch:
        _direct(patch)
        old_frame, old_summary = run()
    assert_identical(old_frame, new_frame)
    assert json.dumps(old_summary, sort_keys=True, default=repr) == json.dumps(new_summary, sort_keys=True, default=repr)


def test_the_ledger_request_list_names_each_pass_once(monkeypatch):
    from scripts.public_ownership import exchequer_ledger as led
    from tests.unit import test_public_ownership_investor_ledger as fixture

    register = rio.CaseRows(fixture._register())
    bill = {"cases": {"private": {}, "p2": {}}}
    requests = led._register_requests(register, bill=bill, balancing=None, tariff=None, institution=None,
                                      investor={"legacy_valuation": None}, catalogue={}, contested=None, hpc=True,
                                      has_upstream_ct=True, wedges={}, private_tax=None)
    keys = [(label, ids) for label, ids, _ in requests]
    assert len(keys) == len(set(keys))
    assert ("register_position", ("p2",)) in keys and ("register_components", ("p2", "private")) in keys
    assert ("tax_upstream", ("p2",)) in keys and ("hpc_state_owns", ("private",)) in keys
    copy.deepcopy(keys)
