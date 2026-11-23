"""F20 phase 0: the output-comparison tool (scripts/public_ownership/compare_outputs.py) on small fixtures."""

import json

import numpy as np
import pandas as pd

from scripts.public_ownership.compare_outputs import Options, compare_file, compare_roots, failed, main
from scripts.public_ownership.levers import DEFAULT_SETTINGS
from scripts.public_ownership.register_io import write_register


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _roots(tmp_path):
    ref, cand = tmp_path / "ref" / "resources", tmp_path / "cand" / "resources"
    ref.mkdir(parents=True)
    cand.mkdir(parents=True)
    return ref, cand


SUMMARY = "case_id,assets,annual_saving_gbp,note\nprivate,3,0.0,base\npublic_ownership,3,0.1,state\n"


def test_identical_roots_pass(tmp_path, capsys):
    ref, cand = _roots(tmp_path)
    for root in (ref, cand):
        _write(root, "public_ownership/g_financing_summary.csv", SUMMARY)
        _write(root, "public_ownership/g_summary.json", json.dumps({"cases": {"private": {"bill": 1.5}}}))
    assert main([str(ref), str(cand)]) == 0
    assert "2 files: equal 2" in capsys.readouterr().out


def test_row_order_and_float_formatting_are_not_differences(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "g_financing_summary.csv", SUMMARY)
    # Rows swapped; 0.1 written with seventeen digits, 0.0 as 0, 3 as 3.0.
    _write(cand, "g_financing_summary.csv",
           "case_id,assets,annual_saving_gbp,note\npublic_ownership,3.0,0.10000000000000001,state\nprivate,3,0,base\n")
    results = compare_roots(ref, cand, Options())
    assert [d.status for d in results] == ["equal"]
    assert results[0].values_compared == 6


def test_an_exact_difference_fails_and_a_tolerance_passes_it(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "g_financing_summary.csv", SUMMARY)
    _write(cand, "g_financing_summary.csv", SUMMARY.replace("0.1,state", "0.1000000001,state"))
    exact = compare_roots(ref, cand, Options())[0]
    assert exact.status == "differs" and exact.values_differ == 1
    assert np.isclose(exact.max_abs, 1e-10) and np.isclose(exact.max_rel, 1e-9)
    assert exact.worst == "annual_saving_gbp" and "public_ownership" in exact.examples[0]
    assert compare_roots(ref, cand, Options(atol=1e-9))[0].status == "equal"
    assert compare_roots(ref, cand, Options(rtol=1e-8))[0].status == "equal"


def test_columns_and_rows_on_one_side_are_reported(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "g_financing_summary.csv", SUMMARY)
    _write(cand, "g_financing_summary.csv",
           "case_id,assets,annual_saving_gbp,extra\nprivate,3,0.0,1\npublic_control,3,0.2,1\n")
    diff = compare_roots(ref, cand, Options())[0]
    assert diff.status == "differs"
    assert diff.columns_added == ["extra"] and diff.columns_removed == ["note"]
    assert diff.rows_only_reference == 1 and diff.rows_only_candidate == 1
    assert any("public_control" in e for e in diff.examples)


def test_text_differences_and_path_replacement(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "g_bill.csv", "case_id,component_id,source\nprivate,cfd,resources/public_ownership/g.parquet\n")
    _write(cand, "g_bill.csv", "case_id,component_id,source\nprivate,cfd,runs/new/resources/public_ownership/g.parquet\n")
    assert compare_roots(ref, cand, Options())[0].status == "differs"
    assert compare_roots(ref, cand, Options(replace=[("runs/new/resources", "resources")]))[0].status == "equal"


def test_repeated_keys_are_matched_in_file_order(tmp_path):
    ref, cand = _roots(tmp_path)
    text = "case_id,step,value\nprivate,a,1.0\nprivate,a,2.0\nprivate,b,3.0\n"
    _write(ref, "g_recon.csv", text)
    _write(cand, "g_recon.csv", "case_id,step,value\nprivate,b,3.0\nprivate,a,1.0\nprivate,a,2.0\n")
    assert compare_roots(ref, cand, Options())[0].status == "equal"
    _write(cand, "g_recon.csv", "case_id,step,value\nprivate,a,2.0\nprivate,a,1.0\nprivate,b,3.0\n")
    assert compare_roots(ref, cand, Options())[0].values_differ == 2


def test_explicit_keys(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "t.csv", "a,b,v\n1,x,1.0\n2,y,2.0\n")
    _write(cand, "t.csv", "a,b,v\n2,y,2.0\n1,x,1.0\n")
    assert compare_roots(ref, cand, Options(keys={"t.csv": ["a"]}))[0].status == "equal"


def test_json_trees(tmp_path):
    ref, cand = _roots(tmp_path)
    a = {"cases": {"private": {"bill": 1.0, "lines": [1, 2, 3]}}, "records": [
        {"case_id": "private", "x": 1}, {"case_id": "p2", "x": 2}], "written_utc": "2026-10-13T00:00:00Z",
        "source": "resources/public_ownership/g.parquet", "flag": True, "missing": None}
    b = json.loads(json.dumps(a))
    b["records"] = list(reversed(b["records"]))          # matched by case_id, not position
    b["written_utc"] = "2026-10-14T00:00:00Z"            # a write time: ignored by default
    _write(ref, "g.json", json.dumps(a))
    _write(cand, "g.json", json.dumps(b))
    assert compare_roots(ref, cand, Options())[0].status == "equal"
    b["cases"]["private"]["bill"] = 1.25
    b["cases"]["private"]["lines"] = [1, 2]
    b["new_key"] = 1
    del b["flag"]
    _write(cand, "g.json", json.dumps(b))
    diff = compare_roots(ref, cand, Options())[0]
    assert diff.status == "differs"
    assert diff.max_abs == 0.25 and diff.worst == "cases.private.bill"
    assert diff.rows_only_reference == 1 and diff.rows_only_candidate == 1
    assert any("list of 3 -> 2" in e for e in diff.examples)


def test_files_on_one_side_fail_unless_allowed(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "a.csv", SUMMARY)
    _write(cand, "a.csv", SUMMARY)
    _write(ref, "b.json", "{}")
    _write(cand, "c.json", "{}")
    _write(cand, "stage_timing/x.csv", "label,total_s\nx,1\n")   # excluded by default
    _write(cand, "notes.txt", "not an output")                   # not a table or JSON
    results = compare_roots(ref, cand, Options())
    assert sorted((d.path, d.status) for d in results) == [
        ("a.csv", "equal"), ("b.json", "only_in_reference"), ("c.json", "only_in_candidate")]
    assert failed(results) and not failed(results, missing_ok=True)
    assert main([str(ref), str(cand)]) == 1
    assert main([str(ref), str(cand), "--missing-ok"]) == 0
    assert main([str(ref), str(cand), "--include", "a.csv"]) == 0


def test_an_unreadable_file_is_an_error(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "g.json", "{}")
    _write(cand, "g.json", "{not json")
    diff = compare_roots(ref, cand, Options())[0]
    assert diff.status == "error" and "JSONDecodeError" in diff.error


def test_parquet_tables(tmp_path):
    ref, cand = _roots(tmp_path)
    frame = pd.DataFrame({"case_id": ["private", "p2"], "value": [1.0, np.nan]})
    frame.to_parquet(ref / "t.parquet")
    frame.iloc[::-1].to_parquet(cand / "t.parquet")
    assert compare_roots(ref, cand, Options())[0].status == "equal"


def _register(cases=("private", "public_ownership"), value=1.0):
    rows = []
    for case in cases:
        for i, asset in enumerate(("a", "b", "c")):
            rows.append({"case_id": case, "component_type": "Generator", "asset_id": asset,
                         "carrier": ["wind_offshore", "solar_pv", None][i], "capacity_mw": 100.0 * (i + 1),
                         "annual_required_revenue_gbp": value * (i + 1) if case != "private" else 2.0 * (i + 1),
                         **{k: ("public" if (k == "new_build_generation" and case != "private") else v)
                            for k, v in DEFAULT_SETTINGS.items()}})
    return pd.DataFrame(rows)


def test_registers_compare_through_register_io(tmp_path):
    ref, cand = _roots(tmp_path)
    rel = "public_ownership/g_asset_finance_register.parquet"
    (ref / "public_ownership").mkdir()
    (cand / "public_ownership").mkdir()
    write_register(_register(), ref / rel)
    # The same rows written in another order and in two blocks (another row-group layout).
    with __import__("scripts.public_ownership.register_io", fromlist=["RegisterWriter"]).RegisterWriter(
            cand / rel, row_group_rows=2) as writer:
        register = _register()
        writer.write(register.loc[register["case_id"].eq("private")].iloc[::-1])
        writer.write(register.loc[register["case_id"].eq("public_ownership")])
    diff = compare_file(rel, ref / rel, cand / rel, Options(register_batch_cases=1))
    assert diff.kind == "register" and diff.status == "equal", diff.examples
    assert diff.rows_reference == diff.rows_candidate == 6
    write_register(_register(value=1.5), cand / rel)
    diff = compare_file(rel, ref / rel, cand / rel, Options())
    assert diff.status == "differs" and diff.values_differ == 3 and diff.worst == "annual_required_revenue_gbp"
    write_register(_register(cases=("private", "public_control")), cand / rel)
    diff = compare_file(rel, ref / rel, cand / rel, Options())
    assert diff.status == "differs" and diff.rows_only_reference == 1 and diff.rows_only_candidate == 1


def test_report_files(tmp_path):
    ref, cand = _roots(tmp_path)
    _write(ref, "a.csv", SUMMARY)
    _write(cand, "a.csv", SUMMARY.replace("0.1", "0.2"))
    report, details = tmp_path / "diff.csv", tmp_path / "diff.json"
    assert main([str(ref), str(cand), "--report", str(report), "--json", str(details), "--quiet"]) == 1
    table = pd.read_csv(report)
    assert table.loc[0, "status"] == "differs" and np.isclose(table.loc[0, "max_abs_diff"], 0.1)
    assert json.loads(details.read_text())[0]["path"] == "a.csv"
