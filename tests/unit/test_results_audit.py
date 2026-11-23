"""The results audit catches each family of bug found on 11-15 Nov 2026 (user, 15 Nov 2026)."""

import os
import time

import pandas as pd

from scripts.public_ownership import results_audit as ra


def _summary(rows):
    header = "output_file\tdate\trule\tlog-file(s)\tinput-file(s)\tshellcmd\tstatus\tplan"
    return "Building DAG of jobs...\n" + "\n".join([header] + ["\t".join(r) for r in rows]) + "\n"


def test_a_stale_output_snakemake_will_not_rebuild_is_caught_unless_declared(tmp_path):
    (tmp_path / "resources" / "public_ownership").mkdir(parents=True)
    out = tmp_path / "resources/public_ownership/g_asset_inputs.csv"
    shares = tmp_path / "resources/public_ownership/g_generator_state_shares.csv"
    month = tmp_path / "resources/market_month.nc"
    for path in (out, shares):
        path.write_text("x")
    old = time.time() - 3600
    os.utime(out, (old, old))
    os.utime(shares, (old, old))
    month.write_text("y")                                   # re-solved after the register was built
    tsv = tmp_path / "summary.tsv"
    tsv.write_text(_summary([
        ["resources/public_ownership/g_asset_inputs.csv", "d", "build_register", "", "resources/market_month.nc", "-",
         "ok", "no update"],
        ["resources/public_ownership/g_generator_state_shares.csv", "d", "shares", "", "resources/market_month.nc", "-",
         "ok", "no update"],
        ["resources/other.csv", "d", "r", "", "resources/market_month.nc", "-", "missing", "update pending"],
    ]), encoding="utf-8")
    stale = ra.stale_outputs(ra.read_detailed_summary(tsv), tmp_path).set_index("output_file")
    assert stale.at["resources/public_ownership/g_asset_inputs.csv", "accepted"] == ""        # unexplained: fails
    assert stale.at["resources/public_ownership/g_generator_state_shares.csv", "accepted"] != ""  # by design
    assert "resources/other.csv" not in stale.index                                           # Snakemake reruns it


def _register_and_valuation(ccs_price):
    register = pd.DataFrame([
        {"asset_id": "farm", "carrier": "wind_offshore", "support_type": "CfD", "acquisition_price_gbp": 1.0e9,
         "acquisition_basis": "market_value", "state_owned": True},
        {"asset_id": "NZT", "carrier": "CCS Gas", "support_type": "DPA", "acquisition_price_gbp": ccs_price,
         "acquisition_basis": "thermal_market_value", "state_owned": True},
        {"asset_id": "Coire Glas", "carrier": "LAES", "support_type": None, "acquisition_price_gbp": float("nan"),
         "acquisition_basis": None, "state_owned": False},
    ]).assign(case_id=ra.S3, finance_vintage="legacy")
    valuation = pd.DataFrame({"asset_id": ["farm", "NZT", "Coire Glas"], "pv_market_gbp": [1.0e9, 3.16e9, 1.0e9],
                              "scope_group": ["valued", "thermal", "excluded"]})
    return register, valuation


def test_a_purchase_priced_off_its_owners_value_fails_and_one_on_it_passes():
    register, valuation = _register_and_valuation(0.56e9)      # the CCS plants without their contract income
    report = ra.purchase_agreement(register, valuation)
    assert not report["ok"] and [r["asset_id"] for r in report["rows_off_over_fail_share"]] == ["NZT"]
    register, valuation = _register_and_valuation(3.16e9)
    assert ra.purchase_agreement(register, valuation)["ok"]


def test_a_plant_outside_the_purchase_scope_must_be_unowned_and_unpriced():
    register, valuation = _register_and_valuation(3.16e9)
    register.loc[register["asset_id"].eq("Coire Glas"), "state_owned"] = True
    report = ra.purchase_agreement(register, valuation)
    assert not report["ok"] and report["excluded_but_owned"] == ["Coire Glas"]


def test_a_fallback_strike_in_any_month_fails(tmp_path):
    clean, dirty = tmp_path / "a_contract_value_bids.csv", tmp_path / "b_contract_value_bids.csv"
    pd.DataFrame({"generator": ["g1", "g2"], "is_fallback_strike": [False, False]}).to_csv(clean, index=False)
    pd.DataFrame({"generator": ["g1", "g2"], "is_fallback_strike": [False, True]}).to_csv(dirty, index=False)
    assert ra.fallback_strikes([clean])["ok"]
    report = ra.fallback_strikes([clean, dirty])
    assert not report["ok"] and report["months_with_fallbacks"] == {dirty.name: ["g2"]}


def test_a_case_whose_levers_differ_between_years_fails_unless_the_lever_is_year_specific():
    a = pd.DataFrame({"case_id": ["s3"], "acquisition_transaction_costs": ["two_pct"], "capacity_procurement": ["auction"]})
    b = pd.DataFrame({"case_id": ["s3"], "acquisition_transaction_costs": ["none"], "capacity_procurement": ["x"]})
    report = ra.lever_parity({2030: a, 2040: b})
    assert not report["ok"] and [d["lever"] for d in report["differences"]] == ["acquisition_transaction_costs"]
    b["acquisition_transaction_costs"] = "two_pct"
    assert ra.lever_parity({2030: a, 2040: b})["ok"]               # capacity_procurement is declared 2030-only


def _write_meta(root, output, **meta):
    import base64
    import json

    folder = root / ".snakemake" / "metadata"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / base64.b64encode(output.encode()).decode().replace("/", "@")).write_text(json.dumps(meta))


def test_an_input_copied_in_with_its_old_time_but_new_content_is_caught(tmp_path):
    """15 Nov 2026: the fixes copied in with old file times left registers built with the old code looking current."""
    import hashlib

    (tmp_path / "data").mkdir()
    table = tmp_path / "data/crosswalk.csv"
    table.write_text("old")
    _write_meta(tmp_path, "resources/register.csv", shellcmd="python x.py",
                input_checksums={"data/crosswalk.csv": hashlib.sha256(b"old").hexdigest()})
    summary = pd.DataFrame({"output_file": ["resources/register.csv"]})
    assert ra.content_changes(summary, tmp_path).empty
    stamp = table.stat().st_mtime
    table.write_text("new")
    os.utime(table, (stamp, stamp))                               # the old file time, new content
    assert ra.content_changes(summary, tmp_path)["changed_input"].tolist() == ["data/crosswalk.csv"]


def test_code_changed_since_an_output_was_built_is_caught_through_the_imports(tmp_path):
    (tmp_path / "scripts" / "pkg").mkdir(parents=True)
    (tmp_path / "scripts" / "pkg" / "__init__.py").write_text("")
    (tmp_path / "scripts" / "pkg" / "helper.py").write_text("RATE = 1\n")
    (tmp_path / "scripts" / "pkg" / "build.py").write_text("from scripts.pkg import helper\n")
    _write_meta(tmp_path, "resources/out.csv", shellcmd="python scripts/pkg/build.py --out resources/out.csv")
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "out.csv").write_text("x")
    summary = pd.DataFrame({"output_file": ["resources/out.csv"]})
    from scripts.utilities import code_fingerprint as cf
    cf._imports.cache_clear()
    ledger = ra.record_ledger(summary, {}, built_since=None, root=tmp_path, baseline=True)
    assert ra.code_changes(summary, ledger, tmp_path).empty
    (tmp_path / "scripts" / "pkg" / "helper.py").write_text("RATE = 2\n")          # an imported module changes
    cf._imports.cache_clear()
    changed = ra.code_changes(summary, ledger, tmp_path)
    assert changed["status"].tolist() == ["changed"] and "scripts/pkg/helper.py" in changed.at[0, "changed_files"]


def test_an_output_the_plan_promised_but_the_run_did_not_rebuild_is_caught():
    summary = pd.DataFrame({"output_file": ["a", "b"], "rule": ["r", "r"], "status": ["ok", "ok"],
                            "plan": ["no update", "update pending"]})
    assert ra.pending_after_run(summary)["output_file"].tolist() == ["b"]
