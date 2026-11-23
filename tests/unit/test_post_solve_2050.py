"""The decision-49 post-solve pipeline: the dry-run parser, the 2030 guard, the planner and the steps.

No Snakemake, no solve: every test works on synthetic dry-run text and temporary files.
"""

import os
import time

import pandas as pd
import pytest

from scripts.public_ownership import post_solve_2050 as ps

R = "resources/public_ownership"

DRY_RUN = f"""Building DAG of jobs...
Job stats:
job                                               count
----------------------------------------------  -------
build_public_ownership_legacy_valuation               2
total                                                 4

[Sat Oct  3 05:02:51 2026]
rule build_public_ownership_legacy_valuation:
    input: {R}/fes_ht_2050_asset_finance_register.parquet, {R}/model_v11_2_assumptions.yaml
    output: {R}/fes_ht_2050_legacy_valuation.csv, {R}/fes_ht_2050_legacy_cash_flow_profile.csv
    jobid: 95
    reason: Updated input files: {R}/fes_ht_2050_asset_finance_register.parquet
    wildcards: run_group=fes_ht_2050
    resources: tmpdir=<TBD>
INFO:snakemake.logging: Rule: build_public_ownership_legacy_valuation, Jobid: 95
Shell command: python scripts/public_ownership/legacy_valuation.py {R}/fes_ht_2050_asset_finance_register.parquet {R}/fes_ht_2050_legacy_valuation.csv
INFO:snakemake.logging:Shell command: python scripts/public_ownership/legacy_valuation.py {R}/fes_ht_2050_asset_finance_register.parquet {R}/fes_ht_2050_legacy_valuation.csv

[Sat Oct  3 05:02:51 2026]
rule build_public_ownership_legacy_valuation:
    input: {R}/cp30_2030_cp30_network_asset_finance_register.parquet
    output: {R}/cp30_2030_cp30_network_legacy_valuation.csv
    jobid: 118
    reason: Updated input files: config/defaults/base/public_ownership.yaml
    wildcards: run_group=cp30_2030_cp30_network
INFO:snakemake.logging: Rule: build_public_ownership_legacy_valuation, Jobid: 118
Shell command: python scripts/public_ownership/legacy_valuation.py {R}/cp30_2030_cp30_network_asset_finance_register.parquet {R}/cp30_2030_cp30_network_legacy_valuation.csv
INFO:snakemake.logging:Shell command: python scripts/public_ownership/legacy_valuation.py {R}/cp30_2030_cp30_network_asset_finance_register.parquet {R}/cp30_2030_cp30_network_legacy_valuation.csv

[Sat Oct  3 05:02:51 2026]
rule build_public_ownership_exchequer_ledger:
    input: {R}/fes_ht_2050_carried_acquisition.json, {R}/fes_ht_2050_legacy_valuation.csv
    output: {R}/fes_ht_2050_exchequer_ledger.json
    jobid: 0
    reason: Input files updated by another job
    wildcards: run_group=fes_ht_2050
INFO:snakemake.logging: Rule: build_public_ownership_exchequer_ledger, Jobid: 0
Shell command: python scripts/public_ownership/exchequer_ledger.py {R}/fes_ht_2050_legacy_valuation.csv {R}/fes_ht_2050_exchequer_ledger.json
INFO:snakemake.logging:Shell command: python scripts/public_ownership/exchequer_ledger.py {R}/fes_ht_2050_legacy_valuation.csv {R}/fes_ht_2050_exchequer_ledger.json

[Sat Oct  3 05:02:51 2026]
rule build_public_ownership_carried_acquisition:
    input: {R}/cp30_2030_cp30_network_legacy_valuation.csv, {R}/cp30_2030_cp30_network_nationalised_bill_summary.json
    output: {R}/fes_ht_2050_carried_acquisition.json
    jobid: 96
    reason: Input files updated by another job: {R}/cp30_2030_cp30_network_legacy_valuation.csv
    wildcards: run_group=fes_ht_2050
INFO:snakemake.logging: Rule: build_public_ownership_carried_acquisition, Jobid: 96
Shell command: python scripts/public_ownership/acquisition_carry_forward.py {R}/cp30_2030_cp30_network_nationalised_bill_summary.json {R}/cp30_2030_cp30_network_legacy_valuation.csv {R}/fes_ht_2050_carried_acquisition.json
INFO:snakemake.logging:Shell command: python scripts/public_ownership/acquisition_carry_forward.py {R}/cp30_2030_cp30_network_nationalised_bill_summary.json {R}/cp30_2030_cp30_network_legacy_valuation.csv {R}/fes_ht_2050_carried_acquisition.json
Reasons:
    (check individual jobs above for details)
This was a dry-run (flag -n). The order of jobs does not reflect the order of execution.
"""


def job(rule, outputs, inputs=(), shell="python x.py", wildcards=""):
    return ps.Job(rule=rule, outputs=list(outputs), inputs=list(inputs), shell=shell, wildcards=wildcards)


# ── the parser ───────────────────────────────────────────────────────────────

def test_parse_dry_run_reads_every_block_and_its_command():
    jobs = ps.parse_dry_run(DRY_RUN)
    assert [j.jobid for j in jobs] == ["95", "118", "0", "96"]
    assert jobs[0].outputs == [f"{R}/fes_ht_2050_legacy_valuation.csv", f"{R}/fes_ht_2050_legacy_cash_flow_profile.csv"]
    assert jobs[0].inputs[1] == f"{R}/model_v11_2_assumptions.yaml"
    assert jobs[0].wildcards == "run_group=fes_ht_2050"
    assert jobs[3].shell.startswith("python scripts/public_ownership/acquisition_carry_forward.py")
    assert all("INFO" not in j.shell and "\n" not in j.shell for j in jobs)


def test_parse_dry_run_script_rule_has_no_command_and_nothing_to_do_is_empty():
    text = ("rule analyze_market_results:\n    output: resources/analysis/FES25_HT_2050_Jan_market_market_summary.json\n"
            "    jobid: 7\nINFO:snakemake.logging: Rule: analyze_market_results, Jobid: 7\nShell command: None\n"
            "INFO:snakemake.logging:Shell command: None\n")
    (j,) = ps.parse_dry_run(text)
    assert j.shell == ""
    assert ps.parse_dry_run("Building DAG of jobs...\nNothing to be done (all requested files are present).\n") == []


# ── the guard ────────────────────────────────────────────────────────────────

def test_classify_job():
    assert ps.classify_job(job("x", [f"{R}/cp30_2030_ledger.json"])) == "2030"
    assert ps.classify_job(job("x", ["resources/market/CP30_FFR_2030_year_fti.csv"])) == "2030"
    assert ps.classify_job(job("x", [f"{R}/fes_ht_2050_ledger.json", "resources/market/FES25_HT_2050_Jan_market.csv"])) == "2050"
    assert ps.classify_job(job("x", [f"{R}/pathway_2030_2050_summary.json"])) == "2050"
    assert ps.classify_job(job("x", ["resources/FES/FES_2025_data.csv"])) == "foreign"
    assert ps.classify_job(job("fes_long_range_validation", [])) == "2050"
    # one 2030 output makes the job a 2030 job, whatever else it writes
    assert ps.classify_job(job("x", [f"{R}/fes_ht_2050_a.csv", f"{R}/cp30_2030_a.csv"])) == "2030"


def test_snakemake_policy_refuses_any_job_writing_a_2030_path():
    jobs = ps.parse_dry_run(DRY_RUN)
    with pytest.raises(ps.GuardError, match="writes a 2030 path"):
        ps.plan_execution(jobs, policy="snakemake")


def test_auto_runs_snakemake_only_when_the_plan_is_all_2050():
    jobs = [j for j in ps.parse_dry_run(DRY_RUN) if ps.classify_job(j) == "2050"]
    plan = ps.plan_execution(jobs, policy="auto")
    assert plan.mode == "snakemake" and len(plan.run) == 3 and not plan.skipped_2030


def test_auto_falls_back_to_the_2050_commands_in_dependency_order():
    plan = ps.plan_execution(ps.parse_dry_run(DRY_RUN), policy="auto")
    assert plan.mode == "commands"
    assert [j.jobid for j in plan.skipped_2030] == ["118"]
    order = [j.rule for j in plan.run]
    # the ledger reads the carried acquisition and the legacy valuation: it runs last
    assert order[-1] == "build_public_ownership_exchequer_ledger"
    assert set(order) == {"build_public_ownership_legacy_valuation", "build_public_ownership_carried_acquisition",
                          "build_public_ownership_exchequer_ledger"}
    assert not any(ps.PATH_2030.search(o) for j in plan.run for o in j.outputs)
    assert plan.reads_2030 == [f"{R}/cp30_2030_cp30_network_legacy_valuation.csv", f"{R}/cp30_2030_cp30_network_nationalised_bill_summary.json"]
    assert plan.counts() == {"build_public_ownership_carried_acquisition": 1,
                             "build_public_ownership_exchequer_ledger": 1,
                             "build_public_ownership_legacy_valuation": 1}


def test_guard_refuses_forbidden_rules_foreign_paths_and_script_rules_in_command_mode():
    with pytest.raises(ps.GuardError, match="never solves"):
        ps.plan_execution([job("solve_balancing_mechanism", ["resources/market/FES25_HT_2050_Jan_market_balancing.nc"])])
    with pytest.raises(ps.GuardError, match="never solves"):
        ps.plan_execution([job("build_public_ownership_deployment_share_tables",
                               [f"{R}/fes_ht_2050_generator_state_shares_deployment_share_25.csv"])])
    with pytest.raises(ps.GuardError, match="neither HT50 nor 2050"):
        ps.plan_execution([job("build_x", ["resources/FES/FES_2025_data.csv"])])
    mixed = [job("a", [f"{R}/cp30_2030_a.csv"]), job("analyze_market_results",
                                                    ["resources/analysis/FES25_HT_2050_Jan_market_x.json"], shell="")]
    with pytest.raises(ps.GuardError, match="no printed shell"):
        ps.plan_execution(mixed, policy="auto")


def test_guard_refuses_a_command_naming_an_undeclared_2030_path():
    jobs = [job("a", [f"{R}/cp30_2030_a.csv"]),
            job("b", [f"{R}/fes_ht_2050_b.csv"], inputs=[f"{R}/fes_ht_2050_in.csv"],
                shell=f"python b.py {R}/fes_ht_2050_in.csv --also-write {R}/cp30_2030_b.csv")]
    with pytest.raises(ps.GuardError, match="not one of its declared inputs"):
        ps.plan_execution(jobs, policy="auto")


def test_done_outputs_leave_the_previous_calls_jobs_out_of_the_plan():
    # a dry run of the carried call drops the chain's jobs, as a real run (after the chain) would
    jobs = ps.parse_dry_run(DRY_RUN)
    done = {f"{R}/fes_ht_2050_legacy_valuation.csv", f"{R}/fes_ht_2050_legacy_cash_flow_profile.csv"}
    plan = ps.plan_execution(jobs, "auto", done)
    assert [j.rule for j in plan.run] == ["build_public_ownership_carried_acquisition",
                                          "build_public_ownership_exchequer_ledger"]
    # a script: rule the chain will run no longer blocks the command plan
    script = job("extract_future_balancing_costs", ["resources/analysis/FES25_HT_2050_x.csv"], shell="")
    assert ps.plan_execution(jobs + [script], "auto", done | {"resources/analysis/FES25_HT_2050_x.csv"}).mode == "commands"


def test_empty_plan_is_nothing_and_topo_order_detects_cycles():
    assert ps.plan_execution([]).mode == "nothing"
    a = job("a", ["HT50_a"], inputs=["HT50_b"])
    b = job("b", ["HT50_b"], inputs=["HT50_a"])
    with pytest.raises(ps.StepError, match="cycle"):
        ps.topo_order([a, b])


# ── the step planner ─────────────────────────────────────────────────────────

def test_select_steps():
    assert ps.select_steps() == list(ps.STEP_NAMES)
    assert ps.select_steps("finance") == ["finance", "strike_diagnostic", "validation", "summary"]
    assert ps.select_steps("3", "4") == ["calibration", "finance"]
    assert ps.select_steps("summary", "summary") == ["summary"]
    with pytest.raises(ValueError, match="unknown step"):
        ps.select_steps("solve")
    with pytest.raises(ValueError, match="comes after"):
        ps.select_steps("summary", "check")


def test_snakemake_command_is_mtime_nolock_and_restricted():
    pipe = ps.Pipeline(ps.Options(root=ps.PROJECT_ROOT, out_dir=ps.PROJECT_ROOT / "unused", dry_run=True))
    cmd = pipe.snakemake_cmd(["t1"], ps.FINANCE_RULES, ["FES25_HT_2050_Jan_market"], dry=True)
    assert cmd[cmd.index("--rerun-triggers") + 1] == "mtime" and "--nolock" in cmd and "-n" in cmd
    assert cmd[cmd.index("--allowed-rules") + 1:] == list(ps.FINANCE_RULES)
    assert cmd[cmd.index("--config") + 1] == 'run_scenarios=["FES25_HT_2050_Jan_market"]'
    assert not any(ps.FORBIDDEN_RULE.match(r) for r in ps.FINANCE_RULES + ps.VALIDATION_RULES)


def test_run_ids_and_requirements_cover_the_decision_49_set():
    assert len(ps.finance_run_ids()) == 72 and len(ps.all_run_ids()) == 87
    req = ps.required_solve_outputs()
    paths = {r.path for r in req}
    assert sum(p.endswith("_balancing.nc") for p in paths) == 12 * 3 + 48 + 3
    assert "resources/market/FES25_HT_2050_Jan_p2_bids_25pct_balancing.nc" in paths
    p2 = next(r for r in req if r.path.endswith("Jan_p2_bids_25pct_balancing.nc"))
    assert p2.after == "resources/market/FES25_HT_2050_Jan_market_wholesale.nc"  # the BM read the market month's Stage 1


def test_check_solve_outputs_and_scan_logs(tmp_path):
    net = tmp_path / "net.nc"
    net.write_text("n")
    old = time.time() - 100
    os.utime(net, (old, old))
    s1, bm, stale = tmp_path / "s1.nc", tmp_path / "bm.nc", tmp_path / "stale.nc"
    for p in (s1, bm, stale):
        p.write_text("x")
    os.utime(stale, (old - 10, old - 10))
    os.utime(bm, (old + 10, old + 10))  # newer than the network, older than its Stage 1
    reqs = [ps.Requirement("s1.nc", "s1"), ps.Requirement("bm.nc", "bm", "s1.nc"),
            ps.Requirement("stale.nc", "x"), ps.Requirement("missing.nc", "x")]
    res = ps.check_solve_outputs(tmp_path, "net.nc", reqs)
    assert res["missing"] == ["missing.nc"] and res["stale"] == ["stale.nc"]
    assert res["out_of_order"] == ["bm.nc is older than s1.nc"]
    log, old_log = tmp_path / "a.log", tmp_path / "b.log"
    log.write_text("ok\nError in rule solve_balancing_mechanism:\nOSError: [Errno 28] No space left on device\n")
    old_log.write_text("Error in rule x:\n")
    os.utime(old_log, (old - 50, old - 50))
    hits = ps.scan_logs([log, old_log, tmp_path / "none.log"], since=old)
    assert [(h["line"], h["pattern"]) for h in hits] == [(2, "Error in rule"), (3, "Errno 28"),
                                                          (3, "No space left on device")]


# ── calibration drift and the summary ────────────────────────────────────────

def test_compare_calibrations_reports_drift_by_scheme():
    old = pd.DataFrame({"key": ["PAR7_2040_onshore", "DPA_2040_CCS Gas_x", "PAR7_2031_solar"],
                        "scheme": ["CfD", "DPA", "CfD"], "calibrated_value": [80.0, 250.0, 58.7],
                        "unit": ["GBP/MWh 2024", "GBP/kW/yr", "GBP/MWh 2024"],
                        "value_in_use": [80.0, 250.0, 58.7]})
    new = pd.DataFrame({"key": ["PAR7_2040_onshore", "DPA_2040_CCS Gas_x", "PAR7_2050_offshore"],
                        "scheme": ["CfD", "DPA", "CfD"], "calibrated_value": [81.5, 248.0, 90.1],
                        "unit": ["GBP/MWh 2024", "GBP/kW/yr", "GBP/MWh 2024"],
                        "value_in_use": [80.0, 250.0, 90.0]})
    summary, table = ps.compare_calibrations(old, new)
    assert summary["cohorts"] == 2 and summary["added"] == ["PAR7_2050_offshore"]
    assert summary["dropped"] == ["PAR7_2031_solar"]
    assert summary["by_scheme"]["CfD"]["max_abs_drift"] == pytest.approx(1.5)
    assert summary["by_scheme"]["DPA"]["max_abs_drift"] == pytest.approx(2.0)
    assert summary["strike_table_vs_bm_basis_max_gap"] == pytest.approx(0.0)
    assert len(table) == 4


def test_ledger_cases_and_identity_residual():
    ledger = {"headline_cases": {"public_ownership": "P2"},
              "cases": {"public_ownership": {"bill_inc_vat": 955.29, "consumer_change_vs_private_gbp": 13e9,
                                             "exchequer_net_vs_private_gbp": -7e9,
                                             "investor_change_vs_private_gbp": 3e9,
                                             "resource_saving_vs_private_gbp": 9e9}}}
    (row,) = ps.ledger_cases(ledger)
    assert row["label"] == "P2" and row["consumers_bn"] == pytest.approx(13.0)
    assert row["identity_residual_gbp"] == pytest.approx(0.0)
    md = ps.summary_markdown({"written": "now", "bills_and_parties": [row], "p2_boundary_private_rent": None,
                              "pathway": {"rows": [], "anchoring_all_pass": True, "anchoring_worst_difference_gbp": 0},
                              "bm_by_month": [{"month": "Year"}], "calibration_drift": None,
                              "validation": {"WARN": 18, "FAIL": 0, "unexplained_warns": 0, "status_counts": {}}})
    assert "| public_ownership | P2 | 955.29 |" in md and "WARN 18, FAIL 0" in md
