"""Post-solve pipeline for the consolidated 2050 re-solve (FES 2050 plan, decision 49): one checked launch.

Everything that follows the decision-49 solves (``scratchpad/year/run_decision49.sh``, phases A to C),
in order, each step logged and stopping on the first error with the step to resume from:

1. ``check``: every solve output exists and is newer than the rebuilt network
   (``resources/network/FES25_HT_2050.nc``): the two annual passes, the market, central and NT+ months
   (both stages), the 48 P2 bid months and the three gate weeks; each BM file is newer than the
   Stage 1 file it read. No run log written since the rebuild shows ``Error in rule`` or Errno 28,
   and no other Snakemake is running.
2. ``recovery``: refresh the 2050 recovery-demand volumes from the twelve market months
   (``scripts.consumer_bill.refresh_recovery_demand``).
3. ``calibration``: bring the register up to date, snapshot the capture prices and strike
   calibration the BM bid on, re-derive both from the register (``vintage_capital_2050 capture`` and
   ``calibrate``) and report the drift. Decision 49 (b) is **one pass**: the drift is reported, never
   iterated. ``--calibration install`` (default) writes the re-derived files to ``data/`` for the
   finance chain (the D42 practice); ``--calibration report`` leaves them in the output directory.
   Decision 71: ``calibrate`` derives at the study's auction margin (``vintage_capital_2050.
   auction_margin_share``, 0.05 for HT50), the step refuses a re-derived file at another margin, and the
   two bound variants (margin none and high) are re-margined from the installed file (``margin``).
4. ``finance``: the 2050 finance chain, the P2 bid-stack bill groups and their merge (Snakemake), then
   the carried acquisition, tariff, institution comparison, Exchequer ledger and the 2030-2050
   pathway. Snakemake always runs with ``--nolock --rerun-triggers mtime``, explicit 2050 targets and
   ``--allowed-rules`` restricted to the 2050 public-ownership and bill rules. Every call is planned
   first (``-n -p``) and guarded:

   * a plan with no 2030 job runs through Snakemake;
   * a plan that pulls 2030 jobs (the carried acquisition, the ledger and the pathway read cp30_2030_cp30_network
     files, and the rule names are shared with 2030) runs **only its 2050 jobs, from Snakemake's printed
     commands**, in dependency order, reading the 2030 files as they stand (listed in the summary);
   * the guard refuses any job it would run that writes a ``cp30_2030``/``CP30`` path, any job naming
     neither HT50 nor 2050, any solve, network or label rule, and any 2050 job without a shell command.

   The step ends with ``calibrate --check`` on the new register: the residual drift, report only.
5. ``strike_diagnostic`` (decision 70, user, 4 Oct 2026): the post-AR7 break-even strikes re-derived
   on the central-dispatch year beside the market year's (``vintage_capital_2050 calibrate
   --dispatch-basis central``, :mod:`central_strike_diagnostic_2050`), against the calibration the
   chain used and with the chain's bill summary for the per-household figures. Only uncompensated
   curtailment counts on either basis. A diagnostic: no file the chain reads is written.
6. ``validation``: the ``fes_long_range_validation`` target (the same plan and guard) and the NT+ year
   comparison (``scripts/market/compare_tyndp26_year_2050.py``).
7. ``summary``: one JSON and one markdown table: bills by case, the three-party columns, the
   P2-boundary private rent, the pathway totals and NPVs, the BM shed and cost by month (market and
   central), the calibration drift, the decision-70 strike diagnostic and the validation WARN/FAIL counts.

``--dry-run`` runs the read-only checks and the Snakemake plans (``-n -p``), prints every command and
planned job, and changes nothing. Usage, from the repository root::

    python scripts/public_ownership/post_solve_2050.py [--dry-run] [--from-step STEP] [--to-step STEP]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import read_config

LOGGER = logging.getLogger("post_solve_2050")

# ── the run ──────────────────────────────────────────────────────────────────

#: The pypsa-gb environment the steps run in: ``PYPSA_GB_ENV`` if set, else the active conda
#: environment, else the interpreter running this script.
ENV_DIR = Path(os.environ.get("PYPSA_GB_ENV") or os.environ.get("CONDA_PREFIX") or sys.prefix)
_ENV_PYTHON = ENV_DIR / ("python.exe" if os.name == "nt" else "bin/python")
PYTHON = str(_ENV_PYTHON) if _ENV_PYTHON.exists() else sys.executable
ENV_PATH_DIRS = ([ENV_DIR, ENV_DIR / "Scripts", ENV_DIR / "Library" / "bin"] if os.name == "nt"
                 else [ENV_DIR / "bin"])
BASH_CANDIDATES = [r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"]

STUDY = "FES25_HT_2050"
GROUP = "fes_ht_2050"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
P2_SHARES = (25, 50, 75, 100)
GATES = ("HT50_1week_gate_d40", "HT50_1week_gate_d40_ga", "HT50_1week_gate_d40_tyndp26")
MARKET_SUFFIX = "_market"
CENTRAL_SUFFIX = "_central_dispatch"
NTP_SUFFIX = "_tyndp2026_prices"

NETWORK = f"resources/network/{STUDY}.nc"
MARKET = "resources/market"
PO = "resources/public_ownership"
RECOVERY = "data/market/consumer_bill_forecast/2050_recovery_demand_assumptions.json"
ASSUMPTIONS = "data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json"
CAPTURE = "data/public_ownership/financing/capture_prices_2050.csv"
CALIBRATION = "data/public_ownership/financing/strike_calibration_2050.csv"
#: Decision 71: the auction-margin bounds' calibrations, re-margined from the installed central file (the 2030
#: lever's values: none 0, high 10%; the central 5% is CALIBRATION itself).
MARGIN_VARIANTS = {"none": "data/public_ownership/financing/strike_calibration_2050_auction_margin_none.csv",
                   "high": "data/public_ownership/financing/strike_calibration_2050_auction_margin_high.csv"}
#: Decision 54: the NESO wind ramp's own calibration, derived on private@private_rate_neso_wind_ramp.
RATE_PATH = "neso_wind_ramp"
RAMP_CALIBRATION = "data/public_ownership/financing/strike_calibration_2050_neso_wind_ramp.csv"
SUPPORT_LABELS = f"{MARKET}/{STUDY}_support_labels.csv"
REGISTER = f"{PO}/{GROUP}_asset_finance_register.parquet"
BOOK = f"{PO}/{GROUP}_support_contract_book.csv"
FINANCE_CONFIG = f"{PO}/{STUDY}_finance_config.yaml"
STRIKE_TABLE = f"{PO}/{STUDY}_cfd_strike_table.csv"
LEDGER = f"{PO}/{GROUP}_exchequer_ledger.json"
PATHWAY = f"{PO}/pathway_2030_2050_summary.json"
VALIDATION_JSON = f"resources/analysis/{STUDY}_fes_energy_validation.json"
#: Decision 70: the strike diagnostic on the central-dispatch year, and the bill summary it converts with.
STRIKE_DIAGNOSTIC = f"{PO}/{GROUP}_central_dispatch_strike_diagnostic.csv"
STRIKE_DIAGNOSTIC_JSON = f"{PO}/{GROUP}_central_dispatch_strike_diagnostic.json"
BILL_SUMMARY = f"resources/analysis/{STUDY}_consumer_bill_future_summary.json"
DEFAULT_OUT_DIR = f"resources/analysis/{STUDY}_post_solve"

REGISTER_TARGETS = (REGISTER, BOOK, FINANCE_CONFIG, STRIKE_TABLE)
CHAIN_TARGETS = (
    f"{PO}/{GROUP}_consumer_bill_by_ownership_case_summary.json",
    f"{PO}/{GROUP}_public_account_summary.json",
    f"{PO}/{GROUP}_consumer_bill_balancing_cases_summary.json",
    f"resources/analysis/{STUDY}_consumer_bill_future_summary.json",
)
CARRIED_TARGETS = (
    f"{PO}/{GROUP}_carried_acquisition.json",
    f"{PO}/{GROUP}_nationalised_bill_summary.json",
    f"{PO}/{GROUP}_nationalised_bill_by_institution.json",
    LEDGER,
    PATHWAY,
)
VALIDATION_TARGETS = ("fes_long_range_validation",)
#: Decision 61 (finding 9): the register bounds re-accounted on the decision-49 months (no solve): the
#: learning-rate and retirement-rule bounds and decision 62's repowering-share sensitivity.
#: The register bounds that re-account P3 (their case list holds the acquisition tariff).
TARIFF_BOUND_GROUPS = ("fes_ht_2050_learning_bound", "fes_ht_2050_retirement_bound", "fes_ht_2050_repowering_bound")
#: Decision 68's sensitivities (round 4, 3 Oct 2026): bounds that re-account the P2 share cases only, so they
#: have no tariff. They arrive with the HT50 round-3 config patch.
COOWNERSHIP_BOUND_GROUPS = ("fes_ht_2050_coownership_reduction_bound", "fes_ht_2050_coownership_pooled_bound")
#: Decision 71's sensitivities (user, 4 Oct 2026): the auction margin at none and high, each on its own strike
#: table (rule build_public_ownership_auction_margin_inputs_2050) and the central months' BM. Their case list
#: is the other bounds' (P3 included), so they take the tariff targets too. They arrive with the HT50 patch.
AUCTION_MARGIN_BOUND_GROUPS = tuple(f"fes_ht_2050_auction_margin_{level}_bound" for level in MARGIN_VARIANTS)
BOUND_GROUPS = TARIFF_BOUND_GROUPS + COOWNERSHIP_BOUND_GROUPS + AUCTION_MARGIN_BOUND_GROUPS
BOUND_TARGETS = tuple(f"{PO}/{g}_{suffix}" for g in TARIFF_BOUND_GROUPS + AUCTION_MARGIN_BOUND_GROUPS
                      for suffix in ("consumer_bill_by_ownership_case_summary.json",
                                     "nationalised_bill_summary.json", "exchequer_ledger.json")) + tuple(
    f"{PO}/{g}_{suffix}" for g in COOWNERSHIP_BOUND_GROUPS
    for suffix in ("consumer_bill_by_ownership_case_summary.json", "exchequer_ledger.json"))

#: The 2050 public-ownership and bill rules (the union of the 2 to 3 Oct chain, P2-bill and
#: pathway launches). No solve, network, label or P2 share-table rule is ever allowed.
FINANCE_RULES = (
    "aggregate_CP30_months", "aggregate_public_ownership_reference_run", "analyze_market_results",
    "build_capacity_market_year", "build_future_ro_scheme_cost", "build_public_bm_bid_decomposition",
    "build_public_ownership_auction_margin_inputs_2050",
    "build_public_ownership_carried_acquisition", "build_public_ownership_cfd_strike_adjustments",
    "build_public_ownership_cfd_strike_table_2050", "build_public_ownership_commissioning_years",
    "build_public_ownership_contract_book", "build_public_ownership_exchequer_ledger",
    "build_public_ownership_fleet_vintages", "build_public_ownership_institution_comparison",
    "build_public_ownership_legacy_valuation", "build_public_ownership_nationalised_bill",
    "build_public_ownership_network_finance_account", "build_public_ownership_pathway_2030_2050",
    "build_public_ownership_public_account", "build_public_ownership_state_shares",
    "build_public_ownership_vintaged_asset_register", "build_public_ownership_vintaged_asset_settlement",
    "calculate_public_ownership_financing", "extract_future_balancing_costs",
    "merge_public_ownership_balancing_groups", "project_future_consumer_bill",
    "project_public_ownership_consumer_bill", "resolve_public_ownership_study_config",
    "verify_public_ownership_ownership_tables",
)
#: Year-neutral names since stage 5 (rules/fes_long_range.smk; were validate_HT50_dispatch etc.).
VALIDATION_RULES = ("validate_fes_dispatch", "diagnose_fes_ic1", "measure_fes_bm_storage_churn",
                    "fes_long_range_validation")
#: Rules the post-solve pipeline must never run, whatever the allowed list says.
FORBIDDEN_RULE = re.compile(r"^(solve_|build_support_labels|build_public_ownership_deployment_share_tables$|"
                            r"finalize_|add_|cluster_|integrate_|apply_marginal_costs|build_ETYS|retrieve_)")
#: A job writing a path matching this is a 2030 job.
PATH_2030 = re.compile(r"cp30_2030|CP30", re.IGNORECASE)
#: Every job the pipeline runs must write paths naming HT50 or 2050 (or be a 2050 target rule).
PATH_2050 = re.compile(r"HT50|2050")
LOG_ERROR_PATTERNS = ("Error in rule", "Errno 28", "No space left on device")


class StepError(RuntimeError):
    """A step that cannot continue; the message says why and what to do."""


class GuardError(StepError):
    """A planned job the pipeline refuses to run."""


# ── run ids and required outputs ─────────────────────────────────────────────

def market_ids() -> list[str]:
    return [f"{STUDY}_{m}{MARKET_SUFFIX}" for m in MONTHS]


def central_ids() -> list[str]:
    return [f"{STUDY}_{m}{CENTRAL_SUFFIX}" for m in MONTHS]


def ntp_ids() -> list[str]:
    return [f"{STUDY}_{m}{NTP_SUFFIX}" for m in MONTHS]


def p2_ids() -> list[str]:
    return [f"{STUDY}_{m}_p2_bids_{p}pct" for m in MONTHS for p in P2_SHARES]


def finance_run_ids() -> list[str]:
    """The run ids the finance and bill rules need configs for (market, central, P2: 72)."""
    out = []
    for m in MONTHS:
        out += [f"{STUDY}_{m}{MARKET_SUFFIX}", f"{STUDY}_{m}{CENTRAL_SUFFIX}"]
        out += [f"{STUDY}_{m}_p2_bids_{p}pct" for p in P2_SHARES]
    return out


def all_run_ids() -> list[str]:
    """Every decision-49 run id (87): finance ids, the NT+ months and the gate weeks."""
    return finance_run_ids() + ntp_ids() + list(GATES)


@dataclass(frozen=True)
class Requirement:
    """A solve output that must exist, be newer than the network and newer than ``after``."""
    path: str
    kind: str
    after: str | None = None


def required_solve_outputs() -> list[Requirement]:
    """Every decision-49 solve output the post-solve steps read."""
    req = [Requirement(f"{MARKET}/{STUDY}_seasonal_trajectory.csv", "annual pass"),
           Requirement(f"{MARKET}/{STUDY}_seasonal_pass.json", "annual pass"),
           Requirement(f"{MARKET}/{STUDY}{NTP_SUFFIX}_seasonal_trajectory.csv", "annual pass (NT+)"),
           Requirement(f"{MARKET}/{STUDY}{NTP_SUFFIX}_seasonal_pass.json", "annual pass (NT+)")]
    for kind, ids in (("market", market_ids()), ("central", central_ids()), ("NT+", ntp_ids()),
                      ("gate", list(GATES))):
        for sid in ids:
            s1 = f"{MARKET}/{sid}_wholesale.nc"
            req += [Requirement(s1, f"{kind} Stage 1"),
                    Requirement(f"{MARKET}/{sid}_wholesale_price.csv", f"{kind} Stage 1", None),
                    Requirement(f"{MARKET}/{sid}_balancing.nc", f"{kind} BM", s1),
                    Requirement(f"{MARKET}/{sid}_constraint_costs.csv", f"{kind} BM", s1)]
    for sid in p2_ids():
        s1 = f"{MARKET}/{sid.split('_p2_bids')[0]}{MARKET_SUFFIX}_wholesale.nc"  # wholesale_from: the market month
        req += [Requirement(f"{MARKET}/{sid}_balancing.nc", "P2 BM", s1),
                Requirement(f"{MARKET}/{sid}_constraint_costs.csv", "P2 BM", s1)]
    return req


def check_solve_outputs(root: Path, network: str = NETWORK,
                        requirements: Sequence[Requirement] | None = None) -> dict:
    """Missing, stale (older than the network) and out-of-order (BM older than its Stage 1) outputs."""
    net = root / network
    if not net.exists():
        return {"network": network, "network_mtime": None, "missing": [network], "stale": [], "out_of_order": [],
                "checked": 0}
    t_net = net.stat().st_mtime
    missing, stale, order = [], [], []
    reqs = list(requirements if requirements is not None else required_solve_outputs())
    for r in reqs:
        p = root / r.path
        if not p.exists():
            missing.append(r.path)
            continue
        t = p.stat().st_mtime
        if t <= t_net:
            stale.append(r.path)
        if r.after and (root / r.after).exists() and t < (root / r.after).stat().st_mtime:
            order.append(f"{r.path} is older than {r.after}")
    return {"network": network, "network_mtime": _iso(t_net), "missing": missing, "stale": stale,
            "out_of_order": order, "checked": len(reqs)}


def run_logs(root: Path) -> list[Path]:
    """The decision-49 launch logs and every per-run solve log the check scans."""
    paths = sorted((root / "logs").glob("ht50_decision49_*.log"))
    lm = root / "logs" / "market"
    for sid in market_ids() + central_ids() + ntp_ids() + list(GATES):
        paths += [lm / f"solve_wholesale_{sid}.log", lm / f"solve_balancing_{sid}.log"]
    paths += [lm / f"solve_balancing_{sid}.log" for sid in p2_ids()]
    paths += [lm / f"solve_seasonal_storage_pass_{s}.log" for s in (STUDY, f"{STUDY}{NTP_SUFFIX}")]
    return paths


def scan_logs(paths: Iterable[Path], since: float | None,
              patterns: Sequence[str] = LOG_ERROR_PATTERNS) -> list[dict]:
    """Lines matching ``patterns`` in the logs written after ``since`` (all logs if None)."""
    hits = []
    for p in paths:
        p = Path(p)
        if not p.exists() or (since is not None and p.stat().st_mtime <= since):
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                for pat in patterns:
                    if pat in line:
                        hits.append({"log": str(p), "line": i, "pattern": pat, "text": line.strip()[:240]})
    return hits


def other_snakemake_running() -> list[str]:
    """Command lines of running Snakemake processes (Windows; [] where it cannot tell)."""
    if os.name != "nt":
        return []
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'snakemake' } | Select-Object -ExpandProperty CommandLine")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    # Snakemake itself, not the job scripts it launches (.snakemake\scripts\tmp*.py)
    return [line.strip() for line in out.splitlines()
            if "snakemake" in line and ".snakemake\\scripts" not in line and ".snakemake/scripts" not in line]


# ── Snakemake plans ──────────────────────────────────────────────────────────

@dataclass
class Job:
    """One job of a Snakemake dry run (``-n -p``)."""
    rule: str
    jobid: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    wildcards: str = ""
    reason: str = ""
    shell: str = ""
    local: bool = False


def _split_files(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_dry_run(text: str) -> list[Job]:
    """The jobs of a Snakemake 9 ``-n -p`` log (each job's block, and its ``Shell command:`` line).

    A block starts at ``rule NAME:`` (or ``localrule NAME:``) and carries indented ``input:``,
    ``output:``, ``jobid:``, ``wildcards:`` and ``reason:`` lines. The shell command follows on a
    ``Shell command:`` line (Snakemake repeats it behind an ``INFO:snakemake.logging:`` prefix, which
    is ignored) and runs until a blank line, a new block or a logging line.
    """
    jobs: list[Job] = []
    by_id: dict[str, Job] = {}
    cur: Job | None = None
    shell_job: Job | None = None
    in_shell = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        m = re.match(r"^(local)?rule (\S+):$", line.strip()) if not line.startswith("INFO:") else None
        if m and not line.startswith(" "):
            cur = Job(rule=m.group(2), local=bool(m.group(1)))
            jobs.append(cur)
            in_shell = False
            shell_job = None
            continue
        if line.startswith("INFO:snakemake.logging:"):
            in_shell = False
            mm = re.search(r"Rule: (\S+), Jobid: (\d+)", line)
            if mm:
                shell_job = by_id.get(mm.group(2))
            continue
        if line.startswith("Shell command:"):
            target = shell_job or cur
            value = line[len("Shell command:"):].strip()
            if target is not None and not target.shell and value != "None":  # "None": a script: rule
                target.shell = value
                in_shell = True
                shell_job = target
            continue
        if in_shell:
            if not line.strip() or re.match(r"^(local)?rule \S+:$", line) or line.startswith("["):
                in_shell = False
            else:
                shell_job.shell += "\n" + line
                continue
        if cur is not None and line.startswith("    "):
            s = line.strip()
            for key in ("input", "output", "jobid", "wildcards", "reason"):
                if s.startswith(key + ":"):
                    value = s[len(key) + 1:].strip()
                    if key == "input":
                        cur.inputs = _split_files(value)
                    elif key == "output":
                        cur.outputs = _split_files(value)
                    elif key == "jobid":
                        cur.jobid = value
                        by_id[value] = cur
                    else:
                        setattr(cur, key, value)
        elif cur is not None and not line.strip():
            cur = None
    return jobs


def classify_job(job: Job) -> str:
    """``2030`` (writes a cp30_2030/CP30 path), ``2050`` (every output names HT50 or 2050, or a
    target rule with no output whose rule is 2050), or ``foreign``."""
    if any(PATH_2030.search(o) for o in job.outputs):
        return "2030"
    if job.outputs:
        return "2050" if all(PATH_2050.search(o) for o in job.outputs) else "foreign"
    return "2050" if PATH_2050.search(job.rule) or job.rule in VALIDATION_RULES else "foreign"


@dataclass
class ExecutionPlan:
    """What a step runs: ``nothing``, ``snakemake`` (the whole plan) or ``commands`` (2050 jobs only)."""
    mode: str
    jobs: list[Job]
    run: list[Job]
    skipped_2030: list[Job]
    reads_2030: list[str]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for j in self.run:
            out[j.rule] = out.get(j.rule, 0) + 1
        return dict(sorted(out.items()))


def topo_order(jobs: Sequence[Job]) -> list[Job]:
    """``jobs`` in dependency order (a job after every job writing one of its inputs); ties keep order.

    Snakemake's dry-run order is not its execution order, so printed commands are re-ordered here.
    """
    producers: dict[str, int] = {}
    for i, j in enumerate(jobs):
        for o in j.outputs:
            producers[o] = i
    deps = {i: {producers[f] for f in j.inputs if f in producers and producers[f] != i} for i, j in enumerate(jobs)}
    done: list[int] = []
    placed: set[int] = set()
    while len(done) < len(jobs):
        ready = [i for i in range(len(jobs)) if i not in placed and deps[i] <= placed]
        if not ready:
            cycle = [jobs[i].rule for i in range(len(jobs)) if i not in placed]
            raise StepError(f"the planned jobs have a dependency cycle: {cycle}")
        for i in ready:
            done.append(i)
            placed.add(i)
    return [jobs[i] for i in done]


def _check_runnable(job: Job, mode: str) -> None:
    if FORBIDDEN_RULE.match(job.rule):
        raise GuardError(f"refusing: the plan includes rule {job.rule} ({', '.join(job.outputs)[:200]}); "
                         "the post-solve pipeline never solves, builds the network or rebuilds labels or "
                         "P2 share tables")
    cls = classify_job(job)
    if cls == "2030":
        raise GuardError(f"refusing: job {job.rule} writes a 2030 path ({', '.join(job.outputs)[:200]})")
    if cls == "foreign":
        raise GuardError(f"refusing: job {job.rule} writes paths naming neither HT50 nor 2050 "
                         f"({', '.join(job.outputs)[:200]})")
    if mode == "commands":
        if not job.shell:
            raise GuardError(f"refusing: 2050 job {job.rule} ({', '.join(job.outputs)[:160]}) has no printed shell "
                             "command (a script: rule), so it cannot run outside Snakemake while the plan pulls "
                             "2030 jobs; bring its inputs up to date first")
        for token in re.split(r"\s+", job.shell):
            token = token.strip("'\"")
            if PATH_2030.search(token) and token not in job.inputs and "/" in token:
                raise GuardError(f"refusing: the command of {job.rule} names the 2030 path {token}, which is not "
                                 "one of its declared inputs (it could be written)")


def plan_execution(jobs: Sequence[Job], policy: str = "auto",
                   done_outputs: Iterable[str] = ()) -> ExecutionPlan:
    """Decide how a planned step runs, and guard it.

    Args:
        jobs: The dry run's jobs.
        policy: ``auto`` (Snakemake if the plan has no 2030 job, else the 2050 jobs' commands),
            ``snakemake`` (refuse any 2030 job) or ``commands``.
        done_outputs: Outputs an earlier call of the same run is planned to write (a dry run's
            simulation of that call): a job writing only these is dropped from the plan.

    Raises:
        GuardError: a job the pipeline would run writes a 2030 or foreign path, is a forbidden rule,
            or (commands) has no shell command or names an undeclared 2030 path.
    """
    done = set(done_outputs)
    jobs = [j for j in jobs if not (j.outputs and set(j.outputs) <= done)]
    j2030 = [j for j in jobs if classify_job(j) == "2030"]
    if not jobs:
        return ExecutionPlan("nothing", [], [], [], [])
    if policy == "snakemake" or (policy == "auto" and not j2030):
        for j in jobs:
            _check_runnable(j, "snakemake")
        return ExecutionPlan("snakemake", jobs, jobs, [], [])
    if policy not in ("auto", "commands"):
        raise ValueError(f"unknown policy {policy!r}")
    run = topo_order([j for j in jobs if classify_job(j) != "2030"])
    for j in run:
        _check_runnable(j, "commands")
    outs_2030 = {o for j in j2030 for o in j.outputs}
    reads = sorted({f for j in run for f in j.inputs if PATH_2030.search(f)})
    stale_reads = sorted({f for j in run for f in j.inputs if f in outs_2030})
    LOGGER.debug("2030 inputs Snakemake would rebuild: %s", stale_reads)
    return ExecutionPlan("commands", jobs, run, j2030, reads)


# ── steps and selection ──────────────────────────────────────────────────────

STEP_NAMES = ("check", "recovery", "calibration", "finance", "strike_diagnostic", "validation", "summary")


def select_steps(from_step: str | None = None, to_step: str | None = None,
                 names: Sequence[str] = STEP_NAMES) -> list[str]:
    """The steps from ``from_step`` to ``to_step`` inclusive (a name or a 1-based number)."""
    def index(s, default):
        if s is None:
            return default
        s = str(s).strip()
        if s.isdigit() and 1 <= int(s) <= len(names):
            return int(s) - 1
        if s in names:
            return names.index(s)
        raise ValueError(f"unknown step {s!r}; steps are {', '.join(f'{i + 1}={n}' for i, n in enumerate(names))}")
    a, b = index(from_step, 0), index(to_step, len(names) - 1)
    if a > b:
        raise ValueError(f"--from-step {from_step} comes after --to-step {to_step}")
    return list(names[a:b + 1])


# ── calibration drift ────────────────────────────────────────────────────────

def compare_calibrations(old, new) -> tuple[dict, "object"]:
    """Drift of each cohort's calibrated value, re-derived (``new``) against the one the BM bid on (``old``).

    Returns a summary (largest and mean absolute drift by scheme, keys added or dropped, and the
    largest gap between ``new.value_in_use`` and ``old.calibrated_value``, which is zero when the
    strike table in use was built from the old calibration) and the per-cohort table.
    """
    import pandas as pd

    o = old[["key", "scheme", "calibrated_value"]].rename(columns={"calibrated_value": "bm_basis_value"})
    n = new[["key", "scheme", "unit", "calibrated_value", "value_in_use"]].rename(
        columns={"calibrated_value": "rederived_value"})
    t = o.merge(n, on=["key", "scheme"], how="outer", indicator=True)
    t["drift"] = t["rederived_value"] - t["bm_basis_value"]
    both = t[t["_merge"] == "both"]
    summary = {"cohorts": int(len(both)), "added": sorted(t.loc[t["_merge"] == "right_only", "key"].astype(str)),
               "dropped": sorted(t.loc[t["_merge"] == "left_only", "key"].astype(str)), "by_scheme": {}}
    for scheme, g in both.groupby("scheme"):
        a = g["drift"].abs()
        top = g.assign(abs_drift=a).sort_values("abs_drift", ascending=False).head(5)
        summary["by_scheme"][str(scheme)] = {
            "unit": str(g["unit"].iloc[0]), "cohorts": int(len(g)), "max_abs_drift": float(a.max()),
            "mean_abs_drift": float(a.mean()),
            "largest": [{"key": str(r.key), "bm_basis": float(r.bm_basis_value), "rederived": float(r.rederived_value),
                         "drift": float(r.drift)} for r in top.itertuples(index=False)]}
    gap = (both["value_in_use"] - both["bm_basis_value"]).abs()
    summary["strike_table_vs_bm_basis_max_gap"] = float(gap.max()) if len(gap) else 0.0
    return summary, t.drop(columns="_merge").assign(status=t["_merge"].astype(str))


def calibration_margins(config_path: Path, bm_path: Path, new_path: Path) -> dict:
    """Decision 71: the study's configured auction margin, the BM basis's and the re-derived calibration's."""
    import yaml

    from scripts.public_ownership import vintage_capital_2050 as vk

    config = read_config(config_path) if Path(config_path).exists() else {}
    configured = vk.basis_from_config(config or {})[vk.AUCTION_MARGIN_KEY]
    return {"configured": round(float(configured), 9),
            "bm_basis": round(vk.calibration_margin(vk.load_calibration(bm_path)), 9),
            "rederived": round(vk.calibration_margin(vk.load_calibration(new_path)), 9)}


def compare_capture(old, new) -> dict:
    """Capture price by carrier, re-derived against the file in use (GBP/MWh)."""
    o = old.set_index("carrier")["capture_gbp_per_mwh"]
    n = new.set_index("carrier")["capture_gbp_per_mwh"]
    idx = sorted(set(o.index) | set(n.index))
    rows = {c: {"in_use": _num(o.get(c)), "rederived": _num(n.get(c)),
                "drift": (_num(n.get(c)) - _num(o.get(c))) if c in o.index and c in n.index else None} for c in idx}
    drifts = [abs(r["drift"]) for r in rows.values() if r["drift"] is not None]
    return {"max_abs_drift": max(drifts) if drifts else 0.0, "by_carrier": rows}


def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _iso(t: float | None) -> str | None:
    return None if t is None else dt.datetime.fromtimestamp(t).isoformat(timespec="seconds")


# ── the pipeline ─────────────────────────────────────────────────────────────

@dataclass
class Options:
    root: Path
    out_dir: Path
    dry_run: bool = False
    cores: int = 6
    mem_mb: int = 24000
    calibration: str = "install"
    ignore_log_errors: bool = False
    allow_concurrent: bool = False
    skip_compare: bool = False
    network: str = NETWORK


class Pipeline:
    """The six steps; each logs and raises :class:`StepError` with what to do next."""

    def __init__(self, opts: Options):
        self.o = opts
        self.root = opts.root
        self.out = opts.out_dir
        self.logs = self.out / "logs"
        self.state_file = self.out / "state.json"
        self.state = self._load_state()
        self.dry_run_errors: list[str] = []

    # helpers
    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        if self.o.dry_run:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")

    def env(self) -> dict:
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join([str(p) for p in ENV_PATH_DIRS] + [env.get("PATH", "")])
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _log_file(self, name: str) -> Path:
        d = self.logs if not self.o.dry_run else self.out / "dry_run_logs"
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def run_cmd(self, cmd: Sequence[str], log_name: str, *, check: bool = True, mutating: bool = True) -> tuple[int, str]:
        """Run ``cmd`` (no shell) from the repository root, its output to ``logs/<log_name>``."""
        shown = " ".join(_quote(c) for c in cmd)
        if self.o.dry_run and mutating:
            LOGGER.info("  [dry-run] would run: %s", shown)
            return 0, ""
        LOGGER.info("  running: %s", shown)
        path = self._log_file(log_name)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {dt.datetime.now().isoformat(timespec='seconds')} {shown}\n")
            fh.flush()
            proc = subprocess.run(list(cmd), cwd=self.root, env=self.env(), stdout=fh, stderr=subprocess.STDOUT)
        text = path.read_text(encoding="utf-8", errors="replace")
        if check and proc.returncode != 0:
            raise StepError(f"command failed (exit {proc.returncode}); see {path}\n  {shown}\n"
                            + "\n".join(text.splitlines()[-25:]))
        return proc.returncode, text

    def snakemake_cmd(self, targets: Sequence[str], rules: Sequence[str], run_ids: Sequence[str],
                      dry: bool) -> list[str]:
        cmd = [PYTHON, "-m", "snakemake", *targets, "--cores", str(self.o.cores), "--resources",
               f"mem_mb={self.o.mem_mb}", "--nolock", "--rerun-triggers", "mtime", "--rerun-incomplete",
               "--config", "run_scenarios=" + json.dumps(list(run_ids), separators=(",", ":"))]
        cmd += ["-n", "-p"] if dry else ["-p", "--keep-going"]
        return cmd + ["--allowed-rules", *rules]

    def plan(self, name: str, targets: Sequence[str], rules: Sequence[str], run_ids: Sequence[str],
             policy: str = "auto", done_outputs: Iterable[str] = ()) -> ExecutionPlan:
        """Dry-run ``targets``, parse and guard the plan, and log it."""
        cmd = self.snakemake_cmd(targets, rules, run_ids, dry=True)
        log = self._log_file(f"{name}_plan.log")
        log.write_text("", encoding="utf-8")
        LOGGER.info("  planning %s (snakemake -n -p): %s", name, " ".join(targets))
        with open(log, "a", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=self.root, env=self.env(), stdout=fh, stderr=subprocess.STDOUT)
        text = log.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise StepError(f"the Snakemake plan for {name} failed (exit {proc.returncode}); see {log}\n"
                            + "\n".join(text.splitlines()[-25:]))
        jobs = parse_dry_run(text)
        if not jobs and "Nothing to be done" not in text:
            LOGGER.warning("  the plan for %s parsed no jobs and did not say 'Nothing to be done'; see %s", name, log)
        done = set(done_outputs)
        if done:
            dropped = sum(1 for j in jobs if j.outputs and set(j.outputs) <= done)
            LOGGER.info("  [dry-run] %d of %d planned job(s) are the previous call's and are left to it", dropped,
                        len(jobs))
        plan = plan_execution(jobs, policy, done)
        LOGGER.info("  plan %s: %d job(s), mode %s; to run %d: %s", name, len(jobs), plan.mode, len(plan.run),
                    plan.counts())
        for j in plan.run:
            LOGGER.info("    %-48s %s", j.rule, ", ".join(j.outputs)[:150])
        if plan.skipped_2030:
            LOGGER.warning("  %d 2030 job(s) in the DAG are NOT run (rule names are shared with 2030): %s", len(
                plan.skipped_2030), sorted({f"{j.rule}[{j.wildcards}]" for j in plan.skipped_2030}))
            for f in plan.reads_2030:
                p = self.root / f
                LOGGER.warning("    2030 file read as it stands: %s (%s)", f,
                               _iso(p.stat().st_mtime) if p.exists() else "MISSING")
        self.state.setdefault("plans", {})[name] = {
            "mode": plan.mode, "jobs": len(jobs), "run": plan.counts(),
            "skipped_2030": sorted({f"{j.rule}[{j.wildcards}]" for j in plan.skipped_2030}),
            "reads_2030": {f: _iso((self.root / f).stat().st_mtime) if (self.root / f).exists() else None
                           for f in plan.reads_2030}}
        return plan

    def execute(self, name: str, targets: Sequence[str], rules: Sequence[str], run_ids: Sequence[str],
                policy: str = "auto", done_outputs: Iterable[str] = ()) -> ExecutionPlan:
        """Plan, guard and (unless a dry run) run one Snakemake call."""
        plan = self.plan(name, targets, rules, run_ids, policy, done_outputs)
        if plan.mode == "nothing":
            LOGGER.info("  %s: nothing to be done", name)
            return plan
        if self.o.dry_run:
            LOGGER.info("  [dry-run] would run the %s plan as %s", name, plan.mode)
            if plan.mode == "commands":
                for j in plan.run:
                    LOGGER.info("    [dry-run] %s: %s", j.rule, j.shell[:300])
            return plan
        if plan.mode == "snakemake":
            self.run_cmd(self.snakemake_cmd(targets, rules, run_ids, dry=False), f"{name}_snakemake.log")
            return plan
        bash = next((b for b in BASH_CANDIDATES if Path(b).exists()), shutil.which("bash"))
        if not bash:
            raise StepError("no bash found to run the printed commands")
        for k, j in enumerate(plan.run, 1):
            LOGGER.info("  [%d/%d] %s", k, len(plan.run), j.rule)
            start = dt.datetime.now().timestamp()
            self.run_cmd([bash, "-c", j.shell], f"{name}_commands.log")
            for o in j.outputs:
                p = self.root / o
                if not p.exists() or p.stat().st_mtime < start - 2:
                    raise StepError(f"{j.rule} finished but did not write {o}")
        return plan

    # 1
    def step_check(self) -> None:
        res = check_solve_outputs(self.root, self.o.network)
        LOGGER.info("  network %s written %s; %d outputs checked", res["network"], res["network_mtime"],
                    res["checked"])
        problems = []
        for key, label in (("missing", "missing"), ("stale", "not newer than the network"),
                           ("out_of_order", "BM older than its Stage 1")):
            if res[key]:
                problems.append(f"{len(res[key])} {label}")
                for p in res[key][:40]:
                    LOGGER.warning("    %s: %s", label, p)
                if len(res[key]) > 40:
                    LOGGER.warning("    ... and %d more", len(res[key]) - 40)
        net = self.root / self.o.network
        since = net.stat().st_mtime if net.exists() else None
        hits = scan_logs(run_logs(self.root), since)
        for h in hits[:30]:
            LOGGER.warning("    log error: %s:%d %s", h["log"], h["line"], h["text"])
        running = other_snakemake_running()
        for r in running:
            LOGGER.warning("    another Snakemake is running: %s", r[:200])
        self.state["check"] = {**{k: (len(v) if isinstance(v, list) else v) for k, v in res.items()},
                               "log_errors": len(hits), "other_snakemake": len(running)}
        if hits and not self.o.ignore_log_errors:
            problems.append(f"{len(hits)} log line(s) with {' / '.join(LOG_ERROR_PATTERNS)} "
                            "(re-run with --ignore-log-errors once checked)")
        if running and not self.o.allow_concurrent:
            problems.append(f"{len(running)} other Snakemake process(es) running (--allow-concurrent to override)")
        if problems:
            msg = "solve outputs not ready: " + "; ".join(problems)
            if self.o.dry_run:
                LOGGER.warning("  [dry-run] %s (a real run stops here)", msg)
                return
            raise StepError(msg)
        LOGGER.info("  all solve outputs present and newer than the network; no log errors")

    # 2
    def step_recovery(self) -> None:
        backup = self.out / "backup" / Path(RECOVERY).name
        if not self.o.dry_run and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / RECOVERY, backup)
            LOGGER.info("  backed up %s to %s", RECOVERY, backup)
        self.run_cmd([PYTHON, "-m", "scripts.consumer_bill.refresh_recovery_demand", "--recovery", RECOVERY,
                      "--scenarios", *market_ids()], "recovery.log")

    # 3
    def step_calibration(self) -> None:
        import pandas as pd

        self.execute("register", REGISTER_TARGETS, FINANCE_RULES, finance_run_ids())
        cal_dir = self.out / "calibration"
        bm_cal, bm_cap = cal_dir / "strike_calibration_2050_bm_basis.csv", cal_dir / "capture_prices_2050_bm_basis.csv"
        new_cal, new_cap = cal_dir / "strike_calibration_2050_rederived.csv", cal_dir / "capture_prices_2050_rederived.csv"
        labels = self.root / SUPPORT_LABELS
        if labels.exists() and (self.root / CALIBRATION).stat().st_mtime > labels.stat().st_mtime:
            LOGGER.warning("  %s changed after the BM labels were built (%s); the drift is measured against the "
                           "file, not the labels", CALIBRATION, SUPPORT_LABELS)
        if not self.o.dry_run:
            cal_dir.mkdir(parents=True, exist_ok=True)
            for src, dst in ((CALIBRATION, bm_cal), (CAPTURE, bm_cap)):
                if dst.exists():
                    LOGGER.info("  keeping the first snapshot %s (a resumed run never overwrites it)", dst)
                else:
                    shutil.copy2(self.root / src, dst)
                    LOGGER.info("  snapshot of the file the BM bid on: %s -> %s", src, dst)
        mod = [PYTHON, "-m", "scripts.public_ownership.vintage_capital_2050"]
        self.run_cmd(mod + ["capture", "--register", REGISTER, "--out", str(new_cap)], "calibration.log")
        if self.o.calibration == "install":
            self._install(new_cap, CAPTURE)
        self.run_cmd(mod + ["calibrate", "--register", REGISTER, "--book", BOOK, "--assumptions", ASSUMPTIONS,
                            "--config", FINANCE_CONFIG, "--out", str(new_cal)], "calibration.log")
        if self.o.dry_run:
            LOGGER.info("  [dry-run] would report the drift of %s against %s", new_cal, bm_cal)
            if self.o.calibration == "install":
                LOGGER.info("  [dry-run] would install %s as %s (one pass; never iterated)", new_cal, CALIBRATION)
            return
        summary, table = compare_calibrations(pd.read_csv(bm_cal), pd.read_csv(new_cal))
        summary["capture"] = compare_capture(pd.read_csv(bm_cap), pd.read_csv(new_cap))
        summary["mode"] = self.o.calibration
        # Decision 71: the margin the BM bid on and the margin re-derived (both the study's 5%, or a stale file).
        summary["auction_margin"] = calibration_margins(self.root / FINANCE_CONFIG, bm_cal, new_cal)
        if summary["auction_margin"]["rederived"] != summary["auction_margin"]["configured"]:
            raise StepError(f"the re-derived calibration is at auction margin {summary['auction_margin']['rederived']}, "
                            f"not the study's {summary['auction_margin']['configured']} (decision 71)")
        if summary["auction_margin"]["bm_basis"] != summary["auction_margin"]["configured"]:
            LOGGER.warning("  the BM bid on a calibration at auction margin %s, not the study's %s: the drift "
                           "includes the margin (decision 71)", summary["auction_margin"]["bm_basis"],
                           summary["auction_margin"]["configured"])
        table.to_csv(cal_dir / "calibration_drift.csv", index=False)
        (cal_dir / "calibration_drift.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        for scheme, s in summary["by_scheme"].items():
            LOGGER.info("  drift %s: largest %.4f %s, mean %.4f over %d cohorts", scheme, s["max_abs_drift"],
                        s["unit"], s["mean_abs_drift"], s["cohorts"])
        LOGGER.info("  capture-price drift: largest %.4f GBP/MWh", summary["capture"]["max_abs_drift"])
        if summary["strike_table_vs_bm_basis_max_gap"] > 0.01:
            LOGGER.warning("  the strike table in use differs from the BM-basis calibration by up to %.4f",
                           summary["strike_table_vs_bm_basis_max_gap"])
        if self.o.calibration == "install":
            self._install(new_cal, CALIBRATION)
        # Decision 71: the bounds' calibrations, the central one re-margined (no register read).
        self.write_margin_variants(CALIBRATION if self.o.calibration == "install" else str(new_cal), cal_dir)
        # Decision 54: the wind ramp's own calibration, on its private case at the strikes in use (one pass).
        ramp_cal = cal_dir / "strike_calibration_2050_neso_wind_ramp_rederived.csv"
        self.run_cmd(mod + ["calibrate", "--register", REGISTER, "--book", BOOK, "--assumptions", ASSUMPTIONS,
                            "--config", FINANCE_CONFIG, "--rate-path", RATE_PATH, "--out", str(ramp_cal)],
                     "calibration.log")
        if self.o.calibration == "install":
            self._install(ramp_cal, RAMP_CALIBRATION)
        self.state["calibration"] = {"mode": self.o.calibration, "drift": str(cal_dir / "calibration_drift.json"),
                                     "rate_path": str(ramp_cal)}

    def write_margin_variants(self, central: str, cal_dir: Path) -> None:
        """Decision 71: the margin bounds' calibrations, re-margined from ``central`` (installed, or the report copy)."""
        mod = [PYTHON, "-m", "scripts.public_ownership.vintage_capital_2050"]
        for level, dst in MARGIN_VARIANTS.items():
            out = dst if self.o.calibration == "install" else str(cal_dir / Path(dst).name)
            if self.o.calibration == "install" and not self.o.dry_run and (self.root / dst).exists():
                backup = self.out / "backup" / Path(dst).name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(self.root / dst, backup)
            self.run_cmd(mod + ["margin", "--calibration", central, "--auction-margin", level, "--out", out],
                         "calibration.log")

    def _install(self, src: Path, dst: str) -> None:
        if self.o.dry_run:
            LOGGER.info("  [dry-run] would install %s as %s", src, dst)
            return
        backup = self.out / "backup" / Path(dst).name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists() and (self.root / dst).exists():   # a first derive has nothing to back up
            shutil.copy2(self.root / dst, backup)
        shutil.copy2(src, self.root / dst)
        LOGGER.info("  installed %s as %s (previous copy in %s)", src.name, dst, backup)

    # 4
    def step_finance(self) -> None:
        chain = self.execute("chain", CHAIN_TARGETS, FINANCE_RULES, finance_run_ids())
        # A real run plans the carried targets after the chain has run; a dry run leaves the chain's
        # jobs to the chain, so the carried plan shows what the real call would run.
        done = {o for j in chain.run for o in j.outputs} if self.o.dry_run else ()
        carried = self.execute("carried", CARRIED_TARGETS, FINANCE_RULES, finance_run_ids(), done_outputs=done)
        # Decision 61: the register bounds on the same months, after the central chain (no solve).
        done_b = ({o for j in chain.run for o in j.outputs} | {o for j in carried.run for o in j.outputs}
                  if self.o.dry_run else ())
        self.execute("bounds", BOUND_TARGETS, FINANCE_RULES, finance_run_ids(), done_outputs=done_b)
        mod = [PYTHON, "-m", "scripts.public_ownership.vintage_capital_2050"]
        code, text = self.run_cmd(mod + ["calibrate", "--register", REGISTER, "--book", BOOK, "--assumptions",
                                         ASSUMPTIONS, "--config", FINANCE_CONFIG, "--check"],
                                  "calibration_residual.log", check=False)
        if not self.o.dry_run:
            m = re.findall(r"largest re-derivation drift: ([-\d.eE+]+)", text)
            residual = float(m[-1]) if m else None
            LOGGER.info("  residual calibration drift on the new register: %s (report only; decision 49 is one "
                        "pass)", residual)
            self.state["calibration_residual"] = {"largest_drift": residual, "check_exit": code}

    # 5
    def step_strike_diagnostic(self) -> None:
        """Decision 70: the post-AR7 strikes on the central-dispatch year, beside the market year's (report only).

        Reads the calibration the chain used (``data/``, installed in step 3 or left as the BM bid
        on), the chain's contract book and bill summary, and the solved market and central months.
        """
        mod = [PYTHON, "-m", "scripts.public_ownership.vintage_capital_2050"]
        self.run_cmd(mod + ["calibrate", "--dispatch-basis", "central", "--book", BOOK, "--register", REGISTER,
                            "--market-calibration",
                            CALIBRATION, "--study", STUDY, "--central-suffix", CENTRAL_SUFFIX, "--bill-summary",
                            BILL_SUMMARY, "--out", STRIKE_DIAGNOSTIC, "--json-out", STRIKE_DIAGNOSTIC_JSON],
                     "strike_diagnostic.log")
        if self.o.dry_run:
            return
        diag = _read_json(self.root / STRIKE_DIAGNOSTIC_JSON) or {}
        total = diag.get("total") or {}
        LOGGER.info("  decision 70: strike %s -> %s GBP/MWh (paid-weighted), CfD line %s GBP m, %s GBP per 3,100 kWh "
                    "ex VAT; material: %s", _f(total.get("market_strike_paid_weighted")),
                    _f(total.get("central_strike_paid_weighted")), _f(total.get("cfd_line_effect_gbp_m"), 1),
                    _f(total.get("cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat")),
                    (diag.get("materiality") or {}).get("material"))
        self.state["strike_diagnostic"] = {"csv": STRIKE_DIAGNOSTIC, "json": STRIKE_DIAGNOSTIC_JSON}

    # 6
    def step_validation(self) -> None:
        self.execute("validation", VALIDATION_TARGETS, VALIDATION_RULES, market_ids())
        if self.o.skip_compare:
            LOGGER.info("  NT+ year comparison skipped (--skip-compare)")
            return
        self.run_cmd([PYTHON, "scripts/market/compare_tyndp26_year_2050.py", "--out-dir",
                      str(self.out / "nt_year")], "nt_year.log")

    # 7
    def step_summary(self) -> None:
        if self.o.dry_run:
            LOGGER.info("  [dry-run] would write %s and %s", self.out / "summary.json", self.out / "summary.md")
            return
        summary = build_summary(self.root, self.out, self.state)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        (self.out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
        LOGGER.info("  wrote %s and %s", self.out / "summary.json", self.out / "summary.md")

    def run(self, steps: Sequence[str]) -> None:
        funcs: dict[str, Callable[[], None]] = {n: getattr(self, f"step_{n}") for n in STEP_NAMES}
        for n in steps:
            k = STEP_NAMES.index(n) + 1
            LOGGER.info("=== step %d/%d %s%s", k, len(STEP_NAMES), n, " [dry-run]" if self.o.dry_run else "")
            try:
                funcs[n]()
            except StepError as exc:
                self._save_state()
                if self.o.dry_run:  # a dry run reports every step; a real run stops here
                    LOGGER.error("  [dry-run] a real run would STOP at step %d (%s): %s", k, n, exc)
                    self.dry_run_errors.append(f"step {k} ({n}): {exc}")
                    continue
                raise StepError(f"STOPPED at step {k} ({n}): {exc}\nFix it, then resume with --from-step {n}") \
                    from exc
            self.state.setdefault("completed", {})[n] = dt.datetime.now().isoformat(timespec="seconds")
            self._save_state()


def _quote(s: str) -> str:
    s = str(s)
    return f'"{s}"' if (" " in s or "[" in s) else s


# ── the summary ──────────────────────────────────────────────────────────────

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ledger_cases(ledger: dict) -> list[dict]:
    """Bills and the three-party columns (GBP bn a year vs private) for every ledger case."""
    labels = ledger.get("headline_cases", {})
    rows = []
    for cid, c in ledger.get("cases", {}).items():
        def bn(key):
            v = c.get(key)
            return None if v is None else float(v) / 1e9
        row = {"case": cid, "label": labels.get(cid, ""), "bill_inc_vat": c.get("bill_inc_vat"),
               "consumers_bn": bn("consumer_change_vs_private_gbp"),
               "exchequer_bn": bn("exchequer_net_vs_private_gbp"),
               "investors_bn": bn("investor_change_vs_private_gbp"),
               "resource_saving_bn": bn("resource_saving_vs_private_gbp")}
        parts = [row[k] for k in ("consumers_bn", "exchequer_bn", "investors_bn", "resource_saving_bn")]
        row["identity_residual_gbp"] = (None if any(p is None for p in parts)
                                        else (parts[0] + parts[1] + parts[2] - parts[3]) * 1e9)
        rows.append(row)
    return rows


def p2_boundary_rent(register_path: Path, top: int = 8) -> dict | None:
    """Private-case producer rent of the plant inside P2's boundary (P2 share > 0), GBP bn.

    ``rent_excl_removal_bn`` leaves out the carbon-removal CfD (the old measure);
    ``rent_bn`` counts it (expired BECCS's removal value).
    """
    if not register_path.exists():
        return None
    import pandas as pd
    from scripts.public_ownership.register_io import read_register

    cols = ["case_id", "asset_id", "carrier", "ownership_share", "producer_rent_gbp", "carbon_removal_cfd_revenue_gbp"]
    reg = read_register(register_path, columns=cols, cases=["private", "public_ownership"])
    pv = reg[reg["case_id"].astype(str) == "private"].reset_index(drop=True)
    p2 = reg[reg["case_id"].astype(str) == "public_ownership"].reset_index(drop=True)
    if len(pv) != len(p2) or not (pv["asset_id"].astype(str).values == p2["asset_id"].astype(str).values).all():
        raise StepError("the register's private and public_ownership rows do not align by asset")
    share = pd.to_numeric(p2["ownership_share"], errors="coerce").fillna(0.0)
    rent = pd.to_numeric(pv["producer_rent_gbp"], errors="coerce").fillna(0.0)
    removal = pd.to_numeric(pv["carbon_removal_cfd_revenue_gbp"], errors="coerce").fillna(0.0)
    inside = share > 0
    by = (pd.DataFrame({"carrier": pv["carrier"].astype(str), "rent": rent})[inside]
          .groupby("carrier")["rent"].sum().div(1e9).sort_values())
    return {"rent_bn": float(rent[inside].sum() / 1e9), "rent_excl_removal_bn": float((rent - removal)[inside].sum() / 1e9),
            "by_carrier_bn": {k: float(v) for k, v in by.head(top).items()}}


def pathway_rows(pathway: dict) -> list[dict]:
    rows = []
    for cid, c in (pathway or {}).get("cases", {}).items():
        parties = c.get("parties", {})

        def tot(party, key):
            return ((parties.get(party) or {}).get("total") or {}).get(key)
        hh = (parties.get("consumers") or {}).get("per_household_3100kwh_gbp2025") or {}
        beyond = (c.get("beyond_2050") or {}).get("parties") or {}
        rows.append({"case": cid, "label": c.get("label", ""),
                     "resource_bn": _div(tot("resource", "undiscounted")),
                     "resource_npv_bn": _div(tot("resource", "npv_stpr_3p5")),
                     "consumers_bn": _div(tot("consumers", "undiscounted")),
                     "consumers_npv_bn": _div(tot("consumers", "npv_stpr_3p5")),
                     "consumers_per_household": hh.get("undiscounted"),
                     "consumers_per_household_npv": hh.get("npv_stpr_3p5"),
                     "beyond_2050_resource_npv_bn": _div(((beyond.get("resource") or {}).get("total") or {})
                                                         .get("npv_stpr_3p5"))})
    return rows


def _div(v, by=1e9):
    return None if v is None else float(v) / by


def bm_by_month(root: Path) -> list[dict]:
    """Stage 1 shed and price, BM shed and BM cost excluding shedding, per month, market and central."""
    import pandas as pd

    rows = []
    for m in MONTHS:
        row = {"month": m}
        for tag, sid in (("mkt", f"{STUDY}_{m}{MARKET_SUFFIX}"), ("cen", f"{STUDY}_{m}{CENTRAL_SUFFIX}")):
            d = root / MARKET / f"{sid}_wholesale_dispatch.csv"
            p = root / MARKET / f"{sid}_wholesale_price.csv"
            c = root / MARKET / f"{sid}_constraint_costs.csv"
            if d.exists():
                hdr = pd.read_csv(d, nrows=0).columns
                shed_cols = [x for x in hdr if "shed" in x.lower()]
                row[f"{tag}_S1_shed_GWh"] = float(pd.read_csv(d, usecols=shed_cols).sum().sum() / 1e3) if shed_cols else 0.0
            if p.exists():
                pr = pd.read_csv(p, index_col=0)
                row[f"{tag}_S1_price"] = float((pr["wholesale_price"] if "wholesale_price" in pr else pr.iloc[:, 0]).mean())
            if c.exists():
                cc = pd.read_csv(c)
                cc = cc[cc["carrier"] != "TOTAL"]
                row[f"{tag}_BM_shed_GWh"] = float(cc.loc[cc["carrier"] == "load_shedding", "increase_MWh"].sum() / 1e3)
                row[f"{tag}_BM_cost_GBPm"] = float(cc.loc[cc["carrier"] != "load_shedding", "net_cost"].sum() / 1e6)
        rows.append(row)
    keys = sorted({k for r in rows for k in r if k != "month"})
    year = {"month": "Year"}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        year[k] = (sum(vals) / len(vals) if k.endswith("_price") else sum(vals)) if vals else None
    return rows + [year]


def validation_counts(root: Path) -> dict | None:
    v = _read_json(root / VALIDATION_JSON)
    if v is None:
        return None
    counts = v.get("status_counts", {})
    return {"status_counts": counts, "WARN": int(counts.get("WARN", 0)), "FAIL": int(counts.get("FAIL", 0)),
            "unexplained_warns": v.get("unexplained_warns")}


#: Decision 70: the report groups the summary carries from the strike diagnostic.
STRIKE_DIAGNOSTIC_GROUPS = ("offshore", "onshore", "solar", "onshore_and_solar", "tidal", "nuclear")
STRIKE_DIAGNOSTIC_FIELDS = ("market_strike_paid_weighted", "central_strike_paid_weighted",
                            "strike_difference_paid_weighted", "market_uncompensated_twh", "central_uncompensated_twh",
                            "market_gross_curtailed_twh", "central_gross_curtailed_twh", "cfd_line_effect_gbp_m",
                            "cfd_and_compensation_effect_gbp_m", "cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat",
                            "cfd_and_compensation_effect_gbp_per_headline_household_ex_vat")


def strike_diagnostic_summary(path: Path) -> dict | None:
    """Decision 70: the strike diagnostic's total and report groups (None before it has run)."""
    diag = _read_json(path)
    if not diag:
        return None
    pick = lambda g: {k: (g or {}).get(k) for k in STRIKE_DIAGNOSTIC_FIELDS}  # noqa: E731
    return {"total": pick(diag.get("total")),
            "groups": {g: pick(diag["groups"][g]) for g in STRIKE_DIAGNOSTIC_GROUPS if g in (diag.get("groups") or {})},
            "materiality": diag.get("materiality"), "reconciliation": diag.get("reconciliation"),
            "settings": {k: (diag.get("settings") or {}).get(k) for k in ("central_price_basis", "negative_price_rule",
                                                                          "market_bm_bid_basis")}}


def build_summary(root: Path, out_dir: Path, state: dict | None = None) -> dict:
    """Everything the decision-49 write-up needs, from the outputs on disk."""
    ledger = _read_json(root / LEDGER) or {}
    pathway = _read_json(root / PATHWAY) or {}
    files = [LEDGER, PATHWAY, REGISTER, VALIDATION_JSON, CALIBRATION, CAPTURE, RECOVERY, STRIKE_DIAGNOSTIC_JSON]
    summary = {
        "written": dt.datetime.now().isoformat(timespec="seconds"),
        "price_basis": ledger.get("price_basis"),
        "bills_and_parties": ledger_cases(ledger),
        # Decision 61: each register bound's bills and parties (learning, retirement, repowering share 0.5;
        # decision 68's co-ownership reduction and pooled bounds).
        "register_bounds": {g: ledger_cases(_read_json(root / f"{PO}/{g}_exchequer_ledger.json") or {})
                            for g in BOUND_GROUPS},
        "p2_boundary_private_rent": p2_boundary_rent(root / REGISTER),
        "pathway": {"rows": pathway_rows(pathway),
                    "anchoring_all_pass": (pathway.get("anchoring") or {}).get("all_pass"),
                    "anchoring_worst_difference_gbp": (pathway.get("anchoring") or {}).get("worst_difference_gbp")},
        "bm_by_month": bm_by_month(root),
        "calibration_drift": _read_json(out_dir / "calibration" / "calibration_drift.json"),
        "calibration_residual": (state or {}).get("calibration_residual"),
        # Decision 70: the strike diagnostic's report groups (offshore apart from onshore and solar).
        "central_dispatch_strike_diagnostic": strike_diagnostic_summary(root / STRIKE_DIAGNOSTIC_JSON),
        "validation": validation_counts(root),
        "nt_year_delta": (_read_json(out_dir / "nt_year" / "year_compare.json") or {}).get("delta"),
        "plans": (state or {}).get("plans"),
        "inputs": {f: _iso((root / f).stat().st_mtime) if (root / f).exists() else None for f in files},
    }
    return summary


def _f(v, nd=2):
    return "" if v is None else (f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v))


def summary_markdown(s: dict) -> str:
    """The summary as markdown tables."""
    out = [f"# HT50 2050 post-solve summary (decision 49)\n\nWritten {s['written']}. {s.get('price_basis') or ''}\n"]
    out.append("## Bills and the three-party columns\n\nBill GBP inc VAT per 3,100 kWh; parties GBP bn a year vs "
               "private.\n\n| Case | Label | Bill | Consumers | Exchequer | Investors | Resource saving | Residual GBP |\n"
               "| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in s["bills_and_parties"]:
        out.append(f"| {r['case']} | {r['label']} | {_f(r['bill_inc_vat'])} | {_f(r['consumers_bn'])} | "
                   f"{_f(r['exchequer_bn'])} | {_f(r['investors_bn'])} | {_f(r['resource_saving_bn'])} | "
                   f"{_f(r['identity_residual_gbp'], 0)} |")
    rent = s.get("p2_boundary_private_rent")
    if rent:
        out.append(f"\n## P2-boundary private rent\n\n{_f(rent['rent_bn'])} bn (removal value counted); "
                   f"{_f(rent['rent_excl_removal_bn'])} bn excluding the carbon-removal CfD. By carrier: "
                   + ", ".join(f"{k} {_f(v)}" for k, v in rent["by_carrier_bn"].items()) + ".")
    out.append("\n## Pathway 2030-2050 (GBP bn; NPV at 3.5% to 2030)\n\n| Case | Label | Resource | Resource NPV | "
               "Consumers | Consumers NPV | Per household | Per household NPV | Beyond 2050 resource NPV |\n"
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in s["pathway"]["rows"]:
        out.append(f"| {r['case']} | {r['label']} | {_f(r['resource_bn'], 1)} | {_f(r['resource_npv_bn'], 1)} | "
                   f"{_f(r['consumers_bn'], 1)} | {_f(r['consumers_npv_bn'], 1)} | {_f(r['consumers_per_household'], 0)} | "
                   f"{_f(r['consumers_per_household_npv'], 0)} | {_f(r['beyond_2050_resource_npv_bn'], 1)} |")
    out.append(f"\nAnchoring all pass: {s['pathway']['anchoring_all_pass']}, worst difference GBP "
               f"{_f(s['pathway']['anchoring_worst_difference_gbp'], 0)}.")
    bounds = s.get("register_bounds") or {}
    if any(bounds.values()):
        # Decision 61: the register bounds (no re-solve), bills and parties against each bound's own private.
        out.append("\n## Register bounds (no re-solve)\n\n| Group | Case | Bill | Consumers | Exchequer | Investors | "
                   "Resource saving |\n| --- | --- | --- | --- | --- | --- | --- |")
        for group, rows in bounds.items():
            for r in rows:
                out.append(f"| {group} | {r['case']} | {_f(r['bill_inc_vat'])} | {_f(r['consumers_bn'])} | "
                           f"{_f(r['exchequer_bn'])} | {_f(r['investors_bn'])} | {_f(r['resource_saving_bn'])} |")
    cols = ["mkt_S1_shed_GWh", "mkt_S1_price", "mkt_BM_shed_GWh", "mkt_BM_cost_GBPm",
            "cen_S1_shed_GWh", "cen_S1_price", "cen_BM_shed_GWh", "cen_BM_cost_GBPm"]
    out.append("\n## BM shed and cost by month (shed GWh, price mean GBP/MWh, BM cost GBP m excluding shedding)\n\n"
               "| Month | " + " | ".join(cols) + " |\n| --- |" + " --- |" * len(cols))
    for r in s["bm_by_month"]:
        out.append(f"| {r['month']} | " + " | ".join(_f(r.get(c), 1) for c in cols) + " |")
    cal = s.get("calibration_drift")
    out.append("\n## Calibration drift (re-derived against the calibration the BM bid on; one pass)\n")
    if cal:
        for scheme, d in cal.get("by_scheme", {}).items():
            out.append(f"- {scheme}: largest {_f(d['max_abs_drift'], 3)} {d['unit']}, mean {_f(d['mean_abs_drift'], 3)} "
                       f"over {d['cohorts']} cohorts; largest: "
                       + ", ".join(f"{x['key']} {_f(x['drift'], 2)}" for x in d["largest"][:3]))
        out.append(f"- capture prices: largest drift {_f(cal.get('capture', {}).get('max_abs_drift'), 3)} GBP/MWh; "
                   f"mode {cal.get('mode')}")
    else:
        out.append("- not run")
    res = s.get("calibration_residual")
    if res:
        out.append(f"- residual on the new register: {_f(res.get('largest_drift'), 4)} (check exit {res.get('check_exit')})")
    sd = s.get("central_dispatch_strike_diagnostic")
    out.append("\n## Strike diagnostic on the central-dispatch year (decision 70)\n")
    if sd:
        out.append("Post-AR7 break-even strikes, paid-weighted GBP/MWh 2024; uncompensated curtailment TWh; the "
                   "strike difference on the CfD line (GBP m) and per 3,100 kWh (GBP ex VAT, CfD line plus "
                   "compensated turn-down).\n\n| Group | Market strike | Central strike | Uncompensated market | "
                   "Uncompensated central | CfD line GBP m | Per 3,100 kWh |\n| --- | --- | --- | --- | --- | --- | --- |")
        for name, g in {**sd["groups"], "total": sd["total"]}.items():
            out.append(f"| {name} | {_f(g['market_strike_paid_weighted'])} | {_f(g['central_strike_paid_weighted'])} | "
                       f"{_f(g['market_uncompensated_twh'])} | {_f(g['central_uncompensated_twh'])} | "
                       f"{_f(g['cfd_line_effect_gbp_m'], 1)} | "
                       f"{_f(g['cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat'])} |")
        m = sd.get("materiality") or {}
        out.append(f"\nMaterial (threshold {_f(m.get('threshold_gbp_per_3100kwh_ex_vat'))} GBP per 3,100 kWh, "
                   f"{m.get('threshold_status')}): {m.get('material')}.")
    else:
        out.append("- not run")
    v = s.get("validation")
    out.append("\n## Validation\n\n" + (f"WARN {v['WARN']}, FAIL {v['FAIL']}, unexplained WARNs {v['unexplained_warns']} "
                                        f"({v['status_counts']})." if v else "not available."))
    if s.get("nt_year_delta"):
        d = s["nt_year_delta"]
        out.append("\n## NT+ minus central (DE), year\n\n" + ", ".join(
            f"{k} {_f(d.get(k))}" for k in ("S1_price_load_weighted", "BM_cost_excl_shed_GBPm", "BM_retrading_GBPm",
                                             "BM_shed_GWh", "per_3100kWh_from_S1_price",
                                             "per_3100kWh_from_BM_and_retrading") if k in d) + ".")
    plans = s.get("plans") or {}
    reads = {f: t for p in plans.values() for f, t in (p.get("reads_2030") or {}).items()}
    if reads:
        out.append("\n## 2030 files read as they stand\n\n" + "\n".join(f"- {f} ({t})" for f, t in sorted(reads.items())))
    return "\n".join(out) + "\n"


# ── command line ─────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="steps: " + ", ".join(f"{i + 1}={n}" for i, n in enumerate(STEP_NAMES)))
    parser.add_argument("--from-step", default=None, help="first step (name or number)")
    parser.add_argument("--to-step", default=None, help="last step (name or number)")
    parser.add_argument("--dry-run", action="store_true", help="checks and Snakemake plans only; change nothing")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cores", type=int, default=6)
    parser.add_argument("--mem-mb", type=int, default=24000)
    parser.add_argument("--calibration", choices=("install", "report"), default="install",
                        help="install the re-derived capture prices and calibration in data/ (default) or report only")
    parser.add_argument("--ignore-log-errors", action="store_true")
    parser.add_argument("--allow-concurrent", action="store_true", help="run although another Snakemake is running")
    parser.add_argument("--skip-compare", action="store_true", help="skip the NT+ year comparison")
    parser.add_argument("--network", default=NETWORK)
    parser.add_argument("--list-steps", action="store_true")
    args = parser.parse_args(argv)
    if args.list_steps:
        for i, n in enumerate(STEP_NAMES, 1):
            print(f"{i} {n}")
        return 0
    steps = select_steps(args.from_step, args.to_step)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ("post_solve_dry_run.log" if args.dry_run else "post_solve.log")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers,
                        force=True)
    opts = Options(root=PROJECT_ROOT, out_dir=out_dir, dry_run=args.dry_run, cores=args.cores, mem_mb=args.mem_mb,
                   calibration=args.calibration, ignore_log_errors=args.ignore_log_errors,
                   allow_concurrent=args.allow_concurrent, skip_compare=args.skip_compare, network=args.network)
    LOGGER.info("post-solve 2050 (decision 49): steps %s%s; output %s", steps, " [DRY RUN]" if args.dry_run else "",
                out_dir)
    pipe = Pipeline(opts)
    try:
        pipe.run(steps)
    except StepError as exc:
        LOGGER.error("%s", exc)
        return 1
    if pipe.dry_run_errors:
        stops = [e.splitlines()[0][:300] for e in pipe.dry_run_errors]
        LOGGER.error("dry run: a real run would stop at:%s", "".join(f"\n  {e}" for e in stops))
        return 1
    LOGGER.info("done: %s", ", ".join(steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
