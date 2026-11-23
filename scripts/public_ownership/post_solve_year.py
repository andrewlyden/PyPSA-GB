"""Post-solve steps for a long-range FES year, in order (user, 11 Nov 2026: hand-made steps into scripts).

After a year's market months are solved on a seed strike calibration (``new_fes_year`` seeds it from 2050's rows):

1. ``recovery``: refresh the year's recovery-demand volumes from its twelve market months
   (``scripts.consumer_bill.refresh_recovery_demand``; the pre-run file is backed up once).
2. ``register``: the year's register, contract book, finance config and strike table (Snakemake, guarded: no solve
   or network rule may run).
3. ``calibration``: snapshot the files the balancing months bid on (once), then capture prices -> install ->
   calibrate -> report the drift -> install -> the auction-margin variants -> the wind-ramp calibration where the
   study names one. The capture prices go in **before** ``calibrate``: a calibration priced on other capture
   prices is refused by the support labels (11 Nov 2026, the 2040 first pass). One pass, never iterated (decision
   49 b).
4. ``labels``: rebuild the year's support labels on the installed calibration. Their calibration inputs are
   ancient (the BM bids on one pass), so a re-derived calibration never rebuilds them by itself.
5. ``check``: every calibrated CfD cohort in the labels carries the installed calibration's strike
   (:func:`labels_match_calibration`), so no balancing month bids on a stale calibration.

The balancing months are re-solved after this (they read the labels). 2050's chain stays
``post_solve_2050.py``.

    python scripts/public_ownership/post_solve_year.py --year 2040 [--steps recovery,register,calibration,labels,check]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
STEPS = ("recovery", "register", "calibration", "labels", "check")
FORBIDDEN_PREFIXES = ("solve_", "add_", "finalize_", "cluster_", "build_ETYS_", "apply_marginal_costs")
TOLERANCE = 1e-6


class StepError(RuntimeError):
    pass


def paths(year: int) -> dict:
    y = int(year)
    fin = "data/public_ownership/financing"
    po = "resources/public_ownership"
    return {
        "study": f"FES25_HT_{y}", "group": f"fes_ht_{y}",
        "recovery": f"data/market/consumer_bill_forecast/{y}_recovery_demand_assumptions.json",
        "assumptions": f"data/market/consumer_bill_benchmarks/{y}_pypsa_informed_assumptions.json",
        "capture": f"{fin}/capture_prices_{y}.csv", "calibration": f"{fin}/strike_calibration_{y}.csv",
        "margins": {lv: f"{fin}/strike_calibration_{y}_auction_margin_{lv}.csv" for lv in ("none", "high")},
        "ramp": f"{fin}/strike_calibration_{y}_neso_wind_ramp.csv",
        "register": f"{po}/fes_ht_{y}_asset_finance_register.parquet",
        "book": f"{po}/fes_ht_{y}_support_contract_book.csv",
        "finance_config": f"{po}/FES25_HT_{y}_finance_config.yaml",
        "strike_table": f"{po}/FES25_HT_{y}_cfd_strike_table.csv",
        "labels": f"resources/market/FES25_HT_{y}_support_labels.csv",
        "out": f"resources/analysis/FES25_HT_{y}_post_solve",
    }


def labels_match_calibration(labels: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """The calibrated CfD cohorts whose label strike differs from the calibration's (empty when they agree)."""
    cfd = labels.loc[labels["support_type"].astype(str).str.lower().eq("cfd")]
    strikes = cfd.groupby("strike_key")["strike_gbp_per_mwh_2024"].agg(["min", "max"])
    cal = calibration.loc[calibration["scheme"].eq("CfD")].set_index("key")["calibrated_value"]
    joined = strikes.join(cal, how="inner")
    gap = joined[["min", "max"]].sub(joined["calibrated_value"], axis=0).abs().max(axis=1)
    return joined.assign(gap=gap).loc[gap > TOLERANCE]


class Runner:
    def __init__(self, year: int, dry_run: bool = False, stamp: str | None = None):
        self.year, self.p, self.dry, self.stamp = int(year), paths(year), dry_run, stamp
        env_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "miniforge3/envs/pypsa-gb"
        self.python = str(env_dir / "python.exe") if (env_dir / "python.exe").exists() else sys.executable
        self.sm = str(env_dir / "Scripts/snakemake.exe") if (env_dir / "Scripts/snakemake.exe").exists() else "snakemake"
        self.env = dict(os.environ, PATH=os.pathsep.join([str(env_dir), str(env_dir / "Library/bin"),
                                                          str(env_dir / "Scripts"), os.environ.get("PATH", "")]),
                        PYTHONIOENCODING="utf-8")
        self.out = ROOT / self.p["out"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = self.out / "post_solve.log"

    def run(self, cmd: list[str], what: str) -> str:
        print(f"[{what}] {' '.join(cmd)}")
        if self.dry:
            return ""
        res = subprocess.run(cmd, cwd=ROOT, env=self.env, capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n### {what}: {' '.join(cmd)}\n{res.stdout}\n{res.stderr}\n")
        if res.returncode:
            raise StepError(f"{what} failed (exit {res.returncode}); see {self.log}")
        return res.stdout

    def snakemake(self, targets: list[str], what: str, extra: list[str] | None = None) -> None:
        base = [self.sm, "--cores", "6", "--resources", "mem_mb=13000", "--rerun-triggers", "mtime",
                "--rerun-incomplete", "--latency-wait", "60", *(extra or [])]
        plan = self.run([*base, "-n", "--", *targets], f"{what} plan") if not self.dry else ""
        stats = plan[plan.find("Job stats"):] if "Job stats" in plan else ""
        bad = [line.split()[0] for line in stats.splitlines()[2:]
               if line.split() and line.split()[0].startswith(FORBIDDEN_PREFIXES)]
        if bad:
            raise StepError(f"{what}: the plan would run {sorted(set(bad))}; refusing (solves and builds are not "
                            "post-solve steps)")
        self.run([*base, "--", *targets], what)

    def py(self, args: list[str], what: str) -> str:
        return self.run([self.python, *args], what)

    def install(self, src: Path, dst: str) -> None:
        backup = self.out / "backup" / Path(dst).name
        if not self.dry:
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists() and (ROOT / dst).exists():
                shutil.copy2(ROOT / dst, backup)
            shutil.copy2(src, ROOT / dst)
        print(f"installed {src.name} as {dst} (first previous copy in {backup})")

    # -- steps ---------------------------------------------------------------------------------------------
    def recovery(self) -> None:
        backup = self.out / "backup" / Path(self.p["recovery"]).name
        if not self.dry and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / self.p["recovery"], backup)
        ids = [f"{self.p['study']}_{m}_market" for m in MONTHS]
        self.py(["-m", "scripts.consumer_bill.refresh_recovery_demand", "--recovery", self.p["recovery"],
                 "--scenarios", *ids], "recovery")

    def register(self) -> None:
        self.snakemake([self.p[k] for k in ("register", "book", "finance_config", "strike_table")], "register")

    def calibration(self) -> None:
        cal_dir = self.out / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        y = self.year
        bm_cal, bm_cap = cal_dir / f"strike_calibration_{y}_bm_basis.csv", cal_dir / f"capture_prices_{y}_bm_basis.csv"
        for src, dst in ((self.p["calibration"], bm_cal), (self.p["capture"], bm_cap)):
            if not dst.exists() and not self.dry:
                shutil.copy2(ROOT / src, dst)
        mod = ["-m", "scripts.public_ownership.vintage_capital_2050"]
        reg = ["--register", self.p["register"]]
        new_cap = cal_dir / f"capture_prices_{y}_rederived.csv"
        self.py([*mod, "capture", *reg, "--out", str(new_cap)], "capture")
        self.install(new_cap, self.p["capture"])            # before calibrate (see the module docstring)
        calibrate = [*mod, "calibrate", *reg, "--book", self.p["book"], "--assumptions", self.p["assumptions"],
                     "--config", self.p["finance_config"]]
        new_cal = cal_dir / f"strike_calibration_{y}_rederived.csv"
        self.py([*calibrate, "--out", str(new_cal)], "calibrate")
        if not self.dry:
            from scripts.public_ownership.post_solve_2050 import compare_calibrations, compare_capture

            summary, table = compare_calibrations(pd.read_csv(bm_cal), pd.read_csv(new_cal))
            summary["capture"] = compare_capture(pd.read_csv(bm_cap), pd.read_csv(new_cap))
            table.to_csv(cal_dir / "calibration_drift.csv", index=False)
            (cal_dir / "calibration_drift.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
            for scheme, s in summary["by_scheme"].items():
                print(f"drift {scheme}: largest {s['max_abs_drift']:.4f} {s['unit']}, mean {s['mean_abs_drift']:.4f} "
                      f"over {s['cohorts']} cohorts")
        self.install(new_cal, self.p["calibration"])
        for level, dst in self.p["margins"].items():
            self.py([*mod, "margin", "--calibration", self.p["calibration"], "--auction-margin", level, "--out", dst],
                    f"margin {level}")
        study = (ROOT / f"config/studies/fes{y}/templates.yaml")
        if study.exists() and Path(self.p["ramp"]).name in study.read_text(encoding="utf-8"):
            ramp = cal_dir / f"strike_calibration_{y}_neso_wind_ramp_rederived.csv"
            self.py([*calibrate, "--rate-path", "neso_wind_ramp", "--out", str(ramp)], "ramp calibration")
            self.install(ramp, self.p["ramp"])

    def labels(self) -> None:
        self.snakemake([self.p["labels"]], "labels", ["-f", "--allowed-rules", "build_support_labels"])

    def check(self) -> None:
        if self.dry:
            return
        bad = labels_match_calibration(pd.read_csv(ROOT / self.p["labels"], low_memory=False),
                                       pd.read_csv(ROOT / self.p["calibration"]))
        if len(bad):
            raise StepError(f"{len(bad)} calibrated CfD cohorts in {self.p['labels']} differ from "
                            f"{self.p['calibration']} (e.g. {bad.index[0]}): rebuild the labels (--steps labels)")
        print(f"check: the labels carry the installed calibration ({self.p['labels']})")
        if self.stamp:
            Path(self.stamp).write_text(json.dumps({"labels": self.p["labels"], "calibration": self.p["calibration"],
                                                    "agrees": True}, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated, in order: {','.join(STEPS)}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stamp", default=None, help="with the check step: write a JSON stamp when the labels agree "
                                                  "(rule verify_support_labels_calibration)")
    a = ap.parse_args(argv)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    if a.year == 2050 and steps != ["check"]:
        print("2050 keeps scripts/public_ownership/post_solve_2050.py (only its check runs here)", file=sys.stderr)
        return 2
    if unknown := [s for s in steps if s not in STEPS]:
        print(f"unknown steps {unknown}; supported: {list(STEPS)}", file=sys.stderr)
        return 2
    runner = Runner(a.year, dry_run=a.dry_run, stamp=a.stamp)
    for step in STEPS:
        if step in steps:
            print(f"== {step}")
            try:
                getattr(runner, step)()
            except StepError as exc:
                print(f"STOPPED at {step}: {exc}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
