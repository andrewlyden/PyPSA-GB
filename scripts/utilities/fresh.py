"""Refuse to read workflow outputs that Snakemake would rebuild (user, 15 Nov 2026: "stale files over and over").

Any check run outside the workflow (a scratch comparison, a notebook, an in-flight check during a run) first asks
Snakemake whether the files it reads are current:

    python -m scripts.utilities.fresh resources/public_ownership/x_asset_finance_register.parquet ...

Exit 0 and "fresh" when a dry run of those files plans nothing; exit 1 with the rules it would run otherwise. It uses
the run's own rerun triggers (mtime; the bill layer declares its code, so a code edit makes its outputs stale too).
Use ``--allow-running`` during a run: it skips Snakemake's lock (the dry run never writes).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def planned_jobs(paths: list[str], *, allow_running: bool = False) -> dict[str, int]:
    """Rule -> job count Snakemake would run to bring ``paths`` up to date (empty when they are current)."""
    command = [sys.executable, "-m", "snakemake", "-n", "--rerun-triggers", "mtime", "--cores", "1"]
    if allow_running:
        command.append("--nolock")
    result = subprocess.run(command + ["--", *paths], cwd=ROOT, capture_output=True, text=True)
    out = result.stdout + result.stderr
    if result.returncode != 0:
        raise SystemExit(f"snakemake dry run failed:\n{out[-3000:]}")
    if "Nothing to be done" in out:
        return {}
    stats = out[out.find("Job stats:"):] if "Job stats:" in out else ""
    jobs = {}
    for line in stats.splitlines()[3:]:
        match = re.match(r"^(\S+)\s+(\d+)$", line.strip())
        if match and match.group(1) != "total":
            jobs[match.group(1)] = int(match.group(2))
        elif line.strip().startswith("total"):
            break
    return jobs


def require_fresh(paths: list[str], *, allow_running: bool = False) -> None:
    """Raise SystemExit naming the rules that would run, unless every path is current."""
    jobs = planned_jobs(paths, allow_running=allow_running)
    if jobs:
        listing = ", ".join(f"{rule} x{n}" for rule, n in sorted(jobs.items()))
        raise SystemExit(f"stale: Snakemake would run {listing} before these files are current; "
                         "build them first (or read the files of a finished run)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--allow-running", action="store_true", help="skip the workflow lock (during a run)")
    args = parser.parse_args(argv)
    require_fresh(args.paths, allow_running=args.allow_running)
    print("fresh: " + ", ".join(args.paths))


if __name__ == "__main__":
    main()
