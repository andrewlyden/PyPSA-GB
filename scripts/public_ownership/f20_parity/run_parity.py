"""F20 phase 0: freeze a reference, run the ownership layer into its own root, compare (13 Oct 2026).

Three steps, each its own sub-command (``README.md`` beside this file has the
exact commands):

``freeze``
    Copy a tree's 2030 ownership outputs (``resources/public_ownership/cp30_2030*``)
    to a reference folder, read-only on the source, with a manifest of what was
    copied and the source's commit, working-tree digest and config hashes.

``run``
    Run the ownership layer from this tree (``rule public_ownership_parity``)
    into ``ROOT/resources/public_ownership``, reading every solved input (the
    market months, the bill forecasts, the capacity market) from another tree's
    ``resources`` (``--input-resources``).  Snakemake gets
    ``--config resources_path=<input> po_resources_path=ROOT/resources f20_parity=true``
    and ``--allowed-rules`` with the ownership rules only (:data:`ALLOWED_RULES`,
    every one of which writes under ``po_resources_path`` alone; a unit test
    checks that against ``rules/public_ownership.smk``), so nothing is written to
    the input tree.  Stage timing is on (``ROOT/stage_timing``, summarised to
    ``ROOT/stage_times.md``).

``compare``
    :mod:`scripts.public_ownership.compare_outputs` over two roots, with each
    root's own path strings (from its manifest) normalised to ``resources``.

Nothing here runs unless asked; importing it (the Snakefile does, for
:func:`check_roots`) has no effect.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: The ownership rules a parity run may execute: every one writes only under
#: ``{po_resources_path}/public_ownership`` (test_f20_parity_harness.py checks the rule file).
ALLOWED_RULES = (
    "aggregate_public_ownership_reference_run",
    "build_public_ownership_asset_register",
    "build_public_ownership_asset_settlement",
    "build_public_ownership_price_level_settlement",
    "build_public_ownership_price_level_twin_settlement",
    # P1/P2 under central dispatch (20 Oct 2026): the institution twin's settlement and the merge.
    "build_public_ownership_institution_twin_settlement",
    "calculate_public_ownership_financing",
    "build_public_ownership_state_shares",
    "build_public_ownership_commissioning_years",
    "verify_public_ownership_ownership_tables",
    "build_public_ownership_legacy_valuation",
    "build_public_ownership_network_finance_account",
    "build_public_ownership_cfd_strike_adjustments",
    "build_public_ownership_public_account",
    "build_public_ownership_consumer_bill_bridge",
    "project_public_ownership_consumer_bill",
    "build_public_bm_bid_decomposition",
    "merge_public_ownership_balancing_groups",
    "merge_public_ownership_tariff_balancing_groups",
    "build_public_ownership_nationalised_bill",
    "build_public_ownership_institution_comparison",
    # F27 items 3, 13 and 15 (24 Oct 2026): the declared items the ledger reads.
    "build_public_ownership_declared_items",
    "build_public_ownership_exchequer_ledger",
    "build_public_ownership_workbook_esm_feedback",
    "merge_public_ownership_institution_groups",
    "build_public_ownership_case_manifest",
    "verify_public_ownership_cost_calibration",
    "public_ownership_parity",
)
#: Rules whose commands pass cmd.exe's 8,191 characters once every month file is an absolute path into the
#: input tree; a parity run on Windows runs them in PowerShell (:func:`parity_shell`, :func:`snakemake_command`).
LONG_COMMAND_RULES = (
    "build_public_ownership_asset_settlement",
    "build_public_ownership_price_level_settlement",
    "build_public_ownership_price_level_twin_settlement",
    "build_public_ownership_institution_twin_settlement",
    "build_public_ownership_cfd_strike_adjustments",
)
#: Every other rule in rules/public_ownership.smk, and why a parity run leaves it out.
EXCLUDED_RULES = {
    "resolve_public_ownership_study_config": "2050 study (decision 31)",
    "extract_public_ownership_workbook_assumptions": "input: the workbook snapshot is read from the input tree",
    "build_public_ownership_fleet_vintages": "2050 (vintaged registers)",
    "build_public_ownership_vintaged_asset_register": "2050 (vintaged registers)",
    "build_public_ownership_cfd_strike_table_2050": "2050",
    "build_public_ownership_auction_margin_inputs_2050": "2050",
    "build_public_ownership_contract_book": "2050",
    "build_public_ownership_vintaged_asset_settlement": "2050",
    "build_public_ownership_deployment_share_tables": "2050 (L4 deployment shares)",
    "build_public_ownership_repowering_sensitivity": "report only",
    "extract_network_price_control_building_blocks": "input: writes resources/analysis",
    "build_public_ownership_carried_acquisition": "2050 (a purchase carried from 2030)",
    "build_public_ownership_lever_workbook": "report: writes resources/analysis",
    "build_public_ownership_explorer": "report: writes resources/analysis",
    "build_dispatch_pricing_explainer": "report: writes resources/analysis",
    "build_cp30_bill_notebook": "report: writes notebooks/",
    "build_bill_docs_values": "report: writes docs/",
    "build_capacity_market": "input: writes resources/market",
    "build_must_run_support_sensitivity": "report only",
    "report_capacity_market_revenue_adequacy": "report only",
    "build_public_ownership_pathway_2030_2050": "2050 pathway report",
    # FES 2050 decision 70 (FES25_HT_2050_cd, a study forecast case; no 2030 case has central_dispatch_strikes).
    "build_central_dispatch_strike_inputs_2050": "2050 (decision 70: a central-dispatch case's strike inputs)",
    "build_central_dispatch_contract_book_2050": "2050 (decision 70: a central-dispatch case's contract book)",
    "build_central_dispatch_case_fti_settlement": "2050 (decision 70), writes resources/market",
    # Long-range years (11 Nov 2026), classified 13 Nov 2026.
    "build_public_ownership_price_anchors": "long-range: built from the 2040/2050 registers (a parity run reads it)",
    "build_capacity_market_year": "2031-2049 Capacity Market, writes resources/market",
    "verify_capacity_market_pin": "check: writes resources/market (CM_OUT)",
    "verify_support_labels_calibration": "long-range year check (2038-2050)",
    "build_public_ownership_explorer_year": "report: writes resources/analysis",
    "build_public_ownership_pathway_explorer": "report: writes resources/analysis",
    # D1 (user, 14 Nov 2026): the later years' fleet continuity against the 2030 purchase.
    "verify_public_ownership_acquisition_baseline": "check: the 2030 register against the committed bought-capacity table",
    "verify_fleet_continuity": "long-range: a later year's register against the 2030 purchase",
    "audit_public_ownership_purchase": "check: each acquired row's price against the owners' value (15 Nov 2026)",
    "verify_public_ownership_tariff_invariants": "check: a buying case returns the DPA the kept line pays",
    "verify_public_ownership_bill_invariants": "check: Sizewell C's public-rate return where its RAB is private",
    "report_public_ownership_headline_moves": "check: every headline move against the accepted reference is declared",
    "verify_fes_bill_inputs": "long-range: a year's committed bill inputs against the current model outputs",
}
INPUT_ENV = "PYPSA_GB_INPUT_RESOURCES"
#: The run root's resources folder (the layer's own outputs), for paths a script reads from the bill's assumptions.
PO_ENV = "PYPSA_GB_PO_RESOURCES"
TIMING_ENV = "PYPSA_GB_STAGE_TIMING"
RUN_MANIFEST = "run_manifest.json"
FREEZE_MANIFEST = "freeze_manifest.json"
DEFAULT_GROUP_PREFIXES = ("cp30_2030",)
#: Files a freeze leaves out: the CSV registers of 29 Sep 2026, superseded by the Parquet ones.
FREEZE_SKIP_SUFFIXES = ("_asset_finance_register.csv",)
#: The full defaults directory while a parity cut is in place (make_reduced_config.py build --in-place).
CONFIG_BACKUP = "config/defaults.f20_full.bak"


# ── Paths ─────────────────────────────────────────────────────────────────────


def short_path(path: str | Path) -> str:
    """``path`` absolute with forward slashes; on Windows its 8.3 short form when it has a space in it.

    Snakemake does not quote the paths it puts in a rule's shell command, so a path with a space
    (a live tree in a folder whose name has spaces) must reach it without one.
    """
    text = str(Path(path).absolute())
    if os.name == "nt" and " " in text:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(text, buffer, len(buffer)):
            text = buffer.value
    return text.replace("\\", "/")


def parity_path(path, resources_root=None, po_root=None):
    """A literal ``resources/...`` path redirected for a parity run; anything else, or no roots, unchanged.

    ``resources/public_ownership/...`` (the layer's own outputs) goes to ``{po_root}/public_ownership/...``,
    any other ``resources/...`` (the solved inputs) to ``{resources_root}/...``.  The roots default to the
    run's environment (:data:`PO_ENV`, :data:`INPUT_ENV`); unset (every normal run), the path is returned
    as given.  For paths named in data rather than in the rules (the bill assumptions'
    ``sourced_externals.*.derivation.cost_from.public_account``).
    """
    if resources_root is None:
        resources_root = os.environ.get(INPUT_ENV)
    if po_root is None:
        po_root = os.environ.get(PO_ENV)   # never the input tree: its ownership outputs are not this run's
    if not isinstance(path, str):
        return path
    text = path.replace("\\", "/")
    if text.startswith("resources/public_ownership/"):
        return f"{str(po_root).rstrip('/')}/{text[len('resources/'):]}" if po_root and po_root != "resources" else path
    if text.startswith("resources/") and resources_root and resources_root != "resources":
        return f"{str(resources_root).rstrip('/')}/{text[len('resources/'):]}"
    return path


def _inside(child: str | Path, parent: str | Path) -> bool:
    child, parent = os.path.realpath(str(child)), os.path.realpath(str(parent))
    try:
        return os.path.normcase(os.path.commonpath([child, parent])) == os.path.normcase(parent)
    except ValueError:  # different drives
        return False


def check_roots(input_root, output_root, acknowledged) -> None:
    """Refuse a run that reads another tree unless it is a parity run writing outside that tree.

    The Snakefile calls this when ``resources_path`` is not ``resources``.

    Raises:
        ValueError: ``f20_parity`` is not set, ``po_resources_path`` is missing, or one root holds the other.
    """
    if not acknowledged or not output_root:
        raise ValueError("--config resources_path=... is for an F20 parity run only: it needs po_resources_path=<root> "
                         "and f20_parity=true, and the run must pass --allowed-rules (run_parity.py run does all three)")
    if _inside(output_root, input_root) or _inside(input_root, output_root):
        raise ValueError(f"po_resources_path {output_root} and resources_path {input_root} must not hold one "
                         "another: a parity run writes outside the tree it reads")


def resources_dir(root: str | Path) -> Path:
    """A run or freeze root's ``resources`` folder (the root itself when it is one)."""
    root = Path(root)
    return root / "resources" if (root / "resources").is_dir() else root


# ── Provenance ────────────────────────────────────────────────────────────────


def _git(tree: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_files(tree: Path) -> list[str]:
    """The config files a tree's state names, relative to its ``config`` folder (forward slashes).

    ``config.yaml``, ``scenarios.yaml``, every defaults file (``config_loader.defaults_files``), a
    pre-restructure ``defaults.yaml`` if the tree still has one, and the studies.
    """
    from config.config_loader import defaults_files

    config = Path(tree) / "config"
    names = ["config.yaml", "scenarios.yaml"]
    if (config / "defaults.yaml").exists():
        names.append("defaults.yaml")
    names += [p.relative_to(config).as_posix() for p in defaults_files(config) if p.exists()]
    names += [f"studies/{p.name}" for p in sorted((config / "studies").glob("*.yaml"))]
    return names


def tree_state(tree: Path) -> dict:
    """The code and config a tree holds: commit, a digest of its uncommitted changes, config hashes."""
    diff = _git(tree, "diff", "HEAD")
    untracked = _git(tree, "ls-files", "--others", "--exclude-standard")
    return {
        "tree": str(tree),
        "commit": _git(tree, "rev-parse", "HEAD").strip(),
        "branch": _git(tree, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "uncommitted_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "untracked_files": len([line for line in untracked.splitlines() if line.strip()]),
        "config_sha256": {name: _sha256(tree / "config" / name) for name in config_files(tree)},
        "config_cut": (tree / CONFIG_BACKUP).exists(),
    }


# ── freeze ────────────────────────────────────────────────────────────────────


def freeze(source_tree: Path, dest: Path, prefixes: Sequence[str] = DEFAULT_GROUP_PREFIXES,
           dry_run: bool = False) -> dict:
    """Copy the source tree's ownership outputs for ``prefixes`` to ``dest/resources/public_ownership`` (read-only on source)."""
    source = Path(source_tree) / "resources" / "public_ownership"
    if not source.is_dir():
        raise FileNotFoundError(f"no {source}")
    if _inside(dest, source_tree) or _inside(source_tree, dest):
        raise ValueError("the reference must be outside the tree it is frozen from")
    target = Path(dest) / "resources" / "public_ownership"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty: freeze into a new folder")
    files = []
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if not path.is_file() or not rel.parts[0].startswith(tuple(prefixes)):
            continue
        if path.name.endswith(FREEZE_SKIP_SUFFIXES):
            continue
        stat = path.stat()
        files.append({"path": rel.as_posix(), "bytes": stat.st_size,
                      "mtime": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")})
        if not dry_run:
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
    manifest = {"kind": "freeze", "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
                "source": tree_state(Path(source_tree)), "prefixes": list(prefixes),
                "path_strings": ["resources"], "files": files,
                "bytes": sum(f["bytes"] for f in files)}
    if not dry_run:
        Path(dest).mkdir(parents=True, exist_ok=True)
        snapshot = Path(dest) / "config_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in ("config.yaml", "defaults.yaml", "scenarios.yaml"):
            if (Path(source_tree) / "config" / name).exists():
                shutil.copy2(Path(source_tree) / "config" / name, snapshot / name)
        if (Path(source_tree) / "config" / "defaults").is_dir():
            shutil.copytree(Path(source_tree) / "config" / "defaults", snapshot / "defaults", dirs_exist_ok=True)
        if (Path(source_tree) / "config" / "studies").is_dir():
            shutil.copytree(Path(source_tree) / "config" / "studies", snapshot / "studies", dirs_exist_ok=True)
        (Path(dest) / FREEZE_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ── run ───────────────────────────────────────────────────────────────────────


def snakemake_command(input_root: str, po_root: str, cores: int, mem_mb: int | None, dry_run: bool,
                      target: str = "public_ownership_parity", extra: Sequence[str] = ()) -> list[str]:
    """The Snakemake invocation of a parity run (ownership rules only, inputs read from ``input_root``)."""
    command = [sys.executable, "-m", "snakemake", target, "--cores", str(cores), "--rerun-triggers", "mtime",
               "--printshellcmds", "--allowed-rules", *ALLOWED_RULES]
    if mem_mb:
        command += ["--resources", f"mem_mb={int(mem_mb)}"]
    shell = parity_shell()
    if shell:
        # 14 Oct 2026: the input tree's 8.3 path is 51 characters against "resources" (9), which takes these
        # rules' commands (50 to 62 month files each) past cmd.exe's 8,191-character limit ("The command line
        # is too long").  They run in Windows PowerShell instead, which passes up to 32,767 to python (Git
        # bash is no use: it silently truncates a native program's arguments at about 8,150 characters).
        # Their commands are paths, flags, numbers and scenario names only, which PowerShell passes as given.
        command += ["--set-resources", *(f"{rule}:shell_exec='{shell}'" for rule in LONG_COMMAND_RULES)]
    if dry_run:
        command.append("--dry-run")
    command += [*extra, "--config", f"resources_path={input_root}", f"po_resources_path={po_root}", "f20_parity=true"]
    return command


def parity_shell() -> str | None:
    """On Windows, Windows PowerShell to run :data:`LONG_COMMAND_RULES` in; None elsewhere or if absent."""
    if os.name != "nt":
        return None
    candidate = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return short_path(candidate) if candidate.exists() else None


def parity_env(input_root: str, timing_dir: str | None, po_root: str | None = None) -> dict:
    """The run's environment: the conda env first on PATH, the input root, the run's own root and the timing folder."""
    env = dict(os.environ)
    prefix = Path(sys.executable).parent
    env["PATH"] = os.pathsep.join([str(prefix), str(prefix / "Library" / "bin"), str(prefix / "Scripts"),
                                   env.get("PATH", "")])
    env[INPUT_ENV] = input_root
    # PowerShell (LONG_COMMAND_RULES) runs the user's profile, whose conda hook would otherwise activate the
    # base env ahead of this one on PATH (base has no xarray; seen 14 Oct 2026).
    env["CONDA_AUTO_ACTIVATE_BASE"] = "false"
    if po_root:
        env[PO_ENV] = po_root
    else:
        env.pop(PO_ENV, None)
    if timing_dir:
        env[TIMING_ENV] = timing_dir
    else:
        env.pop(TIMING_ENV, None)
    return env


def run(root: Path, input_resources: Path, *, full: bool = False, cores: int = 4, mem_mb: int | None = 20000,
        dry_run: bool = False, resume: bool = False, timing: bool = True, label: str | None = None,
        extra: Sequence[str] = ()) -> int:
    """Run the ownership layer from this tree into ``root``; return Snakemake's exit code."""
    input_root = short_path(input_resources)
    if not (Path(input_root) / "market").is_dir():
        raise FileNotFoundError(f"{input_root} has no market/: pass the input tree's resources folder")
    root = Path(root)
    check_roots(input_root, str(root.absolute()), True)          # before anything is created
    root.mkdir(parents=True, exist_ok=True)
    po_root = short_path(root) + "/resources"
    if " " in po_root or " " in input_root:
        raise ValueError(f"{po_root} or {input_root} has a space in it (no 8.3 name): Snakemake's shell commands "
                         "would split it")
    check_roots(input_root, po_root, True)
    if _inside(root, PROJECT_ROOT) and not dry_run:
        print(f"note: {root} is inside the tree; keep it out of git (a 'resources' folder is gitignored)")
    cut = (PROJECT_ROOT / CONFIG_BACKUP).exists()
    if full and cut:
        raise RuntimeError("the config is cut: run make_reduced_config.py restore before a full run")
    if not full and not cut:
        raise RuntimeError("the config is the full one: run make_reduced_config.py build --in-place --input-resources "
                           f"{input_root} first (or pass --full)")
    outputs = Path(po_root) / "public_ownership"
    if outputs.exists() and any(outputs.iterdir()) and not (resume or dry_run):
        raise FileExistsError(f"{outputs} is not empty: run into a new root (or --resume)")
    timing_dir = str(root / "stage_timing") if timing else None
    command = snakemake_command(input_root, po_root, cores, mem_mb, dry_run, extra=extra)
    manifest = {"kind": "run", "label": label or root.name, "full_config": full, "dry_run": dry_run,
                "started": dt.datetime.now().isoformat(timespec="seconds"), "tree": tree_state(PROJECT_ROOT),
                "input_resources": input_root, "input_resources_long": str(Path(input_resources).resolve()),
                "po_resources_path": po_root, "command": command,
                "path_strings": [po_root, input_root, str(Path(input_resources).resolve())]}
    print(" ".join(f'"{c}"' if " " in c else c for c in command))
    log = root / ("snakemake_dry_run.log" if dry_run else "snakemake.log")
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(command, cwd=PROJECT_ROOT, env=parity_env(input_root, timing_dir, po_root),
                              stdout=handle, stderr=subprocess.STDOUT).returncode
    manifest.update({"finished": dt.datetime.now().isoformat(timespec="seconds"), "exit_code": code, "log": str(log)})
    if not dry_run:
        (root / RUN_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Written on a failed run too (14 Oct 2026: the calibration check fails on today's register); the
        # manifest's exit code says whether every stage ran.
        if timing and any(Path(timing_dir).glob("*.csv")):
            from scripts.public_ownership import stage_timing

            table = stage_timing.summarise(timing_dir)
            (root / "stage_times.md").write_text(stage_timing.to_markdown(table), encoding="utf-8")
            table.to_csv(root / "stage_times.csv", index=False)
    print(f"exit {code}; log {log}")
    return code


# ── compare ───────────────────────────────────────────────────────────────────


def path_replacements(*roots: Path) -> list[tuple[str, str]]:
    """Each root's own path strings (its manifest's) mapped to ``resources``, in both slash forms, longest first."""
    strings = set()
    for root in roots:
        for name in (RUN_MANIFEST, FREEZE_MANIFEST):
            path = Path(root) / name
            if path.exists():
                strings.update(json.loads(path.read_text(encoding="utf-8")).get("path_strings") or [])
    strings.discard("resources")
    pairs = set()
    for text in strings:
        for form in (text.replace("\\", "/"), text.replace("/", "\\")):
            pairs.add((form.rstrip("/\\"), "resources"))
    return sorted(pairs, key=lambda pair: -len(pair[0]))


def compare(reference: Path, candidate: Path, extra: Sequence[str] = ()) -> int:
    """:func:`compare_outputs.main` over two roots, their path strings normalised."""
    from scripts.public_ownership.compare_outputs import main as compare_main

    replace = [f"{old}={new}" for old, new in path_replacements(reference, candidate)]
    argv = [str(resources_dir(reference)), str(resources_dir(candidate))]
    if replace:
        argv += ["--replace", *replace]
    return compare_main([*argv, *extra])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="copy a tree's 2030 ownership outputs to a reference folder")
    f.add_argument("--from", dest="source", required=True, help="the tree (its root, holding resources/)")
    f.add_argument("--to", dest="dest", required=True, help="the reference folder (outside both trees)")
    f.add_argument("--prefixes", nargs="*", default=list(DEFAULT_GROUP_PREFIXES),
                   help="run-group prefixes to copy (default cp30_2030: every 2030 group and twin)")
    f.add_argument("--dry-run", action="store_true", help="list, do not copy")
    r = sub.add_parser("run", help="run the ownership layer from this tree into ROOT")
    r.add_argument("root", help="the run root (outputs go to ROOT/resources/public_ownership)")
    r.add_argument("--input-resources", required=True, help="the resources folder holding the solved inputs")
    r.add_argument("--full", action="store_true", help="the full case list (the config must not be cut)")
    r.add_argument("--cores", type=int, default=4)
    r.add_argument("--mem-mb", type=int, default=20000, help="Snakemake's mem_mb budget (0: none)")
    r.add_argument("--dry-run", action="store_true", help="Snakemake --dry-run (writes nothing but the log)")
    r.add_argument("--resume", action="store_true", help="allow a root that already has outputs")
    r.add_argument("--no-timing", action="store_true", help="leave stage timing off")
    r.add_argument("--label", default=None)
    r.add_argument("--snakemake-args", nargs=argparse.REMAINDER, default=[],
                   help="anything after this goes to Snakemake (e.g. --forcerun RULE)")
    c = sub.add_parser("compare", help="compare two roots (compare_outputs.py with path strings normalised)")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("compare_args", nargs=argparse.REMAINDER, help="passed to compare_outputs.py")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        manifest = freeze(Path(args.source), Path(args.dest), args.prefixes, args.dry_run)
        print(f"{len(manifest['files'])} files, {manifest['bytes'] / 1e9:.2f} GB"
              + (" (dry run)" if args.dry_run else f" -> {args.dest}"))
        return 0
    if args.command == "run":
        return run(Path(args.root), Path(args.input_resources), full=args.full, cores=args.cores,
                   mem_mb=args.mem_mb or None, dry_run=args.dry_run, resume=args.resume, timing=not args.no_timing,
                   label=args.label, extra=args.snakemake_args)
    return compare(Path(args.reference), Path(args.candidate), args.compare_args)


if __name__ == "__main__":
    sys.exit(main())
