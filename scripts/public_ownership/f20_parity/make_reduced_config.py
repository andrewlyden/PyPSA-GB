"""The F20 parity run's reduced public-ownership config, cut from a tree's own defaults (13 Oct 2026).

The full 2030 case list is 5,334 cases, 5,216 of them grid positions.  The parity
run (``README.md`` beside this file) keeps every non-grid case and a few axes of
each grid (``reduced_config.yaml``), so one run exercises every lever value,
every private comparator, both balancing groups and the institution case in
minutes rather than hours.  The cut edits only the blocks it changes, as text,
so every other byte of the defaults (``config/defaults/``, comments included) is kept:

* ``public_ownership.sensitivities.toggle_grids``: the kept axes, and ``max_cases``
  (``base/public_ownership/sensitivities.yaml``);
* ``public_ownership.sensitivities.named``: only when ``named_keep`` is a list (the same file);
* a price-level twin's ``case_selection.cases``: any case the cut drops is
  dropped from it too (a twin may name grid cases; ``base/public_ownership/reference_run_groups.yaml``);
* with ``--input-resources ROOT``: every ``resources/...`` value in every defaults file
  (``config_loader.defaults_files``) becomes ``ROOT/...``, so the scripts read config-named
  inputs (the assumptions snapshot, the institution's marginal costs) from the tree that solved them.

Usage (from the parity tree's root)::

    python scripts/public_ownership/f20_parity/make_reduced_config.py build --in-place \\
        --input-resources <main working tree>/resources
    python scripts/public_ownership/f20_parity/make_reduced_config.py report     # case counts and coverage
    python scripts/public_ownership/f20_parity/make_reduced_config.py restore    # put the full config back

``build --in-place`` keeps the whole original defaults directory as ``config/defaults.f20_full.bak/``
and refuses to run while that backup exists, so the full config is never overwritten
by a second cut.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SPEC = Path(__file__).with_name("reduced_config.yaml")
#: The config directory; the defaults are its ``defaults/`` directory (``config_loader.defaults_dir``).
DEFAULT_CONFIG = Path("config")
BACKUP_SUFFIX = ".f20_full.bak"
SENSITIVITIES = ("public_ownership", "sensitivities")
GRID_KINDS = ("grids", "bound_grids")
#: The defaults files the cut edits, relative to ``config/defaults`` (forward slashes). The first holds
#: ``public_ownership.sensitivities``, the second ``public_ownership.reference_run_groups``.
SENSITIVITIES_FILE = "base/public_ownership/sensitivities.yaml"
RUN_GROUPS_FILE = "base/public_ownership/reference_run_groups.yaml"


# ── Text blocks ───────────────────────────────────────────────────────────────


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_code(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def find_block(lines: Sequence[str], path: Sequence[str]) -> tuple[int, int, int]:
    """``(key line, end, indent)`` of the mapping entry at ``path`` (``end`` exclusive, trailing comments excluded).

    Raises:
        KeyError: A key on the path is not a direct child of its parent.
    """
    lo, hi, parent_indent = 0, len(lines), -1
    start = indent = -1
    for key in path:
        child_indent = next((_indent(lines[i]) for i in range(lo, hi)
                             if _is_code(lines[i]) and _indent(lines[i]) > parent_indent), None)
        pattern = re.compile(rf"^ {{{child_indent}}}{re.escape(key)}:(\s|$)") if child_indent is not None else None
        start = next((i for i in range(lo, hi) if pattern is not None and pattern.match(lines[i])), -1)
        if start < 0:
            raise KeyError(f"{'.'.join(path)}: no key {key!r} at this level")
        indent = child_indent
        end = next((i for i in range(start + 1, hi) if _is_code(lines[i]) and _indent(lines[i]) <= indent), hi)
        while end > start + 1 and not (_is_code(lines[end - 1]) or
                                       (lines[end - 1].strip() and _indent(lines[end - 1]) > indent)):
            end -= 1
        lo, hi, parent_indent = start + 1, end, indent
    return start, hi, indent


def _dump(key: str, value, indent: int) -> list[str]:
    """``key: value`` as YAML lines at ``indent`` (flow style for the innermost collections)."""
    body = yaml.safe_dump({key: value}, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)
    return [" " * indent + line for line in body.splitlines()]


def replace_block(text: str, path: Sequence[str], value) -> str:
    """``text`` with the entry at ``path`` replaced by ``value`` (line endings kept)."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    start, end, indent = find_block(lines, path)
    lines[start:end] = _dump(path[-1], value, indent)
    return newline.join(lines)


_RESOURCE_VALUE = re.compile(r"""(?P<lead>(?:(?::|-)\s+|[\[,]\s*)["']?)resources/""")


def rewrite_resource_paths(text: str, root: str) -> tuple[str, int]:
    """Every YAML value starting ``resources/`` pointed at ``root`` instead (comment lines untouched)."""
    root = str(root).replace("\\", "/").rstrip("/")
    count = 0
    out = []
    for line in text.split("\n"):
        if _is_code(line):
            code, hash_, comment = line.partition(" #")
            code, n = _RESOURCE_VALUE.subn(lambda m: f"{m.group('lead')}{root}/", code)
            count += n
            line = code + hash_ + comment
        out.append(line)
    return "\n".join(out), count


# ── The defaults as texts ─────────────────────────────────────────────────────


def defaults_backup(config_dir) -> Path:
    """Where ``build --in-place`` keeps the full defaults: ``<config_dir>/defaults.f20_full.bak``."""
    from config.config_loader import defaults_dir

    root = defaults_dir(Path(config_dir))
    return root.with_name(root.name + BACKUP_SUFFIX)


def read_defaults_texts(defaults_root) -> dict[str, str]:
    """Every defaults file under ``defaults_root`` (the set ``config_loader.defaults_files`` lists) as text,
    keyed by its path relative to ``defaults_root`` (forward slashes)."""
    from config.config_loader import BASE_DIR, BASE_FILE, DEFAULTS_DIR, LAYERS, defaults_files

    root = Path(defaults_root)
    # defaults_files() takes the config directory, so it lists a root named "defaults"; a backup or --out
    # directory (another name) is listed by the same pattern.
    if root.name == DEFAULTS_DIR:
        files = defaults_files(root.parent)
    else:
        files = [root / BASE_FILE] + sorted((root / BASE_DIR).rglob("*.yaml"))
        for layer in LAYERS:
            files += sorted((root / layer).glob("*.yaml"))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"defaults files missing: {missing}")
    return {p.relative_to(root).as_posix(): p.read_bytes().decode("utf-8") for p in files}


def assemble(texts: Mapping[str, str]) -> dict:
    """The base defaults dict of ``texts`` as ``config_loader.load_defaults`` builds it (file path = key path)."""
    from config.config_loader import BASE_DIR, BASE_FILE

    tree: dict = {}
    names = [BASE_FILE] + sorted(n for n in texts if n.startswith(BASE_DIR + "/"))
    for name in names:
        value = yaml.safe_load(texts[name]) or {}
        path = () if name == BASE_FILE else tuple(Path(name[len(BASE_DIR) + 1:]).with_suffix("").parts)
        node = tree
        for part in path:
            node = node.setdefault(part, {})
        clash = sorted(set(node) & set(value))
        if clash:
            raise ValueError(f"{name}: {'.'.join(path + (clash[0],))} is defined in two defaults files")
        node.update(value)
    return tree


def write_defaults_texts(defaults_root, texts: Mapping[str, str], only: Sequence[str] | None = None) -> list[str]:
    """Write ``texts`` (all, or the names in ``only``) under ``defaults_root``; returns the names written."""
    names = list(texts) if only is None else list(only)
    for name in names:
        path = Path(defaults_root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(texts[name].encode("utf-8"))
    return names


# ── The cut ───────────────────────────────────────────────────────────────────


def _axis_id(entry: Mapping) -> str:
    return str(entry.get("token") if "token" in entry else entry.get("selector"))


def reduce_grids(toggle_grids: Mapping, spec: Mapping) -> tuple[dict, list[str]]:
    """The toggle-grid block with only the axes ``spec`` names (and the notes on what it skipped)."""
    out = copy.deepcopy(dict(toggle_grids))
    notes: list[str] = []
    wanted = spec.get("toggle_grids") or {}
    if "max_cases" in wanted:
        out["max_cases"] = wanted["max_cases"]
    for kind in GRID_KINDS:
        grids = dict(out.get(kind) or {})
        keep_by_parent = wanted.get(kind) or {}
        reduced = {}
        for parent, axes in grids.items():
            keep = [str(a) for a in (keep_by_parent.get(parent) or [])]
            if not keep:
                notes.append(f"{kind}.{parent}: no axis kept, grid dropped")
                continue
            kept = [dict(axis) for axis in axes if _axis_id(axis) in keep]
            selectors = {_axis_id(a) for a in kept if "selector" in a}
            for axis in list(kept):
                when = axis.get("when") or {}
                if any(name not in selectors for name in when):
                    kept.remove(axis)
                    notes.append(f"{kind}.{parent}: {_axis_id(axis)} dropped (its selector {sorted(when)} is not kept)")
            missing = sorted(set(keep) - {_axis_id(a) for a in axes})
            if missing:
                notes.append(f"{kind}.{parent}: the config has no axes {missing}; skipped")
            if kept:
                reduced[parent] = kept
        for parent in sorted(set(keep_by_parent) - set(grids)):
            notes.append(f"{kind}: the config has no grid {parent!r}; skipped")
        if kind in out or reduced:
            out[kind] = reduced
    return out, notes


def reduce_named(named: Mapping, keep: Sequence[str] | None) -> dict:
    """The named sensitivities ``keep`` lists, with every case they extend (None keeps all)."""
    if keep is None:
        return dict(named)
    wanted = set(map(str, keep))
    unknown = sorted(wanted - set(named))
    if unknown:
        raise ValueError(f"named_keep names cases the config does not have: {unknown}")
    frontier = list(wanted)
    while frontier:
        parent = (named.get(frontier.pop()) or {}).get("extends")
        if parent is not None and parent not in wanted:
            wanted.add(parent)
            frontier.append(parent)
    return {name: entry for name, entry in named.items() if name in wanted}


def prune_case_selections(po: Mapping, kept: set) -> dict[str, dict]:
    """``{group: selection}`` for each group whose ``case_selection.cases`` names a case the cut drops."""
    changed = {}
    for group, block in (po.get("reference_run_groups") or {}).items():
        selection = (block or {}).get("case_selection")
        if not isinstance(selection, Mapping) or not selection.get("cases"):
            continue
        cases = [str(c) for c in selection["cases"]]
        remaining = [c for c in cases if c in kept]
        if remaining != cases:
            changed[group] = {**selection, "cases": remaining}
    return changed


def build(texts: Mapping[str, str], spec: Mapping, input_resources: str | None = None) -> tuple[dict[str, str], dict]:
    """The reduced defaults and a record of what the cut changed.

    Args:
        texts: every defaults file's text, keyed as :func:`read_defaults_texts` keys them.
        spec: the cut (``reduced_config.yaml``).
        input_resources: point every ``resources/...`` value of every file at this root.

    Returns:
        Every file's text after the cut (same keys) and the record; ``record["changed_files"]`` names the files
        whose text differs.
    """
    from scripts.public_ownership.reference_runs import configured_case_specs

    out = dict(texts)
    po = assemble(texts).get("public_ownership") or {}
    sensitivities = po.get("sensitivities") or {}
    record: dict = {"notes": []}
    grids, notes = reduce_grids(sensitivities.get("toggle_grids") or {}, spec)
    record["notes"] += notes
    out[SENSITIVITIES_FILE] = replace_block(out[SENSITIVITIES_FILE], ("toggle_grids",), grids)
    if spec.get("named_keep") is not None:
        named = reduce_named(sensitivities.get("named") or {}, spec["named_keep"])
        out[SENSITIVITIES_FILE] = replace_block(out[SENSITIVITIES_FILE], ("named",), named)
        record["named_kept"] = list(named)
    reduced_po = assemble(out).get("public_ownership") or {}
    # A twin's case_selection is applied after the global list, so the global list decides what it may name.
    kept = {s.case_id for s in configured_case_specs({**reduced_po, "reference_run_groups": {}})}
    for group, selection in prune_case_selections(reduced_po, kept).items():
        out[RUN_GROUPS_FILE] = replace_block(out[RUN_GROUPS_FILE], (group, "case_selection"), selection)
        record["notes"].append(f"reference_run_groups.{group}.case_selection: cases the cut drops removed")
    if input_resources:
        count = 0
        for name, text in out.items():
            newline = "\r\n" if "\r\n" in text else "\n"
            rewritten, n = rewrite_resource_paths(text.replace("\r\n", "\n"), input_resources)
            out[name] = rewritten.replace("\n", newline)
            count += n
        record["resource_paths_rewritten"] = count
    record["changed_files"] = sorted(name for name in out if out[name] != texts.get(name))
    return out, record


# ── Coverage ──────────────────────────────────────────────────────────────────


def coverage(full_po: Mapping, reduced_po: Mapping) -> dict:
    """What the reduced list covers of the full one: cases by kind, lever values, comparators, groups."""
    from scripts.public_ownership.levers import comparator_for_settings
    from scripts.public_ownership.reference_runs import (
        case_kind, configured_case_specs, group_market_cases, group_tariff_cases, institution_cases,
        market_bill_cases, tariff_cases,
    )

    def describe(po: Mapping) -> dict:
        specs = configured_case_specs(po)
        by_case = {s.case_id: dict(s.settings) for s in specs}
        groups = {}
        for group in po.get("reference_run_groups") or {}:
            try:
                groups[group] = {"market": len(group_market_cases(po, group)),
                                 "tariff": len(group_tariff_cases(po, group)),
                                 "registered": len(configured_case_specs(po, group))}
            except Exception as error:  # noqa: BLE001  (a group the config cannot list is reported, not fatal)
                groups[group] = {"error": str(error)}
        return {
            "cases": len(specs),
            "kinds": dict(collections.Counter(case_kind(s) for s in specs)),
            "market_bill_cases": len(market_bill_cases(po)),
            "tariff_cases": len(tariff_cases(po)),
            "institution_cases": institution_cases(po),
            "register_keys": len({s.register_key() for s in specs}),
            "lever_values": {(k, v) for s in specs for k, v in s.settings.items()},
            "comparators": set(comparator_for_settings(c, by_case) for c in by_case),
            "groups": groups,
        }

    full, reduced = describe(full_po), describe(reduced_po)
    return {
        "full": {k: v for k, v in full.items() if k not in ("lever_values", "comparators")},
        "reduced": {k: v for k, v in reduced.items() if k not in ("lever_values", "comparators")},
        "lever_values_full": len(full["lever_values"]),
        "lever_values_missing": sorted(map(list, full["lever_values"] - reduced["lever_values"])),
        "comparators_full": sorted(full["comparators"]),
        "comparators_missing": sorted(full["comparators"] - reduced["comparators"]),
    }


# ── Command line ──────────────────────────────────────────────────────────────


def _load_spec(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the reduced config")
    b.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="the config directory whose defaults/ is cut (default config)")
    b.add_argument("--spec", default=str(DEFAULT_SPEC))
    b.add_argument("--out", default=None,
                   help="write the reduced defaults directory here (default: print the record only, unless --in-place)")
    b.add_argument("--in-place", action="store_true",
                   help=f"rewrite --config's defaults, keeping the whole directory as defaults{BACKUP_SUFFIX}")
    b.add_argument("--input-resources", default=None,
                   help="point every resources/... config value at this root (the tree whose solved outputs are read)")
    r = sub.add_parser("report", help="case counts and coverage of the reduced against the full config")
    r.add_argument("--config", default=str(DEFAULT_CONFIG))
    r.add_argument("--full", default=None, help=f"the full defaults directory (default: defaults{BACKUP_SUFFIX} "
                                                "if it exists, else --config's defaults cut in memory)")
    r.add_argument("--spec", default=str(DEFAULT_SPEC))
    r.add_argument("--out", default=None, help="also write the report as JSON")
    s = sub.add_parser("restore", help=f"put defaults{BACKUP_SUFFIX} back over --config's defaults")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    from config.config_loader import defaults_dir

    defaults_root = defaults_dir(Path(args.config))
    backup = defaults_backup(args.config)
    if args.command == "restore":
        if not backup.exists():
            print(f"no {backup}: nothing to restore", file=sys.stderr)
            return 1
        if defaults_root.exists():
            shutil.rmtree(defaults_root)
        shutil.move(str(backup), str(defaults_root))
        print(f"restored {defaults_root} from {backup}")
        return 0
    if args.command == "build":
        if args.in_place and backup.exists():
            print(f"{backup} exists: the config is already cut (run restore first)", file=sys.stderr)
            return 1
        texts = read_defaults_texts(defaults_root)
        from scripts.public_ownership.f20_parity.run_parity import short_path

        # Snakemake does not quote paths in shell commands: the live tree's has spaces, its 8.3 form none.
        input_resources = short_path(args.input_resources) if args.input_resources else None
        if input_resources and " " in input_resources:
            print(f"{input_resources} has a space in it and no 8.3 name: Snakemake would split it", file=sys.stderr)
            return 1
        out, record = build(texts, _load_spec(args.spec), input_resources)
        from scripts.public_ownership.reference_runs import validate_public_ownership_config

        validate_public_ownership_config(assemble(out).get("public_ownership") or {})
        if args.in_place:
            shutil.copytree(defaults_root, backup)
            write_defaults_texts(defaults_root, out, record["changed_files"])
            record["written"] = str(defaults_root)
            record["backup"] = str(backup)
        elif args.out:
            write_defaults_texts(args.out, out)
            record["written"] = args.out
        print(json.dumps(record, indent=2))
        return 0
    # report
    if args.full:
        full_texts = read_defaults_texts(args.full)
        reduced_texts = read_defaults_texts(defaults_root)
    elif backup.exists():
        full_texts = read_defaults_texts(backup)
        reduced_texts = read_defaults_texts(defaults_root)
    else:
        full_texts = read_defaults_texts(defaults_root)
        reduced_texts, _ = build(full_texts, _load_spec(args.spec))
    report = coverage(assemble(full_texts).get("public_ownership") or {},
                      assemble(reduced_texts).get("public_ownership") or {})
    text = json.dumps(report, indent=2, default=list)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
