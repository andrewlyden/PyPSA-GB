"""Compare two trees of bill and ownership outputs, file by file.

A speed-up of the bill and ownership layer must leave every
output the bill and ownership layer writes unchanged: the same arithmetic, so
bit-identical numbers.  This finds every CSV, Parquet and JSON file under a
reference root and a candidate root (each a ``resources/``-like tree, or any
subset of one), matches them by relative path and compares them:

* **CSV and Parquet** as tables.  Rows are matched on key columns, not on their
  order (``case_id``, ``component_id``, ``asset_id`` and the like, found by name;
  ``--keys`` overrides), with an ordinal inside any key that repeats.  Numbers are
  parsed exactly (``float_precision="round_trip"``), so ``0.1`` and
  ``0.10000000000000001`` are equal and a number is never compared as text;
  text is compared as text.  Columns added or removed are reported.
* **The asset finance register** (``*_asset_finance_register.parquet``) through
  :func:`register_io.read_register`, a batch of cases at a time, so what is
  compared is what every stage reads, whatever the file layout underneath.
* **JSON** as a tree: keys added or removed, list lengths, every number and
  string.  A list of records with a unique ``case_id`` (or ``component_id``,
  ``id``, ``name``, ``flow_id``) is matched by that id.

The comparison is exact unless ``--atol``/``--rtol`` are given (a number passes
when ``|a - b| <= atol + rtol * |a|``).  Each file reports its largest absolute
and relative difference, the cells or values that differ (with examples), the
rows or keys only on one side and the columns added or removed.  The exit code
is 1 when anything differs beyond the tolerance (a file on one side only
included, unless ``--missing-ok``), 0 otherwise.

Usage::

    python scripts/public_ownership/compare_outputs.py REFERENCE CANDIDATE \\
        [--include 'public_ownership/cp30_2030*'] [--exclude '*_asset_finance_register.csv'] \\
        [--atol 0 --rtol 0] [--replace 'f20_parity_runs/new/resources=resources'] \\
        [--report diff.csv] [--json diff.json]

Paths a script writes into its output (``"source": <register path>``) differ
between roots by construction; ``--replace OLD=NEW`` rewrites text on both sides
before comparing, and ``--ignore-keys`` / ``--ignore-columns`` (regular
expressions on a JSON key path or a column name) leave a field out.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUFFIXES = (".csv", ".parquet", ".json")
REGISTER_SUFFIX = "_asset_finance_register.parquet"
#: Columns that identify a row, in the order they are used when present (``--keys`` overrides).
KEY_COLUMNS = (
    "case_id", "run_group", "scenario", "component_type", "asset_id", "generator", "component_id", "strike_key",
    "route", "group", "consumer_bill_group", "ledger", "step", "year", "finance_technology", "finance_vintage",
    "support_type", "treatment", "sector", "scope_group", "carrier", "stream", "flow_id", "stage", "month",
)
#: JSON record ids, for matching a list of records by id instead of position.
RECORD_IDS = ("case_id", "component_id", "id", "name", "flow_id", "asset_id")
#: Fields every run writes afresh (a write time): never compared unless --ignore-keys is given.
DEFAULT_IGNORE_KEYS = r"(^|\.)(written_utc|written_at|generated_at)$"
DEFAULT_EXCLUDE = ("stage_timing/*",)
ORDINAL = "__row_in_key__"


# ── Results ───────────────────────────────────────────────────────────────────


class Diff:
    """What one file's comparison found."""

    def __init__(self, path: str, kind: str, examples: int):
        self.path, self.kind, self.max_examples = path, kind, examples
        self.status = "equal"
        self.rows_reference = self.rows_candidate = None
        self.rows_only_reference = self.rows_only_candidate = 0
        self.columns_added: list[str] = []
        self.columns_removed: list[str] = []
        self.values_compared = 0
        self.values_differ = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.worst = ""
        self.examples: list[str] = []
        self.error = ""
        self.seconds = 0.0

    def example(self, text: str) -> None:
        if len(self.examples) < self.max_examples:
            self.examples.append(text)

    def numbers(self, where: str, a: np.ndarray, b: np.ndarray, atol: float, rtol: float,
                labels: Sequence[str] | None = None) -> None:
        """Compare two float arrays (NaN equals NaN; an infinity equals the same infinity)."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        self.values_compared += a.size
        both_nan = np.isnan(a) & np.isnan(b)
        with np.errstate(invalid="ignore", over="ignore"):
            same = (a == b) | both_nan
            diff = np.abs(a - b)
            if atol or rtol:
                same |= diff <= atol + rtol * np.abs(a)
        bad = ~same
        if not bad.any():
            return
        self.values_differ += int(bad.sum())
        finite = bad & np.isfinite(diff)
        if (bad & ~finite).any():
            self._worst(where, math.inf, math.inf)
        if finite.any():
            d = diff[finite]
            scale = np.abs(a[finite])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(scale > 0, d / scale, np.where(d > 0, np.inf, 0.0))
            self._worst(where, float(d.max()), float(rel.max()))
        for i in np.flatnonzero(bad)[: self.max_examples]:
            label = labels[i] if labels is not None else str(i)
            self.example(f"{where} [{label}]: {a[i]!r} -> {b[i]!r}")

    def text(self, where: str, a: Sequence, b: Sequence, labels: Sequence[str] | None = None) -> None:
        a = np.asarray(a, dtype=object)
        b = np.asarray(b, dtype=object)
        self.values_compared += a.size
        bad = a != b
        if not bad.any():
            return
        self.values_differ += int(bad.sum())
        self._worst(where, math.nan, math.nan)
        for i in np.flatnonzero(bad)[: self.max_examples]:
            label = labels[i] if labels is not None else str(i)
            self.example(f"{where} [{label}]: {_short(a[i])} -> {_short(b[i])}")

    def _worst(self, where: str, abs_diff: float, rel_diff: float) -> None:
        if not self.worst:
            self.worst = where
        if not math.isnan(abs_diff) and abs_diff > self.max_abs:
            self.max_abs, self.worst = abs_diff, where
        if not math.isnan(rel_diff) and rel_diff > self.max_rel:
            self.max_rel = rel_diff

    def finish(self) -> "Diff":
        if self.status == "equal" and (self.values_differ or self.rows_only_reference or self.rows_only_candidate
                                       or self.columns_added or self.columns_removed):
            self.status = "differs"
        return self

    def record(self) -> dict:
        return {
            "path": self.path, "kind": self.kind, "status": self.status,
            "rows_reference": self.rows_reference, "rows_candidate": self.rows_candidate,
            "rows_only_reference": self.rows_only_reference, "rows_only_candidate": self.rows_only_candidate,
            "columns_added": ";".join(self.columns_added), "columns_removed": ";".join(self.columns_removed),
            "values_compared": self.values_compared, "values_differ": self.values_differ,
            "max_abs_diff": self.max_abs, "max_rel_diff": self.max_rel, "worst": self.worst,
            "examples": " | ".join(self.examples), "error": self.error, "seconds": round(self.seconds, 3),
        }


def _short(value, width: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


# ── Options ───────────────────────────────────────────────────────────────────


class Options:
    """The comparison settings shared by every file."""

    def __init__(self, atol: float = 0.0, rtol: float = 0.0, keys: Mapping[str, Sequence[str]] | None = None,
                 ignore_columns: str | None = None, ignore_keys: str | None = DEFAULT_IGNORE_KEYS,
                 replace: Sequence[tuple[str, str]] = (), examples: int = 5, register_batch_cases: int = 64):
        self.atol, self.rtol = float(atol), float(rtol)
        self.keys = dict(keys or {})
        self.ignore_columns = re.compile(ignore_columns) if ignore_columns else None
        self.ignore_keys = re.compile(ignore_keys) if ignore_keys else None
        self.replace = [(str(old), str(new)) for old, new in replace]
        self.examples = int(examples)
        self.register_batch_cases = int(register_batch_cases)

    def keys_for(self, path: str) -> list[str] | None:
        for pattern, columns in self.keys.items():
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
                return list(columns)
        return None

    def normalise(self, text: str) -> str:
        for old, new in self.replace:
            text = text.replace(old, new)
        return text


# ── Tables ────────────────────────────────────────────────────────────────────


def read_csv(path: Path) -> pd.DataFrame:
    """A CSV with numbers parsed exactly as written (round trip), so formatting never shows as a difference."""
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _text_values(series: pd.Series, options: Options) -> np.ndarray:
    """A column as comparable text: missing as '<NA>', integral floats without '.0', ``--replace`` applied."""
    missing = pd.isna(series).to_numpy()
    if _is_numeric(series):
        values = series.to_numpy(dtype=float)
        out = np.array([str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(float(v)) if not math.isnan(v)
                        else "<NA>" for v in values], dtype=object)
        return out
    text = series.astype(object).map(_scalar_text)
    for old, new in options.replace:
        text = text.str.replace(old, new, regex=False)
    out = text.to_numpy(dtype=object)
    out[missing] = "<NA>"
    return out


def _scalar_text(value) -> str:
    """One cell as text: an integral number without '.0', so 2030 and 2030.0 read alike in a text column."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "<NA>"
        return str(int(value)) if float(value).is_integer() and abs(value) < 1e15 else repr(float(value))
    return str(value)


def key_columns(reference: pd.DataFrame, candidate: pd.DataFrame, path: str, options: Options) -> list[str]:
    """The columns rows are matched on: ``--keys``, else the known identifier columns, else every text column."""
    common = [c for c in reference.columns if c in candidate.columns]
    given = options.keys_for(path)
    if given is not None:
        missing = [c for c in given if c not in common]
        if missing:
            raise ValueError(f"--keys columns {missing} are not in both files")
        return given
    known = [c for c in KEY_COLUMNS if c in common]
    if known:
        return known
    return [c for c in common if not _is_numeric(reference[c]) and not _is_numeric(candidate[c])]


def _keyed(frame: pd.DataFrame, keys: Sequence[str], options: Options) -> pd.DataFrame:
    """``frame`` indexed by its key columns as text, with an ordinal inside a repeated key (file order)."""
    out = frame.copy()
    key_text = {k: _text_values(out[k], options) for k in keys}
    index_frame = pd.DataFrame(key_text, index=out.index) if keys else pd.DataFrame(index=out.index)
    ordinal = (index_frame.groupby(list(keys), sort=False, dropna=False).cumcount() if keys
               else pd.Series(np.arange(len(out)), index=out.index))
    index_frame[ORDINAL] = ordinal.to_numpy()
    out.index = pd.MultiIndex.from_frame(index_frame)
    return out.drop(columns=list(keys))


def _labels(index: pd.Index) -> list[str]:
    return ["|".join(str(part) for part in (entry if isinstance(entry, tuple) else (entry,))) for entry in index]


def compare_frames(reference: pd.DataFrame, candidate: pd.DataFrame, diff: Diff, options: Options,
                   keys: Sequence[str] | None = None, where: str = "") -> None:
    """Match rows on their keys and compare every common column (numbers as numbers, text as text)."""
    if options.ignore_columns is not None:
        reference = reference[[c for c in reference.columns if not options.ignore_columns.search(str(c))]]
        candidate = candidate[[c for c in candidate.columns if not options.ignore_columns.search(str(c))]]
    added = [c for c in candidate.columns if c not in reference.columns]
    removed = [c for c in reference.columns if c not in candidate.columns]
    for column in added:
        if column not in diff.columns_added:
            diff.columns_added.append(column)
    for column in removed:
        if column not in diff.columns_removed:
            diff.columns_removed.append(column)
    keys = list(keys if keys is not None else key_columns(reference, candidate, diff.path, options))
    ref = _keyed(reference, keys, options)
    cand = _keyed(candidate, keys, options)
    only_ref = ref.index.difference(cand.index, sort=False)
    only_cand = cand.index.difference(ref.index, sort=False)
    diff.rows_only_reference += len(only_ref)
    diff.rows_only_candidate += len(only_cand)
    for label in _labels(only_ref)[: diff.max_examples]:
        diff.example(f"{where}row only in reference: {label}")
    for label in _labels(only_cand)[: diff.max_examples]:
        diff.example(f"{where}row only in candidate: {label}")
    common = ref.index.intersection(cand.index, sort=False)
    if not len(common):
        return
    ref = ref.loc[common]
    cand = cand.loc[common]
    labels = None
    for column in [c for c in ref.columns if c in cand.columns]:
        a, b = ref[column], cand[column]
        if _is_numeric(a) and _is_numeric(b):
            if labels is None and diff.values_differ < diff.max_examples:
                labels = _labels(common)
            before = diff.values_differ
            diff.numbers(f"{where}{column}", a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                         options.atol, options.rtol, labels if before < diff.max_examples else None)
            continue
        # Mixed or text: cells that are numbers on both sides compare as numbers, the rest as text.
        an = pd.to_numeric(a, errors="coerce")
        bn = pd.to_numeric(b, errors="coerce")
        a_missing, b_missing = a.isna().to_numpy(), b.isna().to_numpy()
        numeric = (an.notna().to_numpy() | a_missing) & (bn.notna().to_numpy() | b_missing) & ~(a_missing & b_missing)
        numeric &= ~(_bool_like(a) | _bool_like(b))
        if labels is None:
            labels = _labels(common)
        if numeric.any():
            diff.numbers(f"{where}{column}", an.to_numpy(dtype=float)[numeric], bn.to_numpy(dtype=float)[numeric],
                         options.atol, options.rtol, [labels[i] for i in np.flatnonzero(numeric)])
        rest = ~numeric
        if rest.any():
            diff.text(f"{where}{column}", _text_values(a[rest], options), _text_values(b[rest], options),
                      [labels[i] for i in np.flatnonzero(rest)])


def _bool_like(series: pd.Series) -> np.ndarray:
    return np.array([isinstance(v, (bool, np.bool_)) for v in series.to_numpy(dtype=object)], dtype=bool)


def compare_table(reference: Path, candidate: Path, diff: Diff, options: Options) -> None:
    read = read_csv if reference.suffix == ".csv" else pd.read_parquet
    ref, cand = read(reference), read(candidate)
    diff.rows_reference, diff.rows_candidate = len(ref), len(cand)
    compare_frames(ref, cand, diff, options)


def compare_register(reference: Path, candidate: Path, diff: Diff, options: Options) -> None:
    """Two asset finance registers as the stages read them (``register_io``), a batch of cases at a time."""
    from scripts.public_ownership.register_io import read_register, register_cases

    ref_cases, cand_cases = register_cases(reference), register_cases(candidate)
    only_ref = [c for c in ref_cases if c not in set(cand_cases)]
    only_cand = [c for c in cand_cases if c not in set(ref_cases)]
    for case in only_ref[: diff.max_examples]:
        diff.example(f"case only in reference: {case}")
    for case in only_cand[: diff.max_examples]:
        diff.example(f"case only in candidate: {case}")
    common = [c for c in ref_cases if c in set(cand_cases)]
    if [c for c in cand_cases if c in set(common)] != common:
        diff.example("the common cases are in a different order")
        diff.values_differ += 1
    keys = options.keys_for(diff.path) or ["case_id", "component_type", "asset_id"]
    rows_ref = rows_cand = 0
    batch = max(1, options.register_batch_cases)
    for start in range(0, len(common), batch):
        cases = common[start:start + batch]
        ref = read_register(reference, cases=cases, lever_dtype="object")
        cand = read_register(candidate, cases=cases, lever_dtype="object")
        rows_ref += len(ref)
        rows_cand += len(cand)
        compare_frames(ref, cand, diff, options, keys=[k for k in keys if k in ref.columns and k in cand.columns])
    for case in only_ref:
        rows_ref += len(read_register(reference, columns=["case_id"], cases=[case]))
    for case in only_cand:
        rows_cand += len(read_register(candidate, columns=["case_id"], cases=[case]))
    diff.rows_reference, diff.rows_candidate = rows_ref, rows_cand
    diff.rows_only_reference += len(only_ref)
    diff.rows_only_candidate += len(only_cand)


# ── JSON ──────────────────────────────────────────────────────────────────────


def _record_id(items: Sequence) -> str | None:
    if not items or not all(isinstance(item, Mapping) for item in items):
        return None
    for key in RECORD_IDS:
        values = [item.get(key) for item in items]
        if all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in values) and len(set(values)) == len(values):
            return key
    return None


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_json_values(a, b, path: str, diff: Diff, options: Options) -> None:
    """Walk two JSON values together, recording every difference under its dotted path."""
    if options.ignore_keys is not None and path and options.ignore_keys.search(path):
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in a:
            child = f"{path}.{key}" if path else str(key)
            if key not in b:
                if not (options.ignore_keys is not None and options.ignore_keys.search(child)):
                    diff.rows_only_reference += 1
                    diff.example(f"key only in reference: {child}")
                continue
            compare_json_values(a[key], b[key], child, diff, options)
        for key in b:
            if key not in a:
                child = f"{path}.{key}" if path else str(key)
                if not (options.ignore_keys is not None and options.ignore_keys.search(child)):
                    diff.rows_only_candidate += 1
                    diff.example(f"key only in candidate: {child}")
        return
    if isinstance(a, list) and isinstance(b, list):
        ident = _record_id(a)
        if ident is not None and ident == _record_id(b):
            by_b = {item[ident]: item for item in b}
            by_a = {item[ident]: item for item in a}
            for value, item in by_a.items():
                if value in by_b:
                    compare_json_values(item, by_b[value], f"{path}[{ident}={value}]", diff, options)
                else:
                    diff.rows_only_reference += 1
                    diff.example(f"record only in reference: {path}[{ident}={value}]")
            for value in by_b:
                if value not in by_a:
                    diff.rows_only_candidate += 1
                    diff.example(f"record only in candidate: {path}[{ident}={value}]")
            return
        if len(a) != len(b):
            diff.values_differ += 1
            diff.example(f"{path}: list of {len(a)} -> {len(b)}")
        if a and b and all(_is_number(x) or x is None for x in a) and all(_is_number(x) or x is None for x in b) \
                and len(a) == len(b):
            diff.numbers(path, np.array([np.nan if x is None else x for x in a], dtype=float),
                         np.array([np.nan if x is None else x for x in b], dtype=float), options.atol, options.rtol)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare_json_values(x, y, f"{path}[{i}]", diff, options)
        return
    if _is_number(a) and _is_number(b):
        diff.numbers(path, np.array([a], dtype=float), np.array([b], dtype=float), options.atol, options.rtol,
                     [path])
        return
    if isinstance(a, str) and isinstance(b, str):
        a, b = options.normalise(a), options.normalise(b)
    diff.values_compared += 1
    if a != b and not (a is None and b is None):
        diff.values_differ += 1
        diff._worst(path, math.nan, math.nan)
        diff.example(f"{path}: {_short(a)} -> {_short(b)}")


def compare_json(reference: Path, candidate: Path, diff: Diff, options: Options) -> None:
    a = json.loads(reference.read_text(encoding="utf-8"))
    b = json.loads(candidate.read_text(encoding="utf-8"))
    compare_json_values(a, b, "", diff, options)


# ── Files ─────────────────────────────────────────────────────────────────────


def find_outputs(root: Path, include: Sequence[str] = (), exclude: Sequence[str] = DEFAULT_EXCLUDE) -> dict[str, Path]:
    """``{relative posix path: path}`` of every CSV, Parquet and JSON file under ``root`` the patterns select."""
    root = Path(root)
    if root.is_file():
        return {root.name: root}
    out = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if include and not any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(path.name, p) for p in include):
            continue
        if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(path.name, p) for p in exclude):
            continue
        out[rel] = path
    return dict(sorted(out.items()))


def compare_file(rel: str, reference: Path, candidate: Path, options: Options) -> Diff:
    """One file pair's :class:`Diff` (an unreadable file is an ``error``, never a pass)."""
    kind = ("register" if rel.endswith(REGISTER_SUFFIX) else reference.suffix.lower().lstrip("."))
    diff = Diff(rel, kind, options.examples)
    start = time.perf_counter()
    try:
        if kind == "register":
            compare_register(reference, candidate, diff, options)
        elif kind in ("csv", "parquet"):
            compare_table(reference, candidate, diff, options)
        else:
            compare_json(reference, candidate, diff, options)
        diff.finish()
    except Exception as error:  # noqa: BLE001  (reported per file; the run goes on)
        diff.status, diff.error = "error", f"{type(error).__name__}: {error}"
    diff.seconds = time.perf_counter() - start
    return diff


def compare_roots(reference: Path, candidate: Path, options: Options, include: Sequence[str] = (),
                  exclude: Sequence[str] = DEFAULT_EXCLUDE, progress=None) -> list[Diff]:
    """Every file under either root: compared when on both sides, else reported as on one side only."""
    ref_files = find_outputs(reference, include, exclude)
    cand_files = find_outputs(candidate, include, exclude)
    if Path(reference).is_file() and Path(candidate).is_file():
        ref_files, cand_files = {Path(reference).name: Path(reference)}, {Path(reference).name: Path(candidate)}
    results = []
    for rel in sorted(set(ref_files) | set(cand_files)):
        if rel not in cand_files:
            diff = Diff(rel, Path(rel).suffix.lstrip("."), options.examples)
            diff.status = "only_in_reference"
        elif rel not in ref_files:
            diff = Diff(rel, Path(rel).suffix.lstrip("."), options.examples)
            diff.status = "only_in_candidate"
        else:
            diff = compare_file(rel, ref_files[rel], cand_files[rel], options)
        if progress is not None:
            progress(diff)
        results.append(diff)
    return results


def failed(results: Iterable[Diff], missing_ok: bool = False) -> bool:
    bad = {"differs", "error"} | (set() if missing_ok else {"only_in_reference", "only_in_candidate"})
    return any(d.status in bad for d in results)


def _parse_pairs(values: Sequence[str], what: str) -> list[tuple[str, str]]:
    out = []
    for value in values or ():
        if "=" not in value:
            raise SystemExit(f"{what} takes OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        out.append((old, new))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", help="the reference root (or one file)")
    parser.add_argument("candidate", help="the candidate root (or one file)")
    parser.add_argument("--include", nargs="*", default=[], help="glob(s) on the relative path or file name to keep")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE), help="glob(s) to leave out")
    parser.add_argument("--atol", type=float, default=0.0, help="absolute tolerance (default 0: exact)")
    parser.add_argument("--rtol", type=float, default=0.0, help="relative tolerance (default 0: exact)")
    parser.add_argument("--keys", nargs="*", default=[],
                        help="GLOB=col1,col2: the key columns of the tables matching GLOB (default: found by name)")
    parser.add_argument("--ignore-columns", default=None, help="regex: table columns left out")
    parser.add_argument("--ignore-keys", default=DEFAULT_IGNORE_KEYS,
                        help=f"regex on a JSON key's dotted path: left out (default {DEFAULT_IGNORE_KEYS!r})")
    parser.add_argument("--replace", nargs="*", default=[],
                        help="OLD=NEW: text replaced on both sides before comparing (paths a script writes)")
    parser.add_argument("--missing-ok", action="store_true", help="a file on one side only does not fail the run")
    parser.add_argument("--examples", type=int, default=5, help="examples kept per file")
    parser.add_argument("--register-batch-cases", type=int, default=64, help="register cases compared at a time")
    parser.add_argument("--report", default=None, help="write one row per file (CSV)")
    parser.add_argument("--json", default=None, help="write the per-file results (JSON)")
    parser.add_argument("--quiet", action="store_true", help="print only differing files and the totals")
    args = parser.parse_args(argv)

    keys = {}
    for pattern, columns in _parse_pairs(args.keys, "--keys"):
        keys[pattern] = [c for c in columns.split(",") if c]
    options = Options(atol=args.atol, rtol=args.rtol, keys=keys, ignore_columns=args.ignore_columns,
                      ignore_keys=args.ignore_keys or None, replace=_parse_pairs(args.replace, "--replace"),
                      examples=args.examples, register_batch_cases=args.register_batch_cases)

    def progress(diff: Diff) -> None:
        if args.quiet and diff.status == "equal":
            return
        line = f"{diff.status:18s} {diff.path}"
        if diff.status in ("differs", "error"):
            line += (f"  (values differ {diff.values_differ}/{diff.values_compared}, max abs {diff.max_abs:.6g}, "
                     f"max rel {diff.max_rel:.3g}, rows only ref/cand {diff.rows_only_reference}/"
                     f"{diff.rows_only_candidate}, columns +{len(diff.columns_added)}/-{len(diff.columns_removed)})")
            if diff.error:
                line += f"  {diff.error}"
        print(line, flush=True)
        if diff.status in ("differs", "error"):
            for example in diff.examples:
                print(f"    {example}", flush=True)

    for root in (args.reference, args.candidate):
        if not Path(root).exists():
            print(f"no such path: {root}", file=sys.stderr)
            return 2
    results = compare_roots(Path(args.reference), Path(args.candidate), options, args.include, args.exclude, progress)
    counts = pd.Series([d.status for d in results]).value_counts().to_dict() if results else {}
    print(f"{len(results)} files: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    records = [d.record() for d in results]
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.report, index=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    if not results:
        print("no files matched", file=sys.stderr)
        return 1
    return 1 if failed(results, args.missing_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
