"""Opt-in wall-clock timing of the ownership layer's hot loops (F20 phase 0, 13 Oct 2026).

Off unless the environment variable ``PYPSA_GB_STAGE_TIMING`` is set, and then
only measurement: no number a stage writes can move.  Its value is the folder the
timings go to (``1``, ``true`` or ``yes`` mean ``logs/stage_timing``).  Each
process writes one CSV there when it exits, named after the script and its
process id, with one row per label: calls, total, mean and largest seconds, and
the process's own wall time as the label ``<script>.total``; every row also carries
the process's peak memory (``peak_mb``, F20 phase 1), the figure a rule's ``mem_mb``
is set from.

Three ways in, each a no-op while timing is off:

* :func:`instrument` swaps named functions of a module (or methods of a class)
  for timed wrappers, so a stage times its own helpers with one line in
  ``main`` and no reindented code.  A generator function is timed over its
  iterations, so a streamed register read counts the reads, not the call.
* :func:`laps` returns a lap timer: ``lap("label")`` books the time since the
  previous lap, for the steps of one long function.
* :func:`timed` is a context manager for a block.

Labels nest (a wrapped function that calls another wrapped one counts the inner
time under both), so the table is read label by label, not summed.  A
multiprocessing worker exits without running ``atexit`` handlers, so a pool
worker calls :func:`flush` itself.

The table of a run::

    python scripts/public_ownership/stage_timing.py summarise logs/stage_timing [--out stage_times.md]
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import inspect
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable, Mapping

ENV_VAR = "PYPSA_GB_STAGE_TIMING"
DEFAULT_DIR = Path("logs") / "stage_timing"
_TRUE = {"1", "true", "yes", "on"}
WRAPPED_ATTR = "__stage_timing_label__"
COLUMNS = ("script", "run_group", "pid", "label", "calls", "total_s", "mean_s", "max_s", "peak_mb", "argv")

_records: dict[str, list[float]] = {}
_started = time.perf_counter()
_registered = False
_flushes = 0


def enabled() -> bool:
    """Whether timing is on (``PYPSA_GB_STAGE_TIMING`` set to anything but empty, 0, false or no)."""
    value = os.environ.get(ENV_VAR, "").strip()
    return bool(value) and value.lower() not in {"0", "false", "no", "off"}


def output_dir() -> Path:
    """The folder the timings go to."""
    value = os.environ.get(ENV_VAR, "").strip()
    return DEFAULT_DIR if value.lower() in _TRUE else Path(value)


def record(label: str, seconds: float) -> None:
    """Book ``seconds`` under ``label`` (calls, total, largest)."""
    entry = _records.get(label)
    if entry is None:
        _records[label] = [1.0, seconds, seconds]
    else:
        entry[0] += 1.0
        entry[1] += seconds
        entry[2] = max(entry[2], seconds)
    _register_flush()


def _register_flush() -> None:
    global _registered
    if not _registered:
        atexit.register(flush)
        _registered = True


def _timed_function(function, label: str):
    if inspect.isgeneratorfunction(function):
        @functools.wraps(function)
        def generator_wrapper(*args, **kwargs):
            iterator = function(*args, **kwargs)
            try:
                while True:
                    start = time.perf_counter()
                    try:
                        item = next(iterator)
                    except StopIteration:
                        record(label, time.perf_counter() - start)
                        return
                    record(label, time.perf_counter() - start)
                    yield item
            finally:
                iterator.close()        # an abandoned loop closes the wrapped generator as it would have been

        setattr(generator_wrapper, WRAPPED_ATTR, label)
        return generator_wrapper

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            record(label, time.perf_counter() - start)

    setattr(wrapper, WRAPPED_ATTR, label)
    return wrapper


def instrument(namespace, names: Iterable[str], prefix: str) -> list[str]:
    """Replace each named function of ``namespace`` with a timed wrapper (only while timing is on).

    Args:
        namespace: A module, a class (its plain methods) or a mapping such as a module's ``globals()``.
        names: The attribute names to time; a name the namespace lacks, or one already wrapped, is skipped.
        prefix: The label prefix (``<prefix>.<name>``).

    Returns:
        The labels wrapped ([] while timing is off).
    """
    if not enabled():
        return []
    is_mapping = isinstance(namespace, Mapping)
    wrapped = []
    for name in names:
        current = namespace.get(name) if is_mapping else getattr(namespace, name, None)
        if current is None or not callable(current) or getattr(current, WRAPPED_ATTR, None):
            continue
        if inspect.isclass(namespace) and not inspect.isfunction(inspect.getattr_static(namespace, name)):
            continue  # classmethods, staticmethods and properties keep their own binding
        label = f"{prefix}.{name}"
        timed_function = _timed_function(current, label)
        if is_mapping:
            namespace[name] = timed_function
        else:
            setattr(namespace, name, timed_function)
        wrapped.append(label)
    return wrapped


#: The register reads every heavy stage shares (``register_io``), timed by :func:`instrument_register_io`.
REGISTER_IO_FUNCTIONS = ("read_register", "iter_register_cases", "register_cases")
REGISTER_IO_METHODS = {"CaseRows": ("_case", "prefetch"), "RegisterWriter": ("write", "close"),
                       "RegisterReader": ("read",)}


def instrument_register_io() -> list[str]:
    """Time the register's readers and writer (``register_io.*``), once per process; [] while timing is off.

    Call it before a stage imports ``read_register`` by name, so the name it binds is the timed one.
    """
    if not enabled():
        return []
    from scripts.public_ownership import register_io

    labels = instrument(register_io, REGISTER_IO_FUNCTIONS, prefix="register_io")
    for cls, methods in REGISTER_IO_METHODS.items():
        if hasattr(register_io, cls):           # a class an older register_io lacks is skipped
            labels += instrument(getattr(register_io, cls), methods, prefix=f"register_io.{cls}")
    return labels


#: The CfD re-pricing the strike adjustments and the public account share (``cfd_repricing``).
CFD_REPRICING_FUNCTIONS = ("per_row_savings", "register_settings_by_case", "_priced_rows", "settled_energy_by_key",
                           "limit_at_cost_to_deliverable", "route_summary", "_case_savings")
# F20 phase 2: _case_savings counts the savings computed; savings_batches is the strike stage's lean pass.
CFD_REPRICING_METHODS = {"SavingsStream": ("__iter__", "savings_batches"),
                         "SavingsContext": ("contract_revenue_for", "private_for"),
                         "StrikeAdjustments": ("add", "frame")}


def instrument_cfd_repricing() -> list[str]:
    """Time the shared CfD re-pricing steps (``cfd_repricing.*``), once per process; [] while timing is off.

    A stage that imported one of these functions by name times its own binding with :func:`instrument`.
    """
    if not enabled():
        return []
    from scripts.public_ownership import cfd_repricing

    labels = instrument(cfd_repricing, CFD_REPRICING_FUNCTIONS, prefix="cfd_repricing")
    for cls, methods in CFD_REPRICING_METHODS.items():
        labels += instrument(getattr(cfd_repricing, cls), methods, prefix=f"cfd_repricing.{cls}")
    return labels


class _NullLaps:
    def __call__(self, label: str) -> None:
        return None


class Laps:
    """``lap("label")`` books the time since the previous lap (or since the timer was made)."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self._last = time.perf_counter()

    def __call__(self, label: str) -> None:
        now = time.perf_counter()
        record(f"{self.prefix}.{label}", now - self._last)
        self._last = now


def laps(prefix: str):
    """A lap timer under ``prefix`` while timing is on, else one that does nothing."""
    return Laps(prefix) if enabled() else _NullLaps()


def timed(label: str):
    """A context manager booking its block under ``label`` (a null context while timing is off)."""
    if not enabled():
        return contextlib.nullcontext()
    return _block(label)


@contextlib.contextmanager
def _block(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        record(label, time.perf_counter() - start)


def script_name() -> str:
    """The running script's stem (``nationalised_tariff``), or ``python`` for an interactive process."""
    argv0 = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
    return argv0 or "python"


def run_group_from_argv(argv: Iterable[str] | None = None) -> str:
    """The run group named by the first ``{run_group}_asset_finance_register`` (or similar) path on the command line."""
    pattern = re.compile(r"([A-Za-z0-9_]+?)_(asset_finance_register|case_settings|consumer_bill_by_ownership_case"
                         r"|financing_summary|public_account|nationalised_bill|exchequer_ledger)")
    for arg in (sys.argv[1:] if argv is None else argv):
        match = pattern.search(Path(str(arg)).name)
        if match:
            return match.group(1)
    return ""


def peak_memory_mb() -> float | None:
    """The process's peak memory so far in MB (peak working set on Windows, max RSS elsewhere); None if unknown.

    F20 phase 1: written on every row, so a timed run also measures each stage's peak
    (the ``mem_mb`` the rules declare).
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            return peak / 1e6
    except Exception:  # noqa: BLE001  (psutil absent or the platform has no peak)
        pass
    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak_kb / 1e3 if sys.platform != "darwin" else peak_kb / 1e6
    except Exception:  # noqa: BLE001
        return None


def rows() -> list[dict]:
    """This process's timings as table rows (the script total first), each with the process's peak memory."""
    script, group, pid = script_name(), run_group_from_argv(), os.getpid()
    argv = " ".join(sys.argv[1:])[:400]
    wall = time.perf_counter() - _started
    peak = peak_memory_mb()
    out = [{"script": script, "run_group": group, "pid": pid, "label": f"{script}.total", "calls": 1,
            "total_s": wall, "mean_s": wall, "max_s": wall, "peak_mb": peak, "argv": argv}]
    for label, (calls, total, largest) in sorted(_records.items()):
        out.append({"script": script, "run_group": group, "pid": pid, "label": label, "calls": int(calls),
                    "total_s": total, "mean_s": total / calls if calls else 0.0, "max_s": largest, "peak_mb": peak,
                    "argv": argv})
    return out


def flush() -> Path | None:
    """Write this process's timings to the output folder (once per process; nothing while timing is off)."""
    if not enabled() or not _records:
        return None
    import csv

    folder = output_dir()
    folder.mkdir(parents=True, exist_ok=True)
    global _flushes
    _flushes += 1
    path = folder / f"{script_name()}_{os.getpid()}_{time.strftime('%Y%m%d_%H%M%S')}_{_flushes}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows())
    _records.clear()
    return path


def summarise(folder: str | Path):
    """Every timing file in ``folder`` as one table: label totals summed over processes, largest first."""
    import pandas as pd

    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no timing files in {folder}")
    frame = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    # F20 phase 1: each process's peak memory (absent from files written before it: blank).
    frame["peak_mb"] = pd.to_numeric(frame["peak_mb"], errors="coerce") if "peak_mb" in frame.columns else float("nan")
    table = (frame.groupby(["script", "run_group", "label"], as_index=False)
             .agg(processes=("pid", "nunique"), calls=("calls", "sum"), total_s=("total_s", "sum"),
                  max_s=("max_s", "max"), peak_mb=("peak_mb", "max")))
    table["mean_s"] = table["total_s"] / table["calls"].where(table["calls"] > 0)
    script_totals = table.loc[table["label"].eq(table["script"] + ".total")].set_index(["script", "run_group"])["total_s"]
    table["share_of_script"] = [
        total / script_totals.get((script, group), float("nan"))
        for script, group, total in zip(table["script"], table["run_group"], table["total_s"])]
    return table.sort_values(["script", "run_group", "total_s"], ascending=[True, True, False]).reset_index(drop=True)


def to_markdown(table) -> str:
    """The summary as a Markdown table (minutes and seconds, no dependency beyond pandas)."""
    lines = ["| Script | Run group | Label | Processes | Calls | Total (min) | Mean (s) | Max (s) | Share | Peak (MB) |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table.itertuples(index=False):
        peak = getattr(row, "peak_mb", float("nan"))
        lines.append(f"| {row.script} | {row.run_group} | {row.label} | {row.processes} | {row.calls} | "
                     f"{row.total_s / 60:.2f} | {row.mean_s:.3f} | {row.max_s:.3f} | {row.share_of_script:.0%} | "
                     f"{'' if peak != peak else f'{peak:,.0f}'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summarise", help="one stage-time table from a folder of timing files")
    summary.add_argument("folder", nargs="?", default=str(DEFAULT_DIR))
    summary.add_argument("--out", default=None, help="write the table here (.md or .csv) as well as printing it")
    args = parser.parse_args(argv)
    table = summarise(args.folder)
    text = to_markdown(table)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".csv":
            table.to_csv(out, index=False)
        else:
            out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
