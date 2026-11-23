"""Exact branch-constraint screening for a central dispatch Stage 1 (18 Oct 2026).

A central dispatch Stage 1 (``market.wholesale.copperplate_mode: none``) solves
each rolling window as a network-constrained LP on the full ETYS network: every
line and transformer carries a thermal limit (PyPSA's ``Line-fix-s-lower/upper``
and ``Transformer-fix-s-lower/upper`` constraints) on every snapshot, although
only a few dozen of the ~3,000 branches ever bind.

``market.wholesale.central_constraint_screening`` solves each window with the
thermal limits on a *monitored* subset of branches only, then checks the flow on
every branch at every snapshot of the window (the look-ahead blocks included)
against its own limit (``s_max_pu`` x ``s_nom``, after the seasonal ratings, the
outage schedule and the rating overrides) and re-solves with every violated
branch added, until no branch exceeds its limit by more than ``tolerance_mw``.
Everything else in the window LP is untouched: the secured boundary limits
(aggregate, few), Kirchhoff's voltage law, the hydro, storage, look-ahead and
interconnector callbacks.

Why the answer does not change. The screened LP is the full LP with some
constraints left out, so its feasible set contains the full LP's. When its
optimum also satisfies every left-out constraint it is feasible for the full LP,
and no feasible point of the full LP can be cheaper (each is feasible for the
screened LP too), so it is an optimum of the full LP: the same objective, up to
the solver's tolerance. Where the LP has several optima (degenerate dispatch) the
screened solve may return a different one of them at the same cost. The duals
are a valid set for the full LP as well (the left-out limits carry a zero
multiplier, which is complementary to a non-binding limit), so nodal prices are
unchanged up to the same degeneracy. If ``max_iterations`` screened solves still
leave a violation the window is solved once more with every limit monitored,
i.e. exactly as without screening.

Monitored set. The first window starts from the branches named in
``seed_branches_file`` (optional), extendable branches (always monitored) and
nothing else. Each later window starts from every branch an earlier window had to
add (``initial_set: previous_windows``, the default) plus every branch loaded to
at least ``seed_loading_threshold`` of its limit in the previous window's
solution. Each re-solve adds the violated branches plus every unmonitored branch
loaded to ``add_loading_threshold`` or more in that solution, so a branch about
to bind is caught in the same iteration.

The screen works as an ``optimize`` wrapper inside the solver fallback ladder
(``market_utils.solve_with_fallback``): an iteration that a rung does not accept
returns that rung's status at once and the ladder moves on, keeping the branches
found so far; the ladder's post-solve bound check sees only the final, screened
solution, whose branch flows are within ``tolerance_mw`` of their limits.

Off by default (``enabled: false``): the rolling loop then calls exactly what it
called before. Applies only when ``copperplate_mode`` is ``none``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

CONFIG_KEY = "central_constraint_screening"

#: Code defaults; a scenario's ``market.wholesale.central_constraint_screening`` overrides them.
DEFAULT_CONFIG = {
    "enabled": False,
    # previous_windows: start each window from every branch earlier windows had to add,
    # plus the previous window's near-binding branches. empty: start each window from
    # the previous window's near-binding branches only.
    "initial_set": "previous_windows",
    "seed_loading_threshold": 0.9,
    "add_loading_threshold": 0.95,
    "max_iterations": 6,
    "tolerance_mw": 0.1,
    # Optional CSV (columns component, name) of branches monitored from the first window.
    "seed_branches_file": None,
}

INITIAL_SETS = ("previous_windows", "empty")
BRANCH_COMPONENTS = ("Line", "Transformer")
_TABLES = {"Line": ("lines", "lines_t"), "Transformer": ("transformers", "transformers_t")}


def resolve_screening_config(wholesale_config) -> Optional[dict]:
    """The validated screening block, or None when screening is off (the default)."""
    block = (wholesale_config or {}).get(CONFIG_KEY) or {}
    if not isinstance(block, dict):
        raise ValueError(f"market.wholesale.{CONFIG_KEY} must be a mapping, got {block!r}")
    unknown = set(block) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"market.wholesale.{CONFIG_KEY}: unknown keys {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **block}
    if not bool(config["enabled"]):
        return None
    if str(config["initial_set"]) not in INITIAL_SETS:
        raise ValueError(
            f"market.wholesale.{CONFIG_KEY}.initial_set must be one of {INITIAL_SETS}, "
            f"got {config['initial_set']!r}")
    for key in ("seed_loading_threshold", "add_loading_threshold"):
        value = float(config[key])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"market.wholesale.{CONFIG_KEY}.{key} must be in (0, 1], got {value}")
        config[key] = value
    config["max_iterations"] = int(config["max_iterations"])
    if config["max_iterations"] < 1:
        raise ValueError(f"market.wholesale.{CONFIG_KEY}.max_iterations must be >= 1")
    config["tolerance_mw"] = float(config["tolerance_mw"])
    if not config["tolerance_mw"] >= 0.0:
        raise ValueError(f"market.wholesale.{CONFIG_KEY}.tolerance_mw must be >= 0")
    return config


def _extendable(table: pd.DataFrame) -> pd.Series:
    if "s_nom_extendable" not in table.columns:
        return pd.Series(False, index=table.index)
    return table["s_nom_extendable"].fillna(False).astype(bool)


def branch_limits(network, component: str, snapshots=None) -> pd.DataFrame:
    """``s_max_pu`` x ``s_nom`` (MW) of the non-extendable branches, snapshots x names."""
    table = getattr(network, _TABLES[component][0])
    sns = network.snapshots if snapshots is None else pd.Index(snapshots)
    names = table.index[~_extendable(table)]
    if not len(names):
        return pd.DataFrame(index=sns)
    s_max_pu = network.get_switchable_as_dense(component, "s_max_pu", sns)[names]
    s_nom = pd.to_numeric(table.loc[names, "s_nom"], errors="coerce").fillna(0.0)
    return s_max_pu.mul(s_nom, axis=1)


def branch_flows(network, component: str, snapshots=None) -> pd.DataFrame:
    """Solved ``|p0|`` (MW) of every branch, snapshots x names (0 where missing)."""
    table = getattr(network, _TABLES[component][0])
    series = getattr(network, _TABLES[component][1])
    sns = network.snapshots if snapshots is None else pd.Index(snapshots)
    flow = getattr(series, "p0", pd.DataFrame())
    return flow.reindex(index=sns, columns=table.index).astype(float).fillna(0.0).abs()


def branch_excess_and_loading(network, component: str, snapshots=None):
    """``(excess, loading)`` per non-extendable branch: max over snapshots of |p0| - limit (MW) and |p0| / limit."""
    limits = branch_limits(network, component, snapshots)
    if not limits.shape[1]:
        empty = pd.Series(dtype=float)
        return empty, empty
    flows = branch_flows(network, component, snapshots)[limits.columns]
    excess = (flows - limits).max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = flows.to_numpy() / limits.to_numpy()
    ratio = np.where(limits.to_numpy() > 0.0, ratio, np.where(flows.to_numpy() > 0.0, np.inf, 0.0))
    loading = pd.Series(np.nanmax(ratio, axis=0) if ratio.size else [], index=limits.columns)
    return excess, loading


def mask_unmonitored_branch_limits(model, monitored: Dict[str, Iterable[str]]) -> Dict[str, int]:
    """Drop the thermal-limit rows of every branch not in ``monitored`` from a built linopy model.

    Rebuilds ``{Line,Transformer}-fix-s-{lower,upper}`` with a mask that keeps
    the rows already present for the monitored names only (lhs, sign and rhs
    unchanged). Returns the number of branches whose limits were kept, per
    component. A component without those constraints is skipped.
    """
    import xarray as xr

    kept = {}
    for component in BRANCH_COMPONENTS:
        names = set(monitored.get(component, ()))
        for side in ("lower", "upper"):
            con_name = f"{component}-fix-s-{side}"
            if con_name not in model.constraints:
                continue
            con = model.constraints[con_name]
            labels = con.labels
            coord = labels.coords["name"].values
            keep = xr.DataArray(np.isin(coord, list(names)), coords={"name": labels.coords["name"]}, dims="name")
            mask = (labels != -1) & keep
            lhs, sign, rhs = con.lhs, con.sign, con.rhs
            model.remove_constraints(con_name)
            model.add_constraints(lhs, sign, rhs, name=con_name, mask=mask)
            kept[component] = int(keep.sum())
    return kept


def read_seed_branches(path) -> Dict[str, set]:
    """``{component: names}`` from a CSV with columns ``component`` and ``name``."""
    frame = pd.read_csv(path)
    missing = {"component", "name"} - set(frame.columns)
    if missing:
        raise ValueError(f"Seed branch file {path} lacks columns {sorted(missing)}")
    out = {c: set() for c in BRANCH_COMPONENTS}
    for component, name in zip(frame["component"].astype(str), frame["name"].astype(str)):
        if component not in out:
            raise ValueError(f"Seed branch file {path}: component must be Line or Transformer, got {component!r}")
        out[component].add(name)
    return out


class CentralConstraintScreen:
    """Iterative thermal-limit screening for the windows of a central dispatch Stage 1.

    Use: ``start_window(label)`` before a window, solve through
    ``wrap(optimize)``, then ``end_window(network)``. ``summary()`` gives the
    run's totals.
    """

    def __init__(self, network, config: dict, logger: logging.Logger,
                 accepted_conditions: Sequence[str] = ("optimal",)):
        self.config = config
        self.logger = logger
        self.accepted = {str(c).lower() for c in accepted_conditions}
        self.always = {}
        self.totals = {}
        for component in BRANCH_COMPONENTS:
            table = getattr(network, _TABLES[component][0])
            self.always[component] = set(table.index[_extendable(table)])
            self.totals[component] = len(table)
        # Branches some window had to add (or seeded); sticky across windows.
        self.sticky = {c: set(self.always[c]) for c in BRANCH_COMPONENTS}
        seed_file = config.get("seed_branches_file")
        if seed_file:
            for component, names in read_seed_branches(Path(seed_file)).items():
                self.sticky[component] |= names
        self.previous_near = {c: set() for c in BRANCH_COMPONENTS}
        self.current = None
        self.window_label = ""
        self.window_stats = []
        self._reset_window_counters()
        logger.info(
            f"Central constraint screening on: thermal limits on a monitored subset of "
            f"{self.totals['Line']} lines and {self.totals['Transformer']} transformers, checked on every "
            f"branch and snapshot, violations > {config['tolerance_mw']:g} MW added and re-solved "
            f"(at most {config['max_iterations']} screened solves, then every limit); initial set "
            f"{config['initial_set']}, seed loading {config['seed_loading_threshold']:g}, add loading "
            f"{config['add_loading_threshold']:g}"
            + (f", seeded with {sum(len(v) for v in self.sticky.values())} branches from {seed_file}"
               if seed_file else "")
        )

    # ── window bookkeeping ───────────────────────────────────────────────
    def _reset_window_counters(self):
        self.iterations = 0
        self.full_solves = 0
        self.added = 0
        self.solve_seconds = 0.0
        self.max_excess = float("nan")
        self.last_monitored = 0

    def start_window(self, label: str = "") -> None:
        """Set the window's starting monitored set from the earlier windows."""
        self._reset_window_counters()
        self.window_label = label
        if self.config["initial_set"] == "previous_windows":
            base = {c: set(self.sticky[c]) for c in BRANCH_COMPONENTS}
        else:
            base = {c: set(self.always[c]) for c in BRANCH_COMPONENTS}
        for component in BRANCH_COMPONENTS:
            base[component] |= self.previous_near[component]
        self.current = base
        self.started = time.perf_counter()

    def monitored_count(self) -> int:
        return sum(len(v) for v in (self.current or {}).values())

    def _add(self, names_by_component: Dict[str, Iterable[str]]) -> int:
        added = 0
        for component, names in names_by_component.items():
            new = set(names) - self.current[component]
            added += len(new)
            self.current[component] |= new
            self.sticky[component] |= new
        self.added += added
        return added

    def check(self, network):
        """``(violated, near, max_excess)`` of the solved network against every branch limit."""
        violated, near = {}, {}
        worst = 0.0
        for component in BRANCH_COMPONENTS:
            excess, loading = branch_excess_and_loading(network, component)
            if not len(excess):
                violated[component], near[component] = [], []
                continue
            worst = max(worst, float(excess.max()))
            violated[component] = list(excess.index[excess > self.config["tolerance_mw"]])
            near[component] = list(loading.index[loading >= self.config["add_loading_threshold"]])
        return violated, near, worst

    def end_window(self, network) -> dict:
        """Record the window's near-binding branches for the next window and log the window."""
        for component in BRANCH_COMPONENTS:
            _, loading = branch_excess_and_loading(network, component)
            self.previous_near[component] = (
                set(loading.index[loading >= self.config["seed_loading_threshold"]]) if len(loading) else set())
        stats = {
            "window": self.window_label,
            "iterations": self.iterations,
            "full_solves": self.full_solves,
            "monitored": self.last_monitored,
            "added": self.added,
            "max_excess_mw": self.max_excess,
            "solve_seconds": self.solve_seconds,
            "window_seconds": time.perf_counter() - getattr(self, "started", time.perf_counter()),
        }
        self.window_stats.append(stats)
        total = sum(self.totals.values())
        self.logger.info(
            f"Constraint screening {self.window_label}: {self.iterations} solve(s)"
            + (f" ({self.full_solves} with every limit)" if self.full_solves else "")
            + f", {self.last_monitored} of {total} branches monitored ({self.added} added this window), "
            f"max branch excess {self.max_excess:.3g} MW, optimize {self.solve_seconds:.1f}s"
        )
        return stats

    def summary(self) -> dict:
        stats = self.window_stats
        return {
            "windows": len(stats),
            "solves": int(sum(s["iterations"] for s in stats)),
            "full_solves": int(sum(s["full_solves"] for s in stats)),
            "monitored_final": int(sum(len(v) for v in self.sticky.values())),
            "branches": int(sum(self.totals.values())),
            "optimize_seconds": float(sum(s["solve_seconds"] for s in stats)),
        }

    # ── the optimize wrapper ─────────────────────────────────────────────
    def _accepted(self, status, condition) -> bool:
        return str(status) == "ok" and str(condition).lower() in self.accepted

    def wrap(self, optimize: Callable) -> Callable:
        """An ``optimize(network, **kwargs)`` that screens the branch limits around ``optimize``."""
        screen = self

        def screened_optimize(network, extra_functionality=None, **kwargs):
            if screen.current is None:
                screen.start_window()
            max_iterations = screen.config["max_iterations"]
            for iteration in range(1, max_iterations + 2):
                full = iteration > max_iterations
                monitored = None if full else {c: set(v) for c, v in screen.current.items()}

                def masked(n, sns, _monitored=monitored):
                    if extra_functionality is not None:
                        extra_functionality(n, sns)
                    if _monitored is not None:
                        mask_unmonitored_branch_limits(n.model, _monitored)

                started = time.perf_counter()
                status, condition = optimize(network, extra_functionality=masked, **kwargs)
                screen.solve_seconds += time.perf_counter() - started
                screen.iterations += 1
                screen.full_solves += int(full)
                screen.last_monitored = sum(screen.totals.values()) if full else sum(
                    len(v) for v in monitored.values())
                if not screen._accepted(status, condition):
                    screen.logger.info(
                        f"Constraint screening {screen.window_label}: solve {iteration} returned "
                        f"{status} ({condition}); handing back to the solver ladder")
                    return status, condition
                violated, near, worst = screen.check(network)
                screen.max_excess = worst
                n_violated = sum(len(v) for v in violated.values())
                if full or not n_violated:
                    return status, condition
                to_add = {c: set(violated[c]) | set(near[c]) for c in BRANCH_COMPONENTS}
                added = screen._add(to_add)
                screen.logger.info(
                    f"Constraint screening {screen.window_label}: solve {iteration} "
                    f"({screen.last_monitored} monitored) left {n_violated} branch(es) over their limit "
                    f"(max {worst:.1f} MW); adding {added} and re-solving"
                    + (" with every limit" if iteration == max_iterations else "")
                )
            return status, condition  # pragma: no cover - the loop always returns

        return screened_optimize


def build_central_constraint_screen(network, wholesale_config, logger,
                                    accepted_conditions: Sequence[str] = ("optimal",)):
    """A :class:`CentralConstraintScreen` for a central dispatch Stage 1 with screening on, else None."""
    if str((wholesale_config or {}).get("copperplate_mode", "collapse")).lower() != "none":
        if ((wholesale_config or {}).get(CONFIG_KEY) or {}).get("enabled"):
            logger.info("Central constraint screening: not applied (copperplate_mode is not 'none')")
        return None
    config = resolve_screening_config(wholesale_config)
    if config is None:
        return None
    return CentralConstraintScreen(network, config, logger, accepted_conditions)
