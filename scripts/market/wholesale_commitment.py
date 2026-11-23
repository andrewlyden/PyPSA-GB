"""Stage 1 unit commitment with prices (16 Oct 2026, Stage 1 price-gap test, variant s1_uc).

``market.wholesale.unit_commitment`` (``solve_wholesale.apply_wholesale_unit_commitment``) makes the
listed carriers committable, which turns the Stage 1 window into a MILP. A MILP has no duals, so that
overlay alone gives no price. ``unit_commitment.method`` chooses how the price is formed:

``milp`` (code default, the overlay as before)
    one MILP per window; no marginal prices.
``milp_fixed``
    two passes per window, the restricted pricing PLEXOS uses after its commitment step: (1) the MILP
    fixes which units are on; (2) the LP is solved with that commitment held as output bounds,
    ``status x floor(t) <= p <= status x available(t)``, and its duals are the prices. A committed
    unit above its floor sets the price at its marginal cost, part-loaded or not; start-up costs shape
    the commitment but, as in any fixed-commitment pricing, are not in the price.
``linearized``
    PyPSA's linearised (Hua, Liu and Zhang 2017) relaxation, the clustered commitment of a station
    treated as a continuum of small units; one LP, duals direct.

Design points:

* **Availability is never touched.** The commitment enters the pricing LP as constraints, never as a
  lower ``p_max_pu``, so the MR3 scarcity margin (``scarcity_pricing.derated_capacity``: CM factor x
  ``p_max_pu`` x ``p_nom`` less dispatch) counts an uncommitted unit's available capacity, as the BSC
  de-rated margin does. Zeroing ``p_max_pu`` made the margin read switched-off gas as lost margin, the
  +GBP 25/MWh artefact of the 6 Oct 2026 prototype (scratchpad price_tests, mode fixed_lp).
* **Minimum stable generation relative to availability, per module.** A unit's floor when on is
  ``msg x min(1, module_mw / p_nom) x p_max_pu(t)``: the network's gas units are stations (24 CCGT
  stations, mean 943 MW), and a multi-shaft CCGT's minimum generation is a modular level, one gas
  turbine and the steam turbine (NESO Grid Code guidance notes for generators), so a committed station
  holds at least one module at its stable level. ``module_mw`` null: the whole unit is one module.
  Start-up cost per start is ``start_up_cost_per_mw x min(module_mw, p_nom)``.
* **Commitment carried between windows** (``milp_fixed``): each window starts from the up/down run
  the previous window's kept hours ended on (``up_time_before`` / ``down_time_before``); the first
  window of the solve period takes ``initial_status``. A re-solve of the same window (the counterflow
  lock, a fallback rung) reuses that window's commitment.
* **Must-run units are left out:** a unit with a positive minimum output anywhere (CHP, nuclear) is
  already must-run and is not committed.
* **Zero-commitment guard** (16 Oct 2026). A window whose MILP commits no available gas in its kept hours
  first solves the plain LP (gas free). If the plain LP runs at most ``zero_commitment_guard_mwh`` (default 50)
  MWh of gas there, the two LPs are the same and the plain LP's solution is kept (``zero_commitment_windows``).
  If it runs more:
    - when the MILP itself ran gas in those hours, the commitment has been lost (an empty or misaligned status
      frame): ERROR, the window keeps the plain LP, counted in ``guard_fallback_windows``;
    - otherwise the MILP chose to keep gas off (start-up costs, minimum stable generation): that is the
      commitment's answer, so the window is priced with gas off, logged as a WARNING and counted in
      ``zero_commitment_plain_gas_windows``; ``zero_commitment_fallback: true`` makes these fall back to the plain
      LP as well (ERROR, ``guard_fallback_windows``).
  The MILP's "on" status on a unit with no available output (costless, so arbitrary) is dropped before pricing
  and the carry-over.
* **Coarse look-ahead** (``look_ahead_resolution_hours`` > 1): PyPSA counts minimum up and down times
  in snapshots, so inside the look-ahead blocks they are stricter than in hours; the kept hours are
  hourly, and the look-ahead only values the end-of-day state (the same declared approximation as the
  ramp limits there).

The defaults below are those of ``config/defaults/`` ``market.wholesale.unit_commitment``.
"""
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr

COMMITMENT_METHODS = ("milp", "milp_fixed", "linearized")
#: Code defaults of the keys this module reads (``method``: the overlay as before 16 Oct 2026).
DEFAULT_METHOD = "milp"
DEFAULT_MILP_OPTIONS = {"MIPGap": 0.005, "TimeLimit": 600}
#: Gas energy (MWh, kept hours) the plain LP may run in a zero-commitment window before the guard trips:
#: about 2 MW on average over a day, below one OCGT module at its stable level for an hour, and above the
#: few MWh a barrier solve without crossover leaves on gas in a gas-free day (6-8 MWh on 1-2 Jan 2030).
DEFAULT_ZERO_COMMITMENT_GUARD_MWH = 50.0
#: Gurobi options that only apply to barrier and are dropped for the MILP pass.
BARRIER_ONLY_KEYS = ("BarHomogeneous", "BarConvTol", "BarIterLimit")
STATUS_TOL = 0.5


def commitment_method(uc_config) -> str:
    method = str((uc_config or {}).get("method", DEFAULT_METHOD)).strip().lower()
    if method not in COMMITMENT_METHODS:
        raise ValueError(f"market.wholesale.unit_commitment.method {method!r}; choose from {COMMITMENT_METHODS}")
    return method


def priced_commitment(wholesale_config) -> bool:
    """True when unit commitment is on with a method that yields prices (needs the window wrapper)."""
    uc = (wholesale_config or {}).get("unit_commitment") or {}
    return bool(uc.get("enabled", False)) and commitment_method(uc) != "milp"


def capture_bounds(network) -> dict:
    """The generators' minimum-output inputs before the commitment overlay rewrites them."""
    tv = network.generators_t.get("p_min_pu", pd.DataFrame())
    return {"p_min_pu": network.generators["p_min_pu"].astype(float).copy(),
            "p_min_pu_t": (tv if tv is not None else pd.DataFrame()).copy(),
            "start_up_cost": network.generators.get("start_up_cost", pd.Series(0.0, index=network.generators.index)).copy(),
            "shut_down_cost": network.generators.get("shut_down_cost", pd.Series(0.0, index=network.generators.index)).copy()}


class CommitmentPlan:
    """What the overlay committed, with the bounds to restore after the solve."""

    def __init__(self, method: str, units: list, floor_pu: pd.Series, original: dict, uc_config: dict):
        self.method = method
        self.units = list(units)
        self.floor_pu = floor_pu
        self.original = original
        self.uc_config = uc_config

    def restore(self, network, logger=None) -> None:
        """Put the network's commitment inputs back as they were (the saved Stage 1 network stays LP-shaped)."""
        gens = network.generators
        units = [u for u in self.units if u in gens.index]
        gens.loc[units, "committable"] = False
        gens.loc[units, "p_min_pu"] = self.original["p_min_pu"].reindex(units).fillna(0.0).to_numpy()
        for col in ("start_up_cost", "shut_down_cost"):
            if col in gens:
                gens.loc[units, col] = self.original[col].reindex(units).fillna(0.0).to_numpy()
        tv = network.generators_t.get("p_min_pu", pd.DataFrame())
        orig_tv = self.original["p_min_pu_t"]
        if tv is not None and not tv.empty:
            drop = [u for u in units if u in tv.columns and u not in orig_tv.columns]
            tv = tv.drop(columns=drop)
            for u in units:
                if u in orig_tv.columns and u in tv.columns:
                    tv[u] = orig_tv[u].reindex(tv.index).to_numpy()
            network.generators_t["p_min_pu"] = tv
        if logger:
            logger.info(f"Unit commitment: restored the LP inputs of {len(units)} committed units on the saved network")


def configure_commitment(network, wholesale_config, original: dict, logger) -> CommitmentPlan | None:
    """After ``apply_wholesale_unit_commitment``: per-module floors on availability, must-run units out.

    ``original`` is :func:`capture_bounds` taken before the overlay. Returns None when commitment is off
    or its method is the plain MILP.
    """
    uc = (wholesale_config or {}).get("unit_commitment") or {}
    if not priced_commitment(wholesale_config):
        return None
    method = commitment_method(uc)
    gens = network.generators
    com = gens.index[gens["committable"].fillna(False).astype(bool)]
    # Must-run units (a positive minimum output in the input) are already held on: leave them out.
    orig_static = original["p_min_pu"].reindex(com).fillna(0.0)
    orig_tv = original["p_min_pu_t"]
    must_run = set(orig_static.index[orig_static > 1e-9])
    if orig_tv is not None and not orig_tv.empty:
        cols = [c for c in com if c in orig_tv.columns]
        must_run |= {c for c in cols if float(orig_tv[c].max()) > 1e-9}
    if must_run:
        gens.loc[list(must_run), "committable"] = False
        gens.loc[list(must_run), "p_min_pu"] = orig_static.reindex(list(must_run)).to_numpy()
        logger.info(f"Unit commitment: {len(must_run)} must-run units (positive minimum output) left uncommitted")
    com = [c for c in com if c not in must_run]
    if not com:
        raise ValueError("Unit commitment enabled but no unit is committable after the must-run exclusion")
    params = uc.get("carrier_parameters") or {}
    p_nom = gens.loc[com, "p_nom"].astype(float)
    carrier = gens.loc[com, "carrier"].astype(str)
    msg = gens.loc[com, "p_min_pu"].astype(float)          # set by the overlay from carrier p_min_pu
    module = pd.Series([params.get(c, {}).get("module_mw") for c in carrier], index=com, dtype=object)
    module_mw = pd.Series([p if (m is None or pd.isna(m)) else min(float(m), p) for m, p in zip(module, p_nom)],
                          index=com, dtype=float)
    share = (module_mw / p_nom.where(p_nom > 0)).fillna(1.0).clip(upper=1.0)
    floor_pu = msg * share
    start = pd.Series([float(params.get(c, {}).get("start_up_cost_per_mw", 0.0)) for c in carrier], index=com)
    shut = pd.Series([float(params.get(c, {}).get("shut_down_cost_per_mw", 0.0)) for c in carrier], index=com)
    gens.loc[com, "start_up_cost"] = (start * module_mw).to_numpy()
    gens.loc[com, "shut_down_cost"] = (shut * module_mw).to_numpy()
    # Floor relative to availability: static p_max_pu x floor, and per hour where p_max_pu varies.
    static_max = gens.loc[com, "p_max_pu"].astype(float)
    gens.loc[com, "p_min_pu"] = (floor_pu * static_max).to_numpy()
    tv_max = network.generators_t.get("p_max_pu", pd.DataFrame())
    if tv_max is not None and not tv_max.empty:
        cols = [c for c in com if c in tv_max.columns]
        if cols:
            tv_min = network.generators_t.get("p_min_pu", pd.DataFrame())
            tv_min = (tv_min if tv_min is not None else pd.DataFrame()).reindex(network.snapshots)
            add = tv_max[cols].mul(floor_pu[cols], axis=1)
            for c in cols:
                tv_min[c] = add[c].to_numpy()
            network.generators_t["p_min_pu"] = tv_min
    logger.info(
        f"Unit commitment ({method}): {len(com)} units, {p_nom.sum():,.0f} MW; floor (MSG x module share) by "
        f"carrier {floor_pu.groupby(carrier).mean().round(3).to_dict()}, start cost per start by carrier "
        f"{(start * module_mw).groupby(carrier).mean().round(0).to_dict()} GBP"
    )
    return CommitmentPlan(method, com, floor_pu, original, uc)


def _dense(n, attr, names, snapshots) -> pd.DataFrame:
    return n.get_switchable_as_dense("Generator", attr, snapshots)[list(names)]


def _run_lengths(status: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(up_time_before, down_time_before) a window ending on these rows hands the next one."""
    up, down = {}, {}
    values = status.to_numpy(dtype=float) >= STATUS_TOL
    for j, unit in enumerate(status.columns):
        col = values[:, j]
        last = col[-1]
        run = 0
        for v in col[::-1]:
            if v != last:
                break
            run += 1
        up[unit], down[unit] = (run, 0) if last else (0, run)
    return pd.Series(up, dtype=int), pd.Series(down, dtype=int)


class CommitmentOptimizer:
    """The ``optimize`` callable a Stage 1 window solves through (``solve_with_counterflow_lock``)."""

    def __init__(self, plan: CommitmentPlan, original_optimize, window_steps: int, logger=None):
        self.plan = plan
        self.original = original_optimize
        self.window_steps = int(window_steps)
        self.logger = logger or logging.getLogger(__name__)
        self.method = plan.method
        uc = plan.uc_config
        self.initial_status = str(uc.get("initial_status", "off")).lower()
        self.milp_options = {**DEFAULT_MILP_OPTIONS, **(uc.get("milp_options") or {})}
        self.guard_mwh = float(uc.get("zero_commitment_guard_mwh", DEFAULT_ZERO_COMMITMENT_GUARD_MWH))
        self.zero_fallback = bool(uc.get("zero_commitment_fallback", False))
        self._window_milp_gas = np.nan  # the MILP's own gas energy in the kept hours of the current window
        self._window_start = None
        self._window_status = None      # the MILP commitment of the current window (reused on re-solves)
        self._window_tried = False      # the MILP of the current window has run (a failure is not retried)
        self._window_initial = None     # (up, down) this window started from
        self._carry = None              # (up, down) the last solved window hands the next one
        self.records = []

    # ── initial state ───────────────────────────────────────────────────
    def _first_window_state(self, n, units) -> tuple[pd.Series, pd.Series]:
        gens = n.generators.loc[units]
        if self.initial_status == "on":
            return gens["min_up_time"].astype(int).clip(lower=1), pd.Series(0, index=units)
        if self.initial_status == "free":
            return pd.Series(0, index=units), pd.Series(0, index=units)
        return pd.Series(0, index=units), gens["min_down_time"].astype(int).clip(lower=1)

    def _set_initial(self, n, units) -> None:
        start = n.snapshots[0]
        if start != self._window_start:
            # A new window: start from the previous window's end, or the configured state.
            self._window_start = start
            self._window_status = None
            self._window_tried = False
            self._window_milp_gas = np.nan
            if self._carry is not None and self.method == "milp_fixed":
                up, down = self._carry
                self._window_initial = (up.reindex(units).fillna(0).astype(int),
                                        down.reindex(units).fillna(0).astype(int))
            else:
                self._window_initial = self._first_window_state(n, units)
        up, down = self._window_initial
        n.generators.loc[units, "up_time_before"] = up.reindex(units).to_numpy()
        n.generators.loc[units, "down_time_before"] = down.reindex(units).to_numpy()

    # ── the solve ───────────────────────────────────────────────────────
    def __call__(self, n, **kwargs):
        gens = n.generators
        units = [u for u in self.plan.units if u in gens.index and bool(gens.at[u, "committable"])]
        if not units:
            return self.original(n, **kwargs)
        self._set_initial(n, units)
        if self.method == "linearized":
            t0 = time.perf_counter()
            result = self.original(n, linearized_unit_commitment=True, **kwargs)
            self.records.append({"window_start": str(n.snapshots[0]), "method": "linearized",
                                 "status": str(result[0]), "condition": str(result[1]),
                                 "seconds": round(time.perf_counter() - t0, 1)})
            return result
        return self._milp_fixed(n, units, kwargs)

    def _milp_pass(self, n, kwargs) -> tuple[pd.DataFrame | None, str, str, float]:
        options = {k: v for k, v in dict(kwargs.get("solver_options") or {}).items()
                   if k not in BARRIER_ONLY_KEYS and str(k).lower() != "crossover"}
        options.update(self.milp_options)
        n.generators_t["status"] = pd.DataFrame(index=n.snapshots)
        t0 = time.perf_counter()
        status, condition = self.original(n, **{**kwargs, "solver_options": options})
        seconds = time.perf_counter() - t0
        st = n.generators_t.get("status")
        units = [u for u in self.plan.units if u in n.generators.index]
        if st is None or st.empty or not set(units).issubset(st.columns) or str(status) != "ok":
            return None, str(status), str(condition), seconds
        p = n.generators_t.get("p")
        steps = min(self.window_steps, len(n.snapshots))
        if p is not None and set(units).issubset(p.columns):
            w = n.snapshot_weightings.generators.reindex(n.snapshots).fillna(1.0).iloc[:steps]
            self._window_milp_gas = float(p[units].iloc[:steps].sum(axis=1).mul(w).sum())
        return (st.reindex(index=n.snapshots, columns=units).fillna(0.0) >= STATUS_TOL).astype(float), \
            str(status), str(condition), seconds

    def _milp_fixed(self, n, units, kwargs):
        gens = n.generators
        sns = n.snapshots
        milp_status = milp_condition = "reused"
        milp_seconds = 0.0
        if not self._window_tried:
            self._window_tried = True
            commitment, milp_status, milp_condition, milp_seconds = self._milp_pass(n, kwargs)
            if commitment is None:
                self.logger.warning(f"Unit commitment window {sns[0]}: MILP returned {milp_status} "
                                    f"({milp_condition}) without a commitment; this window is priced on "
                                    "the plain LP (no commitment)")
            self._window_status = commitment
        commitment = self._window_status
        floor_mw = _dense(n, "p_min_pu", units, sns).mul(gens.loc[units, "p_nom"].astype(float), axis=1)
        avail_mw = _dense(n, "p_max_pu", units, sns).mul(gens.loc[units, "p_nom"].astype(float), axis=1)
        steps = min(self.window_steps, len(sns))
        weights = n.snapshot_weightings.generators.reindex(sns).fillna(1.0)
        committed_mw = None
        if commitment is not None:
            # A unit with no available output cannot be on: the MILP's "on" there is costless and arbitrary.
            commitment = commitment.reindex(index=sns, columns=units).fillna(0.0).where(avail_mw > 1e-6, 0.0)
            committed_mw = (commitment * avail_mw).sum(axis=1)
        zero_commitment = committed_mw is not None and float(committed_mw.iloc[:steps].sum()) <= 1e-6
        milp_gas = self._window_milp_gas
        plain_gas, guard, plain_gas_kept_off = np.nan, False, False
        lp_seconds = 0.0
        if zero_commitment:
            lp_status, lp_condition, seconds = self._solve_lp(n, units, None, kwargs, avail_mw, floor_mw)
            lp_seconds += seconds
            plain_gas = self._kept_gas(n, units, steps, weights)
            if plain_gas > self.guard_mwh:
                lost = np.isfinite(milp_gas) and milp_gas > self.guard_mwh
                if lost or self.zero_fallback:
                    guard = True
                    why = (f"the MILP ran {milp_gas:,.0f} MWh of gas but its commitment reads as zero available MW "
                           "(status frame lost or misaligned)" if lost else "zero_commitment_fallback is set")
                    self.logger.error(f"Unit commitment window {sns[0]}: no available gas committed in the kept hours "
                                      f"while the plain LP runs {plain_gas:,.0f} MWh of gas; {why}; the window keeps "
                                      "the plain LP (no commitment), counted in guard_fallback_windows")
                else:
                    plain_gas_kept_off = True
                    self.logger.warning(f"Unit commitment window {sns[0]}: the MILP keeps gas off where the plain LP "
                                        f"runs {plain_gas:,.0f} MWh (start-up costs or minimum stable generation); "
                                        "priced with gas off, counted in zero_commitment_plain_gas_windows")
                    lp_status, lp_condition, seconds = self._solve_lp(n, units, commitment, kwargs, avail_mw, floor_mw)
                    lp_seconds += seconds
        else:
            lp_status, lp_condition, lp_seconds = self._solve_lp(n, units, commitment, kwargs, avail_mw, floor_mw)
        lp_gas = self._kept_gas(n, units, steps, weights)
        if commitment is not None:
            self._carry = _run_lengths(commitment.iloc[:steps])
            on_mw = committed_mw.iloc[:steps]
        else:
            on_mw = pd.Series(np.nan, index=sns[:steps])
        self.records.append({"window_start": str(sns[0]), "method": "milp_fixed", "milp_status": milp_status,
                             "milp_condition": milp_condition, "milp_seconds": round(milp_seconds, 1),
                             "lp_status": str(lp_status), "lp_condition": str(lp_condition),
                             "lp_seconds": round(lp_seconds, 1),
                             "committed_available_mw_mean": float(on_mw.mean()),
                             "milp_gas_mwh": milp_gas, "plain_lp_gas_mwh": plain_gas, "lp_gas_mwh": lp_gas,
                             "zero_commitment": bool(zero_commitment), "guard_fallback": guard,
                             "zero_commitment_plain_gas": plain_gas_kept_off})
        priced = ("plain LP (guard fallback)" if guard else
                  "plain LP (no gas committed or needed)" if zero_commitment and not plain_gas_kept_off else
                  "pricing LP")
        self.logger.info(f"Unit commitment window {sns[0]}: MILP {milp_status}/{milp_condition} {milp_seconds:.1f}s "
                         f"(gas {milp_gas:,.0f} MWh in the kept hours), {priced} {lp_status}/{lp_condition} "
                         f"{lp_seconds:.1f}s (gas {lp_gas:,.0f} MWh); committed available "
                         f"{float(on_mw.mean()):,.0f} MW on average over the kept hours")
        return lp_status, lp_condition

    @staticmethod
    def _kept_gas(n, units, steps, weights) -> float:
        p = n.generators_t.get("p")
        if p is None or not set(units).issubset(p.columns):
            return np.nan
        return float(p[units].iloc[:steps].sum(axis=1).mul(weights.iloc[:steps]).sum())

    def _solve_lp(self, n, units, commitment, kwargs, avail_mw, floor_mw):
        """The window's LP: the commitment held as output bounds, or (None) the plain LP with gas free."""
        gens = n.generators
        saved = {"committable": gens.loc[units, "committable"].copy(), "p_min_pu": gens.loc[units, "p_min_pu"].copy()}
        tv = n.generators_t.get("p_min_pu", pd.DataFrame())
        saved_tv = tv.copy() if tv is not None else pd.DataFrame()
        lp_kwargs = dict(kwargs)
        try:
            gens.loc[units, "committable"] = False
            gens.loc[units, "p_min_pu"] = 0.0
            if saved_tv is not None and not saved_tv.empty:
                n.generators_t["p_min_pu"] = saved_tv.drop(columns=[u for u in units if u in saved_tv.columns])
            if commitment is not None:
                upper = (commitment * avail_mw).clip(lower=0.0)
                lower = (commitment * floor_mw).clip(lower=0.0)
                lower = lower.where(lower <= upper, upper)
                previous = lp_kwargs.pop("extra_functionality", None)

                def fixed_commitment(net, snapshots):
                    if previous is not None:
                        previous(net, snapshots)
                    model = net.model
                    var = model.variables["Generator-p"]
                    dim = [d for d in var.dims if d != "snapshot"][0]
                    snaps = pd.Index(snapshots)
                    sel = var.sel({dim: units, "snapshot": snaps})

                    def da(frame):
                        f = frame.reindex(index=snaps, columns=units).fillna(0.0)
                        return xr.DataArray(f.to_numpy(dtype=float), dims=["snapshot", dim],
                                            coords={"snapshot": snaps, dim: pd.Index(units, name=dim)})

                    model.add_constraints(sel <= da(upper), name="Generator-fixed-commitment-upper")
                    model.add_constraints(sel >= da(lower), name="Generator-fixed-commitment-lower")

                lp_kwargs["extra_functionality"] = fixed_commitment
            t1 = time.perf_counter()
            status, condition = self.original(n, **lp_kwargs)
            return status, condition, time.perf_counter() - t1
        finally:
            gens.loc[units, "committable"] = saved["committable"]
            gens.loc[units, "p_min_pu"] = saved["p_min_pu"]
            n.generators_t["p_min_pu"] = saved_tv

    def summary(self) -> dict:
        frame = pd.DataFrame(self.records)
        if frame.empty:
            return {"method": self.method, "windows": 0}
        out = {"method": self.method, "windows": int(frame["window_start"].nunique()), "solves": len(frame)}
        for col in ("milp_seconds", "lp_seconds", "seconds"):
            if col in frame:
                out[f"{col}_total"] = round(float(frame[col].sum()), 1)
        if "milp_status" in frame:
            out["windows_without_commitment"] = int((~frame["milp_status"].isin(["ok", "reused"])).sum())
            last = frame.drop_duplicates("window_start", keep="last")   # a re-solve's record supersedes
            out["zero_commitment_windows"] = int(last["zero_commitment"].sum())
            out["guard_fallback_windows"] = int(last["guard_fallback"].sum())
            out["guard_fallback_window_starts"] = last.loc[last["guard_fallback"], "window_start"].tolist()
            out["zero_commitment_plain_gas_windows"] = int(last["zero_commitment_plain_gas"].sum())
            out["committed_available_mw_mean"] = round(float(last["committed_available_mw_mean"].mean()), 1)
        return out
