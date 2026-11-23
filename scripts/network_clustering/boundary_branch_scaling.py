"""Make the ETYS boundaries bind, not the 2031 circuits (task Y5, method step 3).

The 2050 network is the ETYS 2031 topology (the last year Appendix B
publishes), clustered. Its crossing branches carry 2031 circuit ratings,
while the boundary limits are the NESO secured capabilities for 2045 held to
2050 (decision 10), which assume the Beyond 2030 reinforcements. Where the
summed rating of the branches crossing a boundary is below that capability,
the circuits bind before the boundary does and the model reports congestion
on individual branches that NESO's plan has reinforced.

:func:`scale_boundary_branches` raises the rating of every AC branch that
crosses such a boundary by one factor per boundary, so that

    sum(AC crossing rating) + sum(DC crossing p_nom) >= margin x capability.

DC links keep their ratings (they are named projects, and ETYS capability
already counts them). A branch that crosses several boundaries takes the
largest of their factors, so every boundary meets its target. The factor is
applied to the rating the branch had before any scaling (kept in
``s_nom_pre_boundary_scaling``), so calling the function twice gives the same
network. Reinforcement is represented as added parallel circuits: line
reactance and resistance fall by the same factor and susceptance rises by it
(``scale_impedance``, default true). Transformer impedance is per unit on
``s_nom``, so a larger ``s_nom`` already lowers it in proportion. Seasonal
rating columns (``s_nom_winter`` etc.) scale with ``s_nom`` so the seasonal
factors are unchanged.

Optionally (``internal_floor_fraction``, default off) branches with both ends
on the same side of every boundary ("inside a boundary region") are raised to
at least that fraction of the smallest capability among the boundaries their
region touches. It is off by default because internal branches carry local
flows and no NESO figure sets their rating.

Branches at or above ``relaxed_rating_mva`` (9,999 MVA couplers and the BM
voltage floor) are left alone and counted at their rating.

It is applied only when a scenario asks for it:
``transmission.outage_schedule.secured_capability.branch_scaling.enabled: true``
(see ``scripts/solve/secured_boundaries.py``), which is meant for the
``HT50_*`` scenarios. Nothing else calls it.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MARGIN = 1.25
RELAXED_RATING_MVA = 9999.0
PRE_SCALING_COLUMN = "s_nom_pre_boundary_scaling"
SEASON_COLUMNS = ("s_nom_winter", "s_nom_spring", "s_nom_summer", "s_nom_autumn")


def branch_scaling_config(cfg: Optional[dict]) -> dict:
    """Resolve the ``branch_scaling`` options with their defaults.

    Args:
        cfg: the ``branch_scaling`` sub-config, or ``None``.

    Returns:
        dict with ``enabled``, ``margin``, ``scale_impedance``,
        ``internal_floor_fraction`` and ``relaxed_rating_mva``.
    """
    out = dict(cfg or {})
    out.setdefault("enabled", False)
    out["margin"] = float(out.get("margin", DEFAULT_MARGIN))
    out["scale_impedance"] = bool(out.get("scale_impedance", True))
    frac = out.get("internal_floor_fraction")
    out["internal_floor_fraction"] = None if frac is None else float(frac)
    out["relaxed_rating_mva"] = float(out.get("relaxed_rating_mva", RELAXED_RATING_MVA))
    if out["margin"] <= 0:
        raise ValueError("branch_scaling.margin must be positive")
    return out


def _original_ratings(network, component: str) -> pd.Series:
    """s_nom before any boundary scaling (records it on first call)."""
    df = getattr(network, component)
    if PRE_SCALING_COLUMN not in df.columns:
        df[PRE_SCALING_COLUMN] = df["s_nom"].astype(float)
    return df[PRE_SCALING_COLUMN].astype(float).fillna(df["s_nom"].astype(float))


def _apply_factor(network, component: str, factors: pd.Series, scale_impedance: bool) -> None:
    """Set s_nom (and seasonal columns, line impedances) to original x factor."""
    df = getattr(network, component)
    orig = _original_ratings(network, component)
    f = factors.reindex(df.index).fillna(1.0).clip(lower=1.0)
    current = df["s_nom"].astype(float)
    df["s_nom"] = orig * f
    ratio = (df["s_nom"] / current.where(current > 0)).fillna(1.0)
    for col in SEASON_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * ratio
    if component == "lines" and scale_impedance:
        for col in ("x", "r"):
            if col in df.columns:
                df[col] = df[col].astype(float) / ratio
        for col in ("b", "g"):
            if col in df.columns:
                df[col] = df[col].astype(float) * ratio


def scale_boundary_branches(network, crossings: Dict[str, pd.DataFrame], capability: pd.Series,
                            config: Optional[dict] = None,
                            sides: Optional[pd.DataFrame] = None,
                            log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Raise crossing-branch ratings so each boundary, not its circuits, binds.

    Args:
        network: PyPSA network, modified in place.
        crossings: boundary -> frame of crossing branches as returned by
            ``find_boundary_crossings`` (component, name, sign, rating), after
            any include/exclude corrections.
        capability: boundary -> capability (MW), the limit the solve enforces
            (before the seasonal factor).
        config: ``branch_scaling`` options (see :func:`branch_scaling_config`).
        sides: optional bus x boundary side frame (0/1), needed only for the
            internal floor.
        log: logger.

    Returns:
        per-boundary report: capability_mw, target_mw, ac_before_mva,
        ac_after_mva, dc_mw, factor, before_ratio, after_ratio, scaled
        (branch count).
    """
    log = log or logger
    cfg = branch_scaling_config(config)
    comp_of = {"line": "lines", "transformer": "transformers"}
    relaxed = cfg["relaxed_rating_mva"]
    orig = {c: _original_ratings(network, c) for c in comp_of.values()}
    branch_factor = {c: pd.Series(1.0, index=getattr(network, c).index) for c in comp_of.values()}
    rows = []
    for b, cross in crossings.items():
        if b not in capability.index:
            continue
        cap = float(capability[b])
        ac = cross[cross["component"].isin(comp_of)]
        dc = float(cross.loc[cross["component"] == "link", "rating"].sum())
        ac_orig = pd.Series([float(orig[comp_of[c]].get(n, np.nan)) for c, n in zip(ac["component"], ac["name"])],
                            index=ac.index).fillna(0.0)
        fixed = ac_orig >= relaxed
        scalable = float(ac_orig[~fixed].sum())
        target = cfg["margin"] * cap
        need = target - dc - float(ac_orig[fixed].sum())
        factor = 1.0
        if scalable > 0 and need > scalable:
            factor = need / scalable
            for (c, n), is_fixed in zip(zip(ac["component"], ac["name"]), fixed):
                if not is_fixed:
                    s = branch_factor[comp_of[c]]
                    s[n] = max(s[n], factor)
        rows.append({"boundary": b, "capability_mw": cap, "target_mw": target,
                     "ac_before_mva": float(ac_orig.sum()), "dc_mw": dc, "factor": factor,
                     "scaled": int((~fixed).sum()) if factor > 1.0 else 0})
    internal = 0
    if cfg["internal_floor_fraction"] is not None and sides is not None and len(rows):
        internal = _internal_floor(network, sides, capability, crossings, cfg, branch_factor, orig)
    for c in comp_of.values():
        _apply_factor(network, c, branch_factor[c], cfg["scale_impedance"])
    report = pd.DataFrame(rows).set_index("boundary") if rows else pd.DataFrame(
        columns=["capability_mw", "target_mw", "ac_before_mva", "dc_mw", "factor", "scaled"])
    after = []
    for b, cross in crossings.items():
        if b not in report.index:
            continue
        ac = cross[cross["component"].isin(comp_of)]
        after.append(sum(float(getattr(network, comp_of[c]).at[n, "s_nom"]) for c, n in zip(ac["component"], ac["name"])))
    report["ac_after_mva"] = after
    report["before_ratio"] = (report["ac_before_mva"] + report["dc_mw"]) / report["capability_mw"]
    report["after_ratio"] = (report["ac_after_mva"] + report["dc_mw"]) / report["capability_mw"]
    report.attrs["internal_branches_raised"] = internal
    n_scaled = int((report["factor"] > 1.0).sum())
    log.info(f"  boundary branch scaling (margin {cfg['margin']:.2f}): {n_scaled} of {len(report)} "
             f"boundaries below margin x capability; crossing branches raised"
             + (f"; {internal} internal branches raised to the floor" if internal else ""))
    for b, r in report.iterrows():
        log.info(f"    {b}: AC {r.ac_before_mva:,.0f} -> {r.ac_after_mva:,.0f} MVA, DC {r.dc_mw:,.0f} MW, "
                 f"capability {r.capability_mw:,.0f} MW (ratio {r.before_ratio:.2f} -> {r.after_ratio:.2f})")
    return report


def _internal_floor(network, sides: pd.DataFrame, capability: pd.Series,
                    crossings: Dict[str, pd.DataFrame], cfg: dict,
                    branch_factor: Dict[str, pd.Series], orig: Dict[str, pd.Series]) -> int:
    """Raise internal branches of each boundary region to the declared floor."""
    tables = {"line": network.lines, "transformer": network.transformers, "link": network.links}
    code = sides.astype(str).agg("".join, axis=1)
    crossing_names = set()
    touches: Dict[str, set] = {}          # region code -> boundaries its crossing branches cross
    for b, cross in crossings.items():
        crossing_names |= set(cross["name"])
        for comp, name in zip(cross["component"], cross["name"]):
            t = tables[comp]
            for bus in (t.at[name, "bus0"], t.at[name, "bus1"]):
                if bus in code.index:
                    touches.setdefault(code[bus], set()).add(b)
    raised = 0
    for comp in ("lines", "transformers"):
        df = getattr(network, comp)
        for name, r in df.iterrows():
            if name in crossing_names or r.bus0 not in code.index or r.bus1 not in code.index:
                continue
            if code[r.bus0] != code[r.bus1]:
                continue
            region_caps = [float(capability[b]) for b in touches.get(code[r.bus0], ())
                           if b in capability.index]
            if not region_caps:
                continue
            floor = cfg["internal_floor_fraction"] * min(region_caps)
            base = float(orig[comp][name])
            if 0 < base < floor and base < cfg["relaxed_rating_mva"]:
                branch_factor[comp][name] = max(branch_factor[comp][name], floor / base)
                raised += 1
    return raised
