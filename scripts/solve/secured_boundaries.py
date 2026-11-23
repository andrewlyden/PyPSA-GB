"""Secured boundary limits: NESO's sum-of-flow boundary constraints for future years.

For a modelled year with no NESO day-ahead limits (anything after the published
record), the balancing stage can enforce the ETYS *secured* boundary capability
instead: the largest pre-fault transfer across a boundary that meets the NETS
SQSS, as NESO publishes it in the ETYS boundary charts. This is how NESO
represents the transmission network in its own constraint cost modelling, and it
stands in for the N-1 security margin and planned outages that an intact DC
power flow with thermal ratings does not see.

Three sourced inputs:

* boundary geometry: the NESO ETYS boundary GIS (March 2025), converted to
  OSGB36 (``data/network/ETYS/etys_boundaries_mar25_osgb36.geojson``);
* capability: the ``Capability`` rows of the ETYS 2025 boundary chart data
  (``data/network/etys_boundary_capability.csv``), by boundary, FES pathway
  and year; capability already includes the HVDC links that cross the boundary;
* seasonal scaling: NOA methodology (Issue 9.1, July 2023) Table 2.3, "winter
  adjustment factors" used where the TOs give no seasonal capability: spring
  and autumn thermal 85%, summer thermal 80%, summer outage thermal 70%,
  summer outage voltage 90% (of the winter capability).

The branches that cross a boundary are found from the geometry: a line,
transformer or internal DC link crosses when the straight segment between its
bus coordinates cuts the boundary polyline an odd number of times. Each crossing
is signed by the side of the polyline its ``bus0`` lies on, so the signed sum is
the net transfer between the two sides. A node placed on the wrong side changes
that sum only by its own injection. ``include`` and ``exclude`` lists in the
config correct individual crossings.

The limit is enforced in both directions (ETYS publishes one directional
capability; the reverse direction does not bind in practice).

Two opt-in keys for the FES 2050 runs (task Y5), both off by default so every
existing scenario resolves exactly as before:

* ``hold_last_year: true`` takes a boundary's last published capability for a
  later year (the table stops at 2045; decision 10 holds it to 2050);
* ``branch_scaling: {enabled: true, margin: 1.25}`` raises the ratings of the
  AC branches crossing a boundary whose summed rating is below margin x
  capability, so the boundary binds rather than the 2031 circuits
  (``scripts/network_clustering/boundary_branch_scaling.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from shapely.geometry import LineString

DEFAULT_GEOMETRY = "data/network/ETYS/etys_boundaries_mar25_osgb36.geojson"
DEFAULT_CAPABILITY = "data/network/etys_boundary_capability.csv"

# NOA methodology Table 2.3 (fractions of winter capability).
NOA_SEASONAL_FACTORS = {
    "thermal": {"winter": 1.00, "spring_autumn": 0.85, "summer": 0.80, "summer_outage": 0.70},
    "voltage": {"winter": 1.00, "spring_autumn": 1.00, "summer": 1.00, "summer_outage": 0.90},
    "stability": {"winter": 1.00, "spring_autumn": 1.00, "summer": 1.00, "summer_outage": 1.00},
}
# Month -> season. The NOA methodology gives the factors but not the calendar.
# The seasons are aligned to the line-rating calendar the model uses for
# transmission.seasonal_ratings (Northern Powergrid IMP/001/011 Table 1, after
# ENA ER P27 / ACE 104): winter Nov-Feb, spring Mar-Apr, summer May-Aug, autumn
# Sep-Oct. The summer outage condition is a documented assumption: June to
# August, the heart of the transmission outage season inside summer (NESO's
# own 2024-25 day-ahead SCOTEX, SSE-SP2 and HARSPNBLY limits sit at 50-70% of
# their winter level through the summer months), with May at the plain summer
# factor.
DEFAULT_MONTH_SEASON = {
    1: "winter", 2: "winter", 3: "spring_autumn", 4: "spring_autumn", 5: "summer",
    6: "summer_outage", 7: "summer_outage", 8: "summer_outage", 9: "spring_autumn",
    10: "spring_autumn", 11: "winter", 12: "winter",
}


def load_boundary_geometries(path: str | Path = DEFAULT_GEOMETRY) -> Dict[str, "LineString"]:
    """Return {boundary name: shapely LineString in OSGB36 metres}."""
    from shapely.geometry import LineString

    with open(path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    out = {}
    for feat in gj["features"]:
        out[str(feat["properties"]["boundary"])] = LineString(feat["geometry"]["coordinates"])
    return out


def load_capabilities(path: str | Path, year: int, pathway: str,
                      hold_last_year: bool = False) -> pd.DataFrame:
    """Capability table for one year and FES pathway, indexed by boundary.

    With ``hold_last_year``, a year after the last published year for a
    boundary takes that boundary's last published value (plan decision 10 for
    2050: the ETYS table stops at 2045). Without it, a missing year raises.
    """
    cap = pd.read_csv(path)
    if hold_last_year:
        rows = [capability_row(cap, b, year, pathway, True)[0] for b in sorted(set(cap["boundary"]))]
        sel = pd.DataFrame([r for r in rows if r is not None])
    else:
        sel = cap[(cap["year"] == int(year)) & (cap["fes_pathway"] == pathway)]
    if sel.empty:
        raise ValueError(f"No boundary capability rows for year {year}, pathway {pathway} in {path}")
    return sel.set_index("boundary")


def capability_row(cap_all: pd.DataFrame, boundary: str, year: int, pathway: str,
                   hold_last_year: bool = False):
    """One boundary's capability row, and the year it was taken from.

    Args:
        cap_all: the whole capability table.
        boundary: boundary name.
        year: requested year.
        pathway: FES pathway.
        hold_last_year: when the requested year is after the boundary's last
            published year, use that last year (declared; off by default, so
            the lookup is exact as before).

    Returns:
        (row as a Series, or None if absent; year used).
    """
    sel = cap_all[(cap_all["boundary"] == boundary) & (cap_all["fes_pathway"] == pathway)]
    row = sel[sel["year"] == int(year)]
    if not row.empty:
        return row.iloc[0], int(year)
    if hold_last_year and not sel.empty and int(year) > int(sel["year"].max()):
        last = int(sel["year"].max())
        return sel[sel["year"] == last].iloc[0], last
    return None, int(year)


def _polyline_side(geom, point) -> int:
    """+1 when ``point`` lies left of the directed polyline at its nearest segment, else -1."""
    from shapely.geometry import Point

    d = geom.project(Point(point))
    a = np.asarray(geom.interpolate(max(d - 50.0, 0.0)).coords[0])
    b = np.asarray(geom.interpolate(min(d + 50.0, geom.length)).coords[0])
    v, w = b - a, np.asarray(point) - a
    return 1 if v[0] * w[1] - v[1] * w[0] > 0 else -1


def find_boundary_crossings(network, geom) -> pd.DataFrame:
    """Branches whose bus-to-bus segment crosses ``geom`` an odd number of times.

    Returns a frame with columns component (line/transformer/link), name,
    sign (+1 when bus0 is left of the polyline) and rating (s_nom or p_nom).
    Links are internal DC links only (carrier DC, no external bus).
    """
    from shapely.geometry import LineString

    xy = network.buses[["x", "y"]].astype(float)
    links = network.links
    if len(links):
        ext = links.bus0.astype(str).str.contains("External") | links.bus1.astype(str).str.contains("External")
        links = links[(links.get("carrier", pd.Series("", index=links.index)) == "DC") & ~ext]
    rows = []
    for comp, df, rating in (("line", network.lines, "s_nom"), ("transformer", network.transformers, "s_nom"),
                             ("link", links, "p_nom")):
        for name, r in df.iterrows():
            if r.bus0 not in xy.index or r.bus1 not in xy.index:
                continue
            p0, p1 = tuple(xy.loc[r.bus0]), tuple(xy.loc[r.bus1])
            if p0 == p1:
                continue
            inter = LineString([p0, p1]).intersection(geom)
            k = 0 if inter.is_empty else (len(inter.geoms) if hasattr(inter, "geoms") else 1)
            if k % 2 == 1:
                rows.append({"component": comp, "name": name, "sign": _polyline_side(geom, p0),
                             "rating": float(r[rating])})
    return pd.DataFrame(rows, columns=["component", "name", "sign", "rating"])


def seasonal_factor_series(snapshots, limit_type: str, factors: dict, month_season: dict) -> pd.Series:
    """Per-snapshot fraction of the winter capability for a boundary's limiting type."""
    table = factors.get(limit_type, factors.get("thermal"))
    months = pd.DatetimeIndex(snapshots).month
    return pd.Series([float(table[month_season[int(m)]]) for m in months], index=snapshots)


def secured_boundary_config(outage_cfg: dict, scenario_config: dict) -> dict:
    """Resolve the ``secured_capability`` sub-config with its defaults."""
    cfg = dict(outage_cfg.get("secured_capability", {}) or {})
    year = cfg.get("capability_year") or scenario_config.get("modelled_year")
    if year is None:
        raise ValueError("secured_capability needs capability_year or the scenario's modelled_year")
    cfg["capability_year"] = int(year)
    cfg.setdefault("fes_pathway", "HT")
    cfg.setdefault("geometry_file", DEFAULT_GEOMETRY)
    cfg.setdefault("capability_file", DEFAULT_CAPABILITY)
    cfg.setdefault("boundaries", None)             # None = every boundary with a capability row
    cfg["capability_year_overrides"] = dict(cfg.get("capability_year_overrides", {}) or {})
    cfg["include"] = dict(cfg.get("include", {}) or {})
    cfg["exclude"] = dict(cfg.get("exclude", {}) or {})
    factors = {k: dict(v) for k, v in NOA_SEASONAL_FACTORS.items()}
    for k, v in (cfg.get("seasonal_factors", {}) or {}).items():
        factors.setdefault(k, {}).update(v)
    cfg["seasonal_factors"] = factors
    ms = dict(DEFAULT_MONTH_SEASON)
    ms.update({int(k): str(v) for k, v in (cfg.get("month_season", {}) or {}).items()})
    cfg["month_season"] = ms
    cfg.setdefault("capability_scale", 1.0)
    # 2050 (task Y5, decision 10): a capability year after the last published
    # one takes the last published value. Off by default: the lookup stays exact.
    cfg["hold_last_year"] = bool(cfg.get("hold_last_year", False))
    # 2050 (task Y5): raise crossing-branch ratings below margin x capability so
    # the boundary binds, not the 2031 circuits. Off by default; see
    # scripts/network_clustering/boundary_branch_scaling.py.
    cfg["branch_scaling"] = dict(cfg.get("branch_scaling", {}) or {})
    # Historic outage derate (9 Oct 2026): NESO's day-ahead limits as a fraction of the intact winter limit,
    # in place of the NOA factor on the boundaries the table covers. mode replay = the reference year as it
    # happened; forward = the derate beyond the NOA factor scaled by NA1's target-year construction outage
    # days (scripts/solve/build_boundary_historic_derate.py). Off by default.
    cfg["historic_derate"] = dict(cfg.get("historic_derate", {}) or {})
    # Capability phase-in (9 Oct 2026): before ``from`` every boundary without a capability_year_overrides
    # entry takes its ``before_year`` capability, so projects dated in the capability year arrive on that
    # date, not on 1 January. Off by default.
    cfg["phase_in"] = dict(cfg.get("phase_in", {}) or {})
    return cfg


def historic_derate_series(snapshots, noa_factor: pd.Series, table: pd.DataFrame, mode: str) -> pd.Series:
    """Per-snapshot factor for one boundary from its rows of the historic derate table."""
    if mode not in ("replay", "forward"):
        raise ValueError(f"historic_derate.mode must be replay or forward, got {mode!r}")
    idx = pd.DatetimeIndex(snapshots)
    key = pd.MultiIndex.from_arrays([idx.month, idx.day, idx.hour])
    lookup = table.set_index(["month", "day", "hour"])["factor"]
    # 29 February is not in the table: it takes 28 February's hours.
    feb29 = (idx.month == 2) & (idx.day == 29)
    if feb29.any():
        key = pd.MultiIndex.from_arrays([idx.month, np.where(feb29, 28, idx.day), idx.hour])
    hist = pd.Series(lookup.reindex(key).to_numpy(), index=snapshots)
    if hist.isna().any():
        raise ValueError(f"historic_derate table misses {int(hist.isna().sum())} snapshot hours")
    if mode == "replay":
        return hist
    ratio = float(table["na1_ratio"].iloc[0])
    noa = noa_factor.reindex(snapshots)
    deeper = (noa - hist).clip(lower=0.0)
    return (hist.where(hist >= noa, noa - deeper * ratio)).astype(float)


def _boundary_crossings(network, b: str, geom, cfg: dict) -> pd.DataFrame:
    """Crossing branches of boundary ``b`` after the config's exclude/include corrections."""
    cross = find_boundary_crossings(network, geom)
    drop = set(cfg["exclude"].get(b, []) or [])
    cross = cross[~cross["name"].isin(drop)]
    for extra in cfg["include"].get(b, []) or []:
        comp, name, sign = extra["component"], extra["name"], float(extra.get("sign", 1.0))
        table = {"line": network.lines, "transformer": network.transformers, "link": network.links}[comp]
        if name in table.index and name not in set(cross["name"]):
            rating = float(table.at[name, "p_nom" if comp == "link" else "s_nom"])
            cross = pd.concat([cross, pd.DataFrame([{"component": comp, "name": name, "sign": sign,
                                                     "rating": rating}])], ignore_index=True)
    return cross


def _scale_branches_to_capability(network, cfg: dict, cap_all: pd.DataFrame, geoms: dict,
                                  wanted: List[str], log: logging.Logger) -> pd.DataFrame:
    """Apply the opt-in ``branch_scaling`` (task Y5) before the specs are built.

    Scales the ratings of AC branches crossing a boundary whose summed rating is
    below margin x capability (the winter limit the specs enforce: capability
    x ``capability_scale``, before the seasonal factor). The per-boundary report is
    logged and returned.
    """
    from scripts.network_clustering.boundary_branch_scaling import scale_boundary_branches

    crossings, capability = {}, {}
    for b in wanted:
        year = int(cfg["capability_year_overrides"].get(b, cfg["capability_year"]))
        row, _ = capability_row(cap_all, b, year, cfg["fes_pathway"], cfg["hold_last_year"])
        if row is None or b not in geoms:
            continue   # the main loop raises for these
        crossings[b] = _boundary_crossings(network, b, geoms[b], cfg)
        capability[b] = float(row["capability_mw"]) * float(cfg["capability_scale"])
    sides = None
    if cfg["branch_scaling"].get("internal_floor_fraction") is not None:
        from scripts.network_clustering.boundary_busmap import boundary_sides

        sides = boundary_sides(network.buses[["x", "y"]].astype(float), geoms, list(crossings))
    return scale_boundary_branches(network, crossings, pd.Series(capability, dtype=float),
                                   cfg["branch_scaling"], sides=sides, log=log)


def build_secured_boundary_specs(network, outage_cfg: dict, scenario_config: dict,
                                 logger: Optional[logging.Logger] = None) -> List[dict]:
    """Aggregate-boundary constraint specs (the format the NESO callback consumes).

    One spec per boundary: signed lines/transformers/links and a per-snapshot
    limit = capability(year) x seasonal factor x capability_scale.
    """
    log = logger or logging.getLogger(__name__)
    cfg = secured_boundary_config(outage_cfg, scenario_config)
    geoms = load_boundary_geometries(cfg["geometry_file"])
    cap_all = pd.read_csv(cfg["capability_file"])
    wanted = cfg["boundaries"] or sorted(set(cap_all["boundary"]))
    snapshots = network.snapshots
    if cfg["branch_scaling"].get("enabled", False):
        _scale_branches_to_capability(network, cfg, cap_all, geoms, wanted, log)
    specs = []
    hist_table = None
    for b in wanted:
        year = int(cfg["capability_year_overrides"].get(b, cfg["capability_year"]))
        row, used_year = capability_row(cap_all, b, year, cfg["fes_pathway"], cfg["hold_last_year"])
        if row is None:
            raise ValueError(f"secured_capability: no capability for boundary {b}, year {year}, "
                             f"pathway {cfg['fes_pathway']} in {cfg['capability_file']}")
        if b not in geoms:
            raise ValueError(f"secured_capability: boundary {b} has no geometry in {cfg['geometry_file']}")
        if used_year != year:
            log.warning(f"  secured boundary {b}: no capability published for {year}; holding the last "
                        f"published year {used_year} (hold_last_year, declared)")
            year = used_year
        cross = _boundary_crossings(network, b, geoms[b], cfg)
        if cross.empty:
            log.warning(f"  secured boundary {b}: no crossing branches found, skipped")
            continue
        limit_type = str(row.get("limit_type", "thermal"))
        factor = seasonal_factor_series(snapshots, limit_type, cfg["seasonal_factors"], cfg["month_season"])
        hd = cfg["historic_derate"]
        if hd.get("enabled", False):
            if hist_table is None:
                hist_table = pd.read_csv(hd["file"])
            rows = hist_table[hist_table["boundary"] == b]
            if rows.empty and str(hd.get("unmapped", "noa")) == "default":
                # No day-ahead history for this boundary: the mean outage share of its limit type.
                rows = hist_table[hist_table["boundary"] == f"_default_{limit_type}"]
                if rows.empty:
                    rows = hist_table[hist_table["boundary"] == "_default_thermal"]
            if len(rows):
                noa_mean = float(factor.mean())
                factor = historic_derate_series(snapshots, factor, rows, str(hd.get("mode", "replay")))
                log.info(f"  secured boundary {b}: historic derate ({hd.get('mode', 'replay')}, "
                         f"{rows['group'].iloc[0]}) mean factor {factor.mean():.3f} against NOA {noa_mean:.3f}")
        capability = pd.Series(float(row["capability_mw"]), index=snapshots)
        pi = cfg["phase_in"]
        if pi.get("enabled", False) and b not in cfg["capability_year_overrides"]:
            before_row, before_year = capability_row(cap_all, b, int(pi["before_year"]), cfg["fes_pathway"],
                                                     cfg["hold_last_year"])
            if before_row is None:
                raise ValueError(f"secured_capability.phase_in: no capability for {b} in {pi['before_year']}")
            early = pd.DatetimeIndex(snapshots) < pd.Timestamp(pi["from"])
            capability[early] = float(before_row["capability_mw"])
            if early.any():
                log.info(f"  secured boundary {b}: phase-in, {before_year} capability "
                         f"{float(before_row['capability_mw']):,.0f} MW before {pi['from']} ({int(early.sum())} snapshots)")
        limit = capability * float(cfg["capability_scale"]) * factor
        by = cross.groupby(["component", cross["sign"] > 0])["name"].apply(list)

        def pick(comp, positive):
            return list(by.get((comp, positive), []))

        spec = {
            "boundary_name": f"ETYS_{b}",
            "positive_lines": pick("line", True), "negative_lines": pick("line", False),
            "positive_transformers": pick("transformer", True),
            "negative_transformers": pick("transformer", False),
            "links": [{"name": r["name"], "sign": float(r["sign"])} for _, r in cross[cross.component == "link"].iterrows()],
            "limit_series": limit,
            "soft_penalty_gbp_per_mwh": None,
            "soft_max_violation_mw": None,
        }
        soft = (outage_cfg.get("secured_capability", {}) or {}).get("soft_boundaries", {}) or {}
        if isinstance(soft.get(b), dict) and soft[b].get("enabled", True):
            spec["soft_penalty_gbp_per_mwh"] = float(soft[b].get("penalty_gbp_per_mwh", 1000.0))
            mv = soft[b].get("max_violation_mw")
            spec["soft_max_violation_mw"] = np.inf if mv is None else max(0.0, float(mv))
        ac = cross[cross.component != "link"]["rating"].clip(upper=9998).sum()
        dc = cross[cross.component == "link"]["rating"].sum()
        log.info(f"  secured boundary {b}: {len(cross)} crossings (AC rating {ac:,.0f} MVA, DC {dc:,.0f} MW), "
                 f"capability {row['capability_mw']:,.0f} MW ({year}, {cfg['fes_pathway']}, {limit_type}), "
                 f"limit {limit.min():,.0f}-{limit.max():,.0f} MW over the snapshots")
        specs.append(spec)
    return specs
