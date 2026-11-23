#!/usr/bin/env python3
"""Audit the secured-boundary capability each solved balancing family resolves to (19 Oct 2026).

Report only, from the config alone (no network). For every family in ``run_scenarios`` (the month scenarios
``<head>_<Mon><year>_<tag>`` grouped as ``<head>_<tag>``; any other market scenario is its own family) it writes one
row per boundary with the capability the balancing stage will enforce, resolved exactly as the solve resolves it:
the base ``network.transmission.outage_schedule.secured_capability`` with the keys of
``market.balancing.network_delivery.secured_capability`` replacing its own (``network_delivery.apply_network_delivery``,
keys ``network_delivery.SECURED_CAPABILITY_KEYS``), then ``secured_boundaries.secured_boundary_config``; the
capability row is ``secured_boundaries.capability_row`` of the capability file (per-boundary year overrides,
``hold_last_year``). Columns: the case (central / no_acceleration / seven_late / no_upgrades), capability year asked
and used, capability (MW), limit type, the NOA seasonal factors of that type, historic derate (on/off, mode, the
table's group for the boundary), the mean factor and mean limit over the family's own hours (phase-in included),
the capability file.

Flags (WARN, never a failed run):

* a delay family's capability above its central twin's on any boundary (the twin is the same family with its
  network case swapped for the central one, ``no_upgrades_p2_bids`` -> ``cp30_network_p2_bids``; when that is no
  scenario, the central family ``<head>_<central tag>``, ``cp30_network`` by default: config.yaml ``secured_capability_audit_central``);
* a family reading another capability file than the central family;
* a boundary with no capability row or no geometry (the solve would raise);
* months of one family resolving differently.

Rule ``audit_secured_capability`` (rules/CP30.smk) -> ``resources/analysis/secured_capability_by_family.csv``.
Its params are the resolved settings only (:func:`audit_spec`), so it reruns when a capability setting changes,
not on every config edit.

    python scripts/solve/secured_capability_audit.py [--out resources/analysis/secured_capability_by_family.csv]
"""
# No `from __future__` import: Snakemake's script directive prepends a preamble.

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: <head>_<Mon>_<tag> (CP30_FFR_2030_Jan_no_acceleration), or the older <head>_<Mon><year>_<tag>.
FAMILY_MONTH_RE = re.compile(r"^(?P<head>.+?)_(?P<mon>" + "|".join(MONTHS) + r")(?P<year>\d{4})?_(?P<tag>.+)$")
#: Delay case keyword -> the substrings that name it in a network_delivery case or a family tag.
DELAY_CASES = {"no_acceleration": ("no_acceleration",), "seven_late": ("seven_late", "seven_critical_late"),
               "no_upgrades": ("no_upgrades",)}
DEFAULT_CENTRAL_TAG = "cp30_network"
#: Settings of the resolved secured_capability block that move a capability or its limit series.
RESOLVED_KEYS = ("capability_year", "capability_year_overrides", "capability_file", "fes_pathway", "boundaries",
                 "geometry_file", "seasonal_factors", "month_season", "capability_scale", "hold_last_year",
                 "historic_derate", "phase_in", "soft_boundaries")
SEASONS = ("winter", "spring_autumn", "summer", "summer_outage")
COLUMNS = ["family", "tag", "months", "case", "network_delivery_case", "central_twin", "secured_active", "boundary",
           "capability_year", "capability_year_used", "capability_mw", "limit_type", "fes_pathway", "capability_scale",
           *[f"noa_{s}" for s in SEASONS], "historic_derate", "historic_derate_mode", "historic_derate_group",
           "mean_factor", "mean_limit_mw", "min_limit_mw", "phase_in_capability_mw", "phase_in_from",
           "twin_capability_mw", "central_capability_file", "capability_file", "historic_derate_file",
           "geometry_file", "status", "note"]


def delay_case(case: str | None, tag: str = "") -> str | None:
    """The delay keyword of a network_delivery case or family tag (None for a central family)."""
    for text in (str(case or ""), str(tag or "")):
        for key, names in DELAY_CASES.items():
            if any(n in text for n in names):
                return key
    return None


def twin_tag(tag: str, central_tag: str = DEFAULT_CENTRAL_TAG) -> str:
    """The same family on the central network: a leading network case swapped for the central family's
    (``no_upgrades_p2_bids`` -> ``cp30_network_p2_bids``), or a trailing one dropped (an older family,
    ``fin5_no_upgrades`` -> ``fin5``); the bare network case (``no_upgrades``) has none ("")."""
    for names in DELAY_CASES.values():
        for n in names:
            if tag == n:
                return ""
            if tag.startswith(n + "_"):
                return f"{central_tag}{tag[len(n):]}"
            if tag.endswith("_" + n):
                return tag[: -len(n) - 1]
    return tag


def family_of(scenario_id: str) -> tuple[str, str, str | None, str | None]:
    """(family, tag, head, month) of a scenario id; a scenario outside the month pattern is its own family."""
    m = FAMILY_MONTH_RE.match(str(scenario_id))
    if m is None:
        return str(scenario_id), "", None, None
    return f"{m['head']}_{m['tag']}", m["tag"], m["head"], m["mon"]


def _json_safe(value):
    """Params must survive Snakemake's comparison: dict keys as strings, tuples as lists."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def resolve_secured_capability(scenario: dict) -> dict:
    """The secured-capability settings the balancing stage of ``scenario`` enforces.

    As the solve does it: ``network_delivery_config`` validates the case block, its ``secured_capability`` keys
    replace the base outage block's own (``apply_network_delivery``), then ``secured_boundary_config``.

    Returns:
        {"active": outage schedule on with source secured_capability, "case": the network_delivery case,
         "secured": the resolved settings (``RESOLVED_KEYS``)}.
    """
    from scripts.market.network_delivery import network_delivery_config
    from scripts.solve.secured_boundaries import secured_boundary_config

    balancing = ((scenario.get("market") or {}).get("balancing") or {})
    outage = copy.deepcopy(((((scenario.get("network") or {}).get("transmission")) or {}).get("outage_schedule") or {}))
    block = network_delivery_config({"network_delivery": copy.deepcopy(balancing.get("network_delivery"))})
    cap_block = (block or {}).get("secured_capability") or {}
    if cap_block:
        secured = copy.deepcopy(outage.get("secured_capability") or {})
        for key, value in cap_block.items():
            secured[key] = copy.deepcopy(value)
        outage["secured_capability"] = secured
    active = bool(outage.get("enabled", False)) and str(outage.get("source", "csv")) == "secured_capability"
    resolved = None
    if active:
        cfg = secured_boundary_config(outage, {"modelled_year": scenario.get("modelled_year")})
        cfg["soft_boundaries"] = (outage.get("secured_capability") or {}).get("soft_boundaries") or {}
        resolved = _json_safe({k: cfg.get(k) for k in RESOLVED_KEYS})
    return {"active": active, "case": (block or {}).get("case"), "secured": resolved}


def _period(scenario: dict) -> list[str] | None:
    sp = scenario.get("solve_period") or {}
    if sp.get("enabled") and sp.get("start") and sp.get("end"):
        return [str(sp["start"]), str(sp["end"])]
    return None


def audit_spec(scenarios: dict, run_ids: list[str], central_tag: str = DEFAULT_CENTRAL_TAG) -> dict:
    """The resolved settings of every market family in ``run_ids`` (rule params), with their twins and central.

    Returns:
        {"central_tag": ..., "families": {family: {"tag", "head", "in_run", "months": [{"month", "scenario",
        "period", "timestep_minutes", "case", "active", "secured"}], "delay_case", "twin", "central"}}}
    """
    def months_of(family_ids):
        out = []
        for sid in family_ids:
            sc = scenarios[sid]
            r = resolve_secured_capability(sc)
            out.append({"month": family_of(sid)[3], "scenario": sid, "period": _period(sc),
                        "timestep_minutes": int(sc.get("timestep_minutes") or 60), "case": r["case"],
                        "active": r["active"], "secured": r["secured"]})
        return out

    def ids_for(head, tag):
        """Month scenarios of <head>_<Mon><year>_<tag> known to the config, Jan to Dec."""
        found = {}
        for sid in scenarios:
            m = FAMILY_MONTH_RE.match(str(sid))
            if m and m["head"] == head and m["tag"] == tag:
                found.setdefault(m["mon"], sid)
        return [found[mon] for mon in MONTHS if mon in found]

    groups: dict = {}
    for sid in run_ids:
        sc = scenarios.get(sid) or {}
        if not (sc.get("market") or {}).get("enabled", False):
            continue
        fam, tag, head, _ = family_of(sid)
        groups.setdefault(fam, {"tag": tag, "head": head, "ids": []})["ids"].append(sid)
    families: dict = {}

    def add(fam, tag, head, ids, in_run):
        if fam in families:
            families[fam]["in_run"] = families[fam]["in_run"] or in_run
            return
        order = {m: i for i, m in enumerate(MONTHS)}
        ids = sorted(ids, key=lambda s: order.get(family_of(s)[3], 99))
        families[fam] = {"tag": tag, "head": head, "in_run": in_run, "months": months_of(ids)}

    for fam, g in groups.items():
        add(fam, g["tag"], g["head"], g["ids"], True)
    for fam, g in list(groups.items()):
        entry = families[fam]
        case = next((m["case"] for m in entry["months"] if m["case"]), None)
        entry["delay_case"] = delay_case(case, g["tag"])
        central = f"{g['head']}_{central_tag}" if g["head"] else None
        if central and central not in families:
            ids = ids_for(g["head"], central_tag)
            if ids:
                add(central, central_tag, g["head"], ids, False)
        entry["central"] = central if central in families else None
        entry["twin"] = None
        if entry["delay_case"]:
            base = twin_tag(g["tag"], central_tag)
            twin = f"{g['head']}_{base}" if (g["head"] and base) else None
            ids = ids_for(g["head"], base) if twin else []
            if twin and (twin in families or ids):
                add(twin, base, g["head"], ids, False)
                entry["twin"], entry["twin_basis"] = twin, "tag without the delay suffix"
            else:
                entry["twin"], entry["twin_basis"] = entry["central"], f"central family ({central_tag})"
    for fam, entry in families.items():
        entry.setdefault("delay_case", delay_case(next((m["case"] for m in entry["months"] if m["case"]), None),
                                                  entry["tag"]))
        entry.setdefault("central", None)
        entry.setdefault("twin", None)
    return {"central_tag": central_tag, "families": families}


def spec_files(spec: dict) -> dict:
    """The files the audit reads (rule inputs): capability tables, geometries, historic derate tables."""
    out = {"capability": set(), "geometry": set(), "historic_derate": set()}
    for entry in spec["families"].values():
        for m in entry["months"]:
            s = m["secured"]
            if not s:
                continue
            out["capability"].add(s["capability_file"])
            out["geometry"].add(s["geometry_file"])
            if (s.get("historic_derate") or {}).get("enabled") and s["historic_derate"].get("file"):
                out["historic_derate"].add(s["historic_derate"]["file"])
    return {k: sorted(v) for k, v in out.items()}


# ── The audit ───────────────────────────────────────────────────────────────
def _snapshots(months: list[dict], year: int | None) -> pd.DatetimeIndex:
    parts = []
    for m in months:
        freq = f"{int(m.get('timestep_minutes') or 60)}min"
        if m.get("period"):
            parts.append(pd.date_range(m["period"][0], m["period"][1], freq=freq))
        elif m.get("month") and year:
            start = pd.Timestamp(year=int(year), month=MONTHS.index(m["month"]) + 1, day=1)
            parts.append(pd.date_range(start, start + pd.offsets.MonthBegin(1), freq=freq, inclusive="left"))
    return parts[0].append(parts[1:]) if parts else pd.DatetimeIndex([])


def boundary_rows(secured: dict, snapshots: pd.DatetimeIndex, cap_all: pd.DataFrame, geometry_names: set,
                  hist: pd.DataFrame | None) -> list[dict]:
    """One record per boundary of a resolved secured_capability block (capability, factors, mean limit)."""
    from scripts.solve.secured_boundaries import capability_row, historic_derate_series, seasonal_factor_series

    month_season = {int(k): str(v) for k, v in secured["month_season"].items()}
    factors = secured["seasonal_factors"]
    pathway = secured["fes_pathway"]
    sel = cap_all[cap_all["fes_pathway"] == pathway]
    wanted = secured["boundaries"] or sorted(set(sel["boundary"]))
    hd = secured.get("historic_derate") or {}
    pi = secured.get("phase_in") or {}
    overrides = secured.get("capability_year_overrides") or {}
    rows = []
    for b in wanted:
        year = int(overrides.get(b, secured["capability_year"]))
        row, used = capability_row(cap_all, b, year, pathway, bool(secured.get("hold_last_year")))
        rec = {"boundary": b, "capability_year": year, "capability_year_used": used, "fes_pathway": pathway,
               "capability_scale": float(secured.get("capability_scale", 1.0)),
               "capability_file": secured["capability_file"], "geometry_file": secured["geometry_file"],
               "historic_derate": "on" if hd.get("enabled") else "off",
               "historic_derate_mode": hd.get("mode", "replay") if hd.get("enabled") else "",
               "historic_derate_file": hd.get("file", "") if hd.get("enabled") else "", "problems": []}
        if b not in geometry_names:
            rec["problems"].append("no geometry (the solve would raise)")
        if row is None:
            rec["problems"].append(f"no capability row for {year} {pathway} (the solve would raise)")
            rows.append(rec)
            continue
        limit_type = str(row.get("limit_type", "thermal"))
        table = factors.get(limit_type, factors.get("thermal"))
        rec.update({"capability_mw": float(row["capability_mw"]), "limit_type": limit_type,
                    **{f"noa_{s}": float(table[s]) for s in SEASONS}})
        if len(snapshots):
            factor = seasonal_factor_series(snapshots, limit_type, factors, month_season)
            if hd.get("enabled") and hist is not None:
                rows_b = hist[hist["boundary"] == b]
                if rows_b.empty and str(hd.get("unmapped", "noa")) == "default":
                    rows_b = hist[hist["boundary"] == f"_default_{limit_type}"]
                    if rows_b.empty:
                        rows_b = hist[hist["boundary"] == "_default_thermal"]
                if len(rows_b):
                    factor = historic_derate_series(snapshots, factor, rows_b, str(hd.get("mode", "replay")))
                    rec["historic_derate_group"] = str(rows_b["group"].iloc[0])
                else:
                    rec["historic_derate_group"] = "unmapped: NOA factor"
            capability = pd.Series(float(row["capability_mw"]), index=snapshots)
            if pi.get("enabled") and b not in overrides:
                before, before_year = capability_row(cap_all, b, int(pi["before_year"]), pathway,
                                                     bool(secured.get("hold_last_year")))
                if before is None:
                    rec["problems"].append(f"phase_in: no capability for {pi['before_year']}")
                else:
                    early = snapshots < pd.Timestamp(pi["from"])
                    capability[early] = float(before["capability_mw"])
                    rec["phase_in_capability_mw"] = float(before["capability_mw"])
                    rec["phase_in_from"] = str(pi["from"])
            limit = capability * float(secured.get("capability_scale", 1.0)) * factor
            rec.update({"mean_factor": round(float(factor.mean()), 4), "mean_limit_mw": round(float(limit.mean()), 1),
                        "min_limit_mw": round(float(limit.min()), 1)})
        rows.append(rec)
    return rows


def audit(spec: dict, root: str | Path = ".") -> pd.DataFrame:
    """The audit table (one row per in-run family x boundary, and per family whose capability is not active)."""
    from scripts.solve.secured_boundaries import load_boundary_geometries

    root = Path(root)
    caps, geoms, hists = {}, {}, {}

    def cap(path):
        if path not in caps:
            caps[path] = pd.read_csv(root / path)
        return caps[path]

    def geom(path):
        if path not in geoms:
            geoms[path] = set(load_boundary_geometries(root / path))
        return geoms[path]

    def hist(path):
        if path and path not in hists:
            hists[path] = pd.read_csv(root / path)
        return hists.get(path)

    families = spec["families"]
    per_family: dict = {}
    for fam, entry in families.items():
        # Months that resolve alike form one group (normally one per family).
        groups: dict = {}
        for m in entry["months"]:
            key = json.dumps({"active": m["active"], "secured": m["secured"]}, sort_keys=True, default=str)
            groups.setdefault(key, []).append(m)
        year = None
        out = []
        for key, months in groups.items():
            first = months[0]
            label = "|".join(m["month"] or m["scenario"] for m in months)
            base = {"family": fam, "tag": entry["tag"], "months": label,
                    "case": entry.get("delay_case") or "central", "network_delivery_case": first["case"] or "",
                    "secured_active": bool(first["active"])}
            if not first["active"]:
                out.append({**base, "boundary": "", "problems": [], "info": "secured capability not active "
                            "(outage_schedule off or another source): no boundary limits"})
                continue
            s = first["secured"]
            year = s["capability_year"]
            hd = s.get("historic_derate") or {}
            for rec in boundary_rows(s, _snapshots(months, year), cap(s["capability_file"]), geom(s["geometry_file"]),
                                     hist(hd.get("file")) if hd.get("enabled") else None):
                out.append({**base, **rec})
        per_family[fam] = {"rows": out, "groups": len(groups)}

    rows = []
    for fam, entry in families.items():
        if not entry["in_run"]:
            continue
        twin = entry.get("twin")
        twin_caps = {r["boundary"]: r.get("capability_mw") for r in per_family.get(twin, {}).get("rows", [])} \
            if twin else {}
        twin_pi = {r["boundary"]: r.get("phase_in_capability_mw") for r in per_family.get(twin, {}).get("rows", [])} \
            if twin else {}
        central = entry.get("central")
        central_rows = per_family.get(central, {}).get("rows", []) if central else []
        central_file = next((r.get("capability_file") for r in central_rows if r.get("capability_file")), None)
        for r in per_family[fam]["rows"]:
            warns = list(r.pop("problems", []))
            info = [r.pop("info")] if r.get("info") else []
            checked = False
            if per_family[fam]["groups"] > 1:
                warns.append(f"months resolve to {per_family[fam]['groups']} different settings")
            if r.get("capability_mw") is not None:
                if entry.get("delay_case"):
                    t = twin_caps.get(r["boundary"])
                    r["twin_capability_mw"] = t
                    if t is None or not np.isfinite(t):
                        info.append(f"central twin {twin or '-'} has no capability for this boundary")
                    else:
                        checked = True
                        if r["capability_mw"] > t + 1e-6:
                            warns.append(f"delay capability {r['capability_mw']:,.0f} MW ABOVE central twin "
                                         f"{twin} {t:,.0f} MW")
                        tp = twin_pi.get(r["boundary"])
                        if r.get("phase_in_capability_mw") is not None and tp is not None \
                                and r["phase_in_capability_mw"] > tp + 1e-6:
                            warns.append("phase-in capability above the central twin's")
                if central_file is not None:
                    checked = True
                    r["central_capability_file"] = central_file
                    if r["capability_file"] != central_file:
                        warns.append(f"reads {r['capability_file']}, central {central} reads {central_file}")
                else:
                    info.append("no central family to compare the capability file with")
            status = "WARN" if warns else ("PASS" if checked else "INFO")
            r.update({"central_twin": twin or "", "status": status, "note": "; ".join(warns + info)})
            rows.append(r)
    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def run(spec: dict, output: str | Path, log_path: str | None = None) -> pd.DataFrame:
    from scripts.utilities.logging_config import setup_logging

    log = setup_logging(log_path or "audit_secured_capability")
    table = audit(spec)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    fams = table["family"].nunique() if len(table) else 0
    log.info("Secured capability audit: %d families, %d rows, %s (report only)", fams, len(table),
             table["status"].value_counts().to_dict() if len(table) else {})
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        if len(table):
            wide = table.pivot_table(index="boundary", columns="family", values="capability_mw", aggfunc="first")
            log.info("Capability (MW) by boundary and family:\n%s", wide.to_string(float_format=lambda v: f"{v:,.0f}"))
        for _, r in table[table["status"].eq("WARN")].iterrows():
            log.warning("WARN %s %s: %s", r["family"], r["boundary"], r["note"])
    return table


def spec_from_config(central_tag: str | None = None, run_ids: list[str] | None = None) -> dict:
    """The spec for the repository's own config (outside Snakemake)."""
    import yaml

    sys.path.insert(0, str(PROJECT_ROOT / "config"))
    from config_loader import load_config

    full = load_config(PROJECT_ROOT / "config")
    return audit_spec(full["scenarios"], run_ids or full["run_scenarios"],
                      central_tag or full["study_settings"]["secured_capability_audit_central"] or DEFAULT_CENTRAL_TAG)


if "snakemake" in globals():
    run(dict(snakemake.params.spec), snakemake.output.csv, snakemake.log[0] if snakemake.log else None)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="resources/analysis/secured_capability_by_family.csv")
    parser.add_argument("--central-tag", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    _table = run(spec_from_config(args.central_tag), args.out, args.log)
    print(_table["status"].value_counts().to_dict())
