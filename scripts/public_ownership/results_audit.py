"""Results audit: invariants checked on the real outputs of a bills run (user, 15 Nov 2026).

The bugs of 11-15 Nov 2026 were not in functions a unit test exercises on made-up data; they were the same quantity
computed twice and drifting apart, money booked twice or never, workflow wiring that left outputs stale, silent
fallbacks, and settings that differ between model years. Each check here tests one of those families on the files a
run actually wrote, and the audit fails the run on any violation (``--report-only`` writes the report and never fails).

Checks:

* ``stale``: every output a target depends on is newer than every input it was built from, ``ancient()`` inputs
  included, wherever Snakemake plans no update (``snakemake --detailed-summary``): the P2-bid registers that no plan
  rebuilt after their months were re-solved (14 Nov) are this case.
* ``purchase``: under a market-value purchase each acquired legacy row's price in the register equals the legacy
  valuation file's owners' value (the CCS plants priced without their contract income, 15 Nov); every row the
  valuation leaves out of scope is unowned and unpriced; the rows the case leaves out (energy from waste) are unowned.
* ``fallbacks``: no balancing month bid at a fallback strike.
* ``parity``: a case present in two model years has the same lever settings in both, apart from levers declared
  year-specific (the 2% transaction cost missing from 2040/2050's S3 cases, 13 Nov).

    python -m scripts.public_ownership.results_audit --out <report.json> [--summary-tsv <tsv>] [--report-only]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PO = Path("resources/public_ownership")
MARKET = Path("resources/market")
S3 = "public_ownership@fully_nationalised_market_value"
#: Bases on which the price is the valuation's owners' value (pv_market_gbp).
MARKET_VALUE_BASES = ("market_value", "thermal_market_value")
#: Levers that may differ between a 2030 case and the same case in a long-range year (declared, with the reason).
YEAR_SPECIFIC_LEVERS = {
    "capacity_market_design": "2050 decision 73: vintaged registers only",
    "capacity_procurement": "F27 item 6: 2030 registers only",
    "balancing_bids": "the P2 bid stack is a 2030 run group's lever",
}
#: Seconds of clock skew allowed between an input and its output (OneDrive-synced files).
MTIME_SLACK_S = 2.0
#: Stale outputs accepted by design or explained, as (output path pattern, reason). Anything else stale fails.
ACCEPTED_STALE = (
    (r"^resources/renewable/profiles/", "upstream of every solved network: the site files were rewritten on 10 Sep 2026, "
     "the same day as the profiles, before any network in use was built (a change would need a rebuild and re-solve)"),
    (r"_generator_(state_shares|commissioning_years)\.csv$", "read the register as ancient() so a register rebuild never "
     "re-solves the bid months; rule verify_public_ownership_ownership_tables checks them against the register"),
    (r"^resources/analysis/FES25_HT_20[45]0_ro_(eligible_fleet\.csv|scheme_summary\.json)$", "the recovery-demand "
     "file was rewritten byte-identical by the bill-input builder on 14 Nov 2026 (no content change)"),
    (r"_generator_state_shares_deployment_share_\d+\.csv$", "solve inputs of the P2-bid deployment-share months, read "
     "off the register as ancient(); the rule verifying the year's ownership tables (verify_ownership_tables_2050.py) "
     "checks them against the register within the changes the user accepted on 15 Nov 2026 "
     "(data/public_ownership/financing/accepted_ownership_changes.csv)"),
    (r"^resources/market/FES25_HT_20[45]0_(support_labels|dpa_dispatch)\.(csv|json)$", "solve inputs of the 2040 and 2050 "
     "months, read as ancient(): rebuilt from today's inputs on 16 Nov 2026 the DPA dispatch is byte-identical and the "
     "labels are identical on every field the bids read (component, support type, negative-price rule, reference, "
     "strike and strike key, basis, DPA margin, RO banding, build year); only the split of onshore CfD rows differs. "
     "User acceptance, 16 Nov 2026"),
    (r"^resources/FES/fes2025_holistic_transition_20[45]0_station_schedule\.csv$", "solve input; rebuilt from today's "
     "station list and scripts on 16 Nov 2026 it is byte-identical. User acceptance, 16 Nov 2026"),
)
#: Purchase-price agreement: a row off its owners' value by more than this share fails, and so does a technology
#: (carrier and support type) off by more in total; smaller gaps are listed.
ROW_FAIL_SHARE = 0.05
GROUP_FAIL_SHARE = 0.01


class AuditError(ValueError):
    """A results invariant does not hold."""


# ── stale outputs ────────────────────────────────────────────────────────────

def read_detailed_summary(path: str | Path) -> pd.DataFrame:
    """``snakemake --detailed-summary`` output (log lines before the header are skipped)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("output_file\t"))
    rows = [line.split("\t") for line in lines[start:] if line.count("\t") >= 7]
    return pd.DataFrame(rows[1:], columns=rows[0])


def stale_outputs(summary: pd.DataFrame, root: str | Path = ".") -> pd.DataFrame:
    """Existing outputs older than one of their inputs where Snakemake plans no update (``ancient()`` included)."""
    from scripts.utilities.code_fingerprint import is_tooling

    root = Path(root)
    out = []
    for row in summary.itertuples(index=False):
        plan = str(getattr(row, "plan", "")).strip()
        if plan != "no update":
            continue
        target = root / row.output_file
        if not target.exists():
            continue
        made = target.stat().st_mtime
        # Snakemake records the checksum of each small input it built from and does not rerun when a newer file
        # has the same content (16 Nov 2026: the CM pin check, its summary rewritten byte-identical); such an
        # input is not newer in content, so it does not make the output stale.
        recorded = ((_metadata(row.output_file, root) or {}).get("input_checksums") or {})
        newer = []
        for name in str(getattr(row, "_4", "") or row[4]).split(","):
            name = name.strip()
            path = root / name
            if name and path.is_file() and path.stat().st_mtime > made + MTIME_SLACK_S:
                if name in recorded and _sha256(path) == recorded[name]:
                    continue
                newer.append(name)
        if newer:
            reason = next((why for pattern, why in ACCEPTED_STALE if re.search(pattern, row.output_file)), "")
            if not reason and all(is_tooling(root / name, root.resolve()) for name in newer):
                reason = ("only test tooling is newer (user decision, 15 Nov 2026; code_fingerprint.TOOLING_PREFIXES): "
                          + ", ".join(newer))
            out.append({"output_file": row.output_file, "rule": row.rule, "newer_inputs": len(newer),
                        "example_input": newer[0], "accepted": reason})
    return pd.DataFrame(out, columns=["output_file", "rule", "newer_inputs", "example_input", "accepted"])


# ── the purchase ─────────────────────────────────────────────────────────────

def purchase_agreement(register: pd.DataFrame, valuation: pd.DataFrame, *, case: str = S3,
                       rel_tol: float = 1e-6, abs_tol: float = 1.0) -> dict:
    """The case's purchase price per legacy row against the valuation file, and its scope against the valuation's.

    Rows priced on owners' value: a row off by more than ``ROW_FAIL_SHARE`` or a technology off by more than
    ``GROUP_FAIL_SHARE`` fails; any gap above ``rel_tol`` is counted and the total reported.
    """
    rows = register.loc[register["case_id"].astype(str).eq(case)
                        & register["finance_vintage"].astype(str).eq("legacy")].copy()
    if rows.empty:
        raise AuditError(f"the register has no legacy rows for {case}")
    value = valuation.drop_duplicates("asset_id").set_index("asset_id")
    rows["pv_market_gbp"] = rows["asset_id"].map(pd.to_numeric(value["pv_market_gbp"], errors="coerce"))
    rows["scope_group"] = rows["asset_id"].map(value["scope_group"])
    price = pd.to_numeric(rows["acquisition_price_gbp"], errors="coerce")
    basis = rows["acquisition_basis"].astype(str)
    owned = rows["state_owned"].astype(str).str.lower().isin(["true", "1"])
    on_value = basis.isin(MARKET_VALUE_BASES) & rows["pv_market_gbp"].notna()
    expected = rows["pv_market_gbp"].clip(lower=0.0)
    gap = (price - expected).abs()
    mismatch = rows.loc[on_value & (gap > np.maximum(abs_tol, rel_tol * expected.abs()))]
    big = rows.loc[on_value & (gap > np.maximum(abs_tol, ROW_FAIL_SHARE * expected.abs()))]
    keys = [rows["carrier"].astype(str)]
    if "support_type" in rows.columns:
        keys.append(rows["support_type"].fillna("none").astype(str))
    tech = pd.DataFrame({"price": price, "value": expected}).loc[on_value].groupby(
        [k.loc[on_value] for k in keys]).sum()
    tech["share"] = (tech["price"] - tech["value"]) / tech["value"].where(tech["value"].abs() > 0)
    tech_fail = tech.loc[tech["share"].abs() > GROUP_FAIL_SHARE]
    excluded = rows["scope_group"].astype(str).eq("excluded")
    out_of_scope_owned = rows.loc[excluded & owned]
    out_of_scope_priced = rows.loc[excluded & price.notna() & (price != 0.0)]
    return {
        "case": case, "rows": int(len(rows)), "priced_on_owners_value": int(on_value.sum()),
        "price_gap_total_gbp": float((price[on_value] - expected[on_value]).sum()),
        "rows_off_listed": int(len(mismatch)),
        "rows_off_over_fail_share": [{"asset_id": a, "price_gbp": float(p), "owners_value_gbp": float(v)}
                                     for a, p, v in zip(big["asset_id"], price[big.index], expected[big.index])][:50],
        "technologies_off_over_fail_share": {" / ".join(map(str, k)) if isinstance(k, tuple) else str(k):
                                             round(float(v), 4) for k, v in tech_fail["share"].items()},
        "excluded_but_owned": out_of_scope_owned["asset_id"].astype(str).tolist(),
        "excluded_but_priced": out_of_scope_priced["asset_id"].astype(str).tolist(),
        "purchase_total_gbp": float(price.fillna(0.0).sum()),
        "ok": big.empty and tech_fail.empty and out_of_scope_owned.empty and out_of_scope_priced.empty,
    }


# ── fallbacks ────────────────────────────────────────────────────────────────

def fallback_strikes(bid_files: Iterable[str | Path]) -> dict:
    """Balancing months that bid at a fallback strike, with the generators."""
    out = {}
    for path in bid_files:
        bids = pd.read_csv(path, usecols=lambda c: c in ("generator", "is_fallback_strike"))
        if "is_fallback_strike" not in bids.columns:
            continue
        hit = bids.loc[bids["is_fallback_strike"].fillna(False).astype(bool), "generator"].astype(str).tolist()
        if hit:
            out[Path(path).name] = hit[:10]
    return {"months_with_fallbacks": out, "ok": not out}


# ── cross-year parity ────────────────────────────────────────────────────────

def lever_parity(settings_by_year: Mapping[int, pd.DataFrame], *,
                 year_specific: Mapping[str, str] = YEAR_SPECIFIC_LEVERS) -> dict:
    """Cases present in two years whose lever settings differ outside the declared year-specific levers."""
    frames = {y: s.set_index("case_id") for y, s in settings_by_year.items()}
    years = sorted(frames)
    diffs = []
    for i, a in enumerate(years):
        for b in years[i + 1:]:
            common = frames[a].index.intersection(frames[b].index)
            levers = [c for c in frames[a].columns.intersection(frames[b].columns) if c not in year_specific]
            left = frames[a].loc[common, levers].astype(str)
            right = frames[b].loc[common, levers].astype(str)
            for lever in levers:
                differ = left[lever].ne(right[lever])
                for case in left.index[differ]:
                    diffs.append({"years": f"{a}/{b}", "case_id": case, "lever": lever,
                                  "values": f"{left.at[case, lever]} / {right.at[case, lever]}"})
    return {"differences": diffs[:200], "count": len(diffs), "ok": not diffs}


# ── content and code provenance (user, 15 Nov 2026: "so this never happens again") ─────────────────────────────

LEDGER = Path("resources/analysis/code_fingerprint_ledger.json")
#: Code or content changes accepted for outputs that were not rebuilt, as (output pattern, changed file pattern, reason).
ACCEPTED_PROVENANCE = (
    (r"^resources/market/FES25_HT_20[45]0_support_labels\.(csv|json)$",
     r"build_asset_finance_register\.py|legislation_year\.py|pypsa_finance_technology_crosswalk_2050\.csv",
     "S4/P3/S6 copied in after the 14 Nov label build: the labels rebuilt with them give every generator the same "
     "weight by contract (checked 14 Nov 02:35), so the solved months stand"),
    (r"^resources/market/FES25_HT_20[45]0_(support_labels|dpa_dispatch)\.(csv|json)$",
     r"technology_costs_vintage\.csv|fes2025_hurdle_rates\.csv|technology_economic_lives\.csv",
     "15 Nov 2026 cost-table regeneration (OCGT/gas-engine vintage rows): labels and DPA dispatch rebuilt in scratch "
     "are identical in every field the solves read, 2040 and 2050; user decision, ancient() in both rules"),
    (r".",
     r"scripts/public_ownership/f20_parity/",
     "the F20 parity harness is test tooling, not a code input (user decision, 15 Nov 2026; "
     "code_fingerprint.TOOLING_PREFIXES): its 15 Nov edit added four rule names to a list only its own test reads"),
    (r"^resources/market/FES25_HT_20[45]0_(support_labels|dpa_dispatch)\.(csv|json)$",
     r"pypsa_finance_technology_crosswalk(_2050)?\.csv|cfd_strike_inputs_2050\.csv|fleet_vintages\.py|"
     r"build_asset_finance_register\.py|legislation_year\.py",
     "user acceptance, 16 Nov 2026: rebuilt from today's inputs the DPA dispatch is byte-identical and the labels are "
     "identical on every field the bids read; only the split of onshore CfD rows differs, so the solved months stand"),
    (r"^resources/(market|analysis)/CP30_FFR_2030_",
     r"^data/market/lccc/cfd_strike_prices_by_round\.csv$",
     "user acceptance, 16 Nov 2026: the table's 15 Nov change is note text only (the AR8_modelled notes' capex factors "
     "and private rates); every strike value is unchanged, so the 2030 solves stand"),
    (r"^resources/FES/fes2025_holistic_transition_20[45]0_station_schedule\.csv$",
     r"station_list\.csv|build_fes_station_schedule\.py|fes_supply\.py",
     "user acceptance, 16 Nov 2026: rebuilt from today's station list and scripts the schedule is byte-identical"),
    (r"^resources/(market/CP30_FFR_2030_[A-Za-z0-9_]+_(pro_rata_turn_down\.(csv|json)|turn_down_basis\.json)|"
     r"analysis/CP30_2030_[a-z0-9_]+_pro_rata_turn_down_summary\.json)$",
     r"^scripts/market/turn_down_allocation\.py$",
     "user acceptance, 16 Nov 2026: the script's 15 Nov edit (the 2040/2050 support-label path) was copied in with "
     "its 28 Oct file time, so the 2030 months were not rebuilt; rebuilt today with the current script, January "
     "no_acceleration, July cp30_network_p3_bids and December no_upgrades_p2_bids are byte-identical (csv and json)"),
    (r"^resources/public_ownership/[a-z0-9_]+_generator_(state_shares|commissioning_years)\.csv$",
     r"_case_settings\.csv$",
     "user acceptance, 16 Nov 2026: solve inputs read off the register as ancient(); the rule verifying the "
     "ownership tables recomputed them against today's register and passed (preflight, 16 Nov 2026), within the "
     "changes the user accepted on 15 Nov"),
    (r".",
     r"^scripts/FES/FES_data\.py$|^resources/FES/FES_202[45]_data\.csv$",
     "user acceptance, 20 Nov 2026: FES_data.py's 19 Nov edit names the pathway column of the FES 2021-2023 editions "
     "only (2024 and 2025 already have 'FES Pathway'); FES_2024_data.csv and FES_2025_data.csv were regenerated by "
     "it on 19 Nov from the 26 Oct NESO cache and unchanged inputs, so their content is unchanged; file times set "
     "back to 27 Oct 10:00 (FES_data.py 09:40), so the 2030, 2040 and 2050 builds stand"),
    (r"^resources/FES/fes2025_future_costs_20[0-9]{2}\.json$",
     r"^scripts/FES/fes_commodity_prices\.py$",
     "user acceptance, 20 Nov 2026: the 19 Nov edit adds CPIH 2019-2023 for the FES 2021-2023 workbooks; rebuilt "
     "with the current script the 2040, 2045 and 2050 sources are identical; file time set back to 7 Nov 22:00"),
    (r"^resources/interconnectors/entsoe_day_ahead_prices_2024\.csv$",
     r"^data/interconnectors/entsoe_day_ahead/day_ahead_prices_2024_eur_per_mwh\.csv$",
     "user acceptance, 20 Nov 2026: the shipped 2024 prices (18 Nov) give an output identical to the 26 Sep API "
     "fetch (all 8,785 rows and 7 columns, largest difference 0.0); file time set back to 26 Sep 08:00 UTC"),
    (r".",
     r"^scripts/renewables/large_hydro_inflow\.py$",
     "user acceptance, 20 Nov 2026: the 19 Nov edit only adds apply_historical_metered_hydro, which returns at once "
     "unless targets.source is historical; the functions the 2030, 2040 and 2050 builds call are unchanged; file "
     "time set back to 31 Oct 12:00, before those builds"),
    (r"^resources/(?!.*(FES25_HT_20[45]0|Tutorial_2040_HT_clustered))",
     r"^scripts/generators/(proxy_profiles|integrate_renewable_generators)\.py$",
     "user acceptance, 21 Nov 2026 (issue #51): the edit adds renewables.proxy_profile_members: all_sites, set only "
     "in the gsp_block method overlay; every new line runs only under it, so the brownfield (CP30 2030) and "
     "historical builds take the unchanged path, and the one shared function refactored (the site capacity map) "
     "gives the identical map on the real site files; proxy_profiles.py's file time set back to 19 Oct 05:26 (its "
     "last commit), and the FES 2040 and 2050 builds were forced and re-solved"),
    (r"^resources/public_ownership/fes_ht_20[45]0_generator_state_shares_deployment_share_\d+\.csv$",
     r"^resources/public_ownership/fes_ht_20[45]0_case_settings\.csv$",
     "user acceptance, 23 Nov 2026 (issue #51): solve inputs of the P2-bid deployment-share months, read off the "
     "register as ancient(); the issue #51 registers rewrote case_settings, and the preflight of the 22 Nov bills run "
     "recomputed all eight tables against those registers (fes_ht_20{40,50}_ownership_tables_check.json): every "
     "table matches to 1e-13, within the changes the user accepted on 15 Nov"),
)
# Ledger catch-up (user acceptance, 23 Nov 2026): the ledger had last been recorded on 16 Nov, and the run script's
# audit does not record it, so the code check flagged 1,684 outputs of the issue #51 re-solve and bills run. Every one
# was built from 21 Nov 19:49 on, after the last change to the scripts it names (19 Nov, committed, no uncommitted
# edits); the ledger was recorded for outputs built since 21 Nov 19:00, and no declaration is kept for it.


def _metadata(output: str, root: Path) -> dict | None:
    import base64

    path = root / ".snakemake" / "metadata" / base64.b64encode(output.encode()).decode().replace("/", "@")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_provenance(output: str, changed: str) -> str:
    return next((why for out_pattern, file_pattern, why in ACCEPTED_PROVENANCE
                 if re.search(out_pattern, output) and re.search(file_pattern, changed)), "")


def content_changes(summary: pd.DataFrame, root: str | Path = ".") -> pd.DataFrame:
    """Outputs built from an input whose content has changed since (Snakemake's recorded input checksums).

    File times cannot hide this: an input copied in with its old time but new content is caught.
    """
    root = Path(root)
    out = []
    for output in summary["output_file"]:
        meta = _metadata(output, root)
        for name, recorded in ((meta or {}).get("input_checksums") or {}).items():
            path = root / name
            if path.is_file() and _sha256(path) != recorded:
                out.append({"output_file": output, "changed_input": name, "accepted": _accepted_provenance(output, name)})
    return pd.DataFrame(out, columns=["output_file", "changed_input", "accepted"])


def _command(output: str, root: Path) -> str:
    meta = _metadata(output, root) or {}
    return str(meta.get("shellcmd") or meta.get("code") or "")


def code_changes(summary: pd.DataFrame, ledger: Mapping, root: str | Path = ".") -> pd.DataFrame:
    """Outputs whose code (scripts run and every project module they import) has changed since they were built.

    The ledger holds the fingerprint each output was built with and the file hashes behind it; outputs not in it
    are reported as unknown, never as current.
    """
    from scripts.utilities.code_fingerprint import command_fingerprint

    root = Path(root)
    built = ledger.get("outputs", {})
    files_by_fp = ledger.get("fingerprints", {})
    out = []
    for output in summary["output_file"]:
        fp, files = command_fingerprint(_command(output, root), root)
        if fp is None:
            continue
        recorded = built.get(output)
        if recorded is None:
            out.append({"output_file": output, "changed_files": "", "status": "unknown", "accepted": ""})
            continue
        if recorded == fp:
            continue
        before = files_by_fp.get(recorded, {})
        changed = [f for f in files if before.get(f) != _sha256(root / f)] + [f for f in before if f not in files]
        reasons = [_accepted_provenance(output, f) for f in changed]
        out.append({"output_file": output, "changed_files": ";".join(changed[:20]), "status": "changed",
                    "accepted": reasons[0] if changed and all(reasons) else ""})
    return pd.DataFrame(out, columns=["output_file", "changed_files", "status", "accepted"])


def record_ledger(summary: pd.DataFrame, ledger: dict, *, built_since: float | None, root: str | Path = ".",
                  baseline: bool = False) -> dict:
    """Record the current code fingerprint of outputs rebuilt since ``built_since`` (every output with ``baseline``)."""
    from scripts.utilities.code_fingerprint import command_fingerprint

    root = Path(root)
    ledger.setdefault("outputs", {})
    ledger.setdefault("fingerprints", {})
    for output in summary["output_file"]:
        path = root / output
        if not path.exists():
            continue
        if not baseline and (built_since is None or path.stat().st_mtime < built_since):
            continue
        fp, files = command_fingerprint(_command(output, root), root)
        if fp is None:
            continue
        ledger["outputs"][output] = fp
        ledger["fingerprints"].setdefault(fp, {f: _sha256(root / f) for f in files})
    return ledger


def pending_after_run(summary: pd.DataFrame) -> pd.DataFrame:
    """Outputs a completed run still plans to update: promised by the plan but not rebuilt (the P2 registers, 15 Nov)."""
    rows = summary.loc[summary["plan"].astype(str).str.strip().ne("no update")]
    return rows[["output_file", "rule", "status", "plan"]].reset_index(drop=True)


# ── runner ───────────────────────────────────────────────────────────────────

def run(*, summary_tsv: str | None, groups_2030: Iterable[str], long_range: Mapping[int, str],
        bid_files: Iterable[str | Path], root: str | Path = ".", post_run: bool = False,
        ledger_path: str | Path | None = None) -> dict:
    from scripts.public_ownership.register_io import read_register

    root = Path(root)
    report: dict = {}
    if summary_tsv:
        stale = stale_outputs(read_detailed_summary(summary_tsv), root)
        unexplained = stale.loc[stale["accepted"].eq("")]
        report["stale"] = {"unexplained": unexplained.to_dict(orient="records")[:200],
                           "accepted": stale.loc[stale["accepted"].ne("")].to_dict(orient="records")[:200],
                           "count": int(len(stale)), "ok": unexplained.empty}
        summary = read_detailed_summary(summary_tsv)
        if post_run:
            pending = pending_after_run(summary)
            report["pending_after_run"] = {"outputs": pending.to_dict(orient="records")[:200],
                                           "count": int(len(pending)), "ok": pending.empty}
        content = content_changes(summary, root)
        bad = content.loc[content["accepted"].eq("")]
        report["content"] = {"changed": bad.to_dict(orient="records")[:200],
                             "accepted": content.loc[content["accepted"].ne("")].to_dict(orient="records")[:50],
                             "count": int(len(content)), "ok": bad.empty}
        if ledger_path is not None:
            ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8")) if Path(ledger_path).exists() else {}
            code = code_changes(summary, ledger, root)
            changed = code.loc[code["status"].eq("changed") & code["accepted"].eq("")]
            report["code"] = {"changed": changed.to_dict(orient="records")[:200],
                              "accepted": code.loc[code["accepted"].ne("")].to_dict(orient="records")[:50],
                              "unknown": int(code["status"].eq("unknown").sum()), "ok": changed.empty}
    columns = ["case_id", "asset_id", "carrier", "support_type", "finance_vintage", "state_owned",
               "acquisition_price_gbp", "acquisition_basis"]
    report["purchase"] = {}
    for item in groups_2030:
        # "group:valuation_group" for a group that holds another's purchase price (purchase_price_from).
        group, _, priced_by = str(item).partition(":")
        register = read_register(root / PO / f"{group}_asset_finance_register.parquet", columns=columns, cases=[S3])
        valuation = pd.read_csv(root / PO / f"{priced_by or group}_legacy_valuation.csv", low_memory=False,
                                usecols=["asset_id", "pv_market_gbp", "scope_group"])
        report["purchase"][item] = purchase_agreement(register, valuation)
    report["fallbacks"] = fallback_strikes(bid_files)
    settings = {2030: pd.read_csv(root / PO / f"{str(next(iter(groups_2030))).partition(':')[0]}_case_settings.csv")}
    for year, group in long_range.items():
        settings[int(year)] = pd.read_csv(root / PO / f"{group}_case_settings.csv")
    report["parity"] = lever_parity(settings)
    report["ok"] = all(section.get("ok", True) if "ok" in section else all(v["ok"] for v in section.values())
                       for section in report.values() if isinstance(section, dict))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-tsv", default=None, help="snakemake --detailed-summary output for the run's targets")
    parser.add_argument("--groups-2030", nargs="*",
                        default=["cp30_2030_no_acceleration", "cp30_2030_cp30_network:cp30_2030_no_acceleration"])
    parser.add_argument("--long-range", nargs="*", default=["2040=fes_ht_2040", "2050=fes_ht_2050"])
    parser.add_argument("--bids-glob", default="resources/market/*_contract_value_bids.csv")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--post-run", action="store_true", help="after a completed run: nothing may still be pending")
    parser.add_argument("--ledger", default=str(LEDGER), help="the code-fingerprint ledger")
    parser.add_argument("--record-since", default=None,
                        help="after a passing audit, record the code of outputs rebuilt since this ISO time")
    parser.add_argument("--baseline", action="store_true", help="record every output's current code (first use)")
    parser.add_argument("--purchase-register", default=None,
                        help="purchase check only (a workflow rule): this group's register against --purchase-valuation")
    parser.add_argument("--purchase-valuation", default=None, help="the legacy valuation its purchase is priced from")
    args = parser.parse_args(argv)
    if args.purchase_register:
        # 15 Nov 2026: the purchase check as a rule, so Snakemake guarantees the register and its valuation are
        # current (an ad-hoc check had compared a fresh register with a two-day-old valuation).
        from scripts.public_ownership.register_io import read_register

        columns = ["case_id", "asset_id", "carrier", "support_type", "finance_vintage", "state_owned",
                   "acquisition_price_gbp", "acquisition_basis"]
        register = read_register(args.purchase_register, columns=columns, cases=[S3])
        valuation = pd.read_csv(args.purchase_valuation, low_memory=False,
                                usecols=["asset_id", "pv_market_gbp", "scope_group"])
        result = purchase_agreement(register, valuation)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"purchase {'ok' if result['ok'] else 'FAILS'}: total GBP {result['purchase_total_gbp'] / 1e9:.2f} bn, "
              f"technologies off >1% {result['technologies_off_over_fail_share']}, "
              f"rows off >5% {len(result['rows_off_over_fail_share'])}")
        if not result["ok"] and not args.report_only:
            Path(args.out).unlink()
            raise SystemExit(f"purchase audit fails for {args.purchase_register}")
        return
    long_range = dict(item.split("=", 1) for item in args.long_range)
    bid_files = sorted(Path(".").glob(args.bids_glob))
    report = run(summary_tsv=args.summary_tsv, groups_2030=args.groups_2030, long_range=long_range,
                 bid_files=bid_files, post_run=args.post_run, ledger_path=args.ledger)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, section in report.items():
        if isinstance(section, dict):
            ok = section.get("ok") if "ok" in section else all(v.get("ok") for v in section.values())
            print(f"{name:10s} {'ok' if ok else 'FAILS'}")
    if not report["ok"] and not args.report_only:
        raise SystemExit(f"results audit fails; see {args.out}")
    if args.summary_tsv and (args.record_since or args.baseline) and report["ok"]:
        import datetime as _dt

        since = _dt.datetime.fromisoformat(args.record_since).timestamp() if args.record_since else None
        ledger_path = Path(args.ledger)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
        ledger = record_ledger(read_detailed_summary(args.summary_tsv), ledger, built_since=since,
                               baseline=args.baseline)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
        print(f"ledger: {len(ledger.get('outputs', {}))} outputs recorded")


if __name__ == "__main__":
    main()
