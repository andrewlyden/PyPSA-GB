"""OW11: the ownership lever spreadsheet and the case-settings manifest, generated from the case catalogue.

Two outputs, each by its own rule (``rules/public_ownership.smk``, the report targets):

* ``manifest``: ``{run_group}_case_settings_manifest.json`` beside
  ``{run_group}_case_settings.csv``.  The sha256 of the case settings, of the run
  group's resolved ``public_ownership`` and ``consumer_bill.forecasts`` config
  blocks, and of every source the finance layer reads (``scripts/public_ownership/*.py``
  and ``data/public_ownership/financing/*``).  The rule takes the config and the
  sources as ``ancient()`` inputs, so it is written when the case settings are and
  records the state they were written in.
* ``workbook``: ``resources/analysis/{run_group}_ownership_levers.xlsx``, replacing
  the hand-made copy of 24 September 2026.  One row per lever value (the
  catalogue in :mod:`levers`, the retired values listed beside it), the scenarios'
  values, and the lever audit's five status columns:

  - **implemented**: the value is in the catalogue (``retired`` with its pointer
    when it is in ``levers.RETIRED_VALUES``);
  - **configured**: how many configured cases take the value
    (``reference_runs.configured_case_specs``);
  - **saved**: how many of those have numbers in ``{run_group}_household_totals.csv``;
  - **current**: how many of the saved ones were written on today's settings: the
    manifest's config and source hashes equal today's, the manifest describes the
    case settings on disk, and the saved settings row equals the configured one;
  - **reconciled**: how many saved cases close the three-party identity
    (consumers + Exchequer + investors = resource saving) to GBP 1.

Run as ``python scripts/public_ownership/lever_workbook.py manifest|workbook ...``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd
import yaml

if __name__ == "__main__":  # run as python scripts/public_ownership/lever_workbook.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.public_ownership.levers import LEVERS, RETIRED_VALUES  # noqa: E402
from scripts.public_ownership.reference_runs import configured_case_specs  # noqa: E402

#: The catalogue's own sections (levers.py), keyed by each section's first lever.
SECTION_STARTS = {
    "new_build_generation": "1. Who owns what",
    "private_wacc_basis": "2. Cost of capital",
    "purchase_price_basis": "3. Acquisition terms",
    "cfd_strike_basis": "4. How public plant is paid",
    "wholesale_pricing": "5. Market and institution",
    "state_deficit_funding": "6. Fiscal incidence: who pays",
    "legacy_levy_funding": "7. Background: every case alike",
}
#: The sources the finance layer reads, hashed into the manifest (relative to the repository root).
SOURCE_GLOBS = ("scripts/public_ownership/*.py", "data/public_ownership/financing/*")
#: The three scenarios and P3 (the state also buys the existing fleet), as the hand-made sheet showed them.
SCENARIO_COLUMNS = {"Private": "private", "P1": "public_control", "P2": "public_ownership",
                    "P3": "public_ownership@fully_nationalised_market_value"}
IDENTITY_TOLERANCE_GBP = 1.0
IDENTITY_COLUMNS = ("consumer_change_vs_private_gbp", "exchequer_net_vs_private_gbp",
                    "investor_change_vs_private_gbp", "resource_saving_vs_private_gbp")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_block(block) -> str:
    return hashlib.sha256(json.dumps(block, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict:
    from config.config_loader import read_config

    return read_config(path) or {}


def source_hashes(root: Path) -> dict[str, str]:
    """sha256 of every finance-layer source, keyed by its path relative to ``root`` (forward slashes)."""
    root = Path(root)
    files = sorted({p for pattern in SOURCE_GLOBS for p in root.glob(pattern) if p.is_file()})
    return {p.relative_to(root).as_posix(): _sha256_file(p) for p in files}


def config_hashes(config: Mapping) -> dict[str, str]:
    return {"public_ownership_sha256": _sha256_block(config.get("public_ownership") or {}),
            # The key keeps its old name, so manifests written before the config restructure stay comparable.
            "consumer_bill_forecasts_sha256": _sha256_block((config.get("consumer_bill") or {}).get("forecasts") or {})}


def build_manifest(case_settings: Path, config_path: Path, run_group: str, root: Path) -> dict:
    """The manifest of one run group's case settings (see the module)."""
    frame = pd.read_csv(case_settings, usecols=["case_id"])
    config = load_config(config_path)
    return {
        "run_group": run_group,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case_settings": {"path": Path(case_settings).as_posix(), "sha256": _sha256_file(case_settings),
                          "cases": int(len(frame))},
        "config": {"path": Path(config_path).as_posix(), **config_hashes(config)},
        "sources": source_hashes(root),
        "note": ("Written when the case settings are (the rule's config and sources are ancient() inputs): the "
                 "hashes are the state the case settings were written in. lever_workbook.py compares them with "
                 "today's to mark saved cases current or stale."),
    }


def manifest_status(manifest: Mapping | None, *, case_settings: Path | None, config: Mapping, root: Path,
                    run_group: str) -> dict:
    """Whether a manifest still describes today's config, sources and case settings; what differs if not."""
    if not manifest:
        return {"current": False, "reasons": ["no manifest"]}
    reasons = []
    if manifest.get("run_group") != run_group:
        reasons.append(f"manifest is for run group {manifest.get('run_group')!r}")
    for key, value in config_hashes(config).items():
        if (manifest.get("config") or {}).get(key) != value:
            reasons.append(f"config {key.replace('_sha256', '')} changed")
    today, then = source_hashes(root), dict(manifest.get("sources") or {})
    changed = sorted(p for p in set(today) | set(then) if today.get(p) != then.get(p))
    if changed:
        reasons.append("sources changed: " + ", ".join(changed))
    if case_settings is not None and Path(case_settings).exists():
        if (manifest.get("case_settings") or {}).get("sha256") != _sha256_file(case_settings):
            reasons.append("case settings on disk are not the ones the manifest describes")
    return {"current": not reasons, "reasons": reasons}


def case_status(specs, *, saved_settings: pd.DataFrame | None, household_totals: pd.DataFrame | None,
                manifest_current: bool) -> pd.DataFrame:
    """One row per configured case: kind, stage and the saved / current / reconciled flags."""
    rows = []
    for spec in specs:
        stage = ("market_bill" if spec.market_bill()
                 else "acquisition_tariff" if spec.settings.get("dispatch_institution", "market") == "market"
                 else "institution")
        rows.append({"case_id": spec.case_id, "kind": spec.meta.get("kind"), "parent": spec.base_case,
                     "stage": stage, **{name: str(spec.settings[name]) for name in LEVERS}})
    frame = pd.DataFrame(rows).set_index("case_id")
    saved = pd.Series(False, index=frame.index)
    reconciled = pd.Series(False, index=frame.index)
    if household_totals is not None and not household_totals.empty:
        totals = household_totals.drop_duplicates("case_id").set_index("case_id")
        bill = pd.to_numeric(totals.get("household_total_gbp_inc_vat"), errors="coerce")
        saved = bill.reindex(frame.index).notna()
        if set(IDENTITY_COLUMNS) <= set(totals.columns):
            parts = totals[list(IDENTITY_COLUMNS)].apply(pd.to_numeric, errors="coerce")
            residual = (parts["consumer_change_vs_private_gbp"] + parts["exchequer_net_vs_private_gbp"]
                        + parts["investor_change_vs_private_gbp"] - parts["resource_saving_vs_private_gbp"])
            reconciled = (residual.abs() <= IDENTITY_TOLERANCE_GBP).reindex(frame.index, fill_value=False).astype(bool)
    current = pd.Series(False, index=frame.index)
    if manifest_current and saved_settings is not None and not saved_settings.empty:
        stored = saved_settings.drop_duplicates("case_id").set_index("case_id").astype(str)
        columns = [name for name in LEVERS if name in stored.columns]
        here = frame.index.intersection(stored.index)
        same = (stored.loc[here, columns] == frame.loc[here, columns]).all(axis=1)
        current.loc[here] = same.to_numpy()
    frame["saved"] = saved.to_numpy()
    frame["current"] = (current & saved).to_numpy()
    frame["reconciled"] = (reconciled & saved).to_numpy()
    return frame.reset_index()


def _plain(module, attribute: str) -> Mapping:
    return getattr(module, attribute, {}) if module is not None else {}


def lever_rows(cases: pd.DataFrame, scenario_settings: Mapping[str, Mapping[str, str]], plain=None) -> pd.DataFrame:
    """One row per lever value (and per retired value), with the scenarios' values and the five status columns."""
    lever_labels, value_labels = _plain(plain, "LEVER_LABELS"), _plain(plain, "VALUE_LABELS")
    rows, section, number = [], None, 0
    for name, lever in LEVERS.items():
        section = SECTION_STARTS.get(name, section)
        number += 1
        values = list(lever.values) + [value for (retired_lever, value) in RETIRED_VALUES if retired_lever == name]
        for value in values:
            retired = (name, value) in RETIRED_VALUES
            takes = cases[name].eq(value) if not retired else pd.Series(False, index=cases.index)
            configured = int(takes.sum())
            saved = int((takes & cases["saved"]).sum())
            current = int((takes & cases["current"]).sum())
            reconciled = int((takes & cases["reconciled"]).sum())
            if retired:
                status = "retired"
            elif not configured:
                status = "not configured"
            elif not saved:
                status = "configured, no numbers saved"
            elif current < saved:
                status = "saved, stale" if not current else "saved, partly stale"
            elif reconciled < saved:
                status = "current, identity not closed"
            else:
                status = "current and reconciled"
            row = {
                "Group": section, "No.": number, "Lever": lever_labels.get(name, name), "Lever (config)": name,
                "What it decides": lever.description if value == values[0] else None,
                "Option": (value_labels.get(name) or {}).get(value, value), "Value (config)": value,
                "Case label": lever.label(value),
                "Default (private)": "yes" if value == lever.default else None,
                **{column: ("yes" if str((scenario_settings.get(column) or {}).get(name)) == value else None)
                   for column in SCENARIO_COLUMNS},
                "Implemented": (f"retired: {RETIRED_VALUES[(name, value)]}" if retired else "yes"),
                "Configured (cases)": configured, "Saved (cases)": saved, "Current (cases)": current,
                "Reconciled (cases)": reconciled, "Status": status,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def scenario_rows(scenario_settings: Mapping[str, Mapping[str, str]], plain=None) -> pd.DataFrame:
    lever_labels, value_labels = _plain(plain, "LEVER_LABELS"), _plain(plain, "VALUE_LABELS")
    rows = []
    for name in LEVERS:
        row = {"Lever": lever_labels.get(name, name), "Lever (config)": name}
        for column in SCENARIO_COLUMNS:
            value = str((scenario_settings.get(column) or {}).get(name, ""))
            row[column] = (value_labels.get(name) or {}).get(value, value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_workbook(*, config_path: Path, run_group: str, case_settings: Path | None, manifest_path: Path | None,
                   household_totals: Path | None, root: Path, out: Path) -> dict:
    """Write the spreadsheet; returns the counts it reports."""
    config = load_config(config_path)
    po = config.get("public_ownership") or {}
    specs = configured_case_specs(po, run_group)
    manifest = (json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                if manifest_path and Path(manifest_path).exists() else None)
    status = manifest_status(manifest, case_settings=case_settings, config=config, root=root, run_group=run_group)
    saved_settings = (pd.read_csv(case_settings, dtype=str) if case_settings and Path(case_settings).exists() else None)
    totals = (pd.read_csv(household_totals, low_memory=False)
              if household_totals and Path(household_totals).exists() else None)
    cases = case_status(specs, saved_settings=saved_settings, household_totals=totals,
                        manifest_current=status["current"])
    by_id = {spec.case_id: spec.settings for spec in specs}
    scenario_settings = {column: by_id.get(case_id, {}) for column, case_id in SCENARIO_COLUMNS.items()}
    try:
        import scripts.analysis.build_ownership_explorer_data as plain
    except Exception:  # noqa: BLE001 - plain labels are a convenience; the config names stand without them
        plain = None
    levers = lever_rows(cases, scenario_settings, plain)
    scenarios = scenario_rows(scenario_settings, plain)
    counts = {"cases": int(len(cases)), "saved": int(cases["saved"].sum()), "current": int(cases["current"].sum()),
              "reconciled": int(cases["reconciled"].sum()), "lever_values": int(len(levers)),
              "values_not_configured": int(levers["Status"].eq("not configured").sum())}
    notes = [
        f"{run_group}: ownership levers, generated {datetime.now(timezone.utc):%d %B %Y} from the case catalogue "
        "(scripts/public_ownership/levers.py, reference_runs.configured_case_specs) by "
        "scripts/public_ownership/lever_workbook.py. Do not edit by hand: the rule rewrites it.",
        f"{len(LEVERS)} levers in seven groups, one row per option; the first option of each lever is today's "
        "private arrangement. Retired options are listed with the pointer the config parse gives.",
        f"Cases configured: {counts['cases']:,}; with saved numbers: {counts['saved']:,}; current: "
        f"{counts['current']:,}; closing the three-party identity: {counts['reconciled']:,}.",
        "Implemented: the option is in the catalogue (or retired, with its pointer).",
        "Configured: configured cases that take the option.",
        "Saved: of those, the cases with numbers in the household totals "
        f"({Path(household_totals).as_posix() if household_totals else 'not given'}).",
        "Current: saved cases whose saved settings equal today's, provided the case-settings manifest's config and "
        "source hashes equal today's. Manifest: " + ("current." if status["current"]
                                                     else "stale (" + "; ".join(status["reasons"]) + ")."),
        f"Reconciled: saved cases whose consumers + Exchequer + investors = resource saving to GBP "
        f"{IDENTITY_TOLERANCE_GBP:.0f}.",
    ]
    manifest_rows = pd.DataFrame(
        [{"item": "manifest", "value": Path(manifest_path).as_posix() if manifest_path else None}]
        + [{"item": k, "value": v} for k, v in ((manifest or {}).get("case_settings") or {}).items()]
        + [{"item": k, "value": v} for k, v in ((manifest or {}).get("config") or {}).items()]
        + [{"item": "written_utc", "value": (manifest or {}).get("written_utc")},
           {"item": "current", "value": status["current"]}]
        + [{"item": "stale because", "value": reason} for reason in status["reasons"]])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame({"Notes": notes}).to_excel(writer, sheet_name="Notes", index=False)
        levers.to_excel(writer, sheet_name="Levers", index=False)
        scenarios.to_excel(writer, sheet_name="Scenarios", index=False)
        cases.to_excel(writer, sheet_name="Cases", index=False)
        manifest_rows.to_excel(writer, sheet_name="Manifest", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            for column in sheet.columns:
                width = max(len(str(cell.value or "")) for cell in list(column)[:200])
                sheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 8), 60)
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="write the case-settings manifest")
    manifest.add_argument("case_settings")
    manifest.add_argument("--config", required=True, help="the run group's resolved config (the config directory)")
    manifest.add_argument("--run-group", required=True)
    manifest.add_argument("--out", required=True)
    workbook = sub.add_parser("workbook", help="write the lever spreadsheet")
    workbook.add_argument("--config", required=True)
    workbook.add_argument("--run-group", required=True)
    workbook.add_argument("--case-settings", default=None)
    workbook.add_argument("--manifest", default=None)
    workbook.add_argument("--household-totals", default=None)
    workbook.add_argument("--out", required=True)
    for p in (manifest, workbook):
        p.add_argument("--root", default=".", help="repository root the sources are hashed under")
    args = parser.parse_args(argv)
    if args.command == "manifest":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = build_manifest(Path(args.case_settings), Path(args.config), args.run_group, Path(args.root))
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"run_group": args.run_group, "cases": payload["case_settings"]["cases"],
                          "sources": len(payload["sources"])}))
    else:
        counts = build_workbook(config_path=Path(args.config), run_group=args.run_group,
                                case_settings=Path(args.case_settings) if args.case_settings else None,
                                manifest_path=Path(args.manifest) if args.manifest else None,
                                household_totals=Path(args.household_totals) if args.household_totals else None,
                                root=Path(args.root), out=Path(args.out))
        print(json.dumps(counts))


if __name__ == "__main__":
    main()
