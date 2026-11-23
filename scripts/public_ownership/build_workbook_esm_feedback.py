"""Items J4 and J5 on workbook v2.0 (F29): what the model hands back to the financing workbook.

Workbook v2.0 (``GB_Power_Model_Rebuild_v5.xlsx``) asks the energy system model for two
things: its own wholesale price for the legacy valuation (the model's 2030 price is the
valuation's central path, lever ``legacy_price_path``) and, on its ``Four Accounts`` tab
(rows 101-110, "enter capital outstanding by technology from the PyPSA coupling"), the
amortising capital outstanding by build-out technology and year.  This script writes, for
the model year, for every case the finance layer projects:

* the private and case bill ex VAT per household on the 3,100 kWh basis and the price gap
  as a fraction of the private bill, with a second gap on the energy stack alone;
* the realised pass-through, on-bill saving over financing saving;
* the workbook's gilt-yield shift (assumption B) on the case's fiscal path (Sc2 for a
  stake, Sc3 for the state developer, Sc4 with the fleet bought), the shift the
  ``sovereign_yield`` lever applies;
* in the JSON, the Four Accounts hand-back: capital outstanding (GBP bn, model price year)
  by the workbook's ten technologies, state and private, per case.  A row's balance is the
  level-annuity loan balance on its capex at its own rate after its age (economic life
  less remaining life): ``capex x (1 - ((1 + r)^age - 1) / ((1 + r)^n - 1))``.

v11-2's CPI-multiplier and assumption-C columns went with its ``CPI Impact`` sheet; v2.0
leaves assumption C blank, pending its input-output model.  Cases that acquire the legacy
fleet take their bill from the acquisition tariff; the central-dispatch case from the
institution comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.levers import (  # noqa: E402
    LEVER_LEGACY_FLEET, LEVER_NEW_BUILD_GENERATION, LEVER_PUBLIC_RETURN, LEVER_SOVEREIGN_YIELD, LEGACY_PRIVATE,
    NEW_BUILD_PRIVATE, NEW_BUILD_PUBLIC, NEW_BUILD_STATE_STAKE, RETURN_CEPA_DECOMPOSITION,
)

from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR  # noqa: E402

#: Kept for callers that import it; the script takes the run's own year (item Y4):
#: ``--model-year``, else the finance register's ``model_year``.
MODEL_YEAR = DEFAULT_MODEL_YEAR
ENERGY_STACKS = ("hedged_wholesale", "hedge_uplifts")
WORKBOOK_SCENARIO = {NEW_BUILD_PRIVATE: "Sc1", NEW_BUILD_STATE_STAKE: "Sc2", NEW_BUILD_PUBLIC: "Sc3"}
#: Workbook v2.0's Four Accounts technologies (rows 101-110, column B: the CEPA technology), with the
#: register finance technologies each gathers.
FOUR_ACCOUNTS_TECHNOLOGIES = {
    "Offshore wind": ("Offshore wind", "Floating offshore wind", "Remote island wind"),
    "Onshore wind": ("Onshore wind",),
    "Solar PV": ("Solar PV",),
    "Large-scale nuclear": ("Large-scale nuclear", "Small modular reactors (SMRs)", "Advanced modular reactors (AMRs)"),
    "Lithium batteries": ("Lithium batteries", "New compound batteries"),
    "Novel long duration energy storage (LDES)": ("Novel long duration energy storage (LDES)",
                                                  "Pumped hydro energy storage (PHES)"),
    "Gas generation - unabated": ("Gas generation - unabated",),
    "Gas generation - with CCUS - mature": ("Gas generation - CCUS - mature", "Gas generation - CCUS - maturing"),
    "Hydrogen CCHT / OCHT - mature": ("Hydrogen CCHT/OCHT - mature", "Hydrogen CCHT/OCHT - emerging"),
    "Biomass - with CCUS - mature": ("Biomass - with CCUS - mature", "Biomass - with CCUS - maturing"),
}


def _value(cell):
    return float(cell["value"] if isinstance(cell, Mapping) else cell)


def _key(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def loan_balance(capex, rate, life, remaining) -> np.ndarray:
    """The level-annuity balance after ``life - remaining`` years: capex x (1 - ((1+r)^a - 1) / ((1+r)^n - 1))."""
    capex, rate = np.asarray(capex, dtype=float), np.asarray(rate, dtype=float)
    life, remaining = np.asarray(life, dtype=float), np.asarray(remaining, dtype=float)
    age = np.clip(life - remaining, 0.0, life)
    with np.errstate(invalid="ignore", divide="ignore"):
        paid = np.where(np.abs(rate) > 1e-12, ((1.0 + rate) ** age - 1.0) / ((1.0 + rate) ** life - 1.0), age / life)
    return np.nan_to_num(capex * (1.0 - paid))


def capital_outstanding(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    """``{Four Accounts technology: {state_gbp_bn, private_gbp_bn}}`` for one case's register rows."""
    def column(name, default=0.0):
        return (pd.to_numeric(rows[name], errors="coerce") if name in rows.columns
                else pd.Series(default, index=rows.index)).fillna(default).to_numpy(dtype=float)

    capex = (column("capacity_mw") * 1_000.0 * column("effective_capex_power_gbp_per_kw")
             + column("energy_capacity_mwh") * 1_000.0 * column("capex_energy_gbp_per_kwh"))
    balance = loan_balance(capex, column("wacc"), column("economic_life_years", np.nan),
                           column("remaining_life_years", np.nan))
    share = np.clip(column("ownership_share"), 0.0, 1.0)
    technology = (rows["finance_technology"].map(_key) if "finance_technology" in rows.columns
                  else pd.Series("", index=rows.index))
    out = {}
    for label, members in FOUR_ACCOUNTS_TECHNOLOGIES.items():
        mask = technology.isin({_key(m) for m in members}).to_numpy()
        out[label] = {"state_gbp_bn": float((balance * share)[mask].sum()) / 1e9,
                      "private_gbp_bn": float((balance * (1.0 - share))[mask].sum()) / 1e9}
    return out


def build_feedback(bill: Mapping, tariff: Mapping | None, institution: Mapping | None, financing: pd.DataFrame,
                   register_settings: Mapping[str, Mapping[str, str]], snapshot: Mapping,
                   components: pd.DataFrame | None = None,
                   model_year: int = DEFAULT_MODEL_YEAR, register_rows=None) -> tuple[pd.DataFrame, dict]:
    """One row per case; see the module docstring for the columns (year-named for ``model_year``, item Y4).

    ``register_rows`` (a callable, case id -> its register rows, or None) adds the Four Accounts hand-back.
    """
    from scripts.public_ownership.finance import yield_analogue, yield_shift_cell

    y = int(model_year)
    cases = bill["cases"]
    # F19 (7 Oct 2026): the bill change at system level, each line at its own recovery
    # (scripts/public_ownership/bill_recovery.py); flat on the Exchequer's key on a D1 bill.
    from scripts.public_ownership import bill_recovery

    factor = float(bill["model_rebasing_factor"])
    exchequer_key = bill_recovery.exchequer_key_mwh(bill)
    mwh = float(bill["benchmark_kwh"]) / 1000.0
    two_volume = bill_recovery.is_two_volume(bill)
    saving = financing.set_index("case_id")
    energy_by_case = (components.loc[components["stack_group"].isin(ENERGY_STACKS)]
                      .groupby("case_id")["annual_gbp_per_3100kwh_ex_vat"].sum()
                      if components is not None else None)

    rows = []
    entries: dict[str, dict] = {}
    for case_id, entry in cases.items():
        entries[case_id] = {"ex_vat": float(entry["ex_vat_gbp"]), "comparator": entry.get("comparator_case_id", "private"),
                            "stage": "market_bill"}
    for case_id, report in (tariff or {}).get("cases", {}).items():
        entries[case_id] = {"ex_vat": float(report["bill"]["ex_vat_gbp"]), "comparator": "private",
                            "stage": "acquisition_tariff"}
    if institution is not None:
        for case_id in register_settings:
            if case_id == institution.get("institution_case_id"):
                entries[case_id] = {"ex_vat": float(institution["i3_ex_vat_gbp"]), "comparator": "private",
                                    "stage": "institution_comparison"}
    capital = {}
    for case_id, info in entries.items():
        levers = register_settings.get(case_id, {})
        scenario = WORKBOOK_SCENARIO.get(levers.get(LEVER_NEW_BUILD_GENERATION, NEW_BUILD_PRIVATE), "Sc1")
        if scenario == "Sc3" and levers.get(LEVER_PUBLIC_RETURN) == RETURN_CEPA_DECOMPOSITION:
            scenario = "Sc2"
        if scenario == "Sc3" and levers.get(LEVER_LEGACY_FLEET, LEGACY_PRIVATE) != LEGACY_PRIVATE:
            scenario = "Sc4"
        comparator_ex_vat = entries.get(info["comparator"], entries["private"])["ex_vat"]
        gap = (comparator_ex_vat - info["ex_vat"]) / comparator_ex_vat if comparator_ex_vat else 0.0
        energy_gap = None
        if energy_by_case is not None and case_id in energy_by_case.index and info["comparator"] in energy_by_case.index:
            base = float(energy_by_case[info["comparator"]])
            energy_gap = (base - float(energy_by_case[case_id])) / base if base else None
        financing_saving = float(saving["annual_financing_saving_vs_private_gbp"].get(case_id, 0.0))
        measured = (bill_recovery.bill_change_system_gbp(case_id, bill=bill, comparator=info["comparator"],
                                                         tariff=tariff, institution=institution)
                    if two_volume else None)
        on_bill_system = (-measured if measured is not None
                          else (comparator_ex_vat - info["ex_vat"]) * exchequer_key / factor / mwh)
        realised_pass_through = on_bill_system / financing_saving if financing_saving else None
        analogue = yield_analogue(levers) if scenario != "Sc1" else None
        rows.append({
            "case_id": case_id, "stage": info["stage"], "workbook_scenario": scenario,
            "comparator_case_id": info["comparator"], "model_year": y,
            "private_bill_ex_vat_gbp_per_3100kwh": comparator_ex_vat, "case_bill_ex_vat_gbp_per_3100kwh": info["ex_vat"],
            "price_gap_vs_private_fraction": gap,
            "energy_stack_gap_vs_private_fraction": energy_gap,
            "yield_b_shift_workbook": (_value(yield_shift_cell(snapshot, analogue, "B", y)) if analogue else 0.0),
            "sovereign_yield_lever": levers.get(LEVER_SOVEREIGN_YIELD, "A"),
            "financing_saving_gbp_2024": financing_saving,
            "on_bill_saving_gbp_2024": on_bill_system,
            "realised_pass_through": realised_pass_through,
        })
        if register_rows is not None and info["stage"] != "institution_comparison":
            case_rows = register_rows(case_id)
            if case_rows is not None and len(case_rows):
                capital[case_id] = capital_outstanding(case_rows)
    frame = pd.DataFrame(rows)
    summary = {
        "model_year": y,
        "workbook": snapshot.get("source_workbook"), "workbook_sha256": snapshot.get("workbook_sha256"),
        "four_accounts_capital_outstanding_gbp_bn": {
            "year": y, "cells": "Four Accounts C101:AB110 (the model-year column)", "by_case": capital},
        "sovereign_yield_in_lever": f"workbook Gilt-Adjusted WACCs, assumption B, {y}",
        "notes": [
            "The price gap is on the ex-VAT bill per household: the CPI's electricity item is the bill, not the "
            "wholesale price. The energy-stack gap is offered beside it.",
            "Capital outstanding is each register row's level-annuity balance at its own rate after its age, "
            "split by its state ownership share; the workbook's own rates by investment year are not re-applied.",
            "Realised pass-through is what the routes actually deliver.",
        ],
    }
    return frame, summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bill_summary")
    parser.add_argument("financing_summary")
    parser.add_argument("register")
    parser.add_argument("snapshot")
    parser.add_argument("out_csv")
    parser.add_argument("out_json")
    parser.add_argument("--bill-components", default=None)
    parser.add_argument("--tariff-summary", default=None)
    parser.add_argument("--institution-summary", default=None)
    parser.add_argument("--model-year", type=int, default=None,
                        help="the run group's model year (item Y4); default: the finance register's model_year")
    args = parser.parse_args(argv)
    bill = json.loads(Path(args.bill_summary).read_text(encoding="utf-8"))
    tariff = json.loads(Path(args.tariff_summary).read_text(encoding="utf-8")) if args.tariff_summary and Path(args.tariff_summary).exists() else None
    institution = (json.loads(Path(args.institution_summary).read_text(encoding="utf-8"))
                   if args.institution_summary and Path(args.institution_summary).exists() else None)
    snapshot = yaml.safe_load(Path(args.snapshot).read_text(encoding="utf-8"))
    financing = pd.read_csv(args.financing_summary)
    from scripts.public_ownership.cfd_repricing import register_settings_by_case

    # 30 Sep 2026: only the case settings are needed, read off the case-settings table, not the register.
    settings = register_settings_by_case(args.register)
    components = pd.read_csv(args.bill_components) if args.bill_components else None
    from scripts.public_ownership.model_year import register_model_year
    from scripts.public_ownership.register_io import CaseRows

    model_year = args.model_year if args.model_year is not None else register_model_year(args.register)
    register = CaseRows.of(args.register)
    frame, summary = build_feedback(bill, tariff, institution, financing, settings, snapshot, components,
                                    model_year=model_year,
                                    register_rows=lambda case_id: register.rows(case_id) if case_id in register.cases else None)
    for path in (args.out_csv, args.out_json):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"cases": len(frame),
                      "four_accounts_cases": len(summary["four_accounts_capital_outstanding_gbp_bn"]["by_case"])}, indent=2))


if __name__ == "__main__":
    main()
