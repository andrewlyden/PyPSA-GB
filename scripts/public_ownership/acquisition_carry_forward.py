"""The 2030 acquisition carried to a later model year (FES 2050 plan, task Y12; decision 2).

Decision 2: the state buys the legacy fleet in 2030, reusing the 2030 purchase, and the
annuity and the surviving fleet carry forward.  In 2050 the state is still paying the
30-year gilt-like annuity it took on in 2030 (the last payment is 2059), whatever of the
fleet survives.  This module turns a 2030 tariff run into what a later year needs:

* **The carried annuity** (:func:`carried_acquisition`): per 2030 tariff case, the
  acquisition lines at their 2030 terms (valued group, thermal and storage, the
  cancelled-contract residual, the interconnectors' GB halves, transaction costs), payable
  in the model year while ``model_year < 2030 + term``.  Every lever variant is its own
  2030 case, so the price bases, the thermal basis, the 15-year term and the rest carry
  as they are: a 2050 case reads the 2030 case with the same id.
* **The survival schedule** (:func:`survival_schedule`): the 2030 legacy valuation's
  own retirement rules (RO run-off to 2037, CfD expiry, retirement at commissioning plus
  life, named overrides), per carrier and year: what the 2030 purchase still operates.
  Acquired plant still operating is costed at operating cost only (its capital is in the
  annuity).
* **The replacement build** (:func:`replacement_build`): acquired plant retired by the
  model year, replaced like for like by state new build at the retirement year's vintage
  cost (``vintage_costs.py``, task Y2) and the state rate (the public rate plus the
  graded construction premium).  Thermal retirements are not replaced like for like
  (FES HT 2050 has no unabated gas): their role is in the 2050 fleet's own new build.
* **The ledger** (:func:`acquisition_ledger`): the model year's cost of what 2030
  bought, by part, with the capacity identity (acquired = surviving + retired, retired =
  replaced + not replaced) and the money identity, which must close.

The replacement build is **also inside the 2050 register's gross additions** (task Y10:
gross additions = the net FES change + that year's retirements).  The ledger reports it
to attribute the 2050 new-build line, never to add to it.

Usage::

    python scripts/public_ownership/acquisition_carry_forward.py <2030_tariff_summary.json>
        <2030_legacy_valuation.csv> <snapshot.yaml> <out.json> [--model-year 2050]
        [--acquisition-year 2030] [--case-settings <2030_case_settings.csv>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.finance import capital_recovery_factor  # noqa: E402

#: The 2030 tariff lines that are the purchase, each annuitised on the case's own term.
LINE_LEGACY_RENEWABLES = "nationalised_legacy_acquisition_renewables_nuclear"
LINE_LEGACY_THERMAL = "nationalised_legacy_acquisition_thermal_storage"
LINE_CANCELLED_RESIDUAL = "nationalised_cancelled_contract_residual"
LINE_IC_ACQUISITION = "nationalised_interconnector_gb_half_acquisition"
LINE_TRANSACTION_COSTS = "nationalised_acquisition_transaction_costs"
CARRIED_LINES = (LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, LINE_CANCELLED_RESIDUAL, LINE_IC_ACQUISITION,
                 LINE_TRANSACTION_COSTS)
#: The lines the multi-year view counts as the annuity (``nationalised_tariff.multi_year``).
MULTI_YEAR_ANNUITY_LINES = (LINE_LEGACY_RENEWABLES, LINE_LEGACY_THERMAL, LINE_CANCELLED_RESIDUAL)
ACQUISITION_YEAR = 2030

GROUP_VALUED, GROUP_THERMAL, GROUP_STORAGE, GROUP_EXCLUDED = "valued", "thermal", "storage", "excluded"
#: legacy_fleet lever -> the valuation groups the 2030 case bought.
ACQUIRED_GROUPS = {"acquire_valued_group": (GROUP_VALUED,),
                   "acquire_all": (GROUP_VALUED, GROUP_THERMAL, GROUP_STORAGE)}
#: Carriers replaced like for like on retirement, and the vintage-table technology they are rebuilt as.
REPLACEMENT_TECHNOLOGY = {
    "wind_onshore": "Onshore wind", "wind_offshore": "Offshore wind", "solar_pv": "Solar PV",
    "large_hydro": "Hydropower", "marine": "Tidal stream", "nuclear": "Large-scale nuclear",
    "Battery": "Lithium batteries", "battery": "Lithium batteries",
    "pumped_hydro": "Pumped hydro energy storage (PHES) - 6h",
}
#: Vintage-table technology -> the workbook snapshot's technology (its CEPA risk rating).
RISK_RATING_TECHNOLOGY = {
    "Onshore wind": "Onshore wind", "Offshore wind": "Offshore wind", "Solar PV": "Solar PV",
    "Hydropower": "Hydropower", "Tidal stream": "Tidal stream", "Large-scale nuclear": "Large-scale nuclear",
    "Lithium batteries": "Lithium batteries",
    "Pumped hydro energy storage (PHES) - 6h": "Pumped hydro energy storage (PHES)",
}
LEDGER_TOLERANCE_GBP = 1.0
LEDGER_TOLERANCE_MW = 1e-6


def annuity_payable(model_year: int, acquisition_year: int, term_years: float) -> bool:
    """Whether a purchase annuitised from ``acquisition_year`` over ``term_years`` is paid in ``model_year``."""
    return int(acquisition_year) <= int(model_year) < int(acquisition_year) + float(term_years)


#: F28 (e): the 2030 tariff's schedule each carried line runs off on (``nationalised_tariff.AnnuitySchedule``).
LINE_SCHEDULE = {LINE_LEGACY_RENEWABLES: "legacy", LINE_LEGACY_THERMAL: "legacy", LINE_CANCELLED_RESIDUAL: "legacy",
                 LINE_IC_ACQUISITION: LINE_IC_ACQUISITION, LINE_TRANSACTION_COSTS: LINE_TRANSACTION_COSTS}


def schedule_live_share(record, offset: int) -> float:
    """The share of a schedule's first-year charge still paid ``offset`` years after the purchase (F28 (e))."""
    if offset < 0:
        return 0.0
    return float(sum(float(share) for term, share in record if offset < float(term)))


def line_live_shares(terms: Mapping, model_year: int, acquisition_year: int) -> dict[str, float]:
    """Each carried line's share of its 2030 amount paid in the model year.

    Each asset is repaid over its own term under the remaining-life term (F28 (e)): the 2030
    tariff's ``annuity_schedules`` give how each line runs off.  A summary without them (one term
    for every asset) pays all of each line or none of it.
    """
    records = dict(terms.get("annuity_schedules") or {})
    offset = int(model_year) - int(acquisition_year)
    if not records:
        whole = 1.0 if annuity_payable(model_year, acquisition_year, float(terms["term_years"])) else 0.0
        return {line: whole for line in CARRIED_LINES}
    legacy = records.get("legacy")
    if legacy is None:
        raise ValueError("the 2030 tariff's annuity_schedules lack the legacy schedule")
    return {line: schedule_live_share(records.get(LINE_SCHEDULE[line], legacy), offset) for line in CARRIED_LINES}


def carried_acquisition(case: Mapping, *, model_year: int, acquisition_year: int = ACQUISITION_YEAR) -> dict:
    """One 2030 tariff case's purchase, as the model year pays it.

    Args:
        case: A 2030 case entry of ``*_nationalised_bill_summary.json`` (``cases[case_id]``).
        model_year: The later model year.
        acquisition_year: The year the purchase was made (decision 2: 2030).

    Returns:
        The case's 2030 lines, the model year's lines (zero once the term has run), the
        purchase prices, the term and the last payment year.

    Raises:
        ValueError: The case is not from the acquisition year, or carries no acquisition terms.
    """
    if int(case.get("model_year", acquisition_year)) != int(acquisition_year):
        raise ValueError(f"{case.get('case_id')}: the carried purchase must come from the {acquisition_year} tariff, "
                         f"not model year {case.get('model_year')}")
    terms = case.get("acquisition_terms") or {}
    if "term_years" not in terms:
        raise ValueError(f"{case.get('case_id')}: the 2030 case carries no acquisition term")
    term = float(terms["term_years"])
    # F28 (e): each line's share still paid in the model year (all or nothing under one term).
    live = line_live_shares(terms, model_year, acquisition_year)
    records = dict(terms.get("annuity_schedules") or {})
    last_term = float(terms.get("last_payment_offset_years", term) or term)
    rr = dict(case.get("revenue_requirement_gbp") or {})
    diag = dict(case.get("diagnostics") or {})
    # Lever acquisition_financing: exchequer moved the annuity off the tariff in 2030; it is
    # still owed, and still off the tariff, in the later year.
    funded = dict(diag.get("acquisition_annuity_exchequer_funded_by_line_gbp") or {})
    lines_2030 = {line: float(rr.get(line, 0.0) or 0.0) for line in CARRIED_LINES}
    payable = any(share > 0.0 for share in live.values())
    lines = {line: value * live[line] for line, value in lines_2030.items()}
    exchequer = {line: float(v) * live.get(line, live[LINE_LEGACY_RENEWABLES]) for line, v in funded.items()}
    # Before the acquisition_financing lever moves anything off the tariff: what the later
    # year's tariff starts from, so the lever there moves the same lines (nationalised_tariff).
    gross = {line: lines[line] + exchequer.get(line, 0.0) for line in CARRIED_LINES}
    ic_terms = dict(diag.get("interconnector_acquisition_terms") or {})
    price = float(diag.get("acquisition_price_gbp", 0.0) or 0.0)
    valued_price = float(diag.get("acquisition_price_renewables_nuclear_gbp", 0.0) or 0.0)
    return {
        "case_id": case.get("case_id"),
        "acquisition_year": int(acquisition_year), "model_year": int(model_year), "term_years": term,
        "term_basis": terms.get("term_basis", "fixed"),
        "last_payment_year": int(acquisition_year + round(last_term) - 1), "payable_in_model_year": payable,
        "line_shares_paid_in_model_year": live,
        "annuity_schedules": records,
        "purchase_price_basis": terms.get("purchase_price_basis"), "thermal_basis": terms.get("thermal_basis"),
        "acquisition_financing": terms.get("acquisition_financing"),
        "purchase_price_gbp": price, "purchase_price_renewables_nuclear_gbp": valued_price,
        "purchase_price_thermal_storage_gbp": float(diag.get("acquisition_price_thermal_storage_gbp", 0.0) or 0.0),
        "interconnector_gb_halves_price_gbp": float(diag.get("interconnector_acquisition_price_gbp", 0.0) or 0.0),
        "public_rate": ic_terms.get("public_rate"),
        "annuity_factor_implied": ((lines_2030[LINE_LEGACY_RENEWABLES] + funded.get(LINE_LEGACY_RENEWABLES, 0.0))
                                   / valued_price if valued_price else None),
        "lines_acquisition_year_gbp": lines_2030,
        # F28 (e): the first year's lines with any annuity the Exchequer funded, for the year path.
        "gross_lines_acquisition_year_gbp": {line: lines_2030[line] + float(funded.get(line, 0.0))
                                             for line in CARRIED_LINES},
        "lines_model_year_gbp": lines,
        "gross_lines_model_year_gbp": gross,
        # Lever legacy_contracts_under_acquisition, exchequer route: the residual owed to
        # former owners is a transfer the Exchequer keeps paying on the purchase's term.
        "cancelled_residual_exchequer_model_year_gbp": (
            float(diag.get("cancelled_contract_residual_exchequer_transfer_gbp", 0.0) or 0.0)
            * live[LINE_CANCELLED_RESIDUAL]),
        "exchequer_funded_model_year_by_line_gbp": exchequer,
        "total_model_year_gbp": float(sum(lines.values()) + sum(exchequer.values())),
        "on_tariff_model_year_gbp": float(sum(lines.values())),
        "multi_year_annuity_gbp": float(sum(lines_2030[line] for line in MULTI_YEAR_ANNUITY_LINES)),
        "basis": ((f"the {acquisition_year} purchase at its {acquisition_year} terms (decision 2): each asset "
                   f"repaid at the public rate over its own remaining life from {acquisition_year} (F28), "
                   f"{live[LINE_LEGACY_RENEWABLES]:.1%} of the legacy annuity still paid in {model_year}, the last "
                   f"in {int(acquisition_year + round(last_term) - 1)}") if records else
                  (f"the {acquisition_year} purchase at its {acquisition_year} terms (decision 2): a {term:g}-year "
                   f"annuity at the public rate from {acquisition_year}, last paid {int(acquisition_year + term - 1)}; "
                   f"paid in {model_year} whatever of the fleet survives" if payable else
                   f"the {acquisition_year} purchase's {term:g}-year annuity ended in "
                   f"{int(acquisition_year + term - 1)}")),
    }


def carry_forward_table(tariff_summary: Mapping, *, model_year: int,
                        acquisition_year: int = ACQUISITION_YEAR) -> dict[str, dict]:
    """:func:`carried_acquisition` for every case of a 2030 tariff summary, keyed on case id."""
    cases = dict(tariff_summary.get("cases") or {})
    if not cases:
        raise ValueError("the 2030 tariff summary carries no cases")
    return {cid: carried_acquisition(c, model_year=model_year, acquisition_year=acquisition_year)
            for cid, c in cases.items()}


# ── survival ──────────────────────────────────────────────────────────────────

def effective_retirement(valuation: pd.DataFrame, *, asset_lives: str = "central",
                         extended: Mapping | None = None) -> pd.Series:
    """Each valued row's retirement year under the case's ``legacy_asset_lives`` lever.

    The valuation's own retirement year is the central rule.  ``extended`` (the
    ``acquisition.extended_asset_lives`` block) adds ``extra_years`` for its carriers on
    rows dated by commissioning plus life, and applies its named overrides, as
    ``plan_acquisition`` does for the 2030 price.
    """
    retire = pd.to_numeric(valuation["retirement_year"], errors="coerce").astype(float)
    if str(asset_lives) != "extended":
        return retire
    extended = dict(extended or {})
    carriers = set(extended.get("carriers") or ())
    method = valuation["retirement_method"].astype(str)
    dated = method.isin(["commissioning_plus_life", "carrier_median_commissioning_plus_life"])
    bump = valuation["carrier"].astype(str).isin(carriers) & dated
    retire = retire.where(~bump, retire + float(extended.get("extra_years", 0.0)))
    base = valuation["asset_id"].astype(str).str.split("::", n=1).str[0]
    for name, year in dict(extended.get("retirement_year_overrides") or {}).items():
        retire[base.eq(str(name))] = float(year)
    return retire


def repowering_pieces(rows: pd.DataFrame) -> pd.DataFrame:
    """FES 2050 decision 62: a valuation's rows split into their repowered and life-extended parts.

    A valuation written with the repowering rule (``legacy_valuation.repowering_weights``)
    carries ``repowering_year`` and ``repowering_share`` on the rows it applies to: that share
    of the row leaves at ``repowering_year`` (when it is after the valuation's first year) and
    the rest at ``retire``. Other rows, and every valuation without the columns, are returned
    as they are (one piece each, ``retire`` unchanged).
    """
    if "repowering_year" not in rows.columns or "repowering_share" not in rows.columns:
        return rows
    decision = pd.to_numeric(rows["repowering_year"], errors="coerce")
    share = pd.to_numeric(rows["repowering_share"], errors="coerce")
    split = decision.notna() & share.notna() & decision.lt(rows["retire"])
    if not split.any():
        return rows
    capacity = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0)
    early = rows.loc[split].assign(capacity_mw=capacity[split] * share[split], retire=decision[split])
    late = rows.loc[split].assign(capacity_mw=capacity[split] * (1.0 - share[split]))
    return pd.concat([rows.loc[~split], early, late]).sort_index(kind="mergesort")


def survival_schedule(valuation: pd.DataFrame, years: Iterable[int], *, groups: Iterable[str],
                      asset_lives: str = "central", extended: Mapping | None = None) -> pd.DataFrame:
    """Capacity the purchase still operates, per group, carrier and year.

    A row operates in year ``y`` while ``y < retirement_year`` (the valuation's rule:
    ``alive = years < retire``); its CfD is live while ``y < cfd_contract_end_year``.

    Args:
        valuation: The 2030 ``*_legacy_valuation.csv``.
        years: The years to report.
        groups: The valuation groups the case bought (:data:`ACQUIRED_GROUPS`).
        asset_lives: The case's ``legacy_asset_lives`` lever.
        extended: The ``acquisition.extended_asset_lives`` block.

    Returns:
        One row per (year, scope_group, carrier): acquired, operating and retired MW, and
        the MW still on a live CfD.
    """
    rows = valuation.loc[valuation["scope_group"].astype(str).isin(list(groups))].copy()
    rows["retire"] = effective_retirement(rows, asset_lives=asset_lives, extended=extended)
    rows = repowering_pieces(rows)
    cfd_end = pd.to_numeric(rows["cfd_contract_end_year"], errors="coerce")
    capacity = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0)
    out = []
    for year in years:
        alive = rows["retire"].gt(year)
        cfd = alive & cfd_end.gt(year).fillna(False)
        frame = pd.DataFrame({"scope_group": rows["scope_group"], "carrier": rows["carrier"],
                              "acquired_mw": capacity, "operating_mw": capacity.where(alive, 0.0),
                              "retired_mw": capacity.where(~alive, 0.0), "cfd_live_mw": capacity.where(cfd, 0.0)})
        grouped = frame.groupby(["scope_group", "carrier"], as_index=False).sum()
        grouped.insert(0, "year", int(year))
        out.append(grouped)
    return pd.concat(out, ignore_index=True)


# ── replacement ───────────────────────────────────────────────────────────────

def state_rate(technology: str, public_rate: float, snapshot: Mapping | None, *,
               construction_premium: str = "none") -> float:
    """The rate the state rebuilds a retiring plant at: its cash rate (F29, 26 Oct 2026: workbook v2.0 charges no
    construction premium; ``construction_premium`` is kept as an argument for callers and ignored)."""
    return float(public_rate)


def replacement_build(valuation: pd.DataFrame, model_year: int, public_rate: float, *,
                      groups: Iterable[str], snapshot: Mapping | None = None, asset_lives: str = "central",
                      extended: Mapping | None = None, vintage_table: pd.DataFrame | None = None,
                      learning: bool = False, construction_premium: str = "graded") -> pd.DataFrame:
    """Acquired plant retired by ``model_year`` and rebuilt like for like by the state.

    Each retired row of a replaceable carrier (:data:`REPLACEMENT_TECHNOLOGY`) is rebuilt in
    its retirement year (held at 2025 or later, the vintage table's first year) at that
    vintage's cost on the full DESNZ basis and at :func:`state_rate`; its model-year capital
    charge is ``capex x CRF(state rate, life)`` while the vintage is within its life.
    Storage energy is the vintage row's duration times power.

    Returns:
        One row per (scope_group, carrier, technology, build_year) with MW, capex, the
        rate, the life, and the model-year capital and fixed O&M.
    """
    from scripts.public_ownership.vintage_costs import costs_for, load_vintage_costs

    table = load_vintage_costs() if vintage_table is None else vintage_table
    rows = valuation.loc[valuation["scope_group"].astype(str).isin(list(groups))].copy()
    rows["retire"] = effective_retirement(rows, asset_lives=asset_lives, extended=extended)
    rows = repowering_pieces(rows)
    retired = rows.loc[rows["retire"].le(model_year) & rows["carrier"].astype(str).isin(REPLACEMENT_TECHNOLOGY)]
    if retired.empty:
        return pd.DataFrame(columns=["scope_group", "carrier", "technology", "build_year", "capacity_mw"])
    retired = retired.assign(build_year=retired["retire"].clip(lower=2025).astype(int),
                             technology=retired["carrier"].astype(str).map(REPLACEMENT_TECHNOLOGY))
    grouped = retired.groupby(["scope_group", "carrier", "technology", "build_year"], as_index=False)["capacity_mw"].sum()
    out = []
    for r in grouped.itertuples(index=False):
        cost = costs_for(r.technology, int(r.build_year), learning=learning, table=table)
        life = float(cost["economic_life_years"])
        rate = state_rate(r.technology, public_rate, snapshot, construction_premium=construction_premium)
        duration = float(cost["duration_hours"]) if pd.notna(cost.get("duration_hours")) else 0.0
        capex = (float(r.capacity_mw) * 1e3 * float(cost["capex_power_gbp_per_kw"])
                 + float(r.capacity_mw) * duration * 1e3 * float(cost.get("capex_energy_gbp_per_kwh") or 0.0))
        alive = int(r.build_year) <= model_year < int(r.build_year) + life
        out.append({
            "scope_group": r.scope_group, "carrier": r.carrier, "technology": r.technology,
            "build_year": int(r.build_year), "capacity_mw": float(r.capacity_mw), "capex_gbp": capex,
            "state_rate": rate, "economic_life_years": life,
            "annual_capital_gbp": capex * capital_recovery_factor(rate, life) if alive else 0.0,
            "annual_fixed_om_gbp": (float(r.capacity_mw) * 1e3 * float(cost["fixed_om_gbp_per_kw_year"])
                                    + float(r.capacity_mw) * duration * 1e3
                                    * float(cost.get("fixed_om_gbp_per_kwh_year") or 0.0)) if alive else 0.0,
            "cost_source": cost.get("source"), "cost_status": cost.get("status"),
        })
    return pd.DataFrame(out)


# ── the ledger ────────────────────────────────────────────────────────────────

def acquisition_ledger(carried: Mapping, survival: pd.DataFrame, replacement: pd.DataFrame, model_year: int, *,
                       surviving_operating_cost_gbp: float | None = None) -> dict:
    """The model year's cost of what the acquisition year bought, by part, and its identities.

    Parts: the carried annuity (on the tariff and, under ``acquisition_financing:
    exchequer``, on the Exchequer), the surviving plant's operating cost (from the model
    year's register when given; its capital is zero, being in the annuity), and the
    replacement build's capital and fixed O&M (an attribution of the model year's
    new-build lines, never an addition to them).

    Identities (raise when broken): acquired MW = operating MW + retired MW per carrier;
    retired MW = replaced MW + retired MW not replaced like for like; the parts sum to the
    total.

    Returns:
        The ledger.
    """
    year_rows = survival.loc[survival["year"].eq(int(model_year))]
    if year_rows.empty:
        raise ValueError(f"the survival schedule does not reach {model_year}")
    acquired = float(year_rows["acquired_mw"].sum())
    operating = float(year_rows["operating_mw"].sum())
    retired = float(year_rows["retired_mw"].sum())
    replaced = float(replacement["capacity_mw"].sum()) if not replacement.empty else 0.0
    not_replaceable = year_rows.loc[~year_rows["carrier"].astype(str).isin(REPLACEMENT_TECHNOLOGY), "retired_mw"].sum()
    capacity_residuals = {
        "acquired_minus_operating_minus_retired_mw": acquired - operating - retired,
        "retired_minus_replaced_minus_not_replaced_mw": retired - replaced - float(not_replaceable),
    }
    parts = {
        "carried_annuity_on_tariff_gbp": float(carried["on_tariff_model_year_gbp"]),
        "carried_annuity_exchequer_funded_gbp": float(sum(carried["exchequer_funded_model_year_by_line_gbp"].values())),
        "cancelled_residual_exchequer_transfer_gbp": float(carried.get("cancelled_residual_exchequer_model_year_gbp", 0.0)),
        "surviving_acquired_plant_capital_gbp": 0.0,
        "surviving_acquired_plant_operating_cost_gbp": float(surviving_operating_cost_gbp or 0.0),
        "replacement_build_capital_gbp": float(replacement["annual_capital_gbp"].sum()) if not replacement.empty else 0.0,
        "replacement_build_fixed_om_gbp": float(replacement["annual_fixed_om_gbp"].sum()) if not replacement.empty else 0.0,
    }
    total = float(sum(parts.values()))
    money_residual = total - (float(carried["total_model_year_gbp"]) + parts["cancelled_residual_exchequer_transfer_gbp"]
                              + parts["surviving_acquired_plant_operating_cost_gbp"]
                              + parts["replacement_build_capital_gbp"] + parts["replacement_build_fixed_om_gbp"])
    if max(abs(v) for v in capacity_residuals.values()) > LEDGER_TOLERANCE_MW or abs(money_residual) > LEDGER_TOLERANCE_GBP:
        raise ValueError(f"the acquisition ledger for {model_year} does not close: {capacity_residuals}, money "
                         f"{money_residual:,.2f}")
    by_carrier = year_rows.groupby("carrier")[["acquired_mw", "operating_mw", "retired_mw"]].sum()
    if not replacement.empty:
        by_carrier = by_carrier.join(replacement.groupby("carrier")["capacity_mw"].sum().rename("replaced_mw"))
    by_carrier = by_carrier.fillna(0.0)
    return {
        "model_year": int(model_year), "acquisition_year": int(carried["acquisition_year"]),
        "case_id": carried.get("case_id"),
        "parts_gbp": parts, "total_gbp": total,
        "annuity_gbp": float(carried["total_model_year_gbp"]),
        "capacity_mw": {"acquired": acquired, "operating": operating, "retired": retired, "replaced": replaced,
                        "retired_not_replaced_like_for_like": float(not_replaceable)},
        "capacity_by_carrier_mw": {c: {k: float(v) for k, v in row.items()} for c, row in by_carrier.iterrows()},
        "capacity_residuals_mw": capacity_residuals, "money_residual_gbp": money_residual,
        "closes": True,
        "notes": [
            f"The annuity is paid in {model_year} whatever survives; surviving plant is costed at operating cost "
            "only (its capital is in the annuity).",
            "The replacement build attributes part of the model year's new-build lines (the register's gross additions "
            "include replacements, task Y10): it is never added to them.",
            "Thermal retirements are not replaced like for like: FES HT 2050 has no unabated gas.",
        ],
    }


def check_register_survivors(register_legacy: pd.DataFrame, survival: pd.DataFrame, model_year: int,
                             tolerance_share: float = 0.05) -> dict:
    """Compare the model year's register legacy capacity by carrier with the survival schedule (reported, not enforced).

    Args:
        register_legacy: The model year's legacy register rows (``carrier``, ``capacity_mw``).
        survival: :func:`survival_schedule`.
        model_year: The model year.
        tolerance_share: The share within which a carrier counts as matching.

    Returns:
        Per carrier: register MW, schedule MW and whether they match.
    """
    schedule = survival.loc[survival["year"].eq(int(model_year))].groupby("carrier")["operating_mw"].sum()
    register = register_legacy.groupby("carrier")["capacity_mw"].sum()
    out = {}
    for carrier in sorted(set(schedule.index) | set(register.index)):
        s, r = float(schedule.get(carrier, 0.0)), float(register.get(carrier, 0.0))
        out[str(carrier)] = {"register_mw": r, "schedule_mw": s,
                             "matches": abs(r - s) <= tolerance_share * max(abs(s), 1.0)}
    return out


def register_survivor_report(register_assets: pd.DataFrame, valuation: pd.DataFrame, model_year: int,
                             years: Iterable[int]) -> dict:
    """:func:`check_register_survivors` on a vintaged register's legacy rows, with a count of mismatches.

    The register's legacy rows are its pre-policy cohorts (``finance_vintage == legacy``,
    ``capacity_mw`` by network carrier), against the 2030 purchase's survival schedule for
    every acquired group in the model year. Reported, never enforced: the register's
    legacy fleet comes from the FES stock-flow and the schedule from the 2030 valuation's
    retirement rules, so a gap is a finding about the two rules.
    """
    legacy = register_assets.loc[register_assets["finance_vintage"].astype(str).eq("legacy")
                                 & register_assets["component_type"].astype(str).isin(["Generator", "StorageUnit"])]
    schedule = survival_schedule(valuation, years, groups=ACQUIRED_GROUPS["acquire_all"])
    by_carrier = check_register_survivors(legacy[["carrier", "capacity_mw"]], schedule, int(model_year))
    return {"by_carrier": by_carrier,
            "carriers_not_matching": sorted(c for c, v in by_carrier.items() if not v["matches"]),
            "register_legacy_mw": float(pd.to_numeric(legacy["capacity_mw"], errors="coerce").sum()),
            "schedule_operating_mw": float(schedule.loc[schedule["year"].eq(int(model_year)), "operating_mw"].sum()),
            "rule": "reported, not enforced (tolerance 5% of the schedule by carrier)"}


# ── CLI ───────────────────────────────────────────────────────────────────────

def build(tariff_summary: Mapping, valuation: pd.DataFrame, snapshot: Mapping | None, *, model_year: int,
          acquisition_year: int = ACQUISITION_YEAR, acquisition_settings: Mapping | None = None,
          vintage_table: pd.DataFrame | None = None, register_assets: pd.DataFrame | None = None) -> dict:
    """Every 2030 tariff case carried to ``model_year``, with its survival and replacement ledger.

    ``register_assets`` (the model year's vintaged register, task SETTLE2050) adds
    ``register_survivor_check``: the register's legacy capacity by carrier against the
    survival schedule (:func:`check_register_survivors`, reported, not enforced).
    """
    table = carry_forward_table(tariff_summary, model_year=model_year, acquisition_year=acquisition_year)
    public_rate = float(valuation["public_real_rate"].iloc[0])
    horizon = int(valuation["horizon_year"].iloc[0]) if "horizon_year" in valuation.columns else None
    years = sorted({acquisition_year, 2035, 2037, 2040, 2045, int(model_year), acquisition_year + 29})
    cache: dict = {}
    out_cases = {}
    for case_id, carried in table.items():
        case = tariff_summary["cases"][case_id]
        levers = dict(case.get("levers") or {})
        groups = ACQUIRED_GROUPS.get(str(levers.get("legacy_fleet", "acquire_all")), ACQUIRED_GROUPS["acquire_all"])
        lives = str(levers.get("legacy_asset_lives", "central"))
        premium = str(levers.get("construction_premium", "graded"))
        rate = float(carried["public_rate"]) if carried.get("public_rate") is not None else public_rate
        key = (groups, lives, premium, round(rate, 12))
        if key not in cache:
            extended = (acquisition_settings or {}).get("extended_asset_lives") or case.get(
                "acquisition_settings", {}).get("extended_asset_lives")
            survival = survival_schedule(valuation, years, groups=groups, asset_lives=lives, extended=extended)
            replacement = replacement_build(valuation, model_year, rate, groups=groups, snapshot=snapshot,
                                            asset_lives=lives, extended=extended, vintage_table=vintage_table,
                                            construction_premium=premium)
            cache[key] = (survival, replacement)
        survival, replacement = cache[key]
        ledger = acquisition_ledger(carried, survival, replacement, model_year)
        out_cases[case_id] = {**carried, "ledger": ledger}
    first = next(iter(cache.values()))
    return {
        "model_year": int(model_year), "acquisition_year": int(acquisition_year),
        "valuation_horizon_year": horizon,
        "cases": out_cases,
        "survival_schedule_all_groups": survival_schedule(valuation, years, groups=ACQUIRED_GROUPS["acquire_all"])
                                        .to_dict(orient="records"),
        "replacement_build_first_case": first[1].to_dict(orient="records"),
        **({"register_survivor_check": register_survivor_report(register_assets, valuation, model_year, years)}
           if register_assets is not None else {}),
        "notes": [
            "Decision 2: the 2030 purchase at its 2030 terms; each 2050 tariff case reads the 2030 case with its id.",
            "Survival uses the 2030 legacy valuation's retirement rules (RO run-off to 2037, CfD expiry, retirement "
            "at commissioning plus life, named overrides), with the extended-lives lever applied as in the 2030 price.",
        ],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tariff_summary", help="the acquisition year's *_nationalised_bill_summary.json")
    parser.add_argument("legacy_valuation", help="the acquisition year's *_legacy_valuation.csv")
    parser.add_argument("snapshot", help="the workbook assumptions snapshot (construction premium by risk rating)")
    parser.add_argument("out")
    parser.add_argument("--model-year", type=int, default=2050)
    parser.add_argument("--acquisition-year", type=int, default=ACQUISITION_YEAR)
    parser.add_argument("--register-assets", default=None,
                        help="the model year's vintaged asset inputs: report its legacy rows against the survival schedule")
    args = parser.parse_args(argv)
    summary = json.loads(Path(args.tariff_summary).read_text(encoding="utf-8"))
    valuation = pd.read_csv(args.legacy_valuation)
    snapshot = yaml.safe_load(Path(args.snapshot).read_text(encoding="utf-8"))
    register = pd.read_csv(args.register_assets, low_memory=False) if args.register_assets else None
    result = build(summary, valuation, snapshot, model_year=args.model_year, acquisition_year=args.acquisition_year,
                   register_assets=register)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    primary = summary.get("primary_case_id")
    entry = result["cases"].get(primary) or next(iter(result["cases"].values()))
    print(json.dumps({"cases": len(result["cases"]), "primary": entry["case_id"],
                      "annuity_model_year_gbp_bn": round(entry["total_model_year_gbp"] / 1e9, 3),
                      "operating_mw": round(entry["ledger"]["capacity_mw"]["operating"], 1),
                      "replaced_mw": round(entry["ledger"]["capacity_mw"]["replaced"], 1)}))


if __name__ == "__main__":
    main()
