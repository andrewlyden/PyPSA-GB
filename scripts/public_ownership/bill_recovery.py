"""Household <-> system conversions for the public-ownership layer and its reports (F19, 7 Oct 2026).

Since F18 a bill line is converted between GBP per household and system GBP by its own
recovery entry (:mod:`scripts.consumer_bill.line_recovery`), not by one denominator: the
bill spreads lines over metered supply, the GSP charging volume and their eligible shares,
and charges the GSP-rate lines on the household's loss-uplifted 3.1 MWh. This module is the
public-ownership layer's one reader of those entries:

* a bill summary on the two-volume basis (``recovery_basis.basis`` =
  ``loss_inclusive_two_volume``, the 2030 bill) converts through ``line_recovery`` and
  never through the deprecated summary denominators;
* a summary on the D1 basis (the 2050 bill, which declares one metered volume) keeps its
  own denominators, exactly as before F18, so the 2050 figures do not move.

The Exchequer's per-household key (a system cost's household share on the electricity-use
key) is metered supply in both cases: :func:`exchequer_key_mwh`.
"""

from __future__ import annotations

from typing import Mapping

from scripts.consumer_bill.line_recovery import entry_for, household_to_system_gbp, system_to_household_gbp

TWO_VOLUME_BASIS = "loss_inclusive_two_volume"
DEFAULT_CONSUMPTION_KWH = 3100.0

#: Lines a pre-F18 (D1) summary recovered over CfD-eligible supplier sales; every other
#: line converted flat over final demand. Used only for a summary that is not on the
#: two-volume basis.
D1_CFD_ELIGIBLE_LINES = frozenset({
    "cfd", "allocated_ccs", "dpa", "remaining_ccs", "new_build_contracts_at_cost", "public_cfd_at_cost",
    "public_surplus_rebate", "public_portfolio_shortfall_levy", "public_state_deficit_levy",
    "beccs_carbon_removal_levy", "allocated_ccs_public_financing_adjustment",
    "remaining_ccs_public_financing_adjustment",
})


def recovery_basis(summary: Mapping | None) -> str | None:
    """The summary's recovery basis name (``None`` for a summary that declares none)."""
    block = (summary or {}).get("recovery_basis")
    if isinstance(block, Mapping):
        return block.get("basis")
    return None


def is_two_volume(summary: Mapping | None) -> bool:
    """True when the bill summary (or a case summary in it) is on F18's two-volume basis."""
    if recovery_basis(summary) == TWO_VOLUME_BASIS:
        return True
    cases = (summary or {}).get("cases") or {}
    private = cases.get("private") if isinstance(cases, Mapping) else None
    return bool(private) and recovery_basis(private) == TWO_VOLUME_BASIS


def line_recovery_block(summary: Mapping | None, case_id: str = "private") -> dict | None:
    """The ``line_recovery`` block of ``case_id`` in an ownership-case summary, or a bill summary's own."""
    summary = summary or {}
    cases = summary.get("cases")
    if isinstance(cases, Mapping) and case_id in cases and (cases[case_id] or {}).get("line_recovery"):
        return dict(cases[case_id]["line_recovery"])
    if summary.get("line_recovery"):
        return dict(summary["line_recovery"])
    return None


def consumption_kwh(summary: Mapping | None) -> float:
    summary = summary or {}
    for key in ("benchmark_kwh", "annual_consumption_kwh"):
        if summary.get(key) is not None:
            return float(summary[key])
    return DEFAULT_CONSUMPTION_KWH


def _d1_rate(summary: Mapping, component_id: str) -> float:
    """GBP per household per system GBP on the D1 basis: the summary's own two denominators."""
    if is_two_volume(summary):
        raise ValueError(f"{component_id}: the bill summary is on the two-volume basis (F18) but carries no "
                         "line_recovery entry for it; convert through line_recovery, never the deprecated denominators")
    key = ("cfd_recovery_denominator_mwh" if component_id in D1_CFD_ELIGIBLE_LINES
           else "final_demand_recovery_denominator_mwh")
    return float(summary["model_rebasing_factor"]) * consumption_kwh(summary) / 1000.0 / float(summary[key])


def household_per_system(summary: Mapping, component_id: str, *, case_id: str = "private",
                         entry: Mapping | None = None) -> float:
    """GBP per household (report money) per system GBP (model money) of ``component_id``."""
    if entry is None:
        block = line_recovery_block(summary, case_id) if is_two_volume(summary) else None
        if block is not None and component_id in block:
            entry = block[component_id]
        elif block is not None:
            entry = entry_for(block, component_id)   # raises, naming the line
    if entry is not None:
        return system_to_household_gbp(entry, 1.0, consumption_kwh(summary))
    return _d1_rate(summary, component_id)


def system_from_household(summary: Mapping, component_id: str, gbp_per_household: float, *,
                          case_id: str = "private") -> float:
    """System GBP (model money) a household amount of ``component_id`` stands for."""
    if is_two_volume(summary):
        block = line_recovery_block(summary, case_id)
        return household_to_system_gbp(entry_for(block or {}, component_id), gbp_per_household, consumption_kwh(summary))
    return float(gbp_per_household) / _d1_rate(summary, component_id)


def household_from_system(summary: Mapping, component_id: str, system_gbp: float, *, case_id: str = "private") -> float:
    """GBP per household (report money) of a system amount of ``component_id``."""
    return float(system_gbp) * household_per_system(summary, component_id, case_id=case_id)


def exchequer_key_mwh(summary: Mapping) -> float:
    """The Exchequer's per-household key: metered supply (one household's 3.1 MWh is its share of it).

    The two-volume summary states it (``exchequer_per_household_key``); a D1 summary's
    metered supply is its final-demand denominator.
    """
    key = (summary or {}).get("exchequer_per_household_key")
    if isinstance(key, Mapping) and key.get("volume_mwh") is not None:
        return float(key["volume_mwh"])
    cases = (summary or {}).get("cases") or {}
    private = cases.get("private") if isinstance(cases, Mapping) else None
    key = (private or {}).get("exchequer_per_household_key")
    if isinstance(key, Mapping) and key.get("volume_mwh") is not None:
        return float(key["volume_mwh"])
    if is_two_volume(summary):
        raise ValueError("a two-volume bill summary must state exchequer_per_household_key (F18)")
    return float(summary["final_demand_recovery_denominator_mwh"])


def flat_household_per_system(summary: Mapping) -> float:
    """GBP per household per system GBP on the Exchequer's key (a cost spread flat over metered supply)."""
    return float(summary["model_rebasing_factor"]) * consumption_kwh(summary) / 1000.0 / exchequer_key_mwh(summary)


def bill_change_system_gbp(case_id: str, *, bill: Mapping, comparator: str = "private", tariff: Mapping | None = None,
                           institution: Mapping | None = None) -> float | None:
    """A case's ex-VAT bill change against its comparator at system level, each line at its own recovery.

    Market-bill cases: the system amounts of the lines the ownership projector moves
    (:func:`investor_ledger.bill_line_moves`, each reconciled to its route to GBP 1).
    Tariff cases: the tariff's reconciled system change (against private). The institution
    case: its I1 twin's plus the institution step. ``None`` when the case is in none of them
    (the caller then declares its own reading).
    """
    from scripts.public_ownership.investor_ledger import bill_line_moves

    cases = (bill or {}).get("cases") or {}
    if case_id in cases and comparator in cases:
        moves, base = bill_line_moves(cases[case_id]), bill_line_moves(cases[comparator])
        return float(sum(moves.get(k, 0.0) - base.get(k, 0.0) for k in dict.fromkeys([*moves, *base])))
    tariff_cases = (tariff or {}).get("cases") or {}
    if case_id in tariff_cases and comparator == "private":
        recon = tariff_cases[case_id].get("reconciliation") or {}
        if recon.get("bill_change_system_gbp") is not None:
            return float(recon["bill_change_system_gbp"])
    if institution and case_id == institution.get("institution_case_id"):
        twin = institution.get("i1_case_id")
        recon = (tariff_cases.get(twin) or {}).get("reconciliation") or {}
        if recon.get("bill_change_system_gbp") is not None and comparator == "private":
            return float(recon["bill_change_system_gbp"]) + float(institution["institution_change_system_gbp"])
    return None


# ── the D1 basis (2050): the one place the public-ownership layer reads the old names ──────────

def d1_denominators(summary: Mapping, *, from_diagnostics: bool = False) -> dict:
    """A D1 bill summary's two denominators (``final_demand_mwh``, ``cfd_eligible_mwh``).

    Refused on the two-volume basis, where they are deprecated aliases (F18). With
    ``from_diagnostics`` the CfD volume is the one the summary's diagnostics carry (where the
    Exchequer ledger's HPC sensitivity has always read it; ``None`` when absent).
    """
    if is_two_volume(summary):
        raise ValueError("the D1 denominators are deprecated on the two-volume basis (F18); convert through "
                         "line_recovery")
    summary = summary or {}
    diag = summary.get("diagnostics") or {}
    cfd = diag.get("cfd_recovery_denominator_mwh") if from_diagnostics else summary.get("cfd_recovery_denominator_mwh")
    final = summary.get("final_demand_recovery_denominator_mwh")
    return {"final_demand_mwh": float(final) if final is not None else None,
            "cfd_eligible_mwh": float(cfd) if cfd else None}


def d1_recovery_file_volumes(recovery: Mapping, cm_denominator: str = "final_demand_mwh") -> dict:
    """A D1 recovery file's volumes: final demand, RO-eligible supply and the Capacity Market's volume.

    Refused on the two-volume basis (its ``final_demand_mwh`` is a deprecated alias there).
    ``cm_denominator`` is the Capacity Market derivation's ``recovery_denominator`` (item D1:
    ``gsp_chargeable_volume_mwh``, its household factor carrying the losses, or final demand).
    """
    if (recovery or {}).get("recovery_basis") == TWO_VOLUME_BASIS:
        raise ValueError("a two-volume recovery file's final_demand_mwh is a deprecated alias (F18)")
    final = float(recovery["final_demand_mwh"]["value"])
    ro = recovery.get("ro_obligation_eligible_supply_mwh")
    ro_eligible = float(ro["value"] if isinstance(ro, dict) else ro) if ro is not None else final
    if cm_denominator == "gsp_chargeable_volume_mwh":
        basis = (recovery.get("metered_supply_basis") or {}).get("gsp_chargeable_volume_mwh")
        if basis is None:
            raise ValueError("the Capacity Market is recovered over gsp_chargeable_volume_mwh but the recovery file "
                             "declares no metered_supply_basis.gsp_chargeable_volume_mwh")
        cm_volume = float(basis["value"] if isinstance(basis, dict) else basis)
    elif cm_denominator == "final_demand_mwh":
        cm_volume = final
    else:
        raise ValueError(f"unknown Capacity Market recovery denominator {cm_denominator!r} (a two-volume bill converts "
                         "through line_recovery: pass --bill-summary)")
    return {"final_demand_mwh": final, "ro_eligible_mwh": ro_eligible, "cm_volume_mwh": cm_volume}


def recovery_signature(summary: Mapping) -> dict:
    """What two bills must share to be compared: each line's recovery (two-volume) or the D1 denominators."""
    if is_two_volume(summary):
        block = line_recovery_block(summary, "private") or {}
        return {line: (e["volume_key"], round(float(e["volume_mwh"]), 3), float(e["household_loss_factor"]),
                       float(e["household_allocation_factor"])) for line, e in sorted(block.items())}
    return d1_denominators(summary)


def flat_equivalent_mwh(summary: Mapping, component_id: str, *, case_id: str = "private") -> float:
    """The volume over which a line converts flat (``factor x 3.1 MWh / volume`` = its rate).

    For builders written on denominators (the ownership projector's): on the two-volume basis the
    CfD line's is its eligible GSP volume over the household loss factor. On D1, the old denominators.
    """
    rate = household_per_system(summary, component_id, case_id=case_id)
    return float(summary["model_rebasing_factor"]) * consumption_kwh(summary) / 1000.0 / rate
