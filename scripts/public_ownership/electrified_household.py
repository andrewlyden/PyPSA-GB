"""The electrified household beside the 3,100 kWh benchmark (user, 13 Nov 2026).

Every public-ownership figure per household is on the 3,100 kWh benchmark: a unit-cost index
that holds consumption fixed across 2030-2050. A household with a heat pump and an electric
car uses two to three times that, and pays a standing charge that does not grow with use. This
module gives every case's bill, and its change against private, for the bill's headline
archetype (``household_archetypes.headline_archetype`` in the bill assumptions: a heat-pump
plus EV household, item G6 / task Y13), on the archetypes' own single-rate tariff:

    bill(kWh) = standing charge + (benchmark bill - standing charge) x kWh / benchmark kWh

with each line's standing part ``standing share x its value at the benchmark``
(``scripts/consumer_bill/household_archetypes.py``). So a change on a per-kWh line (wholesale,
CfD, constraints, the S3 tariff's generation lines) scales with the household's use, and a
change on a standing line (TNUoS residual, the standing part of DUoS and supplier costs) does
not.

**The standing charge of each case.**

* Market-bill cases: the benchmark row of the case's own archetype bills (the ownership
  projector writes them per case).
* The S3 tariff cases: private's standing charge less the standing part of the lines the
  tariff replaces or drops, plus that of the lines it adds or adjusts (the tariff bill
  reconciles line by line to its total).
* Bid-stack cases (the P2/P3 balancing months): their base case's. The bid stack moves the
  balancing lines, which are per-kWh lines; the supplier allowance's small move with the bill
  is the approximation.
* The central-dispatch case (I3): its market twin's. The institution step moves wholesale,
  balancing and the forecast-error allowance, all per-kWh lines.

**The taxpayer share.** On the electricity-use key (the headline key) a household's share of
an Exchequer change is in proportion to its use, so the electrified household's share is the
benchmark share times ``kWh / benchmark kWh``. On the legal-incidence key it is the same per
household whatever the household uses.
"""

from __future__ import annotations

from typing import Mapping

BENCHMARK = "benchmark"
#: The headline electrified household and, where the bill declares one, the high-demand one (user, 13 Nov 2026).
PREFIX, HIGH_PREFIX = "electrified", "electrified_high"
#: The bill assumptions' key naming each household.
ARCHETYPE_KEYS = {PREFIX: "headline_archetype", HIGH_PREFIX: "high_demand_archetype"}
_SUFFIXES = ("archetype", "kwh", "bill_inc_vat", "bill_delta_vs_private_inc_vat", "taxpayer_share_gbp_per_household",
             "household_total_delta_vs_private_gbp_inc_vat", "household_total_legal_incidence_delta_vs_private_gbp_inc_vat")
#: Columns this module adds to the household totals (the standing charge and its basis are the case's, shared).
COLUMNS = ("electrified_standing_charge_gbp_ex_vat", "electrified_standing_basis",
           *(f"{prefix}_{suffix}" for prefix in (PREFIX, HIGH_PREFIX) for suffix in _SUFFIXES))


class ElectrifiedHouseholdError(ValueError):
    """The bill does not carry what the electrified household needs."""


def archetype_basis(forecast_summary: Mapping | None, assumptions: Mapping | None,
                    key: str = "headline_archetype") -> dict | None:
    """The named archetype's volume, the benchmark's, private's standing charge and the line shares.

    ``key`` is the summary's key naming the household (``headline_archetype`` or ``high_demand_archetype``);
    ``None`` when the bill declares none (that household is not reported).
    """
    arch = ((forecast_summary or {}).get("diagnostics") or {}).get("household_archetypes") or {}
    name = arch.get(key)
    if not name:
        return None
    bills = {str(b["archetype"]): b for b in arch.get("bills") or []}
    if name not in bills or BENCHMARK not in bills:
        raise ElectrifiedHouseholdError(f"the bill's archetypes have no {name!r} or {BENCHMARK!r} row")
    shares = {str(k): float(v) for k, v in ((arch.get("charge_basis") or {}).get("standing_share") or {}).items()}
    if not shares:
        raise ElectrifiedHouseholdError("the bill's archetype audit carries no standing shares")
    inherit = {str(k): str(v) for k, v in (((assumptions or {}).get("charge_basis") or {}).get("inherit") or {}).items()}
    return {
        "archetype": str(name),
        "kwh": float(bills[name]["annual_consumption_kwh"]),
        "benchmark_kwh": float(bills[BENCHMARK]["annual_consumption_kwh"]),
        "private_standing_ex_vat": float(bills[BENCHMARK]["standing_charge_gbp_ex_vat"]),
        "shares": shares,
        "inherit": inherit,
    }


def share(line: str, basis: Mapping) -> float:
    """A line's standing share: declared, inherited from the line it adjusts, or zero (per kWh)."""
    shares, inherit = basis["shares"], basis.get("inherit") or {}
    return float(shares.get(line, shares.get(inherit.get(line, ""), 0.0)))


def market_standing(case_bill: Mapping | None) -> float | None:
    """The standing charge of a market-bill case: its own benchmark archetype row (ex VAT)."""
    for row in (case_bill or {}).get("household_archetypes") or []:
        if str(row.get("archetype")) == BENCHMARK:
            return float(row["standing_charge_gbp_ex_vat"])
    return None


def tariff_standing(tariff_bill: Mapping, basis: Mapping) -> float:
    """The standing charge of an S3 tariff case, from private's and the tariff's line changes (ex VAT)."""
    def part(key: str) -> float:
        return sum(share(line, basis) * float(value) for line, value in (tariff_bill.get(key) or {}).items())

    return (float(basis["private_standing_ex_vat"])
            - part("replaced_lines_gbp_per_3100kwh_ex_vat")
            - part("dropped_lines_gbp_per_3100kwh_ex_vat")
            + part("new_lines_gbp_per_3100kwh_ex_vat")
            + part("network_adjustments_gbp_per_3100kwh_ex_vat")
            + share("supplier_costs_and_margin", basis)
            * float(tariff_bill.get("supplier_margin_adjustment_gbp_per_3100kwh_ex_vat") or 0.0))


def bill_at(bill_ex_vat: float, standing_ex_vat: float, basis: Mapping) -> float:
    """The electrified household's bill ex VAT: the standing charge plus the unit part scaled to its use."""
    scale = float(basis["kwh"]) / float(basis["benchmark_kwh"])
    return float(standing_ex_vat) + (float(bill_ex_vat) - float(standing_ex_vat)) * scale


def add_columns(rows: list[dict], basis: Mapping | None, *, vat_rate: float, prefix: str = PREFIX) -> None:
    """The electrified household's bill and household totals on every household-totals row (in place).

    ``prefix`` names the columns (``electrified`` for the headline household, ``electrified_high`` for the
    high-demand one); the standing charge is the case's, whatever the household.

    Each row needs ``bill_ex_vat``, ``electrified_standing_charge_gbp_ex_vat`` (set by the
    ledger from the case's stage), ``taxpayer_share_gbp_per_household`` and, with legal
    incidence, ``taxpayer_share_legal_incidence_gbp_per_household``.
    """
    if basis is None:
        return
    by_case = {row["case_id"]: row for row in rows}
    if "private" not in by_case:
        raise ElectrifiedHouseholdError("the household totals carry no private row")
    scale = float(basis["kwh"]) / float(basis["benchmark_kwh"])
    for row in rows:
        if row.get("electrified_standing_charge_gbp_ex_vat") is None:
            raise ElectrifiedHouseholdError(f"{row['case_id']}: no standing charge for the electrified household")
        row[f"{prefix}_bill_inc_vat"] = bill_at(row["bill_ex_vat"], row["electrified_standing_charge_gbp_ex_vat"],
                                                basis) * (1.0 + float(vat_rate))
    private_bill = float(by_case["private"][f"{prefix}_bill_inc_vat"])
    for row in rows:
        row[f"{prefix}_archetype"] = basis["archetype"]
        row[f"{prefix}_kwh"] = float(basis["kwh"])
        delta = float(row[f"{prefix}_bill_inc_vat"]) - private_bill
        row[f"{prefix}_bill_delta_vs_private_inc_vat"] = delta
        row[f"{prefix}_taxpayer_share_gbp_per_household"] = float(row["taxpayer_share_gbp_per_household"]) * scale
        row[f"{prefix}_household_total_delta_vs_private_gbp_inc_vat"] = (
            delta + row[f"{prefix}_taxpayer_share_gbp_per_household"])
        legal = row.get("taxpayer_share_legal_incidence_gbp_per_household")
        row[f"{prefix}_household_total_legal_incidence_delta_vs_private_gbp_inc_vat"] = (
            None if legal is None else delta + float(legal))
