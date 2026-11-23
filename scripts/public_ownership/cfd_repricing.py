"""Route the financing saving to the consumer bill: CfD re-pricing and reconciliation.

Principles applied here:

* every pound of the financing saving is routed or declared:
  ``financing saving = bill-allocated + public-account + unallocated``;
* additive, not proportional: a strike falls by the cohort's capital-charge
  change per MWh of the dispatch settling at that strike;
* an asset is re-priced (route R1) or its saving goes to the public account
  (route R2), never both;
* the private case must reproduce the forecast case exactly, so the case
  strikes are fed through the same calculator as the private strikes.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.market.cfd_strike_table import StrikeTable, contract_technology
from scripts.public_ownership.finance import NATIONALISED_CASE, capital_recovery_factor  # noqa: F401
# base_case and is_variant are re-exported here: several modules have imported
# them from this one since before the axes existed.
from scripts.public_ownership.levers import (  # noqa: F401
    AXIS_CFD_AMORTISATION, AXIS_CFD_STRIKE_BASIS, DEFAULT_SETTINGS, LEVER_CFD_ELIGIBILITY, LEVER_PRE_2025_CFD,
    GATED_LEVERS, LEVERS, PRE_2025_STATE_OWNED, axis_setting, base_case, case_settings, comparator_for_settings, is_variant,
    lever_default, spec_for_row,
)
# F27 item 1 (24 Oct 2026): Net Zero Teesside's pre-2025 DPA joins the pre-2025 option.
from scripts.public_ownership.levers import PRE_2025_DPA_CONTRACTS, PRE_2025_DPA_TERM_YEARS, is_pre_2025_dpa
from scripts.public_ownership.levers import OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT
# F20 phase 2 (27 Oct 2026): the savings once per distinct fingerprint, broadcast to the cases on it.
from scripts.public_ownership import fingerprints as _fp

AXES = LEVERS
_LOG = logging.getLogger(__name__)


# Routes, in reporting order.  Exactly one applies to every register row.
ROUTE_CFD_BILL = "R1_cfd_repricing_bill"
#: F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA re-priced to the state's cost on the bill's
#: remaining_ccs line, as R1 re-prices a CfD strike (the public account treats both as R1).
ROUTE_DPA_BILL = "R1_dpa_repricing_bill"
ROUTE_CFD_PUBLIC_ACCOUNT = "R2_cfd_public_account"
ROUTE_STORAGE = "R3_storage_no_bill_line"
ROUTE_CAPACITY_MARKET = "capacity_market_route_not_modelled"
ROUTE_DISPATCHABLE_CFD = "dispatchable_cfd_route_not_modelled"
ROUTE_MERCHANT = "merchant_generation_unallocated"
ROUTE_UNRESOLVED = "unresolved_cost_or_finance_mapping"
ROUTES = (
    ROUTE_CFD_BILL, ROUTE_DPA_BILL, ROUTE_CFD_PUBLIC_ACCOUNT, ROUTE_STORAGE, ROUTE_CAPACITY_MARKET,
    ROUTE_DISPATCHABLE_CFD, ROUTE_MERCHANT, ROUTE_UNRESOLVED,
)
#: The routes whose saving the bill takes by re-pricing a contract (a strike, or the DPA's payments).
BILL_REPRICING_ROUTES = (ROUTE_CFD_BILL, ROUTE_DPA_BILL)
ROUTE_TREATMENT = {
    ROUTE_CFD_BILL: "bill_allocated",
    ROUTE_DPA_BILL: "bill_allocated",
    ROUTE_CFD_PUBLIC_ACCOUNT: "public_account_pending_settlement_link",
    ROUTE_STORAGE: "unallocated",
    ROUTE_CAPACITY_MARKET: "unallocated",
    ROUTE_DISPATCHABLE_CFD: "unallocated",
    ROUTE_MERCHANT: "unallocated",
    ROUTE_UNRESOLVED: "unallocated",
}
ROUTE_CLASSIFICATIONS = {"capacity_market": ROUTE_CAPACITY_MARKET, "dispatchable_cfd": ROUTE_DISPATCHABLE_CFD}
IDENTITY_TOLERANCE_GBP = 1.0

REQUIRED_REGISTER_COLUMNS = {
    "case_id", "asset_id", "component_type", "carrier", "support_type", "finance_vintage",
    "contract_vintage", "cfd_round", "evidence_status", "dispatch_mwh", "annual_required_revenue_gbp",
    "annual_capital_recovery_gbp", "wacc", "remaining_life_years",
}


def _crf(rates: np.ndarray, years: np.ndarray) -> np.ndarray:
    """The capital recovery factor per row (NaN without a finite rate and a positive life), once per distinct pair."""
    rates = np.asarray(rates, dtype=float)
    years = np.asarray(years, dtype=float)
    out = np.full(len(rates), np.nan)
    usable = np.isfinite(rates) & (years > 0)
    if usable.any():
        pairs, inverse = np.unique(np.column_stack([rates[usable], years[usable]]), axis=0, return_inverse=True)
        values = np.array([capital_recovery_factor(float(r), float(y)) for r, y in pairs])
        out[usable] = values[inverse.ravel()]
    return out


def _blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


VARIANT_SEPARATOR = "@"
AXIS_COLUMNS = tuple(AXES)
STRIKE_BASIS_AT_COST = "at_cost"
#: F10: the register's gbe_part value for GB Energy's offshore stake (calculate_financing_scenarios).
GBE_PART_STAKE = "offshore_stake"
#: User decision 5 (29 Sep 2026): the state keeps the auction-cleared strike, so its
#: owned CfDs are not re-priced at all and join the public account as R2.
STRIKE_BASIS_AUCTION_CLEARING = "auction_clearing"
ELIGIBILITY_CONTRACT_VINTAGE = "contract_vintage"
#: Carriers whose required revenue covers a leg the electricity CfD does not pay
#: (config ``cfd_repricing.at_cost_electricity_leg_carriers``). Power BECCS: the
#: register costs the whole plant, capture included, but the electricity CfD
#: covers the electricity leg only; capture and removals are paid off the bill by
#: the carbon-removal contract (DESNZ GGR and Power BECCS business model, Dec
#: 2023). At cost, such a contract gives the bill its electricity leg's share
#: of required revenue, not the whole of it.
DEFAULT_ELECTRICITY_LEG_CARRIERS = ("CCS Biomass",)


def repricing_settings(block: Mapping) -> dict:
    """Read and check the ``public_ownership.consumer_bill`` parameters used here.

    Item K1: the eligibility rule and the amortisation are levers read from each
    case's register rows (``cfd_repricing_eligibility``, ``cfd_amortisation``);
    the values kept here are the defaults a register written without those
    columns falls back to.  Only the contract term and the route classification
    are parameters.
    """
    repricing = block.get("cfd_repricing") or {}
    # The eligibility lever was retired on 29 Sep 2026 (review C1: structurally inert);
    # contract vintage is the rule, and commissioning vintage is reached through
    # pre_2025_cfd_capacity: public_repriced.  The key is still read for old callers.
    eligibility = repricing.get("eligibility", ELIGIBILITY_CONTRACT_VINTAGE)
    if eligibility not in (ELIGIBILITY_CONTRACT_VINTAGE, "commissioning_vintage"):
        raise ValueError(f"cfd_repricing.eligibility must be contract_vintage or commissioning_vintage, got {eligibility!r}")
    amortisation = repricing.get("amortisation", DEFAULT_SETTINGS[AXIS_CFD_AMORTISATION])
    if amortisation not in LEVERS[AXIS_CFD_AMORTISATION].values:
        raise ValueError(f"cfd_repricing.amortisation must be contract_term or economic_life, got {amortisation!r}")
    term = repricing.get("contract_term_years")
    if isinstance(term, bool) or not isinstance(term, int) or term <= 0:
        raise ValueError("cfd_repricing.contract_term_years must be a positive integer")
    by_round = contract_terms_by_round(repricing)
    corrections = block.get("label_corrections") or {}
    leg_carriers = repricing.get("at_cost_electricity_leg_carriers", list(DEFAULT_ELECTRICITY_LEG_CARRIERS))
    if not isinstance(leg_carriers, (list, tuple)):
        raise ValueError("cfd_repricing.at_cost_electricity_leg_carriers must be a list of carriers")
    return {
        "electricity_leg_carriers": [str(c) for c in leg_carriers],
        "eligibility": eligibility,
        "amortisation": amortisation,
        "contract_term_years": int(term),
        "contract_term_years_by_round": by_round,
        "eligible_award_years_from": int(repricing.get("eligible_award_years_from", 2025)),
        "route_classification": {str(k): str(v) for k, v in (corrections.get("route_classification") or {}).items()},
        "method": f"{eligibility};{amortisation}" + (f"_{_term_label(term, by_round)}" if amortisation == "contract_term" else ""),
        "sensitivities": [],
    }


def contract_terms_by_round(repricing: Mapping) -> dict[str, int]:
    """``cfd_repricing.contract_term_years_by_round``: a term per allocation round, checked.

    CfD terms are round-specific: 15 years to AR6, 20 years from AR7 for offshore
    wind, onshore wind and solar (DESNZ, AR7 core parameters, 2025).  A round not
    listed takes ``contract_term_years``.  Absent, every contract takes that one
    term, as before 26 Sep 2026.
    """
    raw = repricing.get("contract_term_years_by_round") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("cfd_repricing.contract_term_years_by_round must map a round to a term in years")
    out = {}
    for cfd_round, years in raw.items():
        if isinstance(years, bool) or not isinstance(years, int) or years <= 0:
            raise ValueError(f"cfd_repricing.contract_term_years_by_round.{cfd_round} must be a positive integer")
        out[str(cfd_round)] = int(years)
    return out


def _term_label(term: int, by_round: Mapping[str, int]) -> str:
    return f"{term}y" + ("".join(f"_{r}{y}y" for r, y in sorted(by_round.items())) if by_round else "")


def contract_term_by_row(cfd_round: pd.Series, settings: Mapping) -> pd.Series:
    """Each row's contract term in years: its round's term, else ``contract_term_years``.

    A pre-2025 DPA (Net Zero Teesside, F27 item 1) takes its own 15-year term.
    """
    by_round = {**{k: PRE_2025_DPA_TERM_YEARS for k in PRE_2025_DPA_CONTRACTS},
                **(settings.get("contract_term_years_by_round") or {})}
    default = float(settings["contract_term_years"])
    return _blank(cfd_round).map(lambda r: float(by_round.get(r, default))).astype(float)


def savings_with_sensitivities(register: pd.DataFrame, settings: Mapping) -> pd.DataFrame:
    """Kept for callers of the item-D1 name: every sensitivity is a full case in the register now."""
    return per_row_savings(register, settings)


def _case_savings(case_id: str, rows: pd.DataFrame, settings: Mapping, context: "SavingsContext") -> pd.DataFrame:
    """One market case's savings rows before the route rules (the per-case step of :func:`per_row_savings`)."""
    # Contract revenue is a physical settlement of the run, so it is read once
    # from the private rows and shared by every case (house rule 2).
    has_settlement = context.has_settlement
    _private_for = context.private_for
    private = _private_for(case_id)
    contract_revenue = context.contract_revenue_for(case_id)
    private_rr = pd.to_numeric(private["annual_required_revenue_gbp"], errors="coerce")
    private_capital = pd.to_numeric(private["annual_capital_recovery_gbp"], errors="coerce")
    private_wacc = pd.to_numeric(private["wacc"], errors="coerce")
    remaining = pd.to_numeric(private["remaining_life_years"], errors="coerce")
    capex_total = private_capital / _crf(private_wacc.to_numpy(), remaining.fillna(0).to_numpy())
    # Each contract's term, never past the asset's remaining life.
    term_years = np.minimum(contract_term_by_row(private["cfd_round"], settings).to_numpy(),
                            remaining.fillna(np.inf).to_numpy())
    term_crf_private = _crf(private_wacc.to_numpy(), term_years)
    rows = rows.set_index("asset_id")
    if not rows.index.equals(private.index) and set(rows.index) != set(private.index):
        raise ValueError(f"case {case_id!r} does not cover the same assets as the private case")
    rows = rows.reindex(private.index)
    case_rr = pd.to_numeric(rows["annual_required_revenue_gbp"], errors="coerce")
    case_wacc = pd.to_numeric(rows["wacc"], errors="coerce")
    saving_life = (private_rr - case_rr).fillna(0.0)
    term_crf_case = _crf(case_wacc.to_numpy(), term_years)
    life_crf_case = _crf(case_wacc.to_numpy(), remaining.fillna(0).to_numpy())
    # The case's own capex: equal to the private one unless the state bears an
    # overrun on its own plant (lever state_capex_overrun, 29 Sep 2026).
    case_capital = pd.to_numeric(rows["annual_capital_recovery_gbp"], errors="coerce")
    capex_case = pd.Series(np.where(np.isfinite(life_crf_case) & (life_crf_case > 0),
                                    case_capital.to_numpy() / np.where(life_crf_case > 0, life_crf_case, 1.0),
                                    capex_total.to_numpy()), index=private.index)
    # Tax equivalents (lever tax_equivalent) are in the case's required revenue;
    # the term-basis saving is a capital-charge change, so they come off it apart.
    tax_equivalent = _numeric(rows, "annual_tax_equivalent_gbp") - _numeric(private, "annual_tax_equivalent_gbp")
    if settings["amortisation"] == "contract_term":
        saving_term = pd.Series(
            np.nan_to_num(capex_total.to_numpy() * term_crf_private - capex_case.to_numpy() * term_crf_case),
            index=private.index) - tax_equivalent
    else:
        saving_term = saving_life.copy()
    # The case's required revenue with its capital recovered over the contract
    # term instead of the economic life, for the at-cost strike under
    # contract-term amortisation (see the docstring).
    case_rr_term = case_rr + pd.Series(
        np.nan_to_num(capex_case.to_numpy() * (term_crf_case - life_crf_case)), index=private.index)
    axis_values = {axis: _case_axis(rows, axis) for axis in AXIS_COLUMNS}
    # Retired lever (29 Sep 2026): every case re-prices by contract vintage.
    axis_values[LEVER_CFD_ELIGIBILITY] = str(settings.get("eligibility", ELIGIBILITY_CONTRACT_VINTAGE))
    if AXIS_CFD_AMORTISATION not in rows.columns:
        axis_values[AXIS_CFD_AMORTISATION] = str(settings.get("amortisation", axis_values[AXIS_CFD_AMORTISATION]))
    # The whole rent, but only the owner's share of it: a 30% stake can
    # re-price 30% of the contract, not all of it.  Under full ownership the
    # share is 1 and this is the whole rent.  A split contract (Power BECCS)
    # is costed on its electricity leg only, the leg its strike pays.
    leg_share = electricity_leg_share(private, contract_revenue, private_rr,
                                      settings.get("electricity_leg_carriers", DEFAULT_ELECTRICITY_LEG_CARRIERS))
    share = _numeric(rows, "ownership_share")
    # Item B5: the state overhead in case_rr is already the state's share of
    # its holding company (capacity x share x rate), so it is taken off after
    # the share rather than scaled by it again.  Under full ownership this
    # changes nothing; a 30% stake at cost now carries its whole centre.
    overhead = _numeric(rows, "annual_state_overhead_gbp")
    saving_at_cost_life = (share * (contract_revenue - leg_share * (case_rr - overhead))
                           - leg_share * overhead).fillna(0.0)
    saving_at_cost_term = (share * (contract_revenue - leg_share * (case_rr_term - overhead))
                           - leg_share * overhead).fillna(0.0)
    if "co_ownership_state_adjustment_gbp" in rows.columns:
        # Decision 59 (review 2, 3 Oct 2026): on a separately financed co-owned row the state's at-cost
        # saving is its share of the revenue less its OWN cost, not its share of the blended row's.
        co_cost = _numeric(rows, "co_ownership_state_adjustment_gbp").fillna(0.0)
        # Decision 68: at cost, the co-owner's lower cost of capital (the reduction sensitivity) passes
        # into its strike too, so the gain reaches consumers (under a fixed strike it stays as rent).
        if "co_investment_rate_gain_gbp" in rows.columns:
            co_cost = co_cost + _numeric(rows, "co_investment_rate_gain_gbp").fillna(0.0)
        saving_at_cost_life = saving_at_cost_life + leg_share * co_cost
        saving_at_cost_term = saving_at_cost_term + leg_share * co_cost
    if OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT in rows.columns:
        # F27 item 1 (user, 27 Oct 2026): at cost, the state's share re-prices on its own cost, which
        # carries the whole owners'-value premium (its pro rata share of the row's carried s of it).
        premium_adjustment = _numeric(rows, OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT).fillna(0.0)
        saving_at_cost_life = saving_at_cost_life + leg_share * premium_adjustment
        saving_at_cost_term = saving_at_cost_term + leg_share * premium_adjustment
    over_term = axis_values[AXIS_CFD_AMORTISATION] == "contract_term"
    saving_at_cost = saving_at_cost_term if over_term else saving_at_cost_life
    if axis_values[AXIS_CFD_STRIKE_BASIS] == STRIKE_BASIS_AT_COST and not has_settlement:
        raise ValueError(
            f"case {case_id!r} asks for cfd_strike_basis=at_cost, which needs the WP-D settlement "
            "columns on the register; run the settlement before the re-pricing"
        )
    frame = pd.DataFrame({
        "case_id": case_id,
        "asset_id": private.index,
        "component_type": private["component_type"].to_numpy(),
        "carrier": private["carrier"].to_numpy(),
        "finance_technology": private.get("finance_technology", pd.Series(index=private.index)).to_numpy(),
        "support_type": _blank(private["support_type"]).to_numpy(),
        "network_support_type": _blank(
            private["network_support_type"] if "network_support_type" in private.columns else private["support_type"]
        ).to_numpy(),
        "finance_vintage": private["finance_vintage"].to_numpy(),
        "contract_vintage": _blank(private["contract_vintage"]).to_numpy(),
        "cfd_round": _blank(private["cfd_round"]).to_numpy(),
        "cfd_technology": contract_technology(private).to_numpy(),
        "cfd_award_year": pd.to_numeric(private.get("cfd_award_year"), errors="coerce").to_numpy(),
        "evidence_status": rows["evidence_status"].to_numpy(),
        "state_owned": (rows["state_owned"].astype(bool) if "state_owned" in rows.columns
                        else pd.to_numeric(rows["ownership_share"], errors="coerce").fillna(0.0).gt(0)).to_numpy(),
        "dispatch_mwh": pd.to_numeric(private["dispatch_mwh"], errors="coerce").fillna(0.0).to_numpy(),
        "saving_economic_life_gbp": saving_life.to_numpy(),
        "saving_bill_term_gbp": saving_term.to_numpy(),
        "saving_at_cost_gbp": saving_at_cost.to_numpy(),
        "saving_at_cost_economic_life_gbp": saving_at_cost_life.to_numpy(),
        "contract_term_years": term_years,
        "electricity_leg_share": leg_share.to_numpy(),
        # F10: GB Energy's offshore stake keeps P1's contract form (a minority stake beside a private
        # majority on a 20-year CfD: the financing saving over the contract term, never at cost).
        "gbe_stake_contract": (rows["gbe_part"].astype(str).eq(GBE_PART_STAKE).to_numpy()
                               if "gbe_part" in rows.columns else np.zeros(len(rows), dtype=bool)),
        # Option E (user decision, 28 Oct 2026): GB Energy's rows, dated from the GB Energy Act 2025 (only where the
        # register records it): a contract it owns is re-priced as P2 / P1 re-price their own, whatever its vintage
        # under the group's legislation year.
        **({"gbe_owned": pd.to_numeric(rows["gbe_eligible_share"], errors="coerce").notna().to_numpy()}
           if "gbe_eligible_share" in rows.columns else {}),
        # F27 item 1 (24 Oct 2026): Net Zero Teesside's pre-2025 DPA, and the DPA payment the bill's
        # remaining_ccs line makes to it (the most its re-pricing can take off the line).
        "pre_2025_dpa": _pre_2025_dpa_rows(private, rows),
        "dpa_payment_gbp": _numeric(private, "support_revenue_gbp").to_numpy(),
        **axis_values,
    })
    return frame


def _relabelled_savings(frame: pd.DataFrame, case_id: str, settings: Mapping[str, str]) -> pd.DataFrame:
    """Another case's :func:`_case_savings` rows with this case's labels (F20 phase 2).

    The fingerprint holds everything the rows' numbers depend on, so only the labels
    differ: ``case_id`` and the lever columns, which :func:`_case_savings` reads off the
    case's own rows and which equal its settings (``register_settings_by_case``).  The
    eligibility (a constant of the run) and the amortisation (in the fingerprint) are kept.
    """
    out = frame.copy()
    out["case_id"] = case_id
    for axis in AXIS_COLUMNS:
        if axis not in (LEVER_CFD_ELIGIBILITY, AXIS_CFD_AMORTISATION):
            out[axis] = axis_setting(settings, axis)
    return out


def per_row_savings(register: pd.DataFrame | None, settings: Mapping, context: "SavingsContext | None" = None,
                    cases: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per ``(case_id, asset_id)`` with the saving versus private and its route.

    ``saving_economic_life_gbp`` is the register's own required-revenue change
    (resource saving over the asset's economic life).  ``saving_bill_term_gbp``
    is the capital-charge change annuitised over the configured term when
    ``amortisation`` is ``contract_term``, otherwise identical.  ``route`` is
    exactly one of :data:`ROUTES`.

    Item D3, the strike-basis axis.  A case whose register rows carry
    ``cfd_strike_basis = at_cost`` takes, for the rows it re-prices under R1,
    the rent on the contract instead of the financing saving: the strike becomes
    the asset's cost, so the bill gives up ``ownership share x (contract revenue
    - required revenue at the case WACC)`` rather than the change in required
    revenue alone.  The share matters under coordination, where a 30% stake can
    re-price 30% of the contract and no more; under full ownership it is the
    whole rent.  Only ``saving_bill_term_gbp`` moves;
    ``saving_economic_life_gbp`` stays the register's required-revenue change,
    so the route identity against the financing summary still closes and the
    difference is declared as the term adjustment.

    Like for like on a split contract (``electricity_leg_carriers``, Power
    BECCS).  The register's required revenue covers the whole plant, capture
    included, but the private strike covers the electricity leg only; capture
    is paid off the bill by the carbon-removal contract.  Charging the whole
    required revenue to the electricity strike at cost raised the BECCS strike
    from 153.13 to 215.67 GBP/MWh (2024 prices) on the CP30 year: the bill took
    the capture leg the private case never put on it.  Both sides are therefore
    put on the electricity leg: its share of required revenue is what the
    private strike recovers, ``share = clip(contract revenue / private required
    revenue, 0, 1)`` (the private strike taken to price the electricity leg at
    the private cost of capital, the only basis the model has for the split),
    and at cost the bill gives up ``ownership share x (contract revenue - share
    x case required revenue)``: the electricity leg's financing saving.  On the
    financing-saving basis the strike likewise takes only ``share`` of the
    financing saving (``saving_off_bill_leg_gbp`` records the capture leg's
    part, which stays in the route's register saving but off the bill).  The
    share is reported as ``electricity_leg_share`` (1 for every other carrier).

    The amortisation axis on the at-cost basis (audit 26 Sep 2026).  Until this
    audit the at-cost saving always costed the asset over its economic life, so
    ``cfd_amortisation`` did nothing wherever ``cfd_strike_basis`` was ``at_cost``
    and ``public_ownership@amortised_over_term`` reproduced the central case
    exactly (927.84).  Now, under ``contract_term``, the strike at cost recovers
    the case's capital over the contract term: required revenue has its capital
    charge re-annuitised at the case WACC over ``min(term, remaining life)``,
    ``case_rr_term = case_rr + capex x (CRF(case WACC, term) - CRF(case WACC,
    life))``, which raises the 2030 strike and so takes less to the bill;
    ``economic_life`` keeps the life-long charge.  The term is the row's round's
    (``contract_term_years_by_round``), else ``contract_term_years``; the same
    ``min(term, life)`` applies on the financing-saving basis, since no contract
    can amortise capital past the asset's life.

    30 Sep 2026: ``register`` may hold a batch of whole cases when ``context``
    (:class:`SavingsContext`) carries what the batch cannot show: the private
    rows, every case's settings and the comparators (:class:`SavingsStream`).

    F20 phase 2 (27 Oct 2026): a case's rows depend on its savings fingerprint alone
    (:func:`fingerprints.savings_key`: the register body, the comparator, the strike basis
    and the amortisation), so the context computes each fingerprint once
    (:meth:`SavingsContext.case_savings`) and gives every other case on it a copy with its
    own labels, recomputing a sample in full.  ``cases`` (in register order) may name
    cases whose rows ``register`` does not hold, when their fingerprint was computed
    before (:meth:`SavingsStream.savings_batches`); ``register`` may then be None.
    """
    if register is not None:
        missing = REQUIRED_REGISTER_COLUMNS - set(register.columns)
        if missing:
            raise ValueError(f"Finance register lacks columns: {sorted(missing)}")
    if context is None:
        # House rule 3: the baseline for a case is the private case at the same point
        # on every lever the private baseline also takes (the WACC basis, merchant
        # contracts at cost, the storage floor), read off the register's lever columns.
        settings_by_case = {case_id: case_settings(rows) for case_id, rows in register.groupby("case_id", sort=False)}
        context = SavingsContext(register.loc[register["case_id"].eq("private")], settings_by_case,
                                 lambda case_id: register.loc[register["case_id"].eq(case_id)],
                                 keyed=_fp.frame_is_keyed(register))
    groups = ({} if register is None
              else {case_id: rows for case_id, rows in register.groupby("case_id", sort=False)})
    frames = []
    for case_id in (list(groups) if cases is None else [str(c) for c in cases]):
        rows = groups.get(case_id)
        point = case_settings(rows) if rows is not None else context.settings_by_case[case_id]
        if not spec_for_row(point).market_bill():
            # A case that acquires the legacy fleet (or dispatches centrally) has
            # no contracts to re-price and no market surplus to route; the
            # acquisition tariff (item N5) is built from the register itself.
            continue
        frames.append(context.case_savings(case_id, rows, settings))
    savings = pd.concat(frames, ignore_index=True)
    # auction_clearing: nothing is re-priced; the owned CfDs keep their strike and
    # their rent follows surplus_recycling in the public account (route R2).
    auction = savings[AXIS_CFD_STRIKE_BASIS].astype(str).eq(STRIKE_BASIS_AUCTION_CLEARING)
    savings["r1_eligible"] = _r1_eligible(savings, settings) & ~auction
    # F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA, where the state owns it under the pre-2025
    # option, is re-priced like an R1 strike, on the bill's remaining_ccs line (never under
    # auction_clearing, where the state keeps every contract as signed and its rent is R2's).
    savings["dpa_repriced"] = (savings["pre_2025_dpa"].astype(bool) & savings["state_owned"].astype(bool)
                               & savings[LEVER_PRE_2025_CFD].astype(str).isin(PRE_2025_STATE_OWNED) & ~auction)
    repriced = savings["r1_eligible"] | savings["dpa_repriced"]
    savings["route"] = _routes(savings, settings)
    # The amortisation axis chooses how much of a re-priced contract's saving
    # the bill takes: the capital charge annuitised over the contract term, or
    # the whole change in required revenue the route carries. Applied before the
    # strike basis, which overrides it where it applies.
    stake_contract = savings["gbe_stake_contract"].astype(bool) if "gbe_stake_contract" in savings.columns else False
    over_life = (savings[AXIS_CFD_AMORTISATION].astype(str).eq("economic_life") & repriced
                 & ~stake_contract)
    savings["saving_bill_term_gbp"] = savings["saving_bill_term_gbp"].where(
        ~over_life, savings["saving_economic_life_gbp"]
    )
    # A split contract (Power BECCS) passes only its electricity leg's financing
    # saving to the electricity strike: the capture leg's saving belongs to the
    # off-bill carbon-removal contract, exactly as its cost does. The route still
    # carries the whole register saving (the identity with the financing summary
    # closes), and the off-bill part is reported, not billed.
    full_financing = savings["saving_bill_term_gbp"]
    savings["saving_bill_term_gbp"] = full_financing * savings["electricity_leg_share"]
    savings["saving_off_bill_leg_gbp"] = full_financing - savings["saving_bill_term_gbp"]
    # D3: under the at-cost strike basis the re-priced rows give up the whole
    # rent rather than the financing saving. The financing amount is kept so
    # limit_at_cost_to_deliverable can put a contract back on it.
    at_cost = (savings[AXIS_CFD_STRIKE_BASIS].astype(str).eq(STRIKE_BASIS_AT_COST) & repriced
               & ~stake_contract)
    savings["saving_financing_term_gbp"] = savings["saving_bill_term_gbp"]
    savings["saving_bill_term_gbp"] = savings["saving_bill_term_gbp"].where(
        ~at_cost, savings["saving_at_cost_gbp"]
    )
    savings["at_cost_applied"] = at_cost
    # F27 item 1: the DPA's re-pricing can take its payments to zero, never below (the plant does
    # not pay consumers; the rest stays with its owner, as a CfD cut stops at a zero strike).
    dpa = savings["route"].eq(ROUTE_DPA_BILL)
    if dpa.any():
        capped = np.minimum(savings["saving_bill_term_gbp"], savings["dpa_payment_gbp"])
        if (dpa & (capped < savings["saving_bill_term_gbp"])).any():
            _LOG.warning("the DPA re-pricing exceeds the DPA payment in %d case(s); capped at the payment",
                         int((dpa & (capped < savings["saving_bill_term_gbp"])).sum()))
        savings["saving_bill_term_gbp"] = savings["saving_bill_term_gbp"].where(~dpa, capped)
    amortisation = savings[AXIS_CFD_AMORTISATION].astype(str)
    term_label = _term_label(int(settings["contract_term_years"]), settings.get("contract_term_years_by_round") or {})
    term = np.where(amortisation.eq("contract_term"), f"contract_term_{term_label}", amortisation)
    boundary_value = savings[LEVER_PRE_2025_CFD].astype(str)
    boundary = boundary_value.map(LEVERS[LEVER_PRE_2025_CFD].label)
    eligibility = np.where(boundary_value.isin(PRE_2025_STATE_OWNED), "commissioning_vintage",
                           savings[LEVER_CFD_ELIGIBILITY].astype(str))
    basis = savings[AXIS_CFD_STRIKE_BASIS].astype(str)
    savings["method"] = (boundary + ";" + eligibility + ";" + term + ";strike_" + basis)
    return savings


#: The comparator columns :func:`per_row_savings` reads (the rest of a comparator's rows is never used).
COMPARATOR_COLUMNS = frozenset({
    "asset_id", "annual_required_revenue_gbp", "annual_capital_recovery_gbp", "wacc", "remaining_life_years",
    "cfd_round", "annual_tax_equivalent_gbp", "component_type", "carrier", "finance_technology", "support_type",
    "network_support_type", "finance_vintage", "contract_vintage", "cfd_award_year", "dispatch_mwh",
    # FES 2050 decision 54: a comparator on another private rate path settles its contracts at
    # that path's strikes, so its contract revenue is its own (equal to the private case's otherwise).
    "market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp", "cfd_strike_offset_gbp_per_mwh",
})


class SavingsContext:
    """What every case's savings share: the private rows, the case settings and the comparators (30 Sep 2026).

    ``private_rows`` are the central private case's register rows;
    ``settings_by_case`` every case's lever settings, in register order (the
    comparator is the first private-family case that matches, house rule 3);
    ``rows_for(case_id)`` returns a case's register rows (kept as ``rows_for``),
    called once per comparator.  :func:`per_row_savings` builds one from a whole register; the
    streaming stages build one from the register file and hand it to
    :func:`per_row_savings` with one batch of cases at a time.
    """

    def __init__(self, private_rows: pd.DataFrame, settings_by_case: Mapping[str, Mapping[str, str]], rows_for, *,
                 share: bool | None = None, sample_every: int | None = None, keyed: bool = True):
        central_private = private_rows.set_index("asset_id")
        if central_private.index.has_duplicates:
            raise ValueError("The private register repeats an asset_id")
        self.central_private = central_private
        self.settings_by_case = settings_by_case
        self.rows_for = rows_for
        self._comparators: dict[str, pd.DataFrame] = {}
        # Contract revenue is a physical settlement of the run, so it is read once
        # from the private rows and shared by every case (house rule 2).
        self.contract_revenue = (
            _numeric(central_private, "market_revenue_gbp") + _numeric(central_private, "support_revenue_gbp")
            - _numeric(central_private, "fuel_carbon_gbp")
        )
        self.has_settlement = {"market_revenue_gbp", "support_revenue_gbp"}.issubset(central_private.columns)
        # F20 phase 2: one computation per savings fingerprint (fingerprints.savings_key), held while
        # its register key's cases stream past (they are neighbours in register order).
        from scripts.public_ownership.model_year import is_vintaged_model_year

        # A register without its lever columns (a hand-built test frame) cannot be keyed: nothing is shared.
        self.share = (_fp.sharing_enabled() if share is None else bool(share)) and bool(keyed)
        years = (pd.to_numeric(private_rows["model_year"], errors="coerce").dropna().unique()
                 if "model_year" in private_rows.columns else [])
        self.vintaged = any(is_vintaged_model_year(int(y)) for y in years)
        self.sample = _fp.SampleCheck("cfd_repricing.per_row_savings", sample_every)
        self._comparator_of: dict[str, str] = {}
        self._memo: dict[tuple, pd.DataFrame] = {}
        self._run = None
        self.computed = 0

    def contract_revenue_for(self, case_id: str) -> pd.Series:
        """The contract revenue a case's at-cost saving is measured on: its comparator's (decision 54).

        A comparator's revenue equals the private case's unless the comparator settles its
        contracts at other strikes (a private rate path's own calibration), so every case whose
        comparator is ``private``, and every 2030 case, reads the shared private revenue.
        """
        comparator = self.comparator(case_id)
        if comparator == "private" or not self.has_settlement:
            return self.contract_revenue
        frame = self.private_for(case_id)
        if not {"market_revenue_gbp", "support_revenue_gbp"}.issubset(frame.columns):
            return self.contract_revenue
        # Review 2 (3 Oct 2026): only a comparator on a private rate path (its rows carry the path's strike
        # offset) settles at other strikes; any other comparator (every 2030 one) reads the shared revenue.
        offset = (_numeric(frame, "cfd_strike_offset_gbp_per_mwh") if "cfd_strike_offset_gbp_per_mwh" in frame.columns
                  else None)
        if offset is None or not offset.fillna(0.0).ne(0.0).any():
            return self.contract_revenue
        return (_numeric(frame, "market_revenue_gbp") + _numeric(frame, "support_revenue_gbp")
                - _numeric(frame, "fuel_carbon_gbp"))

    def comparator(self, case_id: str) -> str:
        """The case's comparator (house rule 3), looked up once."""
        if case_id not in self._comparator_of:
            self._comparator_of[case_id] = comparator_for_settings(case_id, self.settings_by_case)
        return self._comparator_of[case_id]

    def savings_key(self, case_id: str) -> tuple | None:
        """The case's savings fingerprint (:func:`fingerprints.savings_key`); None when nothing is shared."""
        if not self.share or case_id not in self.settings_by_case:
            return None
        return _fp.savings_key(case_id, self.settings_by_case[case_id], self.comparator(case_id),
                               vintaged=self.vintaged)

    @staticmethod
    def _run_of(key: tuple) -> tuple:
        """The register key a fingerprint sits on: its cases are one run of the register."""
        return key[0][:2]

    def computed_cases(self, cases: Sequence[str]) -> set[str]:
        """Of ``cases`` (market cases in register order), those :meth:`case_savings` will compute.

        Every case when nothing is shared; otherwise the first case of each fingerprint in each
        run of the register (the memo is cleared when the run changes, exactly as here).
        """
        out, seen, run = set(), set(), None
        for case_id in cases:
            key = self.savings_key(case_id)
            if key is None:
                out.add(case_id)
                continue
            if self._run_of(key) != run:
                seen, run = set(), self._run_of(key)
            if key not in seen:
                seen.add(key)
                out.add(case_id)
        return out

    def case_savings(self, case_id: str, rows: pd.DataFrame | None, settings: Mapping) -> pd.DataFrame:
        """One market case's :func:`_case_savings`, computed once per fingerprint (F20 phase 2).

        A case whose fingerprint was computed for an earlier case of its run takes that
        case's rows relabelled (:func:`_relabelled_savings`); :attr:`sample` picks some of
        those to compute in full as well (their rows read when not given) and fails on any
        difference.  ``rows`` may be None only for a case that takes a shared result.
        """
        key = self.savings_key(case_id)
        if key is not None:
            if self._run_of(key) != self._run:
                self._memo.clear()
                self._run = self._run_of(key)
            shared = self._memo.get(key)
            if shared is not None:
                frame = _relabelled_savings(shared, case_id, self.settings_by_case[case_id])
                if self.sample.due():
                    from scripts.public_ownership.register_io import frames_identical

                    own = rows if rows is not None else self.rows_for(case_id)
                    full = _case_savings(case_id, own, settings, self)
                    self.sample.check(case_id, frames_identical(full, frame) and full.index.equals(frame.index))
                return frame
        if rows is None:
            raise ValueError(f"{case_id}: its savings are not shared with an earlier case of its register key and "
                             "its rows were not read")
        frame = _case_savings(case_id, rows, settings, self)
        self.computed += 1
        if key is not None:
            self._memo[key] = frame
        return frame

    def private_for(self, case_id: str) -> pd.DataFrame:
        """The case's comparator rows, indexed and ordered like the private case."""
        comparator = self.comparator(case_id)
        if comparator not in self._comparators:
            frame = self.rows_for(comparator)
            # Only the columns a case's savings read off its comparator (every comparator is held).
            frame = frame[[c for c in frame.columns if c in COMPARATOR_COLUMNS]].set_index("asset_id")
            if frame.empty:
                frame = self.central_private
            self._comparators[comparator] = frame.reindex(self.central_private.index)
        return self._comparators[comparator]


#: Cases per batch when a stage streams the register (30 Sep 2026): about 110,000
#: rows, a few hundred MB, whatever the number of cases.
DEFAULT_BATCH_CASES = 8


def register_settings_by_case(register_path, cases: Sequence[str] | None = None) -> dict[str, dict[str, str]]:
    """Every case's lever settings in register order, as :func:`levers.case_settings` reads them off its rows.

    From the case-settings table beside a Parquet register (a missing value is
    the lever's baseline, as ``case_settings`` gives it); from the lever
    columns of an old CSV register otherwise.  ``cases`` keeps those cases.
    """
    from scripts.public_ownership.register_io import read_register, read_settings, settings_path

    wanted = None if cases is None else {str(c) for c in cases}
    if str(register_path).endswith(".csv") or not settings_path(register_path).exists():
        frame = read_register(register_path, columns=["case_id", *AXIS_COLUMNS])
        return {case_id: case_settings(rows) for case_id, rows in frame.groupby("case_id", sort=False)
                if wanted is None or case_id in wanted}
    table = read_settings(register_path)
    out = {}
    for record in table.to_dict("records"):
        case_id = str(record.pop("case_id"))
        if wanted is not None and case_id not in wanted:
            continue
        out[case_id] = {name: (str(record[name]) if name in record and not pd.isna(record[name])
                               else DEFAULT_SETTINGS[name]) for name in LEVERS}
        # Task E2E: a year-gated lever (2050) the case sets off its default is part of its
        # settings, so an all_cases one (private_rate_path) finds its own private comparator.
        # No 2030 settings table has a gated column, so every 2030 dict is unchanged.
        for name in GATED_LEVERS:
            if name in record and not pd.isna(record[name]) and str(record[name]) != lever_default(name):
                out[case_id][name] = str(record[name])
    return out


class SavingsStream:
    """:func:`per_row_savings` over a register file, a batch of whole cases at a time (30 Sep 2026).

    Iterating yields ``(register rows, savings rows)`` per batch of ``cases``
    (default all), in register order, each batch's savings equal to that
    batch's rows of the whole-register :func:`per_row_savings`.  The private
    case's savings are computed first (``private_savings``): the strike
    entries, their denominators and the settlement the stages share are read
    off them.  Memory is one batch plus the comparators, flat in the number of
    cases.
    """

    def __init__(self, register_path, settings: Mapping, cases: Sequence[str] | None = None,
                 batch_cases: int = DEFAULT_BATCH_CASES):
        from scripts.public_ownership.register_io import read_register

        self.path, self.settings, self.batch_cases = register_path, settings, int(batch_cases)
        self.cases = None if cases is None else [str(c) for c in cases]
        settings_by_case = register_settings_by_case(register_path, self.cases)
        self.private_rows = read_register(register_path, cases=["private"])
        missing = REQUIRED_REGISTER_COLUMNS - set(self.private_rows.columns)
        if missing:
            raise ValueError(f"Finance register lacks columns: {sorted(missing)}")
        self.context = SavingsContext(
            self.private_rows, settings_by_case,
            lambda case_id: self.private_rows if case_id == "private" else read_register(register_path, cases=[case_id]),
            keyed=_fp.register_is_keyed(register_path))
        self.private_savings = per_row_savings(self.private_rows, settings, context=self.context)
        #: Cases whose rows :meth:`savings_batches` read (the rest took a shared result).
        self.rows_read = 0

    def __iter__(self):
        from scripts.public_ownership.register_io import iter_register_cases

        for rows in iter_register_cases(self.path, cases=self.cases, batch_cases=self.batch_cases):
            batch = rows["case_id"].drop_duplicates()
            if any(spec_for_row(self.context.settings_by_case.get(c, {})).market_bill() for c in batch):
                yield rows, per_row_savings(rows, self.settings, context=self.context)

    def savings_batches(self):
        """Each batch's savings as :meth:`__iter__` yields them, reading only the rows it computes (F20 phase 2).

        For a stage that needs the savings and not the rows (the strike adjustments): the
        market cases in register order, ``batch_cases`` at a time, each batch's savings
        equal to the whole-register :func:`per_row_savings` rows of those cases; only the
        first case of each fingerprint (and the sample :attr:`SavingsContext.sample`
        recomputes) is read.  :attr:`rows_read` counts the cases read.
        """
        from scripts.public_ownership.register_io import RegisterReader, register_cases

        settings_by_case = self.context.settings_by_case
        order = [c for c in (self.cases if self.cases is not None else register_cases(self.path))
                 if c in settings_by_case and spec_for_row(settings_by_case[c]).market_bill()]
        computed = self.context.computed_cases(order)
        reader = RegisterReader(self.path, pinned=())
        for start in range(0, len(order), self.batch_cases):
            batch = order[start:start + self.batch_cases]
            needed = [c for c in batch if c in computed]
            self.rows_read += len(needed)
            rows = reader.read(needed) if needed else None
            yield per_row_savings(rows, self.settings, context=self.context, cases=batch)


def electricity_leg_share(private: pd.DataFrame, contract_revenue: pd.Series, private_rr: pd.Series,
                          carriers) -> pd.Series:
    """Share of required revenue the electricity CfD pays, per asset: 1 except on split contracts.

    For a carrier in ``carriers`` it is the private contract revenue (market
    plus support less fuel and carbon) over the private required revenue,
    clipped to [0, 1]: the private strike is taken to recover exactly the
    electricity leg at the private cost of capital, the rest of required
    revenue being the off-bill leg.  An asset with no private required revenue
    keeps 1.
    """
    share = pd.Series(1.0, index=private.index)
    split = private["carrier"].astype(str).isin([str(c) for c in carriers or ()])
    rr = pd.to_numeric(private_rr, errors="coerce")
    usable = split & rr.gt(0)
    share[usable] = (contract_revenue[usable] / rr[usable]).clip(lower=0.0, upper=1.0)
    return share.fillna(1.0)


def _pre_2025_dpa_rows(private: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """F27 item 1: the rows on a pre-2025 DPA (``levers.is_pre_2025_dpa``), aligned to ``private``."""
    support = _blank(private["support_type"]).str.upper()
    rounds = _blank(private["cfd_round"])
    candidate = (support.eq("DPA") & rounds.isin(list(PRE_2025_DPA_CONTRACTS))).to_numpy()
    out = np.zeros(len(private), dtype=bool)
    if candidate.any():
        years = rows["model_year"] if "model_year" in rows.columns else pd.Series(None, index=private.index)
        for i in np.flatnonzero(candidate):
            out[i] = is_pre_2025_dpa(support.iloc[i], rounds.iloc[i], years.iloc[i])
    return out


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _case_axis(rows: pd.DataFrame, axis: str) -> str:
    """The axis value this case sits at, read from its register rows.

    Every row of a case carries the same settings, so the first non-null wins;
    a register written before the axis existed has no column and falls back to
    the axis central value.
    """
    if axis not in rows.columns:
        return axis_setting({}, axis)
    values = rows[axis].dropna().astype(str).unique()
    if len(values) > 1:
        raise ValueError(f"a case carries more than one {axis}: {sorted(values)}")
    return axis_setting({axis: values[0]} if len(values) else {}, axis)


def _r1_eligible(savings: pd.DataFrame, settings: Mapping) -> pd.Series:
    # A contract can only be re-priced in the bill if the network settles it
    # as a CfD: a label corrected in the register alone (tidal) has no CfD
    # payment in the bill, so its saving goes to the public account instead.
    is_cfd = savings["support_type"].str.lower().eq("cfd") & savings["network_support_type"].str.lower().eq("cfd")
    # Under pre_2025_cfd_capacity: public_repriced every post-2025 contract the
    # state owns is re-priced (commissioning vintage); otherwise the case's
    # eligibility lever applies.
    reprice_all = savings[LEVER_PRE_2025_CFD].astype(str).isin(PRE_2025_STATE_OWNED)
    by_commissioning = is_cfd & savings["finance_vintage"].eq("new_build")
    by_contract = is_cfd & savings["contract_vintage"].eq("post_policy")
    by_contract_rule = savings[LEVER_CFD_ELIGIBILITY].astype(str).eq("contract_vintage")
    central = (by_contract & by_contract_rule) | (by_commissioning & ~by_contract_rule)
    eligible = (reprice_all & by_commissioning) | (~reprice_all & central)
    if "gbe_owned" in savings.columns:
        # Option E: a CfD GB Energy owns from its own year (the GB Energy Act 2025) is re-priced like any state CfD.
        eligible = eligible | (is_cfd & savings["gbe_owned"].astype(bool))
    return eligible


def _routes(savings: pd.DataFrame, settings: Mapping) -> pd.Series:
    is_cfd = savings["support_type"].str.lower().eq("cfd")
    classified = savings["carrier"].astype(str).map(settings["route_classification"])
    route = pd.Series(ROUTE_MERCHANT, index=savings.index, dtype=object)
    for label, classified_route in ROUTE_CLASSIFICATIONS.items():
        route[classified.eq(label)] = classified_route
    route[savings["component_type"].eq("StorageUnit")] = ROUTE_STORAGE
    route[is_cfd] = ROUTE_CFD_PUBLIC_ACCOUNT
    route[savings["r1_eligible"]] = ROUTE_CFD_BILL
    if "pre_2025_dpa" in savings.columns:
        # F27 item 1 (24 Oct 2026): Net Zero Teesside is never merchant. On its DPA (private, or a case
        # that keeps it so) it is the dispatchable-contract route, which carries nothing at no state
        # share; owned and kept on the signed DPA (auction_clearing) its rent is R2's, as an owned CfD's.
        dpa = savings["pre_2025_dpa"].astype(bool)
        owned = savings["state_owned"].astype(bool) if "state_owned" in savings.columns else False
        route[dpa] = ROUTE_DISPATCHABLE_CFD
        route[dpa & owned] = ROUTE_CFD_PUBLIC_ACCOUNT
        route[savings["dpa_repriced"].astype(bool)] = ROUTE_DPA_BILL
    route[~savings["evidence_status"].astype(str).eq("resolved")] = ROUTE_UNRESOLVED
    return route


def _priced_rows(savings: pd.DataFrame, table: StrikeTable) -> pd.DataFrame:
    """CfD rows tagged with the strike-table entry they settle at."""
    cfd = savings.loc[savings["support_type"].str.lower().eq("cfd")].copy()
    if cfd.empty:
        return cfd.assign(strike_round="", technology="", strike_key="", is_fallback_strike=False)
    # The contract's own strike technology where the register carries it (the
    # strike the bill and the bids use); otherwise the carrier's.
    cfd["_contract_technology"] = contract_technology(cfd).to_numpy()
    resolved = {}
    for carrier, cfd_round, technology in cfd[["carrier", "cfd_round", "_contract_technology"]]             .drop_duplicates().itertuples(index=False):
        resolved[(carrier, cfd_round, technology)] = table.resolve(carrier, cfd_round, technology)
    keys = list(zip(cfd["carrier"], cfd["cfd_round"], cfd["_contract_technology"]))
    cfd["_entry"] = [resolved[k][0] for k in keys]
    cfd["is_fallback_strike"] = [resolved[k][1] for k in keys]
    cfd = cfd.drop(columns="_contract_technology")
    priced = cfd.loc[cfd["_entry"].notna()].copy()
    priced["strike_round"] = [entry[0] for entry in priced["_entry"]]
    priced["technology"] = [entry[1] for entry in priced["_entry"]]
    priced["strike_key"] = priced["strike_round"] + "_" + priced["technology"]
    return priced


def _settled_energy_by_key(priced: pd.DataFrame,
                           paid_curtailment_mwh_by_key: Mapping[str, float] | None,
                           paid_energy_mwh_by_key: Mapping[str, float] | None = None) -> pd.Series:
    """Energy the bill pays a difference on at each strike entry: the strike cut's denominator.

    A strike cut of ``delta`` moves the bill's CfD line by ``delta`` times the
    energy that line actually pays at that strike, so that is the only
    denominator under which ``delta = saving / energy`` hands the bill exactly
    the saving. ``paid_energy_mwh_by_key`` is that energy measured on the bill's
    own settlement (:func:`scripts.public_ownership.build_cfd_strike_adjustments.bill_paid_energy_by_key`):
    the balancing dispatch in the hours a difference payment accrues, after the
    support start, the negative pricing rule (``gb_contract_terms``: AR4 and
    later unpaid below zero, AR2-AR3 inside six-hour negative runs) and the
    volume collars, plus the pre-AR4 curtailed energy the economic-curtailment
    rule pays on the same terms. When it is given it is used alone.

    Without it the denominator falls back to the register's dispatch plus the
    forecast summary's curtailed energy, which is the paid energy only when the
    bill applies no negative pricing rule and no volume collar binds on a
    re-priced round (the writer refuses the fallback under any negative pricing
    rule; the one collar, Drax's bridging CfD, is never re-priced). Either way the denominator counts each asset
    once whatever the case: it is a physical quantity of the run, not of the
    ownership case.
    """
    private_rows = priced.loc[priced["case_id"].eq("private")]
    if paid_energy_mwh_by_key is not None:
        keys = pd.Index(private_rows["strike_key"].drop_duplicates())
        return pd.Series({key: float(paid_energy_mwh_by_key.get(key, 0.0)) for key in keys}, dtype=float)
    dispatch_by_key = private_rows.groupby("strike_key")["dispatch_mwh"].sum()
    for key, curtailed in (paid_curtailment_mwh_by_key or {}).items():
        if key in dispatch_by_key.index:
            dispatch_by_key[key] += float(curtailed)
    return dispatch_by_key


def limit_at_cost_to_deliverable(
    savings: pd.DataFrame, table: StrikeTable,
    paid_curtailment_mwh_by_key: Mapping[str, float] | None = None,
    paid_energy_mwh_by_key: Mapping[str, float] | None = None,
    settled_energy: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put a contract back on the financing-saving basis when at-cost cannot be delivered.

    Handing a contract's whole rent to the bill means cutting its strike by
    ``rent / energy the contract settles``. A contract that is mostly curtailed
    settles little energy, so that cut can exceed the strike itself and imply a
    negative strike - a generator paying consumers on top of surrendering its
    market revenue, which is outside what a difference contract can express.

    Rather than invent a floor and strand the remainder, such a contract keeps
    the ordinary treatment: the bill receives the financing saving and the rent
    stays with the public owner, exactly as in the cases that do not move the
    strike-basis axis. Decided 19 Sep 2026 on the CP30 2030 year, where it
    affects AR3 remote island wind (85% curtailed) in the bounding-pair case
    alone: GBP 18.4 m of rent over 179 GWh is GBP 102.67/MWh against a
    GBP 55.74/MWh strike.

    The energy is the one the bill pays at the strike (see
    :func:`_settled_energy_by_key`), the same denominator the strike cut uses.

    Returns ``(savings, limited)``; ``limited`` is one row per reverted contract
    group for the audit, empty when every group can deliver its rent.
    ``settled_energy`` (:func:`settled_energy_by_key` of the private rows) lets
    ``savings`` be a batch of cases without the private case (30 Sep 2026).
    """
    savings = savings.copy()
    if "at_cost_applied" not in savings.columns or not savings["at_cost_applied"].any():
        return savings, pd.DataFrame(columns=_LIMIT_COLUMNS)
    priced = _priced_rows(savings, table)
    if priced.empty:
        return savings, pd.DataFrame(columns=_LIMIT_COLUMNS)
    energy = (settled_energy if settled_energy is not None
              else _settled_energy_by_key(priced, paid_curtailment_mwh_by_key, paid_energy_mwh_by_key))
    r1 = priced.loc[priced["route"].eq(ROUTE_CFD_BILL) & priced["at_cost_applied"]]
    rows = []
    for (case_id, strike_key), group in r1.groupby(["case_id", "strike_key"], sort=False):
        settled = float(energy.get(strike_key, 0.0))
        if settled <= 0.0:
            continue
        entry = (group["strike_round"].iloc[0], group["technology"].iloc[0])
        strike = float(table.strikes[entry])
        at_cost = float(group["saving_bill_term_gbp"].sum())
        if at_cost / settled <= strike:
            continue
        financing = float(group["saving_financing_term_gbp"].sum())
        index = group.index
        savings.loc[index, "saving_bill_term_gbp"] = savings.loc[index, "saving_financing_term_gbp"]
        savings.loc[index, "at_cost_applied"] = False
        reason = "at_cost cut exceeds the strike; kept on the financing-saving basis"
        if financing / settled > strike:
            # Review 26 Sep 2026: under the gb_contract_terms rule the paid energy
            # shrinks, so even the financing cut can exceed the strike (AR3 remote
            # island wind, P1 all-repriced). The cut stops at a zero strike and the
            # rest stays with the public owner, whose settlement at the case strike
            # keeps it in the public account, as for the financing-basis fallback.
            scale = strike * settled / financing if financing > 0.0 else 0.0
            savings.loc[index, "saving_bill_term_gbp"] = savings.loc[index, "saving_financing_term_gbp"] * scale
            reason = ("even the financing cut exceeds the strike; capped at the strike (a zero strike), "
                      "the rest stays with the public owner")
            _LOG.warning(
                f"{case_id}: contract {strike_key} cannot deliver its financing saving "
                f"(GBP {financing / settled:,.2f}/MWh over {settled:,.0f} MWh against a "
                f"GBP {strike:,.2f}/MWh strike); cut capped at the strike")
        rows.append({
            "case_id": case_id, "strike_key": strike_key, "assets": int(len(index)),
            "settled_mwh": settled, "strike_gbp_per_mwh_2024": strike,
            "at_cost_saving_gbp_2024": at_cost,
            "at_cost_gbp_per_mwh_2024": at_cost / settled,
            "financing_saving_gbp_2024": financing,
            "reason": reason,
        })
    return savings, pd.DataFrame(rows, columns=_LIMIT_COLUMNS)


_LIMIT_COLUMNS = [
    "case_id", "strike_key", "assets", "settled_mwh", "strike_gbp_per_mwh_2024",
    "at_cost_saving_gbp_2024", "at_cost_gbp_per_mwh_2024", "financing_saving_gbp_2024", "reason",
]


def settled_energy_by_key(savings: pd.DataFrame, table: StrikeTable,
                          paid_curtailment_mwh_by_key: Mapping[str, float] | None = None,
                          paid_energy_mwh_by_key: Mapping[str, float] | None = None) -> pd.Series:
    """The strike cut's denominator per strike entry, from savings rows that include the private case."""
    return _settled_energy_by_key(_priced_rows(savings, table), paid_curtailment_mwh_by_key, paid_energy_mwh_by_key)


class StrikeAdjustments:
    """:func:`build_strike_adjustments` a batch of cases at a time (30 Sep 2026).

    Built from the private case's savings rows: the strike entries, their
    denominators and the generator counts are physical quantities of the run,
    the same for every case, and the whole-register function reads them off
    the private rows too.  :meth:`add` takes each batch's savings in register
    order; :meth:`frame` returns what :func:`build_strike_adjustments` returns
    for all of them together.
    """

    def __init__(self, private_savings: pd.DataFrame, table: StrikeTable, settings: Mapping,
                 paid_curtailment_mwh_by_key: Mapping[str, float] | None = None,
                 paid_energy_mwh_by_key: Mapping[str, float] | None = None,
                 case_offsets: Mapping[str, Mapping[str, float]] | None = None):
        self.table, self.settings = table, settings
        # FES 2050 decision 54: ``{case_id: {strike_key: GBP/MWh}}``, the strike cut every contract of an
        # entry takes in a case on another private rate path (its own calibrated strikes). The output
        # gains ``strike_basis_offset_gbp_per_mwh_2024`` only when given (no 2030 call passes it).
        self.case_offsets = case_offsets
        self.empty = private_savings.loc[private_savings["support_type"].str.lower().eq("cfd")].empty
        priced = _priced_rows(private_savings, table)
        self.dispatch_by_key = _settled_energy_by_key(priced, paid_curtailment_mwh_by_key, paid_energy_mwh_by_key)
        self.assets_by_key = priced.loc[priced["case_id"].eq("private")].groupby("strike_key")["asset_id"].size()
        # Every case covers the private assets with the private contract fields
        # (carrier, round, award year, vintage), so an entry's first row is the
        # same whichever case it is read from.
        self.detail = (
            priced.groupby("strike_key")
            .agg(carrier=("carrier", "first"), cfd_round=("strike_round", "first"), technology=("technology", "first"),
                 cfd_award_year=("cfd_award_year", "first"), contract_vintage=("contract_vintage", "first"),
                 is_fallback_strike=("is_fallback_strike", "first"))
        )
        self._eligible_any = pd.Series(False, index=self.detail.index)
        self._rows: list[dict] = []

    def add(self, savings: pd.DataFrame) -> None:
        """Add a batch of cases' savings rows (whole cases, in register order)."""
        if self.empty or savings.empty:
            return
        priced = _priced_rows(savings, self.table)
        r1 = priced.loc[priced["route"].eq(ROUTE_CFD_BILL)]
        numerator_term = r1.groupby(["case_id", "strike_key"])["saving_bill_term_gbp"].sum()
        numerator_life = r1.groupby(["case_id", "strike_key"])["saving_economic_life_gbp"].sum()
        eligible_assets = r1.groupby(["case_id", "strike_key"])["asset_id"].size()
        # Eligibility can differ by variant, so it is read per case rather than per entry.
        eligible_by_case = priced.groupby(["case_id", "strike_key"])["r1_eligible"].any()
        batch_any = priced.groupby("strike_key")["r1_eligible"].any()
        self._eligible_any = self._eligible_any | batch_any.reindex(self._eligible_any.index, fill_value=False)
        # F20 phase 1: the per-case lookups read off dicts made once per batch, and the entries
        # walked once (each case's rows are the values the per-row lookups returned).
        if "method" in savings:
            first = savings.drop_duplicates("case_id")
            method_by_case = dict(zip(first["case_id"], first["method"]))
        term_by_key, life_by_key = numerator_term.to_dict(), numerator_life.to_dict()
        eligible_assets, eligible_by_case = eligible_assets.to_dict(), eligible_by_case.to_dict()
        entries = list(self.detail.iterrows())
        for case_id in savings["case_id"].drop_duplicates():
            method = method_by_case[case_id] if "method" in savings else self.settings["method"]
            for strike_key, info in entries:
                entry = (info["cfd_round"], info["technology"])
                strike_private = float(self.table.strikes[entry])
                dispatch = float(self.dispatch_by_key.get(strike_key, 0.0))
                saving_term = float(term_by_key.get((case_id, strike_key), 0.0))
                saving_life = float(life_by_key.get((case_id, strike_key), 0.0))
                delta = saving_term / dispatch if dispatch > 0 else 0.0
                offset = float((self.case_offsets or {}).get(case_id, {}).get(strike_key, 0.0))
                delta += offset
                self._rows.append({
                    "case_id": case_id,
                    "carrier": info["carrier"],
                    "technology": info["technology"],
                    "cfd_round": info["cfd_round"],
                    "strike_key": strike_key,
                    "cfd_award_year": info["cfd_award_year"],
                    "contract_vintage": info["contract_vintage"],
                    # None: no row of the case settles at the entry; the entry-wide flag applies (frame()).
                    "eligible": (bool(eligible_by_case[(case_id, strike_key)])
                                 if (case_id, strike_key) in eligible_by_case else None),
                    "is_fallback_strike": bool(info["is_fallback_strike"]),
                    "generators": int(self.assets_by_key.get(strike_key, 0)),
                    "eligible_generators": int(eligible_assets.get((case_id, strike_key), 0)),
                    "strike_private_gbp_per_mwh_2024": strike_private,
                    "delta_gbp_per_mwh_2024": delta,
                    "strike_case_gbp_per_mwh_2024": strike_private - delta,
                    "eligible_dispatch_mwh": dispatch,
                    "r1_saving_gbp_2024": saving_term,
                    "register_saving_gbp_2024": saving_life,
                    "method": method,
                    **({"strike_basis_offset_gbp_per_mwh_2024": offset} if self.case_offsets is not None else {}),
                })

    def frame(self) -> pd.DataFrame:
        """Every case added so far, sorted as :func:`build_strike_adjustments` sorts them."""
        if self.empty:
            return pd.DataFrame(columns=ADJUSTMENT_COLUMNS)
        for row in self._rows:
            if row["eligible"] is None:
                row["eligible"] = bool(self._eligible_any.get(row["strike_key"], False))
        columns = ADJUSTMENT_COLUMNS + (["strike_basis_offset_gbp_per_mwh_2024"] if self.case_offsets is not None else [])
        frame = pd.DataFrame(self._rows, columns=columns)
        return frame.sort_values(["case_id", "r1_saving_gbp_2024"], ascending=[True, False]).reset_index(drop=True)


def build_strike_adjustments(
    savings: pd.DataFrame, table: StrikeTable, settings: Mapping,
    paid_curtailment_mwh_by_key: Mapping[str, float] | None = None,
    paid_energy_mwh_by_key: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Per-case strike changes for every table entry a CfD contract settles at.

    ``delta = eligible saving settling at that entry / all energy the bill pays
    at that strike`` so that applying the delta to every generator of the entry
    in the bill reproduces the eligible saving exactly (auction displacement:
    a cheaper public bidder lowers the clearing strike for every winner).  The
    energy paid is ``paid_energy_mwh_by_key``, measured on the bill's own
    settlement: balancing dispatch in the hours a difference accrues under the
    bill's negative pricing rule and volume collars, plus the pre-AR4 curtailed
    energy the economic-curtailment rule pays.  Without it, the fallback is the
    register dispatch plus ``paid_curtailment_mwh_by_key`` (the forecast bill's
    ``cfd_economic_curtailment_by_strike`` volumes), exact only when no hour is
    left unpaid.  ``eligible_dispatch_mwh`` in the output is that paid energy.  Entries with no
    eligible saving carry a zero delta and are listed so the coverage is
    visible.  Contracts that match no table entry cannot be re-priced in the
    bill; they are routed to the public account by :func:`per_row_savings` and
    reported by :func:`route_summary`.  The streaming stages use
    :class:`StrikeAdjustments` directly; this is it over one frame.
    """
    builder = StrikeAdjustments(savings, table, settings, paid_curtailment_mwh_by_key, paid_energy_mwh_by_key)
    builder.add(savings)
    return builder.frame()


ADJUSTMENT_COLUMNS = [
    "case_id", "carrier", "technology", "cfd_round", "strike_key", "cfd_award_year", "contract_vintage",
    "eligible", "is_fallback_strike", "generators", "eligible_generators",
    "strike_private_gbp_per_mwh_2024", "delta_gbp_per_mwh_2024", "strike_case_gbp_per_mwh_2024",
    "eligible_dispatch_mwh", "r1_saving_gbp_2024", "register_saving_gbp_2024", "method",
]


def case_strike_table(table: StrikeTable, adjustments: pd.DataFrame, case_id: str) -> StrikeTable:
    """The strike table one ownership case settles against."""
    rows = adjustments.loc[adjustments["case_id"].eq(case_id)]
    deltas = {
        (str(r.cfd_round), str(r.technology)): float(r.delta_gbp_per_mwh_2024)
        for r in rows.itertuples(index=False) if float(r.delta_gbp_per_mwh_2024) != 0.0
    }
    return table.adjusted(deltas, label=case_id)


def route_summary(savings: pd.DataFrame, financing_summary: pd.DataFrame) -> pd.DataFrame:
    """Saving by case and route, with the identity against the financing summary.

    Routes carry the economic-life saving, which sums to the register total by
    construction; the R1 row also carries the bill-term amount actually taken
    to the bill, and the difference is declared as ``term_adjustment_gbp``.

    Review 28 Sep 2026: under the at-cost strike basis part of that difference is not
    a term effect at all but the producer rent the re-priced strike hands to the
    bill, a transfer from the contract's owner (house rule 9). It is split out as
    ``at_cost_rent_transfer_gbp`` (bill term less financing term on at-cost rows),
    leaving ``term_adjustment_excluding_rent_gbp``; the two sum to the old column.

    The identity here partitions register rows into routes, so it checks that no row
    is lost or doubled, not that the money is right (``identity_kind``).
    """
    totals = financing_summary.set_index("case_id")["annual_financing_saving_vs_private_gbp"].astype(float)
    rows = []
    for case_id, frame in savings.groupby("case_id", sort=False):
        by_route = frame.groupby("route")["saving_economic_life_gbp"].sum()
        by_route_term = frame.groupby("route")["saving_bill_term_gbp"].sum()
        off_bill = (frame.loc[~frame.get("at_cost_applied", pd.Series(False, index=frame.index)).astype(bool)]
                    .groupby("route")["saving_off_bill_leg_gbp"].sum()
                    if "saving_off_bill_leg_gbp" in frame.columns else pd.Series(dtype=float))
        assets = frame.groupby("route")["asset_id"].size()
        at_cost = (frame["at_cost_applied"].astype(bool) if "at_cost_applied" in frame.columns
                   else pd.Series(False, index=frame.index))
        rent = ((frame.loc[at_cost, "saving_bill_term_gbp"] - frame.loc[at_cost, "saving_financing_term_gbp"])
                .groupby(frame.loc[at_cost, "route"]).sum()
                if at_cost.any() and "saving_financing_term_gbp" in frame.columns else pd.Series(dtype=float))
        # Every case is a full case in the financing summary; the base scenario is the fallback.
        total = float(totals.get(case_id, totals.get(base_case(case_id), np.nan)))
        routed_sum = float(by_route.sum())
        for route in ROUTES:
            life = float(by_route.get(route, 0.0))
            bill = float(by_route_term.get(route, 0.0)) if route in BILL_REPRICING_ROUTES else 0.0
            rows.append({
                "case_id": case_id, "route": route, "treatment": ROUTE_TREATMENT[route],
                "assets": int(assets.get(route, 0)),
                "saving_economic_life_gbp_2024": life,
                "bill_allocated_gbp_2024": bill,
                "term_adjustment_gbp_2024": bill - life if route in BILL_REPRICING_ROUTES else 0.0,
                # Inside the term adjustment: the at-cost rent, a transfer (review 28 Sep 2026).
                "at_cost_rent_transfer_gbp_2024": float(rent.get(route, 0.0)) if route in BILL_REPRICING_ROUTES else 0.0,
                "term_adjustment_excluding_rent_gbp_2024": (
                    bill - life - float(rent.get(route, 0.0)) if route in BILL_REPRICING_ROUTES else 0.0),
                # Inside the term adjustment: the capture leg's financing saving on a split
                # contract (Power BECCS), kept off the electricity strike.
                "off_bill_leg_gbp_2024": float(off_bill.get(route, 0.0)) if route in BILL_REPRICING_ROUTES else 0.0,
                "financing_saving_total_gbp_2024": total,
                "identity_residual_gbp_2024": routed_sum - total,
                "identity_kind": "row_partition",
                "identity_within_tolerance": bool(abs(routed_sum - total) <= IDENTITY_TOLERANCE_GBP)
                if np.isfinite(total) else False,
            })
    return pd.DataFrame(rows)


def assert_single_route(savings: pd.DataFrame) -> None:
    """No asset may be re-priced (R1) and also sit in the public account (R2)."""
    counts = savings.groupby(["case_id", "asset_id"])["route"].nunique()
    if (counts > 1).any():
        raise ValueError("an asset carries more than one route")
    if not set(savings["route"]) <= set(ROUTES):
        raise ValueError(f"unknown routes: {sorted(set(savings['route']) - set(ROUTES))}")
