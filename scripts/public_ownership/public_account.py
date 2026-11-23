"""WP-D, WP-I and WP-E: the public account and what it routes to the bill.

Producer and public account (plan section 10): for every register row that is
not re-priced under R1,

    rent    = market revenue + support revenue + Capacity Market revenue
              + LDES floor top-up paid - fuel and carbon - required revenue
    surplus = ownership share x (rent + state overhead) - state overhead
              - the Capacity Market / CCS amount booked on the bill (booked cases)

split into the financing saving (required revenue private minus case) and the
pre-existing rent the state acquires with the asset (a recipient change, marked
as a transfer).  Rows re-priced under R1 are outside the account: an asset is
re-priced or its saving goes to surplus, never both; what the state keeps of an
R1 row's rent after the strike cut is reported as ``r1_retained_rent_gbp`` and
retained with the R2 surplus (item B12).

Which plant is in the state's contracted-at-cost portfolio: renewables and
hydro only.  Gas is paid by the Capacity Market or a strategic reserve and
BECCS by a dispatchable CfD, both outside the wholesale line and outside this
account's recycling; their financing savings are reported as sensitivities on
the Capacity Market and CCS-levy lines.  Storage surplus is never recycled
(arbitrage-only revenue in the model); LDES goes through the cap-and-floor line
(WP-E) and batteries through the Capacity Market sensitivity.

Recycling rules (config ``public_ownership.surplus_recycling.<case>``):

* ``no_assumption`` / ``retained_by_treasury``: nothing reaches the bill;
* ``rebate_per_mwh_eligible_supply`` / ``offset_policy_levies``: the merchant
  and R2 surplus becomes a negative policy row over CfD eligible supply;
* ``cost_recovery_contracting`` (W1): the state's renewable and hydro plant is
  paid what it costs, its rent returned on the wholesale line, a shortfall
  levied separately;
* ``public_cfd_at_cost`` (W1 presented as the contract it is): the same plant
  on a two-way CfD whose strike is its cost, one policy line over CfD eligible
  supply carrying the return or the top-up.

A negative amount to route is a shortfall funded by ``public_shortfall_funding``:
``levy_on_eligible_supply`` puts it on the bill (gross variant); ``exchequer``
keeps it off the bill and reports the fiscal cost.

LDES cap and floor: Ofgem's scheme floors at a return "similar to the cost of
debt", so the floor is required revenue re-annuitised at a configured debt-like
real rate per case (``None`` = the case WACC, the state's own borrowing cost);
the floor at the technology WACC is kept as an upper sensitivity.

W2 (average-cost tariff) is reported, not routed.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from scripts.public_ownership.cfd_repricing import (
    ROUTE_CAPACITY_MARKET, ROUTE_CFD_BILL, ROUTE_CFD_PUBLIC_ACCOUNT, ROUTE_DISPATCHABLE_CFD, ROUTE_MERCHANT,
    ROUTE_STORAGE, base_case,
)
# F27 item 1 (24 Oct 2026): Net Zero Teesside's DPA re-priced on remaining_ccs is an R1 row here.
from scripts.public_ownership.cfd_repricing import ROUTE_DPA_BILL
from scripts.public_ownership.levers import OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT
from scripts.public_ownership.finance import NATIONALISED_CASE, capital_recovery_factor, co_ownership_rate  # noqa: F401
from scripts.public_ownership import sizewell_c_rab
from scripts.public_ownership import stage_timing
from scripts.public_ownership.legislation_year import STATUS_FLEET as STATUS_PRE_POLICY_FLEET
from scripts.public_ownership.levers import (
    AXIS_LDES_FLOOR_BASIS, AXIS_ROUTE_BOOKING, DEFAULT_SETTINGS, LEVER_PRE_2025_CFD, LEVER_SHORTFALL_FUNDING,
    GATED_LEVERS, LEVER_STATE_DEFICIT_FUNDING, LEVER_SURPLUS_RECYCLING,
    LEVER_WHOLESALE_PRICING, LEVERS, PRICING_PAY_AS_BID_CEILING, axis_setting, case_settings, comparator_for_settings,
    spec_for_row, GBE_VALUES, LEVER_NEW_BUILD_GENERATION, LEVER_NEW_BUILD_STORAGE, STORAGE_PRIVATE,
)

AXES = LEVERS

LDES_FLOOR_COST_OF_DEBT_LIKE = "cost_of_debt_like"
LDES_FLOOR_TECHNOLOGY_WACC = "technology_wacc"


RECYCLING_RULES = frozenset({
    "no_assumption", "retained_by_treasury", "rebate_per_mwh_eligible_supply",
    "offset_policy_levies", "cost_recovery_contracting", "public_cfd_at_cost", "rebate_gross",
})
#: Rules that route the merchant, R2 and R1-remainder balances together (the net).
ROUTE_ALL_NET = ("rebate_per_mwh_eligible_supply", "offset_policy_levies")
DEFICITS_EXCHEQUER, DEFICITS_BILL_LEVY = "exchequer", "bill_levy"
SHORTFALL_FUNDING = frozenset({"levy_on_eligible_supply", "exchequer"})
GROUP_R1 = "R1_repriced_not_in_account"
GROUP_R2 = "R2_cfd_public_account"
GROUP_MERCHANT = "merchant_generation"
GROUP_CM = "capacity_market_route_generation"
GROUP_DPA = "dispatchable_cfd_route_generation"
GROUP_LDES = "storage_ldes"
GROUP_BATTERY = "storage_battery"
GROUPS = (GROUP_R1, GROUP_R2, GROUP_MERCHANT, GROUP_CM, GROUP_DPA, GROUP_LDES, GROUP_BATTERY)
AXIS_COLUMNS = tuple(AXES)


#: FES 2050 decision 58: support-line income columns a 2050 register books (support_lines_2050).
#: FES 2050 decision 73: the state's share of a co-owned row's CM revenue, taken back (capacity_market_design).
CM_STATE_ADJUSTMENT = "capacity_market_state_part_adjustment_gbp"
SUPPORT_INCOME_COLUMNS = ("ancillary_services_revenue_gbp", "efw_gate_fee_revenue_gbp", "life_extension_top_up_gbp",
                          # Decision 63: the battery revenue top-up; decision 67: the batteries' declared
                          # sub-hourly revenue (a 2050 register only).
                          "battery_revenue_top_up_gbp", "battery_subhourly_revenue_gbp")
#: User decision (6 Oct 2026): support-line income an LDES floor nets besides market and Capacity
#: Market revenue (``support_lines_2050``'s storage lines). The gate fee is EfW's only, and the
#: battery and life-extension top-ups are other support lines, never paid to a floored store.
LDES_FLOOR_OTHER_INCOME = ("ancillary_services_revenue_gbp", "battery_subhourly_revenue_gbp")
#: F19 decision 5: the register's private-basis battery top-up, written only when each case's top-up
#: is sized at its owner's cost (``support_lines_2050.COLUMN_BATTERY_TOP_UP_PRIVATE_BASIS``).
BATTERY_TOP_UP_PRIVATE_BASIS = "battery_revenue_top_up_private_basis_gbp"


def account_settings(config: Mapping) -> dict:
    """Read the LDES and Capacity Market decisions.

    Item K1: the recycling rule and the shortfall funding are levers read from
    each case's register rows (``surplus_recycling``, ``shortfall_funding``).
    A ``surplus_recycling`` mapping and a ``public_shortfall_funding`` value are
    still accepted here as the fallback for a register written without those
    columns.
    """
    recycling = dict(config.get("surplus_recycling") or {})
    for case_id, rule in recycling.items():
        if rule not in RECYCLING_RULES:
            raise ValueError(f"surplus_recycling.{case_id}={rule!r}; supported: {sorted(RECYCLING_RULES)}")
    funding = config.get("public_shortfall_funding", "levy_on_eligible_supply")
    if funding not in SHORTFALL_FUNDING:
        raise ValueError(f"public_shortfall_funding={funding!r}; supported: {sorted(SHORTFALL_FUNDING)}")
    bill = config.get("consumer_bill") or {}
    ldes = dict(bill.get("ldes_cap_and_floor") or {})
    cm = dict(bill.get("capacity_market_sensitivity") or {})
    floor_rates = dict((ldes.get("floor_return") or {}).get("real_rate_by_case") or {})
    # FES 2050 decision 56: which rate a row's floor takes. ``case`` (the default, every 2030
    # config): the case's rate on every row. ``row_owner``: a row the state holds any of takes the
    # case's rate, a private row the private rate, so private stores in a P2 share case are made
    # whole at the private floor, not the state's.
    rate_key = str((ldes.get("floor_return") or {}).get("rate_key", "case"))
    if rate_key not in FLOOR_RATE_KEYS:
        raise ValueError(f"ldes_cap_and_floor.floor_return.rate_key={rate_key!r}; supported: {FLOOR_RATE_KEYS}")
    for case_id, rate in floor_rates.items():
        if rate is not None and not (0.0 < float(rate) < 1.0):
            raise ValueError(f"ldes_cap_and_floor.floor_return.real_rate_by_case.{case_id} must be a rate in (0, 1) or null")
    return {
        "recycling": recycling,
        "shortfall_funding": funding,
        "ldes_enabled": bool(ldes.get("enabled", False)),
        "ldes_carriers": [str(c) for c in (ldes.get("carriers") or ["LAES"])],
        "ldes_floor_real_rate_by_case": {k: (None if v is None else float(v)) for k, v in floor_rates.items()},
        **({"ldes_floor_rate_key": rate_key} if rate_key != "case" else {}),
        "cm_sensitivity_enabled": bool(cm.get("enabled", False)),
        # OW10: Sizewell C's RAB return at the public rate in the public cases (the bill reads it).
        sizewell_c_rab.CONFIG_KEY: sizewell_c_rab.settings_from_config(bill),
    }


def _group(route: str, component_type: str, carrier: str, ldes_carriers) -> str:
    if route in (ROUTE_CFD_BILL, ROUTE_DPA_BILL):
        return GROUP_R1
    if route == ROUTE_CFD_PUBLIC_ACCOUNT:
        return GROUP_R2
    if component_type == "StorageUnit" or route == ROUTE_STORAGE:
        return GROUP_LDES if carrier in ldes_carriers else GROUP_BATTERY
    if route == ROUTE_CAPACITY_MARKET:
        return GROUP_CM
    if route == ROUTE_DISPATCHABLE_CFD:
        return GROUP_DPA
    return GROUP_MERCHANT


#: FES 2050 decision 56: the keys a floor rate is looked up by (``floor_return.rate_key``).
FLOOR_RATE_KEYS = ("case", "row_owner")


def _floor_required_revenue(frame: pd.DataFrame, rate_by_case: Mapping[str, float | None],
                            rate_key: str = "case") -> pd.Series:
    """Required revenue with the capital charge re-annuitised at the floor rate (see :func:`_floor_parts`)."""
    return _floor_parts(frame, rate_by_case, rate_key)["row"]


def _num_col(frame: pd.DataFrame, column: str, default=0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _floor_parts(frame: pd.DataFrame, rate_by_case: Mapping[str, float | None], rate_key: str = "case") -> dict:
    """The floor's required revenue per row, and per owner on a separately financed co-owned row.

    Review 2 (3 Oct 2026): the capital is re-annuitised over the years the register annuitised it
    (``capital_annuity_years``: a decision-66 level-annuity row's economic life), so its capex is
    recovered exactly; and under ``row_owner`` a co-owned row (decision 59) is floored per owner:
    under separate financing the state's part at the case's floor rate and the co-owner's at the
    private case's (``state`` and ``private``: each a whole-row floor at that owner's capex and rate,
    with ``separate`` marking the rows), under pooled financing the whole row at the blend of the two
    floor rates. A register without these columns (every 2030 one) is floored as before.
    """
    out = _floor_parts_original(frame, rate_by_case, rate_key)
    basis = (frame["co_ownership_basis"].where(frame["co_ownership_basis"].notna(), "").astype(str)
             if "co_ownership_basis" in frame.columns else pd.Series("", index=frame.index))
    co = basis.ne("") & (rate_key == "row_owner")
    separate = co & basis.ne("pooled")
    out["separate"] = separate
    out["state"] = pd.Series(np.nan, index=frame.index)
    out["private"] = pd.Series(np.nan, index=frame.index)
    if not co.any():
        return out
    other = _num_col(frame, "annual_fixed_om_gbp").fillna(0.0) + _num_col(frame, "annual_variable_om_gbp").fillna(0.0)
    years = _num_col(frame, "capital_annuity_years", np.nan).fillna(_num_col(frame, "remaining_life_years", np.nan))
    share = _num_col(frame, "ownership_share").fillna(0.0)
    private_floor = rate_by_case.get("private")
    row = out["row"].copy()
    for i in np.flatnonzero(co.to_numpy()):
        label = frame.index[i]
        n = float(years.iloc[i])
        case_floor = rate_by_case.get(base_case(frame["case_id"].iloc[i]))
        if not np.isfinite(n) or n <= 0:
            continue
        if separate.iloc[i]:
            state_rate = case_floor if case_floor is not None else float(frame["state_part_rate"].iloc[i])
            co_rate = private_floor if private_floor is not None else float(frame["co_owner_rate"].iloc[i])
            state = float(frame["capex_state_view_gbp"].iloc[i]) * capital_recovery_factor(float(state_rate), n)
            private = float(frame["capex_private_view_gbp"].iloc[i]) * capital_recovery_factor(float(co_rate), n)
            f = float(share.iloc[i])
            out["state"].at[label] = state + float(other.iloc[i])
            out["private"].at[label] = private + float(other.iloc[i])
            row.at[label] = f * state + (1.0 - f) * private + float(other.iloc[i])
        else:
            capital = float(_num_col(frame, "annual_capital_recovery_gbp").fillna(0.0).iloc[i])
            r = float(_num_col(frame, "wacc", np.nan).iloc[i])
            if case_floor is None or private_floor is None or not np.isfinite(r):
                continue
            blend = co_ownership_rate(float(case_floor), float(private_floor), float(share.iloc[i]), n)
            row.at[label] = (capital / capital_recovery_factor(r, n) * capital_recovery_factor(blend, n)
                             + float(other.iloc[i]))
    out["row"] = row
    return out


def _floor_parts_original(frame: pd.DataFrame, rate_by_case: Mapping[str, float | None],
                          rate_key: str = "case") -> dict:
    """Required revenue with the capital charge re-annuitised at the case's floor rate.

    capex = capital recovery / CRF(wacc, remaining life); floor capital charge =
    capex x CRF(floor rate, remaining life); fixed and variable O&M unchanged.
    A missing or null rate keeps the case WACC.  ``rate_key="row_owner"`` (decision 56)
    looks the rate up by the row's owner: the case's rate where the state holds any of the
    row (``ownership_share > 0``), the private case's rate on a wholly private row.
    """
    capital = pd.to_numeric(frame["annual_capital_recovery_gbp"], errors="coerce").fillna(0.0)
    wacc = pd.to_numeric(frame["wacc"], errors="coerce")
    life = pd.to_numeric(frame["remaining_life_years"], errors="coerce")
    if "capital_annuity_years" in frame.columns:
        # Decision 66 (review 2): the years the register annuitised the capital over.
        life = pd.to_numeric(frame["capital_annuity_years"], errors="coerce").fillna(life)
    other = (pd.to_numeric(frame["annual_fixed_om_gbp"], errors="coerce").fillna(0.0)
             + pd.to_numeric(frame["annual_variable_om_gbp"], errors="coerce").fillna(0.0))
    floor_capital = np.zeros(len(frame))
    if rate_key == "row_owner":
        held = pd.to_numeric(frame.get("ownership_share", pd.Series(0.0, index=frame.index)),
                             errors="coerce").fillna(0.0).gt(0.0).to_numpy()
        keys = [base_case(c) if h else "private" for c, h in zip(frame["case_id"], held)]
    else:
        keys = [base_case(c) for c in frame["case_id"]]
        if "finance_vintage" in frame.columns:
            # User decision (16 Nov 2026, the LDES floor follows the build date, not the law year): a store committed
            # before the legislation (``legacy`` or ``in_construction``) that the state holds none of is a private
            # store in every case, floored at the private case's rate as in private (a null case rate would fall to
            # its full WACC), as the F10 rule below does for a wholly private store inside a P2 variant.
            held = pd.to_numeric(frame.get("ownership_share", pd.Series(0.0, index=frame.index)),
                                 errors="coerce").fillna(0.0).gt(0.0).to_numpy()
            committed_before = frame["finance_vintage"].astype(str).ne("new_build").to_numpy()
            keys = ["private" if (cb and not h) else k for k, cb, h in zip(keys, committed_before, held)]
        if LEVER_NEW_BUILD_GENERATION in frame.columns or LEVER_NEW_BUILD_STORAGE in frame.columns:
            # F10: GB Energy as announced leaves storage private inside a P2 variant, whose floor key is the
            # state's own rate (the case WACC); a wholly private store there floors at the private case's rate.
            # Integration (7 Oct 2026, user decision): the same for new_build_storage = private
            # (public_ownership@developer_storage_private), whose P2 floor rate is null and so fell to the
            # technology WACC.
            gbe = (frame[LEVER_NEW_BUILD_GENERATION].astype(str).isin(GBE_VALUES).to_numpy()
                   if LEVER_NEW_BUILD_GENERATION in frame.columns else np.zeros(len(frame), dtype=bool))
            storage_private = (frame[LEVER_NEW_BUILD_STORAGE].astype(str).eq(STORAGE_PRIVATE).to_numpy()
                               if LEVER_NEW_BUILD_STORAGE in frame.columns else np.zeros(len(frame), dtype=bool))
            held = pd.to_numeric(frame.get("ownership_share", pd.Series(0.0, index=frame.index)),
                                 errors="coerce").fillna(0.0).gt(0.0).to_numpy()
            keys = ["private" if (g or sp) and not h else k
                    for k, g, sp, h in zip(keys, gbe, storage_private, held)]
    for i, (key, cap, r, n) in enumerate(zip(keys, capital, wacc, life)):
        floor_rate = rate_by_case.get(key)
        if floor_rate is None or not np.isfinite(r) or not np.isfinite(n) or n <= 0:
            floor_capital[i] = cap
            continue
        floor_capital[i] = cap / capital_recovery_factor(float(r), float(n)) * capital_recovery_factor(float(floor_rate), float(n))
    return {"row": pd.Series(floor_capital, index=frame.index) + other}


def _floor_parts_on(frame: pd.DataFrame, rows: np.ndarray, rate_by_case: Mapping[str, float | None],
                    rate_key: str = "case") -> dict:
    """:func:`_floor_parts` computed on the ``rows`` (a boolean mask) alone, NaN (``separate`` False) elsewhere.

    F20 phase 1 (13 Oct 2026): :func:`ldes_floor_topups` reads the floor on LDES rows only,
    and :func:`_floor_parts_original` walks its rows in Python with two annuity factors
    each, over every row of every market case; each row's floor depends on that row
    alone, so the LDES rows' values are the ones the whole frame gave them.
    """
    positions = np.flatnonzero(rows)
    sub = _floor_parts(frame.iloc[positions], rate_by_case, rate_key)
    out = {}
    for key, value in sub.items():
        if key == "separate":
            full = np.zeros(len(frame), dtype=bool)
        else:
            full = np.full(len(frame), np.nan)
        full[positions] = np.asarray(value)
        out[key] = pd.Series(full, index=frame.index)
    return out


def ldes_floor_eligible_share(frame: pd.DataFrame) -> pd.Series:
    """Each row's share Ofgem's LDES cap and floor covers: plant built from 2025, whatever the law year.

    User decision (16 Nov 2026): the floor is Ofgem's regime for new long-duration storage, which does
    not depend on when an ownership law passes. It had followed ``finance_vintage == "new_build"``,
    which since F28 follows the legislation year, so a 2028 law left CP30's new LAES (committed before
    it) without a floor in every 2030 case. A row is now covered unless it is the pre-2025 fleet
    (``legislation_status`` ``pre_policy_fleet``); the share of a stock-flow cohort that refurbishes
    existing plant (``cohort_refurbishment_share``: the 2040/2050 pumped hydro) is existing plant and
    is left out pro rata. A row without a status keeps the old test.
    """
    if "legislation_status" in frame.columns:
        raw = frame["legislation_status"]
        status = raw.astype(str)
        covered = ~status.eq(STATUS_PRE_POLICY_FLEET)
        if "finance_vintage" in frame.columns:
            no_status = raw.isna() | status.isin(["", "nan", "None"])
            covered = covered.where(~no_status, frame["finance_vintage"].astype(str).eq("new_build"))
    elif "finance_vintage" in frame.columns:
        covered = frame["finance_vintage"].astype(str).eq("new_build")
    else:
        covered = pd.Series(True, index=frame.index)
    share = covered.astype(float)
    if "cohort_refurbishment_share" in frame.columns:
        refurb = pd.to_numeric(frame["cohort_refurbishment_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        share = share * (1.0 - refurb)
    return share


def ldes_floor_topups(frame: pd.DataFrame, settings: Mapping) -> pd.DataFrame:
    """The LDES cap-and-floor top-up per register row, on both floor bases and on the case's own.

    One function for the register (``producer_rent_gbp``) and the account, so
    the owner's rent and the bill line cannot drift (item B2).  A row is floored
    when it is long-duration storage (``settings['ldes_carriers']``) built from 2025
    (:func:`ldes_floor_eligible_share`, user decision 16 Nov 2026):
    Ofgem's scheme is for new projects, and existing pumped hydro is legacy.
    The floor tops up what the store earns, market revenue plus Capacity Market
    revenue (item MR4) plus any ancillary-service or sub-hourly revenue the
    register books it (``LDES_FLOOR_OTHER_INCOME``, user decision 6 Oct 2026),
    to required revenue re-annuitised at the case's
    debt-like floor rate (``ldes_floor_topup_gbp``) or at the technology WACC
    (``ldes_floor_topup_at_wacc_gbp``).  ``ldes_floor_topup_paid_gbp`` is the
    one the case's ``ldes_floor_basis`` lever makes the bill pay, zero when the
    scheme is disabled: that is the money that reaches the store's owners.
    """
    out = pd.DataFrame(index=frame.index)
    eligible = ldes_floor_eligible_share(frame)
    is_storage = frame["component_type"].astype(str).eq("StorageUnit")
    is_ldes = (is_storage & frame["carrier"].astype(str).isin(list(settings.get("ldes_carriers") or []))
               & eligible.gt(0.0))
    resolved = pd.to_numeric(frame["annual_required_revenue_gbp"], errors="coerce").notna()
    is_ldes = is_ldes & resolved
    cm_revenue = (pd.to_numeric(frame["capacity_market_revenue_gbp"], errors="coerce").fillna(0.0)
                  if "capacity_market_revenue_gbp" in frame.columns else pd.Series(0.0, index=frame.index))
    market = pd.to_numeric(frame["market_revenue_gbp"], errors="coerce").fillna(0.0)
    # User decision (6 Oct 2026): the floor nets every income the store earns, as the battery
    # top-up does through the producer rent: the support lines the register books to storage
    # rows (ancillary services, the declared sub-hourly revenue), each where present.
    other = sum((_num_col(frame, c).fillna(0.0) for c in LDES_FLOOR_OTHER_INCOME if c in frame.columns),
                pd.Series(0.0, index=frame.index))
    floor_income = market + cm_revenue + other
    w = eligible.to_numpy(dtype=float)
    out["ldes_capacity_market_revenue_gbp"] = np.where(is_ldes, w * cm_revenue, 0.0)
    parts = _floor_parts_on(frame, is_ldes.to_numpy(dtype=bool), settings.get("ldes_floor_real_rate_by_case") or {},
                            settings.get("ldes_floor_rate_key", "case"))
    floor_rr = parts["row"]
    rr = pd.to_numeric(frame["annual_required_revenue_gbp"], errors="coerce").fillna(0.0)
    topup = np.maximum(floor_rr - floor_income, 0.0)
    separate = parts.get("separate")
    if separate is not None and bool((separate & is_ldes).any()):
        # Review 2: a separately financed co-owned store is floored per owner, each on its share of the income.
        f = _num_col(frame, "ownership_share").fillna(0.0)
        state_part = f * np.maximum(parts["state"] - floor_income, 0.0)
        private_part = (1.0 - f) * np.maximum(parts["private"] - floor_income, 0.0)
        topup = np.where(separate, state_part + private_part, topup)
        out["ldes_floor_topup_state_part_gbp"] = np.where(is_ldes & separate, w * state_part, np.nan)
    out["ldes_floor_topup_gbp"] = np.where(is_ldes, w * topup, 0.0)
    out["ldes_floor_topup_at_wacc_gbp"] = np.where(is_ldes, w * np.maximum(rr - floor_income, 0.0), 0.0)
    basis = (frame[AXIS_LDES_FLOOR_BASIS].fillna(LDES_FLOOR_COST_OF_DEBT_LIKE).astype(str)
             if AXIS_LDES_FLOOR_BASIS in frame.columns else pd.Series(LDES_FLOOR_COST_OF_DEBT_LIKE, index=frame.index))
    paid = np.where(basis.eq(LDES_FLOOR_TECHNOLOGY_WACC), out["ldes_floor_topup_at_wacc_gbp"], out["ldes_floor_topup_gbp"])
    out["ldes_floor_topup_paid_gbp"] = paid if settings.get("ldes_enabled", False) else 0.0
    return out


def co_ownership_adjustments(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Decision 59 (review 2): the state's surplus moved onto its own cost, and onto its own floor top-up.

    ``(cost, floor)``: the register's ``co_ownership_state_adjustment_gbp``, and on a separately
    financed co-owned store the state's own floor top-up less its share of the row's
    (``ldes_floor_topup_state_part_gbp``, debt-like floor basis only). Zero without the columns.
    """
    cost = _num_col(frame, "co_ownership_state_adjustment_gbp").fillna(0.0)
    floor = pd.Series(0.0, index=frame.index)
    if "ldes_floor_topup_state_part_gbp" in frame.columns:
        state_part = pd.to_numeric(frame["ldes_floor_topup_state_part_gbp"], errors="coerce")
        basis = (frame[AXIS_LDES_FLOOR_BASIS].fillna(LDES_FLOOR_COST_OF_DEBT_LIKE).astype(str)
                 if AXIS_LDES_FLOOR_BASIS in frame.columns else pd.Series(LDES_FLOOR_COST_OF_DEBT_LIKE, index=frame.index))
        paid = _num_col(frame, "ldes_floor_topup_paid_gbp").fillna(0.0)
        share = _num_col(frame, "ownership_share").fillna(0.0)
        floor = (state_part - share * paid).where(state_part.notna() & basis.eq(LDES_FLOOR_COST_OF_DEBT_LIKE)
                                                  & (paid > 0), 0.0)
    return cost, floor


def state_surplus(share: pd.Series, rent: pd.Series, overhead: pd.Series) -> pd.Series:
    """The state's surplus on a row: its share of the rent before the holding company, less the holding company.

    Item OH1 charges the overhead off the state's share already
    (``annual_state_overhead`` is capacity x share x rate), so scaling a rent
    that includes it by the share again charged a 30% stake only 9% of its own
    centre (item B5).  The overhead is the state's cost alone: the private
    co-owner does not run the state's holding company.
    """
    return share * (rent + overhead) - overhead


#: ``pay_as_bid_ceiling`` not given: :func:`build_public_account` reads it off the register's private rows.
_FROM_REGISTER = object()


def pay_as_bid_ceiling_gbp(frame: pd.DataFrame) -> float | None:
    """Item K3: the most pay-as-bid could remove, read off the private case's resolved generator rows.

    Holding offers and dispatch fixed, every accepted GB generator is paid its
    short-run cost (fuel, carbon and variable O&M) instead of the clearing
    price; the difference is the wholesale margin the private case pays.  A
    physical quantity of the run, read once from the private rows (house rule
    2), a ceiling never a case (plan Part 7.3).  CfD plant is paid its strike
    whatever the wholesale price, so a lower wholesale payment is made good by
    a larger top-up and the bill saves nothing on it: only non-CfD generation
    carries removable wholesale margin.
    """
    private_rows = frame.loc[frame["case_id"].eq("private") & frame["component_type"].eq("Generator")
                             & frame["evidence_status"].eq("resolved")]
    support = (private_rows["network_support_type"] if "network_support_type" in private_rows.columns
               else private_rows["support_type"]).fillna("").astype(str).str.strip().str.lower()
    private_rows = private_rows.loc[~support.eq("cfd")]
    if "wholesale_revenue_gbp" not in private_rows.columns:
        return None
    margin = (pd.to_numeric(private_rows["wholesale_revenue_gbp"], errors="coerce").fillna(0.0)
              - pd.to_numeric(private_rows["fuel_carbon_gbp"], errors="coerce").fillna(0.0)
              - pd.to_numeric(private_rows["annual_variable_om_gbp"], errors="coerce").fillna(0.0))
    return float(margin.clip(lower=0.0).sum())


def build_public_account(register: pd.DataFrame, savings: pd.DataFrame, settings: Mapping, *,
                         settings_by_case: Mapping[str, Mapping[str, str]] | None = None, rows_for=None,
                         pay_as_bid_ceiling=_FROM_REGISTER) -> tuple[pd.DataFrame, dict]:
    """Return the account by case and group, and the amounts each rule routes to the bill.

    30 Sep 2026: ``register`` may hold a batch of whole cases.  The caller then
    passes what the batch cannot show: ``settings_by_case`` (every case in
    register order, from which each case's comparator is picked, house rule 3),
    ``rows_for(case_id)`` (a comparator's register rows) and the
    ``pay_as_bid_ceiling`` (:func:`pay_as_bid_ceiling_gbp` of the private rows).
    Every step is per row or per case, so a batch's account is its cases' part
    of the whole-register account (``build_public_account.merge_accounts``).
    """
    # F20 phase 0: the steps of one batch, timed only when PYPSA_GB_STAGE_TIMING is set.
    lap = stage_timing.laps("public_account.build")
    required = {"case_id", "asset_id", "component_type", "carrier", "ownership_share", "evidence_status",
                "annual_required_revenue_gbp", "annual_capital_recovery_gbp", "annual_fixed_om_gbp",
                "annual_variable_om_gbp", "wacc", "remaining_life_years", "market_revenue_gbp",
                "support_revenue_gbp", "fuel_carbon_gbp", "dispatch_mwh"}
    if missing := required - set(register.columns):
        raise ValueError(f"Finance register lacks columns: {sorted(missing)}")
    if register["market_revenue_gbp"].isna().all():
        raise ValueError("The register carries no settlement: run the WP-D settlement before the public account")
    # F20 phase 1: the savings indexed once (the route, the R1 cut and the at-cost flag are read off it).
    by_case_asset = savings.set_index(["case_id", "asset_id"])
    routes = by_case_asset["route"]
    # House rule 3: the saving part is measured against the private case at the
    # same point on every all_cases lever (read off the lever columns).
    if settings_by_case is None:
        settings_by_case = {case_id: case_settings(rows) for case_id, rows in register.groupby("case_id", sort=False)}
    batch_cases = list(register["case_id"].drop_duplicates())
    comparator_by_case = {case_id: comparator_for_settings(case_id, settings_by_case) for case_id in batch_cases}
    if rows_for is None:
        rr_by_case_asset = register.set_index(["case_id", "asset_id"])["annual_required_revenue_gbp"]
    else:
        comparators = list(dict.fromkeys(comparator_by_case.values()))
        rr_by_case_asset = pd.concat({c: rows_for(c).set_index("asset_id")["annual_required_revenue_gbp"]
                                      for c in comparators})
    # A case that acquires the legacy fleet has no market account (every
    # payment is intra-public); its tariff is item N5's.  Only market cases are
    # accounted, read off their lever columns.
    market_by_case = {case_id: spec_for_row(settings_by_case[case_id]).market_bill() for case_id in batch_cases}
    market = register["case_id"].map(market_by_case).astype(bool)
    frame = register.loc[register["evidence_status"].eq("resolved") & market].copy()
    if pay_as_bid_ceiling is _FROM_REGISTER:
        pay_as_bid_ceiling = pay_as_bid_ceiling_gbp(frame)
    lap("settings_and_comparators")
    frame["route"] = routes.reindex(pd.MultiIndex.from_arrays([frame["case_id"], frame["asset_id"]])).to_numpy()
    frame["group"] = [
        _group(r, t, c, settings["ldes_carriers"]) for r, t, c in zip(frame["route"], frame["component_type"], frame["carrier"])
    ]
    for column in ("capacity_market_revenue_gbp", "carbon_removal_cfd_revenue_gbp", "annual_tax_equivalent_gbp"):
        if column not in frame.columns:
            frame[column] = 0.0
    for column in ("market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp", "ownership_share",
                   "capacity_market_revenue_gbp", "carbon_removal_cfd_revenue_gbp", "annual_tax_equivalent_gbp"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["rr_private_gbp"] = rr_by_case_asset.reindex(
        pd.MultiIndex.from_arrays([frame["case_id"].map(comparator_by_case), frame["asset_id"]])).to_numpy()
    lap("routes_groups_and_private_rr")
    # LDES cap and floor: floor at a debt-like return; the floor at the
    # technology WACC is kept as the upper sensitivity.  The floor tops up what
    # the store earns, and it earns Capacity Market revenue as well as market
    # revenue (item MR4: CP30's long-duration stores hold ~100 m of agreements,
    # which the bill's Capacity Market line already pays), so both are netted;
    # topping up to required revenue on market revenue alone paid the Capacity
    # Market twice.  Only new build is floored: Ofgem's scheme is for new
    # projects, and existing pumped hydro is registered as legacy
    # (public_ownership.existing_storage_sites).
    floors = ldes_floor_topups(frame, settings)
    private_top_up = (_private_battery_top_up(register, rows_for)
                      if BATTERY_TOP_UP_PRIVATE_BASIS in frame.columns else None)
    for column in floors.columns:
        frame[column] = floors[column].to_numpy()
    lap("ldes_floor")
    overhead = (pd.to_numeric(frame["annual_state_overhead_gbp"], errors="coerce").fillna(0.0)
                if "annual_state_overhead_gbp" in frame.columns else pd.Series(0.0, index=frame.index))
    # Capacity Market revenue (item MR4) is income the owner keeps like any other:
    # the bill's CM line pays it, so a state owner's rent includes it and the
    # recycling rules return it; leaving it out understated every surplus.  The
    # LDES floor top-up the bill pays reaches the store's owners the same way
    # (item B2): until 26 Sep 2026 the bill paid it and no owner received it.
    # User decision 2 (29 Sep 2026): Power BECCS's carbon-removal leg is paid by its own
    # carbon CfD (the register's carbon_removal_cfd_revenue_gbp, the owner's income like
    # any other), so the rent on a split contract is its electricity leg's and the state
    # carries no capture-leg deficit.
    frame["rent_gbp"] = (frame["market_revenue_gbp"] + frame["support_revenue_gbp"]
                         + frame["capacity_market_revenue_gbp"] + frame["ldes_floor_topup_paid_gbp"]
                         + frame["carbon_removal_cfd_revenue_gbp"]
                         - frame["fuel_carbon_gbp"] - frame["annual_required_revenue_gbp"])
    # FES 2050 decision 58: the support lines the register books are the owner's income too
    # (absent from every 2030 register, so no 2030 rent moves).
    for column in SUPPORT_INCOME_COLUMNS:
        if column in frame.columns:
            frame["rent_gbp"] = frame["rent_gbp"] + pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    # FES 2050 decision 54: a case on another private rate path settles its post-AR7 contracts at
    # that path's strikes; the register's support revenue already carries the cut, recorded here by
    # who bears it (the state on its share), for the party flows.
    if "cfd_strike_offset_gbp_per_mwh" in frame.columns and "cfd_paid_mwh" in frame.columns:
        cut = (pd.to_numeric(frame["cfd_strike_offset_gbp_per_mwh"], errors="coerce").fillna(0.0)
               * pd.to_numeric(frame["cfd_paid_mwh"], errors="coerce").fillna(0.0))
        frame["rate_path_cut_gbp"] = cut
        frame["rate_path_cut_borne_by_state_gbp"] = cut * frame["ownership_share"]
    frame["surplus_gbp"] = state_surplus(frame["ownership_share"], frame["rent_gbp"], overhead)
    # Decision 59 (review 2): a separately financed co-owned row's state surplus is on its own cost and floor.
    co_cost, co_floor = co_ownership_adjustments(frame)
    # F27 item 1 (27 Oct 2026): the state's share carries the whole owners'-value premium (zero without it).
    premium_adjustment = _num_col(frame, OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT).fillna(0.0)
    co_cost = co_cost + premium_adjustment
    frame["surplus_gbp"] = frame["surplus_gbp"] + co_cost + co_floor
    if CM_STATE_ADJUSTMENT in frame.columns:
        # FES 2050 decision 73: a co-owned row's CM revenue is its co-owner's alone once the state's share
        # leaves the auction, so the state's share of it comes back out (a rent part, never a saving).
        # Absent from every 2030 register.
        frame["surplus_gbp"] = frame["surplus_gbp"] + _num_col(frame, CM_STATE_ADJUSTMENT).fillna(0.0)
    # The saving part on the same footing: the state's share of the private
    # required revenue less its share of the case's before the holding company,
    # less the holding company (item B5).  On a split contract only the electricity
    # leg's saving is the state's to book (the capture leg's is a smaller carbon CfD,
    # the Exchequer's), and a tax equivalent the state pays is no saving at all.
    leg = (pd.to_numeric(frame["electricity_leg_share"], errors="coerce").fillna(1.0)
           if "electricity_leg_share" in frame.columns else pd.Series(1.0, index=frame.index))
    tax_equivalent = frame["annual_tax_equivalent_gbp"]
    frame["saving_part_gbp"] = state_surplus(
        frame["ownership_share"],
        leg * (frame["rr_private_gbp"] - (frame["annual_required_revenue_gbp"] - tax_equivalent)) - tax_equivalent,
        overhead) + leg * co_cost
    frame["rent_part_gbp"] = frame["surplus_gbp"] - frame["saving_part_gbp"]
    lap("rent_and_surplus")
    # Item B3: under route_booking = booked the bill's Capacity Market and CCS
    # lines are cut by the state's financing saving on the plant they pay, so
    # that plant receives that much less; its rent must fall by the same amount
    # or the state keeps the saving twice (once on the bill, once in its rent).
    # F20 phase 1: each case's lever reads made once per case, not once per row (the same values).
    batch_frame_cases = list(frame["case_id"].drop_duplicates())
    is_public_case = frame["case_id"].map(
        {c: c != "private" and spec_for_row(settings_by_case[c]).is_public() for c in batch_frame_cases}).astype(bool)
    booked = frame["case_id"].map(
        {c: str(settings_by_case[c].get(AXIS_ROUTE_BOOKING, DEFAULT_SETTINGS[AXIS_ROUTE_BOOKING])) == "booked"
         for c in batch_frame_cases})
    booked = booked.astype(bool) & is_public_case
    # F19 decision 5: where the register sizes each case's battery top-up at its owner's cost, the
    # batteries' financing saving already reaches the bill on the top-up line, so the battery part
    # of the Capacity Market booking is retired (it returned the private top-up's overpayment to state
    # batteries through the CM line, which a pay-as-clear auction would not do).
    cm_groups = ([GROUP_CM] if BATTERY_TOP_UP_PRIVATE_BASIS in frame.columns else [GROUP_CM, GROUP_BATTERY])
    cm_booked = booked & frame["group"].isin(cm_groups) & bool(settings["cm_sensitivity_enabled"])
    ccs_booked = booked & frame["group"].eq(GROUP_DPA)
    frame["saving_part_before_booking_gbp"] = frame["saving_part_gbp"]
    frame["booked_to_bill_gbp"] = np.where(cm_booked | ccs_booked, frame["saving_part_gbp"], 0.0)
    # User decision 12 (29 Sep 2026): a negative Capacity Market booking is a state
    # deficit, not a bill line.  The Capacity Market clears at an auction price, so the
    # state's higher cost (its overhead) does not raise it.  Under state_deficit_funding
    # = exchequer (central) the booking is floored at zero at the route total of each
    # case, so the CM line never rises and the negative stays in the state's CM groups
    # (the Exchequer funds it); under bill_levy it passes to the bill as before.
    deficit_rule = frame["case_id"].map(
        {c: str(settings_by_case[c].get(LEVER_STATE_DEFICIT_FUNDING, DEFAULT_SETTINGS[LEVER_STATE_DEFICIT_FUNDING]))
         for c in batch_frame_cases})
    cm_route_total = pd.Series(np.where(cm_booked, frame["booked_to_bill_gbp"], 0.0), index=frame.index).groupby(
        frame["case_id"]).transform("sum")
    floored = cm_booked & deficit_rule.eq(DEFICITS_EXCHEQUER) & (cm_route_total < 0)
    frame["cm_booking_floored_gbp"] = np.where(floored, frame["booked_to_bill_gbp"], 0.0)
    frame["booked_to_bill_gbp"] = np.where(floored, 0.0, frame["booked_to_bill_gbp"])
    frame["surplus_gbp"] = frame["surplus_gbp"] - frame["booked_to_bill_gbp"]
    frame["saving_part_gbp"] = frame["saving_part_gbp"] - frame["booked_to_bill_gbp"]
    lap("per_row_lever_maps_and_booking")
    # Item B12: a re-priced (R1) row's rent after the strike cut stays with its
    # owners.  The bill takes ``saving_bill_term_gbp`` off the row (the strike
    # cut times the energy it settles, exact at the strike-entry level); what
    # remains of the state's share is kept by the state and was unreported.
    bill_cut = (by_case_asset["saving_bill_term_gbp"]
                .reindex(pd.MultiIndex.from_arrays([frame["case_id"], frame["asset_id"]])).to_numpy())
    bill_cut = np.nan_to_num(pd.to_numeric(pd.Series(bill_cut), errors="coerce").to_numpy())
    is_r1 = frame["group"].eq(GROUP_R1).to_numpy()
    frame["r1_bill_cut_gbp"] = np.where(is_r1, bill_cut, 0.0)
    # F27 item 1 (24 Oct 2026): the part of the R1 cut taken off the remaining_ccs line (NZT's DPA).
    frame["dpa_bill_cut_gbp"] = np.where(is_r1 & frame["route"].eq(ROUTE_DPA_BILL).to_numpy(), bill_cut, 0.0)
    # Who funds the cut depends on what it is.  On the financing-saving basis
    # (P1's own contract form) the cut is the WHOLE row's required-revenue fall:
    # the stake refinances the whole JV at the Scenario 2 rate, so the co-owner's
    # (1 - share) of the cut is its own cost saving passed into the auction-cleared
    # strike and its rent is the private rent, untouched; the cut splits pro rata.
    # On the at-cost basis the cut is sized on the state's share alone
    # (``share x (contract revenue - cost)``: "a 30% stake can re-price 30% of a
    # contract and no more", plan of record D3), so it comes out of the state's
    # share and the co-owner is untouched.
    at_cost = (by_case_asset["at_cost_applied"]
               .reindex(pd.MultiIndex.from_arrays([frame["case_id"], frame["asset_id"]])).fillna(False)
               .astype(bool).to_numpy() if "at_cost_applied" in savings.columns else np.zeros(len(frame), bool))
    # Decision 59 (review 2): under separate financing a co-owner's cost is untouched by the stake, so the
    # cut on the financing-saving basis is the state's own part.
    separate_co = (frame["co_ownership_basis"].fillna("").astype(str).isin(["separate", "separate_with_private_reduction"])
                   .to_numpy() if "co_ownership_basis" in frame.columns else np.zeros(len(frame), bool))
    # Decision 68: the co-owner's own cost-of-capital gain (the reduction sensitivity) is the part it bears.
    co_gain = _num_col(frame, "co_investment_rate_gain_gbp").fillna(0.0).to_numpy()
    # F27 item 1 (user, 27 Oct 2026): on the financing-saving basis the whole row's cut is net of the
    # owners'-value premium, which is the state's alone, so the co-owner's part of the cut is its own
    # saving without it and the state's part is its pro rata share less the co-owners' part of the premium.
    frame["r1_cut_borne_by_state_gbp"] = np.where(
        is_r1, np.where(at_cost | separate_co, frame["r1_bill_cut_gbp"] - np.where(separate_co, co_gain, 0.0),
                        frame["ownership_share"] * frame["r1_bill_cut_gbp"] + premium_adjustment.to_numpy()), 0.0)
    frame["r1_retained_rent_gbp"] = np.where(
        is_r1, state_surplus(frame["ownership_share"], frame["rent_gbp"], overhead) + co_cost + co_floor
        - frame["r1_cut_borne_by_state_gbp"], 0.0)

    account = frame.groupby(["case_id", "group"], as_index=False).agg(
        assets=("asset_id", "size"),
        dispatch_mwh=("dispatch_mwh", "sum"),
        market_revenue_gbp=("market_revenue_gbp", "sum"),
        support_revenue_gbp=("support_revenue_gbp", "sum"),
        capacity_market_revenue_gbp=("capacity_market_revenue_gbp", "sum"),
        fuel_carbon_gbp=("fuel_carbon_gbp", "sum"),
        required_revenue_gbp=("annual_required_revenue_gbp", "sum"),
        rent_gbp=("rent_gbp", "sum"),
        public_surplus_gbp=("surplus_gbp", "sum"),
        public_surplus_saving_part_gbp=("saving_part_gbp", "sum"),
        public_surplus_rent_part_gbp=("rent_part_gbp", "sum"),
        public_surplus_saving_part_before_booking_gbp=("saving_part_before_booking_gbp", "sum"),
        booked_to_bill_gbp=("booked_to_bill_gbp", "sum"),
        cm_booking_floored_gbp=("cm_booking_floored_gbp", "sum"),
        tax_equivalent_gbp=("annual_tax_equivalent_gbp", "sum"),
        carbon_removal_cfd_revenue_gbp=("carbon_removal_cfd_revenue_gbp", "sum"),
        r1_bill_cut_gbp=("r1_bill_cut_gbp", "sum"),
        dpa_bill_cut_gbp=("dpa_bill_cut_gbp", "sum"),
        r1_retained_rent_gbp=("r1_retained_rent_gbp", "sum"),
        r1_cut_borne_by_state_gbp=("r1_cut_borne_by_state_gbp", "sum"),
        ldes_floor_topup_paid_gbp=("ldes_floor_topup_paid_gbp", "sum"),
        ldes_floor_topup_gbp=("ldes_floor_topup_gbp", "sum"),
        ldes_floor_topup_at_wacc_gbp=("ldes_floor_topup_at_wacc_gbp", "sum"),
        ldes_capacity_market_revenue_gbp=("ldes_capacity_market_revenue_gbp", "sum"),
        **({"rate_path_cut_gbp": ("rate_path_cut_gbp", "sum"),
            "rate_path_cut_borne_by_state_gbp": ("rate_path_cut_borne_by_state_gbp", "sum")}
           if "rate_path_cut_gbp" in frame.columns else {}),
    )
    lap("r1_cut_and_aggregate")
    account["in_account"] = ~account["group"].eq(GROUP_R1)
    account["recyclable"] = account["group"].isin([GROUP_R2, GROUP_MERCHANT])

    # D4: the shortfall a contract at required revenue would carry on merchant
    # new build (onshore, solar, hydro).  Only the share the state does not own
    # is left to book on the bill: the rest already comes back through the
    # surplus-recycling line, and an asset is booked once (house rule 5).
    new_build_merchant = (
        frame["group"].eq(GROUP_MERCHANT) & frame["finance_vintage"].astype(str).eq("new_build")
        if "finance_vintage" in frame.columns else pd.Series(False, index=frame.index)
    )
    frame["new_build_contract_shortfall_gbp"] = np.where(new_build_merchant, -frame["rent_gbp"], 0.0)
    frame["new_build_contract_bill_gbp"] = np.where(
        new_build_merchant, -frame["rent_gbp"] * (1.0 - frame["ownership_share"]), 0.0
    )

    summary = {}
    for case_id, rows in account.groupby("case_id"):
        rows = rows.set_index("group")
        get = lambda group, column: float(rows[column].get(group, 0.0))  # noqa: E731
        merchant = get(GROUP_MERCHANT, "public_surplus_gbp")
        r2 = get(GROUP_R2, "public_surplus_gbp")
        storage = get(GROUP_LDES, "public_surplus_gbp") + get(GROUP_BATTERY, "public_surplus_gbp")
        # Item B12: what the state keeps of its re-priced (R1) rows' rent after the strike cut.
        r1_retained = get(GROUP_R1, "r1_retained_rent_gbp")
        case_rows = frame.loc[frame["case_id"].eq(case_id)]
        levers = case_settings(case_rows)
        # FES 2050 (Y14): a year-gated lever the case sets off its default (state_deployment_share)
        # is part of its levers, so the bill and the balancing merge can tell the P2 shares
        # apart. settings_by_case carries them (cfd_repricing); no 2030 case sets one.
        levers.update({name: value for name, value in (settings_by_case.get(case_id) or {}).items()
                       if name in GATED_LEVERS})
        if LEVER_SURPLUS_RECYCLING in case_rows.columns:
            rule = levers[LEVER_SURPLUS_RECYCLING]
        else:
            rule = settings["recycling"].get(base_case(case_id), "no_assumption")
        if case_id == "private" or not spec_for_row(levers).is_public():
            rule = "not_applicable"
        funding_rule = (levers[LEVER_SHORTFALL_FUNDING] if LEVER_SHORTFALL_FUNDING in case_rows.columns
                        else settings["shortfall_funding"])
        # The R1 remainder is CfD surplus, so it follows R2 under every rule: retained
        # where R2 is retained, routed where R2 is routed (rebate and offset rules).
        # Under P1b (public_cfd_at_cost) R2 stays in the account, so it does too.
        wholesale_route, policy_route, retained = 0.0, 0.0, merchant + r2 + r1_retained
        if rule in ROUTE_ALL_NET:
            policy_route, retained = merchant + r2 + r1_retained, 0.0
        elif rule == "rebate_gross":
            # Consumer first: every positive balance to the bill, every negative one kept
            # by the state (funded by the Exchequer or levied, state_deficit_funding).
            policy_route = sum(max(v, 0.0) for v in (merchant, r2, r1_retained))
            retained = sum(min(v, 0.0) for v in (merchant, r2, r1_retained))
        elif rule == "cost_recovery_contracting":
            wholesale_route, retained = merchant, r2 + r1_retained
        elif rule == "public_cfd_at_cost":
            policy_route, retained = merchant, r2 + r1_retained
        if rule == "not_applicable":
            retained = 0.0
        routed = wholesale_route + policy_route
        shortfall = -min(routed, 0.0)
        funding = funding_rule if shortfall > 0 else None
        generation = case_rows.loc[case_rows["component_type"].eq("Generator")]
        axis_values = dict(levers)
        floor_basis = axis_values.get(AXIS_LDES_FLOOR_BASIS) or LDES_FLOOR_COST_OF_DEBT_LIKE
        floor_column = ("ldes_floor_topup_at_wacc_gbp" if floor_basis == LDES_FLOOR_TECHNOLOGY_WACC
                        else "ldes_floor_topup_gbp")
        summary[case_id] = {
            "recycling_rule": rule,
            "shortfall_funding_rule": funding_rule,
            "sensitivity_axes": axis_values,
            "levers": axis_values,
            # Item K3: the pay-as-bid ceiling, a bound on the wholesale line the
            # wholesale_pricing lever books under the private scenario alone.
            "pay_as_bid_ceiling_gbp": pay_as_bid_ceiling,
            "pay_as_bid_ceiling_applies": axis_values.get(LEVER_WHOLESALE_PRICING) == PRICING_PAY_AS_BID_CEILING,
            "public_surplus_gbp": {group: get(group, "public_surplus_gbp") for group in GROUPS if group in rows.index},
            "merchant_generation_surplus_gbp": merchant,
            "r2_cfd_surplus_gbp": r2,
            "capacity_market_route_surplus_not_recycled_gbp": get(GROUP_CM, "public_surplus_gbp"),
            "dispatchable_cfd_route_surplus_not_recycled_gbp": get(GROUP_DPA, "public_surplus_gbp"),
            "storage_surplus_not_recycled_gbp": storage,
            "wholesale_route_gbp": wholesale_route,
            "policy_route_gbp": policy_route,
            "routed_to_bill_gbp": routed,
            "shortfall_gbp": shortfall,
            "shortfall_funding": funding,
            "shortfall_on_bill_gbp": shortfall if funding == "levy_on_eligible_supply" else 0.0,
            "shortfall_on_exchequer_gbp": shortfall if funding == "exchequer" else 0.0,
            "retained_in_public_account_gbp": retained,
            "r1_retained_rent_gbp": r1_retained,
            "r1_bill_cut_gbp": get(GROUP_R1, "r1_bill_cut_gbp"),
            # F27 item 1: of which off the remaining_ccs line (NZT's DPA); the rest is the CfD line's.
            "dpa_bill_cut_gbp": get(GROUP_R1, "dpa_bill_cut_gbp"),
            # The part of the cut the state funds: pro rata on the financing basis, all of it at cost.
            "r1_cut_borne_by_state_gbp": get(GROUP_R1, "r1_cut_borne_by_state_gbp"),
            # Item B3: the Capacity Market and CCS amounts booked on the bill,
            # already taken off the owning groups' surplus above.
            "booked_to_bill_gbp": {group: get(group, "booked_to_bill_gbp") for group in (GROUP_CM, GROUP_BATTERY, GROUP_DPA)},
            # User decision 12: the Capacity Market booking the bill actually takes, and
            # the negative kept off it (floored at the route total, state_deficit_funding).
            "capacity_market_booked_gbp": (get(GROUP_CM, "booked_to_bill_gbp") + get(GROUP_BATTERY, "booked_to_bill_gbp")
                                           if settings["cm_sensitivity_enabled"] else None),
            "capacity_market_booking_floored_gbp": (get(GROUP_CM, "cm_booking_floored_gbp")
                                                    + get(GROUP_BATTERY, "cm_booking_floored_gbp")),
            "state_deficit_funding": str(levers.get(LEVER_STATE_DEFICIT_FUNDING,
                                                    DEFAULT_SETTINGS[LEVER_STATE_DEFICIT_FUNDING])),
            # Lever tax_equivalent and user decision 2, for the Exchequer ledger.
            "tax_equivalent_paid_gbp": float(rows["tax_equivalent_gbp"].sum()),
            # OW10: the state's share of its own new build, the share of Sizewell C's RAB it finances.
            "state_new_build_share": (sizewell_c_rab.state_new_build_share(case_rows)
                                      if rule != "not_applicable" else 0.0),
            "carbon_removal_cfd_revenue_gbp": float(rows["carbon_removal_cfd_revenue_gbp"].sum()),
            # Item B2: the LDES floor the bill pays, of which the state's share is in its storage surplus.
            "ldes_floor_topup_paid_to_owners_gbp": get(GROUP_LDES, "ldes_floor_topup_paid_gbp"),
            # The floor basis axis chooses which of the two the bill pays. Under
            # technology_wacc every case floors at its own cost of capital, so
            # the private case no longer gets a floor below what its capital
            # costs; it moves the private column too, hence SCOPE_ALL.
            "ldes_floor_topup_gbp": (
                get(GROUP_LDES, floor_column) if settings["ldes_enabled"] else 0.0
            ),
            "ldes_floor_topup_at_wacc_gbp": get(GROUP_LDES, "ldes_floor_topup_at_wacc_gbp") if settings["ldes_enabled"] else 0.0,
            # Netted from the floor: the Capacity Market line already pays it.
            "ldes_capacity_market_revenue_netted_gbp": get(GROUP_LDES, "ldes_capacity_market_revenue_gbp"),
            "ldes_floor_basis": floor_basis,
            "ldes_floor_real_rate": (
                None if floor_basis == LDES_FLOOR_TECHNOLOGY_WACC
                else (settings.get("ldes_floor_real_rate_by_case") or {}).get(base_case(case_id))
            ),
            "ownership_boundary": axis_values.get(LEVER_PRE_2025_CFD),
            "pre_2025_cfd_capacity": axis_values.get(LEVER_PRE_2025_CFD),
            "ldes_cap_and_floor_enabled": settings["ldes_enabled"],
            # Sensitivities on lines the model does not price: the state's own
            # financing saving on the plant the Capacity Market and the CCS levy pay.
            "capacity_market_sensitivity_gbp": (
                get(GROUP_CM, "public_surplus_saving_part_before_booking_gbp")
                + get(GROUP_BATTERY, "public_surplus_saving_part_before_booking_gbp")
                if settings["cm_sensitivity_enabled"] else None
            ),
            "ccs_levy_sensitivity_gbp": get(GROUP_DPA, "public_surplus_saving_part_before_booking_gbp"),
            # D4: what a contract at required revenue on merchant new build
            # would carry, and the part of it this case would put on the bill.
            "new_build_contract_shortfall_gbp": float(case_rows["new_build_contract_shortfall_gbp"].sum()),
            "new_build_contract_bill_gbp": float(case_rows["new_build_contract_bill_gbp"].sum()),
            # W2: generation required revenue plus fuel and carbon per MWh of generation.
            "w2_average_generation_cost_gbp": float(
                generation["annual_required_revenue_gbp"].sum() + generation["fuel_carbon_gbp"].sum()
            ),
            "w2_generation_mwh": float(generation["dispatch_mwh"].sum()),
        }
        for column in ("ancillary_services_revenue_gbp", "ancillary_voltage_stability_revenue_gbp",
                       "efw_gate_fee_revenue_gbp", "life_extension_top_up_gbp", "battery_revenue_top_up_gbp",
                       "battery_subhourly_revenue_gbp", "battery_revenue_top_up_subhourly_high_gbp"):
            # FES 2050 decisions 58 and 63: the support lines the register books (absent from every 2030
            # register); the 2050 bill's life_extension_operating_top_up and battery_revenue_top_up lines
            # read the private case's top-ups.
            if column in case_rows.columns:
                summary[case_id][column] = float(pd.to_numeric(case_rows[column], errors="coerce").fillna(0.0).sum())
        if BATTERY_TOP_UP_PRIVATE_BASIS in case_rows.columns:
            # F19 decision 5: the case's top-up against the private case's on the same register, by
            # who owns the row (the bill's battery_revenue_top_up_public_financing_adjustment and its
            # party flows; party_flows.register_components measures it the same way). Not against the
            # row's own private-basis column: a balancing group holds its support lines at the market
            # group's amounts (support_from_reference), so that column put private's own move between
            # the two months (+29.06 m on P2's private) into the co-owners' change (14 Oct 2026).
            # A row private does not carry falls back to the private-basis column.
            fallback = pd.to_numeric(case_rows[BATTERY_TOP_UP_PRIVATE_BASIS], errors="coerce").fillna(0.0)
            private_basis = (private_top_up.reindex(case_rows["asset_id"]).to_numpy()
                             if private_top_up is not None else fallback.to_numpy())
            private_basis = pd.Series(private_basis, index=case_rows.index).fillna(fallback)
            change = (pd.to_numeric(case_rows["battery_revenue_top_up_gbp"], errors="coerce").fillna(0.0)
                      - private_basis)
            own_share = pd.to_numeric(case_rows["ownership_share"], errors="coerce").fillna(0.0)
            summary[case_id]["battery_top_up_owner_cost"] = True
            summary[case_id]["battery_top_up_change_state_gbp"] = float((own_share * change).sum())
            summary[case_id]["battery_top_up_change_coowners_gbp"] = float(((1.0 - own_share) * change).sum())
        if "rate_path_cut_gbp" in rows.columns:
            # Decision 54: the rate path's strike cut and the state's part of it.
            summary[case_id]["rate_path_cut_gbp"] = float(rows["rate_path_cut_gbp"].sum())
            summary[case_id]["rate_path_cut_borne_by_state_gbp"] = float(rows["rate_path_cut_borne_by_state_gbp"].sum())
        if "rab_support_gbp" in case_rows.columns:
            # FES 2050 decision 41 (c): Sizewell C's RAB allowance in the register, which the
            # 2050 bill's sizewell_c_rab line recovers (only a register with RAB rows has it).
            rab = pd.to_numeric(case_rows["rab_support_gbp"], errors="coerce").fillna(0.0)
            summary[case_id]["rab_support_gbp"] = float(rab.sum())
            # Independent review (15 Nov 2026): the RAB the case's own state holds. Only then is the levy replaced
            # (the tariff's test); a case leaving Sizewell C private on its RAB keeps the OW10/D4 public-rate return.
            held = (case_rows["state_owned"].astype(str).str.lower().isin(["true", "1"])
                    if "state_owned" in case_rows.columns else pd.Series(True, index=case_rows.index))
            summary[case_id]["rab_support_state_owned_gbp"] = float(rab[held].sum())
        summary[case_id].update(exchequer_position(summary[case_id], rows, rule))
    lap("case_summaries")
    return account, summary


def _private_battery_top_up(register: pd.DataFrame, rows_for=None) -> pd.Series | None:
    """The private case's battery top-up by asset on this register (None when it has no private rows).

    From the batch when it holds private, else from ``rows_for("private")`` (streamed batches).
    """
    if register["case_id"].eq("private").any():
        rows = register.loc[register["case_id"].eq("private")]
    elif rows_for is not None:
        rows = rows_for("private")
    else:
        return None
    if rows is None or not len(rows) or "battery_revenue_top_up_gbp" not in rows.columns:
        return None
    rows = rows.drop_duplicates("asset_id")
    return pd.Series(pd.to_numeric(rows["battery_revenue_top_up_gbp"], errors="coerce").to_numpy(),
                     index=rows["asset_id"].to_numpy())


def exchequer_position(case: Mapping, rows: pd.DataFrame, rule: str) -> dict:
    """What the state's balances leave with the Exchequer, for its ledger (decision D2).

    Everything the account does not route to the bill stays with the state:
    the retained CfD and merchant surplus (by rule), the Capacity Market and
    dispatchable-CfD routes after the booked amounts (item B3) and the storage
    groups (never recycled).  A negative balance is a deficit the Exchequer
    funds; ``state_deficit_exchequer_funded_gbp`` is the sum of the negative
    group balances plus a routed shortfall the Exchequer carries
    (``shortfall_on_exchequer_gbp``), gross of any surplus elsewhere, which is
    reported as ``state_surplus_retained_gbp``.  ``state_retained_net_gbp`` is
    their difference less the Exchequer shortfall.  Zero for a case the state
    does not own.
    """
    def get(group):
        return float(rows["public_surplus_gbp"].get(group, 0.0)) if group in rows.index else 0.0

    if rule == "not_applicable":
        balances = {}
    else:
        retained_cfd = case["r2_cfd_surplus_gbp"] + case["r1_retained_rent_gbp"]
        retained_merchant = case["merchant_generation_surplus_gbp"]
        if rule in ROUTE_ALL_NET:
            retained_cfd, retained_merchant = 0.0, 0.0
        elif rule == "rebate_gross":
            retained_cfd = min(case["r2_cfd_surplus_gbp"], 0.0) + min(case["r1_retained_rent_gbp"], 0.0)
            retained_merchant = min(retained_merchant, 0.0)
        elif rule in ("cost_recovery_contracting", "public_cfd_at_cost"):
            retained_merchant = 0.0
        balances = {
            "retained_cfd_surplus": retained_cfd,
            "retained_merchant_surplus": retained_merchant,
            GROUP_CM: get(GROUP_CM), GROUP_DPA: get(GROUP_DPA),
            GROUP_LDES: get(GROUP_LDES), GROUP_BATTERY: get(GROUP_BATTERY),
        }
    deficits = -sum(min(v, 0.0) for v in balances.values())
    surpluses = sum(max(v, 0.0) for v in balances.values())
    exchequer_shortfall = float(case.get("shortfall_on_exchequer_gbp", 0.0))
    # User decision 3 (29 Sep 2026): state_deficit_funding.  Under exchequer (today)
    # every negative balance is the Exchequer's; under bill_levy the state recovers
    # them through a levy on eligible supply, and only a routed shortfall the
    # shortfall_funding lever sends to the Exchequer stays there.
    on_bill = deficits if case.get("state_deficit_funding") == DEFICITS_BILL_LEVY else 0.0
    return {
        "state_balances_left_with_state_gbp": balances,
        "state_surplus_retained_gbp": surpluses,
        "state_deficit_gross_gbp": deficits,
        "state_deficit_on_bill_gbp": on_bill,
        "state_deficit_exchequer_funded_gbp": deficits - on_bill + exchequer_shortfall,
        "state_retained_net_gbp": surpluses - (deficits - on_bill) - exchequer_shortfall,
    }
