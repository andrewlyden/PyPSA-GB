"""The Capacity Market design under public ownership (FES 2050 decision 73, user, 5 Oct 2026).

Until decision 73 every case paid the state's plant in the Capacity Market like any
participant (P1, P2: its income returned to bills through the surplus) or, under an
acquisition (P3), dropped the acquired fleet's agreements and kept a residual line for what
the state does not buy. Decision 73 makes the following the 2050 **central** design, with
today's treatment as the sensitivity ``@cm_market_participant`` (lever ``capacity_market_design``):

* **P2: the state's plant leaves the auction** (``state_out``). The state covers its own
  de-rated share of the reliability standard through its build plan, so the auction's
  target falls by the state's de-rated eligible capacity and the auction clears on what is
  left (DSR, V2G, interconnectors, private plant) at the residual clearing price. State
  plant earns no CM revenue; the bill's CM line is the residual auction's cost.
  **Co-owned rows (decisions 59 and 68)** leave pro rata: the state's share ``f`` of the
  row's de-rated MW is the state's (out of the auction), the co-owner's ``1 - f`` bids in
  and is paid the residual price, so a co-owned row's CM revenue is its co-owner's alone.
  **P1's minority stakes stay in**: a 30% stake in a privately run project is an investor's
  position, not state plant, so only rows the state holds whole leave (none in the HT50
  register, so P1 is unchanged).
* **P3: adequacy planning** (``adequacy_planning``). The state owns the fleet and plans its
  adequacy; a targeted contract covers what it does not own (DSR, V2G, interconnectors and
  any private or unmodelled capacity), priced off the residual auction clearing exactly as
  in P2. Existing (legacy) agreements are honoured and net out on acquired plant, as in
  2030; in 2050 the register holds none (the longest ends in 2044/45), so the line is the
  targeted contract alone. The acquired interconnector GB halves' share of the contract's
  interconnector payment is paid to the state itself and leaves the line, as the 2030
  tariff's GB-half rule does.
* **Private is unchanged**: the full auction. Every comparison against private is the
  design's effect plus ownership.

**How it is measured.** A finance step, not a solve: the 2050 auction model
(``scripts/market/capacity_market_2050.py``, task Y11, as decision 30 left it) is re-cleared
from its own builder outputs (``cm_fleet_2050.csv``, the classified fleet; and
``cm_calendar_2050_summary.json``, its requirement, settings and bid distributions) with the
state's share of each (technology, MR4 category) cell moved from eligible supply to capacity
outside the auction. The state's share of a cell is the capacity-weighted exit share of the
case's register rows that could hold an agreement (support not CfD, DPA, RAB or RO), read off
the register itself, so it follows every ownership lever. Each row's CM revenue is then its
non-exiting part at the residual auction's rate per de-rated MW of its category, relative to
the full auction's (the register's rates are the full auction's per-kW rates, so the ratio
carries them over without re-deriving participation). Nothing here moves required revenue.

**Who is paid.** The bill's CM line pays: the state's plant (nothing under the design), the
register's private plant and co-owners (their rows' CM revenue), V2G owners and residential
DSR (households: a transfer to consumers, kept visible in the party flows), I&C DSR and the
interconnectors, and plant the register does not carry (the auction's stock-flow fleet is not
the register, a declared Y11 limitation; the gap is paid outside the register, to investors).

Configured by ``public_ownership.capacity_market_design`` (:func:`settings`); absent, as in
every 2030 config, nothing here runs and every number is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.public_ownership.levers import (
    CM_DESIGN_MARKET_PARTICIPANT, CM_DESIGN_STATE_PLANNED, CM_DESIGN_STUDY, LEVER_CM_DESIGN,
    LEVER_NEW_BUILD_GENERATION, NEW_BUILD_STATE_STAKE, lever_default, spec_for_row,
)

CONFIG_KEY = "capacity_market_design"
CONFIG_KEYS = ("central", "fleet", "summary")
AUCTION_ALL, STATE_OUT, ADEQUACY_PLANNING = "auction_all", "state_out", "adequacy_planning"
DESIGNS = (AUCTION_ALL, STATE_OUT, ADEQUACY_PLANNING)
CENTRAL_VALUES = (CM_DESIGN_MARKET_PARTICIPANT, CM_DESIGN_STATE_PLANNED)
#: Register columns the design writes (2050 registers only).
COLUMN_STATE_ADJUSTMENT = "capacity_market_state_part_adjustment_gbp"
COLUMN_EXIT_SHARE = "capacity_market_state_exit_share"
#: The financing summary's column prefix for each case's design figures.
SUMMARY_PREFIX = "capacity_market_design_"
#: Auction rows (``capacity_market_2050.flexibility_rows``) whose owners are households.
HOUSEHOLD_CARRIERS = ("V2G", "DSR_residential")
PAYEE_CARRIERS = {"V2G": "v2g_owners_households", "DSR_residential": "dsr_residential",
                  "DSR_industrial_commercial": "dsr_industrial_commercial", "interconnector": "interconnectors"}
FLEET_COLUMNS = ("carrier", "build_year", "eligible_derated_mw", "non_cm_derated_mw", "category", "rate_key")
TOLERANCE_GBP = 1.0


class CapacityMarketDesignError(ValueError):
    """A capacity_market_design block or input the code does not implement."""


# ── settings ─────────────────────────────────────────────────────────────────

def settings(raw: Mapping | None, root: Path | str = ".") -> dict | None:
    """``public_ownership.capacity_market_design`` checked, with the auction's builder outputs loaded; None without it.

    Keys: ``central`` (what ``study_central`` means: ``state_planned``, decision 73's 2050
    central, or ``market_participant``), ``fleet`` (``cm_fleet_2050.csv``) and ``summary``
    (``cm_calendar_2050_summary.json``), both written by ``build_capacity_market_year.py`` (year 2050).
    """
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise CapacityMarketDesignError(f"public_ownership.{CONFIG_KEY} must be a mapping")
    unknown = sorted(set(raw) - set(CONFIG_KEYS))
    if unknown:
        raise CapacityMarketDesignError(f"public_ownership.{CONFIG_KEY} has unknown keys {unknown}; "
                                        f"supported: {list(CONFIG_KEYS)}")
    central = str(raw.get("central", CM_DESIGN_MARKET_PARTICIPANT))
    if central not in CENTRAL_VALUES:
        raise CapacityMarketDesignError(f"public_ownership.{CONFIG_KEY}.central={central!r}; supported: "
                                        f"{list(CENTRAL_VALUES)}")
    for key in ("fleet", "summary"):
        if not raw.get(key):
            raise CapacityMarketDesignError(f"public_ownership.{CONFIG_KEY}.{key} must name the 2050 auction's "
                                            "builder output (build_capacity_market_year.py)")
    fleet_path, summary_path = Path(root) / str(raw["fleet"]), Path(root) / str(raw["summary"])
    fleet = pd.read_csv(fleet_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return from_inputs(fleet, summary, central=central, fleet_file=str(raw["fleet"]),
                       summary_file=str(raw["summary"]))


def from_inputs(fleet: pd.DataFrame, summary: Mapping, *, central: str = CM_DESIGN_STATE_PLANNED,
                rebasing_factor: float | None = None, fleet_file: str | None = None,
                summary_file: str | None = None) -> dict:
    """The design options from an auction fleet and summary already in memory (tests, :func:`settings`)."""
    missing = sorted(set(FLEET_COLUMNS) - set(fleet.columns))
    if missing:
        raise CapacityMarketDesignError(f"the auction fleet lacks columns {missing} (cm_fleet_2050.csv)")
    for key in ("requirement", "settings", "bid_distributions", "indexation_2025_26_to_2025", "model_year"):
        if key not in summary:
            raise CapacityMarketDesignError(f"the auction summary lacks {key!r} (cm_calendar_2050_summary.json)")
    options = {"central": central, "fleet": fleet.copy(), "summary": dict(summary),
               "rebasing_factor": rebasing_factor, "fleet_file": fleet_file, "summary_file": summary_file}
    options["full"] = clear(options["fleet"], options["summary"])
    reported = summary.get(f"calendar_{int(summary['model_year'])}_cost_gbp")
    if reported is not None:
        # 2040 (11 Nov 2026): a year the register still reaches carries a contract book beside the auction
        # (build_capacity_market_year.py); the re-clear reproduces the auction part. 2050's book is 0.
        reported = float(reported) - float(summary.get("contract_book_gbp") or 0.0)
    if reported is not None and abs(options["full"]["cost_gbp"] - float(reported)) > TOLERANCE_GBP:
        raise CapacityMarketDesignError(
            f"re-clearing the auction's own fleet gives GBP {options['full']['cost_gbp']:,.0f}, its summary "
            f"GBP {float(reported):,.0f}: the fleet and the summary are not one build")
    return options


def resolve(levers: Mapping, options: Mapping | None) -> str:
    """The design a case takes: ``auction_all`` (private, or the market-participant treatment), ``state_out``
    (a public market-bill case) or ``adequacy_planning`` (a case that acquires the legacy fleet)."""
    if not options:
        return AUCTION_ALL
    value = str(levers.get(LEVER_CM_DESIGN, lever_default(LEVER_CM_DESIGN)))
    if value == CM_DESIGN_STUDY:
        value = str(options["central"])
    spec = spec_for_row(levers)
    if value == CM_DESIGN_MARKET_PARTICIPANT or not spec.is_public():
        return AUCTION_ALL
    if value != CM_DESIGN_STATE_PLANNED:
        raise CapacityMarketDesignError(f"capacity_market_design={value!r} is not implemented")
    return ADEQUACY_PLANNING if spec.acquires_legacy() else STATE_OUT


# ── the auction ──────────────────────────────────────────────────────────────

def _cm50():
    from scripts.market import capacity_market_2050 as cm50

    return cm50


def clear(fleet: pd.DataFrame, summary: Mapping) -> dict:
    """Clear the 2050 auction on a classified fleet; the cost, the rate per de-rated MW by category, the payees.

    The arithmetic is ``capacity_market_2050.clear_2050`` and ``revenue_by_carrier``'s: each
    category's award (T-1 pro rata) split across its rows by eligible de-rated capacity, at the
    clearing price in report money (2025).
    """
    cm50 = _cm50()
    auction_settings = cm50.Settings2050(**dict(summary["settings"]))
    distributions = cm50.read_distributions({"distributions": summary["bid_distributions"]})
    cleared = cm50.clear_2050(fleet, summary["requirement"], distributions, auction_settings)
    indexation = float(summary["indexation_2025_26_to_2025"])
    price = float(cleared["price_gbp_per_kw_year_auction_money"]) * indexation
    cost = (float(cleared["awarded_mw"]) + float(cleared["t1_mw"])) * 1e3 * price
    rows = fleet.loc[pd.to_numeric(fleet["eligible_derated_mw"], errors="coerce").fillna(0.0) > 0]
    by_cat = rows.groupby("category")["eligible_derated_mw"].sum()
    award = pd.Series(cleared["awarded_by_category_mw"], dtype=float).reindex(by_cat.index).fillna(0.0)
    t4_total = float(sum(cleared["awarded_by_category_mw"].values()))
    t1_scale = 1.0 + (float(cleared["t1_mw"]) / t4_total if t4_total > 0 else 0.0)
    rate = (award / by_cat.where(by_cat > 0) * t1_scale * price * 1e3).fillna(0.0)
    payment = rows["eligible_derated_mw"].astype(float) * rows["category"].map(rate).fillna(0.0)
    carrier = rows["carrier"].astype(str)
    payees = {label: float(payment[carrier.eq(c)].sum()) for c, label in PAYEE_CARRIERS.items()}
    payees["plant"] = float(payment[~carrier.isin(PAYEE_CARRIERS)].sum())
    if abs(sum(payees.values()) - cost) > max(TOLERANCE_GBP, 1e-9 * cost):
        raise CapacityMarketDesignError(f"the auction's payments ({sum(payees.values()):,.0f}) do not add up to "
                                        f"its cost ({cost:,.0f})")
    return {"cleared": cleared, "price_gbp_per_kw_year_2025": price, "cost_gbp": cost,
            "rate_gbp_per_mw_by_category": {str(k): float(v) for k, v in rate.items()},
            "payments_by_payee_gbp": payees,
            "household_payment_gbp": float(sum(payees[PAYEE_CARRIERS[c]] for c in HOUSEHOLD_CARRIERS))}


def register_cells(rows: pd.DataFrame, model_year: int, lead_years: int) -> pd.DataFrame:
    """Each register row's auction cell: ``cm_carrier`` (a ``capacity_market_2050.CARRIERS`` key, or None),
    ``category`` (MR4's existing, new_build or interconnector) and ``eligible`` (could hold an agreement)."""
    cm50 = _cm50()
    mapped = rows["carrier"].astype(str).map(lambda c: cm50.REGISTER_CARRIER_MAP.get(c, c))
    cm_carrier = mapped.where(mapped.isin(list(cm50.CARRIERS)))
    support = (rows["support_type"].astype(str) if "support_type" in rows.columns
               else pd.Series("", index=rows.index))
    eligible = cm_carrier.notna() & ~support.isin(sorted(cm50.NON_CM_SCHEMES))
    build = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in ("build_year", "commissioning_year"):
        if column in rows.columns:
            build = build.fillna(pd.to_numeric(rows[column], errors="coerce"))
    klass = cm_carrier.map(lambda c: cm50.CARRIERS[c][1] if isinstance(c, str) else None)
    category = np.where(klass.eq("interconnector"), "interconnector",
                        np.where(build > int(model_year) - int(lead_years), "new_build", "existing"))
    return pd.DataFrame({"cm_carrier": cm_carrier, "category": category, "eligible": eligible.to_numpy()},
                        index=rows.index)


def exit_shares(rows: pd.DataFrame, levers: Mapping) -> pd.Series:
    """The share of each row the state takes out of the auction.

    Pro rata on its ownership share (P2, and every acquisition case; a co-owned row's
    ``f``), except a state stake (P1, ``new_build_generation: state_stake``): a minority
    stake in a privately run project stays in, so only rows the state holds whole leave.
    """
    own = pd.to_numeric(rows["ownership_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if str(levers.get(LEVER_NEW_BUILD_GENERATION, "")) == NEW_BUILD_STATE_STAKE:
        return pd.Series(np.where(own >= 1.0 - 1e-9, 1.0, 0.0), index=rows.index)
    return own


def state_shares_by_cell(rows: pd.DataFrame, exit_share: pd.Series, cells: pd.DataFrame) -> dict:
    """The state's capacity-weighted exit share of each (auction carrier, category) cell, and by carrier."""
    capacity = pd.to_numeric(rows["capacity_mw"], errors="coerce").fillna(0.0)
    frame = pd.DataFrame({"cm_carrier": cells["cm_carrier"], "category": cells["category"],
                          "capacity": capacity, "held": capacity * exit_share}).loc[cells["eligible"]]
    by_cell = frame.groupby(["cm_carrier", "category"])[["capacity", "held"]].sum()
    by_carrier = frame.groupby("cm_carrier")[["capacity", "held"]].sum()
    cell = (by_cell["held"] / by_cell["capacity"].where(by_cell["capacity"] > 0)).dropna()
    carrier = (by_carrier["held"] / by_carrier["capacity"].where(by_carrier["capacity"] > 0)).dropna()
    return {"cell": {k: float(v) for k, v in cell.items()}, "carrier": {k: float(v) for k, v in carrier.items()}}


def apply_exit(fleet: pd.DataFrame, shares: Mapping) -> pd.DataFrame:
    """The auction fleet with the state's share of each cell moved from eligible supply to outside the auction.

    A cell the register does not hold takes its carrier's share, and a carrier the register
    does not hold (DSR, V2G) stays whole; the target then falls by exactly what leaves
    (``clear_2050`` procures the requirement less everything outside the auction).
    """
    out = fleet.copy()
    share = [shares["cell"].get((str(c), str(k)), shares["carrier"].get(str(c), 0.0))
             for c, k in zip(out["carrier"], out["category"])]
    eligible = pd.to_numeric(out["eligible_derated_mw"], errors="coerce").fillna(0.0)
    state = eligible * np.asarray(share, dtype=float)
    out["state_planned_derated_mw"] = state
    out["eligible_derated_mw"] = eligible - state
    out["non_cm_derated_mw"] = pd.to_numeric(out["non_cm_derated_mw"], errors="coerce").fillna(0.0) + state
    return out


# ── the register booking ─────────────────────────────────────────────────────

def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def case_design(rows: pd.DataFrame, levers: Mapping, options: Mapping) -> tuple[dict, pd.Series, pd.Series]:
    """One case's auction under its design: ``(result, exit share by row, residual-to-full rate ratio by row)``.

    ``result`` is in report money (2025) and MW; :func:`summary_columns` converts it.
    """
    design = resolve(levers, options)
    full = options["full"]
    summary = options["summary"]
    zero = pd.Series(0.0, index=rows.index)
    cm = _num(rows, "capacity_market_revenue_gbp")
    result = {"design": design, "full_price_gbp_per_kw_year_2025": full["price_gbp_per_kw_year_2025"],
              "full_auction_cost_gbp_2025": full["cost_gbp"]}
    if design == AUCTION_ALL:
        result.update(_auction_result(full, full, state_mw=0.0))
        result["register_revenue_gbp"] = float(cm.sum())
        result["state_forgone_register_gbp"] = 0.0
        return result, zero, pd.Series(1.0, index=rows.index)
    lead = int(dict(summary["settings"]).get("new_build_lead_years", 4))
    cells = register_cells(rows, int(summary["model_year"]), lead)
    exit_share = exit_shares(rows, levers).where(cells["eligible"], 0.0)
    shares = state_shares_by_cell(rows, exit_share, cells)
    classified = apply_exit(options["fleet"], shares)
    case = clear(classified, summary)
    state_mw = float(classified["state_planned_derated_mw"].sum())
    result.update(_auction_result(case, full, state_mw=state_mw))
    # The state's de-rated capacity valued at the full auction's rate: what it forgoes as a participant.
    full_rate = classified["category"].map(full["rate_gbp_per_mw_by_category"]).fillna(0.0)
    result["state_forgone_auction_gbp_2025"] = float((classified["state_planned_derated_mw"] * full_rate).sum())
    result["state_shares_by_cell"] = {f"{c}|{k}": v for (c, k), v in shares["cell"].items()}
    ratio = {k: (case["rate_gbp_per_mw_by_category"].get(k, 0.0) / v if v > 0 else 0.0)
             for k, v in full["rate_gbp_per_mw_by_category"].items()}
    result["rate_ratio_by_category"] = ratio
    row_ratio = pd.Series(cells["category"], index=rows.index).map(ratio).fillna(0.0)
    own = pd.to_numeric(rows["ownership_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    new = cm * (1.0 - exit_share) * row_ratio
    result["register_revenue_gbp"] = float(new.sum())
    # What the state's rows earn in the CM falls from its share of the full revenue to its in-auction share.
    result["state_forgone_register_gbp"] = float((own * cm - (own - exit_share).clip(lower=0.0) * row_ratio * cm).sum())
    return result, exit_share, row_ratio


def _auction_result(case: Mapping, full: Mapping, *, state_mw: float) -> dict:
    cleared = case["cleared"]
    return {
        "price_gbp_per_kw_year_2025": case["price_gbp_per_kw_year_2025"],
        "line_cost_gbp_2025": case["cost_gbp"],
        "target_mw": float(cleared["target_mw"]), "t1_mw": float(cleared["t1_mw"]),
        "awarded_mw": float(cleared["awarded_mw"]) + float(cleared["t1_mw"]),
        "supply_total_mw": float(cleared["supply_total_mw"]), "shortfall": bool(cleared["shortfall"]),
        "state_exit_derated_mw": float(state_mw),
        "household_payment_gbp_2025": case["household_payment_gbp"],
        "household_payment_full_gbp_2025": full["household_payment_gbp"],
        "v2g_payment_gbp_2025": case["payments_by_payee_gbp"]["v2g_owners_households"],
        "interconnector_payment_gbp_2025": case["payments_by_payee_gbp"]["interconnectors"],
        "payments_by_payee_gbp_2025": dict(case["payments_by_payee_gbp"]),
    }


def book(frame: pd.DataFrame, options: Mapping | None, specs: Sequence) -> tuple[pd.DataFrame, dict]:
    """Book each case's design to its register rows: CM revenue, producer rent and the state's surplus.

    Per row, with ``f`` the state's ownership share, ``e`` its exit share (``f`` pro rata, P1's
    stakes 0), ``cm`` the full auction's revenue and ``r`` the residual-to-full rate ratio of the
    row's category: the row earns ``cm' = (1 - e) x r x cm``, the co-owner's in-auction part. The
    rent moves by ``cm' - cm``; the state's surplus by its share of that plus
    ``capacity_market_state_part_adjustment_gbp = -e (1 - f) r cm``, so the state keeps exactly
    its in-auction share ``(f - e) r cm`` (zero under P2). The LDES floor, booked after this,
    nets the new CM revenue. Returns ``(frame, {case_id: result})``; unchanged without options.
    """
    if not options or frame.empty:
        return frame, {}
    frame = frame.copy()
    cm_all = _num(frame, "capacity_market_revenue_gbp")
    frame["capacity_market_revenue_gbp"] = cm_all
    adjustment = pd.Series(0.0, index=frame.index)
    exits = pd.Series(0.0, index=frame.index)
    results = {}
    has_rent = frame["producer_rent_gbp"].notna() if "producer_rent_gbp" in frame.columns \
        else pd.Series(False, index=frame.index)
    for spec in specs:
        mask = frame["case_id"].eq(spec.case_id)
        if not mask.any():
            continue
        rows = frame.loc[mask]
        result, exit_share, ratio = case_design(rows, spec.settings, options)
        results[spec.case_id] = result
        if result["design"] == AUCTION_ALL:
            continue
        cm = cm_all[mask]
        own = pd.to_numeric(rows["ownership_share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        new = cm * (1.0 - exit_share) * ratio
        delta = new - cm
        state_part = -exit_share * (1.0 - own) * ratio * cm
        frame.loc[mask, "capacity_market_revenue_gbp"] = new
        rent = mask & has_rent
        frame.loc[rent, "producer_rent_gbp"] = pd.to_numeric(frame.loc[rent, "producer_rent_gbp"]) + delta[rent[mask]]
        if "public_surplus_before_recycling_gbp" in frame.columns:
            surplus = rent & frame["public_surplus_before_recycling_gbp"].notna()
            frame.loc[surplus, "public_surplus_before_recycling_gbp"] = (
                pd.to_numeric(frame.loc[surplus, "public_surplus_before_recycling_gbp"])
                + (own * delta + state_part)[surplus[mask]])
        adjustment.loc[mask] = state_part
        exits.loc[mask] = exit_share
    frame[COLUMN_STATE_ADJUSTMENT] = adjustment
    frame[COLUMN_EXIT_SHARE] = exits
    return frame, results


def summary_columns(results: Mapping[str, Mapping], rebasing_factor: float | None) -> pd.DataFrame:
    """The financing summary's design columns per case (``capacity_market_design_*``), model money where ``_gbp``.

    Report-money (2025) amounts are divided by the bill's rebasing factor onto the register's
    price year, as ``calculate_financing_scenarios.capacity_market_rates_in_price_year`` does.
    """
    factor = float(rebasing_factor or 1.0)
    rows = []
    for case_id, r in results.items():
        row = {"case_id": case_id, SUMMARY_PREFIX + "design": r["design"]}
        for key, value in r.items():
            if key in ("design", "payments_by_payee_gbp_2025", "state_shares_by_cell", "rate_ratio_by_category"):
                continue
            if key.endswith("_gbp_2025"):
                row[SUMMARY_PREFIX + key[:-len("_2025")]] = float(value) / factor
            else:
                row[SUMMARY_PREFIX + key] = value
        row[SUMMARY_PREFIX + "rate_ratio_by_category"] = json.dumps(r.get("rate_ratio_by_category") or {},
                                                                    sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


# ── the bill and the identity ────────────────────────────────────────────────

def case_moves(financing_summary: pd.DataFrame, case_id: str, private_id: str = "private") -> dict | None:
    """A case's CM design moves against private, model money (system GBP): None without the design columns.

    ``line_change_gbp`` (the bill's CM line), ``register_change_gbp`` (the register rows' CM
    revenue, the state's and the co-owners'), ``household_change_gbp`` (V2G owners and
    residential DSR, households) and ``outside_register_other_change_gbp`` (I&C DSR,
    interconnectors and plant the register does not carry: the rest, so the four add up).
    """
    column = SUMMARY_PREFIX + "line_cost_gbp"
    if financing_summary is None or column not in financing_summary.columns:
        return None
    table = financing_summary.set_index("case_id")
    if case_id not in table.index or private_id not in table.index or pd.isna(table.at[case_id, column]):
        return None
    get = lambda c, key: float(table.at[c, SUMMARY_PREFIX + key])  # noqa: E731
    line = get(case_id, "line_cost_gbp") - get(private_id, "line_cost_gbp")
    register = get(case_id, "register_revenue_gbp") - get(private_id, "register_revenue_gbp")
    household = get(case_id, "household_payment_gbp") - get(private_id, "household_payment_gbp")
    return {"design": str(table.at[case_id, SUMMARY_PREFIX + "design"]),
            "line_cost_gbp": get(case_id, "line_cost_gbp"), "line_cost_private_gbp": get(private_id, "line_cost_gbp"),
            "line_change_gbp": line, "register_change_gbp": register, "household_change_gbp": household,
            "outside_register_other_change_gbp": line - register - household,
            "price_gbp_per_kw_year_2025": get(case_id, "price_gbp_per_kw_year_2025"),
            "price_private_gbp_per_kw_year_2025": get(private_id, "price_gbp_per_kw_year_2025"),
            "state_forgone_auction_gbp": get(case_id, "state_forgone_auction_gbp")
            if SUMMARY_PREFIX + "state_forgone_auction_gbp" in table.columns
            and not pd.isna(table.at[case_id, SUMMARY_PREFIX + "state_forgone_auction_gbp"]) else 0.0,
            "state_forgone_register_gbp": get(case_id, "state_forgone_register_gbp")}
