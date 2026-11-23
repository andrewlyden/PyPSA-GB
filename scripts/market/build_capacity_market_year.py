"""The Capacity Market of any long-range model year, 2031 to 2050 (FES 2040, 11 Nov 2026; 2050 from 14 Nov 2026).

User, 14 Nov 2026 (everything through the generic builders): this builder also builds calendar 2050, which no
agreement in the NESO register reaches (its contract book is empty, so the line is the auction alone); it
reproduces the former 2050-only builder's outputs exactly (cost, every case, price, payees, revenue table and
fleet), and ``build_capacity_market_2050.py`` is now its 2050 entry point. In 2040 the 15-year agreements of the 2025/26 to 2029/30 T-4 auctions are still live
(about 10.5 GW de-rated in 2039/40 and 8.8 GW in 2040/41), so the line has two parts:

* **Contract book.** The register's agreements live in each delivery year the calendar year
  spans (``capacity_market.contract_book``: obligation MW x clearing price, CPI-indexed to 2025),
  weighted half each (the supplier levy runs November to February). Each agreement is matched to
  the FES stock-flow fleet's carrier (:data:`BOOK_CARRIER`); booked capacity leaves the auction's
  eligible supply and counts against its target (``non_cm_derated_mw``, as live support does).
  Where the fleet holds less eligible capacity of a carrier than is booked, the excess agreement
  is taken as terminated: FES retires the plant (declared; reported as ``unmatched``).
* **Auction.** The Y11 model (``capacity_market_2050``: requirement on the 2030/31 ratio, MR4 bid
  distributions, NESO 2030/31 de-rating) clears the residual on the model year's fleet. Agreements
  of the unheld auctions (2030/31 onward) are taken at the model year's clearing price (steady
  state, as 2050 declares for its earlier auctions).

V2G bids in as DSR centrally (FES 2050 decision 30), with the unpaid sensitivity, as 2050. The
three outputs have the 2050 files' shape (``cm_calendar_<Y>_summary.json`` with
``contract_book_gbp`` and ``calendar_<Y>_cost_gbp`` = book + auction,
``cm_revenue_by_technology_<Y>.csv``, ``cm_fleet_<Y>.csv``: the auction's classified fleet, which
``capacity_market_design`` re-clears; its consistency check takes the book out).


Usage::

    python scripts/market/build_capacity_market_year.py --model-year 2040 --out resources/market/capacity_market
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.market import capacity_market as cm  # noqa: E402
from scripts.market import capacity_market_2050 as cm50  # noqa: E402
from scripts.market.build_capacity_market_2050 import (  # noqa: E402
    CENTRAL_V2G, CONSUMER_FLEXIBILITY, UNPAID_CASE, payments_by_row)
from scripts.public_ownership.model_year import fiscal_year_label  # noqa: E402

#: Register technology (first token of "CMU Technology") -> the stock-flow fleet carrier that holds it.
#: Storage: refurbishing units are the pumped-storage stations (Dinorwig, Ffestiniog, Cruachan, Foyers);
#: every new-build storage agreement is taken as battery (declared).
BOOK_CARRIER = {
    "Combined Cycle Gas Turbine (CCGT)": "gas", "Open Cycle Gas Turbine (OCGT)": "gas",
    "Reciprocating engines": "gas", "Combined Heat and Power (CHP)": "CHP", "Energy from Waste": "waste_to_energy",
    "Hydro": "large_hydro", "Offshore Wind": "wind_offshore", "Onshore Wind": "wind_onshore",
    "Solar Photovoltaic": "solar_pv", "Nuclear": "nuclear", "Biomass": "biomass", "Oil-fired steam generators": "oil",
    "DSR": "DSR_industrial_commercial", "Interconnector": "interconnector",
}


def book_carrier(technology: str, category: str) -> str | None:
    """The fleet carrier an agreement's plant is (None: not mapped)."""
    tech = str(technology)
    if tech.startswith("Storage"):
        return "Pumped Storage Hydroelectricity" if str(category) == "refurbishing" else "Battery"
    return BOOK_CARRIER.get(tech)


def contract_book_year(model_year: int, cpi: pd.Series, agreements: pd.DataFrame | None = None) -> pd.DataFrame:
    """Register agreements live in calendar ``model_year``, weighted by delivery year, by fleet carrier.

    Returns:
        One row per agreement and delivery year: ``carrier``, ``derated_mw`` and ``payment_gbp``
        (2025 money), each times the delivery year's calendar weight, and ``delivery_year``.
    """
    agreements = cm.read_agreements() if agreements is None else agreements
    rows = []
    for delivery_year, weight in cm50.calendar_weights(model_year).items():
        live = cm.contract_book(agreements, delivery_year, cpi)
        if live.empty:
            continue
        live = live.assign(carrier=[book_carrier(t, c) for t, c in zip(live["technology"], live["category"])],
                           weight=weight)
        rows.append(live)
    if not rows:
        return pd.DataFrame(columns=["agreement_id", "technology", "category", "carrier", "delivery_year", "weight",
                                     "derated_mw", "payment_gbp"])
    book = pd.concat(rows, ignore_index=True)
    unmapped = sorted(book.loc[book["carrier"].isna(), "technology"].unique())
    if unmapped:
        raise ValueError(f"contract book technologies with no fleet carrier: {unmapped}")
    book["derated_mw"] = book["mw"] * book["weight"]
    book["payment_gbp"] = book["payment_gbp"] * book["weight"]
    return book[["agreement_id", "auction", "technology", "category", "carrier", "delivery_year", "weight",
                 "derated_mw", "payment_gbp"]]


def apply_book(classified: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move each carrier's booked de-rated capacity from eligible supply to outside the auction.

    The booked MW of a carrier is taken from its eligible rows pro rata; where the fleet holds less
    eligible capacity than is booked, the match is capped and the rest of the book is unmatched.

    Returns:
        ``(classified with cm_agreement_derated_mw, book by carrier with matched_share)``.
    """
    out = classified.copy()
    out["cm_agreement_derated_mw"] = 0.0
    by_carrier = book.groupby("carrier").agg(booked_derated_mw=("derated_mw", "sum"),
                                             booked_payment_gbp=("payment_gbp", "sum"))
    matched = []
    for carrier, row in by_carrier.iterrows():
        mask = out["carrier"].astype(str).eq(carrier) & (out["eligible_derated_mw"] > 0)
        available = float(out.loc[mask, "eligible_derated_mw"].sum())
        take = min(float(row["booked_derated_mw"]), available)
        if take > 0:
            share = take / available
            moved = out.loc[mask, "eligible_derated_mw"] * share
            out.loc[mask, "cm_agreement_derated_mw"] = moved
            out.loc[mask, "eligible_derated_mw"] = out.loc[mask, "eligible_derated_mw"] - moved
            out.loc[mask, "non_cm_derated_mw"] = out.loc[mask, "non_cm_derated_mw"] + moved
        matched.append({"carrier": carrier, "fleet_eligible_derated_mw": available, "matched_derated_mw": take,
                        "matched_share": take / float(row["booked_derated_mw"]) if row["booked_derated_mw"] > 0 else 0.0})
    # A year past every register agreement (2045: the 15-year terms end by 2044/45) has an empty book.
    columns = ["carrier", "fleet_eligible_derated_mw", "matched_derated_mw", "matched_share"]
    by_carrier = by_carrier.join(pd.DataFrame(matched, columns=columns).set_index("carrier"))
    by_carrier["matched_payment_gbp"] = by_carrier["booked_payment_gbp"] * by_carrier["matched_share"]
    return out, by_carrier.reset_index()


def run_year(model: dict, cpi: pd.Series, model_year: int, *, settings: cm50.Settings2050,
             book: pd.DataFrame | None = None) -> dict:
    """The calendar Capacity Market of ``model_year``: contract book plus the residual auction."""
    trajectory = cm50.read_fleet_trajectory()
    calibration = cm50.read_calibration()
    derating = cm50.read_derating()
    fleet = cm50.fleet_2050(trajectory, model_year)
    distributions = cm50.read_distributions(model)
    peak = float(cm50._series(trajectory, "acs_peak_mw", "demand")[model_year])
    indexation = float(cpi[cm50.PRICE_YEAR] / cm.cpi_financial_year(cpi, 2025))
    book = contract_book_year(model_year, cpi) if book is None else book

    def one(s: cm50.Settings2050):
        classified = pd.concat([cm50.classify_fleet(fleet, derating, s, model_year),
                                cm50.flexibility_rows(trajectory, derating, s, model_year)], ignore_index=True)
        classified, by_carrier = apply_book(classified, book)
        requirement = cm50.requirement_mw(calibration, peak, s.requirement_rule)
        cleared = cm50.clear_2050(classified, requirement, distributions, s)
        auction_cost = ((cleared["awarded_mw"] + cleared["t1_mw"]) * 1e3
                        * cleared["price_gbp_per_kw_year_auction_money"] * indexation)
        return classified, by_carrier, requirement, cleared, auction_cost

    classified, by_carrier, requirement, cleared, auction_cost = one(settings)
    book_cost = float(by_carrier["matched_payment_gbp"].sum())
    total = auction_cost + book_cost
    revenue = revenue_table(classified, cleared, by_carrier, indexation, model_year)
    key = f"calendar_{model_year}_cost_gbp"
    if not np.isclose(float(revenue[key.replace("_cost_", "_revenue_")].sum()), total, rtol=1e-9, atol=1.0):
        raise ValueError("CM revenue by carrier does not add up to the calendar cost")
    cases = {}
    for label, change in {**cm50.CASES, UNPAID_CASE: {"v2g": "outside_cm"}}.items():
        _, bc, req, clr, cst = one(replace(settings, **change))
        cost = float(cst) + float(bc["matched_payment_gbp"].sum())
        cases[label] = {"changes": change, "requirement_mw": round(req["requirement_mw"], 1),
                        "target_mw": round(clr["target_mw"], 1), "supply_total_mw": round(clr["supply_total_mw"], 1),
                        "price_gbp_per_kw_year_auction_money": round(clr["price_gbp_per_kw_year_auction_money"], 2),
                        "awarded_mw": round(clr["awarded_mw"], 1), "shortfall": clr["shortfall"], key: cost}
    rows = payments_by_row(classified, cleared, indexation)
    carrier = rows["carrier"].astype(str)
    payees = {label: float(rows.loc[carrier.eq(c), "payment_gbp"].sum()) for c, label in CONSUMER_FLEXIBILITY.items()}
    payees["register_plant_and_interconnectors"] = float(rows.loc[~carrier.isin(CONSUMER_FLEXIBILITY), "payment_gbp"].sum())
    # The booked DSR is industrial and commercial; every other agreement is register plant.
    dsr_book = float(by_carrier.loc[by_carrier["carrier"].eq("DSR_industrial_commercial"), "matched_payment_gbp"].sum())
    payees["dsr_industrial_commercial"] += dsr_book
    payees["register_plant_and_interconnectors"] += book_cost - dsr_book
    if abs(sum(payees.values()) - total) > 1.0:
        raise ValueError("the payee split does not add up to the calendar cost")
    v2g = rows.loc[carrier.eq("V2G")]
    payees["v2g_awarded_derated_mw"] = float(v2g["awarded_derated_mw"].sum())
    payees["v2g_eligible_derated_mw"] = float(v2g["eligible_derated_mw"].sum())
    weights = cm50.calendar_weights(model_year)
    summary = {
        "model_year": model_year, "price_year": cm50.PRICE_YEAR, key: float(total),
        "auction_cost_gbp": float(auction_cost),
        "contract_book_gbp": book_cost,
        "contract_book": {
            "rule": ("register agreements live in each delivery year the calendar year spans, weighted "
                     f"{weights}; matched to the FES stock-flow fleet by carrier, the unmatched part taken as "
                     "terminated (FES retires the plant)"),
            "register": str(cm.REGISTER_CMU),
            "by_delivery_year_derated_mw": {dy: float(book.loc[book["delivery_year"].eq(dy), "derated_mw"].sum() / w)
                                            for dy, w in weights.items()},
            "by_carrier": by_carrier.round(3).to_dict(orient="records"),
            "booked_payment_gbp": float(by_carrier["booked_payment_gbp"].sum()),
            "matched_payment_gbp": book_cost,
            "unmatched_payment_gbp": float(by_carrier["booked_payment_gbp"].sum()) - book_cost,
        },
        "delivery_years": {dy: {"weight_in_calendar": w} for dy, w in weights.items()},
        "requirement": requirement, "auction": cleared,
        "price_gbp_per_kw_year_2025": cleared["price_gbp_per_kw_year_auction_money"] * indexation,
        "indexation_2025_26_to_2025": indexation,
        "settings": {k: v for k, v in settings.__dict__.items() if k != "notes"},
        f"calendar_{model_year}_cost_by_case_gbp": {k: v[key] for k, v in cases.items()},
        "cases": cases,
        "bid_distributions": model["distributions"],
        "payments_by_payee_gbp": payees,
        "derating_source": f"{cm50.DERATING_FILE} ({cm50.read_derating()['auction']} factors held to {model_year}, declared)",
        "fleet_source": f"FES 2025 HT ES1 capacity path, stock-flow vintages at the Y2 lives ({cm50.FLEET_FILE})",
        "declared": [
            "Requirement: de-rated capacity per GW of ACS peak held at its 2030/31 value (ratio).",
            "Agreements of the unheld auctions (2030/31 onward) are taken at the model year's clearing price "
            "(steady state, as 2050 declares).",
            "New-build storage agreements are taken as batteries; refurbishing storage agreements as pumped storage.",
            "An agreement whose carrier the FES fleet holds less of than is booked is taken as terminated "
            "(contract_book.by_carrier matched_share).",
            "V2G bids in as DSR centrally (FES 2050 decision 30); unpaid is the v2g_outside_cm_unpaid sensitivity.",
            cm50.DECLARED_DURATION_NOTE["CAES"],
        ],
        "builder": "scripts/market/build_capacity_market_year.py",
    }
    return {"summary": summary, "revenue": revenue, "classified": classified, "book": book}


def revenue_table(classified: pd.DataFrame, cleared: dict, by_carrier: pd.DataFrame, indexation: float,
                  model_year: int) -> pd.DataFrame:
    """Calendar revenue by CM rate key: the auction's (``revenue_by_carrier``) plus the matched book."""
    auction = cm50.revenue_by_carrier(classified, cleared, indexation, model_year)
    label1, label2 = fiscal_year_label(model_year - 1), fiscal_year_label(model_year)
    cal = f"calendar_{model_year}_revenue_gbp"
    rate_key = {c: cm50.CARRIERS[c][0] if c in cm50.CARRIERS else "dsr" for c in by_carrier["carrier"]}
    book = by_carrier.assign(rate_key=by_carrier["carrier"].map(rate_key))
    factor = classified.groupby("carrier")["derating_factor"].mean()
    book["connection_mw"] = book["matched_derated_mw"] / book["carrier"].map(factor).where(lambda s: s > 0)
    b = book.groupby("rate_key").agg(derated=("matched_derated_mw", "sum"), connection=("connection_mw", "sum"),
                                     pay=("matched_payment_gbp", "sum"))
    out = auction.set_index("carrier")
    for key, row in b.iterrows():
        if key not in out.index:
            out.loc[key] = 0.0
        out.loc[key, f"derated_mw_{label1}"] += row["derated"]
        out.loc[key, f"connection_mw_{label1}"] += row["connection"] if np.isfinite(row["connection"]) else 0.0
        for col in (f"revenue_{label1}_gbp", f"revenue_{label2}_gbp", cal):
            out.loc[key, col] += row["pay"]
    out["gbp_per_kw_connection_year"] = out[cal] / (out[f"connection_mw_{label1}"].where(out[f"connection_mw_{label1}"] > 0) * 1e3)
    return out.reset_index().rename(columns={"index": "carrier"})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-year", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--v2g", default=CENTRAL_V2G, choices=cm50.V2G_TREATMENTS)
    args = parser.parse_args(argv)
    from scripts.market.build_capacity_market import auction_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = auction_model(cm.read_register())
    result = run_year(model, cm.read_cpi(), args.model_year, settings=cm50.Settings2050(v2g=args.v2g))
    summary, year = result["summary"], args.model_year
    (out / f"cm_calendar_{year}_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    result["revenue"].to_csv(out / f"cm_revenue_by_technology_{year}.csv", index=False)
    result["classified"].to_csv(out / f"cm_fleet_{year}.csv", index=False)
    result["book"].to_csv(out / f"cm_contract_book_{year}.csv", index=False)
    print("calendar %d CM cost (2025 GBP m): %.0f = book %.0f + auction %.0f at %.2f GBP/kW/yr (target %.0f MW, "
          "supply %.0f MW)" % (year, summary[f"calendar_{year}_cost_gbp"] / 1e6, summary["contract_book_gbp"] / 1e6,
                               summary["auction_cost_gbp"] / 1e6, summary["price_gbp_per_kw_year_2025"],
                               summary["auction"]["target_mw"], summary["auction"]["supply_total_mw"]))
    for label, case in summary["cases"].items():
        print("  %-24s price %6.2f  cost %8.0f m  shortfall %s" % (
            label, case["price_gbp_per_kw_year_auction_money"], case[f"calendar_{year}_cost_gbp"] / 1e6,
            case["shortfall"]))
    print("book by carrier:\n" + pd.DataFrame(summary["contract_book"]["by_carrier"]).to_string(index=False))


if __name__ == "__main__":
    main()
