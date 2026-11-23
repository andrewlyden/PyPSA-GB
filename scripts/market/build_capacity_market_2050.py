"""The calendar-2050 Capacity Market with V2G bidding in as DSR, centrally (FES 2050 decision 30).

User, 14 Nov 2026: built by the generic ``build_capacity_market_year.py`` (rule build_capacity_market_year, year
2050), which reproduces this builder's former outputs exactly; this module keeps the shared V2G and payee
definitions the year builder imports, and ``build``/``main`` are its 2050 entry point.

Task Y11 built the 2050 auction (``scripts/market/capacity_market_2050.py``) with V2G
counted against the target and paid nothing; V2G bidding in as DSR was its high case. The
user's decision 30 (3 October 2026) makes that case the central: V2G (consumer-owned,
decision 24) enters the auction as DSR, the auction clears, and the bill's Capacity Market
line carries the cost. "V2G netted out, unpaid" (Y11's old central) is kept as the
``v2g_outside_cm_unpaid`` sensitivity.

This builder runs the Y11 model with ``Settings2050(v2g=...)`` and writes the same three
files as ``build_capacity_market.py --model-year 2050`` (that script and the model are
declared inputs of 2030 and BM-label rules, so they are left as they are). It adds who is
paid: each auction row's payment (``capacity_market_2050.revenue_by_carrier``'s own
arithmetic), split into V2G owners (households: a consumer-to-consumer transfer outside
every ownership case), other DSR, and the plant and interconnectors the finance register
books.

Usage::

    python scripts/market/build_capacity_market_2050.py --out resources/market/capacity_market [--v2g in_auction]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.market import capacity_market as cm  # noqa: E402
from scripts.market import capacity_market_2050 as cm50  # noqa: E402

#: Decision 30: V2G bids into the 2050 auction as DSR, centrally.
CENTRAL_V2G = "in_auction"
#: Y11's old central, kept as the sensitivity: V2G counts against the target and is paid nothing.
UNPAID_CASE = "v2g_outside_cm_unpaid"
#: Auction rows (``flexibility_rows`` carriers) whose owners are outside the finance register.
CONSUMER_FLEXIBILITY = {"V2G": "v2g_owners_households", "DSR_residential": "dsr_residential",
                        "DSR_industrial_commercial": "dsr_industrial_commercial"}


def payments_by_row(classified: pd.DataFrame, auction: dict, indexation: float) -> pd.DataFrame:
    """Each auction row's calendar payment, as ``capacity_market_2050.revenue_by_carrier`` computes it.

    The category's award (T-1 pro rata) is split across its rows by eligible de-rated
    capacity, times the clearing price in report money.
    """
    price = float(auction["price_gbp_per_kw_year_auction_money"]) * float(indexation)
    rows = classified[classified["eligible_derated_mw"] > 0].copy()
    by_cat = rows.groupby("category")["eligible_derated_mw"].transform("sum")
    award = rows["category"].map(auction["awarded_by_category_mw"]).fillna(0.0)
    t4_total = float(sum(auction["awarded_by_category_mw"].values()))
    t1_scale = 1.0 + (float(auction["t1_mw"]) / t4_total if t4_total > 0 else 0.0)
    rows["awarded_derated_mw"] = rows["eligible_derated_mw"] / by_cat * award * t1_scale
    rows["payment_gbp"] = rows["awarded_derated_mw"] * 1e3 * price
    return rows


def payee_split(classified: pd.DataFrame, auction: dict, indexation: float, total_gbp: float) -> dict:
    """The calendar payment by payee: V2G owners, other DSR, and the register's plant and interconnectors."""
    rows = payments_by_row(classified, auction, indexation)
    carrier = rows["carrier"].astype(str)
    out = {label: float(rows.loc[carrier.eq(c), "payment_gbp"].sum()) for c, label in CONSUMER_FLEXIBILITY.items()}
    out["register_plant_and_interconnectors"] = float(rows.loc[~carrier.isin(CONSUMER_FLEXIBILITY), "payment_gbp"].sum())
    if abs(sum(out.values()) - float(total_gbp)) > 1.0:
        raise ValueError(f"the payee split ({sum(out.values()):,.0f}) does not add up to the cost ({total_gbp:,.0f})")
    v2g = rows.loc[carrier.eq("V2G")]
    out["v2g_awarded_derated_mw"] = float(v2g["awarded_derated_mw"].sum())
    out["v2g_eligible_derated_mw"] = float(v2g["eligible_derated_mw"].sum())
    return out


def build(v2g: str = CENTRAL_V2G) -> dict:
    """The 2050 auction under ``v2g`` (``build_capacity_market_year.run_year`` for 2050; user, 14 Nov 2026).

    Returns:
        ``{"summary": dict, "revenue": DataFrame, "classified": DataFrame, "book": DataFrame}``.
    """
    from scripts.market.build_capacity_market import auction_model
    from scripts.market.build_capacity_market_year import run_year

    model = auction_model(cm.read_register())
    return run_year(model, cm.read_cpi(), cm50.MODEL_YEAR, settings=cm50.Settings2050(v2g=v2g))


def main(argv=None) -> None:
    """The 2050 entry point of ``build_capacity_market_year.py``."""
    from scripts.market.build_capacity_market_year import main as year_main

    args = list(sys.argv[1:] if argv is None else argv)
    year_main(["--model-year", str(cm50.MODEL_YEAR), *args])


if __name__ == "__main__":
    main()
