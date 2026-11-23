"""F9: the 2030/31 T-4 Capacity Market re-cleared with the state's plant bidding at its own cost of capital.

The calendar-2030 Capacity Market (item MR4, ``scripts/market/build_capacity_market.py``)
clears the 2030/31 T-4 on DESNZ's demand curve against the CP30 2030 fleet, each
MR4 category bidding to a distribution fitted to the last two T-4s. A new-build
bid is roughly its missing money: annuitised fixed cost less expected net revenue.
A state owner finances its plant at its own rate, so its bids fall:

* **Which supply.** Every supply row the case's state owns any of: the register's
  new-build rows (``finance_vintage == "new_build"``) with ``ownership_share > 0``,
  in the categories whose bids recover capital (config ``categories``: new build
  and refurbishing). Existing plant bids its going-forward cost, which carries no
  capital charge; new LDES on Ofgem's cap and floor (``ldes_cap_and_floor``) is a
  price-taker centrally, so it never sets the price; DSR and the cables are private.
* **How far it falls.** A row's bid scales by
  ``k = 1 - (1 - CRF_case / CRF_private) x capital share``, the capital share being
  the private row's annuitised capital over its annuitised capital plus fixed O&M,
  and ``CRF_case / CRF_private`` the case's annual capital recovery over the private
  row's on the same capital (the register prices both). A jointly owned row (P1's
  30% stake) bids once, at its own blended cost, as a joint venture's single
  bidding company would. A category's state part bids at its supply-weighted ``k``.
* **The re-clear.** The state part of each such category is split off at ``k`` (its
  bid distribution scaled, :func:`scripts.market.capacity_market.bid_share`), the
  rest keeps the fitted distribution, and the auction clears again on the same
  demand curve. The new price is paid to every 2030/31 T-4 winner and to the 2030/31
  T-1 (pay as clear); the 2029/30 delivery year and every agreement already held
  stay as they are. Calendar 2030 pays half of 2030/31, so

      change = 0.5 x [(q' + T1) p' - (q + T1) p]

  in the builder's money (2025), a share of the calendar-2030 total the bill's line
  pins. The bill books that share of the private case's Capacity Market line.
* **Who loses it.** The winners: the state's part (its rows' share of the
  2030/31 award at the new price) comes off the state's receipts, the rest off
  private investors'. It is a transfer, never a resource saving. The register keeps
  every CM receipt at the central price ("as now", design step 4), so the state's
  part is booked to the Exchequer column in the ledger.

Configured in ``public_ownership.capacity_market_state_bids``; off by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from scripts.market import capacity_market as cm

CONFIG_KEY = "capacity_market_state_bids"
LINE = "capacity_market_state_bids_at_cost"
#: Code defaults: the switch off; the rest is config/defaults/base/public_ownership.yaml's.
DEFAULT_SETTINGS: dict = {
    "enabled": False,
    # MR4 categories whose bids recover capital (new build's missing money, refurbishment capex).
    "categories": ["new_build", "refurbishing"],
    "auction": "T-4 2030/31",
    # The builder's 2030/31 T-1 (build_capacity_market.T1_VOLUME_MW), paid the T-4 price.
    "t1_volume_mw": 1000.0,
    # Calendar 2030 pays half of 2030/31 (capacity_market.CALENDAR_2030_WEIGHTS).
    "calendar_weight": 0.5,
}
#: The builder's price-taking LDES category bids as existing plant (build_capacity_market.BID_DISTRIBUTION_OF).
BID_DISTRIBUTION_OF = {"ldes_cap_and_floor": "existing"}
#: The builder writes prices to the penny; the re-clear without state bids must reproduce it.
REPRODUCTION_TOLERANCE = 0.01


def settings_from_config(public_ownership: Mapping | None) -> dict:
    """``public_ownership.capacity_market_state_bids`` over the defaults, checked."""
    block = dict(((public_ownership or {}).get(CONFIG_KEY)) or {})
    unknown = sorted(set(block) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"public_ownership.{CONFIG_KEY} has unknown keys {unknown}; supported: {sorted(DEFAULT_SETTINGS)}")
    merged = {**DEFAULT_SETTINGS, **block}
    if not isinstance(merged["enabled"], bool):
        raise ValueError(f"public_ownership.{CONFIG_KEY}.enabled must be true or false")
    allowed = {"new_build", "refurbishing", "existing", "ldes_cap_and_floor"}
    if not set(merged["categories"]) <= allowed:
        raise ValueError(f"public_ownership.{CONFIG_KEY}.categories must be among {sorted(allowed)}")
    if not 0.0 < float(merged["calendar_weight"]) <= 1.0 or float(merged["t1_volume_mw"]) < 0.0:
        raise ValueError(f"public_ownership.{CONFIG_KEY}: calendar_weight in (0, 1] and t1_volume_mw >= 0")
    return {"enabled": merged["enabled"], "categories": [str(c) for c in merged["categories"]],
            "auction": str(merged["auction"]), "t1_volume_mw": float(merged["t1_volume_mw"]),
            "calendar_weight": float(merged["calendar_weight"])}


def load_auction(model_path, supply_path, calendar_path, settings: Mapping, curves: Mapping | None = None) -> dict:
    """The builder's central 2030/31 T-4: bid distributions, supply rows, demand curve, price and calendar total.

    Refuses unless clearing the supply rows unchanged reproduces the builder's price.
    """
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    calendar = json.loads(Path(calendar_path).read_text(encoding="utf-8"))
    central = (model.get("t4_2030_31") or {}).get("central")
    if central is None:
        raise ValueError(f"{model_path} has no central 2030/31 T-4 (built before the CP30-fleet auction)")
    bids = {c: (float(v["floor_share"]), float(v["shape"])) for c, v in model["distributions"].items()}
    bids.update({c: bids[d] for c, d in BID_DISTRIBUTION_OF.items()})
    supply = pd.read_csv(supply_path)
    supply = supply.loc[supply["case"].eq("central") & pd.to_numeric(supply["supply_mw"], errors="coerce").gt(0)].copy()
    supply["supply_mw"] = pd.to_numeric(supply["supply_mw"], errors="coerce")
    curve = (curves or cm.read_demand_curves())[settings.get("auction", DEFAULT_SETTINGS["auction"])]
    entered = supply.groupby("category")["supply_mw"].sum()
    base = cm.clear_auction(entered, bids, curve)
    if abs(float(base["price"]) - float(central["price"])) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"the 2030/31 T-4 supply rows clear at {base['price']:.2f}, not the builder's "
                         f"{central['price']:.2f}: the supply table and the auction model are from different builds")
    year = (calendar.get("delivery_years") or {}).get("2030/2031") or {}
    price, awarded = float(central["price"]), float(central["awarded_mw"])
    # The builder's 2030/31 money index (cpi 2025 / cpi FY2025), read back from its T-4 payment.
    index = float(year["t4_modelled_gbp"]) / (awarded * 1e3 * price) if price > 0 and awarded > 0 else 1.0
    # The comparison is against the same supply cleared unchanged (the builder rounds its price to the penny).
    return {"bids": bids, "supply": supply, "curve": curve, "price": float(base["price"]),
            "awarded_mw": float(base["awarded_mw"]), "builder_price": price,
            "money_index": index, "calendar_2030_cost_gbp": float(calendar["calendar_2030_cost_gbp"])}


def row_bid_factors(case_rows: pd.DataFrame, private_rows: pd.DataFrame) -> pd.DataFrame:
    """Per register asset: ``state_owned`` (any state share of new build) and the bid scale ``k`` (see the module)."""
    def frame(rows):
        out = rows.drop_duplicates("asset_id").set_index("asset_id")
        return out

    case, private = frame(case_rows), frame(private_rows)
    num = lambda f, c: pd.to_numeric(f[c], errors="coerce").fillna(0.0)  # noqa: E731
    share = num(case, "ownership_share")
    new_build = case["finance_vintage"].astype(str).eq("new_build")
    cap_case = num(case, "annual_capital_recovery_gbp")
    cap_private = num(private, "annual_capital_recovery_gbp").reindex(case.index).fillna(0.0)
    fom_private = num(private, "annual_fixed_om_gbp").reindex(case.index).fillna(0.0)
    fixed = cap_private + fom_private
    ratio = (cap_case / cap_private).where(cap_private > 0, 1.0)
    capital_share = (cap_private / fixed).where(fixed > 0, 0.0)
    k = (1.0 - (1.0 - ratio) * capital_share).clip(lower=1e-6, upper=1.0)
    return pd.DataFrame({"state_owned": (share > 0) & new_build, "ownership_share": share, "k": k})


def reclear(auction: Mapping, factors: pd.DataFrame, settings: Mapping) -> dict:
    """The 2030/31 T-4 with the case's state-owned supply bidding at ``k``, and what calendar 2030 pays for it."""
    supply = auction["supply"].copy()
    lookup = factors.reindex(supply["asset_id"].astype(str))
    owned = lookup["state_owned"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    state = owned & supply["category"].isin(settings["categories"]).to_numpy()
    supply["state_mw"] = np.where(state, supply["supply_mw"], 0.0)
    supply["k"] = np.where(state, lookup["k"].fillna(1.0).to_numpy(), 1.0)
    # The state's share of what each row wins, whatever its category (the payment split).
    supply["state_owned_share"] = np.where(owned,
                                           lookup["ownership_share"].fillna(0.0).to_numpy(), 0.0)
    entered, bids = {}, {}
    for category, rows in supply.groupby("category"):
        state_mw = float(rows["state_mw"].sum())
        entered[category] = float(rows["supply_mw"].sum()) - state_mw
        bids[category] = auction["bids"][category]
        if state_mw > 0:
            key = f"{category}@state"
            entered[key] = state_mw
            k = float((rows["state_mw"] * rows["k"]).sum() / state_mw)
            bids[key] = (*auction["bids"][category], k)
    entered = pd.Series(entered, dtype=float)
    entered = entered[entered > 0]
    result = cm.clear_auction(entered, bids, auction["curve"])
    price, awarded = float(result["price"]), float(result["awarded_mw"])
    by_category = cm.awarded_by_category(entered, bids, price, awarded, auction["curve"].price_cap)
    # The state's part of the award: its split-off part where bidding at k, else its rows' share pro rata.
    state_award = 0.0
    for category, rows in supply.groupby("category"):
        total = float(rows["supply_mw"].sum())
        if total <= 0:
            continue
        state_split = float(rows["state_mw"].sum())
        private_part = float(by_category.get(category, 0.0))
        if state_split > 0:
            # A jointly owned row bids whole; the state is paid its ownership share of what it wins.
            owned_share = float((rows["state_mw"] * rows["state_owned_share"]).sum()) / state_split
            state_award += float(by_category.get(f"{category}@state", 0.0)) * owned_share
        # Rows not split off: their state share of the category's (private-part) award.
        rest = rows.loc[rows["state_mw"].eq(0)]
        rest_mw = total - state_split
        if rest_mw > 0 and private_part > 0:
            state_award += private_part * float((rest["supply_mw"] * rest["state_owned_share"]).sum()) / rest_mw
    weight, t1, index = float(settings["calendar_weight"]), float(settings["t1_volume_mw"]), float(auction["money_index"])
    before = (auction["awarded_mw"] + t1) * auction["price"]
    after = (awarded + t1) * price
    change = weight * 1e3 * index * (after - before)
    state_share = state_award / awarded if awarded > 0 else 0.0
    return {
        "price_gbp_per_kw_year": price, "price_change_gbp_per_kw_year": price - float(auction["price"]),
        "awarded_mw": awarded, "awarded_change_mw": awarded - float(auction["awarded_mw"]),
        "state_bidding_mw": {c: float(v) for c, v in entered.items() if c.endswith("@state")},
        "state_bid_scale": {c: float(bids[c][2]) for c in bids if c.endswith("@state")},
        "state_award_share": state_share,
        "calendar_2030_cost_change_gbp_2025": change,
        "calendar_2030_cost_change_share": change / float(auction["calendar_2030_cost_gbp"]),
        "state_part_share": state_share,
        "base_price_gbp_per_kw_year": float(auction["price"]),
    }


def method_note(result: Mapping) -> str:
    return (f"F9: the 2030/31 T-4 re-cleared with the state's own plant bidding its missing money at its own cost of "
            f"capital: {result['base_price_gbp_per_kw_year']:.2f} -> {result['price_gbp_per_kw_year']:.2f} GBP/kW/yr, "
            f"paid to every 2030/31 winner and the T-1 (pay as clear); 2029/30 and the agreements already held are "
            f"unchanged, and calendar 2030 pays half of 2030/31. The Capacity Market line moves by "
            f"{result['calendar_2030_cost_change_share']:.3%}, recovered as the line itself (peak-weighted). New LDES "
            "on the cap and floor is a price-taker and does not set the price.")
