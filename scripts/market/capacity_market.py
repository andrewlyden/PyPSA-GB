"""The GB Capacity Market in calendar 2030: contract book and auction model (item MR4).

Calendar 2030 spans two delivery years (October to September). The supplier
charge is levied on 16:00-19:00 weekday demand from November to February, so
consumers in calendar 2030 pay about half of each: January and February of
2029/30 and November and December of 2030/31 (``CALENDAR_2030_WEIGHTS``).

* **Contract book.** Every live agreement in the NESO Capacity Market Register
  for a delivery year, at its auction's clearing price. Clearing prices are set
  in the auction's base-period money and CPI-indexed to the delivery year, so a
  payment in 2025 money is price x CPI(2025) / CPI(base financial year).
* **Auction model.** Auctions not yet held are cleared on their published
  demand curve (price cap at the minimum volume, net CONE at the target, zero
  at the maximum) against a supply curve built from the capacity that entered
  the last T-4, by CMU category, each category bidding to a distribution
  fitted to what cleared in the last two T-4 auctions (both cleared on the
  demand curve: 2028/29 at 60 GBP/kW with 43.06 GW, 2029/30 at 27.10 with
  40.11 GW). Existing plant is almost all a price-taker; new build,
  refurbishing plant and DSR set the price.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/market/capacity_market")
REGISTER_CMU = DATA / "cm_register_cmu_20260923.csv"
REGISTER_COMPONENTS = DATA / "cm_register_components_20260923.csv"
AUCTION_STATIC = DATA / "cm_auction_static_data.csv"
AUCTION_PARAMETERS = DATA / "cm_auction_parameters.csv"
CPI_FILE = DATA / "ons_cpi_d7bt.csv"

CALENDAR_2030_WEIGHTS = {"2029/2030": 0.5, "2030/2031": 0.5}
CATEGORIES = {
    "Existing Generating CMU": "existing",
    "Existing Interconnector CMU": "interconnector",
    "New Build Interconnector CMU": "interconnector",
    "New Build Generating CMU": "new_build",
    "Refurbishing Generating CMU": "refurbishing",
    "Proven DSR CMU": "dsr",
    "Unproven DSR CMU": "dsr",
}


# ── money ────────────────────────────────────────────────────────────────────

def read_cpi(path=CPI_FILE) -> pd.Series:
    """ONS CPI (D7BT, 2015=100) calendar-year averages by year."""
    rows = pd.read_csv(path, header=None, names=["period", "value"], skiprows=8)
    annual = rows[rows["period"].astype(str).str.fullmatch(r"\d{4}")]
    return pd.Series(annual["value"].astype(float).to_numpy(), index=annual["period"].astype(int))


def cpi_financial_year(cpi: pd.Series, start_year: int) -> float:
    """April-to-March average, from calendar averages (three quarters, one quarter)."""
    if start_year + 1 in cpi.index:
        return 0.75 * cpi[start_year] + 0.25 * cpi[start_year + 1]
    return float(cpi[start_year])


def base_financial_year(auction_name: str, static_base) -> int:
    """The financial year an auction's clearing price is expressed in.

    NESO's auction static data gives it for T-4 and T-3 auctions (e.g. "Winter
    2024/2025" for the 2029/30 T-4). Otherwise the year before the auction's
    financial year: a T-1 for delivery year Y is held in Y-1/Y, so base Y-2.
    """
    text = str(static_base or "")
    found = re.search(r"(\d{4})/\d{4}", text)
    if found:
        return int(found.group(1))
    year = int(auction_name[:4])
    lead = {"T-4": 5, "T-3": 4, "T-1": 2, "TR": 2}
    kind = auction_type(auction_name)
    return year - lead.get(kind, 2)


def auction_type(auction_name: str) -> str:
    found = re.search(r"\((T-\d|TR)\)", auction_name)
    return found.group(1) if found else "other"


def read_auction_static(path=AUCTION_STATIC) -> pd.DataFrame:
    static = pd.read_csv(path, encoding="latin-1")
    static.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in static.columns]
    static = static.rename(columns={
        "Auction Name": "auction", "Clearing Price_GBP per kW per year": "clearing_price",
        "Base Period for the Agreement": "base_period", "Auction Target Capacity_MW": "target_mw",
        "Total Volume of Agreements Awarded": "awarded_mw"})
    static["base_year"] = [base_financial_year(a, b) for a, b in zip(static["auction"], static["base_period"])]
    return static.set_index("auction")


# ── register ─────────────────────────────────────────────────────────────────

def _num(frame, column):
    return pd.to_numeric(frame[column], errors="coerce")


def read_register(path=REGISTER_CMU) -> pd.DataFrame:
    """The CMU register with a capacity column that also covers refurbishing CMUs."""
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame["capacity_mw"] = (_num(frame, "De Rated Capacity")
                            .fillna(_num(frame, "Pre Refurbishing De Rated Capacity"))
                            .fillna(_num(frame, "Anticipated De Rated Capacity"))
                            .fillna(_num(frame, "Anticipated Pre Refurbishing De Rated Capacity")))
    frame["obligation_mw"] = _num(frame, "Auction Acquired Capacity Obligation")
    frame["category"] = frame["CM Unit Category"].map(CATEGORIES).fillna("other")
    frame["technology"] = frame["CMU Technology"].astype(str).str.split(",").str[0].str.strip()
    return frame


def read_agreements(register: pd.DataFrame | None = None, static: pd.DataFrame | None = None) -> pd.DataFrame:
    """Live (awarded, unterminated) agreements, one row each, priced and dated."""
    register = read_register() if register is None else register
    static = read_auction_static() if static is None else static
    live = register[register["Capacity Agreement Awarded"].eq("Yes")
                    & ~register["Capacity Agreement has been Terminated"].astype(str).str.strip().eq("Yes")]
    live = live.drop_duplicates("Unique Agreement Identifier").copy()
    live["delivery_years"] = live["Delivery Years for which Agreement Valid"].astype(str).str.split(r",\s*")
    live["clearing_price"] = live["Auction Name"].map(static["clearing_price"])
    live["base_year"] = live["Auction Name"].map(static["base_year"])
    live["mw"] = live["obligation_mw"].fillna(live["capacity_mw"])
    missing = live["clearing_price"].isna()
    if missing.any():
        raise ValueError(f"no clearing price for {sorted(live.loc[missing, 'Auction Name'].unique())}")
    return live.rename(columns={"Unique Agreement Identifier": "agreement_id", "CMU ID": "cmu_id",
                                "CM Unit Name": "cmu_name", "Auction Name": "auction",
                                "Transmission or Distribution": "connection",
                                "Connection or DSR Capacity": "connection_mw",
                                "Duration of Capacity Agreement Awarded": "term_years"})


def contract_book(agreements: pd.DataFrame, delivery_year: str, cpi: pd.Series, price_year: int = 2025) -> pd.DataFrame:
    """Agreements live in ``delivery_year`` with their annual payment in ``price_year`` money."""
    live = agreements[agreements["delivery_years"].map(lambda years: delivery_year in years)].copy()
    live["indexation"] = [cpi[price_year] / cpi_financial_year(cpi, int(b)) for b in live["base_year"]]
    live["payment_gbp"] = live["mw"] * 1e3 * live["clearing_price"] * live["indexation"]
    live["delivery_year"] = delivery_year
    return live


# ── auction model ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DemandCurve:
    target_mw: float
    min_at_cap_mw: float
    max_at_zero_mw: float
    net_cone: float
    price_cap: float

    def quantity(self, price: float) -> float:
        """Capacity demanded at ``price`` (GBP/kW/yr)."""
        if price >= self.price_cap:
            return self.min_at_cap_mw
        if price >= self.net_cone:
            share = (self.price_cap - price) / (self.price_cap - self.net_cone)
            return self.min_at_cap_mw + share * (self.target_mw - self.min_at_cap_mw)
        share = (self.net_cone - max(price, 0.0)) / self.net_cone
        return self.target_mw + share * (self.max_at_zero_mw - self.target_mw)

    def price(self, quantity: float) -> float:
        """Inverse of ``quantity``: the curve's price at ``quantity`` MW."""
        if quantity <= self.min_at_cap_mw:
            return self.price_cap
        if quantity <= self.target_mw:
            share = (quantity - self.min_at_cap_mw) / (self.target_mw - self.min_at_cap_mw)
            return self.price_cap - share * (self.price_cap - self.net_cone)
        if quantity >= self.max_at_zero_mw:
            return 0.0
        share = (quantity - self.target_mw) / (self.max_at_zero_mw - self.target_mw)
        return self.net_cone * (1.0 - share)


def read_demand_curves(path=AUCTION_PARAMETERS) -> dict:
    table = pd.read_csv(path)
    curves = {}
    for _, row in table.dropna(subset=["min_capacity_at_price_cap_mw", "max_capacity_at_zero_mw"]).iterrows():
        curves[row["auction"]] = DemandCurve(float(row["target_mw"]), float(row["min_capacity_at_price_cap_mw"]),
                                             float(row["max_capacity_at_zero_mw"]), float(row["net_cone_gbp_per_kw_year"]),
                                             float(row["price_cap_gbp_per_kw_year"]))
    return curves


def bid_share(price, floor_share: float, shape: float, price_cap: float = 75.0, scale: float = 1.0):
    """Share of a category's entered capacity bidding at or below ``price``.

    ``floor_share`` bids nothing (price-takers at zero); the rest spreads up to
    the price cap as (price / cap) ** shape.  ``scale`` multiplies every bid
    (F9: a state owner's new build bids its missing money at its own cost of
    capital, ``scale`` < 1), so the rest spreads up to ``scale`` x the cap.
    """
    if not scale > 0.0:
        raise ValueError(f"bid scale must be positive, got {scale}")
    x = np.clip(np.asarray(price, dtype=float) / (price_cap * float(scale)), 0.0, 1.0)
    return floor_share + (1.0 - floor_share) * x ** shape


def _category_share(price, distribution, price_cap: float):
    """:func:`bid_share` for one category's ``(floor_share, shape)`` or ``(floor_share, shape, scale)``."""
    floor_share, shape, *rest = distribution
    return bid_share(price, floor_share, shape, price_cap, rest[0] if rest else 1.0)


def fit_bid_distribution(points, price_cap: float = 75.0) -> tuple[float, float]:
    """(floor_share, shape) through observed (price, share cleared) points.

    Least squares on a grid (floor 0 to 0.999, shape 0.05 to 20); a share that
    falls with price across auctions (different populations) is fitted, not forced.
    """
    prices = np.array([p for p, _ in points], dtype=float)
    shares = np.array([s for _, s in points], dtype=float)
    floors = np.linspace(0.0, 0.999, 1000)
    shapes = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    x = np.clip(prices / price_cap, 1e-9, 1.0)
    grid = floors[:, None, None] + (1 - floors[:, None, None]) * x[None, None, :] ** shapes[None, :, None]
    error = ((grid - shares[None, None, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmin(error), error.shape)
    return float(floors[i]), float(shapes[j])


def entered_by_category(register: pd.DataFrame, auction: str) -> pd.DataFrame:
    """Capacity that entered an auction and capacity awarded, by category (MW de-rated)."""
    rows = register[register["Auction Name"].eq(auction)].drop_duplicates("CMU ID")
    awarded = rows["Capacity Agreement Awarded"].eq("Yes")
    entered = rows["CMU confirmed entry into Auction"].eq("Yes") | awarded
    out = pd.DataFrame({
        "entered_mw": rows[entered].groupby("category")["capacity_mw"].sum(),
        "awarded_mw": rows[awarded].groupby("category")["obligation_mw"].sum(),
    }).fillna(0.0)
    out["multi_year_awarded_mw"] = rows[awarded & (pd.to_numeric(rows["Duration of Capacity Agreement Awarded"],
                                                                   errors="coerce") > 1)].groupby("category")[
        "obligation_mw"].sum().reindex(out.index).fillna(0.0)
    return out


def clear_auction(entered: pd.Series, distributions: dict, curve: DemandCurve) -> dict:
    """Clearing price and volume where supply S(p) meets the demand curve D(p)."""
    def supply(p):
        return float(sum(entered[c] * _category_share(p, distributions[c], curve.price_cap) for c in entered.index))

    if supply(0.0) >= curve.max_at_zero_mw:
        return {"price": 0.0, "awarded_mw": curve.max_at_zero_mw, "supply_at_cap_mw": supply(curve.price_cap)}
    if supply(curve.price_cap) < curve.min_at_cap_mw:
        return {"price": curve.price_cap, "awarded_mw": supply(curve.price_cap), "shortfall": True,
                "supply_at_cap_mw": supply(curve.price_cap)}
    lo, hi = 0.0, curve.price_cap
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if supply(mid) >= curve.quantity(mid):
            hi = mid
        else:
            lo = mid
    return {"price": hi, "awarded_mw": curve.quantity(hi), "supply_at_cap_mw": supply(curve.price_cap)}


def awarded_by_category(entered: pd.Series, distributions: dict, price: float, awarded_mw: float,
                        price_cap: float = 75.0) -> pd.Series:
    """Split a cleared auction's awarded volume across categories (item B7).

    Each category clears what it bids at or below the clearing price,
    ``entered x bid_share(price)``; the split is scaled so it sums exactly to the
    awarded volume (the bisection leaves supply and demand a hair apart).
    """
    raw = pd.Series({c: float(entered[c]) * float(_category_share(price, distributions[c], price_cap))
                     for c in entered.index}, dtype=float)
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("No category clears at the auction price; the award cannot be allocated")
    return raw * (float(awarded_mw) / total)


# ── locations (component OS grid references) ────────────────────────────────

_GRID = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def os_grid_to_easting_northing(reference: str):
    """'NO 396 371' -> (339600, 737100) in metres; None if unparseable."""
    text = re.sub(r"\s+", "", str(reference or "")).upper()
    found = re.fullmatch(r"([A-HJ-Z]{2})(\d+)", text)
    if not found or len(found.group(2)) % 2:
        return None
    letters, digits = found.groups()
    i1, i2 = _GRID.index(letters[0]), _GRID.index(letters[1])
    e100 = ((i1 - 2) % 5) * 5 + (i2 % 5)
    n100 = (19 - (i1 // 5) * 5) - (i2 // 5)
    half = len(digits) // 2
    scale = 10 ** (5 - half)
    return e100 * 100000 + int(digits[:half]) * scale, n100 * 100000 + int(digits[half:]) * scale


# ── model assets ─────────────────────────────────────────────────────────────

# Model carrier -> carrier key of ``cm_revenue_by_technology.csv``.
MODEL_CARRIER_RATE_KEY = {
    "CCGT": "CCGT", "OCGT": "OCGT", "gas_engine": "gas_engine", "CHP": "CHP", "oil": "oil",
    "nuclear": "nuclear", "large_hydro": "large_hydro", "small_hydro": "large_hydro",
    "waste_to_energy": "waste_to_energy", "biomass": "biomass", "CCS Biomass": "biomass",
    "wind_onshore": "wind_onshore", "wind_offshore": "wind_offshore", "solar_pv": "solar_pv",
    "Battery": "battery", "battery": "battery",
    "LAES": "long_duration_storage", "Pumped Storage Hydroelectricity": "long_duration_storage",
    "pumped_hydro": "long_duration_storage",
    # Task Y11: carriers of the FES 2050 fleet. A 2030 rates table has no row for
    # them, so a 2030 unit carrying one earns a zero rate at a zero share, as before.
    "CAES": "long_duration_storage", "CCS Gas": "CCS Gas", "Hydrogen": "Hydrogen", "marine": "marine",
}
#: User decision (6 Oct 2026): the rate key of CP30's new LDES on Ofgem's cap and floor. Its row
#: in the 2030 rates table is its own 2030/31 T-4 award, kept apart from long-duration storage's
#: (existing pumped hydro's) contract book, so neither is spread over the other's units.
LDES_CAP_AND_FLOOR_RATE_KEY = "long_duration_storage_cap_and_floor"
#: Model carrier -> the rate key it takes when the rates table has that row; otherwise its
#: ``MODEL_CARRIER_RATE_KEY`` entry (the 2050 tables have no such row, so 2050 is unchanged).
PREFERRED_RATE_KEY = {"LAES": LDES_CAP_AND_FLOOR_RATE_KEY}
# Units on a low-carbon support scheme cannot hold a Capacity Market agreement.
CM_INELIGIBLE_SUPPORT = {"CfD", "ROC", "RO", "DPA"}
#: The 2030 rates table's contracted-connection column (item MR4).
CONNECTION_COLUMN_2030 = "connection_mw_2029_30"


def contracted_connection_column(rates: pd.DataFrame) -> str:
    """The contracted connection-capacity column of a CM rates table.

    The 2030 table names it ``connection_mw_2029_30``; a later year's table (task Y11,
    ``cm_revenue_by_technology_2050.csv``) names its first delivery year, e.g.
    ``connection_mw_2049_50``. The 2030 name wins when present, so 2030 is unchanged.

    Args:
        rates: A ``cm_revenue_by_technology`` table.

    Returns:
        The column name.

    Raises:
        ValueError: None, or more than one, year-labelled connection column.
    """
    if CONNECTION_COLUMN_2030 in rates.columns:
        return CONNECTION_COLUMN_2030
    found = [c for c in rates.columns if re.fullmatch(r"connection_mw_\d{4}_\d{2}", str(c))]
    if len(found) != 1:
        raise ValueError(f"a CM rates table needs one connection_mw_<yyyy>_<yy> column, found {found}")
    return found[0]


def attach_capacity_market_revenue(assets: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Add ``capacity_market_revenue_gbp``: calendar-year CM revenue per model unit (2030, or 2050 for Y11).

    Each eligible unit earns its technology's calendar-2030 CM revenue per kW
    of contracted connection capacity (``rates``, from the contract book and the
    modelled 2030/31 T-4) on its own capacity, times a participation share:
    where the model has more eligible capacity of a technology than holds
    agreements (CP30's long-duration storage, merchant solar), units earn pro
    rata, so no technology earns more than the contract book pays it. Units on a
    CfD or the RO, foreign plant and unmapped carriers earn nothing. A carrier
    with a ``PREFERRED_RATE_KEY`` the table carries (LAES: new LDES on the cap
    and floor, user decision 6 Oct 2026) takes that row, so it earns its own
    award and not a capacity share of existing pumped hydro's agreements.
    """
    table = rates.set_index("carrier")
    out = assets.copy()
    key = out["carrier"].astype(str).map(MODEL_CARRIER_RATE_KEY)
    preferred = out["carrier"].astype(str).map(PREFERRED_RATE_KEY)
    key = key.where(~preferred.isin(list(table.index)), preferred)
    rate = key.map(table["gbp_per_kw_connection_year"]).fillna(0.0)
    support = out["support_type"].astype(str) if "support_type" in out.columns else pd.Series("", index=out.index)
    eligible = key.notna() & ~support.isin(CM_INELIGIBLE_SUPPORT)
    capacity = pd.to_numeric(out["capacity_mw"], errors="coerce").fillna(0.0)
    model_mw = capacity[eligible].groupby(key[eligible]).sum()
    contracted_mw = table[contracted_connection_column(rates)].reindex(model_mw.index).fillna(0.0)
    share = (contracted_mw / model_mw.where(model_mw > 0)).clip(upper=1.0).fillna(0.0)
    out["capacity_market_participation_share"] = np.where(eligible, key.map(share).fillna(0.0), 0.0)
    out["capacity_market_revenue_gbp"] = np.where(
        eligible, capacity * 1e3 * rate * out["capacity_market_participation_share"], 0.0)
    return out
