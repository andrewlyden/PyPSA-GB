"""Build the calendar-2030 Capacity Market: contract book, auction model, assets (item MR4).

Reads the register and auction data, and the CP30 2030 fleet the model builds
(rule ``build_capacity_market``; user decision, 6 Oct 2026)::

    python scripts/market/build_capacity_market.py [--fleet-assets ASSETS --fleet-interconnectors CABLES
                                                    --fleet-es1 ES1 --dsr-fes FLX1]

The fleet is the cp30_2030_cp30_network register's asset inputs (built from the solved market
months, which read no CM output), the CP30 build's mapped interconnectors and the
CP30 ES1 capacity path; none depends on this rule's outputs.

The revenue-adequacy table reads a run group's solved settlement (rule
``report_capacity_market_revenue_adequacy``)::

    python scripts/market/build_capacity_market.py --adequacy SETTLEMENT ASSET_INPUTS OUT

Writes ``resources/market/capacity_market/``:

* ``cm_contract_book_2030.csv``    - every live agreement for 2029/30 and 2030/31
* ``cm_auction_model.json``        - fitted bid distributions, the two T-4 checks,
                                     the 2030/31 T-4 on the CP30 fleet and its cases
* ``cm_calendar_2030_summary.json`` - the bill total (2025 prices) and its parts
* ``cm_revenue_by_technology.csv`` - calendar-2030 CM revenue per kW by model carrier
* ``cm_station_matches.csv``       - gas CMUs matched to the named 2030 gas fleet
* ``cm_t4_2030_31_supply.csv``     - the central 2030/31 T-4 supply row by row, and
                                     every fleet row left out (and why)

and, with ``--adequacy``, energy margin + CM against going-forward and
new-build fixed costs per gas class.

Task Y11 (FES 2050): ``--model-year 2050`` builds the calendar-2050 CM instead
(``scripts/market/capacity_market_2050.py``; rule ``build_capacity_market_2050``)::

    python scripts/market/build_capacity_market.py --model-year 2050 --out DIR

writing ``cm_calendar_2050_summary.json``, ``cm_revenue_by_technology_2050.csv``
(``calendar_2050_revenue_gbp``, ``connection_mw_2049_50``) and
``cm_fleet_2050.csv``. ``--extract-fleet`` re-extracts the FES trajectory table
(``data/market/capacity_market/cm_2050_fleet_fes2025_ht.csv``) by hand. Without
``--model-year`` (or with 2030) it builds 2030. The 2050 build reads only the fitted
bid distributions (``auction_model(register)`` without a fleet), which the 6 Oct 2026
change leaves as they were.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.market import capacity_market as cm  # noqa: E402
from scripts.market import capacity_market_2050 as cm50  # noqa: E402  (read-only: de-rating helpers)

OUT = Path("resources/market/capacity_market")
GAS_SCHEDULE = Path("data/market/CP30/gas_station_schedule_2030.csv")
GAS_COSTS = Path("data/market/capacity_market/gas_plant_costs_2024.csv")
GAS_COST_SOURCE = ("CEPA and GHD for DESNZ, Unabated gas electricity generation costs 2025 (2024 prices), "
                   "https://assets.publishing.service.gov.uk/media/696616c68d599f4c09e1ffae/"
                   "unabated-gas-electricity-generation-costs-2025.pdf; cached at "
                   "data/market/capacity_market/sources/desnz_unabated_gas_generation_costs_2025.pdf")
ASSET_SETTLEMENT = Path("resources/public_ownership/cp30_2030_cp30_network_asset_settlement.csv")
ASSET_INPUTS = Path("resources/public_ownership/cp30_2030_cp30_network_asset_inputs.csv")
OFGEM_LEDGER = Path("data/market/ofgem_price_cap/2024_annual/component_ledger.csv")
ESPENI = Path("data/demand/espeni.csv")
PRICE_YEAR = 2025
MERCHANT_WACC = 0.0824        # the register's imputed merchant WACC for thermal plant
T1_VOLUME_MW = 1000.0         # 2029/30 set-aside (DESNZ, July 2025); 2030/31 assumed equal
T4_2930 = "2029-30 (T-4) Four Year Ahead Capacity Auction"
T4_2829 = "2028-29 (T-4) Four Year Ahead Capacity Auction"

# CM technology class (first listed) -> model carrier
TECHNOLOGY_CARRIER = {
    "Combined Cycle Gas Turbine (CCGT)": "CCGT", "Combined Heat and Power (CHP)": "CHP",
    "CHP and autogeneration": "CHP", "Open Cycle Gas Turbine (OCGT)": "OCGT",
    "Reciprocating engines": "gas_engine", "OCGT and Reciprocating Engines": "gas_engine",
    "OCGT and Reciprocating Engines (Fuel Type - Gas)": "gas_engine",
    "OCGT and Reciprocating Engines (Fuel Type - Diesel)": "oil", "Oil Gen & Recip": "oil",
    "Nuclear": "nuclear", "Hydro": "large_hydro", "Energy from Waste": "waste_to_energy",
    "Coal/biomass": "biomass", "DSR": "dsr", "Onshore Wind": "wind_onshore", "Offshore Wind": "wind_offshore",
    "Solar Photovoltaic": "solar_pv",
}


def technology_carrier(technology: str) -> str:
    if technology.startswith("Storage"):
        hours = pd.to_numeric(pd.Series([technology]).str.extract(r"([\d.]+)h")[0], errors="coerce").iloc[0]
        return "long_duration_storage" if hours == hours and hours >= 8 else "battery"
    if "(" in technology and technology.split(" (")[0] in {
            "IFA", "IFA2", "Eleclink", "BritNED", "NEMO", "NSL", "VikingLink", "NeuConnect", "Moyle", "EWIC",
            "Greenlink"}:
        return "interconnector"
    return TECHNOLOGY_CARRIER.get(technology, "other")


# ── auction model ────────────────────────────────────────────────────────────

def fit_distributions(register) -> tuple[dict, dict, float]:
    """Bid distributions anchored on the 2029/30 T-4, with one price-response shape.

    The 2030/31 T-4 keeps the 2029/30 rules (3-year agreements at 65 GBP/kW of
    capex, down from 170), under which much existing gas entered as
    "refurbishing" and bid like existing plant; the 2028/29 refurbishing
    population is not the same. So each category's share at or below 27.10 is
    the 2029/30 share, and a single shape, common to all categories, is set so
    the 2028/29 entrants clear at that auction's 60 GBP/kW.
    """
    curves = cm.read_demand_curves()
    shares = {}
    points = {}
    for auction, price in ((T4_2930, 27.10), (T4_2829, 60.00)):
        table = cm.entered_by_category(register, auction)
        for category, row in table.iterrows():
            if row["entered_mw"] > 0:
                points.setdefault(category, []).append((price, row["awarded_mw"] / row["entered_mw"]))
                if auction == T4_2930:
                    shares[category] = row["awarded_mw"] / row["entered_mw"]
    x30 = 27.10 / 75.0

    def distributions(shape):
        out = {}
        for category, share in shares.items():
            base = x30 ** shape
            floor = min(max((share - base) / (1 - base), 0.0), share)
            out[category] = (floor, shape)
        return out

    entered_2829 = cm.entered_by_category(register, T4_2829)["entered_mw"]
    best = None
    for shape in np.exp(np.linspace(np.log(0.1), np.log(20.0), 600)):
        price = cm.clear_auction(entered_2829, distributions(shape), curves["T-4 2028/29"])["price"]
        if best is None or abs(price - 60.0) < best[1]:
            best = (shape, abs(price - 60.0))
    return distributions(best[0]), points, float(best[0])


def auction_model(register, fleet: pd.DataFrame | None = None, agreements: pd.DataFrame | None = None,
                  cpi: pd.Series | None = None, levels: dict | None = None) -> dict:
    """The fitted bid distributions, the two held-auction checks and, given the fleet, the 2030/31 T-4.

    Without ``fleet`` (the 2050 build, which needs only the distributions) the
    ``t4_2030_31`` block is left out; the distributions and checks are the same.

    Args:
        register: :func:`capacity_market.read_register` output.
        fleet: :func:`read_fleet_2030` rows (the CP30 2030 fleet).
        agreements: :func:`capacity_market.read_agreements` output (for the 2030/31 contract book).
        cpi: ONS CPI.
        levels: :func:`dsr_levels` output (default: built from ``register``).

    Returns:
        ``distributions``, ``common_shape``, ``fit_points``, ``checks``, ``notes`` and,
        with a fleet, ``t4_2030_31`` (one entry per :data:`CASES_2030` case).
    """
    distributions, points, shape = fit_distributions(register)
    curves = cm.read_demand_curves()
    checks = {}
    for name, auction, actual in (("T-4 2029/30", T4_2930, 27.10), ("T-4 2028/29", T4_2829, 60.00)):
        table = cm.entered_by_category(register, auction)
        result = cm.clear_auction(table["entered_mw"], distributions, curves[name])
        checks[name] = {"modelled_price": round(result["price"], 2), "actual_price": actual,
                        "modelled_awarded_mw": round(result["awarded_mw"], 1),
                        "actual_awarded_mw": round(float(table["awarded_mw"].sum()), 1)}
    out = {"distributions": {c: {"floor_share": round(f, 4), "shape": round(s, 4)} for c, (f, s) in distributions.items()},
           "common_shape": round(shape, 4),
           "fit_points": {c: [[p, round(s, 4)] for p, s in v] for c, v in points.items()},
           "checks": checks,
           "notes": ["Both held T-4s cleared on their demand curve. Category shares are anchored on the 2029/30 T-4 "
                     "and one common shape is set on the 2028/29 price, so both are reproduced by construction: "
                     "a calibration, not a test.",
                     "T-1 auctions (1.0 GW set aside for 2029/30; 1.0 GW assumed for 2030/31) are priced at "
                     "the same delivery year's T-4 price."]}
    if fleet is None:
        return out
    agreements = cm.read_agreements(register) if agreements is None else agreements
    cpi = cm.read_cpi() if cpi is None else cpi
    book = cm.contract_book(agreements, "2030/2031", cpi, PRICE_YEAR)
    netted, matching = net_contract_book(fleet, book)
    levels = dsr_levels(register) if levels is None else levels
    shares = refurbishing_shares(register)
    derating = cm50.read_derating()
    cases = {}
    for label, change in CASES_2030.items():
        settings = replace(Supply2030(), **change)
        rows = supply_rows(netted, shares, derating, levels, settings)
        cleared = clear_2030_31(rows, distributions, curves["T-4 2030/31"])
        cases[label] = {"changes": change, **cleared}
    out["t4_2030_31"] = cases
    out["contract_book_matching_2030_31"] = matching
    out["dsr_levels_mw"] = {k: round(v["derated_mw"], 1) for k, v in levels.items()}
    out["dsr_level_sources"] = {k: v["source"] for k, v in levels.items()}
    out["refurbishing_shares"] = {k: round(v, 4) for k, v in shares.items()}
    out["notes"] += [
        "User decision (6 Oct 2026): the 2030/31 T-4 supply is the CP30 2030 fleet the model builds "
        "(the cp30_2030_cp30_network register's asset inputs, the CP30 cables, the CP30 ES1 capacity path for build years), "
        "not the register's 2029/30 entrants; the old new-build pipeline cases (0x, 1x, 2x) are retired.",
        "The 2030/31 prequalification (autumn 2026) is not yet published. The price on the curve moves about "
        "33 GBP/kW per GW of price-taking supply.",
    ] + list(SUPPLY_DECLARED)
    return out


# ── 2030/31 T-4 supply: the CP30 2030 fleet (user decision, 6 Oct 2026) ─────
#
# The supply is the fleet the model builds, classified as the 2050 auction
# (``capacity_market_2050.classify_fleet``) classifies its own: units on a live
# low-carbon contract cannot enter, everything else enters by MR4 category at
# NESO's 2030/31 T-4 de-rating factors. 2030 differs from 2050 in four ways,
# each declared in ``SUPPLY_DECLARED``: a contract book that already holds 19.2 GW
# of 2030/31 agreements is netted off first (the DESNZ target is net of it); the
# existing fleet splits into existing and refurbishing at the 2029/30 T-4
# entrants' shares (that auction's rules, under which much existing gas entered
# as refurbishing, carry to 2030/31); build years come from the CP30 ES1 capacity
# path, because the register dates few of its 2030 rows; and the model has no
# DSR, so DSR is the register's.

FLEET_ASSETS = ASSET_INPUTS
FLEET_INTERCONNECTORS = Path("resources/interconnectors/CP30_FFR_2030_interconnectors_mapped.csv")
FLEET_ES1 = Path("resources/CP30/CP30_ES1.csv")
#: CP30's cable set (15 Oct 2026; ``config/defaults/method/brownfield.yaml interconnectors.cp30_composition``): each mapped cable
#: at CP30's nominal rating, MaresConnect added from 2030, the Isle of Man cable left out, no pro-rata
#: uplift; the network build applies the same file (``interconnectors/add_to_network.py``).
FLEET_COMPOSITION = Path("data/interconnectors/cp30_interconnector_composition.csv")
CP30_PATHWAY = "Further Flex and Renewables"
#: The 2030/31 delivery year starts on 1 October 2030; new build is plant commissioned in the
#: four calendar years to then (2027-2030), the 2050 auction's ``new_build_lead_years``.
DELIVERY_START_YEAR = 2030
NEW_BUILD_LEAD_YEARS = 4
#: ``interconnectors/add_to_network.py`` sizes each foreign market generator at 1.1 x its cables.
EU_MARKET_HEADROOM = 1.1

#: Register (asset input) carrier -> (netting group, NESO 2030/31 de-rating class, rate key of
#: ``cm_revenue_by_technology.csv``). "storage" de-rates by duration; groups match the
#: contract book's carriers (``BOOK_GROUP``) and the CP30 ES1 sub-types (``ES1_SUBTYPES``).
FLEET_CARRIERS = {
    "CCGT": ("gas", "Combined Cycle Gas Turbine (CCGT)", "CCGT"),
    "OCGT": ("gas", "Open Cycle Gas Turbine (OCGT)", "OCGT"),
    "gas_engine": ("gas", "Reciprocating engines", "gas_engine"),
    "CHP": ("CHP", "Combined Heat and Power (CHP)", "CHP"),
    "oil": ("oil", "Oil-fired steam generators", "oil"),
    "nuclear": ("nuclear", "Nuclear", "nuclear"),
    "biomass": ("biomass", "Biomass", "biomass"),
    "CCS Biomass": ("biomass", "Biomass", "biomass"),
    "CCS Gas": ("CCS Gas", "Combined Cycle Gas Turbine (CCGT)", "CCGT"),
    "waste_to_energy": ("waste_to_energy", "Energy from Waste", "waste_to_energy"),
    "large_hydro": ("large_hydro", "Hydro", "large_hydro"),
    "Battery": ("battery", "storage", "battery"),
    "LAES": ("long_duration_storage", "storage", "long_duration_storage"),
    "pumped_hydro": ("long_duration_storage", "storage", "long_duration_storage"),
    "interconnector": ("interconnector", None, "interconnector"),      # class per cable (CABLE_CLASS)
}
#: Register carriers outside the auction whatever their label, and why.
FLEET_OUTSIDE = {
    "wind_onshore": "merchant_renewable", "wind_offshore": "merchant_renewable", "solar_pv": "merchant_renewable",
    "marine": "merchant_renewable", "biogas": "small_supported_biogas", "EU_market": "replaced_by_cables",
    # Consumer flexibility (F5, scripts/demand/cp30_flexibility.py) is demand, not fleet: DSR is the
    # auction's own category, the rest is household load shifting.
    **{c: "consumer_flexibility" for c in ("EV_smart_charging", "V2G_fleet", "HP_flexibility", "DH_flexibility",
                                           "SH_flexibility", "IC_DSR", "residential_DSR")},
}
#: Contract-book carrier (``technology_carrier``) -> netting group.
BOOK_GROUP = {"CCGT": "gas", "OCGT": "gas", "gas_engine": "gas", "CHP": "CHP", "oil": "oil", "nuclear": "nuclear",
              "biomass": "biomass", "waste_to_energy": "waste_to_energy", "large_hydro": "large_hydro",
              "battery": "battery", "long_duration_storage": "long_duration_storage",
              "interconnector": "interconnector", "dsr": "dsr", "wind_onshore": "merchant_renewable",
              "wind_offshore": "merchant_renewable", "solar_pv": "merchant_renewable"}
#: Netting group -> CP30 ES1 capacity sub-types (the build-year path).
ES1_SUBTYPES = {"gas": ["Gas"], "CHP": ["Gas CHP"], "oil": ["Fuel Oil", "Diesel"],
                "nuclear": ["Nuclear - Large", "Nuclear - SMR"], "biomass": ["Biomass", "Biomass CHP", "CCS Biomass"],
                "waste_to_energy": ["Waste", "Waste CHP"], "large_hydro": ["Hydro"], "battery": ["Battery"],
                "long_duration_storage": ["LDES"], "interconnector": ["Interconnectors"]}
#: CP30 cable (``interconnectors_mapped.csv`` or composition name) -> NESO 2030/31 de-rating class; None has
#: no class. NESO publishes no MaresConnect factor for 2030/31 (it has not prequalified); it takes the Republic
#: of Ireland factor of Greenlink (0.61), the newer of the two Irish cables (EWIC 0.60) and its closest analogue,
#: an Irish Sea HVDC cable from Wales to the SEM. The Isle of Man cable is outside CP30's count and not built.
CABLE_CLASS = {"Britned": "BritNED (Netherlands)", "IFA": "IFA (France)", "IFA2": "IFA2 (France)",
               "Nemo Link": "NEMO (Belgium)", "NS Link": "NSL (Norway)", "Moyle": "Moyle (Northern Ireland)",
               "ElecLink": "Eleclink (France)", "Viking": "VikingLink (Denmark)",
               "East West Interconnector": "EWIC (Republic of Ireland)", "Greenlink": "Greenlink (Republic of Ireland)",
               "NeuConnect Interconnector": "NeuConnect (Germany)", "MaresConnect": "Greenlink (Republic of Ireland)",
               "Isle of Man Interconnector": None}
#: Support labels whose units cannot hold a CM agreement while the contract is live (as 2050).
NON_CM_SUPPORT = cm50.NON_CM_SCHEMES
#: Price-taking category new LDES on Ofgem's cap and floor enters as; it bids as MR4's existing plant.
LDES_CATEGORY = "ldes_cap_and_floor"
BID_DISTRIBUTION_OF = {LDES_CATEGORY: "existing"}
#: NESO's lowest recent 2 h storage factor (2027/28 T-4, 0.1539) over its 2030/31 one (0.2231).
STORAGE_EFC_LOW = round(0.1539 / 0.2231, 4)
#: DSR at peak in 2030, FES 2025 Holistic Transition FLX1 (the 2050 auction's DSR source).
DSR_FES_FILE = cm50.FLEET_FILE
DSR_FES_CARRIERS = ("DSR_industrial_commercial", "DSR_residential")

SUPPLY_DECLARED = (
    "Eligibility follows the 2050 auction: units on a live CfD, RO, DPA or RAB label (the register's 2030 support "
    "label) cannot enter; merchant plant, storage, interconnectors and DSR can. HPC (CfD) is out and Sizewell B "
    "(merchant) in; Drax's biomass units (bridging CfD to 2031), MGT (investment contract), Drax BECCS (CfD) and Net "
    "Zero Teesside (DPA) are out.",
    "Merchant wind, solar and marine stay outside the auction (2030 differs from 2050, where they enter): 0.15 GW "
    "de-rated of wind and solar entered the 2029/30 T-4 against 1.4 GW the CP30 merchant fleet would offer, and the "
    "2050 auction's own 2030/31 calibration counts them outside the CM.",
    "Biogas (CP30 'Other Renewables', 1.07 GW of small AD, landfill and sewage gas engines) stays outside: the "
    "register's merchant label on these undated generic rows is a default, most such plant is on the RO or FiTs, "
    "which bar entry, and the CM rate table has no biogas key to pay it.",
    "Capacity already holding a 2030/31 agreement (the register's contract book, 19.2 GW de-rated) is netted off in "
    "connection MW by technology group before de-rating: an agreement from a T-4 for 2027/28 or later held by new "
    "build nets against the fleet's 2027-2030 build, every other against its older plant, with any excess spilling "
    "to the other vintage; book capacity beyond the eligible fleet is reported, not netted (the wind and solar "
    "agreements, whose plant stays outside the auction). DSR's levels are already net of multi-year awards.",
    "New build is plant commissioned in 2027-2030, from the CP30 ES1 capacity path by technology (additions kept, "
    "falls retiring the oldest plant first). It is placed on the register's new-build rows first, then on supported "
    "(ineligible) rows, then on eligible older rows: the plant CP30 adds in those years is mostly contracted.",
    "Existing plant splits into existing and refurbishing at each technology's 2029/30 T-4 entrant shares, net of "
    "that auction's multi-year awards.",
    "New LDES (CP30's 2029-2030 LDES build, the model's LAES) is on Ofgem's cap and floor. Centrally it enters as a "
    "price-taker (MR4's existing-plant bids, as the 2050 auction assumes), its CM revenue netting against its floor; "
    "the ldes_excluded case leaves it out. Existing pumped hydro is merchant plant.",
    "Interconnectors enter per cable at NESO's 2030/31 cable factors, at the model's ratings: CP30's cable set "
    "(data/interconnectors/cp30_interconnector_composition.csv, 15 Oct 2026), each cable at CP30's nominal rating "
    "with no pro-rata uplift, 12.45 GW in 2030 as in the network build. MaresConnect (750 MW, from 2030) is new "
    "build and takes Greenlink's Republic of Ireland factor (0.61; NESO publishes none for it); the Isle of Man "
    "cable is outside CP30's count and not built.",
    "Storage de-rates by its duration on NESO's 2030/31 storage table (batteries 1.78 h at 0.199); storage_efc_low "
    f"scales every storage factor by {STORAGE_EFC_LOW} (NESO's 2027/28 T-4 2 h factor over its 2030/31 one), as the "
    "2050 storage_efc_scale does.",
    "The model has no DSR. Central DSR is the register's 2029/30 T-4 entrants net of multi-year awards; low is the "
    "2028/29 T-4's on the same basis (DSR entry before the 2029/30 jump); high is FES 2025 Holistic Transition I&C "
    "plus residential DSR at peak in 2030 (FLX1) at NESO's DSR factor, the 2050 auction's DSR source.",
)


@dataclass(frozen=True)
class Supply2030:
    """The declared choices of the 2030/31 T-4 supply (every default is the central case)."""

    storage_efc_scale: float = 1.0
    ldes_cap_and_floor: str = "price_taker"     # or "excluded"
    dsr: str = "central"                        # "low" or "high"


#: The 2030/31 T-4 cases (user decision, 6 Oct 2026): the genuinely uncertain supply inputs.
CASES_2030 = {
    "central": {},
    "storage_efc_low": {"storage_efc_scale": STORAGE_EFC_LOW},
    "ldes_excluded": {"ldes_cap_and_floor": "excluded"},
    "dsr_low": {"dsr": "low"},
    "dsr_high": {"dsr": "high"},
}


def read_es1_capacity(path=FLEET_ES1, pathway: str = CP30_PATHWAY) -> pd.DataFrame:
    """CP30 ES1 capacity (MW) by sub-type and year, transmission and distribution summed."""
    es1 = pd.read_csv(path)
    rows = es1[es1["Pathway"].eq(pathway) & es1["Variable"].eq("Capacity (MW)")]
    years = [c for c in es1.columns if str(c).isdigit() and int(c) <= DELIVERY_START_YEAR]
    table = rows.groupby("SubType")[years].sum(min_count=1)
    if table.empty:
        raise ValueError(f"{path}: no CP30 capacity rows for {pathway}")
    return table


def added_since(path: pd.Series, first_year: int) -> float:
    """MW alive in the path's last year that was built from ``first_year`` on.

    A rise is new build in that year; a fall retires the oldest plant first. The
    path starts in 2023 with unknown ages, so no plant retires on age alone: CP30's
    own falls are its retirements.
    """
    path = path.dropna().sort_index()
    years = [int(y) for y in path.index]
    stock = [(years[0], float(path.iloc[0]))]
    for year, value in zip(years[1:], path.iloc[1:]):
        change = float(value) - sum(mw for _, mw in stock)
        if change >= 0:
            stock.append((year, change))
            continue
        for i, (built, mw) in enumerate(stock):
            take = min(mw, -change)
            stock[i] = (built, mw - take)
            change += take
            if change >= -1e-9:
                break
    return float(sum(mw for built, mw in stock if built >= first_year))


def new_build_by_group(es1: pd.DataFrame) -> dict:
    """MW commissioned 2027-2030 and alive in 2030, by netting group (sub-types summed)."""
    first = DELIVERY_START_YEAR - NEW_BUILD_LEAD_YEARS + 1
    out = {}
    for group, subtypes in ES1_SUBTYPES.items():
        found = [s for s in subtypes if s in es1.index]
        out[group] = float(sum(added_since(es1.loc[s].rename(index=int), first) for s in found))
    return out


def read_fleet_2030(assets=FLEET_ASSETS, cables=FLEET_INTERCONNECTORS, es1=FLEET_ES1,
                    pathway: str = CP30_PATHWAY, composition=FLEET_COMPOSITION) -> pd.DataFrame:
    """The CP30 2030 fleet as auction rows: the register's units, the cables, build years and eligibility.

    Args:
        assets: The cp30_2030_cp30_network register's asset inputs (support label, carrier, MW, MWh, vintage).
        cables: The CP30 build's mapped interconnectors (cable identity and rating).
        es1: CP30 ES1 (the capacity path that dates the build, and the 12.45 GW cable total).
        pathway: CP30 pathway.
        composition: CP30's cable composition (:data:`FLEET_COMPOSITION`), as in the network build.

    Returns:
        One row per unit or cable: ``asset_id, carrier, group, derating_class, rate_key,
        capacity_mw, duration_hours, support_type, finance_vintage, eligible, outside_reason,
        new_share`` (the share of the row commissioned 2027-2030).
    """
    table = pd.read_csv(assets, low_memory=False)
    table = table[table["component_type"].isin(["Generator", "StorageUnit"])]
    unknown = sorted(set(table["carrier"]) - set(FLEET_CARRIERS) - set(FLEET_OUTSIDE))
    if unknown:
        raise ValueError(f"register carriers with no 2030/31 CM treatment: {unknown}")
    capacity = pd.to_numeric(table["capacity_mw"], errors="coerce").fillna(0.0)
    energy = pd.to_numeric(table["energy_capacity_mwh"], errors="coerce")
    storage = table["component_type"].eq("StorageUnit")
    units = pd.DataFrame({
        "asset_id": table["asset_id"].astype(str), "carrier": table["carrier"].astype(str),
        "capacity_mw": capacity,
        "duration_hours": np.where(storage, energy / capacity.where(capacity > 0), np.nan),
        "support_type": table["support_type"].fillna("merchant").astype(str),
        "finance_vintage": table["finance_vintage"].astype(str),
    })
    es1_table = read_es1_capacity(es1, pathway)
    units = pd.concat([units[~units["carrier"].eq("EU_market")], _cable_rows(cables, es1_table, table, composition)],
                      ignore_index=True)
    spec = units["carrier"].map(FLEET_CARRIERS)
    units["group"] = spec.map(lambda s: s[0] if isinstance(s, tuple) else None)
    units["derating_class"] = units["derating_class"].fillna(spec.map(lambda s: s[1] if isinstance(s, tuple) else None))
    units["rate_key"] = spec.map(lambda s: s[2] if isinstance(s, tuple) else None)
    reason = pd.Series(np.where(units["support_type"].isin(sorted(NON_CM_SUPPORT)), "live_low_carbon_contract", None),
                       index=units.index)
    reason = reason.where(reason.notna(), units["carrier"].map(FLEET_OUTSIDE))
    reason = reason.where(reason.notna(), np.where(units["derating_class"].isna(), "no_cm_class", None))
    units["outside_reason"] = reason
    units["eligible"] = units["outside_reason"].isna()
    units["new_share"] = _new_shares(units, new_build_by_group(es1_table))
    return units


def _cable_rows(cables, es1_table: pd.DataFrame, assets: pd.DataFrame, composition=FLEET_COMPOSITION
                ) -> pd.DataFrame:
    """The CP30 cables at the model's ratings, checked against the register's foreign market generators.

    The cable set is the network build's (15 Oct 2026): each mapped cable at CP30's nominal rating or
    left out, plus the cables the composition adds (MaresConnect), summing to the ES1 total with nothing
    scaled (``storage_interconnectors.compose_cp30_interconnectors``); an added cable is new build.
    """
    from scripts.CP30.storage_interconnectors import (
        compose_cp30_interconnectors, load_cp30_interconnector_composition)

    target = float(es1_table.loc["Interconnectors", str(DELIVERY_START_YEAR)])
    mapped = compose_cp30_interconnectors(pd.read_csv(cables), target, load_cp30_interconnector_composition(composition),
                                          DELIVERY_START_YEAR).reset_index(drop=True)
    vintage = pd.Series(np.where(mapped["cp30_composition_action"].eq("add"), "new_build", "legacy"),
                        index=mapped.index)
    unknown = sorted(set(mapped["name"]) - set(CABLE_CLASS))
    if unknown:
        raise ValueError(f"interconnectors with no NESO de-rating class mapping: {unknown}")
    rating = mapped["capacity_mw"].astype(float)
    # The register's EU_market generators are the same cables at 1.1x, one per foreign bus.
    foreign = assets[assets["carrier"].eq("EU_market")].groupby("bus")["capacity_mw"].sum() / EU_MARKET_HEADROOM
    built = rating.groupby(mapped["to_bus"]).sum()
    gap = (built - foreign.reindex(built.index)).abs()
    if foreign.empty or gap.isna().any() or float(gap.max()) > 1.0:
        raise ValueError(f"CP30 cables do not reproduce the register's foreign market generators: {gap.to_dict()}")
    return pd.DataFrame({
        "asset_id": mapped["name"].astype(str), "carrier": "interconnector", "capacity_mw": rating,
        "duration_hours": np.nan, "support_type": "cap_and_floor", "finance_vintage": vintage,
        "derating_class": mapped["name"].map(CABLE_CLASS),
    })


def _new_shares(units: pd.DataFrame, new_mw: dict) -> pd.Series:
    """Each row's share commissioned 2027-2030: a group's new MW on its new-build rows, then supported, then the rest."""
    share = pd.Series(0.0, index=units.index)
    for group, rows in units.groupby("group"):
        if group not in new_mw:
            # No ES1 path (CCS Gas, CP30's committed overlay): the register's vintage stands.
            share[rows.index] = rows["finance_vintage"].eq("new_build").astype(float)
            continue
        remaining = min(float(new_mw[group]), float(rows["capacity_mw"].sum()))
        tiers = (rows["finance_vintage"].eq("new_build"),
                 ~rows["finance_vintage"].eq("new_build") & ~rows["eligible"],
                 ~rows["finance_vintage"].eq("new_build") & rows["eligible"])
        for tier in tiers:
            mw = float(rows.loc[tier, "capacity_mw"].sum())
            if remaining <= 0 or mw <= 0:
                continue
            take = min(remaining, mw)
            share[rows.index[tier]] = take / mw
            remaining -= take
    return share


def net_contract_book(fleet: pd.DataFrame, book: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Take capacity already holding a 2030/31 agreement out of the eligible fleet (connection MW).

    Args:
        fleet: :func:`read_fleet_2030` rows.
        book: The 2030/31 contract book (:func:`capacity_market.contract_book`).

    Returns:
        The fleet with ``free_new_mw`` and ``free_existing_mw`` (eligible MW not holding an
        agreement), and the matching by group (book MW new and existing, held, unmatched).
    """
    book = book.assign(group=book["technology"].map(technology_carrier).map(BOOK_GROUP),
                       connection_mw=pd.to_numeric(book["connection_mw"], errors="coerce").fillna(0.0))
    first = DELIVERY_START_YEAR - NEW_BUILD_LEAD_YEARS + 1
    fresh = book["category"].eq("new_build") & (book["auction"].str[:4].astype(int) >= first)
    out = fleet.copy()
    eligible = out["eligible"]
    out["fleet_new_mw"] = np.where(eligible, out["capacity_mw"] * out["new_share"], 0.0)
    out["fleet_existing_mw"] = np.where(eligible, out["capacity_mw"] * (1.0 - out["new_share"]), 0.0)
    out["free_new_mw"] = out["fleet_new_mw"]
    out["free_existing_mw"] = out["fleet_existing_mw"]
    matching = {}
    # DSR has no fleet rows: its supply (``dsr_levels``) is already net of the multi-year awards.
    for group in sorted((set(book["group"].dropna()) | set(out.loc[eligible, "group"].dropna())) - {"dsr"}):
        rows = out["group"].eq(group)
        e_new, e_old = float(out.loc[rows, "fleet_new_mw"].sum()), float(out.loc[rows, "fleet_existing_mw"].sum())
        in_group = book["group"].eq(group)
        b_new = float(book.loc[in_group & fresh, "connection_mw"].sum())
        b_old = float(book.loc[in_group & ~fresh, "connection_mw"].sum())
        held_new, held_old = min(e_new, b_new), min(e_old, b_old)
        # Excess on one vintage spills to the other (the register's vintage is not a commissioning date).
        held_old_spill = min(e_old - held_old, b_new - held_new)
        held_new_spill = min(e_new - held_new, b_old - held_old)
        held_new, held_old = held_new + held_new_spill, held_old + held_old_spill
        if e_new > 0:
            out.loc[rows, "free_new_mw"] *= 1.0 - held_new / e_new
        if e_old > 0:
            out.loc[rows, "free_existing_mw"] *= 1.0 - held_old / e_old
        matching[group] = {"book_new_mw": round(b_new, 1), "book_existing_mw": round(b_old, 1),
                           "fleet_eligible_new_mw": round(e_new, 1), "fleet_eligible_existing_mw": round(e_old, 1),
                           "held_mw": round(held_new + held_old, 1),
                           "book_beyond_fleet_mw": round(b_new + b_old - held_new - held_old, 1)}
    return out, matching


def _t4_entrants(register, auction: str) -> pd.DataFrame:
    """An auction's entrants by (category, carrier), MW de-rated, net of the multi-year agreements it awarded."""
    rows = register[register["Auction Name"].eq(auction)].drop_duplicates("CMU ID").copy()
    rows["carrier"] = rows["technology"].map(technology_carrier)
    awarded = rows["Capacity Agreement Awarded"].eq("Yes")
    entered = rows["CMU confirmed entry into Auction"].eq("Yes") | awarded
    multi = awarded & (pd.to_numeric(rows["Duration of Capacity Agreement Awarded"], errors="coerce") > 1)
    keys = ["category", "carrier"]
    supply = rows[entered].groupby(keys)["capacity_mw"].sum().sub(
        rows[multi].groupby(keys)["obligation_mw"].sum(), fill_value=0.0)
    return supply.rename("supply_mw").reset_index()


def refurbishing_shares(register) -> dict:
    """Each netting group's refurbishing share of its existing-plant entrants in the 2029/30 T-4."""
    entrants = _t4_entrants(register, T4_2930)
    entrants["group"] = entrants["carrier"].map(BOOK_GROUP)
    old = entrants[entrants["category"].isin(["existing", "refurbishing"])]
    table = old.pivot_table(index="group", columns="category", values="supply_mw", aggfunc="sum").fillna(0.0)
    total = table.sum(axis=1)
    refurb = table["refurbishing"] if "refurbishing" in table else total * 0.0
    return {g: float(refurb[g] / total[g]) for g in table.index if total[g] > 0}


def dsr_levels(register, fes_file=DSR_FES_FILE, derating: dict | None = None) -> dict:
    """The three DSR volumes (MW de-rated): central, low and high, with their sources."""
    derating = cm50.read_derating() if derating is None else derating
    out = {}
    for label, auction in (("central", T4_2930), ("low", T4_2829)):
        entrants = _t4_entrants(register, auction)
        out[label] = {"derated_mw": float(entrants.loc[entrants["category"].eq("dsr"), "supply_mw"].sum()),
                      "source": f"NESO CM register, {auction}: DSR entrants net of multi-year awards"}
    fes = pd.read_csv(fes_file)
    at_peak = fes[fes["quantity"].eq("at_peak_mw") & fes["carrier"].isin(DSR_FES_CARRIERS)]
    if len(at_peak) != len(DSR_FES_CARRIERS):
        raise ValueError(f"{fes_file}: expected one at-peak row for each of {DSR_FES_CARRIERS}")
    factor = float(derating["classes"]["DSR"])
    out["high"] = {"derated_mw": float(at_peak[str(DELIVERY_START_YEAR)].astype(float).sum()) * factor,
                   "source": f"FES 2025 Holistic Transition FLX1 I&C plus residential DSR at peak, "
                             f"{DELIVERY_START_YEAR} ({fes_file}), x NESO 2030/31 DSR factor {factor}"}
    return out


def supply_rows(netted: pd.DataFrame, shares: dict, derating: dict, levels: dict,
                settings: Supply2030) -> pd.DataFrame:
    """The 2030/31 T-4 supply: one row per (unit, category), MW de-rated, plus DSR.

    Args:
        netted: :func:`net_contract_book` rows.
        shares: :func:`refurbishing_shares`.
        derating: :func:`capacity_market_2050.read_derating` (2030/31 T-4 factors).
        levels: :func:`dsr_levels`.
        settings: The case.

    Returns:
        ``asset_id, carrier (rate key), group, category, free_mw, derating_factor, supply_mw``.
    """
    rows = netted[netted["eligible"]].copy()
    storage = rows["derating_class"].eq("storage")
    factor = pd.Series(np.nan, index=rows.index)
    factor[storage] = [cm50.storage_factor(h, derating["storage"]) * float(settings.storage_efc_scale)
                       for h in rows.loc[storage, "duration_hours"]]
    factor[~storage] = rows.loc[~storage, "derating_class"].map(derating["classes"])
    if factor.isna().any():
        raise ValueError(f"no 2030/31 de-rating factor for {sorted(rows.loc[factor.isna(), 'derating_class'].unique())}")
    rows["derating_factor"] = factor
    parts = []
    cable = rows["group"].eq("interconnector")
    parts.append(rows[cable].assign(category="interconnector", free_mw=rows["free_existing_mw"] + rows["free_new_mw"]))
    plant = rows[~cable]
    refurb = plant["group"].map(shares).fillna(0.0)
    parts.append(plant.assign(category="existing", free_mw=plant["free_existing_mw"] * (1.0 - refurb)))
    parts.append(plant.assign(category="refurbishing", free_mw=plant["free_existing_mw"] * refurb))
    ldes = plant["group"].eq("long_duration_storage")
    parts.append(plant[~ldes].assign(category="new_build", free_mw=plant["free_new_mw"]))
    if settings.ldes_cap_and_floor == "price_taker":
        # Its own rate key (user decision, 6 Oct 2026): the award is paid to the new LDES rows alone.
        parts.append(plant[ldes].assign(category=LDES_CATEGORY, free_mw=plant["free_new_mw"],
                                        rate_key=cm.LDES_CAP_AND_FLOOR_RATE_KEY))
    elif settings.ldes_cap_and_floor != "excluded":
        raise ValueError(f"ldes_cap_and_floor must be price_taker or excluded, got {settings.ldes_cap_and_floor!r}")
    out = pd.concat(parts, ignore_index=True)
    out = out[out["free_mw"] > 1e-9]
    out["supply_mw"] = out["free_mw"] * out["derating_factor"]
    out = out[["asset_id", "rate_key", "group", "category", "support_type", "free_mw", "derating_factor", "supply_mw"]]
    out = out.rename(columns={"rate_key": "carrier"})
    dsr = float(levels[settings.dsr]["derated_mw"])
    out = pd.concat([out, pd.DataFrame([{"asset_id": f"dsr_{settings.dsr}", "carrier": "dsr", "group": "dsr",
                                         "category": "dsr", "support_type": "merchant", "free_mw": np.nan,
                                         "derating_factor": np.nan, "supply_mw": dsr}])], ignore_index=True)
    return out


def clear_2030_31(rows: pd.DataFrame, distributions: dict, curve: cm.DemandCurve) -> dict:
    """Clear the 2030/31 T-4 on its demand curve and split the award by category and carrier.

    Args:
        rows: :func:`supply_rows`.
        distributions: The fitted bid distributions ``{category: (floor, shape)}``.
        curve: The DESNZ 2030/31 demand curve.

    Returns:
        Price, awarded MW, shortfall, supply by category and by (category, carrier), award by category.
    """
    bids = {**distributions, **{c: distributions[d] for c, d in BID_DISTRIBUTION_OF.items()}}
    supply = rows.groupby("category")["supply_mw"].sum()
    supply = supply[supply > 0]
    missing = sorted(set(supply.index) - set(bids))
    if missing:
        raise ValueError(f"no bid distribution for categories {missing}")
    result = cm.clear_auction(supply, bids, curve)
    awarded = cm.awarded_by_category(supply, bids, float(result["price"]), float(result["awarded_mw"]), curve.price_cap)
    by_carrier = rows.groupby(["category", "carrier"])["supply_mw"].sum()
    # Connection MW behind each (category, carrier)'s supply (DSR has none): a carrier with no
    # 2029/30 agreement is paid per kW of what it wins here (``revenue_by_technology``).
    connection = (rows.dropna(subset=["free_mw"]).groupby(["category", "carrier"])["free_mw"].sum()
                  if "free_mw" in rows.columns else pd.Series(dtype=float))
    return {"entered_mw": {k: round(float(v), 1) for k, v in supply.items()},
            "entered_total_mw": round(float(supply.sum()), 1),
            "entered_by_carrier_mw": {c: {k: float(v) for k, v in by_carrier[c].items() if v > 0}
                                      for c in supply.index},
            "entered_connection_by_carrier_mw": {c: {k: float(v) for k, v in connection[c].items() if v > 0}
                                                 for c in supply.index if c in connection.index.get_level_values(0)},
            "entered_by_rate_key_mw": {k: round(float(v), 1) for k, v in rows.groupby("carrier")["supply_mw"].sum().items()},
            "price": round(float(result["price"]), 2), "awarded_mw": round(float(result["awarded_mw"]), 1),
            "shortfall": bool(result.get("shortfall", False)),
            "awarded_by_category_mw": {k: round(float(v), 1) for k, v in awarded.items()}}


def supply_table_2030_31(register, fleet: pd.DataFrame, agreements: pd.DataFrame, cpi: pd.Series,
                         levels: dict | None = None, case: str = "central") -> pd.DataFrame:
    """One case's auction rows beside every fleet row left out and why (``cm_t4_2030_31_supply.csv``)."""
    book = cm.contract_book(agreements, "2030/2031", cpi, PRICE_YEAR)
    netted, _ = net_contract_book(fleet, book)
    levels = dsr_levels(register) if levels is None else levels
    rows = supply_rows(netted, refurbishing_shares(register), cm50.read_derating(), levels,
                       replace(Supply2030(), **CASES_2030[case]))
    held = netted["fleet_new_mw"] + netted["fleet_existing_mw"] - netted["free_new_mw"] - netted["free_existing_mw"]
    left_out = netted.assign(held_by_agreement_mw=held)
    left_out = left_out[~left_out["eligible"] | (held > 1e-9)]
    left_out = left_out.assign(category=np.where(left_out["eligible"], "holds_2030_31_agreement", "outside"),
                               carrier=left_out["rate_key"].fillna(left_out["carrier"]))
    cols = ["asset_id", "carrier", "group", "category", "support_type", "outside_reason", "capacity_mw",
            "new_share", "held_by_agreement_mw"]
    return pd.concat([rows.assign(case=case), left_out[cols].assign(case=case)], ignore_index=True)


# ── calendar 2030 ────────────────────────────────────────────────────────────

def calendar_2030(agreements, cpi, model) -> tuple[dict, pd.DataFrame]:
    books = {dy: cm.contract_book(agreements, dy, cpi, PRICE_YEAR) for dy in cm.CALENDAR_2030_WEIGHTS}
    idx_2425 = cpi[PRICE_YEAR] / cm.cpi_financial_year(cpi, 2024)
    idx_2526 = cpi[PRICE_YEAR] / cm.cpi_financial_year(cpi, 2025)
    p3031 = model["t4_2030_31"]["central"]["price"]
    q3031 = model["t4_2030_31"]["central"]["awarded_mw"]
    parts = {
        "2029/2030": {"contract_book_gbp": float(books["2029/2030"]["payment_gbp"].sum()),
                      "t1_gbp": T1_VOLUME_MW * 1e3 * 27.10 * idx_2425},
        "2030/2031": {"contract_book_gbp": float(books["2030/2031"]["payment_gbp"].sum()),
                      "t4_modelled_gbp": q3031 * 1e3 * p3031 * idx_2526,
                      "t1_gbp": T1_VOLUME_MW * 1e3 * p3031 * idx_2526},
    }
    for dy, block in parts.items():
        block["total_gbp"] = sum(block.values())
        block["weight_in_calendar_2030"] = cm.CALENDAR_2030_WEIGHTS[dy]
    total = sum(b["total_gbp"] * b["weight_in_calendar_2030"] for b in parts.values())
    sensitivity = {}
    for label, case in model["t4_2030_31"].items():
        dy = parts["2030/2031"]["contract_book_gbp"] + (case["awarded_mw"] + T1_VOLUME_MW) * 1e3 * case["price"] * idx_2526
        sensitivity[label] = 0.5 * parts["2029/2030"]["total_gbp"] + 0.5 * dy
    # The central modelled T-4 volume at the 2029/30 T-4 clearing price instead (its T-1 as modelled).
    at_2930 = (0.5 * parts["2029/2030"]["total_gbp"]
               + 0.5 * (parts["2030/2031"]["total_gbp"] - parts["2030/2031"]["t4_modelled_gbp"]
                        + q3031 * 1e3 * 27.10 * idx_2526))
    book = pd.concat(books.values(), ignore_index=True)
    central = model["t4_2030_31"]["central"]
    summary = {
        "price_year": PRICE_YEAR,
        "calendar_2030_cost_gbp": total,
        "delivery_years": parts,
        "t4_2030_31_central_price_gbp_per_kw_year": p3031,
        "t4_2030_31_central": {k: central[k] for k in ("price", "awarded_mw", "shortfall", "entered_total_mw",
                                                       "entered_mw", "entered_by_rate_key_mw", "awarded_by_category_mw")},
        "calendar_2030_cost_by_t4_case_gbp": sensitivity,
        "t4_2030_31_price_by_case_gbp_per_kw_year": {k: v["price"] for k, v in model["t4_2030_31"].items()},
        "calendar_2030_cost_t4_at_2029_30_clearing_gbp": at_2930,
        "previous_bill_basis_gbp": 27.10 * 40108.608 * 1e3,
        "method": ("Contract book: every live agreement in the NESO Capacity Market Register for each delivery year, "
                   "at its auction's clearing price carried from the auction's base financial year to 2025 with "
                   "ONS CPI (D7BT). Delivery year 2029/30 adds the 1.0 GW T-1 at the T-4 price; 2030/31 adds the "
                   "modelled T-4 and a 1.0 GW T-1 at that price. The modelled T-4 clears the fitted bid "
                   "distributions on the DESNZ 2030/31 demand curve against the CP30 2030 fleet the model builds "
                   "(user decision, 6 Oct 2026): units without a live low-carbon contract, net of capacity already "
                   "holding 2030/31 agreements, at NESO's 2030/31 T-4 de-rating factors, plus DSR from the "
                   "register. Calendar 2030 takes half of each delivery year, because the supplier charge is levied "
                   "on November-to-February weekday peak demand."),
    }
    return summary, book


# ── household share (MR4.4) ──────────────────────────────────────────────────

def ofgem_domestic_peak_factor(agreements, cpi, year: int = 2024, ledger=OFGEM_LEDGER, espeni=ESPENI) -> dict:
    """Ofgem's CM allowance per 3,100 kWh over a flat energy share of the same year's CM cost.

    The supplier charge is levied on 16:00-19:00 weekday demand, November to
    February, where household demand is concentrated, so Ofgem's cap allows a
    benchmark household more than its annual energy share. The ratio for
    ``year`` (register contract book for the two delivery years in ``year``
    money, over ESPENI GB demand, the same basis as the bill's final demand)
    is the factor applied to the 2030 household share.
    """
    rows = pd.read_csv(ledger)
    rows = rows[rows["component_id"].astype(str).str.startswith("capacity")]
    months = {"Q1": 3, "Q2": 3, "Q3": 3, "Q4": 3, "Q2_Q3": 6}
    weights = rows["cap_period"].astype(str).str.replace(f"{year}_", "", regex=False).map(months)
    by_period = rows.assign(w=weights).groupby("cap_period").agg(amount=("amount_gbp", "mean"), w=("w", "first"))
    allowance = float((by_period["amount"] * by_period["w"]).sum() / by_period["w"].sum())
    books = [cm.contract_book(agreements, dy, cpi, price_year=year) for dy in (f"{year - 1}/{year}", f"{year}/{year + 1}")]
    cost = 0.5 * books[0]["payment_gbp"].sum() + 0.5 * books[1]["payment_gbp"].sum()
    col_t, col_mw = "ELEC_elex_startTime[utc](datetime)", "ELEC_POWER_TOTAL_ESPENI[MW](float32)"
    demand = pd.read_csv(espeni, usecols=[col_t, col_mw])
    stamp = pd.to_datetime(demand[col_t], utc=True)
    demand_mwh = float(demand.loc[stamp.dt.year.eq(year), col_mw].sum() * 0.5)
    flat = cost / demand_mwh * 3.1
    return {"year": year, "ofgem_allowance_gbp_per_3100kwh_nominal": allowance, "cm_cost_gbp_nominal": float(cost),
            "espeni_demand_mwh": demand_mwh, "flat_share_gbp_per_3100kwh": float(flat), "factor": allowance / flat,
            "source": f"{ledger} (Ofgem cap model CM allowance, direct debit single rate, 14 regions); register "
                      f"contract book; {espeni}"}


# ── assets ───────────────────────────────────────────────────────────────────

def modelled_t4_by_carrier(model: dict, case: str = "central") -> pd.Series:
    """The modelled 2030/31 T-4 award (MW) by rate-key carrier (item B7).

    The award is split across categories by what each clears at the modelled
    price (``capacity_market.awarded_by_category``), then within a category in
    proportion to each carrier's share of that category's supply (the CP30 fleet
    rows of ``supply_rows``, carried in the model as ``entered_by_carrier_mw``).
    """
    return _modelled_t4_by_category_carrier(model, case).groupby(level="carrier").sum()


def _modelled_t4_by_category_carrier(model: dict, case: str = "central") -> pd.Series:
    """The modelled 2030/31 T-4 award (MW de-rated) by (category, carrier)."""
    result = model["t4_2030_31"][case]
    supply = pd.Series({(c, k): v for c, by in result["entered_by_carrier_mw"].items() for k, v in by.items()},
                       dtype=float)
    supply.index.names = ["category", "carrier"]
    by_category = supply.groupby(level="category").sum()
    distributions = {c: (d["floor_share"], d["shape"]) for c, d in model["distributions"].items()}
    distributions.update({c: distributions[d] for c, d in BID_DISTRIBUTION_OF.items()})
    curve = cm.read_demand_curves()["T-4 2030/31"]
    awarded = cm.awarded_by_category(by_category, distributions, float(result["price"]),
                                     float(result["awarded_mw"]), curve.price_cap)
    category = supply.index.get_level_values("category")
    return supply / by_category.reindex(category).to_numpy() * awarded.reindex(category).to_numpy()


def modelled_t4_connection_by_carrier(model: dict, case: str = "central") -> pd.Series:
    """The modelled 2030/31 T-4 award in connection MW by carrier: each (category, carrier)'s
    de-rated award times its supply's connection-to-de-rated ratio (zero where none is carried)."""
    award = _modelled_t4_by_category_carrier(model, case)
    result = model["t4_2030_31"][case]
    connection = pd.Series({(c, k): v for c, by in (result.get("entered_connection_by_carrier_mw") or {}).items()
                            for k, v in by.items()}, dtype=float)
    supply = pd.Series({(c, k): v for c, by in result["entered_by_carrier_mw"].items() for k, v in by.items()},
                       dtype=float)
    ratio = (connection.reindex(award.index) / supply.reindex(award.index)).fillna(0.0)
    return (award * ratio.to_numpy()).groupby(level="carrier").sum()


def revenue_by_technology(book: pd.DataFrame, model: dict, cpi, register=None) -> pd.DataFrame:
    """Calendar-2030 CM revenue per kW of contracted connection capacity, by model carrier.

    Item B7 (26 Sep 2026): the carriers' revenue adds up to the bill's calendar-2030
    total. Each delivery year is its contract book plus the volumes the bill adds
    to it, allocated to carriers: the modelled 2030/31 T-4 by what each carrier
    brings to the auction (``modelled_t4_by_carrier``), and each T-1 in proportion
    to that delivery year's T-4 awards. The earlier version re-contracted only the
    2029/30 one-year holders, leaving GBP133m of the bill's total with no payee.
    """
    if register is None:
        register = cm.read_register()
    idx_2425 = cpi[PRICE_YEAR] / cm.cpi_financial_year(cpi, 2024)
    idx_2526 = cpi[PRICE_YEAR] / cm.cpi_financial_year(cpi, 2025)
    p3031 = model["t4_2030_31"]["central"]["price"]
    book = book.assign(carrier=book["technology"].map(technology_carrier),
                       connection_mw=pd.to_numeric(book["connection_mw"], errors="coerce"))
    dy1 = book[book["delivery_year"].eq("2029/2030")]
    dy2 = book[book["delivery_year"].eq("2030/2031")]
    # T-1 2029/30 in proportion to the carriers' 2029/30 T-4 awards.
    t4_2930 = register[register["Auction Name"].eq(T4_2930) & register["Capacity Agreement Awarded"].eq("Yes")]
    t4_2930 = t4_2930.drop_duplicates("CMU ID")
    t4_2930_mw = t4_2930.groupby(t4_2930["technology"].map(technology_carrier))["obligation_mw"].sum()
    t1_2930_mw = T1_VOLUME_MW * t4_2930_mw / t4_2930_mw.sum()
    # T-4 2030/31 modelled (on the CP30 fleet), and its T-1 in proportion to it.
    t4_3031_mw = modelled_t4_by_carrier(model)
    t1_3031_mw = T1_VOLUME_MW * t4_3031_mw / t4_3031_mw.sum()
    t4_3031_connection = modelled_t4_connection_by_carrier(model)
    carriers = sorted(set(book["carrier"]) | set(t4_3031_mw.index) | set(t1_2930_mw.index))
    rows = []
    for carrier in carriers:
        a = dy1[dy1["carrier"].eq(carrier)]
        t1_a = float(t1_2930_mw.get(carrier, 0.0)) * 1e3 * 27.10 * idx_2425
        t4_b = float(t4_3031_mw.get(carrier, 0.0)) * 1e3 * p3031 * idx_2526
        t1_b = float(t1_3031_mw.get(carrier, 0.0)) * 1e3 * p3031 * idx_2526
        dy1_gbp = a["payment_gbp"].sum() + t1_a
        dy2_gbp = dy2[dy2["carrier"].eq(carrier)]["payment_gbp"].sum() + t4_b + t1_b
        revenue = 0.5 * dy1_gbp + 0.5 * dy2_gbp
        connection = a["connection_mw"].sum()
        basis = "contract_book_2029_30"
        t4_mw = float(t4_3031_mw.get(carrier, 0.0))
        if not connection > 0 and t4_mw > 0 and float(t4_3031_connection.get(carrier, 0.0)) > 0:
            # User decision (6 Oct 2026): a carrier holding no 2029/30 agreement (new LDES on the
            # cap and floor) is paid per kW of what it wins in 2030/31, T-4 and T-1, so the
            # register's units of that carrier earn exactly that award.
            connection = (float(t4_3031_connection[carrier]) * (t4_mw + float(t1_3031_mw.get(carrier, 0.0))) / t4_mw)
            basis = "award_2030_31"
        rows.append({"carrier": carrier, "derated_mw_2029_30": a["mw"].sum(), "connection_mw_2029_30": connection,
                     "connection_basis": basis,
                     "revenue_2029_30_gbp": dy1_gbp, "revenue_2030_31_gbp": dy2_gbp,
                     "t1_2029_30_mw": float(t1_2930_mw.get(carrier, 0.0)),
                     "t4_2030_31_modelled_mw": float(t4_3031_mw.get(carrier, 0.0)),
                     "t1_2030_31_mw": float(t1_3031_mw.get(carrier, 0.0)),
                     "calendar_2030_revenue_gbp": revenue,
                     "gbp_per_kw_connection_year": revenue / (connection * 1e3) if connection > 0 else np.nan})
    return pd.DataFrame(rows)


def match_gas_stations(agreements: pd.DataFrame, components_path=cm.REGISTER_COMPONENTS,
                       schedule_path=GAS_SCHEDULE, radius_km: float = 3.0) -> pd.DataFrame:
    """Gas agreements live in 2029/30 matched to the named 2030 gas fleet by location."""
    from pyproj import Transformer

    live = agreements[agreements["delivery_years"].map(lambda y: "2029/2030" in y)].copy()
    live["carrier"] = live["technology"].map(technology_carrier)
    live = live[live["carrier"].isin(["CCGT", "OCGT", "gas_engine", "CHP"])]
    comps = pd.read_csv(components_path, encoding="utf-8-sig", low_memory=False)
    comps = comps.drop_duplicates("CMU ID", keep="last").set_index("CMU ID")
    live["grid_ref"] = live["cmu_id"].map(comps["OS Grid Reference"])
    en = live["grid_ref"].map(cm.os_grid_to_easting_northing)
    to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lonlat = [to_wgs.transform(*p) if p else (np.nan, np.nan) for p in en]
    live["lon"], live["lat"] = [p[0] for p in lonlat], [p[1] for p in lonlat]
    schedule = pd.read_csv(schedule_path)
    out = []
    for _, row in live.iterrows():
        match, dist = None, np.nan
        if row["lat"] == row["lat"]:
            d = _haversine_km(row["lat"], row["lon"], schedule["lat"].to_numpy(), schedule["lon"].to_numpy())
            j = int(np.argmin(d))
            if d[j] <= radius_km:
                match, dist = schedule.iloc[j]["station_name"], float(d[j])
        out.append({"agreement_id": row["agreement_id"], "cmu_id": row["cmu_id"], "cmu_name": row["cmu_name"],
                    "carrier": row["carrier"], "derated_mw": row["mw"], "auction": row["auction"],
                    "clearing_price": row["clearing_price"], "term_years": row["term_years"],
                    "grid_ref": row["grid_ref"], "station_name": match, "distance_km": dist})
    return pd.DataFrame(out)


def _haversine_km(lat, lon, lats, lons):
    lat, lon, lats, lons = map(np.radians, (lat, lon, lats, lons))
    a = np.sin((lats - lat) / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def revenue_adequacy(tech: pd.DataFrame, cpi, settlement=ASSET_SETTLEMENT, inputs=ASSET_INPUTS) -> pd.DataFrame:
    """Energy margin + CM against going-forward (existing) and full new-build fixed costs, per gas class."""
    costs = pd.read_csv(GAS_COSTS).set_index("carrier")
    to_2025 = cpi[PRICE_YEAR] / cpi[2024]
    crf = MERCHANT_WACC / (1 - (1 + MERCHANT_WACC) ** -costs["operating_life_years"])
    margins = energy_margins(settlement, inputs)
    rows = []
    for carrier, c in costs.iterrows():
        going_forward = (c["fixed_fee_gbp_per_kw_year"] + c["insurance_share_of_capex"] * c["capital_cost_gbp_per_kw"]) * to_2025
        capital = (c["capital_cost_gbp_per_kw"] + c["development_cost_gbp_per_kw"]) * crf[carrier] * to_2025
        cm_rate = float(tech.set_index("carrier")["gbp_per_kw_connection_year"].get(carrier, np.nan))
        margin = margins.get(carrier, {})
        energy = margin.get("gbp_per_kw_year", np.nan)
        rows.append({"carrier": carrier, "model_capacity_mw": margin.get("capacity_mw", np.nan),
                     "energy_margin_gbp_per_kw_year": energy, "capacity_market_gbp_per_kw_year": cm_rate,
                     "going_forward_fixed_cost_gbp_per_kw_year": going_forward,
                     "new_build_capital_gbp_per_kw_year": capital,
                     "net_existing_gbp_per_kw_year": energy + cm_rate - going_forward,
                     "net_new_build_gbp_per_kw_year": energy + cm_rate - going_forward - capital,
                     "cost_source": GAS_COST_SOURCE})
    return pd.DataFrame(rows)


def energy_margins(settlement=ASSET_SETTLEMENT, inputs=ASSET_INPUTS) -> dict:
    """Wholesale + balancing revenue less fuel and carbon, per kW, from the solved asset settlement."""
    if not (Path(settlement).exists() and Path(inputs).exists()):
        return {}
    settle = pd.read_csv(settlement)
    inputs = pd.read_csv(inputs)
    # The settlement and the register name assets differently (the register splits
    # legacy and new-build cohorts), so capacity is taken by carrier.
    capacity = inputs[inputs["component_type"].eq("Generator")].groupby("carrier")["capacity_mw"].sum()
    out = {}
    for carrier, g in settle.groupby("carrier"):
        cap = float(capacity.get(carrier, 0.0))
        if cap > 0:
            margin = (g["market_revenue_gbp"] - g["fuel_carbon_gbp"]).sum()
            out[carrier] = {"capacity_mw": cap, "gbp_per_kw_year": float(margin / (cap * 1e3))}
    return out


def main_2050(args, out: Path, cpi) -> None:
    """Task Y11: the calendar-2050 CM on the FES 2025 HT fleet (``capacity_market_2050.run_2050``).

    The bid distributions are MR4's, refitted here from the register exactly as the
    2030 build fits them, so the 2050 rule reads no 2030 output.
    """
    from scripts.market import capacity_market_2050 as cm50

    if args.model_year != cm50.MODEL_YEAR:
        raise ValueError(f"--model-year {args.model_year}: only 2030 and {cm50.MODEL_YEAR} are built")
    if args.extract_fleet:
        table = cm50.extract_fes_trajectory()
        table.to_csv(cm50.FLEET_FILE, index=False)
        print(f"wrote {cm50.FLEET_FILE} ({len(table)} rows)")
        return
    register = cm.read_register()
    model = auction_model(register)
    result = cm50.run_2050(model, cpi)
    summary, revenue = result["summary"], result["revenue"]
    summary["bid_distributions"] = model["distributions"]
    year = cm50.MODEL_YEAR
    (out / f"cm_calendar_{year}_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    revenue.to_csv(out / f"cm_revenue_by_technology_{year}.csv", index=False)
    result["classified"].to_csv(out / f"cm_fleet_{year}.csv", index=False)
    print("calendar %d CM cost (2025 GBP m): %.0f at %.2f GBP/kW/yr (target %.0f MW, supply %.0f MW)" % (
        year, summary[f"calendar_{year}_cost_gbp"] / 1e6, summary["price_gbp_per_kw_year_2025"],
        summary["auction"]["target_mw"], summary["auction"]["supply_total_mw"]))
    for label, case in summary["cases"].items():
        print("  %-22s price %6.2f  cost %8.0f m  shortfall %s" % (
            label, case["price_gbp_per_kw_year_auction_money"], case[f"calendar_{year}_cost_gbp"] / 1e6,
            case["shortfall"]))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--adequacy", nargs=3, metavar=("SETTLEMENT", "ASSET_INPUTS", "OUT_CSV"),
                        help="write only the revenue-adequacy table, from cm_revenue_by_technology.csv in --out")
    parser.add_argument("--model-year", type=int, default=2030,
                        help="2030 (default, item MR4) or 2050 (task Y11, capacity_market_2050.py)")
    parser.add_argument("--extract-fleet", action="store_true",
                        help="with --model-year 2050: re-extract the FES 2025 trajectory table and stop")
    # User decision (6 Oct 2026): the 2030/31 T-4 supply is the CP30 2030 fleet.
    parser.add_argument("--fleet-assets", default=str(FLEET_ASSETS), help="the cp30_2030_cp30_network register's asset inputs")
    parser.add_argument("--fleet-interconnectors", default=str(FLEET_INTERCONNECTORS),
                        help="the CP30 build's mapped interconnectors")
    parser.add_argument("--fleet-es1", default=str(FLEET_ES1), help="CP30 ES1 (capacity path, build years)")
    parser.add_argument("--fleet-composition", default=str(FLEET_COMPOSITION),
                        help="CP30's cable composition, as in the network build (15 Oct 2026)")
    parser.add_argument("--dsr-fes", default=str(DSR_FES_FILE), help="FES 2025 HT FLX1 table (the high DSR case)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cpi = cm.read_cpi()
    if args.model_year != 2030:
        main_2050(args, out, cpi)
        return
    if args.adequacy:
        settlement, inputs, target = args.adequacy
        tech = pd.read_csv(out / "cm_revenue_by_technology.csv")
        adequacy = revenue_adequacy(tech, cpi, settlement, inputs)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        adequacy.to_csv(target, index=False)
        print(adequacy.drop(columns="cost_source").round(2).to_string(index=False))
        return
    static = cm.read_auction_static()
    register = cm.read_register()
    agreements = cm.read_agreements(register, static)
    composition = args.fleet_composition
    fleet = read_fleet_2030(args.fleet_assets, args.fleet_interconnectors, args.fleet_es1, composition=composition)
    levels = dsr_levels(register, args.dsr_fes)
    model = auction_model(register, fleet, agreements, cpi, levels)
    supply = supply_table_2030_31(register, fleet, agreements, cpi, levels)
    summary, book = calendar_2030(agreements, cpi, model)
    summary["fleet_sources"] = {"assets": str(args.fleet_assets), "interconnectors": str(args.fleet_interconnectors),
                                "interconnector_composition": str(composition),
                                "es1": f"{args.fleet_es1} ({CP30_PATHWAY})", "dsr_high": str(args.dsr_fes)}
    tech = revenue_by_technology(book, model, cpi, register)
    # Item B7: every pound of the bill's Capacity Market total has a payee.
    if not np.isclose(tech["calendar_2030_revenue_gbp"].sum(), summary["calendar_2030_cost_gbp"], rtol=1e-9):
        raise ValueError(
            f"CM revenue by technology ({tech['calendar_2030_revenue_gbp'].sum():,.0f}) does not add up to the "
            f"calendar-2030 cost ({summary['calendar_2030_cost_gbp']:,.0f})")
    unpaid = tech.loc[tech["connection_mw_2029_30"].fillna(0.0).le(0.0), "calendar_2030_revenue_gbp"].sum()
    summary["revenue_by_technology_note"] = (
        "cm_revenue_by_technology.csv allocates the whole calendar-2030 cost to carriers (item B7): contract books, "
        "the modelled 2030/31 T-4 by what each carrier brings to it, and each T-1 pro rata to its delivery year's "
        f"T-4 awards. GBP{unpaid / 1e6:,.1f}M falls on carriers with no contracted connection capacity, "
        "which the per-kW rate cannot carry. New LDES on the cap and floor has its own row "
        f"({cm.LDES_CAP_AND_FLOOR_RATE_KEY}, user decision 6 Oct 2026): its 2030/31 award, per kW of the connection "
        "it wins (connection_basis award_2030_31), which the register's LAES rows take in place of "
        "long_duration_storage's, so existing pumped hydro keeps the long-duration contract book.")
    matches = match_gas_stations(agreements)

    cols = ["delivery_year", "agreement_id", "cmu_id", "cmu_name", "auction", "technology", "category", "connection",
            "connection_mw", "mw", "term_years", "clearing_price", "base_year", "indexation", "payment_gbp"]
    book[cols].to_csv(out / "cm_contract_book_2030.csv", index=False)
    (out / "cm_auction_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    summary["gas_station_matches"] = {"agreements": int(len(matches)),
                                      "matched": int(matches["station_name"].notna().sum()),
                                      "matched_derated_mw": float(matches.loc[matches["station_name"].notna(), "derated_mw"].sum()),
                                      "gas_derated_mw": float(matches["derated_mw"].sum())}
    summary["gas_cost_source"] = GAS_COST_SOURCE
    summary["household_peak_factor"] = ofgem_domestic_peak_factor(agreements, cpi)
    (out / "cm_calendar_2030_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tech.to_csv(out / "cm_revenue_by_technology.csv", index=False)
    matches.to_csv(out / "cm_station_matches.csv", index=False)
    supply.to_csv(out / "cm_t4_2030_31_supply.csv", index=False)

    print(json.dumps(model["checks"], indent=1))
    print(json.dumps({k: {x: v[x] for x in ("entered_total_mw", "price", "awarded_mw", "shortfall")}
                      for k, v in model["t4_2030_31"].items()}, indent=1))
    print("central supply by category (MW de-rated):", model["t4_2030_31"]["central"]["entered_mw"])
    print("central supply by carrier (MW de-rated):", model["t4_2030_31"]["central"]["entered_by_rate_key_mw"])
    print("calendar 2030 CM cost (2025 GBP m): %.0f  [previous basis %.0f]" % (
        summary["calendar_2030_cost_gbp"] / 1e6, summary["previous_bill_basis_gbp"] / 1e6))
    for dy, b in summary["delivery_years"].items():
        print("  ", dy, {k: round(v / 1e6, 1) if k.endswith("gbp") else v for k, v in b.items()})
    print("  by T-4 case:", {k: round(v / 1e6) for k, v in summary["calendar_2030_cost_by_t4_case_gbp"].items()})
    print(summary["gas_station_matches"])
    print("household peak factor:", round(summary["household_peak_factor"]["factor"], 4))
    pd.set_option("display.width", 220)
    print(tech.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
