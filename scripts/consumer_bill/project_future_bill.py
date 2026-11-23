"""Project a granular future bill from complete hourly model results and assumptions.

This is a scenario sensitivity, not a statutory tariff or validated forecast.
Network and retail allowances remain explicit external inputs. Every
PyPSA-derived component states its physical quantity, price source, recovery
denominator, price basis and calendar alignment.

Components derived from the model:

* wholesale energy, weighted by a domestic consumption profile rather than by
  total system load, with the difference shown as an explicit shaping row;
* CfD difference payments on balancing dispatch, settled against published
  allocation-round strike prices indexed to the model price basis, and
  recovered over CfD eligible supplier sales;
* BSUoS constraint costs, being accepted offer and bid cost (plus the CfD
  curtailment compensation only when the balancing LP priced CfD bids at a
  scalar; under contract-value bids it is already inside the accepted bid cost
  and is only reconciled), recovered over final demand;
* the regulated pass-through of interconnector congestion rent under Ofgem's
  cap and floor regime (and IFA's use-of-revenues framework), credited through
  the TNUoS demand residual;
* the LDES cap-and-floor top-up (item D4(c), 26 Sep 2026): the private case's
  floor payment to long-duration storage, read from the public account the
  finance register builds from model market revenue, so the forecast bill and
  the ownership projector's private case carry the one private headline.

Recovery denominators come from one sourced file per case so that no component
invents its own. Levies are recovered on METERED supply (user decision D1, 26
Sep 2026): the paying volume is GB electrical load plus the electricity the
electrolysers draw, divided by the GSP-to-meter loss factor. The hydrogen
offtake (MWh of hydrogen, not electricity) pays nothing and does not weight the
system price; electrolysers pay every per-MWh line on their draw.

F18 (user decisions, 6 Oct 2026) moves a recovery file that declares
``recovery_basis: loss_inclusive_two_volume`` (the 2030 file) to two volumes,
because the model's load already contains network losses: metered supply (the RO,
FiT, the battery top-up, the Exchequer's per-household key) and the GSP charging
volume (the CfD family, LDES, the Capacity Market and BSUoS), with Ofgem's household
loss factor on every line charged at a GSP rate. The summary carries each line's
recovery (``line_recovery``, :mod:`scripts.consumer_bill.line_recovery`), the only
conversion between household and system pounds. A file without the declaration (2050)
stays on item D1.

F12 (11 Oct 2026) adds an opt-in price level, the bill case's ``price_level``: ``model``
(the default, the model's own Stage 1 price) or ``cp32``, the same hours quantile-mapped
onto CP30's 2030 duration curve (CP.32) with the whole year ranked together. Under
``cp32`` the wholesale and procurement lines, the CfD top-ups and the interconnector
rents are on the mapped price; volumes and the constraint line are held as solved. The
summary records the level, the mapping and which lines moved (:data:`CP32_REPRICED_LINES`,
:data:`CP32_HELD_LINES`).

F24.2 (24 Oct 2026, user decisions) moves the constraint line's GB-price parts to the level
(:func:`price_level_balancing_record`). The CfD contract-value bids (strike less the Stage 1
reference) are re-priced on the mapped reference by the price-level balancing-cost extraction
(``extract_future_balancing_costs.cfd_bid_change_at_price_level``), unit for unit as the
twin's asset settlement moves each unit's balancing settlement, so the bill-to-register
identities close. The interconnector re-trading is re-valued too: it is valued wholly on the
foreign side (trade value at the IC1 asks and bids plus the counter-trade premium), which the
level holds as it does for the rents, so it keeps its value. The other accepted bids and offers
stay held, the Stage 1-indexed storage prices among them (first order about GBP -1.3 m on the
central twin).
"""

import calendar
import hashlib
import html
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from scripts.market.ofgem_price_cap import demand_weighted_price, require_full_calendar_year
from scripts.market.consumer_bill_breakdown import GROUPS
from scripts.consumer_bill.bill_groupings import annotate as annotate_bill_groups, summary_block as bill_group_totals
from scripts.consumer_bill.bill_groupings import group_of as bill_group_of
from scripts.consumer_bill.line_recovery import (
    check_volumes as check_line_recovery_volumes, inherited_entry as inherited_line_recovery,
    make_entry as make_line_recovery_entry,
)
from scripts.market.revenue_tracking import compute_cfd_payments, negative_price_eligibility, _lookup_strike_price
from scripts.market.cfd_strike_table import CFD_TECHNOLOGY_COLUMN, MODELLED_ROUND, load_strike_table, strike_overrides
from scripts.consumer_bill.build_domestic_demand_profile import peak_mask
from scripts.consumer_bill.extract_future_balancing_costs import check_negative_price_rule
from scripts.market.market_accounting import _is_gb_electrical_bus
from scripts.consumer_bill import price_level_mapping as plm
from scripts.market.turn_down_allocation import balancing_dispatch_in_force  # pro rata turn-down, 20 Oct 2026

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Item F12 (11 Oct 2026): the wholesale price level the bill settles on, from the bill
# case's ``price_level``. ``model`` (the code default, the central bill) is the model's own
# Stage 1 hourly price. ``cp32`` re-prices the same hours onto CP30's own 2030 duration
# curve (CP.32, Further Flex and Renewables, mean about GBP 69.9/MWh) by quantile mapping
# (:mod:`scripts.consumer_bill.price_level_mapping`): the whole year is ranked together,
# never month by month, and every hour takes the curve's price at its rank. Volumes and
# the balancing stage are held as solved, so the result is a static bracket on the price
# level, not a forecast. The mapping settings are the case's ``price_level_mapping`` block
# over the module's defaults (which a test holds equal to config/defaults/base/consumer_bill.yaml).
PRICE_LEVELS = ("model", "cp32")
DEFAULT_PRICE_LEVEL = "model"
# Prices within this band of zero map as one tied block (12 Oct 2026): about a third of the
# model's hours clear at zero to within solver noise, which would otherwise rank them.
CP32_ZERO_BAND_GBP_PER_MWH = 1.0
# Under ``cp32``: the bill lines whose value moves with the mapped hourly price, and the
# ones held at the model's own price level, each with why (reported in the summary).
CP32_REPRICED_LINES = {
    "direct_energy": "the domestic energy cost on the mapped hourly price",
    "shaping_adjustment": "the domestic shaping on the mapped hourly price (method kept)",
    "procurement_*": "the procurement allowances, intensities on the mapped energy cost",
    "cfd": "difference payments: strike less the mapped reference (hourly, and the baseload seasonal "
           "reference recomputed from it); the negative-price rule applies to the mapped reference",
    "allocated_ccs": "the Power BECCS CfD's electricity leg, settled with every other CfD",
    "interconnector_cap_and_floor_pass_through": (
        "each cable's congestion rent moves by efficiency x sum(hours x (mapped - model GB price) x Stage 1 p0); "
        "the foreign side (the trade cost the LP charged) is held as solved"),
}
CP32_HELD_LINES = {
    "bsuos_constraints": (
        "held as solved (items F12, F24.2), apart from the CfD contract-value bids and the interconnector "
        "re-trading (see the repriced lines): the other accepted offers and bids are the balancing stage's, at the "
        "model's price. Gas, ROC and merchant bids carry no GB price. The storage bids and offers indexed to the "
        "Stage 1 price do, and are held by decision (user, 24 Oct 2026): first order about GBP -1.3 m a year on "
        "the central twin, GBP -0.01 a household"),
    "renewables_obligation": "no wholesale price in it: ROCs valued at the buy-out price from the RO scheme summary",
    "ldes_cap_and_floor": "the public account's floor top-up, built from the finance register's market revenue "
                          "at the model's price (re-priced instead when the account is a price-level twin "
                          "run group's: price_level_account_record)",
    "dpa": "the 2050 support settlement's DPA payment (no 2030 line), at the model's price",
    "sourced externals": "network, levy, Capacity Market and supplier lines are external inputs with no model price",
    "scarcity_markup": "report only: the MR3 mark-up inside the model's Stage 1 price",
}
# Item F24.2 (24 Oct 2026): a month's interconnector re-trading (``analyze_market``, per cable) is
# the BM schedule's trade value less the Stage 1 schedule's, plus the counter-trade premium; the
# sign each part enters with. :func:`retrading_at_price_level` re-values it part by part.
RETRADING_COMPONENTS = {"balancing_trade_cost_GBP": 1.0, "day_ahead_trade_cost_GBP": -1.0,
                        "counter_trade_premium_GBP": 1.0}
# GBP a month's parts may sit from its re-trading total (sums of per-cable floats).
RETRADING_COMPONENT_TOLERANCE_GBP = 1.0


EXTERNAL_COMPONENTS = {
    "tnuos": ("network", "TNUoS"), "duos": ("network", "DUoS (aggregate estimate)"),
    "feed_in_tariffs": ("policy", "Feed-in Tariffs"), "capacity_market": ("system_balancing", "Capacity Market"),
    "eco": ("policy", "ECO"), "warm_home_discount": ("policy", "Warm Home Discount"),
    "smart_meter_net_cost_change": ("supplier_operating", "Smart-meter net cost change"),
    "aahedc": ("policy", "AAHEDC"), "sizewell_c_rab": ("policy", "Sizewell C RAB"),
    "remaining_ccs": ("policy", "Remaining CCS levy"),
    "bics_levy": ("policy", "Network Charging Compensation levy (British Industry Supercharger)"), "nuclear_contract_counterparty": ("policy", "Nuclear contract counterparty"),
    "supplier_costs_and_margin": ("supplier_operating", "Supplier costs and margin (aggregate estimate)"),
}


# ``model_linked_rules`` entries the projector actually applies, and those it
# only carries as documentation. Validating the set exactly stops a rule being
# declared in the assumptions file while no code reads it: ``bsuos`` once
# described a redispatch-indexed allowance that was never applied, so the file
# claimed a method the bill did not use.
APPLIED_MODEL_RULES = {"direct_fuel", "cfd", "bsuos_constraints", "renewables_obligation",
                       "interconnector_cap_and_floor", "ldes_cap_and_floor"}
DOCUMENTED_MODEL_RULES = set()
#: Rules a bill may declare or leave out (FES 2050 task SETTLE2050): ``dpa`` is the 2050
#: line for the gas CCS and hydrogen-power agreements, which no 2030 bill carries.
OPTIONAL_MODEL_RULES = {"dpa"}

# Sourced outturn externals the bill cannot be assembled without. Withdrawing
# the flat GBP91 BSUoS allowance leaves the non-constraint half of BSUoS with no
# home unless it is supplied here, and the plan is explicit that the projector
# must refuse to run rather than fall back to "91 minus constraints".
REQUIRED_SOURCED_EXTERNALS = {"bsuos_other_services"}

# Source classes a ``sourced_externals`` line may carry. ``external_primary_source``
# is a WS9 replacement of an analyst placeholder (an official or regulatory
# source with URL, access date and price basis); the placeholder it replaced is
# kept on the row as ``reference_gbp_per_3100kwh_ex_vat`` so the delta is visible.
SOURCED_EXTERNAL_CLASSES = {"external_sourced_outturn", "external_primary_source"}

# Reference-price bases a baseload CfD may settle against. Intermittent CfDs
# settle against the hourly day-ahead reference; baseload CfDs (Hinkley Point
# C) settle against a season-ahead baseload reference. The model has no
# forward curve, so the season-ahead price is proxied by the demand-weighted
# mean wholesale price of the season the snapshot falls in.
BASELOAD_REFERENCE_BASES = ("seasonal_demand_weighted_mean",)
SUMMER_MONTHS = (4, 5, 6, 7, 8, 9)

# Economic-curtailment rules for legacy CfDs. The wholesale LP curtails CfD
# output at ~GBP0 without payment, but contracts allocated before AR4 carry no
# zero-payment term: the generator would run through such hours and be paid
# strike minus reference. "pay_pre_ar4_at_strike" restores that payment on the
# economically curtailed energy of the listed rounds.
#
# The rule's premise is the as-built wholesale, where every CfD unit bids 0 and
# a zero-price tie decides who is curtailed. Under the contract-aware order
# (scripts/market/curtailment_order.py, 26 Sep 2026) pre-AR4 units bid minus
# their strike, so the wholesale curtails them only at a price at or below
# -strike: exactly where the real generator backs off itself. Paying them
# max(strike - price, 0) on that energy pays about twice the strike for output
# nobody would have generated, so the rule is refused on such a run
# (:func:`wholesale_curtailment_order`). It stays available for as-built runs.
ECONOMIC_CURTAILMENT_RULES = ("none", "pay_pre_ar4_at_strike")
DEFAULT_PRE_AR4_ROUNDS = ("AR1", "AR2", "AR3", "Investment Contract", "HPC")
DEFAULT_ZERO_PRICE_THRESHOLD = 1.0
# Solver noise on the wholesale price (as-built and barrier runs show -2e-6 to -9e-6
# GBP/MWh on duals that are mathematically zero): an hour counts as clearing below
# zero only below minus this (OW12, 3 Oct 2026; the final 2030 bill reported 269 such
# hours as negative at a minimum of -8.6e-6). Widened to 0.50 on 12 Oct 2026: the CP30
# Stage 1 (s1_zf) bids AR4+ at -0.01 so they still curtail first, and its surplus hours
# clear at -0.01 plus noise (to -0.01005), which a 0.01 band split at random (65 hours
# counted negative). Genuinely negative hours sit at -1 or below. Matches
# revenue_tracking.NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH.
NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH = 0.50
# Carriers the contract-aware order re-prices (curtailment_order.ORDERED_CARRIERS;
# repeated here so reading the saved network needs no market import).
CONTRACT_AWARE_CARRIERS = frozenset({"wind_offshore", "wind_onshore", "solar_pv", "marine", "nuclear"})
# A CfD renewable whose saved wholesale marginal cost is below this was re-priced
# by the contract-aware order (AR4+ at -1, older rounds at minus strike). As built
# they sit at 0 (cfd_dispatch_cost) or at the resource-cost floor (+0.01).
CONTRACT_AWARE_COST_BELOW_GBP_PER_MWH = -1e-6

# Interconnector regimes the cap-and-floor pass-through understands. Every cable
# carrying congestion rent must be declared with one, so a new cable cannot be
# dropped from (or silently credited in) the bill.
INTERCONNECTOR_REGIMES = ("cap_and_floor", "use_of_revenues_share", "exempt_merchant", "other_jurisdiction")

# Ofgem-named procurement allowances, in report order. Each is an intensity
# relative to the delivery price, so each is applied to the domestic energy cost.
# The imbalance allowance (item D4(d)) is the supplier's own cash-out exposure,
# settled through Elexon; it was once left out as "recovered through BSUoS",
# which is wrong: BSUoS pays NESO's balancing actions, not supplier imbalance.
PROCUREMENT_ALLOWANCES = (
    ("losses", "Transmission and distribution losses"),
    ("transaction_cost", "Transaction-cost uplift"),
    ("day_ahead_rehedging", "Day-ahead re-hedging / forecast error"),
    ("imbalance", "Supplier imbalance (cash-out) allowance"),
    # F18 (6 Oct 2026): Ofgem's name. For electricity the 1% is "additional risk"; there is no
    # separate basis-risk allowance. The key keeps its old name (component id procurement_additional_risk_and_basis).
    ("additional_risk_and_basis", "Additional risk allowance"),
)
# How the procurement allowances combine (``procurement_bridge.rate_composition``). The
# default applies each hedging rate to the shaped energy cost and losses to energy plus
# shaping only; ``ofgem_annex2_1b`` is Ofgem's direct-fuel formula exactly (F18, user
# decision 6 Oct 2026): index x (1 + the sum of the rates) x the loss multiplier.
PROCUREMENT_ON_SHAPED_ENERGY = "on_shaped_energy"
PROCUREMENT_OFGEM_ANNEX2 = "ofgem_annex2_1b"
PROCUREMENT_COMPOSITIONS = (PROCUREMENT_ON_SHAPED_ENERGY, PROCUREMENT_OFGEM_ANNEX2)

# Bus carriers that carry electricity; a load or link end on any other bus
# (the hydrogen bus) is not electrical demand.
ELECTRICAL_BUS_CARRIERS = frozenset({"", "ac", "dc"})


def _finite(value, name, positive=False):
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _source_rebasing_factor(assumptions, report_price_year):
    """Return the factor carrying source-published allowances to the report basis.

    The external allowances and the retail procurement reference both come from
    the published source, so they share its price year. Adding them to a
    rebased model price without stating that year silently mixes bases inside
    a total that is labelled as one.
    """
    source_price_year = int(assumptions["source"]["price_year"])
    block = assumptions.get("source_currency_rebasing")
    if source_price_year == report_price_year:
        if block:
            raise ValueError("source_currency_rebasing is set but the source is already on the report basis")
        return 1.0
    if not block:
        raise ValueError(
            f"Source allowances are quoted in {source_price_year} but the report basis is "
            f"{report_price_year}; add source_currency_rebasing to carry them across."
        )
    if int(block["from_price_year"]) != source_price_year or int(block["to_price_year"]) != report_price_year:
        raise ValueError("source_currency_rebasing does not run from the source year to the report year")
    factor = _finite(block["factor"], "Source currency factor", True)
    if not np.isclose(factor, float(block["to_index"]) / float(block["from_index"]), rtol=1e-9):
        raise ValueError("Source currency factor does not reconcile to its source indices")
    return factor


def _rebasing_factor(block, label, from_price_year, to_price_year):
    """Validate any declared index pair and return its reconciled factor."""
    if int(block["from_price_year"]) != int(from_price_year):
        raise ValueError(f"{label} does not run from price year {from_price_year}")
    if int(block["to_price_year"]) != int(to_price_year):
        raise ValueError(f"{label} does not land on price year {to_price_year}")
    factor = _finite(block["factor"], f"{label} factor", True)
    if not np.isclose(factor, float(block["to_index"]) / float(block["from_index"]), rtol=1e-9):
        raise ValueError(f"{label} factor does not reconcile to its source indices")
    return factor


def load_domestic_profile(path):
    """Read the pinned domestic shape and validate its coverage and normalisation.

    The stored file is a full year of hourly weights. Only its day-type and
    local-hour structure is used when mapping onto model snapshots, because
    that is the only structure the Ofgem base/peak shaping input carries; the
    stored calendar year is validated rather than aligned by date.
    """
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "relative_demand" not in frame.columns:
        raise ValueError("Domestic profile needs a relative_demand column")
    weights = pd.to_numeric(frame["relative_demand"], errors="raise")
    if not np.isfinite(weights.to_numpy()).all() or (weights <= 0).any():
        raise ValueError("Domestic profile weights must be finite and positive")
    years = set(frame.index.year)
    if len(years) != 1:
        raise ValueError(f"Domestic profile spans several years: {sorted(years)}")
    year = years.pop()
    expected = 8784 if calendar.isleap(year) else 8760
    if len(frame) != expected:
        raise ValueError(
            f"Domestic profile covers {len(frame)} hours, not the {expected} of {year}"
        )
    total = float(weights.sum())
    if not np.isclose(total, 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"Domestic profile weights sum to {total}, not one")
    if "in_peak_block" not in frame.columns:
        raise ValueError("Domestic profile needs an in_peak_block column to map onto model snapshots")
    in_peak = frame["in_peak_block"].astype(bool)
    peak_rates = weights[in_peak.to_numpy()].round(15).unique()
    offpeak_rates = weights[~in_peak.to_numpy()].round(15).unique()
    if len(peak_rates) != 1 or len(offpeak_rates) != 1:
        raise ValueError(
            "This profile is expected to carry exactly two rates, peak and off-peak; it carries "
            f"{len(peak_rates)} and {len(offpeak_rates)}. A richer profile needs a date- or "
            "day-type-aligned mapping rather than the two-block expansion used here."
        )
    return {
        "profile_year": int(year),
        "peak_rate": float(peak_rates[0]),
        "offpeak_rate": float(offpeak_rates[0]),
        "peak_share_of_consumption": float(weights[in_peak.to_numpy()].sum()),
        "peak_share_of_hours": float(in_peak.mean()),
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def base_peak_prices_by_quarter(hourly) -> list:
    """Per calendar quarter: time-weighted baseload and Elexon-peak-block prices and household volume.

    ``hourly`` carries ``price_gbp_per_mwh``, ``hours`` and ``domestic_weight``.
    The peak block is weekdays 07:00-19:00 local time, the product Ofgem's index
    buys (Annex 2, tab 3b Demand).
    """
    in_peak = peak_mask(pd.DatetimeIndex(hourly.index)).to_numpy()
    rows = []
    for quarter, frame in hourly.groupby(pd.DatetimeIndex(hourly.index).quarter):
        mask = in_peak[pd.DatetimeIndex(hourly.index).quarter == quarter]
        hours = frame["hours"]
        price = frame["price_gbp_per_mwh"]
        if not mask.any():
            raise ValueError(f"quarter {quarter} has no peak-block hours")
        rows.append({
            "quarter": int(quarter),
            "base_price_gbp_per_mwh": float((price * hours).sum() / hours.sum()),
            "peak_price_gbp_per_mwh": float((price[mask] * hours[mask]).sum() / hours[mask].sum()),
            "domestic_volume_weight": float((frame["domestic_weight"] * hours).sum()),
        })
    return rows


def ofgem_cap_rule_price(by_quarter, base_share: float, peak_share: float, quarter_weights=None):
    """The cap's unshaped energy price: base/peak blend per quarter, weighted across quarters.

    Returns ``(price, detail)``. Ofgem weights its quarterly index by seasonal
    PC1 demand (Annex 2, tab 3b Demand). With ``quarter_weights`` (``{1: w1, ...,
    4: w4}``, from ``direct_fuel.shaping.quarter_weights``, 6 Oct 2026) those
    weights are used; without them, the household profile's volume in each
    quarter, which has no seasonal shape, stands in.
    """
    if not np.isclose(base_share + peak_share, 1.0, rtol=0, atol=1e-12):
        raise ValueError("Ofgem base and peak shares must sum to one")
    if quarter_weights is None:
        weights = {row["quarter"]: row["domestic_volume_weight"] for row in by_quarter}
        basis = "household_profile_volume"
    else:
        weights = {int(q): float(w) for q, w in quarter_weights.items()}
        basis = "declared_quarter_weights"
        if set(weights) != {row["quarter"] for row in by_quarter}:
            raise ValueError(f"quarter_weights cover quarters {sorted(weights)}, the prices "
                             f"{sorted(row['quarter'] for row in by_quarter)}")
    total = sum(weights.values())
    detail = []
    price = 0.0
    for row in by_quarter:
        blend = base_share * row["base_price_gbp_per_mwh"] + peak_share * row["peak_price_gbp_per_mwh"]
        weight = weights[row["quarter"]] / total
        price += blend * weight
        detail.append({**row, "blend_gbp_per_mwh": blend, "quarter_weight": weight, "quarter_weight_basis": basis})
    return price, detail


# F20 phase 0: PYPSA_GB_INPUT_RESOURCES points a parity run in a worktree at another tree's resources.
SCARCITY_SUMMARY_DIR = Path(os.environ.get("PYPSA_GB_INPUT_RESOURCES") or "resources") / "analysis" / "scarcity"
SCARCITY_SUMMARY_KEYS = ("basis", "hours_scarce", "load_weighted_uplift_gbp_per_mwh", "mean_uplift_gbp_per_mwh",
                         "shallow_bands_8_to_20gw", "deep_bands_below_8gw", "consumer_cost_gbp")


def read_scarcity_summary(case_id, directory=SCARCITY_SUMMARY_DIR):
    """The scarcity mark-up in this case's Stage 1 prices, from ``scarcity_diagnostic.py``; reporting only.

    The mark-up is inside the Stage 1 price (item MR3(b)), so it is already in the
    wholesale and CfD lines; this only makes it visible. Absent a summary, the
    record says how to produce one rather than leaving the reader to assume zero.
    """
    path = Path(directory) / f"{case_id}_scarcity_summary.json"
    if not path.exists():
        return {"available": False, "expected_path": path.as_posix(),
                "how": ("python -m scripts.market.scarcity_diagnostic resources/analysis/scarcity/"
                        f"{case_id} <the twelve month scenarios> --prices-include-scarcity")}
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {"available": True, "path": path.as_posix(),
            **{key: summary.get(key) for key in SCARCITY_SUMMARY_KEYS}}


SHAPING_METHODS = ("ofgem_cap_rule", "model_profile")


def shaping_settings(direct_fuel: dict) -> dict:
    """The wholesale shaping method (``direct_fuel.shaping``); ``model_profile`` when undeclared."""
    block = dict(direct_fuel.get("shaping") or {"method": "model_profile"})
    if block.get("method") not in SHAPING_METHODS:
        raise ValueError(f"direct_fuel.shaping.method must be one of {SHAPING_METHODS}, got {block.get('method')!r}")
    if block["method"] == "ofgem_cap_rule":
        for key in ("base_share", "peak_share", "seasonal_to_monthly_rate", "monthly_to_hourly_rate"):
            block[key] = _finite(block[key], f"shaping {key}")
            if block[key] < 0:
                raise ValueError(f"shaping {key} must be non-negative")
        block.setdefault("source", "")
        block.setdefault("source_url", "")
        if block.get("quarter_weights") is not None:
            declared = block["quarter_weights"]
            weights = {int(q[1:]) if str(q).lower().startswith("q") else int(q): _finite(w, f"quarter weight {q}")
                       for q, w in (declared.get("weights") or {}).items()}
            if set(weights) != {1, 2, 3, 4} or min(weights.values()) < 0:
                raise ValueError("direct_fuel.shaping.quarter_weights.weights needs non-negative q1 to q4")
            if not np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-6):
                raise ValueError(f"direct_fuel.shaping.quarter_weights sum to {sum(weights.values())}, not one")
            block["quarter_weights"] = {**declared, "weights": weights}
    return block


def domestic_weights(snapshots, profile) -> pd.Series:
    """Expand a two-block domestic profile onto the model's snapshots."""
    in_peak = peak_mask(pd.DatetimeIndex(snapshots))
    weights = pd.Series(profile["offpeak_rate"], index=pd.DatetimeIndex(snapshots))
    weights[in_peak.to_numpy()] = profile["peak_rate"]
    return weights


# Per-process cache of the tables read from the monthly network files.  The
# ownership projector runs the projection once per case (21 cases on the CP30
# year) and every case reads the same 24 files; the tables cached here depend
# only on the files, never on the case, so they are read once.  Under 1 GB for
# the CP30 year (CfD dispatch subsets and the pre-AR4 curtailment tables).
_TABLE_CACHE: dict = {}


def clear_table_cache():
    """Drop the cached network tables (tests, or a long-lived process changing files)."""
    _TABLE_CACHE.clear()


def _cached(key, loader):
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = loader()
    return _TABLE_CACHE[key]


def read_hourly_inputs(network_path, price_path, links_path=None):
    """Read only load and weighting variables; avoid loading the full network.

    ``links_path`` is the Stage 1 link schedule (``*_wholesale_links.csv``,
    p0 per link), from which the electrolysers' electricity draw is read; the
    saved wholesale network carries no link flows.
    """
    return _cached(("hourly", str(network_path), str(price_path), str(links_path)),
                   lambda: _read_hourly_inputs(network_path, price_path, links_path)).copy()


def _electrical_buses(ds):
    """GB buses that carry electricity, or None when the file carries no bus table."""
    if "buses_i" not in ds or "buses_carrier" not in ds:
        return None
    index = pd.Index(ds["buses_i"].values).astype(str)
    buses = pd.DataFrame({"carrier": pd.Series(ds["buses_carrier"].values, index=index)})
    if "buses_country" in ds:
        buses["country"] = pd.Series(ds["buses_country"].values, index=index)
    return set(buses.index[_is_gb_electrical_bus(buses)])


def _read_hourly_inputs(network_path, price_path, links_path=None):
    """Hourly price, snapshot hours and the demand split the bill needs (item D1).

    ``demand_mw`` is the metered-supply basis at the GSP: loads on GB electrical
    buses plus the electricity the electrolysers draw. It weights the system
    price. ``total_load_mw`` is every Load's p_set, as the market summary counts
    it (it includes the hydrogen offtake, MWh of hydrogen), kept only to check
    the files agree.
    """
    with xr.open_dataset(network_path, engine="netcdf4") as ds:
        snapshots = pd.DatetimeIndex(ds["snapshots_snapshot"].values)
        weights = pd.Series(ds["snapshots_generators"].values, index=snapshots, dtype=float)
        loads = pd.Index(ds["loads_i"].values).astype(str)
        dynamic = ds["loads_t_p_set"].to_pandas() if "loads_t_p_set" in ds else pd.DataFrame(index=snapshots)
        dynamic.index = snapshots
        dynamic.columns = dynamic.columns.astype(str)
        if not np.isfinite(dynamic.to_numpy()).all():
            raise ValueError("Model load time series contains missing or non-finite values")
        static = pd.Series(ds["loads_p_set"].values, index=loads) if "loads_p_set" in ds else pd.Series(0., index=loads)
        electrical = _electrical_buses(ds)
        # Demand carried by consumer flexibility (2 Oct 2026; F5, 27 Oct 2026): storage units
        # on the demand carriers (demand_carriers.DEMAND_FLEX_STORAGE_CARRIERS: EVs, V2G, heat,
        # storage heaters, DSR, DACCS; the 2050 build and CP30 2030 with demand.build.flexibility).
        # Their energy is carved out of the Loads, and what they draw is metered supply like any
        # Load: on a solved network p_store - p_dispatch, when they buy it (charging losses
        # included, V2G and DSR net); without a dispatch, the drain's electricity
        # (market_accounting.consumer_flex_draw). It joins the GB electrical load.
        flexible = pd.Series(0.0, index=snapshots)
        if "storage_units_carrier" in ds and "storage_units_bus" in ds:
            from scripts.market.market_accounting import consumer_flex_draw

            su_index = pd.Index(ds["storage_units_i"].values).astype(str)

            def _su_series(name):
                key = f"storage_units_t_{name}"
                if key not in ds:
                    return None
                frame = ds[key].to_pandas()
                frame.index = snapshots
                frame.columns = frame.columns.astype(str)
                return frame

            carriers = pd.Series(ds["storage_units_carrier"].values, index=su_index).astype(str)
            eta = (pd.Series(ds["storage_units_efficiency_store"].values, index=su_index)
                   if "storage_units_efficiency_store" in ds else None)
            draw = consumer_flex_draw(carriers, _su_series("p_store"), _su_series("p_dispatch"), _su_series("p"),
                                      _su_series("inflow"), eta)
            if len(draw.columns):
                su_bus = pd.Series(ds["storage_units_bus"].values, index=su_index).astype(str)
                keep = (su_bus.reindex(draw.columns).isin(electrical).to_numpy() if electrical is not None
                        else np.ones(len(draw.columns), bool))
                if not np.isfinite(draw.to_numpy()).all():
                    raise ValueError("Consumer flexibility time series contain missing or non-finite values")
                flexible = draw.loc[:, keep].sum(axis=1).reindex(snapshots).fillna(0.0)
        if electrical is None or "loads_bus" not in ds:
            is_electrical = pd.Series(True, index=loads)
        else:
            is_electrical = pd.Series(ds["loads_bus"].values, index=loads).astype(str).isin(electrical)
        links = None
        if "links_i" in ds and electrical is not None:
            names = pd.Index(ds["links_i"].values).astype(str)
            links = pd.DataFrame({"bus0": ds["links_bus0"].values, "bus1": ds["links_bus1"].values},
                                 index=names).astype(str)
            bus_names = set(pd.Index(ds["buses_i"].values).astype(str))
            carrier = pd.Series(ds["buses_carrier"].values, index=pd.Index(ds["buses_i"].values).astype(str))
            all_electrical = set(carrier.index[carrier.fillna("").astype(str).str.strip().str.lower()
                                               .isin(ELECTRICAL_BUS_CARRIERS)])
            # Electricity in at a GB bus, something else (hydrogen) out: electrolysers.
            links = links.index[links.bus0.isin(electrical) & links.bus1.isin(bus_names - all_electrical)]
    per_load = pd.DataFrame(np.broadcast_to(static.reindex(loads).to_numpy(dtype=float), (len(snapshots), len(loads))),
                            index=snapshots, columns=loads)
    dynamic_columns = dynamic.columns.intersection(loads)
    per_load[dynamic_columns] = dynamic[dynamic_columns]
    total = per_load.sum(axis=1) + flexible
    gb_electrical = per_load.loc[:, is_electrical.to_numpy()].sum(axis=1) + flexible
    draw = pd.Series(0.0, index=snapshots)
    if links is not None and len(links):
        if links_path is None:
            raise ValueError(
                f"{network_path} has {len(links)} electrolyser links but no Stage 1 link schedule was given; "
                "their electricity draw is part of the metered-supply denominators (item D1)")
        flows = pd.read_csv(links_path, index_col=0, parse_dates=True)
        if flows.index.has_duplicates:
            raise ValueError(f"{links_path} has duplicate timestamps")
        flows = flows.reindex(index=snapshots)
        present = [name for name in links if name in flows.columns]
        if flows[present].isna().any().any():
            raise ValueError(f"{links_path} does not cover every snapshot of {network_path}")
        draw = flows[present].clip(lower=0.0).sum(axis=1)
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)["wholesale_price"]
    if prices.index.has_duplicates:
        raise ValueError("Wholesale prices have duplicate timestamps")
    price = prices.reindex(snapshots)
    demand = gb_electrical + draw
    demand_weighted_price(price, demand, weights)  # validates completeness and finiteness
    out = pd.DataFrame({"demand_mw": demand, "hours": weights, "price_gbp_per_mwh": price,
                        "total_load_mw": total, "gb_electrical_load_mw": gb_electrical,
                        "electrolyser_draw_mw": draw, "h2_offtake_mw": total - gb_electrical})
    # Kept off the columns so the written hourly inputs are unchanged. The market summary carries
    # its own (balancing) flexibility as consumer_flexibility_MWh; the demand checks compare Loads.
    out.attrs["flexible_demand_mwh"] = float((flexible * weights).sum())
    return out


#: F18 (27 Oct 2026): the low-price bands the CfD difference payments are reported in (GBP/MWh, at or
#: below; the hourly wholesale price the contract settles against). The model's zero-price hours sit a
#: hair either side of zero (solver noise, ``-0.01``), so the GBP 1 band beside the strict one.
CFD_LOW_PRICE_BANDS = {"price_at_or_below_0": 0.0, "price_at_or_below_1": 1.0}


def read_cfd_payments(network_path, reference_price, strike_prices, negative_price_rule="none",
                      reference_price_overrides=None, volume_caps=None, contract_book=None):
    """Recompute CfD payments from physical dispatch using the shared calculator.

    ``reference_price_overrides`` is ``{carrier: Series}`` for contracts that
    settle against a baseload reference rather than the hourly wholesale price;
    the calculator records which basis each generator used.

    ``contract_book`` (a path; FES 2050 task SETTLE2050) settles the 2050 register's
    contract units instead of the network's labels: each unit is its network asset's
    dispatch x the unit's ``energy_factor`` (``contract_book_2050.cfd_units``).
    """
    tables = _cached(("cfd", str(network_path), str(contract_book)),
                     lambda: _cfd_tables(network_path, contract_book))
    snapshots, generators, dispatch, weights = (
        tables["snapshots"], tables["generators"], tables["dispatch"], tables["weights"])
    missing, missing_capacity, absent_contract_columns, cfd_names = (
        tables["missing"], tables["missing_capacity"], tables["absent_contract_columns"], tables["cfd_names"])
    if not snapshots.equals(reference_price.index):
        raise ValueError("Physical dispatch and wholesale snapshots do not match")
    if len(cfd_names) and not strike_prices:
        raise ValueError("CfD-labelled physical dispatch requires configured strike prices")
    network = SimpleNamespace(generators=generators, snapshots=snapshots,
        generators_t=SimpleNamespace(p=dispatch), snapshot_weightings=pd.DataFrame({"generators": weights}))
    details = compute_cfd_payments(network, strike_prices, reference_price=reference_price,
                                  negative_price_rule=negative_price_rule,
                                  reference_price_overrides=reference_price_overrides,
                                  volume_caps=volume_caps, price_bands=CFD_LOW_PRICE_BANDS)
    physical = float(dispatch.mul(weights, axis=0).sum().sum())
    priced = float(details.total_dispatch_MWh.sum()) if not details.empty else 0.
    payment = float(details.net_cfd_cost.sum()) if not details.empty else 0.
    strikes = summarise_strike_provenance(details, strike_prices, absent_contract_columns)
    # F18 (27 Oct 2026, a report): the payments made in low-price hours, by band.
    strikes["low_price_hours"] = {
        label: {"payment_gbp": float(details[f"cfd_payment_{label}"].sum()) if not details.empty else 0.0,
                "dispatch_mwh": float(details[f"dispatch_MWh_{label}"].sum()) if not details.empty else 0.0}
        for label in CFD_LOW_PRICE_BANDS}
    return payment, priced, physical, len(missing), missing_capacity, strikes


def _cfd_tables(network_path, contract_book=None):
    """The file-only part of :func:`read_cfd_payments`: CfD generators and their dispatch."""
    if contract_book is not None:
        return _book_cfd_tables(network_path, contract_book)
    with xr.open_dataset(network_path, engine="netcdf4") as ds:
        snapshots = pd.DatetimeIndex(ds["snapshots_snapshot"].values)
        names = pd.Index(ds["generators_i"].values)
        if "generators_support_type" not in ds:
            raise ValueError("Physical model lacks support labels for CfD accounting")
        support = pd.Series(ds["generators_support_type"].values, index=names)
        cfd_names = support.index[support.eq("CfD")]
        generators = pd.DataFrame(index=cfd_names)
        absent_contract_columns = []
        # The contract's own strike technology (cfd_strike_table.CFD_TECHNOLOGY_COLUMN)
        # where the build recorded one; older networks have none and key on the carrier.
        if f"generators_{CFD_TECHNOLOGY_COLUMN}" in ds:
            generators[CFD_TECHNOLOGY_COLUMN] = pd.Series(
                ds[f"generators_{CFD_TECHNOLOGY_COLUMN}"].values, index=names).reindex(cfd_names)
        for column in ("support_type", "carrier", "bus", "cfd_round", "cfd_support_start"):
            key = f"generators_{column}"
            if key in ds:
                generators[column] = pd.Series(ds[key].values, index=names).reindex(cfd_names)
            else:
                # Substituting "" here is what sends every contract to the
                # carrier-wide ``future_*`` strike and disables support-start
                # gating, so the gap is recorded rather than absorbed.
                generators[column] = ""
                if column in ("cfd_round", "cfd_support_start"):
                    absent_contract_columns.append(column)
        available = pd.Index(ds["generators_t_p_i"].values)
        missing = cfd_names.difference(available)
        missing_capacity = float(ds["generators_p_nom"].sel(generators_i=missing).sum()) if len(missing) else 0.
        # Read the whole table then subset in pandas: a label ``.sel`` on the
        # NetCDF-backed array costs ~0.14 s per generator, the full read <1 s.
        dispatch = ds["generators_t_p"].to_pandas()[cfd_names.intersection(available)]
        dispatch.index = snapshots
        # A balancing month on its turn-down basis (a Stage 1 file is never re-shared).
        dispatch = balancing_dispatch_in_force(dispatch, network_path)
        weights = pd.Series(ds["snapshots_generators"].values, index=snapshots)
    if not np.isfinite(dispatch.to_numpy()).all():
        raise ValueError("CfD dispatch contains missing or non-finite values")
    return {"snapshots": snapshots, "generators": generators, "dispatch": dispatch, "weights": weights,
            "missing": missing, "missing_capacity": missing_capacity,
            "absent_contract_columns": absent_contract_columns, "cfd_names": cfd_names}


def _book_cfd_tables(network_path, contract_book):
    """:func:`_cfd_tables` for the 2050 contract book: the book's CfD units and their dispatch (task SETTLE2050)."""
    from scripts.public_ownership.contract_book_2050 import cfd_units, read_contract_book

    book = read_contract_book(contract_book)
    with xr.open_dataset(network_path, engine="netcdf4") as ds:
        snapshots = pd.DatetimeIndex(ds["snapshots_snapshot"].values)
        names = pd.Index(ds["generators_i"].values)
        if "generators_support_type" in ds:
            labels = pd.Series(ds["generators_support_type"].values, index=names).fillna("").astype(str).str.strip()
            if labels.isin(["CfD", "ROC", "DPA"]).any():
                raise ValueError(f"{network_path}: the network carries support labels and the bill names a contract "
                                 "book; settle one or the other")
        static = pd.DataFrame({"carrier": pd.Series(ds["generators_carrier"].values, index=names).astype(str),
                               "bus": pd.Series(ds["generators_bus"].values, index=names).astype(str)}, index=names)
        available = pd.Index(ds["generators_t_p_i"].values)
        assets = pd.Index(book["network_asset"].unique()).intersection(available)
        frame = ds["generators_t_p"].to_pandas()[assets]
        frame.index = snapshots
        weights = pd.Series(ds["snapshots_generators"].values, index=snapshots)
    units, dispatch = cfd_units(book, static, frame)
    if not np.isfinite(dispatch.to_numpy()).all():
        raise ValueError("CfD dispatch contains missing or non-finite values")
    generators = units[["support_type", "carrier", "bus", "cfd_round", "cfd_support_start"]]
    return {"snapshots": snapshots, "generators": generators, "dispatch": dispatch, "weights": weights,
            "missing": pd.Index([]), "missing_capacity": 0.0, "absent_contract_columns": [],
            "cfd_names": generators.index}


def summarise_strike_provenance(details, strike_prices, absent_contract_columns):
    """Report which strike price each unit of CfD energy was settled against.

    With no ``cfd_round`` on the saved generators every contract falls back to
    the carrier-wide modelled AR8 strike, so a whole fleet of separately
    auctioned farms settles at one price. That is the single largest assumption
    behind the CfD line and it is invisible in the payment total alone.

    Also reported: the share settled on the modelled AR8 round itself
    (``modelled_round_share_of_*``: capacity labelled ``AR8_modelled``, F18),
    which is a modelled clearing price rather than a published one.
    """
    if details.empty:
        return {"contract_metadata_absent": absent_contract_columns, "by_strike": [],
                "fallback_strike_share_of_payment": None, "fallback_strike_share_of_energy": None}
    frame = details.copy()
    frame["cfd_round"] = frame.get("cfd_round", "").fillna("").astype(str).str.strip()
    if "is_fallback_strike" in frame.columns:
        # The strike lookup reports its own fallbacks, so the key no longer has
        # to be reconstructed from the round label.
        frame["is_fallback"] = frame["is_fallback_strike"].astype(bool)
        frame["strike_key"] = frame["strike_key"].fillna("unmatched").astype(str)
    else:
        frame["is_fallback"] = frame["cfd_round"].eq("")
        frame["strike_key"] = [
            f"{row.cfd_round}_{row.carrier}" if row.cfd_round else f"{MODELLED_ROUND}_(carrier fallback)"
            for row in frame.itertuples()
        ]
    available = (
        sorted(strike_prices.strikes) if hasattr(strike_prices, "strikes")
        else sorted(str(key) for key in strike_prices)
    )
    if "reference_price_basis" not in frame.columns:
        frame["reference_price_basis"] = "hourly_wholesale"
    grouped = frame.groupby(["carrier", "strike_key", "strike_price", "is_fallback", "reference_price_basis"],
                            dropna=False).agg(
        generators=("generator", "nunique"),
        dispatch_mwh=("total_dispatch_MWh", "sum"),
        payment_gbp=("net_cfd_cost", "sum"),
    ).reset_index()
    energy = float(frame.total_dispatch_MWh.sum())
    gross = float(frame.net_cfd_cost.abs().sum())
    fallback = frame.loc[frame.is_fallback]
    modelled = frame.loc[frame["strike_key"].astype(str).str.startswith(MODELLED_ROUND + "_")]
    return {
        "contract_metadata_absent": absent_contract_columns,
        "available_strike_keys": [str(key) for key in available],
        "by_strike": grouped.to_dict("records"),
        "fallback_strike_share_of_energy": float(fallback.total_dispatch_MWh.sum() / energy) if energy else None,
        "fallback_strike_share_of_payment": float(fallback.net_cfd_cost.abs().sum() / gross) if gross else None,
        "modelled_round": MODELLED_ROUND,
        "modelled_round_share_of_energy": float(modelled.total_dispatch_MWh.sum() / energy) if energy else None,
        "modelled_round_share_of_payment": float(modelled.net_cfd_cost.abs().sum() / gross) if gross else None,
    }


def season_labels(index) -> pd.Series:
    """Label each snapshot summer (Apr-Sep) or winter (Oct-Mar)."""
    index = pd.DatetimeIndex(index)
    return pd.Series(np.where(index.month.isin(SUMMER_MONTHS), "summer", "winter"), index=index)


def seasonal_demand_weighted_reference(hourly):
    """Season-level demand-weighted mean wholesale price, expanded to every snapshot.

    Baseload CfDs settle against a season-ahead baseload reference price rather
    than the hourly day-ahead price. The model has no forward market, so the
    season-ahead price is proxied by the outturn demand-weighted mean of the
    season the snapshot falls in, computed over the full calendar year. Returns
    the per-snapshot series and a per-season record for the report.
    """
    season = season_labels(hourly.index)
    energy = hourly.demand_mw * hourly.hours
    cost = hourly.price_gbp_per_mwh * energy
    by_season = cost.groupby(season).sum() / energy.groupby(season).sum()
    if by_season.isna().any() or not np.isfinite(by_season.to_numpy()).all():
        raise ValueError("Seasonal reference prices could not be formed from the hourly inputs")
    series = season.map(by_season).astype(float)
    detail = {
        name: {
            "reference_price_gbp_per_mwh": float(by_season[name]),
            "hours": float(hourly.hours[season.eq(name)].sum()),
            "months": "Apr-Sep" if name == "summer" else "Oct-Mar",
        }
        for name in by_season.index
    }
    return series, detail


def read_economic_curtailment(network_path, price, strike_prices, rounds, threshold,
                              physical_network_path=None, negative_price_rule="none", selection_price=None):
    """Value the pre-AR4 CfD output the wholesale LP curtailed at ~zero prices.

    The copperplate wholesale solve turns CfD generators down whenever their
    marginal cost clears the market at ~GBP0, and no difference payment is
    booked on that energy. Contracts allocated before AR4 carry no zero-payment
    term, so the real generator would run and be paid strike minus reference.
    For each eligible generator: curtailed_h = max(p_nom x p_max_pu - p, 0) in
    hours where the wholesale price <= ``threshold``; payment = sum over hours
    of curtailed_h x hours x max(strike - price_h, 0).

    ``p`` is the higher of the wholesale and balancing dispatch when
    ``physical_network_path`` is given. The CfD line itself is settled on
    balancing dispatch, so energy the balancing stage restores (Hinkley Point C
    was turned up 6.9 TWh in the 2030 run) is already paid there and must not
    be paid a second time as curtailment.

    Whole tables are read with ``to_pandas`` and subset in pandas: a label
    ``.sel`` on the NetCDF-backed arrays costs ~0.14 s per generator.

    Only valid on an as-built wholesale (see ECONOMIC_CURTAILMENT_RULES; the
    caller refuses a contract-aware run). Known simplification: Hinkley Point
    C's curtailed energy is valued at strike minus the *hourly* price, while
    its settled output uses the seasonal baseload reference
    (``reference_price_basis``); on the 2030 as-built run that is 1.31 TWh.

    ``selection_price`` (item F12) picks the hours the LP curtailed in when it is not
    the price paid against: under the ``cp32`` price level the curtailment is the
    LP's, in the hours its own price was at or below ``threshold``, and ``price`` (the
    mapped reference) sets the payment. None selects on ``price`` itself.
    """
    rounds = [str(r) for r in rounds]
    tables = _cached(("curtailment", str(network_path), tuple(rounds), str(physical_network_path)),
                     lambda: _curtailment_tables(network_path, rounds, physical_network_path))
    snapshots, eligible, weights = tables["snapshots"], tables["eligible"], tables["weights"]
    if not snapshots.equals(price.index):
        raise ValueError("Wholesale dispatch and wholesale price snapshots do not match")
    selection_price = price if selection_price is None else selection_price
    if not snapshots.equals(selection_price.index):
        raise ValueError("Wholesale dispatch and the curtailment selection price snapshots do not match")
    if not len(eligible):
        return {"payment_gbp": 0.0, "curtailed_mwh": 0.0, "generators": 0,
                "generators_without_dispatch": tables["without_dispatch"],
                "hours_at_or_below_threshold": float(weights[selection_price.le(threshold)].sum()),
                "static_availability_generators": [], "by_strike": []}
    dispatch, availability, p_nom = tables["dispatch"], tables["availability"], tables["p_nom"]
    carrier, cfd_round, without_dispatch, static = (
        tables["carrier"], tables["cfd_round"], tables["without_dispatch"], tables["static"])
    technology = tables.get("technology", pd.Series("", index=carrier.index))
    available_mw = availability.mul(p_nom[eligible], axis=1)
    curtailed_mw = (available_mw - dispatch).clip(lower=0.0)
    low_price = selection_price.le(threshold)
    curtailed_mwh = curtailed_mw.where(low_price, 0.0).mul(weights, axis=0)

    strikes, keys, fallbacks, priced = [], [], [], []
    for gen in eligible:
        strike, key, is_fallback = _lookup_strike_price(carrier[gen], cfd_round[gen], strike_prices,
                                                        technology.get(gen, ""))
        if strike is None:
            continue
        priced.append(gen)
        strikes.append(float(strike))
        keys.append(key)
        fallbacks.append(bool(is_fallback))
    if not priced:
        return {"payment_gbp": 0.0, "curtailed_mwh": 0.0, "generators": 0,
                "generators_without_dispatch": without_dispatch,
                "hours_at_or_below_threshold": float(weights[low_price].sum()),
                "static_availability_generators": [str(g) for g in static], "by_strike": []}
    energy = curtailed_mwh[priced]
    uplift = np.maximum(np.asarray(strikes)[None, :] - price.to_numpy()[:, None], 0.0)
    if negative_price_rule != "none":
        # Paid as if generated, so on the same terms as generated output (AR2-AR3: six-hour rule; AR1: none).
        uplift = uplift * negative_price_eligibility(price.to_numpy(), [cfd_round[g] for g in priced],
                                                     negative_price_rule)
    payment_by_gen = pd.Series((energy.to_numpy() * uplift).sum(axis=0), index=priced)
    frame = pd.DataFrame({
        "generator": priced,
        "carrier": [carrier[g] for g in priced],
        "cfd_round": [cfd_round[g] for g in priced],
        "strike_key": keys, "strike_price": strikes, "is_fallback": fallbacks,
        "curtailed_mwh": energy.sum().to_numpy(),
        "payment_gbp": payment_by_gen.to_numpy(),
    })
    grouped = frame.groupby(["carrier", "cfd_round", "strike_key", "strike_price", "is_fallback"], dropna=False).agg(
        generators=("generator", "nunique"), curtailed_mwh=("curtailed_mwh", "sum"),
        payment_gbp=("payment_gbp", "sum"),
    ).reset_index()
    return {
        "payment_gbp": float(frame.payment_gbp.sum()),
        "curtailed_mwh": float(frame.curtailed_mwh.sum()),
        "generators": int(len(priced)),
        "generators_without_dispatch": without_dispatch,
        "hours_at_or_below_threshold": float(weights[low_price].sum()),
        "static_availability_generators": [str(g) for g in static],
        "by_strike": grouped.to_dict("records"),
    }


def wholesale_curtailment_order(network_path) -> dict:
    """Say which renewable curtailment order the saved wholesale was solved with.

    The order is not written to any audit file, but it is visible in the saved
    network: ``apply_curtailment_order`` rewrites each re-priced CfD unit's
    static marginal cost before the solve (AR4+ to -1, older rounds to minus
    strike), and the export keeps it. As built every CfD renewable sits at 0 or
    at the resource-cost floor. ``unknown`` when the file lacks the columns.
    """
    return _cached(("order", str(network_path)), lambda: _wholesale_curtailment_order(network_path))


def _wholesale_curtailment_order(network_path) -> dict:
    with xr.open_dataset(network_path, engine="netcdf4") as ds:
        needed = ("generators_support_type", "generators_carrier", "generators_marginal_cost")
        if any(key not in ds for key in needed):
            return {"order": "unknown", "repriced_units": 0, "pre_ar4_units_below_zero": 0}
        names = pd.Index(ds["generators_i"].values)
        support = pd.Series(ds["generators_support_type"].values, index=names).fillna("").astype(str)
        carrier = pd.Series(ds["generators_carrier"].values, index=names).fillna("").astype(str)
        cost = pd.Series(ds["generators_marginal_cost"].values, index=names, dtype=float)
        rounds = (pd.Series(ds["generators_cfd_round"].values, index=names).fillna("").astype(str).str.strip()
                  if "generators_cfd_round" in ds else pd.Series("", index=names))
    ordered = support.eq("CfD") & carrier.isin(CONTRACT_AWARE_CARRIERS)
    repriced = ordered & cost.lt(CONTRACT_AWARE_COST_BELOW_GBP_PER_MWH)
    pre_ar4 = repriced & rounds.isin(DEFAULT_PRE_AR4_ROUNDS)
    return {"order": "contract_aware" if repriced.any() else "as_built",
            "repriced_units": int(repriced.sum()), "pre_ar4_units_below_zero": int(pre_ar4.sum())}


def _curtailment_tables(network_path, rounds, physical_network_path):
    """The file-only part of :func:`read_economic_curtailment`: eligible pre-AR4 CfD tables."""
    with xr.open_dataset(network_path, engine="netcdf4") as ds:
        snapshots = pd.DatetimeIndex(ds["snapshots_snapshot"].values)
        names = pd.Index(ds["generators_i"].values)
        for key in ("generators_support_type", "generators_cfd_round", "generators_carrier",
                    "generators_p_nom", "generators_t_p", "generators_t_p_max_pu"):
            if key not in ds:
                raise ValueError(f"Wholesale model lacks {key}; the economic-curtailment rule needs it")
        support = pd.Series(ds["generators_support_type"].values, index=names)
        cfd_round = pd.Series(ds["generators_cfd_round"].values, index=names).fillna("").astype(str).str.strip()
        carrier = pd.Series(ds["generators_carrier"].values, index=names).fillna("").astype(str)
        technology_key = f"generators_{CFD_TECHNOLOGY_COLUMN}"
        technology = (pd.Series(ds[technology_key].values, index=names).fillna("").astype(str).str.strip()
                      if technology_key in ds else pd.Series("", index=names))
        p_nom = pd.Series(ds["generators_p_nom"].values, index=names, dtype=float)
        weights = pd.Series(ds["snapshots_generators"].values, index=snapshots, dtype=float)
        eligible = names[support.eq("CfD").to_numpy() & cfd_round.isin(rounds).to_numpy()]
        dispatched = pd.Index(ds["generators_t_p_i"].values)
        without_dispatch = eligible.difference(dispatched)
        eligible = eligible.intersection(dispatched)
        if not len(eligible):
            return {"snapshots": snapshots, "eligible": eligible, "weights": weights,
                    "without_dispatch": int(len(without_dispatch))}
        dispatch = ds["generators_t_p"].to_pandas()[eligible]
        dispatch.index = snapshots
        availability_table = ds["generators_t_p_max_pu"].to_pandas()
        availability_table.index = snapshots
        varying = eligible.intersection(availability_table.columns)
        # Generators with no time-varying p_max_pu (Hinkley Point C) are
        # available at PyPSA's static default of 1.0 unless the file carries a
        # static column.
        availability = pd.DataFrame(1.0, index=snapshots, columns=eligible)
        availability[varying] = availability_table[varying]
        static = eligible.difference(varying)
        if "generators_p_max_pu" in ds and len(static):
            static_values = pd.Series(ds["generators_p_max_pu"].values, index=names, dtype=float)
            availability[static] = np.broadcast_to(static_values[static].to_numpy(), (len(snapshots), len(static)))
    if not np.isfinite(dispatch.to_numpy()).all() or not np.isfinite(availability.to_numpy()).all():
        raise ValueError("Wholesale dispatch or availability contains non-finite values")
    if physical_network_path is not None:
        # Net off anything the balancing stage restored: the CfD line pays on
        # balancing dispatch, so only energy still curtailed after balancing
        # is uncompensated.
        with xr.open_dataset(physical_network_path, engine="netcdf4") as phys:
            phys_snapshots = pd.DatetimeIndex(phys["snapshots_snapshot"].values)
            if not phys_snapshots.equals(snapshots):
                raise ValueError("Wholesale and balancing snapshots do not match")
            phys_table = phys["generators_t_p"].to_pandas()
            phys_table.index = snapshots
            phys_table = balancing_dispatch_in_force(phys_table, physical_network_path)
        physical = phys_table.reindex(columns=eligible, fill_value=0.0)
        if not np.isfinite(physical.to_numpy()).all():
            raise ValueError("Balancing dispatch contains non-finite values")
        dispatch = np.maximum(dispatch, physical)
    return {"snapshots": snapshots, "eligible": eligible, "weights": weights, "dispatch": dispatch,
            "availability": availability, "p_nom": p_nom, "carrier": carrier, "cfd_round": cfd_round,
            "technology": technology, "without_dispatch": int(len(without_dispatch)), "static": static}


def _combine_curtailment_rows(rows):
    """Roll per-month economic-curtailment breakdowns up to one annual table."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["carrier", "cfd_round", "strike_key", "strike_price", "is_fallback"], dropna=False).agg(
        generators=("generators", "max"), curtailed_mwh=("curtailed_mwh", "sum"),
        payment_gbp=("payment_gbp", "sum"),
    ).reset_index().sort_values("payment_gbp", ascending=False)
    return grouped.to_dict("records")


def read_annual_aggregate(path, scenarios, year):
    """Read the declared twelve-month CP30 aggregate as the annual accounting source.

    The projector no longer re-sums monthly accounting itself: one aggregate
    rule owns that arithmetic and validates completeness, so the bill and the
    NESO comparator cannot drift apart.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    aggregate = document["aggregate"]
    if not aggregate.get("annualisation_allowed"):
        raise ValueError(f"{path} is not a complete annualised aggregate")
    declared = list(document.get("monthly_scenarios", []))
    if declared != list(scenarios):
        raise ValueError(
            "The annual aggregate covers different scenarios from the bill case: "
            f"{sorted(set(declared) ^ set(scenarios))}"
        )
    expected_hours = 8784 if calendar.isleap(int(year)) else 8760
    if not np.isclose(float(aggregate["reported_period_hours"]), expected_hours, rtol=0, atol=1e-6):
        raise ValueError(
            f"The annual aggregate reports {aggregate['reported_period_hours']} hours, "
            f"not the {expected_hours} of {year}"
        )
    return aggregate


# The recovery bases a recovery-demand file may declare (``recovery_basis``). Item D1
# (26 Sep 2026, the 2050 file): one GSP-to-meter loss factor turns (GB load +
# electrolyser draw), taken as a GSP volume, into metered supply. F18 (6 Oct 2026,
# the 2030 file): the load already contains losses, so there are two volumes, metered
# supply and the GSP charging volume, and Ofgem's household loss factor on every line
# charged at a GSP rate. A file that declares nothing is on the D1 basis.
D1_BASIS = "metered_supply_d1"
TWO_VOLUME_BASIS = "loss_inclusive_two_volume"
# F18: the old denominator names, kept as aliases on the two-volume basis while other
# code still reads them (deprecated): name -> the volume it now equals.
DEPRECATED_RECOVERY_ALIASES = {"final_demand_mwh": "metered_supply_mwh",
                               "gsp_chargeable_volume_mwh": "gsp_charging_volume_mwh",
                               "cfd_eligible_supplier_sales_mwh": "ro_eligible_metered_supply_mwh"}
# The volumes that sit at the grid supply point on the two-volume basis: a line spread
# over one is a GSP rate, charged on the household's 3.1 MWh times the household loss
# factor unless its allocation factor already carries losses (the Capacity Market).
GSP_RECOVERY_VOLUMES = {"gsp_charging_volume_mwh", "cfd_eligible_gsp_volume_mwh"}
# How a derived line on a GSP volume takes household losses (``derivation.household_losses``).
HOUSEHOLD_LOSS_TREATMENTS = ("household_loss_factor", "in_household_allocation_factor")

# Denominators a sourced external may name; each is a key of read_recovery_denominators.
# ``gsp_chargeable_volume_mwh`` is the paying volume before the loss factor: the
# Capacity Market line takes it because its household factor already carries
# losses (item D1). The F18 volumes are named on the two-volume basis.
SOURCED_EXTERNAL_DENOMINATORS = {"final_demand_mwh", "cfd_eligible_supplier_sales_mwh", "gsp_chargeable_volume_mwh",
                                 "metered_supply_mwh", "gsp_charging_volume_mwh", "cfd_eligible_gsp_volume_mwh",
                                 "ro_eligible_metered_supply_mwh"}

# How a sourced external's own period maps onto the bill's calendar year (28 Sep
# 2026: the bill is calendar 2030, the year the model dispatches). Enforced once
# ``benchmark.bill_calendar_year`` is declared:
#   calendar_year    the source already covers the calendar year (the Capacity
#                    Market's half-and-half delivery years, a nil scheme);
#   calendar_blend   fiscal-year values blended by the months of the calendar
#                    year inside each (``period_blend``: January to March in FY
#                    Y-1/Y, April to December in FY Y/Y+1);
#   period_average   an average over a span that contains the calendar year
#                    (``period_span``);
#   held_flat_real   a value from an earlier ``source_period`` held flat in real
#                    terms, a declared assumption rather than a 2030 forecast;
#   extrapolated_declared  (task Y13, the 2050 bill) a value carried from its
#                    ``extrapolation.from_period`` to the bill year by a declared
#                    ``extrapolation.rule`` on named drivers, for years no source
#                    reaches. No 2030 line uses it.
PERIOD_BASES = ("calendar_year", "calendar_blend", "period_average", "held_flat_real",
                "extrapolated_declared")


def _fiscal_year_start(label) -> int:
    """``FY2029/30`` -> 2029 (a UK fiscal year runs 1 April to 31 March)."""
    text = str(label).strip().upper().removeprefix("FY").strip()
    first, _, second = text.partition("/")
    start = int(first)
    if not second or (start + 1) % 100 != int(second) % 100:
        raise ValueError(f"fiscal year label {label!r} must read like FY2029/30")
    return start


def calendar_months_in_fiscal_year(calendar_year: int, fiscal_start: int) -> int:
    """Months of ``calendar_year`` that fall in the fiscal year starting April ``fiscal_start``."""
    if fiscal_start == calendar_year - 1:
        return 3
    if fiscal_start == calendar_year:
        return 9
    return 0


def blend_weights(components, calendar_year: int) -> list:
    """Each component's weight: its months of the calendar year over twelve; they must cover the year."""
    weights = [calendar_months_in_fiscal_year(calendar_year, _fiscal_year_start(c["period"])) / 12.0
               for c in components]
    if len({_fiscal_year_start(c["period"]) for c in components}) != len(components):
        raise ValueError("period_blend names the same fiscal year twice")
    if not np.isclose(sum(weights), 1.0, rtol=0, atol=1e-12) or min(weights) <= 0.0:
        raise ValueError(
            f"period_blend {[c['period'] for c in components]} does not cover calendar {calendar_year}: "
            f"it needs FY{calendar_year - 1}/{str(calendar_year)[2:]} and FY{calendar_year}/{str(calendar_year + 1)[2:]}")
    return weights


def check_period_basis(key, block, calendar_year: int) -> str:
    """Refuse a sourced external whose period is not stated against the bill's calendar year."""
    basis = block.get("period_basis")
    if basis not in PERIOD_BASES:
        raise ValueError(
            f"{key}: period_basis must be one of {PERIOD_BASES} against the calendar-{calendar_year} bill, "
            f"got {basis!r}")
    if basis == "calendar_blend":
        components = (block.get("period_blend") or block.get("derivation", {}).get("period_blend") or [])
        blend_weights(components, calendar_year)
    elif basis == "period_average":
        span = block.get("period_span") or {}
        if not (int(span.get("first_fiscal_year_start", 9999)) <= calendar_year - 1
                and int(span.get("last_fiscal_year_start", -1)) >= calendar_year):
            raise ValueError(f"{key}: period_span must contain calendar {calendar_year}")
    elif basis == "held_flat_real" and not block.get("source_period"):
        raise ValueError(f"{key}: held_flat_real needs the source_period it is held from")
    elif basis == "extrapolated_declared":
        extrapolation = block.get("extrapolation") or {}
        if not (extrapolation.get("rule") and extrapolation.get("from_period")):
            raise ValueError(f"{key}: extrapolated_declared needs extrapolation.rule and extrapolation.from_period")
    return basis


def blended_household_value(key, block, calendar_year: int, report_price_year: int):
    """A per-household line blended from fiscal-year components, each on its own price basis.

    Returns ``(value in report prices, text)``. Each component carries ``value``,
    ``price_year`` and, when that is not the report year, its own ``rebasing``
    pair, because fiscal years quoted in money of the day deflate differently.
    """
    components = block["period_blend"]
    weights = blend_weights(components, calendar_year)
    total, parts = 0.0, []
    for weight, component in zip(weights, components):
        year = int(component["price_year"])
        factor = 1.0 if year == report_price_year else _rebasing_factor(
            component["rebasing"], f"{key} {component['period']} rebasing", year, report_price_year)
        value = _finite(component["value"], f"{key} {component['period']}", True) * factor
        total += weight * value
        parts.append(f"{component['period']} GBP{component['value']:.2f} ({year} prices) x {weight:.2f}")
    return total, (f" Calendar {calendar_year} = " + " + ".join(parts)
                   + f" = GBP{total:.2f} in {report_price_year} prices.")


#: Review 2 (3 Oct 2026): register-booked lines read from another run group's public account while an
#: ownership projection is running (:class:`register_lines_from`); empty otherwise, so every bill reads
#: the account its assumptions name.
_REGISTER_LINE_ACCOUNTS: dict = {}


class register_lines_from:
    """Context manager: read the named register-booked lines (``{line key: public account}``) from another account.

    A register bound shares its parent's forecast bill, whose top-up lines read the parent's public
    account; the bound's own register sizes its own top-ups, so its ownership projection re-reads
    those lines from the bound's own account (``project_public_ownership_consumer_bill``).
    """

    def __init__(self, accounts):
        self.accounts = dict(accounts or {})

    def __enter__(self):
        self.saved = dict(_REGISTER_LINE_ACCOUNTS)
        _REGISTER_LINE_ACCOUNTS.update(self.accounts)
        return self

    def __exit__(self, *exc):
        _REGISTER_LINE_ACCOUNTS.clear()
        _REGISTER_LINE_ACCOUNTS.update(self.saved)
        return False


def _parity_path(path):
    """F20 parity (14 Oct 2026): a literal ``resources/...`` account redirected for a parity run (unchanged otherwise)."""
    from scripts.public_ownership.f20_parity.run_parity import parity_path

    return parity_path(path)


def register_booked_cost(key, cost_from):
    """``(cost in the register's price year, source)`` of a line the finance register books (decision 41 (c)).

    ``cost_from`` names ``public_account`` (a ``{run_group}_public_account_summary.json``),
    ``case`` (default ``private``: the forecast bill is the private case) and ``key``.
    """
    if key in _REGISTER_LINE_ACCOUNTS:
        cost_from = {**cost_from, "public_account": _REGISTER_LINE_ACCOUNTS[key]}
    path = Path(_parity_path(cost_from["public_account"]))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        raise FileNotFoundError(f"{key}: cost_from names {cost_from['public_account']}, which does not exist")
    document = json.loads(path.read_text(encoding="utf-8"))
    case_id = str(cost_from.get("case", "private"))
    row = (document.get("cases") or {}).get(case_id)
    if row is None or cost_from["key"] not in row:
        raise ValueError(f"{key}: {path.name} has no {cost_from['key']!r} for case {case_id!r}")
    cost = _finite(row[cost_from["key"]], f"{key} {cost_from['key']}", False)
    if cost < 0:
        raise ValueError(f"{key}: the register books a negative {cost_from['key']} ({cost:,.0f})")
    source = {"file": str(cost_from["public_account"]), "case": case_id, "key": cost_from["key"],
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if "price_level" in document:
        # Item F12 (12 Oct 2026): an account a price-level twin built says so; model accounts carry no key.
        source["price_level"] = str(document["price_level"])
    return cost, source


def register_booked_account(key, cost_from):
    """``(path, price level)`` of the public account a register-booked line reads (``model`` without the key)."""
    if key in _REGISTER_LINE_ACCOUNTS:
        cost_from = {**cost_from, "public_account": _REGISTER_LINE_ACCOUNTS[key]}
    path = Path(_parity_path(cost_from["public_account"]))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        return str(cost_from["public_account"]), DEFAULT_PRICE_LEVEL
    document = json.loads(path.read_text(encoding="utf-8"))
    return str(cost_from["public_account"]), str(document.get("price_level", DEFAULT_PRICE_LEVEL))


def resolve_sourced_externals(sourced, recovery, consumption_kwh, calendar_year=None):
    """Derive the value of every sourced external that is a cost spread over demand.

    A line whose ``derivation`` names a ``recovery_denominator`` takes that volume
    from the recovery-demand file, so no line keeps its own copy of an annual
    demand figure (item F4: the Capacity Market line had kept the pre-rebuild
    311.77 TWh). A stored ``recovery_denominator_mwh`` is refused rather than used.
    """
    resolved = {}
    for key, block in (sourced or {}).items():
        derivation = block.get("derivation") or {}
        if "recovery_denominator_mwh" in derivation:
            raise ValueError(
                f"{key}: derivation stores its own recovery_denominator_mwh; name the denominator "
                "(recovery_denominator: final_demand_mwh) so it is read from the recovery-demand file"
            )
        name = derivation.get("recovery_denominator")
        if name is None:
            resolved[key] = block
            continue
        if name not in SOURCED_EXTERNAL_DENOMINATORS:
            raise ValueError(f"{key}: unknown recovery denominator {name!r}")
        if recovery is None:
            raise ValueError(f"{key}: derived over {name}, which needs the recovery-demand file")
        _check_recovery_volume_name(key, name, recovery)
        denominator = _finite(recovery[name], f"{key} {name}", True)
        loss = household_loss_for(key, derivation, name, recovery)
        if derivation.get("period_blend"):
            # A cost blended from fiscal years on one price basis (calendar 2030 =
            # 3/12 of FY2029/30 + 9/12 of FY2030/31); the block's rebasing applies after.
            if calendar_year is None:
                raise ValueError(f"{key}: a period_blend needs benchmark.bill_calendar_year")
            components = derivation["period_blend"]
            weights = blend_weights(components, calendar_year)
            cost = sum(w * _finite(c["cost_gbp"], f"{key} {c['period']} cost", True)
                       for w, c in zip(weights, components))
            derivation = {**derivation, "cost_gbp_blended": cost,
                          "blend_weights": {c["period"]: w for w, c in zip(weights, components)}}
        elif derivation.get("cost_from"):
            # FES 2050 decision 41 (c): a cost the finance register books (Sizewell C's RAB
            # allowance: allowed revenue less market revenue), read from the run group's
            # public account so the bill line and the register carry the same money.
            cost, source = register_booked_cost(key, derivation["cost_from"])
            derivation = {**derivation, "cost_gbp": cost, "cost_from_resolved": source}
        else:
            # agreement_cost_gbp is the Capacity Market's own name for it; cost_gbp is general (item F1).
            cost_field = "cost_gbp" if "cost_gbp" in derivation else "agreement_cost_gbp"
            # Task E2E (FES 2050): a declared nil (value_basis "nil", as the external lines
            # use it) may cost nothing: the 2050 Capacity Market clears at zero centrally
            # (Y11). Every 2030 derived line is positive.
            nil = block.get("value_basis") == "nil"
            cost = _finite(derivation[cost_field], f"{key} cost", not nil)
            if nil and cost != 0.0:
                raise ValueError(f"{key}: value_basis nil but cost {cost}")
            if cost < 0:
                raise ValueError(f"{key} cost must not be negative")
            if derivation.get("demand_scaled_gbp_per_mwh") is not None:
                # FES 2050 decision 65 (2050 BSUoS lines only): a declared cost (the bottom-up ancillary
                # services) plus a part held flat per MWh of a named volume (NESO's internal cost and the
                # trade categories), the volume read from the recovery-demand file like the denominator.
                volume_name = derivation.get("demand_scaled_volume")
                if volume_name not in SOURCED_EXTERNAL_DENOMINATORS:
                    raise ValueError(f"{key}: demand_scaled_volume {volume_name!r} is not a recovery volume")
                _check_recovery_volume_name(key, volume_name, recovery)
                rate = _finite(derivation["demand_scaled_gbp_per_mwh"], f"{key} demand-scaled rate", False)
                scaled = rate * _finite(recovery[volume_name], f"{key} {volume_name}", True)
                derivation = {**derivation, "declared_cost_gbp": cost, "demand_scaled_cost_gbp": scaled,
                              "cost_gbp_total": cost + scaled}
                cost = cost + scaled
        block = {**block, "derivation": {**derivation, "recovery_denominator_mwh_used": denominator,
                                         "household_loss_factor_used": loss, "cost_gbp_used": cost}}
        # A charge levied on part of the day is not an energy-proportional share: the
        # Capacity Market supplier charge falls on winter-peak demand, where households
        # are concentrated (item MR4.4).
        factor = _finite(derivation.get("household_allocation_factor", 1.0), f"{key} household_allocation_factor", True)
        # F18: a GSP-rate line is charged on the household's loss-uplifted 3.1 MWh.
        block["value"] = cost / denominator * float(consumption_kwh) / 1000.0 * factor * loss
        resolved[key] = block
    return resolved


def _check_recovery_volume_name(key, name, recovery):
    """Refuse a deprecated alias on the two-volume basis, and a volume the recovery output lacks."""
    if recovery.get("recovery_basis") == TWO_VOLUME_BASIS and name in DEPRECATED_RECOVERY_ALIASES:
        raise ValueError(f"{key}: {name} is a deprecated alias on the two-volume recovery basis; name "
                         f"{DEPRECATED_RECOVERY_ALIASES[name]} or another F18 volume (item F18)")
    if recovery.get(name) is None:
        raise ValueError(f"{key}: the recovery denominators carry no {name}")


def household_loss_for(key, derivation, name, recovery):
    """F18: the household loss factor a derived line over ``name`` is charged at.

    On the two-volume basis a line spread over a GSP volume is a GSP rate, charged on
    the household's 3.1 MWh times the household loss factor, and must say so
    (``derivation.household_losses``): ``household_loss_factor``, or
    ``in_household_allocation_factor`` where its allocation factor already carries
    losses (the Capacity Market). On the D1 basis the loss sits in the metered-supply
    division itself, so no line declares one.
    """
    treatment = derivation.get("household_losses")
    if recovery.get("recovery_basis") != TWO_VOLUME_BASIS:
        if treatment is not None:
            raise ValueError(f"{key}: household_losses is declared but the recovery file is on the D1 basis, "
                             "which has no household loss factor")
        return 1.0
    if name in GSP_RECOVERY_VOLUMES:
        if treatment not in HOUSEHOLD_LOSS_TREATMENTS:
            raise ValueError(f"{key}: spread over the GSP volume {name}, so derivation.household_losses must be one "
                             f"of {HOUSEHOLD_LOSS_TREATMENTS} (item F18)")
        return float(recovery["household_loss_factor"]) if treatment == "household_loss_factor" else 1.0
    if treatment is not None:
        raise ValueError(f"{key}: household_losses applies only to a line spread over a GSP volume, not {name}")
    return 1.0


#: Levy exemption settings (``levy_exemptions`` in the assumptions file). The
#: defaults reproduce the bill as recovered today: electrolysers pay every levy on
#: their draw (decision D1) and the EII exemption is the 2024-25 outturn share.
LEVY_EXEMPTION_DEFAULTS = {
    "electrolysers_exempt": False,
    # A further share of the eligible base exempted (e.g. the British Industrial
    # Competitiveness Scheme from 2027), recovered from everyone else.
    "additional_exempt_share": 0.0,
    # Whether that further exemption also reaches the Capacity Market charge.
    "additional_exemption_covers_capacity_market": False,
}


def apply_levy_exemptions(recovery, block=None):
    """Return ``(recovery, record)`` with the levy denominators after the declared exemptions.

    Moves only the levy bases: ``cfd_eligible_supplier_sales_mwh`` (CfD, CCS, FiT;
    the RO follows through :func:`scale_ro_for_exemptions`) and, for electrolysers
    and optionally the further exemption, ``gsp_chargeable_volume_mwh`` (the
    Capacity Market). BSUoS, LDES and the network lines stay on final demand. An
    exempted volume's cost falls on everyone else, so a smaller base raises each
    remaining household's share; nothing is dropped.
    """
    settings = {**LEVY_EXEMPTION_DEFAULTS, **(block or {})}
    unknown = set(settings) - set(LEVY_EXEMPTION_DEFAULTS) - {"note", "source", "rationale", "sensitivities"}
    if unknown:
        raise ValueError(f"levy_exemptions has unknown keys {sorted(unknown)}")
    extra = float(settings["additional_exempt_share"])
    if not 0.0 <= extra < 1.0:
        raise ValueError("levy_exemptions.additional_exempt_share must be in [0, 1)")
    if not settings["electrolysers_exempt"] and extra == 0.0:
        # The defaults: the bill as recovered today, denominators untouched.
        return recovery, {**settings, "eligible_scale": 1.0, "cfd_eligible_scale": 1.0, "ro_eligible_scale": 1.0,
                          "applied": False}
    if recovery.get("recovery_basis") == TWO_VOLUME_BASIS:
        return _two_volume_exemptions(recovery, settings, extra)
    if recovery.get("cfd_eligible_supplier_sales_mwh") is None:
        raise ValueError("levy exemptions need the sourced cfd_eligible_supplier_sales_mwh denominator")
    out = dict(recovery)
    eligible = float(recovery["cfd_eligible_supplier_sales_mwh"])
    gsp = recovery.get("gsp_chargeable_volume_mwh")
    if settings["electrolysers_exempt"]:
        draw = float(recovery.get("electrolyser_draw_mwh") or 0.0)
        loss = float(recovery.get("gsp_to_meter_loss_factor") or 1.0)
        # The draw pays at the GSP; on the metered basis it is draw / loss factor.
        out["cfd_eligible_supplier_sales_mwh"] = eligible - draw / loss * float(recovery["cfd_eligible_share"])
        if gsp is not None:
            out["gsp_chargeable_volume_mwh"] = float(gsp) - draw
    out["cfd_eligible_supplier_sales_mwh"] = float(out["cfd_eligible_supplier_sales_mwh"]) * (1.0 - extra)
    if extra and settings["additional_exemption_covers_capacity_market"] and out.get("gsp_chargeable_volume_mwh"):
        out["gsp_chargeable_volume_mwh"] = float(out["gsp_chargeable_volume_mwh"]) * (1.0 - extra)
    if out["cfd_eligible_supplier_sales_mwh"] <= 0:
        raise ValueError("levy exemptions leave no eligible supply")
    scale = eligible / float(out["cfd_eligible_supplier_sales_mwh"])
    record = {**settings, "applied": True,
              "eligible_supplier_sales_before_mwh": eligible,
              "eligible_supplier_sales_after_mwh": float(out["cfd_eligible_supplier_sales_mwh"]),
              # On the D1 basis the CfD and the RO share one eligible base.
              "eligible_scale": scale, "cfd_eligible_scale": scale, "ro_eligible_scale": scale,
              "gsp_chargeable_volume_before_mwh": gsp,
              "gsp_chargeable_volume_after_mwh": out.get("gsp_chargeable_volume_mwh"),
              "capacity_market_volume_before_mwh": gsp,
              "capacity_market_volume_after_mwh": out.get("gsp_chargeable_volume_mwh")}
    return out, record


def _two_volume_exemptions(recovery, settings, extra):
    """F18: the exemptions on the two-volume basis.

    The electrolysers' draw leaves the CfD's eligible GSP volume and the RO and FiT's
    eligible metered supply at its own size (it is metered at the plant); a further
    exempt share (the British Industrial Competitiveness Scheme) leaves the RO and FiT
    base only, since BICS exempts industry from the RO, FiT and Capacity Market, not the
    CfD (user, 6 Oct 2026). The Capacity Market shares the GSP charging volume with BSUoS
    and LDES, so its exemption is recorded (``capacity_market_volume_*``) for the named
    sensitivities and never moves that volume.
    """
    out = dict(recovery)
    share = float(recovery["cfd_eligible_share"])
    cfd, ro = float(recovery["cfd_eligible_gsp_volume_mwh"]), float(recovery["ro_eligible_metered_supply_mwh"])
    cm = float(recovery["gsp_charging_volume_mwh"])
    cfd_after, ro_after, cm_after = cfd, ro, cm
    if settings["electrolysers_exempt"]:
        draw = float(recovery.get("electrolyser_draw_mwh") or 0.0)
        cfd_after, ro_after, cm_after = cfd - draw * share, ro - draw * share, cm - draw
    ro_after *= (1.0 - extra)
    if extra and settings["additional_exemption_covers_capacity_market"]:
        cm_after *= (1.0 - extra)
    if min(cfd_after, ro_after, cm_after) <= 0:
        raise ValueError("levy exemptions leave no eligible supply")
    out["cfd_eligible_gsp_volume_mwh"] = cfd_after
    out["ro_eligible_metered_supply_mwh"] = ro_after
    out["cfd_eligible_supplier_sales_mwh"] = ro_after   # the deprecated alias follows what it aliases
    record = {**settings, "applied": True,
              "cfd_eligible_gsp_volume_before_mwh": cfd, "cfd_eligible_gsp_volume_after_mwh": cfd_after,
              "ro_eligible_metered_supply_before_mwh": ro, "ro_eligible_metered_supply_after_mwh": ro_after,
              "cfd_eligible_scale": cfd / cfd_after, "ro_eligible_scale": ro / ro_after,
              # The RO line is scaled by the RO's base (scale_ro_for_exemptions).
              "eligible_scale": ro / ro_after,
              "capacity_market_volume_before_mwh": cm, "capacity_market_volume_after_mwh": cm_after,
              "gsp_chargeable_volume_before_mwh": cm, "gsp_chargeable_volume_after_mwh": cm_after}
    return out, record


#: Lines on the RO and FiT eligible base: the scope of the British Industrial
#: Competitiveness Scheme sensitivity, which exempts industry from the RO, FiT and the
#: Capacity Market (user, 6 Oct 2026; the CM joins when the case covers it).
ELIGIBLE_BASE_LINES = ("renewables_obligation", "feed_in_tariffs")
#: Lines on the CfD's eligible base (the CfD levy and the NZT DPA it funds), which the
#: electrolyser exemption also concentrates.
CFD_LEVY_LINES = ("cfd", "allocated_ccs", "remaining_ccs")


def levy_and_cm_sensitivities(frame, assumptions, recovery, consumption_kwh) -> dict:
    """Named sensitivities as changes to the ex-VAT bill (review 28 Sep 2026).

    ``levy_exemptions.sensitivities``: a further exempt share ``x`` raises every line on
    the eligible base by ``1/(1-x) - 1`` (and the Capacity Market when it is covered),
    because the same cost is spread over less supply. ``capacity_market.derivation.
    sensitivity_cost_gbp``: an alternative calendar-2030 CM cost over the same volume.
    ``<line>.derivation.sensitivity_gbp_per_household_year`` (a per-household sourced
    line, 29 Sep 2026: DUoS low and high): the alternative value, in report prices, less
    the line as billed.
    """
    value = frame.groupby("component_id")["annual_gbp_per_3100kwh_ex_vat"].sum()
    out = {}
    for name, block in ((assumptions.get("levy_exemptions") or {}).get("sensitivities") or {}).items():
        if block.get("electrolysers_exempt"):
            # FES 2050 decision 60: the policy-cost exemption case, exact on the run's own volumes:
            # the electrolysers' draw leaves the eligible bases (CfD, CCS, FiT, RO) and the Capacity
            # Market's GSP volume, as ``apply_levy_exemptions`` would remove it.
            _, record = apply_levy_exemptions(recovery, {"electrolysers_exempt": True})
            delta = (sum(float(value.get(line, 0.0)) for line in CFD_LEVY_LINES)
                     * (float(record["cfd_eligible_scale"]) - 1.0)
                     + sum(float(value.get(line, 0.0)) for line in ELIGIBLE_BASE_LINES)
                     * (float(record["ro_eligible_scale"]) - 1.0))
            before, after = record.get("capacity_market_volume_before_mwh"), record.get("capacity_market_volume_after_mwh")
            if before and after:
                delta += float(value.get("capacity_market", 0.0)) * (float(before) / float(after) - 1.0)
            out[f"levy_exemption_{name}"] = delta
            continue
        # F18 (user, 6 Oct 2026): a further exempt share (BICS) concentrates the RO and FiT, and
        # the Capacity Market when the case covers it; the CfD is not in its scope.
        x = float(block.get("additional_exempt_share", 0.0))
        scale = 1.0 / (1.0 - x) - 1.0
        delta = sum(float(value.get(line, 0.0)) for line in ELIGIBLE_BASE_LINES) * scale
        if block.get("additional_exemption_covers_capacity_market"):
            delta += float(value.get("capacity_market", 0.0)) * scale
        out[f"levy_exemption_{name}"] = delta
    cm = ((assumptions.get("sourced_externals") or {}).get("capacity_market") or {}).get("derivation") or {}
    if cm.get("sensitivity_cost_gbp") and recovery and recovery.get(cm.get("recovery_denominator")):
        denominator = float(recovery[cm["recovery_denominator"]])
        factor = float(cm.get("household_allocation_factor", 1.0))
        factor *= household_loss_for("capacity_market", cm, cm["recovery_denominator"], recovery)
        for name, cost in cm["sensitivity_cost_gbp"].items():
            out[f"capacity_market_{name}"] = ((float(cost) - float(cm["agreement_cost_gbp"])) / denominator
                                              * float(consumption_kwh) / 1000.0 * factor)
    report_year = (assumptions.get("benchmark") or {}).get("price_year")
    for key, block in (assumptions.get("sourced_externals") or {}).items():
        # FES 2050 decision 65: another line's alternative declared cost over its own denominator (the
        # 2050 BSUoS lines' ancillary low and high); the Capacity Market keeps its own branch above.
        # Decision 67: a register-booked line names the public-account keys of its alternatives.
        derivation = block.get("derivation") or {}
        if key == "capacity_market":
            continue
        if derivation.get("cost_from") and derivation.get("sensitivity_cost_from_keys"):
            base_cost, _ = register_booked_cost(key, derivation["cost_from"])
            derivation = {**derivation, "cost_gbp": base_cost, "sensitivity_cost_gbp": {
                name: register_booked_cost(key, {**derivation["cost_from"], "key": alt})[0]
                for name, alt in derivation["sensitivity_cost_from_keys"].items()}}
        if not derivation.get("sensitivity_cost_gbp"):
            continue
        if key not in value.index:
            raise ValueError(f"{key}: sensitivities declared for a line the bill does not carry")
        if block.get("loss_uplift") or not recovery or not recovery.get(derivation.get("recovery_denominator")):
            raise ValueError(f"{key}: a cost sensitivity needs a recovery denominator and no loss uplift")
        rebase = 1.0 if report_year is None or int(block["price_year"]) == int(report_year) else _rebasing_factor(
            block["rebasing"], f"{key} rebasing", block["price_year"], report_year)
        denominator = float(recovery[derivation["recovery_denominator"]])
        factor = float(derivation.get("household_allocation_factor", 1.0))
        # F18: a GSP-rate line's alternative is charged on the same loss-uplifted volume.
        factor *= household_loss_for(key, derivation, derivation["recovery_denominator"], recovery)
        for name, cost in derivation["sensitivity_cost_gbp"].items():
            out[f"{key}_{name}"] = ((float(cost) - float(derivation["cost_gbp"])) / denominator
                                    * float(consumption_kwh) / 1000.0 * factor * rebase)
    for key, block in (assumptions.get("sourced_externals") or {}).items():
        alternatives = (block.get("derivation") or {}).get("sensitivity_gbp_per_household_year") or {}
        if alternatives and key not in value.index:
            raise ValueError(f"{key}: sensitivities declared for a line the bill does not carry")
        for name, alt in alternatives.items():
            if report_year is None or int(alt["price_year"]) != int(report_year):
                raise ValueError(f"{key} sensitivity {name}: quote it in the report price year {report_year}")
            out[f"{key}_{name}"] = _finite(alt["value"], f"{key} sensitivity {name}", True) - float(value[key])
    return out


def scale_ro_for_exemptions(ro, record):
    """The RO line on the post-exemption eligible base (its scheme cost is unchanged)."""
    scale = float(record.get("ro_eligible_scale", record["eligible_scale"]))
    if ro is None or np.isclose(scale, 1.0, rtol=0, atol=1e-15):
        return ro
    out = dict(ro)
    out["gbp_per_3100kwh_ex_vat"] = float(ro["gbp_per_3100kwh_ex_vat"]) * scale
    out["gross_gbp_per_3100kwh_ex_vat"] = float(ro["gross_gbp_per_3100kwh_ex_vat"]) * scale
    out["levy_exemption_scale"] = scale
    return out


def recovery_case_id(case):
    """The case whose recovery-denominator file this case reads.

    A case on the same demand year but different balancing bids (the P2 months of
    item BM1) reads its parent's file and names it with ``recovery_case_id``; the
    file's own ``case_id`` is still checked, so it cannot be read by accident.
    """
    return case.get("recovery_case_id", case["case_id"])


def read_recovery_denominators(path, case_id, system_demand_mwh, volumes=None):
    """Read the sourced recovery denominators and check them against the model run.

    Item D1 (26 Sep 2026): levies are recovered on metered supply. The paying
    volume at the GSP is GB electrical load plus the electrolysers' electricity
    draw (``gsp_chargeable_volume_mwh``); the hydrogen offtake, MWh of hydrogen,
    pays nothing. ``final_demand_mwh`` is that volume divided by the GSP-to-meter
    loss factor, the metered supply a household's 3.1 MWh is a share of. Dividing
    a cost by it is the same as dividing by the GSP volume and uplifting the
    household's 3.1 MWh for losses, which is how the per-MWh GSP-rate lines are
    charged; the RO and FiT, levied on metered supply, divide by it directly.

    ``volumes`` (the projector's ``annual['demand_volumes_mwh']``), when given, is
    checked line by line so the file cannot drift from the run it describes.

    A file declaring ``recovery_basis: loss_inclusive_two_volume`` (F18) is read by
    :func:`_two_volume_denominators` instead; the D1 keys it still returns are
    deprecated aliases.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("case_id") != case_id:
        raise ValueError(f"{path} declares case {document.get('case_id')!r}, expected {case_id!r}")
    declared = _finite(document["system_demand_mwh"]["value"], "Declared system demand", True)
    if not np.isclose(declared, system_demand_mwh, rtol=1e-6, atol=1):
        raise ValueError(
            f"The recovery-demand file declares {declared:,.0f} MWh of system demand but the "
            f"model run has {system_demand_mwh:,.0f} MWh; regenerate the file for this run."
        )
    basis = document["metered_supply_basis"]
    gb_load = _finite(basis["gb_electrical_load_mwh"]["value"], "GB electrical load", True)
    draw = _finite(basis["electrolyser_electricity_draw_mwh"]["value"], "Electrolyser draw")
    offtake = _finite(basis["h2_offtake_mwh"]["value"], "Hydrogen offtake")
    if not np.isclose(gb_load + offtake, declared, rtol=1e-9, atol=1):
        raise ValueError("GB electrical load plus the hydrogen offtake does not equal the declared system demand")
    for key, value in (("gb_electrical_load_mwh", gb_load), ("electrolyser_draw_mwh", draw),
                       ("h2_offtake_mwh", offtake)):
        if volumes is not None and not np.isclose(value, float(volumes[key]), rtol=1e-6, atol=1):
            raise ValueError(
                f"The recovery-demand file declares {key} = {value:,.0f} MWh but the model run has "
                f"{float(volumes[key]):,.0f} MWh; regenerate the file for this run."
            )
    if document.get("recovery_basis") == TWO_VOLUME_BASIS:
        return _two_volume_denominators(path, document, gb_load, draw, offtake)
    if document.get("recovery_basis") not in (None, D1_BASIS):
        raise ValueError(f"{path}: unknown recovery_basis {document.get('recovery_basis')!r}; "
                         f"expected {TWO_VOLUME_BASIS!r} or none (item D1)")
    gsp = _finite(basis["gsp_chargeable_volume_mwh"]["value"], "GSP chargeable volume", True)
    if not np.isclose(gsp, gb_load + draw, rtol=1e-9, atol=1e-6):
        raise ValueError("gsp_chargeable_volume_mwh does not equal GB electrical load plus electrolyser draw")
    loss = _finite(basis["gsp_to_meter_loss_factor"]["value"], "GSP-to-meter loss factor", True)
    if loss < 1.0:
        raise ValueError("The GSP-to-meter loss factor cannot be below one")
    final_demand = _finite(document["final_demand_mwh"]["value"], "Final demand", True)
    if not np.isclose(final_demand, gsp / loss, rtol=1e-9, atol=1e-6):
        raise ValueError("final_demand_mwh does not equal the GSP chargeable volume over the loss factor")
    eligible_block = document["cfd_eligible_supply"]
    share = _finite(eligible_block["eligible_share_of_supplier_sales"]["value"], "Eligible share", True)
    if share > 1:
        raise ValueError("CfD eligible supply cannot exceed supplier sales")
    eligible = _finite(eligible_block["eligible_supplier_sales_mwh"]["value"], "Eligible sales", True)
    if not np.isclose(eligible, final_demand * share, rtol=1e-9, atol=1e-6):
        raise ValueError("eligible_supplier_sales_mwh does not equal final demand times its declared share")
    ro_block = document.get("ro_obligation_eligible_supply_mwh")
    if ro_block is not None and not np.isclose(float(ro_block["value"]), eligible, rtol=1e-9, atol=1e-6):
        raise ValueError("ro_obligation_eligible_supply_mwh is declared equal to the CfD eligible supply but is not")
    return {
        "recovery_basis": D1_BASIS,
        "final_demand_mwh": final_demand,
        "metered_supply_mwh": final_demand,
        "cfd_eligible_supplier_sales_mwh": eligible,
        "cfd_eligible_share": share,
        "gsp_chargeable_volume_mwh": gsp,
        "gsp_to_meter_loss_factor": loss,
        "gb_electrical_load_mwh": gb_load,
        "electrolyser_draw_mwh": draw,
        "h2_offtake_mwh": offtake,
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "sources": {
            "gsp_to_meter_loss_factor": basis["gsp_to_meter_loss_factor"]["source"],
            "cfd_eligible_supply": eligible_block["eligible_share_of_supplier_sales"]["source"],
        },
    }


def conversion_denominators(document):
    """The two deprecated summary denominators, read straight from a recovery-demand file.

    ``final_demand_mwh`` is metered supply; ``eligible_mwh`` is the volume a CfD-levy line
    divides by: on the two-volume basis (F18) its eligible GSP volume over the household
    loss factor, so ``system x factor / eligible_mwh x 3.1`` is the household line, as the
    projector's summary reports it (``cfd_recovery_denominator_mwh``). For a caller that
    must build lines before the projector has run (the ownership projector's private case).
    """
    if document.get("recovery_basis") == TWO_VOLUME_BASIS:
        basis = document["metered_supply_basis"]
        return {"final_demand_mwh": float(basis["metered_supply_mwh"]["value"]),
                "eligible_mwh": (float(document["cfd_eligible_supply"]["cfd_eligible_gsp_volume_mwh"]["value"])
                                 / float(basis["household_loss_factor"]["value"]))}
    return {"final_demand_mwh": float(document["final_demand_mwh"]["value"]),
            "eligible_mwh": float(document["cfd_eligible_supply"]["eligible_supplier_sales_mwh"]["value"])}


def _declared_factor(block, name):
    """A recovery factor of at least one, inside its declared range when it declares one."""
    value = _finite(block["value"], name, True)
    if value < 1.0:
        raise ValueError(f"{name} cannot be below one")
    low, high = block.get("range") or (value, value)
    if not float(low) <= value <= float(high):
        raise ValueError(f"{name} {value} is outside its declared range [{low}, {high}]")
    return value


def _two_volume_denominators(path, document, gb_load, draw, offtake):
    """F18 (6 Oct 2026): the recovery denominators on the loss-inclusive two-volume basis.

    The model's GB load is a transmission-entry volume (it contains network losses), so
    metered supply is load / system_to_metered_factor plus the electrolysers' draw, and
    the GSP charging volume is that metered load x metered_to_gsp_factor plus the draw.
    Each identity in the file is checked; the deprecated aliases must equal what they alias.
    """
    basis = document["metered_supply_basis"]
    to_metered = _declared_factor(basis["system_to_metered_factor"], "system_to_metered_factor")
    to_gsp = _declared_factor(basis["metered_to_gsp_factor"], "metered_to_gsp_factor")
    household_loss = _declared_factor(basis["household_loss_factor"], "household_loss_factor")
    metered = _finite(basis["metered_supply_mwh"]["value"], "Metered supply", True)
    if not np.isclose(metered, gb_load / to_metered + draw, rtol=1e-9, atol=1e-6):
        raise ValueError("metered_supply_mwh does not equal GB load / system_to_metered_factor + electrolyser draw")
    gsp = _finite(basis["gsp_charging_volume_mwh"]["value"], "GSP charging volume", True)
    if not np.isclose(gsp, gb_load / to_metered * to_gsp + draw, rtol=1e-9, atol=1e-6):
        raise ValueError("gsp_charging_volume_mwh does not equal GB load / system_to_metered_factor x "
                         "metered_to_gsp_factor + electrolyser draw")
    eligible_block = document["cfd_eligible_supply"]
    share = _finite(eligible_block["eligible_share_of_supplier_sales"]["value"], "Eligible share", True)
    if share > 1:
        raise ValueError("CfD eligible supply cannot exceed supplier sales")
    cfd_gsp = _finite(eligible_block["cfd_eligible_gsp_volume_mwh"]["value"], "CfD eligible GSP volume", True)
    if not np.isclose(cfd_gsp, gsp * share, rtol=1e-9, atol=1e-6):
        raise ValueError("cfd_eligible_gsp_volume_mwh does not equal the GSP charging volume times its eligible share")
    ro = _finite(document["ro_obligation_eligible_supply_mwh"]["value"], "RO eligible supply", True)
    if not np.isclose(ro, metered * share, rtol=1e-9, atol=1e-6):
        raise ValueError("ro_obligation_eligible_supply_mwh does not equal metered supply times the eligible share")
    # The deprecated names must still mean what they alias (F18), or a reader of the old key is misled.
    for name, value, target in (("metered_supply_basis.gsp_chargeable_volume_mwh",
                                 basis["gsp_chargeable_volume_mwh"]["value"], gsp),
                                ("final_demand_mwh", document["final_demand_mwh"]["value"], metered),
                                ("cfd_eligible_supply.eligible_supplier_sales_mwh",
                                 eligible_block["eligible_supplier_sales_mwh"]["value"], ro)):
        if not np.isclose(float(value), target, rtol=1e-9, atol=1e-6):
            raise ValueError(f"{name} is a deprecated alias and must equal the volume it aliases")
    return {
        "recovery_basis": TWO_VOLUME_BASIS,
        "metered_supply_mwh": metered,
        "gsp_charging_volume_mwh": gsp,
        "cfd_eligible_gsp_volume_mwh": cfd_gsp,
        "ro_eligible_metered_supply_mwh": ro,
        "household_loss_factor": household_loss,
        "system_to_metered_factor": to_metered,
        "metered_to_gsp_factor": to_gsp,
        "cfd_eligible_share": share,
        # DEPRECATED aliases (F18): read by code written for item D1 until it reads the volumes above.
        "final_demand_mwh": metered,
        "gsp_chargeable_volume_mwh": gsp,
        "cfd_eligible_supplier_sales_mwh": ro,
        "deprecated_aliases": dict(DEPRECATED_RECOVERY_ALIASES),
        "gb_electrical_load_mwh": gb_load,
        "electrolyser_draw_mwh": draw,
        "h2_offtake_mwh": offtake,
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "sources": {
            "system_to_metered_factor": basis["system_to_metered_factor"]["source"],
            "metered_to_gsp_factor": basis["metered_to_gsp_factor"]["source"],
            "household_loss_factor": basis["household_loss_factor"]["source"],
            "cfd_eligible_supply": eligible_block["eligible_share_of_supplier_sales"]["source"],
        },
    }


def read_ro_scheme_cost(path, relief_case, report_price_year):
    """Read the RO scheme cost and select the declared Exchequer-relief case.

    The relief case has to be chosen explicitly. Ending the 75% domestic relief
    in March 2029 makes calendar 2030 the first full year of gross recovery, and
    the two bookends differ by more than the external allowance the line
    replaces, so a silent default would hide the largest single choice in the
    policy block.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = document["relief_cases"]
    if relief_case not in cases:
        raise ValueError(
            f"ro_relief_case {relief_case!r} is not one of {sorted(cases)}. The RO line depends "
            "on whether the domestic Exchequer relief continues, so the case must be declared."
        )
    value = document["notional_roc_value"]
    if int(value["report_price_year"]) != int(report_price_year):
        raise ValueError(
            f"RO scheme cost is on the {value['report_price_year']} basis, but the report basis "
            f"is {report_price_year}"
        )
    # Task Y13: a bill year after the scheme closed (31 March 2037) carries a declared nil.
    # Only the builder's closed-scheme summary may be zero; any other must be positive.
    closed = document.get("method_label") == "scheme_closed"
    if closed:
        if float(document["gross_gbp_per_3100kwh_ex_vat"]) != 0.0 or any(
                float(c["gbp_per_3100kwh_ex_vat"]) != 0.0 for c in cases.values()):
            raise ValueError("a closed RO scheme summary must carry a nil gross cost and nil relief cases")
        return {
            "relief_case": relief_case, "exchequer_relief_share": 0.0,
            "gbp_per_3100kwh_ex_vat": 0.0, "gross_gbp_per_3100kwh_ex_vat": 0.0,
            "relief_cases": {name: 0.0 for name in cases},
            "reference_gbp_per_3100kwh_ex_vat": float(document["reference_external_gbp_per_3100kwh_ex_vat"]),
            "roc_volume": 0.0, "gb_share_of_uk_obligation": None, "eligible_capacity_mw": 0.0,
            "expired_capacity_mw": float(document["expired_capacity_mw"]),
            "notional_roc_value_gbp_per_roc": 0.0, "obligation_year": document["obligation_year"],
            "method_label": "scheme_closed", "scheme_closed": document["scheme_closed"],
            "report_price_year": int(value["report_price_year"]), "sources": document.get("sources", {}),
            "path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
    return {
        "relief_case": relief_case,
        "exchequer_relief_share": float(cases[relief_case]["exchequer_relief_share"]),
        "gbp_per_3100kwh_ex_vat": _finite(
            cases[relief_case]["gbp_per_3100kwh_ex_vat"], "RO allowance"),
        "gross_gbp_per_3100kwh_ex_vat": _finite(
            document["gross_gbp_per_3100kwh_ex_vat"], "Gross RO allowance", True),
        "relief_cases": {name: float(case["gbp_per_3100kwh_ex_vat"]) for name, case in cases.items()},
        "reference_gbp_per_3100kwh_ex_vat": float(
            document["reference_external_gbp_per_3100kwh_ex_vat"]),
        "roc_volume": float(document["roc_volume"]),
        "gb_share_of_uk_obligation": (document.get("uk_obligation_split") or {}).get("gb_share_of_uk_obligation"),
        "eligible_capacity_mw": float(document["eligible_capacity_mw"]),
        "expired_capacity_mw": float(document["expired_capacity_mw"]),
        "notional_roc_value_gbp_per_roc": float(
            value["notional_roc_value_gbp_per_roc_report_prices"]),
        # F18: the scheme cost (report prices) and the volume it was spread over, so the
        # bill can check the summary was built on the recovery file it reads.
        "gross_scheme_cost_gbp": document.get("gross_scheme_cost_gbp"),
        "obligation_eligible_supply_mwh": document.get("obligation_eligible_supply_mwh"),
        "obligation_year": document["obligation_year"],
        "method_label": document["method_label"],
        "report_price_year": int(value["report_price_year"]),
        "sources": document.get("sources", {}),
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def read_balancing_costs(path, scenarios, model_price_year):
    """Read the monthly constraint extraction produced by the balancing-cost rule.

    Item B8: the constraint cost must exclude the VOLL feasibility slack and the
    non-participant turn-down priced at the penalty bid, both of which sit inside
    the accepted offer and bid costs; an extraction that predates the exclusion is
    refused, and the stated identity is re-checked here.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    annual = document["annual"]
    if int(annual["model_price_year"]) != int(model_price_year):
        raise ValueError(
            f"Balancing costs are on the {annual['model_price_year']} basis, but the case "
            f"declares {model_price_year}"
        )
    if annual.get("institution") == CENTRAL_DISPATCH_INSTITUTION:
        return _read_central_dispatch_costs(path, document)
    if not annual.get("constraint_cost_excludes_slack_and_penalty"):
        raise ValueError(
            f"{path} predates item B8: its constraint cost still carries the VOLL feasibility slack and the "
            "non-participant penalty turn-down. Regenerate the balancing-cost extraction."
        )
    penalty = _finite(annual["non_participant_turn_down_cost_at_penalty_gbp"], "Non-participant penalty turn-down")
    slack_in = _finite(annual["feasibility_slack_cost_in_accepted_costs_gbp"], "Slack inside accepted costs")
    # Item C7 / BM1 P3: the decision-20 tie steps sit inside the accepted bid cost and
    # the extraction nets them out of the constraint cost; zero for every other case.
    tie_steps = _finite(annual.get("tie_break_step_cost_gbp", 0.0), "Tie-break steps inside accepted bid cost")
    expected = (float(annual["accepted_offer_cost_gbp"]) + float(annual["accepted_bid_cost_gbp"])
                - slack_in - penalty - tie_steps)
    if annual.get("cfd_curtailment_compensation_treatment") == "add_post_hoc":
        expected += float(annual["cfd_curtailment_compensation_gbp"])
    if annual.get("include_interconnector_retrading"):
        expected += float(annual.get("interconnector_retrading_cost_gbp") or 0.0)
    # Item F24.2: a price-level extraction moves the CfD contract-value bids to the mapped reference.
    level_change = _finite(annual.get("cfd_bid_price_level_change_gbp", 0.0), "CfD bid change at the price level")
    expected += level_change
    if not np.isclose(float(annual["constraint_cost_gbp"]), expected, rtol=1e-9, atol=1.0):
        raise ValueError(
            f"{path}: constraint cost {float(annual['constraint_cost_gbp']):,.0f} does not equal accepted offer + bid "
            f"- feasibility slack - non-participant penalty - tie-break steps (+ compensation, + re-trading, "
            f"+ the CfD bids' price-level change) = {expected:,.0f}"
        )
    out = {
        "constraint_cost_gbp": _finite(annual["constraint_cost_gbp"], "Constraint cost"),
        "accepted_offer_cost_gbp": _finite(annual["accepted_offer_cost_gbp"], "Accepted offer cost"),
        "accepted_bid_cost_gbp": _finite(annual["accepted_bid_cost_gbp"], "Accepted bid cost"),
        "cfd_curtailment_compensation_gbp": _finite(
            annual["cfd_curtailment_compensation_gbp"], "CfD curtailment compensation"),
        "cfd_curtailed_mwh": _finite(annual["cfd_curtailed_mwh"], "CfD curtailed volume"),
        "feasibility_slack_cost_gbp": _finite(annual["feasibility_slack_cost_gbp"], "Slack cost"),
        "feasibility_slack_energy_mwh": _finite(annual["feasibility_slack_energy_mwh"], "Slack energy"),
        "non_participant_turn_down_cost_at_penalty_gbp": penalty,
        "feasibility_slack_cost_in_accepted_costs_gbp": slack_in,
        "non_participant_turn_down_mwh": _finite(annual.get("non_participant_turn_down_mwh", 0.0),
                                                 "Non-participant turn-down volume"),
        "by_carrier": document.get("by_carrier", []),
        "unlabelled_support": document["annual"].get("unlabelled_support", []),
        # Optional: interconnector re-trading valued at the CP30 bid/ask prices.
        # Written by the balancing-cost extraction when its rule enables it;
        # None when the extraction predates the key.
        "interconnector_retrading_cost_gbp": (
            _finite(annual["interconnector_retrading_cost_gbp"], "Interconnector re-trading cost")
            if annual.get("interconnector_retrading_cost_gbp") is not None else None),
        "include_interconnector_retrading": annual.get("include_interconnector_retrading"),
        # How the CfD compensation entered the constraint cost, so the bill text
        # does not claim it was added when it was only reconciled.
        "cfd_curtailment_compensation_treatment": annual.get("cfd_curtailment_compensation_treatment"),
        # The negative pricing rule the balancing bids applied, as recorded in the
        # contract-value bid audits (None when no audit recorded it).
        "balancing_negative_price_rule": annual.get("balancing_negative_price_rule"),
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    # Only a run with tie steps (P3) carries the key, so every other bill is unchanged.
    if tie_steps:
        out["tie_break_step_cost_gbp"] = tie_steps
    # Item F24.2: only a price-level extraction carries these, so every model-price bill is unchanged.
    if annual.get("price_level") is not None:
        out["price_level"] = str(annual["price_level"])
        out["cfd_bid_price_level_change_gbp"] = level_change
        out["price_level_mapping_check"] = annual.get("price_level_mapping_check")
    return out


#: P1/P2 under central dispatch (user decisions of 14 and 20 Oct 2026): a forecast case's ``institution``, and
#: the balancing summary ``scripts/consumer_bill/extract_central_dispatch_costs.py`` writes for it.
CENTRAL_DISPATCH_INSTITUTION = "central_dispatch"
#: The I3 constraint line's components (GBP), which must add up to it.
CENTRAL_DISPATCH_CONSTRAINT_PARTS = ("cost_recovery_gbp", "firm_access_gbp", "storage_make_whole_gbp",
                                     "interconnector_make_whole_gbp", "forecast_error_charge_gbp",
                                     "counter_trade_premium_gbp")


def _read_central_dispatch_costs(path, document):
    """:func:`read_balancing_costs` for a central-dispatch (I3) extraction: the FTI parts, re-checked to add up.

    There is no balancing market, so no accepted offers or bids, no slack inside them and no
    interconnector re-trading: the line is cost recovery, firm access (with the lost CfD and
    ROC support), the storage and interconnector make-whole, the borrowed forecast-error charge
    and (when declared) the central balancing stage's counter-trade premium.
    """
    annual = document["annual"]
    parts = {key: _finite(annual[key], key) for key in CENTRAL_DISPATCH_CONSTRAINT_PARTS}
    constraint = _finite(annual["constraint_cost_gbp"], "Constraint cost")
    if not np.isclose(constraint, sum(parts.values()), rtol=1e-9, atol=1.0):
        raise ValueError(f"{path}: the central-dispatch constraint cost {constraint:,.0f} is not the sum of its parts "
                         f"({sum(parts.values()):,.0f})")
    if annual.get("include_interconnector_retrading") or float(annual.get("interconnector_retrading_cost_gbp") or 0.0):
        raise ValueError(f"{path}: a central-dispatch year has no interconnector re-trading")
    if (annual.get("solve_vintage") or {}).get("mixed_vintage") and not annual.get("mixed_vintage_allowed"):
        raise ValueError(f"{path}: the market and central-dispatch years are from different solve runs")
    rents = {str(k): _finite(v, f"{k} rent change") for k, v in (annual.get("interconnector_rent_change_by_cable_gbp")
                                                                 or {}).items()}
    out = {
        "institution": CENTRAL_DISPATCH_INSTITUTION,
        "constraint_cost_gbp": constraint,
        **parts,
        "accepted_offer_cost_gbp": 0.0, "accepted_bid_cost_gbp": 0.0,
        "cfd_curtailment_compensation_gbp": _finite(annual["cfd_curtailment_compensation_gbp"], "Lost CfD support"),
        "cfd_curtailed_mwh": _finite(annual["cfd_curtailed_mwh"], "Lost CfD volume"),
        "feasibility_slack_cost_gbp": 0.0, "feasibility_slack_energy_mwh": 0.0,
        "non_participant_turn_down_cost_at_penalty_gbp": 0.0, "feasibility_slack_cost_in_accepted_costs_gbp": 0.0,
        "non_participant_turn_down_mwh": 0.0,
        "by_carrier": document.get("by_carrier", []),
        "unlabelled_support": annual.get("unlabelled_support", []),
        "interconnector_retrading_cost_gbp": 0.0,
        "include_interconnector_retrading": False,
        "cfd_curtailment_compensation_treatment": annual.get("cfd_curtailment_compensation_treatment"),
        "balancing_negative_price_rule": annual.get("balancing_negative_price_rule"),
        "interconnector_rent_change_by_cable_gbp": rents,
        "realtime_correction_replaced_gbp": float(annual.get("realtime_correction_replaced_gbp") or 0.0),
        "counter_trade_premium_on_constraint_line": bool(annual.get("counter_trade_premium_on_constraint_line")),
        "counter_trade_premium_fti_treatment": annual.get("counter_trade_premium_fti_treatment"),
        "market_constraint_cost_gbp": annual.get("market_constraint_cost_gbp"),
        "solve_vintage": annual.get("solve_vintage"),
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return out


def rent_by_cable_for_institution(rent_by_cable: dict, balancing: dict | None) -> dict:
    """The cables' Stage 1 rent the cap-and-floor regimes assess, under the case's institution.

    Under central dispatch (decision 4, 20 Oct 2026) each cable's rent is the market
    summaries' moved by the FTI settlement's own change (the central schedule's rent with
    its day-ahead make-whole, less the market schedule's), so the institution moves the rent
    and the two rent conventions never mix. Unchanged for every market case.
    """
    if not balancing or balancing.get("institution") != CENTRAL_DISPATCH_INSTITUTION:
        return rent_by_cable
    change = balancing.get("interconnector_rent_change_by_cable_gbp") or {}
    if unknown := sorted(set(change) - set(rent_by_cable)):
        raise ValueError(f"the central-dispatch rent change names cables the market summaries do not: {unknown}")
    return {name: float(rent) + float(change.get(name, 0.0)) for name, rent in rent_by_cable.items()}


def interconnector_pass_through(rent_by_cable, rule, model_price_year):
    """Return what the regulated interconnector regimes hand back to GB consumers.

    ``rent_by_cable`` is the whole-cable congestion rent from the Stage 1
    schedule in model prices (GBP). Each cable is declared in ``rule['cables']``:

    * ``cap_and_floor`` (Ofgem, OFG1164 section 4.6): the GB-regulated revenue is
      ``gb_share_of_rent`` x rent; revenue above the cap is paid to consumers and
      a shortfall below the floor is topped up by them, both through TNUoS. Levels
      quoted for the GB share apply as they are; levels quoted for the whole link
      (Nemo Link, regulated jointly with CREG) are scaled by the same share, which
      is the 50:50 split of any excess Ofgem applies there. Levels are indexed by
      the declared index from their price year to the model price year, so they
      are flat in real terms of that index.
    * ``use_of_revenues_share`` (IFA): a declared share of the GB owner's revenue
      is referred to network tariffs under Article 19(3) of the retained
      Electricity Regulation.
    * ``exempt_merchant`` / ``other_jurisdiction``: nothing returns to GB bills.

    The year is assessed on its own (the default regime assesses five-year
    periods, some variations annually); Capacity Market income, which the cap
    test also counts, is not added here (see the rule's notes). Returns the
    signed credit (positive = paid to consumers) and a per-cable audit.
    """
    cables = rule.get("cables") or {}
    index = rule["index"]
    if int(index["to_price_year"]) != int(model_price_year):
        raise ValueError(
            f"interconnector_cap_and_floor.index lands on {index['to_price_year']}, not the model price "
            f"year {model_price_year}; the rent is in model prices"
        )
    to_index = _finite(index["to_index"], "interconnector index to_index", True)
    unknown = sorted(set(rent_by_cable).difference(cables))
    if unknown:
        raise ValueError(
            f"Interconnectors with congestion rent but no declared regime: {unknown}. Declare each in "
            "model_linked_rules.interconnector_cap_and_floor.cables, so none is dropped or credited by default."
        )
    rows = []
    for name, rent in sorted(rent_by_cable.items()):
        spec = cables[name]
        regime = str(spec.get("regime"))
        if regime not in INTERCONNECTOR_REGIMES:
            raise ValueError(f"{name}: regime must be one of {INTERCONNECTOR_REGIMES}, not {regime!r}")
        share = _finite(spec.get("gb_share_of_rent", rule.get("default_gb_share_of_rent", 0.5)),
                        f"{name} gb_share_of_rent")
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"{name}: gb_share_of_rent must lie in [0, 1]")
        row = {"interconnector": name, "regime": regime, "congestion_rent_gbp": float(rent),
               "gb_share_of_rent": share if regime in ("cap_and_floor", "use_of_revenues_share") else 0.0,
               "gb_regulated_revenue_gbp": 0.0, "cap_gbp": None, "floor_gbp": None,
               "above_cap_to_consumers_gbp": 0.0, "floor_top_up_from_consumers_gbp": 0.0,
               "tariff_share_to_consumers_gbp": 0.0, "credit_to_consumers_gbp": 0.0}
        if regime == "cap_and_floor":
            factor = to_index / _finite(spec["index_from"], f"{name} index_from", True)
            scale = share if spec.get("levels_apply_to") == "whole_link" else 1.0
            if spec.get("levels_apply_to") not in ("gb_share", "whole_link"):
                raise ValueError(f"{name}: levels_apply_to must be gb_share or whole_link")
            cap = _finite(spec["cap_gbp_m"], f"{name} cap", True) * 1e6 * factor * scale
            floor = _finite(spec["floor_gbp_m"], f"{name} floor", True) * 1e6 * factor * scale
            if floor > cap:
                raise ValueError(f"{name}: floor above cap")
            revenue = share * float(rent)
            above = max(revenue - cap, 0.0)
            top_up = max(floor - revenue, 0.0)
            row.update(gb_regulated_revenue_gbp=revenue, cap_gbp=cap, floor_gbp=floor,
                       above_cap_to_consumers_gbp=above, floor_top_up_from_consumers_gbp=top_up,
                       credit_to_consumers_gbp=above - top_up)
        elif regime == "use_of_revenues_share":
            tariff_share = _finite(spec["tariff_share_of_gb_revenue"], f"{name} tariff share")
            if not 0.0 <= tariff_share <= 1.0:
                raise ValueError(f"{name}: tariff_share_of_gb_revenue must lie in [0, 1]")
            revenue = share * float(rent)
            # A negative rent year (flows against the spread) refers nothing.
            to_tariffs = max(revenue, 0.0) * tariff_share
            row.update(gb_regulated_revenue_gbp=revenue, tariff_share_to_consumers_gbp=to_tariffs,
                       credit_to_consumers_gbp=to_tariffs)
        rows.append(row)
    frame = pd.DataFrame(rows)
    credit = float(frame.credit_to_consumers_gbp.sum()) if not frame.empty else 0.0
    by_regime = (frame.groupby("regime").congestion_rent_gbp.sum().to_dict() if not frame.empty else {})
    return {"credit_to_consumers_gbp": credit,
            "congestion_rent_gbp": float(sum(rent_by_cable.values())),
            "congestion_rent_by_regime_gbp": {str(k): float(v) for k, v in by_regime.items()},
            # The row dicts, not frame records: a None cap would become NaN and break the JSON.
            "by_cable": rows}


#: Network bounds (user decision, 18-19 Oct 2026): a forecast case may name its network-delivery
#: case's TNUoS adjustment, ``{file: data/.../2030_tnuos_network_delivery.json, case: <name>}``.
TNUOS_NETWORK_DELIVERY_KEY = "tnuos_network_delivery"


def _resolve_repo_path(path) -> Path:
    full = Path(path)
    return full if full.is_absolute() or full.exists() else Path(__file__).resolve().parents[2] / full


def apply_tnuos_network_delivery(assumptions: dict, case: dict | None) -> dict:
    """The assumptions with the TNUoS line moved to the case's network-delivery case (a copy; unchanged without one).

    The case's ``tnuos_network_delivery`` names a case in the network-delivery file
    (``data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json``), which declares
    a change in the onshore TOs' TNUoS income as a share of NESO's Five-Year View income per
    fiscal year (``to_income_gbp_m``). Generation TNUoS is capped, so the whole change falls on
    the demand residual, which is levied per site by band in proportion: one household's residual
    moves by ``change x domestic residual per site / demand residual revenue`` of that year (the
    tnuos line's own NESO derivation, both nominal). Each ``period_blend`` component (and the
    line's quoted FY value) moves by its own year's amount in that year's money, so the line's
    calendar blend and CPIH deflation apply to it unchanged. The adjustment is recorded on the
    line (``network_delivery_adjustment``).

    Raises:
        ValueError: the file or case is missing, a fiscal year of the blend has no TO income in the
            file, or the line's derivation lacks that year's residual figures.
    """
    spec = (case or {}).get(TNUOS_NETWORK_DELIVERY_KEY)
    if not spec:
        return assumptions
    if not isinstance(spec, dict) or not spec.get("file") or not spec.get("case"):
        raise ValueError(f"{TNUOS_NETWORK_DELIVERY_KEY} must be {{file, case}}, got {spec!r}")
    document = json.loads(_resolve_repo_path(spec["file"]).read_text(encoding="utf-8"))
    cases = document.get("cases") or {}
    if spec["case"] not in cases:
        raise ValueError(f"{spec['file']} has no network-delivery case {spec['case']!r}; it has {sorted(cases)}")
    entry = cases[spec["case"]]
    share = _finite(entry.get("change_share"), f"{spec['case']} change_share")
    if not -1.0 <= share <= 1.0:
        raise ValueError(f"{spec['case']}: change_share {share} is not a share of TO income")
    out = copy.deepcopy(assumptions)
    tnuos = out["sourced_externals"]["tnuos"]
    derivation = tnuos["derivation"]
    blend = tnuos.get("period_blend") or []
    if not blend:
        raise ValueError("tnuos_network_delivery needs the tnuos line's period_blend (fiscal-year components)")
    record = {"file": str(spec["file"]), "case": str(spec["case"]), "change_share": share,
              "status": entry.get("status"), "network": entry.get("network"),
              "basis": "onshore TO TNUoS income x change_share, on the demand residual, at the household's "
                       "residual share of each fiscal year (nominal), before the line's blend and deflation",
              "by_period": {}}
    income = document.get("to_income_gbp_m") or {}
    for component in blend:
        period = str(component["period"])
        if period not in income:
            raise ValueError(f"{spec['file']}: to_income_gbp_m has no {period}")
        label = period.removeprefix("FY").replace("/", "_")
        revenue = _finite(derivation[f"demand_residual_revenue_gbp_m_{label}"], f"tnuos demand residual {period}", True)
        domestic = _finite(derivation[f"domestic_residual_gbp_per_site_day_{label}"], f"tnuos domestic residual {period}",
                           True) * 365.0
        change_gbp_m = share * _finite(income[period], f"TO income {period}", True)
        change_household = change_gbp_m / revenue * domestic
        before = float(component["value"])
        component["value"] = before + change_household
        if float(tnuos.get("value", np.nan)) == before and int(component.get("price_year", -1)) == int(
                tnuos.get("price_year", -2)):
            tnuos["value"] = component["value"]
        record["by_period"][period] = {"to_income_change_gbp_m_nominal": change_gbp_m,
                                       "household_change_gbp_nominal": change_household,
                                       "value_before": before, "value_after": component["value"]}
    tnuos["network_delivery_adjustment"] = record
    if share:
        tnuos["label"] = f"{tnuos.get('label', 'TNUoS')}; {spec['case']} network ({share:+.0%} of onshore TO income)"
    return out


def tnuos_household_share(assumptions):
    """Share of a change in TNUoS demand-residual revenue borne by one benchmark household.

    Since the Targeted Charging Review the demand residual is the balancing item of
    TNUoS: a pass-through that lowers the revenue to collect lowers the residual,
    which is levied per site by band. Scaling every band proportionally, one
    household bears its domestic residual over the total residual revenue; both
    come from the tnuos line's own NESO derivation, in the same money, so the ratio
    carries no price basis.
    """
    derivation = assumptions["sourced_externals"]["tnuos"]["derivation"]
    # The calendar-year blend when the line declares one (28 Sep 2026), else the
    # fiscal year starting in the bill's calendar year (item Y4: FY 2030/31 on the
    # CP30 year, read from ``demand_residual_revenue_gbp_m_2030_31``).
    blend = derivation.get("calendar_blend") or {}
    per_household = _finite(blend.get("domestic_residual_gbp_per_year", derivation["domestic_residual_gbp_per_year"]),
                            "domestic TNUoS residual", True)
    if "demand_residual_revenue_gbp_m" in blend:
        revenue_value = blend["demand_residual_revenue_gbp_m"]
    else:
        from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR, fiscal_year_label

        bill_year = int((assumptions.get("benchmark") or {}).get("bill_calendar_year") or DEFAULT_MODEL_YEAR)
        revenue_value = derivation[f"demand_residual_revenue_gbp_m_{fiscal_year_label(bill_year)}"]
    revenue = _finite(revenue_value, "TNUoS demand residual revenue", True)
    return per_household / (revenue * 1e6)


def price_level_of(case) -> str:
    """The bill case's wholesale price level (item F12): ``model`` unless the case says ``cp32``."""
    level = str((case or {}).get("price_level", DEFAULT_PRICE_LEVEL))
    if level not in PRICE_LEVELS:
        raise ValueError(f"price_level must be one of {PRICE_LEVELS}, got {level!r}")
    return level


def load_price_level_curve(block=None):
    """The ``cp32`` mapping settings and duration curve, loaded as the asset settlement at that level loads them."""
    settings = plm.settings_from_config(block)
    source = Path(settings["source_file"])
    curve = plm.load_duration_curve(source if source.exists() else PROJECT_ROOT / source, settings["pathway"],
                                    float(settings["rebasing_to_model_price_year"]))
    return settings, curve


def map_year_to_price_level(hourly, block):
    """``(mapped, record)``: the year's hours on CP.32 exactly as the ``cp32`` bill maps them (item F12).

    ``hourly`` carries ``price_gbp_per_mwh`` and ``hours`` for the whole year on the
    model's price (``read_hourly_inputs``); ``block`` is the case's ``price_level_mapping``.
    The ownership layer's price-level twins map through here too (12 Oct 2026), so their
    settlement and the bill settle on the same hourly series.
    """
    settings, curve = load_price_level_curve(block)
    # Prices within GBP 1 of zero tie (12 Oct 2026): the solver's 1e-6 noise among the
    # zero-price third of the year would otherwise decide their mapped prices.
    mapped = plm.map_market_year(hourly, curve, zero_band=CP32_ZERO_BAND_GBP_PER_MWH)
    record = plm.mapping_summary(hourly, mapped, curve, settings)
    record["zero_band_gbp_per_mwh"] = CP32_ZERO_BAND_GBP_PER_MWH
    return mapped, record


def _apply_cp32_price_level(monthly, hourly, block):
    """Map the whole year onto CP.32 and put the mapped price on every month's frame (item F12).

    ``hourly`` is the year on the model's price, so the year is ranked together; each
    month's frame keeps the model price as ``model_price_gbp_per_mwh`` and settles on
    the mapped ``price_gbp_per_mwh`` from here on. Returns the mapping summary.
    """
    mapped, record = map_year_to_price_level(hourly, block)
    for month in monthly:
        frame = month["frame"]
        frame["model_price_gbp_per_mwh"] = frame["price_gbp_per_mwh"].copy()
        frame["price_gbp_per_mwh"] = mapped.reindex(frame.index).to_numpy()
    return record


def _signed_interconnector_links(network_path):
    """``{cable: (signed link, efficiency)}`` for the piecewise-priced cables of a saved wholesale network."""
    return _cached(("ic_links", str(network_path)), lambda: _read_signed_interconnector_links(network_path))


def _read_signed_interconnector_links(network_path):
    from scripts.interconnectors.market_coupling import _piecewise_links

    with xr.open_dataset(network_path, engine="netcdf4") as ds:
        needed = ("links_i", "links_interconnector_name", "links_foreign_market_pricing", "links_efficiency")
        if any(key not in ds for key in needed):
            return {}
        table = pd.DataFrame({"interconnector_name": ds["links_interconnector_name"].values,
                              "foreign_market_pricing": ds["links_foreign_market_pricing"].values,
                              "efficiency": ds["links_efficiency"].values},
                             index=pd.Index(ds["links_i"].values).astype(str))
    return {cable: (link, float(table.at[link, "efficiency"])) for cable, link in _piecewise_links(table).items()}


def interconnector_rent_change(network_path, links_path, price_change, hours, cables):
    """Each cable's congestion-rent change when the GB price moves by ``price_change`` (item F12).

    The market summary's rent (``analyze_market``, signed cables) is
    sum(hours x (GB price x (delivered import - GB-side export) - trade cost)), and
    delivered import less GB-side export is efficiency x p0 in either direction. With
    the flows and the foreign side (the trade cost the LP charged) held as solved, the
    rent moves by efficiency x sum(hours x price change x p0). Returns ``(changes,
    held)``: the change per re-priced cable, and the cables that cannot be re-priced
    (no signed link on the network, or no Stage 1 schedule for it), whose rent stays
    at the model price.
    """
    cables = list(cables)
    signed = _signed_interconnector_links(network_path) if cables else {}
    flows = None
    if links_path is not None and signed:
        flows = pd.read_csv(links_path, index_col=0, parse_dates=True)
        if flows.index.has_duplicates:
            raise ValueError(f"{links_path} has duplicate timestamps")
        flows = flows.reindex(index=price_change.index)
    changes, held = {}, []
    for cable in cables:
        link = signed.get(cable)
        if link is None or flows is None or link[0] not in flows.columns:
            held.append(cable)
            continue
        p0 = flows[link[0]]
        if p0.isna().any():
            raise ValueError(f"{links_path} does not cover every snapshot for {link[0]} ({cable})")
        changes[cable] = link[1] * float((hours * price_change * p0).sum())
    return changes, held


def aggregate_inputs(scenarios, networks, prices, summaries, year, physical_networks=None,
                     strike_prices=None, domestic_profile=None, negative_price_rule="none",
                     annual_aggregate=None, reference_price_basis=None, economic_curtailment=None,
                     volume_caps=None, wholesale_links=None, contract_book=None,
                     price_level=DEFAULT_PRICE_LEVEL, price_level_mapping=None):
    """Settle the market year: hourly inputs, CfD payments and the annual quantities.

    ``price_level`` (item F12) is ``model`` (the model's own Stage 1 price, central) or
    ``cp32`` (the same hours quantile-mapped onto CP.32, the year ranked together; see
    :data:`PRICE_LEVELS`), with ``price_level_mapping`` the mapping settings block.
    """
    if price_level not in PRICE_LEVELS:
        raise ValueError(f"price_level must be one of {PRICE_LEVELS}, got {price_level!r}")
    if not scenarios or len(set(scenarios)) != len(scenarios) or not len(scenarios) == len(networks) == len(prices) == len(summaries):
        raise ValueError("Model input lists must be non-empty, unique and aligned")
    wholesale_links = list(wholesale_links) if wholesale_links is not None else [None] * len(scenarios)
    if len(wholesale_links) != len(scenarios):
        raise ValueError("Stage 1 link schedules must align with the source scenarios")
    if physical_networks is not None and (len(physical_networks) != len(scenarios) or strike_prices is None):
        raise ValueError("Physical networks and strike-price assumptions must align with source scenarios")
    curtailment = _economic_curtailment_config(economic_curtailment)
    if curtailment["rule"] != "none" and strike_prices is None:
        raise ValueError("The economic-curtailment rule needs the strike table to price curtailed output")

    # Pass 1: hourly inputs and market summaries for every month. The baseload
    # reference price is a season-level quantity, so the full calendar year has
    # to be in hand before any month's contracts are settled.
    frames, monthly = [], []
    summary_flex_mwh = []  # consumer flexibility in each month's market summary (F5)
    rent_by_cable, months_without_cables, orders = {}, [], {}
    for scenario, network, price, summary_path, links in zip(scenarios, networks, prices, summaries, wholesale_links):
        frame = _with_demand_split(read_hourly_inputs(network, price, links))
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        if summary.get("scenario") != scenario:
            raise ValueError(f"Market summary does not identify {scenario}")
        hours = float(frame.hours.sum())
        accounting = summary["accounting"]
        if not np.isclose(hours, float(accounting["reported_period_hours"]), rtol=0, atol=1e-6):
            raise ValueError(f"{scenario}: market summary and hourly model period disagree")
        # The summary counts every Load (the hydrogen offtake included) plus consumer
        # flexibility (consumer_flexibility_MWh, F5; absent in summaries written before it).
        # Flexibility is read here off the Stage 1 network and there off the balancing one,
        # which holds it at Stage 1, so the check compares the Loads, the same on both.
        demand = float((frame.total_load_mw * frame.hours).sum()) - float(frame.attrs.get("flexible_demand_mwh", 0.0))
        summary_loads = float(accounting["demand_MWh"]) - float(accounting.get("consumer_flexibility_MWh", 0.0))
        summary_flex_mwh.append(float(accounting.get("consumer_flexibility_MWh", 0.0)))
        if not np.isclose(demand, summary_loads, rtol=1e-5, atol=1):
            raise ValueError(f"{scenario}: wholesale and physical summary demand disagree")
        revenue = summary.get("revenue", {})
        if contract_book is not None and not revenue:
            # SETTLE2050: a 2050 run tracks no revenue (its network carries no labels); the
            # book is settled here, and nothing archived stands beside it.
            revenue = {"cfd_total_cost": 0.0, "cfd_total_dispatch_MWh": 0.0,
                       "cfd_reference_price_source": "wholesale_system_price",
                       "cfd_strike_source": "contract book (no archived revenue tracking)"}
        if revenue.get("cfd_reference_price_source") != "wholesale_system_price":
            raise ValueError(f"{scenario}: CfD payments need an explicit wholesale reference price")
        frame["source_scenario"] = scenario
        frames.append(frame)
        # Per-cable congestion rent from the Stage 1 schedule (analyze_market's
        # interconnector metrics), for the cap-and-floor pass-through line.
        cables = (summary.get("interconnectors") or {}).get("by_cable")
        month_rent = {}
        if cables is not None:
            for row in cables:
                rent = _finite(row["estimated_congestion_rent_GBP"], f"{scenario} congestion rent")
                name = str(row["interconnector"])
                rent_by_cable[name] = rent_by_cable.get(name, 0.0) + rent
                month_rent[name] = month_rent.get(name, 0.0) + rent
        else:
            months_without_cables.append(scenario)
        # Read whenever the networks are (the projection path always passes the
        # physical networks); an archived-summary aggregation does not open them.
        order = (wholesale_curtailment_order(network)
                 if physical_networks is not None or curtailment["rule"] != "none" else {"order": "not_read"})
        orders[scenario] = order["order"]
        if curtailment["rule"] != "none" and order["order"] == "contract_aware":
            raise ValueError(
                f"{scenario}: economic_curtailment_rule {curtailment['rule']!r} is on, but the wholesale "
                f"network was solved with the contract-aware curtailment order ({order['repriced_units']} "
                f"CfD units re-priced below zero, {order['pre_ar4_units_below_zero']} of them pre-AR4 at minus "
                "strike). The rule assumes pre-AR4 output was curtailed in a zero-price tie; under this "
                "order it is curtailed only at a price at or below -strike, where the generator would back "
                "off itself, and the rule would pay it about twice its strike. Set "
                "model_linked_rules.cfd.economic_curtailment_rule to 'none' for this run."
            )
        monthly.append({
            "scenario": scenario, "network": network, "price": price, "summary_path": summary_path,
            "frame": frame, "hours": hours, "demand": demand, "links": links, "rent_by_cable": month_rent,
            "archived_cfd": _finite(revenue["cfd_total_cost"], "CfD payment"),
            "archived_generation": _finite(revenue["cfd_total_dispatch_MWh"], "CfD generation"),
            "archived_strike_source": revenue.get("cfd_strike_source"),
            "redispatch": _finite(accounting["gross_redispatch_MWh"], "Redispatch"),
        })
    hourly = pd.concat(frames).sort_index()
    # Consumer flexibility: in the bill's volumes (Stage 1) and in the summaries' demand_MWh
    # (balancing); the annual check below compares the Loads alone, as the monthly one does.
    flexible_mwh = float(sum(f.attrs.get("flexible_demand_mwh", 0.0) for f in frames))
    summary_flexible_mwh = float(sum(summary_flex_mwh))
    if set(hourly.index.year) != {year}:
        raise ValueError("Model snapshots do not belong to the configured projection year")
    require_full_calendar_year(hourly.index, hourly.hours)

    # Item F12: under the cp32 price level the whole year is mapped here, before anything
    # is settled, so every use of the hourly price below (the baseload reference, pass 2's
    # CfD and curtailment payments, the system and domestic prices, the quarters' base and
    # peak, the price statistics) reads the mapped price. The interconnector rents, which
    # the market summaries carry at the model price, are moved by the price change.
    price_level_record = {"price_level": price_level}
    if price_level == "cp32":
        mapping = _apply_cp32_price_level(monthly, hourly, price_level_mapping)
        hourly = pd.concat(frames).sort_index()
        rent_model = dict(rent_by_cable)
        rent_change, rent_held = {}, set()
        for month in monthly:
            frame = month["frame"]
            changes, held = interconnector_rent_change(
                month["network"], month["links"], frame.price_gbp_per_mwh - frame.model_price_gbp_per_mwh,
                frame.hours, month["rent_by_cable"])
            for name, change in changes.items():
                rent_change[name] = rent_change.get(name, 0.0) + change
            rent_held.update(held)
        rent_by_cable = {name: rent + rent_change.get(name, 0.0) for name, rent in rent_model.items()}
        held_lines = dict(CP32_HELD_LINES)
        if rent_held:
            held_lines["interconnector_cap_and_floor_pass_through (part)"] = (
                f"cables {sorted(rent_held)} have no signed Stage 1 link schedule to re-price, so their rent "
                "(market summary) is at the model price in at least one month")
        if physical_networks is None:
            held_lines["cfd (archived)"] = ("no physical networks: the CfD payment is the market summaries' "
                                            "archived figure, at the model price")
        held_lines["sources[].archived_cfd_payment_gbp"] = (
            "the market summaries' CfD cross-check is at the model price, so recomputed_less_archived_cfd_gbp "
            "also carries the price-level change")
        price_level_record.update({
            "price_level_mapping": mapping,
            "price_level_repriced_lines": dict(CP32_REPRICED_LINES),
            "price_level_held_at_model_price": held_lines,
            "interconnector_congestion_rent_by_cable_model_price_gbp": rent_model,
            "interconnector_congestion_rent_change_by_cable_gbp": rent_change,
            "interconnector_rent_cables_held_at_model_price": sorted(rent_held),
        })

    # Baseload reference: season-level demand-weighted mean price for the
    # configured carriers; every other contract settles hourly.
    overrides, basis_report = _baseload_reference_overrides(reference_price_basis, hourly, physical_networks)

    # Pass 2: settle each month against its own slice of the annual series.
    evidence = []
    cfd_total, cfd_generation, redispatch = 0., 0., 0.
    physical_cfd_generation = 0.
    strike_rows, absent_contract_columns = [], set()
    low_price = {label: {"payment_gbp": 0.0, "dispatch_mwh": 0.0} for label in CFD_LOW_PRICE_BANDS}
    curtailment_payment, curtailment_mwh, curtailment_rows = 0., 0., []
    curtailment_static, curtailment_hours = set(), 0.
    for position, month in enumerate(monthly):
        frame = month["frame"]
        cfd, generation = month["archived_cfd"], month["archived_generation"]
        archived_cfd = cfd
        physical = None
        missing_dispatch, missing_capacity = 0, 0.
        month_curtailment = {"payment_gbp": 0.0, "curtailed_mwh": 0.0}
        if physical_networks is not None:
            month_overrides = (
                {carrier: series.loc[frame.index] for carrier, series in overrides.items()}
                if overrides else None
            )
            cfd, generation, physical, missing_dispatch, missing_capacity, strikes = read_cfd_payments(
                physical_networks[position], frame.price_gbp_per_mwh, strike_prices, negative_price_rule,
                month_overrides, volume_caps, contract_book=contract_book)
            physical_cfd_generation += physical
            strike_rows.extend(strikes["by_strike"])
            for label, band in (strikes.get("low_price_hours") or {}).items():
                low_price[label]["payment_gbp"] += band["payment_gbp"]
                low_price[label]["dispatch_mwh"] += band["dispatch_mwh"]
            absent_contract_columns.update(strikes["contract_metadata_absent"])
            if curtailment["rule"] == "pay_pre_ar4_at_strike":
                month_curtailment = read_economic_curtailment(
                    month["network"], frame.price_gbp_per_mwh, strike_prices,
                    curtailment["pre_ar4_rounds"], curtailment["zero_price_threshold_gbp_per_mwh"],
                    physical_network_path=physical_networks[position], negative_price_rule=negative_price_rule,
                    # F12: the LP curtailed in its own zero-price hours; the mapped price is paid against.
                    selection_price=frame.model_price_gbp_per_mwh if price_level == "cp32" else None)
                cfd += month_curtailment["payment_gbp"]
                curtailment_payment += month_curtailment["payment_gbp"]
                curtailment_mwh += month_curtailment["curtailed_mwh"]
                curtailment_rows.extend(month_curtailment["by_strike"])
                curtailment_static.update(month_curtailment["static_availability_generators"])
                curtailment_hours += month_curtailment["hours_at_or_below_threshold"]
        cfd_total += cfd
        cfd_generation += generation
        redispatch += month["redispatch"]
        evidence.append({"scenario": month["scenario"], "network": str(month["network"]), "price_csv": str(month["price"]),
                         "market_summary": str(month["summary_path"]), "hours": month["hours"], "demand_mwh": month["demand"],
                         "cfd_payment_gbp": cfd, "cfd_priced_generation_mwh": generation,
                         "cfd_physical_labelled_generation_mwh": physical,
                         "cfd_economic_curtailment_payment_gbp": month_curtailment["payment_gbp"],
                         "cfd_economic_curtailment_mwh": month_curtailment["curtailed_mwh"],
                         "cfd_assets_missing_dispatch": missing_dispatch,
                         "cfd_capacity_missing_dispatch_mw": missing_capacity,
                         "archived_cfd_payment_gbp": archived_cfd,
                         "archived_cfd_strike_source": month["archived_strike_source"],
                         "recomputed_less_archived_cfd_gbp": cfd - archived_cfd,
                         "physical_network": str(physical_networks[position]) if physical_networks is not None else None,
                         "price_sha256": hashlib.sha256(Path(month["price"]).read_bytes()).hexdigest(),
                         "network_modified_time_utc": pd.Timestamp(Path(month["network"]).stat().st_mtime, unit="s", tz="UTC").isoformat(),
                         "summary_sha256": hashlib.sha256(Path(month["summary_path"]).read_bytes()).hexdigest()})
    # Item D1: the system price is weighted by the metered-supply basis (GB
    # electrical load plus electrolyser draw), never by the hydrogen offtake.
    system_price = demand_weighted_price(hourly.price_gbp_per_mwh, hourly.demand_mw, hourly.hours)
    demand_mwh = float((hourly.total_load_mw * hourly.hours).sum())
    volumes = {name: float((hourly[column] * hourly.hours).sum()) for name, column in (
        ("gb_electrical_load_mwh", "gb_electrical_load_mw"), ("electrolyser_draw_mwh", "electrolyser_draw_mw"),
        ("h2_offtake_mwh", "h2_offtake_mw"), ("gsp_chargeable_volume_mwh", "demand_mw"))}

    domestic_price = None
    if domestic_profile is not None:
        # The household MWh is peak-heavy, so it is priced with a domestic shape
        # rather than with total system load. The difference is reported as an
        # explicit shaping row rather than folded into the energy line.
        weights = domestic_weights(hourly.index, domestic_profile)
        hourly["domestic_weight"] = weights
        domestic_price = demand_weighted_price(hourly.price_gbp_per_mwh, weights, hourly.hours)
    # Ofgem's cap prices energy per quarter as baseload and peak products (review
    # 28 Sep 2026); the quarters' base and peak prices and the household volume in
    # each are what the cap rule is formed from (:func:`ofgem_cap_rule_price`).
    base_peak_by_quarter = base_peak_prices_by_quarter(hourly) if domestic_profile is not None else None

    if annual_aggregate is not None and not np.isclose(
        demand_mwh - flexible_mwh, float(annual_aggregate["demand_MWh"]) - summary_flexible_mwh, rtol=1e-6, atol=1
    ):
        raise ValueError(
            f"Hourly model demand ({demand_mwh - flexible_mwh:,.0f} MWh, flexibility units excluded) "
            f"disagrees with the annual aggregate "
            f"({float(annual_aggregate['demand_MWh']) - summary_flexible_mwh:,.0f} MWh, flexibility units excluded)"
        )

    # Zero- and negative-price hours are the evidence for the negative-price
    # rule. On an as-built wholesale every CfD unit bids 0, so hours pile up just
    # above zero instead of crossing it and the rule can never bite; they are the
    # hours in which the economic-curtailment rule restores pre-AR4 payments.
    # Under the contract-aware order AR4+ units bid 0 and merchant renewables
    # +0.5 (decision D1, 27 Sep 2026); older contracts bid minus their strike, so
    # hours still clear negative when those are marginal and the rule applies
    # there. The earlier AR4+ = -1 treatment is kept as a sensitivity
    # (ar4_plus_bid_gbp_per_mwh: -1.0).
    price = hourly.price_gbp_per_mwh
    if price_level == "cp32":
        # F12: the statistics below are on the price the bill settles on (the mapped
        # one, where the negative-price rule now applies); the model's own are kept here.
        model = hourly.model_price_gbp_per_mwh
        price_level_record["model_price_statistics"] = {
            "wholesale_price_gbp_per_mwh": demand_weighted_price(model, hourly.demand_mw, hourly.hours),
            "domestic_profile_price_gbp_per_mwh": (
                None if domestic_profile is None
                else demand_weighted_price(model, hourly.domestic_weight, hourly.hours)),
            "hours_price_at_or_below_1_gbp_per_mwh": float(hourly.hours[model.le(1.0)].sum()),
            "hours_price_below_minus_0_01_gbp_per_mwh": float(
                hourly.hours[model.lt(-NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH)].sum()),
            "minimum_price_gbp_per_mwh": float(model.min()),
        }
    distinct_orders = sorted(set(orders.values()))
    wholesale_order = distinct_orders[0] if len(distinct_orders) == 1 else "mixed"
    annual = {"modelled_year": year, "reported_period_hours": float(hourly.hours.sum()),
              # Every Load, as the market summary counts it (checks only; it
              # includes the hydrogen offtake). The billing volumes are below.
              "demand_mwh": demand_mwh,
              "demand_volumes_mwh": volumes,
              "hours_price_at_or_below_1_gbp_per_mwh": float(hourly.hours[price.le(1.0)].sum()),
              "hours_price_at_or_below_zero": float(hourly.hours[price.le(0.0)].sum()),
              # Below solver noise (NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH): the hours that genuinely
              # clear negative. The two keys are the same count; the second names its tolerance.
              "hours_price_below_zero": float(hourly.hours[price.lt(-NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH)].sum()),
              "hours_price_below_minus_0_01_gbp_per_mwh": float(
                  hourly.hours[price.lt(-NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH)].sum()),
              "minimum_price_gbp_per_mwh": float(price.min()),
              "wholesale_curtailment_order": wholesale_order,
              "wholesale_curtailment_order_by_scenario": orders,
              "interconnector_congestion_rent_by_cable_gbp": rent_by_cable,
              "interconnector_rent_months_without_cables": months_without_cables,
              "wholesale_price_gbp_per_mwh": system_price,
              "domestic_profile_price_gbp_per_mwh": domestic_price,
              "domestic_shaping_gbp_per_mwh": None if domestic_price is None else domestic_price - system_price,
              "base_peak_prices_by_quarter": base_peak_by_quarter,
              "domestic_profile": domestic_profile,
              "cfd_payment_gbp": cfd_total, "cfd_priced_generation_mwh": cfd_generation,
              "cfd_settled_dispatch_payment_gbp": cfd_total - curtailment_payment,
              "cfd_physical_labelled_generation_mwh": physical_cfd_generation if physical_networks is not None else None,
              "cfd_dispatch_coverage": cfd_generation / physical_cfd_generation if physical_cfd_generation > 0 else None,
              "cfd_payment_basis": ("recomputed_physical_dispatch" if physical_networks is not None else "archived_market_summary")
                                   + (" on the 2050 contract book" if contract_book is not None else ""),
              **({"cfd_contract_book": str(contract_book)} if contract_book is not None else {}),
              "cfd_negative_price_rule": negative_price_rule,
              "cfd_reference_price_basis": basis_report,
              "cfd_economic_curtailment_rule": curtailment["rule"],
              "cfd_economic_curtailment_rounds": list(curtailment["pre_ar4_rounds"]),
              "cfd_economic_curtailment_zero_price_threshold_gbp_per_mwh": curtailment["zero_price_threshold_gbp_per_mwh"],
              "cfd_economic_curtailment_payment_gbp": curtailment_payment,
              "cfd_economic_curtailment_mwh": curtailment_mwh,
              "cfd_economic_curtailment_hours_at_or_below_threshold": curtailment_hours,
              "cfd_economic_curtailment_static_availability_generators": sorted(curtailment_static),
              "cfd_economic_curtailment_by_strike": _combine_curtailment_rows(curtailment_rows),
              "cfd_missing_dispatch_asset_periods": sum(row["cfd_assets_missing_dispatch"] for row in evidence),
              "cfd_max_missing_dispatch_capacity_mw": max(row["cfd_capacity_missing_dispatch_mw"] for row in evidence),
              "gross_redispatch_mwh": redispatch, "cfd_strike_provenance": _combine_strike_rows(strike_rows, absent_contract_columns),
              "cfd_payments_in_low_price_hours": (
                  cfd_low_price_record(low_price, hourly, cfd_total - curtailment_payment, curtailment_payment,
                                       curtailment["zero_price_threshold_gbp_per_mwh"])
                  if physical_networks is not None else None),
              "annual_aggregate": annual_aggregate,
              **price_level_record,
              "sources": evidence}
    return hourly, annual


def _with_demand_split(frame):
    """Fill the item-D1 demand split on a frame that carries only ``demand_mw`` (all electrical, no links)."""
    frame = frame.copy()
    if "total_load_mw" not in frame.columns:
        frame["total_load_mw"] = frame["demand_mw"]
    if "gb_electrical_load_mw" not in frame.columns:
        frame["gb_electrical_load_mw"] = frame["demand_mw"]
    for column in ("electrolyser_draw_mw", "h2_offtake_mw"):
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _economic_curtailment_config(block):
    """Validate the economic-curtailment rule block and fill its defaults."""
    block = dict(block or {})
    rule = str(block.get("rule", "none"))
    if rule not in ECONOMIC_CURTAILMENT_RULES:
        raise ValueError(f"economic_curtailment_rule must be one of {ECONOMIC_CURTAILMENT_RULES}, got {rule!r}")
    rounds = block.get("pre_ar4_rounds")
    rounds = list(DEFAULT_PRE_AR4_ROUNDS) if rounds is None else [str(r) for r in rounds]
    if rule != "none" and not rounds:
        raise ValueError("pay_pre_ar4_at_strike needs a non-empty pre_ar4_rounds list")
    threshold = block.get("zero_price_threshold_gbp_per_mwh", DEFAULT_ZERO_PRICE_THRESHOLD)
    threshold = _finite(threshold, "zero_price_threshold_gbp_per_mwh")
    if threshold < 0:
        raise ValueError("zero_price_threshold_gbp_per_mwh must be non-negative")
    return {"rule": rule, "pre_ar4_rounds": rounds, "zero_price_threshold_gbp_per_mwh": threshold}


def _baseload_reference_overrides(reference_price_basis, hourly, physical_networks):
    """Build ``{carrier: Series}`` overrides for baseload CfDs and describe them."""
    block = dict(reference_price_basis or {})
    carriers = [str(c) for c in (block.get("baseload_carriers") or [])]
    if not carriers:
        return None, {"baseload_carriers": [], "baseload_reference": None,
                      "other_carriers": "hourly_wholesale",
                      "note": "Every CfD contract settled against the hourly wholesale reference price."}
    basis = str(block.get("baseload_reference", ""))
    if basis not in BASELOAD_REFERENCE_BASES:
        raise ValueError(f"baseload_reference must be one of {BASELOAD_REFERENCE_BASES}, got {basis!r}")
    if physical_networks is None:
        raise ValueError("A baseload reference price can only be applied when CfD payments are recomputed")
    series, seasons = seasonal_demand_weighted_reference(hourly)
    return ({carrier: series for carrier in carriers},
            {"baseload_carriers": carriers, "baseload_reference": basis, "seasons": seasons,
             "other_carriers": "hourly_wholesale",
             "note": ("Contracts on the listed carriers settle against the demand-weighted mean "
                      "wholesale price of the season (Apr-Sep summer, Oct-Mar winter) the snapshot "
                      "falls in, as a proxy for the season-ahead baseload reference in a baseload "
                      "CfD; the model has no forward curve. All other contracts settle hourly.")})


def cfd_low_price_record(bands, hourly, settled_payment_gbp, curtailment_payment_gbp, curtailment_threshold):
    """The share of the CfD line paid in low-price hours (F18, 27 Oct 2026; a report, no bill change).

    ``bands`` are the settled difference payments and dispatch summed over the months per
    :data:`CFD_LOW_PRICE_BANDS` (physical dispatch at the hourly price it settles against); the
    pre-AR4 economic-curtailment payment is paid only in hours at or below its own threshold, so it
    is reported beside them as a zero-price-hour payment too.
    """
    price, hours = hourly.price_gbp_per_mwh, hourly.hours
    line = float(settled_payment_gbp) + float(curtailment_payment_gbp)
    out = {"settled_payment_gbp": float(settled_payment_gbp),
           "economic_curtailment_payment_gbp": float(curtailment_payment_gbp),
           "economic_curtailment_threshold_gbp_per_mwh": curtailment_threshold,
           "cfd_line_gbp": line, "bands": {}}
    for label, threshold in CFD_LOW_PRICE_BANDS.items():
        paid = float(bands[label]["payment_gbp"])
        out["bands"][label] = {
            "threshold_gbp_per_mwh": float(threshold),
            "hours": float(hours[price.le(threshold)].sum()),
            "settled_payment_gbp": paid,
            "settled_dispatch_mwh": float(bands[label]["dispatch_mwh"]),
            "share_of_settled_payment": paid / float(settled_payment_gbp) if settled_payment_gbp else None,
            "with_economic_curtailment_gbp": paid + float(curtailment_payment_gbp),
            "share_of_cfd_line": (paid + float(curtailment_payment_gbp)) / line if line else None,
        }
    out["note"] = ("difference payments on dispatch in hours at or below each price (the hourly wholesale price "
                   "the contracts settle against; baseload contracts on a seasonal reference are counted by the "
                   "hour's wholesale price), plus the pre-AR4 economic-curtailment payment, which is paid only in "
                   "zero-price hours; reported, not a bill line")
    return out


def _with_household(record, rate):
    """:func:`cfd_low_price_record` with each band's CfD-line amounts per household (the CfD line's rate)."""
    if not record or rate is None:
        return record
    out = copy.deepcopy(record)
    out["cfd_line_gbp_per_3100kwh_ex_vat"] = record["cfd_line_gbp"] * rate
    for band in out["bands"].values():
        band["with_economic_curtailment_gbp_per_3100kwh_ex_vat"] = band["with_economic_curtailment_gbp"] * rate
        band["settled_payment_gbp_per_3100kwh_ex_vat"] = band["settled_payment_gbp"] * rate
    return out


def _combine_strike_rows(strike_rows, absent_contract_columns):
    """Roll the per-scenario strike breakdown up to one annual view."""
    if not strike_rows:
        return {"contract_metadata_absent": sorted(absent_contract_columns), "by_strike": []}
    frame = pd.DataFrame(strike_rows)
    if "is_fallback" not in frame.columns:
        frame["is_fallback"] = frame.strike_key.eq(f"{MODELLED_ROUND}_(carrier fallback)")
    if "reference_price_basis" not in frame.columns:
        frame["reference_price_basis"] = "hourly_wholesale"
    grouped = frame.groupby(["carrier", "strike_key", "strike_price", "is_fallback", "reference_price_basis"],
                            dropna=False).agg(
        generators=("generators", "max"),
        dispatch_mwh=("dispatch_mwh", "sum"),
        payment_gbp=("payment_gbp", "sum"),
    ).reset_index().sort_values("payment_gbp", ascending=False)
    energy = float(grouped.dispatch_mwh.sum())
    gross = float(grouped.payment_gbp.abs().sum())
    fallback = grouped.loc[grouped.is_fallback.astype(bool)]
    modelled = grouped.loc[grouped.strike_key.astype(str).str.startswith(MODELLED_ROUND + "_")]
    return {
        "contract_metadata_absent": sorted(absent_contract_columns),
        "by_strike": grouped.to_dict("records"),
        "fallback_strike_share_of_energy": float(fallback.dispatch_mwh.sum() / energy) if energy else None,
        "fallback_strike_share_of_payment": float(fallback.payment_gbp.abs().sum() / gross) if gross else None,
        "modelled_round": MODELLED_ROUND,
        "modelled_round_share_of_energy": float(modelled.dispatch_mwh.sum() / energy) if energy else None,
        "modelled_round_share_of_payment": float(modelled.payment_gbp.abs().sum() / gross) if gross else None,
    }


def _strike_limitation(provenance):
    """State how much of the CfD line rests on a carrier-wide fallback strike."""
    if not provenance:
        return "CfD strike provenance was not recorded for this run."
    share = provenance.get("fallback_strike_share_of_payment")
    absent = provenance.get("contract_metadata_absent") or []
    modelled = provenance.get("modelled_round_share_of_payment")
    ar8 = (f" {modelled:.0%} of the gross CfD payment settled on the modelled AR8 round (capacity beyond the "
           "published rounds, at a levelised strike calibrated to the last auctions)." if modelled else "")
    if not share:
        return "Every CfD contract settled against a round-specific strike price." + ar8
    detail = (
        f" The saved generators carry no {' or '.join(absent)}, so no round-specific "
        "strike could be selected." if absent else ""
    )
    return (
        f"{share:.0%} of the gross CfD payment settled against a carrier-wide "
        f"fallback strike (the modelled AR8 round's) rather than the contract's auctioned round.{detail} "
        "Separately auctioned farms therefore share one strike price. Published round strikes are "
        "clearing prices indexed from 2012 prices with CPI; the AR8 strike is modelled "
        "(scripts/market/build_ar8_modelled_strikes.py)." + ar8
    )


def _max_fallback_check(provenance, case):
    """Fail a case whose contract labelling is worse than it declares it tolerates."""
    limit = case.get("max_fallback_strike_share_of_payment")
    if limit is None:
        return
    share = (provenance or {}).get("fallback_strike_share_of_payment")
    if share is not None and float(share) > float(limit):
        raise ValueError(
            f"{share:.1%} of the gross CfD payment used a carrier-wide fallback strike, above the "
            f"configured limit of {float(limit):.1%}. Label the CP30 CfD capacity with allocation "
            "rounds (WS6) or raise the limit deliberately."
        )


#: The supplier line, which carries Ofgem's levelisation and margin percentages (F18).
SUPPLIER_LINE = "supplier_costs_and_margin"
#: The recovery of a line quoted per household with no levy base (F18).
PER_HOUSEHOLD_RECOVERY_BASIS = "per-household line; system equivalent at metered supply (declared)"


def supplier_line_parts(key, block, external_factor, value, basis):
    """F18 (user, 6 Oct 2026): the supplier line before EBIT and headroom, plus levelisation.

    Returns ``(value, basis, margin)``. ``derivation.direct_debit_levelisation`` adds
    Ofgem's levelisation for a direct-debit household (Annex 9), a per-customer amount
    on the line's own price basis, held flat in real terms. ``derivation.ebit_and_headroom``
    replaces the cap period's EBIT and headroom cash with Ofgem's percentages of the
    bill's own costs (:func:`apply_supplier_margin`): the line here holds the other
    components. With neither declared the line is its cash value and ``margin`` is None.
    """
    derivation = block.get("derivation") or {}
    levelisation_block = derivation.get("direct_debit_levelisation")
    rule = derivation.get("ebit_and_headroom")
    if levelisation_block is None and rule is None:
        return value, basis, None
    margin = {"rule": rule, "external_factor": float(external_factor)}
    if rule is not None:
        components = derivation[f"components_gbp_per_3100kwh_{int(block['price_year'])}"]
        if not {"ebit", "headroom"} <= set(components):
            raise ValueError(f"{key}: ebit_and_headroom needs the cap period's ebit and headroom components")
        if not np.isclose(sum(float(v) for v in components.values()), float(block["value"]), rtol=0, atol=5e-4):
            raise ValueError(f"{key}: the components do not add up to the line's value")
        ebit, headroom = float(components["ebit"]), float(components["headroom"])
        value = (float(block["value"]) - ebit - headroom) * external_factor
        margin.update(cap_period_ebit_gbp=ebit * external_factor, cap_period_headroom_gbp=headroom * external_factor)
        basis += (f" EBIT and headroom are recomputed at Ofgem's percentages of this bill's costs (item F18), in place "
                  f"of their cap-period cash (GBP{ebit:.2f} and GBP{headroom:.2f}, {block['price_year']} prices).")
    levelisation = 0.0
    if levelisation_block is not None:
        if int(levelisation_block["price_year"]) != int(block["price_year"]):
            raise ValueError(f"{key}: direct_debit_levelisation must be quoted on the line's price year")
        levelisation = _finite(levelisation_block["gbp_per_customer"], f"{key} levelisation", True) * external_factor
        value += levelisation
        basis += (f" Plus Ofgem's direct-debit levelisation, GBP{float(levelisation_block['gbp_per_customer']):.2f} "
                  f"per customer ({block['price_year']} prices), held flat in real terms (item F18).")
    margin["levelisation_gbp"] = levelisation
    return value, basis, margin


def apply_supplier_margin(rows, margin, base=None):
    """F18: add Ofgem's EBIT and headroom, as percentages of the bill's costs, to the supplier row.

    EBIT = variable share x costs + the fixed amount per customer; headroom = its share x
    (costs excluding network costs + EBIT), as Ofgem's cap model builds them
    (ElecSingle_Other_Benchmark: costs are every allowance before EBIT and headroom;
    levelisation sits outside). ``base`` (``costs_gbp``, ``costs_excluding_networks_gbp``,
    GBP per household in report prices) is the bill's own lines unless a caller supplies
    one. Returns the margin record.
    """
    rule = margin.get("rule")
    if rule is None:
        return margin
    row = next(r for r in rows if r["component_id"] == SUPPLIER_LINE)
    if base is None:
        network = rule["network_group"]
        groups = bill_group_of(network["view"])
        costs = sum(float(r["annual_gbp_per_3100kwh_ex_vat"]) for r in rows) - float(margin["levelisation_gbp"])
        networks = sum(float(r["annual_gbp_per_3100kwh_ex_vat"]) for r in rows
                       if groups.get(r["component_id"]) == network["group"])
        base = {"costs_gbp": costs, "costs_excluding_networks_gbp": costs - networks,
                "source": f"this bill's lines ex VAT, less levelisation; networks = the {network['view']!r} view's "
                          f"{network['group']!r} group"}
    else:
        base = {"costs_gbp": _finite(base["costs_gbp"], "supplier margin cost base"),
                "costs_excluding_networks_gbp": _finite(base["costs_excluding_networks_gbp"],
                                                        "supplier margin cost base excluding networks"),
                "source": str(base.get("source", "supplied by the caller"))}
    fixed = _finite(rule["ebit_fixed_gbp_per_customer"], "EBIT fixed", True) * float(margin["external_factor"])
    ebit = _finite(rule["ebit_variable_share"], "EBIT share", True) * base["costs_gbp"] + fixed
    headroom = _finite(rule["headroom_share"], "headroom share", True) * (base["costs_excluding_networks_gbp"] + ebit)
    row["annual_gbp_per_3100kwh_ex_vat"] = float(row["annual_gbp_per_3100kwh_ex_vat"]) + ebit + headroom
    row["method"] += (f" EBIT {float(rule['ebit_variable_share']):.4%} x GBP{base['costs_gbp']:,.2f} + GBP{fixed:.2f} = "
                      f"GBP{ebit:.2f}; headroom {float(rule['headroom_share']):.4%} x (GBP"
                      f"{base['costs_excluding_networks_gbp']:,.2f} excluding networks + EBIT) = GBP{headroom:.2f}.")
    return {**margin, "base": base, "ebit_fixed_gbp": fixed, "ebit_gbp": ebit, "headroom_gbp": headroom,
            "change_against_cap_period_cash_gbp": ebit + headroom - float(margin["cap_period_ebit_gbp"])
            - float(margin["cap_period_headroom_gbp"])}


def build_line_recovery(rows, specs, volumes, factor, model_price_year):
    """F18: ``component_id -> entry`` for every row (:mod:`scripts.consumer_bill.line_recovery`).

    ``specs`` holds ``(volume key, household loss factor, household allocation factor, basis)``
    for the lines the projector builds; a line a caller adds copies the entry of the line it
    adjusts (``INHERITED_LINE_RECOVERY``). Every entry is in the model price year.
    """
    entries = {}
    for row in rows:
        component_id = row["component_id"]
        if component_id in specs:
            key, loss, allocation, basis = specs[component_id]
            entries[component_id] = make_line_recovery_entry(
                key, volumes.get(key), household_loss_factor=loss, household_allocation_factor=allocation,
                price_year_factor=factor, system_price_year=model_price_year, basis=basis)
    for row in rows:
        if row["component_id"] not in entries:
            entries[row["component_id"]] = inherited_line_recovery(entries, row["component_id"])
    check_line_recovery_volumes(entries, volumes)
    return entries


def project_components(annual, assumptions, case, balancing=None, recovery=None, ro=None,
                       extra_components=None, ldes=None, dpa=None, supplier_margin_base=None):
    """Assemble the bill's lines.

    ``ldes`` is ``{"topup_gbp": system GBP (model price year), "source": ..., ...}``,
    the LDES floor top-up the bill carries (item D4(c)); required when
    ``model_linked_rules.ldes_cap_and_floor`` is enabled.

    ``dpa`` (task SETTLE2050) is ``contract_book_2050.read_dpa_line``: the 2050 support
    settlement's net DPA payment (system GBP, model prices), required when
    ``model_linked_rules.dpa`` is declared.

    ``supplier_margin_base`` (F18): the cost base, ``{"costs_gbp", "costs_excluding_networks_gbp"}``
    (GBP per household, report prices), that Ofgem's EBIT and headroom percentages are applied to
    when the supplier line declares them. By default the bill's own lines; the ownership projector
    passes the private case's, so the supplier line is the same in every ownership case.
    """
    year = int(case["modelled_year"])
    expected_hours = 8784 if calendar.isleap(year) else 8760
    if annual["modelled_year"] != year or annual["reported_period_hours"] != expected_hours:
        raise ValueError("Projection requires the correct complete calendar year")
    benchmark = assumptions["benchmark"]
    consumption = _finite(benchmark["annual_consumption_kwh"], "Consumption", True)
    if consumption != 3100:
        raise ValueError("External allowances use a 3,100 kWh benchmark")
    # The assumptions file's own note wins: it states the per-line period bases the
    # projector enforces, which the case note in config/defaults/base/consumer_bill.yaml predates.
    period_note = benchmark.get("period_alignment_note") or case.get("period_alignment_note")
    if not period_note:
        raise ValueError("Specify how the benchmark's period maps to the modelled calendar year")
    # Every row is stamped with the benchmark price basis, so each input has to
    # be carried to that basis explicitly rather than summed as published.
    report_price_year = int(benchmark["price_year"])
    if str(report_price_year) not in str(benchmark["price_basis"]):
        raise ValueError("benchmark.price_year must agree with benchmark.price_basis")
    model_price_year = int(case["model_price_year"])
    rebasing = assumptions["currency_rebasing"]
    if model_price_year != int(rebasing["from_price_year"]):
        raise ValueError("Model price basis does not match the configured rebasing source year")
    if int(rebasing["to_price_year"]) != report_price_year:
        raise ValueError("Currency rebasing does not land on the benchmark price year")
    factor = _finite(rebasing["factor"], "Currency factor", True)
    if not np.isclose(factor, float(rebasing["to_index"]) / float(rebasing["from_index"]), rtol=1e-9):
        raise ValueError("Currency factor does not reconcile to its source indices")
    source_factor = _source_rebasing_factor(assumptions, report_price_year)
    rules = set(assumptions["model_linked_rules"]) - OPTIONAL_MODEL_RULES
    if rules != APPLIED_MODEL_RULES | DOCUMENTED_MODEL_RULES:
        raise ValueError(
            "model_linked_rules schema mismatch: "
            f"{sorted(rules ^ (APPLIED_MODEL_RULES | DOCUMENTED_MODEL_RULES))}. "
            "Every declared rule must be applied by the projector or listed as documentation."
        )
    vat_rate = _finite(benchmark["vat_rate"], "VAT")
    if vat_rate < 0:
        raise ValueError("VAT must be non-negative")
    mwh = consumption / 1000

    if balancing is None:
        raise ValueError(
            "The BSUoS constraint row needs the balancing-cost extraction; the flat external "
            "bsuos_reference allowance has been withdrawn and must not be reinstated implicitly."
        )
    if recovery is None:
        raise ValueError("Every recovery denominator must come from the sourced recovery-demand file")
    if ro is None:
        raise ValueError(
            "The Renewables Obligation row needs the RO scheme-cost extraction. The external "
            "allowance it replaced assumed the domestic Exchequer relief continues, so it must "
            "not be reinstated implicitly."
        )
    # Who is exempt from the levies (review 28 Sep 2026): a switch, defaulting to
    # decision D1 (electrolysers pay) and the 2024-25 EII exemption share.
    ro_volume_before = recovery.get("ro_eligible_metered_supply_mwh")
    recovery, levy_exemptions = apply_levy_exemptions(recovery, assumptions.get("levy_exemptions"))
    two_volume = recovery.get("recovery_basis") == TWO_VOLUME_BASIS
    if two_volume and levy_exemptions.get("applied") and not np.isclose(
            float(levy_exemptions["capacity_market_volume_after_mwh"]),
            float(levy_exemptions["capacity_market_volume_before_mwh"]), rtol=0, atol=1e-6):
        raise ValueError(
            "levy_exemptions exempts volume from the Capacity Market centrally, which the two-volume recovery basis "
            "does not implement (the GSP charging volume also carries BSUoS and LDES); declare it as a named "
            "sensitivity (levy_exemptions.sensitivities) instead")
    ro = scale_ro_for_exemptions(ro, levy_exemptions)
    # Metered supply: the household's share of a per-household system cost and the
    # Exchequer's per-household key (``final_demand_mwh`` is its deprecated alias).
    final_demand = _finite(recovery["metered_supply_mwh"] if two_volume else recovery["final_demand_mwh"],
                           "Metered supply", True)
    procurement_loss = float(assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]
                             ["allowances"]["losses"]["loss_multiplier_gb_average"])
    if two_volume:
        # F18 (6 Oct 2026): the household loss factor is charged on every GSP-rate line and
        # on procurement losses; the two must be one factor.
        loss_factor = _finite(recovery["household_loss_factor"], "Household loss factor", True)
        if not np.isclose(loss_factor, procurement_loss, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"The recovery file's household loss factor ({loss_factor}) differs from the procurement losses "
                f"multiplier ({procurement_loss}); the bill must charge one household loss factor")
        # A GSP-rate line: cost / GSP charging volume x 3.1 MWh x household loss factor.
        gsp_rate_denominator = _finite(recovery["gsp_charging_volume_mwh"], "GSP charging volume", True) / loss_factor
        recovery_basis_text = (
            f"Recovered on the GSP charging volume {float(recovery['gsp_charging_volume_mwh']) / 1e6:,.2f} TWh "
            f"(GB load / {float(recovery['system_to_metered_factor']):.3f} x "
            f"{float(recovery['metered_to_gsp_factor']):.3f} + electrolyser draw) and charged on the household's "
            f"3.1 MWh x the household loss factor {loss_factor:.4f}, a GSP rate (item F18)")
        ro_key = "ro_eligible_metered_supply_mwh"
        declared = ro.get("obligation_eligible_supply_mwh")
        if declared is not None and not np.isclose(float(declared), float(ro_volume_before), rtol=1e-9, atol=1.0):
            raise ValueError(
                f"The RO scheme summary was spread over {float(declared):,.0f} MWh, not the recovery file's RO-eligible "
                f"metered supply {float(ro_volume_before):,.0f} MWh; rebuild it (build_future_ro_scheme_cost)")
    else:
        # Item D1: one loss basis. The GSP-to-meter factor the denominators are
        # built on must be the multiplier the procurement losses line applies, or the
        # household's metered MWh would be grossed up two ways in one bill.
        loss_factor = float(recovery.get("gsp_to_meter_loss_factor") or 1.0)
        if recovery.get("gsp_to_meter_loss_factor") is not None and not np.isclose(
                loss_factor, procurement_loss, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"The recovery file's GSP-to-meter loss factor ({loss_factor}) differs from the procurement losses "
                f"multiplier ({procurement_loss}); the bill must carry one loss basis"
            )
        gsp_rate_denominator = final_demand
        recovery_basis_text = (
            f"Recovered on metered supply {final_demand / 1e6:,.2f} TWh = (GB electrical load + electrolyser draw) "
            f"{float(recovery['gsp_chargeable_volume_mwh']) / 1e6:,.2f} TWh at the GSP / loss factor {loss_factor:.4f}, "
            "the same as the GSP rate on the household's loss-uplifted 3.1 MWh (item D1)"
            if recovery.get("gsp_chargeable_volume_mwh") is not None else "Recovered over final demand")
        ro_key = "cfd_eligible_supplier_sales_mwh" if recovery.get("cfd_eligible_supplier_sales_mwh") else "metered_supply_mwh"
    # F18: each line's recovery, ``component_id -> (volume key, household loss factor,
    # household allocation factor, basis)``; the summary's ``line_recovery`` is built from it.
    metered_key = "metered_supply_mwh"
    gsp_rate_recovery = (("gsp_charging_volume_mwh", loss_factor) if two_volume else (metered_key, 1.0))
    line_recovery_of = {}

    recovery_config = case["cfd_recovery"]
    method = recovery_config["method"]
    sourced_eligible_sales = recovery.get("cfd_eligible_gsp_volume_mwh" if two_volume
                                          else "cfd_eligible_supplier_sales_mwh")
    if method == "modelled_system_demand_proxy":
        # Once the case's recovery-demand file supplies a sourced eligible-sales
        # figure, the proxy is no longer an honest fallback but a choice to
        # divide by a volume the levy is not charged on, which understates the
        # per-MWh rate. Refuse rather than silently under-recover.
        if sourced_eligible_sales:
            raise ValueError(
                "This case has a sourced CfD eligible-supply denominator of "
                f"{float(sourced_eligible_sales):,.0f} MWh, so modelled_system_demand_proxy is "
                "refused: system demand includes volume the CfD levy is not charged on. Set "
                "cfd_recovery.method to eligible_supplier_sales."
            )
        denominator = _finite(annual["demand_mwh"], "System demand", True)
        recovery_note = "Model demand proxy; not eligible supplier sales or a statutory levy forecast"
        cfd_recovery = ("modelled_system_demand_mwh", 1.0)
    elif method == "eligible_supplier_sales" and two_volume:
        # F18: the CfD levy is a GSP rate on eligible supply at the GSP, charged on the
        # household's loss-uplifted 3.1 MWh. ``denominator`` is the volume over the household
        # loss factor, so cost / denominator x 3.1 MWh is the household line.
        cfd_volume = _finite(recovery["cfd_eligible_gsp_volume_mwh"], "CfD eligible GSP volume", True)
        denominator = cfd_volume / loss_factor
        recovery_note = (
            f"CfD eligible supply at the GSP, {recovery['cfd_eligible_share']:.4f} of the GSP charging volume after "
            f"the EII exemption ({cfd_volume / 1e6:,.2f} TWh), charged on the household's 3.1 MWh x the household loss "
            f"factor {loss_factor:.4f} (item F18); contract coverage and reconciliation remain to be validated"
        )
        cfd_recovery = ("cfd_eligible_gsp_volume_mwh", loss_factor)
    elif method == "eligible_supplier_sales":
        denominator = _finite(recovery["cfd_eligible_supplier_sales_mwh"], "Eligible supplier sales", True)
        recovery_note = (
            f"CfD eligible supplier sales, {recovery['cfd_eligible_share']:.4f} of metered supply "
            "after the EII exemption (the GSP-rate levy on the household's loss-uplifted 3.1 MWh, item D1); "
            "contract coverage and reconciliation remain to be validated"
        )
        cfd_recovery = ("cfd_eligible_supplier_sales_mwh", 1.0)
    else:
        raise ValueError("CfD recovery method must be modelled_system_demand_proxy or eligible_supplier_sales")

    _max_fallback_check(annual.get("cfd_strike_provenance"), case)

    rows = []
    group_names = {source: name for name, sources, _ in GROUPS for source in sources}
    def add(key, group, label, value, origin, note, source, reference=None):
        rows.append({"scenario": case["case_id"], "modelled_year": year, "component_id": key,
                     "stack_group": group, "category": group_names[group], "component": label,
                     "annual_gbp_per_3100kwh_ex_vat": _finite(value, label), "source_class": origin,
                     "method": note, "source": source, "currency": benchmark["price_basis"],
                     "reference_gbp_per_3100kwh_ex_vat": (
                         _finite(reference, f"{label} reference") if reference is not None else np.nan)})

    system_price = _finite(annual["wholesale_price_gbp_per_mwh"], "Wholesale price")
    bridge = assumptions["model_linked_rules"]["direct_fuel"]
    domestic_price = annual.get("domestic_profile_price_gbp_per_mwh")
    if domestic_price is None:
        raise ValueError(
            "The wholesale line needs a domestic-profile-weighted price; a system-weighted price "
            "would charge the household a system-shaped MWh."
        )
    domestic_price = _finite(domestic_price, "Domestic profile price")
    shaping = shaping_settings(bridge)
    if shaping["method"] == "ofgem_cap_rule":
        # Review 28 Sep 2026: Ofgem's cap rule. The cap buys the household's energy
        # as a 70/30 baseload/peak blend per quarter and adds shaping allowances
        # (seasonal to monthly, monthly peak/baseload to hourly) for reshaping those
        # blocks into the domestic half-hourly profile. The model's own domestic
        # profile is Ofgem's two-block 70/30 allocation, so it cannot see that
        # hourly step; the allowance is applied instead. Unchanged in Annex 2 from
        # 2019 to Q4 2026 (v1.30), set in 2018 from historical day-ahead price ratios.
        by_quarter = annual.get("base_peak_prices_by_quarter")
        if not by_quarter:
            raise ValueError("The Ofgem cap-rule shaping needs base_peak_prices_by_quarter from the hourly inputs")
        declared_weights = (shaping.get("quarter_weights") or {}).get("weights")
        blend, quarters = ofgem_cap_rule_price(by_quarter, shaping["base_share"], shaping["peak_share"],
                                               quarter_weights=declared_weights)
        rate = shaping["seasonal_to_monthly_rate"] + shaping["monthly_to_hourly_rate"]
        add("direct_energy", "hedged_wholesale", "PyPSA wholesale energy (Ofgem 70/30 base/peak blend)",
            blend * factor * mwh, "pypsa_hourly_dispatch_prices",
            f"Per quarter {shaping['base_share']:.0%} baseload + {shaping['peak_share']:.0%} Elexon peak-block "
            f"price of the model's hourly prices, the quarters weighted by "
            f"{'Ofgem PC1 demand (Annex 2, 3b Demand)' if declared_weights else 'household volume'}: "
            f"GBP{blend:,.4f}/MWh, then currency rebasing (Ofgem cap rule, Annex 2)",
            "Declared wholesale price and network files")
        add("shaping_adjustment", "hedge_uplifts", "Shaping (Ofgem cap allowances)",
            blend * rate * factor * mwh, "external_intensity_on_pypsa_energy",
            f"Blend x (seasonal to monthly {shaping['seasonal_to_monthly_rate']:.6%} + monthly peak/baseload to "
            f"hourly {shaping['monthly_to_hourly_rate']:.6%}). Model's own domestic-profile shaping for "
            f"comparison: GBP{domestic_price - blend:,.4f}/MWh over the blend. {shaping['source']}",
            shaping["source_url"])
        energy_cost = blend * (1.0 + rate) * factor * mwh
        shaping_record = {"method": "ofgem_cap_rule", "blend_gbp_per_mwh": blend, "rate": rate,
                          "quarters": quarters, "model_domestic_price_gbp_per_mwh": domestic_price,
                          "model_shaping_over_blend": domestic_price / blend - 1.0}
    else:
        # The energy line stays on the system-weighted price so it remains
        # comparable across reports, and the household's peak-heavy shape enters as
        # its own row. The two together are the domestic-shaped energy cost, which
        # is the base every procurement allowance is applied to.
        add("direct_energy", "hedged_wholesale", "PyPSA wholesale energy (system-weighted)",
            system_price * factor * mwh, "pypsa_hourly_dispatch_prices",
            "Sum(price × system demand × snapshot hours) / annual demand, then currency rebasing",
            "Declared wholesale price and network files")
        add("shaping_adjustment", "hedge_uplifts", "Domestic shaping adjustment",
            (domestic_price - system_price) * factor * mwh, "pypsa_hourly_dispatch_prices",
            f"Domestic-profile-weighted price GBP{domestic_price:,.4f}/MWh less system-weighted "
            f"GBP{system_price:,.4f}/MWh = GBP{domestic_price - system_price:,.4f}/MWh. Household "
            f"consumption is peak-heavy: the profile puts "
            f"{annual['domestic_profile']['peak_share_of_consumption']:.1%} of volume into the "
            f"{annual['domestic_profile']['peak_share_of_hours']:.1%} of hours forming the peak "
            "block. This row replaces the shaping that was previously inside the flat procurement "
            "bridge; it is not an additional charge on top of it.",
            bridge["domestic_shaping"]["profile_file"])
        energy_cost = domestic_price * factor * mwh
        shaping_record = {"method": "model_profile", "model_domestic_price_gbp_per_mwh": domestic_price}

    procurement = bridge["procurement_bridge"]
    allowances = procurement["allowances"]
    premium = procurement["forward_premium"]
    composition = procurement.get("rate_composition", PROCUREMENT_ON_SHAPED_ENERGY)
    if composition not in PROCUREMENT_COMPOSITIONS:
        raise ValueError(f"procurement_bridge.rate_composition must be one of {PROCUREMENT_COMPOSITIONS}, "
                         f"got {composition!r}")
    if composition == PROCUREMENT_OFGEM_ANNEX2:
        # F18 (user, 6 Oct 2026): Ofgem's direct-fuel formula exactly (Annex 2, tab 1b): the
        # rates are added together on the index, index x (1 + shaping + the hedging rates),
        # and losses apply after every allowance. Each line is its own share of that total.
        loss_multiplier = _finite(allowances["losses"]["loss_multiplier_gb_average"], "Losses", True)
        shaping_rate = shaping_record.get("rate", 0.0)
        index_cost = energy_cost / (1.0 + shaping_rate)
        rates = {key: _finite(allowances[key]["rate"], label) for key, label in PROCUREMENT_ALLOWANCES
                 if key != "losses"}
        if min(rates.values()) < 0 or loss_multiplier < 1.0:
            raise ValueError("Procurement intensities must be non-negative")
        premium_rate = _finite(premium["rate"], "Forward premium")
        total_rate = shaping_rate + sum(rates.values()) + premium_rate
        index_text = ("the cap-rule blend (direct_energy)" if shaping_record["method"] == "ofgem_cap_rule"
                      else "the domestic-profile energy cost")
        for key, label in PROCUREMENT_ALLOWANCES:
            if key == "losses":
                add("procurement_losses", "losses", label,
                    index_cost * (1.0 + total_rate) * (loss_multiplier - 1.0), "external_intensity_on_pypsa_energy",
                    f"Index x (1 + sum of rates {total_rate:.6g}) x (loss multiplier {loss_multiplier:.4f} - 1): Ofgem "
                    "applies losses after every allowance (Annex 2 tab 1b). The multiplier is the household's, "
                    "the factor every GSP-rate line carries (item F18).",
                    allowances["losses"].get("source", procurement["source"]))
            else:
                add(f"procurement_{key}", "hedge_uplifts", label, index_cost * rates[key],
                    "external_intensity_on_pypsa_energy",
                    f"Rate {rates[key]:.6g} x {index_text}: Ofgem adds its rates together on the index (Annex 2 "
                    "tab 1b), so no allowance is charged on another.", procurement["source"])
        add("procurement_forward_premium", "hedge_uplifts", "Forward premium (sensitivity, default zero)",
            premium_rate * index_cost, "external_intensity_on_pypsa_energy",
            f"Rate x {index_text}; status {premium['status']}. {premium['note']}"
            + (f" {premium['backwardation_reference']}" if premium.get("backwardation_reference") else ""),
            procurement["source"])
        # The sensitivities, as changes to the ex-VAT bill: a rate moves the total by itself x the
        # index, grossed up by losses.
        wholesale_sensitivities = {"forward_premium": {
            name: (float(alt) - premium_rate) * index_cost * loss_multiplier
            for name, alt in (premium.get("sensitivities") or {}).items()}}
        if shaping_record["method"] == "ofgem_cap_rule":
            blend = shaping_record["blend_gbp_per_mwh"]
            wholesale_sensitivities["shaping"] = {
                name: blend * (float(alt) - shaping_rate) * factor * mwh * loss_multiplier
                for name, alt in (shaping.get("sensitivity_rates") or {}).items()}
            wholesale_sensitivities["shaping"]["model_profile_only"] = (
                (domestic_price * (1.0 + total_rate - shaping_rate) - blend * (1.0 + total_rate))
                * factor * mwh * loss_multiplier)
        shaping_record["rate_composition"] = composition
        shaping_record["sum_of_rates_on_index"] = total_rate
        shaping_record["loss_multiplier"] = loss_multiplier
        shaping_record["uplift_rate_on_energy_cost"] = (1.0 + total_rate) * loss_multiplier / (1.0 + shaping_rate) - 1.0
    else:
        for key, label in PROCUREMENT_ALLOWANCES:
            block = allowances[key]
            rate = (
                _finite(block["loss_multiplier_gb_average"], label, True) - 1.0
                if key == "losses" else _finite(block["rate"], label)
            )
            if rate < 0:
                raise ValueError(f"{label} intensity must be non-negative")
            add(f"procurement_{key}", "losses" if key == "losses" else "hedge_uplifts", label,
                rate * energy_cost, "external_intensity_on_pypsa_energy",
                f"{block['formula']} at an observed 2020 intensity of {rate:.6g}",
                procurement["source"])
        add("procurement_forward_premium", "hedge_uplifts", "Forward premium (sensitivity, default zero)",
            _finite(premium["rate"], "Forward premium") * energy_cost, "external_intensity_on_pypsa_energy",
            f"{premium['formula']}; status {premium['status']}. {premium['note']}"
            + (f" {premium['backwardation_reference']}" if premium.get("backwardation_reference") else ""),
            procurement["source"])
        # Review 28 Sep 2026: the wholesale sensitivities, each as the change in the
        # ex-VAT bill against this projection. The allowances are intensities on the
        # shaped energy cost, so a change to it carries them with it.
        uplift = sum((_finite(allowances[k]["loss_multiplier_gb_average"], k, True) - 1.0) if k == "losses"
                     else _finite(allowances[k]["rate"], k) for k, _ in PROCUREMENT_ALLOWANCES) \
            + _finite(premium["rate"], "Forward premium")
        wholesale_sensitivities = {"forward_premium": {
            name: (float(rate) - float(premium["rate"])) * energy_cost
            for name, rate in (premium.get("sensitivities") or {}).items()}}
        if shaping_record["method"] == "ofgem_cap_rule":
            blend, rate = shaping_record["blend_gbp_per_mwh"], shaping_record["rate"]
            wholesale_sensitivities["shaping"] = {
                name: blend * (float(alt) - rate) * factor * mwh * (1.0 + uplift)
                for name, alt in (shaping.get("sensitivity_rates") or {}).items()}
            wholesale_sensitivities["shaping"]["model_profile_only"] = (
                (domestic_price - blend * (1.0 + rate)) * factor * mwh * (1.0 + uplift))
        shaping_record["uplift_rate_on_energy_cost"] = uplift
    # F18: every wholesale line is the household's own energy cost; its system equivalent is
    # at metered supply (declared).
    for key in ["direct_energy", "shaping_adjustment", "procurement_forward_premium"] + [
            f"procurement_{k}" for k, _ in PROCUREMENT_ALLOWANCES]:
        line_recovery_of[key] = (metered_key, 1.0, 1.0,
                                 "household energy cost (wholesale and procurement allowances); system equivalent "
                                 "at metered supply (declared)")

    # Item MR7: the CCS plant in the model (BECCS on a Power BECCS CfD) is settled
    # with every other CfD, then booked on the allocated CCS levy line rather
    # than the CfD line. The carriers are declared in the cfd rule.
    cfd_rule = assumptions["model_linked_rules"]["cfd"]
    ccs_carriers = set(cfd_rule.get("ccs_levy_carriers", []) or [])
    by_strike = (annual.get("cfd_strike_provenance") or {}).get("by_strike") or []
    ccs_payment = float(sum(float(row["payment_gbp"]) for row in by_strike
                            if row.get("carrier") in ccs_carriers))
    add("cfd", "policy", "CfD net payments / paybacks",
        (float(annual["cfd_payment_gbp"]) - ccs_payment) * factor / denominator * mwh,
        "pypsa_support_recovery_sensitivity",
        f"{recovery_note}. {_reference_basis_text(annual)} {_economic_curtailment_text(annual)}"
        + (f" Excludes {sorted(ccs_carriers)}, booked on the allocated CCS levy line." if ccs_carriers else ""),
        f"Balancing dispatch settled against {bridge_source(assumptions)}")
    add("allocated_ccs", "policy", "Allocated CCS levy (Power BECCS CfD, electricity leg)",
        ccs_payment * factor / denominator * mwh,
        "pypsa_support_recovery_sensitivity",
        f"{cfd_rule.get('ccs_levy_note', '')} Settled payment GBP{ccs_payment:,.0f} on {sorted(ccs_carriers)}.",
        f"Balancing dispatch settled against {bridge_source(assumptions)}")
    cfd_basis = ("CfD supplier obligation: a GSP rate on eligible supply, charged on the household's loss-uplifted "
                 "3.1 MWh" if two_volume else "CfD supplier obligation over eligible supplier sales (item D1)")
    line_recovery_of["cfd"] = line_recovery_of["allocated_ccs"] = (cfd_recovery[0], cfd_recovery[1], 1.0, cfd_basis)
    # Task SETTLE2050: the 2050 dispatchable power agreements (gas CCS, hydrogen power), the
    # support settlement's net payment, on the CfD levy base (DPAs are funded through the
    # CfD supplier obligation, ECC FBC section 5.6.1).
    dpa_rule = assumptions["model_linked_rules"].get("dpa")
    if dpa_rule is not None:
        if dpa is None:
            raise ValueError("model_linked_rules.dpa is declared but no DPA settlement was supplied")
        dpa_volume = "cfd_eligible_gsp_volume_mwh" if two_volume else "cfd_eligible_supplier_sales_mwh"
        if dpa_rule.get("recovery_denominator", dpa_volume) != dpa_volume:
            raise ValueError(f"dpa.recovery_denominator: only {dpa_volume} (the CfD's own base) is implemented")
        dpa_total = _finite(dpa["net_support_gbp"], "DPA net support")
        parts = dpa.get("summary", {}).get("dpa", {})
        add("dpa", "policy", dpa_rule.get("label", "Dispatchable power agreements (gas CCS and hydrogen power)"),
            dpa_total * factor / denominator * mwh, "pypsa_support_recovery_sensitivity",
            f"{dpa_rule.get('formula', '')} Availability GBP{float(parts.get('availability_payment_gbp', 0.0)) / 1e6:,.0f}M "
            f"+ variable GBP{float(parts.get('variable_payment_gbp', 0.0)) / 1e6:,.0f}M "
            f"- gainshare GBP{float(parts.get('gainshare_gbp', 0.0)) / 1e6:,.0f}M "
            f"= GBP{dpa_total / 1e6:,.0f}M (model prices). {recovery_note}.",
            str(dpa.get("path", "model_linked_rules.dpa.settlement_summary")))
        line_recovery_of["dpa"] = (cfd_recovery[0], cfd_recovery[1], 1.0, cfd_basis)

    constraint_cost = _finite(balancing["constraint_cost_gbp"], "Constraint cost")
    bsuos_rule = assumptions["model_linked_rules"]["bsuos_constraints"]
    retrading_text = ""
    if balancing.get("include_interconnector_retrading") and balancing.get("interconnector_retrading_cost_gbp") is not None:
        retrading_text = (
            f" Includes interconnector re-trading of GBP{balancing['interconnector_retrading_cost_gbp'] / 1e6:,.0f}M, "
            "the interconnector schedule change valued at the CP30 bid/ask trade prices, which NESO's "
            "constraint-management category also carries; it is reported separately so it can be excluded."
        )
    if balancing.get("institution") == CENTRAL_DISPATCH_INSTITUTION:
        bsuos_method = _central_dispatch_constraint_text(balancing) + f" {recovery_basis_text}."
    else:
        bsuos_method = (
            f"{bsuos_rule['formula']}. Accepted offer GBP{balancing['accepted_offer_cost_gbp'] / 1e6:,.0f}M "
            f"+ accepted bid GBP{balancing['accepted_bid_cost_gbp'] / 1e6:,.0f}M "
            f"- feasibility slack GBP"
            f"{float(balancing.get('feasibility_slack_cost_in_accepted_costs_gbp', balancing['feasibility_slack_cost_gbp'])) / 1e6:,.2f}M "
            f"- non-participant penalty turn-down GBP"
            f"{float(balancing.get('non_participant_turn_down_cost_at_penalty_gbp') or 0.0) / 1e6:,.1f}M"
            + (f" - decision-20 tie-break steps GBP{float(balancing['tie_break_step_cost_gbp']) / 1e6:,.1f}M "
               "(they order a tie, they are not paid)" if balancing.get("tie_break_step_cost_gbp") else "")
            + ". "
            f"{_compensation_text(balancing)}{retrading_text} {bsuos_rule['excludes']} {recovery_basis_text}.")
    add("bsuos_constraints", "system_balancing", "BSUoS constraint costs (PyPSA)",
        constraint_cost * factor / gsp_rate_denominator * mwh, "pypsa_redispatch_recovery",
        bsuos_method, balancing["path"])
    gsp_rate_basis = ("a GSP rate charged on the household's loss-uplifted 3.1 MWh (item F18)" if two_volume
                      else "over metered supply, the GSP rate on the household's loss-uplifted 3.1 MWh (item D1)")
    line_recovery_of["bsuos_constraints"] = (*gsp_rate_recovery, 1.0, f"BSUoS: {gsp_rate_basis}")

    ro_rule = assumptions["model_linked_rules"]["renewables_obligation"]
    add("renewables_obligation", "policy", "Renewables Obligation (ROC-eligible fleet)",
        ro["gbp_per_3100kwh_ex_vat"], "pypsa_informed_fleet_register",
        f"{ro_rule['formula']} Relief case {ro['relief_case']} at "
        f"{ro['exchequer_relief_share']:.0%} relief; gross "
        f"GBP{ro['gross_gbp_per_3100kwh_ex_vat']:,.2f} from {ro['roc_volume'] / 1e6:,.1f}m ROCs "
        + (f"(GB suppliers' {ro['gb_share_of_uk_obligation']:.2%} of UK issuance, NIROCs included) "
           if ro.get("gb_share_of_uk_obligation") else "")
        + f"on {ro['eligible_capacity_mw']:,.0f} MW still accredited in {ro['obligation_year']} "
        f"({ro['expired_capacity_mw']:,.0f} MW expired), valued at "
        f"GBP{ro['notional_roc_value_gbp_per_roc']:,.2f}/ROC.",
        ro["path"])
    line_recovery_of["renewables_obligation"] = (
        ro_key, 1.0, 1.0, "levied per MWh supplied on obligation-eligible supply (metered), no loss uplift")

    # The bill's calendar year (28 Sep 2026). Once declared, every sourced external
    # must state how its own period maps onto it (PERIOD_BASES).
    bill_year = benchmark.get("bill_calendar_year")
    if bill_year is not None and int(bill_year) != year:
        raise ValueError(f"benchmark.bill_calendar_year {bill_year} is not the modelled calendar year {year}")
    sourced = resolve_sourced_externals(assumptions.get("sourced_externals"), recovery, consumption,
                                        calendar_year=year if bill_year is not None else None)
    period_bases = {}
    interim_lines = {}
    supplier_margin = None
    if missing := REQUIRED_SOURCED_EXTERNALS.difference(sourced):
        raise ValueError(
            f"sourced_externals is missing {sorted(missing)}. The flat bsuos_reference allowance "
            "has been withdrawn, so the non-constraint half of BSUoS must be supplied as a "
            "sourced outturn with its price year; it must not be inferred as the reference "
            "total minus the modelled constraint cost."
        )
    for key, block in sourced.items():
        unit = block["unit"]
        if bill_year is not None:
            period_bases[key] = check_period_basis(key, block, year)
        if unit == "gbp_per_mwh_final_demand":
            if block["recovery_denominator"] != "final_demand_mwh":
                raise ValueError(f"{key}: unit and recovery denominator disagree")
            # This outturn is quoted on the model's own price year, so the declared
            # model rebasing block is the validated route to the report basis.
            external_factor = 1.0 if int(block["price_year"]) == report_price_year else _rebasing_factor(
                rebasing, f"{key} rebasing", block["price_year"], report_price_year
            )
            value = _finite(block["value"], key, True) * external_factor * mwh
            basis = (f" Recovered over {block['recovery_denominator']}; "
                     f"GBP{block['value']:.4f}/MWh in {block['price_year']} prices.")
            if not block.get("loss_uplift"):
                raise ValueError(
                    f"{key}: a per-MWh outturn rate is a GSP rate and must declare loss_uplift: true (item D1)")
            line_recovery_of[key] = (*gsp_rate_recovery, 1.0, f"a per-MWh GSP rate: {gsp_rate_basis}")
        elif unit == "gbp_per_household_year":
            # A per-customer allowance from a primary source (WS9). It is quoted
            # at the benchmark consumption on its own price year and carries its
            # own index pair when that year is not the report year.
            if int(block.get("consumption_basis_kwh", consumption)) != int(consumption):
                raise ValueError(
                    f"{key}: quoted for {block.get('consumption_basis_kwh')} kWh, not the "
                    f"{consumption:.0f} kWh benchmark"
                )
            external_factor = 1.0 if int(block["price_year"]) == report_price_year else _rebasing_factor(
                block["rebasing"], f"{key} rebasing", block["price_year"], report_price_year
            )
            # A declared nil (value_basis "nil", item F1: ECO is off bills from April
            # 2026) is a sourced zero; any other line must be positive.
            nil = block.get("value_basis") == "nil"
            # A signed line (review 28 Sep 2026: Ofgem's smart-meter net cost change is
            # negative since the July 2025 restructuring) may be below zero.
            signed = block.get("value_basis") == "signed"
            if period_bases.get(key) == "calendar_blend" and block.get("period_blend"):
                # Fiscal-year components, each deflated on its own price year.
                value, basis = blended_household_value(key, block, year, report_price_year)
            else:
                value = _finite(block["value"], key, not (nil or signed)) * external_factor
                if nil and value != 0.0:
                    raise ValueError(f"{key}: value_basis nil but value {block['value']}")
                basis = f" GBP{block['value']:.2f} per household in {block['price_year']} prices."
            if key == SUPPLIER_LINE:
                # F18: Ofgem's direct-debit levelisation, and EBIT and headroom recomputed on the
                # bill's own costs below (both only where the line declares them).
                value, basis, supplier_margin = supplier_line_parts(key, block, external_factor, value, basis)
            derivation = block.get("derivation") or {}
            if "recovery_denominator_mwh_used" in derivation:
                # A cost spread over a recovery volume (resolve_sourced_externals).
                line_recovery_of[key] = (
                    derivation["recovery_denominator"], float(derivation["household_loss_factor_used"]),
                    float(derivation.get("household_allocation_factor", 1.0)),
                    f"declared cost spread over {derivation['recovery_denominator']}"
                    + (" and charged on the household's loss-uplifted 3.1 MWh"
                       if float(derivation["household_loss_factor_used"]) != 1.0 else ""))
            elif block.get("loss_uplift"):
                line_recovery_of[key] = (*gsp_rate_recovery, 1.0,
                                         f"a GSP rate per MWh quoted per household: {gsp_rate_basis}")
            else:
                line_recovery_of[key] = (metered_key, 1.0, 1.0, PER_HOUSEHOLD_RECOVERY_BASIS)
        else:
            raise ValueError(
                f"{key}: sourced externals must be gbp_per_mwh_final_demand or gbp_per_household_year, not {unit!r}"
            )
        # Item D1: a GSP-rate charge falls on the household's loss-uplifted MWh (F18: the
        # household loss factor).
        if block.get("loss_uplift"):
            value *= loss_factor
            basis += (f" Uplifted by the household loss factor {loss_factor:.4f} (item F18)." if two_volume
                      else f" Uplifted by the GSP-to-meter loss factor {loss_factor:.4f} (item D1).")
        source_class = block.get("source_class", "external_sourced_outturn")
        if source_class not in SOURCED_EXTERNAL_CLASSES:
            raise ValueError(f"{key}: source_class must be one of {sorted(SOURCED_EXTERNAL_CLASSES)}")
        if key in EXTERNAL_COMPONENTS and source_class != "external_primary_source":
            raise ValueError(f"{key}: a line replacing an analyst placeholder must be external_primary_source")
        if key in period_bases:
            basis += f" Period basis {period_bases[key]}" + (
                f" (held from {block['source_period']})" if period_bases[key] == "held_flat_real" else "") + (
                f" (from {block['extrapolation']['from_period']})" if period_bases[key] == "extrapolated_declared"
                else "") + "."
        # Task Y13: a line carried at a declared interim value until another task's model
        # output replaces it says so in its method and in the summary.
        if block.get("interim"):
            interim = block["interim"]
            if not (interim.get("until") and interim.get("replaced_by")):
                raise ValueError(f"{key}: an interim line needs interim.until and interim.replaced_by")
            interim_lines[key] = {"until": interim["until"], "replaced_by": interim["replaced_by"]}
            basis += f" INTERIM until {interim['until']}: {interim['replaced_by']}."
        add(key, block["stack_group"], block["label"], value, source_class,
            f"{block['method']}{basis}", block["source_url"],
            reference=block.get("reference_gbp_per_3100kwh_ex_vat"))

    # Congestion rent the regulated interconnectors hand back through TNUoS. The
    # re-trading in the constraint line is NESO's balancing trade on the cables, a different cash
    # flow from the owners' day-ahead capacity-allocation income, so the two do
    # not overlap.
    ic_rule = assumptions["model_linked_rules"]["interconnector_cap_and_floor"]
    pass_through = None
    if ic_rule.get("enabled", True):
        rent_by_cable = annual.get("interconnector_congestion_rent_by_cable_gbp")
        # Review 26 Sep 2026: a month without per-cable rent would otherwise be left
        # out of the credit silently (the aggregate starts as {}, never None).
        missing_months = annual.get("interconnector_rent_months_without_cables") or []
        if rent_by_cable is None or missing_months:
            raise ValueError(
                "The interconnector cap-and-floor line needs the per-cable congestion rent from the "
                "market summaries (interconnectors.by_cable) for every month; missing for "
                f"{missing_months or 'all months'}. Disable the rule to project without it."
            )
        # Central dispatch (decision 4, 20 Oct 2026): the rent on the central schedule with its make-whole.
        rent_by_cable = rent_by_cable_for_institution(rent_by_cable, balancing)
        pass_through = interconnector_pass_through(rent_by_cable, ic_rule, model_price_year)
        household_share = tnuos_household_share(assumptions)
        pass_through.update(household_share_of_tnuos_residual=household_share)
        add("interconnector_cap_and_floor_pass_through", "network",
            "Interconnector cap-and-floor pass-through (TNUoS credit)",
            -pass_through["credit_to_consumers_gbp"] * factor * household_share,
            "pypsa_informed_regulatory_pass_through",
            f"{ic_rule['formula']} GBP{pass_through['credit_to_consumers_gbp'] / 1e6:,.1f}M returned "
            f"of GBP{pass_through['congestion_rent_gbp'] / 1e6:,.0f}M whole-cable Stage 1 congestion rent; "
            f"one household bears {household_share:.3e} of the TNUoS demand residual. {ic_rule['note']}",
            "; ".join(ic_rule.get("sources", [])) or "model_linked_rules.interconnector_cap_and_floor")
        # The credit reaches a household through the TNUoS demand residual, not as an energy share:
        # as an allocation factor on metered supply, one household's residual share x metered supply / 3.1 MWh.
        line_recovery_of["interconnector_cap_and_floor_pass_through"] = (
            metered_key, 1.0, household_share * final_demand / mwh,
            "a credit through the TNUoS demand residual: the household's share of the residual, expressed as an "
            "allocation factor on metered supply")

    # Item D4(c): the LDES cap-and-floor top-up is a line of the forecast bill,
    # not something the ownership projector adds, so there is one private
    # headline. The top-up is the finance register's: max(0, floor at the case's
    # debt-like rate - market revenue) over the long-duration storage, where the
    # market revenue is the model's.
    ldes_rule = assumptions["model_linked_rules"]["ldes_cap_and_floor"]
    ldes_report = None
    if ldes_rule.get("enabled", True):
        if ldes is None:
            raise ValueError(
                "model_linked_rules.ldes_cap_and_floor is enabled but no LDES floor top-up was supplied; pass the "
                "run group's public account (ldes_public_account) or disable the rule"
            )
        ldes_volume = "gsp_charging_volume_mwh" if two_volume else "final_demand_mwh"
        if ldes_rule.get("recovery_denominator", ldes_volume) != ldes_volume:
            raise ValueError(f"ldes_cap_and_floor.recovery_denominator: only {ldes_volume} is implemented on this "
                             "recovery basis (F18: LDES is a GSP-rate line)")
        topup = _finite(ldes["topup_gbp"], "LDES floor top-up")
        if topup < 0:
            raise ValueError("The LDES floor top-up cannot be negative")
        ldes_report = {**ldes, "topup_gbp": topup}
        add("ldes_cap_and_floor", "policy", "LDES cap and floor top-up (floor at required revenue)",
            topup * factor / gsp_rate_denominator * mwh, "pypsa_informed_regulatory_pass_through",
            f"{ldes_rule['formula']} GBP{topup / 1e6:,.0f}M (model prices) for case {ldes.get('case_id', 'private')}"
            f"{' at a ' + format(float(ldes['floor_real_rate']), '.2%') + ' real floor return' if ldes.get('floor_real_rate') is not None else ''}. "
            f"{recovery_basis_text}. {ldes_rule['note']}",
            str(ldes.get("source", "public account")))
        line_recovery_of["ldes_cap_and_floor"] = (*gsp_rate_recovery, 1.0, f"LDES cap and floor: {gsp_rate_basis}")

    external = assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"]
    replaced = {key for key in sourced if key in EXTERNAL_COMPONENTS}
    if replaced & set(external):
        raise ValueError(
            f"{sorted(replaced & set(external))} appear both as a primary-source line and as an analyst "
            "placeholder; keep the placeholder only as reference_gbp_per_3100kwh_ex_vat on the sourced line"
        )
    if set(external) | replaced != set(EXTERNAL_COMPONENTS):
        raise ValueError(
            f"External component schema mismatch: {sorted((set(external) | replaced) ^ set(EXTERNAL_COMPONENTS))}"
        )
    source_note = (
        "Benchmark input carried to the report price basis; no additional model charge"
        if source_factor != 1.0
        else "Benchmark input already on the report price basis; no additional model charge"
    )
    for key, (group, label) in EXTERNAL_COMPONENTS.items():
        if key in replaced:
            continue  # carried above as a primary-source line
        add(key, group, label, external[key] * source_factor, "external_assumption", source_note, assumptions["source"]["page_url"])
        line_recovery_of[key] = (metered_key, 1.0, 1.0, PER_HOUSEHOLD_RECOVERY_BASIS)

    # F18: EBIT and headroom at Ofgem's percentages of the bill's own costs (the projector's lines;
    # the ownership projector supplies the private case's base so every case pays the same margin).
    if supplier_margin is not None:
        supplier_margin = apply_supplier_margin(rows, supplier_margin, supplier_margin_base)

    # Lines a caller adds on the report basis (the public-ownership bill link:
    # LDES cap and floor, public-account recycling, shortfall levy). They go
    # through the same subtotal so VAT is still applied once.
    existing = {row["component_id"] for row in rows}
    # F19 (7 Oct 2026): an extra component may state its own recovery, ``line_recovery``
    # = (volume key, household loss factor, household allocation factor, basis), when it is
    # not recovered like the line it adjusts; ``line_recovery_volumes`` names any volume the
    # recovery output does not carry (the domestic supply the levy offset is spread over).
    extra_volumes = {}
    for extra in extra_components or []:
        if extra["component_id"] in existing:
            raise ValueError(f"extra component {extra['component_id']!r} duplicates a projector line")
        if extra["stack_group"] not in group_names:
            raise ValueError(f"extra component {extra['component_id']!r} names unknown stack group {extra['stack_group']!r}")
        add(extra["component_id"], extra["stack_group"], extra["label"], extra["annual_gbp_per_3100kwh_ex_vat"],
            extra.get("source_class", "public_ownership_bill_link"), extra["method"], extra["source"],
            reference=extra.get("reference_gbp_per_3100kwh_ex_vat"))
        existing.add(extra["component_id"])
        if extra.get("line_recovery") is not None:
            line_recovery_of[extra["component_id"]] = tuple(extra["line_recovery"])
            for key, volume in (extra.get("line_recovery_volumes") or {}).items():
                if key in recovery and recovery[key] is not None and float(recovery[key]) != float(volume):
                    raise ValueError(f"extra component {extra['component_id']!r} restates volume {key!r} differently")
                extra_volumes[key] = float(volume)
    # F18: every line's recovery; lines the ownership projector adds copy the entry of the line they adjust.
    line_recovery = build_line_recovery(
        rows, line_recovery_of, {**extra_volumes, **recovery, metered_key: final_demand,
                                 "modelled_system_demand_mwh": annual.get("demand_mwh")},
        factor, model_price_year)

    # 6 Oct 2026: each line's Ofgem price-cap group (at the current cap release) and its
    # functional group, from bill_groupings.csv; a line with no group is refused here.
    frame = annotate_bill_groups(pd.DataFrame(rows))
    ex_vat = float(frame.annual_gbp_per_3100kwh_ex_vat.sum())
    # Fully model-derived: the physical quantity and its price both come from the
    # run. The RO row is deliberately excluded and counted separately, because
    # only its load factors are model-derived while its fleet and certificate
    # value are external.
    pypsa_classes = {"pypsa_hourly_dispatch_prices", "pypsa_support_recovery_sensitivity",
                     "pypsa_redispatch_recovery"}
    pypsa_derived = float(
        frame.loc[frame.source_class.isin(pypsa_classes), "annual_gbp_per_3100kwh_ex_vat"].sum()
    )
    # The interconnector pass-through is informed the same way: the rent is the
    # model's, the regulatory cap and floor levels are Ofgem's.
    pypsa_informed = float(
        frame.loc[frame.source_class.isin({"pypsa_informed_fleet_register",
                                           "pypsa_informed_regulatory_pass_through"}),
                  "annual_gbp_per_3100kwh_ex_vat"].sum()
    )
    bridge_rows = frame.loc[frame.component_id.str.startswith("procurement_")
                            | frame.component_id.eq("shaping_adjustment")]
    diagnostics = _diagnostics(annual, balancing, assumptions, factor, gsp_rate_denominator, mwh, ex_vat,
                               float(bridge_rows.annual_gbp_per_3100kwh_ex_vat.sum()), ro,
                               denominator=denominator, source_factor=source_factor,
                               pass_through=pass_through, household_loss=loss_factor if two_volume else 1.0,
                               other_services_line=float(frame.loc[frame.component_id.eq("bsuos_other_services"),
                                                                   "annual_gbp_per_3100kwh_ex_vat"].sum()))
    # Review 28 Sep 2026: how the wholesale line was shaped, and its sensitivities
    # (GBP per 3,100 kWh ex VAT against this bill; the scarcity mark-up is inside
    # the Stage 1 price and is reported with the aggregate when the months record it).
    diagnostics["levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat"] = levy_and_cm_sensitivities(
        frame, assumptions, recovery, consumption)
    diagnostics["wholesale_procurement"] = {
        "shaping": shaping_record,
        "sensitivities_gbp_per_3100kwh_ex_vat": wholesale_sensitivities,
        "scarcity_markup": annual.get("scarcity_markup"),
        "forward_premium_status": procurement["forward_premium"].get("status"),
    }
    # F18 (27 Oct 2026): the constraint line split into thermal redispatch and interconnector
    # re-trading, beside NESO's thermal constraint cost for the case's network.
    diagnostics["constraint_line_split"] = constraint_line_split(
        balancing, case, household_per_system=float(factor) / float(gsp_rate_denominator) * float(mwh))
    # WS9: every analyst placeholder replaced by a primary source, with the delta.
    primary = frame.loc[frame.source_class.eq("external_primary_source")]
    diagnostics["primary_source_replacements"] = [
        {"component_id": row.component_id,
         "gbp_per_3100kwh_ex_vat": float(row.annual_gbp_per_3100kwh_ex_vat),
         "reference_gbp_per_3100kwh_ex_vat": (
             None if pd.isna(row.reference_gbp_per_3100kwh_ex_vat) else float(row.reference_gbp_per_3100kwh_ex_vat)),
         "delta_against_reference_gbp_per_3100kwh_ex_vat": (
             None if pd.isna(row.reference_gbp_per_3100kwh_ex_vat)
             else float(row.annual_gbp_per_3100kwh_ex_vat - row.reference_gbp_per_3100kwh_ex_vat)),
         "source": row.source}
        for row in primary.itertuples()
    ]
    diagnostics["external_placeholders_remaining"] = sorted(
        frame.loc[frame.source_class.eq("external_assumption"), "component_id"].tolist()
    )
    # WS9 item 9: the same bill at other consumption levels (Ofgem's 2,700 kWh
    # typical value). Per-MWh lines scale with consumption; primary-source
    # per-household allowances do not; analyst placeholders are scaled as if
    # volumetric because their fixed and unit split is not published.
    # Item G6: archetype bills on a declared standing and unit split, replacing the
    # rule that held every primary-source line fixed and scaled the rest. The
    # alternative volumes keep their old diagnostic key for readers of it.
    alternatives = {}
    if "household_archetypes" in assumptions and "charge_basis" in assumptions:
        from scripts.consumer_bill.household_archetypes import project_archetypes

        archetypes = project_archetypes(frame, assumptions, vat_rate=vat_rate)
        diagnostics["household_archetypes"] = archetypes
        by_name = {row["archetype"]: row for row in archetypes["bills"]}
        for alt_kwh in benchmark.get("alternative_consumption_kwh") or []:
            row = by_name[f"alternative_{int(alt_kwh)}"]
            alternatives[str(int(alt_kwh))] = {
                "annual_consumption_kwh": float(alt_kwh), "ex_vat_gbp": row["ex_vat_gbp"], "vat_gbp": row["vat_gbp"],
                "total_gbp_inc_vat": row["total_gbp_inc_vat"],
                "fixed_per_household_gbp_ex_vat": row["standing_charge_gbp_ex_vat"],
                "unit_rate_p_per_kwh_ex_vat": row["unit_rate_p_per_kwh_ex_vat"],
                "basis": "declared charge_basis (item G6)", "note": benchmark.get("alternative_consumption_note"),
            }
    diagnostics["alternative_consumption_bills"] = alternatives
    summary = {**annual, "sources": annual.get("sources", []), "scenario": case["case_id"],
               "input_mode": "future_projection", "currency": benchmark["price_basis"], "model_price_year": model_price_year,
               "report_price_year": report_price_year, "model_rebasing_factor": factor,
               "source_price_year": int(assumptions["source"]["price_year"]), "source_rebasing_factor": source_factor,
               "period_alignment_note": period_note, "annual_consumption_kwh": consumption,
               "bill_calendar_year": int(bill_year) if bill_year is not None else None,
               "external_period_bases": period_bases,
               "levy_exemptions": levy_exemptions,
               "cfd_recovery_method": method,
               # DEPRECATED on the two-volume basis (F18): the CfD line's eligible GSP volume over the
               # household loss factor, so a household line x this / 3.1 MWh is still the CfD's system
               # cost; and metered supply. Read line_recovery instead.
               "cfd_recovery_denominator_mwh": denominator,
               "final_demand_recovery_denominator_mwh": final_demand,
               # F18: one entry per bill line, the only conversion between household and system pounds.
               "line_recovery": line_recovery,
               # F18: the Exchequer's per-household key (a system cost's household share is 3.1 MWh over it).
               "exchequer_per_household_key": {
                   "volume_key": metered_key, "volume_mwh": final_demand,
                   "households_equivalent": final_demand / mwh,
                   "basis": "metered supply: one household's 3.1 MWh is its share of a cost spread over it",
               },
               "supplier_margin": supplier_margin,
               "recovery_basis": _recovery_basis_summary(recovery, final_demand, two_volume) if two_volume else {
                   "basis": "metered_supply",
                   "metered_supply_mwh": final_demand,
                   "gsp_chargeable_volume_mwh": recovery.get("gsp_chargeable_volume_mwh"),
                   "gsp_to_meter_loss_factor": recovery.get("gsp_to_meter_loss_factor"),
                   "gb_electrical_load_mwh": recovery.get("gb_electrical_load_mwh"),
                   "electrolyser_draw_mwh": recovery.get("electrolyser_draw_mwh"),
                   "h2_offtake_excluded_mwh": recovery.get("h2_offtake_mwh"),
                   "note": ("Every per-MWh line is a share of metered supply: the GSP-rate lines (CfD, allocated "
                            "CCS, remaining CCS, BSUoS constraints and other services, voltage and stability "
                            "constraints, AAHEDC, LCCC/ESC, LDES) are a GSP rate on the household's 3.1 MWh times "
                            "the loss factor; the RO and FiT are levied on metered supply directly. The Capacity "
                            "Market divides by the GSP volume because its household factor already carries losses. "
                            "Electrolysers pay on their Stage 1 draw; the hydrogen offtake pays nothing."),
               },
               "ldes_cap_and_floor": ldes_report,
               "recovery_denominators": recovery, "balancing_costs": balancing,
               "ro_scheme_cost": ro,
               "pypsa_derived_gbp_per_3100kwh_ex_vat": pypsa_derived,
               "pypsa_derived_share_of_ex_vat": pypsa_derived / ex_vat if ex_vat else None,
               "pypsa_informed_gbp_per_3100kwh_ex_vat": pypsa_informed,
               "pypsa_informed_share_of_ex_vat": pypsa_informed / ex_vat if ex_vat else None,
               "diagnostics": diagnostics,
               "groupings": bill_group_totals(frame),
               "ex_vat_gbp": ex_vat, "vat_gbp": ex_vat * vat_rate, "total_gbp_inc_vat": ex_vat * (1 + vat_rate),
               "vat_rate": vat_rate, "is_statutory_cap_reproduction": False, "forecast_validated": False,
               "limitations": [recovery_note, _strike_limitation(annual.get("cfd_strike_provenance")),
                   "CfD uses balancing dispatch and published allocation-round strike prices indexed from 2012 prices with CPI; complete contract coverage is not independently validated here.",
                   "CfD assets without stored dispatch are counted in the source audit; their unreported energy is not imputed. Dispatch coverage excludes those assets.",
                   _zero_price_limitation(annual),
                   _bsuos_limitation(balancing),
                   _interconnector_limitation(pass_through),
                   ("Levies are recovered on two volumes (item F18): metered supply = GB load / "
                    f"{float(recovery.get('system_to_metered_factor') or 0.0):.3f} + electrolyser draw, and the GSP "
                    f"charging volume = metered load x {float(recovery.get('metered_to_gsp_factor') or 0.0):.3f} + draw, "
                    "both declared outturn ratios held for the year; electrolysers pay on their Stage 1 draw (the "
                    "balancing stage moves it slightly; that change is not metered here). GSP-rate lines carry Ofgem's "
                    f"current single-rate household loss factor {loss_factor:.4f}, the procurement losses multiplier."
                    if two_volume else
                    "Levies are recovered on metered supply (item D1): electrolysers pay on their Stage 1 "
                    "electricity draw (the balancing stage moves it slightly; that change is not metered here) and "
                    "the loss factor is Ofgem's 2020 GB-average transmission-to-meter multiplier, the one the "
                    f"procurement losses line applies, held for {year}."),
                   ("Shaping follows the Ofgem cap rule: a 70/30 baseload/peak blend of the model's prices per quarter plus Ofgem's shaping allowances (0.23% + 4.16%, set in 2018 and unchanged to Q4 2026). The model's own domestic profile is the same two-block allocation, so it cannot measure within-block structure such as the post-19:00 evening peak; the allowance stands in for it until Ofgem resets it from smart-meter data (decision planned early 2027)."
                    if shaping_record["method"] == "ofgem_cap_rule" else
                    "Domestic shaping uses Ofgem's two-block base/peak allocation, not an Elexon Profile Class 1 half-hourly regression, so within-block structure such as the post-19:00 domestic evening peak is absent."),
                   "Procurement allowances are Ofgem's named intensities (unchanged from 2019 to Q4 2026) relative to the energy cost. Modelled spot stands in for the forward index with no forward premium: the historical index-to-outturn gap reverses sign with the price trend, so it is not carried; the premium is reported as -3%, +5% and +10% sensitivities.",
                   "Network, RO, FiT, other policy and retail charges use explicit external assumptions.",
                   "DUoS and supplier costs/margin remain aggregates because the source does not provide a defensible finer split.",
                   period_note]}
    if annual.get("price_level") == "cp32":
        summary["limitations"].append(_cp32_price_level_limitation(annual))
    if interim_lines:
        # Task Y13: lines at a declared interim value, so no headline is read from them silently.
        summary["interim_external_lines"] = interim_lines
    return frame, summary


def _cp32_price_level_limitation(annual):
    """Item F12: what the cp32 price level is and is not."""
    mapping = annual.get("price_level_mapping") or {}
    model = (mapping.get("model") or {}).get("time_weighted_gbp_per_mwh")
    mapped = (mapping.get("mapped") or {}).get("time_weighted_gbp_per_mwh")
    level = (f" (time-weighted GBP{float(model):,.2f}/MWh on the model's price, GBP{float(mapped):,.2f}/MWh mapped)"
             if model is not None and mapped is not None else "")
    return (
        "Price level cp32 (item F12): every model hour keeps its rank in the whole year and takes CP30's CP.32 "
        f"{mapping.get('pathway', '')} price at that rank{level}. Static: dispatch, every volume and the "
        "constraint line's accepted bids and offers are as solved at the model's price (its interconnector "
        "re-trading is re-valued, item F24.2: valued on the foreign side, it keeps its value), so this is a "
        "bracket on the price level, not a forecast. The zero- and negative-price hours reported are the mapped series', set by CP.32's curve rather "
        "than by modelled bids; the economic-curtailment rule still selects the hours the LP curtailed in on the "
        "model's own price. Lines held at the model price: "
        + "; ".join(sorted(annual.get("price_level_held_at_model_price") or {})) + "."
    )


def _recovery_basis_summary(recovery, metered, two_volume):
    """F18: the summary's record of the two recovery volumes and their factors."""
    return {
        "basis": TWO_VOLUME_BASIS,
        "metered_supply_mwh": metered,
        "gsp_charging_volume_mwh": recovery["gsp_charging_volume_mwh"],
        "cfd_eligible_gsp_volume_mwh": recovery["cfd_eligible_gsp_volume_mwh"],
        "ro_eligible_metered_supply_mwh": recovery["ro_eligible_metered_supply_mwh"],
        "system_to_metered_factor": recovery["system_to_metered_factor"],
        "metered_to_gsp_factor": recovery["metered_to_gsp_factor"],
        "household_loss_factor": recovery["household_loss_factor"],
        "gb_electrical_load_mwh": recovery.get("gb_electrical_load_mwh"),
        "electrolyser_draw_mwh": recovery.get("electrolyser_draw_mwh"),
        "h2_offtake_excluded_mwh": recovery.get("h2_offtake_mwh"),
        "deprecated_aliases": dict(DEPRECATED_RECOVERY_ALIASES),
        "note": ("Item F18: the model's GB load contains network losses. Metered supply carries the RO and FiT (each "
                 "over its eligible share), the battery top-up and the Exchequer's per-household key; the GSP charging "
                 "volume carries the CfD, allocated CCS and the NZT DPA (over the eligible share), LDES and the three "
                 "BSUoS lines, each charged on the household's 3.1 MWh x the household loss factor, as are AAHEDC and "
                 "the LCCC/ESC levies; the Capacity Market is spread over the GSP charging volume with its own "
                 "household allocation factor. Electrolysers pay on their Stage 1 draw; the hydrogen offtake pays "
                 "nothing. line_recovery states each line's volume and factors."),
    }


def _reference_basis_text(annual):
    basis = annual.get("cfd_reference_price_basis") or {}
    carriers = basis.get("baseload_carriers") or []
    if not carriers:
        return "Every contract settles against the hourly wholesale reference price."
    seasons = basis.get("seasons") or {}
    season_text = "; ".join(
        f"{name} GBP{detail['reference_price_gbp_per_mwh']:,.2f}/MWh" for name, detail in seasons.items()
    )
    return (
        f"Baseload contracts ({', '.join(carriers)}) settle against the {basis.get('baseload_reference')} "
        f"reference ({season_text}); all other contracts settle against the hourly wholesale price."
    )


def _clears_negative(annual) -> bool:
    """True when modelled hours actually clear below zero (not solver noise at -1e-6)."""
    minimum = annual.get("minimum_price_gbp_per_mwh")
    return minimum is not None and float(minimum) < -NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH


def _economic_curtailment_text(annual):
    rule = annual.get("cfd_economic_curtailment_rule", "none")
    rounds = ", ".join(annual.get("cfd_economic_curtailment_rounds") or [])
    if rule == "none":
        if annual.get("wholesale_curtailment_order") == "contract_aware":
            return (
                "Economic-curtailment rule 'none': the wholesale used the contract-aware curtailment order, "
                "so pre-AR4 contracts are curtailed only at a price at or below minus their strike, where "
                "the real generator backs off itself; no payment is due on that energy."
            )
        return (
            "Economic-curtailment rule 'none': output the wholesale LP curtailed at ~GBP0 accrues no "
            "difference payment, including for pre-AR4 contracts that carry no zero-payment term."
        )
    return (
        f"Economic-curtailment rule '{rule}': GBP{annual.get('cfd_economic_curtailment_payment_gbp', 0.0) / 1e6:,.0f}M "
        f"on {annual.get('cfd_economic_curtailment_mwh', 0.0) / 1e6:,.2f} TWh of pre-AR4 output ({rounds}) that "
        f"the wholesale LP curtailed in hours clearing at or below "
        f"GBP{annual.get('cfd_economic_curtailment_zero_price_threshold_gbp_per_mwh', 0.0):,.1f}/MWh is paid at "
        "strike minus the hourly price and included in this row."
    )


def _zero_price_limitation(annual):
    rule = annual.get("cfd_economic_curtailment_rule", "none")
    negative_rule = annual.get("cfd_negative_price_rule")
    if _clears_negative(annual):
        # Contract-aware order (AR4+ at 0, merchant at +0.5, older contracts at
        # minus strike): hours can clear negative and each round's own term
        # decides what is paid in them.
        return (
            f"{annual.get('hours_price_below_minus_0_01_gbp_per_mwh', 0.0):,.0f} modelled hours clear below zero (minimum "
            f"GBP{float(annual['minimum_price_gbp_per_mwh']):,.2f}/MWh; wholesale curtailment order "
            f"'{annual.get('wholesale_curtailment_order', 'unknown')}'). The CfD line applies the "
            f"'{negative_rule}' negative-price rule in them"
            + (": AR4 and later contracts are unpaid in any negative hour, AR2-AR3 inside runs of six or "
               "more negative hours, and AR1, Investment Contracts and Hinkley Point C are paid throughout."
               if negative_rule == "gb_contract_terms" else ".")
            + " Negative prices are set by modelled bids (AR4+ at the declared ar4_plus_bid, older "
            "contracts at minus strike, ROC plant at minus its certificate value), not by an observed "
            "negative-price distribution, so the count and depth of negative hours are model outcomes."
            + ("" if rule == "none" else
               f" The '{rule}' economic-curtailment rule is also on; it was designed for an as-built "
               "wholesale and should be 'none' here.")
        )
    base = (
        f"No modelled hour clears below zero, so the '{negative_rule}' negative-price "
        "rule cannot bite: this wholesale gives CfD units no negative bids."
    )
    if rule == "none":
        return (
            f"{base} The wholesale LP instead curtails CfD output at ~GBP0 in "
            f"{annual.get('hours_price_at_or_below_1_gbp_per_mwh', 0.0):,.0f} hours without payment, which "
            "understates pre-AR4 contracts that carry no zero-payment term; the economic_curtailment_rule "
            "is set to 'none', so that payment is not restored."
        )
    return (
        f"{base} The wholesale LP curtails CfD output at ~GBP0 in "
        f"{annual.get('hours_price_at_or_below_1_gbp_per_mwh', 0.0):,.0f} hours; the '{rule}' rule restores "
        f"strike-minus-reference payments on {annual.get('cfd_economic_curtailment_mwh', 0.0) / 1e6:,.2f} TWh of "
        f"that curtailment for the rounds {', '.join(annual.get('cfd_economic_curtailment_rounds') or [])}, "
        "which carry no zero-payment term. Contracts from AR4 onwards remain unpaid in those hours, as their terms provide."
    )


def _central_dispatch_constraint_text(balancing):
    """The constraint line's method under central dispatch (I3, FTI's settlement; 20 Oct 2026)."""
    m = lambda key: float(balancing.get(key) or 0.0) / 1e6  # noqa: E731
    return (
        "Central dispatch on FTI's method (institution I3): cost recovery "
        f"GBP{m('cost_recovery_gbp'):,.0f}M + firm access GBP{m('firm_access_gbp'):,.0f}M (with the lost CfD top-up "
        f"GBP{m('cfd_curtailment_compensation_gbp'):,.0f}M and ROC value on the turned-down day-ahead volume) + storage "
        f"weekly make-whole GBP{m('storage_make_whole_gbp'):,.0f}M + interconnector day-ahead make-whole "
        f"GBP{m('interconnector_make_whole_gbp'):,.0f}M + the borrowed FTI forecast-error charge "
        f"GBP{m('forecast_error_charge_gbp'):,.0f}M"
        + (f" + the central balancing stage's counter-trade premium GBP{m('counter_trade_premium_gbp'):,.1f}M"
           if balancing.get("counter_trade_premium_on_constraint_line") else "")
        + f". The model's own real-time correction (GBP{m('realtime_correction_replaced_gbp'):,.0f}M) is replaced by "
        "the charge; no interconnector re-trading (there is no balancing market). Market-year line for comparison "
        f"GBP{m('market_constraint_cost_gbp'):,.0f}M.")


def _compensation_text(balancing):
    """How the CfD curtailment compensation entered the constraint cost, as the extraction did it."""
    compensation = float(balancing.get("cfd_curtailment_compensation_gbp") or 0.0) / 1e6
    curtailed = float(balancing.get("cfd_curtailed_mwh") or 0.0) / 1e6
    treatment = balancing.get("cfd_curtailment_compensation_treatment")
    if treatment == "reconcile_against_lp_bid":
        return (f"CfD curtailment compensation of GBP{compensation:,.0f}M on {curtailed:,.1f} TWh of turn-down "
                "is already inside the accepted bid cost (contract-value bids) and is recomputed only as a "
                "reconciliation check, not added.")
    if treatment == "add_post_hoc":
        return (f"Plus CfD curtailment compensation of GBP{compensation:,.0f}M on {curtailed:,.1f} TWh of "
                "turn-down, added because the balancing LP priced CfD bids at a scalar.")
    return (f"CfD curtailment compensation GBP{compensation:,.0f}M on {curtailed:,.1f} TWh of turn-down; "
            "the extraction did not record whether it was added or reconciled.")


def _bsuos_limitation(balancing):
    if balancing.get("institution") == CENTRAL_DISPATCH_INSTITUTION:
        return ("BSUoS constraint cost under central dispatch is FTI's settlement of a perfect-foresight central "
                "schedule against the market twin's copperplate schedule (cost recovery, firm access with the lost "
                "support, the storage and interconnector make-whole), plus FTI's forecast-error charge scaled to this "
                "fleet in place of the model's own real-time correction; it is not NESO settlement expenditure. "
                "Non-constraint balancing services remain a NESO outturn level held flat in real terms.")
    if balancing.get("cfd_curtailment_compensation_treatment") == "reconcile_against_lp_bid":
        middle = ("in which CfD units bid their contract value, so their compensation is inside the "
                  "accepted bid cost rather than a post-hoc term")
    else:
        middle = "plus a post-hoc CfD compensation term"
    return (f"BSUoS constraint cost is modelled redispatch against fixed wholesale positions, {middle}; "
            "it is not NESO settlement expenditure. Non-constraint balancing services remain a NESO outturn "
            "level held flat in real terms.")


def _interconnector_limitation(pass_through):
    if pass_through is None:
        return ("The interconnector cap-and-floor pass-through is disabled; the congestion rent is a "
                "diagnostic only.")
    return (
        "The interconnector credit applies Ofgem cap and floor levels to one modelled year of Stage 1 "
        "congestion rent. The foreign prices are the IC1 supply curves (item IC1, 22 Sep 2026): the "
        "ENTSO-E 2024 hourly shape conditioned on 2013 GB wind, with volume tranches on three coupled "
        "markets, not the flat CP30 trade prices of earlier vintages. The tranche cost enters the rent "
        "as the area under the stepped curve (what the LP charges), not as a uniform clearing price on "
        "the whole flow, so the rent is if anything overstated (methodology note, item B9). The year "
        "stands in for a five-year assessment, Capacity Market income (which the cap test also counts) "
        "is left out. NESO's TNUoS forecast carries no interconnector cap-and-floor adjustment of its own "
        "(checked 28 Sep 2026: its OFTO-and-TICF line is OFTO revenue alone, FYV v3.0 Tables 38 and 39), "
        "so the credit is not counted twice; Ofgem's lagged assessment would move it to a later year."
    )


def bridge_source(assumptions):
    return (
        f"published allocation-round strike prices from "
        f"{assumptions['model_linked_rules']['cfd']['strike_table']}, indexed "
        f"{assumptions['cfd_strike_indexation']['from_price_year']}"
        f"->{assumptions['cfd_strike_indexation']['to_price_year']} by "
        f"{assumptions['cfd_strike_indexation']['factor']:.6f}"
    )


def _unlabelled_support_diagnostics(balancing, assumptions, factor, mwh, annual=None, denominator=None):
    """Size the support labels the CP30 fleet does not carry, and the nuclear CfD it does.

    While the networks carried unlabelled nuclear proxies this measured the
    missing Hinkley Point C line. With HPC labelled as a CfD (round HPC) the
    settled nuclear payment is read from the strike provenance and the
    HPC-strike sensitivity becomes a cross-check on it.
    """
    rows = {row["carrier"]: row for row in (balancing.get("unlabelled_support") or [])}
    exclusion = assumptions["model_linked_rules"]["cfd"].get("nuclear_exclusion") or {}
    nuclear = rows.get("nuclear") or {}
    energy = float(nuclear.get("physical_mwh") or 0.0)
    strike = exclusion.get("hpc_strike_gbp_per_mwh_model_prices")
    reference = exclusion.get("sensitivity_reference_price_gbp_per_mwh")
    sensitivity = None
    if energy > 0 and strike is not None and reference is not None:
        sensitivity = max(float(strike) - float(reference), 0.0) * energy * factor
    provenance = ((annual or {}).get("cfd_strike_provenance") or {}).get("by_strike") or []
    nuclear_rows = [row for row in provenance if str(row.get("carrier")) == "nuclear"]
    nuclear_payment = float(sum(float(row.get("payment_gbp") or 0.0) for row in nuclear_rows))
    nuclear_energy = float(sum(float(row.get("dispatch_mwh") or 0.0) for row in nuclear_rows))
    in_bill = nuclear_payment * factor / denominator * mwh if denominator else None
    return {
        "unlabelled_support_by_carrier": list(rows.values()),
        "unlabelled_nuclear_capacity_mw": float(nuclear.get("capacity_mw") or 0.0),
        "unlabelled_nuclear_twh": energy / 1e6,
        "nuclear_cfd_settled_payment_gbp": nuclear_payment,
        "nuclear_cfd_settled_twh": nuclear_energy / 1e6,
        "nuclear_cfd_in_bill_gbp_per_3100kwh": in_bill,
        "nuclear_cfd_hpc_sensitivity_gbp": sensitivity,
        "nuclear_cfd_exclusion_status": exclusion.get("status"),
        "nuclear_cfd_exclusion_note": exclusion.get("note"),
    }


def _price_basis_register(assumptions, ro, source_factor):
    """List every deflator in play so the reader can see which index carried what."""
    strike = assumptions["cfd_strike_indexation"]
    rebasing = assumptions["currency_rebasing"]
    report_year = int(assumptions["benchmark"]["price_year"])
    register = [
        {"quantity": "CfD strike prices (published 2012-price clearing prices to model basis)",
         "index": strike["index"], "from_price_year": int(strike["from_price_year"]),
         "to_price_year": int(strike["to_price_year"]), "factor": float(strike["factor"]),
         "source": "assumptions.cfd_strike_indexation"},
        {"quantity": "Model prices to report basis (wholesale, CfD, BSUoS constraint, BSUoS other services)",
         "index": rebasing["index"], "from_price_year": int(rebasing["from_price_year"]),
         "to_price_year": int(rebasing["to_price_year"]), "factor": float(rebasing["factor"]),
         "source": "assumptions.currency_rebasing"},
    ]
    rpi = ((ro or {}).get("sources") or {}).get("rpi")
    if rpi:
        register.append({
            "quantity": "Notional ROC value (RPI-indexed buy-out plus recycle) to report basis",
            "index": rpi.get("index"), "from_price_year": rpi.get("from_price_year"),
            "to_price_year": rpi.get("to_price_year"), "factor": rpi.get("factor"),
            "source": "ro_scheme_summary.sources.rpi",
            "note": "RPI, not CPIH: the buy-out is RPI-indexed each obligation year, so it is flat in real RPI terms and drifts upward in real CPIH terms."})
    else:
        register.append({
            "quantity": "Notional ROC value to report basis", "index": None, "factor": None,
            "source": "ro_scheme_summary",
            "note": "The RO summary carries no rpi source block; its report_price_year is validated but the index is not recorded here."})
    register.append({
        "quantity": "External allowances and retail procurement reference (source workbook)",
        "index": None, "from_price_year": int(assumptions["source"]["price_year"]),
        "to_price_year": report_year, "factor": float(source_factor),
        "source": "assumptions.source.price_year",
        "note": ("Assumed already on the report basis" if float(source_factor) == 1.0
                 else "Carried with assumptions.source_currency_rebasing")})
    for key, block in (assumptions.get("sourced_externals") or {}).items():
        own = block.get("rebasing")
        if own and int(block.get("price_year", report_year)) != report_year:
            register.append({
                "quantity": f"{block.get('label', key)} (sourced external) to report basis",
                "index": own.get("index"), "from_price_year": int(own["from_price_year"]),
                "to_price_year": int(own["to_price_year"]), "factor": float(own["factor"]),
                "source": f"assumptions.sourced_externals.{key}.rebasing",
                "note": own.get("note")})
    return register


HPC_CAPACITY_BASIS_NOTE = (
    "Hinkley Point C is modelled at 2,317 MW = unit 1 (1,630 MW) plus unit 2 pro-rated 5/12 for its "
    "31 July 2030 TEC effective date (1,630 x 5/12 = 679 MW), so the annual HPC energy already reflects "
    "the staggered start as a capacity average and no support-start gating is applied to its CfD. Source: "
    "data/market/CP30/nuclear_station_schedule_2030.csv."
)


#: F18 (27 Oct 2026): the network a forecast case's TNUoS names (``tnuos_network_delivery.case``) ->
#: the delay case whose NESO thermal constraint cost the constraint line is set beside
#: (``turn_down_allocation.NETWORK_BENCHMARKS``; ``cp30`` is the recommended network, NESO's 3,570 m).
NESO_CONSTRAINT_NETWORKS = {"cp30": None, "no_acceleration": "no_acceleration", "no_upgrades": "no_upgrades"}


def neso_thermal_constraint_benchmark(case: Mapping | None, benchmarks_file=None) -> dict | None:
    """NESO's thermal constraint cost for the case's network, or None (a case that names no CP30 network)."""
    network = ((case or {}).get(TNUOS_NETWORK_DELIVERY_KEY) or {}).get("case")
    if network is None:
        return None
    if network not in NESO_CONSTRAINT_NETWORKS:
        raise ValueError(f"no NESO constraint benchmark for network {network!r}; known: {sorted(NESO_CONSTRAINT_NETWORKS)}")
    from scripts.market.turn_down_allocation import network_benchmark

    out = network_benchmark(NESO_CONSTRAINT_NETWORKS[network], benchmarks_file)
    out["network_case"] = network
    return out


def constraint_line_split(balancing: Mapping, case: Mapping | None, *, household_per_system: float,
                          benchmarks_file=None) -> dict:
    """The constraint line split into thermal redispatch and interconnector re-trading (F18, 27 Oct 2026).

    Re-trading stays in the line (the bill is unchanged); this record shows it apart, so the
    line is set beside NESO's thermal constraint cost (CP30's 3,570 m for the recommended
    network, the CP.31 chart's thermal part; NAO/NESO figures for the delay networks) on both
    readings: the thermal redispatch alone, and with re-trading (NESO's figure carries the
    interconnector counter-trades by inference from CP.18's IC actions). System GBP in the
    model price year; household GBP ex VAT at the line's own rate.
    """
    line = float(balancing["constraint_cost_gbp"])
    retrading = float(balancing.get("interconnector_retrading_cost_gbp") or 0.0)
    in_line = bool(balancing.get("include_interconnector_retrading"))
    inside = retrading if in_line else 0.0
    thermal = line - inside
    rate = float(household_per_system)
    out = {
        "constraint_line_gbp": line,
        "thermal_redispatch_gbp": thermal,
        "interconnector_retrading_gbp": inside,
        "interconnector_retrading_in_constraint_line": in_line,
        "interconnector_retrading_outside_line_gbp": 0.0 if in_line else retrading,
        "constraint_line_gbp_per_3100kwh_ex_vat": line * rate,
        "thermal_redispatch_gbp_per_3100kwh_ex_vat": thermal * rate,
        "interconnector_retrading_gbp_per_3100kwh_ex_vat": inside * rate,
        "note": ("interconnector re-trading stays in the constraint line and is reported apart from thermal "
                 "redispatch; the line's total is unchanged"),
    }
    if balancing.get("institution") == CENTRAL_DISPATCH_INSTITUTION:
        out["note"] += "; a central-dispatch (I3) year has no re-trading, so the whole line is its FTI parts"
        # 27 Oct 2026: FTI's reading of the counter-trade premium beside the headline line (no bill change).
        fti = balancing.get("counter_trade_premium_fti_treatment")
        if fti:
            out["counter_trade_premium_fti_treatment"] = {
                **fti, "constraint_cost_gbp_per_3100kwh_ex_vat": float(fti["constraint_cost_gbp"]) * rate,
                "consumer_saving_on_constraint_line_gbp_per_3100kwh_ex_vat":
                    float(fti["consumer_saving_on_constraint_line_gbp"]) * rate,
                "headline_consumer_saving_on_constraint_line_gbp_per_3100kwh_ex_vat":
                    float(fti["headline_consumer_saving_on_constraint_line_gbp"]) * rate}
    neso = neso_thermal_constraint_benchmark(case, benchmarks_file)
    if neso is not None:
        value = float(neso["value_GBP"])
        out["neso_thermal_constraint_cost"] = {
            **neso,
            "gbp_per_3100kwh_ex_vat": value * rate,
            "thermal_redispatch_less_neso_gbp": thermal - value,
            "constraint_line_less_neso_gbp": line - value,
            "basis": ("NESO's thermal constraints only (voltage and stability outside, as the bill's own "
                      "bsuos_voltage_and_stability_constraints line); interconnector counter-trades inside by "
                      "inference, so the line with re-trading is the like-for-like reading and thermal "
                      "redispatch alone the lower one"),
        }
    return out


def _diagnostics(annual, balancing, assumptions, factor, final_demand, mwh, ex_vat,
                 procurement_bridge_total, ro, denominator=None, source_factor=1.0, pass_through=None,
                 other_services_line=None, household_loss=1.0):
    """Model quantities the report shows but never charges to the household.

    ``final_demand`` is the GSP-rate lines' denominator (F18: the GSP charging volume over
    the household loss factor ``household_loss``; item D1: metered supply, factor one).
    """
    aggregate = annual.get("annual_aggregate") or {}
    bsuos_rule = assumptions["model_linked_rules"]["bsuos_constraints"]
    other = assumptions["sourced_externals"]["bsuos_other_services"]
    procurement = assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]
    comparator = bsuos_rule["neso_outturn_comparator"]
    bridge_reference = float(procurement["reference_total_gbp_per_3100kwh"])
    constraints_per_customer = balancing["constraint_cost_gbp"] * factor / final_demand * mwh
    # FES 2050 decision 65: a derived (per-household) line is read off the bill as charged.
    other_per_customer = (float(other["value"]) * factor * mwh
                          if other.get("unit", "gbp_per_mwh_final_demand") == "gbp_per_mwh_final_demand"
                          else float(other_services_line))
    cfd_payment = float(annual.get("cfd_payment_gbp") or 0.0)
    return {
        # CfD line and its sensitivities.
        "cfd_payment_gbp": cfd_payment,
        "cfd_settled_dispatch_payment_gbp": annual.get("cfd_settled_dispatch_payment_gbp"),
        "cfd_line_gbp_per_3100kwh": cfd_payment * factor / denominator * mwh if denominator else None,
        "cfd_recovery_denominator_mwh": denominator,
        # Denominator sensitivity: the same payment spread over total system
        # demand instead of CfD eligible supplier sales.
        "cfd_line_if_recovered_over_system_demand_gbp_per_3100kwh": (
            cfd_payment * factor / annual["demand_mwh"] * mwh if annual.get("demand_mwh") else None),
        "cfd_reference_price_basis": annual.get("cfd_reference_price_basis"),
        "cfd_economic_curtailment_rule": annual.get("cfd_economic_curtailment_rule"),
        "cfd_economic_curtailment_rounds": annual.get("cfd_economic_curtailment_rounds"),
        "cfd_economic_curtailment_zero_price_threshold_gbp_per_mwh": annual.get(
            "cfd_economic_curtailment_zero_price_threshold_gbp_per_mwh"),
        "cfd_economic_curtailment_payment_gbp": annual.get("cfd_economic_curtailment_payment_gbp"),
        "cfd_economic_curtailment_twh": (annual.get("cfd_economic_curtailment_mwh") or 0.0) / 1e6,
        "cfd_economic_curtailment_gbp_per_3100kwh": (
            float(annual.get("cfd_economic_curtailment_payment_gbp") or 0.0) * factor / denominator * mwh
            if denominator else None),
        "cfd_economic_curtailment_by_strike": annual.get("cfd_economic_curtailment_by_strike"),
        "cfd_economic_curtailment_static_availability_generators": annual.get(
            "cfd_economic_curtailment_static_availability_generators"),
        "price_basis_register": _price_basis_register(assumptions, ro, source_factor),
        "hpc_capacity_basis_note": HPC_CAPACITY_BASIS_NOTE,
        # Interconnector re-trading inside the constraint line (optional keys
        # from the balancing-cost extraction; None when absent).
        "interconnector_retrading_cost_gbp": balancing.get("interconnector_retrading_cost_gbp"),
        "interconnector_retrading_in_constraint_line": balancing.get("include_interconnector_retrading"),
        "interconnector_retrading_gbp_per_3100kwh": (
            float(balancing["interconnector_retrading_cost_gbp"]) * factor / final_demand * mwh
            if balancing.get("interconnector_retrading_cost_gbp") is not None else None),
        "cp30_pathway_curtailment_twh": bsuos_rule.get("cp30_pathway_curtailment_twh"),
        "ro_relief_case": ro["relief_case"],
        "ro_gross_gbp_per_3100kwh": ro["gross_gbp_per_3100kwh_ex_vat"],
        "ro_relief_case_values_gbp_per_3100kwh": ro["relief_cases"],
        "ro_reference_gbp_per_3100kwh": ro["reference_gbp_per_3100kwh_ex_vat"],
        "ro_relief_swing_gbp_per_3100kwh": (
            max(ro["relief_cases"].values()) - min(ro["relief_cases"].values())),
        "ro_roc_volume": ro["roc_volume"],
        "ro_eligible_capacity_mw": ro["eligible_capacity_mw"],
        "ro_expired_capacity_mw": ro["expired_capacity_mw"],
        "ro_method_label": ro["method_label"],
        "procurement_bridge_gbp_per_3100kwh": procurement_bridge_total,
        "procurement_bridge_reference_gbp_per_3100kwh": bridge_reference,
        "procurement_bridge_delta_against_reference_gbp_per_3100kwh": (
            procurement_bridge_total - bridge_reference
        ),
        "procurement_bridge_note": (
            "The decomposed allowances plus domestic shaping replace the single external "
            "residual. They come out below it mainly because the residual also absorbed a forward "
            "premium, now set to zero. The supplier imbalance allowance is its own row (item D4(d)): "
            "cash-out is settled by suppliers through Elexon, not recovered through BSUoS."
        ),
        "bsuos_constraints_gbp_per_3100kwh": constraints_per_customer,
        "bsuos_other_services_gbp_per_3100kwh": other_per_customer,
        "bsuos_modelled_plus_external_gbp_per_3100kwh": constraints_per_customer + other_per_customer,
        "bsuos_reference_gbp_per_3100kwh": float(bsuos_rule["reference_total_gbp_per_3100kwh"]),
        "bsuos_delta_against_reference_gbp_per_3100kwh": (
            constraints_per_customer + other_per_customer
            - float(bsuos_rule["reference_total_gbp_per_3100kwh"])
        ),
        # Constraint cost against NESO's own published constraint outturn, which
        # is the only pinned figure the modelled quantity is directly comparable
        # to. The 91.0 above is a third-party forecast of *total* BSUoS.
        "neso_constraint_outturn_gbp_per_mwh_final_demand": float(
            comparator["constraint_gbp_per_mwh_final_demand"]),
        "neso_constraint_outturn_gbp_per_3100kwh": (
            float(comparator["constraint_gbp_per_mwh_final_demand"]) * factor * mwh * household_loss),
        # Per MWh of the chargeable volume, as NESO's outturn rate is.
        "modelled_constraint_gbp_per_mwh_final_demand": (
            balancing["constraint_cost_gbp"] / (final_demand * household_loss)),
        "neso_total_bsuos_outturn_gbp_per_3100kwh": (
            float(comparator["total_bsuos_gbp_per_mwh_final_demand"]) * factor * mwh * household_loss),
        "neso_comparator_regulatory_year": comparator["regulatory_year"],
        "neso_comparator_note": comparator["note"],
        **_unlabelled_support_diagnostics(balancing, assumptions, factor, mwh, annual, denominator),
        "cfd_curtailment_twh": balancing["cfd_curtailed_mwh"] / 1e6,
        "cfd_curtailment_compensation_gbp": balancing["cfd_curtailment_compensation_gbp"],
        # Review 28 Sep 2026: inside the constraint line, the part of the strike-basis
        # CfD bids above what the turned-down units forgo: a transfer to generators.
        "cfd_strike_basis_premium_over_contract_value_gbp": balancing.get(
            "cfd_strike_basis_premium_over_contract_value_gbp"),
        "cfd_strike_basis_premium_over_contract_value_gbp_per_3100kwh": (
            float(balancing["cfd_strike_basis_premium_over_contract_value_gbp"]) * factor / final_demand * mwh
            if balancing.get("cfd_strike_basis_premium_over_contract_value_gbp") is not None else None),
        "cfd_strike_basis_premium_label": ("transfer to CfD generators inside BSUoS constraint costs "
                                           "(strike bid basis less contract value), not a resource cost"),
        # F18 (27 Oct 2026): the CfD payments made in low-price hours (a report, no bill change).
        "cfd_payments_in_low_price_hours": _with_household(
            annual.get("cfd_payments_in_low_price_hours"), factor / denominator * mwh if denominator else None),
        # F9 evidence: hours bunch just above zero rather than crossing it.
        "hours_price_at_or_below_1_gbp_per_mwh": annual.get("hours_price_at_or_below_1_gbp_per_mwh"),
        "hours_price_at_or_below_zero": annual.get("hours_price_at_or_below_zero"),
        "hours_price_below_zero": annual.get("hours_price_below_zero"),
        "minimum_price_gbp_per_mwh": annual.get("minimum_price_gbp_per_mwh"),
        "negative_price_rule": annual.get("cfd_negative_price_rule"),
        "negative_price_rule_note": assumptions["model_linked_rules"]["cfd"][
            "negative_price_rule_note"],
        "gross_redispatch_mwh": annual.get("gross_redispatch_mwh"),
        "gross_redispatch_share_of_demand": (
            annual["gross_redispatch_mwh"] / annual["demand_mwh"] if annual.get("demand_mwh") else None
        ),
        "feasibility_slack_energy_mwh": balancing["feasibility_slack_energy_mwh"],
        "feasibility_slack_cost_gbp": balancing["feasibility_slack_cost_gbp"],
        "feasibility_slack_note": (
            "Unserved load priced at VOLL. Subtracted from the constraint line (item B8) and "
            "reported as adequacy context for the Capacity Market line, which an energy-only "
            "dispatch model cannot produce."
        ),
        "non_participant_turn_down_cost_at_penalty_gbp": balancing.get("non_participant_turn_down_cost_at_penalty_gbp"),
        "non_participant_turn_down_mwh": balancing.get("non_participant_turn_down_mwh"),
        "non_participant_turn_down_note": (
            "Forced turn-down of units outside the balancing mechanism, priced at the LP's "
            "non-participant penalty bid rather than any real bid; subtracted from the constraint "
            "line (item B8)."
        ),
        "interconnector_congestion_rent_gbp": aggregate.get("interconnector_congestion_rent_GBP"),
        "interconnector_congestion_rent_note": (
            "Whole-cable Stage 1 congestion rent. The part the regulated regimes return to GB "
            "consumers (above-cap revenue less floor top-ups for cap-and-floor cables, the tariff "
            "share for IFA) is billed as interconnector_cap_and_floor_pass_through, a TNUoS credit; "
            "exempt and non-GB cables keep theirs. Interconnector re-trading, a NESO balancing trade, "
            "stays in the constraint line and does not overlap this."
            if pass_through is not None else
            "Cap-and-floor pass-through disabled: recorded as a diagnostic only."
        ),
        "interconnector_cap_and_floor_pass_through": pass_through,
        "interconnector_cap_and_floor_credit_gbp_per_3100kwh": (
            pass_through["credit_to_consumers_gbp"] * factor * pass_through["household_share_of_tnuos_residual"]
            if pass_through is not None else None),
        "wholesale_curtailment_order": annual.get("wholesale_curtailment_order"),
        "hours_price_below_minus_0_01_gbp_per_mwh": annual.get("hours_price_below_minus_0_01_gbp_per_mwh"),
        "balancing_negative_price_rule": balancing.get("balancing_negative_price_rule"),
        "max_congested_lines": aggregate.get("max_congested_lines"),
        "max_congested_transformers": aggregate.get("max_congested_transformers"),
        "domestic_shaping_gbp_per_mwh": annual.get("domestic_shaping_gbp_per_mwh"),
        "ex_vat_gbp": ex_vat,
    }


def render_projection(frame, summary):
    # The headline grouping is Ofgem's price-cap categories at the current cap release;
    # the functional view (what each cost pays for) is shown beside it.
    groupings = summary["groupings"]
    def group_table(view):
        totals = groupings[view]["totals_gbp_per_3100kwh_ex_vat"]
        table = pd.DataFrame({"group": list(totals), "annual_gbp_per_3100kwh_ex_vat": list(totals.values())})
        table = table.loc[table.annual_gbp_per_3100kwh_ex_vat.abs() > 0.0005].reset_index(drop=True)
        table.loc[len(table)] = ["VAT", summary["vat_gbp"]]
        return table
    groups, functional = group_table("ofgem"), group_table("functional")
    notes = "".join(f"<li>{html.escape(item)}</li>" for item in summary["limitations"])
    order = list(groupings["ofgem"]["totals_gbp_per_3100kwh_ex_vat"])
    details = "".join(f'<details open><summary>{html.escape(group)}</summary>{rows[["component", "annual_gbp_per_3100kwh_ex_vat", "source_class", "method", "source"]].to_html(index=False, escape=True)}</details>'
                      for group, rows in sorted(frame.groupby("ofgem_group"), key=lambda item: order.index(item[0])))
    diagnostics = pd.DataFrame(
        [{"diagnostic": key, "value": value} for key, value in summary["diagnostics"].items()]
    )
    strikes = pd.DataFrame((summary.get("cfd_strike_provenance") or {}).get("by_strike", []))
    curtailment = pd.DataFrame(summary.get("cfd_economic_curtailment_by_strike") or [])
    pass_through = pd.DataFrame(
        ((summary["diagnostics"].get("interconnector_cap_and_floor_pass_through") or {}).get("by_cable")) or [])
    share = summary.get("pypsa_derived_share_of_ex_vat")
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Future electricity bill sensitivity</title>
<style>body{{font:15px system-ui;color:#182838;margin:2rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{padding:.6rem;border:1px solid #ddd;text-align:left}}th{{background:#edf3f8}}details{{overflow:auto;margin:1rem 0}}summary{{font-size:1.2rem;font-weight:bold}}.note{{background:#fff4d9;padding:1rem}}.kpi{{background:#edf3f8;padding:1rem;margin:1rem 0}}</style>
<h1>{html.escape(summary['scenario'])}: future bill sensitivity</h1>
<h2>GBP {summary['total_gbp_inc_vat']:,.2f}/year including VAT</h2>
<p>{summary['modelled_year']} | {summary['annual_consumption_kwh']:,.0f} kWh/year | {html.escape(summary['currency'])}</p>
<div class="kpi"><strong>PyPSA-derived share of the ex-VAT bill:
GBP {summary['pypsa_derived_gbp_per_3100kwh_ex_vat']:,.2f} of GBP {summary['ex_vat_gbp']:,.2f}
({share:.1%})</strong>. The remainder is explicit external assumption.</div>
<div class="note">Scenario sensitivity; forecast validation and statutory cap reproduction are false.<ul>{notes}</ul></div>
<h2>Main categories: {html.escape(groupings['ofgem']['title'])}</h2>{groups.to_html(index=False, float_format=lambda x: f'{x:,.2f}')}
<h2>{html.escape(groupings['functional']['title'])}</h2>{functional.to_html(index=False, float_format=lambda x: f'{x:,.2f}')}
<h2>Every component, including zero allowances and negative credits</h2>{details}
<h2>Diagnostics: model quantities reported but not charged</h2>{diagnostics.to_html(index=False, escape=True)}
<h2>CfD strike provenance</h2>{strikes.to_html(index=False, escape=True) if not strikes.empty else "<p>No CfD contracts settled.</p>"}
<h2>CfD economic-curtailment payments ({html.escape(str(summary.get('cfd_economic_curtailment_rule')))})</h2>{curtailment.to_html(index=False, escape=True) if not curtailment.empty else "<p>No economic-curtailment payment added.</p>"}
<h2>Interconnector congestion rent returned through TNUoS</h2>{pass_through.to_html(index=False, escape=True) if not pass_through.empty else "<p>No interconnector pass-through computed.</p>"}
<h2>Model evidence</h2>{pd.DataFrame(summary.get('sources', [])).to_html(index=False, escape=True)}
</html>'''


def read_ldes_topup(path, case_id="private"):
    """The LDES floor top-up one case of a public account carries (item D4(c)).

    The public account (``{run_group}_public_account_summary.json``) is built from
    the finance register, whose LDES market revenue is the model's; the forecast
    bill is the private case, so it reads ``cases.private``. A case whose account
    has the floor switched off carries a zero line.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = document.get("cases") or {}
    if case_id not in cases:
        raise ValueError(f"{path} has no case {case_id!r} to read the LDES floor top-up from")
    row = cases[case_id]
    enabled = bool(row.get("ldes_cap_and_floor_enabled"))
    record = {
        "case_id": case_id, "enabled": enabled,
        "topup_gbp": float(row.get("ldes_floor_topup_gbp", 0.0)) if enabled else 0.0,
        "topup_at_wacc_gbp": row.get("ldes_floor_topup_at_wacc_gbp"),
        "floor_basis": row.get("ldes_floor_basis"), "floor_real_rate": row.get("ldes_floor_real_rate"),
        "source": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    if "price_level" in document:
        # Item F12 (12 Oct 2026): the account of a price-level twin (a model account carries no key).
        record["price_level"] = str(document["price_level"])
    return record


def price_level_account_record(annual, assumptions, ldes=None):
    """``annual`` with the account-built lines placed by the price level of the account they read (item F12).

    Under ``cp32`` the LDES floor top-up and the register-booked top-up lines
    (``sourced_externals.<line>.derivation.cost_from``) come from public accounts. An
    account a price-level twin built (12 Oct 2026; its summary says ``price_level``) is
    on the finance register settled at that level, so the line is re-priced; an account
    at the model's price is held, and said so. Returns ``annual`` itself at ``model``.
    """
    level = annual.get("price_level", DEFAULT_PRICE_LEVEL)
    if level == DEFAULT_PRICE_LEVEL:
        return annual
    held = dict(annual.get("price_level_held_at_model_price") or {})
    repriced = dict(annual.get("price_level_repriced_lines") or {})
    if ldes is not None and ldes.get("enabled", True):
        if ldes.get("price_level", DEFAULT_PRICE_LEVEL) == level:
            held.pop("ldes_cap_and_floor", None)
            repriced["ldes_cap_and_floor"] = (
                f"the floor top-up from a public account built on the finance register settled at {level} "
                f"({ldes.get('source', 'the caller')})")
    for key, block in (assumptions.get("sourced_externals") or {}).items():
        cost_from = ((block or {}).get("derivation") or {}).get("cost_from")
        if not cost_from:
            continue
        path, account_level = register_booked_account(key, cost_from)
        if account_level == level:
            repriced[key] = f"the register-booked top-up from {path}, a public account settled at {level}"
        else:
            held[key] = f"the register-booked top-up from {path}, a public account at the model's price"
    return {**annual, "price_level_held_at_model_price": held, "price_level_repriced_lines": repriced}


def retrading_at_price_level(summaries, balancing):
    """Item F24.2: the constraint line's interconnector re-trading re-valued at the case's price level.

    A month's re-trading (``accounting.interconnector_retrading_cost_GBP`` of its market summary) is
    the BM cable schedules' trade value less the Stage 1 schedules', both at the IC1 foreign asks and
    bids with the volume tranches (``market_coupling.interconnector_trade_cost``), plus the
    counter-trade premium: a flat GBP/MWh moved and the foreign-plant uplift, a share of the move's
    foreign marginal value (``counter_trade_premium_cost``). No part has the GB price in it: NESO pays
    for the foreign energy a move buys or gives up, and the GB side of the move is priced by the GB
    units' own bids and offers. The price level re-prices the GB price and holds the foreign side as
    solved, as the cables' rent does (:func:`interconnector_rent_change`), so every part keeps its
    value. The re-trading at the level is rebuilt from its parts, each month checked against its
    total, so a re-trading carrying anything else is refused rather than held silently. A summary
    from before the parts were written keeps its total, and is listed.

    Returns ``{model_gbp, at_price_level_gbp, change_gbp, components_gbp, months_without_components}``.
    """
    model, at_level = 0.0, 0.0
    parts = dict.fromkeys(RETRADING_COMPONENTS, 0.0)
    without = []
    for path in summaries:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        name = str(document.get("scenario", path))
        total = (document.get("accounting") or {}).get("interconnector_retrading_cost_GBP")
        total = 0.0 if total is None else _finite(total, f"{name} interconnector re-trading")
        block = document.get("interconnectors") or {}
        model += total
        if not any(block.get(key) is not None for key in RETRADING_COMPONENTS):
            if total:
                without.append(name)
            at_level += total
            continue
        values = {key: _finite(block.get(key) or 0.0, f"{name} {key}") for key in RETRADING_COMPONENTS}
        rebuilt = sum(sign * values[key] for key, sign in RETRADING_COMPONENTS.items())
        if abs(rebuilt - total) > RETRADING_COMPONENT_TOLERANCE_GBP:
            raise ValueError(
                f"{name}: interconnector re-trading GBP{total:,.0f} is not its parts (BM trade value - Stage 1 "
                f"trade value + counter-trade premium = GBP{rebuilt:,.0f}), so it cannot be re-valued at the "
                "price level (item F24.2)")
        for key in parts:
            parts[key] += values[key]
        # Every part is on the foreign side, held as solved at the price level.
        at_level += rebuilt
    extracted = (balancing or {}).get("interconnector_retrading_cost_gbp")
    if extracted is not None and abs(float(extracted) - model) > RETRADING_COMPONENT_TOLERANCE_GBP * max(1, len(summaries)):
        raise ValueError(f"the market summaries' re-trading (GBP{model:,.0f}) is not the balancing-cost "
                         f"extraction's (GBP{float(extracted):,.0f}): different months?")
    return {"model_gbp": model, "at_price_level_gbp": at_level, "change_gbp": at_level - model,
            "components_gbp": parts, "months_without_components": without}


def price_level_balancing_record(annual, balancing, summaries):
    """``(annual, balancing)`` with the constraint line placed by the price level (item F24.2, 24 Oct 2026).

    Under ``cp32``:

    * the CfD contract-value bids are on the mapped reference already: the balancing-cost
      extraction ran at the case's price level (``cfd_bid_price_level_change_gbp``), and the
      twin's asset settlement moves each unit's balancing settlement by the same amount. An
      extraction at another level, or mapped on another year, is refused, since the bill
      and the register would then price one bid two ways;
    * the interconnector re-trading inside the line is re-valued at the level
      (:func:`retrading_at_price_level`; valued on the foreign side, which the level holds,
      so it keeps its model value) and the line takes the change;
    * the other accepted offers and bids stay held (:data:`CP32_HELD_LINES`).

    A central-dispatch year has no bids or re-trading: its line is held whole. Returns both
    unchanged at ``model`` (after refusing a price-level extraction there).
    """
    level = annual.get("price_level", DEFAULT_PRICE_LEVEL)
    extracted_level = (balancing or {}).get("price_level", DEFAULT_PRICE_LEVEL)
    is_central = (balancing or {}).get("institution") == CENTRAL_DISPATCH_INSTITUTION
    if balancing is not None and not is_central and extracted_level != level:
        raise ValueError(
            f"the bill is at price level {level!r} but its balancing-cost extraction ({balancing.get('path')}) is at "
            f"{extracted_level!r}: the CfD contract-value bids in the constraint line must be priced at the bill's "
            "level (item F24.2). Re-run extract_future_balancing_costs for this case.")
    if level == DEFAULT_PRICE_LEVEL or balancing is None:
        return annual, balancing
    held = dict(annual.get("price_level_held_at_model_price") or {})
    repriced = dict(annual.get("price_level_repriced_lines") or {})
    if is_central:
        held["bsuos_constraints"] = ("held as solved: a central-dispatch year's line (cost recovery, firm access, "
                                     "make-whole, the borrowed forecast-error charge) at the model's price; it has "
                                     "no interconnector re-trading")
        return {**annual, "price_level_held_at_model_price": held}, balancing
    _check_price_level_mapping(annual.get("price_level_mapping") or {}, balancing)
    change = float(balancing.get("cfd_bid_price_level_change_gbp", 0.0))
    repriced["bsuos_constraints (CfD contract-value bids)"] = (
        f"re-priced at {level} (item F24.2): each contract-value unit's bid on its turn-down is the foregone "
        "difference payment, strike less the mapped reference (bid basis, floor and negative-price rule as the "
        "bids applied; the state-owned share bids its public price and does not move); volumes as solved; "
        f"change GBP{change / 1e6:,.2f}M, matched unit for unit in the twin's asset settlement")
    record = retrading_at_price_level(summaries, balancing)
    record["in_constraint_line"] = bool(balancing.get("include_interconnector_retrading"))
    if record["in_constraint_line"]:
        balancing = {**balancing,
                     "constraint_cost_gbp": float(balancing["constraint_cost_gbp"]) + record["change_gbp"],
                     "interconnector_retrading_cost_gbp": record["at_price_level_gbp"],
                     "interconnector_retrading_model_price_gbp": record["model_gbp"]}
        repriced["bsuos_constraints (interconnector re-trading)"] = (
            f"re-valued at {level} (item F24.2): BM less Stage 1 trade value at the IC1 foreign asks and bids, plus "
            "the counter-trade premium and foreign-plant uplift, all on the foreign side, which the price level "
            f"holds as solved as it does for the cables' rent; change GBP{record['change_gbp'] / 1e6:,.2f}M on "
            f"GBP{record['model_gbp'] / 1e6:,.0f}M")
    if record["months_without_components"]:
        held["bsuos_constraints (interconnector re-trading, part)"] = (
            f"months {record['months_without_components']} have market summaries without the re-trading's parts, "
            "so their re-trading is their total at the model's price")
    return ({**annual, "price_level_held_at_model_price": held, "price_level_repriced_lines": repriced,
             "interconnector_retrading_at_price_level": record}, balancing)


#: Relative tolerance on the mapped year's levels the bill and its extraction each report (item F24.2).
PRICE_LEVEL_MAPPING_CHECK_RTOL = 1e-9


def _check_price_level_mapping(mapping, balancing):
    """Refuse an extraction whose mapped year is not the bill's own (item F24.2).

    Both map the same months the same way (``map_year_to_price_level``), so the model and
    mapped levels each reports, time- and demand-weighted, must agree.
    """
    check = balancing.get("price_level_mapping_check") or {}
    for side in ("model", "mapped"):
        for key in ("time_weighted_gbp_per_mwh", "demand_weighted_gbp_per_mwh"):
            ours = (mapping.get(side) or {}).get(key)
            theirs = (check.get(side) or {}).get(key)
            if ours is None or theirs is None or not np.isclose(float(ours), float(theirs),
                                                                rtol=PRICE_LEVEL_MAPPING_CHECK_RTOL, atol=1e-9):
                raise ValueError(
                    f"the balancing-cost extraction ({balancing.get('path')}) mapped another year: {side} {key} "
                    f"{theirs} against the bill's {ours} (item F24.2)")


def contract_book_path(cfd_rule):
    """The 2050 contract book a bill's CfD rule names (``model_linked_rules.cfd.contract_book``), or None (2030)."""
    path = (cfd_rule or {}).get("contract_book")
    if not path:
        return None
    full = Path(path) if Path(path).is_absolute() else Path(__file__).resolve().parents[2] / path
    if not full.exists():
        raise FileNotFoundError(f"model_linked_rules.cfd.contract_book {path} does not exist")
    return str(full)


def sibling_link_schedules(networks):
    """The Stage 1 link schedule written beside each wholesale network by the same solve."""
    paths = []
    for network in networks:
        text = str(network)
        if not text.endswith("_wholesale.nc"):
            raise ValueError(f"Cannot locate the Stage 1 link schedule for {network}; pass wholesale_links")
        paths.append(text[: -len("_wholesale.nc")] + "_wholesale_links.csv")
    return paths


def _settlement_key(table, scenarios, networks, prices, physical_networks, cfd_rule, wholesale_links=None):
    """What the settlement depends on: the strikes and the files, never the case label."""
    strikes = tuple(sorted((str(k[0]), str(k[1]), float(v)) for k, v in table.strikes.items()))
    return (strikes, tuple(scenarios), tuple(map(str, networks)), tuple(map(str, prices)),
            tuple(map(str, physical_networks)), json.dumps(cfd_rule, sort_keys=True, default=str),
            tuple(map(str, wholesale_links or ())))


def run_projection(*args, register_line_accounts=None, **kwargs):
    """Settle the case and return ``(hourly, components, summary)`` without writing (:func:`_run_projection`).

    ``register_line_accounts`` (``{sourced_externals line: public account}``; item F12,
    12 Oct 2026) reads those register-booked lines from another account, as
    :class:`register_lines_from` does: a price-level bill case reads the top-ups its
    level's twin run group sized. Empty or None, nothing changes.
    """
    if not register_line_accounts:
        return _run_projection(*args, **kwargs)
    with register_lines_from(register_line_accounts):
        return _run_projection(*args, **kwargs)


def _run_projection(case, assumptions_path, networks, prices, summaries, physical_networks=None,
                    annual_summary=None, balancing_costs=None, recovery_demand=None, strike_table=None,
                    ro_scheme_costs=None, strike_table_object=None, extra_components=None,
                    settlement_cache=None, wholesale_links=None, ldes_public_account=None,
                    ldes_topup_gbp=None, supplier_margin_base=None, ldes_account_price_level=None):
    """Settle the case and return ``(hourly, components, summary)`` without writing.

    ``wholesale_links`` are the Stage 1 link schedules (electrolyser draw, item
    D1); by default the ones the same solve wrote beside each wholesale network.
    The LDES line (item D4(c)) takes ``ldes_topup_gbp`` when a caller settles a
    case of its own (the ownership projector), else the private case of
    ``ldes_public_account``.

    ``strike_table_object`` lets a caller settle the same run against an
    already-built :class:`StrikeTable` (the public-ownership bill link feeds
    re-priced strikes through here so every other rule of the projector is
    shared with the private case). It must carry the case's model price year.

    ``supplier_margin_base`` (F18) is the cost base Ofgem's EBIT and headroom
    percentages apply to (:func:`apply_supplier_margin`); the ownership projector
    passes the private case's, so the supplier line does not move between cases.

    ``ldes_account_price_level`` (item F12) is the price level of the account a
    caller-supplied ``ldes_topup_gbp`` came from; it only labels the line in a
    ``cp32`` summary (:func:`price_level_account_record`).

    ``settlement_cache`` (a dict the caller keeps across calls) memoises the
    settled year by strike table: two cases whose strikes agree settle to the
    same hourly series and CfD payment, and differ only in the bill rows the
    caller adds, so the year is settled once for them.
    """
    if physical_networks is None:
        raise ValueError("Future bill output requires physical networks to verify CfD payments")
    assumptions = json.loads(Path(assumptions_path).read_text(encoding="utf-8"))
    # Network bounds (18-19 Oct 2026): the case's network-delivery TNUoS (unchanged without one).
    assumptions = apply_tnuos_network_delivery(assumptions, case)
    scenarios = case["source_scenarios"]
    year = int(case["modelled_year"])
    if strike_table_object is not None:
        table = strike_table_object
        if int(getattr(table, "price_year", -1)) != int(case["model_price_year"]):
            raise ValueError("The supplied strike table is not on the case's model price year")
    else:
        table = load_strike_table(
            strike_table or assumptions["model_linked_rules"]["cfd"]["strike_table"],
            assumptions["cfd_strike_indexation"], int(case["model_price_year"]),
            overrides=strike_overrides(assumptions),
        )
    profile = load_domestic_profile(
        assumptions["model_linked_rules"]["direct_fuel"]["domestic_shaping"]["profile_file"]
    )
    aggregate = read_annual_aggregate(annual_summary, scenarios, year)
    cfd_rule = assumptions["model_linked_rules"]["cfd"]
    if wholesale_links is None:
        wholesale_links = sibling_link_schedules(networks)
    # Item F12: the price level the year settles on (``model`` unless the case says ``cp32``).
    price_level = price_level_of(case)
    price_level_mapping = case.get("price_level_mapping") if price_level == "cp32" else None
    key = (_settlement_key(table, scenarios, networks, prices, physical_networks, cfd_rule, wholesale_links)
           + (price_level, json.dumps(price_level_mapping, sort_keys=True, default=str))
           if settlement_cache is not None else None)
    if key is not None and key in settlement_cache:
        hourly, annual = settlement_cache[key]
        hourly, annual = hourly.copy(), copy.deepcopy(annual)
    else:
        hourly, annual = aggregate_inputs(
            scenarios, networks, prices, summaries, year, physical_networks, table, profile,
            cfd_rule["negative_price_rule"], aggregate,
            reference_price_basis=cfd_rule.get("reference_price_basis", {}),
            economic_curtailment={
                "rule": cfd_rule.get("economic_curtailment_rule", "none"),
                "pre_ar4_rounds": cfd_rule.get("pre_ar4_rounds"),
                "zero_price_threshold_gbp_per_mwh": cfd_rule.get(
                    "zero_price_threshold_gbp_per_mwh", DEFAULT_ZERO_PRICE_THRESHOLD),
            },
            volume_caps={str(k): float(v["annual_mwh"]) for k, v in (cfd_rule.get("volume_collars") or {}).items()},
            wholesale_links=wholesale_links, contract_book=contract_book_path(cfd_rule),
            price_level=price_level, price_level_mapping=price_level_mapping,
        )
        if key is not None:
            settlement_cache[key] = (hourly.copy(), copy.deepcopy(annual))
    balancing = read_balancing_costs(balancing_costs, scenarios, int(case["model_price_year"]))
    # The CfD line settles each round on the bill's negative pricing rule; the
    # balancing bids (and so the constraint cost) applied the rule recorded in
    # their audit. Two rules would price the same contract two ways.
    check_negative_price_rule(cfd_rule["negative_price_rule"], balancing.get("balancing_negative_price_rule"),
                              where=str(balancing_costs))
    recovery = read_recovery_denominators(recovery_demand, recovery_case_id(case), annual["demand_mwh"],
                                          volumes=annual.get("demand_volumes_mwh"))
    ro = read_ro_scheme_cost(
        ro_scheme_costs, case["ro_relief_case"], int(assumptions["benchmark"]["price_year"])
    )
    if ldes_topup_gbp is not None:
        ldes = {"case_id": "supplied by caller", "enabled": True, "topup_gbp": float(ldes_topup_gbp),
                "source": "ldes_topup_gbp supplied by the caller (the ownership projector's public account)"}
        if ldes_account_price_level is not None and ldes_account_price_level != DEFAULT_PRICE_LEVEL:
            ldes["price_level"] = str(ldes_account_price_level)
    elif ldes_public_account:
        ldes = read_ldes_topup(ldes_public_account, "private")
    else:
        ldes = None
    # The scarcity mark-up the Stage 1 prices carry, for the report only (review
    # 28 Sep 2026); annual is copied so a shared settlement cache is not touched.
    annual = {**annual, "scarcity_markup": read_scarcity_summary(case["case_id"])}
    # Item F12: under cp32, the account-built lines are re-priced or held by their account's level.
    annual = price_level_account_record(annual, assumptions, ldes)
    # Item F24.2: under cp32, the constraint line's interconnector re-trading is re-valued at the level.
    annual, balancing = price_level_balancing_record(annual, balancing, summaries)
    from scripts.public_ownership.contract_book_2050 import read_dpa_line

    components, summary = project_components(annual, assumptions, case, balancing, recovery, ro,
                                             extra_components=extra_components, ldes=ldes,
                                             dpa=read_dpa_line(assumptions["model_linked_rules"].get("dpa")),
                                             supplier_margin_base=supplier_margin_base)
    summary["assumptions_file"] = str(assumptions_path)
    summary["assumptions_sha256"] = hashlib.sha256(Path(assumptions_path).read_bytes()).hexdigest()
    summary["strike_table"] = table.as_records()
    if getattr(table, "adjustment_label", None) is not None:
        summary["strike_table_adjustment"] = {
            "label": table.adjustment_label,
            "adjusted_entries": [
                {"cfd_round": key[0], "technology": key[1], "delta_gbp_per_mwh": float(value)}
                for key, value in sorted(table.adjustments.items())
            ],
        }
    return hourly, components, summary


def write_projection(case, assumptions_path, networks, prices, summaries, output_csv, output_json,
                     output_html, hourly_csv, physical_networks=None, annual_summary=None,
                     balancing_costs=None, recovery_demand=None, strike_table=None,
                     ro_scheme_costs=None, wholesale_links=None, ldes_public_account=None,
                     register_line_accounts=None):
    hourly, components, summary = run_projection(
        case, assumptions_path, networks, prices, summaries, physical_networks, annual_summary,
        balancing_costs, recovery_demand, strike_table, ro_scheme_costs,
        wholesale_links=wholesale_links, ldes_public_account=ldes_public_account,
        register_line_accounts=register_line_accounts,
    )
    for path in (output_csv, output_json, output_html, hourly_csv):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_csv, index=False)
    hourly.to_csv(hourly_csv, index_label="snapshot")
    Path(output_json).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    Path(output_html).write_text(render_projection(components, summary), encoding="utf-8")


if "snakemake" in globals():
    write_projection(snakemake.params.case, snakemake.input.assumptions,
        list(snakemake.input.networks), list(snakemake.input.prices), list(snakemake.input.summaries),
        snakemake.output.components, snakemake.output.summary, snakemake.output.report, snakemake.output.hourly,
        list(snakemake.input.physical_networks), snakemake.input.annual_summary,
        snakemake.input.balancing_costs, snakemake.input.recovery_demand,
        snakemake.input.strike_table, snakemake.input.ro_scheme_costs,
        wholesale_links=list(snakemake.input.wholesale_links),
        ldes_public_account=(list(snakemake.input.ldes_public_account) or [None])[0],
        # Item F12 (12 Oct 2026): a price-level case's register-booked lines from its level's twin account.
        register_line_accounts=dict(getattr(snakemake.params, "register_line_accounts", None) or {}))
