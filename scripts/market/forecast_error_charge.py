"""Carry FTI's forecast-error cost onto this system, because our model produces none.

Why
---
FTI's REMA scheduling and dispatch assessment for NESO (February 2026) charges central
dispatch for resolving day-ahead forecast error in real time, and assumes conservatively
that no intra-day market helps: GBP 7.9 bn over 2030-2040 in their core scenario, 2022
prices. Their headline benefit of GBP 8.2 bn is *net* of it. This model produces
essentially no such cost, because any thermal unit can move from nothing to full output
within the hour at marginal cost and interconnectors absorb the error as a cheap reserve. Our raw result therefore corresponds to FTI's *gross*
figure, and a deduction is needed to state an FTI-comparable one.

Rather than try to make the model generate the cost, this borrows FTI's and scales it.
The FTI settlement (:mod:`scripts.market.fti_settlement`) applies it per window through
:func:`charge_for_period_gbp_m` and reports the gross saving, the charge and the
FTI-comparable net saving, the net being the headline (user decision, 26 Sep 2026).

Bases (config ``public_ownership.institution_comparison.forecast_error_charge.basis``)
-----------------------------------------------------------------------------------
``renewable_capacity`` (**default**): forecast error is driven by how much variable
capacity there is, so this is the right form. FTI say their updated scenario has "a higher
magnitude of RES forecast errors (driven by a ~10% larger renewable build-out)", and that
updated scenario runs on CP30 Further Flex and Renewables, our own background; so our fleet
is about 1.10 times their core fleet in 2030. Checkable on the dominant technology: FES 2022
Leading the Way reaches 50 GW of offshore wind by 2030 and ours is 50.65 GW. Spread over a
window by its share of the year's hours.

User decision (11 Nov 2026): FTI's GBP 7.9 bn covers 2030-2040, so its annual amount belongs to
their fleet across that decade, not to their 2030 fleet. ``fleet_ratio`` is therefore our fleet
against FTI's *discount-weighted 2030-2040* fleet: the 2030-fleet ratio divided by
``FTI_FLEET_GROWTH``, the growth of FES 2022 Leading the Way's wind and solar output over the
decade at the same 3.5% (``data/market/fti_core_fleet_fes2022_ltw.csv``; 1.354). 2030: 1.10 /
1.354 = 0.812; a later year's ratio is 1.10 x its ES1 wind and solar output / 273.0 TWh / 1.354.

``demand``: FTI's annual cost over roughly 300 TWh of British demand, applied per MWh to
ours on the same discounting reading and price year. Simpler but wrong in form, since
demand does not drive forecast error. (Until 26 Sep 2026 this was a fixed 2.40 GBP/MWh,
which is the *simple-total* reading in *2022* prices, so it disagreed with the default
reading on both counts.)

``gross_share``: forecast error is 49% of FTI's gross benefit; apply that share to ours.
Self-scaling, but it ties the cost to how much congestion the reform relieves, which is a
different thing again. Never negative.

Reading of the GBP 7.9 bn
-------------------------
FTI's summary does not say whether it is discounted. It is quoted to show what intra-day
trading could recover from an GBP 8.2 bn NPV, and only an NPV can be set against an NPV, so
the default reads it as one: an equivalent annual amount over eleven years at the Green Book
3.5%. ``as_simple_total`` divides by eleven instead and gives about 18% less.

Price year
----------
Every settlement figure is PyPSA money, which the consumer bill treats as **2024 prices**
and rebases to its 2025 report year with the CPIH factor (``currency_rebasing.factor``,
1.0384). The charge is therefore stated in 2024 prices by default, using FTI's own
restatement of GBP 8.2 bn (2022) as GBP 8.8 bn (2024). Before 26 Sep 2026 it was stated
in 2025 prices (a further round 2.5%) and subtracted from 2024-price savings.

Per household
-------------
House rule 8: never divide by 28 million households. :func:`gbp_per_household` converts
system GBP the way the bill recovers a balancing cost, over final demand, rebased to the
report year and times the 3,100 kWh benchmark, read from the bill's own summary.

The result is borrowed, not derived. It inherits FTI's assumptions about flexibility and
scarcity, and linear proportionality to capacity is conservative in the wrong direction,
since larger errors push further up the merit order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

# FTI core scenario, 2022 prices, 2030-2040
FTI_FORECAST_ERROR_BN_2022 = 7.9
FTI_NET_BENEFIT_BN_2022 = 8.2
# The British demand FTI's cost is spread over for the demand basis
FTI_GB_DEMAND_TWH = 300.0
YEARS = 11
DISCOUNT_RATE = 0.035
# FTI restate GBP 8.2 bn (2022) as GBP 8.8 bn (2024).
PRICE_FACTOR_2022_TO_2024 = 8.8 / 8.2
# The bill's model-to-report rebasing, ONS CPIH 2024 -> 2025
# (2030_pypsa_informed_assumptions.json currency_rebasing.factor).
PRICE_FACTOR_2024_TO_2025 = 1.038374717833
PRICE_FACTORS = {2022: 1.0, 2024: PRICE_FACTOR_2022_TO_2024, 2025: PRICE_FACTOR_2022_TO_2024 * PRICE_FACTOR_2024_TO_2025}
#: PyPSA money, the basis of every settlement figure.
MODEL_PRICE_YEAR = 2024
# Our 2030 fleet against FTI's core 2030 fleet, from their "~10% larger renewable build-out" remark
FLEET_RATIO_2030_FLEET = 1.10
#: FES 2022 Leading the Way's wind and solar output, discount-weighted over 2030-2040 at 3.5%, over its 2030 output
#: (data/market/fti_core_fleet_fes2022_ltw.csv; user decision, 11 Nov 2026). A test re-derives it from the table.
FTI_FLEET_GROWTH = 1.3541
FTI_CORE_FLEET_TABLE = "data/market/fti_core_fleet_fes2022_ltw.csv"
# Our fleet against FTI's discount-weighted 2030-2040 fleet (the charge's own fleet).
FLEET_RATIO = round(FLEET_RATIO_2030_FLEET / FTI_FLEET_GROWTH, 3)
YEAR_HOURS = 8760.0

BASES = ("renewable_capacity", "demand", "gross_share")
DEFAULT_BASIS = "renewable_capacity"
#: Defaults of ``public_ownership.institution_comparison.forecast_error_charge``; they
#: must match config/defaults/ when the block is added there.
DEFAULT_SETTINGS = {"enabled": True, "basis": DEFAULT_BASIS, "as_simple_total": False,
                    "fleet_ratio": FLEET_RATIO, "price_year": MODEL_PRICE_YEAR}


def fti_fleet_growth(table=FTI_CORE_FLEET_TABLE, rate: float = DISCOUNT_RATE) -> float:
    """FTI's discount-weighted 2030-2040 wind and solar output over its 2030 output, from the table."""
    import pandas as pd

    path = Path(table)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[2] / path
    frame = pd.read_csv(path, comment="#").sort_values("year")
    output = frame["wind_solar_twh"].to_numpy(dtype=float)
    weights = (1.0 + rate) ** -(frame["year"].to_numpy(dtype=float) - float(frame["year"].iloc[0]))
    return float((output * weights).sum() / weights.sum() / output[0])


def annuity_factor(years: int = YEARS, rate: float = DISCOUNT_RATE) -> float:
    return (1 - (1 + rate) ** -years) / rate


def _price_factor(price_year: int) -> float:
    if int(price_year) not in PRICE_FACTORS:
        raise ValueError(f"price_year must be one of {sorted(PRICE_FACTORS)}, got {price_year!r}")
    return PRICE_FACTORS[int(price_year)]


def fti_cost_per_year_gbp_bn(as_simple_total: bool = False, price_year: int = MODEL_PRICE_YEAR) -> float:
    """FTI's core forecast-error cost as an annual amount, their fleet, in ``price_year`` prices."""
    total = FTI_FORECAST_ERROR_BN_2022 * _price_factor(price_year)
    return total / (YEARS if as_simple_total else annuity_factor())


def forecast_error_charge_gbp_bn(
    gross_saving_gbp_bn: float,
    demand_twh: float,
    basis: str = DEFAULT_BASIS,
    as_simple_total: bool = False,
    fleet_ratio: float = FLEET_RATIO,
    price_year: int = MODEL_PRICE_YEAR,
) -> float:
    """Annual deduction to turn a gross saving into an FTI-comparable one."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    if basis == "renewable_capacity":
        return fti_cost_per_year_gbp_bn(as_simple_total, price_year) * fleet_ratio
    if basis == "demand":
        return fti_cost_per_year_gbp_bn(as_simple_total, price_year) / FTI_GB_DEMAND_TWH * demand_twh
    share = FTI_FORECAST_ERROR_BN_2022 / (FTI_NET_BENEFIT_BN_2022 + FTI_FORECAST_ERROR_BN_2022)
    return share * max(0.0, gross_saving_gbp_bn)


def charge_for_period_gbp_m(
    gross_saving_gbp_m: float,
    demand_twh: float,
    hours: float,
    *,
    basis: str = DEFAULT_BASIS,
    as_simple_total: bool = False,
    fleet_ratio: float = FLEET_RATIO,
    price_year: int = MODEL_PRICE_YEAR,
    year_hours: float = YEAR_HOURS,
) -> float:
    """The charge over a window of ``hours`` with ``demand_twh`` of GB demand, GBP m.

    ``year_hours``: the hours of the window's calendar year (8,784 in a leap year such as 2040), so twelve
    months still add to the annual figure.

    Capacity basis: the annual charge times the window's share of a year's hours,
    so twelve months add to the annual figure. Demand basis: per MWh of the
    window's own demand. Gross-share basis: the share of the window's own gross
    saving. Each is therefore additive over windows, which is what lets the year
    settlement sum its months.
    """
    if basis == "renewable_capacity":
        annual = forecast_error_charge_gbp_bn(0.0, 0.0, basis, as_simple_total, fleet_ratio, price_year)
        return annual * 1e3 * float(hours) / float(year_hours)
    return forecast_error_charge_gbp_bn(gross_saving_gbp_m / 1e3, demand_twh, basis, as_simple_total,
                                        fleet_ratio, price_year) * 1e3


def settings_from_config(block: Mapping | None) -> dict:
    """Validated forecast-error settings from a config block (missing keys take the defaults)."""
    merged = {**DEFAULT_SETTINGS, **dict(block or {})}
    unknown = sorted(set(merged) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"forecast_error_charge has unknown keys: {unknown}")
    if merged["basis"] not in BASES:
        raise ValueError(f"forecast_error_charge.basis must be one of {BASES}, got {merged['basis']!r}")
    if not isinstance(merged["enabled"], bool) or not isinstance(merged["as_simple_total"], bool):
        raise ValueError("forecast_error_charge.enabled and .as_simple_total must be true or false")
    ratio = float(merged["fleet_ratio"])
    if not ratio > 0:
        raise ValueError("forecast_error_charge.fleet_ratio must be positive")
    _price_factor(merged["price_year"])
    return {"enabled": merged["enabled"], "basis": merged["basis"], "as_simple_total": merged["as_simple_total"],
            "fleet_ratio": ratio, "price_year": int(merged["price_year"])}


def adjusted_saving_gbp_bn(gross_saving_gbp_bn: float, demand_twh: float, **kw) -> float:
    return gross_saving_gbp_bn - forecast_error_charge_gbp_bn(gross_saving_gbp_bn, demand_twh, **kw)


#: The bill line a forecast-error charge is recovered like (a balancing cost, through BSUoS).
RECOVERY_LINE = "bsuos_constraints"


def gbp_per_household(system_gbp_model_year: float, bill_summary: Mapping) -> float:
    """System GBP (2024 model money) as GBP per 3,100 kWh household, 2025 prices, ex VAT (house rule 8).

    The balancing line's recovery rule, through the bill's own entry for the BSUoS constraint
    line (F19, 7 Oct 2026: a GSP rate on the household's loss-uplifted 3.1 MWh on the
    two-volume basis; flat over final demand on a D1 bill). Never over a count of households.
    """
    from scripts.public_ownership import bill_recovery

    return float(system_gbp_model_year) * bill_recovery.household_per_system(bill_summary, RECOVERY_LINE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gross-saving-bn", type=float, default=None,
                    help="gross consumer saving, GBP bn a year in 2024 prices (default: read --settlement)")
    ap.add_argument("--demand-twh", type=float, default=None)
    ap.add_argument("--settlement", default="resources/market/CP30_FFR_2030_year_cp30_network_central_dispatch_fti_settlement_copperplate.csv",
                    help="year FTI settlement CSV supplying the gross saving and demand when not given")
    ap.add_argument("--bill-summary", default="resources/public_ownership/cp30_2030_cp30_network_consumer_bill_by_ownership_case_summary.json",
                    help="consumer bill summary whose recovery denominator converts to GBP per household (house rule 8)")
    ap.add_argument("--basis", default=DEFAULT_BASIS, choices=BASES)
    ap.add_argument("--as-simple-total", action="store_true")
    ap.add_argument("--all-bases", action="store_true", help="print every basis, not just the chosen one")
    a = ap.parse_args(argv)

    gross, demand = a.gross_saving_bn, a.demand_twh
    # The charge replaces the model's own real-time correction (26 Sep 2026), so the
    # net puts that correction back, as the settlement headline does.
    correction = 0.0
    if gross is None or demand is None:
        import pandas as pd

        frame = pd.read_csv(a.settlement).set_index("metric")
        col = "YEAR" if "YEAR" in frame.columns else frame.columns[-1]
        key = "consumer_saving_gross_gbp_m" if "consumer_saving_gross_gbp_m" in frame.index else "consumer_saving_gbp_m"
        gross = float(frame.loc[key, col]) / 1e3 if gross is None else gross
        demand = float(frame.loc["gb_demand_twh", col]) if demand is None else demand
        if "central_realtime_correction_gbp_m" in frame.index:
            correction = float(frame.loc["central_realtime_correction_gbp_m", col]) / 1e3
    bill = None
    if a.bill_summary and Path(a.bill_summary).exists():
        bill = json.loads(Path(a.bill_summary).read_text(encoding="utf-8"))

    rows = []
    if a.all_bases:
        for b in BASES:
            for st in (False, True) if b in ("renewable_capacity", "demand") else (a.as_simple_total,):
                rows.append((b, st))
    else:
        rows.append((a.basis, a.as_simple_total))
    print(f"gross saving {gross:.3f} bn/yr on {demand:.1f} TWh (2024 prices); model real-time correction "
          f"{correction:.3f} bn/yr, replaced by the charge")
    for b, st in rows:
        d = forecast_error_charge_gbp_bn(gross, demand, basis=b, as_simple_total=st)
        net = gross + correction - d
        tag = f"{b}{', simple total' if st else ''}"
        star = "  <- default" if (b == DEFAULT_BASIS and not st and not a.all_bases) else ""
        hh = (f" = {gbp_per_household(net * 1e9, bill):6.2f} GBP per 3,100 kWh ex VAT (2025 prices)"
              if bill else " (no bill summary: per-household figure not shown)")
        print(f"  {tag:<34} charge {d:.3f}  ->  {net:.3f} bn/yr{hh}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
