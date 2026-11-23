"""Bill-level sensitivity: Hinkley Point C on EDF's current schedule (user decision, 29 Sep 2026).

The model keeps CP30's physical assumption for HPC, 2,317 MW all year (about 18.27
TWh settled in 2030).  EDF's 2025 URD puts unit 1's first power in mid-2030 and unit 2
a year later.  This module prices the difference on the bill without a re-solve, for
every case alike (a background sensitivity, like the legacy-levy switch):

* **Market bill** (private, P1, P2 and every market case; HPC stays privately owned
  on its CfD): the HPC CfD settlement falls pro rata with the output,
  ``-dq x t`` with ``t`` the modelled CfD top-up per settled MWh (at the bill's
  89.50 strike), and the missing energy is bought at the modelled price over the
  months it is missing, ``dq x (P - ref)``, booked on the wholesale line: HPC's
  energy had been settled at its CfD reference ``ref = strike - t``, so the two
  rows together replace ``strike x dq`` with ``P x dq`` and the bill closes.
* **Acquisition tariff** (a case whose register owns HPC): the tariff stops paying
  HPC's fuel and variable O&M on ``dq`` and buys ``dq`` at ``P``; the purchase price
  of HPC (market or state present value) falls by the missing share of its 2030 cash
  flow, discounted half a year, annuitised like the purchase.  The completion price
  (``in_construction: public_completion``) does not depend on output.

Declared approximation: dispatch, emissions and constraint effects of the missing
output are not re-solved; the replacement is priced at the market-year Stage 1
system price (time-weighted in each month, weighted across months by the missing
output).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR, calendar_months

HPC_ASSET_PREFIX = "Hinkley Point C"
#: The default year's months, kept for callers that import it; every function takes
#: the run's own year (item Y4), and :func:`schedule_shortfall` reads it off the prices.
MONTHS = calendar_months(DEFAULT_MODEL_YEAR)


def model_year_months(model_year: int | None = None) -> pd.PeriodIndex:
    """The twelve months of the model year (the default year when none is given)."""
    return calendar_months(DEFAULT_MODEL_YEAR if model_year is None else int(model_year))


def prices_model_year(prices: Mapping[str, float]) -> int:
    """The one calendar year a ``{YYYY-MM: price}`` mapping covers."""
    years = {pd.Period(str(month), freq="M").year for month in prices}
    if len(years) != 1:
        raise ValueError(f"the monthly prices span {sorted(years)}; they must be one model year")
    return years.pop()


def load_schedule(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _month_hours(month: pd.Period) -> float:
    return float(month.days_in_month * 24)


def edf_monthly_mwh(schedule: Mapping, first_power: str | None = None,
                    model_year: int | None = None) -> dict[str, float]:
    """Unit 1's model-year output by month: zero before first power, a linear ramp, then full availability."""
    edf = schedule["edf_schedule"]
    start = pd.Timestamp(first_power or edf["unit_1_first_power"])
    ramp_days = float(edf["ramp_months"]) * 365.25 / 12.0
    full_mw = float(edf["unit_1_mw"]) * float(edf["availability_at_full_power"])
    out = {}
    for month in model_year_months(model_year):
        days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
        # Mean fraction of full power over the month's days (ramp evaluated at each day's midpoint).
        elapsed = (days + pd.Timedelta(hours=12) - start) / pd.Timedelta(days=1)
        fraction = np.clip(np.asarray(elapsed, dtype=float) / ramp_days, 0.0, 1.0)
        out[str(month)] = full_mw * 24.0 * float(fraction.sum())
    return out


def cp30_monthly_mwh(settled_mwh: float, model_year: int | None = None) -> dict[str, float]:
    """CP30's HPC output by month: the settled annual total spread by hours (a baseload unit)."""
    months = model_year_months(model_year)
    total_hours = sum(_month_hours(m) for m in months)
    return {str(m): float(settled_mwh) * _month_hours(m) / total_hours for m in months}


def monthly_prices(price_files: Mapping[str, str | Path]) -> dict[str, float]:
    """Time-weighted mean Stage 1 system price per month (``{YYYY-MM: path}`` of *_wholesale_price.csv)."""
    out = {}
    for month, path in price_files.items():
        frame = pd.read_csv(path, usecols=["snapshot", "wholesale_price"])
        out[str(month)] = float(pd.to_numeric(frame["wholesale_price"], errors="coerce").mean())
    return out


def schedule_shortfall(schedule: Mapping, settled_mwh: float, prices: Mapping[str, float],
                       first_power: str | None = None) -> dict:
    """The missing output and the price it is replaced at, for one first-power date (the prices' year)."""
    model_year = prices_model_year(prices) if prices else None
    cp30 = cp30_monthly_mwh(settled_mwh, model_year)
    edf = edf_monthly_mwh(schedule, first_power, model_year)
    missing = {m: max(cp30[m] - edf[m], 0.0) for m in cp30}
    dq = float(sum(missing.values()))
    price = (float(sum(missing[m] * float(prices[m]) for m in missing)) / dq) if dq else 0.0
    return {"cp30_mwh": float(settled_mwh), "edf_mwh": float(sum(edf.values())), "missing_mwh": dq,
            "replacement_price_gbp_per_mwh": price, "edf_mwh_by_month": edf, "missing_mwh_by_month": missing}


def market_bill_change(shortfall: Mapping, *, cfd_payment_gbp: float, settled_mwh: float,
                       strike_gbp_per_mwh: float, conv: Mapping, eligible_mwh: float | None = None,
                       line_rates: Mapping | None = None) -> dict:
    """The two market-bill rows (GBP a household ex VAT, 2025 money) and their system amounts (model money).

    ``line_rates`` (F19, 7 Oct 2026): GBP per household per system GBP of the CfD and wholesale
    lines (``{"cfd": ..., "wholesale": ...}``), from the bill's recovery entries on the two-volume
    basis; without them the CfD over ``eligible_mwh`` and the wholesale line over final demand.
    """
    dq = float(shortfall["missing_mwh"])
    top_up = float(cfd_payment_gbp) / float(settled_mwh) if settled_mwh else 0.0
    reference = float(strike_gbp_per_mwh) - top_up
    cfd_system = -dq * top_up
    wholesale_system = dq * (float(shortfall["replacement_price_gbp_per_mwh"]) - reference)
    factor, mwh = float(conv["factor"]), float(conv["mwh"])
    cfd_rate = float(line_rates["cfd"]) if line_rates else factor / float(eligible_mwh) * mwh
    wholesale_rate = float(line_rates["wholesale"]) if line_rates else factor / float(conv["final_demand_mwh"]) * mwh
    return {"cfd_line_system_gbp": cfd_system, "wholesale_line_system_gbp": wholesale_system,
            "cfd_top_up_gbp_per_mwh": top_up, "cfd_reference_gbp_per_mwh": reference,
            "cfd_line_gbp_per_3100kwh_ex_vat": cfd_system * cfd_rate,
            "wholesale_line_gbp_per_3100kwh_ex_vat": wholesale_system * wholesale_rate}


def tariff_change(shortfall: Mapping, *, variable_cost_gbp_per_mwh: float, annual_flow_gbp: float,
                  settled_mwh: float, basis_rate: float | None, public_rate: float, term_years: float,
                  conv: Mapping, line_rate: float | None = None) -> dict:
    """The acquisition tariff's rows for a case that owns HPC (GBP a household ex VAT, 2025 money).

    ``line_rate`` (F19): the tariff's own rate for its pool lines (GBP per household per system
    GBP); without it, flat over final demand.
    """
    dq = float(shortfall["missing_mwh"])
    energy_system = dq * (float(shortfall["replacement_price_gbp_per_mwh"]) - float(variable_cost_gbp_per_mwh))
    share = dq / float(settled_mwh) if settled_mwh else 0.0
    price_change = (-share * float(annual_flow_gbp) / (1.0 + float(basis_rate)) ** 0.5
                    if basis_rate is not None else 0.0)
    annuity_system = price_change * capital_recovery_factor(float(public_rate), float(term_years))
    per_hh = ((lambda gbp: gbp * float(line_rate)) if line_rate else  # noqa: E731
              (lambda gbp: gbp * float(conv["factor"]) / float(conv["final_demand_mwh"]) * float(conv["mwh"])))
    return {"energy_system_gbp": energy_system, "purchase_price_change_gbp": price_change,
            "annuity_system_gbp": annuity_system,
            "energy_gbp_per_3100kwh_ex_vat": per_hh(energy_system),
            "annuity_gbp_per_3100kwh_ex_vat": per_hh(annuity_system)}


def hpc_inputs(register: pd.DataFrame) -> dict:
    """HPC's private register row: settled output and variable cost per MWh (fuel, carbon and variable O&M)."""
    rows = register.loc[register["case_id"].eq("private")
                        & register["asset_id"].astype(str).str.startswith(HPC_ASSET_PREFIX)]
    if rows.empty:
        return {}
    num = lambda c: float(pd.to_numeric(rows[c], errors="coerce").fillna(0.0).sum()) if c in rows.columns else 0.0  # noqa: E731
    mwh = num("dispatch_mwh")
    return {"asset_ids": sorted(rows["asset_id"].astype(str)), "dispatch_mwh": mwh,
            "variable_cost_gbp_per_mwh": (num("fuel_carbon_gbp") + num("annual_variable_om_gbp")) / mwh if mwh else 0.0}
