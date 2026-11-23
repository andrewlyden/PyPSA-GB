"""The private support regime in calendar 2050 (FES 2050 plan, task Y11; decisions 4, 6, 19, 22).

The private counterfactual is today's arrangements extended (decision 4). This module
prices and settles the schemes the private 2050 fleet holds, beside
``scripts/market/cfd_strikes_2050.py`` (CfDs) and
``scripts/market/capacity_market_2050.py`` (the Capacity Market):

* **Renewables Obligation**: zero; it closed in 2037.
* **Dispatchable power agreements** (gas CCS; hydrogen power by decision 22):
  :func:`dpa_contract` prices a vintage at cost plus the private rate, and
  :func:`dpa_settlement` settles it on a solved year:

      availability payment = APR x capacity x availability
      variable payment     = sum_h max(0, SRMC_plant,h - SRMC_reference CCGT,h + margin) x output_h
      gainshare            = share x max(0, energy margin - threshold)
      support cost (bill)  = availability payment + variable payment - gainshare

  APR = (capex x CRF(private rate, life) + fixed O&M + insurance + use of system) / expected
  availability, from the build year's Y2 vintage row. Insurance is counted once on the
  register's basis (task Y14-prep): DESNZ insurance out of fixed O&M, the finance layer's
  D4(a) line in (``cfd_strikes_2050.insured_fixed_cost``); so is the LDES cap and floor. The 15-year term means CCS and
  hydrogen plant built before 2036 are off their DPA by 2050 and are merchant (they then
  enter the Capacity Market). NZT Power (2028 + 15) has ended.
* **BECCS** on a declared dispatchable CfD for its electricity leg (:func:`beccs_strike`),
  the removals leg paid off-bill, as the 2030 Power BECCS line; 15-year term.
* **LDES cap and floor** on 25-year terms (:func:`ldes_contract`, :func:`ldes_settlement`):
  floor = required revenue with capital at a debt-like 4.5% real (the 2030 private
  floor), cap = required revenue at the CEPA cost of equity; top-up below the floor and
  return above the cap, recovered over final demand.

:func:`support_book_2050` applies all of it to a fleet (the FES 2025 HT stock-flow
fleet by default, the Y10 register when wired) and :func:`bill_line_map` states which
bill line and denominator each scheme reaches. Every settlement takes the solved
year's dispatch and prices as arguments (the hook) and is tested on synthetic inputs.

Run by hand (no solve)::

    python scripts/public_ownership/support_regime_2050.py --out DIR
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market import capacity_market_2050 as cm50  # noqa: E402
from scripts.market import cfd_strikes_2050 as cfd50  # noqa: E402
from scripts.public_ownership import vintage_costs as vc  # noqa: E402
from scripts.public_ownership.finance import capital_recovery_factor  # noqa: E402

PARAMETERS = ROOT / "data" / "market" / "support" / "support_regime_2050.json"
MODEL_YEAR = 2050
#: FES 2025 HT 2050 hydrogen turbine split, ES1 rows 54, 55, 183, 184: transmission hydrogen
#: (15.90 GW) and hydrogen CHP (1.00 GW) as CCHT, hydrogen peaking (5.60 + 3.58 GW) as OCHT.
HYDROGEN_TECHNOLOGY_SPLIT = {"Hydrogen CCHT": (15897.0 + 998.8582) / 26081.8942,
                             "Hydrogen OCHT": (5604.036 + 3582.0) / 26081.8942}


def load_parameters(path: str | Path = PARAMETERS) -> dict:
    """The declared scheme parameters (``support_regime_2050.json``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _value(block) -> float:
    return float(block["value"] if isinstance(block, Mapping) else block)


# ── dispatchable power agreements ────────────────────────────────────────────

@dataclass(frozen=True)
class DPAContract:
    """One DPA vintage: the availability payment rate and the variable-payment terms."""

    carrier: str
    build_year: int
    capacity_mw: float
    apr_gbp_per_kw_year: float
    expected_availability: float
    term_years: int
    live_share: float
    efficiency: float
    variable_om_gbp_per_mwh: float
    residual_emission_tco2_per_mwh: float
    margin_gbp_per_mwh: float
    gainshare_share: float
    gainshare_threshold_gbp_per_kw_year: float
    private_rate: float
    fixed_cost_gbp_per_kw_year: float
    source: str


def dpa_contract(carrier: str, build_year: int, capacity_mw: float, *, snapshot: Mapping,
                 params: Mapping | None = None, rate_path: str = "flat", learning: bool = False,
                 model_year: int = MODEL_YEAR, vintage_table: pd.DataFrame | None = None,
                 insurance: str = cfd50.INSURANCE_RULE, insurance_config: Mapping | None = None,
                 technology: str | None = None) -> DPAContract:
    """A DPA for a gas CCS or hydrogen-power vintage, priced at cost plus the private rate.

    Args:
        carrier: ``CCS Gas`` or ``Hydrogen``.
        build_year: Commissioning year.
        capacity_mw: Net dependable capacity.
        snapshot: The workbook snapshot (private rates).
        params: :func:`load_parameters` output.
        rate_path: ``flat`` (decision 19) or ``neso_wind_ramp`` (no effect on thermal rows).
        learning: Capex from the learning column.
        model_year: The settled year (sets ``live_share``).
        vintage_table: A loaded vintage table.
        insurance: ``exclude`` (the register's rule, the default: DESNZ insurance out of fixed
            O&M and the finance layer's D4(a) insurance added, so the APR equals the register's
            required revenue) or ``include`` (DESNZ insurance kept, no D4(a) line).
        insurance_config: The D4(a) block (default: the 2050 study config's,
            ``cfd_strikes_2050.finance_insurance_config``).
        technology: Price one vintage-table technology instead of the carrier's mix (task
            SETTLE2050: a register cohort knows whether it is ``Hydrogen CCHT`` or ``OCHT``).
            Must be one of the carrier's technologies.

    Returns:
        The contract.
    """
    params = load_parameters() if params is None else params
    dpa, thermal = params["dpa"], params["thermal"]
    if carrier not in dpa["technologies"]:
        raise KeyError(f"no DPA for {carrier!r}; DPAs are {sorted(dpa['technologies'])}")
    split = HYDROGEN_TECHNOLOGY_SPLIT if carrier == "Hydrogen" else {dpa["technologies"][carrier]: 1.0}
    if technology is not None:
        if technology not in split:
            raise KeyError(f"{technology!r} is not a {carrier} DPA technology; they are {sorted(split)}")
        split = {technology: 1.0}
    fixed, rates, sources = 0.0, [], []
    uos = float(dpa["use_of_system_gbp_per_kw_year"][carrier])
    cost_year = max(int(build_year), vc.FIRST_BUILD_YEAR)   # a pre-2025 unit is costed as a 2025 vintage
    for technology, share in split.items():
        cost = vc.costs_for(technology, cost_year, learning=learning, table=vintage_table)
        rate = cfd50.private_rate(technology, cost_year, snapshot=snapshot, rate_path=rate_path,
                                  vintage_table=vintage_table)
        capex = float(cost["capex_power_gbp_per_kw"])
        insured = cfd50.insured_fixed_cost(cost, capex, insurance=insurance, insurance_config=insurance_config)
        annual = (capex * capital_recovery_factor(rate["rate"], float(cost["economic_life_years"]))
                  + insured["fixed_total_gbp_per_kw_year"] + uos)
        fixed += share * annual
        rates.append(share * rate["rate"])
        sources.append(f"{technology} {build_year}: {cost['source_page']}; {rate['source']}; {insured['insurance_rule']}")
    availability = _value(dpa["expected_availability"])
    term = int(_value(dpa["term_years"]))
    ef = _value(thermal["gas_emission_factor_tco2_per_mwh_thermal"])
    capture = _value(thermal["ccs_capture_rate"])
    efficiency = float(thermal["efficiency"][carrier])
    residual = ef * (1.0 - capture) / efficiency if carrier == "CCS Gas" else 0.0
    return DPAContract(
        carrier=carrier, build_year=int(build_year), capacity_mw=float(capacity_mw),
        apr_gbp_per_kw_year=fixed / availability, expected_availability=availability, term_years=term,
        live_share=cm50.live_fraction(int(build_year), term, model_year), efficiency=efficiency,
        variable_om_gbp_per_mwh=float(thermal["variable_om_gbp_per_mwh"][carrier]),
        residual_emission_tco2_per_mwh=residual, margin_gbp_per_mwh=_value(dpa["variable_payment_margin_gbp_per_mwh"]),
        gainshare_share=_value(dpa["gainshare_share"]),
        gainshare_threshold_gbp_per_kw_year=_value(dpa["gainshare_threshold_gbp_per_kw_year"]),
        private_rate=float(sum(rates)), fixed_cost_gbp_per_kw_year=fixed, source=" | ".join(sources))


def reference_ccgt_srmc(gas, carbon, params: Mapping | None = None):
    """Short-run cost of the unabated reference CCGT (GBP/MWh): fuel + carbon + VOM at its efficiency."""
    params = load_parameters() if params is None else params
    thermal = params["thermal"]
    eff = float(thermal["efficiency"]["CCGT"])
    ef = _value(thermal["gas_emission_factor_tco2_per_mwh_thermal"])
    return (np.asarray(gas, dtype=float) + np.asarray(carbon, dtype=float) * ef) / eff + float(
        thermal["variable_om_gbp_per_mwh"]["CCGT"])


def dpa_plant_srmc(contract: DPAContract, *, gas=None, carbon=None, hydrogen_price=None):
    """Short-run cost of the DPA plant (GBP/MWh): gas CCS on gas and residual carbon, hydrogen on the H2 price."""
    if contract.carrier == "CCS Gas":
        if gas is None or carbon is None:
            raise ValueError("gas CCS needs gas and carbon prices")
        return (np.asarray(gas, dtype=float) / contract.efficiency
                + np.asarray(carbon, dtype=float) * contract.residual_emission_tco2_per_mwh
                + contract.variable_om_gbp_per_mwh)
    if hydrogen_price is None:
        raise ValueError("hydrogen power needs the hydrogen price (the solved H2 bus price, GBP/MWh H2)")
    return np.asarray(hydrogen_price, dtype=float) / contract.efficiency + contract.variable_om_gbp_per_mwh


def dpa_payments(contract: DPAContract, *, output_mwh: float, variable_top_up_gbp: float, market_revenue_gbp: float,
                 short_run_cost_gbp: float, availability: Optional[float] = None, period_share: float = 1.0,
                 params: Mapping | None = None) -> dict:
    """One DPA's payments from its period sums: the single formula every 2050 DPA settles on.

    ``variable_top_up_gbp`` is ``sum_h max(SRMC_plant,h - SRMC_ref,h + margin, 0) x output_h``;
    the other three are the period's output, market revenue and short-run cost. All four
    are the unit's own, before the live share, which this applies. The availability
    payment, the gainshare threshold and the fixed cost are annual amounts, scaled by
    ``period_share`` (the settled hours over the calendar year's) so a part year pays its part.

    Args:
        contract: The unit's contract (``capacity_mw`` is the unit's).
        output_mwh: Output.
        variable_top_up_gbp: The variable payment before the live share.
        market_revenue_gbp: Market revenue.
        short_run_cost_gbp: Short-run cost.
        availability: Measured availability; the declared expected value if None.
        period_share: Share of the calendar year settled.
        params: Scheme parameters (:func:`has_variable_payment`: a carrier without the variable
            payment, hydrogen power, is paid none whatever its top-up).

    Returns:
        Availability and variable payments, market revenue, short-run cost, energy margin,
        gainshare and the net support cost to consumers, each scaled by the live share.
    """
    live = contract.live_share
    avail = contract.expected_availability if availability is None else float(availability)
    kw = contract.capacity_mw * 1e3
    share = float(period_share)
    availability_payment = contract.apr_gbp_per_kw_year * kw * avail * live * share
    variable_payment = float(variable_top_up_gbp) * live if has_variable_payment(contract.carrier, params) else 0.0
    market_revenue = float(market_revenue_gbp) * live
    short_run_cost = float(short_run_cost_gbp) * live
    margin = market_revenue - short_run_cost
    threshold = contract.gainshare_threshold_gbp_per_kw_year * kw * live * share
    gainshare = contract.gainshare_share * max(margin - threshold, 0.0)
    return {"carrier": contract.carrier, "build_year": contract.build_year, "capacity_mw": contract.capacity_mw,
            "live_share": live, "output_mwh": float(output_mwh) * live,
            "availability_payment_gbp": availability_payment, "variable_payment_gbp": variable_payment,
            "market_revenue_gbp": market_revenue, "short_run_cost_gbp": short_run_cost,
            "energy_margin_gbp": margin, "gainshare_gbp": gainshare,
            "net_support_cost_gbp": availability_payment + variable_payment - gainshare,
            "plant_net_revenue_gbp": market_revenue - short_run_cost + availability_payment + variable_payment - gainshare,
            "fixed_cost_gbp": contract.fixed_cost_gbp_per_kw_year * kw * live * share}


def has_variable_payment(carrier: str, params: Mapping | None = None) -> bool:
    """Whether a DPA carrier's contract carries the variable payment (``dpa.variable_payment_carriers``).

    Gas CCS does (the NZT Power DPA); hydrogen power does not (user, 26 Oct 2026: DESNZ's
    hydrogen-to-power response, Dec 2024, adopts an availability payment and no variable
    payment). A parameter file without the key gives every DPA carrier the payment.
    """
    params = load_parameters() if params is None else params
    block = params["dpa"].get("variable_payment_carriers")
    if block is None:
        return True
    return str(carrier) in set(block["value"] if isinstance(block, Mapping) else block)


def dpa_settlement(contract: DPAContract, dispatch_mwh: pd.Series, price: pd.Series, *, gas, carbon,
                   hydrogen_price=None, availability: Optional[float] = None, params: Mapping | None = None,
                   plant_srmc=None) -> dict:
    """Settle one DPA on a solved year (:func:`dpa_payments` on the year's hourly sums).

    Args:
        contract: :func:`dpa_contract` output.
        dispatch_mwh: The unit's MWh by snapshot (already weighted by snapshot hours).
        price: The wholesale price it sells at, by snapshot (2024 prices).
        gas: Gas price (scalar or by snapshot), GBP/MWh thermal.
        carbon: Carbon price (scalar or by snapshot), GBP/t.
        hydrogen_price: Hydrogen price by snapshot, GBP/MWh H2 (hydrogen power only).
        availability: Measured availability of low-carbon capacity; the declared expected value if None.
        params: Scheme parameters.
        plant_srmc: The plant's own short-run cost (GBP/MWh, scalar or by snapshot) as the
            model dispatches it, in place of :func:`dpa_plant_srmc` (task SETTLE2050: the
            2050 settlement prices a turbine's hydrogen at its bus and its link efficiency).

    Returns:
        :func:`dpa_payments` for the year.
    """
    if not pd.DatetimeIndex(dispatch_mwh.index).equals(pd.DatetimeIndex(price.index)):
        raise ValueError("dispatch and price snapshots differ")
    out_mwh = dispatch_mwh.to_numpy(dtype=float)
    own = (dpa_plant_srmc(contract, gas=gas, carbon=carbon, hydrogen_price=hydrogen_price)
           if plant_srmc is None else np.asarray(plant_srmc, dtype=float))
    srmc = np.broadcast_to(own, out_mwh.shape)
    reference = np.broadcast_to(reference_ccgt_srmc(gas, carbon, params), out_mwh.shape)
    top_up = np.maximum(srmc - reference + contract.margin_gbp_per_mwh, 0.0)
    return dpa_payments(contract, output_mwh=float(out_mwh.sum()), variable_top_up_gbp=float((top_up * out_mwh).sum()),
                        market_revenue_gbp=float((price.to_numpy(dtype=float) * out_mwh).sum()),
                        short_run_cost_gbp=float((srmc * out_mwh).sum()), availability=availability,
                        params=params)


# ── BECCS dispatchable CfD ───────────────────────────────────────────────────

def beccs_strike(build_year: int, *, snapshot: Mapping, params: Mapping | None = None, rate_path: str = "flat",
                 biomass_price: Optional[float] = None, removal_price: Optional[float] = None, **kwargs) -> dict:
    """The electricity-leg strike of a Power BECCS vintage.

    Levelised cost at the private rate (fuel at the declared biomass price over the
    model efficiency) less the removals revenue per MWh (captured biogenic CO2 x the
    declared removal price), which the off-bill removals contract pays.

    Args:
        build_year: Commissioning year.
        snapshot: The workbook snapshot.
        params: Scheme parameters.
        rate_path: As :func:`cfd_strikes_2050.levelised_strike`.
        biomass_price: GBP/MWh thermal; the declared default if None.
        removal_price: GBP/tCO2; the declared default if None.
        **kwargs: Passed to ``levelised_strike``.

    Returns:
        The strike row, with ``removal_credit_gbp_per_mwh`` and ``captured_tco2_per_mwh``.
    """
    params = load_parameters() if params is None else params
    thermal, beccs, prices = params["thermal"], params["beccs"], params["commodity_prices"]
    eff = float(thermal["efficiency"]["CCS Biomass"])
    fuel = (_value(prices["biomass_gbp_per_mwh_thermal"]) if biomass_price is None else float(biomass_price)) / eff
    captured = _value(thermal["biomass_emission_factor_tco2_per_mwh_thermal"]) * _value(beccs["capture_rate"]) / eff
    credit = captured * (_value(beccs["removal_price_gbp_per_tco2"]) if removal_price is None else float(removal_price))
    row = cfd50.levelised_strike(beccs["technology"], build_year, snapshot=snapshot, rate_path=rate_path,
                                 fuel_gbp_per_mwh=fuel, other_revenue_gbp_per_mwh=credit, **kwargs)
    row.update(removal_credit_gbp_per_mwh=credit, captured_tco2_per_mwh=captured,
               fuel_basis=f"biomass {_value(prices['biomass_gbp_per_mwh_thermal']) if biomass_price is None else biomass_price} "
                          f"GBP/MWh_th / efficiency {eff}")
    return row


# ── LDES cap and floor ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class LDESContract:
    """One LDES vintage's floor and cap (GBP a year) on a 25-year term."""

    carrier: str
    technology: str
    build_year: int
    capacity_mw: float
    floor_gbp: float
    cap_gbp: float
    floor_rate: float
    cap_rate: float
    term_years: int
    live_share: float
    source: str


def cost_of_equity(snapshot: Mapping, wacc_technology: str) -> float:
    """The workbook's pre-tax real cost of equity for a snapshot row (F29: workbook v2.0 ``Sc1 Operative`` K,
    at the technology's operative gearing; v11-2's WACC Calculator F was at CEPA's 50%)."""
    match = [row for row in snapshot.get("technologies", []) if vc._key(row["technology"]) == vc._key(wacc_technology)]
    if len(match) != 1:
        raise KeyError(f"the workbook snapshot has no single technology row for {wacc_technology!r}")
    return float(match[0]["cost_of_equity_pre_tax"])


def ldes_contract(carrier: str, build_year: int, capacity_mw: float, *, snapshot: Mapping,
                  params: Mapping | None = None, learning: bool = False, model_year: int = MODEL_YEAR,
                  vintage_table: pd.DataFrame | None = None, insurance: str = cfd50.INSURANCE_RULE,
                  insurance_config: Mapping | None = None) -> LDESContract:
    """The cap and floor of an LDES vintage (window 1 terms, 25 years).

    Floor and cap are required revenue (capital annuity over the economic life + fixed
    O&M) with capital at the floor rate (debt-like) and at the cap rate (cost of equity).

    Args:
        carrier: ``LAES``, ``CAES`` or ``Pumped Storage Hydroelectricity``.
        build_year: Commissioning year; before the scheme's first build year it is merchant.
        capacity_mw: Power capacity.
        snapshot: The workbook snapshot.
        params: Scheme parameters.
        learning: Capex from the learning column.
        model_year: The settled year.
        vintage_table: A loaded vintage table.
        insurance: As :func:`dpa_contract` (the register's rule by default).
        insurance_config: The D4(a) block.

    Returns:
        The contract (``live_share`` 0 for a merchant vintage).
    """
    params = load_parameters() if params is None else params
    block = params["ldes_cap_and_floor"]
    technology = block["carriers"][carrier]
    term = int(_value(block["term_years"]))
    if int(build_year) < int(_value(block["first_build_year"])):
        return LDESContract(carrier=carrier, technology=technology, build_year=int(build_year),
                            capacity_mw=float(capacity_mw), floor_gbp=np.nan, cap_gbp=np.nan, floor_rate=np.nan,
                            cap_rate=np.nan, term_years=term, live_share=0.0,
                            source="built before the LDES cap and floor (window 1): merchant")
    cost = vc.costs_for(technology, build_year, learning=learning, table=vintage_table)
    capex = float(cost["capex_power_gbp_per_kw"])
    life = float(cost["economic_life_years"])
    fixed = cfd50.insured_fixed_cost(cost, capex, insurance=insurance,
                                     insurance_config=insurance_config)["fixed_total_gbp_per_kw_year"]
    floor_rate = _value(block["floor_rate"])
    cap_rate = cost_of_equity(snapshot, str(cost["wacc_technology"]))
    kw = float(capacity_mw) * 1e3
    return LDESContract(
        carrier=carrier, technology=technology, build_year=int(build_year), capacity_mw=float(capacity_mw),
        floor_gbp=kw * (capex * capital_recovery_factor(floor_rate, life) + fixed),
        cap_gbp=kw * (capex * capital_recovery_factor(cap_rate, life) + fixed),
        floor_rate=floor_rate, cap_rate=cap_rate, term_years=term,
        live_share=cm50.live_fraction(int(build_year), term, model_year),
        source=f"{technology} {build_year}: {cost['source_page']}; floor {block['floor_rate']['source']}; "
               f"cap {block['cap_rate_basis']['source']}")


def ldes_settlement(contract: LDESContract, market_revenue_gbp: float, capacity_market_revenue_gbp: float = 0.0,
                    params: Mapping | None = None) -> dict:
    """Settle one LDES cap and floor on a solved year's revenue.

    Args:
        contract: :func:`ldes_contract` output.
        market_revenue_gbp: Wholesale + balancing revenue less charging cost (the register's).
        capacity_market_revenue_gbp: CM revenue, counted against the floor and cap.
        params: Scheme parameters.

    Returns:
        Floor top-up (a bill cost), cap return (a bill credit) and the net, scaled by the live share.
    """
    params = load_parameters() if params is None else params
    share_returned = _value(params["ldes_cap_and_floor"]["above_cap_returned_share"])
    revenue = float(market_revenue_gbp) + float(capacity_market_revenue_gbp)
    live = contract.live_share
    top_up = max(contract.floor_gbp - revenue, 0.0) * live
    cap_return = share_returned * max(revenue - contract.cap_gbp, 0.0) * live
    return {"carrier": contract.carrier, "build_year": contract.build_year, "live_share": live,
            "revenue_gbp": revenue, "floor_gbp": contract.floor_gbp, "cap_gbp": contract.cap_gbp,
            "floor_topup_gbp": top_up, "cap_return_gbp": cap_return, "net_support_cost_gbp": top_up - cap_return}


# ── the book and the bill map ────────────────────────────────────────────────

#: Fleet carrier -> the scheme it holds when built new (post-AR7), and its CfD technology.
#: Hydro is left out: FES's 2 GW is the legacy fleet refurbished (the stock-flow's
#: "new build" is its replacement of unknown ages, declared merchant). Nuclear is built
#: apart (:func:`nuclear_book_rows`).
CFD_CARRIER_TECHNOLOGY = {"wind_offshore": "Offshore wind", "wind_onshore": "Onshore wind", "solar_pv": "Solar PV",
                          "marine": "Tidal stream"}


def nuclear_book_rows(known: pd.DataFrame, trajectory: pd.DataFrame, strikes: pd.DataFrame,
                      model_year: int = MODEL_YEAR) -> list[dict]:
    """The 2050 nuclear fleet as contracts: HPC (CfD) and Sizewell C (RAB) from the known-contracts
    table, SMR vintages from the FES ES1 'Nuclear - Small' path on post-AR7 CfDs, and any large
    nuclear left over (FES large less HPC and Sizewell C) as merchant.
    """
    rows = []
    large = known[known["technology"].eq("Large-scale nuclear")]
    for _, k in large.iterrows():
        rows.append({"carrier": "nuclear", "build_year": int(np.floor(k["start_year"])), "capacity_mw": float(k["capacity_mw"]),
                     "scheme": k["scheme"], "live_share": float(k["live_share"]), "cfd_round": k["cfd_round"],
                     "strike_gbp_per_mwh_2024": float(k["strike_gbp_per_mwh_2024"]), "apr_gbp_per_kw_year": np.nan,
                     "floor_gbp": np.nan, "cap_gbp": np.nan, "term_years": int(k["term_years"]),
                     "note": f"{k['project']} (known contract)"})
    total = cm50._series(trajectory, "capacity_mw", "nuclear")
    smr = cm50._series(trajectory, "capacity_mw", "nuclear_smr")
    central = strikes[strikes["variant"].eq("central")].set_index(["technology", "build_year"])
    if not smr.empty:
        vintages = cm50.stock_flow_vintages(smr[smr.index <= model_year], 40)
        for year, mw in vintages.items():
            s = central.loc[("Small modular reactors (SMRs)", int(min(max(year, cfd50.FIRST_MODELLED_BUILD_YEAR), MODEL_YEAR)))]
            rows.append({"carrier": "nuclear", "build_year": int(year), "capacity_mw": float(mw), "scheme": "CfD",
                         "live_share": cm50.live_fraction(int(year), int(s["term_years"]), model_year),
                         "cfd_round": f"{cfd50.MODELLED_ROUND_PREFIX}{int(year)}",
                         "strike_gbp_per_mwh_2024": float(s["strike_gbp_per_mwh_2024"]), "apr_gbp_per_kw_year": np.nan,
                         "floor_gbp": np.nan, "cap_gbp": np.nan, "term_years": int(s["term_years"]),
                         "note": "SMR (FES ES1 row 176) on a post-AR7 CfD"})
    residual = float(total.get(model_year, 0.0)) - float(smr.get(model_year, 0.0)) - float(large["capacity_mw"].sum())
    if residual > 1.0:
        rows.append({"carrier": "nuclear", "build_year": model_year, "capacity_mw": residual, "scheme": "merchant",
                     "live_share": 0.0, "strike_gbp_per_mwh_2024": np.nan, "apr_gbp_per_kw_year": np.nan,
                     "floor_gbp": np.nan, "cap_gbp": np.nan, "term_years": 0,
                     "note": "FES large nuclear beyond HPC and Sizewell C (no named station): merchant, declared"})
    return rows


def support_book_2050(fleet: pd.DataFrame | None = None, *, snapshot: Mapping | None = None,
                      params: Mapping | None = None, strikes: pd.DataFrame | None = None,
                      model_year: int = MODEL_YEAR) -> pd.DataFrame:
    """Every 2050 fleet vintage with its scheme, live share and contract terms (no settlement).

    Renewables and SMRs built from 2031 hold post-AR7 CfDs (strike by vintage); AR7/AR7a
    capacity is in the known-contracts table (the stock-flow's 2027-2030 vintages are
    marked ``CfD_known_round``, and 2031-2033 vintages carry a note to take the AR7
    phases out first); large nuclear is HPC (CfD) and Sizewell C (RAB); CCS and hydrogen
    hold DPAs; BECCS a dispatchable CfD; LDES from 2026 a cap and floor; hydro, pre-2027
    renewables, batteries, interconnectors, waste and biomass are merchant.

    Args:
        fleet: ``capacity_market_2050.fleet_2050`` rows (default) or ``fleet_from_register``.
        snapshot: The workbook snapshot.
        params: Scheme parameters.
        strikes: ``cfd_strikes_2050.strike_table_2050`` output (central used).
        model_year: The settled year.

    Returns:
        One row per fleet vintage.
    """
    snapshot = cfd50.load_snapshot() if snapshot is None else snapshot
    params = load_parameters() if params is None else params
    fleet = cm50.fleet_2050() if fleet is None else fleet
    if strikes is None:
        strikes = cfd50.strike_table_2050(snapshot=snapshot, variants={"central": {}})
    central = strikes[strikes["variant"].eq("central")].set_index(["technology", "build_year"])
    vintage = vc.load_vintage_costs()
    trajectory = cm50.read_fleet_trajectory()
    rows = nuclear_book_rows(cfd50.known_contracts_2050(model_year), trajectory, strikes, model_year)
    for _, f in fleet.iterrows():
        carrier, year, mw = f["carrier"], int(f["build_year"]), float(f["capacity_mw"])
        if carrier == "nuclear":
            continue
        row = {"carrier": carrier, "build_year": year, "capacity_mw": mw, "scheme": "merchant", "live_share": 0.0,
               "strike_gbp_per_mwh_2024": np.nan, "apr_gbp_per_kw_year": np.nan, "floor_gbp": np.nan, "cap_gbp": np.nan,
               "term_years": 0, "note": ""}
        if carrier in CFD_CARRIER_TECHNOLOGY:
            technology = CFD_CARRIER_TECHNOLOGY[carrier]
            if year >= cfd50.FIRST_MODELLED_BUILD_YEAR:
                key = (technology, min(year, MODEL_YEAR))
                if key in central.index:
                    s = central.loc[key]
                    row.update(scheme="CfD", cfd_round=f"{cfd50.MODELLED_ROUND_PREFIX}{year}",
                               strike_gbp_per_mwh_2024=float(s["strike_gbp_per_mwh_2024"]),
                               term_years=int(s["term_years"]),
                               live_share=cm50.live_fraction(year, int(s["term_years"]), model_year))
                if year <= 2033:
                    row["note"] = ("AR7 phases start 2031/32 and 2032/33: assign the known-contracts capacity "
                                   "(cfd_live_contracts_2050.csv) first, the rest of the vintage at this strike")
            elif year >= 2027:
                row.update(scheme="CfD_known_round",
                           note="AR6-AR8 era vintage: AR7/AR7a units settle on the known-contracts table; the rest "
                                "(AR6 15-year, AR8 20-year from 2029/30 or earlier, results unpublished) is taken "
                                "as expired by 2050, declared")
            else:
                row.update(note="CfD/RO expired or never held: merchant in 2050")
        elif carrier in ("CCS Gas", "Hydrogen"):
            c = dpa_contract(carrier, year, mw, snapshot=snapshot, params=params, model_year=model_year,
                             vintage_table=vintage)
            row.update(scheme="DPA" if c.live_share > 0 else "merchant", live_share=c.live_share,
                       apr_gbp_per_kw_year=c.apr_gbp_per_kw_year, term_years=c.term_years,
                       note="" if c.live_share > 0 else "DPA ended (15 years): merchant, CM-eligible")
        elif carrier == "CCS Biomass":
            term = int(_value(params["beccs"]["term_years"]))
            share = cm50.live_fraction(year, term, model_year) if year >= cfd50.FIRST_MODELLED_BUILD_YEAR else 0.0
            if share > 0:
                s = beccs_strike(year, snapshot=snapshot, params=params, vintage_table=vintage)
                row.update(scheme="CfD_dispatchable", live_share=share, term_years=term,
                           strike_gbp_per_mwh_2024=s["strike_gbp_per_mwh_2024"])
            else:
                row.update(note="Power BECCS CfD (15 years) ended: merchant, CM-eligible")
        elif carrier in params["ldes_cap_and_floor"]["carriers"] and carrier != "source" and carrier != "status":
            c = ldes_contract(carrier, year, mw, snapshot=snapshot, params=params, model_year=model_year,
                              vintage_table=vintage)
            row.update(scheme="cap_and_floor" if c.live_share > 0 else "merchant", live_share=c.live_share,
                       floor_gbp=c.floor_gbp, cap_gbp=c.cap_gbp, term_years=c.term_years)
        rows.append(row)
    return pd.DataFrame(rows)


def bill_line_map() -> list[dict]:
    """How each 2050 support line reaches the bill: the spec for the Y13 and Y10/Y14 owners."""
    return [
        {"scheme": "CfD (post-AR7 vintages, AR7 remainder, HPC)", "bill_line": "model_linked_rules.cfd (the CfD line)",
         "settles": "SETTLE2050: the register's contract units (contract_book_2050.cfd_units) through "
                    "revenue_tracking.compute_cfd_payments at the 2050 strike table (contract_book_2050.strike_table_frame, "
                    "= cfd_strikes_2050.strike_table_for_bill), on the solved 2050 balancing dispatch; the bill, the "
                    "asset settlement, the strike adjustments and the constraint line's compensation all read the book",
         "denominator": "cfd_eligible_supplier_sales_mwh (metered supply less the EII exemption), as 2030",
         "register": "support_revenue_gbp on CfD rows; strikes trace to the Y2 row and the private rate"},
        {"scheme": "Gas CCS and hydrogen-power DPAs", "bill_line": "the 2050 'dpa' policy line (model_linked_rules.dpa; "
                                                                  "remaining_ccs is nil)",
         "settles": "SETTLE2050: the asset settlement's DPA units (contract_book_2050.settle_dpa_book through "
                    "dpa_payments: availability + variable payment - gainshare); the bill reads its total",
         "denominator": "cfd_eligible_supplier_sales_mwh (DPAs are funded through the CfD Electricity Supplier "
                        "Obligation, ECC FBC section 5.6.1)",
         "register": "support_revenue_gbp on DPA rows; the plant keeps market margin less gainshare"},
        {"scheme": "BECCS dispatchable CfD (electricity leg)",
         "bill_line": "allocated_ccs (the CfD settlement's CCS Biomass share, ccs_levy_carriers; baseload seasonal reference)",
         "settles": "the CfD contract units at the beccs_strike entries (PAR7_<year>, beccs)",
         "denominator": "cfd_eligible_supplier_sales_mwh",
         "register": "support_revenue_gbp; the removals leg (captured CO2 x removal price) is off-bill, Exchequer"},
        {"scheme": "Capacity Market", "bill_line": "sourced_externals.capacity_market (pin calendar_2050_cost_gbp)",
         "settles": "capacity_market_2050.run_2050 (no contract book in 2050)",
         "denominator": "gsp_chargeable_volume_mwh x household_allocation_factor, as 2030",
         "register": "capacity_market_revenue_gbp via attach_capacity_market_revenue(cm_revenue_by_technology_2050.csv)"},
        {"scheme": "LDES cap and floor", "bill_line": "model_linked_rules.ldes_cap_and_floor",
         "settles": "support_regime_2050.ldes_settlement: floor top-up less cap return (CM revenue counted)",
         "denominator": "final_demand_mwh (passes through TNUoS), as 2030", "register": "ldes_floor_topup_paid_gbp"},
        {"scheme": "Sizewell C RAB", "bill_line": "sourced_externals.sizewell_c_rab (operating-phase allowance; Y13)",
         "settles": "external line, not the CfD calculator", "denominator": "metered supply (Ofgem nRAB per MWh), as 2030",
         "register": "Sizewell C row carries no CfD"},
        {"scheme": "Renewables Obligation", "bill_line": "renewables_obligation = 0 (closed 2037)", "settles": "none",
         "denominator": "n/a", "register": "no ROC rows in 2050"},
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build the 2050 private support tables (task Y11; no solve).")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--snapshot", default=str(cfd50.DEFAULT_SNAPSHOT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = cfd50.load_snapshot(args.snapshot)
    params = load_parameters()
    strikes = cfd50.strike_table_2050(snapshot=snapshot)
    known = cfd50.known_contracts_2050()
    book = support_book_2050(snapshot=snapshot, params=params, strikes=strikes)
    strikes.to_csv(out / "cfd_strikes_2050.csv", index=False)
    known.to_csv(out / "cfd_live_contracts_2050.csv", index=False)
    book.to_csv(out / "support_book_2050.csv", index=False)
    dpa = pd.DataFrame([asdict(dpa_contract(c, y, 1.0, snapshot=snapshot, params=params))
                        for c in ("CCS Gas", "Hydrogen") for y in (2036, 2040, 2045, 2050)])
    dpa.to_csv(out / "dpa_parameters_2050.csv", index=False)
    beccs = pd.DataFrame([beccs_strike(y, snapshot=snapshot, params=params) for y in (2036, 2040, 2045, 2050)])
    beccs.to_csv(out / "beccs_strikes_2050.csv", index=False)
    ldes = pd.DataFrame([asdict(ldes_contract(c, y, 1.0, snapshot=snapshot, params=params))
                         for c in ("LAES", "CAES", "Pumped Storage Hydroelectricity") for y in (2030, 2040, 2050)])
    ldes.to_csv(out / "ldes_cap_floor_2050.csv", index=False)
    live = known[known["live_share"] > 0]
    dpa_rows = book[book["scheme"].eq("DPA")]
    availability = _value(params["dpa"]["expected_availability"])
    ap = dpa_rows["apr_gbp_per_kw_year"] * dpa_rows["capacity_mw"] * 1e3 * availability * dpa_rows["live_share"]
    summary_dpa = {"availability_payments_gbp_by_carrier": ap.groupby(dpa_rows["carrier"]).sum().round(0).to_dict(),
                   "live_capacity_mw_by_carrier": (dpa_rows["capacity_mw"] * dpa_rows["live_share"])
                   .groupby(dpa_rows["carrier"]).sum().round(1).to_dict(),
                   "note": "Dispatch-independent part of the DPA line at the declared expected availability; the "
                           "variable payment and the gainshare need the solved year (dpa_settlement)"}
    summary = {
        "dpa_availability": summary_dpa,
        "model_year": MODEL_YEAR, "price_year": 2024,
        "known_contracts_live_mw": live.groupby(["cfd_round", "technology"])["live_mw"].sum().round(1)
                                      .reset_index().to_dict("records"),
        "known_contracts_expired": sorted(set(known.loc[known["live_share"].eq(0), "cfd_round"])),
        "book_capacity_mw_by_scheme": book.assign(live_mw=book["capacity_mw"] * book["live_share"])
                                          .groupby(["carrier", "scheme"])[["capacity_mw", "live_mw"]].sum().round(1)
                                          .reset_index().to_dict("records"),
        "bill_line_map": bill_line_map(),
        "parameters": str(PARAMETERS.relative_to(ROOT)),
    }
    (out / "support_regime_2050_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out}: {len(strikes)} strikes, {len(known)} known contracts, {len(book)} book rows")


if __name__ == "__main__":
    main()
