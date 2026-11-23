"""Build the three bill input files of a long-range FES model year (one builder for every year).

Written for the 2050 files and made year-generic for FES 2040. For model year Y it writes

* ``data/market/consumer_bill_benchmarks/<Y>_pypsa_informed_assumptions.json``: every
  ``sourced_externals`` line given a Y value and a ``period_basis``, the heat-pump plus EV
  household, the declared charge basis, and the ledger switches;
* ``data/market/consumer_bill_forecast/<Y>_recovery_demand_assumptions.json``: the pre-run recovery
  denominators from FES HT calendar Y (``refresh_recovery_demand.py`` replaces the volumes after the solve);
* ``data/market/consumer_bill_forecast/<Y>_ro_scheme_parameters.json``: the closed RO.

How it is year-generic:

* **FES figures** are read for Y on the 2050 builder's own definitions
  (:mod:`scripts.consumer_bill.fes_bill_figures`; each reproduces the 2050 constant exactly).
* **Network lines** take Y12's roll-forward to Y (``network_rav_roll_forward`` on
  ``data/market/network_cost_sources/rav_roll_forward_<Y>.yaml``).
* **Capacity Market** reads the year's builder summary
  (``resources/market/capacity_market/cm_calendar_<Y>_summary.json``; with a contract book where
  the register still reaches Y).
* **Facts that depend on the year** branch on it: the RO (closed from calendar 2038), Feed-in Tariffs
  (the last tariffs end about 2039-41), Net Zero Teesside's DPA (live to 2043; settled on the dpa
  line either way), Sizewell C (operating from 2039), and F.05's efficiency gains (published for
  2050; interpolated for other years).
* **The text** is written for 2050 and moved to Y by :func:`year_text` (references to the 2050 plan,
  its decisions and its modules stay).

A year this builder cannot represent faithfully (RO or FiT still live, Sizewell C in construction)
is refused, not approximated. ``tests/unit/test_fes_bill_inputs.py`` checks that it reproduces the
committed 2050 and 2040 files exactly.

Usage::

    python -m scripts.consumer_bill.build_fes_bill_inputs --year 2045 [--check]
"""

import argparse
import copy
import json
import re
from pathlib import Path

#: The calendar-Y Capacity Market (2025 money): the year's builder summary (rule build_capacity_market_year, every
#: long-range year). Decision 30 (user, 3 Oct 2026): V2G bids in as DSR centrally; V2G
#: netted out and unpaid is the low case, no fresh new build the high case.


def cm_figures(year: int, root=".") -> dict:
    """The CM figures the line carries, from ``cm_calendar_<year>_summary.json``."""
    summary = json.loads((Path(root) / f"resources/market/capacity_market/cm_calendar_{int(year)}_summary.json")
                         .read_text(encoding="utf-8"))
    by_case = summary[f"calendar_{int(year)}_cost_by_case_gbp"]
    book = summary.get("contract_book") or {}
    return {"central_gbp": float(summary[f"calendar_{int(year)}_cost_gbp"]),
            "book_gbp": float(summary.get("contract_book_gbp") or 0.0),
            "auction_gbp": float(summary.get("auction_cost_gbp", summary[f"calendar_{int(year)}_cost_gbp"])),
            "book_derated_mw": dict(book.get("by_delivery_year_derated_mw") or {}),
            "low_gbp": float(by_case["v2g_outside_cm_unpaid"]), "low_case": "v2g_outside_cm_unpaid",
            "high_gbp": float(by_case["no_fresh_new_build"]), "high_case": "no_fresh_new_build",
            "target_mw": float(summary["auction"]["target_mw"]), "supply_mw": float(summary["auction"]["supply_total_mw"]),
            "price_gbp_per_kw_year_2025": float(summary["price_gbp_per_kw_year_2025"]),
            "payments_by_payee_gbp": {k: float(summary["payments_by_payee_gbp"][k]) for k in
                                      ("v2g_owners_households", "dsr_residential", "dsr_industrial_commercial",
                                       "register_plant_and_interconnectors")}}


#: User, 3 Oct 2026 (item 6): TNUoS and DUoS follow Y12's network roll-forward
#: (``network_rav_roll_forward.py`` through the fes_ht_2050 network account), pinned here and
#: kept equal to the account by tests/unit/test_fes2050_decisions_42_to_48.py. The revenue
#: driver is the building-block return plus regulatory depreciation, RAV x allowed WACC
#: (CPIH-real) + depreciation, in the calendar year against 2030. The distribution path's 2030
#: anchor row carries no depreciation, so its driver is the RAV ratio, which the roll-forward
#: sets equal to the FES peak ratio (elasticity 1).
#: FES 2050 decision 65 (user, 3 Oct 2026): NESO's internal BSUoS and incentive in FY2024/25, the
#: twelve monthly "Estimated Internal BSUoS & ESO Incentive" lines of NESO's monthly BSUoS forecasts
#: (GBP 38.76 m to 61.18 m a month; price-control based, not settled). Read 3 Oct 2026; see
#: data/public_ownership/financing/ancillary_services_2050.csv for the services themselves.
NESO_INTERNAL_BSUOS_2024_25 = {
    "gbp": 595.6e6,
    "source": "NESO, Monthly BSUoS forecasts (Apr-24 and Aug-24 editions), 'Estimated Internal BSUoS & ESO Incentive', FY2024/25",
    "source_url": "https://www.neso.energy/document/304841/download | https://www.neso.energy/document/321376/download",
}
ANCILLARY_2050 = "data/public_ownership/financing/ancillary_services_2050.csv"  # the template's; read per year
BSUOS_SPLIT_2024_25 = "data/market/neso_balancing_costs/bsuos_constraint_split_2024_25.json"


def bsuos_rebase_2050(root, ancillary: str = ANCILLARY_2050) -> dict:
    """Decision 65: the 2050 BSUoS lines' services part from the bottom-up ancillary table, the rest demand-scaled.

    ``bsuos_other_services``: NESO's FY2024/25 internal cost and incentive and its non-constraint
    trade categories, less the reserve trades the ancillary table carries, held flat per MWh of final
    (metered) demand; the contracted services and the reserve trades are the
    table's 2050 spend. ``bsuos_voltage_and_stability_constraints``: the table's voltage and inertia
    BM actions. Each services figure is :func:`support_lines_2050.services_held_by_line`, the
    money the register and the providers outside it are paid, so the two reconcile exactly.
    """
    import pandas as pd
    from scripts.public_ownership import support_lines_2050 as sl

    root = Path(root)
    table = sl.load_ancillary(root / ancillary)
    split = json.loads((root / BSUOS_SPLIT_2024_25).read_text(encoding="utf-8"))
    volume = float(split["chargeable_final_demand_mwh"])
    categories = float(split["non_constraint_cost_gbp"]) - float(split["internal_and_contracted_residual_gbp"])
    first = table.groupby("service").first()
    trades_outturn = float(first.loc[first["in_bill_line"].eq(sl.LINE_CATEGORIES), "outturn_spend_gbp_m"].sum()) * 1e6
    held = {case: sl.services_held_by_line(table, case) for case in sl.CASES}
    internal = float(NESO_INTERNAL_BSUOS_2024_25["gbp"])
    return {
        "volume_mwh": volume, "categories_gbp": categories, "trades_outturn_gbp": trades_outturn,
        "internal_gbp": internal,
        "other_rate_gbp_per_mwh": (internal + categories - trades_outturn) / volume,
        "services": {line: {case: held[case][line] for case in sl.CASES} for line in sl.LINE_SERVICES_HELD},
    }


NETWORK_BLOCKS = "resources/analysis/network_price_control_building_blocks.json"


def network_drivers(year: int, root=".") -> dict:
    """Y12's roll-forward to ``year`` (the network account's own function and anchors): RAV and depreciation."""
    from scripts.public_ownership.network_finance_account import roll_forward_result

    extracted = json.loads((Path(root) / NETWORK_BLOCKS).read_text(encoding="utf-8"))
    dist, trans = extracted["distribution"], extracted["transmission"]
    blocks = {"distribution": {"price_basis_key": "2020/21",
                               "closing_rav_gbp_m_2027_28": dist["closing_rav_gbp_m_2027_28"],
                               "net_additions_gbp_m_2027_28": dist["net_additions_gbp_m_2027_28"],
                               "depreciation_gbp_m_2027_28": dist["depreciation_gbp_m_2027_28"]},
              "transmission": {"price_basis_key": "2023/24",
                               "total_rav_gbp_m": trans["implied_total_rav_gbp_m_period_average"]}}
    inputs = f"data/market/network_cost_sources/rav_roll_forward_{int(year)}.yaml"
    result = roll_forward_result(blocks, {"roll_forward_inputs": str(Path(root) / inputs), "roll_forward_years": 2.5},
                                 int(year))
    t = {int(r["year"]): r for r in result["transmission_path"]}
    d = {int(r["year"]): r for r in result["distribution_path"]}
    years = (2030, int(year) - 1, int(year))
    return {
        "transmission": {"allowed_wacc_cpih_real": 0.0457,
                         "rav_gbp": {y: float(t[y]["closing_rav_gbp"]) for y in years},
                         "depreciation_gbp": {y: float(t[y]["depreciation_gbp"]) for y in years}},
        "distribution": {"rav_gbp": {y: float(d[y]["closing_rav_gbp"]) for y in years}},
        "source": f"resources/public_ownership/fes_ht_2050_network_finance_account.json roll_forward (transmission_path, "
                  f"distribution_path); {inputs}",
    }


#: The 2030 DUoS line's FY2030/31 value the 2050 line is carried from (GBP at 3,100 kWh, Oct-Dec 2026
#: money: cap 17a on the RIIO-ED2 path at the ED2 RAV growth, plus the ED3 return uplift, 29 Sep 2026).
#: Pinned when the 2030 line moved to CP30's distribution capex (user decision, 6 Oct 2026).
DUOS_FY2030_31_BASE_2050 = 140.810632


def y12_driver(network: dict, sector: str, year: int) -> float:
    """Y12's revenue driver in ``year`` against 2030 (transmission: return + depreciation; distribution: RAV)."""
    block = network[sector]
    y = int(year)
    if sector == "transmission":
        w = float(block["allowed_wacc_cpih_real"])
        value = lambda t: block["rav_gbp"][t] * w + block["depreciation_gbp"][t]  # noqa: E731
        return value(y) / value(2030)
    return block["rav_gbp"][y] / block["rav_gbp"][2030]


#: F18 (6 Oct 2026) moved the 2030 bill to two recovery volumes, Ofgem's household loss factor, Ofgem's
#: direct-fuel formula, EBIT and headroom on the bill's own costs and direct-debit levelisation. The 2050
#: bill stays on the item D1 basis (its loads exclude losses; user, 6 Oct 2026) until the 2050 session
#: decides, so the fields the builder inherits from the 2030 files keep their pre-F18 values here. The
#: D1 recovery basis itself is read from the 2030 recovery file's ``superseded_d1_basis``.
PRE_F18_2030 = {
    "cfd_formula": ("annual PyPSA CfD payment on balancing dispatch / CfD eligible supplier sales * 3.1 MWh, "
                    "rebased to GBP real 2025"),
    "ldes_formula": ("LDES floor top-up = sum over long-duration storage (carriers in "
                     "public_ownership.consumer_bill.ldes_cap_and_floor.carriers) of max(0, floor - market revenue), "
                     "floor = required revenue with the capital charge at a debt-like real rate; x currency rebasing / "
                     "final demand (metered supply) x 3.1 MWh."),
    "ldes_recovery_denominator": "final_demand_mwh",
    "bsuos_formula": ("(accepted offer cost + accepted bid cost - feasibility slack - non-participant penalty turn-down "
                      "+ interconnector re-trading, plus CfD curtailment compensation only under add_post_hoc) / final "
                      "demand (metered supply) * 3.1 MWh, rebased to GBP real 2025"),
    "bsuos_recovery_denominator": "final_demand_mwh (metered supply, item D1) from the case recovery-demand assumptions file",
    "losses": {"formula": "(loss_multiplier - 1) * domestic energy cost", "source_column": "loss_multiplier_gb_average"},
    "supplier_label": "Supplier costs and margin (Ofgem cap allowances, Oct-Dec 2026)",
    "supplier_method": ("Sum of Ofgem's post-July-2025 supplier allowances for a single-rate, other-payment-method "
                        "customer, GB average of the 14 regions, at 3,100 kWh from the nil and 2,500 kWh benchmark values "
                        "(standing plus unit): cost to serve 110.43; debt related costs 31.58; industry charges 21.51; "
                        "ebit 27.38; headroom 10.95 = GBP201.85 (Oct-Dec 2026 money), held flat in real terms to 2030. "
                        "Smart metering is its own line."),
    "supplier_limitation": ("EBIT and headroom are percentage allowances in Ofgem's method, so they scale with the bill; "
                            "they are held here at their Oct-Dec 2026 cash value, which overstates them by about GBP4 if "
                            "the 2030 bill is about 10% below the Oct-Dec 2026 cap."),
    "payment_method_scope": ("Average across payment methods, so the standing-charge levelisation between payment methods "
                             "(Ofgem Annex 9) nets to zero. A direct-debit customer pays about GBP3.5 a year more (Annex 9 "
                             "v1.11, Oct 2026, average of 14 regions)."),
}


def _pre_f18_rules(rules: dict, loss_factor: float) -> dict:
    """The 2030 model-linked rules with the F18 changes taken back out, for the D1-basis 2050 bill."""
    rules["cfd"]["formula"] = PRE_F18_2030["cfd_formula"]
    rules["ldes_cap_and_floor"]["formula"] = PRE_F18_2030["ldes_formula"]
    rules["ldes_cap_and_floor"]["recovery_denominator"] = PRE_F18_2030["ldes_recovery_denominator"]
    rules["bsuos_constraints"]["formula"] = PRE_F18_2030["bsuos_formula"]
    rules["bsuos_constraints"]["recovery_denominator"] = PRE_F18_2030["bsuos_recovery_denominator"]
    bridge = rules["direct_fuel"]["procurement_bridge"]
    bridge.pop("rate_composition", None)
    bridge.pop("rate_composition_note", None)
    # The procurement losses multiplier must equal the D1 recovery file's loss factor.
    bridge["allowances"]["losses"] = {"loss_multiplier_gb_average": loss_factor, **PRE_F18_2030["losses"]}
    return rules


def _pre_f18_supplier(block: dict) -> dict:
    """The 2030 supplier line without F18's levelisation and EBIT and headroom recompute (2050 holds the cash)."""
    block = copy.deepcopy(block)
    block["derivation"].pop("direct_debit_levelisation", None)
    block["derivation"].pop("ebit_and_headroom", None)
    block["label"] = PRE_F18_2030["supplier_label"]
    block["method"] = PRE_F18_2030["supplier_method"]
    block["limitations"] = [PRE_F18_2030["supplier_limitation"]] + list(block["limitations"][1:])
    return block


#: CP30 2030's ES1 demand excluding electrolysis, GWh ("Demand Total (GWh) - Excluding Electrolysis"; checked against the
#: 2030 network in resources/CP30/validation/CP30_FFR_2030_energy_totals.csv), the base of the renewable share.
CP30_2030_DEMAND_EXCL_ELECTROLYSIS_GWH = 315161.95


def wind_solar_share_factor(year: int, root: str | Path = ".") -> dict:
    """Wind and solar's share of demand in ``year`` over its 2030 share (CP30): the imbalance allowance's driver.

    Output is the driver of the central-dispatch forecast-error charge (decision 18,
    :func:`scripts.sources.new_fes_year.fleet_ratio`: ES1 wind and solar output with non-networked offshore, over CP30
    2030's 273.0 TWh). The allowance is a rate on the energy cost, which already moves with demand and price, so the
    rate takes the output per MWh of demand (ED1 system demand excluding electrolysis; CP30's ES1 equivalent in 2030).
    """
    import json as _json

    from scripts.market.forecast_error_charge import FTI_FLEET_GROWTH
    from scripts.sources import new_fes_year

    path = Path(root) / f"resources/FES/fes2025_holistic_transition_{int(year)}_targets.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: the year's ES1 targets, needed for the imbalance allowance's scaling")
    targets = _json.loads(path.read_text(encoding="utf-8"))
    output_twh = new_fes_year.fleet_ratio(int(year)) * FTI_FLEET_GROWTH / new_fes_year.FTI_FLEET_RATIO         * new_fes_year.CP30_WIND_SOLAR_TWH
    demand_twh = float(targets["demand_gwh"]) / 1e3
    share = output_twh / demand_twh
    share_2030 = new_fes_year.CP30_WIND_SOLAR_TWH / (CP30_2030_DEMAND_EXCL_ELECTROLYSIS_GWH / 1e3)
    return {"factor": share / share_2030, "wind_solar_twh": output_twh, "demand_twh": demand_twh,
            "demand_line": str(targets.get("demand_line", "")), "share": share, "share_2030": share_2030}


def _scale_imbalance_with_wind_and_solar(rules: dict, year: int, root: str | Path) -> None:
    """User decision (16 Nov 2026): the market's forecast-error cost follows wind and solar, as central dispatch's does.

    Under market dispatch the forecast-error cost on the bill is the supplier imbalance allowance; under central
    dispatch (S3's I3) it is the charge borrowed from the 2026 REMA scheduling and dispatch assessment for NESO, scaled
    by the year's wind and solar output (decision 18). The allowance held its 2030 intensity. It is a rate on the
    energy cost (already moving with demand and price), so its rate now scales with wind and solar's share of demand
    relative to 2030 (the user's choice of form; scaling the rate by output alone would count demand growth twice).
    2030 is unchanged.
    """
    imbalance = rules["direct_fuel"]["procurement_bridge"]["allowances"]["imbalance"]
    s = wind_solar_share_factor(year, root)
    rate_2030 = float(imbalance["rate"])
    imbalance["rate"] = rate_2030 * s["factor"]
    imbalance["formula"] = "rate_2030 * wind_solar_share_factor * domestic energy cost"
    imbalance["wind_solar_scaling"] = {
        "rate_2030": rate_2030,
        "wind_solar_share_factor": round(s["factor"], 6),
        "wind_solar_twh": round(s["wind_solar_twh"], 3),
        "demand_twh": round(s["demand_twh"], 3),
        "share": round(s["share"], 6),
        "share_2030": round(s["share_2030"], 6),
        "basis": (f"ES1 {int(year)} wind and solar output (with non-networked offshore) over {s['demand_line']}, against "
                  "CP30 2030's 273.0 TWh over its 315.16 TWh demand excluding electrolysis. User decision 16 Nov 2026: "
                  "the market's forecast-error cost (this allowance) follows wind and solar as the central-dispatch "
                  "charge does (decision 18), in the form of a rate (share of demand, not output alone)"),
    }


#: Ofgem's cap-and-floor regime term (OFG1164 Interconnector Cap and Floor Handbook: 25 years from operation).
IC_REGIME_YEARS = 25
#: The calendar year the RO no longer runs in (support ends 31 March 2037).
RO_CLOSED_FROM = 2038
#: Sizewell C operates from about 2039 (NAO, 20 May 2026); before, its levy is the construction-phase RAB.
SIZEWELL_C_OPERATING_FROM = 2039
#: Every Feed-in Tariff has ended (the last 20-year tariffs end about 2039-41).
FIT_ENDED_FROM = 2042
#: Net Zero Teesside Power's DPA (payments from 2028/29, 15 years).
NZT_DPA_LIVE_UNTIL = 2043
#: Text the builder must keep at 2050 for any year (protected from :func:`year_text`).
FIXED_2050 = "\x02FIXED\x02"


def outputs(year: int) -> dict:
    return {"assumptions": f"data/market/consumer_bill_benchmarks/{int(year)}_pypsa_informed_assumptions.json",
            "recovery": f"data/market/consumer_bill_forecast/{int(year)}_recovery_demand_assumptions.json",
            "ro": f"data/market/consumer_bill_forecast/{int(year)}_ro_scheme_parameters.json"}


OUTPUTS = outputs(2050)


#: Task SETTLE2050: the 2050 bill settles the vintaged register's own contracts. The strike
#: table (Y11's strikes in the published format, ``contract_book_2050.py strike-table``), the
#: fes_ht_2050 register's contract book (``contract_book_2050.py book``) and the support
#: settlement's totals (``build_asset_settlement.py --support-summary``) are rule outputs.
STRIKE_TABLE_2050 = "resources/public_ownership/FES25_HT_2050_cfd_strike_table.csv"
CONTRACT_BOOK_2050 = "resources/public_ownership/fes_ht_2050_support_contract_book.csv"
#: Decision 41: the strikes the solved BM bid on (decision 49 (b): the calibrated contract table).
BM_STRIKE_TABLE_2050 = "resources/public_ownership/FES25_HT_2050_cfd_bm_strike_table.csv"
#: Decision 41 (c): the public account whose private case carries Sizewell C's RAB allowance.
PUBLIC_ACCOUNT_2050 = "resources/public_ownership/fes_ht_2050_public_account_summary.json"
SUPPORT_SUMMARY_2050 = "resources/public_ownership/fes_ht_2050_asset_settlement_support.json"


def build(year: int, root: str | Path = ".") -> dict[str, dict]:
    """The three documents of model year ``year``, keyed as :func:`outputs`.

    Args:
        year: the calendar model year.
        root: the repository root the 2030 bill files are read from.

    Returns:
        ``{"assumptions": ..., "recovery": ..., "ro": ...}``.
    """
    from scripts.consumer_bill import fes_bill_figures

    Y = int(year)
    if Y < RO_CLOSED_FROM:
        raise ValueError(f"{Y}: the Renewables Obligation still runs (support ends 31 March 2037); this builder "
                         "writes the closed scheme only")
    if Y < SIZEWELL_C_OPERATING_FROM:
        raise ValueError(f"{Y}: Sizewell C is in construction; the line here is its operating-phase RAB allowance")
    ROOT = Path(root)
    F = fes_bill_figures.read(Y, ROOT)
    NET = network_drivers(Y, ROOT)
    CM_Y = cm_figures(Y, ROOT)
    ancillary = f"data/public_ownership/financing/ancillary_services_{Y}.csv"
    A2030 = json.loads((ROOT / "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json").read_text(encoding="utf-8"))
    R2030 = json.loads((ROOT / "data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json").read_text(encoding="utf-8"))
    S30 = {**A2030["sourced_externals"],
           "supplier_costs_and_margin": _pre_f18_supplier(A2030["sourced_externals"]["supplier_costs_and_margin"])}
    # F18: the 2030 file moved to the two-volume basis; 2050 stays on item D1, recorded there.
    D1 = R2030["superseded_d1_basis"]
    WB ="data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx"
    BB = "resources/FES/FES_2025_data.csv"
    PLAN = ("FES 2050 public-ownership work, task Y13 (decisions 12, 12a)" if Y == 2050 else
            "FES " + FIXED_2050 + " public-ownership work, task Y13 (decisions 12, 12a), made year-generic for FES 2040")
    ACCESS = "2026-09-30" if Y == 2050 else "2026-11-11"  # the day the year's FES figures were first read

    # ── FES 2025 Holistic Transition. FES year Y = fiscal year April Y to March Y+1 ──
    PEAK_GW = F["peak_gw"]                    # ED1 row 134
    CUSTOMERS = F["customers"]                # Dem_BB001a, HT, sum of GSPs
    CONSUMER_GWH = F["consumer_gwh"]          # ED1 rows 14+79+84+119
    fy_prev, fy_this = f"FY{Y - 1}/{Y % 100:02d}", f"FY{Y}/{(Y + 1) % 100:02d}"
    CAL = {fy_prev: 0.25, fy_this: 0.75}      # calendar Y = 3/12 FY(Y-1) + 9/12 FY(Y)
    FY = {fy_prev: Y - 1, fy_this: Y}


    def ratios(y):
        return {"peak": PEAK_GW[y] / PEAK_GW[2030], "customers": CUSTOMERS[y] / CUSTOMERS[2030],
                "consumer_demand": CONSUMER_GWH[y] / CONSUMER_GWH[2030]}


    DRIVERS = {
        "source": f"{WB}: ED1 row 134 (GBFES Peak Customer Demand: Total Consumption plus Losses, GW), ED1 rows 14, 79, 84 "
                  f"and 119 (consumer demand excluding electrolysis, GWh, via scripts/FES/fes2025_targets.py ed1_demand); "
                  f"{BB}: Dem_BB001a (number of domestic customers), Holistic Transition, summed over GSPs",
        "year_basis": "FES year Y is fiscal year April Y to March Y+1 (ED1 'Annual [Fiscal]'; ED1 row 134 is the winter peak of that year)",
        "peak_gw": {str(k): v for k, v in PEAK_GW.items()},
        "domestic_customers": {str(k): v for k, v in CUSTOMERS.items()},
        "consumer_demand_excluding_electrolysis_gwh": {str(k): v for k, v in CONSUMER_GWH.items()},
        "ratios_to_fes_2030": {str(y): ratios(y) for y in (Y - 1, Y)},
    }
    REB_2026_CAP = copy.deepcopy(S30["supplier_costs_and_margin"]["rebasing"])  # Oct-Dec 2026 cap to 2025
    REB_2030_31 = copy.deepcopy(S30["tnuos"]["period_blend"][1]["rebasing"])     # FY2030/31 money to 2025
    REB_2026_27 = copy.deepcopy(S30["aahedc"]["rebasing"])                      # FY2026/27 money to 2025


    def base_line(key, **extra):
        block = {"unit": "gbp_per_household_year", "consumption_basis_kwh": 3100,
                 "stack_group": S30[key]["stack_group"], "source_class": "external_primary_source",
                 "reference_gbp_per_3100kwh_ex_vat": None}
        block.update(extra)
        return block


    def held_from_2030(key, why, limitations_add=()):
        """A 2030 line whose source value is held flat in real terms again for 2050 (declared, not inherited)."""
        b = copy.deepcopy(S30[key])
        b["reference_gbp_per_3100kwh_ex_vat"] = None
        b["method"] = b["method"].replace(" to 2030", "").rstrip(".") + ". 2050 (task Y13): held flat in real terms from the same source period; " + why
        b["limitations"] = list(b.get("limitations") or []) + list(limitations_add)
        b["period_basis"] = "held_flat_real"
        b.pop("scaling_2030", None)
        b["y13_basis"] = "held_flat_real from the 2030 line's own source period, re-declared for 2050"
        return b


    se = {}

    # 1-2. BSUoS non-constraint and voltage/stability (decision 65, 3 Oct 2026): the services part of
    # each line is the bottom-up 2050 ancillary spend (the money the register's support line and the
    # providers outside it are paid), NESO's internal cost and the other trade categories held flat per
    # MWh of the GSP volume as before. Low and high are a named bill sensitivity (sensitivity_cost_gbp).
    bs = bsuos_rebase_2050(ROOT, ancillary)
    rebase_extrapolation = {"rule": "bottom-up 2050 ancillary services (central) plus the FY2024/25 per-MWh rate of the parts that are not services",
                            "from_period": "FY2024/25 NESO BSUoS outturn"}
    other30, volt30 = S30["bsuos_other_services"], S30["bsuos_voltage_and_stability_constraints"]
    # The 2030 lines were rebased the same way on 3 Oct 2026 (OW16 follow-up); their FY2024/25 figures moved
    # under derivation.fy2024_25. Either shape gives the same numbers.
    other30_fy = (other30.get("derivation") or {}).get("fy2024_25") or {}
    other30_rate = float(other30_fy.get("non_constraint_gbp_per_mwh", other30.get("value")))
    volt30_fy = (volt30.get("derivation") or {}).get("fy2024_25") or volt30["derivation"]
    services_other = bs["services"]["bsuos_other_services"]
    services_volt = bs["services"]["bsuos_voltage_and_stability_constraints"]
    se["bsuos_other_services"] = base_line(
        "bsuos_other_services", price_year=2024, rebasing=copy.deepcopy(A2030["currency_rebasing"]),
        label="BSUoS non-constraint services (bottom-up 2050 ancillary services plus NESO internal cost and trade categories)",
        source_file=ancillary, source_url=other30["source_url"], access_date="2026-10-03",
        source_table=f"{ancillary} (in_bill_line bsuos_other_services and balancing_categories); {BSUOS_SPLIT_2024_25}",
        method=("FES 2050 decision 65 (user, 3 Oct 2026). The line held three things in FY2024/25 (GBP %.1f m over %.0f MWh): "
                "NESO's internal cost and incentive (GBP %.1f m), the daily trade categories (GBP %.1f m) and contracted "
                "ancillary services (the rest). The services are now the bottom-up 2050 spend of the ancillary table "
                "(frequency response, reserve, stability, reactive power, restoration, and the reserve trades the "
                "categories held, GBP %.1f m centrally), the money scripts/public_ownership/support_lines_2050.py pays "
                "the providers, in the register and outside it, so the two reconcile exactly. The internal cost and the "
                "categories less those reserve trades (GBP %.1f m in FY2024/25) stay held flat per MWh of final demand, "
                "GBP %.4f/MWh, on 2050 final (metered) demand: BSUoS has been charged on final demand only since "
                "April 2023, and NESO's FY2024/25 chargeable volume (275.15 TWh) is metered supply (LCCC's eligible "
                "sales were 277.0 TWh), not a loss-inclusive GSP volume (review 2, 3 Oct 2026). Spread over 2050 "
                "metered supply." % (
                    other30_rate * bs["volume_mwh"] / 1e6, bs["volume_mwh"], bs["internal_gbp"] / 1e6,
                    bs["categories_gbp"] / 1e6, services_other["central"] / 1e6,
                    (bs["categories_gbp"] - bs["trades_outturn_gbp"]) / 1e6, bs["other_rate_gbp_per_mwh"])),
        derivation={"cost_gbp": services_other["central"], "recovery_denominator": "final_demand_mwh",
                    "demand_scaled_gbp_per_mwh": bs["other_rate_gbp_per_mwh"],
                    # Review 2: the FY2024/25 rate is per MWh of final (metered) demand, so is its 2050 base.
                    "demand_scaled_volume": "final_demand_mwh",
                    "ancillary_case": "central", "ancillary_file": ancillary,
                    "sensitivity_cost_gbp": {"ancillary_low": services_other["low"], "ancillary_high": services_other["high"]},
                    "fy2024_25": {"non_constraint_gbp": other30_rate * bs["volume_mwh"],
                                  "internal_and_incentive_gbp": bs["internal_gbp"],
                                  "internal_source": NESO_INTERNAL_BSUOS_2024_25["source"],
                                  "internal_source_url": NESO_INTERNAL_BSUOS_2024_25["source_url"],
                                  "trade_categories_gbp": bs["categories_gbp"],
                                  "reserve_trades_in_ancillary_table_gbp": bs["trades_outturn_gbp"],
                                  "chargeable_volume_mwh": bs["volume_mwh"]}},
        limitations=["The services are a bottom-up projection (sources and assumptions per row in the ancillary table); NESO publishes nothing past 2026-27.",
                     "NESO's internal cost is held flat per MWh; its Oct-2026 forecast runs higher (about GBP 66.5 m a month).",
                     "Low and high are a named bill sensitivity; the register's support line stays on the central case."],
        period_basis="extrapolated_declared", extrapolation=dict(rebase_extrapolation),
    )
    se["bsuos_voltage_and_stability_constraints"] = base_line(
        "bsuos_voltage_and_stability_constraints", price_year=2024, rebasing=copy.deepcopy(A2030["currency_rebasing"]),
        label="BSUoS voltage and stability actions (bottom-up 2050 voltage and inertia BM actions)",
        source_file=ancillary, source_url=volt30["source_url"], access_date="2026-10-03",
        source_table=f"{ancillary} (in_bill_line bsuos_voltage_and_stability_constraints)",
        method=("FES 2050 decision 65 (user, 3 Oct 2026): the line is the voltage and inertia actions NESO takes in the BM, "
                "the bottom-up 2050 spend of the ancillary table (GBP %.1f m centrally), the money "
                "scripts/public_ownership/support_lines_2050.py pays the providers, so the two reconcile exactly. The "
                "modelled redispatch (bsuos_constraints) is thermal only: a DC power flow has no voltage or inertia "
                "constraint, so this cost has no other home in the bill (item D4(d)). Spread over 2050 metered supply."
                % (services_volt["central"] / 1e6)),
        derivation={"cost_gbp": services_volt["central"], "recovery_denominator": "final_demand_mwh",
                    "ancillary_case": "central", "ancillary_file": ancillary,
                    "sensitivity_cost_gbp": {"ancillary_low": services_volt["low"], "ancillary_high": services_volt["high"]},
                    "fy2024_25_category_minus_thermal_gbp": volt30_fy["non_thermal_constraints_gbp_2024_25"]},
        limitations=["The 2050 voltage and inertia need of an inverter-dominated system is a declared projection (held at the FY2024/25 outturn centrally).",
                     "Low and high are a named bill sensitivity; the register's support line stays on the central case."],
        period_basis="extrapolated_declared", extrapolation=dict(rebase_extrapolation),
    )

    # 3. Supplier costs and margin, 4. smart meter: Ofgem cap 17a held flat real.
    se["supplier_costs_and_margin"] = held_from_2030(
        "supplier_costs_and_margin",
        "a declared real-terms hold of Ofgem's post-July-2025 allowances; no source reaches 2050.",
        ["2050: EBIT and headroom are percentage allowances, so they would move with the 2050 bill; held at their 2026 cash value.",
         "2050: the heat-pump plus EV household is charged the same supplier allowances per 3,100 kWh-equivalent unit rate "
         "and standing charge as every other household (single rate)."])
    se["smart_meter_net_cost_change"] = held_from_2030(
        "smart_meter_net_cost_change",
        "the rollout is long complete by 2050 and the allowance is a net change against Ofgem's baseline; the second-"
        "generation meter replacement cycle is not modelled.")

    # 5. TNUoS: FY2030/31 carried on Y12's transmission roll-forward (user, 3 Oct 2026).
    tn = S30["tnuos"]["derivation"]
    res_2030_31 = float(tn["domestic_residual_gbp_per_year"])
    loc_2030_31 = float(tn["locational_gbp_per_year"])
    rev_2030_31 = float(tn["demand_residual_revenue_gbp_m_2030_31"])
    tn_fy = {}
    for label, y in FY.items():
        r = ratios(y)
        driver = y12_driver(NET, "transmission", y)
        residual = res_2030_31 * driver / r["customers"]
        tn_fy[label] = {"domestic_residual_gbp_per_year": residual, "locational_gbp_per_year": loc_2030_31,
                        "total_gbp_per_year": residual + loc_2030_31,
                        "demand_residual_revenue_gbp_m": rev_2030_31 * driver, "y12_revenue_driver": driver}
    tn_cal = {k: sum(CAL[p] * tn_fy[p][k] for p in CAL) for k in tn_fy[fy_this]}
    se["tnuos"] = base_line(
        "tnuos", value=tn_cal["total_gbp_per_year"], price_year=2030, rebasing=REB_2030_31,
        label="TNUoS (NESO 2030/31 carried to calendar 2050 on Y12's transmission roll-forward)",
        source_file=S30["tnuos"]["source_file"], source_title=S30["tnuos"]["source_title"], source_url=S30["tnuos"]["source_url"],
        access_date=S30["tnuos"]["access_date"],
        method=("Declared extrapolation on Y12's network account (user, 3 Oct 2026). The NESO five-year view's FY2030/31 "
                "domestic residual (GBP119.62 a site a year) and average locational tariff (GBP4.00 at 3,100 kWh), in FY2030/31 "
                "money, are carried to FY2049/50 and FY2050/51: the demand residual revenue grows with transmission allowed "
                "revenue's building blocks on Y12's roll-forward (RAV x the RIIO-3 CPIH-real WACC 4.57% + regulatory "
                "depreciation, against 2030) and is levied per site, so the domestic residual grows with that over domestic "
                "customers (Dem_BB001a); the locational tariff is held flat real per kWh. Calendar 2050 = 0.25 FY2049/50 + "
                "0.75 FY2050/51, deflated with the FY2030/31 factor."),
        derivation={
            "rule": "residual_Y = residual_FY2030/31 x D_Y / (customers_Y / customers_2030); revenue_Y = revenue_FY2030/31 x D_Y; D_Y = (RAV_Y x WACC + depreciation_Y) / (RAV_2030 x WACC + depreciation_2030) on Y12's transmission roll-forward; locational held",
            "y12_network": {"transmission": NET["transmission"], "source": NET["source"]},
            "base_fy2030_31": {"domestic_residual_gbp_per_year": res_2030_31, "locational_gbp_per_year": loc_2030_31,
                               "demand_residual_revenue_gbp_m": rev_2030_31,
                               "source": "2030_pypsa_informed_assumptions.json sourced_externals.tnuos.derivation (NESO FYV v3.0 Tables 14, 15)"},
            "drivers": DRIVERS,
            "by_fiscal_year": tn_fy,
            "domestic_residual_gbp_per_year": tn_cal["domestic_residual_gbp_per_year"],
            "total_gbp_per_year": tn_cal["total_gbp_per_year"],
            f"demand_residual_revenue_gbp_m_{Y}_{(Y + 1) % 100:02d}": tn_fy[fy_this]["demand_residual_revenue_gbp_m"],
            "calendar_blend": {"domestic_residual_gbp_per_year": tn_cal["domestic_residual_gbp_per_year"],
                               "demand_residual_revenue_gbp_m": tn_cal["demand_residual_revenue_gbp_m"],
                               "weights": CAL,
                               "note": "Calendar-2050 blend of the domestic residual and the demand residual revenue, the ratio the interconnector credit is spread on (tnuos_household_share)"},
        },
        limitations=["No official TNUoS projection exists after 2030/31 (NESO's five-year view ends there); Y12's RAV roll-forward is itself declared (RIIO-3 plus Beyond 2030, then capex = depreciation).",
                     "Fast money (opex) and the tax allowance are assumed to move with the return and depreciation blocks.",
                     "Domestic customers stand in for all chargeable sites; the non-domestic site count is not projected."],
        period_basis="extrapolated_declared",
        extrapolation={"from_period": "NESO FY2030/31 (five-year view v3.0)", "rule": "Y12 transmission building blocks over domestic customers (residual); locational held flat real",
                       "to_periods": list(CAL)},
    )

    # 6. DUoS: FY2030/31 carried on Y12's distribution roll-forward (user, 3 Oct 2026).
    du = S30["duos"]
    # FY2030/31, Oct-Dec 2026 money. Held at the 2030 line's value as committed for 2050 (the ED2-RAV-growth
    # path): the 2030 line moved to CP30's distribution capex (user decision, 6 Oct 2026) and the 2050 line
    # has not been re-decided, so the 2050 file stays as committed.
    du_base = DUOS_FY2030_31_BASE_2050
    fixed_share = float(du["derivation"]["fixed_gbp_per_year_2026"]) / float(du["derivation"]["base_gbp_per_year_2026"])
    du_fy = {}
    for label, y in FY.items():
        r = ratios(y)
        driver = y12_driver(NET, "distribution", y)
        factor = fixed_share * driver / r["customers"] + (1.0 - fixed_share) * driver / r["consumer_demand"]
        du_fy[label] = {"value": du_base * factor, "factor": factor, "y12_revenue_driver": driver}
    du_cal = sum(CAL[p] * du_fy[p]["value"] for p in CAL)
    se["duos"] = base_line(
        "duos", value=du_cal, price_year=2026, rebasing=REB_2026_CAP,
        label="DUoS (the 2030/31 path carried to calendar 2050 on Y12's distribution roll-forward)",
        source_file=du["source_file"], source_url=du["source_url"], access_date=du["access_date"],
        source_table=du["source_table"], source_metadata=du["source_metadata"],
        method=("Declared extrapolation on Y12's network account (user, 3 Oct 2026). The 2030 line's FY2030/31 value (GBP140.81 at 3,100 kWh, Oct-Dec 2026 money: Ofgem "
                "cap 17a on the RIIO-ED2 path plus the ED3 allowed-return uplift) is carried to FY2049/50 and FY2050/51. Distribution "
                "revenue grows with Y12's distribution RAV roll-forward (the ED2 roll-forward scaled by the FES HT winter peak at elasticity 1, so the driver equals the peak ratio). Its fixed part (the cap's nil charge, "
                f"{fixed_share:.4f} of the 3,100 kWh charge) is recovered per customer (Dem_BB001a) and its unit part per MWh of FES "
                "consumer demand excluding electrolysis. Calendar 2050 = 0.25 FY2049/50 + 0.75 FY2050/51."),
        derivation={
            "rule": "value_Y = value_FY2030/31 x [f x D_Y/(customers_Y/customers_2030) + (1-f) x D_Y/(demand_Y/demand_2030)]; D_Y = RAV_Y / RAV_2030 on Y12's distribution roll-forward",
            "y12_network": {"distribution": NET["distribution"], "source": NET["source"]},
            "base_fy2030_31_gbp_2026_money": du_base, "fixed_share_f": fixed_share,
            "fixed_share_source": "2030 duos derivation: fixed_gbp_per_year_2026 / base_gbp_per_year_2026 (Ofgem cap 17a nil and 3,100 kWh charges)",
            "drivers": DRIVERS, "by_fiscal_year": du_fy, "calendar_weights": CAL,
        },
        limitations=["RIIO-ED3 has no determinations and nothing exists for ED4 and ED5; Y12's roll-forward is a declared rule, not a forecast.",
                     "The FY2030/31 base carries the ED3 allowed-return uplift measured on the 2030 network account's RAV, held in the ratio.",
                     "The heat-pump plus EV household's own peak contribution is not charged separately on a single-rate tariff."],
        period_basis="extrapolated_declared",
        extrapolation={"from_period": "FY2030/31 of the 2030 DUoS line (Ofgem cap 17a base, RIIO-ED2 path, ED3 return)",
                       "rule": "Y12 distribution RAV; fixed part per customer, unit part per MWh", "to_periods": list(CAL)},
    )

    # 7. Capacity Market: Y11's 2050 auction model (task E2E, 1 Oct 2026, replacing the interim
    #    peak-scaled 2030 cost). The figures are pinned here and kept equal to the model by
    #    tests/unit/test_fes2050_bill.py (build_capacity_market.py --model-year 2050).
    cm30 = S30["capacity_market"]["derivation"]
    CM_2050 = CM_Y
    if CM_Y["book_gbp"] > 0:
        cm_label = "Capacity Market (calendar 2050: the register's live agreements plus the MR4 auction model on the 2050 fleet)"
        cm_method = ("The 15-year agreements of the 2025/26 to 2029/30 T-4 auctions are still live in 2050: %.0f MW de-rated "
                     "in 2049/2050 and %.0f MW in 2050/2051 in the NESO register, each at its own clearing price CPI-indexed to "
                     "2025, half of each delivery year in the calendar year, matched to the FES stock-flow fleet by carrier "
                     "(GBP %.0f m). The rest is the modelled auction (task Y11's machinery, scripts/market/"
                     "build_capacity_market_year.py): the MR4 demand curve and bid distributions on the FES 2025 HT 2050 fleet, "
                     "with DPA, CfD and RAB plant outside the auction while their support is live and the booked capacity "
                     "outside it as held; agreements of the unheld auctions (2030/31 onward) are taken at its price (steady "
                     "state). V2G bids in as DSR centrally (decision 30). The auction procures %.0f MW de-rated against %.0f MW "
                     "of eligible supply and clears at %.2f GBP/kW/yr (2025 money): GBP %.0f m. Spread over the GSP chargeable "
                     "volume with the 2024 household peak factor."
                     % (CM_Y["book_derated_mw"][f"{Y - 1}/{Y}"], CM_Y["book_derated_mw"][f"{Y}/{Y + 1}"], CM_Y["book_gbp"] / 1e6,
                        CM_Y["target_mw"], CM_Y["supply_mw"], CM_Y["price_gbp_per_kw_year_2025"], CM_Y["auction_gbp"] / 1e6))
    else:
        cm_label = "Capacity Market (calendar 2050, the MR4 auction model on the 2050 fleet, Y11)"
        cm_method = None
    se["capacity_market"] = base_line(
        "capacity_market", price_year=2025, **({"value_basis": "nil"} if CM_2050["central_gbp"] == 0.0 else {}),
        label=cm_label,
        source_file=S30["capacity_market"]["source_file"], source_title=S30["capacity_market"]["source_title"],
        source_url=S30["capacity_market"]["source_url"], access_date=S30["capacity_market"]["access_date"],
        method=("No agreement in the register reaches 2050 (the longest, 15 years from 2030/31, ends in 2045), so the whole "
                "2050 line is the modelled auction (task Y11, scripts/market/capacity_market_2050.py): the MR4 demand "
                "curve and bid distributions on the FES 2025 HT 2050 fleet, with DPA, CfD and RAB plant outside the "
                "auction while their support is live. Decision 30 (3 Oct 2026): V2G bids in as DSR centrally, so the "
                "auction procures %.0f MW de-rated against %.0f MW of eligible supply and clears at %.2f GBP/kW/yr "
                "(2025 money). Spread over the GSP chargeable volume with the 2024 household peak factor."
                % (CM_2050["target_mw"], CM_2050["supply_mw"], CM_2050["price_gbp_per_kw_year_2025"])) if cm_method is None else cm_method,
        derivation={
            "agreement_cost_gbp": CM_2050["central_gbp"], "recovery_denominator": "gsp_chargeable_volume_mwh",
            **({"contract_book_gbp": CM_Y["book_gbp"], "auction_cost_gbp": CM_Y["auction_gbp"]} if CM_Y["book_gbp"] > 0 else {}),
            "household_allocation_factor": float(cm30["household_allocation_factor"]),
            "household_allocation_basis": cm30["household_allocation_basis"],
            "rule": ("calendar_2050_cost_gbp of the 2050 auction model (central case: V2G bids in as DSR, decision 30)"
                     if CM_Y["book_gbp"] == 0 else "calendar_2050_cost_gbp of the year builder: the register's live agreements "
                     "plus the auction (central case: V2G bids in as DSR, decision 30)"),
            "builder": "scripts/market/build_capacity_market_year.py",
            "builder_output": "resources/market/capacity_market/cm_calendar_2050_summary.json",
            "builder_output_key": "calendar_2050_cost_gbp",
            # Decision 30: who the line pays. V2G is consumer-owned (decision 24), outside every
            # ownership case: bill payers pay it and the households that own V2G receive it, the
            # same in every case, so it nets out of every case's change against private.
            "payments_by_payee_gbp": dict(CM_2050["payments_by_payee_gbp"]),
            "consumer_owned_flexibility_payment_gbp": (CM_2050["payments_by_payee_gbp"]["v2g_owners_households"]
                                                       + CM_2050["payments_by_payee_gbp"]["dsr_residential"]),
            "sensitivity_cost_gbp": {"clearing_low": CM_2050["low_gbp"], "clearing_high": CM_2050["high_gbp"]},
            "sensitivity_cases": {"clearing_low": CM_2050["low_case"], "clearing_high": CM_2050["high_case"]},
            "sensitivity_note": (("The model's own cases: low is V2G netted out against the target and unpaid (Y11's "
                                 "old central, decision 30's sensitivity), whose auction clears at zero, leaving the "
                                 "contract book; high is no fresh new build. The cases with a shortfall (V2G not "
                                 "counted, storage at half its EFC) clear at the price cap and are not used.")
                                 if CM_Y["book_gbp"] > 0 else
                                 ("The model's own cases: low is V2G netted out against the target and unpaid (Y11's "
                                 "old central, decision 30's sensitivity), which clears at zero; high is no fresh new "
                                 "build. The cases with a shortfall (V2G not counted, storage at half its EFC) clear "
                                 "at the price cap and are not used.")),
        },
        limitations=["The 2030/31 T-4 de-rating factors are held to 2050 (declared in the model).",
                     "The household peak factor is Ofgem's 2024 ratio; heat pumps and EV charging raise the household share of the winter peak.",
                     "The 2050 fleet is FES ES1 by stock-flow, not the Y10 register (capacity_market_2050.fleet_from_register is not wired)."]
                    + (["New-build storage agreements are matched to batteries and refurbishing ones to pumped storage (declared)."]
                       if CM_Y["book_gbp"] > 0 else []),
        period_basis="calendar_year",
    )

    # 8. Sizewell C RAB (decision 41 (c), 2 Oct 2026): the operating-phase levy is the RAB's allowed
    # revenue less its market revenue, which the vintaged register books to the Sizewell C rows
    # (calculate_financing_scenarios.book_rab_allowed_revenue); the line reads it from the public
    # account, so the bill and the register carry the same money. 2024 money (the register's).
    supply_2025_26 = 253.97e6
    se["sizewell_c_rab"] = base_line(
        "sizewell_c_rab", price_year=2024, rebasing=copy.deepcopy(A2030["currency_rebasing"]),
        label="Sizewell C RAB levy (allowed revenue less market revenue, the 2050 register's RAB allowance)",
        source_file=S30["sizewell_c_rab"]["source_file"], source_url=S30["sizewell_c_rab"]["source_url"],
        access_date=S30["sizewell_c_rab"]["access_date"], source_table=S30["sizewell_c_rab"]["source_table"],
        method=("Decision 41 (c): Sizewell C operates in 2050 (FES HT large nuclear) and its RAB pays the allowed revenue, so "
                "the levy is the allowed revenue (the plant's required revenue at the private rate in the fes_ht_2050 "
                "register: capital, O&M, generator TNUoS and insurance) less its market revenue on the solved year, "
                "booked to the register's RAB rows as support revenue (rent zero, at cost) and read here from the run "
                "group's public account (cases.private.rab_support_gbp), spread over 2050 metered supply. The Ofgem "
                "source is the 2026/27 construction-phase levy, kept as the 2030 comparator."),
        derivation={"cost_from": {"public_account": PUBLIC_ACCOUNT_2050, "case": "private", "key": "rab_support_gbp"},
                    "recovery_denominator": "final_demand_mwh",
                    "construction_phase_reference": {"rate_gbp_per_mwh_oct_dec_2026": 4.6341, "supply_base_mwh": supply_2025_26,
                                                     "cost_gbp_2026_money": 4.6341 * supply_2025_26},
                    "nao_note": "The NAO (20 May 2026) expects the construction-phase charge to reach GBP17-19 a household by opening (2039); after opening, sales revenue offsets allowed revenue."},
        limitations=["The allowed revenue is the register's required revenue (a level annuity of the plant's capex over its remaining life at the "
                     "private rate), not Ofgem's RAB building blocks; the construction-period returns the levy paid before 2039 are not netted.",
                     "A state share of Sizewell C (P2's boundary, decision 32 open) earns the same allowance; the tariff (P3) replaces the line."],
        period_basis="calendar_year",
    )

    # 8b. FES 2050 decision 58: the life-extension operating top-up the register books (support_lines_2050).
    se["life_extension_operating_top_up"] = base_line(
        "sizewell_c_rab", price_year=2024, rebasing=copy.deepcopy(A2030["currency_rebasing"]),
        label="Life-extension operating top-up (expired offshore and onshore wind, unabated biomass, expired BECCS)",
        source_file="scripts/public_ownership/support_lines_2050.py",
        source_url="resources/public_ownership/fes_ht_2050_public_account_summary.json", access_date="2026-10-03",
        source_table="fes_ht_2050 register: life_extension_top_up_gbp on the private case",
        method=("FES 2050 decision 58 (user, 3 Oct 2026): plant past its contract that is life-extended at operating cost only (decisions 41 (a) and 42): expired offshore wind, expired onshore wind (decision 62's life-extended cohorts, added 3 Oct 2026), unabated biomass (no contract in 2050) and expired Power BECCS. Where a row's revenue, ancillary services and every other income included, does not cover its operating cost (fixed and variable O&M, generator TNUoS and insurance, fuel and carbon), the bill tops it up to break even on operating cost, never capital. Sized in the fes_ht_2050 register on the private case after the ancillary-service and gate-fee revenue (scripts/public_ownership/support_lines_2050.py) and read here from the run group's public account (cases.private.life_extension_top_up_gbp), spread over 2050 metered supply. Paid to the row's owner in every market case; the acquisition tariff replaces it on plant the state owns."),
        derivation={"cost_from": {"public_account": PUBLIC_ACCOUNT_2050, "case": "private",
                                  "key": "life_extension_top_up_gbp"},
                    "recovery_denominator": "final_demand_mwh"},
        limitations=['A declared support line: no such scheme exists; it keeps life-extended plant running at operating cost.', "Sized on the solved year's dispatch and the private case's revenue; a state owner is paid the same amount."],
        period_basis="calendar_year",
    )

    # 8c. FES 2050 decision 63: the battery revenue top-up the register books (support_lines_2050).
    se["battery_revenue_top_up"] = base_line(
        "sizewell_c_rab", price_year=2024, rebasing=copy.deepcopy(A2030["currency_rebasing"]),
        label="Battery revenue top-up (batteries short of their required revenue after ancillary services)",
        source_file="scripts/public_ownership/support_lines_2050.py",
        source_url="resources/public_ownership/fes_ht_2050_public_account_summary.json", access_date="2026-10-03",
        source_table="fes_ht_2050 register: battery_revenue_top_up_gbp on the private case",
        method=("FES 2050 decision 63 (user, 3 Oct 2026): batteries (carrier Battery) are merchant in 2050 and stay short of "
                "their required revenue (capital at the private rate as a level annuity over the economic life, decision "
                "66; O&M, generator TNUoS and insurance) after their market, Capacity Market, ancillary-service and declared "
                "sub-hourly revenue (decision 67: +20% of wholesale plus balancing revenue), so the bill tops each battery "
                "row up to break even. "
                "Sized in the fes_ht_2050 register on the private case after every other income "
                "(scripts/public_ownership/support_lines_2050.py) and read here from the run group's public account "
                "(cases.private.battery_revenue_top_up_gbp), spread over 2050 metered supply. The same payment goes to the "
                "row's owner in every market case: a state-held battery needs less at the public rate, and the difference "
                "is the state's surplus, returned to bills under the base incidence. The acquisition tariff replaces it. "
                "Long-duration stores (pumped hydro, LAES, CAES) are on the LDES cap and floor instead."),
        derivation={"cost_from": {"public_account": PUBLIC_ACCOUNT_2050, "case": "private",
                                  "key": "battery_revenue_top_up_gbp"},
                    "recovery_denominator": "final_demand_mwh",
                    # Decision 67's high case (+40% sub-hourly revenue): the top-up the register sizes at it.
                    "sensitivity_cost_from_keys": {"subhourly_high": "battery_revenue_top_up_subhourly_high_gbp"}},
        limitations=["A declared support line: no such scheme exists; it keeps the FES battery fleet whole.",
                     "Sized on the solved year's dispatch and the private case's revenue; a state owner is paid the same amount."],
        period_basis="calendar_year",
    )

    # 9. NCC levy: scheme cost held flat real over 2050 supply (policy-dependent).
    ncc_rate = 1.0436
    se["bics_levy"] = base_line(
        "bics_levy", price_year=2026, rebasing=REB_2026_CAP,
        label="Network Charging Compensation levy (2026/27 scheme cost held flat real over 2050 supply; policy-dependent)",
        source_file=S30["bics_levy"]["source_file"], source_url=S30["bics_levy"]["source_url"],
        access_date=S30["bics_levy"]["access_date"], source_table=S30["bics_levy"]["source_table"],
        method=("Declared: the Network Charging Compensation scheme (the British Industry Supercharger's network element) continues "
                "at its Oct-Dec 2026 cost (1.0436 GBP/MWh on 253.97 TWh, GBP265m, 2026 money) held flat in real terms, spread over "
                "2050 metered supply. The scheme has no statutory end date; its continuation is a policy choice."),
        derivation={"cost_gbp": ncc_rate * supply_2025_26, "recovery_denominator": "final_demand_mwh",
                    "rate_gbp_per_mwh_oct_dec_2026": ncc_rate, "supply_base_mwh": supply_2025_26},
        limitations=["Policy-dependent: nil if the scheme ends; higher if 2050 network charges (which it compensates) rise as the network lines do."],
        period_basis="extrapolated_declared",
        extrapolation={"from_period": "Ofgem cap period 17a, October-December 2026", "rule": "scheme cost held flat real, over 2050 metered supply"},
    )

    # 10. LCCC/ESC operational levies: per-MWh rate held flat real.
    se["nuclear_contract_counterparty"] = held_from_2030(
        "nuclear_contract_counterparty",
        "the counterparty bodies' cost per MWh is held, so their total grows with 2050 supply; the 2050 contract book "
        "(CfDs, the nuclear RAB, DPAs) is larger than 2030's.")

    # 11. AAHEDC: scheme cost held flat real over 2050 volume.
    aa_rate_gbp_per_kwh = 0.044201 / 100.0
    aa_base_mwh = 275151189.05
    se["aahedc"] = base_line(
        "aahedc", price_year=2026, rebasing=REB_2026_27,
        label="AAHEDC (2026/27 scheme cost held flat real over 2050 supply)",
        source_file=S30["aahedc"]["source_file"], source_title=S30["aahedc"]["source_title"],
        source_url=S30["aahedc"]["source_url"], access_date=S30["aahedc"]["access_date"],
        method=("Declared: the statutory scheme continues (Electricity Act 1989 s.184 / Hydro Benefit Replacement) at its 2026/27 "
                "cost held flat in real terms: the Total Scheme tariff 0.044201 p/kWh on NESO's 2024-25 chargeable volume "
                "(275.15 TWh at the GSP) = GBP121.6m, spread over 2050 metered supply (the GSP rate on the household's loss-uplifted "
                "3.1 MWh, so no separate loss uplift)."),
        derivation={"cost_gbp": aa_rate_gbp_per_kwh * aa_base_mwh * 1000.0, "recovery_denominator": "final_demand_mwh",
                    "tariff_p_per_kwh_2026_27": 0.044201, "chargeable_volume_mwh": aa_base_mwh,
                    "chargeable_volume_source": D1["gsp_to_meter_loss_factor"]["cross_check"]["neso_source"]},
        limitations=["The 2026/27 tariff applied to the 2024-25 chargeable volume; the scheme amount is reset annually."],
        period_basis="extrapolated_declared",
        extrapolation={"from_period": "NESO FY2026/27 AAHEDC tariff", "rule": "scheme cost held flat real, over 2050 metered supply"},
    )

    # 12. Warm Home Discount: 2030/31 budget held flat real, over 2050 domestic supply (policy-dependent).
    whd = S30["warm_home_discount"]["derivation"]
    res_cal_gwh, dh_res_cal_gwh = F["res_cal_gwh"], F["dh_res_cal_gwh"]  # ED1 rows 14 and 59, calendar Y (fes2025_targets)
    domestic_mwh = (res_cal_gwh - dh_res_cal_gwh) * 1000.0
    whd_value = float(whd["scheme_cost_gbp_2030_31_nominal"]) * float(whd["electricity_share"]) / domestic_mwh * 3.1
    se["warm_home_discount"] = base_line(
        "warm_home_discount", value=whd_value, price_year=2030, rebasing=REB_2030_31,
        label="Warm Home Discount, electricity half (2030/31 budget held flat real over 2050 domestic supply; policy-dependent)",
        source_file=S30["warm_home_discount"]["source_file"], source_title=S30["warm_home_discount"]["source_title"],
        source_url=S30["warm_home_discount"]["source_url"], access_date=S30["warm_home_discount"]["access_date"],
        method=("Declared: the scheme is renewed after 2030/31, as it has been since 2011, at the OBR's 2030-31 budget (GBP0.970bn "
                "nominal) held flat in real terms, half on electricity, recovered per MWh of domestic electricity supplied. 2050 "
                "domestic supply = FES HT residential demand less residential district heat (whose electricity is bought by heat "
                "network operators), calendar 2050: %.2f - %.2f = %.2f TWh." % (res_cal_gwh / 1e3, dh_res_cal_gwh / 1e3,
                                                                    (res_cal_gwh - dh_res_cal_gwh) / 1e3)),
        derivation={"scheme_cost_gbp_2030_31_nominal": float(whd["scheme_cost_gbp_2030_31_nominal"]),
                    "electricity_share": float(whd["electricity_share"]), "domestic_electricity_supply_mwh": domestic_mwh,
                    "domestic_supply_source": f"{WB} ED1 row 14 (Residential) less row 59 (Residential District Heat), calendar 2050 = 0.75 FY2050 + 0.25 FY2049 (scripts/FES/fes2025_targets.py)",
                    "arithmetic": f"0.970e9 x 0.5 / {domestic_mwh:,.0f} MWh x 3.1 = GBP{whd_value:.4f} (FY2030/31 money)"},
        limitations=["Policy-dependent: the scheme period confirmed in January 2026 ends 31 March 2031; nil if not renewed.",
                     ("The 50:50 fuel split is the 2026/27 price-cap convention; with no gas boilers in 2050 (ED7 row 452) the whole scheme might fall on electricity, doubling the line."
                      if Y == 2050 else "The 50:50 fuel split is the 2026/27 price-cap convention; as gas boilers decline toward " + FIXED_2050 + " (ED7 row 452) more of the scheme might fall on electricity, up to doubling the line.")],
        period_basis="extrapolated_declared",
        extrapolation={"from_period": "OBR EFO March 2026, WHD 2030-31", "rule": "budget held flat real, over FES 2050 domestic supply"},
    )

    # 13. FiT: nil (last tariffs expire early 2040s).
    se["feed_in_tariffs"] = base_line(
        "feed_in_tariffs", value=0.0, value_basis="nil", price_year=2025,
        label=("Feed-in Tariffs (nil: every tariff has expired by the early 2040s)" if Y >= FIT_ENDED_FROM
               else "Feed-in Tariffs (nil, declared: the last 20-year tariffs end about 2039-41)"),
        source_file=S30["feed_in_tariffs"]["source_file"], source_title=S30["feed_in_tariffs"]["source_title"],
        source_url=S30["feed_in_tariffs"]["source_url"], access_date=S30["feed_in_tariffs"]["access_date"],
        method=("The scheme closed to new applicants on 31 March 2019. Tariff lifetimes are 25 years for PV accredited before "
                "1 August 2012 (so ending by July 2037) and 20 years for later PV and for wind, hydro and AD (ending by about 2039-41). "
                "DESNZ's analytical annex (January 2026, p.6) projects scheme costs 'falling to zero in the early 2040s, consistent "
                "with the expected expiry profile of existing generator contracts'. " + (
                    "So the 2050 levelisation is nil." if Y >= FIT_ENDED_FROM else
                    "Calendar 2050 holds only the tail of tariffs accredited in the scheme's last year (2018-19), a small "
                    "fraction of today's cost, which is declared nil (no source projects the tail).")),
        derivation={"desnz_statement": "costs are assumed to decline gradually each year from 2030, falling to zero in the early 2040s (DESNZ FiT analytical annex, January 2026, p.6)",
                    "ofgem_scheme_closure": "https://www.ofgem.gov.uk/environmental-and-social-schemes/feed-tariffs-fit/scheme-closure"},
        period_basis="calendar_year",
    )

    # 14. ECO: nil (off bills since April 2026).
    eco = copy.deepcopy(S30["eco"])
    eco["method"] = ("ECO has not been levied on bills since 1 April 2026 (Budget 2025); home energy efficiency moved to the "
                     "Exchequer-funded Warm Homes Plan. 2050 (task Y13): nil, a declared policy assumption; no successor "
                     "supplier obligation is proposed.")
    eco["label"] = "Energy Company Obligation (nil: off bills since April 2026; policy-dependent)"
    eco["reference_gbp_per_3100kwh_ex_vat"] = None
    eco["limitations"] = list(eco.get("limitations") or []) + ["2050: a policy assumption held for twenty years." if Y == 2050 else f"2050: a policy assumption held for {Y - 2030} years."]
    se["eco"] = eco

    # 15. Remaining CCS (NZT Power DPA): nil, the DPA ends about 2043.
    se["remaining_ccs"] = base_line(
        "remaining_ccs", value=0.0, value_basis="nil", price_year=2025,
        label=("Remaining CCS levy: Net Zero Teesside Power DPA (nil: the 15-year DPA ends about 2043)" if Y >= NZT_DPA_LIVE_UNTIL
               else "Remaining CCS levy: Net Zero Teesside Power DPA (nil here: live in 2050, settled on the model-linked dpa line)"),
        source_file=S30["remaining_ccs"]["source_file"], source_title=S30["remaining_ccs"]["source_title"],
        source_url=S30["remaining_ccs"]["source_url"], access_date=S30["remaining_ccs"]["access_date"],
        method=(("The only CCS contract on bills in 2030, the Net Zero Teesside Power DPA (742.91 MW, signed November 2024, LCCC "
                "payments from 2028/29, 15 years), ends about 2043. Every gas CCS and hydrogen-power unit in 2050 is on a modelled "
                "later agreement that Y11 settles on the solved year (plan Y11: DPA availability plus variable payment); those "
                "payments belong to a model-linked line, not this external one, so it is nil.") if Y >= NZT_DPA_LIVE_UNTIL else
                ("The Net Zero Teesside Power DPA (742.91 MW, signed November 2024, LCCC payments from 2028/29, 15 years, to "
                 "about 2043) is live in 2050. In the vintaged 2050 register it is the 2029 gas CCS cohort, labelled DPA "
                 "(build_support_labels), and every gas CCS and hydrogen-power unit is settled on the solved year on the "
                 "model-linked dpa line (support_regime_2050.dpa_payments: availability plus variable payment less "
                 "gainshare), NZT included on the modelled DPA terms in place of its signed ones. Carrying it here as well "
                 "would count it twice, so this external line is nil.")),
        derivation={"dpa_term_years": 15, "first_payment_fiscal_year": "2028/29",
                    "model_link": "SETTLE2050: the 2050 gas CCS and hydrogen DPAs are the model-linked dpa line "
                                  "(model_linked_rules.dpa); the BECCS CfD is on allocated_ccs (model_linked_rules.cfd.ccs_levy_carriers)"},
        limitations=(["The 2050 DPA payments are on the dpa line, settled on the solved year; this external NZT line stays nil."]
                     if Y >= NZT_DPA_LIVE_UNTIL else
                     ["NZT is settled on the modelled DPA terms (support_regime_2050.json), not its signed contract's.",
                      "The 2050 DPA payments are on the dpa line, settled on the solved year; this external NZT line stays nil."]),
        period_basis="calendar_year",
    )

    # ── Model-linked rules: carried from 2030 with a declared 2050 status per rule ──
    rules = _pre_f18_rules(copy.deepcopy(A2030["model_linked_rules"]), float(D1["gsp_to_meter_loss_factor"]["value"]))
    # 13 Nov 2026: since the 11 Nov 2026 alignment every 2040/2050 balancing family bids its CfDs at contract value
    # with no negative-price exemption (2030's NESO/FTI rule), and the balancing-cost extraction refuses a bill on
    # another rule than the bid audit, so the bill settles the CfD line on the same rule (as 2030's
    # 2030_pypsa_informed_assumptions_cfd_rule_none.json does for the live 2030 cases).
    rules["cfd"]["negative_price_rule"] = "none"
    rules["cfd"]["negative_price_rule_decision"] = (
        "none (13 Nov 2026): since the 11 Nov 2026 alignment of the long-range years on 2030's balancing methods "
        "(user decision) every balancing family bids every CfD at strike minus wholesale with no negative-price "
        "exemption (support.bm_bids.contract_value_bids.negative_price_rule: none), and the balancing-cost "
        "extraction refuses a bill whose rule differs from the bid audit, so the bill settles the CfD line on the "
        "same rule, as 2030_pypsa_informed_assumptions_cfd_rule_none.json does for the live 2030 cases. "
        "Previous decision: " + rules["cfd"]["negative_price_rule_decision"])
    rules["renewables_obligation"].update(
        scheme_parameters="data/market/consumer_bill_forecast/2050_ro_scheme_parameters.json",
        fleet_note="2050 (task Y13): the RO closed to all generation support on 31 March 2037, so no station holds an accreditation in calendar 2050 and the builder writes a declared nil (build_ro_eligible_fleet.closed_scheme_cost).",
        relief_note="2050: no relief case applies; the case declares ro_relief_case scheme_closed.",
        fes2050_status="nil in 2050 (scheme closed 31 March 2037); reference_total kept as the 2030 comparator only")
    rules["cfd"]["volume_collars"] = {}
    # The 2030 file's record of its former HPC override (OW12) is 2030 evidence; 2050
    # writes its own strike_overrides_note below.
    for key in ("hpc_strike_reduction", "strike_overrides_note"):
        rules["cfd"].pop(key, None)
    # SETTLE2050: one settlement path for the 2050 contracts.
    rules["cfd"]["strike_table"] = STRIKE_TABLE_2050
    rules["cfd"]["strike_basis"] = (
        "Y11's 2050 strikes (scripts/market/cfd_strikes_2050.py, contract_book_2050.strike_table_frame) in the published "
        "table's 2012 prices, indexed to the 2024 model basis by cfd_strike_indexation: PAR7_<build year> entries at the "
        "build year's capital over the contract term at the private hurdle rate, less the merchant tail's value at the solved "
        "year's capture price, then calibrated so each private cohort breaks even on the modelled output (decisions 41 and "
        "41 (b), public_ownership.vintage_capital_2050; decision 6's economic-life levelising is a sensitivity; BECCS at "
        "support_regime_2050.beccs_strike, uncalibrated), and the AR7 and HPC contracts live in 2050, each round collapsed "
        "to its live-MW-weighted strike per technology (AR7 English 91.20, Scottish 89.49 and floating 216.49 settle as one "
        "offshore entry, as the 2030 table collapses delivery years).")
    rules["cfd"]["bm_strike_table"] = BM_STRIKE_TABLE_2050
    rules["cfd"]["bm_strike_table_note"] = (
        "Decision 41 (2 Oct 2026): the contract strikes above are set in the bill and settlement layer only "
        "(public_ownership.vintage_capital_2050 on FES25_HT_2050: capital over the contract term less the merchant tail's "
        "value, calibrated to break even on the modelled output). Decision 49 (b): the BM bids at these calibrated strikes "
        "(support_labels.strike_variant: calibrated; vintage_capital_2050.bm_strike_basis: calibrated), so this table is "
        "the contract table itself and the CfD curtailment compensation, which reconciles against the solved BM's "
        "accepted bids, is valued at the same strikes as the difference payments (decision 6's economic-life table is the "
        "basis only if bm_strike_basis is economic_life).")
    _scale_imbalance_with_wind_and_solar(rules, Y, ROOT)
    rules["cfd"]["strike_overrides"] = []
    rules["cfd"]["strike_overrides_note"] = (
        "None in 2050: the 2050 table carries HPC at 89.50 (2012 prices) itself, the 2030 override's value.")
    rules["cfd"]["contract_book"] = CONTRACT_BOOK_2050
    rules["cfd"]["contract_book_note"] = (
        "SETTLE2050: the HT50 network carries no support labels (one clustered generator holds several build years). "
        "The CfD line settles the fes_ht_2050 register's contract units (contract_book_2050.build_contract_book): each a "
        "register row holding a live CfD, its network asset's balancing dispatch x cohort share x live share, through the "
        "same calculator (revenue_tracking.compute_cfd_payments). The asset settlement, the strike adjustments' paid-energy "
        "probe and the constraint line's compensation read the same book.")
    rules["cfd"]["fes2050_status"] = (
        "SETTLE2050 (1 Oct 2026): settled on the 2050 contract book against the 2050 strike table (AR7 units by start year, "
        "post-AR7 vintages at the Y2 levelised cost at the private hurdle rate, each technology's term: 20 years wind and "
        "solar, 15 tidal and BECCS, 35 nuclear). The Drax bridging collar (2027-31) is removed: no bridging CfD runs in 2050. "
        "HPC runs on its 35-year CfD at 89.50. BECCS (CCS Biomass) is booked on allocated_ccs; the gas CCS and hydrogen DPAs "
        "are the dpa line.")
    rules["dpa"] = {
        "formula": ("2050 dispatchable power agreements (gas CCS; hydrogen power by decision 22): availability payment + "
                    "variable payment - gainshare, settled per contract unit on the solved year "
                    "(support_regime_2050.dpa_payments), / CfD eligible supplier sales * 3.1 MWh, rebased to GBP real 2025"),
        "label": "Dispatchable power agreements (gas CCS and hydrogen power)",
        "settlement_summary": SUPPORT_SUMMARY_2050,
        "recovery_denominator": "cfd_eligible_supplier_sales_mwh",
        "recovery_source": ("DPAs are funded through the CfD Electricity Supplier Obligation (ECC full business case, June "
                            "2025, section 5.6.1), so the line is on the CfD levy base."),
        "note": ("The settlement's dpa_support_gbp (build_asset_settlement.py --contract-book) is the money this line "
                 "carries: the bill reads the settlement's total, so the register and the bill pay the plants the same. "
                 "Availability at the declared expected availability (0.90); the variable payment keeps the plant 1 "
                 "GBP/MWh ahead of a reference unabated CCGT at the run's gas and carbon prices; 50% gainshare of the "
                 "energy margin (physical output at the Stage 1 price less the plant's own dispatch cost). Replaces the 2030 "
                 "remaining_ccs line's role (that external line is nil in 2050)."),
        "fes2050_status": "SETTLE2050 (1 Oct 2026): new for 2050.",
    }
    rules["bsuos_constraints"].pop("cp30_pathway_curtailment_twh", None)
    rules["bsuos_constraints"]["fes2050_status"] = (
        "Method inherited. The CP30 pathway curtailment comparator is removed (Y8's FES dispatch validation owns the 2050 "
        "curtailment comparison, %.1f TWh gross in ES1); neso_outturn_comparator and reference_total are the 2024-25 and "
        "2030 comparators, kept for the diagnostic only." % F["curtailment_gross_twh"])
    # User, 3 Oct 2026 (item 5): a cable is under cap and floor only while its own 25-year regime is
    # live in 2050, counted from its start of operation; after that it is merchant, with no
    # pass-through to consumers. Every cable carrying 2050 rent is one of these twelve (the 2050
    # build carries its interconnector capacity on the named cables), so the rule is per cable.
    ended = []
    for name, spec in rules["interconnector_cap_and_floor"]["cables"].items():
        if spec.get("regime") != "cap_and_floor":
            continue
        end = int(spec["operational"]) + IC_REGIME_YEARS
        spec["regime_end_year"] = end
        if end <= Y:
            spec["regime"] = "exempt_merchant"
            spec["post_regime_note"] = (
                f"Cap-and-floor regime ({IC_REGIME_YEARS} years from {spec['operational']}) ended in {end}: merchant in "
                f"{Y}, no consumer pass-through (user, 3 Oct 2026). The regime's levels are kept for reference.")
            ended.append(f"{name} ({end})")
    rules["interconnector_cap_and_floor"]["fes2050_status"] = (
        "Method inherited, with each cable's own regime term (user, 3 Oct 2026): a cap-and-floor regime runs "
        f"{IC_REGIME_YEARS} years from operation (Ofgem OFG1164), so " + ", ".join(ended) + " have left theirs by "
        f"{Y} and are merchant (exempt_merchant: nothing returns to GB bills); cables whose regime is live "
        "still pass through. A regime ending in the model year counts as ended.")
    rules["ldes_cap_and_floor"]["fes2050_status"] = "Method inherited: the 2050 LDES fleet on 25-year cap-and-floor terms (plan Y11), topped up from the fes_ht_2050 group's public account."
    rules["direct_fuel"]["fes2050_status"] = (
        "Method inherited: Ofgem's cap-rule shaping and procurement allowances (unchanged 2019 to Q4 2026) held for 2050. The "
        "domestic profile is the 2020 one; the 2050 heat-pump and EV load shapes are priced in the cost-to-serve diagnostic "
        "(archetype_cost_to_serve.py --composite), not in the benchmark bill.")

    # ── The household (decisions 12, 12a) ──
    app_gwh, light_gwh = F["app_gwh"], F["light_gwh"]  # ED1 rows 19 and 24, FY Y
    base_kwh = (app_gwh + light_gwh) * 1e6 / CUSTOMERS[Y]
    # F.05 publishes the SCOP and EV efficiency gains for 2050 only; another year uses F.05's formulas with each
    # efficiency interpolated 2024 -> 2050 (fes_bill_figures; FES's rule for end-use efficiencies).
    scop_saving_twh = F["scop_saving_twh"]   # F.05 row 9, column AB (SCOP improvements)
    hp_dh_twh = F["hp_dh_twh"]
    hp_dh_total = sum(sum(v.values()) for v in hp_dh_twh.values())
    scop_gain = scop_saving_twh / hp_dh_total
    ev_saving_twh, road_twh = F["ev_saving_twh"], F["road_twh"]   # F.05 row 12 column V; row 12 column R (Energy Flow Diagram row 22)
    ev_gain = ev_saving_twh / road_twh
    ashp_kwh = F["ashp_twh"] * 1e9 / F["ashp_units"]     # ED7 row 7 / row 445
    bev_kwh = F["bev_twh"] * 1e9 / F["bev_cars"]          # ED5 row 201 / row 81
    published = F["scop_basis"] == "published"
    f05_note = "" if published else ", formula on the year's demand with the efficiencies interpolated 2024-" + FIXED_2050
    hp30 = next(a for a in A2030["household_archetypes"]["archetypes"] if a["id"] == "heat_pump")["added"]
    ev30 = next(a for a in A2030["household_archetypes"]["archetypes"] if a["id"] == "electric_vehicle")["added"]

    archetypes = {
        "purpose": ("Task Y13 (decisions 12 and 12a): the 2050 headline is a heat-pump plus EV household on a single-rate tariff, "
                    "composed as base + heat-pump addition + EV addition. Beside it, the same household on FES's own per-unit "
                    "averages. The 3,100 kWh benchmark is reported for comparability with 2030; its time-of-use cost-to-serve is "
                    "archetype_cost_to_serve.py --composite on the 2050 hourly inputs."),
        "benchmark_kwh": 3100,
        "headline_archetype": "heat_pump_ev_2050",
        "report_composition": True,
        "archetypes": [
            {"id": "fes_2050_base", "kwh": base_kwh,
             "source": f"FES 2025 HT FY2050: ED1 row 19 (Residential Appliance Demand {app_gwh} GWh) + row 24 (Residential Light Demand {light_gwh} GWh) over Dem_BB001a domestic customers ({CUSTOMERS[Y]:,.0f}); the smart-meter reduction (row 39) is not netted",
             "source_file": WB},
            {"id": "heat_pump_ev_2050", "base": "fes_2050_base",
             "added": [
                 {"part": "heat_pump",
                  "inputs": {"gas_kwh": hp30["gas_kwh"], "boiler_efficiency": hp30["boiler_efficiency"],
                             "seasonal_performance_factor": hp30["seasonal_performance_factor"],
                             "seasonal_performance_gain": scop_gain},
                  "source": ("The 2030 G6 inputs (gas 9,500 kWh, boiler 0.85, SPF 2.80) with FES F.05's SCOP improvement read by F.05's "
                             f"own method (saving = gain x 2050 heat-pump and district-heat electricity): {scop_saving_twh:.3f} TWh "
                             f"(F.05 row 9{f05_note}) / {hp_dh_total:.3f} TWh (ED7 rows 7-54, HP and DH electricity, all sectors, 2050) = "
                             f"+{scop_gain:.2%}, so a 2050 SPF of {hp30['seasonal_performance_factor'] * (1 + scop_gain):.3f}")},
                 {"part": "electric_vehicle",
                  "inputs": {"miles": ev30["miles"], "kwh_per_mile": ev30["kwh_per_mile"], "efficiency_gain": ev_gain},
                  "source": ("The 2030 G6 inputs (9,400 miles, 0.30 kWh/mile) with FES F.05's EV efficiency gain by F.05's own method: "
                             f"{ev_saving_twh:.3f} TWh (row 12, EV efficiency{f05_note}) / {road_twh} TWh (row 12, road transport 2050) = "
                             f"+{ev_gain:.2%}, so {ev30['kwh_per_mile'] / (1 + ev_gain):.4f} kWh/mile")}],
             "alternative_reading": (("Reading F.05's saving as a share of the no-gain counterfactual instead gives +26.1% and +14.3%, "
                                     "and 6,257 kWh; the F.05 calculation notes (rows 35 and 41) define the saving as the gain "
                                     "times 2050 demand, which is the reading used") if published else
                                     ("F.05 publishes no gains for 2050. The upper bound takes the " + FIXED_2050 + " SCOP and vehicle "
                                      "efficiencies as already reached: +%.2f%% SCOP and +%.2f%% EV efficiency; the reading used "
                                      "interpolates them from 2024, FES's own rule for end-use efficiencies"
                                      % (100 * F["upper_bound"]["scop_saving_twh"] / hp_dh_total,
                                         100 * F["upper_bound"]["ev_saving_twh"] / road_twh)))},
            {"id": "heat_pump_ev_2050_fes_per_unit", "base": "fes_2050_base",
             "added": [
                 {"part": "heat_pump", "inputs": {"annual_kwh": ashp_kwh},
                  "source": "FES 2025 HT 2050 air-source heat pump electricity per unit: ED7 row 7 (%.3f TWh) / row 445 (%s units)" % (F["ashp_twh"], f"{F['ashp_units']:,.0f}") + "; includes FES insulation savings"},
                 {"part": "electric_vehicle", "inputs": {"annual_kwh": bev_kwh},
                  "source": "FES 2025 HT 2050 battery-electric car electricity per car: ED5 row 201 (%.3f TWh) / row 81 (%s cars)" % (F["bev_twh"], f"{F['bev_cars']:,.0f}") + "; fleet-average mileage, all charging locations"}]},
        ],
        "domestic_customer_accounts": {
            "value": CUSTOMERS[Y],
            "source": f"FES 2025 HT Dem_BB001a (number of domestic customers), 2050, summed over GSPs ({BB})",
            "use": "a cross-check of the implied domestic standing-charge revenue only; never a recovery denominator (house rule 8)"},
    }

    # User, 13 Nov 2026: the headline electrified household on today's base, and a high-demand one beside it.
    from scripts.consumer_bill.household_archetypes import add_today_base_households
    add_today_base_households(archetypes, A2030["household_archetypes"], "heat_pump_ev_2050")

    mixed = 0.3084280928349764   # the 2030 calibrated mixed standing share (CP30_2030_cp30_network bill summary)
    charge_basis = {
        "purpose": ("Item G6 on the 2050 bill: the standing share of each line is declared outright (no calibration). The 2030 "
                    "calibration reproduces Ofgem's 2026 nil-consumption charge from 2030 line values; re-running it on 2050 "
                    "values would make the standing share an artefact of the 2026 tariff, so the 2030 calibrated share is held."),
        "standing_share": {"tnuos": tn_cal["domestic_residual_gbp_per_year"] / tn_cal["total_gbp_per_year"],
                           "supplier_costs_and_margin": mixed, "duos": mixed, "smart_meter_net_cost_change": mixed},
        "standing_share_sources": {
            "tnuos": "the 2050 line's own derivation: domestic residual (per site) over the line total (calendar blend)",
            "mixed_lines": "resources/analysis/CP30_2030_cp30_network_consumer_bill_future_summary.json diagnostics.household_archetypes.charge_basis.mixed_standing_share (0.308428), held for 2050"},
        "inherit": copy.deepcopy(A2030["charge_basis"]["inherit"]),
    }

    levy = copy.deepcopy(A2030["levy_exemptions"])
    levy["sensitivities"] = {}

    # ── Recovery denominators, pre-run from FES (refresh_recovery_demand.py replaces the volumes) ──
    LOSS = float(D1["gsp_to_meter_loss_factor"]["value"])
    SHARE = float(R2030["cfd_eligible_supply"]["eligible_share_of_supplier_sales"]["value"])
    sys_excl_cal_gwh, daccs_gwh, electrolysis_cal_gwh = F["sys_excl_cal_gwh"], F["daccs_gwh"], F["electrolysis_cal_gwh"]
    h2_parts = F["h2_non_power_parts"]
    h2_non_power_cal_twh = 0.0
    for part in h2_parts:  # summed in WS1's order, as the 2050 literal sum was
        h2_non_power_cal_twh = h2_non_power_cal_twh + part
    load = (sys_excl_cal_gwh + daccs_gwh) * 1000.0
    draw = electrolysis_cal_gwh * 1000.0
    offtake = h2_non_power_cal_twh * 1e6
    gsp = load + draw
    final = gsp / LOSS
    eligible = final * SHARE
    # FES 2050 decision 60: the exemption case is exact on the run's refreshed volumes (the projector's
    # apply_levy_exemptions), so no pre-run share is written (it was draw / gsp = 0.1230).
    levy["sensitivities"]["electrolysers_2050"] = {
        "electrolysers_exempt": True,
        "note": ("FES 2050 decision 60 (user, 3 Oct 2026): the policy-cost exemption case. Electrolysers stay in every "
                 "recovery denominator centrally (decision D1); this named sensitivity removes their draw from the "
                 "policy-cost bases exactly, on the run's own refreshed volumes (project_future_bill.apply_levy_exemptions "
                 "with electrolysers_exempt: the CfD, CCS, FiT and RO eligible base and the Capacity Market's GSP volume). "
                 "Replaces Task Y13's pre-run share (0.1230 of the GSP volume), which needed recomputing after the refresh. "
                 "BSUoS, LDES and the network lines stay on final demand.")}

    assumptions = {
        "schema_version": A2030["schema_version"],
        "title": "2050 PyPSA-informed household-bill assumptions (FES 2025 Holistic Transition)",
        "benchmark": {
            "annual_consumption_kwh": 3100, "customer_scope": "GB-average single-rate electricity customer",
            "period": "calendar 2050", "price_basis": "GBP real 2025", "vat_rate": 0.05, "price_year": 2025,
            "alternative_consumption_kwh": [],
            "bill_calendar_year": Y,
            "period_alignment_note": ("Calendar 2050, the year the model dispatches (FES 2025 HT, 2013 weather; plan decision 1). Model "
                                      "lines are calendar 2050 by construction. Every sourced external declares period_basis: "
                                      "held_flat_real (a sourced value held flat in real terms from its source_period), "
                                      "extrapolated_declared (carried from extrapolation.from_period by a declared rule on FES drivers; "
                                      "TNUoS, DUoS, Capacity Market, Sizewell C RAB, NCC, AAHEDC, WHD) or calendar_year (the nil "
                                      "schemes: FiT, ECO, remaining CCS). Lines marked interim wait for Y11 or Y12. The RO is nil (closed "
                                      "31 March 2037). The projector refuses a line without a basis."),
            "payment_method_scope": PRE_F18_2030["payment_method_scope"],
        },
        "currency_rebasing": copy.deepcopy(A2030["currency_rebasing"]),
        "cfd_strike_indexation": copy.deepcopy(A2030["cfd_strike_indexation"]),
        "model_linked_rules": rules,
        "sourced_externals": se,
        "sensitivities": {"interconnector_trade_prices": {
            "note": ("2050 foreign prices are TYNDP 2024 Distributed Energy 2050 on IC1's tranche structure (task Y3, decision 27); Global Ambition and the TYNDP 2026 draft are sensitivities (decision 29)."
                     if Y == 2050 else "2050 foreign prices are the TYNDP 2026 draft National Trends+ 2050 on IC1's tranche structure (tyndp_prices.TYNDP2026_WEATHER_WEIGHTS_BY_YEAR), as every " + FIXED_2050 + " family is on NT+."),
            "source": "task Y3 outputs (the TYNDP foreign prices of the FES " + FIXED_2050 + " public-ownership work)"}},
        "external_assumptions_gbp_per_3100kwh_ex_vat": {},
        "source": {"title": "No third-party 2050 bill source", "price_year": 2025,
                   "use": "none: every 2050 line is sourced, extrapolated by a declared rule or nil (task Y13). The key is kept because the projector reads source.price_year for its price-basis register.",
                   "page_url": "https://www.electricitybills.uk/2030"},
        "household_archetypes": archetypes,
        "charge_basis": charge_basis,
        "levy_exemptions": levy,
        "exchequer_ledger": {"hpc_schedule_sensitivity": {
            "enabled": False,
            "reason": ("Both Hinkley Point C units run long before 2050 (EDF's latest unit 2 dates are in the early 2030s), so EDF's "
                       "start schedule moves no 2050 energy; the 2030 bill-level sensitivity does not apply (task Y13).")}},
        "fes2050_provenance": {
            "plan": PLAN, "built": ACCESS,
            "generator": ("scripts/consumer_bill/build_2050_bill_inputs.py (task Y13); every figure is also written in its line's derivation"
                          if Y == 2050 else "scripts/consumer_bill/build_fes_bill_inputs.py (from task Y13's " + FIXED_2050 + " builder); every figure is also written in its line's derivation"),
            "fes_workbook": WB, "building_blocks": BB,
            "inherited_from_2030_declared": ["currency_rebasing (CPIH 2024 to 2025)", "cfd_strike_indexation (CPI 2012 to 2024)",
                                             "model_linked_rules (method; each rule carries fes2050_status)",
                                             "levy_exemptions defaults (decision D1; 2024-25 EII share via the recovery file)",
                                             "charge_basis.inherit and the 2030 calibrated mixed standing share"],
        },
    }

    recovery = {
        "purpose": ("Recovery denominators for the FES25_HT_2050 consumer-bill case (task Y13). The method is the 2030 file's (item D1): "
                    "metered supply = (GB electrical load + electrolyser draw) / GSP-to-meter loss factor; the hydrogen offtake pays "
                    "nothing. PRE-RUN: the volumes are FES 2025 HT calendar-2050 figures, declared so the file is complete before the "
                    "solve; scripts/consumer_bill/refresh_recovery_demand.py replaces them with the solved months' own volumes, which "
                    "the projector requires."),
        "forecast_year": Y,
        "case_id": "FES25_HT_2050",
        "system_demand_mwh": {
            "value": load + offtake,
            "source": "PRE-RUN: gb_electrical_load_mwh + h2_offtake_mwh below (FES 2025 HT, calendar 2050)",
            "price_year": None,
            "note": "As in 2030: every modelled Load's p_set (GB electrical load plus any hydrogen-bus Load). Kept only so the file can be checked against the run.",
            "refreshed": "not yet: refresh after the twelve HT50 market months solve",
        },
        "metered_supply_basis": {
            "method": D1["method"],
            "gb_electrical_load_mwh": {
                "value": load,
                "source": (f"PRE-RUN: FES 2025 HT calendar 2050, ED1 system demand excluding electrolysis ({sys_excl_cal_gwh:,.1f} GWh: "
                           f"consumer demand {F['consumer_cal_gwh']:,.1f} + losses {F['losses_cal_gwh']:,.1f}; ED1 rows 9 less 74, 0.75 FY2050 + 0.25 FY2049) plus DACCS "
                           f"electrical demand ({daccs_gwh:,.2f} GWh, ES1, not in ED1; decision 25)"),
                "mapping_note": ("The 2030 identity treats the model's nodal loads as GSP-level (loss-inclusive) volumes, so the pre-run "
                                 "load is the loss-inclusive ED1 system figure. FINDING (Y13): plan Y6 scales the 2050 base to the ED1 "
                                 "consumer total (excluding losses); if the solved loads exclude losses, dividing by the loss factor "
                                 "understates metered supply by about 8% and overstates every per-MWh levy. Decision needed before "
                                 "the refresh (Y6 or coordinator)."),
            },
            "electrolyser_electricity_draw_mwh": {
                "value": draw,
                "source": "PRE-RUN: FES 2025 HT ED1 row 74 (networked electrolysis), calendar 2050 = 0.75 FY2050 + 0.25 FY2049 (%s GWh)" % f"{electrolysis_cal_gwh:,.0f}",
                "note": "Refreshed from the Stage 1 link schedules of the electrolysis links, as in 2030.",
            },
            "h2_offtake_mwh": {
                "value": offtake,
                "source": ("PRE-RUN: FES 2025 HT WS1 non-power hydrogen demand, calendar 2050 (road %s, residential %s, commercial %s, "
                           "industrial %s, rail %s, shipping %s, aviation %s, transmission losses %s TWh H2; "
                           % tuple(_half_up(h2_parts[i]) for i in (0, 1, 2, 3, 4, 5, 6, 9)) +
                           "scripts/FES/fes2025_targets.py hydrogen_quantities)"),
                "note": "Excluded from every electricity denominator. Decision 15 makes non-power hydrogen an annual volume, which may not be a Load; the refresh then reads 0 here and the identity still holds.",
            },
            "gsp_chargeable_volume_mwh": {"value": gsp, "derivation": "gb_electrical_load_mwh + electrolyser_electricity_draw_mwh",
                                          "used_by": ["capacity_market"], "note": "The paying volume at the GSP, before the loss factor."},
            "gsp_to_meter_loss_factor": {
                **copy.deepcopy(D1["gsp_to_meter_loss_factor"]),
                "limitation": ("A 2020 GB-average multiplier held for 2050 (it must equal the procurement losses multiplier). FES HT "
                               "2050 losses are %.2f TWh on %.2f TWh of consumer demand, a ratio of %.4f, close to it."
                               % (F["losses_cal_gwh"] / 1e3, F["consumer_cal_gwh"] / 1e3,
                                  1 + F["losses_cal_gwh"] / F["consumer_cal_gwh"])),
            },
        },
        "final_demand_mwh": {
            "value": final, "derivation": "metered_supply_basis.gsp_chargeable_volume_mwh / metered_supply_basis.gsp_to_meter_loss_factor",
            "used_by": ["bsuos_constraints", "bsuos_other_services", "bsuos_voltage_and_stability_constraints", "ldes_cap_and_floor",
                        "sizewell_c_rab", "bics_levy", "aahedc"],
            "note": "Metered supply (item D1).",
        },
        "cfd_eligible_supply": {
            "eligible_share_of_supplier_sales": {
                **copy.deepcopy(R2030["cfd_eligible_supply"]["eligible_share_of_supplier_sales"]),
                "note": ("The 2024-25 LCCC outturn share (EII exemption 4.35%% of gross sales) held for 2050, declared. FES HT 2050 "
                         "industrial demand excluding electrolysis is %.1f TWh (ED1 row 79); the exempt share of it is not forecast."
                         % F["industrial_excl_electrolysis_fy_twh"]),
            },
            "eligible_supplier_sales_mwh": {"value": eligible, "derivation": "final_demand_mwh (metered supply) * eligible_share_of_supplier_sales",
                                            "used_by": ["cfd", "allocated_ccs", "remaining_ccs", "feed_in_tariffs"]},
        },
        "ro_obligation_eligible_supply_mwh": {
            "value": eligible,
            "source": "Identified with the CfD eligible-supply volume, as in 2030; the RO is closed in 2050 so no line divides by it",
            "used_by": ["renewables_obligation"],
        },
        "domestic_customer_accounts": {
            "value": CUSTOMERS[Y], "source": f"FES 2025 HT Dem_BB001a, 2050 ({BB})",
            "note": "Cross-check and standing-charge context only; never a recovery denominator.",
        },
        "representative_customer_consumption_kwh": {"value": 3100, "source": "Scenario selection, matching the external allowance benchmark",
                                                    "note": "Used only to express the final representative bill. It is never a recovery denominator."},
        "fes2025_mapping": {
            "note": "How FES 2025 ED1/ES1/WS1 map onto the bill's denominators (task Y13). Calendar 2050 = 0.75 FY2050 + 0.25 FY2049 (Y1's rule).",
            "gb_electrical_load_mwh": "ED1 row 9 (system total) - row 74 (networked electrolysis) + ES1 DACCS: consumer demand + losses + DACCS",
            "electrolyser_electricity_draw_mwh": "ED1 row 74",
            "h2_offtake_mwh": "WS1 hydrogen demand less Power Generation (non-power demand)",
            "final_demand_mwh": "(load + draw) / 1.0919: about consumer demand + DACCS + electrolysis at the meter",
            "cfd_eligible_supplier_sales_mwh": "final demand x the 2024-25 EII share",
            "fy2050_values_gwh": dict(F["fy_values_gwh"]),
        },
        "limitations": list(R2030["limitations"]) + [
            "PRE-RUN: FES volumes stand in until the refresh; the projector refuses the file until its volumes match the run."],
        "refresh": {"stamp": "pre-run FES 2025 HT volumes (task Y13, 30 Sep 2026); not refreshed",
                    "volumes_mwh": {"system_demand_mwh": load + offtake, "gb_electrical_load_mwh": load,
                                    "electrolyser_draw_mwh": draw, "h2_offtake_mwh": offtake},
                    "derived_mwh": {"gsp_chargeable_volume": gsp, "final_demand": final, "eligible_supplier_sales": eligible}},
    }

    ro = {
        "purpose": "Renewables Obligation scheme parameters for calendar 2050 (task Y13): the scheme is closed, so the line is a declared nil.",
        "obligation_year": "calendar 2050",
        "scheme_closed": {
            "closed_on": "2037-03-31",
            "reason": ("The RO closed to new accreditations on 31 March 2017 and ends for all generation support on 31 March 2037; "
                       "no accreditation (20 years from its effective date, capped at the scheme end) runs in calendar 2050, so no "
                       "obligation is set and suppliers pay nothing."),
            "source": "The Renewables Obligation Closure Order 2014 and The Renewables Obligation Order 2015 (support ends 31 March 2037); Ofgem RO register (data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv)",
            "source_url": "https://www.ofgem.gov.uk/environmental-and-social-schemes/renewables-obligation-ro",
            "code_constant": "scripts/consumer_bill/build_ro_eligible_fleet.py RO_SCHEME_END",
        },
        "report_price_year": 2025,
        "relief_cases": {"scheme_closed": 0.0},
        "relief_case_source": "No Exchequer relief applies to a closed scheme; the one case exists because the bill case must name its relief case.",
        "reference_external_gbp_per_3100kwh_ex_vat": 0.0,
        "reference_source": "No external RO allowance for 2050.",
        "limitations": ["Late ROC presentation and mutualisation after 2037 are ignored; they settle within a year or two of the scheme end."],
    }

    return {key: year_text(doc, Y) for key, doc in {"assumptions": assumptions, "recovery": recovery, "ro": ro}.items()}


#: References to the 2050 plan, its decisions and its modules: never moved to the model year.
KEEP_2050 = (
    "FES 2050 decision", "FES 2050 plan", "fes_2050_public_ownership_plan", "fes_2050_research", "support_regime_2050",
    "support_lines_2050", "contract_book_2050", "cfd_strikes_2050", "capacity_market_2050", "vintage_capital_2050",
    "SETTLE2050", "build_capacity_market_2050", "fes2050_status", "fes2050_provenance", "repd_stale_records_2050",
    "bsuos_rebase_2050", "known_contracts_2050", "strike_table_2050", "test_fes2050", "fes2050_pumped",
    "build_2050_bill_inputs", "HT50", "cm_2050_fleet", "cm_2050_requirement", "levelised_strike_2050",
    "fes_ht_2050_network_finance_account",
)


def _half_up(value: float, places: int = 2) -> str:
    """``value`` to ``places`` decimals, halves rounded up (as the 2050 text was written by hand)."""
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def year_text(document, year: int):
    """The document with the template's year (2050, and 2049/2051 beside it) moved to ``year``, keys included.

    The identity for 2050. :data:`KEEP_2050` and :data:`FIXED_2050` text keep 2050.
    """
    y = int(year)

    def move(text: str) -> str:
        if y != 2050:
            tokens = {}
            for i, keep in enumerate(sorted(KEEP_2050, key=len, reverse=True)):
                token = f"\x01{i}\x01"
                tokens[token] = keep
                text = text.replace(keep, token)
            # Fiscal and delivery years first (FY2049/50, 2050/51), then the calendar years.
            text = re.sub(r"(?<!\d)2049/50(?!\d)", f"{y - 1}/{y % 100:02d}", text)
            text = re.sub(r"(?<!\d)2050/51(?!\d)", f"{y}/{(y + 1) % 100:02d}", text)
            text = re.sub(r"(?<!\d)2049_50(?!\d)", f"{y - 1}_{y % 100:02d}", text)
            text = re.sub(r"(?<!\d)2050_51(?!\d)", f"{y}_{(y + 1) % 100:02d}", text)
            text = re.sub(r"(?<!\d)2049(?!\d)", str(y - 1), text)
            text = re.sub(r"(?<!\d)2051(?!\d)", str(y + 1), text)
            text = re.sub(r"(?<!\d)2050(?!\d)", str(y), text)
            for token, keep in tokens.items():
                text = text.replace(token, keep)
        return text.replace(FIXED_2050, "2050")

    def walk(value):
        if isinstance(value, dict):
            return {(move(k) if isinstance(k, str) else k): walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            return move(value)
        return value

    return walk(document)


def render(document: dict) -> str:
    """The committed JSON text of one document."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def keep_committed_refresh(recovery: dict, path) -> dict:
    """The built recovery document with the committed file's refresh applied, if it has one.

    ``scripts/consumer_bill/refresh_recovery_demand.py`` replaces the pre-run FES volumes
    with the solved months' own (2 Oct 2026). Rebuilding must not put the pre-run volumes
    back, so the committed refresh is re-applied to the builder's document.
    """
    path = Path(path)
    if not path.exists():
        return recovery
    committed = json.loads(path.read_text(encoding="utf-8"))
    if "volumes_mwh" not in (committed.get("refresh") or {}):
        return recovery
    from scripts.consumer_bill.refresh_recovery_demand import refresh_document

    doc = refresh_document(recovery, committed["refresh"]["volumes_mwh"], "the solved months")
    doc["refresh"] = committed["refresh"]
    doc["system_demand_mwh"]["refreshed"] = committed["system_demand_mwh"]["refreshed"]
    # The refresh may also name its demand source (refresh_recovery_demand.system_demand_source).
    doc["system_demand_mwh"]["source"] = committed["system_demand_mwh"].get("source", doc["system_demand_mwh"].get("source"))
    return doc


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, required=True, help="the calendar model year")
    parser.add_argument("--check", action="store_true", help="Fail if the committed files differ from the builder")
    parser.add_argument("--stamp", default=None, help="with --check: write this file when the committed files hold")
    args = parser.parse_args(argv)
    documents = build(args.year)
    paths = outputs(args.year)
    documents["recovery"] = keep_committed_refresh(documents["recovery"], paths["recovery"])
    for key, path in paths.items():
        text = render(documents[key])
        if args.check:
            if Path(path).read_text(encoding="utf-8") != text:
                raise SystemExit(f"{path} differs from the builder on the current model outputs; rebuild it: "
                                 f"python -m scripts.consumer_bill.build_fes_bill_inputs --year {args.year}")
            continue
        Path(path).write_text(text, encoding="utf-8")
        print("wrote", path)
    if args.check and args.stamp:
        Path(args.stamp).write_text(f"the {args.year} bill inputs hold\n", encoding="utf-8")
    household = documents["assumptions"]["household_archetypes"]
    from scripts.consumer_bill.household_archetypes import archetype_consumption

    print({k: round(v, 2) for k, v in archetype_consumption(household).items()})


if __name__ == "__main__":
    main()
