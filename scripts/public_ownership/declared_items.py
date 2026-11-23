"""F27 items 3 and 15 (user, 20 Oct 2026): what ownership cannot reach, and what is declared rather than built.

Every item is sized and dated beside the results, never hidden:

* **Out of reach** (contracts and statute ownership does not change; only renegotiation or
  legislation reaches them): the FIDeR investment contracts, the Renewables Obligation and the
  Feed-in Tariffs. Sized by what the bill pays and until when; FIDeR also by its return above
  cost, on as-built costs where the historical cost table is given (item 2), else on 2030 build
  costs (the register's), which overstates it and is labelled so.
* **Declared, not built** (item 15 and the rest of F27): industrial CHP (excluded, with its
  size), offshore transmission owners, other balancing services, the transmission tax allowance
  (nil), a public retail supplier, forward sale at cost (item 4, at its researched size), and the
  legacy valuation's balancing income held at 2030 (item 13).
* **Limitations** (the literature channels, user 24 Oct 2026: not modelled, stated): transition
  effects, operating performance, central dispatch as an upper bound, compensation and treaties,
  the cost of capital (Grout), technology-specific capex overruns, the gas price state and
  attribution to instruments other than ownership; plus the single weather year.

Sizes are in system GBP (2024 prices) a year and GBP a household ex VAT, from the private case's
bill lines and the register's private rows.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.finance import capital_recovery_factor  # noqa: E402

#: The economic lives the as-built capital charge runs over (the 2030 cost table's, by technology).
TECHNOLOGY_COSTS = PROJECT_ROOT / "data" / "public_ownership" / "financing" / "technology_costs.csv"
#: The register's round for the FIDeR investment contracts.
FIDER_ROUND = "Investment Contract"
#: The contract term of the FIDeR investment contracts (15 years from start).
FIDER_TERM_YEARS = 15
#: Bill lines of the statutory schemes and the year their payments end (decision of 20 Oct 2026).
STATUTORY_LINES = {
    "renewables_obligation": ("Renewables Obligation", 2037,
                              "statute: accredited stations are paid to March 2037; ownership changes who receives "
                              "the certificates (P3 buys RO plant with them), not what the bill pays"),
    "feed_in_tariffs": ("Feed-in Tariffs", 2039,
                        "statute: small-scale generation paid for 20 to 25 years from accreditation (last in 2019); "
                        "small plant is not bought in any case"),
}


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def fider_return_above_cost(private_rows: pd.DataFrame, historical: pd.DataFrame | None = None) -> dict:
    """FIDeR's payments, and its return above cost: on the register's 2030 build cost, and as built where known.

    ``historical`` (item 2, ``technology_costs_historical.csv``) gives capex per kW by technology and
    commissioning year in 2024 prices; the as-built capital charge is that capex over the plant's
    economic life at its private rate, the same rate the register charges.
    """
    rows = private_rows.loc[private_rows["cfd_round"].astype(str).eq(FIDER_ROUND)]
    if rows.empty:
        return {}
    support = float(_num(rows, "support_revenue_gbp").sum())
    rent_2030 = float(_num(rows, "producer_rent_gbp").sum())
    started = pd.to_numeric(rows["commissioning_year"], errors="coerce")
    out = {"capacity_mw": float(_num(rows, "capacity_mw").sum()), "farms": sorted(rows["asset_id"].astype(str)),
           "contract_support_gbp": support, "ends": int((started + FIDER_TERM_YEARS).max()) if started.notna().any() else None,
           "return_above_cost_at_2030_build_cost_gbp": rent_2030,
           "return_above_cost_at_2030_build_cost_basis": ("the register's rent: its capital at 2030 build cost "
                                                          "(technology_costs.csv) over the remaining life; overstates "
                                                          "the return on plant built when it cost more")}
    if historical is None or historical.empty:
        return out
    table = historical.copy()
    table["technology"] = table["technology"].astype(str)
    table["commissioning_year"] = pd.to_numeric(table["commissioning_year"], errors="coerce")
    lives = (pd.read_csv(TECHNOLOGY_COSTS).set_index("technology")["economic_life_years"]
             if TECHNOLOGY_COSTS.exists() else pd.Series(dtype=float))
    as_built_capital = 0.0
    matched = 0
    for index, row in rows.iterrows():
        tech = str(row.get("finance_technology") or "")
        year = started.get(index)
        hit = table.loc[table["technology"].eq(tech) & table["commissioning_year"].eq(year)]
        if hit.empty or not np.isfinite(year):
            continue
        capex = float(hit["capex_power_gbp_per_kw"].iloc[0]) * float(row["capacity_mw"]) * 1000.0
        years = float(lives.get(tech, 25.0))
        rate = float(row["wacc"])
        as_built_capital += capex * capital_recovery_factor(rate, years)
        matched += 1
    if matched == len(rows):
        capital_2030 = float(_num(rows, "annual_capital_recovery_gbp").sum())
        out["return_above_cost_as_built_gbp"] = rent_2030 + capital_2030 - as_built_capital
        out["return_above_cost_as_built_basis"] = ("revenue less O&M less the as-built capital charge: historical "
                                                   "capex for the commissioning year over the economic life at the "
                                                   "private rate (technology_costs_historical.csv, F27 item 2)")
    return out


def build(private_lines: Mapping[str, float], private_rows: pd.DataFrame, *, household_per_system: float,
          historical: pd.DataFrame | None = None, legacy_balancing: Mapping | None = None) -> dict:
    """The declared items, each with its size, its end and why ownership does not reach it.

    ``private_lines`` are the private bill's lines (GBP a household ex VAT); ``household_per_system``
    converts system GBP to a household's GBP (the flat rate over final demand, for display).
    ``legacy_balancing`` is item 13's sizing of the balancing income the legacy valuation holds.
    """
    hh = lambda gbp: float(gbp) * float(household_per_system)  # noqa: E731
    items = []
    fider = fider_return_above_cost(private_rows, historical)
    if fider:
        items.append({
            "id": "fider", "kind": "out_of_reach", "label": "FIDeR investment contracts",
            "gbp_m_per_year": fider["contract_support_gbp"] / 1e6,
            "gbp_per_household": hh(fider["contract_support_gbp"]),
            "ends": fider.get("ends"),
            "return_above_cost_gbp_m_per_year": (fider.get("return_above_cost_as_built_gbp",
                                                           fider["return_above_cost_at_2030_build_cost_gbp"]) / 1e6),
            "detail": fider,
            "why": "signed contracts (2014): only renegotiation reaches them; the state could buy the farms (P3) "
                   "but the contract keeps paying its strike"})
    for line, (label, ends, why) in STATUTORY_LINES.items():
        if line in private_lines:
            items.append({"id": line, "kind": "out_of_reach", "label": label,
                          "gbp_per_household": float(private_lines[line]), "ends": ends, "why": why})
    chp = private_rows.loc[private_rows["carrier"].astype(str).eq("CHP")]
    items.append({"id": "chp", "kind": "excluded", "label": "Industrial CHP",
                  "capacity_mw": float(_num(chp, "capacity_mw").sum()),
                  "why": "the carrier has no fuel or vintage split (industrial, mostly behind the meter); it stays "
                         "private in every case (F27 item 5)"})
    items += [
        {"id": "ofto", "kind": "declared", "label": "Offshore transmission owners (OFTOs)",
         "gbp_m_per_year": 1400.0, "saving_gbp_m_per_year": [0.0, 120.0],
         "why": "about GBP 1.4 bn a year of tendered revenue in 2030 (NESO TNUoS Five-Year View 2027/28-2031/32, "
                "Table 39), mostly paid by offshore generators, whose strikes include it; implied returns are close "
                "to the public rate (about 2-3% real), so a public refinancing is worth about 0-0.12 bn a year and "
                "buying at market value pays most of it away (F27 item 7)"},
        {"id": "other_balancing_services", "kind": "declared", "label": "Other balancing services (BSUoS other)",
         "gbp_per_household": float(private_lines.get("bsuos_other_services", 0.0)),
         "why": "income to the plant only: providers earn about GBP 1.07 bn a year; P2's state share comes back "
                "through the rebate and P3's tariff credits it (both circular); a services-at-cost lever saves only "
                "the providers' margin in competitive daily auctions, a few pounds at most (F27 item 15)"},
        # Plan Part 7.7 and the 2030 close-out audit (24 Oct 2026): channels left out of every case on purpose.
        {"id": "private_wacc_spillover", "kind": "declared", "label": "Spillover to the private cost of capital",
         "gbp_m_per_year": 1300.0, "gbp_per_household": hh(1.3e9),
         "why": "per point off the private WACC on post-2025 build (about double on the whole fleet); the case for "
                "coordination beyond a stake, and in the other direction a policy-risk premium on the capital that "
                "stays private while the 2030 fleet's strikes are set (plan Part 7.7)"},
        {"id": "coordinated_planning", "kind": "declared", "label": "Coordinated planning of generation and network",
         "why": "the three network cases bound what delivery is worth; attributing it to ownership rather than "
                "delivery is a judgement, so no case claims it (plan Part 7.7)"},
        {"id": "build_programme_fixed", "kind": "declared", "label": "The build programme held fixed",
         "why": "every case builds CP30's fleet on CP30's dates; that ownership delivers more, faster or slower is a "
                "different scenario (plan Part 7.7)"},
        {"id": "curtailment_redesign", "kind": "declared", "label": "Curtailment redesign",
         "why": "compensation to curtailed public wind becomes an intra-public transfer; the bid stacks reprice the "
                "state's own bids, but the rules for private plant are unchanged (plan Part 7.7)"},
        {"id": "levies_follow_network_charges", "kind": "declared", "label": "Levies that follow network charges",
         "gbp_per_household": -1.0,
         "why": "BICS, NCC and AAHEDC do not scale with TNUoS and DUoS when the wires are bought: about -1 a household "
                "(plan Part 7.7)"},
        {"id": "wires_purchase_costs", "kind": "declared", "label": "Costs of buying the wires beyond the premium",
         "why": "breaking index-linked licensee debt, pensions and closure liabilities are not costed; transaction "
                "costs are advisory fees only, so the acquisition would cost more (plan Part 7.7)"},
        {"id": "targeted_rebates", "kind": "declared", "label": "Targeted rebates and business bills",
         "why": "no social-tariff or per-customer rebate; energy-intensive users are exempt from most levies, so "
                "their saving per MWh would be smaller (plan Part 7.7)"},
        {"id": "weather_year", "kind": "limitation", "label": "One weather year (2013)",
         "why": "the CfD line, constraints and P3's tariff depend on the weather; no other year is run for 2030 "
                "(user, 24 Oct 2026)"},
        # Literature channels (user, 24 Oct 2026): not modelled, stated as limitations; each cuts against
        # the ownership saving or qualifies how it is read.
        {"id": "transition_effects", "kind": "limitation", "label": "Transition effects",
         "why": "a policy-risk premium on the private, partner and network capital that remains while the 2030 "
                "fleet's strikes are set (2025-27), and a slip in delivery while the state's programme starts; every "
                "case takes the policy as in force with no transition cost (user, 24 Oct 2026)"},
        {"id": "state_operating_performance", "kind": "limitation",
         "label": "State operating performance",
         "why": "availability and cost discipline under an at-cost tariff (Davis and Wolfram 2012; Fabrizio, Rose "
                "and Wolfram 2007; Newbery and Pollitt 1997) go beyond the flat O&M premium (user, 24 Oct 2026)"},
        {"id": "central_dispatch_upper_bound", "kind": "limitation", "label": "Central dispatch is an upper bound",
         "why": "I3 is a perfect-foresight cost-minimising LP, so its gain over the market is an upper bound on what "
                "a real central dispatcher achieves (Cicala 2022); the FTI forecast-error charge is borrowed in place "
                "of the model's own real-time correction (user, 24 Oct 2026)"},
        {"id": "compensation_floor", "kind": "limitation", "label": "A legal floor on compensation",
         "why": "investment treaties (the Energy Charter Treaty's protection to 2045, bilateral treaties) make the "
                "below-market purchase cases (book value, RAV at statute) legally contested (user, 24 Oct 2026)"},
        {"id": "systematic_risk_transfer", "kind": "limitation", "label": "Cheaper public capital is partly risk moved",
         "why": "part of the public cost-of-capital saving is systematic risk moved to taxpayers (Grout 2003), so "
                "the resource-saving column is an upper bound; the bill is unaffected (user, 24 Oct 2026)"},
        {"id": "capex_overruns", "kind": "limitation", "label": "Capital overruns are technology-specific",
         "why": "the state_capex_overrun lever applies one uniform overrun to state new build, while observed "
                "overruns differ by technology (largest for nuclear and hydro, small for wind and solar; Sovacool, "
                "Gilbert and Nugent 2014), so the uniform lever misstates the risk of a nuclear-heavy programme "
                "(user, 24 Oct 2026)"},
        {"id": "price_state", "kind": "limitation", "label": "One gas price",
         "why": "every case is at one 2030 gas price (CP30's 100.68 p/therm); the ownership saving depends on it "
                "(merchant rents, CfD top-ups and the legacy purchase all move with gas), so the saving has a "
                "gas-price spread that is not reported (user, 24 Oct 2026)"},
        {"id": "attribution", "kind": "limitation", "label": "Gains reachable without ownership",
         "why": "some gains are available through instruments (BM licence conditions, CfD design, a green power "
                "pool); the bill effect is reported per channel so this can be read off (user, 24 Oct 2026)"},
        {"id": "transmission_tax_allowance", "kind": "declared", "label": "Transmission tax allowance",
         "gbp_m_per_year": 0.0,
         "why": "nil: the RIIO-3 Final Determinations give no tax allowance for NGET, SHET or SPT in any year "
                "(Finance Annex, Appendix 2, Tables 52, 54 and 56)"},
        {"id": "forward_sale_at_cost", "kind": "declared", "label": "State plant selling forward to suppliers at cost",
         "gbp_per_household": [0.14, 0.43], "gross_bill_line_cut_gbp_per_household": [2.35, 7.18],
         "gross_bill_line_cut_gbp_per_household_high": 7.18,
         "why": "a declared item at the researched size, no lever (user, 24 Oct 2026): removes only trading friction "
                "on the volume sold (half the transaction allowance, a quarter of the additional-risk allowance; "
                "Ofgem cap Appendix 4 2018, CRE ARENH), a resource saving of GBP 0.14 (P2, state output pro rata "
                "over all supply) to 0.43 (domestic suppliers first) a household; a shaped full-requirements "
                "product cuts the shaping and risk lines by up to 2.35 (pro rata) or 7.18 (domestic first) gross, "
                "almost all a transfer to the state that nets against its rebate under bill-payers first; 91% of "
                "P2's state output is already on CfDs (F27 item 4)"},
        {"id": "public_retail_supplier", "kind": "declared", "label": "A public retail supplier",
         "why": "not modelled: supplier costs are the price cap's, and EBIT and headroom follow each case's own "
                "costs (F27 item 10)"},
    ]
    if legacy_balancing:
        on_path = (legacy_balancing.get("gas_balancing_path") or {}).get("path")
        if on_path:
            # F27 item 13 (user, 24 Oct 2026): gas follows NESO's constraint-cost path; the rest is held.
            label = "Balancing income in the legacy valuation: gas on NESO's constraint-cost path, the rest held at 2030"
            why = ("gas plant's 2030 balancing margin (settlement less the variable cost of the energy redispatched) "
                   f"follows NESO's thermal constraint-cost projection relative to 2030 ({on_path}; 2025 Annual "
                   "Balancing Costs Report, Figure 30, the 2035 ratio held after it), as network reinforcement "
                   "shrinks it; every other row's 2030 balancing settlement is held for life: for renewables it "
                   "replaces revenue they would otherwise earn (their bids are at contract or opportunity value). "
                   "The value resting on it is at the flat rate, an upper bound for gas (F27 item 13)")
        else:
            label = "Balancing income in the legacy valuation, held at 2030"
            why = ("the acquired fleet's 2030 balancing settlement is held for life; for renewables it replaces "
                   "revenue they would otherwise earn (their bids are at contract or opportunity value), for gas it "
                   "is a margin that network reinforcement would shrink (F27 item 13)")
        items.append({"id": "legacy_balancing_income", "kind": "declared", "label": label, **dict(legacy_balancing),
                      "why": why})
    return {"items": items,
            "note": "Sizes are the private case's (2024 prices, a household ex VAT). Out of reach: no ownership "
                    "case moves them; declared: left out of every case on purpose, with the reason."}


def legacy_balancing_sizing(valuation: pd.DataFrame, settlement: pd.DataFrame) -> dict:
    """Item 13: the balancing settlement the legacy valuation holds, and the value resting on it, by scope group."""
    v = valuation.copy()
    v["base"] = v["asset_id"].astype(str).str.split("::").str[0]
    balancing = settlement.set_index("asset_id")["balancing_settlement_gbp"] if "balancing_settlement_gbp" in settlement.columns \
        else pd.Series(dtype=float)
    share = v["capacity_mw"] / v.groupby("base")["capacity_mw"].transform("sum")
    v["balancing"] = v["base"].map(balancing).fillna(0.0) * share.fillna(0.0)
    flow = pd.to_numeric(v["annual_flow_model_year_gbp"], errors="coerce").replace(0.0, np.nan)
    factor = (pd.to_numeric(v["pv_market_gbp"], errors="coerce") / flow).clip(lower=0.0, upper=40.0).fillna(0.0)
    v["pv"] = v["balancing"] * factor
    groups = v.groupby("scope_group")
    out = {"balancing_gbp_m_per_year": {k: float(x) / 1e6 for k, x in groups["balancing"].sum().items()},
           "value_resting_on_it_gbp_bn": {k: float(x) / 1e9 for k, x in groups["pv"].sum().items()},
           "group_value_gbp_bn": {k: float(x) / 1e9 for k, x in groups["pv_market_gbp"].sum().items()}}
    # F27 item 13 (24 Oct 2026): the rows whose balancing margin follows NESO's path (legacy_valuation).
    if "balancing_income_path" in v.columns:
        path = v["balancing_income_path"].astype(str)
        on = path.ne("flat")
        if on.any():
            margin = pd.to_numeric(v.loc[on, "balancing_margin_model_year_gbp"], errors="coerce").fillna(0.0)
            out["gas_balancing_path"] = {
                "path": str(path[on].iloc[0]),
                "margin_model_year_gbp_m_by_carrier": {str(c): float(x) / 1e6
                                                       for c, x in margin.groupby(v.loc[on, "carrier"]).sum().items()}}
    return out


def main(argv=None) -> None:
    import argparse
    import json

    from scripts.public_ownership.register_io import read_register

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("register", help="{run_group}_asset_finance_register.parquet")
    parser.add_argument("bill_by_case", help="{run_group}_consumer_bill_by_ownership_case.csv")
    parser.add_argument("network_account", help="{run_group}_network_finance_account.csv (the flat household rate)")
    parser.add_argument("out_json")
    parser.add_argument("--historical-costs", default=None, help="technology_costs_historical.csv (F27 item 2)")
    parser.add_argument("--legacy-valuation", default=None)
    parser.add_argument("--asset-settlement", default=None)
    args = parser.parse_args(argv)

    columns = ["case_id", "asset_id", "carrier", "cfd_round", "capacity_mw", "commissioning_year", "finance_technology",
               "support_revenue_gbp", "producer_rent_gbp", "annual_capital_recovery_gbp", "wacc",
               "economic_life_years"]
    private_rows = read_register(args.register, cases=["private"],
                                 columns=[c for c in columns if c != "economic_life_years"])
    bill = pd.read_csv(args.bill_by_case)
    lines = bill.loc[bill["case_id"].eq("private")].set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    network = pd.read_csv(args.network_account)
    rate = float(pd.to_numeric(network["household_gbp_per_system_gbp"], errors="coerce").dropna().iloc[0])
    historical = pd.read_csv(args.historical_costs) if args.historical_costs and Path(args.historical_costs).exists() else None
    legacy = (legacy_balancing_sizing(pd.read_csv(args.legacy_valuation, low_memory=False),
                                      pd.read_csv(args.asset_settlement, low_memory=False))
              if args.legacy_valuation and args.asset_settlement else None)
    out = build(lines.astype(float).to_dict(), private_rows, household_per_system=rate, historical=historical,
                legacy_balancing=legacy)
    out["historical_costs"] = args.historical_costs if historical is not None else None
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
