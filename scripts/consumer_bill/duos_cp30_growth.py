"""DUoS carried into RIIO-ED3 on CP30's distribution investment (user decision, 6 Oct 2026).

The 2030 bill's DUoS line starts from Ofgem's cap 17a DUoS (Oct-Dec 2026, DNO charging year
2026/27) and rolls it along RIIO-ED2's Final Determinations to 2027/28. Until 6 Oct 2026 it then
grew at the ED2 closing-RAV growth (3.1% a year, a declared assumption) into ED3. The user decided
that the bill's network lines follow the network CP30 assumes, so the ED3 years now grow with the
allowed revenue CP30's distribution investment implies:

* **Investment.** CP30 data workbook V003, sheet CP.13, cell L9: distribution network CAPEX
  GBP 3.386 bn a year, the 2025-2030 average of pathway Further Flex and Renewables (the same in
  New Dispatch; EC2: not a pathway variable), in real 2024 prices (CP30 Annex 4 s.2: "All costs
  have been rebased to 2024 prices", CPI). Baringa built it from the RIIO-ED2 Final
  Determinations, extrapolated to 2028-2030 with CP30's peak demand (Annex 4 s.2.1.4).
* **Asset base.** RIIO-ED2 Final Determinations Finance Annex, Appendix 8 (the revised financial
  values, 14 DNOs summed, 2020/21 prices): closing RAV at 31 March 2028 GBP 34.46 bn. In each ED3
  year the RAV adds CP.13's capex (converted to 2020/21 prices with the network account's CPIH
  factor) and depreciates at ED2's final-year rate (depreciation over opening RAV, 5.98%).
* **Allowed revenue** = ED2's 2027/28 calculated revenue plus the change in its building blocks:
  return and tax allowance in proportion to the average RAV (the return at the ED2 WACC; the ED3
  WACC enters as the separate uplift, as before), depreciation as above, fast money (opex) in
  proportion to CP30's peak demand (ED1 row 21; Annex 4: OPEX "grows in line with total network
  capacity using peak demand as the indicator"), pass-through and other allowances flat.
* **Charge.** DUoS per household moves with allowed revenue against 2026/27 (the cap base year),
  as the ED2 roll-forward did. The ED3 allowed-return uplift, RAV x (4.35% - 3.93%) on this path's
  average RAV, is added to allowed revenue and moves the charge the same way (until 6 Oct 2026 it was
  spread over 2030 final demand, which gave the household the average per-MWh share of it rather
  than its own share of DUoS).
* **Charging base** (user decision, 6 Oct 2026). The unit part of the household's charge (cap 17a:
  GBP 27.104/MWh x 3.1 MWh of the 126.41) is spread over CP30's demand growth: its rate falls by the
  2026/27 demand over the charging year's, on CP30 ED1 row 17 ('GB Total System Demand - Minus
  electrolysis', Further Flex and Renewables), each DNO charging year 3/4 of its first calendar year
  and 1/4 of the next, 2031 extrapolated at 2029-2030's growth. The fixed (per-site) part moves with
  revenue alone; site numbers are held. Holding the whole base is the sensitivity
  ``charging_base_held``. Row 17 includes transmission-connected demand and losses, which pay no
  DUoS: only its growth is used, declared.

Low: no real growth in ED3 (ED2 to 2027/28, then flat, RAV held at its 31 March 2028 close).
High: ED2's 2027/28 RAV additions continued, indexed to CP30's peak demand (Baringa's own
extrapolation rule), the reading in which CP.13 counts only the capex part of totex while Ofgem
capitalises at its ED2 rate.

The network account (``scripts/public_ownership/network_finance_account.py``) built its
distribution RAV from ``resources/analysis/network_price_control_building_blocks.json``, which summed
Appendix 5 *and* Appendix 8, so its levels were twice the 14 DNOs' (GBP 90.4 bn for 2030 against about
45 bn here). The extractor now reads Appendix 8 alone
(:func:`extract_network_price_control_building_blocks.extract_ed2_appendix8`, 6 Oct 2026), and the
account follows at its next run. This line takes its RAV from the revised tables directly;
:data:`NETWORK_ACCOUNT_RAV_2030_GBP_2024_PRE_FIX` records the account's double-counted figure.
"""

from __future__ import annotations

from typing import Callable

#: RIIO-ED2 FD Finance Annex (Nov 2022) Appendix 8, 14 DNOs summed, GBP m 2020/21 prices
#: (``extract_ed2_appendix8``; tests/unit/test_duos_cp30_growth.py re-reads the PDF).
ED2_REVISED = {
    "calculated_revenue": {"2026/27": 5578.8, "2027/28": 5494.4},
    "closing_rav": {"2026/27": 33430.8, "2027/28": 34464.9},
    "opening_rav_2027_28": 33430.8,
    "net_additions_2027_28": 3033.1,
    "depreciation_2027_28": 1999.2,
    "return_2027_28": 1287.0,
    "tax_allowance_2027_28": 294.7,
    "fast_pot_2027_28": 1168.3,
    "source": "Ofgem, RIIO-ED2 Final Determinations Finance Annex (30 Nov 2022), Appendix 8 'REVISED Financial values "
              "for electricity distribution networks', 14 DNOs summed (GBP m, 2020/21 prices)",
    "source_file": "data/market/network_cost_sources/price_controls/ofgem_riio_ed2_final_determinations_finance_annex_2022_11.pdf",
}
#: RIIO-ED2 FD Core Methodology (Nov 2022) Table 12: ex ante totex GBP 22,224 m over five years (2020/21 prices),
#: and the load and non-load related capex of the six DNO group annexes (2,638 + 6,684 = 9,322 m).
ED2_TOTEX = {"ex_ante_totex_gbp_m_2020_21": 22224.0, "lre_plus_nlre_gbp_m_2020_21": 9322.0, "years": 5,
             "source": "Ofgem, RIIO-ED2 Final Determinations Core Methodology Document, Table 12 (para 7.5: GBP 22.2 bn); "
                       "the DNO group annexes (load related and non-load related capex)",
             "source_url": "https://www.ofgem.gov.uk/sites/default/files/2022-11/RIIO-ED2%20Final%20Determinations%20Core%20Methodology.pdf"}
#: CP30 data workbook V003, CP.13 L9 (Further Flex and Renewables), GBP bn a year, real 2024.
CP13_DISTRIBUTION_CAPEX_GBP_BN_2024 = 3.386267287018254
#: CP30 ED1 row 21, 'GBFES Peak Customer Demand: Total Consumption plus Losses' (GW), Further Flex and Renewables.
PEAK_GW = {2026: 58.198, 2027: 59.09, 2028: 59.968, 2029: 61.194, 2030: 62.357}
#: The network account's CPIH factor, 2020/21 (financial-year average 109.15) to 2024 (132.9).
CPIH_2020_21_TO_2024 = 132.9 / ((108.7 + 109.1 + 109.3 + 109.5) / 4)
CPIH_2024_TO_2025 = 138.0 / 132.9
#: The bill's Oct-Dec 2026 money to 2025 (the line's own rebasing factor).
CAP_2026_TO_2025 = 0.973235102669969
ED2_WACC, ED3_WACC = 0.0393, 0.0435
#: The cap 17a base (GBP a year at 3,100 kWh, Oct-Dec 2026 money) and its unit part (GBP 27.104/MWh x 3.1 MWh).
CAP_BASE_2026 = 126.4094
CAP_UNIT_GBP_PER_MWH_2026 = 27.104
HOUSEHOLD_MWH = 3.1
UNIT_SHARE = CAP_UNIT_GBP_PER_MWH_2026 * HOUSEHOLD_MWH / CAP_BASE_2026
#: CP30 ED1 row 17, 'GB Total System Demand - Minus electrolysis' (GWh), Further Flex and Renewables.
DEMAND_GWH = {2026: 286411.857089409, 2027: 291754.0572916392, 2028: 299361.2051777449,
              2029: 306355.4492380522, 2030: 315162.7497490148}
ED3_YEARS = {"FY2028/29": 2028, "FY2029/30": 2029, "FY2030/31": 2030}
CALENDAR_2030 = {"FY2029/30": 0.25, "FY2030/31": 0.75}
#: The network account's 2030 distribution RAV (GBP, 2024 money), resources/public_ownership/
#: cp30_2030_cp30_network_network_finance_account.json, distribution.rav_gbp, as at 6 Oct 2026, before the extractor fix
#: (double counted, see above).
NETWORK_ACCOUNT_RAV_2030_GBP_2024_PRE_FIX = 90366825240.49472


def demand_gwh(year: int) -> float:
    """CP30's calendar-year demand; 2031 extrapolated at 2029-2030's growth (declared)."""
    if year in DEMAND_GWH:
        return DEMAND_GWH[year]
    if year == 2031:
        return DEMAND_GWH[2030] + (DEMAND_GWH[2030] - DEMAND_GWH[2029])
    raise ValueError(f"no CP30 demand for {year}")


def charging_year_demand_gwh(first_year: int) -> float:
    """A DNO charging year (April to March): 3/4 of its first calendar year and 1/4 of the next."""
    return 0.75 * demand_gwh(first_year) + 0.25 * demand_gwh(first_year + 1)


def charging_base_factor(first_year: int, spread: bool = True) -> float:
    """The household's charge per unit of revenue in a charging year against 2026/27.

    The fixed part moves with revenue alone; the unit part's rate falls as CP30's demand grows.
    """
    if not spread:
        return 1.0
    return (1.0 - UNIT_SHARE) + UNIT_SHARE * charging_year_demand_gwh(2026) / charging_year_demand_gwh(first_year)


def cp13_additions_gbp_m_2020_21() -> float:
    """CP.13's distribution capex in GBP m a year, 2020/21 prices."""
    return CP13_DISTRIBUTION_CAPEX_GBP_BN_2024 * 1e3 / CPIH_2020_21_TO_2024


def additions_rule(case: str) -> Callable[[int], float]:
    """RAV additions in an ED3 year (GBP m, 2020/21 prices) under ``central`` or ``high``."""
    if case == "central":
        return lambda year: cp13_additions_gbp_m_2020_21()
    if case == "high":
        return lambda year: ED2_REVISED["net_additions_2027_28"] * PEAK_GW[year] / PEAK_GW[2027]
    raise ValueError(f"no additions rule for {case!r}")


def uplift_revenue_gbp_m_2020_21(average_rav_gbp_m_2020_21: float) -> float:
    """The ED3 allowed-return uplift as allowed revenue: RAV x (ED3 - ED2 WACC), GBP m 2020/21 prices."""
    return average_rav_gbp_m_2020_21 * (ED3_WACC - ED2_WACC)


def path(case: str = "central", spread: bool = True) -> dict:
    """The ED3 fiscal years under ``case``: RAV, allowed revenue, the DUoS charge and the uplift.

    ``spread=False`` holds the charging base at 2026/27 (the sensitivity ``charging_base_held``).
    """
    e = ED2_REVISED
    rev26, rev27 = e["calculated_revenue"]["2026/27"], e["calculated_revenue"]["2027/28"]
    open27, close27 = e["opening_rav_2027_28"], e["closing_rav"]["2027/28"]
    avg27 = (open27 + close27) / 2
    rate = e["depreciation_2027_28"] / open27
    years, close = {}, close27
    for label, year in ED3_YEARS.items():
        if case == "low":
            opening, closing, revenue = close27, close27, rev27
            additions = depreciation = None
        else:
            opening = close
            depreciation = rate * opening
            additions = additions_rule(case)(year)
            closing = opening + additions - depreciation
            average = (opening + closing) / 2
            revenue = (rev27 + e["return_2027_28"] * (average / avg27 - 1) + (depreciation - e["depreciation_2027_28"])
                       + e["tax_allowance_2027_28"] * (average / avg27 - 1)
                       + e["fast_pot_2027_28"] * (PEAK_GW[year] / PEAK_GW[2027] - 1))
        close = closing
        average = (opening + closing) / 2
        base = charging_base_factor(year, spread)
        at_ed2 = CAP_BASE_2026 * revenue / rev26 * base
        uplift_revenue = uplift_revenue_gbp_m_2020_21(average)
        uplift_2026 = CAP_BASE_2026 * uplift_revenue / rev26 * base
        uplift = uplift_2026 * CAP_2026_TO_2025
        years[label] = {"rav_additions_gbp_m_2020_21": additions, "depreciation_gbp_m_2020_21": depreciation,
                        "opening_rav_gbp_m_2020_21": opening, "closing_rav_gbp_m_2020_21": closing,
                        "average_rav_gbp_2024": average * 1e6 * CPIH_2020_21_TO_2024,
                        "allowed_revenue_gbp_m_2020_21": revenue, "revenue_over_2026_27": revenue / rev26,
                        "uplift_revenue_gbp_m_2020_21": uplift_revenue,
                        "charging_demand_gwh": charging_year_demand_gwh(year),
                        "charging_base_factor": base,
                        "value_at_ed2_return_gbp_2026": at_ed2,
                        "uplift_gbp_2025": uplift, "uplift_gbp_2026": uplift_2026,
                        "value_gbp_2026": at_ed2 + uplift_2026}
    calendar = sum(w * years[p]["value_gbp_2026"] for p, w in CALENDAR_2030.items()) * CAP_2026_TO_2025
    at_ed2 = sum(w * years[p]["value_at_ed2_return_gbp_2026"] for p, w in CALENDAR_2030.items()) * CAP_2026_TO_2025
    uplift = sum(w * years[p]["uplift_gbp_2025"] for p, w in CALENDAR_2030.items())
    return {"case": case, "charging_base": "spread" if spread else "held", "by_fiscal_year": years, "calendar_2030_gbp_2025": calendar,
            "calendar_2030_at_ed2_return_gbp_2025": at_ed2, "calendar_2030_uplift_gbp_2025": uplift}


def comparison() -> dict:
    """CP.13's distribution capex against RIIO-ED2's own spend, GBP bn a year in 2024 prices."""
    f = CPIH_2020_21_TO_2024
    e, t = ED2_REVISED, ED2_TOTEX
    years = t["years"]
    return {
        "cp30_distribution_capex_2025_2030": CP13_DISTRIBUTION_CAPEX_GBP_BN_2024,
        "ed2_ex_ante_totex": t["ex_ante_totex_gbp_m_2020_21"] / years / 1e3 * f,
        "ed2_rav_additions_2027_28": e["net_additions_2027_28"] / 1e3 * f,
        "ed2_load_and_non_load_capex": t["lre_plus_nlre_gbp_m_2020_21"] / years / 1e3 * f,
        "cp30_over_ed2_rav_additions": CP13_DISTRIBUTION_CAPEX_GBP_BN_2024 / (e["net_additions_2027_28"] / 1e3 * f),
        "cp30_over_ed2_totex": CP13_DISTRIBUTION_CAPEX_GBP_BN_2024 / (t["ex_ante_totex_gbp_m_2020_21"] / years / 1e3 * f),
    }
