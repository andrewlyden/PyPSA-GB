"""The 2030 ancillary services projected on CP30's drivers (user decision, 6 Oct 2026).

The 2030 bill's two non-thermal BSUoS lines (``bsuos_other_services`` and
``bsuos_voltage_and_stability_constraints``) and the register's ancillary-services income
(``support_lines_2050.ancillary_by_row``) read one table,
``data/public_ownership/financing/ancillary_services_2030.csv``. Until 6 Oct 2026 every service
there was its FY2024/25 outturn (user decision, 3 Oct 2026). The user decided on 6 Oct 2026 that
the network-related lines follow the system Clean Power 2030 (NESO, Nov 2024; pathway Further
Flex and Renewables) assumes, so each service is now its FY2024/25 outturn carried to 2030 by a
CP30 driver, as the 2050 table is carried by FES drivers:

* **Stability** (pathfinder and stability-market contracts): the contracted book live in 2030
  (Phase 2 and Phase 3 pathfinders, the mid-term Y-1 market; Phase 1 ends April 2026) plus
  CP.21's *additional* short-circuit level (SCL) requirement priced at the Phase 2 and 3 unit
  cost. CP30 Annex 3 s.2.3.1 says the additional need is "in addition to the volumes already
  procured through our stability pathfinder projects and mid-term stability markets".
* **Reactive power** (commercial contracts): today's book plus CP.21's additional absorption
  and injection requirement priced at the Voltage 2026 tender's unit cost. The obligatory
  service (ORPS) carries today's need and is held.
* **Voltage and inertia BM actions**: held at outturn. CP30 Annex 4 s.2.1.5 costs voltage and
  stability balancing as "a single projection over all pathways, based on NESO's balancing cost
  annual report 2024", and EC2 lists them as no pathway variable: CP30 itself holds them.
* **Response**: NESO's 2030 response need against 2025 (4.2 to 4.8 GW, Operability Strategy
  Report March 2025 p.19; CP30 Annex 3 s.2.3.2), at FY2024/25 clearing prices.
* **Reserve**: CP30 Annex 3's 2030 reserve need (6.4 GW positive, 5.2 GW negative) against
  2025's (4.4 and 2.4 GW, OSR March 2025 p.19), which NESO derives from the largest loss and the
  wind and solar forecast error of the CP30 fleet; at FY2024/25 prices.
* **Restoration**: the Electricity System Restoration Standard, live from end-2026, as the
  2050 table (x1.5); the rest held, or zero where the service is closed.

Low and high: CP.21's ranges give their ends (midpoints are central); response and reserve take
price factors and NESO's other 2030 volume (OSR: 6.5 GW negative reserve); the held lines take
declared factors. All spend is 2024 money (FY2024/25 nominal treated as 2024 money, as the
2050 table); contract unit costs are their published totals over their terms, not indexed.

Usage::

    python scripts/public_ownership/ancillary_services_2030.py [--check]

rewrites the table's spend columns, projection basis and the stability provider rows from the
FY2024/25 outturn columns (``--check`` fails if the file differs).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLE = ROOT / "data/public_ownership/financing/ancillary_services_2030.csv"
WORKBOOK = ROOT / "data/CP30/Clean Power 2030 Data Workbook_V003.xlsx"
CASES = ("central", "low", "high")
SPEND = {case: f"spend_{case}_gbp_m_2024" for case in CASES}
DECISION = "2030 (user decision, 6 Oct 2026): CP30-driven projection of the FY2024/25 outturn, 2024 money."
PROVIDER_NAMES = ("Provider names are the CP30 register's: CCGT (2050 table: CCS Gas), peaking_gas = OCGT and gas_engine "
                  "(2050 table: H2_turbine), biomass (2050 table: CCS Biomass).")

#: CP.21 (CP30 data workbook V003): additional SCL (column F) and reactive absorption (D) and
#: injection (E) requirements by region, rows 5 to 16; the published bands, as printed.
CP21 = {
    "sheet": "CP.21", "cells": "B4:F16",
    "title": "Additional short circuit level requirements for system strength and additional reactive power requirements",
    "regions": {
        "North Scotland": {"absorption_mvar": "1-200", "injection_mvar": 0, "scl_mva": "10001-15000"},
        "South Scotland": {"absorption_mvar": 0, "injection_mvar": 0, "scl_mva": "100-1000"},
        "NE England": {"absorption_mvar": 0, "injection_mvar": 0, "scl_mva": "1001-5000"},
        "East Midlands": {"absorption_mvar": "1-200", "injection_mvar": "501-1100", "scl_mva": "5001-10000"},
        "NW England": {"absorption_mvar": "801-1300", "injection_mvar": "1101-1200", "scl_mva": 0},
        "West Midlands": {"absorption_mvar": None, "injection_mvar": None, "scl_mva": None},
        "East England": {"absorption_mvar": "201-400", "injection_mvar": "1201-1300", "scl_mva": 0},
        "Greater London": {"absorption_mvar": "1-200", "injection_mvar": "1-500", "scl_mva": 0},
        "South Wales": {"absorption_mvar": "801-1300", "injection_mvar": 0, "scl_mva": "5001-10000"},
        "West England": {"absorption_mvar": None, "injection_mvar": None, "scl_mva": None},
        "SW England": {"absorption_mvar": "1-200", "injection_mvar": 0, "scl_mva": "1001-5000"},
        "SE England": {"absorption_mvar": 0, "injection_mvar": 0, "scl_mva": "1001-5000"},
    },
    "blank_regions_note": ("West Midlands and West England carry no values in CP.21 although Annex 3 s.2.3.3 names "
                           "both among the high-voltage (and the West Midlands among the low-voltage) support areas; "
                           "they count as zero here, so the reactive requirement is a floor."),
}

#: NESO's response and reserve requirement, 2025 and 2030 (GW).
FREQUENCY = {
    "response_2025_gw": 4.2, "response_2030_gw": 4.8,
    "positive_reserve_2025_gw": 4.4, "positive_reserve_2030_gw": 6.4,
    "negative_reserve_2025_gw": 2.4, "negative_reserve_2030_gw": 5.2, "negative_reserve_2030_high_gw": 6.5,
    "source": ("NESO, Clean Power 2030 Annex 3 (Nov 2024) s.2.3.2: in 2030 'approximately 4.8 GW of response' and "
               "'reserve need is around 11.6 GW: 6.4 GW of positive reserve ... and 5.2 GW of negative reserve'; NESO, "
               "Operability Strategy Report (March 2025) p.19: by 2030 against 2025 'Response increase from 4.2 to 4.8 GW' "
               "and 'Reserve increase from 6.8 to 12.9 GW, including 4.4 to 6.4 GW of positive reserve ... and 2.4 to "
               "6.5 GW of negative reserve'"),
    "source_url": "https://www.neso.energy/document/346801/download | https://www.neso.energy/document/358121/download",
}

#: Contract unit costs (published totals over their terms; GBP m, treated as 2024 money).
STABILITY_BOOK = {
    "phase2_gbp_m_per_year": 323.0 / 10, "phase2_scl_mva": 11550.0,
    "phase3_gbp_m_per_year": 1300.0 / 10, "phase3_scl_mva": 8720.0, "phase3_inertia_gva_s": 17.08,
    "mid_term_y1_gbp_m_per_year": 10.3, "mid_term_y1_inertia_gva_s": 7.3,
    "source": ("Stability pathfinder Phase 2 (Scotland, ten contracts, GBP 323 m of forward revenue, 2024-2034; 6.75 GVA.s "
               "and 11.55 GVA SCL); Phase 3 (England and Wales, GBP 1.3 bn, 2025-2035; 17.08 GVA.s and 8.72 GVA SCL, "
               "Utility Week and Modern Power Systems); Mid-term (Y-1) Round 2 (GBP 10.3 m for 7.3 GVA.s, Oct 2026 to "
               "Sep 2027). Phase 1 contracts end April 2026 (Modo Energy) and are not in the 2030 book."),
    "source_url": ("https://utilityweek.co.uk/eso-awards-stability-contracts-worth-1-3-billion/ | "
                   "https://www.neso.energy/news/new-contracts-awarded-under-mid-term-y-1-stability-market-round-2 | "
                   "https://modoenergy.com/research/stability-pathfinders-inertia-short-circuit-level-battery-energy-storage"),
}
REACTIVE_UNIT = {
    "voltage_2026_gbp_m": 83.0, "voltage_2026_years": 10, "voltage_2026_mvar": 646.0,
    "source": "NESO, Voltage 2026 tenders awarded: four contracts, 646 Mvar (North of England and London), GBP 83 m, 2026-2036",
    "source_url": "https://www.neso.energy/news/voltage-2026-tenders-awarded",
}
#: FY2025/26 voltage BM outturn over FY2024/25 (nominal), the bill's existing high-side reference.
VOLTAGE_TREND_2025_26 = 212363735.0 / 153851930.0
#: Response and reserve availability prices: ABSS 2024/25 reports clearing prices down by up to 50%;
#: CP30 ES1 has 27.4 GW of batteries in 2030 against 8.0 GW in 2024 (FFR, Capacity (MW), Storage, Battery).
PRICE = {"response": {"low": 0.5, "high": 1.5}, "reserve": {"low": 0.75, "high": 1.25}}
RESTORATION = {"central": 1.5, "low": 1.0, "high": 3.0}


def band(value) -> tuple[float, float]:
    """A CP.21 cell as (low end, high end): ``'1001-5000'`` -> (1001, 5000); 0 or blank -> (0, 0)."""
    if value is None or value == "" or value == 0:
        return 0.0, 0.0
    text = str(value).replace(" ", "")
    if "-" not in text:
        return float(text), float(text)
    lo, hi = text.split("-", 1)
    return float(lo), float(hi)


def cp21_totals(regions: dict | None = None) -> dict:
    """GB totals of CP.21's additional requirements: low ends, midpoints and high ends."""
    regions = CP21["regions"] if regions is None else regions
    out = {}
    for key in ("scl_mva", "absorption_mvar", "injection_mvar"):
        bands = [band(r[key]) for r in regions.values()]
        out[key] = {"low": sum(b[0] for b in bands), "central": sum((b[0] + b[1]) / 2 for b in bands),
                    "high": sum(b[1] for b in bands)}
    out["reactive_mvar"] = {c: out["absorption_mvar"][c] + out["injection_mvar"][c] for c in CASES}
    return out


def read_cp21(path: Path = WORKBOOK) -> dict:
    """CP.21 as printed in the workbook (region name -> the three cells), for the tests."""
    import openpyxl

    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True)[CP21["sheet"]]
    out = {}
    for row in sheet.iter_rows(min_row=5, max_row=16, min_col=3, max_col=6, values_only=True):
        if row[0]:
            out[str(row[0]).strip()] = {"absorption_mvar": row[1], "injection_mvar": row[2], "scl_mva": row[3]}
    return out


def unit_costs() -> dict:
    """GBP per MVA of SCL a year (Phase 2 and 3 together) and per Mvar a year (Voltage 2026)."""
    book = STABILITY_BOOK
    scl = (book["phase2_gbp_m_per_year"] + book["phase3_gbp_m_per_year"]) * 1e6 / (book["phase2_scl_mva"] + book["phase3_scl_mva"])
    reactive = REACTIVE_UNIT["voltage_2026_gbp_m"] * 1e6 / REACTIVE_UNIT["voltage_2026_years"] / REACTIVE_UNIT["voltage_2026_mvar"]
    return {"scl_gbp_per_mva_year": scl, "reactive_gbp_per_mvar_year": reactive}


def ratios() -> dict:
    f = FREQUENCY
    return {
        "response": f["response_2030_gw"] / f["response_2025_gw"],
        "positive": f["positive_reserve_2030_gw"] / f["positive_reserve_2025_gw"],
        "negative": f["negative_reserve_2030_gw"] / f["negative_reserve_2025_gw"],
        "negative_high": f["negative_reserve_2030_high_gw"] / f["negative_reserve_2025_gw"],
        "both": (f["positive_reserve_2030_gw"] + f["negative_reserve_2030_gw"])
                / (f["positive_reserve_2025_gw"] + f["negative_reserve_2025_gw"]),
        "both_high": (f["positive_reserve_2030_gw"] + f["negative_reserve_2030_high_gw"])
                     / (f["positive_reserve_2025_gw"] + f["negative_reserve_2025_gw"]),
    }


def stability_book_gbp_m() -> float:
    b = STABILITY_BOOK
    return b["phase2_gbp_m_per_year"] + b["phase3_gbp_m_per_year"] + b["mid_term_y1_gbp_m_per_year"]


def stability_providers() -> dict:
    """Central-case shares of the 2030 stability spend: Phase 3 synchronous condensers, Phase 2 half
    synchronous condensers and half grid-forming batteries (five of each of its ten contracts), the
    mid-term market to ``other``, and the additional SCL in the Phase 2 and 3 money mix."""
    b = STABILITY_BOOK
    totals = cp21_totals()
    extra = totals["scl_mva"]["central"] * unit_costs()["scl_gbp_per_mva_year"] / 1e6
    book23 = b["phase2_gbp_m_per_year"] + b["phase3_gbp_m_per_year"]
    sync_share_23 = (b["phase3_gbp_m_per_year"] + b["phase2_gbp_m_per_year"] / 2) / book23
    total = stability_book_gbp_m() + extra
    sync = b["phase3_gbp_m_per_year"] + b["phase2_gbp_m_per_year"] / 2 + extra * sync_share_23
    battery = b["phase2_gbp_m_per_year"] / 2 + extra * (1 - sync_share_23)
    other = b["mid_term_y1_gbp_m_per_year"]
    return {"synchronous_condenser": round(sync / total, 4), "Battery": round(battery / total, 4),
            "other": round(1 - round(sync / total, 4) - round(battery / total, 4), 4)}


def rules() -> dict:
    """service -> (central, low, high) multipliers on outturn, or absolute values, and the basis text."""
    r, p = ratios(), PRICE
    t, u = cp21_totals(), unit_costs()
    book = stability_book_gbp_m()
    sb = STABILITY_BOOK
    alt = {"phase3": sb["phase3_gbp_m_per_year"] * 1e6 / sb["phase3_scl_mva"],
           "phase2": sb["phase2_gbp_m_per_year"] * 1e6 / sb["phase2_scl_mva"]}
    scl = {c: book + t["scl_mva"][c] * u["scl_gbp_per_mva_year"] / 1e6 for c in CASES}
    reactive = {c: t["reactive_mvar"][c] * u["reactive_gbp_per_mvar_year"] / 1e6 for c in CASES}
    resp = {"central": r["response"], "low": r["response"] * p["response"]["low"], "high": r["response"] * p["response"]["high"]}
    resp_text = (f"Response: NESO's 2030 need over 2025's, 4.8 / 4.2 GW = x{r['response']:.4f} (CP30 Annex 3 s.2.3.2; OSR "
                 f"Mar 2025 p.19: high 2.1 to 2.2 GW, low 2.1 to 2.6 GW; largest secured loss 1,800 MW, inertia floor 102 GVA.s), "
                 f"at FY2024/25 clearing prices. Low = price x{p['response']['low']} (ABSS: clearing prices down by up to 50%; "
                 f"CP30 ES1 27.4 GW of batteries in 2030 against 8.0 GW in 2024). High = price x{p['response']['high']}.")
    pos = {"central": r["positive"], "low": r["positive"] * p["reserve"]["low"], "high": r["positive"] * p["reserve"]["high"]}
    both = {"central": r["both"], "low": r["both"] * p["reserve"]["low"], "high": r["both_high"] * p["reserve"]["high"]}
    reserve_text = (f"Reserve: CP30 Annex 3 s.2.3.2's 2030 need over 2025's (OSR Mar 2025 p.19), positive 6.4 / 4.4 GW = "
                    f"x{r['positive']:.4f}, negative 5.2 / 2.4 GW = x{r['negative']:.4f}, both 11.6 / 6.8 GW = x{r['both']:.4f}; "
                    "NESO sizes reserve on the largest loss and the wind and solar forecast error (CP30 ES1 wind and solar "
                    "125.3 GW in 2030 against 52.2 GW in 2024; a pure capacity scaling, x2.4, ignores forecasting gains and "
                    f"diversity, which NESO's figure includes). FY2024/25 prices. Low = price x{p['reserve']['low']} (battery "
                    f"competition); high = OSR's 6.5 GW of negative reserve and price x{p['reserve']['high']}.")
    out = {
        **{s: (resp, resp_text) for s in ("dynamic_containment", "dynamic_moderation", "dynamic_regulation",
                                           "static_firm_frequency_response", "mandatory_frequency_response",
                                           "hydro_frequency_services")},
        "enhanced_frequency_response": ({"central": 0.0, "low": 0.0, "high": 1.0},
                                        "Legacy contracts, 'no longer actively procured' (ABSS): zero in 2030; high held flat."),
        "fast_reserve_optional": ({"central": 0.0, "low": 0.0, "high": 0.0},
                                  "Optional Fast Reserve closes December 2025 and its volume moves to Quick Reserve (NESO "
                                  "reserve webinar, Oct 2024); its FY2024/25 spend is in the quick_reserve base. Zero."),
        "quick_reserve": ({c: v * (8.95 + 103.04) / 8.95 for c, v in both.items()},
                          "Base = Quick Reserve 8.95 + Optional Fast Reserve 103.04 = 111.99 (QR replaces OFR; QR buys positive "
                          "and negative), x the both-direction ratio. " + reserve_text),
        "stor_and_slow_reserve": (pos, "Positive reserve (Slow Reserve succeeds STOR). " + reserve_text),
        "balancing_reserve": (pos, "Positive Balancing Reserve (the FY2024/25 service). " + reserve_text),
        "pumped_storage_bilateral_reserve": ({"central": 1.0, "low": 0.5, "high": r["positive"]},
                                             "Bilateral contract on the existing pumped-storage fleet (capacity-limited; the "
                                             "added 2030 need is bought in the Quick and Slow Reserve markets): held flat. Low "
                                             "x0.5 (absorbed into the markets); high x the positive-reserve ratio."),
        "gt_fast_start_and_hydro_rapid_start": ({"central": 0.0, "low": 0.0, "high": 1.0},
                                                "Legacy, 'no longer actively procured' (ABSS): zero; high held flat."),
        "bm_start_up_and_super_sel": ({"central": 0.0, "low": 0.0, "high": 1.0},
                                      "Legacy, no longer actively procured (ABSS): zero; high held flat."),
        "operating_reserve_trades": (pos, "Positive reserve traded across interconnectors. " + reserve_text),
        "negative_reserve_trades": ({"central": r["negative"], "low": r["negative_high"], "high": 1.0},
                                    f"Net receipt to NESO, x the negative-reserve ratio (x{r['negative']:.4f}); low = OSR's "
                                    f"6.5 GW (x{r['negative_high']:.4f}, more receipts); high = FY2024/25 receipts held."),
        "demand_flexibility_service": ({"central": 1.0, "low": 0.0, "high": 11.0 / 0.77},
                                       "Held flat; low 0; high about 11m (the FY2023/24 level implied by the 93% fall in ABSS)."),
        "interconnector_ntc_payments": ({"central": 1.0, "low": 0.0, "high": 2.0}, "Held flat; low 0; high x2."),
        "stability_network_services": ({c: ("abs", v) for c, v in scl.items()},
                                       f"Contracted book live in 2030 GBP {book:.2f} m (Phase 2 {STABILITY_BOOK['phase2_gbp_m_per_year']:.2f}, "
                                       f"Phase 3 {STABILITY_BOOK['phase3_gbp_m_per_year']:.2f}, mid-term Y-1 "
                                       f"{STABILITY_BOOK['mid_term_y1_gbp_m_per_year']:.2f}; Phase 1 ends Apr 2026) plus CP.21's "
                                       f"additional SCL, {t['scl_mva']['central']:,.0f} MVA at band midpoints (low ends "
                                       f"{t['scl_mva']['low']:,.0f}, high ends {t['scl_mva']['high']:,.0f}), at the Phase 2 and 3 "
                                       f"cost of GBP {u['scl_gbp_per_mva_year']:,.0f} per MVA a year (GBP 162.3 m a year for "
                                       "20,270 MVA, inertia included). CP30 Annex 3: the additional need is beyond the volumes "
                                       "the pathfinders and the mid-term market already hold. At Phase 3's own cost (GBP "
                                       f"{alt['phase3']:,.0f}/MVA) the central would be GBP "
                                       f"{book + t['scl_mva']['central'] * alt['phase3'] / 1e6:.1f} m; at Phase 2's (GBP "
                                       f"{alt['phase2']:,.0f}/MVA) GBP {book + t['scl_mva']['central'] * alt['phase2'] / 1e6:.1f} m."),
        "inertia_bm_actions": ({"central": 1.0, "low": 0.5, "high": 35.0 / 15.0},
                               "Held at outturn: CP30 Annex 4 s.2.1.5 costs voltage and stability balancing as one "
                               "projection based on NESO's 2024 balancing cost report (EC2: no pathway variable). Low x0.5 "
                               "(Phase 2 and 3 add 23.8 GVA.s and the floor falls 120 to 102 GVA.s). High x35/15: CP30's "
                               "2030 peak residual inertia need (35 GVA.s, Annex 3 s.2.3.1, met by the markets and the BM) "
                               "over the 2026/27 mid-term requirement (15 GVA.s, half met in the BM)."),
        "voltage_bm_actions": ({"central": 1.0, "low": 0.5, "high": VOLTAGE_TREND_2025_26},
                               "Held at outturn: CP30 Annex 4 s.2.1.5 (one projection from NESO's 2024 balancing cost "
                               "report; EC2: no pathway variable). Low x0.5 (CP.21's new reactive assets also displace "
                               f"today's BM synchronisations). High x{VOLTAGE_TREND_2025_26:.4f}, the FY2025/26 outturn "
                               "(GBP 212.4 m) over FY2024/25."),
        "obligatory_reactive_power_service": ({"central": 1.0, "low": 0.7, "high": VOLTAGE_TREND_2025_26},
                                              "Today's reactive need, paid at the default rate, held: CP.21's additional "
                                              "requirement is bought commercially (commercial_reactive_and_voltage_network_"
                                              "services). Low x0.7 (TO reactive compensation under RIIO-T3, voltage markets). "
                                              f"High x{VOLTAGE_TREND_2025_26:.4f}, the FY2025/26 voltage cost trend."),
        "commercial_reactive_and_voltage_network_services": (
            {c: ("abs", 8.11 + v) for c, v in reactive.items()},
            f"Today's book (Mersey to 2031, Pennine to 2034: GBP 8.11 m) plus CP.21's additional reactive requirement, "
            f"absorption + injection {t['reactive_mvar']['central']:,.0f} Mvar at band midpoints (low ends "
            f"{t['reactive_mvar']['low']:,.0f}, high ends {t['reactive_mvar']['high']:,.0f}), at the Voltage 2026 tender's "
            f"GBP {u['reactive_gbp_per_mvar_year']:,.0f} per Mvar a year (GBP 83 m over ten years for 646 Mvar, taken to be "
            "part of the CP.21 need). " + CP21["blank_regions_note"]),
        "electricity_system_restoration": (RESTORATION,
                                           "x1.5 for the Electricity System Restoration Standard (60% of demand in every "
                                           "region in 24 h, 100% in 5 days, from end-2026; CP30 Annex 3 s.2.3.5), as the 2050 "
                                           "table: the Northern regions tender (GBP 53.8 m over 3 years for 3 of 7 regions) is "
                                           "about 42 m a year pro rata for 7. Low = flat. High = x3."),
    }
    return out


def project(table: pd.DataFrame) -> pd.DataFrame:
    """The table with its spend columns, projection basis and stability providers set by :func:`rules`."""
    spec = rules()
    missing = sorted(set(table["service"]) - set(spec))
    if missing:
        raise ValueError(f"no 2030 rule for {missing}")
    rows = []
    for service, group in table.groupby("service", sort=False):
        factors, text = spec[service]
        outturn = float(group["outturn_spend_gbp_m"].iloc[0])
        if service == "stability_network_services":
            template = group.iloc[0].to_dict()
            shares = stability_providers()
            group = pd.DataFrame([{**template, "provider_carrier": k, "participation_share": v} for k, v in shares.items()])
            group["share_basis"] = ("ASSUMPTION (NESO publishes no payment by contract; user decision, 6 Oct 2026): 2030 money "
                                    "by contract: Phase 3 (all synchronous condensers) and half of Phase 2 to synchronous "
                                    "condensers, the other half of Phase 2 to its five grid-forming batteries, the mid-term "
                                    "Y-1 market to other, and the CP.21 additional SCL in the Phase 2 and 3 money mix "
                                    f"(central shares: {shares}). Phase 1 (Cruachan, Triton) ends April 2026.")
        group = group.copy()
        for case in CASES:
            value = factors[case]
            group[SPEND[case]] = round(value[1] if isinstance(value, tuple) else outturn * value, 4)
        group["projection_basis"] = f"{DECISION} {text} {PROVIDER_NAMES}"
        group["model_year"] = 2030
        rows.append(group)
    return pd.concat(rows, ignore_index=True)[list(table.columns)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", default=str(TABLE))
    parser.add_argument("--check", action="store_true", help="fail if the table is not the projection")
    args = parser.parse_args(argv)
    table = pd.read_csv(args.table)
    out = project(table)
    if args.check:
        pd.testing.assert_frame_equal(out.reset_index(drop=True), table.reset_index(drop=True), check_dtype=False)
        print(f"{args.table}: up to date")
        return 0
    out.to_csv(args.table, index=False, lineterminator="\n")
    print(f"wrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
