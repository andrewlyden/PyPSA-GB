"""The FES 2025 figures a long-range bill year reads, for any model year (FES 2040 plan, 11 Nov 2026).

The 2050 bill-input builder (task Y13) hard-coded its FES figures with their sheet and row.
This module reads the same figures on the same definitions for any year Y, so one builder
(:mod:`scripts.consumer_bill.build_fes_bill_inputs`) serves every year. Each definition reproduces
the 2050 constants exactly (``tests/unit/test_fes_bill_inputs.py``).

FES year Y is fiscal April Y to March Y+1 (ED1 "Annual [Fiscal]"); calendar Y = 0.75 FY(Y) +
0.25 FY(Y-1) (Y1's rule, :func:`scripts.FES.fes2025_targets.ed1_demand`).

* Peak: ED1 row 134 (GBFES Peak Customer Demand: Total Consumption plus Losses), FY.
* Domestic customers: Dem_BB001a, Holistic Transition, summed over GSPs (``FES_2025_data.csv``).
* Consumer demand excluding electrolysis: ED1 rows 14 + 79 + 84 + 119, FY.
* Residential and residential district heat (ED1 rows 14, 59), calendar: the WHD base.
* Household base: ED1 rows 19 and 24 (appliances, lighting), FY.
* Heat-pump and district-heat electricity: ED7 Annual electricity by sector, five technologies.
* SCOP and EV efficiency gains: F.05 AB9 and V12 (SCOP improvements, EV efficiency). F.05
  publishes them for 2050 only (2024 -> 2050); for another year its own formulas are evaluated
  on that year's ED7/ED5 electricity with each efficiency interpolated linearly 2024 -> 2050
  (FES's convention for end-use efficiencies, Energy Flow Diagram: "values between 2024 and
  2050 are interpolated"), HGV electricity split into F.05's two classes by their 2050 shares.
* Road transport electricity: Energy Flow Diagram row 22 (= F.05 R12 in 2050).
* Per-unit heat pump and car: ED7 rows 7 / 445; ED5 rows 201 / 81.
* Recovery volumes: ED1 row 9 less row 74 (calendar), ES1 DACCS, ED1 row 74 (calendar), WS1
  non-power hydrogen (calendar), and the FY values of the mapping block.

Every value is rounded to 6 decimals (the workbook's figures carry at most 5), so a calendar
blend's last-bit float noise does not reach the committed files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WORKBOOK = "data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx"
BUILDING_BLOCKS = "resources/FES/FES_2025_data.csv"
PATHWAY = "Holistic Transition"
#: The year F.05 publishes its efficiency gains for (2024 -> 2050).
F05_YEAR = 2050
F05_BASE_YEAR = 2024
HP_TECHS = ("ASHP", "Low carbon district heating", "GSHP", "Hybrid (ASHP + BioLPG boiler)",
            "Hybrid (ASHP + Electric resistive)")
#: F.05 AB9's SCOP endpoints (today -> 2050): ASHP-type and district heat; GSHP.
SCOP_ASHP_DH = (2.9, 3.5)
SCOP_GSHP = (3.25, 3.92)
#: F.05 V12's vehicle classes: (label, efficiency 2024, efficiency 2050, ED5 item or None, F.05's hard-coded 2050 TWh).
EV_CLASSES = (
    ("buses", 1.1804, 1.0094, "TWh Buses - Electricity", None),
    ("cars", 0.2802, 0.2424, "TWh Cars - Electricity", None),
    ("hgv_class_1", 1.5026, 1.285, None, 6.6),
    ("hgv_class_2", 2.5061, 2.1431, None, 27.87),
    ("motorbikes", 0.0659, 0.06253, "TWh Motorbikes - Electricity", None),
    ("vans", 0.3864, 0.3613, "TWh Vans - Electricity", None),
)
H2_PARTS = ("Road Transport", "Residential", "Commercial", "Industrial", "Rail", "Shipping", "Aviation", "Blending",
            "Direct Air Carbon Capture and Storage", "Transmission Losses")
DECIMALS = 6


def _r(x) -> float:
    return round(float(x), DECIMALS)


def _efd_column(year: int) -> str:
    """The Energy Flow Diagram column of a year: N is 2024, one column a year."""
    from openpyxl.utils import column_index_from_string, get_column_letter

    return get_column_letter(column_index_from_string("N") + int(year) - F05_BASE_YEAR)


def read(year: int, root: str | Path = ".") -> dict:
    """Every figure the bill year ``year`` reads (and Y-1 and 2030 where its rules use them).

    Returns:
        ``{"year", "peak_gw", "customers", "consumer_gwh", "res_cal_gwh", "dh_res_cal_gwh", "app_gwh",
        "light_gwh", "hp_dh_twh", "scop_saving_twh", "scop_basis", "ev_saving_twh", "road_twh", "ev_basis",
        "ashp_twh", "ashp_units", "bev_twh", "bev_cars", "sys_excl_cal_gwh", "consumer_cal_gwh",
        "losses_cal_gwh", "daccs_gwh", "electrolysis_cal_gwh", "h2_non_power_parts", "fy_values_gwh",
        "industrial_excl_electrolysis_fy_twh", "curtailment_gross_twh", "upper_bound"}``.
    """
    import openpyxl

    from scripts.FES import fes2025_targets as ft

    root = Path(root)
    y = int(year)
    years = sorted({2030, y - 1, y})
    workbook = root / WORKBOOK
    shared = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    real_load = openpyxl.load_workbook

    class _Shared:  # one read-only handle for every helper (they close after each table)
        def __getattr__(self, name):
            return getattr(shared, name)

        def __getitem__(self, key):
            return shared[key]

        def close(self):
            pass

    openpyxl.load_workbook = lambda *a, **k: _Shared()  # noqa: E731
    try:
        ed1 = ft.read_ed1(workbook)
        demand = {t: ft.ed1_demand(ed1, PATHWAY, t) for t in years}
        ed5 = ft.read_sheet_table(workbook, "ED5", ("Data item", "Unit", "Pathway", "Fuel"))
        ed5 = ed5[ed5["Pathway"].map(ft._norm).eq(PATHWAY)]
        ed7 = ft.read_sheet_table(workbook, "ED7", ("Pathway", "Sector/Aggregation Level", "Fuel", "Technology",
                                                    "Data Type", "Units"))
        ed7 = ed7[ed7["Pathway"].map(ft._norm).eq(PATHWAY)]
        efd = shared["Energy Flow Diagram"]
        road = float(efd[f"{_efd_column(y)}22"].value)
        f05 = shared["F.05"]
        f05_ab9, f05_v12, f05_r12 = (float(f05[c].value) for c in ("AB9", "V12", "R12"))
        es1, ws1, flx1 = ft.read_es1(workbook), ft.read_ws1(workbook), ft.read_flx1(workbook)
        classified = ft.classify_es1(es1, PATHWAY, y)
        daccs = ft.supply_tables(classified)["daccs_gwh"]
        h2 = ft.hydrogen_quantities(ws1, flx1, classified, PATHWAY, y)["annual_twh_calendar"]["demand"]
    finally:
        openpyxl.load_workbook = real_load
        shared.close()

    def peak(t):
        p = demand[t]["peak_gw"]["GBFES Peak Customer Demand: Total Consumption plus Losses"]
        if int(p["excel_row"]) != 134:
            raise ValueError(f"ED1 peak row moved: {p['excel_row']}")
        return p["gw"]

    tot = lambda t, k, b: demand[t]["totals"][k][b]  # noqa: E731
    ann = lambda t, k, b: demand[t]["annual"][k][b]  # noqa: E731
    bb = pd.read_csv(root / BUILDING_BLOCKS, usecols=["Building Block ID Number", "FES Pathway", *map(str, years)])
    bb = bb[bb["Building Block ID Number"].eq("Dem_BB001a") & bb["FES Pathway"].eq(PATHWAY)]
    customers = {t: _r(pd.to_numeric(bb[str(t)], errors="coerce").fillna(0.0).sum()) for t in years}

    electricity = ed7[ed7["Fuel"].map(ft._norm).eq("electricity") & ed7["Data Type"].map(ft._norm).eq("Annual")]
    tech = electricity["Technology"].map(ft._norm)
    hp_dh = {}
    for sector in ("Residential", "Commercial", "Industrial"):
        hp_dh[sector.lower()] = {}
        for name in HP_TECHS:
            row = electricity[electricity["Sector/Aggregation Level"].map(ft._norm).eq(sector) & tech.eq(name)]
            if len(row) != 1:
                raise ValueError(f"ED7: {sector} {name} not found once")
            hp_dh[sector.lower()][name] = _r(row.iloc[0][str(y)])

    def ed5_item(label):
        row = ed5[ed5["Data item"].map(ft._norm).eq(label)]
        if len(row) != 1:
            raise ValueError(f"ED5: {label!r} not found once")
        return float(row.iloc[0][str(y)])

    def ed_row(table, excel_row):
        row = table[table["excel_row"].eq(excel_row)]
        if len(row) != 1:
            raise ValueError(f"row {excel_row} not found once")
        return float(row.iloc[0][str(y)])

    def scop(fraction):
        ashp = float(pd.to_numeric(electricity.loc[tech.str.contains("ASHP", case=False), str(y)]).sum())
        dh = float(pd.to_numeric(electricity.loc[tech.eq("Low carbon district heating"), str(y)]).sum())
        gshp = float(pd.to_numeric(electricity.loc[tech.eq("GSHP"), str(y)]).sum())
        return ((ashp + dh) * fraction * (SCOP_ASHP_DH[1] / SCOP_ASHP_DH[0] - 1)
                + gshp * fraction * (SCOP_GSHP[1] / SCOP_GSHP[0] - 1))

    def ev(fraction):
        hgv_total, hgv_hard = ed5_item("TWh HGVs - Electricity"), 6.6 + 27.87
        total = 0.0
        for _, e0, e1, item, hard in EV_CLASSES:
            twh = ed5_item(item) if item is not None else (hard if y == F05_YEAR else hgv_total * hard / hgv_hard)
            total += fraction * (e0 - e1) / e0 * twh
        return total

    fraction = (y - F05_BASE_YEAR) / (F05_YEAR - F05_BASE_YEAR)
    if y == F05_YEAR:
        scop_saving, ev_saving, road_twh = f05_ab9, f05_v12, f05_r12
        scop_basis = ev_basis = "published"
    else:
        scop_saving, ev_saving, road_twh = scop(fraction), ev(fraction), road
        scop_basis = ev_basis = "interpolated"
    curtailment = sum(float(r.get("curtailment_gwh") or 0.0) for r in _targets(root, y)["generation"]) / 1e3
    return {
        "year": y,
        "peak_gw": {t: _r(peak(t)) for t in years},
        "customers": customers,
        "consumer_gwh": {t: _r(tot(t, "consumer_excluding_electrolysis", "fy_gwh")) for t in years},
        "res_cal_gwh": _r(ann(y, "residential", "calendar_gwh")),
        "dh_res_cal_gwh": _r(ann(y, "residential_district_heat", "calendar_gwh")),
        "app_gwh": _r(ann(y, "residential_appliances", "fy_gwh")),
        "light_gwh": _r(ann(y, "residential_lighting", "fy_gwh")),
        "hp_dh_twh": hp_dh,
        "scop_saving_twh": float(scop_saving), "scop_basis": scop_basis,
        "ev_saving_twh": float(ev_saving), "road_twh": _r(road_twh), "ev_basis": ev_basis,
        "interpolation_fraction": fraction,
        "upper_bound": {"scop_saving_twh": scop(1.0), "ev_saving_twh": ev(1.0)},
        "ashp_twh": _r(ed_row(ed7, 7)), "ashp_units": _r(ed_row(ed7, 445)),
        "bev_twh": float(ed_row(ed5, 201)), "bev_cars": _r(ed_row(ed5, 81)),
        "sys_excl_cal_gwh": _r(tot(y, "system_excluding_electrolysis", "calendar_gwh")),
        "consumer_cal_gwh": _r(tot(y, "consumer_excluding_electrolysis", "calendar_gwh")),
        "losses_cal_gwh": _r(tot(y, "losses", "calendar_gwh")),
        "daccs_gwh": _r(daccs),
        "electrolysis_cal_gwh": _r(tot(y, "electrolysis", "calendar_gwh")),
        "h2_non_power_parts": [_r(h2[k]) for k in H2_PARTS],
        "fy_values_gwh": {"system_total": _r(ann(y, "system_total", "fy_gwh")),
                          "networked_electrolysis": _r(ann(y, "networked_electrolysis", "fy_gwh")),
                          "consumer_excluding_electrolysis": _r(tot(y, "consumer_excluding_electrolysis", "fy_gwh")),
                          "losses": _r(tot(y, "losses", "fy_gwh"))},
        "industrial_excl_electrolysis_fy_twh": ann(y, "industrial_excluding_electrolysis", "fy_gwh") / 1e3,
        "curtailment_gross_twh": curtailment,
    }


def _targets(root: Path, year: int) -> dict:
    import json

    path = root / f"resources/FES/fes2025_holistic_transition_{int(year)}_targets.json"
    return json.loads(path.read_text(encoding="utf-8"))
