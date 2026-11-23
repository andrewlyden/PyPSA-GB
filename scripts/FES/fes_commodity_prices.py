"""FES commodity prices (sheets CP1 and CP2) read from the Data Workbook, with no fallback.

Task Y1. The old extractor (``extract_FES_prices.py`` before 30 Sep 2026) could not read the
FES 2025 layout and silently wrote built-in placeholders (gas = coal = oil = biomass rising
20 to 32 GBP/MWh, carbon 30 to 110). This module reads the published tables and fails
loudly if it cannot.

**Layout** (FES 2023 to 2025). Each commodity is a block: a title cell such as "Wholesale
NBP gas price (2025 EUR/MWh)", then a row starting "Year" with the year columns, then one
row per case ("Base Case", "High Case", "History", ...). FES 2025 publishes one base case
for every pathway. CP1 rows 99 onward hold the conversion table used by NESO: GBP to EUR
by year and a GDP deflator rebased to the workbook's price year.

**Conversion to model money (declared).**

1. EUR (real P) to GBP (real P): divide by NESO's own GBP-to-EUR rate for year P from the
   CP1 conversion table (FES 2025: 1.17 in 2025). Using the workbook's own rate reverses
   NESO's own conversion, so the GBP figure is the one NESO started from.
2. GBP real P to the model's price base (GBP real 2024, ``marginal_costs.future_costs.
   price_year`` in ``config/defaults/base/marginal_costs.yaml``): divide by the CPIH ratio index(P) /
   index(2024), the same CPIH the consumer bill uses to rebase model money to 2025
   (2025 = 138.0, 2024 = 132.9, factor 1.0384; ``config/defaults/base/public_ownership/consumer_bill.yaml`` consumer-bill CPIH
   path and ``scripts/market/fti_settlement.py`` PRICE_BASIS). NESO's own GDP deflator
   (2024 = 0.976 on 2025 = 1, i.e. 1.0246) is returned beside it for comparison, not used.

FES 2025 HT 2050 in model money: gas 28.72 EUR25/MWh / 1.17 / 1.0384 = 23.64 GBP24/MWh;
GB carbon 134.61 EUR25/t -> 110.80 GBP24/t.
"""

import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


MODEL_PRICE_YEAR = 2024
# CPIH (2015 = 100) annual averages: ONS series L522 (MM23), as the consumer bill uses to rebase model money
# (config/defaults/base/public_ownership/consumer_bill.yaml: 2025 = 138.0, 2024 = 132.9). 2019 to 2023 added
# 19 Nov 2026 so the FES 2021 to 2023 workbooks, priced in their own year's money, can be rebased.
CPIH_INDEX = {2019: 107.8, 2020: 108.9, 2021: 111.6, 2022: 120.5, 2023: 128.6, 2024: 132.9, 2025: 138.0}

# Block titles -> commodity key. Matched case-insensitively on the start of the title.
COMMODITY_TITLES = {
    "wholesale nbp gas price": "gas",
    "wholesale ara coal price": "coal",
    "wholesale brent oil price": "oil",
    "gb total carbon price": "carbon_gb_total",
    "eu ets carbon price": "carbon_eu_ets",
}
FUEL_COMMODITIES = ("gas", "coal", "oil")
CARBON_COMMODITIES = ("carbon_gb_total", "carbon_eu_ets")
_TITLE_UNIT = re.compile(r"\((\d{4})\s+([A-Z]{3})\s*/\s*([A-Za-z]+)\)")


class FESPriceError(ValueError):
    """The workbook's commodity-price tables could not be read as declared."""


def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_year(value: Any) -> Optional[int]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number == int(number) and 1990 <= number <= 2100 else None


def _sheet_rows(workbook: str | Path, sheet: str) -> list[tuple[int, tuple]]:
    import openpyxl

    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        if sheet not in wb.sheetnames:
            raise FESPriceError(f"Sheet {sheet!r} not found in {workbook}")
        return [(i, tuple(row)) for i, row in enumerate(wb[sheet].iter_rows(values_only=True), start=1)]
    finally:
        wb.close()


def parse_price_blocks(rows: list[tuple[int, tuple]], sheet: str) -> pd.DataFrame:
    """Parse every commodity block of a CP sheet into a long table.

    Args:
        rows: (Excel row, values) pairs of the sheet.
        sheet: Sheet name, for provenance.

    Returns:
        Columns commodity, case, year, value, currency, price_year, unit, sheet, excel_row, title.
    """
    records: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    year_cols: Optional[dict[int, int]] = None
    label_col: Optional[int] = None
    for excel_row, values in rows:
        cells = [(idx, _norm(v)) for idx, v in enumerate(values) if _norm(v)]
        if not cells:
            # The case rows follow the Year row with no gap; the first blank row ends the block.
            if year_cols is not None:
                current, year_cols, label_col = None, None, None
            continue
        first = cells[0][1]
        key = next((k for t, k in COMMODITY_TITLES.items() if first.lower().startswith(t)), None)
        if key is not None:
            match = _TITLE_UNIT.search(first)
            if not match:
                raise FESPriceError(f"{sheet} row {excel_row}: title {first!r} has no '(YYYY CUR/unit)' price basis")
            current = {"commodity": key, "title": first, "price_year": int(match.group(1)),
                       "currency": match.group(2), "unit": match.group(3)}
            year_cols, label_col = None, None
            continue
        if current is None:
            continue
        if first.lower() == "year" and year_cols is None:
            label_col = cells[0][0]
            year_cols = {idx: _as_year(v) for idx, v in cells[1:] if _as_year(v) is not None}
            continue
        if year_cols is not None and cells[0][0] == label_col:
            case = first
            if case.lower().startswith(("all prices", "years represent", "counterfactual", "note")):
                continue
            for idx, year in year_cols.items():
                raw = values[idx] if idx < len(values) else None
                value = pd.to_numeric(raw, errors="coerce")
                if pd.notna(value):
                    records.append({**{k: current[k] for k in ("commodity", "currency", "price_year", "unit",
                                                               "title")},
                                    "case": case, "year": year, "value": float(value), "sheet": sheet,
                                    "excel_row": excel_row})
        elif year_cols is not None and cells[0][0] != label_col:
            # Text outside the table (notes) ends the block.
            current, year_cols, label_col = None, None, None
    return pd.DataFrame.from_records(records)


def parse_conversion_table(rows: list[tuple[int, tuple]]) -> pd.DataFrame:
    """CP1 conversion table: year, GBP to EUR, rebased GDP deflator (workbook price year = 1).

    Returns:
        Columns year, gbp_to_eur, gdp_deflator_rebased, excel_row.
    """
    start = None
    for excel_row, values in rows:
        cells = [_norm(v) for v in values]
        if "From:" in cells and "GBP" in cells:
            start = excel_row
            gbp_idx = cells.index("GBP")
            break
    if start is None:
        raise FESPriceError("CP1 conversion table ('From:' / 'GBP' header) not found")
    records = []
    for excel_row, values in rows:
        if excel_row <= start + 1:
            continue
        cells = list(values)
        year = _as_year(cells[gbp_idx - 1]) if gbp_idx >= 1 else None
        if year is None:
            if records:
                break
            continue
        numbers = [pd.to_numeric(v, errors="coerce") for v in cells[gbp_idx:]]
        numbers = [float(v) for v in numbers if pd.notna(v)]
        # Row layout: GBP->EUR, GBP-pence->EUR, USD->EUR, year, GDP deflator (index), rebased deflator.
        if len(numbers) < 6:
            raise FESPriceError(f"CP1 conversion row {excel_row} has {len(numbers)} numbers, expected 6")
        records.append({"year": year, "gbp_to_eur": numbers[0], "gdp_deflator_rebased": numbers[5],
                        "excel_row": excel_row})
    if not records:
        raise FESPriceError("CP1 conversion table has no rows")
    return pd.DataFrame.from_records(records)


#: ``missing_case`` values of ``select_case``: refuse, or take the mean of a High and a Low case.
MISSING_CASE_RULES = ("refuse", "midpoint")


def _midpoint_case(block: pd.DataFrame, high: str, low: str) -> pd.DataFrame:
    """The year-by-year mean of a commodity's High and Low case rows, labelled as such (row of the High case)."""
    hi = block[block["case"].eq(high)].set_index("year")
    lo = block[block["case"].eq(low)].set_index("year")
    years = hi.index.intersection(lo.index)
    out = hi.loc[years].copy()
    out["value"] = (hi.loc[years, "value"] + lo.loc[years, "value"]) / 2.0
    out["case"] = (f"midpoint of {high} (row {int(hi['excel_row'].iloc[0])}) "
                   f"and {low} (row {int(lo['excel_row'].iloc[0])})")
    return out.reset_index()


def select_case(prices: pd.DataFrame, case: str = "Base Case", missing_case: str = "refuse") -> pd.DataFrame:
    """One case per commodity: the named case (case-insensitive), or the only non-History row.

    A commodity with several cases and none the one named fails, unless ``missing_case`` is ``midpoint`` and its
    cases are exactly a High and a Low case: then the year-by-year mean of the two (FES 2024 publishes no Base Case
    gas price; user decision, 18 Nov 2026).
    """
    if missing_case not in MISSING_CASE_RULES:
        raise FESPriceError(f"missing_case must be one of {MISSING_CASE_RULES}, not {missing_case!r}")
    out = []
    for commodity, block in prices.groupby("commodity"):
        cases = sorted(block["case"].unique())
        chosen = [c for c in cases if c.lower() == case.lower()]
        if not chosen:
            non_history = [c for c in cases if c.lower() != "history"]
            if len(non_history) == 1:
                chosen = non_history
            else:
                by_name = {c.lower(): c for c in non_history}
                if missing_case == "midpoint" and set(by_name) == {"high case", "low case"}:
                    out.append(_midpoint_case(block, by_name["high case"], by_name["low case"]))
                    continue
                raise FESPriceError(f"{commodity}: no {case!r} row; the workbook has {cases}. "
                                    "Name the case explicitly rather than falling back.")
        out.append(block[block["case"].eq(chosen[0])])
    return pd.concat(out, ignore_index=True)


def read_fes_commodity_prices(workbook: str | Path, case: str = "Base Case",
                              required: Iterable[str] = FUEL_COMMODITIES + CARBON_COMMODITIES,
                              missing_case: str = "refuse") -> dict[str, Any]:
    """Read CP1 and CP2 for one case, with the conversion table.

    Args:
        workbook: FES Data Workbook.
        case: Case row to take (FES 2025 publishes only "Base Case").
        required: Commodities that must be present.
        missing_case: What to do where a commodity lacks the case (``select_case``).

    Returns:
        ``prices`` (long table in the published currency), ``conversion`` (CP1 table).
    """
    cp1 = _sheet_rows(workbook, "CP1")
    cp2 = _sheet_rows(workbook, "CP2")
    prices = pd.concat([parse_price_blocks(cp1, "CP1"), parse_price_blocks(cp2, "CP2")], ignore_index=True)
    if prices.empty:
        raise FESPriceError(f"No commodity price blocks parsed from {workbook}")
    missing = set(required) - set(prices["commodity"])
    if missing:
        raise FESPriceError(f"Commodity price blocks missing from {workbook}: {sorted(missing)}")
    prices = select_case(prices, case, missing_case)
    bases = prices.groupby("commodity")[["currency", "price_year"]].first()
    if bases["price_year"].nunique() != 1 or set(bases["currency"]) != {"EUR"}:
        raise FESPriceError(f"Expected one EUR price year across blocks; got {bases.to_dict('index')}")
    return {"prices": prices, "conversion": parse_conversion_table(cp1)}


def to_model_money(prices: pd.DataFrame, conversion: pd.DataFrame,
                   model_price_year: int = MODEL_PRICE_YEAR,
                   cpih_index: Optional[dict[int, float]] = None) -> pd.DataFrame:
    """Convert published EUR (real P) prices to GBP real ``model_price_year`` (see module docstring).

    Args:
        prices: Output ``prices`` of :func:`read_fes_commodity_prices`.
        conversion: Output ``conversion`` of the same.
        model_price_year: Target price year.
        cpih_index: CPIH annual index by year; must hold P and ``model_price_year``.

    Returns:
        ``prices`` with value_gbp_price_year, value_model_gbp and the GDP-deflator alternative.
    """
    cpih = dict(CPIH_INDEX if cpih_index is None else cpih_index)
    price_year = int(prices["price_year"].iloc[0])
    fx = conversion.set_index("year")
    if price_year not in fx.index:
        raise FESPriceError(f"No GBP-to-EUR rate for the price year {price_year} in the CP1 conversion table")
    if price_year not in cpih or model_price_year not in cpih:
        raise FESPriceError(f"CPIH index needs {price_year} and {model_price_year}; declared: {sorted(cpih)}")
    gbp_to_eur = float(fx.loc[price_year, "gbp_to_eur"])
    cpih_factor = cpih[price_year] / cpih[model_price_year]
    gdp_factor = (float(fx.loc[price_year, "gdp_deflator_rebased"]) / float(fx.loc[model_price_year, "gdp_deflator_rebased"])
                  if model_price_year in fx.index else np.nan)
    out = prices.copy()
    out["gbp_to_eur"] = gbp_to_eur
    out["value_gbp_price_year"] = out["value"] / gbp_to_eur
    out["rebase_cpih_factor"] = cpih_factor
    out["value_model_gbp"] = out["value_gbp_price_year"] / cpih_factor
    out["value_model_gbp_gdp_deflator"] = out["value_gbp_price_year"] / gdp_factor
    out["model_price_year"] = model_price_year
    unit = out["unit"].str.lower()
    out["model_unit"] = np.where(unit.eq("mwh"), f"GBP{model_price_year}/MWh_th",
                                 np.where(unit.isin(["tonne", "t"]), f"GBP{model_price_year}/tCO2", "?"))
    if (out["model_unit"] == "?").any():
        raise FESPriceError(f"Unknown price units {sorted(out.loc[out['model_unit'] == '?', 'unit'].unique())}")
    return out


def standard_price_tables(model_prices: pd.DataFrame, years: Optional[Iterable[int]] = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two tables ``apply_marginal_costs.py`` and ``process_european_generation.py`` read.

    ``fuel_prices``: year, fuel, price_gbp_per_mwh_thermal (a thermal fuel price, not divided by
    any efficiency). ``carbon_prices``: year, carbon_price_gbp_per_tco2 (GB total carbon price)
    plus eu_ets_gbp_per_tco2. Provenance columns follow.
    """
    df = model_prices if years is None else model_prices[model_prices["year"].isin(list(years))]
    fuels = df[df["commodity"].isin(FUEL_COMMODITIES)]
    fuel_table = pd.DataFrame({
        "year": fuels["year"].astype(int), "fuel": fuels["commodity"],
        "price_gbp_per_mwh_thermal": fuels["value_model_gbp"],
        "price_eur_per_mwh_published": fuels["value"], "price_basis": fuels["model_unit"],
        "source": "FES " + fuels["sheet"] + " row " + fuels["excel_row"].astype(str) + " (" + fuels["case"] + ")",
    }).sort_values(["fuel", "year"]).reset_index(drop=True)
    gb = df[df["commodity"].eq("carbon_gb_total")].set_index("year")
    eu = df[df["commodity"].eq("carbon_eu_ets")].set_index("year")
    carbon_table = pd.DataFrame({
        "year": gb.index.astype(int), "carbon_price_gbp_per_tco2": gb["value_model_gbp"].to_numpy(),
        "eu_ets_gbp_per_tco2": eu["value_model_gbp"].reindex(gb.index).to_numpy(),
        "carbon_eur_per_tco2_published": gb["value"].to_numpy(), "price_basis": gb["model_unit"].to_numpy(),
        "source": ("FES CP2 row " + gb["excel_row"].astype(str) + " (" + gb["case"] + ")").to_numpy(),
    }).sort_values("year").reset_index(drop=True)
    if fuel_table.empty or carbon_table.empty:
        raise FESPriceError("No fuel or carbon prices in the selected years")
    return fuel_table, carbon_table


def future_cost_source(model_prices: pd.DataFrame, year: int, workbook: str | Path) -> dict[str, Any]:
    """A ``marginal_costs.future_costs`` source file for one year (schema of the NESO CP30 file).

    ``apply_marginal_costs.get_future_cost_assumptions`` reads ``assumptions.fuel_prices`` and
    ``assumptions.carbon.{uk_ets, carbon_price_support}`` from it; the case's ``fallback_case``
    supplies what FES does not publish (biomass, waste, hydrogen, variable O&M). The GB total
    carbon price is booked as ``uk_ets`` with Carbon Price Support zero: FES 2025 GB and EU
    prices are identical from 2030, and CPS for power ends April 2028 (HLWS1528).
    """
    rows = model_prices[model_prices["year"].eq(int(year))].set_index("commodity")
    missing = set(FUEL_COMMODITIES + ("carbon_gb_total",)) - set(rows.index)
    if missing:
        raise FESPriceError(f"No {sorted(missing)} price for {year}")
    first = rows.iloc[0]
    return {
        "schema_version": 1,
        "case": f"fes{int(first['price_year'])}_{int(year)}",
        "source": {
            "title": "NESO Future Energy Scenarios Data Workbook, sheets CP1 and CP2",
            "workbook": str(workbook),
            "case": str(first["case"]),
            "published_basis": f"real {int(first['price_year'])} EUR",
            "conversion": (f"EUR / {float(first['gbp_to_eur']):.4g} (CP1 GBP-to-EUR, {int(first['price_year'])}) "
                           f"/ {float(first['rebase_cpih_factor']):.6g} (CPIH {int(first['price_year'])} / "
                           f"{int(first['model_price_year'])})"),
            "rows": {c: f"{rows.loc[c, 'sheet']} row {int(rows.loc[c, 'excel_row'])}" for c in rows.index},
        },
        "price_year": int(first["model_price_year"]),
        "currency": "GBP",
        "year": int(year),
        "assumptions": {
            "fuel_prices": {c: float(rows.loc[c, "value_model_gbp"]) for c in FUEL_COMMODITIES},
            "carbon": {"uk_ets": float(rows.loc["carbon_gb_total", "value_model_gbp"]),
                       "carbon_price_support": 0.0},
        },
        "extracted_values": {c: {"published": float(rows.loc[c, "value"]), "unit": f"{rows.loc[c, 'currency']}/"
                                 f"{rows.loc[c, 'unit']}", "gbp_price_year": float(rows.loc[c, "value_gbp_price_year"]),
                                 "model_gbp_gdp_deflator_alternative": float(rows.loc[c, "value_model_gbp_gdp_deflator"])}
                             for c in rows.index},
        "not_published_by_fes": ["biomass", "waste", "hydrogen", "CO2 transport and storage"],
    }
