"""Extract the network price-control revenue building blocks (backlog item G2, plan item N6).

The network lines are GBP 231 of the 2030 household bill and the largest block the
public-ownership work cannot yet touch.  What an ownership account needs is not the
tariff, which the bill already sources, but the *building blocks underneath it*: the
Regulatory Asset Value, the return allowed on it, the tax allowance and the total
allowed revenue.  The saving from public ownership is then
``(allowed WACC - public borrowing rate) x RAV``, plus the corporation-tax allowance
that a public owner does not pay.

Both regulators' Final Determinations publish those blocks per licensee:

* **Distribution, RIIO-ED2 (1 April 2023 to 31 March 2028).**  Appendix 5 of the Final
  Determinations Finance Annex gives, for each of the fourteen DNOs and each year,
  opening and closing RAV, net additions, depreciation, and the calculated allowances
  split into fast pot, RAV depreciation, return, pass-through, tax and total revenue.
  In GBP m 2020/21 prices, CPIH real.
* **Transmission, RIIO-3 (1 April 2026 to 31 March 2031).**  The Finance Annex gives
  average annual ex ante base revenue (Table 31) and the *equity portion* of NPV-neutral
  RAV (Table 32) for NGET, SHET and SPT, in GBP m 2023/24 prices.  Total RAV is the
  equity portion divided by one minus notional gearing (55% for ET).

**Two bases that must not be mixed.**  The ED2 vanilla WACC of 3.93% is CPIH-real.  The
RIIO-3 ET WACC of 5.64% is *semi-nominal*: the fixed-rate debt allowance carries expected
inflation, so it is not comparable with a real rate and must be converted before it is set
against the state's real borrowing rate.  The converter here is deliberately explicit
rather than hidden in a spreadsheet.

**What this does not give, and why item N6 is still blocked for 2030.**  The modelled year
is 2030.  Transmission is covered: RIIO-3 runs to March 2031.  Distribution is not: ED2
ends in March 2028 and ED3 (April 2028 to March 2033) had only a Sector Specific
Methodology Decision as at May 2026, with draft determinations expected in 2027.  So the
2030 distribution RAV and return are a projection, not a published dataset, and this
module extracts ED2 as the base to project *from*, never as the 2030 answer.

Usage::

    python scripts/consumer_bill/extract_network_price_control_building_blocks.py
        --ed2-finance-annex <pdf> --riio3-finance-annex <pdf>
        --out-licensee <csv> --out-summary <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.model_year import DEFAULT_MODEL_YEAR  # noqa: E402

#: The determined price-control periods a calendar model year is tested against (item Y4):
#: RIIO-3 (transmission) and RIIO-ED3 (distribution; ED2 ended 31 March 2028).
RIIO3_PERIOD = ("2026-04-01", "2031-03-31")
ED3_PERIOD = ("2028-04-01", "2033-03-31")


def calendar_year_inside(model_year: int, period: tuple[str, str]) -> bool:
    """Whether the whole of calendar ``model_year`` falls inside a price-control period."""
    start, end = (pd.Timestamp(d) for d in period)
    return start <= pd.Timestamp(f"{int(model_year)}-01-01") and pd.Timestamp(f"{int(model_year)}-12-31") <= end

DNOS = ("ENWL", "NPgN", "NPgY", "WMID", "EMID", "SWALES", "SWEST",
        "LPN", "SPN", "EPN", "SPD", "SPMW", "SSEH", "SSES")
ED2_YEARS = ("2023/24", "2024/25", "2025/26", "2026/27", "2027/28")
ED2_COLUMNS = ED2_YEARS + ("period_total", "period_average")
ED2_ROWS = (
    "Opening RAV", "Net additions (after disposals)", "Depreciation", "Closing RAV",
    "Fast pot expenditure", "RAV depreciation", "Return", "Pass-through expenditure",
    "Tax allowance", "Calculated revenue",
)
ET_LICENSEES = ("NGET", "SHET", "SPT")

# Published parameters, quoted so the arithmetic below can be checked by hand.
ED2_PARAMETERS = {
    "period": "2023-04-01/2028-03-31", "price_basis": "GBP m 2020/21 prices, CPIH real",
    "notional_gearing": 0.60, "cost_of_equity_cpih_real": 0.0523,
    "cost_of_debt_cpih_real_infrequent_issuers": 0.0307,
    "cost_of_debt_cpih_real_frequent_issuers": 0.0301,
    "vanilla_wacc_cpih_real_infrequent_issuers": 0.0393,
    "vanilla_wacc_cpih_real_frequent_issuers": 0.0390,
    "frequent_issuers": ["EMID", "EPN", "SSES"],
    "source": "Ofgem, RIIO-ED2 Final Determinations Finance Annex and Overview, 30 November 2022, Table 5 and Appendix 5",
}
RIIO3_ET_PARAMETERS = {
    "period": "2026-04-01/2031-03-31", "price_basis": "GBP m 2023/24 prices",
    "notional_gearing": 0.55, "cost_of_equity_cpih_real": 0.0570,
    "wacc_semi_nominal_unweighted_average": 0.0564, "wacc_semi_nominal_range": [0.0553, 0.0574],
    "source": "Ofgem, RIIO-3 Final Determinations Finance Annex, 4 December 2025, Table 11, Tables 31 and 32",
}


def _numbers(text: str) -> list[float]:
    return [float(v.replace(",", "")) for v in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def extract_ed2_appendix5(pdf_path: str | Path) -> pd.DataFrame:
    """Per-DNO, per-year building blocks from every DNO table in the ED2 Finance Annex.

    The annex prints each DNO's table twice (Appendix 5 and its revision, Appendix 8), and this
    reads both, so its rows must never be summed into levels: use :func:`extract_ed2_appendix8`.
    Kept for the comparison of the two copies (tests/unit/test_duos_cp30_growth.py).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    records = []
    for page in reader.pages:
        text = page.extract_text() or ""
        header = re.search(r"^\s*(%s)\s*$" % "|".join(DNOS), text, re.M)
        if not header:
            continue
        dno = header.group(1)
        for row in ED2_ROWS:
            match = re.search(re.escape(row) + r"((?:\s+[-\d,\.]+){7})", text)
            if not match:
                continue
            values = _numbers(match.group(1))
            if len(values) < len(ED2_COLUMNS):
                continue
            records.append({"sector": "distribution", "licensee": dno, "item": row,
                            **dict(zip(ED2_COLUMNS, values[:len(ED2_COLUMNS)]))})
    frame = pd.DataFrame(records)
    missing = set(DNOS) - set(frame["licensee"].unique() if not frame.empty else ())
    if missing:
        raise ValueError(f"ED2 Appendix 5: no table found for {sorted(missing)}")
    return frame


def extract_ed2_appendix8(pdf_path: str | Path) -> pd.DataFrame:
    """Appendix 8 ("REVISED Financial values for electricity distribution networks") only, one table per DNO.

    The Finance Annex prints every DNO's table twice: Appendix 5 and its revision, Appendix 8.
    :func:`extract_ed2_appendix5` reads every page with a DNO header, so it returns both, and a sum
    over its rows counts each DNO twice (ratios survive, levels double). The CP30 DUoS path (user
    decision, 6 Oct 2026, ``duos_cp30_growth.py``) reads levels, so it takes the revised tables alone.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    records, inside = [], False
    for page in reader.pages:
        text = page.extract_text() or ""
        header = re.search(r"^\s*(%s)\s*$" % "|".join(DNOS), text, re.M)
        # The contents page names Appendix 8 too; the appendix starts on the page that also has a DNO table.
        if "Appendix 8 - REVISED Financial values" in text and header:
            inside = True
        elif inside and re.search(r"Appendix (9|1\d)\b", text):
            break
        if not inside:
            continue
        if not header:
            continue
        for row in ED2_ROWS:
            # A negative is printed in brackets (SSEH's pass-through expenditure).
            match = re.search(re.escape(row) + r"((?:\s+\(?[-\d,\.]+\)?){7})", text)
            values = [(-v if cell.startswith("(") else v) for cell, v in
                      ((cell, _numbers(cell)[0]) for cell in match.group(1).split())] if match else []
            if len(values) >= len(ED2_COLUMNS):
                records.append({"sector": "distribution", "licensee": header.group(1), "item": row,
                                **dict(zip(ED2_COLUMNS, values[:len(ED2_COLUMNS)]))})
    frame = pd.DataFrame(records)
    complete = (not frame.empty and set(frame["licensee"]) == set(DNOS)
                and frame.groupby("licensee")["item"].apply(lambda s: sorted(s) == sorted(ED2_ROWS)).all())
    if not complete:
        raise ValueError("ED2 Appendix 8: expected one revised table per DNO, every row read once")
    return frame


def extract_riio3_et(pdf_path: str | Path) -> pd.DataFrame:
    """Average annual base revenue and equity RAV per ET licensee (Tables 31 and 32)."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    def _section(start: str, end: str | None) -> str:
        """The text of one numbered table, so a licensee row cannot be read from another.

        Anchoring matters: Table 30 immediately above lists the same three licensees
        against their business plan incentive, which is a single-digit GBP m figure.
        An unanchored search finds that first and silently returns it as revenue.
        """
        begin = re.search(re.escape(start), text)
        if not begin:
            raise ValueError(f"RIIO-3 Finance Annex: {start!r} not found")
        tail = text[begin.end():]
        stop = re.search(re.escape(end), tail) if end else None
        return tail[: stop.start()] if stop else tail[:4000]

    revenue_table = _section("Table 31:", "Table 32:")
    equity_table = _section("Table 32:", "Annual Iteration Process")
    records = []
    for licensee in ET_LICENSEES:
        revenue = re.search(rf"{licensee}\s*\D{{0,3}}([\d,]+\.\d)m", revenue_table)
        equity = re.search(rf"{licensee}\s*\D{{0,3}}([\d,]+\.\d)m", equity_table)
        if revenue:
            records.append({"sector": "transmission", "licensee": licensee,
                            "item": "Ex ante average annual base revenue",
                            "period_average": _numbers(revenue.group(1))[0]})
        if equity:
            records.append({"sector": "transmission", "licensee": licensee,
                            "item": "Average equity portion of NPV neutral RAV",
                            "period_average": _numbers(equity.group(1))[0]})
    frame = pd.DataFrame(records)
    # ET base revenue is hundreds of millions a year; a single-digit value means the
    # parse has drifted into the incentive table above.
    small = frame.loc[frame["item"].str.startswith("Ex ante") & frame["period_average"].lt(100.0)]
    if not small.empty:
        raise ValueError(f"RIIO-3 base revenue implausibly small for {sorted(small['licensee'])}; check table anchoring")
    for item in ("Ex ante average annual base revenue", "Average equity portion of NPV neutral RAV"):
        found = set(frame.loc[frame["item"].eq(item), "licensee"]) if not frame.empty else set()
        if set(ET_LICENSEES) - found:
            raise ValueError(f"RIIO-3 Finance Annex: {item} missing for {sorted(set(ET_LICENSEES) - found)}")
    return frame


def semi_nominal_to_cpih_real(wacc_semi_nominal: float, *, gearing: float, inflation: float,
                              fixed_rate_debt_share: float) -> float:
    """Convert a semi-nominal WACC to CPIH-real.

    RIIO-3 sets the allowed return on debt on a semi-nominal basis: the fixed-rate
    portion carries expected inflation, the index-linked portion does not.  Only that
    portion of the debt allowance has inflation to remove, so the adjustment is
    ``inflation x gearing x fixed-rate share``.  Equity is already CPIH-real.
    """
    for name, value in (("gearing", gearing), ("fixed_rate_debt_share", fixed_rate_debt_share)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a share between 0 and 1")
    return float(wacc_semi_nominal) - float(inflation) * float(gearing) * float(fixed_rate_debt_share)


def coverage(model_year: int) -> dict:
    """Whether each price control determines the calendar model year (item Y4; the text is the CP30 year's)."""
    y = int(model_year)
    transmission = calendar_year_inside(y, RIIO3_PERIOD)
    in_ed3 = calendar_year_inside(y, ED3_PERIOD)
    return {
        "modelled_year": y,
        "transmission_covered": transmission,
        "transmission_note": (f"RIIO-3 runs to 31 March 2031, so {y} is inside the determined period" if transmission
                              else f"RIIO-3 runs to 31 March 2031, so {y} is outside every determined period"),
        "distribution_covered": False,
        "distribution_note": (
            (f"ED2 ends 31 March 2028. {y} falls in RIIO-ED3 (April 2028 to March 2033), which had only a "
             "Sector Specific Methodology Decision as at 21 May 2026; draft determinations are expected in "
             f"2027. The ED2 values here are the base to project from, never the {y} answer.") if in_ed3 else
            (f"ED2 ends 31 March 2028 and RIIO-ED3 runs to March 2033, so {y} is outside every determined or "
             f"methodology-decided period. The ED2 values here are the base to project from, never the {y} answer.")),
    }


def summarise(ed2: pd.DataFrame, et: pd.DataFrame, model_year: int | None = None) -> dict:
    """Sector totals and the ownership-relevant ratios, with the checks that validate the parse."""
    repeated = ed2.duplicated(["licensee", "item"])
    if repeated.any():
        raise ValueError("ED2 building blocks: a DNO's table appears more than once, so the summed levels would "
                         f"double count ({sorted(set(ed2.loc[repeated, 'licensee']))}); use extract_ed2_appendix8")
    dist = ed2.pivot_table(index="item", values=list(ED2_COLUMNS), aggfunc="sum")
    get = lambda item, col: float(dist.loc[item, col])  # noqa: E731
    opening_rav = get("Opening RAV", "period_average")
    return_allowance = get("Return", "period_average")
    implied_wacc = return_allowance / opening_rav
    equity_rav = float(et.loc[et["item"].str.startswith("Average equity"), "period_average"].sum())
    gearing = RIIO3_ET_PARAMETERS["notional_gearing"]
    total_et_rav = equity_rav / (1.0 - gearing)
    return {
        "distribution": {
            **ED2_PARAMETERS,
            "licensees": len(DNOS),
            "opening_rav_gbp_m_period_average": opening_rav,
            "closing_rav_gbp_m_2027_28": get("Closing RAV", "2027/28"),
            # The final-year additions and depreciation are what a roll-forward past the
            # end of ED2 continues from (item N6); 2030 is outside the determined period.
            "net_additions_gbp_m_2027_28": get("Net additions (after disposals)", "2027/28"),
            "depreciation_gbp_m_2027_28": get("Depreciation", "2027/28"),
            "return_gbp_m_period_average": return_allowance,
            "tax_allowance_gbp_m_period_average": get("Tax allowance", "period_average"),
            "rav_depreciation_gbp_m_period_average": get("RAV depreciation", "period_average"),
            "allowed_revenue_gbp_m_period_average": get("Calculated revenue", "period_average"),
            "return_share_of_allowed_revenue": return_allowance / get("Calculated revenue", "period_average"),
            # The parse is validated by reproducing the published WACC from the
            # extracted return and RAV; a bad parse cannot pass this.
            "implied_vanilla_wacc_from_extract": implied_wacc,
            "published_vanilla_wacc_cpih_real": ED2_PARAMETERS["vanilla_wacc_cpih_real_infrequent_issuers"],
            "implied_wacc_check_passes": abs(implied_wacc - 0.0393) < 0.0015,
        },
        "transmission": {
            **RIIO3_ET_PARAMETERS,
            "licensees": len(ET_LICENSEES),
            "equity_portion_of_rav_gbp_m_period_average": equity_rav,
            "implied_total_rav_gbp_m_period_average": total_et_rav,
            "base_revenue_gbp_m_period_average": float(
                et.loc[et["item"].str.startswith("Ex ante"), "period_average"].sum()),
        },
        "coverage": coverage(DEFAULT_MODEL_YEAR if model_year is None else model_year),
        "basis_warning": (
            "ED2 WACC is CPIH-real and RIIO-3 ET WACC is semi-nominal. Convert with "
            "semi_nominal_to_cpih_real() before setting either against a real public borrowing rate."),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = "data/market/network_cost_sources/price_controls"
    parser.add_argument("--ed2-finance-annex", default=f"{default}/ofgem_riio_ed2_final_determinations_finance_annex_2022_11.pdf")
    parser.add_argument("--riio3-finance-annex", default=f"{default}/ofgem_riio3_final_determinations_finance_annex_2025_12.pdf")
    parser.add_argument("--out-licensee", default="resources/analysis/network_price_control_building_blocks.csv")
    parser.add_argument("--out-summary", default="resources/analysis/network_price_control_building_blocks.json")
    parser.add_argument("--model-year", type=int, default=None,
                        help="the calendar year the coverage flags describe (item Y4; default the CP30 year)")
    args = parser.parse_args(argv)

    # Appendix 8 alone: summing Appendix 5 as well counted every DNO twice (found 6 Oct 2026).
    ed2 = extract_ed2_appendix8(args.ed2_finance_annex)
    et = extract_riio3_et(args.riio3_finance_annex)
    frame = pd.concat([ed2, et], ignore_index=True)
    report = summarise(ed2, et, args.model_year)
    report["sources"] = {"ed2_finance_annex": str(args.ed2_finance_annex),
                         "riio3_finance_annex": str(args.riio3_finance_annex)}
    for path in (args.out_licensee, args.out_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_licensee, index=False)
    Path(args.out_summary).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    d, t = report["distribution"], report["transmission"]
    print(json.dumps({
        "dno_tables": len(DNOS), "et_licensees": len(ET_LICENSEES),
        "distribution_rav_gbp_bn": round(d["opening_rav_gbp_m_period_average"] / 1e3, 1),
        "distribution_return_gbp_bn": round(d["return_gbp_m_period_average"] / 1e3, 2),
        "implied_wacc_check_passes": d["implied_wacc_check_passes"],
        "transmission_total_rav_gbp_bn": round(t["implied_total_rav_gbp_m_period_average"] / 1e3, 1),
        f"distribution_covered_in_{report['coverage']['modelled_year']}": report["coverage"]["distribution_covered"],
    }))


if __name__ == "__main__":
    main()
