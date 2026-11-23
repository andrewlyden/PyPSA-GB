"""One review page for the selected bill, component provenance and model evidence.

All candidate amounts retain their 3,100 kWh benchmark scope. They are never
silently added to a selected household tariff or counted as validation.
"""

import argparse
import html
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.market.consumer_bill_breakdown import build_breakdown
from scripts.market.build_consumer_bill_forecast_registry import _classification


def build_review(bills, register, bundle, bridges=None, signals=None, backtest=None):
    components, breakdown = build_breakdown(bills, register)
    bill = bills.iloc[0]
    components["value_origin"] = np.where(
        components["is_modelled"], "PyPSA wholesale sensitivity",
        "Projected bundle assumption" if bundle["input_mode"] == "future_projection" else "Historical tariff allocation",
    )
    classifications = pd.DataFrame([
        _classification(row.component, row.source_stack_group)
        for row in components.itertuples(index=False)
    ])
    components = pd.concat([components.reset_index(drop=True), classifications], axis=1)
    checks = []

    def check(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    check("Component total", "pass", f"Components sum to GBP {bill['consumer_bill_gbp_inc_vat']:.6f} including VAT")
    vat = register.loc[register["stack_group"].eq("vat"), "annual_gbp_ex_vat"].sum()
    correct_vat = np.isclose(vat, float(bill["consumer_bill_gbp_ex_vat"]) * float(bill["vat_rate"]), atol=1e-6, rtol=0)
    check("VAT", "pass" if correct_vat else "fail", f"VAT amount GBP {vat:.6f}; rate {bill['vat_rate']}")
    check("Year and provenance", "pass", f"{bundle['modelled_year']}: {bundle['input_mode']}; {bundle['snapshot_id']}")
    cfd = components.loc[components["component"].str.contains("CfD", case=False, na=False)]
    check("CfD counted once", "pass" if len(cfd) == 1 else "review",
          f"{len(cfd)} CfD rows; presentation uses policy costs and preserves the source category")
    unresolved = components["component"].str.contains("Network costs|Adjustment costs", case=False, na=False)
    check("Granularity", "review" if unresolved.any() else "pass",
          "Some source allowances remain aggregates; a finer split needs reconciled evidence" if unresolved.any()
          else "Available reconciled source components retained; see per-row source references")
    check("Headline use of PyPSA", "review", "Only the wholesale sensitivity enters this tariff headline. Recovery candidates below are not automatically substituted.")
    if signals is not None:
        values = signals.set_index("metric")["value"]
        shed = values.get("load_shedding_mwh", np.nan)
        check("Model adequacy", "review" if pd.isna(shed) or shed > 1e-6 else "pass",
              f"Modelled load shedding: {shed} MWh; scarcity can materially affect bill estimates")
        dispatch = values.get("cfd_total_dispatch_mwh", 0)
        priced = values.get("cfd_priced_dispatch_mwh", 0)
        check("CfD proxy coverage", "review" if dispatch > priced + 1e-6 else "pass",
              f"Priced dispatch {priced:,.2f} / labelled dispatch {dispatch:,.2f} MWh; see independent contract reconciliation where available")
    else:
        check("Operational evidence", "unavailable", "No bill-signal artifact supplied to this report")
    if bridges is not None:
        pending = bridges["candidate_gbp_per_3100kwh_ex_vat"].isna().sum()
        check("Recovery evidence", "review", f"{len(bridges) - pending}/{len(bridges)} candidate values available. Availability does not establish forecast validity; inspect evidence_status.")
    else:
        check("Recovery evidence", "unavailable", "No recovery bridge supplied; non-wholesale headline allowances come from the bundle")
    if backtest is not None:
        check("Historical comparison", "review",
              f"Benchmark GBP {backtest['historical_bill_gbp']:.2f}; reconstructed GBP {backtest['reconstructed_bill_gbp']:.2f}; difference GBP {backtest['difference_gbp']:.2f}. Same-year reconstruction is not out-of-sample validation.")
    summary = {
        "scenario": str(bill.get("scenario", bundle.get("scenario", "unspecified"))),
        "modelled_year": bundle["modelled_year"], "input_mode": bundle["input_mode"],
        "bill_basis": bundle.get("bill_basis", breakdown.get("bill_basis")),
        "annual_consumption_kwh": float(bill["annual_consumption_kwh"]),
        "bill_gbp_inc_vat": float(bill["consumer_bill_gbp_inc_vat"]),
        "currency": bundle.get("currency", "GBP; see source bundle"),
        "checks": checks, "groups": breakdown["published_cost_stack"]["groups"],
        "forecast_validated": False,
        "interpretation": "Arithmetic reconciliation does not validate a future forecast. Use historical backtests, explicit future contract/recovery inputs and sensitivity ranges.",
    }
    return components, summary


def render_review(components, summary, bridges=None, signals=None, links=None):
    def table(frame):
        return frame.to_html(index=False, escape=True, na_rep="Unavailable", float_format=lambda x: f"{x:,.4f}")

    group_rows = [{"Category": row["group"], "Annual GBP": row["annual_gbp_per_3100kwh"]} for row in summary["groups"]]
    sections = []
    for group, frame in components.groupby("group", sort=False):
        fields = ["component", "annual_gbp_at_selected_consumption", "value_origin", "charge_basis", "source_stack_group", "source_workbook", "source_cells", "forecast_method", "required_input"]
        sections.append(f"<details open><summary>{html.escape(group)}</summary>{table(frame[fields])}</details>")
    links_html = " ".join(f'<a href="{html.escape(path.name, quote=True)}">{html.escape(label)}</a>' for label, path in (links or {}).items())
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Consumer bill review — {html.escape(summary['scenario'])}</title>
<style>body{{font:15px system-ui; margin:2rem; color:#182838}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{padding:.6rem;border:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}th{{background:#edf3f8}}details{{overflow:auto;margin:1rem 0}}summary{{font-size:1.2rem;font-weight:bold;cursor:pointer}}a{{margin-right:1rem}}.note{{background:#fff4d9;padding:1rem}}h1{{color:#174b64}}</style>
<h1>{html.escape(summary['scenario'])}: consumer electricity bill</h1>
<p>{summary['modelled_year']} | {html.escape(summary['input_mode'])} | {summary['annual_consumption_kwh']:,.0f} kWh/year | {html.escape(summary['currency'])}</p>
<h2>GBP {summary['bill_gbp_inc_vat']:,.2f} per year including VAT</h2>
<p class="note">{html.escape(summary['interpretation'])} Household component amounts below use the selected consumption. Recovery candidates retain their separate 3,100 kWh benchmark scope.</p>
<p>{links_html}</p><h2>Checks and evidence gaps</h2>{table(pd.DataFrame(summary['checks']))}
<h2>Main categories</h2>{table(pd.DataFrame(group_rows))}
<h2>Every bill component</h2>{''.join(sections)}
<h2>Model-to-bill recovery candidates — not added to the headline</h2>
{table(bridges) if bridges is not None else '<p>Not supplied for this report.</p>'}
<details><summary>Underlying PyPSA signals — system quantities, not household charges</summary>{table(signals) if signals is not None else '<p>Not supplied.</p>'}</details></html>'''


def write_review(bills_path, register_path, bundle_path, output_csv, output_json, output_html,
                 bridges_path=None, signals_path=None, backtest_path=None):
    bridges = pd.read_csv(bridges_path) if bridges_path else None
    signals = pd.read_csv(signals_path) if signals_path else None
    backtest = json.loads(Path(backtest_path).read_text(encoding="utf-8")) if backtest_path else None
    components, summary = build_review(pd.read_csv(bills_path), pd.read_csv(register_path),
        json.loads(Path(bundle_path).read_text(encoding="utf-8")), bridges, signals, backtest)
    summary["input_artifacts"] = [
        {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
         "modified_time_utc": pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC").isoformat()}
        for path in (bills_path, register_path, bundle_path, bridges_path, signals_path, backtest_path) if path
    ]
    for path in (output_csv, output_json, output_html):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_csv, index=False)
    Path(output_json).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    links = {"Components CSV": Path(output_csv), "Checks JSON": Path(output_json)}
    Path(output_html).write_text(render_review(components, summary, bridges, signals, links), encoding="utf-8")


if "snakemake" in globals():
    def optional(name):
        paths = list(snakemake.input.get(name, []))
        return paths[0] if paths else None
    write_review(snakemake.input.bills, snakemake.input.register, snakemake.input.bundle,
                 snakemake.output.components, snakemake.output.summary, snakemake.output.report,
                 optional("bridges"), optional("signals"), optional("backtest"))
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("bills", "register", "bundle", "output-csv", "output-json", "output-html"):
        parser.add_argument("--" + arg, required=True)
    for arg in ("bridges", "signals", "backtest"):
        parser.add_argument("--" + arg)
    args = parser.parse_args()
    write_review(args.bills, args.register, args.bundle, args.output_csv, args.output_json,
                 args.output_html, args.bridges, args.signals, args.backtest)
