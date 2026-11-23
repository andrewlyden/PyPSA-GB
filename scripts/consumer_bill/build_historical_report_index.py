"""Create the maintained index for historical consumer-bill reports."""

import argparse
import csv
import html
import json
from pathlib import Path


def write_index(report_paths, output_path) -> None:
    output = Path(output_path)
    rows = []
    for value in report_paths:
        report = Path(value)
        checks_path = report.with_name("checks.json")
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        relative = report.relative_to(output.parent).as_posix()
        totals_path = report.with_name("ofgem_model_totals.csv")
        totals = []
        if totals_path.exists():
            with totals_path.open(encoding="utf-8", newline="") as stream:
                totals = list(csv.DictReader(stream))
        published = f"{float(totals[0]['inc_vat_gbp']):,.2f}" if len(totals) == 2 else "Not available"
        hybrid = f"{float(totals[1]['inc_vat_gbp']):,.2f}" if len(totals) == 2 else "Not available"
        rows.append(
            f"<tr><td><a href=\"{html.escape(relative)}\">{html.escape(report.parent.name)}</a></td>"
            f"<td>{checks.get('period_count')}</td><td>{checks.get('region_count')}</td>"
            f"<td>{published}</td><td>{hybrid}</td>"
            f"<td>{checks.get('validated')}</td><td>{checks.get('model_validation_status')}</td></tr>"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Historical consumer bills</title>"
        "<style>body{font:14px system-ui;max-width:1100px;margin:2rem auto}table{border-collapse:collapse}"
        "th,td{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}</style></head><body>"
        "<h1>Historical consumer-bill reports</h1><table><thead><tr><th>Case</th><th>Periods</th>"
        "<th>Regions</th><th>Ofgem GBP incl. VAT</th><th>Ofgem/modelled GBP incl. VAT</th><th>Tariff validated</th><th>Model evidence</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></body></html>",
        encoding="utf-8",
    )


if "snakemake" in globals():
    write_index(snakemake.input.reports, snakemake.output.index)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_index(args.reports, args.output)
