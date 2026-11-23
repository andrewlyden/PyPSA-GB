"""Split published NESO BSUoS expenditure into constraint and non-constraint parts.

The consumer bill takes its constraint cost from PyPSA redispatch, so the
external BSUoS allowance must cover only what the model does not produce:
response, reserve, reactive, restoration and NESO internal costs.  This script
derives that residual from two NESO datasets covering the same regulatory year:

* the half-hourly balancing-cost category breakdown, which gives constraint
  spend explicitly, and
* the Settlement Final BSUoS dataset, which gives total actual BSUoS cost and
  the chargeable final-demand volume the charge is recovered over.

The non-constraint residual is deliberately taken against *total actual BSUoS
cost* rather than against the summed categories: the category file covers
balancing actions only, so the difference between the two totals is NESO
internal and contracted cost, which belongs in the external allowance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


NESO_API = "https://api.neso.energy/dataset"
BALANCING_COST_CATEGORIES = (
    "Energy Imbalance", "Frequency Control", "Positive Reserve",
    "Constraints", "Negative Reserve", "Other",
)
CONSTRAINT_CATEGORY = "Constraints"

SOURCES = {
    "balancing_cost_categories": {
        "dataset": "Daily Balancing Services Use of System (BSUoS) Cost Data",
        "dataset_id": "d13d78fc-60d9-4f4d-87b6-e25a20f669c0",
        "resource_id": "527a5f40-942b-416b-99df-81a51c30d041",
        "filename": "q1_cost_fy2425.csv",
    },
    "bsuos_settlement_final": {
        "dataset": "Current Balancing Services Use of System (BSUoS) Charges, Historic SF 2024-2025",
        "dataset_id": "d6a4bf54-c63f-4014-a716-49fd3878ca52",
        "resource_id": "320cdc0d-457f-4849-a0fc-bf6bdd4e21e5",
        "filename": "2024-2025-sf.csv",
    },
}


def resource_url(source: dict) -> str:
    return f"{NESO_API}/{source['dataset_id']}/resource/{source['resource_id']}/download/{source['filename']}"


def _column(frame: pd.DataFrame, fragment: str, label: str) -> str:
    """Match a column by fragment; the published headers carry a mojibake GBP sign."""
    matches = [column for column in frame.columns if fragment in column]
    if len(matches) != 1:
        raise ValueError(
            f"{label}: expected exactly one column containing {fragment!r}, found {matches}"
        )
    return matches[0]


def derive_balancing_cost_split(categories: pd.DataFrame, settlement: pd.DataFrame,
                                regulatory_year: str, price_year: int) -> dict:
    """Return the constraint / non-constraint BSUoS decomposition for one year."""
    categories = categories.copy()
    categories.columns = [column.strip() for column in categories.columns]
    settlement = settlement.copy()
    settlement.columns = [column.strip() for column in settlement.columns]

    missing = [name for name in BALANCING_COST_CATEGORIES if name not in categories.columns]
    if missing:
        raise ValueError(f"NESO balancing-cost categories are missing columns: {missing}")

    periods = len(categories)
    if not 17_000 <= periods <= 17_600:
        raise ValueError(
            f"Category file covers {periods} settlement periods, not one regulatory year"
        )

    by_category = {
        name: float(pd.to_numeric(categories[name], errors="raise").sum())
        for name in BALANCING_COST_CATEGORIES
    }
    constraint_cost = by_category[CONSTRAINT_CATEGORY]

    volume_column = _column(settlement, "Volume", "BSUoS settlement")
    cost_column = _column(settlement, "Actual BSUoS Cost", "BSUoS settlement")
    volume = float(pd.to_numeric(settlement[volume_column], errors="raise").sum())
    total_cost = float(pd.to_numeric(settlement[cost_column], errors="raise").sum())
    if volume <= 0:
        raise ValueError("BSUoS chargeable volume must be positive")
    if constraint_cost >= total_cost:
        raise ValueError("Constraint spend cannot exceed total actual BSUoS cost")

    category_total = float(sum(by_category.values()))
    other_cost = total_cost - constraint_cost
    return {
        "purpose": (
            "External BSUoS allowance for the parts of balancing the dispatch model does not "
            "produce. Constraint cost is excluded here because the consumer bill takes it from "
            "PyPSA redispatch; counting both would double-count constraints."
        ),
        "regulatory_year": regulatory_year,
        "price_year": price_year,
        "price_basis_note": (
            f"Nominal outturn for regulatory year {regulatory_year}, whose mid-point falls in "
            f"{price_year}. It is treated as {price_year} money and rebased by the projector's "
            "declared index; it is not deflated within the year."
        ),
        "settlement_periods_in_category_file": periods,
        "balancing_cost_by_category_gbp": by_category,
        "balancing_cost_category_total_gbp": category_total,
        "constraint_cost_gbp": constraint_cost,
        "total_actual_bsuos_cost_gbp": total_cost,
        "non_constraint_cost_gbp": other_cost,
        "internal_and_contracted_residual_gbp": total_cost - category_total,
        "chargeable_final_demand_mwh": volume,
        "non_constraint_gbp_per_mwh_final_demand": other_cost / volume,
        "constraint_gbp_per_mwh_final_demand": constraint_cost / volume,
        "total_gbp_per_mwh_final_demand": total_cost / volume,
        "method": (
            "Sum the published half-hourly balancing-cost categories over the regulatory year to "
            "obtain constraint spend. Subtract it from total actual BSUoS cost in the Settlement "
            "Final dataset for the same year, and divide by the summed chargeable volume. The "
            "residual between total actual cost and the summed categories is NESO internal and "
            "contracted cost and is retained inside the non-constraint figure."
        ),
        "caveats": [
            "The 'Constraints' category covers all constraint actions, not only thermal ones, so "
            "the modelled thermal redispatch cost that replaces it is a narrower quantity.",
            "BSUoS has been recovered from final demand only since April 2023, so earlier years "
            "are not comparable on this volume basis.",
            "This is an outturn decomposition for one regulatory year, not a 2030 forecast. Only "
            "its level is carried forward, held flat in real terms.",
        ],
        "sources": SOURCES,
    }


def write_split(categories_path, settlement_path, output_json, regulatory_year, price_year):
    split = derive_balancing_cost_split(
        pd.read_csv(categories_path), pd.read_csv(settlement_path), regulatory_year, price_year
    )
    for key, local in (("balancing_cost_categories", categories_path),
                       ("bsuos_settlement_final", settlement_path)):
        path = Path(str(local))
        if path.exists():
            split["sources"][key] = {
                **split["sources"][key],
                "download_url": resource_url(split["sources"][key]),
                "local_file": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", required=True, help="NESO daily balancing cost CSV or URL")
    parser.add_argument("--settlement", required=True, help="NESO SF BSUoS CSV or URL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--regulatory-year", default="2024-25")
    parser.add_argument("--price-year", type=int, default=2024)
    args = parser.parse_args()
    result = write_split(args.categories, args.settlement, args.output,
                         args.regulatory_year, args.price_year)
    print(f"non-constraint BSUoS: GBP {result['non_constraint_gbp_per_mwh_final_demand']:.4f}/MWh "
          f"final demand ({args.regulatory_year})")
