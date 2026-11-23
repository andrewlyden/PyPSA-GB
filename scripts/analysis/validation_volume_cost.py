"""Balancing volume and cost against outturn, month by month, for a historical validation family.

Cost alone can match for the wrong reason, so each month is compared on volume and unit cost as well:

* **Constraint volume:** the model's one-sided redispatch (turn-up = turn-down) against NESO's thermal constraint
  volume (its daily absolute volumes; almost all of it is turn-down), from ``{scenario}_neso_validation.csv``.
* **Constraint cost and cost per MWh:** the model's balancing cost against NESO's total thermal constraint cost
  (constraint breakdown, ``neso_breakdown_thermal_cost_gbp``: every thermal constraint, local ones included, the
  same scope as the volume and as the full network the model enforces). NESO's cost by constraint group covers only
  the six named boundaries (``neso_thermal_cost_gbp``, GBP 450m in 2020 against 596m in all) and is kept as
  ``neso_six_boundary_cost_gbp_m``.
* **Turn-down by technology:** the model's decrease volumes (``{scenario}_redispatch_summary.csv``) against ELEXON's
  accepted bid volumes (``{scenario}_boav_by_carrier.csv``), which include energy balancing as well as constraints.

    python scripts/analysis/validation_volume_cost.py --pattern "Validation_{Mon}2020" \
        --out resources/analysis/validation_2020_volume_cost.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WIND = ("wind_onshore", "wind_offshore", "embedded_wind")
GROUPS = {"wind": WIND, "gas": ("CCGT", "OCGT"), "hydro": ("large_hydro", "small_hydro"),
          "pumped_storage": ("Pumped Storage Hydroelectricity", "pumped_hydro"), "biomass": ("biomass",),
          "coal": ("coal",), "nuclear": ("nuclear",)}


def month_row(scenario: str, market: Path) -> dict | None:
    neso_path = market / f"{scenario}_neso_validation.csv"
    redispatch_path = market / f"{scenario}_redispatch_summary.csv"
    if not neso_path.exists() or not redispatch_path.exists():
        return None
    neso = pd.read_csv(neso_path).set_index(["category", "metric"])["value"]
    redispatch = pd.read_csv(redispatch_path)
    boav_path = market / f"{scenario}_boav_by_carrier.csv"
    boav = pd.read_csv(boav_path).set_index("carrier") if boav_path.exists() else pd.DataFrame()

    cost = float(neso[("total_cost", "model_bm_cost_gbp")])
    neso_cost = float(neso[("total_cost", "neso_breakdown_thermal_cost_gbp")])
    six_boundary = float(neso[("total_cost", "neso_thermal_cost_gbp")])
    volume = float(neso[("thermal_volume", "model_one_sided_redispatch_mwh")])
    neso_volume = float(neso[("thermal_volume", "neso_thermal_volume_abs_daily_mwh")])
    row = {
        "scenario": scenario,
        "model_cost_gbp_m": cost / 1e6, "neso_cost_gbp_m": neso_cost / 1e6, "cost_ratio": cost / neso_cost,
        "neso_six_boundary_cost_gbp_m": six_boundary / 1e6,
        "model_volume_gwh": volume / 1e3, "neso_volume_gwh": neso_volume / 1e3, "volume_ratio": volume / neso_volume,
        "model_gbp_per_mwh": cost / volume if volume else float("nan"),
        "neso_gbp_per_mwh": neso_cost / neso_volume if neso_volume else float("nan"),
    }
    decrease = redispatch.groupby("carrier")["decrease_MWh"].sum()
    for group, carriers in GROUPS.items():
        row[f"model_{group}_down_gwh"] = float(decrease.reindex(list(carriers)).fillna(0).sum()) / 1e3
        if not boav.empty:
            row[f"elexon_{group}_down_gwh"] = float(
                boav["decrease_mwh"].reindex(list(carriers)).fillna(0).sum()) / 1e3
    wind = redispatch[redispatch["carrier"].isin(WIND)]
    row["model_wind_bid_gbp_per_mwh"] = (float(wind["bid_cost"].sum()) / float(wind["decrease_MWh"].sum())
                                         if wind["decrease_MWh"].sum() > 0 else float("nan"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pattern", default="Validation_{Mon}2020", help="scenario id with {Mon}")
    parser.add_argument("--market", type=Path, default=Path("resources/market"))
    parser.add_argument("--out", type=Path, default=Path("resources/analysis/validation_2020_volume_cost.csv"))
    args = parser.parse_args()
    rows = [r for r in (month_row(args.pattern.format(Mon=m), args.market) for m in MONTHS) if r]
    table = pd.DataFrame(rows)
    totals = table.select_dtypes("number").sum()
    year = {"scenario": "year", **{c: totals[c] for c in table.columns if c.endswith("_gwh") or c.endswith("_gbp_m")}}
    year["cost_ratio"] = year["model_cost_gbp_m"] / year["neso_cost_gbp_m"]
    year["volume_ratio"] = year["model_volume_gwh"] / year["neso_volume_gwh"]
    year["model_gbp_per_mwh"] = year["model_cost_gbp_m"] * 1e3 / year["model_volume_gwh"]
    year["neso_gbp_per_mwh"] = year["neso_cost_gbp_m"] * 1e3 / year["neso_volume_gwh"]
    table = pd.concat([table, pd.DataFrame([year])], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
