"""Snakemake driver for the FES 2025 targets and commodity prices (task Y1, ``rules/fes_long_range.smk``), any year.

Rules:

* ``build_fes2025_energy_targets``: ES1/ED1/BB1/WS1/FLX1 targets for one pathway and year
  (``scripts/FES/fes2025_targets.py``), as JSON, a tidy CSV and the classified ES1 lines.
* ``extract_fes2025_commodity_prices``: CP1/CP2 base case in GBP real 2024, in the standard
  ``fuel_prices`` / ``carbon_prices`` layout (``scripts/FES/fes_commodity_prices.py``).
* ``build_fes2025_future_costs_source``: one year's prices as a ``marginal_costs.future_costs``
  source file, for the FES long-range scenarios.

Run standalone for a trial: ``python scripts/FES/build_fes_inputs.py --year 2040 --out-dir <dir>``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.FES.fes2025_targets import (  # noqa: E402
    FES2025_WORKBOOK,
    fes2025_energy_targets,
    targets_summary_table,
    targets_to_json,
)
from scripts.FES.fes_commodity_prices import (  # noqa: E402
    future_cost_source,
    read_fes_commodity_prices,
    standard_price_tables,
    to_model_money,
)

try:
    from scripts.utilities.logging_config import setup_logging
except ImportError:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    def setup_logging(log_path):
        return logging.getLogger(__name__)


def write_targets(workbook: str, pathway: str, year: int, json_path: str, csv_path: str, lines_path: str,
                  year_basis: str = "calendar", demand_basis: str = "consumer", logger=None) -> dict:
    """Compute the targets (strict: an unmapped ES1 line fails) and write the three outputs."""
    targets = fes2025_energy_targets(workbook, pathway, year, year_basis=year_basis, demand_basis=demand_basis,
                                     include_building_blocks=True, strict=True)
    for path in (json_path, csv_path, lines_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(targets_to_json(targets), handle, indent=1)
    targets_summary_table(targets).to_csv(csv_path, index=False)
    targets["es1_lines"].to_csv(lines_path, index=False)
    if logger:
        gen = targets["generation"]
        logger.info("FES 2025 %s %s: %d carriers, %.1f GW, %.1f TWh generation, %.1f TWh curtailment",
                    pathway, year, len(gen), gen["capacity_mw"].sum() / 1e3, gen["generation_gwh"].sum() / 1e3,
                    gen["curtailment_gwh"].sum() / 1e3)
        logger.info("Demand %.1f TWh (%s, %s); electrolysis %.1f TWh; DACCS %.1f TWh",
                    targets["demand_gwh"] / 1e3, demand_basis, year_basis, targets["electrolysis_gwh"] / 1e3,
                    targets["daccs_gwh"] / 1e3)
        h2 = targets["hydrogen"]
        logger.info("Hydrogen: supply %.1f, demand %.1f (non-power %.1f), storage %.2f TWh; "
                    "electrolysis %.2f GW; reforming %.2f GW", h2["total_supply_twh"], h2["total_demand_twh"],
                    h2["non_power_demand_twh"], h2["storage_twh"], h2["electrolysis_capacity_gw_electric"],
                    h2["gas_reforming_ccs"]["capacity_gw_h2"])
    return targets


def write_prices(workbook: str, fuel_path: str, carbon_path: str, case: str = "Base Case", logger=None) -> None:
    """Write the standard fuel and carbon price tables in GBP real 2024."""
    read = read_fes_commodity_prices(workbook, case=case)
    fuel, carbon = standard_price_tables(to_model_money(read["prices"], read["conversion"]))
    for path in (fuel_path, carbon_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fuel.to_csv(fuel_path, index=False)
    carbon.to_csv(carbon_path, index=False)
    if logger:
        logger.info("Wrote %d fuel and %d carbon price rows", len(fuel), len(carbon))


def write_future_cost_source(workbook: str, year: int, json_path: str, case: str = "Base Case", logger=None) -> dict:
    """Write one year's ``future_costs`` source file."""
    read = read_fes_commodity_prices(workbook, case=case)
    payload = future_cost_source(to_model_money(read["prices"], read["conversion"]), year, workbook)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if logger:
        logger.info("future_costs source %s: %s", year, payload["assumptions"])
    return payload


def _run_snakemake(smk) -> None:
    logger = setup_logging(smk.log[0])
    rule = smk.rule
    if rule == "build_fes2025_energy_targets":
        write_targets(smk.input.workbook, smk.params.pathway, int(smk.wildcards.year), smk.output.json,
                      smk.output.csv, smk.output.es1_lines, smk.params.year_basis, smk.params.demand_basis, logger)
    elif rule == "extract_fes2025_commodity_prices":
        write_prices(smk.input.workbook, smk.output.fuel_prices, smk.output.carbon_prices, smk.params.price_case,
                     logger)
    elif rule == "build_fes2025_future_costs_source":
        write_future_cost_source(smk.input.workbook, int(smk.wildcards.year), smk.output.json,
                                 smk.params.price_case, logger)
    else:
        raise ValueError(f"build_fes_inputs.py does not handle rule {rule!r}")


if "snakemake" in globals():
    _run_snakemake(snakemake)  # noqa: F821
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FES 2025 targets and commodity prices (task Y1)")
    parser.add_argument("--workbook", default=FES2025_WORKBOOK)
    parser.add_argument("--pathway", default="Holistic Transition")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("build_fes_inputs")
    out = Path(args.out_dir)
    slug = args.pathway.lower().replace(" ", "_")
    write_targets(args.workbook, args.pathway, args.year, str(out / f"fes2025_{slug}_{args.year}_targets.json"),
                  str(out / f"fes2025_{slug}_{args.year}_targets.csv"),
                  str(out / f"fes2025_{slug}_{args.year}_es1_lines.csv"), logger=log)
    write_prices(args.workbook, str(out / "fes2025_fuel_prices_gbp2024.csv"),
                 str(out / "fes2025_carbon_prices_gbp2024.csv"), logger=log)
    write_future_cost_source(args.workbook, args.year, str(out / f"fes2025_future_costs_{args.year}.json"), logger=log)
