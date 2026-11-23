"""
Extract fuel and carbon prices from FES Data Workbooks (sheets CP1 and CP2).

Writes ``fuel_prices_{year}.csv`` (year, fuel, price_gbp_per_mwh_thermal) and
``carbon_prices_{year}.csv`` (year, carbon_price_gbp_per_tco2) in GBP real 2024, the
model's price base, with provenance columns. The reading and the EUR-to-GBP and price-base
conversion are in ``scripts/FES/fes_commodity_prices.py``, where they are declared.

**No fallback (task Y1, 30 Sep 2026).** Until then this script fell back silently to
built-in placeholders (gas = coal = oil = biomass rising 20 to 32 GBP/MWh, carbon 30 to
110) whenever it could not parse a workbook, which it never could for FES 2025. It now
fails instead. FES publishes no biomass price, so no biomass row is written.

Note: the files it wrote before this change are left in place; see the Y1 report for
which rules read them (``apply_marginal_costs_to_network`` lists them as inputs but, with
the ``marginal_costs.fuel_prices`` / ``carbon_price`` defaults present, never reads their
values).

Author: PyPSA-GB
License: MIT
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.FES.fes_commodity_prices import (  # noqa: E402
    MODEL_PRICE_YEAR,
    read_fes_commodity_prices,
    standard_price_tables,
    to_model_money,
)

try:
    from scripts.utilities.logging_config import setup_logging
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def setup_logging(log_path):
        return logging.getLogger(__name__)

# Setup logging
logger = setup_logging(snakemake.log[0] if "snakemake" in dir() else "extract_FES_prices.log")


def extract_prices(workbook_path: str, price_case: str = "Base Case", missing_case: str = "refuse"):
    """Read CP1/CP2 for one case and return the (fuel, carbon) tables in model money.

    Args:
        workbook_path: FES Data Workbook.
        price_case: Case row to take; FES 2025 publishes only "Base Case".
        missing_case: ``refuse`` or ``midpoint`` where a commodity lacks the case (FES 2024 gas has only a
            High and a Low case).

    Returns:
        (fuel_prices, carbon_prices) DataFrames. Raises FESPriceError if the workbook
        cannot be read as declared.
    """
    logger.info(f"Extracting commodity prices from {workbook_path} (case {price_case!r}, missing: {missing_case})")
    read = read_fes_commodity_prices(workbook_path, case=price_case, missing_case=missing_case)
    model = to_model_money(read["prices"], read["conversion"])
    first = model.iloc[0]
    logger.info(
        "Published basis real %s %s; conversion EUR / %.4g (CP1 GBP-to-EUR) / %.6g (CPIH) to GBP real %s",
        int(first["price_year"]), first["currency"], float(first["gbp_to_eur"]),
        float(first["rebase_cpih_factor"]), MODEL_PRICE_YEAR,
    )
    return standard_price_tables(model)


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("FES Fuel and Carbon Price Extraction")
    logger.info("=" * 80)

    workbook_path = snakemake.input.workbook
    fes_year = snakemake.params.fes_year
    price_case = getattr(snakemake.params, "price_case", None) or "Base Case"
    missing_case = getattr(snakemake.params, "missing_case", None) or "refuse"

    logger.info(f"FES Year: {fes_year}")
    logger.info(f"Workbook: {workbook_path}")

    fuel_prices, carbon_prices = extract_prices(workbook_path, price_case, missing_case)
    for year in (2030, 2050):
        f = fuel_prices[fuel_prices["year"].eq(year)]
        c = carbon_prices[carbon_prices["year"].eq(year)]
        if len(f) and len(c):
            logger.info("%s: %s; carbon %.2f GBP/tCO2", year,
                        ", ".join(f"{r.fuel} {r.price_gbp_per_mwh_thermal:.2f}" for r in f.itertuples()),
                        float(c["carbon_price_gbp_per_tco2"].iloc[0]))

    logger.info(f"Saving fuel prices to {snakemake.output.fuel_prices}")
    fuel_prices.to_csv(snakemake.output.fuel_prices, index=False)
    logger.info(f"Saving carbon prices to {snakemake.output.carbon_prices}")
    carbon_prices.to_csv(snakemake.output.carbon_prices, index=False)

    logger.info("=" * 80)
    logger.info("FES Price Extraction Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
