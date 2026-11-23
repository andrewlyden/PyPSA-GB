"""Snakemake entry point for rule build_interconnector_price_shape (item IC1).

The method is in :mod:`scripts.interconnectors.price_shape`.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.interconnectors.price_shape import build_conditioned_shape  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(snakemake.log[0]), logging.StreamHandler()],  # noqa: F821
)
build_conditioned_shape(
    snakemake.input.prices,  # noqa: F821
    snakemake.input.wind_shape,  # noqa: F821
    snakemake.input.wind_weather,  # noqa: F821
    snakemake.output.shape,  # noqa: F821
    snakemake.output.audit,  # noqa: F821
    logger=logging.getLogger("build_interconnector_price_shape"),
)
