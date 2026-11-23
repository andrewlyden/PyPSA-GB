"""Snakemake entry point for rule build_tyndp_price_shape (task Y3, FES 2050 plan; any TYNDP year).

The method is in :mod:`scripts.interconnectors.tyndp_prices`.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.interconnectors.tyndp_prices import blend_ntplus_shapes, build_tyndp_shape  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(snakemake.log[0]), logging.StreamHandler()],  # noqa: F821
)
_inputs = snakemake.input  # noqa: F821
if "bracket_lo_shape" in _inputs.keys():
    # User decision (11 Nov 2026): a year NT+ does not publish blends its neighbours' shapes.
    blend_ntplus_shapes(_inputs.bracket_lo_shape, _inputs.bracket_hi_shape, _inputs.bracket_lo_audit,
                        _inputs.bracket_hi_audit, snakemake.output.shape, snakemake.output.audit,  # noqa: F821
                        int(snakemake.wildcards.year), snakemake.wildcards.variant,  # noqa: F821
                        logger=logging.getLogger("build_tyndp_price_shape"))
    sys.exit(0)
_climate_files = {
    int(key.split("_")[-1]): value
    for key, value in _inputs.items()
    if key.startswith("tyndp2024_cy_")
}
build_tyndp_shape(
    snakemake.wildcards.variant,  # noqa: F821
    _inputs.wind_weather,
    snakemake.output.shape,  # noqa: F821
    snakemake.output.audit,  # noqa: F821
    tyndp2024_files=_climate_files or None,
    ntplus_price_file=_inputs.get("ntplus_prices"),
    ntplus_wind_file=_inputs.get("ntplus_wind"),
    ic1_shape_file=_inputs.get("ic1_shape"),
    eur_to_gbp=float(snakemake.params.eur_to_gbp),  # noqa: F821
    calendar_year=int(snakemake.wildcards.weather_year),  # noqa: F821
    logger=logging.getLogger("build_tyndp_price_shape"),
    tyndp_year=int(snakemake.wildcards.year),  # noqa: F821
)
