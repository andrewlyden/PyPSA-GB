"""Snakemake entry point for rule build_cp30_foreign_prices (item F13).

The method is in :mod:`scripts.interconnectors.cp30_foreign_prices`.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.interconnectors.cp30_foreign_prices import build_cp30_basis_prices, build_cp30_refit_prices  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(snakemake.log[0]), logging.StreamHandler()],  # noqa: F821
)
_p = snakemake.params  # noqa: F821
_i = snakemake.input  # noqa: F821
_o = snakemake.output  # noqa: F821
_logger = logging.getLogger("build_cp30_foreign_prices")
if str(_p.method) == "refit_2013":
    build_cp30_refit_prices(
        _i.zonal, _i.cf, _o.prices, _o.audit,
        ireland_shape_file=_i.get("ireland_shape"),
        tyndp_new_ccgt_eur=float(_p.tyndp_new_ccgt_eur),
        foreign_new_ccgt_gbp=float(_p.foreign_new_ccgt_gbp),
        target_price_cap_eur=float(_p.target_price_cap_eur),
        ireland_source=str(_p.ireland_source),
        ireland_offset_gbp=float(_p.ireland_offset_gbp),
        calendar_year=int(snakemake.wildcards.weather_year),  # noqa: F821
        composition_file=_i.get("composition"),
        logger=_logger,
    )
else:
    build_cp30_basis_prices(
        _i.tyndp, _i.wind_weather, _o.prices, _o.audit,
        tyndp_new_ccgt_eur=float(_p.tyndp_new_ccgt_eur),
        foreign_new_ccgt_gbp=float(_p.foreign_new_ccgt_gbp),
        calendar_year=int(snakemake.wildcards.weather_year),  # noqa: F821
        composition_file=_i.get("composition"),
        logger=_logger,
    )
