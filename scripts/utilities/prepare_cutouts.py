"""
Prepare atlite cutouts using a tiered acquisition strategy:

  1. Data dir: check for a cached copy in data/atlite/cutouts/
  2. Zenodo: download from https://zenodo.org/records/18325225 (~minutes)
  3. Atlite/ERA5: full download from CDS API (~2-4 hours per year)

Snakemake handles checking if the output file already exists and will
skip re-running this script if the output is up to date.

This avoids unnecessary multi-hour ERA5 downloads when cutouts are
already available in data_dir or on Zenodo.
"""

import logging
import os
import sys

# Set up logging for both standalone and snakemake usage
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to path so we can import sibling modules
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.utilities.download_cutouts import (
    acquire_cutout,
    query_zenodo_available_files,
)


# Default directory to check for pre-existing cutouts
DATA_DIR = "data/atlite/cutouts"


def prepare_cutouts(years, outputs, enable_zenodo=True, verify_checksum=True):
    """
    Prepare cutouts for the given years using the tiered strategy.

    Parameters
    ----------
    years : list[int]
        Weather years to acquire cutouts for.
    outputs : list[str]
        Corresponding output file paths.
    enable_zenodo : bool
        Whether to try Zenodo before falling back to atlite.
    verify_checksum : bool
        Whether to verify MD5 checksums on Zenodo downloads.
    """
    logger.info(f"Preparing cutouts for years: {years}")
    logger.info(f"Zenodo download: {'enabled' if enable_zenodo else 'disabled'}")

    # Pre-fetch Zenodo file listing once (avoids repeated API calls)
    zenodo_files = None
    if enable_zenodo:
        logger.info("Querying Zenodo for available cutout files...")
        zenodo_files = query_zenodo_available_files()
        if zenodo_files:
            available = sorted(zenodo_files.keys())
            logger.info(f"  Zenodo has {len(available)} cutout files available")
        else:
            logger.warning("  Could not reach Zenodo API; will use hardcoded file list")

    sources = {}
    for i, year in enumerate(years):
        output_path = outputs[i]
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(years)}] Acquiring cutout for {year}")
        logger.info(f"  Target: {output_path}")

        source = acquire_cutout(
            year=year,
            output_path=output_path,
            data_dir=DATA_DIR,
            enable_zenodo=enable_zenodo,
            verify_checksum=verify_checksum,
            zenodo_files=zenodo_files,
        )
        sources[year] = source
        logger.info(f"  Source: {source}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CUTOUT ACQUISITION SUMMARY")
    logger.info(f"{'='*60}")
    for year, source in sources.items():
        icon = {
            "data_dir": "[CACHED] ",
            "zenodo": "[ZENODO] ",
            "atlite": "[ERA5]   ",
        }.get(source, "[?]      ")
        logger.info(f"  {icon} uk-{year}.nc")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    # Get parameters from snakemake
    years = snakemake.params.years
    outputs = snakemake.output

    # Read Zenodo settings from config if available
    snakemake_config = getattr(snakemake.params, "config", {})
    zenodo_config = snakemake_config.get("zenodo", {})
    enable_zenodo = zenodo_config.get("enabled", True)
    verify_checksum = zenodo_config.get("verify_checksum", True)

    prepare_cutouts(
        years=years,
        outputs=outputs,
        enable_zenodo=enable_zenodo,
        verify_checksum=verify_checksum,
    )

