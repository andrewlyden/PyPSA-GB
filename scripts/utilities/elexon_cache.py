"""
ELEXON Calibration Data Cache — Pack and Restore
==================================================

Manages a compressed Parquet bundle of ELEXON calibration data (PN, MID, BOD)
so that Snakemake runs don't need to re-fetch ~365 API calls per year.

Data sizes:
  CSV cache (5 years):   ~1.3 GB
  Parquet bundle:        ~11 MB  (zstd compression)

Two modes:
  1. **Pack** — Convert existing CSV caches → single .tar.gz of Parquet files.
  2. **Restore** — Unpack local archive back to CSV for calibration scripts.

Usage (standalone):
    python scripts/utilities/elexon_cache.py pack
    python scripts/utilities/elexon_cache.py restore --years 2020 2021 2022 2023 2024
    python scripts/utilities/elexon_cache.py status

Via Python:
    from scripts.utilities.elexon_cache import ensure_cache_for_year
    ensure_cache_for_year(2023)  # restore from archive if CSV missing
"""

import argparse
import hashlib
import logging
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

try:
    from scripts.utilities.logging_config import setup_logging
except ImportError:
    def setup_logging(name):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")
        return logging.getLogger(name)

logger = setup_logging("elexon_cache")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ELEXON_CACHE_DIR = PROJECT_ROOT / "resources" / "market" / "elexon"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "market"
ARCHIVE_NAME = "elexon_calibration_cache.tar.gz"

# ─── File manifest per year ───────────────────────────────────────────────────
YEAR_FILES = {
    "pn_data": "pn_data_{year}.csv",
    "mid_prices": "mid_prices_{year}.csv",
    "renewable_bids": "{year}/renewable_bids_cache.csv",
    "renewable_bmu_registry": "{year}/renewable_bmu_registry.csv",
}

# Optional BM validation data (fetched on demand, cached if available)
VALIDATION_FILES = {
    "boalf": "validation/{year}/boalf_data.csv",
    "system_prices_val": "validation/{year}/system_prices.csv",
    "b1610_actual": "validation/{year}/b1610_actual.csv",
}

AVAILABLE_YEARS = [2020, 2021, 2022, 2023, 2024]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _md5(filepath, chunk_size=8192 * 1024):
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _csv_to_parquet(csv_path, parquet_path):
    """Convert a CSV file to Parquet with zstd compression."""
    df = pd.read_csv(csv_path, low_memory=False)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, compression="zstd", index=False)
    return parquet_path


def _parquet_to_csv(parquet_path, csv_path):
    """Convert a Parquet file back to CSV."""
    df = pd.read_parquet(parquet_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
# PACK — Create archive from existing CSV caches
# ══════════════════════════════════════════════════════════════════════════════

def pack_cache(
    years=None,
    cache_dir=None,
    output_path=None,
    logger=logger,
):
    """
    Convert CSV caches to Parquet and bundle into a .tar.gz archive.

    Parameters
    ----------
    years : list of int, optional
        Years to include (default: all available).
    cache_dir : Path, optional
        ELEXON cache directory (default: resources/market/elexon).
    output_path : Path, optional
        Output .tar.gz path (default: data/market/elexon_calibration_cache.tar.gz).

    Returns
    -------
    Path to the created archive.
    """
    cache_dir = Path(cache_dir or ELEXON_CACHE_DIR)
    output_path = Path(output_path or ARCHIVE_DIR / ARCHIVE_NAME)
    years = years or AVAILABLE_YEARS

    logger.info("=" * 70)
    logger.info("ELEXON CACHE — PACK")
    logger.info("=" * 70)
    logger.info(f"Source:  {cache_dir}")
    logger.info(f"Output:  {output_path}")
    logger.info(f"Years:   {years}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        packed = 0

        for year in years:
            for key, pattern in YEAR_FILES.items():
                csv_rel = pattern.format(year=year)
                csv_path = cache_dir / csv_rel
                if not csv_path.exists():
                    logger.warning(f"  Missing: {csv_rel}")
                    continue

                pq_rel = csv_rel.replace(".csv", ".parquet")
                pq_path = tmpdir / pq_rel

                csv_mb = csv_path.stat().st_size / 1e6
                _csv_to_parquet(csv_path, pq_path)
                pq_mb = pq_path.stat().st_size / 1e6
                logger.info(f"  {csv_rel:45s} {csv_mb:8.1f} MB -> {pq_mb:.2f} MB parquet")
                packed += 1

        if packed == 0:
            logger.error("No files found to pack!")
            return None

        # Also include any top-level files
        for extra in ["renewable_bmu_registry.csv", "system_prices.csv"]:
            extra_path = cache_dir / extra
            if extra_path.exists():
                pq_path = tmpdir / extra.replace(".csv", ".parquet")
                _csv_to_parquet(extra_path, pq_path)
                packed += 1

        # Create .tar.gz
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w:gz") as tar:
            for pq_file in sorted(tmpdir.rglob("*.parquet")):
                arcname = str(pq_file.relative_to(tmpdir))
                tar.add(pq_file, arcname=arcname)

        archive_mb = output_path.stat().st_size / 1e6
        checksum = _md5(output_path)
        logger.info(f"\nArchive: {output_path} ({archive_mb:.1f} MB)")
        logger.info(f"MD5:     {checksum}")
        logger.info(f"Files:   {packed}")

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# RESTORE — Unpack archive to CSV
# ══════════════════════════════════════════════════════════════════════════════

def restore_cache(
    years=None,
    cache_dir=None,
    archive_path=None,
    logger=logger,
):
    """
    Restore ELEXON CSV caches from a local Parquet archive.

    Strategy:
      1. Check if CSVs already exist (skip if complete)
      2. Check for local archive -> unpack
      3. If no archive, calibration scripts fall back to ELEXON API

    Parameters
    ----------
    years : list of int
        Years to restore (default: all available).
    cache_dir : Path
        Target ELEXON cache directory.
    archive_path : Path
        Path to .tar.gz archive (default: data/market/elexon_calibration_cache.tar.gz).

    Returns
    -------
    dict mapping year -> list of restored files, or empty dict if nothing needed.
    """
    cache_dir = Path(cache_dir or ELEXON_CACHE_DIR)
    archive_path = Path(archive_path or ARCHIVE_DIR / ARCHIVE_NAME)
    years = years or AVAILABLE_YEARS

    logger.info("=" * 70)
    logger.info("ELEXON CACHE — RESTORE")
    logger.info("=" * 70)

    # Check what's already cached
    missing_years = []
    for year in years:
        pn = cache_dir / f"pn_data_{year}.csv"
        mid = cache_dir / f"mid_prices_{year}.csv"
        if pn.exists() and mid.exists():
            logger.info(f"  {year}: already cached")
        else:
            missing_years.append(year)
            logger.info(f"  {year}: MISSING — needs restore")

    if not missing_years:
        logger.info("All years already cached — nothing to do.")
        return {}

    if not archive_path.exists():
        logger.warning(f"No archive available at {archive_path}")
        logger.warning("Calibration scripts will fall back to ELEXON API (slow).")
        return {}

    # Unpack archive
    logger.info(f"\nUnpacking {archive_path}...")
    restored = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        for year in missing_years:
            year_files = []
            for key, pattern in YEAR_FILES.items():
                pq_rel = pattern.format(year=year).replace(".csv", ".parquet")
                pq_path = tmpdir / pq_rel
                if not pq_path.exists():
                    logger.debug(f"  {pq_rel} not in archive (optional)")
                    continue

                csv_rel = pattern.format(year=year)
                csv_path = cache_dir / csv_rel

                if csv_path.exists():
                    logger.debug(f"  {csv_rel} already exists — skipping")
                    year_files.append(str(csv_path))
                    continue

                _parquet_to_csv(pq_path, csv_path)
                csv_mb = csv_path.stat().st_size / 1e6
                logger.info(f"  Restored: {csv_rel} ({csv_mb:.1f} MB)")
                year_files.append(str(csv_path))

            restored[year] = year_files

        # Also restore top-level files
        for extra in ["renewable_bmu_registry.parquet", "system_prices.parquet"]:
            pq_path = tmpdir / extra
            if pq_path.exists():
                csv_name = extra.replace(".parquet", ".csv")
                csv_path = cache_dir / csv_name
                if not csv_path.exists():
                    _parquet_to_csv(pq_path, csv_path)

    n_years = len(restored)
    n_files = sum(len(v) for v in restored.values())
    logger.info(f"\nRestored {n_files} files for {n_years} years")
    return restored


def cache_status(cache_dir=None, logger=logger):
    """Print a summary of what's cached locally."""
    cache_dir = Path(cache_dir or ELEXON_CACHE_DIR)

    logger.info("=" * 70)
    logger.info("ELEXON CACHE STATUS")
    logger.info("=" * 70)
    logger.info(f"Directory: {cache_dir}")

    for year in AVAILABLE_YEARS:
        files = {}
        total_mb = 0
        for key, pattern in YEAR_FILES.items():
            csv_path = cache_dir / pattern.format(year=year)
            if csv_path.exists():
                mb = csv_path.stat().st_size / 1e6
                files[key] = f"{mb:.1f} MB"
                total_mb += mb
            else:
                files[key] = "MISSING"

        status = "ok" if all(v != "MISSING" for v in files.values()) else "MISSING"
        logger.info(f"\n  {year} [{status}] ({total_mb:.1f} MB total)")
        for key, val in files.items():
            logger.info(f"    {key:25s}: {val}")

    archive = ARCHIVE_DIR / ARCHIVE_NAME
    if archive.exists():
        mb = archive.stat().st_size / 1e6
        logger.info(f"\n  Archive: {archive} ({mb:.1f} MB)")
    else:
        logger.info(f"\n  Archive: not built yet")


# ══════════════════════════════════════════════════════════════════════════════
# SNAKEMAKE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def ensure_cache_for_year(year, cache_dir=None, logger=logger):
    """
    Ensure ELEXON calibration data for a specific year is available.

    Called by calibration scripts before they attempt an API fetch.
    Returns True if cache was restored, False if API fetch needed.
    """
    cache_dir = Path(cache_dir or ELEXON_CACHE_DIR)
    pn = cache_dir / f"pn_data_{year}.csv"
    mid = cache_dir / f"mid_prices_{year}.csv"

    if pn.exists() and mid.exists():
        return True

    logger.info(f"ELEXON cache miss for {year} — attempting restore from archive")
    restored = restore_cache(
        years=[year],
        cache_dir=cache_dir,
        logger=logger,
    )

    return year in restored and len(restored[year]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Manage ELEXON calibration data cache (pack/restore/status)"
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # pack
    p_pack = sub.add_parser("pack", help="Pack CSV caches into compressed archive")
    p_pack.add_argument("--years", type=int, nargs="*", default=None)
    p_pack.add_argument("--output", type=str, default=None)

    # restore
    p_restore = sub.add_parser("restore", help="Restore CSV caches from archive")
    p_restore.add_argument("--years", type=int, nargs="*", default=None)
    p_restore.add_argument("--archive", type=str, default=None)

    # status
    sub.add_parser("status", help="Show cache status")

    args = parser.parse_args()

    if args.command == "pack":
        pack_cache(years=args.years, output_path=args.output, logger=logger)
    elif args.command == "restore":
        restore_cache(years=args.years, archive_path=args.archive, logger=logger)
    elif args.command == "status":
        cache_status(logger=logger)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
