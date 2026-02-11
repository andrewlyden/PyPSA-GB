"""
Download atlite cutouts with a tiered strategy:
  1. Check data directory for cached copy (instant)
  2. Try downloading from Zenodo repository (fast, ~minutes)
  3. Fall back to atlite ERA5 download (slow, ~hours)

Snakemake handles checking if the output file already exists.
Zenodo record: https://zenodo.org/records/18325225
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Zenodo record ID for PyPSA-GB atlite cutouts
ZENODO_RECORD_ID = "18325225"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Known MD5 checksums from the Zenodo record (for integrity verification)
ZENODO_CHECKSUMS = {
    "uk-2010.nc": "3989174c06d8740763d1c26adde6ad86",
    "uk-2011.nc": "037cc3af69320f24215ace60003ce968",
    "uk-2012.nc": "9a8cbf90f822559529bbeef3e27e0e55",
    "uk-2013.nc": "0bb5945d20390214dcab0f69a104066b",
    "uk-2014.nc": "a19bf441cba3966cdc8880676e504903",
    "uk-2015.nc": "85baacbc55195c293350fe61f13232e5",
    "uk-2016.nc": "f80c3f7ce221983cd029a577ff234d20",
    "uk-2017.nc": "bf2d07cd34cbfbf246cabcf86b9c586d",
    "uk-2018.nc": "40c03a68b73a5f9ce365c92885185d9e",
    "uk-2019.nc": "07853e0850eb1f5bbd818ac756f22b30",
    "uk-2020.nc": "9d08ab99530454944a8efffcd547b720",
    "uk-2021.nc": "207ab54d807596b6f51970b1d1007ffa",
    "uk-2022.nc": "45d78b30a2ea7a26f9a9600a04fb997f",
    "uk-2023.nc": "8e5bc5f4d4dc01a87d923f357ef94242",
    "uk-2024.nc": "4e0fe4991d8173e5775b0ab22d5a5536",
}


def _md5_checksum(filepath, chunk_size=8192 * 1024):
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _get_zenodo_file_url(filename):
    """
    Build the direct download URL for a file in the Zenodo record.

    Uses the Zenodo API content endpoint which supports streaming downloads.
    """
    return f"{ZENODO_API_URL}/files/{filename}/content"


def query_zenodo_available_files():
    """
    Query the Zenodo API to get the list of available cutout files.

    Returns a dict mapping filename -> {url, size, checksum} or None on failure.
    """
    try:
        resp = requests.get(ZENODO_API_URL, timeout=30)
        resp.raise_for_status()
        record = resp.json()
        files = {}
        for f in record.get("files", []):
            files[f["key"]] = {
                "url": f["links"]["self"],
                "size": f["size"],
                "checksum": f.get("checksum", "").replace("md5:", ""),
            }
        return files
    except Exception as e:
        logger.warning(f"Could not query Zenodo API: {e}")
        return None


def is_available_on_zenodo(filename, zenodo_files=None):
    """Check whether a cutout file is available on Zenodo."""
    # Fast path: check against hardcoded list
    if filename in ZENODO_CHECKSUMS:
        return True
    # Slow path: query the API (handles future uploads)
    if zenodo_files is not None:
        return filename in zenodo_files
    return False


def download_from_zenodo(filename, output_path, verify_checksum=True, zenodo_files=None):
    """
    Download a cutout file from Zenodo.

    Parameters
    ----------
    filename : str
        The filename on Zenodo, e.g. "uk-2021.nc"
    output_path : str or Path
        Local path to save the file to.
    verify_checksum : bool
        Whether to verify the MD5 checksum after download.
    zenodo_files : dict or None
        Pre-fetched Zenodo file metadata (from query_zenodo_available_files).

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine download URL
    if zenodo_files and filename in zenodo_files:
        url = zenodo_files[filename]["url"]
    else:
        url = _get_zenodo_file_url(filename)

    # Stream download with progress reporting
    tmp_path = output_path.with_suffix(".nc.tmp")
    try:
        logger.info(f"Downloading {filename} from Zenodo...")
        logger.info(f"  URL: {url}")

        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    # Log every 10%
                    if pct >= last_pct + 10:
                        last_pct = pct
                        logger.info(
                            f"  Progress: {pct}% "
                            f"({downloaded / 1e6:.0f} / {total_size / 1e6:.0f} MB)"
                        )

        # Verify checksum
        if verify_checksum:
            expected = ZENODO_CHECKSUMS.get(filename)
            if zenodo_files and filename in zenodo_files:
                expected = zenodo_files[filename].get("checksum", expected)
            if expected:
                actual = _md5_checksum(tmp_path)
                if actual != expected:
                    logger.error(
                        f"Checksum mismatch for {filename}: "
                        f"expected {expected}, got {actual}"
                    )
                    tmp_path.unlink(missing_ok=True)
                    return False
                logger.info(f"  Checksum verified: {actual}")
            else:
                logger.warning(f"  No checksum available for {filename}, skipping verification")

        # Move to final location
        shutil.move(str(tmp_path), str(output_path))
        logger.info(f"  Saved to: {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        logger.warning(f"Zenodo download failed for {filename}: {e}")
        tmp_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {filename} from Zenodo: {e}")
        tmp_path.unlink(missing_ok=True)
        return False


def download_with_atlite(year, output_path):
    """
    Fall back to downloading via atlite/ERA5 (slow, 2-4 hours per year).

    Parameters
    ----------
    year : int
        The weather year to download.
    output_path : str or Path
        Local path to save the cutout to.
    """
    import atlite
    import cartopy.io.shapereader as shpreader
    import geopandas as gpd

    logger.info(f"Falling back to atlite ERA5 download for {year}...")
    logger.info("  This may take 2-4 hours. Please be patient.")

    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    UK = gpd.GeoSeries(
        {r.attributes["NAME_EN"]: r.geometry for r in reader.records()},
        crs={"init": "epsg:4326"},
    ).reindex(["United Kingdom"])

    cutout = atlite.Cutout(
        path=str(output_path),
        module="era5",
        bounds=UK.unary_union.bounds,
        time=str(year),
    )
    cutout.prepare()
    logger.info(f"  atlite download complete for {year}")


def acquire_cutout(year, output_path, data_dir=None, enable_zenodo=True,
                   verify_checksum=True, zenodo_files=None):
    """
    Acquire a cutout file using a tiered strategy:

    1. Check if it exists in data_dir (copy)
    2. Try downloading from Zenodo (fast)
    3. Fall back to atlite ERA5 download (slow)

    Note: Snakemake handles checking if output_path already exists,
    so we don't need to duplicate that logic here.

    Parameters
    ----------
    year : int
        The weather year.
    output_path : str or Path
        Target file path, e.g. "resources/atlite/cutouts/uk-2021.nc"
    data_dir : str or Path or None
        Optional directory to check for pre-existing cutouts (e.g. "data/atlite/cutouts")
    enable_zenodo : bool
        Whether to try Zenodo before falling back to atlite.
    verify_checksum : bool
        Whether to verify MD5 checksums on Zenodo downloads.
    zenodo_files : dict or None
        Pre-fetched Zenodo file metadata.

    Returns
    -------
    str
        The source of the cutout: "data_dir", "zenodo", or "atlite"
    """
    output_path = Path(output_path)
    filename = f"uk-{year}.nc"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Check data directory ---
    if data_dir:
        data_path = Path(data_dir) / filename
        if data_path.exists():
            logger.info(f"[DATA DIR] Found cutout for {year} at {data_path}, copying...")
            shutil.copy2(str(data_path), str(output_path))
            logger.info(f"  Copied to {output_path}")
            return "data_dir"

    # --- Step 2: Try Zenodo download ---
    if enable_zenodo and is_available_on_zenodo(filename, zenodo_files):
        logger.info(f"[ZENODO] Cutout for {year} is available on Zenodo, downloading...")
        success = download_from_zenodo(
            filename, output_path,
            verify_checksum=verify_checksum,
            zenodo_files=zenodo_files,
        )
        if success:
            return "zenodo"
        else:
            logger.warning(f"  Zenodo download failed, falling back to atlite...")

    # --- Step 3: Fall back to atlite ERA5 ---
    logger.info(f"[ATLITE] Downloading cutout for {year} from ERA5 via atlite...")
    download_with_atlite(year, output_path)
    return "atlite"
