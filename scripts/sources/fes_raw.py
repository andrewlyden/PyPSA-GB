"""Cached raw downloads of the NESO FES API tables, one file per edition and table.

The FES API file (``data/FES/FES_api_urls.yaml``) names a URL per table and edition. The first
request for a (table, edition) downloads it to ``resources/sources/raw/fes_{edition}/{table}.csv``
with a sidecar ``{table}.json`` recording the URL, the time fetched, the size and the SHA-256;
every later request reads the cached bytes, so the workflow is offline after the first fetch.
A cached file whose sidecar names another URL is fetched again (a new release of the table).

The bytes are kept exactly as NESO served them. Parsing (encoding, header rows) and cleaning
(GSP aliases, non-ASCII characters) belong to the normalisation step, ``scripts/FES/FES_data.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URLS_FILE = PROJECT_ROOT / "data" / "FES" / "FES_api_urls.yaml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "resources" / "sources" / "raw"

#: Tables the FES API file lists. ``gsp_info`` is the regional breakdown's GSP table (2021-2024).
RAW_TABLES = ("building_blocks", "building_block_definitions", "gsp_info")

#: Seconds between requests to the NESO API.
POLITE_DELAY_S = 1.0

_log = logging.getLogger(__name__)


class RawTableUnavailable(LookupError):
    """No URL for this table and edition, and nothing cached."""


def load_api_urls(urls_file: str | Path = DEFAULT_URLS_FILE) -> dict[str, dict[int, str]]:
    """The FES API file as {table: {edition: url}}.

    Args:
        urls_file: Path to ``FES_api_urls.yaml``.

    Returns:
        URLs by table and edition (editions as int).
    """
    with open(urls_file, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return {str(table): {int(edition): str(url) for edition, url in (entries or {}).items()}
            for table, entries in raw.items()}


def raw_path(edition: int, table: str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    """Cache path of one raw table: ``{raw_dir}/fes_{edition}/{table}.csv``."""
    if table not in RAW_TABLES:
        raise ValueError(f"Unknown FES raw table {table!r}; expected one of {RAW_TABLES}")
    return Path(raw_dir) / f"fes_{int(edition)}" / f"{table}.csv"


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def read_sidecar(edition: int, table: str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> Optional[dict[str, Any]]:
    """The provenance record of a cached table, or None when the table is not cached."""
    side = _sidecar(raw_path(edition, table, raw_dir))
    if not side.is_file():
        return None
    return json.loads(side.read_text(encoding="utf-8"))


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def fetch_raw(
    edition: int,
    table: str,
    urls: Optional[Mapping[str, Mapping[int, str]]] = None,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    session: Any = None,
    refresh: bool = False,
    timeout: float = 60.0,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Path of the cached raw table, downloading it first if needed.

    Args:
        edition: FES edition (publication year), e.g. 2025.
        table: One of :data:`RAW_TABLES`.
        urls: {table: {edition: url}}; default :func:`load_api_urls`.
        raw_dir: Cache root.
        session: Object with ``get(url, timeout=)`` (a ``requests.Session``); default a new session.
        refresh: Download even when a cached copy matches the URL.
        timeout: Request timeout in seconds.
        logger: Logger.

    Returns:
        The cached file path.

    Raises:
        RawTableUnavailable: No URL for the table and edition, and no cached copy.
    """
    logger = logger or _log
    edition = int(edition)
    urls = load_api_urls() if urls is None else urls
    url = (urls.get(table) or {}).get(edition)
    path = raw_path(edition, table, raw_dir)
    meta = read_sidecar(edition, table, raw_dir)
    if path.is_file() and not refresh:
        if url is None or (meta is not None and meta.get("url") == url):
            logger.info("FES %s %s: cached %s", edition, table, path)
            return path
        logger.info("FES %s %s: cached copy is from %s; fetching %s", edition, table,
                    (meta or {}).get("url"), url)
    if url is None:
        raise RawTableUnavailable(
            f"FES {edition} {table}: no URL in the FES API file and no cached copy at {path}")

    if session is None:
        import requests
        session = requests.Session()
    time.sleep(POLITE_DELAY_S)
    logger.info("FES %s %s: downloading %s", edition, table, url)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    _write_atomic(path, content)
    record = {"source": "FES", "edition": edition, "table": table, "url": url,
              "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    _write_atomic(_sidecar(path), (json.dumps(record, indent=2) + "\n").encode("utf-8"))
    logger.info("FES %s %s: %d bytes cached at %s", edition, table, len(content), path)
    return path


def raw_bytes(edition: int, table: str, **kwargs: Any) -> bytes:
    """The raw table's bytes (cached, downloading first if needed); kwargs as :func:`fetch_raw`."""
    return fetch_raw(edition, table, **kwargs).read_bytes()


def cached_editions(table: str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[int]:
    """Editions with a cached copy of ``table``."""
    root = Path(raw_dir)
    editions = []
    for folder in root.glob("fes_*"):
        try:
            edition = int(folder.name.split("_", 1)[1])
        except ValueError:
            continue
        if (folder / f"{table}.csv").is_file():
            editions.append(edition)
    return sorted(editions)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: cache every table of the given editions (default: all in the API file)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edition", type=int, action="append", help="FES edition (repeatable)")
    parser.add_argument("--table", choices=RAW_TABLES, action="append", help="table (repeatable)")
    parser.add_argument("--urls-file", default=str(DEFAULT_URLS_FILE))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    urls = load_api_urls(args.urls_file)
    tables = args.table or list(RAW_TABLES)
    for table in tables:
        editions = args.edition or sorted(urls.get(table, {}))
        for edition in editions:
            if edition not in urls.get(table, {}):
                _log.info("FES %s %s: not served by the API file; skipped", edition, table)
                continue
            fetch_raw(edition, table, urls=urls, raw_dir=args.raw_dir, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
