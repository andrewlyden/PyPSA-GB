"""Write one source edition's contract tables: ``python -m scripts.sources.build --source FES --edition 2025``.

Output: ``resources/sources/{source}_{edition}/`` (``scripts/sources/contract.py``). Nothing in the
workflow reads these tables yet; they sit beside the existing readers until the later stages of the
year-generic plan move the consumers onto them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sources import contract  # noqa: E402


def build(source: str, edition: int, out_root: str | Path = contract.DEFAULT_SOURCES_DIR,
          workbook: Optional[str] = None, fes_data: Optional[str] = None, es1: Optional[str] = None,
          es5: Optional[str] = None, ed1: Optional[str] = None, include_gsp_blocks: bool = True) -> Path:
    """Run the source's adapter and write its tables.

    Args:
        source: "FES" or "CP30".
        edition: Edition.
        out_root: Sources root.
        workbook: FES Data Workbook (FES; default the edition's).
        fes_data: ``FES_{edition}_data.csv`` (FES; default built from the raw cache).
        es1, es5, ed1: CP30 extracted CSVs (CP30; default resources/CP30/).
        include_gsp_blocks: FES: write gsp_blocks.

    Returns:
        The output directory.
    """
    source = source.upper()
    if source == "CP30":
        from scripts.sources import cp30_adapter
        if int(edition) != cp30_adapter.EDITION:
            raise ValueError(f"CP30 edition {edition} unknown; the registered edition is {cp30_adapter.EDITION}")
        kwargs = {k: v for k, v in {"es1_path": es1, "es5_path": es5, "ed1_path": ed1}.items() if v}
        tables, meta = cp30_adapter.build_cp30_tables(**kwargs)
    elif source == "FES":
        from scripts.sources import fes_adapter
        tables, meta = fes_adapter.build_fes_tables(int(edition), workbook=workbook, fes_data=fes_data,
                                                    include_gsp_blocks=include_gsp_blocks)
    else:
        raise ValueError(f"Unknown source {source!r}; expected FES or CP30")
    return contract.write_source(source, int(edition), tables, meta, root=out_root)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write a source edition's contract tables")
    parser.add_argument("--source", required=True, choices=["FES", "CP30", "fes", "cp30"])
    parser.add_argument("--edition", type=int, required=True)
    parser.add_argument("--out-root", default=str(contract.DEFAULT_SOURCES_DIR))
    parser.add_argument("--workbook")
    parser.add_argument("--fes-data")
    parser.add_argument("--es1")
    parser.add_argument("--es5")
    parser.add_argument("--ed1")
    parser.add_argument("--no-gsp-blocks", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = build(args.source, args.edition, args.out_root, args.workbook, args.fes_data, args.es1, args.es5,
                args.ed1, include_gsp_blocks=not args.no_gsp_blocks)
    logging.getLogger(__name__).info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
