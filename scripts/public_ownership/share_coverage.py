"""Check that an ownership share table names the network it is about to bid on.

The share and commissioning-year tables (``ownership_shares.py``) are keyed on generator
names and read by the balancing stage as ``ancient()`` inputs, so a network rebuild that
renames units does not re-run them. The bid code treats a unit missing from the table as
private, so a stale table does not fail: it quietly hands state plant back to private
owners. On 19 Oct 2026 the bus-position rebuild renamed about 1,370 units per P2/P3 case
(4.8% of P2's state capacity) and 48 bid months were solved on the stale tables before
the bill merge's check caught it.

``check_share_coverage`` measures the state capacity in the table whose generator is not
a component of the network and raises above ``MAX_UNMATCHED_STATE_SHARE``. The balancing
stage calls it before it bids (``cfd_bm_bids.resolve_public_ownership_bids``), and the
command line runs it on a saved network before a queue launches bid months:

    python scripts/public_ownership/share_coverage.py \\
        --network resources/market/<parent month>_wholesale.nc \\
        --table resources/public_ownership/cp30_2030_cp30_network_generator_state_shares.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# A rebuilt table on the network it was built from leaves nothing out (22 Oct 2026: 0 MW of
# 96.9 GW for P2, 0 of 197.2 GW for P3); the stale one left out 4.82% and 2.61%.
MAX_UNMATCHED_STATE_SHARE = 0.001

REBUILD_HINT = (
    "rebuild the ownership tables on the current network: snakemake -R "
    "build_public_ownership_state_shares build_public_ownership_commissioning_years "
    "(and the asset finance register first if the network's generators changed), then "
    "re-solve the bid months that used the old tables")

COMPONENT_DIMS = ("generators_i", "storage_units_i", "links_i", "stores_i")


def network_component_names(network) -> set[str]:
    """Names of the generators, storage units, links and stores of a network or a saved .nc."""
    if isinstance(network, (str, Path)):
        import xarray as xr

        with xr.open_dataset(network) as ds:
            return {str(n) for dim in COMPONENT_DIMS if dim in ds.coords for n in ds[dim].values}
    names: set[str] = set()
    for attr in ("generators", "storage_units", "links", "stores"):
        frame = getattr(network, attr, None)
        if frame is not None:
            names.update(map(str, frame.index))
    return names


def share_coverage(table: pd.DataFrame, names: set[str], case_id: str | None = None) -> pd.DataFrame:
    """Per case: state-owned units and MW, and those whose generator the network lacks."""
    if case_id is not None:
        table = table[table["case_id"] == case_id]
    owned = table[pd.to_numeric(table["state_capacity_mw"], errors="coerce").fillna(0.0) > 0.0]
    rows = []
    for case, group in owned.groupby("case_id", sort=False):
        missing = group[~group["generator"].astype(str).isin(names)]
        total = float(group["state_capacity_mw"].sum())
        lost = float(missing["state_capacity_mw"].sum())
        rows.append({
            "case_id": case,
            "owned_units": len(group),
            "owned_mw": total,
            "unmatched_units": len(missing),
            "unmatched_mw": lost,
            "unmatched_share": lost / total if total > 0.0 else 0.0,
            "examples": list(missing.sort_values("state_capacity_mw", ascending=False)["generator"].astype(str)[:5]),
        })
    return pd.DataFrame(rows)


def check_share_coverage(table_path, network, case_id: str | None = None,
                         max_share: float = MAX_UNMATCHED_STATE_SHARE,
                         logger: logging.Logger | None = None) -> pd.DataFrame:
    """Raise if more than ``max_share`` of a case's state capacity is on units the network lacks."""
    logger = logger or logging.getLogger(__name__)
    table = pd.read_csv(table_path)
    if "state_capacity_mw" not in table.columns:
        raise ValueError(f"{table_path} has no state_capacity_mw column; is it an ownership table?")
    if case_id is not None and not (table["case_id"] == case_id).any():
        raise ValueError(f"{table_path} has no rows for case {case_id!r}")
    result = share_coverage(table, network_component_names(network), case_id)
    if result.empty:
        return result
    bad = result[result["unmatched_share"] > max_share]
    for row in result.itertuples():
        logger.info("Ownership table %s, case %s: %d of %d state-owned units (%.0f of %.0f MW, %.2f%%) "
                    "are not in the network", Path(table_path).name, row.case_id, row.unmatched_units,
                    row.owned_units, row.unmatched_mw, row.owned_mw, 100.0 * row.unmatched_share)
    if not bad.empty:
        detail = "; ".join(f"{r.case_id}: {r.unmatched_mw:.0f} MW ({100.0 * r.unmatched_share:.2f}%), "
                           f"e.g. {r.examples}" for r in bad.itertuples())
        raise ValueError(
            f"ownership table {table_path} does not match the network: state capacity on units the "
            f"network lacks exceeds {100.0 * max_share:.2f}% ({detail}). The bids would treat that "
            f"plant as private. To fix: {REBUILD_HINT}.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--network", required=True, help="saved PyPSA network (.nc)")
    parser.add_argument("--table", required=True, nargs="+", help="ownership share table(s)")
    parser.add_argument("--case", default=None, help="check one case only")
    parser.add_argument("--max-share", type=float, default=MAX_UNMATCHED_STATE_SHARE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failed = False
    for table in args.table:
        try:
            check_share_coverage(table, args.network, args.case, args.max_share)
        except ValueError as exc:
            logging.error("%s", exc)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
