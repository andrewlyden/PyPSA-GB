"""How bill lines are grouped for presentation: dated data, not code.

Every bill line has a stable ``component_id``. Which group a line is shown in is
read from ``data/market/consumer_bill_benchmarks/bill_groupings.csv``, one row
per (view, vintage, line), with the views and their current vintage declared in
``bill_grouping_views.csv`` beside it. Two views (6 Oct 2026):

* ``ofgem``: Ofgem's price-cap categories at a named cap release. Ofgem moves
  lines between categories (the CfD left policy costs for the wholesale
  allowance after 2020), so each release is its own set of rows: a new release
  adds a vintage, it never edits an old one. The bill headline uses it.
* ``functional``: what each cost pays for, our own and independent of Ofgem.
  The ownership comparison and the P3 tariff use it.

The source columns ``stack_group`` and ``category`` are left as they are: the
2020 historical breakdown and the 2050 notebook key on them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GROUPINGS_PATH = ROOT / "data" / "market" / "consumer_bill_benchmarks" / "bill_groupings.csv"
VIEWS_PATH = ROOT / "data" / "market" / "consumer_bill_benchmarks" / "bill_grouping_views.csv"
VIEWS = ("ofgem", "functional")
VALUE = "annual_gbp_per_3100kwh_ex_vat"


class GroupingError(ValueError):
    """Raised when the grouping data is malformed or a bill line has no group."""


@lru_cache(maxsize=None)
def _views(path: str) -> pd.DataFrame:
    views = pd.read_csv(path, dtype=str).fillna("")
    if views.duplicated(["view", "vintage"]).any():
        raise GroupingError(f"{path}: a (view, vintage) is declared twice")
    unknown = sorted(set(views["view"]) - set(VIEWS))
    if unknown:
        raise GroupingError(f"{path}: unknown views {unknown}; the views are {list(VIEWS)}")
    for view in VIEWS:
        current = views.loc[views["view"].eq(view) & views["status"].eq("current")]
        if len(current) != 1:
            raise GroupingError(f"{path}: view {view!r} needs exactly one current vintage, has {len(current)}")
    return views


@lru_cache(maxsize=None)
def _groupings(path: str, views_path: str) -> pd.DataFrame:
    rows = pd.read_csv(path, dtype={"vintage": str, "group_order": int})
    declared = set(map(tuple, _views(views_path)[["view", "vintage"]].to_numpy()))
    undeclared = sorted(set(map(tuple, rows[["view", "vintage"]].to_numpy())) - declared)
    if undeclared:
        raise GroupingError(f"{path}: (view, vintage) pairs not declared in {views_path}: {undeclared}")
    twice = rows.loc[rows.duplicated(["view", "vintage", "component_id"]), ["view", "vintage", "component_id"]]
    if not twice.empty:
        raise GroupingError(f"{path}: lines grouped twice: {twice.to_dict('records')}")
    orders = rows.groupby(["view", "vintage", "group"])["group_order"].nunique()
    if (orders > 1).any():
        raise GroupingError(f"{path}: groups with more than one group_order: {list(orders[orders > 1].index)}")
    ranks = rows.drop_duplicates(["view", "vintage", "group"]).groupby(["view", "vintage"])["group_order"]
    if (ranks.nunique() != ranks.size()).any():
        raise GroupingError(f"{path}: two groups of one view share a group_order")
    return rows


def load_views(views_path: Path | str = VIEWS_PATH) -> pd.DataFrame:
    """The declared views and vintages, one current vintage per view."""
    return _views(str(views_path)).copy()


def load_groupings(path: Path | str = GROUPINGS_PATH, views_path: Path | str = VIEWS_PATH) -> pd.DataFrame:
    """Every (view, vintage, component_id) -> group row, validated."""
    return _groupings(str(path), str(views_path)).copy()


def current_vintage(view: str, views_path: Path | str = VIEWS_PATH) -> str:
    """The vintage a view uses unless a caller names another."""
    views = _views(str(views_path))
    return str(views.loc[views["view"].eq(view) & views["status"].eq("current"), "vintage"].iloc[0])


def view_title(view: str, vintage: str | None = None, views_path: Path | str = VIEWS_PATH) -> str:
    """The view's display title, naming its vintage (for captions and notes)."""
    views = _views(str(views_path))
    vintage = vintage or current_vintage(view, views_path)
    return str(views.loc[views["view"].eq(view) & views["vintage"].eq(vintage), "title"].iloc[0])


def _rows(view: str, vintage: str | None, path, views_path) -> pd.DataFrame:
    if view not in VIEWS:
        raise GroupingError(f"unknown view {view!r}; the views are {list(VIEWS)}")
    vintage = vintage or current_vintage(view, views_path)
    rows = _groupings(str(path), str(views_path))
    rows = rows.loc[rows["view"].eq(view) & rows["vintage"].eq(vintage)]
    if rows.empty:
        raise GroupingError(f"view {view!r} has no rows at vintage {vintage!r}")
    return rows


def group_order(view: str, vintage: str | None = None, path: Path | str = GROUPINGS_PATH,
                views_path: Path | str = VIEWS_PATH) -> list[str]:
    """The view's groups in display order."""
    rows = _rows(view, vintage, path, views_path).drop_duplicates("group").sort_values("group_order")
    return rows["group"].tolist()


def group_of(view: str, vintage: str | None = None, path: Path | str = GROUPINGS_PATH,
             views_path: Path | str = VIEWS_PATH) -> pd.Series:
    """``component_id -> group`` for one view and vintage."""
    return _rows(view, vintage, path, views_path).set_index("component_id")["group"]


def annotate(frame: pd.DataFrame, vintages: dict[str, str] | None = None, path: Path | str = GROUPINGS_PATH,
             views_path: Path | str = VIEWS_PATH) -> pd.DataFrame:
    """A copy of a bill-line frame with ``ofgem_group``, ``ofgem_group_vintage`` and ``functional_group``.

    Refuses a frame with a line that has no group in either view, naming the lines,
    so a new bill line cannot reach a chart or a table ungrouped.
    """
    vintages = vintages or {}
    out = frame.copy()
    ids = out["component_id"].astype(str)
    for view in VIEWS:
        vintage = vintages.get(view) or current_vintage(view, views_path)
        groups = group_of(view, vintage, path, views_path)
        missing = sorted(set(ids) - set(groups.index))
        if missing:
            raise GroupingError(
                f"bill lines with no {view!r} group at vintage {vintage!r}: {missing}. Add a row for each to "
                f"{Path(path).name} (view, vintage, component_id, group, group_order, basis).")
        out[f"{view}_group"] = ids.map(groups).to_numpy()
        if view == "ofgem":
            out["ofgem_group_vintage"] = vintage
    return out


def totals_by_group(frame: pd.DataFrame, view: str, vintage: str | None = None, value: str = VALUE,
                    path: Path | str = GROUPINGS_PATH, views_path: Path | str = VIEWS_PATH) -> pd.Series:
    """The frame's lines summed by group, in the view's order (groups with no line show 0)."""
    vintage = vintage or current_vintage(view, views_path)
    grouped = annotate(frame, {view: vintage}, path, views_path)
    sums = grouped.groupby(f"{view}_group")[value].sum()
    return sums.reindex(group_order(view, vintage, path, views_path), fill_value=0.0)


def summary_block(frame: pd.DataFrame, value: str = VALUE, path: Path | str = GROUPINGS_PATH,
                  views_path: Path | str = VIEWS_PATH) -> dict:
    """The per-view group totals a bill summary JSON carries (ex VAT, the frame's own units)."""
    block = {}
    for view in VIEWS:
        vintage = current_vintage(view, views_path)
        totals = totals_by_group(frame, view, vintage, value, path, views_path)
        block[view] = {"vintage": vintage, "title": view_title(view, vintage, views_path),
                       "totals_gbp_per_3100kwh_ex_vat": {g: float(v) for g, v in totals.items()}}
    return block
