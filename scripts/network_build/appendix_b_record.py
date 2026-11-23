"""What every applied ETYS Appendix B row did to a network, so that a delivery case can undo any of them.

``ETYS_upgrades.apply_etys_network_upgrades`` applies the in-scope Appendix B rows (circuit
and transformer additions, removals and changes, and the B-5-1 HVDC links), including the
later-snapshot rows it pulls in: the CP30-dated works (``add_cp30_dated_works``) and the
HVDC terminal enabling works (``add_hvdc_enabling_works``). The build records what each
row did, in ``network.meta[APPENDIX_B_META_KEY]`` (a JSON string that survives pickles
and the netCDF export), with impedances in pu on 100 MVA (``units``) as the build held them:

* ``rows``: one entry per in-scope row: ``id`` (see :func:`row_id`), ``kind`` (circuit,
  transformer, hvdc), ``operation`` (addition, removal, change), ``source_year``, the
  workbook terminals ``node1``/``node2``, ``operator``, ``winter_rating_mva``,
  ``cp30_dated_work`` (the CP30 project code of a pulled dated row), ``hvdc_enabling_works``
  and ``audit_status`` (the row's status in the upgrade audit, whose ``appendix_b_row_id``
  column carries the same id).
* ``ops``: what the rows did, in the order the build did it (``seq``), one entry per
  component touched:

  - ``add``: ``after`` holds the new branch's rating, impedance and seasonal ratings
    (:data:`CHANGE_FIELDS`; links: ``p_nom``);
  - ``change``: ``before`` and ``after`` hold those fields either side of the row (a row
    changes every parallel circuit between its terminals);
  - ``remove``: ``before`` is the full branch row at removal;
  - ``prune``: a branch or bus the dead-island pruning took out (``island``), full row.

* ``islands``: the pruned dead islands, their buses and ``caused_by``: the removal rows
  whose terminals lie in the island (empty for an island that never connected).
* ``bus_rows``: the full rows of every removed branch's terminal buses (at the removal),
  of pruned buses and of buses ``remove_orphan_buses`` dropped, for re-adding them.
* ``buses_added``: buses the upgrade step created that survived the build.

The CP30-dated works keep their own tag: ``cp30_dated_work`` on lines, transformers, links
and buses holds the project code(s), ``;``-joined, of the dated rows that added or changed
the component (buses: that the dated rows' new branches alone created). NaN elsewhere
(the netCDF import reads NaN back as an empty string).

``scripts/market/network_delivery.py`` (``revert_appendix_b``) reads the record: undoing a
set of rows replays each touched component's operations without them, so two changes on one
circuit, a removal of a circuit a later row added and chained re-terminations all come back
to the state the remaining rows alone give.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

APPENDIX_B_META_KEY = "appendix_b_works"
APPENDIX_B_RECORD_VERSION = 1
APPENDIX_B_ROW_ID_COLUMN = "appendix_b_row_id"
CP30_DATED_WORK_COLUMN = "cp30_dated_work"

# Attributes an Appendix B row sets on a branch, recorded either side of it and restored by the
# revert (``seasonal_rating_source`` and ``s_nom_winter`` are recomputed after a revert).
CHANGE_FIELDS = ("s_nom", "r", "x", "b", "g", "s_nom_spring", "s_nom_summer", "s_nom_autumn")
LINK_FIELDS = ("p_nom",)

COMPONENT_CLASS = {"line": "Line", "transformer": "Transformer", "link": "Link", "bus": "Bus"}
COMPONENT_TABLE = {"line": "lines", "transformer": "transformers", "link": "links", "bus": "buses"}

#: Appendix B ``Status`` -> record operation.
OPERATION = {"Addition": "addition", "Removed": "removal", "Change": "change", "Modify": "change"}


def _plain(value):
    """A JSON-safe scalar, or ``None`` for a missing value."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str)):
        return value if value != "" else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def output_attributes(network, component: str) -> set:
    """PyPSA's output (dependent or solved) attributes of ``component``: never recorded or restored."""
    attrs = network.components[COMPONENT_CLASS[component]].defaults
    return set(attrs.index[attrs["status"].astype(str).str.startswith("Output")])


def row_record(network, component: str, name: str) -> Dict:
    """One static row as plain JSON values, without output attributes or missing values."""
    table = getattr(network, COMPONENT_TABLE[component])
    skip = output_attributes(network, component)
    out = {}
    for key, value in table.loc[name].items():
        if key in skip:
            continue
        plain = _plain(value)
        if plain is not None:
            out[str(key)] = plain
    return out


def change_fields(network, component: str, name: str) -> Dict:
    """The fields an Appendix B row sets, as plain values (``None`` for NaN or an absent column)."""
    table = getattr(network, COMPONENT_TABLE[component])
    fields = LINK_FIELDS if component == "link" else CHANGE_FIELDS
    return {f: (_plain(table.at[name, f]) if f in table.columns else None) for f in fields}


def is_tagged(value) -> bool:
    """True for a non-empty tag (NaN and the empty string netCDF gives back are untagged)."""
    plain = _plain(value)
    return plain is not None and str(plain).strip() != ""


def _year(value) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) else None


def row_id(kind: str, operator, year, operation: str, node1, node2, occurrence: int) -> str:
    """``kind|operator|year|operation|node1|node2|occurrence``: one Appendix B row.

    ``occurrence`` counts identical (kind, operator, year, operation, terminals) rows in
    workbook order from 1 (the four MONF42--MFSN21 2027 transformers are ``...|1`` to ``...|4``).
    The terminals are the workbook's, before any documented terminal reconciliation.
    """
    year = _year(year)
    return "|".join([kind, str(operator or "").strip(), "" if year is None else str(year), operation,
                     str(node1).strip(), str(node2).strip(), str(int(occurrence))])


def assign_row_ids(upgrades_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Add ``_row_id`` and ``_source_year`` to every loaded Appendix B row (in place; idempotent)."""
    for key, kind in (("circuits", "circuit"), ("transformers", "transformer"), ("hvdc", "hvdc")):
        df = upgrades_data.get(key)
        if df is None or len(df) == 0 or "_row_id" in df.columns:
            continue
        if kind == "hvdc":
            n1 = df.get("Node 1", df.get("Node1", pd.Series("", index=df.index))).astype(str).str.strip()
            n2 = df.get("Node 2", df.get("Node2", pd.Series("", index=df.index))).astype(str).str.strip()
            year = pd.to_numeric(df.get("Planned from year", pd.Series(np.nan, index=df.index)), errors="coerce")
            operator = pd.Series("HVDC", index=df.index)
            operation = pd.Series("addition", index=df.index)
        else:
            n1 = df["Node1"].astype(str).str.strip()
            n2 = df["Node2"].astype(str).str.strip()
            year = pd.to_numeric(df["Year"], errors="coerce")
            operator = df["operator"].astype(str) if "operator" in df.columns else pd.Series("", index=df.index)
            operation = df["Status"].map(OPERATION).fillna(df["Status"].astype(str).str.lower())
        key_frame = pd.DataFrame({"o": operator.values, "y": year.values, "s": operation.values,
                                  "a": n1.values, "b": n2.values}, index=df.index)
        occurrence = key_frame.fillna(-1).groupby(["o", "y", "s", "a", "b"], sort=False).cumcount() + 1
        df["_row_id"] = [row_id(kind, o, y, s, a, b, k) for o, y, s, a, b, k in zip(
            operator, year, operation, n1, n2, occurrence)]
        df["_source_year"] = year
    return upgrades_data


def read_appendix_b_record(network) -> Optional[dict]:
    """The recorded Appendix B works, or ``None`` when the network carries none."""
    raw = (getattr(network, "meta", None) or {}).get(APPENDIX_B_META_KEY)
    if raw is None:
        return None
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def write_appendix_b_record(network, record: dict) -> None:
    network.meta = dict(getattr(network, "meta", None) or {},
                        **{APPENDIX_B_META_KEY: json.dumps(record, sort_keys=True)})
