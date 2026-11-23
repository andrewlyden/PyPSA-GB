"""Seasonal ratings and transmission licensee carried on ETYS branches.

Seasonal ratings
----------------
ETYS Appendix B circuit sheets (B-2-1a/b/c and the change sheets B-2-2a/b/c)
publish Winter, Spring, Summer and Autumn ratings in MVA; the OFTO sheets
(B-2-1d, B-2-2d) and all transformer sheets (B-3-*) publish one rating. The
build keeps ``s_nom`` as the winter rating and carries

    s_nom_winter, s_nom_spring, s_nom_summer, s_nom_autumn   (MVA, lines)

with ``seasonal_rating_source``:

* ``etys``: the Appendix B (or GB_network.xlsx AC-sheet) seasonal values;
* ``class_median_ratio``: no seasonal value was published (OFTO circuits, the
  GB_network.xlsx wind-farm/BMU edges, repair edges without one). The season
  rating is ``s_nom`` times the median season/winter ratio of the ETYS-sourced
  lines of the same bus0 voltage class (all lines if the class has none);
* ``relaxed``: 9,999 MVA couplers and other lines rated at or above
  ``RELAXED_RATING_MVA`` keep that rating in every season;
* ``winter_only``: transformers, for which Appendix B gives a single rating;
  every season uses it.

Transmission licensee
---------------------
``etys_licensee`` is the ETYS licensee of the source sheet: SHE (``a``
sheets), SPT (``b``), NGET (``c``) and OFTO (``d``). Branches from other
sources (GB_network.xlsx edges, repair edges and couplers) take the licensee
most common among ETYS branches at their substations (4-character site code);
failing that, the side of the Scotland--England border their buses lie on
(``SCOT`` / ``EW``). ``etys_licensee_source`` records which rule applied.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

SEASONS = ("winter", "spring", "summer", "autumn")
SEASON_COLUMNS = {"spring": "s_nom_spring", "summer": "s_nom_summer", "autumn": "s_nom_autumn"}
ETYS_SEASON_FIELDS = {
    "spring": "Spring Rating (MVA)",
    "summer": "Summer Rating (MVA)",
    "autumn": "Autumn Rating (MVA)",
}
RELAXED_RATING_MVA = 9999.0
SCOTTISH_LICENSEES = {"SHE", "SPT", "SCOT"}

# ETYS licensee by Appendix B sheet suffix
SHEET_LICENSEE = {"a": "SHE", "b": "SPT", "c": "NGET", "d": "OFTO"}

# Approximate Scotland--England border in OSGB36 (x, y in m): the Solway coast
# (Galloway north of Cumbria), up the Sark/Liddel past Gretna (Scotland) and
# Longtown/Carlisle (England), Carter Bar, the Tweed to north of Berwick
# (England). A fallback only, for branches whose substations have no ETYS
# licensee and for networks built without the licensee column.
_BORDER_X = [150000.0, 290000.0, 318000.0, 333000.0, 334000.0, 350000.0,
             369000.0, 385000.0, 395000.0, 450000.0]
_BORDER_Y = [520000.0, 528000.0, 560000.0, 562000.0, 570000.0, 582000.0,
             607000.0, 640000.0, 658000.0, 680000.0]


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) and v > 0 else np.nan


def seasonal_ratings_from_record(record) -> Dict[str, float]:
    """``{s_nom_spring, s_nom_summer, s_nom_autumn}`` from an Appendix B row (NaN if absent)."""
    get = record.get if hasattr(record, "get") else (lambda k, d=None: d)
    return {SEASON_COLUMNS[s]: _num(get(field)) for s, field in ETYS_SEASON_FIELDS.items()}


def scotland_by_geography(buses: pd.DataFrame) -> pd.Series:
    """True for buses north of the approximate border (OSGB36 x/y)."""
    x = pd.to_numeric(buses.get("x"), errors="coerce")
    y = pd.to_numeric(buses.get("y"), errors="coerce")
    border = np.interp(x.fillna(0.0), _BORDER_X, _BORDER_Y)
    return pd.Series((y > border) & x.notna() & y.notna(), index=buses.index)


def _fill_seasonal(network, logger) -> pd.DataFrame:
    lines = network.lines
    v0 = lines["bus0"].map(network.buses["v_nom"]).astype(float)
    s_nom = lines["s_nom"].astype(float)
    for col in SEASON_COLUMNS.values():
        if col not in lines.columns:
            lines[col] = np.nan
        lines[col] = pd.to_numeric(lines[col], errors="coerce")
    has_all = lines[list(SEASON_COLUMNS.values())].notna().all(axis=1) & (s_nom > 0)
    relaxed = s_nom >= RELAXED_RATING_MVA
    source = pd.Series("class_median_ratio", index=lines.index, dtype=object)
    source[has_all] = "etys"
    source[relaxed] = "relaxed"
    ref = has_all & ~relaxed
    summary = {}
    for season, col in SEASON_COLUMNS.items():
        ratio = (lines.loc[ref, col] / s_nom[ref]).clip(upper=1.5)
        by_class = ratio.groupby(v0[ref]).median()
        overall = float(ratio.median()) if len(ratio) else 1.0
        fill = v0.map(by_class).fillna(overall)
        missing = lines[col].isna() & ~relaxed
        lines.loc[missing, col] = s_nom[missing] * fill[missing]
        lines.loc[relaxed, col] = s_nom[relaxed]
        summary[season] = by_class.round(3).to_dict()
    lines["seasonal_rating_source"] = source
    lines["s_nom_winter"] = s_nom   # build rating; solve-time ratios are season / winter
    logger.info(
        "Seasonal line ratings: %s; median season/winter ratio by bus0 kV %s",
        source.value_counts().to_dict(), summary,
    )
    trafos = network.transformers
    if len(trafos):
        for col in list(SEASON_COLUMNS.values()) + ["s_nom_winter"]:
            trafos[col] = trafos["s_nom"].astype(float)
        trafos["seasonal_rating_source"] = "winter_only"
    return lines


def _fill_licensee(network, logger) -> None:
    tables = [t for t in (network.lines, network.transformers) if len(t)]
    for table in tables:
        if "etys_licensee" not in table.columns:
            table["etys_licensee"] = np.nan
        table["etys_licensee"] = table["etys_licensee"].replace({"": np.nan, "nan": np.nan})
        table["etys_licensee_source"] = np.where(table["etys_licensee"].notna(), "etys_sheet", None)
    # site code -> most common ETYS licensee among branches touching it
    votes = []
    for table in tables:
        known = table[table["etys_licensee"].notna()]
        for end in ("bus0", "bus1"):
            votes.append(pd.DataFrame({"site": known[end].astype(str).str[:4],
                                       "lic": known["etys_licensee"]}))
    site_lic = {}
    if votes:
        v = pd.concat(votes)
        v = v[v.lic != "OFTO"] if (v.lic != "OFTO").any() else v
        site_lic = v.groupby("site")["lic"].agg(lambda s: s.value_counts().index[0]).to_dict()
    geo_scot = scotland_by_geography(network.buses)
    for table in tables:
        missing = table["etys_licensee"].isna()
        for name in table.index[missing]:
            ends = [str(table.at[name, "bus0"]), str(table.at[name, "bus1"])]
            lic = [site_lic.get(e[:4]) for e in ends if site_lic.get(e[:4])]
            if lic:
                table.at[name, "etys_licensee"] = lic[0]
                table.at[name, "etys_licensee_source"] = "site_majority"
            else:
                scot = all(bool(geo_scot.get(e, False)) for e in ends)
                table.at[name, "etys_licensee"] = "SCOT" if scot else "EW"
                table.at[name, "etys_licensee_source"] = "geography"
    for label, table in (("lines", network.lines), ("transformers", network.transformers)):
        if len(table):
            logger.info("Branch licensee (%s): %s by %s", label,
                        table["etys_licensee"].value_counts().to_dict(),
                        table["etys_licensee_source"].value_counts().to_dict())


def finalize_branch_attributes(network, logger: Optional[logging.Logger] = None):
    """Complete seasonal ratings and licensee on every line and transformer."""
    logger = logger or logging.getLogger(__name__)
    if len(network.lines):
        _fill_seasonal(network, logger)
    elif len(network.transformers):
        for col in list(SEASON_COLUMNS.values()) + ["s_nom_winter"]:
            network.transformers[col] = network.transformers["s_nom"].astype(float)
        network.transformers["seasonal_rating_source"] = "winter_only"
    _fill_licensee(network, logger)
    return network


def complete_branch_attributes_for(network, lines=(), transformers=(),
                                   logger: Optional[logging.Logger] = None) -> None:
    """Complete seasonal ratings and licensee source on named branches of a finished network.

    For branches put back after the build (the network-delivery revert of the
    CP30-dated works): a season missing on a line takes ``s_nom`` times the median
    season/winter ratio of the ``etys``-sourced lines of its bus0 voltage class, as
    :func:`finalize_branch_attributes` does at the build; ``s_nom_winter`` is the
    line's ``s_nom``; transformers use their one rating in every season. The other
    branches are untouched (the build's ratios are recomputed from them).
    """
    logger = logger or logging.getLogger(__name__)
    lines = [n for n in lines if n in network.lines.index]
    if lines:
        table = network.lines
        for col in list(SEASON_COLUMNS.values()) + ["s_nom_winter"]:
            if col not in table.columns:
                table[col] = np.nan
        if "seasonal_rating_source" not in table.columns:
            table["seasonal_rating_source"] = pd.Series(np.nan, index=table.index, dtype=object)
        v0 = table["bus0"].map(network.buses["v_nom"]).astype(float)
        s_nom = table["s_nom"].astype(float)
        source = table["seasonal_rating_source"].astype(object)
        others = ~table.index.isin(lines)
        ref = others & source.eq("etys") & (s_nom > 0)
        target = pd.Index(lines)
        had_all = table.loc[target, list(SEASON_COLUMNS.values())].notna().all(axis=1)
        relaxed = s_nom[target] >= RELAXED_RATING_MVA
        for col in SEASON_COLUMNS.values():
            ratio = (table.loc[ref, col].astype(float) / s_nom[ref]).clip(upper=1.5)
            by_class = ratio.groupby(v0[ref]).median()
            overall = float(ratio.median()) if len(ratio) else 1.0
            fill = v0[target].map(by_class).fillna(overall)
            missing = table.loc[target, col].isna() & ~relaxed
            table.loc[target[missing.to_numpy()], col] = (s_nom[target] * fill)[missing.to_numpy()]
            table.loc[target[relaxed.to_numpy()], col] = s_nom[target][relaxed.to_numpy()]
        table.loc[target, "s_nom_winter"] = s_nom[target]
        table.loc[target, "seasonal_rating_source"] = np.where(
            relaxed, "relaxed", np.where(had_all, "etys", "class_median_ratio"))
    trafos = [n for n in transformers if n in network.transformers.index]
    if trafos:
        table = network.transformers
        for col in list(SEASON_COLUMNS.values()) + ["s_nom_winter"]:
            if col in table.columns:
                table.loc[trafos, col] = table.loc[trafos, "s_nom"].astype(float)
        if "seasonal_rating_source" in table.columns:
            table.loc[trafos, "seasonal_rating_source"] = "winter_only"
    for name_list, table in ((lines, network.lines), (trafos, network.transformers)):
        if name_list and "etys_licensee_source" in table.columns and "etys_licensee" in table.columns:
            known = table.loc[name_list, "etys_licensee"].astype(str).str.strip().isin(["", "nan"])
            source = table.loc[name_list, "etys_licensee_source"].astype(object)
            fill = source.isna() | source.astype(str).str.strip().isin(["", "nan"])
            table.loc[source.index[fill & ~known], "etys_licensee_source"] = "etys_sheet"
    if lines or trafos:
        logger.info("Branch attributes completed on %d restored lines and %d transformers",
                    len(lines), len(trafos))


def is_scottish_branch(network, table: pd.DataFrame) -> pd.Series:
    """True for branches in the SHE/SPT transmission areas.

    Uses ``etys_licensee`` (SHE, SPT, or SCOT for inferred Scottish branches);
    a network without that column falls back on bus geography (both ends north
    of the border).
    """
    if "etys_licensee" in table.columns and table["etys_licensee"].notna().any():
        lic = table["etys_licensee"].astype(str)
        known = table["etys_licensee"].notna()
        out = lic.isin(SCOTTISH_LICENSEES)
        if (~known).any():
            geo = scotland_by_geography(network.buses)
            out[~known] = table.loc[~known, "bus0"].map(geo).fillna(False) & \
                table.loc[~known, "bus1"].map(geo).fillna(False)
        return out.astype(bool)
    geo = scotland_by_geography(network.buses)
    return (table["bus0"].map(geo).fillna(False) & table["bus1"].map(geo).fillna(False)).astype(bool)


# Rating season of each calendar month (config network.transmission.seasonal_ratings
# .season_months overrides it). Appendix B names the four seasons but does not
# define their months. Default: the ER P27 / ACE 104 overhead-line rating
# seasons as tabulated in Northern Powergrid IMP/001/011 v4.0 (Dec 2017)
# Table 1: Winter Nov-Feb, Spring Mar-Apr, Summer May-Aug, Autumn Sep-Oct.
DEFAULT_SEASON_MONTHS = {
    "winter": [11, 12, 1, 2],
    "spring": [3, 4],
    "summer": [5, 6, 7, 8],
    "autumn": [9, 10],
}


def month_to_season(season_months=None) -> Dict[int, str]:
    season_months = season_months or DEFAULT_SEASON_MONTHS
    out = {}
    for season, months in season_months.items():
        if season not in SEASONS:
            raise ValueError(f"Unknown rating season {season!r}; expected one of {SEASONS}")
        for m in months:
            m = int(m)
            if m in out:
                raise ValueError(f"Month {m} is assigned to two rating seasons")
            out[m] = season
    missing = sorted(set(range(1, 13)) - set(out))
    if missing:
        raise ValueError(f"Rating seasons do not cover months {missing}")
    return out


def seasonal_rating_factors(network, snapshots, season_months=None,
                            relaxed_rating: float = RELAXED_RATING_MVA) -> pd.DataFrame:
    """Snapshot x line factor ``s_nom_<season> / s_nom_winter`` (1 where relaxed).

    Lines whose current ``s_nom`` is at or above ``relaxed_rating`` (9,999 MVA
    couplers, the BM voltage floor) and lines without seasonal columns get 1.
    Only lines with a factor different from 1 in some season are returned.
    """
    lines = network.lines
    cols = ["s_nom_winter"] + list(SEASON_COLUMNS.values())
    if not len(lines) or any(c not in lines.columns for c in cols):
        return pd.DataFrame(index=snapshots)
    winter = pd.to_numeric(lines["s_nom_winter"], errors="coerce")
    ratios = {"winter": pd.Series(1.0, index=lines.index)}
    for season, col in SEASON_COLUMNS.items():
        ratios[season] = (pd.to_numeric(lines[col], errors="coerce") / winter).where(winter > 0)
    ratios = pd.DataFrame(ratios).fillna(1.0)
    ratios.loc[lines["s_nom"].astype(float) >= relaxed_rating] = 1.0
    varying = ratios.index[(ratios.sub(1.0).abs() > 1e-9).any(axis=1)]
    season_of = pd.DatetimeIndex(snapshots).month.map(month_to_season(season_months))
    frame = ratios.loc[varying].T.reindex(list(season_of))
    frame.index = snapshots
    return frame
