"""Assign allocation-round labels to CP30 CfD capacity by declared schedule.

Before this module every CfD contract in the CP30 2030 runs settled at one
carrier-wide fallback strike, because no generator carried a `cfd_round`. That is the
largest single assumption behind the CfD line: 43 offshore generators holding
50.6 GW settle at a single price, when the real fleet spans AR2 at 74.75 to AR7
at 65.45 in 2012 prices, with AR4 at 37.35 in between.

The CP30 fleet cannot carry site-level vintages. It is synthetic: generators are
named `CP30_<carrier>_<n>` with `build_year` 0 and no station identity, produced
by scaling an FES spatial proxy to CP30 capacity targets. So this module does
what the improvement plan specifies for the increment - splits each carrier's
CfD capacity into round tranches using a declared, sourced capacity schedule -
and applies it to the whole CfD fleet rather than only to the increment,
because there is no separable brownfield fleet to leave alone.

Allocation is proportional and capacity-preserving. A carrier's CfD capacity is
divided between the scheduled rounds in proportion to their awarded MW, with any
capacity beyond the schedule on the modelled AR8 round (``AR8_modelled``, user
decision 27 Oct 2026, F18: plant generating in 2030 that wins its contract in
Allocation Round 8). Nothing is created or destroyed: the sum of tranche
capacities equals the input capacity.

With a brownfield fleet the schedule is shared: named projects already carry
their own round (REPD or the LCCC register), so the increment may only fill
what each round has left. ``apply_cfd_vintage_labels(named_capacity=...)`` nets
the named MW per (carrier, round) off the schedule and allocates the carrier's
increment across the remainder, the rest to the AR8 round. Without netting AR3
offshore carried 9.2 GW against 5.5 GW awarded (audit 27 Sep 2026).

A schedule row may name a contract technology (``cfd_technology``, e.g.
``floating_offshore``): the round's floating award is then a row of its own.
Named rows with that contract technology net off it, not off the carrier's main
row, and the anonymous increment never fills it (it is the carrier's main
technology). Before this (27 Oct 2026) the two AR7 floating projects (200 MW)
netted off AR7's fixed-offshore 8,245 MW, so fixed AR7 was labelled 200 MW short
and the difference sat on the residual round; Green Volt (AR6 floating, 400 MW)
did the same to AR6.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.market.cfd_strike_table import MODELLED_ROUND as _MODELLED_ROUND


REQUIRED_COLUMNS = {"carrier", "cfd_round", "capacity_mw", "source"}
#: The residual round: capacity beyond the schedule (``cfd_strike_table.MODELLED_ROUND``).
MODELLED_ROUND = _MODELLED_ROUND
#: Schedule and named-capacity column naming a row's contract technology ("" = the carrier's main one).
TECHNOLOGY_COLUMN = "cfd_technology"

#: Carriers whose 2025-26 contracts were awarded in AR7 proper (offshore pots).
#: Onshore wind, solar and tidal were auctioned separately in AR7a (results
#: 10 Feb 2026), but the LCCC portfolio and REPD label both rounds "AR7".
AR7_OFFSHORE_CARRIERS = frozenset({"wind_offshore"})


def normalise_cfd_round(cfd_round, carrier) -> object:
    """Return the strike-table round key for a recorded allocation round.

    ``AR7`` on a non-offshore carrier is the AR7a round (the strike table has
    no AR7 solar or onshore entry, so the label would silently fall back to the
    modelled AR8 strike and be counted as offshore-only AR7 capacity). Missing
    values pass through unchanged.
    """
    if cfd_round is None or (not isinstance(cfd_round, str) and pd.isna(cfd_round)):
        return cfd_round
    text = str(cfd_round).strip()
    if text == "AR7" and str(carrier).strip() not in AR7_OFFSHORE_CARRIERS:
        return "AR7a"
    return text


def normalise_cfd_rounds(frame: pd.DataFrame, carrier_column: str = "technology",
                         round_column: str = "cfd_round") -> pd.DataFrame:
    """Vectorised :func:`normalise_cfd_round` over a site table (returns a copy)."""
    if round_column not in frame.columns or carrier_column not in frame.columns:
        return frame
    result = frame.copy()
    result[round_column] = [
        normalise_cfd_round(value, carrier)
        for value, carrier in zip(result[round_column], result[carrier_column])
    ]
    return result


def named_round_capacity(sites_df: pd.DataFrame, carrier_column: str = "technology",
                         capacity_column: str = "capacity_mw") -> pd.DataFrame:
    """MW per (carrier, round, contract technology) already carried by labelled CfD rows.

    Rows on the modelled AR8 round or without a round are not netted: they are
    not part of any auctioned round the schedule describes. ``cfd_technology``
    is "" where a row records none.
    """
    columns = ["carrier", "cfd_round", TECHNOLOGY_COLUMN, "capacity_mw"]
    if sites_df.empty or "support_type" not in sites_df.columns or "cfd_round" not in sites_df.columns:
        return pd.DataFrame(columns=columns)
    rounds = sites_df["cfd_round"].fillna("").astype(str).str.strip()
    labelled = sites_df["support_type"].fillna("").astype(str).str.strip().eq("CfD") \
        & rounds.ne("") & rounds.ne(MODELLED_ROUND)
    if not labelled.any():
        return pd.DataFrame(columns=columns)
    technology = (_technology(sites_df[TECHNOLOGY_COLUMN]) if TECHNOLOGY_COLUMN in sites_df.columns
                  else pd.Series("", index=sites_df.index))
    frame = pd.DataFrame({
        "carrier": sites_df.loc[labelled, carrier_column].astype(str).str.strip(),
        "cfd_round": rounds[labelled],
        TECHNOLOGY_COLUMN: technology[labelled],
        "capacity_mw": pd.to_numeric(sites_df.loc[labelled, capacity_column], errors="coerce").fillna(0.0),
    })
    return frame.groupby(["carrier", "cfd_round", TECHNOLOGY_COLUMN], as_index=False)["capacity_mw"].sum()


def _technology(values: pd.Series) -> pd.Series:
    """A contract-technology column as stripped text, "" where blank."""
    return values.where(values.notna(), "").astype(str).str.strip().replace({"nan": "", "None": ""})


def netted_round_mw(named: pd.DataFrame | None, carrier: str, schedule: pd.DataFrame) -> dict:
    """Named MW per round that nets off a carrier's main schedule rows.

    A named row nets off the main row of its round unless the schedule gives
    its contract technology a row of its own for that round (a floating award).
    """
    if named is None or named.empty:
        return {}
    rows = named.loc[named["carrier"].astype(str).str.strip().eq(carrier)]
    if rows.empty:
        return {}
    technology = (_technology(rows[TECHNOLOGY_COLUMN]) if TECHNOLOGY_COLUMN in rows.columns
                  else pd.Series("", index=rows.index))
    own_rows = {(str(r.cfd_round).strip(), str(getattr(r, TECHNOLOGY_COLUMN)))
                for r in schedule.itertuples() if str(getattr(r, TECHNOLOGY_COLUMN))}
    keep = [(str(rnd).strip(), tech) not in own_rows for rnd, tech in zip(rows["cfd_round"], technology)]
    return rows.loc[keep].groupby("cfd_round")["capacity_mw"].sum().to_dict()


def load_vintage_schedule(path) -> pd.DataFrame:
    """Read and validate the declared allocation-round capacity schedule."""
    frame = pd.read_csv(path)
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"CfD vintage schedule is missing columns: {sorted(missing)}")
    frame["capacity_mw"] = pd.to_numeric(frame["capacity_mw"], errors="raise")
    if not np.isfinite(frame["capacity_mw"]).all() or (frame["capacity_mw"] <= 0).any():
        raise ValueError("CfD vintage schedule capacities must be finite and positive")
    if frame["source"].isna().any() or frame["source"].astype(str).str.strip().eq("").any():
        raise ValueError("Every CfD vintage schedule row needs a source")
    if TECHNOLOGY_COLUMN not in frame.columns:
        frame[TECHNOLOGY_COLUMN] = ""
    frame[TECHNOLOGY_COLUMN] = _technology(frame[TECHNOLOGY_COLUMN])
    if frame.duplicated(["carrier", "cfd_round", TECHNOLOGY_COLUMN]).any():
        raise ValueError("CfD vintage schedule has duplicate carrier/round/technology rows")
    if frame["cfd_round"].astype(str).str.strip().eq(MODELLED_ROUND).any():
        raise ValueError(
            f"The schedule must not declare the {MODELLED_ROUND!r} round; capacity beyond the "
            "scheduled rounds is assigned to it as the residual."
        )
    return frame


def main_rows(schedule: pd.DataFrame) -> pd.DataFrame:
    """The rows the anonymous increment fills: the carrier's main technology (no ``cfd_technology``)."""
    if TECHNOLOGY_COLUMN not in schedule.columns:
        return schedule
    return schedule.loc[_technology(schedule[TECHNOLOGY_COLUMN]).eq("")]


def allocate_rounds(capacity_mw: float, schedule: pd.DataFrame) -> dict:
    """Split one carrier's CfD capacity between its scheduled rounds.

    Returns round -> MW. Scheduled rounds are filled in proportion to their
    awarded capacity, scaled to fit if the model carries less CfD capacity than
    the schedule awards; any excess sits on the modelled AR8 round. Rows of a
    named contract technology (floating) are not filled.
    """
    capacity_mw = float(capacity_mw)
    if capacity_mw <= 0:
        return {}
    schedule = main_rows(schedule)
    scheduled = float(schedule["capacity_mw"].sum())
    if scheduled <= 0:
        return {MODELLED_ROUND: capacity_mw}
    # Scale down when the model carries less than the schedule awards, so a
    # tranche can never exceed the capacity actually present.
    share = min(1.0, capacity_mw / scheduled)
    allocation = {
        str(row.cfd_round).strip(): float(row.capacity_mw) * share
        for row in schedule.itertuples()
    }
    residual = capacity_mw - sum(allocation.values())
    if residual > 1e-9:
        allocation[MODELLED_ROUND] = residual
    return {name: mw for name, mw in allocation.items() if mw > 1e-9}


def allocate_carrier_rounds(capacity_mw: float, schedule: pd.DataFrame,
                            named_mw: dict | None = None) -> dict:
    """Split a carrier's unlabelled CfD capacity over what each round has left.

    ``named_mw`` (round -> MW) is the capacity other rows already hold under
    that round's label (net off the main rows: :func:`netted_round_mw`). Each
    round offers ``max(awarded - named, 0)``; the increment fills those
    remainders in proportion (scaled down if it is smaller than their sum) and
    anything beyond them sits on the modelled AR8 round. Rows of a named
    contract technology (floating) are not filled.
    Returns round -> MW summing to ``capacity_mw``.
    """
    capacity_mw = float(capacity_mw)
    if capacity_mw <= 0:
        return {}
    schedule = main_rows(schedule)
    named_mw = named_mw or {}
    remaining = {
        str(row.cfd_round).strip(): max(float(row.capacity_mw) - float(named_mw.get(str(row.cfd_round).strip(), 0.0)), 0.0)
        for row in schedule.itertuples()
    }
    open_mw = sum(remaining.values())
    if open_mw <= 1e-9:
        return {MODELLED_ROUND: capacity_mw}
    share = min(1.0, capacity_mw / open_mw)
    allocation = {name: mw * share for name, mw in remaining.items() if mw * share > 1e-9}
    residual = capacity_mw - sum(allocation.values())
    if residual > 1e-9:
        allocation[MODELLED_ROUND] = allocation.get(MODELLED_ROUND, 0.0) + residual
    return allocation


def apply_cfd_vintage_labels(sites_df: pd.DataFrame, schedule_path,
                             logger: logging.Logger | None = None,
                             named_capacity: pd.DataFrame | None = None) -> pd.DataFrame:
    """Split CfD-labelled rows into allocation-round tranches.

    Each unlabelled CfD row is divided into one row per round, preserving the
    row's total capacity and inheriting its bus, profile and every other
    attribute. Capacity and available energy are therefore conserved exactly;
    only the contract label and the row count change.

    The split is set per carrier, not per row: the carrier's unlabelled CfD
    capacity is allocated over the schedule net of ``named_capacity``
    (``carrier``, ``cfd_round``, optional ``cfd_technology``, ``capacity_mw``:
    MW other rows already hold under a round's label, from
    :func:`named_round_capacity`), and every row of that carrier takes the same
    round fractions. Capacity beyond the rounds' remainders goes to the
    modelled AR8 round.
    """
    logger = logger or logging.getLogger(__name__)
    if "support_type" not in sites_df.columns:
        return sites_df
    schedule = load_vintage_schedule(schedule_path)
    frame = sites_df.copy()
    if "cfd_round" not in frame.columns:
        frame["cfd_round"] = ""

    capacity_column = next(
        (column for column in ("capacity_mw", "p_nom") if column in frame.columns), None
    )
    if capacity_column is None:
        raise ValueError("CfD vintage labelling needs a capacity column")

    is_cfd = frame["support_type"].fillna("").astype(str).str.strip().eq("CfD")
    unlabelled = is_cfd & frame["cfd_round"].fillna("").astype(str).str.strip().eq("")
    if not unlabelled.any():
        logger.info("No unlabelled CfD capacity to assign allocation rounds to")
        return frame

    carrier_column = next(
        column for column in ("technology", "carrier", "fuel_type") if column in frame.columns
    )
    carriers = frame[carrier_column].astype(str).str.strip()
    row_capacity = pd.to_numeric(frame[capacity_column], errors="coerce").fillna(0.0)
    named = named_capacity if named_capacity is not None else pd.DataFrame(
        columns=["carrier", "cfd_round", "capacity_mw"])
    fractions = {}
    for carrier in sorted(set(carriers[unlabelled])):
        carrier_schedule = schedule.loc[schedule.carrier.eq(carrier)]
        total = float(row_capacity[unlabelled & carriers.eq(carrier)].sum())
        if carrier_schedule.empty or total <= 0:
            continue
        named_mw = netted_round_mw(named, carrier, carrier_schedule)
        allocation = allocate_carrier_rounds(total, carrier_schedule, named_mw)
        fractions[carrier] = {name: mw / total for name, mw in allocation.items()}
        for row in carrier_schedule.itertuples():
            name = str(row.cfd_round).strip()
            technology = str(getattr(row, TECHNOLOGY_COLUMN))
            if technology:
                logger.info("  CfD round %s %s %s: awarded %.0f MW, held by named rows only",
                            carrier, name, technology, float(row.capacity_mw))
                continue
            logger.info(
                "  CfD round %s %s: awarded %.0f MW, named %.0f MW, increment %.0f MW",
                carrier, name, float(row.capacity_mw), float(named_mw.get(name, 0.0)),
                allocation.get(name, 0.0),
            )
        logger.info("  CfD round %s %s: increment %.0f MW beyond the schedule",
                    carrier, MODELLED_ROUND, allocation.get(MODELLED_ROUND, 0.0))

    kept, created = [], []
    for idx, row in frame.iterrows():
        if not unlabelled.loc[idx]:
            kept.append(row)
            continue
        carrier = carriers.loc[idx]
        if carrier not in fractions:
            allocation = {MODELLED_ROUND: float(row_capacity.loc[idx])} if row_capacity.loc[idx] > 0 else {}
        else:
            allocation = {
                name: float(row_capacity.loc[idx]) * share
                for name, share in fractions[carrier].items()
                if float(row_capacity.loc[idx]) * share > 1e-9
            }
        if not allocation:
            kept.append(row)
            continue
        if len(allocation) == 1 and MODELLED_ROUND in allocation:
            # No schedule for this carrier: leave it on the modelled AR8
            # strike rather than inventing a tranche split.
            tagged = row.copy()
            tagged["cfd_round"] = MODELLED_ROUND
            kept.append(tagged)
            continue
        for round_name, capacity in allocation.items():
            tranche = row.copy()
            tranche[capacity_column] = capacity
            tranche["cfd_round"] = round_name
            for name_column in ("site_name", "station_name"):
                if name_column in tranche.index and pd.notna(tranche.get(name_column)):
                    tranche[name_column] = f"{tranche[name_column]}_{round_name}"
            created.append(tranche)

    result = pd.DataFrame(kept + created).reset_index(drop=True)
    before = float(pd.to_numeric(frame[capacity_column], errors="coerce").sum())
    after = float(pd.to_numeric(result[capacity_column], errors="coerce").sum())
    if not np.isclose(before, after, rtol=1e-9, atol=1e-6):
        raise ValueError(
            f"CfD vintage labelling changed total capacity from {before:,.3f} to {after:,.3f} MW"
        )
    labelled = result.loc[
        result["support_type"].fillna("").astype(str).str.strip().eq("CfD")
        & ~result["cfd_round"].fillna("").astype(str).str.strip().isin(["", MODELLED_ROUND])
    ]
    logger.info(
        "CfD vintage labelling: %d rows became %d; %.0f MW on a published allocation round, "
        "%.0f MW on the modelled AR8 round",
        len(frame), len(result),
        float(pd.to_numeric(labelled[capacity_column], errors="coerce").sum()),
        after - float(pd.to_numeric(labelled[capacity_column], errors="coerce").sum()),
    )
    return result


def summarise_vintage_labels(sites_df: pd.DataFrame, capacity_column="capacity_mw") -> pd.DataFrame:
    """Report labelled CfD capacity by carrier and round, for the acceptance check."""
    if "support_type" not in sites_df.columns or "cfd_round" not in sites_df.columns:
        return pd.DataFrame(columns=["carrier", "cfd_round", "capacity_mw", "generators"])
    carrier_column = next(
        column for column in ("technology", "carrier", "fuel_type") if column in sites_df.columns
    )
    cfd = sites_df.loc[sites_df["support_type"].fillna("").astype(str).str.strip().eq("CfD")]
    if cfd.empty:
        return pd.DataFrame(columns=["carrier", "cfd_round", "capacity_mw", "generators"])
    grouped = cfd.assign(
        cfd_round=cfd["cfd_round"].fillna("").astype(str).str.strip().replace("", MODELLED_ROUND)
    ).groupby([carrier_column, "cfd_round"], as_index=False).agg(
        capacity_mw=(capacity_column, "sum"), generators=(capacity_column, "size")
    )
    return grouped.rename(columns={carrier_column: "carrier"}).sort_values(
        ["carrier", "capacity_mw"], ascending=[True, False]
    ).reset_index(drop=True)
