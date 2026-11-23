"""Committed plant that CP30 omits, added as a declared overlay.

CP30 is authoritative on capacity, but it was published in November 2024 and
some plant has since reached final investment decision and is being built for
operation before 2030. Net Zero Teesside Power (742 MW gas CCS, Dispatchable
Power Agreement signed December 2024, start-up targeted 2028) is on 2030 bills
through the CfD supplier obligation, yet CP30's Further Flex pathway has no gas
CCS in 2030. Leaving it out would charge consumers for a plant the model never
dispatches (user decision, 23 September 2026).

An overlay row is appended to the CP30-scaled thermal table at its declared
connection bus. It is refused where CP30 already gives the carrier capacity in
the modelled year, so an overlay can only add what CP30 omits, never double
count it. CP30 validation reports overlay capacity separately.

A declared carve-out (``spatial_proxy.committed_overlay_carve_out``, user decision
14 Oct 2026: capacities equal CP30's) takes an overlay carrier's capacity out of
one member carrier of a CP30 line instead of adding it on top: NZT's 742 MW of
CCS gas comes out of CCGT, inside CP30's 31,253 MW Gas line, so Gas + Gas CHP +
CCS Gas is CP30's 35,000 MW. The member carrier's target falls before the gas
station schedule fills it, so the schedule's own retention rule decides which
plant makes way (Capacity Market coverage, then FES 2030 retention, then age:
since 15 Oct 2026 the 742 MW comes off the least-evidenced kept CCGTs, which
hold one-year 2029/30 agreements only). CP30 validation then counts the overlay
carrier inside that line.

    carve_out = resolve_overlay_carve_out(spatial_proxy.get("committed_overlay_carve_out"))
    scaled = apply_overlay_carve_out(scaled, overlay_path, carve_out, targets, logger)   # before the schedule
    scaled = apply_committed_overlay(scaled, overlay_path, targets, logger, carve_out=carve_out)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"station_name", "carrier", "capacity_mw", "bus", "support_type", "contract", "source"}
SUPPORT_TYPES = {"DPA", "CfD", "merchant"}
DATA_SOURCE = "CP30_committed_overlay"


def load_committed_overlay(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"Committed overlay is missing columns: {sorted(missing)}")
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    if frame["station_name"].duplicated().any():
        raise ValueError("Committed overlay has duplicate station names")
    if frame["source"].eq("").any():
        raise ValueError("Every committed overlay row needs a source")
    bad = sorted(set(frame["support_type"]) - SUPPORT_TYPES)
    if bad:
        raise ValueError(f"Committed overlay support_type must be one of {sorted(SUPPORT_TYPES)}, got {bad}")
    frame["capacity_mw"] = pd.to_numeric(frame["capacity_mw"], errors="coerce")
    if frame["capacity_mw"].isna().any() or (frame["capacity_mw"] <= 0).any():
        raise ValueError("Committed overlay capacities must be positive numbers")
    for column in ("lat", "lon"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame


def overlay_capacity_by_carrier(path) -> dict:
    """MW the overlay adds, by carrier (for CP30 validation)."""
    if not path:
        return {}
    return load_committed_overlay(path).groupby("carrier")["capacity_mw"].sum().to_dict()


CARVE_OUT_KEYS = {"cp30_line", "from_carrier", "reason"}


def resolve_overlay_carve_out(config) -> dict:
    """``spatial_proxy.committed_overlay_carve_out`` validated: {overlay carrier: {cp30_line, from_carrier, reason}}.

    Empty (the default) adds the overlay on top of CP30, the build before the carve-out.
    """
    out = {}
    for carrier, entry in dict(config or {}).items():
        entry = dict(entry or {})
        if set(entry) != CARVE_OUT_KEYS:
            raise ValueError(f"committed_overlay_carve_out.{carrier} needs exactly {sorted(CARVE_OUT_KEYS)}, "
                             f"got {sorted(entry)}")
        if any(not str(entry[key]).strip() for key in CARVE_OUT_KEYS):
            raise ValueError(f"committed_overlay_carve_out.{carrier}: every field needs a value")
        out[str(carrier)] = {key: str(entry[key]).strip() for key in CARVE_OUT_KEYS}
    return out


def carved_line_by_carrier(carve_out: dict | None) -> dict:
    """{overlay carrier: CP30 line carrier} for CP30 validation."""
    return {carrier: entry["cp30_line"] for carrier, entry in (carve_out or {}).items()}


def carve_out_mw(overlay_path, carve_out: dict | None) -> dict:
    """MW to take out of each member carrier: {from_carrier: MW}."""
    if not overlay_path or not carve_out:
        return {}
    by_carrier = overlay_capacity_by_carrier(overlay_path)
    missing = sorted(set(carve_out) - set(by_carrier))
    if missing:
        raise ValueError(f"committed_overlay_carve_out names carriers the overlay does not add: {missing}")
    out: dict = {}
    for carrier, entry in carve_out.items():
        out[entry["from_carrier"]] = out.get(entry["from_carrier"], 0.0) + float(by_carrier[carrier])
    return out


def apply_overlay_carve_out(scaled: pd.DataFrame, overlay_path, carve_out: dict | None, targets: pd.DataFrame,
                            logger: logging.Logger | None = None,
                            carrier_column: str = "fuel_type") -> pd.DataFrame:
    """Shrink each ``from_carrier``'s CP30-scaled rows pro rata by the overlay MW carved from it.

    Call on the CP30-scaled table before any station schedule, so the schedule fills
    the reduced target. ``from_carrier`` must be a member of ``cp30_line`` in
    ``targets`` (``split_carriers``, or the line's own carrier) and keep a positive total.
    """
    logger = logger or logging.getLogger(__name__)
    if not carve_out:
        return scaled
    from scripts.CP30.generation import target_member_carriers

    lines = {str(row["carrier"]): row for _, row in targets.iterrows()} if len(targets) else {}
    for carrier, entry in carve_out.items():
        line = lines.get(entry["cp30_line"])
        if line is None or float(line["capacity_mw"]) <= 0.0:
            raise ValueError(f"committed_overlay_carve_out.{carrier}: CP30 has no {entry['cp30_line']!r} "
                             "capacity this year to carve from")
        if entry["from_carrier"] not in target_member_carriers(line):
            raise ValueError(f"committed_overlay_carve_out.{carrier}: {entry['from_carrier']!r} is not a member "
                             f"of CP30 line {entry['cp30_line']!r} ({target_member_carriers(line)})")
    scaled = scaled.copy()
    for from_carrier, mw in carve_out_mw(overlay_path, carve_out).items():
        rows = scaled[carrier_column].eq(from_carrier)
        total = float(scaled.loc[rows, "capacity_mw"].sum())
        if total - mw <= 1e-6:
            raise ValueError(f"Carve-out of {mw:,.1f} MW exceeds the {total:,.1f} MW of {from_carrier}")
        scaled.loc[rows, "capacity_mw"] = scaled.loc[rows, "capacity_mw"] * (total - mw) / total
        reasons = "; ".join(e["reason"] for e in carve_out.values() if e["from_carrier"] == from_carrier)
        logger.info("Committed overlay carve-out: %s %.1f MW -> %.1f MW (%.1f MW to the overlay; %s)",
                    from_carrier, total, total - mw, mw, reasons)
    return scaled


def apply_committed_overlay(scaled: pd.DataFrame, overlay_path, targets: pd.DataFrame,
                            logger: logging.Logger | None = None,
                            carrier_column: str = "fuel_type",
                            carve_out: dict | None = None) -> pd.DataFrame:
    """Append the overlay stations to the CP30-scaled thermal table.

    ``targets`` is the CP30 capacity table for the modelled year (``carrier``,
    ``capacity_mw``); an overlay carrier with a positive CP30 target is refused.
    """
    logger = logger or logging.getLogger(__name__)
    overlay = load_committed_overlay(overlay_path)
    cp30 = targets.groupby("carrier")["capacity_mw"].sum() if len(targets) else pd.Series(dtype=float)
    clash = sorted(c for c in overlay["carrier"].unique() if float(cp30.get(c, 0.0) or 0.0) > 0)
    if clash:
        raise ValueError(
            f"Committed overlay adds {clash}, which CP30 already gives capacity in this year; "
            "an overlay may only add plant CP30 omits")
    template = scaled.iloc[0].to_dict() if len(scaled) else {}
    rows = []
    for row in overlay.itertuples(index=False):
        record = {key: np.nan for key in template}
        record.update({
            "technology": row.carrier, carrier_column: row.carrier,
            "capacity_mw": float(row.capacity_mw), "bus": row.bus,
            "station_name": row.station_name, "source_site_name": row.station_name,
            "support_type": row.support_type, "cfd_round": row.contract or np.nan,
            "lat": row.lat, "lon": row.lon, "gsp": np.nan,
            "data_source": DATA_SOURCE, "spatial_proxy_source": row.source,
            "overlay_capacity_basis": (
                f"declared_overlay_carved_from_cp30_{carve_out[row.carrier]['from_carrier']}"
                if carve_out and row.carrier in carve_out else "declared_overlay_not_in_cp30"),
        })
        rows.append(record)
    stations = pd.DataFrame(rows)
    logger.info(
        "Committed overlay (plant CP30 omits%s): %s",
        "; carved out of " + ", ".join(sorted({e["cp30_line"] for e in carve_out.values()})) if carve_out else "",
        "; ".join(f"{r.station_name} {r.capacity_mw:,.0f} MW {r.carrier} at {r.bus} ({r.support_type} {r.contract})"
                  for r in overlay.itertuples()),
    )
    return pd.concat([scaled, stations], ignore_index=True, sort=False)
