"""Place CP30 nuclear capacity at named stations instead of proxy buses.

CP30 publishes nuclear as one national total per pathway (3,515 MW for Further
Flex and Renewables in 2030) and the thermal integration spreads it across the
twenty best-connected buses in each transmission region as
``FES_nuclear_<region>_<bus>`` proxies. Nothing in that fleet can be Hinkley
Point C, so nothing carries its CfD, and the bill has no nuclear difference
payment at all.

The TEC register dates HPC unit 1 at 31 March 2029 and unit 2 at 31 July 2030,
and CP30's total is Sizewell B plus HPC unit 1 plus unit 2 prorated from
August to within eight megawatts. A declared station schedule makes that
inference explicit: fixed stations take their DUKES capacity, one station may
take the residual of the CP30 total, and each carries its own support label
and connection bus. Capacity is conserved to the ES1 total exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"station_name", "bus", "capacity_mw", "support_type", "cfd_round", "source"}
SUPPORT_TYPES = {"CfD", "ROC", "merchant"}
DATA_SOURCE = "CP30_nuclear_station_schedule"
# The same declared-station mechanism places CP30's CCS biomass (BECCS) at Drax
# (item MR7): CP30 converts a Drax biomass unit, so the capacity is one named
# station, transmission-connected, holding a Power BECCS CfD.
CCS_BIOMASS_DATA_SOURCE = "CP30_ccs_biomass_station_schedule"


def load_nuclear_station_schedule(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"Nuclear station schedule is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ("station_name", "bus", "support_type", "cfd_round", "source"):
        frame[column] = frame[column].astype(str).str.strip()
    if frame["station_name"].duplicated().any():
        raise ValueError("Nuclear station schedule has duplicate station names")
    if frame["source"].eq("").any():
        raise ValueError("Every nuclear station schedule row needs a source")
    bad = sorted(set(frame["support_type"]) - SUPPORT_TYPES)
    if bad:
        raise ValueError(f"Nuclear station schedule support_type must be one of {sorted(SUPPORT_TYPES)}, got {bad}")
    is_residual = frame["capacity_mw"].str.strip().str.lower().eq("residual")
    if is_residual.sum() > 1:
        raise ValueError("At most one nuclear station may take the residual of the CP30 total")
    fixed = pd.to_numeric(frame.loc[~is_residual, "capacity_mw"], errors="coerce")
    if fixed.isna().any() or (fixed <= 0).any():
        raise ValueError("Nuclear station schedule capacities must be positive numbers or 'residual'")
    frame["is_residual"] = is_residual
    frame["capacity_mw"] = pd.to_numeric(frame["capacity_mw"], errors="coerce")
    for column in ("lat", "lon"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame


def apply_nuclear_station_schedule(scaled: pd.DataFrame, schedule_path, target_mw: float,
                                   logger: logging.Logger | None = None,
                                   carrier_column: str = "fuel_type",
                                   carrier: str = "nuclear",
                                   data_source: str = DATA_SOURCE) -> pd.DataFrame:
    """Replace proxy rows of ``carrier`` with the declared stations, conserving the total.

    ``scaled`` is the CP30-scaled thermal table; its ``carrier`` rows are dropped
    and the schedule rows appended with the columns the thermal integration
    expects. The rows carry no ``gsp``, so the later "Direct" spatial
    distribution leaves them at their declared bus. Nuclear is the default; the
    CP30 CCS biomass schedule uses the same mechanism.
    """
    logger = logger or logging.getLogger(__name__)
    schedule = load_nuclear_station_schedule(schedule_path)
    target_mw = float(target_mw)
    fixed_mw = float(schedule.loc[~schedule["is_residual"], "capacity_mw"].sum())
    residual_mw = target_mw - fixed_mw
    if schedule["is_residual"].any():
        if residual_mw < -1e-6:
            raise ValueError(
                f"Fixed {carrier} stations ({fixed_mw:,.0f} MW) exceed the CP30 {carrier} total "
                f"({target_mw:,.0f} MW); the residual station would be negative"
            )
        schedule.loc[schedule["is_residual"], "capacity_mw"] = max(residual_mw, 0.0)
    elif not np.isclose(fixed_mw, target_mw, rtol=0, atol=1.0):
        raise ValueError(
            f"{carrier} station schedule totals {fixed_mw:,.0f} MW but CP30 gives {target_mw:,.0f} MW "
            "and no station takes the residual"
        )
    schedule = schedule[schedule["capacity_mw"] > 1e-6].copy()

    proxy = scaled[scaled[carrier_column].eq(carrier)]
    template = proxy.iloc[0].to_dict() if len(proxy) else {}
    rows = []
    for row in schedule.itertuples(index=False):
        record = {key: np.nan for key in template}
        record.update({
            "technology": carrier, carrier_column: carrier,
            "capacity_mw": float(row.capacity_mw), "bus": row.bus,
            "station_name": row.station_name, "source_site_name": row.station_name,
            "support_type": row.support_type, "cfd_round": row.cfd_round or np.nan,
            "lat": row.lat, "lon": row.lon, "gsp": np.nan,
            "data_source": data_source, "spatial_proxy_source": row.source,
            f"{carrier.replace(' ', '_').lower()}_capacity_basis": "cp30_residual" if row.is_residual else "declared",
        })
        for key in ("cp30_pathway", "cp30_target_mw", "cp30_source_type", "cp30_source_subtype"):
            if key in template:
                record[key] = template[key]
        rows.append(record)
    stations = pd.DataFrame(rows)
    result = pd.concat([scaled[~scaled[carrier_column].eq(carrier)], stations], ignore_index=True, sort=False)
    logger.info(
        "%s station schedule: replaced %d proxy rows (%.0f MW) with %d stations (%.0f MW): %s",
        carrier, len(proxy), float(proxy["capacity_mw"].sum()) if len(proxy) else 0.0, len(stations),
        float(stations["capacity_mw"].sum()),
        "; ".join(f"{r.station_name} {r.capacity_mw:,.0f} MW at {r.bus} ({r.support_type}"
                  f"{', ' + r.cfd_round if r.cfd_round else ''})" for r in schedule.itertuples()),
    )
    return result
