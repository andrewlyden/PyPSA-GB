"""Build the CP30 gas fleet from named stations instead of proxy buses (item MR1/MR2).

CP30 ES1 publishes unabated gas as one national total per pathway (31,253 MW
for Further Flex and Renewables in 2030), not split by technology. The build
used to map all of it to CCGT at one class efficiency and spread it over the
twenty best-connected buses, so every CCGT had the same cost, no peaking plant
existed, and the price could never rise above one CCGT's short-run cost.

This module replaces the gas proxy rows with a declared station schedule
(``data/market/CP30/gas_station_schedule_2030.csv``, written by
``scripts/CP30/build_gas_station_schedule.py``):

* The CP30 gas total is split into CCGT, OCGT and gas engines in proportion to
  the FES spatial proxy's own capacities for the same year (done upstream in
  ``scale_proxy_to_cp30_targets``); each carrier then arrives here with its
  own target.
* Named stations fill each carrier's target first, at their DUKES capacity and
  declared connection bus. If they exceed the target, they are kept in order
  of the evidence that they run in 2030 (user decision, 15 Oct 2026):

  1. Capacity Market agreements: capacity contracted for both delivery years
     that calendar 2030 spans (2029/30 and 2030/31), then for one of them, then
     none (``cm_mw_2029_30``, ``cm_mw_2030_31``; NESO CM register). A station
     whose agreements cover only some of its units (Connah's Quay units 1 and
     3, South Humber Bank phase 2) is split into covered and uncovered parts;
  2. FES 2030 retention (``fes_2030``): capacity FES names at the station's
     own GSP or licence-area Direct row, then capacity inside FES's aggregate
     Direct(NGET) total, then capacity FES has dropped by 2030;
  3. age: the newest by commissioning year first, so the oldest retire first.

  The part at the retirement margin is kept at the residual, so the total is
  exact and every other kept station keeps its DUKES capacity. A schedule
  without the CM and FES columns (the biomass schedule) retires by age alone.
  If the named stations fall short, the FES proxy rows of that carrier fill
  the residual at their proxy locations.
* Each named CCGT carries a heat-rate index from its 2024 ELEXON-derived
  empirical marginal cost. The class efficiency sets the fleet level and the
  index sets the spread: the kept fleet's capacity-weighted harmonic mean
  efficiency equals the class value, so the fleet's average cost does not move,
  only its dispersion.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "station_name", "carrier", "capacity_mw", "commissioning_year", "bus",
    "bus_source", "heat_rate_index", "efficiency_evidence", "source",
}
GAS_CARRIERS = ("CCGT", "OCGT", "gas_engine")
DATA_SOURCE = "CP30_gas_station_schedule"
# The same named-station-plus-residual mechanism places transmission biomass
# (item MR6): Drax, Lynemouth and MGT Teesside by name, the distribution
# residual on FES GSP rows.
BIOMASS_CARRIERS = ("biomass",)
BIOMASS_DATA_SOURCE = "CP30_biomass_station_schedule"
OPTIONAL_PASS_THROUGH = ("support_type", "cfd_round")
# Retention evidence (user decision, 15 Oct 2026). Optional columns: a schedule
# without them retires by age alone. Blank fes_2030 is neutral (aggregate).
CM_COLUMNS = ("cm_mw_2029_30", "cm_mw_2030_31")
FES_RETENTION_RANK = {"named": 2, "aggregate": 1, "dropped": 0}
NEUTRAL_FES_RETENTION = "aggregate"
# Bounds on a unit efficiency after the fleet renormalisation. They keep a
# station whose empirical cost reflects something other than its heat rate
# (a CHP host's steam sales, a strategic bid) inside a physical range.
# The CCGT range is CP30's own: workbook chart CP.25 gives CCGT "Min" and "Max"
# fuel costs of 59.165 and 81.782 GBP/MWh at the Clean Power gas price in CP.22
# (100.68 p/therm = 34.354 GBP/MWh thermal), i.e. efficiencies of 0.581 and
# 0.420. CP30 publishes no OCGT or engine range; those carriers take one class
# value each (engines 0.350 from CP.25, OCGT the configured 0.38), so their
# bounds only guard against a bad input.
CP30_CCGT_EFFICIENCY_RANGE = (0.420, 0.581)
EFFICIENCY_BOUNDS = {"CCGT": CP30_CCGT_EFFICIENCY_RANGE, "OCGT": (0.30, 0.45), "gas_engine": (0.30, 0.45)}


def load_gas_station_schedule(path, carriers=GAS_CARRIERS) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station_name": str, "carrier": str, "bus": str,
                                     "support_type": str, "cfd_round": str})
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"Gas station schedule is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ("station_name", "carrier", "bus", "bus_source", "efficiency_evidence", "source"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    if frame["station_name"].duplicated().any():
        raise ValueError("Gas station schedule has duplicate station names")
    bad = sorted(set(frame["carrier"]) - set(carriers))
    if bad:
        raise ValueError(f"Station schedule carrier must be one of {tuple(carriers)}, got {bad}")
    for column in ("bus", "bus_source", "source"):
        if frame[column].eq("").any():
            raise ValueError(f"Every gas station schedule row needs a {column}")
    for column in ("capacity_mw", "commissioning_year", "heat_rate_index", "lat", "lon"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    if frame["capacity_mw"].isna().any() or (frame["capacity_mw"] <= 0).any():
        raise ValueError("Gas station schedule capacities must be positive numbers")
    if frame["commissioning_year"].isna().any():
        raise ValueError("Every gas station needs a commissioning year (retirement order)")
    if (frame["heat_rate_index"] <= 0).any():
        raise ValueError("Gas station heat_rate_index must be positive where given")
    for column in CM_COLUMNS:
        if column not in frame.columns:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if (frame[column] < 0).any():
            raise ValueError(f"Gas station schedule {column} must not be negative")
        over = frame[frame[column] > frame["capacity_mw"] + 1e-6]
        if len(over):
            raise ValueError(f"{column} exceeds the station capacity for {over['station_name'].tolist()}")
    if "fes_2030" in frame.columns:
        frame["fes_2030"] = frame["fes_2030"].fillna("").astype(str).str.strip()
        bad = sorted(set(frame["fes_2030"]) - set(FES_RETENTION_RANK) - {""})
        if bad:
            raise ValueError(f"fes_2030 must be one of {sorted(FES_RETENTION_RANK)} or blank, got {bad}")
    return frame


def retention_blocks(stations: pd.DataFrame) -> pd.DataFrame:
    """Each station split by Capacity Market coverage, in the order blocks are kept.

    One block per station and coverage level: ``cm_years`` 2 for the MW
    contracted in both 2029/30 and 2030/31, 1 for one of them, 0 for none.
    Every agreement live in 2030/31 also covers 2029/30 (the 2030/31 T-4 is not
    yet in the register), so the both-years MW is the smaller of the two
    columns. Blocks sort by coverage, then FES retention, then commissioning
    year (newest first), then name.
    """
    frame = stations.reset_index(drop=True).copy()
    capacity = frame["capacity_mw"].astype(float)
    cover = [frame[c].astype(float).clip(lower=0.0).clip(upper=capacity) if c in frame.columns
             else pd.Series(0.0, index=frame.index) for c in CM_COLUMNS]
    both = np.minimum(cover[0], cover[1])
    either = np.maximum(cover[0], cover[1])
    fes = (frame["fes_2030"].replace("", NEUTRAL_FES_RETENTION) if "fes_2030" in frame.columns
           else pd.Series(NEUTRAL_FES_RETENTION, index=frame.index))
    rank = fes.map(FES_RETENTION_RANK).astype(int)
    blocks = []
    for years, mw in ((2, both), (1, either - both), (0, capacity - either)):
        part = frame.assign(station_capacity_mw=capacity, capacity_mw=mw, cm_years=years,
                            fes_rank=rank, station_order=frame.index)
        blocks.append(part[part["capacity_mw"] > 1e-9])
    blocks = pd.concat(blocks, ignore_index=True)
    return blocks.sort_values(["cm_years", "fes_rank", "commissioning_year", "station_name"],
                              ascending=[False, False, False, True], kind="mergesort").reset_index(drop=True)


def select_stations_to_target(stations: pd.DataFrame, target_mw: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep stations up to ``target_mw`` in retention order (``retention_blocks``).

    Returns ``(kept, retired)``, one row per station in each. ``kept`` sums to
    ``min(target, named total)``. The block at the retirement margin keeps only
    the residual (``capacity_basis`` ``dukes_partial_at_retirement_margin``); a
    station whose units without a Capacity Market agreement retire whole keeps
    its covered units (``dukes_cm_covered_units``); every other kept station
    keeps its DUKES capacity (``dukes``). ``retired`` gives each station's
    retired MW and the highest coverage level it retired at
    (``retired_cm_years``), last kept first. Without the CM and FES columns
    this is retirement by age, oldest first.
    """
    blocks = retention_blocks(stations)
    kept_mw = np.zeros(len(blocks))
    running = 0.0
    for i, mw in enumerate(blocks["capacity_mw"].astype(float)):
        kept_mw[i] = min(mw, max(target_mw - running, 0.0))
        running += kept_mw[i]
    kept_mw[kept_mw <= 1e-6] = 0.0
    blocks["kept_mw"] = kept_mw
    blocks["retired_mw"] = blocks["capacity_mw"] - blocks["kept_mw"]
    blocks["marginal"] = (blocks["kept_mw"] > 0.0) & (blocks["retired_mw"] > 1e-6)

    columns = [c for c in stations.columns if c != "capacity_mw"]
    first = blocks.drop_duplicates("station_order").set_index("station_order")[columns]
    by_station = blocks.groupby("station_order")
    kept_total = by_station["kept_mw"].sum()
    retired_total = by_station["retired_mw"].sum()
    marginal = by_station["marginal"].any()

    order = list(blocks.drop_duplicates("station_order")["station_order"])
    kept_ids = [i for i in order if kept_total[i] > 0.0]
    kept = first.loc[kept_ids].copy()
    kept["capacity_mw"] = kept_total.loc[kept_ids].to_numpy()
    full = (retired_total.loc[kept_ids] <= 1e-6).to_numpy()
    kept["capacity_basis"] = np.where(
        full, "dukes",
        np.where(marginal.loc[kept_ids].to_numpy(), "dukes_partial_at_retirement_margin", "dukes_cm_covered_units"))

    retired_blocks = blocks[blocks["retired_mw"] > 1e-6]
    retired_ids = list(dict.fromkeys(retired_blocks["station_order"]))
    retired = first.loc[retired_ids].copy()
    retired["capacity_mw"] = retired_total.loc[retired_ids].to_numpy()
    retired["retired_cm_years"] = retired_blocks.groupby("station_order")["cm_years"].max().loc[retired_ids].to_numpy()
    return kept.reset_index(drop=True), retired.reset_index(drop=True)


def unit_efficiencies(kept: pd.DataFrame, class_efficiency: float, carrier: str) -> pd.Series:
    """Efficiency per kept station: class level, schedule spread.

    Stations without a heat-rate index sit at the fleet level (index 1). The
    scale factor makes the capacity-weighted harmonic mean equal the class
    efficiency, so the fleet's average fuel-and-carbon cost is unchanged.
    """
    index = kept["heat_rate_index"].fillna(1.0).astype(float)
    weights = kept["capacity_mw"].astype(float)
    low, high = EFFICIENCY_BOUNDS.get(carrier, (0.0, 1.0))
    efficiency = (class_efficiency / index).clip(low, high)
    # Scaling every efficiency by s scales the harmonic mean by s, so rescale
    # to the class value and clip; repeat because clipping moves the mean.
    for _ in range(50):
        harmonic = float(weights.sum() / (weights / efficiency).sum())
        if abs(harmonic - class_efficiency) < 1e-9:
            break
        efficiency = (efficiency * class_efficiency / harmonic).clip(low, high)
    return efficiency


def apply_gas_station_schedule(
    scaled: pd.DataFrame,
    schedule_path,
    targets: dict[str, float],
    class_efficiency: dict[str, float],
    logger: logging.Logger | None = None,
    carrier_column: str = "fuel_type",
    carriers=GAS_CARRIERS,
    data_source: str = DATA_SOURCE,
    residual_rows: str = "all",
) -> pd.DataFrame:
    """Replace proxy rows with named stations, conserving each carrier's target.

    ``scaled`` is the CP30-scaled thermal table; ``targets`` maps each carrier
    to its CP30 MW (gas after the proxy split). Proxy rows of a carrier are
    kept, rescaled, only for the residual the named stations do not cover.
    ``residual_rows="non_direct"`` keeps only GSP-level proxy rows for that
    residual, for a carrier whose transmission-connected capacity the named
    stations already represent (biomass). A carrier without a finite class
    efficiency gets no unit efficiency. ``support_type`` and ``cfd_round`` pass
    through where the schedule declares them.
    """
    logger = logger or logging.getLogger(__name__)
    schedule = load_gas_station_schedule(schedule_path, carriers)
    rest = scaled[~scaled[carrier_column].isin(carriers)]
    frames = [rest]
    for carrier in carriers:
        target_mw = float(targets.get(carrier, 0.0))
        proxy = scaled[scaled[carrier_column].eq(carrier)].copy()
        if residual_rows == "non_direct" and "gsp" in proxy.columns:
            proxy = proxy[~proxy["gsp"].astype(str).str.contains("Direct|Transmission", regex=True, na=False)]
        named = schedule[schedule["carrier"].eq(carrier)]
        if target_mw <= 1e-6:
            if len(named) or len(proxy):
                logger.info("Gas schedule: no CP30 target for %s; %d named stations unused", carrier, len(named))
            continue
        kept, retired = select_stations_to_target(named, target_mw) if len(named) else (named, named)
        named_mw = float(kept["capacity_mw"].sum()) if len(kept) else 0.0
        residual_mw = max(target_mw - named_mw, 0.0)
        efficiency = pd.Series(dtype=float)

        if len(kept):
            template = proxy.iloc[0].to_dict() if len(proxy) else {}
            level = float(class_efficiency.get(carrier, np.nan))
            efficiency = (unit_efficiencies(kept, level, carrier) if np.isfinite(level)
                          else pd.Series(np.nan, index=kept.index))
            rows = []
            for (_, row), eff in zip(kept.iterrows(), efficiency):
                record = {key: np.nan for key in template}
                record.update({
                    "technology": carrier, carrier_column: carrier,
                    "capacity_mw": float(row["capacity_mw"]), "bus": row["bus"],
                    "station_name": row["station_name"], "source_site_name": row["station_name"],
                    "lat": row.get("lat"), "lon": row.get("lon"), "gsp": np.nan,
                    "build_year": int(row["commissioning_year"]),
                    "unit_efficiency": float(eff),
                    "efficiency_source": row["efficiency_evidence"],
                    "data_source": data_source, "spatial_proxy_source": row["bus_source"],
                    "gas_capacity_basis": row["capacity_basis"],
                })
                for column in OPTIONAL_PASS_THROUGH:
                    value = row.get(column)
                    if isinstance(value, str) and value.strip():
                        record[column] = value.strip()
                for key in ("cp30_pathway", "cp30_target_mw", "cp30_source_type", "cp30_source_subtype"):
                    if key in template:
                        record[key] = template[key]
                rows.append(record)
            frames.append(pd.DataFrame(rows))

        if residual_mw > 1e-6:
            if proxy.empty:
                raise ValueError(
                    f"Named {carrier} stations cover {named_mw:,.0f} MW of a {target_mw:,.0f} MW "
                    "CP30 target and there are no proxy rows to fill the residual"
                )
            proxy["capacity_mw"] = proxy["capacity_mw"] / float(proxy["capacity_mw"].sum()) * residual_mw
            proxy["gas_capacity_basis"] = "fes_proxy_residual"
            frames.append(proxy)

        weights = kept["capacity_mw"] if len(kept) else pd.Series(dtype=float)
        harmonic = (
            float(weights.sum() / (weights / efficiency).sum())
            if len(kept) and efficiency.notna().all() else float("nan")
        )
        logger.info(
            "Station schedule %s: target %.0f MW = %d named stations %.0f MW (harmonic-mean efficiency %.4f, "
            "range %.3f-%.3f) + proxy residual %.0f MW; retired %.1f MW, last kept first: %s",
            carrier, target_mw, len(kept), named_mw, harmonic,
            float(efficiency.min()) if len(kept) else float("nan"),
            float(efficiency.max()) if len(kept) else float("nan"),
            residual_mw, float(retired["capacity_mw"].sum()) if len(retired) else 0.0,
            "; ".join(
                f"{r.station_name} {r.capacity_mw:,.1f} MW ({int(r.commissioning_year)}, CM years {int(r.retired_cm_years)})"
                for r in retired.itertuples()
            ) if len(retired) else "none",
        )
    return pd.concat(frames, ignore_index=True, sort=False)
