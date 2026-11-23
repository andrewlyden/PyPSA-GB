#!/usr/bin/env python3
"""Simultaneous charge and discharge in the balancing stage, by storage carrier and month.

The gate week of the 2050 analysis (plan Y7, finding 4) found batteries charging and
discharging at once in the BM: 13.6-15.6 GWh a week before the seasonal pass, 5.6 with it,
1.2 on the final gate. That is the zero-price degeneracy of the 2030 work recurring (the BM
replaced objective with only a net anchor let the gross split float, see
``scripts/market/diagnose_storage_degeneracy.py``); the 18 September 2026 fix anchors both
gross legs in ``solve_balancing.py``. This script measures what is left on a solved year.

Two numbers per unit, summed by carrier:

* **simultaneous energy** ``sum_t min(p_dispatch, p_store) x hours``: the energy both
  charged and discharged in the same hour. This is the gate-week figure (Y7 ``bm_summary``).
* **energy lost** ``simultaneous x (1/eta_dispatch - eta_store)``: what the pair burns
  through the round trip (``diagnose_storage_degeneracy.exact_dissipation``). Consumer
  flexibility and DACCS have unit efficiencies, so their pairs burn nothing.

Also reported: unit-hours with a pair (both legs above 1 kW), charge and discharge, the
pair's share of charging, and, for the window given by ``--gate-window``, the same figures
for the gate week so the year compares like for like.

    python scripts/market/bm_storage_churn.py --study FES25_HT_2050 --gate-year 2050 \
        --output-csv resources/analysis/FES25_HT_2050_bm_storage_churn.csv

The study is required (no default year); the gate week is :func:`gate_window` of the
model year given, or none.
"""


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The repository root: the working directory under Snakemake (which runs a copy of this file),
# this file's grandparent when run as `python scripts/...`.
for _root in (Path.cwd(), Path(__file__).resolve().parents[2]):
    if (_root / "scripts").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.market.diagnose_storage_degeneracy import (  # noqa: E402
    SIMULTANEITY_TOLERANCE_MW,
    read_storage,
)

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def gate_window(year: int) -> tuple[str, str]:
    """The gate week of a model year (plan Y7: 10 to 16 January), the rule's ``window`` param."""
    return (f"{int(year)}-01-10 00:00", f"{int(year)}-01-16 23:00")


def churn_frame(p_dispatch: pd.DataFrame, p_store: pd.DataFrame, hours: pd.Series,
                attributes: pd.DataFrame, tolerance_mw: float = SIMULTANEITY_TOLERANCE_MW,
                bus_price: np.ndarray | None = None) -> pd.DataFrame:
    """Per-unit simultaneous energy, energy lost, pair unit-hours, charge and discharge (MWh).

    ``bus_price`` (snapshots x units, the BM nodal price at each unit's bus) splits the
    simultaneous energy into hours where the unit's node prices below zero, where burning
    energy through the round trip has value to the BM (a dump load), and the rest.
    """
    w = hours.to_numpy()[:, None]
    dispatch, store = p_dispatch.to_numpy(), p_store.to_numpy()
    overlap = np.minimum(dispatch, store).clip(min=0.0) * w
    negative = ((overlap * (bus_price < 0.0)).sum(axis=0) if bus_price is not None
                else np.full(overlap.shape[1], np.nan))
    loss_rate = (1.0 / attributes["efficiency_dispatch"].clip(lower=1e-6) - attributes["efficiency_store"]).to_numpy()
    pairs = ((dispatch > tolerance_mw) & (store > tolerance_mw)).sum(axis=0)
    return pd.DataFrame({
        "carrier": attributes["carrier"].astype(str).to_numpy(),
        "simultaneous_mwh": overlap.sum(axis=0),
        "simultaneous_negative_price_mwh": negative,
        "energy_lost_mwh": (overlap * loss_rate[None, :]).sum(axis=0),
        "pair_unit_hours": pairs,
        "charge_mwh": (store.clip(min=0.0) * w).sum(axis=0),
        "discharge_mwh": (dispatch.clip(min=0.0) * w).sum(axis=0),
        "units_with_pairs": (pairs > 0).astype(int),
    }, index=p_dispatch.columns)


def by_carrier(units: pd.DataFrame) -> pd.DataFrame:
    out = units.groupby("carrier").agg(
        units=("charge_mwh", "size"), units_with_pairs=("units_with_pairs", "sum"),
        simultaneous_mwh=("simultaneous_mwh", "sum"), energy_lost_mwh=("energy_lost_mwh", "sum"),
        simultaneous_negative_price_mwh=("simultaneous_negative_price_mwh", "sum"),
        pair_unit_hours=("pair_unit_hours", "sum"), charge_mwh=("charge_mwh", "sum"),
        discharge_mwh=("discharge_mwh", "sum"))
    out["simultaneous_share_of_charge"] = out["simultaneous_mwh"] / out["charge_mwh"].replace(0.0, np.nan)
    return out.reset_index()


def _unit_bus_price(path: str | Path, units: pd.Index) -> np.ndarray | None:
    """BM nodal price at each unit's bus (snapshots x units); None if the file has none."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if "buses_t_marginal_price" not in ds:
            return None
        bus = pd.Series(ds["storage_units_bus"].values.astype(str), index=ds["storage_units_i"].values.astype(str))
        price = pd.DataFrame(ds["buses_t_marginal_price"].values,
                             columns=ds["buses_t_marginal_price_i"].values.astype(str))
    return price.reindex(columns=bus.reindex(units).to_numpy()).fillna(0.0).to_numpy()


def measure_month(path: str | Path, month: str, window: tuple[str, str] | None = None,
                  top: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One balancing month: the carrier table, the window's (empty if outside it), and the
    ``top`` units by simultaneous energy."""
    _, hours, attributes, gross = read_storage(path)
    if gross is None:
        raise ValueError(f"{path} carries no p_dispatch / p_store split; churn cannot be measured exactly")
    p_dispatch, p_store = gross
    price = _unit_bus_price(path, p_dispatch.columns)
    units = churn_frame(p_dispatch, p_store, hours, attributes, bus_price=price)
    month_table = by_carrier(units)
    month_table.insert(0, "month", month)
    top_units = units.nlargest(top, "simultaneous_mwh")[["carrier", "simultaneous_mwh",
                                                          "simultaneous_negative_price_mwh"]].assign(month=month)
    window_table = pd.DataFrame()
    if window is not None:
        mask = (p_dispatch.index >= pd.Timestamp(window[0])) & (p_dispatch.index <= pd.Timestamp(window[1]))
        if mask.any():
            window_table = by_carrier(churn_frame(p_dispatch[mask], p_store[mask], hours[mask], attributes,
                                                  bus_price=price[mask] if price is not None else None))
            window_table.insert(0, "month", f"window {window[0][:10]} to {window[1][:10]}")
    return month_table, window_table, top_units


def run(networks: list[str], months: list[str], output_csv: str, output_json: str | None = None,
        window: tuple[str, str] | None = None, logger=None) -> pd.DataFrame:
    import logging

    logger = logger or logging.getLogger("bm_storage_churn")
    tables, windows, tops = [], [], []
    for path, month in zip(networks, months):
        table, win, top_units = measure_month(path, month, window)
        tables.append(table)
        tops.append(top_units)
        if not win.empty:
            windows.append(win)
        logger.info("%s: simultaneous %.2f GWh, lost %.2f GWh", month, table["simultaneous_mwh"].sum() / 1e3,
                    table["energy_lost_mwh"].sum() / 1e3)
    monthly = pd.concat(tables, ignore_index=True)
    year = by_carrier_sum(monthly)
    year.insert(0, "month", "Year")
    out = pd.concat([monthly, year, *windows], ignore_index=True)
    for col in ("simultaneous", "simultaneous_negative_price", "energy_lost", "charge", "discharge"):
        out[f"{col}_gwh"] = out.pop(f"{col}_mwh") / 1e3
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    if output_json:
        weeks = sum(len(read_storage(p)[1]) for p in networks) / 168.0
        summary = {
            "months": list(months), "weeks": weeks,
            "gate_week_reference_gwh": {"battery_y7_before_seasonal_pass": [13.6, 15.6],
                                        "battery_y8a_seasonal": 5.6, "battery_final_gate": 1.2},
            "year_by_carrier": year.assign(simultaneous_gwh=year["simultaneous_mwh"] / 1e3,
                                           energy_lost_gwh=year["energy_lost_mwh"] / 1e3,
                                           per_week_simultaneous_gwh=year["simultaneous_mwh"] / 1e3 / weeks)
            [["carrier", "simultaneous_gwh", "energy_lost_gwh", "per_week_simultaneous_gwh", "pair_unit_hours",
              "simultaneous_share_of_charge"]].to_dict("records"),
            "top_units_year_gwh": (pd.concat(tops).reset_index(names="unit").groupby(["unit", "carrier"])
                                   [["simultaneous_mwh", "simultaneous_negative_price_mwh"]].sum().div(1e3)
                                   .sort_values("simultaneous_mwh", ascending=False).head(10)
                                   .reset_index().to_dict("records")),
            "simultaneous_negative_price_share": (float(year["simultaneous_negative_price_mwh"].sum()
                                                        / year["simultaneous_mwh"].sum())
                                                  if year["simultaneous_mwh"].sum() else None),
            "window": ({"start": window[0], "end": window[1],
                        "by_carrier": windows[0].assign(simultaneous_gwh=windows[0]["simultaneous_mwh"] / 1e3,
                                                        energy_lost_gwh=windows[0]["energy_lost_mwh"] / 1e3)
                        [["carrier", "simultaneous_gwh", "energy_lost_gwh", "pair_unit_hours"]].to_dict("records")}
                       if windows else None),
        }
        Path(output_json).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return out


def by_carrier_sum(monthly: pd.DataFrame) -> pd.DataFrame:
    year = monthly.groupby("carrier").agg(
        units=("units", "max"), units_with_pairs=("units_with_pairs", "max"),
        simultaneous_mwh=("simultaneous_mwh", "sum"), energy_lost_mwh=("energy_lost_mwh", "sum"),
        simultaneous_negative_price_mwh=("simultaneous_negative_price_mwh", "sum"),
        pair_unit_hours=("pair_unit_hours", "sum"), charge_mwh=("charge_mwh", "sum"),
        discharge_mwh=("discharge_mwh", "sum")).reset_index()
    year["simultaneous_share_of_charge"] = year["simultaneous_mwh"] / year["charge_mwh"].replace(0.0, np.nan)
    return year


if "snakemake" in globals():
    from scripts.utilities.logging_config import setup_logging

    run(list(snakemake.input.networks), list(snakemake.params.months), snakemake.output.csv,
        snakemake.output.json, tuple(snakemake.params.window) if snakemake.params.window else None,
        setup_logging(snakemake.log[0] if snakemake.log else "bm_storage_churn"))
elif __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, help="the twelve-month study id, e.g. FES25_HT_2050")
    ap.add_argument("--family", default="market", help="the month family: months are <study>_<Mon>_<family>")
    ap.add_argument("--gate-year", type=int, default=None, help="the model year whose gate week is reported")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", default=None)
    a = ap.parse_args()
    nets = [f"resources/market/{a.study}_{m}_{a.family}_balancing.nc" for m in MONTHS]
    run(nets, list(MONTHS), a.output_csv, a.output_json or str(Path(a.output_csv).with_suffix(".json")),
        gate_window(a.gate_year) if a.gate_year else None)
