"""Separate storage round-trip loss from zero-price solver degeneracy (item B2).

**Why this exists.** A solved CP30 year charges 13.73 TWh of storage and
discharges 8.93 TWh. Read as a round trip that is 65%, against a "nominal 85%",
and it looks like the zero-price degeneracy the storage dispatch cost floor was
meant to remove: a barrier solve without crossover returning an interior point
that charges and discharges the same unit at once, burning energy through the
round-trip losses on both legs. That reading is wrong twice over, and this
script is the arithmetic that shows why.

**The 85% is a battery-only figure.** The fleet is two carriers. Grid batteries
carry ``efficiency_store`` = ``efficiency_dispatch`` = 0.92, a nominal round trip
of 84.6%. The FES "other storage" block, modelled as LAES, carries 0.60 on each
leg, a nominal round trip of 36%, and it takes 5.27 TWh of the charging. The
capacity-weighted nominal round trip of the fleet is 66%, not 85%, so almost all
of the apparent gap is the efficiency the model was given rather than energy it
lost to the solver.

**Net dispatch cannot see simultaneity at all.** The saved networks carry only
``storage_units_t.p`` = ``p_dispatch - p_store``. A simultaneous pair cancels in
that difference, so charging and discharging totals are unchanged by it; what
changes is the stored energy, which is not saved. Reconstructing the state of
charge from net p therefore *overstates* it by exactly the dissipated energy,
and because true SoC is bounded above by ``p_nom * max_hours``,

    dissipation >= max_t (SoC_reconstructed(t) - Emax)

is a strict per-unit lower bound. That is the honest test, and it is what the
acceptance criterion "no unit's implied state of charge exceeds its capacity"
was reaching for.

**Carry-over.** The twelve months are independent solves, each starting from the
configured ``state_of_charge_initial``, so part of any gap is a month ending
fuller or emptier than it began. The script reports it separately. It is small
by construction: the whole fleet holds 148 GWh, so twelve months of carry-over
cannot account for TWh.

**Exact mode.** The balancing solver now saves ``p_dispatch`` and ``p_store``
alongside net ``p``. When both are present the dissipation is computed exactly
as ``sum(min(p_dispatch, p_store) * (1/eta_dispatch - eta_store) * hours)`` and
the number of unit-hours with a simultaneous pair is counted; the lower bound
above is still reported for comparison. Networks carrying only net ``p`` fall
back to the bound.

Run it against any set of solved wholesale or balancing networks::

    python scripts/market/diagnose_storage_degeneracy.py \\
        --network resources/market/<scenario>_wholesale.nc [--network ...] \\
        [--price resources/market/<scenario>_wholesale_price.csv ...] \\
        [--output-csv report.csv]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REQUIRED_VARIABLES = (
    "storage_units_i", "storage_units_t_p", "storage_units_p_nom",
    "storage_units_max_hours", "storage_units_efficiency_store",
    "storage_units_efficiency_dispatch",
    "storage_units_carrier", "snapshots_snapshot", "snapshots_generators",
)

# PyPSA omits an attribute from the NetCDF when every unit sits at its default,
# so these are read where present and default to zero, which is PyPSA's own
# default for both.
OPTIONAL_VARIABLES = ("state_of_charge_initial", "standing_loss")

# Written by the balancing solver; older networks carry only net p.
GROSS_VARIABLES = ("storage_units_t_p_dispatch", "storage_units_t_p_store")

# A charge or discharge below this is solver noise, not a simultaneous pair.
SIMULTANEITY_TOLERANCE_MW = 1e-3


def read_storage(path):
    """Return ``(net_dispatch, hours, attributes, gross)`` for one solved network.

    ``gross`` is ``(p_dispatch, p_store)`` when the network carries the split,
    else ``None``.
    """
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        if missing := [v for v in REQUIRED_VARIABLES if v not in dataset.variables]:
            raise ValueError(f"{path} is missing storage variables: {missing}")
        names = pd.Index(dataset["storage_units_i"].values)
        snapshots = pd.DatetimeIndex(dataset["snapshots_snapshot"].values)
        # Read the whole table then subset in pandas: a label .sel on the
        # NetCDF-backed array costs ~0.14 s per unit.
        dispatch = dataset["storage_units_t_p"].to_pandas()
        dispatch.index = snapshots
        dispatch = dispatch.reindex(columns=names, fill_value=0.0)
        gross = None
        if all(variable in dataset.variables for variable in GROSS_VARIABLES):
            frames = []
            for variable in GROSS_VARIABLES:
                frame = dataset[variable].to_pandas()
                frame.index = snapshots
                frames.append(frame.reindex(columns=names, fill_value=0.0))
            gross = tuple(frames)
        hours = pd.Series(dataset["snapshots_generators"].values, index=snapshots,
                          dtype=float)
        attributes = pd.DataFrame({
            key: pd.Series(dataset[f"storage_units_{key}"].values, index=names)
            for key in ("carrier",)
        })
        for key in ("p_nom", "max_hours", "efficiency_store", "efficiency_dispatch",
                    *OPTIONAL_VARIABLES):
            variable = f"storage_units_{key}"
            attributes[key] = (
                pd.Series(dataset[variable].values, index=names, dtype=float)
                if variable in dataset.variables
                else pd.Series(0.0, index=names, dtype=float)
            )
    return dispatch, hours, attributes, gross


def exact_dissipation(gross, hours, attributes, tolerance_mw=SIMULTANEITY_TOLERANCE_MW):
    """Energy burnt by simultaneous charge and discharge, per unit, and the
    unit-hours in which it happened.

    A unit that stores ``x`` and dispatches ``x`` in the same hour moves no net
    power but loses ``x * (1/eta_dispatch - eta_store)`` from its reservoir.
    """
    p_dispatch, p_store = gross
    overlap = np.minimum(p_dispatch.to_numpy(), p_store.to_numpy()).clip(min=0.0)
    loss_rate = (
        1.0 / attributes.efficiency_dispatch.clip(lower=1e-6)
        - attributes.efficiency_store
    ).to_numpy()
    burnt = (overlap * hours.to_numpy()[:, None] * loss_rate[None, :]).sum(axis=0)
    simultaneous = ((p_dispatch > tolerance_mw) & (p_store > tolerance_mw)).sum()
    return (
        pd.Series(burnt, index=p_dispatch.columns),
        pd.Series(simultaneous.to_numpy(), index=p_dispatch.columns, dtype=int),
    )


def reconstruct_state_of_charge(dispatch, hours, attributes) -> pd.DataFrame:
    """State of charge implied by net dispatch, assuming no simultaneity.

    ``soc_t = (1 - standing_loss) * soc_{t-1} + eff_store * store_t
              - dispatch_t / eff_dispatch``
    """
    store = (-dispatch).clip(lower=0.0).mul(hours, axis=0)
    discharge = dispatch.clip(lower=0.0).mul(hours, axis=0)
    delta = (store.mul(attributes.efficiency_store, axis=1)
             - discharge.div(attributes.efficiency_dispatch, axis=1)).to_numpy()
    decay = 1.0 - attributes.standing_loss.to_numpy()
    level = attributes.state_of_charge_initial.to_numpy().astype(float).copy()
    path = np.empty_like(delta)
    for step in range(delta.shape[0]):
        level = level * decay + delta[step]
        path[step] = level
    return pd.DataFrame(path, index=dispatch.index, columns=dispatch.columns)


def diagnose_network(path, label=None) -> tuple[dict, pd.DataFrame]:
    """Return one network's summary and its per-carrier energy balance."""
    dispatch, hours, attributes, gross = read_storage(path)
    energy = dispatch.mul(hours, axis=0)
    charge = (-energy).clip(lower=0.0)
    discharge = energy.clip(lower=0.0)
    capacity = attributes.p_nom * attributes.max_hours
    soc = reconstruct_state_of_charge(dispatch, hours, attributes)
    over_capacity = (soc.max() - capacity).clip(lower=0.0)
    if gross is not None:
        exact, simultaneous = exact_dissipation(gross, hours, attributes)
    else:
        exact = pd.Series(np.nan, index=dispatch.columns)
        simultaneous = pd.Series(np.nan, index=dispatch.columns)

    per_carrier = pd.DataFrame({
        "carrier": attributes.carrier,
        "charge_mwh": charge.sum(),
        "discharge_mwh": discharge.sum(),
        "dissipation_lower_bound_mwh": over_capacity,
        "dissipation_exact_mwh": exact,
        "simultaneous_unit_hours": simultaneous,
        "nominal_roundtrip": attributes.efficiency_store * attributes.efficiency_dispatch,
        "energy_capacity_mwh": capacity,
    })
    per_carrier["expected_discharge_mwh"] = (
        per_carrier.charge_mwh * per_carrier.nominal_roundtrip
    )
    grouped = per_carrier.groupby("carrier", as_index=False).agg(
        units=("charge_mwh", "size"),
        charge_mwh=("charge_mwh", "sum"),
        discharge_mwh=("discharge_mwh", "sum"),
        expected_discharge_mwh=("expected_discharge_mwh", "sum"),
        dissipation_lower_bound_mwh=("dissipation_lower_bound_mwh", "sum"),
        dissipation_exact_mwh=("dissipation_exact_mwh", _sum_or_nan),
        simultaneous_unit_hours=("simultaneous_unit_hours", _sum_or_nan),
        energy_capacity_mwh=("energy_capacity_mwh", "sum"),
    )
    grouped.insert(0, "network", label or Path(path).name)
    grouped["implied_roundtrip"] = np.where(
        grouped.charge_mwh > 0, grouped.discharge_mwh / grouped.charge_mwh, np.nan)
    grouped["nominal_roundtrip"] = np.where(
        grouped.charge_mwh > 0, grouped.expected_discharge_mwh / grouped.charge_mwh, np.nan)

    charge_total = float(charge.to_numpy().sum())
    summary = {
        "network": label or Path(path).name,
        "snapshots": int(len(dispatch)),
        "units": int(len(attributes)),
        "energy_capacity_mwh": float(capacity.sum()),
        "charge_mwh": charge_total,
        "discharge_mwh": float(discharge.to_numpy().sum()),
        "expected_discharge_at_nominal_mwh": float(per_carrier.expected_discharge_mwh.sum()),
        "state_of_charge_initial_mwh": float(attributes.state_of_charge_initial.sum()),
        "state_of_charge_final_mwh": float(soc.iloc[-1].sum()),
        "dissipation_lower_bound_mwh": float(over_capacity.sum()),
        "gross_split_available": gross is not None,
        "dissipation_exact_mwh": float(exact.sum()) if gross is not None else None,
        "simultaneous_unit_hours": (
            int(simultaneous.sum()) if gross is not None else None
        ),
        "units_exceeding_capacity": int((over_capacity > 1e-6).sum()),
        "worst_implied_soc_over_capacity_ratio": float(
            (soc.max() / capacity.replace(0.0, np.nan)).max()),
    }
    summary["carry_over_mwh"] = (
        summary["state_of_charge_final_mwh"] - summary["state_of_charge_initial_mwh"]
    )
    summary["charge_discharge_gap_mwh"] = summary["charge_mwh"] - summary["discharge_mwh"]
    summary["gap_explained_by_nominal_efficiency_mwh"] = (
        summary["charge_mwh"] - summary["expected_discharge_at_nominal_mwh"]
    )
    summary["unexplained_gap_mwh"] = (
        summary["expected_discharge_at_nominal_mwh"] - summary["discharge_mwh"]
    )
    summary["dissipation_lower_bound_share_of_charge"] = (
        summary["dissipation_lower_bound_mwh"] / charge_total if charge_total else None
    )
    return summary, grouped


def _sum_or_nan(values):
    """Sum that stays NaN when nothing in the group carried a value."""
    return values.sum(min_count=1)


def diagnose(networks, labels=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    labels = list(labels or [None] * len(networks))
    if len(labels) != len(networks):
        raise ValueError("Labels must align with networks")
    summaries, carriers = [], []
    for path, label in zip(networks, labels):
        summary, per_carrier = diagnose_network(path, label)
        summaries.append(summary)
        carriers.append(per_carrier)
    per_network = pd.DataFrame(summaries)
    by_carrier = (
        pd.concat(carriers).groupby("carrier", as_index=False).agg(
            units=("units", "max"),
            charge_mwh=("charge_mwh", "sum"),
            discharge_mwh=("discharge_mwh", "sum"),
            expected_discharge_mwh=("expected_discharge_mwh", "sum"),
            dissipation_lower_bound_mwh=("dissipation_lower_bound_mwh", "sum"),
            dissipation_exact_mwh=("dissipation_exact_mwh", _sum_or_nan),
            simultaneous_unit_hours=("simultaneous_unit_hours", _sum_or_nan),
            energy_capacity_mwh=("energy_capacity_mwh", "max"),
        )
    )
    by_carrier["implied_roundtrip"] = by_carrier.discharge_mwh / by_carrier.charge_mwh
    by_carrier["nominal_roundtrip"] = (
        by_carrier.expected_discharge_mwh / by_carrier.charge_mwh)

    charge = float(per_network.charge_mwh.sum())
    discharge = float(per_network.discharge_mwh.sum())
    expected = float(per_network.expected_discharge_at_nominal_mwh.sum())
    dissipation = float(per_network.dissipation_lower_bound_mwh.sum())
    with_gross = per_network[per_network.gross_split_available]
    exact_total = (
        float(with_gross.dissipation_exact_mwh.sum()) if len(with_gross) else None
    )
    simultaneous_total = (
        int(with_gross.simultaneous_unit_hours.sum()) if len(with_gross) else None
    )
    totals = {
        "networks": len(networks),
        "networks_with_gross_split": int(len(with_gross)),
        "dissipation_exact_mwh": exact_total,
        "simultaneous_unit_hours": simultaneous_total,
        "charge_mwh": charge,
        "discharge_mwh": discharge,
        "charge_discharge_gap_mwh": charge - discharge,
        "fleet_energy_capacity_mwh": float(per_network.energy_capacity_mwh.max()),
        "carry_over_mwh": float(per_network.carry_over_mwh.sum()),
        "expected_discharge_at_nominal_mwh": expected,
        "gap_explained_by_nominal_efficiency_mwh": charge - expected,
        "unexplained_gap_mwh": expected - discharge,
        "dissipation_lower_bound_mwh": dissipation,
        "dissipation_lower_bound_share_of_charge": dissipation / charge if charge else None,
        "implied_roundtrip": discharge / charge if charge else None,
        "nominal_roundtrip": expected / charge if charge else None,
        "units_exceeding_capacity": int(per_network.units_exceeding_capacity.max()),
        "note": (
            "gap_explained_by_nominal_efficiency is the round-trip loss the configured "
            "efficiencies imply and is not a defect. dissipation_lower_bound is a strict "
            "floor on the energy lost to simultaneous charge and discharge, derived from "
            "the reconstructed state of charge breaching the unit's energy capacity; it "
            "is a floor because net dispatch hides a simultaneous pair entirely."
        ),
    }
    return per_network, by_carrier, totals


def price_cluster(price_paths, thresholds=(0.0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0)):
    """Hours at or below each threshold, for judging where a cost floor would bite."""
    series = pd.concat([
        pd.read_csv(path, index_col=0, parse_dates=True)["wholesale_price"]
        for path in price_paths
    ]).sort_index()
    return {
        "hours": int(len(series)),
        "hours_at_or_below": {
            f"{threshold:g}": int((series <= threshold).sum()) for threshold in thresholds
        },
        "share_at_or_below": {
            f"{threshold:g}": float((series <= threshold).mean()) for threshold in thresholds
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", action="append", required=True,
                        help="Solved wholesale network (.nc); repeat for several months")
    parser.add_argument("--price", action="append",
                        help="Wholesale price CSV; repeat to report the zero-price cluster")
    parser.add_argument("--output-csv", help="Per-network summary CSV")
    parser.add_argument("--output-json", help="Totals and the price cluster as JSON")
    args = parser.parse_args()

    per_network, by_carrier, totals = diagnose(args.network)
    if args.price:
        totals["price_cluster"] = price_cluster(args.price)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("Per-carrier energy balance (TWh):")
    display = by_carrier.assign(
        charge_twh=by_carrier.charge_mwh / 1e6,
        discharge_twh=by_carrier.discharge_mwh / 1e6,
        dissipation_lower_bound_twh=by_carrier.dissipation_lower_bound_mwh / 1e6,
    )[["carrier", "units", "charge_twh", "discharge_twh", "implied_roundtrip",
       "nominal_roundtrip", "dissipation_lower_bound_twh"]]
    print(display.round(4).to_string(index=False))
    print()
    for key in ("charge_mwh", "discharge_mwh", "charge_discharge_gap_mwh",
                "gap_explained_by_nominal_efficiency_mwh", "carry_over_mwh",
                "unexplained_gap_mwh", "dissipation_lower_bound_mwh"):
        print(f"  {key:<42s} {totals[key] / 1e6:>9.3f} TWh")
    print(f"  {'implied round trip':<42s} {totals['implied_roundtrip']:>9.3f}")
    print(f"  {'nominal round trip':<42s} {totals['nominal_roundtrip']:>9.3f}")
    print(f"  {'dissipation share of charge':<42s} "
          f"{100 * totals['dissipation_lower_bound_share_of_charge']:>8.2f}%")
    if totals["dissipation_exact_mwh"] is not None:
        print(f"  {'dissipation (exact, from gross split)':<42s} "
              f"{totals['dissipation_exact_mwh'] / 1e6:>9.3f} TWh  "
              f"over {totals['networks_with_gross_split']} of {totals['networks']} networks")
        print(f"  {'simultaneous charge+discharge unit-hours':<42s} "
              f"{totals['simultaneous_unit_hours']:>9d}")
    if "price_cluster" in totals:
        print()
        print("Clearing-price cluster:")
        for threshold, count in totals["price_cluster"]["hours_at_or_below"].items():
            share = totals["price_cluster"]["share_at_or_below"][threshold]
            print(f"  hours <= {threshold:>5s} GBP/MWh: {count:>6d} ({100 * share:5.2f}%)")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        per_network.to_csv(args.output_csv, index=False)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps({"totals": totals, "by_carrier": by_carrier.to_dict("records")},
                       indent=2, allow_nan=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
