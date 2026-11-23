"""Represent historical CfD contract tranches explicitly in a PyPSA network.

The public LCCC register often has several contracts for one physical project.
Keeping that project as one generator and multiplying its dispatch by several
shares makes commissioning/support dates ambiguous.  This module replaces the
supported share with one generator per contract and retains any remainder as a
merchant physical asset.  Every tranche inherits the source profile, so total
capacity and available energy are conserved exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generators.integrate_renewable_generators import (
    _parse_operational_dates,
)


REQUIRED = {"cfd_id", "model_generator", "allocation_share"}


def _settlement_dates(actuals_path: str | Path) -> dict[str, pd.Timestamp]:
    actual = pd.read_csv(actuals_path, usecols=["Settlement_Date", "CfD_ID"])
    actual["Settlement_Date"] = pd.to_datetime(actual["Settlement_Date"], errors="coerce")
    return actual.dropna(subset=["Settlement_Date"]).groupby("CfD_ID")["Settlement_Date"].min().to_dict()


def _copy_dynamic_column(network, source: str, target: str) -> None:
    for frame in network.generators_t.values():
        if source in frame.columns:
            frame[target] = frame[source].copy()


def split_historical_cfd_tranches(
    network,
    overrides_path: str | Path,
    actuals_path: str | Path,
    logger=None,
) -> dict:
    """Split shared/partial CfD assets while conserving physical capability.

    ``cfd_support_start`` records the first public settlement date.  It does
    not gate physical output: a commissioned asset may operate as merchant
    generation before its CfD starts.  Physical commissioning is handled from
    the source ``operational_date`` independently.
    """
    overrides = pd.read_csv(overrides_path)
    missing = REQUIRED - set(overrides.columns)
    if missing:
        raise ValueError(f"CfD tranche mapping is missing columns: {sorted(missing)}")
    overrides["allocation_share"] = pd.to_numeric(overrides["allocation_share"], errors="raise")
    dates = _settlement_dates(actuals_path)
    created = []
    residuals = []
    skipped = []

    for source, group in overrides.groupby("model_generator", sort=False):
        if source not in network.generators.index:
            skipped.append(source)
            continue
        total_share = float(group["allocation_share"].sum())
        if total_share > 1.000001:
            raise ValueError(f"CfD tranches exceed physical capacity for {source}: {total_share:.6f}")

        source_row = network.generators.loc[source].copy()
        source_p_nom = float(source_row["p_nom"])
        if len(group) == 1 and abs(total_share - 1.0) <= 1e-9:
            contract = group.iloc[0]
            cfd_id = str(contract["cfd_id"])
            network.generators.at[source, "support_type"] = "CfD"
            network.generators.at[source, "cfd_id"] = cfd_id
            network.generators.at[source, "cfd_allocation_share_of_source"] = 1.0
            network.generators.at[source, "source_generator_name"] = source
            network.generators.at[source, "cfd_support_start"] = dates.get(cfd_id, pd.NaT)
            continue
        for row in group.to_dict("records"):
            cfd_id = str(row["cfd_id"])
            share = float(row["allocation_share"])
            tranche = f"{source}__CfD__{cfd_id}"
            attrs = source_row.to_dict()
            attrs["p_nom"] = source_p_nom * share
            attrs["support_type"] = "CfD"
            attrs["cfd_id"] = cfd_id
            attrs["cfd_allocation_share_of_source"] = share
            attrs["source_generator_name"] = source
            attrs["cfd_support_start"] = dates.get(cfd_id, pd.NaT)
            # Results/dual attributes are outputs, not valid build inputs.
            for field in ("p_nom_opt", "mu_upper", "mu_lower"):
                attrs.pop(field, None)
            network.add("Generator", tranche, **attrs)
            _copy_dynamic_column(network, source, tranche)
            created.append(tranche)

        remainder = max(0.0, 1.0 - total_share)
        if remainder > 1e-9:
            network.generators.at[source, "p_nom"] = source_p_nom * remainder
            network.generators.at[source, "support_type"] = "merchant"
            network.generators.at[source, "cfd_id"] = ""
            network.generators.at[source, "source_generator_name"] = source
            residuals.append(source)
        else:
            network.remove("Generator", source)

    summary = {
        "created_cfd_tranches": len(created),
        "physical_residual_generators": len(residuals),
        "source_generators_not_in_this_build": sorted(skipped),
    }
    if logger:
        logger.info(
            "Historical CfD tranches: %d contract assets, %d merchant residuals, %d absent sources",
            len(created), len(residuals), len(skipped),
        )
    return summary


def apply_physical_commissioning_gates(network, logger=None) -> int:
    """Set pre-commissioning availability and minimum output to zero.

    The time-varying minimum is also capped by the final availability profile.
    This preserves must-run behaviour when a unit is available while preventing
    contradictory ``p_min_pu > p_max_pu`` constraints during commissioning or
    outage periods.
    """
    if "operational_date" not in network.generators.columns:
        return 0
    snapshots = pd.DatetimeIndex(network.snapshots).tz_localize(None)
    dates = _parse_operational_dates(network.generators["operational_date"])
    gated = 0
    for name, date in dates.dropna().items():
        mask = snapshots < date.normalize()
        if name in network.generators_t.p_max_pu.columns:
            profile = network.generators_t.p_max_pu[name].reindex(network.snapshots).copy()
        else:
            profile = pd.Series(float(network.generators.at[name, "p_max_pu"]), index=network.snapshots)
        profile.loc[mask] = 0.0
        network.generators_t.p_max_pu[name] = profile

        static_min = float(network.generators.at[name, "p_min_pu"])
        if name in network.generators_t.p_min_pu.columns:
            minimum = network.generators_t.p_min_pu[name].reindex(
                network.snapshots
            ).fillna(static_min)
        else:
            minimum = pd.Series(static_min, index=network.snapshots)
        bounded_minimum = minimum.clip(upper=profile)
        if not bounded_minimum.equals(minimum):
            network.generators_t.p_min_pu[name] = bounded_minimum

        # PyPSA omits an all-zero dynamic p_min_pu column when exporting to
        # NetCDF because zero is the time-series default. Persist the same zero
        # on the static component too, otherwise its positive must-run floor is
        # resurrected when the network is loaded by a later workflow stage.
        if bounded_minimum.eq(0.0).all() and static_min != 0.0:
            network.generators.at[name, "p_min_pu"] = 0.0

        if mask.any():
            gated += 1
    if logger and gated:
        logger.info(
            "Applied physical commissioning gates with feasible minimum-output "
            "bounds to %d generators",
            gated,
        )
    return gated


def apply_generator_monthly_availability(network, path: str | Path, year: int, logger=None) -> int:
    """Apply auditable station/month availability overrides.

    CSV columns are ``generator``, ``year``, ``month`` and
    ``availability_factor``.  A factor replaces the broad carrier factor for
    the named month; it is not multiplied by it.  An optional ``fixed`` column
    (true) also sets the month's minimum to the factor, holding the station at
    that output (scripts/generators/build_elexon_monthly_availability.py).
    """
    table = pd.read_csv(path)
    required = {"generator", "year", "month", "availability_factor"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Monthly availability file is missing: {sorted(missing)}")
    table = table.loc[pd.to_numeric(table["year"], errors="coerce").eq(int(year))].copy()
    snapshots = pd.DatetimeIndex(network.snapshots)
    applied = 0
    for row in table.to_dict("records"):
        name = str(row["generator"])
        if name not in network.generators.index:
            continue
        factor = float(row["availability_factor"])
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"Availability outside [0,1] for {name}: {factor}")
        mask = (snapshots.year == int(year)) & (snapshots.month == int(row["month"]))
        if not mask.any():
            continue
        if name in network.generators_t.p_max_pu.columns:
            profile = network.generators_t.p_max_pu[name].reindex(network.snapshots).copy()
        else:
            profile = pd.Series(float(network.generators.at[name, "p_max_pu"]), index=network.snapshots)
        profile.loc[mask] = factor
        network.generators_t.p_max_pu[name] = profile
        static_min = float(network.generators.at[name, "p_min_pu"])
        fixed = str(row.get("fixed", "")).strip().lower() in {"true", "1", "yes"}
        if fixed or static_min > factor:
            if name in network.generators_t.p_min_pu.columns:
                minimum = network.generators_t.p_min_pu[name].reindex(network.snapshots).copy()
            else:
                minimum = pd.Series(static_min, index=network.snapshots)
            minimum.loc[mask] = factor
            network.generators_t.p_min_pu[name] = minimum
        # observed availability: random thermal outages (thermal_availability.py) skip this unit
        network.generators.at[name, "observed_availability"] = str(row.get("evidence", "station-month rows"))
        applied += 1
    if logger:
        logger.info("Applied %d station-month availability overrides", applied)
    return applied
