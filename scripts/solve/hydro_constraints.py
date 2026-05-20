"""Hydro and pumped-hydro operational constraints for PyPSA solves."""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd


LARGE_HYDRO_CARRIERS = {"large_hydro"}
PUMPED_HYDRO_CARRIERS = {"Pumped Storage Hydroelectricity", "pumped_hydro"}


def combine_extra_functionalities(*callbacks: Optional[Callable]) -> Optional[Callable]:
    """Combine multiple PyPSA extra_functionality callbacks into one."""
    active_callbacks = [callback for callback in callbacks if callback is not None]
    if not active_callbacks:
        return None

    def combined(network, snapshots):
        for callback in active_callbacks:
            callback(network, snapshots)

    return combined


def get_large_hydro_names(network) -> list[str]:
    """Return large-hydro generator names present in the network."""
    if len(network.generators) == 0:
        return []

    carriers = network.generators.get(
        "carrier", pd.Series(index=network.generators.index, dtype=object)
    )
    return [
        name for name, carrier in carriers.items()
        if str(carrier) in LARGE_HYDRO_CARRIERS
    ]


def get_pumped_hydro_names(network) -> list[str]:
    """Return pumped-hydro storage-unit names present in the network."""
    if len(network.storage_units) == 0:
        return []

    carriers = network.storage_units.get(
        "carrier", pd.Series(index=network.storage_units.index, dtype=object)
    )
    return [
        name for name, carrier in carriers.items()
        if str(carrier) in PUMPED_HYDRO_CARRIERS
    ]


def initialise_large_hydro_storage_state(network, hydro_config: dict) -> Optional[pd.Series]:
    """Create initial reservoir energy state for large-hydro generators."""
    names = get_large_hydro_names(network)
    large_cfg = hydro_config.get("large_hydro", {})

    if not names or not large_cfg.get("reservoir_model", False):
        return None

    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    reservoir_capacity = p_nom * float(large_cfg.get("reservoir_capacity_hours", 168.0))
    initial_storage = p_nom * float(large_cfg.get("initial_storage_hours", 48.0))
    return initial_storage.clip(lower=0.0, upper=reservoir_capacity)


def log_hydro_constraint_setup(network, scenario_config: dict, logger, context: str = "Hydro") -> None:
    """Log hydro policy summaries from the outer solve workflow."""
    hydro_config = scenario_config.get("hydro", {})
    if not hydro_config.get("enabled", False):
        return

    large_hydro_names = get_large_hydro_names(network)
    large_cfg = hydro_config.get("large_hydro", {})
    if large_hydro_names and large_cfg.get("reservoir_model", False):
        logger.info(
            "%s: large-hydro reservoir constraints enabled for %d generators "
            "(initial storage %.1fh, reservoir capacity %.1fh, inflow factor %.2f)",
            context,
            len(large_hydro_names),
            float(large_cfg.get("initial_storage_hours", 48.0)),
            float(large_cfg.get("reservoir_capacity_hours", 168.0)),
            float(large_cfg.get("inflow_profile_factor", 0.55)),
        )

    pumped_hydro_names = get_pumped_hydro_names(network)
    pumped_cfg = hydro_config.get("pumped_hydro", {})
    policy_parts = []

    min_soc_fraction = pumped_cfg.get("min_soc_fraction")
    if min_soc_fraction is not None:
        policy_parts.append(f"min SoC {100 * float(min_soc_fraction):.0f}%")

    final_soc_fraction = pumped_cfg.get("final_soc_fraction")
    if final_soc_fraction is not None:
        policy_parts.append(f"final SoC {100 * float(final_soc_fraction):.0f}%")

    if pumped_hydro_names and policy_parts:
        logger.info(
            "%s: pumped-hydro SOC policy enabled for %d storage units (%s)",
            context,
            len(pumped_hydro_names),
            ", ".join(policy_parts),
        )


def _get_snapshot_weights(network, snapshots) -> pd.Series:
    """Return per-snapshot energy weights in hours."""
    if hasattr(network, "snapshot_weightings") and hasattr(network.snapshot_weightings, "generators"):
        return network.snapshot_weightings.generators.reindex(snapshots).fillna(1.0)
    return pd.Series(1.0, index=pd.Index(snapshots))


def _get_large_hydro_inflow_energy(network, snapshots, hydro_config: dict, names: list[str]) -> pd.DataFrame:
    """Return per-snapshot inflow energy (MWh) for each large-hydro generator."""
    large_cfg = hydro_config.get("large_hydro", {})
    inflow_factor = float(large_cfg.get("inflow_profile_factor", 0.55))

    p_nom = network.generators.loc[names, "p_nom"].astype(float)
    static_p_max = network.generators.loc[names, "p_max_pu"].fillna(1.0).astype(float)

    if getattr(network.generators_t, "p_max_pu", pd.DataFrame()).empty:
        p_max = pd.DataFrame(
            {name: static_p_max[name] for name in names},
            index=pd.Index(snapshots),
        )
    else:
        p_max = network.generators_t.p_max_pu.reindex(index=snapshots).copy()
        for name in names:
            if name in p_max.columns:
                p_max[name] = p_max[name].fillna(static_p_max[name])
            else:
                p_max[name] = static_p_max[name]
        p_max = p_max[names]

    inflow_power = p_max.mul(p_nom, axis=1) * inflow_factor
    weights = _get_snapshot_weights(network, snapshots)
    return inflow_power.mul(weights, axis=0)


def build_hydro_constraints_callback(
    network,
    scenario_config: dict,
    large_hydro_storage_state: Optional[pd.Series] = None,
    pumped_min_soc_override: Optional[float] = None,
) -> Optional[Callable]:
    """Build an extra_functionality callback for hydro-specific constraints.

    Parameters
    ----------
    pumped_min_soc_override : float, optional
        If provided, overrides ``pumped_hydro.min_soc_fraction`` from config.
        Used by the wholesale solve to enforce a higher SoC floor, preserving
        headroom for BM increases.
    """
    hydro_config = scenario_config.get("hydro", {})
    if not hydro_config.get("enabled", False):
        return None

    large_hydro_names = get_large_hydro_names(network)
    pumped_hydro_names = get_pumped_hydro_names(network)
    if not large_hydro_names and not pumped_hydro_names:
        return None

    large_cfg = hydro_config.get("large_hydro", {})
    pumped_cfg = hydro_config.get("pumped_hydro", {})
    if large_hydro_storage_state is None and large_hydro_names:
        large_hydro_storage_state = initialise_large_hydro_storage_state(network, hydro_config)

    def hydro_extra_functionality(n, snapshots):
        import xarray as xr

        model = n.model

        if large_hydro_names and large_cfg.get("reservoir_model", False) and "Generator-p" in model.variables:
            names = [name for name in large_hydro_names if name in n.generators.index]
            if names:
                if large_hydro_storage_state is None:
                    initial_storage = pd.Series(0.0, index=names)
                else:
                    initial_storage = large_hydro_storage_state.reindex(names).fillna(0.0)
                inflow_energy = _get_large_hydro_inflow_energy(n, snapshots, hydro_config, names)
                inflow_cumsum = inflow_energy.cumsum(axis=0)

                gen_p = model.variables["Generator-p"].sel({"name": names})
                weights = _get_snapshot_weights(n, snapshots)
                weight_da = xr.DataArray(
                    weights.values,
                    dims=["snapshot"],
                    coords={"snapshot": snapshots},
                )
                weighted_dispatch = gen_p * weight_da

                for idx in range(len(snapshots)):
                    lhs = weighted_dispatch.isel(snapshot=slice(0, idx + 1)).sum("snapshot")
                    rhs = initial_storage.values + inflow_cumsum.iloc[idx].values
                    model.add_constraints(
                        lhs <= rhs,
                        name=f"Generator-large_hydro_water_budget_{idx}",
                    )

        if pumped_hydro_names and "StorageUnit-state_of_charge" in model.variables:
            names = [name for name in pumped_hydro_names if name in n.storage_units.index]
            if names:
                energy_cap = (
                    n.storage_units.loc[names, "p_nom"].astype(float)
                    * n.storage_units.loc[names, "max_hours"].astype(float)
                )
                state_of_charge = model.variables["StorageUnit-state_of_charge"].sel({"name": names})
                energy_cap_da = xr.DataArray(
                    energy_cap.reindex(names).values,
                    dims=["name"],
                    coords={"name": names},
                )

                min_soc_fraction = pumped_min_soc_override if pumped_min_soc_override is not None else pumped_cfg.get("min_soc_fraction")
                if min_soc_fraction is not None:
                    model.add_constraints(
                        state_of_charge >= energy_cap_da * float(min_soc_fraction),
                        name="StorageUnit-pumped_hydro_min_soc",
                    )

                final_soc_fraction = pumped_cfg.get("final_soc_fraction")
                if final_soc_fraction is not None and len(snapshots) > 0:
                    model.add_constraints(
                        state_of_charge.isel(snapshot=-1) >= energy_cap_da * float(final_soc_fraction),
                        name="StorageUnit-pumped_hydro_final_soc",
                    )

    return hydro_extra_functionality


def update_large_hydro_storage_state(
    network,
    snapshots,
    scenario_config: dict,
    current_storage: Optional[pd.Series],
) -> Optional[pd.Series]:
    """Advance large-hydro reservoir state after a solved window."""
    hydro_config = scenario_config.get("hydro", {})
    large_cfg = hydro_config.get("large_hydro", {})
    names = get_large_hydro_names(network)

    if current_storage is None or not names or not large_cfg.get("reservoir_model", False):
        return current_storage

    reservoir_capacity = (
        network.generators.loc[names, "p_nom"].astype(float)
        * float(large_cfg.get("reservoir_capacity_hours", 168.0))
    )
    inflow_energy = _get_large_hydro_inflow_energy(network, snapshots, hydro_config, names).sum(axis=0)

    dispatch_df = getattr(network.generators_t, "p", pd.DataFrame())
    if dispatch_df.empty:
        dispatch_energy = pd.Series(0.0, index=names)
    else:
        dispatch = dispatch_df.reindex(index=snapshots, columns=names, fill_value=0.0)
        weights = _get_snapshot_weights(network, snapshots)
        dispatch_energy = dispatch.mul(weights, axis=0).sum(axis=0)

    next_storage = current_storage.reindex(names).fillna(0.0) + inflow_energy - dispatch_energy
    return next_storage.clip(lower=0.0, upper=reservoir_capacity)