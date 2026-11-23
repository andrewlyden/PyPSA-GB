"""LP-compatible thermal ramp-rate assumptions for future market scenarios."""

import logging

import pandas as pd


def _carrier_parameters(config, carrier):
    carriers = config.get("carriers", {})
    if carrier in carriers:
        return carriers[carrier]
    carrier_lower = str(carrier).lower()
    for name, value in carriers.items():
        if str(name).lower() == carrier_lower:
            return value
    return None


def apply_thermal_ramp_limits(network, config, logger: logging.Logger):
    """Apply per-unit ramp limits without setting any UC-related attributes."""
    if not config.get("enabled", False):
        return 0

    for column in ("ramp_limit_up", "ramp_limit_down"):
        if column not in network.generators.columns:
            network.generators[column] = pd.NA

    applied = 0
    for name, generator in network.generators.iterrows():
        params = _carrier_parameters(config, generator["carrier"])
        if params is None:
            continue
        up = float(params.get("ramp_limit_up", 1.0))
        down = float(params.get("ramp_limit_down", 1.0))
        if not 0.0 < up <= 1.0 or not 0.0 < down <= 1.0:
            raise ValueError(f"Invalid ramp limits for {generator['carrier']}: {up}, {down}")
        network.generators.at[name, "ramp_limit_up"] = up
        network.generators.at[name, "ramp_limit_down"] = down
        applied += 1

    logger.info("Applied LP ramp limits to %d thermal generators", applied)
    return applied
