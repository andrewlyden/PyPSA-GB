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


def widen_ramp_limits_to_availability(network, logger: logging.Logger) -> int:
    """Keep ramp limits feasible beside a minimum output and an hourly availability.

    A unit with a ramp limit and a positive minimum output (p_min_pu) somewhere is held between
    p_min_pu and p_max_pu in every snapshot. Where availability falls, or the floor rises, by more
    than the ramp limit in one snapshot, no dispatch can satisfy all three. For those units the limit
    in each snapshot becomes the larger of its own value and the step it must follow:

        ramp_limit_down[t] = max(ramp_limit_down, p_max_pu[t-1] - p_max_pu[t])
        ramp_limit_up[t]   = max(ramp_limit_up,   p_min_pu[t]   - p_min_pu[t-1])

    with p_min_pu clipped to p_max_pu (and written back clipped where the floor exceeded
    availability), so a dispatch that stays inside both bounds always exists. An outage is then
    forced, not dispatched. Units with no minimum output can always run at zero and
    keep their static limits; committable units are left to PyPSA's commitment constraints. Limits
    are shares of p_nom per snapshot. Call after p_min_pu and p_max_pu are final, before optimize.

    Returns the number of units given hourly limits.
    """
    gens = network.generators
    if gens.empty or len(network.snapshots) < 2:
        return 0
    static = {}
    for column in ("ramp_limit_up", "ramp_limit_down"):
        static[column] = (pd.to_numeric(gens[column], errors="coerce") if column in gens.columns
                          else pd.Series(float("nan"), index=gens.index))
    limited = static["ramp_limit_up"].notna() | static["ramp_limit_down"].notna()
    if "committable" in gens.columns:
        limited &= ~gens["committable"].fillna(False).astype(bool)
    if not limited.any():
        return 0

    names = gens.index[limited]
    p_max = network.get_switchable_as_dense("Generator", "p_max_pu").reindex(columns=names)
    p_min_raw = network.get_switchable_as_dense("Generator", "p_min_pu").reindex(columns=names)
    p_min = p_min_raw.clip(upper=p_max)
    names = names[(p_min > 0).any().to_numpy()]
    if names.empty:
        return 0
    p_max, p_min = p_max[names], p_min[names]

    # A floor above availability (during an outage) has no feasible dispatch at all: hold it at availability.
    above = names[(p_min_raw[names] > p_max).any().to_numpy()]
    if not above.empty:
        varying = network.generators_t["p_min_pu"]
        network.generators_t["p_min_pu"] = pd.concat(
            [varying.drop(columns=varying.columns.intersection(above)), p_min[above]], axis=1)

    steps = {
        "ramp_limit_down": (p_max.shift(1) - p_max).fillna(0.0),
        "ramp_limit_up": (p_min - p_min.shift(1)).fillna(0.0),
    }
    widened_hours = 0
    for column, step in steps.items():
        base = static[column].reindex(names).dropna()
        if base.empty:
            continue
        hourly = step[base.index].clip(lower=base, axis=1)
        widened_hours += int((hourly > base).to_numpy().sum())
        varying = network.generators_t[column]
        network.generators_t[column] = pd.concat(
            [varying.drop(columns=varying.columns.intersection(base.index)), hourly], axis=1)

    logger.info(
        "Ramp limits beside a minimum output: %d unit(s) given hourly limits, widened in %d "
        "unit-snapshots where availability or the floor steps faster than the limit",
        len(names), widened_hours,
    )
    return len(names)
