import math
import pypsa


def _validate_per_unit_value(name: str, value: float) -> float:
    """Validate and return a finite per-unit value."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")

    try:
        validated_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not math.isfinite(validated_value):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= validated_value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return validated_value


def configure_nuclear(
        network: pypsa.Network,
        min_output: float,
        ramp_limit: float
) -> pypsa.Network:
    """
    Apply static operating constraints to nuclear generators.

    The minimum output and ramp limits are expressed per unit of nominal capacity.

    This function currently does not introduce start-up/down decisions, minimum
    up/down times or cycling costs.

    Parameters
    ----------
    network: pypsa.Network
        PyPSA network containing generators to configure.
    min_output: float,
        Minimum nuclear output per unit of nominal capacity,
        in [0,1].
    ramp_limit: float
        Maximum increase/decrease between adjacent snapshots,
        in [0,1].

    Returns
    -------
    pypsa.Network
        Network with nuclear operating constraints applied.
        
    Raises
    ------
    ValueError
        If either value is invalid or no nuclear generators
        are found.
    """
    min_output = _validate_per_unit_value("min_output", min_output)
    ramp_limit = _validate_per_unit_value("ramp_limit", ramp_limit)

    carriers = (
        network.generators["carrier"]
        .fillna("")
        .astype(str)
        .str.casefold()
    )

    nuclear = carriers.eq("nuclear")

    if not nuclear.any():
        raise ValueError("No nuclear generators found.")

    network.generators.loc[nuclear, "p_min_pu"] = min_output
    network.generators.loc[nuclear, "ramp_limit_up"] = ramp_limit
    network.generators.loc[nuclear, "ramp_limit_down"] = ramp_limit

    return network
