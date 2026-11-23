"""Branch impedance units for the ETYS network build.

ETYS Appendix B publishes R, X and B as "% on 100 MVA".  Every ETYS branch
source (the base sheets, the Appendix B upgrade rows, the GB_network.xlsx
repair and wind-farm edges, the documented bus sections, the CP30
accelerated components, the HND register and the registry defaults) is
carried through the build as **per unit on a 100 MVA base** ("pu100"):
``pu100 = percent / 100``.

PyPSA does not use that unit.  PyPSA 1.0.7 (``calculate_dependent_values``)
reads

* ``Line.x`` and ``Line.r`` in ohm and ``Line.b`` and ``Line.g`` in siemens,
  and forms ``x_pu = x / v_nom**2`` and ``b_pu = b * v_nom**2`` with the
  ``v_nom`` (kV) of ``bus0``: per unit on a 1 MVA base;
* ``Transformer.x`` and ``Transformer.r`` (and ``b``, ``g``) per unit on the
  transformer's own ``s_nom``, and forms ``x_pu = x / s_nom`` and
  ``b_pu = b * s_nom``: again per unit on a 1 MVA base.

So a branch with reactance ``x100`` pu on 100 MVA must be stored as

* line:        ``x = x100 * v_nom**2 / 100`` ohm,  ``b = b100 * 100 / v_nom**2`` S
* transformer: ``x = x100 * s_nom / 100`` pu,      ``b = b100 * 100 / s_nom`` pu

and PyPSA then sees ``x_pu = x100 / 100`` for both.  Before 27 Sep 2026 the
pu100 values were written straight into ``x``, so PyPSA read a 400 kV line as
1,600 times too stiff (275 kV: 756; 132 kV: 174) and each transformer at a
factor that depended on its rating; flows split on the wrong impedances.

The build converts exactly once, in :func:`convert_network_pu100_to_pypsa`,
at the end of ``ETYS_network.create_network`` after every branch source has
been applied.  ``network.meta[BRANCH_UNITS_META_KEY]`` records which units the
branch tables hold so a second conversion, or an upgrade applied in the wrong
units, raises instead of silently corrupting the network.

A transformer's stored impedance is relative to its ``s_nom``: changing
``s_nom`` alone changes its per-unit impedance.  Code that relaxes or rescales
a transformer rating after the build must use
:func:`set_transformer_s_nom_preserving_impedance`.
"""

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from scripts.network_build.etys_file_registry import (
    ELECTRICAL_DEFAULTS, COUPLER_MAX_LENGTH_KM,
)

BASE_MVA = 100.0

BRANCH_UNITS_META_KEY = "branch_impedance_units"
UNITS_PU100 = "pu_on_100MVA"
UNITS_PYPSA = "pypsa_native"  # lines ohm/siemens; transformers pu on own s_nom


def _is_missing(value) -> bool:
    try:
        return value is None or pd.isna(value)
    except (TypeError, ValueError):
        return True


def is_short_connection(circuit_type="", length_km=None) -> bool:
    """True for an Appendix B bus section, tee or other connection <= 1 km."""
    if "zero length" in str(circuit_type or "").lower():
        return True
    try:
        return length_km is not None and not pd.isna(length_km) and float(length_km) <= COUPLER_MAX_LENGTH_KM
    except (TypeError, ValueError):
        return False


def resolve_source_impedance_pu100(r, x, b, kind: str, circuit_type="", length_km=None,
                                   fill_missing: bool = True):
    """Apply the one policy for zero or missing Appendix B impedances.

    Inputs are pu on 100 MVA (``percent / 100``) or NaN.  ``kind`` is
    ``'line'``, ``'cable'`` or ``'transformer'`` and selects the
    ``ELECTRICAL_DEFAULTS`` entry.

    * ``r = 0`` with ``x > 0`` is a valid source value and is kept (series
      compensators, couplers and short cables are published with R = 0).
    * ``b = 0`` is a valid source value and is kept.
    * a published ``x = 0`` on a circuit of type "Zero Length" or no longer
      than ``COUPLER_MAX_LENGTH_KM`` is a bus section: it gets the coupler
      reactance the published 9,999 MVA couplers carry (0.01 % on 100 MVA).
      An unpublished (NaN) x is missing data, never a coupler: the
      GB_network.xlsx BMU edges carry no impedance or length at all and some
      join different voltages.
    * any other missing value (NaN, or ``x = 0`` on a longer circuit) takes
      the type default when ``fill_missing`` and is reported missing otherwise.

    Returns ``(values, assumptions, missing)``: ``values`` maps r/x/b to pu100
    (``None`` where missing and not filled), ``assumptions`` names the fields
    set by rule rather than taken from the source.
    """
    defaults = ELECTRICAL_DEFAULTS[kind]
    coupler = ELECTRICAL_DEFAULTS["coupler"]
    values, assumptions, missing = {}, [], []
    x_val = None if _is_missing(x) else float(x)
    if x_val is None or x_val == 0.0:
        if (x_val == 0.0 and kind != "transformer"
                and is_short_connection(circuit_type, length_km)):
            x_val = float(coupler["x"])
            assumptions.append("x (zero-reactance bus section: coupler)")
        elif fill_missing:
            x_val = float(defaults["x"])
            assumptions.append("x (type default)")
        else:
            missing.append("x")
            x_val = None
    values["x"] = x_val
    for key, value in (("r", r), ("b", b)):
        if _is_missing(value):
            if fill_missing:
                values[key] = float(defaults[key])
                assumptions.append(f"{key} (type default)")
            else:
                values[key] = None
                missing.append(key)
        else:
            values[key] = float(value)
    return values, assumptions, missing


def get_branch_units(network, default: Optional[str] = None) -> Optional[str]:
    """Return the branch-impedance units recorded on ``network.meta``."""
    meta = getattr(network, "meta", None) or {}
    try:
        return meta.get(BRANCH_UNITS_META_KEY, default)
    except AttributeError:
        return default


def set_branch_units(network, units: str) -> None:
    """Record the branch-impedance units on ``network.meta``."""
    if units not in (UNITS_PU100, UNITS_PYPSA):
        raise ValueError(f"Unknown branch impedance units {units!r}")
    meta = dict(getattr(network, "meta", None) or {})
    meta[BRANCH_UNITS_META_KEY] = units
    network.meta = meta


def _line_v_nom(network, lines: pd.DataFrame) -> pd.Series:
    """The ``v_nom`` PyPSA uses for each line: that of ``bus0``."""
    v_nom = pd.to_numeric(lines["bus0"].map(network.buses["v_nom"]), errors="coerce")
    bad = v_nom.isna() | (v_nom <= 0)
    if bad.any():
        raise ValueError(
            "Cannot convert line impedances: bus0 v_nom missing or non-positive for "
            f"{int(bad.sum())} lines, e.g. {list(lines.index[bad][:5])}"
        )
    return v_nom.astype(float)


def _transformer_s_nom(transformers: pd.DataFrame) -> pd.Series:
    s_nom = pd.to_numeric(transformers["s_nom"], errors="coerce")
    bad = s_nom.isna() | (s_nom <= 0)
    if bad.any():
        raise ValueError(
            "Cannot convert transformer impedances: s_nom missing or non-positive for "
            f"{int(bad.sum())} transformers, e.g. {list(transformers.index[bad][:5])}"
        )
    return s_nom.astype(float)


def line_pu100_to_pypsa(values: pd.DataFrame, v_nom_kv) -> pd.DataFrame:
    """Convert line r, x, b, g from pu on 100 MVA to ohm / siemens."""
    out = values.copy()
    z_base = np.asarray(v_nom_kv, dtype=float) ** 2 / BASE_MVA  # ohm per pu100
    for col in ("r", "x"):
        if col in out:
            out[col] = out[col].astype(float) * z_base
    for col in ("b", "g"):
        if col in out:
            out[col] = out[col].astype(float) / z_base
    return out


def transformer_pu100_to_pypsa(values: pd.DataFrame, s_nom) -> pd.DataFrame:
    """Convert transformer r, x, b, g from pu on 100 MVA to pu on own s_nom."""
    out = values.copy()
    ratio = np.asarray(s_nom, dtype=float) / BASE_MVA
    for col in ("r", "x"):
        if col in out:
            out[col] = out[col].astype(float) * ratio
    for col in ("b", "g"):
        if col in out:
            out[col] = out[col].astype(float) / ratio
    return out


def _convert(network, forward: bool) -> None:
    cols = ["r", "x", "b", "g"]
    lines = network.lines
    if len(lines):
        present = [c for c in cols if c in lines.columns]
        z_base = _line_v_nom(network, lines) ** 2 / BASE_MVA
        for col in present:
            factor = z_base if col in ("r", "x") else 1.0 / z_base
            if not forward:
                factor = 1.0 / factor
            lines[col] = lines[col].astype(float) * factor
    transformers = network.transformers
    if len(transformers):
        present = [c for c in cols if c in transformers.columns]
        ratio = _transformer_s_nom(transformers) / BASE_MVA
        for col in present:
            factor = ratio if col in ("r", "x") else 1.0 / ratio
            if not forward:
                factor = 1.0 / factor
            transformers[col] = transformers[col].astype(float) * factor


def convert_network_pu100_to_pypsa(network, logger: Optional[logging.Logger] = None):
    """Convert every line and transformer from pu100 to PyPSA's native units.

    Must be called exactly once per build.  Raises if the network already
    records native units.
    """
    logger = logger or logging.getLogger(__name__)
    units = get_branch_units(network)
    if units == UNITS_PYPSA:
        raise ValueError("Branch impedances are already in PyPSA units; refusing to convert twice")
    if units not in (None, UNITS_PU100):
        raise ValueError(f"Unknown branch impedance units {units!r} on network.meta")
    _convert(network, forward=True)
    set_branch_units(network, UNITS_PYPSA)
    logger.info(
        "Converted %d lines (pu100 -> ohm/S at bus0 v_nom) and %d transformers "
        "(pu100 -> pu on own s_nom) to PyPSA units",
        len(network.lines), len(network.transformers),
    )
    return network


def convert_network_pypsa_to_pu100(network, logger: Optional[logging.Logger] = None):
    """Exact inverse of :func:`convert_network_pu100_to_pypsa`."""
    logger = logger or logging.getLogger(__name__)
    if get_branch_units(network) != UNITS_PYPSA:
        raise ValueError("Network does not record PyPSA branch units; nothing to invert")
    _convert(network, forward=False)
    set_branch_units(network, UNITS_PU100)
    logger.info("Converted branch impedances from PyPSA units back to pu on 100 MVA")
    return network


def branch_values_for_network(network, component: str, values: dict, bus0: str,
                              s_nom: Optional[float] = None) -> dict:
    """Express pu100 branch values in whatever units ``network`` holds.

    Used by layers that may be applied to a network either during the build
    (pu100) or to an already-converted network.
    """
    if get_branch_units(network) != UNITS_PYPSA:
        return dict(values)
    frame = pd.DataFrame([{k: v for k, v in values.items() if k in ("r", "x", "b", "g")}])
    if component == "Line":
        v_nom = float(network.buses.at[bus0, "v_nom"])
        frame = line_pu100_to_pypsa(frame, [v_nom])
    elif component == "Transformer":
        if s_nom is None:
            s_nom = float(values["s_nom"])
        frame = transformer_pu100_to_pypsa(frame, [float(s_nom)])
    else:
        raise ValueError(f"Unsupported branch component {component!r}")
    out = dict(values)
    out.update({k: float(frame.iloc[0][k]) for k in frame.columns})
    return out


def set_transformer_s_nom_preserving_impedance(network, names: Iterable, new_s_nom) -> None:
    """Change transformer ratings without changing their physical impedance.

    PyPSA stores transformer r/x/b/g per unit on ``s_nom``, so a new rating
    rescales r and x by ``new/old`` and b and g by ``old/new``: ``x_pu`` (1 MVA
    base) is unchanged.
    """
    names = pd.Index(list(names))
    if names.empty:
        return
    trafos = network.transformers
    old = pd.to_numeric(trafos.loc[names, "s_nom"], errors="coerce").astype(float)
    new = pd.Series(new_s_nom, index=names, dtype=float) if np.isscalar(new_s_nom) \
        else pd.Series(new_s_nom, dtype=float).reindex(names)
    if (old <= 0).any() or old.isna().any() or (new <= 0).any() or new.isna().any():
        raise ValueError("Transformer ratings must be positive to rescale impedance")
    ratio = new / old
    for col in ("r", "x"):
        if col in trafos.columns:
            trafos.loc[names, col] = trafos.loc[names, col].astype(float) * ratio
    for col in ("b", "g"):
        if col in trafos.columns:
            trafos.loc[names, col] = trafos.loc[names, col].astype(float) / ratio
    trafos.loc[names, "s_nom"] = new


def branch_x_pu100(network) -> pd.DataFrame:
    """Per-branch reactance as PyPSA sees it, expressed in pu on 100 MVA.

    ``x_pu100 = 100 * x_pu`` where ``x_pu`` is PyPSA's 1 MVA per unit.  With
    correct units this equals the ETYS "% on 100 MVA" value divided by 100.
    """
    rows = []
    if len(network.lines):
        l = network.lines
        v0 = l["bus0"].map(network.buses["v_nom"]).astype(float)
        v1 = l["bus1"].map(network.buses["v_nom"]).astype(float)
        rows.append(pd.DataFrame({
            "component": "Line", "v_high": np.maximum(v0, v1), "v_low": np.minimum(v0, v1),
            "x_pu100": BASE_MVA * l["x"].astype(float) / v0 ** 2,
            "s_nom": l["s_nom"].astype(float),
        }, index=l.index))
    if len(network.transformers):
        t = network.transformers
        v0 = t["bus0"].map(network.buses["v_nom"]).astype(float)
        v1 = t["bus1"].map(network.buses["v_nom"]).astype(float)
        rows.append(pd.DataFrame({
            "component": "Transformer", "v_high": np.maximum(v0, v1), "v_low": np.minimum(v0, v1),
            "x_pu100": BASE_MVA * t["x"].astype(float) / t["s_nom"].astype(float),
            "s_nom": t["s_nom"].astype(float),
        }, index=t.index))
    return pd.concat(rows) if rows else pd.DataFrame(
        columns=["component", "v_high", "v_low", "x_pu100", "s_nom"])


def log_branch_impedance_summary(network, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Log median PyPSA reactance by class, in pu on 100 MVA, as a unit check."""
    logger = logger or logging.getLogger(__name__)
    table = branch_x_pu100(network)
    if table.empty:
        return table
    summary = table.groupby(["component", "v_high", "v_low"])["x_pu100"].agg(
        ["size", "median", "min", "max"])
    logger.info("Branch reactance as PyPSA sees it (pu on 100 MVA):\n%s", summary.to_string())
    return summary


class BranchUnitsError(ValueError):
    """A network's branch impedances are not recorded as PyPSA-native units."""


def require_pypsa_branch_units(network, context: str = "solve", network_model: Optional[str] = None,
                               logger: Optional[logging.Logger] = None) -> None:
    """Refuse to power-flow a network whose branch units are not PyPSA-native.

    Networks built before 27 Sep 2026 (ETYS and Reduced) stored pu-on-100 MVA
    values where PyPSA reads ohm (lines) or pu on s_nom (transformers); they
    carry no ``branch_impedance_units`` key.  A network with no lines and no
    transformers (the Zonal model: links only) has no impedances and passes.
    """
    logger = logger or logging.getLogger(__name__)
    n_branches = len(network.lines) + len(network.transformers)
    if n_branches == 0:
        logger.info("Branch units check (%s): no lines or transformers; nothing to check", context)
        return
    units = get_branch_units(network)
    if units == UNITS_PYPSA:
        logger.info("Branch units check (%s): %s", context, units)
        return
    model = f" ({network_model} model)" if network_model else ""
    raise BranchUnitsError(
        f"{context}: network{model} has {n_branches} lines/transformers but "
        f"network.meta['{BRANCH_UNITS_META_KEY}'] is {units!r}, not {UNITS_PYPSA!r}. "
        "It was built before the 27 Sep 2026 impedance-unit fix (branch x stored in "
        "pu on 100 MVA where PyPSA reads ohm / pu on s_nom), so its power flows are "
        "wrong. Rebuild the network (rerun the network build rule and every rule "
        "downstream of it)."
    )
