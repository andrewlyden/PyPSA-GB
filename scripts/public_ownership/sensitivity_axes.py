"""Compatibility shim: the item-D1 axis vocabulary over the lever module (item K1).

The six structural axes became levers in :mod:`scripts.public_ownership.levers`,
alongside the ownership, cost-of-capital, acquisition, recycling and market
levers of plan Part 6.3.  Everything importable from here before is still
importable, with the same meaning:

* ``AXES`` is the lever catalogue, ``CENTRAL_SETTINGS`` the private baseline's
  settings, ``REGISTER_AXES`` the levers that change the register;
* ``parse_axes`` reads the retired ``sensitivity_axes`` declaration and
  ``expand_cases`` accepts either that shape (a list of case names plus the
  parsed axes) or the new one (scenario bundles plus parsed sensitivities).

New code should import from ``levers`` directly.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from scripts.public_ownership.levers import (  # noqa: F401
    AXES, AXIS_CFD_AMORTISATION, AXIS_CFD_STRIKE_BASIS, AXIS_LDES_FLOOR_BASIS, AXIS_NEW_BUILD_CONTRACTS,
    AXIS_OWNERSHIP_BOUNDARY, AXIS_ROUTE_BOOKING, BOUNDING_PAIR_LABEL, CENTRAL_SETTINGS, LEVERS,
    REGISTER_AXES, SCOPE_ALL, SCOPE_PUBLIC, SCOPES, VARIANT_SEPARATOR, AxisConfigError, CaseSpec,
    Lever as Axis, LeverConfigError, all_cases_axes, axis_setting, base_case, comparator_case_id,
    comparator_for_settings, expand_axis_cases, is_variant, parse_axes, settings_columns, variant_label,
)
from scripts.public_ownership.levers import expand_cases as _expand_lever_cases


def expand_cases(cases_or_scenarios, axes_or_sensitivities: Mapping | None = None, *, bounding_pair: bool = True):
    """Expand either the D1 shape (case names + parsed axes) or the K1 shape (bundles + sensitivities)."""
    if isinstance(cases_or_scenarios, Mapping):
        sensitivities = dict(axes_or_sensitivities or {})
        sensitivities.setdefault("bounding_pair", bounding_pair)
        return _expand_lever_cases(cases_or_scenarios, sensitivities)
    return expand_axis_cases(list(cases_or_scenarios), axes_or_sensitivities or {}, bounding_pair=bounding_pair)
