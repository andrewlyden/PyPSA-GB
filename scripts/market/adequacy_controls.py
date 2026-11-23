"""Controlled transmission-relaxation tools for adequacy diagnosis."""
from __future__ import annotations
import logging
from typing import Any
import pandas as pd

from scripts.network_build.branch_units import set_transformer_s_nom_preserving_impedance


def apply_adequacy_relaxation(network, adequacy_config: dict | None, logger=None) -> pd.DataFrame:
    """Apply an explicit diagnostic-only AC capacity relaxation and audit it."""
    logger = logger or logging.getLogger(__name__)
    cfg = (adequacy_config or {}).get("relaxation", {}) or {}
    mode = str(cfg.get("mode", "none")).lower()
    capacity = float(cfg.get("capacity_mw", 1.0e6))
    if mode not in {"none", "all_ac", "components"}:
        raise ValueError("topology_adequacy.relaxation.mode must be none, all_ac, or components")
    if capacity <= 0:
        raise ValueError("topology_adequacy.relaxation.capacity_mw must be positive")
    requested = {
        "lines": set(map(str, cfg.get("lines", []) or [])),
        "transformers": set(map(str, cfg.get("transformers", []) or [])),
    }
    if mode == "components" and not (requested["lines"] or requested["transformers"]):
        raise ValueError("components relaxation requires lines and/or transformers")
    rows: list[dict[str, Any]] = []
    for key, table_name, component_type in (
        ("lines", "lines", "line"),
        ("transformers", "transformers", "transformer"),
    ):
        table = getattr(network, table_name)
        selected = table.index if mode == "all_ac" else table.index.intersection(requested[key])
        missing = requested[key] - set(table.index)
        if missing:
            raise ValueError(f"Unknown {component_type} adequacy-relaxation component(s): " + ", ".join(sorted(missing)))
        for name in selected:
            original = float(table.at[name, "s_nom"])
            applied = max(original, capacity)
            if component_type == "transformer":
                # x is per unit on s_nom: keep the physical impedance.
                set_transformer_s_nom_preserving_impedance(network, [name], applied)
            else:
                table.at[name, "s_nom"] = applied
            rows.append({"mode": mode, "component_type": component_type, "component": str(name), "bus0": str(table.at[name, "bus0"]), "bus1": str(table.at[name, "bus1"]), "original_s_nom_MW": original, "applied_s_nom_MW": applied, "changed": bool(applied != original), "diagnostic_only": True})
    result = pd.DataFrame(rows, columns=["mode", "component_type", "component", "bus0", "bus1", "original_s_nom_MW", "applied_s_nom_MW", "changed", "diagnostic_only"])
    if mode != "none":
        logger.warning("Adequacy diagnostic relaxation '%s' applied to %s AC components; this is not a network reinforcement.", mode, int(result["changed"].sum()) if not result.empty else 0)
    return result
