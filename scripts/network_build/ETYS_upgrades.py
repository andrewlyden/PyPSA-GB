"""
ETYS Network Upgrades Application Module

This module applies network upgrades from ETYS Appendix B to PyPSA networks.
Supports ETYS 2022, 2023 and 2024 data via the etys_file_registry.

Upgrades include:
- Circuit additions, removals, and modifications
- Transformer additions, removals, and modifications
- New HVDC links (internal GB interconnectors)

Usage:
    from scripts.network_build.ETYS_upgrades import apply_etys_network_upgrades
    network = apply_etys_network_upgrades(network, modelled_year=2035,
                                          etys_file='data/network/ETYS/...')
"""

import hashlib
import itertools
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import pypsa

# Add project root to path for imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.network_build.etys_file_registry import (
    VOLTAGE_LEVELS, ELECTRICAL_DEFAULTS, ETYS_UPGRADE_SHEETS,
    INVALID_NODE_PATTERNS, DEFAULT_RATINGS, HVDC_PLACE_NAME_ALIASES,
    HVDC_TERMINAL_ENABLING_WORKS,
)
from scripts.network_build.branch_attributes import seasonal_ratings_from_record
from scripts.network_build.branch_units import (
    UNITS_PU100, UNITS_PYPSA, get_branch_units, is_short_connection,
    convert_network_pu100_to_pypsa, convert_network_pypsa_to_pu100,
)
from scripts.network_build.appendix_b_record import (
    APPENDIX_B_RECORD_VERSION, APPENDIX_B_ROW_ID_COLUMN, CP30_DATED_WORK_COLUMN, COMPONENT_TABLE, _plain,
    assign_row_ids,
    change_fields, row_id, row_record, write_appendix_b_record,
)


# Realistic per-unit parameter ranges for validation (on 100 MVA base).
# Every branch value in this module is pu on 100 MVA (ETYS "% on 100 MVA" / 100);
# conversion to PyPSA units happens once, in branch_units, after the build.
PARAM_RANGES = {
    'line': {
        'r': (0.0001, 0.05),    # pu resistance
        'x': (0.0001, 0.10),    # pu reactance
        'b': (0.0, 5.0),        # pu susceptance
        's_nom': (50, 5000),    # MVA
    },
    'transformer': {
        'r': (0.0001, 0.02),    # pu resistance
        'x': (0.02, 0.20),      # pu reactance
        'b': (0.0, 0.10),       # pu susceptance
        's_nom': (50, 2000),    # MVA
    },
}


# ETYS Appendix B provides circuit terminals and transformer connections, but it
# does not provide the internal busbar / breaker topology of every substation.
# These are *not* generic inferred connections. Each entry below is a
# deliberately small, evidence-backed modelling representation that is needed
# to preserve a documented transmission path after an ETYS upgrade is applied.
#
# Little Horsted, commissioned in 2025, is an in-line 400 kV grid supply point
# on one of the Bolney--Ninfield circuits. ETYS 2024 Appendix B, B-2-2c,
# splits that circuit into BOLN41--LHOR4A and LHOR4B--NINF41 (both 3102 MVA),
# while B-3-2c adds two 240 MVA transformers to LHOR11. The transformer pair
# is the local GSP off-take; it is not the 400 kV through-path. NGET's project
# material independently describes Little Horsted as connected to the existing
# Bolney--Ninfield OHL and says a circuit was upgraded "coming in and out" of
# the substation. See: https://www.nationalgrid.com/national-grids-little-horsted-substation-now-operational
#
# Appendix B does not list internal busbar/breaker assets, hence LHOR4A--LHOR4B
# must be represented explicitly in the simplified PyPSA topology. Its 3102
# MVA rating is the documented OHL rating, not a new or assumed reinforcement.
# Do not add further entries without comparable source evidence.
#
# Glenglass 132 kV (F25.12, 27 Oct 2026). ETYS 2024 Appendix B B-2-2b adds the
# Sanquhar II cable as GLGL1---SANQ1- (2025, 283 MVA, R/X/B 0.0629/0.3444/6.5906 %)
# and the two Glenmuckloch circuits as GHPS1---GLGL1- (2027); the B-2-1b base has
# no GLGL1-, only the busbars GLGL1Q and GLGL1R, so the build created GLGL1- and
# the Sanquhar section tied to the rest of Glenglass only through SANQ1- and GHPS1-.
# The same sheet's 2029 and 2033 change rows name one busbar, GLGL1B, for two
# circuits: GLGL1B--SAKN1- (181, then 166 MVA, R/X/B 0.2541/0.5842/11.0847 %),
# which is the base circuit GLGL1Q--SAKN1- (166 MVA, the same R, X and B), and
# GLGL1B--SANQ1- (241, then 283 MVA, 0.0629/0.3444/6.5906 %), which is the 2025
# addition GLGL1---SANQ1-. GLGL1- is therefore the busbar the base calls GLGL1Q:
# the two are joined by a bus coupler (9,999 MVA, X 0.01 % on 100 MVA, as ETYS
# models couplers), and the change rows are reconciled onto GLGL1Q and GLGL1-
# (ETYS_CHANGE_NODE_REMAPS). GLGL1- keeps its name, so the circuits keep theirs
# (the GHPS1-_GLGL1-_ETYS_UPG_2027 rating overrides still apply).
DOCUMENTED_INTERNAL_BUS_SECTIONS = (
    {
        "correction_id": "little_horsted_2025_bus_section",
        "name": "LHOR4A_LHOR4B_ETYS_2025_BUS_SECTION",
        "bus0": "LHOR4A",
        "bus1": "LHOR4B",
        "in_service_year": 2025,
        "s_nom": 3102.0,
        # A short bus-section line must be electrically close to a busbar, but
        # non-zero impedance avoids a singular DC susceptance matrix.
        "r": 1.0e-5,
        "x": 1.0e-4,
        "b": 0.0,
        "length": 0.001,
        "source": "ETYS 2024 Appendix B B-2-2c/B-3-2c; NGET Little Horsted project",
    },
    {
        "correction_id": "glenglass_2025_terminal_reconciliation",
        "name": "GLGL1-_GLGL1Q_ETYS_2025_BUS_COUPLER",
        "bus0": "GLGL1-",
        "bus1": "GLGL1Q",
        "in_service_year": 2025,
        "s_nom": 9999.0,
        "r": 0.0,
        "x": 1.0e-4,          # 0.01 % on 100 MVA, the ETYS coupler reactance (pu on 100 MVA here)
        "b": 0.0,
        "length": 0.001,
        "source": "ETYS 2024 Appendix B B-2-1b base GLGL1Q--SAKN1-; B-2-2b GLGL1---SANQ1- (2025) and the "
                  "2029/2033 GLGL1B--SAKN1- and GLGL1B--SANQ1- change rows (one Glenglass 132 kV busbar)",
    },
)


# These are deliberately not part of DOCUMENTED_INTERNAL_BUS_SECTIONS.
# ETYS Appendix B lists the Yorkshire GREEN external circuits and transformers,
# but does not list the internal 275 kV busbar/switchgear that would connect
# MFSN21 to MFSN2A/MFSN2B. Without those connections, the four 1,100 MVA
# MONF42--MFSN21 transformers cannot reach the two 1,500 MVA
# MFSN2A/B--TADT2A/B circuits; both circuits consequently carry zero flow.
#
# National Grid describes Yorkshire GREEN as creating a second connection
# between the Knaresborough and Monk Fryston/Tadcaster routes. This supports
# testing the two internal links, but does not establish their breaker design
# or busbar rating. They are therefore an opt-in sensitivity, never a default
# network correction. The 1,500 MVA limit is deliberately conservative: it is
# the documented rating of each associated external MFSN--TADT circuit.
#
# The ``yorkshire_green_full_chain_2027`` sensitivity additionally represents
# the omitted 400 kV internal section between MONF42 and MFSN41. The published
# Yorkshire GREEN description says the four new 275/400 kV SGTs connect into a
# new Monk Fryston 400 kV substation and that the two 4YS circuits to Eggborough
# are reconfigured into it. It does not publish the busbar rating. Its 6,140
# MVA model limit is therefore the *sum* of the two documented 3,070 MVA 4YS
# circuits, rather than a claim about the physical busbar. Keep this as a
# sensitivity for testing the complete documented path, not a default fix.
TOPOLOGY_SENSITIVITY_BUS_SECTIONS = {
    "yorkshire_green_mfsn_2027": (
        {
            "name": "MFSN21_MFSN2A_YORKSHIRE_GREEN_SENSITIVITY_2027",
            "bus0": "MFSN21",
            "bus1": "MFSN2A",
            "in_service_year": 2027,
            "s_nom": 1500.0,
        },
        {
            "name": "MFSN21_MFSN2B_YORKSHIRE_GREEN_SENSITIVITY_2027",
            "bus0": "MFSN21",
            "bus1": "MFSN2B",
            "in_service_year": 2027,
            "s_nom": 1500.0,
        },
    ),
    "yorkshire_green_full_chain_2027": (
        {
            "name": "MFSN21_MFSN2A_YORKSHIRE_GREEN_SENSITIVITY_2027",
            "bus0": "MFSN21",
            "bus1": "MFSN2A",
            "in_service_year": 2027,
            "s_nom": 1500.0,
        },
        {
            "name": "MFSN21_MFSN2B_YORKSHIRE_GREEN_SENSITIVITY_2027",
            "bus0": "MFSN21",
            "bus1": "MFSN2B",
            "in_service_year": 2027,
            "s_nom": 1500.0,
        },
        {
            "name": "MONF42_MFSN41_YORKSHIRE_GREEN_SENSITIVITY_2027",
            "bus0": "MONF42",
            "bus1": "MFSN41",
            "in_service_year": 2027,
            # 2 x 3,070 MVA ETYS B-2-2c 4YS circuits. This is an aggregate
            # external-circuit rating used to avoid inventing a busbar limit.
            "s_nom": 6140.0,
        },
    ),
}
YORKSHIRE_GREEN_SENSITIVITY_SOURCE = (
    "ETYS 2024 Appendix B B-2-2c/B-3-2c; "
    "NGET Yorkshire GREEN Tadcaster/Monk Fryston project"
)


# ETYS 2024 Appendix B B-2-2c records two 2027 Yorkshire GREEN removals
# against the opposite 275 kV Monk Fryston bus section from the one used in
# the B-2-1c base topology.  The base records have the same circuit ratings
# and the three related removals form one coherent reconfiguration:
#
#   B-2-1c base             B-2-2c removal record     modelled removal
#   KNAR21--MONF21 551 MVA  KNAR21--MONF22 551 MVA     KNAR21--MONF21
#   MONF22--POPP21 682 MVA  MONF21--POPP21 682 MVA     MONF22--POPP21
#
# This is a terminal-label reconciliation only.  It neither adds capacity nor
# changes ratings, and is attempted solely when the literal Appendix-B pair is
# absent.  Keep the original record in the audit log so the data correction is
# visible and reviewable.
DOCUMENTED_TOPOLOGY_CORRECTION_IDS = {
    "little_horsted_2025_bus_section",
    "yorkshire_green_2027_terminal_reconciliation",
    # 15 Oct 2026: CONQ42 in the 2027 North Wales rows is the base busbar
    # CONQ41 (ETYS_CIRCUIT_REMOVAL_REMAPS, ETYS_ADDITION_NODE_REMAPS). On unless
    # a scenario's documented_topology_corrections sets it false.
    "connahs_quay_2027_terminal_reconciliation",
    # 27 Oct 2026 (F25.12): GLGL1- is the base busbar GLGL1Q (bus coupler in
    # DOCUMENTED_INTERNAL_BUS_SECTIONS; GLGL1B change rows in ETYS_CHANGE_NODE_REMAPS).
    "glenglass_2025_terminal_reconciliation",
}

def enabled_documented_topology_corrections(corrections=None):
    """Validate and return enabled source-backed topology corrections.

    ``None`` preserves the API default (all source-backed corrections enabled);
    workflow calls should pass the explicit config/defaults/ mapping. A correction
    the mapping does not name is enabled.
    """
    if corrections is None:
        return set(DOCUMENTED_TOPOLOGY_CORRECTION_IDS)
    if not isinstance(corrections, dict):
        raise ValueError("network.upgrades.documented_topology_corrections must be a mapping")
    unknown = set(corrections) - DOCUMENTED_TOPOLOGY_CORRECTION_IDS
    if unknown:
        raise ValueError("Unknown documented ETYS topology correction(s): " + ", ".join(sorted(unknown)))
    return {name for name in DOCUMENTED_TOPOLOGY_CORRECTION_IDS if bool(corrections.get(name, True))}

ETYS_CIRCUIT_REMOVAL_REMAPS = (
    {
        "correction_id": "yorkshire_green_2027_terminal_reconciliation",
        "year": 2027,
        "node1": "KNAR21",
        "node2": "MONF22",
        "mapped_node1": "KNAR21",
        "mapped_node2": "MONF21",
        "rating_mva": 551.0,
        "source": "ETYS 2024 Appendix B B-2-1c/B-2-2c Yorkshire GREEN terminal reconciliation",
    },
    {
        "correction_id": "yorkshire_green_2027_terminal_reconciliation",
        "year": 2027,
        "node1": "MONF21",
        "node2": "POPP21",
        "mapped_node1": "MONF22",
        "mapped_node2": "POPP21",
        "rating_mva": 682.0,
        "source": "ETYS 2024 Appendix B B-2-1c/B-2-2c Yorkshire GREEN terminal reconciliation",
    },
    # North Wales 2027 (Pentir-Bodelwyddan re-arrangement): the B-2-1c base has
    # one Connah's Quay 400 kV busbar, CONQ41, carrying both Gwynedd circuits.
    # B-2-2c removes CONQ41--GWYN4B and "CONQ42--GWYN4A" (same 2,665 MVA
    # rating). The second literal pair is absent, so CONQ41--GWYN4A survived as
    # a stub in NW2's crossing set (network audit 15 Oct 2026, issue 10d).
    {
        "correction_id": "connahs_quay_2027_terminal_reconciliation",
        "year": 2027,
        "node1": "CONQ42",
        "node2": "GWYN4A",
        "mapped_node1": "CONQ41",
        "mapped_node2": "GWYN4A",
        "rating_mva": 2665.0,
        "source": "ETYS 2024 Appendix B B-2-1c/B-2-2c Connah's Quay terminal reconciliation",
    },
)


# Additions whose Appendix B terminal names a busbar the base topology does not
# carry. Without the remap Step 1 creates the busbar as a new bus with no other
# connection, so the addition is a dead-end stub that carries no flow. The
# remap is applied only to the exact (year, terminals, rating) row listed.
#   BODE41--CONQ42 (B-2-2c, 2027, 2,665 MVA): the second Bodelwyddan-Connah's
#   Quay circuit. The base has the single 400 kV busbar CONQ41 (the AENC rows in
#   cp30_n1_accelerated_components.csv map NORM42/BRFO42 the same way), and the
#   same release's removal row also names CONQ42 for a circuit the base puts on
#   CONQ41 (ETYS_CIRCUIT_REMOVAL_REMAPS above).
ETYS_ADDITION_NODE_REMAPS = (
    {
        "correction_id": "connahs_quay_2027_terminal_reconciliation",
        "component": "circuits",
        "year": 2027,
        "node1": "BODE41",
        "node2": "CONQ42",
        "mapped_node1": "BODE41",
        "mapped_node2": "CONQ41",
        "rating_mva": 2665.0,
        "source": "ETYS 2024 Appendix B B-2-1c/B-2-2c Connah's Quay terminal reconciliation",
    },
)


# Change rows whose Appendix B terminal names a busbar the base topology does not
# carry, so the change finds no circuit (audit ``base_topology_terminal_mismatch``).
# Applied only to the exact (year, terminals, rating) row listed.
#   GLGL1B (B-2-2b, 2029 and 2033): one Glenglass 132 kV busbar carrying the base
#   GLGL1Q--SAKN1- circuit and the 2025 GLGL1---SANQ1- cable (same R/X/B as each);
#   see the Glenglass note above DOCUMENTED_INTERNAL_BUS_SECTIONS.
_GLENGLASS_SOURCE = "ETYS 2024 Appendix B B-2-1b/B-2-2b Glenglass GLGL1B terminal reconciliation"
ETYS_CHANGE_NODE_REMAPS = tuple(
    {
        "correction_id": "glenglass_2025_terminal_reconciliation",
        "component": "circuits",
        "year": year,
        "node1": "GLGL1B",
        "node2": far,
        "mapped_node1": near,
        "mapped_node2": far,
        "rating_mva": rating,
        "source": _GLENGLASS_SOURCE,
    }
    for year, far, near, rating in (
        (2029, "SAKN1-", "GLGL1Q", 181.0), (2029, "SANQ1-", "GLGL1-", 241.0),
        (2033, "SAKN1-", "GLGL1Q", 166.0), (2033, "SANQ1-", "GLGL1-", 283.0),
    )
)


def apply_addition_node_remaps(categorized: Dict, enabled_corrections=None,
                               logger: Optional[logging.Logger] = None) -> List[Dict]:
    """Rewrite the terminals of the reviewed rows in ``ETYS_ADDITION_NODE_REMAPS`` and ``ETYS_CHANGE_NODE_REMAPS``.

    Runs before missing buses are created, so no stub busbar is made, and before
    the change rows are applied, so a reconciled change finds its circuit. The
    original terminals are kept on the record (``_source_node1/2``).
    """
    logger = logger or logging.getLogger(__name__)
    enabled = enabled_documented_topology_corrections(enabled_corrections)
    applied = []
    remaps = [(r, "additions") for r in ETYS_ADDITION_NODE_REMAPS] + \
        [(r, "changes") for r in ETYS_CHANGE_NODE_REMAPS]
    for remap, bucket in remaps:
        if remap["correction_id"] not in enabled:
            continue
        for record in categorized.get(remap["component"], {}).get(bucket, []):
            try:
                year = int(record.get("Year"))
                rating = float(record.get("Winter Rating (MVA)"))
            except (TypeError, ValueError):
                continue
            ends = (str(record.get("Node1")).strip(), str(record.get("Node2")).strip())
            if year != remap["year"] or not np.isclose(rating, remap["rating_mva"]):
                continue
            if ends != (remap["node1"], remap["node2"]):
                continue
            record["_source_node1"], record["_source_node2"] = ends
            record["Node1"], record["Node2"] = remap["mapped_node1"], remap["mapped_node2"]
            record["_terminal_reconciliation"] = remap["source"]
            applied.append(record)
            logger.warning(
                "ETYS %s terminal reconciliation: %s--%s -> %s--%s (%s).", bucket[:-1],
                ends[0], ends[1], remap["mapped_node1"], remap["mapped_node2"], remap["source"],
            )
    return applied


_DUPLICATE_VALUE_FIELDS = (
    "R (% on 100 MVA)", "X (% on 100 MVA)", "B (% on 100 MVA)", "Winter Rating (MVA)",
)


def drop_reversed_duplicate_records(records: List[Dict], label: str = "",
                                    logger: Optional[logging.Logger] = None) -> List[Dict]:
    """Drop Appendix B rows that repeat another row with the terminals swapped.

    ETYS 2024 Appendix B lists some SHE assets twice, once in each orientation,
    with identical year, status, R, X, B and ratings (FETT4Q/R--KINT4J/K and
    ANSN2---CRMU2J in B-2-2a, 2027/2029). Both rows were applied, doubling the
    circuit. Parallel circuits are published in the same orientation (e.g. the
    two GRNS4L--NEDE4J cables), so only swapped-terminal repeats are dropped.
    """
    logger = logger or logging.getLogger(__name__)
    seen = set()
    kept, dropped = [], []

    def value(record, field):
        number = _as_etys_float(record.get(field))
        return round(number, 9) if number is not None else _audit_text(record.get(field))

    for record in records:
        n1 = str(record.get("Node1", "")).strip()
        n2 = str(record.get("Node2", "")).strip()
        values = tuple(value(record, field) for field in _DUPLICATE_VALUE_FIELDS)
        common = (_audit_text(record.get("Year")), _audit_text(record.get("Status")),
                  _audit_text(record.get("operator")), values)
        if n1 != n2 and (n2, n1) + common in seen and (n1, n2) + common not in seen:
            dropped.append(f"{n1}--{n2} ({common[0]})")
            seen.add((n1, n2) + common)
            continue
        seen.add((n1, n2) + common)
        kept.append(record)
    if dropped:
        logger.info(
            "  Dropped %d %s row(s) repeating another row with swapped terminals: %s",
            len(dropped), label, ", ".join(dropped),
        )
    return kept


def load_cp30_dated_works(path) -> pd.DataFrame:
    """Read the CP30-dated Appendix B works table (see ``add_cp30_dated_works``)."""
    works = pd.read_csv(path)
    required = {
        "project_code", "in_service_year", "component", "operation", "etys_sheet",
        "operator", "node1", "node2", "appendix_b_year", "occurrence", "r", "x", "b",
        "impedance_units", "winter_rating_mva", "source",
    }
    missing = required - set(works.columns)
    if missing:
        raise ValueError(f"CP30 dated-works file {path} is missing columns: {sorted(missing)}")
    bad_units = set(works["impedance_units"].astype(str)) - {"percent_on_100MVA"}
    if bad_units:
        raise ValueError(
            "CP30 dated-works rows copy Appendix B values, which are '% on 100 MVA'; "
            f"impedance_units must be percent_on_100MVA, got {sorted(bad_units)}"
        )
    if works["source"].astype(str).str.strip().eq("").any() or works["source"].isna().any():
        raise ValueError("Every CP30 dated-works row needs a source")
    bad_ops = set(works["operation"]) - {"addition", "removal", "change"}
    bad_components = set(works["component"]) - {"circuit", "transformer"}
    if bad_ops or bad_components:
        raise ValueError(f"Unsupported CP30 dated-works operation/component: {bad_ops | bad_components}")
    key = ["component", "operation", "node1", "node2", "appendix_b_year", "occurrence"]
    if works.duplicated(key).any():
        raise ValueError("CP30 dated-works rows must be unique by " + ", ".join(key))
    return works


_DATED_STATUS = {"addition": "Addition", "removal": "Removed", "change": "Change"}
_DATED_BUCKET = {"addition": "additions", "removal": "removals", "change": "changes"}


def add_cp30_dated_works(upgrades_data: Dict[str, pd.DataFrame],
                         categorized: Dict[str, Dict[str, List]],
                         modelled_year: int,
                         works: pd.DataFrame,
                         logger: Optional[logging.Logger] = None) -> List[Dict]:
    """Bring CP30-dated Appendix B rows into an earlier year's build.

    Appendix B publishes snapshot years only (2029, then 2032 for SHE and 2033
    for SPT/NGET), so works CP30 sheet N1 requires by 2030-31 sit in a later
    snapshot. Each row of ``works`` names one Appendix B record (component,
    operation, terminals, Appendix B year and occurrence among identical rows)
    with its CP30 delivery year. A row is pulled when its CP30 year is at or
    before ``modelled_year`` and its Appendix B year is after it. The record is
    taken from the workbook unchanged, after checking that its R, X, B (% on
    100 MVA) and winter rating equal the values pinned in ``works``: a changed
    workbook fails here, not silently. Pulled records are tagged
    ``_cp30_dated_work`` and applied by the ordinary addition, removal and
    change code, so they create buses and get audited like any other row.
    """
    logger = logger or logging.getLogger(__name__)
    modelled_year = int(modelled_year)
    due = works[(pd.to_numeric(works["in_service_year"]) <= modelled_year)
                & (pd.to_numeric(works["appendix_b_year"]) > modelled_year)]
    pulled: List[Dict] = []
    sheet_operator = {sheet: op for comp in ("circuits", "transformers")
                      for sheet, op in ETYS_UPGRADE_SHEETS[comp].items()}
    for _, work in due.iterrows():
        component = "circuits" if work["component"] == "circuit" else "transformers"
        df = upgrades_data.get(component)
        if df is None or len(df) == 0:
            raise ValueError(f"CP30 dated work {work['project_code']}: no Appendix B {component} rows loaded")
        if sheet_operator.get(str(work["etys_sheet"])) != str(work["operator"]):
            raise ValueError(f"CP30 dated work {work['project_code']}: sheet {work['etys_sheet']} "
                             f"is not the {work['operator']} sheet")
        years = pd.to_numeric(df["Year"], errors="coerce")
        match = df[(years == int(work["appendix_b_year"]))
                   & (df["Status"] == _DATED_STATUS[work["operation"]])
                   & (df["Node1"].astype(str).str.strip() == work["node1"])
                   & (df["Node2"].astype(str).str.strip() == work["node2"])
                   & (df["operator"] == work["operator"])]
        occurrence = int(work["occurrence"])
        label = (f"{work['project_code']} {work['operation']} {work['node1']}--{work['node2']} "
                 f"(Appendix B {int(work['appendix_b_year'])}, row {occurrence})")
        if len(match) < occurrence:
            raise ValueError(f"CP30 dated work {label} is not in the Appendix B workbook")
        record = match.iloc[occurrence - 1].to_dict()
        for field, pinned in (("R (% on 100 MVA)", "r"), ("X (% on 100 MVA)", "x"),
                              ("B (% on 100 MVA)", "b"), ("Winter Rating (MVA)", "winter_rating_mva")):
            source_value = _as_etys_float(record.get(field))
            pinned_value = _as_etys_float(work[pinned])
            if source_value is None or pinned_value is None or not np.isclose(
                    source_value, pinned_value, rtol=0.0, atol=1e-6):
                raise ValueError(f"CP30 dated work {label}: {field} is {record.get(field)!r} in Appendix B "
                                 f"but {work[pinned]!r} in the dated-works table")
        bucket = categorized[component][_DATED_BUCKET[work["operation"]]]
        already = sum(
            1 for existing in bucket
            if str(existing.get("Node1")).strip() == work["node1"]
            and str(existing.get("Node2")).strip() == work["node2"]
            and _as_etys_float(existing.get("Year")) == float(work["appendix_b_year"])
        )
        if already >= occurrence:
            logger.info("  CP30 dated work %s is already in scope (HVDC enabling works)", label)
            continue
        record["_cp30_dated_work"] = str(work["project_code"])
        record["_cp30_dated_source"] = str(work["source"])
        record["_component"] = component
        bucket.append(record)
        pulled.append(record)       # the same dict: the apply step annotates it (record_appendix_b_works)
    if pulled:
        logger.info(
            "  Pulled forward %d later-year Appendix B rows as CP30-dated works (%s)",
            len(pulled), ", ".join(sorted({r["_cp30_dated_work"] for r in pulled})),
        )
    return pulled


def check_cp30_dated_works_applied(pulled: List[Dict], failures_by_type: Dict[str, List[Dict]]) -> None:
    """Raise when a pulled CP30-dated row failed to apply (it would silently be absent)."""
    failed = {
        (component, str(f.get("node1")), str(f.get("node2")))
        for component, failures in failures_by_type.items() for f in failures
    }
    bad = [
        f"{r['_cp30_dated_work']} {r['Node1']}--{r['Node2']} ({r['Status']} {int(r['Year'])})"
        for r in pulled
        if (r["_component"], str(r["Node1"]).strip(), str(r["Node2"]).strip()) in failed
        or (r["_component"], str(r["Node2"]).strip(), str(r["Node1"]).strip()) in failed
    ]
    if bad:
        raise ValueError("CP30-dated Appendix B works failed to apply: " + "; ".join(bad))


#: Order of every recorded Appendix B operation within a build (the revert replays it).
_OP_SEQUENCE = itertools.count()


def _note_op(record: Dict, entry: Dict) -> Dict:
    entry["seq"] = next(_OP_SEQUENCE)
    record.setdefault("_ops", []).append(entry)
    return entry


def _terminals(network: pypsa.Network, component: str, name: str) -> List[str]:
    table = getattr(network, COMPONENT_TABLE[component])
    return [str(table.at[name, "bus0"]), str(table.at[name, "bus1"])]


def _note_addition(network: pypsa.Network, record: Dict, component: str, name: str) -> None:
    """Name the branch or link an addition row created and record its rating and impedance."""
    record.setdefault("_network_components", []).append((component, name))
    _note_op(record, {"op": "add", "component": component, "name": name,
                      "buses": _terminals(network, component, name),
                      "after": change_fields(network, component, name)})


def _note_removal(network: pypsa.Network, record: Dict, component: str, name: str) -> None:
    """Name the branch a removal row takes out and keep its full row and its terminal buses' rows.

    Called just before the drop, while the branch and its terminal buses are still in the
    network: the network-delivery revert (``revert_appendix_b``) puts it back as it was
    (impedances in pu on 100 MVA, as here).
    """
    record.setdefault("_network_components", []).append((component, name))
    table = getattr(network, COMPONENT_TABLE[component])
    terminals = {}
    for end in ("bus0", "bus1"):
        bus = str(table.at[name, end])
        if bus in network.buses.index:
            terminals[bus] = row_record(network, "bus", bus)
    _note_op(record, {"op": "remove", "component": component, "name": name,
                      "buses": _terminals(network, component, name),
                      "before": row_record(network, component, name), "_terminal_bus_rows": terminals})


def _note_change(network: pypsa.Network, record: Dict, component: str, name: str) -> Dict:
    """Name a branch a change row modifies and keep its fields before; the caller sets ``after``."""
    record.setdefault("_network_components", []).append((component, name))
    return _note_op(record, {"op": "change", "component": component, "name": name,
                             "buses": _terminals(network, component, name),
                             "before": change_fields(network, component, name)})


def _note_missed(record: Dict, op: str, component: str, node1: str, node2: str) -> None:
    """A change or removal row that found no branch between its terminals.

    In the build that can be because another row removed the branch first (the build
    applies every removal, of any year, before any change). Recorded with its place in
    the sequence and, for a change, the values it sets, so that a revert of the other
    row replays it on the branch it puts back (``network_delivery.plan_appendix_b_revert``).
    """
    entry = {"op": op, "component": component, "name": None, "buses": [node1, node2]}
    if op == "change_missed":
        entry["after"] = _change_row_values(record, component)
    else:
        entry["rating"] = _plain(_as_etys_float(record.get("Winter Rating (MVA)")))
    _note_op(record, entry)


def _change_row_values(record: Dict, component: str) -> Dict:
    """The fields an Appendix B change row sets (as ``apply_circuit_changes`` /
    ``apply_transformer_changes`` set them), pu on 100 MVA."""
    out = {}
    quiet = logging.getLogger(__name__ + ".missed")
    quiet.disabled = True
    new_r, new_x = _change_impedance_pu100(record, component, logger=quiet)
    if pd.notna(record.get("Winter Rating (MVA)")):
        out["s_nom"] = float(record["Winter Rating (MVA)"])
        if component == "line":
            out.update({k: _plain(v) for k, v in seasonal_ratings_from_record(record).items()})
    if new_x is not None:
        out["x"] = float(new_x)
        if new_r is not None:
            out["r"] = float(new_r)
    if pd.notna(record.get("B (% on 100 MVA)")):
        out["b"] = float(record["B (% on 100 MVA)"]) / 100.0
    return out


def _note_change_done(network: pypsa.Network, entry: Dict) -> None:
    entry["after"] = change_fields(network, entry["component"], entry["name"])


def _record_rows(categorized: Dict) -> List[Tuple[str, str, Dict]]:
    """``(kind, operation, record)`` for every in-scope row, in application order."""
    out = []
    for component, kind in (("circuits", "circuit"), ("transformers", "transformer")):
        for bucket, operation in (("additions", "addition"), ("removals", "removal"), ("changes", "change")):
            out.extend((kind, operation, r) for r in categorized.get(component, {}).get(bucket, []))
    out.extend(("hvdc", "addition", r) for r in categorized.get("hvdc", {}).get("additions", []))
    return out


def _fallback_row_id(kind: str, operation: str, record: Dict) -> str:
    node1 = record.get("_source_node1", record.get("Node1", record.get("Node 1", "")))
    node2 = record.get("_source_node2", record.get("Node2", record.get("Node 2", "")))
    year = record.get("Year", record.get("Planned from year"))
    return row_id(kind, record.get("operator", "HVDC" if kind == "hvdc" else ""), year, operation,
                  node1, node2, 1)


def record_appendix_b_works(network: pypsa.Network,
                            categorized: Dict,
                            buses_before: set,
                            pruned: List[Dict],
                            orphans: List[Dict],
                            audit: Optional[pd.DataFrame],
                            modelled_year: int,
                            etys_file: str = "",
                            logger: Optional[logging.Logger] = None) -> dict:
    """Record what every applied Appendix B row did (``network.meta['appendix_b_works']``).

    Called once the upgrades, dead-island pruning and orphan removal are done. Also tags
    the CP30-dated works (``cp30_dated_work`` on the branches they added or changed and on
    the new buses only their branches reach). See ``appendix_b_record`` for the format.
    """
    logger = logger or logging.getLogger(__name__)
    status = {}
    if audit is not None and not audit.empty and APPENDIX_B_ROW_ID_COLUMN in audit.columns:
        status = dict(zip(audit[APPENDIX_B_ROW_ID_COLUMN], audit["audit_status"]))
    rows, ops, bus_rows = [], [], {}
    removal_terminals: Dict[str, set] = {}
    seen_ids = set()
    for kind, operation, r in _record_rows(categorized):
        rid = r.get("_row_id") or _fallback_row_id(kind, operation, r)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        rows.append({
            "id": rid, "kind": kind, "operation": operation,
            "source_year": _plain(r.get("_source_year", r.get("Year", r.get("Planned from year")))),
            "node1": str(r.get("_source_node1", r.get("Node1", r.get("Node 1", "")))).strip(),
            "node2": str(r.get("_source_node2", r.get("Node2", r.get("Node 2", "")))).strip(),
            "operator": _plain(r.get("operator")),
            "winter_rating_mva": _plain(_as_etys_float(r.get("Winter Rating (MVA)"))),
            "cp30_dated_work": _plain(r.get("_cp30_dated_work")),
            "hvdc_enabling_works": _plain(r.get("_hvdc_enabling_works")),
            "audit_status": status.get(rid),
        })
        for op in r.get("_ops", []):
            entry = {k: v for k, v in op.items() if not k.startswith("_")}
            entry["row"] = rid
            ops.append(entry)
            for bus, brow in op.get("_terminal_bus_rows", {}).items():
                bus_rows.setdefault(bus, brow)
                removal_terminals.setdefault(bus, set()).add(rid)

    # Dead islands the build pruned, with the removal rows whose terminals they hold
    islands: Dict[int, List[Dict]] = {}
    for entry in pruned or []:
        islands.setdefault(entry["island"], []).append(entry)
    island_records = []
    for island, entries in sorted(islands.items()):
        buses = sorted(e["name"] for e in entries if e["component"] == "bus")
        caused_by = sorted(set().union(*(removal_terminals.get(b, set()) for b in buses))) if buses else []
        island_records.append({"island": island, "buses": buses, "caused_by": caused_by})
        for e in entries:
            if e["component"] == "bus":
                bus_rows.setdefault(e["name"], e["row"])
            else:
                ops.append({"op": "prune", "component": e["component"], "name": e["name"],
                            "before": e["row"], "island": island, "row": None, "seq": next(_OP_SEQUENCE)})
    for entry in orphans or []:
        bus_rows.setdefault(entry["name"], entry["row"])
    ops.sort(key=lambda o: o["seq"])

    record = {"version": APPENDIX_B_RECORD_VERSION, "units": UNITS_PU100, "modelled_year": int(modelled_year),
              "etys_file": Path(etys_file).name if etys_file else "", "rows": rows, "ops": ops,
              "islands": island_records, "bus_rows": bus_rows,
              "buses_added": sorted(set(network.buses.index) - set(buses_before)),
              "orphan_buses": sorted(e["name"] for e in orphans or [])}
    _tag_cp30_dated_works(network, record, buses_before)
    write_appendix_b_record(network, record)
    by_op = pd.Series([o["op"] for o in ops]).value_counts().to_dict() if ops else {}
    logger.info(
        "  Appendix B works recorded: %d rows, operations %s, %d pruned islands, %d new buses",
        len(rows), by_op, len(island_records), len(record["buses_added"]),
    )
    return record


def _tag_cp30_dated_works(network: pypsa.Network, record: dict, buses_before: set) -> None:
    """``cp30_dated_work`` on what the CP30-dated rows added or changed, and on their new buses."""
    code_of = {r["id"]: r["cp30_dated_work"] for r in record["rows"] if r.get("cp30_dated_work")}
    if not code_of:
        return

    def tag(component: str, name: str, code: str) -> None:
        table = getattr(network, COMPONENT_TABLE[component])
        if CP30_DATED_WORK_COLUMN not in table.columns:
            table[CP30_DATED_WORK_COLUMN] = pd.Series(np.nan, index=table.index, dtype=object)
        current = table.at[name, CP30_DATED_WORK_COLUMN]
        codes = [c for c in current.split(";") if c] if isinstance(current, str) else []
        if code not in codes:
            codes.append(code)
        table.at[name, CP30_DATED_WORK_COLUMN] = ";".join(codes)

    added = {}
    for op in record["ops"]:
        code = code_of.get(op.get("row"))
        if code is None or op["op"] not in ("add", "change"):
            continue
        table = getattr(network, COMPONENT_TABLE[op["component"]])
        if op["name"] in table.index:
            tag(op["component"], op["name"], code)
            if op["op"] == "add":
                added[(op["component"], op["name"])] = code
    incident: Dict[str, List] = {}
    for component in ("line", "transformer", "link"):
        table = getattr(network, COMPONENT_TABLE[component])
        for name, b0, b1 in zip(table.index, table["bus0"], table["bus1"]):
            for bus in (b0, b1):
                incident.setdefault(str(bus), []).append((component, name))
    for bus in sorted(set(network.buses.index) - set(buses_before)):
        branches = incident.get(bus, [])
        if branches and all(b in added for b in branches):
            for code in sorted({added[b] for b in branches}):
                tag("bus", bus, code)


def check_appendix_b_recorded(audit: pd.DataFrame, network: pypsa.Network, record: dict) -> None:
    """Raise unless every applied audit row's components are in the record (and dated ones tagged).

    A CP30-dated row that applied without touching any component (a skipped duplicate)
    fails too: the network-delivery revert could not undo it.
    """
    if audit is None or audit.empty or "audit_status" not in audit.columns:
        return
    touched = {}
    for op in record.get("ops", []):
        touched.setdefault(op.get("row"), set()).add((op["component"], op["name"]))
    problems = []
    applied = audit[audit["audit_status"].astype(str).str.startswith("applied")]
    for _, row in applied.iterrows():
        rid = row.get(APPENDIX_B_ROW_ID_COLUMN)
        dated = str(row.get("cp30_dated_work", "") or "").strip() not in ("", "nan")
        label = f"{rid} ({row['cp30_dated_work']})" if dated else str(rid)
        components = [c for c in str(row.get("network_components", "") or "").split("; ")
                      if c and c != "nan"]
        if not components:
            if dated:
                problems.append(f"{label}: no network component")
            continue
        for item in components:
            component, name = item.split(":", 1)
            if (component, name) not in touched.get(rid, set()):
                problems.append(f"{label}: {item} not recorded")
                continue
            if dated and row["operation"] != "removal":
                table = getattr(network, COMPONENT_TABLE[component])
                tags = table[CP30_DATED_WORK_COLUMN] if CP30_DATED_WORK_COLUMN in table.columns else None
                if name not in table.index or tags is None or \
                        str(row["cp30_dated_work"]) not in str(tags.get(name, "")).split(";"):
                    problems.append(f"{label}: {item} not tagged")
    if problems:
        raise ValueError("Appendix B rows applied without a record: " + "; ".join(problems[:20]))


def _remap_circuit_removal(record: Dict, network: pypsa.Network, enabled_corrections=None):
    """Return a reviewed terminal remap for a literal ETYS removal miss.

    A remap is allowed only for the exact year, endpoints and rating documented
    in ``ETYS_CIRCUIT_REMOVAL_REMAPS`` and only when the mapped line exists.
    """
    try:
        year = int(record.get("Year"))
        node1 = str(record.get("Node1")).strip()
        node2 = str(record.get("Node2")).strip()
        rating = float(record.get("Winter Rating (MVA)"))
    except (TypeError, ValueError):
        return None

    enabled = enabled_documented_topology_corrections(enabled_corrections)
    for remap in ETYS_CIRCUIT_REMOVAL_REMAPS:
        if remap["correction_id"] not in enabled:
            continue
        if (
            year != remap["year"]
            or (node1, node2) != (remap["node1"], remap["node2"])
            or not np.isclose(rating, remap["rating_mva"])
        ):
            continue
        matching_lines = network.lines[
            ((network.lines["bus0"] == remap["mapped_node1"])
             & (network.lines["bus1"] == remap["mapped_node2"]))
            | ((network.lines["bus0"] == remap["mapped_node2"])
               & (network.lines["bus1"] == remap["mapped_node1"]))
        ]
        if not matching_lines.empty:
            return remap, matching_lines
    return None


def stable_node_bearing_radians(node_name: str) -> float:
    """Return a process-independent pseudo-bearing for an ETYS node name.

    Python's built-in ``hash`` is salted independently for each interpreter
    process, so it must not be used for persistent network coordinates.  A
    SHA-256 digest preserves the existing visual-offset behaviour while making
    repeated scenario builds bit-for-bit reproducible for the same inputs.
    """
    digest = hashlib.sha256(str(node_name).encode("utf-8")).digest()
    degrees = int.from_bytes(digest[:8], byteorder="big", signed=False) % 360
    return np.deg2rad(degrees)


def validate_electrical_params(component_type, r, x, b, s_nom, component_id,
                                logger=None):
    """
    Warn if electrical parameters are outside realistic ranges.
    Does NOT modify values -- only logs warnings.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ranges = PARAM_RANGES.get(component_type)
    if ranges is None:
        return

    for param_name, value in [('r', r), ('x', x), ('b', b), ('s_nom', s_nom)]:
        if value is None:
            continue
        lo, hi = ranges[param_name]
        if value < lo or value > hi:
            logger.warning(
                f"  {component_type} {component_id}: {param_name}={value:.6f} "
                f"outside expected range [{lo}, {hi}]"
            )


def find_split_transmission_path_candidates(categorized_upgrades: Dict) -> List[Dict]:
    """Return upgrade patterns that may need an internal 400 kV bus section.

    ETYS circuit/transformer sheets omit internal substation switchgear. This
    audit looks only for a narrow pattern: two newly-added 400 kV line sections
    terminate at distinct new bus sections of the same substation, both sections
    connect through small off-take transformers to the same lower-voltage bus,
    and Appendix B contains no direct 400 kV circuit between the two sections.
    A pair whose transformer capacity is comparable with its line capacity is
    treated as a normal transformer-supply arrangement, not a through-path.
    Results are diagnostic candidates, *not* automatic topology changes.
    """
    circuit_additions = categorized_upgrades.get("circuits", {}).get("additions", [])
    transformer_additions = categorized_upgrades.get("transformers", {}).get("additions", [])

    lines_by_bus: Dict[str, List[Dict]] = {}
    all_circuit_pairs = set()
    for record in circuit_additions:
        node1, node2 = str(record["Node1"]).strip(), str(record["Node2"]).strip()
        all_circuit_pairs.add(frozenset((node1, node2)))
        for node in (node1, node2):
            if len(node) >= 6 and node[4] == "4" and node[-1].isalpha():
                lines_by_bus.setdefault(node, []).append(record)

    transformers_by_bus: Dict[str, List[Dict]] = {}
    for record in transformer_additions:
        node1, node2 = str(record["Node1"]).strip(), str(record["Node2"]).strip()
        for node in (node1, node2):
            if len(node) >= 6 and node[4] == "4" and node[-1].isalpha():
                transformers_by_bus.setdefault(node, []).append(record)

    grouped_buses: Dict[str, List[str]] = {}
    for bus in set(lines_by_bus) | set(transformers_by_bus):
        grouped_buses.setdefault(bus[:5], []).append(bus)

    candidates = []
    for site, buses in sorted(grouped_buses.items()):
        for bus0, bus1 in itertools.combinations(sorted(buses), 2):
            # Explicit circuit records already define the bus section; no audit
            # candidate is needed in that case.
            if frozenset((bus0, bus1)) in all_circuit_pairs:
                continue

            bus0_lines = [
                record for record in lines_by_bus.get(bus0, [])
                if str(record["Node1"]).strip()[:5] != site
                or str(record["Node2"]).strip()[:5] != site
            ]
            bus1_lines = [
                record for record in lines_by_bus.get(bus1, [])
                if str(record["Node1"]).strip()[:5] != site
                or str(record["Node2"]).strip()[:5] != site
            ]
            # An in-line split has one external transmission line at each end.
            if len(bus0_lines) != 1 or len(bus1_lines) != 1:
                continue

            def lower_voltage_buses(records, bus):
                result = set()
                for record in records:
                    node1, node2 = str(record["Node1"]).strip(), str(record["Node2"]).strip()
                    other = node2 if node1 == bus else node1
                    if len(other) >= 5 and other[4] != "4":
                        result.add(other)
                return result

            common_lower_voltage = lower_voltage_buses(
                transformers_by_bus.get(bus0, []), bus0
            ) & lower_voltage_buses(
                transformers_by_bus.get(bus1, []), bus1
            )
            if not common_lower_voltage:
                continue

            def shared_offtake_rating(records, bus):
                total = 0.0
                for record in records:
                    node1 = str(record["Node1"]).strip()
                    node2 = str(record["Node2"]).strip()
                    other = node2 if node1 == bus else node1
                    if other in common_lower_voltage:
                        total += float(record.get("Winter Rating (MVA)", 0.0) or 0.0)
                return total

            line_ratings = [
                float(record.get("Winter Rating (MVA)", np.nan))
                for record in (bus0_lines + bus1_lines)
            ]
            offtake_ratings = {
                bus0: shared_offtake_rating(transformers_by_bus.get(bus0, []), bus0),
                bus1: shared_offtake_rating(transformers_by_bus.get(bus1, []), bus1),
            }
            # A high-capacity 400/275 kV transformer bank is a supply endpoint,
            # not evidence of an omitted 400 kV through busbar. Retain only
            # small off-takes, conservatively below half the line rating.
            if (
                not np.isfinite(line_ratings).all()
                or max(offtake_ratings.values()) >= 0.5 * min(line_ratings)
            ):
                continue

            years = {
                int(record["Year"])
                for record in (bus0_lines + bus1_lines)
                if pd.notna(record.get("Year"))
            }
            if len(years) != 1:
                continue
            candidates.append(
                {
                    "site": site[:4],
                    "bus0": bus0,
                    "bus1": bus1,
                    "year": years.pop(),
                    "lower_voltage_buses": sorted(common_lower_voltage),
                    "line_ratings_mva": sorted(line_ratings),
                    "shared_offtake_ratings_mva": offtake_ratings,
                }
            )
    return candidates


def apply_documented_internal_bus_sections(
    network: pypsa.Network,
    modelled_year: int,
    logger: Optional[logging.Logger] = None,
    documented_topology_corrections=None,
) -> int:
    """Apply the small set of explicitly documented ETYS bus-section models.

    The registry above is intentionally explicit. This function must not turn
    candidates from :func:`find_split_transmission_path_candidates` into lines
    automatically, because Appendix B alone does not specify breaker status.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    enabled = enabled_documented_topology_corrections(documented_topology_corrections)
    applied = 0
    for section in DOCUMENTED_INTERNAL_BUS_SECTIONS:
        if section["correction_id"] not in enabled:
            continue
        if int(modelled_year) < section["in_service_year"]:
            continue
        name, bus0, bus1 = section["name"], section["bus0"], section["bus1"]
        if name in network.lines.index:
            continue
        if bus0 not in network.buses.index or bus1 not in network.buses.index:
            logger.warning(
                "Skipping documented ETYS bus section %s: missing bus %s",
                name,
                bus0 if bus0 not in network.buses.index else bus1,
            )
            continue
        direct = network.lines[
            ((network.lines.bus0 == bus0) & (network.lines.bus1 == bus1))
            | ((network.lines.bus0 == bus1) & (network.lines.bus1 == bus0))
        ]
        if not direct.empty:
            logger.info("Documented ETYS bus section already represented: %s", name)
            continue
        network.add(
            "Line",
            name,
            bus0=bus0,
            bus1=bus1,
            s_nom=section["s_nom"],
            r=section["r"],
            x=section["x"],
            b=section["b"],
            length=section["length"],
            carrier="AC",
            build_year=section["in_service_year"],
            etys_topology_source=section["source"],
        )
        applied += 1
        logger.info(
            "Added documented ETYS internal bus section %s (%s--%s, %.0f MVA)",
            name,
            bus0,
            bus1,
            section["s_nom"],
        )
    return applied

def apply_topology_sensitivity_bus_sections(
    network: pypsa.Network,
    modelled_year: int,
    topology_sensitivities: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Apply explicitly requested, non-default ETYS topology sensitivities.

    A sensitivity represents a transparent modelling assumption where Appendix
    B omits internal switchgear. It is separate from documented corrections:
    unknown names fail fast and no sensitivity is enabled unless a scenario
    requests it under network.upgrades.topology_sensitivities.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    requested = list(topology_sensitivities or [])
    unknown = sorted(set(requested) - set(TOPOLOGY_SENSITIVITY_BUS_SECTIONS))
    if unknown:
        raise ValueError(
            "Unknown ETYS topology sensitivity name(s): "
            + ", ".join(unknown)
        )

    applied = 0
    for sensitivity_name in requested:
        for section in TOPOLOGY_SENSITIVITY_BUS_SECTIONS[sensitivity_name]:
            if int(modelled_year) < section["in_service_year"]:
                continue
            name, bus0, bus1 = section["name"], section["bus0"], section["bus1"]
            if name in network.lines.index:
                continue
            if bus0 not in network.buses.index or bus1 not in network.buses.index:
                logger.warning(
                    "Skipping ETYS topology sensitivity %s: missing bus %s",
                    name,
                    bus0 if bus0 not in network.buses.index else bus1,
                )
                continue
            direct = network.lines[
                ((network.lines.bus0 == bus0) & (network.lines.bus1 == bus1))
                | ((network.lines.bus0 == bus1) & (network.lines.bus1 == bus0))
            ]
            if not direct.empty:
                logger.info(
                    "ETYS topology sensitivity already represented: %s", name
                )
                continue

            # Near-zero impedance models a same-substation bus section without
            # creating a singular DC susceptance matrix. It is not a new OHL.
            network.add(
                "Line",
                name,
                bus0=bus0,
                bus1=bus1,
                s_nom=section["s_nom"],
                r=1.0e-5,
                x=1.0e-4,
                b=0.0,
                length=0.001,
                carrier="AC",
                build_year=section["in_service_year"],
                etys_topology_status="sensitivity",
                etys_topology_sensitivity=sensitivity_name,
                etys_topology_source=YORKSHIRE_GREEN_SENSITIVITY_SOURCE,
            )
            applied += 1
            logger.warning(
                "Added opt-in ETYS topology sensitivity %s (%s--%s, %.0f MVA).",
                name,
                bus0,
                bus1,
                section["s_nom"],
            )
    return applied


def load_etys_upgrade_data(etys_file: str, logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """
    Load ETYS upgrade data from Appendix B 2023 Excel file.
    
    Args:
        etys_file: Path to ETYS Appendix B 2023.xlsx
        logger: Optional logger instance
        
    Returns:
        Dictionary with DataFrames for each operator/component type:
        - 'circuits': Combined circuit changes (all operators)
        - 'transformers': Combined transformer changes (all operators)
        - 'hvdc': HVDC interconnector data
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading ETYS upgrade data from {etys_file}")
    
    # Sheet-to-operator mappings from central registry
    circuit_sheets = ETYS_UPGRADE_SHEETS['circuits']
    transformer_sheets = ETYS_UPGRADE_SHEETS['transformers']
    
    # Load circuit data
    circuits = []
    for sheet_name, operator in circuit_sheets.items():
        try:
            df = pd.read_excel(etys_file, sheet_name=sheet_name, skiprows=1)
            # OFTO sheets contain leading spaces in headers such as ' Rating (MVA)'.
            # Strip those before applying the cross-edition name normalisation.
            df.columns = [str(column).strip() for column in df.columns]
            # Normalize column names to handle ETYS year-to-year naming differences
            df.rename(columns={
                'Node 1': 'Node1', 'Node 2': 'Node2',
                'R (% on 100MVA)': 'R (% on 100 MVA)',
                'X (% on 100MVA)': 'X (% on 100 MVA)',
                'B (% on 100MVA)': 'B (% on 100 MVA)',
                'OHL Length(km)': 'OHL Length (km)',
                'Cable Length(km)': 'Cable Length (km)',
                'Rating (MVA)': 'Winter Rating (MVA)',
                'Rating(MVA)': 'Winter Rating (MVA)',
            }, inplace=True)
            df['operator'] = operator
            circuits.append(df)
            logger.info(f"  Loaded {sheet_name} ({operator}): {len(df)} records")
        except Exception as e:
            logger.warning(f"  Failed to load {sheet_name}: {e}")
    
    circuits_df = pd.concat(circuits, ignore_index=True) if circuits else pd.DataFrame()
    
    # Load transformer data
    transformers = []
    for sheet_name, operator in transformer_sheets.items():
        try:
            df = pd.read_excel(etys_file, sheet_name=sheet_name, skiprows=1)
            # OFTO sheets contain leading spaces in headers such as ' Rating (MVA)'.
            # Strip those before applying the cross-edition name normalisation.
            df.columns = [str(column).strip() for column in df.columns]
            # Normalize column names to handle ETYS year-to-year naming differences
            df.rename(columns={
                'Node 1': 'Node1', 'Node 2': 'Node2',
                'R (% on 100MVA)': 'R (% on 100 MVA)',
                'X (% on 100MVA)': 'X (% on 100 MVA)',
                'B (% on 100MVA)': 'B (% on 100 MVA)',
                'Rating (MVA)': 'Winter Rating (MVA)',
                'Rating(MVA)': 'Winter Rating (MVA)',
            }, inplace=True)
            df['operator'] = operator
            transformers.append(df)
            logger.info(f"  Loaded {sheet_name} ({operator}): {len(df)} records")
        except Exception as e:
            logger.warning(f"  Failed to load {sheet_name}: {e}")
    
    transformers_df = pd.concat(transformers, ignore_index=True) if transformers else pd.DataFrame()
    
    # Load HVDC data
    hvdc_sheet = ETYS_UPGRADE_SHEETS['hvdc']
    try:
        hvdc_df = pd.read_excel(etys_file, sheet_name=hvdc_sheet, skiprows=1)
        logger.info(f"  Loaded {hvdc_sheet} (HVDC): {len(hvdc_df)} records")
    except Exception as e:
        logger.warning(f"  Failed to load HVDC data: {e}")
        hvdc_df = pd.DataFrame()
    
    return {
        'circuits': circuits_df,
        'transformers': transformers_df,
        'hvdc': hvdc_df
    }


def filter_upgrades_by_year(upgrades_data: Dict[str, pd.DataFrame], 
                            modelled_year: int,
                            logger: Optional[logging.Logger] = None) -> Dict[str, Dict[str, List]]:
    """
    Filter upgrades by modelled year and categorize by status.
    
    Args:
        upgrades_data: Dictionary with circuits, transformers, hvdc DataFrames
        modelled_year: Target year for upgrades
        logger: Optional logger instance

    Rows Appendix B repeats with the terminals swapped are kept once
    (``drop_reversed_duplicate_records``, network audit 15 Oct 2026).

    Returns:
        Dictionary with categorized upgrades:
        {
            'circuits': {'additions': [...], 'removals': [...], 'changes': [...]},
            'transformers': {'additions': [...], 'removals': [...], 'changes': [...]},
            'hvdc': {'additions': [...]}
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Filtering upgrades for year {modelled_year}")
    
    # Ensure modelled_year is an integer
    modelled_year = int(modelled_year)
    
    categorized = {
        'circuits': {'additions': [], 'removals': [], 'changes': []},
        'transformers': {'additions': [], 'removals': [], 'changes': []},
        'hvdc': {'additions': []}
    }
    
    # Process circuits
    if len(upgrades_data['circuits']) > 0:
        # Ensure Year column is numeric
        circuits_df = upgrades_data['circuits'].copy()
        circuits_df['Year'] = pd.to_numeric(circuits_df['Year'], errors='coerce')
        
        # Filter out records with invalid nodes (NaN, empty, etc.)
        circuits_df = circuits_df[
            circuits_df['Node1'].notna() & 
            circuits_df['Node2'].notna() &
            (circuits_df['Node1'].astype(str).str.strip() != '') &
            (circuits_df['Node2'].astype(str).str.strip() != '') &
            (circuits_df['Node1'].astype(str).str.lower() != 'nan') &
            (circuits_df['Node2'].astype(str).str.lower() != 'nan')
        ]
        
        circuits = circuits_df[circuits_df['Year'] <= modelled_year]
        
        for status in ['Addition', 'Removed', 'Change', 'Modify']:
            matching = circuits[circuits['Status'] == status]
            if status in ['Addition']:
                categorized['circuits']['additions'].extend(matching.to_dict('records'))
            elif status == 'Removed':
                categorized['circuits']['removals'].extend(matching.to_dict('records'))
            elif status in ['Change', 'Modify']:
                categorized['circuits']['changes'].extend(matching.to_dict('records'))
        
        logger.info(f"  Circuits: {len(categorized['circuits']['additions'])} additions, "
                   f"{len(categorized['circuits']['removals'])} removals, "
                   f"{len(categorized['circuits']['changes'])} changes")
    
    # Process transformers
    if len(upgrades_data['transformers']) > 0:
        # Ensure Year column is numeric
        transformers_df = upgrades_data['transformers'].copy()
        transformers_df['Year'] = pd.to_numeric(transformers_df['Year'], errors='coerce')
        
        # Filter out records with invalid nodes (NaN, empty, etc.)
        transformers_df = transformers_df[
            transformers_df['Node1'].notna() & 
            transformers_df['Node2'].notna() &
            (transformers_df['Node1'].astype(str).str.strip() != '') &
            (transformers_df['Node2'].astype(str).str.strip() != '') &
            (transformers_df['Node1'].astype(str).str.lower() != 'nan') &
            (transformers_df['Node2'].astype(str).str.lower() != 'nan')
        ]
        
        transformers = transformers_df[transformers_df['Year'] <= modelled_year]
        
        for status in ['Addition', 'Removed', 'Change']:
            matching = transformers[transformers['Status'] == status]
            if status == 'Addition':
                categorized['transformers']['additions'].extend(matching.to_dict('records'))
            elif status == 'Removed':
                categorized['transformers']['removals'].extend(matching.to_dict('records'))
            elif status == 'Change':
                categorized['transformers']['changes'].extend(matching.to_dict('records'))
        
        logger.info(f"  Transformers: {len(categorized['transformers']['additions'])} additions, "
                   f"{len(categorized['transformers']['removals'])} removals, "
                   f"{len(categorized['transformers']['changes'])} changes")
    
    # Process HVDC
    if len(upgrades_data['hvdc']) > 0:
        # Ensure 'Planned from year' column is numeric
        hvdc_df = upgrades_data['hvdc'].copy()
        hvdc_df['Planned from year'] = pd.to_numeric(hvdc_df['Planned from year'], errors='coerce')
        hvdc = hvdc_df[
            (hvdc_df['Existing'].str.lower() == 'no') &
            (hvdc_df['Planned from year'] <= modelled_year)
        ]
        categorized['hvdc']['additions'].extend(hvdc.to_dict('records'))
        
        logger.info(f"  HVDC: {len(categorized['hvdc']['additions'])} new interconnectors")

    for component in ('circuits', 'transformers'):
        for bucket, records in categorized[component].items():
            categorized[component][bucket] = drop_reversed_duplicate_records(
                records, f"{component[:-1]} {bucket[:-1]}", logger
            )

    return categorized


def _upgrade_length_km(record: Dict) -> float:
    """OHL plus cable length of an Appendix B circuit row (0 for transformers/TBC)."""
    total = 0.0
    for field in ('OHL Length (km)', 'Cable Length (km)'):
        value = pd.to_numeric(pd.Series([record.get(field)]), errors='coerce').iloc[0]
        if pd.notna(value):
            total += float(value)
    return total


def add_hvdc_enabling_works(upgrades_data: Dict[str, pd.DataFrame],
                            categorized: Dict[str, Dict[str, List]],
                            network: pypsa.Network,
                            modelled_year: int,
                            logger: Optional[logging.Logger] = None,
                            links=None) -> List[Dict]:
    """Pull forward the onshore terminal works of an in-scope HVDC link.

    Appendix B can date an HVDC link earlier than the AC works that build its
    converter buses. ETYS 2024 lists PSDC (Spittal--Longside, BANN4R--LOSD4R,
    2,000 MW) and the Western Isles link (LEWI4R--FNLN4R, 1,800 MW) from 2030,
    while the Longside, Lewis and Fanellan substations that create their
    converter buses are 2032 additions. Filtering by year alone leaves such a
    link with no terminals (audit: ``bus_not_found``).

    Only links named in ``links`` (default ``HVDC_TERMINAL_ENABLING_WORKS``)
    are completed. For each of their terminals that is neither in the network
    (exactly or as a same-site, same-voltage busbar, e.g. PEHE4- -> PEHE4J) nor
    created by an in-scope addition, this adds from Appendix B, restricted to
    the commissioning batch (Appendix B year) of the row that creates the
    terminal:

    1. every circuit/transformer with both ends at the terminal's substation
       (four-character site code), so the converter bus reaches the rest of the
       site (LOSD4R -> LOSD4A -> LOSD1A/1B; LEWI4R -> LEWI4J -> LEWI1J/1K);
    2. the shortest circuit(s) joining that site to a bus already in the
       network (LOSD4A/4B -> PEHE4J/4K; FNLN4A/4B -> BEAU4-, 5 km, both
       parallels). Wider reinforcements that also touch the site
       (FNLN--BING/CAAI/COFD/FAUG) are longer and stay in their own year. A
       site with no such circuit (Lewis) is joined by the HVDC link alone.

    Pulled-forward records are appended to ``categorized`` in place, tagged
    ``_hvdc_enabling_works`` so the audit reports them, and returned.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if links is None:
        links = HVDC_TERMINAL_ENABLING_WORKS
    links = {str(name).strip().lower() for name in links}
    modelled_year = int(modelled_year)
    hvdc_additions = categorized.get('hvdc', {}).get('additions', [])
    if not hvdc_additions or not links:
        return []

    def in_scope_nodes() -> set:
        nodes = set()
        for component in ('circuits', 'transformers'):
            for record in categorized[component]['additions']:
                nodes.add(str(record.get('Node1', '')).strip())
                nodes.add(str(record.get('Node2', '')).strip())
        return nodes

    def resolvable(node: str) -> bool:
        if node in network.buses.index or node in in_scope_nodes():
            return True
        # Same site and voltage digit, as _resolve_hvdc_bus step 2 (PEHE4- -> PEHE4J)
        return any(str(bus)[:5] == node[:5] for bus in network.buses.index)

    def later_additions(component: str) -> List[Dict]:
        df = upgrades_data.get(component)
        if df is None or len(df) == 0 or 'Year' not in df.columns:
            return []
        df = df.copy()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df[(df['Status'] == 'Addition') & (df['Year'] > modelled_year)]
        df = df[df['Node1'].notna() & df['Node2'].notna()]
        return df.to_dict('records')

    later = {component: later_additions(component) for component in ('circuits', 'transformers')}
    pulled: List[Dict] = []
    pulled_keys = set()

    def ends(record: Dict) -> Tuple[str, str]:
        return str(record['Node1']).strip(), str(record['Node2']).strip()

    def pull(component: str, record: Dict, link_name: str) -> None:
        key = (component, *ends(record), record.get('Year'))
        if key in pulled_keys:
            return
        pulled_keys.add(key)
        record = dict(record)
        record['_hvdc_enabling_works'] = link_name
        categorized[component]['additions'].append(record)
        pulled.append({**record, '_component': component})

    for hvdc in hvdc_additions:
        link_name = str(hvdc.get('Interconnector Name', hvdc.get('Name', 'HVDC link'))).strip()
        if link_name.lower() not in links:
            # Record why an unlisted link will fail, so the audit says
            # "terminals not yet built" rather than a generic application error.
            deferred = sorted({
                int(r['Year'])
                for field in ('Node 1', 'Node1', 'Node 2', 'Node2') if field in hvdc
                for records in later.values() for r in records
                if str(hvdc.get(field, '')).strip() in ends(r)
                and not resolvable(str(hvdc.get(field, '')).strip())
            })
            if deferred:
                hvdc['_terminals_from_year'] = deferred[0]
                logger.info(
                    f"  HVDC {link_name}: terminals are Appendix B {deferred[0]} additions and the "
                    "link is not in HVDC_TERMINAL_ENABLING_WORKS; left out of this year"
                )
            continue
        for field in ('Node 1', 'Node1', 'Node 2', 'Node2'):
            if field not in hvdc:
                continue
            node = str(hvdc.get(field, '')).strip()
            if len(node) < 5 or any(p in node.lower() for p in INVALID_NODE_PATTERNS):
                continue
            if resolvable(node):
                continue
            creating_years = [
                r['Year'] for records in later.values() for r in records if node in ends(r)
            ]
            if not creating_years:
                logger.warning(
                    f"  HVDC {link_name}: terminal {node} is not created by any Appendix B "
                    "addition; the link cannot be applied"
                )
                continue
            batch_year = min(creating_years)
            site = node[:4]
            # (1) site-internal works of the batch that creates the terminal
            for component, records in later.items():
                for record in records:
                    n1, n2 = ends(record)
                    if record['Year'] == batch_year and n1[:4] == site and n2[:4] == site:
                        pull(component, record, link_name)
            site_nodes = {n for n in in_scope_nodes() if n[:4] == site}
            # (2) shortest same-batch tie from the site to the existing network
            ties = []
            for record in later['circuits']:
                n1, n2 = ends(record)
                if record['Year'] == batch_year and (
                        (n1 in site_nodes and n2 in network.buses.index) or
                        (n2 in site_nodes and n1 in network.buses.index)):
                    ties.append(record)
            if ties:
                shortest = min(_upgrade_length_km(r) for r in ties)
                for record in ties:
                    if abs(_upgrade_length_km(record) - shortest) < 1e-9:
                        pull('circuits', record, link_name)

    if pulled:
        logger.info(
            f"  Pulled forward {len(pulled)} later-year Appendix B additions as HVDC "
            "terminal enabling works:"
        )
        for record in pulled:
            logger.info(
                f"    {record['_hvdc_enabling_works']}: {record['_component'][:-1]} "
                f"{record['Node1']}--{record['Node2']} (Appendix B year "
                f"{int(record['Year'])}, rating {record.get('Winter Rating (MVA)')})"
            )
    return pulled


# VOLTAGE_LEVELS imported from etys_file_registry (shared constant)


def add_missing_buses_from_upgrades(network: pypsa.Network,
                                    categorized_upgrades: Dict,
                                    substation_coords: Optional[Dict[str, Tuple[float, float]]] = None,
                                    logger: Optional[logging.Logger] = None,
                                    max_iterations: int = 25,
                                    bus_coords: Optional[Dict[str, Tuple[float, float]]] = None) -> int:
    """
    Add missing buses to the network that are referenced in upgrade data.

    When ETYS upgrades reference new nodes (e.g., BLYT22 for a new 275kV bus at Blyth),
    these buses need to be added to the network before circuits can be created.

    The function uses a multi-pass approach:
    0. Strategy 0: Exact site lookup in substation_coords dict (lat/lon â†’ OSGB36)
    1. Collects all node names and their connections from circuit/transformer additions
    2. Identifies which nodes don't exist in the network
    3. Infers voltage level from node name (e.g., '2' suffix = 275kV)
    4. First pass: Uses coordinates from existing bus at same site (4-char prefix match)
    5. Second pass: Uses coordinates from connected buses in the upgrade data
    6. Third pass: Uses line length to estimate position from known connected bus
    7. Final fallback: Network centroid (should rarely be needed)

    Args:
        network: PyPSA network object
        categorized_upgrades: Dictionary with 'circuits' and 'transformers' additions
        substation_coords: Optional dict mapping 4-char site code â†’ (lat, lon) in WGS84.
            When provided, used as highest-priority coordinate source (Strategy 0).
        bus_coords: Optional dict mapping bus name -> (lat, lon) in WGS84, the reviewed
            per-bus overrides (``bus_coordinate_overrides.csv``). A bus named here is
            placed there whatever the site-level strategies give.
        logger: Optional logger instance

    Returns:
        Number of buses added
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Collect all nodes referenced in additions AND their connections
    nodes_needed = set()
    node_connections = {}  # Map: node -> list of (connected_node, length_km)
    
    def is_valid_node(node_name):
        """Check if node name is valid (not a placeholder or invalid pattern)."""
        if not node_name or pd.isna(node_name):
            return False
        node_str = str(node_name).strip().lower()
        return node_str and not any(p in node_str for p in INVALID_NODE_PATTERNS)
    
    def safe_float(val, default=0.0):
        """Safely convert a value to float, handling 'TBC', NaN, etc."""
        if val is None or pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip().lower()
        if val_str in ('', 'tbc', 'n/a', 'nan', 'none'):
            return default
        try:
            return float(val_str)
        except ValueError:
            return default
    
    # From circuit additions - collect nodes AND connections
    for record in categorized_upgrades['circuits']['additions']:
        node1 = str(record.get('Node1', '')).strip()
        node2 = str(record.get('Node2', '')).strip()
        
        # Get line length (OHL + cable) - handle 'TBC' and other invalid values
        ohl_length = safe_float(record.get('OHL Length (km)', 0))
        cable_length = safe_float(record.get('Cable Length (km)', 0))
        total_length = ohl_length + cable_length
        
        if is_valid_node(node1):
            nodes_needed.add(node1)
            if node1 not in node_connections:
                node_connections[node1] = []
            if is_valid_node(node2):
                node_connections[node1].append((node2, total_length))
                
        if is_valid_node(node2):
            nodes_needed.add(node2)
            if node2 not in node_connections:
                node_connections[node2] = []
            if is_valid_node(node1):
                node_connections[node2].append((node1, total_length))
    
    # From transformer additions (transformers have no length, use 0)
    for record in categorized_upgrades['transformers']['additions']:
        node1 = str(record.get('Node1', '')).strip()
        node2 = str(record.get('Node2', '')).strip()
        
        if is_valid_node(node1):
            nodes_needed.add(node1)
            if node1 not in node_connections:
                node_connections[node1] = []
            if is_valid_node(node2):
                node_connections[node1].append((node2, 0.0))  # Transformers are co-located
                
        if is_valid_node(node2):
            nodes_needed.add(node2)
            if node2 not in node_connections:
                node_connections[node2] = []
            if is_valid_node(node1):
                node_connections[node2].append((node1, 0.0))
    
    # Remove empty strings
    nodes_needed.discard('')
    
    # Find missing nodes
    existing_buses = set(network.buses.index)
    # Sets are deliberately sorted before they enter the multi-pass coordinate
    # resolver.  The resolver mutates ``network`` and ``added_buses`` as it
    # proceeds, so iteration order otherwise changes which coordinate source is
    # available to later nodes.
    missing_nodes = sorted(nodes_needed - existing_buses)
    
    if not missing_nodes:
        logger.info("  No missing buses to add from upgrade data")
        return 0
    
    logger.info(f"  Found {len(missing_nodes)} missing buses referenced in upgrades")
    
    # Calculate network centroid for final fallback
    centroid_x = network.buses['x'].median()
    centroid_y = network.buses['y'].median()

    # Build WGS84 â†’ OSGB36 transformer once (if substation_coords provided)
    wgs_to_osgb = None
    if substation_coords or bus_coords:
        try:
            from pyproj import Transformer as ProjTransformer
            wgs_to_osgb = ProjTransformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        except ImportError:
            logger.warning("  pyproj not available â€” Strategy 0 (substation_coords) disabled")

    # Track which buses were added and their coordinates
    added_buses = {}  # node -> (x, y)
    buses_at_centroid = []  # Track buses that ended up at centroid
    
    def get_bus_coords(bus_name):
        """Get coordinates for a bus (from network or already-added buses)."""
        if bus_name in network.buses.index:
            x = network.buses.loc[bus_name, 'x']
            y = network.buses.loc[bus_name, 'y']
            # Check if this is NOT at the centroid (valid coordinates)
            if abs(x - centroid_x) > 100 or abs(y - centroid_y) > 100:
                return x, y
        if bus_name in added_buses:
            x, y = added_buses[bus_name]
            if abs(x - centroid_x) > 100 or abs(y - centroid_y) > 100:
                return x, y
        return None, None
    
    def estimate_coords_from_connection(node, connected_node, length_km):
        """
        Estimate coordinates for a node based on a connected node with known coords.
        Uses line length to add a small offset (for visualization clarity).
        """
        x, y = get_bus_coords(connected_node)
        if x is None:
            return None, None
        
        # If length is 0 or very small (transformers, short cables), co-locate
        if length_km < 0.5:
            return x, y
        
        # Add a small offset based on line length for visualization.
        # Use 1000 meters per km as rough scale (OSGB36 is in meters)
        # The bearing uses a stable digest rather than Python's process-random
        # built-in hash, so separate scenario builds receive identical coords.
        angle = stable_node_bearing_radians(node)
        offset = min(length_km * 500, 5000)  # Cap offset at 5km
        new_x = x + offset * np.cos(angle)
        new_y = y + offset * np.sin(angle)
        
        return new_x, new_y
    
    # Multi-pass approach to resolve coordinates. A chain of new buses needs
    # one pass per link; five passes (before 15 Oct 2026) left COST1-
    # (Caithness, 2029) at the network centroid in the Borders, where it
    # crossed B0-B4. The loop stops as soon as a pass adds nothing.
    max_iterations = int(max_iterations)
    for iteration in range(max_iterations):
        nodes_to_add = [n for n in missing_nodes if n not in added_buses]
        if not nodes_to_add:
            break
            
        added_this_iteration = 0
        
        for node in nodes_to_add:
            # Extract site code and voltage
            site_code = node[:4] if len(node) >= 4 else node
            voltage_suffix = node[4] if len(node) > 4 else '4'
            v_nom = VOLTAGE_LEVELS.get(voltage_suffix, 400)
            
            x, y = None, None
            coord_source = None

            # Strategy 0: Explicit substation_coords lookup (highest priority)
            if x is None and substation_coords and wgs_to_osgb:
                if site_code in substation_coords:
                    lat, lon = substation_coords[site_code]
                    osgb_x, osgb_y = wgs_to_osgb.transform(lon, lat)
                    x, y = osgb_x, osgb_y
                    coord_source = f"substation_coords ({lat:.4f}, {lon:.4f})"

            # Strategy 1: Same-site bus with valid coordinates (not at centroid)
            site_buses = network.buses[
                network.buses.index.str.startswith(site_code)
            ].sort_index()
            for _, site_bus in site_buses.iterrows():
                if abs(site_bus['x'] - centroid_x) > 100 or abs(site_bus['y'] - centroid_y) > 100:
                    x, y = site_bus['x'], site_bus['y']
                    coord_source = f"same-site bus"
                    break
            
            # Strategy 2: Already-added bus at same site
            if x is None:
                for added_node in sorted(added_buses):
                    ax, ay = added_buses[added_node]
                    if added_node[:4] == site_code:
                        if abs(ax - centroid_x) > 100 or abs(ay - centroid_y) > 100:
                            x, y = ax, ay
                            coord_source = f"added same-site bus {added_node}"
                            break
            
            # Strategy 3: Connected bus from upgrade data
            if x is None and node in node_connections:
                connections = sorted(
                    node_connections[node],
                    key=lambda item: (str(item[0]), float(item[1])),
                )
                for connected_node, length_km in connections:
                    cx, cy = estimate_coords_from_connection(node, connected_node, length_km)
                    if cx is not None:
                        x, y = cx, cy
                        coord_source = f"connected bus {connected_node} (len={length_km:.1f}km)"
                        break
            
            # Strategy -1: a reviewed per-bus override wins over every site-level source
            if bus_coords and node in bus_coords and wgs_to_osgb:
                lat, lon = bus_coords[node]
                x, y = wgs_to_osgb.transform(lon, lat)
                coord_source = f"bus_coordinate_overrides ({lat:.4f}, {lon:.4f})"

            # If we found coordinates, add the bus
            if x is not None and y is not None:
                try:
                    network.add('Bus', node,
                               v_nom=v_nom,
                               x=x,
                               y=y,
                               carrier='AC',
                               country='GB')
                    added_buses[node] = (x, y)
                    added_this_iteration += 1
                    logger.debug(f"  Added bus {node} ({v_nom}kV) at ({x:.0f}, {y:.0f}) via {coord_source}")
                except Exception as e:
                    logger.warning(f"  Failed to add bus {node}: {e}")
        
        logger.debug(f"  Iteration {iteration + 1}: added {added_this_iteration} buses")
        
        if added_this_iteration == 0:
            break  # No progress made, remaining nodes have no resolvable coordinates
    
    # Final pass: Add remaining buses at centroid (with warning)
    remaining = [n for n in missing_nodes if n not in added_buses]
    for node in remaining:
        site_code = node[:4] if len(node) >= 4 else node
        voltage_suffix = node[4] if len(node) > 4 else '4'
        v_nom = VOLTAGE_LEVELS.get(voltage_suffix, 400)
        
        try:
            network.add('Bus', node,
                       v_nom=v_nom,
                       x=centroid_x,
                       y=centroid_y,
                       carrier='AC',
                       country='GB')
            added_buses[node] = (centroid_x, centroid_y)
            buses_at_centroid.append(node)
            logger.warning(f"  Added bus {node} ({v_nom}kV) at CENTROID - no coordinate reference found")
        except Exception as e:
            logger.warning(f"  Failed to add bus {node}: {e}")
    
    buses_added = len(added_buses)
    logger.info(f"  Added {buses_added} new buses from upgrade data")
    if buses_at_centroid:
        logger.warning(f"  WARNING: {len(buses_at_centroid)} buses placed at centroid (no coords): {buses_at_centroid[:5]}{'...' if len(buses_at_centroid) > 5 else ''}")
    
    return buses_added


def _as_etys_float(value):
    """Return a numeric ETYS cell value, treating TBC and blank cells as missing."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_enabled(engineering_fallbacks) -> bool:
    return bool((engineering_fallbacks or {}).get("enabled", False))


def _fallback_rating_mva(component: str, voltage_kv, engineering_fallbacks) -> float:
    """Return an explicit, conservative rating assumption for a missing ETYS value."""
    config = engineering_fallbacks or {}
    ratings = config.get(f"{component}_rating_mva_by_voltage_kv", {}) or {}
    voltage = _as_etys_float(voltage_kv)
    if voltage is not None:
        for key, value in ratings.items():
            if _as_etys_float(key) == voltage and _as_etys_float(value) is not None:
                return float(value)
    default = config.get(f"default_{component}_rating_mva")
    if _as_etys_float(default) is not None:
        return float(default)
    return float(DEFAULT_RATINGS["line" if component == "circuit" else "transformer"])


def _resolve_addition_electrical_data(record: Dict, component: str,
                                      engineering_fallbacks) -> Tuple[Optional[Dict], List[str]]:
    """Use ETYS values first, or explicit configured defaults when enabled.

    The function never guesses values in the base case.  A fallback scenario
    must opt in via ``network.upgrades.engineering_fallbacks.enabled``.  Applied
    assumptions are retained on the source record for the row-level audit.
    """
    is_hvdc = component == "circuit" and "hvdc" in str(
        record.get("Interconnection Type", "")
    ).lower()
    parameter_fields = [] if is_hvdc else [
        "R (% on 100 MVA)", "X (% on 100 MVA)", "B (% on 100 MVA)"
    ]
    rating_field = "Winter Rating (MVA)"
    missing = [field for field in parameter_fields + [rating_field]
               if _as_etys_float(record.get(field)) is None]
    if missing and not _fallback_enabled(engineering_fallbacks):
        return None, missing

    fallback_fields = []
    circuit_type = str(record.get("Circuit Type", "")).lower()
    defaults = ELECTRICAL_DEFAULTS["cable" if "cable" in circuit_type else
                                   ("line" if component == "circuit" else "transformer")]
    values = {}
    for field, key in (("R (% on 100 MVA)", "r"),
                       ("X (% on 100 MVA)", "x"),
                       ("B (% on 100 MVA)", "b")):
        if field not in parameter_fields:
            continue
        value = _as_etys_float(record.get(field))
        if value is None:
            values[key] = float(defaults[key])
            fallback_fields.append(field)
        else:
            values[key] = value / 100.0
    rating = _as_etys_float(record.get(rating_field))
    if rating is None:
        voltage = record.get("Rated AC Voltage (kV)", record.get("Voltage(kV)"))
        rating = _fallback_rating_mva(component, voltage, engineering_fallbacks)
        fallback_fields.append(rating_field)
    values["s_nom"] = rating
    values["is_hvdc"] = is_hvdc

    if fallback_fields:
        record["_engineering_fallback"] = True
        record["_engineering_fallback_fields"] = fallback_fields
        record["_engineering_fallback_detail"] = (
            "Configured conservative defaults: " + "; ".join(fallback_fields)
        )
    return values, missing

def _resolve_upgrade_node(network: pypsa.Network, node: str, allow_alias: bool,
                          logger: logging.Logger) -> Optional[str]:
    """Return the network bus an ETYS upgrade node refers to, or None.

    ETYS names a busbar-specific node (``COCK21``: Cockenzie, 275 kV, busbar 1)
    while the base network may only carry the site's generic busbar at that
    voltage (``COCK2-``). When the specific node is absent and the generic one
    exists, the generic busbar is used and the substitution is logged. This is
    what connects Inch Cape's 2025 export cables; without it the INCO/INCW
    buses were left as a dead island with degenerate nodal prices.
    """
    if node in network.buses.index:
        return node
    if not allow_alias or len(node) != 6:
        return None
    generic = node[:5] + '-'
    if generic != node and generic in network.buses.index:
        logger.info(f"  ETYS node {node} not in network; using site busbar {generic}")
        return generic
    return None


def _branch_graph(network: pypsa.Network):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(network.buses.index)
    for comp_df in (network.lines, network.transformers, network.links):
        if len(comp_df):
            G.add_edges_from(zip(comp_df['bus0'], comp_df['bus1']))
    return G


def _main_component_buses(network: pypsa.Network) -> set:
    import networkx as nx
    G = _branch_graph(network)
    if G.number_of_nodes() == 0:
        return set()
    return set(max(nx.connected_components(G), key=len))


def prune_dead_islands(network: pypsa.Network, main_component_before: Optional[set],
                       logger: Optional[logging.Logger] = None,
                       removed_sink: Optional[List[Dict]] = None) -> List[str]:
    """Remove electrical islands that carry no generation, load or storage.

    Islands arise when ETYS removals disconnect a pocket (Robin Rigg West's
    HARK12 pocket after the 2027 Harker 275/132 kV transformer removal) or when
    an addition could not be applied. A pocket with nothing attached still
    receives a load-shedding generator later and then reports degenerate nodal
    prices (the CP30 run showed mean duals above 27,000 GBP/MWh there). Islands
    with any attached component are kept and reported instead.

    ``removed_sink``, when given, receives ``{island, component, name, row}`` for every
    branch and bus removed (``record_appendix_b_works`` keeps the islands a removal row
    cut off, so the network-delivery revert can restore them).
    """
    import networkx as nx
    if logger is None:
        logger = logging.getLogger(__name__)
    G = _branch_graph(network)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) <= 1:
        return []
    attached = set()
    for comp_df in (network.generators, network.loads, network.storage_units):
        if len(comp_df) and 'bus' in comp_df.columns:
            attached.update(comp_df['bus'].dropna())
    removed: List[str] = []
    kept = 0
    for island, comp in enumerate(components[1:]):
        if comp & attached:
            kept += 1
            continue
        reason = ("disconnected by ETYS removals"
                  if main_component_before and (comp & main_component_before)
                  else "never connected to the main network")
        buses = sorted(comp)
        for comp_name, comp_df in (("Line", network.lines),
                                   ("Transformer", network.transformers),
                                   ("Link", network.links)):
            if len(comp_df):
                touching = comp_df.index[comp_df['bus0'].isin(comp) | comp_df['bus1'].isin(comp)]
                if len(touching):
                    if removed_sink is not None:
                        removed_sink.extend(
                            {"island": island, "component": comp_name.lower(), "name": name,
                             "row": row_record(network, comp_name.lower(), name)}
                            for name in touching)
                    network.remove(comp_name, list(touching))
        if removed_sink is not None:
            removed_sink.extend({"island": island, "component": "bus", "name": bus,
                                 "row": row_record(network, "bus", bus)} for bus in buses)
        network.remove("Bus", buses)
        removed.extend(buses)
        logger.info(
            f"  Pruned dead island of {len(buses)} buses ({reason}): {', '.join(buses[:8])}"
            + (" ..." if len(buses) > 8 else "")
        )
    if kept:
        logger.info(f"  {kept} island(s) with attached components were kept")
    return removed


def apply_circuit_additions(network: pypsa.Network, 
                           additions: List[Dict],
                           logger: Optional[logging.Logger] = None,
                           engineering_fallbacks=None,
                           busbar_alias: bool = True) -> Tuple[int, int, List[Dict]]:
    """
    Add new circuits (lines) to the network.
    
    Args:
        network: PyPSA network object
        additions: List of circuit addition records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_additions, failed_additions, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    # Track line IDs to make parallel circuits unique
    line_id_counter = {}
    
    for record in additions:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            year = int(record['Year']) if pd.notna(record.get('Year')) else 2024
            
            # Skip invalid nodes (generic names, NaN, etc.)
            if any(p in node1.lower() for p in INVALID_NODE_PATTERNS) or any(p in node2.lower() for p in INVALID_NODE_PATTERNS):
                failure_details.append({'type': 'addition', 'reason': 'invalid_node_name', 'node1': node1, 'node2': node2, 'year': year})
                logger.debug(f"Skipping circuit with generic node name: {node1}-{node2}")
                failed += 1
                continue
            
            # Skip self-loops (bus-section switches, series compensators, etc.)
            # These are "Zero Length" or "SC" circuit types in ETYS that connect
            # a bus to itself.  They cannot be modelled as AC lines (zero
            # impedance â†’ singular susceptance matrix â†’ SVD failure).
            if node1 == node2:
                circuit_type = str(record.get('Circuit Type', '')).strip()
                logger.debug(
                    f"Skipping self-loop circuit {node1}-{node2} "
                    f"(type={circuit_type}): bus-section switch / SC"
                )
                failed += 1
                failure_details.append({
                    'type': 'addition', 'reason': 'self_loop',
                    'node1': node1, 'node2': node2, 'year': year,
                })
                continue
            
            # Check if buses exist (a site-busbar alias is allowed, see _resolve_upgrade_node)
            resolved1 = _resolve_upgrade_node(network, node1, busbar_alias, logger)
            resolved2 = _resolve_upgrade_node(network, node2, busbar_alias, logger)
            if resolved1 is None or resolved2 is None:
                missing = [n for n, r in ((node1, resolved1), (node2, resolved2)) if r is None]
                failure_details.append({'type': 'addition', 'reason': 'bus_not_found', 'node1': node1, 'node2': node2, 'year': year, 'missing_buses': missing})
                logger.debug(f"Skipping circuit {node1}-{node2}: bus not found ({', '.join(missing)})")
                failed += 1
                continue
            node1, node2 = resolved1, resolved2
            
            # Create unique ID with counter for parallel circuits
            base_line_id = f"{node1}_{node2}_ETYS_UPG_{year}"
            if base_line_id in line_id_counter:
                line_id_counter[base_line_id] += 1
                line_id = f"{base_line_id}_{line_id_counter[base_line_id]}"
            else:
                line_id_counter[base_line_id] = 0
                line_id = base_line_id
            
            # Skip if this exact ID already exists in network
            if line_id in network.lines.index and "hvdc" not in str(record.get("Interconnection Type", "")).lower():
                logger.debug(f"Line {line_id} already exists, skipping")
                continue

            # Check for existing parallel circuits (info/warning for visibility)
            existing_parallels = network.lines[
                ((network.lines['bus0'] == node1) & (network.lines['bus1'] == node2)) |
                ((network.lines['bus0'] == node2) & (network.lines['bus1'] == node1))
            ]
            if not existing_parallels.empty:
                n_existing = len(existing_parallels)
                ratings = existing_parallels['s_nom'].tolist()
                logger.info(
                    f"  Adding parallel circuit {node1}â€“{node2}: "
                    f"{n_existing} circuit(s) already exist (ratings: {ratings} MVA)"
                )
                if n_existing >= 4:
                    logger.warning(
                        f"  High parallel circuit count for {node1}â€“{node2}: "
                        f"{n_existing} existing + 1 new = {n_existing + 1} total"
                    )

            electrical, missing_fields = _resolve_addition_electrical_data(
                record, "circuit", engineering_fallbacks
            )
            if electrical is None:
                failure_details.append({
                    "type": "addition", "reason": "missing_electrical_data",
                    "node1": node1, "node2": node2, "year": year,
                    "missing_fields": missing_fields,
                })
                logger.debug(
                    "Skipping circuit %s-%s: ETYS electrical data missing (%s); "
                    "enable engineering_fallbacks to test stated assumptions.",
                    node1, node2, ", ".join(missing_fields),
                )
                failed += 1
                continue

            if electrical["is_hvdc"]:
                # B-2-2d includes OFTO collector HVDC routes.  They are Links,
                # not AC Lines; retain the ETYS rating and make the internal
                # transfer bidirectional, as for the base ETYS HVDC network.
                link_id = f"{line_id}_HVDC"
                if link_id not in network.links.index:
                    network.add(
                        "Link", link_id, bus0=node1, bus1=node2,
                        p_nom=electrical["s_nom"], p_min_pu=-1.0,
                        p_max_pu=1.0, efficiency=1.0, carrier="DC",
                        under_construction=False,
                    )
                    _note_addition(network, record, "link", link_id)
            else:
                r, x, b, s_nom = (
                    electrical["r"], electrical["x"], electrical["b"],
                    electrical["s_nom"],
                )
                # Zero impedances (branch_units.resolve_source_impedance_pu100
                # policy): R = 0 with X > 0 is a valid published value (series
                # compensators, SSSCs and the 0.01 % couplers all carry R = 0).
                # A published X = 0 on a zero-length or <= 1 km circuit is a bus
                # section and takes the coupler reactance.  X = 0 on a longer
                # circuit is a placeholder and needs the explicit fallback.
                if x == 0.0:
                    length_km = _upgrade_length_km(record)
                    if is_short_connection(record.get("Circuit Type", ""), length_km):
                        x = float(ELECTRICAL_DEFAULTS["coupler"]["x"])
                        record["_impedance_assumption"] = (
                            f"zero-reactance bus section ({length_km:g} km): coupler "
                            f"X = {100 * x:g} % on 100 MVA"
                        )
                    elif not _fallback_enabled(engineering_fallbacks):
                        failure_details.append({
                            "type": "addition", "reason": "missing_electrical_data",
                            "node1": node1, "node2": node2, "year": year,
                            "missing_fields": ["zero X placeholder"],
                        })
                        failed += 1
                        continue
                    else:
                        defaults = ELECTRICAL_DEFAULTS[
                            "cable" if "cable" in str(record.get("Circuit Type", "")).lower() else "line"
                        ]
                        x = defaults["x"]
                        record["_engineering_fallback"] = True
                        record["_engineering_fallback_fields"] = ["zero X placeholder"]
                        record["_engineering_fallback_detail"] = "Configured conservative defaults: zero X placeholder"
                validate_electrical_params("line", r, x, b, s_nom, line_id, logger)
                # OHL + cable length as metadata (length_km, as the base ETYS circuits carry
                # it) for the position checks; PyPSA's own `length` is left alone, and r/x/b
                # come from the row's impedances, never from a length.
                route_km = _upgrade_length_km(record)
                network.add(
                    "Line", line_id, bus0=node1, bus1=node2, r=r, x=x,
                    b=b, s_nom=s_nom, carrier="AC", under_construction=False,
                    etys_licensee=record.get("operator"),
                    length_km=route_km if route_km > 0 else np.nan,
                    **seasonal_ratings_from_record(record),
                )
                _note_addition(network, record, "line", line_id)
            successful += 1
        except Exception as e:
            failure_details.append({'type': 'addition', 'reason': 'exception', 'node1': node1, 'node2': node2, 'year': year, 'error': str(e)})
            logger.debug(f"Failed to add circuit {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_circuit_removals(network: pypsa.Network,
                          removals: List[Dict],
                          logger: Optional[logging.Logger] = None,
                          documented_topology_corrections=None) -> Tuple[int, int, List[Dict]]:
    """
    Remove circuits (lines) from the network.
    
    Args:
        network: PyPSA network object
        removals: List of circuit removal records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_removals, failed_removals, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    for record in removals:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Find line connecting these nodes (check both directions)
            matching_lines = network.lines[
                ((network.lines['bus0'] == node1) & (network.lines['bus1'] == node2)) |
                ((network.lines['bus0'] == node2) & (network.lines['bus1'] == node1))
            ]
            
            remap = None
            if len(matching_lines) == 0:
                resolved = _remap_circuit_removal(record, network, documented_topology_corrections)
                if resolved is None:
                    failure_details.append({'type': 'removal', 'reason': 'line_not_found', 'node1': node1, 'node2': node2})
                    _note_missed(record, "remove_missed", "line", node1, node2)
                    logger.debug(f"No line found for removal: {node1}-{node2}")
                    failed += 1
                    continue
                remap, matching_lines = resolved

            # Remove ONE line per removal record (not all parallel circuits).
            # If the removal record specifies a rating, prefer the closest match.
            if len(matching_lines) > 1 and pd.notna(record.get('Winter Rating (MVA)')):
                target_rating = float(record['Winter Rating (MVA)'])
                rating_diff = (matching_lines['s_nom'] - target_rating).abs()
                line_id = rating_diff.idxmin()
            else:
                line_id = matching_lines.index[0]

            _note_removal(network, record, "line", line_id)
            network.lines.drop(line_id, inplace=True)
            successful += 1
            if remap is not None:
                logger.warning(
                    "ETYS removal terminal reconciliation: %s--%s -> %s--%s "
                    "(line %s; %s).",
                    node1, node2, remap["mapped_node1"], remap["mapped_node2"],
                    line_id, remap["source"],
                )
            if len(matching_lines) > 1:
                logger.debug(f"Removed 1 of {len(matching_lines)} parallel circuits "
                             f"between {node1}-{node2} (line {line_id})")

        except Exception as e:
            failure_details.append({'type': 'removal', 'reason': 'exception', 'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to remove circuit {node1}-{node2}: {e}")
            failed += 1

    return successful, failed, failure_details


def _change_impedance_pu100(record: Dict, component: str, line_desc: str = "",
                            logger: Optional[logging.Logger] = None):
    """Return ``(r, x)`` in pu on 100 MVA for an Appendix B change row.

    ``None`` means "keep the existing value": the field is blank, or X = 0 on
    a transformer or on a circuit longer than a bus section (a placeholder, in
    which case R is not applied either). A published R = 0 is returned as 0.
    """
    logger = logger or logging.getLogger(__name__)
    r_raw = _as_etys_float(record.get('R (% on 100 MVA)'))
    x_raw = _as_etys_float(record.get('X (% on 100 MVA)'))
    r = None if r_raw is None else r_raw / 100.0
    if x_raw is None:
        return r, None
    x = x_raw / 100.0
    if x == 0.0:
        if component != "transformer" and is_short_connection(
                record.get("Circuit Type", ""), _upgrade_length_km(record)):
            return r, float(ELECTRICAL_DEFAULTS["coupler"]["x"])
        logger.warning(f"  Zero-X placeholder in {component} change {line_desc}: keeping existing R and X")
        return None, None
    return r, x


def apply_circuit_changes(network: pypsa.Network,
                         changes: List[Dict],
                         logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Modify existing circuit parameters.
    
    Args:
        network: PyPSA network object
        changes: List of circuit modification records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_modifications, failed_modifications, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    for record in changes:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Find line connecting these nodes
            matching_lines = network.lines[
                ((network.lines['bus0'] == node1) & (network.lines['bus1'] == node2)) |
                ((network.lines['bus0'] == node2) & (network.lines['bus1'] == node1))
            ]
            
            if len(matching_lines) == 0:
                failure_details.append({'type': 'change', 'reason': 'line_not_found', 'node1': node1, 'node2': node2})
                _note_missed(record, "change_missed", "line", node1, node2)
                logger.debug(f"No line found for modification: {node1}-{node2}")
                failed += 1
                continue
            
            # Update parameters (% on 100 MVA -> pu on 100 MVA), with the
            # additions' policy (branch_units.resolve_source_impedance_pu100):
            # a published R = 0 is a real value and is applied; a published
            # X = 0 on a zero-length / <= 1 km circuit is a bus section (coupler
            # X); X = 0 on a longer circuit is a placeholder, so the existing R
            # and X are kept (writing it would make the DC matrix singular).
            new_r, new_x = _change_impedance_pu100(record, "line", line_desc=f"{node1}-{node2}", logger=logger)
            seasonal = seasonal_ratings_from_record(record)
            for line_id in matching_lines.index:
                noted = _note_change(network, record, "line", line_id)
                if pd.notna(record.get('Winter Rating (MVA)')):
                    network.lines.at[line_id, 's_nom'] = float(record['Winter Rating (MVA)'])
                    # the change row's seasonal ratings replace the old ones (NaN
                    # where unpublished; completed after the build)
                    for col, value in seasonal.items():
                        network.lines.at[line_id, col] = value
                if new_x is not None:
                    network.lines.at[line_id, 'x'] = new_x
                    if new_r is not None:
                        network.lines.at[line_id, 'r'] = new_r
                if pd.notna(record.get('B (% on 100 MVA)')):
                    network.lines.at[line_id, 'b'] = float(record['B (% on 100 MVA)']) / 100.0
                _note_change_done(network, noted)
                
                successful += 1
                
        except Exception as e:
            failure_details.append({'type': 'change', 'reason': 'exception', 'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to modify circuit {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_transformer_additions(network: pypsa.Network,
                               additions: List[Dict],
                               logger: Optional[logging.Logger] = None,
                               engineering_fallbacks=None,
                               busbar_alias: bool = True) -> Tuple[int, int, List[Dict]]:
    """
    Add new transformers to the network.
    
    Args:
        network: PyPSA network object
        additions: List of transformer addition records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_additions, failed_additions, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    # Track transformer IDs to make parallel transformers unique
    xfmr_id_counter = {}
    
    for record in additions:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Skip invalid nodes (NaN, empty, generic names)
            if (node1.lower() == 'nan' or node2.lower() == 'nan' or not node1 or not node2 or
                any(p in node1.lower() for p in INVALID_NODE_PATTERNS) or
                any(p in node2.lower() for p in INVALID_NODE_PATTERNS)):
                year = int(record['Year']) if pd.notna(record.get('Year')) else 2024
                failure_details.append({'type': 'addition', 'reason': 'invalid_node_name', 'node1': node1, 'node2': node2, 'year': year})
                logger.debug(f"Skipping transformer with invalid nodes: {node1}-{node2}")
                failed += 1
                continue
                
            year = int(record['Year']) if pd.notna(record.get('Year')) else 2024
            
            # Check if buses exist (a site-busbar alias is allowed, see _resolve_upgrade_node)
            resolved1 = _resolve_upgrade_node(network, node1, busbar_alias, logger)
            resolved2 = _resolve_upgrade_node(network, node2, busbar_alias, logger)
            if resolved1 is None or resolved2 is None:
                missing = [n for n, r in ((node1, resolved1), (node2, resolved2)) if r is None]
                failure_details.append({'type': 'addition', 'reason': 'bus_not_found', 'node1': node1, 'node2': node2, 'year': year, 'missing_buses': missing})
                logger.debug(f"Skipping transformer {node1}-{node2}: bus not found ({', '.join(missing)})")
                failed += 1
                continue

            # Skip self-loops, as for circuits: a transformer from a bus to itself (e.g. the 2029
            # HIGM41-HIGM41 row) carries no flow and is not a modellable branch.
            if resolved1 == resolved2:
                failure_details.append({'type': 'addition', 'reason': 'self_loop', 'node1': node1, 'node2': node2, 'year': year})
                logger.debug(f"Skipping self-loop transformer {node1}-{node2}")
                failed += 1
                continue
            node1, node2 = resolved1, resolved2
            
            # Create unique ID with counter for parallel transformers
            base_xfmr_id = f"{node1}_{node2}_ETYS_XFMR_{year}"
            if base_xfmr_id in xfmr_id_counter:
                xfmr_id_counter[base_xfmr_id] += 1
                xfmr_id = f"{base_xfmr_id}_{xfmr_id_counter[base_xfmr_id]}"
            else:
                xfmr_id_counter[base_xfmr_id] = 0
                xfmr_id = base_xfmr_id
            
            # Skip if already exists
            if xfmr_id in network.transformers.index:
                logger.debug(f"Transformer {xfmr_id} already exists, skipping")
                continue

            # Check for existing parallel transformers (info/warning for visibility)
            existing_parallels = network.transformers[
                ((network.transformers['bus0'] == node1) & (network.transformers['bus1'] == node2)) |
                ((network.transformers['bus0'] == node2) & (network.transformers['bus1'] == node1))
            ]
            if not existing_parallels.empty:
                n_existing = len(existing_parallels)
                ratings = existing_parallels['s_nom'].tolist()
                logger.info(
                    f"  Adding parallel transformer {node1}â€“{node2}: "
                    f"{n_existing} transformer(s) already exist (ratings: {ratings} MVA)"
                )
                if n_existing >= 4:
                    logger.warning(
                        f"  High parallel transformer count for {node1}â€“{node2}: "
                        f"{n_existing} existing + 1 new = {n_existing + 1} total"
                    )

            electrical, missing_fields = _resolve_addition_electrical_data(
                record, "transformer", engineering_fallbacks
            )
            if electrical is None:
                failure_details.append({
                    "type": "addition", "reason": "missing_electrical_data",
                    "node1": node1, "node2": node2, "year": year,
                    "missing_fields": missing_fields,
                })
                logger.debug(
                    "Skipping transformer %s-%s: ETYS electrical data missing (%s); "
                    "enable engineering_fallbacks to test stated assumptions.",
                    node1, node2, ", ".join(missing_fields),
                )
                failed += 1
                continue
            r, x, b, s_nom = (
                electrical["r"], electrical["x"], electrical["b"],
                electrical["s_nom"],
            )
            # R = 0 with X > 0 is accepted (as for circuits); a transformer
            # cannot be a zero-reactance bus section, so X = 0 is a placeholder.
            if x == 0.0:
                if not _fallback_enabled(engineering_fallbacks):
                    failure_details.append({
                        "type": "addition", "reason": "missing_electrical_data",
                        "node1": node1, "node2": node2, "year": year,
                        "missing_fields": ["zero X placeholder"],
                    })
                    failed += 1
                    continue
                x = ELECTRICAL_DEFAULTS["transformer"]["x"]
                record["_engineering_fallback"] = True
                record["_engineering_fallback_fields"] = ["zero X placeholder"]
                record["_engineering_fallback_detail"] = "Configured conservative defaults: zero X placeholder"
            validate_electrical_params("transformer", r, x, b, s_nom, xfmr_id, logger)
            network.add(
                "Transformer", xfmr_id, bus0=node1, bus1=node2, model="pi",
                r=r, x=x, b=b, s_nom=s_nom, under_construction=False,
                etys_licensee=record.get("operator"),
            )
            _note_addition(network, record, "transformer", xfmr_id)
            successful += 1
        except Exception as e:
            failure_details.append({'type': 'addition', 'reason': 'exception', 'node1': node1, 'node2': node2, 'year': year, 'error': str(e)})
            logger.debug(f"Failed to add transformer {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_transformer_removals(network: pypsa.Network,
                              removals: List[Dict],
                              logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Remove transformers from the network.
    
    Args:
        network: PyPSA network object
        removals: List of transformer removal records from ETYS
        logger: Optional logger instance
        
    Returns:
        Tuple of (successful_removals, failed_removals, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    successful, failed = 0, 0
    failure_details = []
    
    for record in removals:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()
            
            # Find transformer connecting these nodes
            matching = network.transformers[
                ((network.transformers['bus0'] == node1) & (network.transformers['bus1'] == node2)) |
                ((network.transformers['bus0'] == node2) & (network.transformers['bus1'] == node1))
            ]
            
            if len(matching) == 0:
                failure_details.append({'type': 'removal', 'reason': 'transformer_not_found', 'node1': node1, 'node2': node2})
                _note_missed(record, "remove_missed", "transformer", node1, node2)
                logger.debug(f"No transformer found for removal: {node1}-{node2}")
                failed += 1
                continue

            # Remove ONE transformer per removal record (not all parallel transformers).
            # If the removal record specifies a rating, prefer the closest match.
            if len(matching) > 1 and pd.notna(record.get('Winter Rating (MVA)')):
                target_rating = float(record['Winter Rating (MVA)'])
                rating_diff = (matching['s_nom'] - target_rating).abs()
                xfmr_id = rating_diff.idxmin()
            else:
                xfmr_id = matching.index[0]

            _note_removal(network, record, "transformer", xfmr_id)
            network.transformers.drop(xfmr_id, inplace=True)
            successful += 1
            if len(matching) > 1:
                logger.debug(f"Removed 1 of {len(matching)} parallel transformers "
                             f"between {node1}-{node2} (transformer {xfmr_id})")
                
        except Exception as e:
            failure_details.append({'type': 'removal', 'reason': 'exception', 'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to remove transformer {node1}-{node2}: {e}")
            failed += 1
    
    return successful, failed, failure_details


def apply_transformer_changes(network: pypsa.Network,
                              changes: List[Dict],
                              logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Modify existing transformer parameters (rating, impedance).

    Mirrors apply_circuit_changes() but operates on network.transformers.

    Args:
        network: PyPSA network object
        changes: List of transformer modification records from ETYS
        logger: Optional logger instance

    Returns:
        Tuple of (successful_modifications, failed_modifications, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    successful, failed = 0, 0
    failure_details = []

    for record in changes:
        try:
            node1 = str(record['Node1']).strip()
            node2 = str(record['Node2']).strip()

            # Find transformer connecting these nodes (check both directions)
            matching = network.transformers[
                ((network.transformers['bus0'] == node1) & (network.transformers['bus1'] == node2)) |
                ((network.transformers['bus0'] == node2) & (network.transformers['bus1'] == node1))
            ]

            if len(matching) == 0:
                failure_details.append({'type': 'change', 'reason': 'transformer_not_found',
                                       'node1': node1, 'node2': node2})
                _note_missed(record, "change_missed", "transformer", node1, node2)
                logger.debug(f"No transformer found for modification: {node1}-{node2}")
                failed += 1
                continue

            # Update parameters (% on 100 MVA -> pu on 100 MVA; policy as for
            # circuit changes: R = 0 applied, X = 0 is a placeholder and the
            # existing impedance is kept). The build is in pu100 here, so a
            # rating change does not alter the impedance.
            new_r, new_x = _change_impedance_pu100(record, "transformer", line_desc=f"{node1}-{node2}", logger=logger)
            for xfmr_id in matching.index:
                noted = _note_change(network, record, "transformer", xfmr_id)
                if pd.notna(record.get('Winter Rating (MVA)')):
                    network.transformers.at[xfmr_id, 's_nom'] = float(record['Winter Rating (MVA)'])
                if new_x is not None:
                    network.transformers.at[xfmr_id, 'x'] = new_x
                    if new_r is not None:
                        network.transformers.at[xfmr_id, 'r'] = new_r
                if pd.notna(record.get('B (% on 100 MVA)')):
                    network.transformers.at[xfmr_id, 'b'] = float(record['B (% on 100 MVA)']) / 100.0
                _note_change_done(network, noted)

                successful += 1

        except Exception as e:
            failure_details.append({'type': 'change', 'reason': 'exception',
                                   'node1': node1, 'node2': node2, 'error': str(e)})
            logger.debug(f"Failed to modify transformer {node1}-{node2}: {e}")
            failed += 1

    return successful, failed, failure_details


def _resolve_hvdc_bus(network: pypsa.Network, node_name: str,
                     logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Resolve an HVDC node name to an existing network bus.

    ETYS B-5-1 uses generic node references like 'PEHE4-' (Peterhead 400kV)
    or 'WALP4-' (Walpole 400kV) that don't correspond exactly to the bus names
    in the ETYS network (which use suffixes like '41', '4J', '4K', etc.).

    Resolution strategy:
    0. Place-name alias â€” lookup in HVDC_PLACE_NAME_ALIASES (e.g. 'Ballantrae' â†’ 'AUCH2-')
    1. Exact match â€” return immediately
    2. Prefix match â€” find buses at the same site with the same voltage digit
       (e.g. PEHE4- â†’ PEHE41, PEHE4J, ...)
    3. Site match at any voltage â€” last resort, find any bus at the same 4-char site

    Args:
        network: PyPSA network
        node_name: Node name from ETYS B-5-1 (e.g. 'DRAX41', 'PEHE4-', 'Ballantrae')
        logger: Optional logger

    Returns:
        Resolved bus name, or None if unresolvable
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    node_name = str(node_name).strip()
    if not node_name or node_name.lower() in ('nan', '', 'tbc', 'n/a'):
        return None

    # 0. Place-name alias lookup (e.g. 'Ballantrae' â†’ 'AUCH2-')
    alias = HVDC_PLACE_NAME_ALIASES.get(node_name.lower())
    if alias:
        # Resolve the alias through the normal strategies below
        logger.debug(f"  HVDC alias: '{node_name}' â†’ '{alias}'")
        node_name = alias

    # 1. Exact match
    if node_name in network.buses.index:
        return node_name

    # 2. Prefix match: same site + same voltage digit
    #    ETYS convention: first 4 chars = site code, 5th char = voltage digit
    if len(node_name) >= 5:
        site = node_name[:4]
        voltage_digit = node_name[4]  # e.g. '4' for 400kV, '2' for 275kV
        candidates = [b for b in network.buses.index
                      if b.startswith(site) and len(b) > 4 and b[4] == voltage_digit]
        if candidates:
            # Prefer the simplest name (e.g. 'PEHE41' over 'PEHE4J')
            candidates.sort(key=len)
            resolved = candidates[0]
            logger.debug(f"  Resolved HVDC bus '{node_name}' â†’ '{resolved}' (prefix match)")
            return resolved

    # 3. Site match at any voltage (fallback)
    site = node_name[:4] if len(node_name) >= 4 else node_name
    site_buses = [b for b in network.buses.index if b.startswith(site)]
    if site_buses:
        # Prefer 400kV buses
        v400 = [b for b in site_buses if len(b) > 4 and b[4] == '4']
        if v400:
            resolved = sorted(v400, key=len)[0]
            logger.debug(f"  Resolved HVDC bus '{node_name}' â†’ '{resolved}' (site 400kV fallback)")
            return resolved
        resolved = sorted(site_buses, key=len)[0]
        logger.debug(f"  Resolved HVDC bus '{node_name}' â†’ '{resolved}' (site any-voltage fallback)")
        return resolved

    return None


def apply_hvdc_additions(network: pypsa.Network,
                         additions: List[Dict],
                         logger: Optional[logging.Logger] = None) -> Tuple[int, int, List[Dict]]:
    """
    Add new HVDC interconnectors/links to the network.

    These come from B-5-1 rows where Existing='No' and Planned from year <= modelled_year.
    Uses the same column conventions as process_ETYS_data.py for the base HVDC links.

    Args:
        network: PyPSA network object
        additions: List of HVDC addition records from ETYS B-5-1
        logger: Optional logger instance

    Returns:
        Tuple of (successful_additions, failed_additions, failure_details)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    successful, failed = 0, 0
    failure_details = []

    # Track link IDs to make parallel links unique
    link_id_counter = {}

    for record in additions:
        try:
            # B-5-1 uses 'Node 1'/'Node 2' (may have been normalized to 'Node1'/'Node2')
            node1 = str(record.get('Node 1', record.get('Node1', ''))).strip()
            node2 = str(record.get('Node 2', record.get('Node2', ''))).strip()

            if not node1 or not node2 or node1.lower() == 'nan' or node2.lower() == 'nan':
                failure_details.append({'type': 'hvdc_addition', 'reason': 'invalid_node_name',
                                       'node1': node1, 'node2': node2})
                failed += 1
                continue

            # Skip records with invalid/generic node names (offshore platforms, etc.)
            if (any(p in node1.lower() for p in INVALID_NODE_PATTERNS) or
                    any(p in node2.lower() for p in INVALID_NODE_PATTERNS)):
                failure_details.append({'type': 'hvdc_addition', 'reason': 'invalid_node_name',
                                       'node1': node1, 'node2': node2,
                                       'name': record.get('Interconnector Name', 'Unknown')})
                failed += 1
                continue

            # Resolve node names to actual network buses (handles PEHE4- â†’ PEHE4J etc.)
            bus0 = _resolve_hvdc_bus(network, node1, logger)
            bus1 = _resolve_hvdc_bus(network, node2, logger)

            if bus0 is None or bus1 is None:
                missing = [n for n, b in [(node1, bus0), (node2, bus1)] if b is None]
                failure_details.append({'type': 'hvdc_addition', 'reason': 'bus_not_found',
                                       'node1': node1, 'node2': node2, 'missing_buses': missing})
                failed += 1
                continue

            # Use resolved bus names
            node1, node2 = bus0, bus1

            year = record.get('Planned from year', 2025)
            if pd.notna(year):
                year = int(year)

            # Create unique ID
            base_link_id = f"{node1}_{node2}_HVDC_{year}"
            if base_link_id in link_id_counter:
                link_id_counter[base_link_id] += 1
                link_id = f"{base_link_id}_{link_id_counter[base_link_id]}"
            else:
                link_id_counter[base_link_id] = 0
                link_id = base_link_id

            if link_id in network.links.index:
                logger.debug(f"HVDC link {link_id} already exists, skipping")
                continue

            # Extract capacity â€” B-5-1 uses 'Winter Rating (MVA)' for power capacity
            p_nom_raw = record.get('Winter Rating (MVA)', record.get('Rating (MVA)', DEFAULT_RATINGS['hvdc']))
            p_nom = float(p_nom_raw) if pd.notna(p_nom_raw) else DEFAULT_RATINGS['hvdc']

            network.add('Link', link_id,
                       bus0=node1,
                       bus1=node2,
                       p_nom=p_nom,
                       p_min_pu=-1.0,  # Bidirectional
                       p_max_pu=1.0,
                       carrier='DC',
                       under_construction=False)
            _note_addition(network, record, "link", link_id)

            successful += 1
            logger.debug(f"Added HVDC link {link_id}: {node1}-{node2}, {p_nom:.0f} MW")

        except Exception as e:
            failure_details.append({'type': 'hvdc_addition', 'reason': 'exception',
                                   'node1': node1, 'node2': node2, 'error': str(e)})
            failed += 1

    return successful, failed, failure_details


def remove_orphan_buses(network: pypsa.Network, logger: Optional[logging.Logger] = None,
                        removed_sink: Optional[List[Dict]] = None) -> int:
    """
    Remove buses that have no attached components (orphan buses).
    
    Orphan buses can occur when:
    - Buses are added from upgrade data but their connecting components failed to add
    - Future infrastructure where only partial data is available
    
    Args:
        network: PyPSA network object
        logger: Optional logger instance
        
    Returns:
        Number of buses removed
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Find buses with no attached components â€” pre-compute connected bus sets
    # for O(1) membership lookup instead of O(B Ã— C) per-bus Series scans
    connected_buses = set()
    for comp_df in [network.generators, network.loads, network.storage_units]:
        if len(comp_df) > 0 and 'bus' in comp_df.columns:
            connected_buses.update(comp_df['bus'].dropna())
    for comp_df in [network.lines, network.transformers, network.links]:
        if len(comp_df) > 0:
            if 'bus0' in comp_df.columns:
                connected_buses.update(comp_df['bus0'].dropna())
            if 'bus1' in comp_df.columns:
                connected_buses.update(comp_df['bus1'].dropna())

    orphan_buses = [b for b in network.buses.index if b not in connected_buses]
    
    if orphan_buses:
        if removed_sink is not None:
            removed_sink.extend({"name": bus, "row": row_record(network, "bus", bus)} for bus in orphan_buses)
        logger.info(f"  Removing {len(orphan_buses)} orphan buses (no attached components)")
        logger.debug(f"  Orphan buses: {orphan_buses[:20]}" + (" ..." if len(orphan_buses) > 20 else ""))
        network.buses.drop(orphan_buses, inplace=True)
    else:
        logger.debug("  No orphan buses found")
    
    return len(orphan_buses)


def log_failure_summary(failures_by_type: Dict[str, List[Dict]], logger: logging.Logger):
    """
    Log a detailed summary of upgrade failures for investigation.
    
    Args:
        failures_by_type: Dictionary mapping component type to list of failure records
        logger: Logger instance
    """
    total_failures = sum(len(failures) for failures in failures_by_type.values())
    
    if total_failures == 0:
        logger.info("  âœ“ No failures - all upgrades applied successfully")
        return
    
    logger.info(f"\n  Failure Analysis ({total_failures} total failures):")
    logger.info("  " + "-" * 68)
    
    for component_type, failures in failures_by_type.items():
        if not failures:
            continue
            
        logger.info(f"\n  {component_type.upper()} FAILURES ({len(failures)} total):")
        
        # Group by failure reason
        by_reason = {}
        for f in failures:
            reason = f.get('reason', 'unknown')
            by_reason.setdefault(reason, []).append(f)
        
        for reason, records in sorted(by_reason.items(), key=lambda x: len(x[1]), reverse=True):
            logger.info(f"    â€¢ {reason}: {len(records)} cases")
            
            # Log a few examples
            for record in records[:3]:
                node1 = record.get('node1', 'N/A')
                node2 = record.get('node2', 'N/A')
                year = record.get('year', 'N/A')
                
                if reason == 'bus_not_found':
                    missing = ', '.join(record.get('missing_buses', []))
                    logger.debug(f"      - {node1}-{node2} ({year}): missing buses: {missing}")
                elif reason == 'exception':
                    error = record.get('error', 'unknown')
                    logger.debug(f"      - {node1}-{node2} ({year}): {error}")
                else:
                    logger.debug(f"      - {node1}-{node2} ({year})")
            
            if len(records) > 3:
                logger.debug(f"      ... and {len(records) - 3} more")



def _audit_text(value) -> str:
    """Return a stable, CSV-friendly representation of an ETYS cell value."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _is_numeric_etys_value(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def write_upgrade_audit(categorized_upgrades: Dict[str, Dict[str, List[Dict]]],
                        failures_by_type: Dict[str, List[Dict]],
                        modelled_year: int,
                        etys_file: str,
                        audit_path: Optional[str],
                        logger: logging.Logger) -> pd.DataFrame:
    """Write a row-level ETYS completeness audit and return it.

    Appendix B can contain future assets with provisional terminal names or
    ``TBC`` ratings/impedances. This audit makes each in-scope row explicit:
    applied, intentionally excluded, or needing a source-backed completion.
    It does not infer electrical parameters or add capacity to the base case.
    ``network_components`` names the branches a row added, removed or changed
    (``line:NAME; ...``) and ``cp30_dated_work`` the CP30 project of a dated row.
    The frame is built even without ``audit_path`` (the dated-works tag check
    reads it); only the write needs the path.
    """
    failure_queues: Dict[Tuple[str, str, str, str, str], List[Dict]] = {}
    for component, failures in failures_by_type.items():
        for failure in failures:
            operation = str(failure.get("type", "")).lower().replace("hvdc_", "")
            key = (
                component,
                operation,
                _audit_text(failure.get("node1")),
                _audit_text(failure.get("node2")),
                _audit_text(failure.get("year")),
            )
            failure_queues.setdefault(key, []).append(failure)

    operation_groups = (
        ("circuits", "addition", categorized_upgrades["circuits"]["additions"]),
        ("circuits", "removal", categorized_upgrades["circuits"]["removals"]),
        ("circuits", "change", categorized_upgrades["circuits"]["changes"]),
        ("transformers", "addition", categorized_upgrades["transformers"]["additions"]),
        ("transformers", "removal", categorized_upgrades["transformers"]["removals"]),
        ("transformers", "change", categorized_upgrades["transformers"]["changes"]),
        ("hvdc", "addition", categorized_upgrades["hvdc"]["additions"]),
    )
    rows = []
    for component, operation, records in operation_groups:
        for record in records:
            node1 = _audit_text(record.get("Node1", record.get("Node 1")))
            node2 = _audit_text(record.get("Node2", record.get("Node 2")))
            year = _audit_text(record.get("Year", record.get("Planned from year")))
            key = (component, operation, node1, node2, year)
            matched_failures = failure_queues.get(key, [])
            # Removal/change code paths historically did not always record the
            # source year on a failure. Match that case without losing the
            # one-row-per-source-record accounting.
            if not matched_failures:
                key_without_year = (component, operation, node1, node2, "")
                matched_failures = failure_queues.get(key_without_year, [])
            # A row that touched the network applied: its failure-queue entry (removals and
            # changes queue without a year) belongs to a same-terminal row that did not.
            applied_here = bool(record.get("_network_components"))
            failure = matched_failures.pop(0) if matched_failures and not applied_here else None
            reason = _audit_text(failure.get("reason")) if failure else ""
            error = _audit_text(failure.get("error")) if failure else ""

            invalid_terminal = any(
                token in node.lower()
                for node in (node1, node2)
                for token in INVALID_NODE_PATTERNS
            )
            parameter_fields = () if (
                component == "hvdc" or
                (component == "circuits" and "hvdc" in str(record.get("Interconnection Type", "")).lower())
            ) else (
                "R (% on 100 MVA)", "X (% on 100 MVA)", "B (% on 100 MVA)",
            )
            missing_numeric = []
            if operation == "addition":
                missing_numeric.extend(
                    field for field in parameter_fields
                    if not _is_numeric_etys_value(record.get(field))
                )
                if component != "hvdc" and not _is_numeric_etys_value(
                    record.get("Winter Rating (MVA)")
                ):
                    missing_numeric.append("Winter Rating (MVA)")

            if failure is None and record.get("_cp30_dated_work"):
                audit_status = "applied_as_cp30_dated_work"
                recommendation = (
                    f"Appendix B {year} row brought into the build for CP30 N1 "
                    f"{record['_cp30_dated_work']} ({record.get('_cp30_dated_source', '')})."
                )
            elif failure is None and record.get("_hvdc_enabling_works"):
                audit_status = "applied_as_hvdc_enabling_works"
                recommendation = (
                    f"Later-year terminal works pulled forward so {record['_hvdc_enabling_works']} "
                    "has onshore terminals; confirm they energise with the link."
                )
            elif failure is not None and record.get("_terminals_from_year"):
                audit_status = "deferred_terminals_later_year"
                recommendation = (
                    f"Converter terminals are Appendix B {record['_terminals_from_year']} additions; "
                    "the link applies from that year. Add it to HVDC_TERMINAL_ENABLING_WORKS "
                    "only if the scenario assumes it earlier."
                )
            elif failure is None and record.get("_engineering_fallback"):
                audit_status = "applied_with_engineering_fallback"
                recommendation = "Scenario-specific engineering assumption applied; review before treating as a central result."
            elif failure is None:
                audit_status = "applied"
                recommendation = "No action required."
            elif invalid_terminal:
                audit_status = "unmappable_source_terminal"
                recommendation = "Obtain the named ETYS/RDB terminal mapping before modelling this asset."
            elif missing_numeric:
                audit_status = "named_asset_missing_electrical_data"
                recommendation = "Obtain source-backed rating and electrical parameters; do not infer them in the base case."
            elif reason in {"line_not_found", "transformer_not_found"}:
                audit_status = "base_topology_terminal_mismatch"
                recommendation = "Reconcile Appendix B terminal names against the base topology before applying this row."
            elif reason == "self_loop":
                audit_status = "intentional_non_line_bus_section"
                recommendation = "Review only if source evidence identifies a required internal bus-section connection."
            else:
                audit_status = "application_error"
                recommendation = "Review this source row and the recorded application error."

            rows.append({
                "audit_status": audit_status,
                "recommendation": recommendation,
                "component": component,
                "operation": operation,
                "modelled_year": int(modelled_year),
                "source_workbook": Path(etys_file).name,
                "source_year": year,
                "node1": node1,
                "node2": node2,
                "failure_reason": reason,
                "failure_error": error,
                "missing_numeric_fields": "; ".join(missing_numeric),
                "engineering_fallback_fields": "; ".join(record.get("_engineering_fallback_fields", [])),
                "engineering_fallback_detail": _audit_text(record.get("_engineering_fallback_detail")),
                "impedance_assumption": _audit_text(record.get("_impedance_assumption")),
                "winter_rating_mva": _audit_text(record.get("Winter Rating (MVA)")),
                "r_percent_on_100_mva": _audit_text(record.get("R (% on 100 MVA)")),
                "x_percent_on_100_mva": _audit_text(record.get("X (% on 100 MVA)")),
                "b_percent_on_100_mva": _audit_text(record.get("B (% on 100 MVA)")),
                "circuit_type": _audit_text(record.get("Circuit Type")),
                "operator": _audit_text(record.get("operator")),
                "wind_farm": _audit_text(record.get("Wind Farm")),
                "cp30_dated_work": _audit_text(record.get("_cp30_dated_work")),
                APPENDIX_B_ROW_ID_COLUMN: _audit_text(record.get("_row_id")),
                "network_components": "" if failure is not None else "; ".join(
                    f"{component}:{name}" for component, name in record.get("_network_components", [])),
            })

    audit = pd.DataFrame(rows)
    if not audit_path:
        return audit
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    resolved_statuses = {"applied", "applied_with_engineering_fallback", "applied_as_hvdc_enabling_works",
                         "applied_as_cp30_dated_work", "deferred_terminals_later_year"}
    unresolved = int((~audit["audit_status"].isin(resolved_statuses)).sum()) if not audit.empty else 0
    logger.info("Wrote ETYS upgrade completeness audit: %s (%s unresolved rows)", audit_path, unresolved)
    return audit

def apply_etys_network_upgrades(network: pypsa.Network,
                               modelled_year: int,
                               etys_file: str = None,
                               substation_coords_file: Optional[str] = None,
                               topology_sensitivities: Optional[List[str]] = None,
                               documented_topology_corrections=None,
                               engineering_fallbacks=None,
                               busbar_alias_fallback: bool = True,
                               prune_dead_islands_enabled: bool = True,
                               audit_path: Optional[str] = None,
                               logger: Optional[logging.Logger] = None,
                               cp30_dated_works_file: Optional[str] = None,
                               bus_coordinate_overrides_file: Optional[str] = None) -> pypsa.Network:
    """
    Apply ETYS network upgrades from Appendix B to the network.

    This function modifies the network topology by:
    1. Adding missing buses referenced in upgrade data
    2. Adding/removing/modifying circuits commissioned by modelled_year
    3. Adding/removing/modifying transformers
    4. Applying narrowly scoped, documented internal bus sections
    5. Adding new HVDC links (from B-5-1)
    6. Removing orphan buses (no attached components)

    Args:
        network: PyPSA network object (ETYS network expected)
        modelled_year: Target year for upgrades (e.g., 2030, 2035)
        etys_file: Path to ETYS Appendix B Excel file (required)
        substation_coords_file: Optional path to substation_coordinates.csv.
            When provided, used as the highest-priority coordinate source for
            upgrade buses that have no anchor in the base network (e.g. new
            island substations like KERG, GREM, SYEL).
        topology_sensitivities: Explicit non-default ETYS topology sensitivity names.
        audit_path: Optional path for a CSV recording the treatment of every
            in-scope Appendix B upgrade row.
        logger: Optional logger instance
        cp30_dated_works_file: Optional CSV of later-year Appendix B rows that
            CP30 sheet N1 requires by the modelled year (``add_cp30_dated_works``).
            Passed only for CP30 scenarios (``network.reinforcements``).

    Returns:
        Modified PyPSA network object
    """
    if etys_file is None:
        raise ValueError(
            "etys_file must be provided. Pass the ETYS Appendix B file path "
            "from the Snakemake rule input (snakemake.input.etys_file)."
        )
    if logger is None:
        logger = logging.getLogger(__name__)

    # Upgrade rows are pu on 100 MVA. A network already in PyPSA units (a
    # saved build, applied standalone) is converted to pu100 for the upgrade
    # and back afterwards, so each branch is converted exactly once.
    input_units = get_branch_units(network)
    if input_units == UNITS_PYPSA:
        convert_network_pypsa_to_pu100(network, logger)
    elif input_units not in (None, UNITS_PU100):
        raise ValueError(f"Unknown branch impedance units {input_units!r} on network.meta")
    elif input_units is None:
        logger.warning(
            "Network records no branch impedance units; assuming pu on 100 MVA "
            "(networks built before 27 Sep 2026 stored ETYS pu100 values as PyPSA inputs)."
        )

    enabled_corrections = enabled_documented_topology_corrections(documented_topology_corrections)
    logger.info(f"Applying ETYS network upgrades for year {modelled_year}")
    logger.info("  Enabled documented topology corrections: %s", ", ".join(sorted(enabled_corrections)) or "none")
    logger.info("  Engineering fallbacks: %s", "enabled" if _fallback_enabled(engineering_fallbacks) else "disabled")

    # Load substation coordinates (Strategy 0 lookup for upgrade buses)
    substation_coords: Optional[Dict[str, Tuple[float, float]]] = None
    if substation_coords_file:
        try:
            coords_df = pd.read_csv(substation_coords_file)
            substation_coords = {
                row['site_code']: (float(row['lat']), float(row['lon']))
                for _, row in coords_df.iterrows()
                if pd.notna(row.get('lat')) and pd.notna(row.get('lon'))
            }
            logger.info(f"  Loaded {len(substation_coords)} substation coordinates from {substation_coords_file}")
        except Exception as e:
            logger.warning(f"  Failed to load substation_coords_file '{substation_coords_file}': {e}")

    # Load upgrade data
    upgrades_data = assign_row_ids(load_etys_upgrade_data(etys_file, logger))

    # Filter by year and categorize
    categorized = filter_upgrades_by_year(upgrades_data, modelled_year, logger)
    add_hvdc_enabling_works(upgrades_data, categorized, network, modelled_year, logger)
    cp30_dated_pulled: List[Dict] = []
    if cp30_dated_works_file:
        cp30_dated_pulled = add_cp30_dated_works(
            upgrades_data, categorized, modelled_year,
            load_cp30_dated_works(cp30_dated_works_file), logger,
        )
    apply_addition_node_remaps(categorized, documented_topology_corrections, logger)
    split_path_candidates = find_split_transmission_path_candidates(categorized)
    documented_pairs = {
        frozenset((section["bus0"], section["bus1"]))
        for section in DOCUMENTED_INTERNAL_BUS_SECTIONS
        if int(modelled_year) >= section["in_service_year"]
    }
    unreviewed_split_path_candidates = [
        candidate
        for candidate in split_path_candidates
        if frozenset((candidate["bus0"], candidate["bus1"])) not in documented_pairs
    ]
    
    # Track summary statistics
    summary = {
        'circuits_added': 0,
        'circuits_removed': 0,
        'circuits_modified': 0,
        'circuits_failed': 0,
        'transformers_added': 0,
        'transformers_removed': 0,
        'transformers_modified': 0,
        'transformers_failed': 0,
        'hvdc_added': 0,
        'hvdc_failed': 0,
        'buses_added': 0,
        'buses_removed': 0,
        'documented_bus_sections_added': 0,
        'unreviewed_split_path_candidates': len(unreviewed_split_path_candidates),
        'topology_sensitivity_bus_sections_added': 0,
    }
    
    # Track failure details for logging
    all_failures = {
        'circuits': [],
        'transformers': [],
        'hvdc': [],
    }
    
    # STEP 1: Add missing buses that are referenced in the upgrade data
    # This is critical - many upgrades reference new nodes that don't exist in the base network
    logger.info("Step 1: Adding missing buses from upgrade data...")
    buses_before_upgrades = set(network.buses.index)
    bus_coords = None
    if bus_coordinate_overrides_file:
        # Reviewed per-bus positions (bus position audit, 19 Oct 2026): an Appendix B bus
        # named in bus_coordinate_overrides.csv is placed there, before any other source.
        from scripts.network_build.ETYS_network import bus_coordinate_override_positions
        bus_coords = bus_coordinate_override_positions(bus_coordinate_overrides_file)
    buses_added = add_missing_buses_from_upgrades(network, categorized, substation_coords, logger,
                                                  bus_coords=bus_coords)
    summary['buses_added'] = buses_added
    
    # STEP 2: Apply circuit additions
    if categorized['circuits']['additions']:
        added, failed, failures = apply_circuit_additions(network, categorized['circuits']['additions'], logger, engineering_fallbacks, busbar_alias=busbar_alias_fallback)
        summary['circuits_added'] = added
        summary['circuits_failed'] += failed
        all_failures['circuits'].extend(failures)
        logger.info(f"Added {added} circuits ({failed} failed)")
    
    # Remember which buses were on the main network before removals so that
    # pockets disconnected by ETYS removals can be reported as such.
    main_before_removals = _main_component_buses(network)
    # Apply circuit removals
    if categorized['circuits']['removals']:
        removed, failed, failures = apply_circuit_removals(network, categorized['circuits']['removals'], logger, documented_topology_corrections)
        summary['circuits_removed'] = removed
        summary['circuits_failed'] += failed
        all_failures['circuits'].extend(failures)
        logger.info(f"Removed {removed} circuits ({failed} failed)")
    
    # Apply circuit modifications
    if categorized['circuits']['changes']:
        modified, failed, failures = apply_circuit_changes(network, categorized['circuits']['changes'], logger)
        summary['circuits_modified'] = modified
        summary['circuits_failed'] += failed
        all_failures['circuits'].extend(failures)
        logger.info(f"Modified {modified} circuits ({failed} failed)")
    
    # Apply transformer additions
    if categorized['transformers']['additions']:
        added, failed, failures = apply_transformer_additions(network, categorized['transformers']['additions'], logger, engineering_fallbacks, busbar_alias=busbar_alias_fallback)
        summary['transformers_added'] = added
        summary['transformers_failed'] += failed
        all_failures['transformers'].extend(failures)
        logger.info(f"Added {added} transformers ({failed} failed)")
    
    # Apply transformer removals
    if categorized['transformers']['removals']:
        removed, failed, failures = apply_transformer_removals(network, categorized['transformers']['removals'], logger)
        summary['transformers_removed'] = removed
        summary['transformers_failed'] += failed
        all_failures['transformers'].extend(failures)
        logger.info(f"Removed {removed} transformers ({failed} failed)")

    # Apply transformer modifications
    if categorized['transformers']['changes']:
        modified, failed, failures = apply_transformer_changes(
            network, categorized['transformers']['changes'], logger)
        summary['transformers_modified'] = modified
        summary['transformers_failed'] += failed
        all_failures['transformers'].extend(failures)
        logger.info(f"Modified {modified} transformers ({failed} failed)")

    # ETYS does not enumerate internal substation switchgear. The audit above
    # reports possible split transmission paths, but this is deliberately not
    # a generic inference: only entries in DOCUMENTED_INTERNAL_BUS_SECTIONS
    # are applied. This currently adds the source-backed Little Horsted 2025
    # 400 kV bus section, preserving the documented Bolney--Ninfield route.
    if split_path_candidates:
        for candidate in split_path_candidates:
            pair = frozenset((candidate["bus0"], candidate["bus1"]))
            if pair in documented_pairs:
                logger.info(
                    "ETYS topology audit: documented split-path correction %s--%s (%s).",
                    candidate["bus0"], candidate["bus1"], candidate["site"],
                )
            else:
                logger.warning(
                    "ETYS topology audit: possible split path %s--%s (%s) needs source review; no connection was added.",
                    candidate["bus0"], candidate["bus1"], candidate["site"],
                )
    documented_added = apply_documented_internal_bus_sections(
        network, modelled_year, logger, documented_topology_corrections
    )
    summary['documented_bus_sections_added'] = documented_added

    sensitivity_added = apply_topology_sensitivity_bus_sections(
        network, modelled_year, topology_sensitivities, logger
    )
    summary['topology_sensitivity_bus_sections_added'] = sensitivity_added

    # Apply HVDC additions (new interconnectors)
    # A link whose converter terminals are later-year additions and that is not
    # in HVDC_TERMINAL_ENABLING_WORKS stays out of this year explicitly (and is
    # audited), rather than relying on its terminal bus being absent: the
    # CP30-dated Fanellan works create FNLN4A, which the site fallback in
    # _resolve_hvdc_bus would otherwise take for the Western Isles terminal FNLN4R.
    for record in categorized['hvdc']['additions']:
        if record.get('_terminals_from_year'):
            summary['hvdc_failed'] += 1
            all_failures['hvdc'].append({
                'type': 'hvdc_addition', 'reason': 'terminals_later_year',
                'node1': str(record.get('Node 1', record.get('Node1', ''))).strip(),
                'node2': str(record.get('Node 2', record.get('Node2', ''))).strip(),
            })
    hvdc_to_apply = [r for r in categorized['hvdc']['additions']
                     if not r.get('_terminals_from_year')]
    if hvdc_to_apply:
        added, failed, failures = apply_hvdc_additions(network, hvdc_to_apply, logger)
        summary['hvdc_added'] = added
        summary['hvdc_failed'] += failed
        all_failures['hvdc'].extend(failures)
        logger.info(f"Added {added} HVDC links ({failed} failed)")

    # STEP 2b: Remove dead islands created by removals or failed additions
    pruned_rows: List[Dict] = []
    if prune_dead_islands_enabled:
        logger.info("Step 2b: Pruning dead islands (no generation, load or storage)...")
        pruned = prune_dead_islands(network, main_before_removals, logger,
                                    removed_sink=pruned_rows)
        summary['dead_island_buses_removed'] = len(pruned)
    else:
        summary['dead_island_buses_removed'] = 0
    # STEP 3: Remove orphan buses
    logger.info("Step 3: Cleaning up orphan buses...")
    orphan_rows: List[Dict] = []
    buses_removed = remove_orphan_buses(network, logger, removed_sink=orphan_rows)
    summary['buses_removed'] = buses_removed
    
    # Log detailed failure analysis
    log_failure_summary(all_failures, logger)
    audit = write_upgrade_audit(
        categorized,
        all_failures,
        modelled_year,
        etys_file,
        audit_path,
        logger,
    )
    check_cp30_dated_works_applied(cp30_dated_pulled, all_failures)
    # Record what every applied row did (and tag the CP30-dated works), so a
    # network-delivery case can undo any set of rows
    # (market.balancing.network_delivery.revert_appendix_b).
    appendix_b_record = record_appendix_b_works(
        network, categorized, buses_before_upgrades, pruned_rows, orphan_rows, audit,
        modelled_year, etys_file, logger,
    )
    check_appendix_b_recorded(audit, network, appendix_b_record)
    
    # Log summary
    logger.info("\n" + "="*70)
    logger.info(f"ETYS Network Upgrades Applied (Year {modelled_year})")
    logger.info("="*70)
    logger.info(f"  Buses Added:          {summary['buses_added']:4d}")
    logger.info(f"  Buses Removed:        {summary['buses_removed']:4d} (orphans)")
    logger.info(f"  Dead-Island Buses Removed: {summary['dead_island_buses_removed']:4d}")
    logger.info(f"  Circuits Added:       {summary['circuits_added']:4d}")
    logger.info(f"  Circuits Removed:     {summary['circuits_removed']:4d}")
    logger.info(f"  Circuits Modified:    {summary['circuits_modified']:4d}")
    logger.info(f"  Circuits Failed:      {summary['circuits_failed']:4d}")
    logger.info(f"  Transformers Added:   {summary['transformers_added']:4d}")
    logger.info(f"  Transformers Removed: {summary['transformers_removed']:4d}")
    logger.info(f"  Transformers Modified:{summary['transformers_modified']:4d}")
    logger.info(f"  Transformers Failed:  {summary['transformers_failed']:4d}")
    logger.info(f"  Documented Bus Sections Added: {summary['documented_bus_sections_added']:2d}")
    logger.info(f"  Unreviewed Split-Path Candidates: {summary['unreviewed_split_path_candidates']:2d}")
    logger.info(f"  Opt-In Sensitivity Bus Sections Added: {summary['topology_sensitivity_bus_sections_added']:2d}")
    logger.info(f"  HVDC Links Added:     {summary['hvdc_added']:4d}")
    logger.info(f"  HVDC Links Failed:    {summary['hvdc_failed']:4d}")
    logger.info(f"\nNetwork Summary:")
    logger.info(f"  Buses: {len(network.buses)}")
    logger.info(f"  Lines: {len(network.lines)}")
    logger.info(f"  Transformers: {len(network.transformers)}")
    logger.info(f"  Links: {len(network.links)}")
    logger.info("="*70 + "\n")
    
    if input_units == UNITS_PYPSA:
        convert_network_pu100_to_pypsa(network, logger)
    return network


if __name__ == "__main__":
    # Example usage
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load a network and apply upgrades
    network = pypsa.Network("resources/network/HT35_network.nc")
    network_upgraded = apply_etys_network_upgrades(
        network, 2035,
        etys_file="data/network/ETYS/ETYS 2024 Appendix-B V1.xlsx",
        logger=logger,
    )
    network_upgraded.export_to_netcdf("resources/network/HT35_network_upgraded.nc")
