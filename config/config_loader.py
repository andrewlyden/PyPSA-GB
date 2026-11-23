"""
Configuration Loading Utility for PyPSA-GB

Handles the hierarchical configuration system, merged in this order:
  1. defaults/base.yaml + defaults/base/**  - the base defaults, complete and neutral (file path = key path)
  2. config.yaml                            - run selection + global overrides
  3. defaults/method/<method>.yaml          - the build method's overlay (attached by ``method``)
  4. defaults/source/<source>.yaml          - the targets' source overlay (attached by ``targets.source``)
  5. defaults/dispatch/<mode>.yaml          - the dispatch overlay (attached by ``market.dispatch``; central only)
  6. scenarios.yaml, studies/<study>/       - scenario entries, month families and templates (``inherits``)
  clustering.yaml holds the clustering presets ``network.clustering`` may name.

Usage:
    from config_loader import load_config, get_scenario
    
    config = load_config()
    scenario = get_scenario("HT35", config)
"""

import csv
import logging
import os
import sys
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, List, Collection
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # `python config/config_loader.py` puts only config/ on the path
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sources import registry as _registry  # noqa: E402

logger = logging.getLogger(__name__)

#: Names of NESO's Clean Power 2030 pathways (data/scenario_sources/pathways.csv).
CP30_PATHWAYS = {p.name for p in _registry.pathways("CP30")}
CP30_REQUIRED_RESOURCE_KEYS = {"es1", "es5", "ed1", "reference", "validation", "interconnector_prices"}


# ══════════════════════════════════════════════════════════════════════════════
# TARGETS AND METHOD
# ══════════════════════════════════════════════════════════════════════════════
# Two independent keys on every merged scenario:
#
#   targets: {source, edition, pathway}   where the capacity, energy and demand numbers come from
#   method:  brownfield | gsp_block | historical   how the fleet and network are built
#
# source is "CP30", "FES" or "historical". edition is the publication year: FES by its edition
# (FES_year, e.g. 2025), CP30 by NESO's Clean Power 2030 advice (2024; CP30_data_version names the
# workbook revision, V003, within that edition), None for historical. pathway is the short code
# (HT, FFR, ND, ...; None for historical).
#
# method "brownfield" is the CP30 chain: named REPD/TEC projects, ETYS Appendix B, ES5 regions and
# the defaults/method/brownfield.yaml overlay. "gsp_block" builds from FES GSP building
# blocks (the FES long-range chain, FES25_HT_2050 and the older HT35-style scenarios). "historical" is
# DUKES/REPD/ESPENI for modelled_year <= 2024.
#
# A scenario that writes ``targets`` must also write ``method`` (method is never derived for a new
# scenario). A legacy scenario (LEGACY_SCENARIO_IDS_FILE) that writes neither has both derived from its
# old keys:
#
#   modelled_year <= 2024                                 -> historical / historical
#   scenario_source (or source, data_source) "CP30",
#     or a CP30_pathway key                               -> CP30 2024 <CP30_pathway code> / brownfield
#   FES_year and FES_scenario                             -> FES <FES_year> <FES_scenario code> / gsp_block
#   none of these (a future year)                         -> error: state targets and method
#
# A written ``method`` with no ``targets`` keeps the derived targets. A scenario that writes
# ``targets`` but none of the old keys gets those filled from it (scenario_source and CP30_pathway,
# or FES_year and FES_scenario), so the scripts that still read them keep working.

HISTORICAL_LAST_YEAR = 2024
TARGET_SOURCES = ("CP30", "FES", "historical")
BUILD_METHODS = ("brownfield", "gsp_block", "historical")
#: Pairs the chain can build. Any other pair (a FES brownfield build, for example) is
#: refused here rather than failing mid-build.
SUPPORTED_TARGET_METHODS = {("CP30", "brownfield"), ("FES", "gsp_block"), ("historical", "historical")}
#: The first FES edition a scenario can be built on. NESO published no GSP regional breakdown for FES 2020 (the
#: series starts with FES 2021), so its GSP rows cannot be placed (user, 19 Nov 2026: leave FES 2020 out).
FES_FIRST_BUILD_EDITION = 2021
#: CP30 has one edition: NESO's Clean Power 2030 advice, published November 2024.
CP30_EDITION = 2024
#: The scenario ids that existed when ``targets`` and ``method`` were introduced (frozen 27 Oct 2026).
#: Only these may still derive both from the old keys; a new scenario writes them.
LEGACY_SCENARIO_IDS_FILE = PROJECT_ROOT / "data" / "scenario_sources" / "legacy_scenario_ids.txt"


def _pathway(source: str, value: Any, edition: Any, scenario_id: str):
    """The registry pathway a scenario names by name, alias or code, in its edition."""
    try:
        try:
            return _registry.pathway_by_name(source, int(edition), str(value))
        except _registry.UnknownPathway:
            return _registry.pathway_by_code(source, int(edition), str(value))
    except (_registry.UnknownPathway, _registry.UnknownEdition) as exc:
        raise ValueError(f"[{scenario_id}] unknown {source} {edition} pathway {value!r} "
                         f"(data/scenario_sources/pathways.csv): {exc}") from None


def _pathway_code(source: str, name: Any, scenario_id: str, edition: Any = None, any_edition: bool = False) -> str:
    """Short code (HT, FFR, ...) of a pathway name or code in its source edition.

    any_edition is for the frozen legacy scenarios only: some name a pathway of another FES
    edition than their FES_year (Test_Future_2030: FES 2024 with Leading the Way, a FES 2020-2023
    pathway). Their code is then looked up in every edition of the source.
    """
    edition = CP30_EDITION if source == "CP30" else edition
    if edition is None:
        raise ValueError(f"[{scenario_id}] a {source} pathway needs its edition")
    try:
        return _pathway(source, name, edition, scenario_id).code
    except ValueError:
        if not any_edition:
            raise
    codes = set()
    for other in _registry.editions(source):
        try:
            codes.add(_pathway(source, name, other, scenario_id).code)
        except ValueError:
            continue
    if len(codes) != 1:
        raise ValueError(f"[{scenario_id}] {source} pathway {name!r} matches codes {sorted(codes)} across editions")
    return codes.pop()


def _pathway_name(source: str, code: Any, scenario_id: str, edition: Any = None) -> str:
    """Pathway name of a code (or name) in its source edition."""
    edition = CP30_EDITION if source == "CP30" else edition
    if edition is None:
        raise ValueError(f"[{scenario_id}] a {source} pathway needs its edition")
    return _pathway(source, code, edition, scenario_id).name


@lru_cache(maxsize=2)
def legacy_scenario_ids(path: str = str(LEGACY_SCENARIO_IDS_FILE)) -> frozenset:
    """The frozen ids that may derive ``targets`` and ``method`` from the old keys."""
    with open(path, encoding="utf-8") as handle:
        return frozenset(line.strip() for line in handle if line.strip() and not line.startswith("#"))


def _is_historical_year(scenario: Dict[str, Any]) -> bool:
    year = scenario.get("modelled_year") or scenario.get("year")
    return year is not None and int(year) <= HISTORICAL_LAST_YEAR


def legacy_target_source(scenario: Dict[str, Any]) -> Optional[str]:
    """The target source today's keys imply: "historical", "CP30", "FES", or None when they say nothing.

    A historical year wins, then an explicit CP30 marker, then a future FES scenario.
    """
    if _is_historical_year(scenario):
        return "historical"
    source = scenario.get("scenario_source") or scenario.get("source") or scenario.get("data_source") or ""
    if str(source).upper() == "CP30" or "CP30_pathway" in scenario:
        return "CP30"
    if "FES_year" in scenario or "FES_scenario" in scenario:
        return "FES"
    return None


def _legacy_targets(source: str, scenario: Dict[str, Any], scenario_id: str) -> Dict[str, Any]:
    if source == "historical":
        return {"source": "historical", "edition": None, "pathway": None}
    if source == "CP30":
        pathway = scenario.get("CP30_pathway")
        return {"source": "CP30", "edition": CP30_EDITION,
                "pathway": _pathway_code("CP30", pathway, scenario_id) if pathway else None}
    missing = [key for key in ("FES_year", "FES_scenario") if key not in scenario]
    if missing:
        raise ValueError(f"[{scenario_id}] a future FES scenario needs {missing} (or targets and method)")
    return {"source": "FES", "edition": int(scenario["FES_year"]),
            "pathway": _pathway_code("FES", scenario["FES_scenario"], scenario_id, scenario["FES_year"],
                                      any_edition=True)}


DERIVED_METHOD = {"historical": "historical", "CP30": "brownfield", "FES": "gsp_block"}


def resolve_targets_and_method(entry: Dict[str, Any], merged: Dict[str, Any],
                               scenario_id: str = "unknown",
                               legacy_ids: Optional[Collection[str]] = None) -> tuple:
    """A scenario's ``(targets, method, legacy_fill)``.

    ``entry`` is the scenario's own entry (inherits resolved): it says what the scenario wrote.
    ``merged`` is the config the old keys are read from (defaults, overrides and the entry,
    before the method's defaults attach). ``legacy_fill`` holds the old keys a ``targets``-only
    scenario needs filled in. ``legacy_ids`` are the scenarios allowed to derive both from the old
    keys (default: the frozen list, :data:`LEGACY_SCENARIO_IDS_FILE`). See the block comment above.
    """
    written_targets = entry.get("targets")
    written_method = entry.get("method")
    legacy = legacy_target_source(merged)
    # A historical year is historical whatever is written; any other answer is a config error.
    if written_targets is None:
        if legacy is None:
            raise ValueError(
                f"[{scenario_id}] future scenario (modelled_year {merged.get('modelled_year')}) states no "
                "targets: write targets: {source, edition, pathway} and method: brownfield | gsp_block")
        if scenario_id not in (legacy_scenario_ids() if legacy_ids is None else legacy_ids):
            raise ValueError(
                f"[{scenario_id}] a new scenario must write targets: {{source, edition, pathway}} and method "
                f"({' | '.join(BUILD_METHODS)}); only the scenarios in {LEGACY_SCENARIO_IDS_FILE.name} "
                "(those that existed on 27 Oct 2026) still derive them from FES_year/FES_scenario/scenario_source")
        targets = _legacy_targets(legacy, merged, scenario_id)
    else:
        if written_method is None:
            raise ValueError(
                f"[{scenario_id}] a scenario that writes targets must also write method "
                f"({' | '.join(BUILD_METHODS)}): method is never derived for a new study")
        if not isinstance(written_targets, dict):
            raise ValueError(f"[{scenario_id}] targets must be a mapping {{source, edition, pathway}}")
        unknown = sorted(set(written_targets) - {"source", "edition", "pathway"})
        if unknown:
            raise ValueError(f"[{scenario_id}] targets has unknown keys {unknown}")
        source = written_targets.get("source")
        if source not in TARGET_SOURCES:
            raise ValueError(f"[{scenario_id}] targets.source {source!r} must be one of {TARGET_SOURCES}")
        if source == "historical":
            targets = {"source": "historical", "edition": None, "pathway": None}
        else:
            edition = written_targets.get("edition", CP30_EDITION if source == "CP30" else None)
            if edition is None:
                raise ValueError(f"[{scenario_id}] targets.edition is required for source {source}")
            if written_targets.get("pathway") is None:
                raise ValueError(f"[{scenario_id}] targets.pathway is required for source {source}")
            targets = {"source": source, "edition": int(edition),
                       "pathway": _pathway_code(source, written_targets["pathway"], scenario_id, edition)}
        # Old keys written beside targets must say the same thing.
        if legacy is not None:
            implied = _legacy_targets(legacy, merged, scenario_id)
            clash = {k: (implied[k], targets[k]) for k in implied
                     if implied[k] is not None and implied[k] != targets[k]}
            if clash:
                raise ValueError(f"[{scenario_id}] targets disagree with the old keys (old, targets): {clash}")
    if targets["source"] != "historical" and _is_historical_year(merged):
        raise ValueError(f"[{scenario_id}] modelled_year {merged.get('modelled_year')} is historical, "
                         f"but targets.source is {targets['source']}")

    method = written_method if written_method is not None else DERIVED_METHOD[targets["source"]]
    if method not in BUILD_METHODS:
        raise ValueError(f"[{scenario_id}] method {method!r} must be one of {BUILD_METHODS}")
    if targets["source"] == "FES" and int(targets.get("edition") or 0) < FES_FIRST_BUILD_EDITION:
        raise ValueError(
            f"[{scenario_id}] FES {targets.get('edition')} is not supported: NESO published no GSP regional "
            f"breakdown for it, so its capacity cannot be placed; use FES {FES_FIRST_BUILD_EDITION} or later")
    if (targets["source"], method) not in SUPPORTED_TARGET_METHODS:
        raise ValueError(
            f"[{scenario_id}] targets.source {targets['source']} with method {method} is not supported yet "
            f"(supported: {sorted(SUPPORTED_TARGET_METHODS)})")

    legacy_fill: Dict[str, Any] = {}
    if written_targets is not None and legacy is None:
        if targets["source"] == "CP30":
            legacy_fill = {"scenario_source": "CP30",
                           "CP30_pathway": _pathway_name("CP30", targets["pathway"], scenario_id)}
        elif targets["source"] == "FES":
            legacy_fill = {"FES_year": targets["edition"],
                           "FES_scenario": _pathway_name("FES", targets["pathway"], scenario_id,
                                                         targets["edition"])}
    return targets, method, legacy_fill


# ══════════════════════════════════════════════════════════════════════════════
# YEAR PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
#: Calibrated scalars by (year, pathway): data/FES/fes_year_parameters.csv. A config value written
#: as ``year_parameter:<key>`` takes the table's value for the scenario's modelled_year and
#: targets.pathway; a study that wants another value writes the number instead.
YEAR_PARAMETERS_FILE = PROJECT_ROOT / "data" / "FES" / "fes_year_parameters.csv"
YEAR_PARAMETER_PREFIX = "year_parameter:"


@lru_cache(maxsize=2)
def year_parameter_table(path: str = str(YEAR_PARAMETERS_FILE)) -> Dict[tuple, float]:
    """{(year, pathway code, key): value} from the year-parameter table (every row needs a source)."""
    table: Dict[tuple, float] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not (row.get("source") or "").strip():
                raise ValueError(f"{path}: {row.get('key')} {row.get('pathway')} {row.get('year')} has no source")
            key = (int(row["year"]), row["pathway"].strip(), row["key"].strip())
            if key in table:
                raise ValueError(f"{path}: {key} appears twice")
            table[key] = float(row["value"])
    return table


def resolve_year_parameters(merged: Dict[str, Any], scenario_id: str,
                            table: Optional[Dict[tuple, float]] = None) -> Dict[str, Any]:
    """Replace every ``year_parameter:<key>`` value of ``merged`` (in place) by its table value.

    Raises:
        ValueError: the table has no row for the scenario's (modelled_year, targets.pathway, key).
    """
    found = []

    def walk(node, path):
        items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for key, value in items:
            if isinstance(value, str) and value.startswith(YEAR_PARAMETER_PREFIX):
                found.append((node, key, path + (str(key),), value[len(YEAR_PARAMETER_PREFIX):].strip()))
            elif isinstance(value, (dict, list)):
                walk(value, path + (str(key),))

    walk(merged, ())
    if not found:
        return merged
    table = year_parameter_table() if table is None else table
    year = merged.get("modelled_year")
    pathway = (merged.get("targets") or {}).get("pathway")
    for node, key, path, name in found:
        lookup = (int(year) if year is not None else None, pathway, name)
        if lookup not in table:
            raise ValueError(
                f"[{scenario_id}] {'.'.join(path)}: no {name!r} for pathway {pathway} year {year} in "
                f"{YEAR_PARAMETERS_FILE.relative_to(PROJECT_ROOT).as_posix()}; add the row with its source, "
                "or write the value in the study")
        node[key] = table[lookup]
    return merged


def _validate_existing_path(path_value: Any, field_name: str, errors: List[str], scenario_id: str) -> None:
    if not path_value:
        errors.append(f"[{scenario_id}] Missing required CP30 path: {field_name}")
        return

    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        errors.append(f"[{scenario_id}] CP30 path does not exist for {field_name}: {path_value}")


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.
    Nested dicts are merged recursively, not replaced entirely.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(filepath: Path) -> Dict:
    """Load a YAML file, return empty dict if not found."""
    if not filepath.exists():
        return {}
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # the C parser: same data, much faster
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=loader) or {}


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULTS_DIR = "defaults"
BASE_FILE = "base.yaml"
BASE_DIR = "base"
#: Overlay layers, merged over the base in this order when a scenario declares the axis (``method``,
#: ``targets.source``, ``market.dispatch``). One whole config tree per file: ``defaults/<layer>/<name>.yaml``.
LAYERS = ("method", "source", "dispatch")

#: ``market.dispatch``. ``self``: the market (no overlay). ``central``: the system operator dispatches
#: at cost (``defaults/dispatch/central.yaml``).
DISPATCH_MODES = ("self", "central")
#: Under central dispatch these keys must keep the overlay's value: a central dispatch with a copperplate
#: Stage 1, scarcity rent or bid and offer prices is not one, so a scenario that sets them otherwise is
#: refused rather than silently overridden (the renewable floors and the solver remain its choice).
CENTRAL_DISPATCH_FIXED = (
    ("market", "wholesale", "copperplate_mode"),
    ("market", "wholesale", "scarcity_pricing", "enabled"),
    ("market", "balancing", "pricing_basis"),
    ("support", "bm_bids", "enabled"),
    ("support", "bm_bids", "contract_value_bids", "enabled"),
)


def dispatch_mode(scenario_config: Dict[str, Any], merged: Dict[str, Any], scenario_id: str = "") -> str:
    """The scenario's ``market.dispatch``: its own entry's, else the value merged so far."""
    market = scenario_config.get("market") if isinstance(scenario_config.get("market"), dict) else {}
    if "dispatch" not in market:
        market = merged.get("market") if isinstance(merged.get("market"), dict) else {}
    mode = str(market.get("dispatch", "self")).lower()
    if mode not in DISPATCH_MODES:
        raise ValueError(f"[{scenario_id}] market.dispatch must be one of {DISPATCH_MODES}, "
                         f"got {market.get('dispatch')!r}")
    return mode


def check_dispatch_mode(merged: Dict[str, Any], central_layer: Dict[str, Any], scenario_id: str = "") -> None:
    """Refuse a central-dispatch scenario whose own entry changes a key ``CENTRAL_DISPATCH_FIXED`` holds."""
    if str((merged.get("market") or {}).get("dispatch", "self")).lower() != "central":
        return

    def get(tree: Dict[str, Any], path: tuple) -> Any:
        for key in path:
            if not isinstance(tree, dict) or key not in tree:
                return None
            tree = tree[key]
        return tree

    changed = [f"{'.'.join(p)}: {get(merged, p)!r} (central dispatch: {get(central_layer, p)!r})"
               for p in CENTRAL_DISPATCH_FIXED if get(merged, p) != get(central_layer, p)]
    if changed:
        raise ValueError(
            f"[{scenario_id}] market.dispatch: central fixes these keys, but the scenario sets them otherwise: "
            f"{'; '.join(changed)}. Drop them from the scenario, or use market.dispatch: self and set "
            "the keys directly.")


def defaults_dir(config_dir: Optional[Path] = None) -> Path:
    return (Path(config_dir) if config_dir is not None else CONFIG_DIR) / DEFAULTS_DIR


def defaults_files(config_dir: Optional[Path] = None) -> List[Path]:
    """Every file of the defaults (base and overlays): a rule's inputs when it reads them."""
    root = defaults_dir(config_dir)
    files = [root / BASE_FILE] + sorted((root / BASE_DIR).rglob("*.yaml"))
    for layer in LAYERS:
        files += sorted((root / layer).glob("*.yaml"))
    return files


def _place(tree: Dict[str, Any], path: tuple, value: Any, source: Path) -> None:
    """Put one file's content at its key path; a key defined by two files is an error."""
    if not isinstance(value, dict):
        raise ValueError(f"{source}: a defaults file must hold a mapping")
    cur = tree
    for part in path:
        nxt = cur.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"{source}: {'.'.join(path)} is also set as a value elsewhere")
        cur = nxt
    clash = sorted(set(cur) & set(value))
    if clash:
        raise ValueError(f"{source}: {'.'.join(path + (clash[0],))} is defined in two defaults files")
    cur.update(value)


def load_defaults(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """The base defaults as one dict: ``defaults/base.yaml`` plus ``defaults/base/**`` (file path = key path).

    ``defaults/base/market/balancing.yaml`` holds the contents of ``market.balancing``. Overlays are not
    included (``load_layer``). Read the defaults only through this, ``load_layer`` or ``read_config``.
    """
    root = defaults_dir(config_dir)
    base_file = root / BASE_FILE
    if not base_file.exists():
        raise FileNotFoundError(f"defaults not found: {base_file}")
    tree: Dict[str, Any] = {}
    _place(tree, (), load_yaml(base_file), base_file)
    for path in sorted((root / BASE_DIR).rglob("*.yaml")):
        rel = path.relative_to(root / BASE_DIR).with_suffix("")
        _place(tree, tuple(rel.parts), load_yaml(path), path)
    return tree


def load_layer(layer: str, name: Optional[str], config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """The ``layer`` overlay called ``name`` (``defaults/<layer>/<name>.yaml``); {} when there is none."""
    if layer not in LAYERS:
        raise ValueError(f"unknown defaults layer {layer!r}; layers: {LAYERS}")
    if not name:
        return {}
    path = defaults_dir(config_dir) / layer / f"{name}.yaml"
    content = load_yaml(path)
    check_renamed_keys(content, str(path))
    return content


def read_config(path: Any) -> Dict[str, Any]:
    """A config a script is handed by path (``--config``).

    The config directory (or its ``defaults`` directory) gives ``load_defaults()``; any other file is a
    resolved config written in the defaults' layout (a study's, ``study_config.py``) and is read as is.
    """
    path = Path(path)
    if path.is_dir():
        if (path / DEFAULTS_DIR).is_dir():
            return load_defaults(path)
        if path.name == DEFAULTS_DIR:
            return load_defaults(path.parent)
        raise FileNotFoundError(f"{path} is neither the config directory nor its defaults directory")
    if path.name == "defaults.yaml":
        raise FileNotFoundError(f"{path}: config/defaults.yaml is now the directory config/defaults/; "
                                "pass the config directory")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return load_yaml(path)


# ══════════════════════════════════════════════════════════════════════════════
# RENAMED KEYS
# ══════════════════════════════════════════════════════════════════════════════
#: Old key path -> new key path. A scenario entry, template or overlay that still writes an old path is
#: refused with the new one named (no aliases). ``None``: the key was removed.
RENAMED_KEYS = {
    'solver': 'solve.solver',
    'solve_mode': 'solve.mode',
    'voll': 'solve.voll',
    'etys': 'network.etys',
    'etys_upgrades': 'network.upgrades',
    'transmission': 'network.transmission',
    'spatial_mapping': 'network.spatial_mapping',
    'clustering': 'network.clustering',
    'component_aggregation': 'network.component_aggregation',
    'cp30_reinforcements': 'network.reinforcements',
    'hnd_hybrid': 'network.hnd_hybrid',
    'topology_adequacy': 'network.topology_adequacy',
    'renewable_aggregation': 'renewables.aggregation',
    'renewable_performance_factors': 'renewables.performance_factors',
    'renewable_performance_factor_registry': 'renewables.performance_factor_registry',
    'embedded_generation': 'renewables.embedded',
    'hydro': 'renewables.hydro',
    'thermal_availability_factors': 'thermal.availability_factors',
    'thermal_outage_availability': 'thermal.outages',
    'thermal_ramping': 'thermal.ramping',
    'nuclear_availability_smoothing': 'thermal.nuclear_smoothing',
    'historical_dispatch_anchors': 'thermal.dispatch_anchors',
    'storage_round_trip_efficiencies': 'storage.round_trip_efficiencies',
    'storage_dispatch_costs': 'storage.dispatch_costs',
    'storage_standing_losses': 'storage.standing_losses',
    'storage_placement': 'storage.placement',
    'demand_timeseries': 'demand.timeseries',
    'demand_disaggregation': 'demand.disaggregation',
    'demand_flexibility': 'demand.flexibility',
    'cp30_demand': 'demand.build',
    'cp30_brownfield': 'fleet.brownfield',
    'spatial_proxy': 'fleet.spatial_proxy',
    'station_identity_registry': 'fleet.station_registry',
    'historical_commissioning': 'fleet.commissioning',
    'fes_supply': 'fleet.fes_supply',
    'subsidy_tracking': 'support.labels',
    'marginal_costs.subsidies': 'support.dispatch_costs',
    'market.balancing.renewable_support_pricing': 'support.bm_bids',
    'market.revenue_tracking': 'support.revenue',
    'support_scheme_audit': 'support.audit',
    'cp30_stage1_validation': 'validation.stage1',
    'cp30_balancing_validation': 'validation.balancing',
    'cp30_fleet_location_validation': 'validation.fleet_location',
    'cp30_locational_validation': 'validation.locational',
    'CP30_workbook': 'source_data.workbook',
    'CP30_data_version': 'source_data.version',
    'CP30_resources': 'source_data.files',
    'network_assumption': 'source_data.network_assumption',
    'consumer_bill_forecasts': 'consumer_bill.forecasts',
    'price_level_mapping': 'consumer_bill.price_level_mapping',
    'market.consumer_bill': 'consumer_bill.price_cap',
    'demand_source': None,
    'fes2050_supply': 'fleet.fes_supply',
    'demand_disaggregation.fes2050': 'demand.disaggregation.fes',
    'demand.disaggregation.fes2050': 'demand.disaggregation.fes',
    'hydrogen.fes2050': 'hydrogen.fes',
}
#: The scenario-level ``consumer_bill`` alias of ``market.consumer_bill`` is not in the table:
#: ``consumer_bill`` is now the bill domain itself (its price-cap calculator is ``consumer_bill.price_cap``).


def check_renamed_keys(entry: Any, where: str) -> None:
    if not isinstance(entry, dict):
        return
    for old, new in RENAMED_KEYS.items():
        cur = entry
        parts = old.split(".")
        for part in parts[:-1]:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, dict) and parts[-1] in cur:
            hint = f"write {new}" if new else "it was removed"
            raise ValueError(f"{where} writes {old}, which was renamed in the config restructure: {hint} "
                             "(config/README.md)")


#: Scenario blocks (dotted paths) that no network build or solve script reads (checked against the scripts of
#: the model rules and everything they import). If one of those scripts starts reading a block, remove it here.
NOT_READ_BY_MODEL = (
    "consumer_bill",
    "description",
    "logging",
    "public_ownership",
    "source_data.version",
    "support.audit",
    "validation",
)


def _without_paths(config: Dict, paths) -> Dict:
    """``config`` without the dotted ``paths`` (copied only along the paths it changes)."""
    out = dict(config)
    for path in paths:
        parts = path.split(".")
        cur = out
        for part in parts[:-1]:
            if not isinstance(cur.get(part), dict):
                cur = None
                break
            cur[part] = dict(cur[part])
            cur = cur[part]
        if cur is not None:
            cur.pop(parts[-1], None)
    return out


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value, key=lambda k: (type(k).__name__, str(k)))}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_canonical(v) for v in value)
    return value


class ScenarioConfigParam(dict):
    """A scenario config passed to a rule's script, as a Snakemake param.

    The script receives a plain dict holding every key, exactly as before; pickling turns this
    object back into a dict. Snakemake decides whether a param changed by comparing ``repr(value)``
    with the repr stored at the last run. Here ``repr`` gives the config with keys sorted and
    without the ``hidden`` blocks (dotted paths). So reordering keys, or editing a block this rule
    never reads, does not make its outputs look out of date.
    """

    def __init__(self, data: Dict, hidden=()):
        super().__init__(data)
        self._hidden = tuple(sorted(hidden))

    def __repr__(self) -> str:
        return repr(_canonical(_without_paths(dict(self), self._hidden)))

    def __reduce__(self):
        return (dict, (dict(self),))


def scenario_param(config: Dict) -> ScenarioConfigParam:
    """A scenario config as a rule param, compared on its sorted contents."""
    return ScenarioConfigParam(config)


def model_scenario_param(config: Dict) -> ScenarioConfigParam:
    """A scenario config as a build or solve rule's param: also blind to ``NOT_READ_BY_MODEL``."""
    return ScenarioConfigParam(config, NOT_READ_BY_MODEL)


TEMPLATES_KEY = "templates"
INHERITS_KEY = "inherits"


#: A study folder's files, read in this order (``config/studies/<study>/``; README there).
STUDY_FILES = ("study.yaml", "templates.yaml", "families.yaml", "scenarios.yaml")
#: A family entry's own keys; everything else in it is a setting every month gets.
FAMILY_META = ("id", "year", "months", "stage1", "network_from", "month_overrides", "description")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


#: Environment variable that replaces config.yaml's ``studies`` list: ``all`` (every live study that is not a
#: draft, after the listed ones) or a comma list of study names. The test suite sets ``all``, so its checks of the
#: studies' scenarios run whatever config.yaml loads (the public config.yaml loads none).
STUDIES_ENV = "PYPSA_GB_STUDIES"


def _live_studies(config_dir: Path) -> List[str]:
    """Every live study folder (``studies/<study>/``, not the archive), drafts excepted."""
    studies_dir = config_dir / "studies"
    if not studies_dir.is_dir():
        return []
    return sorted(p.name for p in studies_dir.iterdir()
                  if p.is_dir() and p.name != "archive" and not _is_draft_study(p))


def read_main_config(config_dir: Optional[Path] = None) -> Dict:
    """config.yaml, with its ``studies`` list replaced when ``PYPSA_GB_STUDIES`` is set (``STUDIES_ENV``)."""
    config_dir = Path(config_dir) if config_dir is not None else Path(__file__).parent
    main_config = dict(load_yaml(config_dir / "config.yaml") or {})
    override = os.environ.get(STUDIES_ENV, "").strip()
    if override:
        listed = list(main_config.get("studies") or [])
        if override.lower() == "all":
            main_config["studies"] = listed + [s for s in _live_studies(config_dir) if s not in listed]
        else:
            main_config["studies"] = [s.strip() for s in override.split(",") if s.strip()]
    return main_config


def scenario_files(config_dir: Optional[Path] = None, main_config: Optional[Dict] = None) -> List[Path]:
    """The files that define scenarios: ``scenarios.yaml``, then each study listed in config.yaml.

    A study is a folder ``config/studies/<name>/`` (``STUDY_FILES``) or a single file
    ``config/studies/<name>.yaml``, loaded only when ``config.yaml`` lists it under ``studies`` (or
    ``PYPSA_GB_STUDIES`` names it: ``read_main_config``). An archived study is ``archive/<name>``.
    """
    config_dir = Path(config_dir) if config_dir is not None else Path(__file__).parent
    if main_config is None:
        main_config = read_main_config(config_dir)
    files = [config_dir / "scenarios.yaml"]
    for name in main_config.get("studies") or []:
        folder, flat = config_dir / "studies" / name, config_dir / "studies" / f"{name}.yaml"
        if folder.is_dir():
            files += [folder / f for f in STUDY_FILES if (folder / f).exists()]
        elif flat.exists():
            files.append(flat)
        else:
            raise FileNotFoundError(f"config.yaml lists study {name!r}, but neither {folder} nor {flat} exists")
    return files


def _is_draft_study(folder: Path) -> bool:
    """A study folder whose ``study.yaml`` declares ``draft: true``."""
    study_file = folder / "study.yaml"
    return study_file.exists() and bool((load_yaml(study_file) or {}).get("draft"))


def main_config_with_archive(config_dir: Optional[Path] = None, main_config: Optional[Dict] = None) -> Dict:
    """config.yaml with every study listed: each live study (``studies/<study>/``), whether or not config.yaml
    loads it (the public config.yaml loads none), and every archived study (``studies/archive/<study>/``),
    including the archive of a study that is no longer live. For checks over every scenario id.

    A live study whose ``study.yaml`` says ``draft: true`` (prepared, not yet complete enough to merge: fes2045)
    is added only when config.yaml lists it."""
    config_dir = Path(config_dir) if config_dir is not None else Path(__file__).parent
    main_config = dict(main_config if main_config is not None else read_main_config(config_dir))
    studies = list(main_config.get("studies") or [])
    studies += [s for s in _live_studies(config_dir) if s not in studies]
    archive = config_dir / "studies" / "archive"
    archived = sorted(p.name for p in archive.iterdir() if p.is_dir()) if archive.is_dir() else []
    main_config["studies"] = studies + [f"archive/{s}" for s in archived if f"archive/{s}" not in studies]
    return main_config


def _parents(entry: Dict) -> List[str]:
    """An entry's ``inherits``: one name or a list (merged left to right, the entry's own body last)."""
    parents = entry.get(INHERITS_KEY)
    if not parents:
        return []
    return [parents] if isinstance(parents, str) else list(parents)


def _leaves(node: Any, path: tuple = ()):
    if isinstance(node, dict) and node:
        for k, v in node.items():
            yield from _leaves(v, path + (k,))
    else:
        yield path, node


def load_scenario_entries(config_dir: Optional[Path] = None, main_config: Optional[Dict] = None,
                          include_archive: bool = False) -> Dict[str, Dict]:
    """Every scenario's own entry, with its ``inherits`` resolved, before defaults are applied.

    Each file may hold a ``templates`` mapping beside its scenarios; a study folder keeps them in
    ``templates.yaml`` and its month families in ``families.yaml``. An entry (scenario or template) may name
    ``inherits: <name>`` or ``inherits: [<name>, ...]``: the parents are resolved first and deep-merged in
    order, the entry's own body last. Two parents may not set the same leaf to different values unless the
    entry sets it itself, so a composition never depends on the order of its list. This is
    what ``load_config`` merges over the defaults, and what anything reading a scenario's "raw" entry must
    use (the dispatch-mode check judges what a scenario set explicitly on it).
    """
    if include_archive:
        main_config = main_config_with_archive(config_dir, main_config)
    sources = read_scenario_sources(config_dir, main_config)
    templates, entries = sources.templates, sources.entries

    resolved: Dict[str, Dict] = {}

    def resolve(name: str, stack: tuple) -> Dict:
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ValueError(f"inheritance cycle: {' -> '.join(stack + (name,))}")
        entry = templates[name] if name in templates else entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"{stack[-1] if stack else name!r} inherits {name!r}, which is not a template or scenario")
        parents = _parents(entry)
        body = {k: v for k, v in entry.items() if k != INHERITS_KEY}
        own = [path for path, _ in _leaves(body)]
        merged: Dict[str, Any] = {}
        seen: Dict[tuple, tuple] = {}
        for parent in parents:
            tree = resolve(parent, stack + (name,))
            if len(parents) > 1:
                for path, value in _leaves(tree):
                    first = seen.setdefault(path, (parent, value))
                    if first[1] != value and not any(path[:len(o)] == o for o in own):
                        raise ValueError(
                            f"{name!r} inherits {first[0]!r} and {parent!r}, which set {'.'.join(map(str, path))} "
                            f"differently ({first[1]!r} / {value!r}); set it in {name!r} itself, so the result does not depend on the order of inherits")
            merged = deep_merge(merged, tree)
        result = deep_merge(merged, body)
        resolved[name] = result
        return result

    return {name: resolve(name, ()) for name, body in entries.items() if isinstance(body, dict)}


class ScenarioSources:
    """What the scenario files hold, ``inherits`` unresolved: templates, entries (families expanded), each
    name's file, each family's month ids and each study folder's ``study.yaml``."""

    def __init__(self):
        self.templates: Dict[str, Dict] = {}
        self.entries: Dict[str, Any] = {}
        self.origin: Dict[str, Path] = {}
        self.families: Dict[str, List[str]] = {}         # "<study>:<family>" -> its month ids, in month order
        self.studies: Dict[str, Dict] = {}               # study name -> its study.yaml

    def add(self, kind: str, name: str, body: Any, path: Path) -> None:
        table = self.templates if kind == "template" else self.entries
        if name in table:
            raise ValueError(f"{kind} {name!r} is defined in both {self.origin[name]} and {path}")
        check_renamed_keys(body, f"{path.name}: {kind} {name}")
        table[name] = (body or {}) if kind == "template" else body
        self.origin[name] = path


def _month_period(year: int, month: int) -> Dict[str, str]:
    import calendar

    last = calendar.monthrange(int(year), month)[1]
    return {"start": f"{year}-{month:02d}-01 00:00", "end": f"{year}-{month:02d}-{last:02d} 23:00"}


def expand_families(study: str, spec: Dict[str, Any], families: Dict[str, Dict]):
    """A study's month families (``families.yaml``) as ``({scenario id: entry}, {family: [ids]})``.

    A family ``<name>: {inherits: [...], <settings>..., year, months, stage1, network_from, month_overrides,
    description}`` becomes one entry per month, named by the family's ``id`` or the study's ``family_id``
    (``{Mon}``, ``{year}``, ``{family}``). Each month gets its own ``solve_period``; ``stage1: <family>`` sets
    ``wholesale_from`` to that family's same month; ``month_overrides: {Mar: {...}}`` holds a month's exceptions.
    """
    families = {name: (family or {}) for name, family in families.items()}

    def ids(name: str) -> Dict[str, str]:
        family = families[name]
        pattern = family.get("id") or spec.get("family_id")
        if not pattern:
            raise ValueError(f"study {study}: family {name!r} has no id and the study no family_id")
        year = family.get("year", spec.get("year"))
        months = family.get("months") or MONTH_NAMES
        bad = sorted(set(months) - set(MONTH_NAMES)) + sorted(set(family.get("month_overrides") or {}) - set(months))
        if bad:
            raise ValueError(f"study {study}: family {name!r} names months it does not have: {bad}")
        return {mon: pattern.format(Mon=mon, year=year, family=name) for mon in months}

    out: Dict[str, Dict] = {}
    index: Dict[str, List[str]] = {}
    for name, family in families.items():
        year = family.get("year", spec.get("year"))
        stage1 = family.get("stage1")
        if stage1 is not None and stage1 not in families:
            raise ValueError(f"study {study}: family {name!r} takes its Stage 1 from {stage1!r}, not a family here")
        own = ids(name)
        index[name] = list(own.values())
        for mon, sid in own.items():
            body = {k: deepcopy(v) for k, v in family.items() if k not in FAMILY_META}
            # the family's own solve_period settings (e.g. enabled) with this month's start and end
            body["solve_period"] = {**(body.get("solve_period") or {}), **_month_period(year, MONTH_NAMES.index(mon) + 1)}
            if stage1 is not None:
                body["wholesale_from"] = ids(stage1)[mon]
            if family.get("network_from"):
                body["network_from"] = family["network_from"]
            body["description"] = f"{spec.get('label', study)}: {mon} {year}, {name}" + (
                f": {family['description']}" if family.get("description") else "")
            body = deep_merge(body, (family.get("month_overrides") or {}).get(mon) or {})
            if sid in out:
                raise ValueError(f"study {study}: scenario id {sid!r} produced twice")
            out[sid] = body
    return out, index


def read_scenario_sources(config_dir: Optional[Path] = None, main_config: Optional[Dict] = None) -> ScenarioSources:
    """Every template, entry and family as written in the scenario files (``ScenarioSources``)."""
    config_dir = Path(config_dir) if config_dir is not None else Path(__file__).parent
    studies_dir = (config_dir / "studies").resolve()
    src = ScenarioSources()
    for path in scenario_files(config_dir, main_config):
        content = load_yaml(path)
        folder = path.parent.resolve()
        study = folder.relative_to(studies_dir).as_posix() if folder != studies_dir and studies_dir in folder.parents else None
        if study and path.name == "study.yaml":
            src.studies[study] = content
            continue
        if study and path.name == "templates.yaml":
            for name, body in content.items():
                src.add("template", name, body, path)
            continue
        if study and path.name == "families.yaml":
            entries, index = expand_families(study, src.studies.get(study) or {}, content)
            for sid, body in entries.items():
                src.add("scenario", sid, body, path)
            for name, sids in index.items():
                src.families[f"{study}:{name}"] = sids
            continue
        for name, body in (content.pop(TEMPLATES_KEY, None) or {}).items():
            src.add("template", name, body, path)
        for name, body in content.items():
            src.add("scenario", name, body, path)
    clash = sorted(set(src.templates) & set(src.entries))
    if clash:
        raise ValueError(f"names used for both a template and a scenario: {clash[:5]}")
    return src


def _read_scenario_files(config_dir: Optional[Path] = None, main_config: Optional[Dict] = None):
    """(templates, entries, origin) as written in the scenario files, ``inherits`` unresolved."""
    src = read_scenario_sources(config_dir, main_config)
    return src.templates, src.entries, src.origin


#: config.yaml keys that moved into a study's study.yaml: old key -> new key.
MOVED_TO_STUDIES = {
    "cp30_monthly_aggregates": "monthly_aggregates",
    "cp30_balancing_validation_families": "balancing_validation_families",
    "secured_capability_audit_central": "secured_capability_audit_central",
}


def study_settings(config_dir: Optional[Path] = None, main_config: Optional[Dict] = None,
                   sources: Optional[ScenarioSources] = None) -> Dict[str, Any]:
    """The loaded studies' own settings, merged: ``monthly_aggregates`` (``{study: {scenario_family, scenarios}}``,
    a ``family:`` entry expanded to its months), ``balancing_validation_families`` and
    ``secured_capability_audit_central``. Reads the study files without merging any scenario (cheap)."""
    sources = sources or read_scenario_sources(config_dir, main_config)
    out: Dict[str, Any] = {"monthly_aggregates": {}, "balancing_validation_families": [],
                           "secured_capability_audit_central": None}
    for study, spec in sources.studies.items():
        for name, agg in (spec.get("monthly_aggregates") or {}).items():
            if name in out["monthly_aggregates"]:
                raise ValueError(f"monthly aggregate {name!r} is declared by two studies")
            agg = dict(agg or {})
            if "family" in agg:
                agg["scenarios"] = expand_selection([f"{study}:{agg.pop('family')}"], sources)
            out["monthly_aggregates"][name] = agg
        out["balancing_validation_families"] += list(spec.get("balancing_validation_families") or [])
        if spec.get("secured_capability_audit_central"):
            if out["secured_capability_audit_central"]:
                raise ValueError("secured_capability_audit_central is set by two studies")
            out["secured_capability_audit_central"] = spec["secured_capability_audit_central"]
    return out


def expand_selection(names, sources: ScenarioSources) -> List[str]:
    """Scenario ids for a selection (``run_scenarios``, an aggregate) that may name families.

    A name that is a scenario id stays; a family name (``<family>`` when unique across the loaded studies,
    else ``<study>:<family>``) becomes its months in order; anything else is kept for the caller to report.
    """
    by_name: Dict[str, List[str]] = {}
    for key in sources.families:
        by_name.setdefault(key.split(":", 1)[1], []).append(key)
    out: List[str] = []
    for name in names or []:
        if name in sources.entries:
            out.append(name)
        elif name in sources.families:
            out += sources.families[name]
        elif len(by_name.get(name, [])) == 1:
            out += sources.families[by_name[name][0]]
        elif len(by_name.get(name, [])) > 1:
            raise ValueError(f"family {name!r} exists in several studies {by_name[name]}; name it as <study>:<family>")
        else:
            out.append(name)
    return list(dict.fromkeys(out))


def explain_scenario(scenario_id: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Where each value of a scenario's merged config comes from (``--explain``).

    Returns ``{"layers": [labels in merge order], "values": [(dotted path, value, label)]}``. The label is
    the last layer that set the path: a defaults file, ``config.yaml``, a ``method/`` or ``source/``
    overlay, a template or the scenario's own entry; ``loader`` for what ``load_config`` derives itself
    (``targets``, ``method``, year parameters, the clustering preset).
    """
    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    config = load_config(config_dir)
    if scenario_id not in config["scenarios"]:
        raise KeyError(f"no scenario {scenario_id!r}")
    merged = config["scenarios"][scenario_id]
    root = defaults_dir(config_dir)

    layers = []
    base_file = root / BASE_FILE
    layers.append((str(base_file.relative_to(config_dir.parent)), load_yaml(base_file)))
    for path in sorted((root / BASE_DIR).rglob("*.yaml")):
        tree: Dict[str, Any] = {}
        _place(tree, tuple(path.relative_to(root / BASE_DIR).with_suffix("").parts), load_yaml(path), path)
        layers.append((str(path.relative_to(config_dir.parent)), tree))
    if config["global_overrides"]:
        layers.append(("config/config.yaml", config["global_overrides"]))
    for layer, name in (("method", merged.get("method")), ("source", (merged.get("targets") or {}).get("source")),
                        ("dispatch", (merged.get("market") or {}).get("dispatch"))):
        path = root / layer / f"{name}.yaml"
        if path.exists():
            layers.append((str(path.relative_to(config_dir.parent)), load_yaml(path)))
    templates, entries, origin = _read_scenario_files(config_dir)
    def chain(name):
        """The entries ``name`` is built from, in merge order (parents first, list ``inherits`` left to right)."""
        body = (templates.get(name) if name in templates else entries.get(name)) or {}
        kind = "template" if name in templates else "scenario"
        out = [layer for parent in _parents(body) for layer in chain(parent)]
        return out + [(f"{origin[name].relative_to(config_dir.parent)}: {kind} {name}",
                       {k: v for k, v in body.items() if k != INHERITS_KEY})]
    layers.extend(chain(scenario_id))

    who: Dict[tuple, str] = {}

    def record(node, path, label):
        if isinstance(node, dict):               # deep_merge: a mapping merges (an empty one changes nothing)
            if not node and not any(p[:len(path)] == path for p in who):
                who[path] = label                # ... unless nothing is there yet: then it sets {}
            for k, v in node.items():
                record(v, path + (k,), label)
        else:
            for key in [p for p in who if p[:len(path)] == path]:
                del who[key]                       # a value replaces whatever was below it
            who[path] = label

    for label, tree in layers:
        record(tree, (), label)

    values = []

    def walk(node, path):
        if isinstance(node, dict) and node:
            for k, v in node.items():
                walk(v, path + (k,))
        else:
            values.append((".".join(map(str, path)), node, who.get(path, "loader")))
    walk(merged, ())
    return {"layers": [label for label, _ in layers], "values": values}

def resolve_clustering(clustering_value: Any, clustering_presets: Dict) -> Dict:
    """
    Resolve clustering configuration.
    
    Args:
        clustering_value: Either a string (preset name) or dict (inline config)
        clustering_presets: Dictionary of preset configurations
    
    Returns:
        Resolved clustering configuration dict
        
    Note:
        If a dict has 'method' specified, clustering is enabled regardless of 
        the 'enabled' key (which may come from defaults). The 'enabled: false'
        only disables clustering if no method is specified.
    """
    if clustering_value is None:
        return {"enabled": False}
    
    if isinstance(clustering_value, str):
        # Reference to preset
        if clustering_value not in clustering_presets:
            raise ValueError(f"Unknown clustering preset: '{clustering_value}'. "
                           f"Available: {list(clustering_presets.keys())}")
        config = deepcopy(clustering_presets[clustering_value])
        config["enabled"] = True
        return config
    
    if isinstance(clustering_value, dict):
        # Allow referencing preset while extending with custom options
        if "preset" in clustering_value:
            preset_name = clustering_value.get("preset")
            if preset_name not in clustering_presets:
                raise ValueError(f"Unknown clustering preset: '{preset_name}'. "
                               f"Available: {list(clustering_presets.keys())}")
            config = deep_merge(
                deepcopy(clustering_presets[preset_name]),
                {k: v for k, v in clustering_value.items() if k != "preset"}
            )
            config["enabled"] = True
            return config
        
        # If 'method' is specified, clustering should be enabled
        # (even if 'enabled: false' was inherited from defaults)
        if "method" in clustering_value:
            config = deepcopy(clustering_value)
            config["enabled"] = True
            return config
        
        # No method specified - respect the enabled flag
        if not clustering_value.get("enabled", False):
            return {"enabled": False}
        
        # Enabled but no method - unusual, but keep the config
        config = deepcopy(clustering_value)
        config["enabled"] = True
        return config
    
    return {"enabled": False}


def load_config(config_dir: Optional[Path] = None, include_archive: bool = False) -> Dict[str, Any]:
    """
    Load the complete configuration with proper inheritance.
    
    Returns a dict with:
        - defaults: Default settings
        - scenarios: Dict of scenario_id -> merged scenario config
        - run_scenarios: List of scenario IDs to run
        - clustering_presets: Available clustering presets
        - global_overrides: Overrides from config.yaml
    ``include_archive`` loads every study, live or archived, whether or not config.yaml lists it (checks over
    every scenario id; ``main_config_with_archive``).
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    # Load all config files
    defaults = load_defaults(config_dir)
    main_config = read_main_config(config_dir)
    if include_archive:
        main_config = main_config_with_archive(config_dir, main_config)
    scenarios_raw = load_scenario_entries(config_dir, main_config)
    sources = read_scenario_sources(config_dir, main_config)
    clustering_config = load_yaml(config_dir / "clustering.yaml")

    # Extract clustering presets
    clustering_presets = clustering_config.get("presets", {})
    aggregation_strategies = clustering_config.get("aggregation_strategies", {})
    
    # Extract global overrides from main config (excluding the run selection). Study settings (monthly
    # aggregates, validation families, the audit central) live in each study's study.yaml (study_settings).
    moved = sorted(set(main_config) & set(MOVED_TO_STUDIES))
    if moved:
        raise ValueError(f"config.yaml sets {moved}: these are study settings now; put them in "
                         f"config/studies/<study>/study.yaml as {[MOVED_TO_STUDIES[k] for k in moved]}")
    special_keys = {"run_scenarios", "studies"}
    global_overrides = {k: v for k, v in main_config.items() if k not in special_keys}
    
    check_renamed_keys(global_overrides, "config.yaml")

    # Build merged scenarios: base, config.yaml's global overrides, the method's overlay
    # (defaults/method/<method>.yaml), the targets' source overlay (defaults/source/<source>.yaml),
    # the dispatch overlay (defaults/dispatch/<market.dispatch>.yaml),
    # then the scenario's own entry. A method or source without a file has no overlay.
    base_defaults = deepcopy(defaults)
    method_layers = {method: load_layer("method", method, config_dir) for method in BUILD_METHODS}
    source_layers = {source: load_layer("source", source, config_dir) for source in TARGET_SOURCES}
    dispatch_layers = {mode: load_layer("dispatch", mode, config_dir) for mode in DISPATCH_MODES}
    merged_scenarios = {}
    for scenario_id, scenario_config in scenarios_raw.items():
        if scenario_config is None:
            continue
            
        # Skip template comments (strings that start with template markers)
        if isinstance(scenario_config, str):
            continue
        
        # Start with defaults
        merged = deepcopy(base_defaults)
        
        # Apply global overrides
        merged = deep_merge(merged, global_overrides)

        # Targets and method, written or derived from the old keys. The old keys are
        # all top-level scalars, so a shallow view of defaults + entry is enough to read them.
        targets, method, legacy_fill = resolve_targets_and_method(
            scenario_config, {**merged, **scenario_config}, scenario_id)

        # The overlays attach by the scenario's declared axes, never by its name.
        merged = deep_merge(merged, method_layers[method])
        merged = deep_merge(merged, source_layers[targets["source"]])
        merged = deep_merge(merged, dispatch_layers[dispatch_mode(scenario_config, merged, scenario_id)])

        # Apply scenario-specific settings
        merged = deep_merge(merged, scenario_config)
        for key, value in legacy_fill.items():
            merged.setdefault(key, value)
        merged["targets"] = targets
        merged["method"] = method
        resolve_year_parameters(merged, scenario_id)

        # Resolve clustering references
        network = merged.get("network")
        if isinstance(network, dict) and "clustering" in network:
            network["clustering"] = resolve_clustering(network["clustering"], clustering_presets)
        
        # Add scenario ID for reference
        merged["_scenario_id"] = scenario_id
        check_dispatch_mode(merged, dispatch_layers["central"], scenario_id)

        merged_scenarios[scenario_id] = merged
    
    return {
        "defaults": defaults,
        "scenarios": merged_scenarios,
        "run_scenarios": expand_selection(main_config.get("run_scenarios", []), sources),
        "families": sources.families,
        "study_settings": study_settings(sources=sources),
        "studies": sources.studies,
        "clustering_presets": clustering_presets,
        "aggregation_strategies": aggregation_strategies,
        "global_overrides": global_overrides,
        "logging": defaults.get("logging", {}),
        "solve_mode": (defaults.get("solve") or {}).get("mode", "LP"),
    }


def get_scenario(scenario_id: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get a fully resolved scenario configuration.
    
    Args:
        scenario_id: The scenario identifier
        config: Pre-loaded config (optional, will load if not provided)
    
    Returns:
        Merged scenario configuration dict
    """
    if config is None:
        config = load_config()
    
    if scenario_id not in config["scenarios"]:
        available = list(config["scenarios"].keys())
        raise ValueError(f"Unknown scenario: '{scenario_id}'. Available: {available}")
    
    return config["scenarios"][scenario_id]


def get_active_scenarios(config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Get list of scenarios that are marked to run in config.yaml.
    
    Returns:
        List of merged scenario configuration dicts
    """
    if config is None:
        config = load_config()
    
    scenarios = []
    for scenario_id in config["run_scenarios"]:
        if scenario_id in config["scenarios"]:
            scenarios.append(config["scenarios"][scenario_id])
        else:
            print(f"Warning: Scenario '{scenario_id}' in run_scenarios not found")
    
    return scenarios


def validate_scenario(scenario: Dict[str, Any]) -> List[str]:
    """
    Validate a scenario configuration.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    scenario_id = scenario.get("_scenario_id", "unknown")
    
    # Required fields
    required = ["modelled_year", "renewables_year", "demand_year"]
    for field in required:
        if field not in scenario:
            errors.append(f"[{scenario_id}] Missing required field: {field}")
    
    # Future scenarios need either FES config or explicit CP30 config.
    modelled_year = scenario.get("modelled_year", 0)
    source = (scenario.get("targets") or {}).get("source") or legacy_target_source(scenario)
    if source != "historical":
        if source == "CP30":
            pathway = scenario.get("CP30_pathway")
            if not pathway:
                errors.append(f"[{scenario_id}] CP30 scenario requires CP30_pathway")
            elif pathway not in CP30_PATHWAYS:
                errors.append(
                    f"[{scenario_id}] Invalid CP30_pathway: {pathway}. "
                    f"Must be one of {sorted(CP30_PATHWAYS)}"
                )

            if "FES_year" in scenario or "FES_scenario" in scenario:
                errors.append(
                    f"[{scenario_id}] CP30 scenarios must not use top-level FES_year/FES_scenario; "
                    "put FES proxy assumptions under fleet.spatial_proxy"
                )

            source_data = scenario.get("source_data") or {}
            _validate_existing_path(source_data.get("workbook"), "source_data.workbook", errors, scenario_id)

            resources = source_data.get("files")
            if not isinstance(resources, dict):
                errors.append(f"[{scenario_id}] CP30 scenario requires a source_data.files mapping")
            else:
                missing = sorted(CP30_REQUIRED_RESOURCE_KEYS - set(resources))
                for key in missing:
                    errors.append(f"[{scenario_id}] source_data.files missing key: {key}")
                for key in sorted(CP30_REQUIRED_RESOURCE_KEYS & set(resources)):
                    _validate_existing_path(resources.get(key), f"source_data.files.{key}", errors, scenario_id)

            proxy = (scenario.get("fleet") or {}).get("spatial_proxy") or {}
            if not proxy:
                errors.append(
                    f"[{scenario_id}] CP30 scenario requires explicit fleet.spatial_proxy until CP30 spatial integration exists"
                )
            elif str(proxy.get("source", "")).upper() == "FES":
                if "FES_year" not in proxy:
                    errors.append(f"[{scenario_id}] FES spatial_proxy requires FES_year")
                if "FES_scenario" not in proxy:
                    errors.append(f"[{scenario_id}] FES spatial_proxy requires FES_scenario")
            else:
                errors.append(
                    f"[{scenario_id}] Unsupported CP30 fleet.spatial_proxy source: {proxy.get('source')}"
                )
        else:
            if "FES_year" not in scenario:
                errors.append(f"[{scenario_id}] Future FES scenario (year {modelled_year}) requires FES_year")
            if "FES_scenario" not in scenario:
                errors.append(f"[{scenario_id}] Future FES scenario (year {modelled_year}) requires FES_scenario")
    
    # Validate network_model
    valid_networks = ["ETYS", "Reduced", "Zonal"]
    network_model = scenario.get("network_model", "ETYS")
    if network_model not in valid_networks:
        errors.append(f"[{scenario_id}] Invalid network_model: {network_model}. Must be one of {valid_networks}")
    
    return errors


# ══════════════════════════════════════════════════════════════════════════════
# CLI for testing
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="PyPSA-GB Configuration Loader")
    parser.add_argument("--scenario", "-s", help="Show specific scenario config")
    parser.add_argument("--list", "-l", action="store_true", help="List all scenarios")
    parser.add_argument("--active", "-a", action="store_true", help="List active scenarios")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate all scenarios")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--dump-merged", metavar="FILE",
                        help="Write every scenario's merged config and own entry (sorted keys, JSON lines), "
                             "to check that a config restructure changed nothing")
    parser.add_argument("--explain", metavar="SCENARIO",
                        help="Show the layers a scenario is built from and which file set each value")
    parser.add_argument("--path", metavar="PREFIX", default="",
                        help="With --explain: only the values under this dotted path (e.g. market.wholesale)")

    args = parser.parse_args()

    if args.explain:
        out = explain_scenario(args.explain)
        print(f"{args.explain} is merged from, in order:")
        for i, label in enumerate(out["layers"], 1):
            print(f"  {i:2d}. {label}")
        print("  then the loader's own derivations (targets, method, year parameters, presets)\n")
        shown = [row for row in out["values"] if row[0].startswith(args.path)]
        width = min(70, max((len(p) for p, _, _ in shown), default=10))
        for path, value, label in shown:
            print(f"{path:{width}s} = {str(value)[:60]:60s}  [{label}]")
        sys.exit(0)

    config = load_config()

    if args.dump_merged:
        raw = load_scenario_entries()
        with open(args.dump_merged, "w", encoding="utf-8") as handle:
            for sid in sorted(config["scenarios"]):
                handle.write(json.dumps({"id": sid, "entry": raw[sid], "config": config["scenarios"][sid]},
                                        sort_keys=True, default=repr) + "\n")
        print(f"wrote {len(config['scenarios'])} scenarios to {args.dump_merged}")

    elif args.list:
        print("Available scenarios:")
        for sid in sorted(config["scenarios"].keys()):
            desc = config["scenarios"][sid].get("description", "")
            print(f"  {sid}: {desc}")
    
    elif args.active:
        print("Active scenarios (from run_scenarios):")
        for sid in config["run_scenarios"]:
            if sid in config["scenarios"]:
                desc = config["scenarios"][sid].get("description", "")
                print(f"  [ok] {sid}: {desc}")
            else:
                print(f"  [missing] {sid}: NOT FOUND")
    
    elif args.validate:
        print("Validating scenarios...")
        all_errors = []
        for sid, scenario in config["scenarios"].items():
            errors = validate_scenario(scenario)
            all_errors.extend(errors)
        
        if all_errors:
            print(f"\n{len(all_errors)} validation errors:")
            for error in all_errors:
                print(f"  FAIL {error}")
        else:
            print(f"  OK: all {len(config['scenarios'])} scenarios valid")
    
    elif args.scenario:
        scenario = get_scenario(args.scenario, config)
        if args.json:
            print(json.dumps(scenario, indent=2, default=str))
        else:
            print(f"Scenario: {args.scenario}")
            print("-" * 40)
            for key, value in sorted(scenario.items()):
                if not key.startswith("_"):
                    print(f"  {key}: {value}")
    
    else:
        parser.print_help()
