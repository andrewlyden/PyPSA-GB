"""A study's finance and bill settings on its parent scenario entry (FES 2050 plan, decision 31; task D31).

The 2030 finance layer reads one config: ``public_ownership`` and
``consumer_bill.forecasts`` in ``config/defaults/``.  A later study (the FES 2050
year) keeps its own settings on its **parent scenario entry** in
``config/scenarios.yaml`` instead (user decision, 1 Oct 2026: no separate YAML file):

* A *study* is a scenario entry with no ``network_from`` that carries a
  ``public_ownership`` block, a ``consumer_bill.forecasts`` block, or both
  (:data:`FINANCE_BLOCKS`).  ``FES25_HT_2050`` is the one study today.
* A *study group* is a run group its ``public_ownership.reference_run_groups``
  declares.  A run group finds its parent scenario by where it is declared: no key on
  the group names it.  A *study case* is a forecast case its
  ``consumer_bill.forecasts.cases`` declares.
* A study group's settings are ``config/defaults/`` with the study's blocks
  deep-merged over it (:func:`deep_merge`: mappings merge key by key, every other
  value replaces).  Every group declared in defaults (cp30_2030_cp30_network, the P2 and P3 groups,
  the historical groups) resolves to defaults itself, so nothing 2030 can move.

The parent's monthly and gate-week children (``network_from: <study>``) inherit the
blocks through the YAML merge key; a child must carry exactly its parent's blocks or
none, so a finance setting is never declared on a solved month.

Refused at load: a study redefining a defaults group or forecast case, two studies
declaring one group or case, a study overriding a key the rules read only from
defaults (:data:`DEFAULTS_ONLY_KEYS`), and any resolved config that
``reference_runs.validate_public_ownership_config`` rejects.

The rules write a study's resolved config to ``{study}_finance_config.yaml``
(:data:`STUDY_CONFIG_NAME`) and pass it where a 2030 command passes
the config directory; ``python scripts/public_ownership/study_config.py --study S --out F``
writes it.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

if __name__ == "__main__":  # run as python scripts/public_ownership/study_config.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: The scenario-entry blocks a study may carry (key paths).
FINANCE_BLOCKS: tuple[tuple[str, ...], ...] = (("public_ownership",), ("consumer_bill", "forecasts"))
#: ``public_ownership`` keys the rules read once, from defaults, for every group (the
#: layer switch and the workbook snapshot rule).  A study may not set them.
DEFAULTS_ONLY_KEYS: tuple[str, ...] = ("enabled", "source_workbook", "assumptions_snapshot")
#: File name of a study's resolved config under ``resources/public_ownership``.
STUDY_CONFIG_NAME = "{study}_finance_config.yaml"


class StudyConfigError(ValueError):
    """A study block that cannot be resolved."""


def _fail(message: str) -> None:
    raise StudyConfigError(f"study finance config: {message}")


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """``override`` over ``base``: mappings merge key by key, every other value replaces.

    The same rule as ``config/config_loader.deep_merge`` (scenario over defaults) and
    the E2E harness's merge; neither input is modified.
    """
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get_path(tree: Mapping, path: tuple[str, ...]):
    """The value at ``path`` in ``tree``, or None where any level is missing or not a mapping."""
    node = tree
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _blocks(entry: Mapping) -> dict:
    """The finance blocks ``entry`` carries, nested at their own paths (``{"consumer_bill": {"forecasts": ...}}``)."""
    out: dict = {}
    for path in FINANCE_BLOCKS:
        value = _get_path(entry, path)
        if value is None:
            continue
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return out


def _flat_blocks(blocks: Mapping) -> dict:
    """``{"public_ownership": ..., "consumer_bill.forecasts": ...}``: each finance block by its dotted path."""
    return {".".join(path): _get_path(blocks, path) for path in FINANCE_BLOCKS if _get_path(blocks, path) is not None}


def _declared(blocks: Mapping, block: str, key: str) -> dict:
    value = ((_get_path(blocks, tuple(block.split("."))) or {}).get(key)) or {}
    if not isinstance(value, Mapping):
        _fail(f"{block}.{key} must be a mapping")
    return dict(value)


@dataclass
class Studies:
    """The studies of a scenario file, resolved over one defaults config.

    Attributes:
        defaults: the defaults (``config_loader.load_defaults()``) as loaded (never modified).
        configs: study -> its resolved config (defaults with the study's blocks merged over).
        group_study: study group -> the study that declares it.
        case_study: study forecast case -> the study that declares it.
    """

    defaults: Mapping
    configs: dict = field(default_factory=dict)
    group_study: dict = field(default_factory=dict)
    case_study: dict = field(default_factory=dict)

    def study_of(self, run_group: str | None) -> str | None:
        """The study that declares ``run_group``, or None for a defaults group (or an unknown name)."""
        return self.group_study.get(run_group)

    def config(self, run_group: str | None = None) -> Mapping:
        """The full config ``run_group`` resolves to: its study's, else defaults itself."""
        study = self.study_of(run_group)
        return self.defaults if study is None else self.configs[study]

    def public_ownership(self, run_group: str | None = None) -> Mapping:
        """``public_ownership`` as ``run_group`` resolves it."""
        return self.config(run_group).get("public_ownership") or {}

    def forecast_cases(self, run_group: str | None = None) -> Mapping:
        """``consumer_bill.forecasts.cases`` as ``run_group`` resolves them."""
        return (((self.config(run_group).get("consumer_bill") or {}).get("forecasts") or {}).get("cases")) or {}

    def case(self, case_id: str) -> dict:
        """A study forecast case's entry (the dict in its study's resolved config)."""
        return self.configs[self.case_study[case_id]]["consumer_bill"]["forecasts"]["cases"][case_id]

    def groups(self, study: str | None = None) -> list[str]:
        """Study groups in declaration order (one study's, or every study's)."""
        return [g for g, s in self.group_study.items() if study is None or s == study]

    def cases(self, study: str | None = None) -> list[str]:
        """Study forecast cases in declaration order (one study's, or every study's)."""
        return [c for c, s in self.case_study.items() if study is None or s == study]


def finance_studies(defaults: Mapping, raw_scenarios: Mapping, *, validate: bool = True) -> Studies:
    """Find and resolve every study in ``raw_scenarios`` (``config/scenarios.yaml`` as loaded).

    Args:
        defaults: the defaults (``config_loader.load_defaults()``).
        raw_scenarios: the scenario entries before the defaults merge.
        validate: run ``validate_public_ownership_config`` on each resolved config.

    Returns:
        The :class:`Studies`; empty when no entry carries a finance block.
    """
    studies = Studies(defaults=defaults)
    default_groups = set((defaults.get("public_ownership") or {}).get("reference_run_groups") or {})
    default_cases = set((((defaults.get("consumer_bill") or {}).get("forecasts") or {}).get("cases")) or {})
    default_po = defaults.get("public_ownership") or {}
    for scenario_id, entry in (raw_scenarios or {}).items():
        if not isinstance(entry, Mapping):
            continue
        blocks = _blocks(entry)
        if not blocks:
            continue
        parent = entry.get("network_from")
        if parent:
            if blocks != _blocks(raw_scenarios.get(parent) or {}):
                _fail(f"{scenario_id} shares {parent}'s network (network_from) but carries finance settings that "
                      f"differ from {parent}'s; declare them on the parent entry {parent} only")
            continue
        for block, value in _flat_blocks(blocks).items():
            if not isinstance(value, Mapping):
                _fail(f"{scenario_id}.{block} must be a mapping")
        po = blocks.get("public_ownership") or {}
        for key in DEFAULTS_ONLY_KEYS:
            if key in po and po[key] != default_po.get(key):
                _fail(f"{scenario_id}.public_ownership.{key}: read once from config/defaults/ for every group; "
                      "a study may not change it")
        for group in _declared(blocks, "public_ownership", "reference_run_groups"):
            if group in default_groups:
                _fail(f"{scenario_id} redefines run group {group!r} of config/defaults/")
            if group in studies.group_study:
                _fail(f"run group {group!r} is declared by {studies.group_study[group]} and {scenario_id}")
            studies.group_study[group] = scenario_id
        for case in _declared(blocks, "consumer_bill.forecasts", "cases"):
            if case in default_cases:
                _fail(f"{scenario_id} redefines forecast case {case!r} of config/defaults/")
            if case in studies.case_study:
                _fail(f"forecast case {case!r} is declared by {studies.case_study[case]} and {scenario_id}")
            studies.case_study[case] = scenario_id
        studies.configs[scenario_id] = deep_merge(defaults, blocks)
    if validate:
        from scripts.public_ownership.reference_runs import validate_public_ownership_config

        for study, config in studies.configs.items():
            if not studies.groups(study):
                continue
            try:
                validate_public_ownership_config(
                    config["public_ownership"],
                    (((config.get("consumer_bill") or {}).get("forecasts") or {}).get("cases")) or {})
            except ValueError as error:
                raise StudyConfigError(f"{study}: {error}") from error
    return studies


def _load_yaml(path: Path) -> dict:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=loader) or {}


def load_studies(config_dir: str | Path = "config", defaults: Mapping | None = None, *,
                 validate: bool = True) -> Studies:
    """:func:`finance_studies` on the scenario entries of ``<config_dir>`` (scenarios.yaml and the studies
    config.yaml lists, ``inherits`` resolved) and its defaults (``load_defaults``) unless given."""
    from config.config_loader import load_defaults, load_scenario_entries

    config_dir = Path(config_dir)
    if defaults is None:
        defaults = load_defaults(config_dir)
    return finance_studies(defaults, load_scenario_entries(config_dir), validate=validate)


def group_config(run_group: str | None, config_dir: str | Path = "config") -> Mapping:
    """The full config ``run_group`` resolves to (defaults for a defaults group), for scripts and tests."""
    return load_studies(config_dir).config(run_group)


def dump_config(config: Mapping) -> str:
    """The resolved config as the rules write it (the E2E harness's YAML form)."""
    return yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True, help="the parent scenario entry carrying the finance blocks")
    parser.add_argument("--out", required=True, help="the resolved config (YAML, the defaults' layout)")
    parser.add_argument("--config-dir", default="config", help="holds defaults/ and scenarios.yaml")
    args = parser.parse_args(argv)
    studies = load_studies(args.config_dir)
    if args.study not in studies.configs:
        _fail(f"{args.study} carries no finance block; studies: {sorted(studies.configs)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_config(studies.configs[args.study]), encoding="utf-8")
    print(f"{args.study}: {len(studies.groups(args.study))} run groups, {len(studies.cases(args.study))} forecast "
          f"cases -> {out}")


if __name__ == "__main__":
    main()
