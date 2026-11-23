"""Scenario config params: scripts get the full dict; Snakemake compares a sorted, narrowed repr."""

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "config"))

from config_loader import (  # noqa: E402
    NOT_READ_BY_MODEL, ScenarioConfigParam, model_scenario_param, scenario_param,
)


def test_repr_ignores_key_order():
    a = scenario_param({"market": {"x": 1, "y": [2, {"b": 1, "a": 2}]}, "solve": {"voll": 6000.0}})
    b = scenario_param({"solve": {"voll": 6000.0}, "market": {"y": [2, {"a": 2, "b": 1}], "x": 1}})
    assert repr(a) == repr(b)
    assert repr(scenario_param({"market": {"x": 2}})) != repr(scenario_param({"market": {"x": 1}}))


def test_model_param_hides_blocks_the_model_never_reads():
    base = {"market": {"x": 1}, "public_ownership": {"cases": 1}, "description": "a"}
    edited = {"market": {"x": 1}, "public_ownership": {"cases": 2}, "description": "b"}
    assert repr(model_scenario_param(base)) == repr(model_scenario_param(edited))
    assert repr(scenario_param(base)) != repr(scenario_param(edited))
    assert {"public_ownership", "consumer_bill", "description"} <= set(NOT_READ_BY_MODEL)
    assert "market" not in NOT_READ_BY_MODEL


def test_script_receives_the_full_plain_dict():
    cfg = {"market": {"x": 1}, "public_ownership": {"cases": 1}}
    param = model_scenario_param(cfg)
    assert param == cfg and param["public_ownership"] == {"cases": 1}
    restored = pickle.loads(pickle.dumps(param))
    assert type(restored) is dict and restored == cfg


def test_snakemake_serialises_the_narrowed_repr():
    from snakemake.persistence import Persistence

    param = model_scenario_param({"market": {"x": 1}, "public_ownership": {"cases": 1}})
    serialised = Persistence._serialize_param_builtin(None, param)
    assert serialised == repr(param)
    assert "public_ownership" not in serialised
    assert isinstance(param, ScenarioConfigParam)
