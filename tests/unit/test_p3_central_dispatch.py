"""P3's central case is central dispatch (the user's decision, 23 September 2026).

The public power system's headline is the market-value purchase run by a
nationalised operator dispatching at resource cost. The institution comparison
therefore takes the market-value tariff as its I1 and moves four measured lines;
the market-kept case is P3's alternative. There is one institution case, and its
I1 twin must differ from it in the dispatch institution alone.
"""

import copy
from pathlib import Path

import pytest
import yaml

from scripts.public_ownership.reference_runs import (
    configured_case_specs, institution_market_twin, validate_public_ownership_config,
)
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
P3 = "public_ownership@fully_nationalised_market_value_central_dispatch"
P3_MARKET = "public_ownership@fully_nationalised_market_value"


@pytest.fixture()
def config():
    defaults = load_defaults()
    return copy.deepcopy(defaults["public_ownership"])


def test_the_central_case_is_the_market_value_purchase_under_central_dispatch(config):
    assert institution_market_twin(config) == (P3, P3_MARKET)
    specs = {spec.case_id: spec for spec in configured_case_specs(config)}
    moved = {k for k in specs[P3].settings if specs[P3].settings[k] != specs[P3_MARKET].settings[k]}
    assert moved == {"dispatch_institution"}
    assert specs[P3].settings["purchase_price_basis"] == "market_value"


def test_a_second_institution_case_is_refused(config):
    config["sensitivities"]["named"]["state_value_central"] = {
        "extends": "fully_nationalised", "dispatch_institution": "nationalised_dispatch"}
    with pytest.raises(ValueError, match="one I1/I3 pair"):
        validate_public_ownership_config(config)


def test_an_institution_case_without_a_market_twin_is_refused(config):
    named = config["sensitivities"]["named"]
    named["fully_nationalised_market_value_central_dispatch"]["acquisition_term_years"] = "15"
    del named["fully_nationalised_market_value_term_15y"]
    with pytest.raises(ValueError, match="identical but for dispatch_institution"):
        validate_public_ownership_config(config)


def test_the_comparison_takes_the_named_tariff_case_as_i1(tmp_path, monkeypatch):
    from scripts.public_ownership import nationalised_institutions as ni

    def report(ex_vat):
        return {"conversion": {"final_demand_mwh": 3.1e8, "factor": 1.0, "mwh": 3.1, "vat_rate": 0.05},
                "revenue_requirement_gbp": {ni.LINE_FUEL: 1e9},
                "bill": {"ex_vat_gbp": ex_vat, "private_ex_vat_gbp": 900.0, "private_total_gbp_inc_vat": 945.0,
                         "delta_inc_vat_gbp": ex_vat * 1.05 - 945.0}}
    tariff = {**report(800.0), "primary_case_id": "state_value",
              "cases": {"state_value": report(800.0), "market_value": report(780.0)}}
    path = tmp_path / "tariff.json"
    path.write_text(__import__("json").dumps(tariff), encoding="utf-8")
    measured = {"fuel_carbon_gbp": 0.0, "fuel_gbp": 0.0, "carbon_gbp": 0.0, "variable_om_gbp": 0.0,
                "interconnector_cost_gbp": 0.0}
    monkeypatch.setattr(ni, "measure_dispatch", lambda scenarios, rates, heat_credit=None, stage="balancing": dict(measured))
    monkeypatch.setattr(ni, "measure_link_trade", lambda scenarios, stage="balancing", schedule_scenarios=None: {})
    monkeypatch.setattr(ni.pd, "read_csv", lambda *a, **k: ni.pd.DataFrame({"generator": ["g"]}))
    monkeypatch.setattr(ni, "energy_table", lambda market, central: ni.pd.DataFrame())
    out = tmp_path / "summary.json"
    ni.main([str(path), str(out), str(tmp_path / "comparison.csv"), "--market-scenarios", "m",
             "--central-scenarios", "c", "--marginal-costs", "x.csv", "--case", "market_value",
             "--institution-case", "p3"])
    summary = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert summary["i1_ex_vat_gbp"] == pytest.approx(780.0)
    assert (summary["i1_case_id"], summary["institution_case_id"]) == ("market_value", "p3")
    with pytest.raises(ValueError, match="is not a case"):
        ni.main([str(path), str(out), str(tmp_path / "comparison.csv"), "--market-scenarios", "m",
                 "--central-scenarios", "c", "--marginal-costs", "x.csv", "--case", "nope"])
