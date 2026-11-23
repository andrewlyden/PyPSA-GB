"""F27 item 13 (user, 24 Oct 2026): gas balancing income after 2030 follows NESO's constraint-cost forecast.

* A gas row's balancing margin (settlement less the variable cost of the energy redispatched)
  scales by NESO_t / NESO_2030 after the model year; the model year is as solved.
* Years after NESO's last (2035) hold its ratio; renewables stay flat; ``flat`` is the old rule.
* The series follows the run group's network case (cp30: the recommended network).
* The code default equals config/defaults/base/public_ownership/acquisition.yaml, and the data file is the figure's values.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.public_ownership import legacy_valuation as lv

from tests.unit.test_public_ownership_legacy_valuation import PUBLIC_RATE, _row, _stations
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
PATH_FILE = ROOT / "data/public_ownership/financing/neso_constraint_cost_path.csv"


def _gas(**extra):
    # 2 TWh in the wholesale market, 3 TWh after the BM: 1 TWh offered at 120/MWh on a 75/MWh fuel cost.
    return {**_row("ccgt", "CCGT", "merchant", market=200.0e6 + 120.0e6, fuel=225.0e6, commissioning=2020),
            "wholesale_mwh": 2.0e6, "balancing_mwh": 3.0e6, "balancing_settlement_gbp": 120.0e6, **extra}


def _profile(rows, **overrides):
    settings = lv.settings_from_config({"horizon_year": 2070, **overrides})
    valuation, profile, summary = lv.value_legacy_fleet(pd.DataFrame(rows), PUBLIC_RATE, settings, _stations())
    return valuation, profile.groupby(["carrier", "stream", "year"])["gbp"].sum(), summary


def _path(series):
    table = pd.read_csv(PATH_FILE).set_index("year")[series]
    return table / table[2030]


class TestGasMargin:
    def test_margin_is_settlement_less_the_redispatched_variable_cost(self):
        rows = pd.DataFrame([_gas(annual_variable_om_gbp=15.0e6)])
        # (225 + 15) m over 3 TWh = 80/MWh on the 1 TWh moved: 120 - 80 = 40 m.
        assert lv.balancing_margin(rows)[0] == pytest.approx(40.0e6)
        assert lv.balancing_margin(rows.drop(columns="balancing_settlement_gbp"))[0] == 0.0

    def test_gas_margin_follows_the_neso_expected_network_path(self):
        valuation, p, summary = _profile([_gas()])
        ratio = _path("ht_expected_network")
        margin = 45.0e6  # 120 m less 1 TWh at 75/MWh
        model_year = p[("CCGT", "net_operating_cash", 2030)]
        assert model_year == pytest.approx(320.0e6 - 225.0e6)  # as solved
        for year in (2031, 2033, 2035):
            assert p[("CCGT", "net_operating_cash", year)] == pytest.approx(model_year - margin * (1 - ratio[year]))
        # After NESO's last year the 2035 ratio holds.
        assert p[("CCGT", "net_operating_cash", 2040)] == pytest.approx(p[("CCGT", "net_operating_cash", 2035)])
        assert valuation["balancing_income_path"].iloc[0] == "neso_constraint_cost:ht_expected_network"
        assert valuation["balancing_margin_model_year_gbp"].iloc[0] == pytest.approx(margin)
        block = summary["balancing_income_path"]
        assert block["ratio_to_model_year"][2030] == 1.0
        assert block["ratio_to_model_year"][2031] == pytest.approx(ratio[2031], abs=1e-4)

    def test_flat_holds_the_model_year_and_renewables_never_move(self):
        _, flat, summary = _profile([_gas()], balancing_income_path={"method": "flat"})
        assert flat[("CCGT", "net_operating_cash", 2035)] == pytest.approx(flat[("CCGT", "net_operating_cash", 2030)])
        assert summary["balancing_income_path"]["method"] == "flat"
        wind = {**_row("farm", "wind_offshore", "merchant", market=300.0, commissioning=2025),
                "wholesale_mwh": 3.0, "balancing_mwh": 2.0, "balancing_settlement_gbp": 50.0}
        _, p, _ = _profile([wind])
        assert p[("wind_offshore", "net_operating_cash", 2031)] == pytest.approx(300.0)

    def test_a_vintaged_register_keeps_the_old_rule(self):
        _, p, summary = _profile([_gas(model_year=2050, commissioning_year=2045)], horizon_year=2090)
        assert p[("CCGT", "net_operating_cash", 2051)] == pytest.approx(p[("CCGT", "net_operating_cash", 2050)])
        assert summary["balancing_income_path"]["method"] == "flat"

    def test_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="balancing_income_path.method"):
            lv.settings_from_config({"balancing_income_path": {"method": "linear"}})


class TestSeriesAndConfig:
    def test_the_series_follows_the_group_network_case(self):
        settings = lv.settings_from_config({})
        block = {"reference_run_groups": {"cp30_2030_cp30_network": {"network_case": "cp30"},
                                          "cp30_2030_no_acceleration": {"network_case": "no_acceleration"}}}
        lower = lv.group_balancing_path_settings(settings, block, "cp30_2030_cp30_network")
        assert lower["balancing_income_path"]["series"] == "ht_recommended_network"
        assert lv.group_balancing_path_settings(settings, block, "cp30_2030_no_acceleration") == settings
        # A raw block without the key takes the default path's mapping.
        assert lv.group_balancing_path_settings({}, block, "cp30_2030_cp30_network")["balancing_income_path"]["series"] \
            == "ht_recommended_network"

    def test_code_default_equals_defaults_yaml(self):
        block = load_defaults()["public_ownership"]["acquisition"]["valuation"]["balancing_income_path"]
        assert block == lv.DEFAULT_SETTINGS["balancing_income_path"]

    def test_path_file_is_neso_figure_30(self):
        table = pd.read_csv(PATH_FILE).set_index("year")
        # The report: without the three critical projects ~GBP 4 bn more in 2030; identical networks after it.
        gap = table.loc[2030, "ht_expected_network"] - table.loc[2030, "ht_recommended_network"]
        assert gap == pytest.approx(3.94, abs=0.01)
        later = table.loc[table.index != 2030]
        assert (later["ht_expected_network"] == later["ht_recommended_network"]).all()
        assert (later["ee_expected_network"] == later["ee_recommended_network"]).all()
        assert table.index.min() == 2025 and table.index.max() == 2035


def test_declared_item_names_the_gas_path():
    from scripts.public_ownership.declared_items import build, legacy_balancing_sizing

    valuation, _, _ = _profile([_gas(), {**_row("farm", "wind_offshore", "merchant", market=300.0, commissioning=2025),
                                          "wholesale_mwh": 3.0, "balancing_mwh": 2.0, "balancing_settlement_gbp": 50.0}])
    settlement = pd.DataFrame({"asset_id": ["ccgt", "farm"], "balancing_settlement_gbp": [120.0e6, 50.0]})
    sizing = legacy_balancing_sizing(valuation, settlement)
    assert sizing["gas_balancing_path"]["path"] == "neso_constraint_cost:ht_expected_network"
    assert sizing["gas_balancing_path"]["margin_model_year_gbp_m_by_carrier"] == {"CCGT": pytest.approx(45.0)}
    rows = pd.DataFrame({"asset_id": ["x"], "carrier": ["CCGT"], "cfd_round": [""], "capacity_mw": [1.0]})
    item = next(i for i in build({}, rows, household_per_system=1e-8, legacy_balancing=sizing)["items"]
                if i["id"] == "legacy_balancing_income")
    assert "NESO" in item["label"] and "Figure 30" in item["why"]
    flat, _, _ = _profile([_gas()], balancing_income_path={"method": "flat"})
    item = next(i for i in build({}, rows, household_per_system=1e-8,
                                 legacy_balancing=legacy_balancing_sizing(flat, settlement))["items"]
                if i["id"] == "legacy_balancing_income")
    assert item["label"] == "Balancing income in the legacy valuation, held at 2030"
