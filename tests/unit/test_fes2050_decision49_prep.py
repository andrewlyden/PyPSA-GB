"""FES 2050 decision 49 prep: onshore at 25 years in the stock-flow (decision 48), the BM at the
bill's calibrated strikes (49 (b)), and the configuration that keeps the two sides in step."""

import pandas as pd
import pytest
import yaml

from scripts.public_ownership import fleet_vintages as fv
from scripts.public_ownership import reference_runs as rr
from scripts.public_ownership import vintage_capital_2050 as vk
from config.config_loader import load_scenario_entries
from tests.helpers import requires_outputs


# ── decision 48: per-technology life overrides ───────────────────────────────

def test_life_override_changes_only_the_named_technology():
    base = fv.load_lives()
    onshore = fv.load_lives(overrides={"Onshore wind": 25})
    assert onshore["Onshore wind"] == 25.0
    assert {k: v for k, v in onshore.items() if k != "Onshore wind"} == \
           {k: v for k, v in base.items() if k != "Onshore wind"}
    assert fv.load_lives(overrides={}) == base == fv.load_lives(overrides=None)


def test_life_override_refuses_unknown_technology_and_bad_life():
    with pytest.raises(ValueError, match="unknown stock-flow"):
        fv.load_lives(overrides={"Onshore": 25})
    with pytest.raises(ValueError, match="positive"):
        fv.load_lives(overrides={"Onshore wind": 0})


def test_parse_life_overrides_from_command_line_and_config():
    assert fv.parse_life_overrides(["Onshore wind=25"]) == {"Onshore wind": 25.0}
    assert fv.parse_life_overrides({"Onshore wind": 25}) == {"Onshore wind": 25.0}
    assert fv.parse_life_overrides([]) == {} == fv.parse_life_overrides(None)
    with pytest.raises(ValueError):
        fv.parse_life_overrides(["Onshore wind"])


def test_reference_runs_validates_life_overrides_only_when_set():
    group = {"fleet_vintages": {"life_source": "central"}}
    rr._validate_vintaged_group("g", group, 2050)
    rr._validate_vintaged_group("g", {"fleet_vintages": {"life_overrides": {"Onshore wind": 25}}}, 2050)
    with pytest.raises(ValueError, match="life_overrides"):
        rr._validate_vintaged_group("g", {"fleet_vintages": {"life_overrides": {"Onshore": 25}}}, 2050)


# ── decision 49 (b): the BM strike basis ─────────────────────────────────────

def test_bm_strike_basis_defaults_to_economic_life_and_is_validated():
    assert vk.basis_from_config({})["bm_strike_basis"] == "economic_life"
    with pytest.raises(vk.VintageCapitalError):
        vk.basis_from_config({vk.CONFIG_KEY: {"bm_strike_basis": "contract"}})
    with pytest.raises(vk.VintageCapitalError, match="calibrated needs"):
        vk.basis_from_config({vk.CONFIG_KEY: {"bm_strike_basis": "calibrated"}})


def _ht50_months():
    scenarios = load_scenario_entries("config")
    return scenarios


def test_ht50_labels_and_bill_agree_on_the_bm_strikes():
    """The months' support_labels.strike_variant and FES25_HT_2050's bm_strike_basis say the same thing."""
    from scripts.public_ownership.study_config import load_studies

    scenarios = _ht50_months()
    block = (scenarios["FES25_HT_2050_Jan_market"]["support"]["bm_bids"]
             ["contract_value_bids"]["support_labels"])
    basis = vk.basis_from_config(load_studies("config", validate=False).configs["FES25_HT_2050"])
    assert (block["strike_variant"] == "calibrated") == (basis["bm_strike_basis"] == "calibrated")
    assert block["fleet_vintages"]["life_overrides"] == {"Onshore wind": 25}
    assert not basis["held_private_carriers"]


@requires_outputs("resources/public_ownership/workbook_v5_assumptions.yaml")
def test_calibrated_labels_price_cfds_at_the_bill_strikes(tmp_path):
    """The labels' calibrated StrikeTable equals the bill's strike table (BECCS left unpriced), and
    the strike-table rule's --bm-out file is the contract table itself."""
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.market import support_labels_2050 as sl
    from scripts.public_ownership import contract_book_2050 as cb
    from scripts.public_ownership.study_config import load_studies

    config = tmp_path / "FES25_HT_2050_finance_config.yaml"
    config.write_text(yaml.safe_dump(load_studies("config", validate=False).configs["FES25_HT_2050"]), encoding="utf-8")
    table, bm = tmp_path / "table.csv", tmp_path / "bm.csv"
    cb.build_strike_table_file(table, config_path=config, bm_out=bm)
    pd.testing.assert_frame_equal(pd.read_csv(table), pd.read_csv(bm))
    labels = sl.strike_table("calibrated", cfd50.known_contracts(2050), model_year=2050, study="FES25_HT_2050")
    bill = cfd50.strike_table_from_frame(pd.read_csv(table))
    expected = {k: v for k, v in bill.strikes.items() if k[1] != "beccs"}
    assert set(labels.strikes) == set(expected)
    assert all(abs(labels.strikes[k] - expected[k]) < 1e-9 for k in expected)
