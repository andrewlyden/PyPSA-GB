"""F9 and F11: the 2030/31 T-4 re-cleared with state bids at cost, and the Capacity Market's recovery options.

F9: the state's own new-build and refurbishing supply bids its missing money at its own
cost of capital, the auction clears lower, every 2030/31 winner is paid the new price,
and the bill books the change on its own row; the state's winners' part is the
Exchequer's, the rest private investors', and the three-party identity closes with no
resource saving.  F11: flat and taxation recovery are reported beside the headline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.market import capacity_market as cm
from scripts.public_ownership import capacity_market_state_bids as sb
from scripts.public_ownership import investor_ledger as inv
from scripts.public_ownership import party_flows
from scripts.public_ownership.exchequer_ledger import cm_recovery_option_columns, per_household
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    LINE_CM_STATE_BIDS, NEW_LINES, cm_recovery_options, cm_state_bids_row, cm_state_bids_share,
)
from config.config_loader import load_defaults

M = 1.0e6
ON = {**sb.DEFAULT_SETTINGS, "enabled": True}
CURVE = cm.DemandCurve(target_mw=40900.0, min_at_cap_mw=39400.0, max_at_zero_mw=42400.0, net_cone=49.0, price_cap=75.0)
BIDS = {"existing": (0.9966, 5.2135), "refurbishing": (0.8633, 5.2135), "new_build": (0.4991, 5.2135),
        "dsr": (0.8232, 5.2135), "interconnector": (1.0, 5.2135), "ldes_cap_and_floor": (0.9966, 5.2135)}


def _auction(tmp_path):
    supply = pd.DataFrame({
        "asset_id": ["gas_old", "gas_refurb", "bat_state", "bat_private", "nb_state", "ldes_state", "dsr", "ic"],
        "carrier": ["CCGT", "CCGT", "battery", "battery", "OCGT", "long_duration_storage_cap_and_floor", "dsr",
                    "interconnector"],
        "category": ["existing", "refurbishing", "refurbishing", "refurbishing", "new_build", "ldes_cap_and_floor",
                     "dsr", "interconnector"],
        "supply_mw": [16000.0, 9000.0, 2000.0, 1000.0, 1500.0, 4000.0, 3000.0, 7600.0],
        "case": "central"})
    entered = supply.groupby("category")["supply_mw"].sum()
    model = {"distributions": {c: {"floor_share": f, "shape": s} for c, (f, s) in BIDS.items() if c != "ldes_cap_and_floor"}}
    base = cm.clear_auction(entered, BIDS, CURVE)
    model["t4_2030_31"] = {"central": {"price": round(base["price"], 2), "awarded_mw": round(base["awarded_mw"], 1)}}
    calendar = {"calendar_2030_cost_gbp": 2.4e9,
                "delivery_years": {"2030/2031": {"t4_modelled_gbp": model["t4_2030_31"]["central"]["awarded_mw"] * 1e3
                                                 * model["t4_2030_31"]["central"]["price"]}}}
    paths = []
    for name, payload in (("model.json", model), ("calendar.json", calendar)):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    supply_path = tmp_path / "supply.csv"
    supply.to_csv(supply_path, index=False)
    return sb.load_auction(paths[0], supply_path, paths[1], ON, curves={"T-4 2030/31": CURVE})


def _rows(share, cap_case, ids=("bat_state", "nb_state", "ldes_state")):
    return pd.DataFrame({"asset_id": list(ids), "ownership_share": share, "finance_vintage": "new_build",
                         "annual_capital_recovery_gbp": cap_case, "annual_fixed_om_gbp": 20.0})


PRIVATE = _rows(0.0, 80.0)


class TestBidsAndSettings:
    def test_a_scaled_bid_reaches_the_same_share_at_a_lower_price(self):
        assert cm.bid_share(30.0, 0.5, 2.0, 75.0, 0.8) == pytest.approx(cm.bid_share(37.5, 0.5, 2.0, 75.0))
        assert cm.bid_share(30.0, 0.5, 2.0) == cm.bid_share(30.0, 0.5, 2.0, 75.0, 1.0)
        with pytest.raises(ValueError, match="positive"):
            cm.bid_share(30.0, 0.5, 2.0, 75.0, 0.0)

    def test_code_defaults_equal_the_config_and_bad_keys_are_refused(self):
        po = load_defaults()["public_ownership"]
        settings = sb.settings_from_config(po)
        assert settings["enabled"] is True
        assert {k: v for k, v in settings.items() if k != "enabled"} == {
            k: v for k, v in sb.DEFAULT_SETTINGS.items() if k != "enabled"}
        with pytest.raises(ValueError, match="unknown keys"):
            sb.settings_from_config({sb.CONFIG_KEY: {"price": 1}})
        with pytest.raises(ValueError, match="categories"):
            sb.settings_from_config({sb.CONFIG_KEY: {"categories": ["interconnector"]}})

    def test_the_bid_scale_is_the_capital_part_at_the_case_rate(self):
        # Capital 80 of fixed 100 at the private rate; the state's capital charge is 40: k = 1 - 0.5 x 0.8 = 0.6.
        factors = sb.row_bid_factors(_rows(1.0, 40.0), PRIVATE)
        assert factors.loc["bat_state", "k"] == pytest.approx(0.6)
        assert factors["state_owned"].all()
        # The private case's own rows bid as they are.
        assert sb.row_bid_factors(PRIVATE, PRIVATE)["k"].eq(1.0).all()
        assert not sb.row_bid_factors(PRIVATE, PRIVATE)["state_owned"].any()


class TestReclear:
    def test_unchanged_supply_reproduces_the_builder_and_moves_nothing(self, tmp_path):
        auction = _auction(tmp_path)
        result = sb.reclear(auction, sb.row_bid_factors(PRIVATE, PRIVATE), ON)
        assert result["calendar_2030_cost_change_gbp_2025"] == pytest.approx(0.0, abs=1e-6)
        assert result["state_award_share"] == 0.0

    def test_state_bids_at_cost_lower_the_price_paid_to_every_winner(self, tmp_path):
        auction = _auction(tmp_path)
        result = sb.reclear(auction, sb.row_bid_factors(_rows(1.0, 40.0), PRIVATE), ON)
        assert result["price_change_gbp_per_kw_year"] < 0.0
        assert result["calendar_2030_cost_change_gbp_2025"] < 0.0
        # Half of 2030/31: the T-4 award and the 1 GW T-1 at the new price.
        expected = 0.5 * 1e3 * auction["money_index"] * (
            (result["awarded_mw"] + 1000.0) * result["price_gbp_per_kw_year"]
            - (auction["awarded_mw"] + 1000.0) * auction["price"])
        assert result["calendar_2030_cost_change_gbp_2025"] == pytest.approx(expected)
        assert result["calendar_2030_cost_change_share"] == pytest.approx(expected / 2.4e9)
        # New build and refurbishing split off; the price-taking LDES never does.
        assert set(result["state_bidding_mw"]) == {"new_build@state", "refurbishing@state"}
        assert 0.0 < result["state_award_share"] < 1.0

    def test_a_stake_bids_whole_at_its_blended_cost_and_is_paid_its_share(self, tmp_path):
        auction = _auction(tmp_path)
        full = sb.reclear(auction, sb.row_bid_factors(_rows(1.0, 40.0), PRIVATE), ON)
        # A 30% stake: the joint venture's capital charge 0.3 x 40 + 0.7 x 80 = 68.
        stake = sb.reclear(auction, sb.row_bid_factors(_rows(0.3, 68.0), PRIVATE), ON)
        assert full["price_change_gbp_per_kw_year"] < stake["price_change_gbp_per_kw_year"] < 0.0
        assert stake["state_award_share"] < full["state_award_share"]

    def test_only_capital_categories_move(self, tmp_path):
        auction = _auction(tmp_path)
        none = sb.reclear(auction, sb.row_bid_factors(_rows(1.0, 40.0), PRIVATE), dict(ON, categories=["existing"]))
        assert none["calendar_2030_cost_change_gbp_2025"] == pytest.approx(0.0, abs=1e-6)

    def test_a_mismatched_supply_table_is_refused(self, tmp_path):
        _auction(tmp_path)
        model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
        model["t4_2030_31"]["central"]["price"] += 5.0
        (tmp_path / "model.json").write_text(json.dumps(model), encoding="utf-8")
        with pytest.raises(ValueError, match="different builds"):
            sb.load_auction(tmp_path / "model.json", tmp_path / "supply.csv", tmp_path / "calendar.json", ON,
                            curves={"T-4 2030/31": CURVE})


CONV = {"factor": 1.04, "final_demand_mwh": 300.0e6, "mwh": 3.1, "vat_rate": 0.05, "domestic_mwh": 100.0e6}


class TestBill:
    def test_the_row_is_its_share_of_the_private_line(self):
        block = {"calendar_2030_cost_change_share": -0.003, "state_part_share": 0.12,
                 "base_price_gbp_per_kw_year": 49.54, "price_gbp_per_kw_year": 49.2}
        account = {sb.CONFIG_KEY: block}
        private = {sb.CONFIG_KEY: dict(block, calendar_2030_cost_change_share=0.0)}
        assert cm_state_bids_share(account, private) == pytest.approx(-0.003)
        rows = cm_state_bids_row(account, private_account=private, cm_line=30.0, account_source="a")
        assert rows[0]["component_id"] == LINE_CM_STATE_BIDS and LINE_CM_STATE_BIDS in NEW_LINES
        assert rows[0]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-0.09)
        assert "49.54 -> 49.20" in rows[0]["method"]
        assert cm_state_bids_row({}, private_account=private, cm_line=30.0, account_source="a") == []

    def test_recovery_options_flat_and_taxation(self):
        options = cm_recovery_options(30.0, cm_factor=1.348, final_demand_mwh=CONV["final_demand_mwh"],
                                      factor=CONV["factor"], mwh=CONV["mwh"])
        flat = options["b_flat_per_mwh"]
        assert flat["household_line_gbp_per_3100kwh_ex_vat"] == pytest.approx(30.0 / 1.348)
        assert flat["bill_change_gbp_per_3100kwh_ex_vat"] == pytest.approx(30.0 / 1.348 - 30.0)
        assert -flat["household_share_change"] == pytest.approx(0.258, abs=1e-3)   # "about a quarter"
        # The whole system cost, which the household line overstates by its peak factor.
        assert options["system_cost_gbp_2024"] == pytest.approx(per_household(1.0, CONV) ** -1 * 30.0 / 1.348)
        assert options["c_taxation"]["bill_change_gbp_per_3100kwh_ex_vat"] == -30.0

        row = {"bill_inc_vat": 1000.0, "taxpayer_share_gbp_per_household": 5.0, "cm_recovery_options": options}
        cm_recovery_option_columns(row, CONV)
        assert row["cm_option_b_flat_bill_gbp_inc_vat"] == pytest.approx(1000.0 + (30.0 / 1.348 - 30.0) * 1.05)
        assert row["cm_option_c_taxation_bill_gbp_inc_vat"] == pytest.approx(1000.0 - 31.5)
        # The taxpayer pays the system cost and the domestic VAT no longer charged.
        assert row["cm_option_c_taxpayer_share_gbp_per_household"] > 5.0 + 30.0 / 1.348
        assert row["cm_option_c_household_total_gbp_inc_vat"] == pytest.approx(
            row["cm_option_c_taxation_bill_gbp_inc_vat"] + row["cm_option_c_taxpayer_share_gbp_per_household"])
        untouched = {"bill_inc_vat": 1.0}
        cm_recovery_option_columns(untouched, CONV)
        assert untouched == {"bill_inc_vat": 1.0}


class TestThreeParty:
    POSITION = {"private_rent_gbp": 0.0, "required_revenue_gbp": 0.0, "capital_recovery_gbp": 0.0,
                "corporation_tax_forgone_gbp": 0.0}

    def test_a_transfer_from_the_winners_with_no_resource_saving(self):
        from tests.unit.test_public_ownership_investor_ledger import _bill_case

        case = _bill_case(capacity_market_state_bids_gbp_2024=-10 * M,
                          capacity_market_state_bids_state_part_gbp_2024=-1.2 * M)
        parties = inv.market_case_parties(bill_case=case, bill_private=_bill_case(), account_case={},
                                          position=self.POSITION, position_private=self.POSITION,
                                          network=inv.network_parties(None, "private", None))
        row = inv.three_party_row(parties, vat_change_gbp=0.0, exchequer_change_gbp=-1.2 * M, case_id="p2")
        assert row["consumer_change_vs_private_gbp"] == pytest.approx(10 * M)
        assert row["investor_change_vs_private_gbp"] == pytest.approx(-8.8 * M)
        assert row["resource_saving_vs_private_gbp"] == pytest.approx(0.0, abs=1e-6)

        flows = party_flows.Flows("p2", lambda gbp: gbp / 1e6)
        components = {k: 0.0 for k in ("state_capital", "private_capital", "state_operating", "private_operating",
                                       "corporation_tax", "takeover_positive", "takeover_negative",
                                       "carbon_cfd_private", "tax_equivalent_private", "ldes_state", "ldes_private")}
        account = {"capacity_market_state_bids": {"state_part_share": 0.12}}
        party_flows.market_flows(flows, parties=parties, components=components, account_case=account,
                                 network=inv.network_parties(None, "private", None), levers={})
        # A lower line reverses each flow: consumers are paid by the Exchequer and by the private winners.
        assert sum(abs(r["gbp"]) for r in flows.rows) == pytest.approx(10 * M)
        party_flows.check(flows.rows, {"case_id": "p2", "exchequer_net_vs_private_gbp": -1.2 * M, **row})
