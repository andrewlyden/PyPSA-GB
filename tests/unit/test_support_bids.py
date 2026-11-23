"""Unit tests for the 2050 balancing bid stack (task BM2050; decisions 35, 22, 13).

The 2050 generators are clustered: each holds several build years and so several
contracts. These tests pin the blend (weights from the label table), the CfD bid
per contract, the DPA net cost both ways, the hydrogen turbines' DPA prices, the
state share under P2, and that no 2030 path can reach any of it.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market import support_bids as sb
from scripts.market import support_labels as sl
from scripts.market.cfd_bm_bids import build_contract_value_bid_frames
from scripts.market.cfd_strike_table import StrikeTable
from config.config_loader import load_defaults

LOG = logging.getLogger("test_support_bids")
REF_SRMC, MARGIN = 86.45, 1.0


def _label(component, carrier, weight, support, *, year=2040, strike=np.nan, live=np.nan, rule="",
           reference="", component_type="Generator", cfd_round=None):
    dpa = support == "dpa"
    return {"component": component, "component_type": component_type, "carrier": carrier,
            "cohort_id": f"{component}::b{year}", "stock_flow_technology": "", "build_year": year,
            "cohort_capacity_mw": weight, "weight": weight, "support_type": support, "cfd_round": cfd_round,
            "cfd_contract_id": None, "live_share": live, "strike_gbp_per_mwh_2024": strike, "strike_key": "",
            "negative_price_rule": rule, "reference_price": reference,
            "dpa_reference_srmc_gbp_per_mwh": REF_SRMC if dpa else np.nan,
            "dpa_margin_gbp_per_mwh": MARGIN if dpa else np.nan, "support_basis": ""}


@pytest.fixture
def labels():
    rows = [
        # offshore: 60% PAR7 at 80 (live), 40% merchant (pre-2025 build)
        _label("off", "wind_offshore", 0.6, "cfd", year=2035, strike=80.0, live=1.0, rule="ar4_plus",
               reference="intermittent_hourly", cfd_round="PAR7_2035"),
        _label("off", "wind_offshore", 0.4, "merchant", year=2020),
        # onshore: one AR7a phase half live in the year
        _label("on", "wind_onshore", 1.0, "cfd", year=2028, strike=72.24, live=0.5, rule="ar4_plus",
               reference="intermittent_hourly", cfd_round="AR7a"),
        # gas CCS: half on a DPA, half merchant (pre-2036)
        _label("ccs", "CCS Gas", 0.5, "dpa", year=2040, live=1.0, cfd_round="DPA_2040"),
        _label("ccs", "CCS Gas", 0.5, "merchant", year=2030),
        # nuclear is excluded from support pricing
        _label("nuc", "nuclear", 1.0, "cfd", year=2045, strike=105.9, live=1.0, rule="ar4_plus",
               reference="baseload_seasonal", cfd_round="PAR7_2045"),
        # hydrogen turbine link: 80% DPA
        _label("h2t", "H2_turbine", 0.8, "dpa", year=2040, live=1.0, component_type="Link", cfd_round="DPA_2040"),
        _label("h2t", "H2_turbine", 0.2, "merchant", year=2030, component_type="Link"),
    ]
    return pd.DataFrame(rows, columns=list(sl.LABEL_COLUMNS))


@pytest.fixture
def network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-10", periods=4, freq="h"))
    n.add("Bus", "bus", v_nom=400, carrier="AC")
    n.add("Bus", "h2", carrier="H2_gas")
    n.add("Load", "load", bus="bus", p_set=1000.0)
    for name, carrier, mc in (("off", "wind_offshore", 0.0), ("on", "wind_onshore", 0.0), ("ccs", "CCS Gas", 66.33),
                              ("nuc", "nuclear", 10.0), ("bio", "biogas", 81.0)):
        n.add("Generator", name, bus="bus", carrier=carrier, p_nom=1000.0, marginal_cost=mc)
    n.add("Link", "h2t", bus0="h2", bus1="bus", carrier="H2_turbine", p_nom=1000.0, efficiency=0.38, marginal_cost=5.0)
    return n


@pytest.fixture
def reference(network):
    # one hour above every strike, one negative, two in between
    return pd.Series([100.0, -5.0, 30.0, 60.0], index=network.snapshots)


def _bm_bids(**extra):
    """A scenario's ``support.bm_bids`` block."""
    cvb = {"enabled": True, "bid_basis": "contract_value", "negative_price_rule": "gb_contract_terms",
           "turn_up_floor_gbp_per_mwh": 2.0, "merchant": {"bid_price": 20.0, "carriers": ["wind_offshore",
                                                                                            "wind_onshore", "solar_pv"]},
           "support_labels": {"file": "unused.csv", "dpa_carriers": ["CCS Gas"]}}
    cvb.update(extra)
    return {"enabled": True, "mode": "fixed",
            "exclude_carriers": ["nuclear", "CCS Biomass", "biomass"],
            "contract_value_bids": cvb}


def _scenario(base=None, **extra):
    """``base`` (a scenario config) with ``support.bm_bids`` set to ``_bm_bids(**extra)``."""
    scenario = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (base or {}).items()}
    scenario["support"] = {**(scenario.get("support") or {}), "bm_bids": _bm_bids(**extra)}
    return scenario


STATIC_OFFER = {"off": 0.0, "on": 0.0, "ccs": 85.57, "nuc": 999.0, "bio": 89.1}
STATIC_BID = {"off": 90.0, "on": 25.0, "ccs": -59.70, "nuc": 150.0, "bio": -72.9}


def _frames(network, labels, reference, **extra):
    offer = pd.Series(STATIC_OFFER).reindex(network.generators.index)
    bid = pd.Series(STATIC_BID).reindex(network.generators.index)
    return sb.build_label_bid_frames(network, _scenario(**extra), {}, offer, bid, reference, LOG, labels=labels)


@pytest.mark.unit
class TestGeneratorBlend:
    def test_cfd_and_merchant_cohorts_blend_by_weight(self, network, labels, reference):
        frames = _frames(network, labels, reference)
        cfd = np.array([0.0, 0.0, 50.0, 20.0])          # max(80 - ref, 0), 0 in the negative hour
        np.testing.assert_allclose(frames.bid_tv["off"].to_numpy(), 0.6 * cfd + 0.4 * 20.0)

    def test_a_part_live_contract_is_merchant_for_the_rest_of_the_year(self, network, labels, reference):
        frames = _frames(network, labels, reference)
        cfd = np.array([0.0, 0.0, 42.24, 12.24])
        np.testing.assert_allclose(frames.bid_tv["on"].to_numpy(), 0.5 * cfd + 0.5 * 20.0)

    def test_strike_basis_bids_the_strike_in_paying_hours(self, network, labels, reference):
        frames = _frames(network, labels, reference, bid_basis="strike")
        np.testing.assert_allclose(frames.bid_tv["off"].to_numpy(), 0.6 * np.array([80.0, 0.0, 80.0, 80.0]) + 8.0)

    def test_dpa_share_prices_at_its_net_cost_both_ways(self, network, labels, reference):
        frames = _frames(network, labels, reference)
        net = min(66.33, REF_SRMC - MARGIN)
        np.testing.assert_allclose(frames.offer_tv["ccs"].to_numpy(), 0.5 * net + 0.5 * 85.57)
        np.testing.assert_allclose(frames.bid_tv["ccs"].to_numpy(), -0.5 * net + 0.5 * -59.70)

    def test_excluded_and_unlabelled_generators_keep_their_static_prices(self, network, labels, reference):
        frames = _frames(network, labels, reference)
        for name in ("nuc", "bio"):
            assert (frames.bid_tv[name] == STATIC_BID[name]).all()
            assert (frames.offer_tv[name] == STATIC_OFFER[name]).all()
        assert set(frames.audit["generator"]) == {"off", "on", "ccs"}

    def test_the_turn_up_floor_covers_renewables_only(self, network, labels, reference):
        frames = _frames(network, labels, reference)
        assert set(frames.offer_regulariser.index) == {"off", "on"}
        assert frames.offer_regulariser["off"] == 2.0

    def test_an_unlabelled_support_priced_generator_is_refused(self, network, labels, reference):
        with pytest.raises(ValueError, match="no support label"):
            _frames(network, labels[labels.component.ne("on")], reference)

    def test_the_p3_tie_break_is_refused(self, network, labels, reference):
        with pytest.raises(ValueError, match="tie-break"):
            _frames(network, labels, reference, public_ownership_bids={"enabled": False,
                                                                      "tie_break": {"enabled": True}})


@pytest.mark.unit
class TestStateShares:
    def test_the_state_share_comes_out_of_post_2025_cohorts(self, network, labels, reference, tmp_path):
        table = tmp_path / "shares.csv"
        pd.DataFrame({"case_id": "public_ownership@deployment_share_50", "generator": ["off"], "capacity_mw": 1000.0,
                      "state_capacity_mw": 300.0, "state_share": [0.3], "state_commissioning_year": 2040.0}
                     ).to_csv(table, index=False)
        frames = _frames(network, labels, reference, public_ownership_bids={
            "enabled": True, "case_id": "public_ownership@deployment_share_50", "shares_file": str(table),
            "public_bid_price": 0.0, "state_first_step_gbp_per_mwh": 1.0})
        cfd = np.array([0.0, 0.0, 50.0, 20.0])
        # the 0.6 post-2025 CfD cohort loses 0.3 to the state; the 0.4 merchant (2020) keeps its weight
        np.testing.assert_allclose(frames.bid_tv["off"].to_numpy(), 0.3 * cfd + 0.4 * 20.0 + 0.3 * 0.0)
        assert frames.bid_regulariser["off"] == pytest.approx(0.4)
        row = frames.audit.set_index("generator").loc["off"]
        assert row["state_share"] == pytest.approx(0.3)


@pytest.mark.unit
class TestHydrogenTurbines:
    def test_dpa_turbines_offer_and_bid_at_their_net_cost(self, network, labels):
        lam = pd.Series(36.75, index=network.snapshots)
        inc = pd.DataFrame(1.1 * (5.0 + 36.75), index=network.snapshots, columns=["h2t"])
        dec = pd.DataFrame(-0.9 * (5.0 + 36.75), index=network.snapshots, columns=["h2t"])
        up, down, audit = sb.dpa_link_prices(network, ["h2t"], inc, dec, lam, labels, LOG)
        resource_e = (36.75 + 5.0) / 0.38
        net_e = min(resource_e, REF_SRMC - MARGIN)
        np.testing.assert_allclose(up["h2t"].to_numpy(), 0.2 * inc["h2t"].to_numpy() + 0.8 * 0.38 * net_e)
        np.testing.assert_allclose(down["h2t"].to_numpy(), 0.2 * dec["h2t"].to_numpy() - 0.8 * 0.38 * net_e)
        assert audit["dpa_weight"].iloc[0] == pytest.approx(0.8)

    def test_cheap_hydrogen_keeps_the_turbine_at_its_resource_cost(self, network, labels):
        lam = pd.Series(10.0, index=network.snapshots)
        frame = pd.DataFrame(0.0, index=network.snapshots, columns=["h2t"])
        up, _, _ = sb.dpa_link_prices(network, ["h2t"], frame, frame, lam, labels, LOG)
        np.testing.assert_allclose(up["h2t"].to_numpy(), 0.8 * (10.0 + 5.0))

    def test_an_unlabelled_turbine_is_refused(self, network, labels):
        frame = pd.DataFrame(0.0, index=network.snapshots, columns=["h2t"])
        with pytest.raises(ValueError, match="no support label"):
            sb.dpa_link_prices(network, ["h2t"], frame, frame, pd.Series(36.75, index=network.snapshots),
                               labels[labels.component.ne("h2t")], LOG)


@pytest.mark.unit
class TestDpaNetCost:
    def test_net_cost_is_srmc_less_the_variable_payment(self):
        from scripts.public_ownership import support_regime_2050 as sr

        for srmc in (50.0, 85.0, 109.65, 200.0):
            top_up = max(srmc - REF_SRMC + MARGIN, 0.0)       # support_regime_2050.dpa_settlement
            assert sb.dpa_net_cost(srmc, REF_SRMC, MARGIN) == pytest.approx(srmc - top_up)
        params = sr.load_parameters()
        assert sr._value(params["dpa"]["variable_payment_margin_gbp_per_mwh"]) == 1.0


@pytest.mark.unit
class TestLabelTable:
    def _cohorts(self):
        return pd.DataFrame([
            {"component_type": "Generator", "asset_id": "W::b2035", "carrier": "wind_offshore", "capacity_mw": 60.0,
             "support_type": "CfD", "cfd_round": "PAR7_2035", "cfd_contract_id": None, "cfd_live_share": 1.0,
             "build_year": 2035, "stock_flow_technology": "Offshore wind", "support_basis": "post_ar7_modelled"},
            {"component_type": "Generator", "asset_id": "W::b2030", "carrier": "wind_offshore", "capacity_mw": 30.0,
             "support_type": "CfD", "cfd_round": "AR7", "cfd_contract_id": "AR7_X#2", "cfd_live_share": 0.75,
             "build_year": 2030, "stock_flow_technology": "Offshore wind", "support_basis": "known_contract"},
            {"component_type": "Generator", "asset_id": "W::b2020", "carrier": "wind_offshore", "capacity_mw": 10.0,
             "support_type": "merchant", "cfd_round": None, "cfd_contract_id": None, "cfd_live_share": np.nan,
             "build_year": 2020, "stock_flow_technology": "Offshore wind", "support_basis": "contract_expired_merchant"},
            {"component_type": "Generator", "asset_id": "B::b2031", "carrier": "CCS Biomass", "capacity_mw": 5.0,
             "support_type": "CfD", "cfd_round": "PAR7_2031", "cfd_contract_id": None, "cfd_live_share": 1.0,
             "build_year": 2031, "stock_flow_technology": "Biomass - with CCUS - mature", "support_basis": ""},
            {"component_type": "StorageUnit", "asset_id": "S", "carrier": "Battery", "capacity_mw": 5.0,
             "support_type": None, "cfd_round": None, "cfd_contract_id": None, "cfd_live_share": np.nan,
             "build_year": 2040, "stock_flow_technology": "Battery", "support_basis": ""},
        ])

    def _known(self):
        return pd.DataFrame([{"contract_id": "AR7_X", "phase": 2, "strike_gbp_per_mwh_2024": 89.49,
                              "negative_price_rule": "ar4_plus", "reference_price": "intermittent_hourly",
                              "live_share": 0.75}])

    def _table(self):
        return StrikeTable({("PAR7_2035", "offshore"): 82.0}, 2024, {}, {})

    def test_weights_strikes_and_contracts(self):
        out = sl.label_table(self._cohorts(), strike_table=self._table(), known=self._known(),
                             dpa_reference_srmc=REF_SRMC, dpa_margin=MARGIN).set_index("cohort_id")
        assert set(out["component"]) == {"W", "B"}                         # storage is not a contract holder
        assert out.loc["W::b2035", "weight"] == pytest.approx(0.6)
        assert out.loc["W::b2035", "strike_gbp_per_mwh_2024"] == 82.0
        assert out.loc["W::b2030", "strike_gbp_per_mwh_2024"] == 89.49     # the contract's own (Scottish) strike
        assert out.loc["W::b2030", "live_share"] == 0.75
        assert out.loc["W::b2020", "support_type"] == "merchant"
        assert out.loc["B::b2031", "strike_key"] == "not_priced_excluded_carrier"

    def test_a_cfd_without_a_strike_is_refused(self):
        cohorts = self._cohorts()
        cohorts.loc[0, "cfd_round"] = "PAR7_2036"
        with pytest.raises(ValueError, match="no strike in the model year"):
            sl.label_table(cohorts, strike_table=self._table(), known=self._known(),
                           dpa_reference_srmc=REF_SRMC, dpa_margin=MARGIN)

    def test_component_name_strips_the_cohort(self):
        assert sl.component_name("H2_turbine_gen_H2_674::b2029::hydrogen_ocht") == "H2_turbine_gen_H2_674"


@pytest.mark.unit
class TestTwentyThirtyIsUntouched:
    def test_without_support_labels_the_2030_builder_runs(self, network, reference):
        from scripts.market import cfd_bm_bids

        called = {}

        def fake(*args, **kwargs):
            called["yes"] = True

        original = sb.build_label_bid_frames
        try:
            sb.build_label_bid_frames = fake
            with pytest.raises(ValueError, match="support_type"):
                build_contract_value_bid_frames(network,
                                                {"support": {"labels": {},
                                                             "bm_bids": {"contract_value_bids": {"enabled": True}}}},
                                                {},
                                                pd.Series(0.0, index=network.generators.index),
                                                pd.Series(0.0, index=network.generators.index), reference, LOG)
        finally:
            sb.build_label_bid_frames = original
        assert "yes" not in called
        assert cfd_bm_bids is not None

    def test_store_band_mode_defaults_to_the_stage1_band(self):
        from scripts.market.solve_balancing import resolve_store_band_mode

        assert resolve_store_band_mode({}) == "stage1"
        assert resolve_store_band_mode({"hydrogen_links": {"hold_carriers": ["H2_turbine"]}}) == "stage1"
        assert resolve_store_band_mode({"hydrogen_links": {"store_band": "none"}}) == "none"
        with pytest.raises(ValueError):
            resolve_store_band_mode({"hydrogen_links": {"store_band": "physical"}})

    def test_dpa_link_pricing_is_off_unless_declared(self):
        assert sb.dpa_link_block({}) == {}
        assert sb.dpa_link_block({"hydrogen_links": {"hold_carriers": ["H2_turbine"]}}) == {}
        assert sb.support_labels_block({"contract_value_bids": {"enabled": True}}) == {}

    def test_code_defaults_match_defaults_yaml(self):
        import yaml

        text = load_defaults()
        excl = text["support"]["bm_bids"]["exclude_carriers"]
        assert tuple(excl) == sb.DEFAULT_EXCLUDE == sl.UNPRICED_CARRIERS


def _scenarios():
    import importlib.util
    import sys

    sys.path.insert(0, "config")
    spec = importlib.util.spec_from_file_location("config_loader", "config/config_loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config(Path("config").resolve(), include_archive=True)["scenarios"]   # the gate weeks are archived


@pytest.mark.unit
class TestScenarioEntries:
    @pytest.fixture(scope="class")
    def scenarios(self):
        return _scenarios()

    def test_months_carry_the_2050_bid_stack(self, scenarios):
        for month in ("Jan", "Jul", "Dec"):
            bal = scenarios[f"FES25_HT_2050_{month}_market"]["market"]["balancing"]
            cvb = scenarios[f"FES25_HT_2050_{month}_market"]["support"]["bm_bids"]["contract_value_bids"]
            assert cvb["enabled"] and cvb["support_labels"]["file"] == "resources/market/FES25_HT_2050_support_labels.csv"
            # The hydrogen turbines have no DPA variable payment (user, 26 Oct 2026). Since 11 Nov 2026 (user: the
            # long-range years on 2030's balancing methods) electrolysers and hydrogen turbines are held at Stage 1.
            assert bal["hydrogen_links"]["hold_carriers"] == ["H2_turbine", "electrolysis"]
            assert not bal["hydrogen_links"]["dpa_pricing"]["enabled"]

    def test_the_y8a_gate_keeps_its_bm(self, scenarios):
        bal = scenarios["HT50_1week_gate_seasonal"]["market"]["balancing"]
        assert not scenarios["HT50_1week_gate_seasonal"]["support"]["bm_bids"].get("enabled", False)
        assert bal.get("hydrogen_links", {}).get("hold_carriers", ["H2_turbine"]) == ["H2_turbine"]

    def test_bm_twins_differ_from_their_stage1_source_only_in_balancing(self, scenarios):
        # The balancing stage's settings: market.balancing and the BM bid rules, support.bm_bids.
        twins = ["HT50_1week_gate_seasonal_bm2050", "HT50_1week_gate_seasonal_bm2050_strike",
                 "HT50_1week_gate_seasonal_bm2050_h2band", "HT50_1week_gate_final_contract_value"] + [
            f"FES25_HT_2050_{m}_p2_bids_{p}pct" for m in ("Jan", "Feb", "Dec") for p in (25, 50, 75, 100)]
        for sid in twins:
            child = scenarios[sid]
            parent = scenarios[child["wholesale_from"]]
            for key in set(child) | set(parent):
                if key in ("wholesale_from", "description", "_scenario_id", "market", "support"):
                    continue
                assert child.get(key) == parent.get(key), (sid, key)
            strip = lambda m, drop: {k: v for k, v in (m or {}).items() if k != drop}  # noqa: E731
            assert strip(child["market"], "balancing") == strip(parent["market"], "balancing"), sid
            assert strip(child.get("support"), "bm_bids") == strip(parent.get("support"), "bm_bids"), sid

    def test_p2_months_bid_on_their_share_case(self, scenarios):
        from scripts.public_ownership.ownership_shares import deployment_share_case_id

        for pct in (25, 50, 75, 100):
            for month in ("Jan", "Jun"):
                bids = (scenarios[f"FES25_HT_2050_{month}_p2_bids_{pct}pct"]["support"]["bm_bids"]
                        ["contract_value_bids"]["public_ownership_bids"])
                assert bids["enabled"] and bids["case_id"] == deployment_share_case_id(pct)
                assert bids["shares_file"].endswith(f"fes_ht_2050_generator_state_shares_deployment_share_{pct}.csv")
            assert scenarios[f"FES25_HT_2050_Jun_p2_bids_{pct}pct"]["solve_period"]["start"] == "2050-06-01 00:00"


@pytest.mark.unit
class TestDpaDispatch:
    """Coordinator follow-up: DPA plant in Stage 1 and the annual pass (the 2030 rule at solve time)."""

    def _write(self, labels, tmp_path):
        path = tmp_path / "labels.csv"
        labels.to_csv(path, index=False)
        return path

    def test_turbine_dpa_share_dispatches_at_the_reference_less_margin(self, network, labels, tmp_path):
        block = {"enabled": True, "file": str(self._write(labels, tmp_path)), "generator_carriers": ["CCS Gas"],
                 "link_carriers": ["H2_turbine"], "hydrogen_value_gbp_per_mwh_h2": 36.75}
        sb.apply_dpa_dispatch(network, block, LOG)
        mc = network.links.at["h2t", "marginal_cost"]
        resource_e = (36.75 + 5.0) / 0.38
        dispatch_e = (36.75 + mc) / 0.38
        assert dispatch_e == pytest.approx(0.2 * resource_e + 0.8 * (REF_SRMC - MARGIN))

    def test_gas_ccs_below_the_reference_is_unchanged_and_above_it_is_capped(self, network, labels, tmp_path):
        block = {"enabled": True, "file": str(self._write(labels, tmp_path)), "link_carriers": [],
                 "generator_carriers": ["CCS Gas"]}
        sb.apply_dpa_dispatch(network, block, LOG)
        assert network.generators.at["ccs", "marginal_cost"] == pytest.approx(66.33)
        network.generators.at["ccs", "marginal_cost"] = 120.0
        sb.apply_dpa_dispatch(network, block, LOG)
        assert network.generators.at["ccs", "marginal_cost"] == pytest.approx(0.5 * 120.0 + 0.5 * (REF_SRMC - MARGIN))
        assert network.generators.at["bio", "marginal_cost"] == 81.0

    def test_unknown_keys_and_a_missing_hydrogen_value_are_refused(self, network, labels, tmp_path):
        path = str(self._write(labels, tmp_path))
        with pytest.raises(ValueError, match="unknown keys"):
            sb.apply_dpa_dispatch(network, {"enabled": True, "file": path, "typo": 1}, LOG)
        with pytest.raises(ValueError, match="hydrogen_value"):
            sb.apply_dpa_dispatch(network, {"enabled": True, "file": path}, LOG)

    def test_the_cost_basis_hook_applies_it_only_when_declared(self, network, labels, tmp_path):
        from scripts.market.marginal_cost_basis import apply_marginal_cost_basis

        before = network.links["marginal_cost"].copy()
        assert apply_marginal_cost_basis(network, {"mode": "as_built"}, LOG) == {"mode": "as_built"}
        assert apply_marginal_cost_basis(network, {}, LOG) == {"mode": "as_built"}
        pd.testing.assert_series_equal(network.links["marginal_cost"], before)
        apply_marginal_cost_basis(network, {"mode": "as_built", "dpa_dispatch": {
            "enabled": True, "file": str(self._write(labels, tmp_path)), "hydrogen_value_gbp_per_mwh_h2": 36.75}}, LOG)
        assert network.links.at["h2t", "marginal_cost"] < before["h2t"]


@pytest.mark.unit
class TestFinalSettings:
    @pytest.fixture(scope="class")
    def scenarios(self):
        return _scenarios()

    def test_months_bid_at_the_strike_and_dispatch_dpa_in_stage1(self, scenarios):
        # Bid basis: the archived gate week keeps the strike; since 11 Nov 2026 (user: 2030's balancing methods)
        # the 2050 families bid every CfD at contract value.
        for sid, basis in (("FES25_HT_2050_Mar_market", "contract_value"), ("HT50_1week_gate_final", "strike"),
                           ("FES25_HT_2050_Mar_p2_bids_75pct", "contract_value")):
            market = scenarios[sid]["market"]
            assert scenarios[sid]["support"]["bm_bids"]["contract_value_bids"]["bid_basis"] == basis, sid
            dpa = market["wholesale"]["marginal_cost_basis"]["dpa_dispatch"]
            # Gas CCS only (user, 26 Oct 2026); the declared value follows market.hydrogen_supply_price.
            assert dpa["enabled"] and dpa["generator_carriers"] == ["CCS Gas"] and dpa["link_carriers"] == []
            assert dpa["hydrogen_value_gbp_per_mwh_h2"] == market["hydrogen_supply_price"]["gbp_per_mwh_h2"] == 64.0

    def test_records_keep_their_settings(self, scenarios):
        old = scenarios["HT50_1week_gate_seasonal"]["market"]["wholesale"]
        assert "dpa_dispatch" not in (old.get("marginal_cost_basis") or {})
        for sid, basis in (("HT50_1week_gate_seasonal_bm2050", "contract_value"),
                           ("HT50_1week_gate_seasonal_bm2050_strike", "strike"),
                           ("HT50_1week_gate_seasonal_bm2050_h2band", "contract_value"),
                           ("HT50_1week_gate_final_contract_value", "contract_value")):
            cvb = scenarios[sid]["support"]["bm_bids"]["contract_value_bids"]
            assert cvb["bid_basis"] == basis, sid
        for sid in ("HT50_1week_gate_seasonal_bm2050", "HT50_1week_gate_seasonal_bm2050_strike",
                    "HT50_1week_gate_seasonal_bm2050_h2band"):
            assert scenarios[sid]["market"]["wholesale"] == old, sid

    def test_central_dispatch_months_take_no_dpa_cost(self, scenarios):
        ws = scenarios["FES25_HT_2050_Jan_central_dispatch"]["market"]["wholesale"]
        assert "dpa_dispatch" not in (ws.get("marginal_cost_basis") or {})


# ── year-generic stage 5 (27 Oct 2026) ───────────────────────────────────────

def _cohort(asset, component_type, carrier, mw, support, *, year=2040, live=np.nan, technology="", basis="",
            contract=None, cfd_round=None):
    return {"asset_id": asset, "component_type": component_type, "carrier": carrier, "capacity_mw": mw,
            "support_type": support, "cfd_round": cfd_round, "cfd_contract_id": contract, "cfd_live_share": live,
            "build_year": year, "stock_flow_technology": technology, "support_basis": basis}


@pytest.mark.unit
class TestRenewablesObligation:
    SCENARIO = {"modelled_year": 2030, "support": {"dispatch_costs": {"roc_buyout_prices": {2030: 78.04}}}}

    def _ro_labels(self, labels):
        out = labels.copy()
        on = out["component"].eq("on")
        out.loc[on, ["support_type", "live_share", "ro_banding", "strike_gbp_per_mwh_2024"]] = ["ro", 0.5, 0.9, np.nan]
        return out

    def test_an_ro_cohort_bids_banding_times_buyout_on_its_live_share(self, network, labels, reference):
        offer = pd.Series(STATIC_OFFER).reindex(network.generators.index)
        bid = pd.Series(STATIC_BID).reindex(network.generators.index)
        frames = sb.build_label_bid_frames(network, _scenario(self.SCENARIO), {}, offer, bid, reference, LOG,
                                           labels=self._ro_labels(labels))
        # The 2030 chain's rule: banding x buyout x multiplier (1), on the live half; the rest merchant.
        np.testing.assert_allclose(frames.bid_tv["on"].to_numpy(), 0.5 * 0.9 * 78.04 + 0.5 * 20.0)
        assert (frames.offer_tv["on"] == STATIC_OFFER["on"]).all()
        audit = frames.audit.set_index("generator")
        assert audit.loc["on", "ro_weight"] == pytest.approx(0.5) and audit.loc["off", "ro_weight"] == 0.0

    def test_ro_labels_without_a_buyout_are_refused(self, network, labels, reference):
        offer = pd.Series(STATIC_OFFER).reindex(network.generators.index)
        bid = pd.Series(STATIC_BID).reindex(network.generators.index)
        with pytest.raises(ValueError, match="roc_buyout_prices"):
            sb.build_label_bid_frames(network, _scenario({"modelled_year": 2030}), {}, offer, bid, reference, LOG,
                                      labels=self._ro_labels(labels))

    def test_the_six_hour_rule_pays_through_a_short_negative_run(self, network, labels, reference):
        six = labels.copy()
        six.loc[six["component"].eq("off") & six["support_type"].eq("cfd"), "negative_price_rule"] = "six_hour"
        frames = _frames(network, six, reference)
        cfd = np.array([0.0, 85.0, 50.0, 20.0])          # the -5 hour is paid: a run of one hour
        np.testing.assert_allclose(frames.bid_tv["off"].to_numpy(), 0.6 * cfd + 0.4 * 20.0)


@pytest.mark.unit
class TestSupportSchemes:
    def _cohorts(self):
        return pd.DataFrame([
            _cohort("ON::b2010", "Generator", "wind_onshore", 10.0, "merchant", year=2010, technology="Onshore wind",
                    basis="contract_expired_merchant"),
            _cohort("OFF::b2017", "Generator", "wind_offshore", 10.0, "merchant", year=2017,
                    technology="Offshore wind", basis="contract_expired_merchant"),
            _cohort("OFF::b2018", "Generator", "wind_offshore", 10.0, "CfD", year=2018, technology="Offshore wind",
                    contract="AR1_OFFSHORE_2018_19#1", cfd_round="AR1", live=1.0),
            _cohort("PV::b2012", "Generator", "solar_pv", 10.0, "merchant", year=2012, technology="Solar PV"),
        ])

    def test_ro_live_share_follows_accreditation_term_and_scheme_end(self):
        schemes = sl.load_support_schemes()
        out = sl.label_support_schemes(self._cohorts(), schemes, model_year=2030).set_index("asset_id")
        # Accredited through 2010, 20 years: expires through 2030, so half of 2030.
        assert out.loc["ON::b2010", "support_type"] == "RO"
        assert out.loc["ON::b2010", "cfd_live_share"] == pytest.approx(0.5, abs=1e-3)
        assert out.loc["ON::b2010", "ro_banding"] == 0.9
        # Offshore closed to new plant on 31 March 2017: only the year's first quarter is accredited.
        assert out.loc["OFF::b2017", "cfd_live_share"] == pytest.approx(0.2466, abs=2e-3)
        assert out.loc["OFF::b2018", "support_type"] == "CfD"                 # a CfD cohort is left alone
        # FiT is declared but assigns nothing without a cohort share (small-scale plant is not separated),
        # so a 2012 solar cohort is RO (an upper bound until a FiT share is declared).
        assert out.loc["PV::b2012", "support_type"] == "RO"

    def test_the_ro_ends_on_31_march_2037(self):
        schemes = sl.load_support_schemes()
        y37 = sl.label_support_schemes(self._cohorts(), schemes, model_year=2037).set_index("asset_id")
        # Offshore accredited Jan-Mar 2017 expires Jan-Mar 2037: the mean of (t + 20 - 2037) over that quarter.
        assert y37.loc["OFF::b2017", "cfd_live_share"] == pytest.approx(0.2466 ** 2 / 2, abs=2e-3)
        for year in (2038, 2040, 2050):
            out = sl.label_support_schemes(self._cohorts(), schemes, model_year=year)
            assert out["ro_banding"].isna().all() and not out["support_type"].eq("RO").any(), year

    def test_ro_cohorts_reach_the_label_table(self):
        cohorts = sl.label_support_schemes(self._cohorts().iloc[[0, 3]], sl.load_support_schemes(), model_year=2030)
        out = sl.label_table(cohorts, strike_table=StrikeTable({}, 2024, {}, {}), known=pd.DataFrame(
            columns=["contract_id", "phase"]), dpa_reference_srmc=REF_SRMC, dpa_margin=MARGIN).set_index("cohort_id")
        assert out.loc["ON::b2010", "support_type"] == "ro" and out.loc["ON::b2010", "ro_banding"] == 0.9
        assert out.loc["ON::b2010", "live_share"] == pytest.approx(0.5, abs=1e-3)


@pytest.mark.unit
class TestDpaDispatchTable:
    def _cohorts(self):
        return pd.DataFrame([
            _cohort("ccs::b2040", "Generator", "CCS Gas", 300.0, "DPA", live=0.7, cfd_round="DPA_2040"),
            _cohort("ccs::b2030", "Generator", "CCS Gas", 200.0, "merchant", year=2030),
            _cohort("ccs::b2041", "Generator", "CCS Gas", 100.0, "DPA", year=2041, live=1.0, cfd_round="DPA_2041"),
            _cohort("h2t::b2040", "Link", "H2_turbine", 800.0, "DPA", live=1.0, cfd_round="DPA_2040"),
            _cohort("h2t::b2030", "Link", "H2_turbine", 200.0, "merchant", year=2030),
            _cohort("bio::b2030", "Generator", "biogas", 50.0, "merchant", year=2030),
        ])

    def test_the_rate_free_table_gives_stage_1_the_label_table_costs_exactly(self, network, tmp_path):
        cohorts = self._cohorts()
        labels = sl.label_table(cohorts, strike_table=StrikeTable({}, 2024, {}, {}),
                                known=pd.DataFrame(columns=["contract_id", "phase"]), dpa_reference_srmc=REF_SRMC,
                                dpa_margin=MARGIN)
        table = sl.dpa_dispatch_table(cohorts, dpa_reference_srmc=REF_SRMC, dpa_margin=MARGIN)
        assert list(table.columns) == list(sl.DPA_DISPATCH_COLUMNS)
        assert "strike_gbp_per_mwh_2024" not in table.columns
        labels.to_csv(tmp_path / "labels.csv", index=False)
        table.to_csv(tmp_path / "dpa.csv", index=False)
        costs = []
        for name in ("labels.csv", "dpa.csv"):
            n = network.copy()
            n.generators.at["ccs", "marginal_cost"] = 120.0
            block = {"enabled": True, "file": str(tmp_path / name), "generator_carriers": ["CCS Gas"],
                     "link_carriers": ["H2_turbine"], "hydrogen_value_gbp_per_mwh_h2": 36.75}
            sb.apply_dpa_dispatch(n, block, LOG)
            costs.append((n.generators["marginal_cost"].copy(), n.links["marginal_cost"].copy()))
        pd.testing.assert_series_equal(costs[0][0], costs[1][0], check_exact=True)
        pd.testing.assert_series_equal(costs[0][1], costs[1][1], check_exact=True)
        held = table.set_index("component")
        assert held.loc["ccs", "dpa_weight"] == pytest.approx((300 * 0.7 + 100) / 600)
        assert held.loc["bio", "dpa_cohorts"] == 0 and np.isnan(held.loc["bio", "dpa_margin_gbp_per_mwh"])

    def test_a_dpa_link_missing_from_the_table_is_refused(self, network, tmp_path):
        table = sl.dpa_dispatch_table(self._cohorts().iloc[:3], dpa_reference_srmc=REF_SRMC, dpa_margin=MARGIN)
        table.to_csv(tmp_path / "dpa.csv", index=False)
        with pytest.raises(ValueError, match="DPA-carrier links have no support label"):
            sb.apply_dpa_dispatch(network, {"enabled": True, "file": str(tmp_path / "dpa.csv"),
                                            "link_carriers": ["H2_turbine"], "hydrogen_value_gbp_per_mwh_h2": 36.75},
                                  LOG)


@pytest.mark.unit
def test_stage1_reads_the_rate_free_dpa_table():
    scenarios = _scenarios()
    for sid in ("FES25_HT_2050_Mar_market", "HT50_1week_gate_final", "FES25_HT_2050_Mar_p2_bids_75pct"):
        dpa = scenarios[sid]["market"]["wholesale"]["marginal_cost_basis"]["dpa_dispatch"]
        assert dpa["file"] == "resources/market/FES25_HT_2050_dpa_dispatch.csv", sid


@pytest.mark.unit
def test_an_unknown_vintage_source_is_refused_not_labelled():
    mix = pd.DataFrame({"technology": ["Hydropower", "Onshore wind"], "build_year": [2030, 2031],
                        "capacity_mw": [1.0, 2.0], "source": ["refurbishment", "gross_addition"]})
    with pytest.raises(ValueError, match="refurbishment"):
        sl.check_vintage_sources(mix)
    sl.check_vintage_sources(mix.iloc[[1]])
    with pytest.raises(ValueError, match="no source column"):
        sl.check_vintage_sources(mix.drop(columns=["source"]))
