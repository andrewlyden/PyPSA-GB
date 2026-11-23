"""The NESO-consistent CP30 2030 package (user decisions, 16 Oct 2026).

Pins, in the repository code:

* CfD bids at contract value without the zero floor (FTI for NESO 2025: "CfD strike price - Wholesale
  price"): the sign in the ESO-cost convention, every vintage, the part-CfD remainder rows (merchant), the
  P2/P3 state-bid paths, and the settlement/compensation readers following the audit;
* merchant renewables at a bid of 0;
* must-take rooftop solar in Stage 1 (p_min_pu = p_max_pu, never curtailed) and in the BM (held at Stage 1);
* the NESO-comparable cost field (ordinary redispatch + IC re-trading, slack and penalties apart);
* the churn sink outside the redispatch volumes, its held energy reported;
* the foreign-plant uplift on interconnector moves, offers and bids apart, on GB gas's multipliers
  (LTMNC eqs 6 and 1);
* the boundaries-only balancing transmission (network_delivery) and the secured boundary list;
* the CP30 gas price override (CP.22, 100.68 p/therm) and the Stage 1 check config equality.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml

from scripts.market.cfd_bm_bids import (
    CONTRACT_VALUE_FLOORS,
    DEFAULT_CONTRACT_VALUE_FLOOR,
    contract_value_rate,
    foregone_rate,
    load_scenario_strike_table,
    resolve_contract_value_floor,
)
from config.config_loader import load_defaults, load_layer
from tests.unit.test_cfd_bm_bids import (  # noqa: F401  (pytest fixtures)
    _frames,
    balancing_config,
    network,
    reference_price,
    scenario_config,
    strike_table_file,
)

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("test_neso_consistent_package")


def _defaults() -> dict:
    return load_defaults()


def _contract_value_bids(scenario_config):
    """The scenario's ``support.bm_bids.contract_value_bids`` block."""
    return scenario_config["support"]["bm_bids"]["contract_value_bids"]


def _unfloored(scenario_config):
    cfg = _contract_value_bids(scenario_config)
    cfg["contract_value_floor"] = "none"
    cfg["merchant"]["bid_price"] = 0.0
    return scenario_config


# ── CfD contract value ─────────────────────────────────────────────────────
@pytest.mark.unit
class TestContractValueFloor:

    def test_rate_helpers_and_sign(self):
        ref = pd.Series([0.0, 40.0, 120.0, -10.0])
        assert contract_value_rate(80.0, ref, "zero").tolist() == [80.0, 40.0, 0.0, 90.0]
        # Unfloored (FTI): strike - reference, negative when the reference exceeds the strike. In the
        # ESO-cost convention a negative bid is money NESO receives: the unit saves the difference payment
        # it would have paid back (ELEXON's own convention would show +40).
        assert contract_value_rate(80.0, ref, "none").tolist() == [80.0, 40.0, -40.0, 90.0]
        assert foregone_rate(80.0, ref, "strike", "none").tolist() == [80.0] * 4
        assert foregone_rate(80.0, ref, "contract_value", "none").tolist() == [80.0, 40.0, -40.0, 90.0]
        arr = contract_value_rate(np.array([[80.0, 50.0]]), np.array([[100.0], [10.0]]), "none")
        assert arr.tolist() == [[-20.0, -50.0], [70.0, 40.0]]

    def test_floor_values_and_default(self):
        assert CONTRACT_VALUE_FLOORS == ("zero", "none") and DEFAULT_CONTRACT_VALUE_FLOOR == "zero"
        assert resolve_contract_value_floor(None) == "zero"
        with pytest.raises(ValueError):
            resolve_contract_value_floor("negative")
        block = load_layer("source", "CP30")["support"]["bm_bids"]["contract_value_bids"]
        assert block["contract_value_floor"] == DEFAULT_CONTRACT_VALUE_FLOOR

    def test_unfloored_bid_goes_negative_above_the_strike_for_every_vintage(
            self, network, scenario_config, balancing_config, reference_price):
        frames = _frames(network, _unfloored(scenario_config), balancing_config, reference_price)
        table = load_scenario_strike_table(scenario_config)
        for name, carrier, cfd_round in (("ic_offshore", "wind_offshore", "Investment Contract"),
                                         ("ar4_offshore", "wind_offshore", "AR4"),
                                         ("ar6_solar", "solar_pv", "AR6"),
                                         ("future_solar", "solar_pv", "AR8_modelled")):
            strike, _, _ = table.lookup(carrier, cfd_round)
            expected = strike - reference_price
            pd.testing.assert_series_equal(frames.bid_tv[name], expected, check_names=False)
            # The 400 GBP/MWh hour clears above every strike: NESO receives the saved difference payment.
            assert frames.bid_tv[name].iloc[3] < 0.0
        audit = frames.audit.set_index("generator")
        cfd = audit[audit.pricing.eq("cfd_contract_value")]
        assert set(cfd.contract_value_floor) == {"none"} and set(cfd.bid_basis) == {"contract_value"}
        # Excluded carrier (HPC, nuclear) and ROC keep their static bids.
        assert frames.bid_tv["hpc"].tolist() == pytest.approx([150.0] * 4)
        assert frames.bid_tv["roc_offshore"].tolist() == pytest.approx([78.04] * 4)

    def test_floored_default_is_unchanged(self, network, scenario_config, balancing_config, reference_price):
        frames = _frames(network, scenario_config, balancing_config, reference_price)
        assert (frames.bid_tv[["ic_offshore", "ar4_offshore", "ar6_solar"]] >= 0.0).all().all()
        assert set(frames.audit.loc[frames.audit.pricing.eq("cfd_contract_value"), "contract_value_floor"]) == {"zero"}

    def test_merchant_and_part_cfd_remainder_bid_zero(self, network, scenario_config, balancing_config,
                                                      reference_price):
        # A part-CfD project's uncontracted remainder is its own merchant row (cfd_contract_part remainder).
        network.add("Generator", "seagreen_non_CfD", bus="bus", carrier="wind_offshore", p_nom=300.0,
                    marginal_cost=0.0, support_type="merchant", cfd_round="", cfd_contract_part="remainder")
        frames = _frames(network, _unfloored(scenario_config), balancing_config, reference_price)
        for name in ("merchant_offshore", "merchant_solar", "seagreen_non_CfD"):
            assert (frames.bid_tv[name] == 0.0).all()
        # Merchant (0) goes before a CfD unit forgoing a positive top-up, after one that pays NESO to stop.
        assert frames.bid_tv["merchant_offshore"].iloc[0] < frames.bid_tv["ar4_offshore"].iloc[0]
        assert frames.bid_tv["merchant_offshore"].iloc[3] > frames.bid_tv["ar4_offshore"].iloc[3]

    def test_state_owned_share_blends_the_unfloored_private_bid(self, network, scenario_config,
                                                                balancing_config, reference_price, tmp_path):
        shares = tmp_path / "shares.csv"
        table = pd.DataFrame({"case_id": ["p2", "p2", "p2"],
                              "generator": ["ar4_offshore", "merchant_solar", "roc_offshore"],
                              "state_share": [0.5, 1.0, 1.0]})
        table["state_capacity_mw"] = table["state_share"] * network.generators.p_nom.reindex(table["generator"]).values
        table.to_csv(shares, index=False)
        cfg = _contract_value_bids(_unfloored(scenario_config))
        cfg["public_ownership_bids"] = {"enabled": True, "case_id": "p2", "shares_file": str(shares),
                                        "public_bid_price": 0.0, "renewables_at_cost": True}
        frames = _frames(network, scenario_config, balancing_config, reference_price)
        strike, _, _ = load_scenario_strike_table(scenario_config).lookup("wind_offshore", "AR4")
        expected = 0.5 * (strike - reference_price) + 0.5 * 0.0
        pd.testing.assert_series_equal(frames.bid_tv["ar4_offshore"], expected, check_names=False)
        # P3's renewables_at_cost path: owned RO and merchant plant bid the public price.
        assert (frames.bid_tv["merchant_solar"] == 0.0).all() and (frames.bid_tv["roc_offshore"] == 0.0).all()

    def test_state_thermal_at_cost_is_untouched_by_the_floor(self):
        from scripts.market.cfd_bm_bids import apply_state_thermal_at_cost
        snaps = pd.date_range("2030-01-01", periods=2, freq="h")
        net = SimpleNamespace(generators=pd.DataFrame({"marginal_cost": [100.0]}, index=["ccgt"]))
        offer = pd.DataFrame(160.0, index=snaps, columns=["ccgt"])
        bid = pd.DataFrame(-90.0, index=snaps, columns=["ccgt"])
        moved = apply_state_thermal_at_cost(net, {"thermal_at_cost": True}, pd.Series({"ccgt": 1.0}), offer, bid,
                                            pd.Series({"ccgt": 160.0}), pd.Series({"ccgt": -90.0}),
                                            pd.Series({"ccgt": "ccgt"}), LOG)
        assert moved == 1 and (offer == 100.0).all().all() and (bid == -100.0).all().all()

    def test_audit_reader_and_2050_guard(self, tmp_path):
        from scripts.consumer_bill.extract_future_balancing_costs import contract_value_floor_from_audit
        path = tmp_path / "audit.csv"
        assert contract_value_floor_from_audit(path) == "zero"
        pd.DataFrame({"generator": ["a"], "pricing": ["cfd_contract_value"], "bid_basis": ["contract_value"],
                      "contract_value_floor": ["none"]}).to_csv(path, index=False)
        assert contract_value_floor_from_audit(path) == "none"
        pd.DataFrame({"generator": ["a"], "pricing": ["cfd_contract_value"]}).to_csv(path, index=False)
        assert contract_value_floor_from_audit(path) == "zero"      # an older audit: floored

    def test_settlement_imports_the_shared_rate(self):
        import scripts.public_ownership.settlement as settlement
        assert settlement.contract_value_rate is contract_value_rate


# ── Must-take rooftop solar ────────────────────────────────────────────────
def _solar_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-06-01 10:00", periods=3, freq="h"))
    n.add("Bus", "b")
    n.add("Load", "load", bus="b", p_set=100.0)
    n.add("Generator", "rooftop", bus="b", carrier="solar_pv", p_nom=80.0, marginal_cost=0.5,
          p_max_pu=pd.Series([0.5, 1.0, 0.25], index=n.snapshots))
    n.add("Generator", "farm", bus="b", carrier="solar_pv", p_nom=100.0, marginal_cost=-0.01,
          p_max_pu=pd.Series([1.0, 1.0, 1.0], index=n.snapshots))
    n.add("Generator", "gas", bus="b", carrier="CCGT", p_nom=200.0, marginal_cost=100.0)
    n.generators["data_source"] = ["CP30_small_scale_solar_gsp", "REPD_named_fleet", ""]
    return n


MUST_TAKE = {"must_take_generation": {"data_sources": ["CP30_small_scale_solar_gsp"]}}


@pytest.mark.unit
class TestMustTake:

    def test_code_default_matches_defaults_yaml_and_is_off(self):
        from scripts.market.must_take import DEFAULT_DATA_SOURCES, must_take_generators
        assert _defaults()["market"]["must_take_generation"] == {"data_sources": list(DEFAULT_DATA_SOURCES)}
        assert len(must_take_generators(_solar_network(), {})) == 0

    def test_unknown_label_and_keys_are_refused(self):
        from scripts.market.must_take import must_take_generators, resolve_must_take_config
        with pytest.raises(ValueError):
            must_take_generators(_solar_network(), {"must_take_generation": {"data_sources": ["typo"]}})
        with pytest.raises(ValueError):
            resolve_must_take_config({"must_take_generation": {"carriers": ["solar_pv"]}})

    def test_stage1_holds_rooftop_at_availability_and_curtails_the_farm(self):
        from scripts.market.must_take import apply_must_take_stage1
        n = _solar_network()
        held = apply_must_take_stage1(n, MUST_TAKE, LOG)
        assert list(held) == ["rooftop"]
        pd.testing.assert_series_equal(n.generators_t.p_min_pu["rooftop"], n.generators_t.p_max_pu["rooftop"])
        status, _ = n.optimize(solver_name="highs")
        assert status == "ok"
        p = n.generators_t.p
        # Rooftop (merchant +0.5, the first renewable the contract-aware order curtails) runs at availability;
        # the cheaper-to-keep farm gives way instead, and energy balances.
        assert p["rooftop"].tolist() == pytest.approx([40.0, 80.0, 20.0], abs=1e-6)
        assert p["farm"].tolist() == pytest.approx([60.0, 20.0, 80.0], abs=1e-6)
        assert p["gas"].abs().max() == pytest.approx(0.0, abs=1e-6)

    def test_stage1_without_must_take_curtails_rooftop_first(self):
        n = _solar_network()
        n.optimize(solver_name="highs")
        assert n.generators_t.p["rooftop"].sum() < 140.0 - 1.0

    def test_bm_takes_rooftop_out_and_holds_it_at_stage1(self):
        from scripts.market.must_take import apply_must_take_balancing, hold_at_stage1
        n = _solar_network()
        n._bm_eligible_generators = n.generators.index
        n._bm_participating_generators = n.generators.index
        n._bm_fixed_generators = pd.Index([])
        held = apply_must_take_balancing(n, MUST_TAKE, LOG)
        assert "rooftop" not in n._bm_participating_generators and "rooftop" not in n._bm_eligible_generators
        assert list(n._bm_fixed_generators) == ["rooftop"]
        stage1 = pd.DataFrame({"rooftop": [40.0 + 1e-7, 80.0, 20.0], "farm": [60.0, 20.0, 80.0]}, index=n.snapshots)
        clipped = hold_at_stage1(n, stage1, held, LOG)
        assert clipped["rooftop"].tolist() == pytest.approx([40.0, 80.0, 20.0])
        assert clipped["farm"].tolist() == stage1["farm"].tolist()

    def test_bm_fixed_set_is_used_by_the_anchor(self):
        """The balancing extra functionality fixes a generator in the fixed set and not a participant."""
        from scripts.market.must_take import apply_must_take_balancing
        from scripts.market.solve_balancing import _build_balancing_extra_functionality
        n = _solar_network()
        n._bm_participating_generators = n.generators.index
        n._bm_fixed_generators = pd.Index([])
        names = apply_must_take_balancing(n, MUST_TAKE, LOG)
        stage1 = pd.DataFrame({"rooftop": [40.0, 80.0, 20.0], "farm": [60.0, 20.0, 80.0], "gas": 0.0},
                              index=n.snapshots)
        n.generators_t.p_max_pu["rooftop"] = [0.5, 1.0, 0.25]
        n.loads_t.p_set = pd.DataFrame({"load": [130.0, 100.0, 100.0]}, index=n.snapshots)
        extra = _build_balancing_extra_functionality(
            wholesale_gen=stage1, wholesale_su=pd.DataFrame(index=n.snapshots),
            wholesale_links=pd.DataFrame(index=n.snapshots),
            gen_offer_prices=pd.Series({"rooftop": 0.0, "farm": 500.0, "gas": 160.0}),
            gen_bid_prices=pd.Series({"rooftop": 0.0, "farm": 0.0, "gas": -90.0}),
            su_offer_prices=pd.Series(dtype=float), su_bid_prices=pd.Series(dtype=float),
            fix_interconnectors=False, logger=LOG,
            participating_generators=n._bm_participating_generators, fixed_generators=n._bm_fixed_generators)
        status, _ = n.optimize(solver_name="highs", extra_functionality=extra)
        assert status == "ok" and list(names) == ["rooftop"]
        assert n.generators_t.p["rooftop"].tolist() == pytest.approx([40.0, 80.0, 20.0], abs=1e-6)


# ── NESO-comparable cost and the churn sink ────────────────────────────────
def _bm_network(sink=True):
    snaps = pd.date_range("2030-01-01", periods=2, freq="h")
    gens = pd.DataFrame({"carrier": ["CCGT", "load_shedding", "churn_sink"] if sink else ["CCGT", "load_shedding"],
                         "marginal_cost": [100.0, 6000.0, 0.0] if sink else [100.0, 6000.0],
                         "bus": "b"}, index=["g", "shed", "churn_sink b"] if sink else ["g", "shed"])
    p = pd.DataFrame({"g": [10.0, 10.0], "shed": [1.0, 0.0]}, index=snaps)
    if sink:
        p["churn_sink b"] = [-30.0, -20.0]
    loads = pd.DataFrame({"bus": ["b"], "carrier": ["AC"]}, index=["l"])
    return SimpleNamespace(
        snapshots=snaps, snapshot_weightings=pd.DataFrame({"generators": 1.0, "objective": 1.0}, index=snaps),
        generators=gens, generators_t=SimpleNamespace(p=p), loads=loads,
        loads_t=SimpleNamespace(p_set=pd.DataFrame({"l": [100.0, 100.0]}, index=snaps)),
        buses=pd.DataFrame({"carrier": ["AC"]}, index=["b"]), storage_units=pd.DataFrame(),
        storage_units_t=SimpleNamespace(p_dispatch=pd.DataFrame(), p_store=pd.DataFrame()),
        links=pd.DataFrame(columns=["bus0", "bus1", "carrier"]), links_t=SimpleNamespace(p0=pd.DataFrame()))


@pytest.mark.unit
class TestCostMeasure:

    def _accounting(self):
        from scripts.market.market_accounting import attach_interconnector_retrading
        accounting = {"demand_MWh": 1000.0, "modelled_redispatch_cost_GBP": 1_000_000.0,
                      "ordinary_redispatch_cost_excluding_feasibility_slack_GBP": 600_000.0,
                      "definitions": {}}
        report = pd.DataFrame(columns=["metric", "value", "unit", "scope"])
        report = attach_interconnector_retrading(accounting, report, 250_000.0)
        return accounting, report

    def test_neso_comparable_is_ordinary_plus_retrading(self):
        accounting, report = self._accounting()
        assert accounting["neso_comparable_redispatch_cost_GBP"] == pytest.approx(850_000.0)
        assert accounting["modelled_redispatch_cost_including_retrading_GBP"] == pytest.approx(1_250_000.0)
        assert "neso_comparable_redispatch_cost_GBP" in set(report.metric)
        assert "neso_comparable_redispatch_cost" in accounting["definitions"]

    def test_none_without_retrading(self):
        from scripts.market.market_accounting import attach_interconnector_retrading
        accounting = {"modelled_redispatch_cost_GBP": 1.0,
                      "ordinary_redispatch_cost_excluding_feasibility_slack_GBP": 1.0}
        attach_interconnector_retrading(accounting, pd.DataFrame(columns=["metric", "value", "unit", "scope"]), None)
        assert accounting["neso_comparable_redispatch_cost_GBP"] is None

    def test_aggregate_sums_it_and_reports_the_excluded_parts(self, tmp_path):
        import json
        from scripts.market.aggregate_neso_comparator import aggregate_monthly_comparator_summaries
        months = pd.date_range("2030-01-01", periods=12, freq="MS")
        paths, names = [], []
        for i, m in enumerate(months):
            hours = float(m.days_in_month * 24)
            acc = {"reported_period_hours": hours, "demand_MWh": 1e6, "accepted_offer_cost_GBP": 2e6,
                   "accepted_bid_cost_GBP": 1e6, "modelled_redispatch_cost_GBP": 3e6,
                   "ordinary_redispatch_cost_excluding_feasibility_slack_GBP": 2e6,
                   "feasibility_slack_energy_MWh": 100.0, "feasibility_slack_cost_GBP": 6e5,
                   "non_participant_turn_down_cost_at_penalty_GBP": 4e5, "gross_redispatch_MWh": 1e5,
                   "interconnector_retrading_cost_GBP": 5e5, "churn_sink_held_MWh": 1e3,
                   "is_short_run": True, "annualisation_allowed": False}
            name = f"S{i}"
            p = tmp_path / f"{name}.json"
            p.write_text(json.dumps({"scenario": name, "accounting": acc}), encoding="utf-8")
            paths.append(p)
            names.append(name)
        agg = aggregate_monthly_comparator_summaries(paths, names, tmp_path / "m.csv", tmp_path / "a.json")
        assert agg["neso_comparable_redispatch_cost_GBP"] == pytest.approx(12 * 2.5e6)
        assert agg["modelled_redispatch_cost_including_retrading_GBP"] == pytest.approx(12 * 3.5e6)
        assert agg["non_participant_turn_down_cost_at_penalty_GBP"] == pytest.approx(12 * 4e5)
        assert agg["feasibility_slack_cost_GBP"] == pytest.approx(12 * 6e5)
        assert agg["churn_sink_held_MWh"] == pytest.approx(12e3)

    def test_churn_sink_held_energy_is_reported_and_not_a_volume(self):
        from scripts.market.market_accounting import churn_sink_held_mwh
        from scripts.market.market_utils import align_redispatch_generators
        n = _bm_network()
        hours = n.snapshot_weightings["generators"]
        assert churn_sink_held_mwh(n, hours) == pytest.approx(50.0)
        assert churn_sink_held_mwh(_bm_network(sink=False), hours) == 0.0
        # The F22 alignment drops the sink from the redispatch tables.
        wholesale = n.generators_t.p.drop(columns=["churn_sink b"])
        ws, phys, sink = align_redispatch_generators(wholesale, n.generators_t.p, n.snapshots)
        assert "churn_sink b" not in phys.columns and sink == pytest.approx(50.0)


# ── Interconnector foreign-plant uplift ────────────────────────────────────
@pytest.mark.unit
class TestForeignUplift:
    """LTMNC 2017 section 12: eq. 6 (offer) and eq. 1 (bid), the foreign plant on GB gas's multipliers.

    Toy (test_interconnector_counter_trade): a 200 MW cable to France, ask 50.5, bid 49.5, losses 10%, so
    an export MWh is worth 0.9^2 x 49.5 = 40.095. ESO-cost convention: positive = NESO pays.
    """

    EXPORT = 0.9 ** 2 * 49.5    # the toy cable's export value per MWh (test_interconnector_counter_trade)
    ASK = 50.5
    GAS = {"CCGT": {"mode": "markup", "offer_markup": 0.6, "bid_discount": 0.1}}

    def _toy(self):
        from tests.unit.test_interconnector_counter_trade import LINK, _balancing, _network, _schedule, _solve
        return LINK, _balancing, _network, _schedule, _solve

    def _gb_gas(self, carrier_overrides=None, **ct):
        _, _balancing, _, _, _ = self._toy()
        bal = _balancing(premium_gbp_per_mwh=0.0, bid_premium_gbp_per_mwh=0.0, foreign_plant_multipliers="gb_gas",
                         **ct)
        bal["carrier_overrides"] = self.GAS if carrier_overrides is None else carrier_overrides
        return bal

    # Config ------------------------------------------------------------------
    def test_defaults_match_the_yaml_and_are_off(self):
        from scripts.market.interconnector_counter_trade import DEFAULT_COUNTER_TRADE, resolve_counter_trade_config
        block = _defaults()["market"]["balancing"]["interconnector_counter_trade"]
        for key in ("foreign_plant_multipliers", "foreign_offer_uplift_share", "foreign_bid_uplift_share",
                    "foreign_uplift_scale"):
            assert DEFAULT_COUNTER_TRADE[key] == block[key], key
        assert (block["foreign_plant_multipliers"], block["foreign_offer_uplift_share"],
                block["foreign_bid_uplift_share"], block["foreign_uplift_scale"]) == (None, 0.0, 0.0, 1.0)
        # The retired single share is not a default any more (it would clash with the split keys).
        assert "foreign_uplift_share" not in block and "foreign_uplift_share" not in DEFAULT_COUNTER_TRADE
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": dict(block)})
        assert (cfg["foreign_offer_uplift_share"], cfg["foreign_bid_uplift_share"]) == (0.0, 0.0)

    def test_bounds_and_conflicts_are_refused(self):
        from scripts.market.interconnector_counter_trade import resolve_counter_trade_config

        def resolve(carrier_overrides=None, **ct):
            bal = {"interconnector_counter_trade": ct}
            if carrier_overrides is not None:
                bal["carrier_overrides"] = carrier_overrides
            return resolve_counter_trade_config(bal)

        for bad in ({"foreign_offer_uplift_share": -0.1}, {"foreign_bid_uplift_share": 1.5},
                    {"foreign_uplift_share": 1.5}, {"foreign_offer_uplift_share": 0.6, "foreign_uplift_scale": -1.0},
                    # gb_gas at a scale of 2 puts the offer share at 1.2 > 1
                    {"foreign_plant_multipliers": "gb_gas", "foreign_uplift_scale": 2.0},
                    {"foreign_plant_multipliers": "dutch_gas"},
                    # two places holding a share: refused
                    {"foreign_plant_multipliers": "gb_gas", "foreign_offer_uplift_share": 0.6},
                    {"foreign_uplift_share": 0.2, "foreign_bid_uplift_share": 0.1},
                    {"foreign_uplift_share": 0.2, "foreign_plant_multipliers": "gb_gas"}):
            with pytest.raises(ValueError):
                resolve(self.GAS, **bad)
        with pytest.raises(ValueError):    # an absolute-priced CCGT has no multipliers to lend
            resolve({"CCGT": {"mode": "absolute", "offer_price": 120.0}}, foreign_plant_multipliers="gb_gas")
        # The defaults zeros beside the alias or gb_gas are not a conflict.
        cfg = resolve(self.GAS, foreign_plant_multipliers="gb_gas", foreign_offer_uplift_share=0.0,
                      foreign_bid_uplift_share=0.0)
        assert (cfg["foreign_offer_uplift_share"], cfg["foreign_bid_uplift_share"]) == (0.6, 0.1)

    def test_retired_single_share_sets_both_sides(self):
        from scripts.market.interconnector_counter_trade import resolve_counter_trade_config
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {
            "foreign_uplift_share": 0.25, "foreign_offer_uplift_share": 0.0, "foreign_bid_uplift_share": 0.0}})
        assert (cfg["foreign_offer_uplift_share"], cfg["foreign_bid_uplift_share"]) == (0.25, 0.25)
        assert "foreign_uplift_share" not in cfg
        # cal_nc1, the no-multiplier record, resolves to no uplift.
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {"foreign_uplift_share": 0.0}})
        assert (cfg["foreign_offer_uplift_share"], cfg["foreign_bid_uplift_share"]) == (0.0, 0.0)

    def test_gb_gas_reads_the_scenarios_ccgt_multipliers(self):
        from scripts.market.interconnector_counter_trade import resolve_counter_trade_config
        cfg = resolve_counter_trade_config(self._gb_gas())
        assert (cfg["foreign_offer_uplift_share"], cfg["foreign_bid_uplift_share"]) == (0.6, 0.1)
        assert cfg["foreign_uplift_source"].startswith("gb_gas")
        # It follows the override, not a copy of it.
        cfg = resolve_counter_trade_config(self._gb_gas({"CCGT": {"offer_markup": 0.29, "bid_discount": 0.05}}))
        assert (cfg["foreign_offer_uplift_share"], cfg["foreign_bid_uplift_share"]) == (0.29, 0.05)
        # A CCGT override without a key falls back to the balancing defaults, then the code defaults (0.10).
        bal = self._gb_gas({"CCGT": {"offer_markup": 0.6}})
        bal["default_bid_discount"] = 0.2
        assert resolve_counter_trade_config(bal)["foreign_bid_uplift_share"] == pytest.approx(0.2)
        assert resolve_counter_trade_config(self._gb_gas({}))["foreign_offer_uplift_share"] == pytest.approx(0.10)

    def test_gb_gas_on_the_cp30_balancing_block_takes_its_ccgt_override(self):
        from scripts.market.interconnector_counter_trade import resolve_counter_trade_config
        bal = dict(load_layer("source", "CP30")["market"]["balancing"])
        ccgt = bal["carrier_overrides"]["CCGT"]
        bal["interconnector_counter_trade"] = {"enabled": True, "foreign_plant_multipliers": "gb_gas"}
        cfg = resolve_counter_trade_config(bal)
        assert cfg["foreign_offer_uplift_share"] == pytest.approx(
            ccgt.get("offer_markup", bal.get("default_offer_markup", 0.10)))
        assert cfg["foreign_bid_uplift_share"] == pytest.approx(
            ccgt.get("bid_discount", bal.get("default_bid_discount", 0.10)))

    def test_scale_moves_both_shares_in_proportion(self):
        from scripts.market.interconnector_counter_trade import resolve_counter_trade_config
        for scale in (0.0, 0.5, 1.0, 1.5):
            cfg = resolve_counter_trade_config(self._gb_gas(foreign_uplift_scale=scale))
            assert cfg["foreign_offer_uplift_share"] == pytest.approx(0.6 * scale)
            assert cfg["foreign_bid_uplift_share"] == pytest.approx(0.1 * scale)
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {
            "foreign_offer_uplift_share": 0.4, "foreign_bid_uplift_share": 0.2, "foreign_uplift_scale": 0.5}})
        assert (cfg["foreign_offer_uplift_share"], cfg["foreign_bid_uplift_share"]) == (0.2, 0.1)

    # Signs and magnitudes ----------------------------------------------------
    def test_direction_premiums_signs_and_magnitudes(self):
        # Offer move: + s_o x m_up, the move costing (1 + s_o) x m_up. Bid move: + s_b x |m_down| on top of
        # m_down < 0, NESO receiving (1 - s_b) x |m_down|. Exporting and importing, both directions.
        from scripts.interconnectors.market_coupling import counter_trade_direction_premiums, marginal_trade_cost
        LINK, _, _network, _, _ = self._toy()
        n = _network(hours=2)
        sched = pd.DataFrame({LINK: [-200.0, 100.0]}, index=n.snapshots)   # hour 0 exports, hour 1 imports
        ask_t = n.links_t.marginal_cost
        zero = pd.Series(0.0, index=[LINK])
        up, down = counter_trade_direction_premiums(sched, n.links, [LINK], zero, ask_t, zero,
                                                    pd.Series(0.6, index=[LINK]), pd.Series(0.1, index=[LINK]))
        m_up, m_down = marginal_trade_cost(sched, n.links, ask_t)
        assert m_up[LINK].tolist() == pytest.approx([self.EXPORT, self.ASK])      # export given up; ask bought
        assert m_down[LINK].tolist() == pytest.approx([-self.EXPORT, -self.ASK])  # export earned; ask saved
        assert up[LINK].tolist() == pytest.approx([0.6 * self.EXPORT, 0.6 * self.ASK])
        assert down[LINK].tolist() == pytest.approx([0.1 * self.EXPORT, 0.1 * self.ASK])
        assert (up[LINK] >= 0).all() and (down[LINK] >= 0).all()   # both raise NESO's cost
        total_up = m_up[LINK] + up[LINK]
        total_down = m_down[LINK] + down[LINK]
        assert total_up.tolist() == pytest.approx([1.6 * self.EXPORT, 1.6 * self.ASK])
        assert total_down.tolist() == pytest.approx([-0.9 * self.EXPORT, -0.9 * self.ASK])
        # The sides are independent.
        up0, down0 = counter_trade_direction_premiums(sched, n.links, [LINK], zero, ask_t, zero,
                                                      pd.Series(0.6, index=[LINK]), pd.Series(0.0, index=[LINK]))
        assert down0[LINK].tolist() == [0.0, 0.0] and up0[LINK].tolist() == pytest.approx(up[LINK].tolist())

    def test_gb_gas_stamps_both_sides_and_values_offers_and_bids(self):
        from scripts.interconnectors.market_coupling import (COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN,
                                                             COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN,
                                                             COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN,
                                                             counter_trade_premium_cost)
        from scripts.market.solve_balancing import prepare_interconnector_counter_trade
        LINK, _, _network, _schedule, _ = self._toy()
        n = _network()
        _, links = _schedule(n)      # Stage 1 exports 200
        prepare_interconnector_counter_trade(n, self._gb_gas(), links, LOG)
        assert n.links.at[LINK, COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN] == pytest.approx(0.6)
        assert n.links.at[LINK, COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN] == pytest.approx(0.1)
        assert COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN not in n.links.columns
        # Offer: the export cut to zero pays 0.6 x the export value given up.
        offer = counter_trade_premium_cost(links * 0.0, links, n.links, n.links_t.marginal_cost)
        assert float(offer.to_numpy().sum()) == pytest.approx(200 * 0.6 * self.EXPORT, rel=1e-9)
        # Bid: an import of 100 cut to zero pays 0.1 x the ask saved.
        sched = pd.DataFrame({LINK: [100.0]}, index=n.snapshots)
        bid = counter_trade_premium_cost(sched * 0.0, sched, n.links, n.links_t.marginal_cost)
        assert float(bid.to_numpy().sum()) == pytest.approx(100 * 0.1 * self.ASK, rel=1e-9)

    def test_retrading_reports_carry_both_sides(self):
        from scripts.market.analyze_market import calculate_interconnector_retrading_metrics
        from scripts.market.fti_settlement import interconnector_retrading_cost
        from scripts.market.solve_balancing import prepare_interconnector_counter_trade
        LINK, _, _network, _, _ = self._toy()
        n = _network(hours=2)
        sched = pd.DataFrame({LINK: [-200.0, 100.0]}, index=n.snapshots)
        physical = pd.DataFrame({LINK: [0.0, 0.0]}, index=n.snapshots)   # export cut (offer), import cut (bid)
        prepare_interconnector_counter_trade(n, self._gb_gas(), sched, LOG)
        metrics = calculate_interconnector_retrading_metrics(n, sched, physical).set_index("interconnector")
        uplift = 200 * 0.6 * self.EXPORT + 100 * 0.1 * self.ASK
        trade = 200 * self.EXPORT - 100 * self.ASK      # export income lost, ask saved
        assert metrics.at["TestCable", "counter_trade_premium_GBP"] == pytest.approx(uplift, rel=1e-6)
        assert metrics.at["TestCable", "retrading_cost_GBP"] == pytest.approx(trade + uplift, rel=1e-6)
        # = 1.6 x 8,019 for the offer hour, -0.9 x 5,050 for the bid hour.
        assert trade + uplift == pytest.approx(1.6 * 200 * self.EXPORT - 0.9 * 100 * self.ASK)
        assert interconnector_retrading_cost(physical, sched, n.links, n.links_t.marginal_cost) == pytest.approx(
            metrics.at["TestCable", "retrading_cost_GBP"], rel=1e-9)

    def test_columns_survive_a_netcdf_round_trip(self, tmp_path):
        from scripts.interconnectors.market_coupling import (COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN,
                                                             COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN)
        LINK, _, _network, _, _solve = self._toy()
        n, _, _ = _solve(_network(), self._gb_gas(max_deviation_share=None, max_deviation_share_by_country={}))
        path = tmp_path / "bm.nc"
        n.export_to_netcdf(path)
        m = pypsa.Network(str(path))
        assert m.links.at[LINK, COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN] == pytest.approx(0.6)
        assert m.links.at[LINK, COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN] == pytest.approx(0.1)

    def test_retired_column_is_still_valued_on_both_sides(self):
        from scripts.interconnectors.market_coupling import (COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN,
                                                             COUNTER_TRADE_PREMIUM_COLUMN, counter_trade_premium_cost)
        LINK, _, _network, _, _ = self._toy()
        n = _network(hours=2)
        sched = pd.DataFrame({LINK: [-200.0, 100.0]}, index=n.snapshots)
        lk = n.links.copy()
        lk[COUNTER_TRADE_PREMIUM_COLUMN] = 0.0
        lk[COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN] = 0.0
        lk.loc[LINK, COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN] = 0.5
        cost = counter_trade_premium_cost(sched * 0.0, sched, lk, n.links_t.marginal_cost)
        assert float(cost.to_numpy().sum()) == pytest.approx(0.5 * (200 * self.EXPORT + 100 * self.ASK), rel=1e-9)

    def test_off_writes_no_column(self):
        from scripts.interconnectors.market_coupling import (COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN,
                                                             COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN,
                                                             COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN)
        from scripts.market.solve_balancing import prepare_interconnector_counter_trade
        _, _balancing, _network, _schedule, _ = self._toy()
        n = _network()
        _, links = _schedule(n)
        prepare_interconnector_counter_trade(n, _balancing(premium_gbp_per_mwh=0.0), links, LOG)
        for column in (COUNTER_TRADE_FOREIGN_UPLIFT_COLUMN, COUNTER_TRADE_FOREIGN_OFFER_UPLIFT_COLUMN,
                       COUNTER_TRADE_FOREIGN_BID_UPLIFT_COLUMN):
            assert column not in n.links.columns

    # The LP --------------------------------------------------------------------
    def test_offer_uplift_steers_the_lp_between_unwinding_and_gas(self):
        # Premium 0, caps off: unwinding costs (1 + s_o) x 40.1 per MWh against gas at 13,932 for the 200 MW.
        # s_o = 0.5: 12,030, the export is cut; s_o = 1.0: 16,038, gas runs instead. The bid share plays no part.
        LINK, _balancing, _network, _, _solve = self._toy()
        kw = dict(premium_gbp_per_mwh=0.0, bid_premium_gbp_per_mwh=0.0, max_deviation_share=None,
                  max_deviation_share_by_country={}, foreign_bid_uplift_share=1.0)
        n, _, _ = _solve(_network(), _balancing(foreign_offer_uplift_share=0.5, **kw))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        n, _, _ = _solve(_network(), _balancing(foreign_offer_uplift_share=1.0, **kw))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(-200.0, abs=1e-3)

    def test_gb_gas_at_scale_one_still_unwinds_the_toy_export(self):
        # (1 + 0.6) x 40.1 x 200 = 12,830 < 13,932 of gas: the export is cut; at scale 1.5 (offer 0.9:
        # 15,234) gas runs. The controller's lever moves the outcome through both shares.
        LINK, _, _network, _, _solve = self._toy()
        kw = dict(max_deviation_share=None, max_deviation_share_by_country={})
        n, _, _ = _solve(_network(), self._gb_gas(**kw))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(0.0, abs=1e-3)
        n, _, _ = _solve(_network(), self._gb_gas(foreign_uplift_scale=1.5, **kw))
        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(-200.0, abs=1e-3)

    def test_caps_off_lets_the_whole_cable_move(self):
        from scripts.market.interconnector_counter_trade import link_shares, resolve_counter_trade_config
        _, _, _network, _, _ = self._toy()
        n = _network()
        cfg = resolve_counter_trade_config({"interconnector_counter_trade": {
            "max_deviation_share": 2.0, "max_deviation_share_by_country": {"France": 2.0},
            "min_spread_rule": "none"}})
        assert link_shares(n, list(n.links.index[n.links.index.str.startswith("IC_")]), cfg).eq(2.0).all()


# ── Boundaries-only balancing transmission ─────────────────────────────────
@pytest.mark.unit
class TestBoundariesOnly:

    BLOCK = {"case": "outage_neso_boundaries_only",
             "transmission": {"min_bm_constraint_voltage_kv": 1000,
                              "bm_voltage_floor": {"scottish_transmission_constrained": False}},
             "secured_capability": {"boundaries": ["B1a", "B2", "B4", "B6", "B7a", "B8", "B9", "EC5", "LE1",
                                                   "SW1", "B13"]}}

    def test_overrides_reach_the_balancing_scenario_config_only(self):
        from scripts.market.network_delivery import apply_network_delivery
        n = pypsa.Network()
        n.add("Bus", "a"); n.add("Bus", "b")
        n.add("Line", "ab", bus0="a", bus1="b", x=0.1, s_nom=100.0)
        cfg = {"network": {"transmission": {"min_bm_constraint_voltage_kv": 220,
                                            "bm_voltage_floor": {"scottish_transmission_constrained": True,
                                                                 "scottish_min_constrained_kv": 132},
                                            "outage_schedule": {"secured_capability": {"capability_year": 2030}}}}}
        record = apply_network_delivery(n, cfg, {"network_delivery": self.BLOCK}, LOG)
        t = cfg["network"]["transmission"]
        assert t["min_bm_constraint_voltage_kv"] == 1000
        assert t["bm_voltage_floor"] == {"scottish_transmission_constrained": False, "scottish_min_constrained_kv": 132}
        assert t["outage_schedule"]["secured_capability"]["boundaries"] == self.BLOCK["secured_capability"]["boundaries"]
        assert t["outage_schedule"]["secured_capability"]["capability_year"] == 2030
        assert record["transmission"]["min_bm_constraint_voltage_kv"] == 1000
        assert "ab" in n.lines.index      # relaxed later by the voltage floor, never removed

    def test_every_neso_zone_boundary_has_a_capability_row(self):
        cap = pd.read_csv(ROOT / "data/network/etys_boundary_capability.csv")
        assert set(self.BLOCK["secured_capability"]["boundaries"]) <= set(cap["boundary"])

    def test_unknown_keys_are_refused(self):
        from scripts.market.network_delivery import network_delivery_config
        with pytest.raises(ValueError):
            network_delivery_config({"network_delivery": {"case": "x", "transmission": {"line_rating_overrides": {}}}})
        with pytest.raises(ValueError):
            network_delivery_config({"network_delivery": {"case": "x", "secured_capability": {"boundaries": []}}})


# ── Gas price and the Stage 1 check config ─────────────────────────────────
@pytest.mark.unit
class TestConfig:

    def test_cp30_gas_override_is_cp22(self):
        from scripts.generators.apply_marginal_costs import (DEFAULT_GAS_PRICE_OVERRIDE_PENCE_PER_THERM,
                                                             THERM_TO_MWH, _apply_gas_price_override)
        d = _defaults()
        assert d["marginal_costs"]["future_costs"]["gas_price_override_pence_per_therm"] \
            == DEFAULT_GAS_PRICE_OVERRIDE_PENCE_PER_THERM
        cp30 = load_layer("source", "CP30")["marginal_costs"]["future_costs"][
            "gas_price_override_pence_per_therm"]
        assert cp30 == 100.68
        selected = {"case": "neso_cp2030", "fuel_prices": {"gas": 34.121425}}
        _apply_gas_price_override(selected, {"gas_price_override_pence_per_therm": cp30})
        assert selected["fuel_prices"]["gas"] == pytest.approx(34.354, abs=5e-3)
        assert selected["fuel_prices"]["gas"] == pytest.approx(1.0068 / THERM_TO_MWH)
        unchanged = {"fuel_prices": {"gas": 30.0}}
        _apply_gas_price_override(unchanged, {})
        assert unchanged == {"fuel_prices": {"gas": 30.0}}

    def test_stage1_check_block_equals_the_code_defaults(self):
        from scripts.CP30.stage1_validation import DEFAULT_CONFIG, stage1_config
        block = load_layer("source", "CP30")["validation"]["stage1"]
        assert block == DEFAULT_CONFIG
        assert stage1_config(block) == DEFAULT_CONFIG
