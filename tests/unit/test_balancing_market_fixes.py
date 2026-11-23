"""Balancing-mechanism fixes of 27 Sep 2026 (market fix list D5, D6, C8, C9, C10, C12).

* D5: the hydrogen store level is carried between BM windows, its Stage 1 path
  is rebuilt from the link schedule and floors each window softly; turbines are
  held at Stage 1 (the held-turbine LP test is in test_hydrogen_link_balancing).
* D6: storage burn is measured on the gross legs with a p_nom-relative
  tolerance and reported as curtailed energy in the market accounting.
* C8: the general turn-up floor applies without contract-value bids.
* C9: the reported BM cost is net of the Stage 1 valuation of the cable schedule.
* C10: the low-rating warning lists only ratings still in force after the
  overrides, and radial stubs apart.
* The rolling loop carries the store and holds the turbines (D5) and nets the IC valuation (C9).
"""

import logging

import pandas as pd
import pypsa
import pytest

from scripts.market.market_accounting import (
    STORAGE_BURN_TOLERANCE_PU,
    create_market_accounting_report,
    storage_simultaneous_burn,
)
from scripts.market.solve_balancing import (
    DEFAULT_HELD_HYDROGEN_CARRIERS,
    DEFAULT_STORE_MONTH_END_TOLERANCE_PU,
    build_store_band_callback,
    ensure_price_frames,
    resolve_held_hydrogen_carriers,
    resolve_offer_regulariser,
    solve_rolling_balancing,
    stage1_store_levels,
    stage1_store_band,
    store_band_violation,
)
from config.config_loader import load_defaults, load_layer

LOGGER = logging.getLogger("test_balancing_market_fixes")


def _defaults():
    import yaml
    return load_defaults()


@pytest.mark.unit
class TestDefaults:

    def test_code_defaults_match_defaults_yaml(self):
        bal = _defaults()["market"]["balancing"]
        assert tuple(bal["hydrogen_links"]["hold_carriers"]) == DEFAULT_HELD_HYDROGEN_CARRIERS
        assert resolve_held_hydrogen_carriers({}) == DEFAULT_HELD_HYDROGEN_CARRIERS
        assert resolve_held_hydrogen_carriers({"hydrogen_links": {"hold_carriers": []}}) == ()
        assert bal["hydrogen_links"]["store_month_end_tolerance_pu"] == DEFAULT_STORE_MONTH_END_TOLERANCE_PU
        from scripts.market.cfd_bm_bids import DEFAULT_MERCHANT_BID
        cvb = load_layer("source", "CP30")["support"]["bm_bids"]["contract_value_bids"]
        # 27 Sep 2026 (user): merchant wind is paid ~20 to turn down in the BM (2024
        # evidence), no longer minus the Stage 1 merchant variable cost (-0.5).
        assert cvb["merchant"]["bid_price"] == DEFAULT_MERCHANT_BID == 20.0
        stage1 = _defaults()["market"]["wholesale"]["renewable_curtailment_order"]
        assert stage1["merchant_variable_cost_gbp_per_mwh"] == 0.5   # Stage 1 order unchanged


# ── D6 ───────────────────────────────────────────────────────────────────────

def _storage_network(dispatch, store, p_nom=100.0, eta=(0.9, 0.9)):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=len(dispatch), freq="h"))
    n.add("Bus", "b")
    n.add("StorageUnit", "bat", bus="b", p_nom=p_nom, max_hours=4, carrier="Battery",
          efficiency_store=eta[0], efficiency_dispatch=eta[1])
    n.storage_units_t.p_dispatch = pd.DataFrame({"bat": dispatch}, index=n.snapshots)
    n.storage_units_t.p_store = pd.DataFrame({"bat": store}, index=n.snapshots)
    return n


@pytest.mark.unit
class TestStorageBurn:

    def test_burn_is_the_overlap_times_the_round_trip_loss(self):
        n = _storage_network([10.0, 50.0, 0.0], [10.0, 20.0, 30.0])
        burn = storage_simultaneous_burn(n)
        loss = 1 / 0.9 - 0.9
        assert burn["burn_MWh"] == pytest.approx((10.0 + 20.0) * loss)
        assert burn["unit_hours"] == 2
        assert burn["burn_by_carrier_MWh"] == {"Battery": pytest.approx(30.0 * loss)}

    def test_barrier_noise_below_the_relative_tolerance_is_not_counted(self):
        tiny = 0.5 * STORAGE_BURN_TOLERANCE_PU * 100.0
        n = _storage_network([tiny, 5.0], [tiny, 5.0])
        burn = storage_simultaneous_burn(n)
        assert burn["unit_hours"] == 1
        assert burn["burn_MWh"] == pytest.approx(5.0 * (1 / 0.9 - 0.9))
        assert burn["noise_MWh"] == pytest.approx(tiny * (1 / 0.9 - 0.9))

    def test_accounting_reports_the_burn_as_curtailed_energy(self):
        n = _storage_network([10.0, 0.0], [10.0, 0.0])
        n.add("Load", "l", bus="b", p_set=1.0)
        redispatch = pd.DataFrame(columns=["carrier", "increase_MWh", "decrease_MWh", "offer_cost", "bid_cost"])
        accounting, report = create_market_accounting_report(n, redispatch)
        expected = 10.0 * (1 / 0.9 - 0.9)
        assert accounting["storage_burn_MWh"] == pytest.approx(expected)
        assert accounting["curtailed_energy_storage_burn_MWh"] == pytest.approx(expected)
        row = report.set_index("metric").loc["storage_burn_MWh"]
        assert row["scope"] == "curtailment" and row["value"] == pytest.approx(expected)
        assert any("curtailed energy" in w for w in accounting["warnings"])

    def test_no_gross_split_is_no_burn(self):
        n = _storage_network([1.0], [1.0])
        n.storage_units_t.p_store = pd.DataFrame(index=n.snapshots)
        assert storage_simultaneous_burn(n)["burn_MWh"] == 0.0


# ── C8 ───────────────────────────────────────────────────────────────────────

def _renewable_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "wind", bus="b", p_nom=10.0, carrier="wind_onshore")
    n.add("Generator", "gas", bus="b", p_nom=10.0, carrier="CCGT")
    return n


@pytest.mark.unit
class TestTurnUpFloorWithoutContractValueBids:

    def test_floor_applies_when_contract_value_bids_are_off(self):
        n = _renewable_network()
        cfg = {"contract_value_bids": {"enabled": False, "turn_up_floor_gbp_per_mwh": 2.0}}
        reg = resolve_offer_regulariser(n, cfg, LOGGER)
        assert reg.to_dict() == {"wind": 2.0}

    def test_with_contract_value_bids_the_folded_regulariser_is_kept(self):
        n = _renewable_network()
        n._bm_offer_regulariser = pd.Series({"wind": 2.0})
        cfg = {"contract_value_bids": {"enabled": True, "turn_up_floor_gbp_per_mwh": 2.0}}
        assert resolve_offer_regulariser(n, cfg, LOGGER) is n._bm_offer_regulariser

    def test_no_floor_configured_is_no_regulariser(self):
        assert resolve_offer_regulariser(_renewable_network(), {}, LOGGER) is None

    def test_frames_broadcast_the_static_prices_together(self):
        n = _renewable_network()
        offer, bid = ensure_price_frames(n, pd.Series({"wind": 0.0, "gas": 80.0}),
                                         pd.Series({"wind": 5.0, "gas": -60.0}), None, None)
        assert offer.shape == bid.shape == (2, 2)
        assert offer["gas"].tolist() == [80.0, 80.0] and bid["wind"].tolist() == [5.0, 5.0]
        kept = pd.DataFrame(1.0, index=n.snapshots, columns=n.generators.index)
        assert ensure_price_frames(n, offer.iloc[0], bid.iloc[0], kept, None)[0] is kept


# ── C9 ───────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBmCostNetOfStage1Valuation:

    def test_net_bm_cost_is_gb_redispatch_when_the_cable_keeps_its_schedule(self):
        from tests.unit.test_interconnector_counter_trade import (
            GAS_OFFER, LINK, WIND_BID, _balancing, _network, _solve,
        )
        n, links, _ = _solve(_network(), _balancing(premium_gbp_per_mwh=100.0, max_deviation_share=None,
                                                    max_deviation_share_by_country={}))
        from scripts.interconnectors.market_coupling import stage1_interconnector_trade_value

        assert n.links_t.p0.at[n.snapshots[0], LINK] == pytest.approx(-200.0, abs=1e-3)
        valuation = stage1_interconnector_trade_value(n)
        assert valuation < 0.0   # an export schedule earns
        # Whatever the export slope, the net cost is the GB bids and offers alone.
        assert n.objective - valuation == pytest.approx(200 * WIND_BID + 180 * GAS_OFFER, rel=1e-5)

    def test_no_callback_no_valuation(self):
        from scripts.interconnectors.market_coupling import stage1_interconnector_trade_value

        assert stage1_interconnector_trade_value(_renewable_network()) == 0.0


# ── D5 ───────────────────────────────────────────────────────────────────────

def _h2_network(hours=4):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=hours, freq="h"))
    n.add("Bus", "E")
    n.add("Bus", "GB_H2", carrier="H2_gas")
    n.add("Link", "ely", bus0="E", bus1="GB_H2", p_nom=100.0, efficiency=0.7, carrier="electrolysis")
    n.add("Link", "turb", bus0="GB_H2", bus1="E", p_nom=50.0, efficiency=0.5, carrier="H2_turbine")
    n.add("Store", "GB_H2_storage", bus="GB_H2", e_nom=1000.0, carrier="H2_gas", standing_loss=0.0)
    n.add("Load", "offtake", bus="GB_H2", p_set=10.0)
    return n


@pytest.mark.unit
class TestHydrogenStoreInTheBm:

    def test_stage1_path_is_rebuilt_from_the_link_schedule(self):
        n = _h2_network()
        links = pd.DataFrame({"ely": [100.0, 0.0, 50.0, 0.0], "turb": [0.0, 20.0, 0.0, 0.0]}, index=n.snapshots)
        path = stage1_store_levels(n, links, pd.Series({"GB_H2_storage": 500.0}))
        # +70-10, -20-10, +35-10, -10
        assert path["GB_H2_storage"].tolist() == pytest.approx([560.0, 530.0, 555.0, 545.0])

    def test_band_closes_on_the_stage1_month_end_level_and_is_reachable(self):
        n = _h2_network()
        links = pd.DataFrame({"ely": [100.0, 0.0, 50.0, 0.0], "turb": [0.0, 20.0, 0.0, 0.0]}, index=n.snapshots)
        lower, upper = stage1_store_band(n, links, pd.Series({"GB_H2_storage": 500.0}), 1e-3)
        path = [560.0, 530.0, 555.0, 545.0]
        # Floor: the Stage 1 path; last snapshot widened by 1e-3 x e_nom = 1 MWh.
        assert lower["GB_H2_storage"].tolist() == pytest.approx(path[:3] + [544.0])
        # Ceiling: path + what later snapshots can take back (0.7 x the electrolysis after t).
        assert upper["GB_H2_storage"].tolist() == pytest.approx([560.0 + 35.0, 530.0 + 35.0, 555.0, 546.0])

    def test_soft_band_is_used_only_when_needed_and_its_violation_is_measured(self):
        n = _h2_network(hours=2)
        n.add("Generator", "grid", bus="E", p_nom=1000.0, marginal_cost=10.0)
        n.stores.loc["GB_H2_storage", ["e_initial", "e_cyclic"]] = [100.0, False]
        n.links.loc["ely", "p_nom"] = 0.0   # nothing can refill the store
        lower, upper = pd.Series({"GB_H2_storage": 150.0}), pd.Series({"GB_H2_storage": 160.0})
        callback = build_store_band_callback(lower, upper, 200.0, LOGGER)
        status, _ = n.optimize(solver_name="highs", extra_functionality=callback)
        assert status == "ok"
        # 100 - 2 x 10 offtake = 80 at the end, 70 MWh short, paid at 200 in the objective only.
        assert store_band_violation(n, lower, upper) == pytest.approx((70.0, 0.0), abs=1e-6)
        assert n.objective == pytest.approx(70.0 * 200.0, rel=1e-6)

    def test_ceiling_side_binds_too(self):
        n = _h2_network(hours=2)
        n.add("Generator", "grid", bus="E", p_nom=1000.0, marginal_cost=-10.0)  # wants to run the electrolyser
        n.stores.loc["GB_H2_storage", ["e_initial", "e_cyclic"]] = [100.0, False]
        lower, upper = pd.Series({"GB_H2_storage": 80.0}), pd.Series({"GB_H2_storage": 90.0})
        callback = build_store_band_callback(lower, upper, 200.0, LOGGER)
        status, _ = n.optimize(solver_name="highs", extra_functionality=callback)
        assert status == "ok"
        assert n.stores_t.e["GB_H2_storage"].iloc[-1] == pytest.approx(90.0, abs=1e-4)

    def test_no_band_no_callback(self):
        assert build_store_band_callback(None, None, 200.0) is None
        assert build_store_band_callback(pd.Series({"s": 1.0}), pd.Series({"s": 2.0}), 0.0) is None


# ── The rolling loop (D5 carry and hold) ─────────────────────────────────────

def _rolling_network(hours=4):
    n = _h2_network(hours)
    n.add("Bus", "W")
    n.add("Line", "W_E", bus0="W", bus1="E", x=0.1, r=0.01, s_nom=60.0)
    n.add("Generator", "wind", bus="W", p_nom=200.0, carrier="wind_onshore",
          p_max_pu=pd.Series([1.0, 1.0, 0.5, 0.5], index=n.snapshots[:4])[:hours])
    n.add("Generator", "gas", bus="E", p_nom=200.0, carrier="CCGT", marginal_cost=80.0)
    n.add("Generator", "ls_dead", bus="W", p_nom=10.0, carrier="load_shedding", marginal_cost=6000.0, p_max_pu=0.0)
    n.add("Load", "demand", bus="E", p_set=50.0)
    n.stores.loc["GB_H2_storage", ["e_initial", "e_cyclic"]] = [500.0, False]
    return n


def _rolling(hours=4):
    n = _rolling_network(hours)
    snaps = n.snapshots
    gen = pd.DataFrame({"wind": [110.0, 110.0, 100.0, 100.0][:hours], "gas": 0.0, "ls_dead": 0.0}, index=snaps)
    links = pd.DataFrame({"ely": 100.0, "turb": 0.0}, index=snaps)
    offer = pd.Series({"wind": 0.0, "gas": 90.0, "ls_dead": 6000.0})
    bid = pd.Series({"wind": 20.0, "gas": -70.0, "ls_dead": 0.0})
    start = pd.Series({"GB_H2_storage": 500.0})
    band = stage1_store_band(n, links, start, 1e-3)
    from scripts.market.market_utils import hydrogen_link_prices
    names = ["ely"]
    inc, dec = hydrogen_link_prices(n, names, pd.Series(60.0, index=snaps), 0.1, 0.1, ("H2_gas",))
    result = solve_rolling_balancing(
        n, gen, pd.DataFrame(index=snaps), links, offer, bid, pd.Series(dtype=float), pd.Series(dtype=float),
        {"window_hours": 2, "fix_interconnectors": True}, "highs", {}, {}, LOGGER,
        anchored_links=names, link_increase_prices=inc, link_decrease_prices=dec,
        fixed_links=["turb"], store_start=start, store_band=band, store_penalty=200.0,
    )
    return n, result


@pytest.mark.unit
class TestRollingBm:

    def test_store_level_is_carried_and_turbines_held(self):
        n, (gen, _, _, _) = _rolling()
        e = n.stores_t.e["GB_H2_storage"]
        assert len(e) == 4
        # Window 2 starts where window 1 ended (no jump back to 500).
        links = n.links_t.p0
        assert links["turb"].abs().max() == pytest.approx(0.0, abs=1e-6)
        net = 0.7 * links["ely"] - links["turb"] - 10.0
        assert e.tolist() == pytest.approx((500.0 + net.cumsum()).tolist(), abs=1e-4)


# ── C10 ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLowRatingWarning:

    def _network(self, s132_rating):
        from scripts.network_build.branch_attributes import finalize_branch_attributes
        from tests.unit.test_branch_seasonal_and_licensee import _network
        n = _network()
        n.lines.loc["S132", "s_nom"] = s132_rating
        n.add("Load", "scot_load", bus="SCOT12", p_set=10.0)
        # A radial 132 kV stub to an empty bus, like CHAP1-_ECCF1K_0.
        n.add("Bus", "SCOTSTUB", v_nom=132, x=283000, y=702000)
        n.add("Line", "STUB", bus0="SCOT11", bus1="SCOTSTUB", x=1.0, s_nom=21.0, etys_licensee="SPT",
              voltage_kv=132)
        n.add("Generator", "ls_stub", bus="SCOTSTUB", p_nom=10.0, carrier="load_shedding")
        finalize_branch_attributes(n, LOGGER)
        return n

    @staticmethod
    def _cfg(overrides):
        return {"network": {"transmission": {"min_bm_constraint_voltage_kv": 220, "line_rating_overrides": overrides,
                                             "bm_voltage_floor": {"scottish_transmission_constrained": True,
                                                                  "scottish_min_constrained_kv": 132}}}}

    def test_warning_comes_after_the_overrides(self, caplog):
        from scripts.solve.solve_network import apply_line_rating_overrides
        n = self._network(18.0)
        with caplog.at_level(logging.INFO):
            apply_line_rating_overrides(n, self._cfg({"S132": 120}), LOGGER)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("S132" in w for w in warnings)          # corrected before the check
        infos = [r.getMessage() for r in caplog.records if "radial stub" in r.getMessage()]
        assert infos and "STUB" in infos[0]

    def test_a_live_low_rating_still_warns(self, caplog):
        from scripts.solve.solve_network import apply_line_rating_overrides
        n = self._network(18.0)
        with caplog.at_level(logging.INFO):
            apply_line_rating_overrides(n, self._cfg({}), LOGGER)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("S132" in w and "after overrides" in w for w in warnings)
        assert not any("STUB" in w for w in warnings)
