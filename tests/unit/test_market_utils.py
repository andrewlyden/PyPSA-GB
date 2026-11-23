"""
Unit tests for market simulation utilities.

Tests:
  - Bid/offer price calculation (derived from marginal cost)
  - Wholesale position extraction
  - Redispatch volume computation
  - Congestion identification
"""

import pytest
import pypsa
import pandas as pd
import numpy as np

from scripts.market.market_utils import (
    calculate_bid_offer_prices,
    extract_wholesale_positions,
    compute_redispatch_volumes,
    identify_congested_boundaries,
)
from config.config_loader import load_defaults


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def market_network():
    """
    Create a small network for testing market simulation utilities.

    3 buses, 1 line, 1 transformer, generators of different carriers,
    1 storage unit, 1 interconnector link.
    """
    n = pypsa.Network()
    snapshots = pd.date_range("2035-01-01", periods=6, freq="h")
    n.set_snapshots(snapshots)

    # Buses
    n.add("Bus", "bus_A", v_nom=400, x=0, y=0)
    n.add("Bus", "bus_B", v_nom=400, x=100000, y=0)
    n.add("Bus", "bus_ext", v_nom=400, x=200000, y=0)  # external

    # Line (constrained)
    n.add("Line", "line_AB", bus0="bus_A", bus1="bus_B",
          r=0.01, x=0.1, s_nom=500, length=100)

    # Generators
    n.add("Generator", "wind_1", bus="bus_A", p_nom=600,
          carrier="wind_onshore", marginal_cost=0.5)
    n.add("Generator", "ccgt_1", bus="bus_B", p_nom=400,
          carrier="CCGT", marginal_cost=50.0)
    n.add("Generator", "nuclear_1", bus="bus_B", p_nom=300,
          carrier="nuclear", marginal_cost=8.0)
    n.add("Generator", "load_shed", bus="bus_B", p_nom=1000,
          carrier="load_shedding", marginal_cost=6000.0)

    # Storage
    n.add("StorageUnit", "battery_1", bus="bus_A", p_nom=100,
          carrier="battery", marginal_cost=0.1,
          max_hours=4, efficiency_store=0.9, efficiency_dispatch=0.9,
          cyclic_state_of_charge=True)

    # Interconnector link
    n.add("Link", "IC_France", bus0="bus_B", bus1="bus_ext",
          p_nom=1000, p_min_pu=-1.0, carrier="DC", marginal_cost=0.0)

    # Load
    n.add("Load", "load_B", bus="bus_B", p_set=500)

    return n


@pytest.fixture
def default_market_config():
    """Default market config dict matching the config/defaults/ market structure."""
    return {
        "enabled": True,
        "wholesale": {"transmission_relaxation": 1e6},
        "balancing": {
            "bid_offer_source": "derived",
            "default_offer_markup": 0.10,
            "default_bid_discount": 0.10,
            "carrier_overrides": {
                "nuclear": {"offer_markup": 0.50, "bid_discount": 0.05},
                "wind_onshore": {"offer_markup": 0.0, "bid_discount": 0.05},
                "battery": {"offer_markup": 0.15, "bid_discount": 0.15},
            },
            "fix_interconnectors": True,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# BID / OFFER PRICE TESTS
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCalculateBidOfferPrices:
    """Tests for calculate_bid_offer_prices()."""

    def test_basic_prices(self, market_network, default_market_config, caplog):
        """Offer and bid prices are derived correctly from marginal cost."""
        import logging

        logger = logging.getLogger("test_market")
        gen_offer, gen_bid, su_offer, su_bid = calculate_bid_offer_prices(
            market_network, default_market_config, logger
        )

        # CCGT: marginal_cost=50, default markup=10%, discount=10%. The bid is
        # ESO-cost: turning the unit down saves 50, NESO receives 45 (the raw
        # ELEXON bid of 0.9 x mc, negated as the ELEXON path does).
        assert gen_offer["ccgt_1"] == pytest.approx(55.0)
        assert gen_bid["ccgt_1"] == pytest.approx(-45.0)

    def test_legacy_bid_convention_charges_neso_for_turn_down(
        self, market_network, default_market_config
    ):
        """derived_bid_convention legacy_cost restores the pre-26 Sep 2026 +mc x 0.9 bid."""
        import logging

        cfg = {
            **default_market_config,
            "balancing": {**default_market_config["balancing"], "derived_bid_convention": "legacy_cost"},
        }
        _, gen_bid, _, su_bid = calculate_bid_offer_prices(
            market_network, cfg, logging.getLogger("test_market")
        )
        assert gen_bid["ccgt_1"] == pytest.approx(45.0)
        assert su_bid["battery_1"] == pytest.approx(0.1 * 0.85)

    def test_unknown_bid_convention_is_refused(self, market_network, default_market_config):
        import logging

        cfg = {
            **default_market_config,
            "balancing": {**default_market_config["balancing"], "derived_bid_convention": "maybe"},
        }
        with pytest.raises(ValueError, match="derived_bid_convention"):
            calculate_bid_offer_prices(market_network, cfg, logging.getLogger("test_market"))

    def test_bid_sign_leaves_absolute_bids_alone_and_negates_round_trip_storage(
        self, market_network, default_market_config
    ):
        """Absolute bids are configured in ESO-cost convention and kept; a round_trip
        storage bid_price is what the unit pays (raw ELEXON sign), so it is negated."""
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "carrier_overrides": {
                    "nuclear": {"mode": "absolute", "offer_price": 999.0, "bid_price": 150.0},
                    "battery": {"mode": "round_trip", "bid_price": 50.0},
                },
            },
        }
        _, gen_bid, _, su_bid = calculate_bid_offer_prices(
            market_network, cfg, logging.getLogger("test_market")
        )
        assert gen_bid["nuclear_1"] == pytest.approx(150.0)
        assert su_bid["battery_1"] == pytest.approx(-50.0)

    def test_negative_marginal_cost_bid_pays_the_unit_its_loss(
        self, market_network, default_market_config
    ):
        """A markup-mode unit with mc -50 (a ROC unit) loses 50 when turned down: NESO pays 55."""
        import logging

        market_network.generators.loc["ccgt_1", "marginal_cost"] = -50.0
        _, gen_bid, _, _ = calculate_bid_offer_prices(
            market_network, default_market_config, logging.getLogger("test_market")
        )
        assert gen_bid["ccgt_1"] == pytest.approx(55.0)

    def test_carrier_overrides(self, market_network, default_market_config):
        """Per-carrier overrides are applied correctly."""
        import logging

        logger = logging.getLogger("test_market")
        gen_offer, gen_bid, _, _ = calculate_bid_offer_prices(
            market_network, default_market_config, logger
        )

        # Nuclear: offer_markup=0.50, bid_discount=0.05
        # marginal_cost=8.0
        assert gen_offer["nuclear_1"] == pytest.approx(12.0)  # 8 * 1.50
        assert gen_bid["nuclear_1"] == pytest.approx(-7.6)  # NESO receives 8 * 0.95

    def test_wind_offer_zero_markup(self, market_network, default_market_config):
        """Wind generators with offer_markup=0.0 have offer_price == mc."""
        import logging

        logger = logging.getLogger("test_market")
        gen_offer, _, _, _ = calculate_bid_offer_prices(
            market_network, default_market_config, logger
        )

        # Wind: mc=0.5, offer_markup=0.0 → offer=0.5
        assert gen_offer["wind_1"] == pytest.approx(0.5)

    def test_bid_price_is_not_floored(self, market_network, default_market_config):
        """Near-zero marginal-cost assets keep their derived bid without a floor."""
        import logging

        logger = logging.getLogger("test_market")
        _, gen_bid, _, _ = calculate_bid_offer_prices(
            market_network, default_market_config, logger
        )

        # Wind: mc=0.5, bid_discount=0.05 → NESO receives 0.5*0.95=0.475
        assert gen_bid["wind_1"] == pytest.approx(-0.475)

    def test_storage_prices(self, market_network, default_market_config):
        """Storage units receive bid/offer prices."""
        import logging

        logger = logging.getLogger("test_market")
        _, _, su_offer, su_bid = calculate_bid_offer_prices(
            market_network, default_market_config, logger
        )

        # battery: mc=0.1, offer_markup=0.15, bid_discount=0.15
        assert len(su_offer) == 1
        assert su_offer["battery_1"] == pytest.approx(0.1 * 1.15)

    def test_round_trip_mode_derives_offer_from_bid(
        self, market_network, default_market_config
    ):
        """offer = bid_price / (eta_store * eta_dispatch) + offer_premium; bid = -bid_price."""
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "storage_churn_guard_gbp_per_mwh": 0.0,
                "carrier_overrides": {
                    "battery": {"mode": "round_trip", "bid_price": 50.0},
                },
            },
        }
        _, _, su_offer, su_bid = calculate_bid_offer_prices(
            market_network, cfg, logging.getLogger("test_market")
        )
        # battery_1: efficiency 0.9 per leg -> round trip 0.81
        assert su_bid["battery_1"] == pytest.approx(-50.0)
        assert su_offer["battery_1"] == pytest.approx(50.0 / 0.81)

    def test_round_trip_premium_is_additive(self, market_network, default_market_config):
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "storage_churn_guard_gbp_per_mwh": 0.0,
                "carrier_overrides": {
                    "battery": {
                        "mode": "round_trip", "bid_price": 50.0, "offer_premium": 10.0,
                    },
                },
            },
        }
        _, _, su_offer, _ = calculate_bid_offer_prices(
            market_network, cfg, logging.getLogger("test_market")
        )
        assert su_offer["battery_1"] == pytest.approx(50.0 / 0.81 + 10.0)

    def test_churn_guard_raises_a_short_premium(self, market_network, default_market_config, caplog):
        """A pair that leaves SoC unchanged costs (offer x eta_rt + bid) per MWh charged and
        destroys (1 - eta_rt); the offer is lifted until that is >= the guard per MWh destroyed."""
        import logging

        guard = 250.0
        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "storage_churn_guard_gbp_per_mwh": guard,
                "carrier_overrides": {
                    "battery": {"mode": "round_trip", "bid_price": 40.0, "offer_premium": 10.0},
                },
            },
        }
        logger = logging.getLogger("test_market_guard")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            _, _, su_offer, su_bid = calculate_bid_offer_prices(market_network, cfg, logger)
        eta = 0.81
        assert su_offer["battery_1"] == pytest.approx((guard * (1 - eta) + 40.0) / eta)
        churn_cost = (su_offer["battery_1"] * eta + su_bid["battery_1"]) / (1 - eta)
        assert churn_cost == pytest.approx(guard)
        assert any("offer_premium too small" in r.getMessage() for r in caplog.records)

    def test_a_sufficient_premium_is_kept(self, market_network, default_market_config):
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "storage_churn_guard_gbp_per_mwh": 250.0,
                "carrier_overrides": {
                    "battery": {"mode": "round_trip", "bid_price": 40.0, "offer_premium": 70.0},
                },
            },
        }
        _, _, su_offer, _ = calculate_bid_offer_prices(
            market_network, cfg, logging.getLogger("test_market")
        )
        assert su_offer["battery_1"] == pytest.approx(40.0 / 0.81 + 70.0)

    def test_cross_unit_arbitrage_is_refused(self, market_network, default_market_config):
        """Condition (b): no storage offer may sit below what another storage unit pays to
        charge, or the BM discharges one unit into the other for profit."""
        import logging

        market_network.add(
            "StorageUnit", "dear_charger", bus="bus_B", p_nom=50, carrier="Domestic Battery",
            marginal_cost=0.1, max_hours=2, efficiency_store=0.95, efficiency_dispatch=0.95,
        )
        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "storage_churn_guard_gbp_per_mwh": 0.0,
                "carrier_overrides": {
                    "battery": {"mode": "round_trip", "bid_price": 10.0},     # offer 12.3
                    "Domestic Battery": {"mode": "round_trip", "bid_price": 40.0},  # pays 40
                },
            },
        }
        with pytest.raises(ValueError, match="cross-unit arbitrage"):
            calculate_bid_offer_prices(market_network, cfg, logging.getLogger("test_market"))
        # a premium that lifts the cheap offer to the dear charger's bid clears it
        cfg["balancing"]["carrier_overrides"]["battery"]["offer_premium"] = 40.0 - 10.0 / 0.81
        calculate_bid_offer_prices(market_network, cfg, logging.getLogger("test_market"))

    def test_storage_bids_follow_the_stage1_price(self, market_network, default_market_config):
        """storage_bid_reference stage1_price: paid = price - bid_margin per snapshot, so at a
        zero Stage 1 price NESO pays the battery to charge (no receipt to farm against
        curtailed wind) and at 100 the battery pays 56; the churn cost per MWh destroyed
        is the same in every snapshot."""
        import logging

        price = pd.Series([0.0, 0.0, 50.0, 100.0, 100.0, 0.0], index=market_network.snapshots)
        market_network._bm_reference_price = price
        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "storage_bid_reference": "stage1_price",
                "storage_churn_guard_gbp_per_mwh": 250.0,
                "carrier_overrides": {
                    "battery": {"mode": "round_trip", "bid_margin": 44.0, "bid_price": 40.0,
                                "offer_premium": 67.0},
                },
            },
        }
        calculate_bid_offer_prices(
            market_network, cfg, logging.getLogger("test_market"), time_varying=True
        )
        bid = market_network._bm_su_bid_tv["battery_1"]
        offer = market_network._bm_su_offer_tv["battery_1"]
        eta = 0.81
        assert bid.tolist() == pytest.approx((44.0 - price).tolist())
        assert offer.tolist() == pytest.approx(((price - 44.0) / eta + 67.0).tolist())
        churn = (offer * eta + bid) / (1 - eta)
        assert churn.tolist() == pytest.approx([67.0 * eta / (1 - eta)] * len(price))

    def test_fixed_reference_keeps_static_storage_prices(self, market_network, default_market_config):
        import logging

        market_network._bm_reference_price = pd.Series(0.0, index=market_network.snapshots)
        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "storage_bid_reference": "fixed",
                "carrier_overrides": {
                    "battery": {"mode": "round_trip", "bid_margin": 44.0, "bid_price": 40.0,
                                "offer_premium": 67.0},
                },
            },
        }
        _, _, _, su_bid = calculate_bid_offer_prices(
            market_network, cfg, logging.getLogger("test_market"), time_varying=True
        )
        assert getattr(market_network, "_bm_su_bid_tv", None) is None
        assert su_bid["battery_1"] == pytest.approx(-40.0)

    def test_configured_indexed_prices_hold_the_cross_unit_condition(self):
        """At every Stage 1 price the 2030 runs clear (0-215, sampled to 300) the cheapest
        configured storage offer is at least the most any storage unit pays to charge."""
        from pathlib import Path

        import numpy as np
        import yaml

        root = Path(__file__).resolve().parents[2]
        defaults = load_defaults()
        overrides = defaults["market"]["balancing"]["carrier_overrides"]
        trips = dict(defaults["storage"]["round_trip_efficiencies"])
        trips.update({"LAES": trips["LDES"], "CAES": trips["LDES"], "battery": trips["Battery"],
                      "pumped_hydro": trips["Pumped Storage Hydroelectricity"]})
        for lam in np.linspace(0.0, 300.0, 61):
            offers, paid = [], []
            for carrier, spec in overrides.items():
                if spec.get("mode") != "round_trip":
                    continue
                pay = lam - spec["bid_margin"]
                paid.append(pay)
                offers.append(pay / trips[carrier] + spec["offer_premium"])
            assert min(offers) >= max(paid), lam

    def test_configured_storage_prices_pass_both_guards(self):
        """config/defaults/base/market/balancing.yaml storage prices, on the configured round trips, clear the
        churn guard unaided and satisfy the cross-unit condition."""
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[2]
        defaults = load_defaults()
        balancing = defaults["market"]["balancing"]
        guard = float(balancing["storage_churn_guard_gbp_per_mwh"])
        overrides = balancing["carrier_overrides"]
        trips = dict(defaults["storage"]["round_trip_efficiencies"])
        trips["LAES"] = trips["LDES"]          # the CP30 LDES fleet is built as carrier LAES
        trips["CAES"] = trips["LDES"]
        trips["battery"] = trips["Battery"]
        trips["pumped_hydro"] = trips["Pumped Storage Hydroelectricity"]
        offers, paid = {}, {}
        for carrier, spec in overrides.items():
            if spec.get("mode") != "round_trip":
                continue
            eta = trips[carrier]
            offers[carrier] = spec["bid_price"] / eta + spec.get("offer_premium", 0.0)
            paid[carrier] = spec["bid_price"]
            churn = (offers[carrier] * eta - spec["bid_price"]) / (1 - eta)
            assert churn >= guard, carrier
        assert min(offers.values()) >= max(paid.values())

    def test_round_trip_without_bid_price_is_refused(
        self, market_network, default_market_config
    ):
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "carrier_overrides": {"battery": {"mode": "round_trip"}},
            },
        }
        with pytest.raises(ValueError, match="bid_price"):
            calculate_bid_offer_prices(
                market_network, cfg, logging.getLogger("test_market")
            )

    def test_inconsistent_absolute_storage_prices_are_warned(
        self, market_network, default_market_config, caplog
    ):
        """An offer below bid / round trip lets the BM charge for less than the
        energy the round trip loses; legacy absolute config still works but says so."""
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "carrier_overrides": {
                    "battery": {"mode": "absolute", "offer_price": 50.0, "bid_price": 60.0},
                },
            },
        }
        logger = logging.getLogger("test_market_inconsistent")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            _, _, su_offer, su_bid = calculate_bid_offer_prices(market_network, cfg, logger)
        assert su_offer["battery_1"] == pytest.approx(50.0)
        assert su_bid["battery_1"] == pytest.approx(60.0)
        assert any("round-trip-consistent" in r.getMessage() for r in caplog.records)

    def test_csv_source_without_files_raises(self, market_network):
        """bid_offer_source='csv' without file paths raises ValueError."""
        import logging

        logger = logging.getLogger("test_market")
        bad_config = {
            "balancing": {"bid_offer_source": "csv"},
        }
        with pytest.raises(ValueError, match="csv"):
            calculate_bid_offer_prices(market_network, bad_config, logger)

    def test_generator_participation_priced_out(self, market_network, default_market_config):
        """Non-participants can be priced out instead of acting like normal BM units."""
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "participation": {
                    "generators": {
                        "mode": "all",
                        "exclude_carriers": ["wind_onshore"],
                        "behavior": "priced_out",
                        "penalty_offer_price": 5000.0,
                        "penalty_bid_price": 4000.0,
                    }
                },
            },
        }
        logger = logging.getLogger("test_market")
        gen_offer, gen_bid, _, _ = calculate_bid_offer_prices(
            market_network, cfg, logger
        )

        assert gen_offer["wind_1"] == pytest.approx(5000.0)
        assert gen_bid["wind_1"] == pytest.approx(4000.0)
        assert gen_offer["ccgt_1"] == pytest.approx(55.0)
        assert "wind_1" in market_network._bm_participating_generators
        assert "wind_1" not in market_network._bm_eligible_generators
        assert len(market_network._bm_fixed_generators) == 0

    def test_generator_participation_fixed(self, market_network, default_market_config):
        """Strict non-participants are tagged for fixing at wholesale in BM solve."""
        import logging

        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "participation": {
                    "generators": {
                        "mode": "all",
                        "exclude_carriers": ["wind_onshore"],
                        "behavior": "fixed",
                    }
                },
            },
        }
        logger = logging.getLogger("test_market")
        gen_offer, gen_bid, _, _ = calculate_bid_offer_prices(
            market_network, cfg, logger
        )

        assert gen_offer["wind_1"] == pytest.approx(0.5)
        assert gen_bid["wind_1"] == pytest.approx(-0.475)
        assert "wind_1" in market_network._bm_fixed_generators
        assert "wind_1" not in market_network._bm_participating_generators

    def test_generator_participation_elexon_mapped_mode(
        self, market_network, default_market_config
    ):
        """elexon_mapped mode can restrict BM access to directly matched units."""
        import logging

        market_network._bm_direct_matched_generators = pd.Index(["ccgt_1"])
        cfg = {
            **default_market_config,
            "balancing": {
                **default_market_config["balancing"],
                "participation": {
                    "generators": {
                        "mode": "elexon_mapped",
                        "behavior": "fixed",
                    }
                },
            },
        }
        logger = logging.getLogger("test_market")
        calculate_bid_offer_prices(market_network, cfg, logger)

        # Only the directly matched unit is eligible on its own merits.
        assert list(market_network._bm_eligible_generators) == ["ccgt_1", "load_shed"]
        assert list(market_network._bm_participating_generators) == ["ccgt_1", "load_shed"]
        assert "wind_1" in market_network._bm_fixed_generators
        assert "nuclear_1" in market_network._bm_fixed_generators
        # Load shedding is carved out of `behavior: fixed` on purpose: it is the
        # slack that keeps the balancing LP feasible, so fixing it at its
        # wholesale position would make an infeasible hour unsolvable rather than
        # expensive. It stays available and priced out, not eligible in the
        # Elexon sense. This assertion is what the test originally missed.
        assert "load_shed" not in market_network._bm_fixed_generators


# ══════════════════════════════════════════════════════════════════════════════
# WHOLESALE POSITION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestExtractWholesalePositions:
    """Tests for extract_wholesale_positions()."""

    def test_extraction_shapes(self, market_network):
        """Dispatch DataFrames have correct shapes after a trivial solve-like setup."""
        import logging

        logger = logging.getLogger("test_market")

        # Manually set generators_t.p to simulate a solved network
        market_network.generators_t.p = pd.DataFrame(
            np.random.rand(6, 4) * 100,
            index=market_network.snapshots,
            columns=market_network.generators.index,
        )
        market_network.storage_units_t.p = pd.DataFrame(
            np.random.rand(6, 1) * 50,
            index=market_network.snapshots,
            columns=market_network.storage_units.index,
        )

        gen_d, su_d, link_d = extract_wholesale_positions(market_network, logger)

        assert gen_d.shape == (6, 4)
        assert su_d.shape == (6, 1)


# ══════════════════════════════════════════════════════════════════════════════
# REDISPATCH VOLUME COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestComputeRedispatchVolumes:
    """Tests for compute_redispatch_volumes()."""

    def test_zero_redispatch(self, market_network):
        """No redispatch if wholesale == physical."""
        import logging

        logger = logging.getLogger("test_market")

        snapshots = market_network.snapshots
        gen_names = ["wind_1", "ccgt_1"]
        dispatch_ws = pd.DataFrame(
            {"wind_1": [100] * 6, "ccgt_1": [200] * 6}, index=snapshots
        )
        dispatch_phys = dispatch_ws.copy()

        offer = pd.Series({"wind_1": 0.5, "ccgt_1": 55.0})
        bid = pd.Series({"wind_1": 0.5, "ccgt_1": 45.0})

        gen_summary, su_summary = compute_redispatch_volumes(
            wholesale_gen=dispatch_ws,
            physical_gen=dispatch_phys,
            wholesale_su=pd.DataFrame(index=snapshots),
            physical_su=pd.DataFrame(index=snapshots),
            gen_offer_prices=offer,
            gen_bid_prices=bid,
            su_offer_prices=pd.Series(dtype=float),
            su_bid_prices=pd.Series(dtype=float),
            network=market_network,
            logger=logger,
        )

        assert gen_summary["increase_MWh"].sum() == pytest.approx(0.0)
        assert gen_summary["decrease_MWh"].sum() == pytest.approx(0.0)
        assert gen_summary["net_cost"].sum() == pytest.approx(0.0)

    def test_known_redispatch(self, market_network):
        """Known increase/decrease produces expected costs."""
        import logging

        logger = logging.getLogger("test_market")

        snapshots = market_network.snapshots  # 6 hourly snapshots

        # Wholesale: wind=100 all hours, ccgt=200 all hours
        dispatch_ws = pd.DataFrame(
            {"wind_1": [100.0] * 6, "ccgt_1": [200.0] * 6}, index=snapshots
        )
        # Physical: wind curtailed to 50, ccgt increased to 250
        dispatch_phys = pd.DataFrame(
            {"wind_1": [50.0] * 6, "ccgt_1": [250.0] * 6}, index=snapshots
        )

        offer = pd.Series({"wind_1": 0.5, "ccgt_1": 55.0})
        bid = pd.Series({"wind_1": 0.5, "ccgt_1": 45.0})

        gen_summary, _ = compute_redispatch_volumes(
            wholesale_gen=dispatch_ws,
            physical_gen=dispatch_phys,
            wholesale_su=pd.DataFrame(index=snapshots),
            physical_su=pd.DataFrame(index=snapshots),
            gen_offer_prices=offer,
            gen_bid_prices=bid,
            su_offer_prices=pd.Series(dtype=float),
            su_bid_prices=pd.Series(dtype=float),
            network=market_network,
            logger=logger,
        )

        # Wind: decrease = 50 MW * 6h = 300 MWh, bid_cost = 300 * 0.5 = 150
        wind_row = gen_summary[gen_summary["component"] == "wind_1"].iloc[0]
        assert wind_row["decrease_MWh"] == pytest.approx(300.0)
        assert wind_row["increase_MWh"] == pytest.approx(0.0)
        assert wind_row["bid_cost"] == pytest.approx(150.0)

        # CCGT: increase = 50 MW * 6h = 300 MWh, offer_cost = 300 * 55 = 16500
        ccgt_row = gen_summary[gen_summary["component"] == "ccgt_1"].iloc[0]
        assert ccgt_row["increase_MWh"] == pytest.approx(300.0)
        assert ccgt_row["decrease_MWh"] == pytest.approx(0.0)
        assert ccgt_row["offer_cost"] == pytest.approx(16500.0)

    def test_time_varying_prices_override_static_costs(self, market_network):
        """Time-varying BM prices should drive redispatch cost accounting when provided."""
        import logging

        logger = logging.getLogger("test_market")

        snapshots = market_network.snapshots
        dispatch_ws = pd.DataFrame(
            {"wind_1": [100.0] * 6, "ccgt_1": [200.0] * 6}, index=snapshots
        )
        dispatch_phys = pd.DataFrame(
            {"wind_1": [50.0] * 6, "ccgt_1": [250.0] * 6}, index=snapshots
        )

        offer = pd.Series({"wind_1": 0.5, "ccgt_1": 55.0})
        bid = pd.Series({"wind_1": 0.5, "ccgt_1": 45.0})
        offer_tv = pd.DataFrame(
            {"ccgt_1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}, index=snapshots
        )
        bid_tv = pd.DataFrame(
            {"wind_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=snapshots
        )

        gen_summary, _ = compute_redispatch_volumes(
            wholesale_gen=dispatch_ws,
            physical_gen=dispatch_phys,
            wholesale_su=pd.DataFrame(index=snapshots),
            physical_su=pd.DataFrame(index=snapshots),
            gen_offer_prices=offer,
            gen_bid_prices=bid,
            su_offer_prices=pd.Series(dtype=float),
            su_bid_prices=pd.Series(dtype=float),
            network=market_network,
            logger=logger,
            gen_offer_prices_tv=offer_tv,
            gen_bid_prices_tv=bid_tv,
        )

        wind_row = gen_summary[gen_summary["component"] == "wind_1"].iloc[0]
        ccgt_row = gen_summary[gen_summary["component"] == "ccgt_1"].iloc[0]

        # 50 MW decrease each hour * sum([1,2,3,4,5,6]) = 50 * 21 = 1050
        assert wind_row["bid_cost"] == pytest.approx(1050.0)
        # 50 MW increase each hour * sum([10,20,30,40,50,60]) = 50 * 210 = 10500
        assert ccgt_row["offer_cost"] == pytest.approx(10500.0)


# ══════════════════════════════════════════════════════════════════════════════
# CONGESTION IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIdentifyCongestedBoundaries:
    """Tests for identify_congested_boundaries()."""

    def test_no_congestion(self, market_network):
        """No congestion if flows are well below s_nom."""
        import logging

        logger = logging.getLogger("test_market")

        # Set low flows (well below s_nom=500)
        market_network.lines_t.p0 = pd.DataFrame(
            {"line_AB": [100.0] * 6}, index=market_network.snapshots
        )

        result = identify_congested_boundaries(market_network, threshold=0.95, logger=logger)
        assert len(result) == 0

    def test_congestion_detected(self, market_network):
        """Congestion detected when flow exceeds threshold * s_nom."""
        import logging

        logger = logging.getLogger("test_market")

        # Set flows at 98% of s_nom=500 → above 95% threshold
        market_network.lines_t.p0 = pd.DataFrame(
            {"line_AB": [490.0] * 6}, index=market_network.snapshots
        )

        result = identify_congested_boundaries(market_network, threshold=0.95, logger=logger)
        assert len(result) == 1
        assert result.iloc[0]["component"] == "line_AB"
        assert result.iloc[0]["hours_congested"] == 6
        assert result.iloc[0]["max_loading_fraction"] == pytest.approx(0.98, abs=0.01)
