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
    """Default market config dict matching defaults.yaml structure."""
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

        # CCGT: marginal_cost=50, default markup=10%, discount=10%
        assert gen_offer["ccgt_1"] == pytest.approx(55.0)
        assert gen_bid["ccgt_1"] == pytest.approx(45.0)

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
        assert gen_bid["nuclear_1"] == pytest.approx(7.6)  # 8 * 0.95

    def test_wind_offer_zero_markup(self, market_network, default_market_config):
        """Wind generators with offer_markup=0.0 have offer_price == mc."""
        import logging

        logger = logging.getLogger("test_market")
        gen_offer, _, _, _ = calculate_bid_offer_prices(
            market_network, default_market_config, logger
        )

        # Wind: mc=0.5, offer_markup=0.0 → offer=0.5
        assert gen_offer["wind_1"] == pytest.approx(0.5)

    def test_bid_floor_for_zero_cost(self, market_network, default_market_config):
        """Bid price has a minimum floor (MIN_BID_FLOOR=0.50) for near-zero mc assets."""
        import logging

        logger = logging.getLogger("test_market")
        _, gen_bid, _, _ = calculate_bid_offer_prices(
            market_network, default_market_config, logger
        )

        # Wind: mc=0.5, bid_discount=0.05 → 0.5*0.95=0.475, but floor=0.50
        assert gen_bid["wind_1"] >= 0.50

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

    def test_csv_source_without_files_raises(self, market_network):
        """bid_offer_source='csv' without file paths raises ValueError."""
        import logging

        logger = logging.getLogger("test_market")
        bad_config = {
            "balancing": {"bid_offer_source": "csv"},
        }
        with pytest.raises(ValueError, match="csv"):
            calculate_bid_offer_prices(market_network, bad_config, logger)


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
