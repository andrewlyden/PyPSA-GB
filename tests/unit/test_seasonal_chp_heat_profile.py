"""Heat-led CHP: the heat credit and the must-run floor follow heat demand.

Before 26 Sep 2026 gas CHP carried a flat 66 GBP/MWh heat-credited cost and a
30% floor all year, so it ran at CF 0.58 against CP30's 0.36 and its busiest
month was July. The user decision is a seasonal credit and floor; these tests
pin the shape (mean 1 over the year, winter above summer), the time-varying
marginal cost and floor it writes, and that other carriers are untouched.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.generators.apply_marginal_costs import (
    DEFAULT_MONTHLY_HEATING_DEGREE_DAYS,
    apply_marginal_costs_to_network,
    apply_seasonal_heat_profile,
    seasonal_heat_factors,
)
from scripts.market.market_utils import calculate_bid_offer_prices

FUEL = {"gas": 30.0, "CHP": 30.0, "CCGT": 30.0}
CARBON = 73.0


def _scenario(seasonal=None):
    chp = {"mode": "heat_credit", "heat_fuel": "gas", "heat_to_power": 1.1, "boiler_efficiency": 0.85}
    if seasonal is not None:
        chp["seasonal"] = seasonal
    return {"modelled_year": 2030, "marginal_costs": {"byproduct_adjustments": {"CHP": chp}}}


@pytest.fixture
def chp_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h"))
    n.add("Bus", "bus", carrier="AC")
    n.add("Generator", "chp", bus="bus", carrier="CHP", p_nom=100.0, efficiency=0.42, p_min_pu=0.3)
    n.add("Generator", "ccgt", bus="bus", carrier="CCGT", p_nom=500.0, efficiency=0.52, p_min_pu=0.0)
    return n


def _build(n, seasonal=None):
    config = _scenario(seasonal)
    mc_df = apply_marginal_costs_to_network(n, FUEL, CARBON, config)
    flat = float(n.generators.at["chp", "marginal_cost"])
    mc_df = apply_seasonal_heat_profile(n, mc_df, config)
    return mc_df, flat


@pytest.mark.unit
class TestSeasonalHeatFactors:

    def test_mean_one_over_the_calendar_year_and_winter_above_summer(self):
        snaps = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")
        factor = seasonal_heat_factors(snaps, {"base_load_share": 0.5})
        assert factor.mean() == pytest.approx(1.0, abs=1e-9)
        by_month = factor.groupby(snaps.month).first()
        assert by_month.idxmax() == int(np.argmax(DEFAULT_MONTHLY_HEATING_DEGREE_DAYS)) + 1
        assert by_month.loc[1] == pytest.approx(1.285, abs=1e-3) and by_month.loc[8] == pytest.approx(0.710, abs=1e-3)

    def test_mean_is_a_calendar_property_not_the_solve_window(self):
        july = pd.date_range("2030-07-01", periods=24 * 31, freq="h")
        assert seasonal_heat_factors(july, None).mean() < 0.75

    def test_base_one_is_flat_and_bad_input_is_refused(self):
        snaps = pd.date_range("2030-01-01", periods=48, freq="h")
        assert seasonal_heat_factors(snaps, {"base_load_share": 1.0}).tolist() == pytest.approx([1.0] * 48)
        with pytest.raises(ValueError):
            seasonal_heat_factors(snaps, {"monthly_heating_degree_days": [1.0] * 11})


@pytest.mark.unit
class TestSeasonalHeatProfile:

    def test_chp_cost_and_floor_follow_heat_demand(self, chp_network):
        mc_df, flat = _build(chp_network)
        mc_t = chp_network.generators_t.marginal_cost["chp"]
        by_month = mc_t.groupby(mc_t.index.month).mean()
        # Credit larger in winter: cheaper to run in January than in August.
        assert by_month.loc[1] < flat < by_month.loc[8]
        # Nothing clipped at zero here, so the annual mean is the flat value.
        assert mc_t.mean() == pytest.approx(flat, rel=1e-9)
        assert chp_network.generators.at["chp", "marginal_cost"] == pytest.approx(flat, rel=1e-9)
        floor = chp_network.generators_t.p_min_pu["chp"]
        floor_by_month = floor.groupby(floor.index.month).mean()
        assert floor_by_month.loc[1] == pytest.approx(0.3 * 1.285, abs=1e-3) and floor_by_month.loc[8] == pytest.approx(0.3 * 0.710, abs=1e-3)
        assert floor.mean() == pytest.approx(0.3, rel=1e-9)
        row = mc_df.set_index("generator").loc["chp"]
        assert row["marginal_cost_seasonal_min"] < row["marginal_cost"] < row["marginal_cost_seasonal_max"]
        assert "seasonal_heat" in row["source"]

    def test_other_carriers_untouched(self, chp_network):
        _build(chp_network)
        assert "ccgt" not in chp_network.generators_t.marginal_cost.columns
        assert "ccgt" not in chp_network.generators_t.p_min_pu.columns

    def test_switch_off_keeps_the_flat_credit(self, chp_network):
        _build(chp_network, seasonal={"enabled": False})
        assert "chp" not in chp_network.generators_t.marginal_cost.columns
        assert "chp" not in chp_network.generators_t.p_min_pu.columns

    def test_floor_can_be_left_flat(self, chp_network):
        _build(chp_network, seasonal={"apply_to_min_output": False})
        assert "chp" in chp_network.generators_t.marginal_cost.columns
        assert "chp" not in chp_network.generators_t.p_min_pu.columns

    def test_balancing_prices_follow_the_seasonal_cost(self, chp_network):
        """Derived BM prices read the time-varying cost, not the annual mean."""
        _build(chp_network)
        market = {"balancing": {"bid_offer_source": "derived", "default_offer_markup": 0.1,
                                "default_bid_discount": 0.1}}
        calculate_bid_offer_prices(chp_network, market, logging.getLogger("test_chp"), time_varying=True)
        mc_t = chp_network.generators_t.marginal_cost["chp"]
        offer = chp_network._bm_offer_tv["chp"]
        bid = chp_network._bm_bid_tv["chp"]
        assert offer.to_numpy() == pytest.approx((mc_t * 1.1).to_numpy())
        assert bid.to_numpy() == pytest.approx((-0.9 * mc_t).to_numpy())
        # A unit without a time-varying cost keeps its static price in the frame.
        ccgt_mc = chp_network.generators.at["ccgt", "marginal_cost"]
        assert chp_network._bm_bid_tv["ccgt"].unique() == pytest.approx([-0.9 * ccgt_mc])
