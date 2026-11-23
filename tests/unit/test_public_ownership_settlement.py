"""The WP-D settlement must not pay a curtailed CfD generator twice.

``cfd_curtailment_compensation_gbp`` is the foregone difference payment restored
through the balancing bid. When the balancing solve priced CfD bids on contract
value it is the same money as the CfD share of ``balancing_settlement_gbp``, so
support revenue must count it once. On the CP30 2030 year adding both overstated
CfD contract revenue by GBP 0.994 bn (5.1%), inflated the public account's rent
and drove the at-cost strike basis to negative strikes.
"""

import pandas as pd
import pytest

from scripts.public_ownership.settlement import (
    COMPENSATION_RECONCILIATION_FLOOR_GBP,
    COMPENSATION_RECONCILIATION_TOLERANCE,
    combine_settlement_revenue,
)


def _frame(compensation=1_000.0, balancing=1_000.0):
    """One curtailed CfD generator and one unsupported thermal unit."""
    return pd.DataFrame({
        "component_type": ["Generator", "Generator"],
        "asset_id": ["wind_cfd", "ccgt"],
        "carrier": ["wind_offshore", "CCGT"],
        "support_type": ["CfD", ""],
        "wholesale_revenue_gbp": [5_000.0, 8_000.0],
        "balancing_settlement_gbp": [balancing, 300.0],
        "cfd_support_gbp": [2_000.0, 0.0],
        "cfd_curtailment_compensation_gbp": [compensation, 0.0],
        "roc_support_gbp": [0.0, 0.0],
    })


class TestContractValueBids:
    def test_compensation_is_not_added_again(self):
        out = combine_settlement_revenue(_frame(), priced_on_contract_value=True)
        # The bid payment is already in market revenue; support revenue is the
        # difference payments alone.
        assert out.loc[0, "market_revenue_gbp"] == pytest.approx(6_000.0)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(2_000.0)
        total = out.loc[0, "market_revenue_gbp"] + out.loc[0, "support_revenue_gbp"]
        assert total == pytest.approx(8_000.0)

    def test_a_divergence_from_the_balancing_settlement_is_refused(self):
        # A compensation that is not the CfD fleet's balancing settlement means
        # a different strike table or reference series, so dropping it is unsafe.
        with pytest.raises(ValueError, match="contract value"):
            combine_settlement_revenue(_frame(compensation=1.0e9, balancing=5.0e8),
                                       priced_on_contract_value=True)

    def test_a_residual_inside_the_tolerance_is_accepted(self):
        within = 1_000.0 * (1.0 - COMPENSATION_RECONCILIATION_TOLERANCE / 2.0)
        out = combine_settlement_revenue(_frame(compensation=1_000.0, balancing=within),
                                         priced_on_contract_value=True)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(2_000.0)

    def test_offers_to_a_cfd_unit_do_not_break_the_reconciliation(self):
        # A CfD biomass unit turned up earns offers the compensation never covers
        # (Drax, 25 Sep 2026 run: GBP 15.5 m, a 2.39% "gap" against the net).
        # The identity holds on the bid side, so that is what is compared.
        frame = _frame(compensation=1_000.0, balancing=1_500.0)
        frame["balancing_bid_gbp"] = [1_000.0, 0.0]
        out = combine_settlement_revenue(frame, priced_on_contract_value=True)
        assert out.loc[0, "market_revenue_gbp"] == pytest.approx(6_500.0)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(2_000.0)

    def test_a_bid_side_divergence_is_still_refused(self):
        frame = _frame(compensation=1.0e9, balancing=1.0e9)
        frame["balancing_bid_gbp"] = [5.0e8, 0.0]
        with pytest.raises(ValueError, match="contract value"):
            combine_settlement_revenue(frame, priced_on_contract_value=True)

    def test_only_the_private_share_is_reconciled_under_state_owned_bids(self):
        # Item BM1: a unit 60% state-owned bids the public price on that share, so
        # only 40% of its compensation is in the bids (P2, 25 Sep 2026: 181.7 of
        # 1,542.6 m, 0.01% from the bids).
        frame = _frame(compensation=1_000.0, balancing=400.0)
        frame["balancing_bid_gbp"] = [400.0, 0.0]
        frame["state_share"] = [0.6, 0.0]
        out = combine_settlement_revenue(frame, priced_on_contract_value=True)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(2_000.0)

    def test_a_private_share_divergence_is_still_refused(self):
        frame = _frame(compensation=1.0e9, balancing=1.0e9)
        frame["balancing_bid_gbp"] = [1.0e9, 0.0]
        frame["state_share"] = [0.6, 0.0]
        with pytest.raises(ValueError, match="privately owned"):
            combine_settlement_revenue(frame, priced_on_contract_value=True)

    def test_an_immaterial_residual_passes_on_any_ratio(self):
        # P3, 29 Sep 2026: the fleet is state-owned, so GBP 879 of private
        # compensation met GBP 712 of bids, 18.96% apart and worth nothing.
        frame = _frame(compensation=879.0, balancing=712.0)
        frame["balancing_bid_gbp"] = [712.0, 0.0]
        out = combine_settlement_revenue(frame, priced_on_contract_value=True)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(2_000.0)

    def test_a_ratio_breach_above_the_floor_is_refused(self):
        compensation = 10.0 * COMPENSATION_RECONCILIATION_FLOOR_GBP / COMPENSATION_RECONCILIATION_TOLERANCE
        frame = _frame(compensation=compensation, balancing=compensation * 0.9)
        frame["balancing_bid_gbp"] = [compensation * 0.9, 0.0]
        with pytest.raises(ValueError, match="contract value"):
            combine_settlement_revenue(frame, priced_on_contract_value=True)

    def test_a_thermal_cfd_on_fuel_cost_bids_is_left_out_of_the_check(self):
        # Drax keeps its fuel-cost bids (cfd_dispatch_cost_exempt_carriers), so its bids are not contract value.
        compensation = 10.0 * COMPENSATION_RECONCILIATION_FLOOR_GBP / COMPENSATION_RECONCILIATION_TOLERANCE
        frame = _frame(compensation=compensation, balancing=compensation * 0.7)
        frame["balancing_bid_gbp"] = [compensation * 0.7, 0.0]
        frame.loc[0, "carrier"] = "biomass"
        out = combine_settlement_revenue(frame, priced_on_contract_value=True)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(2_000.0)

    def test_no_cfd_compensation_at_all_is_not_a_reconciliation_failure(self):
        out = combine_settlement_revenue(_frame(compensation=0.0, balancing=0.0),
                                         priced_on_contract_value=True)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(2_000.0)


class TestScalarBids:
    def test_compensation_is_added_when_the_solve_did_not_price_contract_value(self):
        # A scalar-zero LP bid leaves the generator uncompensated in the
        # balancing settlement, so the term has to be added back.
        out = combine_settlement_revenue(_frame(compensation=1_000.0, balancing=0.0),
                                         priced_on_contract_value=False)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(3_000.0)

    def test_no_reconciliation_is_required(self):
        # The two need not agree in this regime, so no error is raised.
        out = combine_settlement_revenue(_frame(compensation=1_000.0, balancing=500.0),
                                         priced_on_contract_value=False)
        assert out.loc[0, "support_revenue_gbp"] == pytest.approx(3_000.0)


def test_non_cfd_assets_are_untouched_either_way():
    for priced in (True, False):
        out = combine_settlement_revenue(_frame(), priced_on_contract_value=priced)
        assert out.loc[1, "market_revenue_gbp"] == pytest.approx(8_300.0)
        assert out.loc[1, "support_revenue_gbp"] == pytest.approx(0.0)
