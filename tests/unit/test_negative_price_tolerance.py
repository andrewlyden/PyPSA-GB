"""The negative-price rule ignores the AR4+ curtailment bid (-0.01) and its solver noise (12 Oct 2026)."""
import numpy as np
import pytest

from scripts.consumer_bill import project_future_bill as bill
from scripts.market.revenue_tracking import NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH, negative_price_eligibility


def test_tolerances_agree():
    assert NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH == bill.NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH == pytest.approx(0.50)


def test_bid_floor_and_noise_keep_ar4_paid_while_real_negatives_do_not():
    # an hour at the AR4+ bid, one with noise past it, one genuinely negative, one positive
    ref = np.array([-0.01, -0.01005, -1.0, 40.0])
    eligible = negative_price_eligibility(ref, ["AR4", "AR6"], "gb_contract_terms")
    assert eligible[:, 0].tolist() == [True, True, False, True]
    assert eligible[:, 1].tolist() == [True, True, False, True]


def test_six_hour_rule_counts_only_real_negatives():
    # seven hours at the bid floor do not make a six-hour negative run for AR2/AR3
    floor = np.full(7, -0.01005)
    assert negative_price_eligibility(floor, ["AR3"], "gb_contract_terms")[:, 0].all()
    real = np.full(7, -2.0)
    assert not negative_price_eligibility(real, ["AR3"], "gb_contract_terms")[:, 0].any()
