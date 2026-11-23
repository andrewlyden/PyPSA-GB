"""Storage churn share (market.wholesale.storage_churn_share), 9 Oct 2026."""
import logging

import pypsa
import pytest

from scripts.market.market_utils import apply_storage_churn_share


def _net():
    n = pypsa.Network()
    n.add("Bus", "b")
    for i, p in enumerate([100.0, 100.0, 200.0]):
        n.add("StorageUnit", f"bat{i}", bus="b", carrier="Battery", p_nom=p, max_hours=2,
              efficiency_store=0.92, efficiency_dispatch=0.92, marginal_cost=1.0)
    n.add("StorageUnit", "laes", bus="b", carrier="LAES", p_nom=50.0, max_hours=8,
          efficiency_store=0.70 ** 0.5, efficiency_dispatch=0.70 ** 0.5, marginal_cost=1.5)
    return n


def test_absent_share_changes_nothing():
    n = _net()
    assert apply_storage_churn_share(n, {}, logging.getLogger()) == 0
    assert list(n.storage_units.marginal_cost) == [1.0, 1.0, 1.0, 1.5]


def test_share_blocks_the_rest_of_each_carrier_at_the_no_churn_cost():
    n = _net()
    cfg = {"storage_churn_share": 0.5, "renewable_curtailment_order": {"ar4_plus_bid_gbp_per_mwh": -10.0}}
    apply_storage_churn_share(n, cfg, logging.getLogger())
    mc = n.storage_units.marginal_cost
    rte = 0.92 * 0.92
    assert mc["bat0"] == 1.0 and mc["bat1"] == 1.0                      # first half of battery power
    assert mc["bat2"] == pytest.approx(10 * (1 - rte) / rte + 1)          # blocked
    assert mc["laes"] == 1.5                                              # a lone unit starts inside the share


def test_share_one_keeps_every_unit_and_bad_share_is_refused():
    n = _net()
    assert apply_storage_churn_share(n, {"storage_churn_share": 1.0}, logging.getLogger()) == 0
    with pytest.raises(ValueError):
        apply_storage_churn_share(n, {"storage_churn_share": 1.5}, logging.getLogger())
