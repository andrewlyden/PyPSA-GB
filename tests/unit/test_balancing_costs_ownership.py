"""Item BM1, step 2 of the P2 bill chain: the balancing-cost extract knows who owns what.

Under P2 a CfD unit bids ``(1 - s) x contract value + s x (public price + tie step)``,
where ``s`` is the state's share of it. The extract recomputes what the balancing
mechanism paid each unit for being turned down and checks it against the LP's own
accepted bid cost; priced at full contract value it would refuse every P2 month. It
now prices the same blend, taken from the month's own bid audit, and reports the
private and the state parts apart, because the state's part is a transfer to itself.
"""

import pandas as pd
import pytest

from scripts.consumer_bill.extract_future_balancing_costs import (
    compute_cfd_curtailment_compensation, extract_month, ownership_from_audit, tie_break_step_cost,
)
from scripts.market.cfd_strike_table import StrikeTable

from tests.unit.test_future_consumer_bill import _redispatch_with_cfd_bid_cost, _toy_month_inputs, _toy_networks

SNAPSHOTS = pd.date_range("2030-01-01", periods=2, freq="h")
TABLE = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0}, 2024, {}, {})


def _audit(tmp_path, *, share, step=0.0, public_price=0.0, contract_value_mean=80.0):
    path = tmp_path / "contract_value_bids.csv"
    mean_bid = (1 - share) * contract_value_mean + share * public_price + step
    pd.DataFrame([{"generator": "curtailed", "pricing": "cfd_contract_value", "state_share": share,
                   "contract_value_mean_bid_gbp_per_mwh": contract_value_mean,
                   "mean_bid_gbp_per_mwh": mean_bid, "tie_break_step_gbp_per_mwh": step}]).to_csv(path, index=False)
    return path


def test_the_audit_gives_the_share_the_step_and_the_public_price(tmp_path):
    ownership = ownership_from_audit(_audit(tmp_path, share=0.75, step=0.5, public_price=2.0))
    assert ownership["state_share"]["curtailed"] == pytest.approx(0.75)
    assert ownership["tie_break_step"]["curtailed"] == pytest.approx(0.5)
    assert ownership["public_bid_price"] == pytest.approx(2.0)


def test_an_audit_without_state_ownership_changes_nothing(tmp_path):
    assert ownership_from_audit(_audit(tmp_path, share=0.0)) is None
    path = tmp_path / "old.csv"
    pd.DataFrame([{"generator": "curtailed", "pricing": "cfd_contract_value"}]).to_csv(path, index=False)
    assert ownership_from_audit(path) is None


def test_inconsistent_public_prices_are_refused(tmp_path):
    path = tmp_path / "contract_value_bids.csv"
    rows = []
    for name, price in (("a", 0.0), ("b", 5.0)):
        rows.append({"generator": name, "pricing": "cfd_contract_value", "state_share": 1.0,
                     "contract_value_mean_bid_gbp_per_mwh": 80.0, "mean_bid_gbp_per_mwh": price,
                     "tie_break_step_gbp_per_mwh": 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="public bid price"):
        ownership_from_audit(path)


def test_the_state_share_is_paid_its_blended_bid_and_reported_apart(tmp_path):
    wholesale, balancing = _toy_networks(
        tmp_path, [[80.0, 50.0], [60.0, 50.0]], [[30.0, 50.0], [60.0, 50.0]], SNAPSHOTS)
    prices = pd.Series([20.0, 40.0], index=SNAPSHOTS)
    ownership = ownership_from_audit(_audit(tmp_path, share=0.75, step=0.5))
    details, _ = compute_cfd_curtailment_compensation(wholesale, balancing, prices, TABLE, ownership=ownership)
    row = details.set_index("generator").loc["curtailed"]
    # 50 MWh turned down at a foregone 80 GBP/MWh: the private quarter is paid 1,000;
    # the state's three quarters bid 0 plus its share-weighted 0.5 step on 50 MWh, 25.
    assert row["compensation_private_gbp"] == pytest.approx(1000.0)
    assert row["compensation_state_gbp"] == pytest.approx(25.0)
    assert row["compensation_gbp"] == pytest.approx(1025.0)


def test_a_p2_month_reconciles_with_the_lp_blended_bid_cost(tmp_path):
    wholesale, balancing, price, summary, _ = _toy_month_inputs(tmp_path, SNAPSHOTS)
    _, month, _ = extract_month(
        "toy", wholesale, balancing, price, summary, TABLE,
        compensation_treatment="reconcile_against_lp_bid",
        redispatch_summary_path=_redispatch_with_cfd_bid_cost(tmp_path, 1025.0),
        contract_value_bids_path=_audit(tmp_path, share=0.75, step=0.5),
    )
    assert month["cfd_curtailment_compensation_gbp"] == pytest.approx(1025.0)
    assert month["cfd_curtailment_compensation_private_gbp"] == pytest.approx(1000.0)
    assert month["cfd_curtailment_compensation_state_gbp"] == pytest.approx(25.0)
    assert month["cfd_compensation_reconciliation_gap_gbp"] == pytest.approx(0.0)


def test_p3_tie_steps_are_taken_out_of_the_constraint_cost(tmp_path):
    """Item C7: the tie step orders a tie, so step x turn-down leaves the bill's constraint line."""
    wholesale, balancing, price, summary, _ = _toy_month_inputs(tmp_path, SNAPSHOTS)
    audit = _audit(tmp_path, share=0.75, step=0.5)

    def _redispatch(name, bid_cost):
        path = tmp_path / name
        pd.DataFrame({"component": ["curtailed", "steady"], "type": ["generator", "generator"],
                      "carrier": ["wind_offshore", "CCGT"], "increase_MWh": [0.0, 0.0],
                      "decrease_MWh": [50.0, 0.0], "offer_cost": [0.0, 0.0],
                      "bid_cost": [bid_cost, 0.0]}).to_csv(path, index=False)
        return path

    redispatch = _redispatch("redispatch_steps.csv", 1025.0)
    # Same netting as decompose_public_bm_bids: 0.5 x 50 MWh.
    assert tie_break_step_cost(redispatch, audit) == pytest.approx(25.0)
    _, with_steps, _ = extract_month(
        "toy", wholesale, balancing, price, summary, TABLE,
        compensation_treatment="reconcile_against_lp_bid",
        redispatch_summary_path=redispatch, contract_value_bids_path=audit,
    )
    _, without, _ = extract_month(
        "toy", wholesale, balancing, price, summary, TABLE,
        compensation_treatment="reconcile_against_lp_bid",
        redispatch_summary_path=_redispatch("redispatch_no_steps.csv", 1000.0),
        contract_value_bids_path=_audit(tmp_path, share=0.75, step=0.0),
    )
    assert with_steps["tie_break_step_cost_gbp"] == pytest.approx(25.0)
    assert without["tie_break_step_cost_gbp"] == pytest.approx(0.0)
    assert without["constraint_cost_gbp"] - with_steps["constraint_cost_gbp"] == pytest.approx(25.0)
    # The recompute keeps the step, so it still reconciles with the LP bid cost.
    assert with_steps["cfd_compensation_reconciliation_gap_gbp"] == pytest.approx(0.0)
