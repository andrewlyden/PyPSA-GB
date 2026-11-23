"""Item F24.2 (user decision, 24 Oct 2026): the CP.32 twins move the CfD contract-value bids.

A contract-value bid is the foregone difference payment, strike less the Stage 1
reference, so at a price level the bid on the same turn-down moves with the reference.
The twin's balancing-cost extraction puts the change on the constraint line and the
twin's asset settlement puts it, unit for unit, on each unit's balancing settlement, from
one function, so the bill and the register move by the same money. The state-owned share
bids its public price and does not move; nor does a unit outside contract-value pricing
or one turned down at the non-participant penalty.
"""

import json

import pandas as pd
import pytest

from scripts.consumer_bill.extract_future_balancing_costs import (
    cfd_bid_change_at_price_level, extract_month,
)
from scripts.market.cfd_strike_table import StrikeTable
from tests.unit.test_future_consumer_bill import _redispatch_with_cfd_bid_cost, _toy_month_inputs, _toy_networks

SNAPSHOTS = pd.date_range("2030-01-01", periods=2, freq="h")
TABLE = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0}, 2024, {}, {})
MODEL = pd.Series([20.0, 40.0], index=SNAPSHOTS)
# The level's reference: hour 1, where 'curtailed' is turned down 50 MWh, moves from 20 to 50.
LEVEL = pd.Series([50.0, 70.0], index=SNAPSHOTS)


def _audit(tmp_path, rows, name="contract_value_bids.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _priced(share=0.0, generator="curtailed"):
    return {"generator": generator, "pricing": "cfd_contract_value", "state_share": share,
            "contract_value_mean_bid_gbp_per_mwh": 80.0, "mean_bid_gbp_per_mwh": (1 - share) * 80.0,
            "tie_break_step_gbp_per_mwh": 0.0}


def _networks(tmp_path):
    return _toy_networks(tmp_path, [[80.0, 50.0], [60.0, 50.0]], [[30.0, 50.0], [60.0, 50.0]], SNAPSHOTS)


def test_the_bid_moves_by_the_turn_down_times_the_reference_change(tmp_path):
    wholesale, balancing = _networks(tmp_path)
    moved = cfd_bid_change_at_price_level(wholesale, balancing, MODEL, LEVEL, TABLE,
                                          _audit(tmp_path, [_priced()])).set_index("generator")
    # 50 MWh x ((100 - 50) - (100 - 20)) = -1,500: a higher reference forgoes less.
    assert moved.at["curtailed", "compensation_model_gbp"] == pytest.approx(50.0 * 80.0)
    assert moved.at["curtailed", "compensation_at_price_level_gbp"] == pytest.approx(50.0 * 50.0)
    assert moved.at["curtailed", "change_gbp"] == pytest.approx(-1500.0)
    assert moved.at["curtailed", "gross_change_gbp"] == pytest.approx(-1500.0)
    # 'steady' is not in the audit's contract-value pricing: it does not move.
    assert "steady" not in moved.index


def test_only_the_private_share_of_a_state_owned_unit_moves(tmp_path):
    wholesale, balancing = _networks(tmp_path)
    moved = cfd_bid_change_at_price_level(wholesale, balancing, MODEL, LEVEL, TABLE,
                                          _audit(tmp_path, [_priced(share=0.75)])).set_index("generator")
    assert moved.at["curtailed", "gross_change_gbp"] == pytest.approx(-1500.0)
    assert moved.at["curtailed", "change_gbp"] == pytest.approx(-375.0)


def test_a_unit_turned_down_at_the_penalty_does_not_move(tmp_path):
    wholesale, balancing = _networks(tmp_path)
    redispatch = tmp_path / "redispatch.csv"
    # 50 MWh at a penalty-sized bid: outside the balancing mechanism, priced at the penalty, not contract value.
    pd.DataFrame({"component": ["curtailed", "steady"], "type": "generator", "carrier": "wind_offshore",
                  "decrease_MWh": [50.0, 0.0], "bid_cost": [50.0 * 1e5, 0.0]}).to_csv(redispatch, index=False)
    moved = cfd_bid_change_at_price_level(wholesale, balancing, MODEL, LEVEL, TABLE,
                                          _audit(tmp_path, [_priced()]), redispatch)
    assert moved.empty


def test_bids_off_contract_value_and_a_short_reference_are_refused(tmp_path):
    wholesale, balancing = _networks(tmp_path)
    scalar = _audit(tmp_path, [{"generator": "curtailed", "pricing": "scalar"}], "scalar.csv")
    with pytest.raises(ValueError, match="contract value"):
        cfd_bid_change_at_price_level(wholesale, balancing, MODEL, LEVEL, TABLE, scalar)
    with pytest.raises(ValueError, match="misses 1"):
        cfd_bid_change_at_price_level(wholesale, balancing, MODEL, LEVEL.iloc[:1], TABLE,
                                      _audit(tmp_path, [_priced()]))


def test_the_month_constraint_cost_takes_the_change_and_reconciles_at_the_model_price(tmp_path):
    wholesale, balancing, price, summary, _ = _toy_month_inputs(tmp_path, SNAPSHOTS)
    audit = _audit(tmp_path, [_priced()])
    redispatch = _redispatch_with_cfd_bid_cost(tmp_path, 4000.0)
    kwargs = dict(compensation_treatment="reconcile_against_lp_bid", redispatch_summary_path=redispatch,
                  contract_value_bids_path=audit)
    _, model, _ = extract_month("toy", wholesale, balancing, price, summary, TABLE, **kwargs)
    _, level, _ = extract_month("toy", wholesale, balancing, price, summary, TABLE, level_price=LEVEL, **kwargs)
    assert "cfd_bid_price_level_change_gbp" not in model
    assert level["cfd_bid_price_level_change_gbp"] == pytest.approx(-1500.0)
    assert level["constraint_cost_gbp"] - model["constraint_cost_gbp"] == pytest.approx(-1500.0)
    # The LP bid it reconciles against is the model-price bid, so the check is unchanged.
    assert level["cfd_compensation_reconciliation_gap_gbp"] == model["cfd_compensation_reconciliation_gap_gbp"]


def test_the_bills_identity_check_counts_the_change(tmp_path):
    from scripts.consumer_bill.project_future_bill import read_balancing_costs

    annual = {"model_price_year": 2024, "constraint_cost_excludes_slack_and_penalty": True,
              "accepted_offer_cost_gbp": 500.0, "accepted_bid_cost_gbp": 4000.0,
              "non_participant_turn_down_cost_at_penalty_gbp": 0.0,
              "feasibility_slack_cost_in_accepted_costs_gbp": 0.0, "cfd_curtailment_compensation_gbp": 4000.0,
              "cfd_curtailed_mwh": 50.0, "feasibility_slack_cost_gbp": 0.0, "feasibility_slack_energy_mwh": 0.0,
              "cfd_curtailment_compensation_treatment": "reconcile_against_lp_bid",
              "price_level": "cp32", "cfd_bid_price_level_change_gbp": -1500.0,
              "price_level_mapping_check": {"model": {}, "mapped": {}}, "constraint_cost_gbp": 3000.0}
    path = tmp_path / "balancing.json"
    path.write_text(json.dumps({"annual": annual}), encoding="utf-8")
    out = read_balancing_costs(path, ["toy"], 2024)
    assert out["constraint_cost_gbp"] == 3000.0
    assert out["price_level"] == "cp32" and out["cfd_bid_price_level_change_gbp"] == -1500.0
    path.write_text(json.dumps({"annual": {**annual, "constraint_cost_gbp": 4500.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="price-level change"):
        read_balancing_costs(path, ["toy"], 2024)


# ── the twin's asset settlement takes the same change, unit for unit ──────────

SETTLEMENT = pd.DataFrame({
    "component_type": ["Generator", "Generator", "StorageUnit"], "asset_id": ["curtailed", "ccgt", "curtailed"],
    "wholesale_revenue_gbp": [4000.0, 3000.0, -20.0], "balancing_settlement_gbp": [-3000.0, 1500.0, 50.0],
    "cfd_curtailment_compensation_gbp": [4000.0, 0.0, 0.0], "market_revenue_gbp": [1000.0, 4500.0, 30.0],
    "support_revenue_gbp": [500.0, 0.0, 0.0],
})


def test_the_settlement_moves_each_units_balancing_settlement_by_its_change():
    from scripts.public_ownership.price_level_settlement import apply_cfd_bid_changes

    changes = pd.DataFrame({"change_gbp": [-375.0], "gross_change_gbp": [-1500.0]}, index=["curtailed"])
    out = apply_cfd_bid_changes(SETTLEMENT, changes)
    gen = out.iloc[0]
    assert gen["balancing_settlement_gbp"] == pytest.approx(-3375.0)
    assert gen["cfd_curtailment_compensation_gbp"] == pytest.approx(2500.0)
    assert gen["market_revenue_gbp"] == pytest.approx(4000.0 - 3375.0)
    # Support revenue never held the compensation under contract-value bids; other rows are untouched,
    # the storage unit of the same name included.
    assert gen["support_revenue_gbp"] == 500.0
    pd.testing.assert_frame_equal(out.iloc[1:][["balancing_settlement_gbp", "cfd_curtailment_compensation_gbp"]],
                                  SETTLEMENT.iloc[1:][["balancing_settlement_gbp", "cfd_curtailment_compensation_gbp"]])
    assert float(out["balancing_settlement_gbp"].sum() - SETTLEMENT["balancing_settlement_gbp"].sum()) == \
        pytest.approx(float(changes["change_gbp"].sum()))
    with pytest.raises(ValueError, match="no settlement row"):
        apply_cfd_bid_changes(SETTLEMENT, pd.DataFrame({"change_gbp": [1.0], "gross_change_gbp": [1.0]},
                                                       index=["ghost"]))


def test_the_settlement_and_the_extraction_take_the_same_total(tmp_path):
    """Both read cfd_bid_change_at_price_level on the same months, so the totals are one number."""
    from scripts.public_ownership import price_level_settlement as pls

    wholesale, balancing, price, summary, redispatch = _toy_month_inputs(tmp_path, SNAPSHOTS)
    audit = _audit(tmp_path, [_priced(share=0.75)])
    assumptions = tmp_path / "assumptions.json"
    assumptions.write_text(json.dumps({"cfd_strike_indexation": {"factor": 1.0}, "model_linked_rules": {"cfd": {}}}),
                           encoding="utf-8")
    from scripts.market import cfd_strike_table

    original = cfd_strike_table.load_strike_table
    try:
        cfd_strike_table.load_strike_table = lambda *args, **kwargs: TABLE
        changes = pls.cfd_bid_changes([wholesale], [balancing], [price], [redispatch], [audit], LEVEL,
                                      assumptions=assumptions, strike_table="unused.csv", model_price_year=2024)
    finally:
        cfd_strike_table.load_strike_table = original
    _, month, _ = extract_month("toy", wholesale, balancing, price, summary, TABLE, level_price=LEVEL,
                                compensation_treatment="reconcile_against_lp_bid",
                                redispatch_summary_path=_redispatch_with_cfd_bid_cost(tmp_path, 1000.0),
                                contract_value_bids_path=audit)
    assert float(changes["change_gbp"].sum()) == pytest.approx(month["cfd_bid_price_level_change_gbp"])
    assert changes.at["curtailed", "change_gbp"] == pytest.approx(-375.0)


def test_the_bid_audit_is_found_beside_the_redispatch_summary(tmp_path):
    from scripts.public_ownership.price_level_settlement import audit_beside

    found = audit_beside(tmp_path / "CP30_X_Jan2030_fin5_redispatch_summary.csv")
    assert found == tmp_path / "CP30_X_Jan2030_fin5_contract_value_bids.csv"
    with pytest.raises(ValueError, match="not a redispatch summary"):
        audit_beside(tmp_path / "something.csv")
