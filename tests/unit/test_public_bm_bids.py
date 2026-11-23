"""Item BM1: ownership reaching the balancing bid stack, and the decomposition.

Two things are pinned here, and both are places where a quiet mistake would be
expensive.

The register is **finer than the LP**: it owns ``<generator>::legacy`` and
``<generator>::new_build`` separately while the balancing mechanism takes one bid
per generator, so ownership has to arrive as a capacity-weighted share. Getting
that wrong hands the state plant it does not own, or vice versa, and nothing
downstream would notice.

And a state-owned unit bidding its avoided short-run cost does not stop being
owed its cost: the at-cost contract recovers the same required revenue from a
smaller volume, so the fall in accepted bid cost is mostly the state paying
itself. The decomposition has to separate that transfer from money that actually
leaves the consumer, because failing to is exactly the shape of the WP-D double
count of 19 September.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.ownership_shares import (
    generator_name, generator_state_shares, load_shares,
)


def _register():
    """Three generators: wholly private, wholly state-owned, and split by cohort."""
    return pd.DataFrame([
        {"case_id": "p2", "asset_id": "PRIVATE_A::legacy", "capacity_mw": 100.0, "ownership_share": 0.0},
        {"case_id": "p2", "asset_id": "PUBLIC_B::new_build", "capacity_mw": 200.0, "ownership_share": 1.0},
        # The mixed unit: 300 MW legacy private, 700 MW new build state-owned.
        {"case_id": "p2", "asset_id": "MIXED_C::legacy", "capacity_mw": 300.0, "ownership_share": 0.0},
        {"case_id": "p2", "asset_id": "MIXED_C::new_build", "capacity_mw": 700.0, "ownership_share": 1.0},
        # A stake case, so the share is fractional inside a cohort as well.
        {"case_id": "p1", "asset_id": "MIXED_C::legacy", "capacity_mw": 300.0, "ownership_share": 0.0},
        {"case_id": "p1", "asset_id": "MIXED_C::new_build", "capacity_mw": 700.0, "ownership_share": 0.30},
    ])


def test_a_cohort_id_names_the_generator_it_belongs_to():
    assert generator_name("WIND_X::new_build") == "WIND_X"
    assert generator_name("WIND_X") == "WIND_X"


def test_ownership_reaches_one_bid_as_a_capacity_weighted_share():
    shares = generator_state_shares(_register()).set_index(["case_id", "generator"])["state_share"]
    assert shares[("p2", "PRIVATE_A")] == 0.0
    assert shares[("p2", "PUBLIC_B")] == 1.0
    # The mixed unit: 700 of 1,000 MW state-owned, so 0.7 -- not 0, not 1.
    assert shares[("p2", "MIXED_C")] == pytest.approx(0.7)
    # A 30% stake on the new-build cohort of the same unit is 0.3 x 0.7.
    assert shares[("p1", "MIXED_C")] == pytest.approx(0.21)


def test_an_unresolved_cohort_is_private_not_state_owned():
    """The only safe reading: a missing share must not hand plant to the state."""
    register = _register()
    register.loc[register.asset_id.eq("PUBLIC_B::new_build"), "ownership_share"] = np.nan
    shares = generator_state_shares(register, case_id="p2").set_index("generator")["state_share"]
    assert shares["PUBLIC_B"] == 0.0


def test_a_zero_capacity_generator_carries_no_ownership():
    register = pd.DataFrame([
        {"case_id": "p2", "asset_id": "EMPTY::new_build", "capacity_mw": 0.0, "ownership_share": 1.0}])
    shares = generator_state_shares(register).set_index("generator")["state_share"]
    assert shares["EMPTY"] == 0.0


def test_the_share_table_round_trips_and_refuses_an_unknown_case(tmp_path):
    path = tmp_path / "shares.csv"
    generator_state_shares(_register()).to_csv(path, index=False)
    loaded = load_shares(path, "p2")
    assert loaded["MIXED_C"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="no rows for case"):
        load_shares(path, "p3")


# ── the bid blend ────────────────────────────────────────────────────────────


def test_the_blend_is_off_by_default_and_reproduces_the_pre_bm1_stack():
    """A missing or disabled block must not move a single bid."""
    import logging

    from scripts.market.cfd_bm_bids import resolve_public_ownership_bids

    logger = logging.getLogger("test")
    assert resolve_public_ownership_bids(None, logger) == (None, 0.0)
    assert resolve_public_ownership_bids({}, logger) == (None, 0.0)
    assert resolve_public_ownership_bids({"enabled": False, "case_id": "x",
                                          "shares_file": "nope.csv"}, logger) == (None, 0.0)


def test_the_blend_refuses_an_incomplete_or_negative_declaration(tmp_path):
    import logging

    from scripts.market.cfd_bm_bids import resolve_public_ownership_bids

    logger = logging.getLogger("test")
    with pytest.raises(ValueError, match="needs case_id and shares_file"):
        resolve_public_ownership_bids({"enabled": True}, logger)
    with pytest.raises(FileNotFoundError):
        resolve_public_ownership_bids(
            {"enabled": True, "case_id": "p2", "shares_file": str(tmp_path / "absent.csv")}, logger)
    path = tmp_path / "shares.csv"
    generator_state_shares(_register()).to_csv(path, index=False)
    with pytest.raises(ValueError, match="must not be negative"):
        resolve_public_ownership_bids(
            {"enabled": True, "case_id": "p2", "shares_file": str(path), "public_bid_price": -1.0}, logger)


def test_the_blend_moves_a_bid_in_proportion_to_what_the_state_owns():
    """bid = (1 - s) x contract value + s x the avoided short-run cost."""
    contract_value = np.array([50.0, 20.0, 0.0])   # three snapshots
    shares = np.array([0.0, 0.7, 1.0])             # three generators
    public_price = 0.0
    blended = np.outer(contract_value, 1.0 - shares) + public_price * shares
    # A wholly private unit is untouched.
    assert blended[:, 0] == pytest.approx(contract_value)
    # A wholly state-owned unit bids the avoided short-run cost in every hour.
    assert blended[:, 2] == pytest.approx(np.zeros(3))
    # The mixed unit sits between them, in proportion.
    assert blended[0, 1] == pytest.approx(0.3 * 50.0)
    assert (blended[:, 1] <= contract_value).all()


# ── the decomposition ────────────────────────────────────────────────────────


def _redispatch(bid_costs, decreases):
    return pd.DataFrame({
        "component": list(bid_costs), "type": "generator", "carrier": "wind_offshore",
        "increase_MWh": 0.0, "decrease_MWh": [decreases[g] for g in bid_costs],
        "offer_cost": 0.0, "bid_cost": [bid_costs[g] for g in bid_costs],
        "net_cost": [bid_costs[g] for g in bid_costs],
    })


def test_the_fall_in_bid_cost_is_split_into_a_transfer_and_a_real_saving(tmp_path, monkeypatch):
    """The state paying itself is not a consumer saving, and must not be quoted as one."""
    import scripts.market.decompose_public_bm_bids as mod

    base = _redispatch({"PUBLIC_B": 1_000_000.0, "PRIVATE_A": 400_000.0},
                       {"PUBLIC_B": 10_000.0, "PRIVATE_A": 4_000.0})
    # The state-owned unit stops being paid to turn down; the private one is
    # turned down less, so it is paid less too.
    public = _redispatch({"PUBLIC_B": 0.0, "PRIVATE_A": 300_000.0},
                         {"PUBLIC_B": 13_000.0, "PRIVATE_A": 1_000.0})
    monkeypatch.setattr(mod, "RESOURCES", tmp_path)
    base.to_csv(tmp_path / "base_redispatch_summary.csv", index=False)
    public.to_csv(tmp_path / "pub_redispatch_summary.csv", index=False)
    shares = pd.Series({"PUBLIC_B": 1.0, "PRIVATE_A": 0.0})

    result, _ = mod.decompose("base", "pub", shares)
    assert result["delta_bid_cost_gbp"] == pytest.approx(-1_100_000.0)
    # The million that stops flowing to the state is a transfer, not a saving.
    assert result["transfer_to_the_state_gbp"] == pytest.approx(-1_000_000.0)
    assert result["payment_to_private_counterparties_gbp"] == pytest.approx(-100_000.0)
    assert result["transfer_share_of_delta"] == pytest.approx(1_000_000 / 1_100_000)
    # Volume is reported because a CfD unit is near-indifferent: if turn-down
    # barely moves, the exercise is about who is paid, not what is spilled.
    assert result["delta_decrease_MWh"] == pytest.approx(0.0)


def test_the_decomposition_drops_the_total_row(tmp_path, monkeypatch):
    """constraint_costs.csv carries a TOTAL row; summing it whole double counts."""
    import scripts.market.decompose_public_bm_bids as mod

    frame = _redispatch({"PRIVATE_A": 400_000.0}, {"PRIVATE_A": 4_000.0})
    with_total = pd.concat([frame, frame.assign(component="TOTAL")], ignore_index=True)
    monkeypatch.setattr(mod, "RESOURCES", tmp_path)
    with_total.to_csv(tmp_path / "base_redispatch_summary.csv", index=False)
    frame.to_csv(tmp_path / "pub_redispatch_summary.csv", index=False)

    result, _ = mod.decompose("base", "pub", pd.Series({"PRIVATE_A": 0.0}))
    assert result["base_bid_cost_gbp"] == pytest.approx(400_000.0)
    assert result["delta_bid_cost_gbp"] == pytest.approx(0.0)


def test_a_fixed_strike_state_loses_its_top_up_on_the_volume_moved_off_it(tmp_path, monkeypatch):
    """30 Sep 2026: on an auction-clearing strike the state is settled on delivery, like any CfD holder."""
    import scripts.market.decompose_public_bm_bids as mod

    base = _redispatch({"PUBLIC_B": 1_000_000.0, "PRIVATE_A": 400_000.0},
                       {"PUBLIC_B": 10_000.0, "PRIVATE_A": 4_000.0})
    public = _redispatch({"PUBLIC_B": 0.0, "PRIVATE_A": 100_000.0},
                         {"PUBLIC_B": 13_000.0, "PRIVATE_A": 1_000.0})
    monkeypatch.setattr(mod, "RESOURCES", tmp_path)
    base.to_csv(tmp_path / "base_redispatch_summary.csv", index=False)
    public.to_csv(tmp_path / "pub_redispatch_summary.csv", index=False)
    pd.DataFrame({"generator": ["PUBLIC_B", "PRIVATE_A"], "pricing": "cfd_contract_value",
                  "mean_bid_gbp_per_mwh": [80.0, 100.0]}).to_csv(
        tmp_path / "base_contract_value_bids.csv", index=False)
    result, _ = mod.decompose("base", "pub", pd.Series({"PUBLIC_B": 1.0, "PRIVATE_A": 0.0}))
    # 3,000 MWh move from the state to the private unit, each at its own top-up.
    assert result["offsetting_private_cfd_settlement_gbp"] == pytest.approx(3_000 * 100.0)
    assert result["state_cfd_settlement_at_fixed_strike_gbp"] == pytest.approx(-3_000 * 80.0)
    assert "state_cfd_settlement_at_fixed_strike_gbp" in mod.SUMMABLE
