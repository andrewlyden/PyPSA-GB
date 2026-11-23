"""FES 2050 plan, task Y14: the balancing merge never picks a P2 share's twin by order."""

import pytest

from scripts.public_ownership.merge_balancing_groups import market_twin

BASE = {"new_build_generation": "public", "balancing_bids": "contract_value"}


def test_a_gated_lever_in_the_summaries_picks_the_share_twin():
    market = {"cases": {
        "public_ownership": {"levers": dict(BASE)},
        "public_ownership@deployment_share_25": {"levers": dict(BASE, state_deployment_share="0.25")},
        "public_ownership@deployment_share_50": {"levers": dict(BASE, state_deployment_share="0.50")},
    }}
    levers = dict(BASE, balancing_bids="state_at_cost", state_deployment_share="0.50")
    assert market_twin("public_ownership@deployment_share_50_state_bids", levers, market) == \
        "public_ownership@deployment_share_50"


def test_summaries_without_the_gated_lever_are_refused_not_resolved_by_order():
    market = {"cases": {"public_ownership": {"levers": dict(BASE)},
                        "public_ownership@deployment_share_25": {"levers": dict(BASE)}}}
    with pytest.raises(ValueError, match="market twins"):
        market_twin("public_ownership@deployment_share_25_state_bids", dict(BASE, balancing_bids="state_at_cost"),
                    market)


def test_one_twin_and_no_twin_behave_as_before():
    market = {"cases": {"public_ownership": {"levers": dict(BASE)}}}
    assert market_twin("public_ownership@state_bids_at_cost", dict(BASE, balancing_bids="state_at_cost"),
                       market) == "public_ownership"
    assert market_twin("public_ownership@state_bids_at_cost", None, market) == "public_ownership"

