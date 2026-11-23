"""Contract-aware curtailment in the wholesale and each round's negative pricing term (26 Sep 2026).

Every CfD and merchant renewable reached the wholesale at a marginal cost of 0, so
which of ~9,000 tied units gave way in an oversupplied hour was the solver's
choice (and it loaded small units), and it decided how much CfD top-up was paid.
Now each unit backs off where its support terms say it would.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.market.curtailment_order import apply_curtailment_order
from scripts.market.revenue_tracking import negative_price_eligibility
from config.config_loader import load_defaults

LOG = logging.getLogger("test")


# ── negative pricing rule ────────────────────────────────────────────────────

def test_ar4_and_later_lose_payment_in_any_strictly_negative_hour():
    ref = np.array([5.0, 0.0, -1.0, 3.0])
    eligible = negative_price_eligibility(ref, ["AR4", "AR7a", "AR8_modelled"], "gb_contract_terms")
    # A zero price still pays (the term is "negative"), a negative one does not.
    assert eligible[:, 0].tolist() == [True, True, False, True]
    assert (eligible[:, 1] == eligible[:, 0]).all() and (eligible[:, 2] == eligible[:, 0]).all()


def test_solver_noise_around_zero_is_not_a_negative_price():
    # Hours the merchant fleet sets at zero come back as -3e-5 from a barrier solve, and the
    # CP30 Stage 1 clears its surplus at the AR4+ bid of -0.01 (plus noise): none of these is a
    # negative price. The band is 0.50 (12 Oct 2026); genuinely negative hours sit at -1 or below.
    ref = np.array([-3e-5, -0.009, -0.02, -0.6])
    assert negative_price_eligibility(ref, ["AR5"], "gb_contract_terms")[:, 0].tolist() == [True, True, True, False]


def test_ar1_to_ar3_lose_payment_only_inside_a_six_hour_negative_run():
    ref = np.array([-1.0] * 5 + [2.0] + [-1.0] * 6 + [4.0])
    eligible = negative_price_eligibility(ref, ["AR2"], "gb_contract_terms")[:, 0]
    assert eligible[:5].all()                # five negative hours: still paid
    assert not eligible[6:12].any()          # six in a row: not paid
    assert eligible[5] and eligible[12]


def test_ar1_investment_contracts_and_hpc_carry_no_negative_pricing_term():
    # The rule applies to contracts offered from 1 January 2016; AR1 was offered in February 2015.
    ref = np.full(8, -50.0)
    assert negative_price_eligibility(ref, ["AR1", "Investment Contract", "HPC"], "gb_contract_terms").all()


def test_the_legacy_rules_are_unchanged():
    ref = np.array([0.0, 1.0])
    assert negative_price_eligibility(ref, ["AR1"], "none").all()
    assert negative_price_eligibility(ref, ["AR1"], "ar4_zero_payment_when_price_le_0")[:, 0].tolist() == [False, True]
    with pytest.raises(ValueError, match="negative_price_rule"):
        negative_price_eligibility(ref, ["AR4"], "made_up")


# ── wholesale curtailment order ──────────────────────────────────────────────

class _Strikes:
    def lookup(self, carrier, cfd_round):
        table = {("wind_offshore", "AR3"): 56.7, ("nuclear", "HPC"): 128.9}
        strike = table.get((carrier, cfd_round))
        return (strike, f"{cfd_round}_{carrier}", False) if strike else (None, None, True)


def _network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "GB")
    for name, carrier, support, rnd, mc in [
        ("merchant", "wind_onshore", "merchant", "", 0.0),
        ("ar6", "wind_offshore", "CfD", "AR6", 0.0),
        ("AR8_modelled", "solar_pv", "CfD", "AR8_modelled", 0.0),
        ("ar3", "wind_offshore", "CfD", "AR3", 0.0),
        ("hpc", "nuclear", "CfD", "HPC", 10.0),
        ("roc", "wind_onshore", "ROC", "", -80.0),
        ("drax", "biomass", "CfD", "DRAX_BRIDGE", 95.0),
    ]:
        n.add("Generator", name, bus="GB", p_nom=100.0, carrier=carrier, marginal_cost=mc)
        n.generators.loc[name, "support_type"] = support
        n.generators.loc[name, "cfd_round"] = rnd
    return n


def test_each_unit_backs_off_where_its_terms_say():
    # Decision D1 (27 Sep 2026): merchant +0.5 gives way first, AR4+ at 0 (its
    # reservation price: paid whenever the price is not negative) next.
    n = _network()
    audit = apply_curtailment_order(n, {"mode": "contract_aware"}, _Strikes(), LOG)
    mc = n.generators["marginal_cost"]
    assert mc["merchant"] == pytest.approx(0.5)               # small variable cost: gives way first
    assert mc["ar6"] == 0.0 and mc["AR8_modelled"] == 0.0           # AR4+: next, the price stays at zero
    assert mc["ar3"] == pytest.approx(-56.7) and mc["hpc"] == pytest.approx(-128.9)   # paid through negatives
    assert mc["roc"] == -80.0 and mc["drax"] == 95.0          # ROC and thermal CfDs untouched
    assert set(audit["generator"]) == {"merchant", "ar6", "AR8_modelled", "ar3", "hpc"}


def test_the_pre_d1_treatment_is_kept_as_a_sensitivity():
    n = _network()
    audit = apply_curtailment_order(
        n, {"mode": "contract_aware", "ar4_plus_bid_gbp_per_mwh": -1.0, "merchant_variable_cost_gbp_per_mwh": 0.0},
        _Strikes(), LOG)
    mc = n.generators["marginal_cost"]
    assert mc["merchant"] == 0.0 and mc["ar6"] == -1.0 and mc["AR8_modelled"] == -1.0
    assert set(audit["generator"]) == {"ar6", "AR8_modelled", "ar3", "hpc"}


def test_merchant_adder_only_touches_listed_renewable_carriers_and_keeps_varying_costs():
    n = _network()
    n.add("Generator", "merchant_nuke", bus="GB", p_nom=100.0, carrier="nuclear", marginal_cost=10.0)
    n.generators.loc["merchant_nuke", "support_type"] = "merchant"
    n.add("Generator", "tv_solar", bus="GB", p_nom=10.0, carrier="solar_pv", marginal_cost=0.0)
    n.generators.loc["tv_solar", "support_type"] = "merchant"
    n.generators_t.marginal_cost = pd.DataFrame({"tv_solar": [0.0, 1.0]}, index=n.snapshots)
    apply_curtailment_order(n, {"mode": "contract_aware"}, _Strikes(), LOG)
    assert n.generators.at["merchant_nuke", "marginal_cost"] == 10.0
    assert n.generators_t.marginal_cost["tv_solar"].tolist() == [0.5, 1.5]


@pytest.mark.parametrize("block", [{"ar4_plus_bid_gbp_per_mwh": 0.5}, {"merchant_variable_cost_gbp_per_mwh": -0.1}])
def test_out_of_range_order_parameters_are_refused(block):
    with pytest.raises(ValueError):
        apply_curtailment_order(_network(), {"mode": "contract_aware", **block}, _Strikes(), LOG)


def test_code_defaults_match_defaults_yaml():
    from pathlib import Path

    import yaml
    from scripts.market import curtailment_order as co

    block = load_defaults()["market"]["wholesale"]["renewable_curtailment_order"]
    assert block["ar4_plus_bid_gbp_per_mwh"] == co.DEFAULT_AR4_PLUS_BID
    assert block["merchant_variable_cost_gbp_per_mwh"] == co.DEFAULT_MERCHANT_VARIABLE_COST
    assert tuple(block["merchant_carriers"]) == co.DEFAULT_MERCHANT_CARRIERS


def test_central_dispatch_and_as_built_are_left_alone():
    for block, copperplate in (({"mode": "as_built"}, "collapse"), ({"mode": "contract_aware"}, "none")):
        n = _network()
        assert apply_curtailment_order(n, block, _Strikes(), LOG, copperplate_mode=copperplate) is None
        assert n.generators.at["ar6", "marginal_cost"] == 0.0


def test_the_general_turn_up_floor_covers_every_renewable_and_nothing_else():
    # The contract-aware order moved the free swap onto merchant plant (69 -> 161 GWh on the test week).
    from scripts.market.cfd_bm_bids import resolve_general_turn_up_floor, resolve_turn_up_floor

    carriers = pd.Series({"MERCH_WIND": "wind_onshore", "CFD_SOLAR": "solar_pv", "CCGT": "CCGT", "TIDE": "marine"})
    floor = resolve_general_turn_up_floor({"turn_up_floor_gbp_per_mwh": 2.0}, carriers, LOG)
    assert floor.to_dict() == {"MERCH_WIND": 2.0, "CFD_SOLAR": 2.0, "TIDE": 2.0}
    assert resolve_general_turn_up_floor({}, carriers, LOG) is None
    with pytest.raises(ValueError, match="at most"):
        resolve_general_turn_up_floor({"turn_up_floor_gbp_per_mwh": 9.0}, carriers, LOG)
    # Combined with P3's ownership floor, each unit takes the larger.
    p3 = resolve_turn_up_floor({"enabled": True, "turn_up_floor_gbp_per_mwh": 3.0},
                               pd.Series({"CFD_SOLAR": 0.5}), carriers, LOG)
    combined = pd.concat([floor, p3], axis=1).max(axis=1)
    assert combined["CFD_SOLAR"] == 2.0 and combined["MERCH_WIND"] == 2.0


def test_a_cfd_unit_without_a_strike_is_refused():
    n = _network()
    n.generators.loc["ar3", "cfd_round"] = "AR2"
    with pytest.raises(ValueError, match="no strike"):
        apply_curtailment_order(n, {"mode": "contract_aware"}, _Strikes(), LOG)
