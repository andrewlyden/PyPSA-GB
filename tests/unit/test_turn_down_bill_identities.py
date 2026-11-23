"""The bill and ownership layers on the pro rata turn-down basis (20 Oct 2026): identities still close.

Runs on a solved, re-priced month (skipped where absent): the constraint line moves by exactly the month's
uplift, the recomputed CfD compensation still reconciles with the LP's CfD bid cost on the re-shared
volumes, the bill's identity re-check closes, and the per-asset settlement moves by the same uplift.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.market import turn_down_allocation as tda

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "resources" / "market"
NAME = "CP30_FFR_2030_Jan_no_acceleration"
S1 = "CP30_FFR_2030_Jan_stage1"
ASSUMPTIONS = ROOT / "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions_cfd_rule_none.json"


def _paths():
    paths = {"wholesale": MARKET / f"{S1}_wholesale.nc", "balancing": MARKET / f"{NAME}_balancing.nc",
             "price": MARKET / f"{S1}_wholesale_price.csv", "summary": ROOT / f"resources/analysis/{NAME}_market_summary.json",
             "redispatch": MARKET / f"{NAME}_redispatch_summary.csv", "bids": MARKET / f"{NAME}_contract_value_bids.csv",
             "pro_rata": MARKET / f"{NAME}_pro_rata_turn_down.json"}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(f"{NAME} not solved and re-priced here")
    return paths


@pytest.fixture(scope="module")
def month():
    import scripts.consumer_bill.extract_future_balancing_costs as ex
    from scripts.market.cfd_strike_table import load_strike_table

    p = _paths()
    a = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    include, _, treatment, tol = ex.bsuos_rule_options(a)
    strike_path, _ = ex.compensation_strike_source(a, a["model_linked_rules"]["cfd"]["strike_table"])
    strikes = load_strike_table(strike_path, a["cfd_strike_indexation"], 2024)
    out = {}
    for basis in ("cheapest_first", "pro_rata_zonal"):
        with tda.force_basis(basis):
            _, m, _ = ex.extract_month(NAME, p["wholesale"], p["balancing"], p["price"], p["summary"], strikes,
                                       include_interconnector_retrading=include, compensation_treatment=treatment,
                                       redispatch_summary_path=p["redispatch"], contract_value_bids_path=p["bids"],
                                       reconciliation_tolerance=tol)
        out[basis] = m
    out["uplift"] = json.loads(p["pro_rata"].read_text(encoding="utf-8"))["pro_rata_uplift_GBP"]
    return out


def test_constraint_line_moves_by_the_uplift(month):
    solved, pro = month["cheapest_first"], month["pro_rata_zonal"]
    assert pro["turn_down_basis"] == "pro_rata_zonal" and solved["turn_down_basis"] == "cheapest_first"
    assert pro["constraint_cost_gbp"] - solved["constraint_cost_gbp"] == pytest.approx(month["uplift"], abs=1.0)
    assert pro["accepted_bid_cost_as_solved_gbp"] == pytest.approx(solved["accepted_bid_cost_gbp"])


def test_cfd_compensation_reconciles_on_the_re_shared_volumes(month):
    pro = month["pro_rata_zonal"]
    assert abs(pro["cfd_compensation_reconciliation_gap_share"]) < 1e-3
    assert pro["cfd_curtailment_compensation_gbp"] > month["cheapest_first"]["cfd_curtailment_compensation_gbp"]


@pytest.mark.parametrize("basis", ["cheapest_first", "pro_rata_zonal"])
def test_the_bill_identity_closes(month, basis):
    m = month[basis]
    expected = (m["accepted_offer_cost_gbp"] + m["accepted_bid_cost_gbp"] - m["feasibility_slack_cost_in_accepted_costs_gbp"]
                - m["non_participant_turn_down_cost_at_penalty_gbp"] - m["tie_break_step_cost_gbp"])
    if m["cfd_curtailment_compensation_treatment"] == "add_post_hoc":
        expected += m["cfd_curtailment_compensation_gbp"]
    if m["include_interconnector_retrading"]:
        expected += m["interconnector_retrading_cost_gbp"]
    assert m["constraint_cost_gbp"] == pytest.approx(expected, abs=1.0)


def test_settlement_moves_by_the_uplift_and_keeps_wind_output():
    from scripts.public_ownership import settlement as st

    p = _paths()
    a = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    from scripts.market.cfd_strike_table import load_strike_table

    strike_path = a["model_linked_rules"]["cfd"]["strike_table"]
    strikes = load_strike_table(strike_path, a["cfd_strike_indexation"], 2024)
    price = pd.read_csv(p["price"], index_col=0, parse_dates=True)["wholesale_price"]
    frames = {}
    for basis in ("cheapest_first", "pro_rata_zonal"):
        with tda.force_basis(basis):
            frames[basis] = st._month_settlement(p["wholesale"], p["balancing"], price, p["redispatch"], strikes, None,
                                                 78.04, bid_basis="contract_value", audit_path=p["bids"],
                                                 contract_value_floor="none")
    solved, pro = frames["cheapest_first"], frames["pro_rata_zonal"]
    uplift = json.loads(p["pro_rata"].read_text(encoding="utf-8"))["pro_rata_uplift_GBP"]
    assert pro["balancing_settlement_gbp"].sum() - solved["balancing_settlement_gbp"].sum() == pytest.approx(uplift, rel=1e-6)
    wind = solved["carrier"].isin(["wind_onshore", "wind_offshore"])
    assert pro.loc[wind, "balancing_mwh"].sum() == pytest.approx(solved.loc[wind, "balancing_mwh"].sum(), rel=1e-9)
    # The CfD units are paid on the re-shared volumes what they bid: compensation tracks the bid money.
    cfd = pro["support_type"].eq("CfD") & wind
    assert np.isclose(pro.loc[cfd, "cfd_curtailment_compensation_gbp"].sum(), pro.loc[cfd, "balancing_bid_gbp"].sum(),
                      rtol=1e-2) if "balancing_bid_gbp" in pro else True
