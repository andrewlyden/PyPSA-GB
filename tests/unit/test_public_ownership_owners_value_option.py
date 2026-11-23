"""F27 item 1 (user, 20 Oct 2026): pre-2025 CfD plant bought at its owners' value, beside the at-cost option.

The state pays the owners their cost plus the present value, at their own rate, of the rent they
earn above their required return over the contract's remaining years; the premium is annuitised
at the state's rate on its share and is a payment to the former owners, not a resource cost.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import calculate_financing_scenarios as cfs
from scripts.public_ownership import cfd_repricing as cr
from scripts.public_ownership import party_flows as pf
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.levers import (
    DEFAULT_SETTINGS, PRE_2025_DPA_CONTRACTS, PRE_2025_PUBLIC_OWNERS_VALUE, PRE_2025_PUBLIC_REPRICED, is_pre_2025_dpa,
)
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    LINE_DPA_REPRICED, NEW_LINES, dpa_household, dpa_system, extra_lines,
)
from tests.unit.test_public_ownership_levers import ROW

PRIVATE_RATE = ROW["wacc"]["private"]   # workbook v2.0's Sc1 offshore rate (F29)


def _rows(n=3):
    return SimpleNamespace(technology=np.array(["Offshore wind"] * n), pre_policy_cfd=np.array([True, True, False]),
                           commissioning_year=np.array([2026.0, np.nan, 2027.0]))


def _terms(value, rent=(10.0e6, 4.0e6, 99.0e6)):
    return SimpleNamespace(settings=dict(DEFAULT_SETTINGS, new_build_generation="public", pre_2025_cfd_capacity=value),
                           rates={"Offshore wind": ROW}, comparator={"rent": np.array(rent)},
                           tax_basis={"model_year": 2030})


def test_the_premium_is_the_owners_rent_valued_at_their_rate_and_financed_at_the_states():
    idx, own = np.arange(3), np.ones(3, dtype=bool)
    vintage = np.array(["new_build"] * 3)
    share, wacc, remaining = np.ones(3), np.full(3, 0.03), np.full(3, 28.0)
    out = cfs._owners_value_premium(_rows(), _terms(PRE_2025_PUBLIC_OWNERS_VALUE), idx, own, vintage, share, wacc,
                                    remaining, finance_snapshot={})
    annuity = lambda years: (1 - (1 + PRIVATE_RATE) ** -years) / PRIVATE_RATE  # noqa: E731
    crf = capital_recovery_factor(0.03, 28.0)
    # Started 2026: 11 of its 15 contract years left in 2030; undated: the whole term; row 3 is post-2025 (no premium).
    assert out[0] == pytest.approx(10.0e6 * annuity(11.0) * crf)
    assert out[1] == pytest.approx(4.0e6 * annuity(15.0) * crf)
    assert out[2] == 0.0


def test_other_lever_values_pay_no_premium_and_the_option_needs_the_comparator():
    args = (np.arange(3), np.ones(3, dtype=bool), np.array(["new_build"] * 3), np.ones(3), np.full(3, 0.03),
            np.full(3, 28.0))
    assert not cfs._owners_value_premium(_rows(), _terms(PRE_2025_PUBLIC_REPRICED), *args, finance_snapshot={}).any()
    terms = _terms(PRE_2025_PUBLIC_OWNERS_VALUE)
    terms.comparator = None
    with pytest.raises(ValueError, match="needs the comparator"):
        cfs._owners_value_premium(_rows(), terms, *args, finance_snapshot={})


# ── Net Zero Teesside joins the option (user, 24 Oct 2026) ──────────────────────────────────

NZT = {"support_type": "DPA", "cfd_round": "NZT_POWER_DPA", "model_year": 2030, "finance_vintage": "new_build",
       "carrier": "CCS Gas", "component_type": "Generator", "contract_vintage": "not_applicable",
       "evidence_status": "resolved"}


def test_only_the_signed_2024_dpa_is_a_pre_2025_contract_and_only_while_live():
    assert PRE_2025_DPA_CONTRACTS == {"NZT_POWER_DPA": 2028}
    assert is_pre_2025_dpa("DPA", "NZT_POWER_DPA", 2030)
    assert not is_pre_2025_dpa("DPA", "NZT_POWER_DPA", 2050)       # 2050 does not move
    assert is_pre_2025_dpa("DPA", "NZT_POWER_DPA", 2040)           # live 2028-2043 (2040, 11 Nov 2026)
    assert not is_pre_2025_dpa("DPA", "NZT_POWER_DPA", 2043)
    assert not is_pre_2025_dpa("DPA", "DPA_2031", 2030)            # a modelled later agreement
    assert not is_pre_2025_dpa("CfD", "NZT_POWER_DPA", 2030)


@pytest.mark.parametrize("settings, owned", [
    (dict(new_build_generation="public", pre_2025_cfd_capacity="private"), False),
    (dict(new_build_generation="public", pre_2025_cfd_capacity=PRE_2025_PUBLIC_REPRICED), True),
    (dict(new_build_generation="public", pre_2025_cfd_capacity=PRE_2025_PUBLIC_OWNERS_VALUE), True),
    (dict(new_build_generation="state_stake", pre_2025_cfd_capacity=PRE_2025_PUBLIC_OWNERS_VALUE), True),
    # P3 (and every acquiring case) keeps it on its DPA: the tariff pays the kept remaining_ccs line.
    (dict(new_build_generation="public", pre_2025_cfd_capacity=PRE_2025_PUBLIC_REPRICED,
          legacy_fleet="acquire_all"), False),
])
def test_nzt_follows_the_pre_2025_lever_only_where_the_bill_is_the_market_bill(settings, owned):
    assert cfs._state_owned(pd.Series(NZT), dict(DEFAULT_SETTINGS, **settings), None, 0) is owned
    # A 2050 row on a DPA is ordinary new build again (the rule never reaches a vintaged year).
    later = pd.Series(dict(NZT, model_year=2050))
    assert cfs._state_owned(later, dict(DEFAULT_SETTINGS, new_build_generation="public"), None, 0)


def test_the_nzt_premium_runs_over_the_dpa_years_left_from_its_2028_start():
    rows = SimpleNamespace(technology=np.array(["Offshore wind"] * 2), pre_policy_cfd=np.array([True, True]),
                           commissioning_year=np.array([np.nan, np.nan]),
                           pre_2025_dpa=np.array([True, False]), pre_2025_dpa_start=np.array([2028.0, np.nan]))
    terms = _terms(PRE_2025_PUBLIC_OWNERS_VALUE, rent=(300.0e6, 4.0e6))
    out = cfs._owners_value_premium(rows, terms, np.arange(2), np.ones(2, dtype=bool), np.array(["new_build"] * 2),
                                    np.ones(2), np.full(2, 0.03), np.full(2, 25.0), finance_snapshot={})
    annuity = lambda years: (1 - (1 + PRIVATE_RATE) ** -years) / PRIVATE_RATE  # noqa: E731
    crf = capital_recovery_factor(0.03, 25.0)
    assert out[0] == pytest.approx(300.0e6 * annuity(13.0) * crf)     # 2028 + 15 - 2030
    assert out[1] == pytest.approx(4.0e6 * annuity(15.0) * crf)       # an undated CfD: the whole term


def _nzt_register(value, *, basis="at_cost", case_rr=150.0e6, support=403.0e6, market=245.0e6, fuel=145.0e6,
                  owned=True):
    """The NZT row in private and one owning case (system GBP), at the settlement the register books."""
    common = dict(NZT, asset_id="Net Zero Teesside Power::new_build", dispatch_mwh=1.94e6, wacc=0.0824,
                  remaining_life_years=25.0, market_revenue_gbp=market, support_revenue_gbp=support,
                  fuel_carbon_gbp=fuel, annual_state_overhead_gbp=0.0, cfd_technology=None, cfd_award_year=np.nan)
    private = dict(common, case_id="private", annual_required_revenue_gbp=204.0e6,
                   annual_capital_recovery_gbp=157.0e6, ownership_share=0.0, state_owned=False,
                   cfd_strike_basis="financing_saving_only")
    case = dict(common, case_id="public_ownership@nzt", annual_required_revenue_gbp=case_rr,
                annual_capital_recovery_gbp=case_rr - 47.0e6, wacc=0.035, ownership_share=1.0 if owned else 0.0,
                state_owned=owned, new_build_generation="public", pre_2025_cfd_capacity=value,
                cfd_strike_basis=basis, cfd_amortisation="economic_life")
    return pd.DataFrame([private, case])


def _settings():
    return cr.repricing_settings({"cfd_repricing": {"contract_term_years": 20}})


def test_owned_nzt_is_repriced_at_cost_on_its_own_route():
    savings = cr.per_row_savings(_nzt_register(PRE_2025_PUBLIC_OWNERS_VALUE), _settings())
    row = savings.set_index("case_id").loc["public_ownership@nzt"]
    assert row.route == cr.ROUTE_DPA_BILL and bool(row.at_cost_applied) and not bool(row.r1_eligible)
    # The DPA falls by contract revenue (market + DPA - fuel and carbon) less the state's cost.
    assert row.saving_bill_term_gbp == pytest.approx(245.0e6 + 403.0e6 - 145.0e6 - 150.0e6)
    assert row.contract_term_years == 15.0
    # Private keeps it on its DPA: the dispatchable-contract route (never merchant), carrying nothing.
    assert savings.set_index("case_id").loc["private"].route == cr.ROUTE_DISPATCHABLE_CFD
    summary = cr.route_summary(savings, pd.DataFrame({"case_id": ["private", "public_ownership@nzt"],
                                                      "annual_financing_saving_vs_private_gbp": [0.0, 54.0e6]}))
    dpa = summary.set_index(["case_id", "route"]).loc[("public_ownership@nzt", cr.ROUTE_DPA_BILL)]
    assert dpa.bill_allocated_gbp_2024 == pytest.approx(353.0e6) and dpa.treatment == "bill_allocated"


def test_the_dpa_cut_stops_at_the_dpa_payment_and_auction_clearing_keeps_the_contract():
    capped = cr.per_row_savings(_nzt_register(PRE_2025_PUBLIC_REPRICED, support=100.0e6, market=600.0e6,
                                              fuel=100.0e6), _settings()).set_index("case_id")
    assert capped.loc["public_ownership@nzt"].saving_bill_term_gbp == pytest.approx(100.0e6)
    kept = cr.per_row_savings(_nzt_register(PRE_2025_PUBLIC_REPRICED, basis="auction_clearing"),
                              _settings()).set_index("case_id")
    assert kept.loc["public_ownership@nzt"].route == cr.ROUTE_CFD_PUBLIC_ACCOUNT
    unowned = cr.per_row_savings(_nzt_register("private", owned=False), _settings()).set_index("case_id")
    assert unowned.loc["public_ownership@nzt"].route == cr.ROUTE_DISPATCHABLE_CFD


def test_the_bill_row_carries_the_account_dpa_cut_on_the_remaining_ccs_recovery():
    assert LINE_DPA_REPRICED in NEW_LINES
    rows = extra_lines("public_ownership@nzt", {"dpa_bill_cut_gbp": 100.0e6}, factor=1.0, final_demand_mwh=250.0e6,
                       eligible_mwh=200.0e6, mwh=3.1, account_source="account")
    row = next(r for r in rows if r["component_id"] == LINE_DPA_REPRICED)
    assert row["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-100.0e6 / 200.0e6 * 3.1)
    kwargs = dict(line_recovery=None, eligible_mwh=200.0e6, factor=1.0, mwh=3.1)
    assert dpa_system(dpa_household(123.0e6, **kwargs), **kwargs) == pytest.approx(123.0e6)
    # The private case adds no row.
    assert not [r for r in extra_lines("private", {"dpa_bill_cut_gbp": 100.0e6}, factor=1.0, final_demand_mwh=1.0,
                                       eligible_mwh=1.0, mwh=3.1, account_source="a")
                if r["component_id"] == LINE_DPA_REPRICED]


def test_the_party_flows_take_the_dpa_line_as_part_of_the_r1_cut():
    flows = pf.Flows("public_ownership@nzt", lambda gbp: gbp / 1e7)
    components = {k: 0.0 for k in ("state_capital", "private_capital", "state_operating", "private_operating",
                                   "corporation_tax", "ldes_state", "ldes_private", "carbon_cfd_private",
                                   "tax_equivalent_private")}
    account = {"r1_bill_cut_gbp": 353.0e6, "r1_cut_borne_by_state_gbp": 353.0e6, "dpa_bill_cut_gbp": 353.0e6}
    parties = {"bill_lines_gbp": {"cfd": 0.0, pf.DPA_BILL_LINE: -353.0e6}}
    pf.market_flows(flows, parties=parties, components=components, account_case=account, network={}, levers={})
    cut = [r for r in flows.rows if r["flow_id"] == "strike_cut_state"]
    assert len(cut) == 1 and cut[0]["gbp"] == pytest.approx(353.0e6) and cut[0]["payee"] == "consumers"
    assert not [r for r in flows.rows if r["flow_id"] == pf.DPA_BILL_LINE]     # carried once, by the cut


def test_the_premium_adjustment_moves_the_coowners_part_onto_the_state():
    adj = cfs.owners_value_premium_state_adjustment(np.array([60.0e6, 60.0e6, 0.0]), np.array([0.3, 1.0, 0.3]))
    assert adj == pytest.approx([-0.7 * 60.0e6, 0.0, 0.0])


def _p1_register(share, premium, *, with_adjustment=True):
    base = dict(asset_id="ar5_farm", annual_state_overhead_gbp=0.0, annual_tax_equivalent_gbp=0.0,
                carbon_removal_cfd_revenue_gbp=0.0, ldes_floor_topup_paid_gbp=0.0, corporation_tax_forgone_gbp=0.0)
    case = dict(base, case_id="public_control@p1", ownership_share=share, annual_capital_recovery_gbp=140.0e6 + premium,
                annual_required_revenue_gbp=190.0e6 + premium, producer_rent_gbp=250.0e6,
                owners_value_premium_gbp=premium)
    if with_adjustment:
        case[cfs.OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT] = -(1.0 - share) * premium
    return pd.DataFrame([
        dict(base, case_id="private", ownership_share=0.0, annual_capital_recovery_gbp=150.0e6,
             annual_required_revenue_gbp=200.0e6, producer_rent_gbp=300.0e6, owners_value_premium_gbp=0.0),
        case])


def _exchequer(flows):
    return sum(r["gbp"] for r in flows.rows if r["payee"] == "exchequer") - sum(
        r["gbp"] for r in flows.rows if r["payer"] == "exchequer")


@pytest.mark.parametrize("at_cost", [False, True])
def test_on_a_p1_stake_the_state_books_the_whole_premium_and_the_flows_close(at_cost):
    """User, 27 Oct 2026: the premium is booked on the state's share only; the co-owners carry none of it."""
    share, premium = 0.3, 60.0e6
    adj = -(1 - share) * premium
    components = pf.register_components(_p1_register(share, premium), "public_control@p1")
    assert components["owners_value_premium"] == pytest.approx(premium)
    assert components["owners_value_premium_coowners"] == pytest.approx(0.0)
    if at_cost:
        # At cost the cut is sized on the state's share alone, on its own cost (the whole premium).
        # (its share of the rent with the co-owners' part of the premium moved onto it: it keeps nothing).
        cut = share * 250.0e6 + adj
        borne = cut
    else:
        # Financing-saving basis: the whole row's cut, the state's part less the co-owners' part of the premium.
        cut = 200.0e6 - (190.0e6 + premium)
        borne = share * cut + adj
    account = {"r1_bill_cut_gbp": cut, "r1_cut_borne_by_state_gbp": borne}
    flows = pf.Flows("public_control@p1", lambda gbp: gbp / 1e7)
    pf.market_flows(flows, parties={"bill_lines_gbp": {"cfd": -cut}}, components=components, account_case=account,
                    network={}, levers={})
    # The account's view: the state's share of the case's rent plus the adjustment, less the cut it bears.
    assert _exchequer(flows) == pytest.approx(share * 250.0e6 + adj - borne)
    if not at_cost:
        # On the financing basis the Exchequer's position is the same as before the fix (the premium it
        # now pays in full is matched by the part of the cut the co-owners take on).
        old = pf.Flows("public_control@p1", lambda gbp: gbp / 1e7)
        old_components = pf.register_components(_p1_register(share, premium, with_adjustment=False), "public_control@p1")
        pf.market_flows(old, parties={"bill_lines_gbp": {"cfd": -cut}}, components=old_components,
                        account_case={"r1_bill_cut_gbp": cut, "r1_cut_borne_by_state_gbp": share * cut},
                        network={}, levers={})
        assert _exchequer(flows) == pytest.approx(_exchequer(old))


def test_at_cost_the_state_share_reprices_on_a_cost_carrying_the_whole_premium():
    share, premium = 0.3, 6.0e6
    frame = _nzt_register(PRE_2025_PUBLIC_OWNERS_VALUE, case_rr=150.0e6)
    owned = frame["case_id"].ne("private")
    frame.loc[owned, "ownership_share"] = share
    plain = cr.per_row_savings(frame.assign(owners_value_premium_gbp=np.where(owned, premium, 0.0)),
                               _settings()).set_index("case_id").loc["public_ownership@nzt"]
    adjusted = cr.per_row_savings(frame.assign(**{
        "owners_value_premium_gbp": np.where(owned, premium, 0.0),
        cfs.OWNERS_VALUE_PREMIUM_STATE_ADJUSTMENT: np.where(owned, -(1 - share) * premium, 0.0)}),
        _settings()).set_index("case_id").loc["public_ownership@nzt"]
    assert adjusted.saving_at_cost_gbp == pytest.approx(plain.saving_at_cost_gbp - (1 - share) * premium)


def test_on_a_p1_stake_the_exchequer_flows_carry_only_the_states_share_of_the_premium():
    """Live bill refresh, 24 Oct 2026, on a register without the 27 Oct adjustment column: the pro rata split."""
    share, premium = 0.3, 60.0e6
    base = dict(asset_id="ar5_farm", annual_state_overhead_gbp=0.0, annual_tax_equivalent_gbp=0.0,
                carbon_removal_cfd_revenue_gbp=0.0, ldes_floor_topup_paid_gbp=0.0, corporation_tax_forgone_gbp=0.0)
    register = pd.DataFrame([
        dict(base, case_id="private", ownership_share=0.0, annual_capital_recovery_gbp=150.0e6,
             annual_required_revenue_gbp=200.0e6, producer_rent_gbp=300.0e6, owners_value_premium_gbp=0.0),
        # The stake refinances the row (capital 140m) and buys at owners' value: the premium is in its capital.
        dict(base, case_id="public_control@p1", ownership_share=share, annual_capital_recovery_gbp=140.0e6 + premium,
             annual_required_revenue_gbp=190.0e6 + premium, producer_rent_gbp=250.0e6, owners_value_premium_gbp=premium),
    ])
    components = pf.register_components(register, "public_control@p1")
    assert components["owners_value_premium"] == pytest.approx(share * premium)
    assert components["owners_value_premium_coowners"] == pytest.approx((1 - share) * premium)
    # P1's financing-saving basis: the bill takes the whole row's required-revenue change, split pro rata.
    cut = 200.0e6 - (190.0e6 + premium)
    borne = share * cut
    account = {"r1_bill_cut_gbp": cut, "r1_cut_borne_by_state_gbp": borne}
    flows = pf.Flows("public_control@p1", lambda gbp: gbp / 1e7)
    pf.market_flows(flows, parties={"bill_lines_gbp": {"cfd": -cut}}, components=components, account_case=account,
                    network={}, levers={})
    exchequer = sum(r["gbp"] for r in flows.rows if r["payee"] == "exchequer") - sum(
        r["gbp"] for r in flows.rows if r["payer"] == "exchequer")
    # The account's view: the state's share of the case's rent, less the part of the cut it bears.
    assert exchequer == pytest.approx(share * 250.0e6 - borne)
