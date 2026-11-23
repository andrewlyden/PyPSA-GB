"""The R1 strike cut divides by the energy the bill pays, not by all dispatch (negative pricing rule).

Under ``gb_contract_terms`` the CfD line pays no difference in AR4-and-later
hours with a negative reference price, and none inside AR1-AR3 six-hour negative
runs. A strike cut of ``delta`` then moves the bill by ``delta`` x the *paid*
energy, so ``delta = saving / all dispatch`` hands the bill less than the saving
and the route identity (<= 1 GBP) fails by the unpaid share. These tests settle
a synthetic year with negative hours through the bill's own calculator.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.market.cfd_strike_table import load_strike_table
from scripts.market.revenue_tracking import compute_cfd_payments
from scripts.public_ownership import build_cfd_strike_adjustments as builder
from scripts.public_ownership.cfd_repricing import (
    ROUTE_CFD_BILL, build_strike_adjustments, case_strike_table, limit_at_cost_to_deliverable, per_row_savings,
    repricing_settings,
)
from tests.unit.test_public_ownership_cfd_repricing import _synthetic_register
from config.config_loader import load_defaults

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
STRIKE_TABLE = Path("data/market/lccc/cfd_strike_prices_by_round.csv")
RULE = "gb_contract_terms"
HOURS = 24
#: Six negative hours in a row (AR1-AR3 lose them too) and one isolated negative
#: hour (only AR4 and later lose it); the rest clear at 40.
REFERENCE = pd.Series([-5.0] * 6 + [40.0] * 4 + [-1.0] + [40.0] * 13,
                      index=pd.date_range("2030-01-01", periods=HOURS, freq="h"))
#: Curtailed pre-AR4 energy the economic-curtailment rule pays at strike minus a zero price.
CURTAILED_MWH = {"AR3_offshore": 1000.0}


@pytest.fixture(scope="module")
def table():
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    return load_strike_table(STRIKE_TABLE, assumptions["cfd_strike_indexation"], 2024)


@pytest.fixture(scope="module")
def settings():
    block = deepcopy(load_defaults()["public_ownership"]["consumer_bill"])
    block["cfd_repricing"]["amortisation"] = "economic_life"
    return repricing_settings(block)


@pytest.fixture(scope="module")
def savings(settings):
    register, _ = _synthetic_register()
    return per_row_savings(register, settings)


def _network(savings):
    """The private CfD fleet, each unit's register dispatch spread evenly over the day."""
    private = savings.loc[savings["case_id"].eq("private") & savings["support_type"].str.lower().eq("cfd")
                          & savings["network_support_type"].str.lower().eq("cfd")]
    generators = pd.DataFrame({
        "support_type": "CfD", "carrier": private["carrier"].to_numpy(), "bus": "bus",
        "cfd_round": private["cfd_round"].to_numpy(), "cfd_support_start": pd.NaT,
    }, index=private["asset_id"].to_numpy())
    dispatch = pd.DataFrame(np.repeat(private["dispatch_mwh"].to_numpy()[None, :] / HOURS, HOURS, axis=0),
                            index=REFERENCE.index, columns=generators.index)
    return SimpleNamespace(generators=generators, snapshots=REFERENCE.index,
                           generators_t=SimpleNamespace(p=dispatch),
                           snapshot_weightings=pd.DataFrame({"generators": 1.0}, index=REFERENCE.index))


def _bill_cfd_line(network, strikes) -> float:
    """The bill's CfD line on the synthetic year: settled dispatch plus the curtailment payment."""
    details = compute_cfd_payments(network, strikes, reference_price=REFERENCE, negative_price_rule=RULE)
    curtailment = sum(mwh * max(strikes.strikes[tuple(key.split("_", 1))] - 0.0, 0.0)
                      for key, mwh in CURTAILED_MWH.items())
    return float(details["net_cfd_cost"].sum()) + curtailment


@pytest.fixture()
def fake_settlement(monkeypatch, savings):
    """Stand the projector's ``aggregate_inputs`` in with the same calculator on the synthetic year."""
    from scripts.consumer_bill import project_future_bill
    from scripts.consumer_bill.project_future_bill import summarise_strike_provenance

    network = _network(savings)
    calls = []

    def aggregate_inputs(scenarios, networks, prices, summaries, year, physical_networks, strike_prices,
                         domestic_profile, negative_price_rule, annual_aggregate, **kwargs):
        calls.append({"rule": negative_price_rule, **kwargs})
        details = compute_cfd_payments(network, strike_prices, reference_price=REFERENCE,
                                       negative_price_rule=negative_price_rule,
                                       volume_caps=kwargs.get("volume_caps"))
        curtailment = [{"strike_key": key, "curtailed_mwh": mwh,
                        "payment_gbp": mwh * strike_prices.strikes[tuple(key.split("_", 1))]}
                       for key, mwh in CURTAILED_MWH.items()]
        return None, {"cfd_strike_provenance": summarise_strike_provenance(details, strike_prices, []),
                      "cfd_economic_curtailment_by_strike": curtailment}

    monkeypatch.setattr(project_future_bill, "aggregate_inputs", aggregate_inputs)
    return network, calls


def _paid_energy(table):
    case = {"source_scenarios": ["m1"], "modelled_year": 2030}
    assumptions = {"model_linked_rules": {"cfd": {"negative_price_rule": RULE, "volume_collars": {}}}}
    return builder.bill_paid_energy_by_key(case, assumptions, ["w.nc"], ["p.csv"], ["s.json"], ["b.nc"], table)


def test_paid_energy_is_the_dispatch_in_the_hours_the_round_is_paid(table, savings, fake_settlement):
    _, calls = fake_settlement
    paid = _paid_energy(table)
    assert len(calls) == 2 and {c["rule"] for c in calls} == {RULE}
    private = savings.loc[savings["case_id"].eq("private")].set_index("asset_id")["dispatch_mwh"]
    # AR7 (post-AR4): every negative hour unpaid, 7 of 24.
    ar7 = private[["ar7_a", "ar7_b"]].sum()
    assert paid["AR7_offshore"] == pytest.approx(ar7 * 17 / 24, rel=1e-12)
    # AR3: only the six-hour run is unpaid; plus the curtailed energy the bill pays.
    assert paid["AR3_offshore"] == pytest.approx(private["ar3_old"] * 18 / 24 + 1000.0, rel=1e-12)
    # The modelled AR8 round keeps the AR4 terms.
    future = private[["solar_future::legacy", "solar_future::new_build"]].sum()
    assert paid["AR8_modelled_solar"] == pytest.approx(future * 17 / 24, rel=1e-12)


def test_strike_cut_on_paid_energy_moves_the_bill_by_exactly_the_r1_saving(table, settings, savings,
                                                                          fake_settlement):
    network, _ = fake_settlement
    paid = _paid_energy(table)
    adjustments = build_strike_adjustments(savings, table, settings, paid_energy_mwh_by_key=paid)
    private_line = _bill_cfd_line(network, table)
    for case_id in ("public_coordination", "public_ownership"):
        r1 = savings.loc[savings["case_id"].eq(case_id) & savings["route"].eq(ROUTE_CFD_BILL),
                         "saving_bill_term_gbp"].sum()
        assert r1 > 0
        bill_change = private_line - _bill_cfd_line(network, case_strike_table(table, adjustments, case_id))
        assert bill_change == pytest.approx(r1, abs=1.0), case_id
        rows = adjustments.loc[adjustments["case_id"].eq(case_id)].set_index("strike_key")
        assert rows.loc["AR7_offshore", "eligible_dispatch_mwh"] == pytest.approx(paid["AR7_offshore"])

    # The old denominator (all dispatch) misses by the unpaid share: 7/24 of the AR7 and future cut.
    stale = build_strike_adjustments(savings, table, settings)
    r1 = savings.loc[savings["case_id"].eq("public_ownership") & savings["route"].eq(ROUTE_CFD_BILL),
                     "saving_bill_term_gbp"].sum()
    stale_change = private_line - _bill_cfd_line(network, case_strike_table(table, stale, "public_ownership"))
    assert stale_change == pytest.approx(r1 * 17 / 24, rel=1e-9)
    assert r1 - stale_change > 1.0


def test_at_cost_limit_reads_the_paid_energy(table, savings):
    """A contract whose rent per *paid* MWh exceeds its strike is put back on the financing basis."""
    frame = savings.copy()
    at_cost = frame["case_id"].eq("public_ownership") & frame["route"].eq(ROUTE_CFD_BILL) & frame["cfd_round"].eq("AR7")
    frame["at_cost_applied"] = at_cost
    frame["saving_financing_term_gbp"] = frame["saving_bill_term_gbp"]
    strike = table.strikes[("AR7", "offshore")]
    dispatch = float(frame.loc[frame["case_id"].eq("private") & frame["cfd_round"].eq("AR7"), "dispatch_mwh"].sum())
    # A rent of 0.9 x strike per dispatched MWh is deliverable on all dispatch...
    rent = 0.9 * strike * dispatch
    frame.loc[at_cost, "saving_bill_term_gbp"] = rent * frame.loc[at_cost, "dispatch_mwh"] / frame.loc[at_cost, "dispatch_mwh"].sum()
    _, limited = limit_at_cost_to_deliverable(frame, table)
    assert limited.empty
    # ...but not on the 17/24 of it the bill pays.
    _, limited = limit_at_cost_to_deliverable(frame, table, paid_energy_mwh_by_key={"AR7_offshore": dispatch * 17 / 24})
    assert list(limited["strike_key"]) == ["AR7_offshore"]
    assert limited["settled_mwh"].iloc[0] == pytest.approx(dispatch * 17 / 24)


def test_writer_refuses_the_dispatch_denominator_under_a_negative_pricing_rule(tmp_path, settings):
    register, summary = _synthetic_register()
    register_path, summary_path = tmp_path / "register.csv", tmp_path / "summary.csv"
    register.to_csv(register_path, index=False)
    summary.to_csv(summary_path, index=False)
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    assumptions["model_linked_rules"]["cfd"]["negative_price_rule"] = RULE
    assumptions_path = tmp_path / "assumptions.json"
    assumptions_path.write_text(json.dumps(assumptions), encoding="utf-8")
    block = load_defaults()["public_ownership"]["consumer_bill"]
    with pytest.raises(ValueError, match="negative_price_rule"):
        builder.write_strike_adjustments(
            register_path, summary_path, assumptions_path, block, model_price_year=2024,
            adjustments_output=tmp_path / "a.csv", routes_output=tmp_path / "r.csv", strike_table_path=STRIKE_TABLE)


def test_the_paid_energy_settles_on_the_stage_1_link_schedules(table, fake_settlement):
    """Item D1: the electrolysers' Stage 1 draw reaches the settlement, given or found beside each network."""
    _, calls = fake_settlement
    case = {"source_scenarios": ["m1"], "modelled_year": 2030}
    assumptions = {"model_linked_rules": {"cfd": {"negative_price_rule": RULE, "volume_collars": {}}}}
    builder.bill_paid_energy_by_key(case, assumptions, ["w.nc"], ["p.csv"], ["s.json"], ["b.nc"], table,
                                    links=["parent_wholesale_links.csv"])
    assert calls[-1]["wholesale_links"] == ["parent_wholesale_links.csv"]
    builder.bill_paid_energy_by_key(case, assumptions, ["M_wholesale.nc"], ["p.csv"], ["s.json"], ["b.nc"], table)
    assert calls[-1]["wholesale_links"] == ["M_wholesale_links.csv"]
    with pytest.raises(ValueError, match="align with the scenarios"):
        builder.bill_paid_energy_by_key(case, assumptions, ["w.nc"], ["p.csv"], ["s.json"], ["b.nc"], table,
                                        links=["a.csv", "b.csv"])
