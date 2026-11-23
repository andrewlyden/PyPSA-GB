"""Regression checks for year routing, monetary recovery and report accounting."""

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill.bill_scope import bill_scope
from scripts.consumer_bill.extract_pypsa_bill_signals import _price_series, _roc_price, _roc_price_series
from scripts.consumer_bill.extract_pypsa_bill_signals import extract_signals
from scripts.consumer_bill.build_recovery_bridges import _per_customer, _share, build_bridges
from scripts.consumer_bill.review_consumer_bill import build_review, render_review
from scripts.market.build_household_bill_component_register import build_consumer_bill_register
from scripts.market.build_consumer_bill_forecast_registry import _classification
from scripts.market.ofgem_price_cap import calculate_tariffs, calculate_bills, require_full_calendar_year, demand_weighted_price


def fixture_bill(year=2030):
    rates = pd.DataFrame([dict(payment_method="direct_debit", region="Example",
        nil_gbp_per_year_ex_vat=90, benchmark_gbp_per_year_ex_vat=500, benchmark_kwh=3100)])
    bills = calculate_bills(calculate_tariffs(rates, regions=["Example"], payment_methods=["direct_debit"]), [2500])
    source = pd.DataFrame([
        ("hedged_wholesale", "Direct energy", 220),
        ("hedged_wholesale", "Published CfD allowance embedded in direct fuel", -20),
        ("network", "Ofgem Annex 3 — DUoS fixed charges", 30),
        ("network", "Ofgem Annex 3 — TNUoS", 70),
        ("policy", "Ofgem Annex 4 — WHD", 10),
        ("supplier_operating", "Supplier operating costs", 190),
    ], columns=["stack_group", "component", "annual_gbp_per_3100kwh_ex_vat"])
    scope = bill_scope({"modelled_year": year}, {"effective_from": f"{year}-01-01", "effective_to": f"{year}-12-31", "currency": "GBP nominal"})
    for key, value in scope.items():
        bills[key] = value
    register, _ = build_consumer_bill_register(source, bills)
    return bills, register, {**scope, "snapshot_id": f"{year}_example"}


def test_future_review_preserves_categories_sources_and_negative_cfd():
    bills, register, bundle = fixture_bill()
    rows, summary = build_review(bills, register, bundle)
    cfd = rows.loc[rows.component.str.contains("CfD")].iloc[0]
    assert cfd["source_stack_group"] == "hedged_wholesale"
    assert cfd["group"] == "Policy costs"
    assert cfd["annual_gbp_at_selected_consumption"] == pytest.approx(-20 * 2500 / 3100)
    assert rows.annual_gbp_at_selected_consumption.sum() == pytest.approx(bills.consumer_bill_gbp_inc_vat.iloc[0])
    assert summary["forecast_validated"] is False
    assert "Projected bundle assumption" in render_review(rows, summary)
    assert "2020" not in render_review(rows, summary)


def test_known_fixed_components_do_not_scale_with_consumption():
    _, register, _ = fixture_bill()
    fixed = register.loc[register.component.str.contains("fixed charges|WHD")]
    assert fixed.annual_gbp_ex_vat.sum() == pytest.approx(40)
    assert fixed.unit_charge_at_benchmark_gbp_ex_vat.sum() == 0


def test_aggregate_network_cost_is_allocated_between_standing_and_unit_rates():
    rates = pd.DataFrame([dict(payment_method="direct_debit", region="Example",
        nil_gbp_per_year_ex_vat=150, benchmark_gbp_per_year_ex_vat=500, benchmark_kwh=3100)])
    bills = calculate_bills(calculate_tariffs(rates, regions=["Example"], payment_methods=["direct_debit"]), [3100])
    source = pd.DataFrame([("hedged_wholesale", "Energy", 200), ("network", "Ofgem Annex 3 — Network costs", 250),
                           ("supplier_operating", "Operating", 50)],
                          columns=["stack_group", "component", "annual_gbp_per_3100kwh_ex_vat"])
    rows, _ = build_consumer_bill_register(source, bills)
    assert rows.annual_gbp_ex_vat.sum() == pytest.approx(525)
    network = rows.loc[rows.stack_group.eq("network")].iloc[0]
    assert network.charge_basis == "mixed_allocated"
    assert network.standing_charge_component_gbp_ex_vat == pytest.approx(125)


def test_roc_price_uses_year_and_requires_missing_future_assumption():
    config = {"support": {"dispatch_costs": {"roc_buyout_prices": {2020: 50, "2024": 64}}}}
    assert _roc_price(config, 2024, True) == 64
    with pytest.raises(ValueError, match="2030"):
        _roc_price(config, 2030, True)


def test_reference_price_gaps_are_not_filled(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"snapshot": ["2024-01-01 00:00"], "wholesale_price": [20]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="every PyPSA snapshot"):
        _price_series(str(path), pd.date_range("2024-01-01", periods=2, freq="h"))


def test_missing_prices_cannot_be_silently_dropped_from_weighted_average():
    with pytest.raises(ValueError, match="complete, finite"):
        demand_weighted_price(pd.Series([20., np.nan]), pd.Series([1., 1.]))


def test_zero_domestic_share_is_zero_and_cfd_paybacks_stay_negative():
    assert _per_customer(100, 10, _share({"domestic_recovery_share": 0})) == 0
    assert _per_customer(-100, 10) == -31
    with pytest.raises(ValueError, match="between zero and one"):
        _per_customer(100, 10, 1.1)


def test_aggregate_network_bundle_remains_explicit_without_invented_split():
    register = pd.DataFrame({"component": ["Published CfD allowance embedded in direct fuel", "Ofgem Annex 4 — RO", "Ofgem Annex 3 — Network costs", "Capacity Market"],
                             "annual_gbp_per_3100kwh_ex_vat": [20, 40, 200, 10]})
    rows, summary = build_bridges(register, pd.DataFrame({"modelled_demand_mwh": [100]}), {}, {})
    assert rows.loc[rows.component.str.contains("embedded"), "historical_gbp_per_3100kwh_ex_vat"].iloc[0] == 20
    missing = rows.loc[rows.evidence_status.eq("baseline_component_not_disaggregated")]
    assert len(missing) == 4
    assert missing.candidate_gbp_per_3100kwh_ex_vat.isna().all()
    assert summary["undisaggregated_baseline_components"]


def test_fresh_dispatch_signals_take_precedence_over_archived_revenue():
    register = pd.DataFrame({"component": ["Published CfD allowance embedded in direct fuel", "Ofgem Annex 4 — RO", "Capacity Market"],
                             "annual_gbp_per_3100kwh_ex_vat": [20, 40, 10]})
    signals = pd.DataFrame({"metric": ["cfd_payment_proxy_gbp", "roc_obligation_proxy_gbp"], "value": [100., 200.]})
    rows, _ = build_bridges(register, pd.DataFrame({"modelled_demand_mwh": [100]}),
                           {"revenue": {"cfd_total_cost": 9999, "roc_total_income": 9999}}, {}, signals)
    assert rows.loc[rows.bridge.eq("CfD levy recovery"), "modelled_signal_value"].iloc[0] == 100
    assert rows.loc[rows.bridge.eq("Renewables Obligation recovery"), "modelled_signal_value"].iloc[0] == 200


@pytest.mark.parametrize("year", [2020, 2024, 2030])
def test_annual_snapshot_coverage_including_leap_year(year):
    index = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="30min", inclusive="left")
    require_full_calendar_year(index, pd.Series(.5, index=index))
    weights = pd.Series(.5, index=index.delete(1))
    weights.iloc[-1] += .5
    with pytest.raises(ValueError, match="gaps or overlaps"):
        require_full_calendar_year(weights.index, weights)


def test_annual_historical_benchmark_is_not_statutory_cap():
    _, _, bundle = fixture_bill(2020)
    assert bundle["is_statutory_cap_reproduction"] is False
    assert bundle["bill_basis"] == "annualised_historical_benchmark"


def test_ro_is_classified_as_operational_recovery():
    assert _classification("Ofgem Annex 4 — RO", "policy")["forecast_driver"] == "operational_model"


def test_model_signals_use_scenario_prices_and_retain_midperiod_cfd_details():
    import pypsa
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=2, freq="h"))
    n.add("Bus", "GB")
    n.add("Load", "load", bus="GB", p_set=10)
    n.add("Generator", "wind", bus="GB", carrier="wind_offshore", p_nom=10)
    n.add("Generator", "roc", bus="GB", carrier="wind_onshore", p_nom=10)
    n.generators["support_type"] = ["CfD", "ROC"]
    n.generators["cfd_round"] = ["AR1", ""]
    n.generators["cfd_support_start"] = ["2024-01-01 01:00", None]
    n.generators["ro_banding"] = [0.0, 1.0]
    n.generators_t.p = pd.DataFrame({"wind": [4., 4.], "roc": [6., 6.]}, index=n.snapshots)
    config = {"modelled_year": 2024,
              "support": {"labels": {"cfd_strike_prices": {"AR1_offshore": 50}},
                          "dispatch_costs": {"roc_buyout_prices": {2020: 1, 2023: 60, 2024: 65}}}}
    signals, summary, details = extract_signals(n, pd.Series(70., index=n.snapshots), config, {})
    values = signals.set_index("metric").value
    assert values["model_demand_mwh"] == 20
    assert values["roc_obligation_proxy_gbp"] == 720
    assert values["cfd_payment_proxy_gbp"] == -80
    assert details.has_strike_price.all()
    assert details.physical_dispatch_MWh.iloc[0] == 8
    assert details.total_dispatch_MWh.iloc[0] == 4
    assert summary["modelled_year"] == 2024


def test_roc_dispatch_cost_changes_at_april_obligation_boundary():
    import pypsa
    from scripts.market.revenue_tracking import compute_roc_income
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-03-31 23:00", periods=2, freq="h"))
    n.add("Bus", "GB")
    n.add("Generator", "wind", bus="GB", p_nom=10, carrier="wind_onshore")
    n.generators["support_type"] = "ROC"
    n.generators["ro_banding"] = 1.0
    n.generators_t.p = pd.DataFrame({"wind": [10., 20.]}, index=n.snapshots)
    config = {"support": {"dispatch_costs": {"roc_buyout_prices": {2023: 50, 2024: 60}}}}
    prices = _roc_price_series(config, n.snapshots, True)
    assert prices.tolist() == [50, 60]
    assert compute_roc_income(n, prices).roc_income_total.iloc[0] == 1700
