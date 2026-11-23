import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill.project_future_bill import (
    project_components, aggregate_inputs, read_hourly_inputs, read_cfd_payments,
    load_domestic_profile, domestic_weights, read_annual_aggregate,
    read_recovery_denominators, read_balancing_costs,
)
from config.config_loader import load_defaults
from tests.helpers import requires_outputs


ASSUMPTIONS_PATH = "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json"
STRIKE_TABLE_PATH = "data/market/lccc/cfd_strike_prices_by_round.csv"
PROFILE_PATH = "data/market/consumer_bill_forecast/domestic_demand_profile_2020.csv"


def fixture_inputs():
    """A neutral fixture: unit currency factor, zero externals, zero bridge rates.

    Only the rows under test carry value, so an assertion about one component
    cannot be satisfied by another moving in the opposite direction.
    """
    assumptions = json.loads(Path(ASSUMPTIONS_PATH).read_text())
    assumptions["currency_rebasing"].update(factor=1, from_index=100, to_index=100)
    assumptions["benchmark"]["vat_rate"] = .10
    bridge = assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]
    bridge["allowances"]["losses"]["loss_multiplier_gb_average"] = 1.0
    for key in ("transaction_cost", "day_ahead_rehedging", "imbalance", "additional_risk_and_basis"):
        bridge["allowances"][key]["rate"] = 0.0
    bridge["forward_premium"]["rate"] = 0.0
    # The neutral fixture prices energy on the model's own profile; the Ofgem
    # cap-rule shaping (28 Sep 2026) is exercised in test_bill_review_2026_09_28.py.
    assumptions["model_linked_rules"]["direct_fuel"]["shaping"]["method"] = "model_profile"
    # The LDES line (item D4(c)) needs a public account; its own tests switch it on. The fixture's
    # recovery volumes are on the item D1 basis, so the line is spread over final demand (F18).
    assumptions["model_linked_rules"]["ldes_cap_and_floor"]["enabled"] = False
    assumptions["model_linked_rules"]["ldes_cap_and_floor"]["recovery_denominator"] = "final_demand_mwh"
    assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"] = {
        key: 0 for key in assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"]
    }
    # Every sourced external is neutralised (values must stay positive), so the
    # fixture cannot be satisfied by a primary-source line moving instead.
    for block in assumptions["sourced_externals"].values():
        block["value"] = 1e-12
        # A line derived over the recovery file (item F4) would take its value from
        # its cost and the fixture's volumes, and a declared nil (item F1) must stay
        # zero, so both are turned into plain quoted lines here.
        (block.get("derivation") or {}).pop("recovery_denominator", None)
        # A register-booked line (cost_from, the battery top-up) reads its alternatives
        # from the live public account; the neutral fixture quotes it plainly instead.
        (block.get("derivation") or {}).pop("sensitivity_cost_from_keys", None)
        # A declared cost's alternatives (the 2030 BSUoS services lines' ancillary low and high, 6 Oct 2026)
        # need the denominator popped above; a plainly quoted line has none.
        (block.get("derivation") or {}).pop("sensitivity_cost_gbp", None)
        # F18: the supplier line's levelisation and EBIT/headroom recompute read its real components;
        # the neutral fixture quotes the line plainly (tests/unit/test_line_recovery.py covers them).
        (block.get("derivation") or {}).pop("household_losses", None)
        (block.get("derivation") or {}).pop("direct_debit_levelisation", None)
        (block.get("derivation") or {}).pop("ebit_and_headroom", None)
        block.pop("value_basis", None)
        # A calendar blend (28 Sep 2026) is valued from its fiscal-year components.
        for component in block.get("period_blend") or []:
            component["value"] = 1e-12
    # Item G6's calibration needs the real line values; a neutral fixture has none,
    # so the archetypes are declared by the tests that exercise them.
    assumptions.pop("charge_basis", None)
    assumptions.pop("household_archetypes", None)
    case = {"case_id": "test", "modelled_year": 2030, "model_price_year": 2024,
            "period_alignment_note": "Test calendar-year assumptions",
            "cfd_recovery": {"method": "modelled_system_demand_proxy"}}
    annual = {"modelled_year": 2030, "reported_period_hours": 8760, "demand_mwh": 100,
              "wholesale_price_gbp_per_mwh": 10, "domestic_profile_price_gbp_per_mwh": 10,
              "domestic_shaping_gbp_per_mwh": 0.0,
              "domestic_profile": {"peak_share_of_consumption": .55, "peak_share_of_hours": .36},
              "cfd_payment_gbp": -200, "cfd_negative_price_rule": "none",
              "gross_redispatch_mwh": 0.0,
              # No interconnector rent: the cap-and-floor credit is exercised in its own tests.
              "interconnector_congestion_rent_by_cable_gbp": {}}
    balancing = {"constraint_cost_gbp": 0.0, "accepted_offer_cost_gbp": 0.0,
                 "accepted_bid_cost_gbp": 0.0, "cfd_curtailment_compensation_gbp": 0.0,
                 "cfd_curtailed_mwh": 0.0, "feasibility_slack_cost_gbp": 0.0,
                 "feasibility_slack_energy_mwh": 0.0, "path": "test-balancing.json"}
    # No sourced eligible-sales figure by default, so the system-demand proxy is
    # still permitted. A case whose recovery file supplies one may not use it.
    recovery = {"final_demand_mwh": 100, "cfd_eligible_supplier_sales_mwh": None,
                "cfd_eligible_share": 0.5, "system_to_final_demand_factor": 1.0}
    # The RO row is exercised in tests/unit/test_phase2_support_labels.py; here
    # it is neutral so it cannot mask another component's movement.
    ro = {"relief_case": "relief_ends_march_2029", "exchequer_relief_share": 0.0,
          "gbp_per_3100kwh_ex_vat": 0.0, "gross_gbp_per_3100kwh_ex_vat": 1e-12,
          "relief_cases": {"relief_ends_march_2029": 0.0, "relief_extended": 0.0},
          "reference_gbp_per_3100kwh_ex_vat": 17.0, "roc_volume": 0.0,
          "eligible_capacity_mw": 0.0, "expired_capacity_mw": 0.0,
          "notional_roc_value_gbp_per_roc": 75.95, "obligation_year": "2030/31",
          "method_label": "pypsa_informed_fleet_register", "report_price_year": 2025,
          "path": "test-ro.json"}
    return annual, assumptions, case, balancing, recovery, ro


def as_placeholder(assumptions, key, value):
    """Turn a sourced line back into an analyst placeholder, for the tests of that path.

    Item F1 sourced the last five placeholders, so the repository file has none;
    the placeholder machinery is still exercised, on a line the test demotes.
    """
    assumptions["sourced_externals"].pop(key, None)
    assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"][key] = value
    return assumptions


def test_future_bill_uses_modelled_cfd_credit_once_and_recalculates_vat():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    assert rows.loc[rows.component_id.eq("cfd"), "annual_gbp_per_3100kwh_ex_vat"].iloc[0] == -6.2
    assert summary["ex_vat_gbp"] == pytest.approx(24.8)
    assert summary["vat_gbp"] == pytest.approx(2.48)
    assert summary["total_gbp_inc_vat"] == pytest.approx(27.28)
    assert len(rows.loc[rows.component_id.eq("cfd")]) == 1
    assert {"allocated_ccs", "remaining_ccs", "eco"}.issubset(rows.component_id)
    assert summary["forecast_validated"] is False


def test_eligible_sales_change_only_cfd_recovery():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    case["cfd_recovery"] = {"method": "eligible_supplier_sales"}
    recovery["cfd_eligible_supplier_sales_mwh"] = 50
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    assert rows.loc[rows.component_id.eq("cfd"), "annual_gbp_per_3100kwh_ex_vat"].iloc[0] == -12.4
    assert rows.loc[rows.component_id.eq("direct_energy"), "annual_gbp_per_3100kwh_ex_vat"].iloc[0] == 31
    assert summary["cfd_recovery_denominator_mwh"] == 50


@pytest.mark.parametrize("change, message", [("hours", "complete calendar"), ("price_year", "price basis"), ("period", "maps"), ("unknown", "schema mismatch")])
def test_future_projection_rejects_mixed_scopes(change, message):
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    if change == "hours": annual["reported_period_hours"] = 4344
    if change == "price_year": case["model_price_year"] = 2020
    if change == "period":
        case.pop("period_alignment_note")
        assumptions["benchmark"].pop("period_alignment_note", None)
    if change == "unknown": assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"]["unmapped levy"] = 10
    with pytest.raises(ValueError, match=message):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_hourly_reader_includes_static_loads_and_demand_weights(tmp_path):
    import xarray as xr
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    ds = xr.Dataset({"snapshots_snapshot": ("snapshots", snapshots), "snapshots_generators": ("snapshots", [1., 1.]),
                     "loads_p_set": ("loads_i", [99., 5.]),
                     "loads_t_p_set": (("snapshots", "loads_t_p_set_i"), [[5.], [15.]])},
                    coords={"loads_i": ["dynamic", "static"], "loads_t_p_set_i": ["dynamic"]})
    path = tmp_path / "model.nc"
    ds.to_netcdf(path)
    price = tmp_path / "price.csv"
    pd.DataFrame({"wholesale_price": [10., 40.]}, index=snapshots).to_csv(price)
    inputs = read_hourly_inputs(path, price)
    assert inputs.demand_mw.tolist() == [10, 20]
    assert (inputs.price_gbp_per_mwh * inputs.demand_mw).sum() / inputs.demand_mw.sum() == 30


def test_physical_cfd_reader_recomputes_paybacks_and_reports_missing_dispatch(tmp_path):
    import xarray as xr
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    ds = xr.Dataset({
        "snapshots_snapshot": ("snapshots", snapshots), "snapshots_generators": ("snapshots", [1., 1.]),
        "generators_support_type": ("generators_i", ["CfD", "CfD"]),
        "generators_carrier": ("generators_i", ["wind_offshore", "wind_offshore"]),
        "generators_bus": ("generators_i", ["GB", "GB"]),
        "generators_cfd_round": ("generators_i", ["AR8_modelled", "AR8_modelled"]),
        "generators_p_nom": ("generators_i", [20., 5.]),
        "generators_t_p": (("snapshots", "generators_t_p_i"), [[10.], [20.]]),
    }, coords={"generators_i": ["wind", "missing"], "generators_t_p_i": ["wind"]})
    path = tmp_path / "physical.nc"
    ds.to_netcdf(path)
    payment, priced, physical, missing, capacity, strikes = read_cfd_payments(path, pd.Series([50., 70.], index=snapshots), {"AR8_modelled_offshore": 60})
    assert payment == -100
    assert priced == physical == 30
    assert missing == 1
    assert capacity == 5
    assert strikes["contract_metadata_absent"] == ["cfd_support_start"]
    # These generators declare cfd_round "AR8_modelled", so "AR8_modelled_offshore" is an
    # explicit round match. A fallback is only recorded when the round is
    # missing or unmatched and the carrier-wide strike has to be substituted.
    assert strikes["fallback_strike_share_of_payment"] == 0.0


def test_annual_aggregation_uses_hourly_demand_weighting_not_monthly_mean(monkeypatch, tmp_path):
    import scripts.consumer_bill.project_future_bill as module
    index = pd.date_range("2030-01-01", "2031-01-01", freq="h", inclusive="left")
    demand = pd.Series(1., index=index)
    demand.iloc[len(index)//2:] = 3
    prices = pd.Series(10., index=index)
    prices.iloc[len(index)//2:] = 50
    frame = pd.DataFrame({"demand_mw": demand, "hours": 1., "price_gbp_per_mwh": prices})
    monkeypatch.setattr(module, "read_hourly_inputs", lambda *args: frame)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"scenario": "test", "prices": {"mean_wholesale_price": 30},
        "accounting": {"reported_period_hours": 8760, "demand_MWh": demand.sum(), "gross_redispatch_MWh": 0},
        "revenue": {"cfd_reference_price_source": "wholesale_system_price", "cfd_total_cost": 100, "cfd_total_dispatch_MWh": 20}}))
    network, price_path = tmp_path / "model", tmp_path / "prices"
    network.write_text("mock model")
    price_path.write_text("mock prices")
    _, annual = aggregate_inputs(["test"], [network], [price_path], [path], 2030)
    assert annual["wholesale_price_gbp_per_mwh"] == 40


def test_declared_dead_model_rule_is_rejected():
    """A rule declared in the assumptions file must be one the projector applies.

    The file previously described a redispatch-indexed BSUoS allowance that no
    code read, so the stated method and the published bill disagreed.
    """
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["model_linked_rules"]["bsuos"] = {"formula": "indexed to modelled redispatch"}
    with pytest.raises(ValueError, match="model_linked_rules schema mismatch"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_source_allowances_on_another_price_year_need_an_explicit_bridge():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["source"]["price_year"] = 2022
    with pytest.raises(ValueError, match="add source_currency_rebasing"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_source_rebasing_is_applied_to_every_external():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["source"]["price_year"] = 2022
    assumptions["source_currency_rebasing"] = {
        "from_price_year": 2022, "to_price_year": 2025,
        "from_index": 100.0, "to_index": 110.0, "factor": 1.1,
    }
    as_placeholder(assumptions, "remaining_ccs", 100.0)
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    assert value["remaining_ccs"] == pytest.approx(110.0)
    assert summary["source_rebasing_factor"] == pytest.approx(1.1)


def test_source_rebasing_factor_must_reconcile_to_its_indices():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["source"]["price_year"] = 2022
    assumptions["source_currency_rebasing"] = {
        "from_price_year": 2022, "to_price_year": 2025,
        "from_index": 100.0, "to_index": 110.0, "factor": 1.5,
    }
    with pytest.raises(ValueError, match="does not reconcile"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_benchmark_price_year_must_agree_with_the_stated_basis():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["benchmark"]["price_year"] = 2023
    with pytest.raises(ValueError, match="must agree with benchmark.price_basis"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_rebasing_must_land_on_the_benchmark_price_year():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["currency_rebasing"]["to_price_year"] = 2026
    with pytest.raises(ValueError, match="does not land on the benchmark price year"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def _strike_details(rounds):
    return pd.DataFrame({
        "generator": [f"g{i}" for i in range(len(rounds))],
        "carrier": ["wind_offshore"] * len(rounds),
        "cfd_round": rounds,
        "strike_price": [80.0] * len(rounds),
        "total_dispatch_MWh": [100.0] * len(rounds),
        "net_cfd_cost": [1000.0] * len(rounds),
    })


def test_carrier_fallback_strike_share_is_reported():
    from scripts.consumer_bill.project_future_bill import summarise_strike_provenance
    provenance = summarise_strike_provenance(
        _strike_details(["", "AR4", ""]), {"future_offshore": 80.0}, ["cfd_round"]
    )
    assert provenance["fallback_strike_share_of_energy"] == pytest.approx(2 / 3)
    assert provenance["fallback_strike_share_of_payment"] == pytest.approx(2 / 3)
    assert provenance["contract_metadata_absent"] == ["cfd_round"]


def test_round_specific_strikes_report_no_fallback():
    from scripts.consumer_bill.project_future_bill import summarise_strike_provenance
    provenance = summarise_strike_provenance(_strike_details(["AR4", "AR6"]), {}, [])
    assert provenance["fallback_strike_share_of_payment"] == pytest.approx(0.0)


def test_strike_limitation_names_the_missing_contract_metadata():
    from scripts.consumer_bill.project_future_bill import _strike_limitation
    text = _strike_limitation({
        "fallback_strike_share_of_payment": 1.0,
        "contract_metadata_absent": ["cfd_round", "cfd_support_start"],
    })
    assert "100%" in text
    assert "cfd_round or cfd_support_start" in text
    assert "2012" in text


def test_strike_limitation_is_clean_when_every_contract_matched_a_round():
    from scripts.consumer_bill.project_future_bill import _strike_limitation
    text = _strike_limitation({"fallback_strike_share_of_payment": 0.0, "contract_metadata_absent": []})
    assert "round-specific" in text


# --------------------------------------------------------------------------
# WS1: constraint costs and CfD curtailment compensation
# --------------------------------------------------------------------------

def _toy_networks(tmp_path, wholesale_mw, balancing_mw, snapshots):
    """Two CfD generators; the first is turned down in balancing, the second is not."""
    import xarray as xr
    paths = []
    for name, dispatch in (("wholesale", wholesale_mw), ("balancing", balancing_mw)):
        ds = xr.Dataset({
            "snapshots_snapshot": ("snapshots", snapshots),
            "snapshots_generators": ("snapshots", [1.0] * len(snapshots)),
            "generators_support_type": ("generators_i", ["CfD", "CfD"]),
            "generators_carrier": ("generators_i", ["wind_offshore", "wind_onshore"]),
            "generators_cfd_round": ("generators_i", ["", ""]),
            "generators_p_nom": ("generators_i", [100.0, 100.0]),
            "generators_t_p": (("snapshots", "generators_t_p_i"), dispatch),
        }, coords={"generators_i": ["curtailed", "steady"],
                   "generators_t_p_i": ["curtailed", "steady"]})
        path = tmp_path / f"{name}.nc"
        ds.to_netcdf(path)
        paths.append(path)
    return paths


def test_cfd_compensation_is_turndown_times_foregone_difference(tmp_path):
    """The acceptance test from the plan: turn-down x (strike - price) x hours."""
    from scripts.consumer_bill.extract_future_balancing_costs import (
        compute_cfd_curtailment_compensation,
    )
    from scripts.market.cfd_strike_table import StrikeTable
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing = _toy_networks(
        tmp_path, [[80.0, 50.0], [60.0, 50.0]], [[30.0, 50.0], [60.0, 50.0]], snapshots
    )
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0}, 2024, {}, {})
    prices = pd.Series([20.0, 40.0], index=snapshots)
    details, _ = compute_cfd_curtailment_compensation(wholesale, balancing, prices, table)
    curtailed = details.set_index("generator")
    # Hour 1: 80 - 30 = 50 MW turned down, foregone 100 - 20 = 80 GBP/MWh -> 4,000
    # Hour 2: no turn-down
    assert curtailed.loc["curtailed", "curtailed_mwh"] == pytest.approx(50.0)
    assert curtailed.loc["curtailed", "compensation_gbp"] == pytest.approx(50.0 * 80.0)
    assert curtailed.loc["steady", "compensation_gbp"] == pytest.approx(0.0)


def _toy_month_inputs(tmp_path, snapshots):
    """A whole month's worth of extract_month inputs around the toy networks."""
    wholesale, balancing = _toy_networks(
        tmp_path, [[80.0, 50.0], [60.0, 50.0]], [[30.0, 50.0], [60.0, 50.0]], snapshots
    )
    price_path = tmp_path / "price.csv"
    pd.DataFrame({"wholesale_price": [20.0, 40.0]}, index=snapshots).to_csv(
        price_path, index_label="snapshot")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({
        "scenario": "toy",
        "accounting": {
            "reported_period_hours": 2.0, "demand_MWh": 1000.0,
            "accepted_offer_cost_GBP": 500.0, "accepted_bid_cost_GBP": 4000.0,
            "feasibility_slack_cost_GBP": 0.0, "feasibility_slack_energy_MWh": 0.0,
            "interconnector_retrading_cost_GBP": 0.0,
        },
    }), encoding="utf-8")
    redispatch_path = tmp_path / "redispatch.csv"
    pd.DataFrame({"bid_cost": [3800.0, 0.0]}, index=["curtailed", "steady"]).to_csv(
        redispatch_path, index_label="name")
    return wholesale, balancing, price_path, summary_path, redispatch_path


def _redispatch_with_cfd_bid_cost(tmp_path, bid_cost):
    path = tmp_path / "redispatch_alt.csv"
    pd.DataFrame({"bid_cost": [bid_cost, 0.0]}, index=["curtailed", "steady"]).to_csv(
        path, index_label="name")
    return path


def _bid_audit(tmp_path, contract_value: bool):
    path = tmp_path / "contract_value_bids.csv"
    rows = ([{"generator": "curtailed", "pricing": "cfd_contract_value"}]
            if contract_value else [])
    pd.DataFrame(rows, columns=["generator", "pricing"]).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# B1: the compensation becomes a check once the LP prices CfD bids on contract
# value. Adding it on top of a contract-value bid double counts it.
# --------------------------------------------------------------------------

def test_post_hoc_treatment_adds_the_compensation_to_the_constraint_cost(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import extract_month
    from scripts.market.cfd_strike_table import StrikeTable
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing, price, summary, redispatch = _toy_month_inputs(tmp_path, snapshots)
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0},
                        2024, {}, {})
    _, month, _ = extract_month(
        "toy", wholesale, balancing, price, summary, table,
        compensation_treatment="add_post_hoc",
        redispatch_summary_path=redispatch,
        contract_value_bids_path=_bid_audit(tmp_path, contract_value=False),
    )
    assert month["cfd_curtailment_compensation_gbp"] == pytest.approx(4000.0)
    assert month["constraint_cost_gbp"] == pytest.approx(500.0 + 4000.0 + 4000.0)
    assert month["cfd_bids_priced_on_contract_value"] is False


def test_reconcile_treatment_reports_the_compensation_without_adding_it(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import extract_month
    from scripts.market.cfd_strike_table import StrikeTable
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing, price, summary, redispatch = _toy_month_inputs(tmp_path, snapshots)
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0},
                        2024, {}, {})
    _, month, _ = extract_month(
        "toy", wholesale, balancing, price, summary, table,
        compensation_treatment="reconcile_against_lp_bid",
        redispatch_summary_path=_redispatch_with_cfd_bid_cost(tmp_path, 3980.0),
        contract_value_bids_path=_bid_audit(tmp_path, contract_value=True),
    )
    # Still computed and reported...
    assert month["cfd_curtailment_compensation_gbp"] == pytest.approx(4000.0)
    # ...but the constraint cost is offers plus bids only, no double count.
    assert month["constraint_cost_gbp"] == pytest.approx(500.0 + 4000.0)
    # The LP's own CfD bid cost is what it reconciles against.
    assert month["lp_cfd_accepted_bid_cost_gbp"] == pytest.approx(3980.0)
    assert month["cfd_compensation_reconciliation_gap_gbp"] == pytest.approx(20.0)
    assert month["cfd_compensation_reconciliation_gap_share"] == pytest.approx(0.005)


@pytest.mark.parametrize("declared,contract_value", [
    ("reconcile_against_lp_bid", False),
    ("add_post_hoc", True),
])
def test_a_treatment_that_contradicts_the_solve_is_refused(tmp_path, declared, contract_value):
    """Enforced, not documented: getting this wrong is worth billions either way."""
    from scripts.consumer_bill.extract_future_balancing_costs import extract_month
    from scripts.market.cfd_strike_table import StrikeTable
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing, price, summary, redispatch = _toy_month_inputs(tmp_path, snapshots)
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0},
                        2024, {}, {})
    with pytest.raises(ValueError, match="cfd_curtailment_compensation_treatment"):
        extract_month(
            "toy", wholesale, balancing, price, summary, table,
            compensation_treatment=declared,
            redispatch_summary_path=redispatch,
            contract_value_bids_path=_bid_audit(tmp_path, contract_value=contract_value),
        )


def test_a_gap_under_the_absolute_floor_is_accepted(tmp_path):
    """13 Nov 2026: a breach must exceed GBP 1,000 as well as the share (2040 Jan P2 100%: one geothermal unit,
    GBP 3 against 5). The toy month's GBP 200 gap (5%) passes under the floor."""
    from scripts.consumer_bill import extract_future_balancing_costs as efb
    from scripts.market.cfd_strike_table import StrikeTable
    assert efb.RECONCILIATION_TOLERANCE_ABS_GBP == 1000.0
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing, price, summary, _ = _toy_month_inputs(tmp_path, snapshots)
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0},
                        2024, {}, {})
    efb.extract_month("toy", wholesale, balancing, price, summary, table,
                      compensation_treatment="reconcile_against_lp_bid",
                      redispatch_summary_path=_redispatch_with_cfd_bid_cost(tmp_path, 3800.0),
                      contract_value_bids_path=_bid_audit(tmp_path, contract_value=True))


def test_a_reconciliation_gap_beyond_tolerance_is_refused(tmp_path, monkeypatch):
    """The recompute and the LP bid cost are the same quantity by construction.

    A breach is the signature of the two sides pricing different contracts, which
    is what the untraced 4.49bn against 3.96bn discrepancy would have looked like
    had anything been checking. (The toy month's money is below the GBP 1,000 floor, so the floor is lowered.)
    """
    from scripts.consumer_bill import extract_future_balancing_costs as efb
    from scripts.consumer_bill.extract_future_balancing_costs import extract_month
    from scripts.market.cfd_strike_table import StrikeTable
    monkeypatch.setattr(efb, "RECONCILIATION_TOLERANCE_ABS_GBP", 100.0)
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing, price, summary, _ = _toy_month_inputs(tmp_path, snapshots)
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0},
                        2024, {}, {})
    with pytest.raises(ValueError, match="beyond the"):
        extract_month(
            "toy", wholesale, balancing, price, summary, table,
            compensation_treatment="reconcile_against_lp_bid",
            # 3,800 against a 4,000 recompute is a 5% gap, five times the tolerance.
            redispatch_summary_path=_redispatch_with_cfd_bid_cost(tmp_path, 3800.0),
            contract_value_bids_path=_bid_audit(tmp_path, contract_value=True),
        )


def test_the_tolerance_does_not_police_the_post_hoc_treatment(tmp_path):
    """On a scalar-bid run the LP charges nothing for CfD turn-down, so a 100%
    gap is the expected state and must not stop the extraction."""
    from scripts.consumer_bill.extract_future_balancing_costs import extract_month
    from scripts.market.cfd_strike_table import StrikeTable
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    wholesale, balancing, price, summary, _ = _toy_month_inputs(tmp_path, snapshots)
    table = StrikeTable({("AR8_modelled", "offshore"): 100.0, ("AR8_modelled", "onshore"): 100.0},
                        2024, {}, {})
    _, month, _ = extract_month(
        "toy", wholesale, balancing, price, summary, table,
        compensation_treatment="add_post_hoc",
        redispatch_summary_path=_redispatch_with_cfd_bid_cost(tmp_path, 0.0),
        contract_value_bids_path=_bid_audit(tmp_path, contract_value=False),
    )
    assert month["cfd_compensation_reconciliation_gap_share"] == pytest.approx(1.0)


def test_an_unknown_treatment_is_refused(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import (
        bsuos_rule_options, constraint_components,
    )
    with pytest.raises(ValueError, match="must be one of"):
        constraint_components("add_it_twice")
    with pytest.raises(ValueError, match="must be one of"):
        bsuos_rule_options({"model_linked_rules": {"bsuos_constraints": {
            "cfd_curtailment_compensation_treatment": "add_it_twice"}}})
    with pytest.raises(ValueError, match="must lie in"):
        bsuos_rule_options({"model_linked_rules": {"bsuos_constraints": {
            "cfd_compensation_reconciliation_tolerance": 1.5}}})


def test_the_assumptions_file_declares_a_supported_treatment():
    from scripts.consumer_bill.extract_future_balancing_costs import (
        COMPENSATION_TREATMENTS, bsuos_rule_options,
    )
    assumptions = json.loads(Path(
        "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json"
    ).read_text(encoding="utf-8"))
    _, _, treatment, tolerance = bsuos_rule_options(assumptions)
    assert treatment in COMPENSATION_TREATMENTS
    assert 0.0 < tolerance < 1.0


def test_turn_up_and_above_strike_prices_create_no_compensation_claim(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import (
        compute_cfd_curtailment_compensation,
    )
    from scripts.market.cfd_strike_table import StrikeTable
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    # Hour 1 is a turn-*up*; hour 2 is a turn-down but the price is above strike.
    wholesale, balancing = _toy_networks(
        tmp_path, [[10.0, 0.0], [80.0, 0.0]], [[40.0, 0.0], [20.0, 0.0]], snapshots
    )
    table = StrikeTable({("AR8_modelled", "offshore"): 50.0, ("AR8_modelled", "onshore"): 50.0}, 2024, {}, {})
    prices = pd.Series([20.0, 90.0], index=snapshots)
    details, _ = compute_cfd_curtailment_compensation(wholesale, balancing, prices, table)
    assert details.compensation_gbp.sum() == pytest.approx(0.0)
    # The turn-down volume is still recorded; only its value is nil.
    assert details.curtailed_mwh.sum() == pytest.approx(60.0)


def test_bsuos_constraint_row_sums_offer_bid_and_compensation():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    balancing.update(accepted_offer_cost_gbp=10.0, accepted_bid_cost_gbp=5.0,
                     cfd_curtailment_compensation_gbp=20.0, constraint_cost_gbp=35.0)
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    # 35 GBP over 100 MWh of final demand, charged on 3.1 MWh
    assert value["bsuos_constraints"] == pytest.approx(35.0 / 100 * 3.1)
    assert summary["diagnostics"]["bsuos_reference_gbp_per_3100kwh"] == 91.0


def test_withdrawn_flat_bsuos_allowance_cannot_be_reinstated_implicitly():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    with pytest.raises(ValueError, match="bsuos_reference"):
        project_components(annual, assumptions, case, None, recovery, ro)


def test_projector_refuses_to_run_without_the_sourced_non_constraint_allowance():
    """It must not infer the residual as "reference 91 minus modelled constraints"."""
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["sourced_externals"] = {}
    with pytest.raises(ValueError, match="bsuos_other_services"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_bsuos_reference_is_no_longer_summed_into_the_bill():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    rows, _ = project_components(annual, assumptions, case, balancing, recovery, ro)
    assert "bsuos_reference" not in set(rows.component_id)


# --------------------------------------------------------------------------
# WS9: primary sources for the external allowances
# --------------------------------------------------------------------------


def _primary_line(value, price_year=2025, **extra):
    block = {"value": value, "unit": "gbp_per_household_year", "consumption_basis_kwh": 3100,
             "price_year": price_year, "stack_group": "policy", "label": "Feed-in Tariffs (test)",
             "source_class": "external_primary_source", "reference_gbp_per_3100kwh_ex_vat": 26.0,
             "method": "test", "source_url": "https://example.org",
             # The repository file declares a calendar bill year, so every line states its period.
             "period_basis": "held_flat_real", "source_period": "test"}
    block.update(extra)
    return block


def test_primary_source_line_replaces_the_placeholder_and_keeps_it_as_reference():
    # Item F1 sourced FiT, WHD and ECO; a line that is still a placeholder is the example.
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    assumptions["sourced_externals"]["remaining_ccs"] = _primary_line(110.0, reference_gbp_per_3100kwh_ex_vat=4.0)
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    tnuos = rows.set_index("component_id").loc["remaining_ccs"]
    assert tnuos["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(110.0)
    assert tnuos["source_class"] == "external_primary_source"
    assert tnuos["reference_gbp_per_3100kwh_ex_vat"] == 4.0
    # The repository file already carries the three WS9 lines (neutralised by the fixture); the test line joins them.
    replacement = [entry for entry in summary["diagnostics"]["primary_source_replacements"]
                   if entry["component_id"] == "remaining_ccs"]
    assert replacement == [{"component_id": "remaining_ccs", "gbp_per_3100kwh_ex_vat": 110.0,
                            "reference_gbp_per_3100kwh_ex_vat": 4.0,
                            "delta_against_reference_gbp_per_3100kwh_ex_vat": pytest.approx(106.0),
                            "source": "https://example.org"}]
    assert "remaining_ccs" not in summary["diagnostics"]["external_placeholders_remaining"]
    assert rows.loc[rows.source_class.eq("external_assumption"), "reference_gbp_per_3100kwh_ex_vat"].isna().all()


def test_primary_source_line_is_rebased_with_its_own_index_pair():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    rebasing = {"from_price_year": 2020, "to_price_year": 2025, "index": "CPIH", "from_index": 100.0,
                "to_index": 125.0, "factor": 1.25}
    assumptions["sourced_externals"]["remaining_ccs"] = _primary_line(80.0, price_year=2020, rebasing=rebasing, label="Remaining CCS (test)")
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    assert rows.set_index("component_id").loc["remaining_ccs", "annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(100.0)
    register = {entry["source"]: entry for entry in summary["diagnostics"]["price_basis_register"]}
    assert register["assumptions.sourced_externals.remaining_ccs.rebasing"]["factor"] == 1.25
    assumptions["sourced_externals"]["remaining_ccs"]["rebasing"]["factor"] = 1.3
    with pytest.raises(ValueError, match="does not reconcile"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


# Item F1 sourced FiT, so a line that is still a placeholder is the example.
@pytest.mark.parametrize("mutate, message", [
    (lambda a: a["sourced_externals"].__setitem__("remaining_ccs", _primary_line(110.0)), "both as a primary-source line"),
    (lambda a: (a["external_assumptions_gbp_per_3100kwh_ex_vat"].pop("remaining_ccs"),
                a["sourced_externals"].__setitem__("remaining_ccs", _primary_line(110.0, source_class="external_sourced_outturn"))),
     "must be external_primary_source"),
    (lambda a: (a["external_assumptions_gbp_per_3100kwh_ex_vat"].pop("remaining_ccs"),
                a["sourced_externals"].__setitem__("remaining_ccs", _primary_line(110.0, consumption_basis_kwh=2700))),
     "kWh benchmark"),
    (lambda a: (a["external_assumptions_gbp_per_3100kwh_ex_vat"].pop("remaining_ccs"),
                a["sourced_externals"].__setitem__("remaining_ccs", _primary_line(110.0, unit="gbp_per_year"))), "gbp_per_household_year"),
    (lambda a: a["external_assumptions_gbp_per_3100kwh_ex_vat"].pop("remaining_ccs"), "schema mismatch"),
])
def test_primary_source_schema_is_enforced(mutate, message):
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    as_placeholder(assumptions, "remaining_ccs", 4.0)
    mutate(assumptions)
    with pytest.raises(ValueError, match=message):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_the_cp30_case_has_primary_sources_for_the_three_largest_external_lines():
    """WS9 items 1 to 3 (plan section 4): supplier costs, DUoS and TNUoS carry an official source."""
    assumptions = json.loads(Path(ASSUMPTIONS_PATH).read_text(encoding="utf-8"))
    sourced = assumptions["sourced_externals"]
    for key in ("supplier_costs_and_margin", "duos", "tnuos"):
        block = sourced[key]
        assert key not in assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"]
        assert block["source_class"] == "external_primary_source"
        assert block["unit"] == "gbp_per_household_year" and block["consumption_basis_kwh"] == 3100
        assert block["source_url"].startswith("https://") and "electricitybills" not in block["source_url"]
        assert block["access_date"] and block["reference_gbp_per_3100kwh_ex_vat"] > 0
        assert Path(block["source_file"]).exists(), block["source_file"]
        if int(block["price_year"]) != 2025:
            assert block["rebasing"]["to_price_year"] == 2025
    # Review 28 Sep 2026: the Oct-Dec 2026 cap (post-July-2025 structure) at 3,100 kWh, GB average,
    # from the stored extract of the cap model: cost to serve + debt + industry charges + EBIT + headroom.
    extract = pd.read_csv("data/market/ofgem_price_cap/2026_Q4/cap_17a_electricity_single_rate_components.csv")
    regional = extract[extract.source.eq("cap_model_v1.31") & ~extract.region.astype(str).str.contains("GB|Total", case=False)]
    means = regional.groupby(["component", "level"]).gbp_per_customer_year.mean()
    at_3100 = {c: means[(c, "nil")] + (means[(c, "2500_kwh")] - means[(c, "nil")]) * 3100 / 2500
               for c in ("CO", "DRC", "IC", "E", "H")}
    assert sourced["supplier_costs_and_margin"]["value"] == pytest.approx(sum(at_3100.values()), abs=1e-2)
    assert sourced["supplier_costs_and_margin"]["price_year"] == 2026
    # TNUoS: Domestic band residual tariff x 365 plus the locational NHH charge, NESO five-year view
    # 2027/28-2031/32 v1.0 (9 Oct 2026).
    derivation = sourced["tnuos"]["derivation"]
    assert derivation["domestic_residual_gbp_per_site_day_2030_31"] == 0.376877
    assert sourced["tnuos"]["value"] == pytest.approx(
        derivation["domestic_residual_gbp_per_year"] + derivation["locational_gbp_per_year"], abs=1e-6)
    # DUoS (29 Sep 2026): the Ofgem Annex 3 v1.25 Oct-Dec 2026 base (fixed charge plus unit rate at
    # 3,100 kWh) rolled forward on the RIIO-ED2 path to 2027/28, then on CP30's distribution capex into
    # ED3 (user decision, 6 Oct 2026; tests/unit/test_duos_cp30_growth.py), blended 3/12 : 9/12.
    duos = extract[extract.component.eq("DUoS")].groupby("level").gbp_per_customer_year.mean()
    unit = (duos["2500_kwh"] - duos["nil"]) / 2.5
    block = sourced["duos"]
    ed2 = block["derivation"]
    base = duos["nil"] + unit * 3.1
    assert ed2["base_gbp_per_year_2026"] == pytest.approx(base, abs=1e-2)
    # The ED2 path to 2027/28, then (6 Oct 2026) allowed revenue on CP30's distribution capex, plus
    # (29 Sep 2026) the ED3 allowed-return uplift.
    # The unit rate is spread over CP30's demand growth (user decision, 6 Oct 2026; derivation.charging_base).
    growth = {p: v["revenue_over_2026_27"] for p, v in ed2["ed3_cp30_growth"]["by_fiscal_year"].items()}
    spread = ed2["charging_base"]["factor_by_fiscal_year"]
    components = {c["period"]: c for c in block["period_blend"]}
    uplift = {p: u["gbp_per_household_2026_prices"] for p, u in ed2["ed3_allowed_return_uplift"]["by_period"].items()}
    for period in ("FY2029/30", "FY2030/31"):
        assert components[period]["value_at_ed2_return"] == pytest.approx(base * growth[period] * spread[period], abs=1e-2)
        assert components[period]["value"] == pytest.approx(base * growth[period] * spread[period] + uplift[period],
                                                            abs=1e-2)
    assert block["value"] == pytest.approx(0.25 * components["FY2029/30"]["value"]
                                           + 0.75 * components["FY2030/31"]["value"], abs=1e-6)
    assert block["period_basis"] == "calendar_blend"
    assert block["rebasing"]["from_price_year"] == 2026
    assert all(c["rebasing"] == block["rebasing"] and c["price_year"] == 2026 for c in components.values())
    registry = pd.read_csv("data/market/network_cost_sources/source_registry.csv")
    assert registry["local_file"].str.contains("neso_tnuos_five_year_view_2026_27_to_2030_31_v3.pdf").any()
    assert registry["local_file"].str.contains("ofgem_riio_ed2_final_determinations_overview_2022_11.pdf").any()


@requires_outputs("resources/analysis/CP30_2030_cp30_network_future_balancing_costs_summary.json")
def test_annual_constraint_total_reconciles_to_the_sum_of_months(tmp_path):
    """The plan's reconciliation check, to within GBP1."""
    from scripts.consumer_bill.extract_future_balancing_costs import (
        DEFAULT_COMPENSATION_TREATMENT,
        constraint_components,
    )
    summary = json.loads(
        Path("resources/analysis/CP30_2030_cp30_network_future_balancing_costs_summary.json").read_text()
    )
    monthly = pd.read_csv("resources/analysis/CP30_2030_cp30_network_future_balancing_costs_monthly.csv")
    annual = summary["annual"]
    assert len(monthly) == 12
    # The CfD compensation is a constraint-cost row only when the balancing LP
    # priced CfD bids at a scalar. Once it prices them on contract value the
    # settlement is inside accepted_bid_cost and the term is a check, so the
    # component list has to follow the treatment the run recorded.
    components = list(constraint_components(
        annual.get("cfd_curtailment_compensation_treatment", DEFAULT_COMPENSATION_TREATMENT)
    ))
    # Interconnector re-trading sits inside the constraint line only when the
    # extraction was configured to include it; the monthly file carries the flag.
    if bool(annual.get("include_interconnector_retrading")):
        assert monthly["include_interconnector_retrading"].all()
        components.append("interconnector_retrading_cost_gbp")
    expected = float(monthly[components].to_numpy().sum())
    # Slack and penalty-priced non-participant turn-down sit inside the accepted
    # costs but are not constraint payments; the extraction nets them when it
    # says so (29 Sep 2026 run: 0.24 m + 0.66 m).
    if bool(annual.get("constraint_cost_excludes_slack_and_penalty")):
        expected -= float(monthly["feasibility_slack_cost_in_accepted_costs_gbp"].sum())
        expected -= float(monthly["non_participant_turn_down_cost_at_penalty_gbp"].sum())
    assert annual["constraint_cost_gbp"] == pytest.approx(expected, abs=1.0)
    assert annual["constraint_cost_gbp"] == pytest.approx(float(monthly["constraint_cost_gbp"].sum()), abs=1.0)
    assert annual["reported_period_hours"] == 8760


# --------------------------------------------------------------------------
# WS2: domestic shaping and the decomposed procurement bridge
# --------------------------------------------------------------------------

def test_domestic_profile_covers_a_full_year_and_sums_to_one():
    profile = load_domestic_profile(PROFILE_PATH)
    assert profile["profile_year"] == 2020
    assert profile["peak_rate"] > profile["offpeak_rate"]
    # Household consumption is peak-heavy: more of the volume than of the hours.
    assert profile["peak_share_of_consumption"] > profile["peak_share_of_hours"]


def test_domestic_profile_rejects_a_partial_year(tmp_path):
    frame = pd.read_csv(PROFILE_PATH, index_col=0, parse_dates=True).head(100)
    path = tmp_path / "short.csv"
    frame.to_csv(path)
    with pytest.raises(ValueError, match="hours, not the"):
        load_domestic_profile(path)


def test_domestic_profile_rejects_weights_that_do_not_sum_to_one(tmp_path):
    frame = pd.read_csv(PROFILE_PATH, index_col=0, parse_dates=True)
    frame["relative_demand"] *= 2
    path = tmp_path / "unnormalised.csv"
    frame.to_csv(path)
    with pytest.raises(ValueError, match="sum to"):
        load_domestic_profile(path)


def test_profile_weighted_price_differs_from_system_weighted_on_a_two_hour_series():
    """A peak-weighted profile must price a peak-priced hour more heavily."""
    profile = {"peak_rate": 2.0, "offpeak_rate": 1.0}
    # 2030-01-02 is a Wednesday: 10:00 local is in the peak block, 03:00 is not.
    snapshots = pd.DatetimeIndex(["2030-01-02 03:00", "2030-01-02 10:00"])
    weights = domestic_weights(snapshots, profile)
    assert weights.tolist() == [1.0, 2.0]
    # New Year's Day (a Tuesday) is a bank holiday, so off-peak all day (F18, 27 Oct 2026).
    assert domestic_weights(pd.DatetimeIndex(["2030-01-01 10:00"]), profile).tolist() == [1.0]
    prices = pd.Series([10.0, 40.0], index=snapshots)
    hours = pd.Series([1.0, 1.0], index=snapshots)
    from scripts.market.ofgem_price_cap import demand_weighted_price
    assert demand_weighted_price(prices, weights, hours) == pytest.approx(30.0)
    assert demand_weighted_price(prices, pd.Series([1.0, 1.0], index=snapshots), hours) == 25.0


def test_shaping_row_carries_the_difference_and_energy_row_stays_system_weighted():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    annual["domestic_profile_price_gbp_per_mwh"] = 12.0
    rows, _ = project_components(annual, assumptions, case, balancing, recovery, ro)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    assert value["direct_energy"] == pytest.approx(10.0 * 3.1)
    assert value["shaping_adjustment"] == pytest.approx(2.0 * 3.1)


def test_wholesale_line_requires_a_domestic_profile_price():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    annual["domestic_profile_price_gbp_per_mwh"] = None
    with pytest.raises(ValueError, match="domestic-profile-weighted price"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_procurement_bridge_rows_replace_the_single_residual():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    bridge = assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]
    bridge["allowances"]["losses"]["loss_multiplier_gb_average"] = 1.10
    bridge["allowances"]["transaction_cost"]["rate"] = 0.01
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    energy = 10.0 * 3.1
    # F18: Ofgem's formula, losses after every allowance (index x (1 + rates) x (loss - 1)).
    assert value["procurement_transaction_cost"] == pytest.approx(0.01 * energy)
    assert value["procurement_forward_premium"] == 0.0
    assert "procurement_bridge" not in set(rows.component_id)
    assert summary["diagnostics"]["procurement_bridge_reference_gbp_per_3100kwh"] == 52.0


def test_forward_premium_is_a_sensitivity_that_defaults_to_zero():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    published = json.loads(Path(ASSUMPTIONS_PATH).read_text())
    premium = published["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["forward_premium"]
    assert premium["rate"] == 0.0
    # Zero central with declared sensitivities; the 2020-fitted multiplier is retired (28 Sep 2026).
    assert premium["status"] == "zero_central_with_sensitivities"
    assert set(premium["sensitivities"]) == {"minus_3pct", "plus_5pct", "stress_plus_10pct"}
    assert "calibrated_2020_coefficient" not in premium and premium["retired_calibration"]["reason"]
    assumptions["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["forward_premium"]["rate"] = 0.5
    rows, _ = project_components(annual, assumptions, case, balancing, recovery, ro)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    assert value["procurement_forward_premium"] == pytest.approx(0.5 * 10.0 * 3.1)


# --------------------------------------------------------------------------
# WS3: contract-aware strike prices
# --------------------------------------------------------------------------

def _indexation():
    return json.loads(Path(ASSUMPTIONS_PATH).read_text())["cfd_strike_indexation"]


def test_ar2_offshore_indexes_from_2012_to_the_expected_2024_value():
    from scripts.market.cfd_strike_table import load_strike_table
    table = load_strike_table(STRIKE_TABLE_PATH, _indexation(), 2024)
    strike, key, is_fallback = table.lookup("wind_offshore", "AR2")
    # AR2 offshore cleared at 74.75 (860 MW, 2021/22) and 57.50 (2336 MW,
    # 2022/23); the capacity-weighted 2012-price mean indexes by CPI.
    expected_2012 = (74.75 * 860.0 + 57.50 * 2336.0) / (860.0 + 2336.0)
    assert strike == pytest.approx(expected_2012 * 1.393340270551509)
    assert key == "AR2_offshore"
    assert is_fallback is False


def test_ar7_published_dual_price_basis_validates_the_indexation_factor():
    """DESNZ publishes AR7 in both 2012 and 2024 prices; we must reproduce it."""
    from scripts.market.cfd_strike_table import load_strike_table
    table = load_strike_table(STRIKE_TABLE_PATH, _indexation(), 2024)
    strike, _, _ = table.lookup("wind_offshore", "AR7")
    assert strike == pytest.approx(91.20, abs=0.02)


def test_a_labelled_round_never_takes_the_ar8_fallback_strike():
    from scripts.market.cfd_strike_table import load_strike_table
    table = load_strike_table(STRIKE_TABLE_PATH, _indexation(), 2024)
    ar6, ar6_key, ar6_fallback = table.lookup("wind_offshore", "AR6")
    future, future_key, future_fallback = table.lookup("wind_offshore", "")
    assert ar6_key == "AR6_offshore" and ar6_fallback is False
    assert future_key == "AR8_modelled_offshore" and future_fallback is True
    assert ar6 != future
    # AR6 cleared at 58.87 against AR7's 65.45, both in 2012 prices.
    assert ar6 == pytest.approx(58.87 * 1.393340270551509)


def test_strike_table_must_land_on_the_model_price_year():
    from scripts.market.cfd_strike_table import load_strike_table
    with pytest.raises(ValueError, match="CfD payments would mix bases"):
        load_strike_table(STRIKE_TABLE_PATH, _indexation(), 2020)


def test_strike_indexation_factor_must_reconcile_to_its_indices():
    from scripts.market.cfd_strike_table import load_strike_table
    indexation = _indexation()
    indexation["factor"] = 1.5
    with pytest.raises(ValueError, match="does not reconcile"):
        load_strike_table(STRIKE_TABLE_PATH, indexation, 2024)


def test_every_strike_table_row_is_quoted_in_2012_prices_with_a_source():
    frame = pd.read_csv(STRIKE_TABLE_PATH)
    assert (frame["price_basis"] == "2012 prices").all()
    assert frame["source"].str.strip().ne("").all()
    assert frame["strike_price_gbp_per_mwh_2012"].gt(0).all()
    # Every technology the CP30 fleet can fall back on must have a modelled AR8 row.
    future = frame.loc[frame.cfd_round.eq("AR8_modelled"), "technology"]
    assert {"offshore", "onshore", "solar"}.issubset(set(future))


def test_negative_price_rule_zeroes_payments_at_or_below_zero():
    from scripts.market.revenue_tracking import compute_cfd_payments
    from types import SimpleNamespace
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    network = SimpleNamespace(
        generators=pd.DataFrame({"carrier": ["wind_offshore"], "bus": ["GB"],
                                 "support_type": ["CfD"], "cfd_round": [""]}, index=["w"]),
        snapshots=snapshots,
        generators_t=SimpleNamespace(p=pd.DataFrame({"w": [10.0, 10.0]}, index=snapshots)),
        snapshot_weightings=pd.DataFrame({"generators": [1.0, 1.0]}, index=snapshots),
    )
    prices = pd.Series([0.0, 50.0], index=snapshots)
    strikes = {"future_offshore": 100.0}
    default = compute_cfd_payments(network, strikes, reference_price=prices)
    gated = compute_cfd_payments(network, strikes, reference_price=prices,
                                 negative_price_rule="ar4_zero_payment_when_price_le_0")
    assert default.net_cfd_cost.iloc[0] == pytest.approx(100 * 10 + 50 * 10)
    assert gated.net_cfd_cost.iloc[0] == pytest.approx(50 * 10)


def test_unknown_negative_price_rule_is_rejected():
    from scripts.market.revenue_tracking import compute_cfd_payments
    from types import SimpleNamespace
    network = SimpleNamespace(generators=pd.DataFrame({"support_type": ["CfD"]}, index=["w"]))
    with pytest.raises(ValueError, match="negative_price_rule"):
        compute_cfd_payments(network, {}, negative_price_rule="whatever")


def test_configured_fallback_limit_fails_an_unlabelled_fleet():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    annual["cfd_strike_provenance"] = {"fallback_strike_share_of_payment": 1.0,
                                       "contract_metadata_absent": ["cfd_round"]}
    case["max_fallback_strike_share_of_payment"] = 0.2
    with pytest.raises(ValueError, match="above the\n?\\s*configured limit|configured limit"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


# --------------------------------------------------------------------------
# WS4: sourced recovery denominators
# --------------------------------------------------------------------------

RECOVERY_PATH = "data/market/consumer_bill_forecast/2030_recovery_demand_assumptions.json"


def test_recovery_denominators_reconcile_to_their_declared_shares():
    document = json.loads(Path(RECOVERY_PATH).read_text())
    recovery = read_recovery_denominators(
        RECOVERY_PATH, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"]
    )
    assert recovery["cfd_eligible_share"] < 1.0
    assert recovery["cfd_eligible_supplier_sales_mwh"] < recovery["final_demand_mwh"]


def test_recovery_file_must_match_the_model_run(tmp_path):
    document = json.loads(Path(RECOVERY_PATH).read_text())
    with pytest.raises(ValueError, match="regenerate the file"):
        read_recovery_denominators(RECOVERY_PATH, "CP30_2030_cp30_network",
                                   document["system_demand_mwh"]["value"] * 1.5)


def test_recovery_file_must_match_the_case(tmp_path):
    document = json.loads(Path(RECOVERY_PATH).read_text())
    with pytest.raises(ValueError, match="declares case"):
        read_recovery_denominators(RECOVERY_PATH, "SomeOtherCase",
                                   document["system_demand_mwh"]["value"])


def test_inconsistent_eligible_sales_are_rejected(tmp_path):
    document = json.loads(Path(RECOVERY_PATH).read_text())
    document["cfd_eligible_supply"]["eligible_supplier_sales_mwh"]["value"] *= 1.2
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="deprecated alias"):
        read_recovery_denominators(path, "CP30_2030_cp30_network", document["system_demand_mwh"]["value"])


def test_ro_denominator_is_populated_and_states_its_shared_basis():
    """WS5 consumes it, so it is populated; the sharing with CfD must be explicit."""
    document = json.loads(Path(RECOVERY_PATH).read_text())
    block = document["ro_obligation_eligible_supply_mwh"]
    assert block["value"] > 0
    assert "renewables_obligation" in block["used_by"]
    # It equals the CfD eligible supply, which is an identification, not a
    # claim that the two exemption scopes are identical.
    assert block["value"] == pytest.approx(
        document["cfd_eligible_supply"]["eligible_supplier_sales_mwh"]["value"]
    )
    assert "separate instruments" in block["note"]
    assert "sourced separately" in block["note"]


def test_every_recovery_denominator_comes_from_the_file():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    with pytest.raises(ValueError, match="sourced recovery-demand file"):
        project_components(annual, assumptions, case, balancing, None, ro)


def test_system_demand_proxy_is_refused_once_eligible_sales_are_sourced():
    """The plan's WS4 acceptance test: the proxy is not available to this case."""
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    recovery["cfd_eligible_supplier_sales_mwh"] = 50
    case["cfd_recovery"] = {"method": "modelled_system_demand_proxy"}
    with pytest.raises(ValueError, match="modelled_system_demand_proxy is refused"):
        project_components(annual, assumptions, case, balancing, recovery, ro)


def test_the_cp30_case_is_configured_for_eligible_supplier_sales():
    import yaml
    defaults = load_defaults()
    case = defaults["consumer_bill"]["forecasts"]["cases"]["CP30_2030_cp30_network"]
    assert case["cfd_recovery"]["method"] == "eligible_supplier_sales"
    assert case["recovery_demand_assumptions"] == RECOVERY_PATH
    assert case["annual_aggregate_study"] == "CP30_2030_cp30_network"


# --------------------------------------------------------------------------
# WS0: the annual aggregate owns the annual accounting
# --------------------------------------------------------------------------

# The Run B market year the 2030 recovery file is refreshed from (18 Oct 2026; the LOWER network's family).
AGGREGATE_PATH = "resources/analysis/CP30_2030_cp30_network_annual_summary.json"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
CP30_SCENARIOS = [
    f"CP30_FFR_2030_{month}_cp30_network"
    for month in MONTHS
]


@requires_outputs("resources/analysis/CP30_2030_cp30_network_annual_summary.json")
def test_annual_aggregate_covers_a_complete_calendar_year():
    """The aggregate is the sole source of the annual demand the recovery file quotes.

    The expected figure is read from the recovery assumptions rather than restated here,
    so a rebuild that moves demand cannot leave the two disagreeing silently. Since the
    8 Nov 2026 refresh the file is measured on the twelve Stage 1 months directly
    (refresh_recovery_demand.py); the balancing months carry the same loads, so the
    aggregate agrees to summation rounding (3 MWh of 325 TWh on 13 Nov 2026), held here
    at one part in ten million.
    """
    aggregate = read_annual_aggregate(AGGREGATE_PATH, CP30_SCENARIOS, 2030)
    recovery = json.loads(Path(RECOVERY_PATH).read_text())
    assert aggregate["reported_period_hours"] == 8760
    assert aggregate["demand_MWh"] == pytest.approx(
        recovery["system_demand_mwh"]["value"], rel=1e-7
    )
    assert aggregate["accepted_offer_cost_GBP"] > 0
    assert aggregate["interconnector_congestion_rent_GBP"] > 0


@requires_outputs("resources/analysis/CP30_2030_cp30_network_annual_summary.json")
def test_annual_aggregate_must_cover_the_case_scenarios():
    with pytest.raises(ValueError, match="different scenarios"):
        read_annual_aggregate(AGGREGATE_PATH, CP30_SCENARIOS[:-1] + ["something_else"], 2030)


@requires_outputs("resources/analysis/CP30_2030_cp30_network_annual_summary.json")
def test_annual_aggregate_rejects_the_wrong_year():
    with pytest.raises(ValueError, match="not the 8784"):
        read_annual_aggregate(AGGREGATE_PATH, CP30_SCENARIOS, 2032)


# --------------------------------------------------------------------------
# WS7: reporting
# --------------------------------------------------------------------------

def test_summary_reports_the_pypsa_derived_share_of_the_bill():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    as_placeholder(assumptions, "remaining_ccs", 31.0)
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    # Energy 31 and the CfD credit -6.2 are model-derived; the CCS placeholder 31 is not.
    assert summary["pypsa_derived_gbp_per_3100kwh_ex_vat"] == pytest.approx(24.8)
    assert summary["pypsa_derived_share_of_ex_vat"] == pytest.approx(24.8 / 55.8)


def test_diagnostics_show_constraints_against_the_reference_and_never_charge_slack():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    balancing.update(feasibility_slack_cost_gbp=1e9, feasibility_slack_energy_mwh=5.0,
                     constraint_cost_gbp=0.0)
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    diagnostics = summary["diagnostics"]
    assert diagnostics["feasibility_slack_cost_gbp"] == 1e9
    assert diagnostics["bsuos_constraints_gbp_per_3100kwh"] == 0.0
    # A billion pounds of slack must not reach any bill row.
    assert rows.annual_gbp_per_3100kwh_ex_vat.abs().max() < 1e6


def test_diagnostics_compare_constraint_cost_with_the_pinned_neso_outturn():
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    balancing["constraint_cost_gbp"] = 500.0
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    diagnostics = summary["diagnostics"]
    comparator = assumptions["model_linked_rules"]["bsuos_constraints"]["neso_outturn_comparator"]
    assert diagnostics["neso_comparator_regulatory_year"] == "2024-25"
    assert diagnostics["neso_constraint_outturn_gbp_per_mwh_final_demand"] == pytest.approx(
        comparator["constraint_gbp_per_mwh_final_demand"]
    )
    # 500 GBP over 100 MWh of final demand
    assert diagnostics["modelled_constraint_gbp_per_mwh_final_demand"] == pytest.approx(5.0)


def test_zero_price_hours_are_counted_and_reported(monkeypatch, tmp_path):
    """F9 evidence: hours bunch just above zero because the LP has no negative bids."""
    import scripts.consumer_bill.project_future_bill as module
    index = pd.date_range("2030-01-01", "2031-01-01", freq="h", inclusive="left")
    prices = pd.Series(50.0, index=index)
    prices.iloc[:10] = 0.5      # at or below GBP1, but not below zero
    prices.iloc[10:13] = 0.0    # exactly zero
    frame = pd.DataFrame({"demand_mw": 1.0, "hours": 1.0, "price_gbp_per_mwh": prices})
    monkeypatch.setattr(module, "read_hourly_inputs", lambda *args: frame)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"scenario": "test",
        "accounting": {"reported_period_hours": 8760, "demand_MWh": 8760.0, "gross_redispatch_MWh": 0},
        "revenue": {"cfd_reference_price_source": "wholesale_system_price",
                    "cfd_total_cost": 0, "cfd_total_dispatch_MWh": 0}}))
    network, price_path = tmp_path / "model", tmp_path / "prices"
    network.write_text("mock model")
    price_path.write_text("mock prices")
    _, annual = aggregate_inputs(["test"], [network], [price_path], [path], 2030)
    assert annual["hours_price_at_or_below_1_gbp_per_mwh"] == 13
    assert annual["hours_price_at_or_below_zero"] == 3
    assert annual["hours_price_below_zero"] == 0
    assert annual["minimum_price_gbp_per_mwh"] == 0.0


def test_solver_noise_below_zero_is_not_a_negative_hour(monkeypatch, tmp_path):
    """OW12: hours_price_below_zero uses the 0.01 tolerance, as the named key does."""
    import scripts.consumer_bill.project_future_bill as module
    index = pd.date_range("2030-01-01", "2031-01-01", freq="h", inclusive="left")
    prices = pd.Series(50.0, index=index)
    prices.iloc[:269] = -8.6e-6   # barrier noise on a zero dual (the final 2030 bill)
    prices.iloc[269:273] = -5.0   # genuinely negative
    frame = pd.DataFrame({"demand_mw": 1.0, "hours": 1.0, "price_gbp_per_mwh": prices})
    monkeypatch.setattr(module, "read_hourly_inputs", lambda *args: frame)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"scenario": "test",
        "accounting": {"reported_period_hours": 8760, "demand_MWh": 8760.0, "gross_redispatch_MWh": 0},
        "revenue": {"cfd_reference_price_source": "wholesale_system_price",
                    "cfd_total_cost": 0, "cfd_total_dispatch_MWh": 0}}))
    network, price_path = tmp_path / "model", tmp_path / "prices"
    network.write_text("mock model")
    price_path.write_text("mock prices")
    _, annual = aggregate_inputs(["test"], [network], [price_path], [path], 2030)
    assert module.NEGATIVE_PRICE_TOLERANCE_GBP_PER_MWH == 0.50
    assert annual["hours_price_at_or_below_zero"] == 273
    assert annual["hours_price_below_zero"] == 4
    assert annual["hours_price_below_minus_0_01_gbp_per_mwh"] == 4


# Solver noise floor. The wholesale LP admits no negative bids, so any hour reported
# below zero is a rounding residual on a dual that is mathematically zero. A hour that
# breaches this is a real negative price and must fail the test, because the AR4
# negative-price rule would then be capable of binding and the bill could not go on
# reporting negative_price_rule == "none".
ZERO_PRICE_SOLVER_TOLERANCE_GBP_PER_MWH = -0.01
SOLVER_NOISE_GBP_PER_MWH = 1.0e-3


@requires_outputs("resources/analysis/CP30_2030_cp30_network_consumer_bill_future_summary.json")
def test_saved_report_records_the_modelled_zero_price_hours():
    """The CP30 run has many near-zero hours and none meaningfully below zero."""
    summary = json.loads(
        Path("resources/analysis/CP30_2030_cp30_network_consumer_bill_future_summary.json").read_text()
    )
    diagnostics = summary["diagnostics"]
    # Prices within 1e-3 GBP/MWh of the floor are solver noise (user, 27 Oct 2026: a minimum of
    # -0.0101 against the -0.01 floor); the floor itself is unchanged.
    assert (
        diagnostics["minimum_price_gbp_per_mwh"]
        >= ZERO_PRICE_SOLVER_TOLERANCE_GBP_PER_MWH - SOLVER_NOISE_GBP_PER_MWH
    ), (
        "the run clears meaningfully below zero, so the AR4 negative-price rule can bind "
        "and reporting it as inapplicable is no longer safe"
    )
    assert diagnostics["hours_price_at_or_below_1_gbp_per_mwh"] > 1000
    # The report records the rule the bill settled on: the case's own assumptions file
    # (CP30_2030_cp30_network reads the _cfd_rule_none twin since 12 Oct 2026, to match its bid audit).
    import yaml
    defaults = load_defaults()
    case_assumptions = defaults["consumer_bill"]["forecasts"]["cases"]["CP30_2030_cp30_network"]["assumptions"]
    assumptions = json.loads(Path(case_assumptions).read_text(encoding="utf-8"))
    assert diagnostics["negative_price_rule"] == assumptions["model_linked_rules"]["cfd"]["negative_price_rule"]


def test_report_renders_diagnostics_and_the_pypsa_share():
    from scripts.consumer_bill.project_future_bill import render_projection
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    annual["cfd_strike_provenance"] = {"by_strike": []}
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, ro)
    html = render_projection(rows, summary)
    assert "PyPSA-derived share" in html
    assert "Diagnostics" in html
    assert "CfD strike provenance" in html


def test_beccs_cfd_is_carved_out_of_the_cfd_line_onto_the_ccs_levy_line():
    """Item MR7: the CCS plant's CfD settles with the rest and is booked on allocated_ccs."""
    annual, assumptions, case, balancing, recovery, ro = fixture_inputs()
    baseline, _ = project_components(annual, assumptions, case, balancing, recovery, ro)
    base = baseline.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    total_payment = float(annual["cfd_payment_gbp"])
    beccs_share = 0.25
    annual["cfd_strike_provenance"] = {"by_strike": [
        {"carrier": "CCS Biomass", "strike_key": "POWER_BECCS_beccs", "payment_gbp": beccs_share * total_payment},
        {"carrier": "wind_offshore", "strike_key": "AR4_offshore", "payment_gbp": (1 - beccs_share) * total_payment},
    ]}
    assert "CCS Biomass" in assumptions["model_linked_rules"]["cfd"]["ccs_levy_carriers"]
    rows, _ = project_components(annual, assumptions, case, balancing, recovery, ro)
    value = rows.set_index("component_id")["annual_gbp_per_3100kwh_ex_vat"]
    assert value["allocated_ccs"] == pytest.approx(beccs_share * base["cfd"])
    assert value["cfd"] == pytest.approx((1 - beccs_share) * base["cfd"])
    assert value["cfd"] + value["allocated_ccs"] == pytest.approx(base["cfd"] + base["allocated_ccs"])
