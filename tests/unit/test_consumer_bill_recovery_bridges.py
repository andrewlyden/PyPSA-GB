import pandas as pd

from scripts.consumer_bill.build_recovery_bridges import build_bridges


def test_recovery_bridges_do_not_pass_redispatch_directly_to_bsuos():
    components = [
        "Ofgem Annex 4 — CfD", "Ofgem Annex 4 — RO", "Ofgem Annex 3 — BSUoS",
        "Capacity Market", "Ofgem Annex 3 — TNUoS", "Ofgem Annex 3 — DUoS unit charges",
        "Ofgem Annex 3 — DUoS fixed charges",
    ]
    register = pd.DataFrame({"component": components, "annual_gbp_per_3100kwh_ex_vat": [1.0] * len(components)})
    wholesale = pd.DataFrame({"modelled_demand_mwh": [100.0]})
    summary = {"redispatch": {"total_net_cost": 50.0}}
    assumptions = {"representative_customer_consumption_kwh": 3100}

    bridges, bridge_summary = build_bridges(register, wholesale, summary, assumptions)

    bsuos = bridges.loc[bridges["component"].eq("Ofgem Annex 3 — BSUoS")].iloc[0]
    assert bsuos["candidate_gbp_per_3100kwh_ex_vat"] is None
    assert bsuos["modelled_signal_value"] is None
    assert bridge_summary["not_a_bill_pass_through"] == ["PyPSA redispatch cost", "line flows", "peak demand"]


def test_cfd_bridge_calculates_only_when_recovery_inputs_are_complete():
    components = [
        "Ofgem Annex 4 — CfD", "Ofgem Annex 4 — RO", "Ofgem Annex 3 — BSUoS",
        "Capacity Market", "Ofgem Annex 3 — TNUoS", "Ofgem Annex 3 — DUoS unit charges",
        "Ofgem Annex 3 — DUoS fixed charges",
    ]
    register = pd.DataFrame({"component": components, "annual_gbp_per_3100kwh_ex_vat": [1.0] * len(components)})
    assumptions = {
        "representative_customer_consumption_kwh": 3100,
        "cfd": {"modelled_recovery_cost_gbp": 1000.0, "billable_supplier_sales_mwh": 100.0, "domestic_recovery_share": 0.5},
    }
    bridges, _ = build_bridges(register, pd.DataFrame({"modelled_demand_mwh": [100.0]}), {"redispatch": {"total_net_cost": 0.0}}, assumptions)
    cfd = bridges.loc[bridges["component"].eq("Ofgem Annex 4 — CfD")].iloc[0]
    assert cfd["candidate_gbp_per_3100kwh_ex_vat"] == 15.5
    assert cfd["evidence_status"] == "candidate_unvalidated"
    assert cfd["backtest_substitution_allowed"] == False


def test_bsuos_uses_gross_redispatch_volume_as_a_calibrated_index_not_a_pass_through():
    components = [
        "Ofgem Annex 4 — CfD", "Ofgem Annex 4 — RO", "Ofgem Annex 3 — BSUoS",
        "Capacity Market", "Ofgem Annex 3 — TNUoS", "Ofgem Annex 3 — DUoS unit charges",
        "Ofgem Annex 3 — DUoS fixed charges",
    ]
    register = pd.DataFrame({"component": components, "annual_gbp_per_3100kwh_ex_vat": [1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0]})
    assumptions = {"representative_customer_consumption_kwh": 3100, "bsuos": {"calibration_gross_redispatch_mwh_per_mwh": 0.5}}
    summary = {"redispatch": {"total_net_cost": 100.0, "total_increase_MWh": 50.0, "total_decrease_MWh": 50.0}}
    bridges, _ = build_bridges(register, pd.DataFrame({"modelled_demand_mwh": [100.0]}), summary, assumptions)
    bsuos = bridges.loc[bridges["component"].eq("Ofgem Annex 3 — BSUoS")].iloc[0]
    assert bsuos["candidate_gbp_per_3100kwh_ex_vat"] == 20.0
    assert bsuos["evidence_status"] == "candidate_unvalidated"
    assert bsuos["modelled_signal_unit"] == "gross redispatch MWh per MWh model demand"


def test_subsidy_signals_are_exposed_without_claiming_a_consumer_allocation():
    components = [
        "Ofgem Annex 4 — CfD", "Ofgem Annex 4 — RO", "Ofgem Annex 3 — BSUoS",
        "Imbalance uplift", "Capacity Market", "Ofgem Annex 3 — TNUoS",
        "Ofgem Annex 3 — DUoS unit charges", "Ofgem Annex 3 — DUoS fixed charges",
    ]
    register = pd.DataFrame({"component": components, "annual_gbp_per_3100kwh_ex_vat": [1.0] * len(components)})
    signals = pd.DataFrame({"metric": ["cfd_payment_proxy_gbp", "roc_obligation_proxy_gbp"], "value": [100.0, 200.0]})
    bridges, _ = build_bridges(register, pd.DataFrame({"modelled_demand_mwh": [100.0]}), {"redispatch": {"total_net_cost": 0.0}}, {"representative_customer_consumption_kwh": 3100}, signals)
    cfd = bridges.loc[bridges["component"].eq("Ofgem Annex 4 — CfD")].iloc[0]
    ro = bridges.loc[bridges["component"].eq("Ofgem Annex 4 — RO")].iloc[0]
    assert cfd["evidence_status"] == "modelled_cost_pending_recovery_allocation"
    assert ro["modelled_signal_value"] == 200.0


def test_calendar_reconciled_roc_cost_is_preferred_for_historical_bridge():
    dash = "\u00e2\u20ac\u201d"
    components = [
        f"Ofgem Annex 4 {dash} CfD", f"Ofgem Annex 4 {dash} RO", f"Ofgem Annex 3 {dash} BSUoS",
        "Capacity Market", f"Ofgem Annex 3 {dash} TNUoS", f"Ofgem Annex 3 {dash} DUoS unit charges",
        f"Ofgem Annex 3 {dash} DUoS fixed charges",
    ]
    register = pd.DataFrame({"component": components, "annual_gbp_per_3100kwh_ex_vat": [1.0] * len(components)})
    signals = pd.DataFrame({
        "metric": ["roc_obligation_proxy_gbp", "roc_reconciled_obligation_proxy_gbp"],
        "value": [200.0, 300.0],
    })

    bridges, _ = build_bridges(
        register, pd.DataFrame({"modelled_demand_mwh": [100.0]}),
        {"redispatch": {"total_net_cost": 0.0}},
        {"representative_customer_consumption_kwh": 3100}, signals,
    )

    ro = bridges.loc[bridges["bridge"].eq("Renewables Obligation recovery")].iloc[0]
    assert ro["modelled_signal_value"] == 300.0


def test_independent_cfd_formula_is_preferred_to_generic_revenue_proxy():
    dash = "\u00e2\u20ac\u201d"
    components = [
        f"Ofgem Annex 4 {dash} CfD", f"Ofgem Annex 4 {dash} RO",
        f"Ofgem Annex 3 {dash} BSUoS", "Capacity Market",
        f"Ofgem Annex 3 {dash} TNUoS",
        f"Ofgem Annex 3 {dash} DUoS unit charges",
        f"Ofgem Annex 3 {dash} DUoS fixed charges",
    ]
    register = pd.DataFrame({
        "component": components,
        "annual_gbp_per_3100kwh_ex_vat": [1.0] * len(components),
    })
    signals = pd.DataFrame({
        "metric": ["cfd_payment_proxy_gbp", "cfd_independent_formula_payment_gbp"],
        "value": [100.0, 250.0],
    })
    summary = {
        "redispatch": {"total_net_cost": 0.0},
        "revenue": {"cfd_total_cost": 150.0},
    }

    bridges, _ = build_bridges(
        register, pd.DataFrame({"modelled_demand_mwh": [100.0]}), summary,
        {"representative_customer_consumption_kwh": 3100}, signals,
    )

    cfd = bridges.loc[bridges["bridge"].eq("CfD levy recovery")].iloc[0]
    assert cfd["modelled_signal_value"] == 250.0


def test_network_and_capacity_indices_are_explicit_same_year_assumptions():
    dash = "\u00e2\u20ac\u201d"
    components = [
        f"Ofgem Annex 4 {dash} CfD", f"Ofgem Annex 4 {dash} RO",
        f"Ofgem Annex 3 {dash} BSUoS", "Capacity Market",
        "GB-average regional electricity losses", f"Ofgem Annex 3 {dash} TNUoS",
        f"Ofgem Annex 3 {dash} DUoS unit charges",
        f"Ofgem Annex 3 {dash} DUoS fixed charges",
    ]
    register = pd.DataFrame({
        "component": components,
        "annual_gbp_per_3100kwh_ex_vat": [1.0, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    signals = pd.DataFrame({
        "metric": ["transmission_loss_rate", "wind_solar_residual_peak_mw", "transmission_capacity_mw"],
        "value": [0.02, 100.0, 200.0],
    })
    assumptions = {
        "representative_customer_consumption_kwh": 3100,
        "losses": {"use_current_run_as_calibration_baseline": True},
        "capacity_market": {"use_current_run_as_calibration_baseline": True},
        "tnuos": {"use_current_run_as_calibration_baseline": True},
        "duos_unit": {"assumed_multiplier": 1.0},
        "duos_fixed": {"assumed_multiplier": 1.0},
    }

    bridges, _ = build_bridges(
        register, pd.DataFrame({"modelled_demand_mwh": [100.0]}),
        {"redispatch": {"total_net_cost": 0.0}}, assumptions, signals,
    )

    expected = {
        "Loss-factor recovery": 30.0,
        "Capacity Market recovery": 20.0,
        "Transmission allowed-revenue recovery": 40.0,
        "Distribution volumetric allowed-revenue recovery": 50.0,
        "Distribution fixed allowed-revenue recovery": 60.0,
    }
    for bridge, amount in expected.items():
        row = bridges.loc[bridges["bridge"].eq(bridge)].iloc[0]
        assert row["candidate_gbp_per_3100kwh_ex_vat"] == amount
        assert row["backtest_substitution_allowed"] == False


def test_loss_bridge_prefers_the_calibrated_i2r_proxy_over_zero_represented_losses():
    dash = "â€”"
    components = [
        f"Ofgem Annex 4 {dash} CfD", f"Ofgem Annex 4 {dash} RO",
        f"Ofgem Annex 3 {dash} BSUoS", "Capacity Market",
        "GB-average regional electricity losses", f"Ofgem Annex 3 {dash} TNUoS",
        f"Ofgem Annex 3 {dash} DUoS unit charges", f"Ofgem Annex 3 {dash} DUoS fixed charges",
    ]
    register = pd.DataFrame({"component": components, "annual_gbp_per_3100kwh_ex_vat": [1.0] * len(components)})
    signals = pd.DataFrame({
        "metric": ["transmission_loss_rate", "transmission_i2r_loss_proxy_rate"],
        "value": [0.0, 0.02],
    })

    bridges, _ = build_bridges(
        register, pd.DataFrame({"modelled_demand_mwh": [100.0]}),
        {"redispatch": {"total_net_cost": 0.0}},
        {"representative_customer_consumption_kwh": 3100, "losses": {"use_current_run_as_calibration_baseline": True}},
        signals,
    )

    loss = bridges.loc[bridges["bridge"].eq("Loss-factor recovery")].iloc[0]
    assert loss["candidate_gbp_per_3100kwh_ex_vat"] == 1.0
    assert loss["modelled_signal_value"] == 0.02
    assert loss["modelled_signal_unit"] == "resistance-weighted I²R proxy MWh per MWh model demand"


def test_fit_requires_specific_generation_and_levelisation_recovery_inputs():
    dash = "\u00e2\u20ac\u201d"
    components = [
        f"Ofgem Annex 4 {dash} CfD", f"Ofgem Annex 4 {dash} RO",
        f"Ofgem Annex 4 {dash} FiT", f"Ofgem Annex 3 {dash} BSUoS",
        "Capacity Market", f"Ofgem Annex 3 {dash} TNUoS",
        f"Ofgem Annex 3 {dash} DUoS unit charges", f"Ofgem Annex 3 {dash} DUoS fixed charges",
    ]
    register = pd.DataFrame({"component": components, "annual_gbp_per_3100kwh_ex_vat": [1.0] * len(components)})
    assumptions = {
        "representative_customer_consumption_kwh": 3100,
        "fit": {"modelled_recovery_cost_gbp": 1000.0, "billable_supplier_sales_mwh": 100.0, "domestic_recovery_share": 0.5},
    }
    signals = pd.DataFrame({"metric": ["fit_dispatch_mwh"], "value": [10.0]})

    bridges, _ = build_bridges(
        register, pd.DataFrame({"modelled_demand_mwh": [100.0]}),
        {"redispatch": {"total_net_cost": 0.0}}, assumptions, signals,
    )

    fit = bridges.loc[bridges["bridge"].eq("FiT levelisation recovery")].iloc[0]
    assert fit["candidate_gbp_per_3100kwh_ex_vat"] == 15.5
    assert fit["evidence_status"] == "candidate_unvalidated"


def test_bsuos_volume_index_is_unchanged_by_modelled_bid_offer_cost():
    components = [
        "Ofgem Annex 4 â€” CfD", "Ofgem Annex 4 â€” RO", "Ofgem Annex 3 â€” BSUoS",
        "Capacity Market", "Ofgem Annex 3 â€” TNUoS", "Ofgem Annex 3 â€” DUoS unit charges",
        "Ofgem Annex 3 â€” DUoS fixed charges",
    ]
    register = pd.DataFrame({
        "component": components,
        "annual_gbp_per_3100kwh_ex_vat": [1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0],
    })
    assumptions = {
        "representative_customer_consumption_kwh": 3100,
        "bsuos": {"calibration_gross_redispatch_mwh_per_mwh": 1.0},
    }
    wholesale = pd.DataFrame({"modelled_demand_mwh": [100.0]})
    cheap, _ = build_bridges(
        register, wholesale,
        {"redispatch": {"total_net_cost": 1.0, "total_increase_MWh": 50.0, "total_decrease_MWh": 50.0}},
        assumptions,
    )
    expensive, _ = build_bridges(
        register, wholesale,
        {"redispatch": {"total_net_cost": 1_000_000.0, "total_increase_MWh": 50.0, "total_decrease_MWh": 50.0}},
        assumptions,
    )
    cheap_bsuos = cheap.loc[cheap["bridge"].eq("Balancing / BSUoS recovery"), "candidate_gbp_per_3100kwh_ex_vat"].iloc[0]
    expensive_bsuos = expensive.loc[expensive["bridge"].eq("Balancing / BSUoS recovery"), "candidate_gbp_per_3100kwh_ex_vat"].iloc[0]
    assert cheap_bsuos == expensive_bsuos == 10.0
