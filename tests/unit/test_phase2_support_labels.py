"""Phase 2: Renewables Obligation from the ROC fleet, and CfD contract vintages.

These cover WS5 and WS6 of the 2030 consumer-bill improvement plan. The bill
rows they produce take effect immediately for RO, which is derived from the
Ofgem register rather than the network, and only after the twelve monthly
networks are rebuilt for the CfD vintage labels, which are generator attributes.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.consumer_bill.project_future_bill import project_components, read_ro_scheme_cost
from tests.unit.test_future_consumer_bill import (
    fixture_inputs, STRIKE_TABLE_PATH, _indexation,
)
from config.config_loader import load_defaults
from tests.helpers import requires_outputs


RO_PARAMETERS_PATH = "data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json"
RO_SUMMARY_PATH = "resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json"
RO_REGISTER_PATH = "data/market/ofgem_renewables_obligation/accredited_stations_ro_current.csv"
VINTAGE_SCHEDULE_PATH = "data/market/CP30/cfd_vintage_schedule_2030.csv"


def _ro_block(relief_case="relief_ends_march_2029"):
    return read_ro_scheme_cost(RO_SUMMARY_PATH, relief_case, 2025)


# --------------------------------------------------------------------------
# WS5: Renewables Obligation from the ROC-eligible fleet
# --------------------------------------------------------------------------

def test_roc_accreditation_expires_twenty_years_after_the_effective_date():
    """Regression test for the 20-year cut-off falling inside 2030."""
    from scripts.consumer_bill.build_ro_eligible_fleet import read_eligible_fleet
    fleet = read_eligible_fleet(RO_REGISTER_PATH)
    # The bill is calendar 2030 (28 Sep 2026): a station counts for the share of
    # 2030 its 20 years still run. Accredited before 2010, none of 2030; during
    # 2010, part of it; from 2011, all of it.
    before = fleet[fleet.effective_date < pd.Timestamp("2010-01-01")]
    during = fleet[(fleet.effective_date > pd.Timestamp("2010-01-15"))
                   & (fleet.effective_date < pd.Timestamp("2010-12-15"))]
    after = fleet[fleet.effective_date >= pd.Timestamp("2011-01-01")]
    assert not before.eligible.any()
    assert during.eligible.all()
    assert ((during.accredited_share > 0.0) & (during.accredited_share < 1.0)).all()
    assert after.eligible.all() and (after.accredited_share == 1.0).all()
    # The cut-off must actually bite: every side is populated.
    assert len(before) > 100 and len(during) > 10 and len(after) > 100


def test_accredited_share_counts_the_days_of_the_calendar_year():
    from scripts.consumer_bill.build_ro_eligible_fleet import accredited_share_of_period
    effective = pd.Series(pd.to_datetime(["2009-06-01", "2010-04-01", "2010-10-01", "2015-01-01"]))
    expiry = effective + pd.DateOffset(years=20)
    share = accredited_share_of_period(effective, expiry).round(4).tolist()
    # Expired mid-2029; Jan-Mar 2030 (90 days); Jan-Sep 2030 (273 days); all of 2030.
    assert share == [0.0, round(90 / 365, 4), round(273 / 365, 4), 1.0]
    # The old single-date rule, kept for other callers.
    from scripts.consumer_bill.build_ro_eligible_fleet import read_eligible_fleet
    old = read_eligible_fleet(RO_REGISTER_PATH, period=None)
    assert set(old.accredited_share.unique()) <= {0.0, 1.0}


def test_roc_eligibility_is_capped_at_the_scheme_end_date():
    from scripts.consumer_bill.build_ro_eligible_fleet import read_eligible_fleet, RO_SCHEME_END
    fleet = read_eligible_fleet(RO_REGISTER_PATH)
    assert fleet.roc_expiry_capped.max() <= RO_SCHEME_END
    # Some stations' raw 20-year window runs past the scheme's own end.
    assert (fleet.roc_expiry > RO_SCHEME_END).any()


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_ro_scheme_cost_reports_both_relief_cases_and_the_reference():
    summary = json.loads(Path(RO_SUMMARY_PATH).read_text())
    assert set(summary["relief_cases"]) == {"relief_ends_march_2029", "relief_extended"}
    gross = summary["gross_gbp_per_3100kwh_ex_vat"]
    assert summary["relief_cases"]["relief_ends_march_2029"]["gbp_per_3100kwh_ex_vat"] == pytest.approx(gross)
    assert summary["relief_cases"]["relief_extended"]["gbp_per_3100kwh_ex_vat"] == pytest.approx(gross * 0.25)
    assert summary["reference_external_gbp_per_3100kwh_ex_vat"] == 17.0
    assert summary["method_label"] == "pypsa_informed_fleet_register"


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_ro_eligible_capacity_is_compared_with_the_register_total():
    summary = json.loads(Path(RO_SUMMARY_PATH).read_text())
    # Item MR8's activity rule drops still-accredited fuelled stations with no recent
    # certificates (Fawley's 1,000 MW among them) from the eligible fleet without
    # expiring them, so they are a third bucket: eligible + expired + inactive is the
    # register. Summaries built before the bucket was reported carry it only by carrier.
    inactive = summary.get("inactive_capacity_mw")
    if inactive is None:
        inactive = sum(summary["certificate_evidence"]["inactive_capacity_mw_by_carrier"].values())
    # Calendar 2030 (28 Sep 2026): a station accredited for part of the year counts for that
    # part, and the rest of its capacity is a fourth bucket (summaries built before carry none).
    part_year = summary.get("unaccredited_part_of_year_capacity_mw", 0.0)
    assert summary["eligible_capacity_mw"] + summary["expired_capacity_mw"] + inactive + part_year == pytest.approx(
        summary["register_total_capacity_mw"], rel=1e-6
    )
    # Roughly half the accredited fleet has aged out of the scheme by 2030.
    assert 0.4 < summary["eligible_capacity_mw"] / summary["register_total_capacity_mw"] < 0.6


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_every_register_technology_maps_to_a_carrier():
    summary = json.loads(Path(RO_SUMMARY_PATH).read_text())
    assert summary["unmapped_register_technologies"] == []


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_ro_row_uses_the_declared_relief_case():
    annual, assumptions, case, balancing, recovery, _ = fixture_inputs()
    ends = _ro_block("relief_ends_march_2029")
    extended = _ro_block("relief_extended")
    rows_ends, _ = project_components(annual, assumptions, case, balancing, recovery, ends)
    rows_ext, summary_ext = project_components(
        annual, assumptions, case, balancing, recovery, extended
    )
    value_ends = rows_ends.set_index("component_id").loc[
        "renewables_obligation", "annual_gbp_per_3100kwh_ex_vat"]
    value_ext = rows_ext.set_index("component_id").loc[
        "renewables_obligation", "annual_gbp_per_3100kwh_ex_vat"]
    assert value_ends == pytest.approx(value_ext * 4, rel=1e-6)
    assert summary_ext["diagnostics"]["ro_relief_case"] == "relief_extended"
    assert summary_ext["diagnostics"]["ro_relief_swing_gbp_per_3100kwh"] == pytest.approx(
        value_ends - value_ext
    )


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_unknown_relief_case_is_rejected():
    with pytest.raises(ValueError, match="not one of"):
        read_ro_scheme_cost(RO_SUMMARY_PATH, "relief_maybe", 2025)


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_ro_scheme_cost_must_be_on_the_report_price_basis():
    with pytest.raises(ValueError, match="basis"):
        read_ro_scheme_cost(RO_SUMMARY_PATH, "relief_ends_march_2029", 2022)


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_external_ro_allowance_is_no_longer_summed_into_the_bill():
    annual, assumptions, case, balancing, recovery, _ = fixture_inputs()
    assert "renewables_obligation" not in assumptions["external_assumptions_gbp_per_3100kwh_ex_vat"]
    rows, _ = project_components(annual, assumptions, case, balancing, recovery, _ro_block())
    ro_rows = rows[rows.component_id.eq("renewables_obligation")]
    assert len(ro_rows) == 1
    assert ro_rows.source_class.iloc[0] == "pypsa_informed_fleet_register"


def test_ro_row_requires_the_scheme_cost_extraction():
    annual, assumptions, case, balancing, recovery, _ = fixture_inputs()
    with pytest.raises(ValueError, match="Renewables Obligation row needs"):
        project_components(annual, assumptions, case, balancing, recovery, None)


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_ro_line_is_counted_as_informed_not_derived():
    """Only its load factors are model-derived; the fleet and ROC value are not."""
    annual, assumptions, case, balancing, recovery, _ = fixture_inputs()
    _, summary = project_components(annual, assumptions, case, balancing, recovery, _ro_block())
    assert summary["pypsa_informed_gbp_per_3100kwh_ex_vat"] > 0
    # The RO row must not inflate the fully-model-derived share.
    assert summary["pypsa_derived_gbp_per_3100kwh_ex_vat"] == pytest.approx(24.8)


def test_buyout_price_needs_no_inflation_forecast():
    """It is RPI-indexed, so restating the sourced value gives the 2030/31 level."""
    from scripts.consumer_bill.build_ro_eligible_fleet import notional_roc_value
    parameters = json.loads(Path(RO_PARAMETERS_PATH).read_text())
    value = notional_roc_value(parameters, parameters["rpi_restatement"])
    base = parameters["base_obligation_year"]["final_notional_roc_value_gbp_per_roc"]
    factor = 402.7 / 288.8
    assert value["rpi_restatement_factor"] == pytest.approx(factor)
    assert value["notional_roc_value_gbp_per_roc_report_prices"] == pytest.approx(base * factor)


def test_rpi_restatement_factor_must_reconcile_to_its_indices():
    from scripts.consumer_bill.build_ro_eligible_fleet import notional_roc_value
    parameters = json.loads(Path(RO_PARAMETERS_PATH).read_text())
    rpi = dict(parameters["rpi_restatement"], factor=1.9)
    with pytest.raises(ValueError, match="does not reconcile"):
        notional_roc_value(parameters, rpi)


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_eligible_fleet.csv")
def test_ro_load_factors_come_from_physical_not_copperplate_dispatch():
    """ROCs are issued on metered output, so curtailed wind earns none."""
    fleet = pd.read_csv("resources/analysis/CP30_2030_cp30_network_ro_eligible_fleet.csv").set_index("carrier")
    # Offshore wind load factors come from the balancing (post-curtailment)
    # dispatch of the rebuilt networks and sit inside the observed GB range.
    # Since 26 Sep 2026 the value is the ROC-labelled units' own dispatch where the
    # model holds enough of them (source ..._roc_units); ROC units bid minus their
    # certificate value and are rarely curtailed, so they run near 0.45.
    assert str(fleet.at["wind_offshore", "load_factor_source"]).startswith("pypsa_balancing_dispatch")
    assert 0.25 < fleet.at["wind_offshore", "pypsa_load_factor"] < 0.50
    # Biomass carries a must-run floor of 0.35 (thermal.availability_factors in
    # the CP30 scenarios, below the pathway-implied 0.39) as a daily energy budget
    # under a 0.80 availability cap (must-run package, 28 Sep 2026; was 0.53), so
    # its modelled load factor sits between the two. The ROC units (short-run cost
    # ~8 GBP/MWh) now run up to the cap. This test once expected it below 0.1,
    # from before the floor existed (item E3).
    # The fleet still applies the declared CP30 ES1 generation override.
    assert 0.35 - 1e-6 <= fleet.at["biomass", "pypsa_load_factor"] <= 0.80 + 1e-6
    assert fleet.at["biomass", "applied_load_factor"] == pytest.approx(12155 / 3544 / 8.76, abs=0.005)
    assert str(fleet.at["biomass", "load_factor_source"]).startswith("override:")


# --------------------------------------------------------------------------
# WS6: CfD allocation-round vintages and the nuclear position
# --------------------------------------------------------------------------

def test_vintage_schedule_rows_are_sourced_and_exclude_the_modelled_round():
    from scripts.generators.cfd_vintage_labels import load_vintage_schedule
    schedule = load_vintage_schedule(VINTAGE_SCHEDULE_PATH)
    assert schedule.source.str.strip().ne("").all()
    assert schedule.capacity_mw.gt(0).all()
    assert "AR8_modelled" not in set(schedule.cfd_round)
    assert {"wind_offshore", "wind_onshore", "solar_pv"} <= set(schedule.carrier)


def test_round_allocation_is_proportional_and_capacity_preserving():
    from scripts.generators.cfd_vintage_labels import allocate_rounds
    schedule = pd.DataFrame({"cfd_round": ["AR4", "AR6"], "capacity_mw": [300.0, 100.0]})
    allocation = allocate_rounds(200.0, schedule)
    assert sum(allocation.values()) == pytest.approx(200.0)
    assert allocation["AR4"] == pytest.approx(150.0)
    assert allocation["AR6"] == pytest.approx(50.0)


def test_capacity_beyond_the_schedule_goes_to_the_modelled_ar8_round():
    from scripts.generators.cfd_vintage_labels import allocate_rounds
    schedule = pd.DataFrame({"cfd_round": ["AR7"], "capacity_mw": [100.0]})
    allocation = allocate_rounds(250.0, schedule)
    assert allocation["AR7"] == pytest.approx(100.0)
    assert allocation["AR8_modelled"] == pytest.approx(150.0)
    assert sum(allocation.values()) == pytest.approx(250.0)


def test_labelling_splits_cfd_rows_without_changing_total_capacity():
    from scripts.generators.cfd_vintage_labels import apply_cfd_vintage_labels
    sites = pd.DataFrame({
        "site_name": ["off_a", "on_a", "merch"],
        "technology": ["wind_offshore", "wind_onshore", "wind_offshore"],
        "capacity_mw": [1000.0, 500.0, 250.0],
        "support_type": ["CfD", "CfD", "merchant"],
    })
    result = apply_cfd_vintage_labels(sites, VINTAGE_SCHEDULE_PATH)
    assert result.capacity_mw.sum() == pytest.approx(1750.0)
    merchant = result[result.support_type.eq("merchant")]
    assert len(merchant) == 1 and merchant.capacity_mw.iloc[0] == pytest.approx(250.0)
    offshore = result[result.technology.eq("wind_offshore") & result.support_type.eq("CfD")]
    assert offshore.capacity_mw.sum() == pytest.approx(1000.0)
    assert set(offshore.cfd_round) == {"AR2", "AR3", "AR4", "AR6", "AR7"}


def test_labelling_leaves_already_labelled_contracts_alone():
    from scripts.generators.cfd_vintage_labels import apply_cfd_vintage_labels
    sites = pd.DataFrame({
        "technology": ["wind_offshore"], "capacity_mw": [900.0],
        "support_type": ["CfD"], "cfd_round": ["AR4"],
    })
    result = apply_cfd_vintage_labels(sites, VINTAGE_SCHEDULE_PATH)
    assert len(result) == 1
    assert result.cfd_round.iloc[0] == "AR4"


def test_a_carrier_with_no_schedule_takes_the_modelled_ar8_strike():
    from scripts.generators.cfd_vintage_labels import apply_cfd_vintage_labels
    sites = pd.DataFrame({
        "technology": ["shoreline_wave"], "capacity_mw": [10.0], "support_type": ["CfD"],
    })
    result = apply_cfd_vintage_labels(sites, VINTAGE_SCHEDULE_PATH)
    assert len(result) == 1
    assert result.cfd_round.iloc[0] == "AR8_modelled"
    assert result.capacity_mw.iloc[0] == pytest.approx(10.0)


def test_marine_cfd_proxies_take_the_tidal_stream_rounds_and_each_has_a_strike():
    """Item C4: AR4-AR7a tidal awards, the rest on the modelled AR8 tidal strike."""
    from scripts.generators.cfd_vintage_labels import apply_cfd_vintage_labels
    from scripts.market.cfd_strike_table import load_strike_table
    sites = pd.DataFrame({"site_name": ["CP30_marine_0", "CP30_marine_1"], "technology": ["marine", "marine"],
                          "capacity_mw": [140.0, 98.4], "support_type": ["CfD", "CfD"]})
    result = apply_cfd_vintage_labels(sites, VINTAGE_SCHEDULE_PATH)
    by_round = result.groupby("cfd_round")["capacity_mw"].sum()
    assert result["capacity_mw"].sum() == pytest.approx(238.4)
    # AR6 tidal net of the 6 MW LCCC lists Terminated (rebuild batch, 14 Oct 2026).
    for cfd_round, mw in {"AR4": 40.82, "AR5": 53.04, "AR6": 22.00, "AR7a": 20.90}.items():
        assert by_round[cfd_round] == pytest.approx(mw)
    assert by_round["AR8_modelled"] == pytest.approx(238.4 - 136.76)
    table = load_strike_table(STRIKE_TABLE_PATH, _indexation(), 2024)
    for cfd_round in ("AR4", "AR5", "AR6", "AR7a"):
        strike, key, fallback = table.lookup("marine", cfd_round)
        assert not fallback and strike > 150.0, (cfd_round, key)
    # The rest settles at the modelled AR8 tidal strike (F18), its own price, not a copy of AR7a.
    ar8_tidal = table.lookup("marine", "AR8_modelled")
    assert not ar8_tidal[2] and ar8_tidal[0] > 150.0 and ar8_tidal[0] != pytest.approx(table.lookup("marine", "AR7a")[0])


def test_labelled_rounds_resolve_to_different_strike_prices():
    """The point of the exercise: rounds must not all settle at one price."""
    from scripts.market.cfd_strike_table import load_strike_table
    table = load_strike_table(STRIKE_TABLE_PATH, _indexation(), 2024)
    strikes = {
        cfd_round: table.lookup("wind_offshore", cfd_round)[0]
        for cfd_round in ("AR2", "AR3", "AR4", "AR6", "AR7")
    }
    assert len({round(value, 4) for value in strikes.values()}) == 5
    # AR4 was the record low; AR2 is a capacity-weighted mean of 74.75 and
    # 57.50, so it lands between AR6 and AR7 rather than above both.
    assert strikes["AR4"] < strikes["AR3"] < strikes["AR6"] < strikes["AR2"] < strikes["AR7"]


def test_aggregation_never_mixes_allocation_rounds_on_one_bus():
    """The aggregated row inherits the largest member's attributes, so a mixed
    group would silently relabel the smaller tranches with the wrong round."""
    import pypsa
    from scripts.generators.aggregate_renewable_generators import aggregate_renewables_by_bus
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    network.add("Bus", "B1")
    for name, capacity, cfd_round in (("big", 900.0, "AR7"), ("small", 100.0, "AR4")):
        network.add("Generator", name, bus="B1", carrier="wind_offshore", p_nom=capacity)
        network.generators.loc[name, "support_type"] = "CfD"
        network.generators.loc[name, "cfd_round"] = cfd_round
    network, _ = aggregate_renewables_by_bus(network, carriers=["wind_offshore"])
    assert set(network.generators.cfd_round) == {"AR7", "AR4"}
    assert network.generators.p_nom.sum() == pytest.approx(1000.0)


def test_aggregation_still_merges_generators_of_the_same_vintage():
    import pypsa
    from scripts.generators.aggregate_renewable_generators import aggregate_renewables_by_bus
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    network.add("Bus", "B1")
    for name in ("a", "b"):
        network.add("Generator", name, bus="B1", carrier="wind_offshore", p_nom=100.0)
        network.generators.loc[name, "support_type"] = "CfD"
        network.generators.loc[name, "cfd_round"] = "AR7"
    network, removed = aggregate_renewables_by_bus(network, carriers=["wind_offshore"])
    assert len(network.generators) == 1
    assert removed > 0
    assert network.generators.p_nom.iloc[0] == pytest.approx(200.0)


def test_nuclear_carries_no_cfd_label_and_the_exclusion_is_recorded():
    annual, assumptions, case, balancing, recovery, _ = fixture_inputs()
    exclusion = assumptions["model_linked_rules"]["cfd"]["nuclear_exclusion"]
    # The rebuilt networks label Hinkley Point C as a CfD (round HPC) from the
    # station schedule, so the settled payment is inside the CfD line and the
    # file says so.
    assert exclusion["status"] == "resolved_hpc_cfd_settled_in_bill"
    assert "nuclear_station_schedule_2030.csv" in exclusion["status_note"]
    # 89.50 since Sizewell C's financial close (4 Nov 2025): in the strike table since OW12.
    assert exclusion["hpc_strike_gbp_per_mwh_2012"] == 89.5
    # Sizewell C is RAB-financed, so its levy is a different instrument.
    assert "Regulated Asset Base" in exclusion["note"]
    assert "sizewell_c_rab" in exclusion["note"]
    assert "nuclear_contract_counterparty" in exclusion["note"]


def test_fleet_wide_nuclear_cfd_fraction_has_been_replaced_by_a_station_schedule():
    """`nuclear: 1.0` labelled all nuclear CfD "for Sizewell C", which is RAB-financed.

    The fraction was live in the thermal path, not dead as first recorded; it was
    wrong in principle and could not express one CfD station beside one
    unsupported station, which the schedule can.
    """
    import yaml
    defaults = load_defaults()
    fractions = defaults["support"]["labels"]["fes_cfd_fractions"]
    assert "nuclear" not in fractions
    assert "nuclear_cfd_tranches" not in defaults["support"]["labels"]
    assert Path("data/market/CP30/nuclear_station_schedule_2030.csv").is_file()


@requires_outputs("resources/analysis/CP30_2030_cp30_network_ro_scheme_summary.json")
def test_nuclear_diagnostic_prices_the_gap_without_charging_it():
    annual, assumptions, case, balancing, recovery, _ = fixture_inputs()
    balancing["unlabelled_support"] = [
        {"carrier": "nuclear", "capacity_mw": 3515.0, "physical_mwh": 25_000_000.0,
         "generators": 20},
    ]
    rows, summary = project_components(annual, assumptions, case, balancing, recovery, _ro_block())
    diagnostics = summary["diagnostics"]
    assert diagnostics["unlabelled_nuclear_capacity_mw"] == 3515.0
    assert diagnostics["unlabelled_nuclear_twh"] == pytest.approx(25.0)
    assert diagnostics["nuclear_cfd_in_bill_gbp_per_3100kwh"] == 0.0
    assert diagnostics["nuclear_cfd_hpc_sensitivity_gbp"] > 1e9
    # It is a diagnostic only: no bill row carries a nuclear CfD.
    assert not any("nuclear_cfd" in str(value) for value in rows.component_id)


@requires_outputs("resources/analysis/CP30_2030_cp30_network_consumer_bill_future_summary.json")
def test_saved_report_records_the_unlabelled_nuclear_fleet():
    summary = json.loads(
        Path("resources/analysis/CP30_2030_cp30_network_consumer_bill_future_summary.json").read_text()
    )
    diagnostics = summary["diagnostics"]
    # After the brownfield rebuild no nuclear proxy is unlabelled: Hinkley Point C
    # (2,317 MW) settles as a CfD and Sizewell B is merchant by schedule.
    assert diagnostics["unlabelled_nuclear_capacity_mw"] == 0.0
    assert diagnostics["nuclear_cfd_exclusion_status"] == "resolved_hpc_cfd_settled_in_bill"
    # HPC at 2,317 MW runs about 18 TWh a year; the settled payment is inside the CfD line.
    # On an as-built wholesale HPC tied at zero with every other CfD and was curtailed
    # 1.31 TWh in 2030, which the economic-curtailment rule then paid; under the
    # contract-aware order it bids minus its strike and is not curtailed there, and
    # the rule is off. The energy the bill pays HPC on is the same either way.
    curtailed = sum(float(row["curtailed_mwh"]) for row in diagnostics.get("cfd_economic_curtailment_by_strike") or []
                    if row.get("carrier") == "nuclear") / 1e6
    assert 17.0 < diagnostics["nuclear_cfd_settled_twh"] + curtailed < 22.0
    assert diagnostics["nuclear_cfd_in_bill_gbp_per_3100kwh"] > 10.0


def test_cp30_scenarios_declare_the_vintage_schedule():
    import sys
    sys.path.insert(0, str(Path("config").resolve()))
    from config_loader import load_config
    scenarios = load_config()["scenarios"]
    name = ("CP30_FFR_2030_Jan_cp30_network")
    subsidy = scenarios[name]["support"]["labels"]
    assert subsidy["cfd_vintage_schedule_file"] == VINTAGE_SCHEDULE_PATH
    assert subsidy["enabled"] is True
