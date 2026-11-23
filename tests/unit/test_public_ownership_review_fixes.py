"""Review fixes of 26 Sep 2026: LDES floor, existing pumped hydro, BECCS like for like, CM price year, merge check.

* The LDES floor nets the store's Capacity Market revenue (the CM line already
  pays it) and floors new build only.
* Existing pumped hydro that CP30 labels LAES is registered as legacy
  ``pumped_hydro`` at its named bus, the expansion above it as new-build LAES.
* Under at-cost re-pricing a split contract (Power BECCS) gives the bill its
  electricity leg's share of required revenue, not the capture leg too.
* The CM rates, in 2025 money, are carried onto the register's 2024 prices.
* The balancing-group merge checks each bill's lines against its total.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership.build_asset_finance_register import (
    EXISTING_SITE_EXPANSION_METHOD, EXISTING_SITE_VINTAGE_METHOD, _split_post_policy_cohorts,
    existing_storage_site_flags, parse_existing_storage_sites, split_existing_storage_sites,
)
from scripts.public_ownership.calculate_financing_scenarios import (
    CM_RATES_PRICE_YEAR, capacity_market_rates_in_price_year,
)
from scripts.public_ownership.cfd_repricing import per_row_savings, repricing_settings
from scripts.public_ownership.merge_balancing_groups import merge
from scripts.public_ownership.public_account import account_settings, build_public_account
from scripts.public_ownership.settlement import attach_settlement

from tests.unit.test_merge_balancing_groups import CASE, bills  # noqa: F401  (fixture)
from tests.unit.test_public_ownership_public_account import _settled_register
from config.config_loader import load_defaults

SITES = [
    {"name": "Dinorwig", "bus": "DINO41", "network_carrier": "LAES", "existing_mw": 1728, "register_carrier": "pumped_hydro"},
    {"name": "Cruachan", "bus": "CRUA2R", "network_carrier": "LAES", "existing_mw": 440, "register_carrier": "pumped_hydro"},
]


@pytest.fixture(scope="module")
def defaults() -> dict:
    return load_defaults()


# ----------------------------------------------------------------------------- LDES floor


def test_ldes_floor_nets_capacity_market_revenue(defaults):
    register, _ = _settled_register()
    settings = account_settings(defaults["public_ownership"])
    savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    _, without = build_public_account(register, savings, settings)
    with_cm = register.copy()
    with_cm["capacity_market_revenue_gbp"] = np.where(with_cm["asset_id"].eq("ldes"), 2.0e6, 0.0)
    _, report = build_public_account(with_cm, savings, settings)
    for case_id in ("private", "public_ownership"):
        # Both floors fall by exactly the CM revenue while they stay positive.
        assert without[case_id]["ldes_floor_topup_at_wacc_gbp"] > 2.0e6
        assert report[case_id]["ldes_floor_topup_at_wacc_gbp"] == pytest.approx(
            without[case_id]["ldes_floor_topup_at_wacc_gbp"] - 2.0e6)
        assert report[case_id]["ldes_floor_topup_gbp"] == pytest.approx(
            max(without[case_id]["ldes_floor_topup_gbp"] - 2.0e6, 0.0))
        assert report[case_id]["ldes_capacity_market_revenue_netted_gbp"] == pytest.approx(2.0e6)


def test_ldes_floor_is_for_new_build_only(defaults):
    register, _ = _settled_register()
    settings = account_settings(defaults["public_ownership"])
    savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    legacy = register.copy()
    legacy.loc[legacy["asset_id"].eq("ldes"), "finance_vintage"] = "legacy"
    _, report = build_public_account(legacy, savings, settings)
    assert report["private"]["ldes_floor_topup_gbp"] == 0.0
    assert report["private"]["ldes_floor_topup_at_wacc_gbp"] == 0.0


# ----------------------------------------------------------------------------- existing pumped hydro


def _storage_register():
    base = {"model_year": 2030, "component_type": "StorageUnit", "commissioning_year": np.nan,
            "development_status": None, "support_type": None, "network_support_type": None, "cfd_round": None,
            "cfd_award_year": np.nan, "contract_vintage": "not_applicable", "data_source": None,
            "capacity_aggregation_method": "static_consistent_across_reference_runs",
            "finance_vintage": "new_build", "vintage_method": "cp30_source_attribute"}
    rows = [
        dict(base, asset_id="CP30_LAES_3_572", carrier="LAES", bus="DINO41", capacity_mw=1550.0,
             energy_capacity_mwh=19515.0, dispatch_mwh=428932.0),
        dict(base, asset_id="CP30_LAES_98_667", carrier="LAES", bus="CRUA2R", capacity_mw=672.0,
             energy_capacity_mwh=8457.0, dispatch_mwh=111832.0),
        dict(base, asset_id="CP30_LAES_2_571", carrier="LAES", bus="FROD41", capacity_mw=1357.0,
             energy_capacity_mwh=17083.0, dispatch_mwh=379363.0),
    ]
    return pd.DataFrame(rows)


def test_existing_pumped_hydro_is_legacy_and_the_expansion_new_build():
    register = split_existing_storage_sites(_storage_register(), SITES)
    by_id = register.set_index("asset_id")
    # Dinorwig: the model's 1,550 MW is all inside the existing 1,728 MW.
    dinorwig = by_id.loc["CP30_LAES_3_572::legacy"]
    assert (dinorwig["carrier"], dinorwig["finance_vintage"], dinorwig["vintage_method"]) == (
        "pumped_hydro", "legacy", EXISTING_SITE_VINTAGE_METHOD)
    assert dinorwig["capacity_mw"] == pytest.approx(1550.0) and dinorwig["cohort_legacy_share"] == 1.0
    assert "CP30_LAES_3_572::new_build" not in by_id.index
    # Cruachan: 440 MW existing, the other 232 MW is new-build LAES; capacity, energy and dispatch split pro rata.
    legacy, expansion = by_id.loc["CP30_LAES_98_667::legacy"], by_id.loc["CP30_LAES_98_667::new_build"]
    assert legacy["capacity_mw"] == pytest.approx(440.0) and expansion["capacity_mw"] == pytest.approx(232.0)
    assert expansion["carrier"] == "LAES" and expansion["vintage_method"] == EXISTING_SITE_EXPANSION_METHOD
    assert legacy["dispatch_mwh"] + expansion["dispatch_mwh"] == pytest.approx(111832.0)
    assert legacy["cohort_legacy_share"] == pytest.approx(440.0 / 672.0) == expansion["cohort_legacy_share"]
    # Any other LAES is untouched.
    assert by_id.loc["CP30_LAES_2_571", "carrier"] == "LAES"
    # The baseline split keeps the site shares (the 2024 run has no LAES and no pumped_hydro carrier).
    split = _split_post_policy_cohorts(register, pd.Series(dtype=float))
    kept = split.set_index("asset_id")
    assert kept.loc["CP30_LAES_98_667::legacy", "cohort_legacy_share"] == pytest.approx(440.0 / 672.0)
    assert kept.loc["CP30_LAES_2_571::new_build", "cohort_legacy_share"] == 0.0
    # The settlement splits by the same share and sums back to the unit.
    settlement = pd.DataFrame({"component_type": "StorageUnit", "asset_id": ["CP30_LAES_98_667"],
                               "market_revenue_gbp": [672.0], "support_revenue_gbp": [0.0], "fuel_carbon_gbp": [0.0]})
    settled = attach_settlement(split, settlement).set_index("asset_id")
    assert settled.loc["CP30_LAES_98_667::legacy", "market_revenue_gbp"] == pytest.approx(440.0)
    assert settled.loc["CP30_LAES_98_667::new_build", "market_revenue_gbp"] == pytest.approx(232.0)


def test_existing_site_spread_over_sibling_busbars_is_legacy_and_named_units_go_first():
    """Fix S3: the build places Cruachan as its own 440 MW unit, shared over CRUA2Q and CRUA2R."""
    base = _storage_register().iloc[0].to_dict()
    rows = [
        dict(base, asset_id="CP30_LAES_existing_Cruachan_570", bus="CRUA2Q", capacity_mw=283.0,
             energy_capacity_mwh=4567.0, dispatch_mwh=100.0),
        dict(base, asset_id="CP30_LAES_40_609", bus="CRUA2R", capacity_mw=300.0, energy_capacity_mwh=4500.0,
             dispatch_mwh=50.0),
        dict(base, asset_id="CP30_LAES_existing_Cruachan_CRUA2R_680", bus="CRUA2R", capacity_mw=157.0,
             energy_capacity_mwh=2533.0, dispatch_mwh=60.0),
    ]
    register = split_existing_storage_sites(pd.DataFrame(rows), [SITES[1]]).set_index("asset_id")
    legacy = register[register["finance_vintage"].eq("legacy")]
    assert set(legacy.index) == {"CP30_LAES_existing_Cruachan_570::legacy",
                                 "CP30_LAES_existing_Cruachan_CRUA2R_680::legacy"}
    assert legacy["capacity_mw"].sum() == pytest.approx(440.0)
    assert (legacy["carrier"] == "pumped_hydro").all()
    # The unrelated new-build LDES unit at the same substation is untouched.
    assert register.loc["CP30_LAES_40_609", "carrier"] == "LAES"
    assert register.loc["CP30_LAES_40_609", "finance_vintage"] == "new_build"


def test_existing_site_config_round_trips_through_the_rule_flags_and_refuses_a_missing_bus():
    flags = existing_storage_site_flags(SITES)
    assert flags.startswith("--existing-storage-site ")
    assert parse_existing_storage_sites(flags.split()[1:]) == parse_existing_storage_sites(SITES)
    assert existing_storage_site_flags(None) == ""
    moved = [dict(SITES[0], bus="PENT41")]
    with pytest.raises(ValueError, match="no LAES StorageUnit at bus 'PENT41'"):
        split_existing_storage_sites(_storage_register(), moved)
    # A sibling busbar of the same substation still matches (fix S3, 27 Sep 2026).
    sibling = split_existing_storage_sites(_storage_register(), [dict(SITES[0], bus="DINO42")])
    assert "CP30_LAES_3_572::legacy" in set(sibling["asset_id"])
    # A run that does not use the LAES label at all (the historical runs) is left alone.
    historical = _storage_register().assign(carrier="Pumped Storage Hydroelectricity")
    assert split_existing_storage_sites(historical, SITES).equals(historical)
    with pytest.raises(ValueError, match="repeat a bus"):
        parse_existing_storage_sites([SITES[0], dict(SITES[0], name="Copy")])


# ----------------------------------------------------------------------------- BECCS like for like


def _beccs_register(strike_basis="at_cost"):
    """Power BECCS on the CP30 year's register values beside an ordinary AR7 farm."""
    common = {"component_type": "Generator", "support_type": "CfD", "network_support_type": "CfD",
              "finance_vintage": "new_build", "contract_vintage": "post_policy", "evidence_status": "resolved",
              "remaining_life_years": 25.0, "cfd_award_year": 2027}
    beccs = dict(common, asset_id="Drax BECCS::new_build", carrier="CCS Biomass", cfd_round="POWER_BECCS",
                 dispatch_mwh=3_596_850.0, market_revenue_gbp=139_918_828.0, support_revenue_gbp=401_026_110.0,
                 fuel_carbon_gbp=534_389_143.0)
    wind = dict(common, asset_id="ar7", carrier="wind_offshore", cfd_round="AR7", dispatch_mwh=4.0e6,
                market_revenue_gbp=2.0e8, support_revenue_gbp=1.5e8, fuel_carbon_gbp=0.0)
    rows = []
    for asset, private_rr, case_rr in ((beccs, 299_730_463.0, 231_492_444.0), (wind, 3.0e8, 2.0e8)):
        rows.append(dict(asset, case_id="private", annual_required_revenue_gbp=private_rr,
                         annual_capital_recovery_gbp=0.6 * private_rr, wacc=0.082, ownership_share=0.0,
                         cfd_strike_basis="financing_saving_only"))
        # Economic-life amortisation, as in the P2 bundle: since 26 Sep 2026 the at-cost
        # strike honours cfd_amortisation, and this test pins the life-long rent.
        rows.append(dict(asset, case_id="public_ownership", annual_required_revenue_gbp=case_rr,
                         annual_capital_recovery_gbp=0.6 * case_rr, wacc=0.035, ownership_share=1.0,
                         cfd_strike_basis=strike_basis, cfd_amortisation="economic_life"))
    return pd.DataFrame(rows)


def test_beccs_at_cost_takes_only_the_electricity_leg(defaults):
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    assert "CCS Biomass" in settings["electricity_leg_carriers"]
    savings = per_row_savings(_beccs_register(), settings).set_index(["case_id", "asset_id"])
    row = savings.loc[("public_ownership", "Drax BECCS::new_build")]
    contract = 139_918_828.0 + 401_026_110.0 - 534_389_143.0
    share = contract / 299_730_463.0
    assert row["electricity_leg_share"] == pytest.approx(share)
    # The electricity leg's financing saving: positive, where charging all of required revenue raised the strike.
    assert row["saving_at_cost_gbp"] == pytest.approx(contract * (1.0 - 231_492_444.0 / 299_730_463.0))
    assert row["saving_at_cost_gbp"] > 0
    # The ordinary farm is unchanged: share 1, the whole rent.
    wind = savings.loc[("public_ownership", "ar7")]
    assert wind["electricity_leg_share"] == 1.0
    assert wind["saving_at_cost_gbp"] == pytest.approx(3.5e8 - 2.0e8)
    # With no split carriers configured, the old basis (the whole required revenue) comes back.
    block = deepcopy(defaults["public_ownership"]["consumer_bill"])
    block["cfd_repricing"]["at_cost_electricity_leg_carriers"] = []
    old = per_row_savings(_beccs_register(), repricing_settings(block)).set_index(["case_id", "asset_id"])
    old_row = old.loc[("public_ownership", "Drax BECCS::new_build")]
    assert old_row["saving_at_cost_gbp"] == pytest.approx(contract - 231_492_444.0)
    # 153.13 -> 215.67 GBP/MWh on the CP30 year: a 62.54 rise, now a small cut.
    assert -old_row["saving_at_cost_gbp"] / 3_596_850.0 == pytest.approx(62.54, abs=0.01)


# ----------------------------------------------------------------------------- CM price year


def test_capacity_market_rates_are_carried_onto_the_register_price_year():
    rates = pd.DataFrame({"carrier": ["CCGT"], "gbp_per_kw_connection_year": [38.62],
                          "connection_mw_2029_30": [23223.6], "calendar_2030_revenue_gbp": [8.97e8]})
    factor = 1.038374717833
    out = capacity_market_rates_in_price_year(rates, 2024, (2024, 2025, factor))
    assert out["gbp_per_kw_connection_year"].iloc[0] == pytest.approx(38.62 / factor)
    assert out["connection_mw_2029_30"].iloc[0] == pytest.approx(23223.6)  # volumes are not deflated
    assert capacity_market_rates_in_price_year(rates, CM_RATES_PRICE_YEAR, None).equals(rates)
    with pytest.raises(ValueError, match="--capacity-market-rebasing"):
        capacity_market_rates_in_price_year(rates, 2024, None)
    with pytest.raises(ValueError, match="2024 to 2025"):
        capacity_market_rates_in_price_year(rates, 2024, (2023, 2025, factor))


# ----------------------------------------------------------------------------- merge check


def test_merge_refuses_lines_that_do_not_match_their_total(bills):  # noqa: F811
    market, market_summary, group, group_summary = bills
    stale = group.copy()
    stale.loc[stale["case_id"].eq(CASE) & stale["component_id"].eq("cfd"), "annual_gbp_per_3100kwh_ex_vat"] += 0.5
    with pytest.raises(ValueError, match="not the same projection"):
        merge(market, market_summary, [{"name": "p2", "by_case": stale, "summary": group_summary, "decomposition": None}])


# ----------------------------------------------------------------------------- follow-ups (26 Sep 2026)


def test_beccs_on_the_financing_basis_passes_only_the_electricity_leg(defaults):
    """Follow-up 1: the capture leg's financing saving stays off the electricity strike."""
    settings = repricing_settings(defaults["public_ownership"]["consumer_bill"])
    register = _beccs_register(strike_basis="financing_saving_only")
    savings = per_row_savings(register, settings).set_index(["case_id", "asset_id"])
    row = savings.loc[("public_ownership", "Drax BECCS::new_build")]
    share = row["electricity_leg_share"]
    assert 0.0 < share < 0.05
    full = row["saving_bill_term_gbp"] + row["saving_off_bill_leg_gbp"]
    assert row["saving_bill_term_gbp"] == pytest.approx(share * full)
    assert row["saving_off_bill_leg_gbp"] == pytest.approx((1.0 - share) * full)
    # The register saving the route carries is unchanged, so the identity still closes.
    assert row["saving_economic_life_gbp"] == pytest.approx(299_730_463.0 - 231_492_444.0)
    wind = savings.loc[("public_ownership", "ar7")]
    assert wind["saving_off_bill_leg_gbp"] == 0.0


def test_existing_pumped_hydro_is_costed_on_the_network_carriers_mapping():
    """Follow-up 2: the legacy cohort is finance-eligible (acquired and costed), not held unchanged."""
    from scripts.public_ownership.build_asset_finance_register import cost_existing_storage_sites

    crosswalk = pd.DataFrame([
        {"component_type": "StorageUnit", "pypsa_carrier": "LAES",
         "workbook_technology": "Novel long duration energy storage (LDES)", "finance_eligible": True},
        {"component_type": "StorageUnit", "pypsa_carrier": "pumped_hydro", "workbook_technology": np.nan,
         "finance_eligible": False},
    ])
    register = split_existing_storage_sites(_storage_register(), SITES)
    register = register.merge(crosswalk, how="left", left_on=["component_type", "carrier"],
                              right_on=["component_type", "pypsa_carrier"]).rename(
        columns={"workbook_technology": "finance_technology"})
    costed = cost_existing_storage_sites(register, crosswalk, SITES).set_index("asset_id")
    legacy = costed.loc["CP30_LAES_98_667::legacy"]
    assert legacy["carrier"] == "pumped_hydro" and legacy["finance_eligible"]
    assert legacy["finance_technology"] == "Novel long duration energy storage (LDES)"
    assert legacy["finance_technology_source"] == "existing_site_network_carrier:LAES"
    # The expansion keeps its own (LAES) mapping; nothing else is touched.
    assert costed.loc["CP30_LAES_98_667::new_build", "finance_technology_source"] is None
    with pytest.raises(ValueError, match="no finance technology"):
        cost_existing_storage_sites(register, crosswalk.iloc[1:], SITES)


def test_merchant_rent_includes_capacity_market_revenue(defaults):
    """Follow-up 4: the state owner's rent, and so the recycled surplus, includes its CM income."""
    register, _ = _settled_register()
    settings = account_settings(defaults["public_ownership"])
    savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    _, without = build_public_account(register, savings, settings)
    with_cm = register.copy()
    with_cm["capacity_market_revenue_gbp"] = np.where(with_cm["asset_id"].eq("solar"), 3.0e6, 0.0)
    _, report = build_public_account(with_cm, savings, settings)
    share = float(register.loc[register["case_id"].eq("public_ownership") & register["asset_id"].eq("solar"),
                               "ownership_share"].iloc[0])
    assert share > 0
    assert report["public_ownership"]["merchant_generation_surplus_gbp"] == pytest.approx(
        without["public_ownership"]["merchant_generation_surplus_gbp"] + share * 3.0e6)
    assert report["private"]["merchant_generation_surplus_gbp"] == pytest.approx(
        without["private"]["merchant_generation_surplus_gbp"])


def test_register_producer_rent_includes_capacity_market_revenue():
    """Follow-up 4, register side: producer rent and the pre-recycling surplus carry CM income too."""
    from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
    from tests.unit.test_public_ownership_cfd_repricing import _asset, _costs, _snapshot

    assets = pd.DataFrame([_asset("gas", "CCGT", "Gas generation - unabated", "new_build", "", "",
                                  "not_applicable", np.nan, 400.0, 1.0e6)])
    settlement = pd.DataFrame({"component_type": ["Generator"], "asset_id": ["gas"], "market_revenue_gbp": [1.2e8],
                               "support_revenue_gbp": [0.0], "fuel_carbon_gbp": [1.0e8]})
    base = attach_settlement(assets, settlement)
    with_cm = base.assign(capacity_market_revenue_gbp=5.0e6)
    kwargs = dict(price_year=2024, state_operating_overhead={"enabled": False})
    plain, _ = calculate_financing_scenarios(base, _costs(), _snapshot(), **kwargs)
    booked, _ = calculate_financing_scenarios(with_cm, _costs(), _snapshot(), **kwargs)
    for case_id in ("private", "public_ownership"):
        a = plain.loc[plain["case_id"].eq(case_id)].iloc[0]
        b = booked.loc[booked["case_id"].eq(case_id)].iloc[0]
        assert b["producer_rent_gbp"] == pytest.approx(a["producer_rent_gbp"] + 5.0e6)
        assert b["public_surplus_before_recycling_gbp"] == pytest.approx(
            a["public_surplus_before_recycling_gbp"] + 5.0e6 * b["ownership_share"])


def test_a_second_unit_with_its_own_settlement_row_takes_its_own_money():
    """F27 item 11 (23 Oct 2026): both Drax biomass rows had carried the first unit's settlement."""
    register = pd.DataFrame({"component_type": "Generator",
                             "asset_id": ["Drax Power Station (biomass units)",
                                          "Drax Power Station (biomass units)::DRAX42", "Wind::legacy", "Wind::new_build"],
                             "cohort_legacy_share": [np.nan, np.nan, 0.25, 0.25]})
    settlement = pd.DataFrame({"component_type": "Generator",
                               "asset_id": ["Drax Power Station (biomass units)",
                                            "Drax Power Station (biomass units)::DRAX42", "Wind"],
                               "market_revenue_gbp": [395.0, 200.0, 100.0], "support_revenue_gbp": [291.0, 146.0, 0.0],
                               "fuel_carbon_gbp": [677.0, 341.0, 0.0]})
    settled = attach_settlement(register, settlement).set_index("asset_id")
    assert settled["support_revenue_gbp"].tolist() == [291.0, 146.0, 0.0, 0.0]
    assert settled["market_revenue_gbp"].tolist() == [395.0, 200.0, 25.0, 75.0]
