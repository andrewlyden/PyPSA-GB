"""The 29 September 2026 user decisions on the public-ownership layer.

One test (or a small group) per decision and per new lever: the auction
calibration (1), the BECCS carbon-removal CfD (2), state deficits (3), the
household total (4), the presets' resource invariant (5), the guarantee's
pass-through (6), the premium over RAV (7), Hinkley Point C completed by the
state (10), and the Capacity Market floor (12); plus the itemised party flows,
the background levy switches and the register toggles.  The expansion itself
(presets, bounds, grids, retired levers) is in test_public_ownership_levers.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership import party_flows as pf
from scripts.public_ownership.calculate_financing_scenarios import (
    LEVER_PARAMETER_DEFAULTS, apply_cost_calibration, book_carbon_removal_cfd, hpc_completion_price,
    lever_parameters, load_cost_calibration, undelivered_rows,
)
from scripts.public_ownership.cfd_repricing import per_row_savings, repricing_settings
from scripts.public_ownership.exchequer_ledger import (
    check_resource_invariant, contested_fiscal_costs, legal_incidence_per_household, load_contested_costs,
    load_tax_incidence, per_household,
)
from scripts.public_ownership.finance import capital_recovery_factor
from scripts.public_ownership.nationalised_tariff import (
    IC_ACQUISITION_DEFAULTS, break_even_purchase_price, headline_years, interconnector_acquisition_settings,
    interconnector_price_settings,
)
from scripts.public_ownership.network_finance_account import (
    incidence_amounts, load_pass_through, load_premiums, select_rows, settings as network_settings,
)
from scripts.public_ownership.project_public_ownership_consumer_bill import (
    DOMESTIC_LEVY_LINES, LINE_BECCS_LEVY, background_rows, offset_split,
)
from scripts.public_ownership.public_account import account_settings, build_public_account

from tests.unit.test_public_ownership_cfd_repricing import _synthetic_register
from tests.unit.test_public_ownership_public_account import _central, _settled_register
from config.config_loader import load_defaults

FINANCING = Path("data/public_ownership/financing")


@pytest.fixture(scope="module")
def defaults() -> dict:
    return load_defaults()


# ── decision 1: trust the auction ───────────────────────────────────────────


def test_the_calibration_table_is_valid_and_scales_only_its_cohorts():
    table = load_cost_calibration(FINANCING / "private_cost_calibration.csv")
    # One factor per (carrier, round), every finance vintage pooled (user, 6 Nov 2026).
    assert "finance_vintage" not in table.columns and not table.duplicated(["carrier", "cfd_round"]).any()
    factors = table.set_index(["carrier", "cfd_round"])["capex_factor"]
    # Re-derived on the cp30_2030_no_acceleration register of the single re-solve (6 Nov 2026, F28 and workbook v6.2).
    # 11 Nov 2026: the table was re-derived again on that register (derive_cost_calibration.py --out, committed in
    # 65ff9c72 after the 10 Nov decisions): AR7 0.897027 -> 0.896697, AR4 0.341925 -> 0.341773.
    assert factors[("wind_offshore", "AR7")] == pytest.approx(0.896697)
    assert factors[("wind_offshore", "AR4")] == pytest.approx(0.341773)
    # F19 decision 2 (6 Oct): onshore and solar AR7a and AR8, calibrated both ways (k > 1 on the re-solve).
    assert set(table["carrier"]) == {"wind_offshore", "marine", "wind_onshore", "solar_pv"}
    both_ways = table.loc[table["carrier"].isin(["wind_onshore", "solar_pv"])]
    assert set(both_ways["cfd_round"]) == {"AR7a", "AR8_modelled"} and (both_ways["capex_factor"] > 1.0).all()
    # Second round (29 Sep): the earlier offshore rounds above their strike.
    assert {r for c, r in factors.index if c == "wind_offshore"} == {"AR2", "AR3", "AR4", "AR6", "AR7", "AR8_modelled"}
    # A round's factor applies to every vintage the register gives it (the legislation year sets the vintage).
    merged = pd.DataFrame({
        "carrier": ["wind_offshore", "wind_offshore", "wind_offshore", "marine"],
        "cfd_round": ["AR7", "AR6", "AR7", "AR4"],
        "finance_vintage": ["new_build", "new_build", "legacy", "new_build"],
        "capex_power_gbp_per_kw": [2000.0, 2000.0, 2000.0, 5000.0], "capex_energy_gbp_per_kwh": [0.0] * 4,
    })
    small = table.loc[((table["carrier"] == "wind_offshore") & (table["cfd_round"] == "AR7"))
                      | ((table["carrier"] == "marine") & (table["cfd_round"] == "AR4"))]
    out = apply_cost_calibration(merged, small)
    assert out["capex_power_gbp_per_kw"].tolist() == pytest.approx(
        [2000.0 * factors[("wind_offshore", "AR7")], 2000.0, 2000.0 * factors[("wind_offshore", "AR7")],
         5000.0 * factors[("marine", "AR4")]])
    # A cohort in the table with no row in the fleet means the two have drifted apart.
    with pytest.raises(ValueError, match="no row of wind_offshore AR8_modelled"):
        apply_cost_calibration(merged, table)


def test_a_table_keyed_on_finance_vintage_is_refused(tmp_path):
    """6 Nov 2026: the per-vintage key (before the 2025 legislation bound) is retired with a pointer."""
    path = tmp_path / "old.csv"
    pd.DataFrame({"carrier": ["wind_offshore"], "cfd_round": ["AR7"], "finance_vintage": ["legacy"],
                  "capex_factor": [0.9]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one factor per"):
        load_cost_calibration(path)


def test_a_capex_factor_out_of_range_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"carrier": ["x"], "cfd_round": ["AR7"], "capex_factor": [3.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="capex_factor"):
        load_cost_calibration(path)


# ── decision 2: the BECCS carbon-removal CfD ────────────────────────────────


def _beccs_register():
    base = {"carrier": "CCS Biomass", "component_type": "Generator", "asset_id": "beccs",
            "market_revenue_gbp": 60.0, "support_revenue_gbp": 20.0, "fuel_carbon_gbp": 30.0}
    rows = [dict(base, case_id="private", annual_required_revenue_gbp=100.0, ownership_share=0.0,
                 producer_rent_gbp=50.0 - 100.0, public_surplus_before_recycling_gbp=0.0),
            dict(base, case_id="public_ownership", annual_required_revenue_gbp=70.0, ownership_share=1.0,
                 producer_rent_gbp=50.0 - 70.0, public_surplus_before_recycling_gbp=50.0 - 70.0)]
    return pd.DataFrame(rows)


def test_the_capture_leg_is_paid_by_its_own_contract_so_no_owner_carries_it():
    out = book_carbon_removal_cfd(_beccs_register(), ["CCS Biomass"]).set_index("case_id")
    # Leg share = private contract revenue / private required revenue = 50 / 100.
    assert out["electricity_leg_share"].tolist() == pytest.approx([0.5, 0.5])
    assert out.loc["private", "carbon_removal_cfd_revenue_gbp"] == pytest.approx(50.0)
    assert out.loc["public_ownership", "carbon_removal_cfd_revenue_gbp"] == pytest.approx(35.0)
    # The private electricity leg breaks even by construction; the state carries no capture deficit.
    assert out.loc["private", "producer_rent_gbp"] == pytest.approx(0.0)
    assert out.loc["public_ownership", "producer_rent_gbp"] == pytest.approx(15.0)
    assert out.loc["public_ownership", "public_surplus_before_recycling_gbp"] == pytest.approx(15.0)
    # Nothing is booked for a carrier outside the split contract.
    untouched = book_carbon_removal_cfd(_beccs_register(), [])
    assert untouched["carbon_removal_cfd_revenue_gbp"].eq(0.0).all()


# ── the register toggles ─────────────────────────────────────────────────────


def test_lever_parameter_code_defaults_equal_the_repository_config(defaults):
    """House rule: a missing config block cannot move a number."""
    block = defaults["public_ownership"]["lever_parameters"]
    assert lever_parameters(None) == lever_parameters(block)
    assert {k: v for k, v in block.items()} == {k: (dict(v) if isinstance(v, dict) else v)
                                                 for k, v in LEVER_PARAMETER_DEFAULTS.items()}
    ic = defaults["public_ownership"]["acquisition"]["interconnectors"]
    assert interconnector_acquisition_settings(ic) == interconnector_acquisition_settings(None) | {
        k: ic[k] for k in ic}
    assert IC_ACQUISITION_DEFAULTS["owner_real_rate_by_setting"] == ic["owner_real_rate_by_setting"]


def test_the_interconnector_price_lever_swaps_only_the_owner_rate():
    central = interconnector_acquisition_settings(None)
    assert interconnector_price_settings(central, None, "central") == central
    low = interconnector_price_settings(central, {"owner_real_rate_by_setting": {"low": 0.083}}, "low")
    assert low["owner_real_rate"] == 0.083 and {k for k in central if central[k] != low[k]} == {"owner_real_rate"}
    with pytest.raises(ValueError, match="owner_real_rate_by_setting"):
        interconnector_price_settings(central, {}, "high")


def test_half_delivery_splits_each_carrier_by_capacity_deterministically():
    merged = pd.DataFrame({
        "asset_id": ["a", "b", "c", "d", "e"], "carrier": ["wind", "wind", "wind", "solar", "solar"],
        "finance_vintage": ["new_build"] * 4 + ["legacy"], "evidence_status": ["resolved"] * 5,
        "capacity_mw": [100.0, 100.0, 200.0, 50.0, 50.0],
    })
    # Wind: midpoints 50, 150, 300 of 400 -> c lies past half; solar: one new-build row, midpoint 25 of 50.
    assert undelivered_rows(merged, 0.5) == {2}
    assert undelivered_rows(merged, 0.5) == undelivered_rows(merged.sample(frac=1.0, random_state=1), 0.5)


def test_register_toggles_move_the_state_rows_and_never_private():
    base, _ = _synthetic_register()
    for levers, column in (({"state_capex_overrun": "plus_15pct"}, "annual_capital_recovery_gbp"),
                           ({"state_om_premium": "plus_10pct"}, "annual_fixed_om_gbp")):
        moved, _ = _synthetic_register(axes=_central(**levers))
        key = ["case_id", "asset_id"]
        a, b = base.set_index(key)[column].dropna(), moved.set_index(key)[column].dropna()
        assert (b.loc["private"] - a.loc["private"]).abs().max() == pytest.approx(0.0)
        assert (b.loc["public_ownership"] >= a.loc["public_ownership"] - 1e-6).all()
        assert (b.loc["public_ownership"] - a.loc["public_ownership"]).max() > 0, levers


def test_a_tax_equivalent_is_the_corporation_tax_a_private_owner_would_pay(monkeypatch):
    import scripts.public_ownership.calculate_financing_scenarios as cfs

    wedge = {"private": 0.02, "public_coordination": 0.015}
    monkeypatch.setattr(cfs, "corporation_tax_wedges", lambda snapshot: {
        t: wedge for t in ("Offshore wind", "Solar PV", "Gas generation - unabated", "Lithium batteries")})
    base, _ = _synthetic_register()
    taxed, summary = _synthetic_register(axes=_central(tax_equivalent="on"))
    rows = taxed.loc[taxed.case_id.eq("public_ownership") & taxed.finance_vintage.eq("new_build")
                     & taxed.annual_tax_equivalent_gbp.gt(0)].set_index("asset_id")
    assert len(rows)
    assert rows["annual_tax_equivalent_gbp"].tolist() == pytest.approx(rows["corporation_tax_forgone_gbp"].tolist())
    before = base.loc[base.case_id.eq("public_ownership")].set_index("asset_id")["annual_required_revenue_gbp"]
    assert (rows["annual_required_revenue_gbp"] - before.reindex(rows.index)).tolist() == pytest.approx(
        rows["annual_tax_equivalent_gbp"].tolist())
    assert taxed.loc[taxed.case_id.eq("private"), "annual_tax_equivalent_gbp"].fillna(0.0).eq(0.0).all()


# ── decisions 3 and 12: state deficits and the Capacity Market floor ─────────


def _negative_cm(register):
    """Make the state's Capacity Market plant dearer than private's, so its booking turns negative."""
    register = register.copy()
    private = register.loc[register.case_id.eq("private")].set_index("asset_id")["annual_required_revenue_gbp"]
    rows = register.case_id.eq("public_ownership") & register.asset_id.isin(["gas", "battery"])
    register.loc[rows, "annual_required_revenue_gbp"] = register.loc[rows, "asset_id"].map(private) * 1.5
    return register


@pytest.mark.parametrize("rule, floored", [("exchequer", True), ("bill_levy", False)])
def test_a_negative_capacity_market_booking_is_floored_unless_deficits_go_on_bills(defaults, rule, floored):
    register, _ = _settled_register(_central(state_deficit_funding=rule, route_booking="booked"))
    register = _negative_cm(register)
    savings = per_row_savings(register, repricing_settings(defaults["public_ownership"]["consumer_bill"]))
    _, report = build_public_account(register, savings, account_settings(defaults["public_ownership"]))
    case = report["public_ownership"]
    assert case["state_deficit_funding"] == rule
    if floored:
        assert case["capacity_market_booked_gbp"] == pytest.approx(0.0)
        assert case["capacity_market_booking_floored_gbp"] < 0
    else:
        assert case["capacity_market_booked_gbp"] < 0
        assert case["capacity_market_booking_floored_gbp"] == pytest.approx(0.0)


def test_state_deficits_go_on_bills_only_under_the_bill_levy():
    from scripts.public_ownership.public_account import GROUP_BATTERY, GROUP_LDES, exchequer_position

    rows = pd.DataFrame({"public_surplus_gbp": [-40.0, 10.0]}, index=[GROUP_LDES, GROUP_BATTERY])
    case = {"r2_cfd_surplus_gbp": 0.0, "r1_retained_rent_gbp": 0.0, "merchant_generation_surplus_gbp": 0.0,
            "shortfall_on_exchequer_gbp": 5.0}
    for rule, on_bill in (("exchequer", 0.0), ("bill_levy", 40.0)):
        out = exchequer_position(dict(case, state_deficit_funding=rule), rows, "public_cfd_at_cost")
        assert out["state_deficit_gross_gbp"] == pytest.approx(40.0)
        assert out["state_deficit_on_bill_gbp"] == pytest.approx(on_bill)
        # A shortfall the shortfall_funding lever sends to the Exchequer stays there either way.
        assert out["state_deficit_exchequer_funded_gbp"] == pytest.approx(40.0 - on_bill + 5.0)


# ── decision 4: the household total ─────────────────────────────────────────


CONV = {"factor": 1.04, "final_demand_mwh": 280e6, "mwh": 3.1, "vat_rate": 0.05, "domestic_mwh": 95e6}


def test_the_legal_incidence_key_is_the_household_share_over_households():
    incidence = load_tax_incidence(FINANCING / "household_tax_incidence.csv")
    assert incidence["theta"] == 0.61 and incidence["theta_low"] < 0.61 < incidence["theta_high"]
    assert incidence["households"] == 29_003_000
    assert legal_incidence_per_household(1e9, CONV, incidence) == pytest.approx(1e9 * 1.04 * 0.61 / 29_003_000)
    # The central key is the bill's own flat allocation over final demand.
    assert per_household(1e9, CONV) == pytest.approx(1e9 * 1.04 / 280e6 * 3.1)


def test_the_contested_fiscal_costs_follow_their_sources():
    contested = load_contested_costs(FINANCING / "contested_fiscal_costs.csv")
    assert contested["marginal_cost_of_public_funds"] == 1.26
    row = {"exchequer_net_vs_private_gbp": -2e9}
    # Workbook v6.2 Build-Out section F: 2 bp per pp of (debt + 25% stress loss) / GDP, on the
    # marketable stock (2,452 bn in 2024, grown with GDP at 1.5%) refinanced since 2025 over 14 years.
    gdp_2030 = 2.905e12 * 1.015 ** 6
    borrowing = 0.01 * gdp_2030 / 1.25
    out = contested_fiscal_costs(row, borrowing, contested, CONV, 2030)
    assert out["contested_gilt_yield_shift_bp"] == pytest.approx(2.0)
    assert out["contested_gilt_stock_externality_gbp"] == pytest.approx(2e-4 * 2.452e12 * 1.015 ** 6 * 5 / 14)
    assert out["contested_mcpf_deadweight_gbp"] == pytest.approx(0.26 * 2e9)
    # By 2039 the whole stock has refinanced.
    late = contested_fiscal_costs(row, borrowing, contested, CONV, 2050)
    assert late["contested_gilt_stock_repriced_gbp"] == pytest.approx(2.452e12 * 1.015 ** 26)
    # A net Exchequer gain has no deadweight.
    assert contested_fiscal_costs({"exchequer_net_vs_private_gbp": 1e9}, 0.0, contested, CONV, 2030)[
        "contested_mcpf_deadweight_gbp"] == 0.0


# ── decision 5: the presets keep the resource saving ─────────────────────────


def test_a_preset_that_moves_the_resource_saving_is_refused():
    rows = [{"case_id": "public_ownership", "resource_saving_vs_private_gbp": 100.0},
            {"case_id": "public_ownership@consumer_first", "resource_saving_vs_private_gbp": 100.4},
            {"case_id": "public_ownership@treasury_first", "resource_saving_vs_private_gbp": 250.0}]
    catalogue = {"public_ownership@consumer_first": {"kind": "preset", "parent": "public_ownership",
                                                     "preset": "consumer_first"}}
    out = check_resource_invariant(rows, catalogue)
    assert out[0]["resource_saving_gap_gbp"] == pytest.approx(0.4)
    catalogue["public_ownership@treasury_first"] = {"kind": "preset", "parent": "public_ownership",
                                                    "preset": "treasury_first"}
    with pytest.raises(ValueError, match="treasury_first"):
        check_resource_invariant(rows, catalogue)


# ── the itemised party flows ─────────────────────────────────────────────────


def _ledger_row(**columns):
    return {"case_id": "x", **columns}


def test_party_flows_close_on_the_three_party_columns_and_the_resource_saving():
    flows = pf.Flows("x", lambda gbp: gbp / 1e6)
    flows.add("finance", "Cheaper finance", "resource", "exchequer", 100.0, "financing")
    flows.add("rebate", "Surplus to bills", "exchequer", "consumers", 60.0, "other")
    flows.add("vat", "VAT", "consumers", "exchequer", -3.0, "VAT")        # negative: written the other way round
    flows.add("tiny", "Rounding", "consumers", "exchequer", 1e-9, "other")  # dropped
    assert [r["payer"] for r in flows.rows] == ["resource", "exchequer", "exchequer"]
    row = _ledger_row(consumer_change_vs_private_gbp=63.0, exchequer_net_vs_private_gbp=37.0,
                      investor_change_vs_private_gbp=0.0, resource_saving_vs_private_gbp=100.0)
    out = pf.check(flows.rows, row)
    assert out["transfer_net_gbp"] == pytest.approx(0.0) and out["resource_gap_gbp"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="missing or doubled"):
        pf.check(flows.rows, dict(row, exchequer_net_vs_private_gbp=40.0))
    with pytest.raises(ValueError, match="resource flows"):
        pf.check(flows.rows, dict(row, resource_saving_vs_private_gbp=90.0))
    with pytest.raises(ValueError, match="unknown party"):
        flows.add("bad", "Bad", "consumers", "somebody", 1.0, "other")


def test_every_bill_line_a_case_can_add_has_a_party():
    for line in (*DOMESTIC_LEVY_LINES.values(), LINE_BECCS_LEVY, "public_state_deficit_levy",
                 "public_surplus_offset_domestic_levies"):
        assert line in pf.BILL_LINE_PARTY, line


# ── the background levers and the repaired offset ────────────────────────────


LEVIES = {"renewables_obligation": 60.0, "feed_in_tariffs": 25.0, "warm_home_discount": 15.0}


def test_the_offset_is_capped_at_the_domestic_levies_and_the_rest_is_rebated():
    cap = sum(v / 3.1 * 95e6 / 1.04 for v in LEVIES.values())
    offset, rebate = offset_split(cap / 2, factor=1.04, domestic_mwh=95e6, mwh=3.1, levy_lines=LEVIES)
    assert (offset, rebate) == pytest.approx((cap / 2, 0.0))
    offset, rebate = offset_split(cap + 5e6, factor=1.04, domestic_mwh=95e6, mwh=3.1, levy_lines=LEVIES)
    assert (offset, rebate) == pytest.approx((cap, 5e6))
    # Once the levies have moved to taxation there is nothing left to offset.
    assert offset_split(1e6, factor=1.04, domestic_mwh=95e6, mwh=3.1, levy_lines=LEVIES,
                        levies_on_bills=False) == pytest.approx((0.0, 1e6))
    assert offset_split(-1e6, factor=1.04, domestic_mwh=95e6, mwh=3.1, levy_lines=LEVIES) == (0.0, 0.0)


def test_the_background_switches_move_the_bill_and_name_who_pays():
    case = {"sensitivity_axes": {"legacy_levy_funding": "domestic_to_taxation", "beccs_removal_funding": "bill_levy"},
            "carbon_removal_cfd_revenue_gbp": 200e6}
    rows = {r["component_id"]: r for r in background_rows(
        case, factor=1.04, eligible_mwh=250e6, mwh=3.1, domestic_mwh=95e6, levy_lines=LEVIES, account_source="t")}
    assert {DOMESTIC_LEVY_LINES[k] for k in LEVIES} <= set(rows)
    for line, value in LEVIES.items():
        assert rows[DOMESTIC_LEVY_LINES[line]]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(-value)
    assert rows[LINE_BECCS_LEVY]["annual_gbp_per_3100kwh_ex_vat"] == pytest.approx(200e6 * 1.04 / 250e6 * 3.1)
    assert background_rows({"sensitivity_axes": {}}, factor=1.04, eligible_mwh=250e6, mwh=3.1, domestic_mwh=95e6,
                           levy_lines=LEVIES, account_source="t") == []


# ── decisions 6 and 7: the wires ─────────────────────────────────────────────


def test_the_network_evidence_tables_load_with_the_decided_values(defaults):
    cfg = network_settings(defaults["public_ownership"]["network_account"])
    assert load_premiums(cfg) == {"central": 0.5, "low": 0.2, "high": 0.75, "statutory_at_rav": 0.0, "quarter": 0.25}
    shares = load_pass_through(cfg)
    assert shares["full_stock"] == {"transmission": 1.0, "distribution": 1.0}
    assert 0 < shares["new_debt_low"]["transmission"] < shares["new_and_refinanced"]["transmission"] < 1
    # The ED3/RIIO-3 correction: Ofgem's own CPIH-real allowances.
    assert cfg["allowed_wacc_cpih_real"] == {"transmission": 0.0457, "distribution": 0.0435}
    assert cfg["premium_annuity_years"] == 30
    with pytest.raises(ValueError, match="not both"):
        network_settings(dict(defaults["public_ownership"]["network_account"], acquisition_premium_over_rav=0.1))


def _lines():
    rows = []
    for treatment, premium, pass_through in (("acquired", "central", ""), ("acquired", "high", ""),
                                            ("debt_guarantee", "", "new_and_refinanced"),
                                            ("debt_guarantee", "", "full_stock")):
        for sector, line, saving in (("transmission", "tnuos", 100.0), ("distribution", "duos", 200.0)):
            scale = 0.5 if premium == "high" or pass_through == "new_and_refinanced" else 1.0
            tax = 50.0 if (treatment == "acquired" and sector == "distribution") else 0.0
            rows.append({"treatment": treatment, "sector": sector, "bill_line_adjusted": line,
                         "network_acquisition_premium": premium, "public_rate_basis": "",
                         "guarantee_pass_through": pass_through, "financing_saving_gbp": saving * scale,
                         "tax_allowance_saving_gbp": tax, "total_saving_gbp": saving * scale + tax,
                         "household_gbp_per_system_gbp": 1e-8, "guarantee_expected_loss_central_gbp": 10.0})
    return pd.DataFrame(rows)


def test_a_case_reads_the_network_rows_at_its_own_variant():
    lines = _lines()
    central = select_rows(lines, "acquired", {})
    assert set(central["network_acquisition_premium"]) == {"central"} and len(central) == 2
    high = select_rows(lines, "acquired", {"network_acquisition_premium": "high"})
    assert high["financing_saving_gbp"].tolist() == [50.0, 100.0]
    stock = select_rows(lines, "debt_guarantee", {"guarantee_pass_through": "full_stock"})
    assert stock["financing_saving_gbp"].tolist() == [100.0, 200.0]


@pytest.mark.parametrize("treatment, levers, on_bill, fee, retained", [
    ("acquired", {}, 350.0, 0.0, 0.0),
    ("acquired", {"tax_equivalent": "on"}, 300.0, 0.0, 50.0),
    ("acquired", {"network_incidence": "return_retained"}, 0.0, 0.0, 350.0),
    ("debt_guarantee", {}, 150.0, 0.0, 0.0),
    ("debt_guarantee", {"network_incidence": "guarantee_fee_on_bills"}, 130.0, 20.0, 0.0),
    ("debt_guarantee", {"network_incidence": "return_retained"}, 0.0, 150.0, 0.0),
])
def test_network_incidence_splits_the_saving_and_keeps_its_total(treatment, levers, on_bill, fee, retained):
    rows = select_rows(_lines(), treatment, levers)
    amounts = incidence_amounts(rows, treatment, levers)
    assert sum(v["saving_on_bill_gbp"] for v in amounts.values()) == pytest.approx(on_bill)
    assert sum(v["fee_to_exchequer_gbp"] for v in amounts.values()) == pytest.approx(fee)
    assert sum(v["retained_by_exchequer_gbp"] for v in amounts.values()) == pytest.approx(retained)


# ── decision 8 and the break-even price ──────────────────────────────────────


def test_the_multi_year_view_is_on_the_headline_private_bill():
    years = pd.DataFrame({"year": ["2030", "30-year average"], "bill_delta_gbp_per_3100kwh_ex_vat": [-100.0, -40.0]})
    summary = {"private_total_gbp_inc_vat": 1006.32}
    out = headline_years(years, summary, CONV, {"total_gbp_inc_vat": 1024.25})
    assert out["private_bill_gbp_inc_vat"] == 1024.25 and out["basis"] == "against the headline private bill"
    assert out["2030 snapshot"]["bill_gbp_inc_vat"] == pytest.approx(1024.25 - 105.0)
    assert out["30-year average"]["delta_vs_private_gbp_inc_vat"] == pytest.approx(-42.0)


def test_the_break_even_price_sets_the_bill_saving_to_zero():
    summary = {"total_gbp_inc_vat": 900.0, "private_total_gbp_inc_vat": 1000.0}
    diagnostics = {"acquisition_price_gbp": 100e9}
    out = break_even_purchase_price(summary, diagnostics, None, CONV, public_rate=0.0275, term_years=30.0,
                                    headline_private=None, exchequer_funded=False)
    slope = capital_recovery_factor(0.0275, 30.0) * (1.04 / 280e6 * 3.1) * 1.05
    assert out["break_even_purchase_price_bill_gbp"] == pytest.approx(100e9 + 100.0 / slope)
    funded = break_even_purchase_price(summary, diagnostics, None, CONV, public_rate=0.0275, term_years=30.0,
                                       headline_private=None, exchequer_funded=True)
    assert funded["break_even_purchase_price_bill_gbp"] is None


# ── decision 10: Hinkley Point C completed by the state ──────────────────────


def test_the_hpc_completion_price_is_sunk_cost_plus_remaining_capex_at_the_public_rate():
    table = pd.read_csv(FINANCING / "hpc_completion_cost.csv")
    asset = {"asset_id": "Hinkley Point C::legacy", "capacity_mw": 1630.0}
    none = hpc_completion_price(asset, table, public_rate=0.0275, term_years=30.0, premium=0.0)
    assert none["capacity_share_of_project"] == pytest.approx(0.5)
    assert none["price_gbp"] == pytest.approx((25.1e9 + 9.9e9) * 1.339 * 0.5)
    with_premium = hpc_completion_price(asset, table, public_rate=0.0275, term_years=30.0, premium=0.005)
    uplift = capital_recovery_factor(0.0325, 30.0) / capital_recovery_factor(0.0275, 30.0)
    assert with_premium["price_gbp"] == pytest.approx((25.1e9 + 9.9e9 * uplift) * 1.339 * 0.5)
    assert with_premium["construction_premium_capitalised_gbp"] == pytest.approx(9.9e9 * 1.339 * 0.5 * (uplift - 1))
    assert hpc_completion_price({"asset_id": "Sizewell B", "capacity_mw": 1200.0}, table, public_rate=0.0275,
                                term_years=30.0, premium=0.0) is None


def test_the_retired_acquisition_key_and_lever_are_refused(defaults):
    from copy import deepcopy
    from scripts.public_ownership.reference_runs import validate_public_ownership_config

    broken = deepcopy(defaults["public_ownership"])
    broken["acquisition"]["include_in_net_bill"] = False
    with pytest.raises(ValueError, match="acquisition_financing"):
        validate_public_ownership_config(broken)


def test_the_ledger_reads_the_network_tax_allowance_once_at_the_case_variant():
    """The first end-to-end run found the allowance summed over every acquired variant row (x12)."""
    from scripts.public_ownership.exchequer_ledger import network_tax_allowance

    lines = _lines()
    assert network_tax_allowance(lines) == pytest.approx(50.0)
    assert network_tax_allowance(lines, {"network_acquisition_premium": "high"}) == pytest.approx(50.0)


# ── second round (29 Sep 2026): the HPC strike ───────────────────────────────


def test_the_bill_and_the_solves_settle_hpc_at_89_50_from_the_table():
    """OW12 (3 Oct 2026): the 89.50 HPC strike is in the strike table itself, so the
    solves and the bill read one strike and the bill declares no override."""
    import json

    from scripts.market.cfd_strike_table import load_strike_table, strike_overrides

    assumptions = json.loads(Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
                             .read_text(encoding="utf-8"))
    table_path = Path("data/market/lccc/cfd_strike_prices_by_round.csv")
    index = assumptions["cfd_strike_indexation"]
    assert strike_overrides(assumptions) == []
    bill_table = load_strike_table(table_path, index, 2024, overrides=strike_overrides(assumptions))
    solve_table = load_strike_table(table_path, index, 2024)
    assert solve_table.lookup("nuclear", "HPC")[0] == pytest.approx(89.5 * index["factor"])
    assert solve_table.lookup("nuclear", "AR8_modelled")[0] == pytest.approx(89.5 * index["factor"])
    assert bill_table.strikes == solve_table.strikes
    # The former override's evidence is kept as a record.
    record = assumptions["model_linked_rules"]["cfd"]["hpc_strike_reduction"]
    assert (record["strike_price_gbp_per_mwh_2012"], record["replaced"]) == (89.5, 92.5)
    assert set(record["sizewell_c_offset"]) >= {
        "question_a_payment_inside_szc_rab", "question_b_owners_compensated_in_full",
        "question_c_net_saving_to_bill_payers", "sizing_2030"}
    # The override mechanism still guards a stale value (the table no longer holds 92.50).
    stale = {k: record[k] for k in ("cfd_round", "technology", "strike_price_gbp_per_mwh_2012",
                                    "price_basis", "source")}
    with pytest.raises(ValueError, match="replaces"):
        load_strike_table(table_path, index, 2024, overrides=[dict(stale, replaces=92.5)])


# ── second round: HPC on EDF's current schedule (bill-level sensitivity) ──────


def _schedule():
    from scripts.public_ownership import hpc_schedule as hs
    return hs.load_schedule(FINANCING / "hpc_schedule_edf_2025.json")


def test_edf_schedule_output_is_unit_1_from_june_with_a_six_month_ramp():
    from scripts.public_ownership import hpc_schedule as hs

    monthly = hs.edf_monthly_mwh(_schedule())
    assert all(monthly[m] == 0.0 for m in ("2030-01", "2030-05"))
    full = 1630.0 * 0.9 * 24 * 31
    assert monthly["2030-12"] == pytest.approx(full)
    assert 0 < monthly["2030-06"] < monthly["2030-09"] < monthly["2030-12"]
    assert sum(monthly.values()) / 1e6 == pytest.approx(4.32, abs=0.02)
    # EDF's unfavourable scenario (12 months later) has nothing in 2030.
    assert sum(hs.edf_monthly_mwh(_schedule(), "2031-06-01").values()) == 0.0


def test_the_market_rows_replace_the_strike_with_the_market_price_and_close():
    from scripts.public_ownership import hpc_schedule as hs

    prices = {str(m): 50.0 for m in hs.MONTHS}
    shortfall = hs.schedule_shortfall(_schedule(), 18.0e6, prices)
    dq = shortfall["missing_mwh"]
    conv = {"factor": 1.0, "final_demand_mwh": 300e6, "mwh": 3.1}
    out = hs.market_bill_change(shortfall, cfd_payment_gbp=18.0e6 * 80.0, settled_mwh=18.0e6,
                                strike_gbp_per_mwh=125.0, conv=conv, eligible_mwh=300e6)
    assert out["cfd_top_up_gbp_per_mwh"] == pytest.approx(80.0) and out["cfd_reference_gbp_per_mwh"] == pytest.approx(45.0)
    # Together: HPC's strike on dq is replaced by the market price on dq.
    assert out["cfd_line_system_gbp"] + out["wholesale_line_system_gbp"] == pytest.approx(dq * (50.0 - 125.0))


def test_a_tariff_that_owns_hpc_buys_the_energy_and_pays_less_for_the_plant():
    from scripts.public_ownership import hpc_schedule as hs

    prices = {str(m): 50.0 for m in hs.MONTHS}
    shortfall = hs.schedule_shortfall(_schedule(), 18.0e6, prices)
    conv = {"factor": 1.0, "final_demand_mwh": 300e6, "mwh": 3.1}
    out = hs.tariff_change(shortfall, variable_cost_gbp_per_mwh=10.0, annual_flow_gbp=2e9, settled_mwh=18.0e6,
                           basis_rate=0.07, public_rate=0.0275, term_years=30.0, conv=conv)
    assert out["energy_system_gbp"] == pytest.approx(shortfall["missing_mwh"] * 40.0)
    assert out["purchase_price_change_gbp"] < 0 and out["annuity_system_gbp"] < 0
    # The completion price does not depend on output.
    assert hs.tariff_change(shortfall, variable_cost_gbp_per_mwh=10.0, annual_flow_gbp=2e9, settled_mwh=18.0e6,
                            basis_rate=None, public_rate=0.0275, term_years=30.0, conv=conv)["annuity_system_gbp"] == 0.0


def test_a_drifted_calibration_names_every_cohort_and_how_to_re_derive():
    """A table round the fleet does not hold is named, with every other such round and how to re-derive."""
    table = pd.DataFrame({"carrier": ["wind_offshore", "solar_pv"], "cfd_round": ["AR9", "AR8_modelled"],
                          "capex_factor": [0.9, 1.4]})
    merged = pd.DataFrame({"carrier": ["wind_offshore", "solar_pv"], "cfd_round": ["AR7", "future"],
                           "finance_vintage": ["legacy", "new_build"], "capex_power_gbp_per_kw": [2000.0, 500.0],
                           "capex_energy_gbp_per_kwh": [0.0, 0.0]})
    with pytest.raises(ValueError, match="no row of wind_offshore AR9") as raised:
        apply_cost_calibration(merged, table)
    message = str(raised.value)
    assert "solar_pv AR8_modelled (the fleet holds no such round)" in message
    assert "cost_calibration.enabled: false" in message and "derive_cost_calibration.py" in message


def test_the_2025_legislation_bound_takes_the_held_factor_on_its_new_build_rows():
    """6 Nov 2026: legislation in 2025 makes AR7 new build; the table derived at 2028 (legacy) still applies."""
    table = pd.DataFrame({"carrier": ["wind_offshore"], "cfd_round": ["AR7"], "capex_factor": [0.9]})
    for vintage in ("legacy", "new_build"):
        merged = pd.DataFrame({"carrier": ["wind_offshore"], "cfd_round": ["AR7"], "finance_vintage": [vintage],
                               "capex_power_gbp_per_kw": [2000.0], "capex_energy_gbp_per_kwh": [0.0]})
        out = apply_cost_calibration(merged, table)
        assert out["capex_power_gbp_per_kw"].iloc[0] == pytest.approx(1800.0)
        assert out["capex_calibration_factor"].iloc[0] == pytest.approx(0.9)
