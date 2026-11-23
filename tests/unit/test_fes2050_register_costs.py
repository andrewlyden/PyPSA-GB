"""FES 2050 plan, task Y10: the vintaged register's costs, the insurance rule and the private rate path."""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership.calculate_financing_scenarios import (
    LEVER_COLUMNS, VINTAGE_INSURANCE_RULE, calculate_financing_scenarios, vintage_cost_registry, vintage_options_for,
)
from scripts.public_ownership.finance import capital_recovery_factor, neso_wind_ramp_deltas, private_rate_path_delta
from scripts.public_ownership.vintage_costs import costs_for

PRIVATE = {"Offshore wind": 0.0827, "Solar PV": 0.066645, "Onshore wind": 0.0747}
SCENARIOS = {
    "private": {},
    "public_ownership": {"new_build_generation": "public", "public_return": "public_rate",
                         "public_rate_basis": "real_cpi"},
}
INSURANCE = {"insurance": {"enabled": True, "share_of_capex_per_year": 0.004,
                           "gbp_per_kw_year_by_technology": {"Offshore wind": 11.22},
                           "excluded_technologies": ["Large-scale nuclear", "Lithium batteries"]}}
NO_OVERHEAD = {"enabled": False}


def _snapshot():
    return {"rate_basis": "real_CPI",
            "assumptions": {"public_coordination_state_equity_share": {"value": 0.30},
                            "public_borrowing_real": {"value": 0.02745}},
            "technologies": [{"technology": t, "risk_rating": "Medium",
                              "wacc": {"private": r, "public_coordination": r - 0.008,
                                       "public_ownership_new_build": 0.0325, "public_ownership_legacy": 0.02745}}
                             for t, r in PRIVATE.items()]}


def _assets(model_year=2050, years=(2020, 2030, 2033, 2045), with_solar=True):
    rows = []
    for year in years:
        rows.append({"asset_id": f"OFF_{year}", "carrier": "wind_offshore", "finance_technology": "Offshore wind",
                     "build_year": year, "capacity_mw": 100.0})
    if with_solar:
        rows.append({"asset_id": "SOL_2040", "carrier": "solar_pv", "finance_technology": "Solar PV",
                     "build_year": 2040, "capacity_mw": 50.0})
    frame = pd.DataFrame(rows)
    frame["model_year"] = model_year
    frame["component_type"] = "Generator"
    frame["finance_vintage"] = np.where(frame["build_year"] < 2025, "legacy", "new_build")
    frame["commissioning_year"] = frame["build_year"].astype(float)
    frame["dispatch_mwh"] = frame["capacity_mw"] * 3000.0
    frame["evidence_status"] = "resolved"
    return frame


def _old_costs():
    from pathlib import Path
    return pd.read_csv(Path("data/public_ownership/financing/technology_costs.csv"))


def _specs(sensitivities=None):
    from scripts.public_ownership.levers import expand_cases, parse_scenarios, parse_sensitivities

    bundles = parse_scenarios(SCENARIOS)
    return expand_cases(bundles, parse_sensitivities(dict({"bounding_pair": False}, **(sensitivities or {})), bundles))


def _run(assets, sensitivities=None, **kwargs):
    kwargs.setdefault("case_specs", _specs(sensitivities))
    return calculate_financing_scenarios(assets, _old_costs(), _snapshot(), price_year=2024,
                                         state_operating_overhead=NO_OVERHEAD, generator_costs=INSURANCE, **kwargs)


def test_the_registry_takes_desnz_insurance_out_of_fixed_om():
    registry = vintage_cost_registry()
    assert VINTAGE_INSURANCE_RULE == "exclude"
    offshore = registry.loc[registry["technology"].eq("Offshore wind") & registry["cost_build_year"].eq(2040)].iloc[0]
    row = costs_for("Offshore wind", 2040)
    assert offshore["fixed_om_gbp_per_kw_year"] == pytest.approx(row["fixed_om_gbp_per_kw_year"]
                                                                 - row["insurance_gbp_per_kw_year"])
    assert offshore["vintage_insurance_excluded_gbp_per_kw_year"] == pytest.approx(row["insurance_gbp_per_kw_year"])
    kept = vintage_cost_registry(insurance="include")
    assert kept.loc[kept["technology"].eq("Offshore wind") & kept["cost_build_year"].eq(2040),
                    "fixed_om_gbp_per_kw_year"].iloc[0] == pytest.approx(row["fixed_om_gbp_per_kw_year"])


def test_a_vintaged_register_is_costed_by_build_year_with_one_insurance_line():
    register, _ = _run(_assets(), vintage_costs=vintage_cost_registry())
    private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
    for asset_id, technology, year in (("OFF_2045", "Offshore wind", 2045), ("OFF_2030", "Offshore wind", 2030),
                                       ("OFF_2020", "Offshore wind", 2025), ("SOL_2040", "Solar PV", 2040)):
        row = costs_for(technology, year)
        got = private.loc[asset_id]
        assert got["cost_build_year"] == year
        # Year-generic stage 5b: the strike path's capex factor (AR7's offshore k; 1 elsewhere) on the vintage row.
        assert got["effective_capex_power_gbp_per_kw"] == pytest.approx(
            row["capex_power_gbp_per_kw"] * float(got["capex_calibration_factor"]))
        site = got["capacity_mw"] * 1000.0 * (row["fixed_om_gbp_per_kw_year"] - row["insurance_gbp_per_kw_year"])
        assert got["annual_site_fixed_om_gbp"] == pytest.approx(site), asset_id
        # the D4(a) line is the only insurance: 11.22 GBP/kW offshore, 0.4% of capex otherwise
        expected = (got["capacity_mw"] * 1000.0 * 11.22 if technology == "Offshore wind"
                    else 0.004 * got["capacity_mw"] * 1000.0 * row["capex_power_gbp_per_kw"])
        assert got["annual_insurance_gbp"] == pytest.approx(expected), asset_id
        assert got["annual_fixed_om_gbp"] == pytest.approx(site + expected), asset_id
    # remaining life follows the build year: a 2045 offshore farm has 25 of its 30 years left in 2050
    assert private.loc["OFF_2045", "remaining_life_years"] == pytest.approx(25.0)
    crf = capital_recovery_factor(PRIVATE["Offshore wind"], 25.0)
    assert private.loc["OFF_2045", "annual_capital_recovery_gbp"] == pytest.approx(
        100.0 * 1000.0 * costs_for("Offshore wind", 2045)["capex_power_gbp_per_kw"] * crf
        * float(private.loc["OFF_2045", "capex_calibration_factor"]))


def test_vintage_costs_are_refused_for_a_2030_register_and_for_undated_rows():
    with pytest.raises(ValueError, match="vintaged model years only"):
        _run(_assets(model_year=2030, years=(2020, 2028)), vintage_costs=vintage_cost_registry())
    undated = _assets()
    undated.loc[0, ["build_year", "commissioning_year"]] = np.nan
    with pytest.raises(ValueError, match="build year on every costed row"):
        _run(undated, vintage_costs=vintage_cost_registry())


def test_without_vintage_costs_a_register_keeps_technology_costs_csv():
    register, _ = _run(_assets())
    offshore = register.loc[register["case_id"].eq("private") & register["asset_id"].eq("OFF_2045")].iloc[0]
    assert offshore["effective_capex_power_gbp_per_kw"] == pytest.approx(2670.0)
    assert "cost_build_year" not in register.columns


def test_the_group_options_gate_vintage_costs_on_the_model_year():
    block = {"reference_run_groups": {"cp30_2030_cp30_network": {"model_year": 2030},
                                      "cp30_bad": {"model_year": 2030, "vintage_costs": {"table": "x.csv"}},
                                      "fes_ht_2050": {"model_year": 2050},
                                      "fes_off": {"model_year": 2050, "vintage_costs": {"enabled": False}}}}
    assert vintage_options_for(block, "cp30_2030_cp30_network", 2030) is None
    with pytest.raises(ValueError, match="vintage_costs is set"):
        vintage_options_for(block, "cp30_bad", 2030)
    registry = vintage_options_for(block, "fes_ht_2050", 2050)
    assert set(registry["cost_basis"]) == {"vintage_central_insurance_exclude"}
    assert vintage_options_for(block, "fes_off", 2050) is None


def test_the_reference_run_validation_refuses_vintage_blocks_before_2040():
    from scripts.public_ownership.reference_runs import _validate_vintaged_group

    with pytest.raises(ValueError, match="applies from model year 2040"):
        _validate_vintaged_group("cp30_2030_cp30_network", {"fleet_vintages": {}}, 2030)
    with pytest.raises(ValueError, match="twice"):
        _validate_vintaged_group("fes", {"vintage_costs": {"insurance": "include"}}, 2050)
    with pytest.raises(ValueError, match="unknown keys"):
        _validate_vintaged_group("fes", {"fleet_vintages": {"lives": 1}}, 2050)
    _validate_vintaged_group("fes", {"fleet_vintages": {"retirement_rule": "strict_economic_life"},
                                     "vintage_costs": {"learning": True}}, 2050)


# ── decision 19: the private rate path ───────────────────────────────────────


def test_the_neso_ramp_is_a_linear_step_down_over_2031_to_2035():
    deltas = neso_wind_ramp_deltas()
    offshore = deltas["Offshore wind"]
    assert offshore[2030] == 0.0 and offshore[2031] == pytest.approx(-0.004)
    assert offshore[2034] == pytest.approx(-0.016) and offshore[2035] == pytest.approx(-0.020)
    assert offshore[2050] == pytest.approx(-0.020)
    assert deltas["Onshore wind"][2035] == pytest.approx(-0.010)
    assert deltas["Floating offshore wind"][2035] == pytest.approx(-0.020)
    ramp = {"private_rate_path": "neso_wind_ramp"}
    assert private_rate_path_delta(ramp, "Offshore wind", 2045, deltas) == pytest.approx(-0.020)
    assert private_rate_path_delta(ramp, "Offshore wind", 2030, deltas) == 0.0
    assert private_rate_path_delta(ramp, "Solar PV", 2045, deltas) == 0.0
    assert private_rate_path_delta({}, "Offshore wind", 2045, deltas) == 0.0
    assert private_rate_path_delta(ramp, "Offshore wind", float("nan"), deltas) == 0.0


def test_the_ramp_moves_private_wind_vintages_after_2030_only_and_carries_its_comparator():
    sens = {"one_at_a_time": ["private_rate_path"]}
    register, totals = _run(_assets(), vintage_costs=vintage_cost_registry(), sensitivities=sens)
    wacc = register.pivot_table(index="asset_id", columns="case_id", values="wacc", aggfunc="first")
    ramp = "private@private_rate_neso_wind_ramp"
    assert wacc.loc["OFF_2045", ramp] == pytest.approx(PRIVATE["Offshore wind"] - 0.020)
    assert wacc.loc["OFF_2033", ramp] == pytest.approx(PRIVATE["Offshore wind"] - 0.012)
    assert wacc.loc["OFF_2030", ramp] == pytest.approx(PRIVATE["Offshore wind"])
    assert wacc.loc["SOL_2040", ramp] == pytest.approx(PRIVATE["Solar PV"])
    # the flat central is the snapshot rate, untouched
    assert wacc.loc["OFF_2045", "private"] == pytest.approx(PRIVATE["Offshore wind"])
    # state-owned rows keep the public rate; the private legacy row is pre-2031, unmoved
    public_ramp = "public_ownership@private_rate_neso_wind_ramp"
    assert wacc.loc["OFF_2045", public_ramp] == pytest.approx(wacc.loc["OFF_2045", "public_ownership"])
    by_case = totals.set_index("case_id")
    assert by_case.loc[public_ramp, "comparator_case_id"] == ramp
    assert by_case.loc[ramp, "annual_required_revenue_gbp"] < by_case.loc["private", "annual_required_revenue_gbp"]
    assert "private_rate_path" in register.columns and "private_rate_path" in totals.columns


def test_gated_levers_add_no_column_to_a_register_that_does_not_use_them():
    register, totals = _run(_assets())
    for name in ("state_deployment_share", "private_rate_path"):
        assert name not in register.columns and name not in totals.columns
    assert [c for c in register.columns if c in LEVER_COLUMNS] == list(LEVER_COLUMNS)


def test_a_vintaged_register_carries_the_2030_purchase_and_never_re_prices_the_survivors():
    """Tasks Y10/Y12: acquired survivors cost their operation only; the annuity is Y12's carried line."""
    from scripts.public_ownership.acquisition_carry_forward import check_register_survivors
    from scripts.public_ownership.calculate_financing_scenarios import CARRIED_ACQUISITION_BASIS

    sens = {"named": {"fully_nationalised": {"scenario": "public_ownership", "legacy_fleet": "acquire_all",
                                             "pre_2025_cfd_capacity": "public_repriced"}}}
    register, totals = _run(_assets(), sensitivities=sens, vintage_costs=vintage_cost_registry(),
                            acquisition={"valuation": {}})
    case = register.loc[register["case_id"].eq("public_ownership@fully_nationalised")].set_index("asset_id")
    legacy = case.loc["OFF_2020"]
    assert bool(legacy["state_owned"]) and legacy["acquisition_basis"] == CARRIED_ACQUISITION_BASIS
    assert legacy["acquisition_price_gbp"] == 0.0 and legacy["annual_capital_recovery_gbp"] == 0.0
    assert legacy["annual_required_revenue_gbp"] == pytest.approx(
        legacy["annual_fixed_om_gbp"] + legacy["annual_variable_om_gbp"])
    assert totals.set_index("case_id").loc["public_ownership@fully_nationalised",
                                           "acquisition_annual_charge_gbp"] == 0.0
    # new build is still financed at the public rate on its vintage capex
    assert case.loc["OFF_2045", "annual_capital_recovery_gbp"] > 0.0
    survivors = case.loc[case["finance_vintage"].eq("legacy"), ["carrier", "capacity_mw"]]
    schedule = pd.DataFrame({"year": [2050], "carrier": ["wind_offshore"], "operating_mw": [100.0]})
    assert check_register_survivors(survivors, schedule, 2050)["wind_offshore"]["matches"]
