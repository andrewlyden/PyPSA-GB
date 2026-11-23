"""Item N2: the ``public_nationalised`` case in the finance layer.

Every GB asset is state-owned; a legacy asset costs what it costs to buy,
annuitised at the public rate over the acquisition term, plus its O&M; new
build is financed at the public new-build rate; foreign plant is not bought.
The three market cases must not move when the fourth is added, and the
downstream bill stages must skip it rather than choke on it.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership.calculate_financing_scenarios import calculate_financing_scenarios
from scripts.public_ownership.cfd_repricing import per_row_savings, repricing_settings
from scripts.public_ownership.finance import (
    CASES, MARKET_CASES, NATIONALISED_CASE, capital_recovery_factor, state_equity_share, wacc_for_case,
)
from scripts.public_ownership.reference_runs import validate_public_ownership_config
from scripts.public_ownership.sensitivity_axes import expand_cases, parse_axes

from tests.unit.test_public_ownership_cfd_repricing import _asset, _costs, _snapshot
from tests.unit.test_public_ownership_legacy_valuation import _stations
from config.config_loader import load_defaults


#: Item OH1: the state's holding-company cost, switched off in fixtures that
#: pin other mechanics.  See tests/unit/test_public_ownership_state_overhead.py.
NO_OVERHEAD = {"enabled": False}

PUBLIC_RATE = 0.02745


def _snapshot_with_public_rate():
    snapshot = _snapshot()
    snapshot["assumptions"]["public_borrowing_real"] = {"value": PUBLIC_RATE}
    return snapshot


def _acquisition(**overrides):
    """The acquisition parameters; the price basis, thermal basis and term are levers (item K1)."""
    block = {
        "include_in_net_bill": False, "purchase_price_gbp": None, "workbook_placeholder_gbp": 100.0e9,
        "thermal_price_gbp": None, "valuation": {"horizon_year": 2070},
    }
    block.update(overrides)
    return block


def _terms(**levers):
    """Acquisition terms as lever settings applied to every case (axis central values)."""
    return {name: {"central": str(value)} for name, value in levers.items()}


def _register_inputs():
    """Legacy ROC offshore (dated), legacy gas, legacy foreign plant, new-build CfD and merchant solar."""
    assets = pd.DataFrame([
        _asset("old_wind", "wind_offshore", "Offshore wind", "legacy", "ROC", "", "not_applicable", np.nan, 600.0, 2.5e6,
               commissioning_year=2014),
        _asset("old_cfd", "wind_offshore", "Offshore wind", "legacy", "CfD", "AR1", "pre_policy", 2015, 400.0, 1.6e6,
               commissioning_year=2018),
        _asset("gas::legacy", "CCGT", "Gas generation - unabated", "legacy", "", "", "not_applicable", np.nan, 800.0, 2.0e6),
        _asset("EU_market_FR", "EU_market", "Missing", "legacy", "", "", "not_applicable", np.nan, 2000.0, 8.0e6,
               evidence_status="unresolved_cost_or_finance_mapping"),
        _asset("ar7", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6),
        _asset("solar", "solar_pv", "Solar PV", "new_build", "merchant", "", "not_applicable", np.nan, 500.0, 5.0e5),
    ])
    for column, values in {
        "market_revenue_gbp": [1.2e8, 8.0e7, 1.6e8, 6.0e8, 2.0e8, 4.0e7],
        "support_revenue_gbp": [1.5e8, 9.0e7, 0.0, 0.0, 1.0e8, 0.0],
        "cfd_support_gbp": [0.0, 9.0e7, 0.0, 0.0, 1.0e8, 0.0],
        "cfd_curtailment_compensation_gbp": [0.0] * 6,
        "roc_support_gbp": [1.5e8, 0.0, 0.0, 0.0, 0.0, 0.0],
        "fuel_carbon_gbp": [0.0, 0.0, 2.0e8, 0.0, 0.0, 0.0],
    }.items():
        assets[column] = values
    assets["cfd_award_year"] = assets["cfd_award_year"].astype(float)
    return assets


def _run(cases=("private", "public_coordination", "public_ownership", NATIONALISED_CASE), acquisition=None, axes=None):
    # Item OH1 off: this fixture pins the acquisition accounting.  The tariff's
    # own overhead line is covered in test_public_ownership_state_overhead.py.
    return calculate_financing_scenarios(
        _register_inputs(), _costs(), _snapshot_with_public_rate(), cases=tuple(cases), axes=axes or {},
        acquisition=acquisition if acquisition is not None else _acquisition(), ro_stations=_stations(),
        state_operating_overhead=NO_OVERHEAD,
    )


class TestFinancePrimitives:
    def test_case_sets(self):
        assert NATIONALISED_CASE in CASES and NATIONALISED_CASE not in MARKET_CASES
        assert {"private", "public_control", "public_ownership"} <= MARKET_CASES

    def test_wacc_and_share(self):
        rates = _snapshot()["technologies"][0]
        assert wacc_for_case(rates, NATIONALISED_CASE, "legacy") == rates["wacc"]["public_ownership_legacy"]
        assert wacc_for_case(rates, NATIONALISED_CASE, "new_build") == rates["wacc"]["public_ownership_new_build"]
        assert state_equity_share(NATIONALISED_CASE, "legacy") == 1.0
        assert state_equity_share(NATIONALISED_CASE, "new_build") == 1.0
        # The market cases are untouched.
        assert wacc_for_case(rates, "public_ownership", "legacy") == rates["wacc"]["private"]
        assert state_equity_share("public_ownership", "legacy") == 0.0

    def test_axes_never_vary_the_nationalised_case(self):
        axes, bounding = parse_axes({"axes": {
            "cfd_strike_basis": {"alternatives": ["at_cost"]},
            "ldes_floor_basis": {"alternatives": ["technology_wacc"]},
        }})
        specs = expand_cases(["private", "public_ownership", NATIONALISED_CASE], axes, bounding_pair=bounding)
        ids = [s.case_id for s in specs]
        assert NATIONALISED_CASE in ids
        assert not any(i.startswith(NATIONALISED_CASE + "@") for i in ids)
        assert "public_ownership@strike_at_cost" in ids and "private@ldes_floor_at_wacc" in ids


class TestRegister:
    @pytest.fixture(scope="class")
    def result(self):
        return _run()

    def test_market_cases_are_unchanged_by_the_fourth_case(self, result):
        register, summary = result
        without, summary_without = _run(cases=("private", "public_coordination", "public_ownership"))
        for case in MARKET_CASES:
            a = register.loc[register["case_id"].eq(case)].set_index("asset_id")
            b = without.loc[without["case_id"].eq(case)].set_index("asset_id")
            for column in ("annual_required_revenue_gbp", "annual_capital_recovery_gbp", "wacc", "ownership_share"):
                pd.testing.assert_series_equal(a[column].astype(float), b[column].astype(float), check_names=False)
        assert set(summary["case_id"]) == set(summary_without["case_id"]) | {NATIONALISED_CASE}

    def test_every_gb_row_is_owned_and_foreign_plant_is_not(self, result):
        register, _ = result
        nat = register.loc[register["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
        assert nat.loc[["old_wind", "old_cfd", "gas::legacy", "ar7", "solar"], "ownership_share"].eq(1.0).all()
        assert nat.loc[["old_wind", "old_cfd", "gas::legacy", "ar7", "solar"], "state_owned"].all()
        assert nat.loc["EU_market_FR", "ownership_share"] == 0.0 and not nat.loc["EU_market_FR", "state_owned"]
        assert nat.loc["old_wind", "legacy_scope_group"] == "valued"
        assert nat.loc["gas::legacy", "legacy_scope_group"] == "thermal"
        assert nat.loc["EU_market_FR", "legacy_scope_group"] == "excluded"

    def test_legacy_capital_is_the_annuitised_purchase_price(self, result):
        register, summary = result
        nat = register.loc[register["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
        legacy = nat.loc[["old_wind", "old_cfd", "gas::legacy"]]
        # F28 (e): each row repaid over its own remaining life.
        crf = legacy["acquisition_term_years_applied"].astype(float).map(lambda n: capital_recovery_factor(PUBLIC_RATE, n))
        # Identity: capital recovery / CRF sums back to the purchase price to the pound.
        implied_price = float((legacy["annual_capital_recovery_gbp"] / crf).sum())
        row = summary.loc[summary["case_id"].eq(NATIONALISED_CASE)].iloc[0]
        assert implied_price == pytest.approx(float(row["acquisition_purchase_price_gbp"]), abs=1.0)
        assert float(row["acquisition_annual_charge_gbp"]) == pytest.approx(float(legacy["annual_capital_recovery_gbp"].sum()), abs=1.0)
        assert row["acquisition_purchase_price_basis"] == "state_present_value"
        # F28 (e): the summary's term is the price-weighted mean of the rows' own terms.
        price, term = legacy["acquisition_price_gbp"].astype(float), legacy["acquisition_term_years_applied"].astype(float)
        assert row["acquisition_term_years"] == pytest.approx(float((price * term).sum() / price.sum()))
        # F19 P3 decision 3 (6 Oct 2026): thermal at its owners' value is the central basis.
        assert legacy["acquisition_basis"].tolist() == ["state_present_value", "state_present_value", "thermal_market_value"]
        assert legacy["wacc"].eq(PUBLIC_RATE).all()
        # O&M is untouched: required revenue = capital + fixed + variable.
        for asset_id, r in legacy.iterrows():
            assert r["annual_required_revenue_gbp"] == pytest.approx(
                r["annual_capital_recovery_gbp"] + r["annual_fixed_om_gbp"] + r["annual_variable_om_gbp"])

    def test_valued_rows_are_priced_at_their_state_present_value(self, result):
        register, _ = result
        nat = register.loc[register["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
        private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
        # old_wind: ROC 1.5e8 running off with the synthetic register, plus net cash
        # (market 1.2e8 - fixed O&M 600 MW x 46 GBP/kW = 2.76e7) to retirement in 2039.
        price = float(nat.loc["old_wind", "acquisition_price_gbp"])
        assert price > 0
        net_cash = 1.2e8 - 600.0 * 1000.0 * 46.0
        years = np.arange(2030, 2039)
        pv_net = float((net_cash * (1 + PUBLIC_RATE) ** -(years - 2030)).sum())
        assert price > pv_net  # ROC income adds to it
        # A row priced at acquisition is cheaper to carry than at as-new private capital.
        assert nat.loc["old_wind", "annual_capital_recovery_gbp"] < private.loc["old_wind", "annual_capital_recovery_gbp"]

    def test_book_value_prices_thermal_by_remaining_life(self):
        # F29 (26 Oct 2026): book value is workbook v2.0's A1 for the whole fleet (purchase_price_basis book_value);
        # a row with no workbook cohort takes its as-new capex written down to its retirement.
        register, _ = _run(axes=_terms(purchase_price_basis="book_value"))
        nat = register.loc[register["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
        # Undated CCGT: 15 declared remaining years of the valuation's 25-year CCGT life (continuity audit,
        # 14 Nov 2026: GHD for DESNZ 2025 and FES 2025), as-new 800 MW x 1020 GBP/kW.
        book = 800.0 * 1000.0 * 1020.0 * 15.0 / 25.0
        assert float(nat.loc["gas::legacy", "acquisition_price_gbp"]) == pytest.approx(book)

    def test_new_build_matches_full_public_ownership(self, result):
        register, _ = result
        nat = register.loc[register["case_id"].eq(NATIONALISED_CASE)].set_index("asset_id")
        own = register.loc[register["case_id"].eq("public_ownership")].set_index("asset_id")
        for asset_id in ("ar7", "solar"):
            assert nat.loc[asset_id, "annual_required_revenue_gbp"] == pytest.approx(own.loc[asset_id, "annual_required_revenue_gbp"])
            assert nat.loc[asset_id, "wacc"] == own.loc[asset_id, "wacc"]


class TestPriceBases:
    def test_the_retired_placeholder_is_refused_with_a_pointer(self):
        # F29 (26 Oct 2026): v11-2's GBP 100 bn placeholder is retired; book value is workbook v2.0's A1.
        with pytest.raises(ValueError, match="book_value"):
            _run(axes=_terms(purchase_price_basis="workbook_placeholder", thermal_basis="zero"))

    def test_market_value_is_below_state_value(self):
        _, state = _run()
        _, market = _run(axes=_terms(purchase_price_basis="market_value"))
        s = float(state.loc[state["case_id"].eq(NATIONALISED_CASE), "acquisition_purchase_price_gbp"].iloc[0])
        m = float(market.loc[market["case_id"].eq(NATIONALISED_CASE), "acquisition_purchase_price_gbp"].iloc[0])
        assert m < s

    def test_missing_inputs_fail_loudly(self):
        with pytest.raises(ValueError, match="acquisition"):
            calculate_financing_scenarios(_register_inputs(), _costs(), _snapshot_with_public_rate(),
                                          cases=("private", NATIONALISED_CASE))
        with pytest.raises(ValueError, match="acquisition"):
            _run(acquisition={})
        assets = _register_inputs().drop(columns=["market_revenue_gbp", "support_revenue_gbp", "fuel_carbon_gbp",
                                                  "cfd_support_gbp", "roc_support_gbp", "cfd_curtailment_compensation_gbp"])
        with pytest.raises(ValueError, match="settlement"):
            calculate_financing_scenarios(assets, _costs(), _snapshot_with_public_rate(),
                                          cases=("private", NATIONALISED_CASE), acquisition=_acquisition(), ro_stations=_stations())


class TestDownstreamSkips:
    def test_repricing_ignores_the_nationalised_case(self):
        register, _ = _run()
        register["network_support_type"] = register["support_type"]
        settings = repricing_settings(load_defaults()["public_ownership"]["consumer_bill"])
        savings = per_row_savings(register, settings)
        assert NATIONALISED_CASE not in set(savings["case_id"])
        assert {"public_coordination", "public_ownership"} <= set(savings["case_id"])


def test_repository_defaults_declare_full_nationalisation_as_a_sensitivity():
    """Item K1: the workbook's Scenario 3 is public_ownership@fully_nationalised, a named sensitivity."""
    from scripts.public_ownership.reference_runs import expanded_cases, tariff_cases

    config = load_defaults()["public_ownership"]
    named = config["sensitivities"]["named"]["fully_nationalised"]
    assert named["scenario"] == "public_ownership" and named["legacy_fleet"] == "acquire_all"
    # User decision 10: HPC is bought as an operating plant at its contract value in the
    # central; completing it at the public rate is its own case.  30 Sep 2026 (user
    # decision 2): the wires are out of every base, P3's included, so the bundle no
    # longer names them (they sit at the private baseline); buying them is P3's grid.
    assert "networks" not in named and "in_construction" not in named
    assert config["sensitivities"]["named"]["fully_nationalised_market_value_hpc_completion"][
        "in_construction"] == "public_completion"
    # include_in_net_bill is retired: lever acquisition_financing replaces it.
    assert "include_in_net_bill" not in config["acquisition"]
    assert "enabled" not in config["acquisition"] and "purchase_price_basis" not in config["acquisition"]
    assert validate_public_ownership_config(config) is config
    assert "public_ownership@fully_nationalised" in tariff_cases(config)
    # Switching acquisition off is a consistent configuration too: drop every named position that
    # acquires, and everything that extends one.  Matched on what a position does, not on its name:
    # the policy propositions of 22 September acquire without carrying the fully_nationalised prefix.
    off = deepcopy(config)
    acquiring, kept = set(), {}
    for name, spec in off["sensitivities"]["named"].items():
        parent = spec.get("extends")
        if parent in acquiring or spec.get("legacy_fleet", "private") != "private":
            acquiring.add(name)
            continue
        kept[name] = spec
    off["sensitivities"]["named"] = kept
    # Presets, bounds and toggle grids on an acquiring parent go with it.
    sens = off["sensitivities"]
    for preset in sens["presets"].values():
        preset["applies_to"] = [t for t in preset["applies_to"] if "nationalised" not in t]
        # 30 Sep 2026: a per-parent value for a dropped parent goes with it (a stale key is refused).
        preset["levers_by_case"] = {k: v for k, v in (preset.get("levers_by_case") or {}).items()
                                    if "nationalised" not in k}
    sens["bounds"]["applies_to"] = [t for t in sens["bounds"]["applies_to"] if "nationalised" not in t]
    sens["bounds"]["exclude"] = {k: v for k, v in sens["bounds"]["exclude"].items() if "nationalised" not in k}
    sens["toggle_grids"]["grids"] = {k: v for k, v in sens["toggle_grids"]["grids"].items() if "nationalised" not in k}
    sens["toggle_grids"]["bound_grids"] = {k: v for k, v in (sens["toggle_grids"].get("bound_grids") or {}).items()
                                           if "nationalised" not in k}
    off["sensitivities"]["one_at_a_time"] = [x for x in off["sensitivities"]["one_at_a_time"] if x != "legacy_fleet"]
    # Item BM1, P3: a group whose months bid on a tariff case goes with the case.
    off["reference_run_groups"] = {name: group for name, group in off["reference_run_groups"].items()
                                   if "bid_case" not in group}
    assert validate_public_ownership_config(off) is off
    assert tariff_cases(off) == [] and "public_ownership@fully_nationalised" not in expanded_cases(off)
    # The retired keys fail the parse with a pointer at the levers.
    for key, value in (("cases", ["private"]), ("existing_assets", "acquired"), ("network_ownership", "acquired")):
        broken = deepcopy(config)
        broken[key] = value
        with pytest.raises(ValueError, match="retired"):
            validate_public_ownership_config(broken)
