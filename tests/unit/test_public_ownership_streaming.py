"""The heavy public-ownership stages a batch at a time (30 Sep 2026): the same numbers as the whole register.

The financing register is written one register key at a time (``register_sink``,
``RegisterWriter``), and the CfD re-pricing and the public account read it back
a few cases at a time (``SavingsStream``, ``StrikeAdjustments``,
``PublicAccount``).  Each test builds a register with several cases sharing
register keys and checks the streamed result against the whole-register one,
exactly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.market.cfd_strike_table import load_strike_table
from scripts.public_ownership.calculate_financing_scenarios import (
    _effective_capex_power, _remaining_life, _state_owned, calculate_financing_scenarios,
    run_group_case_specs,
)
from scripts.public_ownership.cfd_repricing import (
    SavingsStream, StrikeAdjustments, build_strike_adjustments, limit_at_cost_to_deliverable, per_row_savings,
    register_settings_by_case, repricing_settings, route_summary, settled_energy_by_key,
)
from scripts.public_ownership.levers import case_settings
from scripts.public_ownership.build_public_account import merge_accounts
from scripts.public_ownership.public_account import account_settings, build_public_account, pay_as_bid_ceiling_gbp
from scripts.public_ownership.register_io import (
    RegisterWriter, _normalised_column, _normalised_column_by_value, iter_register_cases, read_register,
    read_settings, write_register,
)
from scripts.public_ownership.settlement import attach_settlement

from tests.unit.test_public_ownership_cfd_repricing import ASSUMPTIONS, STRIKE_TABLE, _asset, _costs, _snapshot
from config.config_loader import load_defaults

NO_OVERHEAD = {"enabled": False}


@pytest.fixture(scope="module")
def block() -> dict:
    return load_defaults()["public_ownership"]


def _inputs():
    """Assets with a settlement: re-priced and pre-policy CfDs, merchant solar, gas, BECCS, LDES and a battery."""
    assets = pd.DataFrame([
        _asset("ar7", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR7", "post_policy", 2026, 1000.0, 4.0e6),
        _asset("ar6", "wind_offshore", "Offshore wind", "new_build", "CfD", "AR6", "pre_policy", 2024, 800.0, 3.0e6),
        _asset("solar", "solar_pv", "Solar PV", "new_build", "merchant", "", "not_applicable", np.nan, 500.0, 5.0e5),
        _asset("gas", "CCGT", "Gas generation - unabated", "new_build", "", "", "not_applicable", np.nan, 400.0, 1.0e6),
        _asset("beccs", "CCS Biomass", "Biomass - with CCUS - mature", "new_build", "", "", "not_applicable", np.nan,
               100.0, 1.0e5),
        _asset("ldes", "LAES", "Novel long duration energy storage (LDES)", "new_build", "", "", "not_applicable", np.nan,
               100.0, 5.0e4, component_type="StorageUnit", energy_capacity_mwh=400.0),
        _asset("battery", "Battery", "Lithium batteries", "new_build", "", "", "not_applicable", np.nan, 100.0, 5.0e4,
               component_type="StorageUnit", energy_capacity_mwh=200.0),
        _asset("unmapped", "mystery", "Missing", "new_build", "CfD", "AR7", "post_policy", 2026, 10.0, 1.0e3,
               evidence_status="unmapped_carrier"),
    ])
    settlement = pd.DataFrame({
        "component_type": ["Generator"] * 5 + ["StorageUnit"] * 2 + ["Generator"],
        "asset_id": ["ar7", "ar6", "solar", "gas", "beccs", "ldes", "battery", "unmapped"],
        "market_revenue_gbp": [2.0e8, 1.5e8, 4.0e7, 1.2e8, 1.5e7, 5.0e6, 8.0e6, 1.0e5],
        "support_revenue_gbp": [1.5e8, 1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "wholesale_revenue_gbp": [1.9e8, 1.4e8, 4.0e7, 1.2e8, 1.5e7, 5.0e6, 8.0e6, 1.0e5],
        "fuel_carbon_gbp": [0.0, 0.0, 0.0, 1.0e8, 1.0e7, 0.0, 0.0, 0.0],
    })
    costs = pd.concat([_costs(), pd.DataFrame([{
        "technology": "Biomass - with CCUS - mature", "capex_power_gbp_per_kw": 4423.57, "capex_energy_gbp_per_kwh": 0.0,
        "fixed_om_gbp_per_kw_year": 202.73, "variable_om_gbp_per_mwh": 5.06, "economic_life_years": 25.0,
        "price_year": 2024, "real_or_nominal": "real", "status": "reviewed"}, {
        "technology": "Novel long duration energy storage (LDES)", "capex_power_gbp_per_kw": 977.99,
        "capex_energy_gbp_per_kwh": 0.0, "duration_hours": 4, "fixed_om_gbp_per_kw_year": 32.61,
        "variable_om_gbp_per_mwh": 0.0, "economic_life_years": 20.0, "price_year": 2024, "real_or_nominal": "real",
        "status": "provisional"}])], ignore_index=True)
    snapshot = _snapshot()
    for technology, private in (("Biomass - with CCUS - mature", 0.095), ("Novel long duration energy storage (LDES)", 0.108)):
        snapshot["technologies"].append({"technology": technology,
                                         "wacc": {"private": private, "public_coordination": private - 0.005,
                                                  "public_ownership_new_build": 0.0425, "public_ownership_legacy": 0.0275}})
    return attach_settlement(assets, settlement), costs, snapshot


#: Two alternatives on a lever that is not a register key (several cases per key)
#: and one that is (several keys).
AXES = {"cfd_strike_basis": {"alternatives": ["at_cost"]},
        "ownership_boundary": {"alternatives": ["ownership_all_cfd_repriced"]}}


def _register(block, sink=None, **extra):
    assets, costs, snapshot = _inputs()
    return calculate_financing_scenarios(assets, costs, snapshot, price_year=2024, axes=AXES,
                                         state_operating_overhead=NO_OVERHEAD,
                                         ldes_account_settings=account_settings(block),
                                         electricity_leg_carriers=["CCS Biomass"], register_sink=sink, **extra)


def _streamed(block, path):
    with RegisterWriter(path) as writer:
        _, totals = _register(block, sink=writer.write)
    return totals


def test_streamed_register_reads_back_as_the_whole_register(tmp_path, block):
    register, totals = _register(block)
    assert register["case_id"].nunique() >= 6          # several cases per register key, and several keys
    whole = tmp_path / "w_asset_finance_register.parquet"
    write_register(register, whole)
    streamed = tmp_path / "s_asset_finance_register.parquet"
    streamed_totals = _streamed(block, streamed)
    pd.testing.assert_frame_equal(totals, streamed_totals, check_exact=True)
    pd.testing.assert_frame_equal(read_register(whole), read_register(streamed), check_exact=True)
    pd.testing.assert_frame_equal(read_settings(whole), read_settings(streamed))
    import pyarrow.parquet as pq

    a, b = pq.read_schema(whole), pq.read_schema(streamed)
    assert [(f.name, str(f.type)) for f in a] == [(f.name, str(f.type)) for f in b]
    assert json.loads(a.metadata[b"pandas"])["columns"] == json.loads(b.metadata[b"pandas"])["columns"]
    meta_a, meta_b = (json.loads(s.metadata[b"pypsa_gb_register"]) for s in (a, b))
    assert {k: meta_a[k] for k in ("column_order", "object_kinds")} == {k: meta_b[k] for k in ("column_order", "object_kinds")}
    # The case index reads each case alone, exactly as a filter over the file does.
    assert set(meta_b["case_row_groups"]) == set(register["case_id"])
    for case_id in register["case_id"].unique():
        pd.testing.assert_frame_equal(read_register(whole, cases=[case_id]), read_register(streamed, cases=[case_id]),
                                      check_exact=True)


def test_small_row_groups_keep_every_case_readable(tmp_path, block):
    register, _ = _register(block)
    path = tmp_path / "g_asset_finance_register.parquet"
    with RegisterWriter(path, row_group_rows=5) as writer:
        for _, rows in register.groupby("case_id", sort=False):
            writer.write(rows)
    cases = list(register["case_id"].unique())
    got = pd.concat(list(iter_register_cases(path, batch_cases=2)), ignore_index=True)
    expected = pd.concat([read_register(path, cases=[c]) for c in cases], ignore_index=True)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    runs = list(iter_register_cases(path, batch_cases=None))
    assert sum(len(frame) for frame in runs) == len(register)


def test_vectorised_rows_match_the_scalar_rules(block):
    """The per-row helpers the account was written with, against the register's arrays."""
    register, _ = _register(block)
    assets, costs, snapshot = _inputs()
    rows = register.loc[register["case_id"].eq("public_ownership")].set_index("asset_id")
    merged = assets.merge(costs.assign(_k=costs["technology"]), how="left", left_on="finance_technology", right_on="_k",
                          suffixes=("", "_cost"))
    for _, asset in merged.iterrows():
        row = rows.loc[asset["asset_id"]]
        assert bool(row["state_owned"]) == _state_owned(asset, case_settings(rows.reset_index()), None, None)
        if row["evidence_status"] == "resolved" and pd.notna(asset.get("technology")):
            assert row["effective_capex_power_gbp_per_kw"] == _effective_capex_power(asset)[0]
            assert row["remaining_life_years"] == _remaining_life(asset)


def test_savings_stream_and_strike_adjustments_match_the_whole_register(tmp_path, block):
    register, summary = _register(block)
    path = tmp_path / "g_asset_finance_register.parquet"
    _streamed(block, path)
    settings = repricing_settings(block["consumer_bill"])
    whole = per_row_savings(read_register(path), settings)
    stream = SavingsStream(path, settings, batch_cases=1)
    batches = [savings for _, savings in stream]
    assert len(batches) == whole["case_id"].nunique()
    pd.testing.assert_frame_equal(pd.concat(batches, ignore_index=True), whole, check_exact=True)
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    table = load_strike_table(STRIKE_TABLE, assumptions["cfd_strike_indexation"], 2024)
    paid = {f"{r}_{t}": 1.0e6 for r, t in table.strikes}
    limited_whole, _ = limit_at_cost_to_deliverable(whole, table, None, paid)
    energy = settled_energy_by_key(stream.private_savings, table, None, paid)
    builder = StrikeAdjustments(stream.private_savings, table, settings, None, paid)
    parts = []
    for savings in batches:
        savings, _ = limit_at_cost_to_deliverable(savings, table, None, paid, settled_energy=energy)
        builder.add(savings)
        parts.append(savings)
    pd.testing.assert_frame_equal(pd.concat(parts, ignore_index=True), limited_whole, check_exact=True)
    pd.testing.assert_frame_equal(builder.frame(), build_strike_adjustments(limited_whole, table, settings, None, paid),
                                  check_exact=True)
    pd.testing.assert_frame_equal(pd.concat([route_summary(p, summary) for p in parts], ignore_index=True),
                                  route_summary(limited_whole, summary), check_exact=True)


def test_public_account_in_batches_matches_the_whole_register(tmp_path, block):
    path = tmp_path / "g_asset_finance_register.parquet"
    _streamed(block, path)
    register = read_register(path)
    settings = repricing_settings(block["consumer_bill"])
    savings = per_row_savings(register, settings)
    account, summary = build_public_account(register, savings, account_settings(block))
    stream = SavingsStream(path, settings, batch_cases=2)
    accounts, summaries = [], []
    for rows, batch in stream:
        part, part_summary = build_public_account(
            rows, batch, account_settings(block), settings_by_case=stream.context.settings_by_case,
            rows_for=stream.context.rows_for, pay_as_bid_ceiling=pay_as_bid_ceiling_gbp(stream.private_rows))
        accounts.append(part)
        summaries.append(part_summary)
    assert len(accounts) > 1
    streamed_account, streamed_summary = merge_accounts(accounts, summaries)
    pd.testing.assert_frame_equal(streamed_account, account, check_exact=True)
    assert json.dumps(streamed_summary) == json.dumps(summary)


def test_case_settings_come_from_the_settings_table_as_from_the_rows(tmp_path, block):
    path = tmp_path / "g_asset_finance_register.parquet"
    _streamed(block, path)
    register = read_register(path)
    assert register_settings_by_case(path) == {c: case_settings(r) for c, r in register.groupby("case_id", sort=False)}


def test_a_register_restricted_to_some_cases_keeps_their_rows_exactly(block):
    register, totals = _register(block)
    kept = ["private", "public_ownership@strike_at_cost", "public_coordination"]
    kept = [c for c in kept if c in set(register["case_id"])]
    subset, subset_totals = _register(block, register_cases=kept)
    expected = register.loc[register["case_id"].isin(kept)].reset_index(drop=True)
    for lever in [c for c in expected.columns if isinstance(expected[c].dtype, pd.CategoricalDtype)]:
        expected[lever], subset[lever] = expected[lever].astype(object), subset[lever].astype(object)
    pd.testing.assert_frame_equal(subset, expected, check_exact=True)
    pd.testing.assert_frame_equal(
        subset_totals, totals.loc[totals["case_id"].isin(kept)].reset_index(drop=True), check_exact=True)


def test_a_balancing_group_registers_its_own_cases(block):
    from scripts.public_ownership.reference_runs import configured_case_specs, group_market_cases

    specs = configured_case_specs(block)
    for group, spec in (block.get("reference_run_groups") or {}).items():
        kept = run_group_case_specs(block, group, specs)
        if str(spec.get("balancing_bids", "contract_value")) == "contract_value":
            assert len(kept) == len(specs)
        else:
            ids = [s.case_id for s in kept]
            assert "private" in ids and set(group_market_cases(block, group)) <= set(ids)
            assert ids == [s.case_id for s in specs if s.case_id in set(ids)]


@pytest.mark.parametrize("values", [
    ["a", None, np.nan, ""], [True, None, False], [np.bool_(True), np.nan], [None, None], ["", np.nan],
    [1, "x", None], [True, "x"], ["2025", None], [pd.NA, "x"], [pd.NaT, "x"], [1.5, None], [np.float64("nan"), "q"],
])
def test_normalised_columns_follow_the_per_value_rule(values):
    series = pd.Series(values, dtype=object)
    fast, fast_kind = _normalised_column(series)
    slow, slow_kind = _normalised_column_by_value(series)
    assert fast_kind == slow_kind and fast.dtype == slow.dtype
    assert [repr(v) for v in fast] == [repr(v) for v in slow]
