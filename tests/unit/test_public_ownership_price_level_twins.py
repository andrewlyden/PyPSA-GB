"""Item F12 at the ownership layer (12 Oct 2026): price-level twin run groups.

A twin (``reference_run_groups.<twin>.price_level_of: <group>``) re-accounts its
parent's solved months at its forecast case's price level (CP.32), so private, P1, P2
and P3 exist at both levels under distinct names without a re-solve. These tests hold:

* the config checks (one twin per parent and level, same months, a cp32 case naming the
  parent's case, a balancing twin measured against the market group's twin) and that
  the one-group-per-role counts are unchanged by twins;
* the twin's settlement re-prices the Stage 1 positions and CfD top-ups on the bill's own
  mapped hours, and holds the balancing settlement, fuel, ROCs, the DPA and volumes;
* the strike cut's paid energy, the public account and the bills settle at the twin's
  level, while every model-price call is exactly what it was;
* the two approved code fixes: the LDES lookup through ``price_level_parent`` and the
  ``churn_sink`` exclusion from the finance register.
"""

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.consumer_bill.project_future_bill as pfb
from scripts.consumer_bill import price_level_mapping as plm
from scripts.public_ownership import reference_runs as rr
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Network bounds (user decision, 18-19 Oct 2026): the model-price parents with the full case list are the
# CENTRAL network's (no_acceleration); cp30_2030* (the LOWER network) carries the headline cases.
MARKET, P2, P3 = ("cp30_2030_no_acceleration", "cp30_2030_no_acceleration_public_bids_p2",
                  "cp30_2030_no_acceleration_public_bids_p3")
NETWORK = "no_acceleration"
TWINS = {f"{MARKET}_cp32": MARKET, f"{P2}_cp32": P2, f"{P3}_cp32": P3}


# ── config: the twins on the live defaults, built in memory ───────────────────

@pytest.fixture(scope="module")
def defaults():
    return load_defaults()


#: The cp32 twins' reduced case list (user decision, 12 Oct 2026): the headline cases, no grids.
HEADLINE_SELECTION = {"kinds": ["scenario", "named", "preset", "bound"],
                      "cases": ["public_ownership@state_bids_at_cost"]}


def _with_twins(defaults, selection=None):
    """The live public_ownership block and forecast cases with a cp32 twin of the three 2030 bill groups."""
    po = copy.deepcopy(defaults["public_ownership"])
    cases = copy.deepcopy(defaults["consumer_bill"]["forecasts"]["cases"])
    groups = po["reference_run_groups"]
    for twin, parent in TWINS.items():
        parent_case = rr.group_forecast_case_id(po, parent)
        twin_case = f"{parent_case}_cp32"
        cases[twin_case] = {**copy.deepcopy(cases[parent_case]), "price_level": "cp32",
                            "price_level_mapping": dict(plm.DEFAULT_SETTINGS), "price_level_parent": parent_case,
                            "recovery_case_id": "CP30_2030_cp30_network"}
        groups[twin] = {**copy.deepcopy(groups[parent]), "price_level_of": parent, "forecast_case": twin_case}
        if selection is not None:
            groups[twin]["case_selection"] = copy.deepcopy(selection)
    for case_id, case in cases.items():
        case["case_id"] = case_id
    return po, cases


@pytest.fixture(scope="module")
def twinned(defaults):
    return _with_twins(defaults)


def test_the_model_price_groups_and_their_roles_are_unchanged_by_twins(defaults):
    po = defaults["public_ownership"]
    cases = copy.deepcopy(defaults["consumer_bill"]["forecasts"]["cases"])
    twins = rr.price_level_twin_groups(po)
    on_network = lambda g: po["reference_run_groups"][g].get("network_case") == NETWORK  # noqa: E731
    assert {t for t in twins if on_network(t)} <= set(TWINS)   # none before the cp32_ownership config patch
    # P1/P2 under central dispatch (20 Oct 2026): the institution twin is a bill group of its own, not a price twin.
    institution = rr.institution_twin_groups(po)
    # F28: the legislation-year bound (a register bound) re-accounts the market group's months; not counted here.
    assert [g for g in rr.consumer_bill_groups(po, cases) if g not in twins and g not in institution
            and on_network(g) and g not in rr.register_bound_groups(po)] == [MARKET, P2, P3]


def test_twins_validate_and_take_their_parents_roles_at_cp32(twinned):
    po, cases = twinned
    rr.validate_public_ownership_config(po, cases)
    on_network = lambda g: po["reference_run_groups"][g].get("network_case") == NETWORK  # noqa: E731
    assert {t: p for t, p in rr.price_level_twin_groups(po).items() if on_network(t)} == TWINS
    assert [g for g in rr.consumer_bill_groups(po, cases) if on_network(g)
            and g not in rr.institution_twin_groups(po) and g not in rr.register_bound_groups(po)] == [MARKET, P2, P3, *TWINS]
    for twin, parent in TWINS.items():
        assert rr.price_level_parent_group(po, twin) == parent
        assert rr.group_price_level(po, twin, cases) == "cp32"
        assert rr.group_price_level(po, parent, cases) == "model"
        assert rr.group_market_cases(po, twin) == rr.group_market_cases(po, parent)
        assert rr.group_tariff_cases(po, twin) == rr.group_tariff_cases(po, parent)
        assert rr.group_bid_case(po, twin) == rr.group_bid_case(po, parent)
        assert rr.is_tariff_balancing_group(po, twin) == rr.is_tariff_balancing_group(po, parent)
    assert rr.price_level_parent_group(po, MARKET) is None


def _twin_check(po, cases):
    rr._validate_price_level_twins(po, cases)


def test_a_twin_must_repeat_its_parents_months(twinned):
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"][f"{P2}_cp32"]["scenario_pattern"] = "Other_{month}2030"
    with pytest.raises(ValueError, match="repeats its parent"):
        _twin_check(po, cases)


def test_a_twins_case_must_be_off_the_model_level_and_name_the_parent_case(twinned):
    po, cases = copy.deepcopy(twinned)
    cases["CP30_2030_no_acceleration_cp32"].pop("price_level")
    with pytest.raises(ValueError, match="model's own price level"):
        _twin_check(po, cases)
    po, cases = copy.deepcopy(twinned)
    cases["CP30_2030_no_acceleration_cp32"]["price_level_parent"] = "CP30_2030_no_acceleration_public_bids_p2"
    with pytest.raises(ValueError, match="price_level_parent"):
        _twin_check(po, cases)
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"][f"{MARKET}_cp32"]["forecast_case"] = "CP30_2030_no_acceleration"
    with pytest.raises(ValueError, match="names its own forecast_case"):
        _twin_check(po, cases)


def test_twin_of_a_twin_and_two_twins_at_one_level_are_refused(twinned):
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"]["twin_of_twin"] = {**po["reference_run_groups"][f"{MARKET}_cp32"],
                                                  "price_level_of": f"{MARKET}_cp32"}
    with pytest.raises(ValueError, match="itself a price-level twin"):
        _twin_check(po, cases)
    po, cases = copy.deepcopy(twinned)
    po["reference_run_groups"]["second_cp32"] = copy.deepcopy(po["reference_run_groups"][f"{MARKET}_cp32"])
    with pytest.raises(ValueError, match="two twins at price level"):
        _twin_check(po, cases)


def test_a_balancing_twin_needs_the_market_twin_at_its_level(twinned):
    po, cases = copy.deepcopy(twinned)
    del po["reference_run_groups"][f"{MARKET}_cp32"]
    with pytest.raises(ValueError, match="contract-value twin"):
        _twin_check(po, cases)


# ── the rules' own helpers, executed from the rule files ──────────────────────

def _smk_function(path, name, namespace):
    """Exec one top-level ``def`` of a rule file in ``namespace`` (the rules are not importable)."""
    text = (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.search(rf"^def {name}\(.*?(?=^\S)", text, flags=re.S | re.M)
    assert match, name
    exec(match.group(0), namespace)
    return namespace[name]


def _bill_rule_namespace(po, cases, assumptions):
    return {"re": re, "resources_path": "resources", "_consumer_defaults": {"public_ownership": po},
            "_consumer_future_cases": cases, "_consumer_all_future_cases": cases,
            "_CONSUMER_STUDIES": SimpleNamespace(case_study={}),
            "_consumer_future_assumptions": lambda case_id: assumptions}


def test_the_ldes_line_reads_the_twin_account_else_the_parents(twinned):
    """Approved fix (12 Oct 2026): a cp32 case with no group of its own reads its price_level_parent's account."""
    po, cases = twinned
    assumptions = {"model_linked_rules": {"ldes_cap_and_floor": {"enabled": True}}}
    lookup = _smk_function("rules/consumer_bill.smk", "_consumer_ldes_public_account",
                           _bill_rule_namespace(po, cases, assumptions))
    account = lambda g: [f"resources/public_ownership/{g}_public_account_summary.json"]
    assert lookup(SimpleNamespace(bill_case="CP30_2030_no_acceleration")) == account(MARKET)
    assert lookup(SimpleNamespace(bill_case="CP30_2030_no_acceleration_cp32")) == account(f"{MARKET}_cp32")
    assert lookup(SimpleNamespace(bill_case="CP30_2030_no_acceleration_public_bids_p3_cp32")) == account(f"{P3}_cp32")
    # The bill-wiring config alone (cp32 cases, no twin groups): the parent's account.
    bare = copy.deepcopy(po)
    for twin in TWINS:
        del bare["reference_run_groups"][twin]
    lookup = _smk_function("rules/consumer_bill.smk", "_consumer_ldes_public_account",
                           _bill_rule_namespace(bare, cases, assumptions))
    assert lookup(SimpleNamespace(bill_case="CP30_2030_no_acceleration_cp32")) == account(MARKET)
    assert lookup(SimpleNamespace(bill_case="CP30_2030_no_acceleration_public_bids_p2_cp32")) == account(P2)
    broken = copy.deepcopy(cases)
    broken["CP30_2030_no_acceleration_cp32"]["price_level_parent"] = "nope"
    lookup = _smk_function("rules/consumer_bill.smk", "_consumer_ldes_public_account",
                           _bill_rule_namespace(bare, broken, assumptions))
    with pytest.raises(ValueError, match="price_level_parent"):
        lookup(SimpleNamespace(bill_case="CP30_2030_no_acceleration_cp32"))


def test_register_booked_lines_read_the_twin_of_the_named_account(twinned):
    po, cases = twinned
    assumptions = {"sourced_externals": {"battery_revenue_top_up": {"derivation": {"cost_from": {
        "public_account": "resources/public_ownership/cp30_2030_cp30_network_public_account_summary.json",
        "case": "private", "key": "battery_revenue_top_up_gbp"}}}, "tnuos": {"derivation": {}}}}
    accounts = _smk_function("rules/consumer_bill.smk", "_consumer_register_line_accounts",
                             _bill_rule_namespace(po, cases, assumptions))
    account = lambda g: {"battery_revenue_top_up": f"resources/public_ownership/{g}_public_account_summary.json"}  # noqa: E731
    # The LOWER network's cases read the account the assumptions name, its twin at cp32.
    assert accounts("CP30_2030_cp30_network") == {}
    assert accounts("CP30_2030_cp30_network_public_bids_p2") == {}
    for case in ("CP30_2030_cp30_network_cp32", "CP30_2030_cp30_network_public_bids_p2_cp32", "CP30_2030_cp30_network_public_bids_p3_cp32"):
        assert accounts(case) == account("cp30_2030_cp30_network_cp32")
    # Network bounds (18-19 Oct 2026): another network's cases read their own market group's account
    # (register_account_group), and its twin at cp32.
    for case in ("CP30_2030_no_acceleration", "CP30_2030_no_acceleration_public_bids_p2"):
        assert accounts(case) == account(MARKET)
    for case in ("CP30_2030_no_acceleration_cp32", "CP30_2030_no_acceleration_public_bids_p3_cp32"):
        assert accounts(case) == account(f"{MARKET}_cp32")
    assert accounts("CP30_2030_no_upgrades_cp32") == account("cp30_2030_no_upgrades_cp32")


def test_a_balancing_twin_is_measured_against_the_market_twin(twinned):
    po, cases = twinned
    groups = po["reference_run_groups"]
    bill = rr.consumer_bill_groups(po, cases)
    market = [g for g in bill if rr.group_balancing_bids(groups[g]) == "contract_value"]
    from scripts.public_ownership.model_year import group_model_year

    namespace = {"PUBLIC_OWNERSHIP_GROUPS": groups, "_po_group_model_year": group_model_year,
                 "_PO_MARKET_BID_GROUPS": market, "_PO_REGISTER_BOUND_GROUPS": rr.register_bound_groups(po),
                 "_PO_PRICE_LEVEL_TWINS": rr.price_level_twin_groups(po), "_PO_INSTITUTION_TWINS": rr.institution_twin_groups(po),
                 "_po_parent_forecast_case": lambda g: cases[rr.group_forecast_case_id(po, g)],
                 "_po_group_network_case": rr.group_network_case}
    measured_against = _smk_function("rules/public_ownership.smk", "_po_market_bid_group_for", namespace)
    assert measured_against(P2) == measured_against(P3) == MARKET
    assert measured_against(f"{P2}_cp32") == measured_against(f"{P3}_cp32") == f"{MARKET}_cp32"


def test_the_solve_side_rules_never_match_a_twin():
    """State shares, commissioning years, their check and the decomposition are the parent's (solve quantities)."""
    text = (ROOT / "rules/public_ownership.smk").read_text(encoding="utf-8")

    def block(rule):
        start = text.index(f"rule {rule}:")
        end = text.find("\nrule ", start + 1)
        return text[start:end]

    for rule in ("build_public_ownership_state_shares", "build_public_ownership_commissioning_years"):
        assert "run_group=PUBLIC_OWNERSHIP_SOLVED_GROUP_REGEX" in block(rule), rule
    assert "run_group=PUBLIC_OWNERSHIP_SOLVED_MARKET_BID_GROUP_REGEX" in block("verify_public_ownership_ownership_tables")
    # Network bounds (18-19 Oct 2026): the calibration is checked on the group it is derived from only
    # (cost_calibration.derived_from_group, never a twin), and the merges read the stamp of the tables
    # their bid months bid on (_po_bid_tables_group: cp30_2030_cp30_network's for every network).
    assert "run_group=PUBLIC_OWNERSHIP_CALIBRATION_CHECK_GROUP_REGEX" in block("verify_public_ownership_cost_calibration")
    for rule in ("merge_public_ownership_balancing_groups", "merge_public_ownership_tariff_balancing_groups"):
        assert "_po_bid_tables_group(wc.run_group)}_ownership_tables_check.json" in block(rule), rule
        assert "_po_solved_group(g)}_bm_bid_decomposition.json" in block(rule), rule
    assert "run_group=PUBLIC_OWNERSHIP_PRICE_LEVEL_TWIN_REGEX" in block("build_public_ownership_price_level_twin_settlement")


# ── the twin's settlement ──────────────────────────────────────────────────────

def _hourly_year(tmp_path, months=2):
    """Two 'months' of hourly prices (with a near-zero band) and their price CSVs."""
    index = pd.date_range("2030-01-01", periods=24 * 10 * months, freq="h")
    rng = np.random.default_rng(7)
    price = rng.uniform(5.0, 120.0, len(index))
    price[rng.random(len(index)) < 0.3] = rng.uniform(-0.5, 0.5, 1)[0] * 1e-3
    frames, paths = [], []
    for number, chunk in enumerate(np.array_split(np.arange(len(index)), months)):
        idx = index[chunk]
        path = tmp_path / f"month{number}_wholesale_price.csv"
        pd.DataFrame({"wholesale_price": price[chunk], "spread": 1.0}, index=pd.Index(idx, name="snapshot")).to_csv(path)
        frames.append(pd.DataFrame({"price_gbp_per_mwh": price[chunk], "hours": 1.0, "demand_mw": 100.0,
                                    "price_file": str(path)}, index=idx))
        paths.append(path)
    return pd.concat(frames), paths


SETTLEMENT_ROWS = pd.DataFrame({
    "component_type": ["Generator", "Generator", "StorageUnit"], "asset_id": ["farm", "ccgt", "battery"],
    "carrier": ["wind_offshore", "CCGT", "Battery"], "support_type": ["CfD", "merchant", "merchant"],
    "payee_class": "register_asset", "bus": "B", "bus_lat": 55.0, "bus_lon": -2.0,
    "wholesale_mwh": [1000.0, 500.0, -10.0], "balancing_mwh": [900.0, 520.0, -10.0],
    "wholesale_revenue_gbp": [40000.0, 30000.0, -200.0], "balancing_settlement_gbp": [-3000.0, 1500.0, 50.0],
    "cfd_support_gbp": [50000.0, 0.0, 0.0], "cfd_curtailment_compensation_gbp": [3000.0, 0.0, 0.0],
    "roc_support_gbp": [0.0, 0.0, 0.0], "dpa_support_gbp": [0.0, 0.0, 0.0], "fuel_carbon_gbp": [0.0, 25000.0, 0.0],
    "fuel_carbon_basis": "resource_breakdown", "market_revenue_gbp": [37000.0, 31500.0, -150.0],
    "support_revenue_gbp": [50000.0, 0.0, 0.0], "months": 2,
})


def test_the_twin_settlement_reprices_on_the_bills_hours_and_holds_the_rest(monkeypatch, tmp_path):
    from scripts.public_ownership import price_level_settlement as pls
    from scripts.public_ownership import build_asset_settlement
    from scripts.public_ownership.settlement import SETTLEMENT_COLUMNS

    hourly, prices = _hourly_year(tmp_path)
    monkeypatch.setattr(plm, "read_market_year", lambda wholesale, price_paths: hourly.copy())
    seen = {}

    def fake_settle(scenarios, wholesale, balancing, price_paths, *args, **kwargs):
        seen["prices"] = [Path(p) for p in price_paths]
        seen["contract_value_bids"] = kwargs.get("contract_value_bids")
        mapped = pd.concat(pd.read_csv(p, index_col=0, parse_dates=True)["wholesale_price"] for p in price_paths)
        out = SETTLEMENT_ROWS.copy()
        out["wholesale_revenue_gbp"] = out["wholesale_mwh"] * float(mapped.mean())
        out["cfd_support_gbp"] = np.where(out["support_type"].eq("CfD"), 1000.0 * (100.0 - float(mapped.mean())), 0.0)
        out["balancing_settlement_gbp"] = 999.0       # anything: it must be held from the central settlement
        out["fuel_carbon_gbp"] = 1.0
        return out

    monkeypatch.setattr(build_asset_settlement, "write_asset_settlement", fake_settle)

    # Item F24.2: the CfD bid changes come from the extraction's own function; here a stub.
    def fake_changes(wholesale, balancing, price_paths, redispatch, audits, mapped, **kwargs):
        seen["audits"] = list(audits)
        seen["change_mapped"] = mapped
        return pd.DataFrame({"change_gbp": [-400.0], "gross_change_gbp": [-500.0]}, index=["farm"])

    monkeypatch.setattr(pls, "cfd_bid_changes", fake_changes)
    central = tmp_path / "central.csv"
    SETTLEMENT_ROWS.to_csv(central, index=False)
    # Targets given in the months' order under names that do not sort like the sources.
    targets = [tmp_path / "out" / "z_Jan.csv", tmp_path / "out" / "a_Feb.csv"]
    case = {"price_level": "cp32", "price_level_mapping": dict(plm.DEFAULT_SETTINGS)}
    frame, summary = pls.build_at_bill_price_level(
        central, tmp_path / "twin.csv", tmp_path / "twin.json", targets, case=case, scenarios=["m0", "m1"],
        wholesale=["w0", "w1"], balancing=["b0", "b1"], prices=prices,
        redispatch=["m0_redispatch_summary.csv", "m1_redispatch_summary.csv"],
        assumptions="a.json", ro_scheme_summary="ro.json", model_price_year=2024)

    # The mapped hours are the bill's own (the same function, the same zero band).
    expected, _ = pfb.map_year_to_price_level(hourly, case["price_level_mapping"])
    for source, target in zip(prices, targets):
        written = pd.read_csv(target, index_col=0, parse_dates=True)
        model = pd.read_csv(source, index_col=0, parse_dates=True)
        np.testing.assert_allclose(written["wholesale_price"].to_numpy(),
                                   expected.reindex(written.index).to_numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(written["model_wholesale_price"], model["wholesale_price"])
        np.testing.assert_allclose(written["spread"], model["spread"])   # other columns held
    assert seen["prices"] == targets and seen["contract_value_bids"] is None
    # The bid audits default to the ones beside each redispatch summary; the mapped year is the bill's.
    assert [p.name for p in seen["audits"]] == ["m0_contract_value_bids.csv", "m1_contract_value_bids.csv"]
    pd.testing.assert_series_equal(seen["change_mapped"], expected, check_names=False)
    assert summary["totals_gbp"]["cfd_bid_change_at_price_level"] == -400.0
    assert summary["price_level"] == "cp32" and summary["zero_band_gbp_per_mwh"] == pfb.CP32_ZERO_BAND_GBP_PER_MWH
    assert summary["mapped"]["time_weighted_gbp_per_mwh"] == pytest.approx(
        plm.curve_mean(pfb.load_price_level_curve(case["price_level_mapping"])[1]), rel=1e-9)

    written = pd.read_csv(tmp_path / "twin.csv")
    assert list(written.columns) == SETTLEMENT_COLUMNS
    out = written.set_index("asset_id")
    base = SETTLEMENT_ROWS.set_index("asset_id")
    mean = float(expected.mean())
    # Re-priced: the Stage 1 positions and the CfD top-ups.
    np.testing.assert_allclose(out["wholesale_revenue_gbp"], base["wholesale_mwh"] * mean)
    assert out.at["farm", "cfd_support_gbp"] == pytest.approx(1000.0 * (100.0 - mean))
    # Item F24.2: the CfD unit's contract-value bid moves by its change; nothing else in the balancing settlement.
    assert out.at["farm", "balancing_settlement_gbp"] == pytest.approx(base.at["farm", "balancing_settlement_gbp"] - 400.0)
    assert out.at["farm", "cfd_curtailment_compensation_gbp"] == pytest.approx(
        base.at["farm", "cfd_curtailment_compensation_gbp"] - 500.0)
    others = ["ccgt", "battery"]
    for column in ("balancing_settlement_gbp", "cfd_curtailment_compensation_gbp"):
        np.testing.assert_allclose(out.loc[others, column], base.loc[others, column], err_msg=column)
    # Held as solved: fuel, ROCs, DPA and every volume.
    for column in ("fuel_carbon_gbp", "roc_support_gbp", "dpa_support_gbp", "wholesale_mwh", "balancing_mwh"):
        np.testing.assert_allclose(out[column], base[column], err_msg=column)
    # The derived revenues follow: market = wholesale + balancing; support moves by the CfD change.
    np.testing.assert_allclose(out["market_revenue_gbp"], out["wholesale_revenue_gbp"] + out["balancing_settlement_gbp"])
    np.testing.assert_allclose(out["support_revenue_gbp"],
                               base["support_revenue_gbp"] + out["cfd_support_gbp"] - base["cfd_support_gbp"])


def test_a_model_level_case_is_refused_by_the_twin_settlement(tmp_path):
    from scripts.public_ownership import price_level_settlement as pls

    with pytest.raises(ValueError, match="off the model's own price level"):
        pls.build_at_bill_price_level(tmp_path / "c.csv", tmp_path / "o.csv", tmp_path / "s.json", [tmp_path / "p.csv"],
                                      case={}, scenarios=["m"], wholesale=["w"], balancing=["b"],
                                      prices=[tmp_path / "x.csv"], redispatch=["r"], assumptions="a",
                                      ro_scheme_summary="ro", model_price_year=2024)


def test_mapped_price_files_by_directory_are_unchanged(tmp_path):
    hourly, prices = _hourly_year(tmp_path)
    mapped = hourly["price_gbp_per_mwh"] * 2.0
    by_dir = plm.write_mapped_price_files(hourly, mapped, tmp_path / "dir")
    assert [p.name for p in by_dir] == [p.name for p in prices]
    by_target = plm.write_mapped_price_files(hourly, mapped, targets=[tmp_path / "t" / p.name for p in prices])
    for a, b in zip(by_dir, by_target):
        assert a.read_bytes() == b.read_bytes()
    with pytest.raises(ValueError, match="target paths"):
        plm.write_mapped_price_files(hourly, mapped, targets=[tmp_path / "only_one.csv"])


# ── the strike cut's paid energy at the twin's level ──────────────────────────

class _Table:
    strikes = {("AR4", "offshore"): 50.0}

    def adjusted(self, deltas, label=None):
        return self


def test_the_paid_energy_settles_at_the_cases_level_and_a_model_call_is_unchanged(monkeypatch):
    from scripts.public_ownership.build_cfd_strike_adjustments import bill_paid_energy_by_key

    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return None, {"cfd_strike_provenance": {"by_strike": [{"strike_key": "AR4_offshore", "payment_gbp": 1.0}]}}

    monkeypatch.setattr(pfb, "aggregate_inputs", fake)
    assumptions = {"model_linked_rules": {"cfd": {"negative_price_rule": "none"}}}
    inputs = dict(networks=["n.nc"], prices=["p.csv"], summaries=["s.json"], physical_networks=["b.nc"],
                  table=_Table(), links=["l.csv"])
    bill_paid_energy_by_key({"source_scenarios": ["m"], "modelled_year": 2030}, assumptions, **inputs)
    assert all("price_level" not in call and "price_level_mapping" not in call for call in calls)
    calls.clear()
    block = {"pathway": "further_flex_and_renewables"}
    bill_paid_energy_by_key({"source_scenarios": ["m"], "modelled_year": 2030, "price_level": "cp32",
                             "price_level_mapping": block}, assumptions, **inputs)
    assert calls and all(c["price_level"] == "cp32" and c["price_level_mapping"] == block for c in calls)


def test_price_level_settings_come_from_the_forecast_case():
    from scripts.public_ownership.build_cfd_strike_adjustments import price_level_settings

    config = {"consumer_bill": {"forecasts": {"cases": {"A": {}, "B": {"price_level": "cp32", "price_level_mapping": {"x": 1}}}}}}
    assert price_level_settings(config, "A") == {}
    assert price_level_settings(config, "B") == {"price_level": "cp32", "price_level_mapping": {"x": 1}}
    with pytest.raises(ValueError, match=r"not a consumer_bill[_.]forecasts case"):
        price_level_settings(config, "C")


# ── the public account and the bill lines it feeds ─────────────────────────────

def test_the_public_account_states_a_non_model_level_only():
    from scripts.public_ownership.build_public_account import stamp_price_level

    report = {"settings": {}, "cases": {}}
    assert stamp_price_level(report) is report and stamp_price_level(report, "model") is report
    assert stamp_price_level(report, "cp32") == {**report, "price_level": "cp32"}


def _account(path, level=None):
    document = {"cases": {"private": {"ldes_cap_and_floor_enabled": True, "ldes_floor_topup_gbp": 5.0e6,
                                      "battery_revenue_top_up_gbp": 2.0e6}}}
    if level:
        document["price_level"] = level
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_model_accounts_read_exactly_as_before(tmp_path):
    model = _account(tmp_path / "model.json")
    record = pfb.read_ldes_topup(model)
    assert "price_level" not in record and record["topup_gbp"] == 5.0e6
    _, source = pfb.register_booked_cost("battery_revenue_top_up", {"public_account": str(model),
                                                                    "key": "battery_revenue_top_up_gbp"})
    assert "price_level" not in source
    twin = _account(tmp_path / "twin.json", "cp32")
    assert pfb.read_ldes_topup(twin)["price_level"] == "cp32"


def test_account_built_lines_are_repriced_or_held_by_their_accounts_level(tmp_path):
    model, twin = _account(tmp_path / "model.json"), _account(tmp_path / "twin.json", "cp32")
    assumptions = {"sourced_externals": {"battery_revenue_top_up": {"derivation": {"cost_from": {
        "public_account": str(model), "key": "battery_revenue_top_up_gbp"}}}}}
    annual_model = {"price_level": "model"}
    assert pfb.price_level_account_record(annual_model, assumptions, {"enabled": True}) is annual_model
    annual = {"price_level": "cp32", "price_level_held_at_model_price": dict(pfb.CP32_HELD_LINES),
              "price_level_repriced_lines": dict(pfb.CP32_REPRICED_LINES)}
    # A model account: both lines held, and said so.
    held = pfb.price_level_account_record(annual, assumptions, pfb.read_ldes_topup(model))
    assert "ldes_cap_and_floor" in held["price_level_held_at_model_price"]
    assert "battery_revenue_top_up" in held["price_level_held_at_model_price"]
    assert "battery_revenue_top_up" not in held["price_level_repriced_lines"]
    # The twin's account (the line re-read from it): both re-priced.
    with pfb.register_lines_from({"battery_revenue_top_up": str(twin)}):
        moved = pfb.price_level_account_record(annual, assumptions, pfb.read_ldes_topup(twin))
    assert "ldes_cap_and_floor" not in moved["price_level_held_at_model_price"]
    assert {"ldes_cap_and_floor", "battery_revenue_top_up"} <= set(moved["price_level_repriced_lines"])
    assert annual["price_level_held_at_model_price"] == pfb.CP32_HELD_LINES   # the input is not mutated


def test_run_projection_reads_register_lines_from_the_given_accounts_only_during_the_call(monkeypatch):
    seen = {}
    monkeypatch.setattr(pfb, "_run_projection", lambda *a, **k: seen.setdefault("accounts", dict(pfb._REGISTER_LINE_ACCOUNTS)))
    pfb.run_projection("case", register_line_accounts={"battery_revenue_top_up": "twin.json"})
    assert seen["accounts"] == {"battery_revenue_top_up": "twin.json"}
    assert pfb._REGISTER_LINE_ACCOUNTS == {}
    seen.clear()
    pfb.run_projection("case")
    assert seen["accounts"] == {}


# ── the tariff's Stage 1 congestion rent on the twin's prices ─────────────────

def test_congestion_rent_reads_the_given_price_files(monkeypatch, tmp_path):
    import xarray as xr

    from scripts.interconnectors import market_coupling
    from scripts.market import fti_settlement
    from scripts.public_ownership import nationalised_institutions as ni

    snapshots = pd.date_range("2030-01-01", periods=3, freq="h")
    monkeypatch.setattr(ni, "MARKET_DIR", tmp_path)
    xr.Dataset({"snapshots_snapshot": ("snapshots", snapshots),
                "snapshots_generators": ("snapshots", np.ones(3))}).to_netcdf(tmp_path / "m_balancing.nc")
    pd.DataFrame({"IFA": [100.0, -50.0, 0.0]}, index=pd.Index(snapshots, name="snapshot")).to_csv(
        tmp_path / "m_wholesale_links.csv")
    for name, level in (("m_wholesale_price.csv", 10.0), ("mapped.csv", 70.0)):
        pd.DataFrame({"wholesale_price": level}, index=pd.Index(snapshots, name="snapshot")).to_csv(tmp_path / name)
    monkeypatch.setattr(market_coupling, "link_price_tables_from_dataset", lambda ds: (pd.DataFrame(), None))
    monkeypatch.setattr(fti_settlement, "congestion_rent_by_cable",
                        lambda flows, price, links, ask, weights: (flows.mul(price, axis=0)).sum())
    assert ni.measure_congestion_rent(["m"])["IFA"] == pytest.approx(10.0 * 50.0)
    assert ni.measure_congestion_rent(["m"], price_paths=[tmp_path / "mapped.csv"])["IFA"] == pytest.approx(70.0 * 50.0)
    with pytest.raises(ValueError, match="price files"):
        ni.measure_congestion_rent(["m"], price_paths=[])


# ── approved fix: churn_sink never enters the finance register ────────────────

def test_churn_sink_units_are_kept_out_of_the_finance_register():
    from scripts.public_ownership import build_asset_finance_register as bafr

    assert {"churn_sink", "load_shedding"} <= bafr.NON_FINANCE_CARRIERS
    snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
    generators = pd.DataFrame({"carrier": ["wind_offshore", "churn_sink", "load_shedding"],
                               "p_nom": [100.0, 50.0, 1.0e5]}, index=["farm", "sink_bus1", "shed_bus1"])
    dispatch = pd.DataFrame({"farm": [40.0, 60.0], "sink_bus1": [-5.0, 0.0], "shed_bus1": [0.0, 0.0]}, index=snapshots)
    network = SimpleNamespace(generators=generators, generators_t=SimpleNamespace(p=dispatch), snapshots=snapshots,
                              snapshot_weightings=pd.DataFrame({"generators": 1.0}, index=snapshots))
    rows = bafr._component_rows(network, "Generator", 2030, 2025)
    assert [row["asset_id"] for row in rows] == ["farm"]


# ── the three-party identity at cp32 ───────────────────────────────────────────

def test_the_three_party_identity_closes_at_cp32_with_the_models_resource_saving():
    """The twin settlement moves each asset's revenue identically in every case, so the identity closes at cp32.

    CfD units: Stage 1 revenue up, top-up down by the same amount (the strike is fixed);
    merchant plant: Stage 1 revenue up. Price is a transfer, so the resource saving is the
    model level's, and the consumers', Exchequer's and investors' changes still sum to it.
    """
    import tests.unit.test_public_ownership_investor_ledger as led

    def repriced(register):
        out = register.copy()
        wind, nuke = out["asset_id"].eq("wind"), out["asset_id"].eq("nuke")
        out.loc[wind, "market_revenue_gbp"] += 20 * led.M          # mapped reference up ...
        out.loc[wind, "support_revenue_gbp"] -= 20 * led.M         # ... top-up down: strike x energy fixed
        out.loc[nuke, "market_revenue_gbp"] += 50 * led.M          # merchant: higher price, more rent
        out["producer_rent_gbp"] = (out["market_revenue_gbp"] + out["support_revenue_gbp"] - out["fuel_carbon_gbp"]
                                    - out["annual_required_revenue_gbp"])
        out["public_surplus_before_recycling_gbp"] = out["ownership_share"] * out["producer_rent_gbp"]
        return out

    def run(register):
        bill = {"cases": {"private": led._bill_case(),
                          "p2": led._bill_case(ex_vat_gbp=899.8, total_gbp_inc_vat=899.8 * 1.05,
                                               bill_allocated_r1_gbp_2024=20 * led.M)}}
        return led.build(register=register, bill=bill, account=led._account(p2_retained=20 * led.M),
                         network_lines=led.NETWORK, conv=led.CONV, wedges={}, carbon=led.CARBON,
                         investor={"network_public_rate": 0.03})[1]["cases"]["p2"]

    model, cp32 = run(led._register()), run(repriced(led._register()))
    assert abs(cp32["three_party_identity_residual_gbp"]) <= led.inv.IDENTITY_TOLERANCE_GBP
    assert cp32["resource_saving_vs_private_gbp"] == pytest.approx(model["resource_saving_vs_private_gbp"])
    assert cp32["investor_change_vs_private_gbp"] == pytest.approx(model["investor_change_vs_private_gbp"])


# ── the reduced case list of the cp32 twins (user decision, 12 Oct 2026) ─────

@pytest.fixture(scope="module")
def reduced(defaults):
    po, cases = _with_twins(defaults, HEADLINE_SELECTION)
    rr.validate_public_ownership_config(po, cases)
    return po, cases


HEADLINE = ["private", "public_control", "public_ownership", "public_ownership@state_bids_at_cost",
            "public_ownership@fully_nationalised_market_value",
            "public_ownership@fully_nationalised_market_value_central_dispatch"]


def test_the_reduced_twins_keep_the_headline_cases_and_no_grid(reduced):
    from scripts.public_ownership.levers import comparator_for_settings

    po, _ = reduced
    full = rr.configured_case_specs(po)
    by_id = {spec.case_id: spec for spec in full}
    settings = {spec.case_id: dict(spec.settings) for spec in full}
    for twin, parent in TWINS.items():
        kept = rr.expanded_cases(po, twin)
        assert len(rr.expanded_cases(po, parent)) == len(full) > 3500     # the parents keep the full list (3,967 since F29)
        assert set(HEADLINE) <= set(kept)
        assert {rr.case_kind(by_id[c]) for c in kept} >= {"scenario", "named", "preset", "bound"}
        assert not any(rr.case_kind(by_id[c]) == "grid" for c in kept)
        assert len(kept) < 100
        assert kept == [c for c in by_id if c in set(kept)]                    # the full list's order
        # Every kept case's private comparator (on the full list) is kept, and is the same on the kept list.
        kept_settings = {c: settings[c] for c in kept}
        for case in kept:
            assert comparator_for_settings(case, settings) in kept
            assert comparator_for_settings(case, kept_settings) == comparator_for_settings(case, settings)
        # A kept case's settings are the full list's.
        assert all(dict(spec.settings) == settings[spec.case_id] for spec in rr.configured_case_specs(po, twin))


def test_the_twin_roles_and_merges_run_over_kept_cases_only(reduced):
    from scripts.public_ownership.calculate_financing_scenarios import run_group_case_specs
    from scripts.public_ownership.merge_balancing_groups import market_twin

    po, _ = reduced
    market, p2, p3 = (f"{g}_cp32" for g in (MARKET, P2, P3))
    kept = set(rr.expanded_cases(po, market))
    specs = {spec.case_id: spec for spec in rr.configured_case_specs(po, market)}
    # The register (financing) holds the kept cases; a balancing twin holds its own and their comparators.
    assert [s.case_id for s in run_group_case_specs(po, market, list(specs.values()))] == list(specs)
    p2_specs = rr.configured_case_specs(po, p2)
    assert {s.case_id for s in run_group_case_specs(po, p2, p2_specs)} <= kept
    # The bills, the excluded list, the tariff family and the ledger's catalogue are over kept cases.
    for group in (market, p2, p3):
        assert set(rr.group_market_cases(po, group)) <= kept
        assert set(rr.group_excluded_cases(po, group)) <= kept
        assert set(rr.group_tariff_cases(po, group)) <= kept
        assert set(rr.configured_case_catalogue(po, group)) <= kept
    assert rr.group_bid_case(po, p2) == "public_ownership"
    assert rr.group_bid_case(po, p3) == "public_ownership@fully_nationalised_market_value"
    assert set(rr.group_tariff_cases(po, p3)) <= set(rr.group_tariff_cases(po, market))
    assert rr.institution_market_twin(po, market) == ("public_ownership@fully_nationalised_market_value_central_dispatch",
                                                      "public_ownership@fully_nationalised_market_value")
    # The P2 merge finds each bid-stack case's own market twin on the market twin's projected cases.
    summary = {"cases": {c: {"levers": dict(specs[c].settings)} for c in rr.group_market_cases(po, market)}}
    for case in rr.group_market_cases(po, p2):
        if case == "private":
            continue
        twin = market_twin(case, dict(specs[case].settings), summary)
        assert twin in summary["cases"]
        assert {k: v for k, v in specs[twin].settings.items() if k != "balancing_bids"} == {
            k: v for k, v in specs[case].settings.items() if k != "balancing_bids"}


def test_a_selection_naming_an_unknown_case_or_kind_is_refused(defaults):
    po, _ = _with_twins(defaults, {**HEADLINE_SELECTION, "cases": ["public_ownership@no_such_case"]})
    with pytest.raises(ValueError, match="does not expand"):
        rr.expanded_cases(po, f"{MARKET}_cp32")
    po, _ = _with_twins(defaults, {"kinds": ["scenario", "toggles"]})
    with pytest.raises(ValueError, match="unknown kinds"):
        rr.expanded_cases(po, f"{MARKET}_cp32")
    po, _ = _with_twins(defaults, {**HEADLINE_SELECTION, "extra": 1})
    with pytest.raises(ValueError, match="unknown keys"):
        rr.expanded_cases(po, f"{MARKET}_cp32")


def test_a_kept_case_without_its_comparator_is_refused(defaults):
    from scripts.public_ownership.levers import comparator_for_settings

    full = rr.configured_case_specs(defaults["public_ownership"])
    settings = {spec.case_id: dict(spec.settings) for spec in full}
    # A case whose comparator is not plain private (it moves a background lever).
    case = next(s.case_id for s in full if rr.case_kind(s) in ("named", "bound") and not s.case_id.startswith("private")
                and comparator_for_settings(s.case_id, settings) != "private")
    with pytest.raises(ValueError, match="comparators it drops"):
        rr.select_cases(full, {"cases": ["private", case], "comparators": False})
    kept = rr.select_cases(full, {"cases": ["private", case]})                  # added by default
    assert comparator_for_settings(case, settings) in {s.case_id for s in kept}


def test_a_model_price_group_may_reduce_its_list_in_2030_only(defaults):
    """Network bounds (18-19 Oct 2026): the LOWER and UPPER market groups carry the headline cases at model
    price too; a reduced list stays refused on a vintaged (2050) group or a register bound."""
    po = copy.deepcopy(defaults["public_ownership"])
    assert rr.group_case_selection(po, "cp30_2030_cp30_network") is not None
    assert rr.price_level_parent_group(po, "cp30_2030_cp30_network") is None
    po["reference_run_groups"]["historical_2024"]["case_selection"] = copy.deepcopy(HEADLINE_SELECTION)
    po["reference_run_groups"]["fake_2050"] = {"model_year": 2050, "mode": "twelve_months",
                                               "scenario_pattern": "X_{month}2050",
                                               "case_selection": copy.deepcopy(HEADLINE_SELECTION)}
    with pytest.raises(ValueError, match=r"unvintaged \(2030\) groups only"):
        rr.validate_public_ownership_config(po)


def test_a_balancing_twin_whose_market_counterparts_are_dropped_is_refused(defaults):
    po, cases = _with_twins(defaults, HEADLINE_SELECTION)
    # The market twin drops the presets; the P2 twin keeps its state-bids presets, whose twins they are.
    po["reference_run_groups"][f"{MARKET}_cp32"]["case_selection"] = {
        "kinds": ["scenario", "named", "bound"], "cases": ["public_ownership@state_bids_at_cost"]}
    with pytest.raises(ValueError, match="market twin|tariff cases"):
        rr._validate_price_level_twins(po, cases)
