"""Task SETTLE2050 (FES 2050 plan): one settlement path for the 2050 CfDs and DPAs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.market import cfd_strikes_2050 as cfd50
from scripts.market.cfd_strike_table import StrikeTable, load_strike_table
from scripts.market.revenue_tracking import compute_cfd_payments, negative_price_eligibility
from scripts.public_ownership import contract_book_2050 as cb
from scripts.public_ownership import support_regime_2050 as sr
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]
needs_snapshot = pytest.mark.skipif(not Path(cfd50.DEFAULT_SNAPSHOT).exists(), reason="workbook snapshot not extracted")
INDEXATION = {"from_price_year": 2012, "to_price_year": 2024, "from_index": 96.1, "to_index": 133.9,
              "factor": 133.9 / 96.1, "index": "ONS CPI D7BT"}


# ── the strike table ─────────────────────────────────────────────────────────

def _strikes():
    rows = []
    for technology, base in (("Offshore wind", 90.0), ("Floating offshore wind", 150.0), ("Onshore wind", 50.0),
                             ("Small modular reactors (SMRs)", 120.0), ("Hydropower", 200.0)):
        for year in (2035, 2045):
            rows.append({"technology": technology, "build_year": year, "variant": "central",
                         "strike_gbp_per_mwh_2024": base - (year - 2035) * 0.5, "private_rate": 0.08})
    return pd.DataFrame(rows)


def _known():
    return pd.DataFrame([
        {"contract_id": "A", "phase": 1, "cfd_round": "AR7", "technology": "Offshore wind", "region": "England",
         "scheme": "CfD", "live_share": 1.0, "live_mw": 300.0, "strike_gbp_per_mwh_2012": 65.45, "start_year": 2031.25,
         "source": "s"},
        {"contract_id": "B", "phase": 1, "cfd_round": "AR7", "technology": "Offshore wind", "region": "Scotland",
         "scheme": "CfD", "live_share": 0.5, "live_mw": 100.0, "strike_gbp_per_mwh_2012": 64.23, "start_year": 2030.25,
         "source": "s"},
        {"contract_id": "F", "phase": 1, "cfd_round": "AR7", "technology": "Floating offshore wind", "region": "Wales",
         "scheme": "CfD", "live_share": 0.0, "live_mw": 0.0, "strike_gbp_per_mwh_2012": 155.37, "start_year": 2029.25,
         "source": "s"},
        {"contract_id": "H", "phase": 1, "cfd_round": "HPC", "technology": "Large-scale nuclear", "region": "England",
         "scheme": "CfD", "live_share": 1.0, "live_mw": 1630.0, "strike_gbp_per_mwh_2012": 89.5, "start_year": 2030.25,
         "source": "s"},
        {"contract_id": "S", "phase": 1, "cfd_round": "SZC_RAB", "technology": "Large-scale nuclear", "region": "England",
         "scheme": "RAB", "live_share": 1.0, "live_mw": 3260.0, "strike_gbp_per_mwh_2012": np.nan, "start_year": 2039.25,
         "source": "s"}])


def test_strike_table_frame_reads_back_as_one_table(tmp_path):
    frame = cb.strike_table_frame(_strikes(), _known())
    # Floating and hydro are not PAR7 vintages: no collision on (PAR7_<year>, offshore).
    assert set(frame.loc[frame["cfd_round"].str.startswith("PAR7_"), "technology"]) == {"offshore", "onshore", "nuclear"}
    assert not frame["cfd_round"].eq("SZC_RAB").any() and frame["price_basis"].eq("2012 prices").all()
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    table = load_strike_table(path, INDEXATION, 2024)
    assert table.strikes[("PAR7_2035", "offshore")] == pytest.approx(90.0)
    # AR7 offshore collapses to its live-MW-weighted strike (the expired floating phase weighs nothing).
    expected = (65.45 * 300 + 64.23 * 100) / 400 * 133.9 / 96.1
    assert table.strikes[("AR7", "offshore")] == pytest.approx(expected)
    assert table.strikes[("HPC", "nuclear")] == pytest.approx(89.5 * 133.9 / 96.1)
    # strike_table_for_bill is the same table, in memory.
    bill = cfd50.strike_table_for_bill(_strikes(), _known())
    assert set(bill.strikes) == set(table.strikes)
    for key, value in table.strikes.items():
        assert bill.strikes[key] == pytest.approx(value, rel=1e-12)


# ── the book ─────────────────────────────────────────────────────────────────

def _register():
    rows = [
        # A CfD asset split into a PAR7 cohort, an AR7 cohort (part live) and a merchant cohort.
        ("OFF::b2040::offshore", "Generator", "wind_offshore", "CfD", "PAR7_2040", 1.0, 2040, 100.0, "Offshore wind", 0.5),
        ("OFF::b2030::offshore", "Generator", "wind_offshore", "CfD", "AR7", 0.71875, 2030, 60.0, "Offshore wind", 0.3),
        ("OFF::b2020::offshore", "Generator", "wind_offshore", "merchant", None, np.nan, 2020, 40.0, "Offshore wind", 0.2),
        ("CCS::b2040::ccs", "Generator", "CCS Gas", "DPA", "DPA_2040", 1.0, 2040, 500.0,
         "Gas generation - CCUS - mature", 1.0),
    ]
    frame = pd.DataFrame(rows, columns=["asset_id", "component_type", "carrier", "support_type", "cfd_round",
                                        "cfd_live_share", "build_year", "capacity_mw", "stock_flow_technology",
                                        "cohort_share"])
    frame["model_year"] = 2050
    return frame


def _table():
    return StrikeTable({("PAR7_2040", "offshore"): 80.0, ("AR7", "offshore"): 91.2}, 2024, {}, {})


@needs_snapshot
def test_the_book_carries_each_unit_with_its_share_and_terms():
    book = cb.build_contract_book(_register(), _table(), model_year=2050).set_index("unit")
    assert set(book.index) == {"OFF::b2040::offshore", "OFF::b2030::offshore", "CCS::b2040::ccs"}
    par7 = book.loc["OFF::b2040::offshore"]
    assert par7["network_asset"] == "OFF" and par7["strike_gbp_per_mwh_2024"] == 80.0
    assert par7["energy_factor"] == pytest.approx(0.5)
    assert book.loc["OFF::b2030::offshore", "energy_factor"] == pytest.approx(0.3 * 0.71875)
    ccs = book.loc["CCS::b2040::ccs"]
    contract = sr.dpa_contract("CCS Gas", 2040, 1.0, snapshot=cfd50.load_snapshot())
    assert ccs["dpa_apr_gbp_per_kw_year"] == pytest.approx(contract.apr_gbp_per_kw_year)
    assert ccs["energy_factor"] == 1.0


def test_the_book_refuses_a_cfd_without_a_2050_strike():
    register = _register().iloc[:1].copy()
    register["cfd_round"] = "PAR7_2041"
    with pytest.raises(cb.ContractBookError, match="no 2050 strike"):
        cb.build_contract_book(register, _table(), model_year=2050)


def _book_frame():
    return pd.DataFrame([
        {"unit": "OFF::b2040", "network_asset": "OFF", "component_type": "Generator", "carrier": "wind_offshore",
         "scheme": "CfD", "cfd_round": "PAR7_2040", "energy_factor": 0.5, "settlement_share": 0.5, "live_share": 1.0},
        {"unit": "OFF::b2030", "network_asset": "OFF", "component_type": "Generator", "carrier": "wind_offshore",
         "scheme": "CfD", "cfd_round": "AR7", "energy_factor": 0.2, "settlement_share": 0.4, "live_share": 0.5}])


def test_cfd_units_settle_through_the_bill_calculator():
    index = pd.date_range("2050-01-01", periods=3, freq="h")
    generators = pd.DataFrame({"carrier": ["wind_offshore"], "bus": ["b"]}, index=["OFF"])
    dispatch = pd.DataFrame({"OFF": [100.0, 50.0, 0.0]}, index=index)
    units, unit_dispatch = cb.cfd_units(_book_frame(), generators, dispatch)
    assert unit_dispatch["OFF::b2040"].tolist() == [50.0, 25.0, 0.0]
    assert unit_dispatch["OFF::b2030"].tolist() == pytest.approx([20.0, 10.0, 0.0])
    price = pd.Series([40.0, -5.0, 60.0], index=index)
    network = SimpleNamespace(generators=units, snapshots=index, generators_t=SimpleNamespace(p=unit_dispatch),
                              snapshot_weightings=pd.DataFrame({"generators": 1.0}, index=index))
    out = compute_cfd_payments(network, _table(), reference_price=price,
                               negative_price_rule="gb_contract_terms").set_index("generator")
    # PAR7 and AR7 keep the AR4 rule: the -5 hour pays nothing.
    assert out.loc["OFF::b2040", "net_cfd_cost"] == pytest.approx(50 * (80 - 40))
    assert out.loc["OFF::b2030", "net_cfd_cost"] == pytest.approx(20 * (91.2 - 40))
    missing = pd.DataFrame({"carrier": ["solar_pv"], "bus": ["b"]}, index=["SOL"])
    with pytest.raises(cb.ContractBookError):
        cb.cfd_units(_book_frame(), missing, None)


def test_post_ar7_rounds_keep_the_ar4_negative_price_rule():
    eligible = negative_price_eligibility(np.array([10.0, -1.0]), ["PAR7_2045", "HPC"], "gb_contract_terms")
    assert eligible[:, 0].tolist() == [True, False]
    assert eligible[:, 1].tolist() == [True, True]


@pytest.mark.parametrize("carrier", ["CCS Gas", "Hydrogen"])
def test_dpa_book_matches_the_single_dpa_settlement(carrier):
    """The book's per-asset energy terms give each unit exactly what dpa_settlement gives it alone."""
    index = pd.date_range("2050-01-01", periods=4, freq="h")
    price = pd.Series([150.0, 60.0, 100.0, 300.0], index=index)
    asset = pd.Series([200.0, 0.0, 100.0, 50.0], index=index)
    srmc = pd.Series([70.0, 70.0, 90.0, 120.0], index=index)
    reference, margin = 86.0, 1.0
    contract = sr.DPAContract(carrier=carrier, build_year=2040, capacity_mw=40.0, apr_gbp_per_kw_year=150.0,
                              expected_availability=0.9, term_years=15, live_share=0.5, efficiency=0.4,
                              variable_om_gbp_per_mwh=5.0, residual_emission_tco2_per_mwh=0.0, margin_gbp_per_mwh=margin,
                              gainshare_share=0.5, gainshare_threshold_gbp_per_kw_year=0.0, private_rate=0.08,
                              fixed_cost_gbp_per_kw_year=135.0, source="synthetic")
    k = 0.25
    params = sr.load_parameters()
    gas = 30.0
    carbon = (reference - 2.0) * 0.52 / 0.183 - gas / 0.183     # makes reference_ccgt_srmc == 86
    alone = sr.dpa_settlement(contract, asset * k, price, gas=gas, carbon=carbon, params=params,
                              plant_srmc=srmc.to_numpy())
    terms = cb.dpa_energy_terms(asset.to_frame("H2"), price, srmc.to_frame("H2"),
                                float(sr.reference_ccgt_srmc(gas, carbon, params)), margin)
    book = pd.DataFrame([{"unit": "H2::b2040", "network_asset": "H2", "scheme": "DPA", "energy_factor": k,
                          "dpa_carrier": carrier, "build_year": 2040, "capacity_mw": 40.0,
                          "dpa_apr_gbp_per_kw_year": 150.0, "dpa_expected_availability": 0.9, "dpa_term_years": 15,
                          "live_share": 0.5, "dpa_margin_gbp_per_mwh": margin, "dpa_gainshare_share": 0.5,
                          "dpa_gainshare_threshold_gbp_per_kw_year": 0.0, "dpa_private_rate": 0.08,
                          "dpa_fixed_cost_gbp_per_kw_year": 135.0, "source": "synthetic"}])
    unit = cb.settle_dpa_book(book, terms).iloc[0]
    for key in ("availability_payment_gbp", "variable_payment_gbp", "gainshare_gbp", "net_support_cost_gbp"):
        assert unit[key] == pytest.approx(alone[key]), key
    # Hydrogen power's DPA has no variable payment (user, 26 Oct 2026); gas CCS's does.
    assert (alone["variable_payment_gbp"] > 0) == (carrier == "CCS Gas") and alone["gainshare_gbp"] > 0
    # A part year pays its part of the availability payment.
    part = sr.dpa_payments(replace(contract, live_share=1.0), output_mwh=0.0, variable_top_up_gbp=0.0,
                           market_revenue_gbp=0.0, short_run_cost_gbp=0.0, period_share=0.5)
    assert part["availability_payment_gbp"] == pytest.approx(150.0 * 40e3 * 0.9 * 0.5)


# ── the register's labels ────────────────────────────────────────────────────

def test_modelled_contracts_take_their_technology_terms_and_are_network_settled():
    from scripts.public_ownership import fleet_vintages as fv

    assert fv.modelled_contract_term("Tidal stream", 2040) == 15
    assert fv.modelled_contract_term("Biomass - with CCUS - mature", 2040) == 15
    assert fv.modelled_contract_term("Small modular reactors (SMRs)", 2040) == 35
    assert fv.modelled_contract_term("Offshore wind", 2040) == 20
    assert fv.modelled_contract_term("Gas generation - CCUS - mature", 2040) == 15
    cohorts = pd.DataFrame([
        {"asset_id": f"{t[:1]}{y}", "component_type": "Generator", "stock_flow_technology": t, "build_year": float(y),
         "capacity_mw": 10.0, "network_support_type": None}
        for t, y in (("Tidal stream", 2034), ("Tidal stream", 2035), ("Tidal stream", 2036), ("Offshore wind", 2031),
                     ("Gas generation - CCUS - mature", 2035))])
    empty = pd.DataFrame(columns=["contract_id", "phase", "cfd_round", "technology", "scheme", "capacity_mw",
                                  "start_year", "live_share"])
    out, _ = fv.label_support_contracts(cohorts, empty, model_year=2050)
    out = out.set_index("asset_id")
    assert out.loc["T2034", "support_type"] == "merchant"                 # 2034 + 15 < 2050
    assert out.loc["T2035", "cfd_live_share"] == pytest.approx(0.5)       # expires through 2050
    assert out.loc["T2036", "cfd_live_share"] == 1.0 and out.loc["T2036", "network_support_type"] == "CfD"
    assert out.loc["Offshore wind" == out["stock_flow_technology"]].iloc[0]["cfd_round"] == "PAR7_2031"
    ccs = out.loc[out["stock_flow_technology"].eq("Gas generation - CCUS - mature")].iloc[0]
    assert ccs["support_type"] == "DPA" and ccs["cfd_live_share"] == pytest.approx(0.5)
    assert ccs["network_support_type"] == "DPA"
    assert pd.isna(out.loc["T2034", "network_support_type"])


# ── the register-row settlement ──────────────────────────────────────────────

def test_register_rows_take_their_share_and_their_own_support():
    from scripts.public_ownership.settlement import (
        SETTLEMENT_COLUMNS, attach_settlement, combine_register_rows, register_row_settlement)

    network = pd.DataFrame([{"component_type": "Generator", "asset_id": "OFF", "carrier": "wind_offshore",
                             "support_type": "", "payee_class": "register_asset", "bus": "b", "bus_lat": 55.0,
                             "bus_lon": -2.0, "wholesale_mwh": 1000.0, "balancing_mwh": 900.0,
                             "wholesale_revenue_gbp": 50_000.0, "balancing_settlement_gbp": 2_000.0,
                             "balancing_bid_gbp": 2_000.0, "cfd_support_gbp": 0.0,
                             "cfd_curtailment_compensation_gbp": 0.0, "roc_support_gbp": 0.0, "dpa_support_gbp": 0.0,
                             "fuel_carbon_gbp": 0.0, "fuel_carbon_basis": "no_fuel", "months": 1}])
    register = pd.DataFrame([
        {"component_type": "Generator", "asset_id": "OFF::b2040", "carrier": "wind_offshore", "support_type": "CfD",
         "cohort_share": 0.6},
        {"component_type": "Generator", "asset_id": "OFF::b2020", "carrier": "wind_offshore", "support_type": "merchant",
         "cohort_share": 0.4}])
    cfd = pd.DataFrame([{"unit": "OFF::b2040", "cfd_support_gbp": 7_000.0, "cfd_curtailment_compensation_gbp": 300.0}])
    dpa = pd.DataFrame(columns=["unit", "net_support_cost_gbp"])
    rows = register_row_settlement(network, register, cfd, dpa).set_index("asset_id")
    assert rows.loc["OFF::b2040", "wholesale_revenue_gbp"] == pytest.approx(30_000.0)
    assert rows.loc["OFF::b2020", "wholesale_revenue_gbp"] == pytest.approx(20_000.0)
    assert rows.loc["OFF::b2040", "cfd_support_gbp"] == 7_000.0 and rows.loc["OFF::b2020", "cfd_support_gbp"] == 0.0
    book = pd.DataFrame([{"unit": "OFF::b2040", "network_asset": "OFF", "scheme": "CfD", "energy_factor": 0.6}])
    combined = combine_register_rows(rows.reset_index(), book, priced_on_contract_value=False).set_index("asset_id")
    # A scalar-priced BM: the compensation is added post hoc, on the unit that holds the contract.
    assert combined.loc["OFF::b2040", "support_revenue_gbp"] == pytest.approx(7_300.0)
    assert combined.loc["OFF::b2020", "support_revenue_gbp"] == 0.0
    settlement = combined.reset_index()[SETTLEMENT_COLUMNS + ["settlement_level", "network_asset_id", "settlement_share"]]
    attached = attach_settlement(register.assign(capacity_mw=[60.0, 40.0]), settlement).set_index("asset_id")
    assert attached.loc["OFF::b2040", "support_revenue_gbp"] == pytest.approx(7_300.0)
    assert attached.loc["OFF::b2020", "market_revenue_gbp"] == pytest.approx(20_800.0)
    assert "settlement_share" not in attached.columns
    with pytest.raises(ValueError, match="one source"):
        register_row_settlement(network.assign(cfd_support_gbp=1.0), register, cfd, dpa)


# ── the bill and the audits ──────────────────────────────────────────────────

def test_a_support_label_audit_declares_contract_value_bids(tmp_path):
    from scripts.consumer_bill.extract_future_balancing_costs import (
        audit_declares_contract_value_bids, bid_basis_from_audit, label_priced_cfd_assets,
        negative_price_rule_from_audit)

    audit = pd.DataFrame([
        {"generator": "A", "pricing": "support_labels_2050", "cfd_weight": 1.0, "negative_price_rule": "gb_contract_terms",
         "bid_basis": "contract_value"},
        {"generator": "B", "pricing": "support_labels_2050", "cfd_weight": 0.4, "negative_price_rule": "gb_contract_terms",
         "bid_basis": "contract_value"},
        {"generator": "C", "pricing": "support_labels_2050", "cfd_weight": 0.0, "negative_price_rule": "gb_contract_terms",
         "bid_basis": "contract_value"}])
    path = tmp_path / "audit.csv"
    audit.to_csv(path, index=False)
    assert audit_declares_contract_value_bids(path)
    assert negative_price_rule_from_audit(path) == "gb_contract_terms" and bid_basis_from_audit(path) == "contract_value"
    assert label_priced_cfd_assets(path) == {"A"}
    audit.loc[:, "cfd_weight"] = 0.0
    audit.to_csv(path, index=False)
    assert not audit_declares_contract_value_bids(path)


def test_the_dpa_line_is_optional_and_replaced_by_the_tariff():
    from scripts.consumer_bill.project_future_bill import APPLIED_MODEL_RULES, OPTIONAL_MODEL_RULES
    from scripts.public_ownership.nationalised_tariff import (
        ALL_REPLACED_LINES, REPLACED_LINES, replaced_lines_reported)

    assert "dpa" in OPTIONAL_MODEL_RULES and "dpa" not in APPLIED_MODEL_RULES
    assert "dpa" not in REPLACED_LINES and "dpa" in ALL_REPLACED_LINES
    assert replaced_lines_reported(pd.DataFrame({"component_id": ["cfd"]})) == REPLACED_LINES
    assert "dpa" in replaced_lines_reported(pd.DataFrame({"component_id": ["cfd", "dpa"]}))
    assert cb.read_dpa_line(None) is None


def test_the_2050_bill_names_the_settlement_files():
    import json

    doc = json.loads((ROOT / "data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json")
                     .read_text(encoding="utf-8"))
    cfd = doc["model_linked_rules"]["cfd"]
    assert cfd["strike_table"] == "resources/public_ownership/FES25_HT_2050_cfd_strike_table.csv"
    assert cfd["contract_book"] == "resources/public_ownership/fes_ht_2050_support_contract_book.csv"
    assert cfd["strike_overrides"] == []
    assert doc["model_linked_rules"]["dpa"]["settlement_summary"].endswith("fes_ht_2050_asset_settlement_support.json")
    # The 2030 bill names none of them.
    doc30 = json.loads((ROOT / "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
                       .read_text(encoding="utf-8"))
    assert "contract_book" not in doc30["model_linked_rules"]["cfd"] and "dpa" not in doc30["model_linked_rules"]


# ── guards ───────────────────────────────────────────────────────────────────

def test_a_study_group_handed_defaults_is_refused():
    import yaml

    from scripts.public_ownership.reference_runs import configured_group

    defaults = load_defaults()["public_ownership"]
    assert configured_group(defaults, "cp30_2030_cp30_network") == "cp30_2030_cp30_network"
    assert configured_group(defaults, "a_scratch_name") is None
    with pytest.raises(ValueError, match="study"):
        configured_group(defaults, "fes_ht_2050")


def test_strikes_read_the_study_config_insurance(tmp_path):
    import yaml

    block = cfd50.finance_insurance_config()
    assert block["enabled"] and block["share_of_capex_per_year"] == 0.004
    # A resolved config file wins over the study default.
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"public_ownership": {"generator_costs": {"insurance": {
        "enabled": True, "share_of_capex_per_year": 0.01}}}}), encoding="utf-8")
    assert cfd50.finance_insurance_config(path)["share_of_capex_per_year"] == 0.01


def test_run_bundle_refuses_a_vintaged_group():
    from scripts.public_ownership.run_financing import run_bundle

    with pytest.raises(ValueError, match="vintaged"):
        run_bundle(ROOT / "config", "fes_ht_2050")
