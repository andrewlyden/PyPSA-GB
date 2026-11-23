"""FES 2050 decision 71 (user, 4 Oct 2026): the auction margin in the post-AR7 strike calibration (synthetic tests)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.public_ownership import vintage_capital_2050 as vk
from config.config_loader import load_scenario_entries

ROOT = Path(__file__).resolve().parents[2]
KEY = ("PAR7_2040", "onshore")


class _Table:
    """A strike table with one post-AR7 entry (the bill's ``StrikeTable`` interface the calibration uses)."""

    strikes = {KEY: 50.0, ("PAR7_2041", "offshore"): 90.0}

    def resolve(self, carrier, rnd):
        return (rnd, "onshore" if carrier == "wind_onshore" else "offshore"), False


def _register(rent=(-3.0e6, -1.0e6, 2.5e6, -2.0e6, 4.0e5)):
    return pd.DataFrame({
        "case_id": "private", "asset_id": ["a", "b", "c", "d", "s"],
        "carrier": ["wind_onshore", "wind_onshore", "wind_offshore", "CCS Gas", "nuclear"],
        "support_type": ["CfD", "CfD", "CfD", "DPA", "RAB"],
        "cfd_round": ["PAR7_2040", "PAR7_2040", "PAR7_2041", "DPA_2040", "SZC_RAB"],
        "producer_rent_gbp": list(rent), "cfd_paid_mwh": [1.0e5, 1.0e5, 3.0e5, 0.0, 0.0]})


BOOK = pd.DataFrame({"unit": ["d"], "scheme": "DPA", "dpa_carrier": "CCS Gas", "dpa_technology": [""],
                     "build_year": [2040], "dpa_apr_gbp_per_kw_year": [300.0], "capacity_mw": [100.0],
                     "dpa_expected_availability": [0.9], "live_share": [1.0]})
BASE = {"strikes": {KEY: 48.0, ("PAR7_2041", "offshore"): 88.0}, "dpa": {("CCS Gas", "", 2040): 290.0}}


def _derive(register=None, **kwargs):
    return vk.derive_calibration(_register() if register is None else register, BOOK, _Table(),
                                 amortisation=vk.CONTRACT_TERM_WITH_TAIL, base_strikes=BASE["strikes"],
                                 base_dpa=BASE["dpa"], source="test", **kwargs).set_index("key")


def _rent_at(frame, key):
    """The cohort's rent at its calibrated strike: rent moves by the paid energy per GBP of strike."""
    r = frame.loc[key]
    return r["rent_at_value_in_use_gbp"] + (r["calibrated_value"] - r["value_in_use"]) * r["denominator"]


# ── the rule ─────────────────────────────────────────────────────────────────

def test_zero_margin_is_the_break_even_calibration_exactly():
    before = _derive()
    at_zero = _derive(auction_margin=0.0)
    pd.testing.assert_frame_equal(before, at_zero)
    cfd = at_zero.loc[at_zero["scheme"].eq("CfD")]
    # Bit for bit the decision-41 (b) expression: value in use less rent over paid energy.
    expected = cfd["value_in_use"] - cfd["rent_at_value_in_use_gbp"] / cfd["denominator"]
    assert (cfd["calibrated_value"] == expected).all()
    assert (at_zero[vk.AUCTION_MARGIN_COLUMN] == 0.0).all()
    for key in cfd.index:
        assert _rent_at(at_zero, key) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("margin", [0.05, 0.10, "central", "high"])
def test_with_a_margin_rent_is_that_share_of_contract_revenue(margin):
    m = vk.auction_margin_share(margin)
    frame = _derive(auction_margin=margin)
    for key in frame.index[frame["scheme"].eq("CfD")]:
        r = frame.loc[key]
        contract_revenue = r["calibrated_value"] * r["denominator"]   # strike x paid MWh
        assert _rent_at(frame, key) / contract_revenue == pytest.approx(m, rel=1e-12)
        assert r[vk.AUCTION_MARGIN_COLUMN] == m
    # The margin strike is the break-even strike over (1 - m).
    even = _derive()
    assert frame.loc["PAR7_2040_onshore", "calibrated_value"] == pytest.approx(
        vk.margin_strike(even.loc["PAR7_2040_onshore", "calibrated_value"], m))


def test_dpa_cohorts_take_no_margin_and_rab_rows_are_never_calibrated():
    even, high = _derive(), _derive(auction_margin=0.10)
    dpa = "DPA_2040_CCS Gas_"
    assert high.loc[dpa, "calibrated_value"] == even.loc[dpa, "calibrated_value"]
    assert high.loc[dpa, vk.AUCTION_MARGIN_COLUMN] == 0.0
    assert _rent_at(high, dpa) == pytest.approx(0.0, abs=1e-6)          # the DPA still breaks even
    assert not high.index.str.contains("RAB").any() and set(high["scheme"]) == {"CfD", "DPA"}
    assert vk.AUCTION_SCHEMES == ("CfD",)
    # A RAB row handed to the re-margin is left exactly as it was.
    table = high.reset_index()
    rab = table.iloc[[0]].assign(key="SZC_RAB", scheme="RAB", calibrated_value=123.0, **{vk.AUCTION_MARGIN_COLUMN: 0.0})
    out = vk.apply_auction_margin(pd.concat([table, rab], ignore_index=True), "high").set_index("key")
    assert out.loc["SZC_RAB", "calibrated_value"] == 123.0 and out.loc["SZC_RAB", vk.AUCTION_MARGIN_COLUMN] == 0.0


def test_the_declared_output_strike_carries_the_margin():
    rows = pd.DataFrame({"capacity_mw": [10.0], "cfd_live_share": [1.0], "dispatch_mwh": [2.0e4],
                         "annual_variable_om_gbp": [1.0e5], "fuel_carbon_gbp": [0.0],
                         "annual_required_revenue_gbp": [1.6e6]})
    even = vk.declared_output_strike(rows, "onshore", 2040)
    high = vk.declared_output_strike(rows, "onshore", 2040, auction_margin=0.10)
    assert high["declared_output_strike"] == pytest.approx(even["declared_output_strike"] / 0.9)


# ── the levels and the basis ─────────────────────────────────────────────────

def test_the_levels_are_the_2030_levers():
    assert vk.AUCTION_MARGIN_LEVELS == {"none": 0.0, "central": 0.05, "high": 0.10}
    try:
        from scripts.public_ownership.calculate_financing_scenarios import LEVER_PARAMETER_DEFAULTS
        from scripts.public_ownership.levers import ALL_LEVERS, LEVER_AUCTION_MARGIN
    except ImportError:
        pytest.skip("the 2030 lever auction_margin is not in this tree")
    assert set(ALL_LEVERS[LEVER_AUCTION_MARGIN].values) == set(vk.AUCTION_MARGIN_LEVELS)
    assert dict(LEVER_PARAMETER_DEFAULTS["auction_margin_share"]) == vk.AUCTION_MARGIN_LEVELS


def test_the_basis_reads_the_margin_and_defaults_to_break_even(tmp_path):
    block = {"amortisation": vk.CONTRACT_TERM, "strike_calibration": "x.csv"}
    assert vk.basis_from_config({vk.CONFIG_KEY: block})[vk.AUCTION_MARGIN_KEY] == 0.0
    assert vk.basis_from_config({vk.CONFIG_KEY: {**block, "auction_margin_share": 0.05}})[vk.AUCTION_MARGIN_KEY] == 0.05
    assert vk.basis_from_config(None)[vk.AUCTION_MARGIN_KEY] == 0.0
    assert vk.auction_margin_share("high") == 0.10
    for bad in (-0.01, 0.5, "huge"):
        with pytest.raises(vk.VintageCapitalError):
            vk.auction_margin_share(bad)


def test_a_calibration_at_another_margin_is_refused():
    central = vk.apply_auction_margin(_derive().reset_index(), "central")
    assert vk.calibration_margin(central) == 0.05
    vk.check_calibration_margin(central, 0.05)
    with pytest.raises(vk.VintageCapitalError, match="re-margin"):
        vk.check_calibration_margin(central, 0.0)
    # A file from before decision 71 has no column: it broke every cohort even.
    old = _derive().reset_index().drop(columns=[vk.AUCTION_MARGIN_COLUMN])
    assert vk.calibration_margin(old) == 0.0
    mixed = central.copy()
    mixed.loc[0, vk.AUCTION_MARGIN_COLUMN] = 0.10
    with pytest.raises(vk.VintageCapitalError, match="mixes"):
        vk.calibration_margin(mixed)


def test_load_fills_the_margin_of_an_older_file(tmp_path):
    path = tmp_path / "old.csv"
    _derive().reset_index().drop(columns=[vk.AUCTION_MARGIN_COLUMN]).to_csv(path, index=False)
    assert (vk.load_calibration(path)[vk.AUCTION_MARGIN_COLUMN] == 0.0).all()


# ── the bound variants ───────────────────────────────────────────────────────

def test_the_bound_variant_files_differ_only_in_the_margin(tmp_path):
    central = vk.apply_auction_margin(_derive(auction_margin=0.05).reset_index(), "central")
    # Re-margining the central file at its own margin reproduces it exactly (the derivation's expression).
    pd.testing.assert_frame_equal(central, _derive(auction_margin=0.05).reset_index())
    variants = {level: vk.apply_auction_margin(central, level) for level in ("none", "high")}
    margin_columns = {"calibrated_value", "declared_output_strike", vk.AUCTION_MARGIN_COLUMN}
    for level, frame in variants.items():
        same = [c for c in central.columns if c not in margin_columns]
        pd.testing.assert_frame_equal(frame[same], central[same])
        dpa = frame["scheme"].eq("DPA")
        assert (frame.loc[dpa, "calibrated_value"] == central.loc[dpa, "calibrated_value"]).all()
        m = vk.AUCTION_MARGIN_LEVELS[level]
        assert vk.calibration_margin(frame) == m
        f = frame.set_index("key")
        for key in f.index[f["scheme"].eq("CfD")]:
            assert _rent_at(f, key) == pytest.approx(m * f.loc[key, "calibrated_value"] * f.loc[key, "denominator"],
                                                     rel=1e-12, abs=1e-3)
        vk.check_bound_variant(central, frame, f"{level}.csv")
    # The none variant is exactly the break-even calibration.
    even = _derive().reset_index()
    assert (variants["none"]["calibrated_value"] == even["calibrated_value"]).all()
    # A file written from another register's derivation is refused as a bound variant.
    other = variants["high"].assign(rent_at_value_in_use_gbp=variants["high"]["rent_at_value_in_use_gbp"] + 1.0)
    with pytest.raises(vk.VintageCapitalError, match="re-margined"):
        vk.check_bound_variant(central, other, "high.csv")


def test_the_margin_command_writes_a_variant(tmp_path, capsys):
    src, out = tmp_path / "central.csv", tmp_path / "high.csv"
    vk.apply_auction_margin(_derive().reset_index(), "central").to_csv(src, index=False)
    assert vk._cli(["margin", "--calibration", str(src), "--auction-margin", "high", "--out", str(out)]) == 0
    assert vk.calibration_margin(vk.load_calibration(out)) == 0.10
    assert "0.05 -> 0.1" in capsys.readouterr().out


def test_bound_spec_and_assumptions():
    config = {"public_ownership": {"reference_run_groups": {
        "g_high": {"register_bound_of": "fes_ht_2050",
                   "auction_margin": {"level": "high", "strike_calibration": "data/x_high.csv"}},
        "g_central": {"register_bound_of": "fes_ht_2050",
                      "auction_margin": {"level": "central", "strike_calibration": "data/x.csv"}},
        "g_extra": {"register_bound_of": "fes_ht_2050",
                    "auction_margin": {"level": "none", "strike_calibration": "data/x.csv", "share": 0.0}},
        "g_central_group": {"auction_margin": {"level": "high", "strike_calibration": "data/x.csv"}}}}}
    spec = vk.bound_margin_spec(config, "g_high", root=ROOT)
    assert spec["share"] == 0.10 and spec["parent"] == "fes_ht_2050" and spec["strike_calibration"] == ROOT / "data/x_high.csv"
    for bad in ("g_central", "g_extra", "g_central_group", "missing"):
        with pytest.raises(vk.VintageCapitalError):
            vk.bound_margin_spec(config, bad)
    assumptions = {"model_linked_rules": {
        "cfd": {"strike_table": "resources/public_ownership/FES25_HT_2050_cfd_strike_table.csv",
                "bm_strike_table": "resources/public_ownership/FES25_HT_2050_cfd_bm_strike_table.csv",
                "contract_book": "resources/public_ownership/fes_ht_2050_support_contract_book.csv",
                "strike_basis": "Y11."},
        "dpa": {"settlement_summary": "resources/public_ownership/fes_ht_2050_asset_settlement_support.json"}},
        "sourced_externals": {"x": {"source": "resources/public_ownership/fes_ht_2050_public_account_summary.json"}}}
    out = vk.bound_assumptions(assumptions, parent="fes_ht_2050", group="g_high", strike_table="resources/t.csv")
    cfd = out["model_linked_rules"]["cfd"]
    assert cfd["strike_table"] == "resources/t.csv"
    assert cfd["bm_strike_table"] == assumptions["model_linked_rules"]["cfd"]["bm_strike_table"]   # central labels
    assert cfd["contract_book"] == "resources/public_ownership/g_high_support_contract_book.csv"
    assert out["model_linked_rules"]["dpa"]["settlement_summary"] == \
        "resources/public_ownership/g_high_asset_settlement_support.json"
    assert out["sourced_externals"] == assumptions["sourced_externals"]                           # the parent's
    assert assumptions["model_linked_rules"]["cfd"]["contract_book"].endswith("fes_ht_2050_support_contract_book.csv")
    with pytest.raises(vk.VintageCapitalError):
        vk.bound_assumptions(assumptions, parent="other", group="g_high", strike_table="t.csv")


def test_the_register_bound_compares_the_margin_block():
    from scripts.public_ownership import reference_runs as rr

    assert rr.GROUP_AUCTION_MARGIN_KEY in rr.REGISTER_BOUND_BLOCKS
    parent = {"fleet_vintages": {}, "vintage_costs": {}}
    bound = {**parent, "auction_margin": {"level": "high", "strike_calibration": "x.csv"}}
    assert rr._resolved_bound_blocks(bound) != rr._resolved_bound_blocks(parent)
    assert rr._resolved_bound_blocks(parent)[rr.GROUP_AUCTION_MARGIN_KEY] == {}


# ── the run order and the post-solve pipeline ────────────────────────────────

def test_the_post_solve_pipeline_carries_the_margin(tmp_path):
    from scripts.public_ownership import post_solve_2050 as ps

    assert set(ps.MARGIN_VARIANTS) == {"none", "high"}
    assert set(ps.AUCTION_MARGIN_BOUND_GROUPS) <= set(ps.BOUND_GROUPS)
    assert "build_public_ownership_auction_margin_inputs_2050" in ps.FINANCE_RULES
    for group in ps.AUCTION_MARGIN_BOUND_GROUPS:
        assert any(t.endswith(f"/{group}_nationalised_bill_summary.json") for t in ps.BOUND_TARGETS)
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"public_ownership": {vk.CONFIG_KEY: {
        "amortisation": vk.CONTRACT_TERM, "strike_calibration": "x.csv", "auction_margin_share": 0.05}}}),
        encoding="utf-8")
    bm, new = tmp_path / "bm.csv", tmp_path / "new.csv"
    _derive().reset_index().drop(columns=[vk.AUCTION_MARGIN_COLUMN]).to_csv(bm, index=False)
    _derive(auction_margin=0.05).reset_index().to_csv(new, index=False)
    assert ps.calibration_margins(config, bm, new) == {"configured": 0.05, "bm_basis": 0.0, "rederived": 0.05}


def test_the_ht50_config_when_patched_names_the_margin_and_its_bounds():
    """Live only once the held HT50 patch is applied (the FES25_HT_2050 scenario entry); skipped before."""
    from scripts.public_ownership import post_solve_2050 as ps

    entry = load_scenario_entries(ROOT / "config").get("FES25_HT_2050") or {}
    block = entry.get("public_ownership") or {}
    if vk.AUCTION_MARGIN_KEY not in (block.get(vk.CONFIG_KEY) or {}):
        pytest.skip("the HT50 decision-71 patch is not applied yet")
    assert block[vk.CONFIG_KEY][vk.AUCTION_MARGIN_KEY] == vk.AUCTION_MARGIN_LEVELS["central"]
    groups = block["reference_run_groups"]
    for level, path in ps.MARGIN_VARIANTS.items():
        group = groups[f"fes_ht_2050_auction_margin_{level}_bound"]
        assert group["auction_margin"] == {"level": level, "strike_calibration": path}
        assert vk.bound_margin_spec({"public_ownership": block}, f"fes_ht_2050_auction_margin_{level}_bound")
