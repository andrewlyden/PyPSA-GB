"""Pathway-lite 2030-2050 (FES 2050 plan task Y15, scripts/public_ownership/pathway_2030_2050.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import pathway_2030_2050 as pw
from scripts.public_ownership.finance import capital_recovery_factor, embedded_corporation_tax

ROOT = Path(__file__).resolve().parents[2]
RP = ROOT / "resources" / "public_ownership"


# ── arithmetic ───────────────────────────────────────────────────────────────

def test_crf_matches_finance():
    for r, n in ((0.035, 20), (0.0824, 15), (0.0, 10), (0.1063, 25)):
        assert pw.crf(r, n) == pytest.approx(capital_recovery_factor(r, n), rel=1e-12)
    assert float(pw.crf(0.05, 0)) == 0.0


def test_embedded_tax_matches_finance():
    assert float(pw.embedded_tax(1e9, 0.0824, 0.02, 20)) == pytest.approx(
        embedded_corporation_tax(1e9, 0.0824, 0.02, 20), rel=1e-12)


def test_live_fraction_matrix_is_the_register_rule():
    from scripts.market.capacity_market_2050 import live_fraction

    builds, terms = [2030, 2031, 2035, 2046.0, 2028], [20, 20, 15, 20, 15]
    m = pw.live_fraction_matrix(builds, terms, pw.YEARS)
    for i, (b, t) in enumerate(zip(builds, terms)):
        for j, y in enumerate(pw.YEARS):
            want = live_fraction(b, t, y) if y >= b else 0.0
            assert m[i, j] == pytest.approx(want, abs=1e-12)
    # A 2035 cohort on a 15-year term is half live in 2050, as the register labels it.
    assert m[2, pw.YEARS.index(2050)] == pytest.approx(0.5)


def test_interpolation_is_linear_between_snapshots():
    s = pw.interpolate({2030: 10.0, 2050: 30.0})
    assert s[2030] == 10.0 and s[2050] == 30.0
    assert s[2040] == pytest.approx(20.0)
    assert s[2035] == pytest.approx(15.0)
    assert list(s.index) == list(pw.YEARS)


def test_npv_base_year_undiscounted():
    flat = pd.Series(1.0, index=list(pw.YEARS))
    # User decision (10 Nov 2026): the view runs 2027-2050, discounted to 2027.
    assert pw.npv(flat, 0.0) == pytest.approx(len(pw.YEARS)) and pw.BASE_YEAR == pw.YEARS[0] == 2027
    r = 0.035
    want = sum(1.0 / (1 + r) ** (y - pw.BASE_YEAR) for y in pw.YEARS)
    assert pw.npv(flat, r) == pytest.approx(want)
    assert pw.npv(pd.Series({2027: 5.0}), 0.05) == pytest.approx(5.0)
    assert pw.npv(pd.Series({2028: 1.05}), 0.05) == pytest.approx(1.0)


# ── financing on a toy fleet ─────────────────────────────────────────────────

def _toy():
    capex = 1e9
    rp, rc = 0.08, 0.035
    return pd.DataFrame([
        # A contracted row built 2032, 20-year term, live in 2050 at 1.0 (2032 + 20 > 2051).
        {"tech_group": "offshore_wind", "build_year": 2032, "kind": "term", "term_years": 20, "life_years": 30,
         "live_share_2050": 1.0, "annual_capital_case_gbp": capex * pw.crf(rc, 20),
         "annual_capital_private_gbp": capex * pw.crf(rp, 20), "tax_forgone_gbp": 1e7, "overhead_gbp": 2e6},
        # A pre-2031 CfD row on a 15-year term: recovered 2028-2042, sunk after.
        {"tech_group": "solar", "build_year": 2028, "kind": "expired", "term_years": 15, "life_years": 38,
         "live_share_2050": np.nan, "annual_capital_case_gbp": capex * pw.crf(rc, 15),
         "annual_capital_private_gbp": capex * pw.crf(rp, 15), "tax_forgone_gbp": 5e6, "overhead_gbp": 1e6},
        # A merchant battery built 2030, 15-year life: level 2030-2044, retired after.
        {"tech_group": "storage", "build_year": 2030, "kind": "level", "term_years": np.nan, "life_years": 15,
         "live_share_2050": np.nan, "annual_capital_case_gbp": capex * pw.crf(rc, 15),
         "annual_capital_private_gbp": capex * pw.crf(rp, 15), "tax_forgone_gbp": 3e6, "overhead_gbp": 5e5},
    ])


def test_financing_saving_on_a_toy_fleet():
    rows = _toy()
    f = pw.schedule_financing(rows).set_index(["year", "tech_group"])
    saving = lambda i: rows.at[i, "annual_capital_private_gbp"] - rows.at[i, "annual_capital_case_gbp"]  # noqa: E731
    # Offshore: nothing before 2032, the full level saving 2032-2050.
    assert f.at[(2031, "offshore_wind"), "capital_saving_gbp"] == 0.0
    assert f.at[(2032, "offshore_wind"), "capital_saving_gbp"] == pytest.approx(saving(0))
    assert f.at[(2050, "offshore_wind"), "capital_saving_gbp"] == pytest.approx(saving(0))
    # Solar: 2030-2042 in full, the expiry year 2043 half, then sunk; overhead runs on.
    assert f.at[(2042, "solar"), "capital_saving_gbp"] == pytest.approx(saving(1))
    assert f.at[(2043, "solar"), "capital_saving_gbp"] == pytest.approx(0.5 * saving(1))
    assert f.at[(2044, "solar"), "capital_saving_gbp"] == 0.0
    assert f.at[(2044, "solar"), "tax_forgone_gbp"] == 0.0
    assert f.at[(2050, "solar"), "overhead_gbp"] == pytest.approx(1e6)
    # Battery: 2030-2044, nothing after retirement (capital or overhead).
    assert f.at[(2044, "storage"), "capital_saving_gbp"] == pytest.approx(saving(2))
    assert f.at[(2045, "storage"), "capital_saving_gbp"] == 0.0
    assert f.at[(2045, "storage"), "overhead_gbp"] == 0.0
    # The saving is the rate spread on the same capital.
    assert saving(0) == pytest.approx(1e9 * (capital_recovery_factor(0.08, 20) - capital_recovery_factor(0.035, 20)))


def test_term_row_pays_its_register_share_in_2050():
    rows = _toy().iloc[[0]].copy()
    rows["live_share_2050"] = 0.74
    f = pw.schedule_financing(rows).set_index("year")
    level = rows.iloc[0]["annual_capital_private_gbp"] - rows.iloc[0]["annual_capital_case_gbp"]
    assert f.at[2050, "capital_saving_gbp"] == pytest.approx(0.74 * level)
    assert f.at[2049, "capital_saving_gbp"] == pytest.approx(level)


def test_beyond_2050_continues_each_row_on_its_own_rule():
    """Decision 46 (b): after 2050 a live contract runs to the end of its term, an expired one pays nothing,
    and a merchant row runs to the end of its life; P3's annuity runs to its last payment."""
    rows = _toy()
    rows.loc[2, "build_year"] = 2045                       # a merchant battery alive 2045-2059
    years = pw.beyond_2050_years(rows, last_payment_year=2059)
    assert years[0] == 2051 and years[-1] == 2059          # battery to 2059; offshore to 2052 (2032 + 20)
    f = pw.beyond_2050_financing(rows, years).set_index(["year", "tech_group"])
    saving = lambda i: rows.at[i, "annual_capital_private_gbp"] - rows.at[i, "annual_capital_case_gbp"]  # noqa: E731
    assert f.at[(2051, "offshore_wind"), "capital_saving_gbp"] == pytest.approx(saving(0))
    assert f.at[(2052, "offshore_wind"), "capital_saving_gbp"] == pytest.approx(0.5 * saving(0))
    assert f.at[(2053, "offshore_wind"), "capital_saving_gbp"] == 0.0
    assert f.at[(2051, "solar"), "capital_saving_gbp"] == 0.0
    assert f.at[(2059, "storage"), "capital_saving_gbp"] == pytest.approx(saving(2))
    carried = {"acquisition_year": 2030, "last_payment_year": 2059, "total_model_year_gbp": 4.0e9}
    entry, long_rows = pw.beyond_2050_case(rows, carried, None, key_2050=1e-9)
    assert entry["years"] == [2051, 2059]
    annuity = entry["by_line"]["p3_annuity"]
    assert annuity["consumers"]["undiscounted"] == pytest.approx(-9 * 4.0e9)
    assert annuity["investors"]["undiscounted"] == pytest.approx(9 * 4.0e9)
    assert annuity["consumers"]["npv_stpr_3p5"] == pytest.approx(
        -sum(4.0e9 / 1.035 ** (y - pw.BASE_YEAR) for y in range(2051, 2060)))
    for party_totals in (entry["parties"],):
        resource = party_totals["resource"]["total"]["undiscounted"]
        assert sum(party_totals[p]["total"]["undiscounted"] for p in ("consumers", "exchequer", "investors")) == \
            pytest.approx(resource)
    assert {r["mechanism"] for r in long_rows} == {"beyond_2050"}
    # No carried purchase (a market case): the annuity line is zero.
    entry, _ = pw.beyond_2050_case(rows, None, None, key_2050=1e-9)
    assert entry["by_line"]["p3_annuity"]["consumers"]["undiscounted"] == 0.0


def test_beyond_2050_counts_the_acquired_fleets_cash_beside_the_annuity():
    """User, 13 Nov 2026 (results review, finding 2): both sides after 2050, the annuity and the fleet's own cash,
    the cash to the last payment only; former owners give up the cash, so it reaches consumers."""
    rows = _toy()
    carried = {"acquisition_year": 2030, "last_payment_year": 2055, "total_model_year_gbp": 4.0e9}
    profile = pd.DataFrame({"total": [3.0e9] * 10}, index=range(2051, 2061))      # cash to 2060
    entry, _ = pw.beyond_2050_case(rows, carried, profile, key_2050=1e-9)
    cash = entry["by_line"]["p3_fleet_cash"]
    assert cash["consumers"]["undiscounted"] == pytest.approx(5 * 3.0e9)          # 2051-2055 only
    assert cash["investors"]["undiscounted"] == pytest.approx(-5 * 3.0e9)
    assert cash["resource"]["undiscounted"] == 0.0
    assert entry["acquired_fleet_cash_after_2050"]["counted"] is True
    p = entry["parties"]
    assert sum(p[x]["total"]["undiscounted"] for x in ("consumers", "exchequer", "investors")) == \
        pytest.approx(p["resource"]["total"]["undiscounted"])


def test_p2_cases_are_the_state_bids_cases():
    """Decision 46 (a): P2 and its shares are on their own bid stacks, anchored on the 2030 P2 bid-stack case."""
    p2 = {c: src for c, (label, src) in pw.CASES.items() if label.startswith("P2")}
    assert set(p2) == {f"public_ownership@deployment_share_{p}_state_bids" for p in (25, 50, 75, 100)}
    assert p2["public_ownership@deployment_share_100_state_bids"] == pw.P2_2030_STATE_BIDS
    for pct in (25, 50, 75):
        assert p2[f"public_ownership@deployment_share_{pct}_state_bids"] == (pw.P2_2030_STATE_BIDS, pct / 100)
    assert pw.PRE_AR7_CFD_TERM_YEARS == 15                 # decision 46 (c)


def test_financing_parties_close():
    p = pw.financing_parties(10.0, 3.0, 1.0)
    assert p == {"consumers": 9.0, "exchequer": -3.0, "investors": 0.0, "resource": 6.0}
    pw.check_identity(p, "toy")


def test_legacy_parties_close():
    p = pw.legacy_parties(margin=12.0, annuity=5.0, legacy_tax=2.0, overhead=0.2)
    assert p["investors"] == pytest.approx(-5.0)
    assert p["consumers"] == pytest.approx(12.0 - 5.0 - 0.2)
    pw.check_identity(p, "toy")


# ── anchoring ────────────────────────────────────────────────────────────────

def _record(**over):
    rec = {"resource_saving_vs_private_parts_gbp": {
        "capital charges at the financing party's rate (register)": 10.0,
        "O&M and the state's overhead (register required revenue less its capital charge)": -1.0,
        "less the corporation tax inside the private rate the state does not pay (a transfer from the Exchequer)": -3.0,
        "interconnector re-trading no longer paid abroad": 0.5},
        "investor_change_vs_private_parts_gbp": {
            "legacy operating margin the owners give up (market + support - fuel - O&M on acquired plant)": -8.0,
            "legacy acquisition annuity, to the vendors and the lenders who fund the price (cash basis)": 4.0,
            "new-build producer rent private owners no longer earn (at their required return)": 2.0},
        "corporation_tax_forgone_new_build_gbp": 3.0, "corporation_tax_forgone_legacy_gbp": 1.0,
        "consumer_change_vs_private_gbp": 12.0, "exchequer_net_vs_private_gbp": -5.0,
        "investor_change_vs_private_gbp": -0.5, "resource_saving_vs_private_gbp": 6.5}
    rec.update(over)
    return rec


def test_snapshot_mechanisms_split_and_close():
    m = pw.snapshot_mechanisms(_record())
    assert m["financing"]["resource"] == pytest.approx(6.0)
    assert m["financing"]["exchequer"] == pytest.approx(-3.0)
    assert m["financing"]["consumers"] == pytest.approx(9.0)
    assert m["legacy"]["investors"] == pytest.approx(-4.0)
    assert m["legacy"]["exchequer"] == pytest.approx(-1.0)
    assert m["legacy"]["consumers"] == pytest.approx(5.0)
    for mech in pw.MECHANISMS:
        pw.check_identity(m[mech], mech, tolerance=1e-9)
    for party, column in pw.LEDGER_TOTALS.items():
        assert sum(m[mech][party] for mech in pw.MECHANISMS) == pytest.approx(_record()[column])
    moved = pw.snapshot_mechanisms(_record(), legacy_overhead_gbp=0.3)
    assert moved["financing"]["resource"] == pytest.approx(6.3)
    assert moved["legacy"]["resource"] == pytest.approx(-0.3)


def test_assemble_case_anchors_each_snapshot_exactly():
    snaps = {2030: pw.snapshot_mechanisms(_record()),
             2050: pw.scaled(pw.snapshot_mechanisms(_record()), 2.0)}
    fin = pw.schedule_financing(_toy())
    legacy = pd.DataFrame({"margin": np.linspace(12, 1, len(pw.YEARS)), "annuity": 5.0, "legacy_tax": 1.0,
                           "overhead": 0.1}, index=list(pw.YEARS))
    out = pw.assemble_case("toy", snapshots=snaps, financing=fin, legacy=legacy)
    check = pw.anchor_check(out, {"toy": snaps}, tolerance=1e-6)
    assert check["passes"].all()
    # Every year closes, and between the snapshots the modelled lines move off the straight line.
    for year, g in out.groupby("year"):
        pw.check_identity({p: g.loc[g["party"].eq(p), "gbp"].sum() for p in pw.PARTIES}, str(year), tolerance=1e-6)
    res = out.loc[out["party"].eq("resource")].groupby("year")["gbp"].sum()
    straight = pw.interpolate({2030: res[2030], 2050: res[2050]})
    assert not np.allclose(res.to_numpy(), straight.to_numpy())


def test_scaled_snapshot_for_l4_shares():
    m = pw.scaled(pw.snapshot_mechanisms(_record()), 0.25)
    assert m["financing"]["resource"] == pytest.approx(1.5)
    assert m["market_and_other"]["investors"] == pytest.approx(0.25 * (-0.5 + 4.0))


# ── the real files (skipped when absent) ─────────────────────────────────────

REGISTER = RP / "fes_ht_2050_asset_finance_register.parquet"
LEDGER_2050 = RP / "fes_ht_2050_exchequer_ledger.json"


@pytest.mark.skipif(not (REGISTER.exists() and LEDGER_2050.exists()), reason="2050 register or ledger not built")
def test_2050_register_rows_reproduce_the_ledger_financing():
    """Live-contract rows reproduce the register exactly; the 2050 capital part matches the ledger."""
    import json

    import yaml

    from scripts.public_ownership.calculate_financing_scenarios import canonical_technology, vintage_capital_options
    from scripts.public_ownership.finance import corporation_tax_wedges
    from scripts.public_ownership.register_io import read_register

    cfg = yaml.safe_load((RP / "FES25_HT_2050_finance_config.yaml").read_text(encoding="utf-8"))["public_ownership"]
    basis = vintage_capital_options(cfg, ROOT)
    snap = yaml.safe_load((ROOT / cfg["assumptions_snapshot"]).read_text(encoding="utf-8"))
    wedges = {canonical_technology(k): v for k, v in corporation_tax_wedges(snap).items()}
    private = read_register(REGISTER, columns=list(pw.REGISTER_COLUMNS), cases=["private"])
    case = read_register(REGISTER, columns=list(pw.REGISTER_COLUMNS), cases=["public_ownership"])
    rows = pw.register_rows(case, private, basis=basis, wedges=wedges, cepa_jv=False)
    term = rows["kind"].eq("term")
    got = ((rows.loc[term, "annual_capital_private_gbp"] - rows.loc[term, "annual_capital_case_gbp"])
           * rows.loc[term, "live_share_2050"]).sum()
    want = (rows.loc[term, "register_capital_private_2050_gbp"] - rows.loc[term, "register_capital_case_2050_gbp"]).sum()
    assert got == pytest.approx(want, abs=1.0)
    ledger = json.loads(LEDGER_2050.read_text(encoding="utf-8"))["cases"]["public_ownership"]
    capital = ledger["resource_saving_vs_private_parts_gbp"]["capital charges at the financing party's rate (register)"]
    register = (rows["register_capital_private_2050_gbp"] - rows["register_capital_case_2050_gbp"]).sum()
    assert register == pytest.approx(capital, abs=1e3)
    # Expired rows pay nothing in 2050 on their terms, as the register says.
    assert (rows.loc[rows["kind"].eq("expired"), "register_capital_case_2050_gbp"] == 0).all()


def test_the_legacy_cash_and_overhead_leave_out_what_the_purchase_leaves_private():
    """Review finding 9 (13 Nov 2026): energy-from-waste sits in the valuation's 'thermal' group, but S3 leaves it
    with its owners unless ``efw_acquisition: included``, so its cash and overhead are not the state's."""
    profile = pd.DataFrame({
        "scope_group": ["thermal", "thermal", "valued", "excluded"],
        "carrier": ["waste_to_energy", "CCGT", "wind_offshore", "biogas"],
        "support_type": ["merchant"] * 4, "stream": ["wholesale"] * 4, "year": [2040] * 4,
        "gbp": [100.0, 10.0, 1.0, 1000.0]})
    assert pw.purchase_excluded_carriers({"efw_acquisition": "excluded"}) == ("waste_to_energy",)
    assert pw.purchase_excluded_carriers({}) == ("waste_to_energy",)
    assert pw.purchase_excluded_carriers({"efw_acquisition": "included"}) == ()
    cash = pw.legacy_cash_by_year(profile, years=[2040], exclude_carriers=("waste_to_energy",))
    assert cash.loc[2040, "total"] == pytest.approx(11.0)
    assert pw.legacy_cash_by_year(profile, years=[2040]).loc[2040, "total"] == pytest.approx(111.0)
    schedule = [{"year": y, "scope_group": "thermal", "carrier": c, "operating_mw": mw}
                for y in (2030, 2050) for c, mw in (("waste_to_energy", 50.0), ("CCGT", 20.0))]
    mw = pw.operating_mw_by_year(schedule, years=[2030, 2040, 2050], exclude_carriers=("waste_to_energy",))
    assert list(mw) == pytest.approx([20.0, 20.0, 20.0])


def test_retired_battery_cohorts_built_before_the_law_stay_private():
    """S1/S2 audit (13 Nov 2026): a retired cohort committed before the legislation year (build year less the
    carrier's investment lead) is pre-law plant: private in every case, so it moves no financing line."""
    flows = pd.DataFrame({"technology": ["Lithium batteries"] * 2, "year": [2027, 2031],
                          "gross_addition_mw": [100.0, 100.0]})
    survivors = pd.DataFrame({"technology": [], "build_year": [], "capacity_mw": []})
    register = pd.DataFrame({
        "finance_vintage": ["new_build"], "stock_flow_technology": ["Lithium batteries"], "carrier": ["Battery"],
        "capacity_mw": [50.0], "wacc": [0.03], "wacc_private": [0.07], "ownership_share": [1.0],
        "annual_state_overhead_gbp": [50.0 * 1000 * 3.0], "finance_technology": ["Lithium batteries"]})
    rows, _ = pw.retired_cohort_rows(flows, survivors, register, wedges={}, cepa_jv=False,
                                     legislated_year=2028, lead_years={"Battery": 1.0})
    by = rows.set_index("build_year")
    assert by.loc[2027.0, "ownership_share"] == 0.0 and by.loc[2027.0, "overhead_gbp"] == 0.0
    assert by.loc[2027.0, "annual_capital_case_gbp"] == pytest.approx(by.loc[2027.0, "annual_capital_private_gbp"])
    assert by.loc[2031.0, "ownership_share"] == 1.0
    assert by.loc[2031.0, "annual_capital_case_gbp"] < by.loc[2031.0, "annual_capital_private_gbp"]
