"""F18 (user decision 27 Oct 2026): the 2030 chain's modelled AR8 CfD round.

The ``future`` cohort (a copy of the latest round's strike) became ``AR8_modelled``: a levelised strike calibrated
to the last cleared auctions (AR7 offshore, AR7a onshore, solar and tidal), its own cost-calibration cohort, and a
bill-only range (AR6 clears low, AR8 administrative strikes high).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.consumer_bill.project_future_bill import _combine_strike_rows, summarise_strike_provenance
from scripts.generators import cfd_vintage_labels as labels
from scripts.market import build_ar8_modelled_strikes as ar8
from scripts.market import cfd_strikes as s
from scripts.market.cfd_strike_table import MODELLED_ROUND, load_strike_table, strike_overrides
from scripts.public_ownership import derive_cost_calibration as calib

ROOT = Path(__file__).resolve().parents[2]
STRIKE_TABLE = ROOT / "data" / "market" / "lccc" / "cfd_strike_prices_by_round.csv"
SCHEDULE = ROOT / "data" / "market" / "CP30" / "cfd_vintage_schedule_2030.csv"
ASSUMPTIONS = ROOT / "data" / "market" / "consumer_bill_benchmarks" / "2030_pypsa_informed_assumptions_cfd_rule_none.json"
needs_snapshot = pytest.mark.skipif(not Path(s.DEFAULT_SNAPSHOT).exists(), reason="workbook snapshot not extracted")


@pytest.fixture(scope="module")
def snapshot():
    return s.load_snapshot() if Path(s.DEFAULT_SNAPSHOT).exists() else None


@pytest.fixture(scope="module")
def indexation():
    return json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["cfd_strike_indexation"]


# ── the anchors: AR7 and AR7a reproduce, and the AR7a ones are 2030-only ─────

@needs_snapshot
@pytest.mark.parametrize("technology, year, cleared", [
    ("Offshore wind", 2030, 91.20), ("Onshore wind", 2028, 72.24), ("Solar PV", 2028, 65.23),
    ("Tidal stream", 2028, 265.00)])
def test_the_2030_strike_path_reproduces_ar7_and_ar7a(snapshot, technology, year, cleared):
    strike = s.levelised_strike(technology, year, snapshot=snapshot, model_year=2030)
    assert strike["strike_gbp_per_mwh_2024"] == pytest.approx(cleared, abs=1e-9)


@needs_snapshot
def test_the_ar7a_anchors_never_reach_a_caller_without_model_year_2030(snapshot):
    # The 2050 strike path (and any caller naming no year) keeps offshore-only calibration.
    for model_year in (None, 2040, 2050):
        for technology in ("Onshore wind", "Solar PV", "Tidal stream"):
            assert s.cost_calibration_factor(technology, snapshot=snapshot, model_year=model_year) is None
        assert s.levelised_strike("Offshore wind", 2030, snapshot=snapshot,
                                  model_year=model_year)["strike_gbp_per_mwh_2024"] == pytest.approx(91.20, abs=1e-9)
    assert s.cost_calibration_factor("Onshore wind", snapshot=snapshot, model_year=2030) > 1.0


def test_overlapping_anchors_are_refused(tmp_path):
    table = s.load_cost_calibration()
    assert dict(zip(table["technology"], table["model_years"])) == {
        "Offshore wind": "all", "Onshore wind": "2030", "Solar PV": "2030", "Tidal stream": "2030"}
    clash = pd.concat([table, table.iloc[[1]].assign(model_years="2030;2050")])
    clash.to_csv(tmp_path / "anchors.csv", index=False)
    with pytest.raises(ValueError, match="two anchors for one model year"):
        s.load_cost_calibration(tmp_path / "anchors.csv")
    disjoint = pd.concat([table, table.iloc[[1]].assign(model_years="2050")])
    disjoint.to_csv(tmp_path / "ok.csv", index=False)
    assert len(s.load_cost_calibration(tmp_path / "ok.csv")) == len(table) + 1
    assert s.anchor_applies("all", None) and not s.anchor_applies("2030", None) and s.anchor_applies("2030", 2030)


# ── the strike table: one AR8 strike per technology, no future round ────────

def test_the_strike_table_holds_one_ar8_strike_per_technology_and_no_future_round(indexation):
    frame = pd.read_csv(STRIKE_TABLE)
    assert not frame["cfd_round"].eq("future").any()
    ar8_rows = frame.loc[frame["cfd_round"].eq(MODELLED_ROUND)]
    assert sorted(ar8_rows["technology"]) == sorted(ar8.ROW_ORDER)
    assert not ar8_rows["technology"].duplicated().any()
    table = load_strike_table(STRIKE_TABLE, indexation, 2024)
    # Published rounds unchanged: AR7 offshore 91.20 and AR7a's clears in 2024 money.
    assert table.lookup("wind_offshore", "AR7")[0] == pytest.approx(91.20, abs=0.01)
    assert table.lookup("wind_onshore", "AR7a")[0] == pytest.approx(72.24, abs=0.01)
    assert table.lookup("solar_pv", "AR7a")[0] == pytest.approx(65.23, abs=0.01)
    assert table.lookup("marine", "AR7a")[0] == pytest.approx(265.00, abs=0.01)
    # AR8 is its own strike; an unlabelled unit falls back to it and says so.
    strike, key, fallback = table.lookup("wind_offshore", MODELLED_ROUND)
    assert key == f"{MODELLED_ROUND}_offshore" and not fallback and 91.20 < strike < 95.0
    assert table.lookup("wind_offshore", "")[1:] == (f"{MODELLED_ROUND}_offshore", True)
    assert table.lookup("wind_offshore", "future")[2]  # an old label is a fallback, not a round


@needs_snapshot
def test_the_committed_ar8_files_are_the_builders_output(snapshot):
    strikes = ar8.modelled_strikes(snapshot)
    # The mean of build years 2029 and 2030; offshore 2030 is the AR7 anchor itself.
    assert strikes.at["offshore", "strike_2030_gbp_per_mwh_2024"] == pytest.approx(91.20, abs=1e-9)
    for technology, row in strikes.iterrows():
        assert row["strike_gbp_per_mwh_2024"] == pytest.approx(
            (row["strike_2029_gbp_per_mwh_2024"] + row["strike_2030_gbp_per_mwh_2024"]) / 2)
    for path, text in ar8.build(snapshot, strikes=strikes).items():
        assert path.read_text(encoding="utf-8") == text, path


def test_the_range_twins_bracket_the_central_strike(indexation):
    central = load_strike_table(STRIKE_TABLE, indexation, 2024)
    for case, path in ar8.TWINS.items():
        assumptions = json.loads(path.read_text(encoding="utf-8"))
        cfd = assumptions["model_linked_rules"]["cfd"]
        assert cfd["bm_strike_table"] == cfd["strike_table"]
        # Everything but the CfD block is the central assumptions file.
        base = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
        base["model_linked_rules"]["cfd"] = cfd
        assert base == assumptions
        moved = load_strike_table(STRIKE_TABLE, indexation, 2024, overrides=strike_overrides(assumptions))
        changed = {k for k in central.strikes if central.strikes[k] != moved.strikes[k]}
        assert changed and all(k[0] == MODELLED_ROUND for k in changed)
        for key in changed:
            assert (moved.strikes[key] > central.strikes[key]) == (case == "high"), (case, key)
    high = json.loads(ar8.TWINS["high"].read_text(encoding="utf-8"))
    asp = {o["technology"]: o["strike_price_gbp_per_mwh_2012"] * s.CPI_2012_TO_2024
           for o in high["model_linked_rules"]["cfd"]["strike_overrides"]}
    assert asp == pytest.approx({"offshore": 113.0, "offshore_scotland": 113.0, "floating_offshore": 271.0,
                                 "onshore": 92.0, "solar": 75.0, "tidal": 371.0}, abs=0.001)


# ── nothing keys the old round ──────────────────────────────────────────────

def test_no_table_or_module_keys_the_future_round():
    for path in ("data/market/lccc/cfd_strike_prices_by_round.csv",
                 "data/public_ownership/financing/cfd_round_award_years.csv",
                 "data/public_ownership/financing/private_cost_calibration.csv",
                 "data/market/CP30/cfd_vintage_schedule_2030.csv"):
        frame = pd.read_csv(ROOT / path)
        assert not frame["cfd_round"].astype(str).eq("future").any(), path
    for module in ("scripts/generators/cfd_vintage_labels.py", "scripts/market/cfd_strike_table.py",
                   "scripts/market/cfd_strikes.py", "scripts/market/revenue_tracking.py",
                   "scripts/market/support_bids.py", "scripts/market/support_labels.py",
                   "scripts/public_ownership/commissioning_years.py", "scripts/public_ownership/legislation_year.py",
                   "scripts/public_ownership/derive_cost_calibration.py", "scripts/CP30/brownfield_fleet.py",
                   "scripts/analysis/build_bill_docs_values.py", "scripts/generators/integrate_renewable_generators.py"):
        text = (ROOT / module).read_text(encoding="utf-8")
        assert '"future"' not in text and "'future'" not in text, module


def test_ar8_keeps_the_ar4_negative_price_rule_and_its_award_year():
    from scripts.market.revenue_tracking import POST_AR4_ROUNDS
    from scripts.public_ownership.build_asset_finance_register import contract_vintage, load_award_years
    from scripts.public_ownership.legislation_year import MODELLED_ROUNDS

    assert MODELLED_ROUND in POST_AR4_ROUNDS and MODELLED_ROUND in MODELLED_ROUNDS
    years = load_award_years(ROOT / "data/public_ownership/financing/cfd_round_award_years.csv")
    assert years[MODELLED_ROUND] == 2027
    assert contract_vintage("CfD", MODELLED_ROUND, years, 2025) == ("post_policy", 2027.0)


# ── the fallback and modelled-round shares are reported ─────────────────────

def _details(rounds):
    return pd.DataFrame({"generator": [f"g{i}" for i in range(len(rounds))], "carrier": ["wind_offshore"] * len(rounds),
                         "cfd_round": rounds, "strike_price": [90.0] * len(rounds),
                         "total_dispatch_MWh": [100.0, 100.0, 200.0][:len(rounds)],
                         "net_cfd_cost": [1000.0, 1000.0, 2000.0][:len(rounds)]})


def test_the_bill_reports_the_fallback_and_the_modelled_round_shares():
    provenance = summarise_strike_provenance(_details(["AR4", MODELLED_ROUND, MODELLED_ROUND]), {}, [])
    assert provenance["fallback_strike_share_of_payment"] == 0.0
    assert provenance["modelled_round"] == MODELLED_ROUND
    assert provenance["modelled_round_share_of_energy"] == pytest.approx(0.75)
    assert provenance["modelled_round_share_of_payment"] == pytest.approx(0.75)
    unlabelled = summarise_strike_provenance(_details(["AR4", ""]), {}, ["cfd_round"])
    assert unlabelled["fallback_strike_share_of_payment"] == pytest.approx(0.5)
    combined = _combine_strike_rows(provenance["by_strike"], [])
    assert combined["modelled_round_share_of_payment"] == pytest.approx(0.75)


# ── the cost calibration: AR8 is its own cohort, both ways ──────────────────

def _register(rows, factors=None):
    """Private new-build offshore rows: (round, revenue); capital 500, other 200 per row."""
    factors = factors or {}
    out = []
    for cfd_round, revenue in rows:
        k = factors.get(cfd_round, 1.0)
        out.append({"case_id": "private", "evidence_status": "resolved", "carrier": "wind_offshore",
                    "cfd_round": cfd_round, "finance_vintage": "new_build", "capacity_mw": 10.0, "dispatch_mwh": 10.0,
                    "market_revenue_gbp": revenue, "support_revenue_gbp": 0.0, "fuel_carbon_gbp": 0.0,
                    "annual_capital_recovery_gbp": 500.0 * k, "annual_insurance_gbp": 0.0,
                    "annual_required_revenue_gbp": 500.0 * k + 200.0, "capex_calibration_factor": k})
    return pd.DataFrame(out)


def test_the_ar8_cohort_is_calibrated_apart_from_ar7_and_both_ways():
    # AR7 below its cost (k < 1); AR8 above it (its modelled strike over the cost): k > 1, rent zero.
    register = _register([("AR7", 600.0), (MODELLED_ROUND, 900.0)])
    table = calib.derive(register, "test").set_index("cfd_round")
    assert set(table["pooled_rounds"]) == {"AR7", MODELLED_ROUND}
    assert table.at["AR7", "capex_factor"] == pytest.approx(0.8)
    assert table.at[MODELLED_ROUND, "capex_factor"] == pytest.approx(1.4)
    assert table.at[MODELLED_ROUND, "status"] == calib.STATUS_BOTH_WAYS
    # The verify step passes on a register calibrated with the table, and catches a drift.
    recalibrated = _register([("AR7", 600.0), (MODELLED_ROUND, 900.0)],
                             factors=table["capex_factor"].to_dict())
    fresh = calib.derive(recalibrated, "test")
    assert not calib.compare(table.reset_index(), fresh)["drifted"]
    assert calib.compare(table.reset_index().assign(capex_factor=lambda f: f["capex_factor"] + 0.01), fresh)["drifted"]
    # At the factor the AR8 cohort's private rent is zero.
    row = recalibrated.set_index("cfd_round").loc[MODELLED_ROUND]
    assert row["annual_required_revenue_gbp"] == pytest.approx(row["market_revenue_gbp"])


@pytest.mark.parametrize("carrier, technology", [("wind_offshore", "Offshore wind"), ("marine", "Tidal stream"),
                                                 ("wind_onshore", "Onshore wind"), ("solar_pv", "Solar PV")])
def test_every_ar8_technology_has_its_own_cohort(carrier, technology):
    cohorts = [c for c in calib.COHORTS if c[0] == carrier and MODELLED_ROUND in c[2]]
    assert len(cohorts) == 1 and cohorts[0][2] == (MODELLED_ROUND,) and cohorts[0][1] == technology
    assert cohorts[0][4] is True                                   # both ways
    assert all(len(c[2]) == 1 for c in calib.COHORTS)              # nothing is pooled any more


# ── the vintage schedule: floating awards net off their own rows ────────────

def test_floating_projects_net_off_their_own_row_so_fixed_ar7_labels_in_full():
    schedule = labels.load_vintage_schedule(SCHEDULE)
    offshore = schedule.loc[schedule["carrier"].eq("wind_offshore")].set_index(["cfd_round", "cfd_technology"])
    assert offshore.at[("AR7", ""), "capacity_mw"] == pytest.approx(8245.0)
    assert offshore.at[("AR7", "floating_offshore"), "capacity_mw"] == pytest.approx(192.5)
    assert offshore.at[("AR6", "floating_offshore"), "capacity_mw"] == pytest.approx(400.0)
    named = pd.DataFrame({
        "technology": ["wind_offshore"] * 3, "capacity_mw": [100.0, 100.0, 500.0],
        "support_type": ["CfD"] * 3, "cfd_round": ["AR7", "AR7", "AR7"],
        "cfd_technology": ["floating_offshore", "floating_offshore", ""]})
    netted = labels.named_round_capacity(named)
    assert set(netted.columns) >= {"carrier", "cfd_round", "cfd_technology", "capacity_mw"}
    increment = pd.DataFrame({"technology": ["wind_offshore"], "capacity_mw": [40000.0], "support_type": ["CfD"]})
    result = labels.apply_cfd_vintage_labels(increment, SCHEDULE, named_capacity=netted)
    by_round = result.groupby("cfd_round")["capacity_mw"].sum()
    # Fixed AR7: 8,245 less the 500 MW named fixed project; the 200 MW floating does not net off it.
    assert by_round["AR7"] == pytest.approx(8245.0 - 500.0)
    assert by_round.sum() == pytest.approx(40000.0)
    assert MODELLED_ROUND in by_round and "future" not in by_round
    # A terminated-capacity frame without a technology column still nets off the main row.
    terminated = pd.DataFrame({"carrier": ["wind_offshore"], "cfd_round": ["AR7"], "capacity_mw": [45.0]})
    again = labels.apply_cfd_vintage_labels(increment, SCHEDULE,
                                            named_capacity=pd.concat([netted, terminated], ignore_index=True))
    assert again.groupby("cfd_round")["capacity_mw"].sum()["AR7"] == pytest.approx(8245.0 - 545.0)
