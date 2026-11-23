"""FES 2050 decision 70: the post-AR7 strikes on the central-dispatch year (synthetic tests)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import central_strike_diagnostic_2050 as csd
from scripts.public_ownership import post_solve_2050 as ps
from scripts.public_ownership import vintage_capital_2050 as vc

HOURS = pd.date_range("2050-01-01", periods=4, freq="h")
RULE = "gb_contract_terms"


def _frame(values, columns=("A",)):
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(len(HOURS), len(columns)), index=HOURS,
                        columns=list(columns))


def _weights(assets=("A",), keys=("PAR7_2035_offshore",), w=None):
    w = np.ones((len(assets), len(keys))) if w is None else np.asarray(w, dtype=float)
    return pd.DataFrame(w, index=list(assets), columns=list(keys))


def _cohorts(keys=("PAR7_2035_offshore",), strikes=(80.0,), paid=(0.0,)):
    rows = []
    for k, s, d in zip(keys, strikes, paid):
        rnd, tech = k.rsplit("_", 1)
        rows.append({"key": k, "cfd_round": rnd, "technology": tech, "build_year": int(rnd.split("_")[1]),
                     "market_strike": s, "calibration_paid_mwh": d})
    return pd.DataFrame(rows)


def _diagnose(market_d, central_d, firm, avail, price=(30.0, 30.0, 30.0, 30.0), strike=80.0):
    hours = pd.Series(1.0, index=HOURS)
    w = _weights()
    market = csd.basis_energy(_frame(market_d), _frame(firm), _frame(avail), w, hours)
    central = csd.basis_energy(_frame(central_d), _frame(firm), _frame(avail), w, hours)
    p = pd.Series(price, index=HOURS, dtype=float)
    paid = float(np.sum(market_d))
    return csd.diagnose(_cohorts(strikes=(strike,), paid=(paid,)), market, central, p, p,
                        negative_price_rule=RULE).iloc[0]


# ── the rule ─────────────────────────────────────────────────────────────────

def test_same_output_reproduces_the_market_strike():
    row = _diagnose([100, 80, 50, 0], [100, 80, 50, 0], firm=[100, 80, 50, 0], avail=[100, 100, 100, 100])
    assert row["central_strike_gbp_per_mwh"] == pytest.approx(80.0)
    assert row["strike_difference_gbp_per_mwh"] == pytest.approx(0.0)
    assert row["cfd_line_effect_gbp"] == pytest.approx(0.0)


def test_less_uncompensated_curtailment_lowers_the_strike():
    # The day-ahead schedule spills 20 + 50 + 100 MWh of surplus; central dispatch uses 60 of it.
    firm, avail = [100, 80, 50, 0], [100, 100, 100, 100]
    row = _diagnose(firm, [100, 100, 90, 0], firm=firm, avail=avail)
    assert row["market_uncompensated_mwh"] == pytest.approx(170.0)
    assert row["central_uncompensated_mwh"] == pytest.approx(110.0)
    # Every hour pays: the same contract revenue spread over 290 MWh instead of 230.
    assert row["central_strike_gbp_per_mwh"] == pytest.approx(80.0 * 230.0 / 290.0)
    assert row["strike_difference_gbp_per_mwh"] < 0.0
    assert row["cfd_line_effect_gbp"] == pytest.approx(row["strike_difference_gbp_per_mwh"] * 290.0)


def test_compensated_central_turn_down_does_not_move_the_strike():
    # Central dispatch turns the unit down below its firm day-ahead schedule: firm access restores it.
    firm, avail = [100, 80, 50, 0], [100, 100, 100, 100]
    row = _diagnose(firm, [60, 20, 50, 0], firm=firm, avail=avail)
    assert row["central_gross_curtailed_mwh"] > row["market_gross_curtailed_mwh"]
    assert row["central_compensated_mwh"] == pytest.approx(100.0)
    assert row["central_strike_gbp_per_mwh"] == pytest.approx(80.0)
    # Without firm access the same turn-down would count against the strike.
    assert row["central_strike_without_firm_access_info"] == pytest.approx(80.0 * 230.0 / 130.0)
    # The strike difference is zero, so no money moves, though the compensated volume is reported.
    assert row["cfd_and_compensation_effect_gbp"] == pytest.approx(0.0)


def test_market_bm_turn_down_is_compensated_too():
    # BM turn-down in the market year counts as contract-value output, as the calibration's rent books it.
    firm, avail = [100, 80, 50, 0], [100, 100, 100, 100]
    row = _diagnose([40, 80, 10, 0], firm, firm=firm, avail=avail)
    assert row["market_compensated_mwh"] == pytest.approx(100.0)
    assert row["market_uncompensated_mwh"] == pytest.approx(row["central_uncompensated_mwh"])
    assert row["central_strike_gbp_per_mwh"] == pytest.approx(80.0)
    assert row["market_paid_mwh"] == pytest.approx(130.0)


def test_output_in_a_withheld_negative_price_hour_earns_the_price_not_the_strike():
    # Hour 3 is negative (post-AR4: no difference payment); central runs 50 MWh more there.
    firm, avail = [100, 80, 0, 0], [100, 100, 100, 100]
    price = (30.0, 30.0, -10.0, 30.0)
    row = _diagnose(firm, [100, 80, 50, 0], firm=firm, avail=avail, price=price)
    assert row["central_paid_mwh"] == pytest.approx(180.0)
    assert row["central_contract_value_unpaid_mwh"] == pytest.approx(50.0)
    # B unchanged, A falls by 50 x 10: the strike rises by 500 / 180.
    assert row["central_strike_gbp_per_mwh"] == pytest.approx(80.0 + 500.0 / 180.0)


def test_extra_output_carries_its_variable_cost():
    # Central uses 60 MWh more; at 10 GBP/MWh variable cost the saving is smaller than at zero.
    firm, avail = [100, 80, 50, 0], [100, 100, 100, 100]
    hours = pd.Series(1.0, index=HOURS)
    w = _weights()
    market = csd.basis_energy(_frame(firm), _frame(firm), _frame(avail), w, hours)
    central = csd.basis_energy(_frame([100, 100, 90, 0]), _frame(firm), _frame(avail), w, hours)
    p = pd.Series(30.0, index=HOURS)
    row = csd.diagnose(_cohorts(paid=(230.0,)), market, central, p, p, negative_price_rule=RULE,
                       variable_cost={"PAR7_2035_offshore": 10.0}).iloc[0]
    assert row["variable_cost_shift_gbp"] == pytest.approx(600.0)
    assert row["central_strike_gbp_per_mwh"] == pytest.approx((80.0 * 230.0 + 600.0) / 290.0)
    # Compensated turn-down saves its variable cost: central below firm lowers the strike a little.
    central_down = csd.basis_energy(_frame([60, 80, 50, 0]), _frame(firm), _frame(avail), w, hours)
    row = csd.diagnose(_cohorts(paid=(230.0,)), market, central_down, p, p, negative_price_rule=RULE,
                       variable_cost={"PAR7_2035_offshore": 10.0}).iloc[0]
    assert row["central_strike_gbp_per_mwh"] == pytest.approx((80.0 * 230.0 - 400.0) / 230.0)


def _diagnose_margin(market_d, central_d, firm, avail, margin, strike=80.0, variable_cost=0.0):
    hours = pd.Series(1.0, index=HOURS)
    w = _weights()
    market = csd.basis_energy(_frame(market_d), _frame(firm), _frame(avail), w, hours)
    central = csd.basis_energy(_frame(central_d), _frame(firm), _frame(avail), w, hours)
    p = pd.Series(30.0, index=HOURS)
    cohorts = _cohorts(strikes=(strike,), paid=(float(np.sum(market_d)),)).assign(auction_margin=margin)
    return csd.diagnose(cohorts, market, central, p, p, negative_price_rule=RULE,
                        variable_cost={"PAR7_2035_offshore": variable_cost}).iloc[0]


def test_decision_71_same_output_with_a_margin_reproduces_the_market_strike():
    firm = [100, 80, 50, 0]
    row = _diagnose_margin(firm, firm, firm=firm, avail=[100] * 4, margin=0.05)
    assert row["central_strike_gbp_per_mwh"] == pytest.approx(80.0)
    assert row["auction_margin_share"] == 0.05


def test_decision_71_the_central_strike_earns_the_margin_on_its_own_contract_revenue():
    # Central turns the unit down below its firm schedule (firm access pays 100 MWh): the contract pays on
    # 130 MWh instead of 230, so the 5% margin is earned on a smaller contract revenue.
    firm, avail, m, v = [100, 80, 50, 0], [100] * 4, 0.05, 10.0
    row = _diagnose_margin(firm, [60, 20, 50, 0], firm=firm, avail=avail, margin=m, variable_cost=v)
    s_m, s_c = 80.0, row["central_strike_gbp_per_mwh"]
    q_m, q_c = row["market_paid_mwh"], row["central_paid_mwh"]
    b_m, b_c = row["market_contract_value_paid_mwh"], row["central_contract_value_paid_mwh"]
    assert (q_m, q_c, b_m, b_c) == (230.0, 130.0, 230.0, 230.0)
    # Net contract income less the margin rent is the same on both bases (A = 0: every hour pays).
    lhs = s_c * b_c - v * row["central_delivered_mwh"] - m * s_c * q_c
    rhs = s_m * b_m - v * row["market_delivered_mwh"] - m * s_m * q_m
    assert lhs == pytest.approx(rhs)
    assert row["central_margin_rent_gbp"] == pytest.approx(m * s_c * q_c)
    assert row["market_margin_rent_gbp"] == pytest.approx(m * s_m * q_m)
    # Without the margin term it is the decision-70 break-even formula.
    even = _diagnose_margin(firm, [60, 20, 50, 0], firm=firm, avail=avail, margin=0.0, variable_cost=v)
    assert even["central_strike_gbp_per_mwh"] == pytest.approx((80.0 * 230.0 - 1000.0) / 230.0)
    assert s_c == pytest.approx((80.0 * (230.0 - m * 230.0) - 1000.0) / (230.0 - m * 130.0))
    assert s_c != pytest.approx(even["central_strike_gbp_per_mwh"])


def test_decision_71_market_strikes_read_the_calibration_margin():
    cal = pd.DataFrame({"scheme": ["CfD", "CfD", "DPA"], "key": ["PAR7_2035_offshore", "PAR7_2036_onshore", "DPA_x"],
                        "cfd_round": ["PAR7_2035", "PAR7_2036", "DPA_2040"], "technology": ["offshore", "onshore", ""],
                        "build_year": [2035, 2036, 2040], "calibrated_value": [80.0, 60.0, 300.0],
                        "denominator": [1.0, 2.0, 3.0], "auction_margin_share": [0.05, 0.05, 0.0]})
    assert (csd.market_strikes(cal)["auction_margin"] == 0.05).all()
    assert (csd.market_strikes(cal.drop(columns="auction_margin_share"))["auction_margin"] == 0.0).all()
    with pytest.raises(csd.StrikeDiagnosticError, match="mixes"):
        csd.market_strikes(cal.assign(auction_margin_share=[0.05, 0.10, 0.0]))


def test_register_variable_costs_by_cohort():
    book = pd.DataFrame({"scheme": "CfD", "unit": ["A::b2035", "B::b2035", "B::b2036"],
                         "strike_key": ["PAR7_2035_nuclear", "PAR7_2035_nuclear", "PAR7_2036_nuclear"]})
    register = pd.DataFrame({"case_id": ["private", "private", "private", "public_ownership"],
                             "asset_id": ["A::b2035", "B::b2035", "B::b2036", "A::b2035"],
                             "dispatch_mwh": [100.0, 300.0, 0.0, 100.0],
                             "annual_variable_om_gbp": [500.0, 1500.0, 0.0, 9e9],
                             "fuel_carbon_gbp": [500.0, 1500.0, 0.0, 9e9]})
    v = csd.register_variable_costs(register, book, ["PAR7_2035_nuclear", "PAR7_2036_nuclear"])
    assert v == {"PAR7_2035_nuclear": pytest.approx(10.0)}


def test_offshore_and_onshore_can_move_in_opposite_directions():
    hours = pd.Series(1.0, index=HOURS)
    keys = ("PAR7_2035_offshore", "PAR7_2035_onshore")
    w = _weights(("OFF", "ON"), keys, [[1.0, 0.0], [0.0, 1.0]])
    firm = _frame([[100, 50], [100, 50], [100, 20], [100, 20]], ("OFF", "ON"))
    avail = _frame([[100, 50]] * 4, ("OFF", "ON"))
    # Central turns offshore down (compensated) and runs onshore up (less uncompensated curtailment).
    central_d = _frame([[60, 50], [60, 50], [60, 50], [60, 50]], ("OFF", "ON"))
    market = csd.basis_energy(firm, firm, avail, w, hours)
    central = csd.basis_energy(central_d, firm, avail, w, hours)
    p = pd.Series(30.0, index=HOURS)
    frame = csd.diagnose(_cohorts(keys, (90.0, 60.0), (400.0, 140.0)), market, central, p, p, negative_price_rule=RULE)
    off, on = frame.set_index("technology").loc["offshore"], frame.set_index("technology").loc["onshore"]
    assert off["central_gross_curtailed_mwh"] > off["market_gross_curtailed_mwh"]
    assert off["strike_difference_gbp_per_mwh"] == pytest.approx(0.0)
    assert on["central_gross_curtailed_mwh"] < on["market_gross_curtailed_mwh"]
    assert on["strike_difference_gbp_per_mwh"] < 0.0
    summary = csd.summarise(frame, conversion=None, settings={})
    assert set(summary["groups"]) >= {"offshore", "onshore", "onshore_and_solar"}
    assert summary["reconciliation"]["within_tolerance"]


def test_cohort_weights_follow_the_book_energy_factor():
    book = pd.DataFrame({"scheme": ["CfD", "CfD", "CfD", "DPA"], "component_type": ["Generator"] * 4,
                         "network_asset": ["A", "A", "B", "C"],
                         "strike_key": ["PAR7_2035_offshore", "PAR7_2036_offshore", "PAR7_2035_offshore", ""],
                         "energy_factor": [0.25, 0.5, 1.0, 1.0]})
    w = csd.cohort_weights(book, ["PAR7_2035_offshore", "PAR7_2036_offshore"])
    assert w.loc["A", "PAR7_2035_offshore"] == 0.25 and w.loc["A", "PAR7_2036_offshore"] == 0.5
    assert w.loc["B", "PAR7_2035_offshore"] == 1.0 and w.loc["B", "PAR7_2036_offshore"] == 0.0
    with pytest.raises(csd.StrikeDiagnosticError, match="no unit in the contract book"):
        csd.cohort_weights(book, ["PAR7_2040_solar"])


def test_bill_conversion_uses_the_bill_summary():
    summary = {"cfd_payment_gbp": 2.0e9, "cfd_recovery_denominator_mwh": 6.0e8,
               "final_demand_recovery_denominator_mwh": 6.5e8, "vat_rate": 0.05,
               "diagnostics": {"cfd_line_gbp_per_3100kwh": 2.0e9 / 6.0e8 * 3.1 * 1.0384,
                               "household_archetypes": {"headline_archetype": "hp_ev",
                                                        "bills": [{"archetype": "hp_ev", "annual_consumption_kwh": 6200.0}]}}}
    c = csd.bill_conversion(summary)
    assert c["price_rebase"] == pytest.approx(1.0384)
    assert c["final_demand_per_gbp_per_3100kwh"] == pytest.approx(3.1 * 1.0384 / 6.5e8)
    assert c["headline_kwh"] == 6200.0
    frame = pd.DataFrame({"cfd_line_effect_gbp": [-1.0e8], "compensation_effect_gbp": [0.0]})
    out = csd.add_household_columns(frame, c)
    assert out["cfd_and_compensation_effect_gbp_per_headline_household_ex_vat"].iloc[0] == pytest.approx(
        2.0 * out["cfd_and_compensation_effect_gbp_per_3100kwh_ex_vat"].iloc[0])


# ── end to end on synthetic solved files ─────────────────────────────────────

def _write_month(root, study, month, *, firm, market_d, central_d, avail_pu, p_nom, price):
    import xarray as xr

    d = root / "resources" / "market"
    d.mkdir(parents=True, exist_ok=True)
    gens = list(firm.columns)
    for name, frame in ((f"{study}_{month}_market_wholesale_dispatch.csv", firm),
                        (f"{study}_{month}_market_balancing_dispatch.csv", market_d),
                        (f"{study}_{month}_central_dispatch_balancing_dispatch.csv", central_d)):
        frame.rename_axis("snapshot").to_csv(d / name)
    pd.DataFrame({"wholesale_price": price, "price_spread": 0.0, "hydrogen_price": 36.75},
                 index=HOURS).rename_axis("snapshot").to_csv(d / f"{study}_{month}_market_wholesale_price.csv")
    pd.DataFrame({"generator": gens, "pricing": "cfd_contract_value", "negative_price_rule": RULE,
                  "bid_basis": "strike"}).to_csv(d / f"{study}_{month}_market_contract_value_bids.csv", index=False)
    ds = xr.Dataset({
        "snapshots_snapshot": ("snapshots", HOURS.values), "snapshots_generators": ("snapshots", np.ones(len(HOURS))),
        "generators_i": ("generators_i", gens), "generators_p_nom": ("generators_i", np.asarray(p_nom, dtype=float)),
        "generators_p_max_pu": ("generators_i", np.ones(len(gens))),
        "generators_t_p_max_pu_i": ("generators_t_p_max_pu_i", gens[:1]),
        "generators_t_p_max_pu": (("snapshots", "generators_t_p_max_pu_i"), avail_pu[:, None]),
    })
    ds.to_netcdf(d / f"{study}_{month}_market_wholesale.nc", engine="netcdf4")


def _book(units):
    from scripts.public_ownership.contract_book_2050 import BOOK_COLUMNS

    rows = []
    for unit, asset, key, factor in units:
        row = {c: "" for c in BOOK_COLUMNS}
        rnd = key.rsplit("_", 1)[0]
        row.update({"unit": unit, "network_asset": asset, "component_type": "Generator", "carrier": "wind_offshore",
                    "scheme": "CfD", "cfd_round": rnd, "strike_key": key, "settlement_share": factor,
                    "live_share": 1.0, "energy_factor": factor, "build_year": int(rnd.split("_")[1])})
        rows.append(row)
    return pd.DataFrame(rows, columns=list(BOOK_COLUMNS))


def test_run_writes_the_diagnostic_from_solved_files(tmp_path):
    study = "FES25_HT_2050"
    gens = ["OFF_A", "OFF_B"]
    firm = _frame([[100, 50], [80, 50], [40, 50], [0, 50]], gens)
    market_d = _frame([[100, 30], [80, 50], [40, 50], [0, 50]], gens)        # 20 MWh BM turn-down on B
    central_d = _frame([[100, 50], [100, 50], [70, 50], [0, 50]], gens)      # 50 MWh more on A than firm
    _write_month(tmp_path, study, "Jan", firm=firm, market_d=market_d, central_d=central_d,
                 avail_pu=np.array([1.0, 1.0, 1.0, 1.0]), p_nom=[100.0, 50.0], price=[30.0, 30.0, 30.0, 30.0])
    book = _book([("OFF_A::b2035", "OFF_A", "PAR7_2035_offshore", 1.0),
                  ("OFF_B::b2035", "OFF_B", "PAR7_2035_offshore", 0.5),
                  ("OFF_B::b2036", "OFF_B", "PAR7_2036_offshore", 0.5)])
    (tmp_path / "resources" / "public_ownership").mkdir(parents=True)
    book.to_csv(tmp_path / "resources" / "public_ownership" / "book.csv", index=False)
    # Calibration paid MWh as the settlement computes them: A 220 + B half of 180, and B half of 180.
    cal = pd.DataFrame({"scheme": "CfD", "key": ["PAR7_2035_offshore", "PAR7_2036_offshore"],
                        "cfd_round": ["PAR7_2035", "PAR7_2036"], "technology": "offshore", "build_year": [2035, 2036],
                        "calibrated_value": [90.0, 88.0], "denominator": [310.0, 90.0]})
    cal.to_csv(tmp_path / "cal.csv", index=False)
    frame, summary = csd.run(study=study, months=("Jan",), book_path="resources/public_ownership/book.csv",
                             calibration_path="cal.csv", bill_summary_path=None, out_csv="out/diag.csv",
                             root=tmp_path)
    assert (tmp_path / "out" / "diag.csv").exists() and (tmp_path / "out" / "diag.json").exists()
    row = frame.set_index("key").loc["PAR7_2035_offshore"]
    # Market contract-value energy: A 220 + B 0.5 x 200 = 320; central: A 270 + B 0.5 x 200 = 370.
    assert row["market_contract_value_mwh"] == pytest.approx(320.0)
    assert row["central_contract_value_mwh"] == pytest.approx(370.0)
    assert row["central_strike_gbp_per_mwh"] == pytest.approx(90.0 * 320.0 / 370.0)
    other = frame.set_index("key").loc["PAR7_2036_offshore"]
    assert other["central_strike_gbp_per_mwh"] == pytest.approx(88.0)      # B only: its turn-down was compensated
    assert summary["reconciliation"]["within_tolerance"]
    assert summary["settings"]["market_bm_bid_basis"] == "strike"
    assert summary["materiality"]["material"] is None                    # no bill summary, no per-household figure
    on_disk = json.loads((tmp_path / "out" / "diag.json").read_text(encoding="utf-8"))
    assert on_disk["groups"]["offshore"]["cohorts"] == 2


def test_run_refuses_missing_months(tmp_path):
    cal = pd.DataFrame({"scheme": "CfD", "key": ["PAR7_2035_offshore"], "cfd_round": ["PAR7_2035"],
                        "technology": "offshore", "build_year": [2035], "calibrated_value": [90.0], "denominator": [1.0]})
    cal.to_csv(tmp_path / "cal.csv", index=False)
    _book([("OFF_A::b2035", "OFF_A", "PAR7_2035_offshore", 1.0)]).to_csv(tmp_path / "book.csv", index=False)
    with pytest.raises(csd.StrikeDiagnosticError, match="solved files missing"):
        csd.run(months=("Jan",), book_path="book.csv", calibration_path="cal.csv", bill_summary_path=None,
                out_csv="o.csv", root=tmp_path)


# ── the calibration CLI and the post-solve hook ──────────────────────────────

def test_market_basis_still_needs_the_register():
    with pytest.raises(SystemExit):
        vc._cli(["calibrate", "--book", "book.csv"])


def test_central_basis_refuses_check():
    with pytest.raises(vc.VintageCapitalError, match="diagnostic"):
        vc._cli(["calibrate", "--dispatch-basis", "central", "--book", "book.csv", "--check"])


def test_central_basis_delegates_to_the_diagnostic(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame(), {"settings": {"central_price_basis": "market_copperplate", "negative_price_rule": RULE},
                                "groups": {}, "total": {"market_strike_paid_weighted": 1.0,
                                                        "central_strike_paid_weighted": 1.0,
                                                        "market_uncompensated_twh": 0.0, "central_uncompensated_twh": 0.0,
                                                        "cfd_line_effect_gbp_m": 0.0},
                                "reconciliation": {"worst_cohort_gap": 0.0, "within_tolerance": True}}

    monkeypatch.setattr(csd, "run", fake_run)
    assert vc._cli(["calibrate", "--dispatch-basis", "central", "--book", "b.csv", "--market-calibration", "c.csv",
                    "--out", "o.csv"]) == 0
    assert seen["calibration_path"] == "c.csv" and seen["out_csv"] == "o.csv"
    assert seen["central_suffix"] == "_central_dispatch" and len(seen["months"]) == 12


def test_post_solve_runs_the_diagnostic_after_finance(tmp_path):
    assert ps.STEP_NAMES.index("strike_diagnostic") == ps.STEP_NAMES.index("finance") + 1
    assert ps.select_steps("strike_diagnostic", "strike_diagnostic") == ["strike_diagnostic"]
    assert ps.strike_diagnostic_summary(tmp_path / "missing.json") is None
    diag = {"total": {"market_strike_paid_weighted": 80.0, "central_strike_paid_weighted": 79.0},
            "groups": {"offshore": {"cfd_line_effect_gbp_m": -1.0}, "solar": {}},
            "materiality": {"material": False}, "settings": {"central_price_basis": "market_copperplate"}}
    (tmp_path / "d.json").write_text(json.dumps(diag), encoding="utf-8")
    out = ps.strike_diagnostic_summary(tmp_path / "d.json")
    assert out["total"]["central_strike_paid_weighted"] == 79.0
    assert set(out["groups"]) == {"offshore", "solar"} and out["materiality"]["material"] is False
