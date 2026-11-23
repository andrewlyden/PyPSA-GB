"""FES 2050 decision 70, the full private-under-central-dispatch case (25 Oct 2026).

The pieces it adds, on small hand-checkable frames: the FTI settlement's CfD terms from a 2050
contract book (several cohorts per network asset), the year settlement's named months, and the
central-dispatch strike calibration and bill assumptions.
"""

import numpy as np
import pandas as pd
import pytest

IDX = pd.date_range("2050-01-01", periods=3, freq="h")


# ── the FTI settlement on a contract book ─────────────────────────────────────

def _book():
    """Two cohorts on wind A (paid shares 0.5 at 80 and 0.25 at 60), one on wind B, a DPA row ignored."""
    return pd.DataFrame({
        "unit": ["A::b1", "A::b2", "B::b1", "G::dpa"],
        "network_asset": ["A", "A", "B", "G"],
        "component_type": ["Generator"] * 4,
        "scheme": ["CfD", "CfD", "CfD", "DPA"],
        "strike_gbp_per_mwh_2024": [80.0, 60.0, 100.0, np.nan],
        "energy_factor": [0.5, 0.25, 1.0, 1.0],
    })


def _audit():
    # The 2050 audit labels a multi-cohort unit "mixed" with one blended strike, which support_terms skips.
    return pd.DataFrame({"generator": ["A", "B"], "support_type": ["mixed", "mixed"],
                         "strike_price_gbp_per_mwh": [73.3, 100.0],
                         "negative_price_rule": ["gb_contract_terms", "none"]})


def _terms():
    from scripts.market.fti_settlement import book_cfd_terms, support_terms

    gens = pd.DataFrame({"carrier": ["wind_onshore", "wind_offshore"], "support_type": ["mixed", "mixed"]},
                        index=["A", "B"])
    base = support_terms(gens, _audit(), roc_buyout=0.0)
    assert base["cfd_strike"].empty            # why the book is needed: mixed units carry no CfD terms
    return book_cfd_terms(base, _book(), _audit())


def test_book_terms_carry_each_cohort_at_its_own_strike_and_paid_share():
    terms = _terms()
    assert sorted(terms["cfd_strike"].index) == ["A", "B"]
    assert terms["cfd_weight"]["A"] == pytest.approx(0.75)
    assert terms["cfd_strike"]["A"] == pytest.approx((0.5 * 80 + 0.25 * 60) / 0.75)
    assert terms["cfd_cohorts"].loc["A", 80.0] == pytest.approx(0.5)
    assert bool(terms["cfd_negative_price_rule"]["A"]) and not bool(terms["cfd_negative_price_rule"]["B"])


def test_lost_support_on_a_book_sums_each_cohorts_floored_top_up():
    from scripts.market.fti_settlement import lost_support

    price = pd.Series([-10.0, 70.0, 90.0], index=IDX)
    market = pd.DataFrame({"A": [10.0, 10.0, 10.0], "B": [2.0, 2.0, 2.0]}, index=IDX)
    central = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=IDX)
    s = lost_support(central, market, price, _terms())
    # A (negative-price rule): hour 1 nothing; hour 2 0.5 x 10 + 0 (60 < 70); hour 3 nothing (both below 90).
    assert s.loc["A", "cfd"] == pytest.approx(0.5 * 10 * 10)
    # B (no rule): (100 + 10) x 2 + 30 x 2 + 10 x 2.
    assert s.loc["B", "cfd"] == pytest.approx(110 * 2 + 30 * 2 + 10 * 2)


def test_support_on_output_on_a_book_pays_the_weighted_strike_on_the_paid_share():
    from scripts.market.fti_settlement import support_on_output

    price = pd.Series([50.0, 50.0, 50.0], index=IDX)
    output = pd.DataFrame({"A": [4.0, 4.0, 4.0], "B": [0.0, 0.0, 0.0]}, index=IDX)
    s = support_on_output(output, price, _terms())
    assert s.loc["A", "cfd"] == pytest.approx(3 * 4 * (0.5 * (80 - 50) + 0.25 * (60 - 50)))


def test_lost_volume_counts_only_the_paid_share_on_a_book():
    from scripts.market.fti_settlement import lost_volume_mwh

    market = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=IDX)
    central = pd.DataFrame({"A": [0.0, 5.0, 10.0]}, index=IDX)
    terms = _terms()
    assert lost_volume_mwh(central, market, ["A"]) == pytest.approx(15.0)
    assert lost_volume_mwh(central, market, ["A"], terms["cfd_weight"]) == pytest.approx(0.75 * 15.0)


def test_book_terms_refuse_a_cfd_row_without_a_strike():
    from scripts.market.fti_settlement import book_cfd_terms

    book = _book()
    book.loc[0, "strike_gbp_per_mwh_2024"] = np.nan
    with pytest.raises(ValueError, match="without a strike"):
        book_cfd_terms({"roc_value": pd.Series(dtype=float)}, book, _audit())


def test_named_months_replace_the_cp30_stem():
    from scripts.market.fti_settle_year import MONTHS, market_stage_1, month_scenarios

    names = [f"FES25_HT_2050_{m}_market" for m in MONTHS]
    named = month_scenarios(names, "--market-scenarios")
    assert named["Mar"] == "FES25_HT_2050_Mar_market"
    assert market_stage_1("ignored", None, named) == named
    assert month_scenarios(None, "--market-scenarios") is None
    with pytest.raises(ValueError, match="not a Feb month"):
        month_scenarios([names[0], names[0]] + names[2:], "--market-scenarios")
    with pytest.raises(ValueError, match="12 names"):
        month_scenarios(names[:11], "--market-scenarios")


# ── the central-dispatch strikes ─────────────────────────────────────────────

def _calibration():
    return pd.DataFrame({
        "scheme": ["CfD", "CfD", "DPA"], "key": ["PAR7_2031_offshore", "PAR7_2031_onshore", "DPA_2031_gas"],
        "cfd_round": ["PAR7_2031", "PAR7_2031", np.nan], "technology": ["offshore", "onshore", np.nan],
        "calibrated_value": [105.0, 82.0, 300.0], "source": ["derived", "derived", "derived"],
        "auction_margin_share": [0.05, 0.05, 0.0],
    })


def _diagnostic():
    return pd.DataFrame({"key": ["PAR7_2031_onshore", "PAR7_2031_offshore"],
                         "market_strike_gbp_per_mwh": [82.0, 105.0],
                         "central_strike_gbp_per_mwh": [76.4, 103.3]})


def test_central_calibration_moves_only_the_post_ar7_cfd_strikes():
    from scripts.public_ownership.vintage_capital_2050 import central_dispatch_calibration

    out = central_dispatch_calibration(_calibration(), _diagnostic())
    assert out["calibrated_value"].tolist() == pytest.approx([103.3, 76.4, 300.0])
    assert out["auction_margin_share"].tolist() == [0.05, 0.05, 0.0]
    assert "decision 70" in out.loc[0, "source"] and out.loc[2, "source"] == "derived"


def test_central_calibration_refuses_a_stale_or_mismatched_diagnostic():
    from scripts.public_ownership.vintage_capital_2050 import VintageCapitalError, central_dispatch_calibration

    stale = _diagnostic()
    stale.loc[0, "market_strike_gbp_per_mwh"] = 81.0
    with pytest.raises(VintageCapitalError, match="another calibration"):
        central_dispatch_calibration(_calibration(), stale)
    with pytest.raises(VintageCapitalError, match="cohorts"):
        central_dispatch_calibration(_calibration(), _diagnostic().iloc[:1])
    with pytest.raises(VintageCapitalError, match="lacks"):
        central_dispatch_calibration(_calibration(), _diagnostic().drop(columns="central_strike_gbp_per_mwh"))


def _assumptions():
    return {"model_linked_rules": {
        "cfd": {"strike_table": "resources/public_ownership/FES25_HT_2050_cfd_strike_table.csv",
                "bm_strike_table": "resources/public_ownership/FES25_HT_2050_cfd_bm_strike_table.csv",
                "contract_book": "resources/public_ownership/fes_ht_2050_support_contract_book.csv",
                "strike_basis": "Basis."},
        "dpa": {"settlement_summary": "resources/public_ownership/fes_ht_2050_asset_settlement_support.json"}}}


def test_central_assumptions_name_the_case_table_and_book_and_hold_the_rest():
    from scripts.public_ownership.vintage_capital_2050 import central_dispatch_assumptions, central_dispatch_files

    files = central_dispatch_files("FES25_HT_2050_cd")
    parent = _assumptions()
    out = central_dispatch_assumptions(parent, case="FES25_HT_2050_cd", strike_table=files["strike_table"],
                                       contract_book=files["contract_book"])
    cfd = out["model_linked_rules"]["cfd"]
    assert cfd["strike_table"] == "resources/public_ownership/FES25_HT_2050_cd_cfd_strike_table.csv"
    assert cfd["contract_book"] == "resources/public_ownership/FES25_HT_2050_cd_support_contract_book.csv"
    assert cfd["bm_strike_table"] == parent["model_linked_rules"]["cfd"]["bm_strike_table"]
    assert out["model_linked_rules"]["dpa"] == parent["model_linked_rules"]["dpa"]
    assert "Decision 70" in cfd["strike_basis"]
    assert parent["model_linked_rules"]["cfd"]["strike_table"].endswith("FES25_HT_2050_cfd_strike_table.csv")  # not mutated


def test_central_assumptions_refuse_a_bill_without_a_contract_book():
    from scripts.public_ownership.vintage_capital_2050 import VintageCapitalError, central_dispatch_assumptions

    parent = _assumptions()
    del parent["model_linked_rules"]["cfd"]["contract_book"]
    with pytest.raises(VintageCapitalError, match="contract_book"):
        central_dispatch_assumptions(parent, case="X", strike_table="t.csv", contract_book="b.csv")


def test_central_strikes_spec_checks_the_block():
    from scripts.public_ownership.vintage_capital_2050 import VintageCapitalError, central_strikes_spec

    block = {"assumptions_from": "a.json", "market_group": "fes_ht_2050", "diagnostic": "d.csv"}
    case = {"institution": "central_dispatch", "central_dispatch_strikes": block}
    assert central_strikes_spec(case, "FES25_HT_2050_cd") == block
    with pytest.raises(VintageCapitalError, match="central-dispatch case"):
        central_strikes_spec({"central_dispatch_strikes": block}, "FES25_HT_2050_cd")
    with pytest.raises(VintageCapitalError, match="unknown"):
        central_strikes_spec({**case, "central_dispatch_strikes": {**block, "extra": 1}}, "FES25_HT_2050_cd")
    with pytest.raises(VintageCapitalError, match="lacks"):
        central_strikes_spec({**case, "central_dispatch_strikes": {"market_group": "g"}}, "FES25_HT_2050_cd")
