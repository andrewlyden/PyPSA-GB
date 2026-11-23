"""REPD support-label hygiene and the sourced CfD round corrections."""

import json

import pandas as pd
import pytest

from scripts.market.cfd_strike_table import ROUND_TECHNOLOGY_ALIASES, load_strike_table
from scripts.renewables.repd_support_labels import (
    apply_cfd_round_corrections, load_cfd_round_corrections, normalise_label,
    normalise_support_labels,
)

CORRECTIONS = "data/market/lccc/cfd_round_corrections.csv"
STRIKE_TABLE = "data/market/lccc/cfd_strike_prices_by_round.csv"
ASSUMPTIONS = "data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json"
INDEXATION_FACTOR = 1.393340270551509


def _table():
    indexation = json.load(open(ASSUMPTIONS, encoding="utf-8"))["cfd_strike_indexation"]
    return load_strike_table(STRIKE_TABLE, indexation, int(indexation["to_price_year"]))


def test_trailing_and_non_breaking_spaces_collapse_to_one_label():
    values = pd.Series(["AR1", "AR1 ", "AR1\xa0", "  AR6  ", None, ""])
    normalised = normalise_label(values)
    assert list(normalised[:4]) == ["AR1", "AR1", "AR1", "AR6"]
    assert normalised.isna().iloc[4] and normalised.isna().iloc[5]
    # The point of the fix: one group, not two.
    assert normalised[:4].nunique() == 2


def test_normalisation_leaves_numeric_ro_banding_alone():
    frame = pd.DataFrame({"cfd_round": ["AR1 "], "ro_banding": [0.9]})
    result = normalise_support_labels(frame)
    assert result.cfd_round.iloc[0] == "AR1"
    assert result.ro_banding.iloc[0] == pytest.approx(0.9)


def test_shipped_corrections_move_hornsea_1_and_beatrice_to_their_contracts():
    corrections = load_cfd_round_corrections(CORRECTIONS)
    frame = pd.DataFrame({
        "repd_ref_id": ["2521", "2525", "2533", "9999"],
        "site_name": ["Beatrice", "Hornsea 1 - Heron & Njord", "Walney 3", "Other"],
        "cfd_round": ["AR1 ", "AR3", "Investment Contract", "AR6"],
        "capacity_mw": [588.0, 1218.0, 659.0, 100.0],
    })
    result = apply_cfd_round_corrections(frame, corrections)
    by_ref = result.set_index("repd_ref_id").cfd_round
    assert by_ref["2521"] == "Investment Contract"
    assert by_ref["2525"] == "Investment Contract"
    # Already right, and rows outside the file are untouched.
    assert by_ref["2533"] == "Investment Contract"
    assert by_ref["9999"] == "AR6"


def test_a_correction_whose_repd_value_has_changed_is_an_error():
    corrections = load_cfd_round_corrections(CORRECTIONS)
    frame = pd.DataFrame({
        "repd_ref_id": ["2525"], "site_name": ["Hornsea 1"], "cfd_round": ["AR6"],
    })
    with pytest.raises(ValueError, match="Recheck the evidence"):
        apply_cfd_round_corrections(frame, corrections)


def test_float_shaped_ref_ids_still_match():
    corrections = load_cfd_round_corrections(CORRECTIONS)
    frame = pd.DataFrame({"repd_ref_id": [2525.0], "cfd_round": ["AR3"]})
    assert apply_cfd_round_corrections(frame, corrections).cfd_round.iloc[0] == \
        "Investment Contract"


def test_corrections_need_a_source(tmp_path):
    path = tmp_path / "c.csv"
    pd.DataFrame({
        "repd_ref_id": ["1"], "site_name": ["x"], "cfd_round_repd": ["AR1"],
        "cfd_round": ["Investment Contract"], "source": [""], "note": [""],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="needs a source"):
        load_cfd_round_corrections(path)


def test_corrections_reject_duplicate_ref_ids(tmp_path):
    path = tmp_path / "c.csv"
    pd.DataFrame({
        "repd_ref_id": ["1", "1"], "site_name": ["x", "y"],
        "cfd_round_repd": ["AR1", "AR1"], "cfd_round": ["A", "B"],
        "source": ["s", "s"], "note": ["", ""],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate Ref IDs"):
        load_cfd_round_corrections(path)


def test_investment_contract_resolves_to_the_published_fider_strike():
    table = _table()
    strike, key, is_fallback = table.lookup("wind_offshore", "Investment Contract")
    assert key == "Investment Contract_offshore" and is_fallback is False
    # NAO Figure 1: GBP150/MWh for the 2017 cohort (Burbo Bank Extension,
    # Dudgeon, Walney Extension) and GBP140 for Beatrice and Hornsea 1, in 2012
    # prices; the round collapses to a capacity-weighted mean like AR1 and AR2.
    weighted_2012 = (150 * (258 + 90 + 210 + 102 + 330 + 330)
                     + 140 * (280 + 384 + 400 + 400 + 400)) / 3184
    assert strike == pytest.approx(weighted_2012 * INDEXATION_FACTOR, rel=1e-9)
    assert 140 * INDEXATION_FACTOR < strike < 150 * INDEXATION_FACTOR
    # It must beat the future-offshore fallback it used to receive by a wide margin.
    fallback, _, _ = table.lookup("wind_offshore", None)
    assert strike > 2 * fallback


def test_ar3_onshore_settles_at_the_remote_island_wind_strike():
    """AR3 had no Pot 1, so onshore capacity labelled AR3 is remote island wind."""
    assert ROUND_TECHNOLOGY_ALIASES[("AR3", "onshore")] == "remote_island_wind"
    table = _table()
    strike, key, is_fallback = table.lookup("wind_onshore", "AR3")
    assert key == "AR3_remote_island_wind" and is_fallback is False
    island, _, _ = table.lookup("wind_onshore", None)
    assert strike != island
    # 39.650 (225.72 MW) and 41.611 (49.50 MW), capacity-weighted, in 2012 prices.
    expected = (39.650 * 225.72 + 41.611 * 49.50) / (225.72 + 49.50)
    assert strike == pytest.approx(expected * INDEXATION_FACTOR, rel=1e-9)


def test_the_alias_never_masks_a_round_that_has_its_own_row():
    table = _table()
    for round_name in ("AR4", "AR5"):
        _, key, _ = table.lookup("wind_onshore", round_name)
        assert key == f"{round_name}_onshore"


def test_every_strike_table_row_still_carries_a_source_and_2012_basis():
    frame = pd.read_csv(STRIKE_TABLE)
    investment = frame[frame.cfd_round.eq("Investment Contract")]
    assert len(investment) == 11
    assert investment.source.str.contains("nao.org.uk").all()
    assert (investment.price_basis == "2012 prices").all()
    assert investment.capacity_mw.sum() == pytest.approx(3184.0)
