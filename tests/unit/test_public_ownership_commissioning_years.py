"""Decision 20: every state-owned CfD unit P3 ranks needs a commissioning year.

The register dated only plant with an operational date, so on the CP30 2030 fleet
5,423 of 5,476 state-owned CfD units had none and every P3 solve refused to start
(25 Sep 2026). ``commissioning_years.py`` dates the pipeline from the LCCC
portfolio, in a table of its own so the register and the P2 share table are
untouched.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from scripts.market.cfd_bm_bids import resolve_tie_break
from scripts.public_ownership.commissioning_years import (
    asset_years, generator_commissioning_years, load_portfolio,
)

LOG = logging.getLogger("test")


def _portfolio(tmp_path):
    path = tmp_path / "portfolio.csv"
    pd.DataFrame([
        # A named AR6 offshore farm with its own expected start.
        {"Name_of_CFD_Unit": "Moor House Wind Farm", "Allocation_Round": "Allocation Round 6",
         "Technology_Type": "Offshore Wind", "Status": "Pre-MDD", "Expected_Start_Date": "2027-10-01 00:00:00.0000000"},
        {"Name_of_CFD_Unit": "Other Farm", "Allocation_Round": "Allocation Round 6",
         "Technology_Type": "Offshore Wind", "Status": "Pre-MDD", "Expected_Start_Date": "2029-04-01 00:00:00.0000000"},
        {"Name_of_CFD_Unit": "Solar A", "Allocation_Round": "Allocation Round 7",
         "Technology_Type": "Solar PV", "Status": "Pre-MDD", "Expected_Start_Date": "2028-01-01 00:00:00.0000000"},
        # AR7's offshore pot delivers after the model year; a terminated unit is ignored.
        {"Name_of_CFD_Unit": "Late Farm", "Allocation_Round": "Allocation Round 7",
         "Technology_Type": "Offshore Wind", "Status": "Pre-MDD", "Expected_Start_Date": "2031-04-01 00:00:00.0000000"},
        {"Name_of_CFD_Unit": "Gone", "Allocation_Round": "Allocation Round 7",
         "Technology_Type": "Solar PV", "Status": "Terminated", "Expected_Start_Date": "2020-01-01 00:00:00.0000000"},
    ]).to_csv(path, index=False)
    return load_portfolio(path)


def _row(asset_id, carrier, support, cfd_round, year=np.nan, share=1.0, capacity=100.0, case="p3"):
    return {"case_id": case, "asset_id": asset_id, "carrier": carrier, "capacity_mw": capacity,
            "ownership_share": share, "commissioning_year": year, "support_type": support,
            "cfd_round": cfd_round, "model_year": 2030}


def test_each_source_is_used_in_order(tmp_path):
    register = pd.DataFrame([
        _row("DATED::legacy", "wind_onshore", "CfD", "AR1", year=2016),
        _row("Moor_House_Wind_Farm_CfD::new_build", "wind_offshore", "CfD", "AR6"),
        _row("Unnamed_offshore::new_build", "wind_offshore", "CfD", "AR6"),
        _row("Rooftop_CfD::new_build", "solar_pv", "CfD", "AR7a"),
        _row("CP30_increment_solar_pv_1_CfD_future::new_build", "solar_pv", "CfD", "AR8_modelled"),
        _row("CHP_1::legacy", "CHP", "merchant", ""),
    ])
    out = asset_years(register, _portfolio(tmp_path)).set_index("asset_id")
    assert out.loc["DATED::legacy", ["tie_break_year", "year_source"]].tolist() == [2016.0, "asset_date"]
    assert out.loc["Moor_House_Wind_Farm_CfD::new_build", "year_source"] == "lccc_unit_expected_start"
    assert out.loc["Moor_House_Wind_Farm_CfD::new_build", "tie_break_year"] == 2027.0
    # The AR6 offshore median of 2027 and 2029.
    assert out.loc["Unnamed_offshore::new_build", ["tie_break_year", "year_source"]].tolist() == [
        2028.0, "lccc_round_technology_median"]
    # AR7a sits in the portfolio's "Allocation Round 7"; the terminated unit is not counted.
    assert out.loc["Rooftop_CfD::new_build", "tie_break_year"] == 2028.0
    assert out.loc["CP30_increment_solar_pv_1_CfD_future::new_build", ["tie_break_year", "year_source"]].tolist() == [
        2030.0, "ar8_round_model_year"]
    # Merchant plant never carries a tie step, so it is left undated.
    assert np.isnan(out.loc["CHP_1::legacy", "tie_break_year"])
    assert out.loc["CHP_1::legacy", "year_source"] == "none"


def test_no_unit_is_dated_after_the_model_year(tmp_path):
    register = pd.DataFrame([_row("AR7_offshore::new_build", "wind_offshore", "CfD", "AR7")])
    out = asset_years(register, _portfolio(tmp_path))
    assert out["tie_break_year"].iloc[0] == 2030.0


def test_the_generator_year_is_weighted_by_state_capacity(tmp_path):
    register = pd.DataFrame([
        _row("MIX::legacy", "wind_onshore", "CfD", "AR1", year=2020, capacity=300.0),
        _row("MIX::new_build", "wind_onshore", "CfD", "AR1", year=2028, capacity=100.0),
        _row("PRIV::legacy", "wind_onshore", "CfD", "AR1", year=2015, share=0.0),
    ])
    table = generator_commissioning_years(register, _portfolio(tmp_path)).set_index("generator")
    assert table.at["MIX", "state_commissioning_year"] == pytest.approx(2022.0)
    assert table.at["MIX", "year_source"] == "asset_date"
    # Nothing state-owned to date.
    assert np.isnan(table.at["PRIV", "state_commissioning_year"])


def test_the_tie_break_ranks_only_the_units_that_carry_a_step(tmp_path):
    years = tmp_path / "years.csv"
    pd.DataFrame([
        {"case_id": "p3", "generator": "WIND", "state_capacity_mw": 100.0, "state_commissioning_year": 2029.0,
         "year_source": "lccc_round_technology_median"},
        {"case_id": "p3", "generator": "SOLAR", "state_capacity_mw": 100.0, "state_commissioning_year": 2016.0,
         "year_source": "asset_date"},
        # Undated merchant plant, state-owned under P3.
        {"case_id": "p3", "generator": "CHP", "state_capacity_mw": 100.0, "state_commissioning_year": np.nan,
         "year_source": "none"},
    ]).to_csv(years, index=False)
    tie = {"enabled": True, "rule": "lifo_commissioning", "step_gbp_per_mwh": 0.05, "years_file": str(years)}
    block = {"shares_file": "unused-when-years_file-is-set", "case_id": "p3"}
    shares = pd.Series({"WIND": 1.0, "SOLAR": 1.0, "CHP": 1.0})
    with pytest.raises(ValueError, match="commissioning year"):
        resolve_tie_break(tie, block, shares, LOG)
    steps = resolve_tie_break(tie, block, shares, LOG, eligible=pd.Index(["WIND", "SOLAR"]))
    assert "CHP" not in steps.index
    assert steps["WIND"] == 0.0 and steps["SOLAR"] == pytest.approx(0.05 * 13)


def test_round_median_dates_a_technology_with_no_dated_unit(tmp_path):
    """18 Oct 2026: AR1's waste CfDs are live in LCCC's portfolio with no expected start, and waste has no
    technology median, so they were undated and stopped every P3 solve. The round's median over dated units of
    any technology now dates them; a round with no dated unit at all stays undated."""
    register = pd.DataFrame([
        _row("Energy_Works_(Hull)::legacy", "waste_to_energy", "CfD", "AR6"),
        _row("Nowhere::legacy", "waste_to_energy", "CfD", "AR4"),
    ])
    out = asset_years(register, _portfolio(tmp_path)).set_index("asset_id")
    # AR6's dated units start 2027 and 2029: median 2028.
    assert out.loc["Energy_Works_(Hull)::legacy", ["tie_break_year", "year_source"]].tolist() == [2028.0, "lccc_round_median"]
    assert np.isnan(out.loc["Nowhere::legacy", "tie_break_year"])
    assert out.loc["Nowhere::legacy", "year_source"] == "none"
