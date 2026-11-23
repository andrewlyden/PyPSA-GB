"""P1/P2 under central dispatch (20 Oct 2026): the I3 constraint line and what the projector does with it."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill import extract_central_dispatch_costs as ecd
from scripts.consumer_bill import project_future_bill as pfb

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def settlement(**overrides) -> pd.DataFrame:
    """A year FTI settlement table (metric rows, the months and YEAR) with round numbers, GBP m."""
    monthly = {"central_cost_recovery_gbp_m": 1.0, "central_firm_access_gbp_m": 10.0,
               "central_firm_access_margin_gbp_m": 6.0, "central_firm_access_cfd_support_gbp_m": 3.0,
               "central_firm_access_roc_support_gbp_m": 1.0, "central_storage_make_whole_gbp_m": 0.5,
               "central_ic_make_whole_gbp_m": 2.0, "forecast_error_charge_gbp_m": 4.0,
               "central_counter_trade_premium_gbp_m": 0.25, "central_realtime_correction_gbp_m": -0.5,
               "central_realtime_correction_replaced_gbp_m": -0.5, "central_firm_access_cfd_lost_twh": 0.1,
               "market_constraint_cost_gbp_m": 30.0, "gb_demand_twh": 25.0, "hours": 730.0,
               "market_ic_rent_gbp_m::Nemo": 3.0, "central_ic_rent_gbp_m::Nemo": 3.5,
               "market_ic_rent_gbp_m::Moyle": 1.0, "central_ic_rent_gbp_m::Moyle": 0.75}
    monthly.update(overrides)
    table = pd.DataFrame({m: dict(monthly) for m in MONTHS})
    year = {k: v * 12 for k, v in monthly.items()}
    year["hours"] = 8760.0
    table["YEAR"] = pd.Series(year)
    table = table.astype(object)
    for row, value in (("national_price_basis", "copperplate"), ("firm_access_support_basis", "lost_support")):
        table.loc[row, MONTHS] = value
        table.loc[row, "YEAR"] = ""
    table.loc["months_settled", :] = ""
    table.loc["months_settled", "YEAR"] = 12
    table.index.name = "metric"
    return table


MARKET = {"annual": {"model_price_year": 2024, "constraint_cost_gbp": 360e6, "interconnector_retrading_cost_gbp": 90e6,
                     "include_interconnector_retrading": True, "balancing_negative_price_rule": "none",
                     "unlabelled_support": [{"carrier": "EU_market"}]}}


def test_the_i3_line_is_the_fti_parts_with_retrading_at_zero():
    monthly, by_carrier, annual = ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024)
    parts = {k: annual[k] for k in pfb.CENTRAL_DISPATCH_CONSTRAINT_PARTS}
    assert parts == pytest.approx({"cost_recovery_gbp": 12e6, "firm_access_gbp": 120e6, "storage_make_whole_gbp": 6e6,
                                   "interconnector_make_whole_gbp": 24e6, "forecast_error_charge_gbp": 48e6,
                                   "counter_trade_premium_gbp": 3e6})
    assert annual["constraint_cost_gbp"] == pytest.approx(213e6)
    assert annual["interconnector_retrading_cost_gbp"] == 0.0 and annual["include_interconnector_retrading"] is False
    # The lost CfD top-up is paid as firm access, inside the line; the I1 line is beside it.
    assert annual["cfd_curtailment_compensation_gbp"] == pytest.approx(36e6)
    assert annual["cfd_curtailed_mwh"] == pytest.approx(1.2e6)
    assert annual["market_constraint_cost_gbp"] == 360e6
    assert annual["institution_step_on_constraint_line_gbp"] == pytest.approx(213e6 - 360e6)
    assert annual["balancing_negative_price_rule"] == "none"
    # The real-time correction is replaced by the charge, so it is reported, never added.
    assert annual["realtime_correction_replaced_gbp"] == pytest.approx(-6e6)
    assert annual["interconnector_rent_change_by_cable_gbp"] == pytest.approx({"Moyle": -3e6, "Nemo": 6e6})
    assert len(monthly) == 12 and monthly["constraint_cost_gbp"].sum() == pytest.approx(213e6)
    _, _, without = ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024, counter_trade_premium=False)
    assert without["constraint_cost_gbp"] == pytest.approx(210e6) and without["counter_trade_premium_gbp"] == 0.0


def test_the_fti_reading_of_the_counter_trade_premium_is_reported_beside_the_headline(tmp_path):
    """27 Oct 2026: the premium stays on the I3 line; FTI count it inside the replaced real-time correction."""
    _, _, annual = ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024)
    fti = annual["counter_trade_premium_fti_treatment"]
    assert annual["constraint_cost_gbp"] == pytest.approx(213e6)          # the headline is unchanged
    assert fti["counter_trade_premium_gbp"] == pytest.approx(3e6)
    assert fti["constraint_cost_gbp"] == pytest.approx(210e6)
    assert fti["headline_consumer_saving_on_constraint_line_gbp"] == pytest.approx(360e6 - 213e6)
    assert fti["consumer_saving_on_constraint_line_gbp"] == pytest.approx(360e6 - 210e6)
    assert fti["institution_step_on_constraint_line_gbp"] == pytest.approx(210e6 - 360e6)
    _, _, without = ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024, counter_trade_premium=False)
    assert without["counter_trade_premium_fti_treatment"]["constraint_cost_gbp"] == pytest.approx(210e6)
    # The projector carries it to the bill's constraint-line split, in household pounds too.
    out = tmp_path / "x"
    table, market = tmp_path / "s.csv", tmp_path / "m.json"
    settlement().to_csv(table)
    market.write_text(json.dumps(MARKET), encoding="utf-8")
    ecd.write_central_dispatch_costs(table, market, f"{out}_monthly.csv", f"{out}_carriers.csv", f"{out}_summary.json",
                                     model_price_year=2024)
    read = pfb.read_balancing_costs(f"{out}_summary.json", MONTHS, 2024)
    split = pfb.constraint_line_split(read, {}, household_per_system=1e-8)
    assert split["interconnector_retrading_gbp"] == 0.0
    carried = split["counter_trade_premium_fti_treatment"]
    assert carried["constraint_cost_gbp"] == pytest.approx(210e6)
    assert carried["consumer_saving_on_constraint_line_gbp_per_3100kwh_ex_vat"] == pytest.approx(150e6 * 1e-8)


def test_the_extraction_refuses_what_the_i3_bill_cannot_be_built_on():
    table = settlement()
    table.loc["months_settled", "YEAR"] = 11
    with pytest.raises(ValueError, match="twelve"):
        ecd.central_dispatch_costs(table, MARKET, model_price_year=2024)
    table = settlement()
    table.loc["firm_access_support_basis", "Mar"] = "metered_only"
    with pytest.raises(ValueError, match="lost_support"):
        ecd.central_dispatch_costs(table, MARKET, model_price_year=2024)
    table = settlement()
    table.loc["national_price_basis", MONTHS] = "nodal"
    with pytest.raises(ValueError, match="copperplate"):
        ecd.central_dispatch_costs(table, MARKET, model_price_year=2024)
    with pytest.raises(ValueError, match="different solve runs"):
        ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024,
                                   vintage={"mixed_vintage": True, "gap_between_runs_hours": 80.0})
    ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024,
                               vintage={"mixed_vintage": True, "gap_between_runs_hours": 80.0}, allow_mixed_vintage=True)
    with pytest.raises(ValueError, match="2023"):
        ecd.central_dispatch_costs(settlement(), {"annual": {**MARKET["annual"], "model_price_year": 2023}},
                                   model_price_year=2024)
    with pytest.raises(KeyError, match="central_ic_make_whole"):
        ecd.central_dispatch_costs(settlement().drop(index="central_ic_make_whole_gbp_m"), MARKET, model_price_year=2024)


def test_per_unit_transfers_must_add_up_to_the_settlement():
    units = pd.DataFrame({"component_type": ["Generator", "Generator", "StorageUnit"], "asset_id": ["g", "w", "b"],
                          "carrier": ["CCGT", "wind_offshore", "battery"],
                          "cost_recovery_gbp": [12e6, 0.0, 0.0], "firm_access_margin_gbp": [0.0, 72e6, 0.0],
                          "firm_access_cfd_support_gbp": [0.0, 36e6, 0.0], "firm_access_roc_support_gbp": [0.0, 12e6, 0.0],
                          "storage_make_whole_gbp": [0.0, 0.0, 6e6]})
    _, by_carrier, _ = ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024, units=units)
    assert list(by_carrier["carrier"]) == ["wind_offshore", "CCGT", "battery"]
    units.loc[0, "cost_recovery_gbp"] = 1e6
    with pytest.raises(ValueError, match="cost_recovery_gbp"):
        ecd.central_dispatch_costs(settlement(), MARKET, model_price_year=2024, units=units)


def test_the_projector_reads_the_i3_summary_and_moves_the_cable_rents(tmp_path):
    out = tmp_path / "x"
    table, market = tmp_path / "s.csv", tmp_path / "m.json"
    settlement().to_csv(table)
    market.write_text(json.dumps(MARKET), encoding="utf-8")
    annual = ecd.write_central_dispatch_costs(table, market, f"{out}_monthly.csv", f"{out}_carriers.csv",
                                              f"{out}_summary.json", model_price_year=2024)
    read = pfb.read_balancing_costs(f"{out}_summary.json", MONTHS, 2024)
    assert read["institution"] == "central_dispatch"
    assert read["constraint_cost_gbp"] == pytest.approx(annual["constraint_cost_gbp"])
    assert read["accepted_offer_cost_gbp"] == 0.0 and read["include_interconnector_retrading"] is False
    rents = pfb.rent_by_cable_for_institution({"Nemo": 50e6, "Moyle": 10e6}, read)
    assert rents == pytest.approx({"Nemo": 56e6, "Moyle": 7e6})
    # A market case's rent is untouched.
    assert pfb.rent_by_cable_for_institution({"Nemo": 50e6}, {"constraint_cost_gbp": 1.0}) == {"Nemo": 50e6}
    # The parts must add up to the line.
    document = json.loads((tmp_path / "x_summary.json").read_text(encoding="utf-8"))
    document["annual"]["constraint_cost_gbp"] += 10.0
    (tmp_path / "bad.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="sum of its parts"):
        pfb.read_balancing_costs(tmp_path / "bad.json", MONTHS, 2024)
    assert "Central dispatch on FTI's method" in pfb._central_dispatch_constraint_text(read)
    assert "central dispatch" in pfb._bsuos_limitation(read)


def test_a_rent_change_for_an_unknown_cable_is_refused():
    with pytest.raises(ValueError, match="names cables"):
        pfb.rent_by_cable_for_institution({"Nemo": 1.0}, {"institution": "central_dispatch",
                                                          "interconnector_rent_change_by_cable_gbp": {"Viking": 1.0}})
