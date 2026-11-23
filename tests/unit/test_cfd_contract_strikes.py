"""Contract-level CfD strikes and the round-by-round fixes (rebuild batch, 14 Oct 2026).

A generator may carry ``cfd_technology``: the strike-table row of its own
contract (floating, AR6 permitted reduction, AR7 Scottish pot, remote island
wind). Every consumer looks the strike up with it, and the anonymous increment
keeps the (round, carrier) lookup.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.generators.cfd_vintage_labels import load_vintage_schedule
from scripts.market.cfd_strike_table import CFD_TECHNOLOGY_COLUMN, contract_technology, load_strike_table
from scripts.market.revenue_tracking import compute_cfd_payments
from scripts.renewables.support_register_labels import (
    CFD_CONTRACT_PART_COLUMN,
    CFD_ROUND_PART_SUFFIX,
    PARTIAL_CFD_REMAINDER_SUFFIX,
    apply_checked_cfd_contracts,
    apply_lccc_cfd_evidence,
    checked_cfd_exclusions,
    load_checked_cfd_contracts,
    terminated_round_capacity,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "market" / "lccc" / "checked_cfd_contracts.csv"
STRIKES = ROOT / "data" / "market" / "lccc" / "cfd_strike_prices_by_round.csv"
SCHEDULE = ROOT / "data" / "market" / "CP30" / "cfd_vintage_schedule_2030.csv"
UNIT = {"from_price_year": 2012, "to_price_year": 2012, "from_index": 1.0, "to_index": 1.0,
        "factor": 1.0, "index": "none"}


@pytest.fixture(scope="module")
def strikes():
    return load_strike_table(STRIKES, UNIT, 2012)


@pytest.fixture(scope="module")
def table():
    return load_checked_cfd_contracts(TABLE)


def test_contract_technology_selects_its_own_strike_row(strikes):
    assert strikes.lookup("wind_offshore", "AR6", "floating_offshore")[:2] == (139.93, "AR6_floating_offshore")
    assert strikes.lookup("wind_offshore", "AR7", "floating_offshore")[:2] == (155.37, "AR7_floating_offshore")
    assert strikes.lookup("wind_offshore", "AR6", "offshore_permitted_reduction")[:2] == (
        54.23, "AR6_offshore_permitted_reduction")
    assert strikes.lookup("wind_offshore", "AR7", "offshore_scotland")[:2] == (64.23, "AR7_offshore_scotland")
    assert strikes.lookup("wind_onshore", "AR4", "remote_island_wind")[:2] == (46.39, "AR4_remote_island_wind")
    # No technology, or a technology the round lacks: the carrier's row as before.
    assert strikes.lookup("wind_offshore", "AR6")[:2] == (58.87, "AR6_offshore")
    assert strikes.lookup("wind_offshore", "AR6", "")[:2] == (58.87, "AR6_offshore")
    assert strikes.lookup("wind_offshore", "AR3", "floating_offshore")[1] == "AR3_offshore"
    assert strikes.lookup("wind_onshore", "AR3")[1] == "AR3_remote_island_wind"   # the alias still holds


def test_contract_technology_column_reader():
    frame = pd.DataFrame({CFD_TECHNOLOGY_COLUMN: ["floating_offshore", np.nan, None, "nan"]}, index=list("abcd"))
    assert contract_technology(frame).tolist() == ["floating_offshore", "", "", ""]
    assert contract_technology(pd.DataFrame(index=["x"])).tolist() == [""]


def test_cfd_payments_settle_each_contract_at_its_own_strike(strikes):
    snaps = pd.date_range("2030-01-01", periods=2, freq="h")
    generators = pd.DataFrame({
        "support_type": "CfD", "carrier": "wind_offshore", "cfd_round": ["AR6", "AR6", "AR7"],
        "bus": "B", CFD_TECHNOLOGY_COLUMN: ["floating_offshore", "", "floating_offshore"]},
        index=["green_volt", "increment", "erebus"])
    network = SimpleNamespace(
        generators=generators, snapshots=snaps,
        generators_t=SimpleNamespace(p=pd.DataFrame(10.0, index=snaps, columns=generators.index)),
        snapshot_weightings=pd.DataFrame({"generators": 1.0}, index=snaps))
    details = compute_cfd_payments(network, strikes, reference_price=pd.Series(30.0, index=snaps)) \
        .set_index("generator")
    assert details.at["green_volt", "strike_price"] == pytest.approx(139.93)
    assert details.at["increment", "strike_price"] == pytest.approx(58.87)
    assert details.at["erebus", "strike_price"] == pytest.approx(155.37)
    assert details.at["green_volt", "net_cfd_cost"] == pytest.approx((139.93 - 30.0) * 20.0)


def _site(name, carrier, mw, ref, round_="AR6", status="Awaiting Construction", support="CfD"):
    return {"site_name": name, "technology": carrier, "capacity_mw": mw, "repd_ref_id": ref, "fleet_origin": "named",
            "development_status": status, "support_type": support, "cfd_round": round_, "cfd_capacity_mw": np.nan,
            "cfd_id": "", "ro_banding": np.nan, "bus": "B"}


def test_checked_parts_carry_their_contract_technology(table):
    sites = pd.DataFrame([
        _site("Moray West Offshore Wind Farm Project", "wind_offshore", 882.0, "6158"),
        _site("Green Volt - Offshore Wind Farm", "wind_offshore", 560.0, "11851"),
        _site("Pentland Floating Offshore Wind Farm", "wind_offshore", 100.0, "11989", "AR7"),
        _site("Viking Wind Farm", "wind_onshore", 443.0, "6828", "AR5", "Operational"),
        _site("Hornsea 3", "wind_offshore", 2955.0, "2472", "AR4"),
    ])
    out, _ = apply_checked_cfd_contracts(sites, table)
    tech = out.set_index("site_name")[CFD_TECHNOLOGY_COLUMN]
    rnd = out.set_index("site_name")["cfd_round"]
    moray = "Moray West Offshore Wind Farm Project"
    assert rnd[moray] == "AR4" and pd.isna(tech[moray])
    assert tech[moray + CFD_ROUND_PART_SUFFIX + "AR6"] == "offshore_permitted_reduction"
    assert pd.isna(tech[moray + PARTIAL_CFD_REMAINDER_SUFFIX])
    assert tech["Green Volt - Offshore Wind Farm"] == "floating_offshore"
    assert tech["Pentland Floating Offshore Wind Farm"] == "floating_offshore"
    # Pentland: 92.5 of 100 MW contracted; the 7.5% remainder is absorbed, so all 100 MW settle on AR7 floating.
    assert out.set_index("site_name").at["Pentland Floating Offshore Wind Farm", "capacity_mw"] == pytest.approx(100.0)
    # Viking: AR5 223.6 MW holds the site's name, AR4 220 MW is the part; both remote island wind.
    assert rnd["Viking Wind Farm"] == "AR5" and tech["Viking Wind Farm"] == "remote_island_wind"
    assert tech["Viking Wind Farm" + CFD_ROUND_PART_SUFFIX + "AR4"] == "remote_island_wind"
    assert tech[ "Hornsea 3" + CFD_ROUND_PART_SUFFIX + "AR6"] == "offshore_permitted_reduction" and pd.isna(tech["Hornsea 3"])


def test_label_clean_ups(table):
    sites = pd.DataFrame([
        _site("Netley Landfill Solar", "solar_pv", 12.0, "2044", "AR1", "Operational"),
        _site("High Constellation Wind Farm", "wind_onshore", 48.0, "7345", "AR4"),
        _site("Mopes Lane, Purton - Solar Farm", "solar_pv", 20.0, "9849", "AR5"),
        _site("Thaxted - Solar farm", "solar_pv", 40.0, "8475"),
        _site("Crouch Solar Farm", "solar_pv", 49.9, "8357", "AR4", "Operational"),
    ])
    out, audit = apply_checked_cfd_contracts(sites, table)
    by = out.set_index("site_name")
    # Netley operates: it stays, without a CfD.
    assert by.at["Netley Landfill Solar", "support_type"] == "merchant"
    assert by.at["Netley Landfill Solar", CFD_CONTRACT_PART_COLUMN] == "no_contract"
    assert "2044" not in set(checked_cfd_exclusions(table)["repd_ref_id"])
    # High Constellation settles on its live AR7a contract, not the terminated AR4.
    assert by.at["High Constellation Wind Farm", "cfd_round"] == "AR7"
    assert by.at["High Constellation Wind Farm", "cfd_id"] == "AR7-HIC-610"
    # Mopes Lane is AR5-PSF-700 (15 MW of 20, remainder under the solar threshold:
    # the whole site settles on AR5); AR7-MOL-700 is the Bentham House Farm extension.
    assert by.at["Mopes Lane, Purton - Solar Farm", "capacity_mw"] == pytest.approx(20.0)
    assert by.at["Mopes Lane, Purton - Solar Farm", "cfd_round"] == "AR5"
    assert by.at["Mopes Lane, Purton - Solar Farm", "cfd_id"] == "AR5-PSF-700"
    assert "Mopes Lane, Purton - Solar Farm" + PARTIAL_CFD_REMAINDER_SUFFIX not in by.index
    # Solar sites wholly CfD beyond their contracts are split (LCCC-matched and REPD-only).
    assert by.at["Thaxted - Solar farm", "capacity_mw"] == pytest.approx(19.76)
    assert by.at["Crouch Solar Farm", "capacity_mw"] == pytest.approx(15.0)
    assert by.at["Crouch Solar Farm" + PARTIAL_CFD_REMAINDER_SUFFIX, "support_type"] == "merchant"
    # The terminated AR4 award of High Constellation is netted off the onshore AR4 schedule.
    held = terminated_round_capacity(table)
    onshore_ar4 = held[held["carrier"].eq("wind_onshore") & held["cfd_round"].eq("AR4")]["capacity_mw"].sum()
    assert onshore_ar4 == pytest.approx(81.6 + 71.5 + 50.0)


def test_lccc_evidence_records_floating_and_remote_island_technology():
    portfolio = pd.DataFrame([
        {"unit_id": "AR7-ERE-910", "unit_name": "Erebus", "carrier": "wind_offshore", "unit_capacity_mw": 100.0,
         "cfd_round": "AR7", "Status": "Pre-MDD", "Technology_Type": "Floating Offshore Wind", "postcode": "",
         "award_year": 2025},
    ])
    sites = pd.DataFrame([_site("Erebus", "wind_offshore", 100.0, "7861", None)])
    sites["cfd_round"] = pd.NA
    out, _, _ = apply_lccc_cfd_evidence(sites, portfolio)
    assert out.at[0, "cfd_round"] == "AR7" and out.at[0, CFD_TECHNOLOGY_COLUMN] == "floating_offshore"


def test_vintage_schedule_carries_the_contracts_not_the_awards():
    schedule = load_vintage_schedule(SCHEDULE)
    # The main (fixed) rows; the floating awards are rows of their own (27 Oct 2026).
    schedule = schedule.loc[schedule["cfd_technology"].eq("")].set_index(["carrier", "cfd_round"])["capacity_mw"]
    assert schedule[("wind_offshore", "AR4")] == pytest.approx(5627.47)    # 6,994.34 less 1,366.87 relinquished
    assert schedule[("wind_offshore", "AR2")] == pytest.approx(3143.34)    # the LCCC contracts, not the award
    assert schedule[("marine", "AR6")] == pytest.approx(22.0)              # net of 6 MW terminated tidal


def test_post_2030_contracts_settle_on_their_own_round(table):
    inch = table[table["repd_ref_id"].eq("2527")].set_index("cfd_round")
    assert inch.at["AR4", "contracted_mw"] == pytest.approx(810.0)
    assert "settles on its own round and strike" in inch.at["AR4", "note"]
    assert table.loc[table["repd_ref_id"].eq("11989"), "cfd_technology"].iloc[0] == "floating_offshore"
