"""F19 and its follow-up (rebuild batch, 13 Oct 2026): named sites' CfD contracts checked against LCCC.

A part-contracted site splits into its CfD rounds and a merchant remainder; a
multi-round site splits so each round settles at its own strike; a site whose
contracts are terminated, or that holds none, is merchant. Capacity is
conserved, every part keeps the site's profile name, and REPD-only partial
figures are reported, not split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.generators.cfd_vintage_labels import (
    apply_cfd_vintage_labels,
    named_round_capacity,
)
from scripts.market.cfd_strike_table import load_strike_table
from scripts.public_ownership.calculate_financing_scenarios import offshore_local_tariffs
from scripts.renewables.support_register_labels import (
    CFD_CONTRACT_PART_COLUMN,
    CFD_ROUND_PART_SUFFIX,
    PARTIAL_CFD_REMAINDER_SUFFIX,
    apply_checked_cfd_contracts,
    load_checked_cfd_contracts,
    load_crosswalk_overrides,
    partial_cfd_base_name,
    terminated_round_capacity,
)
from config.config_loader import load_layer

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "market" / "lccc" / "checked_cfd_contracts.csv"
PORTFOLIO = ROOT / "data" / "market" / "lccc" / "cfd_contract_portfolio_status.csv"


def _site(name, carrier, mw, ref, cfd_mw=np.nan, round_="AR6", status="Awaiting Construction", bus="B", support="CfD",
          origin="named"):
    return {"site_name": name, "technology": carrier, "capacity_mw": mw, "repd_ref_id": ref, "fleet_origin": origin,
            "development_status": status, "support_type": support, "cfd_round": round_, "cfd_capacity_mw": cfd_mw,
            "cfd_id": "", "ro_banding": np.nan, "bus": bus}


def _sites():
    return pd.DataFrame([
        _site("Seagreen", "wind_offshore", 1075.0, "2540", 454.0, "AR3", "Operational", "TEAL2K"),
        _site("Moray West Offshore Wind Farm Project", "wind_offshore", 882.0, "6158", 73.5, "AR6"),
        _site("Inch Cape", "wind_offshore", 1080.0, "2527", 266.11, "AR6"),
        _site("East Anglia 3 (EA 3)", "wind_offshore", 1400.0, "2470.0", 158.9, "AR6"),
        _site("Sofia (Teesside B)", "wind_offshore", 1400.0, "6556", 14.0, "AR3"),
        _site("Hornsea 4", "wind_offshore", 2600.0, "2473", 2400.0, "AR6"),
        _site("Bhlaraidh Wind Farm Extension", "wind_onshore", 84.0, "7473", 5.6, "AR5"),
        _site("Forthwind", "wind_offshore", 20.0, "14484", 12.0, "AR8_modelled"),
        _site("Dudgeon Extension Project", "wind_offshore", 402.0, "7692", np.nan, "AR8_modelled"),
        _site("Unchecked Solar", "solar_pv", 50.0, "999", 10.0, "AR6"),
        _site("Pentlow Hill", "solar_pv", 24.0, "7666", 15.0, "AR4", "Operational"),
        # A proxy row carrying a checked reference is never touched.
        _site("wind_offshore_proxy_1", "wind_offshore", 100.0, "2540", np.nan, "AR3", origin="increment"),
    ])


@pytest.fixture(scope="module")
def table():
    return load_checked_cfd_contracts(TABLE)


def _parts(out, base):
    rows = out[out["site_name"].map(partial_cfd_base_name).eq(base)]
    return rows.set_index("site_name")


def test_capacity_is_conserved_and_proxies_untouched(table):
    sites = _sites()
    out, _ = apply_checked_cfd_contracts(sites, table)
    assert out["capacity_mw"].sum() == pytest.approx(sites["capacity_mw"].sum())
    proxy = out[out["site_name"].eq("wind_offshore_proxy_1")]
    assert len(proxy) == 1 and proxy.iloc[0]["capacity_mw"] == pytest.approx(100.0)


def test_part_contracted_site_splits_into_cfd_and_merchant(table):
    out, audit = apply_checked_cfd_contracts(_sites(), table)
    parts = _parts(out, "Seagreen")
    assert parts.at["Seagreen", "capacity_mw"] == pytest.approx(454.0)
    assert parts.at["Seagreen", "support_type"] == "CfD" and parts.at["Seagreen", "cfd_round"] == "AR3"
    assert parts.at["Seagreen", "cfd_id"] == "AR3-SEW-101;AR3-SEW-102;AR3-SEW-103"
    rest = parts.loc["Seagreen" + PARTIAL_CFD_REMAINDER_SUFFIX]
    assert rest["capacity_mw"] == pytest.approx(621.0) and rest["support_type"] == "merchant"
    assert pd.isna(rest["cfd_round"]) and pd.isna(rest["cfd_id"]) and pd.isna(rest["cfd_capacity_mw"])
    assert rest["bus"] == "TEAL2K" and rest["source_site_name"] == "Seagreen"
    assert rest[CFD_CONTRACT_PART_COLUMN] == "remainder"
    # Solar 15 of 24 MW: the 37.5% remainder clears the 35% solar threshold.
    pentlow = _parts(out, "Pentlow Hill")["capacity_mw"]
    assert pentlow["Pentlow Hill"] == pytest.approx(15.0)
    assert pentlow["Pentlow Hill" + PARTIAL_CFD_REMAINDER_SUFFIX] == pytest.approx(9.0)


def test_multi_round_sites_settle_each_round_at_its_own_strike(table):
    out, _ = apply_checked_cfd_contracts(_sites(), table)
    moray = _parts(out, "Moray West Offshore Wind Farm Project")
    main = "Moray West Offshore Wind Farm Project"
    assert moray.at[main, "cfd_round"] == "AR4" and moray.at[main, "capacity_mw"] == pytest.approx(220.5)
    assert moray.at[main, "cfd_id"] == "AR4-MRY-010"
    ar6 = moray.loc[main + CFD_ROUND_PART_SUFFIX + "AR6"]
    assert ar6["cfd_round"] == "AR6" and ar6["capacity_mw"] == pytest.approx(73.5)
    assert ar6["support_type"] == "CfD" and ar6["source_site_name"] == main
    assert ar6[CFD_CONTRACT_PART_COLUMN] == "round"
    assert moray.loc[main + PARTIAL_CFD_REMAINDER_SUFFIX, "capacity_mw"] == pytest.approx(588.0)
    # Inch Cape: AR4 810 + AR6 266.11 of 1,080 MW; the 3.89 MW remainder is absorbed.
    inch = _parts(out, "Inch Cape")
    assert inch.at["Inch Cape", "cfd_round"] == "AR4"
    assert inch.at["Inch Cape", "capacity_mw"] == pytest.approx(1080.0 * 810.0 / 1076.11)
    assert inch.loc["Inch Cape" + CFD_ROUND_PART_SUFFIX + "AR6", "capacity_mw"] == pytest.approx(1080.0 * 266.11 / 1076.11)
    assert not (inch["support_type"] == "merchant").any()
    # EA3: AR4 1,054.53 + AR6 158.9 of 1,400 MW; the 13% remainder is absorbed.
    ea3 = _parts(out, "East Anglia 3 (EA 3)")
    assert ea3.at["East Anglia 3 (EA 3)", "cfd_round"] == "AR4"
    assert ea3.at["East Anglia 3 (EA 3)", "cfd_id"] == "AR4-EAS-011;AR4-EAS-012;AR4-EAS-013"
    assert ea3["capacity_mw"].sum() == pytest.approx(1400.0)
    assert ea3.loc["East Anglia 3 (EA 3)" + CFD_ROUND_PART_SUFFIX + "AR6", "capacity_mw"] == pytest.approx(
        1400.0 * 158.9 / 1213.43)
    # Sofia: REPD says 14 MW, LCCC holds all 1,400 MW in AR3.
    sofia = _parts(out, "Sofia (Teesside B)")
    assert len(sofia) == 1 and sofia.iloc[0]["support_type"] == "CfD" and sofia.iloc[0]["cfd_round"] == "AR3"


def test_terminated_and_uncontracted_sites_are_merchant(table):
    out, audit = apply_checked_cfd_contracts(_sites(), table)
    for name in ("Hornsea 4", "Bhlaraidh Wind Farm Extension", "Forthwind", "Dudgeon Extension Project"):
        row = _parts(out, name)
        assert len(row) == 1, name
        row = row.iloc[0]
        assert row["support_type"] == "merchant" and pd.isna(row["cfd_round"]), name
        assert row[CFD_CONTRACT_PART_COLUMN] == "no_contract"
    notes = audit.set_index("site_name")["note"]
    assert notes["Hornsea 4"].startswith("no live CfD (LCCC Terminated)")
    assert notes["Dudgeon Extension Project"].startswith("no live CfD (LCCC none)")
    assert notes["Unchecked Solar"].startswith("not split")


def test_split_rounds_resolve_to_their_own_strikes(table):
    out, _ = apply_checked_cfd_contracts(_sites(), table)
    indexation = {"from_price_year": 2012, "to_price_year": 2012, "from_index": 1.0, "to_index": 1.0,
                  "factor": 1.0, "index": "none"}
    strikes = load_strike_table(ROOT / "data/market/lccc/cfd_strike_prices_by_round.csv", indexation, 2012)
    moray = _parts(out, "Moray West Offshore Wind Farm Project")
    keys = {r: strikes.lookup("wind_offshore", r)[1:] for r in moray["cfd_round"].dropna()}
    assert keys == {"AR4": ("AR4_offshore", False), "AR6": ("AR6_offshore", False)}


def test_terminated_awards_stay_netted_from_the_vintage_schedule(table, tmp_path):
    held = terminated_round_capacity(table)
    offshore = held[held["carrier"].eq("wind_offshore")].set_index("cfd_round")["capacity_mw"]
    assert offshore["AR4"] == pytest.approx(1396.0) and offshore["AR6"] == pytest.approx(2400.0)
    # AR4 Remote Island Wind is not in the onshore schedule, so it is not netted.
    onshore_ar4 = held[held["carrier"].eq("wind_onshore") & held["cfd_round"].eq("AR4")]["capacity_mw"].sum()
    assert onshore_ar4 == pytest.approx(81.6 + 71.5 + 50.0)   # Chirmorie, Douglas West Ext, High Constellation

    schedule = tmp_path / "schedule.csv"
    pd.DataFrame([{"carrier": "wind_offshore", "cfd_round": "AR4", "capacity_mw": 2000.0, "delivery_years": "",
                   "source": "s", "notes": ""}]).to_csv(schedule, index=False)
    increment = pd.DataFrame([{"site_name": "proxy", "technology": "wind_offshore", "capacity_mw": 3000.0,
                               "support_type": "CfD", "cfd_round": ""}])
    named = pd.DataFrame([{"site_name": "Boreas", "technology": "wind_offshore", "capacity_mw": 1400.0,
                           "support_type": "merchant", "cfd_round": pd.NA}])
    plain = apply_cfd_vintage_labels(increment, schedule, named_capacity=named_round_capacity(named))
    netted = apply_cfd_vintage_labels(increment, schedule,
                                      named_capacity=pd.concat([named_round_capacity(named), held]))
    ar4 = lambda frame: frame.loc[frame["cfd_round"].eq("AR4"), "capacity_mw"].sum()
    assert ar4(plain) == pytest.approx(2000.0)            # the increment would refill Boreas's award
    assert ar4(netted) == pytest.approx(2000.0 - 1396.0)  # the terminated award stays off the increment


def test_part_names_share_the_farm_lookups():
    for suffix in (PARTIAL_CFD_REMAINDER_SUFFIX, CFD_ROUND_PART_SUFFIX + "AR6"):
        assert partial_cfd_base_name("Seagreen" + suffix) == "Seagreen"
    assert partial_cfd_base_name("Seagreen") == "Seagreen"
    assert partial_cfd_base_name("site_CfD") == "site_CfD"           # the FES fraction split is not a part
    table = pd.read_csv(ROOT / "data/public_ownership/financing/neso_tnuos_offshore_local_tariffs_2026_27.csv")
    names = pd.Series(["Seagreen", "Seagreen" + PARTIAL_CFD_REMAINDER_SUFFIX, "Seagreen" + CFD_ROUND_PART_SUFFIX + "AR4"])
    values, basis = offshore_local_tariffs(names, table)
    assert values.nunique() == 1 and set(basis) == {"neso_table"}


def test_table_matches_the_lccc_portfolio(table):
    portfolio = pd.read_csv(PORTFOLIO).set_index("CFD_ID")
    rounds = portfolio["Allocation_Round"].str.replace("Allocation Round ", "AR", regex=False)
    for row in table.itertuples():
        if row.lccc_status == "none":
            assert row.contracted_mw == 0 and row.cfd_ids == ""
            continue
        if row.lccc_status == "repd_only":
            # REPD records the contract but no LCCC unit can be tied to the site by name.
            assert row.cfd_ids == "" and row.contracted_mw > 0 and row.source.endswith("repd-q2-jul-2025.csv")
            continue
        ids = row.cfd_ids.split(";")
        assert set(ids) <= set(portfolio.index), row.site_name
        assert set(rounds[ids]) == {row.cfd_round}, row.site_name
        statuses = set(portfolio.loc[ids, "Status"])
        assert (statuses == {"Terminated"}) == (row.lccc_status == "Terminated"), row.site_name
        assert row.contracted_mw == pytest.approx(portfolio.loc[ids, "Maximum_Contract_Capacity_MW"].sum(), abs=1e-3)
    # Every unit is listed once.
    listed = [i for ids in table["cfd_ids"] for i in ids.split(";") if i]
    assert len(listed) == len(set(listed))


def test_crosswalk_pairs_dudgeon_and_east_anglia_to_their_own_farms():
    overrides = load_crosswalk_overrides(ROOT / "data/market/support/support_register_crosswalk_overrides.csv")
    pairs = overrides[overrides["register"].eq("lccc_cfd")].set_index("unit_id")["repd_ref_id"]
    assert {pairs[u] for u in ("INV-DUD-001", "INV-DUD-002", "INV-DUD-003")} == {"2538"}
    assert {pairs[u] for u in ("AR4-EAS-011", "AR4-EAS-012", "AR4-EAS-013", "AR6-EAT-010")} == {"2470"}
    assert {pairs[u] for u in ("AR6-EAA-011", "AR6-EAA-012", "AR6-EAA-013")} == {"5867"}


def test_table_validation(tmp_path):
    path = tmp_path / "t.csv"
    base = {"repd_ref_id": "1", "site_name": "a", "cfd_round": "AR4", "contracted_mw": 1, "cfd_ids": "x",
            "lccc_status": "Pre-MDD", "source": "s"}
    pd.DataFrame([base, base]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row per REPD Ref ID and round"):
        load_checked_cfd_contracts(path)
    # A terminated round beside a live one is allowed (kept for the netting) ...
    pd.DataFrame([base, {**base, "cfd_round": "AR6", "lccc_status": "Terminated"}]).to_csv(path, index=False)
    assert len(load_checked_cfd_contracts(path)) == 2
    # ... but such a site cannot be declared not built.
    pd.DataFrame([{**base, "action": "not_built_refill", "reason": "r"},
                  {**base, "cfd_round": "AR6", "lccc_status": "Terminated", "action": "not_built_refill",
                   "reason": "r"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="live contract"):
        load_checked_cfd_contracts(path)


def test_the_defaults_name_the_table():
    from config.config_loader import load_defaults

    registers = load_defaults()["support"]["labels"]["support_registers"]
    assert (ROOT / registers["checked_cfd_contracts"]).exists()
