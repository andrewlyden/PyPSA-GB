"""The 2030/31 T-4 cleared on the CP30 2030 fleet (user decision, 6 Oct 2026)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.market import build_capacity_market as bcm
from scripts.market import capacity_market as cm
from scripts.market import capacity_market_2050 as cm50

ASSUMPTIONS = Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")


def test_added_since_keeps_additions_and_retires_the_oldest_first():
    path = pd.Series({2023: 10.0, 2024: 10.0, 2027: 14.0, 2028: 12.0, 2030: 15.0})
    # 2027 adds 4; the 2028 fall retires 2023 plant, not the new; 2030 adds 3.
    assert bcm.added_since(path, 2027) == pytest.approx(7.0)
    falling = pd.Series({2026: 5.0, 2027: 7.0, 2030: 1.0})
    # The fall takes all the 2026 plant, then 1 of the 2 built in 2027.
    assert bcm.added_since(falling, 2027) == pytest.approx(1.0)


def test_new_build_goes_to_new_rows_then_supported_then_eligible():
    units = pd.DataFrame({
        "group": ["nuclear", "nuclear", "battery", "battery", "CCS Gas"],
        "finance_vintage": ["legacy", "legacy", "new_build", "legacy", "new_build"],
        "eligible": [False, True, True, True, False],
        "capacity_mw": [2317.0, 1198.0, 100.0, 50.0, 742.0]})
    share = bcm._new_shares(units, {"nuclear": 1670.0, "battery": 120.0})
    # HPC (CfD) takes nuclear's 2030 build, not Sizewell B.
    assert share.tolist()[:2] == pytest.approx([1670.0 / 2317.0, 0.0])
    # Batteries: the new-build rows first, the rest on the older eligible row.
    assert share.tolist()[2:4] == pytest.approx([1.0, 20.0 / 50.0])
    # No ES1 path: the register's vintage stands.
    assert share.iloc[4] == 1.0


def _book(rows):
    return pd.DataFrame(rows, columns=["technology", "connection_mw", "category", "auction"])


def test_contract_book_nets_by_vintage_with_spill_and_reports_the_excess():
    fleet = pd.DataFrame({
        "asset_id": ["b", "o", "w"], "group": ["battery", "oil", None], "eligible": [True, True, False],
        "capacity_mw": [100.0, 20.0, 500.0], "new_share": [0.3, 0.0, 0.0]})
    book = _book([("Storage (Duration 2h)", 50.0, "new_build", "2028-29 (T-4) Four Year Ahead Capacity Auction"),
                  ("Storage (Duration 2h)", 10.0, "new_build", "2020-21 (T-4) Four Year Ahead Capacity Auction"),
                  ("Oil Gen & Recip", 50.0, "existing", "2029-30 (T-4) Four Year Ahead Capacity Auction"),
                  ("Onshore Wind", 40.0, "new_build", "2027-28 (T-4) Four Year Ahead Capacity Auction")])
    out, matching = bcm.net_contract_book(fleet, book)
    out = out.set_index("asset_id")
    # 30 MW of new batteries take 30 of the 50 MW of new agreements; 20 spill to the older 70 MW.
    assert out.loc["b", "free_new_mw"] == pytest.approx(0.0)
    assert out.loc["b", "free_existing_mw"] == pytest.approx(70.0 - 10.0 - 20.0)
    assert matching["battery"]["held_mw"] == pytest.approx(60.0)
    # Oil has 20 MW for 50 MW of agreements: 30 MW reported beyond the fleet; wind is outside.
    assert out.loc["o", "free_existing_mw"] == 0.0 and matching["oil"]["book_beyond_fleet_mw"] == pytest.approx(30.0)
    assert matching["merchant_renewable"]["book_beyond_fleet_mw"] == pytest.approx(40.0)
    assert out.loc["w", "free_existing_mw"] == 0.0


def _netted():
    return pd.DataFrame({
        "asset_id": ["ccgt", "laes", "bat", "cfd", "ifa"],
        "rate_key": ["CCGT", "long_duration_storage", "battery", "wind_offshore", "interconnector"],
        "group": ["gas", "long_duration_storage", "battery", None, "interconnector"],
        "derating_class": ["Combined Cycle Gas Turbine (CCGT)", "storage", "storage", None, "IFA (France)"],
        "duration_hours": [np.nan, 15.0, 1.784203, np.nan, np.nan],
        "support_type": ["merchant", "merchant", "merchant", "CfD", "cap_and_floor"],
        "eligible": [True, True, True, False, True],
        "free_existing_mw": [1000.0, 0.0, 0.0, 0.0, 1000.0],
        "free_new_mw": [0.0, 100.0, 50.0, 0.0, 0.0]})


LEVELS = {"central": {"derated_mw": 300.0, "source": "x"}, "low": {"derated_mw": 200.0, "source": "x"},
          "high": {"derated_mw": 400.0, "source": "x"}}


def test_supply_rows_classify_eligible_units_and_move_with_the_cases():
    derating = cm50.read_derating()
    rows = bcm.supply_rows(_netted(), {"gas": 0.5}, derating, LEVELS, bcm.Supply2030())
    by = rows.groupby("category")["supply_mw"].sum()
    assert by["existing"] == pytest.approx(500.0 * 0.9048) and by["refurbishing"] == pytest.approx(500.0 * 0.9048)
    assert by["interconnector"] == pytest.approx(1000.0 * 0.55)                      # IFA's own factor
    assert by[bcm.LDES_CATEGORY] == pytest.approx(100.0 * 0.9075)                    # 15 h: longest class
    # New LDES is paid under its own rate key (user decision, 6 Oct 2026).
    assert set(rows.loc[rows["category"].eq(bcm.LDES_CATEGORY), "carrier"]) == {cm.LDES_CAP_AND_FLOOR_RATE_KEY}
    battery = cm50.storage_factor(1.784203, derating["storage"])
    assert by["new_build"] == pytest.approx(50.0 * battery) and 0.19 < battery < 0.21
    assert by["dsr"] == 300.0 and "cfd" not in set(rows["asset_id"])                # CfD plant stays out
    excluded = bcm.supply_rows(_netted(), {"gas": 0.5}, derating, LEVELS, bcm.Supply2030(ldes_cap_and_floor="excluded"))
    assert bcm.LDES_CATEGORY not in set(excluded["category"])
    low = bcm.supply_rows(_netted(), {}, derating, LEVELS, bcm.Supply2030(storage_efc_scale=0.5, dsr="high"))
    low_by = low.groupby("category")["supply_mw"].sum()
    assert low_by["new_build"] == pytest.approx(0.5 * by["new_build"]) and low_by["dsr"] == 400.0
    assert low_by["existing"] == pytest.approx(1000.0 * 0.9048)                       # no refurbishing share


def test_new_ldes_bids_as_existing_plant_and_the_award_splits_by_carrier():
    distributions = {"existing": (0.9966, 5.2), "new_build": (0.5, 5.2), "dsr": (0.82, 5.2),
                     "interconnector": (1.0, 5.2), "refurbishing": (0.86, 5.2)}
    curve = cm.DemandCurve(40900, 39400, 42400, 49, 75)
    rows = pd.DataFrame({"category": [bcm.LDES_CATEGORY, "existing", "new_build"],
                         "carrier": ["long_duration_storage", "CCGT", "battery"],
                         "supply_mw": [4000.0, 35000.0, 3000.0]})
    cleared = bcm.clear_2030_31(rows, distributions, curve)
    same = cm.clear_auction(pd.Series({"existing": 39000.0, "new_build": 3000.0}), distributions, curve)
    assert cleared["price"] == pytest.approx(round(same["price"], 2))
    assert sum(cleared["awarded_by_category_mw"].values()) == pytest.approx(cleared["awarded_mw"], abs=0.2)
    model = {"distributions": {c: {"floor_share": f, "shape": s} for c, (f, s) in distributions.items()},
             "t4_2030_31": {"central": cleared}}
    by_carrier = bcm.modelled_t4_by_carrier(model)
    assert by_carrier.sum() == pytest.approx(cleared["awarded_mw"])
    assert set(by_carrier.index) == {"long_duration_storage", "CCGT", "battery"}


def _write_fleet(tmp_path, foreign_mw=990.0, extra=None):
    assets = pd.DataFrame({
        "component_type": ["Generator", "Generator", "Generator", "StorageUnit", "Generator"],
        "asset_id": ["g", "w", "eu", "b", "h"], "carrier": ["CCGT", "wind_onshore", "EU_market", "Battery", "nuclear"],
        "bus": ["A", "B", "HVDC_External_France_Calais", "C", "D"],
        "capacity_mw": [500.0, 100.0, foreign_mw, 200.0, 2317.0], "energy_capacity_mwh": [0.0, 0.0, 0.0, 400.0, 0.0],
        "support_type": ["merchant", "merchant", None, None, "CfD"],
        "finance_vintage": ["legacy", "legacy", "new_build", "new_build", "legacy"]})
    if extra is not None:
        assets = pd.concat([assets, extra], ignore_index=True)
    cables = pd.DataFrame({"name": ["IFA", "Isle of Man Interconnector"], "capacity_mw": [900.0, 100.0],
                           "to_bus": ["HVDC_External_France_Calais", "HVDC_External_Isle_Of_Man_Douglas"]})
    assets = pd.concat([assets, pd.DataFrame({
        "component_type": ["Generator"], "asset_id": ["eu_iom"], "carrier": ["EU_market"],
        "bus": ["HVDC_External_Isle_Of_Man_Douglas"], "capacity_mw": [110.0], "energy_capacity_mwh": [0.0],
        "support_type": [None], "finance_vintage": ["new_build"]})], ignore_index=True)
    years = [str(y) for y in range(2023, 2031)]
    es1 = pd.DataFrame([{"Pathway": bcm.CP30_PATHWAY, "Variable": "Capacity (MW)", "SubType": sub,
                         **dict(zip(years, path))}
                        for sub, path in (("Interconnectors", [800.0] * 7 + [1000.0]),
                                          ("Battery", [100.0, 100.0, 100.0, 100.0, 150.0, 150.0, 200.0, 200.0]),
                                          ("Gas", [600.0] * 7 + [500.0]))])
    paths = [tmp_path / "assets.csv", tmp_path / "cables.csv", tmp_path / "es1.csv"]
    for frame, path in zip((assets, cables, es1), paths):
        frame.to_csv(path, index=False)
    # A two-cable composition summing to ES1's 1,000 MW: each cable kept at its rating.
    composition = tmp_path / "composition.csv"
    pd.DataFrame({"name": ["IFA", "Isle of Man Interconnector"], "action": "keep", "cp30_mw": [900.0, 100.0],
                  "first_year": [None, None]}).to_csv(composition, index=False)
    return [*paths, bcm.CP30_PATHWAY, composition]


def test_read_fleet_2030_builds_cables_and_labels_what_stays_out(tmp_path):
    fleet = bcm.read_fleet_2030(*_write_fleet(tmp_path)).set_index("asset_id")
    assert fleet.loc["IFA", "capacity_mw"] == pytest.approx(900.0)                  # its composition rating
    assert fleet.loc["IFA", "derating_class"] == "IFA (France)" and fleet.loc["IFA", "eligible"]
    assert fleet.loc["Isle of Man Interconnector", "outside_reason"] == "no_cm_class"
    assert fleet.loc["w", "outside_reason"] == "merchant_renewable"
    assert fleet.loc["h", "outside_reason"] == "live_low_carbon_contract"
    assert "eu" not in fleet.index                                                  # replaced by its cables
    assert fleet.loc["b", "duration_hours"] == pytest.approx(2.0)
    # Batteries: 100 MW built 2027-2030 of the 200 MW new-build row.
    assert fleet.loc["b", "new_share"] == pytest.approx(0.5)
    assert fleet.loc["g", "new_share"] == 0.0


def test_read_fleet_2030_refuses_mismatched_cables_or_unknown_carriers(tmp_path):
    with pytest.raises(ValueError, match="foreign market generators"):
        bcm.read_fleet_2030(*_write_fleet(tmp_path, foreign_mw=1500.0))
    unknown = pd.DataFrame({"component_type": ["Generator"], "asset_id": ["x"], "carrier": ["fusion"], "bus": ["A"],
                            "capacity_mw": [1.0], "energy_capacity_mwh": [0.0], "support_type": ["merchant"],
                            "finance_vintage": ["new_build"]})
    with pytest.raises(ValueError, match="no 2030/31 CM treatment"):
        bcm.read_fleet_2030(*_write_fleet(tmp_path, extra=unknown))


# ── CP30's cable composition (15 Oct 2026) ──────────────────────────────────

#: The CP30 build's mapped register cables (name, foreign bus, register MW), as the network build reads them.
MAPPED_2030 = {
    "Britned": ("HVDC_External_Netherlands_Maasvlakte", 1000.0), "IFA": ("HVDC_External_France_Calais", 1988.0),
    "IFA2": ("HVDC_External_France_Calais", 1000.0), "Nemo Link": ("HVDC_External_Belgium_Zeebrugge", 1020.0),
    "NS Link": ("HVDC_External_Norway_Kvilldal", 1400.0),
    "Moyle": ("HVDC_External_Northern_Ireland_Ballycronan_More", 500.0),
    "ElecLink": ("HVDC_External_France_Calais", 1000.0), "Viking": ("HVDC_External_Denmark_Revsing", 1400.0),
    "East West Interconnector": ("HVDC_External_Ireland_Rush_North_Beach", 505.0),
    "Greenlink": ("HVDC_External_Ireland_Great_Island", 504.0),
    "Isle of Man Interconnector": ("HVDC_External_Isle_Of_Man_Douglas", 74.0),
    "NeuConnect Interconnector": ("HVDC_External_Germany_Fedderwarden", 1400.0),
}


def _write_composed_fleet(tmp_path, es1_2030=12450.0, foreign=None):
    cables = pd.DataFrame([{"name": n, "capacity_mw": mw, "to_bus": bus} for n, (bus, mw) in MAPPED_2030.items()])
    # The rebuilt register's foreign market generators: 1.1 x the composed cables on each foreign bus.
    foreign = foreign or {"HVDC_External_France_Calais": 4000.0, "HVDC_External_Netherlands_Maasvlakte": 1000.0,
                          "HVDC_External_Belgium_Zeebrugge": 1000.0, "HVDC_External_Norway_Kvilldal": 1400.0,
                          "HVDC_External_Northern_Ireland_Ballycronan_More": 500.0,
                          "HVDC_External_Denmark_Revsing": 1400.0, "HVDC_External_Ireland_Rush_North_Beach": 500.0,
                          "HVDC_External_Ireland_Great_Island": 500.0, "HVDC_External_Germany_Fedderwarden": 1400.0,
                          "HVDC_External_Ireland_Maynooth": 750.0}
    assets = pd.DataFrame([{"component_type": "Generator", "asset_id": f"eu_{bus}", "carrier": "EU_market", "bus": bus,
                            "capacity_mw": 1.1 * mw, "energy_capacity_mwh": 0.0, "support_type": None,
                            "finance_vintage": "new_build"} for bus, mw in foreign.items()]
                          + [{"component_type": "Generator", "asset_id": "g", "carrier": "CCGT", "bus": "A",
                              "capacity_mw": 500.0, "energy_capacity_mwh": 0.0, "support_type": "merchant",
                              "finance_vintage": "legacy"}])
    years = [str(y) for y in range(2023, 2031)]
    es1 = pd.DataFrame([{"Pathway": bcm.CP30_PATHWAY, "Variable": "Capacity (MW)", "SubType": sub,
                         **dict(zip(years, path))}
                        for sub, path in (("Interconnectors", [8400.0, 10300.0, 10300.0, 10300.0, 10300.0,
                                                               11700.0, 11700.0, es1_2030]),
                                          ("Gas", [600.0] * 8))])
    paths = [tmp_path / "assets.csv", tmp_path / "cables.csv", tmp_path / "es1.csv"]
    for frame, path in zip((assets, cables, es1), paths):
        frame.to_csv(path, index=False)
    return paths


def test_composed_cables_take_cp30_ratings_with_maresconnect_and_no_isle_of_man(tmp_path):
    fleet = bcm.read_fleet_2030(*_write_composed_fleet(tmp_path), composition=bcm.FLEET_COMPOSITION)
    cables = fleet[fleet["carrier"].eq("interconnector")].set_index("asset_id")
    assert cables["capacity_mw"].sum() == pytest.approx(12450.0)
    assert "Isle of Man Interconnector" not in cables.index
    # Nominal ratings, no x1.0559 uplift.
    assert cables.loc["IFA", "capacity_mw"] == 2000.0 and cables.loc["Viking", "capacity_mw"] == 1400.0
    assert cables.loc["East West Interconnector", "capacity_mw"] == 500.0
    mares = cables.loc["MaresConnect"]
    assert mares["capacity_mw"] == 750.0 and mares["finance_vintage"] == "new_build" and mares["eligible"]
    assert mares["derating_class"] == "Greenlink (Republic of Ireland)"
    assert (cables.drop("MaresConnect")["finance_vintage"] == "legacy").all()
    # NESO's 2030/31 Republic of Ireland factor for it.
    factors = cm50.read_derating()["classes"]
    assert factors[mares["derating_class"]] == pytest.approx(0.61)


def test_composed_cables_must_match_es1_and_the_register(tmp_path):
    with pytest.raises(ValueError, match="CP30 ES1 has"):
        bcm.read_fleet_2030(*_write_composed_fleet(tmp_path, es1_2030=12524.0), composition=bcm.FLEET_COMPOSITION)
    # A register built before the composition (no MaresConnect generator abroad) is refused, not scaled.
    stale = pd.DataFrame(list(MAPPED_2030.values()), columns=["bus", "mw"]).groupby("bus")["mw"].sum().to_dict()
    with pytest.raises(ValueError, match="foreign market generators"):
        bcm.read_fleet_2030(*_write_composed_fleet(tmp_path, foreign=stale), composition=bcm.FLEET_COMPOSITION)


def test_every_composition_cable_built_in_2030_has_a_cm_class():
    from scripts.CP30.storage_interconnectors import load_cp30_interconnector_composition

    comp = load_cp30_interconnector_composition(bcm.FLEET_COMPOSITION)
    built = comp.loc[comp["action"].isin(["keep", "add"]), "name"]
    assert set(built) <= set(bcm.CABLE_CLASS)
    assert all(bcm.CABLE_CLASS[name] for name in built)
    classes = set(cm50.read_derating()["classes"])
    assert {bcm.CABLE_CLASS[name] for name in built} <= classes


def test_every_capacity_market_sensitivity_reaches_the_exchequer_ledger():
    """The bill's CM cases (the assumptions JSON) and the ledger's uncertainty sources are one list."""
    from scripts.public_ownership import exchequer_ledger as el

    derivation = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))["sourced_externals"]["capacity_market"]["derivation"]
    bill = {f"capacity_market_{name}" for name in derivation["sensitivity_cost_gbp"]}
    ledger = {source for _, source in el._LEVY_SOURCES_2030 if source.startswith("capacity_market_")}
    assert bill == ledger
    assert {f"t4_case_{c}" for c in bcm.CASES_2030 if c != "central"} <= set(derivation["sensitivity_cost_gbp"])
    assert set(derivation["calendar_2030_cost_by_t4_case_gbp"]) == set(bcm.CASES_2030)
    for key, _ in el._LEVY_SOURCES_2030:
        assert key in el.UNCERTAINTIES
    # The 2050 list explains every 2030 CM item it has no counterpart for.
    cm_2030 = {k for k, (line, _) in el.UNCERTAINTIES.items() if line == "capacity_market"}
    assert cm_2030 <= set(el.UNCERTAINTIES_NOT_APPLICABLE_2050)


def test_storage_efc_low_is_nesos_own_low():
    table = pd.read_csv(cm50.DERATING_FILE, encoding="utf-8-sig")
    two_hour = table[table["Generating Technology Class"].eq("Storage (Duration 2h)")
                     & table["Auction Name"].str.contains(r"\(T-4\)")].set_index("Auction Name")["De Rating Factor"]
    recent = two_hour[[a for a in two_hour.index if a[:4] >= "2027"]]
    assert bcm.STORAGE_EFC_LOW == pytest.approx(recent.min() / two_hour[two_hour.index.str.startswith("2030-31")].iloc[0],
                                                abs=1e-4)


# ── new LDES paid its own award (user decision, 6 Oct 2026) ──────────────────

def test_laes_takes_the_cap_and_floor_row_and_pumped_hydro_keeps_the_book():
    assets = pd.DataFrame({"asset_id": ["laes", "phs"], "carrier": ["LAES", "pumped_hydro"],
                           "support_type": [None, None], "capacity_mw": [5000.0, 2800.0]})
    rates = pd.DataFrame({"carrier": ["long_duration_storage", cm.LDES_CAP_AND_FLOOR_RATE_KEY],
                          "gbp_per_kw_connection_year": [47.0, 22.5],
                          "connection_mw_2029_30": [2200.0, 4600.0]})
    out = cm.attach_capacity_market_revenue(assets, rates).set_index("asset_id")["capacity_market_revenue_gbp"]
    # Each earns its own row in full: the award on 4.6 GW, the book on 2.2 GW.
    assert out["laes"] == pytest.approx(4600.0 * 1e3 * 22.5)
    assert out["phs"] == pytest.approx(2200.0 * 1e3 * 47.0)
    # Without the row (every 2050 table) LAES falls back to long_duration_storage, pro rata as before.
    pooled = cm.attach_capacity_market_revenue(assets, rates.iloc[:1]).set_index("asset_id")["capacity_market_revenue_gbp"]
    assert pooled.sum() == pytest.approx(2200.0 * 1e3 * 47.0)
    assert pooled["laes"] / pooled["phs"] == pytest.approx(5000.0 / 2800.0)


def test_the_t4_award_in_connection_mw_follows_each_carriers_own_ratio():
    distributions = {"existing": (0.9966, 5.2), "new_build": (0.5, 5.2)}
    curve = cm.DemandCurve(40900, 39400, 42400, 49, 75)
    rows = pd.DataFrame({"category": [bcm.LDES_CATEGORY, "existing"],
                         "carrier": [cm.LDES_CAP_AND_FLOOR_RATE_KEY, "CCGT"],
                         "free_mw": [4000.0 / 0.9075, 37000.0 / 0.9048], "supply_mw": [4000.0, 37000.0]})
    cleared = bcm.clear_2030_31(rows, distributions, curve)
    model = {"distributions": {c: {"floor_share": f, "shape": s} for c, (f, s) in distributions.items()},
             "t4_2030_31": {"central": cleared}}
    derated = bcm.modelled_t4_by_carrier(model)
    connection = bcm.modelled_t4_connection_by_carrier(model)
    assert connection[cm.LDES_CAP_AND_FLOOR_RATE_KEY] == pytest.approx(derated[cm.LDES_CAP_AND_FLOOR_RATE_KEY] / 0.9075)
    assert connection["CCGT"] == pytest.approx(derated["CCGT"] / 0.9048)


def test_the_ldes_floor_nets_every_income_the_store_earns():
    """User decision (6 Oct 2026): ancillary and sub-hourly revenue count against the floor, where booked."""
    from scripts.public_ownership.public_account import LDES_FLOOR_OTHER_INCOME, ldes_floor_topups

    frame = pd.DataFrame({
        "case_id": ["private"] * 3, "component_type": ["StorageUnit"] * 3, "carrier": ["LAES", "LAES", "Battery"],
        "finance_vintage": ["new_build"] * 3, "annual_required_revenue_gbp": [100.0] * 3,
        "annual_capital_recovery_gbp": [0.0] * 3, "annual_fixed_om_gbp": [100.0] * 3,
        "annual_variable_om_gbp": [0.0] * 3, "wacc": [0.08] * 3, "remaining_life_years": [20.0] * 3,
        "market_revenue_gbp": [30.0] * 3, "capacity_market_revenue_gbp": [10.0] * 3,
        "ancillary_services_revenue_gbp": [5.0, 0.0, 50.0], "battery_subhourly_revenue_gbp": [2.0, 0.0, 9.0]})
    settings = {"ldes_carriers": ["LAES"], "ldes_enabled": True, "ldes_floor_real_rate_by_case": {"private": 0.045}}
    out = ldes_floor_topups(frame, settings)
    assert out["ldes_floor_topup_paid_gbp"].tolist() == pytest.approx([100.0 - 47.0, 100.0 - 40.0, 0.0])
    # A register without the support-line columns (every 2030 register before 3 Oct) floors on market and CM.
    bare = ldes_floor_topups(frame.drop(columns=list(LDES_FLOOR_OTHER_INCOME)), settings)
    assert bare["ldes_floor_topup_paid_gbp"].tolist() == pytest.approx([60.0, 60.0, 0.0])
