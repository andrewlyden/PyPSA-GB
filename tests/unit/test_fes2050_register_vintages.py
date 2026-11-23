"""FES 2050 plan, task Y10: the stock-flow vintaging and the join onto a register."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import fleet_vintages as fv

LIVES = {name: 10.0 for name in fv.TECHNOLOGIES}


def _trajectory(values: dict[str, list[float]], start: int = 2024) -> pd.DataFrame:
    rows = []
    for tech, series in values.items():
        for i, value in enumerate(series):
            rows.append({"technology": tech, "year": start + i, "capacity_mw": float(value)})
    # every other technology at zero
    years = range(start, start + len(next(iter(values.values()))))
    for tech in fv.TECHNOLOGIES:
        if tech not in values:
            rows += [{"technology": tech, "year": y, "capacity_mw": 0.0} for y in years]
    return pd.DataFrame(rows)


def _evidence(rows):
    return pd.DataFrame(rows, columns=["technology", "commissioning_year", "capacity_mw", "source", "name"])


def _run(values, evidence, *, target, rule=fv.RULE_CENTRAL, lives=LIVES, closures=None):
    return fv.build_fleet_vintages(_trajectory(values), evidence, lives, base_year=2024, target_year=target,
                                   rule=rule, closure_years=closures)


# ── the stock-flow ───────────────────────────────────────────────────────────


def test_every_positive_es1_line_maps_once_and_every_technology_has_a_life():
    assert len(fv.ES1_LINE_TECHNOLOGY) == sum(len(t.es1_lines) for t in fv.STOCK_FLOW_TECHNOLOGIES)
    lives = fv.load_lives()
    assert set(lives) == set(fv.TECHNOLOGIES)
    # Y2's lives (2030 parity) where a row exists, declared ones otherwise.
    # F28 (g, user 26 Oct 2026): onshore wind lives 25 years in both years.
    assert lives["Offshore wind"] == 30 and lives["Onshore wind"] == 25 and lives["Solar PV"] == 38
    assert lives["Lithium batteries"] == 15 and lives["Large-scale nuclear"] == 60
    assert lives["Small modular reactors (SMRs)"] == 40 and lives["Pumped hydro energy storage (PHES)"] == 41
    assert lives["Gas CHP"] == lives["Gas generation - unabated"] == 25
    assert lives["Oil"] == 25 and lives["Geothermal"] == 25
    assert fv.load_lives(source="fes_e02")["Onshore wind"] == 25   # the FES E.02 alternative


def test_a_cohort_retires_at_its_life_and_the_gap_is_new_build_of_that_year():
    # 100 MW built 2020, life 10: in the stock 2020-2029, retires 2030 and is repowered then.
    out = _run({"Onshore wind": [100.0] * 9}, _evidence([("Onshore wind", 2020, 50.0, "REPD", "a")]), target=2032)
    mix = out["mix"].loc[out["mix"]["technology"].eq("Onshore wind")]
    assert mix["build_year"].tolist() == [2030]
    assert mix["capacity_mw"].sum() == pytest.approx(100.0)
    flows = out["flows"].set_index(["technology", "year"]).loc["Onshore wind"]
    assert flows.loc[2030, "scheduled_retirement_mw"] == pytest.approx(100.0)
    assert flows.loc[2030, "gross_addition_mw"] == pytest.approx(100.0)
    assert flows.loc[2030, "repowered_mw"] == pytest.approx(100.0)
    # gross additions = net FES change + retirements, every year
    assert (flows["gross_addition_mw"] - flows["net_fes_change_mw"]
            - flows["scheduled_retirement_mw"] - flows["early_retirement_mw"]).abs().max() == pytest.approx(0.0)


def test_the_target_year_mix_sums_to_fes_capacity_for_every_technology_every_year():
    values = {"Offshore wind": [10, 20, 15, 30, 30, 50, 45, 45], "Solar PV": [5, 5, 5, 4, 3, 8, 9, 20],
              "Lithium batteries": [0, 0, 2, 2, 0, 0, 6, 6]}
    evidence = _evidence([("Offshore wind", 2016, 3, "REPD", "x"), ("Offshore wind", 2010, 5, "REPD", "y"),
                          ("Solar PV", 2012, 1, "REPD", "z")])
    for target in range(2025, 2032):
        out = _run(values, evidence, target=target)
        by = out["mix"].groupby("technology")["capacity_mw"].sum()
        for tech, series in values.items():
            assert by.get(tech, 0.0) == pytest.approx(series[target - 2024]), (tech, target)
        assert out["summary"]["conservation_max_abs_gap_mw"] == pytest.approx(0.0, abs=1e-9)


def test_the_base_stock_takes_the_evidence_shape_and_the_fes_level():
    evidence = _evidence([("Solar PV", 2020, 30.0, "REPD", "a"), ("Solar PV", 2022, 10.0, "REPD", "b")])
    stock = fv.base_stock({"Solar PV": 80.0}, evidence, LIVES, base_year=2024)
    cohorts = {c.build_year: c.capacity_mw for c in stock["Solar PV"]}
    assert cohorts == {2020: pytest.approx(60.0), 2022: pytest.approx(20.0)}
    # no evidence: uniform over the life before the base year
    stock = fv.base_stock({"Gas CHP": 50.0}, _evidence([]), LIVES, base_year=2024)
    assert [c.build_year for c in stock["Gas CHP"]] == list(range(2015, 2025))
    assert sum(c.capacity_mw for c in stock["Gas CHP"]) == pytest.approx(50.0)


def test_past_life_plant_is_extended_centrally_and_retired_in_the_first_year_by_the_bound():
    evidence = _evidence([("Hydropower", 1950, 100.0, "DUKES", "old")])
    central = _run({"Hydropower": [100.0] * 7}, evidence, target=2030)
    assert central["mix"]["build_year"].tolist() == [1950]            # life-extended, still standing
    strict = _run({"Hydropower": [100.0] * 7}, evidence, target=2030, rule=fv.RULE_STRICT)
    assert strict["mix"]["build_year"].tolist() == [2025]             # retired 2025, repowered 2025
    assert strict["flows"].set_index(["technology", "year"]).loc[("Hydropower", 2025), "repowered_mw"] == 100.0


def test_a_run_down_technology_is_not_repowered_like_for_like():
    # FES runs gas down (target below base): a cohort reaching its life in the horizon is extended, not replaced.
    evidence = _evidence([("Gas generation - unabated", 2016, 100.0, "DUKES", "ccgt")])
    out = _run({"Gas generation - unabated": [100, 100, 100, 90, 80, 60, 40, 20, 10]}, evidence, target=2032)
    flows = out["flows"].loc[out["flows"]["technology"].eq("Gas generation - unabated")]
    assert flows["gross_addition_mw"].sum() == 0.0
    assert out["mix"]["build_year"].tolist() == [2016]
    assert "Gas generation - unabated" in out["summary"]["run_down_technologies"]
    # the bound retires it at its life (2026) and replaces it
    strict = _run({"Gas generation - unabated": [100, 100, 100, 90, 80, 60, 40, 20, 10]}, evidence, target=2032,
                  rule=fv.RULE_STRICT)
    assert strict["flows"].loc[strict["flows"]["technology"].eq("Gas generation - unabated"),
                               "gross_addition_mw"].sum() > 0.0


def test_fes_decline_retires_the_oldest_first():
    evidence = _evidence([("Offshore wind", 2018, 50.0, "REPD", "a"), ("Offshore wind", 2022, 50.0, "REPD", "b")])
    out = _run({"Offshore wind": [100, 100, 60, 60]}, evidence, target=2027)
    mix = out["mix"].set_index("build_year")["capacity_mw"]
    assert mix.to_dict() == {2018: pytest.approx(10.0), 2022: pytest.approx(50.0)}
    events = out["retirements"]
    assert events["reason"].eq(fv.REASON_FES_DECLINE).all() and events["build_year"].eq(2018).all()


def test_a_published_closure_after_the_base_year_holds_the_station_through_that_year():
    evidence = _evidence([("Large-scale nuclear", 1995, 1200.0, "DUKES", "Sizewell B"),
                          ("Large-scale nuclear", 1983, 1000.0, "DUKES", "Hartlepool")])
    closures = {"Sizewell B": 2026, "Hartlepool": 2024}                  # Hartlepool's date is stale
    lives = dict(LIVES, **{"Large-scale nuclear": 60.0})
    out = _run({"Large-scale nuclear": [2200, 2200, 2200, 1000, 1000]}, evidence, target=2028,
               lives=lives, closures=closures)
    events = out["retirements"]
    closed = events.loc[events["reason"].eq(fv.REASON_PUBLISHED_CLOSURE)]
    assert closed["year"].tolist() == [2027] and closed["build_year"].tolist() == [1995]
    assert out["mix"]["build_year"].tolist() == [1983]                  # stale date: left to the FES decline


def test_retirements_by_decade_and_the_vintage_summary_add_up():
    evidence = _evidence([("Onshore wind", 2016, 10.0, "REPD", "a")])
    out = _run({"Onshore wind": [10.0] * 27}, evidence, target=2050, lives=dict(LIVES, **{"Onshore wind": 12.0}))
    decades = out["retired_by_decade"]
    total = decades.loc[decades["technology"].eq("TOTAL")].groupby("decade")["capacity_mw"].sum()
    assert set(total.index) <= {"2025-2030", "2031-2040", "2041-2050"}
    assert total.sum() == pytest.approx(out["retirements"]["capacity_mw"].sum())
    summary = out["vintage_summary"].set_index("technology").loc["Onshore wind"]
    assert summary["total_mw"] == pytest.approx(10.0)
    assert fv.decade_label(2030) == "2025-2030" and fv.decade_label(2031) == "2031-2040"
    assert fv.decade_label(2050) == "2041-2050"


@pytest.mark.skipif(not Path("data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx").exists(),
                    reason="FES 2025 workbook not present")
def test_the_real_ht_2050_mix_sums_to_es1_by_technology():
    from scripts.FES.fes2025_targets import classify_es1, read_es1

    es1 = read_es1()
    trajectory = fv.fes_capacity_trajectory(es1, years=range(2024, 2051))
    repd = pd.read_csv(fv.DEFAULT_REPD, encoding="latin1", low_memory=False) if fv.DEFAULT_REPD.exists() else None
    dukes = pd.read_csv(fv.DEFAULT_DUKES) if fv.DEFAULT_DUKES.exists() else None
    out = fv.build_fleet_vintages(trajectory, fv.existing_fleet_evidence(repd, dukes), fv.load_lives(),
                                  closure_years=fv.load_closure_dates())
    es1_2050 = classify_es1(es1, "Holistic Transition", 2050)
    es1_2050 = es1_2050.loc[es1_2050["quantity"].eq("capacity_mw") & es1_2050["role"].isin(fv.CAPACITY_ROLES)]
    assert out["mix"]["capacity_mw"].sum() == pytest.approx(es1_2050["value_std"].sum(), rel=1e-12)
    by = out["mix"].groupby("technology")["capacity_mw"].sum()
    lines =es1_2050.assign(tech=[fv.ES1_LINE_TECHNOLOGY[(t, s)] for t, s in zip(es1_2050["type"], es1_2050["subtype"])])
    for tech, value in lines.groupby("tech")["value_std"].sum().items():
        assert by.get(tech, 0.0) == pytest.approx(value, abs=1e-6), tech
    assert by["Offshore wind"] == pytest.approx(103930.5, abs=1.0)      # on-grid, off-grid block excluded


# ── the join ─────────────────────────────────────────────────────────────────


def _mix(cells):
    return pd.DataFrame([{"technology": t, "build_year": y, "capacity_mw": c} for t, y, c in cells])


def _register(rows):
    return pd.DataFrame([{"component_type": "Generator", "asset_id": a, "carrier": c, "bus": b, "capacity_mw": mw,
                          "dispatch_mwh": mw * 1000.0, "commissioning_year": year}
                         for a, c, b, mw, year in rows])


def test_pro_rata_join_reproduces_the_mix_exactly_and_keeps_each_row():
    mix = _mix([("Offshore wind", 2020, 30.0), ("Offshore wind", 2035, 50.0), ("Offshore wind", 2045, 20.0)])
    register = _register([("OFF_A", "wind_offshore", "B1", 60.0, np.nan), ("OFF_B", "wind_offshore", "B2", 40.0, np.nan)])
    cohorts, report = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives())
    by_year = cohorts.groupby("build_year")["capacity_mw"].sum()
    assert by_year.to_dict() == {2020: pytest.approx(30.0), 2035: pytest.approx(50.0), 2045: pytest.approx(20.0)}
    per_row = cohorts.assign(base=cohorts["asset_id"].str.split("::").str[0]).groupby("base")
    assert per_row["capacity_mw"].sum().to_dict() == {"OFF_A": pytest.approx(60.0), "OFF_B": pytest.approx(40.0)}
    assert per_row["cohort_share"].sum().to_dict() == {"OFF_A": pytest.approx(1.0), "OFF_B": pytest.approx(1.0)}
    assert cohorts["dispatch_mwh"].sum() == pytest.approx(100_000.0)
    assert set(cohorts.loc[cohorts["build_year"].eq(2020), "finance_vintage"]) == {"legacy"}
    assert set(cohorts.loc[cohorts["build_year"] >= 2025, "finance_vintage"]) == {"new_build"}
    assert (cohorts["commissioning_year"] == cohorts["build_year"]).all()
    legacy = cohorts.loc[cohorts["asset_id"].str.startswith("OFF_A")]
    assert legacy["cohort_legacy_share"].iloc[0] == pytest.approx(0.3)


def test_a_dated_row_keeps_its_year_or_takes_its_repowering_year():
    lives = {**fv.load_lives(), "Onshore wind": 35.0}                   # onshore 35 years (the test's own life)
    mix = _mix([("Onshore wind", 2020, 10.0), ("Onshore wind", 2045, 10.0), ("Onshore wind", 2040, 5.0)])
    register = _register([("ON_NEW", "wind_onshore", "B", 10.0, 2020.0),   # still standing in 2050
                          ("ON_OLD", "wind_onshore", "B", 10.0, 2010.0),   # 2010 + 35 = 2045: repowered
                          ("ON_X", "wind_onshore", "B", 5.0, np.nan)])
    cohorts, _ = fv.assign_build_years(register, mix, model_year=2050, lives=lives)
    rows = cohorts.set_index("asset_id")
    assert rows.loc["ON_NEW", "build_year"] == 2020 and rows.loc["ON_NEW", "vintage_method"] == fv.METHOD_DATED
    assert rows.loc["ON_OLD", "build_year"] == 2045 and rows.loc["ON_OLD", "vintage_method"] == fv.METHOD_REPOWERED
    assert rows.loc["ON_OLD", "site_commissioning_year"] == 2010
    assert rows.loc["ON_X", "build_year"] == 2040
    assert fv.repowering_year(2010, 15, 2050) == 2040 and fv.repowering_year(2040, 15, 2050) == 2040


def test_bus_located_survivors_go_to_their_bus_first():
    mix = _mix([("Solar PV", 2015, 20.0), ("Solar PV", 2040, 80.0)])
    register = _register([("S1", "solar_pv", "NORTH", 50.0, np.nan), ("S2", "solar_pv", "SOUTH", 50.0, np.nan)])
    sites = pd.DataFrame([{"technology": "Solar PV", "bus": "SOUTH", "build_year": 2015, "capacity_mw": 20.0}])
    cohorts, report = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives(), site_vintages=sites)
    south = cohorts.loc[cohorts["asset_id"].str.startswith("S2")].set_index("build_year")["capacity_mw"]
    north = cohorts.loc[cohorts["asset_id"].str.startswith("S1")].set_index("build_year")["capacity_mw"]
    assert south.to_dict() == {2015: pytest.approx(20.0), 2040: pytest.approx(30.0)}
    assert north.to_dict() == {2040: pytest.approx(50.0)}
    assert report.iloc[0]["assigned_bus_located_survivor_mw"] == pytest.approx(20.0)


def test_whole_units_give_one_year_per_row_within_half_a_unit_per_cell():
    rng = np.random.default_rng(3)
    sizes = rng.uniform(5, 50, 60)
    register = _register([(f"B{i:03d}", "Battery", "X", float(s), np.nan) for i, s in enumerate(sizes)])
    register["component_type"] = "StorageUnit"
    total = float(sizes.sum())
    mix = _mix([("Lithium batteries", 2038, 0.2 * total), ("Lithium batteries", 2044, 0.5 * total),
                ("Lithium batteries", 2049, 0.3 * total)])
    cohorts, _ = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives(), mode=fv.JOIN_WHOLE_UNITS)
    assert len(cohorts) == len(register) and cohorts["asset_id"].tolist() == register["asset_id"].tolist()
    achieved = cohorts.groupby("build_year")["capacity_mw"].sum()
    target = mix.set_index("build_year")["capacity_mw"]
    assert (achieved - target).abs().max() <= sizes.max() / 2 + 1e-9
    again, _ = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives(), mode=fv.JOIN_WHOLE_UNITS)
    assert again["build_year"].tolist() == cohorts["build_year"].tolist()   # deterministic


def test_a_register_not_equal_to_fes_keeps_its_capacity_and_takes_the_mix_shares():
    mix = _mix([("Offshore wind", 2030, 60.0), ("Offshore wind", 2040, 40.0)])       # FES 100 MW
    register = _register([("OFF", "wind_offshore", "B", 50.0, np.nan)])               # the network holds 50
    cohorts, report = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives())
    assert cohorts["capacity_mw"].sum() == pytest.approx(50.0)
    assert cohorts.set_index("build_year")["capacity_mw"].to_dict() == {2030: pytest.approx(30.0),
                                                                        2040: pytest.approx(20.0)}
    assert report.iloc[0]["mix_scale"] == pytest.approx(0.5)


def test_a_carrier_naming_two_technologies_pools_them_unless_the_row_says_which():
    mix = _mix([("Large-scale nuclear", 2035, 3000.0), ("Small modular reactors (SMRs)", 2045, 1000.0)])
    pooled = _register([("NUC", "nuclear", "B", 4000.0, np.nan)])
    cohorts, _ = fv.assign_build_years(pooled, mix, model_year=2050, lives=fv.load_lives())
    assert cohorts.groupby("stock_flow_technology")["capacity_mw"].sum().to_dict() == {
        "Large-scale nuclear": pytest.approx(3000.0), "Small modular reactors (SMRs)": pytest.approx(1000.0)}
    hinted = _register([("SMR1", "nuclear", "B", 1000.0, np.nan), ("HPC", "nuclear", "B", 3000.0, np.nan)])
    hinted["stock_flow_technology"] = ["Small modular reactors (SMRs)", "Large-scale nuclear"]
    cohorts, _ = fv.assign_build_years(hinted, mix, model_year=2050, lives=fv.load_lives())
    assert cohorts.set_index("asset_id")["build_year"].to_dict() == {"SMR1": 2045, "HPC": 2035}


def test_rows_without_a_stock_flow_technology_pass_through_unchanged():
    register = _register([("EU", "EU_market", "X", 100.0, np.nan)])
    cohorts, _ = fv.assign_build_years(register, _mix([("Solar PV", 2040, 1.0)]), model_year=2050,
                                       lives=fv.load_lives())
    assert cohorts["asset_id"].tolist() == ["EU"] and cohorts["vintage_method"].tolist() == [fv.METHOD_NONE]


def test_the_vintaged_register_path_is_refused_below_the_vintaged_years():
    from scripts.public_ownership.build_asset_finance_register import vintage_register

    register = _register([("OFF", "wind_offshore", "B", 50.0, np.nan)])
    with pytest.raises(ValueError, match="vintaged model years"):
        vintage_register(register, _mix([("Offshore wind", 2030, 50.0)]), model_year=2030)
    with pytest.raises(ValueError, match="baseline"):
        vintage_register(register, _mix([("Offshore wind", 2030, 50.0)]), model_year=2050, baseline_given=True)
    out = vintage_register(register, _mix([("Offshore wind", 2030, 50.0)]), model_year=2050)
    # (the known contracts' LCCC unit phases may split the row; the build year is unchanged)
    assert set(out["build_year"].tolist()) == {2030} and out["capacity_mw"].sum() == pytest.approx(50.0)


# ── support contracts on the vintaged register (with task Y11) ───────────────


def _contracts(rows):
    return pd.DataFrame([{"contract_id": cid, "phase": 1, "cfd_round": rnd, "technology": tech, "scheme": "CfD",
                          "capacity_mw": mw, "start_year": start, "live_share": share}
                         for cid, rnd, tech, mw, start, share in rows])


def test_known_contracts_go_first_then_post_ar7_cfds_and_dpas_on_their_terms():
    mix = _mix([("Offshore wind", 2030, 100.0), ("Offshore wind", 2040, 100.0), ("Offshore wind", 2027, 50.0),
                ("Gas generation - CCUS - mature", 2034, 100.0), ("Gas generation - CCUS - mature", 2040, 100.0),
                ("Lithium batteries", 2045, 10.0)])
    register = _register([(f"OFF{i}", "wind_offshore", "B", 25.0, np.nan) for i in range(10)]
                         + [("CCS1", "CCS Gas", "B", 100.0, np.nan), ("CCS2", "CCS Gas", "B", 100.0, np.nan)])
    cohorts, _ = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives(), mode=fv.JOIN_WHOLE_UNITS)
    contracts = _contracts([("AR7_X", "AR7", "Offshore wind", 50.0, 2030.25, 0.71875),
                            ("AR7_OLD", "AR7", "Offshore wind", 50.0, 2027.25, 0.0)])     # expired: not used
    labelled, report = fv.label_support_contracts(cohorts, contracts, model_year=2050, award_years={"AR7": 2026})
    off = labelled.loc[labelled["stock_flow_technology"].eq("Offshore wind")]
    known = off.loc[off["support_basis"].eq(fv.SUPPORT_BASIS_KNOWN)]
    assert known["capacity_mw"].sum() == pytest.approx(50.0) and set(known["build_year"]) == {2030}
    assert set(known["cfd_round"]) == {"AR7"} and known["cfd_live_share"].eq(0.71875).all()
    assert known["cfd_award_year"].eq(2026).all() and known["contract_vintage"].eq("post_policy").all()
    assert report.set_index("contract_id").loc["AR7_X", "assigned_mw"] == pytest.approx(50.0)
    par7 = off.loc[off["build_year"].eq(2040)]
    assert set(par7["cfd_round"]) == {"PAR7_2040"} and set(par7["support_type"]) == {"CfD"}
    # User decision (13 Nov 2026): the rest of the 2030 cohort holds the 2030 chain's modelled AR8 CfD (20 years
    # from 2030, so half live in 2050); the 2027 cohort's earlier CfDs have expired.
    ar8 = off.loc[off["build_year"].eq(2030) & ~off["support_basis"].eq(fv.SUPPORT_BASIS_KNOWN)]
    assert set(ar8["cfd_round"]) == {fv.AR8_MODELLED_ROUND} and set(ar8["support_type"]) == {"CfD"}
    from scripts.market.capacity_market_2050 import live_fraction

    assert ar8["cfd_live_share"].eq(live_fraction(2030, 20, 2050)).all()
    expired = off.loc[off["build_year"].eq(2027) & ~off["support_basis"].eq(fv.SUPPORT_BASIS_KNOWN)]
    assert set(expired["support_type"]) == {"merchant"} and expired["cfd_round"].isna().all()
    ccs = labelled.loc[labelled["stock_flow_technology"].eq("Gas generation - CCUS - mature")].set_index("build_year")
    assert ccs.loc[2040, "support_type"] == "DPA" and ccs.loc[2040, "cfd_round"] == "DPA_2040"   # 2040 + 15 > 2050
    assert ccs.loc[2034, "support_type"] == "merchant"                                            # 2034 + 15 = 2049


def test_known_contracts_hold_exactly_their_mw_by_splitting_the_last_row():
    """Review 2 (rebuilt 24 Oct 2026): HPC 1,630 + 1,630, Sizewell C 3,260, exact; capacity and shares kept."""
    mix = _mix([("Large-scale nuclear", 2031, 3260.0), ("Large-scale nuclear", 2038, 4750.0)])
    register = _register([("NUC_A", "nuclear", "B1", 2000.0, np.nan), ("NUC_B", "nuclear", "B2", 1733.0, np.nan),
                          ("NUC_C", "nuclear", "B3", 4277.0, np.nan)])
    register["stock_flow_technology"] = "Large-scale nuclear"
    cohorts, _ = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives())
    contracts = pd.DataFrame([
        {"contract_id": "HPC_UNIT_1", "phase": 1, "cfd_round": "HPC", "technology": "Large-scale nuclear",
         "scheme": "CfD", "capacity_mw": 1630.0, "start_year": 2031.0, "live_share": 1.0},
        {"contract_id": "HPC_UNIT_2", "phase": 1, "cfd_round": "HPC", "technology": "Large-scale nuclear",
         "scheme": "CfD", "capacity_mw": 1630.0, "start_year": 2031.5, "live_share": 1.0},
        {"contract_id": "SIZEWELL_C", "phase": 1, "cfd_round": "SZC_RAB", "technology": "Large-scale nuclear",
         "scheme": "RAB", "capacity_mw": 3260.0, "start_year": 2038.75, "live_share": 1.0}])
    labelled, report = fv.label_support_contracts(cohorts, contracts, model_year=2050, award_years={"HPC": 2016})
    # every phase exact (the whole-row fill tagged HPC 3,733 and Sizewell C 2,947 MW on the live build)
    by_report = report.set_index("contract_id")["assigned_mw"]
    held = labelled.dropna(subset=["cfd_contract_id"]).groupby("cfd_contract_id")["capacity_mw"].sum()
    for contract, mw in (("HPC_UNIT_1", 1630.0), ("HPC_UNIT_2", 1630.0), ("SIZEWELL_C", 3260.0)):
        assert by_report[contract] == pytest.approx(mw, abs=1e-6)
        assert held[f"{contract}#1"] == pytest.approx(mw, abs=1e-6)
    assert set(labelled.loc[labelled["cfd_contract_id"].eq("SIZEWELL_C#1"), "build_year"]) == {2038}
    assert labelled.loc[labelled["cfd_contract_id"].eq("SIZEWELL_C#1"), "support_type"].eq("RAB").all()
    # a split part is named for its contract; the rest keeps the row's id; every id is unique
    assert labelled["asset_id"].is_unique
    split = report["split_asset_id"].dropna()
    assert len(split) >= 1
    for c in report.loc[report["split_asset_id"].notna()].itertuples(index=False):
        part = f"{c.split_asset_id}{fv.COHORT_SEPARATOR}k{c.contract_id}_{c.phase}"
        assert labelled.set_index("asset_id").loc[part, "cfd_contract_id"] == f"{c.contract_id}#{c.phase}"
    # capacity, dispatch and each component's cohort shares are unchanged, per component and in total
    base = lambda f: f["asset_id"].str.split("::").str[0]  # noqa: E731
    for column in ("capacity_mw", "dispatch_mwh", "cohort_share"):
        before = cohorts.groupby(base(cohorts))[column].sum()
        after = labelled.groupby(base(labelled))[column].sum()
        pd.testing.assert_series_equal(after, before, check_exact=False, rtol=1e-12, atol=1e-9)
    # the 2038 cohort's rest is outside every known contract (it takes the modelled PAR7_2038 label)
    rest = labelled.loc[labelled["build_year"].eq(2038) & labelled["cfd_contract_id"].isna()]
    assert rest["capacity_mw"].sum() == pytest.approx(4750.0 - 3260.0)


def test_label_rows_are_unchanged_when_no_split_is_needed():
    mix = _mix([("Offshore wind", 2030, 100.0), ("Offshore wind", 2040, 100.0)])
    register = _register([(f"OFF{i}", "wind_offshore", "B", 25.0, np.nan) for i in range(8)])
    cohorts, _ = fv.assign_build_years(register, mix, model_year=2050, lives=fv.load_lives(), mode=fv.JOIN_WHOLE_UNITS)
    contracts = _contracts([("AR7_X", "AR7", "Offshore wind", 50.0, 2030.25, 0.71875)])
    labelled, report = fv.label_support_contracts(cohorts, contracts, model_year=2050, award_years={"AR7": 2026})
    assert labelled["asset_id"].tolist() == cohorts["asset_id"].tolist()
    assert labelled["capacity_mw"].tolist() == cohorts["capacity_mw"].tolist()
    assert report["split_asset_id"].isna().all()


def test_y11_live_contracts_reach_the_register_through_the_vintaged_path():
    from scripts.market.cfd_strikes_2050 import known_contracts_2050
    from scripts.public_ownership.build_asset_finance_register import vintage_register

    contracts = known_contracts_2050(2050)
    hpc = contracts.loc[contracts["cfd_round"].eq("HPC")]
    mix = _mix([("Large-scale nuclear", 2030, 1630.0), ("Large-scale nuclear", 2031, 1630.0),
                ("Large-scale nuclear", 2039, 3260.0)])
    register = _register([("HPC1", "nuclear", "B", 1630.0, np.nan), ("HPC2", "nuclear", "B", 1630.0, np.nan),
                          ("SZC", "nuclear", "B", 3260.0, np.nan)])
    register["stock_flow_technology"] = "Large-scale nuclear"
    out = vintage_register(register, mix, model_year=2050, mode=fv.JOIN_WHOLE_UNITS,
                           support_contracts=contracts, award_years={"HPC": 2016})
    rows = out.set_index("asset_id")
    assert set(rows.loc[["HPC1", "HPC2"], "cfd_round"]) == {"HPC"} and len(hpc) == 2
    assert rows.loc["SZC", "support_type"] == "RAB"
    assert rows.loc["HPC1", "cfd_award_year"] == 2016
    assert rows.loc["HPC1", "contract_vintage"] == "pre_policy"          # a 2016 award: outside P2's boundary


def test_consumer_owned_flexibility_daccs_and_the_hydrogen_system_leave_the_vintaged_register():
    from scripts.demand.demand_carriers import CONSUMER_OWNED_CARRIERS, NON_ELECTRICITY_ASSET_CARRIERS
    from scripts.public_ownership.build_asset_finance_register import non_ownership_carriers, vintage_register

    carriers = non_ownership_carriers()
    assert set(CONSUMER_OWNED_CARRIERS) | set(NON_ELECTRICITY_ASSET_CARRIERS) <= set(carriers)
    assert {"V2G_fleet", "EV_smart_charging", "DACCS", "gas_reforming_ccs", "gas_supply"} <= set(carriers)
    register = _register([("OFF", "wind_offshore", "B", 50.0, np.nan), ("V2G_B", "V2G_fleet", "B", 900.0, np.nan),
                          ("DAC", "DACCS", "B", 100.0, np.nan), ("GASS", "gas_supply", "G", 5000.0, np.nan)])
    out = vintage_register(register, _mix([("Offshore wind", 2040, 50.0)]), model_year=2050, label_support=False)
    assert out["asset_id"].tolist() == ["OFF"]
    assert out.attrs["non_ownership_excluded"].sum() == pytest.approx(6000.0)



def test_the_2029_30_cohort_takes_the_ar8_round_on_the_cp30_cfd_shares():
    """User decision (13 Nov 2026): 2029-30 renewables off a known contract hold AR8_modelled (solar half)."""
    mix = _mix([("Offshore wind", 2029, 100.0), ("Solar PV", 2030, 100.0), ("Onshore wind", 2028, 100.0)])
    register = _register([("OFF", "wind_offshore", "B", 100.0, np.nan), ("SOL", "solar_pv", "B", 100.0, np.nan),
                          ("ONS", "wind_onshore", "B", 100.0, np.nan)])
    cohorts, _ = fv.assign_build_years(register, mix, model_year=2040, lives=fv.load_lives(), mode=fv.JOIN_WHOLE_UNITS)
    labelled, _ = fv.label_support_contracts(cohorts, _contracts([("OLD", "AR1", "Offshore wind", 1.0, 2016.25, 0.0)]), model_year=2040,
                                             award_years={fv.AR8_MODELLED_ROUND: 2027})
    by = labelled.set_index("stock_flow_technology")
    assert by.loc["Offshore wind", "cfd_round"] == fv.AR8_MODELLED_ROUND
    assert by.loc["Offshore wind", "cfd_live_share"] == pytest.approx(1.0)
    assert by.loc["Offshore wind", "cfd_award_year"] == 2027 and by.loc["Offshore wind", "contract_vintage"] == "post_policy"
    assert by.loc["Solar PV", "cfd_round"] == fv.AR8_MODELLED_ROUND
    assert by.loc["Solar PV", "cfd_live_share"] == pytest.approx(0.5)
    assert by.loc["Onshore wind", "support_type"] == "merchant"        # 2028: before the AR8 build years
