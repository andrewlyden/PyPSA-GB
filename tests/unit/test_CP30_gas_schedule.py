"""The CP30 gas station schedule: named stations, per-unit heat rates (MR1, MR2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.CP30.gas_schedule import (
    DATA_SOURCE,
    apply_gas_station_schedule,
    load_gas_station_schedule,
    select_stations_to_target,
    unit_efficiencies,
)

SCHEDULE = Path("data/market/CP30/gas_station_schedule_2030.csv")
COLUMNS = ["station_name", "carrier", "capacity_mw", "commissioning_year", "bus", "bus_source",
           "lat", "lon", "heat_rate_index", "efficiency_evidence", "source"]


def _schedule(tmp_path, rows):
    path = tmp_path / "gas.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def _row(name, carrier, mw, year, bus, index=np.nan):
    return [name, carrier, mw, year, bus, "declared", 52.0, -1.0, index, "evidence", "DUKES"]


def test_oldest_stations_retire_first_and_the_margin_keeps_the_residual():
    stations = pd.DataFrame({
        "station_name": ["old", "mid", "new"],
        "capacity_mw": [500.0, 400.0, 300.0],
        "commissioning_year": [1990, 2000, 2020],
    })
    kept, retired = select_stations_to_target(stations, 500.0)
    kept = kept.set_index("station_name")["capacity_mw"]
    assert kept.to_dict() == {"new": 300.0, "mid": 200.0}
    assert retired.set_index("station_name")["capacity_mw"].to_dict() == {"mid": 200.0, "old": 500.0}


def test_unit_efficiencies_keep_the_class_level_and_spread_by_index():
    kept = pd.DataFrame({"capacity_mw": [1000.0, 500.0, 500.0], "heat_rate_index": [1.1, 0.9, np.nan]})
    eff = unit_efficiencies(kept, 0.52, "CCGT")
    harmonic = kept["capacity_mw"].sum() / (kept["capacity_mw"] / eff).sum()
    assert harmonic == pytest.approx(0.52, abs=1e-9)
    # Lower heat-rate index means higher efficiency; no evidence sits between.
    assert eff.iloc[1] > eff.iloc[2] > eff.iloc[0]


def test_unit_efficiencies_respect_bounds():
    kept = pd.DataFrame({"capacity_mw": [100.0, 100.0], "heat_rate_index": [0.5, 1.5]})
    eff = unit_efficiencies(kept, 0.52, "CCGT")
    # CP30's own CCGT range (workbook chart CP.25).
    assert eff.max() <= 0.581 + 1e-12
    assert eff.min() >= 0.420 - 1e-12


def test_schedule_conserves_each_target_and_fills_the_residual_from_proxy(tmp_path):
    path = _schedule(tmp_path, [
        _row("A", "CCGT", 600.0, 1995, "BUS1", 1.1),
        _row("B", "CCGT", 600.0, 2010, "BUS2", 0.9),
        _row("P", "OCGT", 100.0, 2000, "BUS3"),
    ])
    scaled = pd.DataFrame({
        "fuel_type": ["CCGT", "CCGT", "OCGT", "OCGT", "gas_engine", "CHP"],
        "capacity_mw": [500.0, 500.0, 200.0, 200.0, 300.0, 50.0],
        "gsp": ["Direct(NGET)", "Direct(NGET)", "Direct(NGET)", "Grid", "Grid", "Grid"],
        "station_name": [f"CP30_x_{i}" for i in range(6)],
    })
    targets = {"CCGT": 1000.0, "OCGT": 400.0, "gas_engine": 300.0}
    result = apply_gas_station_schedule(scaled, path, targets, {"CCGT": 0.52, "OCGT": 0.38, "gas_engine": 0.40})
    by_carrier = result.groupby("fuel_type")["capacity_mw"].sum()
    for carrier, mw in targets.items():
        assert by_carrier[carrier] == pytest.approx(mw)
    assert by_carrier["CHP"] == pytest.approx(50.0)

    ccgt = result[result["fuel_type"].eq("CCGT")]
    # Named stations replace every CCGT proxy row; the older one is trimmed.
    assert set(ccgt["data_source"]) == {DATA_SOURCE}
    assert ccgt.set_index("station_name")["capacity_mw"].to_dict() == {"B": 600.0, "A": 400.0}
    assert ccgt.set_index("station_name").loc["B", "unit_efficiency"] > ccgt.set_index("station_name").loc["A", "unit_efficiency"]
    assert ccgt["gsp"].isna().all()
    assert ccgt.set_index("station_name")["build_year"].to_dict() == {"B": 2010, "A": 1995}

    ocgt = result[result["fuel_type"].eq("OCGT")]
    named = ocgt[ocgt["data_source"].eq(DATA_SOURCE)]
    proxy = ocgt[~ocgt["data_source"].eq(DATA_SOURCE)]
    assert named["capacity_mw"].sum() == pytest.approx(100.0)
    assert proxy["capacity_mw"].sum() == pytest.approx(300.0)
    assert set(proxy["gas_capacity_basis"]) == {"fes_proxy_residual"}


@pytest.mark.skipif(not SCHEDULE.exists(), reason="gas schedule not built")
def test_declared_schedule_is_valid_gb_gas():
    schedule = load_gas_station_schedule(SCHEDULE)
    assert set(schedule["carrier"]) == {"CCGT", "OCGT", "gas_engine"}
    assert not schedule["station_name"].isin(["EP Ballylumford CCGT", "CoolkeeraghESB C30"]).any()
    ccgt = schedule[schedule["carrier"].eq("CCGT")]
    # Most CCGT capacity carries 2024 ELEXON evidence, so the fleet has a spread.
    evidenced = ccgt["heat_rate_index"].notna()
    assert ccgt.loc[evidenced, "capacity_mw"].sum() / ccgt["capacity_mw"].sum() > 0.7
    assert ccgt["heat_rate_index"].max() / ccgt["heat_rate_index"].min() > 1.3
    # The busbars the coordinate mapping got wrong are declared.
    buses = schedule.set_index("station_name")["bus"]
    assert buses["Grain"] == "GRAI41"
    assert buses["Keadby"] == "KEAD41"


# ── Retention order (user decision, 15 Oct 2026): CM coverage, FES 2030, age ──

def _stations(rows):
    return pd.DataFrame(rows, columns=["station_name", "capacity_mw", "commissioning_year",
                                       "cm_mw_2029_30", "cm_mw_2030_31", "fes_2030"])


def test_capacity_market_coverage_outranks_age():
    stations = _stations([
        ("old_both_years", 500.0, 1990, 500.0, 500.0, "aggregate"),
        ("mid_one_year", 400.0, 2000, 400.0, 0.0, "aggregate"),
        ("new_uncontracted", 300.0, 2020, 0.0, 0.0, "aggregate"),
    ])
    kept, retired = select_stations_to_target(stations, 700.0)
    assert kept.set_index("station_name")["capacity_mw"].to_dict() == {"old_both_years": 500.0, "mid_one_year": 200.0}
    assert kept.set_index("station_name").loc["mid_one_year", "capacity_basis"] == "dukes_partial_at_retirement_margin"
    assert retired.set_index("station_name")["capacity_mw"].to_dict() == {"mid_one_year": 200.0, "new_uncontracted": 300.0}
    assert retired.set_index("station_name")["retired_cm_years"].to_dict() == {"mid_one_year": 1, "new_uncontracted": 0}


def test_units_without_an_agreement_retire_before_contracted_units_of_older_stations():
    # Connah's Quay pattern: half the station is contracted, half is not.
    stations = _stations([
        ("part_contracted", 1000.0, 2005, 400.0, 0.0, "aggregate"),
        ("older_contracted", 600.0, 1995, 600.0, 0.0, "aggregate"),
    ])
    kept, retired = select_stations_to_target(stations, 1000.0)
    kept = kept.set_index("station_name")
    assert kept["capacity_mw"].to_dict() == {"part_contracted": 400.0, "older_contracted": 600.0}
    assert kept.loc["part_contracted", "capacity_basis"] == "dukes_cm_covered_units"
    assert kept.loc["older_contracted", "capacity_basis"] == "dukes"
    assert retired.set_index("station_name")["capacity_mw"].to_dict() == {"part_contracted": 600.0}


def test_fes_retention_breaks_ties_within_a_coverage_level_before_age():
    stations = _stations([
        ("old_fes_named", 300.0, 1990, 0.0, 0.0, "named"),
        ("new_fes_dropped", 300.0, 2010, 0.0, 0.0, "dropped"),
        ("mid_fes_aggregate", 300.0, 2000, 0.0, 0.0, ""),
    ])
    kept, _ = select_stations_to_target(stations, 600.0)
    assert list(kept["station_name"]) == ["old_fes_named", "mid_fes_aggregate"]


def test_retention_columns_are_validated(tmp_path):
    rows = [_row("A", "CCGT", 600.0, 1995, "BUS1", 1.1)]
    frame = pd.DataFrame(rows, columns=COLUMNS).assign(cm_mw_2029_30=700.0, cm_mw_2030_31=0.0, fes_2030="named")
    frame.to_csv(tmp_path / "over.csv", index=False)
    with pytest.raises(ValueError, match="exceeds the station capacity"):
        load_gas_station_schedule(tmp_path / "over.csv")
    frame.assign(cm_mw_2029_30=600.0, fes_2030="kept").to_csv(tmp_path / "fes.csv", index=False)
    with pytest.raises(ValueError, match="fes_2030 must be one of"):
        load_gas_station_schedule(tmp_path / "fes.csv")


CP30_FFR_2030_CCGT_MW = 23377.95 - 742.0   # NZT carved out of CP30's Gas line (14 Oct 2026)


@pytest.mark.skipif(not SCHEDULE.exists(), reason="gas schedule not built")
def test_cp30_2030_ccgt_fleet_follows_capacity_market_agreements():
    schedule = load_gas_station_schedule(SCHEDULE)
    ccgt = schedule[schedule["carrier"].eq("CCGT")].set_index("station_name")
    # Contracted for 2030 but absent from DUKES 5.11: declared at their TEC connection buses.
    assert ccgt.loc["Sutton Bridge", "bus"] == "WALP41" and ccgt.loc["Sutton Bridge", "cm_mw_2029_30"] == 850.0
    assert ccgt.loc["Eggborough CCGT", "bus"] == "EGGB41" and ccgt.loc["Eggborough CCGT", "cm_mw_2030_31"] == 850.0
    # Only units 1 and 3 of Connah's Quay and phase 2 of South Humber Bank hold agreements.
    assert ccgt.loc["Connahs Quay", "cm_mw_2029_30"] == pytest.approx(690.0)
    assert ccgt.loc["EP SHB LTD", "cm_mw_2030_31"] == pytest.approx(565.5)

    kept, retired = select_stations_to_target(ccgt.reset_index(), CP30_FFR_2030_CCGT_MW)
    assert kept["capacity_mw"].sum() == pytest.approx(CP30_FFR_2030_CCGT_MW, abs=1e-6)
    kept = kept.set_index("station_name")["capacity_mw"]
    covered = ccgt[["cm_mw_2029_30", "cm_mw_2030_31"]].max(axis=1)
    # Every kept MW is under a Capacity Market agreement ...
    assert (kept <= covered.reindex(kept.index) + 1e-6).all()
    # ... and every station contracted for both 2029/30 and 2030/31 is kept whole.
    both = ccgt.index[ccgt["cm_mw_2030_31"] > 0]
    assert (kept.reindex(both).fillna(0.0) >= ccgt.loc[both, "cm_mw_2030_31"] - 1e-6).all()
    assert {"Sutton Bridge", "Eggborough CCGT", "Kings Lynn"} <= set(kept.index)
    # The CP30 total is below the contracted fleet, so one-year holders still retire, oldest first.
    one_year_out = retired[retired["retired_cm_years"].eq(1)]
    assert set(one_year_out["station_name"]) == {
        "Keadby", "Rye House", "Little Barford", "Connahs Quay", "Rocksavage Power Company Ltd", "Didcot B"}
    assert one_year_out["commissioning_year"].max() <= ccgt.loc[kept.index, "commissioning_year"][
        ccgt.loc[kept.index, "cm_mw_2030_31"].eq(0)].min()
    assert "Peterhead" in set(retired.loc[retired["retired_cm_years"].eq(0), "station_name"])


@pytest.mark.skipif(not SCHEDULE.exists(), reason="gas schedule not built")
def test_kept_fleet_keeps_the_class_efficiency_level():
    table = pd.DataFrame({"fuel_type": ["CCGT"], "capacity_mw": [CP30_FFR_2030_CCGT_MW], "gsp": ["Direct(NGET)"],
                          "station_name": ["CP30_CCGT_0"]})
    out = apply_gas_station_schedule(table, SCHEDULE, {"CCGT": CP30_FFR_2030_CCGT_MW}, {"CCGT": 0.52})
    ccgt = out[out["fuel_type"].eq("CCGT")]
    assert set(ccgt["data_source"]) == {DATA_SOURCE}
    harmonic = ccgt["capacity_mw"].sum() / (ccgt["capacity_mw"] / ccgt["unit_efficiency"]).sum()
    assert harmonic == pytest.approx(0.52, abs=1e-9)
    basis = ccgt.set_index("station_name")["gas_capacity_basis"]
    assert basis["EP SHB LTD"] == "dukes_cm_covered_units"
    assert basis["Didcot B"] == "dukes_partial_at_retirement_margin"


@pytest.mark.skipif(not Path("data/market/capacity_market/cm_register_cmu_20260923.csv").exists(),
                    reason="CM register not present")
def test_schedule_cm_columns_match_the_register():
    from scripts.CP30.build_gas_station_schedule import cm_coverage

    schedule = pd.read_csv(SCHEDULE)
    derived = cm_coverage(schedule.drop(columns=["cm_mw_2029_30", "cm_mw_2030_31", "cm_agreements"]))
    for column in ("cm_mw_2029_30", "cm_mw_2030_31"):
        assert derived[column].to_numpy() == pytest.approx(schedule[column].to_numpy())


CM_REGISTER_PATH = Path("data/market/capacity_market/cm_register_cmu_20260923.csv")


@pytest.mark.skipif(not (SCHEDULE.exists() and CM_REGISTER_PATH.exists()), reason="schedule or CM register absent")
def test_retention_check_passes_on_the_committed_schedule_and_catches_drift(tmp_path):
    """F25.9: --check is what --refresh-retention would write, cell for cell."""
    import subprocess
    import sys

    from scripts.CP30.build_gas_station_schedule import check_retention

    report = check_retention(str(SCHEDULE), str(CM_REGISTER_PATH))
    assert report["ok"], report
    assert report["stations"] > 40

    drifted = pd.read_csv(SCHEDULE, dtype={"notes": str})
    drifted.loc[drifted["station_name"].eq("Grain"), "cm_mw_2030_31"] = 1.0
    drifted.loc[drifted["station_name"].eq("Shoreham"), "fes_2030"] = "dropped"
    drifted = drifted[~drifted["station_name"].eq("Sutton Bridge")]   # a declared station goes missing
    path = tmp_path / "gas.csv"
    drifted.to_csv(path, index=False)
    report = check_retention(str(path), str(CM_REGISTER_PATH))
    found = {(d["station_name"], d["column"]) for d in report["schedule_drift"]}
    assert ("Grain", "cm_mw_2030_31") in found
    assert ("Shoreham", "fes_2030") in found
    assert ("Sutton Bridge", None) in found
    assert not report["ok"]

    stamp = tmp_path / "stamp.json"
    run = subprocess.run([sys.executable, "scripts/CP30/build_gas_station_schedule.py", "--check", "--output",
                          str(path), "--cm-register", str(CM_REGISTER_PATH), "--stamp", str(stamp)],
                         capture_output=True, text=True)
    assert run.returncode != 0
    assert "--refresh-retention" in run.stderr
    assert not stamp.exists()
    # The check never writes the schedule.
    assert pd.read_csv(path, dtype={"notes": str}).shape == drifted.shape


def _fes_rows(values):
    rows = [{"Building Block ID Number": "Gen_BB009", "FES Pathway": "Holistic Transition", "GSP": gsp, "2030": mw}
            for gsp, mw in values.items()]
    rows.append({"Building Block ID Number": "Gen_BB009", "FES Pathway": "Electric Engagement", "GSP": "Bolney",
                 "2030": 0.0})
    return pd.DataFrame(rows)


def test_fes_claims_are_checked_against_the_fes_table():
    from scripts.CP30.build_gas_station_schedule import FES_2030, FES_2030_ROWS, fes_claim_drift

    assert set(FES_2030_ROWS) == set(FES_2030)
    good = {"Bolney": 460.0, "Norwich Main": 420.0, "Walpole (_A)": 395.0, "Direct(SHETL)": 0.0, "Grendon": 0.0,
            "Direct(NGET)": 19975.0}
    assert fes_claim_drift(_fes_rows(good)) == []

    bad = dict(good, Bolney=0.0, Grendon=120.0, **{"Direct(NGET)": 18000.0})
    bad.pop("Norwich Main")
    problems = {d["station_name"]: d["problem"] for d in fes_claim_drift(_fes_rows(bad))}
    assert problems == {
        "Shoreham": "declared named but the FES row is zero",
        "Great Yarmouth": "GSP row absent",
        "Corby Power Station": "declared dropped but FES keeps capacity",
        "(aggregate)": "2030 MW differs from the declared value",
    }


def test_verify_rule_reads_the_builder_register():
    """The verify rule's CM register is the one the builder refreshes from."""
    from scripts.CP30.build_gas_station_schedule import CM_REGISTER

    rules = Path("rules/CP30.smk").read_text(encoding="utf-8")
    assert f'GAS_SCHEDULE_CHECK_CM_REGISTER = "{CM_REGISTER}"' in rules
    assert "rule verify_CP30_gas_station_schedule:" in rules


def test_region_split_converts_osgb36_northings():
    import pypsa
    from scripts.generators.integrate_thermal_generators import _get_buses_in_region
    n = pypsa.Network()
    # London, Glasgow area, Inverness area in OSGB36 metres.
    n.add("Bus", "LOND", x=530000.0, y=180000.0)
    n.add("Bus", "GLAS", x=259000.0, y=665000.0)
    n.add("Bus", "INVE", x=266000.0, y=845000.0)
    assert _get_buses_in_region(n, "NGET") == ["LOND"]
    assert _get_buses_in_region(n, "SPTL") == ["GLAS"]
    assert _get_buses_in_region(n, "SHETL") == ["INVE"]


def test_unit_efficiency_takes_precedence_over_the_class_value():
    from scripts.generators.apply_marginal_costs import resolve_thermal_efficiency
    config = {
        "modelled_year": 2030,
        "marginal_costs": {"future_costs": {
            "enabled": True, "case": "central",
            "efficiency_defaults": {"CCGT": 0.52},
            "cases": {"central": {"fuel_prices": {"gas": 30.0}, "carbon": {"uk_ets": 50.0, "carbon_price_support": 18.0}}},
        }},
    }
    named = pd.Series({"carrier": "CCGT", "unit_efficiency": 0.57, "efficiency": 0.5})
    proxy = pd.Series({"carrier": "CCGT", "unit_efficiency": np.nan, "efficiency": 0.5})
    assert resolve_thermal_efficiency("Grain", named, config) == pytest.approx(0.57)
    assert resolve_thermal_efficiency("CP30_CCGT_0", proxy, config) == pytest.approx(0.52)


def test_retention_check_tolerates_metre_scale_position_noise():
    """Positions re-derived on refresh may move about 2 m with the geocoding source; 1 km still drifts."""
    from scripts.CP30.build_gas_station_schedule import schedule_drift

    committed = pd.read_csv(SCHEDULE, dtype={"notes": str})
    nudged = committed.copy()
    nudged.loc[nudged["station_name"].eq("Eggborough CCGT"), ["lat", "lon"]] += [2e-5, -4e-6]
    assert schedule_drift(committed, nudged) == []
    moved = committed.copy()
    moved.loc[moved["station_name"].eq("Sutton Bridge"), "lat"] += 0.01
    assert [(d["station_name"], d["column"]) for d in schedule_drift(committed, moved)] == [("Sutton Bridge", "lat")]
