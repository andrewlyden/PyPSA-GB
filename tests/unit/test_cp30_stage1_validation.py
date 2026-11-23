"""The Stage 1 check against CP30 (rule validate_CP30_stage1, 16 Oct 2026)."""
import json

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.CP30.energy_validation import (
    build_energy_validation,
    cp30_energy_targets,
    overlay_carved_into,
)
from scripts.CP30.stage1_validation import (
    DEFAULT_CONFIG,
    build_stage1_validation,
    curve_column,
    curve_share_at_or_below,
    model_price_at_share,
    price_rows,
    run,
    stage1_config,
)
from scripts.consumer_bill.price_level_mapping import curve_mean

PATHWAY = "Further Flex and Renewables"
CARVE_OUT = {"fleet": {"spatial_proxy": {"committed_overlay_carve_out": {
    "CCS Gas": {"cp30_line": "gas", "from_carrier": "CCGT", "reason": "test: NZT inside the Gas line"}}}}}

# ES1 lines (GWh unless stated). Marine is under 1 TWh: judged on its absolute difference.
ES1 = [
    ("Capacity (MW)", "Thermal", "Gas", "Gas", 1000.0),
    ("Generation (GWh)", "Thermal", "Gas", "Gas", 2000.0),
    ("Capacity (MW)", "Low Carbon", "CCS", "CCS Gas", np.nan),     # CP30 FFR 2030 has no gas CCS
    ("Generation (GWh)", "Low Carbon", "CCS", "CCS Gas", np.nan),
    ("Capacity (MW)", "Renewable", "Solar", "Solar PV", 5000.0),
    ("Generation (GWh)", "Renewable", "Solar", "Solar PV", 4000.0),
    ("Curtailment (TWh)", "Renewable", "Solar", "Solar PV", 1.0),
    ("Capacity (MW)", "Renewable", "Biomass", "Biomass", 1000.0),
    ("Generation (GWh)", "Renewable", "Biomass", "Biomass", 3000.0),
    ("Capacity (MW)", "Renewable", "Other Renewables", "Other Renewables", 1000.0),
    ("Generation (GWh)", "Renewable", "Other Renewables", "Other Renewables", 3000.0),
    ("Capacity (MW)", "Renewable", "Marine", "Marine", 100.0),
    ("Generation (GWh)", "Renewable", "Marine", "Marine", 400.0),
    ("Capacity (MW)", "Storage", "Storage", "Battery", 1000.0),
    ("Generation (GWh)", "Storage", "Storage", "Battery", 2000.0),
    ("Capacity (MW)", "Interconnectors", "Interconnectors", "Interconnectors", 1000.0),
    ("Import (GWh)", "Interconnectors", None, None, 1000.0),
    ("Export (GWh)", "Interconnectors", None, None, 2000.0),
    ("Demand Total (GWh) - Excluding Electrolysis", "Demand", None, None, 10000.0),
]


def _es1(tmp_path):
    frame = pd.DataFrame(ES1, columns=["Variable", "Category", "Type", "SubType", "2030"])
    frame.insert(0, "Pathway", PATHWAY)
    frame.insert(0, "Connection", "Both")
    path = tmp_path / "es1.csv"
    frame.to_csv(path, index=False)
    return path


def _tables(gas=1700.0, ccs=300.0, solar=4000.0, solar_avail=5000.0, biomass=3000.0, biogas=3000.0,
            marine=400.0, imports=1000.0, exports=2000.0, demand=10000.0, battery=500.0):
    """Model energy tables (GWh in, MWh stored) as dataset_energy_tables returns them."""
    gens = pd.DataFrame(
        [("gas1", "CCGT", gas, 1e5), ("nzt", "CCS Gas", ccs, 1e4), ("pv", "solar_pv", solar, solar_avail),
         ("bio", "biomass", biomass, 8e3), ("eng", "biogas", biogas, 8e3), ("tide", "marine", marine, marine)],
        columns=["name", "carrier", "dispatched", "available"]).set_index("name")
    generators = pd.DataFrame({"carrier": gens["carrier"], "bus": "gb", "p_nom": 100.0,
                               "available_mwh": gens["available"] * 1e3, "dispatched_mwh": gens["dispatched"] * 1e3})
    return {"generators": generators, "storage_dispatch_mwh": pd.Series({"Battery": battery * 1e3}),
            "storage_dispatch_existing_pumped_hydro_mwh": pd.Series(dtype=float), "demand_mwh": demand * 1e3,
            "load_shedding_mwh": 0.0, "import_mwh": imports * 1e3, "export_mwh": exports * 1e3,
            "electrolysis_mwh": 0.0, "h2_turbine_mwh": 0.0, "hours": 8760.0,
            "demand_components_mwh": pd.Series(dtype=float)}


def _curve():
    share = np.round(np.arange(0, 101) / 100, 2)
    price = np.maximum(70.0 - 100.0 * share, 0.0)                    # 70 down to 0 at 70% of hours
    return pd.Series(price, index=pd.Index(share, name="share_of_hours"), name="further_flex_and_renewables")


def _prices(n=8760):
    """Hourly prices spread like the curve: 30% of hours at zero."""
    rank = (np.arange(n) + 0.5) / n
    return pd.Series(np.maximum(70.0 - 100.0 * rank, 0.0))


def _table(tmp_path, **kw):
    targets = cp30_energy_targets(_es1(tmp_path), PATHWAY, 2030)
    return build_stage1_validation(targets, _tables(**kw), _prices(), _curve(), None,
                                   overlay_carved_into(CARVE_OUT)).set_index(["component", "item"])


# ── CCS gas inside ES1's Gas line ────────────────────────────────────────────

def test_gas_row_counts_ccs_gas_when_it_is_carved_out_of_the_gas_line(tmp_path):
    targets = cp30_energy_targets(_es1(tmp_path), PATHWAY, 2030)
    assert overlay_carved_into(CARVE_OUT) == {"CCS Gas": "gas"}
    carved = build_energy_validation(targets, _tables(), "dispatch",
                                     carved_into=overlay_carved_into(CARVE_OUT)).set_index(["component", "item"])
    row = carved.loc[("dispatched_energy", "gas (CCGT + OCGT + gas_engine + CCS Gas)")]
    assert row["model_gwh"] == pytest.approx(2000.0)            # 1,700 unabated + 300 NZT
    assert "carved out" in row["note"]
    assert ("model_only_carrier", "CCS Gas") not in carved.index
    # Without the carve-out NZT is a model-only carrier and the Gas row is short by its output.
    plain = build_energy_validation(targets, _tables(), "dispatch").set_index(["component", "item"])
    assert plain.loc[("dispatched_energy", "gas (CCGT + OCGT + gas_engine)"), "model_gwh"] == pytest.approx(1700.0)
    assert plain.loc[("model_only_carrier", "CCS Gas"), "model_gwh"] == pytest.approx(300.0)
    assert overlay_carved_into({}) == {}


def test_a_carved_carrier_with_its_own_es1_line_is_refused(tmp_path):
    targets = cp30_energy_targets(_es1(tmp_path), PATHWAY, 2030)
    with pytest.raises(ValueError, match="own ES1 line"):
        build_energy_validation(targets, _tables(), "dispatch", carved_into={"solar_pv": "gas"})


# ── tolerances ───────────────────────────────────────────────────────────────

def test_main_generation_lines_warn_above_ten_and_fail_above_fifteen_percent(tmp_path):
    gas = ("dispatched_energy", "gas (CCGT + OCGT + gas_engine + CCS Gas)")
    assert _table(tmp_path).loc[gas, "status"] == "PASS"
    assert _table(tmp_path, gas=1700.0 - 240.0).loc[gas, "status"] == "WARN"     # -12%
    assert _table(tmp_path, gas=1700.0 - 320.0).loc[gas, "status"] == "FAIL"     # -16%
    assert _table(tmp_path, biomass=3000 * 0.84).loc[("dispatched_energy", "biomass"), "status"] == "FAIL"
    # biogas is not a main line: it warns but never fails.
    assert _table(tmp_path, biogas=3000 * 0.7).loc[("dispatched_energy", "biogas"), "status"] == "WARN"


def test_lines_under_one_twh_are_judged_on_the_absolute_difference(tmp_path):
    marine = ("dispatched_energy", "marine")
    row = _table(tmp_path, marine=330.0).loc[marine]             # -17.5% of 400 GWh, but 70 GWh of 1 TWh
    assert row["relative_difference"] == pytest.approx(-0.175)
    assert row["judged_difference"] == pytest.approx(-0.07)
    assert row["basis"].startswith("absolute")
    assert row["status"] == "PASS"
    assert _table(tmp_path, marine=280.0).loc[marine, "status"] == "WARN"         # 120 GWh of 1 TWh


def test_curtailment_warns_but_never_fails_and_storage_is_info(tmp_path):
    table = _table(tmp_path, solar=3000.0, solar_avail=5000.0)                   # curtailment 2.0 against 1.0 TWh
    assert table.loc[("curtailment", "solar_pv"), "status"] == "WARN"
    assert table.loc[("dispatched_energy", "solar_pv"), "status"] == "FAIL"       # -25%, a main line
    storage = table.loc[("storage_discharge", "Battery")]
    assert storage["status"] == "INFO" and "simultaneous charge and discharge" in storage["note"]


def test_interconnectors_fail_above_twenty_percent_and_demand_above_half_a_percent(tmp_path):
    imports = ("interconnectors", "Imports")
    assert _table(tmp_path, imports=1150.0).loc[imports, "status"] == "WARN"
    assert _table(tmp_path, imports=1210.0).loc[imports, "status"] == "FAIL"
    assert _table(tmp_path, exports=1580.0).loc[("interconnectors", "Exports"), "status"] == "FAIL"
    demand = ("demand", "Electricity demand excluding electrolysis")
    assert _table(tmp_path, demand=10040.0).loc[demand, "status"] == "PASS"
    assert _table(tmp_path, demand=10060.0).loc[demand, "status"] == "FAIL"


def test_config_refuses_unknown_keys_and_keeps_its_defaults():
    with pytest.raises(ValueError, match="unknown keys"):
        stage1_config({"generation_fail": 0.2})
    assert stage1_config({"generation_fail_rel": 0.2})["generation_warn_rel"] == DEFAULT_CONFIG["generation_warn_rel"]
    assert curve_column("Further Flex and Renewables") == "further_flex_and_renewables"
    assert curve_column("New Dispatch") == "new_dispatch"
    with pytest.raises(ValueError, match="No CP.32 price curve"):
        curve_column("Holistic Transition")


# ── price against CP.32 ──────────────────────────────────────────────────────

def test_price_rows_compare_mean_percentiles_and_zero_share():
    curve = _curve()
    rows = {r["item"]: r for r in price_rows(_prices(), curve, stage1_config(None))}
    assert rows["mean"]["cp30"] == pytest.approx(curve_mean(curve))
    assert rows["mean"]["model"] == pytest.approx(curve_mean(curve), abs=0.05)
    assert rows["mean"]["status"] == "PASS"
    p50 = rows["price exceeded in 50% of hours"]
    assert p50["cp30"] == pytest.approx(20.0) and p50["model"] == pytest.approx(20.0, abs=0.05)
    assert rows["price exceeded in 90% of hours"]["status"] == "PASS"      # both zero
    zero = rows["share of hours at or below GBP 0.5/MWh"]
    assert zero["cp30"] == pytest.approx(0.305, abs=1e-3)
    assert zero["model"] == pytest.approx(0.305, abs=1e-3)
    assert zero["status"] == "PASS"
    assert set(r["status"] for r in rows.values()) == {"PASS"}


def test_price_rows_warn_on_a_low_level_and_on_the_zero_share():
    low = _prices() * 0.6                                                   # mean -40%
    rows = {r["item"]: r for r in price_rows(low, _curve(), stage1_config(None))}
    assert rows["mean"]["status"] == "WARN"
    assert rows["price exceeded in 50% of hours"]["status"] == "WARN"
    # A percentile under GBP 5/MWh away never warns, however large in relative terms.
    near_zero = {r["item"]: r for r in price_rows(_prices() + 1.0, _curve(), stage1_config(None))}
    assert near_zero["price exceeded in 70% of hours"]["status"] == "PASS"
    many_zero = pd.Series(np.where(np.arange(8760) < 8760 * 0.45, 0.0, 80.0))
    zero = {r["item"]: r for r in price_rows(many_zero, _curve(), stage1_config(None))}
    assert zero["share of hours at or below GBP 0.5/MWh"]["status"] == "WARN"    # 0.45 against 0.305
    relaxed = stage1_config({"price_share_warn_rel": {0.5: 0.5}})
    rows = {r["item"]: r for r in price_rows(low, _curve(), relaxed)}
    assert rows["price exceeded in 50% of hours"]["status"] == "PASS"


def test_price_helpers_read_the_curve_and_the_model_the_same_way():
    curve = _curve()
    assert curve_share_at_or_below(curve, 0.5) == pytest.approx(0.305, abs=1e-4)
    assert model_price_at_share(pd.Series(np.arange(101, dtype=float)), 0.1) == pytest.approx(90.0)


# ── the rule: outputs, then a non-zero exit on a FAIL ────────────────────────

def _month_file(path, hours, gas_mw, demand_mw):
    snapshots = pd.date_range("2030-01-01", periods=hours, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "gb", carrier="AC")
    n.add("Bus", "eu", carrier="AC")
    n.add("Load", "demand", bus="gb", carrier="electricity", p_set=pd.Series(demand_mw, index=snapshots))
    n.add("Link", "ic", bus0="gb", bus1="eu", carrier="DC", p_nom=100.0)
    n.add("StorageUnit", "bat", bus="gb", carrier="Battery", p_nom=10.0)
    n.links_t.p0 = pd.DataFrame({"ic": 0.0}, index=snapshots)
    n.storage_units_t.p_dispatch = pd.DataFrame({"bat": 0.0}, index=snapshots)
    n.add("Generator", "gas1", bus="gb", carrier="CCGT", p_nom=1000.0)
    n.add("Generator", "nzt", bus="gb", carrier="CCS Gas", p_nom=100.0)
    n.generators_t.p = pd.DataFrame({"gas1": gas_mw, "nzt": 50.0}, index=snapshots)
    n.export_to_netcdf(path)


def _price_file(path, hours):
    pd.DataFrame({"snapshot": pd.date_range("2030-01-01", periods=hours, freq="h"),
                  "wholesale_price": _prices(hours).to_numpy()}).to_csv(path, index=False)


def test_run_writes_its_outputs_then_fails_and_keeps_a_failed_run_copy(tmp_path):
    es1 = tmp_path / "es1.csv"
    pd.DataFrame([("Generation (GWh)", "Thermal", "Gas", "Gas", 20.0),
                  ("Capacity (MW)", "Thermal", "Gas", "Gas", 1000.0),
                  ("Demand Total (GWh) - Excluding Electrolysis", "Demand", None, None, 40.0)],
                 columns=["Variable", "Category", "Type", "SubType", "2030"]).assign(
        Pathway=PATHWAY, Connection="Both").to_csv(es1, index=False)
    curve = tmp_path / "curve.csv"
    _curve().rename("further_flex_and_renewables").reset_index().assign(
        actual_2023=0.0, counterfactual=0.0, new_dispatch=0.0).to_csv(curve, index=False)
    nets, prices = [], []
    for i in range(2):                                    # two "months" of 24 h: 2 x 24 x (360 + 50) MW
        _month_file(tmp_path / f"m{i}.nc", 24, 360.0, 40.0e3 / 48)
        _price_file(tmp_path / f"m{i}_price.csv", 24)
        nets.append(str(tmp_path / f"m{i}.nc"))
        prices.append(str(tmp_path / f"m{i}_price.csv"))
    out_csv, out_json = tmp_path / "s1.csv", tmp_path / "s1.json"
    # Gas + NZT: 19.68 GWh against 20 (PASS); demand exact.
    table = run(str(es1), PATHWAY, 2030, nets, prices, str(out_csv), str(out_json),
                carved_into={"CCS Gas": "gas"}, curve_file=str(curve), log_path=str(tmp_path / "log.txt"))
    assert not (tmp_path / "s1_failed_run.csv").exists()
    gas = table.set_index(["component", "item"]).loc[("dispatched_energy", "gas (CCGT + OCGT + gas_engine + CCS Gas)")]
    assert gas["model"] == pytest.approx(2 * 24 * 410.0 / 1e3)
    assert json.loads(out_json.read_text())["carved_into"] == {"CCS Gas": "gas"}
    # Without the carve-out the Gas line is 17.3 GWh against 20 (-13.6%, WARN only); a demand gap FAILS.
    for i in range(2):
        _month_file(tmp_path / f"m{i}.nc", 24, 360.0, 41.0e3 / 48)       # demand +2.5%
    with pytest.raises(ValueError, match="Stage 1 validation failed"):
        run(str(es1), PATHWAY, 2030, nets, prices, str(out_csv), str(out_json), curve_file=str(curve),
            log_path=str(tmp_path / "log.txt"))
    kept = pd.read_csv(tmp_path / "s1_failed_run.csv")
    assert out_csv.exists() and out_json.exists()
    assert kept.loc[kept["component"].eq("demand") & kept["cp30"].notna(), "status"].tolist() == ["FAIL"]
    assert json.loads(out_json.read_text())["status_counts"]["FAIL"] == 1
    # fail_on_fail off: the same table, no exception.
    run(str(es1), PATHWAY, 2030, nets, prices, str(out_csv), str(out_json), {"fail_on_fail": False},
        curve_file=str(curve), log_path=str(tmp_path / "log.txt"))


def test_files_without_link_flows_report_electrolysis_and_hydrogen_as_info(tmp_path):
    targets = cp30_energy_targets(_es1(tmp_path), PATHWAY, 2030)
    tables = _tables()
    targets["electrolysis_gwh"] = 1000.0                                  # model reads 0 without link flows
    judged = build_stage1_validation(targets, tables, _prices(), _curve(), link_flows=True)
    row = judged.set_index(["component", "item"]).loc[("electrolysis", "Electrolysis demand")]
    assert row["status"] == "WARN"
    info = build_stage1_validation(targets, tables, _prices(), _curve(), link_flows=False)
    row = info.set_index(["component", "item"]).loc[("electrolysis", "Electrolysis demand")]
    assert row["status"] == "INFO" and "no link flows" in row["note"]


def test_energy_basis_info_reports_es1_energy_as_info_and_keeps_demand_judged():
    """energy_basis info (19 Oct 2026): a Stage 1 fitted to another target (Run B) reports ES1 energy as INFO."""
    import pandas as pd
    from scripts.CP30.stage1_validation import judge_energy, stage1_config
    table = pd.DataFrame([
        {"component": "dispatched_energy", "item": "gas", "cp30_gwh": 15000.0, "model_gwh": 18000.0,
         "note": "", "es1_lines": "Gas"},
        {"component": "curtailment", "item": "wind_offshore", "cp30_gwh": 10000.0, "model_gwh": 5000.0,
         "note": "", "es1_lines": "Offshore Wind"},
        {"component": "demand", "item": "demand", "cp30_gwh": 300000.0, "model_gwh": 290000.0,
         "note": "", "es1_lines": "Demand"},
    ])
    es1 = judge_energy(table, stage1_config({}))
    info = judge_energy(table, stage1_config({"energy_basis": "info"}))
    assert list(es1["status"]) == ["FAIL", "WARN", "FAIL"]
    assert list(info["status"]) == ["INFO", "INFO", "FAIL"]
    with pytest.raises(ValueError):
        judge_energy(table, stage1_config({"energy_basis": "other"}))
