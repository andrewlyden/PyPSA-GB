"""FES 2050 plan, task Y14-prep: four follow-ups to Y10 and Y11.

1. Settlement of a vintaged register's build-year cohorts (``cohort_share``).
2. Strikes and DPA rates price insurance on the register's basis (exclude + D4(a)).
3. Hydrogen-turbine Links in the vintaged register.
4. The 2050 crosswalk supplement (CCS gas, SMR, hydrogen turbines, CAES, geothermal; LAES life).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import fleet_vintages as fv
from scripts.public_ownership.settlement import attach_settlement, vintaged_settlement_share

ROOT = Path(__file__).resolve().parents[2]
CROSSWALK = ROOT / "data" / "public_ownership" / "financing" / "pypsa_finance_technology_crosswalk.csv"
CROSSWALK_2050 = ROOT / "data" / "public_ownership" / "financing" / "pypsa_finance_technology_crosswalk_2050.csv"
SNAPSHOT = ROOT / "resources" / "public_ownership" / "workbook_v5_assumptions.yaml"
needs_snapshot = pytest.mark.skipif(not SNAPSHOT.exists(), reason="workbook snapshot not extracted")


def _settlement(rows):
    return pd.DataFrame([{"component_type": c, "asset_id": a, "market_revenue_gbp": m, "support_revenue_gbp": s,
                          "fuel_carbon_gbp": f} for c, a, m, s, f in rows])


# ── 1. settlement of build-year cohorts ──────────────────────────────────────


def test_build_year_cohorts_take_their_cohort_share_and_sum_back_to_the_asset():
    assets = pd.DataFrame([
        {"component_type": "Generator", "asset_id": "OFF::b2031", "cohort_share": 0.25, "cohort_legacy_share": 0.0},
        {"component_type": "Generator", "asset_id": "OFF::b2045", "cohort_share": 0.75, "cohort_legacy_share": 0.0},
        {"component_type": "Generator", "asset_id": "SOL", "cohort_share": 1.0, "cohort_legacy_share": 0.0},
        {"component_type": "Link", "asset_id": "H2_turbine_X", "cohort_share": 1.0, "cohort_legacy_share": 0.0},
    ])
    settlement = _settlement([("Generator", "OFF", 1000.0, 400.0, 0.0), ("Generator", "SOL", 50.0, 0.0, 0.0),
                              ("Link", "H2_turbine_X", 80.0, 0.0, 0.0)])
    out = attach_settlement(assets, settlement).set_index("asset_id")
    assert out.loc["OFF::b2031", "market_revenue_gbp"] == pytest.approx(250.0)
    assert out.loc["OFF::b2045", "market_revenue_gbp"] == pytest.approx(750.0)
    assert out["market_revenue_gbp"].sum() == pytest.approx(1130.0)          # once, not once per build year
    assert out["support_revenue_gbp"].sum() == pytest.approx(400.0)
    assert out.loc["H2_turbine_X", "settlement_status"] == "settled"


def test_a_site_split_then_build_year_split_keeps_both_shares():
    # Dinorwig-style unit: 60% legacy site cohort (itself two build years), 40% expansion.
    assets = pd.DataFrame([
        {"component_type": "StorageUnit", "asset_id": "PHS::legacy::b1984", "cohort_share": 0.5,
         "site_cohort_share": 0.6, "cohort_legacy_share": 1.0},
        {"component_type": "StorageUnit", "asset_id": "PHS::legacy::b1990", "cohort_share": 0.5,
         "site_cohort_share": 0.6, "cohort_legacy_share": 1.0},
        {"component_type": "StorageUnit", "asset_id": "PHS::new_build", "cohort_share": 1.0,
         "site_cohort_share": 0.4, "cohort_legacy_share": 0.0},
    ])
    out = attach_settlement(assets, _settlement([("StorageUnit", "PHS", 100.0, 0.0, 0.0)]))
    assert out["market_revenue_gbp"].tolist() == pytest.approx([30.0, 30.0, 40.0])


def test_shares_adding_to_more_than_one_are_refused():
    assets = pd.DataFrame([{"component_type": "Generator", "asset_id": f"OFF::b{y}", "cohort_share": 0.8}
                           for y in (2031, 2040)])
    with pytest.raises(ValueError, match="double counted"):
        vintaged_settlement_share(assets)


def test_a_2030_register_without_cohort_share_splits_as_before():
    assets = pd.DataFrame([
        {"component_type": "Generator", "asset_id": "farm::legacy", "cohort_legacy_share": 0.25},
        {"component_type": "Generator", "asset_id": "farm::new_build", "cohort_legacy_share": 0.25},
        {"component_type": "Generator", "asset_id": "dated", "cohort_legacy_share": 0.0},
    ])
    out = attach_settlement(assets, _settlement([("Generator", "farm", 100.0, 0.0, 0.0),
                                                 ("Generator", "dated", 7.0, 0.0, 0.0)]))
    assert out["market_revenue_gbp"].tolist() == pytest.approx([25.0, 75.0, 7.0])
    assert "cohort_share" not in out.columns


def test_the_vintaged_register_writes_the_site_share_the_settlement_reads():
    from scripts.public_ownership.build_asset_finance_register import site_cohort_shares

    register = pd.DataFrame([
        {"vintage_method": "existing_storage_site", "cohort_legacy_share": 0.6},
        {"vintage_method": "existing_storage_site_expansion", "cohort_legacy_share": 0.6},
        {"vintage_method": "asset_date", "cohort_legacy_share": 1.0},
    ])
    assert site_cohort_shares(register).tolist() == pytest.approx([0.6, 0.4, 1.0])


# ── 2. insurance on the register's basis ─────────────────────────────────────


def test_strikes_use_the_register_insurance_rule():
    from scripts.market import cfd_strikes_2050 as s
    from scripts.public_ownership.calculate_financing_scenarios import VINTAGE_INSURANCE_RULE

    assert s.INSURANCE_RULE == VINTAGE_INSURANCE_RULE == "exclude"
    block = s.finance_insurance_config()
    assert block["enabled"] and block["share_of_capex_per_year"] == 0.004
    assert s.finance_insurance_gbp_per_kw_year("Offshore wind", 2460.0, block) == 11.22
    assert s.finance_insurance_gbp_per_kw_year("Onshore wind", 1600.0, block) == pytest.approx(6.4)
    assert s.finance_insurance_gbp_per_kw_year("Large-scale nuclear", 9000.0, block) == 0.0
    assert s.finance_insurance_gbp_per_kw_year("Solar PV", 500.0, {"enabled": False}) == 0.0


def test_include_with_the_d4a_line_is_refused_as_a_double_count():
    from scripts.market import cfd_strikes_2050 as s
    from scripts.public_ownership import vintage_costs as vc

    cost = vc.costs_for("Onshore wind", 2040)
    with pytest.raises(ValueError, match="twice"):
        s.insured_fixed_cost(cost, 1600.0, insurance="include", finance_insurance=True)
    kept = s.insured_fixed_cost(cost, 1600.0, insurance="include")
    assert kept["finance_insurance_gbp_per_kw_year"] == 0.0
    assert kept["fixed_total_gbp_per_kw_year"] == pytest.approx(cost["fixed_om_gbp_per_kw_year"])


@needs_snapshot
def test_the_strike_fixed_cost_equals_the_register_fixed_cost_per_kw():
    """At cost, a strike's fixed O&M plus insurance per kW is what the vintaged register books for that row."""
    from scripts.market import cfd_strikes_2050 as s
    from scripts.public_ownership.calculate_financing_scenarios import (
        calculate_financing_scenarios, vintage_cost_registry,
    )
    from scripts.public_ownership.levers import expand_cases, parse_scenarios, parse_sensitivities

    snapshot = s.load_snapshot()
    rows = [("OFF", "wind_offshore", "Offshore wind", 2040), ("ONS", "wind_onshore", "Onshore wind", 2040),
            ("SOL", "solar_pv", "Solar PV", 2045), ("CCS", "CCS Gas", "Gas generation - CCUS - mature", 2040)]
    frame = pd.DataFrame([{"asset_id": a, "carrier": c, "finance_technology": t, "build_year": y, "capacity_mw": 100.0}
                          for a, c, t, y in rows])
    frame["model_year"], frame["component_type"], frame["finance_vintage"] = 2050, "Generator", "new_build"
    frame["commissioning_year"] = frame["build_year"].astype(float)
    frame["dispatch_mwh"], frame["evidence_status"] = 300_000.0, "resolved"
    bundles = parse_scenarios({"private": {}})
    specs = expand_cases(bundles, parse_sensitivities({"bounding_pair": False}, bundles))
    register, _ = calculate_financing_scenarios(
        frame, pd.read_csv(ROOT / "data" / "public_ownership" / "financing" / "technology_costs.csv"), snapshot,
        price_year=2024, state_operating_overhead={"enabled": False},
        generator_costs={"insurance": s.finance_insurance_config()}, vintage_costs=vintage_cost_registry(),
        case_specs=specs)
    private = register.loc[register["case_id"].eq("private")].set_index("asset_id")
    for asset_id, _, technology, year in rows[:3]:
        strike = s.levelised_strike(technology, year, snapshot=snapshot)
        per_kw = strike["fixed_om_gbp_per_kw_year"] + strike["finance_insurance_gbp_per_kw_year"]
        assert private.loc[asset_id, "annual_fixed_om_gbp"] / 1e5 == pytest.approx(per_kw), asset_id
        assert strike["insurance_rule"] == "vintage_insurance_exclude_plus_d4a"
    from scripts.public_ownership import support_regime_2050 as sr
    from scripts.public_ownership import vintage_costs as vc
    from scripts.public_ownership.finance import capital_recovery_factor

    dpa = sr.dpa_contract("CCS Gas", 2040, 100.0, snapshot=snapshot)
    cost = vc.costs_for("Gas generation - CCUS - mature", 2040)
    rate = s.private_rate("Gas generation - CCUS - mature", 2040, snapshot=snapshot)["rate"]
    fixed = private.loc["CCS", "annual_fixed_om_gbp"] / 1e5
    expected = cost["capex_power_gbp_per_kw"] * capital_recovery_factor(rate, cost["economic_life_years"]) + fixed + 4.4
    assert dpa.fixed_cost_gbp_per_kw_year == pytest.approx(expected)
    assert "plus_d4a" in dpa.source


# ── 3. hydrogen-turbine links ────────────────────────────────────────────────


def _network(tmp_path):
    import pypsa

    n = pypsa.Network()
    index = pd.date_range("2050-01-01", periods=4, freq="h")
    n.set_snapshots(index)
    n.add("Bus", "AC1", carrier="AC")
    n.add("Bus", "GB_H2", carrier="H2_gas")
    n.add("Bus", "GAS", carrier="natural_gas")
    n.add("Generator", "CCS1", bus="AC1", carrier="CCS Gas", p_nom=500.0)
    n.add("Link", "H2_turbine_A", bus0="GB_H2", bus1="AC1", carrier="H2_turbine", efficiency=0.5, p_nom=400.0)
    n.add("Link", "H2_turbine_B", bus0="GB_H2", bus1="AC1", carrier="H2_turbine", efficiency=0.5, p_nom=200.0)
    n.add("Link", "electrolysis_AC1", bus0="AC1", bus1="GB_H2", carrier="electrolysis", efficiency=0.8, p_nom=300.0)
    n.add("Link", "reformer", bus0="GAS", bus1="GB_H2", carrier="gas_reforming_ccs", efficiency=0.83, p_nom=100.0)
    n.generators_t.p = pd.DataFrame({"CCS1": [100.0, 200.0, 0.0, 50.0]}, index=index)
    p0 = pd.DataFrame({"H2_turbine_A": [400.0, 200.0, 0.0, 100.0], "H2_turbine_B": [0.0, 0.0, 200.0, 0.0],
                       "electrolysis_AC1": [300.0] * 4, "reformer": [100.0] * 4}, index=index)
    n.links_t.p0 = p0
    n.links_t.p1 = -p0.mul(pd.Series({"H2_turbine_A": 0.5, "H2_turbine_B": 0.5, "electrolysis_AC1": 0.8,
                                       "reformer": 0.83}), axis=1)
    path = tmp_path / "ht50_month.nc"
    n.export_to_netcdf(path)
    return path


def _crosswalk_base():
    return pd.read_csv(CROSSWALK)


def _mix(rows):
    return pd.DataFrame([{"technology": t, "build_year": b, "capacity_mw": c} for t, b, c in rows])


def test_hydrogen_turbine_links_join_a_vintaged_register_at_their_electrical_rating(tmp_path):
    from scripts.public_ownership.build_asset_finance_register import build_asset_finance_register

    path = _network(tmp_path)
    mix = _mix([("Hydrogen CCHT", 2040, 200.0), ("Hydrogen OCHT", 2045, 100.0),
                ("Gas generation - CCUS - mature", 2040, 500.0)])
    register = build_asset_finance_register([path], _crosswalk_base(), model_year=2050, fleet_vintage_mix=mix)
    links = register.loc[register["component_type"].eq("Link")]
    base = links["asset_id"].str.split("::").str[0]
    assert set(base) == {"H2_turbine_A", "H2_turbine_B"}          # electrolysis and reforming stay out
    # capacity: the electrical output rating (p_nom on the H2 side x efficiency)
    assert links.groupby(base)["capacity_mw"].sum().to_dict() == pytest.approx({"H2_turbine_A": 200.0,
                                                                                "H2_turbine_B": 100.0})
    # dispatch: the AC-side delivery, -p1
    assert links.groupby(base)["dispatch_mwh"].sum().to_dict() == pytest.approx({"H2_turbine_A": 0.5 * 700.0,
                                                                                 "H2_turbine_B": 0.5 * 200.0})
    assert links.groupby(base)["cohort_share"].sum().to_dict() == pytest.approx({"H2_turbine_A": 1.0,
                                                                                 "H2_turbine_B": 1.0})
    assert set(links["bus"]) == {"AC1"}
    by_tech = links.groupby("stock_flow_technology")["capacity_mw"].sum()
    assert by_tech.to_dict() == pytest.approx({"Hydrogen CCHT": 200.0, "Hydrogen OCHT": 100.0})
    assert set(links["finance_technology"]) == {"Hydrogen CCHT/OCHT - mature"}
    assert (links["cost_technology"] == links["stock_flow_technology"]).all()
    assert links["evidence_status"].eq("resolved").all()
    # Hydrogen turbines hold DPAs (live in 2050 from build year 2036).
    assert links["support_type"].eq("DPA").all()
    assert set(links["cfd_round"]) == {"DPA_2040", "DPA_2045"}
    ccs = register.set_index("asset_id").loc["CCS1"]
    assert ccs["finance_technology"] == "Gas generation - CCUS - mature" and ccs["evidence_status"] == "resolved"
    assert ccs["support_type"] == "DPA"


def test_a_2030_style_register_takes_no_link_and_no_supplement(tmp_path):
    from scripts.public_ownership.build_asset_finance_register import build_asset_finance_register

    register = build_asset_finance_register([_network(tmp_path)], _crosswalk_base(), model_year=2030)
    assert set(register["component_type"]) == {"Generator"}
    # F27 item 5 (23 Oct 2026): the base crosswalk carries cost_technology (OCGT, engines, NZT), so a
    # 2030 register does too; it still takes no 2050 supplement column.
    assert "site_cohort_share" not in register.columns and "crosswalk_source" not in register.columns
    # F27 item 1 (24 Oct 2026): the 2030 crosswalk costs CCS Gas (NZT) so it can join the pre-2025
    # option; the ownership rule, not the crosswalk, keeps it private in every other case
    ccs = register.set_index("asset_id").loc["CCS1"]
    assert ccs["evidence_status"] == "resolved" and ccs["finance_technology"] == "Gas generation - CCUS - mature"


def test_the_h2_turbine_carrier_pools_both_hydrogen_technologies():
    assert fv.carrier_technologies()["H2_turbine"] == ("Hydrogen CCHT", "Hydrogen OCHT")


# ── 4. the 2050 crosswalk ─────────────────────────────────────────────────────


def test_the_2050_crosswalk_names_every_new_carrier_with_provenance():
    table = pd.read_csv(CROSSWALK_2050)
    assert table[["source", "status", "notes"]].notna().all().all()
    keys = set(zip(table["component_type"], table["pypsa_carrier"]))
    for key in (("Generator", "CCS Gas"), ("Generator", "nuclear"), ("Link", "H2_turbine"), ("StorageUnit", "CAES"),
                ("Generator", "geothermal"), ("StorageUnit", "LAES")):
        assert key in keys, key
    smr = table.loc[table["stock_flow_technology"].eq("Small modular reactors (SMRs)")].iloc[0]
    assert smr["pypsa_carrier"] == "nuclear" and smr["workbook_technology"] == smr["cost_technology"]
    # every eligible row costs on a vintage-table row and rates on a snapshot row
    from scripts.public_ownership.vintage_costs import load_vintage_costs

    vintage = load_vintage_costs()
    eligible = table.loc[table["finance_eligible"].astype(str).str.lower().eq("true")]
    for _, row in eligible.iterrows():
        cost = vintage.loc[vintage["technology"].eq(row["cost_technology"])]
        assert len(cost), row["cost_technology"]
        assert set(cost["wacc_technology"]) == {row["workbook_technology"]}, row["cost_technology"]


def test_laes_has_one_life_in_2050():
    from scripts.public_ownership.vintage_costs import load_vintage_costs

    table = pd.read_csv(CROSSWALK_2050)

    def cost_of(carrier):
        return table.loc[table["pypsa_carrier"].eq(carrier), "cost_technology"].iloc[0]

    cost_row = cost_of("LAES")
    vintage = load_vintage_costs()
    register_life = float(vintage.loc[vintage["technology"].eq(cost_row), "economic_life_years"].iloc[0])
    assert register_life == fv.load_lives()["Liquid air energy storage"] == 25.0
    caes = cost_of("CAES")
    assert float(vintage.loc[vintage["technology"].eq(caes), "economic_life_years"].iloc[0]) == \
        fv.load_lives()["Compressed air energy storage"] == 25.0
    # the 2030 crosswalk is untouched: LAES keeps the 20-year LDES proxy there
    base = pd.read_csv(CROSSWALK).set_index(["component_type", "pypsa_carrier"])
    assert base.loc[("StorageUnit", "LAES"), "workbook_technology"] == "Novel long duration energy storage (LDES)"


def test_supplement_rows_key_on_the_stock_flow_technology_first():
    from scripts.public_ownership.build_asset_finance_register import apply_crosswalk_supplement

    register = pd.DataFrame([
        {"component_type": "Generator", "carrier": "nuclear", "stock_flow_technology": "Small modular reactors (SMRs)",
         "finance_technology": "Large-scale nuclear", "finance_eligible": True},
        {"component_type": "Generator", "carrier": "nuclear", "stock_flow_technology": "Large-scale nuclear",
         "finance_technology": "Large-scale nuclear", "finance_eligible": True},
        {"component_type": "Generator", "carrier": "geothermal", "stock_flow_technology": "Geothermal",
         "finance_technology": np.nan, "finance_eligible": np.nan},
        {"component_type": "Generator", "carrier": "solar_pv", "stock_flow_technology": "Solar PV",
         "finance_technology": "Solar PV", "finance_eligible": True},
    ])
    out = apply_crosswalk_supplement(register, pd.read_csv(CROSSWALK_2050))
    assert out["finance_technology"].tolist()[:2] == ["Small modular reactors (SMRs)", "Large-scale nuclear"]
    # User, 3 Oct 2026: geothermal is costed (DESNZ Annex A Deep Granite Geothermal) and finance-eligible.
    assert out.loc[2, "finance_technology"] == "Deep geothermal" and bool(out.loc[2, "finance_eligible"])
    assert out["crosswalk_source"].tolist() == ["2050_supplement"] * 3 + ["base"]
    assert out.loc[3, "cost_technology"] == "Solar PV"
