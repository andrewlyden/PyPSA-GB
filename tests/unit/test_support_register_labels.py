"""Support labels from the scheme registers (Ofgem RO, LCCC CfD) for named sites.

The review of the final CP30 2030 build (26 Sep 2026) found Race Bank, Rampion,
Galloper and EOWDC labelled merchant (RO grace-period accreditations the date
rule cannot see), solar over-labelled ROC (the date rule makes Feed-in Tariff
farms ROC), and Triton Knoll merchant despite its AR2 CfD (REPD gives the round
but not the contracted capacity).
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.generators.integrate_renewable_generators import _assign_subsidy_attributes_by_origin
from scripts.renewables.support_register_labels import (
    apply_lccc_cfd_evidence,
    apply_ro_register_labels,
    apply_ro_register_share_to_proxies,
    default_banding_for_accreditation,
    distinctive_overlap,
    lccc_round_label,
    load_crosswalk_overrides,
    load_lccc_portfolio,
    load_ro_register,
    markers_compatible,
    match_units_to_sites,
    normalise_station_name,
    phase_free_key,
    phase_marker,
    register_proxy_carrier,
    register_reconciliation,
)
from config.config_loader import load_defaults, load_layer

LOG = logging.getLogger("test_support_register_labels")
REGISTER_COLUMNS = ["Accreditation reference", "Station name", "Organisation name", "Station DNC in kW",
                    "Technology", "Commissioning date", "Effective date", "Country", "Contract type",
                    "Organisation address", "Station address"]
BANDINGS = {"wind_onshore": 0.9, "wind_offshore": 2.0, "solar_pv": 1.2}
FRACTIONS = {"wind_offshore": 1.0, "wind_onshore": 0.3, "solar_pv": 0.5}


def _register(tmp_path, rows):
    frame = pd.DataFrame([{
        "Accreditation reference": ref, "Station name": name, "Organisation name": "x",
        "Station DNC in kW": f"{kw:,.0f}", "Technology": technology, "Commissioning date": date,
        "Effective date": date, "Country": country, "Contract type": "General",
        "Organisation address": "x", "Station address": address,
    } for ref, name, kw, technology, date, country, address in rows], columns=REGISTER_COLUMNS)
    path = tmp_path / "register.csv"
    frame.to_csv(path, index=False)
    return path


def _certificates(tmp_path, rows):
    path = tmp_path / "certificates.csv"
    pd.DataFrame(rows, columns=["Accreditation reference", "Scheme", "Compliance Period",
                                "MWh per certificate"]).to_csv(path, index=False)
    return path


def _site(name, technology, capacity, operational_date, *, status="Operational", origin="named",
          ref=None, cfd_round=None, cfd_capacity=np.nan, ro_banding=np.nan):
    return {"site_name": name, "source_site_name": name, "technology": technology,
            "capacity_mw": capacity, "operational_date": operational_date, "development_status": status,
            "fleet_origin": origin, "repd_ref_id": ref or name, "cfd_round": cfd_round,
            "cfd_capacity_mw": cfd_capacity, "ro_banding": ro_banding, "lat": 53.0, "lon": 0.5}


# --------------------------------------------------------------------------- names

def test_names_normalise_unicode_spaces_punctuation_and_boilerplate():
    assert normalise_station_name("Triton Knoll\xa0") == "triton knoll"
    assert phase_free_key("Triton Knoll Offshore Wind Farm Phase 1") == "triton knoll"
    assert phase_free_key("Race Bank (Phase 2)") == phase_free_key("Race Bank")
    assert normalise_station_name("Owl's Hatch Solar Park") == normalise_station_name("Owls Hatch")
    assert normalise_station_name("Blyth Offshore\xa0Wind Test Site") == "blyth test"


def test_phase_markers_and_project_numbers():
    assert phase_marker("Walney\xa01") == phase_marker("Walney Offshore Wind Phase I") == "1"
    assert phase_marker("Kentish Flats Extension Wind Farm") == "ext"
    assert phase_marker("Rothes2 Wind Farm") == "2"
    assert markers_compatible("ext", "2") and markers_compatible("", "1")
    assert not markers_compatible("1", "2")
    # Hornsea Project 2 is not Hornsea 3, though "Hornsea" alone may be either.
    assert not distinctive_overlap(phase_free_key("Hornsea Project 2 Phase 3"), phase_free_key("Hornsea 3"))
    assert distinctive_overlap(phase_free_key("Hornsea Offshore Wind Farm Phase 1"),
                               phase_free_key("Hornsea 1 - Heron & Njord"))
    # Generic words alone never identify a site.
    assert not distinctive_overlap("manor farm", "manor farm")


def test_lccc_round_labels():
    assert lccc_round_label("Allocation Round 2") == "AR2"
    assert lccc_round_label("Investment Contract") == "Investment Contract"


# --------------------------------------------------------------------------- matching

def test_one_accreditation_spans_two_repd_phases_in_proportion():
    units = pd.DataFrame([{"unit_id": "R1", "unit_name": "Race Bank", "carrier": "wind_offshore",
                           "unit_capacity_mw": 565.0, "unit_year": 2017}])
    sites = pd.DataFrame([
        {"name": "Race Bank (Phase 1)", "carrier": "wind_offshore", "capacity_mw": 289.8,
         "repd_ref_id": "2515", "operating_year": 2017},
        {"name": "Race Bank (Phase 2)", "carrier": "wind_offshore", "capacity_mw": 283.5,
         "repd_ref_id": "6501", "operating_year": 2018},
    ], index=["a", "b"])
    mapping = match_units_to_sites(units, sites)
    assert set(mapping["site"]) == {"a", "b"}
    assert mapping["allocated_mw"].sum() == pytest.approx(565.0)
    assert mapping.set_index("site").at["a", "allocated_mw"] == pytest.approx(565.0 * 289.8 / 573.3)


def test_timing_rules_out_an_older_namesake():
    # An LCCC AR5 contract is not a wind farm operating since 2012, and an
    # accreditation from 2016 is not a site built in 2004.
    units = pd.DataFrame([
        {"unit_id": "C1", "unit_name": "Pearie Law Wind Farm", "carrier": "wind_onshore",
         "unit_capacity_mw": 20.0, "award_year": 2023},
        {"unit_id": "R9", "unit_name": "Black Law Extension", "carrier": "wind_onshore",
         "unit_capacity_mw": 60.0, "unit_year": 2016},
    ])
    sites = pd.DataFrame([
        {"name": "Pearie Law Wind Farm", "carrier": "wind_onshore", "capacity_mw": 19.2,
         "repd_ref_id": "1", "operating_year": 2012},
        {"name": "Black Law", "carrier": "wind_onshore", "capacity_mw": 62.0,
         "repd_ref_id": "2", "operating_year": 2004},
    ])
    assert match_units_to_sites(units, sites).empty


def test_postcode_and_sourced_override_matches(tmp_path):
    units = pd.DataFrame([
        {"unit_id": "R2", "unit_name": "Orta Port Farms Solar", "carrier": "solar_pv",
         "unit_capacity_mw": 28.6, "unit_year": 2015, "postcode": "SN66JT"},
        {"unit_id": "R3", "unit_name": "PyC Wind Farm", "carrier": "wind_onshore",
         "unit_capacity_mw": 221.7, "unit_year": 2016},
    ])
    sites = pd.DataFrame([
        {"name": "Water Eaton Farm / Port Farm", "carrier": "solar_pv", "capacity_mw": 34.8,
         "repd_ref_id": "2085", "operating_year": 2015, "site_postcode": "SN66JT"},
        {"name": "Pen Y Cymoedd", "carrier": "wind_onshore", "capacity_mw": 228.0,
         "repd_ref_id": "4595", "operating_year": 2017},
    ], index=["solar", "pyc"])
    overrides_path = tmp_path / "overrides.csv"
    pd.DataFrame([{"register": "ofgem_ro", "unit_id": "R3", "repd_ref_id": "4595", "action": "match",
                   "unit_name": "PyC", "site_name": "Pen Y Cymoedd", "evidence": "abbreviation"}]
                 ).to_csv(overrides_path, index=False)
    mapping = match_units_to_sites(units, sites, load_crosswalk_overrides(overrides_path))
    assert mapping.set_index("unit_id")["site"].to_dict() == {"R2": "solar", "R3": "pyc"}
    assert mapping.set_index("unit_id").at["R3", "method"] == "explicit_override"


def test_override_file_needs_evidence(tmp_path):
    path = tmp_path / "overrides.csv"
    pd.DataFrame([{"register": "ofgem_ro", "unit_id": "R3", "repd_ref_id": "4595", "action": "match",
                   "evidence": ""}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="evidence"):
        load_crosswalk_overrides(path)


# --------------------------------------------------------------------------- RO labels

@pytest.fixture
def ro_case(tmp_path):
    register = _register(tmp_path, [
        ("R00032RPEN", "Race Bank", 565_000, "Offshore wind", "08/06/2017", "England", "Wisbech, PE14 7JE"),
        ("R00034RPEN", "Rampion", 397_416, "Offshore wind", "26/11/2017", "England", "Newhaven, BN9 0BN"),
        ("R00001SQSC", "Old Glen Wind Farm", 30_000, "Onshore wind", "01/06/2005", "Scotland", "IV1 1AA"),
        ("R00002SQSC", "Big Moor Wind Farm", 50_000, "Onshore wind", "01/06/2014", "Scotland", "IV2 2BB"),
        ("R00003SQSC", "Contract Hill Wind Farm", 40_000, "Onshore wind", "01/06/2014", "Scotland", "IV3 3CC"),
        ("R00004NGNI", "Ulster Wind Farm", 20_000, "Onshore wind", "01/06/2014", "Northern Ireland", "BT1 1AA"),
    ])
    certificates = _certificates(tmp_path, [
        ("R00032RPEN", "RO", "2024/25", 1 / 1.8),
        ("R00002SQSC", "ROS", "2024/25", 1 / 0.9),
    ])
    sites = pd.DataFrame([
        _site("Race Bank (Phase 1)", "wind_offshore", 289.8, "08/06/2017"),
        _site("Race Bank (Phase 2)", "wind_offshore", 283.5, "01/02/2018"),
        _site("Rampion Offshore Wind Farm (Hastings Zone)", "wind_offshore", 400.0, "26/11/2017"),
        _site("Old Glen Wind Farm", "wind_onshore", 30.0, "01/06/2005"),
        _site("Big Moor Wind Farm", "wind_onshore", 100.0, "01/06/2014"),
        _site("Contract Hill Wind Farm", "wind_onshore", 40.0, "01/06/2014", cfd_round="AR1",
              cfd_capacity=40.0),
        _site("Sunny Meadow Solar Farm", "solar_pv", 5.0, "01/06/2014"),
        _site("CP30_wind_onshore_1", "wind_onshore", 200.0, None, origin="increment", status=None),
    ])
    assigned = _assign_subsidy_attributes_by_origin(sites, 2030, 20, FRACTIONS, LOG, "2017-04-01", BANDINGS)
    frame = load_ro_register(register, "2030-04-01", certificates, BANDINGS)
    return assigned, frame


def test_date_rule_misses_grace_period_and_over_labels_fit_solar(ro_case):
    assigned, _ = ro_case
    labels = assigned.set_index("site_name")["support_type"]
    assert labels["Race Bank (Phase 1)"] == "merchant"   # accredited after 1 April 2017
    assert labels["Sunny Meadow Solar Farm"] == "ROC"    # date rule, no accreditation


def test_register_relabels_named_sites(ro_case):
    assigned, register = ro_case
    result, mapping, audit = apply_ro_register_labels(assigned, register, logger=LOG)
    by_name = result.groupby(["site_name", "support_type"])["capacity_mw"].sum()
    # Grace-period offshore accreditations: ROC with the observed or 2016/17 band.
    rows = result.set_index("site_name")
    assert rows.at["Race Bank (Phase 1)", "support_type"] == "ROC"
    assert rows.at["Race Bank (Phase 1)", "ro_banding"] == pytest.approx(1.8)
    assert rows.at["Rampion Offshore Wind Farm (Hastings Zone)", "support_type"] == "ROC"
    assert rows.at["Rampion Offshore Wind Farm (Hastings Zone)", "ro_banding"] == pytest.approx(1.8)
    # Expired accreditation (2005 + 20 years) and no accreditation: not ROC.
    assert rows.at["Old Glen Wind Farm", "support_type"] == "merchant"
    assert rows.at["Sunny Meadow Solar Farm", "support_type"] == "merchant"
    assert rows.at["Sunny Meadow Solar Farm", "ro_support_source"] == "ofgem_ro_register_no_accreditation"
    # Half accredited: split, both rows keep the site name (and so its profile).
    assert by_name[("Big Moor Wind Farm", "ROC")] == pytest.approx(50.0)
    assert by_name[("Big Moor Wind Farm", "merchant")] == pytest.approx(50.0)
    roc_row = result[(result.site_name == "Big Moor Wind Farm") & (result.support_type == "ROC")].iloc[0]
    assert roc_row["ro_banding"] == pytest.approx(0.9)
    assert roc_row["ro_accreditation_reference"] == "R00002SQSC"
    # CfD precedence, and proxy rows untouched.
    assert rows.at["Contract Hill Wind Farm", "support_type"] == "CfD"
    assert audit.set_index("site_name").at["Contract Hill Wind Farm", "note"].startswith("CfD precedence")
    proxy = result[result.fleet_origin.eq("increment")]
    assert proxy["capacity_mw"].sum() == pytest.approx(200.0)
    assert set(proxy["support_type"]) <= {"CfD", "merchant"}
    # Capacity is conserved.
    assert result["capacity_mw"].sum() == pytest.approx(assigned["capacity_mw"].sum())
    # Northern Ireland stations are excluded from a GB network's labels and
    # reported separately in the reconciliation.
    recon = register_reconciliation(register, mapping, result).set_index("carrier")
    assert recon.at["wind_onshore", "register_eligible_ni_mw"] == pytest.approx(20.0)
    assert recon.at["wind_offshore", "unmatched_gb_mw"] == pytest.approx(0.0)


def test_default_banding_follows_accreditation_year():
    assert default_banding_for_accreditation("wind_offshore", pd.Timestamp("2014-06-01"), BANDINGS) == 2.0
    assert default_banding_for_accreditation("wind_offshore", pd.Timestamp("2015-06-01"), BANDINGS) == 1.9
    assert default_banding_for_accreditation("wind_offshore", pd.Timestamp("2017-06-08"), BANDINGS) == 1.8
    assert default_banding_for_accreditation("wind_onshore", pd.Timestamp("2014-06-01"), BANDINGS) == 0.9
    assert default_banding_for_accreditation("solar_pv", pd.Timestamp("2014-06-01"), BANDINGS) == 1.2


def test_register_eligibility_matches_the_bill_rule(tmp_path):
    # The network label is the single-date rule (period=None in load_ro_register,
    # restored for the 2030 re-solve, OW12); the calendar-2030 share is the bill's.
    register = _register(tmp_path, [
        ("A", "Early", 10_000, "Onshore wind", "15/03/2010", "Scotland", ""),
        ("B", "Later", 10_000, "Onshore wind", "15/04/2010", "Scotland", ""),
    ])
    frame = load_ro_register(register, "2030-04-01").set_index("unit_id")
    # 20 years from the effective date must still run on 1 April 2030.
    assert not frame.at["A", "ro_eligible"]
    assert frame.at["B", "ro_eligible"]


# --------------------------------------------------------------------------- CfD evidence

@pytest.fixture
def lccc_portfolio(tmp_path):
    path = tmp_path / "lccc.csv"
    pd.DataFrame([
        ("AR2-TKN-103", "Triton Knoll Offshore Wind Farm Phase 1", "Allocation Round 2", "Offshore Wind",
         "Live (Post-FIC)", 216.6),
        ("AR2-TKN-203", "Triton Knoll Offshore Wind Farm Phase 2", "Allocation Round 2", "Offshore Wind",
         "Live (Post-FIC)", 207.2),
        ("AR2-TKN-303", "Triton Knoll Offshore Wind Farm Phase 3", "Allocation Round 2", "Offshore Wind",
         "Live (Post-FIC)", 423.7),
        ("AR4-EST-700", "Eastfields", "Allocation Round 4", "Solar PV", "Live (Post-FIC)", 25.0),
        ("AR6-XYZ-700", "Kitland Solar Farm", "Allocation Round 6", "Solar PV", "Pre-MDD", 20.0),
        ("AR4-DEAD-700", "Gone Solar", "Allocation Round 4", "Solar PV", "Terminated", 30.0),
    ], columns=["CFD_ID", "Name_of_CFD_Unit", "Allocation_Round", "Technology_Type", "Status",
                "Maximum_Contract_Capacity_MW"]).to_csv(path, index=False)
    return load_lccc_portfolio(path)


def test_lccc_confirms_rounds_repd_leaves_without_capacity(lccc_portfolio):
    sites = pd.DataFrame([
        _site("Triton Knoll\xa0", "wind_offshore", 857.0, "13/01/2022", cfd_round="AR2"),
        _site("Eastfields Farm, Deppers Bridge - Solar Farm", "solar_pv", 34.8, None,
              status="Under Construction", cfd_round="AR4"),
        _site("Docking Farm Solar", "solar_pv", 28.0, None, status="Awaiting Construction", cfd_round="AR4"),
        _site("Kitland Solar Farm", "solar_pv", 19.95, None, status="Awaiting Construction"),
        _site("Gone Solar", "solar_pv", 30.0, None, status="Awaiting Construction"),
    ])
    assert "Gone Solar" not in set(lccc_portfolio["unit_name"])   # terminated contracts drop out
    before = _assign_subsidy_attributes_by_origin(sites, 2030, 20, FRACTIONS, LOG, "2017-04-01", BANDINGS)
    assert before.set_index("site_name").at["Triton Knoll\xa0", "support_type"] == "merchant"

    evidenced, mapping, audit = apply_lccc_cfd_evidence(sites, lccc_portfolio, logger=LOG)
    rows = evidenced.set_index("site_name")
    assert rows.at["Triton Knoll\xa0", "cfd_capacity_mw"] == pytest.approx(847.5)
    assert rows.at["Triton Knoll\xa0", "cfd_round"] == "AR2"
    assert rows.at["Kitland Solar Farm", "cfd_round"] == "AR6"
    # Round without capacity and no contract: withdrawn, kept for audit.
    assert pd.isna(rows.at["Docking Farm Solar", "cfd_round"])
    assert rows.at["Docking Farm Solar", "repd_cfd_round_unconfirmed"] == "AR4"

    after = _assign_subsidy_attributes_by_origin(evidenced, 2030, 20, FRACTIONS, LOG, "2017-04-01", BANDINGS)
    support = after.groupby("source_site_name")["support_type"].agg(lambda s: set(s))
    assert support["Triton Knoll\xa0"] == {"CfD"}
    assert support["Eastfields Farm, Deppers Bridge - Solar Farm"] == {"CfD"}
    assert support["Kitland Solar Farm"] == {"CfD"}
    # Docking now takes the future-auction split like any unlabelled project.
    assert support["Docking Farm Solar"] == {"CfD", "merchant"}


def test_lccc_never_overwrites_a_repd_round_with_capacity(lccc_portfolio):
    sites = pd.DataFrame([_site("Kitland Solar Farm", "solar_pv", 19.95, None, status="Awaiting Construction",
                                cfd_round="AR5", cfd_capacity=19.95)])
    evidenced, _, audit = apply_lccc_cfd_evidence(sites, lccc_portfolio, logger=LOG)
    assert evidenced["cfd_round"].iloc[0] == "AR5"
    assert audit["note"].iloc[0] == "REPD round kept; LCCC round differs"


# --------------------------------------------------------------------------- dispatchable proxies

def test_biomass_proxies_take_the_active_register_share(tmp_path):
    register = _register(tmp_path, [
        ("B1", "Brigg REP", 40_000, "Fuelled", "20/10/2015", "England", ""),
        ("B2", "Sheffield ERF", 20_000, "Fuelled", "29/08/2013", "England", ""),
        ("B3", "Idle Biomass", 50_000, "Fuelled", "01/01/2014", "England", ""),       # no recent ROCs
        ("B4", "Ulster Biomass", 10_000, "Fuelled", "01/01/2014", "Northern Ireland", ""),
        ("B5", "Old Landfill", 5_000, "Landfill gas", "01/01/2004", "England", ""),   # expired
    ])
    certificates = _certificates(tmp_path, [
        ("B1", "RO", "2024/25", 1 / 1.5), ("B2", "RO", "2024/25", 1.0),
        ("B3", "RO", "2019/20", 1 / 2.0), ("B4", "NIRO", "2024/25", 1.0),
    ])
    frame = load_ro_register(register, "2030-04-01", certificates, BANDINGS)
    proxies = pd.DataFrame([
        {"station_name": "CP30_biomass_0", "fuel_type": "biomass", "capacity_mw": 60.0, "support_type": "merchant"},
        {"station_name": "CP30_biomass_1", "fuel_type": "biomass", "capacity_mw": 40.0, "support_type": "merchant"},
        {"station_name": "CP30_waste_to_energy_0", "fuel_type": "waste_to_energy", "capacity_mw": 10.0,
         "support_type": "merchant"},
    ])
    result, audit = apply_ro_register_share_to_proxies(proxies, frame, logger=LOG)
    audit = audit.set_index("carrier")
    assert register_proxy_carrier("Sheffield ERF") == "waste_to_energy"
    assert register_proxy_carrier("Brigg REP") == "biomass"
    assert audit.at["biomass", "register_active_mw"] == pytest.approx(40.0)
    assert audit.at["biomass", "roc_share"] == pytest.approx(0.4)
    assert audit.at["waste_to_energy", "roc_share"] == pytest.approx(1.0)
    roc = result[result.support_type.eq("ROC")]
    assert roc.loc[roc.fuel_type.eq("biomass"), "capacity_mw"].sum() == pytest.approx(40.0)
    assert set(roc.loc[roc.fuel_type.eq("biomass"), "ro_banding"]) == {1.5}
    assert "CP30_biomass_0_ROC" in set(result["station_name"])
    assert result["capacity_mw"].sum() == pytest.approx(110.0)


def test_marine_proxies_take_the_register_share_without_activity_rule(tmp_path):
    register = _register(tmp_path, [
        ("M1", "Ness of Quoys", 6_000, "Tidal stream", "08/11/2016", "Scotland", ""),
        ("M2", "Test Berth", 2_000, "Wave power", "30/01/2013", "Scotland", ""),     # no certificates
    ])
    certificates = _certificates(tmp_path, [("M1", "ROS", "2024/25", 0.2)])
    frame = load_ro_register(register, "2030-04-01", certificates, BANDINGS)
    proxies = pd.DataFrame([
        {"site_name": "CP30_marine_0", "technology": "marine", "capacity_mw": 80.0,
         "support_type": "merchant", "fleet_origin": "increment"},
        {"site_name": "Named Tidal", "technology": "marine", "capacity_mw": 5.0,
         "support_type": "merchant", "fleet_origin": "named"},
    ])
    result, audit = apply_ro_register_share_to_proxies(
        proxies, frame, carriers=("marine",), min_certificate_year=None, carrier_col="technology",
        proxy_mask=proxies["fleet_origin"].ne("named"), register_carrier="marine",
        classify=lambda name: "marine", logger=LOG)
    assert audit.set_index("carrier").at["marine", "register_active_mw"] == pytest.approx(8.0)
    roc = result[result.support_type.eq("ROC")]
    assert roc["capacity_mw"].sum() == pytest.approx(8.0)
    assert roc["site_name"].tolist() == ["CP30_marine_0_ROC"]
    assert roc["ro_banding"].iloc[0] == pytest.approx(5.0)   # banding from the stations that issue
    assert result.loc[result.site_name.eq("Named Tidal"), "support_type"].tolist() == ["merchant"]
    # Item C4: with remainder_support the rest of each proxy is CfD tidal stream,
    # capacity conserved and named rows untouched.
    cfd, audit = apply_ro_register_share_to_proxies(
        proxies, frame, carriers=("marine",), min_certificate_year=None, carrier_col="technology",
        proxy_mask=proxies["fleet_origin"].ne("named"), register_carrier="marine",
        classify=lambda name: "marine", remainder_support="CfD", logger=LOG)
    by_support = cfd.groupby("support_type")["capacity_mw"].sum()
    assert by_support["ROC"] == pytest.approx(8.0) and by_support["CfD"] == pytest.approx(72.0)
    assert by_support["merchant"] == pytest.approx(5.0)   # the named row
    assert audit.set_index("carrier").at["marine", "remainder_mw"] == pytest.approx(72.0)
    # No register share at all: the whole proxy takes the remainder label.
    empty = frame.iloc[0:0]
    whole, _ = apply_ro_register_share_to_proxies(
        proxies, empty, carriers=("marine",), min_certificate_year=None, carrier_col="technology",
        proxy_mask=proxies["fleet_origin"].ne("named"), register_carrier="marine",
        classify=lambda name: "marine", remainder_support="CfD", logger=LOG)
    assert whole.set_index("site_name").at["CP30_marine_0", "support_type"] == "CfD"


def test_the_marine_proxy_remainder_is_cfd_in_the_defaults():
    import yaml
    share = load_defaults()["support"]["labels"]["support_registers"]["renewable_proxy_share"]
    assert share["carriers"] == ["marine"] and share["remainder_support"] == "CfD"


# --------------------------------------------------------------------------- workflow wiring

def _rule_block(text, rule):
    start = text.index(f"rule {rule}:")
    following = re.search(r"\n(rule |def )", text[start + 1:])
    return text[start:start + 1 + following.start()] if following else text[start:]


def test_rules_declare_the_register_files():
    text = Path("rules/generators.smk").read_text(encoding="utf-8")
    renewables = _rule_block(text, "add_renewables_to_network")
    for name in ("ro_register", "ro_certificates", "lccc_cfd_portfolio", "support_repd",
                 "support_crosswalk_overrides", "support_register_logic"):
        assert f"{name}=lambda wildcards: _support_register_input" in renewables, name
    thermal = _rule_block(text, "add_thermal_generators_to_network")
    for name in ("ro_register", "ro_certificates", "support_register_logic"):
        assert f"{name}=lambda wildcards: _support_register_input" in thermal, name


def test_the_defaults_point_at_existing_register_files():
    registers = load_defaults()["support"]["labels"]["support_registers"]
    assert registers["enabled"] is True
    for key in ("ro_register", "ro_certificates", "lccc_portfolio", "repd", "crosswalk_overrides"):
        assert Path(registers[key]).exists(), registers[key]
    overrides = load_crosswalk_overrides(registers["crosswalk_overrides"])
    assert set(overrides["register"]) <= {"ofgem_ro", "lccc_cfd"}


def test_real_register_holds_the_grace_period_offshore_farms():
    registers = load_defaults()["support"]["labels"]["support_registers"]
    frame = load_ro_register(registers["ro_register"], "2030-04-01", default_bandings=BANDINGS)
    offshore = frame[frame.carrier.eq("wind_offshore") & frame.ro_eligible].set_index("Station name")
    for name in ("Race Bank", "Rampion", "Galloper Wind Farm", "Aberdeen Offshore Windfarm - Demonstration"):
        assert name in offshore.index, name
    assert offshore.at["Race Bank", "banding"] == pytest.approx(1.8)
    portfolio = load_lccc_portfolio(registers["lccc_portfolio"])
    triton = portfolio[portfolio.unit_name.str.contains("Triton Knoll")]
    assert set(triton["cfd_round"]) == {"AR2"}
    assert triton["unit_capacity_mw"].sum() == pytest.approx(847.5)
