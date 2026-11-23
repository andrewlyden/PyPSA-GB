"""Brownfield CP30 fleet: named REPD projects inside CP30's ES5 envelopes.

Toy fleets against toy regional targets, one test per branch of the
reconciliation, plus the properties the design depends on: capacity is
conserved to the ES5 total, operational plant is never trimmed, and the
historical site-file schema is untouched.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.CP30.brownfield_fleet import (
    BROWNFIELD_CARRIERS, MATURITY_RANK, apply_offshore_landing_crosswalk, build_fleet_audit,
    load_named_fleet, reconcile_with_es5, retire_by_lifetime,
)
from tests.helpers import requires_outputs


def _named(rows):
    frame = pd.DataFrame(rows)
    frame["maturity_rank"] = frame["development_status"].map(MATURITY_RANK)
    frame["fleet_origin"] = "named"
    for column in ("cfd_round", "ro_banding", "planning_granted_date", "repd_ref_id"):
        if column not in frame.columns:
            frame[column] = None
    return frame


def _targets(rows):
    return pd.DataFrame(rows, columns=["carrier", "cp30_region", "capacity_mw"])


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------

def test_residual_is_target_minus_named_when_named_fits():
    named = _named([
        {"site_name": "a", "technology": "wind_onshore", "cp30_region": "North England",
         "capacity_mw": 300.0, "development_status": "Operational"},
    ])
    kept, residual, trimmed, audit = reconcile_with_es5(
        named, _targets([("wind_onshore", "North England", 1000.0)])
    )
    assert kept.capacity_mw.sum() == 300.0
    assert residual.set_index(["carrier", "cp30_region"]).loc[("wind_onshore", "North England"), "capacity_mw"] == 700.0
    assert trimmed.empty
    assert audit.iloc[0]["named_share_of_target"] == pytest.approx(0.3)


def test_overshoot_trims_least_mature_pipeline_first_and_never_operational():
    named = _named([
        {"site_name": "built", "technology": "solar_pv", "cp30_region": "Midlands",
         "capacity_mw": 500.0, "development_status": "Operational"},
        {"site_name": "building", "technology": "solar_pv", "cp30_region": "Midlands",
         "capacity_mw": 300.0, "development_status": "Under Construction"},
        {"site_name": "consented_old", "technology": "solar_pv", "cp30_region": "Midlands",
         "capacity_mw": 200.0, "development_status": "Awaiting Construction",
         "planning_granted_date": "2021-01-01"},
        {"site_name": "consented_new", "technology": "solar_pv", "cp30_region": "Midlands",
         "capacity_mw": 200.0, "development_status": "Awaiting Construction",
         "planning_granted_date": "2024-06-01"},
    ])
    kept, residual, trimmed, audit = reconcile_with_es5(
        named, _targets([("solar_pv", "Midlands", 900.0)])
    )
    # 1200 named against 900: the most recently consented project goes first
    # (200), then 100 is derated from the older consented one.
    assert kept.capacity_mw.sum() == pytest.approx(900.0)
    assert set(trimmed.site_name) == {"consented_new", "consented_old"}
    by_name = trimmed.set_index("site_name")
    assert by_name.loc["consented_new", "trim_action"] == "dropped"
    assert by_name.loc["consented_old", "trim_action"] == "derated"
    assert by_name.loc["consented_old", "trimmed_mw"] == pytest.approx(100.0)
    assert "built" in set(kept.site_name) and "building" in set(kept.site_name)
    assert residual.capacity_mw.sum() == 0.0


def test_operational_excess_is_carried_nationally_not_trimmed():
    named = _named([
        {"site_name": "big_built", "technology": "wind_onshore", "cp30_region": "South Scotland",
         "capacity_mw": 1200.0, "development_status": "Operational"},
        {"site_name": "small", "technology": "wind_onshore", "cp30_region": "North England",
         "capacity_mw": 100.0, "development_status": "Operational"},
    ])
    targets = _targets([("wind_onshore", "South Scotland", 1000.0),
                        ("wind_onshore", "North England", 500.0)])
    kept, residual, trimmed, audit = reconcile_with_es5(named, targets)
    assert trimmed.empty
    assert kept.capacity_mw.sum() == 1300.0
    # National total holds: 1500 target = 1300 named + 200 residual, with the
    # 200 MW built-fleet excess taken off the other region's residual (400 -> 200).
    assert residual.capacity_mw.sum() == pytest.approx(200.0)
    south = audit.set_index("cp30_region").loc["South Scotland"]
    assert south["operational_excess_mw"] == pytest.approx(200.0)


def test_operational_excess_beyond_national_target_is_an_error():
    named = _named([
        {"site_name": "x", "technology": "solar_pv", "cp30_region": "Wales",
         "capacity_mw": 2000.0, "development_status": "Operational"},
    ])
    with pytest.raises(ValueError, match="exceeds CP30 nationally"):
        reconcile_with_es5(named, _targets([("solar_pv", "Wales", 1000.0)]))


def test_error_policy_refuses_any_overshoot():
    named = _named([
        {"site_name": "p", "technology": "wind_offshore", "cp30_region": "East England",
         "capacity_mw": 900.0, "development_status": "Awaiting Construction"},
    ])
    with pytest.raises(ValueError, match="overshoot policy is 'error'"):
        reconcile_with_es5(named, _targets([("wind_offshore", "East England", 500.0)]), policy="error")


def test_unknown_policy_is_rejected():
    empty = pd.DataFrame(columns=["site_name", "technology", "cp30_region", "capacity_mw",
                                  "development_status", "maturity_rank"])
    with pytest.raises(ValueError, match="Overshoot policy"):
        reconcile_with_es5(empty, _targets([]), policy="whatever")


def test_named_capacity_plus_residual_equals_es5_everywhere():
    """The invariant validation relies on, across regions and carriers."""
    rows = []
    for carrier in BROWNFIELD_CARRIERS:
        for region, mw in (("A", 250.0), ("B", 1200.0), ("C", 0.0)):
            rows.append({"site_name": f"{carrier}-{region}", "technology": carrier, "cp30_region": region,
                         "capacity_mw": mw, "development_status": "Awaiting Construction"})
    named = _named(rows)
    targets = _targets([(c, r, 800.0) for c in BROWNFIELD_CARRIERS for r in ("A", "B", "C")])
    kept, residual, trimmed, audit = reconcile_with_es5(named, targets)
    total = kept.groupby(["technology", "cp30_region"]).capacity_mw.sum().add(
        residual.set_index(["carrier", "cp30_region"]).capacity_mw, fill_value=0.0
    )
    assert (total.round(6) == 800.0).all()


# --------------------------------------------------------------------------
# Retirement
# --------------------------------------------------------------------------

def test_retirement_removes_plant_past_its_lifetime_only():
    operational = pd.DataFrame({
        "site_name": ["old", "young", "undated"],
        "technology": ["wind_onshore"] * 3,
        "capacity_mw": [10.0, 20.0, 30.0],
        "operational_date": ["01/06/1998", "01/06/2015", None],
    })
    kept, retired = retire_by_lifetime(operational, 2030, {"wind_onshore": 30})
    assert list(retired.site_name) == ["old"]
    assert set(kept.site_name) == {"young", "undated"}
    # A carrier with no declared lifetime is never retired.
    kept2, retired2 = retire_by_lifetime(operational, 2030, {"solar_pv": 35})
    assert retired2.empty and len(kept2) == 3


# --------------------------------------------------------------------------
# Named fleet loading
# --------------------------------------------------------------------------

def _write(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_named_fleet_applies_status_rule_and_developer_dates(tmp_path):
    operational = _write(tmp_path, "op.csv", pd.DataFrame({
        "site_name": ["op1"], "capacity_mw": [100.0], "lat": [55.0], "lon": [-3.0],
        "operational_date": ["01/01/2010"], "repd_ref_id": [1], "cfd_round": [None], "ro_banding": [0.9],
    }))
    pipeline = _write(tmp_path, "pipe.csv", pd.DataFrame({
        "site_name": ["uc", "ac", "late", "submitted"],
        "capacity_mw": [50.0, 60.0, 70.0, 80.0], "lat": [55.0] * 4, "lon": [-3.0] * 4,
        "development_status": ["Under Construction", "Awaiting Construction",
                               "Awaiting Construction", "Application Submitted"],
        "expected_operational_date": [None, None, "2032-01-01", None],
        "repd_ref_id": [2, 3, 4, 5],
    }))
    fleet, retired = load_named_fleet(
        {"wind_onshore": operational}, {"wind_onshore": pipeline}, 2030,
        {"statuses": ["Operational", "Under Construction", "Awaiting Construction"]},
    )
    # Operational kept; UC and AC kept; the AC project dated 2032 excluded;
    # Application Submitted excluded by the status rule.
    assert set(fleet.site_name) == {"op1", "uc", "ac"}
    assert (fleet.fleet_origin == "named").all()
    assert fleet.set_index("site_name").loc["op1", "ro_banding"] == 0.9
    assert retired.empty


def test_named_fleet_can_include_submitted_applications_as_a_sensitivity(tmp_path):
    pipeline = _write(tmp_path, "pipe.csv", pd.DataFrame({
        "site_name": ["submitted"], "capacity_mw": [80.0], "lat": [55.0], "lon": [-3.0],
        "development_status": ["Application Submitted"], "repd_ref_id": [5],
    }))
    fleet, _ = load_named_fleet({}, {"solar_pv": pipeline}, 2030,
                                {"statuses": ["Application Submitted"]})
    assert list(fleet.site_name) == ["submitted"]


def test_unknown_status_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown brownfield statuses"):
        load_named_fleet({}, {}, 2030, {"statuses": ["Built"]})


# --------------------------------------------------------------------------
# Offshore landing crosswalk
# --------------------------------------------------------------------------

class _Network:
    def __init__(self, buses):
        self.buses = pd.DataFrame(index=pd.Index(buses, name="Bus"))


def test_crosswalk_lands_named_offshore_and_records_gaps(tmp_path):
    crosswalk = _write(tmp_path, "xw.csv", pd.DataFrame({
        "repd_ref_id": ["10", "11"], "site_name": ["A", "B"],
        "landing_bus": ["NORM41", ""], "evidence": ["tec_register", "not_represented_in_etys"],
        "source": ["tec", "tec"],
    }))
    fleet = pd.DataFrame({
        "site_name": ["A", "B", "C", "onshore"],
        "technology": ["wind_offshore"] * 3 + ["wind_onshore"],
        "repd_ref_id": ["10", "11", "12", "13"],
        "bus": ["XXX", "YYY", "ZZZ", "WWW"], "capacity_mw": [1.0] * 4,
    })
    result = apply_offshore_landing_crosswalk(fleet, crosswalk, _Network(["NORM41", "XXX", "YYY", "ZZZ", "WWW"]))
    by = result.set_index("site_name")
    assert by.loc["A", "bus"] == "NORM41"
    assert by.loc["A", "landing_evidence"] == "tec_register"
    # The same marker the explicit allocation uses, so BMU remapping leaves it alone.
    assert by.loc["A", "data_source"] == "CP30_connection_allocation"
    assert by.loc["B", "bus"] == "YYY" and by.loc["B", "landing_evidence"] == "not_represented_in_etys"
    assert by.loc["C", "landing_evidence"] == "none"
    assert by.loc["onshore", "landing_evidence"] == "not_applicable"


def test_crosswalk_error_policy_refuses_unmatched_offshore(tmp_path):
    crosswalk = _write(tmp_path, "xw.csv", pd.DataFrame({
        "repd_ref_id": ["10"], "site_name": ["A"], "landing_bus": ["NORM41"],
        "evidence": ["tec_register"], "source": ["tec"],
    }))
    fleet = pd.DataFrame({"site_name": ["C"], "technology": ["wind_offshore"], "repd_ref_id": ["12"],
                          "bus": ["ZZZ"], "capacity_mw": [1.0]})
    with pytest.raises(ValueError, match="no landing crosswalk row"):
        apply_offshore_landing_crosswalk(fleet, crosswalk, _Network(["NORM41", "ZZZ"]), policy="error")


def test_crosswalk_bus_must_exist_in_network(tmp_path):
    crosswalk = _write(tmp_path, "xw.csv", pd.DataFrame({
        "repd_ref_id": ["10"], "site_name": ["A"], "landing_bus": ["NOPE41"],
        "evidence": ["tec_register"], "source": ["tec"],
    }))
    fleet = pd.DataFrame({"site_name": ["A"], "technology": ["wind_offshore"], "repd_ref_id": ["10"],
                          "bus": ["XXX"], "capacity_mw": [1.0]})
    with pytest.raises(ValueError, match="absent from this network"):
        apply_offshore_landing_crosswalk(fleet, crosswalk, _Network(["XXX"]))


class _VoltageNetwork:
    def __init__(self):
        self.buses = pd.DataFrame(
            {"v_nom": [132.0, 400.0, 400.0], "x": [-3.0, -3.1, -6.0], "y": [58.0, 58.05, 55.0]},
            index=pd.Index(["LOW1-", "HIGH41", "FAR41"], name="Bus"),
        )


def test_large_fallback_landing_moves_to_nearest_transmission_bus(tmp_path):
    """A 2 GW farm must not be placed on a 132 kV busbar by proximity alone."""
    crosswalk = tmp_path / "xw.csv"
    pd.DataFrame({"repd_ref_id": ["99"], "site_name": ["gap"], "landing_bus": [""],
                  "evidence": ["not_represented_in_etys"], "source": ["tec"]}).to_csv(crosswalk, index=False)
    fleet = pd.DataFrame({
        "site_name": ["big", "small", "gap"], "technology": ["wind_offshore"] * 3,
        "repd_ref_id": ["1", "2", "99"], "bus": ["LOW1-"] * 3,
        "capacity_mw": [2000.0, 90.0, 900.0], "lat": [58.0] * 3, "lon": [-3.0] * 3,
    })
    result = apply_offshore_landing_crosswalk(fleet, crosswalk, _VoltageNetwork()).set_index("site_name")
    # Large fallback moved to the nearest >=275 kV bus; the evidence flag says so.
    assert result.loc["big", "bus"] == "HIGH41"
    assert result.loc["big", "landing_evidence"] == "none_nearest_transmission_bus"
    # A small farm genuinely on 132 kV is left where it is.
    assert result.loc["small", "bus"] == "LOW1-"
    assert result.loc["small", "landing_evidence"] == "none"
    # A recorded ETYS gap keeps its provenance through the voltage move.
    assert result.loc["gap", "bus"] == "HIGH41"
    assert result.loc["gap", "landing_evidence"] == "not_represented_in_etys_nearest_transmission_bus"


def test_shipped_crosswalk_is_well_formed_and_keyed_by_repd_ref_id():
    crosswalk = pd.read_csv("data/network/ETYS/repd_offshore_landing_crosswalk.csv", dtype=str).fillna("")
    assert not crosswalk.repd_ref_id.duplicated().any()
    assert crosswalk.evidence.str.strip().ne("").all()
    assert crosswalk.source.str.strip().ne("").all()
    # Every referenced project is a real REPD offshore row.
    repd = pd.read_csv("data/renewables/repd-q2-jul-2025.csv", encoding="latin-1", low_memory=False)
    offshore_ids = set(repd.loc[repd["Technology Type"].eq("Wind Offshore"), "Ref ID"].astype(str))
    assert set(crosswalk.repd_ref_id) <= offshore_ids
    # OW12 (3 Oct 2026): every evidenced landing cites a source that was checked, and
    # no row is left to the nearest-bus fallback: Moray East moved to its ETYS 2025
    # onshore bus MORF4- and West of Orkney to Spittal SPIT2J (user decisions, 3 Oct 2026).
    assert not crosswalk.evidence.eq("project_publication_unverified").any()
    assert crosswalk.landing_bus.str.strip().ne("").all()
    by_id = crosswalk.set_index("repd_ref_id")
    assert by_id.at["2534", "landing_bus"] == "MORF4-"
    assert by_id.at["10772", "landing_bus"] == "SPIT2J"
    # Green Volt connects at New Deer (developer FAQ; ETYS 2024 Appendix B GRNV4- to NEDE4J).
    assert crosswalk.set_index("repd_ref_id").at["11851", "landing_bus"] == "NEDE4J"


# --------------------------------------------------------------------------
# Audit and data products
# --------------------------------------------------------------------------

def test_audit_reports_named_share_and_support_coverage():
    sites = pd.DataFrame({
        "technology": ["wind_offshore"] * 3,
        "capacity_mw": [600.0, 400.0, 1000.0],
        "fleet_origin": ["named", "named", "increment"],
        "development_status": ["Operational", "Awaiting Construction", None],
        "support_type": ["CfD", "CfD", "CfD"],
        "cfd_round": ["AR2", None, "AR7"],
        "landing_evidence": ["evidence_register", "none", None],
    })
    regional = pd.DataFrame([{"carrier": "wind_offshore", "cp30_region": "R", "es5_target_mw": 2000.0}])
    audit = build_fleet_audit(sites, regional, pd.DataFrame(), pd.DataFrame())
    entry = audit["carriers"]["wind_offshore"]
    assert entry["named_share"] == pytest.approx(0.5)
    assert entry["named_cfd_with_round_share"] == pytest.approx(0.6)
    assert entry["landing_evidence_mw"] == {"evidence_register": 600.0, "none": 400.0}
    assert audit["regional"][0]["es5_target_mw"] == 2000.0


@requires_outputs("resources/renewable/wind_offshore_pipeline_sites.csv")
def test_pipeline_site_files_exist_with_the_expected_schema():
    for carrier in BROWNFIELD_CARRIERS:
        path = Path(f"resources/renewable/{carrier}_pipeline_sites.csv")
        assert path.is_file(), path
        frame = pd.read_csv(path)
        assert {"site_name", "repd_ref_id", "capacity_mw", "lat", "lon", "cfd_round", "ro_banding",
                "development_status", "planning_granted_date"} <= set(frame.columns)
        assert frame.development_status.isin(["Under Construction", "Awaiting Construction",
                                              "Application Submitted"]).all()
        assert frame.capacity_mw.gt(0).all()


@requires_outputs("resources/renewable/wind_offshore_sites.csv")
def test_historical_site_file_schema_is_unchanged():
    """The brownfield fleet adds new files; the historical path's inputs must not move."""
    frame = pd.read_csv("resources/renewable/wind_offshore_sites.csv", nrows=1)
    assert {"site_name", "capacity_mw", "lat", "lon", "operational_date", "cfd_round", "ro_banding"} <= set(frame.columns)
    assert "development_status" not in frame.columns


def test_cp30_reinforced_scenarios_enable_the_brownfield_fleet():
    import sys
    sys.path.insert(0, str(Path("config").resolve()))
    from config_loader import load_config
    scenario = load_config()["scenarios"][
        "CP30_FFR_2030_Jan_cp30_network"]
    brownfield = scenario["fleet"]["spatial_proxy"]["brownfield"]
    assert brownfield["enabled"] is True
    assert "Application Submitted" not in brownfield["statuses"]
    assert Path(brownfield["offshore_landing_crosswalk"]).is_file()
    # Defaults supply the retirement lifetimes the scenario block does not repeat.
    defaults = scenario["fleet"]["brownfield"]
    assert defaults["retirement_lifetimes_years"] == {"wind_onshore": 30, "wind_offshore": 30, "solar_pv": 35}


# --------------------------------------------------------------------------
# Declared named-fleet exclusions (Lewis farms behind the Western Isles link)
# --------------------------------------------------------------------------

def _exclusions(tmp_path, first_year=2031):
    path = tmp_path / "exclusions.csv"
    pd.DataFrame([{
        "repd_ref_id": "6665", "site_name": "Stornoway Wind Farm (second resubmission)",
        "capacity_mw": 196.0, "first_modelled_year": first_year,
        "reason": "Needs the Western Isles HVDC link", "evidence": "TEC 2030-10-31",
    }]).to_csv(path, index=False)
    return path


def test_declared_exclusion_drops_project_and_capacity_passes_to_the_residual(tmp_path):
    from scripts.CP30.brownfield_fleet import apply_named_fleet_exclusions
    named = _named([
        {"site_name": "Stornoway", "technology": "wind_onshore", "cp30_region": "North Scotland",
         "capacity_mw": 196.0, "development_status": "Awaiting Construction", "repd_ref_id": "6665.0"},
        {"site_name": "Other", "technology": "wind_onshore", "cp30_region": "North Scotland",
         "capacity_mw": 300.0, "development_status": "Operational", "repd_ref_id": "100"},
    ])
    kept, excluded = apply_named_fleet_exclusions(named, _exclusions(tmp_path), 2030)
    assert kept.site_name.tolist() == ["Other"]
    assert excluded.exclusion_reason.tolist() == ["Needs the Western Isles HVDC link"]

    targets = _targets([("wind_onshore", "North Scotland", 1000.0)])
    kept_r, residual, _, _ = reconcile_with_es5(kept, targets)
    before_r, residual_before, _, _ = reconcile_with_es5(named, targets)
    total = kept_r.capacity_mw.sum() + residual.capacity_mw.sum()
    assert total == pytest.approx(1000.0)
    assert residual.capacity_mw.sum() - residual_before.capacity_mw.sum() == pytest.approx(196.0)


def test_exclusion_lapses_from_its_first_modelled_year(tmp_path):
    from scripts.CP30.brownfield_fleet import apply_named_fleet_exclusions
    named = _named([
        {"site_name": "Stornoway", "technology": "wind_onshore", "cp30_region": "North Scotland",
         "capacity_mw": 196.0, "development_status": "Awaiting Construction", "repd_ref_id": "6665"},
    ])
    kept, excluded = apply_named_fleet_exclusions(named, _exclusions(tmp_path), 2031)
    assert len(kept) == 1 and excluded.empty


def test_repo_exclusion_file_lists_the_lewis_projects():
    from scripts.CP30.brownfield_fleet import DEFAULT_NAMED_EXCLUSIONS_FILE
    table = pd.read_csv(DEFAULT_NAMED_EXCLUSIONS_FILE, dtype=str)
    assert set(table.repd_ref_id) == {"6665", "4523", "3901", "4448"}
    assert (table.first_modelled_year.astype(int) > 2030).all()
    assert table.evidence.str.contains("TEC register").all()
