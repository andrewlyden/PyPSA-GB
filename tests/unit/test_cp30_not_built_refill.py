"""Sites not built by 2030 leave the named fleet and the ES5 proxy refills them (user decision, 14 Oct 2026).

A site whose every LCCC contract is terminated, or which holds none, carries
``action: not_built_refill`` in the checked CfD table. The brownfield fleet
excludes it through the named-fleet exclusions, so its (carrier, region) ES5
target is kept and the proxy residual grows by its MW: CP30's capacity is met.
Its terminated award stays netted off its round, so the proxy does not take it
at the old strike.
"""
from pathlib import Path

import pandas as pd
import pytest

from scripts.CP30.brownfield_fleet import apply_named_fleet_exclusions, reconcile_with_es5
from scripts.renewables.support_register_labels import (
    checked_cfd_exclusions,
    load_checked_cfd_contracts,
    terminated_round_capacity,
)

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data" / "market" / "lccc" / "checked_cfd_contracts.csv"
EXCLUSIONS_FILE = ROOT / "data" / "renewables" / "cp30_named_fleet_exclusions.csv"


def _named():
    return pd.DataFrame([
        {"site_name": "Hornsea 4", "repd_ref_id": "2473.0", "technology": "wind_offshore", "cp30_region": "R1",
         "capacity_mw": 2600.0, "maturity_rank": 2},
        {"site_name": "Hornsea 2", "repd_ref_id": "2502", "technology": "wind_offshore", "cp30_region": "R1",
         "capacity_mw": 1320.0, "maturity_rank": 0},
        {"site_name": "Seagreen", "repd_ref_id": "2540", "technology": "wind_offshore", "cp30_region": "R2",
         "capacity_mw": 1075.0, "maturity_rank": 0},
        {"site_name": "Chirmorie Wind Farm", "repd_ref_id": "2767", "technology": "wind_onshore",
         "cp30_region": "R2", "capacity_mw": 80.0, "maturity_rank": 2},
    ]).assign(operational_date="01/01/2028", planning_granted_date="2024-01-01")


def _es5():
    return pd.DataFrame([
        {"carrier": "wind_offshore", "cp30_region": "R1", "capacity_mw": 10000.0},
        {"carrier": "wind_offshore", "cp30_region": "R2", "capacity_mw": 5000.0},
        {"carrier": "wind_onshore", "cp30_region": "R2", "capacity_mw": 3000.0},
    ])


@pytest.fixture(scope="module")
def table():
    return load_checked_cfd_contracts(TABLE)


def test_not_built_sites_are_terminated_or_uncontracted_with_evidence(table):
    not_built = table[table["action"].eq("not_built_refill")]
    assert set(not_built["lccc_status"]) <= {"Terminated", "none"}
    # Every site whose contracts are all terminated (or that holds none) is not built, except
    # operating plant (Netley Landfill Solar, terminated_operating); a site with a live round
    # beside a terminated one (High Constellation) applies its live contract.
    site_live = (~table["lccc_status"].isin(["Terminated", "none"])).groupby(table["repd_ref_id"]).transform("any")
    dead = table[~site_live]
    assert set(dead["action"]) == {"not_built_refill", "terminated_operating"}
    assert set(dead.loc[dead["action"].eq("terminated_operating"), "repd_ref_id"]) == {"2044"}
    assert not_built["reason"].str.contains("refilled by the CP30 ES5 proxy").all()
    assert not_built["reason"].str.contains("Not an AR7").all()
    exclusions = checked_cfd_exclusions(table).set_index("repd_ref_id")
    assert len(exclusions) == 24
    assert set(exclusions["first_modelled_year"]) == {2031}
    assert exclusions.at["2473", "evidence"] == "LCCC Terminated (AR6-HPF-010)"
    assert exclusions.at["7692", "evidence"] == "LCCC none"
    assert "2044" not in exclusions.index          # Netley operates: it keeps its plant, loses its CfD
    assert "12005" in exclusions.index             # Church Farm, Withington: AR5-CSP-700 terminated


def test_excluded_sites_are_refilled_by_the_proxy_in_their_region(table):
    named, es5 = _named(), _es5()
    _, residual_before, _, _ = reconcile_with_es5(named, es5)
    kept, excluded = apply_named_fleet_exclusions(named, None, 2030, extra=checked_cfd_exclusions(table))
    assert set(excluded["site_name"]) == {"Hornsea 4", "Chirmorie Wind Farm"}
    assert excluded.set_index("site_name").at["Hornsea 4", "exclusion_evidence"] == "LCCC Terminated (AR6-HPF-010)"
    kept, residual_after, _, _ = reconcile_with_es5(kept, es5)
    before = residual_before.set_index(["carrier", "cp30_region"])["capacity_mw"]
    after = residual_after.set_index(["carrier", "cp30_region"])["capacity_mw"]
    assert after[("wind_offshore", "R1")] - before[("wind_offshore", "R1")] == pytest.approx(2600.0)
    assert after[("wind_onshore", "R2")] - before[("wind_onshore", "R2")] == pytest.approx(80.0)
    assert after[("wind_offshore", "R2")] == pytest.approx(before[("wind_offshore", "R2")])
    # Named plus proxy meets every ES5 target: no shortfall.
    built = kept.groupby(["technology", "cp30_region"])["capacity_mw"].sum()
    for key, target in es5.set_index(["carrier", "cp30_region"])["capacity_mw"].items():
        assert built.get(key, 0.0) + after[key] == pytest.approx(target)


def test_not_built_exclusions_apply_only_through_2030(table):
    kept, excluded = apply_named_fleet_exclusions(_named(), None, 2031, extra=checked_cfd_exclusions(table))
    assert excluded.empty and len(kept) == len(_named())


def test_exclusions_file_and_checked_table_combine_without_overlap(table):
    combined_named = pd.concat([_named(), pd.DataFrame([
        {"site_name": "Stornoway Wind Farm", "repd_ref_id": "6665", "technology": "wind_onshore",
         "cp30_region": "R2", "capacity_mw": 196.0, "maturity_rank": 2}])], ignore_index=True)
    _, excluded = apply_named_fleet_exclusions(combined_named, EXCLUSIONS_FILE, 2030,
                                               extra=checked_cfd_exclusions(table))
    assert {"Hornsea 4", "Chirmorie Wind Farm", "Stornoway Wind Farm"} == set(excluded["site_name"])
    file_ids = set(pd.read_csv(EXCLUSIONS_FILE, dtype=str)["repd_ref_id"])
    assert not file_ids & set(checked_cfd_exclusions(table)["repd_ref_id"])
    duplicate = checked_cfd_exclusions(table).iloc[[0, 0]]
    with pytest.raises(ValueError, match="twice"):
        apply_named_fleet_exclusions(_named(), None, 2030, extra=duplicate)


def test_terminated_awards_stay_netted_so_the_refill_is_not_on_the_old_rounds(table):
    held = terminated_round_capacity(table)
    offshore = held[held["carrier"].eq("wind_offshore")].set_index("cfd_round")["capacity_mw"]
    assert offshore["AR4"] == pytest.approx(1396.0) and offshore["AR6"] == pytest.approx(2400.0)
    assert offshore["AR3"] == pytest.approx(12.0)


def test_loader_refuses_inconsistent_not_built_rows(tmp_path):
    path = tmp_path / "t.csv"
    base = {"repd_ref_id": "1", "site_name": "a", "cfd_round": "AR4", "contracted_mw": 1, "cfd_ids": "x",
            "lccc_status": "Terminated", "source": "s", "action": "not_built_refill", "reason": "r"}
    pd.DataFrame([{**base, "reason": ""}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="needs a reason"):
        load_checked_cfd_contracts(path)
    pd.DataFrame([{**base, "lccc_status": "Pre-MDD"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="live contract"):
        load_checked_cfd_contracts(path)
    pd.DataFrame([{**base, "action": "remove_2030"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown actions"):
        load_checked_cfd_contracts(path)
