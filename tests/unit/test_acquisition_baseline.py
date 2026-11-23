"""D1 (user, 14 Nov 2026): a later year's buying case owns exactly what the acquisition year's purchase bought."""

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import acquisition_baseline as ab
from scripts.public_ownership import calculate_financing_scenarios as cfs
from scripts.public_ownership.levers import DEFAULT_SETTINGS

S3 = dict(DEFAULT_SETTINGS, legacy_fleet="acquire_all", new_build_generation="public")


def _rows(spec):
    return pd.DataFrame(spec, columns=["carrier", "finance_vintage", "build_year", "capacity_mw", "cfd_round",
                                       "excluded"])


def test_rows_are_acquired_oldest_first_and_plant_on_an_unbought_contract_never():
    rows = _rows([
        ("wind_offshore", "legacy", 2015, 400.0, "", False),
        ("wind_offshore", "legacy", 2025, 400.0, "AR4", False),
        ("wind_offshore", "in_construction", 2031, 300.0, "PAR7_2031", False),
        ("wind_offshore", "in_construction", 2032, 200.0, "AR7", False),     # a round the purchase bought: first
        ("wind_offshore", "legacy", 2030, 500.0, "", True),                   # excluded: never a candidate
        ("solar_pv", "legacy", 2020, 100.0, "", False),                       # no solar bought: private
    ])
    got = ab.acquired_rows(rows, {"offshore_wind": 1100.0}, carried_rounds=("AR7", "AR4"))
    # Built by 2030: 400 + 400; then AR7 (a bought round) 200 = 1,000. PAR7_2031 is a contract the purchase never
    # bought: never owned, however much headroom is left (15 Nov 2026; it used to fill the gap).
    assert list(got) == [True, True, False, True, False, False]
    got = ab.acquired_rows(rows, {"offshore_wind": 5000.0}, carried_rounds=("AR7", "AR4"))
    assert list(got) == [True, True, False, True, False, False]


def test_the_carrier_groups_join_cp30_and_fes_names():
    assert ab.carrier_group("LAES") == ab.carrier_group("Pumped Storage Hydroelectricity") == "long_duration_storage"
    assert ab.carrier_group("OCGT") == ab.carrier_group("gas_engine") == "gas_peaking"
    assert ab.carrier_group("geothermal") == "other:geothermal"


def _row_inputs(vintages, rounds, carriers, mws, years):
    n = len(vintages)
    rows = cfs._RowInputs.__new__(cfs._RowInputs)
    rows.n = n
    rows.base_vintage = np.array(vintages, dtype=object)
    rows._vintage_override = None
    rows.pre_legislation_contract = np.zeros(n, dtype=bool)
    rows.excluded = np.zeros(n, dtype=bool)
    rows.in_construction_carried_round = np.zeros(n, dtype=bool)
    rows.in_construction_hpc = (np.array(vintages) == "in_construction") & (np.array(rounds) == "HPC")
    rows.acquired_in_construction = np.zeros(n, dtype=bool)
    rows.baseline_acquired = None
    rows.baseline_legacy_not_bought = None
    rows.gbe_owned_rows = lambda settings, plan: None
    merged = pd.DataFrame({"carrier": carriers, "build_year": years, "capacity_mw": mws, "cfd_round": rounds})
    return rows, merged


def test_a_later_year_carries_bought_plant_built_later_and_leaves_unbought_legacy_private():
    rows, merged = _row_inputs(["legacy", "legacy", "in_construction", "in_construction"],
                               ["", "", "PAR7_2031", "SZC_RAB"],
                               ["wind_offshore", "wind_offshore", "wind_offshore", "nuclear"],
                               [500.0, 500.0, 300.0, 3260.0], [2010, 2034, 2031, 2038])
    rows.set_acquisition_baseline(merged, {"offshore_wind": 800.0})
    plan = pd.DataFrame({"acquirable": [True] * 4})
    cfs._RowInputs.use_vintage(rows, S3, plan)
    view = rows._vintage_override if rows._vintage_override is not None else rows.base_vintage
    # Only the 2010 farm was there at the purchase; the 2034 row (no contract) and the PAR7 and Sizewell C contracts
    # were never bought (15 Nov 2026: the old rule filled the 800 MW with the PAR7 twin). Built after the purchase,
    # they are S3's own new capacity (user, 15 Nov 2026: "if it's new capacity added then S3 should have it").
    assert list(rows.baseline_acquired) == [True, False, False, False]
    assert list(view) == ["legacy", "new_build", "new_build", "new_build"]
    assert list(rows.state_new_capacity) == [False, True, True, True]
    assert list(rows.baseline_legacy_not_bought) == [False, True, False, False]


def test_the_acquisition_year_comes_from_the_table_so_a_2035_purchase_needs_no_code_change(tmp_path):
    for group in ("cp30_2030_no_acceleration", "cp30_2030_no_acceleration_legislated_2025", "cp30_2030_cp30_network"):
        assert ab.baseline_acquisition_year(ab.baseline_path(group)) == 2030
    table = tmp_path / "acquisition_baseline_x.csv"
    pd.DataFrame({"group": ["solar", "battery"], "bought_mw": [10.0, 5.0], "carriers": ["solar_pv", "Battery"],
                  "acquisition_year": [2035, 2035], "source_case": "c", "source": "s"}).to_csv(table, index=False)
    assert ab.baseline_acquisition_year(table) == 2035
    # plant built in 2033 was running when a 2035 purchase was made: tier 0, ahead of a carried round
    rows = _rows([["solar_pv", "legacy", 2033, 6.0, "", False], ["solar_pv", "in_construction", 2034, 6.0, "AR6", False]])
    assert ab.acquired_rows(rows, {"solar": 6.0}, carried_rounds=("AR6",), acquisition_year=2035).tolist() == [True, False]
    pd.DataFrame({"group": ["solar"], "bought_mw": [1.0], "carriers": ["solar_pv"], "acquisition_year": [2030],
                  "source_case": "c", "source": "s"}).pipe(lambda f: pd.concat([f, f.assign(acquisition_year=2035)])) \
        .to_csv(table, index=False)
    with pytest.raises(ValueError, match="one table is one purchase"):
        ab.baseline_acquisition_year(table)


def test_a_contract_is_owned_only_up_to_what_the_purchase_bought_of_it():
    """15 Nov 2026: Sizewell B's 1,198 MW retired by 2040 and was handed to Hinkley unit 2 and Sizewell C."""
    rows = _rows([
        ("nuclear", "in_construction", 2031, 1630.0, "HPC", False),
        ("nuclear", "in_construction", 2031, 1630.0, "HPC", False),
        ("nuclear", "in_construction", 2038, 281.0, "SZC_RAB", False),
    ])
    caps = {"nuclear": {"HPC": 2317.0, "none": 1198.0}}
    got = ab.acquired_rows(rows, {"nuclear": 3515.0}, round_caps=caps)
    assert list(got) == [True, False, False]            # one unit within HPC's 2,317 MW; never Sizewell C


def test_the_2030_ccs_contract_is_recognised_under_its_later_name(tmp_path):
    table = tmp_path / "acquisition_baseline_x.csv"
    pd.DataFrame({"group": ["gas_ccs"], "bought_mw": [742.0], "carriers": ["CCS Gas"], "acquisition_year": [2030],
                  "source_case": "c", "source": "s", ab.ROUND_COLUMN: ["NZT_POWER_DPA=742.0"]}).to_csv(table, index=False)
    caps = ab.load_round_caps(table)
    rows = _rows([("CCS Gas", "legacy", 2029, 860.0, "DPA_2029", False),
                  ("CCS Gas", "in_construction", 2031, 1820.0, "DPA_2031", False)])
    assert list(ab.acquired_rows(rows, {"gas_ccs": 742.0}, round_caps=caps)) == [True, False]


def test_the_committed_tables_carry_the_purchase_by_contract():
    for group in ("cp30_2030_no_acceleration", "cp30_2030_no_acceleration_legislated_2025", "cp30_2030_cp30_network"):
        caps = ab.load_round_caps(ab.baseline_path(group))
        totals = ab.load_baseline(ab.baseline_path(group))
        for g, by_round in caps.items():
            own = {k: v for k, v in by_round.items() if k not in {e for es in ab.ROUND_EQUIVALENTS.values() for e in es}}
            assert sum(own.values()) == pytest.approx(totals[g], abs=1.0), (group, g)


def test_hinkley_counts_at_the_capacity_the_purchase_valued():
    """Independent review (15 Nov 2026): the valuation prices HPC at 3,260 MW from 2031, so S3 bought all of it."""
    rows = pd.DataFrame({"case_id": ab.BASELINE_CASE, "asset_id": ["Hinkley Point C::legacy", "Sizewell B::legacy"],
                         "carrier": "nuclear", "finance_vintage": "legacy", "state_owned": True,
                         "capacity_mw": [2317.0, 1198.0], "cfd_round": ["HPC", None]})
    table = ab.baseline_from_register(rows)
    assert table.loc[0, "bought_mw"] == pytest.approx(3260.0 + 1198.0)
    assert "HPC=3260.000000" in table.loc[0, ab.ROUND_COLUMN] and "none=1198.000000" in table.loc[0, ab.ROUND_COLUMN]


def test_contract_plant_the_purchase_bought_comes_before_uncontracted_plant():
    """Independent review (15 Nov 2026): 2040 marine owned 79 MW of uncontracted FES rows against 16 MW bought, and
    left 104 MW of the AR4-AR8 rows the purchase bought private (still paid their CfD beside the carried annuity)."""
    rows = _rows([
        ("marine", "legacy", 2015, 40.0, "", False),
        ("marine", "legacy", 2016, 39.0, "", False),
        ("marine", "legacy", 2027, 41.0, "AR4", False),
        ("marine", "legacy", 2029, 53.0, "AR5", False),
    ])
    caps = {"marine": {"none": 16.0, "AR4": 41.0, "AR5": 53.0}}
    got = ab.acquired_rows(rows, {"marine": 110.0}, round_caps=caps)
    # The two contracts first (94 MW), then uncontracted plant within the group's remaining 16 MW (half a row).
    assert list(got) == [False, False, True, True]
