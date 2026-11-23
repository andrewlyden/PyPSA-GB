"""Hinkley Point C bought in 2030 stays the state's in 2040/2050 (user, results review, 13 Nov 2026, finding 1).

Before the fix the 2040/2050 registers left HPC's cohorts private on their CfD (in-construction, ownership 0) while the
carried 2030 annuity still paid for HPC: consumers paid HPC twice in every later S3 year.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.public_ownership import calculate_financing_scenarios as cfs
from scripts.public_ownership.levers import DEFAULT_SETTINGS

S3 = dict(DEFAULT_SETTINGS, legacy_fleet="acquire_all", new_build_generation="public")
S3_LEFT_PRIVATE = dict(S3, in_construction="left_private")
S3_COMPLETION = dict(S3, in_construction="public_completion")
S2 = dict(DEFAULT_SETTINGS, new_build_generation="public")


@pytest.mark.parametrize("settings,expected", [(S3, True), (S3_COMPLETION, True), (S3_LEFT_PRIVATE, False),
                                               (S2, False), (DEFAULT_SETTINGS, False)])
def test_which_cases_carry_hpc_as_bought(settings, expected):
    assert cfs.carries_acquired_in_construction(settings) is expected


def _rows(vintages, rounds):
    in_construction = np.array(vintages) == "in_construction"
    rounds = np.array(rounds, dtype=object)
    rows = SimpleNamespace(base_vintage=np.array(vintages, dtype=object), _vintage_override=None,
                           pre_legislation_contract=np.zeros(len(vintages), dtype=bool),
                           in_construction_carried_round=in_construction & np.isin(rounds, cfs.CARRIED_IN_CONSTRUCTION_ROUNDS),
                           in_construction_hpc=in_construction & (rounds == cfs.HPC_CONTRACT_ROUND),
                           acquired_in_construction=np.zeros(len(vintages), dtype=bool))
    rows.gbe_owned_rows = lambda settings, plan: None
    return rows


def _view(rows):
    return rows.base_vintage if rows._vintage_override is None else rows._vintage_override


def test_a_buying_case_reads_hpc_cohorts_as_legacy_and_nothing_else_moves():
    rows = _rows(["in_construction", "in_construction", "legacy", "new_build"], ["HPC", "SZC_RAB", "", "PAR7_2035"])
    plan = pd.DataFrame({"acquirable": [True] * 4})
    cfs._RowInputs.use_vintage(rows, S3, plan)
    # Sizewell C, not in the 2030 purchase, is the buying case's own new capacity (user, 15 Nov 2026).
    assert list(_view(rows)) == ["legacy", "new_build", "legacy", "new_build"]
    for settings in (S3_LEFT_PRIVATE, S2):
        rows = _rows(["in_construction"], ["HPC"])
        cfs._RowInputs.use_vintage(rows, settings, plan if settings is S3_LEFT_PRIVATE else None)
        assert list(_view(rows)) == ["in_construction"]


def test_the_scalar_reference_owns_hpc_under_a_buying_case_only():
    hpc = pd.Series({"finance_vintage": "in_construction", "cfd_round": "HPC", "carrier": "nuclear",
                     "evidence_status": "resolved", "component_type": "Generator"})
    plan = pd.DataFrame({"acquirable": [True], "legacy_scope_group": [np.nan]})
    assert cfs._state_owned(hpc, S3, plan, 0)
    assert not cfs._state_owned(hpc, S3_LEFT_PRIVATE, plan, 0)
    szc = hpc.copy()
    szc["cfd_round"] = "SZC_RAB"
    assert cfs._state_owned(szc, S3, plan, 0)                      # not bought: the case's own new capacity
    assert not cfs._state_owned(szc, S2, None, 0)                  # a case that buys nothing leaves it private


# ── review finding 9 (13 Nov 2026): every round the 2030 purchase bought, and its costing ──────────────────────

def test_ar7_bought_in_2030_is_carried_and_hpc_follows_its_own_lever():
    """AR7 offshore bought operating in 2030 is in construction in the FES 2040 vintages: a buying case still owns it.
    HPC left private (``in_construction: left_private``) stays out; post-AR7 FES cohorts and Sizewell C, never bought,
    are the buying case's own new capacity (user, 15 Nov 2026)."""
    rows = _rows(["in_construction"] * 4, ["AR7", "HPC", "PAR7_2031", "SZC_RAB"])
    plan = pd.DataFrame({"acquirable": [True] * 4})
    cfs._RowInputs.use_vintage(rows, S3, plan)
    assert list(_view(rows)) == ["legacy", "legacy", "new_build", "new_build"]
    rows = _rows(["in_construction"] * 2, ["AR7", "HPC"])
    cfs._RowInputs.use_vintage(rows, S3_LEFT_PRIVATE, plan)
    assert list(_view(rows)) == ["legacy", "in_construction"]
    rows = _rows(["in_construction"] * 2, ["AR7", "HPC"])
    cfs._RowInputs.use_vintage(rows, S2, None)
    assert list(_view(rows)) == ["in_construction"] * 2


def test_the_carried_mask_needs_a_buying_case():
    rounds = np.array([True, True])
    hpc = np.array([False, True])
    assert list(cfs.carried_in_construction_mask(rounds, hpc, S3)) == [True, True]
    assert list(cfs.carried_in_construction_mask(rounds, hpc, S3_LEFT_PRIVATE)) == [True, False]
    assert list(cfs.carried_in_construction_mask(rounds, hpc, S2)) == [False, False]


def test_the_scalar_reference_owns_ar7_under_a_buying_case():
    ar7 = pd.Series({"finance_vintage": "in_construction", "cfd_round": "AR7", "carrier": "wind_offshore",
                     "evidence_status": "resolved", "component_type": "Generator"})
    plan = pd.DataFrame({"acquirable": [True], "legacy_scope_group": [np.nan]})
    assert cfs._state_owned(ar7, S3, plan, 0)
    assert cfs._state_owned(ar7, S3_LEFT_PRIVATE, plan, 0)          # the HPC lever does not reach AR7
    par7 = ar7.copy()
    par7["cfd_round"] = "PAR7_2031"
    assert cfs._state_owned(par7, S3, plan, 0)                     # never bought: the case's own new capacity
    assert not cfs._state_owned(par7, S2, None, 0)
