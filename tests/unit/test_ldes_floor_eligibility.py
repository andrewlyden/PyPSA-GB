"""The LDES cap and floor follows the build date, not the ownership law's year (user decision, 16 Nov 2026)."""

import numpy as np
import pandas as pd

from scripts.public_ownership.public_account import ldes_floor_eligible_share, ldes_floor_topups

SETTINGS = {"ldes_enabled": True, "ldes_carriers": ["LAES", "Pumped Storage Hydroelectricity"],
            "ldes_floor_real_rate_by_case": {"private": 0.05674, "public_control": 0.05086, "public_ownership": None}}


def _rows(case_id="private", **overrides):
    base = {
        "case_id": case_id, "component_type": "StorageUnit", "carrier": "LAES",
        "annual_required_revenue_gbp": 100.0, "annual_capital_recovery_gbp": 80.0, "wacc": 0.10,
        "remaining_life_years": 30.0, "annual_fixed_om_gbp": 20.0, "annual_variable_om_gbp": 0.0,
        "market_revenue_gbp": 10.0, "capacity_market_revenue_gbp": 0.0, "ownership_share": 0.0,
    }
    rows = [
        {**base, "asset_id": "existing", "finance_vintage": "legacy", "legislation_status": "pre_policy_fleet"},
        {**base, "asset_id": "before_law", "finance_vintage": "legacy", "legislation_status": "pre_legislation"},
        {**base, "asset_id": "building", "finance_vintage": "in_construction", "legislation_status": "pre_legislation"},
        {**base, "asset_id": "after_law", "finance_vintage": "new_build", "legislation_status": "post_legislation"},
    ]
    frame = pd.DataFrame(rows)
    for key, value in overrides.items():
        frame[key] = value
    return frame


def test_every_store_built_from_2025_is_floored_whatever_the_law_year():
    floors = ldes_floor_topups(_rows(), SETTINGS)["ldes_floor_topup_paid_gbp"].to_numpy()
    assert floors[0] == 0.0
    assert np.all(floors[1:] > 0.0)
    # The same store gets the same floor whether it was committed before or after the law.
    assert floors[1] == floors[2] == floors[3]


def test_a_refurbished_share_of_existing_plant_is_left_out_pro_rata():
    frame = _rows(cohort_refurbishment_share=[0.0, 1.0, 0.5, 0.0])
    share = ldes_floor_eligible_share(frame).to_numpy()
    assert list(share) == [0.0, 0.0, 0.5, 1.0]
    floors = ldes_floor_topups(frame, SETTINGS)["ldes_floor_topup_paid_gbp"].to_numpy()
    assert floors[1] == 0.0 and np.isclose(floors[2], 0.5 * floors[3])


def test_a_register_without_status_keeps_the_new_build_test():
    frame = _rows().drop(columns="legislation_status")
    assert list(ldes_floor_eligible_share(frame)) == [0.0, 0.0, 0.0, 1.0]


def test_a_private_store_in_a_public_case_floors_at_the_private_rate():
    """S2's own floor rate is null (its WACC); a store it holds none of is a private store, floored as in private."""
    private = ldes_floor_topups(_rows("private"), SETTINGS)["ldes_floor_topup_paid_gbp"].to_numpy()
    public = ldes_floor_topups(_rows("public_ownership"), SETTINGS)["ldes_floor_topup_paid_gbp"].to_numpy()
    assert np.allclose(public[1:3], private[1:3])
