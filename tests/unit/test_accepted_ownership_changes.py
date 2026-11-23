"""Declared ownership changes the solved P2 months are accepted with (user, 15 Nov 2026): capped and recorded."""

import pandas as pd
import pytest

from scripts.public_ownership import verify_ownership_tables_2050 as v


def _tables():
    keys = {"case_id": ["c", "c", "c"], "generator": ["solar_a", "onshore_b", "gas_c"]}
    expected = pd.DataFrame({**keys, "state_capacity_mw": [60.0, 120.0, 0.0]})
    saved = pd.DataFrame({**keys, "state_capacity_mw": [100.0, 100.0, 0.0]})
    register = pd.DataFrame({"asset_id": ["solar_a::b2030", "onshore_b::b2031", "gas_c::b2030"],
                             "carrier": ["solar_pv", "wind_onshore", "CCGT"]})
    return expected, saved, register


def _accepted(cap_solar=41.0):
    return pd.DataFrame({"run_group": "g", "table": "deployment_share_100", "carrier": ["solar_pv", "wind_onshore"],
                         "cap_gross_mw": [cap_solar, 21.0], "decided": "user", "reason": "measured, negligible"})


def test_declared_moves_within_their_caps_leave_the_check_and_are_recorded():
    expected, saved, register = _tables()
    e, s, records = v.split_declared(expected, saved, register, "x_deployment_share_100.csv", _accepted())
    assert e["generator"].tolist() == ["gas_c"] and s["generator"].tolist() == ["gas_c"]
    assert {r["carrier"]: r["net_state_mw_change"] for r in records} == {"solar_pv": -40.0, "wind_onshore": 20.0}


def test_a_move_above_its_cap_still_fails():
    expected, saved, register = _tables()
    with pytest.raises(SystemExit, match="above its declared cap"):
        v.split_declared(expected, saved, register, "x_deployment_share_100.csv", _accepted(cap_solar=30.0))


def test_a_table_without_declarations_is_untouched():
    expected, saved, register = _tables()
    e, s, records = v.split_declared(expected, saved, register, "x_deployment_share_25.csv", _accepted())
    assert len(e) == 3 and len(s) == 3 and records == []
