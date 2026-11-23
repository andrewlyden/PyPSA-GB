"""F10 (user decision, 6 Oct 2026): GB Energy as announced, a named P2 variant.

The state holds a 30% stake in the post-2025 offshore wind cohort (P1's stake
mechanics and contract form) and 8 GW of local solar and onshore wind, taken pro rata
from P2's cohort and owned at the public rate (P2's); everything else is private. The
partnership twin holds the 8 GW at 50%.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.public_ownership import calculate_financing_scenarios as cfs
from scripts.public_ownership import finance, levers
from scripts.public_ownership.reference_runs import configured_case_specs, expanded_cases
from config.config_loader import load_defaults


def _config():
    return load_defaults()["public_ownership"]


class TestLever:
    def test_the_values_and_the_named_cases(self):
        values = levers.LEVERS[levers.LEVER_NEW_BUILD_GENERATION].values
        assert levers.NEW_BUILD_GBE in values and levers.NEW_BUILD_GBE_PARTNERSHIP in values
        cases = expanded_cases(_config())
        assert "public_ownership@gbe_announced" in cases
        assert "public_ownership@gbe_announced_partnership_half" in cases
        specs = {s.case_id: s for s in configured_case_specs(_config())}
        gbe = specs["public_ownership@gbe_announced"]
        # P2's financing and contract levers; a public, market-bill case.
        assert gbe.settings["new_build_generation"] == levers.NEW_BUILD_GBE
        assert gbe.settings["public_return"] == "public_rate" and gbe.settings["cfd_strike_basis"] == "at_cost"
        assert gbe.is_public() and gbe.market_bill()
        assert specs["public_ownership@gbe_announced_partnership_half"].settings["new_build_generation"] == \
            levers.NEW_BUILD_GBE_PARTNERSHIP

    def test_p2_financing_levers_bite_and_acquisition_levers_do_not(self):
        bundle = {levers.LEVER_NEW_BUILD_GENERATION: levers.NEW_BUILD_GBE, levers.LEVER_PUBLIC_RETURN: "public_rate"}
        assert levers.LEVERS[levers.LEVER_PUBLIC_RATE_BASIS].applies("pwlb_standard", bundle)
        assert levers.LEVERS[levers.LEVER_SYSTEMATIC_PREMIUM].applies("social_flat_19", bundle)  # meeting notes (10 Nov 2026): none is the central
        assert levers.LEVERS[levers.LEVER_STATE_CAPEX_OVERRUN].applies("plus_15pct", bundle)
        assert not levers.LEVERS[levers.LEVER_LEGACY_FLEET].applies("acquire_all", bundle)
        assert not levers.LEVERS[levers.LEVER_DELIVERED_SHARE].applies("half", bundle)

    def test_an_owned_row_takes_its_part_later(self):
        for value in levers.GBE_VALUES:
            assert finance.ownership_share_for_settings({levers.LEVER_NEW_BUILD_GENERATION: value},
                                                        finance_vintage="new_build", owned=True) == 1.0


class TestParameters:
    def test_code_defaults_equal_the_config(self):
        block = _config()["lever_parameters"]
        assert cfs.lever_parameters(block)["gbe_announced"] == cfs.LEVER_PARAMETER_DEFAULTS["gbe_announced"]
        assert block["gbe_announced"] == cfs.LEVER_PARAMETER_DEFAULTS["gbe_announced"]

    def test_bad_parameters_are_refused(self):
        with pytest.raises(ValueError, match="gbe_announced takes"):
            cfs.lever_parameters({"gbe_announced": {"stake": 0.3}})
        with pytest.raises(ValueError, match="offshore_stake_share"):
            cfs.lever_parameters({"gbe_announced": {"offshore_stake_share": 0.0}})


def _rows():
    carrier = np.array(["wind_offshore", "wind_offshore", "solar_pv", "solar_pv", "solar_pv", "wind_onshore",
                        "wind_onshore", "Battery", "solar_pv", "wind_offshore"])
    vintage = np.array(["new_build"] * 8 + ["legacy", "new_build"])
    capacity = np.array([1000.0, 500.0, 100.0, 200.0, 300.0, 150.0, 250.0, 400.0, 999.0, 700.0])
    rows = SimpleNamespace(
        carrier=carrier, vintage=vintage, capacity=capacity, resolved=np.ones(10, dtype=bool),
        excluded=np.zeros(10, dtype=bool),
        # The last offshore row holds a pre-2025 CfD: outside P2's boundary.
        pre_policy_cfd=np.array([False] * 9 + [True]))
    rows.cells = lambda: [(c, 2028) for c in carrier]
    rows.order_keys = lambda: [f"{i:02d}" for i in range(10)]
    return rows


class TestSelection:
    def test_the_stake_cohort_and_the_local_power_pro_rata(self):
        gbe = dict(cfs.LEVER_PARAMETER_DEFAULTS["gbe_announced"], local_capacity_mw=500.0)
        out = cfs.gbe_selection(_rows(), gbe)
        # Offshore: the post-2025 cohort inside the boundary (not the pre-2025 CfD).
        assert out["offshore"].tolist() == [True, True] + [False] * 8
        assert out["offshore_mw"] == 1500.0
        # Local: 500 of the 1,000 MW cohort (legacy solar and the battery out), the same share of each cell.
        assert out["local_cohort_mw"] == pytest.approx(1000.0)
        assert out["local_share"] == pytest.approx(0.5)
        assert not out["local"][[0, 1, 7, 8, 9]].any()
        assert 300.0 <= out["local_mw"] <= 700.0

    def test_more_local_power_than_the_cohort_is_refused(self):
        with pytest.raises(ValueError, match="exceeds the post-2025 cohort"):
            cfs.gbe_selection(_rows(), cfs.LEVER_PARAMETER_DEFAULTS["gbe_announced"])


class TestOtherStages:
    def test_a_private_store_in_a_gbe_case_floors_at_the_private_rate(self):
        import pandas as pd

        from scripts.public_ownership.public_account import _floor_parts_original
        from scripts.public_ownership.finance import capital_recovery_factor

        frame = pd.DataFrame({"case_id": ["public_ownership@gbe_announced", "public_ownership"],
                              "new_build_generation": [levers.NEW_BUILD_GBE, levers.NEW_BUILD_PUBLIC],
                              "ownership_share": [0.0, 1.0], "annual_capital_recovery_gbp": [100.0, 100.0],
                              "wacc": [0.0883, 0.0883], "remaining_life_years": [30.0, 30.0],
                              "annual_fixed_om_gbp": [0.0, 0.0], "annual_variable_om_gbp": [0.0, 0.0]})
        rates = {"private": 0.045, "public_ownership": None}
        floor = _floor_parts_original(frame, rates)["row"]
        assert floor.iloc[0] == pytest.approx(100.0 / capital_recovery_factor(0.0883, 30) * capital_recovery_factor(0.045, 30))
        # P2's own store keeps the case's rate (null: its WACC).
        assert floor.iloc[1] == pytest.approx(100.0)

    def test_gbe_takes_on_none_of_sizewell_c(self):
        import pandas as pd

        from scripts.public_ownership import sizewell_c_rab as szc

        rows = pd.DataFrame({"finance_vintage": ["new_build"], "ownership_share": [1.0],
                             "new_build_generation": [levers.NEW_BUILD_GBE]})
        assert szc.state_new_build_share(rows) == 0.0
        assert szc.state_new_build_share(rows.assign(new_build_generation="public")) == 1.0
