"""The purchase's scope options (user decisions, 10 Nov 2026).

Energy-from-waste is left out of the purchase of the existing fleet unless a case buys it
(``efw_acquisition: included``), and ``in_construction: left_private`` leaves Hinkley Point C out.
Only existing-fleet (legacy) rows are affected: a state-built energy-from-waste plant is not.
"""
import pandas as pd

from scripts.public_ownership import levers
from scripts.public_ownership.calculate_financing_scenarios import (
    acquisition_terms, case_purchase_exclusions, _plan_key,
)

MERGED = pd.DataFrame({
    "asset_id": ["Hinkley Point C::legacy", "Riverside EfW::legacy", "New EfW::new_build", "Walney_3"],
    "carrier": ["nuclear", "waste_to_energy", "waste_to_energy", "wind_offshore"],
    "finance_vintage": ["legacy", "legacy", "new_build", "legacy"],
})


def _excluded(**settings):
    return list(MERGED["asset_id"][case_purchase_exclusions(MERGED, acquisition_terms({}, settings))])


def test_the_central_leaves_energy_from_waste_out_and_buys_hinkley():
    assert levers.LEVERS["efw_acquisition"].default == "excluded"
    assert levers.LEVERS["in_construction"].default == "private"
    assert _excluded() == ["Riverside EfW::legacy"]


def test_the_options_buy_energy_from_waste_or_leave_hinkley_private():
    assert _excluded(efw_acquisition="included") == []
    assert _excluded(in_construction="left_private") == ["Hinkley Point C::legacy", "Riverside EfW::legacy"]
    assert _excluded(efw_acquisition="included", in_construction="left_private") == ["Hinkley Point C::legacy"]
    assert _excluded(in_construction="public_completion") == ["Riverside EfW::legacy"]


def test_each_option_is_its_own_purchase_plan():
    keys = {_plan_key(acquisition_terms({}, s), 0.027) for s in
            ({}, {"efw_acquisition": "included"}, {"in_construction": "left_private"})}
    assert len(keys) == 3
