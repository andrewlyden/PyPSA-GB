"""F27 (user, 24 Oct 2026): the literature channels and the forward sale at cost are declared beside the results."""

import pandas as pd

from scripts.public_ownership.declared_items import build

LITERATURE_CHANNELS = {"transition_effects", "state_operating_performance", "central_dispatch_upper_bound",
                       "compensation_floor", "systematic_risk_transfer", "capex_overruns", "price_state",
                       "attribution"}


def _items():
    rows = pd.DataFrame({"asset_id": ["x"], "carrier": ["CCGT"], "cfd_round": [""], "capacity_mw": [1.0]})
    return {i["id"]: i for i in build({}, rows, household_per_system=1e-8)["items"]}


def test_every_literature_channel_is_stated_as_a_limitation():
    items = _items()
    assert LITERATURE_CHANNELS <= set(items)
    for key in LITERATURE_CHANNELS:
        assert items[key]["kind"] == "limitation" and len(items[key]["why"]) > 60
    assert "Grout 2003" in items["systematic_risk_transfer"]["why"]
    assert "Sovacool" in items["capex_overruns"]["why"]
    assert "Cicala 2022" in items["central_dispatch_upper_bound"]["why"]
    assert "Energy Charter Treaty" in items["compensation_floor"]["why"]
    assert len(items) == len(set(items))


def test_the_forward_sale_at_cost_is_declared_at_its_researched_size():
    item = _items()["forward_sale_at_cost"]
    assert item["kind"] == "declared"
    assert item["gbp_per_household"] == [0.14, 0.43]
    assert item["gross_bill_line_cut_gbp_per_household"] == [2.35, 7.18]
