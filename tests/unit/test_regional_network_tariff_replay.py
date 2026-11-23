import pandas as pd
import pytest

from scripts.consumer_bill.build_regional_network_tariff_replay import (
    PES_TO_OFGEM_REGION,
    annualise_regional_network_components,
)


def test_regional_replay_uses_domestic_meter_weights_not_equal_regions():
    weights = pd.DataFrame({
        "pes_id": list(PES_TO_OFGEM_REGION),
        "domestic_consumption_weight": [1 / 14] * 14,
        "domestic_meter_weight": [0.5] + [0.5 / 13] * 13,
    })
    period = pd.DataFrame({
        "pes_id": list(PES_TO_OFGEM_REGION),
        "ofgem_region": list(PES_TO_OFGEM_REGION.values()),
        "component": ["TNUoS"] * 14,
        "gbp_per_3100kwh_ex_vat": [10.0] + [2.0] * 13,
        "calendar_weight_2020": [1.0] * 14,
    })

    annual, summary = annualise_regional_network_components(period, weights)

    assert annual.loc[annual["pes_id"].eq(10), "annual_gbp_per_3100kwh_ex_vat"].item() == 10.0
    assert summary["meter_weighted_gbp_per_3100kwh_ex_vat"]["TNUoS"] == pytest.approx(6.0)
    assert summary["consumption_weighted_gbp_per_3100kwh_ex_vat"]["TNUoS"] == pytest.approx(2 + 8 / 14)


def test_regional_replay_requires_all_pes_areas():
    weights = pd.DataFrame({
        "pes_id": [10], "domestic_consumption_weight": [1.0], "domestic_meter_weight": [1.0]
    })
    period = pd.DataFrame({
        "pes_id": [10], "ofgem_region": ["Eastern"], "component": ["TNUoS"],
        "gbp_per_3100kwh_ex_vat": [1.0], "calendar_weight_2020": [1.0],
    })

    with pytest.raises(ValueError, match="14 PES"):
        annualise_regional_network_components(period, weights)
