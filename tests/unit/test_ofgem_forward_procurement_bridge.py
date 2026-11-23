import pandas as pd
import pytest

from scripts.consumer_bill.build_ofgem_forward_procurement_bridge import (
    build_ofgem_forward_procurement_bridge,
)


def test_forward_procurement_reconstructs_observed_components_without_spot_pass_through():
    register = pd.DataFrame({
        "component": ["Published seasonal base forward allowance", "Published seasonal peak forward allowance"],
        "annual_gbp_per_3100kwh_ex_vat": [20.0, 10.0],
    })
    periods = pd.DataFrame({
        "cap_period": ["winter", "summer", "winter", "summer"],
        "delivery_start": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
        "delivery_end": ["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"],
        "component": ["seasonal_base_forward_index", "seasonal_base_forward_index", "seasonal_peak_forward_index", "seasonal_peak_forward_index"],
        "gbp_per_benchmark_customer_ex_vat": [20.0, 20.0, 10.0, 10.0],
    })
    wholesale = pd.DataFrame({
        "cap_period": ["winter", "summer"], "modelled_demand_mwh": [1.0, 1.0],
        "modelled_demand_weighted_clearing_price_gbp_per_mwh": [30.0, 40.0],
        "published_forward_index_gbp_per_mwh": [50.0, 60.0],
    })
    bridge, summary = build_ofgem_forward_procurement_bridge(register, periods, wholesale)
    assert bridge["candidate_allowance_gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(30.0)
    assert bridge["difference_gbp_per_3100kwh_ex_vat"].abs().max() == pytest.approx(0.0)
    assert bridge["recovery_status"].eq("reproduced_from_observed_ofgem_forward_inputs").all()
    assert summary["pypsa_diagnostic"]["treatment"].endswith("the household bill")
