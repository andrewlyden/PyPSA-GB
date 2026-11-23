"""The cap rule weights its quarterly base/peak blends by Ofgem's PC1 demand (6 Oct 2026)."""

import json
from pathlib import Path

import pytest

from scripts.consumer_bill import project_future_bill as bill

ROWS = [{"quarter": q, "base_price_gbp_per_mwh": base, "peak_price_gbp_per_mwh": base + 10.0,
         "domestic_volume_weight": 0.25} for q, base in ((1, 60.0), (2, 40.0), (3, 60.0), (4, 20.0))]


def test_declared_weights_replace_the_household_volume():
    weights = {1: 0.3, 2: 0.2, 3: 0.2, 4: 0.3}
    price, detail = bill.ofgem_cap_rule_price(ROWS, 0.7, 0.3, quarter_weights=weights)
    blends = [row["base_price_gbp_per_mwh"] + 3.0 for row in ROWS]
    assert price == pytest.approx(sum(b * weights[q] for b, q in zip(blends, (1, 2, 3, 4))))
    assert detail[0]["quarter_weight"] == pytest.approx(0.3)
    assert detail[0]["quarter_weight_basis"] == "declared_quarter_weights"


def test_without_declared_weights_the_household_volume_is_used():
    price, detail = bill.ofgem_cap_rule_price(ROWS, 0.7, 0.3)
    assert price == pytest.approx(sum(row["base_price_gbp_per_mwh"] + 3.0 for row in ROWS) / 4)
    assert detail[0]["quarter_weight_basis"] == "household_profile_volume"


def test_weights_must_cover_the_priced_quarters():
    with pytest.raises(ValueError, match="quarter_weights cover"):
        bill.ofgem_cap_rule_price(ROWS, 0.7, 0.3, quarter_weights={1: 0.5, 2: 0.5})


def test_the_2030_bill_declares_ofgems_latest_pc1_weights():
    assumptions = json.loads(Path("data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumptions.json")
                             .read_text(encoding="utf-8"))
    shaping = bill.shaping_settings(assumptions["model_linked_rules"]["direct_fuel"])
    weights = shaping["quarter_weights"]["weights"]
    assert set(weights) == {1, 2, 3, 4}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[1] > weights[2] and weights[4] > weights[3]      # winter-heavy, as PC1 is
    assert "3b Demand" in shaping["quarter_weights"]["source"]


def test_shaping_settings_refuse_weights_that_do_not_sum_to_one():
    block = {"shaping": {"method": "ofgem_cap_rule", "base_share": 0.7, "peak_share": 0.3,
                         "seasonal_to_monthly_rate": 0.0, "monthly_to_hourly_rate": 0.0,
                         "quarter_weights": {"weights": {"q1": 0.3, "q2": 0.3, "q3": 0.3, "q4": 0.3}}}}
    with pytest.raises(ValueError, match="not one"):
        bill.shaping_settings(block)
