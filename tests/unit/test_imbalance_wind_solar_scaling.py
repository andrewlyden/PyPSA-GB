"""The market's forecast-error cost follows wind and solar's share of demand (user decision, 16 Nov 2026)."""

import json
from pathlib import Path

import pytest

from scripts.consumer_bill.build_fes_bill_inputs import CP30_2030_DEMAND_EXCL_ELECTROLYSIS_GWH, wind_solar_share_factor

ROOT = Path(__file__).resolve().parents[2]


def _imbalance(year):
    doc = json.loads((ROOT / f"data/market/consumer_bill_benchmarks/{year}_pypsa_informed_assumptions.json").read_text(
        encoding="utf-8"))
    return doc["model_linked_rules"]["direct_fuel"]["procurement_bridge"]["allowances"]["imbalance"]


def test_2030_keeps_ofgems_rate():
    assert _imbalance(2030)["rate"] == pytest.approx(0.013104191358335954)
    assert "wind_solar_scaling" not in _imbalance(2030)


@pytest.mark.parametrize("year", [2040, 2045, 2050])
def test_a_later_year_scales_the_2030_rate_by_the_share_of_wind_and_solar(year):
    if not (ROOT / f"resources/FES/fes2025_holistic_transition_{year}_targets.json").exists():
        pytest.skip("the year's ES1 targets are not built here")
    entry = _imbalance(year)
    s = wind_solar_share_factor(year, ROOT)
    assert entry["wind_solar_scaling"]["rate_2030"] == pytest.approx(_imbalance(2030)["rate"])
    assert entry["rate"] == pytest.approx(entry["wind_solar_scaling"]["rate_2030"] * s["factor"])
    assert s["share_2030"] == pytest.approx(273.0 / (CP30_2030_DEMAND_EXCL_ELECTROLYSIS_GWH / 1e3))
    # A rate on the energy cost: the share of demand, never output alone (which would count demand growth twice).
    assert 1.0 < s["factor"] < 1.3
