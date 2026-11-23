import pytest

from scripts.market.fetch_neso_cp2030_annex4_costs import (
    THERM_TO_MWH,
    extract_annex4_assumptions,
)


def test_extract_annex4_gas_and_total_carbon_assumptions():
    text = (
        "A gas price of around 100p/therm has been assumed for the economic "
        "impact analysis. For the clean power analysis we modelled a UK carbon "
        "price that is @25/tCO2 above this price, at @147/tCO2 in 2030. The "
        "Carbon Price Support scheme has been in place for a decade, currently "
        "set at @18/tCO2, raising the UK carbon price."
    )

    extracted = extract_annex4_assumptions(text)

    assert extracted["fuel_prices"]["gas"] == pytest.approx(
        1.0 / THERM_TO_MWH
    )
    assert extracted["carbon"] == {
        "uk_ets": 129.0,
        "carbon_price_support": 18.0,
    }
    assert extracted["extracted_values"]["total_uk_carbon_gbp_per_tco2"] == 147.0

