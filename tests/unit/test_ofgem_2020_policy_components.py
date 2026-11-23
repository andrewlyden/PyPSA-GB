from pathlib import Path

import pandas as pd
import pytest

from scripts.market.extract_ofgem_2020_policy_components import validate_against_cost_stack
from scripts.market.extract_ofgem_2020_network_components import validate_against_cost_stack as validate_network
from scripts.market.build_consumer_bill_component_register import build_component_register
from tests.helpers import requires_outputs


ROOT = Path(__file__).resolve().parents[2]


def test_annex_4_extract_reconciles_and_exposes_2020_cfd_allowance():
    components = pd.read_csv(ROOT / "data/market/ofgem_price_cap/2020_annual/policy_components.csv")
    validate_against_cost_stack(
        components,
        ROOT / "data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv",
    )
    cfd = components.loc[components["component"].eq("CfD"), "annualised_gbp_per_3100kwh_ex_vat"].sum()
    assert cfd == pytest.approx(27.544489079832335)


@requires_outputs("resources/analysis/Validation_2020_retail_cost_stack.csv")
def test_detailed_register_replaces_network_and_smart_meter_aggregates():
    cost_stack = pd.read_csv(ROOT / "resources/analysis/Validation_2020_retail_cost_stack.csv")
    policy = pd.read_csv(ROOT / "data/market/ofgem_price_cap/2020_annual/policy_components.csv")
    network = pd.read_csv(ROOT / "data/market/ofgem_price_cap/2020_annual/network_components.csv")
    smart = pd.read_csv(ROOT / "data/market/ofgem_price_cap/2020_annual/smart_meter_components.csv")
    validate_network(network, ROOT / "data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv")

    register = build_component_register(cost_stack, policy, network, smart)

    assert set(register.loc[register["stack_group"].eq("network"), "component"]) == {
        "Ofgem Annex 3 — TNUoS", "Ofgem Annex 3 — DUoS fixed charges",
        "Ofgem Annex 3 — DUoS unit charges", "Ofgem Annex 3 — BSUoS",
    }
    assert "Smart-metering net cost change" not in set(register["component"])
    assert register["annual_gbp_per_3100kwh_ex_vat"].sum() == pytest.approx(
        cost_stack.loc[~cost_stack["stack_group"].eq("total"), "annual_gbp_per_3100kwh_ex_vat"].sum()
    )
