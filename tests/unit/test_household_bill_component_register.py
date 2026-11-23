from pathlib import Path

import pandas as pd

from scripts.market.build_household_bill_component_register import build_household_register
from scripts.market.consumer_bill_breakdown import build_breakdown
from tests.helpers import requires_outputs


ROOT = Path(__file__).resolve().parents[2]


@requires_outputs(
    ROOT / "resources/analysis/Validation_2020_ofgem_benchmark_component_register.csv",
    ROOT / "resources/analysis/Validation_2020_consumer_bill.csv",
)
def test_household_register_reconciles_to_selected_bill_and_breakdown():
    detailed = pd.read_csv(ROOT / "resources/analysis/Validation_2020_ofgem_benchmark_component_register.csv")
    bills = pd.read_csv(ROOT / "resources/analysis/Validation_2020_consumer_bill.csv")
    household, summary = build_household_register(detailed, bills)

    assert abs(household["annual_gbp_ex_vat"].sum() - bills.loc[0, "consumer_bill_gbp_inc_vat"]) < 1e-6
    assert abs(household.loc[household["stack_group"].eq("vat"), "annual_gbp_ex_vat"].sum() - bills.loc[0, "vat_gbp"]) < 1e-6
    assert summary["standing_charge_ex_vat"] == bills.loc[0, "standing_charge_gbp_ex_vat"]

    grouped, grouped_summary = build_breakdown(bills, household)
    assert not grouped.empty
    assert abs(grouped_summary["published_cost_stack"]["total_gbp_inc_vat"] - bills.loc[0, "consumer_bill_gbp_inc_vat"]) < 1e-6
    assert grouped["component"].str.contains("CfD").any()
