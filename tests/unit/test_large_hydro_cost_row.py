"""Large hydro is costed on its own IRENA-based row; small hydro stays on the BEIS 5-16 MW Hydropower row."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / "data" / "public_ownership" / "financing"


def test_large_hydro_has_its_own_cost_row_below_small_hydro():
    costs = pd.read_csv(FIN / "technology_costs.csv").set_index("technology")
    assert costs.loc["Hydropower", "capex_power_gbp_per_kw"] == 4120.24
    # IRENA 2023 Europe large hydro: USD 2,318/kW (2023) -> GBP 2024 real
    expected = 2318 / 1.2435 * (97.72656 / 95.24944)
    assert abs(costs.loc["Large hydro", "capex_power_gbp_per_kw"] - expected) < 0.5
    assert costs.loc["Large hydro", "capex_power_gbp_per_kw"] < costs.loc["Hydropower", "capex_power_gbp_per_kw"]


def test_crosswalk_costs_large_hydro_on_large_hydro_and_rates_on_hydropower():
    base = pd.read_csv(FIN / "pypsa_finance_technology_crosswalk.csv").fillna("").set_index("pypsa_carrier")
    assert base.loc["large_hydro", "cost_technology"] == "Large hydro"
    assert base.loc["large_hydro", "workbook_technology"] == "Hydropower"
    # small_hydro and hydro have no base row of their own: they are never costed on Large hydro
    assert "Large hydro" not in set(base.drop(index="large_hydro")["cost_technology"])


def test_2050_supplement_keeps_large_hydro_on_the_vintage_hydropower_row():
    sup = pd.read_csv(FIN / "pypsa_finance_technology_crosswalk_2050.csv").fillna("")
    row = sup[sup["pypsa_carrier"] == "large_hydro"].iloc[0]
    assert row["cost_technology"] == "Hydropower"
    vintage = pd.read_csv(FIN / "technology_costs_vintage.csv", usecols=["technology"])
    assert "Large hydro" not in set(vintage["technology"])
    assert "Hydropower" in set(vintage["technology"])
