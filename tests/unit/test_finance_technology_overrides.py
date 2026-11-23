"""Storage costed on its own technology in every year (user, 15 Nov 2026)."""

import pandas as pd
import pytest

from scripts.public_ownership import build_asset_finance_register as bafr

TABLE = pd.read_csv("data/public_ownership/financing/finance_technology_overrides.csv")


def _register(ids):
    return pd.DataFrame({"asset_id": ids, "carrier": "LAES", "finance_technology": "Novel long duration energy storage (LDES)",
                         "finance_eligible": True})


def test_cp30_pumped_storage_is_costed_as_pumped_storage_and_liquid_air_is_not_touched():
    ids = ["CP30_LAES_committed_CoireGlas_576::pre_legislation", "CP30_LAES_existing_Dinorwig_572::legacy",
           "CP30_LAES_committed_HighviewHunterston_580::pre_legislation", "CP30_LAES_committed_CRUA_expansion_578",
           "CP30_LAES_committed_LochNaCathrach_577", "CP30_LAES_committed_Glenmuckloch_579",
           "CP30_LAES_existing_Ffestiniog_574::legacy", "CP30_LAES_existing_Foyers_575::legacy",
           "CP30_LAES_existing_Cruachan_573::legacy"]
    out = bafr.apply_finance_technology_overrides(_register(ids), TABLE, model_year=2030)
    phes = out["cost_technology"].eq("Pumped hydro energy storage (PHES) - 20h")
    assert phes.tolist() == [True, True, False, True, True, True, True, True, True]
    assert out.loc[2, "finance_technology"] == "Novel long duration energy storage (LDES)"   # Highview: liquid air
    # Another model year is untouched (2040 and 2050 cost pumped storage on their own crosswalk rows).
    later = bafr.apply_finance_technology_overrides(_register(ids), TABLE, model_year=2040)
    assert later["finance_technology"].eq("Novel long duration energy storage (LDES)").all()


def test_a_renamed_project_is_refused():
    ids = ["CP30_LAES_committed_CoireGlass_576", "CP30_LAES_existing_Dinorwig_572::legacy"]
    with pytest.raises(ValueError, match="CoireGlas"):
        bafr.apply_finance_technology_overrides(_register(ids), TABLE, model_year=2030)


def test_every_override_cost_row_exists_in_the_2030_cost_table():
    costs = pd.read_csv("data/public_ownership/financing/technology_costs.csv")
    assert set(TABLE["cost_technology"]) <= set(costs["technology"])
    crosswalk = pd.read_csv("data/public_ownership/financing/pypsa_finance_technology_crosswalk.csv")
    laes = crosswalk.loc[crosswalk["pypsa_carrier"].eq("LAES")].iloc[0]
    assert laes["cost_technology"] in set(costs["technology"])
