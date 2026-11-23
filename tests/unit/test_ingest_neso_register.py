"""Staged NESO register rows describe one asset, not several."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.interconnectors.combine_datasets import combine_datasets
from scripts.interconnectors.ingest_neso_register import standardize_neso_data

REGISTER = Path("data/interconnectors/NESO_interconnector_register.csv")


def _register():
    columns = ["Project Name", "Connection Site", "Stage", "MW Import - Total", "MW Export - Total",
               "Project Status", "HOST TO", "MW Effective From"]
    rows = [
        ["Auchencrosh (interconnector CCT)", "Auchencrosh 275kV", 1, 475, 500, "Built", "SPT", "01/04/2022"],
        ["Auchencrosh (interconnector CCT)", "Auchencrosh 275kV", 2, 500, 500, "Built", "SPT", "01/08/2022"],
        ["Britned", "Grain 400kV Substation", None, 1200, 1200, "Built", "NGET", None],
        ["Cronos", "Kemsley 400kV Substation", None, 1400, 1400, "Scoping", "NGET", "31/10/2029"],
    ]
    return pd.DataFrame(rows, columns=columns)


def test_staged_rows_collapse_to_the_final_cumulative_stage():
    out = standardize_neso_data(_register())
    moyle = out[out["name"] == "Auchencrosh (interconnector CCT)"]
    assert len(moyle) == 1
    assert moyle["capacity_mw"].iloc[0] == 500
    assert set(out["name"]) == {"Auchencrosh (interconnector CCT)", "Britned"}


def test_moyle_appears_once_after_the_dukes_merge():
    neso = standardize_neso_data(_register())
    dukes = pd.DataFrame({
        "name": ["Moyle", "BritNed"],
        "capacity_mw": [500.0, 1000.0],
        "counterparty_country": ["Northern Ireland", "Netherlands"],
        "commissioning_year": [2002, 2011],
        "source": ["DUKES", "DUKES"],
    })
    combined = combine_datasets(dukes, neso)
    ni = combined[combined["counterparty_country"] == "Northern Ireland"]
    assert ni["name"].tolist() == ["Moyle"]
    assert ni["capacity_mw"].iloc[0] == pytest.approx(500.0)


def test_neuconnect_is_germany_and_britned_the_netherlands():
    # NeuConnect was listed under both countries and matched the Netherlands first,
    # so it was priced on the Dutch zone (26 Sep 2026).
    from scripts.interconnectors.enrich_locations import get_international_coordinates
    from scripts.interconnectors.fetch_entsoe_prices import COUNTRY_ZONE_MAP
    from scripts.interconnectors.ingest_neso_register import extract_counterparty_country

    assert extract_counterparty_country("NeuConnect Interconnector") == "Germany"
    assert extract_counterparty_country("Britned") == "Netherlands"
    assert COUNTRY_ZONE_MAP["Germany"] == "DE_LU"
    lat, lon = get_international_coordinates("Germany")
    assert 53.0 < lat < 54.0 and 7.5 < lon < 8.5      # Fedderwarden, Lower Saxony


@pytest.mark.skipif(not REGISTER.exists(), reason="NESO register not available")
def test_shipped_register_has_one_row_per_operational_project():
    out = standardize_neso_data(pd.read_csv(REGISTER))
    assert not out.duplicated(subset=["name", "landing_point_gb"]).any()
