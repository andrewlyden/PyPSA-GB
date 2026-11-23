"""One bill-input builder for every long-range FES year.

``build_fes_bill_inputs`` must reproduce the 2050 builder (task Y13) byte for byte, and the committed
2040 files must be its output.
"""

import json
from pathlib import Path

import pytest

from scripts.consumer_bill import build_fes_bill_inputs as g

ROOT = Path(__file__).resolve().parents[2]
NEEDS = [ROOT / "data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx",
         ROOT / "resources/FES/FES_2025_data.csv", ROOT / g.NETWORK_BLOCKS]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDS), reason="needs the FES workbook and built inputs")


def test_the_generic_builder_reproduces_the_2050_builder_exactly():
    from scripts.consumer_bill import build_2050_bill_inputs as old

    if not (ROOT / "resources/market/capacity_market/cm_calendar_2050_summary.json").exists():
        pytest.skip("needs the 2050 Capacity Market build")
    new, ref = g.build(2050, ROOT), old.build(ROOT)
    for key in ref:
        assert g.render(new[key]) == old.render(ref[key]), key


def test_the_committed_2040_files_are_the_builders_output():
    if not (ROOT / "resources/market/capacity_market/cm_calendar_2040_summary.json").exists():
        pytest.skip("needs the 2040 Capacity Market build")
    documents = g.build(2040, ROOT)
    paths = g.outputs(2040)
    documents["recovery"] = g.keep_committed_refresh(documents["recovery"], ROOT / paths["recovery"])
    for key, path in paths.items():
        assert (ROOT / path).read_text(encoding="utf-8") == g.render(documents[key]), path
    data = json.loads((ROOT / paths["assumptions"]).read_text(encoding="utf-8"))
    assert data["benchmark"]["bill_calendar_year"] == 2040
    assert "FES25_HT_2040" in data["model_linked_rules"]["cfd"]["strike_table"]
    assert data["sourced_externals"]["remaining_ccs"]["value"] == 0.0   # NZT live, settled on the dpa line


def test_year_text_moves_the_year_and_keeps_the_2050_references():
    doc = {"calendar_2050_cost_gbp": "FY2049/50 and FY2050/51; FES 2050 decision 60; support_regime_2050.json; 2050",
           "fixed": "to " + g.FIXED_2050}
    out = g.year_text(doc, 2045)
    assert out == {"calendar_2045_cost_gbp": "FY2044/45 and FY2045/46; FES 2050 decision 60; support_regime_2050.json; 2045",
                   "fixed": "to 2050"}
    assert g.year_text(doc, 2050)["fixed"] == "to 2050"


def test_years_it_cannot_represent_are_refused():
    with pytest.raises(ValueError, match="Renewables Obligation"):
        g.build(2035, ROOT)
