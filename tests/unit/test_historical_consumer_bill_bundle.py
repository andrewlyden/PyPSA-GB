import hashlib
from pathlib import Path

import openpyxl
import pandas as pd
import yaml

from scripts.consumer_bill.build_historical_consumer_bill_bundle import COMPONENT_CODES, compile_bundle
from scripts.consumer_bill.validate_consumer_bill_input_bundle import validate_bundle


ROOT = Path(__file__).resolve().parents[2]


def _write_minimal_ofgem_workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    other = workbook.active
    other.title = "ElecSingle_Other_3100kWh"
    nil = workbook.create_sheet("ElecSingle_Other_Nil")
    for sheet in (other, nil):
        sheet.cell(12, 7).value = "January 2024 - March 2024"
    row = 15
    for number in range(1, 15):
        region, total = f"Region {number}", 0.0
        for index, code in enumerate(COMPONENT_CODES, start=1):
            value = 10.0 + index
            other.cell(row, 2).value = "Category"
            other.cell(row, 3).value = code
            other.cell(row, 4).value = region
            other.cell(row, 7).value = value
            total += value
            row += 1
        other.cell(row, 2).value = "Total"
        other.cell(row, 3).value = f"Total_{region}"
        other.cell(row, 4).value = region
        other.cell(row, 7).value = total
        nil.cell(row, 3).value = f"Total_{region}"
        nil.cell(row, 4).value = region
        nil.cell(row, 7).value = 20.0
        row += 1
    workbook.save(path)


def test_compiler_builds_valid_bundle_from_pinned_local_source(tmp_path):
    workbook = tmp_path / "cap.xlsx"
    _write_minimal_ofgem_workbook(workbook)
    manifest = {
        "calendar_year": 2024,
        "raw_cache_dir": str(tmp_path / "raw"),
        "sources": [{
            "cap_period": "2024_Q1",
            "delivery_start": "2024-01-01",
            "delivery_end": "2024-12-31",
            "workbook_period_label": "January 2024 - March 2024",
            "filename": "cap.xlsx",
            "url": workbook.as_uri(),
            "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        }],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    bundle = tmp_path / "bundle"
    compile_bundle(manifest_path, bundle)
    result = validate_bundle(
        bundle,
        {"scenario_id": "test", "modelled_year": 2024, "consumer_bill": {"price_cap": {"input_mode": "historical_backtest"}}},
        bundle / "hedged_wholesale_inputs.csv",
        bundle / "policy_components.csv",
        bundle / "network_components.csv",
        bundle / "smart_meter_components.csv",
        bundle / "adjustment_components.csv",
    )
    assert result["rate_rows"] == 14
    assert result["procurement_period_count"] == 1
    assert (bundle / "source_provenance.json").exists()


def test_2020_legacy_layout_manifest_builds_full_period_ledger(tmp_path):
    manifest = ROOT / "data" / "market" / "consumer_bill_sources" / "2020.yaml"
    bundle = tmp_path / "2020_bundle"
    compile_bundle(manifest, bundle)

    tariffs = pd.read_csv(bundle / "period_tariffs.csv")
    ledger = pd.read_csv(bundle / "component_ledger.csv")
    assert tariffs["cap_period"].nunique() == 3
    assert tariffs["region"].nunique() == 14
    assert set(tariffs["delivery_days"]) == {91, 92, 183}
    assert ledger["component_id"].notna().all()
    wholesale = pd.read_csv(bundle / "wholesale_components.csv")
    assert wholesale["cap_period"].nunique() == 3
    assert wholesale["detail_status"].eq("published_aggregate_cost_model_row").all()
