from pathlib import Path
import json
import shutil

import pandas as pd
import pytest

from scripts.consumer_bill.validate_consumer_bill_input_bundle import validate_bundle


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data" / "market" / "ofgem_price_cap" / "2020_annual"


def _validate(config):
    return validate_bundle(
        BUNDLE,
        config,
        BUNDLE / "hedged_wholesale_inputs.csv",
        BUNDLE / "policy_components.csv",
        BUNDLE / "network_components.csv",
        BUNDLE / "smart_meter_components.csv",
        BUNDLE / "adjustment_components.csv",
    )


def test_historical_bundle_is_year_agnostic():
    result = _validate(
        {
            "scenario_id": "historical_example",
            "modelled_year": 2020,
            "consumer_bill": {"price_cap": {"input_mode": "historical_backtest"}},
        }
    )
    assert result["snapshot_id"] == "2020_annual"
    assert result["procurement_period_count"] == 3
    assert result["component_rows"]["policy"] > 0


def test_future_bundle_uses_same_schema_but_is_labelled_projection(tmp_path):
    root = tmp_path / "2030_projection"
    shutil.copytree(BUNDLE, root)
    metadata = json.loads((root / "metadata.json").read_text())
    metadata.update(snapshot_id="2030_projection", effective_from="2030-01-01", effective_to="2030-12-31")
    (root / "metadata.json").write_text(json.dumps(metadata))
    periods = pd.read_csv(root / "hedged_wholesale_inputs.csv")
    for column in ("delivery_start", "delivery_end"):
        periods[column] = periods[column].str.replace("2020", "2030")
    periods.to_csv(root / "hedged_wholesale_inputs.csv", index=False)
    for name in ("policy_components.csv", "network_components.csv", "smart_meter_components.csv", "adjustment_components.csv"):
        frame = pd.read_csv(root / name)
        for column in ("delivery_start", "delivery_end"):
            if column in frame:
                frame[column] = frame[column].str.replace("2020", "2030")
        frame.to_csv(root / name, index=False)
    result = validate_bundle(root, {"modelled_year": 2030, "consumer_bill": {"price_cap": {"input_mode": "future_projection"}}},
        *[root / name for name in ("hedged_wholesale_inputs.csv", "policy_components.csv", "network_components.csv", "smart_meter_components.csv", "adjustment_components.csv")])
    assert result["input_mode"] == "future_projection"
    assert result["future_projection_rule"]
    assert result["is_statutory_cap_reproduction"] is False


def test_input_mode_is_explicit_not_inferred_from_the_year():
    result = _validate(
        {
            "scenario_id": "reanalysis_example",
            "modelled_year": 2020,
            "consumer_bill": {"price_cap": {"input_mode": "future_projection"}},
        }
    )
    assert result["input_mode"] == "future_projection"


@pytest.mark.parametrize("mode", ["historical_backtest", "future_projection"])
def test_old_bundle_cannot_masquerade_as_another_year(mode):
    with pytest.raises(ValueError, match="not modelled_year"):
        _validate({"modelled_year": 2030, "consumer_bill": {"price_cap": {"input_mode": mode}}})
