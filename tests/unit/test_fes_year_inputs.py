"""A new FES year's inputs by script (user, 11 Nov 2026): they reproduce the 2040 files made by hand."""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.sources import fes_year_inputs as f

TARGETS_2040 = Path("resources/FES/fes2025_holistic_transition_2040_targets.json")
needs_2040 = pytest.mark.skipif(not TARGETS_2040.exists(), reason="needs the 2040 FES targets")


def test_guarantee_shares_follow_the_trailing_windows():
    assert f.guarantee_shares(2040) == {"transmission": 1.0, "distribution": 0.8824}   # 15 of 17 years from 2026
    assert f.guarantee_shares(2050) == {"transmission": 1.0, "distribution": 1.0}
    assert f.guarantee_shares(2035) == {"transmission": round(10 / 14, 4), "distribution": round(10 / 17, 4)}


@needs_2040
def test_inputs_reproduce_the_2040_hand_files():
    t = f._targets(2040)
    made = pd.read_csv(io.StringIO(f.ancillary(2040, t)))
    hand = pd.read_csv("data/public_ownership/financing/ancillary_services_2040.csv")
    values = [c for c in hand.columns if c != "projection_basis"]
    assert list(made.columns) == list(hand.columns)
    assert made[values].astype(str).equals(hand[values].astype(str))
    efw = json.loads(f.efw(2040, t))
    assert "about 18.3 Mt" in efw["projection_2040"]["fes_context"] and "projection_2050" not in efw
    assert '"2040": 96.508' in f.rav(2040)
    g = pd.read_csv(io.StringIO(f.guarantee(2040))).set_index("setting")
    assert g.loc["new_debt_low", "distribution_share"] == pytest.approx(0.8824)


def test_seed_rows_stop_at_the_year(tmp_path, monkeypatch):
    fin = tmp_path / f.FIN
    fin.mkdir(parents=True)
    pd.DataFrame({"key": ["a", "b"], "build_year": [2035, 2045], "source": ["x", "x"]}).to_csv(
        fin / "strike_calibration_2050.csv", index=False)
    pd.DataFrame({"carrier": ["Battery"], "source": ["x"]}).to_csv(fin / "capture_prices_2050.csv", index=False)
    monkeypatch.setattr(f, "ROOT", tmp_path)
    f.seeds(2040)
    seeded = pd.read_csv(fin / "strike_calibration_2040.csv")
    assert list(seeded["key"]) == ["a"] and seeded["source"].str.startswith("SEED").all()
    assert pd.read_csv(fin / "capture_prices_2040.csv")["source"].str.startswith("SEED").all()


@needs_2040
def test_parameters_reproduce_the_2040_rows():
    p = f.parameters(2040)
    assert p["hydrogen_end_of_window_value_gbp_per_mwh_h2"] == pytest.approx(36.49)
    assert p["hydrogen_reserve_twh"] == pytest.approx(2.22)
    assert (p["hydrogen_supply_price_gbp_per_mwh_h2"], p["ccs_gas_efficiency"], p["ccs_gas_variable_om_gbp_per_mwh"]) == (64.0, 0.47, 18.09)
