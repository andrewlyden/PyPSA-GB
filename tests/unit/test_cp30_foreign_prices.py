"""Item F13: foreign prices on CP30's fuel basis (scripts/interconnectors/cp30_foreign_prices.py)."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.interconnectors import add_to_network as atn
from scripts.interconnectors.cp30_foreign_prices import (
    DEFAULT_FOREIGN_NEW_CCGT_GBP,
    DEFAULT_TYNDP_NEW_CCGT_EUR,
    build_cp30_basis_prices,
    rebase_ratio,
)

NODES = ["UK00", "UKNI", "IE00", "FR00", "BE00", "NL00", "DE00", "DKW1", "NOS0"]


def _tyndp_file(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    hours = 8736
    frame = pd.DataFrame({"time": [f"h{k}" for k in range(hours)]})
    for node in NODES:
        price = np.where(rng.random(hours) < 0.2, 0.0, rng.choice([82.0, 84.0, 87.0, 30.0], hours))
        frame[f"mc_{node}"] = price
    frame["UK00_wind_offshore_mw"] = rng.uniform(0, 20000, hours)
    frame["UK00_dumped_mw"] = 0.0
    path = tmp_path / "tyndp.csv"
    frame.to_csv(path, index=False)
    return path, frame


def _wind_file(tmp_path, seed=1):
    index = pd.date_range("2013-01-01", periods=8760, freq="h")
    wind = pd.DataFrame({"a": np.random.default_rng(seed).uniform(0, 1, 8760)}, index=index)
    path = tmp_path / "wind.csv"
    wind.to_csv(path)
    return path


def test_defaults_map_new_ccgt_plateau_to_cp30_basis():
    # GB's best CCGT (103.0) less the UK-EU carbon gap (7 GBP/t x 0.34 t/MWh).
    assert DEFAULT_FOREIGN_NEW_CCGT_GBP == pytest.approx(103.0 - 7.0 * 0.34, abs=1e-6)
    assert rebase_ratio(DEFAULT_TYNDP_NEW_CCGT_EUR, DEFAULT_FOREIGN_NEW_CCGT_GBP) == pytest.approx(100.62 / 82.0)
    with pytest.raises(ValueError):
        rebase_ratio(0.0, 100.0)


def test_build_keeps_rebased_mean_and_zero_floor(tmp_path):
    tyndp, frame = _tyndp_file(tmp_path)
    out = build_cp30_basis_prices(tyndp, _wind_file(tmp_path), tmp_path / "p.csv", tmp_path / "a.json")
    ratio = rebase_ratio(DEFAULT_TYNDP_NEW_CCGT_EUR, DEFAULT_FOREIGN_NEW_CCGT_GBP)
    assert len(out) == 8760
    assert list(out.columns) == ["France", "Belgium", "Netherlands", "Germany", "Denmark", "Norway", "Ireland"]
    assert (out.min() >= -1e-9).all()  # TYNDP has no negative bids; the wind swap is held at the floor
    # The annual level is the rebased TYNDP level (31 December repeats 24 December, so compare on the extended year).
    expected_fr = np.concatenate([frame["mc_FR00"].to_numpy(), frame["mc_FR00"].to_numpy()[-168:-144]]).mean() * ratio
    assert out["France"].mean() == pytest.approx(expected_fr, rel=1e-9)
    audit = json.loads((tmp_path / "a.json").read_text())
    assert audit["gbp_per_tyndp_eur"] == pytest.approx(ratio, rel=1e-6)


def test_cp30_basis_is_an_own_level_source():
    assert "cp30_basis" in atn.OWN_LEVEL_PRICE_SOURCES
    assert "tyndp" in atn.OWN_LEVEL_PRICE_SOURCES


# --- method refit_2013 (7 Oct 2026) ---------------------------------------------------------------

from scripts.interconnectors.cp30_foreign_prices import (  # noqa: E402
    OUTPUT_ZONE,
    PRICE_NODES,
    REFIT_TECHS,
    REFIT_ZONES,
    build_cp30_refit_prices,
)


def _refit_inputs(tmp_path, seed=3):
    """Synthetic TYNDP zonal file whose prices follow regional residual load, and 2013 capacity factors."""
    rng = np.random.default_rng(seed)
    n09 = 8736
    cols = {}
    wind = rng.uniform(0, 1, n09)
    for z in REFIT_ZONES:
        cols[f"{z}|demand"] = 1000.0 + 200.0 * np.sin(np.arange(n09) / 24.0 * 2 * np.pi)
        cols[f"{z}|onwind"] = 800.0 * wind
        cols[f"{z}|offwind"] = 0.0 * wind
        cols[f"{z}|solar"] = 0.0 * wind
    rl = 1000.0 - 800.0 * wind
    price = np.where(rl > 600, 82.0, np.where(rl > 300, 40.0, 0.0))
    for nodes in PRICE_NODES.values():
        for node in nodes:
            cols[f"price|{node}"] = price
    zonal = pd.DataFrame(cols)
    zonal_path = tmp_path / "zonal.csv"
    zonal.to_csv(zonal_path, index_label="tyndp_hour")
    idx = pd.date_range("2013-01-01", periods=8760, freq="h")
    cf = pd.DataFrame({f"{z}|onwind": rng.uniform(0, 1, 8760) for z in REFIT_ZONES}, index=idx)
    cf_path = tmp_path / "cf.csv"
    cf.to_csv(cf_path)
    shape = pd.DataFrame({"Ireland": np.full(8784, 90.0)},
                         index=pd.date_range("2024-01-01", periods=8784, freq="h", tz="UTC"))
    shape_path = tmp_path / "shape.csv"
    shape.to_csv(shape_path)
    return zonal_path, cf_path, shape_path


def test_refit_predicts_tyndp_plateaus_on_new_weather_and_rebases(tmp_path):
    zonal, cf, shape = _refit_inputs(tmp_path)
    out = build_cp30_refit_prices(zonal, cf, tmp_path / "p.csv", tmp_path / "a.json", ireland_shape_file=shape,
                                  foreign_new_ccgt_gbp=102.9, cv_folds=0)
    ratio = 102.9 / 82.0
    assert len(out) == 8760 and list(out.columns) == list(PRICE_NODES)
    # The learnt response maps the top plateau to 82 EUR, rebased by the ratio.
    assert out["France"].max() == pytest.approx(82.0 * ratio, rel=0.06)   # boosted fit of a step: a few % overshoot
    assert out["France"].min() >= 0.0
    # The SEM column is the 2024 shape plus the IC1 offset, not the refit.
    assert out["Ireland"].to_numpy() == pytest.approx(97.5)
    audit = json.loads((tmp_path / "a.json").read_text())
    assert audit["method"] == "refit_2013" and audit["ireland_source"] == "entsoe_2024"


def test_refit_rejects_unknown_ireland_source(tmp_path):
    zonal, cf, shape = _refit_inputs(tmp_path)
    with pytest.raises(ValueError):
        build_cp30_refit_prices(zonal, cf, tmp_path / "p.csv", tmp_path / "a.json", ireland_source="flat",
                                cv_folds=0)


def test_refit_zone_tables_are_consistent():
    assert set(OUTPUT_ZONE) == set(PRICE_NODES)
    assert set(OUTPUT_ZONE.values()) <= set(REFIT_ZONES)
    assert REFIT_TECHS == ("onwind", "offwind", "solar")


# --- F25.3: SEM node weights from CP30's cable set ---------------------------------------------------

from pathlib import Path  # noqa: E402

from scripts.interconnectors.cp30_foreign_prices import NODE_WEIGHTS, sem_node_weights  # noqa: E402
from scripts.interconnectors.tyndp_prices import SEM_NODE_WEIGHTS_REGISTER  # noqa: E402

COMPOSITION = Path(__file__).resolve().parents[2] / "data" / "interconnectors" / "cp30_interconnector_composition.csv"


def test_sem_weights_from_cp30_cables_count_maresconnect_in_2030():
    assert sem_node_weights(COMPOSITION) == {"IE00": 1750.0, "UKNI": 500.0}
    assert sem_node_weights(COMPOSITION, year=2029) == {"IE00": 1000.0, "UKNI": 500.0}
    # Without the composition: today's register cables, unchanged.
    assert sem_node_weights() == {"IE00": 1009.0, "UKNI": 500.0} == SEM_NODE_WEIGHTS_REGISTER
    assert NODE_WEIGHTS == SEM_NODE_WEIGHTS_REGISTER


def test_rebase_sem_column_follows_the_cp30_cables(tmp_path):
    tyndp, frame = _tyndp_file(tmp_path)
    wind = _wind_file(tmp_path)
    old = build_cp30_basis_prices(tyndp, wind, tmp_path / "p0.csv", tmp_path / "a0.json")
    new = build_cp30_basis_prices(tyndp, wind, tmp_path / "p1.csv", tmp_path / "a1.json",
                                  composition_file=COMPOSITION)
    ratio = rebase_ratio(DEFAULT_TYNDP_NEW_CCGT_EUR, DEFAULT_FOREIGN_NEW_CCGT_GBP)
    sem = (1750.0 * frame["mc_IE00"] + 500.0 * frame["mc_UKNI"]) / 2250.0
    expected = np.concatenate([sem.to_numpy(), sem.to_numpy()[-168:-144]]).mean() * ratio
    assert new["Ireland"].mean() == pytest.approx(expected, rel=1e-9)
    # Only the SEM column moves.
    pd.testing.assert_frame_equal(old.drop(columns="Ireland"), new.drop(columns="Ireland"))
    audit = json.loads((tmp_path / "a1.json").read_text())
    assert audit["sem_weights_source"] == "cp30_composition"
    assert audit["zones"]["Ireland"]["IE00"] == pytest.approx(1750.0 / 2250.0)


def test_refit_sem_weights_do_not_reach_the_entsoe_2024_sem_column(tmp_path):
    zonal, cf, shape = _refit_inputs(tmp_path)
    kw = dict(ireland_shape_file=shape, foreign_new_ccgt_gbp=102.9, cv_folds=0)
    old = build_cp30_refit_prices(zonal, cf, tmp_path / "p0.csv", tmp_path / "a0.json", **kw)
    new = build_cp30_refit_prices(zonal, cf, tmp_path / "p1.csv", tmp_path / "a1.json",
                                  composition_file=COMPOSITION, **kw)
    pd.testing.assert_frame_equal(old, new)
    assert json.loads((tmp_path / "a1.json").read_text())["sem_node_weights_mw"] == {"IE00": 1750.0, "UKNI": 500.0}
