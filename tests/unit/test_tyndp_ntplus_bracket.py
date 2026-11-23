"""A modelled year NT+ 2026 does not publish blends its neighbours' shapes (user decision, 11 Nov 2026)."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.interconnectors import tyndp_prices as tp


def test_the_bracket_of_a_year_between_published_years():
    assert tp.ntplus_bracket(2045) == (2040, 2050, 0.5)
    assert tp.ntplus_bracket(2042) == (2040, 2050, pytest.approx(0.2))
    assert tp.ntplus_bracket(2040) is None and tp.ntplus_bracket(2050) is None
    with pytest.raises(ValueError, match="outside the declared"):
        tp.ntplus_bracket(2060)


def test_a_bracketed_year_reads_both_neighbours_files():
    files = tp.tyndp_input_files("NTplus2026", 2045, "d")
    assert set(files) == {"ntplus_prices_2040", "ntplus_wind_2040", "ntplus_prices_2050", "ntplus_wind_2050"}
    assert files["ntplus_prices_2040"] == "d/tyndp2026_NTplus2040_mc.csv"


def _shape(path, level):
    idx = pd.date_range("2013-01-01", periods=48, freq="h", name="datetime")
    pd.DataFrame({"France": np.full(48, level), "Norway": np.linspace(0, level, 48)}, index=idx).to_csv(path)
    path.with_name(path.stem + "_audit.json").write_text(json.dumps({"shape_source": f"level {level}", "mean_gbp": {}}))


def test_the_blend_is_hourly_and_distance_weighted(tmp_path):
    lo, hi = tmp_path / "lo.csv", tmp_path / "hi.csv"
    _shape(lo, 80.0)
    _shape(hi, 40.0)
    out = tmp_path / "y2045.csv"
    shaped = tp.blend_ntplus_shapes(lo, hi, tmp_path / "lo_audit.json", tmp_path / "hi_audit.json", out,
                                    tmp_path / "y2045_audit.json", 2045, "NTplus2026")
    assert shaped["France"].eq(60.0).all()
    assert shaped["Norway"].iloc[-1] == pytest.approx(60.0)
    audit = json.loads((tmp_path / "y2045_audit.json").read_text())
    assert audit["bracket"] == {"lower": 2040, "upper": 2050, "upper_share": 0.5}
    assert pd.read_csv(out, index_col=0).shape == (48, 2)


def test_the_blend_refuses_shapes_on_different_calendars(tmp_path):
    lo, hi = tmp_path / "lo.csv", tmp_path / "hi.csv"
    _shape(lo, 80.0)
    _shape(hi, 40.0)
    frame = pd.read_csv(hi, index_col=0)
    frame.iloc[:24].to_csv(hi)
    with pytest.raises(ValueError, match="same calendar"):
        tp.blend_ntplus_shapes(lo, hi, tmp_path / "lo_audit.json", tmp_path / "hi_audit.json", tmp_path / "o.csv",
                               tmp_path / "o_audit.json", 2045, "NTplus2026")
