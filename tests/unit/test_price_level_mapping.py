"""The model's 2030 prices mapped onto CP30's own duration curve (CP.32; the legacy price path).

Every hour keeps its rank; the mapped year averages the curve exactly; tied prices map
to one value; the asset settlement at that level re-prices only the Stage 1 positions
and the CfD top-ups.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.consumer_bill import price_level_mapping as plm
from scripts.public_ownership.price_level_settlement import MODEL_PRICE_COLUMNS, carrier_capture, combine
from config.config_loader import load_defaults

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def curve():
    return plm.load_duration_curve(ROOT / plm.DEFAULT_SETTINGS["source_file"], plm.DEFAULT_SETTINGS["pathway"])


def test_the_cp32_extract_is_the_further_flex_curve_at_69_9(curve):
    """CP.32 Further Flex and Renewables averages GBP 69.9/MWh with 31% of hours at or below GBP 1."""
    assert plm.curve_mean(curve) == pytest.approx(69.89, abs=0.01)
    assert curve.index[0] == 0.0 and curve.index[-1] == 1.0 and len(curve) == 101
    assert (curve.le(1.0).mean()) == pytest.approx(0.307, abs=0.001)
    frame = pd.read_csv(ROOT / plm.DEFAULT_SETTINGS["source_file"])
    assert plm.curve_mean(plm.load_duration_curve(ROOT / plm.DEFAULT_SETTINGS["source_file"], "actual_2023")) \
        == pytest.approx(108.47, abs=0.01)
    assert set(plm.PATHWAYS) <= set(frame.columns)


def test_a_rising_curve_or_a_short_one_is_refused(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"share_of_hours": [0.0, 0.5, 1.0], "new_dispatch": [10.0, 20.0, 5.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="not a duration curve"):
        plm.load_duration_curve(bad, "new_dispatch")
    short = tmp_path / "short.csv"
    pd.DataFrame({"share_of_hours": [0.0, 0.5], "new_dispatch": [10.0, 5.0]}).to_csv(short, index=False)
    with pytest.raises(ValueError, match="from 0 to 1"):
        plm.load_duration_curve(short, "new_dispatch")


def test_mapping_keeps_rank_hits_the_curve_mean_and_ties_map_together(curve):
    rng = np.random.default_rng(3)
    index = pd.date_range("2030-01-01", periods=8760, freq="h")
    prices = pd.Series(np.where(rng.random(8760) < 0.35, 0.0, rng.gamma(4.0, 15.0, 8760)), index=index)
    mapped = plm.quantile_map(prices, curve)
    assert plm.weighted_mean(mapped, pd.Series(1.0, index=index)) == pytest.approx(plm.curve_mean(curve), rel=1e-12)
    order = prices.sort_values(kind="mergesort").index
    assert mapped.loc[order].diff().dropna().ge(-1e-9).all()  # a higher model price never maps lower
    assert mapped[prices.eq(0.0)].nunique() == 1  # one tie, one value
    # The zero-price tie (35% of hours) covers the curve's bottom 35%: its mean is that slice's mean.
    share = float(prices.eq(0.0).mean())
    lower = plm._cumulative_integral(curve, np.array([1.0 - share, 1.0]))
    assert float(mapped[prices.eq(0.0)].iloc[0]) == pytest.approx((lower[1] - lower[0]) / share)


def test_hour_weights_are_shares_of_the_year(curve):
    prices = pd.Series([100.0, 50.0, 10.0])
    even = plm.quantile_map(prices, curve)
    weighted = plm.quantile_map(prices, curve, hours=pd.Series([2.0, 1.0, 1.0]))
    assert plm.weighted_mean(weighted, pd.Series([2.0, 1.0, 1.0])) == pytest.approx(plm.curve_mean(curve))
    assert weighted.iloc[0] < even.iloc[0]  # the top hour now spans half the year
    with pytest.raises(ValueError, match="weights"):
        plm.quantile_map(prices, curve, hours=pd.Series([1.0, 0.0, 1.0]))


def test_settings_match_the_config_block_and_refuse_unknown_keys():
    block = load_defaults()["consumer_bill"]["price_level_mapping"]
    assert plm.settings_from_config(block) == plm.DEFAULT_SETTINGS  # code defaults equal config
    with pytest.raises(ValueError, match="unknown keys"):
        plm.settings_from_config({"level": 70})
    with pytest.raises(ValueError, match="pathway"):
        plm.settings_from_config({"pathway": "steady_progression"})


def test_mapped_price_files_keep_their_shape(tmp_path, curve):
    index = pd.date_range("2030-01-01", periods=48, freq="h")
    for month, part in (("a", index[:24]), ("b", index[24:])):
        pd.DataFrame({"wholesale_price": np.linspace(0, 90, 24), "price_spread": 1.0, "hydrogen_price": 2.0},
                     index=pd.Index(part, name="snapshot")).to_csv(tmp_path / f"{month}_wholesale_price.csv")
    hourly = pd.concat([
        pd.DataFrame({"price_gbp_per_mwh": np.linspace(0, 90, 24), "hours": 1.0, "demand_mw": 1.0,
                      "price_file": str(tmp_path / f"{m}_wholesale_price.csv")}, index=part)
        for m, part in (("a", index[:24]), ("b", index[24:]))])
    mapped = plm.map_market_year(hourly, curve)
    paths = plm.write_mapped_price_files(hourly, mapped, tmp_path / "out")
    back = pd.concat(pd.read_csv(p, index_col=0, parse_dates=True) for p in paths)
    assert np.allclose(back["wholesale_price"].to_numpy(), mapped.to_numpy())
    assert np.allclose(back["model_wholesale_price"].to_numpy(), hourly["price_gbp_per_mwh"].to_numpy())
    assert back["hydrogen_price"].eq(2.0).all()  # held as solved
    summary = plm.mapping_summary(hourly, mapped, curve, plm.DEFAULT_SETTINGS)
    assert summary["mapped"]["time_weighted_gbp_per_mwh"] == pytest.approx(plm.curve_mean(curve))


def _settlement(wholesale, cfd, *, energy=10.0):
    return pd.DataFrame({
        "component_type": ["Generator", "Generator"], "asset_id": ["a", "b"], "carrier": ["wind_offshore", "CCGT"],
        "wholesale_mwh": [energy, 5.0], "wholesale_revenue_gbp": wholesale, "balancing_settlement_gbp": [-3.0, 2.0],
        "cfd_support_gbp": cfd, "roc_support_gbp": [0.0, 0.0], "fuel_carbon_gbp": [0.0, 100.0],
        "market_revenue_gbp": [w + b for w, b in zip(wholesale, [-3.0, 2.0])],
        "support_revenue_gbp": [c + 1.0 for c in cfd],
    })


def test_combine_reprices_positions_and_top_ups_and_holds_the_rest():
    central = _settlement([400.0, 500.0], [600.0, 0.0])
    mapped = _settlement([700.0, 800.0], [300.0, 0.0])
    mapped["balancing_settlement_gbp"] = 99.0  # never taken from the re-settlement
    out = combine(central, mapped).set_index("asset_id")
    assert out.loc["a", "wholesale_revenue_gbp"] == 700.0 and out.loc["a", MODEL_PRICE_COLUMNS["wholesale_revenue_gbp"]] == 400.0
    assert out.loc["a", "market_revenue_gbp"] == 697.0  # mapped wholesale + central balancing
    assert out.loc["a", "cfd_support_gbp"] == 300.0
    assert out.loc["a", "support_revenue_gbp"] == pytest.approx(301.0)  # central support less the top-up's fall
    assert out.loc["b", "fuel_carbon_gbp"] == 100.0
    capture = carrier_capture(out.reset_index())
    assert capture.loc["wind_offshore", "capture_mapped_gbp_per_mwh"] == pytest.approx(70.0)
    with pytest.raises(ValueError, match="Stage 1 energy"):
        combine(central, _settlement([700.0, 800.0], [300.0, 0.0], energy=11.0))
