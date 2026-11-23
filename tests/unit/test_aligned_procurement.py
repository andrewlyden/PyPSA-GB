"""Chronological fitting must not use held-out prices or targets."""
import numpy as np
import pandas as pd
import pytest

from scripts.consumer_bill.align_historical_procurement import fit_transfer


def inputs():
    rows=[]
    for year in [2020,2024]:
        for q in range(1,5):
            start=pd.Timestamp(year,3*q-2,1)
            end=start+pd.offsets.QuarterEnd()
            rows.append(dict(year=year,cap_period=f'{year}_Q{q}',delivery_start=start,delivery_end=end,
                             delivery_days=(end-start).days+1,spot_gbp_per_mwh=20*q,
                             forward_index_gbp_per_mwh=30*q))
    return pd.DataFrame(rows)


def test_future_targets_do_not_affect_training():
    frame=inputs()
    _,first=fit_transfer(frame)
    frame.loc[frame.year.eq(2024),'forward_index_gbp_per_mwh']*=10
    _,second=fit_transfer(frame)
    assert first['coefficient']==pytest.approx(1.5)
    assert first['coefficient']==second['coefficient']
    assert second['fitted_test_mae_gbp_per_mwh']>first['fitted_test_mae_gbp_per_mwh']
    assert not second['forecast_validated']


def test_missing_test_period_rejected():
    with pytest.raises(ValueError,match='Incomplete calendar year'):
        fit_transfer(inputs().iloc[:-1])


def test_outturn_is_the_70_30_block_blend_and_refuses_thin_coverage():
    from scripts.consumer_bill.align_historical_procurement import outturn_blend
    from scripts.consumer_bill.build_domestic_demand_profile import peak_mask

    index = pd.date_range("2024-01-01", "2024-03-31 23:30", freq="30min")
    peak = peak_mask(index).to_numpy()
    prices = pd.Series(np.where(peak, 100.0, 50.0), index=index)
    out = outturn_blend(prices, "2024-01-01", "2024-03-31")
    assert out["coverage"] == pytest.approx(1.0)
    assert out["peak"] == pytest.approx(100.0)
    assert out["blend"] == pytest.approx(0.7 * prices.mean() + 0.3 * 100.0)
    thin = outturn_blend(prices.iloc[::2], "2024-01-01", "2024-03-31")
    assert thin["blend"] is None and thin["coverage"] == pytest.approx(0.5)


def test_each_index_is_compared_on_the_window_it_priced_and_the_fit_is_retired():
    from scripts.consumer_bill.align_historical_procurement import PRICED_DELIVERY_WINDOWS

    # 6-2-12 semi-annual: 12 months from the cap window's start; 3-1.5-12: the quarter.
    assert PRICED_DELIVERY_WINDOWS["2020_Apr_Sep"][:2] == ("2020-04-01", "2021-03-31")
    assert PRICED_DELIVERY_WINDOWS["2024_Q3"][:2] == ("2024-07-01", "2024-09-30")
    _, summary = fit_transfer(inputs())
    assert summary["coefficient_status"] == "retired" and "falling markets" in summary["coefficient_retired_reason"]
