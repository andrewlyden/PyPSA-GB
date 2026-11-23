"""Extract comparable pre-uplift forward indices from pinned Annex 2 releases.

Rebuild with python -m scripts.consumer_bill.align_historical_procurement.
Cached published indices are evidence; absent daily quotes are never imputed.
"""
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.utils import get_column_letter

from scripts.consumer_bill.build_historical_evidence_panel import SPOT_FILES, validate_periods
from scripts.consumer_bill.build_domestic_demand_profile import peak_mask


# What each published index actually prices (review 28 Sep 2026). The 2020 caps used
# Ofgem's 6-2-12 semi-annual index: a six-month observation window, a two-month
# lag and season contracts for the 12 months from the cap window's start (Default
# Tariff Cap decision, Appendix 6, May 2018, para 2.6 and Table A6.1), so the index
# for January-March 2020 prices October 2019 to September 2020, not the quarter.
# The 2024 caps used 3-1.5-12: each delivery quarter's contract averaged over 12
# months of observations (Annex 2 v1.19-v1.22, sheet 2a(iii) Elec 3-1.5-12), so the
# delivery window is the quarter but the prices were struck up to a year before it.
PRICED_DELIVERY_WINDOWS = {
    "2020_Jan_Mar": ("2019-10-01", "2020-09-30", "6-2-12 semi-annual"),
    "2020_Apr_Sep": ("2020-04-01", "2021-03-31", "6-2-12 semi-annual"),
    "2020_Oct_Dec": ("2020-10-01", "2021-09-30", "6-2-12 semi-annual"),
    "2024_Q1": ("2024-01-01", "2024-03-31", "3-1.5-12"),
    "2024_Q2": ("2024-04-01", "2024-06-30", "3-1.5-12"),
    "2024_Q3": ("2024-07-01", "2024-09-30", "3-1.5-12"),
    "2024_Q4": ("2024-10-01", "2024-12-31", "3-1.5-12"),
}
ELEXON_CACHE = Path("data/market/elexon_calibration_cache.tar.gz")
#: Below this share of the window's half-hours the outturn is not reported.
MIN_OUTTURN_COVERAGE = 0.9
RETIRED_REASON = (
    "Retired 28 Sep 2026: the coefficient compares each index with the cap period's own delivery quarter, "
    "but the 2020 indices price 12-month delivery windows and the 2024 indices average 12 months of "
    "observations; both test years fell in falling markets, so the apparent held-out improvement is an "
    "artefact of the windows and the price trend. Against actual outturn on the windows the indices priced, "
    "the index-to-outturn gap changes sign with the price trend (index_minus_outturn), which a structural "
    "forward premium would not.")


def load_mid_prices(years, cache=ELEXON_CACHE) -> pd.Series:
    """Elexon MID half-hourly prices (GBP/MWh) for ``years`` from the pinned calibration cache."""
    frames = []
    with tarfile.open(cache) as archive:
        names = set(archive.getnames())
        for year in years:
            name = f"mid_prices_{year}.parquet"
            if name in names:
                frames.append(pd.read_parquet(io.BytesIO(archive.extractfile(name).read())))
    if not frames:
        return pd.Series(dtype=float)
    frame = pd.concat(frames)
    return pd.Series(frame["mid_price"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(frame["datetime"])).sort_index()


def outturn_blend(prices: pd.Series, start, end, base_share=0.7, peak_share=0.3) -> dict:
    """The 70/30 baseload/peak-block outturn over ``start``..``end`` inclusive, with its half-hour coverage."""
    start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    window = prices[(prices.index >= start) & (prices.index < end)]
    expected = (end - start) / pd.Timedelta(minutes=30)
    coverage = float(len(window) / expected) if expected else 0.0
    if coverage < MIN_OUTTURN_COVERAGE or window.empty:
        return {"coverage": coverage, "base": None, "peak": None, "blend": None}
    in_peak = peak_mask(window.index).to_numpy()
    base, peak = float(window.mean()), float(window[in_peak].mean())
    return {"coverage": coverage, "base": base, "peak": peak, "blend": base_share * base + peak_share * peak}


def extract_index(path: Path, column: str, modern: bool) -> dict:
    """Follow the published uplift and loss references; reconcile cached outputs."""
    sheet = "1b Direct Fuel Cost Component" if modern else "1a Direct Fuel Cost Component"
    uplift_row, loss_row = (26, 31) if modern else (24, 28)
    with_values = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_links=False)
    formulas = openpyxl.load_workbook(path, read_only=True, data_only=False, keep_links=False)
    try:
        values = {c.coordinate: c.value for row in with_values[sheet].iter_rows(max_row=91, max_col=40) for c in row if c.value is not None}
        formula = {c.coordinate: c.value for row in formulas[sheet].iter_rows(max_row=44, max_col=40) for c in row if c.value is not None}
        index = float(values[f"{column}16"])
        uplift = float(values[f"{column}{uplift_row}"])
        # Follow sheet references instead of guessing period-specific columns.
        uplift_formula = formula[f"{column}{uplift_row}"]
        match = re.search(r"'3a Allowances'!([A-Z]+)\$?(\d+):([A-Z]+)\$?(\d+)", uplift_formula)
        if match:
            col, start, last_col, end = match.groups()
            if col != last_col:
                raise ValueError("Unexpected allowance range")
            coords = [f"{col}{r}" for r in range(int(start), int(end)+1)]
        else:
            # Older releases use a fixed six-row allowance range.
            match = re.search(r"'3a Allowances'!\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)", uplift_formula)
            if not match:
                raise ValueError(f"Unsupported allowance formula: {uplift_formula}")
            col, start, last_col, end = match.groups()
            coords = [f"{col}{r}" for r in range(int(start), int(end)+1)]
        allowance_values = {c.coordinate: c.value for row in with_values['3a Allowances'].iter_rows(max_row=40,max_col=60) for c in row if c.value is not None}
        rate = sum(float(allowance_values[c]) for c in coords)
        loss_values = {c.coordinate: c.value for row in with_values['3c Electricity losses'].iter_rows(max_row=40,max_col=60) for c in row if c.value is not None}
        loss_refs = []
        for row in range(loss_row, loss_row+14):
            match = re.search(r"'3c Electricity losses'!\$?([A-Z]+)\$?(\d+)", formula[f"{column}{row}"])
            if not match:
                raise ValueError("Unsupported loss reference")
            loss_refs.append(''.join(match.groups()))
        loss = float(np.mean([float(loss_values[c]) for c in loss_refs]))
        cached = float(np.mean([float(values[f"{column}{r}"]) for r in range(loss_row,loss_row+14)])) * 3.1
        reconstructed = index * (1+rate) * loss * 3.1
        if not np.isclose(index*(1+rate), uplift, atol=1e-8, rtol=0) or not np.isclose(cached,reconstructed,atol=1e-6,rtol=0):
            raise ValueError("Annex index/uplift/loss reconstruction mismatch")
        return dict(forward_index_gbp_per_mwh=index, allowance_rate=rate, loss_multiplier=loss,
                    reconstructed_energy_gbp=reconstructed, cached_energy_gbp=cached,
                    index_cell=f"{sheet}!{column}16", allowance_cells="3a Allowances!"+';'.join(coords),
                    loss_cells="3c Electricity losses!"+';'.join(loss_refs), source_file=str(path),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        with_values.close(); formulas.close()


def fit_transfer(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit a dimensionless 2020 coefficient and hold every 2024 target out."""
    result = frame.copy()
    for year, selected in result.groupby('year'):
        validate_periods(selected, int(year))
    if set(result.year) != {2020,2024} or not np.isfinite(result[['spot_gbp_per_mwh','forward_index_gbp_per_mwh']]).all().all():
        raise ValueError("Complete finite 2020/2024 inputs required")
    train = result.loc[result.year.eq(2020)]
    w = train.delivery_days
    x,y = train.spot_gbp_per_mwh,train.forward_index_gbp_per_mwh
    if (w*x*x).sum() <= 0:
        raise ValueError("No training signal")
    coefficient=float((w*x*y).sum()/(w*x*x).sum())
    result['split']=np.where(result.year.eq(2020),'train','test')
    result['predicted_index_gbp_per_mwh']=result.spot_gbp_per_mwh*coefficient
    result['raw_error_gbp_per_mwh']=result.spot_gbp_per_mwh-result.forward_index_gbp_per_mwh
    result['fitted_error_gbp_per_mwh']=result.predicted_index_gbp_per_mwh-result.forward_index_gbp_per_mwh
    test=result.loc[result.year.eq(2024)];w=test.delivery_days/test.delivery_days.sum()
    return result,dict(training_year=2020,test_year=2024,coefficient=coefficient,
                       raw_test_mae_gbp_per_mwh=float((test.raw_error_gbp_per_mwh.abs()*w).sum()),
                       fitted_test_mae_gbp_per_mwh=float((test.fitted_error_gbp_per_mwh.abs()*w).sum()),
                       aligned_boundary='Published forward energy index before allowances, losses, CfD, backwardation and Capacity Market',
                       forecast_validated=False,interpretation='Chronological delivery-price transfer diagnostic; not an observation-date forward-price forecast. Procurement horizons differ between regimes.',
                       coefficient_status='retired',coefficient_retired_reason=RETIRED_REASON)


def add_outturn_on_priced_windows(result: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """The index against actual 70/30 outturn over the window it priced (and over its cap period)."""
    out = result.copy()
    rows = []
    for r in out.itertuples():
        start, end, regime = PRICED_DELIVERY_WINDOWS[r.cap_period]
        priced = outturn_blend(prices, start, end)
        period = outturn_blend(prices, r.delivery_start, r.delivery_end)
        rows.append({"index_regime": regime, "priced_delivery_start": start, "priced_delivery_end": end,
                     "outturn_priced_window_coverage": priced["coverage"],
                     "outturn_priced_window_blend_gbp_per_mwh": priced["blend"],
                     "index_minus_outturn_priced_window_gbp_per_mwh": (
                         r.forward_index_gbp_per_mwh - priced["blend"] if priced["blend"] is not None else None),
                     "outturn_cap_period_blend_gbp_per_mwh": period["blend"],
                     "index_minus_outturn_cap_period_gbp_per_mwh": (
                         r.forward_index_gbp_per_mwh - period["blend"] if period["blend"] is not None else None),
                     "model_minus_outturn_cap_period_gbp_per_mwh": (
                         r.spot_gbp_per_mwh - period["blend"] if period["blend"] is not None else None)})
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def main() -> None:
    output=Path('resources/analysis/consumer_bill_evidence');output.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv('data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv')
    modern=pd.read_csv('data/market/ofgem_price_cap/2024_annual/hedged_wholesale_inputs.csv')
    downloads=json.loads(Path('data/market/consumer_bill_sources/procurement_alignment.json').read_text())
    sources={r['cap_period']:Path(r['path']) for r in downloads['sources']}
    for row in downloads['sources']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Downloaded source checksum changed')
    sources['2024_Q2']=Path('data/market/consumer_bill_sources/raw/2024/2024_Q2_annex_2_wholesale.xlsx')
    old_names=['2019_10_to_2020_03_annex_2_wholesale_v1.4.xlsx','2020_04_to_2020_09_annex_2_wholesale_v1.5.xlsx','2020_10_to_2020_03_annex_2_wholesale_v1.6.xlsx']
    rows=[]
    for year,inputs in [(2020,old),(2024,modern)]:
        spot=pd.read_csv(Path('resources/analysis')/SPOT_FILES[year]).set_index('delivery_start')
        for i,row in enumerate(inputs.to_dict('records')):
            path=Path('data/market/ofgem_price_cap/2020_source')/old_names[i] if year==2020 else sources[row['cap_period']]
            extracted=extract_index(path,get_column_letter(19+i if year==2020 else 31+i),year==2024)
            expected=row['forward_index_gbp_per_mwh']*3.1 if year==2020 else row['direct_fuel_energy_cost_gbp_per_3100kwh']
            # For 2020 the compatibility table contains a pre-uplift index.
            if year==2020:
                expected_index=row['forward_index_gbp_per_mwh']
                if not np.isclose(extracted['forward_index_gbp_per_mwh'],expected_index,atol=1e-8):raise ValueError('2020 index mismatch')
                parent=pd.read_csv('data/market/ofgem_price_cap/2020_historical_annual/wholesale_components.csv')
                expected=float(parent.loc[parent.delivery_start.eq(row['delivery_start']),'annual_gbp_per_3100kwh_ex_vat'].iloc[0])
            extracted['cap_energy_residual_gbp']=extracted['reconstructed_energy_gbp']-expected
            if abs(extracted['cap_energy_residual_gbp'])>1e-5:raise ValueError('Cap energy reconciliation failed')
            obs_start,obs_end=row.get('observation_start'),row.get('observation_end')
            date_source='existing 2020 hedge input table; six-month observation period'
            if year==2024:
                w=openpyxl.load_workbook(path,read_only=True,data_only=True,keep_links=False)
                try:
                    matches=[]
                    for number,values in enumerate(w['3b Demand'].iter_rows(min_row=40,max_row=160,max_col=10,values_only=True),40):
                        dated=[(j,value) for j,value in enumerate(values) if hasattr(value,'year')]
                        if str(values[1]).startswith('Q') and len(dated)==3 and pd.Timestamp(dated[0][1])==pd.Timestamp(row['delivery_start']):
                            matches.append((number,dated))
                    if len(matches)!=1:raise ValueError('Observation calendar mismatch')
                    number,dates=matches[0]
                    obs_start=pd.Timestamp(dates[1][1]).date().isoformat();obs_end=pd.Timestamp(dates[2][1]).date().isoformat()
                    date_source=f'3b Demand!{get_column_letter(dates[1][0]+1)}{number};{get_column_letter(dates[2][0]+1)}{number}'
                finally:w.close()
            if pd.Timestamp(obs_end)>=pd.Timestamp(row['delivery_start']):raise ValueError('Observation after delivery start')
            rows.append(dict(**extracted,year=year,cap_period=row['cap_period'],delivery_start=row['delivery_start'],delivery_end=row['delivery_end'],
                             delivery_days=(pd.Timestamp(row['delivery_end'])-pd.Timestamp(row['delivery_start'])).days+1,
                             observation_start=obs_start,observation_end=obs_end,observation_source=date_source,
                             spot_gbp_per_mwh=float(spot.loc[row['delivery_start'],'pypsa_spot_price_gbp_per_mwh'])))
            print('Reconciled',row['cap_period'],flush=True)
    result,summary=fit_transfer(pd.DataFrame(rows))
    mid = load_mid_prices(range(2019, 2025))
    result = add_outturn_on_priced_windows(result, mid)
    summary['outturn_source']=(f'Elexon MID half-hourly prices, {ELEXON_CACHE.as_posix()} (70/30 baseload/peak block; '
                               f'coverage below {MIN_OUTTURN_COVERAGE:.0%} not reported)')
    for key in ('index_minus_outturn_priced_window_gbp_per_mwh', 'index_minus_outturn_cap_period_gbp_per_mwh',
                'model_minus_outturn_cap_period_gbp_per_mwh'):
        summary[key]={r['cap_period']: r[key] for r in result.to_dict('records')}
    result.to_csv(output/'aligned_procurement_periods.csv',index=False)
    summary['daily_quote_status']='2024 Q2 published seasonal and quarterly price input D:E cells contain no numeric quotes; published cached index can be used but cannot independently reconstruct daily forward prices.'
    summary['observation_dates_recovered']=True
    summary['max_cap_reconciliation_residual_gbp']=float(result.cap_energy_residual_gbp.abs().max())
    summary['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path('data/market/consumer_bill_sources/procurement_alignment.json'),
        Path('data/market/ofgem_price_cap/2020_annual/hedged_wholesale_inputs.csv'),
        Path('data/market/ofgem_price_cap/2024_annual/hedged_wholesale_inputs.csv'),
        *[Path('resources/analysis')/name for name in SPOT_FILES.values()]]}
    (output/'aligned_procurement_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report = '# Aligned historical procurement validation\n\n'
    report += 'The forward-energy index is now separated from shaping, risk, imbalance, transaction costs, regional losses, CfD, backwardation and Capacity Market charges. All seven periods reconcile through the published allowance and loss formulas to their cap energy amounts.\n\n'
    report += f"Maximum reconciliation residual: GBP{summary['max_cap_reconciliation_residual_gbp']:.8f} per 3,100 kWh household, excluding VAT.\n\n"
    report += 'The transfer diagnostic fits a dimensionless coefficient on all 2020 periods and holds all 2024 quarters out. Each year retains its nominal GBP basis; no cross-year cash intercept is estimated.\n\n'
    report += f"Fitted coefficient: {summary['coefficient']:.6f}. Held-out 2024 MAE: raw spot GBP{summary['raw_test_mae_gbp_per_mwh']:.2f}/MWh; fitted GBP{summary['fitted_test_mae_gbp_per_mwh']:.2f}/MWh.\n\n"
    report += f"**The coefficient is retired.** {RETIRED_REASON}\n\n"
    report += _outturn_table(result)
    report += '| Period | Observation start | Observation end | Published index GBP/MWh | Raw error GBP/MWh | Fitted error GBP/MWh |\n| --- | --- | --- | --- | --- | --- |\n'
    for r in result.itertuples():
        report += f'| {r.cap_period} | {r.observation_start} | {r.observation_end} | {r.forward_index_gbp_per_mwh:.2f} | {r.raw_error_gbp_per_mwh:.2f} | {r.fitted_error_gbp_per_mwh:.2f} |\n'
    report += '\n2020 rows are training results, not validation. 2024 rows are held-out delivery-price diagnostics. The procurement regimes have different observation/delivery horizons, so a common accounting boundary does not imply an unchanged economic relationship.\n\n'
    report += 'Observation dates for 2024 are extracted from each release\'s 3b Demand calendar. The 2020 dates retain the existing six-month observation input table. The calendar is not evidence of available daily prices: the inspected Q2 2024 seasonal and quarterly D:E price columns contain no numeric quotes, while the workbook publishes cached indices.\n\n'
    report += '**Forecast validation remains false.** An independent observation-date reconstruction requires the dated seasonal/quarterly base and peak quotes feeding the published index and the applicable product/demand weights. A prospective PyPSA bridge additionally requires scenario information available at the observation date. Delivery spot prices are not those inputs. No future bill assumption was changed.\n\n'
    report += 'Artifacts: [aligned periods and cell references](aligned_procurement_periods.csv), [summary and fingerprints](aligned_procurement_summary.json). Sources are pinned in `data/market/consumer_bill_sources/procurement_alignment.json`; existing releases retain their existing source locations.\n'
    (output/'aligned_procurement_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2))


def _outturn_table(result: pd.DataFrame) -> str:
    def fmt(value):
        return "" if value is None or not np.isfinite(float(value)) else f"{float(value):.2f}"
    text = ("Index against actual outturn (Elexon MID, 70/30 baseload/peak block), GBP/MWh:\n\n"
            "| Period | Index regime | Priced delivery window | Index | Index minus outturn, priced window | "
            "Index minus outturn, cap period | Model minus outturn, cap period |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n")
    for r in result.to_dict("records"):
        text += (f"| {r['cap_period']} | {r['index_regime']} | {r['priced_delivery_start']} to "
                 f"{r['priced_delivery_end']} (coverage {r['outturn_priced_window_coverage']:.0%}) | "
                 f"{r['forward_index_gbp_per_mwh']:.2f} | {fmt(r['index_minus_outturn_priced_window_gbp_per_mwh'])} | "
                 f"{fmt(r['index_minus_outturn_cap_period_gbp_per_mwh'])} | "
                 f"{fmt(r['model_minus_outturn_cap_period_gbp_per_mwh'])} |\n")
    return text + "\n"


if __name__=='__main__':main()
