"""Quick analysis of wholesale market merit order with correction factors."""
import pandas as pd
import pypsa
import numpy as np
import sys

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else 'Historical_2024_lowwind'

# 1. Load marginal cost breakdown
mc = pd.read_csv(f'resources/generators/{SCENARIO}_marginal_costs_breakdown.csv')
print('=' * 80)
print(f'MARGINAL COST BREAKDOWN — {SCENARIO}')
print('=' * 80)

# Merit order by carrier
active = mc[mc['protected'] == False]
merit = active.groupby('carrier').agg(
    count=('generator', 'count'),
    capacity_MW=('p_nom_MW', 'sum'),
    avg_MC=('marginal_cost_total', 'mean'),
    min_MC=('marginal_cost_total', 'min'),
    max_MC=('marginal_cost_total', 'max')
).sort_values('avg_MC')

print('\nMERIT ORDER (by average marginal cost):')
print(f'{"Carrier":>22s} {"Units":>6s} {"Cap(MW)":>10s} {"Avg MC":>10s} {"Min MC":>10s} {"Max MC":>10s}')
print('-' * 70)
for carrier, row in merit.iterrows():
    print(f'{carrier:>22s} {row["count"]:>6.0f} {row["capacity_MW"]:>10.1f} '
          f'{row["avg_MC"]:>10.2f} {row["min_MC"]:>10.2f} {row["max_MC"]:>10.2f}')

total_cap = mc['p_nom_MW'].sum()
print(f'\nTotal generation capacity: {total_cap:,.1f} MW')

# 2. Load wholesale dispatch results
print('\n' + '=' * 80)
print('WHOLESALE MARKET DISPATCH (copperplate)')
print('=' * 80)

n = pypsa.Network(f'resources/market/{SCENARIO}_wholesale.nc')

# Generator dispatch by carrier
gen_dispatch = n.generators_t.p  # MW per timestep
carrier_map = n.generators.carrier

dispatch_by_carrier = gen_dispatch.T.groupby(carrier_map).sum().T  # timestep x carrier
total_gen_MWh = dispatch_by_carrier.sum()  # sum across all timesteps (MWh since hourly)
capacity_by_carrier = n.generators.groupby('carrier')['p_nom'].sum()

print('\nDISPATCH SUMMARY (24 hours):')
print(f'{"Carrier":>22s} {"Cap(MW)":>10s} {"Gen(MWh)":>12s} {"CF(%)":>8s} {"Avg MC":>10s}')
print('-' * 66)
for carrier in total_gen_MWh.sort_values(ascending=False).index:
    mwh = total_gen_MWh[carrier]
    if mwh < 0.01:
        continue
    cap = capacity_by_carrier.get(carrier, 0)
    cf = (mwh / (cap * 24) * 100) if cap > 0 else 0
    avg_mc = n.generators.loc[n.generators.carrier == carrier, 'marginal_cost'].mean()
    print(f'{carrier:>22s} {cap:>10.1f} {mwh:>12.1f} {cf:>8.1f} {avg_mc:>10.2f}')

total_demand = n.loads_t.p_set.sum().sum()
total_generation = gen_dispatch.sum().sum()
print(f'\nTotal demand: {total_demand:,.1f} MWh')
print(f'Total generation: {total_generation:,.1f} MWh')

# Storage dispatch
if len(n.storage_units) > 0:
    su_dispatch = n.storage_units_t.p  # positive = discharging
    su_by_carrier = su_dispatch.T.groupby(n.storage_units.carrier).sum().T
    print('\nSTORAGE DISPATCH:')
    for carrier in su_by_carrier.columns:
        discharge = su_by_carrier[carrier][su_by_carrier[carrier] > 0].sum()
        charge = su_by_carrier[carrier][su_by_carrier[carrier] < 0].sum()
        print(f'  {carrier}: discharge={discharge:,.1f} MWh, charge={charge:,.1f} MWh')

# 3. Wholesale price
print('\n' + '=' * 80)
print('WHOLESALE PRICE')
print('=' * 80)

prices = pd.read_csv(f'resources/market/{SCENARIO}_wholesale_price.csv',
                      index_col=0, parse_dates=True)
if isinstance(prices, pd.DataFrame) and len(prices.columns) > 0:
    # May have multiple bus columns or a single price column
    if prices.shape[1] == 1:
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices.mean(axis=1)  # average across buses
    
    print(f'Mean wholesale price:   {price_series.mean():>8.2f} GBP/MWh')
    print(f'Min wholesale price:    {price_series.min():>8.2f} GBP/MWh')
    print(f'Max wholesale price:    {price_series.max():>8.2f} GBP/MWh')
    print(f'Price std dev:          {price_series.std():>8.2f} GBP/MWh')
    
    print('\nHourly price statistics (first 48h shown if >48 snapshots):')
    shown = price_series.head(48)
    for ts, price in shown.items():
        print(f'  {ts}: {price:>8.2f} GBP/MWh')
    if len(price_series) > 48:
        print(f'  ... ({len(price_series) - 48} more hours)')

# 4. Curtailment analysis
print('\n' + '=' * 80)
print('CURTAILMENT ANALYSIS')
print('=' * 80)

renewable_carriers = ['wind_onshore', 'wind_offshore', 'solar_pv']
for carrier in renewable_carriers:
    gens = n.generators[n.generators.carrier == carrier]
    if len(gens) == 0:
        continue
    p_max = (n.generators_t.p_max_pu[gens.index] * gens.p_nom).sum(axis=1)
    p_actual = n.generators_t.p[gens.index].sum(axis=1)
    curtailed = (p_max - p_actual).clip(lower=0)
    total_available = p_max.sum()
    total_curtailed = curtailed.sum()
    pct = (total_curtailed / total_available * 100) if total_available > 0 else 0
    print(f'{carrier}: available={total_available:,.1f} MWh, curtailed={total_curtailed:,.1f} MWh ({pct:.1f}%)')

# 5. Hourly dispatch stack (key carriers)
print('\n' + '=' * 80)
print('HOURLY DISPATCH STACK (MW by carrier, first 48h)')
print('=' * 80)

key_carriers = ['wind_offshore', 'wind_onshore', 'solar_pv', 'nuclear', 'CCGT',
                'OCGT', 'biomass', 'Coal', 'Oil', 'large_hydro']
gen_dispatch = n.generators_t.p
carrier_map = n.generators.carrier
stack = gen_dispatch.T.groupby(carrier_map).sum().T

# Also get storage and links dispatch
su_total = n.storage_units_t.p.sum(axis=1) if len(n.storage_units) > 0 else pd.Series(0, index=n.snapshots)
link_flow = n.links_t.p0.sum(axis=1) if len(n.links) > 0 else pd.Series(0, index=n.snapshots)
demand = n.loads_t.p_set.sum(axis=1)

avail_cols = [c for c in key_carriers if c in stack.columns and stack[c].sum() > 0]
header = f'{"Hour":>16s} {"Demand":>8s}'
for c in avail_cols:
    header += f' {c:>12s}'
header += f' {"Storage":>10s} {"Links":>10s} {"Price":>8s}'
print(header)
print('-' * len(header))

if isinstance(prices, pd.DataFrame) and len(prices.columns) > 0:
    if prices.shape[1] == 1:
        price_s = prices.iloc[:, 0]
    else:
        price_s = prices.mean(axis=1)
else:
    price_s = pd.Series(0, index=n.snapshots)

for i, ts in enumerate(n.snapshots[:48]):
    row = f'{str(ts)[:16]:>16s} {demand.loc[ts]:>8.0f}'
    for c in avail_cols:
        row += f' {stack.loc[ts, c]:>12.0f}'
    row += f' {su_total.loc[ts]:>10.0f}'
    row += f' {link_flow.loc[ts]:>10.0f}'
    p = price_s.loc[ts] if ts in price_s.index else 0
    row += f' {p:>8.1f}'
    print(row)
if len(n.snapshots) > 48:
    print(f'  ... ({len(n.snapshots) - 48} more hours)')
