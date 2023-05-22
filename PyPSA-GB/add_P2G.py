import pandas as pd
import numpy as np
import os
import shutil
import re
import distance_calculator as dc

def gsp_to_bus(row, df_gsp_data):
    output = {}
    gsp = row['GSP']
    gsp_ = re.sub('\(.*?\)','', gsp)
    if gsp in df_gsp_data.index:
        bus = df_gsp_data.loc[gsp].Bus
        output['bus'] = bus
    elif gsp_ == 'Direct' or gsp_ == 'Not Connected':
        output['bus'] = gsp
    elif gsp_[:-1] in df_gsp_data.index:
        bus = df_gsp_data.loc[gsp_[:-1]].Bus
        output['bus'] = bus
    else:
        try:
            output['bus'] = df_gsp_data[df_gsp_data.index.str.contains(gsp)].Bus
        except:
            print(gsp)
            output['bus'] = np.nan
    if type(output['bus']) == pd.Series:
        output['bus'] = output['bus'].tolist()[0]
    return pd.Series(output)

def add_P2G(year, scenario = None, path = 'LOPF_data', replace = False):

    if path[-1] == '/':
        path = path[:-1]
    if replace:
        save_path = path + '/'
    else:
        save_path = path + '_P2G/'
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        shutil.copytree(path, save_path)
    path = path + '/'

    if scenario =='Leading The Way':
        scenario ='Leading the Way'

    buses_scotland = ['Beauly', 'Peterhead', 'Errochty', 'Denny/Bonnybridge', 'Neilston', 'Strathaven', 'Torness', 'Eccles']

    buses_rgb = ['Harker', 'Stella West', 'Penwortham', 'Deeside', 'Daines', 'Th. Marsh/Stocksbridge', 
                 'Thornton/Drax/Eggborough', 'Keadby', 'Ratcliffe', 'Feckenham', 'Walpole', 'Bramford',
                 'Pelham', 'Sundon/East Claydon', 'Melksham', 'Bramley', 'London', 'Kemsley', 'Sellindge',
                 'Lovedean', 'S.W.Penisula']
    
    pd_generators = pd.read_csv(path+'generators.csv', index_col=0)
    pd_generators_p_max_pu = pd.read_csv(path+'generators-p_max_pu.csv', index_col=0)
    carrier_list = pd_generators[pd_generators.index.isin(pd_generators_p_max_pu.columns.tolist())].carrier.drop_duplicates().tolist()
    pd_generators.loc[pd_generators.index[pd_generators.carrier.isin(carrier_list)], 'marginal_cost'] = -1 # fixed marginal cost

    pd_generators.to_csv(save_path+'generators.csv')

    df_gsp_data = pd.read_csv('../data/FES2022/GSP_data.csv', encoding='cp1252', index_col=3)
    df_gsp_data = df_gsp_data[['Latitude', 'Longitude']]
    df_gsp_data.rename(columns={'Latitude': 'y', 'Longitude': 'x'}, inplace=True)
    df_gsp_data['Bus'] = dc.map_to_bus(df_gsp_data)

    df_FES_bb = pd.read_excel('../data/FES2022/FES2022 Workbook V4.xlsx', sheet_name='BB1')
    df_P2G = df_FES_bb[(df_FES_bb['FES Scenario']==scenario) & (df_FES_bb['Building Block ID Number']== 'Dem_BB009')].copy()
    df_P2G.insert(6, 'bus', np.nan)
    df_P2G['bus'] = df_P2G.apply(lambda r: gsp_to_bus(r, df_gsp_data), axis = 1)
    df_P2G_year = df_P2G.groupby(df_P2G.bus).sum()[year]

    p_available = pd_generators_p_max_pu.multiply(pd_generators.loc[pd_generators.index[pd_generators.carrier.isin(carrier_list)], 'p_nom'])
    p_available_by_bus = p_available.groupby(pd_generators.bus, axis=1).sum().sum()

    P2G_nom_scotland = p_available_by_bus[p_available_by_bus.index.isin(buses_scotland)] / \
    p_available_by_bus[p_available_by_bus.index.isin(buses_scotland)].sum() * \
    (df_P2G_year[df_P2G_year.index.isin(buses_scotland)].sum() + df_P2G_year[['Direct(SHETL)', 'Direct(SPTL)']].sum()) + \
    p_available_by_bus[p_available_by_bus.index.isin(buses_scotland)] / \
    p_available_by_bus.sum() * df_P2G_year['Not Connected']

    P2G_nom_rgb = p_available_by_bus[p_available_by_bus.index.isin(buses_rgb)] / \
    p_available_by_bus[p_available_by_bus.index.isin(buses_rgb)].sum() * \
    (df_P2G_year[df_P2G_year.index.isin(buses_rgb)].sum() + df_P2G_year['Direct(NGET)']) + \
    p_available_by_bus[p_available_by_bus.index.isin(buses_rgb)] / \
    p_available_by_bus.sum() * df_P2G_year['Not Connected']

    P2G_nom = pd.concat([P2G_nom_scotland, P2G_nom_rgb])

    pd_G2G_storages = pd.DataFrame({'name': [n + ' P2G' for n  in P2G_nom.index.tolist()],
                                    'p_nom': P2G_nom.values,
                                    'carrier': 'P2G',
                                    'marginal_cost': 500,
                                    'max_hours': 999999999,
                                    'efficiency_store': 0.95,
                                    'efficiency_dispatch': 0.95,
                                    'state_of_charge_initial': 0,
                                    'bus': P2G_nom.index})

    pd_storage_units = pd.read_csv(path+'storage_units.csv', index_col=0)
    pd.concat([pd_storage_units, pd_G2G_storages.set_index('name')]).to_csv(save_path+'storage_units.csv')

  



if __name__ == "__main__":
    year = 2045
    scenario = 'System Transformation'
    add_P2G(year, scenario = scenario)
