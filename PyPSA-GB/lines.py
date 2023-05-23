import pandas as pd
import os
from allocate_to_zone import map_to_zone

def write_lines(networkmodel):
    if networkmodel == 'Reduced':
        file = '../data/network/BusesBasedGBsystem/network/lines.csv'
        df = pd.read_csv(file)
        df.to_csv('LOPF_data/lines.csv', index=False, header=True)
    elif networkmodel == 'Zonal':
        file = '../data/network/ZonesBasedGBsystem/network/links.csv'
        df = pd.read_csv(file)
        df.to_csv('LOPF_data/lines.csv', index=False, header=True)

def zone_postprocess_generators():
    path = 'LOPF_data/generators.csv'
    df_generators = pd.read_csv(path, index_col=0)
    df_buses = pd.read_csv('../data/network/BusesBasedGBsystem/network/buses.csv')
    df_buses['zone'] = map_to_zone(df_buses, warm=False)
    bus_zone = df_buses.set_index(['name'])['zone'].to_dict()
    df_generators['bus'].replace(bus_zone, inplace=True)
    df_generators.to_csv('LOPF_data/generators.csv', index=True, header=True)

def zone_postprocess_lines_links():
    pd_lines = pd.read_csv('LOPF_data/lines.csv')
    pd_links = pd.read_csv('LOPF_data/links.csv')

    pd.concat([pd_links, pd_lines[pd_links.columns.tolist()]]).to_csv('LOPF_data/links.csv', index=False, header=True)
    os.remove('LOPF_data/lines.csv')
