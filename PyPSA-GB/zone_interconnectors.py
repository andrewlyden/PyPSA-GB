import allocate_to_zone
import pandas as pd
import os
import initialisation_zone_based

if os.path.exists('../data/BusesBasedGBsystem/network/buses.csv'):
    initialisation_zone_based.copy_buses_based()
pd_buses = pd.read_csv('../data/network/buses.csv')

pd_buses['zone'] = allocate_to_zone.map_to_zone(pd_buses)

def repalce_to_zone(row, pd_buses):
    out = {}
    bus1 = row['bus1']
    zone = pd_buses[pd_buses['name'] == bus1]['zone'].tolist()[0]
    out['bus1'] = zone
    return pd.Series(out)

if os.path.exists('../data/BusesBasedGBsystem/interconnectors/links.csv'):
    initialisation_zone_based.copy_buses_based()
pd_links = pd.read_csv('../data/BusesBasedGBsystem/interconnectors/links.csv')
pd_links['bus1'] = pd_links.apply(lambda r: repalce_to_zone(r, pd_buses), axis = 1)

if not os.path.exists('../data/ZonesBasedGBsystem/interconnectors/'):
     initialisation_zone_based.create_path()
pd_links.to_csv('../data/ZonesBasedGBsystem/interconnectors/links.csv', index = None)

if os.path.exists('../data/BusesBasedGBsystem/interconnectors/links.csv'):
    initialisation_zone_based.copy_buses_based()
pd_links_future = pd.read_csv('../data/BusesBasedGBsystem/interconnectors/links.csv')
pd_links_future['bus1'] = pd_links_future.apply(lambda r: repalce_to_zone(r, pd_buses), axis = 1)

pd_links_future.to_csv('../data/ZonesBasedGBsystem/interconnectors/links_future.csv', index = None)