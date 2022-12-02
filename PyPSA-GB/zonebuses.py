import pandas as pd
import ruamel.yaml
import os
import initialisation_zone_based

# Note the buses' lat and lon in this context is just the equivalent Centroid points in each zone.
file = '../data/network/model.yaml' #sourcs: uk-calliope project https://github.com/calliope-project/uk-calliope

with open(file ) as stream:
    data = ruamel.yaml.safe_load(stream)

k = pd.json_normalize(data['locations'])

# parse yaml file to get bus name, lat and lon
k.columns = k.columns.str.split('.', expand=True)
main_df = k.T.unstack()[0]
df_zonebuses = main_df[['lat','lon']][main_df['lon'].notnull()].reset_index().drop('level_1', axis=1)
df_zonebuses.columns = ['name', 'lat', 'lon']

# change lon->x and lat->y, to match the format used in 29bus pypsa-gb
df_zonebuses.rename(columns={'lon':'x', 'lat':'y'}, inplace=True)

# add arbitary values to voltage level and carrier columns to buses, to match the format used in 29bus pypsa-gb
df_zonebuses['v_nom'] = 400
df_zonebuses['carrier'] = 'AC'

if not os.path.exists('../data/ZonesBasedGBsystem/network/'):
    initialisation_zone_based.create_path()
df_zonebuses.to_csv('../data/ZonesBasedGBsystem/network/buses.csv', index=False, header=True)
