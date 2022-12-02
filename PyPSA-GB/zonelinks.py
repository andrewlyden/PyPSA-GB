import pandas as pd
import ruamel.yaml
import os
import initialisation_zone_based

file = '../data/network/transmission_grid_2030.yaml'

with open(file ) as stream:
    data = ruamel.yaml.safe_load(stream)

k = pd.json_normalize(data)
k.columns = k.columns.str.split('.', expand=True)
main_df = k.T.unstack()[0]
df_zonelinks = main_df[main_df['energy_cap_equals'].notnull()].reset_index()[['level_1', 'level_3', 'energy_cap_equals']]
df_zonelinks.columns = ['name', 'carrier', 'p_nom']

def preprocess(row):
    out = {}
    name = row['name'].split(',')
    out['bus0'] = name[0]
    out['bus1'] = name[1]

    carrier = row['carrier']
    if carrier == 'hvac':
        out['carrier'] = 'AC'
    else:
        out['carrier'] = 'DC'
    return pd.Series(out)

df_zonelinks[['bus0', 'bus1', 'carrier']] = df_zonelinks.apply(preprocess, axis=1)
df_zonelinks = df_zonelinks[['name', 'bus0', 'bus1', 'carrier', 'p_nom']]

df_zonelinks['marginal_cost'] = 0
df_zonelinks['p_min_pu'] = -1
df_zonelinks['p_max_pu'] = 1

if not os.path.exists('../data/ZonesBasedGBsystem/network/'):
    initialisation_zone_based.create_path()
df_zonelinks.to_csv('../data/ZonesBasedGBsystem/network/links.csv', index=False, header=True)




#     file = '../data/network/lines.csv'
#     df = pd.read_csv(file)
#     df.to_csv('LOPF_data/lines.csv', index=False, header=True)
