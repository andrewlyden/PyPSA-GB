import allocate_to_zone
import pandas as pd
import os

# todo: add a loop for [Leading_the_Way], and change the input and output file names and path accordingly 

def gsp_location(name, GSP_info):
    out = {}
    name_list = GSP_info['gsp_name'].tolist()
    if name in name_list:
        info = GSP_info[GSP_info['gsp_name'] == name].copy()
        lat = info['gsp_lat'].tolist()[0]
        lon = info['gsp_lon'].tolist()[0]
    else:
        lat = float('nan')
        lon = float('nan')
        for _ in name.split(';'):
            if _ in name_list:
                info = GSP_info[GSP_info['gsp_name'] == _].copy()
                lat = info['gsp_lat'].tolist()[0]
                lon = info['gsp_lon'].tolist()[0]
                break
    out['x'] = lon
    out['y'] = lat
    return pd.Series(out)

def sum_damand(csv_demand):

    sum_damand_df = pd.DataFrame()
    sum_damand_df.index = [ 'Z1_'+str(i) for i in range(1,5)] + ['Z'+str(i) for i in range(2,18)]
    sum_damand_df.index.name = 'Node'

    sum_damand_df = pd.merge(sum_damand_df,
                             csv_demand.loc[:,'2021 (MW)':].groupby(['Node']).sum().loc[:, '2021 (MW)':'2050 (MW)'].sort_index(),
                             how = 'left',
                             left_index=True,
                             right_index=True)
    sum_damand_df = sum_damand_df.fillna(0)
    sum_damand_df.columns = [x for x in range(2021, 2051)]
    return sum_damand_df

if not os.path.exists('../data/ZonesBasedGBsystem/demand'):
     os.makedirs('../data/ZonesBasedGBsystem/demand')
if not os.path.exists('../data/FES2022/Distributions'):
     os.makedirs('../data/FES2022/Distributions')

# demandpk-all
file_path = '../data/FES2022/FES-2021--Leading_the_Way--demandpk-all--gridsupplypoints.csv'
csv_demand = pd.read_csv(file_path)
file_path = '../data/FES2022/gsp_gnode_directconnect_region_lookup.csv'
csv_GSP = pd.read_csv(file_path)

csv_demand[['x', 'y']] = csv_demand.apply(lambda r: gsp_location(r['Name'], csv_GSP), axis = 1)
csv_demand['Node'] = allocate_to_zone.map_to_zone(csv_demand) # zone

# print(csv_demand[csv_demand.isnull().values == True])

sum_damand_df = sum_damand(csv_demand)
sum_damand_df.to_csv('../data/ZonesBasedGBsystem/demand/Demand_Distribution.csv')

# wind
file_path = '../data/FES2022/FES-2021--Leading_the_Way--mxcapacity-wind--gridsupplypoints.csv'
csv_capacity = pd.read_csv(file_path)
file_path = '../data/FES2022/gsp_gnode_directconnect_region_lookup.csv'
csv_GSP = pd.read_csv(file_path)

csv_capacity[['x', 'y']] = csv_capacity.apply(lambda r: gsp_location(r['Name'], csv_GSP), axis = 1)
csv_capacity['Node'] = allocate_to_zone.map_to_zone(csv_capacity) # zone
sum_damand_df = sum_damand(csv_capacity)
sum_damand_df.to_csv('../data/FES2022/Distributions/Wind Distribution LW.csv')
sum_damand_df.to_csv('../data/ZonesBasedGBsystem/Wind Distribution LW.csv')

# solar, storage, hydro
for capacity in ['solar', 'storage', 'hydro', 'other']:
    file_path = '../data/FES2022/FES-2021--Leading_the_Way--dxcapacity-' + capacity + '--gridsupplypoints.csv'
    csv_capacity = pd.read_csv(file_path)

    csv_capacity[['x', 'y']] = csv_capacity.apply(lambda r: gsp_location(r['Name'], csv_GSP), axis = 1)
    csv_capacity['Node'] = allocate_to_zone.map_to_zone(csv_capacity) # zone
    sum_damand_df = sum_damand(csv_capacity)
    sum_damand_df.to_csv('../data/FES2022/Distributions/'+ capacity.capitalize() + ' Distribution LW.csv')
    sum_damand_df.to_csv('../data/ZonesBasedGBsystem/'+ capacity.capitalize() + ' Distribution LW.csv')


