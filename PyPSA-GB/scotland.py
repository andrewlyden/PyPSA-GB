import pandas as pd
import matplotlib.pyplot as plt    

def read_FES_workbook():
    # read in the FES building blocks worksheet
    df_FES = pd.read_excel(
                    '../data/FES2022/FES2022 Workbook V4.xlsx',
                    sheet_name='BB1')
    return df_FES

def read_building_block_ids():
    df_id = pd.read_excel(
                '../data/FES2022/Building Block Definitions.xlsx', index_col=0)
    return df_id

def GSP_data():
    df = pd.read_csv('../data/FES2022/GSP in Scotland.csv', encoding='cp1252')
    return df

def read_building_block(df_id, df_FES, tech, scenario):

    if tech == 'Hydro':
        ids = ['Gen_BB018']
    elif tech == 'Hydrogen':
        ids = ['Gen_BB023']
    elif tech == 'Natural Gas':
        ids = ['Gen_BB008', 'Gen_BB009']
    elif tech == 'Batteries':
        ids = ['Srg_BB001']
    elif tech == 'Domestic Batteries':
        ids = ['Srg_BB002']
    elif tech == 'Pumped Hydro':
        ids = ['Srg_BB003']
    elif tech == 'Other':
        ids = ['Srg_BB004']
    elif tech == 'V2G':
        ids = ['Srg_BB005']
    else:
        ids = df_id.filter(like=tech, axis=0)['Building Block ID Number'].values

    df_FES = df_FES[df_FES['FES Scenario'].str.contains(scenario, case=False, na=False)]
    GSP_list = GSP_data()['GSP'].values
    df_FES = df_FES[df_FES['GSP'].isin(GSP_list)]
    df_FES = df_FES[df_FES['Building Block ID Number'].isin(ids)]
    return df_FES

def scotland_total_tech(df_id, df_FES, tech, scenario, year):
    df = read_building_block(df_id, df_FES, tech, scenario)
    # print(df)
    return df[year].sum()

def generation_capacities(scenario, year=2050):

    df_id = read_building_block_ids()
    df_FES = read_FES_workbook()
    generation_caps = {'Marine': scotland_total_tech(df_id, df_FES, 'Marine', scenario, year),
                       'Biomass': scotland_total_tech(df_id, df_FES, 'Biomass & Energy Crops (including CHP)', scenario, year),
                       'Interconnector': scotland_total_tech(df_id, df_FES, 'Interconnector', scenario, year),
                       'Natural Gas': scotland_total_tech(df_id, df_FES, 'Natural Gas', scenario, year),
                       'Nuclear': scotland_total_tech(df_id, df_FES, 'Nuclear', scenario, year),
                       'Hydrogen': scotland_total_tech(df_id, df_FES, 'Hydrogen', scenario, year),
                       'Hydro': scotland_total_tech(df_id, df_FES, 'Hydro', scenario, year),
                       'Solar Photovoltaics': scotland_total_tech(df_id, df_FES, 'Solar Generation', scenario, year),
                       'Wind Onshore': scotland_total_tech(df_id, df_FES, 'Wind Onshore', scenario, year),
                       'Wind Offshore': scotland_total_tech(df_id, df_FES, 'Wind Offshore', scenario, year)}
    print(generation_caps)

def storage_capacities(scenario, year=2050):

    df_id = read_building_block_ids()
    df_FES = read_FES_workbook()
    generation_caps = {'Batteries': scotland_total_tech(df_id, df_FES, 'Batteries', scenario, year),
                       'Domestic Batteries': scotland_total_tech(df_id, df_FES, 'Domestic Batteries', scenario, year),
                       'Pumped Hydro': scotland_total_tech(df_id, df_FES, 'Pumped Hydro', scenario, year),
                       'Other': scotland_total_tech(df_id, df_FES, 'Other', scenario, year),
                       'V2G': scotland_total_tech(df_id, df_FES, 'V2G', scenario, year)}
    print(generation_caps)

def plot():
    # bar chart
    plt.figure(figsize=(12, 8))
    col_map = plt.get_cmap('Paired')
    plt.bar(generators_p_nom.index, generators_p_nom.values / 1000, color=col_map.colors, edgecolor='k')
    plt.xticks(generators_p_nom.index, rotation=90)
    # plt.ylim([0, 50])
    plt.ylabel('GW')
    plt.grid(color='grey', linewidth=1, axis='both', alpha=0.5)
    plt.title('Installed capacity in year ' + str(year))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    scenario = 'Leading the Way'
    # tech = 'Wind Onshore'
    # year = 2050
    # # print(building_block_ids(tech))
    # # ids = building_block_ids(tech)
    # # read_building_block(ids)
    # scotland_total_tech(tech, year)

    # generation_capacities(year=2021)
    for year in [2021, 2030, 2035, 2040, 2045]:
        print(year)
        generation_capacities(scenario, year=year)
        storage_capacities(scenario, year=year)
