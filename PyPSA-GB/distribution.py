# distributions can be changed via this script
# using FES2021 data to do this.

import pandas as pd


class Distribution(object):

    def __init__(self, year, scenario):
        path = 'LOPF_data/generators.csv'
        self.df_generators = pd.read_csv(path, index_col=0)

        path = 'LOPF_data/storage_units.csv'
        self.df_storage = pd.read_csv(path, index_col=0)

        self.year = year
        self.scenario = scenario

        # if scenario == 'Leading the Way':
        #     self.scenario = 'LW'
        # elif scenario == 'Consumer Transformation':
        #     self.scenario = 'CT'
        # elif scenario == 'System Transformation':
        #     self.scenario = 'ST'
        # elif scenario == 'Steady Progression':
        #     self.scenario = 'SP'
        # else:
        #     raise NameError('Invalid scenario passed to Distribution class')

        self.df_FES_bb = pd.read_excel('../data/FES2022/FES2022 Workbook V4.xlsx', sheet_name='BB1')
        self.df_id = pd.read_excel('../data/FES2022/Building Block Definitions.xlsx', index_col=0)
        self.df_gsp = pd.read_csv('../data/FES2022/GSP in Scotland.csv', encoding='cp1252')

    def read_building_block(self, tech):

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
            ids = self.df_id.filter(like=tech, axis=0)['Building Block ID Number'].values

        df_FES = self.df_FES_bb[self.df_FES_bb['FES Scenario'].str.contains(self.scenario, case=False, na=False)]
        GSP_list = self.df_gsp['GSP'].values
        df_FES = df_FES[df_FES['GSP'].isin(GSP_list)]
        df_FES = df_FES[df_FES['Building Block ID Number'].isin(ids)]
        return df_FES

    def scotland_total_tech(self, tech):
        df = self.read_building_block(tech)
        # print(df)
        return df[self.year].sum()

    def generation_capacities(self):

        df_id = self.df_id
        df_FES = self.df_FES_bb
        scenario = self.scenario
        year = self.year
        generation_caps = {'Marine': self.scotland_total_tech('Marine'),
                           'Biomass': self.scotland_total_tech('Biomass & Energy Crops (including CHP)'),
                           'Interconnector': self.scotland_total_tech('Interconnector'),
                           'Natural Gas': self.scotland_total_tech('Natural Gas'),
                           'Nuclear': self.scotland_total_tech('Nuclear'),
                           'Hydrogen': self.scotland_total_tech('Hydrogen'),
                           'Hydro': self.scotland_total_tech('Hydro'),
                           'Solar Photovoltaics': self.scotland_total_tech('Solar Generation'),
                           'Wind Onshore': self.scotland_total_tech('Wind Onshore'),
                           'Wind Offshore': self.scotland_total_tech('Wind Offshore')}
        return generation_caps

    def read_scotland_generators(self):
        print(self.df_generators)

    def PV_data(self):

        year = self.year
        scenario = self.scenario
        # get generators dataframe with p_noms to be scaled
        df = self.df_generators
        df_PV = df.loc[df['carrier'] == 'Solar Photovoltaics'].reset_index(drop=True)
        PV_by_bus = df_PV.groupby('bus').sum()["p_nom"]
        PV_by_bus_norm = PV_by_bus / PV_by_bus.sum()

        # read in the future distribution for PV
        path = '../data/FES2021/Distributions/Solar Distribution ' + scenario + '.csv'
        df_PV = pd.read_csv(path, index_col=0)
        df_PV = df_PV[df_PV.index.notnull()]
        df_PV = df_PV.loc[:, ~df_PV.columns.str.contains('^Unnamed')]
        df_PV = df_PV.astype('float')
        # compare indexes
        missing = list(set(df_PV.index.values) - set(PV_by_bus_norm.index.values))
        # drop the missing one in future PV
        df_PV.drop(missing, inplace=True)
        # normalise dataseries
        PV_norm = df_PV / df_PV.sum()

        return {'original': PV_by_bus_norm, 'future': PV_norm[str(year)]}

    def PV_scale(self):
        PV = self.PV_data()
        PV_original = PV['original']
        PV_future = PV['future']

        scaling_factor = (PV_future / PV_original).fillna(0)

        # scale PV for each bus
        for bus in PV_future.index:
            self.df_generators.loc[(self.df_generators.carrier == "Solar Photovoltaics") & (self.df_generators.bus == bus), "p_nom"] *= scaling_factor[bus]

    def wind_onshore_data(self):

        year = self.year
        scenario = self.scenario

        # get generators dataframe with p_noms to be scaled
        df = self.df_generators
        df_wind = df.loc[df['carrier'] == 'Wind Onshore'].reset_index(drop=True)
        wind_by_bus = df_wind.groupby('bus').sum()["p_nom"]
        wind_by_bus_norm = wind_by_bus / wind_by_bus.sum()

        # read in the future distribution for wind
        path = '../data/FES2021/Distributions/Wind Distribution ' + scenario + '.csv'
        df_wind = pd.read_csv(path, index_col=0)
        df_wind = df_wind[df_wind.index.notnull()]
        df_wind = df_wind.loc[:, ~df_wind.columns.str.contains('^Unnamed')]
        df_wind = df_wind.astype('float')
        # compare indexes
        missing = list(set(df_wind.index.values) - set(wind_by_bus_norm.index.values))
        # drop the missing one in future wind
        df_wind.drop(missing, inplace=True)
        # normalise dataseries
        wind_norm = df_wind / df_wind.sum()

        return {'original': wind_by_bus_norm, 'future': wind_norm[str(year)]}

    def wind_onshore_scale(self):
        wind = self.wind_onshore_data()
        wind_original = wind['original']
        wind_future = wind['future']

        scaling_factor = (wind_future / wind_original).fillna(0)
        # scale wind for each bus
        for bus in wind_future.index:
            self.df_generators.loc[(self.df_generators.carrier == 'Wind Onshore') & (self.df_generators.bus == bus), "p_nom"] *= scaling_factor[bus]

    def storage_data(self):

        year = self.year
        scenario = self.scenario

        # get generators dataframe with p_noms to be scaled
        df = self.df_storage
        df_storage = df[df['carrier'].str.contains('Pumped Storage Hydroelectric')==False]
        storage_by_bus = df_storage.groupby('bus').sum()["p_nom"]
        storage_by_bus_norm = storage_by_bus / storage_by_bus.sum()

        # read in the future distribution for storage
        path = '../data/FES2021/Distributions/Storage Distribution ' + scenario + '.csv'
        df_storage = pd.read_csv(path, index_col=0)
        df_storage = df_storage[df_storage.index.notnull()]
        df_storage = df_storage.loc[:, ~df_storage.columns.str.contains('^Unnamed')]
        df_storage = df_storage.astype('float')
        # compare indexes
        missing = list(set(df_storage.index.values) - set(storage_by_bus_norm.index.values))
        # drop the missing one in future storage
        df_storage.drop(missing, inplace=True)
        # normalise dataseries
        storage_norm = df_storage / df_storage.sum()

        return {'original': storage_by_bus_norm, 'future': storage_norm[str(year)]}

    def storage_scale(self):
        storage = self.storage_data()
        storage_original = storage['original']
        storage_future = storage['future']

        df = self.df_storage
        df_pumped_hydro = df[df['carrier'].str.contains('Pumped Storage Hydroelectric')==True]
        df_storage = df[df['carrier'].str.contains('Pumped Storage Hydroelectric')==False]

        scaling_factor = (storage_future / storage_original).fillna(0)
        # scale storage for each bus
        for bus in storage_future.index:
            df_storage.loc[df_storage.bus == bus, "p_nom"] *= scaling_factor[bus]
            df_storage.loc[df_storage.bus == bus, "state_of_charge_initial"] *= scaling_factor[bus]

        # add in pumped hydro again
        self.df_storage = df_storage.append(df_pumped_hydro)

    def update(self):

        # run scaling functions
        self.PV_scale()
        self.wind_onshore_scale()
        self.storage_scale()

        # write generators file
        self.df_generators.to_csv('LOPF_data/generators.csv', index=True, header=True)

        # write storage file
        self.df_storage.to_csv('LOPF_data/storage_units.csv', index=True, header=True)

    

if __name__ == '__main__':
    year = 2050
    scenario = 'Leading the Way'
    myDistribution = Distribution(year, scenario)

    # print(myDistribution.generation_capacities())
    print(myDistribution.read_scotland_generators())

    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Solar Photovoltaics'])
    # myDistribution.PV_scale()
    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Solar Photovoltaics'])

    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Wind Onshore'])
    # myDistribution.wind_onshore_scale()
    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Wind Onshore'])

    # print(myDistribution.df_storage)
    # myDistribution.storage_scale()
    # print(myDistribution.df_storage)


