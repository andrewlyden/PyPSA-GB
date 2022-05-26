import pypsa
import pandas as pd
import matplotlib.pyplot as plt

# essentially want a set of functions for creating a district heating which can connect to one of the PyPSA-GB buses

class EnergyCentre(object):

    def __init__(self, network=None, timestep=None, timestamp_from=None, timestamp_to=None):
        if network is None:
            # if no network is passed then create empty one
            self.network = pypsa.Network()

        self.network.set_snapshots(pd.date_range(timestamp_from, timestamp_to, freq=timestep))
        self.timestep = timestep
        self.timestamp_from = timestamp_from
        self.timestamp_to = timestamp_to

    def get_data_easter_bush(self):

        # read in data here
        # half-hourly
        heat_demand = pd.read_csv('..\data\easter_bush_data\heat_demand.csv', index_col=0)[['Values']].loc[timestamp_from:timestamp_to]['Values']
        heat_demand.index = pd.to_datetime(heat_demand.index, infer_datetime_format=True)
        self.heat_demand = heat_demand / 1000
        # half-hourly
        elec_demand = pd.read_csv('..\data\easter_bush_data\elec_demand.csv', index_col=0)[['Values']].loc[timestamp_from:timestamp_to]['Values']
        elec_demand.index = self.network.snapshots
        self.elec_demand = elec_demand / 1000
        temperature = pd.read_csv(r'..\data\easter_bush_data\air_temp.csv', index_col=0, skiprows=3).dropna(axis=1).drop(columns=['local_time'])
        temperature.index = pd.to_datetime(temperature.index, infer_datetime_format=True)
        temp = pd.DataFrame(temperature.temperature.resample('0.5H').interpolate())
        df_temp_new = pd.DataFrame(
            data=[temp.loc[temp.tail(1).index.values].values[0]],
            columns=temp.columns,
            index=[pd.to_datetime(['2020-12-31 23:30:00'])][0])
        # add to existing dataframe
        temp = temp.append(df_temp_new, sort=False)
        # half-hourly
        date_time_1 = pd.to_datetime([timestamp_from])
        date_time_2 = pd.to_datetime([timestamp_to])
        self.temperature = temp[date_time_1[0]:date_time_2[0]]

        # half-hourly
        self.carbon_intensity = pd.read_csv(r'..\data\easter_bush_data\regional_emissions.csv', index_col=0).loc[timestamp_from:timestamp_to]
        # half-hourly
        tariff = pd.read_csv(r'..\data\easter_bush_data\tariffs.csv', index_col=0)['Wholesale price (p/kWh)']
        tariff.index = pd.to_datetime(tariff.index)
        tariff[tariff < 0] = 0
        # multiply by 10 to convert from p/kWh to £/MWh
        self.tariff = tariff.loc[timestamp_from:timestamp_to] * 10
        self.tariff.index = self.network.snapshots
    
    def get_data_scottish_borders(self):

        timestamp_from = self.timestamp_from
        timestamp_to = self.timestamp_to

        # read in data here
        # heat_demand = pd.read_csv('..\data\scottish_borders_data\heat_demand.csv', index_col=0)[['Values']].loc[timestamp_from:timestamp_to]['Values']
        # heat_demand.index = self.network.snapshots
        # self.heat_demand = heat_demand / 1000
    
        heat_demand = pd.read_csv('..\data\scottish_borders_data\heat_demand.csv', index_col=0)
        heat_demand.index = pd.to_datetime(heat_demand.index, infer_datetime_format=True)
        heat = pd.DataFrame(heat_demand.Values.resample('0.5H').interpolate())
        df_heat_demand_new = pd.DataFrame(
            data=[heat.loc[heat.tail(1).index.values].values[0]],
            columns=heat.columns,
            index=[pd.to_datetime(['2019-12-31 23:30:00'])][0])
        # add to existing dataframe
        heat = heat.append(df_heat_demand_new, sort=False)
        # half-hourly
        date_time_1 = pd.to_datetime([timestamp_from])
        date_time_2 = pd.to_datetime([timestamp_to])
        self.heat_demand = heat[date_time_1[0]:date_time_2[0]]['Values'] / 1000
    
        elec_demand = pd.read_csv('..\data\scottish_borders_data\elec_demand.csv', index_col=0)
        elec_demand.index = pd.to_datetime(elec_demand.index, infer_datetime_format=True)
        elec = pd.DataFrame(elec_demand.Values.resample('0.5H').interpolate())
        df_elec_demand_new = pd.DataFrame(
            data=[elec.loc[elec.tail(1).index.values].values[0]],
            columns=elec.columns,
            index=[pd.to_datetime(['2019-12-31 23:30:00'])][0])
        # add to existing dataframe
        elec = elec.append(df_elec_demand_new, sort=False)
        # half-hourly
        date_time_1 = pd.to_datetime([timestamp_from])
        date_time_2 = pd.to_datetime([timestamp_to])
        self.elec_demand = elec[date_time_1[0]:date_time_2[0]]['Values'] / 1000
    
        temperature = pd.read_csv(r'..\data\easter_bush_data\air_temp.csv', index_col=0, skiprows=3).dropna(axis=1).drop(columns=['local_time'])
        temperature.index = pd.to_datetime(temperature.index, infer_datetime_format=True)
        temp = pd.DataFrame(temperature.temperature.resample('0.5H').interpolate())
        df_temp_new = pd.DataFrame(
            data=[temp.loc[temp.tail(1).index.values].values[0]],
            columns=temp.columns,
            index=[pd.to_datetime(['2020-12-31 23:30:00'])][0])
        # add to existing dataframe
        temp = temp.append(df_temp_new, sort=False)
        # half-hourly
        date_time_1 = pd.to_datetime([timestamp_from])
        date_time_2 = pd.to_datetime([timestamp_to])
        self.temperature = temp[date_time_1[0]:date_time_2[0]]

        # half-hourly
        self.carbon_intensity = pd.read_csv(r'..\data\easter_bush_data\regional_emissions.csv', index_col=0).loc[timestamp_from:timestamp_to]
        # half-hourly
        tariff = pd.read_csv(r'..\data\easter_bush_data\tariffs.csv', index_col=0)['Wholesale price (p/kWh)']
        tariff.index = pd.to_datetime(tariff.index)
        tariff[tariff < 0] = 0
        # multiply by 10 to convert from p/kWh to £/MWh
        self.tariff = tariff.loc[timestamp_from:timestamp_to] * 10
        self.tariff.index = self.network.snapshots

    def add_bus(self, carrier):
        # carrier can be 'heat', 'elec', 'hydrogen'
        self.network.add('Bus', name=carrier, carrier=carrier)

    def add_heat_demand(self,):
        self.network.add('Load', name='Heat_Demand', bus='heat', p_set=self.heat_demand)

    def add_elec_demand(self,):
        self.network.add('Load', name='Elec_Demand', bus='elec', p_set=self.elec_demand)

    def add_grid_connection(self):
        self.network.add('Generator', name='Grid', bus='elec', marginal_cost=self.tariff, p_nom=5000)

    def add_gas_boiler(self):
        self.network.add(
            'Generator',
            name='Gas_Boiler', 
            bus='heat', 
            marginal_cost=30, 
            p_nom=5000,
            efficiency=0.9,
            )

    def add_heat_pump(self):
        self.network.add(
            'Link',
            name="Heat_Pump",
            bus0="elec",
            bus1="heat",
            p_nom=1000000,
            efficiency=3.0,
            # marginal_cost=self.tariff,
            # p_nom_extendable=True,
            # p_nom_min=0,
            # p_nom_max=5000,
            )

    def add_resistive_heater(self):
        self.network.add(
            'Link',
            name="Resistive_Heater",
            bus0="elec",
            bus1="heat",
            p_nom=1000000,
            efficiency=1.0,
            marginal_cost=0.001,
            # p_nom_extendable=True,
            # p_nom_min=0,
            # p_nom_max=5000,
            )

    def add_short_term_store(self,
                sts_cap=14,
                sts_charge=2.940,
                sts_discharge=1.260
                ):
        '''
        Default parameters are as in Renaldi, Friedrich 2018
        (Multiple time grids in operational optimisation of energy systems
        with short- and long-term thermal energy storage)
        (units are in megawatts/megawatthours)
        
        Args:
            rest: capacity, maximal charge and discharge rates of short-term storage respectively
        '''

        self.network.add('Bus', name='heat_sts')

        self.network.add('Store', 
                    name='sts', 
                    bus='heat_sts', 
                    e_nom=sts_cap,
                    )

        self.network.add('Link', 
                    name='heat_sts_charge', 
                    bus0='heat', 
                    bus1='heat_sts',
                    p_nom=sts_charge,
                    )
        self.network.add('Link',
                    name='heat_sts_to_demand',
                    bus0='heat_sts', 
                    bus1='heat',
                    p_nom=sts_discharge,
                    )

    def add_long_term_store(self,
                lts_cap=900,
                lts_charge=0.17,
                lts_discharge=0.17,
                ):
        '''
        Default parameters are as in Renaldi, Friedrich 2018
        (Multiple time grids in operational optimisation of energy systems
        with short- and long-term thermal energy storage)
        (units are in megawatts/megawatthours)
        
        Args:
            rest: capacity, maximal charge and discharge rates of long-term storage respectively
        '''

        self.network.add('Bus', name='heat_lts')

        self.network.add('Store', 
                    name='lts', 
                    bus='heat_lts', 
                    e_nom=lts_cap,
                    )

        self.network.add('Link', 
                    name='heat_lts_charge', 
                    bus0='heat_sts', 
                    bus1='heat_lts',
                    p_nom=lts_charge,
                    )
        self.network.add('Link',
                    name='heat_lts_to_sts', 
                    bus0='heat_lts', 
                    bus1='heat_sts',
                    p_nom=lts_discharge,
                    )

if __name__ == '__main__':

    timestep = '0.5H'
    timestamp_from = '2019-01-01 00:00:00'
    timestamp_to = '2019-12-31 23:30:00'
    myEC = EnergyCentre(timestep=timestep, timestamp_from=timestamp_from, timestamp_to=timestamp_to)
    # read in the data for easter bush
    myEC.get_data_easter_bush()
    # print(myEC.__dict__)

    # add heat and elec buses
    myEC.add_bus('heat')
    myEC.add_bus('elec')
    # print(myEC.network.buses)

    # add heat and elec demands
    myEC.add_heat_demand()
    myEC.add_elec_demand()
    # print(myEC.network.loads_t.p_set)

    # add grid connection as elec generator
    myEC.add_grid_connection()
    # print(myEC.network.generators_t.marginal_cost)
    # add gas boiler
    myEC.add_gas_boiler()

    # add heat pump as link
    myEC.add_heat_pump()
    # add resistive heater as link
    myEC.add_resistive_heater()
    # print(myEC.network.links)
    # print(myEC.network.links_t.marginal_cost)

    # add storage
    myEC.add_short_term_store()
    myEC.add_long_term_store()

    # run LOPF
    myEC.network.lopf(myEC.network.snapshots,
                      solver_name="gurobi",
                    #   pyomo=False,
                    #   keep_shadowprices=True,
                      )
    print(myEC.network.links_t.p0)
    print(myEC.network.links_t.p1)
    print(myEC.network.generators_t.p)
    print(myEC.network.loads_t.p)
    # print(myEC.network.generators)
    print(myEC.network.generators_t.marginal_cost)
    # print(myEC.network.buses_t.marginal_price)

    myEC.network.generators_t.p['Heat_Pump'] = myEC.network.links_t.p1.Heat_Pump * -1
    myEC.network.generators_t.p['Resistive_Heater'] = myEC.network.links_t.p1.Resistive_Heater * -1
    myEC.network.generators_t.marginal_cost['Gas_Boiler'] = myEC.network.generators.marginal_cost.Gas_Boiler

    fig, axs = plt.subplots(4, 1, figsize=(16, 16))

    myEC.network.loads_t.p_set.rename_axis('').plot(ax=axs[0], title='Demands')
    myEC.network.generators_t.p.rename_axis('').drop(columns=['Grid']).plot.area(ax=axs[1], linewidth=0, title='Heat production')
    myEC.network.generators_t.marginal_cost.rename_axis('').plot(ax=axs[2], title='Grid and gas costs')
    myEC.network.stores_t.e.rename_axis('').plot(ax=axs[3], title='Storages state of charge')

    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.show()
