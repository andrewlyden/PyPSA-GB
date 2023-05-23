import pandas as pd
import matplotlib.pyplot as plt
import imageio

import data_reader_writer
import renewables


def read_storage_data(year):
    """reads the storage data from prepared csv file

    NB: this currently only consists of Pumped Storage Hydroelectricity

    Parameters
    ----------
    year : int/str
        year of simulation

    Returns
    -------
    dataframe
        data on storage units
    """

    file = '../data/storage_data.csv'
    df = pd.read_csv(file)
    df1 = renewables.REPD_date_corrected(year)
    df2 = df1.loc[df1['Technology Type'] == 'Pumped Storage Hydroelectricity'].reset_index(
        drop=True)
    df['x'] = df2['lon']
    df['y'] = df2['lat']
    return df


def future_storage(FES):

    if FES == 2021:
        df_FES = pd.read_excel('../data/FES2021/FES 2021 Data Workbook V04.xlsx',
                            sheet_name='FLX1', header=9)
    elif FES == 2022:
        df_FES = pd.read_excel('../data/FES2022/FES2022 Workbook V4.xlsx',
                            sheet_name='FLX1', header=9)
                
    df_battery = df_FES[df_FES.Detail.str.contains('Battery', case=False)]
    cols = [0, 1, 3, 5, 6]
    df_battery.drop(df_battery.columns[cols], axis=1, inplace=True)
    df_battery.dropna(axis='rows', inplace=True)
    df_battery.set_index('Scenario', drop=True, inplace=True)
    df_battery.iloc[:4, 0] = 'p_nom'
    df_battery.iloc[4:, 0] = 'energy capacity'

    df_battery.loc[:, 'type'] = 'Battery'

    df_compressed_air = df_FES[df_FES.Detail.str.contains('Compressed Air', case=False)]
    cols = [0, 1, 3, 5, 6]
    df_compressed_air.drop(df_compressed_air.columns[cols], axis=1, inplace=True)
    df_compressed_air.dropna(axis='rows', inplace=True)
    df_compressed_air.set_index('Scenario', drop=True, inplace=True)
    df_compressed_air.iloc[:4, 0] = 'p_nom'
    df_compressed_air.iloc[4:, 0] = 'energy capacity'
    df_compressed_air.loc[:, 'type'] = 'Compressed Air'

    df_liquid_air = df_FES[df_FES.Detail.str.contains('Liquid Air', case=False)]
    cols = [0, 1, 3, 5, 6]
    df_liquid_air.drop(df_liquid_air.columns[cols], axis=1, inplace=True)
    df_liquid_air.dropna(axis='rows', inplace=True)
    df_liquid_air.set_index('Scenario', drop=True, inplace=True)
    df_liquid_air.iloc[:4, 0] = 'p_nom'
    df_liquid_air.iloc[4:, 0] = 'energy capacity'
    df_liquid_air.loc[:, 'type'] = 'Liquid Air'

    df_pumped_hydro = df_FES[df_FES.Detail.str.contains('Pumped Hydro', case=False)]
    cols = [0, 1, 3, 5, 6]
    df_pumped_hydro.drop(df_pumped_hydro.columns[cols], axis=1, inplace=True)
    df_pumped_hydro.dropna(axis='rows', inplace=True)
    df_pumped_hydro.set_index('Scenario', drop=True, inplace=True)
    df_pumped_hydro.iloc[:4, 0] = 'p_nom'
    df_pumped_hydro.iloc[4:, 0] = 'energy capacity'
    df_pumped_hydro.loc[:, 'type'] = 'Pumped Hydro'

    df_storage = df_battery.append([df_compressed_air, df_liquid_air, df_pumped_hydro])

    return df_storage


def write_storage_units(year, scenario=None, FES=None, networkmodel='Reduced'):
    """writes the buses csv file

    Parameters
    ----------
    year : int/str
        year of simulation
    Returns
    -------
    """

    if networkmodel == 'Reduced':
        from distance_calculator import map_to_bus as map_to
    elif networkmodel == 'Zonal':
        from allocate_to_zone import map_to_zone as map_to

    df = read_storage_data(year)
    df_storage_data_by_type = pd.read_csv('../data/storage_data_by_type.csv', index_col=0)

    if year < 2021:
        df.loc[:, 'state_of_charge_initial'] = df.loc[:, 'p_nom'] * df.loc[:, 'max_hours']
        df_UC = df.drop(columns=['x', 'y'])
        df_UC.to_csv('UC_data/storage_units.csv', index=False, header=True)

        # for the LOPF want to map the storage units to the closest bus
        df_LOPF = df.drop(columns=['x', 'y', 'bus'])
        df_LOPF['bus'] = map_to(df)

        df_LOPF.to_csv('LOPF_data/storage_units.csv', index=False, header=True)

    elif year >= 2021:
        df_future_storage = future_storage(FES)

        # as first pass distribute the batteries across the nodes evenly
        # read in the buses
        df_buses = pd.read_csv('LOPF_data/buses.csv', index_col=0)
        # remove interconnector buses
        df_buses = df_buses[~df_buses.carrier.str.contains('DC')]

        if scenario == 'Leading The Way':
            scenario = 'Leading the Way'

        # HYDRO

        df_hydro = df_future_storage[df_future_storage.type.str.contains('Pumped Hydro', case=False)]
        df_hydro.columns = df_hydro.columns.map(str)
        date = str(year) + '-01-01 00:00:00'
        df_hydro_capacity = df_hydro[df_hydro['Data item'].str.contains('energy capacity', case=False)]
        df_hydro_p_nom = df_hydro[df_hydro['Data item'].str.contains('p_nom', case=False)]
        hydro_capacity = df_hydro_capacity.loc[scenario, date] * 1000
        hydro_p_nom = df_hydro_p_nom.loc[scenario, date] * 1000

        # first simply scale up pumped hydro
        hydro_historical_p_nom = df.loc[:, 'p_nom'].sum()
        scaling_factor_hydro_p_nom = hydro_p_nom / hydro_historical_p_nom

        hydro_historical_capacity = (df.loc[:, 'p_nom'] * df.loc[:, 'max_hours']).sum()
        scaling_factor_hydro_capacity = hydro_capacity / hydro_historical_capacity

        # scale the p_nom and max_hours (used as capacity, max_hours * p_nom = capacity)
        df.loc[:, 'p_nom'] *= scaling_factor_hydro_p_nom
        df.loc[:, 'max_hours'] *= scaling_factor_hydro_capacity
        df.set_index('name', inplace=True)
        # state of charge is set to full

        # BATTERY

        df_battery = df_future_storage[df_future_storage.type.str.contains('Battery', case=False)]
        df_battery.columns = df_battery.columns.map(str)
        date = str(year) + '-01-01 00:00:00'
        df_battery_capacity = df_battery[df_battery['Data item'].str.contains('energy capacity', case=False)]
        df_battery_p_nom = df_battery[df_battery['Data item'].str.contains('p_nom', case=False)]
        battery_capacity = df_battery_capacity.loc[scenario, date] * 1000
        battery_p_nom = df_battery_p_nom.loc[scenario, date] * 1000

        store_list = []
        for bus in df_buses.index.values:
            store = {}
            # create a new row to append to storage dataframe
            store['name'] = bus + ' Battery'
            # can map the stores to bus later
            store['bus'] = 'bus'
            # distribute the p_nom, and max_hours, evenly
            store['p_nom'] = battery_p_nom / len(df_buses.index)
            store['max_hours'] = (battery_capacity / battery_p_nom)
            store['carrier'] = 'Battery'
            store['marginal_cost'] = df_storage_data_by_type.loc['Battery', 'marginal_cost']
            store['efficiency_store'] = df_storage_data_by_type.loc['Battery', 'efficiency_store']
            store['efficiency_dispatch'] = df_storage_data_by_type.loc['Battery', 'efficiency_dispatch']
            store['x'] = df_buses.loc[bus, 'x']
            store['y'] = df_buses.loc[bus, 'y']
            store_list.append(store)

        df_battery = pd.DataFrame(store_list)
        df_battery.set_index('name', inplace=True)

        # Compressed Air

        df_compressed_air = df_future_storage[df_future_storage.type.str.contains('Compressed Air', case=False)]
        df_compressed_air.columns = df_compressed_air.columns.map(str)
        date = str(year) + '-01-01 00:00:00'
        df_compressed_air_capacity = df_compressed_air[df_compressed_air['Data item'].str.contains('energy capacity', case=False)]
        df_compressed_air_p_nom = df_compressed_air[df_compressed_air['Data item'].str.contains('p_nom', case=False)]
        compressed_air_capacity = df_compressed_air_capacity.loc[scenario, date] * 1000
        compressed_air_p_nom = df_compressed_air_p_nom.loc[scenario, date] * 1000

        store_list = []
        for bus in df_buses.index.values:
            store = {}
            # create a new row to append to storage dataframe
            store['name'] = bus + ' Compressed Air'
            # can map the stores to bus later
            store['bus'] = 'bus'
            # distribute the p_nom, and max_hours, evenly
            store['p_nom'] = compressed_air_p_nom / len(df_buses.index)
            store['max_hours'] = (compressed_air_capacity / compressed_air_p_nom)
            store['carrier'] = 'Compressed Air'
            store['marginal_cost'] = df_storage_data_by_type.loc['Compressed Air', 'marginal_cost']
            store['efficiency_store'] = df_storage_data_by_type.loc['Compressed Air', 'efficiency_store']
            store['efficiency_dispatch'] = df_storage_data_by_type.loc['Compressed Air', 'efficiency_dispatch']
            store['x'] = df_buses.loc[bus, 'x']
            store['y'] = df_buses.loc[bus, 'y']
            store_list.append(store)

        df_compressed_air = pd.DataFrame(store_list)
        df_compressed_air.set_index('name', inplace=True)

        # Liquid Air

        df_liquid_air = df_future_storage[df_future_storage.type.str.contains('Liquid Air', case=False)]
        df_liquid_air.columns = df_liquid_air.columns.map(str)
        date = str(year) + '-01-01 00:00:00'
        df_liquid_air_capacity = df_liquid_air[df_liquid_air['Data item'].str.contains('energy capacity', case=False)]
        df_liquid_air_p_nom = df_liquid_air[df_liquid_air['Data item'].str.contains('p_nom', case=False)]
        liquid_air_capacity = df_liquid_air_capacity.loc[scenario, date] * 1000
        liquid_air_p_nom = df_liquid_air_p_nom.loc[scenario, date] * 1000

        store_list = []
        for bus in df_buses.index.values:
            store = {}
            # create a new row to append to storage dataframe
            store['name'] = bus + ' Liquid Air'
            # can map the stores to bus later
            store['bus'] = 'bus'
            # distribute the p_nom, and max_hours, evenly
            store['p_nom'] = liquid_air_p_nom / len(df_buses.index)
            store['max_hours'] = (liquid_air_capacity / liquid_air_p_nom)
            store['carrier'] = 'Liquid Air'
            store['marginal_cost'] = df_storage_data_by_type.loc['Liquid Air', 'marginal_cost']
            store['efficiency_store'] = df_storage_data_by_type.loc['Liquid Air', 'efficiency_store']
            store['efficiency_dispatch'] = df_storage_data_by_type.loc['Liquid Air', 'efficiency_dispatch']
            store['x'] = df_buses.loc[bus, 'x']
            store['y'] = df_buses.loc[bus, 'y']
            store_list.append(store)

        df_liquid_air = pd.DataFrame(store_list)
        df_liquid_air.set_index('name', inplace=True)

        # append dataframes
        df_storage = df.append([df_battery, df_compressed_air, df_liquid_air])

        # WRITE FILES
        df_storage.loc[:, 'state_of_charge_initial'] = df_storage.loc[:, 'p_nom'] * df_storage.loc[:, 'max_hours']
        df_UC = df_storage.drop(columns=['x', 'y'])
        df_UC.to_csv('UC_data/storage_units.csv', index=True, header=True)

        # for the LOPF want to map the storage units to the closest bus
        df_LOPF = df_storage.drop(columns=['x', 'y', 'bus'])
        df_LOPF['bus'] = map_to(df_storage)
        df_LOPF.to_csv('LOPF_data/storage_units.csv', index=True, header=True)

        return df_LOPF


def plot_future_capacities(year):

    start = str(year) + '-12-02 00:00:00'
    end = str(year) + '-12-02 03:30:00'
    # time step as fraction of hour
    time_step = 0.5
    if year > 2020:
        data_reader_writer.data_writer(start, end, time_step, year, year_baseline=2020, scenario='Leading The Way')
    if year <= 2020:
        data_reader_writer.data_writer(start, end, time_step, year)

    df_storage = pd.read_csv('LOPF_data/storage_units.csv', index_col=0)
    storage_p_nom = df_storage.p_nom.groupby(df_storage.carrier).sum()
    # storage_p_nom.drop('Unmet Load', inplace=True)
    # storage_p_nom.drop(storage_p_nom[storage_p_nom < 500].index, inplace=True)

    # bar chart
    plt.figure(figsize=(15, 10))
    col_map = plt.get_cmap('Paired')
    plt.bar(storage_p_nom.index, storage_p_nom.values / 1000, color=col_map.colors, edgecolor='k')
    plt.xticks(storage_p_nom.index)
    plt.ylabel('GW')
    plt.grid(color='grey', linewidth=1, axis='both', alpha=0.5)
    plt.title('Installed capacity in year ' + str(year))
    plt.tight_layout()
    # plt.show()
    plt.savefig('../data/FES2021/Capacities Pics/storage_' + str(year) + '.png')


def gif_future_capacities():

    filenames = []
    for year in range(2021, 2050 + 1):
        plot_future_capacities(year)
        # list of filenames
        filenames.append('../data/FES2021/Capacities Pics/storage_' + str(year) + '.png')

    with imageio.get_writer('../data/FES2021/Capacities Pics/FES_storage_installed_capacities.gif', mode='I', duration=1.) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    year = 2050
    # future_storage()
    # write_storage_units(year, scenario='Leading the Way')
    # plot_future_capacities(year)
    gif_future_capacities()
