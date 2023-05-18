import pandas as pd
import distance_calculator as dc


def read_tidal_stream(year, scenario):

    # scenarios are Low, Mid, High
    # only data for 2025, 2030, 2035, 2040, 2045, 2050
    if year in [2025, 2030, 2035, 2040, 2045, 2050]:

        # read in new marine scenarios
        df_tidal_stream = pd.read_excel('../data/renewables/Marine/tidal_stream_future_deployment_scenarios.xlsx', sheet_name=None)
        if scenario == 'Low':
            sheet_name = 'tidal_stream_low'
        elif scenario == 'Mid':
            sheet_name = 'tidal_stream_mid'
        elif scenario == 'High':
            sheet_name = 'tidal_stream_high'

        df_tidal_stream_capacities = df_tidal_stream[sheet_name].T
        df_tidal_stream_capacities.columns = df_tidal_stream_capacities.iloc[1]
        df_tidal_stream_capacities = df_tidal_stream_capacities.drop(['Lat', 'Lon', 'Site ID', 'Site Name'])
        df_tidal_stream_capacities = df_tidal_stream_capacities.iloc[:, :-1]

        # replace NaN with zero
        df_tidal_stream_capacities = df_tidal_stream_capacities.fillna(0)

        df_tidal_stream_locations = df_tidal_stream[sheet_name].iloc[:, -3:]
        df_tidal_stream_locations.index = df_tidal_stream[sheet_name]['Site ID']
        # drop the last row as this is a total row
        df_tidal_stream_locations.drop(df_tidal_stream_locations.tail(1).index, inplace=True)
        df_tidal_stream_locations.rename(columns={'Lat': 'lat', 'Lon': 'lon', 2050: 'p_nom'}, inplace=True)

        # need to use lat and lon to figure out the nearest bus - then add column called bus
        df = df_tidal_stream_locations.rename(columns={'lat': 'y', 'lon': 'x'})
        buses = dc.map_to_bus(df)
        df_tidal_stream_locations['bus'] = buses

        dic = {'capacities': df_tidal_stream_capacities.loc[year, :], 'locations': df_tidal_stream_locations}

        return dic

def read_wave_power(year, scenario):

    # scenarios are Low, Mid, High
    # only data for 2025, 2030, 2035, 2040, 2045, 2050
    if year in [2025, 2030, 2035, 2040, 2045, 2050]:

        # read in new marine scenarios
        df_wave_power = pd.read_excel('../data/renewables/Marine/wave_power_future_deployment_scenarios.xlsx', sheet_name=None)
        if scenario == 'Low':
            sheet_name = 'wave_power_low'
        elif scenario == 'Mid':
            sheet_name = 'wave_power_mid'
        elif scenario == 'High':
            sheet_name = 'wave_power_high'

        df_wave_power_capacities = df_wave_power[sheet_name].T
        df_wave_power_capacities.columns = df_wave_power_capacities.iloc[1]
        df_wave_power_capacities = df_wave_power_capacities.drop(['Lat', 'Lon', 'Site ID', 'Site Name'])
        df_wave_power_capacities = df_wave_power_capacities.iloc[:, :-1]

        # replace NaN with zero
        df_wave_power_capacities = df_wave_power_capacities.fillna(0)

        df_wave_power_locations = df_wave_power[sheet_name].iloc[:, -3:]
        df_wave_power_locations.index = df_wave_power[sheet_name]['Site ID']
        # drop the last row as this is a total row
        df_wave_power_locations.drop(df_wave_power_locations.tail(1).index, inplace=True)
        df_wave_power_locations.rename(columns={'Lat': 'lat', 'Lon': 'lon', 2050: 'p_nom'}, inplace=True)

        # need to use lat and lon to figure out the nearest bus - then add column called bus
        df = df_wave_power_locations.rename(columns={'lat': 'y', 'lon': 'x'})
        buses = dc.map_to_bus(df)
        df_wave_power_locations['bus'] = buses

        dic = {'capacities': df_wave_power_capacities.loc[year, :], 'locations': df_wave_power_locations}

        return dic

def rewrite_generators_for_marine(year, scenario):

    dic_tidal_stream = read_tidal_stream(year, scenario)
    dic_wave_power = read_wave_power(year, scenario)

    df_generators = pd.read_csv('LOPF_data/generators.csv', index_col=0)

    # get names of buses
    df_buses = pd.read_csv('LOPF_data/buses.csv')[:29]
    bus_location = []
    for i in range(len(df_buses)):
        bus_location.append({'lon': df_buses['x'][i], 'lat': df_buses['y'][i]})
    bus_names = df_buses['name'].values

    # remove the Wave power and Tidal stream in the df_generators
    df_generators = df_generators[~df_generators.carrier.str.contains('Wave power')]
    df_generators = df_generators[~df_generators.carrier.str.contains('Tidal stream')]
    for bus in bus_names:
        # sum the p_nom at each bus for wave and tidal
        p_nom_wave_bus = dic_wave_power['locations'].loc[dic_wave_power['locations']['bus'] == bus].p_nom.sum()
        p_nom_tidal_stream_bus = dic_tidal_stream['locations'].loc[dic_tidal_stream['locations']['bus'] == bus].p_nom.sum()
        # then add the new generation at each bus
        # note 1000 multiplier as data is in GW
        if p_nom_wave_bus > 0.:
            df_generators.loc['Wave power ' + bus] = {'carrier': 'Wave power', 'type': 'Wave power', 'p_nom': p_nom_wave_bus * 1000.,
                                                      'bus': bus, 'marginal_cost': 0., 'ramp_limit_up': 1., 'ramp_limit_down': 1.}
        if p_nom_tidal_stream_bus > 0.:
            df_generators.loc['Tidal stream ' + bus] = {'carrier': 'Tidal stream', 'type': 'Tidal stream', 'p_nom': p_nom_tidal_stream_bus * 1000.,
                                                      'bus': bus, 'marginal_cost': 0., 'ramp_limit_up': 1., 'ramp_limit_down': 1.}
    # save the new dataframe
    # note NOT included for Unit Committment data
    df_generators.to_csv('LOPF_data/generators.csv', header=True)


if __name__ == '__main__':

    for year in [2025, 2030, 2035, 2040, 2045, 2050]:
        for scenario in ['Low', 'Mid', 'High']:
            # print(read_wave_power(year, scenario))
            rewrite_generators_for_marine(year, scenario)
