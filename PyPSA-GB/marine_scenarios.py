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

    # remove the Wave power and Tidal stream in the df_generators
    df_generators = df_generators[~df_generators.carrier.str.contains('Wave power')]
    df_generators = df_generators[~df_generators.carrier.str.contains('Tidal stream')]
    for wave_generator in dic_wave_power['locations'].index:
        # add wave generator to df_generators
        df_generators.loc[wave_generator] = {'carrier': 'Wave power', 'type': 'Wave power',
                                             'p_nom': dic_wave_power['locations'].loc[wave_generator, 'p_nom'] * 1000.,
                                             'bus': dic_wave_power['locations'].loc[wave_generator, 'bus'],
                                              'marginal_cost': 0., 'ramp_limit_up': 1., 'ramp_limit_down': 1.}
    for tidal_stream_generator in dic_tidal_stream['locations'].index:
        # add tidal stream generator to df_generators
        df_generators.loc[tidal_stream_generator] = {'carrier': 'Tidal stream', 'type': 'Tidal stream',
                                                     'p_nom': dic_tidal_stream['locations'].loc[tidal_stream_generator, 'p_nom'] * 1000.,
                                                     'bus': dic_tidal_stream['locations'].loc[tidal_stream_generator, 'bus'],
                                                      'marginal_cost': 0., 'ramp_limit_up': 1., 'ramp_limit_down': 1.}
        
    # check_consistency_with_p_max_pu()

    # save the new dataframe
    # note NOT included for Unit Committment data
    df_generators.to_csv('LOPF_data/generators.csv', header=True)

def check_consistency_with_p_max_pu():

    df_generators = pd.read_csv('LOPF_data/generators.csv', index_col=0)
    df_p_max_pu = pd.read_csv('LOPF_data/generators-p_max_pu.csv', index_col=0)
    print(all(item in df_p_max_pu.columns for item in df_generators.loc[df_generators['carrier'] == 'Wave power'].index))


if __name__ == '__main__':

    for year in [2025, 2030, 2035, 2040, 2045, 2050]:
        for scenario in ['Low', 'Mid', 'High']:
            # print(read_wave_power(year, scenario))
            rewrite_generators_for_marine(year, scenario)
