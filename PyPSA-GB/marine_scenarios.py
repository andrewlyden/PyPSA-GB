import pandas as pd
import distance_calculator as dc


def read_tidal_stream(year, scenario, networkmodel):

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

        if networkmodel == 'Reduced':
            from distance_calculator import map_to_bus as map_to
        elif networkmodel == 'Zonal':
            from allocate_to_zone import map_to_zone as map_to

        # need to use lat and lon to figure out the nearest bus - then add column called bus
        df = df_tidal_stream_locations.rename(columns={'lat': 'y', 'lon': 'x'})
        buses = map_to(df)
        df_tidal_stream_locations['bus'] = buses

        dic = {'capacities': df_tidal_stream_capacities.loc[year, :], 'locations': df_tidal_stream_locations}

        return dic

def read_wave_power(year, scenario, networkmodel):

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

        if networkmodel == 'Reduced':
            from distance_calculator import map_to_bus as map_to
        elif networkmodel == 'Zonal':
            from allocate_to_zone import map_to_zone as map_to

        # need to use lat and lon to figure out the nearest bus - then add column called bus
        df = df_wave_power_locations.rename(columns={'lat': 'y', 'lon': 'x'})
        buses = map_to(df)
        df_wave_power_locations['bus'] = buses

        dic = {'capacities': df_wave_power_capacities.loc[year, :], 'locations': df_wave_power_locations}

        return dic
    

def read_floating_wind(year, scenario):

    # scenarios are Low, Mid, High
    # only data for 2025, 2030, 2035, 2040, 2045, 2050
    if year >= 2025:

        # read in new marine scenarios
        df_floating_wind = pd.read_excel('../data/renewables/Marine/floating_wind_deployment_scenarios.xlsx', sheet_name=None)
        if scenario == 'Low':
            sheet_name = 'Low'
        elif scenario == 'Mid':
            sheet_name = 'Mid'
        elif scenario == 'High':
            sheet_name = 'High'

        df_floating_wind_capacities = df_floating_wind[sheet_name].T
        df_floating_wind_capacities.columns = df_floating_wind_capacities.iloc[1]
        df_floating_wind_capacities = df_floating_wind_capacities.drop(['Lat', 'Lon', 'Site ID', 'Site Name'])
        df_floating_wind_capacities = df_floating_wind_capacities.iloc[:, :-1]

        # replace NaN with zero
        # divide by 1000 to convert to same units as other marine
        df_floating_wind_capacities = df_floating_wind_capacities.fillna(0)
        df_floating_wind_capacities = df_floating_wind_capacities / 1000.

        df_floating_wind_locations = df_floating_wind[sheet_name].iloc[:, -3:]
        df_floating_wind_locations.index = df_floating_wind[sheet_name]['Site ID']
        # drop the last row as this is a total row
        df_floating_wind_locations.drop(df_floating_wind_locations.tail(1).index, inplace=True)
        df_floating_wind_locations.rename(columns={'Lat': 'lat', 'Lon': 'lon', 2050: 'p_nom'}, inplace=True)

        # need to use lat and lon to figure out the nearest bus - then add column called bus
        df = df_floating_wind_locations.rename(columns={'lat': 'y', 'lon': 'x'})
        buses = dc.map_to_bus(df)
        df_floating_wind_locations['bus'] = buses

        dic = {'capacities': df_floating_wind_capacities.loc[year, :], 'locations': df_floating_wind_locations}

        return dic


def rewrite_generators_for_marine(year, tech, scenario, networkmodel='Reduced'):

    if tech == 'Floating wind':
        dic = read_floating_wind(year, scenario)
    elif tech == 'Tidal stream':
        dic = read_tidal_stream(year, scenario, networkmodel)
    elif tech == 'Wave power':
        dic = read_wave_power(year, scenario, networkmodel)

    df_generators = pd.read_csv('LOPF_data/generators.csv', index_col=0)

    # remove the Wave power and Tidal stream in the df_generators
    if tech == 'Floating wind':
        df_generators = df_generators[~df_generators.type.str.contains('Floating wind')]
    elif tech == 'Tidal stream':
        df_generators = df_generators[~df_generators.carrier.str.contains('Tidal stream')]
    elif tech == 'Wave power':    
        df_generators = df_generators[~df_generators.carrier.str.contains('Wave power')]
    
    for generator in dic['locations'].index:
        # add wave generator to df_generators
        if tech == 'Tidal stream' or tech == 'Wave power':
            df_generators.loc[generator] = {'carrier': tech, 'type': tech,
                                            'p_nom': dic['locations'].loc[generator, 'p_nom'] * 1000.,
                                            'bus': dic['locations'].loc[generator, 'bus'],
                                            'marginal_cost': 0., 'ramp_limit_up': 1., 'ramp_limit_down': 1.,
                                            'p_max_pu': 1}
        elif tech == 'Floating wind':
            df_generators.loc[generator] = {'carrier': 'Wind Offshore', 'type': tech,
                                            'p_nom': dic['locations'].loc[generator, 'p_nom'] * 1000.,
                                            'bus': dic['locations'].loc[generator, 'bus'],
                                            'marginal_cost': 0., 'ramp_limit_up': 1., 'ramp_limit_down': 1.,
                                            'p_max_pu': 1, 'p_min_pu': 0}
        
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
