import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import osgb
import imageio

import renewables_ninja_data_analysis
import snapshots
import distance_calculator as dc
import data_reader_writer


def read_REPD():
    """reads the REDP (renewable energy planning database) and converts to dataframe

    Parameters
    ----------
    currently none, as file is fixed below

    Returns
    -------
    dataframe
        dataframe of REPD with important fields
    """

    # name of file here
    file = '../data/renewables/renewable-energy-planning-database-q1-march-2021.csv'

    # only important fields
    fields = ['Site Name', 'Technology Type', 'Installed Capacity (MWelec)',
              'CHP Enabled', 'Country',
              'Turbine Capacity (MW)', 'No. of Turbines', 'Height of Turbines (m)',
              'Mounting Type for Solar', 'Development Status',
              'X-coordinate', 'Y-coordinate', 'Operational']
    # reads csv
    df = pd.read_csv(file, encoding='unicode_escape', usecols=fields,
                     lineterminator='\n')
    # remove northern island sites
    df.drop(df[df['Country'] == 'Northern Ireland'].index, inplace=True)
    # then drop the country column
    df.drop(columns=['Country'], inplace=True)
    # only want operational sites
    df = df.loc[df['Development Status'] == 'Operational']
    # drop the entries without a capacity value or a given technology type or site name
    df = df.dropna(subset=['Installed Capacity (MWelec)', 'Technology Type', 'Site Name'])
    # reset the index
    df = df.reset_index(drop=True)

    # check for missing location data
    if df['X-coordinate'].isnull().values.any() is True:
        raise Exception("All X-coordinate values must be provided, please fill in missing data in csv file")
    if df['Y-coordinate'].isnull().values.any() is True:
        raise Exception("All Y-coordinate values must be provided, please fill in missing data in csv file")

    # create two lists of conversions from OSGB to lat/lon
    lon = []
    lat = []
    for i in range(len(df.index)):
        x = df['X-coordinate'][i]
        y = df['Y-coordinate'][i]
        coord = osgb.grid_to_ll(x, y)
        lat.append(coord[0])
        lon.append(coord[1])
    df['lon'] = lon
    df['lat'] = lat

    # different technology types
    # currently no capacity of the following:
    # Advanced Conversion Technology, Fuel Cell (Hydrogen), Hot Dry Rocks (HDR),

    # print(df.loc[df['Technology Type'] == 'Advanced Conversion Technology'])
    # print(df.loc[df['Technology Type'] == 'Anaerobic Digestion'])
    # print(df.loc[df['Technology Type'] == 'Biomass (co-firing)'])
    # print(df.loc[df['Technology Type'] == 'Biomass (dedicated)'])
    # print(df.loc[df['Technology Type'] == 'EfW Incineration'])
    # print(df.loc[df['Technology Type'] == 'Fuel Cell (Hydrogen)'])
    # print(df.loc[df['Technology Type'] == 'Hot Dry Rocks (HDR)'])
    # print(df.loc[df['Technology Type'] == 'Landfill Gas'])
    # print(df.loc[df['Technology Type'] == 'Large Hydro'])
    # print(df.loc[df['Technology Type'] == 'Pumped Storage Hydroelectricity'])
    # print(df.loc[df['Technology Type'] == 'Sewage Sludge Digestion'])
    # print(df.loc[df['Technology Type'] == 'Shoreline Wave'])
    # print(df.loc[df['Technology Type'] == 'Small Hydro'])
    # print(df.loc[df['Technology Type'] == 'Solar Photovoltaics'])
    # print(df.loc[df['Technology Type'] == 'Tidal Barrage and Tidal Stream'])
    # print(df.loc[df['Technology Type'] == 'Wind Offshore'])
    # print(df.loc[df['Technology Type'] == 'Wind Onshore'])

    return df


def REPD_date_corrected(year):
    """corrects the REDP (renewable energy planning database) according to year of simulation

    Parameters
    ----------
    year : int/str
        year of simulation

    Returns
    -------
    dataframe
        dataframe of REPD filtered by operational in year of simulation
    """

    df = read_REPD()
    df2 = df['Operational']
    df2 = pd.to_datetime(df2).dt.to_period('D')
    # cut off is the end of the year being simulated
    date = '31/12/' + str(year)
    df = df[~(df2 > date)]

    return df


def fix_timeseries_res_for_year(path, year, tech, future):
    """fixes the timeseries of renewable profiles according to year

    looks at the REPD and filters out from the timeseries those which
    were not operational on they year to be modelled.
    Note that in another function values are set to zero according to
    the date operational within this year.

    Parameters
    ----------
    path : str
        The file path location of the timeseries
    year: int/str
        Year of simulation
    tech: str
        Technology type, e.g. wind offshore, solar photovoltaics
    future: bool
        Is the year to be modelled in future or not, e.g., future is true, past is false

    Returns
    -------
    dataframe
        the fixed timeseries
    """
    if tech == 'Solar Photovoltaics':
        # the solar outputs csv files needed to be split up so this appends them together again
        path = '../data/renewables/atlite/outputs/PV/PV_' + str(year) + '_1' + '.csv'
        df1 = pd.read_csv(path, index_col=0)
        for c in range(2, 5):
            path = '../data/renewables/atlite/outputs/PV/PV_' + str(year) + '_' + str(c) + '.csv'
            df = pd.read_csv(path, index_col=0, header=None)
            df.columns = df1.columns
            df1 = df1.append(df)

    else:
        # just read csv file using given path
        df1 = pd.read_csv(path, index_col=0)

    df1.index = pd.to_datetime(df1.index)
    # want to ensure no duplicate names
    cols = pd.Series(df1.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    # rename the columns with the cols list.
    df1.columns = cols
    df1.columns = df1.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df1.columns = df1.columns.astype(str).str.replace(u'\xa0', '')
    df1.columns = df1.columns.astype(str).str.replace('ì', 'i')
    df1.columns = df1.columns.str.strip()

    # only filter historical years, for future years want all the RES units
    if future is False:

        df_res = REPD_date_corrected(year)
        df_tech = df_res.loc[df_res['Technology Type'] == tech]
        df_tech['Site Name'] = df_tech['Site Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        df_tech['Site Name'] = df_tech['Site Name'].astype(str).str.replace(u'\xa0', '')
        df_tech['Site Name'] = df_tech['Site Name'].astype(str).str.replace('ì', 'i')
        df_tech['Site Name'] = df_tech['Site Name'].str.strip()

        # check for duplicates
        # check names are unique
        duplicateDFRow = df_tech[df_tech.duplicated(['Site Name'], keep='first')]
        # print(duplicateDFRow)
        # rename duplicates
        for i in range(len(duplicateDFRow.index.values)):
            # print(df_tech['name'][duplicateDFRow.index.values[i]])
            df_tech.at[duplicateDFRow.index.values[i], 'Site Name'] = (
                df_tech['Site Name'][duplicateDFRow.index.values[i]] + '.1')
            # print(df['name'][duplicateDFRow.index.values[i]])

        # narrow dataframe timeseries to those operational in the required year
        # print(df1)
        # print(len(df_tech['Operational'].values))
        df1 = df1[df_tech['Site Name'].values]
        # print(df1)
        # print('here?')

        # also want to return zeroes for before date
        # change to datetime to compare
        df2 = pd.to_datetime(df_tech['Operational']).dt.to_period('d')
        # df_tech['date'] = df2
        # df_tech.loc['date'] = df2
        mask = df2 > '01/01/' + str(year)
        # filtered df to just the year in question
        filtered_df = df_tech.loc[mask]
        # list of sites to change timeseries
        sites = filtered_df['Site Name'].values
        # change value of the timeseries based on date operational
        length = len(sites)

        for name in range(length):

            # get the operational date and convert to datetime
            date_operational = pd.to_datetime(filtered_df['Operational'].values[name], dayfirst=True)
            # times before operational dates set to zero
            df1[sites[name]].loc[df1.index[0]:date_operational] = 0.

    return df1


def read_hydro(year):
    """reads hydro data from REDP

    Parameters
    ----------
    year : int/str
        year of simulation

    Returns
    -------
    dataframe
        dataframe of hydro data
    """

    df = REPD_date_corrected(year)
    df1 = df.loc[df['Technology Type'] == 'Large Hydro']
    df2 = df.loc[df['Technology Type'] == 'Small Hydro']
    df3 = df.loc[df['Technology Type'] == 'Pumped Storage Hydroelectricity']
    df_REDP = df1.append([df2, df3], ignore_index=True, sort=False)
    df_REDP = df_REDP.rename(columns={'Site Name': 'name', 'Technology Type': 'type',
                                      'Installed Capacity (MWelec)': 'p_nom'})
    df_REDP = df_REDP[['name', 'type', 'p_nom', 'lat', 'lon']]
    # print(df_REDP)

    file = '../data/renewables/hydro_DUKES_2020.csv'
    df_dukes = pd.read_csv(file, encoding='unicode_escape')
    df_dukes.loc[:, 'Geocoordinates'] = df_dukes['Geocoordinates'].str.replace(',', '')
    df_dukes.loc[:, 'Geocoordinates'] = df_dukes['Geocoordinates'].str.split()

    lat = []
    lon = []
    for i in range(len(df_dukes['Geocoordinates'])):
        lat.append(df_dukes['Geocoordinates'].values[i][0])
        lon.append(df_dukes['Geocoordinates'].values[i][1])

    df_dukes['lat'] = lat
    df_dukes['lon'] = lon

    df_dukes = df_dukes.rename(columns={'Station Name': 'name', 'Type': 'type',
                                        'Installed Capacity (MW)': 'p_nom'})
    df_dukes = df_dukes[['name', 'type', 'p_nom', 'lat', 'lon']]

    df_hydro = df_dukes.append(df_REDP, ignore_index=True, sort=False)

    df_hydro2 = df_hydro.drop_duplicates(subset=['name'])

    conditions = [
        (df_hydro2['p_nom'] > 5.0) & (df_hydro2['type'] != 'Pumped Storage Hydroelectricity'),
        (df_hydro2['p_nom'] <= 5.0) & (df_hydro2['type'] != 'Pumped Storage Hydroelectricity'),
        (df_hydro2['type'] == 'Pumped Storage Hydroelectricity')]
    type_ = ['Large Hydro', 'Small Hydro', 'Pumped Storage Hydroelectricity']
    df_hydro2.loc[:, 'type'] = np.select(conditions, type_)
    df_hydro2 = df_hydro2.reset_index(drop=True)
    df_hydro2['carrier'] = df_hydro2['type']

    return df_hydro2


def read_hydro_time_series(year):
    """reads hydro timeseries as saved in relevant folder

    currently this is reading ELEXON data, see the read_csv below

    Parameters
    ----------
    year : int/str
        year of simulation

    Returns
    -------
    dict
        contains two dataframes of unnorm and normalised timeseries for hydro output
    """

    df = pd.read_csv('../data/renewables/generation_2015-02-22_2020-12-30_ELEXON.csv')
    dti = pd.date_range(start='2015-02-22 00:00:00', end='2020-12-31 23:30:00', freq='0.5H')
    df = df.set_index(dti)
    df_hydro = df[['npshyd']]

    # want to distribute this among all of the non-pumped hydro generators
    df2 = read_hydro(year)
    df2.index = df2['name']
    # delete pumped storage
    df2 = df2[~df2['type'].isin(['Pumped Storage Hydroelectricity'])]
    total_capacity = df2['p_nom'].sum()
    # add a normalised value for each hydro scheme
    df2['normalised'] = df2['p_nom'] / total_capacity
    # multiply the normalised by the time series to get
    # time series for each hydro scheme
    total_time_series = df_hydro['npshyd'].values

    df_hydro_norm = pd.DataFrame(index=df_hydro.index)

    for i in range(len(df2)):

        name = df2.index[i]
        df_hydro.loc[:, name] = df2.loc[name, 'normalised'] * total_time_series
        df_hydro_norm.loc[:, name] = (
            df2.loc[name, 'normalised'] * total_time_series / df2.loc[name, 'p_nom'])

    # drop the total column
    df_hydro = df_hydro.drop(columns=['npshyd'])

    return {'time_series': df_hydro, 'time_series_norm': df_hydro_norm}


def read_non_dispatchable_continuous(year):
    """reads the continuous renewable generators from REPD and converts to dataframe

    Parameters
    ----------
    year : int/str
        year of simulation

    Returns
    -------
    dataframe
        data on continuous renewable generators
    """

    df = REPD_date_corrected(year)
    df1 = df.loc[df['Technology Type'] == 'Anaerobic Digestion']
    df2 = df.loc[df['Technology Type'] == 'EfW Incineration']
    df3 = df.loc[df['Technology Type'] == 'Landfill Gas']
    df4 = df.loc[df['Technology Type'] == 'Sewage Sludge Digestion']
    df5 = df.loc[df['Technology Type'] == 'Shoreline Wave']
    df6 = df.loc[df['Technology Type'] == 'Tidal Barrage and Tidal Stream']
    df_NDC = df1.append([df2, df3, df4, df5, df6], ignore_index=True, sort=False)
    df_NDC = df_NDC.rename(columns={'Site Name': 'name', 'Technology Type': 'type',
                                    'Installed Capacity (MWelec)': 'p_nom'})
    df_NDC = df_NDC[['name', 'type', 'p_nom', 'lat', 'lon']]
    df_NDC['carrier'] = df_NDC['type']
    return df_NDC


def read_biomass(year):
    """reads the biomass generators from REPD and converts to dataframe

    Parameters
    ----------
    year : int/str
        year of simulation

    Returns
    -------
    dataframe
        data on biomass
    """

    df = REPD_date_corrected(year)
    df1 = df.loc[df['Technology Type'] == 'Biomass (co-firing)']
    df2 = df.loc[df['Technology Type'] == 'Biomass (dedicated)']
    df_biomass = df1.append(df2, ignore_index=True, sort=False)
    df_biomass = df_biomass.rename(columns={'Site Name': 'name', 'Technology Type': 'type',
                                            'Installed Capacity (MWelec)': 'p_nom'})
    df_biomass = df_biomass[['name', 'type', 'p_nom', 'lat', 'lon']]
    df_biomass['carrier'] = df_biomass['type']
    return df_biomass


def scale_biomass_p_nom(year, scenario):

    tech = 'Biomass'
    future_capacities_dict = future_RES_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

    t1 = 'Biomass (co-firing)'
    t2 = 'Biomass (dedicated)'

    # get generators dataframe with p_noms to be scaled
    path = 'LOPF_data/generators.csv'
    generators = pd.read_csv(path, index_col=0)
    gen_tech1 = generators.loc[generators['carrier'] == t1]
    gen_tech2 = generators.loc[generators['carrier'] == t2]
    gen_tech = gen_tech1.append(gen_tech2)

    path_UC = 'UC_data/generators.csv'
    generators_UC = pd.read_csv(path_UC, index_col=0)
    gen_tech_UC1 = generators_UC.loc[generators_UC['carrier'] == t1]
    gen_tech_UC2 = generators_UC.loc[generators_UC['carrier'] == t2]
    gen_tech_UC = gen_tech_UC1.append(gen_tech_UC2)

    # then consider what scaling factor is required
    scaling_factor = round(tech_cap_FES / tech_cap_year, 2)

    # scale the p_noms of the RES generators
    for g in gen_tech.index:
        gen_tech.loc[g, 'p_nom'] *= scaling_factor
        gen_tech_UC.loc[g, 'p_nom'] *= scaling_factor

    # write new generators.csv file
    # save the dataframes to csv
    # need to remove the original tech
    # but first keep Biomass CCS
    biomass_CCS = generators.loc[generators['carrier'] == 'CCS Biomass']
    biomass_CCS_UC = generators_UC.loc[generators_UC['carrier'] == 'CCS Biomass']

    generators = generators[~generators.carrier.str.contains(tech)]
    generators_UC = generators_UC[~generators_UC.carrier.str.contains(tech)]

    # then add the new p_nom tech as well as CCS Biomass
    generators = generators.append([gen_tech, biomass_CCS])
    generators_UC = generators_UC.append([gen_tech_UC, biomass_CCS_UC])

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def read_tidal_lagoon(year, scenario):

    df_tidal_lagoon = pd.read_excel('../data/renewables/Marine/tidal_lagoon_future_deployment_scenarios.xlsx', sheet_name=None)
    # print(df_tidal_lagoon)

    if scenario == 'Leading The Way':
        sheet_name = 'tidal_lagoon_LW'
    elif scenario == 'Consumer Transformation':
        sheet_name = 'tidal_lagoon_CT'
    elif scenario == 'System Transformation':
        sheet_name = 'tidal_lagoon_ST'
    elif scenario == 'Steady Progression':
        sheet_name = 'tidal_lagoon_SP'

    df_tidal_lagoon_capacities = df_tidal_lagoon[sheet_name].T
    df_tidal_lagoon_capacities.columns = df_tidal_lagoon_capacities.iloc[0]
    df_tidal_lagoon_capacities = df_tidal_lagoon_capacities.drop(['Lat', 'Long', 'Suggested Bus', 'Site ID', 'Site Name'])
    df_tidal_lagoon_capacities = df_tidal_lagoon_capacities.drop(columns=['Total'])

    # need to get values for years 2020 - 2050
    index_ = ['2025-01-01', '2030-01-01', '2035-01-01', '2040-01-01', '2045-01-01', '2050-01-01']
    df_tidal_lagoon_capacities.index = index_
    df_tidal_lagoon_capacities.index = pd.to_datetime(df_tidal_lagoon_capacities.index, infer_datetime_format=True, utc=True)
    df_tidal_lagoon_capacities = df_tidal_lagoon_capacities.resample('12MS').asfreq()
    df_tidal_lagoon_capacities = df_tidal_lagoon_capacities.astype(float)
    df_tidal_lagoon_capacities = df_tidal_lagoon_capacities.interpolate(method='linear', limit_direction='forward')
    df_tidal_lagoon_capacities = df_tidal_lagoon_capacities.fillna(0)

    df_tidal_lagoon_locations = df_tidal_lagoon[sheet_name].iloc[:, -3:]
    df_tidal_lagoon_locations.index = df_tidal_lagoon[sheet_name]['Site Name']
    # drop the last row as this is a total row
    df_tidal_lagoon_locations.drop(df_tidal_lagoon_locations.tail(1).index, inplace=True)
    df_tidal_lagoon_locations.rename(columns={'Lat': 'lat', 'Long': 'lon', 'Suggested Bus': 'bus'}, inplace=True)
    if year < 2025:
        year = 2025
    date = str(year) + '-01-01'
    dic = {'capacities': df_tidal_lagoon_capacities.loc[date, :], 'locations': df_tidal_lagoon_locations}

    return dic


def read_tidal_stream(year, scenario):

    df_tidal_stream = pd.read_excel('../data/renewables/Marine/tidal_stream_future_deployment_scenarios.xlsx', sheet_name=None)
    # print(df_tidal_stream)

    if scenario == 'Leading The Way':
        sheet_name = 'tidal_stream_LW'
    elif scenario == 'Consumer Transformation':
        sheet_name = 'tidal_stream_CT'
    elif scenario == 'System Transformation':
        sheet_name = 'tidal_stream_ST'
    elif scenario == 'Steady Progression':
        sheet_name = 'tidal_stream_SP'

    df_tidal_stream_capacities = df_tidal_stream[sheet_name].T
    df_tidal_stream_capacities.columns = df_tidal_stream_capacities.iloc[1]
    df_tidal_stream_capacities = df_tidal_stream_capacities.drop(['Lat', 'Long', 'Suggested Bus', 'Site ID', 'Site Name'])
    df_tidal_stream_capacities = df_tidal_stream_capacities.iloc[:, :-1]

    # need to get values for years 2020 - 2050
    index_ = ['2025-01-01', '2030-01-01', '2035-01-01', '2040-01-01', '2045-01-01', '2050-01-01']
    df_tidal_stream_capacities.index = index_
    df_tidal_stream_capacities.index = pd.to_datetime(df_tidal_stream_capacities.index, infer_datetime_format=True, utc=True)
    df_tidal_stream_capacities = df_tidal_stream_capacities.resample('12MS').asfreq()
    df_tidal_stream_capacities = df_tidal_stream_capacities.astype(float)
    df_tidal_stream_capacities = df_tidal_stream_capacities.interpolate(method='linear', limit_direction='forward')
    # replace NaN with zero
    df_tidal_stream_capacities = df_tidal_stream_capacities.fillna(0)

    df_tidal_stream_locations = df_tidal_stream[sheet_name].iloc[:, -3:]
    df_tidal_stream_locations.index = df_tidal_stream[sheet_name]['Site ID']
    # drop the last row as this is a total row
    df_tidal_stream_locations.drop(df_tidal_stream_locations.tail(1).index, inplace=True)
    df_tidal_stream_locations.rename(columns={'Lat': 'lat', 'Long': 'lon', 'Suggested Bus': 'bus'}, inplace=True)
    if year < 2025:
        year = 2025
    date = str(year) + '-01-01'
    dic = {'capacities': df_tidal_stream_capacities.loc[date, :], 'locations': df_tidal_stream_locations}

    return dic


def read_wave_power(year, scenario):

    df_wave_power = pd.read_excel('../data/renewables/Marine/wave_power_future_deployment_scenarios.xlsx', sheet_name=None)
    # print(df_wave_power)

    if scenario == 'Leading The Way':
        sheet_name = 'wave_power_LW'
    elif scenario == 'Consumer Transformation':
        sheet_name = 'wave_power_CT'
    elif scenario == 'System Transformation':
        sheet_name = 'wave_power_ST'
    elif scenario == 'Steady Progression':
        sheet_name = 'wave_power_SP'

    df_wave_power_capacities = df_wave_power[sheet_name].T
    df_wave_power_capacities.columns = df_wave_power_capacities.iloc[1]
    df_wave_power_capacities = df_wave_power_capacities.drop(['Lat', 'Long', 'Suggested Bus', 'Site ID', 'Site Name'])
    df_wave_power_capacities = df_wave_power_capacities.iloc[:, :-1]

    # need to get values for years 2020 - 2050
    index_ = ['2025-01-01', '2030-01-01', '2035-01-01', '2040-01-01', '2045-01-01', '2050-01-01']
    df_wave_power_capacities.index = index_
    df_wave_power_capacities.index = pd.to_datetime(df_wave_power_capacities.index, infer_datetime_format=True, utc=True)
    df_wave_power_capacities = df_wave_power_capacities.resample('12MS').asfreq()
    df_wave_power_capacities = df_wave_power_capacities.astype(float)
    df_wave_power_capacities = df_wave_power_capacities.interpolate(method='linear', limit_direction='forward')
    # replace NaN with zero
    df_wave_power_capacities = df_wave_power_capacities.fillna(0)

    df_wave_power_locations = df_wave_power[sheet_name].iloc[:, -3:]
    df_wave_power_locations.index = df_wave_power[sheet_name]['Site ID']
    # drop the last row as this is a total row
    df_wave_power_locations.drop(df_wave_power_locations.tail(1).index, inplace=True)
    df_wave_power_locations.rename(columns={'Lat': 'lat', 'Long': 'lon', 'Suggested Bus': 'bus'}, inplace=True)
    if year < 2025:
        year = 2025
    date = str(year) + '-01-01'
    dic = {'capacities': df_wave_power_capacities.loc[date, :], 'locations': df_wave_power_locations}

    return dic


def write_marine_generators(year, scenario):
    # ADD NEW GENERATORS FOR MARINE

    # get generators
    path = 'LOPF_data/generators.csv'
    df_LOPF = pd.read_csv(path, index_col=0)
    path_UC = 'UC_data/generators.csv'
    df_UC = pd.read_csv(path_UC, index_col=0)

    # read marine generators
    read_tidal_lagoon_ = read_tidal_lagoon(year, scenario)

    df_tidal_lagoon = pd.DataFrame(read_tidal_lagoon_['capacities'])
    df_tidal_lagoon.columns = ['p_nom']
    # GW to MW
    df_tidal_lagoon.loc[:, 'p_nom'] *= 1000
    df_tidal_lagoon.index.name = 'name'
    df_tidal_lagoon['carrier'] = 'Marine'
    df_tidal_lagoon['type'] = 'Tidal lagoon'
    df_tidal_lagoon['bus'] = read_tidal_lagoon_['locations']['bus']
    df_tidal_lagoon['marginal_cost'] = 0.0
    df_tidal_lagoon['ramp_limit_up'] = 1.0
    df_tidal_lagoon['ramp_limit_down'] = 1.0
    df_tidal_lagoon['p_max_pu'] = 1.0

    read_tidal_stream_ = read_tidal_stream(year, scenario)

    df_tidal_stream = pd.DataFrame(read_tidal_stream_['capacities'])
    df_tidal_stream.columns = ['p_nom']
    df_tidal_stream.loc[:, 'p_nom'] *= 1000
    df_tidal_stream.index.name = 'name'
    df_tidal_stream['carrier'] = 'Marine'
    df_tidal_stream['type'] = 'Tidal stream'
    df_tidal_stream['bus'] = read_tidal_stream_['locations']['bus']
    df_tidal_stream['marginal_cost'] = 0.0
    df_tidal_stream['ramp_limit_up'] = 1.0
    df_tidal_stream['ramp_limit_down'] = 1.0
    df_tidal_stream['p_max_pu'] = 1.0

    read_wave_power_ = read_wave_power(year, scenario)

    df_wave_power = pd.DataFrame(read_wave_power_['capacities'])
    df_wave_power.columns = ['p_nom']
    df_wave_power.loc[:, 'p_nom'] *= 1000
    df_wave_power.index.name = 'name'
    df_wave_power['carrier'] = 'Marine'
    df_wave_power['type'] = 'Wave power'
    df_wave_power['bus'] = read_wave_power_['locations']['bus']
    df_wave_power['marginal_cost'] = 0.0
    df_wave_power['ramp_limit_up'] = 1.0
    df_wave_power['ramp_limit_down'] = 1.0
    df_wave_power['p_max_pu'] = 1.0

    # in shape to add to LOPF generators
    df_LOPF = df_LOPF.append([df_tidal_lagoon, df_tidal_stream, df_wave_power])
    df_LOPF.to_csv('LOPF_data/generators.csv', header=True)

    # additional params for UC problem
    df_tidal_lagoon['committable'] = False
    df_tidal_lagoon['min_up_time'] = 0
    df_tidal_lagoon['min_down_time'] = 0
    df_tidal_lagoon['p_min_pu'] = 0
    df_tidal_lagoon['up_time_before'] = 0
    df_tidal_lagoon['start_up_cost'] = 0

    df_tidal_stream['committable'] = False
    df_tidal_stream['min_up_time'] = 0
    df_tidal_stream['min_down_time'] = 0
    df_tidal_stream['p_min_pu'] = 0
    df_tidal_stream['up_time_before'] = 0
    df_tidal_stream['start_up_cost'] = 0

    df_wave_power['committable'] = False
    df_wave_power['min_up_time'] = 0
    df_wave_power['min_down_time'] = 0
    df_wave_power['p_min_pu'] = 0
    df_wave_power['up_time_before'] = 0
    df_wave_power['start_up_cost'] = 0

    # in shape to add to UC generators
    df_UC = df_UC.append([df_tidal_lagoon, df_tidal_stream, df_wave_power])
    df_UC.bus = 'bus'
    df_UC.to_csv('UC_data/generators.csv', header=True)


def add_marine_timeseries(year, year_baseline, scenario, time_step):

    path = 'LOPF_data/generators-p_max_pu.csv'
    df_LOPF = pd.read_csv(path, index_col=0)

    # in 5 year increments
    if year < 2030:
        year = 2025
    elif year >= 2030 and year < 2035:
        year = 2030
    elif year >= 2035 and year < 2040:
        year = 2035
    elif year >= 2040 and year < 2045:
        year = 2040
    elif year >= 2045 and year < 2050:
        year = 2045
    elif year >= 2050:
        year = 2050

    # for interpolating
    if time_step == 0.5:
        freq = '0.5H'
    elif time_step == 1:
        freq = 'H'

    # TIDAL LAGOON

    df_tidal_lagoon = pd.read_excel('../data/renewables/Marine/tidal_lagoon_full.xlsx', sheet_name=str(year))
    df_tidal_lagoon.index = df_tidal_lagoon['Date/time']
    df_tidal_lagoon.drop(['Date/time'], axis=1, inplace=True)
    df_tidal_lagoon.index = pd.to_datetime(df_tidal_lagoon.index, infer_datetime_format=True)
    df_tidal_lagoon.index = df_tidal_lagoon.index.round('H')
    df_tidal_lagoon.drop(df_tidal_lagoon.tail(1).index, inplace=True)
    df_tidal_lagoon.dropna(axis='columns', inplace=True)

    # FUTURE WORK, NEED TO INCLUDE NEGATIVE VALUES AS A LOAD, BUT DROPPING FOR NOW

    # fix inconsistent name
    df_tidal_lagoon.rename(columns={'Colwyn Bay': 'Colwyn'}, inplace=True)
    # interpolate to correct timestep
    df_tidal_lagoon = df_tidal_lagoon.resample(freq).interpolate('polynomial', order=2)

    if len(df_tidal_lagoon.index) < len(df_LOPF.index):

        # add end value
        end = df_LOPF.index.values[-1]
        df_new_tidal_lagoon = pd.DataFrame(
            data=df_tidal_lagoon.tail(1).values,
            columns=df_tidal_lagoon.columns,
            index=[end])
        # add to existing dataframe
        df_tidal_lagoon = df_tidal_lagoon.append(df_new_tidal_lagoon, sort=False)

    df_tidal_lagoon[df_tidal_lagoon < 0] = 0
    df_tidal_lagoon[df_tidal_lagoon > 1] = 1

    df_tidal_lagoon.index = pd.to_datetime(df_tidal_lagoon.index, infer_datetime_format=True)
    df_LOPF.index = pd.to_datetime(df_LOPF.index, infer_datetime_format=True)

    # pick out required timeseries
    df_tidal_lagoon = df_tidal_lagoon.loc[df_LOPF.index.values]
    df_tidal_lagoon.index = df_LOPF.index

    # TIDAL STREAM

    path = '../data/renewables/Marine/tidal_stream_' + str(year) + '_full.xlsx'
    df_tidal_stream = pd.read_excel(path)
    df_tidal_stream.index = df_tidal_stream['Date/time']
    df_tidal_stream.drop(['Date/time'], axis=1, inplace=True)
    df_tidal_stream.index = pd.to_datetime(df_tidal_stream.index, infer_datetime_format=True)
    df_tidal_stream.index = df_tidal_stream.index.round('H')
    df_tidal_stream.drop(df_tidal_stream.tail(1).index, inplace=True)
    # interpolate to correct timestep
    df_tidal_stream = df_tidal_stream.resample(freq).interpolate('polynomial', order=2)

    if len(df_tidal_stream.index) < len(df_LOPF.index):

        # add end value
        end = df_LOPF.index.values[-1]
        df_new_tidal_stream = pd.DataFrame(
            data=df_tidal_stream.tail(1).values,
            columns=df_tidal_stream.columns,
            index=[end])
        # add to existing dataframe
        df_tidal_stream = df_tidal_stream.append(df_new_tidal_stream, sort=False)

    df_tidal_stream[df_tidal_stream < 0] = 0
    df_tidal_stream[df_tidal_stream > 1] = 1

    df_tidal_stream.index = pd.to_datetime(df_tidal_stream.index, infer_datetime_format=True)
    df_LOPF.index = pd.to_datetime(df_LOPF.index, infer_datetime_format=True)

    # pick out required timeseries
    df_tidal_stream = df_tidal_stream.loc[df_LOPF.index.values]
    df_tidal_stream.index = df_LOPF.index

    # WAVE POWER

    df_wave_power = pd.read_csv('../data/renewables/Marine/capacity_factors_wave_full - Open Source.csv', index_col=0)
    df_wave_power.index = pd.to_datetime(df_wave_power.index, infer_datetime_format=True)
    # df_wave_power.index = df_wave_power.index.round('H')
    # interpolate to correct timestep
    df_wave_power = df_wave_power.resample(freq).interpolate('linear').round(5)

    if len(df_wave_power.index) < len(df_LOPF.index):

        # add end value
        end = df_LOPF.index.values[-1]
        df_new_wave_power = pd.DataFrame(
            data=df_wave_power.tail(1).values,
            columns=df_wave_power.columns,
            index=[end])
        # add to existing dataframe
        df_wave_power = df_wave_power.append(df_new_wave_power, sort=False)

    df_wave_power[df_wave_power < 0] = 0
    df_wave_power[df_wave_power > 1] = 1

    df_wave_power.index = pd.to_datetime(df_wave_power.index, infer_datetime_format=True)
    df_LOPF.index = pd.to_datetime(df_LOPF.index, infer_datetime_format=True)

    # pick out required timeseries
    period = df_LOPF.index
    period = pd.to_datetime(period, infer_datetime_format=True)
    # change year to baseline year
    period = period.map(lambda t: t.replace(year=year_baseline))
    df_wave_power = df_wave_power.loc[period.values]
    df_wave_power.index = df_LOPF.index

    # concat the DFs together
    df_LOPF = pd.concat([df_LOPF, df_tidal_lagoon, df_tidal_stream, df_wave_power], axis=1)

    # want to ensure no duplicate names
    cols = pd.Series(df_LOPF.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    # rename the columns with the cols list.
    df_LOPF.columns = cols
    # make sure there are no missing values
    df_LOPF = df_LOPF.fillna(0)
    # make sure there are no negative values
    df_LOPF[df_LOPF < 0] = 0
    # fix the column names
    df_LOPF.columns = df_LOPF.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_LOPF.columns = df_LOPF.columns.astype(str).str.replace(u'\xa0', '')
    df_LOPF.columns = df_LOPF.columns.astype(str).str.replace('ì', 'i')
    df_LOPF.columns = df_LOPF.columns.str.strip()

    df_LOPF.to_csv('LOPF_data/generators-p_max_pu.csv', header=True)
    df_LOPF.to_csv('UC_data/generators-p_max_pu.csv', header=True)


def aggregate_renewable_generation(start, end, year, time_step):

    # want to aggregate renewable generation to speed up UC solving

    # read in the generator file
    df = pd.read_csv('UC_data/generators.csv')
    # print(df)

    freq = snapshots.write_snapshots(start, end, time_step)

    # PV
    df_PV = df.loc[df['carrier'] == 'Solar Photovoltaics'].reset_index(drop=True)
    # delete PV from original dataframe
    df = df[df.carrier != 'Solar Photovoltaics']
    # add in row of total PV power
    df = df.append({'name': 'PV',
                    'carrier': 'Solar Photovoltaics',
                    'type': 'Solar Photovoltaics',
                    'p_nom': df_PV['p_nom'].sum(),
                    'bus': 'bus',
                    'marginal_cost': df_PV['marginal_cost'].mean(),
                    'committable': df_PV['committable'][0],
                    'min_up_time': df_PV['min_up_time'][0],
                    'min_down_time': df_PV['min_down_time'][0],
                    'ramp_limit_up': df_PV['ramp_limit_up'][0],
                    'ramp_limit_down': df_PV['ramp_limit_down'][0],
                    'up_time_before': df_PV['up_time_before'][0],
                    'p_min_pu': df_PV['p_min_pu'][0],
                    'start_up_cost': df_PV['start_up_cost'][0]}, ignore_index=True)
    # read in the PV time series
    df_PV_series = renewables_ninja_data_analysis.PV_corrected_series(year)
    df_PV_series = df_PV_series.loc[start:end]

    df_PV_aggregated_norm = df_PV_series.sum(axis=1) / 1000. / df_PV['p_nom'].sum()
    df_PV_aggregated_norm = pd.DataFrame(df_PV_aggregated_norm, columns=['PV'])
    # resample to half hourly timesteps
    df_PV_aggregated_norm = df_PV_aggregated_norm.resample(freq).interpolate('polynomial', order=1)
    # need to add a row at end
    # the data being passed is the values of the last row
    # the tail function is used to get the last index value
    df_new_PV = pd.DataFrame(
        data=[df_PV_aggregated_norm.loc[df_PV_aggregated_norm.tail(1).index.values].values[0]],
        columns=df_PV_aggregated_norm.columns,
        index=[end])
    # add to existing dataframe
    df_PV_aggregated_norm = df_PV_aggregated_norm.append(df_new_PV, sort=False)
    df_PV_aggregated_norm.index.name = 'name'

    # ONSHORE WIND
    df_onshore = df.loc[df['carrier'] == 'Wind Onshore'].reset_index(drop=True)
    # delete PV from original dataframe
    df = df[df.carrier != 'Wind Onshore']
    # add in row of total PV power
    df = df.append({'name': 'wind_onshore',
                    'carrier': 'Wind Onshore',
                    'type': 'Wind Onshore',
                    'p_nom': df_onshore['p_nom'].sum(),
                    'bus': 'bus',
                    'marginal_cost': df_onshore['marginal_cost'].mean(),
                    'committable': df_onshore['committable'][0],
                    'min_up_time': df_onshore['min_up_time'][0],
                    'min_down_time': df_onshore['min_down_time'][0],
                    'ramp_limit_up': df_onshore['ramp_limit_up'][0],
                    'ramp_limit_down': df_onshore['ramp_limit_down'][0],
                    'up_time_before': df_onshore['up_time_before'][0],
                    'p_min_pu': df_onshore['p_min_pu'][0],
                    'start_up_cost': df_onshore['start_up_cost'][0]}, ignore_index=True)
    # read in the onshore wind time series
    df_onshore_series = renewables_ninja_data_analysis.wind_onshore_corrected_series(year)

    df_onshore_series = df_onshore_series.loc[start:end]

    df_onshore_aggregated_norm = df_onshore_series.sum(axis=1) / 1000. / df_onshore['p_nom'].sum()
    df_onshore_aggregated_norm = pd.DataFrame(df_onshore_aggregated_norm, columns=['wind_onshore'])

    # resample to half hourly timesteps
    df_onshore_aggregated_norm = df_onshore_aggregated_norm.resample(freq).interpolate('polynomial', order=1)
    # need to add a row at end
    # the data being passed is the values of the last row
    # the tail function is used to get the last index value
    df_new_offshore = pd.DataFrame(
        data=[df_onshore_aggregated_norm.loc[df_onshore_aggregated_norm.tail(1).index.values].values[0]],
        columns=df_onshore_aggregated_norm.columns,
        index=[end])
    # add to existing dataframe
    df_onshore_aggregated_norm = df_onshore_aggregated_norm.append(df_new_offshore, sort=False)
    df_onshore_aggregated_norm.index.name = 'name'

    # OFFSHORE WIND
    df_offshore = df.loc[df['carrier'] == 'Wind Offshore'].reset_index(drop=True)
    # delete PV from original dataframe
    df = df[df.carrier != 'Wind Offshore']
    # add in row of total PV power
    df = df.append({'name': 'wind_offshore',
                    'carrier': 'Wind Offshore',
                    'type': 'Wind Offshore',
                    'p_nom': df_offshore['p_nom'].sum(),
                    'bus': 'bus',
                    'marginal_cost': df_offshore['marginal_cost'].mean(),
                    'committable': df_offshore['committable'][0],
                    'min_up_time': df_offshore['min_up_time'][0],
                    'min_down_time': df_offshore['min_down_time'][0],
                    'ramp_limit_up': df_offshore['ramp_limit_up'][0],
                    'ramp_limit_down': df_offshore['ramp_limit_down'][0],
                    'up_time_before': df_offshore['up_time_before'][0],
                    'p_min_pu': df_offshore['p_min_pu'][0],
                    'start_up_cost': df_offshore['start_up_cost'][0]}, ignore_index=True)
    # print(df)
    # read in the onshore wind time series
    df_offshore_series = renewables_ninja_data_analysis.wind_offshore_corrected_series(year)
    df_offshore_series = df_offshore_series.loc[start:end]

    df_offshore_aggregated_norm = df_offshore_series.sum(axis=1) / 1000. / df_offshore['p_nom'].sum()
    df_offshore_aggregated_norm = pd.DataFrame(df_offshore_aggregated_norm, columns=['wind_offshore'])
    # resample to half hourly timesteps
    df_offshore_aggregated_norm = df_offshore_aggregated_norm.resample(freq).interpolate('polynomial', order=1)
    # need to add a row at end
    # the data being passed is the values of the last row
    # the tail function is used to get the last index value
    df_new_offshore = pd.DataFrame(
        data=[df_offshore_aggregated_norm.loc[df_offshore_aggregated_norm.tail(1).index.values].values[0]],
        columns=df_offshore_aggregated_norm.columns,
        index=[end])
    # add to existing dataframe
    df_offshore_aggregated_norm = df_offshore_aggregated_norm.append(df_new_offshore, sort=False)
    df_offshore_aggregated_norm.index.name = 'name'

    # HYDRO

    df_hydro_small = df.loc[df['carrier'] == 'Small Hydro'].reset_index(drop=True)
    df_hydro_large = df.loc[df['carrier'] == 'Large Hydro'].reset_index(drop=True)
    p_nom = df_hydro_small['p_nom'].sum() + df_hydro_large['p_nom'].sum()
    # delete PV from original dataframe
    df = df[df.carrier != 'Small Hydro']
    df = df[df.carrier != 'Large Hydro']
    # add in row of total PV power
    df = df.append({'name': 'hydro',
                    'carrier': 'Large Hydro',
                    'type': 'Large Hydro',
                    'p_nom': p_nom,
                    'bus': 'bus',
                    'marginal_cost': df_hydro_large['marginal_cost'].mean(),
                    'committable': df_hydro_large['committable'][0],
                    'min_up_time': df_hydro_large['min_up_time'][0],
                    'min_down_time': df_hydro_large['min_down_time'][0],
                    'ramp_limit_up': df_hydro_large['ramp_limit_up'][0],
                    'ramp_limit_down': df_hydro_large['ramp_limit_down'][0],
                    'up_time_before': df_hydro_large['up_time_before'][0],
                    'p_min_pu': df_hydro_large['p_min_pu'][0],
                    'start_up_cost': df_hydro_large['start_up_cost'][0]}, ignore_index=True)

    df_series = pd.read_csv('UC_data/generators-p_max_pu.csv')
    # print(df_series)
    # limited to using bonnington which has been there for years so should work,
    # but this could be improved
    df_hydro_aggregated_norm = df_series['Bonnington'].values[:len(df_offshore_aggregated_norm.index)]
    df_hydro_aggregated_norm = pd.DataFrame(
        df_hydro_aggregated_norm, columns=['hydro'], index=df_offshore_aggregated_norm.index)

    # concat the time series for the RES tech
    df_res = pd.concat(
        [df_offshore_aggregated_norm, df_onshore_aggregated_norm,
         df_PV_aggregated_norm, df_hydro_aggregated_norm], axis=1)
    # print(df_res)
    # df_res = df_res.loc[start:end]
    # print(df_res)

    # save the new generators and generators p max pu files
    df_res.to_csv('UC_data/generators-p_max_pu.csv', header=True)
    df.to_csv('UC_data/generators.csv', index=False, header=True)


def RES_correction_factors():
    """correction factor to match annual model and actual output for RES generation

    Returns
    -------
    dict
        correction factor for all tech and years
    """

    offshore_data = {'2010': 3060,
                     '2011': 5149,
                     '2012': 7603,
                     '2013': 11472,
                     '2014': 13405,
                     '2015': 17423,
                     '2016': 16406,
                     '2017': 20916,
                     '2018': 26525,
                     '2019': 31975,
                     '2020': 40681}

    onshore_data = {'2010': 7226,
                    '2011': 10814,
                    '2012': 12244,
                    '2013': 16925,
                    '2014': 18555,
                    '2015': 22852,
                    '2016': 20754,
                    '2017': 28725,
                    '2018': 30382,
                    '2019': 31820,
                    '2020': 34688}

    PV_data = {'2010': 40,
               '2011': 244,
               '2012': 1354,
               '2013': 2010,
               '2014': 4054,
               '2015': 7533,
               '2016': 10398,
               '2017': 11457,
               '2018': 12668,
               '2019': 12580,
               '2020': 13158}

    gen_data_dict = {'Wind_Offshore': offshore_data,
                     'Wind_Onshore': onshore_data,
                     'PV': PV_data}
    years = list(range(2010, 2020 + 1))
    tech = ['Wind_Offshore', 'Wind_Onshore', 'PV']
    factor_dict = {}
    # tech = ['PV']
    for y in years:
        factor_dict_year = {}
        for t in tech:
            if y == 2010 and t == 'PV':
                factor_dict_year[t] = np.nan
            else:
                gen_year = gen_data_dict[t][str(y)]
                path = '../data/renewables/atlite/outputs/' + t + '/' + t + '_' + str(y) + '.csv'
                t_ = t.replace("_", " ")
                if t == 'PV':
                    t_ = 'Solar Photovoltaics'
                df = fix_timeseries_res_for_year(path, y, t_, future=False)
                modelled_generation = df.sum().sum() / 1000
                factor = gen_year / modelled_generation
                factor_dict_year[t] = factor
        factor_dict[y] = factor_dict_year

    path = '../data/renewables/atlite/'
    file = 'RES_correction_factors.csv'
    df_factors = pd.DataFrame(factor_dict)
    df_factors.to_csv(path + file, header=True)

    return df_factors


def historical_RES_timeseries(year, tech, future=False):

    tech_ = tech.replace(" ", "_")
    if tech == 'Solar Photovoltaics':
        tech_ = 'PV'
    path = '../data/renewables/atlite/outputs/' + tech_ + '/' + tech_ + '_' + str(year) + '.csv'
    # this returns the atlite time series, but corrected by date of operation
    df = fix_timeseries_res_for_year(path, year, tech, future=future)
    # fix the column names
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df.columns = df.columns.astype(str).str.replace(u'\xa0', '')
    df.columns = df.columns.astype(str).str.replace('ì', 'i')
    df.columns = df.columns.str.strip()

    # # now want to normalise using capacities...
    # # read in the renewable generators
    df_gen = pd.read_csv('LOPF_data/generators.csv', index_col=0)
    df_res_tech = df_gen.loc[df_gen['carrier'] == tech]

    # need to remove future and pipeline wind offshore
    if tech == 'Wind Offshore' and future is True:
        # get index of future and pipeline names
        # then get pipeline timeseries
        path = '../data/renewables/atlite/outputs/Wind_Offshore/wind_offshore_pipeline/'
        file = 'wind_offshore_pipeline_' + str(2020) + '.csv'  # year dosent matter
        df_pipeline = pd.read_csv(path + file, index_col=0)
        # fix the column names
        df_pipeline.columns = df_pipeline.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        df_pipeline.columns = df_pipeline.columns.astype(str).str.replace(u'\xa0', '')
        df_pipeline.columns = df_pipeline.columns.astype(str).str.replace('ì', 'i')
        df_pipeline.columns = df_pipeline.columns.str.strip()

        # first get the timeseries for these areas
        path = '../data/renewables/atlite/outputs/Wind_Offshore/wind_offshore_future/'
        file = 'wind_offshore_future_' + str(2020) + '.csv'  # year dosent matter
        df_future = pd.read_csv(path + file, index_col=0)
        # fix the column names
        df_future.columns = df_future.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        df_future.columns = df_future.columns.astype(str).str.replace(u'\xa0', '')
        df_future.columns = df_future.columns.astype(str).str.replace('ì', 'i')
        df_future.columns = df_future.columns.str.strip()

        # remove them from df_res_tech
        df_res_tech.drop(df_pipeline.columns, inplace=True)
        df_res_tech.drop(df_future.columns, inplace=True)

    df_norm = df.copy()
    for gen in df_norm.columns:
        # print(gen)
        # print(df_norm.columns)
        p_max = df_norm.loc[:, gen].max()
        df_norm.loc[:, gen] /= p_max

    return_dict = {'timeseries': df, 'norm': df_norm}

    return return_dict


def future_RES_scale_p_nom(year, tech, scenario):

    future_capacities_dict = future_RES_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

    if tech == 'Hydro':
        t1 = 'Small Hydro'
        t2 = 'Large Hydro'

        # get generators dataframe with p_noms to be scaled
        path = 'LOPF_data/generators.csv'
        generators = pd.read_csv(path, index_col=0)
        # this deletes the hydrogen generators so need to keep these
        hydrogen = generators.loc[generators['carrier'] == 'Hydrogen']
        gen_tech1 = generators.loc[generators['carrier'] == t1]
        gen_tech2 = generators.loc[generators['carrier'] == t2]
        gen_tech = gen_tech1.append([gen_tech2, hydrogen])

        path_UC = 'UC_data/generators.csv'
        generators_UC = pd.read_csv(path_UC, index_col=0)
        # this deletes the hydrogen generators so need to keep these
        hydrogen_UC = generators_UC.loc[generators_UC['carrier'] == 'Hydrogen']
        gen_tech_UC1 = generators_UC.loc[generators_UC['carrier'] == t1]
        gen_tech_UC2 = generators_UC.loc[generators_UC['carrier'] == t2]
        gen_tech_UC = gen_tech_UC1.append([gen_tech_UC2, hydrogen_UC])

    elif tech == 'Wind Onshore' or tech == 'Solar Photovoltaics':

        # get generators dataframe with p_noms to be scaled
        path = 'LOPF_data/generators.csv'
        generators = pd.read_csv(path, index_col=0)
        gen_tech = generators.loc[generators['carrier'] == tech]

        path_UC = 'UC_data/generators.csv'
        generators_UC = pd.read_csv(path_UC, index_col=0)
        gen_tech_UC = generators_UC.loc[generators_UC['carrier'] == tech]

    # then consider what scaling factor is required
    scaling_factor = round(tech_cap_FES / tech_cap_year, 2)

    # scale the p_noms of the RES generators
    for g in gen_tech.index:
        gen_tech.loc[g, 'p_nom'] *= scaling_factor
        gen_tech_UC.loc[g, 'p_nom'] *= scaling_factor

    # write new generators.csv file
    # save the dataframes to csv
    # need to remove the original tech
    # print(generators.type.to_list())
    generators = generators[~generators.type.str.contains(tech)]
    generators_UC = generators_UC[~generators_UC.type.str.contains(tech)]
    # then add the new p_nom tech
    generators = generators.append(gen_tech)
    generators_UC = generators_UC.append(gen_tech_UC)

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_offshore_timeseries(year, year_baseline, scenario):

    future_capacities_dict = future_offshore_capacity(year, year_baseline, scenario)
    # print(future_capacities_dict)
    offshore_cap_year = future_capacities_dict['offshore_cap_year']
    offshore_cap_pipeline = future_capacities_dict['offshore_cap_pipeline']
    offshore_cap_scotland_planning = future_capacities_dict['offshore_cap_scotland_planning']
    offshore_cap_FES = future_capacities_dict['offshore_cap_FES']

    # first get timeseries for baseline year
    tech = 'Wind Offshore'
    df_baseline = historical_RES_timeseries(year_baseline, tech, future=True)['timeseries']

    # then get pipeline timeseries
    path = '../data/renewables/atlite/outputs/Wind_Offshore/wind_offshore_pipeline/'
    file = 'wind_offshore_pipeline_' + str(year_baseline) + '.csv'
    df_pipeline = pd.read_csv(path + file, index_col=0)
    df_pipeline.index = df_baseline.index

    # first get the timeseries for these areas
    path = '../data/renewables/atlite/outputs/Wind_Offshore/wind_offshore_future/'
    file = 'wind_offshore_future_' + str(year_baseline) + '.csv'
    df_future = pd.read_csv(path + file, index_col=0)
    df_future.index = df_baseline.index

    # check if baseline year is a leap year and simulated year is not and remove 29th Feb
    if year_baseline % 4 == 0:
        # and the year modelled is also not a leap year
        if year % 4 != 0:
            # remove 29th Feb
            df_baseline = df_baseline[~((df_baseline.index.month == 2) & (df_baseline.index.day == 29))]
            df_pipeline = df_pipeline[~((df_pipeline.index.month == 2) & (df_pipeline.index.day == 29))]
            df_future = df_future[~((df_future.index.month == 2) & (df_future.index.day == 29))]

    # now want to normalise using capacities...
    path = 'LOPF_data/generators.csv'
    generators = pd.read_csv(path, index_col=0)
    gen_tech = generators.loc[generators['carrier'] == 'Wind Offshore']

    # now want to normalise using capacities...
    path_UC = 'UC_data/generators.csv'
    generators_UC = pd.read_csv(path_UC, index_col=0)
    gen_tech_UC = generators_UC.loc[generators_UC['carrier'] == 'Wind Offshore']

    # combine the timeseries
    result = pd.concat([df_baseline, df_pipeline, df_future], axis=1)

    # clean up strings in these original dataframes
    df_baseline.columns = df_baseline.columns.astype(str).str.replace(u'\xa0', ' ')
    df_baseline = df_baseline.rename(columns=lambda x: x.strip())
    df_baseline.columns = df_baseline.columns.astype(str).str.replace('ì', 'i')
    df_pipeline.columns = df_pipeline.columns.astype(str).str.replace(u'\xa0', ' ')
    df_pipeline = df_pipeline.rename(columns=lambda x: x.strip())
    df_pipeline.columns = df_pipeline.columns.astype(str).str.replace('ì', 'i')
    df_future.columns = df_future.columns.astype(str).str.replace(u'\xa0', ' ')
    df_future = df_future.rename(columns=lambda x: x.strip())
    df_future.columns = df_future.columns.astype(str).str.replace('ì', 'i')

    # clean up the strings
    result.columns = result.columns.astype(str).str.replace(u'\xa0', ' ')
    result.columns = result.columns.astype(str).str.replace('ì', 'i')
    result = result.rename(columns=lambda x: x.strip())

    gen_tech.index = gen_tech.index.str.strip()
    gen_tech.index = gen_tech.index.astype(str).str.replace(u'\xa0', ' ')
    gen_tech_UC.index = gen_tech_UC.index.str.strip()
    gen_tech_UC.index = gen_tech_UC.index.astype(str).str.replace(u'\xa0', ' ')

    # then normalise all the different units
    result_norm = pd.DataFrame(result)
    # print(result_norm.columns)
    # print(gen_tech)
    for gen in result_norm.columns:
        p_max = result_norm.loc[:, gen].max()
        result_norm.loc[:, gen] /= p_max

    # change the year on the indexes to the year simulated
    result.index = result.index + pd.DateOffset(year=year)
    result_norm.index = result_norm.index + pd.DateOffset(year=year)

    # now want to scale the p_nom of offshore units
    # print(offshore_cap_FES, 'required')
    # print(offshore_cap_year, '2020??')
    # print(offshore_cap_pipeline, 'pipeline for year')
    # print(gen_tech.p_nom.sum(), 'sum to begin with')
    # then consider what capacity still needs to be built
    cap_req = round(offshore_cap_FES - offshore_cap_year, 2)
    # print(cap_req, 'capacity required over 2020 capacity')

    if cap_req <= offshore_cap_pipeline:
        # then pipeline available capacity should be scaled down
        # or by 1 if it is exactly right
        # scale by the total pipeline available as the dict value is the pipeline available in modelled year
        # but need to scaled total pipeline
        pipeline_factor = cap_req / 18.3059
    elif cap_req > offshore_cap_pipeline:
        # if still need more capacity just use the pipeline as is
        pipeline_factor = offshore_cap_pipeline / 18.3059

    # print(pipeline_factor, 'pipeline scaling factor')

    # scale the p_noms down for pipeline wind turbines
    for g in df_pipeline.columns:
        gen_tech.loc[g, 'p_nom'] *= pipeline_factor
        gen_tech_UC.loc[g, 'p_nom'] *= pipeline_factor

    # print(gen_tech.p_nom.sum(), 'sum after including pipeline')

    # then check if still need more capacity built
    cap_req = round(offshore_cap_FES - offshore_cap_year - offshore_cap_pipeline, 2)
    # print(cap_req, 'cap required after pipeline')
    # print(cap_req, 'capacity required over 2020 capacity + pipeline capacity')
    # next step is to use capacity from Marine Sector Plan for Scotland

    if cap_req <= 0:
        # if not capacity required then zero
        future_factor = 0
    elif cap_req <= offshore_cap_scotland_planning:
        # then future available capacity should be scaled down
        # or by 1 if it is exactly right
        future_factor = cap_req / offshore_cap_scotland_planning
    elif cap_req > offshore_cap_pipeline:
        # if still need more capacity just use the future timeseries as is
        future_factor = 1
    # print(future_factor, 'future planned scaling factor')

    # scale the future wind offshore units
    for g in df_future.columns:
        gen_tech.loc[g, 'p_nom'] *= future_factor
        gen_tech_UC.loc[g, 'p_nom'] *= future_factor

    # print(gen_tech.p_nom.sum(), 'sum after future sites included')

    # now check if 2020 + pipeline + future sites is enough
    cap_req = round(
        offshore_cap_FES - offshore_cap_year - offshore_cap_pipeline - offshore_cap_scotland_planning, 2)
    # print(cap_req, 'capacity required over 2020 capacity + pipeline capacity + planned capacity')
    if cap_req > 0:
        # final step is to linearly scale the combined timeseries to get to required capacity
        scaling_factor = round(
            offshore_cap_FES / (offshore_cap_year + offshore_cap_pipeline + offshore_cap_scotland_planning),
            2)
    else:
        scaling_factor = 1
    for g in result.columns:
        gen_tech.loc[g, 'p_nom'] *= scaling_factor
        gen_tech_UC.loc[g, 'p_nom'] *= scaling_factor

    # write new generators.csv file
    # save the dataframes to csv
    generators.loc[generators['carrier'] == 'Wind Offshore'] = gen_tech
    generators_UC.loc[generators_UC['carrier'] == 'Wind Offshore'] = gen_tech_UC

    # print(gen_tech.p_nom.sum(), 'final sum')

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)

    return_dict = {'timeseries': result, 'norm': result_norm}

    return return_dict


def future_offshore_sites(year):
    # how much wind in 2020
    # read in the renewable generators
    df_res = REPD_date_corrected(year)
    # start with the offshore wind farms
    df_res_offshore = df_res.loc[df_res['Technology Type'] == 'Wind Offshore'].reset_index(drop=True)
    # installed capacity 2020
    print(df_res_offshore['Installed Capacity (MWelec)'].sum() / 1000, 'GW installed 2020')

    # pipeline data
    df_pipeline = pd.read_csv('data/renewables/future_offshore_sites/offshore_pipeline.csv',
                              encoding='unicode_escape', index_col=2)
    df_pipeline.drop(columns=['Record Last Updated (dd/mm/yyyy)', 'Operator (or Applicant)',
                              'Under Construction', 'Technology Type',
                              'Planning Permission Expired', 'Operational',
                              'Heat Network Ref', 'Planning Authority',
                              'Planning Application Submitted', 'Region',
                              'Country', 'County'], inplace=True)
    df_pipeline.dropna(axis='columns', inplace=True)
    # capacity in pipeline
    # print(df_pipeline['Installed Capacity (MWelec)'].sum() / 1000, 'GW in pipeline')
    # This takes us as far as 2027 for the Leading the Way scenario

    # lets look at pipeline output for these years
    df = df_pipeline['Expected Operational']
    df = pd.to_datetime(df).dt.to_period('D')
    # cut off is the end of the year being simulated
    for y in [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]:
        date = '31/12/' + str(y)
        df2 = df_pipeline[~(df > date)]
        new_gw = df2['Installed Capacity (MWelec)'].sum() / 1000
        total_gw = new_gw + df_res_offshore['Installed Capacity (MWelec)'].sum() / 1000
        # print(new_gw, 'New GW in ' + str(y))
        # print(total_gw, 'Total GW in ' + str(y))

    # pipeline + operational takes us to 25.4GW in 2027
    # lets look at future fields and the potential capacity which exists there...

    # max 26GW in Scottish waters... takes us to 51.4GW...
    # this is exceeded in 2031 in leading the way scenario
    # highest offshore wind capacity in FES in 113.2GW
    # need to simply scale up distributions by assuming far offshore from 2030


def future_offshore_capacity(year, year_baseline, scenario):
    # how much wind in baseline year
    # read in the renewable generators
    df_res = REPD_date_corrected(year_baseline)
    # start with the offshore wind farms
    df_res_offshore = df_res.loc[df_res['Technology Type'] == 'Wind Offshore'].reset_index(drop=True)
    # installed capacity in baseline year
    offshore_cap_year = df_res_offshore['Installed Capacity (MWelec)'].sum() / 1000

    # pipeline data
    df_pipeline = pd.read_csv('../data/renewables/future_offshore_sites/offshore_pipeline.csv',
                              encoding='unicode_escape', index_col=2)
    df_pipeline.drop(columns=['Record Last Updated (dd/mm/yyyy)', 'Operator (or Applicant)',
                              'Under Construction', 'Technology Type',
                              'Planning Permission Expired', 'Operational',
                              'Heat Network Ref', 'Planning Authority',
                              'Planning Application Submitted', 'Region',
                              'Country', 'County'], inplace=True)
    df_pipeline.dropna(axis='columns', inplace=True)
    # pipeline up to 2030, but still add in pipeline after 2030
    if year > 2030:
        year_pipeline = 2030
    else:
        year_pipeline = year
    # lets look at pipeline output for these years
    df = df_pipeline['Expected Operational']
    df = pd.to_datetime(df).dt.to_period('D')

    date = '31/12/' + str(year_pipeline)
    df2 = df_pipeline[~(df > date)]
    offshore_cap_pipeline = df2['Installed Capacity (MWelec)'].sum() / 1000
    # print(offshore_cap_pipeline, 'New GW in ' + str(year_pipeline))

    df_scotland = pd.read_csv('../data/renewables/future_offshore_sites/Sectoral Marine Plan 2020.csv',
                              encoding='unicode_escape')
    offshore_cap_scotland_planning = df_scotland['max capacity (GW)'].sum()

    # offshore wind capacity from FES2021
    df_FES = pd.read_excel(
        '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
        sheet_name='SV.28', usecols="M:AS", header=7, dtype=str,
        index_col=1)
    df_FES.drop(columns=['Unnamed: 12'], inplace=True)
    df_FES.dropna(axis='rows', inplace=True)
    date = str(year) + '-01-01'
    if scenario == 'Leading The Way':
        scenario = 'Leading the Way'
    offshore_cap_FES = float(df_FES.loc[scenario, date])
    # print(type(offshore_cap_FES))

    capacity_dict = {'offshore_cap_year': offshore_cap_year,
                     'offshore_cap_pipeline': offshore_cap_pipeline,
                     'offshore_cap_scotland_planning': offshore_cap_scotland_planning,
                     'offshore_cap_FES': offshore_cap_FES}
    # print(capacity_dict)
    return capacity_dict


def future_RES_capacity(year, tech, scenario):

    if tech == 'Hydro':
        df_hydro = read_hydro(year)
        # drop pumped hydro
        df_hydro = df_hydro[~df_hydro.type.str.contains('Pumped Storage Hydroelectricity')]
        tech_cap_year = df_hydro['p_nom'].sum() / 1000

    elif tech == 'Wind Onshore' or tech == 'Solar Photovoltaics':
        # how much RES in year
        # read in the renewable generators
        df_res = REPD_date_corrected(year)
        # start with the offshore wind farms
        df_res_tech = df_res.loc[df_res['Technology Type'] == tech].reset_index(drop=True)
        # installed capacity year
        tech_cap_year = df_res_tech['Installed Capacity (MWelec)'].sum() / 1000

    elif tech == 'Biomass':
        df_biomass = read_biomass(year)
        tech_cap_year = df_biomass['p_nom'].sum() / 1000

    # how much RES in year to be simulated
    if tech == 'Wind Onshore':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='SV.29', usecols="M:AS", header=6, dtype=str,
            index_col=1)
        df_FES.drop(columns=['Unnamed: 12'], inplace=True)
        df_FES.dropna(axis='rows', inplace=True)

    elif tech == 'Solar Photovoltaics':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='SV.31', usecols="L:AR", header=5, dtype=str,
            index_col=1)
        df_FES.drop(columns=['Unnamed: 11'], inplace=True)
        df_FES.dropna(axis='rows', inplace=True)

    elif tech == 'Hydro':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        df_FES.dropna(axis='rows', inplace=True)
        df_FES = df_FES[df_FES.Type.str.contains('Hydro', case=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        # df_FES = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        cols = [2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)

        df_FES_LTW = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        df_FES_LTW = df_FES_LTW.append(df_FES_LTW.sum(numeric_only=True), ignore_index=True)
        df_FES_LTW.dropna(axis='columns', inplace=True)
        df_FES_LTW.drop([0, 1], inplace=True)
        df_FES_LTW.index = ['Leading the Way']

        df_FES_CT = df_FES[df_FES.index.str.contains('Consumer Transformation', case=False)]
        df_FES_CT = df_FES_CT.append(df_FES_CT.sum(numeric_only=True), ignore_index=True)
        df_FES_CT.dropna(axis='columns', inplace=True)
        df_FES_CT.drop([0, 1], inplace=True)
        df_FES_CT.index = ['Consumer Transformation']

        df_FES_ST = df_FES[df_FES.index.str.contains('System Transformation', case=False)]
        df_FES_ST = df_FES_ST.append(df_FES_ST.sum(numeric_only=True), ignore_index=True)
        df_FES_ST.dropna(axis='columns', inplace=True)
        df_FES_ST.drop([0, 1], inplace=True)
        df_FES_ST.index = ['System Transformation']

        df_FES_SP = df_FES[df_FES.index.str.contains('Steady Progression', case=False)]
        df_FES_SP = df_FES_SP.append(df_FES_SP.sum(numeric_only=True), ignore_index=True)
        df_FES_SP.dropna(axis='columns', inplace=True)
        df_FES_SP.drop([0, 1], inplace=True)
        df_FES_SP.index = ['Steady Progression']

        df_FES = df_FES_SP.append([df_FES_LTW, df_FES_CT, df_FES_ST])

    elif tech == 'Biomass':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        df_FES = df_FES[df_FES.Type.str.contains('Biomass', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)

        df_FES_LTW = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        df_FES_LTW = df_FES_LTW.append(df_FES_LTW.sum(numeric_only=True), ignore_index=True)
        df_FES_LTW.drop([0, 1, 2], inplace=True)
        df_FES_LTW.dropna(axis='columns', inplace=True)
        df_FES_LTW.index = ['Leading the Way']

        df_FES_CT = df_FES[df_FES.index.str.contains('Consumer Transformation', case=False)]
        df_FES_CT = df_FES_CT.append(df_FES_CT.sum(numeric_only=True), ignore_index=True)
        df_FES_CT.drop([0, 1, 2], inplace=True)
        df_FES_CT.dropna(axis='columns', inplace=True)
        df_FES_CT.index = ['Consumer Transformation']

        df_FES_ST = df_FES[df_FES.index.str.contains('System Transformation', case=False)]
        df_FES_ST = df_FES_ST.append(df_FES_ST.sum(numeric_only=True), ignore_index=True)
        df_FES_ST.drop([0, 1, 2], inplace=True)
        df_FES_ST.dropna(axis='columns', inplace=True)
        df_FES_ST.index = ['System Transformation']

        df_FES_SP = df_FES[df_FES.index.str.contains('Steady Progression', case=False)]
        df_FES_SP = df_FES_SP.append(df_FES_SP.sum(numeric_only=True), ignore_index=True)
        df_FES_SP.drop([0, 1, 2], inplace=True)
        df_FES_SP.dropna(axis='columns', inplace=True)
        df_FES_SP.index = ['Steady Progression']

        df_FES = df_FES_SP.append([df_FES_LTW, df_FES_CT, df_FES_ST])

    date = str(year) + '-01-01'
    if scenario == 'Leading The Way':
        scenario = 'Leading the Way'

    if tech == 'Wind Onshore' or tech == 'Solar Photovoltaics':
        tech_cap_FES = float(df_FES.loc[scenario, date])
    else:
        tech_cap_FES = float(df_FES.loc[scenario, date]) / 1000.

    capacity_dict = {'tech_cap_year': tech_cap_year,
                     'tech_cap_FES': tech_cap_FES}

    return capacity_dict


def plot_future_capacities(year):

    start = str(year) + '-12-02 00:00:00'
    end = str(year) + '-12-02 03:30:00'
    # time step as fraction of hour
    time_step = 0.5
    if year > 2020:
        data_reader_writer.data_writer(start, end, time_step, year, year_baseline=2020, scenario='Leading The Way')
    if year <= 2020:
        data_reader_writer.data_writer(start, end, time_step, year)

    df_generators = pd.read_csv('LOPF_data/generators.csv', index_col=0)
    generators_p_nom = df_generators.p_nom.groupby(df_generators.carrier).sum()
    generators_p_nom.drop('Unmet Load', inplace=True)
    try:
        generators_p_nom.drop('Coal', inplace=True)
    except:
        pass
    # generators_p_nom.drop(generators_p_nom[generators_p_nom < 500].index, inplace=True)

    # bar chart
    plt.figure(figsize=(15, 10))
    col_map = plt.get_cmap('Paired')
    plt.bar(generators_p_nom.index, generators_p_nom.values / 1000, color=col_map.colors, edgecolor='k')
    plt.xticks(generators_p_nom.index, rotation=90)
    plt.ylabel('GW')
    plt.grid(color='grey', linewidth=1, axis='both', alpha=0.5)
    plt.title('Installed capacity in year ' + str(year))
    plt.tight_layout()
    plt.savefig('../data/FES2021/Capacities Pics/' + str(year) + '.png')


def gif_future_capacities():

    filenames = []
    for year in range(2021, 2050 + 1):
        plot_future_capacities(year)
        # list of filenames
        filenames.append('../data/FES2021/Capacities Pics/' + str(year) + '.png')

    with imageio.get_writer('../data/FES2021/Capacities Pics/FES_installed_capacities.gif', mode='I', duration=1.) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":

    # year = 2020
    # future_offshore_sites(year)

    # year = 2027
    # future_offshore_capacity(year)

    year = 2017
    tech = 'Solar Photovoltaics'
    historical_RES_timeseries(year, tech)['norm']
    historical_RES_timeseries(year, tech)['timeseries']

    # RES_correction_factors()

    # year = 2025
    # future_offshore_capacity(year)

    # year = 2050
    # future_offshore_timeseries(year)
    # output = future_offshore_timeseries(year)
    # print(output['timeseries'])
    # print(output['norm'])

    # year = 2050
    # tech = 'Wind Onshore'
    # tech = 'Hydro'
    # tech = 'Solar Photovoltaics'

    # future_RES_capacity(year, tech)
    # future_RES_scale_p_nom(year, tech)

    # gif_future_capacities()

    # scenario = 'Consumer Transformation'
    # scenario = 'Leading The Way'
    # scenario = 'System Transformation'
    # scenario = 'Steady Progression'
    # year = 2050
    # print(read_tidal_lagoon(year, scenario))
    # print(read_tidal_stream(year, scenario))
    # print(read_wave_power(year, scenario))

    # write_marine_generators(year, scenario)
    # year_baseline = 2012
    # add_marine_timeseries(year, year_baseline, scenario, time_step=0.5)
