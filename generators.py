import pandas as pd
import numpy as np
import os

import osgb

import renewables
import distance_calculator as dc
from utils.cleaning import unify_index
from utils.cleaning import remove_double


def read_power_stations_data(year):
    """reads the power station data from prepared csv file

    Parameters
    ----------
    year : int/str
        year of simulation

    Returns
    -------
    dataframe
        power station data
    """
    # short term fix to run data_reader_writer.py
    if year > 2020:
        year = 2020

    # want to read in the conventional power generators from
    # the data prepared from DUKES
    file = '../data/power stations/power_stations_locations_' + str(year) + '.csv'
    # read the csv
    df = pd.read_csv(file, encoding='unicode_escape')
    # fix the formatting to have each element be a list of
    # latitude [0] and longitude [1]
    df['Geolocation'] = df['Geolocation'].str.replace(',', '')
    df['Geolocation'] = df['Geolocation'].str.split()

    return df


def read_generator_data_by_fuel():
    """reads the generator data by fuel from prepared csv file

    Parameters
    ----------

    Returns
    -------
    dataframe
        generator data by fuel
    """

    file = '../data/generator_data_by_fuel.csv'
    df = pd.read_csv(file)
    df = df.set_index('fuel')
    return df


def write_generators(time_step, year):
    """writes the generators csv file

    Parameters
    ----------
    year : int/str
        year of simulation
    time_step : float
        defined as fraction of an hour, e.g., 0.5 is half hour
        currently set up as only hour or half hour
    Returns
    -------
    """

    # GENERATOR CSV FILE
    # read in conventional power plants
    df_pp = read_power_stations_data(year)

    df_pp = df_pp.drop(
        columns=['Company Name', 'Year of commission or year generation began',
                 'Location'])
    df_pp = df_pp.rename(columns={'Installed Capacity (MW)': 'p_nom', 'Fuel': 'carrier',
                                  'Technology': 'type', 'Station Name': 'name'})

    # only do this if there is comma in power
    # pass this step otherwise
    try:
        df_pp['p_nom'] = df_pp['p_nom'].str.replace(',', '')
    except:
        pass

    df_pp_UC = df_pp.drop(
        columns=['x', 'y', 'Geolocation'])
    df_pp_UC['bus'] = 'bus'

    df_pp_LOPF = df_pp.drop(
        columns=['x', 'y', 'Geolocation'])
    df_pp_LOPF['bus'] = dc.map_to_bus(df_pp)

    # corrections factors for RES generators: Onshore, Offshore, PV
    df_correction = pd.read_csv('../data/renewables/atlite/RES_correction_factors.csv', index_col=0)

    # read in the renewable generators
    df_res = renewables.REPD_date_corrected(year)
    # start with the offshore wind farms
    df_res_offshore = df_res.loc[df_res['Technology Type'] == 'Wind Offshore'].reset_index(drop=True)
    df_res_offshore = df_res_offshore.drop(
        columns=['CHP Enabled', 'No. of Turbines', 'Development Status',
                 'X-coordinate', 'Y-coordinate', 'Operational',
                 'Height of Turbines (m)', 'Mounting Type for Solar',
                 'Turbine Capacity (MW)'])
    df_res_offshore = df_res_offshore.rename(columns={'Installed Capacity (MWelec)': 'p_nom',
                                                      'Technology Type': 'carrier',
                                                      'Site Name': 'name',
                                                      'lon': 'x',
                                                      'lat': 'y'})
    df_res_offshore['type'] = 'Wind Offshore'

    # scale the wind offshore to real data, see correction factors in Atlite
    if year <= 2020:
        df_res_offshore.loc[:, 'p_nom'] *= df_correction.loc['Wind_Offshore', str(year)]

    # add in future sites for future scenarios
    if year > 2020:
        path = '../data/renewables/future_offshore_sites/'
        file1 = 'offshore_pipeline.csv'
        df_pipeline = pd.read_csv(path + file1, encoding='unicode_escape')
        df_pipeline.drop(columns=['Record Last Updated (dd/mm/yyyy)', 'Operator (or Applicant)',
                                  'Under Construction', 'Technology Type',
                                  'Planning Permission Expired', 'Operational',
                                  'Heat Network Ref', 'Planning Authority',
                                  'Planning Application Submitted', 'Region',
                                  'Country', 'County', 'Expected Operational',
                                  'Turbine Capacity (MW)', 'No. of Turbines',
                                  'Development Status', 'Development Status (short)'], inplace=True)
        df_pipeline.dropna(axis='columns', inplace=True)
        df_pipeline['carrier'] = 'Wind Offshore'

        # create two lists of conversions from OSGB to lat/lon
        lon = []
        lat = []
        for i in range(len(df_pipeline.index)):
            x = df_pipeline['X-coordinate'][i]
            y = df_pipeline['Y-coordinate'][i]
            coord = osgb.grid_to_ll(x, y)
            lat.append(coord[0])
            lon.append(coord[1])
        df_pipeline['lon'] = lon
        df_pipeline['lat'] = lat
        df_pipeline['type'] = df_pipeline['carrier']
        df_pipeline.rename(columns={'Installed Capacity (MWelec)': 'p_nom',
                                    'Site Name': 'name',
                                    'lon': 'x',
                                    'lat': 'y'}, inplace=True)
        df_pipeline.drop(columns=['X-coordinate', 'Y-coordinate'], inplace=True)

        file2 = 'Sectoral Marine Plan 2020 - Fixed.csv'
        df_future_FBOW = pd.read_csv(path + file2, encoding='unicode_escape')
        df_future_FBOW['carrier'] = 'Wind Offshore'
        df_future_FBOW['type'] = 'Wind Offshore'
        df_future_FBOW.drop(columns=['area (km2)'], inplace=True)
        df_future_FBOW.rename(columns={'max capacity (GW)': 'p_nom',
                                  'lon': 'x',
                                  'lat': 'y'}, inplace=True)
        
        file3 = 'Sectoral Marine Plan 2020 - Floating.csv'
        df_future_FOW = pd.read_csv(path + file3, encoding='unicode_escape')
        df_future_FOW['carrier'] = 'Wind Offshore'
        df_future_FOW['type'] = 'Floating Wind'
        df_future_FOW.drop(columns=['area (km2)'], inplace=True)
        df_future_FOW.rename(columns={'max capacity (GW)': 'p_nom',
                                  'lon': 'x',
                                  'lat': 'y'}, inplace=True)
        
#         df_future = df_future.append([df_future_FBOW, df_future_FOW], ignore_index=True)
        
        # convert from GW to MW
        df_future_FBOW.loc[:, 'p_nom'] *= 1000
        df_future_FOW.loc[:, 'p_nom'] *= 1000

        df_res_offshore = df_res_offshore.append([df_pipeline, df_future_FBOW, df_future_FOW], ignore_index=True)

    # amend type of offshore wind farms
          # find index of values N2, N3, NE1, NE2, NE3, NE6, NE7, NE8, E1, E2, E3
    
#     df_res_offshore[df_res_offshore['name'] == 'N2'].index.values
    
#     df_res_offshore.name[df_res_offshore.name == 'N2'].index
    
#     index_N2 = df_res_offshore.loc[df_res_offshore['name'] == 'N2']
    
#     df_res_offshore.loc[index_N2,'type']='Floating Wind'
    
    
    
    
    
#     findL = ['N2', 'Mexico', 'United States', 'hob']
#     replaceL = ['Kangaroo', 'Spider Monkey', 'Eagle', 'Test']

#     # Select column in which to execute replacement (can be A,B,C,D)
#     col = 'D';

#     # Find and replace values in the selected column
#     df[col] = df[col].replace(findL, replaceL)
#     print(df)   
        
        
        
        
    df_res_offshore_UC = df_res_offshore.drop(
        columns=['x', 'y'])
    df_res_offshore_UC['bus'] = 'bus'

    df_res_offshore_LOPF = df_res_offshore.drop(
        columns=['x', 'y'])
    df_res_offshore_LOPF['bus'] = dc.map_to_bus(df_res_offshore)
    
    
    
    
    
    
    

    # join to previous df of thermal power plants
    df_UC = df_pp_UC.append(df_res_offshore_UC, ignore_index=True, sort=False)
    df_LOPF = df_pp_LOPF.append(df_res_offshore_LOPF, ignore_index=True, sort=False)

    # check names are unique for UC
    duplicateDFRow = df_UC[df_UC.duplicated(['name'], keep='first')]
    for i in range(len(duplicateDFRow.index.values)):
        # print(df_UC['name'][duplicateDFRow.index.values[i]])
        df_UC.at[duplicateDFRow.index.values[i], 'name'] = (
            df_UC['name'][duplicateDFRow.index.values[i]] + '.1')
        # print(df_UC['name'][duplicateDFRow.index.values[i]])

    # check names are unique for LOPF
    duplicateDFRow = df_pp_LOPF[df_pp_LOPF.duplicated(['name'], keep='first')]
    for i in range(len(duplicateDFRow.index.values)):
        # print(df_pp_LOPF['name'][duplicateDFRow.index.values[i]])
        df_pp_LOPF.at[duplicateDFRow.index.values[i], 'name'] = (
            df_pp_LOPF['name'][duplicateDFRow.index.values[i]] + '.1')
        # print(df_pp_LOPF['name'][duplicateDFRow.index.values[i]])

    # then the onshore wind farms
    df_res_onshore = df_res.loc[df_res['Technology Type'] == 'Wind Onshore'].reset_index(drop=True)
    df_res_onshore = df_res_onshore.drop(
        columns=['CHP Enabled', 'No. of Turbines', 'Development Status',
                 'X-coordinate', 'Y-coordinate', 'Operational',
                 'Height of Turbines (m)', 'Mounting Type for Solar',
                 'Turbine Capacity (MW)'])
    df_res_onshore = df_res_onshore.rename(columns={'Installed Capacity (MWelec)': 'p_nom',
                                                    'Technology Type': 'carrier',
                                                    'Site Name': 'name',
                                                    'lon': 'x',
                                                    'lat': 'y'})
    df_res_onshore['type'] = 'Wind Onshore'

    # scale the wind onshore to real data, see correction factors in Atlite
    if year <= 2020:
        df_res_onshore.loc[:, 'p_nom'] *= df_correction.loc['Wind_Onshore', str(year)]

    df_res_onshore_UC = df_res_onshore.drop(
        columns=['x', 'y'])
    df_res_onshore_UC['bus'] = 'bus'

    df_res_onshore_LOPF = df_res_onshore.drop(
        columns=['x', 'y'])
    df_res_onshore_LOPF['bus'] = dc.map_to_bus(df_res_onshore)

    # join to previous df of thermal power plants
    df_UC = df_UC.append(df_res_onshore_UC, ignore_index=True, sort=False)
    df_LOPF = df_LOPF.append(df_res_onshore_LOPF, ignore_index=True, sort=False)

    # then the PV farms
    df_res_PV = df_res.loc[df_res['Technology Type'] == 'Solar Photovoltaics'].reset_index(drop=True)
    df_res_PV = df_res_PV.drop(
        columns=['CHP Enabled', 'No. of Turbines', 'Development Status',
                 'X-coordinate', 'Y-coordinate', 'Operational',
                 'Height of Turbines (m)', 'Mounting Type for Solar',
                 'Turbine Capacity (MW)'])
    df_res_PV = df_res_PV.rename(columns={'Installed Capacity (MWelec)': 'p_nom',
                                          'Technology Type': 'carrier',
                                          'Site Name': 'name',
                                          'lon': 'x',
                                          'lat': 'y'})
    df_res_PV['type'] = 'Solar Photovoltaics'

    # scale the PV to real data, see correction factors in Atlite
    if year <= 2020:
        df_res_PV.loc[:, 'p_nom'] *= df_correction.loc['PV', str(year)]

    df_res_PV_UC = df_res_PV.drop(
        columns=['x', 'y'])
    df_res_PV_UC['bus'] = 'bus'

    df_res_PV_LOPF = df_res_PV.drop(
        columns=['x', 'y'])
    df_res_PV_LOPF['bus'] = dc.map_to_bus(df_res_PV)

    # join to previous df of thermal power plants and offshore wind
    df_UC = df_UC.append(df_res_PV_UC, ignore_index=True, sort=False)
    df_LOPF = df_LOPF.append(df_res_PV_LOPF, ignore_index=True, sort=False)

    # add hydro data
    df_hydro = renewables.read_hydro(year)
    # drop pumped storage as this is going to be a storage unit
    df_hydro = df_hydro[~df_hydro.type.str.contains("Pumped Storage Hydroelectricity")]
    df_hydro = df_hydro.rename(columns={'lon': 'x',
                                        'lat': 'y'})
    df_hydro['x'] = df_hydro['x'].astype(float)
    df_hydro['y'] = df_hydro['y'].astype(float)

    # UC bus is same
    df_hydro_UC = df_hydro.drop(
        columns=['x', 'y'])
    df_hydro_UC['bus'] = 'bus'

    df_hydro_LOPF = df_hydro.drop(
        columns=['x', 'y'])
    df_hydro_LOPF['bus'] = dc.map_to_bus(df_hydro)

    # join to previous df of thermal power plants and offshore wind
    df_UC = df_UC.append(df_hydro_UC, ignore_index=True, sort=False)
    df_LOPF = df_LOPF.append(df_hydro_LOPF, ignore_index=True, sort=False)

    # add non dispatchable generators "RES"
    df_NDC = renewables.read_non_dispatchable_continuous(year)

    df_NDC = df_NDC.rename(columns={'lon': 'x',
                                    'lat': 'y'})

    # UC bus is same
    df_NDC_UC = df_NDC.drop(
        columns=['x', 'y'])
    df_NDC_UC['bus'] = 'bus'

    df_NDC_LOPF = df_NDC.drop(
        columns=['x', 'y'])
    df_NDC_LOPF['bus'] = dc.map_to_bus(df_NDC)

    # join to previous df of thermal power plants and offshore wind
    df_UC = df_UC.append(df_NDC_UC, ignore_index=True, sort=False)
    df_LOPF = df_LOPF.append(df_NDC_LOPF, ignore_index=True, sort=False)

    # add biomass boilers
    df_bio = renewables.read_biomass(year)
    df_bio = df_bio.rename(columns={'lon': 'x',
                                    'lat': 'y'})

    # UC bus is same
    df_bio_UC = df_bio.drop(
        columns=['x', 'y'])
    df_bio_UC['bus'] = 'bus'

    df_bio_LOPF = df_bio.drop(
        columns=['x', 'y'])
    df_bio_LOPF['bus'] = dc.map_to_bus(df_bio)

    # join to previous df of thermal power plants and offshore wind
    df_UC = df_UC.append(df_bio_UC, ignore_index=True, sort=False)
    df_LOPF = df_LOPF.append(df_bio_LOPF, ignore_index=True, sort=False)

    # run additional data for both UC and LOPF
    df_UC = generator_additional_data(df_UC, time_step)
    df_LOPF = generator_additional_data(df_LOPF, time_step)
    # remove the unit committent constraints
    df_LOPF = df_LOPF.drop(
        columns=['committable', 'min_up_time', 'min_down_time',
                 'p_min_pu', 'up_time_before', 'start_up_cost'])

    # remove a-cirumflex characters
    # cols = df_LOPF.select_dtypes(include=[np.object]).columns
    # df_LOPF[cols] = df_LOPF[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    df_LOPF['name'] = df_LOPF['name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_LOPF['name'] = df_LOPF['name'].astype(str).str.replace(u'\xa0', '')
    df_LOPF['name'] = df_LOPF['name'].astype(str).str.replace('ì', 'i')
    df_LOPF['name'] = df_LOPF['name'].str.strip()

    # df_UC[cols] = df_UC[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    df_UC['name'] = df_UC['name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_UC['name'] = df_UC['name'].astype(str).str.replace(u'\xa0', '')
    df_UC['name'] = df_UC['name'].astype(str).str.replace('ì', 'i')
    df_UC['name'] = df_UC['name'].str.strip()

    # check names are unique for UC
    duplicateDFRow = df_UC[df_UC.duplicated(['name'], keep='first')]
    for i in range(len(duplicateDFRow.index.values)):
        # print(df_UC['name'][duplicateDFRow.index.values[i]])
        df_UC.at[duplicateDFRow.index.values[i], 'name'] = (
            df_UC['name'][duplicateDFRow.index.values[i]] + '.1')
        # print(df_UC['name'][duplicateDFRow.index.values[i]])

    # check names are unique for LOPF
    duplicateDFRow = df_LOPF[df_LOPF.duplicated(['name'], keep='first')]
    for i in range(len(duplicateDFRow.index.values)):
        # print(df_LOPF['name'][duplicateDFRow.index.values[i]])
        df_LOPF.at[duplicateDFRow.index.values[i], 'name'] = (
            df_LOPF['name'][duplicateDFRow.index.values[i]] + '.1')
        # print(df_LOPF['name'][duplicateDFRow.index.values[i]])

    # save the dataframes to csv
    df_UC.to_csv('UC_data/generators.csv', index=False, header=True)
    df_LOPF.to_csv('LOPF_data/generators.csv', index=False, header=True)


def generator_additional_data(df, time_step):
    """adds data to the generators csv file

    add marginal costs, min up/down time, ramp up/down rates
    to the generators csv file

    Parameters
    ----------
    df : dataframe
        dataframe containing the generator data to be added to
    time_step : float
        defined as fraction of an hour, e.g., 0.5 is half hour
        currently set up as only hour or half hour
    Returns
    dataframe
        data with all required generator for PyPSA UC or LOPF
    -------
    """

    # add marginal costs, min up/down time, ramp up/down rates
    df_data = read_generator_data_by_fuel()

    conditions = [
        (df['carrier'] == 'Coal'),
        (df['carrier'] == 'Oil'),
        (df['carrier'] == 'Natural Gas') & (df['type'] == 'CCGT'),
        (df['carrier'] == 'Natural Gas') & (df['type'] == 'OCGT'),
        (df['carrier'] == 'Natural Gas') & (df['type'] == 'Sour gas'),
        (df['carrier'] == 'Nuclear'),
        (df['carrier'] == 'Wind Offshore'),
        (df['carrier'] == 'Large Hydro'),
        (df['carrier'] == 'Small Hydro'),
        (df['carrier'] == 'Anaerobic Digestion'),
        (df['carrier'] == 'EfW Incineration'),
        (df['carrier'] == 'Landfill Gas'),
        (df['carrier'] == 'Sewage Sludge Digestion'),
        (df['carrier'] == 'Shoreline Wave'),
        (df['carrier'] == 'Tidal Barrage and Tidal Stream'),
        (df['carrier'] == 'Biomass (dedicated)'),
        (df['carrier'] == 'Biomass (co-firing)'),
        (df['carrier'] == 'Wind Onshore'),
        (df['carrier'] == 'Solar Photovoltaics'),
        (df['carrier'] == 'CCS Gas'),
        (df['carrier'] == 'CCS Biomass'),
        (df['carrier'] == 'Hydrogen')]

    marg_cos = [df_data['marginal_costs']['Coal'],
                df_data['marginal_costs']['Oil'],
                df_data['marginal_costs']['CCGT'],
                df_data['marginal_costs']['OCGT'],
                df_data['marginal_costs']['Sour gas'],
                df_data['marginal_costs']['Nuclear'],
                df_data['marginal_costs']['Wind Offshore'],
                df_data['marginal_costs']['Large Hydro'],
                df_data['marginal_costs']['Small Hydro'],
                df_data['marginal_costs']['Anaerobic Digestion'],
                df_data['marginal_costs']['EfW Incineration'],
                df_data['marginal_costs']['Landfill Gas'],
                df_data['marginal_costs']['Sewage Sludge Digestion'],
                df_data['marginal_costs']['Shoreline Wave'],
                df_data['marginal_costs']['Tidal Barrage and Tidal Stream'],
                df_data['marginal_costs']['Biomass (dedicated)'],
                df_data['marginal_costs']['Biomass (co-firing)'],
                df_data['marginal_costs']['Wind Onshore'],
                df_data['marginal_costs']['Solar Photovoltaics'],
                df_data['marginal_costs']['CCS Gas'],
                df_data['marginal_costs']['CCS Biomass'],
                df_data['marginal_costs']['Hydrogen']]

    min_up_time_ = [df_data['min_up_time']['Coal'],
                    df_data['min_up_time']['Oil'],
                    df_data['min_up_time']['CCGT'],
                    df_data['min_up_time']['OCGT'],
                    df_data['min_up_time']['Sour gas'],
                    df_data['min_up_time']['Nuclear'],
                    df_data['min_up_time']['Wind Offshore'],
                    df_data['min_up_time']['Large Hydro'],
                    df_data['min_up_time']['Small Hydro'],
                    df_data['min_up_time']['Anaerobic Digestion'],
                    df_data['min_up_time']['EfW Incineration'],
                    df_data['min_up_time']['Landfill Gas'],
                    df_data['min_up_time']['Sewage Sludge Digestion'],
                    df_data['min_up_time']['Shoreline Wave'],
                    df_data['min_up_time']['Tidal Barrage and Tidal Stream'],
                    df_data['min_up_time']['Biomass (dedicated)'],
                    df_data['min_up_time']['Biomass (co-firing)'],
                    df_data['min_up_time']['Wind Onshore'],
                    df_data['min_up_time']['Solar Photovoltaics'],
                    df_data['min_up_time']['CCS Gas'],
                    df_data['min_up_time']['CCS Biomass'],
                    df_data['min_up_time']['Hydrogen']]

    min_down_time_ = [df_data['min_down_time']['Coal'],
                      df_data['min_down_time']['Oil'],
                      df_data['min_down_time']['CCGT'],
                      df_data['min_down_time']['OCGT'],
                      df_data['min_down_time']['Sour gas'],
                      df_data['min_down_time']['Nuclear'],
                      df_data['min_down_time']['Wind Offshore'],
                      df_data['min_down_time']['Large Hydro'],
                      df_data['min_down_time']['Small Hydro'],
                      df_data['min_down_time']['Anaerobic Digestion'],
                      df_data['min_down_time']['EfW Incineration'],
                      df_data['min_down_time']['Landfill Gas'],
                      df_data['min_down_time']['Sewage Sludge Digestion'],
                      df_data['min_down_time']['Shoreline Wave'],
                      df_data['min_down_time']['Tidal Barrage and Tidal Stream'],
                      df_data['min_down_time']['Biomass (dedicated)'],
                      df_data['min_down_time']['Biomass (co-firing)'],
                      df_data['min_down_time']['Wind Onshore'],
                      df_data['min_down_time']['Solar Photovoltaics'],
                      df_data['min_down_time']['CCS Gas'],
                      df_data['min_down_time']['CCS Biomass'],
                      df_data['min_down_time']['Hydrogen']]

    ramp_limit_up_ = [df_data['ramp_limit_up']['Coal'],
                      df_data['ramp_limit_up']['Oil'],
                      df_data['ramp_limit_up']['CCGT'],
                      df_data['ramp_limit_up']['OCGT'],
                      df_data['ramp_limit_up']['Sour gas'],
                      df_data['ramp_limit_up']['Nuclear'],
                      df_data['ramp_limit_up']['Wind Offshore'],
                      df_data['ramp_limit_up']['Large Hydro'],
                      df_data['ramp_limit_up']['Small Hydro'],
                      df_data['ramp_limit_up']['Anaerobic Digestion'],
                      df_data['ramp_limit_up']['EfW Incineration'],
                      df_data['ramp_limit_up']['Landfill Gas'],
                      df_data['ramp_limit_up']['Sewage Sludge Digestion'],
                      df_data['ramp_limit_up']['Shoreline Wave'],
                      df_data['ramp_limit_up']['Tidal Barrage and Tidal Stream'],
                      df_data['ramp_limit_up']['Biomass (dedicated)'],
                      df_data['ramp_limit_up']['Biomass (co-firing)'],
                      df_data['ramp_limit_up']['Wind Onshore'],
                      df_data['ramp_limit_up']['Solar Photovoltaics'],
                      df_data['ramp_limit_up']['CCS Gas'],
                      df_data['ramp_limit_up']['CCS Biomass'],
                      df_data['ramp_limit_up']['Hydrogen']]

    ramp_limit_down_ = [df_data['ramp_limit_down']['Coal'],
                        df_data['ramp_limit_down']['Oil'],
                        df_data['ramp_limit_down']['CCGT'],
                        df_data['ramp_limit_down']['OCGT'],
                        df_data['ramp_limit_down']['Sour gas'],
                        df_data['ramp_limit_down']['Nuclear'],
                        df_data['ramp_limit_down']['Wind Offshore'],
                        df_data['ramp_limit_down']['Large Hydro'],
                        df_data['ramp_limit_down']['Small Hydro'],
                        df_data['ramp_limit_down']['Anaerobic Digestion'],
                        df_data['ramp_limit_down']['EfW Incineration'],
                        df_data['ramp_limit_down']['Landfill Gas'],
                        df_data['ramp_limit_down']['Sewage Sludge Digestion'],
                        df_data['ramp_limit_down']['Shoreline Wave'],
                        df_data['ramp_limit_down']['Tidal Barrage and Tidal Stream'],
                        df_data['ramp_limit_down']['Biomass (dedicated)'],
                        df_data['ramp_limit_down']['Biomass (co-firing)'],
                        df_data['ramp_limit_down']['Wind Onshore'],
                        df_data['ramp_limit_down']['Solar Photovoltaics'],
                        df_data['ramp_limit_down']['CCS Gas'],
                        df_data['ramp_limit_down']['CCS Biomass'],
                        df_data['ramp_limit_down']['Hydrogen']]

    committable_ = [df_data['committable']['Coal'],
                    df_data['committable']['Oil'],
                    df_data['committable']['CCGT'],
                    df_data['committable']['OCGT'],
                    df_data['committable']['Sour gas'],
                    df_data['committable']['Nuclear'],
                    df_data['committable']['Wind Offshore'],
                    df_data['committable']['Large Hydro'],
                    df_data['committable']['Small Hydro'],
                    df_data['committable']['Anaerobic Digestion'],
                    df_data['committable']['EfW Incineration'],
                    df_data['committable']['Landfill Gas'],
                    df_data['committable']['Sewage Sludge Digestion'],
                    df_data['committable']['Shoreline Wave'],
                    df_data['committable']['Tidal Barrage and Tidal Stream'],
                    df_data['committable']['Biomass (dedicated)'],
                    df_data['committable']['Biomass (co-firing)'],
                    df_data['committable']['Wind Onshore'],
                    df_data['committable']['Solar Photovoltaics'],
                    df_data['committable']['CCS Gas'],
                    df_data['committable']['CCS Biomass'],
                    df_data['committable']['Hydrogen']]

    p_min_pu_ = [df_data['p_min_pu']['Coal'],
                 df_data['p_min_pu']['Oil'],
                 df_data['p_min_pu']['CCGT'],
                 df_data['p_min_pu']['OCGT'],
                 df_data['p_min_pu']['Sour gas'],
                 df_data['p_min_pu']['Nuclear'],
                 df_data['p_min_pu']['Wind Offshore'],
                 df_data['p_min_pu']['Large Hydro'],
                 df_data['p_min_pu']['Small Hydro'],
                 df_data['p_min_pu']['Anaerobic Digestion'],
                 df_data['p_min_pu']['EfW Incineration'],
                 df_data['p_min_pu']['Landfill Gas'],
                 df_data['p_min_pu']['Sewage Sludge Digestion'],
                 df_data['p_min_pu']['Shoreline Wave'],
                 df_data['p_min_pu']['Tidal Barrage and Tidal Stream'],
                 df_data['p_min_pu']['Biomass (dedicated)'],
                 df_data['p_min_pu']['Biomass (co-firing)'],
                 df_data['p_min_pu']['Wind Onshore'],
                 df_data['p_min_pu']['Solar Photovoltaics'],
                 df_data['p_min_pu']['CCS Gas'],
                 df_data['p_min_pu']['CCS Biomass'],
                 df_data['p_min_pu']['Hydrogen']]

    p_max_pu_ = [df_data['p_max_pu']['Coal'],
                 df_data['p_max_pu']['Oil'],
                 df_data['p_max_pu']['CCGT'],
                 df_data['p_max_pu']['OCGT'],
                 df_data['p_max_pu']['Sour gas'],
                 df_data['p_max_pu']['Nuclear'],
                 df_data['p_max_pu']['Wind Offshore'],
                 df_data['p_max_pu']['Large Hydro'],
                 df_data['p_max_pu']['Small Hydro'],
                 df_data['p_max_pu']['Anaerobic Digestion'],
                 df_data['p_max_pu']['EfW Incineration'],
                 df_data['p_max_pu']['Landfill Gas'],
                 df_data['p_max_pu']['Sewage Sludge Digestion'],
                 df_data['p_max_pu']['Shoreline Wave'],
                 df_data['p_max_pu']['Tidal Barrage and Tidal Stream'],
                 df_data['p_max_pu']['Biomass (dedicated)'],
                 df_data['p_max_pu']['Biomass (co-firing)'],
                 df_data['p_max_pu']['Wind Onshore'],
                 df_data['p_max_pu']['Solar Photovoltaics'],
                 df_data['p_max_pu']['CCS Gas'],
                 df_data['p_max_pu']['CCS Biomass'],
                 df_data['p_max_pu']['Hydrogen']]

    up_time_before_ = [df_data['up_time_before']['Coal'],
                       df_data['up_time_before']['Oil'],
                       df_data['up_time_before']['CCGT'],
                       df_data['up_time_before']['OCGT'],
                       df_data['up_time_before']['Sour gas'],
                       df_data['up_time_before']['Nuclear'],
                       df_data['up_time_before']['Wind Offshore'],
                       df_data['up_time_before']['Large Hydro'],
                       df_data['up_time_before']['Small Hydro'],
                       df_data['up_time_before']['Anaerobic Digestion'],
                       df_data['up_time_before']['EfW Incineration'],
                       df_data['up_time_before']['Landfill Gas'],
                       df_data['up_time_before']['Sewage Sludge Digestion'],
                       df_data['up_time_before']['Shoreline Wave'],
                       df_data['up_time_before']['Tidal Barrage and Tidal Stream'],
                       df_data['up_time_before']['Biomass (dedicated)'],
                       df_data['up_time_before']['Biomass (co-firing)'],
                       df_data['up_time_before']['Wind Onshore'],
                       df_data['up_time_before']['Solar Photovoltaics'],
                       df_data['up_time_before']['CCS Gas'],
                       df_data['up_time_before']['CCS Biomass'],
                       df_data['up_time_before']['Hydrogen']]

    start_up_cost_ = [df_data['start_up_cost']['Coal'],
                      df_data['start_up_cost']['Oil'],
                      df_data['start_up_cost']['CCGT'],
                      df_data['start_up_cost']['OCGT'],
                      df_data['start_up_cost']['Sour gas'],
                      df_data['start_up_cost']['Nuclear'],
                      df_data['start_up_cost']['Wind Offshore'],
                      df_data['start_up_cost']['Large Hydro'],
                      df_data['start_up_cost']['Small Hydro'],
                      df_data['start_up_cost']['Anaerobic Digestion'],
                      df_data['start_up_cost']['EfW Incineration'],
                      df_data['start_up_cost']['Landfill Gas'],
                      df_data['start_up_cost']['Sewage Sludge Digestion'],
                      df_data['start_up_cost']['Shoreline Wave'],
                      df_data['start_up_cost']['Tidal Barrage and Tidal Stream'],
                      df_data['start_up_cost']['Biomass (dedicated)'],
                      df_data['start_up_cost']['Biomass (co-firing)'],
                      df_data['start_up_cost']['Wind Onshore'],
                      df_data['start_up_cost']['Solar Photovoltaics'],
                      df_data['start_up_cost']['CCS Gas'],
                      df_data['start_up_cost']['CCS Biomass'],
                      df_data['start_up_cost']['Hydrogen']]

    df.loc[:, 'marginal_cost'] = np.select(conditions, marg_cos)
    df.loc[:, 'committable'] = np.select(conditions, committable_)
    df.loc[:, 'committable'] = df['committable'].astype('bool')
    df.loc[:, 'min_up_time'] = np.select(conditions, min_up_time_) / (time_step * 60)
    df.loc[:, 'min_up_time'] = df['min_up_time'].astype('int')
    # df.loc[:, 'min_up_time'] = 0
    df.loc[:, 'min_down_time'] = np.select(conditions, min_down_time_) / (time_step * 60)
    df.loc[:, 'min_down_time'] = df['min_down_time'].astype('int')
    # df.loc[:, 'min_down_time'] = 0
    # need to ensure the capacity is a float and not a string
    df.loc[:, 'p_nom'] = pd.to_numeric(df["p_nom"], downcast="float")
    df.loc[:, 'ramp_limit_up'] = np.select(conditions, ramp_limit_up_) * (time_step * 60) / 100
    df.loc[:, 'ramp_limit_up'].values[df['ramp_limit_up'].values > 1.0] = 1.0
    df.loc[:, 'ramp_limit_down'] = np.select(conditions, ramp_limit_down_) * (time_step * 60) / 100
    df.loc[:, 'ramp_limit_down'].values[df['ramp_limit_down'].values > 1.0] = 1.0
    df.loc[:, 'p_min_pu'] = np.select(conditions, p_min_pu_) / 100
    df.loc[:, 'p_max_pu'] = np.select(conditions, p_max_pu_) / 100
    # df.loc[:, 'p_min_pu'] = 0
    df.loc[:, 'up_time_before'] = np.select(conditions, up_time_before_)
    df.loc[:, 'up_time_before'] = df['up_time_before'].astype('int')
    df.loc[:, 'start_up_cost'] = np.select(conditions, start_up_cost_)
    df.loc[:, 'start_up_cost'] *= df['p_nom']

    return df


def future_coal_p_nom(year):
    # read in phase out of coal dates
    file = '../data/power stations/coal_phase_out_dates.csv'
    df = pd.read_csv(file, index_col=1)
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)
    end_date = str(year) + '-01-01'
    filtered_df = df.loc[:end_date]
    pp_to_remove = filtered_df.name.values

    # get generators dataframe with p_noms to be scaled
    path = 'LOPF_data/generators.csv'
    generators = pd.read_csv(path, index_col=0)
    path_UC = 'UC_data/generators.csv'
    generators_UC = pd.read_csv(path_UC, index_col=0)

    # error occurs because deleting PV farm with Ratcliffe in name
    # just add in later
    try:
        PV_ratcliffe = generators.loc[['Ld NW Of Ratcliffe House Farm']]
        PV_ratcliffe_UC = generators_UC.loc[['Ld NW Of Ratcliffe House Farm']]
    except:
        pass

    for i in range(len(pp_to_remove)):
        # remove those who are not in date
        # just look at the coal generators
        # keep PV row with Ratcliffe in name

        generators = generators[~generators.index.str.contains(pp_to_remove[i])]
        generators_UC = generators_UC[~generators_UC.index.str.contains(pp_to_remove[i])]

    # append the PV farm with Ratcliffe in name, only in years Ratcliffe is removed
    try:
        if year > 2024:
            generators = generators.append(PV_ratcliffe)
            generators_UC = generators_UC.append(PV_ratcliffe_UC)
    except:
        pass

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_gas_p_nom(year, scenario, tech):
    # going to scale the OCGT and CCGT based on FES
    future_capacities_dict = future_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

    if tech == 'OCGT' or tech == 'CCGT':
        # get generators dataframe with p_noms to be scaled
        path = 'LOPF_data/generators.csv'
        generators = pd.read_csv(path, index_col=0)
        gen_tech = generators.loc[generators['type'] == tech]

        path_UC = 'UC_data/generators.csv'
        generators_UC = pd.read_csv(path_UC, index_col=0)
        gen_tech_UC = generators_UC.loc[generators_UC['type'] == tech]

    # then consider what scaling factor is required
    scaling_factor = round(tech_cap_FES / tech_cap_year, 2)

    # scale the p_noms of the RES generators
    for g in gen_tech.index:
        gen_tech.loc[g, 'p_nom'] *= scaling_factor
        gen_tech_UC.loc[g, 'p_nom'] *= scaling_factor

    # write new generators.csv file
    # save the dataframes to csv
    # need to remove the original tech
    # print(generators.loc[generators['type'] == tech])
    generators = generators[~generators.type.str.contains(tech)]
    generators_UC = generators_UC[~generators_UC.type.str.contains(tech)]
    # then add the new p_nom tech
    generators = generators.append(gen_tech)
    generators_UC = generators_UC.append(gen_tech_UC)
    # print(generators.loc[generators['type'] == tech])

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_nuclear_p_nom(year, scenario):
    # read in phase out of nuclear dates
    file = '../data/power stations/nuclear_phase_out_dates.csv'
    df = pd.read_csv(file, index_col=1)
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)
    end_date = str(year) + '-01-01'
    filtered_df = df.loc[:end_date]
    pp_to_remove = filtered_df.name.values

    # get generators dataframe with p_noms to be scaled
    path = 'LOPF_data/generators.csv'
    generators = pd.read_csv(path, index_col=0)
    path_UC = 'UC_data/generators.csv'
    generators_UC = pd.read_csv(path_UC, index_col=0)
    # print(generators.loc[generators['carrier'] == 'Nuclear'])
    for i in range(len(pp_to_remove)):
        # remove those who are not in date
        generators = generators[~generators.index.str.contains(pp_to_remove[i])]
        generators_UC = generators_UC[~generators_UC.index.str.contains(pp_to_remove[i])]
    # print(generators.loc[generators['carrier'] == 'Nuclear'])

    # read in new nuclear power plant data
    file = '../data/power stations/nuclear_new_dates.csv'
    df = pd.read_csv(file, index_col=1)
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)
    to_date = str(year) + '-01-01'
    filtered_df = df.loc[:to_date]
    pp_to_add = filtered_df.name.values
    # print(pp_to_add)
    # print(generators.loc[generators['carrier'] == 'Nuclear'])
    generators2 = pd.read_csv(path, index_col=0)
    new_nuclear = generators2.loc[generators2['carrier'] == 'Nuclear'].iloc[[0]]
    # print(new_nuclear)
    df_new_nuclear = pd.DataFrame()
    for i in range(len(pp_to_add)):
        # add new generators
        # template
        new_nuclear = generators2.loc[generators2['carrier'] == 'Nuclear'].iloc[[0]]
        p_nom = filtered_df.loc[filtered_df['name'] == pp_to_add[i]]['p_nom'].values[0]
        new_nuclear.loc[:, 'p_nom'] = p_nom
        new_nuclear.loc[:, 'type'] = 'PWR'
        new_nuclear.index = [pp_to_add[i]]
        new_nuclear['x'] = filtered_df.loc[filtered_df['name'] == pp_to_add[i]]['x'].values[0]
        new_nuclear['y'] = filtered_df.loc[filtered_df['name'] == pp_to_add[i]]['y'].values[0]
        # need to map to bus
        new_nuclear['bus'] = dc.map_to_bus(new_nuclear)
        df_new_nuclear = df_new_nuclear.append(new_nuclear)

    # only append to generators dataframe if there are rows in df_new_nuclear
    if not df_new_nuclear.empty:
        df_new_nuclear.drop(columns=['x', 'y'], inplace=True)
        # now add new nuclear to generators df
        generators = generators.append(df_new_nuclear)
        generators_UC = generators_UC.append(df_new_nuclear)
        # print(generators.loc[generators['carrier'] == 'Nuclear'])

    # this gets us to 2030, but want to scale the old nuclear sites
    # for > 2030
    if year > 2030:
        tech = 'Nuclear'
        future_capacities_dict = future_capacity(year, tech, scenario)
        tech_cap_year = future_capacities_dict['tech_cap_year']
        tech_cap_FES = future_capacities_dict['tech_cap_FES']

        # get generators dataframe with p_noms to be scaled
        path = 'LOPF_data/generators.csv'
        generators3 = pd.read_csv(path, index_col=0)
        gen_tech = generators3.loc[generators3['carrier'] == tech]

        path_UC = 'UC_data/generators.csv'
        generators_UC3 = pd.read_csv(path_UC, index_col=0)
        gen_tech_UC = generators_UC3.loc[generators_UC3['carrier'] == tech]

        # then consider what scaling factor is required
        scaling_factor = round(tech_cap_FES / tech_cap_year, 2)

        # scale the p_noms of the RES generators
        for g in gen_tech.index:
            gen_tech.loc[g, 'p_nom'] *= scaling_factor
            gen_tech_UC.loc[g, 'p_nom'] *= scaling_factor

        # write new generators.csv file
        # save the dataframes to csv
        # need to remove the original tech
        generators3 = generators3[~generators3.carrier.str.contains(tech)]
        generators_UC3 = generators_UC3[~generators_UC3.carrier.str.contains(tech)]
        # then add the new p_nom tech
        generators3 = generators3.append(gen_tech)
        generators_UC3 = generators_UC3.append(gen_tech_UC)

    else:
        generators_UC3 = generators_UC.copy()
        generators3 = generators.copy()

    generators_UC3.to_csv('UC_data/generators.csv', header=True)
    generators3.to_csv('LOPF_data/generators.csv', header=True)


def future_oil_p_nom(year, scenario):
    tech = 'Oil'
    # going to scale the oil based on FES
    future_capacities_dict = future_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

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
    # print(generators.loc[generators['carrier'] == tech])
    generators = generators[~generators.carrier.str.contains(tech)]
    generators_UC = generators_UC[~generators_UC.carrier.str.contains(tech)]
    # then add the new p_nom tech
    generators = generators.append(gen_tech)
    generators_UC = generators_UC.append(gen_tech_UC)
    # print(generators.loc[generators['carrier'] == tech])

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_waste_p_nom(year, scenario):
    tech = 'Waste'
    # going to scale the oil based on FES
    future_capacities_dict = future_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

    tech = 'EfW Incineration'
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
    # print(generators.loc[generators['carrier'] == tech])
    generators = generators[~generators.carrier.str.contains(tech)]
    generators_UC = generators_UC[~generators_UC.carrier.str.contains(tech)]
    # then add the new p_nom tech
    generators = generators.append(gen_tech)
    generators_UC = generators_UC.append(gen_tech_UC)
    # print(generators.loc[generators['carrier'] == tech])

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_gas_CCS(year, scenario):
    tech = 'CCS Gas'
    # going to scale the existing gas sites based on FES
    # but add as new tech
    future_capacities_dict = future_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

    tech_ = 'CCGT'
    # get CCGT generators as they are in year
    # need to ensure doing this function before scaling gas
    path = 'LOPF_data/generators.csv'
    generators = pd.read_csv(path, index_col=0)
    gen_tech = generators.loc[generators['type'] == tech_]

    path_UC = 'UC_data/generators.csv'
    generators_UC = pd.read_csv(path_UC, index_col=0)
    gen_tech_UC = generators_UC.loc[generators_UC['type'] == tech_]

    # then consider what scaling factor is required
    scaling_factor = round(tech_cap_FES / tech_cap_year, 2)

    # don't want to scale original CCGT so need copy
    gen_tech2 = gen_tech.copy()
    gen_tech_UC2 = gen_tech_UC.copy()

    # scale the p_noms of the RES generators
    for g in gen_tech.index:
        gen_tech2.loc[g, 'p_nom'] *= scaling_factor
        gen_tech_UC2.loc[g, 'p_nom'] *= scaling_factor

    # write new generators.csv file
    # save the dataframes to csv
    # need to change the carrier and type before appending
    # also note that not removing CCGT generators
    gen_tech2.loc[:, 'type'] = 'CCS Gas'
    gen_tech2.loc[:, 'carrier'] = 'CCS Gas'
    gen_tech_UC2.loc[:, 'type'] = 'CCS Gas'
    gen_tech_UC2.loc[:, 'carrier'] = 'CCS Gas'
    # need to modify names by adding CCS Gas
    gen_tech2.index += ' CCS Gas'
    gen_tech_UC2.index += ' CCS Gas'
    # then add the new p_nom tech
    generators = generators.append(gen_tech2)
    generators_UC = generators_UC.append(gen_tech_UC2)
    # print(generators.loc[generators['carrier'] == tech])
    # print(generators.loc[generators['carrier'] == 'Natural Gas'])

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_biomass_CCS(year, scenario):
    tech = 'CCS Biomass'
    # going to scale the existing gas sites based on FES
    # but add as new tech
    future_capacities_dict = future_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

    # CCS biomass sometimes coming out as nan, replace with zero
    if np.isnan(tech_cap_FES):
        tech_cap_FES = 0.0

    tech_ = 'CCGT'
    # get CCGT generators as they are in year
    # need to ensure doing this function before scaling gas
    path = 'LOPF_data/generators.csv'
    generators = pd.read_csv(path, index_col=0)
    gen_tech = generators.loc[generators['type'] == tech_]

    path_UC = 'UC_data/generators.csv'
    generators_UC = pd.read_csv(path_UC, index_col=0)
    gen_tech_UC = generators_UC.loc[generators_UC['type'] == tech_]

    # then consider what scaling factor is required
    scaling_factor = round(tech_cap_FES / tech_cap_year, 2)

    # don't want to scale original CCGT so need copy
    gen_tech2 = gen_tech.copy()
    gen_tech_UC2 = gen_tech_UC.copy()

    # scale the p_noms of the RES generators
    for g in gen_tech.index:
        gen_tech2.loc[g, 'p_nom'] *= scaling_factor
        gen_tech_UC2.loc[g, 'p_nom'] *= scaling_factor

    # write new generators.csv file
    # save the dataframes to csv
    # need to change the carrier and type before appending
    # also note that not removing CCGT generators
    gen_tech2.loc[:, 'type'] = 'CCS Biomass'
    gen_tech2.loc[:, 'carrier'] = 'CCS Biomass'
    gen_tech_UC2.loc[:, 'type'] = 'CCS Biomass'
    gen_tech_UC2.loc[:, 'carrier'] = 'CCS Biomass'
    # need to modify names by adding CCS Biomass
    gen_tech2.index += ' CCS Biomass'
    gen_tech_UC2.index += ' CCS Biomass'
    # then add the new p_nom tech
    generators = generators.append(gen_tech2)
    generators_UC = generators_UC.append(gen_tech_UC2)
    # print(generators.loc[generators['carrier'] == 'Natural Gas'])

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_hydrogen(year, scenario):
    tech = 'Hydrogen'
    # going to scale the existing gas sites based on FES
    # but add as new tech
    future_capacities_dict = future_capacity(year, tech, scenario)
    tech_cap_year = future_capacities_dict['tech_cap_year']
    tech_cap_FES = future_capacities_dict['tech_cap_FES']

    # hydrogen sometimes coming out as nan, replace with zero
    if np.isnan(tech_cap_FES):
        tech_cap_FES = 0.0

    tech_ = 'CCGT'
    # get CCGT generators as they are in year
    # need to ensure doing this function before scaling gas
    path = 'LOPF_data/generators.csv'
    generators = pd.read_csv(path, index_col=0)
    gen_tech = generators.loc[generators['type'] == tech_]

    path_UC = 'UC_data/generators.csv'
    generators_UC = pd.read_csv(path_UC, index_col=0)
    gen_tech_UC = generators_UC.loc[generators_UC['type'] == tech_]

    # then consider what scaling factor is required
    scaling_factor = round(tech_cap_FES / tech_cap_year, 2)
    # print(scaling_factor, 'hydrogen scaling factor')

    # don't want to scale original CCGT so need copy
    gen_tech2 = gen_tech.copy()
    gen_tech_UC2 = gen_tech_UC.copy()

    # scale the p_noms of the RES generators
    for g in gen_tech.index:
        gen_tech2.loc[g, 'p_nom'] *= scaling_factor
        gen_tech_UC2.loc[g, 'p_nom'] *= scaling_factor

    # write new generators.csv file
    # save the dataframes to csv
    # need to change the carrier and type before appending
    # also note that not removing CCGT generators
    gen_tech2.loc[:, 'type'] = 'Hydrogen'
    gen_tech2.loc[:, 'carrier'] = 'Hydrogen'
    gen_tech_UC2.loc[:, 'type'] = 'Hydrogen'
    gen_tech_UC2.loc[:, 'carrier'] = 'Hydrogen'
    # need to modify names by adding Hydrogen
    gen_tech2.index += ' Hydrogen'
    gen_tech_UC2.index += ' Hydrogen'
    # then add the new p_nom tech
    generators = generators.append(gen_tech2)
    generators_UC = generators_UC.append(gen_tech_UC2)
    # print(generators.loc[generators['carrier'] == 'Hydrogen'])
    # print(generators.loc[generators['carrier'] == 'CCS Biomass'])

    generators_UC.to_csv('UC_data/generators.csv', header=True)
    generators.to_csv('LOPF_data/generators.csv', header=True)


def future_capacity(year, tech, scenario):

    df_pp = read_power_stations_data(year)
    if tech == 'Nuclear' or tech == 'Oil':
        df_pp = df_pp[df_pp.Fuel.str.contains(tech)]
        tech_cap_year = df_pp['Installed Capacity (MW)'].sum() / 1000
    elif tech == 'CCGT' or tech == 'OCGT':
        df_pp = df_pp[df_pp.Technology.str.contains(tech)]
        tech_cap_year = df_pp['Installed Capacity (MW)'].sum() / 1000
    elif tech == 'Waste':
        df_pp = renewables.read_non_dispatchable_continuous(year)
        df_pp = df_pp[df_pp.type.str.contains('EfW Incineration')]
        # this number excludes EFW CHP but FES includes it
        # which explains discrepancy
        tech_cap_year = df_pp['p_nom'].sum() / 1000
    elif tech == 'CCS Gas' or tech == 'CCS Biomass' or tech == 'Hydrogen':
        df_pp = df_pp[df_pp.Technology.str.contains('CCGT')]
        tech_cap_year = df_pp['Installed Capacity (MW)'].sum() / 1000

    if tech == 'CCGT':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        df_FES = df_FES[df_FES.SubType.str.contains('CCGT', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)
        # print(df_FES)

        df_FES_LTW = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        df_FES_LTW = df_FES_LTW.append(df_FES_LTW.sum(numeric_only=True), ignore_index=True)
        df_FES_LTW.drop([0, 1], inplace=True)
        df_FES_LTW.dropna(axis='columns', inplace=True)
        df_FES_LTW.index = ['Leading the Way']

        df_FES_CT = df_FES[df_FES.index.str.contains('Consumer Transformation', case=False)]
        df_FES_CT = df_FES_CT.append(df_FES_CT.sum(numeric_only=True), ignore_index=True)
        df_FES_CT.drop([0, 1], inplace=True)
        df_FES_CT.dropna(axis='columns', inplace=True)
        df_FES_CT.index = ['Consumer Transformation']

        df_FES_ST = df_FES[df_FES.index.str.contains('System Transformation', case=False)]
        df_FES_ST = df_FES_ST.append(df_FES_ST.sum(numeric_only=True), ignore_index=True)
        df_FES_ST.drop([0, 1], inplace=True)
        df_FES_ST.dropna(axis='columns', inplace=True)
        df_FES_ST.index = ['System Transformation']

        df_FES_SP = df_FES[df_FES.index.str.contains('Steady Progression', case=False)]
        df_FES_SP = df_FES_SP.append(df_FES_SP.sum(numeric_only=True), ignore_index=True)
        df_FES_SP.drop([0, 1], inplace=True)
        df_FES_SP.dropna(axis='columns', inplace=True)
        df_FES_SP.index = ['Steady Progression']

        df_FES = df_FES_SP.append([df_FES_LTW, df_FES_CT, df_FES_ST])

    elif tech == 'OCGT':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        df_FES = df_FES[df_FES.SubType.str.contains('OCGT', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)
        # print(df_FES)

        df_FES_LTW = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        df_FES_LTW = df_FES_LTW.append(df_FES_LTW.sum(numeric_only=True), ignore_index=True)
        df_FES_LTW.drop([0, 1], inplace=True)
        df_FES_LTW.dropna(axis='columns', inplace=True)
        df_FES_LTW.index = ['Leading the Way']

        df_FES_CT = df_FES[df_FES.index.str.contains('Consumer Transformation', case=False)]
        df_FES_CT = df_FES_CT.append(df_FES_CT.sum(numeric_only=True), ignore_index=True)
        df_FES_CT.drop([0, 1], inplace=True)
        df_FES_CT.dropna(axis='columns', inplace=True)
        df_FES_CT.index = ['Consumer Transformation']

        df_FES_ST = df_FES[df_FES.index.str.contains('System Transformation', case=False)]
        df_FES_ST = df_FES_ST.append(df_FES_ST.sum(numeric_only=True), ignore_index=True)
        df_FES_ST.drop([0, 1], inplace=True)
        df_FES_ST.dropna(axis='columns', inplace=True)
        df_FES_ST.index = ['System Transformation']

        df_FES_SP = df_FES[df_FES.index.str.contains('Steady Progression', case=False)]
        df_FES_SP = df_FES_SP.append(df_FES_SP.sum(numeric_only=True), ignore_index=True)
        df_FES_SP.drop([0, 1], inplace=True)
        df_FES_SP.dropna(axis='columns', inplace=True)
        df_FES_SP.index = ['Steady Progression']

        df_FES = df_FES_SP.append([df_FES_LTW, df_FES_CT, df_FES_ST])

    elif tech == 'Nuclear':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        df_FES.dropna(axis='rows', inplace=True)
        df_FES = df_FES[df_FES.Type.str.contains('Nuclear', case=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [0, 1, 2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)

    elif tech == 'Oil':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        # df_FES.dropna(axis='rows', inplace=True)
        df_FES = df_FES[df_FES.Type.str.contains('Other Thermal', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [0, 1, 2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)

        df_FES_LTW = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        df_FES_LTW = df_FES_LTW.append(df_FES_LTW.sum(numeric_only=True), ignore_index=True)
        df_FES_LTW.drop([0, 1], inplace=True)
        df_FES_LTW.index = ['Leading the Way']

        df_FES_CT = df_FES[df_FES.index.str.contains('Consumer Transformation', case=False)]
        df_FES_CT = df_FES_CT.append(df_FES_CT.sum(numeric_only=True), ignore_index=True)
        df_FES_CT.drop([0, 1], inplace=True)
        df_FES_CT.index = ['Consumer Transformation']

        df_FES_ST = df_FES[df_FES.index.str.contains('System Transformation', case=False)]
        df_FES_ST = df_FES_ST.append(df_FES_ST.sum(numeric_only=True), ignore_index=True)
        df_FES_ST.drop([0, 1], inplace=True)
        df_FES_ST.index = ['System Transformation']

        df_FES_SP = df_FES[df_FES.index.str.contains('Steady Progression', case=False)]
        df_FES_SP = df_FES_SP.append(df_FES_SP.sum(numeric_only=True), ignore_index=True)
        df_FES_SP.drop([0, 1], inplace=True)
        df_FES_SP.index = ['Steady Progression']

        df_FES = df_FES_SP.append([df_FES_LTW, df_FES_CT, df_FES_ST])

    elif tech == 'Waste':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        # df_FES.dropna(axis='rows', inplace=True)
        df_FES = df_FES[df_FES.Type.str.contains('Waste', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [0, 1, 2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)

        df_FES_LTW = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        df_FES_LTW = df_FES_LTW.append(df_FES_LTW.sum(numeric_only=True), ignore_index=True)
        df_FES_LTW.drop([0, 1, 2, 3], inplace=True)
        df_FES_LTW.index = ['Leading the Way']

        df_FES_CT = df_FES[df_FES.index.str.contains('Consumer Transformation', case=False)]
        df_FES_CT = df_FES_CT.append(df_FES_CT.sum(numeric_only=True), ignore_index=True)
        df_FES_CT.drop([0, 1, 2, 3], inplace=True)
        df_FES_CT.index = ['Consumer Transformation']

        df_FES_ST = df_FES[df_FES.index.str.contains('System Transformation', case=False)]
        df_FES_ST = df_FES_ST.append(df_FES_ST.sum(numeric_only=True), ignore_index=True)
        df_FES_ST.drop([0, 1, 2, 3], inplace=True)
        df_FES_ST.index = ['System Transformation']

        df_FES_SP = df_FES[df_FES.index.str.contains('Steady Progression', case=False)]
        df_FES_SP = df_FES_SP.append(df_FES_SP.sum(numeric_only=True), ignore_index=True)
        df_FES_SP.drop([0, 1, 2, 3], inplace=True)
        df_FES_SP.index = ['Steady Progression']

        df_FES = df_FES_SP.append([df_FES_LTW, df_FES_CT, df_FES_ST])

    elif tech == 'CCS Gas':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        df_FES = df_FES[df_FES.SubType.str.contains('CCS Gas', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [0, 1, 2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)
        df_LTW = pd.DataFrame(0, columns=df_FES.columns, index=['Leading the Way'])
        df_CT = pd.DataFrame(0, columns=df_FES.columns, index=['Consumer Transformation'])
        df_FES = df_FES.append([df_LTW, df_CT], sort=True)

    elif tech == 'CCS Biomass':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        df_FES = df_FES[df_FES.SubType.str.contains('CCS Biomass', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [0, 1, 2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)
        df_FES_SP = pd.DataFrame(0, columns=df_FES.columns, index=['Steady Progression'])
        df_FES = df_FES.append(df_FES_SP, sort=True)

    elif tech == 'Hydrogen':
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ES1', header=9, index_col=1)
        # df_FES.dropna(axis='rows', inplace=True)
        df_FES = df_FES[df_FES.SubType.str.contains('Hydrogen', case=False, na=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains('Generation')]
        cols = [0, 1, 2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)

        df_FES_LTW = df_FES[df_FES.index.str.contains('Leading The Way', case=False)]
        df_FES_LTW.index = ['Leading the Way']

        df_FES_CT = df_FES[df_FES.index.str.contains('Consumer Transformation', case=False)]
        df_FES_CT = df_FES_CT.append(df_FES_CT.sum(numeric_only=True), ignore_index=True)
        df_FES_CT.drop([0, 1], inplace=True)
        df_FES_CT.index = ['Consumer Transformation']

        df_FES_ST = df_FES[df_FES.index.str.contains('System Transformation', case=False)]
        df_FES_ST = df_FES_ST.append(df_FES_ST.sum(numeric_only=True), ignore_index=True)
        df_FES_ST.drop([0, 1], inplace=True)
        df_FES_ST.index = ['System Transformation']

        df_FES_SP = pd.DataFrame(0, columns=df_FES.columns, index=['Steady Progression'])

        df_FES = df_FES_SP.append([df_FES_LTW, df_FES_CT, df_FES_ST])

    elif tech == 'Marine':
        pass

    date = str(year) + '-01-01'

    if scenario == 'Leading The Way':
        try:
            scenario = 'Leading the Way'
            tech_cap_FES = float(df_FES.loc[scenario, date]) / 1000.
        except:
            scenario = 'Leading The Way'
            tech_cap_FES = float(df_FES.loc[scenario, date]) / 1000.

    else:
        tech_cap_FES = float(df_FES.loc[scenario, date]) / 1000.

    capacity_dict = {'tech_cap_year': tech_cap_year,
                     'tech_cap_FES': tech_cap_FES}

    return capacity_dict


def write_generators_p_max_pu(start, end, freq, year, year_baseline=None, scenario=None):
    """writes the generators p_max_pu csv file

    writes the timeseries maximum power output file for the
    non-dispatchable renewable generators

    Parameters
    ----------
    start : str
        start of simulation
    end : str
        end of simulation
    freq : str
        frequency of timestep, only 'H' or '0.5H' allowed currently
    Returns
    -------
    """

    # GENERATORS-P_MAX_PU FILE

    # WIND OFFSHORE

    # file = 'data/renewables/' + str(year) + '/offshore_time_series_norm.pkl'
    # df = pd.read_pickle(file)
    # fix according to operational dates
    tech = 'Wind Offshore'
    if year <= 2020:
        df_offshore = renewables.historical_RES_timeseries(year, tech, future=False)['norm']
    elif year > 2020:
        df_offshore = renewables.future_offshore_timeseries(year, year_baseline, scenario)['norm']
    df_offshore = df_offshore.loc[start:end]

    if freq == '0.5H':
        # resample to half hourly timesteps
        df_offshore = df_offshore.resample(freq).interpolate('polynomial', order=2)
        # need to add a row at end
        # the data being passed is the values of the last row
        # the tail function is used to get the last index value
        df_offshore_new = pd.DataFrame(
            data=[df_offshore.loc[df_offshore.tail(1).index.values].values[0]],
            columns=df_offshore.columns,
            index=[end])
        # add to existing dataframe
        df_offshore = df_offshore.append(df_offshore_new, sort=False)

    # name the index
    df_offshore.index.name = 'name'
    df_offshore.index = pd.to_datetime(df_offshore.index)

    # WIND ONSHORE

    # file = 'data/renewables/' + str(year) + '/onshore_time_series_norm.pkl'
    # df_onshore = pd.read_pickle(file)
    # fix according to operational dates
    tech = 'Wind Onshore'
    if year > 2020:
        df_onshore = renewables.historical_RES_timeseries(year_baseline, tech, future=True)['norm']
        # edit the years on the start and end to match the baseline year
        start = str(year_baseline) + start[4:]
        end = str(year_baseline) + end[4:]

    elif year <= 2020:
        # either overwrite optional argument or define it as equal to year
        df_onshore = renewables.historical_RES_timeseries(year, tech, future=False)['norm']

    df_onshore = df_onshore.loc[start:end]

    if freq == '0.5H':
        # resample to half hourly timesteps
        df_onshore = df_onshore.resample(freq).interpolate('polynomial', order=2)
        # need to add a row at end
        # the data being passed is the values of the last row
        # the tail function is used to get the last index value
        df_new_onshore = pd.DataFrame(
            data=[df_onshore.loc[df_onshore.tail(1).index.values].values[0]],
            columns=df_onshore.columns,
            index=[end])
        # add to existing dataframe
        df_onshore = df_onshore.append(df_new_onshore, sort=False)
    # name the index
    df_onshore.index.name = 'name'
    df_onshore.index = df_offshore.index

    # check if baseline year is a leap year and simulated year is not and remove 29th Feb
    if year_baseline is not None:
        if year_baseline % 4 == 0:
            # and the year modelled is also not a leap year
            if year % 4 != 0:
                # remove 29th Feb
                df_onshore = df_onshore[~((df_onshore.index.month == 2) & (df_onshore.index.day == 29))]

    # PV

    # file = 'data/renewables/' + str(year) + '/PV_time_series_norm.pkl'
    # df_PV = pd.read_pickle(file)
    # fix according to operational dates
    tech = 'Solar Photovoltaics'
    if year > 2020:
        df_PV = renewables.historical_RES_timeseries(year_baseline, tech, future=True)['norm']
        # edit the years on the start and end to match the baseline year
        # will change for onshore but including here to be explicit
        # incase no onshore wind
        start = str(year_baseline) + start[4:]
        end = str(year_baseline) + end[4:]

    elif year <= 2020:
        # note that use baseline year normalised distribution for RES timeseries for future
        # baseline year equated to modelled year for historical
        df_PV = renewables.historical_RES_timeseries(year, tech, future=False)['norm']

    df_PV = df_PV.loc[start:end]

    # resample to half hourly timesteps
    if freq == '0.5H':
        df_PV = df_PV.resample(freq).interpolate('polynomial', order=1)
        # need to add a row at end
        # the data being passed is the values of the last row
        # the tail function is used to get the last index value
        df_new_PV = pd.DataFrame(
            data=[df_PV.loc[df_PV.tail(1).index.values].values[0]],
            columns=df_PV.columns,
            index=[end])
        # add to existing dataframe
        df_PV = df_PV.append(df_new_PV, sort=False)
    # name the index
    df_PV.index.name = 'name'
    df_PV.index = df_offshore.index

    # check if baseline year is a leap year and simulated year is not and remove 29th Feb
    if year_baseline is not None:
        if year_baseline % 4 == 0:
            # and the year modelled is also not a leap year
            if year % 4 != 0:
                # remove 29th Feb
                df_PV = df_PV[~((df_PV.index.month == 2) & (df_PV.index.day == 29))]

    # HYDRO
    # hydro data is between 2015-02-22 and 2020-12-31
    # if dates are before then ATM use 2016 data
    tech = 'Hydro'
    if year > 2020:
        # edit the years on the start and end to match the baseline year
        # will change for onshore but including here to be explicit
        # incase no onshore wind
        start = str(year_baseline) + start[4:]
        end = str(year_baseline) + end[4:]

        df_hydro1 = renewables.read_hydro_time_series(year_baseline)['time_series_norm']

    elif year <= 2020:
        df_hydro1 = renewables.read_hydro_time_series(year)['time_series_norm']

    df_hydro = df_hydro1.loc[start:end]
    # some February values for 2015 are being overwritten here...
    # might be a better solution to this out there
    if df_hydro.empty or start[:7] == '2015-02':
        start = '2016' + start[4:]
        end = '2016' + end[4:]
        df_hydro = df_hydro1.loc[start:end]

    df_hydro = df_hydro.resample(freq).mean()
    if freq == 'H':
        df_hydro = df_hydro.resample(freq).interpolate('polynomial', order=1)
        # df_hydro = df_hydro.iloc[:-1, :]
        # # need to add a row at end
        # # the data being passed is the values of the last row
        # # the tail function is used to get the last index value
        # df_new_hydro = pd.DataFrame(
        #     data=[df_hydro.loc[df_hydro.tail(1).index.values].values[0]],
        #     columns=df_hydro.columns,
        #     index=[end])
        # # add to existing dataframe
        # df_hydro = df_hydro.append(df_new_hydro, sort=False)

    if year > 2020:
        if year_baseline % 4 == 0:
            # and the year modelled is also not a leap year
            if year % 4 != 0:
                # remove 29th Feb
                df_hydro = df_hydro[~((df_hydro.index.month == 2) & (df_hydro.index.day == 29))]
    df_hydro.index = df_offshore.index

    # MARINE TECHNOLOGIES

    # want to join the three dataframes together
    dfs = [df_offshore, df_onshore, df_PV, df_hydro]
    # if year <= 2020:
    #     dfs = unify_index(dfs, freq)
    df = pd.concat(dfs, axis=1)

    # make sure there are no missing values
    df = df.fillna(0)

    # make sure there are no negative values
    df[df < 0] = 0
    df[df > 1] = 1
    # fix the column names
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df.columns = df.columns.astype(str).str.replace(u'\xa0', '')
    df.columns = df.columns.astype(str).str.replace('ì', 'i')
    df.columns = df.columns.str.strip()

    # want to ensure no duplicate names
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    # rename the columns with the cols list.
    df.columns = cols

    df.to_csv('UC_data/generators-p_max_pu.csv', header=True)
    df.to_csv('LOPF_data/generators-p_max_pu.csv', header=True)
    # this fixes the output from this
    # df.to_csv('UC_data/generators-p_min_pu.csv', header=True)


def future_p_nom(year, time_step, scenario):
    # need to do CCS first as they are scaling based on
    # 2020 data... make sure to do this before scaling gas p_nom
    future_gas_CCS(year, scenario)
    future_biomass_CCS(year, scenario)
    future_hydrogen(year, scenario)
    renewables.scale_biomass_p_nom(year, scenario)
    future_coal_p_nom(year)
    future_gas_p_nom(year, scenario, tech='CCGT')
    future_gas_p_nom(year, scenario, tech='OCGT')
    future_nuclear_p_nom(year, scenario)
    future_oil_p_nom(year, scenario)
    future_waste_p_nom(year, scenario)
    renewables.future_RES_scale_p_nom(year, 'Wind Onshore', scenario)
    renewables.future_RES_scale_p_nom(year, 'Solar Photovoltaics', scenario)
    renewables.future_RES_scale_p_nom(year, 'Hydro', scenario)

    # ensure all generator data is added
    df_UC = pd.read_csv('UC_data/generators.csv', index_col=0)
    df_LOPF = pd.read_csv('LOPF_data/generators.csv', index_col=0)

    # run additional data for both UC and LOPF
    df_UC = generator_additional_data(df_UC, time_step)
    df_LOPF = generator_additional_data(df_LOPF, time_step)
    # remove the unit committent constraints
    df_LOPF = df_LOPF.drop(
        columns=['committable', 'min_up_time', 'min_down_time',
                 'p_min_pu', 'up_time_before', 'start_up_cost'])

    # save the dataframes to csv
    df_UC.to_csv('UC_data/generators.csv', index=True, header=True)
    df_LOPF.to_csv('LOPF_data/generators.csv', index=True, header=True)

    renewables.write_marine_generators(year, scenario)


def unmet_load():
    # ADD NEW GENERATOR FOR UNMET LOAD

    # get generators
    path = 'LOPF_data/generators.csv'
    df_LOPF = pd.read_csv(path, index_col=0)

    # # check names are unique for LOPF
    # duplicateDFRow = df_LOPF[df_LOPF.duplicated(['Unnamed: 0'], keep='first')]
    # for i in range(len(duplicateDFRow.index.values)):
    #     print(df_LOPF['Unnamed: 0'][duplicateDFRow.index.values[i]])
    #     df_LOPF.at[duplicateDFRow.index.values[i], 'Unnamed: 0'] = (
    #         df_LOPF['Unnamed: 0'][duplicateDFRow.index.values[i]] + '.1')
    #     print(df_LOPF['Unnamed: 0'][duplicateDFRow.index.values[i]])

    path_UC = 'UC_data/generators.csv'
    df_UC = pd.read_csv(path_UC, index_col=0)

    # add one to the UC problem
    dic_unmet = {'carrier': 'Unmet Load',
                 'type': 'Unmet Load', 'p_nom': 999999999,
                 'bus': 'bus', 'marginal_cost': 999999999,
                 'committable': True, 'min_up_time': 0,
                 'min_down_time': 0, 'ramp_limit_up': 1,
                 'ramp_limit_down': 1, 'p_min_pu': 0,
                 'up_time_before': 0, 'start_up_cost': 0,
                 'p_max_pu': 1}
    df_unmet = pd.DataFrame(dic_unmet, index=['Unmet Load'])
    df_UC = df_UC.append(df_unmet)
    df_UC.index.name = 'name'

    # for LOPF need to add to each bus

    # read in all buses with loads
    df_buses = pd.read_csv('LOPF_data/loads.csv', index_col=0)
    # add to each bus
    for bus in df_buses.bus.values:
        dic_unmet = {'carrier': 'Unmet Load',
                     'type': 'Unmet Load', 'p_nom': 999999999,
                     'bus': bus, 'marginal_cost': 999999999,
                     'ramp_limit_up': 1, 'ramp_limit_down': 1,
                     'p_max_pu': 1}
        index = 'Unmet Load ' + bus
        df_unmet = pd.DataFrame(dic_unmet, index=[index])
        df_LOPF = df_LOPF.append(df_unmet)

    df_LOPF.index.name = 'name'

    df_UC.to_csv('UC_data/generators.csv', header=True)
    df_LOPF.to_csv('LOPF_data/generators.csv', header=True)


    # DEFINING A NEW FUNCTION TO AMEND 'WIND OFFSHORE' TO 'FLOATING WIND' AT SITES I, E, F, G, NE8, NE7, E3, E2, NE1, E1, NE2, NE3, NE6, N2, N3 - FOR LOPF ONLY
    
# def floating_wind():
        
#     # read list of generators
#     path = 'LOPF_data/generators.csv'
#     df_FW = pd.read_csv(path, 'type')

#     # for sites I, E, F, G, NE8, NE7, E3, E2, NE1, E1, NE2, NE3, NE6, N2, N3, change the type to 'Floating Wind'
#     df_FW = d_FW.replace('I
    
    
#     # read in all buses with loads
#     df_buses = pd.read_csv('LOPF_data/loads.csv', index_col=0)
    
#     # add new type of 'Floating Wind' to each bus, under the carrier of 'Offshore Wind'
#     for bus in df_buses.bus.values:
#         dic_floating_wind = {'carrier': 'Wind Offshore',
#                      'type': 'Floating Wind', 'p_nom': 999999999,
#                      'bus': bus, 'marginal_cost': 999999999,
#                      'ramp_limit_up': 1, 'ramp_limit_down': 1,
#                      'p_max_pu': 1}
#         index = 'Floating Wind' + bus
#         df_floating_wind = pd.DataFrame(dic_floating_wind, index=[index])
#         df_LOPF = df_LOPF.append(df_unmet)

#     df_LOPF.index.name = 'name'
    
#     df_LOPF.to_csv('LOPF_data/generators.csv', header=True)
    
    
    
def merge_generation_buses(year):

    # get generators
    path = 'LOPF_data/generators.csv'
    df_gen = pd.read_csv(path, index_col=0)

    path = 'LOPF_data/generators-p_max_pu.csv'
    df_gen_p = pd.read_csv(path, index_col=0)

    carriers = ['Wind Offshore', 'Wind Onshore', 'Solar Photovoltaics', 'Large Hydro', 'Small Hydro', 'Interconnector', 'Floating Wind']
    buses = df_gen['bus'].unique()
    df_list = []
    df_gen_p_list = []

    for c in carriers:
        for b in buses:
            df_carrier_bus = df_gen.loc[(df_gen.carrier == c) & (df_gen.bus == b)]
            carrier_bus_aggregated = df_carrier_bus.p_nom.sum()
            index = [c + ' ' + b]

            # change the p-max_pu
            list_of_sites = df_gen.loc[(df_gen.carrier == c) & (df_gen.bus == b)].index
            gen_p_bus = []
            for gen in list_of_sites:

                try:
                    gen_p_bus.append(df_gen_p.loc[:, gen] * df_carrier_bus.loc[gen, 'p_nom'])
                except KeyError:
                    pass
            try:
                df_gen_p_new = pd.concat(gen_p_bus, axis=1)
                df_gen_p_new['sum'] = df_gen_p_new.sum(axis=1)
                df_gen_p_new[c + ' ' + b] = df_gen_p_new['sum'] / carrier_bus_aggregated
                df_gen_p_list.append(df_gen_p_new[c + ' ' + b])
            except ValueError:
                pass

            # change generators
            if carrier_bus_aggregated > 0:
                df_carrier_bus = pd.DataFrame([df_carrier_bus.iloc[-1, :]], index=index)
                df_carrier_bus.p_nom = carrier_bus_aggregated
                df_list.append(df_carrier_bus)

        df_gen = df_gen[~df_gen.carrier.str.contains(c)]

    df_gen_p = pd.concat(df_gen_p_list, axis=1)
    df_gen_p = df_gen_p.fillna(0)
    # just to ensure no negative values
    df_gen_p[df_gen_p < 0] = 0
    # add in interconnectors p_max_pu

    df_gen_res = pd.concat(df_list)
    # add in new generators
    df_gen = df_gen.append(df_gen_res)
       
    df_gen.to_csv('LOPF_data/generators.csv', header=True)
    df_gen_p.to_csv('LOPF_data/generators-p_max_pu.csv', header=True)
    if year < 2021:
        # fix interconnectors
        # inter_cols = [col for col in df_gen_p.columns if 'Interconnector' in col]
        # print(inter_cols)
        # df_interconnectors = df_gen_p[[inter_cols]]
        df_interconnectors = df_gen_p.filter(regex='Interconnector')
        df_interconnectors.to_csv('LOPF_data/generators-p_min_pu.csv', header=True)

    if year >= 2021:
        # check if generators-p_min_pu exists and delete if so
        # used in historical simulations but not wanted in future sims
        try:
            file = 'LOPF_data/generators-p_min_pu.csv'
            os.remove(file)
        except Exception:
            pass
        try:
            file = 'UC_data/generators-p_min_pu.csv'
            os.remove(file)
        except Exception:
            pass


if __name__ == "__main__":
    year = 2050
    # future_coal_p_nom(year)
    # tech = 'Gas'
    # future_capacity(year, tech)
    # tech = 'CCGT'
    # tech = 'OCGT'
    # future_gas_p_nom(year, tech)
    # future_nuclear_p_nom(year)
    # future_oil_p_nom(year)
    # future_waste_p_nom(year)
    # future_gas_CCS(year)
    # future_biomass_CCS(year)
    # future_hydrogen(year)
    merge_generation_buses()

    
  
    
    