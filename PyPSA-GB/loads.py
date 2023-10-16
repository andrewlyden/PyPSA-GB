from logging import raiseExceptions
import pandas as pd
import distance_calculator as dc


def read_historical_demand_data():
    """reads the historical demand data from ESPENI database

    Parameters
    ----------

    Returns
    -------
    dataframe
        ESPENI demand data, see dates below
    """

    # this stuff can be used to get data from national grid csv files
    # file1 = 'data/demand/DemandData_2005-2010.csv'
    # file2 = 'data/demand/DemandData_2011-2016.csv'
    # file3 = 'data/demand/DemandData_2017.csv'
    # file4 = 'data/demand/DemandData_2018.csv'

    # # reads csvs
    # df1 = pd.read_csv(file1)
    # df2 = pd.read_csv(file2)
    # df3 = pd.read_csv(file3)
    # df4 = pd.read_csv(file4)

    # frames = [df2, df3, df4]
    # df = df1.append(frames, ignore_index=True, sort=False)

    # using espeni data set
    file = '../data/demand/espeni.csv'
    df = pd.read_csv(file)

    dti = pd.date_range(
        start='2008-11-05 22:00:00', end='2021-06-06 23:30:00', freq='0.5H')
    df = df.set_index(dti)
    df = df[['POWER_ESPENI_MW']]
    # df = df.drop(columns=['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD'])
    return df


def read_future_profile_data():
    """reads the future demand profile data from Staffel et al - https://doi.org/10.1016/j.energy.2015.06.082

    Parameters
    ----------

    Returns
    -------
    dataframe
        eload and DESSTINEE demand data, see dates below
    """

    # this stuff can be used to get data from national grid csv files
    # file1 = 'data/demand/DemandData_2005-2010.csv'
    # file2 = 'data/demand/DemandData_2011-2016.csv'
    # file3 = 'data/demand/DemandData_2017.csv'
    # file4 = 'data/demand/DemandData_2018.csv'

    # # reads csvs
    # df1 = pd.read_csv(file1)
    # df2 = pd.read_csv(file2)
    # df3 = pd.read_csv(file3)
    # df4 = pd.read_csv(file4)

    # frames = [df2, df3, df4]
    # df = df1.append(frames, ignore_index=True, sort=False)

    # using espeni data set
    file = '..\data\demand\egy_7649_mmc1.xlsx'
    df = pd.read_excel(file, sheet_name=None)
    df_eload = df['ELOAD'].drop([0, 1, 2, 3, 4, 5])[['eLOAD Model (2050).2']].reset_index(drop=True) * 1000
    df_eload.rename(columns={'eLOAD Model (2050).2': 'eLOAD'}, inplace=True)
    dti = pd.date_range(
        start='2050-01-01 00:00:00', end='2050-12-31 23:00:00', freq='H')
    df_eload = df_eload.set_index(dti)
    # resample to half hour frequency
    df_eload['eLOAD'] = pd.to_numeric(df_eload['eLOAD'])
    df_eload = df_eload.resample('0.5H').interpolate('polynomial', order=2)

    # add end value
    df_new_eload = pd.DataFrame(
        data=df_eload.tail(1).values,
        columns=df_eload.columns,
        index = pd.date_range(start='2050-12-31 23:30:00', end='2050-12-31 23:30:00', freq='0.5H'))
    # add to existing dataframe
    df_eload = pd.concat([df_eload, df_new_eload], sort=False)

    return df_eload


def write_loads(year, networkmodel='Reduced'):
    """writes the loads csv file

    Parameters
    ----------
    Returns
    -------
    """
    # LOADS CSV FILE
    # first off the loads is just at a single bus
    data = {'name': 'load', 'bus': 'bus'}
    df = pd.DataFrame(data=data, index=[0])
    df.to_csv('UC_data/loads.csv', index=False, header=True)

    df_buses = pd.read_csv('LOPF_data/buses.csv')
    df_buses = df_buses.drop(columns=['v_nom', 'carrier', 'x', 'y'])
    df_buses['bus'] = df_buses['name']
    df_buses = df_buses.set_index('name')
    if year > 2020:
        # delete the interconnectors if future years
        try:
            df_buses = df_buses.drop(['Belgium', 'France1', 'France2',
                                    'Netherlands', 'Ireland', 'N. Ireland'])
        except:
            pass
    # delete the IC loads
    df_buses.to_csv('LOPF_data/loads.csv', index=True, header=True)


def write_loads_p_set(start, end, year, time_step, dataset, year_baseline=None, scenario=None, FES=None, scale_to_peak=False, networkmodel='Reduced'):
    """writes the loads power timeseries csv file

    Parameters
    ----------
    start : str
        start of simulation period
    end : str
        end of simulation period
    dataset : str
        can be 'historical' or 'eload'
    Returns
    -------
    """
    # LOADS-P_SET CSV FILE
    if dataset == 'historical':
        df_hd = read_historical_demand_data()
        df_hd.rename(columns={'POWER_ESPENI_MW': 'load'}, inplace=True)
    elif dataset == 'eload':
        df_hd = read_future_profile_data()
        df_hd.rename(columns={'eLOAD': 'load'}, inplace=True)
        # if modelled year is a leap year need to add in 29th feb (copy 28th Feb)
        if year % 4 == 0:
            df_hd.index = pd.to_datetime({
                'year': 2040,
                'month': df_hd.index.month,
                'day': df_hd.index.day,
                'hour': df_hd.index.hour,
                'minute': df_hd.index.minute,
                'second': df_hd.index.second})
            df_leap_day = df_hd[pd.to_datetime('2040-02-28 00:00:00'):pd.to_datetime('2040-02-28 23:30:00')]
            df_leap_day.index = pd.to_datetime({'year': df_leap_day.index.year,
                                                'month': df_leap_day.index.month,
                                                'day': 29,
                                                'hour': df_leap_day.index.hour,
                                                'minute': df_leap_day.index.minute,
                                                'second': df_leap_day.index.second})
            df_hd_appended = pd.concat([df_hd, df_leap_day])
            df_hd = df_hd_appended.sort_index()

    # need an index for the period to be simulated
    if time_step == 0.5:
        freq = '0.5H'
    elif time_step == 1.0:
        freq = 'H'
    else:
        raise Exception("Time step not recognised")

    dti = pd.date_range(
        start=start,
        end=end,
        freq=freq)

    df_distribution = pd.read_csv(
        '../data/demand/Demand_Distribution.csv', index_col=0)
    df_distribution = df_distribution.loc[:, ~df_distribution.columns.str.contains('^Unnamed')]
    df_distribution.dropna(inplace=True)

    if networkmodel == 'Reduced':
        df_distribution = pd.read_csv(
            '../data/demand/Demand_Distribution.csv', index_col=0)
        df_distribution = df_distribution.loc[:, ~df_distribution.columns.str.contains('^Unnamed')]
        df_distribution.dropna(inplace=True)
    elif networkmodel == 'Zonal':
        df_distribution = pd.read_csv(
            '../data/demand/Demand_Distribution_Zonal.csv', index_col=0)

    # for historical years using the 2020 FES distribution data
    if year < 2020:
        year_dist = 2020
    elif year >= 2020:
        year_dist = year

    # normalise the data so distribution can be applied
    data = df_distribution[[str(year_dist)]].T
    normalised = data.values / data.values.sum()
    norm = pd.DataFrame(data=normalised, columns=data.columns)
    norm.index.name = 'name'

    if year <= 2020 and dataset == 'historical':
        # if using historical data then use this to
        # distribute to different nodes
        # locate from historical data load for simulated period
        df = df_hd.loc[start:end]
        df_loads_p_set_UC = df
        # create empty dataframe to populate
        df_loads_p_set_LOPF = pd.DataFrame(index=df.index)
        for j in norm.columns:
            # scale load for each node/bus
            df_loads_p_set_LOPF[j] = df['load'] * norm[j].values

        if time_step == 0.5:
            df_loads_p_set_LOPF.index = dti
        elif time_step == 1:
            df_loads_p_set_LOPF = df_loads_p_set_LOPF.resample(freq).mean()
        if time_step == 0.5:
            df_loads_p_set_UC.index = dti
        elif time_step == 1:
            df_loads_p_set_UC = df_loads_p_set_UC.resample(freq).mean()

    elif year > 2020:
        # if future scenarios need to scale historical
        # data using FES demand data
        if scenario == 'Leading The Way':
            scenario = 'Leading the Way'
        if FES == 2021:
            df_FES = pd.read_excel(
                '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
                sheet_name='ED1', header=4, dtype=str)
        elif FES == 2022:
            df_FES = pd.read_excel(
                '../data/FES2022/FES2022 Workbook V4.xlsx',
                sheet_name='ED1', header=4, dtype=str)
        elif FES == None:
            raise Exception("Please choose a FES year.")
            
        df_FES_demand = df_FES.loc[df_FES['Data item'] == 'GBFES System Demand: Total']
        df_FES_demand = df_FES_demand.loc[df_FES_demand['Scenario'] == scenario]
        date = str(year) + '-01-01 00:00:00'
        df_FES_demand.columns = df_FES_demand.columns.astype(str)
        # future demand in GWh/yr
        future_demand = float(df_FES_demand[date].values[0])

        # scale historical demand using baseline year
        year_start = str(year_baseline) + '-01-01 00:00:00'
        year_end = str(year_baseline) + '-12-31 23:30:00'
        # using the baseline year as basis for demand distribution

        if dataset == 'historical':
            # load timeseries in baseline year
            df_year = df_hd.loc[year_start:year_end]
        elif dataset == 'eload':
            df_year = df_hd

        # summed load in baseline year
        # factor of two because historical demand is read in half hourly, so need to convert to GWh/yr
        historical_demand = df_year.sum().values[0] * 0.5 / 1000
        scale_factor = float(future_demand) / float(historical_demand)

        # load for simulation dates in baseline year
        if dataset == 'historical':
            yr = str(year_baseline)[-2:]
            start_yr = start[:2] + yr + start[4:]
            end_yr = end[:2] + yr + end[4:]
            df_sim = df_year[start_yr:end_yr]
        elif dataset == 'eload':
            if year % 4 == 0:
                yr = '40'
            else:
                yr = '50'
            start_yr = start[:2] + yr + start[4:]
            end_yr = end[:2] + yr + end[4:]
            df_sim = df_year[start_yr:end_yr]
        scaled_load = scale_factor * df_sim

        # check if baseline year is a leap year and simulated year is not and remove 29th Feb
        if year_baseline % 4 == 0:
            # and the year modelled is also not a leap year
            if year % 4 != 0:
                # remove 29th Feb
                scaled_load = scaled_load[~((scaled_load.index.month == 2) & (scaled_load.index.day == 29))]

        if time_step == 0.5:
            scaled_load.index = dti
        elif time_step == 1:
            scaled_load = scaled_load.resample(freq).mean()
            # for some reason need to get rid of this date again?
            # probably due to resampling step...
            if year_baseline % 4 == 0:
                # and the year modelled is also not a leap year
                if year % 4 != 0:
                    # remove 29th Feb
                    scaled_load = scaled_load[~((scaled_load.index.month == 2) & (scaled_load.index.day == 29))]
            scaled_load.index = dti

        # can use this for UC
        df_loads_p_set_UC = scaled_load.copy()

        # need to distribute for the LOPF
        df_loads_p_set_LOPF = pd.DataFrame(index=dti)
        for j in norm.columns:
            df_loads_p_set_LOPF[j] = scaled_load * norm[j].values
    
    if FES == 2022 and scale_to_peak == True:
        # if FES is 22 then going to scale again using the peak demand from regional breakdown
        df_year_LOPF = pd.DataFrame()
        for j in norm.columns:
            df_year_LOPF[j] = df_year * scale_factor * norm[j].values
        peak_bus_regional = read_regional_breakdown_load(scenario, year, networkmodel)
        for bus in df_year_LOPF.columns:
            scaling_factor = peak_bus_regional[bus] / (df_year_LOPF[bus].max())
            df_loads_p_set_LOPF[bus] *= scaling_factor

    df_loads_p_set_UC.index.name = 'name'
    df_loads_p_set_LOPF.index.name = 'name'

    # if time_step == 0.5:

    #     appendix = df_loads_p_set_LOPF.iloc[-1:]
    #     new_index = df_loads_p_set_LOPF.index[-1] + pd.Timedelta(minutes=30)
    #     appendix.rename(index={appendix.index[0]: new_index}, inplace=True)
    #     df_loads_p_set_LOPF = df_loads_p_set_LOPF.append(appendix)

    #     appendix = df_loads_p_set_UC.iloc[-1:]
    #     new_index = df_loads_p_set_LOPF.index[-1] + pd.Timedelta(minutes=30)
    #     appendix.rename(index={appendix.index[0]: new_index}, inplace=True)
    #     df_loads_p_set_UC = df_loads_p_set_UC.append(appendix)

    df_loads_p_set_LOPF.to_csv('LOPF_data/loads-p_set.csv', header=True)
    df_loads_p_set_UC.to_csv('UC_data/loads-p_set.csv', header=True)

    return df_loads_p_set_LOPF

def read_regional_breakdown_load(scenario, year, networkmodel):

    if scenario == 'Leading the Way':
        df_regional = pd.read_excel('../data/FES2022/FES22_regional_peak_load_leading_the_way.xlsx', sheet_name=str(year), header=5)
    elif scenario == 'Consumer Transformation':
        df_regional = pd.read_excel('../data/FES2022/FES22_regional_peak_load_consumer_transformation.xlsx', sheet_name=str(year), header=5)
    elif scenario == 'System Transformation':
        df_regional = pd.read_excel('../data/FES2022/FES22_regional_peak_load_system_transformation.xlsx', sheet_name=str(year), header=5)
    elif scenario == 'Falling Short':
        df_regional = pd.read_excel('../data/FES2022/FES22_regional_peak_load_falling_short.xlsx', sheet_name=str(year), header=5)
    # remove first row
    df_regional = df_regional.iloc[1: , :].set_index('Name')
    # delete all duplicates in index (removes electrolysis stuff which is zero for winter peak anyway)
    df_regional = df_regional[~df_regional.index.duplicated(keep=False)]
    # print(df_regional['P(Gross'])
    df_regional = df_regional[['P(Gross)']]
    df_regional = df_regional.loc[~(df_regional==0).all(axis=1)]

    df_gsp_data = pd.read_csv('../data/FES2022/GSP_data.csv', encoding='cp1252', index_col=3)
    df_gsp_data = df_gsp_data[['Latitude', 'Longitude']]
    df_gsp_data.rename(columns={'Latitude': 'y', 'Longitude': 'x'}, inplace=True)

    if networkmodel == 'Reduced':
        from distance_calculator import map_to_bus as map_to
    elif networkmodel == 'Zonal':
        from allocate_to_zone import map_to_zone as map_to

    df_gsp_data['Bus'] = map_to(df_gsp_data)
    # now
    GSP_to_bus = []
    for GSP in df_regional.index:
        GSP_to_bus.append(df_gsp_data['Bus'][GSP])

    df_regional['Bus'] = GSP_to_bus
    # list of buses
    peak_bus = {}
    for bus in df_regional['Bus'].unique():
        peak_bus[bus] = df_regional.loc[df_regional['Bus'] == bus]['P(Gross)'].sum()
    
    return peak_bus

def distribution_zonal_loads():
    scenario = 'Leading the Way'
    data = {}
    for year in range(2021, 2051, 1):
        dic = read_regional_breakdown_load(scenario, year, networkmodel='Zonal')
        data[year] = pd.Series(data=dic.values(), index=dic.keys(), name=year)
    df = pd.DataFrame(data)
    df.to_csv('../data/demand/Demand_Distribution_Zonal.csv', header=True)

if __name__ == '__main__':
    read_future_profile_data()
    # read_historical_demand_data()
    # read_regional_breakdown_load()
    # distribution_zonal_loads()
