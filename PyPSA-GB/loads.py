import pandas as pd


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


def write_loads(year):
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
        df_buses = df_buses.drop(['Belgium', 'France1', 'France2',
                                  'Netherlands', 'Ireland', 'N. Ireland'])

    # delete the IC loads
    df_buses.to_csv('LOPF_data/loads.csv', index=True, header=True)


def write_loads_p_set(start, end, year, time_step, year_baseline=None):
    """writes the loads power timeseries csv file

    Parameters
    ----------
    start : str
        start of simulation period
    end : str
        end of simulation period
    Returns
    -------
    """
    # LOADS-P_SET CSV FILE
    df_hd = read_historical_demand_data()
    df_hd.rename(columns={'POWER_ESPENI_MW': 'load'}, inplace=True)

    # need an index for the period to be simulated
    frequency = str(time_step) + 'H'
    dti = pd.date_range(start=start, end=end, freq=frequency)
    df_distribution = pd.read_csv(
        '../data/demand/Demand_Distribution.csv', index_col=0)
    df_distribution = df_distribution.loc[:, ~df_distribution.columns.str.contains('^Unnamed')]
    df_distribution.dropna(inplace=True)

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

    if year <= 2020:
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

    elif year > 2020:
        # if future scenarios need to scale historical
        # data using FES demand data
        df_FES = pd.read_excel(
            '../data/FES2021/FES 2021 Data Workbook V04.xlsx',
            sheet_name='ED1', header=4, dtype=str)
        df_FES_demand = df_FES.loc[df_FES['Data item'] == 'GBFES System Demand: Total']
        scenario = 'Leading the Way'
        df_FES_demand = df_FES_demand.loc[df_FES_demand['Scenario'] == scenario]
        date = str(year) + '-01-01 00:00:00'
        df_FES_demand.columns = df_FES_demand.columns.astype(str)
        # future demand in GWh/yr
        future_demand = df_FES_demand[date].values[0]

        # scale historical demand using baseline year
        year_start = str(year_baseline) + '-01-01 00:00:00'
        year_end = str(year_baseline) + '-12-31 23:30:00'
        # using the baseline year as basis for demand distribution
        # load timeseries in baseline year
        df_year = df_hd.loc[year_start:year_end]
        # summed load in baseline year
        historical_demand = df_year.sum().values[0] * time_step / 1000
        scale_factor = float(future_demand) / float(historical_demand)

        # load for simulation dates in baseline year
        yr = str(year_baseline)[-2:]
        start_yr = start[:2] + yr + start[4:]
        end_yr = end[:2] + yr + end[4:]
        df_sim = df_year[start_yr:end_yr]
        scaled_load = scale_factor * df_sim

        try:
            scaled_load.index = dti
        except ValueError:
            scaled_load = scaled_load.resample(frequency).mean()
            scaled_load.index = dti

        # can use this for UC
        df_loads_p_set_UC = scaled_load.copy()

        # need to distribute for the LOPF
        df_loads_p_set_LOPF = pd.DataFrame(index=dti)
        for j in norm.columns:
            df_loads_p_set_LOPF[j] = scaled_load * norm[j].values

    df_loads_p_set_UC.index.name = 'name'
    df_loads_p_set_LOPF.index.name = 'name'

    if year > 2020 and time_step == 1.:

        appendix = df_loads_p_set_LOPF.iloc[-1:]
        new_index = df_loads_p_set_LOPF.index[-1] + pd.Timedelta(minutes=30)
        appendix.rename(index={appendix.index[0]: new_index}, inplace=True)
        df_loads_p_set_LOPF = df_loads_p_set_LOPF.append(appendix)

        appendix = df_loads_p_set_UC.iloc[-1:]
        new_index = df_loads_p_set_LOPF.index[-1] + pd.Timedelta(minutes=30)
        appendix.rename(index={appendix.index[0]: new_index}, inplace=True)
        df_loads_p_set_UC = df_loads_p_set_UC.append(appendix)

    df_loads_p_set_LOPF.to_csv('LOPF_data/loads-p_set.csv', header=True)
    df_loads_p_set_UC.to_csv('UC_data/loads-p_set.csv', header=True)

    return df_loads_p_set_LOPF
