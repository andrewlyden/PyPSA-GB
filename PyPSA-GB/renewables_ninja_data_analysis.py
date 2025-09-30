"""renewables time series script

contains functions for interacting with renewables ninja API
and fixing these. Also contains functions for introducing correction
factors based upon reported historical renewable generation. This has
scope for improvement as it currently involves a linear correction factor
which is flat across the timeseries and locations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

from . import renewables
from . import renewables_ninja_API


def fix_PV():
    """for joining time series pickle files together"""

    # # # read in the PV files
    # df = pickle.load(open("data/renewables/2016/onshore_time_series.pkl", "rb"))
    # print(df)
    # df2 = pickle.load(open("data/renewables/2016/onshore_time_series2.pkl", "rb"))
    # print(df2)
    # # # df3 = pickle.load(open("data/renewables/2018/PV_time_series_norm3.pkl", "rb"))
    # # # print(df2)
    # # # join the two dataframes
    # result = pd.concat([df, df2], axis=1)
    # print(result)

    # year = 2019
    # # read in the renewables dataframe
    # df = data_reader_writer.REPD_date_corrected(year)
    # df_pv = df.loc[df['Technology Type'] == 'Solar Photovoltaics'].reset_index(drop=True)
    # print(df_pv)

    # z = result.columns.to_list()
    # item = df_pv['Site Name'].to_list()
    # not_in_list = [x for x in item if x not in z]
    # print(not_in_list)
    # print(len(not_in_list))
    # result = result.reset_index().T.drop_duplicates().T
    # print(result)

    meta = pickle.load(open("data/renewables/2016/onshore_metadata.pkl", "rb"))
    print(len(meta))
    meta2 = pickle.load(open("data/renewables/2016/onshore_metadata2.pkl", "rb"))
    print(len(meta2))

    meta.update(meta2)
    print(len(meta))

    # save the new dataframe
    # result.to_pickle("data/renewables/2016/onshore_time_series_norm.pkl")
    # result.to_pickle("data/renewables/2016/onshore_time_series.pkl")
    with open("data/renewables/2016/onshore_metadata.pkl", "wb") as handle:
        pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)


def offshore_wind_time_series(df, year, date_from, date_to):
    """offshore wind time series download from renewables ninja

     writes the timeseries to pickle file in relevant folder
     note that only single year can be accessed

    Parameters
     ----------
     df : dataframe
         REPD data, should use date corrected version
     year: int/str
         Year of simulation
     date_from: str
         time series date from
     date_to: str
         time series date to

     Returns
     -------
    """

    # aim is to extract power series for
    # offshore wind farms from renewables.ninja
    df_offshore_wind = df.loc[df["Technology Type"] == "Wind Offshore"]
    df_offshore_wind = df_offshore_wind.reset_index()

    # check that data is available
    if df_offshore_wind["No. of Turbines"].isnull().values.any() == True:
        raise Exception(
            "Number of turbines values must be provided, please fill in missing data in csv file"
        )
    if df_offshore_wind["Turbine Capacity (MW)"].isnull().values.any() == True:
        raise Exception(
            "Number of turbines values must be provided, please fill in missing data in csv file"
        )

    output_series = []
    output_series_normalised = []
    output_series_metadata = {}

    length = df_offshore_wind.index
    # length = range(3)
    print(length, " number of requests.")
    for i in range(len(length)):

        print(i)
        print(df_offshore_wind["Site Name"][i])

        # multiplied by 1000 to convert to kW
        capacity = df_offshore_wind["Turbine Capacity (MW)"][i] * 1000

        # check if height given
        if df_offshore_wind["Height of Turbines (m)"][i] == "NaN":
            height = df_offshore_wind["Height of Turbines (m)"][i]
        else:
            # need to estimate height from turbine capacity using regression
            height = wind_hub_height_to_capacity_regression(capacity)

        # height is limited to 150m
        height = min(height, 150)

        # need to figure out which turbine to use for power curve
        # vestas has a wide range, let just use their ones
        turbine = turbine_from_capacity(capacity)

        # location parameters
        lat = df_offshore_wind["lat"][i]
        lon = df_offshore_wind["lon"][i]

        print(lat, lon, date_from, date_to, capacity, height, turbine)

        # request data from renewables ninja
        wind = renewables_ninja_API.request_wind(
            token, lat, lon, date_from, date_to, capacity, height, turbine
        )

        wind["data"] = wind["data"].rename(
            columns={"electricity": df_offshore_wind["Site Name"][i]}
        )
        # multiply by number of wind turbines
        wind["data"] = wind["data"] * df_offshore_wind["No. of Turbines"][i]
        wind["normalised"] = wind["data"] / (
            capacity * df_offshore_wind["No. of Turbines"][i]
        )
        output_series.append(wind["data"])
        output_series_normalised.append(wind["normalised"])
        output_series_metadata[df_offshore_wind["Site Name"][i]] = wind["metadata"]

        # gathers all the time series into one dataframe
        df_out = pd.concat(output_series, axis=1)
        df_out_norm = pd.concat(output_series_normalised, axis=1)

        # want to save these into pickle files
        df_out.to_pickle("data/renewables/" + str(year) + "/offshore_time_series.pkl")
        df_out_norm.to_pickle(
            "data/renewables/" + str(year) + "/offshore_time_series_norm.pkl"
        )
        with open(
            "data/renewables/" + str(year) + "/offshore_metadata.pkl", "wb"
        ) as handle:
            pickle.dump(
                output_series_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL
            )


def onshore_wind_time_series(df, year, date_from, date_to):
    """onshore wind time series download from renewables ninja

     writes the timeseries to pickle file in relevant folder
     note that only single year can be accessed

    Parameters
     ----------
     df : dataframe
         REPD data, should use date corrected version
     year: int/str
         Year of simulation
     date_from: str
         time series date from
     date_to: str
         time series date to

     Returns
     -------
    """

    # aim is to extract power series for
    # offshore wind farms from renewables.ninja
    df_onshore_wind = df.loc[df["Technology Type"] == "Wind Onshore"]
    df_onshore_wind = df_onshore_wind.reset_index()

    # check that data is available
    if df_onshore_wind["No. of Turbines"].isnull().values.any() == True:
        raise Exception(
            "Number of turbines values must be provided, please fill in missing data in csv file"
        )
    if df_onshore_wind["Turbine Capacity (MW)"].isnull().values.any() == True:
        raise Exception(
            "Number of turbines values must be provided, please fill in missing data in csv file"
        )

    output_series = []
    output_series_normalised = []
    output_series_metadata = {}

    length = df_onshore_wind.index
    # length = range(3)
    print(length, " number of requests.")
    for i in range(len(length)):

        print(i)
        print(df_onshore_wind["Site Name"][i])

        # multiplied by 1000 to convert to kW
        capacity = df_onshore_wind["Turbine Capacity (MW)"][i] * 1000

        # check if height given
        if df_onshore_wind["Height of Turbines (m)"][i] == "NaN":
            height = df_onshore_wind["Height of Turbines (m)"][i]
        else:
            # need to estimate height from turbine capacity using regression
            height = wind_hub_height_to_capacity_regression(capacity)

        # height is limited to 150m
        height = min(height, 150)

        # need to figure out which turbine to use for power curve
        # vestas has a wide range, let just use their ones
        turbine = turbine_from_capacity(capacity)

        # location parameters
        lat = df_onshore_wind["lat"][i]
        lon = df_onshore_wind["lon"][i]

        print(lat, lon, date_from, date_to, capacity, height, turbine)

        # request data from renewables ninja
        wind = renewables_ninja_API.request_wind(
            token, lat, lon, date_from, date_to, capacity, height, turbine
        )

        wind["data"] = wind["data"].rename(
            columns={"electricity": df_onshore_wind["Site Name"][i]}
        )
        # multiply by number of wind turbines
        wind["data"] = wind["data"] * df_onshore_wind["No. of Turbines"][i]
        wind["normalised"] = wind["data"] / (
            capacity * df_onshore_wind["No. of Turbines"][i]
        )
        output_series.append(wind["data"])
        output_series_normalised.append(wind["normalised"])
        output_series_metadata[df_onshore_wind["Site Name"][i]] = wind["metadata"]

        # save for each timestep due to loss of connection and other errors

        # gathers all the time series into one dataframe
        df_out = pd.concat(output_series, axis=1)
        df_out_norm = pd.concat(output_series_normalised, axis=1)

        # want to save these into pickle files
        df_out.to_pickle("data/renewables/" + str(year) + "/onshore_time_series.pkl")
        df_out_norm.to_pickle(
            "data/renewables/" + str(year) + "/onshore_time_series_norm.pkl"
        )
        with open(
            "data/renewables/" + str(year) + "/onshore_metadata.pkl", "wb"
        ) as handle:
            pickle.dump(
                output_series_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL
            )


def PV_time_series(df, year, date_from, date_to):
    """PV time series download from renewables ninja

     writes the timeseries to pickle file in relevant folder
     note that only single year can be accessed

    Parameters
     ----------
     df : dataframe
         REPD data, should use date corrected version
     year: int/str
         Year of simulation
     date_from: str
         time series date from
     date_to: str
         time series date to

     Returns
     -------
    """
    # aim is to extract power series for
    # PV from renewables.ninja
    df_PV = df.loc[df["Technology Type"] == "Solar Photovoltaics"]
    df_PV = df_PV.reset_index()

    output_series = []
    output_series_normalised = []
    output_series_metadata = {}

    length = df_PV.index
    # length = range(3)
    print(length)
    for i in range(len(length)):

        print(i)
        print(df_PV["Site Name"][i])

        # multiplied by 1000 to convert to kW
        capacity = df_PV["Installed Capacity (MWelec)"][i] * 1000
        system_loss = 0.1
        tracking = 0
        tilt = 35
        azim = 180

        # location parameters
        lat = df_PV["lat"][i]
        lon = df_PV["lon"][i]

        print(lat, lon, date_from, date_to, capacity, system_loss, tracking, tilt, azim)

        # request data from renewables ninja
        PV = renewables_ninja_API.request_PV(
            token,
            lat,
            lon,
            date_from,
            date_to,
            capacity,
            system_loss,
            tracking,
            tilt,
            azim,
        )

        PV["data"] = PV["data"].rename(columns={"electricity": df_PV["Site Name"][i]})
        PV["normalised"] = PV["data"] / capacity
        output_series.append(PV["data"])
        output_series_normalised.append(PV["normalised"])
        output_series_metadata[df_PV["Site Name"][i]] = PV["metadata"]

        # save for each timestep due to loss of connection and other errors

        # gathers all the time series into one dataframe
        df_out = pd.concat(output_series, axis=1)
        df_out_norm = pd.concat(output_series_normalised, axis=1)

        # want to save these into pickle files
        df_out.to_pickle("data/renewables/" + str(year) + "/PV_time_series.pkl")
        df_out_norm.to_pickle(
            "data/renewables/" + str(year) + "/PV_time_series_norm.pkl"
        )
        with open("data/renewables/" + str(year) + "/PV_metadata.pkl", "wb") as handle:
            pickle.dump(
                output_series_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL
            )


def wind_hub_height_to_capacity_regression(capacity):
    """guesses hub height from capacity

    Parameters
     ----------
     capacity : int/float
         capacity of wind turbine

     Returns
     -------
     hub_height : int
         predicted hub height of wind turbine
    """
    file = "data/renewables/wind_regression.csv"
    df = pd.read_csv(file)
    # values converts it into a numpy array
    X = df["Rated Capacity"].values.reshape(-1, 1)
    # -1 means that calculate the dimension of rows, but have 1 column
    Y = df["Hub Height"].values.reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression

    # get value and reshape into single value and convert to int
    return int(linear_regressor.predict([[capacity]])[:, 0][0])


def turbine_from_capacity(capacity):
    """guesses turbine type from capacity using closest capacity in list

    Parameters
     ----------
     capacity : int/float
         capacity of wind turbine

     Returns
     -------
     turbine : str
         predicted turbine type
    """
    turbine_data = {
        225: "Vestas V29 225",
        500: "Vestas V39 500",
        600: "Vestas V44 600",
        660: "Vestas V47 660",
        850: "Vestas V52 850",
        1000: "NEG Micon NM60 1000",
        1650: "Vestas V66 1650",
        1750: "Vestas V66 1750",
        1800: "Vestas V80 1800",
        2000: "Vestas V90 2000",
        3000: "Vestas V90 3000",
        3300: "Vestas V112 3300",
        # 3400: 'REpower 3 4M',
        5000: "REpower 5M",
        6000: "REpower 6M",
    }

    turbine = (
        turbine_data.get(capacity)
        or turbine_data[min(turbine_data.keys(), key=lambda key: abs(key - capacity))]
    )
    # turbine = turbine.replace(".", " ")
    return turbine


def wind_onshore_corrected(year):
    """corrects wind onshore normalised using historical annual generation data

    Parameters
     ----------
     year : int/float/str
         year of time series

     Returns
     -------
     dataframe
         timeseries dataframe with corrected factor included
    """

    file = "data/renewables/" + str(year) + "/onshore_time_series.pkl"
    df_onshore = renewables.fix_timeseries_res_for_year(file, year, "Wind Onshore")
    # print(df_onshore)
    # print(df_onshore.columns)
    # total = df_onshore.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # data from https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/875384/Wind_powered_electricity_in_the_UK.pdf
    # reported generation data for onshore wind in UK
    gen_data = {
        "2010": 7.226,
        "2011": 10.814,
        "2012": 12.244,
        "2013": 16.925,
        "2014": 18.555,
        "2015": 22.852,
        "2016": 20.749,
        "2017": 28.717,
        "2018": 30.217,
        "2019": 32.205,
        "2020": 35.0,
    }  # 2020 assumed not from data
    gen_year = gen_data[str(year)]

    # then need to adjust the normalised time series
    factor = gen_year / (df_onshore.to_numpy().sum() / 1000000000)
    file = "data/renewables/" + str(year) + "/onshore_time_series_norm.pkl"
    df_onshore_norm = renewables.fix_timeseries_res_for_year(file, year, "Wind Onshore")

    # df2 = df_onshore * factor
    # total = df2.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    df_onshore_corrected = df_onshore_norm * factor
    # print(df_onshore_corrected)

    return df_onshore_corrected


def wind_onshore_corrected_series(year):
    """corrects wind onshore series using historical annual generation data

    Parameters
     ----------
     year : int/float/str
         year of time series

     Returns
     -------
     dataframe
         timeseries dataframe with corrected factor included
    """
    file = "data/renewables/" + str(year) + "/onshore_time_series.pkl"
    df_onshore = renewables.fix_timeseries_res_for_year(file, year, "Wind Onshore")
    # print(df_onshore)
    # print(df_onshore.columns)
    # total = df_onshore.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # data from https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/875384/Wind_powered_electricity_in_the_UK.pdf
    # reported generation data for onshore wind in UK
    gen_data = {
        "2010": 7.226,
        "2011": 10.814,
        "2012": 12.244,
        "2013": 16.925,
        "2014": 18.555,
        "2015": 22.852,
        "2016": 20.749,
        "2017": 28.717,
        "2018": 30.217,
        "2019": 32.205,
        "2020": 35.0,
    }  # 2020 assumed not from data
    gen_year = gen_data[str(year)]

    # then need to adjust the normalised time series
    factor = gen_year / (df_onshore.to_numpy().sum() / 1000000000)
    file = "data/renewables/" + str(year) + "/onshore_time_series_norm.pkl"
    # df_onshore_norm = renewables.fix_timeseries_res_for_year(file, year, 'Wind Onshore')

    df2 = df_onshore * factor

    # total = df2.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # df_onshore_corrected = df_onshore_norm * factor
    # print(df_onshore_corrected)

    return df2


def wind_offshore_corrected(year):
    """corrects wind offshore normalised using historical annual generation data

    Parameters
     ----------
     year : int/float/str
         year of time series

     Returns
     -------
     dataframe
         timeseries dataframe with corrected factor included
    """
    file = "data/renewables/" + str(year) + "/offshore_time_series.pkl"
    df_offshore = renewables.fix_timeseries_res_for_year(file, year, "Wind Offshore")
    # print(df_offshore.to_numpy().sum() / 1000000000, 'TWh')
    # print(df_onshore)
    # print(df_onshore.columns)
    # total = df_onshore.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # data from https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/875384/Wind_powered_electricity_in_the_UK.pdf
    # reported generation data for onshore wind in UK
    gen_data = {
        "2010": 3.060,
        "2011": 5.149,
        "2012": 7.603,
        "2013": 11.472,
        "2014": 13.405,
        "2015": 17.423,
        "2016": 16.406,
        "2017": 20.916,
        "2018": 26.687,
        "2019": 31.929,
        "2020": 38.0,
    }  # 2020 assumed not from data
    gen_year = gen_data[str(year)]

    # then need to adjust the normalised time series
    factor = gen_year / (df_offshore.to_numpy().sum() / 1000000000)
    file = "data/renewables/" + str(year) + "/offshore_time_series_norm.pkl"
    df_offshore_norm = renewables.fix_timeseries_res_for_year(
        file, year, "Wind Offshore"
    )

    # df2 = df_offshore * factor
    # total = df2.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    df_offshore_corrected = df_offshore_norm * factor
    # total = df_offshore_corrected.to_numpy().sum() / 1000000000
    # print(total, 'TWh')
    # print(df_offshore_corrected)

    return df_offshore_corrected


def wind_offshore_corrected_series(year):
    """corrects wind offshore series using historical annual generation data

    Parameters
     ----------
     year : int/float/str
         year of time series

     Returns
     -------
     dataframe
         timeseries dataframe with corrected factor included
    """
    file = "data/renewables/" + str(year) + "/offshore_time_series.pkl"
    df_offshore = renewables.fix_timeseries_res_for_year(file, year, "Wind Offshore")
    # print(df_offshore.to_numpy().sum() / 1000000000, 'TWh')
    # print(df_onshore)
    # print(df_onshore.columns)
    # total = df_onshore.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # data from https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/875384/Wind_powered_electricity_in_the_UK.pdf
    # reported generation data for onshore wind in UK
    gen_data = {
        "2010": 3.060,
        "2011": 5.149,
        "2012": 7.603,
        "2013": 11.472,
        "2014": 13.405,
        "2015": 17.423,
        "2016": 16.406,
        "2017": 20.916,
        "2018": 26.687,
        "2019": 31.929,
        "2020": 38.0,
    }  # 2020 assumed not from data
    gen_year = gen_data[str(year)]

    # then need to adjust the normalised time series
    factor = gen_year / (df_offshore.to_numpy().sum() / 1000000000)
    file = "data/renewables/" + str(year) + "/offshore_time_series_norm.pkl"
    # df_offshore_norm = data_reader_writer.fix_timeseries_res_for_year(file, year, 'Wind Offshore')

    df2 = df_offshore * factor
    # total = df2.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # df_offshore_corrected = df_offshore_norm * factor
    # total = df_offshore_corrected.to_numpy().sum() / 1000000000
    # print(total, 'TWh')
    # print(df_offshore_corrected)

    return df2


def PV_corrected(year):
    """corrects PV normalised using historical annual generation data

    Parameters
     ----------
     year : int/float/str
         year of time series

     Returns
     -------
     dataframe
         timeseries dataframe with corrected factor included
    """
    file = "data/renewables/" + str(year) + "/PV_time_series.pkl"
    df_PV = renewables.fix_timeseries_res_for_year(file, year, "Solar Photovoltaics")
    # print(df_PV.to_numpy().sum() / 1000000000, 'TWh')
    # print(df_PV)
    # print(df_PV.columns)
    # total = df_PV.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # data from https://en.wikipedia.org/wiki/Solar_power_in_the_United_Kingdom
    # reported generation data for PV in UK
    gen_data = {
        "2010": 0.033,
        "2011": 0.259,
        "2012": 1.328,
        "2013": 2.015,
        "2014": 4.050,
        "2015": 7.561,
        "2016": 10.292,
        "2017": 11.525,
        "2018": 12.922,
        "2019": 13.616,
        "2020": 14.3,
    }  # 2020 assumed, not from data
    gen_year = gen_data[str(year)]

    # then need to adjust the normalised time series
    factor = gen_year / (df_PV.to_numpy().sum() / 1000000000)
    file = "data/renewables/" + str(year) + "/PV_time_series_norm.pkl"
    df_PV_norm = renewables.fix_timeseries_res_for_year(
        file, year, "Solar Photovoltaics"
    )

    # df2 = df_PV * factor
    # total = df2.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    df_PV_corrected = df_PV_norm * factor
    # total = df_PV_corrected.to_numpy().sum() / 1000000000

    return df_PV_corrected


def PV_corrected_series(year):
    """corrects PV series using historical annual generation data

    Parameters
     ----------
     year : int/float/str
         year of time series

     Returns
     -------
     dataframe
         timeseries dataframe with corrected factor included
    """

    file = "data/renewables/" + str(year) + "/PV_time_series.pkl"
    df_PV = renewables.fix_timeseries_res_for_year(file, year, "Solar Photovoltaics")
    # print(df_PV.to_numpy().sum() / 1000000000, 'TWh')
    # print(df_PV)
    # print(df_PV.columns)
    # total = df_PV.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # data from https://en.wikipedia.org/wiki/Solar_power_in_the_United_Kingdom
    # reported generation data for PV in UK
    gen_data = {
        "2010": 0.033,
        "2011": 0.259,
        "2012": 1.328,
        "2013": 2.015,
        "2014": 4.050,
        "2015": 7.561,
        "2016": 10.292,
        "2017": 11.525,
        "2018": 12.922,
        "2019": 13.616,
        "2020": 14.3,
    }  # 2020 assumed, not from data
    gen_year = gen_data[str(year)]

    # then need to adjust the normalised time series
    factor = gen_year / (df_PV.to_numpy().sum() / 1000000000)
    file = "data/renewables/" + str(year) + "/PV_time_series_norm.pkl"
    # df_PV_norm = data_reader_writer.fix_timeseries_res_for_year(file, year, 'Solar Photovoltaics')

    df2 = df_PV * factor
    # total = df2.to_numpy().sum() / 1000000000
    # print(total, 'TWh')

    # df_PV_corrected = df_PV_norm * factor
    # total = df_PV_corrected.to_numpy().sum() / 1000000000

    return df2


if __name__ == "__main__":

    year = 2011
    # read in the renewables dataframe
    df = renewables.REPD_date_corrected(year)
    # df_pv = df.loc[df['Technology Type'] == 'Solar Photovoltaics'].reset_index(drop=True)
    # print(df_pv)

    # token for renewables.ninja API

    # insert your own here
    token = "INSERT_OWN_API_TOKEN"

    # dates required
    date_from = "2011-01-01"
    date_to = "2011-12-31"

    # fix_PV()

    offshore_wind_time_series(df, year, date_from, date_to)
    onshore_wind_time_series(df, year, date_from, date_to)
    PV_time_series(df, year, date_from, date_to)

    # wind_onshore_corrected(year)
    # wind_offshore_corrected(year)
    # PV_corrected(year)

    # df = pickle.load(open("data/renewables/PV_time_series.pkl", "rb"))
    # print(df.head(24))
    # df2 = pickle.load(open("data/renewables/PV_time_series_norm.pkl", "rb"))
    # print(df2.head(24))
