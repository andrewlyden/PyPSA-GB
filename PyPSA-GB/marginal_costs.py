import pandas as pd
import matplotlib.pyplot as plt

from .utils.cleaning import unify_index


def fuel_prices_df(df):

    # from https://www.gov.uk/government/statistical-data-sets/prices-of-fuels-purchased-by-major-power-producers
    dti_quarterly = pd.date_range(start="2009-12-31", end="2020-12", freq="QS")

    fuel_prices = df["Fuel prices"]
    fuel_prices.set_index(dti_quarterly, inplace=True)
    fuel_prices.drop(columns=["Unnamed: 0", "Unnamed: 1"], inplace=True)

    fuel_prices = fuel_prices.resample("0.5h").ffill()

    dti_end = pd.date_range(
        start="2020-10-01 00:30:00", end="2020-12-31 23:30:00", freq="0.5h"
    )
    cols = ["Coal (p/kWh)", "Oil (p/kWh)", "Gas (p/kWh)"]
    # data = [0.752, 2.815, 1.167] * len(dti_end)
    df_end = pd.DataFrame(columns=cols, index=dti_end)
    df_end["Coal (p/kWh)"] = 0.752
    df_end["Oil (p/kWh)"] = 2.815
    df_end["Gas (p/kWh)"] = 1.167

    fuel_prices = pd.concat([fuel_prices, df_end])

    # print(fuel_prices.loc['2020-09-30 22:00':'2020-10-02 00:00'])

    # want to multiply by efficiency to get prices in kWh of electricity generation
    fuel_prices["Coal (p/kWh)"] /= 0.35
    fuel_prices["Oil (p/kWh)"] /= 0.35
    fuel_prices["Gas (p/kWh)"] /= 0.50

    return fuel_prices


def future_fuel_prices_df(FES):

    if FES == 2021:
        df_FES = pd.read_excel(
            "../data/FES2021/FES 2021 Data Workbook V04.xlsx",
            sheet_name="CP1",
            usecols="L:BA",
            header=8,
            dtype=str,
            index_col=0,
        )
    elif FES == 2022:
        df_FES = pd.read_excel(
            "../data/FES2022/FES2022 Workbook V4.xlsx",
            sheet_name="CP1",
            usecols="L:BA",
            header=8,
            dtype=str,
            index_col=0,
        )
    df_FES.dropna(axis="rows", inplace=True)
    df_FES.drop(["Year"], inplace=True)

    df_FES = df_FES.T
    df_FES.drop(range(2010, 2021), inplace=True)
    df_FES.index = pd.to_datetime(df_FES.index, format="%Y")
    df_FES = df_FES.resample("0.5h").ffill()
    dti = pd.date_range(
        start="2050-01-01 00:30:00", end="2050-12-31 23:30:00", freq="0.5h"
    )
    values = df_FES.loc["2050-01-01 00:00:00"]
    df_2050 = pd.DataFrame([values], columns=df_FES.columns, index=dti)
    df_FES = pd.concat([df_FES, df_2050])

    return df_FES


def carbon_support_price_df(df):

    csp = df["Carbon support price"]
    # datetimeindex from 2010 to 2020 to match the fuel prices dataframe
    dti = pd.date_range(
        start="2010-01-01 00:00:00", end="2020-12-31 23:30:00", freq="0.5h"
    )
    csp_df = pd.DataFrame(columns=["Carbon support price (Pounds/tonne)"], index=dti)
    csp_df.loc["2010-01-01":"2013-03-31 23:30"] = 0.0
    csp_df.loc["2013-04-01":"2014-04-01"] = csp.loc[0, "Carbon Support Price (£/tonne)"]
    csp_df.loc["2014-04-01":"2015-04-01"] = csp.loc[1, "Carbon Support Price (£/tonne)"]
    csp_df.loc["2015-04-01":"2016-04-01"] = csp.loc[2, "Carbon Support Price (£/tonne)"]
    csp_df.loc["2016-04-01":"2020-12-31"] = csp.loc[3, "Carbon Support Price (£/tonne)"]

    return csp_df


def EU_ETS_df(df):

    ets = df["EU ETS"]
    ets.loc[:, "Date"] = pd.to_datetime(ets["Date"])
    ets.set_index("Date", inplace=True)
    # fill in missing data
    ets = ets.resample("D").interpolate()
    # narrow to 2010 to 2020
    ets = ets.loc["2010-01-01":"2020-12-31"]
    # downsample to half hourly
    ets = ets.resample("0.5h").ffill()
    # rename the column name
    ets.rename(columns={"Price (Euros/tonne)": "EU ETS (Euros/tonne)"}, inplace=True)
    # need to add last day of values
    dti_end = pd.date_range(
        start="2020-12-31 00:30:00", end="2020-12-31 23:30:00", freq="0.5h"
    )
    cols = ["EU ETS (Euros/tonne)"]
    # data = [0.752, 2.815, 1.167] * len(dti_end)
    df_end = pd.DataFrame(columns=cols, index=dti_end)
    df_end.loc[:, "EU ETS (Euros/tonne)"] = 37.72
    ets = pd.concat([ets, df_end])

    return ets


def future_carbon_prices_df(FES):

    if FES == 2021:
        df_FES = pd.read_excel(
            "../data/FES2021/FES 2021 Data Workbook V04.xlsx",
            sheet_name="CP2",
            usecols="N:BC",
            header=7,
            dtype=str,
            index_col=0,
        )
    elif FES == 2022:
        df_FES = pd.read_excel(
            "../data/FES2022/FES2022 Workbook V4.xlsx",
            sheet_name="CP2",
            usecols="N:BC",
            header=7,
            dtype=str,
            index_col=0,
        )
    df_FES.drop(columns=range(2010, 2021), inplace=True)
    df_FES.dropna(axis="rows", inplace=True)
    df_FES.drop(["Year"], inplace=True)
    # remove last three rows
    df_FES = df_FES[:-3].T
    df_FES.index = pd.to_datetime(df_FES.index, format="%Y")
    df_FES = df_FES.resample("0.5h").ffill()
    dti = pd.date_range(
        start="2050-01-01 00:30:00", end="2050-12-31 23:30:00", freq="0.5h"
    )
    values = df_FES.loc["2050-01-01 00:00:00"]
    df_2050 = pd.DataFrame([values], columns=df_FES.columns, index=dti)
    df_FES = pd.concat([df_FES, df_2050])

    return df_FES


def exchange_year_average():

    # euros to pound historical data, average over year
    data = {
        2010: 0.8583,
        2011: 0.8678,
        2012: 0.8113,
        2013: 0.8492,
        2014: 0.8061,
        2015: 0.7263,
        2016: 0.8193,
        2017: 0.8766,
        2018: 0.8850,
        2019: 0.8773,
        2020: 0.8897,
    }

    return data


def marginal_price_dataframe(FES):

    df = pd.read_excel("../data/marginal_cost_data.xlsx", sheet_name=None)

    fuel_prices = fuel_prices_df(df)
    carbon_support_price = carbon_support_price_df(df)
    EU_ETS = EU_ETS_df(df)
    EU_ETS.set_index(fuel_prices.index, inplace=True)

    result = pd.concat([fuel_prices, carbon_support_price, EU_ETS], axis=1)

    # exchange euros for pounds
    exch = exchange_year_average()
    for year in range(2010, 2021):
        result.loc[:, "EU ETS (Euros/tonne)"].loc[str(year) : str(year)] *= exch[year]
    result.rename(
        columns={"EU ETS (Euros/tonne)": "EU ETS (Pounds/tonne)"}, inplace=True
    )

    # now to calculate the carbon tax and total prices
    # using conversion factors from 2021 DUKES data,
    # this should probably change from year to year

    # these are from
    # https://www.parliament.uk/globalassets/documents/post/postpn_383-carbon-footprint-electricity-generation.pdf
    coal_emission_factor = 846
    # really should be different between OCGT and CCGT
    gas_emission_factor = 488
    # need better reference for oil
    # https://www.jcm.go.jp/cl-jp/methodologies/68/attached_document2
    oil_emission_factor = 533

    result.loc[:, "Gas carbon tax (p/kWh)"] = (
        (
            result["Carbon support price (Pounds/tonne)"]
            + result["EU ETS (Pounds/tonne)"]
        )
        * gas_emission_factor
        / 10000
    )
    result.loc[:, "Coal carbon tax (p/kWh)"] = (
        (
            result["Carbon support price (Pounds/tonne)"]
            + result["EU ETS (Pounds/tonne)"]
        )
        * coal_emission_factor
        / 10000
    )
    result.loc[:, "Oil carbon tax (p/kWh)"] = (
        (
            result["Carbon support price (Pounds/tonne)"]
            + result["EU ETS (Pounds/tonne)"]
        )
        * oil_emission_factor
        / 10000
    )

    # converting to £/MWh
    result.loc[:, "Gas"] = (
        result["Gas carbon tax (p/kWh)"] + result["Gas (p/kWh)"]
    ) * 10
    result.loc[:, "Coal"] = (
        result["Coal carbon tax (p/kWh)"] + result["Coal (p/kWh)"]
    ) * 10
    result.loc[:, "Oil"] = (
        result["Oil carbon tax (p/kWh)"] + result["Oil (p/kWh)"]
    ) * 10

    # frame = {'Coal (£/MWh)': result['Coal (p/kWh)'] * 10,
    #          'Carbon support price (£/MWh)': result['Carbon support price (Pounds/tonne)'] * coal_emission_factor / 10000 * 10,
    #          'EU ETS (£/MWh)': result['EU ETS (Pounds/tonne)'] * coal_emission_factor / 10000 * 10}
    # df_coal = pd.DataFrame(frame)
    # df_coal.plot.area()
    # plt.show()
    # print(result)

    # compare to electricity mix shown here
    # https://theconversation.com/britains-electricity-since-2010-wind-surges-to-second-place-coal-collapses-and-fossil-fuel-use-nearly-halves-129346
    # plt.plot(result['Coal'], label='Coal fuel + carbon tax')
    # plt.plot(result['Gas'], label='Gas fuel + carbon tax')
    # plt.legend(loc='best')
    # plt.show()

    marginal_prices = result[["Coal", "Gas", "Oil"]].copy()

    # plt.plot(marginal_prices)
    # plt.legend(['Coal price + carbon (£/MWh)', 'Gas price + carbon (£/MWh)', 'Oil price + carbon (£/MWh)'], loc='best')
    # plt.show()
    # print(marginal_prices)

    # add the future prices

    future_fuel_price = future_fuel_prices_df(FES)
    future_fuel_price = future_fuel_price.apply(pd.to_numeric, errors="coerce")
    future_carbon_price = future_carbon_prices_df(FES)
    future_carbon_price = future_carbon_price.apply(pd.to_numeric, errors="coerce")

    future_result = pd.concat([future_fuel_price, future_carbon_price], axis=1)

    future_result.loc[:, "Gas carbon tax (p/kWh)"] = (
        (future_result["High case"]) * gas_emission_factor / 10000
    )
    future_result.loc[:, "Coal carbon tax (p/kWh)"] = (
        (future_result["High case"]) * coal_emission_factor / 10000
    )
    future_result.loc[:, "Oil carbon tax (p/kWh)"] = (
        (future_result["High case"]) * oil_emission_factor / 10000
    )

    # converting to £/MWh
    # gas is in p/therm, divide by 29.3 * 0.5 = 14.65, assume elec efficiency 50%
    # this gets us to p/kWh, then * 10 to get £/MWh
    future_result.loc[:, "Gas"] = (
        future_result["Gas carbon tax (p/kWh)"] * 10
        + future_result["Gas price"] * 10 / 14.65
    )
    # coal is in USD per tonne, * 0.75 to get pounds
    # one tonne can create 2.460 MWh electricity, divide by this
    # to get £/MWh
    # also a 1.4 fudge factor to get closer to 2020 price
    future_result.loc[:, "Coal"] = (
        future_result["Coal carbon tax (p/kWh)"] * 10
        + future_result["Coal price"] * 1.4 * 0.8 / 2.460
    )
    # oil is in $ per barrel, so * 0.75 to get pounds
    # 1.7 MWh per barrel thermal, so *0.3 to get 0.85 MWh elec
    # divide by this to get £/MWh
    future_result.loc[:, "Oil"] = (
        future_result["Oil carbon tax (p/kWh)"] * 10
        + future_result["Oil price"] * 0.8 / 0.51
    )

    marginal_prices2 = future_result[["Coal", "Gas", "Oil"]].copy()

    marginal_prices = pd.concat([marginal_prices, marginal_prices2])

    return marginal_prices


def write_marginal_costs_series(start, end, freq, year, FES):

    # this will be the same for unit commitment and LOPF
    # read in the list of generators
    df_gens = pd.read_csv("LOPF_data/generators.csv")
    start_ = pd.to_datetime(start)
    end_ = pd.to_datetime(end)
    marginal_cost_df = marginal_price_dataframe(FES).loc[start_:end_]

    # Add the coal marginal prices
    coal_gens = df_gens.loc[df_gens["carrier"] == "Coal"]
    # Create a new dataframe with columns as generator names and rows filled with marginal cost values
    df1 = pd.DataFrame(
        data={
            gen_name: marginal_cost_df["Coal"].values
            for gen_name in coal_gens.name.values
        },
        index=marginal_cost_df.index,
    )

    # Add the CCGT marginal prices
    CCGT_gens = df_gens.loc[df_gens["type"] == "CCGT"]
    # Create a new dataframe with columns as generator names and rows filled with marginal cost values
    df2 = pd.DataFrame(
        data={
            gen_name: marginal_cost_df["Gas"].values
            for gen_name in CCGT_gens.name.values
        },
        index=marginal_cost_df.index,
    )

    # Add the OCGT marginal prices
    OCGT_gens = df_gens.loc[df_gens["type"] == "OCGT"]
    # Create a new dataframe with columns as generator names and rows filled with marginal cost values
    df3 = pd.DataFrame(
        data={
            gen_name: marginal_cost_df["Gas"].values
            for gen_name in OCGT_gens.name.values
        },
        index=marginal_cost_df.index,
    )

    # Add the oil marginal prices
    oil_gens = df_gens.loc[df_gens["carrier"] == "Oil"]
    # Create a new dataframe with columns as generator names and rows filled with marginal cost values
    df4 = pd.DataFrame(
        data={
            gen_name: marginal_cost_df["Oil"].values
            for gen_name in oil_gens.name.values
        },
        index=marginal_cost_df.index,
    )

    df = pd.concat([df1, df2, df3, df4], axis=1)
    df.index.name = "name"

    if freq == "h":

        df = df.astype(float)
        # df.index = pd.to_datetime(df.index)
        # print(df)
        df = df.resample(freq).mean()
        # appendix = df.iloc[-1:]
        # appendix.index = appendix.index + pd.Timedelta(minutes=30)
        # df = df.append(appendix)

    elif year > 2020:
        if freq == "h":
            # appendix = df.iloc[-1:]
            # appendix.index = appendix.index + pd.Timedelta(minutes=30)
            # appendix.index = appendix.index  + pd.Timedelta(minutes=30)
            # df = df.resample(freq).mean()
            df = df.iloc[::2, :]
            # df = df.append(appendix)

    df.to_csv("UC_data/generators-marginal_cost.csv", header=True)
    df.to_csv("LOPF_data/generators-marginal_cost.csv", header=True)


if __name__ == "__main__":

    # df = pd.read_excel('../data/marginal_cost_data.xlsx', sheet_name=None)
    # fuel_prices_df(df)
    # carbon_support_price_df(df)
    # EU_ETS_df(df)
    # marginal_price_dataframe()
    # future_fuel_prices_df()
    # future_carbon_prices_df()

    marginal_price_dataframe()
