import pandas as pd


def read_DUKES(year):

    # this contains all the data for all years
    df = pd.read_excel('DUKES_5.11.xls', sheet_name=None)

    if not year == 2020:
        sheet_name = 'DUKES ' + str(year)
    else:
        sheet_name = '5.11'

    df = df[sheet_name]

    if sheet_name == 'DUKES 2019' or sheet_name == '5.11':
        # set the column names
        df.columns = df.iloc[4]
        df = df.drop(range(5), axis=0)
        # drop the last column
        df = df.iloc[:, :-1]

    else:
        # set the column names
        df.columns = df.iloc[2]
        df = df.drop(range(3), axis=0)
        # drop the last column
        # df = df.iloc[:, :-1]

    # column name changes after 2016 for installed capacity
    if year >= 2016:
        df.rename(columns={'Installed Capacity\n(MW)': 'Installed Capacity (MW)'}, inplace=True)

    # drop northern ireland plants
    try:
        df.drop(df[df['Location'] == 'Northern Ireland'].index, inplace=True)
    except:
        df.drop(df[df['Location Scotland, Wales, Northern Ireland, or English region'] == 'Northern Ireland'].index, inplace=True)
        df.rename(columns={'Location Scotland, Wales, Northern Ireland, or English region': 'Location'}, inplace=True)

    # The following uses the various names for fuel in the
    # DUKES data and groups them accordingly
    # Note where more than one fuel is named, the first is
    # taken as the fuel to group to
    # i.e. gas/coal is grouped to gas
    df_coal = df.loc[df['Fuel'] == 'Coal']

    df_coal2 = df.loc[df['Fuel'] == 'Coal/oil']
    df_coal2.loc[:, 'Technology'] = df_coal2.loc[:, ['Fuel']]
    df_coal2.loc[:, 'Fuel'] = 'Coal'

    df_coal3 = df.loc[df['Fuel'] == 'Coal / oil']
    df_coal3.loc[:, 'Technology'] = df_coal3.loc[:, ['Fuel']]
    df_coal3.loc[:, 'Fuel'] = 'Coal'

    df_coal4 = df.loc[df['Fuel'] == 'coal/oil']
    df_coal4.loc[:, 'Technology'] = df_coal4.loc[:, ['Fuel']]
    df_coal4.loc[:, 'Fuel'] = 'Coal'

    df_coal5 = df.loc[df['Fuel'] == 'Coal/biomass']
    df_coal5.loc[:, 'Technology'] = df_coal5.loc[:, ['Fuel']]
    df_coal5.loc[:, 'Fuel'] = 'Coal'

    df_coal6 = df.loc[df['Fuel'] == 'coal/biomass']
    df_coal6.loc[:, 'Technology'] = df_coal6.loc[:, ['Fuel']]
    df_coal6.loc[:, 'Fuel'] = 'Coal'

    df_coal7 = df.loc[df['Fuel'] == 'coal']
    df_coal7.loc[:, 'Technology'] = df_coal7.loc[:, ['Fuel']]
    df_coal7.loc[:, 'Fuel'] = 'Coal'

    df_coal8 = df.loc[df['Fuel'] == 'coal/gas']
    df_coal8.loc[:, 'Technology'] = df_coal8.loc[:, ['Fuel']]
    df_coal8.loc[:, 'Fuel'] = 'Coal'

    df_nuclear = df.loc[df['Fuel'] == 'Nuclear']

    df_nuclear2 = df.loc[df['Fuel'] == 'nuclear']
    df_nuclear2.loc[:, 'Technology'] = df_nuclear2.loc[:, ['Fuel']]
    df_nuclear2.loc[:, 'Fuel'] = 'Nuclear'

    df_gas = df.loc[df['Fuel'] == 'Natural Gas']

    # Sour gas is natural gas or any other gas containing
    # significant amounts of hydrogen sulfide
    df_gas2 = df.loc[df['Fuel'] == 'Sour gas']
    df_gas2.loc[:, 'Technology'] = df_gas2.loc[:, ['Fuel']]
    df_gas2.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas3 = df.loc[df['Fuel'] == 'CCGT']
    df_gas3.loc[:, 'Technology'] = df_gas3.loc[:, ['Fuel']]
    df_gas3.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas4 = df.loc[df['Fuel'] == 'OCGT']
    df_gas4.loc[:, 'Technology'] = df_gas4.loc[:, ['Fuel']]
    df_gas4.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas5 = df.loc[df['Fuel'] == 'Gas']
    df_gas5.loc[:, 'Technology'] = df_gas5.loc[:, ['Fuel']]
    df_gas5.loc[:, 'Technology'] = 'OCGT'
    df_gas5.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas6 = df.loc[df['Fuel'] == 'gas']
    df_gas6.loc[:, 'Technology'] = df_gas6.loc[:, ['Fuel']]
    df_gas6.loc[:, 'Technology'] = 'OCGT'
    df_gas6.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas7 = df.loc[df['Fuel'] == 'Gas / oil']
    df_gas7.loc[:, 'Technology'] = df_gas7.loc[:, ['Fuel']]
    df_gas7.loc[:, 'Technology'] = 'OCGT'
    df_gas7.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas8 = df.loc[df['Fuel'] == 'gas/oil']
    df_gas8.loc[:, 'Technology'] = df_gas8.loc[:, ['Fuel']]
    df_gas8.loc[:, 'Technology'] = 'OCGT'
    df_gas8.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas9 = df.loc[df['Fuel'] == 'gas/coal/oil']
    df_gas9.loc[:, 'Technology'] = df_gas9.loc[:, ['Fuel']]
    df_gas9.loc[:, 'Technology'] = 'OCGT'
    df_gas9.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas10 = df.loc[df['Fuel'] == 'gas CHP']
    df_gas10.loc[:, 'Technology'] = df_gas10.loc[:, ['Fuel']]
    df_gas10.loc[:, 'Technology'] = 'OCGT'
    df_gas10.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas11 = df.loc[df['Fuel'] == 'Gas/Coal/Oil']
    df_gas11.loc[:, 'Technology'] = df_gas11.loc[:, ['Fuel']]
    df_gas11.loc[:, 'Technology'] = 'OCGT'
    df_gas11.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas12 = df.loc[df['Fuel'] == 'gas/oil/OCGT']
    df_gas12.loc[:, 'Technology'] = df_gas12.loc[:, ['Fuel']]
    df_gas12.loc[:, 'Technology'] = 'OCGT'
    df_gas12.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas13 = df.loc[df['Fuel'] == 'gas turbine']
    df_gas13.loc[:, 'Technology'] = df_gas13.loc[:, ['Fuel']]
    df_gas13.loc[:, 'Technology'] = 'OCGT'
    df_gas13.loc[:, 'Fuel'] = 'Natural Gas'

    df_gas14 = df.loc[df['Fuel'] == 'Natural gas']
    # df_gas14.loc[:, 'Technology'] = 'OCGT'
    df_gas14.loc[:, 'Fuel'] = 'Natural Gas'

    df_oil = df.loc[df['Fuel'] == 'Diesel/gas Diesel/Gas oil']
    df_oil.loc[:, 'Technology'] = df_oil.loc[:, 'Fuel']
    df_oil.loc[:, 'Fuel'] = 'Oil'

    df_oil2 = df.loc[df['Fuel'] == 'Diesel/Gas oil']
    df_oil2.loc[:, 'Technology'] = df_oil2.loc[:, 'Fuel']
    df_oil2.loc[:, 'Fuel'] = 'Oil'

    df_oil3 = df.loc[df['Fuel'] == 'Gas oil']
    df_oil3.loc[:, 'Technology'] = df_oil2.loc[:, 'Fuel']
    df_oil3.loc[:, 'Fuel'] = 'Oil'

    df_oil4 = df.loc[df['Fuel'] == 'gas oil']
    df_oil4.loc[:, 'Technology'] = df_oil4.loc[:, 'Fuel']
    df_oil4.loc[:, 'Fuel'] = 'Oil'

    df_oil5 = df.loc[df['Fuel'] == 'Gas oil/kerosene']
    df_oil5.loc[:, 'Technology'] = df_oil5.loc[:, 'Fuel']
    df_oil5.loc[:, 'Fuel'] = 'Oil'

    df_oil6 = df.loc[df['Fuel'] == 'Gas oil / kerosene']
    df_oil6.loc[:, 'Technology'] = df_oil6.loc[:, 'Fuel']
    df_oil6.loc[:, 'Fuel'] = 'Oil'

    df_oil7 = df.loc[df['Fuel'] == 'Light oil']
    df_oil7.loc[:, 'Technology'] = df_oil7.loc[:, 'Fuel']
    df_oil7.loc[:, 'Fuel'] = 'Oil'

    df_oil8 = df.loc[df['Fuel'] == 'light oil']
    df_oil8.loc[:, 'Technology'] = df_oil8.loc[:, 'Fuel']
    df_oil8.loc[:, 'Fuel'] = 'Oil'

    df_oil9 = df.loc[df['Fuel'] == 'Diesel']
    df_oil9.loc[:, 'Technology'] = df_oil9.loc[:, 'Fuel']
    df_oil9.loc[:, 'Fuel'] = 'Oil'

    df_oil10 = df.loc[df['Fuel'] == 'diesel']
    df_oil10.loc[:, 'Technology'] = df_oil10.loc[:, 'Fuel']
    df_oil10.loc[:, 'Fuel'] = 'Oil'

    df_oil11 = df.loc[df['Fuel'] == 'Oil']
    df_oil11.loc[:, 'Technology'] = df_oil11.loc[:, 'Fuel']
    df_oil11.loc[:, 'Fuel'] = 'Oil'

    df_oil12 = df.loc[df['Fuel'] == 'gas oil/kerosene']
    df_oil12.loc[:, 'Technology'] = df_oil12.loc[:, 'Fuel']
    df_oil12.loc[:, 'Fuel'] = 'Oil'

    df_oil13 = df.loc[df['Fuel'] == 'Diesel/gas oil']
    df_oil13.loc[:, 'Technology'] = df_oil13.loc[:, 'Fuel']
    df_oil13.loc[:, 'Fuel'] = 'Oil'

    # join all of these dataframes together
    df_powerplants = df_coal.append(
        [df_coal2, df_coal3, df_coal4, df_coal5,
         df_coal6, df_coal7, df_coal8,
         df_nuclear, df_nuclear2,
         df_gas, df_gas2, df_gas3, df_gas4, df_gas5,
         df_gas6, df_gas7, df_gas8, df_gas9, df_gas10,
         df_gas11, df_gas12, df_gas13, df_gas14,
         df_oil, df_oil2, df_oil3, df_oil4, df_oil5,
         df_oil6, df_oil7, df_oil8, df_oil9, df_oil10,
         df_oil11, df_oil12, df_oil13])
    df_powerplants = df_powerplants.reset_index(drop=True)
    # get rid of empty rows
    try:
        df_powerplants = df_powerplants.dropna(subset=['Installed Capacity\n(MW)'])
    except:
        df_powerplants = df_powerplants.dropna(subset=['Installed Capacity (MW)'])
    df_powerplants = df_powerplants.set_index('Station Name')
    # print(df_powerplants)
    # print(df_powerplants.loc[df_powerplants['Fuel'] == 'Natural Gas'])

    return df_powerplants

def geolocation_data(year, df_powerplants):

    # now want to add the location of these power plants
    # read in the csv file which has lots of locations
    # this should have all power stations and their locations
    # will need updated with new power stations coming online
    df_location = pd.read_csv(
        '../power_stations_locations.csv', encoding='unicode_escape')
    df_location = df_location.set_index('Station Name')

    missing = []
    coordinates = {}

    for i in range(len(df_powerplants)):
        # get the station name
        station_name = df_powerplants.index[i]
        # print(station_name)

        try:
            coordinates[station_name] = df_location.loc[station_name, 'Geolocation']
            # print(coordinates)
        except:
            missing.append(station_name)

    # Fiddlers Ferry GT not working for some reason I cant figure out
    # so manually adding here
    # decomissioned in 2020 so this is just a workaround for now
    if not year == 2020:
        coordinates['Fiddler`s Ferry GT'] = '53.372,\xa0-2.687'
        coordinates['Fiddler`s Ferry'] = '53.372,\xa0-2.687'
        df_powerplants.rename(index={'Fiddler’s Ferry GT': 'Fiddler`s Ferry GT'}, inplace=True)
        df_powerplants.rename(index={'Fiddler’s Ferry': 'Fiddler`s Ferry'}, inplace=True)

    # remove duplicates
    df_powerplants = df_powerplants[~df_powerplants.index.duplicated(keep='first')]

    # need to add a column called Geolocation which maps to the station name
    for i in range(len(df_powerplants.index)):
        station_name = df_powerplants.index[i]
        location = coordinates[station_name]
        df_powerplants.loc[station_name, 'Geolocation'] = location
        x = df_powerplants['Geolocation'][station_name].split(',')[1]
        y = df_powerplants['Geolocation'][station_name].split(',')[0]
        df_powerplants.loc[station_name, 'x'] = float(x)
        df_powerplants.loc[station_name, 'y'] = float(y)

    df_powerplants = df_powerplants.loc[:, df_powerplants.columns.notnull()]
    return df_powerplants

def DUKES_dataframe(year):

    df_powerplants = read_DUKES(year)
    df_powerplants_with_location = geolocation_data(year, df_powerplants)
    return df_powerplants_with_location

def write_csv_file(dataframe, year):

    dataframe.to_csv('../power_stations_locations_' + str(year) + '.csv')


if __name__ == '__main__':

    # going to use the 2020 worksheet which is called 5.11
    # sheetname is 5.11 for 2020 but follows 'DUKES 2015' format for rest
    # year = 2012
    # df = DUKES_dataframe(year)
    # write_csv_file(df, year)

    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    for y in range(len(years)):
        year = years[y]
        df = DUKES_dataframe(year)
        write_csv_file(df, year)
