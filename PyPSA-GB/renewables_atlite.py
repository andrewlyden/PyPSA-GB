import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import geopandas as gpd


import cartopy.crs as ccrs
from cartopy.crs import PlateCarree as plate
import cartopy.io.shapereader as shpreader

import xarray as xr
import atlite

import logging
import warnings

import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

warnings.simplefilter('ignore')
logging.captureWarnings(False)
logging.basicConfig(level=logging.INFO)


def prepare_cutouts_years():

    years = list(range(2010, 2022 + 1))
    for y in years:
        shpfilename = shpreader.natural_earth(resolution='10m',
                                              category='cultural',
                                              name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        UK = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                            for r in reader.records()},
                           crs={'init': 'epsg:4326'}).reindex(['United Kingdom'])

        # Define the cutout; this will not yet trigger any major operations
        path = '../../atlite/cutouts/' + 'uk-' + str(y)
        time = str(y)
        cutout = atlite.Cutout(path=path,
                               module="era5",
                               bounds=UK.unary_union.bounds,
                               time=time)
        
        # print(cutout.available_features)

        # This is where all the work happens
        # (this can take some time, for 2018 it took circa 3 hours).
        # features = ['height', 'wind', 'temperature', 'runoff']
        cutout.prepare()
                 
def offshore_wind_farm_timeseries(sites, year, name):

    shpfilename = shpreader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    UK = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                        for r in reader.records()},
                       crs={'init': 'epsg:4326'}).reindex(['United Kingdom'])

    # Define the cutout; this will not yet trigger any major operations
    file = 'uk-' + str(year)
    time = str(year)
    path = '../../atlite/cutouts/' + file
    cutout = atlite.Cutout(path=path,
                           module="era5",
                           bounds=UK.unary_union.bounds,
                           time=time)
    # This is where all the work happens
    # (this can take some time, for us it took ~15 minutes).
    cutout.prepare()
    # projection = ccrs.Orthographic(-10, 35)
    cells = cutout.grid
    # df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # country_bound = gpd.GeoSeries(cells.unary_union)
    sites = sites.loc[[name]]

    nearest = cutout.data.sel(
        {'x': sites.x.values, 'y': sites.y.values}, 'nearest').coords
    sites['x'] = nearest.get('x').values
    sites['y'] = nearest.get('y').values
    cells_generation = sites.merge(
        cells, how='inner').rename(pd.Series(sites.index))

    layout = cutout.layout_from_capacity_list(sites, col='capacity')

    # fig, axes = plt.subplots(len(sites), sharex=True, figsize=(9, 4))
    # figure out a turbine type
    offshore_turbine_types = {3: 'Vestas_V112_3MW_offshore',
                              5: 'NREL_ReferenceTurbine_5MW_offshore',
                              7: 'Vestas_V164_7MW_offshore',
                              8: 'NREL_ReferenceTurbine_2016CACost_8MW_offshore',
                             10: 'NREL_ReferenceTurbine_2016CACost_10MW_offshore',
                             12: 'NREL_ReferenceTurbine_2020ATB_12MW_offshore',
                             15: 'NREL_ReferenceTurbine_2020ATB_15MW_offshore'}
    possible_cap = [3, 5, 7]
    find_closest = lambda num, collection: min(possible_cap, key=lambda x: abs(x - num))
    cap = sites['capacity'].values[0]
    nearest_capacity = find_closest(cap, possible_cap)
    turbine_type = offshore_turbine_types[nearest_capacity]

    power_generation = cutout.wind(turbine_type, layout=layout,
                                   shapes=cells_generation.geometry)
    factor = sites['Installed Capacity (MWelec)'].values[0] / cap
    power_generation_farm = power_generation * factor
    df_power_generation_farm = power_generation_farm.to_pandas()

    # df_power_generation_farm.plot(subplots=True, ax=axes)
    # axes.set_xlabel('date')
    # axes.set_ylabel('Generation [MW]')
    # fig.tight_layout()
    # plt.show()

    return df_power_generation_farm


def onshore_wind_farm_timeseries(sites, year, name):

    shpfilename = shpreader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    UK = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                        for r in reader.records()},
                       crs={'init': 'epsg:4326'}).reindex(['United Kingdom'])

    # Define the cutout; this will not yet trigger any major operations
    file = 'uk-' + str(year)
    time = str(year)
    path = '../data/renewables/atlite/cutouts/' + file
    cutout = atlite.Cutout(path=path,
                           module="era5",
                           bounds=UK.unary_union.bounds,
                           time=time)
    # This is where all the work happens
    # (this can take some time, for us it took ~15 minutes).
    cutout.prepare()
    # projection = ccrs.Orthographic(-10, 35)
    cells = cutout.grid
    # df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # country_bound = gpd.GeoSeries(cells.unary_union)
    sites = sites.loc[[name]]

    nearest = cutout.data.sel(
        {'x': sites.x.values, 'y': sites.y.values}, 'nearest').coords
    sites['x'] = nearest.get('x').values
    sites['y'] = nearest.get('y').values
    cells_generation = sites.merge(
        cells, how='inner').rename(pd.Series(sites.index))

    layout = cutout.layout_from_capacity_list(sites, col='capacity')

    # fig, axes = plt.subplots(len(sites), sharex=True, figsize=(9, 4))
    # figure out a turbine type
    onshore_turbine_types = {0.66: 'Vestas_V47_660kW',
                             2.3: 'Siemens_SWT_2300kW',
                             1.5: 'Suzlon_S82_1.5_MW',
                             3.0: 'Vestas_V112_3MW',
                             3.6: 'Siemens_SWT_107_3600kW',
                             2.0: 'Vestas_V80_2MW_gridstreamer',
                             1.0: 'Bonus_B1000_1000kW',
                             0.2: 'Vestas_V25_200kW',
                             7.5: 'Enercon_E126_7500kW',
                             1.75: 'Vestas_V66_1750kW'}
    possible_cap = list(onshore_turbine_types.keys())
    find_closest = lambda num, collection: min(possible_cap, key=lambda x: abs(x - num))
    cap = sites['capacity'].values[0]
    nearest_capacity = find_closest(cap, possible_cap)
    turbine_type = onshore_turbine_types[nearest_capacity]

    power_generation = cutout.wind(turbine_type, layout=layout,
                                   shapes=cells_generation.geometry)
    factor = sites['Installed Capacity (MWelec)'].values[0] / cap
    power_generation_farm = power_generation * factor
    df_power_generation_farm = power_generation_farm.to_pandas()

    # df_power_generation_farm.plot(subplots=True, ax=axes)
    # axes.set_xlabel('date')
    # axes.set_ylabel('Generation [MW]')
    # fig.tight_layout()
    # plt.show()

    return df_power_generation_farm


def PV_timeseries(sites, year, name):

    shpfilename = shpreader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    UK = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                        for r in reader.records()},
                       crs={'init': 'epsg:4326'}).reindex(['United Kingdom'])

    # Define the cutout; this will not yet trigger any major operations
    file = 'uk-' + str(year)
    time = str(year)
    path = '../data/renewables/atlite/cutouts/' + file
    cutout = atlite.Cutout(path=path,
                           module="era5",
                           bounds=UK.unary_union.bounds,
                           time=time)
    # This is where all the work happens
    # (this can take some time, for us it took ~15 minutes).
    cutout.prepare()
    # projection = ccrs.Orthographic(-10, 35)
    # cells = cutout.grid
    # df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # country_bound = gpd.GeoSeries(cells.unary_union)
    sites = sites.loc[[name]]

    nearest = cutout.data.sel(
        {'x': sites.x.values, 'y': sites.y.values}, 'nearest').coords
    sites['x'] = nearest.get('x').values
    sites['y'] = nearest.get('y').values
    # cells_generation = sites.merge(
    #     cells, how='inner').rename(pd.Series(sites.index))

    layout = cutout.layout_from_capacity_list(sites, col='capacity')

    power_generation = cutout.pv(
        panel="CSi", orientation='latitude_optimal', layout=layout)
    df_power_generation_farm = power_generation.to_pandas()
    df_power_generation_farm.rename(columns={0: name}, inplace=True)

    # fig, axes = plt.subplots(len(sites), sharex=True, figsize=(9, 4))
    # df_power_generation_farm.plot(subplots=True, ax=axes)
    # axes.set_xlabel('date')
    # axes.set_ylabel('Generation [MW]')
    # fig.tight_layout()
    # plt.show()

    return df_power_generation_farm


def multiple_offshore_wind(sites, year):

    print(sites)
    list_outputs = []
    for i in sites.index:
        list_outputs.append(offshore_wind_farm_timeseries(sites, year, name=i))
    df = pd.concat(list_outputs, axis=1, ignore_index=False)
    print(df)

    path = '../data/renewables/atlite/outputs/Wind_Offshore/'

    df.index.name = 'name'
    df.to_csv(path + 'Wind_Offshore_' + str(year) + '.csv', header=True)


def multiple_years_offshore_wind(sites):

    years = list(range(2010, 2020 + 1))
    for y in years:
        multiple_offshore_wind(sites, year=y)


def multiple_onshore_wind(sites, year):

    list_outputs = []
    for i in sites.index:
        list_outputs.append(onshore_wind_farm_timeseries(sites, year, name=i))
    df = pd.concat(list_outputs, axis=1, ignore_index=False)
    print(df)

    path = '../data/renewables/atlite/outputs/Wind_Onshore/'

    df.index.name = 'name'
    df.to_csv(path + 'Wind_Onshore_' + str(year) + '.csv', header=True)


def multiple_years_onshore_wind(sites_year):

    years = list(range(2010, 2020 + 1))
    # want to model 2020 list of sites for all weather years
    sites = onshore_historical_sites(sites_year)
    for y in years:
        multiple_onshore_wind(sites, year=y)


def multiple_PV(sites, year):

    list_outputs = []
    for i in sites.index:
        list_outputs.append(PV_timeseries(sites, year, name=i))
    df = pd.concat(list_outputs, axis=1, ignore_index=False)
    print(df)

    path = '../data/renewables/atlite/outputs/PV/'

    df.index.name = 'name'
    df.to_csv(path + 'PV_' + str(year) + '.csv', header=True)


def multiple_years_PV(sites):

    # empty for 2010, must start at 2011 for PV
    years = list(range(2015, 2020 + 1))
    for y in years:
        multiple_PV(sites, year=y)


def offshore_wind_pipeline_timeseries(sites, year):

    list_outputs = []
    for i in sites.index:
        list_outputs.append(offshore_wind_farm_timeseries(sites, year, name=i))
    df = pd.concat(list_outputs, axis=1, ignore_index=False)

    path = '../data/renewables/atlite/outputs/Wind_Offshore/wind_offshore_pipeline/'

    df.index.name = 'name'
    df.to_csv(
        path + 'wind_offshore_pipeline' + '_' + str(year) + '.csv',
        header=True)


def offshore_wind_future_timeseries(sites, year):

    list_outputs = []
    for i in sites.index:
        list_outputs.append(offshore_wind_farm_timeseries(sites, year, name=i))
    df = pd.concat(list_outputs, axis=1, ignore_index=False)

    path = '../data/renewables/atlite/outputs/Wind_Offshore/wind_offshore_future/'

    df.index.name = 'name'
    df.to_csv(
        path + 'wind_offshore_future' + '_' + str(year) + '.csv',
        header=True)
    

def offshore_wind_floating_timeseries(sites, year):

    list_outputs = []
    for i in sites.index:
        list_outputs.append(offshore_wind_farm_timeseries(sites, year, name=i))
    df = pd.concat(list_outputs, axis=1, ignore_index=False)

    path = '../data/renewables/atlite/outputs/Wind_Offshore/wind_offshore_floating/'

    df.index.name = 'name'
    print(df)
    df.to_csv(
        path + 'wind_offshore_floating' + '_' + str(year) + '.csv',
        header=True)


def offshore_pipeline_sites():

    # want to use PyPSA-GB data on renewable generators in different years
    # lets start with 2019
    file = '../data/renewables/atlite/inputs/offshore_pipeline_2030.csv'
    df = pd.read_csv(file)
    df['capacity'] = df['Turbine Capacity (MW)']
    df.drop(columns=['Turbine Capacity (MW)', 'No. of Turbines'], inplace=True)
    df.rename(columns={'Site Name': 'name'}, inplace=True)
    sites = gpd.GeoDataFrame(df).set_index('name')
    return sites


def offshore_floating_sites():

    # want to use PyPSA-GB data on renewable generators in different years
    # lets start with 2019
    file = '../data/renewables/future_offshore_sites/Sectoral Marine Plan 2020 - Floating.csv'
    df_plan = pd.read_csv(file, encoding='unicode_escape', index_col=0)
    df_plan.rename(columns={'max capacity (GW)': 'Installed Capacity (MWelec)',
                            'Turbine Capacity (MW)': 'capacity',
                            'lon': 'x',
                            'lat': 'y'}, inplace=True)
    df_plan.drop(columns=['area (km2)'], inplace=True)
    df_plan.loc[:, 'Installed Capacity (MWelec)'] *= 1000
    df_plan['capacity'] = 12.
    df_plan['No. of Turbines'] = df_plan['Installed Capacity (MWelec)'] / df_plan['capacity']
    df_plan['No. of Turbines'] = df_plan['No. of Turbines'].astype(int)
    df_plan.index.name = 'name'
    sites = gpd.GeoDataFrame(df_plan)

    return sites


def offshore_historical_sites(year):

    file = '../data/renewables/atlite/inputs/Wind_Offshore/Wind_Offshore_' + str(year) + '.csv'
    df = pd.read_csv(file)
    print(df)
    df['capacity'] = df['Turbine Capacity (MW)']
    df['Installed Capacity (MWelec)'] = df['Turbine Capacity (MW)'] * df['No. of Turbines']
    df.drop(columns=['Turbine Capacity (MW)', 'No. of Turbines'], inplace=True)
    df.rename(columns={'Site Name': 'name'}, inplace=True)
    sites = gpd.GeoDataFrame(df).set_index('name')
    return sites


def offshore_historical_sites_dict():

    years = list(range(2010, 2020 + 1))
    dic = {}
    for y in years:
        dic[y] = offshore_historical_sites(y)
    return dic


def offshore_future_sites():

    df_plan = pd.read_csv('../data/renewables/future_offshore_sites/Sectoral Marine Plan 2020.csv',
                          encoding='unicode_escape', index_col=0)
    df_plan.rename(columns={'max capacity (GW)': 'Installed Capacity (MWelec)',
                            'Turbine Capacity (MW)': 'capacity',
                            'lon': 'x',
                            'lat': 'y'}, inplace=True)
    df_plan.drop(columns=['area (km2)'], inplace=True)
    df_plan.loc[:, 'Installed Capacity (MWelec)'] *= 1000
    df_plan['capacity'] = 12.
    df_plan['No. of Turbines'] = df_plan['Installed Capacity (MWelec)'] / df_plan['capacity']
    df_plan['No. of Turbines'] = df_plan['No. of Turbines'].astype(int)
    df_plan.index.name = 'name'
    sites = gpd.GeoDataFrame(df_plan)

    return sites

def onshore_historical_sites(year):

    file = '../data/renewables/atlite/inputs/Wind_Onshore/Wind_Onshore_' + str(year) + '.csv'
    df = pd.read_csv(file)
    print(df)
    df['capacity'] = df['Turbine Capacity (MW)']
    df['Installed Capacity (MWelec)'] = df['Turbine Capacity (MW)'] * df['No. of Turbines']
    df.drop(columns=['Turbine Capacity (MW)', 'No. of Turbines'], inplace=True)
    df.rename(columns={'Site Name': 'name'}, inplace=True)
    sites = gpd.GeoDataFrame(df).set_index('name')
    return sites


def PV_historical_sites(year):

    file = '../data/renewables/atlite/inputs/Solar_Photovoltaics/Solar_Photovoltaics_' + str(year) + '.csv'
    df = pd.read_csv(file)
    print(df)
    df['capacity'] = df['Installed Capacity (MWelec)']
    df['Installed Capacity (MWelec)'] = df['Turbine Capacity (MW)'] * df['No. of Turbines']
    df.drop(columns=['Turbine Capacity (MW)', 'No. of Turbines'], inplace=True)
    df.rename(columns={'Site Name': 'name'}, inplace=True)
    sites = gpd.GeoDataFrame(df).set_index('name')
    return sites


def PV_historical_sites_dict():

    years = list(range(2010, 2020 + 1))
    dic = {}
    for y in years:
        dic[y] = PV_historical_sites(y)
    return dic


def plot_UK_and_data():

    shpfilename = shpreader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    UK = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                        for r in reader.records()},
                       crs={'init': 'epsg:4326'}).reindex(['United Kingdom'])

    # Define the cutout; this will not yet trigger any major operations
    y = 2019
    path = 'uk-' + str(y)
    time = str(y)
    cutout = atlite.Cutout(path=path,
                           module="era5",
                           bounds=UK.unary_union.bounds,
                           time=time)

    # This is where all the work happens
    # (this can take some time, for us it took ~15 minutes).
    cutout.prepare()

    projection = ccrs.Orthographic(-10, 35)

    cells = cutout.grid
    df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    country_bound = gpd.GeoSeries(cells.unary_union)

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(3, 3, figure=fig)

    ax = fig.add_subplot(gs[:, 0:2], projection=projection)
    plot_grid_dict = dict(alpha=0.1, edgecolor='k', zorder=4, aspect='equal',
                          facecolor='None', transform=plate())
    UK.plot(ax=ax, zorder=1, transform=plate())
    cells.plot(ax=ax, **plot_grid_dict)
    country_bound.plot(ax=ax, edgecolor='orange',
                       facecolor='None', transform=plate())
    ax.outline_patch.set_edgecolor('white')

    ax1 = fig.add_subplot(gs[0, 2])
    cutout.data.wnd100m.mean(['x', 'y']).plot(ax=ax1)
    ax1.set_frame_on(False)
    ax1.xaxis.set_visible(False)

    ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
    cutout.data.influx_direct.mean(['x', 'y']).plot(ax=ax2)
    ax2.set_frame_on(False)
    ax2.xaxis.set_visible(False)

    ax3 = fig.add_subplot(gs[2, 2], sharex=ax1)
    cutout.data.runoff.mean(['x', 'y']).plot(ax=ax3)
    ax3.set_frame_on(False)
    ax3.set_xlabel(None)
    fig.tight_layout()
    plt.show()

    cap_factors = cutout.wind(turbine='Vestas_V112_3MW', capacity_factor=True)

    fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(9, 7))
    cap_factors.name = 'Capacity Factor'
    cap_factors.plot(ax=ax, transform=plate(), alpha=0.8)
    cells.plot(ax=ax, **plot_grid_dict)
    ax.outline_patch.set_edgecolor('white')
    fig.tight_layout()
    plt.show()

    # sites = gpd.GeoDataFrame([['london', 0.7, 51.3, 20],
    #                           ['edinburgh', -3.13, 55.5, 10]],
    #                          columns=['name', 'x', 'y', 'capacity']
    #                          ).set_index('name')

    # want to use PyPSA-GB data on renewable generators in different years
    # lets start with 2019
    file = '../data/renewables/atlite/inputs/offshore_pipeline_2030.csv'
    df = pd.read_csv(file, index_col=False)
    df['capacity'] = df['Turbine Capacity (MW)'] * df['No. of Turbines']
    df.drop(columns=['Turbine Capacity (MW)', 'No. of Turbines'], inplace=True)
    df.rename(columns={'Site Name': 'name'}, inplace=True)
    sites = gpd.GeoDataFrame(df).set_index('name')

    nearest = cutout.data.sel(
        {'x': sites.x.values, 'y': sites.y.values}, 'nearest').coords
    sites['x'] = nearest.get('x').values
    sites['y'] = nearest.get('y').values
    cells_generation = sites.merge(
        cells, how='inner').rename(pd.Series(sites.index))

    layout = cutout.layout_from_capacity_list(sites, col='capacity')

    # layout = xr.DataArray(cells_generation.set_index(['y', 'x']).capacity.unstack())\
    #                     .reindex_like(cap_factors).rename('Installed Capacity [MW]')

    fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(9, 7))

    UK.plot(ax=ax, zorder=1, transform=plate(), alpha=0.3)
    cells.plot(ax=ax, **plot_grid_dict)
    layout.plot(ax=ax, transform=plate(), cmap='Reds', vmin=0,
                label='Installed Capacity [MW]')
    # country_bound.plot(ax=ax, edgecolor='orange',
    #                    facecolor='None', transform=plate())
    ax.outline_patch.set_edgecolor('black')
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(len(sites), sharex=True, figsize=(9, 4))
    power_generation = cutout.wind('Vestas_V112_3MW', layout=layout,
                                   shapes=cells_generation.geometry)

    power_generation.to_pandas().plot(subplots=True, ax=axes)
    axes[1].set_xlabel('date')
    axes[1].set_ylabel('Generation [MW]')
    fig.tight_layout()
    plt.show()


# # want to use PyPSA-GB data on renewable generators in different years
# # lets start with 2019
# tech = 'Wind Onshore'
# year = 2019
# file = 'data/renewables/atlite/inputs/' + tech + '_' + str(year) + '.csv'
# df = pd.read_csv(file, index_col=False)
# df['capacity'] = df['Turbine Capacity (MW)'] * df['No. of Turbines']
# df.drop(columns=['Turbine Capacity (MW)', 'No. of Turbines'], inplace=True)
# df.rename(columns={'Site Name': 'name'}, inplace=True)
# sites = gpd.GeoDataFrame(df
#                          ).set_index('name')

# nearest = cutout.data.sel(
#     {'x': sites.x.values, 'y': sites.y.values}, 'nearest').coords
# sites['x'] = nearest.get('x').values
# sites['y'] = nearest.get('y').values
# cells_generation = sites.merge(
#     cells, how='inner').rename(pd.Series(sites.index))

# layout = cutout.layout_from_capacity_list(df, col='capacity')

# # layout = xr.DataArray(cells_generation.set_index(['y', 'x']).capacity.unstack())\
# #                     .reindex_like(cap_factors).rename('Installed Capacity [MW]')

# fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(9, 7))

# UK.plot(ax=ax, zorder=1, transform=plate(), alpha=0.3)
# plot_grid_dict = dict(alpha=0.1, edgecolor='k', zorder=4, aspect='equal',
#                       facecolor='None', transform=plate())
# cells.plot(ax=ax, **plot_grid_dict)
# layout.plot(ax=ax, transform=plate(), cmap='Reds', vmin=0,
#             label='Installed Capacity [MW]')
# ax.outline_patch.set_edgecolor('white')
# fig.tight_layout()
# plt.show()

# fig, axes = plt.subplots(len(sites), sharex=True, figsize=(9, 4))
# power_generation = cutout.wind('Vestas_V112_3MW', layout=layout,
#                                shapes=cells_generation.geometry)

# power_generation.to_pandas().plot(subplots=True, ax=axes)
# axes[1].set_xlabel('date')
# axes[1].set_ylabel('Generation [MW]')
# fig.tight_layout()
# plt.show()

if __name__ == '__main__':

    # plot_UK_and_data()
    # prepare_cutouts_years()

    # year = 2019
    # sites = offshore_pipeline_sites()
    # print(sites)
    # # sites = offshore_future_sites()
    # # print(sites)
    # offshore_wind_pipeline_timeseries(sites, year)

    year = 2019
    sites = offshore_floating_sites()
    # print(sites)
    # # sites = offshore_future_sites()
    # # print(sites)
    for year in range(2010, 2022 + 1):
        print(year)
        offshore_wind_floating_timeseries(sites, year)

    # # name = 'Hornsea 3'
    # # df = offshore_wind_farm_timeseries(sites, year, name)
    # # print(df)
    # sites_year = 2020
    # sites = offshore_historical_sites(sites_year)
    # # multiple_offshore_wind(sites, year)
    # 

    # year = 2020
    # sites = onshore_historical_sites(year)
    # name = 'Marshill Farm'
    # onshore_wind_farm_timeseries(sites, year, name)
    # year = 2013
    # sites = onshore_historical_sites(year)
    # multiple_onshore_wind(sites, year)

    # 2020 is year which we are building from
    # sites_year = 2020
    # multiple_years_onshore_wind(sites_year)

    # sites_year = 2020
    # sites = PV_historical_sites(sites_year)
    # multiple_years_PV(sites)
