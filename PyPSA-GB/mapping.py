"""mapping functions

short script for plotting maps related to PyPSA-GB
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature

import imageio

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point

from statistics import mean

import renewables

def generator_map_plotter(tech, color, marker_scaler, year):

    df = renewables.REPD_date_corrected(year)
    df_res = df.loc[df['Technology Type'] == tech].reset_index(drop=True)

    lon = df_res['lon'].values
    lat = df_res['lat'].values

    sizes = df_res['Installed Capacity (MWelec)'].values

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())

    extent = [-8.09782, 2.40511, 60, 49.5]
    ax.set_extent(extent)
    # ax.stock_img()
    # ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.scatter(lon, lat, s=sizes * marker_scaler, c=color, edgecolors='black')
    ax.set_title(tech + ' - ' + str(year), fontsize=16)
    l1 = ax.scatter([], [], s=min(sizes) * marker_scaler, edgecolors='black', color=color)
    l2 = ax.scatter([], [], s=mean(sizes) * marker_scaler, edgecolors='black', color=color)
    l3 = ax.scatter([], [], s=max(sizes) * marker_scaler, edgecolors='black', color=color)

    label1 = round(min(sizes), 0)
    label2 = round(mean(sizes), 0)
    label3 = round(max(sizes), 0)
    labels = [label1, label2, label3]
    ax.legend([l1, l2, l3], labels, frameon=True, fontsize=12,
              loc=1, borderpad=1.5, labelspacing=1.5,
              title='Installed Capacity (MWelec)', scatterpoints=1)
    plt.savefig('../data/renewables/' + str(year) + '_' + tech + '.png')
    # plt.show()

def network_plotter():

    df_network = pd.read_csv('../data/network/buses.csv')[:99]
    lon = df_network['x'].values
    lat = df_network['y'].values
    coordinates = np.zeros(shape=(len(lon), 2))
    # print(coordinates)
    for i in range(len(lon)):
        coordinates[i][0] = lon[i]
        coordinates[i][1] = lat[i]

    # df_load = pd.read_csv('LOPF_data/loads-p_set.csv', index_col=0)[:99]
    # sizes = df_load.sum(axis=0).values
    # marker_scaler = 0.0025

    df_lines = pd.read_csv('../data/network/lines.csv', index_col=0)
    print(df_lines)

    df_buses = pd.read_csv('../data/network/buses.csv', index_col=0)
    print(df_buses[:29].index)
    # print(df_buses['x']['Beauly'])

    line_coordinates = []
    for i in range(len(df_lines['bus0'].values)):

        bus0 = df_lines['bus0'].iloc[i]
        bus1 = df_lines['bus1'].iloc[i]
        bus0_coord = [df_buses['x'][bus0], df_buses['y'][bus0]]
        bus1_coord = [df_buses['x'][bus1], df_buses['y'][bus1]]
        line_coordinates.append([bus0_coord, bus1_coord])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())

    extent = [-8.09782, 2.40511, 60, 49.5]
    ax.set_extent(extent)
    # ax.stock_img()
    # ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')

    # print(line_coordinates[0])
    # print(line_coordinates[0][0][0], line_coordinates[0][1][0])

    for i in range(len(df_lines['bus0'].values)):
        ax.plot([line_coordinates[i][0][0], line_coordinates[i][1][0]],
                [line_coordinates[i][0][1], line_coordinates[i][1][1]],
                c='blue')

    # ax.scatter(lon, lat, s=sizes * marker_scaler, c='black', edgecolors='black')
    ax.scatter(lon, lat, c='black', edgecolors='black')
    for i, txt in enumerate(df_buses[:29].index):
        ax.annotate(txt, (lon[i], lat[i]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center',
                    fontweight='extra bold',
                    color='black',
                    fontsize='small')

    ax.set_title('Reduced Network GB', fontsize=16)

    plt.show()

def gif_maker(tech):

    # tech_ = tech.replace(" ", "_")

    file1 = '../data/renewables/' + '/2011' + '_' + tech + '.png'
    file2 = '../data/renewables/' + '/2012' + '_' + tech + '.png'
    file3 = '../data/renewables/' + '/2013' + '_' + tech + '.png'
    file4 = '../data/renewables/' + '/2014' + '_' + tech + '.png'
    file5 = '../data/renewables/' + '/2015' + '_' + tech + '.png'
    file6 = '../data/renewables/' + '/2016' + '_' + tech + '.png'
    file7 = '../data/renewables/' + '/2017' + '_' + tech + '.png'
    file8 = '../data/renewables/' + '/2018' + '_' + tech + '.png'
    file9 = '../data/renewables/' + '/2019' + '_' + tech + '.png'
    file10 = '../data/renewables/' + '/2020' + '_' + tech + '.png'

    filenames = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]

    with imageio.get_writer('../data/renewables/' + tech + '.gif', mode='I', duration=1.) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == '__main__':

    # tech = 'Advanced Conversion Technology'
    # tech = 'Anaerobic Digestion'
    # tech = 'Biomass (co-firing)'
    # tech = 'Biomass (dedicated)'
    # tech = 'EfW Incineration'
    # tech = 'Fuel Cell (Hydrogen)'
    # tech = 'Hot Dry Rocks (HDR)'
    # tech = 'Landfill Gas'
    # tech = 'Large Hydro'
    # tech = 'Pumped Storage Hydroelectricity'
    # tech = 'Sewage Sludge Digestion'
    # tech = 'Shoreline Wave'
    # tech = 'Small Hydro'
    tech = 'Solar Photovoltaics'
    # tech = 'Tidal Barrage and Tidal Stream'
    # tech = 'Wind Offshore'
    # tech = 'Wind Onshore'
    # tech = 'Biomass (co-firing)'
    color = 'deepskyblue'
    marker_scaler = 2.
    year = 2015
    # generator_map_plotter(tech, color, marker_scaler, year)

    network_plotter()

    # gif_maker(tech)
