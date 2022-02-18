"""distance calculator

short script which contains functions for calculating distance between
coordinates and for mapping locations to buses
"""

from math import cos, asin, sqrt
import pandas as pd


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))


def closest(data, v):
    return min(data, key=lambda p: distance(v['lat'], v['lon'], p['lat'], p['lon']))

def map_to_bus(df):

    # for the LOPF want to map the storage units to the closest bus
    # read buses data
    df_buses = pd.read_csv('LOPF_data/buses.csv')
    bus_location = []
    for i in range(len(df_buses)):
        bus_location.append({'lon': df_buses['x'][i], 'lat': df_buses['y'][i]})
    bus_names = df_buses['name'].values

    object_to_bus = []
    for i in range(len(df)):
        data_point = {'lon': df['x'][i], 'lat': df['y'][i]}
        closest_bus_location = closest(bus_location, data_point)
        closest_bus_index = bus_location.index(closest_bus_location)
        object_to_bus.append(bus_names[closest_bus_index])

    return object_to_bus


if __name__ == "__main__":

    tempDataList = [{'lat': 40.7612992, 'lon': -86.1519681},
                    {'lat': 50.922412, 'lon': -86.1584361},
                    {'lat': 38.7622292, 'lon': -86.1578917}]

    v = {'lat': 39.7622290, 'lon': -86.1519750}
    print(closest(tempDataList, v))
