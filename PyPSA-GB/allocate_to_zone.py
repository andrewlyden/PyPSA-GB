"""distance calculator
Allocate  points to the zone they belong to based on the divided zones
"""

import json
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon

def load_zone(json_file):
    zones_list = list()
    for i in range(len(json_file['features'])):
        try:
            zone = Polygon(json_file['features'][i]['geometry']['coordinates'][0])
        except:
            polygon_list=list()
            for j in range(len(json_file['features'][i]['geometry']['coordinates'])):
                polygon = Polygon(json_file['features'][i]['geometry']['coordinates'][j][0])
                polygon_list.append(polygon)
            zone = MultiPolygon(polygon_list)
        zones_list.append(zone)
    return zones_list


def map_to_zone(df):
    file_path = '../data/zone/zones_json.geojson'
    json_file = json.loads(open(file_path).read())
    zones_list = load_zone(json_file)

    object_to_zone = []
    for i in range(len(df)):
        data_point = Point(df['x'][i],df['y'][i])
        n = 0
        for j in range(len(json_file['features'])):
            zone = zones_list[j]
            if zone.contains(data_point):
                n += 1
                object_to_zone.append(json_file['features'][j]['properties']['Name_1'])
                
        if (n == 0) & (not np.isnan(df['x'][i])) & (not np.isnan(df['y'][i])):
            print('point {} is not inside any zone, use the nearest zone instead'.format((df['x'][i],df['y'][i])))
            min_poly = min(zones_list, key=data_point.distance)
            index_min_poly = zones_list.index(min_poly)
            object_to_zone.append(json_file['features'][index_min_poly]['properties']['Name_1'])
            
        elif n != 1:
            print('Error while allocated for point {}, set to nan value'.format((df['x'][i],df['y'][i])))
            object_to_zone.append(float('nan'))

    return object_to_zone

if __name__ == "__main__":
    from storage import read_storage_data
    df = read_storage_data(2050)
    df['x'] = df['x']-10
    print(map_to_zone(df))



