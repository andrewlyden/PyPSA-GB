import os
import shutil

def create_path():
    path = ['../data/ZonesBasedGBsystem/demand/',
            '../data/ZonesBasedGBsystem/interconnectors/',
            '../data/ZonesBasedGBsystem/network/']
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

def copy_buses_based(year=2021):
    if not os.path.exists('../data/BusesBasedGBsystem/demand/'):
        os.makedirs('../data/BusesBasedGBsystem/demand/')
        shutil.copy('../data/demand/Demand_Distribution.csv', 
                '../data/BusesBasedGBsystem/demand/Demand_Distribution.csv')

    if not os.path.exists('../data/BusesBasedGBsystem/Distributions/'):
        shutil.copytree('../data/FES'+str(year)+'/Distributions', 
                        '../data/BusesBasedGBsystem/Distributions')

    if not os.path.exists('../data/BusesBasedGBsystem/interconnectors/'):
        shutil.copytree('../data/interconnectors', 
                        '../data/BusesBasedGBsystem/interconnectors')

    if not os.path.exists('../data/BusesBasedGBsystem/network/'):
        os.makedirs('../data/BusesBasedGBsystem/network/')
        shutil.copy('../data/network/buses.csv', 
                    '../data/BusesBasedGBsystem/network/buses.csv')
        shutil.copy('../data/network/GBreducednetwork.m', 
                    '../data/BusesBasedGBsystem/network/GBreducednetwork.m')
        shutil.copy('../data/network/lines.csv', 
                    '../data/BusesBasedGBsystem/network/lines.csv')

if __name__ == "__main__":
    create_path()
    copy_buses_based()
