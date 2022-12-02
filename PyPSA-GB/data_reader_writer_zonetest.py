"""
data_reader_writer.py is a script for reading in data and writing into
csv files which can be read by PyPSA

separate folders are populated for both a unit commitment problem and a
network constrained linear optimal power flow problem

at the moment this script is a collection of functions,
but classes could be used to improve readability
"""


import snapshots
import buses
import lines
import storage
import generators
import renewables
import marginal_costs
import loads
import interconnectors
import distribution
import pandas as pd
import os
import shutil

# turn off chained assignment errors
pd.options.mode.chained_assignment = None  # default='warn'
class copy_file():
    def __init__(self, dir1, dir2):
        dlist = os.listdir(dir1)
        if not os.path.exists(dir2):
            os.mkdir(dir2)
        for f in dlist:
            file1 = os.path.join(dir1, f)
            file2 = os.path.join(dir2, f)
            if os.path.isfile(file1):
                shutil.copyfile(file1, file2)
            if os.path.isdir(file1):
                self.__init__(file1,file2)

def data_writer(start, end, time_step, year, year_baseline=None, scenario=None, merge_generators=False, networkmodel=True):
    """writes all the required csv files for UC and LOPF

    Parameters
    ----------
    start : str
        start of simulation period
    end : str
        end of simulation period
    time_step : float
        currently 'H', or '0.5H'
    year : str/int
        year of simulation
    Returns
    -------
    """
    # TODO: 
    # networkmodel = true, busesnework will be used. networkmodel = false, zones model will be used
    if networkmodel:
        copy_file('../data/BusesBasedGBsystem','../data')
        from distance_calculator import map_to_bus as map_to
    else:
        copy_file('../data/ZoneBasedGBsystem','../data')
        from allocate_to_zone import map_to_zones as map_to

    # make sure that end time is in accordance with timestep
    if time_step == 1. or time_step == 'H' or time_step == '1H':
        end = pd.Timestamp(end)
        end = end.replace(minute=0)
        end = str(end)

    freq = snapshots.write_snapshots(start, end, time_step)

    buses.write_buses(year)
    lines.write_lines()
    loads.write_loads(year)
    loads.write_loads_p_set(start, end, year, time_step, year_baseline=year_baseline)
    
    # TODO:
    #     if networkmodel:
    #         import distance_calculator.map_to_bus as dc.map_to_bus
    #     else:
    #         import allocate_to_zone.map_to_zones as dc.map_to_bus
    # the challeges are that imports occur within generators.py and storage.py renewables.py distribution.py
    
    generators.write_generators(time_step, year)

    if year > 2020:
        storage.write_storage_units(year, scenario=scenario)
        generators.future_p_nom(year, time_step, scenario)
        generators.write_generators_p_max_pu(start, end, freq, year, year_baseline=year_baseline, scenario=scenario)
        renewables.add_marine_timeseries(year, year_baseline, scenario, time_step)
        generators.unmet_load()
        distribution.Distribution(year, scenario).update()

    elif year <= 2020:
        storage.write_storage_units(year)
        generators.write_generators_p_max_pu(start, end, freq, year)

    marginal_costs.write_marginal_costs_series(start, end, freq, year)

    if year > 2020:
        interconnectors.future_interconnectors(year)
    elif year <= 2020:
        interconnectors.write_interconnectors(start, end, freq)

    # TODO:
    if networkmodel = false: #for zone model, combine links between zones and linke of interconeectors
        append the links.csv genrated after interconnectors.write_interconnectors with links.csv at zonebasednetwork/network/links.csv

    # merge the non-dispatchable generators at each bus to lower memory requirements
    if merge_generators is True:
        generators.merge_generation_buses(year)



if __name__ == "__main__":

    start = '2050-06-02 00:00:00'
    end = '2050-06-02 23:30:00'
    # year of simulation
    year = int(start[0:4])
    # time step as fraction of hour
    time_step = 0.5
    if year > 2020:
        data_writer(start, end, time_step, year, year_baseline=2020, scenario='Leading The Way', merge_generators=True)
    if year <= 2020:
        data_writer(start, end, time_step, year)
