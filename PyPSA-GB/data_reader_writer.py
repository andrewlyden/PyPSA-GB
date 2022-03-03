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

from utils.cleaning import unify_snapshots
# turn off chained assignment errors
pd.options.mode.chained_assignment = None  # default='warn'


def data_writer(start, end, time_step, year, year_baseline=None, scenario=None):
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
    # renewables.aggregate_renewable_generation(start, end, year, time_step)

    if year > 2020:
        interconnectors.future_interconnectors(year)
    elif year <= 2020:
        interconnectors.write_interconnectors(start, end, freq)

    if year <= 2020:
        unify_snapshots('snapshots.csv',
                        ['generators-marginal_cost.csv',
                         'generators-p_max_pu.csv',
                         'generators-p_min_pu.csv',
                         'loads-p_set.csv'],
                        'LOPF_data')

    '''
    elif year > 2020:
        unify_snapshots('snapshots.csv', [
                                    'generators-marginal_cost.csv',
                                    'generators-p_max_pu.csv',
                                    'loads-p_set.csv'
                                    ], 'LOPF_data')
    '''


if __name__ == "__main__":

    start = '2050-12-02 00:00:00'
    end = '2050-12-02 03:30:00'
    # year of simulation
    year = int(start[0:4])
    # time step as fraction of hour
    time_step = 0.5
    if year > 2020:
        data_writer(start, end, time_step, year, year_baseline=2020, scenario='Leading The Way')
    if year <= 2020:
        data_writer(start, end, time_step, year)
