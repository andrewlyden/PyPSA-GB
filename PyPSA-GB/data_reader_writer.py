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

# turn off chained assignment errors
pd.options.mode.chained_assignment = None  # default='warn'


def data_writer(start, end, time_step, year, demand_dataset, year_baseline=None, scenario=None, FES=2021, merge_generators=False, scale_to_peak=False):
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
    loads.write_loads_p_set(start, end, year, time_step, demand_dataset, year_baseline=year_baseline, scenario=scenario, FES=FES, scale_to_peak=scale_to_peak)

    generators.write_generators(time_step, year)

    if year > 2020:
        interconnectors.future_interconnectors(year, scenario, FES)
        storage.write_storage_units(year, scenario=scenario, FES=FES)
        generators.future_p_nom(year, time_step, scenario, FES)
        generators.write_generators_p_max_pu(start, end, freq, year, FES, year_baseline=year_baseline, scenario=scenario)
        renewables.add_marine_timeseries(year, year_baseline, scenario, time_step)
        generators.unmet_load()
        # distribution.Distribution(year, scenario).update()
        interconnectors.future_interconnectors(year, scenario, FES)
        if FES == 2022:
            distribution.Distribution(year, scenario).building_block_update()

    elif year <= 2020:
        storage.write_storage_units(year)
        generators.write_generators_p_max_pu(start, end, freq, year, FES)
        interconnectors.write_interconnectors(start, end, freq)

    marginal_costs.write_marginal_costs_series(start, end, freq, year)

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
        data_writer(start, end, time_step, year, demand_dataset='eload', year_baseline=2020, scenario='Leading The Way', merge_generators=True)
    if year <= 2020:
        data_writer(start, end, time_step, year)
