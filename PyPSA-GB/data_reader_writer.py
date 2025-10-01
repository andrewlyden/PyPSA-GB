"""
data_reader_writer.py is a script for reading in data and writing into
csv files which can be read by PyPSA

separate folders are populated for both a unit commitment problem and a
network constrained linear optimal power flow problem

at the moment this script is a collection of functions,
but classes could be used to improve readability

"""

from . import snapshots
from . import buses
from . import lines
from . import storage
from . import generators
from . import renewables
from . import marginal_costs
from . import loads
from . import interconnectors
from . import distribution
from . import marine_scenarios
from . import add_P2G
import pandas as pd
import os
import shutil

# turn off chained assignment errors
pd.options.mode.chained_assignment = None  # default='warn'


class copy_file:
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
                self.__init__(file1, file2)


def data_writer(
    start,
    end,
    time_step,
    year,
    demand_dataset=None,
    year_baseline=None,
    scenario=None,
    FES=2021,
    merge_generators=False,
    scale_to_peak=False,
    networkmodel="Reduced",
    floating_wind_scenario="None",
    wave_scenario="None",
    tidal_stream_scenario="None",
    P2G=False,
):
    """writes all the required csv files for UC and LOPF

    Parameters
    ----------
    start : str
        start of simulation period
    end : str
        end of simulation period
    time_step : float
        currently 'h', or '0.5h'
    year : str/int
        year of simulation
    Returns
    -------
    """

    if year > 2020:
        if year % 4 == 0 and year_baseline % 4 != 0:
            print(
                "Exiting because inputting a simulation leap year and a baseline non-leap year are not compatible..."
            )
            exit()

    if networkmodel == "Reduced":
        copy_file("../data/network/BusesBasedGBsystem", "../data")
    elif networkmodel == "Zonal":
        copy_file("../data/network/ZonesBasedGBsystem", "../data")

    # make sure that end time is in accordance with timestep
    if time_step == 1.0 or time_step == "h" or time_step == "1h":
        end = pd.Timestamp(end)
        end = end.replace(minute=0)
        end = str(end)

    freq = snapshots.write_snapshots(start, end, time_step)

    buses.write_buses(year, networkmodel=networkmodel)
    lines.write_lines(networkmodel)
    loads.write_loads(year)
    loads.write_loads_p_set(
        start,
        end,
        year,
        time_step,
        demand_dataset,
        year_baseline=year_baseline,
        scenario=scenario,
        FES=FES,
        scale_to_peak=scale_to_peak,
        networkmodel=networkmodel,
    )
    generators.write_generators(time_step, year)

    if year > 2020:
        storage.write_storage_units(
            year, scenario=scenario, FES=FES, networkmodel=networkmodel
        )
        generators.future_p_nom(
            year, time_step, scenario, FES, networkmodel=networkmodel
        )

        if floating_wind_scenario != "None":
            marine_scenarios.rewrite_generators_for_marine(
                year, "Floating wind", floating_wind_scenario, networkmodel=networkmodel
            )
        if wave_scenario != "None":
            marine_scenarios.rewrite_generators_for_marine(
                year, "Wave power", wave_scenario, networkmodel=networkmodel
            )
        if tidal_stream_scenario != "None":
            marine_scenarios.rewrite_generators_for_marine(
                year, "Tidal stream", tidal_stream_scenario, networkmodel=networkmodel
            )

        generators.write_generators_p_max_pu(
            start, end, freq, year, FES, year_baseline=year_baseline, scenario=scenario
        )
        renewables.add_marine_timeseries(year, year_baseline, scenario, time_step)
        generators.unmet_load()
        # distribution.Distribution(year, scenario).update()
        if networkmodel == "Reduced":
            interconnectors.future_interconnectors(year, scenario, FES)
        if networkmodel == "Zonal":
            lines.zone_postprocess_generators()
        if FES == 2022:
            distribution.Distribution(
                year, scenario, networkmodel=networkmodel
            ).building_block_update()

    elif year <= 2020:
        storage.write_storage_units(year, networkmodel=networkmodel)
        generators.write_generators_p_max_pu(start, end, freq, year)
        interconnectors.write_interconnectors(start, end, freq)

    marginal_costs.write_marginal_costs_series(start, end, freq, year, FES)

    if P2G is True:
        add_P2G.add_P2G(year, scenario=scenario)
    if networkmodel == "Zonal":
        lines.zone_postprocess_lines_links()

    # merge the non-dispatchable generators at each bus to lower memory requirements
    if merge_generators is True:
        generators.merge_generation_buses(year)


if __name__ == "__main__":

    start = "2050-02-28 00:00:00"
    end = "2050-03-01 23:30:00"
    year = int(start[0:4])
    # time_step = 1.
    # year_baseline = 2012

    # scenario = 'Leading The Way'
    # scenario = 'Consumer Transformation'
    # scenario = 'System Transformation'
    # scenario = 'Steady Progression'

    # data_writer(start, end, time_step, year, demand_dataset='eload', year_baseline=year_baseline,
    #             scenario=scenario, FES=2022, merge_generators=True, scale_to_peak=True,
    #             networkmodel='Reduced', marine_modify=True, marine_scenario='Mid', P2G=False)

    # for scenario in ['Consumer Transformation', 'System Transformation', 'Falling Short']:
    #     FES = 2022
    #     time_step = 1.
    #     year_baseline = 2012
    #     print(scenario)
    #     data_writer(start, end, time_step, year, demand_dataset='eload', year_baseline=year_baseline,
    #                 scenario=scenario, FES=FES, merge_generators=True, scale_to_peak=True,
    #                 networkmodel='Reduced', P2G=True)

    # time step as fraction of hour
    for scenario in [
        "Leading The Way",
        "Consumer Transformation",
        "System Transformation",
        "Falling Short",
    ]:
        for demand_dataset in ["eload", "historical"]:
            for time_step in [1.0, 0.5]:
                for year_baseline in [2012, 2013]:
                    print("inputs:", scenario, demand_dataset, time_step, year_baseline)
                    data_writer(
                        start,
                        end,
                        time_step,
                        year,
                        demand_dataset=demand_dataset,
                        year_baseline=year_baseline,
                        scenario=scenario,
                        scenario=scenario,  # type: ignore
                        FES=2022,
                        merge_generators=True,
                        scale_to_peak=True,
                        networkmodel="Reduced",
                        P2G=True,
                        marine_modify=True,
                        marine_scenario="Mid",
                        floating_wind_scenario="Mid",
                        wave_scenario="Mid",
                        tidal_stream_scenario="Mid",
                    )

    # start = '2040-02-28 00:00:00'
    # end = '2040-03-01 23:30:00'
    # year = int(start[0:4])
    # data_writer(start, end, 0.5, year, demand_dataset='eload',
    #             year_baseline=2013, scenario='Leading The Way', FES=2022,
    #             merge_generators=False, scale_to_peak=True,
    #             networkmodel='Reduced', P2G=True)
