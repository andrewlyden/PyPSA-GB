import pandas as pd
import os

from .utils.cleaning import unify_index


def read_interconnectors():
    """reads the interconnector import/export data from ESPENI database

    Parameters
    ----------

    Returns
    -------
    dataframe
        ESPENI interconnector data, see dates below
    """

    # using espeni data set
    file = "../data/demand/espeni.csv"
    df = pd.read_csv(file)

    dti = pd.date_range(
        start="2008-11-05 22:00:00", end="2021-06-06 23:30:00", freq="0.5h"
    )
    df = df.set_index(dti)

    df = df[
        [
            "POWER_NGEM_BRITNED_FLOW_MW",
            "POWER_NGEM_EAST_WEST_FLOW_MW",
            "POWER_NGEM_MOYLE_FLOW_MW",
            "POWER_NGEM_NEMO_FLOW_MW",
            "POWER_NGEM_IFA_FLOW_MW",
            "POWER_NGEM_IFA2_FLOW_MW",
        ]
    ]

    return df


def write_interconnectors(start, end, freq):

    # add lines which are used to handle flows of interconnectors
    df_IC = pd.read_csv("../data/interconnectors/links.csv", index_col=0)
    df_lines = pd.read_csv("LOPF_data/lines.csv", index_col=0)
    df_IC2 = df_IC.copy()
    df_IC2["r"] = 0.0
    df_IC2["x"] = 0.0001
    df_IC2["b"] = 0.0
    df_IC2["s_nom"] = df_IC2.loc[:, "p_nom"]
    df_IC2.drop(columns=["p_nom", "carrier"], inplace=True)
    df_IC2 = df_IC2.reset_index(drop=True)

    # append interconnectors as lines
    df_lines = pd.concat([df_lines, df_IC2], ignore_index=True, sort=False)

    df_lines.to_csv("LOPF_data/lines.csv", header=True)

    # check if links.csv file exists, if it does then delete it
    filePath = "LOPF_data/links.csv"
    if os.path.exists(filePath):
        os.remove(filePath)
    # check if links.csv file exists, if it does then delete it
    filePath = "UC_data/links.csv"
    if os.path.exists(filePath):
        os.remove(filePath)

    # going to add interconnectors as import -> gen, export -> load
    # first add the interconnectors as named loads
    df_load = pd.read_csv("UC_data/loads.csv")
    name_dic = {
        "name": ["BritNed", "EastWest", "Moyle", "Nemo", "IFA", "IFA2"],
        "bus": "bus",
    }
    IC_load = pd.DataFrame(name_dic)
    df_load = pd.concat([df_load, IC_load], ignore_index=True)

    df_load.to_csv("UC_data/loads.csv", index=False, header=True)

    # then add the interconnectors as named generators with basic attributes
    df_gen = pd.read_csv("UC_data/generators.csv", index_col=0)
    # use hydro row as a template
    IC_gen = df_gen.iloc[[0]]
    result = df_gen
    for i in name_dic["name"]:
        IC_gen.index = [i]
        IC_gen["carrier"] = "Interconnector"
        IC_gen["type"] = "Interconnector"
        IC_gen["p_nom"] = df_IC["p_nom"][i]
        IC_gen["marginal_cost"] = 20
        IC_gen["committable"] = False
        IC_gen["min_up_time"] = 0
        IC_gen["min_down_time"] = 0
        IC_gen["ramp_limit_up"] = 1
        IC_gen["ramp_limit_down"] = 1
        IC_gen["p_min_pu"] = 0
        IC_gen["up_time_before"] = 0
        IC_gen["start_up_cost"] = 0

        result = pd.concat([result, IC_gen])

    # write the appended csv file
    result.index.name = "name"
    result.to_csv("UC_data/generators.csv", header=True)

    # for LOPF need to also provide bus
    IC_names_buses = dict(zip(df_IC.index, df_IC["bus1"].values))
    df_gen2 = pd.read_csv("LOPF_data/generators.csv", index_col=0)
    # use row zero as a template
    IC_gen2 = df_gen2.iloc[[0]]
    df_gen2 = pd.read_csv("LOPF_data/generators.csv", index_col=0)
    result2 = df_gen2
    for i in name_dic["name"]:
        IC_gen2.index = [i]
        IC_gen2["carrier"] = "Interconnector"
        IC_gen2["type"] = "Interconnector"
        IC_gen2["p_nom"] = df_IC["p_nom"][i]
        IC_gen2["bus"] = IC_names_buses[i]
        IC_gen2["marginal_cost"] = 20

        result2 = pd.concat([result2, IC_gen2])

    # write the appended csv file
    result2.index.name = "name"
    result2.to_csv("LOPF_data/generators.csv", header=True)

    # now add the flows as fixed generators and loads
    df_gen_series = pd.read_csv(
        "UC_data/generators-p_max_pu.csv", index_col=0, parse_dates=True
    )

    df = read_interconnectors()
    df = df.loc[start:end]
    df.rename(
        columns={
            "POWER_NGEM_BRITNED_FLOW_MW": "BritNed",
            "POWER_NGEM_EAST_WEST_FLOW_MW": "EastWest",
            "POWER_NGEM_MOYLE_FLOW_MW": "Moyle",
            "POWER_NGEM_NEMO_FLOW_MW": "Nemo",
            "POWER_NGEM_IFA_FLOW_MW": "IFA",
            "POWER_NGEM_IFA2_FLOW_MW": "IFA2",
        },
        inplace=True,
    )

    df, df_gen_series = unify_index([df, df_gen_series], freq)
    df.index = df_gen_series.index

    # exports will be loads
    df_IC_load = df.where(df < 0, 0) * -1

    # set p_min_pu and p_max_pu to per unit of p_nom
    for IC in range(len(df_IC.index)):
        p_nom = df_IC.iloc[IC, 3]
        df.iloc[:, IC] /= p_nom

    # split the exports and imports
    # where imports will be generators
    df_IC_gen = df.where(df > 0, 0)

    result = pd.concat([df_gen_series, df_IC_gen], axis=1)
    # # fix the interconnector flows by setting max and min
    df_IC_gen.to_csv("UC_data/generators-p_min_pu.csv", header=True)
    result.to_csv("UC_data/generators-p_max_pu.csv", header=True)

    # do similar for LOPF
    df_gen_series2 = pd.read_csv(
        "LOPF_data/generators-p_max_pu.csv", index_col=0, parse_dates=True
    )
    result = pd.concat([df_gen_series2, df_IC_gen], axis=1)
    df_IC_gen.to_csv("LOPF_data/generators-p_min_pu.csv", header=True)
    result.to_csv("LOPF_data/generators-p_max_pu.csv", header=True)

    # add the exports as loads
    df_load_series = pd.read_csv(
        "UC_data/loads-p_set.csv", index_col=0, parse_dates=True
    )
    df_load_series, df = unify_index([df_load_series, df], freq)
    df_load_series.index = df.index

    result1 = pd.concat([df_load_series, df_IC_load], axis=1)

    result1.to_csv("UC_data/loads-p_set.csv", header=True)

    # do similar for LOPF
    df_load_series2 = pd.read_csv(
        "LOPF_data/loads-p_set.csv", index_col=0, parse_dates=True
    )
    df_load_series2, df = unify_index([df_load_series2, df], freq)
    df_load_series2.index = df.index
    df_IC_load.rename(
        columns={
            "BritNed": "Netherlands",
            "EastWest": "Ireland",
            "Moyle": "N. Ireland",
            "Nemo": "Belgium",
            "IFA": "France1",
            "IFA2": "France2",
        },
        inplace=True,
    )

    result1 = pd.concat([df_load_series2, df_IC_load], axis=1)

    result1.to_csv("LOPF_data/loads-p_set.csv", header=True)


def future_interconnectors(year, scenario, FES):

    # what interconnectors in future
    # read in future interconnector csv data
    # https://www.ofgem.gov.uk/energy-policy-and-regulation/policy-and-regulatory-programmes/interconnectors
    df_IC_future = pd.read_csv("../data/interconnectors/links_future.csv", index_col=0)
    df_IC_future["p_min_pu"] = -1
    # filter by date
    df = df_IC_future.reset_index().set_index(["installed date"])
    df.index = pd.to_datetime(df.index)
    to_date = str(year) + "-01-01"
    filtered_df = df.loc[:to_date]
    df_IC = filtered_df.set_index(["name"])
    # only scaling if year > 2025
    if year > 2025:
        # what is the 2025 capacity
        IC_2025_cap = df_IC["p_nom"].sum() / 1000
        # need to scale the interconnectors for beyond 2025 using FES data
        # read in the FES data
        if FES == 2021:
            df_FES = pd.read_excel(
                "../data/FES2021/FES 2021 Data Workbook V04.xlsx",
                sheet_name="ES1",
                header=9,
                index_col=1,
            )
        if FES == 2022:
            df_FES = pd.read_excel(
                "../data/FES2022/FES2022 Workbook V4.xlsx",
                sheet_name="ES1",
                header=9,
                index_col=1,
                sheet_name="ES1",  # type: ignore
                header=9,  # type: ignore
                index_col=1,  # type: ignore
            )
        else:
            raise ValueError("FES year not supported")
        df_FES.dropna(axis="rows", inplace=True)
        df_FES = df_FES[df_FES.Type.str.contains("Interconnectors", case=False)]
        df_FES = df_FES[~df_FES.Variable.str.contains(r"\(TWh\)")]
        cols = [0, 1, 2, 3, 4]
        df_FES.drop(df_FES.columns[cols], axis=1, inplace=True)
        date = str(year) + "-01-01"
        try:
            IC_cap_FES = float(df_FES.loc[scenario, date]) / 1000.0
        except:
            IC_cap_FES = float(df_FES.loc[scenario, year]) / 1000.0

        # then consider what scaling factor is required
        scaling_factor = round(IC_cap_FES / IC_2025_cap, 2)
        # print(scaling_factor, 'scaling factor for interconnectors')
        # scale the p_noms of the RES generators
        for g in df_IC.index:
            df_IC.loc[g, "p_nom"] *= scaling_factor
            # gen_tech_UC.loc[g, 'p_nom'] *= scaling_factor

    # save csv for LOPF problem, but not UC which does not include lines??
    df_IC.to_csv("LOPF_data/links.csv", index=True, header=True)

    # try links with UC, but probably need to change buses
    df_IC_UC = df_IC.copy()
    df_IC_UC["bus1"] = "bus"
    df_IC_UC.to_csv("UC_data/links.csv", index=True, header=True)

    # want to ensure that buses.csv has new buses from new interconnectors
    # get original bus file
    df_buses = pd.read_csv("LOPF_data/buses.csv", index_col=0)
    df_buses_DC = df_buses.loc[df_buses["carrier"] == "DC"]
    # compare buses to those of interconnectors
    difference = list(set(df_IC.bus0.values) - set(df_buses_DC.index.values))
    # read in csv file with new bus details
    df_buses_new = pd.read_csv(
        "../data/interconnectors/links_new_buses.csv", index_col=0
    )
    df_buses_new = df_buses_new.dropna(axis="columns")
    df_buses_to_add = df_buses_new.loc[difference]
    # add buses to original buses file
    df_buses = pd.concat([df_buses, df_buses_to_add])

    df_buses.to_csv("LOPF_data/buses.csv", index=True, header=True)

    # want to ensure that buses.csv has new buses from new interconnectors
    # get original bus file
    df_buses = pd.read_csv("UC_data/buses.csv", index_col=0)
    # compare buses to those of interconnectors
    difference = list(set(df_IC.bus0.values) - set(df_buses.index.values))
    # read in csv file with new bus details
    df_buses_new = pd.read_csv(
        "../data/interconnectors/links_new_buses.csv", index_col=0
    )
    df_buses_new = df_buses_new.dropna(axis="columns")
    df_buses_to_add = df_buses_new.loc[difference]
    # add buses to original buses file
    df_buses = pd.concat([df_buses, df_buses_to_add])

    df_buses.to_csv("UC_data/buses.csv", index=True, header=True)

    # check if generators-p_min_pu exists and delete if so
    # used in historical simulations but not wanted in future sims
    try:
        file = "LOPF_data/generators-p_min_pu.csv"
        os.remove(file)
    except Exception:
        pass
    try:
        file = "UC_data/generators-p_min_pu.csv"
        os.remove(file)
    except Exception:
        pass


if __name__ == "__main__":
    year = 2050
    future_interconnectors(year)
