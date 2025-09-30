import pandas as pd
import numpy as np
import os

import osgb

from . import renewables


def renewable_data_write_atlite(year, tech):

    df = renewables.REPD_date_corrected(year)
    df_res = df.loc[df["Technology Type"] == tech].reset_index(drop=True)
    df_res.drop(
        columns=[
            "Technology Type",
            "CHP Enabled",
            "lon",
            "lat",
            "Development Status",
            "Operational",
            "Mounting Type for Solar",
            "Height of Turbines (m)",
        ],
        inplace=True,
    )

    # convert from OSGB to lat/lon
    lon = []
    lat = []
    for i in range(len(df_res.index)):
        x = df_res["X-coordinate"][i]
        y = df_res["Y-coordinate"][i]
        coord = osgb.grid_to_ll(x, y)
        lat.append(coord[0])
        lon.append(coord[1])
    df_res["x"] = lon
    df_res["y"] = lat
    df_res.drop(columns=["X-coordinate", "Y-coordinate"], inplace=True)
    # Directory
    directory = tech.replace(" ", "_")
    # Parent Directory path
    parent_dir = "../data/renewables/atlite/inputs/"
    # Path
    path = os.path.join(parent_dir, directory)
    try:
        os.mkdir(path)
    except FileExistsError:
        # directory already exists
        pass

    df_res.to_csv(
        path + "/" + tech.replace(" ", "_") + "_" + str(year) + ".csv",
        header=True,
        index=False,
    )


def write_data_all_tech_and_years():

    tech_list = ["Solar Photovoltaics"]
    years = list(range(2010, 2020 + 1))
    for t in tech_list:
        for y in years:
            renewable_data_write_atlite(y, t)


def offshore_wind_pipeline(year):

    # pipeline data
    df_pipeline = pd.read_csv(
        "../data/renewables/future_offshore_sites/offshore_pipeline.csv",
        encoding="unicode_escape",
        index_col=2,
    )

    df_pipeline.drop(
        columns=[
            "Record Last Updated (dd/mm/yyyy)",
            "Operator (or Applicant)",
            "Under Construction",
            "Technology Type",
            "Planning Permission Expired",
            "Operational",
            "Heat Network Ref",
            "Planning Authority",
            "Planning Application Submitted",
            "Region",
            "Country",
            "County",
            "Development Status",
            "Development Status (short)",
        ],
        inplace=True,
    )
    df_pipeline.dropna(axis="columns", inplace=True)

    # pipeline up to 2030, but still add in pipeline after 2030
    if year > 2030:
        year = 2030
    # lets look at pipeline output for these years
    df = df_pipeline["Expected Operational"]
    df = pd.to_datetime(df).dt.to_period("D")

    date = "31/12/" + str(year)
    df2 = df_pipeline[~(df > date)]
    # convert from OSGB to lat/lon
    lon = []
    lat = []
    for i in range(len(df2.index)):
        x = df2["X-coordinate"][i]
        y = df2["Y-coordinate"][i]
        coord = osgb.grid_to_ll(x, y)
        lat.append(coord[0])
        lon.append(coord[1])
    df2["x"] = lon
    df2["y"] = lat
    df2.drop(columns=["X-coordinate", "Y-coordinate"], inplace=True)
    print(df2)
    df2.to_csv(
        "../data/renewables/atlite/inputs/offshore_pipeline_" + str(year) + ".csv",
        header=True,
        index=True,
    )


def offshore_wind_scotland_planned():
    df_plan = pd.read_csv(
        "../data/renewables/future_offshore_sites/Sectoral Marine Plan 2020.csv",
        encoding="unicode_escape",
        index_col=0,
    )
    df_plan.rename(
        columns={
            "max capacity (GW)": "Installed Capacity (MWelec)",
            "lon": "x",
            "lat": "y",
        },
        inplace=True,
    )
    df_plan.drop(columns=["area (km2)"], inplace=True)
    df_plan.loc[:, "Installed Capacity (MWelec)"] *= 1000
    df_plan["Turbine Capacity (MW)"] = 12.0
    df_plan["No. of Turbines"] = (
        df_plan["Installed Capacity (MWelec)"] / df_plan["Turbine Capacity (MW)"]
    )
    df_plan["No. of Turbines"] = df_plan["No. of Turbines"].astype(int)
    df_plan.index.name = "Site Name"
    print(df_plan)


if __name__ == "__main__":
    # year = 2020
    # tech = 'Wind Onshore'
    # renewable_data_write_atlite(year, tech)

    write_data_all_tech_and_years()

    # year = 2030
    # offshore_wind_pipeline(year)

    # offshore_wind_scotland_planned()
