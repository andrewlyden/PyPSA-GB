
import os
import shutil
try:
    import ruamel.yaml
except ImportError:
    pass
import pandas as pd
from . import allocate_to_zone


def create_path():
    path = [
        "../data/ZonesBasedGBsystem/demand/",
        "../data/ZonesBasedGBsystem/interconnectors/",
        "../data/ZonesBasedGBsystem/network/",
    ]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)


def copy_buses_based(year=2021):
    if not os.path.exists("../data/BusesBasedGBsystem/demand/"):
        os.makedirs("../data/BusesBasedGBsystem/demand/")
        shutil.copy(
            "../data/demand/Demand_Distribution.csv",
            "../data/BusesBasedGBsystem/demand/Demand_Distribution.csv",
        )

    if not os.path.exists("../data/BusesBasedGBsystem/Distributions/"):
        shutil.copytree(
            "../data/FES" + str(year) + "/Distributions",
            "../data/BusesBasedGBsystem/Distributions",
        )

    if not os.path.exists("../data/BusesBasedGBsystem/interconnectors/"):
        shutil.copytree(
            "../data/interconnectors", "../data/BusesBasedGBsystem/interconnectors"
        )

    if not os.path.exists("../data/BusesBasedGBsystem/network/"):
        os.makedirs("../data/BusesBasedGBsystem/network/")
        shutil.copy(
            "../data/network/buses.csv", "../data/BusesBasedGBsystem/network/buses.csv"
        )
        shutil.copy(
            "../data/network/GBreducednetwork.m",
            "../data/BusesBasedGBsystem/network/GBreducednetwork.m",
        )
        shutil.copy(
            "../data/network/lines.csv", "../data/BusesBasedGBsystem/network/lines.csv"
        )


def zone_interconnectors():
    if os.path.exists("../data/BusesBasedGBsystem/network/buses.csv"):
        copy_buses_based()
    pd_buses = pd.read_csv("../data/network/buses.csv")

    pd_buses["zone"] = allocate_to_zone.map_to_zone(pd_buses)

    def repalce_to_zone(row, pd_buses):
        out = {}
        bus1 = row["bus1"]
        zone = pd_buses[pd_buses["name"] == bus1]["zone"].tolist()[0]
        out["bus1"] = zone
        return pd.Series(out)

    if os.path.exists("../data/BusesBasedGBsystem/interconnectors/links.csv"):
        copy_buses_based()
    pd_links = pd.read_csv("../data/BusesBasedGBsystem/interconnectors/links.csv")
    pd_links["bus1"] = pd_links.apply(lambda r: repalce_to_zone(r, pd_buses), axis=1)

    if not os.path.exists("../data/ZonesBasedGBsystem/interconnectors/"):
        create_path()
    pd_links.to_csv("../data/ZonesBasedGBsystem/interconnectors/links.csv", index=None)

    if os.path.exists("../data/BusesBasedGBsystem/interconnectors/links.csv"):
        copy_buses_based()
    pd_links_future = pd.read_csv(
        "../data/BusesBasedGBsystem/interconnectors/links.csv"
    )
    pd_links_future["bus1"] = pd_links_future.apply(
        lambda r: repalce_to_zone(r, pd_buses), axis=1
    )

    pd_links_future.to_csv(
        "../data/ZonesBasedGBsystem/interconnectors/links_future.csv", index=None
    )


def zone_buses():
    # Note the buses' lat and lon in this context is just the equivalent Centroid points in each zone.
    file = "../data/network/model.yaml"  # sourcs: uk-calliope project https://github.com/calliope-project/uk-calliope

    with open(file) as stream:
        data = ruamel.yaml.safe_load(stream)

    k = pd.json_normalize(data["locations"])

    # parse yaml file to get bus name, lat and lon
    k.columns = k.columns.str.split(".", expand=True)
    main_df = k.T.unstack()[0]
    df_zonebuses = (
        main_df[["lat", "lon"]][main_df["lon"].notnull()]
        .reset_index()
        .drop("level_1", axis=1)
    )
    df_zonebuses.columns = ["name", "lat", "lon"]

    # change lon->x and lat->y, to match the format used in 29bus pypsa-gb
    df_zonebuses.rename(columns={"lon": "x", "lat": "y"}, inplace=True)

    # add arbitary values to voltage level and carrier columns to buses, to match the format used in 29bus pypsa-gb
    df_zonebuses["v_nom"] = 400
    df_zonebuses["carrier"] = "AC"

    if not os.path.exists("../data/ZonesBasedGBsystem/network/"):
        create_path()
    df_zonebuses.to_csv(
        "../data/ZonesBasedGBsystem/network/buses.csv", index=False, header=True
    )


def zone_links():
    file = "../data/network/transmission_grid_2030.yaml"

    with open(file) as stream:
        data = ruamel.yaml.safe_load(stream)

    k = pd.json_normalize(data)
    k.columns = k.columns.str.split(".", expand=True)
    main_df = k.T.unstack()[0]
    df_zonelinks = main_df[main_df["energy_cap_equals"].notnull()].reset_index()[
        ["level_1", "level_3", "energy_cap_equals"]
    ]
    df_zonelinks.columns = ["name", "carrier", "p_nom"]

    def preprocess(row):
        out = {}
        name = row["name"].split(",")
        out["bus0"] = name[0]
        out["bus1"] = name[1]

        carrier = row["carrier"]
        if carrier == "hvac":
            out["carrier"] = "AC"
        else:
            out["carrier"] = "DC"
        return pd.Series(out)

    df_zonelinks[["bus0", "bus1", "carrier"]] = df_zonelinks.apply(preprocess, axis=1)
    df_zonelinks = df_zonelinks[["name", "bus0", "bus1", "carrier", "p_nom"]]

    df_zonelinks["marginal_cost"] = 0
    df_zonelinks["p_min_pu"] = -1
    df_zonelinks["p_max_pu"] = 1

    if not os.path.exists("../data/ZonesBasedGBsystem/network/"):
        create_path()
    df_zonelinks.to_csv(
        "../data/ZonesBasedGBsystem/network/links.csv", index=False, header=True
    )


if __name__ == "__main__":
    create_path()
    copy_buses_based()
