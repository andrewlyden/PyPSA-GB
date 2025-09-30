import numpy as np
import pandas as pd
import os
import shutil


def scotland(path="LOPF_data", replace=False):
    if path[-1] == "/":
        path = path[:-1]
    if replace:
        save_path = path + "/"
    else:
        save_path = path + "_Scotland/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    path = path + "/"
    file_list = os.listdir(path)

    done_list = list()

    bus_list = [
        "Beauly",
        "Peterhead",
        "Errochty",
        "Denny/Bonnybridge",
        "Neilston",
        "Strathaven",
        "Torness",
        "Eccles",
    ]

    if "buses.csv" in file_list:
        csv = pd.read_csv(path + "buses.csv", index_col=0)
        inscot = [_ in bus_list for _ in csv.index.tolist()]
        csv = csv.iloc[inscot]
        bus_list = csv.index.tolist()
        if len(inscot) == 0:
            print("None in scotland in buses.csv.")
        else:
            csv.to_csv(save_path + "buses.csv")
            done_list.append("buses.csv")

    if "loads.csv" in file_list:
        csv = pd.read_csv(path + "loads.csv", index_col=0)
        inscot = [_ in bus_list for _ in csv.index.tolist()]
        csv = csv.iloc[inscot]
        if len(inscot) == 0:
            print("None in scotland in loads.csv.")
        else:
            csv.to_csv(save_path + "loads.csv")
            done_list.append("loads.csv")

    if "generators.csv" in file_list:
        csv = pd.read_csv(path + "generators.csv", index_col=0)
        inscot = [_ in bus_list for _ in csv["bus"].tolist()]
        csv = csv.iloc[inscot]
        generators_list = csv.index.tolist()
        if len(inscot) == 0:
            print("None in scotland in generators.csv.")
        else:
            csv.to_csv(save_path + "generators.csv")
            done_list.append("generators.csv")

    if "storage_units.csv" in file_list:
        csv = pd.read_csv(path + "storage_units.csv", index_col=0)
        inscot = [_ in bus_list for _ in csv["bus"].tolist()]
        csv = csv.iloc[inscot]
        storage_units_list = csv.index.tolist()
        if len(inscot) == 0:
            print("None in scotland in storage_units.csv.")
        else:
            csv.to_csv(save_path + "storage_units.csv")
            done_list.append("storage_units.csv")

    file = [
        "generators-marginal_cost.csv",
        "generators-p_max_pu.csv",
        "generators-p_min_pu.csv",
    ]

    for f in file:
        if f in file_list:
            csv = pd.read_csv(path + f, index_col=0)
            inscot = [_ for _ in csv.columns.tolist() if _ in generators_list]
            csv = csv[inscot]
            if len(inscot) == 0:
                print("None in scotland in {}.".format(f))
            else:
                csv.to_csv(save_path + f)
                done_list.append(f)

    if "loads-p_set.csv" in file_list:
        csv = pd.read_csv(path + "loads-p_set.csv", index_col=0)
        inscot = [_ for _ in csv.columns.tolist() if _ in bus_list]
        csv = csv[inscot]
        if len(inscot) == 0:
            print("None in scotland in loads-p_set.csv")
        else:
            csv.to_csv(save_path + "loads-p_set.csv")
            done_list.append("loads-p_set.csv")

    if "links.csv" in file_list:
        csv = pd.read_csv(path + "links.csv", index_col=0)
        inscot0 = [_ in bus_list for _ in csv["bus0"].tolist()]
        inscot1 = [_ in bus_list for _ in csv["bus1"].tolist()]
        inscot = [i for i in range(len(inscot0)) if inscot0[i] if inscot1[i]]

        for i in range(len(inscot0)):
            n = 0
            if inscot0[i]:
                n += 1
            if inscot1[i]:
                n += 1
            if n == 1:
                inscot.append(i)
        inscot = list(set(inscot))
        csv = csv.iloc[inscot]
        if "NorthConnect" in csv.index:
            csv.drop("NorthConnect", inplace=True)
        if len(inscot) == 0:
            print("None in scotland in links.csv.")
        else:
            csv.to_csv(save_path + "links.csv")
            done_list.append("links.csv")

    if "lines.csv" in file_list:
        csv = pd.read_csv(path + "lines.csv", index_col=0)
        inscot0 = [_ in bus_list for _ in csv["bus0"].tolist()]
        inscot1 = [_ in bus_list for _ in csv["bus1"].tolist()]
        inscot = [i for i in range(len(inscot0)) if inscot0[i] if inscot1[i]]
        csv = csv.iloc[inscot]
        if len(inscot) == 0:
            print("None in scotland in lines.csv.")
        else:
            csv.to_csv(save_path + "lines.csv")
            done_list.append("lines.csv")
        index = list()
        for i in range(len(inscot0)):
            n = 0
            if inscot0[i]:
                n += 1
            if inscot1[i]:
                n += 1
            if n == 1:
                index.append(i)
        if len(index) == 0:
            print("No additional links from lines.")
        else:
            pd_lines = pd.read_csv(path + "lines.csv", index_col=0).iloc[index]
            pd_links = pd.DataFrame(
                columns=[
                    "name",
                    "bus0",
                    "bus1",
                    "carrier",
                    "p_nom",
                    "marginal_cost",
                    "p_min_pu",
                ]
            )

            def scot_links(row):
                out = {}
                bus0 = row["bus0"]
                bus1 = row["bus1"]
                out["name"] = bus0 + "-" + bus1
                out["bus0"] = bus0
                out["bus1"] = bus1
                out["carrier"] = "DC"
                out["p_nom"] = row["s_nom"]
                out["marginal_cost"] = 0
                out["p_min_pu"] = -1
                return pd.Series(out)

            pd_links = pd_lines.apply(scot_links, axis=1)
            pd_links.set_index(["name"], inplace=True)
            if os.path.exists(save_path + "links.csv"):
                pd_links0 = pd.read_csv(save_path + "links.csv", index_col=0)
                pd_links = pd.concat([pd_links0, pd_links])
            pd_links.to_csv(save_path + "links.csv")
            done_list.append("links.csv (from lines)")

    if "snapshots.csv" in file_list:
        shutil.copy(path + "snapshots.csv", save_path + "snapshots.csv")
        done_list.append("snapshots.csv")

    print("Processed: " + ", \n".join(done_list))

    save_path_file = os.listdir(save_path)
    if "links.csv (from lines)" in done_list:
        done_list.append("links.csv")

    for f in [f for f in save_path_file if f not in done_list]:
        os.remove(save_path + f)


def interconnector(path="LOPF_data_Scotland/"):
    if path[-1] != "/":
        path = path + "/"
    pd_links = pd.read_csv(path + "links.csv", index_col=0)
    pd_generators = pd.read_csv(path + "generators.csv", index_col=0)

    def connector_to_generator(row, pd_generators):
        name = row.name
        bus0 = row["bus0"]
        bus1 = row["bus1"]
        p_nom = row["p_nom"]
        if name == str(bus0 + "-" + bus1):
            carrier = "Englandconnector"
        else:
            carrier = "Interconnector"
        pd_generators.loc[name, "carrier"] = carrier
        if np.isnan(pd_generators.loc[name, "p_nom"]):
            pd_generators.loc[name, "p_nom"] = p_nom
        else:
            pd_generators.loc[name, "p_nom"] += p_nom

    pd_links.apply(lambda r: connector_to_generator(r, pd_generators), axis=1)
    pd_generators.to_csv(path + "generators.csv")


if __name__ == "__main__":
    scotland()
    interconnector()
