# distributions can be changed via this script
# using FES2021 data to do this.

import pandas as pd
from . import distance_calculator as dc


class Distribution(object):

    def __init__(self, year, scenario, networkmodel="Reduced"):
        path = "LOPF_data/generators.csv"
        self.df_generators = pd.read_csv(path, index_col=0)

        path = "LOPF_data/storage_units.csv"
        self.df_storage = pd.read_csv(path, index_col=0)

        path = "LOPF_data/loads-p_set.csv"
        self.df_loads = pd.read_csv(path, index_col=0)

        # path = 'LOPF_data/links.csv'
        # self.df_interconnector = pd.read_csv(path, index_col=0)

        self.year = year
        self.scenario = scenario

        if networkmodel == "Reduced":
            self.buses_scotland = [
                "Beauly",
                "Peterhead",
                "Errochty",
                "Denny/Bonnybridge",
                "Neilston",
                "Strathaven",
                "Torness",
                "Eccles",
            ]

            self.buses_rgb = [
                "Harker",
                "Stella West",
                "Penwortham",
                "Deeside",
                "Daines",
                "Th. Marsh/Stocksbridge",
                "Thornton/Drax/Eggborough",
                "Keadby",
                "Ratcliffe",
                "Feckenham",
                "Walpole",
                "Bramford",
                "Pelham",
                "Sundon/East Claydon",
                "Melksham",
                "Bramley",
                "London",
                "Kemsley",
                "Sellindge",
                "Lovedean",
                "S.W.Penisula",
            ]
        elif networkmodel == "Zonal":
            self.buses_scotland = [
                "Z1_1",
                "Z1_2",
                "Z1_3",
                "Z1_4",
                "Z2",
                "Z3",
                "Z4",
                "Z5",
                "Z6",
            ]

            self.buses_rgb = [
                "Z7",
                "Z8",
                "Z9",
                "Z10",
                "Z11",
                "Z12",
                "Z13",
                "Z14",
                "Z15",
                "Z16",
                "Z17",
            ]

        # if scenario == 'Leading the Way':
        #     self.scenario = 'LW'
        # elif scenario == 'Consumer Transformation':
        #     self.scenario = 'CT'
        # elif scenario == 'System Transformation':
        #     self.scenario = 'ST'
        # elif scenario == 'Steady Progression':
        #     self.scenario = 'SP'
        # else:
        #     raise NameError('Invalid scenario passed to Distribution class')

        self.df_FES_bb = pd.read_excel(
            "../data/FES2022/FES2022 Workbook V4.xlsx", sheet_name="BB1"
        )
        self.df_id = pd.read_excel(
            "../data/FES2022/Building Block Definitions.xlsx", index_col=0
        )
        self.df_gsp = pd.read_csv(
            "../data/FES2022/GSP in Scotland.csv", encoding="cp1252"
        )

    def read_building_block_scotland(self, tech):

        if tech == "Hydro":
            ids = ["Gen_BB018"]
        elif tech == "Hydrogen":
            ids = ["Gen_BB023"]
        elif tech == "Natural Gas":
            ids = ["Gen_BB008", "Gen_BB009"]
        elif tech == "Batteries":
            ids = ["Srg_BB001"]
        elif tech == "Domestic Batteries":
            ids = ["Srg_BB002"]
        elif tech == "Pumped Hydro":
            ids = ["Srg_BB003"]
        elif tech == "Other":
            ids = ["Srg_BB004"]
        elif tech == "V2G":
            ids = ["Srg_BB005"]
        else:
            ids = self.df_id.filter(like=tech, axis=0)[
                "Building Block ID Number"
            ].values

        df_FES = self.df_FES_bb[
            self.df_FES_bb["FES Scenario"].str.contains(
                self.scenario, case=False, na=False
            )
        ]
        GSP_list = self.df_gsp["GSP"].values
        df_FES = df_FES[df_FES["GSP"].isin(GSP_list)]
        df_FES = df_FES[df_FES["Building Block ID Number"].isin(ids)]
        return df_FES

    def read_building_block_rgb(self, tech):

        if tech == "Hydro":
            ids = ["Gen_BB018"]
        elif tech == "Hydrogen":
            ids = ["Gen_BB023"]
        elif tech == "Natural Gas":
            ids = ["Gen_BB008", "Gen_BB009"]
        elif tech == "Batteries":
            ids = ["Srg_BB001"]
        elif tech == "Domestic Batteries":
            ids = ["Srg_BB002"]
        elif tech == "Pumped Hydro":
            ids = ["Srg_BB003"]
        elif tech == "Other":
            ids = ["Srg_BB004"]
        elif tech == "V2G":
            ids = ["Srg_BB005"]
        else:
            ids = self.df_id.filter(like=tech, axis=0)[
                "Building Block ID Number"
            ].values

        df_FES = self.df_FES_bb[
            self.df_FES_bb["FES Scenario"].str.contains(
                self.scenario, case=False, na=False
            )
        ]
        GSP_list = self.df_gsp["GSP"].values
        df_FES = df_FES[~df_FES["GSP"].isin(GSP_list)]
        df_FES = df_FES[df_FES["Building Block ID Number"].isin(ids)]
        return df_FES

    def scotland_total_tech(self, tech):
        df = self.read_building_block_scotland(tech)
        # print(df)
        return df[self.year].sum()

    def rgb_total_tech(self, tech):
        df = self.read_building_block_rgb(tech)
        # print(df)
        return df[self.year].sum()

    def generation_capacities_scotland(self):

        generation_caps = {
            "Marine": self.scotland_total_tech("Marine"),
            "Biomass": self.scotland_total_tech(
                "Biomass & Energy Crops (including CHP)"
            ),
            "Interconnector": self.scotland_total_tech("Interconnector"),
            "Natural Gas": self.scotland_total_tech("Natural Gas"),
            "Nuclear": self.scotland_total_tech("Nuclear"),
            "Hydrogen": self.scotland_total_tech("Hydrogen"),
            "Hydro": self.scotland_total_tech("Hydro"),
            "Solar Photovoltaics": self.scotland_total_tech("Solar Generation"),
            "Wind Onshore": self.scotland_total_tech("Wind Onshore"),
            "Wind Offshore": self.scotland_total_tech("Wind Offshore"),
        }
        return generation_caps

    def generation_capacities_rgb(self):

        generation_caps = {
            "Marine": self.rgb_total_tech("Marine"),
            "Biomass": self.rgb_total_tech("Biomass & Energy Crops (including CHP)"),
            "Interconnector": self.rgb_total_tech("Interconnector"),
            "Natural Gas": self.rgb_total_tech("Natural Gas"),
            "Nuclear": self.rgb_total_tech("Nuclear"),
            "Hydrogen": self.rgb_total_tech("Hydrogen"),
            "Hydro": self.rgb_total_tech("Hydro"),
            "Solar Photovoltaics": self.rgb_total_tech("Solar Generation"),
            "Wind Onshore": self.rgb_total_tech("Wind Onshore"),
            "Wind Offshore": self.rgb_total_tech("Wind Offshore"),
        }
        return generation_caps

    def storage_capacities_scotland(self):

        generation_caps = {
            "Batteries": self.scotland_total_tech("Batteries"),
            "Domestic Batteries": self.scotland_total_tech("Domestic Batteries"),
            "Pumped Hydro": self.scotland_total_tech("Pumped Hydro"),
            "Other": self.scotland_total_tech("Other"),
            "V2G": self.scotland_total_tech("V2G"),
        }
        return generation_caps

    def storage_capacities_rgb(self):

        generation_caps = {
            "Batteries": self.rgb_total_tech("Batteries"),
            "Domestic Batteries": self.rgb_total_tech("Domestic Batteries"),
            "Pumped Hydro": self.rgb_total_tech("Pumped Hydro"),
            "Other": self.rgb_total_tech("Other"),
            "V2G": self.rgb_total_tech("V2G"),
        }
        return generation_caps

    def interconnector_capacities_scotland(self):

        generation_caps = {"Interconnector": self.scotland_total_tech("Interconnector")}
        return generation_caps

    def interconnector_capacities_rgb(self):

        generation_caps = {"Interconnector": self.rgb_total_tech("Interconnector")}
        return generation_caps

    def read_scotland_generators(self):

        buses_scotland = self.buses_scotland
        # select generators in the buses in Scotland
        df_generators = self.df_generators[self.df_generators.bus.isin(buses_scotland)]
        generators_p_nom = (
            df_generators.p_nom.groupby(df_generators.carrier).sum().sort_values()
        )
        try:
            generators_p_nom.drop("Unmet Load", inplace=True)
        except:
            pass
        # generators_p_nom.drop(generators_p_nom[generators_p_nom < 50].index, inplace=True)
        return generators_p_nom

    def read_rgb_generators(self):

        buses_rgb = self.buses_rgb  # select generators in the buses in Scotland
        df_generators = self.df_generators[self.df_generators.bus.isin(buses_rgb)]
        generators_p_nom = (
            df_generators.p_nom.groupby(df_generators.carrier).sum().sort_values()
        )
        try:
            generators_p_nom.drop("Unmet Load", inplace=True)
        except:
            pass
        # generators_p_nom.drop(generators_p_nom[generators_p_nom < 50].index, inplace=True)
        return generators_p_nom

    def read_scotland_interconnector(self):

        buses_scotland = self.buses_scotland
        # select generators in the buses in Scotland
        df_interconnector = self.df_interconnector[
            self.df_interconnector.bus1.isin(buses_scotland)
        ]
        interconnector_p_nom = (
            df_interconnector.p_nom.groupby(df_interconnector.carrier)
            .sum()
            .sort_values()
        )
        return interconnector_p_nom

    def read_rgb_interconnector(self):

        buses_rgb = self.buses_rgb  # select generators in the buses in Scotland
        df_interconnector = self.df_interconnector[
            self.df_interconnector.bus1.isin(buses_rgb)
        ]
        interconnector_p_nom = (
            df_interconnector.p_nom.groupby(df_interconnector.carrier)
            .sum()
            .sort_values()
        )
        return interconnector_p_nom

    def read_scotland_storage(self):

        buses_scotland = self.buses_scotland
        # select storage in the buses in Scotland
        df_storage = self.df_storage[self.df_storage.bus.isin(buses_scotland)]
        storage_p_nom = df_storage.p_nom.groupby(df_storage.carrier).sum().sort_values()
        # storage_p_nom.drop(storage_p_nom[storage_p_nom < 50].index, inplace=True)
        return storage_p_nom

    def read_rgb_storage(self):

        buses_rgb = self.buses_rgb
        # select storage in the buses in Scotland
        df_storage = self.df_storage[self.df_storage.bus.isin(buses_rgb)]
        storage_p_nom = df_storage.p_nom.groupby(df_storage.carrier).sum().sort_values()
        # storage_p_nom.drop(storage_p_nom[storage_p_nom < 50].index, inplace=True)
        return storage_p_nom

    def modify_generators(self):

        # modify the generators using the buildings block data
        # scales generation to match Scotland and rest of GB separately

        buses_scotland = self.buses_scotland
        # rgb is rest of GB
        buses_rgb = self.buses_rgb
        # generation from unmodified distribution
        generators_p_nom_scotland = self.read_scotland_generators()
        # generation according to building blocks data
        generators_p_nom_bb_scotland = self.generation_capacities_scotland()

        # generation from unmodified distribution
        generators_p_nom_rgb = self.read_rgb_generators()
        # generation according to building blocks data
        generators_p_nom_bb_rgb = self.generation_capacities_rgb()

        # Scale offshore wind
        offshore_wind_unmodified_scotland = generators_p_nom_scotland["Wind Offshore"]
        offshore_wind_bb_scotland = generators_p_nom_bb_scotland["Wind Offshore"]
        scaling_factor_offshore_scotland = (
            offshore_wind_unmodified_scotland / offshore_wind_bb_scotland
        )

        offshore_wind_unmodified_rgb = generators_p_nom_rgb["Wind Offshore"]
        offshore_wind_bb_rgb = generators_p_nom_bb_rgb["Wind Offshore"]
        scaling_factor_offshore_rgb = (
            offshore_wind_unmodified_rgb / offshore_wind_bb_rgb
        )

        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Offshore') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Offshore') & (self.df_generators.bus == bus), "p_nom"])
        for bus in buses_scotland:
            self.df_generators.loc[
                (self.df_generators.carrier == "Wind Offshore")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_offshore_scotland
        for bus in buses_rgb:
            self.df_generators.loc[
                (self.df_generators.carrier == "Wind Offshore")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_offshore_rgb
        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Offshore') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Offshore') & (self.df_generators.bus == bus), "p_nom"])

        # Scale onshore wind
        onshore_wind_unmodified_scotland = generators_p_nom_scotland["Wind Onshore"]
        onshore_wind_bb_scotland = generators_p_nom_bb_scotland["Wind Onshore"]
        scaling_factor_onshore_scotland = (
            onshore_wind_unmodified_scotland / onshore_wind_bb_scotland
        )

        onshore_wind_unmodified_rgb = generators_p_nom_rgb["Wind Onshore"]
        onshore_wind_bb_rgb = generators_p_nom_bb_rgb["Wind Onshore"]
        scaling_factor_onshore_rgb = onshore_wind_unmodified_rgb / onshore_wind_bb_rgb

        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Onshore') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Onshore') & (self.df_generators.bus == bus), "p_nom"])
        for bus in buses_scotland:
            self.df_generators.loc[
                (self.df_generators.carrier == "Wind Onshore")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_onshore_scotland
        for bus in buses_rgb:
            self.df_generators.loc[
                (self.df_generators.carrier == "Wind Onshore")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_onshore_rgb
        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Onshore') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Wind Onshore') & (self.df_generators.bus == bus), "p_nom"])

        # Scale PV
        PV_unmodified_scotland = generators_p_nom_scotland["Solar Photovoltaics"]
        PV_bb_scotland = generators_p_nom_bb_scotland["Solar Photovoltaics"]
        scaling_factor_PV_scotland = PV_unmodified_scotland / PV_bb_scotland

        PV_unmodified_rgb = generators_p_nom_rgb["Solar Photovoltaics"]
        PV_bb_rgb = generators_p_nom_bb_rgb["Solar Photovoltaics"]
        scaling_factor_PV_rgb = PV_unmodified_rgb / PV_bb_rgb

        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Solar Photovoltaics') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Solar Photovoltaics') & (self.df_generators.bus == bus), "p_nom"])
        for bus in buses_scotland:
            self.df_generators.loc[
                (self.df_generators.carrier == "Solar Photovoltaics")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_PV_scotland
        for bus in buses_rgb:
            self.df_generators.loc[
                (self.df_generators.carrier == "Solar Photovoltaics")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_PV_rgb
        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Solar Photovoltaics') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Solar Photovoltaics') & (self.df_generators.bus == bus), "p_nom"])

        # Move nuclear to England
        if self.year < 2028:
            nuclear_unmodified_scotland = generators_p_nom_scotland["Nuclear"]
            nuclear_bb_scotland = generators_p_nom_bb_scotland["Nuclear"]
            if nuclear_bb_scotland > 0.0:
                scaling_factor_nuclear_scotland = (
                    nuclear_bb_scotland / nuclear_unmodified_scotland
                )
            else:
                scaling_factor_nuclear_scotland = 0
        else:
            scaling_factor_nuclear_scotland = 0

        nuclear_unmodified_rgb = generators_p_nom_rgb["Nuclear"]
        nuclear_bb_rgb = generators_p_nom_bb_rgb["Nuclear"]
        scaling_factor_nuclear_rgb = nuclear_unmodified_rgb / nuclear_bb_rgb

        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Nuclear') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Nuclear') & (self.df_generators.bus == bus), "p_nom"])
        for bus in buses_scotland:
            self.df_generators.loc[
                (self.df_generators.carrier == "Nuclear")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] *= scaling_factor_nuclear_scotland
        for bus in buses_rgb:
            self.df_generators.loc[
                (self.df_generators.carrier == "Nuclear")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_nuclear_rgb
        # bus = 'Beauly'
        # print(self.df_generators.loc[(self.df_generators.carrier == 'Nuclear') & (self.df_generators.bus == bus), "p_nom"])
        # bus = 'Harker'
        # # print(self.df_generators.loc[(self.df_generators.carrier == 'Nuclear') & (self.df_generators.bus == bus), "p_nom"])
        # if self.year >= 2028:
        #     # Scale and move CCS gas to Peterhead bus
        #     CCS_gas_unmodified_scotland = generators_p_nom_scotland['CCS Gas']
        #     CCS_gas_bb_scotland = generators_p_nom_bb_scotland['Natural Gas']
        #     scaling_factor_CCS_gas_scotland = CCS_gas_unmodified_scotland / CCS_gas_bb_scotland

        #     CCS_gas_unmodified_rgb = generators_p_nom_rgb['CCS Gas']
        #     CCS_gas_bb_rgb = generators_p_nom_bb_rgb['Natural Gas']
        #     scaling_factor_CCS_gas_rgb = CCS_gas_unmodified_rgb / CCS_gas_bb_rgb

        #     for bus in buses_scotland:
        #         self.df_generators.loc[(self.df_generators.carrier == 'CCS Gas') & (self.df_generators.bus == bus), "p_nom"] /= scaling_factor_CCS_gas_scotland
        #     for bus in buses_rgb:
        #         self.df_generators.loc[(self.df_generators.carrier == 'CCS Gas') & (self.df_generators.bus == bus), "p_nom"] /= scaling_factor_CCS_gas_rgb

        if self.year >= 2030:
            # Scale hydrogen
            hydrogen_unmodified_scotland = generators_p_nom_scotland["Hydrogen"]
            hydrogen_bb_scotland = generators_p_nom_bb_scotland["Hydrogen"]
            if hydrogen_bb_scotland > 0.0:
                scaling_factor_hydrogen_scotland = (
                    hydrogen_unmodified_scotland / hydrogen_bb_scotland
                )
            else:
                scaling_factor_hydrogen_scotland = 1

            hydrogen_unmodified_rgb = generators_p_nom_rgb["Hydrogen"]
            hydrogen_bb_rgb = generators_p_nom_bb_rgb["Hydrogen"]
            # zero for falling short scenario so don't want infinite value
            if hydrogen_bb_rgb > 0:
                scaling_factor_hydrogen_rgb = hydrogen_unmodified_rgb / hydrogen_bb_rgb
            else:
                scaling_factor_hydrogen_rgb = 1

            for bus in buses_scotland:
                if scaling_factor_hydrogen_scotland == 0.0:
                    scaling_factor_hydrogen_scotland = 1
                self.df_generators.loc[
                    (self.df_generators.carrier == "Hydrogen")
                    & (self.df_generators.bus == bus),
                    "p_nom",
                ] /= scaling_factor_hydrogen_scotland
            for bus in buses_rgb:
                self.df_generators.loc[
                    (self.df_generators.carrier == "Hydrogen")
                    & (self.df_generators.bus == bus),
                    "p_nom",
                ] /= scaling_factor_hydrogen_rgb

        # Scale biomass
        biomass_unmodified_scotland = generators_p_nom_scotland["Biomass (dedicated)"]
        biomass_bb_scotland = generators_p_nom_bb_scotland["Biomass"]
        if biomass_bb_scotland > 0.0:
            scaling_factor_biomass_scotland = (
                biomass_unmodified_scotland / biomass_bb_scotland
            )
        else:
            scaling_factor_biomass_scotland = 1

        biomass_unmodified_rgb = generators_p_nom_rgb["Biomass (dedicated)"]
        biomass_bb_rgb = generators_p_nom_bb_rgb["Biomass"]
        # zero for falling short scenario so don't want infinite value
        if biomass_bb_rgb > 0:
            scaling_factor_biomass_rgb = biomass_unmodified_rgb / biomass_bb_rgb
        else:
            scaling_factor_biomass_rgb = 1

        for bus in buses_scotland:
            if scaling_factor_biomass_scotland == 0.0:
                scaling_factor_biomass_scotland = 1
            self.df_generators.loc[
                (self.df_generators.carrier == "Biomass (dedicated)")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_biomass_scotland
        for bus in buses_rgb:
            self.df_generators.loc[
                (self.df_generators.carrier == "Biomass (dedicated)")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] /= scaling_factor_biomass_rgb

        self.df_generators["p_nom"].fillna(0, inplace=True)

        # # generation from unmodified distribution
        # print(self.read_scotland_generators())
        # # generation according to building blocks data
        # print(pd.Series(self.generation_capacities_scotland()))

    def modify_storage(self):

        # modify the storage using the buildings block data
        # scales storage to match Scotland and rest of GB separately

        buses_scotland = self.buses_scotland
        # rgb is rest of GB
        buses_rgb = self.buses_rgb

        # storage from unmodified distribution
        storage_p_nom_scotland = self.read_scotland_storage()
        # storage according to building blocks data
        storage_p_nom_bb_scotland = self.storage_capacities_scotland()

        # storage from unmodified distribution
        storage_p_nom_rgb = self.read_rgb_storage()
        # storage according to building blocks data
        storage_p_nom_bb_rgb = self.storage_capacities_rgb()

        # Scale batteries
        batteries_unmodified_scotland = storage_p_nom_scotland["Battery"]
        batteries_bb_scotland = storage_p_nom_bb_scotland["Batteries"]
        scaling_factor_batteries_scotland = (
            batteries_unmodified_scotland / batteries_bb_scotland
        )

        batteries_unmodified_rgb = storage_p_nom_rgb["Battery"]
        batteries_bb_rgb = storage_p_nom_bb_rgb["Batteries"]
        scaling_factor_batteries_rgb = batteries_unmodified_rgb / batteries_bb_rgb

        for bus in buses_scotland:
            self.df_storage.loc[
                (self.df_storage.carrier == "Battery") & (self.df_storage.bus == bus),
                "p_nom",
            ] /= scaling_factor_batteries_scotland
            # self.df_storage.loc[(self.df_storage.carrier == 'Battery') & (self.df_storage.bus == bus), "max_hours"] /= scaling_factor_batteries_scotland
            self.df_storage.loc[
                (self.df_storage.carrier == "Battery") & (self.df_storage.bus == bus),
                "state_of_charge_initial",
            ] /= scaling_factor_batteries_scotland
        for bus in buses_rgb:
            self.df_storage.loc[
                (self.df_storage.carrier == "Battery") & (self.df_storage.bus == bus),
                "p_nom",
            ] /= scaling_factor_batteries_rgb
            # self.df_storage.loc[(self.df_storage.carrier == 'Battery') & (self.df_storage.bus == bus), "max_hours"] /= scaling_factor_batteries_rgb
            self.df_storage.loc[
                (self.df_storage.carrier == "Battery") & (self.df_storage.bus == bus),
                "state_of_charge_initial",
            ] /= scaling_factor_batteries_rgb

        # Scale pumped hydro
        pumped_hydro_unmodified_scotland = storage_p_nom_scotland[
            "Pumped Storage Hydroelectric"
        ]
        pumped_hydro_bb_scotland = storage_p_nom_bb_scotland["Pumped Hydro"]
        scaling_factor_pumped_hydro_scotland = (
            pumped_hydro_unmodified_scotland / pumped_hydro_bb_scotland
        )

        pumped_hydro_unmodified_rgb = storage_p_nom_rgb["Pumped Storage Hydroelectric"]
        pumped_hydro_bb_rgb = storage_p_nom_bb_rgb["Pumped Hydro"]
        scaling_factor_pumped_hydro_rgb = (
            pumped_hydro_unmodified_rgb / pumped_hydro_bb_rgb
        )

        for bus in buses_scotland:
            self.df_storage.loc[
                (self.df_storage.carrier == "Pumped Storage Hydroelectric")
                & (self.df_storage.bus == bus),
                "p_nom",
            ] /= scaling_factor_pumped_hydro_scotland
            self.df_storage.loc[
                (self.df_storage.carrier == "Pumped Storage Hydroelectric")
                & (self.df_storage.bus == bus),
                "state_of_charge_initial",
            ] /= scaling_factor_pumped_hydro_scotland
        for bus in buses_rgb:
            self.df_storage.loc[
                (self.df_storage.carrier == "Pumped Storage Hydroelectric")
                & (self.df_storage.bus == bus),
                "p_nom",
            ] /= scaling_factor_pumped_hydro_rgb
            self.df_storage.loc[
                (self.df_storage.carrier == "Pumped Storage Hydroelectric")
                & (self.df_storage.bus == bus),
                "state_of_charge_initial",
            ] /= scaling_factor_pumped_hydro_rgb

        # # storage from unmodified distribution
        # print(self.read_scotland_storage())
        # # storage according to building blocks data
        # print(pd.Series(self.storage_capacities_scotland()))

    def modify_interconnector(self):

        # modify the interconnector using the buildings block data
        # scales interconnector to match Scotland and rest of GB separately

        buses_scotland = self.buses_scotland
        # rgb is rest of GB
        buses_rgb = self.buses_rgb
        # interconnector from unmodified distribution
        interconnector_p_nom_scotland = self.read_scotland_interconnector()
        # interconnector according to building blocks data
        interconnector_p_nom_bb_scotland = self.interconnector_capacities_scotland()

        # interconnector from unmodified distribution
        interconnector_p_nom_rgb = self.read_rgb_interconnector()
        # interconnector according to building blocks data
        interconnector_p_nom_bb_rgb = self.interconnector_capacities_rgb()

        # Scale interconnector
        interconnector_unmodified_scotland = interconnector_p_nom_scotland
        interconnector_bb_scotland = interconnector_p_nom_bb_scotland["Interconnector"]
        scaling_factor_interconnector_scotland = (
            interconnector_unmodified_scotland / interconnector_bb_scotland
        )

        interconnector_unmodified_rgb = interconnector_p_nom_rgb
        interconnector_bb_rgb = interconnector_p_nom_bb_rgb["Interconnector"]
        scaling_factor_interconnector_rgb = (
            interconnector_unmodified_rgb / interconnector_bb_rgb
        )

        for bus in buses_scotland:
            self.df_interconnector.loc[
                self.df_interconnector.bus1 == bus, "p_nom"
            ] /= scaling_factor_interconnector_scotland
        for bus in buses_rgb:
            self.df_interconnector.loc[
                self.df_interconnector.bus1 == bus, "p_nom"
            ] /= scaling_factor_interconnector_rgb

    def PV_data(self):

        year = self.year
        scenario = self.scenario
        # get generators dataframe with p_noms to be scaled
        df = self.df_generators
        df_PV = df.loc[df["carrier"] == "Solar Photovoltaics"].reset_index(drop=True)
        PV_by_bus = df_PV.groupby("bus").sum()["p_nom"]
        PV_by_bus_norm = PV_by_bus / PV_by_bus.sum()

        # read in the future distribution for PV
        path = "../data/FES2021/Distributions/Solar Distribution " + scenario + ".csv"
        df_PV = pd.read_csv(path, index_col=0)
        df_PV = df_PV[df_PV.index.notnull()]
        df_PV = df_PV.loc[:, ~df_PV.columns.str.contains("^Unnamed")]
        df_PV = df_PV.astype("float")
        # compare indexes
        missing = list(set(df_PV.index.values) - set(PV_by_bus_norm.index.values))
        # drop the missing one in future PV
        df_PV.drop(missing, inplace=True)
        # normalise dataseries
        PV_norm = df_PV / df_PV.sum()

        return {"original": PV_by_bus_norm, "future": PV_norm[str(year)]}

    def PV_scale(self):
        PV = self.PV_data()
        PV_original = PV["original"]
        PV_future = PV["future"]

        scaling_factor = (PV_future / PV_original).fillna(0)

        # scale PV for each bus
        for bus in PV_future.index:
            self.df_generators.loc[
                (self.df_generators.carrier == "Solar Photovoltaics")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] *= scaling_factor[bus]

    def wind_onshore_data(self):

        year = self.year
        scenario = self.scenario

        # get generators dataframe with p_noms to be scaled
        df = self.df_generators
        df_wind = df.loc[df["carrier"] == "Wind Onshore"].reset_index(drop=True)
        wind_by_bus = df_wind.groupby("bus").sum()["p_nom"]
        wind_by_bus_norm = wind_by_bus / wind_by_bus.sum()

        # read in the future distribution for wind
        path = "../data/FES2021/Distributions/Wind Distribution " + scenario + ".csv"
        df_wind = pd.read_csv(path, index_col=0)
        df_wind = df_wind[df_wind.index.notnull()]
        df_wind = df_wind.loc[:, ~df_wind.columns.str.contains("^Unnamed")]
        df_wind = df_wind.astype("float")
        # compare indexes
        missing = list(set(df_wind.index.values) - set(wind_by_bus_norm.index.values))
        # drop the missing one in future wind
        df_wind.drop(missing, inplace=True)
        # normalise dataseries
        wind_norm = df_wind / df_wind.sum()

        return {"original": wind_by_bus_norm, "future": wind_norm[str(year)]}

    def wind_onshore_scale(self):
        wind = self.wind_onshore_data()
        wind_original = wind["original"]
        wind_future = wind["future"]

        scaling_factor = (wind_future / wind_original).fillna(0)
        # scale wind for each bus
        for bus in wind_future.index:
            self.df_generators.loc[
                (self.df_generators.carrier == "Wind Onshore")
                & (self.df_generators.bus == bus),
                "p_nom",
            ] *= scaling_factor[bus]

    def storage_data(self):

        year = self.year
        scenario = self.scenario

        # get generators dataframe with p_noms to be scaled
        df = self.df_storage
        df_storage = df[
            df["carrier"].str.contains("Pumped Storage Hydroelectric") == False
        ]
        storage_by_bus = df_storage.groupby("bus").sum()["p_nom"]
        storage_by_bus_norm = storage_by_bus / storage_by_bus.sum()

        # read in the future distribution for storage
        path = "../data/FES2021/Distributions/Storage Distribution " + scenario + ".csv"
        df_storage = pd.read_csv(path, index_col=0)
        df_storage = df_storage[df_storage.index.notnull()]
        df_storage = df_storage.loc[:, ~df_storage.columns.str.contains("^Unnamed")]
        df_storage = df_storage.astype("float")
        # compare indexes
        missing = list(
            set(df_storage.index.values) - set(storage_by_bus_norm.index.values)
        )
        # drop the missing one in future storage
        df_storage.drop(missing, inplace=True)
        # normalise dataseries
        storage_norm = df_storage / df_storage.sum()

        return {"original": storage_by_bus_norm, "future": storage_norm[str(year)]}

    def storage_scale(self):
        storage = self.storage_data()
        storage_original = storage["original"]
        storage_future = storage["future"]

        df = self.df_storage
        df_pumped_hydro = df[
            df["carrier"].str.contains("Pumped Storage Hydroelectric") == True
        ]
        df_storage = df[
            df["carrier"].str.contains("Pumped Storage Hydroelectric") == False
        ]

        scaling_factor = (storage_future / storage_original).fillna(0)
        # scale storage for each bus
        for bus in storage_future.index:
            df_storage.loc[df_storage.bus == bus, "p_nom"] *= scaling_factor[bus]
            df_storage.loc[
                df_storage.bus == bus, "state_of_charge_initial"
            ] *= scaling_factor[bus]

        # add in pumped hydro again
        self.df_storage = pd.concat([df_storage, df_pumped_hydro])

    def update(self):

        # run scaling functions
        self.PV_scale()
        self.wind_onshore_scale()
        self.storage_scale()

        # write generators file
        self.df_generators.to_csv("LOPF_data/generators.csv", index=True, header=True)

        # write storage file
        self.df_storage.to_csv("LOPF_data/storage_units.csv", index=True, header=True)

    def building_block_update(self):

        # run scaling functions
        self.modify_generators()
        self.modify_storage()
        # self.modify_interconnector()

        # write generators file
        self.df_generators.to_csv("LOPF_data/generators.csv", index=True, header=True)
        # write storage file
        self.df_storage.to_csv("LOPF_data/storage_units.csv", index=True, header=True)
        # write interconnector file
        # self.df_interconnector.to_csv('LOPF_data/links.csv', index=True, header=True)


if __name__ == "__main__":
    year = 2050
    scenario = "Leading the Way"
    myDistribution = Distribution(year, scenario)

    # print(myDistribution.generation_capacities())
    # print(myDistribution.read_scotland_generators())
    # myDistribution.modify_generators()
    # myDistribution.modify_storage()
    # myDistribution.read_regional_breakdown_load()
    myDistribution.scale_load()

    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Solar Photovoltaics'])
    # myDistribution.PV_scale()
    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Solar Photovoltaics'])

    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Wind Onshore'])
    # myDistribution.wind_onshore_scale()
    # print(myDistribution.df_generators.loc[myDistribution.df_generators.carrier == 'Wind Onshore'])

    # print(myDistribution.df_storage)
    # myDistribution.storage_scale()
    # print(myDistribution.df_storage)
