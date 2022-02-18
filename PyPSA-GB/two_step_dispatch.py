"""Script for running a two-stage dispatch of the PyPSA-GB model
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import data_reader_writer

import cartopy.crs as ccrs

# dates need to include the last hour to be modelled + horizon
# i.e. to model one day, need to include next day too
start = '2019-06-02 00:00:00'
end = '2019-06-04 23:30:00'
# year of simulation
year = int(start[0:4])
# time step as fraction of hour
time_step = 0.5
# want unit commitment marginal prices? True or False
UC_marginal_prices = False

if year > 2020:
    # choose FES scenario
    scenario = 'Leading The Way'
    # scenario = 'Consumer Transformation'
    # scenario = 'System Transformation'
    # scenario = 'Steady Progression'
    year_baseline = 2012
    data_reader_writer.data_writer(start, end, time_step, year, year_baseline=year_baseline, scenario=scenario)
elif year <= 2020:
    data_reader_writer.data_writer(start, end, time_step, year)

network = pypsa.Network()
network.import_from_csv_folder(csv_folder_name='UC_data')

network2 = pypsa.Network()
network2.import_from_csv_folder(csv_folder_name='LOPF_data')

# set the horizon length and number of batches
horizon = 48
batches = 48

for i in range(int(batches)):

    print('Batch number: ', i)

    # set the initial state of charge based on previous round
    if i > 0:
        # both UC and LOPF use the updated state of charge from LOPF
        network.storage_units.state_of_charge_initial = (
            network2.storage_units_t.state_of_charge.loc[network.snapshots[i - 1]])
        network2.storage_units.state_of_charge_initial = (
            network2.storage_units_t.state_of_charge.loc[network.snapshots[i - 1]])

        # set the up time and down time before inputs for committable UC gens
        # as this represents the actually run outputs

        # want to do the below for all generators
        gen_ = network.generators
        # drop interconnectors
        gen_ = gen_[gen_.carrier != 'Interconnector']
        # drop PV
        gen_ = gen_[gen_.carrier != 'Solar Photovoltaics']
        # drop wind onshore
        gen_ = gen_[gen_.carrier != 'Wind Onshore']
        # drop wind offshore
        gen_ = gen_[gen_.carrier != 'Wind Offshore']
        # drop hydro
        gen_ = gen_[gen_.carrier != 'Small Hydro']
        gen_ = gen_[gen_.carrier != 'Large Hydro']
        # drop unmet load
        gen_ = gen_[gen_.carrier != 'Unmet Load']

        # generator index with above removed
        gen_index = gen_.index

        for gen in gen_index:

            # print(network2.generators_t.p.iloc[i - 1][gen], ' power in previ timestep')
            if network2.generators_t.p.iloc[i - 1][gen] > 0:
                # this means the generator was on in previous timestep...
                # down time before goes to zero
                network.generators.down_time_before.loc[gen] = 0
                # add one to the up_time_before...
                network.generators.up_time_before.loc[gen] += 1

            elif network2.generators_t.p.iloc[i - 1][gen] == 0:
                # this means the generator was off in previous timestep...
                # add one to the down_time_before
                network.generators.down_time_before.loc[gen] += 1
                # up time goes to zero
                network.generators.up_time_before.loc[gen] = 0

        # print(network.generators.up_time_before['Hartlepool'], 'nuclear uptime')
        # print(network.generators.down_time_before['Hartlepool'], 'nuclear downtime')
        # print(network.generators.up_time_before['Cottam'], 'coal uptime')
        # print(network.generators.down_time_before['Cottam'], 'coal downtime')

    # only need the second step of UC if interested in marginal prices from this step

    if UC_marginal_prices is False:
        network.lopf(network.snapshots[i:i + horizon], solver_name="gurobi")

    elif UC_marginal_prices is True:

        # Build a pyomo model corresponding to the pypsa network
        model = pypsa.opf.network_lopf_build_model(
            network,
            snapshots=network.snapshots[i:i + horizon])
        # Prepare a pyomo optimizer object, here with gurobi
        opt = pypsa.opf.network_lopf_prepare_solver(
            network, solver_name="gurobi")
        # solve the MILP
        opt.solve(model).write()
        # generator status print out
        # model.generator_status.pprint()
        # fixing the generator_status makes the problem linear (LP)
        model.generator_status.fix()
        # solve again with fixed generators
        network.results = opt.solve(model)
        network.results.write()
        # import back to pypsa datastructures
        pypsa.opf.extract_optimisation_results(
            network, network.snapshots[i:i + horizon])

        # can now see marginal prices
        # print(network.buses_t.marginal_price)
        # print(network.generators_t.status)
        # print(network.generators_t.p)

    # to approximate n-1 security and allow room for reactive power flows,
    # don't allow any line to be loaded above 70% of their thermal rating
    contingency_factor = 0.7
    network2.lines.s_max_pu = contingency_factor

    # committable generators
    gen_committ = network.generators[network.generators.committable != True].index
    s = network.generators_t.p
    # drop non-committable generator units
    s = s.drop(gen_committ, axis=1)
    s = s.iloc[i]
    # index of names of generators which are non-zero power
    gen_non_zero = s[s > 0].index
    print(gen_non_zero)

    # enforce a must run condition
    # on non-zero generators from UC using p_min_pu
    for gen in gen_non_zero:
        network2.generators.p_min_pu.loc[gen] = (
            network.generators.p_min_pu.loc[gen])
        print(network.generators.p_min_pu.loc[gen])
        print('break')
        print(network2.generators.p_min_pu.loc[gen])

    print(network2.snapshots[i:i + 2])
    # network constrained on single snapshot
    network2.lopf(network2.snapshots[i:i + 2], solver_name="gurobi")

# can now see marginal prices from unit commitment
print(network.buses_t.marginal_price)
print(network.generators_t.status)
print(network.generators_t.p)

# can now see marginal prices from LOPF
print(network2.buses_t.marginal_price.head(batches))

p_by_carrier = network2.generators_t.p.groupby(
    network2.generators.carrier, axis=1).sum()

storage_by_carrier = network2.storage_units_t.p.groupby(
    network2.storage_units.carrier, axis=1).sum()
print(network2.storage_units_t.p)

# to show on graph set the negative storage values to zero
storage_by_carrier[storage_by_carrier < 0] = 0
p_by_carrier = pd.concat([p_by_carrier, storage_by_carrier], axis=1)

if year <= 2020:

    # interconnector exports
    exports = network2.loads_t.p
    # multiply by negative one to convert it as a generator
    # i.e. export is a positive load, but negative generator
    exports['Interconnectors Export'] = exports.iloc[:, -6:].sum(axis=1) * -1
    interconnector_export = exports[['Interconnectors Export']].iloc[0:batches]

elif year > 2020:
    print(network2.links_t.p0)
    print(network2.links_t.p1)
    imp = network2.links_t.p0.copy()
    imp[imp < 0] = 0
    imp['Interconnectors Import'] = imp.sum(axis=1)
    interconnector_import = imp[['Interconnectors Import']]
    print(interconnector_import)
    p_by_carrier = pd.concat([p_by_carrier, interconnector_import], axis=1)

    exp = network2.links_t.p0.copy()
    exp[exp > 0] = 0
    exp['Interconnectors Export'] = exp.sum(axis=1)
    interconnector_export = exp[['Interconnectors Export']].iloc[0:batches]
    print(interconnector_export)

# group biomass stuff
p_by_carrier['Biomass'] = (
    p_by_carrier['Biomass (dedicated)'] + p_by_carrier['Biomass (co-firing)'] +
    p_by_carrier['Landfill Gas'] + p_by_carrier['Anaerobic Digestion'] +
    p_by_carrier['Sewage Sludge Digestion'])

# rename the hydro and interconnector import
p_by_carrier = p_by_carrier.rename(
    columns={'Large Hydro': 'Hydro'})
p_by_carrier = p_by_carrier.rename(
    columns={'Interconnector': 'Interconnectors Import'}).iloc[0:batches]

print(p_by_carrier)
# cols = ["Nuclear", "Coal", "Diesel/Gas oil", "Diesel/gas Diesel/Gas oil",
#         "Natural Gas", "Sour gas",
#         'Shoreline Wave', 'Tidal Barrage and Tidal Stream',
#         'Biomass (dedicated)', 'Biomass (co-firing)',
#         'Landfill Gas', 'Anaerobic Digestion', 'EfW Incineration', 'Sewage Sludge Digestion',
#         'Large Hydro', 'Pumped Storage Hydroelectricity', 'Small Hydro',
#         "Wind Offshore"
#         ]
#

if year > 2020:

    cols = ["Nuclear", 'Shoreline Wave', 'Biomass',
            'EfW Incineration', "Oil", "Natural Gas",
            'Hydrogen', 'CCS Gas', 'CCS Biomass',
            "Pumped Storage Hydroelectric", 'Hydro',
            'Battery', 'Compressed Air', 'Liquid Air',
            "Wind Offshore", 'Wind Onshore', 'Solar Photovoltaics',
            'Interconnectors Import', 'Unmet Load'
            ]

else:
    cols = ["Nuclear", 'Shoreline Wave', 'Biomass',
            'EfW Incineration',
            "Coal", "Oil", "Natural Gas",
            "Pumped Storage Hydroelectric", 'Hydro',
            "Wind Offshore", 'Wind Onshore', 'Solar Photovoltaics',
            'Interconnectors Import'
            ]

p_by_carrier = p_by_carrier[cols]

p_by_carrier.drop(
    (p_by_carrier.max()[p_by_carrier.max() < 50.0]).index,
    axis=1, inplace=True)


colors = {"Coal": "grey",
          "Diesel/Gas oil": "black",
          "Diesel/gas Diesel/Gas oil": "black",
          'Oil': 'black',
          'Unmet Load': 'black',
          'Anaerobic Digestion': 'green',
          'EfW Incineration': 'chocolate',
          'Sewage Sludge Digestion': 'green',
          'Landfill Gas': 'green',
          'Biomass (dedicated)': 'green',
          'Biomass (co-firing)': 'green',
          'Biomass': 'green',
          'CCS Biomass': 'darkgreen',
          'Interconnectors Import': 'pink',
          "Sour gas": "lightcoral",
          "Natural Gas": "lightcoral",
          'CCS Gas': "lightcoral",
          'Hydrogen': "lightcoral",
          "Nuclear": "orange",
          'Shoreline Wave': 'aqua',
          'Tidal Barrage and Tidal Stream': 'aqua',
          'Hydro': "turquoise",
          "Large Hydro": "turquoise",
          "Small Hydro": "turquoise",
          "Pumped Storage Hydroelectric": "darkturquoise",
          'Battery': 'lime',
          'Compressed Air': 'greenyellow',
          'Liquid Air': 'lawngreen',
          "Wind Offshore": "lightskyblue",
          'Wind Onshore': 'deepskyblue',
          'Solar Photovoltaics': 'yellow'}

fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(12, 6)
(p_by_carrier / 1e3).plot(
    kind="area", ax=ax, linewidth=0,
    color=[colors[col] for col in p_by_carrier.columns])

# stacked area plot of negative values, prepend column names with '_' such that they don't appear in the legend
(interconnector_export / 1e3).plot.area(ax=ax, stacked=True, linewidth=0.)
# rescale the y axis
ax.set_ylim([(interconnector_export / 1e3).sum(axis=1).min(), (p_by_carrier / 1e3).sum(axis=1).max()])

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

ax.set_ylabel("GW")

ax.set_xlabel("")

plt.show()

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 6)

p_storage = network2.storage_units_t.p.sum(axis=1).iloc[0:batches]
state_of_charge = network2.storage_units_t.state_of_charge.sum(axis=1).iloc[0:batches]
p_storage.plot(label="Pumped hydro dispatch", ax=ax, linewidth=3)
state_of_charge.plot(label="State of charge", ax=ax, linewidth=3)

ax.legend()
ax.grid()
ax.set_ylabel("MWh")
ax.set_xlabel("")
plt.show()

now = network2.snapshots[2]

print("With the linear load flow, there is the following per unit loading:")
loading = network2.lines_t.p0.loc[now] / network2.lines.s_nom
print(loading.describe())

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
fig.set_size_inches(6, 6)

network2.plot(ax=ax, line_colors=abs(loading), line_cmap=plt.cm.jet, title="Line loading")
plt.show()

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
fig.set_size_inches(6, 4)

network2.plot(ax=ax, line_widths=pd.Series(0.5, network2.lines.index))
plt.hexbin(network2.buses.x, network2.buses.y,
           gridsize=20,
           C=network2.buses_t.marginal_price.loc[now],
           cmap=plt.cm.jet)

# for some reason the colorbar only works with graphs plt.plot
# and must be attached plt.colorbar

cb = plt.colorbar()
cb.set_label('Locational Marginal Price (EUR/MWh)')
plt.show()

carrier = "Wind Onshore"

capacity = network2.generators.groupby("carrier").sum().at[carrier, "p_nom"]
p_available = network2.generators_t.p_max_pu.multiply(network2.generators["p_nom"])
p_available_by_carrier = p_available.groupby(network2.generators.carrier, axis=1).sum()
p_curtailed_by_carrier = p_available_by_carrier - p_by_carrier
p_df = pd.DataFrame({carrier + " available": p_available_by_carrier[carrier],
                     carrier + " dispatched": p_by_carrier[carrier],
                     carrier + " curtailed": p_curtailed_by_carrier[carrier]}).iloc[0:batches]

p_df[carrier + " capacity"] = capacity
p_df["Wind Onshore curtailed"][p_df["Wind Onshore curtailed"] < 0.] = 0.
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 6)
p_df[[carrier + " dispatched", carrier + " curtailed"]].plot(kind="area", ax=ax, linewidth=0)

ax.set_xlabel("")
ax.set_ylabel("Power [MW]")
ax.legend()
plt.show()
