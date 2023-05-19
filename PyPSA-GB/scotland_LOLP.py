import os
import shutil
import numpy as np
import pypsa
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.style.use('ggplot')
import matplotlib.ticker as ticker 
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import data_reader_writer
import scotland_network


plt.style.use('plot_style.txt')

def get_rate(row, name_list, carrier, conversion_dict, power_stations, breakdown_rate, breakdowwn_rate_battery=None, storage_units= None):
    for i in range(len(carrier)):
        # if row[i] != [0]: only running stations
        name = name_list[i]
        fuel_type = convert_type(name, carrier, conversion_dict, power_stations)
        if fuel_type != 'Battery':
            rate = breakdown_rate[fuel_type]
        else:
            max_hours = storage_units[storage_units.index == name]['max_hours'].tolist()[0]
            max_hours = np.rint(max_hours * 2) / 2
            max_hours = min(max_hours, 4)
            rate = breakdowwn_rate_battery[str(max_hours) + 'h']        
        row[i] = rate
    return row


def convert_type(name, carrier, conversion_dict, power_stations):
    type0 = carrier[name]
    try:
        type1 = conversion_dict[type0]
    except:
        if type0 == 'Natural Gas':
            type1 = power_stations[power_stations['Station Name']==name]['Technology'].tolist()[0]
        else:
            type1 = 'Excluded'
#             if 'new_type_list' not in globals():
#                 global new_type_list
#                 new_type_list = list()
#             if type0 not in new_type_list:
#                 new_type_list.append(type0)
            print('Do not have break downrate for type ' + type0)
    return type1


def LOLP(network, year, year_baseline=None, failures_type=None, failures_rate=None):
    if year > 2020:
        year = year_baseline
    file = '../data/power stations/power_stations_locations_' + str(year) + '.csv'
    power_stations = pd.read_csv(file, encoding='unicode_escape')
    
    bd_conversion_csv = pd.read_csv('../data/LOLE/bd_conversion_type.csv',index_col=0)
    bd_conversion_dict = bd_conversion_csv.to_dict()['1']
    
    breakdown_rate = pd.read_csv('../data/LOLE/breakdown_rate.csv',index_col=0)
    breakdowwn_rate_battery = pd.read_csv('../data/LOLE/breakdowwn_rate_battery.csv',index_col=0)
    
    if (failures_type is not None) and (failures_type is not None):
        if type(failures_type) is not list:
            failures_type= [failures_type]
        if type(failures_rate) is not list:
            failures_rate = [failures_rate]
        if len(failures_type) != len(failures_rate):
            failures_rate = failures_rate * len(failures_type)
        for _ in range(len(failures_type)):
            generator_type = failures_type[_]
            if generator_type != 'Battery':
                breakdown_rate.loc[generator_type] = 1 - (1 - breakdown_rate.loc[generator_type]) * failures_rate[_]
            else:
                breakdowwn_rate_battery['Breakdown Rate'] = 1 - (1-breakdowwn_rate_battery['Breakdown Rate']) * failures_rate[_]
            
    breakdown_rate = breakdown_rate.to_dict()['Breakdown Rate']
    breakdowwn_rate_battery = breakdowwn_rate_battery.to_dict()['Breakdown Rate']
    
    generators_name_list = network.generators.index.tolist() ##
    
    #  generator units' breakdown rate time-series dataframe: generators_rate
    generators_rate_col = generators_name_list
    generators_rate_index = network.snapshots.copy()
    generators_rate = pd.DataFrame(columns=generators_rate_col, index=generators_rate_index) 
    generators_rate.apply(lambda r: get_rate(r, generators_name_list, network.generators.carrier, bd_conversion_dict, power_stations, breakdown_rate), axis = 1)
    
    # storage units' breakdown rate time-series dataframe: storage_rate
    storage_units_name_list = network.storage_units.index.tolist() 
    storage_units_rate_col = storage_units_name_list
    storage_units_rate_index = network.snapshots.copy()
    storage_units_rate = pd.DataFrame(columns=storage_units_rate_col, index=storage_units_rate_index) ##
    storage_units_rate.apply(lambda r: get_rate(r, storage_units_name_list, network.storage_units.carrier, bd_conversion_dict, power_stations, breakdown_rate, breakdowwn_rate_battery, network.storage_units), axis = 1)
    
    # all units' breakdown rate
    pd_rate = pd.concat([generators_rate, storage_units_rate], axis=1)
    # pd_rate2 = pd_rate.copy()
    
    # caculate time series of weather dependent generators' outputs from input data 

    pd_stations = pd_rate.copy()
    pd_stations_w = pd_stations[network.generators.index.tolist()][(pd_rate == 0)].dropna(axis=1,how='all').fillna(0)
    pd_stations[network.generators.p_nom.index] = network.generators.p_nom.values
    pd_stations[network.storage_units.p_nom.index] = network.storage_units.p_nom.values
    pd_stations[pd_stations_w.columns] = network.generators_t.p_max_pu[pd_stations_w.columns] * network.generators.p_nom[pd_stations_w.columns]

    pd_stations_all = pd_stations.copy()
    
    pd_stations_w = pd_stations[pd_stations_w.columns.tolist()]#[(pd_rate == 0)].dropna(axis=1,how='all').fillna(0)
    ## Shoreline Wave, Tidal Barrage and Tidal Stream

    pd_stations = pd_stations[(pd_rate < 1) & (pd_rate > 0)].dropna(axis=1,how='all').fillna(0)
    
    # Note: now, the time series of breakdown rate ONLY include non weather dependent units (i.e. convertional units)
    pd_rate = pd_rate[pd_stations.columns]
    
    # net_demand
    # calculate net_demand from input data (value is empty)
    # net_demand = total demand - renewable (Weather Dependent) output.
    net_demand = network.loads_t.p_set.sum(axis=1) - pd_stations_w.sum(axis=1)
    
    
    # non renewable generator units' installed capacity
    installed_capacity = pd.concat([network.generators.p_nom, network.storage_units.p_nom], axis=0)[pd_stations.columns].to_numpy()
    
    return installed_capacity, pd_rate.to_numpy()[0], net_demand, pd_stations_all, pd_stations_w


def Margin(network, pd_stations,year, system_reserve_requirment, year_baseline=None):
    if year > 2020:
        year = year_baseline
    file = '../data/power stations/power_stations_locations_' + str(year) + '.csv'
    power_stations = pd.read_csv(file, encoding='unicode_escape')
    
    de_conversion_csv = pd.read_csv('../data/LOLE/de_conversion_type.csv',index_col=0)
    de_conversion_dict = de_conversion_csv.to_dict()['1']
    
    de_rate = pd.read_csv('../data/LOLE/de_rate.csv',index_col=0)
    de_rate = de_rate.to_dict()['De-Rate']
    
    generators_name_list = network.generators.index.tolist() ##
    
    #  generator units' breakdown rate time-series dataframe: generators_rate
    generators_rate_col = generators_name_list
    generators_rate_index = network.snapshots.copy()
    generators_rate = pd.DataFrame(columns=generators_rate_col, index=generators_rate_index) 
    generators_rate.apply(lambda r: get_rate(r, generators_name_list, network.generators.carrier, de_conversion_dict, power_stations, de_rate), axis = 1)
    
    # storage units' breakdown rate time-series dataframe: storage_rate
    storage_units_name_list = network.storage_units.index.tolist() 
    storage_units_rate_col = storage_units_name_list
    storage_units_rate_index = network.snapshots.copy()
    storage_units_rate = pd.DataFrame(columns=storage_units_rate_col, index=storage_units_rate_index) ##
    storage_units_rate.apply(lambda r: get_rate(r, storage_units_name_list, network.storage_units.carrier, de_conversion_dict, power_stations, de_rate), axis = 1)
    
    # all units' breakdown rate
    pd_rate = pd.concat([generators_rate, storage_units_rate], axis=1) 
    
    # caculate time series of weather dependent generators' outputs from input data 

    pd_installed_capacity = pd.concat([network.generators.p_nom, network.storage_units.p_nom], axis=0)
    de_rated_capacity = pd_installed_capacity * pd_rate
    margin = de_rated_capacity.sum(axis =1)[0] - network.loads_t.p_set.sum(axis=1) - system_reserve_requirment # reserve

    # index = pd.concat([network.generators, network.storage_units])[~pd.concat([network.generators, network.storage_units]).carrier.isin[]]
    
    return de_rated_capacity, margin


def split_generators(installed_capacity, breakdwon_rate, num = None, value = 0, Round = False, Print = False):
    if num == None:
        num = installed_capacity.shape[0]
    sorted_capacity = np.sort(installed_capacity[installed_capacity >= value])
    boundary = sorted_capacity[max(sorted_capacity.shape[0] - num, 0)]
    large_capacity = np.copy(installed_capacity[installed_capacity >= boundary])
    large_breakdwon_rate = np.copy(breakdwon_rate[installed_capacity >= boundary])
    expect_small_capacity = sum(installed_capacity[installed_capacity < boundary] * (1 - breakdwon_rate[installed_capacity < boundary]))
    if Round:
        large_capacity = np.rint(large_capacity)
    if Print:
        print('Number of laege generators: ' + str(large_capacity.shape[0]))
        print('Boundary: ' + str(boundary))
    
    return large_capacity, large_breakdwon_rate, expect_small_capacity


def dict_add(A,B):
    for key,value in B.items():
        try:
            A[key] += value
        except:
            A[key] = value
    return A


def probability_function(capacity, breakdwon_rate):
    pdf = dict()
    pdf[0] = 1

    for i in range(capacity.shape[0]):
        prob_down = dict()
        prob_up = dict()
    
        for key,value in pdf.items():
            avail_value = np.float64(key)+capacity[i]
            try:
                prob_up[avail_value] += value * (1-breakdwon_rate[i])
            except:
                prob_up[avail_value] = value * (1-breakdwon_rate[i])
            try:
                prob_down[key] += value * breakdwon_rate[i]
            except:
                prob_down[key] = value * breakdwon_rate[i]
        pdf = dict_add(prob_up,prob_down)
    pdf = dict(sorted(pdf.items(),key=lambda d:d[0]))
    xx = sorted(pdf)
    yy = np.zeros(len(xx))
    cdf = np.zeros(len(xx))
    cdf_i = 0
    for i in range(len(pdf)):
        key = xx[i]
        yy[i] = pdf[key]
        cdf_i += yy[i]
        cdf[i] = cdf_i
    return xx, yy, pdf, cdf

def rate_table(nuclear=True):
    bd_conversion_type = [['Coal', 'Nuclear', 'Oil', 'Wind Offshore', 'Wind Onshore', 'Solar Photovoltaics', 'Large Hydro', 'Small Hydro', 'Anaerobic Digestion', 'EfW Incineration', 'Landfill Gas', 'Sewage Sludge Digestion', 'Shoreline Wave', 'Tidal Barrage and Tidal Stream', 'Biomass (co-firing)', 'Biomass (dedicated)','Pumped Storage Hydroelectric', 'Battery', 'Compressed Air', 'Liquid Air', 'Interconnector', 'Englandconnector',
                       'CCS Gas', 'CCS Biomass', 'Hydrogen', 'Unmet Load', 'Tidal lagoon', 'Tidal stream', 'Wave power', 'Waste'],
                      ['Coal', 'Nuclear', 'Simil-OCGT', 'Weather Dependent', 'Weather Dependent', 'Weather Dependent', 'Hydro', 'Hydro', 'Biomass', 'Biomass', 'Biomass', 'Biomass', 'Excluded', 'Excluded', 'Biomass', 'Biomass','Pumped storage', 'Battery', 'Simil-CCGT', 'Simil-CCGT', 'Interconnector', 'Englandconnector',
                       'Simil-CCGT', 'Biomass', 'Hydrogen', 'Excluded', 'Excluded', 'Excluded', 'Excluded', 'Biomass']]
    bd_conversion_type_csv = pd.DataFrame(bd_conversion_type).T

    breakdowwn_rate = {
        'Coal': 0.1,
        'CCGT': 0.06,
        'Simil-CCGT': 0.06,
        'Hydrogen': 0.06,
        'Nuclear': 0.1,
        'OCGT': 0.07,
        'Simil-OCGT': 0.06, 
        'Biomass': 0.06,
        'Hydro': 0.08,
        'Wind': 0.16,
        'Pumped storage': 0.03, # arbitrary, need to update
        'Interconnector': 0.2,
        'Englandconnector': 0.36,
        'Weather Dependent': 0,
        'Excluded': 1
    }

    if nuclear == False:
        breakdowwn_rate['Nuclear'] = 1

    br_csv = pd.DataFrame.from_dict(breakdowwn_rate,orient='index',columns=['Breakdown Rate'])
    br_csv = br_csv.reset_index().rename(columns = {'index':'Type'})

    # breakdowwn_rate_battery = {
    #     '0.5h': 1-0.1789,
    #     '1.0h': 1-0.3644,
    #     '1.5h': 1-0.5228,
    #     '2.0h': 1-0.6479,
    #     '2.5h': 1-0.7547,
    #     '3.0h': 1-0.8203,
    #     '3.5h': 1-0.8575,
    #     '4.0h': 1-0.9611 # 4+h
    # }

    breakdowwn_rate_battery = {
        '0.5h': 0.1789,
        '1.0h': 0.3644,
        '1.5h': 0.5228,
        '2.0h': 0.6479,
        '2.5h': 0.7547,
        '3.0h': 0.8203,
        '3.5h': 0.8575,
        '4.0h': 0.9611 # 4+h
    }
    
    brb_csv = pd.DataFrame.from_dict(breakdowwn_rate_battery,orient='index',columns=['Breakdown Rate'])
    brb_csv = brb_csv.reset_index().rename(columns = {'index':'Duration'})

    de_conversion_type = [['Coal', 'Nuclear', 'Oil', 'Wind Offshore', 'Wind Onshore', 'Solar Photovoltaics', 'Large Hydro', 'Small Hydro', 'Anaerobic Digestion', 'EfW Incineration', 'Landfill Gas', 'Sewage Sludge Digestion', 'Shoreline Wave', 'Tidal Barrage and Tidal Stream', 'Biomass (co-firing)', 'Biomass (dedicated)','Pumped Storage Hydroelectric', 'Battery', 'Compressed Air', 'Liquid Air', 'Interconnector', 'Englandconnector', 
                       'CCS Gas', 'CCS Biomass', 'Hydrogen', 'Unmet Load', 'Tidal lagoon', 'Tidal stream', 'Wave power','Waste'],
                      ['Coal', 'Nuclear', 'OCGT', 'Wind', 'Wind', 'Solar', 'Hydro', 'Hydro', 'Waste', 'Waste', 'Waste', 'Waste', 'Marine', 'Marine', 'Biomass', 'Biomass','Pumped storage', 'Battery storage', 'OCGT', 'OCGT', 'Interconnector', 'Interconnector',
                       'CCGT', 'Biomass', 'CCGT', 'Excluded', 'Marine', 'Marine', 'Marine','Waste']]
    de_conversion_type_csv = pd.DataFrame(de_conversion_type).T

    de_rate = {
        'Biomass': 0.88,
        'Waste': 0.745, # average for various
        'Coal': 0.76,
        'CCGT': 0.913, # CHP/cogeneration 
        'OCGT': 0.952, # gas & diesel reciprocating engines
        'Nuclear': 0.744,
        'Battery storage': 0.597,
        'Pumped storage': 0.952,
        'Hydro': 0.911,
        'Solar': 0.022,
        'Marine': 0.22,
        'Wind': 0.174, #offshore & onshore
        'DSR': 0.715,
        'Interconnector': 0.099,
        'Excluded': 0
    }

    if nuclear == False:
        de_rate['Nuclear'] = 0

    de_csv = pd.DataFrame.from_dict(de_rate,orient='index',columns=['De-Rate'])
    de_csv = de_csv.reset_index().rename(columns = {'index':'Type'})

    if not os.path.exists('../data/LOLE'):
        os.makedirs('../data/LOLE')
    bd_conversion_type_csv.to_csv('../data/LOLE/bd_conversion_type.csv',index=False)
    br_csv.to_csv('../data/LOLE/breakdown_rate.csv',index=False)
    brb_csv.to_csv('../data/LOLE/breakdowwn_rate_battery.csv',index=False)
    de_conversion_type_csv.to_csv('../data/LOLE/de_conversion_type.csv',index=False)
    de_csv.to_csv('../data/LOLE/de_rate.csv',index=False)


def main(year, scenario, demand_dataset='eload', year_baseline = 2020, system_reserve_requirment = 1200, step=100, nuclear=True):
    start = str(year) + '-01-01 00:00:00'
    end = str(year) + '-12-31 23:30:00'
    time_step = 1.

    if nuclear == False:
        rate_table(nuclear=False)

    try:
        data_reader_writer.data_writer(start, end, time_step, year, demand_dataset=demand_dataset, year_baseline=year_baseline,
            scenario=scenario, FES=2022, merge_generators=True, scale_to_peak=True)
    except:
        shutil.rmtree('LOPF_data')
        os.mkdir('LOPF_data')
        data_reader_writer.data_writer(start, end, time_step, year, demand_dataset=demand_dataset, year_baseline=year_baseline,
            scenario=scenario, FES=2022, merge_generators=True, scale_to_peak=True)
        
    scotland_network.scotland()
    scotland_network.interconnector()

    network = pypsa.Network()
    network.import_from_csv_folder('LOPF_data_Scotland')
    contingency_factor = 4
    network.lines.s_max_pu *= contingency_factor

    output_margin = pd.DataFrame(index=network.snapshots)
    output_lolp = pd.DataFrame(columns=['peak_lolp', 'lole', 'lole_week'])
    output_lolp_self = pd.DataFrame(columns=['peak_lolp', 'lole', 'lole_week'])
    lole_loop = pd.DataFrame(columns=[i*step for i in range(round(10000/step))])

    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline)
    de_rated_capacity, margin = Margin(network, pd_stations_all, year, system_reserve_requirment, year_baseline=year_baseline)

    output_margin['margin'] = margin
    output_margin['demand'] = network.loads_t.p_set.sum(axis=1)
    output_margin['net_demand'] = net_demand
    output_margin['weather_dependent'] = pd_stations_w.sum(axis=1)

    print('De_rated_capacity in ' + str(year) + ': ' + str(de_rated_capacity.sum(axis =1)[0]))
    print('System margin:')
    print(margin.describe())
    print('Demand:')
    print(network.loads_t.p_set.sum(axis=1).describe())
    print('Net demand:')
    print(pd.Series(net_demand).describe())

    if os.path.exists('../data/LOLE/peak_demand.csv'):
        pd_peak_demand = pd.read_csv('../data/LOLE/peak_demand.csv', index_col=0)
    else:
        pd_peak_demand = pd.DataFrame(columns=['demand'])
    pd_peak_demand.loc[year] = {'demand': network.loads_t.p_set.sum(axis=1).max()}
    pd_peak_demand.to_csv('../data/LOLE/peak_demand.csv')

    generators_p_nom_scotland = pd.concat([network.generators, network.storage_units]).p_nom.groupby(
        pd.concat([network.generators, network.storage_units]).carrier).sum().sort_values()
    if year > 2020:
        generators_p_nom_scotland.drop(['Unmet Load', 'CCS Biomass'], inplace=True)
    generators_p_nom_scotland.drop(generators_p_nom_scotland[generators_p_nom_scotland < 50].index, inplace=True)
    print(generators_p_nom_scotland)
    generators_p_nom_scotland.to_csv('../data/LOLE/generators_p_nom_'+str(year)+'_'+re.sub("[^A-Z]","",scenario)+'.csv')

    plt.figure(figsize=(10, 4))
    plt.bar(generators_p_nom_scotland.index, generators_p_nom_scotland.values / 1000)
    plt.xticks(generators_p_nom_scotland.index, rotation=90)
    plt.ylabel('GW')
    plt.grid(color='grey', linewidth=1, axis='both', alpha=0.5)
    # plt.title('Scotland installed generation capacity in year ' + str(year))
    plt.tight_layout()
    plt.savefig('../data/LOLE/Installed_capacity_'+str(year)+'_'+re.sub("[^A-Z]","",scenario)+'.png', dpi = 600)
    plt.show()

    pd_de_rated_capacity = pd.concat([network.generators, network.storage_units])
    pd_de_rated_capacity['de_rated_capacit'] = de_rated_capacity.loc[de_rated_capacity.index[0]].tolist()
    pd_de_rated_capacity = pd_de_rated_capacity.de_rated_capacit.groupby(
        pd_de_rated_capacity.carrier).sum().sort_values()
    if year > 2020:
        pd_de_rated_capacity.drop(['Unmet Load', 'CCS Biomass'], inplace=True)
    pd_de_rated_capacity.drop(pd_de_rated_capacity[pd_de_rated_capacity < 50].index, inplace=True)
    print(pd_de_rated_capacity)
    pd_de_rated_capacity.to_csv('../data/LOLE/de_rated_capacity_'+str(year)+'_'+re.sub("[^A-Z]","",scenario)+'.csv')

    plt.figure(figsize=(10, 4))
    plt.bar(pd_de_rated_capacity.index, pd_de_rated_capacity.values / 1000)
    plt.xticks(pd_de_rated_capacity.index, rotation=90)
    plt.ylabel('GW')
    plt.grid(color='grey', linewidth=1, axis='both', alpha=0.5)
    # plt.title('Scotland installed de-rate capacity in year ' + str(year))
    plt.tight_layout()
    plt.savefig('../data/LOLE/De-rate_capacity_'+str(year)+'_'+re.sub("[^A-Z]","",scenario)+'.png', dpi = 600)
    plt.show()


    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)

    pd_cdf = pd.DataFrame()
    pd_cdf['xx']=xx
    pd_cdf['yy']=cdf
    pd_cdf.to_csv('../data/LOLE/cdf_'+str(year)+'_'+re.sub("[^A-Z]","",scenario)+'.csv')

    # plt.plot(xx,yy)
    # plt.figure(figsize=(6,8))
    # plt.plot(xx,yy)
    # plt.xlim(max(xx)*.6,max(xx)*1.1)

    # int_x = [round(x/100)*100 for x in xx]
    # plt.figure(figsize=(6,8))
    # plt.plot(int_x,yy)
    # plt.xlim(max(xx)*.6,max(xx)*1.1)

    # plt.plot(xx,cdf)
    # plt.show()

    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lolp_base = lolp
   
    peak_demand = network.loads_t.p_set.sum(axis=1).max()
    index_peak_demand = np.where(network.loads_t.p_set.sum(axis=1).to_numpy() == peak_demand)[0][0]
    pd_peakload_period = pd.DataFrame(pd_stations_w.sum(axis=1),columns=['weather dependent capacity'])
    pd_peakload_period = pd_peakload_period[((pd_peakload_period.index.month<4)|(pd_peakload_period.index.month>10)) &
                (pd_peakload_period.index.weekday<6) &
                (pd_peakload_period.index.hour>6) & (pd_peakload_period.index.hour<20)]
    wdc = pd_peakload_period['weather dependent capacity'].to_numpy()

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp.loc['Base case'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Base case'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])
    


    # ### 2.1 Largest offshore wind farm failure 
    largest_windfarm = network.generators_t.p_max_pu[network.generators[network.generators.carrier.isin(['Wind Offshore'])].p_nom.idxmax()] \
        * network.generators[network.generators.carrier.isin(['Wind Onshore', 'Wind Offshore'])].p_nom.max()
    pd_peakload_period['largest windfarm supply'] = largest_windfarm[largest_windfarm.index.isin(pd_peakload_period.index)]
    lws = pd_peakload_period['largest windfarm supply'].to_numpy()
    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] + largest_windfarm[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]+lws[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp.loc['Largest offshore failure'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Largest offshore failure'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


    # ### 2.2 Long period of low RES power scenario
    net_demand = network.loads_t.p_set.sum(axis=1) - pd_stations_w.sum(axis=1) * 0.8
    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]*0.8-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]
    
    output_lolp.loc['Low RES power'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Low RES power'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


    # ### 1 Gas supply issues 
    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline, failures_type=['CCGT','OCGT'], failures_rate=0.)
    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)

    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp.loc['Gas supply issues'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Gas supply issues'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


    # ### 3.1 Storage failures
    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline, failures_type='Battery', failures_rate=0.)
    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)

    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())

    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp.loc['Storage failures'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*100].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Storage failures'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


    # ### 3.2 Interconnector failure
    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline, failures_type='Interconnector', failures_rate=0.)
    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)
    
    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp.loc['Interconnector failure'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Interconnector failure'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


     # ### 3.3 Scot-Eng links failure
    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline, failures_type='Englandconnector', failures_rate=0.)
    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)
    
    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp.loc['B6 failure'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['B6 failure'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


    # ## self-sufficient Scotland
    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline, failures_type=['Interconnector','Englandconnector'], failures_rate=0.)
    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)

    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])
    lolp_base = lolp

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp.loc['Self-sufficient'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}
    output_lolp_self.loc['Base case (Self-sufficient)'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole}

    peak_demand = network.loads_t.p_set.sum(axis=1).max()
    index_peak_demand = np.where(network.loads_t.p_set.sum(axis=1).to_numpy() == peak_demand)[0][0]
    
    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Self-sufficient'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])
    

    #################### THE FOLLOWING ARE ON THE TOP OF SELF-SUFFICENT SCOTLAND ########################

    # ### 2.1 Largest offshore wind farm failure 
    largest_windfarm = network.generators_t.p_max_pu[network.generators[network.generators.carrier.isin(['Wind Offshore'])].p_nom.idxmax()] \
        * network.generators[network.generators.carrier.isin(['Wind Onshore', 'Wind Offshore'])].p_nom.max()
    pd_peakload_period['largest windfarm supply'] = largest_windfarm[largest_windfarm.index.isin(pd_peakload_period.index)]
    lws = pd_peakload_period['largest windfarm supply'].to_numpy()
    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] + largest_windfarm[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]+lws[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp_self.loc['Largest offshore failure'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Self_Largest offshore failure'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


    # ### 2.2 Long period of low RES power scenario
    net_demand = network.loads_t.p_set.sum(axis=1) - pd_stations_w.sum(axis=1) * 0.8
    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]*0.8-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]
    
    output_lolp_self.loc['Low RES power'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Self_Low RES power'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])


    # ### 1 Gas supply issues 
    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline, failures_type=['CCGT', 'OCGT', 'Interconnector','Englandconnector'], failures_rate=0.)
    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)

    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())
    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp_self.loc['Gas supply issues'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Self_Gas supply issues'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])

                                       
    # ### 3.1 Storage failures
    installed_capacity, breakdwon_rate, net_demand, pd_stations_all, pd_stations_w = LOLP(network, year, year_baseline=year_baseline, failures_type=['Battery', 'Interconnector', 'Englandconnector'], failures_rate=0.)
    large_capacity, large_breakdwon_rate, expect_small_capacity = split_generators(installed_capacity, breakdwon_rate, value = 0, Round=True)
    xx, yy, pdf, cdf = probability_function(large_capacity, large_breakdwon_rate)

    lolp = list()
    for i in range(len(net_demand)):
        lolp.append(yy[xx<net_demand[i] - expect_small_capacity + system_reserve_requirment].sum())

    lole = sum(lolp)
    lole_week = sum(lolp_base[: index_peak_demand-85] + lolp[index_peak_demand-85: index_peak_demand+84] + lolp_base[index_peak_demand+84:])

    lolp_p = 0
    for i in range(wdc.shape[0]):
        lolp_p += yy[xx<peak_demand-wdc[i]-expect_small_capacity+system_reserve_requirment].sum()/wdc.shape[0]

    output_lolp_self.loc['Storage failures'] = {'peak_lolp':lolp_p, 'lole':lole, 'lole_week': lole_week}

    lole_list = [lole]
    i_ = 0
    while (lole > 0.3):
        i_ += 1
        lolp = list()
        for i in range(len(net_demand)):
            lolp.append(yy[xx<net_demand[i]-expect_small_capacity+system_reserve_requirment-i_*step].sum())
        lole = sum(lolp)
        lole_list.append(lole)
        print(f'for {i_*step}MW increased firm capacity, lole is {lole}')
    lole_loop.loc['Storage failures'] = dict([[i*step, lole_list[i]] for i in range(i_+1)])

    if nuclear == False:
        rate_table()

    output_lolp.to_csv('../data/LOLE/lolp_lole_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv')
    output_lolp_self.to_csv('../data/LOLE/self_lolp_lole_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv')
    output_margin.to_csv('../data/LOLE/margin_demand_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv')
    lole_loop.dropna(axis=1, how='all').T.to_csv('../data/LOLE/lole_loop_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv')


     
def main_plot(scenario, year_list):
    pd_plot = pd.DataFrame()
    pd_plot_self = pd.DataFrame()
    md_rate = list()
    pd_de_rate = pd.DataFrame()
    for year in year_list:
        pd_lolp = pd.read_csv('../data/LOLE/lolp_lole_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv', index_col=0)
        pd_lolp.index = pd.MultiIndex.from_arrays([[year]*8, pd_lolp.index.tolist()])
        pd_plot = pd.concat([pd_plot, pd_lolp])

        pd_lolp_self = pd.read_csv('../data/LOLE/self_lolp_lole_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv', index_col=0)
        pd_lolp_self.index = pd.MultiIndex.from_arrays([[year]*5, pd_lolp_self.index.tolist()])
        pd_plot_self = pd.concat([pd_plot_self, pd_lolp_self])

        pd_margin = pd.read_csv('../data/LOLE/margin_demand_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv', index_col=0)
        max_demand = pd_margin['demand'].max()
        min_margin = pd_margin['margin'].min()
        md_rate.append(min_margin / (max_demand + min_margin))

        pd_de_rate = pd.concat([pd_de_rate, pd.read_csv('../data/LOLE/de_rated_capacity_'+str(year)+'_'+re.sub("[^A-Z]","", scenario)+'.csv', index_col=0).rename(columns={'de_rated_capacit': year})], axis=1).fillna(0)

    pd_plot = pd_plot.swaplevel()
    pd_plot_self = pd_plot_self.swaplevel()
    for case in pd_plot.index.get_level_values(0).drop_duplicates().tolist():
        sub_pd = pd_plot.loc[case]
        for col in sub_pd.columns.tolist():
            plt.figure(figsize=(10,4))
            if col == 'lole':
                plt.plot([min(year_list)-5,max(year_list)+5], [0.2,0.2], 'k:')
                plt.plot([min(year_list)-5,max(year_list)+5], [0.3,0.3], 'k:')
                plt.text(year_list[0]-4, 0.25 , 'The LOLE reported in National Grid’s Winter Outlook in 2021\nand 2022 were 0.3 and 0.2 hrs/year.' , fontsize = 14 , color = 'k' , ha = 'left' )
            plt.plot(sub_pd.index, sub_pd[col], color='dodgerblue')
            plt.scatter(sub_pd.index, sub_pd[col], color='darkorange', marker='o')
            # plt.title(case+' - '+col)
            plt.ylabel('Hours')
            plt.grid(True)
            plt.xticks(year_list)
            plt.xlim(min(year_list)-5,max(year_list)+5)
            plt.savefig('../data/LOLE/'+case+'_'+col+'_'+re.sub("[^A-Z]","",scenario)+'.png', bbox_inches='tight', dpi=600)
            plt.show()

    for col in pd_plot.columns.tolist():
        plt.figure(figsize=(10,4))
        # plt.axes(yscale='log')
        for case in pd_plot.index.get_level_values(0).drop_duplicates().tolist()[:-1]:
            sub_pd = pd_plot.loc[case]
            if case == 'Gas supply issues':
                case = 'Gas power generation in Scotland unavailable'
            if case == 'Largest offshore failure':
                case = 'Offshore wind farm failures'  
            plt.plot(sub_pd.index, sub_pd[col], marker='o', label=case)
        # if col == 'lole_week':
        #     plt.title('all_scenario - '+'lole')
        # else:
        #     plt.title('all_scenario - '+col)
        # plt.title('all_scenario - '+col)
        plt.ylabel('Hours')
        plt.xticks(year_list)
        plt.xlim(min(year_list)-5,max(year_list)+5)
        plt.grid(True)
        plt.legend()
        plt.savefig('../data/LOLE/all_scenario_'+col+'_'+re.sub("[^A-Z]","",scenario)+'.png', bbox_inches='tight', dpi=600)
        plt.show()

        plt.figure(figsize=(10,4))
        # plt.axes(yscale='log')
        for case in pd_plot.index.get_level_values(0).drop_duplicates().tolist()[:-1]:
            sub_pd = pd_plot.loc[case]
            if case == 'Gas supply issues':
                case = 'Gas power generation in Scotland unavailable'
            if case == 'Largest offshore failure':
                case = 'Offshore wind farm failures'  
            plt.plot(sub_pd.index, sub_pd[col], marker='o', label=case)
        plt.plot([min(year_list)-5,max(year_list)+5], [3,3], 'r--')
        plt.text(year_list[0]-4, 2 , 'The current reliability standard for LOLE in GB\nis set to no more than three hours a year.' , fontsize = 14 , color = 'k' , ha = 'left' )
        # if col == 'lole_week':
        #     plt.title('all_scenario - '+'lole')
        # else:
        #     plt.title('all_scenario - '+col)
        plt.ylabel('Hours')
        plt.xticks(year_list)
        plt.xlim(min(year_list)-5,max(year_list)+5)
        plt.grid(True)
        plt.legend()
        plt.savefig('../data/LOLE/all_scenario_'+col+'_l_'+re.sub("[^A-Z]","",scenario)+'.png', bbox_inches='tight', dpi=600)
        plt.show()
        
    for col in pd_plot_self.columns.tolist():
        plt.figure(figsize=(10,4))
        # plt.axes(yscale='log')
        for case in pd_plot_self.index.get_level_values(0).drop_duplicates().tolist():
            sub_pd = pd_plot_self.loc[case]
            plt.plot(sub_pd.index, sub_pd[col], marker='o', label=case)
        plt.title('all_scenario - '+col + ' (self-sufficient)')
        plt.xticks(year_list)
        plt.xlim(min(year_list)-5,max(year_list)+5)
        plt.legend()
        plt.savefig('../data/LOLE/self_all_scenario'+'_'+col+'_'+re.sub("[^A-Z]","",scenario)+'.png', bbox_inches='tight', dpi=600)
        plt.show()

    plt.figure()
    plt.plot(year_list, md_rate, color='dodgerblue')
    plt.scatter(year_list, md_rate, color='darkorange', marker='o')
    plt.title('De-rated Supply margin')
    plt.xticks(year_list)
    plt.xlim(min(year_list)-2,max(year_list)+2)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
    plt.savefig('../data/LOLE/margin_demand.png', dpi=600)
    plt.show()

    plt.figure()
    cm = plt.get_cmap('gnuplot')
    bottom = np.zeros(pd_de_rate.shape[1])
    for i in range(pd_de_rate.shape[0]):
        carrier = pd_de_rate.index.tolist()[i]
        plt.bar(year_list, 
                pd_de_rate.loc[carrier].to_numpy(),
                bottom=bottom,
                # color=cm(.5+.5*(-1.)**i+(1.)**i*i/pd_de_rate.shape[0]),
                label=carrier)
        bottom += pd_de_rate.loc[carrier].to_numpy()
    plt.title('De-rated Supply margin in relation to De-rated generation capacity ')
    plt.xticks(year_list)
    plt.xlim(min(year_list)-2,max(year_list)+2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    plt.show()
    pd_de_rate.to_csv('../data/LOLE/combi_de-rated_cap_types.csv')

if __name__ == "__main__":
    rate_table()

    # scenario = 'Leading The Way'
    # scenario = 'Consumer Transformation'
    scenario = 'System Transformation'
    # scenario = 'Steady Progression'

    year_list = [2025, 2030, 2035, 2040, 2045]

    import time
    st = time.time()

    # main(2020, scenario, demand_dataset='historical')
    # main(2021, scenario)
    main(2025, scenario)
    # main(2025, scenario, nuclear=False)
    # main(2030, scenario)
    # main(2035, scenario)
    # main(2040, scenario, demand_dataset='historical')
    # main(2045, scenario)

    # main_plot(scenario, year_list)

    print(time.time() - st)
