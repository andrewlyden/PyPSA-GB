import pandas as pd

from rapidfuzz import process, fuzz
import pypsa

generator_dict = {
    'CHP (>=1MW)':              'Gen_BB001',
    'CHP (<1MW)':               'Gen_BB002',
    'Micro CHP':                'Gen_BB003',
    'Renewable Engines':        'Gen_BB004',
    'Diesel Engines':           'Gen_BB005',
    'Gas Engines':              'Gen_BB006',
    'Fuel Cells':               'Gen_BB007',
    'OCGT':                     'Gen_BB008',
    'CCGT':                     'Gen_BB009',
    'Biomass':                  'Gen_BB010',
    'Waste':                    'Gen_BB011',
    'PV (Large)':               'Gen_BB012',
    'PV (Small)':               'Gen_BB013',
    'Wind (Offshore)':          'Gen_BB014',
    'Wind (Onshore >=1MW)':     'Gen_BB015',
    'Wind (Onshore <1MW)':      'Gen_BB016',
    'Marine':                   'Gen_BB017',
    'Hydro':                    'Gen_BB018',
    'Geothermal':               'Gen_BB019',
    'Nuclear':                  'Gen_BB020',
    'Coal':                     'Gen_BB021',
    'Interconnector':           'Gen_BB022',
    'Hydrogen':                 'Gen_BB023',
    'Offshore-wind (off-grid)': 'Gen_BB024',
}

TEC_dict = {
    'CHP (>=1MW)':              'CHP (Combined Heat and Power)',
    'OCGT':                     'OCGT (Open Cycle Gas Turbine)',
    'CCGT':                     'CCGT (Combined Cycle Gas Turbine)',
    'Biomass':                  'Biomass',
    'Waste':                    'Waste',
    'PV (Large)':               'PV Array (Photo Voltaic/solar)',
    'Wind (Offshore)':          'Wind Offshore',
    'Wind (Onshore >=1MW)':     'Wind Onshore',
    'Marine':                   'Tidal',
    'Hydro':                    'Hydro',
    'Nuclear':                  'Nuclear',
}

direct_non_TEC_dict = {
    'CHP (<1MW)':               'Wind (Onshore >=1MW)',
    'Micro CHP':                'Wind (Onshore >=1MW)',
    'Renewable Engines':        'Wind (Onshore >=1MW)',
    'Diesel Engines':           'Wind (Onshore >=1MW)',
    'Gas Engines':              'Wind (Onshore >=1MW)',
    'Fuel Cells':               'Wind (Onshore >=1MW)',
    'PV (Small)':               'PV (Large)',
    'Wind (Onshore <1MW)':      'Wind (Onshore >=1MW)',
    'Geothermal':               'Wind (Onshore >=1MW)',
    'Coal':                     'CCGT',
    'Hydrogen':                 'CCGT',
    'Offshore-wind (off-grid)': 'Wind (Onshore >=1MW)',
}


def FES_generator_data_to_node(FES_scenario, modelled_year):
    # Read in FES data
    FES_data = pd.read_csv(snakemake.input[0], low_memory=False)
    FES_data = FES_data[FES_data.iloc[:, 1] == FES_scenario]

    # Filter the DataFrame once using the values from the dictionary
    bb_ids = list(generator_dict.values())
    filtered_data = FES_data[FES_data['Building Block ID Number'].isin(bb_ids)].copy()
    # Map the generator types to their corresponding building block IDs
    filtered_data.loc[:, 'generator_type'] = filtered_data['Building Block ID Number'].map(
        {v: k for k, v in generator_dict.items()}
    )
    # New column which combines the generator type and the node ID
    filtered_data.loc[:, 'Name'] = filtered_data['Node ID'] + ' ' + filtered_data['generator_type']
    # Make 'Name' the index
    filtered_data = filtered_data.set_index('Name')
    # Assuming 'modelled_year' is the name of the column you want to sum
    modelled_year_column = str(modelled_year)
    # Group by the index and sum only the 'modelled_year' column
    summed_modelled_year = filtered_data.groupby(filtered_data.index)[modelled_year_column].sum()
    # Drop the 'modelled_year' column from the original DataFrame
    filtered_data = filtered_data.drop(columns=[modelled_year_column])
    # Drop duplicate rows based on the index, keeping the first occurrence
    filtered_data = filtered_data[~filtered_data.index.duplicated(keep='first')]
    # Combine the summed 'modelled_year' column back into the DataFrame using .loc
    filtered_data.loc[summed_modelled_year.index, modelled_year_column] = summed_modelled_year
    # remove nan values in the index
    filtered_data = filtered_data.dropna(subset=[modelled_year_column])
    # remove rows with 0 values in modelled year
    filtered_data = filtered_data[filtered_data[modelled_year_column] != 0]

    return filtered_data


def TEC_register_data():
    TEC_raw_data = pd.read_csv(snakemake.input[1], low_memory=False, index_col=0)
    # only keep columns, 'Project Name', 'Connection Site', 'Cumulative Total Capacity (MW)', 'Project Status', 'HOST TO', and 'Plant Type'
    TEC_raw_data = TEC_raw_data[['Connection Site', 'Cumulative Total Capacity (MW)', 'MW Effective From', 'Project Status', 'Agreement Type', 'HOST TO', 'Plant Type']]
    # filter Project Status column for 'Built'
    # TEC_raw_data = TEC_raw_data[TEC_raw_data['Project Status'] == 'Built']
    # filter Agreement Type column for 'Direct Connection'
    # TEC_raw_data = TEC_raw_data[TEC_raw_data['Agreement Type'] == 'Direct Connection']
    # create a copy
    TEC_data = TEC_raw_data.copy()

    ETYS_node_data = pd.ExcelFile(snakemake.input[3])
    sheets = {
        'B-2-1a': ETYS_node_data.parse('B-1-1a', skiprows=1),
        'B-2-1b': ETYS_node_data.parse('B-1-1b', skiprows=1),
        'B-2-1c': ETYS_node_data.parse('B-1-1c', skiprows=1),
        'B-2-1d': ETYS_node_data.parse('B-1-1d', skiprows=1),
    }
    dfa, dfb, dfc, dfd = sheets['B-2-1a'], sheets['B-2-1b'], sheets['B-2-1c'], sheets['B-2-1d']
    # combine these sheets into one DataFrame
    df_map = pd.concat([dfa, dfb, dfc, dfd], axis=0)
    # ensure that the 'Site Name' column in df_map has unique values before setting it as the index.
    df_map = df_map.drop_duplicates(subset='Site Name')

    # Convert 'Connection Site' and 'Site Name' to lowercase for case-insensitive matching
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.lower()
    # Remove word 'substation' from 'Connection Site'
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' substation', '')
    # Remove parts of string which are 132kv, 275kv, 33kv
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 132kv', '')
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 275kv', '')
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 33kv', '')
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 33/132kv', '')
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 132/33kv', '')
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 32/132kv', '')
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 400kv', '')
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 400/132kv', '')
    # remove GSP
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' gsp', '')
    # remove wind farm
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' wind farm', '')
    # remove 400/275/132/33kv
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(' 400/275/132/33kv', '')
    # rmove anything between brackets
    TEC_data.loc[:, 'Connection Site'] = TEC_data['Connection Site'].str.replace(r"\(.*\)", "")

   # Find close matches for 'Connection Site' in 'Site Name'
    site_names = df_map['Site Name'].tolist()
    # remove any duplicates
    site_names = list(set(site_names))
    # make lower case
    site_names = [x.lower() for x in site_names]
    def find_best_match(site):
        match, score, _ = process.extractOne(site, site_names, scorer=fuzz.token_set_ratio)
        return match if score > 80 else None  # Adjust the threshold as needed

    TEC_data['Matched Site Name'] = TEC_data['Connection Site'].apply(find_best_match)
    # make it all upper case
    TEC_data['Matched Site Name'] = TEC_data['Matched Site Name'].str.upper()
    # replace the connection site with the TEC_raw_data connection site
    TEC_data['Connection Site'] = TEC_raw_data['Connection Site']

    # for the column matced site name, add the site code from the map
    TEC_data['Site Code'] = TEC_data['Matched Site Name'].map(df_map.set_index('Site Name')['Site Code'])
    # remove rows with nan values in the 'Site Code' column
    # TEC_data = TEC_data.dropna(subset=['Site Code'])

    # read TEC manually added data
    TEC_manual_data = pd.read_csv(snakemake.input[2], low_memory=False, index_col=0)
    # concat the two dataframes
    TEC_data = pd.concat([TEC_data, TEC_manual_data], axis=0)
    # remove rows with nan values in the 'Site Code' column
    TEC_data = TEC_data.dropna(subset=['Site Code'])

    # Match the site code to a bus in the network
    network = pypsa.Network(snakemake.input[4])
    # Create a dictionary of site code and bus name
    network.buses['name'] = network.buses.index
    site_code_bus_dict = network.buses['name'].to_dict()
    # Map the site code to the bus name
    def find_best_bus(site_code):
        match, score, _ = process.extractOne(site_code, site_code_bus_dict.keys(), scorer=fuzz.partial_ratio)
        return site_code_bus_dict[match] if score > 80 else None  # Adjust the threshold as needed

    # Map the site code to the bus name with partial matching
    TEC_data['Bus'] = TEC_data['Site Code'].apply(find_best_bus)
    # remove rows with nan values in the 'Bus' column
    TEC_data = TEC_data.dropna(subset=['Bus'])
    # remove part of string before ; in plant type
    TEC_data.loc[:, 'Plant Type'] = TEC_data['Plant Type'].str.split(';').str[-1]
    # save to csv for plotting
    TEC_data.to_csv(snakemake.output[0])

    return TEC_data

def TEC_distributions(TEC_data, TEC_dict):

    # Reverse the TEC_dict dictionary
    reversed_TEC_dict = {v: k for k, v in TEC_dict.items()}
    # Map the values in the 'Plant Type' column using the reversed dictionary
    TEC_data['Plant Type Standard'] = TEC_data['Plant Type'].map(reversed_TEC_dict)
    # Calculate the total capacity for each Plant Type Standard
    total_capacity_per_type = TEC_data.groupby('Plant Type Standard')['Cumulative Total Capacity (MW)'].transform('sum')
    # Calculate the normalized capacity for each entry within its group
    TEC_data['Normalized Capacity'] = TEC_data['Cumulative Total Capacity (MW)'] / total_capacity_per_type
    # Combine rows with the same index and sum their 'Normalized Capacity'
    # Keep the first occurrence of other columns
    agg_dict = {col: 'first' for col in TEC_data.columns if col not in ['Normalized Capacity', 'Cumulative Total Capacity (MW)']}
    agg_dict.update({'Normalized Capacity': 'sum', 'Cumulative Total Capacity (MW)': 'sum'})
    TEC_data = TEC_data.groupby(TEC_data.index).agg(agg_dict)
    # add 'Direct' to the start of each index entry
    TEC_data.index = 'Direct ' + TEC_data.index

    return TEC_data

def non_TEC_distributions(TEC_dist, direct_non_TEC_dict):

    # List to hold processed DataFrames
    processed_dfs = []
    for key in direct_non_TEC_dict.keys():
        # Get df where Plant Type Standard is direct_non_TEC_dict[key]
        df = TEC_dist[TEC_dist['Plant Type Standard'] == direct_non_TEC_dict[key]]
        # Change Plant Type Standard to the key
        df.loc[:, 'Plant Type Standard'] = key
        # also drop plant type
        df = df.drop(columns=['Plant Type'])
        # Change project name to add the key at the end
        df.index = 'Direct ' + df.index + f' {key}'
        # Append the processed DataFrame to the list
        processed_dfs.append(df)
    # Concatenate all processed DataFrames into one
    combined_df = pd.concat(processed_dfs)

    return combined_df

def add_generator_to_network(network, generator_data, TEC_distributions, non_TEC_dist, modelled_year):

    # create separate df for rows with Direct in generator_data in the index
    direct_df = generator_data[generator_data.index.str.contains('Direct')]
    # remove these from generator_data
    generator_data = generator_data[~generator_data.index.str.contains('Direct')]
    # remove where [str(modelled_year)] is 0, i.e., installed capacity is 0
    generator_data = generator_data[generator_data[str(modelled_year)] != 0]

    # Add the generator data to the network
    network = pypsa.Network(network)
    # PyPSA 1.0.2: madd() was replaced with add() in a loop
    for gen_name in generator_data.index:
        network.add("Generator",
                    gen_name,
                    bus=generator_data.at[gen_name, 'Node ID'],
                    p_nom=generator_data.at[gen_name, str(modelled_year)],
                    type=generator_data.at[gen_name, 'generator_type'])

    direct_gens = pd.concat([TEC_distributions, non_TEC_dist])
    # rename TEC_distributions column Plant Type to generator_type
    direct_gens = direct_gens.rename(columns={'Plant Type Standard': 'generator_type'})
    # for each row of TEC_distrubtions, multiply the normalized capacity by the total capacity for each generator_type in direct_df
    for index, row in direct_gens.iterrows():
        # get the generator type
        generator_type = row['generator_type']
        # get the total capacity for the generator type
        total_capacity = direct_df[direct_df['generator_type'] == generator_type][str(modelled_year)].sum()
        # multiply the normalized capacity by the total capacity
        direct_gens.at[index, 'p_nom'] = row['Normalized Capacity'] * total_capacity
    # drop rows with nan in the 'p_nom' column
    direct_gens = direct_gens.dropna(subset=['p_nom'])
    
    # NEED TO GET DISTRIBUTIONTS FOR TYPES NOT IN TEC DATA, E.G., HYDROGEN
    # drop nan for generator_type
    direct_gens = direct_gens.dropna(subset=['generator_type'])
    # drop nan or zero for p_nom
    direct_gens = direct_gens[direct_gens['p_nom'] != 0]

    # add the TEC_distributions data to the network
    # PyPSA 1.0.2: madd() was replaced with add() in a loop
    for gen_name in direct_gens.index:
        network.add("Generator",
                    gen_name,
                    bus=direct_gens.at[gen_name, 'Bus'],
                    p_nom=direct_gens.at[gen_name, 'p_nom'],
                    type=direct_gens.at[gen_name, 'generator_type'])

    network.export_to_netcdf(snakemake.output[1])
    network.export_to_csv_folder('test_generator')

if __name__ == "__main__":

    generation_data = FES_generator_data_to_node(snakemake.config['FES_scenario'][0], snakemake.config['modelled_year'][0])
    TEC_data = TEC_register_data()
    TEC_dist = TEC_distributions(TEC_data, TEC_dict)
    non_TEC_dist = non_TEC_distributions(TEC_dist, direct_non_TEC_dict)
    add_generator_to_network(snakemake.input[4], generation_data, TEC_dist, non_TEC_dist, snakemake.config['modelled_year'][0])

