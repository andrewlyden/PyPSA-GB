import os
import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
import pypsa


mapping = {
    'PV (Large)':                            'Solar Photovoltaics',  # Large solar PV directly relates to solar power
    'PV (Small)':                            'Solar Photovoltaics',  # Small solar still relates to PV
    'Wind (Offshore)':                       'Wind Offshore',  # Direct match to offshore wind
    'Wind (Onshore >=1MW)':                  'Wind Onshore',  # Larger scale onshore wind
    'Wind (Onshore <1MW)':                   'Wind Onshore',  # Smaller scale energy production
    'Marine':                                ['Tidal Stream', 'Shoreline Wave'],  # Closest match for marine energy generation
    'Hydro':                                 ['Large Hydro', 'Small Hydro'],  # Direct match for large hydro
}

def REPD_data():

    # Read in REPD data
    repd = pd.read_csv(snakemake.input[0])
    # Keep specific columns
    repd = repd[['Site Name', 'Technology Type', 'Installed Capacity (MWelec)', 'Development Status', 'X-coordinate', 'Y-coordinate']]
    # Set 'Site Name' as the index
    repd = repd.set_index('Site Name')
    # Filter 'Development Status' to only include 'Operational'
    repd = repd[repd['Development Status'] == 'Operational']
    # Define the projection transformation
    from_crs = "epsg:27700"  # Example: British National Grid EPSG code
    to_crs = "epsg:4326"     # Example: WGS84 (Latitude/Longitude) EPSG code
    # Create a transformer object
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    # Apply the transformation to convert X, Y to Lon, Lat
    repd['Longitude'], repd['Latitude'] = transformer.transform(repd['X-coordinate'].values, repd['Y-coordinate'].values)
 
    return repd

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two sets of points.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


import faiss

import pandas as pd
import numpy as np
import pypsa
import os
from scipy.spatial.distance import cdist

def create_spatial_grid(data, grid_size):
    """
    Create a spatial grid and assign each point to a grid cell.
    """
    # Calculate the minimum and maximum coordinates
    min_x, min_y = data[['Longitude', 'Latitude']].min()
    max_x, max_y = data[['Longitude', 'Latitude']].max()

    # Create grid cell identifiers
    x_bins = np.linspace(min_x, max_x, grid_size)
    y_bins = np.linspace(min_y, max_y, grid_size)

    # Assign each point to a grid cell
    data['x_bin'] = np.digitize(data['Longitude'], x_bins)
    data['y_bin'] = np.digitize(data['Latitude'], y_bins)

    return data, x_bins, y_bins

def get_nearby_cells(x_bin, y_bin, grid_size):
    """
    Get the neighboring cells for a given cell.
    """
    nearby_cells = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            new_x = x_bin + dx
            new_y = y_bin + dy
            if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                nearby_cells.append((new_x, new_y))
    return nearby_cells

def atlite_to_normalised_timeseries_bus(tech, atlite_input, REPD, network_path, grid_size=10, max_plants=5):
    # Load the PyPSA network
    network = pypsa.Network(network_path)

    # Load and prepare the atlite input data
    if tech == 'Solar Photovoltaics':
        # Dynamically load and concatenate part files
        df_parts = []
        base_path = atlite_input.replace('.csv', '')
        part_number = 1

        while True:
            part_file = f"{base_path}_part_{part_number}.csv"
            if os.path.exists(part_file):
                df_parts.append(pd.read_csv(part_file))
                part_number += 1
            else:
                break

        if not df_parts:
            raise FileNotFoundError("No part files found for Solar Photovoltaics.")

        # Concatenate all part files
        df_tech = pd.concat(df_parts, axis=1)
    else:
        df_tech = pd.read_csv(atlite_input)

    # Clean up the dataframe
    df_tech = df_tech.dropna(axis=1)
    # drop duplicate columns
    df_tech = df_tech.loc[:, ~df_tech.columns.duplicated()]
    df_tech = df_tech.apply(pd.to_numeric, errors='coerce')

    # Filter the REPD for the specified technology
    REPD_filtered = REPD[REPD['Technology Type'].str.contains(tech, case=False, na=False)].copy()

    # Ensure 'Installed Capacity (MWelec)' is numeric and drop plants with missing capacity
    REPD_filtered['Installed Capacity (MWelec)'] = pd.to_numeric(REPD_filtered['Installed Capacity (MWelec)'], errors='coerce')
    REPD_filtered = REPD_filtered.dropna(subset=['Installed Capacity (MWelec)', 'Latitude', 'Longitude'])

    # Create a spatial grid for plants
    REPD_filtered, x_bins, y_bins = create_spatial_grid(REPD_filtered, grid_size)

    # Prepare bus coordinates and add them to the grid
    buses = network.buses[['x', 'y']].copy()
    buses['bus_id'] = buses.index
    buses['x_bin'] = np.digitize(buses['x'], x_bins)
    buses['y_bin'] = np.digitize(buses['y'], y_bins)

    # Initialize dictionary to store timeseries for each bus
    bus_timeseries = {bus_id: np.zeros(df_tech.shape[0]) for bus_id in network.buses.index}

    # Loop over each bus and search for nearby plants within the grid
    for _, bus in buses.iterrows():
        x_bin, y_bin = bus['x_bin'], bus['y_bin']

        # Get nearby cells
        nearby_cells = get_nearby_cells(x_bin, y_bin, grid_size)

        # Filter plants within the nearby cells
        nearby_plants = REPD_filtered[
            REPD_filtered[['x_bin', 'y_bin']].apply(tuple, axis=1).isin(nearby_cells)
        ]

        # If there are no nearby plants, skip this bus
        if nearby_plants.empty:
            continue

        # Calculate Euclidean distances from the bus to all nearby plants
        bus_coords = np.array([bus['x'], bus['y']]).reshape(1, -1)
        plant_coords = nearby_plants[['Longitude', 'Latitude']].values
        distances = cdist(bus_coords, plant_coords, metric='euclidean').flatten()

        # Avoid division by zero by setting a minimum distance
        distances = np.maximum(distances, 1e-6)

        # Compute weights as the inverse of distance
        weights = 1 / distances
        weights /= weights.sum()  # Normalize weights

        # Select the top `max_plants` closest plants
        sorted_indices = np.argsort(distances)[:max_plants]

        # Aggregate the timeseries data for the bus using the weights
        for i in sorted_indices:
            plant = nearby_plants.iloc[i]
            plant_id = plant.name
            if plant_id in df_tech.columns:
                timeseries = df_tech[plant_id].values
                bus_timeseries[bus['bus_id']] += weights[i] * timeseries * plant['Installed Capacity (MWelec)']

    # Create DataFrame from the aggregated bus timeseries
    timeseries_df = pd.DataFrame(bus_timeseries)

    # Add suffix to column names based on technology type
    if tech == 'Wind Onshore':
        suffix = 'Wind (Onshore >=1MW)'
    elif tech == 'Solar Photovoltaics':
        suffix = 'PV (Large)'
    elif tech == 'Wind Offshore':
        suffix = 'Wind (Offshore)'
    else:
        suffix = tech

    timeseries_df.columns = [f'{col} {suffix}' for col in timeseries_df.columns]

    # Save the normalized timeseries to a CSV file
    timeseries_df.to_csv(f'{tech}_normalized_timeseries.csv', index=False)

    return timeseries_df


def add_to_network(df1, df2, df3):
    # Concatenate all the dataframes along columns
    df = pd.concat([df1, df2, df3], axis=1)
    # Save to a CSV file
    df.to_csv('generator_timeseries.csv')

    # Load the PyPSA network
    network = pypsa.Network(snakemake.input[1])

    # Set snapshots to the length of the dataframe
    network.set_snapshots(df.index)

    # Add generator timeseries data to the network
    generators_to_add = []
    for generator in network.generators.index:
        # Ensure the generator is in df columns and add its timeseries to network
        if generator in df.columns:
            generators_to_add.append(generator)

    # Concatenate all generator timeseries at once
    network.generators_t.p_max_pu = pd.concat([df[generator] for generator in generators_to_add], axis=1)
    network.generators_t.p_max_pu.columns = generators_to_add

    # Save the network to a file
    network.export_to_netcdf(snakemake.output[0])

    # Export the network to a folder of CSVs (if needed)
    network.export_to_csv_folder('network_csvs')


if __name__ == '__main__':
    df1 = atlite_to_normalised_timeseries_bus('Wind Onshore', snakemake.input[2], REPD_data(), snakemake.input[1])
    df2 = atlite_to_normalised_timeseries_bus('Solar Photovoltaics', snakemake.input[3], REPD_data(), snakemake.input[1])
    df3 = atlite_to_normalised_timeseries_bus('Wind Offshore', snakemake.input[4], REPD_data(), snakemake.input[1])
    
    add_to_network(df1, df2, df3)

