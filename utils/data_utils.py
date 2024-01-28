import numpy as np
import csv as csv
import pandas as pd
from sandpile import BTW
import os
def load_data_txt(path: str) -> list:
    """Load data from a file."""
    with open(path, "r") as f:
        data = f.readlines()[0]
        data = [int(x.strip()) for x in data.split(",")]
    return data


def load_data_csv(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path, index_col=0)

# def to_bin(data: pd.DataFrame, bin_size: int) -> pd.DataFrame:
#     """Convert data(timestep,spikes_input, spikes_neighbours) to in bin units."""
#     data['bin'] = (data['timestep'] - 1) // bin_size + 1 #!! to be determined if the logic works for float timesteps'
#     df = data.groupby('bin').agg({'spikes_input': 'sum', 'spikes_neighbours': 'sum'}).reset_index()
#     return df

# def avg_spike_density_gird(data: pd.DataFrame, data_bin:pd.DataFrame, size: int, refractory_period: int, bin_size: int) -> float:
#     """Calculate the average spike density."""
#     # avg_spake_density is defined as avg[spikes_neighbours / （the number of neurons - the sum of spikes in last bin)] in each bin
#     data['bin'] = (data['timestep'] - 1) // bin_size + 1
#     data_bin['refractory_sum'] = 0
#     for bin_number in data_bin['bin'].unique():
#         bin_refractory_sum = 0
#         timesteps_in_bin = data[data['bin'] == bin_number]['timestep']
#         for timestep in timesteps_in_bin:
#             bin_refractory_sum += data.loc[
#                 (data['timestep'] < timestep) & 
#                 (data['timestep'] >= timestep - refractory_period),
#                 ['spikes_neighbours', 'spikes_input']
#             ].sum().sum()
#         data_bin.loc[data_bin['bin'] == bin_number, 'refractory_sum'] = bin_refractory_sum
#     data_bin['density'] = data_bin['spikes_neighbours'] / (size**2 - data_bin['refractory_sum'])
#     return data_bin['density'].mean()

    # avg_spake_density is defined as avg[spikes_neighbours / the number of neurons] in each bin
    #return np.mean(data_bin['spikes_neighbours'] / size**2 ])
def ref_avg_spike_density(data:pd.DataFrame, size:int, refractory_period:int) -> float:
    """Calculate the average spike density."""
    # avg_spake_density is defined as avg[spikes_total / the number of neurons] in each time step
    # return np.mean(data['spikes_total']) / float(size)**2
    # avg_spake_density is defined as avg[spikes_total / （the number of neurons - the sum of spikes in last timestep)] in each timestep
    densities = []

    for index, row in data.iterrows():
        if row['spikes_neighbours'] != 0:
            if int(index) - refractory_period < 0:
                refractory_sum = data.iloc[:int(index)]['spikes_total'].sum()
            else:
                refractory_sum = data.iloc[int(index)-refractory_period:int(index)]['spikes_total'].sum()
            if size**2 - refractory_sum > 0:
                current_density = row['spikes_total'] / (size**2 - refractory_sum)
                densities.append(current_density)
    avg_density = np.mean(densities) if densities else 0
    return avg_density

def avg_spike_density(data:pd.DataFrame, size:int) -> float:
    """Calculate the average spike density."""
    # avg_spake_density is defined as avg[spikes_total / the number of neurons] in each time step
    # Only count rows where spikes_neighbours is not zero
    filtered_data = data[data['spikes_neighbours'] != 0]
    avg_density = np.mean(filtered_data['spikes_total']) / (size ** 2)
    print(avg_density)
    return avg_density
    
def branching_prameter(df: pd.DataFrame) -> float:
    """
    Calculates the branching parameter sigma.
    Sigma is defined as the ratio of next timestep's spikes_neighbours to this timestep's spikes_total,
    excluding cases where spikes_total is or NaN. The result is divided by the count of non-null spikes_total.
    """
    df_copy = df.copy()

    # Check if the last number of spikes_neighbours is nonzero
    # If it's nonzero, add a new row with 0 spikes_neighbours after it
    if df_copy['spikes_neighbours'].iloc[-1] != 0:
        new_row = {col: 0 for col in df_copy.columns}
        new_row_df = pd.DataFrame([new_row])
        df_copy = pd.concat([df_copy, new_row_df], ignore_index=True)
    
        df_copy['next_spikes_neighbours'] = df_copy['spikes_neighbours'].shift(-1)
        df_copy['ratio'] = df_copy['next_spikes_neighbours'] / df_copy['spikes_total'] 
        valid_ratios = df_copy['ratio'][df_copy['spikes_total'] > 0]
        valid_ratios_sum = valid_ratios.sum()
        non_zero_spikes_total_count = (df_copy['spikes_total'] > 0).sum()
        sigma = valid_ratios_sum / (non_zero_spikes_total_count-1) if non_zero_spikes_total_count > 0 else 0
    else:
        df_copy['next_spikes_neighbours'] = df_copy['spikes_neighbours'].shift(-1)
        df_copy['ratio'] = df_copy['next_spikes_neighbours'] / df_copy['spikes_total'] 
        valid_ratios = df_copy['ratio'][df_copy['spikes_total'] > 0]
        valid_ratios_sum = valid_ratios.sum()
        non_zero_spikes_total_count = (df_copy['spikes_total'] > 0).sum()
        sigma = valid_ratios_sum / non_zero_spikes_total_count if non_zero_spikes_total_count > 0 else 0

    return sigma


def avalanche_distributions(df: pd.DataFrame) -> (list, list):
    sizes = []
    durations = []
    avalanche_in_progress = False
    current_size = 0
    current_duration = 0

    for index, row in df.iterrows():
        # Start of DataFrame or end of an avalanche
        if (index == 0 or not avalanche_in_progress) and row['spikes_input'] > 0 and row['spikes_neighbours'] == 0:
            # Look ahead to check if the next row has spikes_neighbours > 0 to confirm the start of an avalanche
            if index < len(df) - 1 and df.iloc[index + 1]['spikes_neighbours'] > 0:
                avalanche_in_progress = True
                current_size = row['spikes_total']  # Start counting the size
                current_duration = 1  # Reset duration to 1 for a new avalanche
            continue  # Skip to the next iteration
        
        # If already in an avalanche
        if avalanche_in_progress:
            # If there are neighbours in the current timestep, add to the size
            if row['spikes_neighbours'] > 0:
                current_size += row['spikes_total']
                current_duration += 1
            # If there are no neighbours, it's the end of the current avalanche
            if row['spikes_neighbours'] == 0:
                avalanche_in_progress = False
                sizes.append(current_size)
                durations.append(current_duration)
                current_size = 0
                current_duration = 0
                # Immediately check if this is also the start of a new avalanche
                if row['spikes_input'] > 0:
                    if index < len(df) - 1 and df.iloc[index + 1]['spikes_neighbours'] > 0:
                        avalanche_in_progress = True
                        current_size = row['spikes_total']
                        current_duration = 1
                else:
                    avalanche_in_progress = False
    
    # Check if an avalanche was in progress at the end and didn't get a chance to end
    if avalanche_in_progress:
        sizes.append(current_size)
        durations.append(current_duration)
        
    sizes = [size for size in sizes if size > 0]
    durations = [duration for duration in durations if duration > 0]

    return sizes, durations


def one_dim_neighbormap(max_distance: float, N: int) -> list:
    """
    Generate a 1D relative position map of neighbors.
    """
    neighbour_map = []
    for x in range(int(np.floor(-max_distance)), int(np.ceil(max_distance+1))):
        for y in range(int(np.floor(-max_distance)), int(np.ceil(max_distance+1))):
            if x**2 + y**2 <= max_distance**2 and (x, y) != (0, 0):
                one_dim_index = N * x + y
                neighbour_map.append(one_dim_index)
    return neighbour_map


def raster_to_transmission(raster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raster data to transmission data.
    """
    assert raster_df['random_connection'].iloc[0] == False, "Random connection not supported yet."
    df = raster_df.copy()

    # Get the neighbour map
    max_distance = df['max_distance'].iloc[0]
    N = np.sqrt(df['grid_size'].iloc[0])
    neighbour_map = one_dim_neighbormap(max_distance, N)

    # Drop parameter columns
    param_cols = ['max_distance', 'grid_size', 'random_connection', 'refractory_period', 'height', 'probability_of_spontaneous_activity']
    df = df.drop(columns=param_cols)

    transmission = []
    for index, row in df.iterrows():    # Loop through all time steps
        not_last_row = index != len(df) - 1
        active_neurons_index = [int(col) for col in df.columns if row[col] != 0]
        if not_last_row:
            next_row = df.iloc[index + 1]
            for i in active_neurons_index:  # Loop through all active neurons, i is the 1D index of the active neuron
                for j in neighbour_map:    # Loop through all neighbors of the active neuron
                    n = int(i + j)   # 1D index of the neighbor
                    if n >= 0 and n < len(row) and next_row[n] == 2:
                        transmission.append([index, i, n])  # [time, active neuron -> neighbor]
    return pd.DataFrame(transmission, columns=['time', 'ancestor', 'descendant'])


def transmission_to_avalanche(transmission_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the transmission data to avalanche data.
    """
    df = transmission_df.copy()
    