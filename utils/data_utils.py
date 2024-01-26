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
    return pd.read_csv(path)

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
def avg_spike_density(data:pd.DataFrame, size:int) -> float:
    """Calculate the average spike density."""
    # avg_spake_density is defined as avg[spikes_total / the number of neurons] in each time step
    return np.mean(data['spikes_total']) / float(size)**2
    # avg_spake_density is defined as avg[spikes_total / （the number of neurons - the sum of spikes in last timestep)] in each timestep
    #!! not sure whether it's needed
    for time_step in data['timestep'].unique():
        data.loc[data['timestep'] == time_step, 'refractory_sum'] = data.loc[
            (data['timestep'] < time_step) & 
            (data['timestep'] >= time_step - refractory_period),
            ['spikes_neighbours', 'spikes_input']
        ].sum().sum()
    
def branching_prameter(df: pd.DataFrame) -> float:
    """
    Calculates the branching parameter sigma.
    Sigma is defined as the ratio of next timestep's spikes_neighbours to this timestep's spikes_total,
    excluding cases where spikes_total is or NaN. The result is divided by the count of non-null spikes_total.
    """
    df_copy = df.copy()

    # Check if the last row(s) of df have zeros in spikes_neighbors
    # If yes, truncate these rows
    while df_copy['spikes_neighbors'].iloc[-1] != 0:
        df_copy = df_copy.iloc[:-1]
    
    df_copy['next_spikes_neighbors'] = df_copy['spikes_neighbors'].shift(-1)
    df_copy['ratio'] = df_copy['next_spikes_neighbors'] / df_copy['spikes_total'] 
    valid_ratios = df_copy['ratio'][df_copy['spikes_total'] > 0]
    valid_ratios_sum = valid_ratios.sum()
    non_zero_spikes_total_count = (df_copy['spikes_total'] > 0).sum()
    sigma = valid_ratios_sum / non_zero_spikes_total_count if non_zero_spikes_total_count > 0 else 0

    return sigma


    return paths