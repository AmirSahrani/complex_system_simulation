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
def avg_spike_density(data:pd.DataFrame, size:int, refractory_period: int) -> float:
    """Calculate the average spike density."""
    # avg_spake_density is defined as avg[spikes_total / the number of neurons] in each time step
    return np.mean(data['spikes_total']) / float(size)**2
    # avg_spake_density is defined as avg[spikes_total / （the number of neurons - the sum of spikes in last timestep)] in each timestep
    # densities = []

    # for index, row in data.iterrows():
    #     if index - refractory_period < 0:
    #         refractory_sum = data.iloc[:index]['spikes_total'].sum()
    #     else:
    #         refractory_sum = data.iloc[index-refractory_period:index]['spikes_total'].sum()
    #     if size**2 - refractory_sum > 0:
    #         current_density = row['spikes_total'] / (size**2 - refractory_sum)
    #         densities.append(current_density)
    # avg_density = np.mean(densities) if densities else 0
    # return avg_density

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
    sigma = valid_ratios_sum / non_zero_spikes_total_count if non_zero_spikes_total_count > 0 else 0

    return sigma