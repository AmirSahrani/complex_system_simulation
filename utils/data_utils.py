import numpy as np
import csv as csv
import pandas as pd
from sandpile import BTW
import os
from typing import List


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


def merge_trees(tree1: list, tree2: list) -> list:
    """
    Merge two trees.
    """
    # Make sure tree1 is the smaller tree
    if len(tree1) > len(tree2):
        tree1, tree2 = tree2, tree1
    tree1 = tree1[::-1]  
    tree2 = tree2[::-1]
    
    # Merge tree1 and tree2 from bottom up
    new_tree = tree2.copy()
    for i, row in enumerate(tree1):
        for node in row:
            if node not in new_tree[i]:
                new_tree[i].append(node)

    return new_tree[::-1]


def transmission_to_avalanche(transmission_df: pd.DataFrame) -> list:
    """
    Convert the transmission data to avalanche data.
    """
    df = transmission_df.copy()
    avalanches = []
    trees = []

    for i, t in enumerate(df['time'].unique()):     # Loop through all time steps
        current_df = df[df['time']==t]
        merge_operations = []

        for _, row in current_df.iterrows():    # Loop through all transmissions in the current time step
            target_tree_index = []
            for j, tree_dic in enumerate(trees):    # Loop through all trees
                # If the tree is from a previous time step, update the time step and add a new row to the tree
                if tree_dic['time'] != t:
                    tree_dic['time'] = t
                    tree_dic['tree'].append([])
                # If the ancestor of this transmission is in the tree's last time step, add the descendant to the tree
                if row['ancestor'] in tree_dic['tree'][-2]:
                    tree_dic['tree'][-1].append(row['descendant'])
                    target_tree_index.append(j)
            # If the ancestor of this transmission is not in any tree, create a new tree
            if len(target_tree_index) == 0:
                trees.append({'time': t, 'tree': [[row['ancestor']], [row['descendant']]]})
            # Record merge operations if the ancestor is in more than one tree
            elif len(target_tree_index) > 1 and target_tree_index not in merge_operations:
                merge_operations.append(target_tree_index)

        # Define a set to store the indices of trees to remove after merging
        trees_to_remove = set()

        # Merge trees
        for indices in merge_operations:
            # Merge trees and update the list
            merged_tree = trees[indices[0]]['tree']
            for index in indices[1:]:
                merged_tree = merge_trees(merged_tree, trees[index]['tree'])
                # Add the index of the tree to the remove set
                trees_to_remove.add(index)

            # Update the first tree with the merged tree
            trees[indices[0]]['tree'] = merged_tree

        # Remove trees from the end of the list to avoid index errors
        for index in sorted(trees_to_remove, reverse=True):
            trees.pop(index)

        # Check if any trees have no descendant in this time step and remove them
        trees_to_remove = []
        for j, tree_dic in enumerate(trees):
            # If the tree is empty, record its index
            if not tree_dic['tree'][-1]:
                trees_to_remove.append(j)

        # Remove trees from the end of the list to avoid index errors
        for index in sorted(trees_to_remove, reverse=True):
            complete_tree_dic = trees.pop(index)
            complete_tree_dic['tree'].pop(-1)   # Remove the empty row
            complete_avalanche = [len(row) for row in complete_tree_dic['tree']]
            avalanches.append(complete_avalanche)

    # Check if any trees are left at the end
    if trees:
        for tree_dic in trees:
            complete_avalanche = [len(row) for row in tree_dic['tree']]
            avalanches.append(complete_avalanche)

    return avalanches


def avalanche_to_statistics(avalanches: list) -> pd.DataFrame:
    """
    Convert the avalanche data to avalanche size and duration.
    """
    statistics = []
    for avalanche in avalanches:
        size = sum(avalanche)
        duration = len(avalanche)
        statistics.append([size, duration])
    return pd.DataFrame(statistics, columns=['size', 'duration'])

def str_to_list(s):
    if not isinstance(s, str):
        return s
    if s == '[]':
        return []
    elif ',' not in s:
        return [float(s)]
    return [float(x) for x in s.strip('[]').split(',')]

def write_data(data: List, file_name: str) -> None:
    """Write data to a CSV file."""
    if os.path.exists(file_name):
        mode = "a"
    else:
        mode = "w"
    with open(file_name, mode) as f:
        header = list(data[0].keys())
        if mode == "w":
            f.write(",".join(header) + "\n")
        for run in data:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(run)