from sandpile import BTW
from utils.data_utils import *
from pandas.testing import assert_frame_equal
import numpy as np
import pandas as pd
import pytest
import csv

def test_init_grid():
    '''
    Test the init_grid method of the BTW class.
    Currently the random method relies on not accidentally picking the same point twice.
    '''
    btw = BTW([10, 10], 4)
    btw = BTW([10, 10], 4)
    btw.init_grid("random", 5)
    assert np.where(btw.grid > 0)[0].shape[0] == 5, "Grid not initialized correctly using random method."

    btw.init_grid("center", 5)
    assert np.sum(btw.grid) == 5, "Grid not initialized correctly using center method."

    btw.init_grid("custom", 5, lambda x: x + 1)
    grid_surface_area = btw.grid.shape[0] * btw.grid.shape[1]
    assert np.sum(btw.grid) == grid_surface_area, "Grid not initialized correctly using custom method."


def test_add_grain():
    btw = BTW([20, 20], probability_of_spontaneous_activity=0.03)
    btw.add_grain()
    sum_spikes = np.count_nonzero(btw.grid)
    assert sum_spikes > 0 and sum_spikes < 100, "Grains are not added correctly."


def test_run():
    btw = BTW(grid_size[10, 10], height=4, probability_of_spontaneous_activity=0.03, max_distance=2, visualize=False)

    num_steps = 100
    btw.run(num_steps)
    
    assert len(btw.spikes_input) == num_steps, "Incorrect length of spikes_input list."
    assert len(btw.spikes_total) == num_steps, "Incorrect length of spikes_total list."

    for input_spikes, total_spikes in zip(btw.spikes_input, btw.spikes_total):
        assert total_spikes >= 0, "Total spikes should be non-negative."
        assert total_spikes >= input_spikes, "Total spikes should be greater than or equal to input spikes."



def test_check_neighbors():
    grid_test_1 = np.array([[4, 0, 4], 
                            [0, 0, 0], 
                            [4, 0, 4]])
    grid_cont_1 = np.array([[0, 0, 0], 
                            [0, 4, 0], 
                            [0, 0, 0]])

    grid_test_2 = np.array([[4, 0, 4, 0],
                            [0, 4, 0, 4],
                            [4, 0, 4, 0],
                            [0, 4, 0, 4]])
    grid_cont_2 = np.array([[0, 0, 0, 0],
                            [0, 0, 4, 0],
                            [0, 4, 0, 0],
                            [0, 0, 0, 0]])

    grid_test_3 = np.array([[0, 4, 0, 4, 0],
                            [0, 4, 0, 4, 0],
                            [0, 0, 0, 0, 0],
                            [0, 4, 0, 4, 0],
                            [0, 0, 0, 0, 0]])
    grid_cont_3 = np.array([[0, 0, 4, 0, 0],
                            [0, 0, 4, 0, 0],
                            [0, 0, 4, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

    
    for i, (test, control) in enumerate(zip([grid_test_1, grid_test_2, grid_test_3], [grid_cont_1, grid_cont_2, grid_cont_3]), start=3):
        btw = BTW([i, i], height=4, max_distance=1.5)
        btw.grid = test
        btw.check_neighbors()
        assert np.all(btw.grid == control), f'Grid not correctly updated. \n {btw.grid} \n {control}'
        

def test_writing():
    btw = BTW([10, 10], 4)
    # Initialize the grid
    btw.init_grid("random", 5)
    # Initialize the BTW class with durations and sizes
    btw.spikes_input = [1, 2, 3]
    btw.spikes_neighbours = [7,5,9]
    # Write the grid to a file
    btw.write_data(path="data/spikes_btw.csv")
    # Read the csv data
    with open("data/spikes_btw.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        data_read = [(int(time_steps), int(spikes_input), int(spikes_total), int(spikes_neighbors)) for time_steps, spikes_input, spikes_total, spikes_neighbors in reader]

    # Prepare expected data for comparison
    expected_data = [(i, btw.spikes_input[i], btw.spikes_input[i]+btw.spikes_neighbours[i], btw.spikes_neighbours[i]) for i in range(len(btw.spikes_input))]

    # Check if the data is correct
    assert data_read == expected_data

def test_check_neighbors():
    grid_test_1 = np.array([[4, 0, 4], 
                            [0, 0, 0], 
                            [4, 0, 4]])
    grid_cont_1 = np.array([[0, 0, 0], 
                            [0, 4, 0], 
                            [0, 0, 0]])

    grid_test_2 = np.array([[4, 0, 4, 0],
                            [0, 4, 0, 4],
                            [4, 0, 4, 0],
                            [0, 4, 0, 4]])
    grid_cont_2 = np.array([[0, 0, 0, 0],
                            [0, 0, 4, 0],
                            [0, 4, 0, 0],
                            [0, 0, 0, 0]])

    grid_test_3 = np.array([[0, 4, 0, 4, 0],
                            [0, 4, 0, 4, 0],
                            [0, 0, 0, 0, 0],
                            [0, 4, 0, 4, 0],
                            [0, 0, 0, 0, 0]])
    grid_cont_3 = np.array([[0, 0, 4, 0, 0],
                            [0, 0, 4, 0, 0],
                            [0, 0, 4, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

    
    for i, (test, control) in enumerate(zip([grid_test_1, grid_test_2, grid_test_3], [grid_cont_1, grid_cont_2, grid_cont_3]), start=3):
        btw = BTW([i, i], height=4, max_distance=1.5)
        btw.grid = test
        btw.check_neighbors()
        assert np.all(btw.grid == control), f'Grid not correctly updated. \n {btw.grid} \n {control}'

# def test_to_bin():
#     df = pd.DataFrame({
#     'timestep': [1, 2, 3, 4, 5, 6],
#     'spikes_input': [1, 1, 2, 2, 3, 1],
#     'spikes_neighbours': [1, 1, 2, 2, 3, 1]})
#     df_to_bin = pd.DataFrame({
#     'bin':[1,2,3],
#     'spikes_input': [2, 4, 4],
#     'spikes_neighbours': [2, 4, 4]
#     })
#     binned_df = to_bin(df, 2)
#     # assert df_to_bin is the same as binned_df
#     assert_frame_equal(df_to_bin, binned_df)

def test_avg_spike_density_noref():
    df = pd.DataFrame({
    'time_step': [0, 1, 2, 3, 4],
    'spikes_total': [2,2,3,3,5],
    'spikes_neighbours': [1, 1, 2, 2, 3],
    'spikes_input': [1, 1, 1, 1, 2]})
    a = avg_spike_density(df, 10)
    assert a == 0.03
    
def test_spike_density_withref():
    df = pd.DataFrame({
    'time_step': [0, 1, 2, 3, 4],
    'spikes_total': [2,2,3,3,5],
    'spikes_neighbours': [1, 1, 2, 2, 3],
    'spikes_input': [1, 1, 1, 1, 2]})
    refractory_period = 2
    a = ref_avg_spike_density(df, 10, refractory_period)
    assert a == 0.0313
def test_branching_parameter():
    df = pd.DataFrame({
    'time_step':[0,1,2,3,4,5,6],
    'spikes_total': [0,2,0,1,4,4,0],
    'spikes_neighbours': [0,0,0,0,2,3,0],
    'spikes_input': [0,2,0,1,2,1,0],
    })
    df_2 = pd.DataFrame({
    'time_step':[0,1,2,3,4,5,6,7,8],
    'spikes_total': [0,2,0,1,4,4,0,1,2],
    'spikes_neighbours': [0,0,0,0,2,3,0,0,1],
    'spikes_input': [0,2,0,1,2,1,0,1,1],
    })
    
    a = branching_prameter(df)
    b = branching_prameter(df_2)
    assert a == 0.6875
    assert b == 0.75

