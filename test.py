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
    btw = BTW(grid_size=[10, 10], height=4, probability_of_spontaneous_activity=0.03, max_distance=2, visualize=False)

    num_steps = 100
    btw.run(num_steps)
    
    assert len(btw.spikes_neighbours) == num_steps, "Incorrect length of spikes_neighbours list."
    assert len(btw.spikes_total) == num_steps, "Incorrect length of spikes_total list."

    for neighbour_spikes, total_spikes in zip(btw.spikes_neighbours, btw.spikes_total):
        assert total_spikes >= 0, "Total spikes should be non-negative."
        assert total_spikes >= neighbour_spikes, "Total spikes should be greater than or equal to input spikes."



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
    kwargs = {"grid_size": [6, 6], "height": 6, "max_distance": 6, "refractory_period": 6, "probability_of_spontaneous_activity": 0.06, "random_connection": False}
    btw = BTW(**kwargs)
    btw.spikes_neighbours = [1, 2, 3]
    btw.spikes_total = [7, 5, 9]
    btw.write_data(path="data/test")
    data_read = load_data_csv(path="data/test")

    # Check if the data is correct
    assert data_read['grid_size'][0] == 36, "Grid size is not correct."
    assert data_read['height'][0] == 6, "Height is not correct."
    assert data_read['max_distance'][0] == 6, "Max distance is not correct."
    assert data_read['refractory_period'][0] == 6, "Refractory period is not correct."
    assert data_read['probability_of_spontaneous_activity'][0] == 0.06, "Probability of spontaneous activity is not correct."
    assert data_read['random_connection'][0] == False, "Random connection is not correct."
    assert data_read['spikes_total'].tolist() == [7, 5, 9], "Total spikes are not correct."
    assert data_read['spikes_neighbours'].tolist() == [1, 2, 3], "Neighbour spikes are not correct."
    assert data_read['spikes_input'].tolist() == [6, 3, 6], "Input spikes are not correct."


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
    
def test_branching_parameter():
    df = pd.DataFrame({
    'time_step':[0,1,2,3,4,5,6],
    'spikes_total': [0,2,0,1,4,4,0],
    'spikes_neighbors': [0,0,0,0,2,3,0],
    'spikes_input': [0,2,0,1,2,1,0],
    })
    df_2 = pd.DataFrame({
    'time_step':[0,1,2,3,4,5,6,7,8,9],
    'spikes_total': [0,2,0,1,4,4,0,1,2,3],
    'spikes_neighbors': [0,0,0,0,2,3,0,0,1,2],
    'spikes_input': [0,2,0,1,2,1,0,1,1,1],
    })
    
    a = branching_prameter(df)
    b = branching_prameter(df_2)
    assert a == 0.6875
    assert b == 0.6875

