from sandpile import BTW
from utils.data_utils import *
import numpy as np
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
    
    assert len(btw.spikes_input) == num_steps, "Incorrect length of spikes"

def test_writing():
    btw = BTW([10, 10], 4)
    # Initialize the grid
    btw.init_grid("random", 5)
    # Initialize the BTW class with durations and sizes
    btw.avalanches_sizes = [1, 2, 3]
    btw.avalanches_durations = [1, 2, 3]
    # Write the grid to a file
    btw.write_data()
    # Read the csv data
    with open("data/avalanches.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        data_read = [(int(size), int(duration)) for size, duration in reader]

    # Prepare expected data for comparison
    expected_data = list(zip(btw.avalanches_sizes, btw.avalanches_durations))

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
