from sandpile import BTW
from utils.data_utils import *
import numpy as np
import os
import csv


def test_init_grid():
    '''
    Test the init_grid method of the BTW class.
    Currently the random method relies on not accidentally picking the same point twice.
    '''
    btw = BTW([10, 10], 4, 0)
    btw.init_grid("random", 5)
    assert np.where(btw.grid > 0)[0].shape[0] == 5, "Grid not initialized correctly using random method."

    btw.init_grid("center", 5)
    assert np.sum(btw.grid) == 5, "Grid not initialized correctly using center method."

    btw.init_grid("custom", 5, lambda x: x + 1)
    grid_surface_area = btw.grid.shape[0] * btw.grid.shape[1]
    assert np.sum(btw.grid) == grid_surface_area, "Grid not initialized correctly using custom method."


def test_grain():
    btw = BTW([3, 3], 4, 0)
    btw.add_grain()
    assert np.sum(btw.grid) == 1


def test_writing():
    # Create an instance of the BTW class
    btw = BTW([10, 10], 4, 0)
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

    # Clean up: remove the test file
    os.remove("data/avalanches.csv")
    
    


if __name__ == "__main__":
    unittest.main()






    # btw = BTW([4, 4], 4, 0)
    # btw.grid = grid_test_2
    # btw.check_neighbors()
    # assert np.all(btw.grid == grid_cont_2), f'Grid not correctly updated. \n {btw.grid} \n {grid_cont_2}'

    # btw = BTW([5, 5], 4, 0)
    # btw.grid = grid_test_3
    # btw.check_neighbors()
    # assert np.all(btw.grid == grid_cont_3), f'Grid not correctly updated. \n {btw.grid} \n {grid_cont_3}'
