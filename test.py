from sandpile import BTW
import numpy as np


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


def test_check_height():
    grid_test = np.array([[0, 0, 0], 
                         [0, 4, 0], 
                         [0, 0, 0]])
    grid_control_1 = np.array([[0, 0, 0], 
                              [0, 0, 0], 
                              [0, 0, 0]])

    btw = BTW([3, 3], 4, 0)
    btw.grid = grid_test
    btw.check_height()
    assert np.all(btw.grid == grid_control_1), f'Grid not correctly updated. \n {btw.grid} \n {grid_control_1}'
