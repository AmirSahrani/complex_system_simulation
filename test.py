from cellular_automata import CA
from utils.data_utils import *
from pandas.testing import assert_frame_equal
from branching import *
from utils.utils import *
import numpy as np
import pandas as pd
import pytest
import csv
import random


def test_init_grid():
    '''
    Test the init_grid method of the CA class.
    Currently the random method relies on not accidentally picking the same point twice.
    '''
    ca = CA([10, 10], 4)
    ca.init_grid("random", 5)
    assert np.where(ca.grid > 0)[
        0].shape[0] == 5, "Grid not initialized correctly using random method."

    ca.init_grid("custom", 5, lambda x: x + 5)
    grid_surface_area = ca.grid.shape[0] * ca.grid.shape[1]
    assert np.sum(ca.grid) == 5 * \
        grid_surface_area, "Grid not initialized correctly using custom method."


def test_add_grain():
    ca = CA([20, 20], probability_of_spontaneous_activity=0.03)
    ca.add_grain()
    sum_spikes = np.count_nonzero(ca.grid)
    assert sum_spikes > 0 and sum_spikes < 100, "Grains are not added correctly."


def test_run():
    ca = CA(grid_size=[10, 10], threshold=4,
            probability_of_spontaneous_activity=0.03, max_distance=2, visualize=False)

    num_steps = 100
    ca.run(num_steps)

    assert len(
        ca.spikes_neighbours) == num_steps, "Incorrect length of spikes_neighbours list."
    assert len(
        ca.spikes_total) == num_steps, "Incorrect length of spikes_total list."

    for neighbour_spikes, total_spikes in zip(ca.spikes_neighbours, ca.spikes_total):
        assert total_spikes >= 0, "Total spikes should be non-negative."
        assert total_spikes >= neighbour_spikes, "Total spikes should be greater than or equal to input spikes."


def test_writing():
    kwargs = {"grid_size": [6, 6], "threshold": 6, "max_distance": 6, "refractory_period": 6,
              "probability_of_spontaneous_activity": 0.06, "random_connection": False}
    ca = CA(**kwargs)
    ca.spikes_neighbours = [1, 2, 3]
    ca.spikes_total = [7, 5, 9]
    ca.write_data(path="data/test")
    data_read = load_data_csv(path="data/test")

    # Check if the data is correct
    assert data_read['grid_size'][0] == 36, "Grid size is not correct."
    assert data_read['threshold'][0] == 6, "Height is not correct."
    assert data_read['max_distance'][0] == 6, "Max distance is not correct."
    assert data_read['refractory_period'][0] == 6, "Refractory period is not correct."
    assert data_read['probability_of_spontaneous_activity'][0] == 0.06, "Probability of spontaneous activity is not correct."
    assert not data_read['random_connection'][0], "Random connection is not correct."
    assert data_read['spikes_total'].tolist(
    ) == [7, 5, 9], "Total spikes are not correct."
    assert data_read['spikes_neighbours'].tolist(
    ) == [1, 2, 3], "Neighbour spikes are not correct."
    assert data_read['spikes_input'].tolist(
    ) == [6, 3, 6], "Input spikes are not correct."


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
        ca = CA([i, i], threshold=4, max_distance=1.5)
        ca.grid = test
        ca.check_neighbors()
        assert np.all(
            ca.grid == control), f'Grid not correctly updated. \n {ca.grid} \n {control}'


def test_avg_spike_density_noref():
    df = pd.DataFrame({
        'time_step': [0, 1, 2, 3, 4],
        'spikes_total': [2, 2, 3, 3, 5],
        'spikes_neighbours': [1, 1, 2, 2, 3],
        'spikes_input': [1, 1, 1, 1, 2]})
    a = avg_spike_density(df, 10)
    assert a == 0.03


def test_spike_density_withref():
    df = pd.DataFrame({
        'time_step': [0, 1, 2, 3, 4],
        'spikes_total': [2, 2, 3, 3, 5],
        'spikes_neighbours': [1, 1, 2, 2, 3],
        'spikes_input': [1, 1, 1, 1, 2]})
    refractory_period = 2
    a = ref_avg_spike_density(df, 10, refractory_period)
    assert np.round(a, 4) == 0.0313


def test_branching_parameter():
    df = pd.DataFrame({
        'time_step': [0, 1, 2, 3, 4, 5, 6],
        'spikes_total': [0, 2, 0, 1, 4, 4, 0],
        'spikes_neighbours': [0, 0, 0, 0, 2, 3, 0],
        'spikes_input': [0, 2, 0, 1, 2, 1, 0],
    })
    df_2 = pd.DataFrame({
        'time_step': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'spikes_total': [0, 2, 0, 1, 4, 4, 0, 1, 2],
        'spikes_neighbours': [0, 0, 0, 0, 2, 3, 0, 0, 1],
        'spikes_input': [0, 2, 0, 1, 2, 1, 0, 1, 1],
    })

    a = branching_prameter(df)
    b = branching_prameter(df_2)
    assert a == 0.6875
    assert b == 0.75


def avalanche_size_duration(df):
    # Implementation of the avalanche_size_duration function
    pass


def test_avalanche_size_duration():
    df = pd.DataFrame({
        'spikes_total': [0, 2, 0, 1, 4, 4, 0, 1, 2, 3],
        'spikes_neighbours': [0, 0, 0, 0, 2, 3, 0, 0, 1, 2],
        'spikes_input': [0, 2, 0, 1, 2, 1, 0, 1, 1, 1],
    })
    expected_size = [9, 6]
    expected_duration = [3, 3]
    actual_size, actual_duration = avalanche_distributions(df)
    assert actual_size == expected_size and actual_duration == expected_duration


def test_transmission_to_avalanche():
    df = pd.DataFrame({
        'time': [0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
        'ancestor': [168, 279, 13, 15, 15, 22, 22, 22, 26, 26],
        'descendant': [15, 22, 22, 199, 198, 116, 26, 204, 11, 71],
    })
    expected = [[1, 1, 2], [2, 1, 3, 2]]
    actual = transmission_to_avalanche(df)
    assert actual == expected


def test_avalanche_to_statistics():
    avalanches = [[1, 3, 2], [2, 1, 2, 1], [1, 1, 2, 1, 1]]
    expected = pd.DataFrame({'size': [6, 6, 6], 'duration': [3, 4, 5]})
    actual = avalanche_to_statistics(avalanches)
    assert (actual == expected).all().all()


def test_init_network():
    network = BranchingNeurons(20, 3, 1, False)
    assert len(network.neurons) == 20, "Incorrect number of neurons"
    assert all(len(neuron.neighbors) <=
               3 for neuron in network.neurons), "Incorrect number of neighbors"


def test_neuron():
    branching_ratio = 3
    neuron = Neuron((1, 1), branching_ratio)
    neuron.neighbors = [Neuron((1, 2), 3), Neuron(
        (2, 1), 3), Neuron((2, 2), 3)]
    neuron.generate_probabilities()
    assert len(neuron.probabilities) == 3, "Incorrect number of probabilities"
    assert np.isclose(sum(neuron.probabilities.values()),
                      branching_ratio), "Probabilities don't sum to branching ratio"


def test_density():
    array = [2, 2, 2, 1, 1, 3]
    true_density = [2/6, 3/6, 1/6]
    values, density = get_density(array)
    assert all([x == y for x, y in zip([1, 2, 3], values)]), "Incorrect values"
    for true, test in zip(true_density, density):
        assert np.isclose(true, test), "Incorrect density"


def test_mutual_info():
    inpt = list(range(1, 20))
    output1 = list(range(2, 21))
    output2 = [random.randint(1, 10) for _ in range(19)]
    mi1 = mutual_info(inpt, output1)
    mi2 = mutual_info(inpt, output2)
    assert mi1 > mi2, "Incorrect mutual information"


def test_dynamic_range():
    output = [1, 3, 1, 2, 1, 1, 4, 2, 3, 2]
    dr = dynamic_range(output)
    expected = [[1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1]]


def test_susceptibility():
    spike_history = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    neuron_num = 2
    expected = 0.06
    sus = susceptibility(spike_history, neuron_num)
    assert round(sus, 2) == expected, "Incorrect susceptibility"
