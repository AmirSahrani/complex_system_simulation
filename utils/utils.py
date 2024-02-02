from typing import Callable
from tqdm import tqdm
import numpy as np


# Configurations for showing different patterns of neural activity.
kwargs_round_spiral = {"threshold": 4, "refractory_period": 5, "probability_of_spontaneous_activity": 0.02, "max_distance": 3, "visualize": True, "random_connection": False}
kwargs_pulse_wave = {"threshold": 5, "refractory_period": 4, "probability_of_spontaneous_activity": 0.03, "max_distance": 3, "visualize": True, "random_connection": False}
kwargs_synchronous = {"threshold": 3, "refractory_period": 5, "probability_of_spontaneous_activity": 0.015, "max_distance": 2.5, "visualize": True, "random_connection": True}
kwargs_oscillatory = {"threshold": 2, "refractory_period": 4, "probability_of_spontaneous_activity": 0.02, "max_distance": 3, "visualize": True, "random_connection": False}
kwargs_repeating = {"threshold": 2, "refractory_period": 4, "probability_of_spontaneous_activity": 0.02, "max_distance": 3, "visualize": True, "random_connection": True}
kwargs_random = {"threshold": 5, "refractory_period": 5, "probability_of_spontaneous_activity": 0.02, "max_distance": 3, "visualize": True, "random_connection": False}


def simulate(simulation: Callable, n_runs: int, duration: int, **kwargs):
    results = []
    for i in range(n_runs):
        sim = simulation(**kwargs)
        sim.run(duration)
        results.append({
            "run": i,
            "evalanche_size": sim.evalanche_size,
            "evalanche_duration": sim.evalanche_duration,
            "density": np.mean(sim.density),
            "cooldown": sim.cooldown,
            'branching_ratio': kwargs['branching_ratio'],
            'emperical_branching_ratio': np.mean(sim.branching)
        })
        sim.reset()
    return results
        
def get_density(data):
    density = []
    values = np.unique(data)
    for value in values:
        density.append(np.sum(data == value) / len(data))
    return values,density

def closest_index_to_value(array, value):
    return np.argmin(np.abs(array - value))