from typing import Callable
from tqdm import tqdm
import numpy as np


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