from typing import Callable
from tqdm import tqdm
import numpy as np


def simulate(simulation: Callable, n_runs: int, duration: int, **kwargs):
    results = []
    for _ in tqdm(range(n_runs)):
        sim = simulation(**kwargs)
        sim.run(duration)
        results.append([sim.evalanche_size, sim.evalanche_duration])
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