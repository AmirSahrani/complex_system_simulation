from typing import Callable
from tqdm import tqdm


def simulate(simulation: Callable, n_runs: int, duration: int, **kwargs):
    results = []
    for _ in tqdm(range(n_runs)):
        sim = simulation(**kwargs)
        sim.run(duration)
        results.append([sim.evalanche_size, sim.evalanche_duration])
        sim.reset()
    return results
        
