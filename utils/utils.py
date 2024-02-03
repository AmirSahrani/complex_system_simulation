from typing import Callable
from tqdm import tqdm
import numpy as np
import multiprocessing
from cellular_automata import CA


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


def run_simulation(params, steps, file_name):
    ca = CA(**params)
    ca.run(steps)
    ca.write_data(file_name)


def simulation_wrapper(args):
    param_name, param_value = args
    grid_size = [50, 50]
    steps = 10000
    params = {
        "grid_size": grid_size,
        "threshold": 8,
        "refractory_period": 8,
        "probability_of_spontaneous_activity": 0.03,
        "max_distance": 3,
        "visualize": False,
        "random_connection": False
    }

    params[param_name] = param_value

    file_name = f"data/new_varying_{param_name}_{param_value}_h_8.csv"
    run_simulation(params, steps, file_name)


def run_multiprocess_simulation(param_name, param_values):
    args_list = [(param_name, value) for value in param_values]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(simulation_wrapper, args_list)

    pool.close()
    pool.join()


if __name__ == "__main__":
    run_multiprocess_simulation("refractory_period", range(1, 10))