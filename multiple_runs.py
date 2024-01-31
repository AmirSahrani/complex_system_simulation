import multiprocessing
from sandpile import BTW

def run_simulation(params, steps, file_name):
    btw = BTW(**params)
    btw.run(steps)
    btw.write_data(file_name)

def simulation_wrapper(args):
    param_name, param_value = args
    grid_size = [50, 50]
    steps = 10000
    params = {
        "grid_size": grid_size,
        "height": 8,
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