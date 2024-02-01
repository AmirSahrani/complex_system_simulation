# import multiprocessing
# from sandpile import BTW

# def run_simulation(params, steps, file_name):
#     btw = BTW(**params)
#     btw.run(steps)
#     btw.write_data(file_name)

# def simulation_wrapper(args):
#     param_name, param_value = args
#     grid_size = [50, 50]
#     steps = 10000
#     params = {
#         "grid_size": grid_size,
#         "height": 8,
#         "refractory_period": 8,
#         "probability_of_spontaneous_activity": 0.03,
#         "max_distance": 3,
#         "visualize": False,
#         "random_connection": False
#     }

#     params[param_name] = param_value

#     file_name = f"data/new_varying_{param_name}_{param_value}_h_8.csv"
#     run_simulation(params, steps, file_name)


# def run_multiprocess_simulation(param_name, param_values):
#     args_list = [(param_name, value) for value in param_values]

#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
#     pool.map(simulation_wrapper, args_list)

#     pool.close()
#     pool.join()

# if __name__ == "__main__":
#     run_multiprocess_simulation("refractory_period", range(1, 10))

# import multiprocessing
# from sandpile import CA_continuous_threshold
# import numpy as np

# def run_simulation(params, steps, file_name):
#     model = CA_continuous_threshold(**params)
#     model.run(steps)
#     model.write_data(file_name)


# def simulation_wrapper(args):
#     param_name, param_value = args
#     grid_size = [50, 50]
#     steps = 10000
#     params = {
#         "grid_size": grid_size, 
#         "height": 3,
#         "refractory_period": 5,
#         "probability_of_spontaneous_activity": 0.03,
#         "max_distance": 3,
#         "visualize": False,
#         "random_connection": False,
#         "threshold": 2.5
#     }

#     params[param_name] = param_value

#     file_name = f"data/new_varying_{param_name}_{param_value}.csv"
#     run_simulation(params, steps, file_name)

# def run_multiprocess_simulation(param_name, param_values):
#     args_list = [(param_name, value) for value in param_values]

#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
#     pool.map(simulation_wrapper, args_list)

#     pool.close()
#     pool.join()

# if __name__ == "__main__":
#     threshold_values = np.arange(1.0, 8.1, 0.1)
#     run_multiprocess_simulation("threshold", threshold_values)

import multiprocessing
from sandpile import BTW
from utils.data_utils import branching_prameter  # Ensure correct import path
import pandas as pd

def run_simulation(params, steps, file_name):
    btw = BTW(**params)
    btw.run(steps)
    btw.write_data(file_name)
    return file_name  # Return the filename for further processing

def simulation_wrapper(args):
    refractory_period, height = args
    grid_size = [50, 50]
    steps = 10000
    params = {
        "grid_size": grid_size,
        "height": height,
        "refractory_period": refractory_period,
        "probability_of_spontaneous_activity": 0.02,
        "max_distance": 3,
        "visualize": False,
        "random_connection": False
    }

    file_name = f"data/ref_{refractory_period}_height_{height}.csv"
    run_simulation(params, steps, file_name)

    # Moved data processing outside of this function for multiprocessing compatibility

    return (refractory_period, height, file_name)  # Return tuple for post-processing

def calculate_branching_parameters(results):
    branching_params_table = pd.DataFrame(index=[rp for rp, _, _ in results], columns=[h for _, h, _ in results])
    for rp, h, file_name in results:
        data = pd.read_csv(file_name)
        sigma = branching_prameter(data)  # Ensure function is correctly named and imported
        branching_params_table.loc[rp, h] = sigma
    return branching_params_table

def run_multiprocess_simulation(refractory_periods, heights):
    args_list = [(rp, h) for rp in refractory_periods for h in heights]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(simulation_wrapper, args_list)

    branching_params_table = calculate_branching_parameters(results)
    branching_params_table.to_csv('branching_params_table.csv')

if __name__ == "__main__":
    refractory_periods = range(1, 8)  
    heights = range(1, 7)  
    run_multiprocess_simulation(refractory_periods, heights)