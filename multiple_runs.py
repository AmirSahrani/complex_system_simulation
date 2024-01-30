from sandpile import BTW
from config import kwargs_oscillatory

# Varying parameter
# height_values = range(1, 10)

# for height in height_values:
#     current_kwargs = kwargs_oscillatory.copy()
#     current_kwargs['height'] = height

#     btw = BTW(grid_size=[50, 50], **current_kwargs)

#     btw.run(10000)
#     btw.write_data(f"data/varying_height_{height}.csv")

refractory_period_values = range(1, 10)

for refractory_period in refractory_period_values:
    current_kwargs = kwargs_oscillatory.copy()
    current_kwargs['refractory_period'] = refractory_period

    btw = BTW(grid_size=[50, 50], **current_kwargs)

    btw.run(10000)
    btw.write_data(f"data/varying_refractory_{refractory_period}.csv")