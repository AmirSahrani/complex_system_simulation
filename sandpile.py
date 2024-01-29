import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from config import *


class BTW():
    """Bak-Tang-Wiesenfeld sandpile model.
    Initialize with a grid size, model will be initialized with all zeros.
    """
    def __init__(self, grid_size: List, height: int=2, visualize: bool=False, max_distance: float=3, refractory_period: int=3, probability_of_spontaneous_activity: float=0.02, random_connection: bool=False) -> None:
        self.grid = np.zeros(grid_size)
        self.max_height = height
        self.direction = []
        self.visualize = visualize
        self.refractory_period = refractory_period
        self.refractory_matrix = np.zeros(grid_size)
        self.spikes_total = []
        self.spikes_neighbours = []
        self.probability_of_spontaneous_activity = probability_of_spontaneous_activity
        self.random_connection = random_connection
        self.max_distance = max_distance

        if random_connection:
            self.generate_rand_conn()
        else:
            self.neighbormap()
        if self.visualize:
            self.cm = plt.get_cmap("viridis", self.max_height + 1)
            self.setup_plot()


    def init_grid(self, method: str, N: Optional[int], func: Optional[callable] = None) -> None:
        """
        Initialize the grid with a method.
        Methods can be:
            Random: Randomly assign N grains to random points on the grid.
            Center: Assign N grains to the center of the grid.
            Custom: Use a custom function to initialize the grid.
        """
        assert method in ["random", "center", "custom"], "Invalid method."
        assert N > 0, "N must be positive."

        if any(sum(self.grid)):
            self.grid = np.zeros(self.grid.shape)

        grid_points = (np.random.randint(0, self.grid.shape[0], size=(N)), np.random.randint(0, self.grid.shape[0], size=(N)))
        if method == "random":
            self.grid[grid_points[0], grid_points[1]] = self.max_height
        elif method == "center":
            for i in range(N):
                self.grid[self.grid.shape[0] // 2, self.grid.shape[1] // 2] = self.max_height
                self.check_neighbors()
                if self.visualize:
                    self.plot()
        elif method == "custom":
            self.grid = func(self.grid)


    def generate_rand_conn(self) -> None:
        """
        Generate random connections between neurons.
        """
        n = 0   # Number of connections
        for x in range(int(np.floor(-self.max_distance)), int(np.ceil(self.max_distance+1))):
            for y in range(int(np.floor(-self.max_distance)), int(np.ceil(self.max_distance+1))):
                if x**2 + y**2 <= self.max_distance**2 and (x, y) != (0, 0):
                    n += 1  # Count the number of connections
        N = self.grid.shape[0] * self.grid.shape[1] # Number of neurons
        numbers = list(range(N))    # List of numbers to randomly choose from
        conn_matrix = np.zeros((N, n)) # Initialize a matrix to store connections
        for i in range(N):
            conn_matrix[i, :] = random.sample(numbers, n) # Randomly choose n numbers from the list
        conn_matrix = conn_matrix.astype(int)
        self.direction = conn_matrix


    def add_grain(self) -> None:
        """Add a grain to a random point on the grid."""
        # Loop through all neurons in the grid
        # Check neurons not in the refractory period
        not_in_ref = self.refractory_matrix == 0
        # Randomly activate neurons based on probability
        add_matrix = np.random.random(self.grid.shape) < self.probability_of_spontaneous_activity
        # Activate neurons that are not in refractory and have been randomly chosen
        self.grid[not_in_ref & add_matrix] = self.max_height


    def neighbormap(self) -> None:
        """
        Generate a relative position map of neighbors.
        """
        for x in range(int(np.floor(-self.max_distance)), int(np.ceil(self.max_distance+1))):
            for y in range(int(np.floor(-self.max_distance)), int(np.ceil(self.max_distance+1))):
                if x**2 + y**2 <= self.max_distance**2 and (x, y) != (0, 0):
                    self.direction.append((x, y))


    def check_neighbors(self) -> None:
        """
        Check if any points on the grid are over the critical height. If so, topple them.
        """
        toppled = np.where(self.grid >= self.max_height)

        if self.random_connection:
            for location in zip(*toppled):
                self.grid[location] = 0
                self.refractory_matrix[location] = self.refractory_period + 1
                index = location[0] * self.grid.shape[0] + location[1]
                for neighbor in self.direction[index]:
                    x, y = neighbor // self.grid.shape[0], neighbor % self.grid.shape[0]
                    if x >= 0 and x < self.grid.shape[0] and y >= 0 and y < self.grid.shape[1] and self.refractory_matrix[x, y] == 0:
                        self.grid[x, y] += 1
        else:
            for location in zip(*toppled):
                self.grid[location] = 0
                self.refractory_matrix[location] = self.refractory_period + 1
                for d in self.direction:
                    x, y = location[0] + d[0], location[1] + d[1]
                    if x >= 0 and x < self.grid.shape[0] and y >= 0 and y < self.grid.shape[1] and self.refractory_matrix[x, y] == 0:
                        self.grid[x, y] += 1
        self.grid = self.grid >= self.max_height
        self.grid = self.grid.astype(int) * self.max_height


    def run(self, steps: int) -> None:
        """
        Run the model for a number of steps.
        """
        for i in range(steps):
            # Initialize variables for avalanche statistics
            total_spikes = 0
            neighbour_spikes = 0

            # Clear all active neurons from the previous step and update neighbor spikes
            self.check_neighbors()
            # Question: HOw do we now these non-zero grids are neighbour spikes, and not added grain from the previous step?
            neighbour_spikes = np.sum(self.grid != 0)

            # Generate some spontaneous spikes randomly and update total spikes
            self.add_grain()
            total_spikes = np.sum(self.grid != 0)

            # Save the number of spikes for this step
            self.spikes_total.append(total_spikes)
            self.spikes_neighbours.append(neighbour_spikes)

            if self.visualize:
                self.plot()

            self.refractory_matrix[self.refractory_matrix > 0] -= 1


    def setup_plot(self) -> None:
        self.fig, self.ax = plt.subplots()
        self.fig.colorbar(plt.cm.ScalarMappable(cmap=self.cm), ax=self.ax)


    def plot(self) -> None:
        self.ax.imshow(self.grid, cmap=self.cm)
        plt.pause(0.001)
        self.ax.clear()


    def write_data(self, path: str) -> None:
        """
        Writes single set of self.spikes_neighbors, self.spikes_total and spikes_input to one csv file.
        """
        args = {"grid_size": self.grid.shape[0] * self.grid.shape[1], 
                "height": self.max_height, 
                "max_distance": self.max_distance, 
                "refractory_period": self.refractory_period, 
                "probability_of_spontaneous_activity": self.probability_of_spontaneous_activity, 
                "random_connection": self.random_connection}
        args_df = pd.DataFrame(args, index=[0])

        spikes_total, spikes_neighbours = np.array(self.spikes_total), np.array(self.spikes_neighbours)
        results_df = pd.DataFrame({"spikes_total": spikes_total, 
                                "spikes_neighbours": spikes_neighbours, 
                                "spikes_input": spikes_total - spikes_neighbours})
        
        combined_df = pd.concat([args_df, results_df], axis=1)
        combined_df.to_csv(path, index=True)
        print("Data written to: ", path)



if __name__ == "__main__":
    btw = BTW(grid_size=[100, 100], **kwargs_round_spiral)
    # btw.init_grid("random", 4)
    btw.run(10000)
    btw.write_data("data/different_max_height_2.csv")