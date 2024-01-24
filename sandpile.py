import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt


class BTW():
    """Bak-Tang-Wiesenfeld sandpile model.
    Initialize with a grid size, model will be initialized with all zeros.
    """
    def __init__(self, grid_size :List,  height: int, offset: int, visualize: bool=False, max_distance: int=1) -> None:
        self.grid = np.zeros(grid_size) + offset
        self.max_height = height
        self.offset = offset
        self.direction = []
        self.visualize = visualize
        self.refractory_period = 3
        self.refractory_matrix = np.zeros(grid_size)
        self.probability_of_spontaneous_activity = 0.02
        self.avalanches = []

        self.cm = plt.get_cmap("viridis", self.max_height + 1)
        self.setup_plot()
        self.neighbormap(max_distance)


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

        if any(sum(self.grid-self.offset)):
            self.grid = np.zeros(self.grid.shape) + self.offset

        grid_points = (np.random.randint(0, self.grid.shape[0], size=(N)), np.random.randint(0, self.grid.shape[0], size=(N)))
        if method == "random":
            self.grid[grid_points[0], grid_points[1]] += np.random.randint(self.max_height)
        elif method == "center":
            for i in range(N):
                self.grid[self.grid.shape[0] // 2, self.grid.shape[1] // 2] += 1
                self.check_neighbors()
                if self.visualize:
                    self.plot()
        elif method == "custom":
            self.grid = func(self.grid)

    def calculate_avalanch_size(self):



    
    def (self) -> None:
        """Add a grain to a random point on the grid."""
        # TODO: make it so this function activates neurons like the book/paper says
        # Loop through all neurons in the grid
        # Check neurons not in the refractory period
        not_in_ref = self.refractory_matrix == 0
        # Randomly activate neurons based on probability
        add_matrix = np.random.random(self.grid.shape) < self.probability_of_spontaneous_activity
        # Activate neurons that are not in refractory and have been randomly chosen
        self.grid[not_in_ref & add_matrix] += 1

    def neighbormap(self, max_distance) -> None:
        for x in range(-max_distance, max_distance+1):
            for y in range(-max_distance, max_distance+1):
                if abs(x)**2 + abs(y)**2 <= max_distance**2 and (x, y) != (0, 0):
                    self.direction.append((x, y))

    def check_neighbors(self) -> None:
        """
        Check if any points on the grid are over the critical height. 
        Any points on the edges "fall off" the grid.
        """
        #TODO: Make this function check neighbors according to the rules in the book/paper
        toppled = np.where(self.grid >= self.max_height)

        for location in zip(*toppled):
            self.grid[location] = 0
            self.refractory_matrix[location] = self.refractory_period
            for d in self.direction:
                x, y = location[0] + d[0], location[1] + d[1]
                if x >= 0 and x < self.grid.shape[0] and y >= 0 and y < self.grid.shape[1] and self.refractory_matrix[x, y] == 0:
                    self.grid[x, y] += 1


    def run(self, steps: int) -> None:
        #TODO: Add docstring and tweak it so it works with the new check_neighbors function
        #TODO: Revise avalanche size/duration counting

        # Initialize a list to store the size of each avalanche
        

        for i in range(steps):
            # Initialize a variable for the current avalanche size
            avalanche_sizes = []
            current_avalanche_size = 0

            # Save the current state of the gride before adding grains
            prev_grid_state = np.copy(self.grid)

            self.add_grain()

            avalanche_duration = 0
            
            while self.grid.max() >= self.max_height:
                self.check_neighbors()

                # Count the number of neurons activated at this step
                # and add it to the current avalanche size
                newly_activated = (self.grid > prev_grid_state) & (prev_grid_state < self.max_height)
                current_avalanche_size += np.sum(newly_activated)

                # Record the current avalanche size if it's greater than 0
                if current_avalanche_size > 0:
                    avalanche_sizes.append(current_avalanche_size)

                if self.visualize:
                    self.plot()
                avalanche_duration += 1

            if avalanche_duration > 0:
                self.avalanches.append(avalanche_duration)

    def setup_plot(self) -> None:
        self.fig, self.ax = plt.subplots()
        self.fig.colorbar(plt.cm.ScalarMappable(cmap=self.cm), ax=self.ax)


    def plot(self) -> None:
        self.ax.imshow(self.grid, cmap=self.cm)
        plt.pause(0.001)
        self.ax.clear()


    def write_data(self) -> None:
        '''Writes data to file'''
        # TODO: Make it so this function doesn't overwrite anything
        with open("data/avalanches.txt", "w") as f:
            f.write(",".join([str(i) for i in self.avalanches]))




if __name__ == "__main__":
    btw = BTW(grid_size=[100, 100], height=4, offset=2, visualize=False)
    btw.init_grid("random", 5)
    btw.run(10000)
