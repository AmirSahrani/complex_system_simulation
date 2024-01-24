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
        self.avalanches_sizes = []
        self.avalanches_durations = []

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

    
    def add_grain(self) -> None:
        """Add a grain to a random point on the grid."""
        #TODO: make it so this function activates neurons like the book/paper says
        grid_point = (np.random.randint(0, self.grid.shape[0]), np.random.randint(0, self.grid.shape[1]))
        self.grid[grid_point] += 1

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

        for i in range(steps):
            self.add_grain()

            avalanche_duration = 0
            
            while self.grid.max() >= self.max_height:
                self.check_neighbors()
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
        '''Writes self.avalanches_sizes and self_avalanches_durations to one csv file'''
        with open("data/avalanches.csv", "w") as f:
            f.write("size,duration\n")
            for size, duration in zip(self.avalanches_sizes, self.avalanches_durations):
                f.write(f"{size},{duration}\n")



if __name__ == "__main__":
    btw = BTW(grid_size=[100, 100], height=4, offset=2, visualize=False)
    btw.init_grid("random", 5)
    btw.run(10000)
