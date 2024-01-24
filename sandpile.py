import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt


class BTW():
    """Bak-Tang-Wiesenfeld sandpile model.
    Initialize with a grid size, model will be initialized with all zeros.
    """
    def __init__(self, grid_size :List,  height: int, offset: int, visualize: bool=False) -> None:
        self.grid = np.zeros(grid_size) + offset
        self.max_height = height
        self.offset = offset
        self.direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.visualize = visualize

        self.avalanches = []

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

        if any(sum(self.grid-self.offset)):
            self.grid = np.zeros(self.grid.shape) + self.offset

        grid_points = (np.random.randint(0, self.grid.shape[0], size=(N)), np.random.randint(0, self.grid.shape[0], size=(N)))
        if method == "random":
            self.grid[grid_points[0], grid_points[1]] += np.random.randint(self.max_height)
        elif method == "center":
            for i in range(N):
                self.grid[self.grid.shape[0] // 2, self.grid.shape[1] // 2] += 1
                self.check_height()
                if self.visualize:
                    self.plot()
        elif method == "custom":
            self.grid = func(self.grid)

    
    def add_grain(self) -> None:
        """Add a grain to a random point on the grid."""
        grid_point = (np.random.randint(0, self.grid.shape[0]), np.random.randint(0, self.grid.shape[1]))
        self.grid[grid_point] += 1


    def check_height(self) -> None:
        """
        Check if any points on the grid are over the critical height. 
        Any points on the edges "fall off" the grid.
        """
        toppled = np.where(self.grid >= self.max_height)

        for location in zip(*toppled):

            self.grid[location] -= self.max_height
            self.grid[location[0] + 1, location[1]] += 1
            self.grid[location[0] - 1, location[1]] += 1
            self.grid[location[0], location[1] + 1] += 1
            self.grid[location[0], location[1] - 1] += 1

        self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = 0 



    def run(self, steps: int) -> None:
        for i in range(steps):
            self.add_grain()

            avalanche_duration = 0
            
            while self.grid.max() >= self.max_height:
                self.check_height()
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



if __name__ == "__main__":
    btw = BTW(grid_size=[100, 100], height=4, offset=2, visualize=False)
    btw.init_grid("random", 5)
    btw.run(10000)
    with open("data/avalanches.txt", "w") as f:
        f.write(",".join([str(i) for i in btw.avalanches]))
