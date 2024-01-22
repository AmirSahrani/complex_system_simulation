import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt


class BTW():
    """Bak-Tang-Wiesenfeld sandpile model.
    Initialize with a grid size, model will be initialized with all zeros.
    """
    def __init__(self, grid_size :List,  height: int) -> None:
        self.grid = np.zeros(grid_size) + 1
        self.max_height = height

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

        grid_points = (np.random.randint(0, self.grid.shape[0], size=(N)), np.random.randint(0, self.grid.shape[0], size=(N)))
        if method == "random":
            self.grid[*grid_points] = np.random.randint(0, 4, N)
        elif method == "center":
            self.grid[self.grid.shape[0]//2, self.grid.shape[1]//2] = 4
        elif method == "custom":
            self.grid = func["grid"]
    

    
    def add_grain(self) -> None:
        """Add a grain to a random point on the grid."""
        grid_point = (np.random.randint(0, self.grid.shape[0]), np.random.randint(0, self.grid.shape[0]))
        self.grid[grid_point] += 1


    def check_height(self) -> None:
        """Check if any points on the grid are over the critical height."""
        toppled = np.where(self.grid > self.max_height)
        self.grid[toppled] -= self.n_neighbors
        
        for i in range(len(toppled[0])):
            location = [toppled[0][i], toppled[1][i]]
            location[0] = location[0] % self.grid.shape[0] -1
            location[1] = location[1] % self.grid.shape[1] - 1

            self.grid[location[0] + 1, location[1]] += 1
            self.grid[location[0] - 1, location[1]] += 1
            self.grid[location[0], location[1] + 1] += 1
            self.grid[location[0], location[1] - 1] += 1



    def run(self, steps: int, visualize: bool =False) -> None:
        for i in range(steps):
            self.add_grain()
            self.check_height()

            if visualize and i > 15000:
                plt.title(f"Step {i}")
                plt.cla()
                plt.imshow(self.grid)
                plt.draw()
                plt.pause(0.001)



if __name__ == "__main__":
    btw = BTW([100, 100], 4, 4)
    btw.init_grid("random", 1000)
    btw.run(20000, True)