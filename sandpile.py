import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt


class BTW():
    """Bak-Tang-Wiesenfeld sandpile model.
    Initialize with a grid size, model will be initialized with all zeros.
    """
    def __init__(self, grid_size :List, neighbors: int) -> None:
        self.grid = np.zeros(grid_size)
        self.n_neighbors = neighbors

    def init_grid(self, method: str, N: int, func: Optional[callable] = None) -> None:
        """
        Initialize the grid with a method.
        Methods can be:
            Random: Randomly assign N grains to random points on the grid.
            Center: Assign N grains to the center of the grid.
            Custom: Use a custom function to initialize the grid.
        """
        assert method in ["random", "center", "custom"], "Invalid method."
        assert N > 0, "N must be positive."

        grid_points = list(zip(*np.random.randint(0, self.grid.shape[0], size=(N, 2))))
        if method == "random":
            self.grid[*grid_points] = np.random.randint(0, 4, N)
        elif method == "center":
            self.grid[self.grid.shape[0]//2, self.grid.shape[1]//2] = 4
        elif method == "custom":
            self.grid = func["grid"]
