import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from tqdm import tqdm
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


class Neuron():
    """
    Represents a neuron in a complex system simulation.

    Attributes:
        neighbors (list): List of neighboring neurons.
        origins (list): List of neurons that activate this neuron.
        active (int): Indicator of whether the neuron is currently active (1) or not (0).
        location (Tuple): Location of the neuron.
        activated_neighbors (list): List of neighbors that have activated this neuron.
        cooldown (int): Cooldown period after activation.
        activation_probability (float): Probability of the neuron getting activated.
    """

    def __init__(self, location: Tuple, activation_p: float) -> None:
        self.neighbors = []
        self.origins = []
        self.active = 0
        self.location = location
        self.activated_neighbors = []
        self.cooldown = 0
        self.activation_probability = activation_p
        self.avalanche: Avalanche_graph =Avalanche_graph(self)


class Avalanche_graph():
    """
    Graph representing an avalanche.
    """
    def __init__(self, neuron: Neuron) -> None:
        self.nodes = [neuron]
        self.edges = []
        self.ad_matrix: csr_matrix
        self.neuron_counter = 0

    def add_node(self, neuron: Neuron) -> None:
        self.nodes.append(neuron)
        self.neuron_counter += 1

    def add_edge(self, neuron_a: Neuron, neuron_b: Neuron) -> None:
        self.edges.append((self.neuron_counter, self.neuron_counter + 1))
    
    def adjacency_matrix(self):
        """
        Returns the adjacency matrix of the avalanche.
        """
        dim = self.neuron_counter + 2
        rows, cols = zip(*self.edges)  
        data = np.ones(len(rows), dtype=np.int8)
        self.ad_matrix = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.int8)

    def get_diameter(self):
        if len(self.nodes) == 2:
            return 1

        self.adjacency_matrix()
        dist = dijkstra(self.ad_matrix)
        dist[dist == np.inf] = 0

        return np.max(dist)
    


class BranchingNeurons():
    """
    Class representing a branching model for neurons.

    Attributes:
        neurons (list): List of Neuron objects representing the neurons in the network.
        p (float): Probability of connection between two neurons.
        visual (bool): Boolean indicating whether to visualize a run of the simulation.
        evalanche_size (list): List to store the size of each avalanche during the simulation.
        evalanche_duration (list): List to store the duration of each avalanche during the simulation.
        active (list): List of active neurons during the simulation.

    Methods:
        __init__(self, connection_probability: float, N:int, spread_probability: float, visual: bool=False) -> None:
            Initializes the Branching model for neurons.

        init_network(self) -> None:
            Initializes the network by connecting neurons based on the connection probability.

        propage_activations(self):
            Propagates activations through the network until no more activations are possible.

        reset(self):
            Resets the state of the neurons and the active neuron list.

        random_activation(self):
            Randomly activates neurons based on the connection probability.

        setup_plot(self):
            Sets up the plot for visualization.

        plot(self):
            Plots the current state of the network.

        run(self, steps):
            Runs the simulation for the specified number of steps.

        plot_evalanche_size(self):
            Plots the size of each avalanche during the simulation.

        plot_evalanche_duration(self):
            Plots the duration of each avalanche during the simulation.
    """

    def __init__(self, N:int, connection_probability: float, 
                 visual: bool=False) -> None:

        self.neurons = [Neuron(tuple(np.random.random(2)), 
                               np.random.random()) for i in range(N)]
        self.connection_probability = connection_probability
        self.visual = visual

        self.evalanche_size = []
        self.evalanche_duration = []
        self.active = []
        self.next_active = []  
        self.branching = []

        self.init_network()
        if visual:
            self.setup_plot()
    
    def init_network(self) -> None:
        for neuron_a in self.neurons:
            for neuron_b in self.neurons:
                if np.random.random() < self.connection_probability:
                    neuron_a.neighbors.append(neuron_b)
                    neuron_b.neighbors.append(neuron_a) 

    def propage_activations(self, neuron: Neuron):

        assert len(neuron.origins) <= 1, "Neuron has more than one origin."
        for origin in neuron.origins:
            if neuron in origin.activated_neighbors:
                origin.activated_neighbors.remove(neuron)

        for neighbor in neuron.neighbors:
            if not neighbor.active and np.random.random() < neighbor.activation_probability:
                if not neighbor.cooldown:

                    neuron.avalanche.add_node(neighbor)
                    neuron.avalanche.add_edge(neuron, neighbor)
                    neighbor.avalanche = neuron.avalanche
                    neuron.avalanche = Avalanche_graph(neuron)

                    neighbor.origins.append(neuron)
                    neighbor.active = 1
                    self.next_active.append(neighbor)
                    neuron.activated_neighbors.append(neighbor)
                else:
                    neuron.cooldown -= 1
        self.branching.append(len(neuron.activated_neighbors))
        if not neuron.activated_neighbors and len(neuron.avalanche.nodes) > 1:
            self.evalanche_size.append(len(neuron.avalanche.nodes))
            self.evalanche_duration.append(neuron.avalanche.get_diameter())
            neuron.avalanche = Avalanche_graph(neuron)
            neuron.origins = []


    def reset(self):
        for neuron in self.neurons:
            neuron.active = 0
            neuron.activated_neighbors = []
        self.active = []
    
    def random_activation(self):
        for neuron in self.neurons:
            if np.random.random() < 1e-1 and not neuron.active:
                neuron.active = 1
                self.active.append(neuron)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
    
    def plot(self):
        self.ax.clear()
        for neuron in self.neurons:
            if neuron.active:
                self.ax.scatter(neuron.location[0], neuron.location[1], c="red")
                for neighbor in neuron.activated_neighbors:
                    self.ax.plot([neuron.location[0], neighbor.location[0]], [neuron.location[1], neighbor.location[1]], c="red", linewidth=0.7)
            else:
                self.ax.scatter(neuron.location[0], neuron.location[1], c="gray")
                for neighbor in neuron.neighbors:
                    self.ax.plot([neuron.location[0], neighbor.location[0]], [neuron.location[1], neighbor.location[1]], c="gray", alpha=0.2, linewidth=0.2)


    def run(self, steps: int):
        for i in tqdm(range(steps)):
            self.random_activation()

            for neuron in self.active:

                self.propage_activations(neuron)
                neuron.active = 0
                neuron.cooldown = 1

            self.active = self.next_active.copy()
            self.next_active = []

            if self.visual:
                self.plot()
                plt.pause(0.00001)

    def plot_evalanche_size(self):
        plt.plot(self.evalanche_size)
        plt.show()
    
    def plot_evalanche_duration(self):
        plt.plot(self.evalanche_duration)
        plt.show()
    

if __name__ == "__main__":
    sim = BranchingNeurons(1000, 0.05, visual=True)
    sim.run(1000)
    print(sum(sim.evalanche_size)/len(sim.evalanche_size), sum(sim.evalanche_duration)/len(sim.evalanche_duration), sum(sim.branching)/len(sim.branching))