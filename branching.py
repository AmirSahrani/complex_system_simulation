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
        self.probabilities = {}

        self.generate_probabilities()

    def generate_probabilities(self):
        n_probs = len(self.neighbors)
        rand_vec = np.random.random(n_probs)
        rand_vec /= np.sum(rand_vec)
        rand_vec *= 1
        for neighbor, prob in zip(self.neighbors, rand_vec):
            self.probabilities[neighbor] = prob


class Avalanche_graph():
    """
    Graph representing an avalanche.
    """
    def __init__(self, neuron: Neuron) -> None:
        self.nodes = [neuron]
        self.edges = []
        self.ad_matrix: csr_matrix
        self.neuron_edge_dict = {}

    def add_node(self, neuron: Neuron) -> None:
        self.nodes.append(neuron)


    def add_edge(self, neuron_a: Neuron, neuron_b: Neuron) -> None:
        self.edges.append((neuron_a, neuron_b))
    

    def generate_neuron_edge_dict(self):
        for i,node in enumerate(set(self.nodes)):
            self.neuron_edge_dict[node] = i


    def edges_to_indices(self):
        for i, (node_a, node_b) in enumerate(self.edges):
            self.edges[i] = (self.neuron_edge_dict[node_a], self.neuron_edge_dict[node_b])

    def adjacency_matrix(self):
        """
        Returns the adjacency matrix of the avalanche.
        """
        self.generate_neuron_edge_dict()
        self.edges_to_indices()
        dim = len(set(self.nodes))
        rows, cols = zip(*self.edges)  
        data = np.ones(len(rows), dtype=np.int8)
        self.ad_matrix = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.int8)

    def get_diameter(self):
        if len(self.nodes) == 2:
            return 1

        self.adjacency_matrix()
        dist = dijkstra(self.ad_matrix)
        dist[dist == np.inf] = 0

        return np.max(dist[0, :])
    


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

    def __init__(self, N:int, max_neighbors: int, visual: bool=False) -> None:

        self.neurons = [Neuron(tuple(np.random.random(2)), 
                               np.random.random()) for i in range(N)]
        self.visual = visual
        self.max_neighbors = max_neighbors

        self.evalanche_size = []
        self.evalanche_duration = []
        self.active = []
        self.next_active = []  
        self.branching = []

        self.init_network()
        if visual:
            self.setup_plot()
    
    def init_network(self) -> None:

        for neuron in self.neurons:
            neighbors = np.random.choice(self.neurons, self.max_neighbors - len(neuron.neighbors), replace=False).tolist()
            neuron.neighbors = neighbors
            for neighbor in neighbors:
                neighbor.neighbors.append(neuron)
            neuron.generate_probabilities()
        
        assert all([len(neuron.neighbors) == self.max_neighbors for neuron in self.neurons]), "Not all neurons have the same number of neighbors."

                

    def propage_activations(self, neuron: Neuron):

        assert neuron.active, "Neuron is not active."

        for neighbor in neuron.neighbors:
            if not neighbor.active and np.random.random() < neuron.probabilities[neighbor]:
                if not neighbor.cooldown:

                    neuron.avalanche.add_node(neighbor)
                    neuron.avalanche.add_edge(neuron, neighbor)
                    neighbor.avalanche = neuron.avalanche
                    neuron.avalanche = Avalanche_graph(neuron)

                    neighbor.active = 1
                    self.next_active.append(neighbor)
                    neuron.activated_neighbors.append(neighbor)

        if not neuron.activated_neighbors and len(neuron.avalanche.nodes) > 1:
            self.evalanche_size.append(len(neuron.avalanche.nodes))
            self.evalanche_duration.append(neuron.avalanche.get_diameter())
            neuron.avalanche = Avalanche_graph(neuron)


    def reset(self):
        for neuron in self.neurons:
            neuron.active = 0
            neuron.activated_neighbors = []
        self.active = []
    
    def random_activation(self):
        for neuron in self.neurons:
            if np.random.random() < 1e-5 and not neuron.active:
                neuron.active = 1
                self.active.append(neuron)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
    
    def plot(self):
        self.ax.clear()
        for neuron in self.neurons:
            if neuron.active:
                if len(neuron.avalanche.nodes) > 0:
                    self.ax.scatter(neuron.location[0], neuron.location[1], s= len(neuron.avalanche.nodes)*2, c="blue")
                    for neighbor in neuron.activated_neighbors:
                        self.ax.plot([neuron.location[0], neighbor.location[0]], [neuron.location[1], neighbor.location[1]], c="blue", linewidth=0.7)
                else:
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
                neuron.cooldown = 0

            for neuron in self.neurons:
                if neuron.cooldown:
                    neuron.cooldown -= 1

            if self.visual:
                self.plot()
                plt.pause(0.5)

            if self.active:
                self.branching.append(len(self.next_active)/len(self.active))

            self.active = self.next_active.copy()
            self.next_active = []

    def plot_evalanche_size(self):
        plt.plot(self.evalanche_size)
        plt.show()
    
    def plot_evalanche_duration(self):
        plt.plot(self.evalanche_duration)
        plt.show()
    

if __name__ == "__main__":
    sim = BranchingNeurons(N=1000, max_neighbors=7, visual=False)
    sim.run(10000)
    print(f'Max avalance size: {max(sim.evalanche_size)}\nMax avalance duration: {max(sim.evalanche_duration)}')
    print(f'Mean branching ratio: {np.mean(sim.branching)}')