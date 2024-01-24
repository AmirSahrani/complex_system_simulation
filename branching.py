import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from tqdm import tqdm


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

        for origin in neuron.origins:
            origin.activated_neighbors.remove(neuron)

        for neighbor in neuron.neighbors:
            if not neighbor.active and np.random.random() < neighbor.activation_probability:
                neighbor.origins.append(neuron)
                neighbor.active = 1
                self.next_active.append(neighbor)
                neuron.activated_neighbors.append(neighbor)

        self.active.remove(neuron)

    def reset(self):
        for neuron in self.neurons:
            neuron.active = 0
            neuron.activated_neighbors = []
        self.active = []
    
    def random_activation(self):
        for neuron in self.neurons:
            if np.random.random() < neuron.activation_probability and not neuron.active:
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
                    self.ax.plot([neuron.location[0], neighbor.location[0]], [neuron.location[1], neighbor.location[1]], c="red")
            else:
                self.ax.scatter(neuron.location[0], neuron.location[1], c="blue")

    def run(self, steps: int):
        self.reset()
        self.random_activation()
        for i in tqdm(range(steps)):

            for neuron in self.active:
                self.propage_activations(neuron)
                neuron.active = 0

            self.active = self.next_active.copy()
            self.next_active = []

            if self.visual:
                self.plot()
                plt.pause(0.001)
    
    def plot_evalanche_size(self):
        plt.plot(self.evalanche_size)
        plt.show()
    
    def plot_evalanche_duration(self):
        plt.plot(self.evalanche_duration)
        plt.show()


if __name__ == "__main__":
    bn = BranchingNeurons(connection_probability=0.1, N=10, visual=True)
    bn.run(300)