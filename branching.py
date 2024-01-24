import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class Neuron():
    def __init__(self, location: Tuple) -> None:
        self.neighbors = []
        self.active = 0
        self.location = location
        self.activated_neighbors = []


class BranchingNeurons():

    def __init__(self, connection_probability: float, N:int, spread_probability: float) -> None:
        self.neurons = [Neuron(tuple(np.random.random(2))) for i in range(N)]
        self.p = connection_probability
        self.init_network()
    

    def init_network(self) -> None:
        for neuron_a in self.neurons:
            for neuron_b in self.neurons:
                if np.random.random() < self.p:
                    neuron_a.neighbors.append(neuron_b)
                    neuron_b.neighbors.append(neuron_a) 

    def propage_activations(self):
        for neuron in self.neurons:
            neuron.activated_neighbors = []

        for neuron in self.neurons:
            if neuron.active:
                for neighbor in neuron.neighbors:
                    if not neighbor.active and np.random.random() < self.p:
                        neighbor.active = 1
                        neuron.activated_neighbors.append(neighbor)
    def reset(self):
        for neuron in self.neurons:
            neuron.active = 0
    
    def random_activation(self):
        for neuron in self.neurons:
            if np.random.random() < self.p:
                neuron.active = 1

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

    def run(self, steps):
        self.setup_plot()
        for i in range(steps):
            self.reset()
            self.random_activation()
            self.propage_activations()
            self.plot()
            plt.pause(0.1)
        plt.show()

if __name__ == "__main__":
    bn = BranchingNeurons(connection_probability=0.1, N=100, spread_probability=0.1)
    bn.run(100)