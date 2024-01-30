import numpy as np
import matplotlib.pyplot as plt
from typing import Counter, List, Optional, Tuple
from tqdm import tqdm

if __name__ != "__main__":
    def tqdm(x):
        return x

class Neuron():
    """
    Represents a neuron in a complex system simulation.

    Attributes:
        location (Tuple): The location of the neuron.
        branching_ratio (float): The branching ratio used to generate the probabilities of activation for each neighboring neuron.
        neighbors (list): List of neighboring neuronns.
        origins (list): List of neurons that activated this neuron.
        active (int): Indicates whether the neuron is active or not.
        activated_neighbors (list): List of activated neighboring neurons.
        cooldown (int): Cooldown period for the neuron.
        avalanche (Avalanche_graph): The avalanche graph associated with the neuron, representing the leaves of the avalanche.
        probabilities (dict): Probabilities of activation for each neighboring neuron.
    """

    def __init__(self, location: Tuple, branching_ratio: float) -> None:
        self.branching_ratio = branching_ratio
        self.location = location

        self.neighbors = []
        self.origins = []
        self.active = 0
        self.activated_neighbors = []
        self.cooldown = 0
        self.avalanche_id = None
        self.probabilities = {}


    def generate_probabilities(self):
        assert len(self.neighbors) > 0, "Neuron has no neighbors."

        n_probs = len(self.neighbors)
        rand_vec = np.random.random(n_probs)
        rand_vec /= np.sum(rand_vec)
        rand_vec *= self.branching_ratio

        for neighbor, prob in zip(self.neighbors, rand_vec):
            self.probabilities[neighbor] = prob

        assert np.isclose(np.sum(list(self.probabilities.values())), self.branching_ratio), "Probabilities do not sum to the branching ratio."


    def __repr__(self) -> str:
        return f""""Neuron at {self.location} 
        Active: {self.active}
        Cooldown: {self.cooldown}
        probablities: {self.probabilities}
        number of neighbors: {len(self.neighbors)}
        number of activated neighbors: {len(self.activated_neighbors)}\n"""


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

    def __init__(self, N:int, max_neighbors: int, branching_ratio: float, visual: bool=False) -> None:

        self.neurons = [Neuron(tuple(np.random.random(2)), 
                               branching_ratio) for i in range(N)]
        self.visual = visual
        assert max_neighbors < N, "Max neighbors must be less than the number of neurons."
        self.max_neighbors = max_neighbors
        self.branching_ratio = branching_ratio

        self.evalanche_size = []
        self.evalanche_duration = []
        self.active = []
        self.next_active = []  
        self.branching = []
        self.density = []
        self.tracked_durations = {}
        self.tracked_sizes = {}
        self.activity = []
        self.active_from_random = 0
        self.neuron_ids = {neuron: i for i, neuron in enumerate(self.neurons)}


        self.init_network()
        if visual:
            self.setup_plot()
    

    def init_network(self) -> None:
        copy_neurons = self.neurons.copy()

        if __name__ == "__main__":
            print(f'Initializing network with {len(self.neurons)} neurons.')

        for neuron in tqdm(list(self.neurons)):
            if neuron in copy_neurons:
                copy_neurons.remove(neuron)

                while len(neuron.neighbors) < self.max_neighbors and copy_neurons:
                    needed_neighbors = self.max_neighbors - len(neuron.neighbors)

                    if all(n in neuron.neighbors for n in copy_neurons):
                        break

                    neighbors = np.random.choice(copy_neurons, needed_neighbors)

                    for neighbor in neighbors:
                        if neighbor not in neuron.neighbors:
                            neuron.neighbors.append(neighbor)
                            neighbor.neighbors.append(neuron)

                            if len(neighbor.neighbors) == self.max_neighbors:
                                copy_neurons.remove(neighbor)
            
            neuron.generate_probabilities()


    def propage_activations(self, neuron: Neuron):
        assert neuron.active, "Neuron is not active."

        neuron.activated_neighbors = []
        if not neuron.avalanche_id:
            neuron.avalanche_id = np.random.rand() + 1e-8
            self.tracked_durations[neuron.avalanche_id] = 0
            self.tracked_sizes[neuron.avalanche_id] = 0

        moving_avalance_id = neuron.avalanche_id
        changed_avalanche = False

        for neighbor in neuron.neighbors:
            if not neighbor.active and np.random.random() < neuron.probabilities[neighbor]:
                if not neighbor.cooldown:

                    neighbor.avalanche_id = moving_avalance_id
                    self.tracked_sizes[moving_avalance_id] += 1



                    if not changed_avalanche:
                        self.tracked_durations[moving_avalance_id] += 1
                        neuron.avalanche_id = None
                        changed_avalanche = True

                    neighbor.active = 1
                    self.next_active.append(neighbor)
                    neuron.activated_neighbors.append(neighbor)

        if all(moving_avalance_id != n.avalanche_id for n in self.neurons) and self.tracked_durations[moving_avalance_id] > 1:
            assert moving_avalance_id == neuron.avalanche_id, "Avalanche graph not updated correctly."
            self.evalanche_duration.append(self.tracked_durations[moving_avalance_id])
            self.evalanche_size.append(self.tracked_sizes[moving_avalance_id])
            self.tracked_durations.pop(moving_avalance_id)
            self.tracked_sizes.pop(moving_avalance_id)
            neuron.avalanche_id = None


    def reset(self):
        for neuron in self.neurons:
            neuron.active = 0
            neuron.activated_neighbors = []
        self.evalanche_size = []
        self.evalanche_duration = []
        self.active = []
        self.next_active = []  
        self.branching = []
        self.density = []
        self.tracked_durations = {}
        self.tracked_sizes = {}
        self.activity = []
        self.active_from_random = 0
    
    
    def random_activation(self):
        self.active_from_random = 0
        for neuron in self.neurons:
            if np.random.random() < 1e-5 and not neuron.active: 
                neuron.active = 1
                self.active.append(neuron)
                self.active_from_random += 1

    
    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
    

    def plot(self):
        self.ax.clear()
        for neuron in self.neurons:
            if neuron.active:
                if self.tracked_sizes[neuron.avalanche_id] > 1:
                    self.ax.scatter(neuron.location[0], neuron.location[1], s= self.tracked_sizes[neuron.avalanche_id], c="blue")
                else:
                    self.ax.scatter(neuron.location[0], neuron.location[1], c="red")
                for neighbor in neuron.activated_neighbors:
                    self.ax.plot([neuron.location[0], neighbor.location[0]], [neuron.location[1], neighbor.location[1]], c="blue", linewidth=0.7)
            else:
                self.ax.scatter(neuron.location[0], neuron.location[1], c="gray")
                for neighbor in neuron.neighbors:
                    self.ax.plot([neuron.location[0], neighbor.location[0]], [neuron.location[1], neighbor.location[1]], c="gray", alpha=0.2, linewidth=0.2)


    def run(self, steps: int):
        for i in tqdm(range(steps)):
            warmup = i > 0.1*steps
            self.random_activation()

            for neuron in self.active:

                self.propage_activations(neuron)
                neuron.active = 0
                neuron.cooldown = 5

            for neuron in self.neurons:
                if neuron.cooldown:
                    neuron.cooldown -= 1

            if self.visual and (self.active or self.next_active):
                self.plot()
                plt.pause(0.001)

            if  warmup and self.active:
                self.branching.append(len(self.next_active)/len(self.active))

            if not warmup:
                self.density.append(len(self.active)/len(self.neurons))
            

            self.active = self.next_active.copy()
            self.next_active = []
            self.activity.append([neuron.active for neuron in self.neurons])

        self.evalanche_duration.extend(self.tracked_durations.values())
        self.evalanche_size.extend(self.tracked_sizes.values())




if __name__ == "__main__":

    kwargs = {
        'N': 1000,
        'max_neighbors': 8,
        'branching_ratio': 3,
        'visual': False,
    }
    sim = BranchingNeurons(**kwargs)
    sim.run(10000)
    sim.activity_plot()
    print(f'Max avalance size: {max(sim.evalanche_size)}\nMax avalance duration: {max(sim.evalanche_duration)} \nMean density: {np.mean(sim.density)}')
    print(f'Mean branching ratio: {np.mean(sim.branching)} ±{np.std(sim.branching)}, with {len(sim.branching)} samples.')