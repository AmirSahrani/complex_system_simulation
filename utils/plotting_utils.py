import matplotlib.pyplot as plt
import pandas as pd
from .data_utils import *


def power_law_plot(data, data_type='size'):
    """Plot a distribution on log-log axes."""
    plt.hist(data, bins=len(data), log=True)

    if data_type == 'size':
        plt.xlabel("Avalanche Size")
        plt.title('Avalanche Size Distribution')
    elif data_type == 'duration':
        plt.xlabel('Avalanche Duration')
        plt.title('Avalanche Duration Distribution')

    plt.ylabel('Frequency')
    plt.show()


def phase_transition_plot(data: pd.DataFrame):
    """Plot a phase transition."""
    plt.plot(data["Control parameter"], data["Order parameter"])
    plt.xlabel("CONROL PARAMETER") #!! Need to determine these
    plt.ylabel("ORDER PARAMETER") #EITHER AVALANCHE SIZE OR DURATION
    plt.title("Phase transition")
    plt.show()
    
def spike_density_plot(paths: list, size: int) -> None:
    """Plot the spike density."""
    plt.figure(10, 8)
    plt.title(f"Average spike density vs. branching prameters")
    plt.xlabel("Average Branching Frequency", fontsize=14)
    plt.ylabel("Average Spike Density", fontsize=14)
    for path in paths:
        data = load_data_csv(path)
        # Plot the average spike density vs. the branching parameter
        density = avg_spike_density(data, size)
        m = branching_prameter(data)
        plt.scatter(m, density)
    plt.show()
        
        
        