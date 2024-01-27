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
    plt.figure(figsize=(10, 8)) 
    plt.title("Average spike density vs. branching ratio", fontsize=16)
    plt.xlabel("Branching Ratio", fontsize=14)
    plt.ylabel("Average Spike Density", fontsize=14)
    for path in paths:  
        df = load_data_csv(path)
        # Plot the average spike density vs. the branching parameter
        density = avg_spike_density(df, size)
        print(density)
        m = branching_prameter(df)
        plt.scatter(m, density)
    plt.show()

def ref_spike_density_plot(paths: list, size: int, refractory_periods: list) -> None:
    """Plot the spike density."""
    plt.figure(figsize=(10, 8)) 
    plt.title("Average spike density vs. Branching Ratio", fontsize=16)
    plt.xlabel("Branching Ratio", fontsize=14)
    plt.ylabel("Average Spike Density", fontsize=14)
    for path, refractory_period in zip(paths, refractory_periods):  
        df = load_data_csv(path)
        # Plot the average spike density vs. the branching parameter
        density = ref_avg_spike_density(df, size,refractory_period)
        print(density)
        m = branching_prameter(df)
        plt.scatter(m, density)
    plt.show()
    
def powerlaw_size_plot(paths: list) -> None:
    """Plot the powerlaw distribution of avalanche size."""
    plt.figure(figsize=(10,8))
    plt.xlabel("s / Avalanche Size", fontsize=14)
    plt.ylabel("f(s)", fontsize=14)
    plt.title("Distribution of Avalanche Size", fontsize=16)
    for path in paths:
        df = load_data_csv(path)
        # Calculate the avalanche size distribution
        
        
    plt.show()