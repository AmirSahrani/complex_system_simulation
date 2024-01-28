import matplotlib.pyplot as plt
import pandas as pd
from .data_utils import *
import powerlaw


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
    
def powerlaw_avalanche_plots(paths: list) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for path in paths:
        df = load_data_csv(path)
        sizes, durations = avalanche_distributions(df)
        sizes = [size for size in sizes if size > 0]
        durations = [duration for duration in durations if duration > 0]

        print(f"Sizes for path {path}: {sizes}")  # Debugging line
        print(f"Durations for path {path}: {durations}")  # Debugging line

        fit_sizes = powerlaw.Fit(sizes)
        fit_sizes.power_law.plot_pdf(ax=ax1, color=np.random.rand(3,), linestyle='-', label=f'Fit size: {path}')
        powerlaw.plot_pdf(sizes, ax=ax1, color=np.random.rand(3,), linestyle='--', label=f'Data size: {path}')
        
        fit_durations = powerlaw.Fit(durations)
        fit_durations.power_law.plot_pdf(ax=ax2, color=np.random.rand(3,), linestyle='-', label=f'Fit duration: {path}')
        powerlaw.plot_pdf(durations, ax=ax2, color=np.random.rand(3,), linestyle='--', label=f'Data duration: {path}')
        

    
    ax1.set_xlabel("Size (s)", fontsize=14)
    ax1.set_ylabel("PDF", fontsize=14)
    ax1.set_title("Avalanche Size Distribution", fontsize=16)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    
    ax2.set_xlabel("Duration", fontsize=14)
    ax2.set_ylabel("PDF", fontsize=14)
    ax2.set_title("Avalanche Duration Distribution", fontsize=16)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
