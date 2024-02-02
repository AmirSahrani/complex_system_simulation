import matplotlib.pyplot as plt
import pandas as pd
from .data_utils import *
import powerlaw
from typing import Optional
from branching import BranchingNeurons


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



def spike_density_plot(paths: list, size: int) -> None:
    """Plot the spike density vs. branching ratio."""
    plt.figure(figsize=(8, 6)) 
    plt.title("Not Considering Refractory Peirod", fontsize=20)
    plt.xlabel("σ(Branching Ratio)", fontsize=18)
    plt.ylabel("Average Spike Density", fontsize=18)
    plt.xlim(0, 4)
    plt.xticks(np.arange(0, 4.2, 0.2))
    plt.grid(True)
    for path in paths:  
        df = load_data_csv(path)
        # Plot the average spike density vs. the branching parameter
        density = avg_spike_density(df, size)
        m = branching_prameter(df)
        plt.scatter(m, density, color='black', s=10)
    plt.axvline(x=1, color='orange', linestyle='-', linewidth=2)
    plt.show()


def ref_spike_density_plot(paths: list, size: int, refractory_periods: list) -> None:
    """Plot the spike density vs. branching ration while taking account of refractory period."""
    plt.figure(figsize=(8, 6)) 
    plt.title("Considering Refractory Period", fontsize=20)
    plt.xlabel("σ(Branching Ratio)", fontsize=18)
    plt.ylabel("Average Spike Density", fontsize=18)
    plt.xlim(0, 4)
    plt.xticks(np.arange(0, 4.2, 0.2))
    plt.grid(True)
    for path, refractory_period in zip(paths, refractory_periods):  
        df = load_data_csv(path)
        # Plot the average spike density vs. the branching parameter
        density = ref_avg_spike_density(df, size,refractory_period)
        #print(density)
        m = branching_prameter(df)
        plt.scatter(m, density, color='black', s=10)
    plt.axvline(x=1, color='orange', linestyle='-', linewidth=2)
    plt.show()


def grid_activity_timestep(paths: list, size: int):
    """Plot spike density vs. timestep. for ordered, complex(critical), chaotic stages."""
    fig, axes = plt.subplots(len(paths), 1, sharex=True, figsize=(8, 6))
    
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df['spike_density'] = df.apply(lambda x: 0 if x['spikes_neighbours'] == 0 else x['spikes_total'] / (size ** 2), axis=1)
        
        axes[i].plot(df.index, df['spike_density'])
        if i == 0:
            axes[i].set_title('Ordered')
        elif i == 1:
            axes[i].set_title('Complex')
        elif i == 2:
            axes[i].set_title('Disordered')
        
        axes[i].set_ylabel('Population Activity', fontsize=20)
        axes[i].set_ylim(0, 0.3)
    
    plt.xlabel('Time Steps', fontsize=20)
    plt.xlim(0, 800)
    plt.tight_layout()
    plt.show()


def loglog_plotting(type: str, data: pd.DataFrame, grouped_branching: pd.DataFrame):
    fig, ax =plt.subplots(3,3, figsize=(15,15))
    ax = ax.ravel()
    for i in range(9):
        offset = 4
        all_critical_points =  data.loc[data['branching_ratio'] == grouped_branching.index[offset+i]]
        all_data = np.concatenate(all_critical_points[type].values)
        all_data = all_data[all_data > 0]

        fit = powerlaw.Fit(all_data, verbose=False, xmin=1, discrete=True)
        fitted_line =  np.linspace(0, max(all_data), 100) ** -fit.alpha 

        powerlaw.plot_pdf(all_data, ax=ax[i], color='red', label='Empirical data' , linestyle='None', marker='o', markersize=3, alpha=0.5)
        ax[i].plot(fitted_line, color='black', linestyle='--', label='Power law fit')

        ax[i].set_title(f'Branching ratio: {grouped_branching.index[offset+i]:.2f}')
        ax[i].text(0.6, 0.9, f'Alpha: {fit.alpha:.2f}\n ', transform=ax[i].transAxes)
        ax[i].set_xlabel(type.split('_')[0].capitalize() + ' ' + type.split('_')[1])
        ax[i].set_ylabel('Frequency')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlim([1, 1e3])
        ax[i].set_ylim([1e-5, 1e0])

    plt.show()


def plot_activity_per_time_step(n_steps: int, ax: Optional[plt.plot]=None, **kwargs) -> None:
    aspect_ratio = kwargs['N']/ n_steps
    if ax is None:
        fig, ax = plt.subplots(3, figsize=(30, 10))
    
    for i,branching_ratio in enumerate([0.8, 1.2, 2.0]):
        kwargs["branching_ratio"] = branching_ratio 
        sim = BranchingNeurons(**kwargs)
        sim.run(n_steps)
        data = np.array(sim.activity).T
        ax[i].imshow(data, cmap='binary', interpolation='nearest')
        ax[i].set_title(f'Branching ratio: {branching_ratio}')


def spike_activity_plot(paths: list, size: int):
    """Plot the raster spike activity."""
    num_plots = len(paths)
    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(7, num_plots * 4))
    
    if num_plots == 1:
        axs = [axs] 
    for idx, file_path in enumerate(paths):
        df = pd.read_csv(file_path)
        df = df[[str(i) for i in range(400)]]
        timesteps = []
        neuron_numbers = []
        for timestep, row in df.iterrows():
            spiked_neurons = [int(col) for col, val in row.items() if val in [1, 2]]
            
            # Store the timestep and neuron numbers
            timesteps.extend([timestep] * len(spiked_neurons))
            neuron_numbers.extend(spiked_neurons)
            
        axs[idx].scatter(timesteps, neuron_numbers, color='black', s=0.1)
        
        # Set labels and limits for better clarity
        axs[idx].set_ylabel(f"Neuron Number", fontsize = 16)
        axs[idx].set_ylim(-0.5, 400.5)  
        axs[idx].set_xlim(0, 500)
        
    # Set common labels
    plt.xlabel("Time Steps", fontsize = 16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the common title
    
    # Show the plot
    plt.show()




def raster_to_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raster data to basic data:spikes_total, spikes_neighbours, spikes_input.
    """
    params_columns = ['grid_size', 'height', 'max_distance', 'refractory_period', 
                      'probability_of_spontaneous_activity', 'random_connection']
    df = df.drop(columns=params_columns, errors='ignore')
    spikes_total = df.apply(lambda row: (row == 2).sum() + (row == 1).sum(), axis=1)
    spikes_neighbours = df.apply(lambda row: (row == 2).sum(), axis=1)
    spikes_input = df.apply(lambda row: (row == 1).sum(), axis=1)

    df_basic = pd.DataFrame({
        'spikes_total': spikes_total,
        'spikes_neighbours': spikes_neighbours,
        'spikes_input': spikes_input
    })
    return df_basic

        
def mutual_info_plot(mutual_info: list, branching_ratios: list) -> None:
    """
    Plot mutual information vs. branching ratio.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(branching_ratios, mutual_info)
    plt.xlabel("Branching Ratio", fontsize=14)
    plt.ylabel("Mutual Information", fontsize=14)
    plt.title("Mutual Information vs. Branching Ratio", fontsize=16)
    plt.show()


def dynamic_range_plot(spike_num: list, probability: list, branching_ratio: float) -> None:
    """
    Plot the dynamic range of a certain branching ratio.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(spike_num, probability)
    plt.xlabel("Response, number of neurons", fontsize=14)
    plt.ylabel("Probability (Response)", fontsize=14)
    plt.title(f"Branching Ratio = {branching_ratio}", fontsize=16)
    plt.show()


def susceptibility_plot(susceptibilities: list, branching_ratios: list) -> None:
    """
    Plot susceptibility vs. branching ratio.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(branching_ratios, susceptibilities)
    plt.xlabel("Branching Ratio", fontsize=14)
    plt.ylabel("Susceptibility", fontsize=14)
    plt.title("Susceptibility vs. Branching Ratio", fontsize=16)
    plt.show()