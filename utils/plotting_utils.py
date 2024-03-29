import matplotlib.pyplot as plt
import pandas as pd
from .data_utils import *
import powerlaw
from typing import Optional
from branching import BranchingNeurons


def power_law_plot(data, data_type='size'):
    """Plot a distribution on log-log axes."""
    plt.figure(figsize=(8, 6))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
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
    plt.figure(figsize=(8, 6))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    plt.plot(data["Control parameter"], data["Order parameter"])
    plt.xlabel("CONROL PARAMETER") #!! Need to determine these
    plt.ylabel("ORDER PARAMETER") #EITHER AVALANCHE SIZE OR DURATION
    plt.title("Phase transition")
    plt.show()


def spike_density_plot(paths: list, size: int) -> None:
    """Plot the spike density vs. branching ratio."""
    plt.figure(figsize=(8, 6))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    plt.title("Not Considering Refractory Peirod", fontsize=20)
    plt.xlabel("σ(Branching Ratio)", fontsize=18)
    plt.ylabel("Average Spike Density", fontsize=18)
    plt.xlim(0, 4)
    plt.xticks(np.arange(0, 4.2, 0.2))
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
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    plt.title("Considering Refractory Period", fontsize=20)
    plt.xlabel("σ(Branching Ratio)", fontsize=18)
    plt.ylabel("Average Spike Density", fontsize=18)
    plt.xlim(0, 4)
    plt.xticks(np.arange(0, 4.2, 0.2))
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
    fig, axes = plt.subplots(len(paths), 1, sharex=True, figsize=(8, 10))
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
        
        axes[i].set_ylabel('Population Activity', fontsize=15)
        axes[i].set_ylim(0, 0.3)
    
    plt.xlabel('Time Steps', fontsize=20)
    plt.xlim(0, 800)
    plt.tight_layout()
    plt.show()




def loglog_plotting(type: str, data: pd.DataFrame, grouped_branching: pd.DataFrame):
    """Plot the log log probability distribution of the data."""
    fig, ax =plt.subplots(2,3, figsize=(12,8))
    ax = ax.ravel()
    for i in range(6):
        offset = 5

        all_critical_points =  data.loc[data['branching_ratio'] == grouped_branching.index[offset+i]]
        all_data = np.concatenate(all_critical_points[type].values)
        all_data = all_data[all_data > 0]

        fit = powerlaw.Fit(all_data, verbose=False)
        lognormal = powerlaw.Lognormal(verbose=False)
        lognormal.fit(all_data)
        mu, sigma = lognormal.mu, lognormal.sigma

        log_llkhood, p_value = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)

        x_values = np.linspace(min(all_data), max(all_data), len(all_data))
        fitted_line = (x_values ** -fit.alpha)
        log_fitted = lognormal.pdf(x_values)
        

        powerlaw.plot_pdf(all_data, ax=ax[i], color='red', label='Empirical data' , linestyle='None', marker='o', markersize=3, alpha=0.5)
        ax[i].plot(x_values, fitted_line, color='black', linestyle='--', label='Power law fit')
        ax[i].plot(x_values, log_fitted, color='blue', linestyle='--', label='Log Normal fit')

        ax[i].set_title(f'Branching ratio: {grouped_branching.index[offset+i]:.2f}')
        ax[i].text(0.1, 0.1, 
           f'$\\alpha$: {fit.alpha:.2f}\n$\\mu$: {mu:.2f}\n$\\sigma$: {sigma:.2f}\n$p$: {p_value:.3f}\nLog Likelihood: {log_llkhood:.3f}', 
           transform=ax[i].transAxes)

        
        if i == 2:
            ax[i].legend(loc='upper right') 
    
    fig.supylabel('Frequency')
    fig.supxlabel(type.split('_')[0].capitalize() + ' ' + type.split('_')[1])
    plt.tight_layout()
    plt.show()

def plot_activity_per_time_step(n_steps: int, ax: Optional[plt.plot]=None, **kwargs) -> None:
    """Plot activity per time step for branching model's different branching ratio."""
    if ax is None:
        fig, ax = plt.subplots(3, figsize=(15, 6))
    
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    for i,branching_ratio in enumerate([0.8, 1.1, 2.0]):
        kwargs["branching_ratio"] = branching_ratio 
        sim = BranchingNeurons(**kwargs)
        sim.run(n_steps)
        data = np.array(sim.activity).T
        ax[i].imshow(data, cmap='binary', interpolation='nearest')
        ax[i].set_title(f'Branching ratio: {branching_ratio}', fontsize=16)
    fig.supxlabel('Time step', fontsize=14)
    fig.supylabel('Neuron', fontsize=14)
    plt.tight_layout()
    plt.show()


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
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    # Show the plot
    plt.show()





        
def mutual_info_plot(mutual_info: list, branching_ratios: list) -> None:
    """
    Plot mutual information vs. branching ratio.
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    plt.plot(branching_ratios, mutual_info)
    plt.xlabel("Branching Ratio", fontsize=14)
    plt.ylabel("Mutual Information", fontsize=14)
    plt.title("Mutual Information vs. Branching Ratio", fontsize=16)
    plt.show()


def dynamic_range_plot(dynamic_ranges: list, branching_ratios: list, ax: plt.plot = None) -> None:
    """
    Show the dynamic range of a certain branching ratio
    """
    width = []
    for i in range(len(branching_ratios)):
        probs = np.cumsum(dynamic_ranges[i][1])
        start = np.argmax(probs > 0.1)
        end = np.argmax(probs > 0.9)
        width.append(len(dynamic_ranges[i][1][start:end]))

    std_err = np.std(width) / np.sqrt(len(width))
    plt.figure(figsize=(12, 8))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    plt.errorbar(branching_ratios, width, yerr=std_err, fmt='--', markersize=3, alpha=0.8)
    plt.xlabel("Branching Ratio", fontsize=14)
    plt.ylabel("Dynamic Range (width)", fontsize=14)
    plt.title('Dynamic Ranges vs Branching Ratios', fontsize=16)
    plt.show()


def susceptibility_plot(susceptibilities: list, errors: list, branching_ratios: list) -> None:
    """
    Plot susceptibility vs. branching ratio.
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('tableau-colorblind10')
    plt.errorbar(branching_ratios, susceptibilities, yerr=errors, fmt='--', capsize=5) 
    plt.xlabel("Branching Ratio", fontsize=14)
    plt.ylabel("Susceptibility", fontsize=14)
    plt.title("Susceptibility vs. Branching Ratio", fontsize=16)
    plt.grid(True)
    plt.show()


def loglog_plotting_size_duration(type: List[str], data: pd.DataFrame, grouped_branching: pd.DataFrame):
    """
    Plot loglog plot of size vs duration for different branching ratios.
    """
    fig, ax =plt.subplots(2,3, figsize=(12,8))
    ax = ax.ravel()
    for i in range(6):
        offset = 4

        all_data=  data.loc[data['branching_ratio'] == grouped_branching.index[offset+i]]

        all_size = np.concatenate(all_data[type[0]].values)
        all_size = all_size[all_size > 0]

        all_duration = np.concatenate(all_data[type[1]].values)
        all_duration = all_duration[all_duration > 0]

        ax[i].loglog(all_size, all_duration, linestyle='None', marker='o', markersize=3, alpha=0.5)
        ax[i].set_title(f'Branching ratio: {grouped_branching.index[offset+i]:.2f}')
        ax[i].grid(True)
    
    fig.supxlabel('Avalanche Size')
    fig.supylabel('Avalanche Duration')
    fig.suptitle(f'Loglog plot of {type[0]} vs {type[1]}', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_mean_density_vs_branching_ratio(grouped_branching, critical_point):
    """
    Plot mean density vs branching ratio.
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    plt.plot(grouped_branching.index, grouped_branching['mean_density'])
    plt.scatter(grouped_branching.index, grouped_branching['mean_density'])

    plt.scatter(grouped_branching.index.values[critical_point], grouped_branching['mean_density'].values[critical_point], c='r')
    plt.text(grouped_branching.index.values[critical_point] + 0.2, grouped_branching['mean_density'].values[critical_point], 'Critical Point')
    plt.xlabel('Branching Ratio')
    plt.xlim(0.5, 3)
    plt.ylabel('Mean Density')
    plt.title('Mean Density vs Branching Ratio')
    plt.show()


def plot_phase_transition_cooldowns(grouped_cooldowns):
    """
    Plot phase transition for different refractory periods.
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    for i in [0,1,3,5]:
        plt.plot(grouped_cooldowns.loc[i].index, grouped_cooldowns.loc[i]['density']['mean'], label=f'Refractory period: {i}')
        plt.fill_between(grouped_cooldowns.loc[i].index, grouped_cooldowns.loc[i]['density']['mean'] - grouped_cooldowns.loc[i]['density']['std'], grouped_cooldowns.loc[i]['density']['mean'] + grouped_cooldowns.loc[i]['density']['std'], alpha=0.2)
    plt.xlabel('Branching ratio')
    plt.ylabel('Density')
    plt.title('Density vs Branching Ratio for different Refractory Periods')
    plt.xlim(0.5, 5)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    
def plot_emperical_branching_ratio(grouped_branching):
    """
    Plot emperical branching ratio vs branching ratio.
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('tableau-colorblind10')
    plt.grid(True)
    plt.scatter(grouped_branching.index, grouped_branching['emperical_branching_ratio'], label='Branching ratio')
    plt.plot(np.linspace(np.min(grouped_branching.index), np.max(grouped_branching.index), 100), np.linspace(np.min(grouped_branching.index), np.max(grouped_branching.index), 100), label='1:1 line', color='black', linestyle='--' , alpha=0.5)
    plt.xlabel('Branching ratio')
    plt.ylabel('Emperical branching ratio')
    plt.title('Emperical branching ratio vs Branching ratio')
    plt.legend()
    plt.show()