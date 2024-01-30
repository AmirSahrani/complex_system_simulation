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
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 5.2, 0.2))
    plt.grid(True)
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
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 5.2, 0.2))
    plt.grid(True)
    for path, refractory_period in zip(paths, refractory_periods):  
        df = load_data_csv(path)
        # Plot the average spike density vs. the branching parameter
        density = ref_avg_spike_density(df, size,refractory_period)
        print(density)
        m = branching_prameter(df)
        plt.scatter(m, density)
    plt.show()
    
def powerlaw_avalanche_plots(paths: list, method: list, thresh_m: float) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for path in paths:
        try:
            df = load_data_csv(path)
            sizes, durations = avalanche_distributions(df)
            sizes = [size for size in sizes if size > 0]
            durations = [duration for duration in durations if duration > 0]
        
            if sizes and durations:
                if method == 'fit':
                    if abs(branching_prameter(df) - 1) < thresh_m:
                        fit_sizes = powerlaw.Fit(sizes)
                        fit_sizes.power_law.plot_pdf(ax=ax1, color=np.random.rand(3,), linestyle='-')
                        fit_durations = powerlaw.Fit(durations)
                        fit_durations.power_law.plot_pdf(ax=ax2, color=np.random.rand(3,), linestyle='-')
                elif method == 'plot':
                    if abs(branching_prameter(df) - 1) < thresh_m:
                        powerlaw.plot_pdf(sizes, ax=ax1, color=np.random.rand(3,), linestyle='--')
                        powerlaw.plot_pdf(durations, ax=ax2, color=np.random.rand(3,), linestyle='--')
                elif method == 'scatter':
                    ax1.scatter(range(len(sizes)), sizes, color=np.random.rand(3,), linestyle='--')
                    ax2.scatter(range(len(durations)), durations, color=np.random.rand(3,), linestyle='--')
                elif method == 'histogram':
                    ax1.hist(sizes, bins=len(sizes), log=True, color=np.random.rand(3,), linestyle='--')
                    ax2.hist(durations, bins=len(durations), log=True, color=np.random.rand(3,), linestyle='--')
            else:
                print(f"{path} has no enough data")
        except Exception as e:
            print(f"Error processing file {path}: {e}")

    
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
        fig, ax = plt.subplots(3, figsize=(10, 30))
    
    for i,branching_ratio in enumerate([0.8, 1.2, 2.0]):
        kwargs["branching_ratio"] = branching_ratio 
        sim = BranchingNeurons(**kwargs)
        sim.run(n_steps)
        data = np.array(sim.activity).T
        ax[i].imshow(data, cmap='binary', interpolation='nearest')
        ax[i].set_title(f'Branching ratio: {branching_ratio}')



        
