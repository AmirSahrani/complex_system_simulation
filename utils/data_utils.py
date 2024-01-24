import matplotlib.pyplot as plt


def power_law_plot(data):
    """Plot a power law distribution."""
    plt.hist(data, bins=len(data), log=True)
    plt.xlabel("Avalanche duration")
    plt.ylabel("Frequency")
    plt.title("Avalanche duration distribution")
    plt.show()