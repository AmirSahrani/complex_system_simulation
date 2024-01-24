import matplotlib.pyplot as plt
import pandas as pd


def power_law_plot(data):
    """Plot a distribution on log-log axes."""
    plt.hist(data, bins=len(data), log=True)
    plt.xlabel("Avalanche duration")
    plt.ylabel("Frequency")
    plt.title("Avalanche duration distribution")
    plt.show()


def phase_transition_plot(data: pd.DataFrame):
    """Plot a phase transition."""
    plt.plot(data["Control parameter"], data["Order parameter"])
    plt.xlabel("CONROL PARAMETER") #!! Need to determine these
    plt.ylabel("ORDER PARAMETER") #EITHER AVALANCHE SIZE OR DURATION
    plt.title("Phase transition")
    plt.show()