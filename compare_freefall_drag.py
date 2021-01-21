# Creates plot with comparison for exact solution of free fall with and without drag

import numpy as np
import matplotlib.pyplot as plt

# Settings for all plots
large = 32; med = 22; small = 16
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)


def compare_freefall_drag(g, d, m, h0, t_sim):
    """
    Exact solution of differential equation for free fall withput drag and with Stoke's drag
    Note: v0 = 0
    :param g: gravitational acc
    :param d: drag factor
    :param m: mass of falling item
    :param x0: initial height
    :param t_sim: simulation time for plots
    """
    t = np.linspace(0, t_sim, 1000)
    # Drag free fall exact solution
    x_drag = +g/(d/m)**2 * (1-np.e**(-d*t/m)) - g*t/(d/m) + h0
    x_free = -0.5*g*t**2 + h0

    plt.plot(t, x_free, 'r-', t, x_drag, 'g-', linewidth=4)
    plt.title("Vergleich freier Fall vs. freier Fall mit Luftreibung")
    plt.ylabel("Höhe [m]")
    plt.xlabel("t [s]")
    plt.ylim([0, h0])
    plt.xlim([0, t_sim])
    plt.legend(["Freier Fall", "Freier Fall mit Luftreibung"])
    plt.grid()
    plt.show()
