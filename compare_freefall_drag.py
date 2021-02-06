# Creates plot with comparison for exact solution of free fall with and without drag

import numpy as np
import matplotlib.pyplot as plt


def compare_freefall_drag(g, d, m, z0, t_sim, steps):
    """
    Exact solution of differential equation for free fall withpot drag and with Stoke's drag
    Note: v0 = 0
    :param steps: number of steps
    :param g: gravitational acc
    :param d: drag factor
    :param m: mass of falling item
    :param h0: initial height
    :param t_sim: simulation time for plots
    :return: np.array (:,2) with time and h values for free fall with drag
    """
    t = np.linspace(0, t_sim, steps)
    # Drag free fall exact solution
    h_drag = +g/(d/m)**2 * (1-np.e**(-d*t/m)) - g*t/(d/m) + z0
    h_free = -0.5*g*t**2 + z0
    plt.figure()
    plt.plot(t, h_free, 'r-', t, h_drag, 'g-', linewidth=4)
    plt.title("Vergleich freier Fall vs. freier Fall mit Luftreibung")
    plt.ylabel("Höhe [m]")
    plt.xlabel("t [s]")
    plt.ylim([0, z0])
    plt.xlim([0, t_sim])
    plt.legend(["Freier Fall", "Freier Fall mit Luftreibung"])
    plt.grid()
    return np.vstack([t, h_drag]).T
