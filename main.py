# This script calls all subscripts with given input

from compare_freefall_drag import compare_freefall_drag
from odeint_test import odeint_test

import matplotlib.pyplot as plt
import numpy as np


# Settings for all plots
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (8, 6),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)

if __name__ == "__main__":
    g = 9.81  # m/s^2
    d = 0.3  # kg/s
    m = 2  # kg
    z0 = 1000  # Initial height in m
    t_sim = 24.99  # s
    steps = 10000  # In whole t0-t_sim time

    step_size = t_sim/steps

    # ODE 2nd Order to solve: zdotdot = -d/m * zdot - g

    # --------- Exact Solutions ----------

    # This will call a graph for comparing free fall with and without drag and their exact solution
    z_t_ref = compare_freefall_drag(g, d, m, z0, t_sim, steps)
    ts = z_t_ref[:, 0]

    # ---------- SciPy Solution ----------

    # ODEINT, which is based on fortran (older api)
    #h_odeint = odeint_test(ts, g, d, m, h0)

    # Or using the new solve_ivp, where we could choose our ODE solver
    def f(t, Y):
        return [Y[1], -d/m * Y[1] - g]

    def hit_ground(t, Y):
        return Y[0]
    hit_ground.terminal = True

    from scipy.integrate import solve_ivp
    res = solve_ivp(fun=f, t_span=[0, t_sim], y0=[z0, 0], t_eval=ts, method='RK45', events=hit_ground)
    h_RK45 = res.y[0, :]
    t_RK45 = res.t

    plt.figure()
    plt.plot(z_t_ref[:, 0], z_t_ref[:, 1], 'r-', t_RK45, h_RK45, 'g--', linewidth=2)
    plt.title("RK45")
    plt.ylabel("Höhe[m]")
    plt.xlabel("t[s]")
    #plt.ylim([0, z0])
    plt.xlim([0, t_sim])
    plt.legend(["Reference", "solve_ivp()"])
    plt.grid()

    # ---------- Explicit Euler Method (non vectorized) -----------

    dts = [4, 2, 1]
    z_n_all_res = []
    for dt in dts:
        t_acc = 0
        z_n = z0
        v_n = 0
        z_n_res = [z_n]
        while t_acc < t_sim:
            # First eq
            f_1 = v_n
            z_n = z_n + dt * f_1
            # Second eq
            f_2 = (-d/m * v_n - g)  # Basically f_zdot
            v_n = v_n + dt * f_2

            # Saving values in list
            z_n_res.append(z_n)

            # Add sim time
            t_acc += dt
        z_n_all_res.append(z_n_res)

    # Plot and compare results of Euler Method
    plt.figure()
    plt.plot(z_t_ref[:, 0], z_t_ref[:, 1], 'r-', linewidth=2, label="Reference")
    for count, dt in enumerate(dts):
        plt.plot(np.arange(0, t_sim+dt, dt), z_n_all_res[count], '--', linewidth=2, label="Euler h={}s".format(str(dt)))
    plt.title("Euler")
    plt.ylabel("Höhe[m]")
    plt.xlabel("t[s]")
    plt.ylim([0, z0])
    plt.xlim([0, t_sim])
    plt.legend()
    plt.grid()

    # --------- Explicit Euler Vectorized ----------
    t_acc = 0
    z_n = z0
    v_n = 0
    Y_n = np.array([z_n, v_n])  # See state vector or scipy solution!
    dt = 2
    z_n_euler_res = [Y_n[0]]
    #t_sim = 100

    def f(t, Y):
        return np.array([Y[1], -d/m * Y[1] - g])

    def euler(fun, t_n, h, Y_n):
        return Y_n + h * fun(t_n, Y_n)

    while t_acc < t_sim:
        # Calculate next time step
        Y_n = euler(f, t_acc, dt, Y_n)
        # Save result (height z)
        z_n_euler_res.append(Y_n[0])
        # Add sim time
        t_acc += dt

    # Plot and compare results of Euler Method
    plt.figure()
    plt.plot(z_t_ref[:, 0], z_t_ref[:, 1], 'r-', linewidth=2, label="Reference")
    plt.plot(np.arange(0, t_sim+dt, dt), z_n_euler_res, 'b-', linewidth=2, label="Euler Vectorized")
    plt.title("Euler Vectorized Check")
    plt.ylabel("Höhe[m]")
    plt.xlabel("t[s]")
    plt.ylim([0, z0])
    plt.xlim([0, t_sim])
    plt.legend()
    plt.grid()

    # --------- Runge-Kutta-Method 4 Vectorized ----------

    t_acc = 0
    z_n = z0
    v_n = 0
    Y_n = np.array([z_n, v_n])  # See state vector or scipy solution!
    dt = 2
    z_n_RK4_res = [Y_n[0]]

    def f(t, Y):
        return np.array([Y[1], -d/m * Y[1] - g])

    def RK4(fun, t_n, h, Y_n):
        k1 = fun(t_n, Y_n)
        k2 = fun(t_n+h/2, Y_n+h*k1/2)
        k3 = fun(t_n+h/2, Y_n+h*k2/2)
        k4 = fun(t_n+h, Y_n+h*k3)
        Y_nplus1 = Y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return Y_nplus1

    while t_acc < t_sim:
        # Calculate next time step
        Y_n = RK4(f, t_acc, dt, Y_n)
        # Save result (height z)
        z_n_RK4_res.append(Y_n[0])
        # Add sim time
        t_acc += dt

    # Other formula alternative
    # def RK4(fun, t_n, h, Y_n):
    #     k1 = h * fun(t_n, Y_n)
    #     k2 = h * fun(t_n+h/2, Y_n+k1/2)
    #     k3 = h * fun(t_n+h/2, Y_n+k2/2)
    #     k4 = h * fun(t_n+h, Y_n+k3)
    #     Y_nplus1 = Y_n + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    #     return Y_nplus1

    # Plot and compare results of Euler Method
    plt.figure()
    plt.plot(z_t_ref[:, 0], z_t_ref[:, 1], 'r-', linewidth=2, label="Reference")
    plt.plot(np.arange(0, t_sim+dt, dt), z_n_RK4_res, 'b--*', linewidth=1, label="RK4")
    plt.plot(np.arange(0, t_sim+dt, dt), z_n_euler_res, 'g--*', linewidth=1, label="Explicit Euler")
    plt.title("RK4-Euler Vergleich h={}s".format(dt))
    plt.ylabel("Höhe[m]")
    plt.xlabel("t[s]")
    plt.ylim([0, z0])
    plt.xlim([0, t_sim])
    plt.legend()
    plt.grid()

    plt.show()

    # # Test other function to proof RK4
    # def f(t,y):
    #     return t * np.sqrt(y)
    #
    # t_acc = 0
    # Y_n = 1
    # Y_e = 1
    # t_sim = 0.99
    # dt = 0.1
    # while t_acc < t_sim:
    #     # Calculate next time step
    #     Y_n = RK4(f, t_acc, dt, Y_n)
    #     Y_e = euler(f, t_acc, dt, Y_e)
    #     t_acc += dt
    #
    # print(t_acc, Y_n, Y_e)
