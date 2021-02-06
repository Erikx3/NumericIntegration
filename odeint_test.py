from scipy.integrate import odeint


def odeint_test(ts, g, d, m, h0):

    # Careful: This is an older API with different order
    def f(u, t):
        return [u[1], -d / m * u[1] - g]

    us = odeint(f, [h0, 0], ts)  # Function, h0 and zdot0, time array
    ys = us[:, 0]
    return ys
