# This script calls all subscripts with given input

from compare_freefall_drag import compare_freefall_drag

if __name__ == "__main__":
    g = 9.81  # m/s^2
    d = 0.3  # kg/s
    m = 2  # kg
    h0 = 1000  # Initial height in m
    t_sim = 25  # s

    compare_freefall_drag(g, d, m, h0, t_sim)
