import gc
from time import time

import numpy as np
from matplotlib import pyplot as plt

from src.operators.convolution_operator import ConvolutionOperator
from src.solvers.fw import FW


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse), N)
    y = y0 + w
    return y


if __name__ == '__main__':
    plot = False

    n_spikes = np.arange(5, 95, 10)

    duration = {"SFW": [], "SFW_PSO": [], "PFW": [], "Sliding_PFW": []}
    candidates_duration = {"SFW": [], "SFW_PSO": [], "PFW": [], "Sliding_PFW": []}
    corrections_duration = {"SFW": [], "SFW_PSO": [], "PFW": [], "Sliding_PFW": []}
    sliding_duration = {"SFW": [], "SFW_PSO": [], "Sliding_PFW": []}
    for n in n_spikes:
        np.random.seed(1)

        x0 = np.random.uniform(0.05, 0.55, size=(n, 2))
        a0 = np.random.uniform(1, 3, n)

        x_dim = 2
        bounds = np.array([0, 1])

        fwhm = 0.1
        n_measurements_per_pixel = 3
        forward_op = ConvolutionOperator(x0, fwhm, bounds, x_dim, n_measurements_per_pixel)
        N = forward_op.n_measurements
        n_particles = 100

        # Get measurements
        y0 = forward_op(a0)

        # add noise
        psnr = 20
        y = add_psnr(y0, psnr, N)

        # Get lambda
        lambda_max = max(abs((forward_op.adjoint(y))))
        lambda_ = 0.1 * lambda_max

        x_dim = 2
        bounds = np.array([0, 1])

        gc.collect()
        options = {"initialization": "smoothing", "polyatomic": False, "swarm": False, "sliding": True, "positive_constraint": True,
               "max_iter": 200, "dual_certificate_tol": 1e-1, "smooth_sigma": 1, "n_particles": n_particles}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        duration["SFW"].append(time() - t1)
        print("SFW - Time: ", time() - t1)
        solver.time_results()
        mst = solver._mstate
        candidates_duration["SFW"].append(sum(mst['candidates_search_durations']))
        corrections_duration["SFW"].append(sum(mst['correction_durations']))
        sliding_duration["SFW"].append(sum(mst['sliding_durations']))

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

        gc.collect()
        options = {"polyatomic": False, "swarm": True, "sliding": True, "swarm_c1": 0.5, "swarm_c2": 0.75, "positive_constraint": True,
                   "max_iter": 200, "dual_certificate_tol": 1e-1, "n_particles": n_particles}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        duration["SFW_PSO"].append(time() - t1)
        print("SFW_swarm - Time: ", time() - t1)
        solver.time_results()
        mst = solver._mstate
        candidates_duration["SFW_PSO"].append(sum(mst['candidates_search_durations']))
        corrections_duration["SFW_PSO"].append(sum(mst['correction_durations']))
        sliding_duration["SFW_PSO"].append(sum(mst['sliding_durations']))

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

        gc.collect()
        options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False, "positive_constraint": True,
               "max_iter": 20, "dual_certificate_tol": 1e-1, "smooth_sigma": 1}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        duration["PFW"].append(time() - t1)
        print("PFW - Time: ", time() - t1)
        solver.time_results()
        mst = solver._mstate
        candidates_duration["PFW"].append(sum(mst['candidates_search_durations']))
        corrections_duration["PFW"].append(sum(mst['correction_durations']))

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

        gc.collect()
        options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": True, "positive_constraint": True,
               "max_iter": 100, "dual_certificate_tol": 1e-1, "smooth_sigma": 1}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        duration["Sliding_PFW"].append(time() - t1)
        print("Sliding_PFW - Time: ", time() - t1)
        solver.time_results()
        mst = solver._mstate
        candidates_duration["Sliding_PFW"].append(sum(mst['candidates_search_durations']))
        corrections_duration["Sliding_PFW"].append(sum(mst['correction_durations']))
        sliding_duration["Sliding_PFW"].append(sum(mst['sliding_durations']))

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

    plt.plot(n_spikes, duration["SFW"], label="SFW", marker='o')
    plt.plot(n_spikes, duration["SFW_PSO"], label="SFW_PSO", marker='o')
    plt.plot(n_spikes, duration["PFW"], label="PFW", marker='o')
    plt.plot(n_spikes, duration["Sliding_PFW"], label="Sliding_PFW", marker='o')
    plt.xlabel("Number of Spikes")
    plt.ylabel("Time (s)")
    plt.semilogy()
    plt.legend()
    plt.show()

    plt.plot(n_spikes, duration["SFW"], label="SFW", marker='o')
    plt.plot(n_spikes, duration["SFW_PSO"], label="SFW_PSO", marker='o')
    plt.plot(n_spikes, duration["PFW"], label="PFW", marker='o')
    plt.plot(n_spikes, duration["Sliding_PFW"], label="Sliding_PFW", marker='o')
    plt.xlabel("Number of Spikes")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

    plt.plot(n_spikes, candidates_duration["PFW"], label="PFW - Candidates Search Step", marker='o')
    plt.plot(n_spikes, corrections_duration["PFW"], label="PFW - Correction step", marker='o')
    plt.plot(n_spikes, candidates_duration["SFW"], label="SFW - Candidates Search Step", marker='o')
    plt.plot(n_spikes, corrections_duration["SFW"], label="SFW - Correction step", marker='o')
    plt.plot(n_spikes, sliding_duration["SFW"], label="SFW - Sliding Step", marker='o')
    plt.xlabel("Number of Spikes")
    plt.ylabel("Time (s)")
    plt.semilogy()
    plt.legend()
    plt.show()

    plt.plot(n_spikes, candidates_duration["PFW"], label="PFW - Candidates Search Step", marker='o')
    plt.plot(n_spikes, corrections_duration["PFW"], label="PFW - Correction step", marker='o')
    plt.plot(n_spikes, candidates_duration["SFW"], label="SFW - Candidates Search Step", marker='o')
    plt.plot(n_spikes, corrections_duration["SFW"], label="SFW - Correction step", marker='o')
    plt.plot(n_spikes, sliding_duration["SFW"], label="SFW - Sliding Step", marker='o')
    plt.xlabel("Number of Spikes")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()