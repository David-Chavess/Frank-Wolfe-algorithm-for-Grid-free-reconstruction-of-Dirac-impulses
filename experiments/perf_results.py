import gc
from time import time

import numpy as np
from matplotlib import pyplot as plt

from src.operators.fourier_operator import FourierOperator
from src.solvers.fw import FW


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse / 2), (N, 2))
    y = y0 + w
    return y


if __name__ == '__main__':
    plot = False
    cost_plot = False

    x0 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    a0 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, -1.5, 1, 1, 1, 3, 1, 1])

    if plot:
        plt.stem(x0, a0, linefmt='k.--', markerfmt='ko', basefmt=" ", label='Ground Truth')
        plt.grid(True)
        plt.legend()
        plt.show()

    costs = {}
    lambda_grid = np.logspace(-3, -1, 25)
    for frequency in ["low", "high", "very_high"]:
        np.random.seed(1)
        N = 10 * len(x0)

        if frequency == "low":
            freq = 10
        elif frequency == "high":
            freq = 100
        elif frequency == "very_high":
            freq = 1000
        else:
            raise ValueError("Invalid frequency")

        n_particles = freq

        freq_bounds = np.array([-freq, freq])
        forward_op = FourierOperator.get_RandomFourierOperator(x0, N, freq_bounds)

        # Get measurements
        y0 = forward_op(a0)

        # add noise
        psnr = 20
        y = add_psnr(y0, psnr, N)

        # Get lambda
        lambda_max = max(abs((forward_op.adjoint(y))))
        lambda_ = 0.1 * lambda_max

        x_dim = 1

        bounds = np.array([-1, 1])
        lambdas = [0.001, 0.01, 0.02, 0.1]

        gc.collect()
        options = {"initialization": "smoothing", "polyatomic": False, "swarm": False, "sliding": True,
                   "max_iter": 100, "dual_certificate_tol": 1e-2, "smooth_sigma": 5}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        print("SFW - Time: ", time() - t1)
        solver.time_results()
        solver.flat_norm_results(x0, a0, lambdas)

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

        if cost_plot:
            costs[f"SFW_{frequency}_frequency"] = solver.get_flat_norm_values(x0, a0, lambda_grid)

        gc.collect()
        options = {"polyatomic": False, "swarm": True, "sliding": True, "swarm_c1": 0.5, "swarm_c2": 0.75,
                   "max_iter": 100, "dual_certificate_tol": 1e-2, "n_particles": n_particles}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        print("SFW_swarm - Time: ", time() - t1)
        solver.time_results()
        solver.flat_norm_results(x0, a0, lambdas)

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

        if cost_plot:
            costs[f"SFW_swarm_{frequency}_frequency"] = solver.get_flat_norm_values(x0, a0, lambda_grid)

        gc.collect()
        options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False,
                   "max_iter": 100, "dual_certificate_tol": 1e-2, "smooth_sigma": 2.5}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        print("PFW - Time: ", time() - t1)
        solver.time_results()
        solver.flat_norm_results(x0, a0, lambdas)

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

        if cost_plot:
            costs[f"PFW_{frequency}_frequency"] = solver.get_flat_norm_values(x0, a0, lambda_grid)

        gc.collect()
        options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": True,
                   "max_iter": 100, "dual_certificate_tol": 1e-2, "smooth_sigma": 2.5}
        solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                    options=options)
        t1 = time()
        solver.fit()
        print("Sliding_PFW - Time: ", time() - t1)
        solver.time_results()
        solver.flat_norm_results(x0, a0, lambdas)

        if plot:
            solver.plot(x0, a0)
            solver.plot_solution(x0, a0)

        if cost_plot:
            costs[f"Sliding_PFW_{frequency}_frequency"] = solver.get_flat_norm_values(x0, a0, lambda_grid)

    if cost_plot:
        for name, cost in costs.items():
            plt.plot(lambda_grid, cost, label=name)

        plt.xlabel("$\gamma$")
        plt.ylabel("Flat norm")
        plt.semilogx()
        plt.legend()
        plt.show()
