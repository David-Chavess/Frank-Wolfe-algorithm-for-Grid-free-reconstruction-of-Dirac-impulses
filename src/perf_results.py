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
    np.random.seed(1)

    plot = False

    x1 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    a1 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, -1.5, 1, 1, 1, 3, 1, 1])

    if plot:
        plt.stem(x1, a1, linefmt='k.--', markerfmt='ko', basefmt=" ", label='Ground Truth')
        plt.grid(True)
        plt.legend()
        plt.show()

    size = 50
    x2 = np.random.uniform(-0.95, 0.95, 50)
    a2 = np.random.uniform(1, 4, 50) * np.random.choice([-1, 1], 50)

    costs = {}
    # lambda_grid = np.logspace(-3, 0, 25)
    lambda_grid = np.linspace(0.001, 1, 25)
    # for i, (x0, a0) in enumerate(zip([x1, x2], [a1, a2])):
    for i, (x0, a0) in enumerate(zip([x1], [a1])):
        for frequency in ["low", "high"]:
            N = 10 * len(x0)
            freq = 10 if frequency == "low" else 100
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

            bounds = np.array([[-1], [1]])
            lambdas = [0.001, 0.01, 0.02, 0.1]

            smooth_grid_size = 1000 if frequency == "low" else 10000
            options = {"initialization": "smoothing", "add_one": True, "swarm": False, "sliding": True,
                       "max_iter": 100, "dual_certificate_tol": 1e-2, "smooth_sigma": 25, "smooth_grid_size": smooth_grid_size}
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

            costs[f"SFW_{frequency}_S{i+1}"] = solver.get_flat_norm_values(x0, a0, lambda_grid)

            options = {"initialization": "smoothing", "add_one": False, "swarm": False, "sliding": False,
                       "max_iter": 100, "dual_certificate_tol": 1e-2, "smooth_sigma": 25, "smooth_grid_size": smooth_grid_size}
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

            costs[f"PFW_{frequency}_S{i+1}"] = solver.get_flat_norm_values(x0, a0, lambda_grid)

    solver.flat_norm_results(x0, a0, lambdas)

    for name, cost in costs.items():
        plt.plot(lambda_grid, cost, label=name)

    # plt.semilogx()
    plt.legend()
    plt.show()
