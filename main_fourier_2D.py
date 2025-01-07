from time import time

import numpy as np

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

    x0 = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.1, 0.9], [0.5, 0.5]])
    a0 = np.array([1, 1.5, 3, 2, 5])

    # x0 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, -1.5, 1, 1, 1, 3, 1, 1])

    x_dim = 2
    bounds = np.array([-1, 1])

    N = 10 * len(x0)
    freq_bounds = np.array([-10, 10])
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, freq_bounds, x_dim)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    lambdas = [0.001, 0.01, 0.02, 0.1]

    options = {"initialization": "smoothing", "polyatomic": False, "swarm": False, "sliding": True, "positivity_constraint": False,
               "max_iter": 20, "dual_certificate_tol": 1e-2, "smooth_sigma": 1}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    solver.time_results()
    solver.plot(x0, a0)
    solver.flat_norm_results(x0, a0, lambdas)
    solver.plot_solution(x0, a0)

    options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False, "positivity_constraint": False,
               "max_iter": 20, "dual_certificate_tol": 1e-2, "smooth_sigma": 1}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    solver.time_results()
    solver.plot(x0, a0)
    solver.flat_norm_results(x0, a0, lambdas)
    solver.plot_solution(x0, a0)
