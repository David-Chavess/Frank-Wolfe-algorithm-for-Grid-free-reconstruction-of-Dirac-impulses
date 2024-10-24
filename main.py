from time import time

import matplotlib.pyplot as plt
import numpy as np

from src.operators.fourier_operator import FourierOperator
from src.solvers.fw import FW


def add_snr(y0, snr, N):
    signal_power = np.mean(np.square(y0))
    mse_db = 10 * np.log10(signal_power) - snr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse / 2), (N, 2))
    y = y0 + w
    return y


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse / 2), (N, 2))
    y = y0 + w
    return y


if __name__ == '__main__':
    np.random.seed(1)

    x0 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    # x0 = np.array([-.5,-.1,.1,.5])
    # a0 = np.array([0.8,0.8,0.8,0.8])
    # a0 = np.array([1, 1.5, 0.5, 2, 5])
    # a0 = np.array([1, 15, 0.5, -3, 5])
    a0 = np.array([1, 1, 1, 1, 1])

    # x0 = np.random.uniform(-0.95, 0.95, 20) * 10
    # a0 = np.random.uniform(0.5, 3, 20)

    # x0 = np.array([0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # a0 = np.array([-1, 0.5, 1, 1, 1, 3, 1, 1])

    x0 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    a0 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, -1.5, 1, 1, 1, 3, 1, 1])
    # a0 = np.abs(a0)$

    # bounds = np.array([[0], [1]])
    bounds = np.array([[-1], [1]])
    # bounds = np.array([[-10], [10]])

    # N = 100
    # grid = np.linspace(-1, 1, N)
    #
    # sigma = 0.01
    # kernel = lambda x: np.exp(-1 * x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    # forward = [kernel(grid - i) for i in x0]
    #
    # out = np.stack(forward, axis=1) @ a0
    # print(out)
    #
    # conv = FFTConvolve(a0, [kernel(grid)] * 5, x0)
    # print(conv(a0))
    #
    # plt.plot(grid, out)
    # plt.show()
    #
    # print("Check adjoint:")
    # y = np.random.uniform(-1, 1, N)
    # print(conv(a0) @ y)
    # print(conv.adjoint(y) @ a0)
    #
    # exit(0)

    N = 10 * len(x0)
    freq_bounds = np.array([-10, 10])
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, freq_bounds)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)
    # y = add_snr(y0, psnr, N)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    x_dim = 1

    lambdas = [0.001, 0.01, 0.02, 0.1]

    # options = {"initialization": "smoothing", "add_one": True, "swarm": False, "sliding": True,
    #            "max_iter": 20, "dual_certificate_tol": 1e-3, "smooth_sigma": 25, "smooth_grid_size": 1000}
    # solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    # t1 = time()
    # solver.fit()
    # print("Time: ", time() - t1)
    # solver.time_results()
    # solver.plot(x0, a0)
    # solver.flat_norm_results(x0, a0, lambdas)
    # solver.plot_solution(x0, a0)
    #
    # costs1 = solver.get_flat_norm_values(x0, a0, np.logspace(-3, 0, 25))

    options = {"initialization": "smoothing", "add_one": False, "swarm": False, "sliding": False,
               "max_iter": 20, "dual_certificate_tol": 1e-3, "smooth_sigma": 25, "smooth_grid_size": 1000}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    solver.time_results()
    solver.plot(x0, a0)
    solver.flat_norm_results(x0, a0, lambdas)
    solver.plot_solution(x0, a0)
    # costs2 = solver.get_flat_norm_values(x0, a0, np.logspace(-3, 0, 25))
    #
    # plt.plot(np.logspace(-3, 0, 25), costs1, label="Sliding")
    # plt.plot(np.logspace(-3, 0, 25), costs2, label="Polyatomic")
    # plt.semilogx()
    # plt.legend()
    # plt.show()
