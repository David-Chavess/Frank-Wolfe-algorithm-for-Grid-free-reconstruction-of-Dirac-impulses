from time import time

import numpy as np

from src.operators.convolution_operator import ConvolutionOperator
from src.operators.dual_certificate import DualCertificate
from src.solvers.fw import FW


def add_snr(y0, snr, N):
    signal_power = np.mean(np.square(y0))
    mse_db = 10 * np.log10(signal_power) - snr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse), N)
    y = y0 + w
    return y


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse), N)
    y = y0 + w
    return y


if __name__ == '__main__':
    np.random.seed(1)

    x0 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    a0 = np.array([1, 1.5, 3.5, 2, 5])
    # a0 = np.array([1, 15, 0.5, -3, 5])
    # a0 = np.array([1, 1, 1, 1, 1])

    x_dim = 1
    bounds = np.array([[0], [1]])
    # bounds = np.array([[-1], [1]])

    # Full width at half maximum
    fwhm = 0.1
    forward_op = ConvolutionOperator(x0, fwhm, bounds)
    N = forward_op.n_measurements

    # for i in x0:
    #     gauss = kernel(grid - i)
    #     plt.plot(grid, gauss)
    #
    # plt.plot(grid, forward)
    # plt.stem(x0, a0, 'r')
    # plt.show()

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

    # options = {"initialization": "smoothing", "polyatomic": False, "swarm": True, "sliding": True,
    #            "max_iter": 20, "dual_certificate_tol": 1e-2, "smooth_sigma": 5}
    # solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=True, options=options)
    # t1 = time()
    # solver.fit()
    # print("Time: ", time() - t1)
    # solver.time_results()
    # solver.plot(x0, a0)
    # solver.flat_norm_results(x0, a0, lambdas)
    # solver.plot_solution(x0, a0)

    options = {"initialization": "random", "polyatomic": True, "swarm": True, "sliding": False,
               "max_iter": 20, "dual_certificate_tol": 1e-2, "smooth_sigma": 2.5}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=True, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    solver.time_results()
    solver.plot(x0, a0)
    solver.flat_norm_results(x0, a0, lambdas)
    solver.plot_solution(x0, a0)
