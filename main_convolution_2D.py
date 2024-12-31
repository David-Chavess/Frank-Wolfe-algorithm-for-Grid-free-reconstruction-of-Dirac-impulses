from time import time

import matplotlib.pyplot as plt
import numpy as np

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
    np.random.seed(1)

    x0 = np.array([[0.1, 0.1], [0.2, 0.5], [0.75, 0.25], [0.1, 0.9], [0.5, 0.5]])
    a0 = np.array([1, 1.5, 2.5, 2, 3])

    n = 25
    x0 = np.random.uniform(0, 1, size=(n, 2))
    a0 = np.random.uniform(1, 3, n)

    x_dim = 2
    bounds = np.array([0, 1])

    fwhm = 0.1
    n_measurements_per_pixel = 10
    forward_op = ConvolutionOperator(x0, fwhm, bounds, x_dim, n_measurements_per_pixel)
    N = forward_op.n_measurements
    print("N = ", N)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)

    # n = np.sqrt(N).astype(int)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    # im = ax1.imshow(np.rot90(y0.reshape(n, n)), extent=(0, 1, 0, 1))
    # plt.colorbar(im, ax=ax1)
    # ax1.scatter(x0[:, 0], x0[:, 1], s=np.abs(a0) * 20, marker="+", c='k', label='Ground Truth')
    # ax1.set_title("Ground Truth")
    #
    # im = ax2.imshow(np.rot90(y.reshape(n, n)), extent=(0, 1, 0, 1))
    # plt.colorbar(im, ax=ax2)
    # ax2.scatter(x0[:, 0], x0[:, 1], s=np.abs(a0) * 20, marker="+", c='k', label='Ground Truth')
    # ax2.set_title("Observed Noisy Signal")
    # plt.show()
    # exit(0)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    lambdas = [0.001, 0.01, 0.02, 0.1]

    options = {"initialization": "smoothing", "polyatomic": False, "swarm": False, "sliding": True, "positive_constraint": True,
               "max_iter": 100, "dual_certificate_tol": 1e-2, "smooth_sigma": 2}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    solver.time_results()
    solver.plot(x0, a0)
    solver.flat_norm_results(x0, a0, lambdas)
    solver.plot_solution(x0, a0)

    options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False, "positive_constraint": True,
               "max_iter": 100, "dual_certificate_tol": 1e-2, "smooth_sigma": 2}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    solver.time_results()
    solver.plot(x0, a0)
    solver.flat_norm_results(x0, a0, lambdas)
    solver.plot_solution(x0, a0)
