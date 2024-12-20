from time import time

import matplotlib.pyplot as plt
import numpy as np

from src.operators.dual_certificate import SmoothDualCertificate, DualCertificate
from src.operators.fourier_operator import FourierOperator


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
    # a0 = np.array([1, 1.5, 0.5, 2, 5])
    a0 = np.array([1, 1, 1, 1, 1])

    # x0 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, 0.5, 1, 1, 1, 3, 1, 1])

    # x0 = np.array([0.2, 0.5, 0.8])
    # a0 = np.array([1, 2, 1.5])

    N = 100
    bounds = np.array([-100, 100])
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, bounds)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    x_dim = 1

    bounds = np.array([0, 1])
    # bounds = np.array([-1, 1])

    grid = np.linspace(bounds[0], bounds[1], 10000)
    empty = np.array([])
    cert = DualCertificate(empty, empty, y, forward_op, lambda_)
    plt.plot(grid, cert(grid), label='Dual Certificate', color='tab:blue')

    t1 = time()
    sigma = 50
    smooth_dual_cert = SmoothDualCertificate(empty, empty, y, forward_op, lambda_, sigma, grid, discrete=True)
    particles = smooth_dual_cert.get_peaks()
    print("Time: ", time() - t1)
    print(len(particles))
    plt.plot(grid, smooth_dual_cert.z_smooth, label='Smooth Dual Certificate', color='tab:orange')
    plt.plot(particles, np.zeros_like(particles), 'x', color='tab:orange', label='Initialization Points')

    t1 = time()
    sigma = 100
    smooth_dual_cert = SmoothDualCertificate(empty, empty, y, forward_op, lambda_, sigma, grid, discrete=False)
    particles = smooth_dual_cert.get_peaks()
    print("Time: ", time() - t1)
    print(len(particles))
    plt.plot(grid, smooth_dual_cert.z_smooth, label='Analytical Smooth Dual Certificate', color='tab:red')
    plt.plot(particles, np.zeros_like(particles), 'x', color='tab:red')

    plt.stem(x0, a0, linefmt='k.--', markerfmt='ko', basefmt=" ", label='Ground Truth')
    plt.legend()
    plt.show()
