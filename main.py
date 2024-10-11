from time import time

import numpy as np

from src.metrics.flat_norm import flat_norm
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

    # x0 = np.random.uniform(-0.95, 0.95, 20)
    # a0 = np.random.uniform(0.5, 3, 20)

    # x0 = np.array([0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    # a0 = np.array([-1, 0.5, 1, 1, 1, 3, 1, 1])

    # x0 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    # a0 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, 0.5, 1, 1, 1, 3, 1, 1])
    # a0 = np.abs(a0)

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

    N = 100
    bounds = np.array([-100, 100])
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, bounds)

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

    bounds = np.array([[-1], [1]])

    options = {"initialization": "smoothing", "add_one": True, "swarm": True, "sliding": True,
               "swarm_n_particles": 100, "max_iter": 20, "dual_certificate_tol": 1e-2, "smooth_sigma": 50, "smooth_grid_size": 10000}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    print("Candidates search Time: ", solver._mstate["candidates_search_durations"])
    print("Correction Time: ", solver._mstate["correction_durations"])
    print("Correction iterations: ", solver._mstate["correction_iterations"])
    print("Sliding Time: ", solver._mstate["sliding_durations"])
    print("Iterations: ", solver._astate["idx"])
    print("Dual certificate: ", solver._mstate["dual_certificate"])
    solver.plot(x0, a0)
    solver.plot_solution(x0, a0, merged=False)

    x, a = solver.solution()
    # x, a = solver.merged_solution()
    print("Distance: ", flat_norm(x0, x, a0, a, 0.01).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.02).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.05).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.1).cost)

    options = {"initialization": "smoothing", "add_one": False, "swarm": True, "sliding": False,
               "swarm_n_particles": 100, "max_iter": 20, "dual_certificate_tol": 1e-2, "smooth_sigma": 50, "smooth_grid_size": 10000}
    solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False, options=options)
    t1 = time()
    solver.fit()
    print("Time: ", time() - t1)
    print("Candidates search Time: ", solver._mstate["candidates_search_durations"])
    print("Correction Time: ", solver._mstate["correction_durations"])
    print("Correction iterations: ", solver._mstate["correction_iterations"])
    print("Sliding Time: ", solver._mstate["sliding_durations"])
    print("Iterations: ", solver._astate["idx"])
    print("Dual certificate: ", solver._mstate["dual_certificate"])
    solver.plot(x0, a0)
    solver.plot_solution(x0, a0, merged=False)
    solver.plot_solution(x0, a0, merged=True)

    x, a = solver.solution()
    print("Distance: ", flat_norm(x0, x, a0, a, 0.01).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.02).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.05).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.1).cost)

    x, a = solver.merged_solution()
    print("Merged")
    print("Distance: ", flat_norm(x0, x, a0, a, 0.01).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.02).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.05).cost)
    print("Distance: ", flat_norm(x0, x, a0, a, 0.1).cost)
