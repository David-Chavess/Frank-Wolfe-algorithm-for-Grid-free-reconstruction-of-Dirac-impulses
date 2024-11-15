import matplotlib.pyplot as plt
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

    x1 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    a1 = np.array([1, 1, 1, 1, 1])
    bounds1 = np.array([0, 1])

    x2 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    a2 = np.array([1, 1.5, 0.5, 2, 5])
    bounds2 = np.array([0, 1])

    x3 = np.array([0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    a3 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    bounds3 = np.array([0, 1])

    x4 = np.array([0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    a4 = np.array([-1, 0.5, 1, 1, 1, 3, 1, 1])
    bounds4 = np.array([0, 1])

    x5 = np.random.uniform(-0.95, 0.95, 20)
    a5 = np.random.uniform(0.5, 3, 20)
    bounds5 = np.array([-1, 1])

    x6 = np.random.uniform(-0.95, 0.95, 20) * 10
    a6 = np.random.uniform(0.5, 3, 20)
    bounds6 = np.array([-10, 10])

    x7 = np.array([-0.89, -0.7, -0.68, -0.55, -0.46, - 0.24, -0.2, -0.05, 0.1, 0.25, 0.5, 0.51, 0.7, 0.75, 0.9, 0.92])
    a7 = np.array([3, 4.5, -1.5, -3, 4, 3, 1, 2.5, -1, 0.5, 1, 1, 1, 3, 1, 1])
    bounds7 = np.array([-1, 1])

    ls = list(zip([x1, x2, x3, x4, x5, x6, x7], [a1, a2, a3, a4, a5, a6, a7],
             [bounds1, bounds2, bounds3, bounds4, bounds5, bounds6, bounds7]))

    s = list(zip([x1, x2], [a1, a2],
                 [bounds1, bounds2]))
    N = 100

    for sigma in [10, 25, 50, 100]:
        res = []
        for idx, (x0, a0, bounds) in enumerate(ls):
            new_res = []
            for f_max in [10, 50, 100, 500, 1000, 5000, 10000]:
                print(f"Signal {idx + 1}, f_max {f_max}, sigma {sigma}")
                bounds = np.array([-f_max, f_max])
                forward_op = FourierOperator.get_RandomFourierOperator(x0, N, bounds)

                # Get measurements
                y0 = forward_op(a0)

                # add noise
                psnr = 20
                y = add_psnr(y0, psnr, N)

                # Get lambda
                lambda_max = max(abs((forward_op.adjoint(y))))
                lambda_ = 0.1 * lambda_max

                x_dim = 1

                options = {"initialization": "smoothing", "polyatomic": True, "swarm": False, "sliding": False,
                           "swarm_n_particles": 100,
                           "max_iter": 5, "dual_certificate_tol": 1e-2, "smooth_sigma": sigma, "smooth_grid_size": 10000}
                solver = FW(y, forward_op, lambda_, x_dim, bounds=bounds, verbose=False, show_progress=False,
                            options=options)
                solver.fit()

                new_res.append((f_max, np.mean(solver._mstate["n_candidates_smooth"])))

            res.append(new_res)

        for idx, r in enumerate(res):
            x, y = zip(*r)
            plt.plot(x, y, label=f"Signal {idx + 1} - sigma {sigma}")

    plt.xlabel("f_max")
    plt.ylabel("n_candidates_smooth")
    plt.xscale("log")
    plt.show()
