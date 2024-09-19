from time import time

import numpy as np
from matplotlib import pyplot as plt
from pyxu.operator.linop.fft.filter import FFTConvolve
from pyxu.util import view_as_real, view_as_complex

from src.operators.conv_operator import ConvOperator
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
    a0 = np.array([1, 1.5, 0.5, 2, 5])
    # a0 = np.array([1, 15, 0.5, -3, 5])
    # a0 = np.array([1, 1, 1, 1, 1])

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

    # print("Check adjoint:")
    # y = np.random.uniform(-1, 1, N)
    # print(conv(a0) @ y)
    # print(conv.adjoint(y) @ a0)
    #
    # exit(0)

    # fc = 20
    # N = 2 * fc + 1
    # forward_op = FourierOperator.get_PeriodicFourierOperator(x0, N, fc)

    N = 100
    bounds = np.array([-10, 10])
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, bounds)

    # print("Check forward:")
    # w = forward_op.w
    # op = NUFFT3(x0, w)
    # f1 = forward_op(a0)
    # f2 = view_as_complex(op(view_as_real(a0.astype(np.complex128))))
    # print(np.allclose(f1, f2, atol=1e-2))
    #
    # a1 = forward_op.adjoint(f1)
    # a2 = np.real(view_as_complex(op.adjoint(view_as_real(f1))))
    # print(np.allclose(a1, a2, atol=1e-2))

    print("Check adjoint:")
    y = np.random.uniform(-1, 1, (N, 2))
    print(np.real(view_as_complex(forward_op(a0)).conj().T @ view_as_complex(y)))
    print(forward_op.adjoint(y) @ a0)
    print(forward_op.adjoint_function(y)(forward_op.x) @ a0)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)
    # y = add_snr(y0, psnr, N)

    # Get lambda
    lambda_max = np.linalg.norm(forward_op.adjoint(y), np.inf)
    print("lambda_max = ", lambda_max)
    lambda_ = 0.05 * lambda_max

    solver = FW(y, forward_op, lambda_)
    solver.fit()
    solver.plot(x0, a0)
