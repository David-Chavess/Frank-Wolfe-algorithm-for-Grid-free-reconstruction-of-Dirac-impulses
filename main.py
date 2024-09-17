import numpy as np

from src.operators.fourier_operator import FourierOperator
from src.solvers.fw import FW

if __name__ == '__main__':
    np.random.seed(1)

    x0 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    # a0 = np.array([1, 1.5, 0.5, 2, 5])
    # a0 = np.array([1, 15, 0.5, -3, 5])
    a0 = np.array([1, 1, 1, 1, 1])

    # fc = 20
    # N = 2 * fc + 1
    # forward_op = FourierOperator.get_PeriodicFourierOperator(x0, N, fc)

    N = 100
    bounds = np.array([-10, 10])
    forward_op = FourierOperator.get_RandomFourierOperator(x0, N, bounds)

    print("Check adjoint:")
    y = np.random.uniform(-1, 1, N) + 1.j * np.random.uniform(-1, 1, N)
    print(np.real(forward_op(a0).conj().T @ y))
    print(forward_op.adjoint(y) @ a0)
    print(forward_op.adjoint_function(y)(forward_op.x) @ a0)

    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse / 2), (N, 2)).view(np.complex128).ravel()
    y = y0 + w

    lambda_max = np.linalg.norm(forward_op.adjoint(y), np.inf)
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    solver = FW(y, forward_op, lambda_)
    solver.fit()
    solver.plot(x0, a0)
    # print(solvers.solution())