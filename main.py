import numpy as np

from src.solvers.fw import FW
from src.operators.fourier_operator import PeriodicFourierOperator, RandomFourierOperator

if __name__ == '__main__':
    x0 = np.array([0.1, 0.25, 0.5, 0.7, 0.9])
    a0 = np.array([1, 1, 1, 1, 1])

    fc = 6
    N = 2 * fc + 1
    P = 2048 * 8
    u = np.linspace(0, 1, P).reshape(-1, 1)

    lambda_ = 1

    forward_op = PeriodicFourierOperator(x0, N, fc)

    # N = 10
    # forward_op = RandomFourierOperator(x0, N)

    y = np.random.uniform(-1, 1, N) + 1.j * np.random.uniform(-1, 1, N)
    print("Check adjoint:")
    print(np.real(forward_op(a0).conj().T @ y))
    print(forward_op.adjoint(y)(x0))
    print(np.sum(forward_op.adjoint(y)(x0)))

    y0 = forward_op(a0)
    sigma = 0.12 * np.linalg.norm(y0)
    w = np.fft.fftshift(np.fft.fft(np.random.randn(N, 1))).ravel()
    w = w / np.linalg.norm(w) * sigma
    y = y0 + w

    solver = FW(y, forward_op, lambda_)
    solver.fit()
    # print(solvers.solution())