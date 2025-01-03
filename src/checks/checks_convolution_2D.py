import numpy as np
from pyxu.util import view_as_complex
from scipy.optimize import minimize

from src.operators.convolution_operator import ConvolutionOperator
from src.operators.dual_certificate import DualCertificate
from src.operators.fourier_operator import FourierOperator


def add_psnr(y0, psnr, N):
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    w = np.random.normal(0, np.sqrt(mse), N)
    y = y0 + w
    return y


if __name__ == '__main__':
    np.random.seed(1)

    x0 = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.1, 0.9], [0.5, 0.5]])
    a0 = np.array([1, 1.5, 3, 2, 5])

    x_dim = 2
    bounds = np.array([-1, 1])

    fwhm = 0.5
    forward_op = ConvolutionOperator(x0, fwhm, bounds, x_dim)
    N = forward_op.n_measurements
    print("N = ", N)

    print("Check adjoint:")
    y = np.random.uniform(-1, 1, N)
    print(forward_op(a0).T @ y)
    print(forward_op.adjoint(y) @ a0)
    print(forward_op.adjoint_function(y)(forward_op.x) @ a0)

    # Get measurements
    y0 = forward_op(a0)

    # add noise
    psnr = 20
    y = add_psnr(y0, psnr, N)

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    op = forward_op.get_DiffOperator()


    def fun(xa):
        a = np.split(xa, 1 + x_dim)[-1]
        z = op(xa) - y
        return z.T @ z / 2 + lambda_ * np.sum(np.abs(a))


    def grad(xa):
        a = np.split(xa, 1 + x_dim)[-1]
        z = op(xa) - y
        grad_x = op.grad_x(xa) @ z
        grad_a = op.grad_a(xa) @ z + lambda_ * np.sign(a)
        return np.concatenate([grad_x, grad_a])


    def finite_grad(xa):
        finite_grad = np.zeros_like(xa)
        eps = 1e-10
        for i in range(len(xa)):
            xa_eps = xa.copy()
            xa_eps[i] += eps
            finite_grad[i] = (fun(xa_eps) - fun(xa)) / eps
        return finite_grad

    xa = np.concatenate([x0[:, 0], x0[:, 1], a0])
    assert np.allclose(grad(xa), finite_grad(xa), atol=1e-2)
    xa = np.concatenate([[0.1], [0.6], [1.]])
    assert np.allclose(grad(xa), finite_grad(xa), atol=1e-2)
    xa = np.concatenate([[0.11111], [0.222222], [0.001]])
    assert np.allclose(grad(xa), finite_grad(xa), atol=1e-2)

    n = -len(a0)
    out = minimize(fun, np.concatenate([x0[:, 0] + 0.01, x0[:, 1] + 0.01, a0 - 0.1]), method="BFGS")
    print(np.split(out.x, 3))
    print(out)
    x = out.x[:n].reshape(-1, x_dim)
    a = out.x[n:]
    print("Distance: ", np.sum(np.abs(x - x0)))

    out = minimize(fun, np.concatenate([x0[:, 0] + 0.01, x0[:, 1] + 0.01, a0 - 0.1]), method="BFGS", jac=grad)
    print(np.split(out.x, 3))
    print(out)
    x = out.x[:n].reshape(-1, x_dim)
    a = out.x[n:]
    print("Distance: ", np.sum(np.abs(x - x0)))

    dual_certificates = DualCertificate(x0, a0, y, forward_op, lambda_, x_dim)

    def finite_grad_2d(x):
        finite_grad = np.zeros_like(x)
        eps = 1e-10
        for i, v in enumerate(x):
            dx = v + np.array([eps, 0])
            finite_grad[i][0] = (dual_certificates(dx) - dual_certificates(v)) / eps
            dy = v + np.array([0, eps])
            finite_grad[i][1] = (dual_certificates(dy) - dual_certificates(v)) / eps
        return finite_grad.reshape(-1, x_dim)


    assert np.allclose(dual_certificates.grad(x0), finite_grad_2d(x0), atol=1e-2)
    x0 = np.array([[0.1, 0.1]])
    assert np.allclose(dual_certificates.grad(x0), finite_grad_2d(x0), atol=1e-2)
    x0 = np.array([[0.55555, 0.88888], [0.001, -0.0001], [0.76346, -0.34687], [0., 0.], [1., 1.]])
    assert np.allclose(dual_certificates.grad(x0), finite_grad_2d(x0), atol=1e-2)
