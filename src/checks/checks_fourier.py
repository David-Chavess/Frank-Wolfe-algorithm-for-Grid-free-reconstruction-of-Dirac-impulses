import numpy as np
from pyxu.util import view_as_complex
from scipy.optimize import minimize

from src.operators.dual_certificate import DualCertificate
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
    a0 = np.array([1, 1.5, 0.5, -2, 5])

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

    # Get lambda
    lambda_max = max(abs((forward_op.adjoint(y))))
    print("lambda_max = ", lambda_max)
    lambda_ = 0.1 * lambda_max

    op = forward_op.get_DiffOperator()

    x_dim = 1

    def fun(xa):
        a = np.split(xa, 1 + x_dim)[-1]
        z = op(xa) - view_as_complex(y)
        return np.real(z.T.conj() @ z) / 2 + lambda_ * np.sum(np.abs(a))


    def grad(xa):
        a = np.split(xa, 1 + x_dim)[-1]
        z = op(xa) - view_as_complex(y)
        grad_x = op.grad_x(xa) @ z
        grad_a = op.grad_a(xa) @ z + lambda_ * np.sign(a)
        return np.real(np.concatenate([grad_x, grad_a]))


    def finite_grad(xa):
        finite_grad = np.zeros_like(xa)
        eps = 1e-10
        for i in range(len(xa)):
            xa_eps = xa.copy()
            xa_eps[i] += eps
            finite_grad[i] = (fun(xa_eps) - fun(xa)) / eps
        return finite_grad


    xa = np.concatenate([x0, a0])
    assert np.allclose(grad(xa), finite_grad(xa), atol=1e-2)
    xa = np.concatenate([[0.1], [1.]])
    assert np.allclose(grad(xa), finite_grad(xa), atol=1e-2)
    xa = np.concatenate([[0.11111], [0.001]])
    assert np.allclose(grad(xa), finite_grad(xa), atol=1e-2)

    out = minimize(fun, np.concatenate([x0 + 0.01, a0 - 0.1]), method="BFGS")
    print(np.split(out.x, 2))
    print(out)
    x = out.x[:len(x0)]
    a = out.x[len(x0):]
    print("Distance: ", np.sum(np.abs(x - x0)))

    out = minimize(fun, np.concatenate([x0 + 0.01, a0 - 0.1]), method="BFGS", jac=grad)
    print(np.split(out.x, 2))
    print(out)
    x = out.x[:len(x0)]
    a = out.x[len(x0):]
    print("Distance: ", np.sum(np.abs(x - x0)))

    dual_certificates = DualCertificate(x0, a0, y, forward_op, lambda_, x_dim)

    def finite_grad(x):
        finite_grad = np.zeros_like(x)
        eps = 1e-10
        for i, v in enumerate(x):
            finite_grad[i] = (dual_certificates(v + eps) - dual_certificates(v)) / eps
        return finite_grad.reshape(-1, x_dim)

    assert np.allclose(dual_certificates.grad(x0), finite_grad(x0), atol=1e-2)
    x0 = np.array([0.1])
    assert np.allclose(dual_certificates.grad(x0), finite_grad(x0), atol=1e-2)
    x0 = np.array([0.55555, 0.001, 0.76346, 0., 1.])
    assert np.allclose(dual_certificates.grad(x0), finite_grad(x0), atol=1e-2)
