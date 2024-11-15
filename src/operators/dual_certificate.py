import numpy as np
from pyxu.abc import Func
from pyxu.info import ptype as pxt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.operators.my_lin_op import MyLinOp


class DualCertificate(Func):

    def __init__(self,
                 xk: pxt.NDArray,
                 ak: pxt.NDArray,
                 measurements: pxt.NDArray,
                 operator: MyLinOp,
                 lambda_: float,
                 x_dim: int = 1):
        self.xk = xk
        self.ak = ak
        self.op = operator.get_new_operator(xk)
        self.lambda_ = lambda_
        self.x_dim = x_dim

        phi = self.op
        phiS = phi.adjoint_function(measurements - phi(ak))

        self.fun = lambda t: np.abs(phiS(t) / self.lambda_)

        def grad(t):
            p = measurements - phi(ak)
            grad_phiS = self.op.adjoint_function_grad(p)
            return np.sign(phiS(t)).reshape(-1, 1) * grad_phiS(t) / self.lambda_

        self.grad = grad

        super().__init__(x_dim, 1)

    def apply(self, t: pxt.NDArray) -> pxt.NDArray:
        return self.fun(t).ravel()

    def grad(self, t: pxt.NDArray) -> pxt.NDArray:
        return self.grad(t)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        pass

    def _meta(self):
        pass


class SmoothDualCertificate(DualCertificate):

    def __init__(self,
                 xk: pxt.NDArray,
                 ak: pxt.NDArray,
                 measurements: pxt.NDArray,
                 operator: MyLinOp,
                 lambda_: float,
                 sigma: float,
                 grid: pxt.NDArray,
                 discrete: bool = True,
                 x_dim: int = 1):
        super().__init__(xk, ak, measurements, operator, lambda_, x_dim)

        self.sigma = sigma
        self.grid = grid

        if discrete:
            z = self.fun(self.grid).ravel()
            self.z_smooth = gaussian_filter1d(z, sigma)
        else:
            sigma = 1 / sigma
            gaussian_fourier = lambda x: np.exp(-1 * sigma ** 2 * x ** 2 / 2)
            phi = self.op
            yi = (measurements - phi(ak)) * gaussian_fourier(self.op.w)
            phiS = phi.adjoint_function(yi)
            self.fun = lambda t: np.abs(phiS(t) / self.lambda_)
            self.z_smooth = self.fun(self.grid)

    def get_peaks(self):
        peaks = find_peaks(self.z_smooth)[0]
        return self.grid[peaks]
