from typing import Callable

import numpy as np
import pyxu.info.ptype as pxt
from pyxu.abc.operator import Func
from pyxu.util import view_as_real, view_as_complex

from src.operators.my_lin_op import MyLinOp


class FourierOperator(MyLinOp):

    def __init__(self, x: pxt.NDArray, w: pxt.NDArray, n_measurements: int):
        self.x = x
        self.w = w.reshape(-1, 1)
        self.n_measurements = n_measurements

        self.fourier = np.exp(-2j * np.pi * np.outer(self.w, x))

        super().__init__(max(len(x), 1), (n_measurements, 2))

    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        return view_as_real(self.fourier @ a)

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        # return self.adjoint_function(y)(self.x)
        y = view_as_complex(y)
        return np.real(self.fourier.conj().T @ y)

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        y = view_as_complex(y)
        return lambda t: np.real(np.exp(2j * np.pi * np.outer(self.w, t)).T @ y)

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return FourierOperator(x, self.w, self.n_measurements)

    @staticmethod
    def get_RandomFourierOperator(x: pxt.NDArray, n_measurements: int, bounds: pxt.NDArray) -> MyLinOp:
        w = np.random.uniform(bounds[0], bounds[1], n_measurements)
        return FourierOperator(x, w, n_measurements)

    @staticmethod
    def get_PeriodicFourierOperator(x: pxt.NDArray, n_measurements: int, fc: int) -> MyLinOp:
        w = np.arange(-fc, fc + 1)
        return FourierOperator(x, w, n_measurements)


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
        self.op = operator
        self.lambda_ = lambda_

        phi = operator
        phiS = phi.adjoint_function(measurements - phi(ak))

        self.fun = lambda t: np.abs(phiS(t) / self.lambda_)
        super().__init__(x_dim, 1)

    def apply(self, t: pxt.NDArray) -> pxt.NDArray:
        return self.fun(t)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        pass

    def _meta(self):
        pass
