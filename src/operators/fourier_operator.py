from typing import Self

import numpy as np
from pyxu.abc.operator import LinOp, Func
import pyxu.info.ptype as pxt
import pyxu.util as pxu


class MyLinOp(LinOp):
    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        pass

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        pass

    def _meta(self):
        pass

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        pass

    def get_new_operator(self, x: pxt.NDArray) -> Self:
        pass


class FourierOperator(MyLinOp):

    def __init__(self, x: pxt.NDArray, w: pxt.NDArray, n_measurements: int):
        self.x = x
        self.w = w.reshape(-1, 1)
        self.n_measurements = n_measurements

        xp = pxu.get_array_module(self.x)
        self.fourier = xp.exp(-2j * np.pi * np.outer(self.w, x))

        super().__init__(max(len(x), 1), n_measurements)

    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        return self.fourier @ a.ravel()

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        return lambda t: np.real(np.exp(2j * np.pi * np.outer(self.w, t)).T @ y)

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return FourierOperator(x, self.w, self.n_measurements)


class RandomFourierOperator(FourierOperator):

    def __init__(self, x: pxt.NDArray, n_measurements: int):
        w = np.random.uniform(-1, 1, n_measurements)
        super().__init__(x, w, n_measurements)

    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        return self.fourier @ a.ravel()

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return RandomFourierOperator(x, self.n_measurements)


class PeriodicFourierOperator(FourierOperator):

    def __init__(self, x: pxt.NDArray, n_measurements: int, fc: int, ):
        self.fc = fc
        super().__init__(x, np.arange(-fc, fc + 1), n_measurements)

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return PeriodicFourierOperator(x, self.n_measurements, self.fc)


class TMP(PeriodicFourierOperator):

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        return super().adjoint(y)(self.x)

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return TMP(x, self.n_measurements, self.fc)


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
        phiS = phi.adjoint(measurements - phi(ak))

        self.fun = lambda t: np.abs(phiS(t) / self.lambda_)
        super().__init__(x_dim, 1)

    def apply(self, t: pxt.NDArray) -> pxt.NDArray:
        return self.fun(t)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        pass

    def _meta(self):
        pass
