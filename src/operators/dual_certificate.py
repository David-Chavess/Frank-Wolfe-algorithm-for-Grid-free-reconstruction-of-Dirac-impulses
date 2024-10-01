import numpy as np
from pyxu.abc import Func
from pyxu.info import ptype as pxt

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

        phi = self.op
        phiS = phi.adjoint_function(measurements - phi(ak))

        self.fun = lambda t: np.abs(phiS(t) / self.lambda_)

        def grad(t):
            p = measurements - phi(ak)
            grad_phiS = self.op.adjoint_function_grad(p)
            return np.sign(phiS(t)) * grad_phiS(t) / self.lambda_

        self.grad = grad

        super().__init__(x_dim, 1)

    def apply(self, t: pxt.NDArray) -> pxt.NDArray:
        return self.fun(t)

    def grad(self, t: pxt.NDArray) -> pxt.NDArray:
        return self.grad(t)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        pass

    def _meta(self):
        pass
