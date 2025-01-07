import numpy as np
from pyxu.abc import Func
from pyxu.info import ptype as pxt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import find_peaks
from skimage.feature import peak_local_max

from src.operators.my_lin_op import MyLinOp


class DualCertificate(Func):
    """Class for the empirical dual certificate function: $\eta(x) \stackrel{def}{=} \frac{1}{\lambda} \Phi^{*} (y - \Phi m)$."""

    def __init__(self,
                 xk: pxt.NDArray,
                 ak: pxt.NDArray,
                 measurements: pxt.NDArray,
                 operator: MyLinOp,
                 lambda_: float,
                 positivity_constraint: bool = False,
                 x_dim: int = 1):
        """
        Parameters
        ----------
        xk : pxt.NDArray
            Current estimate of the signal positions at iteration k.
        ak : pxt.NDArray
            Current estimate of the signal amplitudes at iteration k.
        measurements : pxt.NDArray
            Measurements vector.
        operator : MyLinOp
            Linear operator of the problem.
        lambda_ : float
            Regularization parameter.
        positivity_constraint : bool, optional
            Whether the minimization has a positivity constraint, by default False.
        x_dim : int, optional
            Dimension of the Diracs positions, by default 1.
        """
        self.xk = xk
        self.ak = ak
        self.op = operator.get_new_operator(xk)
        self.lambda_ = lambda_
        self.x_dim = x_dim

        phi = self.op
        phiS = phi.adjoint_function(measurements - phi(ak))

        if positivity_constraint:
            self.fun = lambda t: phiS(t) / self.lambda_
        else:
            self.fun = lambda t: np.abs(phiS(t) / self.lambda_)

        def grad(t):
            p = measurements - phi(ak)
            grad_phiS = self.op.adjoint_function_grad(p)

            if positivity_constraint:
                return grad_phiS(t) / self.lambda_
            else:
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
    """Class for the smoothed empirical dual certificate function."""

    def __init__(self,
                 xk: pxt.NDArray,
                 ak: pxt.NDArray,
                 measurements: pxt.NDArray,
                 operator: MyLinOp,
                 lambda_: float,
                 sigma: float,
                 grid: pxt.NDArray,
                 positivity_constraint: bool = False,
                 discrete: bool = True,
                 x_dim: int = 1):
        """
        Parameters
        ----------
        xk : pxt.NDArray
            Current estimate of the signal positions at iteration k.
        ak : pxt.NDArray
            Current estimate of the signal amplitudes at iteration k.
        measurements : pxt.NDArray
            Measurements vector.
        operator : MyLinOp
            Linear operator of the problem.
        lambda_ : float
            Regularization parameter.
        sigma : float
            Standard deviation of the Gaussian kernel.
        grid : pxt.NDArray
            Grid where the smoothed function is evaluated.
        positivity_constraint : bool, optional
            Whether the minimization has a positivity constraint, by default False.
        discrete : bool, optional
            Whether the smoothing is discrete or continuous, by default True.
        x_dim : int, optional
            Dimension of the Diracs positions, by default 1.
        """
        super().__init__(xk, ak, measurements, operator, lambda_, positivity_constraint, x_dim)

        self.sigma = sigma
        self.grid = grid
        self.x_dim = x_dim

        if discrete:
            z = self.fun(self.grid).ravel()
            if self.x_dim == 1:
                self.z_smooth = gaussian_filter1d(z, sigma)
            elif self.x_dim == 2:
                n = int(np.sqrt(z.size))
                z = z.reshape(n, n)
                self.z_smooth = gaussian_filter(z, sigma, mode='constant')
        else:
            if self.x_dim != 1:
                raise ValueError("Continuous smoothing only supported for 1D signals.")

            sigma = 1 / sigma
            gaussian_fourier = lambda x: np.exp(-1 * sigma ** 2 * x ** 2 / 2)
            phi = self.op
            yi = (measurements - phi(ak)) * gaussian_fourier(self.op.w)
            phiS = phi.adjoint_function(yi)

            if positivity_constraint:
                self.fun = lambda t: phiS(t) / self.lambda_
            else:
                self.fun = lambda t: np.abs(phiS(t) / self.lambda_)

            self.z_smooth = self.fun(self.grid)

    def get_peaks(self):
        """Get the peaks positions of the smoothed function."""
        if self.x_dim == 1:
            idx = find_peaks(self.z_smooth)[0]
            peaks = self.grid[idx]
        elif self.x_dim == 2:
            idx = peak_local_max(self.z_smooth)
            idx = [i[0] * self.z_smooth.shape[0] + i[1] for i in idx]
            peaks = self.grid[idx]
        else:
            raise ValueError("Only 1D and 2D signals are supported.")

        return peaks
