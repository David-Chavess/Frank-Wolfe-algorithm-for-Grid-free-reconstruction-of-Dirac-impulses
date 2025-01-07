from typing import Callable

import numpy as np
import pyxu.info.ptype as pxt

from src.operators.my_lin_op import MyLinOp


class ConvolutionOperator(MyLinOp):
    """Convolution operator for 1D or 2D."""

    def __init__(self, x: pxt.NDArray, fwhm: float, bounds: np.ndarray, x_dim: int = 1, n_measurements_per_gaussan: int = 3):
        """
        Parameters
        ----------
        x : pxt.NDArray
            Position of the Diracs.
        fwhm : float
            Full width at half maximum of the Gaussian kernel.
        bounds : np.ndarray
            Bounds of the domain where the Diracs postions live.
        x_dim : int, optional
            Dimension of the Diracs positions, by default 1.
        n_measurements_per_gaussan : int, optional
            Number of measurements per Gaussian, by default 3.
        """
        self.x = x.reshape(-1, x_dim)
        self.bounds = bounds
        self.x_dim = x_dim
        self.n_measurements_per_gaussan = n_measurements_per_gaussan

        self.fwhm = fwhm
        self.sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = bounds[1] - bounds[0]
        self.n_measurements = int(n_measurements_per_gaussan * size / fwhm) + 1

        self.kernel = lambda t: np.exp(-1 * np.sum(t ** 2, axis=2) / (2 * self.sigma ** 2)) / ((2 * np.pi) ** (x_dim / 2) * self.sigma ** x_dim)

        if x_dim == 1:
            grid = np.linspace(bounds[0], bounds[1], self.n_measurements)
            self.outer_sub = lambda t: np.subtract.outer(grid, t.ravel())[:,:,None]
        elif x_dim == 2:
            grid = np.linspace(bounds[0], bounds[1], self.n_measurements)
            grid = np.array(np.meshgrid(grid, grid)).T.reshape(-1, 2)
            self.grid = grid
            self.n_measurements = self.n_measurements ** x_dim
            self.outer_sub = lambda t: grid[:, None, :] - t.reshape(-1, x_dim)[None, :, :]
        else:
            raise ValueError("x_dim must be 1 or 2")

        self.forward_pre = self.kernel(self.outer_sub(self.x))

        self.scaling = 1 / self.fwhm

        super().__init__(max(len(x), 1), self.n_measurements)

    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        return self.forward_pre @ a

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        # Equivalent to self.adjoint_function(y)(self.x)
        return self.forward_pre.T @ y

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        return lambda t: self.kernel(self.outer_sub(t)).T @ y

    def adjoint_function_grad(self, y: pxt.NDArray) -> Callable:
        def tmp(t):
            w = self.outer_sub(t)
            out = ((w * self.kernel(w)[:,:,None] / self.sigma ** 2).T @ y).T
            return out
        return tmp

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return ConvolutionOperator(x, self.fwhm, self.bounds, self.x_dim, self.n_measurements_per_gaussan)

    def is_complex(self) -> bool:
        return False

    def get_DiffOperator(self) -> MyLinOp:
        return DiffConvolutionOperator(self.fwhm, self.bounds, 2 * len(self.x), self.x_dim, self.n_measurements_per_gaussan)

    def get_scaling(self) -> pxt.Real:
        return self.scaling


class DiffConvolutionOperator(MyLinOp):

    def __init__(self, fwhm: float, bounds: np.ndarray, input_size: int, x_dim: int = 1, n_measurements_per_pixel: int = 3):
        """
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel.
        bounds : np.ndarray
            Bounds of the domain where the Diracs postions live.
        x_dim : int, optional
            Dimension of the Diracs positions, by default 1.
        n_measurements_per_gaussan : int, optional
            Number of measurements per Gaussian, by default 3.
        """
        self.bounds = bounds
        self.input_size = input_size
        self.x_dim = x_dim
        self.n_measurements_per_pixel = n_measurements_per_pixel

        self.fwhm = fwhm
        self.sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = bounds[1] - bounds[0]
        self.n_measurements = int(n_measurements_per_pixel * size / fwhm) + 1

        self.kernel = lambda t: np.exp(-1 * np.sum(t ** 2, axis=2) / (2 * self.sigma ** 2)) / (
                    (2 * np.pi) ** (x_dim / 2) * self.sigma ** x_dim)

        if x_dim == 1:
            grid = np.linspace(bounds[0], bounds[1], self.n_measurements)
            self.outer_sub = lambda t: np.subtract.outer(grid, t.ravel())[:, :, None]
        elif x_dim == 2:
            grid = np.linspace(bounds[0], bounds[1], self.n_measurements)
            grid = np.array(np.meshgrid(grid, grid)).T.reshape(-1, 2)
            self.grid = grid
            self.n_measurements = self.n_measurements ** x_dim
            self.outer_sub = lambda t: grid[:, None, :] - t.reshape(-1, x_dim)[None, :, :]
        else:
            raise ValueError("x_dim must be 1 or 2")

        self.scaling = 1 / self.fwhm

        super().__init__(max(input_size, 1), self.n_measurements)

    def apply(self, xa: pxt.NDArray) -> pxt.NDArray:
        a = np.split(xa, 1 + self.x_dim)[-1]
        x = xa[:-len(a)]
        return self.kernel(self.outer_sub(x)) @ a

    def grad_x(self, xa: pxt.NDArray) -> pxt.NDArray:
        a = np.split(xa, 1 + self.x_dim)[-1].reshape(-1, 1)
        x = xa[:-len(a)]
        w = self.outer_sub(x)
        tmp = (w * self.kernel(w)[:,:,None] / self.sigma ** 2).T.reshape(-1, self.n_measurements)

        if self.x_dim == 1:
            out = a * tmp
        elif self.x_dim == 2:
            out = np.empty(tmp.shape)
            out[0::2] = a * tmp[:len(tmp)//2]
            out[1::2] = a * tmp[len(tmp)//2:]
        else:
            raise ValueError("x_dim must be 1 or 2")
        return out

    def grad_a(self, xa: pxt.NDArray) -> pxt.NDArray:
        a = np.split(xa, 1 + self.x_dim)[-1]
        x = xa[:-len(a)]
        return self.kernel(self.outer_sub(x)).T

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        pass

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        return lambda t: self.kernel(self.outer_sub(t)).T @ y

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return DiffConvolutionOperator(self.fwhm, self.bounds, self.input_size, self.x_dim, self.n_measurements_per_pixel)

    def is_complex(self) -> bool:
        return False

    def get_scaling(self) -> pxt.Real:
        return self.scaling
