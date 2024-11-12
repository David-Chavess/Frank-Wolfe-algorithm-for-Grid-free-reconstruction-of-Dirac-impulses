from typing import Callable

import numpy as np
import pyxu.info.ptype as pxt

from src.operators.my_lin_op import MyLinOp


class ConvolutionOperator(MyLinOp):

    def __init__(self, x: pxt.NDArray, fwhm: float, bounds: np.ndarray):
        self.x = x.ravel()
        self.bounds = bounds
        self.fwhm = fwhm
        self.sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = bounds[1] - bounds[0]
        self.n_measurements = int(3 * size / fwhm) + 1
        grid = np.linspace(bounds[0], bounds[1], self.n_measurements).ravel()

        self.kernel = lambda t: np.exp(-1 * t ** 2 / (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))
        self.outer_sub = lambda t: np.subtract.outer(grid, t)
        self.forward_pre = self.kernel(self.outer_sub(self.x))

        self.w = 1 / self.fwhm

        super().__init__(max(len(x), 1), self.n_measurements)

    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        return self.forward_pre @ a

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        # Equivalent to self.adjoint_function(y)(self.x)
        return self.forward_pre.T @ y

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        return lambda t: self.kernel(self.outer_sub(t)).T @ y

    def adjoint_function_grad(self, y: pxt.NDArray) -> Callable:
        return lambda t: (self.outer_sub(t) * self.kernel(self.outer_sub(t)) / self.sigma ** 2).T @ y

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return ConvolutionOperator(x, self.fwhm, self.bounds)

    def is_complex(self) -> bool:
        return False

    def get_DiffOperator(self) -> MyLinOp:
        return DiffConvolutionOperator(self.fwhm, self.bounds, 2*len(self.x))


class DiffConvolutionOperator(MyLinOp):

    def __init__(self, fwhm: float, bounds: np.ndarray, input_size: int):
        self.bounds = bounds
        self.input_size = input_size
        self.fwhm = fwhm
        self.sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = bounds[1] - bounds[0]
        self.n_measurements = int(3 * size / fwhm) + 1
        grid = np.linspace(bounds[0], bounds[1], self.n_measurements).ravel()

        self.kernel = lambda t: np.exp(-1 * t ** 2 / (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))
        self.outer_sub = lambda t: np.subtract.outer(grid, t)

        super().__init__(max(input_size, 1), self.n_measurements)

    def apply(self, xa: pxt.NDArray) -> pxt.NDArray:
        x, a = np.split(xa, 2)
        return self.kernel(self.outer_sub(x)) @ a

    def grad_x(self, xa: pxt.NDArray) -> pxt.NDArray:
        x, a = np.split(xa, 2)
        return (self.outer_sub(x) * self.kernel(self.outer_sub(x)) / self.sigma ** 2).T

    def grad_a(self, xa: pxt.NDArray) -> pxt.NDArray:
        x, a = np.split(xa, 2)
        return self.kernel(self.outer_sub(x)).T

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        pass

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        return lambda t: self.kernel(self.outer_sub(t)).T @ y

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return DiffConvolutionOperator(self.fwhm, self.bounds, self.input_size)

    def is_complex(self) -> bool:
        return False
