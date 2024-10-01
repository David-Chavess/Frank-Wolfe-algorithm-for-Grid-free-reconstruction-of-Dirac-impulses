from typing import Callable

import numpy as np
import pyxu.info.ptype as pxt
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
        # Equivalent to self.adjoint_function(y)(self.x)
        y = view_as_complex(y)
        return np.real(self.fourier.conj().T @ y)

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        y = view_as_complex(y)
        return lambda t: np.real(np.exp(2j * np.pi * np.outer(self.w, t)).T @ y)

    def adjoint_function_grad(self, y: pxt.NDArray) -> Callable:
        y = view_as_complex(y)
        return lambda t: np.real(2j * np.pi * self.w.T * np.exp(2j * np.pi * np.outer(self.w, t)).T @ y)

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


class DiffFourierOperator(MyLinOp):

    def __init__(self, w: pxt.NDArray, n_measurements: int, input_size: int):
        self.w = w.reshape(-1, 1)
        self.n_measurements = n_measurements
        self.input_size = input_size

        self.fourier = lambda t: np.exp(-2j * np.pi * np.outer(self.w, t))

        super().__init__((input_size, 1), (n_measurements, 2))

    def apply(self, xa: pxt.NDArray) -> pxt.NDArray:
        x, a = np.split(xa, 2)
        return self.fourier(x) @ a

    def grad_x(self, arr: pxt.NDArray) -> pxt.NDArray:
        x, a = np.split(arr, 2)
        return (self.fourier(x) * -2j * np.pi * self.w).T.conj()

    def grad_a(self, arr: pxt.NDArray) -> pxt.NDArray:
        x, a = np.split(arr, 2)
        return self.fourier(x).T.conj()

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        pass

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        y = view_as_complex(y)
        return lambda t: np.real(np.exp(2j * np.pi * np.outer(self.w, t)).T @ y)

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return DiffFourierOperator(self.w, self.n_measurements, self.input_size)

    @staticmethod
    def get_DiffFourierOperator(op: FourierOperator) -> MyLinOp:
        return DiffFourierOperator(op.w, op.n_measurements, 2*len(op.x))
