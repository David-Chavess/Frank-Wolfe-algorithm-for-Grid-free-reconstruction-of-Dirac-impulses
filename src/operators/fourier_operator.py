from typing import Callable, List

import numpy as np
import pyxu.info.ptype as pxt
from pyxu.util import view_as_real, view_as_complex

from src.operators.my_lin_op import MyLinOp


class FourierOperator(MyLinOp):

    def __init__(self, x: pxt.NDArray, w: pxt.NDArray, n_measurements: int, x_dim: int = 1):
        """
        Parameters
        ----------
        x : pxt.NDArray
            Position of the Diracs.
        w : pxt.NDArray
            Frequencies of the Fourier transform to measure.
        n_measurements : int
            Number of measurements.
        x_dim : int, optional
            Dimension of the position of the Diracs, by default 1.
        """
        self.x = x
        self.w = w
        self.n_measurements = n_measurements
        self.x_dim = x_dim
        self.scaling = np.max(np.abs(self.w))

        self.fourier = np.exp(-2j * np.pi * self._compute_outer(self.x))

        super().__init__(max(len(x), 1), (n_measurements, 2))

    def _compute_outer(self, x):
        return self.w @ x.reshape(-1, self.x_dim).T

    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        return view_as_real(self.fourier @ a)

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        # Equivalent to self.adjoint_function(y)(self.x)
        y = view_as_complex(y)
        return np.real(self.fourier.conj().T @ y)

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        y = view_as_complex(y)
        return lambda t: np.real(np.exp(2j * np.pi * self._compute_outer(t)).T @ y)

    def adjoint_function_grad(self, y: pxt.NDArray) -> Callable:
        y = view_as_complex(y)
        w = self.w.reshape(-1, 1, self.x_dim)
        return lambda t: np.real(2j * np.pi * w.T * np.exp(2j * np.pi * self._compute_outer(t)).T @ y).T.reshape(-1, self.x_dim)

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return FourierOperator(x, self.w, self.n_measurements, self.x_dim)

    def is_complex(self) -> bool:
        return True

    def get_DiffOperator(self) -> MyLinOp:
        return DiffFourierOperator(self.w, self.n_measurements, 2*len(self.x), self.x_dim)

    def get_scaling(self) -> pxt.Real:
        return self.scaling

    @staticmethod
    def get_RandomFourierOperator(x: pxt.NDArray, n_measurements: int, bounds: pxt.NDArray, x_dim: int = 1) -> MyLinOp:
        w = np.random.uniform(bounds[0], bounds[1], size=(n_measurements, x_dim))
        return FourierOperator(x, w, n_measurements, x_dim)

    @staticmethod
    def get_PeriodicFourierOperator(x: pxt.NDArray, n_measurements: int, fc: int) -> MyLinOp:
        w = np.arange(-fc, fc + 1)
        return FourierOperator(x, w, n_measurements)


class DiffFourierOperator(MyLinOp):

    def __init__(self, w: pxt.NDArray, n_measurements: int, input_size: int, x_dim: int = 1):
        """
        Parameters
        ----------
        w : pxt.NDArray
            Frequencies of the Fourier transform to measure.
        n_measurements : int
            Number of measurements.
        input_size : int
            Size of the input, used to set to dimension of the operator.
        x_dim : int, optional
            Dimension of the position of the Diracs, by default 1.
        """
        self.w = w
        self.n_measurements = n_measurements
        self.input_size = input_size
        self.x_dim = x_dim
        self.scaling = np.max(np.abs(self.w))

        self.fourier = lambda t: np.exp(-2j * np.pi * self._compute_outer(t))

        super().__init__((input_size, 1), (n_measurements, 2))

    def _compute_outer(self, x):
        return self.w @ x.reshape(-1, self.x_dim).T

    def apply(self, xa: pxt.NDArray) -> pxt.NDArray:
        a = np.split(xa, 1 + self.x_dim)[-1]
        x = xa[:-len(a)]
        return self.fourier(x) @ a

    def grad_x(self, xa: pxt.NDArray) -> pxt.NDArray:
        a = np.split(xa, 1 + self.x_dim)[-1].reshape(-1, 1)
        x = xa[:-len(a)]
        if self.x_dim == 1:
            return a * (self.fourier(x) * -2j * np.pi * self.w).T.conj()
        elif self.x_dim == 2:
            w1, w2 = self.w[:, 0].reshape(-1, 1), self.w[:, 1].reshape(-1, 1)

            f = self.fourier(x) * -2j * np.pi
            dx = a * (f * w1).T.conj()
            dy = a * (f * w2).T.conj()
            return np.stack([dx, dy], axis=1).reshape(-1, self.n_measurements)

    def grad_a(self, xa: pxt.NDArray) -> pxt.NDArray:
        a = np.split(xa, 1 + self.x_dim)[-1]
        x = xa[:-len(a)]
        return self.fourier(x).T.conj()

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        pass

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        y = view_as_complex(y)
        return lambda t: np.real(np.exp(2j * np.pi * self._compute_outer(t)).T @ y)

    def get_new_operator(self, x: pxt.NDArray) -> MyLinOp:
        return DiffFourierOperator(self.w, self.n_measurements, self.input_size, self.x_dim)

    def is_complex(self) -> bool:
        return True

    def get_scaling(self) -> pxt.Real:
        return self.scaling
