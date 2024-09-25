from typing import Self, Callable

import numpy as np
import pyxu.info.ptype as pxt
import pyxu.util as pxu
from pyxu.operator import FFTConvolve

from src.operators.my_lin_op import MyLinOp


class ConvOperator(MyLinOp):

    def __init__(self, spikes: pxt.NDArray, grid: pxt.NDArray, n_measurements: int):
        self.spikes = spikes
        self.grid = grid
        self.n_measurements = n_measurements

        sigma = 1

        kernel = np.exp(-1 * spikes**2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

        print(kernel)

        self.convolve = FFTConvolve(n_measurements, kernel, grid)

        super().__init__(len(x), n_measurements)

    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        return self.convolve(a.ravel())

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        return self.convolve.adjoint(y.ravel())