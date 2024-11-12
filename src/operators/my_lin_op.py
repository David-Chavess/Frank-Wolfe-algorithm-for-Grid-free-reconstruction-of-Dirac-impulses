from typing import Callable, Self

from pyxu.abc import LinOp
from pyxu.info import ptype as pxt


class MyLinOp(LinOp):
    def apply(self, a: pxt.NDArray) -> pxt.NDArray:
        pass

    def adjoint(self, y: pxt.NDArray) -> pxt.NDArray:
        pass

    def adjoint_function(self, y: pxt.NDArray) -> Callable:
        pass

    def _meta(self):
        pass

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        pass

    def get_new_operator(self, x: pxt.NDArray) -> Self:
        pass

    def is_complex(self) -> bool:
        pass

    def get_DiffOperator(self) -> Self:
        pass
