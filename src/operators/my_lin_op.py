from typing import Callable, Self

from pyxu.abc import LinOp
from pyxu.info import ptype as pxt


class MyLinOp(LinOp):
    """A custom linear operator class that inherits from LinOp."""
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
        """Construct a new operator with the same properties as the current operator for different positions x"""
        pass

    def is_complex(self) -> bool:
        """Return True if the operator is complex-valued"""
        pass

    def get_DiffOperator(self) -> Self:
        """Return the differential operator of the current operator used for the sliding step optimization"""
        pass

    def get_scaling(self) -> pxt.Real:
        """Return a scaling factor of the operator used to create grids"""
        pass
