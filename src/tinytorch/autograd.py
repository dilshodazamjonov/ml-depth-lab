from numpy.typing import NDArray
from .tensor import Tensor_CP
import numpy as np

class Function:
    """Base class for Autograd classes"""
    def __init__(self, *operands) -> None:
        self.saved_tensors = operands

    def apply(self, grad_output: NDArray):
        raise NotImplementedError("No derivative rule yet.")


# Helper functions

def _sum_to_shape(gradient: NDArray, target_shape: tuple[int, ...]) -> NDArray:

    """
    Fixing the broadcasting so gradient and target shape match
        [1, 1, 1
        1, 1, 1] becomes in b.grad -> [2, 2, 2]
    """

    reduced_gradient = gradient

    while reduced_gradient.ndim > len(target_shape):

        reduced_gradient =  reduced_gradient.sum(axis=0)

    for axis, dimention_size in enumerate(target_shape):
        if dimention_size == 1:
            reduced_gradient = reduced_gradient.sum(
                axis=axis,
                keepdims=True
            )

    return reduced_gradient


class AddBackward(Function):

    def apply(self, grad_output: NDArray) -> tuple[NDArray| None, NDArray | None]:
        gradients = []

        for operand in self.saved_tensors:
            if isinstance(operand, Tensor_CP) and operand.requires_grad:
                operand_gradients = _sum_to_shape(
                    grad_output, 
                    operand.shape
                )
                gradients.append(operand_gradients)
            else:
                gradients.append(None)

        return tuple(gradients)

class MulBackward(Function):

    def apply(self, grad_output: NDArray) -> tuple[NDArray| None, NDArray | None]:
        left, right = self.saved_tensors

        left_gradient = None
        right_gradient = None

        if isinstance(left, Tensor_CP) and left.requires_grad:

            if isinstance(right, Tensor_CP):
                data_right = right.data
            else:
                data_right = right

            left_gradient = grad_output * data_right

            left_gradient = _sum_to_shape(left_gradient, left.shape)

        if isinstance(right, Tensor_CP) and right.requires_grad:

            if isinstance(left, Tensor_CP):
                data_left = left.data
            else:
                data_left = left

            right_gradient = grad_output * data_left

            right_gradient = _sum_to_shape(right_gradient, right.shape)
            
        return left_gradient, right_gradient


print(_sum_to_shape(

    np.ones((2, 3)),
    (3,)
).shape == (3,)
)