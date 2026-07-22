from typing import Any

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

def _validate_data(data: Any):

    if isinstance(data, Tensor_CP):
        return data.data
    else: 
        return data


class AddBackward(Function):
    """For backward gradient computation of Addition"""
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
    """For backward gradient computation of Multiplication"""
    def apply(self, grad_output: NDArray) -> tuple[NDArray| None, NDArray | None]:
        left, right = self.saved_tensors

        left_gradient = None
        right_gradient = None

        if isinstance(left, Tensor_CP) and left.requires_grad:

            data_right = _validate_data(right)

            left_gradient = grad_output * data_right

            left_gradient = _sum_to_shape(left_gradient, left.shape)

        if isinstance(right, Tensor_CP) and right.requires_grad:

            data_left = _validate_data(left)

            right_gradient = grad_output * data_left

            right_gradient = _sum_to_shape(right_gradient, right.shape)
            
        return left_gradient, right_gradient


class MatMulBackward(Function):
    """Backward computation for 2D matrix multiplication only."""

    def apply(self, grad_output: NDArray) -> tuple[NDArray | None, NDArray|None]:
        left, right = self.saved_tensors

        expected_shape =  (left.shape[0], right.shape[1])

        if grad_output.shape != expected_shape:
            raise ValueError(
                f"Expected grad_output shape is {expected_shape}, got {grad_output.shape} instead"
            )

        grad_left, grad_right = None, None

        if isinstance(left, Tensor_CP) and left.requires_grad:

            grad_left = grad_output @ right.data.T

        if isinstance(right, Tensor_CP) and right.requires_grad:

            grad_right = left.data.T @ grad_output

        return grad_left, grad_right


class SubBackward(Function):
    """Backward computation for element-wise subtraction."""

    def apply(
        self,
        grad_output: NDArray
    ) -> tuple[NDArray | None, NDArray | None]:

        left, right = self.saved_tensors

        left_gradient = None
        right_gradient = None

        if isinstance(left, Tensor_CP) and left.requires_grad:

            left_gradient = _sum_to_shape(
                grad_output,
                left.shape
            )

        if isinstance(right, Tensor_CP) and right.requires_grad:

            right_gradient = _sum_to_shape(
                -grad_output,
                right.shape
            )

        return left_gradient, right_gradient

class SumBackward(Function):

    def __init__(
            self,
            tensor: Tensor_CP,
            axis=None,
            keepdims: bool = False
        ) -> None:

        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def apply(self, grad_output: NDArray) -> tuple[NDArray | None]:
        tensor, = self.saved_tensors
        
        if not tensor.requires_grad:
            return (None,)

        input_gradient = grad_output

        # restore the dimentions removed by the forward sum
        if self.axis is not None and not self.keepdims:

            input_gradient = np.expand_dims(
                input_gradient,
                axis=self.axis
            )

        # Give every original input element its gradient
        input_gradient = np.broadcast_to(
            input_gradient,
            tensor.shape
        )

        return (input_gradient,)

class MeanBackward(Function):

    def __init__(
        self,
        tensor: Tensor_CP,
        axis=None,
        keepdims: bool = False
    ) -> None:
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def apply(
        self,
        grad_output: NDArray
    ) -> tuple[NDArray | None]:

        tensor, = self.saved_tensors

        if not tensor.requires_grad:
            return (None,)

        if self.axis is None:
            reduced_count = tensor.size

        elif isinstance(self.axis, tuple):
            reduced_count = np.prod([
                tensor.shape[axis]
                for axis in self.axis
            ])

        else:
            # A single integer axis, including axis=0
            reduced_count = tensor.shape[self.axis]

        input_gradient = grad_output

        if self.axis is not None and not self.keepdims:
            input_gradient = np.expand_dims(
                input_gradient,
                axis=self.axis
            )

        input_gradient = np.broadcast_to(
            input_gradient,
            tensor.shape
        )

        input_gradient = input_gradient / reduced_count

        return (input_gradient,)



class DivBackward(Function):

    def apply(self, grad_output: NDArray) -> tuple[NDArray | None, NDArray | None]:
        left, right = self.saved_tensors

        left_gradient, right_gradient = None, None
        
        data_right = _validate_data(right)
        data_left = _validate_data(left)

        if isinstance(left, Tensor_CP) and left.requires_grad:
            left_gradient = grad_output / data_right

            left_gradient = _sum_to_shape(left_gradient, left.shape)

        if isinstance(right, Tensor_CP) and right.requires_grad:
            right_gradient = grad_output * (-data_left/data_right**2)

            right_gradient = _sum_to_shape(right_gradient, right.shape)

        return left_gradient, right_gradient

class ReshapeBackward(Function):
    """Backward computation for reshape."""

    def apply(
        self,
        grad_output: NDArray
    ) -> tuple[NDArray | None]:

        tensor, = self.saved_tensors

        if not tensor.requires_grad:
            return (None,)

        # Reshape grad_output back to tensor.shape
        input_gradient = grad_output.reshape(tensor.shape)

        return (input_gradient,)


class TransposeBackward(Function):
    """Backward computation for Numpy Transpose"""

    def apply(self, grad_output: NDArray) -> tuple[NDArray | None]:

        tensor, = self.saved_tensors

        if not tensor.requires_grad:
            return (None,)

        output = grad_output.T

        return (output,)