from typing import Any

import numpy as np

from .autograd import (
    AddBackward,
    DivBackward,
    GetItemBackward,
    MatMulBackward,
    MeanBackward,
    MulBackward,
    ReshapeBackward,
    SubBackward,
    SumBackward,
    TransposeBackward,
)


class Tensor_CP:
    """
    Manually built class imitating PyTorch class - Tensor.
    """

    # ========================= Basic Methods =========================
    def __init__(self, data: Any, requires_grad: bool = False) -> None:

        self.data = np.array(data, dtype=np.float32)

        if type(requires_grad) is not bool:
            raise TypeError(f"Expected boolean type for requires_grad, got {type(requires_grad)}")
        
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __len__(self):
        """Returns the size of tensor's first dimention"""

        return len(self.data)
    
    def __getitem__(self, idx) -> "Tensor_CP":
        """Return an indexed tensor while preserving the computation graph."""

        output = Tensor_CP(
            self.data[idx],
            requires_grad=self.requires_grad
        )

        if self.requires_grad:
            output._grad_fn = GetItemBackward(self, idx)

        return output
      
    # ========================= Basic arithmetics ========================= 
    def __add__(self, other):
        """
        Add two Tensor elements with broadcasting support  

        Ex: 
            1. (1, 3) + (3,) => (3, 3)
            2. (3, 1) + (3, 1) => (3, 1)
        """
        other_data = other.data if isinstance(other, Tensor_CP) else other

        requires_grad = (
            self.requires_grad
            or (
                isinstance(other, Tensor_CP)
                and other.requires_grad
            )
        )

        output = Tensor_CP(self.data + other_data, requires_grad=requires_grad)

        if requires_grad:
            output._grad_fn = AddBackward(self, other)

        return output

    def __radd__(self, other):
        """Handles right-side addition"""
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract one tensor from another with broadcasting support 
        """
        other_data = other.data if isinstance(other, Tensor_CP) else other

        requires_grad = (
            self.requires_grad
            or (
                isinstance(other, Tensor_CP)
                and other.requires_grad
            )
        )

        output = Tensor_CP(self.data - other_data, requires_grad=requires_grad)

        if requires_grad:
            output._grad_fn = SubBackward(self, other)

        return output

    def __rsub__(self, other):
        other_data = other.data if isinstance(other, Tensor_CP) else other

        requires_grad = (
            self.requires_grad
            or (isinstance(other, Tensor_CP) and other.requires_grad)
        )

        output = Tensor_CP(
            other_data - self.data,
            requires_grad=requires_grad
        )

        if requires_grad:
            output._grad_fn = SubBackward(other, self)

        return output


    def __mul__(self, other):
        """
            Multiply one tensor by another with broadcasting support 
        """
        try:
            other_data = other.data if isinstance(other, Tensor_CP) else other

            requires_grad = (
                self.requires_grad
                or (
                    isinstance(other, Tensor_CP)
                    and other.requires_grad
                )
            )

            output = Tensor_CP(self.data * other_data, requires_grad=requires_grad)

            if requires_grad:
                output._grad_fn = MulBackward(self, other)

            return output
        
        except ValueError as e:
            raise ValueError(f"Incompatible shape for broadcasting: {e}")

    def __rmul__(self, other):
        """
        Handles right-side multiplication (e.g., scalar * Tensor_CP).
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Handles the devision by one and other element
        """

        try:
            other_data = other.data if isinstance(other, Tensor_CP) else other

            requires_grad = (
                self.requires_grad
                or (
                    isinstance(other, Tensor_CP)
                    and other.requires_grad
                )
            )

            output = Tensor_CP(self.data / other_data, requires_grad=requires_grad)

            if requires_grad:
                output._grad_fn = DivBackward(self, other)

            return output
        
        except ValueError as e:
            raise ValueError(f"Sizes of a matrices does not match: {e}")

    def __rtruediv__(self, other):
        other_data = other.data if isinstance(other, Tensor_CP) else other

        requires_grad = (
            self.requires_grad
            or (isinstance(other, Tensor_CP) and other.requires_grad)
        )

        output = Tensor_CP(
            other_data / self.data,
            requires_grad=requires_grad
        )

        if requires_grad:
            output._grad_fn = DivBackward(other, self)

        return output

    # ========================= MatMUL, Transpose, Reshape =========================

    def matmul(self, other) -> "Tensor_CP":

        other_data = other.data if isinstance(other, Tensor_CP) else np.asarray(other, dtype=np.float32)

        if self.data.ndim != 2 or other_data.ndim != 2:
            raise ValueError(
                "matmul() currently supports only two-dimensional inputs"
            )

        if self.data.shape[-1] != other_data.shape[0]:
            raise ValueError(f"Given matrices with innner parts {self.data.shape} and {other.shape} do not match")


        c_shape = (self.data.shape[0], other_data.shape[-1])
        C = np.zeros(c_shape, dtype=np.float32) 

        # logic of computation

        for i in range(self.data.shape[0]):
            for j in range(other_data.shape[1]):
                for k in range(other_data.shape[0]):
                    C[i][j] += self.data[i][k] * other_data[k][j]

        # requires gradient logic 
        requires_grad = (
            self.requires_grad
            or (isinstance(other, Tensor_CP)
                and other.requires_grad
            )
        )

        output = Tensor_CP(C, requires_grad=requires_grad)

        if requires_grad:
            output._grad_fn = MatMulBackward(self, other)
                
        return output


    def transpose(self):

        if self.data.ndim != 2:

            raise ValueError(
                f"Only 2D transpose is currently supported, "
                f"received {self.data.ndim} dimensions"
            )
        
        output = Tensor_CP(self.data.T, requires_grad=self.requires_grad)

        if self.requires_grad:
            output._grad_fn = TransposeBackward(self)

        return output
    
    def reshape(self, shape: tuple[int, ...]):
        """
        Needed checks: 
            1. instance of shape shold be tuple,
            2. len(shape) > 1 
            3. check if all passed shapes are positive integers
            4. Reject Booleans too.
            5. check for the shape if all are positive
            6. if more than 1 element is -1 inside of shape -> error
        """

        if not isinstance(shape, tuple):
            raise TypeError(f"Expected tuple type shape, got {type(shape)}")

        if len(shape) < 1: 
            raise ValueError(f"Expected length of shape bigger than 1 got: {len(shape)}")

        if any(
            isinstance(dim, (bool, np.bool_))
            or not isinstance(dim, (int, np.integer))
            for dim in shape
        ):
            raise TypeError("Every reshape dimension must be an integer")

        if any(
            not isinstance(dim, (int, np.integer))
            for dim in shape
        ):
            raise TypeError("Expected elements to be integers detected non-numeric or not integer values")

        shape = tuple([int(dim) for dim in shape])

        if any(dim < -1 for dim in shape):
            raise ValueError(f"Dimentions smaller than -1 are not supported: {shape}")

        if shape.count(-1) > 1:
            raise ValueError(f"Only one inferred dimention is allowed, got {shape.count(-1)}")

        try:
            reshaped_data = self.data.reshape(shape)
        except ValueError as e:
            raise ValueError(
                f"Cannot shape tensor with shape {self.shape} and size {self.size} to {shape}: {e}"
            ) from e
        
        output = Tensor_CP(reshaped_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            output._grad_fn = ReshapeBackward(self)

        return output

    # ======================= Axis Semantics =========================

    def sum(self, axis=None, keepdims=False):
        """
        Sum of Tensor along a specified axis
        
        keepdips = True preserves the reduced dimention as size 1, usefull for broadcasting
        """

        result = np.sum(self.data, axis=axis, keepdims=keepdims)

        output = Tensor_CP(result, requires_grad=self.requires_grad)
        # pass the Tensor_CP instance (self) to the backward function, not the raw ndarray
        if self.requires_grad:
            output._grad_fn = SumBackward(tensor=self, axis=axis, keepdims=keepdims)

        return output
    
    def mean(self, axis=None, keepdims=False):
        """Mean of Tensor accross all the elements"""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)

        output = Tensor_CP(result, requires_grad=self.requires_grad)

        if self.requires_grad:
            output._grad_fn = MeanBackward(self, axis=axis, keepdims=keepdims)

        return output


    def backward(self, gradient=None):
        """Compute gradients via backpropagation"""

        if not self.requires_grad:
            return

        # Initialize gradient for scalar outputs 
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError("backward() requires gradient for non-scalar tensors")

        # Accumulate gradient (vectorized NumPy operation)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        self.grad += gradient

        # Propagate to parent tensors 
        if hasattr(self, '_grad_fn') and self._grad_fn is not None:
            grads = self._grad_fn.apply(gradient) # Compute input gradients using vectorized ops

            for tensor, grad in zip(self._grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor_CP) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad) # Recursive call

            