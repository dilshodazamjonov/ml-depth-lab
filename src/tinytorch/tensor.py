import numpy as np
from typing import Any

class Tensor_CP:
    """
    Manually built class imitating PyTorch class - Tensor.
    """

    # ========================= Basic Methods =========================
    def __init__(self, data: Any) -> None:

        self.data = np.array(data, dtype=np.float32)

        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    # ========================= Basic arithmetics ========================= 
    def __add__(self, other):
        """
        Add two Tensor elements with broadcasting support  

        Ex: 
            1. (1, 3) + (3,) => (3, 3)
            2. (3, 1) + (3, 1) => (3, 1)
        """
        if isinstance(other, Tensor_CP):
            # Unwrap, compute with NumPy, re-wrap
            return Tensor_CP(self.data + other.data)

        else:
            # Scalar Broadcast
            return Tensor_CP(self.data + other)

    def __sub__(self, other):
        """
        Subtract one tensor from another with broadcasting support 
        """

        if isinstance(other, Tensor_CP):
            return Tensor_CP(self.data - other.data)

        else:
            return Tensor_CP(self.data - other)

    def __mul__(self, other):
        """
            Multiply one tensor by another with broadcasting support 
        """
        try:
            if isinstance(other, Tensor_CP):
                return Tensor_CP(self.data * other.data)

            else:
                return Tensor_CP(self.data * other)
        except ValueError as e:
            raise ValueError(f"Incompatibli shape for broadcasting: {e}")

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
            if isinstance(other, Tensor_CP):
                return Tensor_CP(self.data / other.data)
            else:
                return Tensor_CP(self.data / other)
        
        except ValueError as e:
            raise ValueError(f"Sizes of a matrices does not match: {e}")

    # ========================= MatMUL =========================

    def matmul(self, other):

        if isinstance(other, Tensor_CP):

            if self.data.shape[-1] != other.data.shape[0]:
                raise ValueError(f"Given matrices with innner parts {self.data.shape} and {other.shape} do not match")

            c_shape = tuple([self.data.shape[0], other.data.shape[-1]]) 
            C = np.zeros(c_shape) 

            # logic of computation

            for i in range(self.data.shape[0]):
                for j in range(other.data.shape[1]):
                    for k in range(other.data.shape[0]):
                        C[i][j] += self.data[i][k] * other.data[k][j]
                 
            return Tensor_CP(data=C)

        else:
            raise ValueError(f"Expected tensor-like object got: {type(other)}")
            