import numpy as np
from typing import Any, Tuple

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
        """Return an item at a given index"""

        return Tensor_CP(self.data[idx])
      
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

    # ========================= MatMUL, Transpose, Reshape =========================

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


    def transpose(self):

        if self.data.ndim == 2:

            return Tensor_CP(self.data.transpose())
            
        else:
            raise ValueError(f"Currently only up to 2 dimaetions are supported, received {self.data.ndim}")


    def reshape(self, shape: Tuple[int, ...]):
        """
        Needed checks: 
            1. shape sizes multiplication == size of the self.data.shape
            2. Maximum of one -1 dimension.reshape(-1, 3) 
            3. -1 only from negatives and only once 
        """
        size = self.data.size


        if len(shape) < 1:
            raise ValueError(f"Shape expected as Tuple type with length >= 1 got {len(shape)}")
        
        if sum([1 for i in shape if i < -1]) > 0:
            raise ValueError(f"Negative shape is not supported. Got shape: {shape}")
        
        if shape.count(-1) == 1:
            known_dims = [i for i in shape if i > -1]
            

            idx = shape.index(-1) # index at which 1 should be inserted 

            prod = 1
            for i in known_dims:
                prod *= i

            try: 
               remainder = size % prod

            except ZeroDivisionError:
                raise ZeroDivisionError(f"Devision by zero encountered, passed shape: {shape}")

            if remainder != 0:
                raise ValueError(f'Error happened while dealing with -1, shape: {shape}, cannot be reshaped to {self.data.shape}')
            
            last_sh = int(size / prod)
            known_dims.insert(idx, last_sh)
            return Tensor_CP(self.data.reshape(tuple(known_dims)))
        
        elif -1 not in shape:            
            prod = 1

            for i in shape: 
                prod *= i 
            
            if prod != size:
                raise ValueError(f"Cannot reshape a tensor with shape: {self.data.shape} to shape: {shape}")
            
            return Tensor_CP(self.data.reshape(shape))

        
        else:
            raise ValueError(f"Expected only one -1 value got 2 or more in provided shape: {shape}")
        
    # ======================= Axis Semantics =========================

    def sum(self, axis=None, keepdims=False):
        """
        Sum of Tensor along a specified axis
        
        keepdips = True preserves the reduced dimention as size 1, usefull for broadcasting
        """
            
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor_CP(result)
    
    def mean(self, axis=None, keepdims=False):
        """Mean of Tensor accross all the elements"""
        
        return Tensor_CP(np.mean(self.data, axis=axis, keepdims=keepdims))
