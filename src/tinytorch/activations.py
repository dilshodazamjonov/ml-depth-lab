from .tensor import Tensor_CP
import numpy as np

class ReLU_CP:

    def forward(self, X: Tensor_CP) -> Tensor_CP:
        # formula to use : max(0, x)
        result  = np.maximum(0.00, X.data)
        return Tensor_CP(result)

    def __call__(self, X:Tensor_CP):
        return self.forward(X=X)

    def backward(self, grad:Tensor_CP):
        # Stub — autograd adds gradient computation later
        pass

class Sigmoid_CP:

    def forward(self, X: Tensor_CP) -> Tensor_CP:
        # Main implementation
        z = np.clip(X.data, -500, 500)
        result_data = np.zeros_like(z)

        # Positive mask for the values of X.data
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        # Negative mask for the values of X.data
        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor_CP(result_data)

    def __call__(self, X:Tensor_CP) -> Tensor_CP:
        return self.forward(X=X)

    def backward(self, grad: Tensor_CP):
        # Stub — autograd adds gradient computation later
        pass

class Tanh_CP:

    def forward(self, X: Tensor_CP) -> Tensor_CP:
        # Main implementation
        result = np.tanh(X.data)

        return Tensor_CP(result)

    def __call__(self, X:Tensor_CP) -> Tensor_CP:
        return self.forward(X=X)

    def backward(self, grad: Tensor_CP):
        # Stub — autograd adds gradient computation later
        pass

class GeLU_CP:
    def forward(self, X: Tensor_CP) -> Tensor_CP:
        # Sigmoid approximation of GELU: x * σ(1.702x)
        return Sigmoid_CP()(X * 1.702) * X
    def __call__(self, X: Tensor_CP) -> Tensor_CP:
        return self.forward(X=X)
    def backward(self, grad: Tensor_CP):
        # Stub — autograd adds gradient computation later
        pass

class Softmax_CP():

    def forward(self, X: Tensor_CP, dim: int = -1) -> Tensor_CP:
        """Apply softmax activation along the given dimension"""
        # Numerical stability: substract max to prevent overflow
        x_max_data = np.max(X.data, axis=dim, keepdims=True)
        x_max = Tensor_CP(x_max_data)
        x_shifted = X - x_max

        # compute exponensials 
        exp_values = Tensor_CP(np.exp(x_shifted.data))

        # sum along the dimension

        exp_sum_data = np.sum(exp_values.data, axis=dim, keepdims=True)
        exp_sum = Tensor_CP(exp_sum_data)

        result = exp_values / exp_sum
        return result

    def __call__(self, X: Tensor_CP, dim: int = -1) -> Tensor_CP:
        return self.forward(X=X, dim=dim)



        