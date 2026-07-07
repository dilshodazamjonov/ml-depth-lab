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