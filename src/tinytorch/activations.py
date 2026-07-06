from .tensor import Tensor_CP

class ReLU:

    def forward(self, X: Tensor_CP):
        # formula to use : max(0, x)
        pass

    def __call__(self, X:Tensor_CP):
        return self.forward(X=X)

    def backward(self, grad:Tensor_CP):
        # Stub — autograd adds gradient computation later
        pass