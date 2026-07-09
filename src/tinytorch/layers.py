from typing import Any

from .tensor import Tensor_CP
import numpy as np


class Linear_CP:

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initializing a class with given parameters and attributes"""

        self.in_features = in_features
        self.out_features = out_features

        # LeCun-style initialization for stable gradients
        scale = np.sqrt(1/in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor_CP(weight_data)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor_CP(bias_data)
        else:
            self.bias = None
          
    def forward(self, X: Tensor_CP) -> Tensor_CP:
        """Forward pass through linear layer."""
        # Linear transformation: y = X * W

        output = X.matmul(self.weight)

        if self.bias is not None:
            output = output + self.bias

        return output

    def parameters(self) -> list:
        """Return all available parameters"""

        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, X: Tensor_CP, *args: Any, **kwds: Any) -> Any:
        """Allow layer to be called like a function"""

        return self.forward(X, *args, **kwds)

