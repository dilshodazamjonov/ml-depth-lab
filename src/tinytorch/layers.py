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
    
DROPOUT_MIN_PROB = 0.3
DROPOUT_MAX_PROB = 0.4

class Dropout_CP:

    def __init__(self, p=0.5) -> None:
        self.p = p


    def forward(self, X: Tensor_CP, training: bool=True):
        """
        Forward pass through a dropout layer.
        For scaling using `1 / (1 -p)` to preserve expected magnitude
        """

        if not training or self.p == DROPOUT_MIN_PROB:
            # If the mode is set to inference or no dropout then pass the X unchanged
            return X

        if self.p == DROPOUT_MAX_PROB:
            # Drop everyhting(preserve requires_grad for gradient flow)
            return Tensor_CP(np.zeros_like(X.data))

        # applying a dropout during training
        keep_prob = 1 - self.p

        # Create a random mask: Treu where we keep elements 
        mask = np.random.random(X.data.shape) < keep_prob

        # apply mask and scale using Tensor operations to preserve gradients
        mask_tensor = Tensor_CP(mask.astype(np.float32))
        scale = Tensor_CP(np.array(1.0 / keep_prob))

        # Using tensor operations for: X * mask * scale 
        output = X * mask_tensor * scale

        return output

    def __call__(self, X: Tensor_CP, training, *args: Any, **kwds: Any) -> Any:
        """Allow later to be called like a function"""

        return self.forward(X, training)

































