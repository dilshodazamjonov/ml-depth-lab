from typing import Any, Protocol

from .tensor import Tensor_CP
import numpy as np

class Layer(Protocol):
    """A layer class to avoid type errors inside of a class"""
    def forward(self, X: Tensor_CP) -> Tensor_CP:
        ...

    def parameters(self) -> list:
        ... 

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

    def parameters(self) -> list[Tensor_CP]:
        """Return all available parameters"""

        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, X: Tensor_CP, *args: Any, **kwds: Any) -> Any:
        """Allow layer to be called like a function"""

        return self.forward(X, *args, **kwds)


class Dropout_CP:

    def __init__(self, p=0.5) -> None:
        
        if 0 <= p < 1:
            self.p = p
        else:
            raise ValueError(f"Passed parameter p outside of probability range. Expected between [0, 1) got {p}")

    def forward(self, X: Tensor_CP, training: bool=True):
        """
        Forward pass through a dropout layer.
        For scaling using `1 / (1 -p)` to preserve expected magnitude
        """

        if not training or self.p == 0:
            # If the mode is set to inference or no dropout then pass the X unchanged
            return X

        # applying a dropout during training
        keep_prob = 1 - self.p

        # Create a random mask: True where we keep elements 
        mask = np.random.random(X.data.shape) < keep_prob

        # apply mask and scale using Tensor operations to preserve gradients
        mask_tensor = Tensor_CP(mask.astype(np.float32))
        scale = Tensor_CP(np.array(1.0 / keep_prob))

        # Using tensor operations for: X * mask * scale 
        output = X * mask_tensor * scale

        return output

    def parameters(self) -> list:
        return []
    
    def __call__(self, X: Tensor_CP, training: bool = True, *args: Any, **kwds: Any) -> Any:
        """Allow layer to be called like a function"""

        return self.forward(X, training)

class Sequential:
    """Container that chains layers sequentially."""
    def __init__(self, *layers: Layer) -> None:
        """Initialize which layers to chain together"""

        if len(layers) == 1 and isinstance(layers[0], (tuple, list)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self, X: Tensor_CP) -> Tensor_CP:
        """Run every layer with its forward method."""

        for layer  in self.layers:
            X = layer.forward(X)

        return X

    def parameters(self) -> list[Tensor_CP]:
        """Return the parameters of a Sequential layers"""
        params = []
        
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
    
    def __call__(self, X: Tensor_CP, *args: Any, **kwds: Any) -> Any:
        
        return self.forward(X=X)
























