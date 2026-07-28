import numpy as np

from .tensor import Tensor_CP


class Optimizer:
    """A base class for all optimizers that will be built in this module"""
    def __init__(self, params: list[Tensor_CP]) -> None:

        # Validate Tensors and the list
        if not isinstance(params, list):
            raise TypeError(f"Expected params to be List got {type(params)}")

        for tensor in params:
            if not isinstance(tensor, Tensor_CP):
                raise TypeError(f"Expected params[..] to be {Tensor_CP} got {type(tensor)}")

        self.params = params

    def zero_grad(self) -> None:
        """Zero gradient should set tensor's gradient to none"""
        for tensor in self.params:
            tensor.zero_grad()

    def step(self) -> None:
        """Update parameters (implemented by subclasses"""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(self, params: list[Tensor_CP], lr=0.01, momentum=0.0, weight_decay=0.0) -> None:

        if momentum > 1.0 or momentum < 0.0:
            raise ValueError(f"Expected momentum to be in range (0.0, 1.0), got {momentum}")
            
        # For different behaviours if momentum exists or no 
        self.has_momentum = 0.0 != momentum
        self.weight_decay_enabled = weight_decay != 0.0

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

        self.momentum_buffers = {}

        super().__init__(params)

    def step(self) -> None:

        for tensor in self.params:
            if tensor.grad is None:
                continue

            gradient = tensor.grad.copy()

            if self.weight_decay_enabled:
                gradient = gradient + self.weight_decay * tensor.data

            if self.has_momentum:

                tensor_id = id(tensor)
                old_buffer = self.momentum_buffers.get(tensor_id)

                if old_buffer is None:
                    old_buffer = np.zeros_like(tensor.data)

                updated_buffer = (
                    self.momentum * old_buffer + gradient
                )

                self.momentum_buffers[tensor_id] = updated_buffer

                update_direction = updated_buffer

            else:
                update_direction = gradient
            
            tensor.data -= self.lr * update_direction
