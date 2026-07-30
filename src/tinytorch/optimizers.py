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
        raise NotImplementedError()

class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(self, params: list[Tensor_CP], lr=0.01, momentum=0.0, weight_decay=0.0) -> None:

        if momentum > 1.0 or momentum < 0.0:
            raise ValueError(f"Expected momentum to be in range (0.0, 1.0), got {momentum}")

        if lr <= 0:
            raise ValueError(f"Learning Rate should be greater than 0. Got {lr}")

        if weight_decay < 0:
            raise ValueError(f"Weight decay should be greater than or equal to 0, got {weight_decay}")
        
            
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

class Adam(Optimizer):

    def __init__(self, params: list[Tensor_CP], lr=0.001, betas = (0.9, 0.999), epsilon=1e-8, weight_decay= 0.0) -> None:

        beta1, beta2 = betas

        if lr <= 0:
            raise ValueError(f"Learning Rate should be greater than 0. Got {lr}")
                
        if weight_decay < 0:
            raise ValueError(f"Weight decay should be greater than or equal to 0, got {weight_decay}")

        if not 0 <= beta1 < 1:
            raise ValueError(f"Expected Beta 1 in range 0 and 1 got: {beta1}")
        
        if not 0 <= beta2 < 1:
            raise ValueError(f"Expected Beta 2 in range 0 and 1 got: {beta2}")

        if epsilon <= 0:
            raise ValueError(
                f"Epsilon should be greater than 0, got {epsilon}"
            )

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay        

        self.first_moment_buffers = {}
        self.second_moment_buffers = {}
        self.step_counts = {}

        super().__init__(params)

    def step(self) -> None:

        if not any(tensor.grad is not None for tensor in self.params):
            return

        for tensor in self.params:

            if tensor.grad is None:
                continue

            tensor_id = id(tensor)

            gradient = tensor.grad.copy()

            # Classical Adam: weight decay is added to the gradient
            if self.weight_decay != 0.0:
                gradient += self.weight_decay * tensor.data

            tensor_step = self.step_counts.get(tensor_id, 0) + 1
            self.step_counts[tensor_id] = tensor_step

            # Initialize this parameter's moment buffers
            if tensor_id not in self.first_moment_buffers:
                self.first_moment_buffers[tensor_id] = np.zeros_like(tensor.data)

            if tensor_id not in self.second_moment_buffers:
                self.second_moment_buffers[tensor_id] = np.zeros_like(tensor.data)

            old_first_moment = self.first_moment_buffers[tensor_id]
            old_second_moment = self.second_moment_buffers[tensor_id]

            updated_first_moment = (
                self.beta1 * old_first_moment 
                + (1 - self.beta1) * gradient 
            )

            updated_second_moment = (
                self.beta2 * old_second_moment
                + (1 - self.beta2) * gradient**2
            )

            self.first_moment_buffers[tensor_id] = updated_first_moment
            self.second_moment_buffers[tensor_id] = updated_second_moment

            # Calculating the bias-corrected moments
            corrected_first_moment = (
                updated_first_moment
                / (1 - self.beta1 ** tensor_step)
            )

            corrected_second_moment = (
                updated_second_moment
                / (1 - self.beta2 ** tensor_step)
            )

            tensor.data -= (
                self.lr * corrected_first_moment
                / (np.sqrt(corrected_second_moment) + self.epsilon)
            )


class AdamW(Adam):

    """
        Decoupled Adam - Weight decay does not accumulate inside the momentum or variance estimates.
    """

    def __init__(
        self,
        params: list[Tensor_CP],
        lr=0.001,
        betas=(0.9,0.999),
        epsilon=1e-8,
        weight_decay=0.01
    ) -> None:

        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            epsilon=epsilon,
            weight_decay=weight_decay
        )

    def step(self) -> None:

        for tensor in self.params:

            if tensor.grad is None:
                continue

            tensor_id = id(tensor)

            gradient = tensor.grad.copy()

            if self.weight_decay != 0.0:
                tensor.data *= 1 - self.lr * self.weight_decay

            # Increment only this parametr's update count
            tensor_step = self.step_counts.get(tensor_id, 0) + 1
            self.step_counts[tensor_id] = tensor_step

            # Initialize this parameter's moment buffers
            if tensor_id not in self.first_moment_buffers:
                self.first_moment_buffers[tensor_id] = np.zeros_like(tensor.data)

            if tensor_id not in self.second_moment_buffers:
                self.second_moment_buffers[tensor_id] = np.zeros_like(tensor.data)

            old_first_moment = self.first_moment_buffers[tensor_id]
            old_second_moment = self.second_moment_buffers[tensor_id]

            updated_first_moment = (
                self.beta1 * old_first_moment 
                + (1 - self.beta1) * gradient 
            )

            updated_second_moment = (
                self.beta2 * old_second_moment
                + (1 - self.beta2) * gradient**2
            )

            self.first_moment_buffers[tensor_id] = updated_first_moment
            self.second_moment_buffers[tensor_id] = updated_second_moment

            # Calculating the bias-corrected moments
            corrected_first_moment = (
                updated_first_moment
                / (1 - self.beta1 ** tensor_step)
            )

            corrected_second_moment = (
                updated_second_moment
                / (1 - self.beta2 ** tensor_step)
            )

            tensor.data -= (
                self.lr * corrected_first_moment
                / (np.sqrt(corrected_second_moment) + self.epsilon)
            )


        

