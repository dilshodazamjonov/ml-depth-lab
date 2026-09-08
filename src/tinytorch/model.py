"""
Model.py is the final module inside of the foundation tier.

Things to be implemented: 
1. CosineScheduler - learning rate annealing 
2. clip_grad_norm() - Global gradient clipping for stability 
3. Trainer class with methods: 
    train_epoch() -  complete training loop with scheduling
    evaluate() - Evaluating the model without the gradient update
    save/load_checkpoint() - Training state persistance
"""
import numpy as np

from .tensor import Tensor_CP


class CosineSchedule:
    """
    Given the current epoch, what learning rate should i use?

    gets: max_lr - upper bound, min_lr - lower bound, total_epoch - how many epochs should decay span
    """

    def __init__(self, max_lr, min_lr, total_epochs) -> None:

        if total_epochs < 0:
            raise ValueError(f"total epochs should be bigger than 0, got: {total_epochs}")
    
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
    

    def get_lr(self, epoch: int) -> float:

        if epoch < 0:
            raise ValueError(f"Expected epoch to be positive integer, got: {epoch}")

        if epoch >= self.total_epochs:
            return self.min_lr

        lr: float = self.min_lr + (self.max_lr - self.min_lr) * ((1 + float(np.cos(np.pi*epoch/self.total_epochs))) / 2)
        return lr


def clip_grad_norm(parameters: list[Tensor_CP], max_norm: float=1.0) -> float:
    """
    Helper to normalize the exploding gradients by clipping the gradient that is concentrated in a single vector
    """
    grad_sum = 0.0

    for param in parameters:

        if not isinstance(param, Tensor_CP):
            raise TypeError(f"Expected Tensor_CP type parameters, got {type(param)}")

        if param.grad is None:
            continue

        grad_sum += np.sum(param.grad ** 2)


    grad_norm = np.sqrt(grad_sum)

    if grad_norm > max_norm:

        scale = max_norm / grad_norm

        for param in parameters:

            if param.grad is not None:
                param.grad *= scale

    return grad_norm


        

    