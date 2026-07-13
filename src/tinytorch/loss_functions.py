from .tensor import Tensor_CP
import numpy as np 


class MSELoss_CP:

    def forward(self, predictions: Tensor_CP, targets: Tensor_CP) -> Tensor_CP:
        """Mean squared Error Loss, shows how much prediction differs from target."""

        # Edge case: shapes should be equal and should be Tensor_CP type
        if not isinstance(predictions, Tensor_CP) or not isinstance(targets, Tensor_CP):
            raise TypeError(f"Expected Tensor_CP type, got {type(predictions)} and {type(targets)}")
        
        if predictions.shape != targets.shape:
            raise ValueError(f"Expected equal shape Tensors got: Prediction shape: {predictions.shape} and Target shape: {targets.shape}")

        num_elements = predictions.size

        if num_elements == 0: 
            raise ValueError('Got a tensor with 0 elements')

        output = Tensor_CP(np.mean((predictions.data - targets.data)**2))

        return output

    def __call__(self, predictions: Tensor_CP, targets: Tensor_CP) -> Tensor_CP:
        """To enable calling a class like a function"""
        return self.forward(predictions, targets)

class CrossEntropy:

    def log_softmax(self, X: Tensor_CP, dim: int = -1) -> Tensor_CP:
        """Compute Log-softmax with numerical stability"""

        # Step 1. Find max along dimention for numerical stability
        X_max = np.max(X.data, axis=dim, keepdims=True)

        # Step 2. Substract max to prevent overflow
        shifted = X.data - X_max

        # Step 3. Compute the log(sum(exp(shifted)))
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

        result = shifted - log_sum_exp

        return Tensor_CP(result)

    def forward(self, logits: Tensor_CP, targets: Tensor_CP) -> Tensor_CP:
        """Compute cross-entropy loss between logits and target class indices"""

        log_probs = self.log_softmax(logits)

        # Select log probabilities for correct classes
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        # Selecting correct class log-probabilities using advanced indexing 
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        cross_entropy = -np.mean(selected_log_probs)

        return Tensor_CP(cross_entropy)
    