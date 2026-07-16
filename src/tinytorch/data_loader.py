from .tensor import Tensor_CP
from abc import ABC, abstractmethod

class Dataset(ABC):
    """To abstract the data for tensors."""  

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of samples in a given dataset"""

    @abstractmethod
    def __getitem__(self, idx: int):
        """Returns an item at a given index"""


class TensorDataset(Dataset):

    def __init__(self, *tensors: Tensor_CP) -> None:

        if len(tensors) < 1:
            raise ValueError("Expected tensor got an empty tuple")

        # Check if all passed elements are tensors
        for idx, tensor in enumerate(tensors):
            if not isinstance(tensor, Tensor_CP):
                raise TypeError(f"Expected an element with type Tensor_CP got {type(tensor)}")

            if len(tensor.shape) == 0: 
                raise ValueError(f"Got a scalar tensor at index: {idx}")
            
        first_dim = tensors[0].shape[0]

        # Dimention check
        for tensor in tensors:
            if tensor.shape[0] != first_dim:
                raise ValueError(f'Expected all tensors to be same first dimention, got: {tensor.shape}')        

        self.tensors = tensors

    def __len__(self) -> int:
        """Returning count of samples"""

        return len(self.tensors[0])

    def __getitem__(self, idx) -> tuple[Tensor_CP, ...]:
        """Getting an item with a specified index"""
        output = []

        for tensor in self.tensors:
            output.append(tensor[idx])

        return tuple(output)

    