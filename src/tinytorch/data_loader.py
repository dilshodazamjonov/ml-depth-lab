from .tensor import Tensor_CP
from abc import ABC, abstractmethod
import numpy as np
import random


class Dataset(ABC):
    """To abstract the data for tensors.""" 

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of samples in a given dataset"""

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Tensor_CP, ...]:
        """Returns an item at a given index"""


class TensorDataset(Dataset):
    """To store Tensors in-memory"""

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

class DataLoader:
    """Prepare batches, shuffle data and enabl iteration"""
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False) -> None:

        if not isinstance(dataset, Dataset): 
            raise TypeError(f"Expected {Dataset} type dataset, got {type(dataset)}")
        
        if type(batch_size) is not int:
            raise TypeError(f'Expected batch size as integer, got: {type(batch_size)}')

        if batch_size <= 0: 
            raise ValueError(f"Batch size should be greater than 0 got: {batch_size}")

        if type(shuffle) is not bool:
            raise TypeError(f'Expected shuffle as boolean, got: {type(shuffle)}')
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _collate_batch(self, batch: list[tuple[Tensor_CP, ...]]) -> tuple[Tensor_CP, ...]:
        """Callate individual samples into batch tensors."""

        if len(batch) == 0:
            return ()

        # Determine number of tensors per sample
        num_tensors = len(batch[0])

        # Group tensors by position
        batched_tensors = []

        for tensor_idx in range(num_tensors):
            # Extract all the tensors from that postion
            tensor_list = [sample[tensor_idx] for sample in batch]

            # Stack into batch tensor
            batched_data = np.stack(
                [tensor.data for tensor in tensor_list], axis=0
            )
            batched_tensors.append(Tensor_CP(batched_data))

        return tuple(batched_tensors)

    def __len__(self) -> int:
        """Returns the number of batches"""

        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
            
    def __iter__(self):
        """Returns an iterator over batches"""

        indices = list(range(len(self.dataset)))

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):

            # breakdown of slicing from i till i + self.batch_size for example: start at 0 -> first batch indices are from 0 till 5(batch size )
            batch_indices = indices[i:i+self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # Collate the batch
            yield self._collate_batch(batch)
        
