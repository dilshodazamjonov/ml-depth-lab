from .tensor import Tensor_CP
from abc import ABC, abstractmethod
import numpy as np

import random

from numpy.typing import NDArray

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
        

class RandomHorizontalFlip:
    """
    Flips image given the value of p

    following:
        p=0.0 → never flip
        p=0.5 → approximately 50% of calls flip
        p=1.0 → always flip
    """
    def __init__(self, p: int | float = 0.5) -> None:

        if not isinstance(p, (float, int)):
            raise TypeError(f"Expected float, int type value for p, got {type(p)} instead")

        if isinstance(p, bool):
            raise TypeError(f"got {bool} type value for p")

        if not 0 <= p <= 1:
            # p must be between 0 and 1
            raise ValueError(f"Expected p value between 0 and 1 got: {p}")

        self.p = p

    def __call__(self, image: Tensor_CP | NDArray) -> NDArray | Tensor_CP:
        """Enabled transforming like a function"""

        r = random.random()

        if isinstance(image, np.ndarray):
            data = image
            is_tensor = False

        elif isinstance(image, Tensor_CP):
            data = image.data
            is_tensor = True

        else:
            raise TypeError(f"Expected {NDArray} or {Tensor_CP} got: {type(image)}")

        # Identifying the dimentions and axis
        if data.ndim == 2:
            axis = -1

        elif data.ndim == 3 and data.shape[0] <= 4:
            axis = -1

        elif data.ndim == 3 and data.shape[0] > 4:
            axis = -2

        else:
            raise ValueError(f"Invalid shapes or dimentions: got dimention as {data.ndim} and shape as {data.shape}")


        if r < self.p:

            flipped_data = np.flip(data, axis=axis).copy()

            if is_tensor:
                return Tensor_CP(flipped_data)

            else:
                return flipped_data

        return image


def _pad_image(data: NDArray, padding: int) -> NDArray : 
    """Helper function that adds padding to a image"""

    if type(padding) is bool:
        raise TypeError('Got boolean, expected positive integer for the padding parameter')

    if not isinstance(padding, int):
        raise TypeError(f'Padding parameter is expected to be a type integer got {type(padding)} instead')

    if padding < 0:
        raise ValueError(f'Expected padding value to be positive integer got: {padding} instead')

    if not isinstance(data, np.ndarray):
        raise TypeError(f'Expected data to be {np.ndarray} got {type(data)}')

    if data.ndim not in (2, 3):
        raise ValueError(f"Expected an input data to be 2D or 3D got {data.ndim} instead")

    if data.ndim == 2: 
        # (H, W)
        pad_width = (
            (padding, padding),
            (padding, padding)
        )

    elif data.shape[0] <= 4:
        # (C, H, W)
        pad_width = (
            (0, 0),
            (padding, padding),
            (padding, padding)
        )

    else:
        # (H, W, C)
        pad_width = (
            (padding, padding),
            (padding, padding),
            (0, 0)
        )


    output = np.pad(data, 
                    pad_width=pad_width,
                    mode='constant',
                    constant_values=0
                )
    return output

def _random_crop_region(
    padded_h: int,
    padded_w: int,
    target_h: int,
    target_w: int
) -> tuple[int, int]:   
    """
    Sample a valid random top-left coordinate for cropping an image.

    The returned coordinates ensure that a crop of shape
    (target_h, target_w) fits completely inside an image of shape
    (padded_h, padded_w).

    Args:
        padded_h: Height of the padded image.
        padded_w: Width of the padded image.
        target_h: Height of the desired crop.
        target_w: Width of the desired crop.

    Returns:
        A tuple (top, left), representing the starting row and column
        of the crop.

    Raises:
        TypeError: If any dimension is not an integer.
        ValueError: If any dimension is non-positive or the target crop
            is larger than the padded image.
    """
    dimensions = (padded_h, padded_w, target_h, target_w)
    
    if any(type(dim) is not int for dim in dimensions):
        raise TypeError(
                f"Expected integer for dimensions, got: "
                f"{{'padded_h': {type(padded_h).__name__}, "
                f"'padded_w': {type(padded_w).__name__}, "
                f"'target_h': {type(target_h).__name__}, "
                f"'target_w': {type(target_w).__name__}}}"
            )
    
    if any(dim <= 0 for dim in [padded_h, padded_w, target_h, target_w]):
        raise ValueError("Expected positive integer for dimentions got Negative value")
    
    max_top = padded_h - target_h
    max_left = padded_w - target_w

    if any(top_left < 0 for top_left in [max_top, max_left]):
        raise ValueError("Got target height or width > padded height and width")

    top = random.randint(0, max_top)
    left = random.randint(0, max_left)

    return top, left


class RandomCrop:
    """Identify the random size and crop the image"""

    def __init__(self, size: int | tuple[int, int], padding: int = 4) -> None:
        """
        Size can be a tuple: (28, 32) or an integer 20 -> size = (20, 20)
        """   

        # Checks for size dtypes
        if type(size) is int:  
            if size > 0:
                size = (size, size)
            else:
                raise ValueError('size should be a positive integer')

        elif isinstance(size, tuple):
            if len(size) == 2:
                if all(type(num) is int for num in size):

                    if any(num <= 0 for num in size):
                        raise ValueError("Got negative integers inside of the tuple")
                    
                else:
                    raise TypeError("Got tuple with elements not integer")
                
            else:
                raise ValueError(f"Expected tuple with length 2 got {len(size)}")
                
        else:
            raise TypeError(f"Expected parameter dtype as {int} or {tuple} got {type(size)}")

        # Checks for padding types
        if type(padding) is not int:
            raise TypeError(f'Padding parameter is expected to be a type integer got {type(padding)} instead')

        if padding < 0:
            raise ValueError(f'Expected padding value to be positive integer got: {padding} instead')
        

        self.size = size
        self.padding = padding
    
    def __call__(self, image: Tensor_CP | NDArray) -> NDArray | Tensor_CP:
        
        if isinstance(image, Tensor_CP):
            data = image.data
            is_tensor = True

        elif isinstance(image, np.ndarray):
            data = image
            is_tensor = False
        
        else:
            raise TypeError(f'Expected image as type of {NDArray} or {Tensor_CP} got {type(image)}')

        
        padded = _pad_image(data, self.padding)
        target_h, target_w = self.size

        if data.ndim == 2:
            image_format = "HW"
            padded_h, padded_w = padded.shape

        elif data.shape[0] <= 4:
            image_format = "CHW"
            padded_h = padded.shape[1]
            padded_w = padded.shape[2]

        else:
            image_format = "HWC"
            padded_h = padded.shape[0]
            padded_w = padded.shape[1]        
                
        
        top, left = _random_crop_region(
            padded_h,
            padded_w,
            target_h,
            target_w
        )

        if image_format == "HW":
            output = padded[top:top + target_h, left:left + target_w].copy()
        
        elif image_format == "CHW":
            output = padded[:, top:top + target_h, left:left + target_w].copy()
        
        elif image_format == "HWC":
            output = padded[top:top + target_h, left:left + target_w, :].copy()


        if is_tensor:
            return Tensor_CP(output)

        return output
