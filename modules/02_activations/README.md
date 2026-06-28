# Module 01 — Tensors

## Goal

Build a 5 activation layers such as: ReLU, Sigmoid, Tanh, GeLU and SoftMax

## Why it matters

Tensors are universal Data Structure for ML. They represent the information in a multidimention allowing to make us fast computations and transformations on them.

## Core concepts

1. Transpose of a matrix with shape `[A, B]` is a matrix that has a same elements with the shape of `[B, A]`
2. Reshape is a method for changing the shape of a matrix
3. `MatMul` vs `__mul__` -> `__mul__` multiplies each of element at A at the same indices with the same indices at B.
   `MatMul` algorithm for computation:
   ```
    1. Create a tensor filled with 0s and shape of a [i,j] where i, j are outer dimentions of A and B respectively
    2. Loop through A from index 0 till A.shape[0]
    3. Loop through B from index 0 till B.shape[-1]
    4. Loop through from A.shape[-1] and B.shape[0] as inner radiuses must be equal
    5. Compute the sum of products for each index at A[i, k] and B[k, j]
   ```

## Mathematics and rules

1. MatMul

$$
C_{i, j} = \sum_k A_{i, k} * B_{i, j}
$$

#### Yet to be written

## What I implemented

1. Class: Tensor
2. Methods
3. Supported oprations

## Experiment

All of the experiments are included in a experiment.ipynb file inside of the 01_tensors

1. Checking matrix multiplication with various arrays and some fail which is expected.
2. Comparing my class performance against PyTorch's matmul -> it was faster than mine, used 100x faster time for computation

Comparion results:

```
   Custom: 0.35976630001096055
   NumPy: 0.00021169998217374086
```

## What I learned

Tensors are fundamental and most used datastructures in DeepLearning and Model training, which helps with computing much faster with the help of library PyTorch

## Difficulties and open questions

Nothing. Everything was super simple, especially because I was moving slowly but understanding a topic

## Resources

NumPy: Array Programming - Harris et al. (2020). The definitive reference for NumPy, which underlies your Tensor implementation. Explains broadcasting, views, and the design philosophy. Systems Implication: Standardized memory layouts (strides) and contiguous blocks allowed C-level operations to bypass the slow Python interpreter, maximizing memory bandwidth. [Nature](https://www.nature.com/articles/s41586-020-2649-2)

https://mlsysbook.ai/tinytorch/modules/01_tensor.html
