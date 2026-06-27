# Module 01 — Tensors

## Goal

Build a class called Tensor which will support arithmetic operations, transpose, matmul, slicing, reshaping and broadcasting

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

$$ C_{i, j} = \sum_k A_{i, k} * B_{i, j} $$

#### Yet to be written

## What I implemented

1. Class: Tensor
2. Methods
3. Supported oprations

## Design decisions

Important implementation choices and why I made them.

## Experiment

What I explored in the notebook and what happened.

## What I learned

The concept explained in my own words.

## Difficulties and open questions

What was confusing or remains unclear.

## Resources

TinyTorch and a small number of useful additional sources.
