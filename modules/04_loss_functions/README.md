# Module 04 — Loss Functions

## Goal

To Implement 3 loss functions: `Mean Squared Error (MSE)`, `CrossEntropy` and `Binary Cross-Entropy`. Along the way I will implement the `log-sum-exp trick` - the one numerical method that separates a softmax that trains from one that returns nan on tht first batch with large logits

## Why it matters

Losses are backbone part of Neural Network, without it NN would turn to a guessing machine. It turns one forward method into a learning step, and shows the direction in which optimization should move.

## Core concepts

![alt text](image.png)

1. `MSELoss` - Mean squared error for continuous predictions
2. `log softmax` - Log-sum-exp trick for numerical stability
3. `CrossEntropyLoss` - Negative log-likelihood for multi-class classification
4. `BinaryCrossEntropyLoss` - Cross-entropy specialized for binary decisions

## Mathematics and rules

1. `MSELoss` - formula: 

$$ Loss = \frac{1}{N} \sum_i^n({\text{predictions} - \text{targets}})^2 $$

Where `N`is the number of elements in Tensor

2. `Cross Entropy` 

3. `Log_softmax` - is a numerical stability technique in classification. It saves overflow of the exponent calcluation from `float32` resulting in $-\infty$


## What I implemented

Classes: 
1. `Layer`- A base class to prevent type erros 
2. `Linear_CP` - Class for linear layer to work
3. `Dropout_CP` - Class for Dropout layer
4. `Sequential` - Class that orchestrates all the layers 

Each was implemented with `forward, parameters and __call__` methods

## Experiment

All experiments are included in the `experiment.ipynb` file inside `03_layers`.

Each layer is compared with its PyTorch equivalent using the same inputs and parameters:

1. `Linear_CP`: its weights and bias are copied into `torch.nn.Linear`. The weight is transposed because TinyTorch stores it as `(in_features, out_features)`, while PyTorch uses `(out_features, in_features)`. The outputs match within `atol=1e-5`, and both layers return the same number of parameters.
2. `Dropout_CP`: exact values are not compared because NumPy and PyTorch generate different random masks. Instead, the experiment checks that the fraction of zeroed values is close to `p`, inverted scaling preserves the expected mean, and inference mode returns the input unchanged.
3. `Sequential`: both implementations use `Linear(4, 5) -> ReLU -> Linear(5, 2)` with synchronized parameters. Their outputs match within `atol=1e-5`, and both containers collect the same number of parameters.

The correctness output from the executed notebook was:

```text
Linear_CP
  Max absolute difference: 1.19e-07
  Output and parameter count match PyTorch: PASS

Dropout_CP (p = 0.25)
  Expected zero fraction:  0.250
  TinyTorch zero fraction: 0.251, mean: 0.999
  PyTorch zero fraction:   0.250, mean: 1.000
  Statistics and inference behavior match PyTorch: PASS

Sequential
  Maximum output difference: approximately 2.98e-08
  Parameters: TinyTorch = 37, PyTorch = 37
  Output and collected parameters match PyTorch: PASS
```

### Efficiency results

The timer performs warm-up calls and reports the median CPU forward-pass time per call. Inputs and layer construction are outside the timed region.

| Layer and input | TinyTorch | PyTorch | TinyTorch / PyTorch | Result |
|---|---:|---:|---:|---|
| Linear, batch `32`, `64 -> 32` | `38.7865 ms` | `0.0073 ms` | `5331.48x` | PyTorch was about 5,331 times faster |
| Dropout, shape `256 x 256` | `0.6309 ms` | `0.7892 ms` | `0.80x` | TinyTorch was about 1.25 times faster in this run |
| Sequential, batch `16`, `32 -> 48 -> 16` | `20.6633 ms` | `0.0261 ms` | `791.85x` | PyTorch was about 792 times faster |

PyTorch is much faster for `Linear` and `Sequential` because it uses optimized compiled matrix-multiplication kernels, while `Tensor_CP.matmul` currently performs the multiplication with three Python loops. `Dropout_CP` is competitive in this small CPU benchmark because its work is handled by vectorized NumPy operations. The dropout result should not be interpreted as a general advantage over PyTorch; timings vary with tensor size, CPU, thread settings, and system load.

All correctness checks pass.

## What I learned

Layers are the functions with weights and parameters. 

What I learnt:
1. Linear model formula $y = X*W + b$
2. Dropout layer without scaling would give less applicable results as a magnitudes during training and inference will not be aligned

## Resources

Understanding the difficulty of training deep feedforward neural networks - Glorot and Bengio (2010). Introduces Xavier/Glorot initialization and analyzes why proper weight scaling matters for gradient flow. The foundation for modern initialization schemes - https://proceedings.mlr.press/v9/glorot10a.html 