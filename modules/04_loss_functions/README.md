# Module 03 — Layers

## Goal

Is to build 3 Layers: Linear($y = xW + b$), Dropout and Sequential layers, so in the end it's possible to make `Sequential(Linear(784, 256), ReLU(), Linear(256, 10))`

Implement everything with methods of `forward()` and `parameters()`

## Why it matters

Layers are what make the neural networks work, it adds to a model weights and makes predictions via activations layers.

## Core concepts

1. `Linear layer` applies $y = x*Weight + bias$ and outputs the result as a Tensor. Bias can be ignored if parameter bias set to False
2. `Dropout layer` is a regularization technique which turns off temporarily specific percentage of neurons in our neural network in order to prevent overfitting. Works only during training and all neurons are then turned back on during validation and test
3. `Sequential layer`

## Mathematics and rules

1. `Linear Layer`. For
* Weight initialization we use formula: 
$$ \sigma = \sqrt{\frac{1}{\text{InFeats}}} $$
This initialization known as a `LeCun initialization` - it shrinks a random variable proprtionally to how many inputs the layer has(Normal distribution)
* Bias initialized at 0 because there is no gain from randomizing it.

![alt text](image.png)

2. `Dropout Layer`. Formula of a scaling unit: 

$$ \text{scale} = \frac{1}{1-p} $$
Algorithm: 

```python
Edge Case Checks: 
    If not training or prob == DropoutMinProb -> return the same result with no transformation
    If prob == DropoutMaxProb -> return the all zeros Tensor with shape of X.data.shape

    else:
        1. keep_prob = 1 - prob
        2. maskout the values in X where p < keep_prob(fill them with true)
        3. create a mask_tensor -> as type float32
        4. scale = Tensor(np.array(1 / keep_prob))
        5. compute the output = X * mask_tensor * scale
        6. Return Output
```
Why do we scale? 

Because when some neurons are turned off during training, others should keep up with the same power, hence multiplying remaining neurons with $\frac{1}{1-p}$ multiplies alive neuros to work harder. This results in equal sums from training and inference.

3. `Sequencial Layer`. Is a Container which sequentially runs layers one by one.

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