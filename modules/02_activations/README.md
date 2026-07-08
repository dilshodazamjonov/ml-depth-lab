# Module 02 — Activations

## Goal

Build a 5 activation layers such as: ReLU, Sigmoid, Tanh, GeLU and SoftMax

## Why it matters

Activation layers are a key component in DeepLearning model. Without it model would just get an input as a Tensor and return that Tensor with no transformations. Activation layers are the parts in that pipeline which help to make a predictions by transforming, changing the passed data.

Concept of activation layers and where they lie during a whole workflow:  

![1782647813661](image/README/1782647813661.png)

## Core concepts

1. ReLU.forward(): Sparsing data through zeroing negatives. Helps with feature selection and reduce computation time 
2. Sigmoid.forward(): Mapping to (0, 1) for probabilities 
3. Tanh.forward(): Hyperbolic tangent, zero-centered activation for better gradients
4. GELU.forward(): Smooth non-linearity for transformers
5. Softmax.forward(): Probability distributions with numerical stability

## Mathematics and rules

1. ReLU (Rectified Linear Unit) - replace the negatives with 0, math formula being: 

$$ f(x) = \max(0, x) $$

2. Sigmoid - Transforms values between 0 and 1 fits good for transforming data into raw probabilities, formula: 

$$ f(x)_{\text{[positive mask]}} = \frac{1}{1 + \exp^{-x}} $$
$$ f(x)_{\text{[negative mask]}} = \frac{\exp_z}{1 + \exp_z} $$ 

where $\exp_z = \exp^{z[\text{negativemask}]}$

3. Tanh - zero-centered needs, formula: 
$$ \tanh(x) = \frac{\exp^x - \exp^{-x}}{\exp^x + \exp^{-x}} $$

4. GELU (Gaussian Error Linear Unit)
Math Formula: 

$$ GeLU(X) = \sigma (X * 1.702) * X $$

5. Softmax - Turnes the given tensor to a distribution probabilities, where sum of them becomes 1. This makes it essensial for the multiclass classification 

Algorithm: 
* Identify the max from the given axis 
* Substract it from the Tensor, and get X_shifted
* Get the exponent of that shifted x 
* Sum up the exponents and finally to the result devide the exp_values / exp_sum 

Formula is: 

$$ Softmax(x)_i = \frac{\exp(x_i - \max(x))}{\sum_j{\exp(x_j - \max{x})}} = \frac{\exp(x_i)}{\exp(\max(x))} : \frac{\sum_j{\exp(x_j)}}{\exp(\max(x))} = \frac{\exp(x_i)} {\sum_j{\exp(x_j)}} $$ 

## What I implemented

1. Classes: ReLU, GeLU, Sigmoid, Tanh, Softmax each with the methods of forward and `__call__` and a placeholder for the method called `backward`

## Experiment

All of the experiments are included in the experiment.ipynb file inside of 02_activations.

Each activation is tested by feeding the same input into both my `_CP` implementation and the matching PyTorch layer, then comparing the outputs with `np.allclose`:

1. ReLU: checked that negatives are zeroed and positives pass through unchanged.
2. Sigmoid: matches `torch.nn.Sigmoid` within `atol=1e-5`.
3. Tanh: matches `torch.nn.Tanh` within `atol=1e-5`.
4. GeLU: my version uses the sigmoid approximation `x * sigmoid(1.702x)`, while PyTorch's `GELU()` is the exact erf-based version, so they are compared with a looser `atol=0.05`. The measured max absolute difference is about 0.02.
5. Softmax: matches `torch.nn.Softmax(dim=-1)` within `atol=1e-5`, and each row sums to 1.

All comparisons pass.

## What I learned

Activations - are fundamental blocks in Neural Networks, without them stacking a one neural network on top of another would just mean a one NN, which would not affect or change the data.

What I learnt:
1.  that `ReLU` is a activation layer that replaces negative numbers with 0, but have drawbacks for the data with all negatives then it changes everything into 0 and then gradient descent for backward propagation becomes 0. 
2.  `Sigmoid` outputs values between 0 and 1 making it good for transforming and representing raw values in a view of `probabilities`
3.  in `Softmax` we substract max from the intire Matrix in order to normalize the vector, so the huge numbers in exponents will not explode, for example: $\exp^{100}≈ 2.7 * 10^{43}$ hence it overflows to inf, then numerator and denominator -> inf resulting in $\frac{inf}{inf}= nan$

## When to use which? 

For Hidden Layers: 
* Default: `ReLU` - fast, no vanishing gradients, creates sparsity
* Transformers: `GeLU` - smooth, better gradient flow, the de-facto choice in `GPT/BERT`
* Recurrent Networks: `Tanh` - zero-weighed, plays well with repeated weighed applicaitons
* When `ReLU` is dying on you: `LeakyReLU`, `ELU`, `Swish`

For Output Layers: 
* In `binary classification`: `Sigmoid` - outputs [0, 1] probabilities
* Multi-Class classification: `Softmax` - outputs multiple probabilities in range of [0, 1] where probability distribution sums up to 1
* Regression: None - leave the linear output alone
  
Computational cost (relative to ReLU):
* ReLU - 1x
* `Sigmoid / Tanh` - 3-4x (one exponential per element) 
* `GELU`: 4–5x (exponential plus the approximation polynomial)
* `Softmax` - 5x+ (exponential, sum-reduction, division)

## Resources

https://proceedings.mlr.press/v15/glorot11a.html - Deep Sparse Rectifier Neural Networks
