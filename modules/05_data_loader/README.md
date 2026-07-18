# Module 04 — Loss Functions

## Goal

1. Implement a `Dataset abstraction` and `TensorDataset` for in-memory data storage
2. Build a `DataLoader` with intelligent batching, shuffling and memory-efficient iteration
3. Master python iterator protocol for streaming dataset without loading the entire dataset
4. Compare it with Pytorch's data loading patterns and see complexity issues

## Why it matters

`DataLoaders` help to load huge datasets efficiently by batching them. For example a ImageNet with 1.2M images would result in 600GB of RAM, this exceeds on almost any machine around the world. 

![alt text](image.png)

## Core concepts

1. `Dataset` - abstract base class, universal data access interface
2. `TensorDataset(Dataset)` - Tensor-based in-memory storage
3. `DataLoader.__init__()` - Store datasets, batch size, shuffle flag
4. `DataLoader.__iter__()` - Index shuffling and batch grouping
5. `DataLoader.__collate_batch()` - Stack samples into batch tensors

## Mathematics and rules

Pipeline to enable: 

```python 
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_features, batch_labels in loader: 
    # batch_features: (32, features_dim) - ready for model.forward()
    predictions = model(batch_features)
```

## What I implemented

Classes: 
1. `Dataset` - For data abstraction
2. `TensorDataset` - For managing memory, and loading dataset into memery efficiently with giving data, only when requested
3. `DataLoader` - For distributing data into batches and efficiently iterating over the whole dataset

## Experiment

All experiments are included in the `experiment.ipynb` file inside `04_loss_functions`.

Each loss is compared with its PyTorch equivalent on a small synthetic dataset:

1. `MSELoss_CP`: evaluated on a noisy linear regression dataset `y = 2x - 1 + noise`. Its loss matches `torch.nn.MSELoss` within `atol=1e-5`, correctly ranks a good model below a mean baseline and below a wrong model, and is minimized exactly at the true slope when the slope is swept.
2. `CrossEntropy`: evaluated on random logits over 4 classes with integer class targets. Its `log_softmax` matches `torch.log_softmax` and the loss matches `torch.nn.CrossEntropyLoss` within `atol=1e-5`. The log-sum-exp trick keeps the loss finite for logits near 1000, where a naive `exp` overflows to `inf`, and the loss falls monotonically toward zero as the true-class logit is boosted.

The correctness output from the executed notebook was:

```text
TinyTorch MSELoss_CP: 0.231074
PyTorch  nn.MSELoss: 0.231074
Absolute difference:  1.49e-08
MSELoss_CP matches PyTorch

log_softmax max difference:   2.38e-07
TinyTorch CrossEntropy:       1.624034
PyTorch  nn.CrossEntropyLoss: 1.624034
Absolute difference:          1.19e-07
CrossEntropy and log_softmax match PyTorch

Loss with logits near 1000: 0.407606
Naive exp(logits):          [[inf inf inf]]
log-sum-exp keeps the loss finite while naive exp overflows
```

### Efficiency results

The timer performs warm-up calls and reports the median CPU forward-pass time per call. Tensors are created outside the timed region.

| Loss and input | TinyTorch | PyTorch | TinyTorch / PyTorch | Result |
|---|---:|---:|---:|---|
| MSELoss, `256` elements | `0.0046 ms` | `0.0090 ms` | `0.51x` | TinyTorch was about 2 times faster in this run |
| CrossEntropy, batch `256`, `4` classes | `0.0393 ms` | `0.0352 ms` | `1.12x` | PyTorch was about 1.1 times faster |

Both losses are close to PyTorch because their heavy work is vectorized NumPy rather than the Python-loop `matmul` used by the layer module. `MSELoss_CP` is a single mean-of-squared-differences expression and was slightly faster than PyTorch in this run, while `CrossEntropy` makes a few more array passes (`max`, `exp`, `sum`, `log`, and the gather of correct-class log-probabilities) and was marginally slower than PyTorch's fused kernel. These are tiny tensors on CPU, so the results should not be read as a general speed advantage; timings vary with tensor size, CPU, thread settings, and system load.

All correctness checks pass.

## What I learned

Loss functions usage between Cross Entropy loss and Binary Cross Entropy loss 

Scenario: 

`Binary Cross Entropy loss` - used in datasets where targets are independent binary decisions
`Cross Entropy` - target consists of mutually exclusive classes 


## Resources

Focal Loss for Dense Object Detection - Lin et al. (2017). Addresses class imbalance by reshaping the loss curve to down-weight easy examples. Shows how loss function design directly impacts model performance on real problems. - https://arxiv.org/abs/1708.02002