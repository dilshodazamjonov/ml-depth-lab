# Module 06 - Autograd

## Goal

1. Implement SDG with momentum to reduce oscillations and accelarete convergence in narrow valleys
2. Master Adam's adaptive learning rate mechanism with first and second moment estimation
3. Understand memory trade-offs(SDG: 2x memory vs Adam: 3x memory) and computational complexity
4. Connect optimizer state management to checkpointing and distributed training considerations

Implementation roadmap: 

1. `Optimizer` - base class - common interface: zero_grad(), step()
2. `SGD with momentum` - Velocity buffers to reduce oscillations 
3. `Adam optimizer` - First and second moment esitmation with bias correction 
4. `AdamW optimizer` - Decoupled wight decay for proper regulirization.

## Why it matters

We can picture optimization as hiking in a foggy forest and you can feel the slope under your feet but cant see the valley. Each optimizer gives different strategies for you to choose as your next step to finding that slope. Overall, an optimizer is the rule that turns gradient into a parameter update

## Core concepts

![alt text](https://mlsysbook.ai/tinytorch/assets/images/diagrams/07_optimizers-diag-1.svg)

## Mathematics and rules



## What I implemented

Classes: 
1. `Cross Entropy and MSE`
2. `Log_softmax` - Overflow saving numerical stability technique used in classification.

## Experiment

All experiments are included in the `experiment.ipynb` file inside `06_autograd`.

Every backward rule and the `_sum_to_shape` helper is tested directly: a known `grad_output` is fed into `.apply(...)` and the returned gradients are asserted against the closed-form derivative of each operation. The checks cover numerical values (`np.allclose`, `atol=1e-6`), exact gradient shapes, `None` gradients for constants and `requires_grad=False` operands, broadcasting reduction, and every `axis`/`keepdims` combination for `Sum`/`Mean`.

A final end-to-end section drives the real `backward()` on `y = x * x` at `x = 3.0`: the first call gives `x.grad = 6.0`, a second call accumulates to `12.0`, `zero_grad()` on the leaf resets it to `None`, and the next `backward()` starts fresh at `6.0` again.

The summary from the executed notebook:

```text
component            | passed | failed |  status
----------------------------------------------------
_sum_to_shape        |      7 |      0 |  OK
AddBackward          |      4 |      0 |  OK
SubBackward          |      3 |      0 |  OK
MulBackward          |      4 |      0 |  OK
DivBackward          |      3 |      0 |  OK
MatMulBackward       |      4 |      0 |  OK
SumBackward          |      8 |      0 |  OK
MeanBackward         |      8 |      0 |  OK
ReshapeBackward      |      4 |      0 |  OK
TransposeBackward    |      3 |      0 |  OK
backward/zero_grad   |      4 |      0 |  OK
----------------------------------------------------
TOTAL                |     52 |      0 |

Overall: 52 passed, 0 failed out of 52 checks.
```

### Efficiency results

This module's notebook is correctness-only; no timing benchmark was run. Every backward rule is a vectorized NumPy expression (no Python loops over elements), and `backward()` visits each node once in topological order, so the cost of a backward pass stays proportional to the cost of the forward pass.

All 52 correctness checks pass.

## What I learned

## Resources