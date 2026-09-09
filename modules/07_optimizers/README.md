# Module 07 - Optimizers

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

1. `SGD - Stochastic Gradient Descent` - an optimization technique that works with a random subset of dataset in order to find best possible settings(Weights)

   * `With weight decay enabled`: 

   $$ \text{grad} = \text{grad} + \omega_i * \theta_\text{old} $$ 
   and then data will be updated as
   $$ \theta_\text{new} = \theta_\text{old} - \eta * grad $$

   Where $\eta$ is a Learning Rate and $\theta_\text{old} \space \text{and} \space \theta_\text{new}$ are old and new data respectively.
      
    * `With momentum enabled` algorithm:

   ```
   1. obtain the old_buffer by id from momentum buffers if exist
   2. if not add it to the momentum buffer with `np.zeros_like(tensor.data)`

   ```

   $$ \text{buffer}_\text{new} = \text{self.momentum} * \text{buffer}_\text{old} + \text{grad} $$

   then grad equal to updated_buffer

   * `No momentum and no weight decay`: 

then data change will be done in an ordinary way using formula: 

$$ \theta_\text{new} = \theta_\text{old} - \eta * grad $$


2. `Adam - Adaptive moment estimation`

Algorithm: 

```
1. Store first and second moment buffers 
2. Retrieve the tensor_id and the tensor.grad
3. If weight decay passed then gradient += weight_decay * tensor.data
4. Get old moment buffers from first and 2nd moment buffers dictionary by Tensor_id
5. Updated_first and 2nd moments calculation 
6. Update the buffers in dictionaries by the Tensor_id
7. Calculate the corrected first and 2nd moments
```

formulas: 

* Updated_moments:
   $$ m_t = \beta_ 1 * m_{t-1} + (1 - \beta_1) g_t$$

   $$ v_t = \beta_2 * v_{t-1} + (1 - \beta_2) g_t^2 $$

* bias_corrected_moments

   $$ \hat{m_t} = \frac{m_t}{1 - \beta_1 ^ t}$$
   $$ \hat{v_t} = \frac{v_t}{1 - \beta_2 ^ t}$$

Finally, updated parameter: 

$$ \theta_t = \theta_{t - 1} - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} $$

3. `AdamW - ADAM with decoupled weight decay`

Instead of updating the gradient for each tensor like `ADAM` does, `AdamW` updates the data by $\theta_t = \theta_t(1 - \eta * \text{weight decay})$. Hence, lost gradient remains unchanged

## What I implemented

Classes:
1. `Optimizer` - base class holding the parameter list, `zero_grad()`, and the `step()` interface.
2. `SGD` - plain gradient descent with optional momentum (velocity buffers per parameter) and coupled weight decay.
3. `Adam` - first/second moment estimation with bias correction and **per-parameter step counters** (`step_counts` keyed by tensor identity, like PyTorch's `state[param]["step"]`).
4. `AdamW` - Adam with decoupled weight decay: the parameter is shrunk in place before the Adam update, so the moment buffers never see the decay term.

## Experiment

All experiments are included in the `experiment.ipynb` file inside `07_optimizers`. They verify `SGD`, `Adam`, and `AdamW` in four passes:

1. **Manual correctness** - every update rule is checked against hand-computed values: plain SGD, weight decay only, momentum over two steps, momentum + weight decay together, Adam's moments/bias correction/coupled decay/epsilon stability, and AdamW's decay-before-update first step.
2. **PyTorch parity** - identical initial parameters and manually assigned, per-step-changing gradients, compared against `torch.optim.SGD` / `Adam` / `AdamW` (loop implementation, `foreach=False`, `fused=False`) over 1, 2, 20, and 100 steps and scalar/vector/matrix shapes.
3. **Integration training** - real `backward()` + `step()` + `zero_grad()` loops: a quadratic `L = (x - 3)^2` and a 64-point linear regression through the full tensor → autograd → `MSELoss_CP` → optimizer chain.
4. **Performance** - milliseconds per `step()`, optimizer-state memory, and memory stability across repeated steps, with `torch.optim` timed on the identical schedule as a baseline.

Key outcomes from the executed notebook:

- Constructor validation rejects malformed inputs for all three optimizers (non-list params, non-`Tensor_CP` elements, `lr <= 0`, negative `weight_decay`, momentum outside [0, 1], betas outside [0, 1), `epsilon <= 0`) - 29 rejection checks pass.
- Common behavior holds for six configurations: `step()` returns `None`, never mutates `.grad`, skips `grad=None` parameters (including AdamW's decay), preserves shapes and float32 dtype, and `zero_grad()` resets every gradient to `None`.
- SGD matches `torch.optim.SGD` with **max diff 0.00e+00** in every configuration (plain, weight decay, momentum, momentum + weight decay) for up to 20 steps.
- Adam matches `torch.optim.Adam` at `atol=1e-6` in all configurations (worst 20-step diff 2.98e-08) and at 1.49e-08 over 100 steps. The intermittent `grad=None` case - a parameter receiving its first gradient at step 4 - matches PyTorch exactly (0.999 vs 0.999) thanks to the per-parameter step counters; an earlier global-counter design gave 0.99941885 here and is now guarded against by a regression test.
- AdamW matches `torch.optim.AdamW` (worst diff 5.96e-08 at 20 steps, 2.98e-08 at 100 steps). Decoupling is verified structurally: AdamW(wd=0.1)'s moment buffers equal plain Adam(wd=0)'s exactly, while coupled Adam(wd=0.1)'s differ, and AdamW(wd=0) is bit-for-bit identical to Adam over 10 random steps.
- End-to-end: SGD reaches `x = 3.000000` on the quadratic (loss 9.0 → 2.27e-13) and Adam reaches loss 0.0; the regression recovers `w = 2.0000, b = 1.0000` with both optimizers.
- Documented (pinned, not enforced) behaviors: an empty parameter list is a no-op, duplicate tensors in `params` update twice per step, and `lr=True` is accepted as `lr=1.0` because `bool` subclasses `int`.

### Efficiency results

Benchmark setup: NumPy 2.5.0, PyTorch 2.12.1+cpu, Windows 11, single CPU. Gradients are pre-created outside the timed region and 5 warm-up steps allocate the state buffers first. Four layouts: one tensor of 1k / 100k / 1M elements, plus 1,000 tensors of 1k elements (1M total) to expose per-tensor Python overhead. `torch ms` is `torch.optim` on the identical schedule; `max diff` compares final parameters against that PyTorch run.

```text
optimizer     layout              mean ms  median ms  torch ms  state MB  growth MB   max diff
SGD           1 tensor x 1k        0.0016     0.0015    0.0093       0.0       0.00   7.57e-05
SGD           1 tensor x 100k      0.0197     0.0193    0.0622       0.4       0.00   2.28e-04
SGD           1 tensor x 1M        3.6973     3.6566    0.1924       0.0       0.00   6.48e-05
SGD           1000 tensors x 1k    1.7600     1.5943    1.8820       0.0       0.00   6.48e-05
SGD+momentum  1 tensor x 1k        0.0029     0.0026    0.0169       0.0       0.00   1.53e-05
SGD+momentum  1 tensor x 100k      0.0523     0.0457    0.0674       1.1       0.00   1.53e-05
SGD+momentum  1 tensor x 1M        7.1564     7.1025    0.4916       3.8       0.00   1.53e-05
SGD+momentum  1000 tensors x 1k    3.1633     2.9870    6.3991       4.1       0.00   1.53e-05
Adam          1 tensor x 1k        0.0086     0.0081    0.0361       0.0       0.00   5.60e-06
Adam          1 tensor x 100k      0.2147     0.1929    0.2864       0.3       0.00   8.34e-06
Adam          1 tensor x 1M       25.2712    25.2264    0.9572       7.6       0.00   1.07e-05
Adam          1000 tensors x 1k    9.4931     9.2782   20.5064       9.0       0.00   1.07e-05
AdamW         1 tensor x 1k        0.0086     0.0084    0.0353       0.0       0.00   5.72e-06
AdamW         1 tensor x 100k      0.2216     0.1999    0.3483       1.2       0.00   7.09e-06
AdamW         1 tensor x 1M       22.0013    21.8036    1.1625       7.6       0.00   1.11e-05
AdamW         1000 tensors x 1k   10.5232    10.3040   23.5324       9.1       0.00   1.11e-05
```

What the numbers show:

- **Time scales O(P)**, as expected: SGD goes from 0.0016 ms at 1k elements to 3.7 ms at 1M (~1000x elements, ~2300x time - the extra factor is memory bandwidth once arrays leave cache).
- **Per-step cost ordering matches the arithmetic**: momentum roughly doubles SGD (one extra buffer pass), and Adam/AdamW cost 3-7x SGD because each step makes several elementwise passes (two moment updates, two bias corrections, sqrt, divide).
- **Memory matches the theory**: for 1M float32 parameters (~4 MB) the momentum state settles at ~4 MB and the Adam/AdamW state at ~8 MB (two buffers) - exactly the 2x vs 3x parameter-memory trade-off this module is about. The `growth MB` column is 0.00 everywhere, so no state is recreated or leaked across steps.
- **TinyTorch is competitive - or faster - where Python overhead dominates**: it beats PyTorch's loop implementation on small tensors (0.0086 vs 0.0361 ms for Adam at 1k) and on the 1,000-tensor layout (9.5 vs 20.5 ms), because `torch.optim`'s per-parameter bookkeeping costs more per tensor.
- **PyTorch wins decisively on one large tensor** (0.96 vs 25.3 ms for Adam at 1M, ~26x): its kernels update in place and use multithreading, while every TinyTorch elementwise expression allocates a fresh temporary array, so the 1M-element step is bound by allocation and memory traffic. Fusing the update into in-place NumPy operations (`np.multiply(..., out=...)`) would be the next optimization.
- **Numerical agreement survives long runs**: after 200-1,000 float32 steps the final parameters stay within 2.3e-04 of PyTorch (SGD's larger drift comes from its higher learning rate in this benchmark; Adam/AdamW stay near 1e-05).

Timings vary with CPU, thread settings, and system load - rerun the benchmark cell on the target machine rather than treating one run as universal.

## What I learned

I learnt :

1. that SGG - Stochastic Gradient descent works by selecting a random subset of a dataset and changing it's directions with or without momentum and weight_decay. Preserving the momentum buffer inside of a dictionary for each tensor.

2. That Adam and AdamW differ as one changes gradient by weight decay, while other changes parameters of passed tensor by 1 - lr * weight buffer.

3. Adam uses 2x more memory than plain SGD 

## Resources