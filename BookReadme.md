## Section 2. ML systems

### Types of ml systems: 
1. Cloud ML(Datacenters connected across wires and more)
2. Edge ML (Inside an enterprises like Banks and pharmasies to prevent data leakage)
3. Mobile ML (ML for phones)
4. TinyML (ML for watches, microcontrollers and etc.)



### Limits behind those groups: 

## 1. The Light Barrier

The round-trip latency of light travelling through an optical fiber is:

$$
L = \frac{2D}{c_{\text{fiber}}}
$$

where:

- $L$ is the round-trip latency,
- $D$ is the one-way distance,
- $c_{\text{fiber}} \approx 200{,}000\ \text{km/s}$.

---

## 2. The Power Wall

Dynamic power consumption is approximately:

$$
P_{\text{dynamic}} \propto C V^2 f
$$

where:

- $C$ is capacitance,
- $V$ is voltage,
- $f$ is clock frequency.

If increasing frequency also requires a roughly proportional increase in voltage, then:

$$
V \propto f
$$

Therefore:

$$
P_{\text{dynamic}} \propto f^3
$$

Thus, doubling the clock frequency can require approximately:

$$
2^3 = 8\times
$$

more dynamic power. This power and heat limitation pushed the industry toward:

- **Parallelism:** multi-core CPUs
- **Specialized hardware:** GPUs and TPUs (Tensor Processing Units)

---

## 3. The Memory Wall

If compute performance grows by $1.6\times$ per period while memory bandwidth grows by only $1.2\times$, their relative growth is:

$$
\frac{\text{Compute growth factor}}
     {\text{Memory-bandwidth growth factor}}
=
\frac{1.6}{1.2}
\approx 1.33
$$

This means compute capability improves approximately **33% faster** than memory bandwidth, causing processors to spend an increasing amount of time waiting for data.

### Analyzing the workloads

Iron law of ML Systems: 
$$ T = \frac{D_\text{vol}}{\text{BW}} + \frac{O}{R_\text{peak} * \eta_\text{hw}} + L_\text{lat}$$

total latency here is decomposed into 3 terms: `Data movement` -> $\frac{D_\text{vol}}{\text{BW}}$, `Compute` -> $\frac{O}{R_\text{peak} * \eta_\text{hw}}$ and `Fixed overhead` $L_\text{lat}$


### 