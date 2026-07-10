## Section 2. ML systems

### Types of ml systems: 
1. Cloud ML(Datacenters connected across wires and more)
2. Edge ML (Inside an enterprises like Banks and pharmasies to prevent data leakage)
3. Mobile ML (ML for phones)
4. TinyML (ML for watches, microcontrollers and etc.)



### Limits behind those groups: 

1. The light barrier: Formula for computations of a Latency for a light transfer through a fiber: 
$$ L = \frac{2 * \text{Distance}}{c_\text{fiber}} $$ 
where $c_\text{fiber} = 200 000 \text{ km/s}$

2. The power wall - Due to thermodynamics -> $Power ∝ C * V^2 * f ->  P ∝ f^2$ Meaning doubling frequency results in 8x more Power. Hence industry moved towards `parallelism(multi-core)` and specialization `(GPUs, TPUs(Tensor processing unit))`
3. The memory wall - $\frac{\text{compute growth rate}}{\text{Memory Bandwidth Growth Rate}} = \frac{1.6}{1.2} = 1.33$

### Analyzing the workloads

Iron law of ML Systems: 
$$ T = \frac{D_\text{vol}}{\text{BW}} + \frac{O}{R_\text{peak} * \eta_\text{hw}} + L_\text{lat}$$

total latency here is decomposed into 3 terms: `Data movement` -> $\frac{D_\text{vol}}{\text{BW}}$, `Compute` -> $\frac{O}{R_\text{peak} * \eta_\text{hw}}$ and `Fixed overhead` $L_\text{lat}$