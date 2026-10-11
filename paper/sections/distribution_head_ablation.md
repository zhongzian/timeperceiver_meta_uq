# Distribution Head Ablation

Gate 5a compares matched Gaussian, Laplace, fixed-df Student-t, and per-channel learnable-df Student-t heads on frozen official TimePerceiver checkpoints for ETTh1, Weather, and Electricity under the corrected random-0.3 fixed-mask protocol. The formal decision is `FIXED_HEAVY_TAIL_SUFFICIENT`. All heads share the same location and scale projections; Laplace and fixed-t8 have exactly the Gaussian trainable capacity, while learnable-t adds one df parameter per target channel.

The comparison uses a single probability-head training seed with fixed backbones. It does not establish formal significance, superiority beyond the evaluated scope, or novelty of Student-t forecasting itself. Table 8 reports every distribution and dataset rather than selecting only the best head. Fixed df=8 passes the matched Gaussian gates in all three datasets, whereas learned per-channel df does not add a consistent benefit over fixed df=8.
