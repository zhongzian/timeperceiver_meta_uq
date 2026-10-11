# Heavy-Tail Mechanism

PIT, distribution-normalized QQ, Gaussian-residual exceedance, residual-percentile NLL decomposition, and channel-level kurtosis/df analyses are secondary descriptive evidence. They do not replace the formal metric gates and do not support p-value or significance claims.

- Gaussian residual heavy tail observed: `True`
- Heavy-tail gain concentrated in extremes: `True`
- Fixed-t explains most Student-t gain: `True`
- Learned df adds value beyond fixed-t: `False`
- Per-channel df associated with channel tails: `True`

The controlled latency benchmark uses the same GPU, batch, precision, warmup, and iteration count within each dataset. Formal training wall times from different GPU models are retained but not treated as strict speed comparisons.
