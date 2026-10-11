# Gate 4a Plan

Status: **PLANNING_PASS**

The minimal matrix contains two adapter-only ETTh1 tasks: a static residual
bottleneck adapter and a missingness-aware adapter with the same main adapter
and a 7-dimensional pooled missingness condition. The official backbone is
frozen. No probability head, meta-learning, test-time adaptation, or test-mask
access during training is included.

Both GPU training smokes passed. The validated formal queue contains exactly
two full-training tasks and uses `required_free_mib=4096`. Scheduler PID
`1591346` is started and currently waiting for resources.
