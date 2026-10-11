# Gate 4a Training Smoke

Status: **PASS**

Both variants completed two GPU smoke epochs with 32 train batches per epoch,
8 fixed-mask validation batches, and 8 fixed-mask test batches. Losses,
metrics, and all new-parameter gradients were finite. The optimizer contained
no backbone parameters and the frozen backbone had no gradients.

| Variant | Trainable params | Peak GPU MiB | Checkpoint reload |
|---|---:|---:|---|
| static_adapter | 33,600 | 239.520 | PASS |
| missingness_aware_adapter | 39,136 | 239.585 | PASS |

Smoke metrics are diagnostic only and are not formal results.
