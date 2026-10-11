# Gate 5a Parameter and Training Efficiency

Status: **PASS**

Formal wall times are hardware-tagged in each run and are not treated as controlled speed comparisons.

```json
{
  "status": "PASS",
  "rows": [
    {
      "dataset": "electricity",
      "head": "Gaussian",
      "variant": "gaussian_head",
      "trainable_parameter_count": 24624,
      "total_parameter_count": 4672072,
      "parameter_overhead_ratio": 0.005298391719498529,
      "training_wall_seconds": 613.886513710022,
      "epochs_completed": 4,
      "mean_epoch_seconds": 153.4716284275055,
      "peak_gpu_memory_mib": 5036.68408203125,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__gate4b_cross_dataset__gaussian_head__random_rate0.3__seed2025__20260716_062020"
    },
    {
      "dataset": "electricity",
      "head": "Laplace",
      "variant": "laplace_head",
      "trainable_parameter_count": 24624,
      "total_parameter_count": 4672072,
      "parameter_overhead_ratio": 0.005298391719498529,
      "training_wall_seconds": 846.2012827396393,
      "epochs_completed": 4,
      "mean_epoch_seconds": 211.55032068490982,
      "peak_gpu_memory_mib": 5036.68408203125,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__gate5a__laplace_head__random_rate0.3__seed2025__20260720_140542"
    },
    {
      "dataset": "electricity",
      "head": "Fixed-t8",
      "variant": "student_t_fixed_df8_head",
      "trainable_parameter_count": 24624,
      "total_parameter_count": 4672072,
      "parameter_overhead_ratio": 0.005298391719498529,
      "training_wall_seconds": 2075.9062662124634,
      "epochs_completed": 4,
      "mean_epoch_seconds": 518.9765665531158,
      "peak_gpu_memory_mib": 5036.685546875,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__gate5a__student_t_fixed_df8__random_rate0.3__seed2025__20260720_141956"
    },
    {
      "dataset": "electricity",
      "head": "Learnable-t",
      "variant": "student_t_head",
      "trainable_parameter_count": 24945,
      "total_parameter_count": 4672393,
      "parameter_overhead_ratio": 0.005367461884457879,
      "training_wall_seconds": 1786.6152453422546,
      "epochs_completed": 4,
      "mean_epoch_seconds": 446.65381133556366,
      "peak_gpu_memory_mib": 5036.68994140625,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__gate4b_cross_dataset__student_t_head__random_rate0.3__seed2025__20260716_184923"
    },
    {
      "dataset": "etth1",
      "head": "Gaussian",
      "variant": "gaussian_head",
      "trainable_parameter_count": 6168,
      "total_parameter_count": 1142564,
      "parameter_overhead_ratio": 0.0054276854195192525,
      "training_wall_seconds": 73.73548150062561,
      "epochs_completed": 9,
      "mean_epoch_seconds": 8.19283127784729,
      "peak_gpu_memory_mib": 242.89013671875,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__gate4b_corrected_minimal__gaussian_head__random_rate0.3__seed2025__20260713_130557"
    },
    {
      "dataset": "etth1",
      "head": "Laplace",
      "variant": "laplace_head",
      "trainable_parameter_count": 6168,
      "total_parameter_count": 1142564,
      "parameter_overhead_ratio": 0.0054276854195192525,
      "training_wall_seconds": 69.64980578422546,
      "epochs_completed": 9,
      "mean_epoch_seconds": 7.738867309358385,
      "peak_gpu_memory_mib": 242.89013671875,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__gate5a__laplace_head__random_rate0.3__seed2025__20260720_134744"
    },
    {
      "dataset": "etth1",
      "head": "Fixed-t8",
      "variant": "student_t_fixed_df8_head",
      "trainable_parameter_count": 6168,
      "total_parameter_count": 1142564,
      "parameter_overhead_ratio": 0.0054276854195192525,
      "training_wall_seconds": 121.77316212654114,
      "epochs_completed": 10,
      "mean_epoch_seconds": 12.177316212654114,
      "peak_gpu_memory_mib": 242.890625,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__gate5a__student_t_fixed_df8__random_rate0.3__seed2025__20260720_134856"
    },
    {
      "dataset": "etth1",
      "head": "Learnable-t",
      "variant": "student_t_head",
      "trainable_parameter_count": 6175,
      "total_parameter_count": 1142571,
      "parameter_overhead_ratio": 0.005433845244087449,
      "training_wall_seconds": 97.1397156715393,
      "epochs_completed": 9,
      "mean_epoch_seconds": 10.793301741282145,
      "peak_gpu_memory_mib": 242.89208984375,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__gate4b_corrected_minimal__student_t_head__random_rate0.3__seed2025__20260713_130713"
    },
    {
      "dataset": "weather",
      "head": "Gaussian",
      "variant": "gaussian_head",
      "trainable_parameter_count": 24624,
      "total_parameter_count": 4518472,
      "parameter_overhead_ratio": 0.005479491073129309,
      "training_wall_seconds": 188.6423122882843,
      "epochs_completed": 4,
      "mean_epoch_seconds": 47.160578072071075,
      "peak_gpu_memory_mib": 690.17138671875,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__gate4b_cross_dataset__gaussian_head__random_rate0.3__seed2025__20260715_233846"
    },
    {
      "dataset": "weather",
      "head": "Laplace",
      "variant": "laplace_head",
      "trainable_parameter_count": 24624,
      "total_parameter_count": 4518472,
      "parameter_overhead_ratio": 0.005479491073129309,
      "training_wall_seconds": 362.26164960861206,
      "epochs_completed": 8,
      "mean_epoch_seconds": 45.28270620107651,
      "peak_gpu_memory_mib": 690.17138671875,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__gate5a__laplace_head__random_rate0.3__seed2025__20260720_135107"
    },
    {
      "dataset": "weather",
      "head": "Fixed-t8",
      "variant": "student_t_fixed_df8_head",
      "trainable_parameter_count": 24624,
      "total_parameter_count": 4518472,
      "parameter_overhead_ratio": 0.005479491073129309,
      "training_wall_seconds": 497.3863773345947,
      "epochs_completed": 6,
      "mean_epoch_seconds": 82.89772955576579,
      "peak_gpu_memory_mib": 690.171875,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__gate5a__student_t_fixed_df8__random_rate0.3__seed2025__20260720_135719"
    },
    {
      "dataset": "weather",
      "head": "Learnable-t",
      "variant": "student_t_head",
      "trainable_parameter_count": 24645,
      "total_parameter_count": 4518493,
      "parameter_overhead_ratio": 0.005484164128381735,
      "training_wall_seconds": 412.71444296836853,
      "epochs_completed": 6,
      "mean_epoch_seconds": 68.78574049472809,
      "peak_gpu_memory_mib": 690.17333984375,
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__gate4b_cross_dataset__student_t_head__random_rate0.3__seed2025__20260715_234156"
    }
  ]
}
```
