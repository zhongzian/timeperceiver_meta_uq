# Gate 2 Official Baseline Plan

Created: 2026-07-01T20:30:54

Scope: official TimePerceiver single-seed baselines only; no innovation, no multi-seed, no ablation.

| Dataset | Command source | Protocol match | Seed | Output dir | Risk |
| ------- | -------------- | -------------- | ---: | ---------- | ---- |
| ETTh1 | upstream/TimePerceiver/scripts/ETT/TimePerceiver_ETTh1.sh first pred_len=96 block; run.py defaults for unspecified args | True | 2025 | `/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official_baseline__seed2025__<timestamp>` | formal training may be long or OOM; no silent config shrinking |
| Weather | upstream/TimePerceiver/scripts/Weather/TimePerceiver.sh first pred_len=96 block; run.py defaults for unspecified args | True | 2025 | `/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__official_baseline__seed2025__<timestamp>` | formal training may be long or OOM; no silent config shrinking |
| Electricity | upstream/TimePerceiver/scripts/ECL/TimePerceiver.sh first pred_len=96 block; run.py defaults for unspecified args | True | 2025 | `/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__official_baseline__seed2025__<timestamp>` | formal training may be long or OOM; no silent config shrinking |

## Commands

### ETTh1

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python -u /mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver/run.py --is_training 1 --root_path /mnt/data/zza/code/timeperceiver_meta_uq/data/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_384_96 --model TimePerceiver --data ETTh1 --features M --target OT --freq h --seq_len 384 --label_len 0 --pred_len 96 --enc_in 7 --des Exp --d_model 256 --d_ff 512 --n_heads 4 --itr 1 --num_latents 32 --latent_dim 64 --latent_d_ff 128 --dropout 0.2 --standard 0 --generalized 1 --use_latent 1 --num_latent_blocks 1 --patch_len 12 --separate_ratio 1 --batch_size 128 --train_epochs 50 --warmup_epochs 5 --patience 10 --learning_rate 0.0005 --weight_decay 0.05 --num_workers 4 --checkpoints '/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official_baseline__seed2025__<timestamp>/checkpoints' --gpu 0
```

### Weather

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python -u /mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver/run.py --is_training 1 --root_path /mnt/data/zza/code/timeperceiver_meta_uq/data/weather/ --data_path weather.csv --model_id weather_384_96 --model TimePerceiver --data custom --features M --target OT --freq h --seq_len 384 --label_len 0 --pred_len 96 --enc_in 21 --des Exp --d_model 512 --d_ff 1024 --n_heads 8 --itr 1 --num_latents 32 --latent_dim 128 --latent_d_ff 256 --dropout 0.2 --standard 0 --generalized 1 --use_latent 1 --num_latent_blocks 1 --patch_len 24 --separate_ratio 1 --batch_size 128 --train_epochs 100 --warmup_epochs 5 --patience 10 --learning_rate 0.0001 --weight_decay 0.05 --num_workers 4 --checkpoints '/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__official_baseline__seed2025__<timestamp>/checkpoints' --gpu 0
```

### Electricity

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python -u /mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver/run.py --is_training 1 --root_path /mnt/data/zza/code/timeperceiver_meta_uq/data/electricity/ --data_path electricity.csv --model_id electricity_384_96 --model TimePerceiver --data custom --features M --target OT --freq h --seq_len 384 --label_len 0 --pred_len 96 --enc_in 321 --des Exp --d_model 512 --d_ff 1024 --n_heads 8 --itr 1 --num_latents 32 --latent_dim 128 --latent_d_ff 256 --dropout 0.2 --standard 0 --generalized 1 --use_latent 1 --num_latent_blocks 1 --patch_len 24 --separate_ratio 0 --batch_size 64 --train_epochs 100 --warmup_epochs 5 --patience 10 --learning_rate 0.0005 --weight_decay 0.05 --num_workers 4 --checkpoints '/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__official_baseline__seed2025__<timestamp>/checkpoints' --gpu 0
```
