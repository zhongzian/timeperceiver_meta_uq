#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official_baseline__seed2025__20260701_202833
export PYTHONPATH=/mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=1
/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python -u /mnt/data/zza/code/timeperceiver_meta_uq/upstream/TimePerceiver/run.py --is_training 1 --root_path /mnt/data/zza/code/timeperceiver_meta_uq/data/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_384_96 --model TimePerceiver --data ETTh1 --features M --target OT --freq h --seq_len 384 --label_len 0 --pred_len 96 --enc_in 7 --des Exp --d_model 256 --d_ff 512 --n_heads 4 --itr 1 --num_latents 32 --latent_dim 64 --latent_d_ff 128 --dropout 0.2 --standard 0 --generalized 1 --use_latent 1 --num_latent_blocks 1 --patch_len 12 --separate_ratio 1 --batch_size 128 --train_epochs 50 --warmup_epochs 5 --patience 10 --learning_rate 0.0005 --weight_decay 0.05 --num_workers 4 --checkpoints /mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official_baseline__seed2025__20260701_202833/checkpoints --gpu 1
