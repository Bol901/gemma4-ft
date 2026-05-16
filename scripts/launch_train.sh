#!/usr/bin/env bash
# Launch multi-GPU LoRA training on 4x A6000 with DeepSpeed ZeRO-2.
#
# 1) Cache slices first:
#    python scripts/preprocess.py --input_jsonl data/train.jsonl \
#        --cache_dir /scratch/slice_cache --n_slices 16 --n_workers 16
# 2) Smoke test the pipeline by overfitting 10 cases:
#    bash scripts/launch_train.sh --overfit
# 3) Full run: bash scripts/launch_train.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/train_config.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
EXTRA_ARGS="$*"

export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_zero2.json \
  training/train.py \
  --config "${CONFIG}" \
  ${EXTRA_ARGS}
