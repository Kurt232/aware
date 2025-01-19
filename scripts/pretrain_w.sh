#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/hal/0118"
MODEL="w_ae"
SETTING_ID=1
# FLAG="${SETTING_ID}"

DATA_CONFIG="data/aware.yaml"
# TIMESTAMP="_"$(date +%m%d%H%M)
TRAIN_DIR="${ROOT}/pretrain/${MODEL}${FLAG}${TIMESTAMP}"

mkdir -p "$TRAIN_DIR"

GPUS="4,5"
MASTER_PORT=2433
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

mkdir -p "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
    aware_train.py --data_config "$DATA_CONFIG" \
    --batch_size 1536 \
    --epochs 400 \
    --warmup_epochs 10 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --weight_decay 5e-6 \
    --output_dir "$TRAIN_DIR" \
    --setting_id $SETTING_ID \
    --enable_aware \
    --seed 42 \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 3 \
    --patch_len 16 \
    --stride 16 \
    --dropout 0.1 \
    --prompt_num 10 \
    --right_prob 0.2 \
    --min_mask_ratio 0.3 \
    --max_mask_ratio 0.6 \
    > "$TRAIN_DIR/output.log"

# setting_id 0: no augmentation, 1: 1 round, 2: 2 rounds, 3: 5 rounds