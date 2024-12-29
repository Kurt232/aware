#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/aware"
MODEL="UniTS_HEAD"
SETTING_ID=1
FLAG="_${SETTING_ID}"

DATA_CONFIG="data/config.yaml"
TIMESTAMP=$(date +%m%d%H%M)
TRAIN_DIR="${ROOT}/pretrain/${MODEL}${FLAG}_${TIMESTAMP}"

mkdir -p "$TRAIN_DIR"

GPUS="4,5,6,7"
MASTER_PORT=4233
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

mkdir -p "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
    pretrain.py --data_config "$DATA_CONFIG" \
    --batch_size 1024 \
    --epochs 400 \
    --warmup_epochs 10 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --weight_decay 5e-6 \
    --output_dir "$TRAIN_DIR" \
    --setting_id $SETTING_ID \
    --seed 42 \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 3 \
    --patch_len 8 \
    --stride 8 \
    --dropout 0.1 \
    --prompt_num 10 \
    --right_prob 0.5 \
    --min_mask_ratio 0.5 \
    --max_mask_ratio 0.8 \
    > "$TRAIN_DIR/output.log"

# setting_id 0: no augmentation, 1: 1 round, 2: 2 rounds, 3: 5 rounds