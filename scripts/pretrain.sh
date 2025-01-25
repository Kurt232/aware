#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/hal1/0124"
MODEL="wo_aware"
SETTING_ID=1
# FLAG="${SETTING_ID}"

DATA_CONFIG="data/aware.yaml"
# TIMESTAMP="_"$(date +%m%d%H%M)
TRAIN_DIR="${ROOT}/pretrain/${MODEL}${FLAG}${TIMESTAMP}"

mkdir -p "$TRAIN_DIR"

GPUS="0,1,2,3,4,5,6,7"
MASTER_PORT=2633
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

mkdir -p "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
    aware_train.py --data_config "$DATA_CONFIG" \
    --batch_size 768 \
    --epochs 200 \
    --warmup_epochs 10 \
    --lr 1e-3 \
    --min_lr 1e-5 \
    --weight_decay 5e-6 \
    --output_dir "$TRAIN_DIR" \
    --setting_id $SETTING_ID \
    --seed 42 \
    --d_model 128 \
    --n_heads 4 \
    --e_layers 3 \
    --patch_len 8 \
    --stride 8 \
    --dropout 0.1 \
    --min_mask_ratio 0.2 \
    --max_mask_ratio 0.4 \
    --lambda_recon 0.5 \
    --temperature 0.07 \
    --is_masked \
    > "$TRAIN_DIR/output.log"

# setting_id 0: no augmentation, 1: 1 round, 2: 2 rounds, 3: 5 rounds