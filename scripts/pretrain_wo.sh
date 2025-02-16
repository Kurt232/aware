#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/hal4/"
MODEL="wo"
SETTING_ID=0
# FLAG="${SETTING_ID}"

GPUS="0,1,3,4,5,6,7"
MASTER_PORT=4333
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

DATA_CONFIG="data/aware.yaml"

# TIMESTAMP="_"$(date +%m%d%H%M)
TRAIN_DIR="${ROOT}/pretrain/${MODEL}${TIMESTAMP}"
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
    --d_model 256 \
    --n_heads 8 \
    --e_layers 3 \
    --patch_len 8 \
    --stride 8 \
    --dropout 0.1 \
    > "$TRAIN_DIR/output.log"