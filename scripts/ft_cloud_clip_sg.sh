#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="3"
MASTER_PORT=2242
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

# w_clip
ROOT=$1
MODEL="w_clip_sg"
SETTING_ID=1
PHASE="all"
MARK=""

LOAD_PATH="${ROOT}/pretrain/${MODEL}${MARK}/checkpoint-399.pth"
DATA_CONFIG=$2

TRAIN_DIR="${ROOT}/ft_cloud/${MODEL}${MARK}"

mkdir -p "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
    train.py --data_config "$DATA_CONFIG" --batch_size 512 \
    --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
    --load_path "$LOAD_PATH" \
    --output_dir "$TRAIN_DIR" \
    --seed 42 \
    --setting_id $SETTING_ID \
    --phase $PHASE \
    --enable_aware \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 3 \
    --patch_len 8 \
    --stride 8 \
    --dropout 0.1 \
    --prompt_num 10 \
    > "$TRAIN_DIR"/output.log


OUTPUT_DIR="${ROOT}/result/ft_cloud/${MODEL}${MARK}"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" --enable_aware > "${OUTPUT_DIR}/output.log"
CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"