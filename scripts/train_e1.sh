#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="0,1,2,3,4,5,6,7"

ROOT="/data/wjdu/aware"
MODEL="w_aware_lr"
SETTING_ID=1
PHASE="all"
MARK="_1"

MASTER_PORT=2233
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))
CONFIGS="data/train"
# LOAD_PATH="${ROOT}/pretrain/UniTS_HEAD_1_12292347/checkpoint-399.pth"

DATA_CONFIG="data/train/s_all.yaml"
FLAG=$(basename ${DATA_CONFIG%.yaml})
TRAIN_DIR="${ROOT}/output/${MODEL}${MARK}/${MODEL}_${FLAG}"
OUTPUT_DIR="${ROOT}/result/${MODEL}${MARK}/${MODEL}_${FLAG}"

# if exists TRAIN_DIR, skip
# if [ -d "$TRAIN_DIR" ]; then
#     continue
# fi
mkdir -p "$TRAIN_DIR"

echo "Data config: $DATA_CONFIG"
echo "Output directory: $TRAIN_DIR"
# if file exists LOAD_PATH, echo fine-tuning
if [ -f "$LOAD_PATH" ]; then
    echo "Fine-tuning"
fi

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
    train.py --data_config "$DATA_CONFIG" --batch_size 512 \
    --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
    --load_path "$LOAD_PATH" \
    --output_dir "$TRAIN_DIR" \
    --seed 42 \
    --setting_id $SETTING_ID \
    --enable_aware \
    --phase $PHASE \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 3 \
    --patch_len 8 \
    --stride 8 \
    --dropout 0.1 \
    --prompt_num 10 \
    > "$TRAIN_DIR"/output.log

DATA_CONFIG="data/eval/all.yaml"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" > "${OUTPUT_DIR}/output.log"
CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"