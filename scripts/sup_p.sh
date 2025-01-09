#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="0,1,2,3"

ROOT="/data/wjdu/hal"
MODEL="w_aware"
SETTING_ID=1
PHASE="all"
MARK=""
AFFIX=""

MASTER_PORT=3633
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

DATA_CONFIG="data/ft_cloud.yaml"
FLAG=$(basename ${DATA_CONFIG%.yaml})
TRAIN_DIR="${ROOT}/sup_p/${MODEL}${MARK}"

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

OUTPUT_DIR="${ROOT}/result/sup_p/${MODEL}${MARK}"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" --enable_aware > "${OUTPUT_DIR}/output.log"
CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"


MODEL="wo_aware"
TRAIN_DIR="${ROOT}/sup_p/${MODEL}${MARK}"

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
    --phase $PHASE \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 3 \
    --patch_len 8 \
    --stride 8 \
    --dropout 0.1 \
    --prompt_num 10 \
    > "$TRAIN_DIR"/output.log

OUTPUT_DIR="${ROOT}/result/sup_p/${MODEL}${MARK}"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" > "${OUTPUT_DIR}/output.log"
CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"
