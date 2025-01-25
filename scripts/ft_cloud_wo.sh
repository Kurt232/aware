#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT=$1
CONFIGS="data/ft_cloud"
MODEL=$2
SETTING_ID=1
PHASE="all"
MARK=""

LOAD_PATH="${ROOT}/pretrain/${MODEL}${MARK}/checkpoint-199.pth"

MASTER_PORT=$3
GPUS=$4
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

TASK_LEN=$(ls $CONFIGS/*.yaml | wc -l)
CURRENT_IDX=0

for DATA_CONFIG in $CONFIGS/*.yaml; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))

    if [[ "$(basename "$DATA_CONFIG")" =~ ^_ ]]; then
        echo "Skip $DATA_CONFIG"
        continue
    fi

    FLAG=$(basename ${DATA_CONFIG%.yaml})
    TRAIN_DIR="${ROOT}/ft_cloud/${MODEL}${MARK}_${FLAG}"

    # if [ -d "$TRAIN_DIR" ]; then
    #     echo "$TRAIN_DIR exists, skip"
    #     continue
    # fi
    
    mkdir -p "$TRAIN_DIR"

    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
        train.py --data_config "$DATA_CONFIG" --batch_size 512 \
        --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
        --load_path "$LOAD_PATH" \
        --output_dir "$TRAIN_DIR" \
        --seed 42 \
        --setting_id $SETTING_ID \
        --phase $PHASE \
        --d_model 128 \
        --n_heads 4 \
        --e_layers 3 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.1 \
        > "$TRAIN_DIR"/output.log

    OUTPUT_DIR="${ROOT}/result/ft_cloud/${MODEL}${MARK}_${FLAG}"
    mkdir -p "$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES="$GPUS" python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" --enable_cross > "${OUTPUT_DIR}/output.log"
    CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"
done