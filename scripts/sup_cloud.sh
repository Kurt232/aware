#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="4,5,6,7"

ROOT=$1
MODEL="w_sup"
SETTING_ID=1
PHASE="all"
MARK=""

MASTER_PORT=2600
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))
CONFIGS="data/sup"

# Count total number of tasks
TASK_LEN=$(ls $CONFIGS/*.yaml | wc -l)

# Initialize loop index
CURRENT_IDX=0

# read all configs from `data/benchmark/`
for DATA_CONFIG in $CONFIGS/*.yaml; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))

    if [[ "$(basename "$DATA_CONFIG")" =~ ^_ ]]; then
        echo "Skip $DATA_CONFIG"
        continue
    fi
    
    FLAG=$(basename ${DATA_CONFIG%.yaml})
    TRAIN_DIR="${ROOT}/sup/${MODEL}${MARK}_${FLAG}"
    OUTPUT_DIR="${ROOT}/result/sup/${MODEL}${MARK}_${FLAG}"
    
    echo "Task: $CURRENT_IDX/$TASK_LEN"
    echo "Data config: $DATA_CONFIG"
    echo "TRAIN directory: $TRAIN_DIR"
    echo "OUTPUT directory: $OUTPUT_DIR"

    # if exists TRAIN_DIR, skip
    if [ -d "$TRAIN_DIR" ]; then
        echo "TRAIN_DIR exists, skip"
        continue
    fi

    mkdir -p "$TRAIN_DIR"

    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
        train.py --data_config "$DATA_CONFIG" --batch_size 512 \
        --epochs 80 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
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
    
    mkdir -p "$OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES="$GPUS" python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" --enable_aware --enable_cross > "${OUTPUT_DIR}/output.log"
    CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"
done