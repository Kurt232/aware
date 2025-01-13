#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPUS="6"

ROOT=$1
MODEL="w_clip_sg"
SETTING_ID=1
PHASE="all"
MARK=""

CONFIGS=$2
LOAD_PATH="${ROOT}/pretrain/${MODEL}${MARK}/checkpoint-399.pth"

if [[ $3 =~ ^[0-9]+$ ]]; then
    MASTER_PORT=$(( $3 + 30 ))
else
    echo "Error: The third argument must be a number."
    exit 1
fi
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

# Count total number of tasks
TASK_LEN=$(ls $CONFIGS/*.yaml | wc -l)

# Initialize loop index
CURRENT_IDX=0

# read all configs from `data/benchmark/`
for DATA_CONFIG in $CONFIGS/*.yaml; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))

    FLAG=$(basename ${DATA_CONFIG%.yaml})
    TRAIN_DIR="${ROOT}/ft_edge/${MODEL}${MARK}/${MODEL}_${FLAG}"
    
    # if exists TRAIN_DIR, skip
    if [ -d "$TRAIN_DIR" ]; then
        continue
    fi
    mkdir -p "$TRAIN_DIR"

    echo "Task: $CURRENT_IDX/$TASK_LEN"
    echo "Data config: $DATA_CONFIG"
    echo "Output directory: $TRAIN_DIR"

    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
        train.py --data_config "$DATA_CONFIG" --batch_size 32 \
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
    
    OUTPUT_DIR="${ROOT}/result/ft_edge/${MODEL}${MARK}/${MODEL}_${FLAG}"
    mkdir -p "$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES="$GPUS" python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" --enable_aware --enable_cross > "${OUTPUT_DIR}/output.log"
    CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"
done