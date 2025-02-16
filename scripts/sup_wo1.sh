#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT=$1
CONFIGS=data/intra1
MODEL=$2
SETTING_ID=0
PHASE="all"
MARK=""

MASTER_PORT=$3
GPUS=$4
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))
commit_hash=$(git rev-parse --short HEAD 2>/dev/null)

# LOAD_PATH="/data/wjdu/hal4/pretrain/w/checkpoint-199.pth"

for DATA_CONFIGS in $CONFIGS/*; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))

    if [[ "$(basename "$DATA_CONFIGS")" =~ ^_ ]]; then
        echo "Skip $DATA_CONFIGS"
        continue
    fi

    DATA_FLAG=$(basename ${DATA_CONFIGS})

    echo "Current data: $DATA_CONFIGS"
    TASK_LEN=$(ls $DATA_CONFIGS/*.yaml | wc -l)
    CURRENT_IDX=0
    
    for DATA_CONFIG in $DATA_CONFIGS/*.yaml; do
        # Increment loop index
        CURRENT_IDX=$((CURRENT_IDX + 1))

        FLAG=$(basename ${DATA_CONFIG%.yaml})
        if [[ "$(basename "$FLAG")" =~ ^_ ]]; then
            echo "Skip $FLAG"
            continue
        fi

        TRAIN_DIR="${ROOT}/sup1/${MODEL}${MARK}_${DATA_FLAG}/${MODEL}_${FLAG}"
        # if [ -d "$TRAIN_DIR" ]; then
        #     echo "$TRAIN_DIR exists, skip"
        #     continue
        # fi

        mkdir -p "$TRAIN_DIR"

        echo "Task: $CURRENT_IDX/$TASK_LEN"
        echo "Data config: $DATA_CONFIG"
        echo "Output directory: $TRAIN_DIR"

        CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
            train.py --data_config "$DATA_CONFIG" --batch_size 512 \
            --epochs 80 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-5 --weight_decay 5e-6 \
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
            --commit $commit_hash \
            > "$TRAIN_DIR"/output.log
        
        OUTPUT_DIR="${ROOT}/result/sup1/${MODEL}${MARK}_${DATA_FLAG}/${MODEL}_${FLAG}"
        mkdir -p "$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES="$GPUS" python infer1.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" > "${OUTPUT_DIR}/output.log"
        CUDA_VISIBLE_DEVICES="$GPUS" python eval.py "$OUTPUT_DIR" > "${OUTPUT_DIR}/output_still.log"
    done
done