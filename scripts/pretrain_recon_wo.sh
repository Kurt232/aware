#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/hal2/0208"
MODEL="wo_recon"
SETTING_ID=0
# FLAG="${SETTING_ID}"
CONF="pretrain"

GPUS="3,4,5,6,7"
MASTER_PORT=2333
NNODE=$(($(echo $GPUS | tr -cd , | wc -c) + 1))

CONFIGS="data/${CONF}"

for DATA_CONFIG in $CONFIGS/*.yaml; do
    # Increment loop index
    CURRENT_IDX=$((CURRENT_IDX + 1))

    if [[ "$(basename "$DATA_CONFIG")" =~ ^_ ]]; then
        echo "Skip $DATA_CONFIG"
        continue
    fi

    FLAG=$(basename ${DATA_CONFIG%.yaml})
    # TIMESTAMP="_"$(date +%m%d%H%M)
    TRAIN_DIR="${ROOT}/${CONF}/${FLAG}/${MODEL}${TIMESTAMP}"
    mkdir -p "$TRAIN_DIR"

    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NNODE --master_port=$MASTER_PORT \
        pretrain.py --data_config "$DATA_CONFIG" \
        --batch_size 1024 \
        --epochs 200 \
        --warmup_epochs 10 \
        --lr 5e-4 \
        --min_lr 1e-5 \
        --weight_decay 5e-6 \
        --output_dir "$TRAIN_DIR" \
        --setting_id $SETTING_ID \
        --seed 42 \
        --d_model 256 \
        --n_heads 4 \
        --e_layers 3 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.1 \
        --min_mask_ratio 0.2 \
        --max_mask_ratio 0.4 \
        > "$TRAIN_DIR/output.log"
done