ROOT="/data/wjdu/aware"
MODEL="UniTS_HEAD"
SETTING_ID=0
MARK="_cls"

DATA_CONFIG="data/3datasets/m_th_c.yaml"
LOAD_PATH="${ROOT}/pretrain/UniTS_HEAD_12291627/checkpoint-399.pth"

TRAIN_DIR="${ROOT}/output/${MODEL}${MARK}/${MODEL}_m_th_c"
mkdir -p "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES="0" python -u -m debugpy --wait-for-client --listen localhost:5678 -m torch.distributed.launch --nproc_per_node=1 --master_port=3334 --use_env \
        train.py --data_config "$DATA_CONFIG" --batch_size 1024 \
        --epochs 40 --warmup_epochs 10 --blr 1e-4 --min_lr 1e-6 --weight_decay 5e-6 \
        --load_path "$LOAD_PATH" \
        --output_dir "$TRAIN_DIR" \
        --seed 42 \
        --setting_id $SETTING_ID \
        --phase cls \
        --d_model 256 \
        --n_heads 8 \
        --e_layers 3 \
        --patch_len 8 \
        --stride 8 \
        --dropout 0.1 \
        --prompt_num 10
