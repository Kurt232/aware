#!/usr/bin/env bash
set -e

ROOT=$1
GPUS="1"
# check pretrain is in the root directory
if [ ! -d "$ROOT/pretrain" ]; then
    echo "No pretrain directory found in $ROOT"
    exit 1
fi

MODELS="w_ae w_clip w_clip_sg"
MASTER_PORT=6000
# Store background process IDs
pids=()

# Function to kill all background processes
cleanup() {
    echo "Caught interrupt signal, stopping all processes..."
    for pid in "${pids[@]}"; do
        kill $pid 2>/dev/null || true
    done
    exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

CURRENT_IDX=0
for model in $MODELS; do
    echo "Running ft_edge.sh"
    bash scripts/ft_edge.sh "$ROOT" "$model" $((MASTER_PORT + 10 * $CURRENT_IDX)) $GPUS &
    # Store the process ID
    pids+=($!)
    CURRENT_IDX=$((CURRENT_IDX + 1))
done

bash scripts/ft_edge_wo.sh "$ROOT" "base" $((MASTER_PORT + 10 * $CURRENT_IDX)) $GPUS &
wait