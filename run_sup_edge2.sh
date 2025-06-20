#!/usr/bin/env bash
set -e

ROOT=$1

MASTER_PORT=5300
OFFSET=6
MODEL="clr"
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
echo "Running sup_edge_w.sh"
GPUS="$(( (CURRENT_IDX + OFFSET) % 8 ))"
bash scripts/sup_edge_w2.sh "$ROOT" "w_${MODEL}" $((MASTER_PORT + 10 * $CURRENT_IDX)) $GPUS &
# Store the process ID
pids+=($!)
CURRENT_IDX=$((CURRENT_IDX + 1))

echo "Running sup_edge_wo.sh"
GPUS="$(( (CURRENT_IDX + OFFSET) % 8 ))"
bash scripts/sup_edge_wo2.sh "$ROOT" "wo_${MODEL}" $((MASTER_PORT + 10 * $CURRENT_IDX)) $GPUS &
# Store the process ID
pids+=($!)
CURRENT_IDX=$((CURRENT_IDX + 1))

wait