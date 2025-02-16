#!/usr/bin/env bash
set -e

ROOT=$1

MASTER_PORT=4200
OFFSET=6
MODEL="loc"
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
echo "Running sup_w1.sh"
GPUS="$(( (CURRENT_IDX + OFFSET) % 8 ))"
bash scripts/sup_w1.sh "$ROOT" "w_${MODEL}" $((MASTER_PORT + 10 * $CURRENT_IDX)) $GPUS &
# Store the process ID
pids+=($!)
CURRENT_IDX=$((CURRENT_IDX + 1))

echo "Running sup_wo1.sh"
GPUS="$(( (CURRENT_IDX + OFFSET) % 8 ))"
bash scripts/sup_wo1.sh "$ROOT" "wo_${MODEL}" $((MASTER_PORT + 10 * $CURRENT_IDX)) $GPUS &
# Store the process ID
pids+=($!)
CURRENT_IDX=$((CURRENT_IDX + 1))

wait