#!/usr/bin/env bash
set -e


TARGET=$1
ROOT=$2
DATA_CONFIG=$3

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

for script in $(ls scripts/*.sh); do
    if [[ "$script" =~ ^$TARGET.*\.sh$ ]]; then
        echo "Running $script"
        bash $script "$ROOT" "$DATA_CONFIG" &
        # Store the process ID
        pids+=($!)
    fi
done

wait
