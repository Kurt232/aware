set -e

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

bash scripts/batch_train.sh "/data/wjdu/hal/0113" &
pids+=($!)
bash scripts/batch_train1.sh "/data/wjdu/hal/0113" &
pids+=($!)

wait