set -e

ROOT=$1
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

bash scripts/sup_edge_w1.sh "$ROOT" &
pids+=($!)
bash scripts/sup_edge_wo1.sh "$ROOT" &
pids+=($!)

wait