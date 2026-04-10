#!/bin/bash

set -euo pipefail

start_time=$(date +%s)

CONFIG_PATH="${1:-config/origin_bird_dev.yaml}"

echo "=========================================="
echo "Running Origin Mode"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python -m alphasql.runner.mcts_runner "${CONFIG_PATH}"

echo "=========================================="
echo "Origin Mode Completed!"
echo "=========================================="

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
