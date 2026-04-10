#!/bin/bash

set -euo pipefail

start_time=$(date +%s)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG_PATH="${1:-config/llm_nodescore_bird.yaml}"

echo "=========================================="
echo "Running LLM-Guided NodeScore Mode"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python -m alphasql.runner.llm_guided_nodescore_runner "${CONFIG_PATH}"

echo "=========================================="
echo "LLM-Guided NodeScore Mode Completed!"
echo "=========================================="

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
