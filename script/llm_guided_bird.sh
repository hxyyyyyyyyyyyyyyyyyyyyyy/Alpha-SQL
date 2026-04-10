#!/bin/bash

set -euo pipefail

start_time=$(date +%s)

CONFIG_PATH="${1:-config/llm_guided_bird.yaml}"

echo "=========================================="
echo "Running LLM-Guided Mode (No MCTS)"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python -m alphasql.runner.llm_guided_runner "${CONFIG_PATH}"

echo "=========================================="
echo "LLM-Guided Mode Completed!"
echo "=========================================="

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
