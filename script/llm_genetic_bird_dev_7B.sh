#!/bin/bash

set -euo pipefail

start_time=$(date +%s)

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG_PATH="${1:-config/llm_genetic_bird_dev_7B.yaml}"

echo "Start running standalone LLM+GA path generation experiment"
echo "Config: ${CONFIG_PATH}"

python -m alphasql.runner.llm_genetic_runner "${CONFIG_PATH}"

echo "Standalone LLM+GA path generation experiment completed"

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"