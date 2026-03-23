#!/bin/bash

start_time=$(date +%s)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Running LLM-Guided Mode (No MCTS)"
echo "Config: config/llm_guided_bird_dev_8B.yaml"
echo "=========================================="

python -m alphasql.runner.llm_guided_runner config/llm_guided_bird_dev_7B.yaml

echo "=========================================="
echo "LLM-Guided Mode Completed!"
echo "=========================================="

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
