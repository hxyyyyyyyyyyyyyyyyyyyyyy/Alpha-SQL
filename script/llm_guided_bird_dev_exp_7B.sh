#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Running LLM-Guided Mode (No MCTS)"
echo "Config: config/llm_guided_bird_dev.yaml"
echo "=========================================="

python -m alphasql.runner.llm_guided_runner config/llm_guided_bird_dev_7B.yaml

echo "=========================================="
echo "LLM-Guided Mode Completed!"
echo "=========================================="
