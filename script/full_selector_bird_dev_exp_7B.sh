#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Running Full-Selector Mode"
echo "Config: config/full_selector_bird_dev_7B.yaml"
echo "=========================================="

python -m alphasql.runner.full_selector_runner config/full_selector_bird_dev_7B.yaml

echo "=========================================="
echo "Full-Selector Mode Completed!"
echo "=========================================="
