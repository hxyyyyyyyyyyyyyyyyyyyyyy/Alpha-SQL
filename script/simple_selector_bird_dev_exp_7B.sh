#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Running Simple-Selector Guided Mode"
echo "Config: config/simple_path_selector_bird_dev_7B.yaml"
echo "=========================================="

python -m alphasql.runner.simple_path_selector_runner config/simple_path_selector_bird_dev_7B.yaml

echo "=========================================="
echo "Simple-Selector Guided Mode Completed!"
echo "=========================================="
