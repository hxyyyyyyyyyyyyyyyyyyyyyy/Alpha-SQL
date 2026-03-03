#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Running Path-Selector Guided Mode"
echo "Config: config/path_selector_bird_dev_7B.yaml"
echo "=========================================="

python -m alphasql.runner.path_selector_runner config/path_selector_bird_dev_7B.yaml

echo "=========================================="
echo "Path-Selector Guided Mode Completed!"
echo "=========================================="
