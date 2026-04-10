#!/bin/bash

set -euo pipefail

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CONFIG_PATH="${1:-config/path_selector_bird_dev_7B.yaml}"

echo "=========================================="
echo "Running Path-Selector Guided Mode"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python -m alphasql.runner.path_selector_runner "${CONFIG_PATH}"

echo "=========================================="
echo "Path-Selector Guided Mode Completed!"
echo "=========================================="
