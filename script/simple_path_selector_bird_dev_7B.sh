#!/bin/bash

set -euo pipefail

CONFIG_PATH="${1:-config/simple_path_selector_bird_dev_7B.yaml}"

echo "=========================================="
echo "Running Simple-Selector Guided Mode"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python -m alphasql.runner.path_selector_runner "${CONFIG_PATH}"

echo "=========================================="
echo "Simple-Selector Guided Mode Completed!"
echo "=========================================="
