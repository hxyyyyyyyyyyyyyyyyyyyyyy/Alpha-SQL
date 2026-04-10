#!/bin/bash

set -euo pipefail

CONFIG_PATH="${1:-config/full_selector_bird_dev_7B.yaml}"

echo "=========================================="
echo "Running Full-Selector Mode"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python -m alphasql.runner.full_selector_runner "${CONFIG_PATH}"

echo "=========================================="
echo "Full-Selector Mode Completed!"
echo "=========================================="
