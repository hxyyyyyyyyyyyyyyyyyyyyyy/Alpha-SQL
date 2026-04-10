#!/bin/bash

set -euo pipefail

start_time=$(date +%s)

CONFIG_PATH="${1:-config/sql_generation_bird.yaml}"

echo "=========================================="
echo "Running SQL Generation Baseline Mode"
echo "Fixed Path : ROOT -> SQL_GENERATION -> END"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python -m alphasql.runner..root_sql_generation_end_runner "${CONFIG_PATH}"

echo "=========================================="
echo "SQL Generation Baseline Mode Completed!"
echo "=========================================="

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
