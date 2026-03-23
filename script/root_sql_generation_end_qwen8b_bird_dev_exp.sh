#!/bin/bash

start_time=$(date +%s)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Running Fixed Path Mode: ROOT -> SQL_GENERATION -> END"
echo "Config: config/root_sql_generation_end_qwen8b_bird_dev.yaml"
echo "=========================================="

python -m alphasql.runner.root_sql_generation_end_runner config/root_sql_generation_end_qwen8b_bird_dev.yaml

echo "=========================================="
echo "Fixed Path Mode Completed!"
echo "=========================================="

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
