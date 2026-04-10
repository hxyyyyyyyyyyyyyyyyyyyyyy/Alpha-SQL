#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

RESULTS_DIR="results/full_selector/Qwen2.5-Coder-7B-Instruct/bird/dev"
GROUND_TRUTH_PATH="data/bird/dev/dev.sql"
DIFF_JSON_PATH="data/bird/dev/dev.json"
DB_ROOT_DIR="data/bird/dev/dev_databases"
OUTPUT_PATH="results/full_selector_path_template_accuracy.json"
PROCESS_NUM="${PROCESS_NUM:-16}"
META_TIME_OUT="${META_TIME_OUT:-30.0}"
TOP_K="${TOP_K:-20}"

echo "=========================================="
echo "Path Template Accuracy (Full Selector)"
echo "results_dir: ${RESULTS_DIR}"
echo "output:      ${OUTPUT_PATH}"
echo "=========================================="

python -m alphasql.runner.path_template_accuracy \
  --results_dir "${RESULTS_DIR}" \
  --ground_truth_path "${GROUND_TRUTH_PATH}" \
  --diff_json_path "${DIFF_JSON_PATH}" \
  --db_root_dir "${DB_ROOT_DIR}" \
  --output_path "${OUTPUT_PATH}" \
  --process_num "${PROCESS_NUM}" \
  --meta_time_out "${META_TIME_OUT}" \
  --top_k "${TOP_K}"

echo "=========================================="
echo "Done"
echo "=========================================="
