#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

db_root_path='data/bird/dev/dev_databases/'
data_mode='dev'
diff_json_path='data/bird/dev/dev.json'
predicted_sql_path_kg="${1:-results/pred_sqls_7B.json}"
ground_truth_path='data/bird/dev/dev.sql'
num_cpus="${NUM_CPUS:-16}"
meta_time_out="${META_TIME_OUT:-30.0}"
mode_gt='gt'
mode_predict='gpt'
output_path="${2:-results/evaluation_results_7B.txt}"

echo "Evaluating SQLs..."
python3 -u -m alphasql.runner.evaluation --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out} > ${output_path}