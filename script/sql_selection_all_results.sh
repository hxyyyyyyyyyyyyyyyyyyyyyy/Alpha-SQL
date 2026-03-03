#! /bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DB_ROOT_DIR="data/bird/dev/dev_databases"
DATA_MODE="dev"
DIFF_JSON_PATH="data/bird/dev/dev.json"
GROUND_TRUTH_PATH="data/bird/dev/dev.sql"
MODE_GT="gt"
MODE_PREDICT="gpt"

PROCESS_NUM="${PROCESS_NUM:-32}"
NUM_CPUS="${NUM_CPUS:-16}"
META_TIME_OUT="${META_TIME_OUT:-30.0}"
FILL_MISSING_ERROR="${FILL_MISSING_ERROR:-false}"

RESULTS_ROOT="${1:-results}"
SUMMARY_DIR="${RESULTS_ROOT}/summary"

mkdir -p "$SUMMARY_DIR"

mapfile -t RESULT_DIRS < <(find "$RESULTS_ROOT" -type d -path "*/bird/dev" | sort)

if [ ${#RESULT_DIRS[@]} -eq 0 ]; then
    echo "No result folders found under: $RESULTS_ROOT"
    exit 0
fi

echo "Found ${#RESULT_DIRS[@]} folders to process."

for RESULTS_DIR in "${RESULT_DIRS[@]}"; do
    if ! compgen -G "${RESULTS_DIR}/*.pkl" > /dev/null; then
        echo "Skip ${RESULTS_DIR}: no .pkl files"
        continue
    fi

    REL_DIR="${RESULTS_DIR#${RESULTS_ROOT}/}"
    SAFE_NAME="${REL_DIR//\//__}"

    PRED_SQL_PATH="${SUMMARY_DIR}/${SAFE_NAME}_pred_sqls.json"
    EVAL_OUTPUT_PATH="${SUMMARY_DIR}/${SAFE_NAME}_evaluation.txt"

    if [ -f "$PRED_SQL_PATH" ] && [ -f "$EVAL_OUTPUT_PATH" ]; then
        echo "Skipping ${RESULTS_DIR}: Output files already exist."
        continue
    fi

    echo "============================================================"
    echo "Processing: ${RESULTS_DIR}"
    echo "Pred SQL:   ${PRED_SQL_PATH}"
    echo "Eval file:  ${EVAL_OUTPUT_PATH}"

    python -m alphasql.runner.sql_selection \
        --results_dir "$RESULTS_DIR" \
        --db_root_dir "$DB_ROOT_DIR" \
        --process_num "$PROCESS_NUM" \
        --output_path "$PRED_SQL_PATH" \
        --fill_missing_error "$FILL_MISSING_ERROR"

    python3 -u -m alphasql.runner.evaluation \
        --db_root_path "${DB_ROOT_DIR}/" \
        --predicted_sql_path "$PRED_SQL_PATH" \
        --data_mode "$DATA_MODE" \
        --ground_truth_path "$GROUND_TRUTH_PATH" \
        --num_cpus "$NUM_CPUS" \
        --mode_gt "$MODE_GT" \
        --mode_predict "$MODE_PREDICT" \
        --diff_json_path "$DIFF_JSON_PATH" \
        --meta_time_out "$META_TIME_OUT" > "$EVAL_OUTPUT_PATH"

    echo "Done: ${RESULTS_DIR}"
done

echo "All finished. Outputs are in ${SUMMARY_DIR}"