#! /bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v conda > /dev/null 2>&1; then
	echo "[ERROR] conda command not found. Please install/initialize conda first."
	exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate alphasql

if ! python -c "import openai" > /dev/null 2>&1; then
	echo "[INFO] openai dependency missing in env 'alphasql', installing..."
	python -m pip install openai
fi

RESULTS_ROOT="${1:-results}"
DETAIL="${2:-${DETAIL:-0}}"

DB_ROOT_DIR="data/bird/dev/dev_databases"
GROUND_TRUTH_PATH="data/bird/dev/dev.sql"

PROCESS_NUM="${PROCESS_NUM:-8}"
META_TIME_OUT="${META_TIME_OUT:-30.0}"
ANALYZE_WORKERS="${ANALYZE_WORKERS:-8}"

mapfile -t RESULT_DIRS < <(find "$RESULTS_ROOT" -type d -path "*/bird/dev" | sort)

if [ ${#RESULT_DIRS[@]} -eq 0 ]; then
	echo "No result folders found under: $RESULTS_ROOT"
	exit 0
fi

echo "Found ${#RESULT_DIRS[@]} folders to process under: $RESULTS_ROOT"

echo "[Step 1] Run analyze_origin_rollouts.py"
for RESULTS_DIR in "${RESULT_DIRS[@]}"; do
	if ! compgen -G "${RESULTS_DIR}/*.pkl" > /dev/null; then
		echo "Skip ${RESULTS_DIR}: no .pkl files"
		continue
	fi

	OUTPUT_BIRD_DIR="$(dirname "$RESULTS_DIR")"
	ANALYSIS_DIR="${OUTPUT_BIRD_DIR}/rollout_analysis_points"
	if [ "$DETAIL" = "1" ] || [ "$DETAIL" = "true" ] || [ "$DETAIL" = "TRUE" ]; then
		ANALYSIS_DIR="${ANALYSIS_DIR}_detail"
	fi

	echo "Analyze: ${RESULTS_DIR} -> ${ANALYSIS_DIR}"
	DETAIL_FLAG=()
	if [ "$DETAIL" = "1" ] || [ "$DETAIL" = "true" ] || [ "$DETAIL" = "TRUE" ]; then
		DETAIL_FLAG=(--detail)
	fi
	python tools/analyze_origin_rollouts.py \
		--results_dir "$RESULTS_DIR" \
		--ground_truth_path "$GROUND_TRUTH_PATH" \
		--db_root_path "$DB_ROOT_DIR" \
		--timeout "$META_TIME_OUT" \
		--num_workers "$ANALYZE_WORKERS" \
		--output_dir "$ANALYSIS_DIR" \
		"${DETAIL_FLAG[@]}"
done

echo "[Step 2] Run sql selection"
bash tools/sql_selection.sh "$RESULTS_ROOT"

echo "[Step 3] Collect fail points after selection"
for RESULTS_DIR in "${RESULT_DIRS[@]}"; do
	if ! compgen -G "${RESULTS_DIR}/*.pkl" > /dev/null; then
		continue
	fi

	OUTPUT_BIRD_DIR="$(dirname "$RESULTS_DIR")"
	PRED_SQL_PATH="${OUTPUT_BIRD_DIR}/pred_sqls.json"
	ANALYSIS_DIR="${OUTPUT_BIRD_DIR}/rollout_analysis_points"
	FAIL_DIR="${OUTPUT_BIRD_DIR}/fail"

	if [ "$DETAIL" = "1" ] || [ "$DETAIL" = "true" ] || [ "$DETAIL" = "TRUE" ]; then
		FAIL_DIR="${FAIL_DIR}_detail"
		ANALYSIS_DIR="${ANALYSIS_DIR}_detail"
	fi

	if [ ! -f "$PRED_SQL_PATH" ]; then
		echo "Skip ${RESULTS_DIR}: ${PRED_SQL_PATH} not found"
		continue
	fi

	python tools/collect_fail_after_selection.py \
		--pred_sql_path "$PRED_SQL_PATH" \
		--ground_truth_path "$GROUND_TRUTH_PATH" \
		--db_root_dir "$DB_ROOT_DIR" \
		--analysis_dir "$ANALYSIS_DIR" \
		--fail_dir "$FAIL_DIR" \
		--timeout "$META_TIME_OUT"
done

echo "Done."
