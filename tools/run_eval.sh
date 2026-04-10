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

RESULTS_ROOT="${1:-results/llm_guided24}"

if [ ! -d "$RESULTS_ROOT" ] && [ "$RESULTS_ROOT" = "results/llm_guided24" ] && [ -d "results/llm_guide24" ]; then
    RESULTS_ROOT="results/llm_guide24"
fi

is_step_b_done() {
    local results_dir
    while IFS= read -r results_dir; do
        if ! compgen -G "${results_dir}/*.pkl" > /dev/null; then
            continue
        fi
        local output_bird_dir
        output_bird_dir="$(dirname "$results_dir")"
        if [ ! -f "${output_bird_dir}/pred_sqls.json" ] || [ ! -f "${output_bird_dir}/evaluation.txt" ]; then
            return 1
        fi
    done < <(find "$RESULTS_ROOT" -type d -path "*/bird/dev" | sort)
    return 0
}

is_step_c_done() {
    local results_dir
    while IFS= read -r results_dir; do
        if ! compgen -G "${results_dir}/*.pkl" > /dev/null; then
            continue
        fi
        local output_bird_dir
        output_bird_dir="$(dirname "$results_dir")"
        if [ ! -f "${output_bird_dir}/summary_statistics.md" ]; then
            return 1
        fi
    done < <(find "$RESULTS_ROOT" -type d -path "*/bird/dev" | sort)
    return 0
}

echo "[Step A] Extract subsets from: ${RESULTS_ROOT}"
python tools/subset_extractor.py --results-root "$RESULTS_ROOT"

if is_step_b_done; then
    echo "[Step B] Skip: all tasks are already completed."
else
    echo "[Step B] Run sql selection + evaluation for all subsets"
    bash tools/sql_selection_all_results.sh "$RESULTS_ROOT"
fi

if is_step_c_done; then
    echo "[Step C] Skip: all tasks are already completed."
else
    echo "[Step C] Run summarize_data.py (summary_paths disabled by default)"
    python tools/summarize_data.py --results-dir "$RESULTS_ROOT"
fi

echo "Done."
