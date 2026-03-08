#!/bin/bash

set -euo pipefail

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SAMPLE_SIZE=4
EXTRA_ARGS=()
HAS_DESTINATION_DIR=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		-n|--sample-size)
			SAMPLE_SIZE="$2"
			shift 2
			;;
		--sample-size=*)
			SAMPLE_SIZE="${1#*=}"
			shift
			;;
		--destination_dir|--destination-dir)
			HAS_DESTINATION_DIR=true
			EXTRA_ARGS+=("$1" "$2")
			shift 2
			;;
		--destination_dir=*|--destination-dir=*)
			HAS_DESTINATION_DIR=true
			EXTRA_ARGS+=("$1")
			shift
			;;
		*)
			EXTRA_ARGS+=("$1")
			shift
			;;
	esac
done

AUTO_DESTINATION_DIR="results/random_selector${SAMPLE_SIZE}/Qwen2.5-Coder-7B-Instruct/bird/dev"

CMD=(
	python script/random_selector_from_full_selector.py
	--strict_expected_size
	--sample_size "${SAMPLE_SIZE}"
)

if [[ "${HAS_DESTINATION_DIR}" == false ]]; then
	CMD+=(--destination_dir "${AUTO_DESTINATION_DIR}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "=========================================="
echo "Random Selector from Full Selector"
echo "Sample size: ${SAMPLE_SIZE}"
if [[ "${HAS_DESTINATION_DIR}" == false ]]; then
	echo "Destination dir: ${AUTO_DESTINATION_DIR}"
fi
echo "=========================================="

"${CMD[@]}"

echo "=========================================="
echo "Done"
echo "=========================================="
