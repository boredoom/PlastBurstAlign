#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NCORES="${1:-10}"
if (( $# > 0 )); then
	shift
fi

if [[ ! "$NCORES" =~ ^[1-9][0-9]*$ ]]; then
	echo "Usage: $0 [NCORES] [DATASET ...]" >&2
	echo "NCORES must be a positive integer." >&2
	exit 2
fi

if (( $# > 0 )); then
	DATASETS=("$@")
else
	DATASETS=(
		benchmark_ZhangNNEtAl2025_53angiosperms
		benchmark_YangEtAl2022_Asteraceae
		benchmark_YangEtAl2022_monocots
		benchmark_YangEtAl2022_angiosperms
		benchmark_ZhangNNEtAl2025_13724angiosperms
	)
fi

PYTHON="${PYTHON:-python}"
FAILED_DATASETS=()

for DATASET in "${DATASETS[@]}"; do
	LOG_FILE="${DATASET}.log"
	echo "Working on $DATASET using $NCORES CPUs (log: $LOG_FILE) ..."
	if "$PYTHON" benchmarking_CDS.py "$DATASET" "$NCORES" > "$LOG_FILE" 2>&1; then
		echo "Completed $DATASET."
	else
		STATUS=$?
		echo "Failed $DATASET (exit status $STATUS; see $LOG_FILE)." >&2
		FAILED_DATASETS+=("$DATASET")
	fi
done

if (( ${#FAILED_DATASETS[@]} > 0 )); then
	printf 'Failed datasets: %s\n' "${FAILED_DATASETS[*]}" >&2
	exit 1
fi
