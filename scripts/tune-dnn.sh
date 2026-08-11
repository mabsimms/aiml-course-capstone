#!/bin/bash

set -euo pipefail

DIRECTORY="/home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1"
LOG_FILE="artifacts/tuner/dnn_search.log"

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1


mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "artifacts/tuner/dnn_search"

uv run python -m capstone.cli --verbose info dnn tune \
    --directory "$DIRECTORY" \
    --config configs/dnn_search_space.json \
    --max-trials 20 \
    --epochs 15 \
    --tuner-dir artifacts/tuner/dnn_search \
    --output artifacts/dnn_tuned_arch.json \
    2>&1 | tee "$LOG_FILE"
