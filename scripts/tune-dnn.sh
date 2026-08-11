#!/bin/bash

set -euo pipefail

LOG_FILE="artifacts/tuner/dnn_search.log"

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

# Download data and set KAGGLE_DIRECTORY
source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "artifacts/tuner/dnn_search"

uv run python -m capstone.cli --verbose info dnn tune \
    --directory "$KAGGLE_DIRECTORY" \
    --config configs/dnn_search_space.json \
    --max-trials 20 \
    --epochs 15 \
    --tuner-dir artifacts/tuner/dnn_search \
    --output artifacts/dnn_tuned_arch.json \
    2>&1 | tee "$LOG_FILE"
