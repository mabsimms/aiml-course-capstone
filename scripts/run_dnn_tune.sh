#!/bin/bash

set -euo pipefail

DIRECTORY="/home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1"
LOG_FILE="artifacts/tuner/dnn_search.log"

mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m capstone.cli --verbose info dnn tune \
    --directory "$DIRECTORY" \
    --config configs/dnn/search_space.json \
    --max-trials 15 \
    --epochs 15 \
    --tuner-dir artifacts/tuner/dnn_search \
    --output configs/dnn/tuned.json \
    2>&1 | tee "$LOG_FILE"