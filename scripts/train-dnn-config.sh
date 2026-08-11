#!/bin/bash

set -euo pipefail

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

# Download data and set KAGGLE_DIRECTORY
source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

CONFIG=${1:?Usage: train-dnn-config.sh <config.json> [output]}
NAME=$(basename "$CONFIG" .json)
OUTPUT=${2:-artifacts/dnn_${NAME}}
  
uv run python -m capstone.cli --verbose info dnn train \
    --directory "$KAGGLE_DIRECTORY" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    2>&1 | tee "${OUTPUT}.log"

