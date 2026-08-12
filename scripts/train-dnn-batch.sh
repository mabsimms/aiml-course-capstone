#!/bin/bash

set -euo pipefail

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

# Download data and set KAGGLE_DIRECTORY
source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

for config in configs/dnn/*.json; do
    name=$(basename "$config" .json)
    echo "Running training on DNN model: $name"

    OUTPUT="artifacts/dnn_${name}"

    if [ ! -f ${OUTPUT}.metrics.json ]; then
        uv run python -m capstone.cli --verbose info dnn train \
            --directory "${KAGGLE_DIRECTORY}" \
            --config "$config" \
            --output "${OUTPUT}" \
            2>&1 | tee "${OUTPUT}.log"
    else
        echo "Output file ${OUTPUT}.metrics.json exists; skipping"
    fi
    
done

echo "Running training on DNN model baseline (forced CPU_ONLY)"
OUTPUT="artifacts/dnn_baseline_nocpu"

if [ ! -f "${OUTPUT}.metrics.json" ]; then
    GPU_MODE=force_cpu uv run python -m capstone.cli --verbose info dnn train \
            --directory "${KAGGLE_DIRECTORY}" \
            --config ./configs/dnn/baseline.json \
            --output "${OUTPUT}" \
            2>&1 | tee "${OUTPUT}.log"
else
    echo "Output file ${OUTPUT}.metrics.json exists; skipping"
fi