#!/bin/bash

set -euo pipefail

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

# Download data and set KAGGLE_DIRECTORY
source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

for config in configs/dnn/*.json; do
    name=$(basename "$config" .json)
    echo "Running training on DNN model: $name"

    OUTPUT="artifacts/dnn_${name}.keras"

    if [ ! -f ${OUTPUT} ]; then
        uv run python -m capstone.cli --verbose info dnn train \
            --directory "${KAGGLE_DIRECTORY}" \
            --config "$config" \
            --output "artifacts/dnn_${name}.keras" \
            2>&1 | tee "${OUTPUT}.log"
    else
        echo "Output file ${OUTPUT} exists; skipping"
    fi
    
done

echo "Running training on DNN model baseline (forced CPU_ONLY)"
if [ ! -f "artifacts/classic_baseline_nocpu.keras" ]; then
    GPU_MODE=force_cpu uv run python -m capstone.cli --verbose info dnn train \
            --directory "${KAGGLE_DIRECTORY}" \
            --config ./configs/dnn/baseline.json \
            --output "artifacts/dnn_baseline_nocpu.keras" \
            2>&1 | tee "artifacts/dnn_baseline_nocpu.keras.log"
else
    echo "Output file artifacts/dnn__baseline_nocpu.keras exists; skipping"
fi