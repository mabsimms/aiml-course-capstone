#!/bin/bash

set -euo pipefail

DIRECTORY="/home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1"

for config in configs/dnn/*.json; do
    name=$(basename "$config" .json)
    echo "Running training on DNN model: $name"

    OUTPUT="artifacts/classic_${name}.keras"

    if [ ! -f ${OUTPUT} ]; then
        uv run python -m capstone.cli --verbose info dnn train \
            --directory "$DIRECTORY" \
            --config "$config" \
            --output "artifacts/classic_${name}.keras" \
            2>&1 | tee "${OUTPUT}.log"
    else
        echo "Output file ${OUTPUT} exists; skipping"
    fi
    
done

echo "Running training on DNN model baseline (forced CPU_ONLY)"
if [ ! -f "artifacts/classic_baseline_nocpu.keras" ]; then
    GPU_MODE=force_cpu uv run python -m capstone.cli --verbose info dnn train \
            --directory "$DIRECTORY" \
            --config ./configs/dnn/baseline.json \
            --output "artifacts/classic_baseline_nocpu.keras" \
            2>&1 | tee "artifacts/classic_baseline_nocpu.keras.log"
else
    echo "Output file artifacts/classic_baseline_nocpu.keras exists; skipping"
fi