#!/bin/bash

set -euo pipefail

DIRECTORY="/home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1"

for config in configs/classic/*.json; do
    name=$(basename "$config" .json)
    echo "Running training on classic model: $name"

    uv run python -m capstone.cli --verbose info classic train \
        --directory "$DIRECTORY" \
        --config "$config" \
        --output "artifacts/classic_${name}.joblib"
done