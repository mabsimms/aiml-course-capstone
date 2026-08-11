#!/bin/bash

set -euo pipefail

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

DIRECTORY="/home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1"

for config in configs/classic/*.json; do
    name=$(basename "$config" .json)
    echo "Running training on classic model: $name"

    uv run python -m capstone.cli --verbose info classic train \
        --directory "$DIRECTORY" \
        --config "$config" \
        --output "artifacts/classic_${name}.joblib"
done