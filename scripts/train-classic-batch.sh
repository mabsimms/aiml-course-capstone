#!/bin/bash

set -euo pipefail

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

# Download data and set KAGGLE_DIRECTORY
source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

for config in configs/classic/*.json; do
    name=$(basename "$config" .json)
    echo "Running training on classic model: $name"

    uv run python -m capstone.cli --verbose info classic train \
        --directory "$KAGGLE_DIRECTORY" \
        --config "$config" \
        --output "artifacts/classic_${name}.joblib"
        2>&1 | tee artifacts/classic_${name}.log
done