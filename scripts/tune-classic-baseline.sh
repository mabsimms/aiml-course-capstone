#!/bin/bash

set -euo pipefail

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

# Download data and set KAGGLE_DIRECTORY
source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

uv run python -m capstone.cli --verbose info classic tune \
    --directory "$KAGGLE_DIRECTORY" \
    --config configs/classic_search_space.json \
    --output artifacts/classic_tuned.joblib \
    --cv 5 \
    --n-jobs -1 \
    2>&1 | tee artifacts/classic_tuned.log
