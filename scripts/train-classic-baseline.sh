#!/bin/bash

set -euo pipefail

DIRECTORY="/home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1"

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

uv run python -m capstone.cli --verbose info classic tune \
    --directory "$DIRECTORY" \
    --config configs/classic_search_space.json \
    --output artifacts/classic_tuned.joblib \
    --cv 5 \
    --n-jobs -1 \
    2>&1 | tee artifacts/classic_tuned.log
