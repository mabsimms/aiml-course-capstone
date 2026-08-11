#!/bin/bash

set -euo pipefail

# Require clean repo
bash "$(dirname "${BASH_SOURCE[0]}")/check-git.sh" || exit 1

# Download data and set KAGGLE_DIRECTORY
source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

uv run python -m capstone.cli --verbose info dnn train \
	--directory "${KAGGLE_DIRECTORY}" \
	--config configs/dnn/baseline.json \
	--output artifacts/dnn_combined_cli \
	--epochs 15 \
	2>&1 | tee artifacts/dnn_combined_cli.log 
