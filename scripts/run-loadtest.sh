#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/resolve-kaggle-directory.sh"

CONFIG=${1:-configs/loadtest-baseline.json}
TAG=${2:?Usage: run_loadtest.sh [config] <tag>}

uv run --group loadtest python "$(dirname "${BASH_SOURCE[0]}")/run-loadtest.py" \
    --config "$CONFIG" --tag "$TAG"
