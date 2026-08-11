#!/bin/bash

# Resolve local directory for email-spam-dataset via kagglehub
_resolve_kaggle() { 
    uv run python -c "import kagglehub; print(kagglehub.dataset_download('nitishabharathi/email-spam-dataset'))"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then    
    set -euo pipefail
    _resolve_kaggle
else
    export KAGGLE_DIRECTORY="$(_resolve_kaggle)"
fi