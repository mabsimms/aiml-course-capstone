#!/bin/bash

#MODEL_MANIFEST_PATH=artifacts/dnn_baseline.manifest.json \
#      uv run uvicorn capstone.app:app --host 127.0.0.1 --port 8001

#MODEL_MANIFEST_PATH=artifacts/classic_baseline.manifest.json \
#      uv run uvicorn capstone.app:app --host 127.0.0.1 --port 8001

MODEL_MANIFEST_PATH=artifacts/dnn_combined_cli.manifest.json \
      uv run uvicorn capstone.app:app --host 127.0.0.1 --port 8001
