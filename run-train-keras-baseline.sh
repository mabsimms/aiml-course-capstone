#!/bin/bash

uv run python -m capstone.cli --verbose info dnn train \
	--directory /home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1 \
	--config configs/dnn/baseline.json \
	--output artifacts/dnn_combined_cli.keras \
	--epochs 15 \
	2>&1 | tee artifacts/dnn_combined_cli.keras.log \
