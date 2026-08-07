#!/bin/bash

uv run python -m capstone.cli classic train \
	--directory /home/masimms/.cache/kagglehub/datasets/nitishabharathi/email-spam-dataset/versions/1 \
	--config configs/classic.json \
	--output artifacts/classic_combined_cli.joblib
