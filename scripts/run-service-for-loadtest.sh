#!/bin/bash

set -euo pipefail

MANIFEST_DIR=${1:-artifacts/baseline}
MANIFEST_NAME=${2:-classic_baseline.manifest.json}
CONTAINER_NAME=${3:-capstone-serving-loadtest}
CPU_LIMIT=${4:-4}
MEMORY_LIMIT=${5:-8g}
TIMEOUT_SECONDS=${6:-60}

# Clean up any leftover container from a previous run
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d -p 8001:8001 \
    --cpus="$CPU_LIMIT" \
    --memory="$MEMORY_LIMIT" \
    --memory-swap="$MEMORY_LIMIT" \
    -e MODEL_MANIFEST_PATH="/models/${MANIFEST_NAME}" \
    -v "$(pwd)/${MANIFEST_DIR}:/models:ro" \
    --name "$CONTAINER_NAME" \
    capstone-serving:latest

echo "Started $CONTAINER_NAME with --cpus=$CPU_LIMIT --memory=$MEMORY_LIMIT"
echo "Waiting for $CONTAINER_NAME to report healthy..."

elapsed=0
until [ "$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null)" = "healthy" ]; do
    if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
        echo "ERROR: $CONTAINER_NAME did not become healthy within ${TIMEOUT_SECONDS}s" >&2
        docker logs "$CONTAINER_NAME" >&2
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

echo "$CONTAINER_NAME is healthy and ready for load testing."
echo
echo "Next steps:"
echo "  ./scripts/run-loadtest.sh configs/loadtest-baseline.json <tag>"
echo "  docker stop $CONTAINER_NAME"
