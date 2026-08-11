#!/bin/bash

TUNER_RESULTS=${1:-artifacts/dnn_tuned_arch.json}
NAME=${2:-tuned_from_search}

if [ ! -f "$TUNER_RESULTS" ]; then
    echo "ERROR: tuner results file not found: $TUNER_RESULTS" >&2
    echo "Run scripts/tune-dnn.sh first, or pass the path to a dnn tune output JSON as the first argument." >&2
    exit 1
fi  
    
CONFIG="configs/dnn/${NAME}.json"
cp "$TUNER_RESULTS" "$CONFIG"

echo "Wrote tuner-selected hyperparameters to $CONFIG:"
cat "$CONFIG"
echo
echo "Next steps:"
echo "  git add $CONFIG && git commit -m 'Add config for best dnn tune result'"
echo "  ./scripts/train-dnn-config.sh $CONFIG"

