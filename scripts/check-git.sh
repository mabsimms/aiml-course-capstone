#!/bin/bash

# Refuse to proceed if the working tree has uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: uncommitted git changes present.  Commit or stash before running"
    git status --short >&2
    exit 1
fi