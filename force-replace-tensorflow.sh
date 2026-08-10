#!/bin/bash

uv cache clean tensorflow
uv sync --reinstall-package tensorflow
uv run python -c "import tensorflow as tf; print(tf.__file__); print(tf.sysconfig.get_build_info())"
