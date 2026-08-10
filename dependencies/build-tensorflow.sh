#!/bin/bash

# Build Tensorflow (this takes about 2-4 hours)
bazel build //tensorflow/tools/pip_package:wheel   \
	--repo_env=WHEEL_NAME=tensorflow   \
	--repo_env=ML_WHEEL_VERSION_SUFFIX=+rtx5090.cuda13   \
	--config=cuda_wheel   \
	--config=opt   \
	--copt=-march=native \
	--host_copt=-march=native \
	-c opt   \
	--repo_env=HERMETIC_CUDA_VERSION=13.0.0   \
	--repo_env=HERMETIC_CUDNN_VERSION=9.12.0   \
	--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=12.0   \
	--@local_config_cuda//:cuda_compiler=nvcc


