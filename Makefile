IMAGE_NAME ?= capstone-serving
GIT_SHA := $(shell git rev-parse --short HEAD)

.PHONY: build built-force build-clean clean

build:
	@bash scripts/check-git.sh
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE_NAME):$(GIT_SHA) -t $(IMAGE_NAME):latest .
	@echo "Built $(IMAGE_NAME):$(GIT_SHA) (tagged latest)"

# Bypass git-clean check, for development builds
build-dev:
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE_NAME):$(GIT_SHA) -t $(IMAGE_NAME):latest .
	@echo "Built $(IMAGE_NAME):$(GIT_SHA) (tagged latest) - WARNING: developer build"

clean:
	docker rmi $(IMAGE_NAME):latest 2>/dev/null || true