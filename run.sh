#!/usr/bin/env bash
set -euo pipefail

docker run --gpus all -it --rm \
  -p 6000:6000 \
  -v "$(pwd)/checkpoints":/workspace/sam-3d-objects/checkpoints \
  --shm-size=16g \
  sam-3d-objects:latest
