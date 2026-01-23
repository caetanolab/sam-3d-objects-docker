#!/usr/bin/env bash
set -euo pipefail

docker run --gpus all -it --rm \
  -p 8000:8000 \
  -v "$(pwd)/checkpoints":/workspace/sam-3d-objects/checkpoints \
  --shm-size=16g \
  sam-3d-objects:latest \
  python /workspace/sam-3d-objects/server.py
