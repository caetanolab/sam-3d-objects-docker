# syntax=docker/dockerfile:1

FROM condaforge/miniforge3:latest
SHELL ["/bin/bash", "-lc"]

# ---- system dependencies (OpenMP, X11, OpenGL) ----
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libgomp1 \
    libx11-6 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*
# --------------------------------------------------

WORKDIR /workspace
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git \
 && rm -rf /workspace/sam-3d-objects/checkpoints # we will mount a volume with the checkpoints here when running the image
WORKDIR /workspace/sam-3d-objects
COPY server.py /workspace/sam-3d-objects/server.py


ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

RUN mamba env create -y -f environments/default.yml && mamba clean -a -y
ENV ENV_NAME=sam3d-objects

# Patch out nvidia-pyindex (it is a broken transitive dep in many builds)
RUN set -eux; \
    for f in pyproject.toml setup.cfg setup.py requirements.txt; do \
      [ -f "$f" ] && sed -i '/nvidia-pyindex/d' "$f" || true; \
    done; \
    find . -maxdepth 4 -type f -iname "*requirements*.txt" -exec sed -i '/nvidia-pyindex/d' {} \; || true

# NVIDIA RTX A6000
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN mamba run -n ${ENV_NAME} python -m pip install --upgrade pip setuptools wheel \
    && mamba run -n ${ENV_NAME} python -m pip install --no-cache-dir \
        wrapt hatchling hatch-requirements-txt editables \
    && mamba run -n ${ENV_NAME} python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio \
    && mamba run -n ${ENV_NAME} python -m pip install --no-cache-dir --no-build-isolation -e ".[dev]" \
    && mamba run -n ${ENV_NAME} python -m pip install --no-cache-dir --no-build-isolation -e ".[p3d]" \
    && mamba run -n ${ENV_NAME} python -m pip install --no-cache-dir --no-build-isolation -e ".[inference]" \
    && mamba run -n ${ENV_NAME} python -m pip install --no-cache-dir 'huggingface-hub[cli]<1.0' \
    && mamba run -n ${ENV_NAME} python ./patching/hydra

# ---- prevent "base" auto-activation in interactive shells ----
RUN conda config --system --set auto_activate_base false \
 && sed -i -E 's/^(.*(conda|mamba)\s+activate\s+base.*)$/# \1/g' /root/.bashrc || true
# -------------------------------------------------------------

WORKDIR /workspace/sam-3d-objects

RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda activate sam3d-objects' \
'cd /workspace' \
'exec "$@"' \
> /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash", "-lc", "cd /workspace/sam-3d-objects && exec python server.py"]
