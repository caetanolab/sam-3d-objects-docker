# syntax=docker/dockerfile:1

FROM condaforge/miniforge3:latest
SHELL ["/bin/bash", "-lc"]

# ---- system dependencies (OpenMP, X11, OpenGL, SSH) ----
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libgomp1 \
    libx11-6 \
    libgl1 \
    openssh-server \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# ------------------------------------------------------

# ---- SSH configuration ----
RUN mkdir -p /var/run/sshd \
 && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && sed -i 's@#AuthorizedKeysFile.*@AuthorizedKeysFile .ssh/authorized_keys@' /etc/ssh/sshd_config

# ---- SSH authorized key (REPLACE THIS) ----
RUN mkdir -p /root/.ssh \
 && chmod 700 /root/.ssh \
 && echo "ssh-ed25519 AAAA_REPLACE_WITH_YOUR_PUBLIC_KEY user@host" >> /root/.ssh/authorized_keys \
 && chmod 600 /root/.ssh/authorized_keys
# ------------------------------------------------------

WORKDIR /workspace
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git \
 && rm -rf /workspace/sam-3d-objects/checkpoints
WORKDIR /workspace/sam-3d-objects

ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

RUN mamba env create -y -f environments/default.yml && mamba clean -a -y
ENV ENV_NAME=sam3d-objects

# Patch out nvidia-pyindex (broken transitive dep)
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

# ---- prevent "base" auto-activation ----
RUN conda config --system --set auto_activate_base false \
 && sed -i -E 's/^(.*(conda|mamba)\s+activate\s+base.*)$/# \1/g' /root/.bashrc || true
# --------------------------------------

WORKDIR /workspace

# ---- entrypoint: SSH + conda env ----
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'/usr/sbin/sshd' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda activate sam3d-objects' \
'cd /workspace' \
'exec "$@"' \
> /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
