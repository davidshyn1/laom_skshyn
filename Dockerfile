FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

USER root

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    bzip2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh
ENV PATH=${CONDA_DIR}/bin:$PATH

WORKDIR /workspace

COPY environment.yaml /tmp/environment.yaml
RUN sed -i '/^prefix:/d' /tmp/environment.yaml

# mamba is pre-installed in Miniforge and significantly faster than conda
RUN mamba env create -f /tmp/environment.yaml && \
    mamba clean -afy

ENV PATH=/opt/conda/envs/laom/bin:$PATH

RUN pip uninstall -y opencv opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir --upgrade --force-reinstall opencv-python

COPY . /workspace
